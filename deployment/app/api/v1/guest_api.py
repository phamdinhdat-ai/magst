import uuid
import asyncio
import os
import time
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
import json

from app.db.session import get_db_session
from app.db.models.guest import (
    GuestWorkflowRequest, GuestWorkflowResponse, GuestWorkflowStreamEvent,
    GuestSessionCreate, GuestSessionResponse, GuestInteractionCreate, GuestInteractionResponse,
    GuestFeedback, GuestAnalytics, GuestInteractionUpdate
)
from app.db.models.base_models import GuestInteractionType
from app.crud.crud_guest import (
    create_guest_session, get_guest_session, get_or_create_guest_session,
    update_guest_session_activity, create_guest_interaction, update_guest_interaction, 
    add_interaction_feedback, get_guest_analytics, get_guest_interaction
)
from app.api.v1.guest_request_queue import GuestRequestQueue
from loguru import logger
# Try to import guest workflow, but make it optional for testing
try:
    from app.agents.workflow.guest_workflow import GuestWorkflow
    WORKFLOW_AVAILABLE = True
except ImportError as e:
    WORKFLOW_AVAILABLE = False
    print(f"Warning: Guest workflow not available: {e}")

# Initialize guest workflow if available
guest_workflow = None
if WORKFLOW_AVAILABLE:
    try:
        guest_workflow = GuestWorkflow()
    except Exception as e:
        print(f"Warning: Failed to initialize guest workflow: {e}")
        WORKFLOW_AVAILABLE = False

# Initialize the request queue system
# Configure based on environment or default values
MAX_CONCURRENT = int(os.getenv("GUEST_MAX_CONCURRENT_REQUESTS", "3"))
MAX_QUEUE_SIZE = int(os.getenv("GUEST_MAX_QUEUE_SIZE", "50"))
REQUEST_TIMEOUT = int(os.getenv("GUEST_REQUEST_TIMEOUT_SEC", "60"))

# Create and configure the request queue
request_queue = GuestRequestQueue(
    max_concurrent=MAX_CONCURRENT,
    max_queue_size=MAX_QUEUE_SIZE,
    request_timeout=REQUEST_TIMEOUT
)

# If workflow is available, set it in the queue
if WORKFLOW_AVAILABLE and guest_workflow:
    request_queue.set_workflow(guest_workflow)
    # Start workers asynchronously (will be done when app starts)
    asyncio.create_task(request_queue.start_workers())

router = APIRouter(prefix="/api/v1/guest", tags=["guest"])

# Dependency to get chat history cache manager
async def get_history_cache_manager():
    """Dependency to get the chat history cache manager if available"""
    if WORKFLOW_AVAILABLE and guest_workflow and hasattr(guest_workflow, 'cache_manager'):
        return guest_workflow.cache_manager
    return None

@router.post("/session", response_model=GuestSessionResponse)
async def create_session(
    request: Dict[str, Any],
    db: AsyncSession = Depends(get_db_session)
):
    """Create a new guest session"""
    session_id = request.get("session_id") or str(uuid.uuid4())
    
    session_create = GuestSessionCreate(
        session_id=session_id,
        ip_address=request.get("ip_address"),
        user_agent=request.get("user_agent"),
        preferred_language=request.get("preferred_language", "vi")
    )
    
    try:
        session = await create_guest_session(db, session_in=session_create)
        return session
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create session: {str(e)}"
        )

@router.get("/session/{session_id}", response_model=GuestSessionResponse)
async def get_session(
    session_id: str,
    db: AsyncSession = Depends(get_db_session)
):
    """Get guest session information"""
    session = await get_guest_session(db, session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    return session

@router.post("/chat")
async def chat_with_workflow(
    request: GuestWorkflowRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db_session)
):
    """Stream chat responses from the guest workflow using the request queue system"""
    
    if not WORKFLOW_AVAILABLE or not guest_workflow:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Guest workflow service is not available"
        )
    
    # Get or create session atomically
    session_id = request.session_id or str(uuid.uuid4())
    session = await get_or_create_guest_session(
        db,
        session_id=session_id,
        preferred_language=request.preferred_language
    )
    
    # Create interaction record
    interaction_create = GuestInteractionCreate(
        session_id=session.id,
        original_query=request.query,
        interaction_type=GuestInteractionType.GENERAL_SEARCH
    )
    interaction = await create_guest_interaction(
        db, 
        interaction_in=interaction_create
    )
    
    # Generate a guest ID using the session id
    guest_id = str(session.id)
    
    # Update session activity in background
    background_tasks.add_task(update_guest_session_activity, db, session_id)
    
    # Get chat history if available
    chat_history = None
    history_cache_manager = None
    if hasattr(guest_workflow, 'cache_manager'):
        history_cache_manager = guest_workflow.cache_manager
        if history_cache_manager and history_cache_manager.is_active():
            try:
                chat_history = await history_cache_manager.get_chat_history(session_id)
                logger.debug(f"Retrieved chat history for session {session_id}: {len(chat_history) if chat_history else 0} messages")
            except Exception as e:
                logger.warning(f"Failed to retrieve chat history: {str(e)}")
    
    # Enqueue the request
    try:
        # Check if this is a priority request (for paying customers or special users)
        is_priority = request.metadata and request.metadata.get("is_priority", False)
        
        request_id, future = await request_queue.enqueue_request(
            query=request.query,
            guest_id=guest_id,
            session_id=session_id,
            chat_history=chat_history,
            priority=is_priority
        )
        
        # Store request ID in the interaction for later reference
        background_tasks.add_task(
            update_guest_interaction,
            db,
            interaction.id,
            GuestInteractionUpdate(metadata={"request_id": request_id})
        )
        
    except HTTPException as e:
        # Re-raise queue-related errors
        raise
    except Exception as e:
        logger.error(f"Failed to enqueue request: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process request: {str(e)}"
        )
    
    async def generate_response():
        """Generate streaming response from the queued request"""
        start_time = datetime.now()
        queue_position = request_queue.queue.qsize()  # Approximate position
        
        # Send initial queue status
        initial_event = {
            "event": "queued",
            "data": {
                "request_id": request_id,
                "queue_position": queue_position,
                "message": f"Request queued for processing at position {queue_position}."
            },
            "timestamp": datetime.now().isoformat()
        }
        yield f"data: {json.dumps(initial_event)}\n\n"
        
        full_response = ""
        suggested_questions = []
        agents_used = []
        
        try:
            # Wait for the request to be processed
            results_generator = await future
            
            # Stream the processing started event
            processing_event = {
                "event": "processing",
                "data": {
                    "request_id": request_id,
                    "message": "Request processing started."
                },
                "timestamp": datetime.now().isoformat()
            }
            yield f"data: {json.dumps(processing_event)}\n\n"
            
            # Stream all events from the workflow
            async for event in results_generator:
                event_data = {
                    "event": event.get("event", "unknown"),
                    "data": event.get("data", {}),
                    "timestamp": event.get("metadata", {}).get("timestamp", datetime.now().isoformat())
                }
                
                # Track agent usage
                if event.get("event") == "node_start":
                    node_name = event.get("data", {}).get("node")
                    if node_name and node_name not in agents_used:
                        agents_used.append(node_name)
                
                # Stream the event
                yield f"data: {json.dumps(event_data)}\n\n"
                # Collect final response data
                if event.get("event") == "answer_chunk":
                    full_response = event.get("data", "")
                elif event.get("event") == "final_result":
                    final_data = event.get("data", {})
                    suggested_questions = final_data.get("suggested_questions", [])
                    full_response = final_data.get("full_final_answer", full_response)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Update interaction with results in background
            background_tasks.add_task(
                _save_guest_interaction_in_background,
                db, 
                interaction.id,
                session_id,
                {
                    "agent_response": full_response,
                    "suggested_questions": suggested_questions,
                    "agents_used": agents_used
                },
                processing_time / 1000  # Convert to seconds
            )
            
            # Also store in chat history if available
            if history_cache_manager and history_cache_manager.is_active():
                background_tasks.add_task(
                    _save_to_chat_history,
                    history_cache_manager,
                    session_id,
                    guest_id,
                    request.query,
                    full_response,
                    {
                        "suggested_questions": suggested_questions,
                        "agents_used": agents_used,
                        "timestamp": datetime.now().isoformat()
                    }
                )
            
            # If we didn't get a final event from the workflow, send our own
            if not any(e.get("event") == "workflow_complete" for e in [event]):
                final_event = {
                    "event": "workflow_complete",
                    "data": {
                        "interaction_id": str(interaction.id),
                        "session_id": session_id,
                        "processing_time_ms": int(processing_time),
                        "agents_used": agents_used
                    },
                    "timestamp": datetime.now().isoformat()
                }
                yield f"data: {json.dumps(final_event)}\n\n"
            
        except asyncio.CancelledError:
            # Request was cancelled
            cancelled_event = {
                "event": "cancelled",
                "data": {"message": "Request was cancelled"},
                "timestamp": datetime.now().isoformat()
            }
            yield f"data: {json.dumps(cancelled_event)}\n\n"
            
        except Exception as e:
            logger.error(f"Error in streaming response: {str(e)}", exc_info=True)
            error_event = {
                "event": "error",
                "data": {"error": str(e)},
                "timestamp": datetime.now().isoformat()
            }
            yield f"data: {json.dumps(error_event)}\n\n"
            
            # Update interaction with error status
            background_tasks.add_task(
                update_guest_interaction,
                db,
                interaction.id,
                GuestInteractionUpdate(
                    error=str(e),
                    completed_at=datetime.now()
                )
            )
    
    return StreamingResponse(
        generate_response(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Prevent Nginx buffering
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*"
        }
    )

@router.post("/chat/simple", response_model=GuestWorkflowResponse)
async def simple_chat(
    request: GuestWorkflowRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db_session)
):
    """Simple non-streaming chat endpoint using request queue"""
    
    if not WORKFLOW_AVAILABLE or not guest_workflow:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Guest workflow service is not available"
        )
    
    # Get or create session atomically
    session_id = request.session_id or str(uuid.uuid4())
    session = await get_or_create_guest_session(
        db,
        session_id=session_id,
        preferred_language=request.preferred_language
    )
    
    # Create interaction record
    interaction_create = GuestInteractionCreate(
        session_id=session.id,
        original_query=request.query,
        interaction_type=GuestInteractionType.GENERAL_SEARCH
    )
    interaction = await create_guest_interaction(
        db, 
        interaction_in=interaction_create
    )
    
    # Generate a guest ID using the session id
    guest_id = str(session.id)
    
    # Get chat history if available
    chat_history = None
    history_cache_manager = None
    if hasattr(guest_workflow, 'cache_manager'):
        history_cache_manager = guest_workflow.cache_manager
        if history_cache_manager and history_cache_manager.is_active():
            try:
                chat_history = await history_cache_manager.get_chat_history(session_id)
                logger.debug(f"Retrieved chat history for session {session_id}: {len(chat_history) if chat_history else 0} messages")
            except Exception as e:
                logger.warning(f"Failed to retrieve chat history: {str(e)}")
    
    try:
        start_time = datetime.now()
        
        # Enqueue the request
        # Check if this is a priority request (for paying customers or special users)
        is_priority = request.metadata and request.metadata.get("is_priority", False)
        
        request_id, future = await request_queue.enqueue_request(
            query=request.query,
            guest_id=guest_id,
            session_id=session_id,
            chat_history=chat_history,
            priority=is_priority
        )
        
        # Wait for the request to be processed
        results_generator = await future
        
        full_response = ""
        suggested_questions = []
        agents_used = []
        
        # Process all events from workflow
        async for event in results_generator:
            if event.get("event") == "node_start":
                node_name = event.get("data", {}).get("node")
                if node_name and node_name not in agents_used:
                    agents_used.append(node_name)
            elif event.get("event") == "answer_chunk":
                full_response = event.get("data", "")
            elif event.get("event") == "final_result":
                final_data = event.get("data", {})
                suggested_questions = final_data.get("suggested_questions", [])
                full_response = final_data.get("full_final_answer", full_response)
            elif event.get("event") == "timeout":
                # Handle timeout case
                full_response = "I'm sorry, but your request timed out due to high demand. Please try again in a moment."
                suggested_questions = ["Tell me about GeneStory", "What services do you provide?"]
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Update interaction
        interaction_update = GuestInteractionUpdate(
            agent_response=full_response,
            suggested_questions=suggested_questions,
            processing_time_ms=int(processing_time),
            completed_at=datetime.now(),
            metadata={"request_id": request_id}
        )
        
        await update_guest_interaction(
            db, 
            interaction.id, 
            interaction_update
        )
        
        # Update session activity
        background_tasks.add_task(update_guest_session_activity, db, session_id)
        
        # Also store in chat history if available
        if history_cache_manager and history_cache_manager.is_active():
            background_tasks.add_task(
                _save_to_chat_history,
                history_cache_manager,
                session_id,
                guest_id,
                request.query,
                full_response,
                {
                    "suggested_questions": suggested_questions,
                    "agents_used": agents_used,
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        return GuestWorkflowResponse(
            agent_response=full_response,
            suggested_questions=suggested_questions,
            session_id=session_id,
            interaction_id=interaction.id,
            processing_time_ms=int(processing_time),
            agents_used=agents_used
        )
        
    except asyncio.TimeoutError:
        # Handle explicit timeout
        logger.warning(f"Request timed out for query: {request.query[:50]}...")
        timeout_response = "I'm sorry, but I couldn't process your request in time due to high demand. Please try again in a moment."
        
        # Update interaction with timeout status
        await update_guest_interaction(
            db,
            interaction.id,
            GuestInteractionUpdate(
                agent_response=timeout_response,
                suggested_questions=["Tell me about GeneStory", "What services do you provide?"],
                error="Request timed out",
                completed_at=datetime.now()
            )
        )
        
        return GuestWorkflowResponse(
            agent_response=timeout_response,
            suggested_questions=["Tell me about GeneStory", "What services do you provide?"],
            session_id=session_id,
            interaction_id=interaction.id,
            processing_time_ms=int((datetime.now() - start_time).total_seconds() * 1000),
            agents_used=[]
        )
    except Exception as e:
        logger.error(f"Simple chat workflow failed: {str(e)}", exc_info=True)
        # Update interaction with error
        await update_guest_interaction(
            db,
            interaction.id,
            GuestInteractionUpdate(
                error=str(e),
                completed_at=datetime.now()
            )
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Workflow execution failed: {str(e)}"
        )

@router.post("/workflow", response_model=GuestWorkflowResponse)
async def guest_workflow_query(
    request: GuestWorkflowRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db_session)
):
    """Execute guest workflow query - optimized for speed"""
    
    if not WORKFLOW_AVAILABLE or not guest_workflow:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Guest workflow service is not available"
        )
    
    # Get or create session atomically
    session_id = request.session_id or str(uuid.uuid4())
    session = await get_or_create_guest_session(
        db,
        session_id=session_id,
        preferred_language=request.preferred_language
    )
    
    # Create interaction record
    interaction_create = GuestInteractionCreate(
        session_id=session.id,
        original_query=request.query,
        interaction_type=GuestInteractionType.GENERAL_SEARCH
    )
    interaction = await create_guest_interaction(
        db, 
        interaction_in=interaction_create
    )
    
    try:
        start_time = datetime.now()
        
        # Check if fast mode is enabled
        import os
        fast_mode_enabled = os.getenv("FAST_MODE_ENABLED", "false").lower() == "true"
        
        if fast_mode_enabled:
            # Use fast mode processing
            result = await guest_workflow.process_fast_mode(
                query=request.query,
                session_id=session_id,
                guest_id=str(session.id)
            )
            
            full_response = result.get("agent_response", "")
            suggested_questions = result.get("suggested_questions", [])
            processing_metadata = result.get("metadata", {})
            
        else:
            # Use load balancing processing (original method)
            result = await guest_workflow.process_with_load_balancing(
                query=request.query,
                session_id=session_id,
                guest_id=str(session.id)
            )
            
            full_response = result.get("agent_response", "")
            suggested_questions = result.get("suggested_questions", [])
            processing_metadata = result.get("metadata", {})
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Get agents used from metadata or set default
        agents_used = processing_metadata.get("agents_used", ["guest_workflow"])
        
        # Update interaction
        interaction_update = GuestInteractionUpdate(
            agent_response=full_response,
            suggested_questions=suggested_questions,
            processing_time_ms=int(processing_time),
            completed_at=datetime.now()
        )
        
        await update_guest_interaction(
            db, 
            interaction.id, 
            interaction_update
        )
        
        # Update session activity
        background_tasks.add_task(update_guest_session_activity, db, session_id)
        
        return GuestWorkflowResponse(
            agent_response=full_response,
            suggested_questions=suggested_questions,
            session_id=session_id,
            interaction_id=interaction.id,
            processing_time_ms=int(processing_time),
            agents_used=agents_used
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Workflow execution failed: {str(e)}"
        )

@router.post("/feedback")
async def submit_feedback(
    feedback: GuestFeedback,
    db: AsyncSession = Depends(get_db_session)
):
    """Submit feedback for an interaction"""
    
    try:
        # Use the unified feedback system
        from app.crud.crud_feedback import update_feedback, create_feedback
        
        # First, update the interaction
        interaction = await update_feedback(
            db=db,
            interaction_id=feedback.interaction_id,
            user_type='guest',
            rating=feedback.rating,
            feedback_text=feedback.feedback_text,
            was_helpful=feedback.was_helpful,
            feedback_type=feedback.feedback_type.value if feedback.feedback_type else None
        )
        
        # Now create a dedicated feedback entry with session_id and message_id
        await create_feedback(
            db=db,
            interaction_id=feedback.interaction_id,
            user_type='guest',
            rating=feedback.rating,
            feedback_text=feedback.feedback_text,
            was_helpful=feedback.was_helpful,
            feedback_type=feedback.feedback_type.value if feedback.feedback_type else None,
            message_id=feedback.message_id,
            session_id=feedback.session_id
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    if not interaction:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Interaction not found"
        )
    
    return {"message": "Feedback submitted successfully"}

@router.get("/analytics", response_model=GuestAnalytics)
async def get_analytics(
    days: int = 30,
    db: AsyncSession = Depends(get_db_session)
):
    """Get guest usage analytics"""
    try:
        analytics_data = await get_guest_analytics(db)
        return analytics_data
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get analytics: {str(e)}"
        )

@router.get("/chat-history/{session_id}")
async def get_guest_chat_history(
    session_id: str,
    limit: Optional[int] = None,
    cache_manager = Depends(get_history_cache_manager)
):
    """Get chat history for a guest session"""
    if not cache_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat history service is not available"
        )
    
    try:
        # Get chat history from cache
        chat_history = await cache_manager.get_chat_history(
            session_id=session_id,
            limit=limit,
            format_for_workflow=False  # Get raw messages for the frontend
        )
        
        if not chat_history:
            return {"messages": [], "session_id": session_id}
        
        return {
            "messages": chat_history,
            "session_id": session_id,
            "message_count": len(chat_history)
        }
    except Exception as e:
        logger.error(f"Failed to retrieve chat history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve chat history: {str(e)}"
        )

@router.delete("/chat-history/{session_id}")
async def clear_guest_chat_history(
    session_id: str,
    cache_manager = Depends(get_history_cache_manager)
):
    """Clear chat history for a guest session"""
    if not cache_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat history service is not available"
        )
    
    try:
        # Delete session and its chat history
        success = await cache_manager.delete_guest_session(session_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session {session_id} not found or could not be deleted"
            )
        
        return {
            "status": "success", 
            "message": f"Chat history for session {session_id} has been cleared",
            "session_id": session_id
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to clear chat history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear chat history: {str(e)}"
        )

@router.get("/queue/status")
async def get_queue_status(detailed: bool = False):
    """Get current queue statistics and status
    
    Args:
        detailed: Whether to include detailed statistics (default: False)
    """
    if not WORKFLOW_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Guest workflow service is not available"
        )
    
    queue_stats = request_queue.get_queue_stats()
    
    # Calculate estimated wait time based on queue size and average processing time
    queue_size = queue_stats.get("current_queue_size", 0)
    avg_time = queue_stats.get("avg_processing_time_sec", 5) # Default to 5 seconds if not available
    estimated_wait = queue_size * avg_time
    
    response = {
        "queue_stats": queue_stats,
        "estimated_wait_time_sec": estimated_wait,
        "is_busy": queue_size > request_queue.max_concurrent,
        "timestamp": datetime.now().isoformat()
    }
    
    # Include additional details if requested
    if detailed:
        # Get active requests (sanitize sensitive info)
        active_requests = []
        for req_id, req in request_queue.active_requests.items():
            active_requests.append({
                "request_id": req_id,
                "status": req.status.value,
                "elapsed_time": time.time() - (req.start_time or req.timestamp),
                "priority": req.priority if hasattr(req, "priority") else False
            })
        
        response["active_requests"] = active_requests
        response["system_info"] = {
            "max_concurrent": request_queue.max_concurrent,
            "max_queue_size": request_queue.max_queue_size,
            "queue_utilization": queue_size / request_queue.max_queue_size if request_queue.max_queue_size > 0 else 0,
            "workers_active": request_queue._workers_started
        }
    
    return response

@router.get("/queue/request/{request_id}")
async def get_request_status(request_id: str):
    """Get status of a specific request"""
    if not WORKFLOW_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Guest workflow service is not available"
        )
    
    request_status = request_queue.get_request_status(request_id)
    if "error" in request_status:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Request {request_id} not found"
        )
    
    return {
        "request_status": request_status,
        "timestamp": datetime.now().isoformat()
    }

@router.delete("/queue/request/{request_id}")
async def cancel_request(request_id: str):
    """Cancel a specific request if it's still in the queue"""
    if not WORKFLOW_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Guest workflow service is not available"
        )
    
    # Check if we have a cancel method
    if not hasattr(request_queue, "cancel_request"):
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Request cancellation is not supported"
        )
    
    # Try to cancel the request
    cancelled = await request_queue.cancel_request(request_id)
    
    if cancelled:
        return {
            "status": "success",
            "message": f"Request {request_id} was cancelled successfully",
            "timestamp": datetime.now().isoformat()
        }
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Request {request_id} could not be cancelled (not found or already processing)"
        )

@router.get("/health")
async def health_check():
    """Enhanced health check endpoint including queue status"""
    service_status = "healthy" if WORKFLOW_AVAILABLE else "unavailable"
    
    response = {
        "status": service_status,
        "service": "guest_workflow",
        "timestamp": datetime.now().isoformat(),
    }
    
    # Add queue information if available
    if WORKFLOW_AVAILABLE:
        queue_stats = request_queue.get_queue_stats()
        response["queue_health"] = {
            "active_requests": queue_stats["active_requests"],
            "current_queue_size": queue_stats["current_queue_size"],
            "is_queue_full": queue_stats["current_queue_size"] >= request_queue.max_queue_size,
            "max_queue_size": request_queue.max_queue_size,
            "max_concurrent": request_queue.max_concurrent
        }
    
    return response

@router.post("/workflow/stream")
async def guest_workflow_stream(
    request: GuestWorkflowRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db_session)
):
    """Stream workflow responses from the guest workflow"""
    
    if not WORKFLOW_AVAILABLE or not guest_workflow:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Guest workflow service is not available"
        )
    
    # Get or create session atomically
    session_id = request.session_id or str(uuid.uuid4())
    session = await get_or_create_guest_session(
        db,
        session_id=session_id,
        preferred_language=request.preferred_language
    )
    
    # Create interaction record
    interaction_create = GuestInteractionCreate(
        session_id=session.id,
        original_query=request.query,
        interaction_type=GuestInteractionType.GENERAL_SEARCH
    )
    interaction = await create_guest_interaction(
        db, 
        interaction_in=interaction_create
    )
    
    # Update session activity
    background_tasks.add_task(update_guest_session_activity, db, session_id)
    
    async def generate_response():
        """Generate streaming response from guest workflow"""
        start_time = datetime.now()
        config = {"configurable": {"thread_id": session_id}}
        
        full_response = ""
        suggested_questions = []
        agents_used = []
        
        try:
            async for event in guest_workflow.astreaming_workflow(
                query=request.query, 
                config=config, 
                guest_id=str(session.id),
                chat_history=request.chat_history or []
            ):
                # Preserve original event structure for frontend compatibility
                event_data = event
                # Add timestamp if needed
                if "timestamp" not in event_data:
                    event_data["timestamp"] = datetime.now().isoformat()
                
                # Track agent usage
                if event.get("event") == "node_start":
                    node_name = event.get("data", {}).get("node")
                    if node_name and node_name not in agents_used:
                        agents_used.append(node_name)
                
                # Stream the event
                yield f"data: {json.dumps(event_data)}\n\n"
                # await asyncio.sleep(0.03)  # Yield control to event loop

                # Collect final response data
                if event.get("event") == "answer_chunk":
                    full_response = event.get("data", "")
                elif event.get("event") == "final_result":
                    final_data = event.get("data", {})
                    suggested_questions = final_data.get("suggested_questions", [])
                    full_response = final_data.get("full_final_answer", full_response)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Update interaction with results
            interaction_update = GuestInteractionUpdate(
                agent_response=full_response,
                suggested_questions=suggested_questions,
                processing_time_ms=int(processing_time),
                completed_at=datetime.now()
            )
            
            await update_guest_interaction(
                db, 
                interaction.id, 
                interaction_update
            )
            
            # Send final event
            final_event = {
                "event": "workflow_complete",
                "data": {
                    "interaction_id": str(interaction.id),
                    "session_id": session_id,
                    "processing_time_ms": int(processing_time),
                    "agents_used": agents_used
                },
                "timestamp": datetime.now().isoformat()
            }
            yield f"data: {json.dumps(final_event)}\n\n"
            
        except Exception as e:
            error_event = {
                "event": "error",
                "data": {"error": str(e)},
                "timestamp": datetime.now().isoformat()
            }
            yield f"data: {json.dumps(error_event)}\n\n"
    
    return StreamingResponse(
        generate_response(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*"
        }
    )





# ------------------------------------------------------------------------------
# HELPER FUNCTIONS FOR CHAT HISTORY MANAGEMENT
# ------------------------------------------------------------------------------
async def _save_to_chat_history(
    cache_manager, 
    session_id: str, 
    guest_id: str,
    user_message: str, 
    assistant_response: str,
    metadata: dict = None
):
    """
    Helper function to save chat history using the cache manager.
    Can be used from any endpoint that needs to save chat history.
    """
    if not cache_manager or not session_id:
        return
        
    try:
        # Ensure we have a session
        guest_id, cache_session_id = await cache_manager.create_guest_session(
            guest_id=guest_id,
            session_id=session_id
        )
        
        # Add the conversation turn
        user_metadata = {"timestamp": datetime.now().isoformat()}
        assistant_metadata = metadata or {}
        
        await cache_manager.add_conversation_turn(
            session_id=cache_session_id,
            user_message=user_message,
            assistant_response=assistant_response,
            user_metadata=user_metadata,
            assistant_metadata=assistant_metadata
        )
        
        logger.info(f"Saved conversation to chat history for session {session_id}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save to chat history: {e}")
        return False

# ------------------------------------------------------------------------------
# HÀM XỬ LÝ NỀN (BACKGROUND TASK)
# ------------------------------------------------------------------------------
async def _save_guest_interaction_in_background(
    db: AsyncSession,
    interaction_id: uuid.UUID,
    session_id: str,
    final_result_data: dict,
    processing_time_sec: float
):
    """
    Chạy trong nền để lưu lại đầy đủ thông tin tương tác của Guest.
    Hàm này được thiết kế để không bao giờ làm sập ứng dụng chính.
    """
    logger.info(f"Background task started for guest interaction {interaction_id}")
    try:
        # BƯỚC 1: TRÍCH XUẤT DỮ LIỆU TỪ KẾT QUẢ WORKFLOW
        full_answer = final_result_data.get("full_final_answer", "")
        suggested_questions = final_result_data.get("suggested_questions", [])
        agents_used = final_result_data.get("agents_used", [])
        processing_time_ms = int(processing_time_sec * 1000)

        # BƯỚC 2: CẬP NHẬT INTERACTION ĐÃ ĐƯỢC TẠO TỪ TRƯỚC
        interaction = await get_guest_interaction(db, interaction_id)
        if interaction:
            workflow_metadata = {
                "agents_used": agents_used,
                "session_id": session_id,
                "error": final_result_data.get("error")
            }
            interaction_update = GuestInteractionUpdate(
                agent_response=full_answer,
                suggested_questions=suggested_questions,
                processing_time_ms=processing_time_ms,
                workflow_metadata=workflow_metadata,
                completed_at=datetime.utcnow()
            )
            await update_guest_interaction(db, interaction.id, interaction_update)

        # BƯỚC 3: CẬP NHẬT HOẠT ĐỘNG CỦA SESSION
        await update_guest_session_activity(db, session_id)
        
        # BƯỚC 4: COMMIT TRANSACTION
        await db.commit()
        logger.info(f"Background task finished successfully for guest interaction {interaction_id}.")

    except Exception as e:
        logger.error(f"Error in background task for guest interaction {interaction_id}: {e}", exc_info=True)
        await db.rollback()
    finally:
        await db.close()
