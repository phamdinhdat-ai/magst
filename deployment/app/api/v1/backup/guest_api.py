import uuid
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
    create_guest_session, get_guest_session, update_guest_session_activity,
    create_guest_interaction, update_guest_interaction, add_interaction_feedback,
    get_guest_analytics, get_guest_interaction
)
from loguru import logger
from app.api.deps import get_db_session, get_guest_workflow_manager
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

router = APIRouter(prefix="/api/v1/guest", tags=["guest"])

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
    """Stream chat responses from the guest workflow"""
    
    if not WORKFLOW_AVAILABLE or not guest_workflow:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Guest workflow service is not available"
        )
    
    # Ensure session exists
    session_id = request.session_id or str(uuid.uuid4())
    session = await get_guest_session(db, session_id)
    
    if not session:
        # Create new session if it doesn't exist
        session_create = GuestSessionCreate(
            session_id=session_id,
            preferred_language=request.preferred_language
        )
        session = await create_guest_session(db, session_in=session_create)
    
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
            async for event in guest_workflow.arun_streaming(request.query, config):
                event_data = {
                    "event": event.get("event", "unknown"),
                    "data": event.get("data", {}),
                    "timestamp": datetime.now().isoformat()
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

@router.post("/chat/simple", response_model=GuestWorkflowResponse)
async def simple_chat(
    request: GuestWorkflowRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db_session)
):
    """Simple non-streaming chat endpoint"""
    
    if not WORKFLOW_AVAILABLE or not guest_workflow:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Guest workflow service is not available"
        )
    
    # Ensure session exists
    session_id = request.session_id or str(uuid.uuid4())
    session = await get_guest_session(db, session_id)
    
    if not session:
        session_create = GuestSessionCreate(
            session_id=session_id,
            preferred_language=request.preferred_language
        )
        session = await create_guest_session(db, session_in=session_create)
    
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
        config = {"configurable": {"thread_id": session_id}}
        
        full_response = ""
        suggested_questions = []
        agents_used = []
        
        # Process workflow
        async for event in guest_workflow.arun_streaming(request.query, config):
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
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
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

@router.post("/workflow", response_model=GuestWorkflowResponse)
async def guest_chat_simple(
    request: GuestWorkflowRequest,
    db: AsyncSession = Depends(get_db_session),
    guest_workflow: GuestWorkflow = Depends(get_guest_workflow_manager)
):
    """
    Thực thi workflow và trả về kết quả cuối cùng. Endpoint này sẽ đợi cho đến khi 
    workflow hoàn thành. Hữu ích cho các kịch bản backend-to-backend hoặc test.
    """
    if not guest_workflow:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Guest workflow service is not available")
        
    session_id = request.session_id or str(uuid.uuid4())
    session = await get_guest_session(db, session_id)
    if not session:
        session = await create_guest_session(db, GuestSessionCreate(session_id=session_id))
    interaction = await create_guest_interaction(db, GuestInteractionCreate(
        session_id=session.id, original_query=request.query
    ))
    await db.commit()
    await db.refresh(interaction)

    try:
        start_time = datetime.utcnow()
        config = {"configurable": {"thread_id": session_id}}
        
        # Giả sử workflow của bạn có hàm `arun_simple` để lấy kết quả cuối cùng
        result = await guest_workflow.arun_simple(request.query, config)
        
        end_time = datetime.utcnow()
        processing_time_ms = int((end_time - start_time).total_seconds() * 1000)
        
        full_answer = result.get("full_final_answer", "")
        suggested_questions = result.get("suggested_questions", [])
        agents_used = result.get("agents_used", [])

        # Cập nhật DB trực tiếp vì đây là luồng non-stream, người dùng đang đợi
        interaction_update = GuestInteractionUpdate(
            agent_response=full_answer,
            suggested_questions=suggested_questions,
            processing_time_ms=processing_time_ms,
            workflow_metadata={"agents_used": agents_used},
            completed_at=datetime.utcnow()
        )
        await update_guest_interaction(db, interaction.id, interaction_update)
        await update_guest_session_activity(db, session_id)
        await db.commit()

        return GuestWorkflowResponse(
            agent_response=full_answer,
            suggested_questions=suggested_questions,
            session_id=session_id,
            interaction_id=interaction.id,
            processing_time_ms=processing_time_ms,
            agents_used=agents_used
        )

    except Exception as e:
        logger.error(f"Error during simple guest chat for {interaction.id}: {e}", exc_info=True)
        await update_guest_interaction(db, interaction.id, GuestInteractionUpdate(workflow_metadata={"error": str(e)}, completed_at=datetime.utcnow()))
        await db.commit()
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"Workflow execution failed: {str(e)}")
    
    
    
# async def guest_workflow_query(
#     request: GuestWorkflowRequest,
#     background_tasks: BackgroundTasks,
#     db: AsyncSession = Depends(get_db_session)
# ):
#     """Execute guest workflow query - non-streaming endpoint"""
    
#     if not WORKFLOW_AVAILABLE or not guest_workflow:
#         raise HTTPException(
#             status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
#             detail="Guest workflow service is not available"
#         )
    
#     # Ensure session exists
#     session_id = request.session_id or str(uuid.uuid4())
#     session = await get_guest_session(db, session_id)
    
#     if not session:
#         session_create = GuestSessionCreate(
#             session_id=session_id,
#             preferred_language=request.preferred_language
#         )
#         session = await create_guest_session(db, session_in=session_create)
    
#     # Create interaction record
#     interaction_create = GuestInteractionCreate(
#         session_id=session.id,
#         original_query=request.query,
#         interaction_type=GuestInteractionType.GENERAL_SEARCH
#     )
#     interaction = await create_guest_interaction(
#         db, 
#         interaction_in=interaction_create
#     )
    
#     try:
#         start_time = datetime.now()
#         config = {"configurable": {"thread_id": session_id}}
        
#         full_response = ""
#         suggested_questions = []
#         agents_used = []
        
#         # Process workflow
#         async for event in guest_workflow.arun_streaming(request.query, config):
#             if event.get("event") == "node_start":
#                 node_name = event.get("data", {}).get("node")
#                 if node_name and node_name not in agents_used:
#                     agents_used.append(node_name)
#             elif event.get("event") == "answer_chunk":
#                 full_response = event.get("data", "")
#             elif event.get("event") == "final_result":
#                 final_data = event.get("data", {})
#                 suggested_questions = final_data.get("suggested_questions", [])
#                 full_response = final_data.get("full_final_answer", full_response)
        
#         processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
#         # Update interaction
#         interaction_update = GuestInteractionUpdate(
#             agent_response=full_response,
#             suggested_questions=suggested_questions,
#             processing_time_ms=int(processing_time),
#             completed_at=datetime.now()
#         )
        
#         await update_guest_interaction(
#             db, 
#             interaction.id, 
#             interaction_update
#         )
        
#         # Update session activity
#         background_tasks.add_task(update_guest_session_activity, db, session_id)
        
#         return GuestWorkflowResponse(
#             agent_response=full_response,
#             suggested_questions=suggested_questions,
#             session_id=session_id,
#             interaction_id=interaction.id,
#             processing_time_ms=int(processing_time),
#             agents_used=agents_used
#         )
        
#     except Exception as e:
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Workflow execution failed: {str(e)}"
#         )

@router.post("/feedback")
async def submit_feedback(
    feedback: GuestFeedback,
    db: AsyncSession = Depends(get_db_session)
):
    """Submit feedback for an interaction"""
    
    interaction = await add_interaction_feedback(
        db,
        feedback.interaction_id,
        feedback.rating,
        feedback.feedback_text,
        feedback.was_helpful
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

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "guest_workflow",
        "timestamp": datetime.now().isoformat()
    }

@router.post("/workflow/stream")
async def guest_chat_stream(
    request: GuestWorkflowRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db_session),
    guest_workflow: GuestWorkflow = Depends(get_guest_workflow_manager)
):
    """
    Xử lý query của guest, stream câu trả lời và giao việc lưu trữ cho background task.
    Đây là endpoint được khuyến nghị sử dụng cho các giao diện người dùng.
    """
    if not guest_workflow:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Guest workflow service is not available")

    # BƯỚC 1: CHUẨN BỊ SESSION VÀ INTERACTION
    session_id = request.session_id or str(uuid.uuid4())
    session = await get_guest_session(db, session_id)
    if not session:
        session = await create_guest_session(db, GuestSessionCreate(session_id=session_id))

    interaction = await create_guest_interaction(db, GuestInteractionCreate(
        session_id=session.id, original_query=request.query
    ))
    await db.commit()
    await db.refresh(interaction)

    config = {"configurable": {"thread_id": session_id}}

    # BƯỚC 2: KHỞI CHẠY STREAM GENERATOR
    async def stream_generator():
        final_result_data, agents_used, full_answer = {}, [], ""
        start_time = datetime.utcnow()
        try:
            async with AsyncSession() as session:  # Create a new session for the stream
                async for event in guest_workflow.arun_streaming(request.query, config):
                    event_data = {
                        "event": event.get("event", "unknown"),
                        "data": event.get("data", {}),
                        "timestamp": datetime.utcnow().isoformat()
                    }

                    # Track agent usage and responses
                    if event.get("event") == "node_start":
                        node_name = event.get("data", {}).get("node")
                        if node_name and node_name not in agents_used:
                            agents_used.append(node_name)
                    elif event.get("event") == "answer_chunk":
                        full_answer = event.get("data", "")
                    elif event.get("event") == "final_result":
                        final_result_data = event.get("data", {})
                        
                    # Stream the event
                    yield f"data: {json.dumps(event_data)}\n\n"

                # Calculate processing time and prepare final event
                processing_time = (datetime.utcnow() - start_time).total_seconds()
                final_event = {
                    "event": "workflow_complete",
                    "data": {
                        "interaction_id": str(interaction.id),
                        "session_id": session_id,
                        "processing_time_ms": int(processing_time * 1000),
                        "agents_used": agents_used
                    },
                    "timestamp": datetime.utcnow().isoformat()
                }
                yield f"data: {json.dumps(final_event)}\n\n"

                # Schedule background task for saving results
                background_tasks.add_task(
                    _save_guest_interaction_in_background,
                    db,
                    interaction.id,
                    session_id,
                    final_result_data,
                    processing_time
                )

        except Exception as e:
            logger.error(f"Error in guest workflow stream for {interaction.id}: {e}", exc_info=True)
            error_event = {
                "event": "error",
                "data": {"error": str(e)},
                "timestamp": datetime.utcnow().isoformat()
            }
            yield f"data: {json.dumps(error_event)}\n\n"
            
            # Save error state in background
            background_tasks.add_task(
                _save_guest_interaction_in_background,
                db,
                interaction.id,
                session_id,
                {"error": str(e)},
                (datetime.utcnow() - start_time).total_seconds()
            )
    return StreamingResponse(stream_generator(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "*"
    })
        

# ------------------------------------------------------------------------------
# HÀM XỬ LÝ NỀN (BACKGROUND TASK) - TRÁI TIM CỦA VIỆC LƯU TRỮ
# ------------------------------------------------------------------------------
async def _save_guest_interaction_in_background(
    db: AsyncSession,
    interaction_id: str,
    session_id: str,
    result_data: Dict[str, Any],
    processing_time: float
):
    """Background task to save interaction results after streaming completes"""
    try:
        async with db as session:  # Ensure proper session handling
            # Update interaction with results
            interaction_update = GuestInteractionUpdate(
                agent_response=result_data.get("full_final_answer", ""),
                suggested_questions=result_data.get("suggested_questions", []),
                processing_time_ms=int(processing_time * 1000),
                workflow_metadata={"agents_used": result_data.get("agents_used", [])},
                completed_at=datetime.utcnow()
            )
            
            await update_guest_interaction(session, interaction_id, interaction_update)
            await update_guest_session_activity(session, session_id)
            await session.commit()
            
    except Exception as e:
        logger.error(f"Failed to save interaction results for {interaction_id}: {e}", exc_info=True)
        # Try to save error state
        try:
            async with db as session:
                error_update = GuestInteractionUpdate(
                    workflow_metadata={"error": str(e)},
                    completed_at=datetime.utcnow()
                )
                await update_guest_interaction(session, interaction_id, error_update)
                await session.commit()
        except Exception as inner_e:
            logger.error(f"Failed to save error state for {interaction_id}: {inner_e}", exc_info=True)
