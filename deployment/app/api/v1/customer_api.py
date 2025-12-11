from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger
import json
import uuid
import time
import asyncio
from functools import lru_cache
from contextvars import ContextVar
from prometheus_client import Counter, Histogram
from sqlalchemy import select, update, func
from tenacity import retry, stop_after_attempt, wait_exponential

from app.api.deps import (
    get_current_customer, get_current_active_customer, rate_limit_workflow_requests,
    verify_chat_thread_access, track_customer_activity, update_customer_last_activity, verify_interaction_access
)
from app.core.security import (
    create_access_token, create_refresh_token, verify_token, hash_refresh_token,
    generate_password_reset_token, verify_password_reset_token, CustomerPermissions,
    get_customer_permissions
)
from app.core.config import settings

from app.db.session import get_db_session
from app.db.models.customer import (
    Customer, CustomerRole, CustomerStatus, CustomerInteractionType,
    CustomerChatThread, CustomerChatMessage, CustomerRegister, CustomerLogin,
    CustomerLoginResponse, TokenRefresh, PasswordChange, PasswordReset,
    PasswordResetConfirm, CustomerResponse, CustomerUpdate, CustomerStats,
    CustomerSessionResponse, CustomerInteractionResponse, CustomerInteractionUpdate,
    CustomerChatThreadResponse, CustomerChatThreadCreate, CustomerChatMessageResponse,
    CustomerChatMessageCreate, CustomerWorkflowRequest, CustomerWorkflowResponse,
    CustomerFeedback, CustomerInteractionCreate, CustomerInteractionUpdate, CustomerChatThreadUpdate,
    CustomerInteraction
)
from app.crud.crud_customer import (
    get_customer, get_customer_by_email, get_customer_by_username, create_customer,
    authenticate_customer, update_customer_last_login, update_customer,
    change_customer_password, verify_customer_email, admin_update_customer,
    get_customers_with_filters, get_customer_stats, create_customer_session,
    get_customer_session, get_active_customer_sessions, update_customer_session_activity,
    expire_old_customer_sessions, create_customer_interaction, get_customer_interaction,
    get_customer_interactions, update_customer_interaction, update_customer_interaction_feedback,
    create_customer_chat_thread, get_customer_chat_thread, get_customer_chat_threads,
    update_customer_chat_thread, archive_customer_chat_thread, delete_customer_chat_thread,
    create_customer_chat_message, get_next_message_sequence_number,
    create_customer_refresh_token, get_customer_refresh_token_by_hash,
    revoke_customer_refresh_token, revoke_all_customer_tokens,
    get_customer_chat_thread_messages
)
from app.agents.workflow.customer_workflow import CustomerWorkflow, create_customer_workflow_session
from app.api.v1.customer_request_queue import customer_request_queue
from app.agents.workflow.state import GraphState as AgentState
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Initialize synchronous database engine for background tasks
SYNC_ENGINE = create_engine(settings.POSTGRES_DATABASE_URL.replace("postgresql+asyncpg", "postgresql"))
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=SYNC_ENGINE)

# Prometheus metrics
WORKFLOW_REQUESTS = Counter("workflow_requests_total", "Total workflow requests", ["endpoint"])
WORKFLOW_LATENCY = Histogram("workflow_latency_seconds", "Workflow latency", ["endpoint"])
ERROR_COUNT = Counter("workflow_errors_total", "Total workflow errors", ["endpoint"])

# Request tracing
REQUEST_ID = ContextVar("request_id", default=None)

router = APIRouter(prefix="/api/v1/customer", tags=["customer"])

# Dependency for singleton workflow manager
def get_workflow_manager():
    """
    Helper function to get the initialized customer workflow instance.
    If the workflow was not initialized during application startup,
    this creates a new instance. Workers will be started when needed.
    """
    # Get the globally initialized workflow from the workflow_manager
    from app.core.workflow_manager import get_customer_workflow, init_customer_workflow
    
    workflow = get_customer_workflow()
    if workflow is None:
        # Fallback in case the workflow wasn't initialized in main.py
        # This shouldn't normally happen with proper initialization
        from app.agents.workflow.customer_workflow import CustomerWorkflow
        workflow = CustomerWorkflow()  # Same value as in main.py
        
        # Store the workflow in the global singleton
        init_customer_workflow(workflow)
        
        # Initialize the queue with the workflow manager
        customer_request_queue.set_workflow(workflow)
        
        # Do NOT start workers here - this is not an async function
        # Workers will be auto-started when the first request comes in
        # The customer_request_queue.enqueue_request method handles this
        
        logger.warning("Customer workflow was initialized on demand. Queue workers will start with first request.")
    
    return workflow


# Authentication endpoints (unchanged for brevity)
@router.post("/auth/register", response_model=CustomerResponse, status_code=status.HTTP_201_CREATED)
async def register_customer(customer_in: CustomerRegister, db: AsyncSession = Depends(get_db_session)):
    existing_customer = await get_customer_by_email(db, email=customer_in.email)
    if existing_customer:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")
    if customer_in.username:
        try:
            existing_username = await get_customer_by_username(db, username=customer_in.username)
            if existing_username:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username already taken")
        except ValueError:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username must be an integer")
    customer = await create_customer(db, customer_in)
    customer.is_verified = True
    return customer

@router.post("/auth/login", response_model=CustomerLoginResponse)
async def login_customer(login_data: CustomerLogin, request: Request, db: AsyncSession = Depends(get_db_session)):
    customer = None
    if login_data.username:
        customer = await authenticate_customer(db, username=login_data.username, password=login_data.password)
    if not customer:
        customer_by_email = await get_customer_by_email(db, login_data.email)
        if customer_by_email and customer_by_email.verify_password(login_data.password):
            customer = customer_by_email
    if not customer:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect email or password")
    if customer.is_locked:
        raise HTTPException(status_code=status.HTTP_423_LOCKED, detail="Account is temporarily locked")
    if not customer.is_active:
        from app.crud.crud_customer import set_customer_active
        await set_customer_active(db, customer.id, True)
        customer.status = CustomerStatus.ACTIVE
        logger.info(f"Auto-activated customer {customer.id} on login")
    permissions = get_customer_permissions(customer.role.value)
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(subject=str(customer.id), expires_delta=access_token_expires, scopes=permissions)
    refresh_token_expires = timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    refresh_token = create_refresh_token(subject=str(customer.id), expires_delta=refresh_token_expires)
    refresh_token_hash = hash_refresh_token(refresh_token)
    await create_customer_refresh_token(
        db, customer_id=customer.id, token_hash=refresh_token_hash, expires_at=datetime.utcnow() + refresh_token_expires,
        device_info=login_data.device_info, ip_address=request.client.host, user_agent=request.headers.get("user-agent")
    )
    await update_customer_last_login(db, customer, request.client.host)
    await create_customer_session(db, customer_id=customer.id, ip_address=request.client.host,
                                 user_agent=request.headers.get("user-agent"), device_info=login_data.device_info)
    if customer.total_sessions is None:
        customer.total_sessions = 0
    if customer.total_interactions is None:
        customer.total_interactions = 0
    await db.commit()
    await db.refresh(customer)
    return CustomerLoginResponse(
        access_token=access_token, refresh_token=refresh_token, token_type="bearer",
        expires_in=int(access_token_expires.total_seconds()), customer=customer
    )


@router.post("/auth/refresh", response_model=Dict[str, Any])
async def refresh_token(
    token_data: TokenRefresh,
    db: AsyncSession = Depends(get_db_session)
):
    """Refresh access token using refresh token"""
    # Verify refresh token
    token_payload = verify_token(token_data.refresh_token, token_type="refresh")
    if not token_payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )
    
    # Get token from database
    refresh_token_hash = hash_refresh_token(token_data.refresh_token)
    db_token = await get_customer_refresh_token_by_hash(db, refresh_token_hash)
    if not db_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token not found or expired"
        )
    
    # Get customer
    customer = await get_customer(db, db_token.customer_id)
    if not customer or not customer.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Customer not found or inactive"
        )
    
    # Get customer permissions
    permissions = get_customer_permissions(customer.role.value)
    
    # Create new access token
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        subject=str(customer.id),
        expires_delta=access_token_expires,
        scopes=permissions
    )
    
    # Update token last used
    db_token.last_used_at = datetime.utcnow()
    await db.commit()
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": int(access_token_expires.total_seconds())
    }

@router.post("/auth/logout")
async def logout_customer(
    token_data: TokenRefresh,
    db: AsyncSession = Depends(get_db_session)
):
    """Logout customer and revoke refresh token"""
    refresh_token_hash = hash_refresh_token(token_data.refresh_token)
    await revoke_customer_refresh_token(db, refresh_token_hash)
    
    return {"message": "Successfully logged out"}

@router.post("/auth/logout-all")
async def logout_all_sessions(
    current_customer: Customer = Depends(get_current_customer),
    db: AsyncSession = Depends(get_db_session)
):
    """Logout from all sessions and revoke all refresh tokens"""
    await revoke_all_customer_tokens(db, current_customer.id)
    
    return {"message": "Successfully logged out from all sessions"}

@router.post("/auth/change-password")
async def change_password(
    password_data: PasswordChange,
    current_customer: Customer = Depends(get_current_customer),
    db: AsyncSession = Depends(get_db_session)
):
    """Change customer password"""
    # Verify current password
    if not current_customer.verify_password(password_data.current_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect"
        )
    
    # Change password
    await change_customer_password(db, current_customer, password_data.new_password)
    
    # Revoke all existing tokens to force re-login
    await revoke_all_customer_tokens(db, current_customer.id)
    
    return {"message": "Password changed successfully"}

@router.post("/auth/request-password-reset")
async def request_password_reset(
    reset_data: PasswordReset,
    db: AsyncSession = Depends(get_db_session)
):
    """Request password reset"""
    customer = await get_customer_by_email(db, email=reset_data.email)
    if not customer:
        # Don't reveal if email exists
        return {"message": "If the email exists, a reset link has been sent"}
    
    # Generate reset token with customer ID
    reset_token = generate_password_reset_token(str(customer.id))
    
    # TODO: Send password reset email with token
    
    return {"message": "If the email exists, a reset link has been sent"}

@router.post("/auth/reset-password")
async def reset_password(
    reset_data: PasswordResetConfirm,
    db: AsyncSession = Depends(get_db_session)
):
    """Reset password using token"""
    # Verify reset token
    customer_id_str = verify_password_reset_token(reset_data.token)
    if not customer_id_str:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset token"
        )
    
    try:
        customer_id = int(customer_id_str)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid token data"
        )
    
    # Get customer by ID
    customer = await get_customer(db, customer_id)
    if not customer:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Customer not found"
        )
    
    # Change password
    await change_customer_password(db, customer, reset_data.new_password)
    
    # Revoke all existing tokens
    await revoke_all_customer_tokens(db, customer.id)
    
    return {"message": "Password reset successfully"}

# Customer profile endpoints
@router.get("/profile", response_model=CustomerResponse)
async def get_customer_profile(
    current_customer: Customer = Depends(get_current_customer)
):
    """Get current customer profile"""
    return current_customer

@router.put("/profile", response_model=CustomerResponse)
async def update_customer_profile(
    customer_update: CustomerUpdate,
    current_customer: Customer = Depends(get_current_customer),
    db: AsyncSession = Depends(get_db_session)
):
    """Update customer profile"""
    updated_customer = await update_customer(db, current_customer, customer_update)
    return updated_customer

@router.get("/profile/stats", response_model=CustomerStats)
async def get_customer_profile_stats(
    current_customer: Customer = Depends(get_current_customer),
    db: AsyncSession = Depends(get_db_session)
):
    """Get customer statistics"""
    stats = await get_customer_stats(db, current_customer.id)
    return CustomerStats(
        customer_id=current_customer.id,
        **stats
    )

# Session management endpoints
@router.get("/sessions", response_model=List[CustomerSessionResponse])
async def get_customer_sessions(
    current_customer: Customer = Depends(get_current_customer),
    db: AsyncSession = Depends(get_db_session)
):
    """Get customer active sessions"""
    sessions = await get_active_customer_sessions(db, current_customer.id)
    return sessions

# Interaction endpoints
@router.get("/interactions", response_model=List[CustomerInteractionResponse])
async def get_customer_interactions_list(
    skip: int = 0,
    limit: int = 100,
    interaction_type: Optional[CustomerInteractionType] = None,
    current_customer: Customer = Depends(get_current_customer),
    db: AsyncSession = Depends(get_db_session)
):
    """Get customer interactions"""
    interactions = await get_customer_interactions(
        db, current_customer.id, skip=skip, limit=limit, interaction_type=interaction_type
    )
    return interactions

@router.post("/interactions/{interaction_id}/feedback")
async def submit_interaction_feedback(
    interaction_id: uuid.UUID,
    feedback: CustomerFeedback,
    current_customer: Customer = Depends(get_current_customer),
    db: AsyncSession = Depends(get_db_session),
    _: bool = Depends(verify_interaction_access)
):
    """Submit feedback for an interaction"""
    try:
        # Use the unified feedback system
        from app.crud.crud_feedback import update_feedback
        interaction = await update_feedback(
            db=db,
            interaction_id=interaction_id,
            user_type='customer',
            rating=feedback.rating,
            feedback_text=feedback.feedback_text,
            was_helpful=feedback.was_helpful,
            feedback_type=feedback.feedback_type.value if feedback.feedback_type else None
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

# Chat thread endpoints
@router.get("/chat-threads", response_model=List[CustomerChatThreadResponse])
async def get_customer_chat_threads_list(
    skip: int = 0,
    limit: int = 100,
    include_archived: bool = False,
    current_customer: Customer = Depends(get_current_customer),
    db: AsyncSession = Depends(get_db_session)
):
    """Get customer chat threads"""
    threads = await get_customer_chat_threads(
        db, current_customer.id, skip=skip, limit=limit, include_archived=include_archived
    )
    return threads

@router.post("/chat-threads", response_model=CustomerChatThreadResponse)
async def create_customer_chat_thread_endpoint(
    thread_data: CustomerChatThreadCreate,
    current_customer: Customer = Depends(get_current_customer),
    db: AsyncSession = Depends(get_db_session)
):
    """Create new chat thread"""
    thread_data.customer_id = current_customer.id
    thread = await create_customer_chat_thread(db, thread_data)
    return thread

@router.get("/chat-threads/{thread_id}", response_model=CustomerChatThreadResponse)
async def get_customer_chat_thread_detail(
    thread_id: uuid.UUID,
    current_customer: Customer = Depends(get_current_customer),
    db: AsyncSession = Depends(get_db_session),
    _: bool = Depends(verify_chat_thread_access)
):
    """Get chat thread details"""
    thread = await get_customer_chat_thread(db, thread_id)
    if not thread:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat thread not found"
        )
    return thread

@router.put("/chat-threads/{thread_id}", response_model=CustomerChatThreadResponse)
async def update_customer_chat_thread_endpoint(
    thread_id: uuid.UUID,
    thread_update: CustomerChatThreadUpdate,
    current_customer: Customer = Depends(get_current_customer),
    db: AsyncSession = Depends(get_db_session),
    _: bool = Depends(verify_chat_thread_access)
):
    """Update chat thread"""
    thread = await get_customer_chat_thread(db, thread_id)
    if not thread:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat thread not found"
        )
    
    updated_thread = await update_customer_chat_thread(db, thread, thread_update)
    return updated_thread

@router.post("/chat-threads/{thread_id}/archive")
async def archive_customer_chat_thread_endpoint(
    thread_id: uuid.UUID,
    current_customer: Customer = Depends(get_current_customer),
    db: AsyncSession = Depends(get_db_session),
    _: bool = Depends(verify_chat_thread_access)
):
    """Archive chat thread"""
    thread = await archive_customer_chat_thread(db, thread_id)
    if not thread:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat thread not found"
        )
    return {"message": "Chat thread archived successfully"}

@router.delete("/chat-threads/{thread_id}")
async def delete_customer_chat_thread_endpoint(
    thread_id: uuid.UUID,
    current_customer: Customer = Depends(get_current_customer),
    db: AsyncSession = Depends(get_db_session),
    _: bool = Depends(verify_chat_thread_access)
):
    """Delete chat thread"""
    thread = await delete_customer_chat_thread(db, thread_id)
    if not thread:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat thread not found"
        )
    return {"message": "Chat thread deleted successfully"}

@router.get("/chat-threads/{thread_id}/messages", response_model=List[CustomerChatMessageResponse])
async def get_chat_thread_messages(
    thread_id: uuid.UUID,
    skip: int = 0,
    limit: int = 100,
    current_customer: Customer = Depends(get_current_customer),
    db: AsyncSession = Depends(get_db_session),
    _: bool = Depends(verify_chat_thread_access)
):
    """Get messages for a chat thread"""
    thread = await get_customer_chat_thread(db, thread_id)
    if not thread:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat thread not found"
        )
    if thread.customer_id != current_customer.id:
        raise HTTPException(status_code=403, detail="Access denied")
    messages = await get_customer_chat_thread_messages(db, thread_id, skip=skip, limit=limit)
    return messages

# Workflow endpoint
# Within the /workflow endpoint
@router.post("/workflow", response_model=CustomerWorkflowResponse)
async def customer_workflow(
    request: CustomerWorkflowRequest,
    current_customer: Customer = Depends(get_current_customer),
    db: AsyncSession = Depends(get_db_session),
    workflow_manager: CustomerWorkflow = Depends(get_workflow_manager),
    _: bool = Depends(rate_limit_workflow_requests)
):
    """
    Process a customer workflow request through the queue system
    """
    request_id = str(uuid.uuid4())
    REQUEST_ID.set(request_id)
    logger.info(f"[REQUEST:{request_id}] Processing workflow request for customer {current_customer.id}")
    WORKFLOW_REQUESTS.labels(endpoint="workflow").inc()

    # Check if premium customer for priority queue
    is_premium = current_customer.role.value == "PREMIUM_CUSTOMER"
    
    try:
        with WORKFLOW_LATENCY.labels(endpoint="workflow").time():
            # Find or create chat thread
            session_id = request.session_id or await create_customer_workflow_session(current_customer.id, current_customer.role.value)
            stmt = select(CustomerChatThread).where(
                CustomerChatThread.customer_id == current_customer.id,
                CustomerChatThread.conversation_metadata.op('->>')('session_key') == session_id,
                CustomerChatThread.is_deleted == False,
                CustomerChatThread.is_archived == False
            )
            result_thread = await db.execute(stmt)
            chat_thread = result_thread.scalar_one_or_none()

            if not chat_thread:
                thread_data = CustomerChatThreadCreate(
                    customer_id=current_customer.id,
                    title=request.query[:100] + "..." if len(request.query) > 100 else request.query,
                    summary=None,
                    primary_topic="workflow_conversation"
                )
                chat_thread = await create_customer_chat_thread(db, thread_data)
                chat_thread.conversation_metadata = {
                    "session_key": session_id, "session_type": "workflow", "created_by": "customer_workflow",
                    "initial_query": request.query, "context_data": {}, "agent_memory": {},
                    "conversation_turns": 0, "total_processing_time_ms": 0, "status": "active"
                }
                await db.commit()
                await db.refresh(chat_thread)

            # Add user message
            user_sequence = await get_next_message_sequence_number(db, chat_thread.id)
            user_message_data = CustomerChatMessageCreate(
                thread_id=chat_thread.id, sequence_number=user_sequence, speaker="customer", content=request.query
            )
            user_message = await create_customer_chat_message(db, user_message_data)

            # Create interaction record
            interaction_data = CustomerInteractionCreate(
                customer_id=current_customer.id, original_query=request.query,
                interaction_type=CustomerInteractionType.GENERAL_SEARCH,
                metadata={"request_id": request_id, "timestamp": datetime.utcnow().isoformat()}
            )
            interaction = await create_customer_interaction(db, interaction_data)

            # Get chat history
            messages = await get_customer_chat_thread_messages(db, chat_thread.id, skip=0, limit=20)
            chat_history = [
                {"role": "user" if msg.speaker == "customer" else "assistant", "content": msg.content}
                for msg in messages
            ]
            if len(chat_history) > 20:
                summary_state = await workflow_manager._summarize_context_node(
                    AgentState(chat_history=chat_history, original_query=request.query)
                )
                chat_history = summary_state.get("chat_history", chat_history[-20:])

            # Start processing time measurement
            start_time = datetime.utcnow()
            
            # Enqueue the request using the queue system
            queue_id, future = await customer_request_queue.enqueue_request(
                query=request.query,
                customer_id=str(current_customer.id),
                session_id=session_id,
                chat_history=chat_history,
                priority=current_customer.role in [CustomerRole.PREMIUM, CustomerRole.VIP]
            )
            
            # Wait for the result (blocking approach)
            try:
                # Set a generous timeout to ensure we get results
                # Use a higher timeout for better reliability (3 minutes)
                result_generator = await asyncio.wait_for(future, timeout=180)
                
                # Process the results
                result = {}
                final_response = ""
                agents_used = []
                suggested_questions = []
                
                async for event in result_generator:
                    if isinstance(event, dict) and "event" in event:
                        if event["event"] == "answer_chunk":
                            final_response += event["data"] if isinstance(event["data"], str) else ""
                        elif event["event"] == "workflow_complete" and "data" in event:
                            data = event["data"]
                            if "agents_used" in data:
                                agents_used = data["agents_used"]
                            if "suggested_questions" in data:
                                suggested_questions = data["suggested_questions"]
                    else:
                        final_response += str(event)
                
                # Add to result
                result["full_answer"] = final_response
                result["agents_used"] = agents_used
                result["suggested_questions"] = suggested_questions
                
            except asyncio.TimeoutError:
                # Handle timeout 
                logger.error(f"Request {request_id} timed out waiting for response")
                raise HTTPException(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                    detail="Request processing timed out. Please try again later."
                )

            processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)

            # Add AI response
            ai_sequence = await get_next_message_sequence_number(db, chat_thread.id)
            ai_message_data = CustomerChatMessageCreate(
                thread_id=chat_thread.id, sequence_number=ai_sequence, speaker="assistant",
                content=result.get("full_answer", "")
            )
            ai_message = await create_customer_chat_message(db, ai_message_data)
            ai_message.agent_used = result.get("agents_used", [None])[0] if result.get("agents_used") else None
            ai_message.processing_time_ms = processing_time
            ai_message.message_metadata = {
                "suggested_questions": result.get("suggested_questions", []),
                "agents_used": result.get("agents_used", []),
                "interaction_id": str(interaction.id),
                "workflow_metadata": result.get("workflow_metadata", {}),
                "processing_time_ms": processing_time
            }

            # Update interaction
            update_data = CustomerInteractionUpdate(
                agent_response=result.get("full_answer", ""),
                suggested_questions=result.get("suggested_questions", []),
                processing_time_ms=processing_time,
                workflow_metadata={
                    "agents_used": result.get("agents_used", []),
                    "session_id": session_id,
                    "customer_role": current_customer.role.value,
                    "chat_thread_id": str(chat_thread.id),
                    "user_message_id": str(user_message.id),
                    "ai_message_id": str(ai_message.id)
                },
                completed_at=datetime.utcnow()
            )
            await update_customer_interaction(db, interaction, update_data)

            # Update thread metadata atomically
            current_metadata = chat_thread.conversation_metadata or {}
            if not isinstance(current_metadata, dict):
                current_metadata = {}
            current_metadata.setdefault("context_data", {})
            current_metadata.setdefault("agent_memory", {})
            current_metadata.setdefault("conversation_turns", 0)
            current_metadata.setdefault("total_processing_time_ms", 0)

            # Prepare updated metadata
            context_update = {
                "last_query": request.query,
                "last_response_time": datetime.utcnow().isoformat(),
                "agents_used": result.get("agents_used", []),
                "primary_topic": result.get("primary_topic", "general")
            }
            current_metadata["context_data"].update(context_update)
            current_metadata["conversation_turns"] += 1
            current_metadata["total_processing_time_ms"] += processing_time
            current_metadata["agent_memory"].update({
                "topics_discussed": current_metadata["agent_memory"].get("topics_discussed", []) + [result.get("primary_topic", "general")],
                "user_preferences": current_metadata["agent_memory"].get("user_preferences", {}),
                "conversation_summary": result.get("conversation_summary", "")
            })

            # Update conversation_metadata and last_message_at in one query
            stmt = update(CustomerChatThread).where(
                CustomerChatThread.id == chat_thread.id
            ).values(
                conversation_metadata=current_metadata,
                last_message_at=datetime.utcnow()
            )
            await db.execute(stmt)

            # Update customer activity
            current_customer.last_activity_at = datetime.utcnow()
            current_customer.total_interactions = (current_customer.total_interactions or 0) + 1
            await db.commit()

            return CustomerWorkflowResponse(
                agent_response=result.get("full_answer", ""),
                suggested_questions=result.get("suggested_questions", []),
                session_id=session_id,
                interaction_id=interaction.id,
                processing_time_ms=processing_time,
                agents_used=result.get("agents_used", [])
            )

    except Exception as e:
        ERROR_COUNT.labels(endpoint="workflow").inc()
        logger.error(f"[REQUEST:{request_id}] Error in workflow: {str(e)}", exc_info=True)
        if 'interaction' in locals():
            update_data = CustomerInteractionUpdate(
                agent_response="Error processing request. Please try again.",
                workflow_metadata={"error": str(e)},
                completed_at=datetime.utcnow()
            )
            await update_customer_interaction(db, interaction, update_data)
        if 'chat_thread' in locals():
            error_sequence = await get_next_message_sequence_number(db, chat_thread.id)
            error_message_data = CustomerChatMessageCreate(
                thread_id=chat_thread.id, sequence_number=error_sequence, speaker="assistant",
                content="Error processing request. Please try again."
            )
            await create_customer_chat_message(db, error_message_data)
            await db.commit()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error_code": "WORKFLOW_FAILED", "message": str(e), "retry_suggested": "Try again later"}
        )
    
    
# Streaming workflow endpoint
@router.post("/workflow/stream")
async def customer_workflow_stream(
    request_data: CustomerWorkflowRequest,
    background_tasks: BackgroundTasks,
    current_customer: Customer = Depends(get_current_active_customer),
    db: AsyncSession = Depends(get_db_session),
    workflow_manager: CustomerWorkflow = Depends(get_workflow_manager),
    _: bool = Depends(rate_limit_workflow_requests)
):
    """
    Process a customer workflow request with streaming response through the queue system
    """
    request_id = str(uuid.uuid4())
    REQUEST_ID.set(request_id)
    logger.info(f"[REQUEST:{request_id}] Streaming workflow request for customer {current_customer.id}")
    WORKFLOW_REQUESTS.labels(endpoint="workflow_stream").inc()
    
    # Check if premium customer for priority queue
    is_premium = current_customer.role.value == "PREMIUM_CUSTOMER"
    
    try:
        with WORKFLOW_LATENCY.labels(endpoint="workflow_stream").time():
            # Find or create chat thread
            session_id = request_data.session_id or await create_customer_workflow_session(current_customer.id, current_customer.role.value)
            stmt = select(CustomerChatThread).where(
                CustomerChatThread.customer_id == current_customer.id,
                CustomerChatThread.conversation_metadata.op('->>')('session_key') == session_id,
                CustomerChatThread.is_deleted == False
            )
            result = await db.execute(stmt)
            chat_thread = result.scalar_one_or_none()
            if not chat_thread:
                thread_data = CustomerChatThreadCreate(customer_id=current_customer.id, title=request_data.query[:100])
                chat_thread = await create_customer_chat_thread(db, thread_data)
                chat_thread.conversation_metadata = {"session_key": session_id}
                await db.commit()
                await db.refresh(chat_thread)

            # Get chat history
            messages = await get_customer_chat_thread_messages(db, chat_thread.id, skip=0, limit=20)
            chat_history = [
                {"role": "user" if msg.speaker == "customer" else "assistant", "content": msg.content}
                for msg in messages
            ]
            if len(chat_history) > 20:
                summary_state = await workflow_manager._summarize_context_node(
                    AgentState(chat_history=chat_history, original_query=request_data.query)
                )
                chat_history = summary_state.get("chat_history", chat_history[-20:])

            # Create interaction record
            interaction = await create_customer_interaction(db, CustomerInteractionCreate(
                customer_id=current_customer.id, original_query=request_data.query,
                interaction_type=CustomerInteractionType.GENERAL_SEARCH,
                metadata={"request_id": request_id, "timestamp": datetime.utcnow().isoformat()}
            ))
            
            # Create user message
            user_msg = await create_customer_chat_message(
                db,
                CustomerChatMessageCreate(
                    thread_id=chat_thread.id, 
                    sequence_number=await get_next_message_sequence_number(db, chat_thread.id),
                    speaker="customer",
                    content=request_data.query
                )
            )
            
            await db.commit()
            await db.refresh(interaction)
            await db.refresh(chat_thread)

            # Stream generator
            async def stream_generator():

                full_response = ""
                agents_used = []
                suggested_questions = []
                processing_time_sec = 0.0
                start_time = time.time()

                try:
                    # Enqueue the request
                    queue_id, future = await customer_request_queue.enqueue_request(
                        query=request_data.query,
                        customer_id=str(current_customer.id),
                        session_id=session_id,
                        chat_history=chat_history,
                        priority=is_premium
                    )
                    
                    # First yield a queue event to client
                    yield f"data: {json.dumps({'event': 'queue', 'data': {'queue_id': queue_id, 'request_id': request_id}})}\n\n"
                    
                    # Wait for the result
                    result_generator = await future
                    
                    # Stream the results as they arrive
                    async for event in result_generator:
                        if isinstance(event, dict) and "event" in event:
                            # Pass through the event
                            yield f"data: {json.dumps(event, default=str)}\n\n"
                            
                            if event["event"] == "answer_chunk" and "data" in event:
                                if isinstance(event["data"], str):
                                    full_response += event["data"]
                            
                            elif event["event"] == "workflow_complete" and "data" in event:
                                data = event.get("data", {})
                                if isinstance(data, dict):
                                    agents_used = data.get("agents_used", [])
                                    suggested_questions = data.get("suggested_questions", [])
                        else:
                            # Handle raw string responses
                            chunk = str(event)
                            full_response += chunk
                            yield f"data: {json.dumps({'event': 'answer_chunk', 'data': chunk}, default=str)}\n\n"
                except Exception as e:
                    ERROR_COUNT.labels(endpoint="workflow_stream").inc()
                    logger.error(f"[REQUEST:{request_id}] Error in streaming: {str(e)}", exc_info=True)
                    error_event = {
                        "event": "error",
                        "data": {"error_code": "STREAMING_FAILED", "message": str(e), "retry_suggested": "Try again later"}
                    }
                    yield f"data: {json.dumps(error_event)}\n\n"
                    error_info = str(e)
                finally:
                    processing_time_sec = (time.time() - start_time)
                    
                    # Prepare final result data
                    final_result_data = {
                        "full_final_answer": full_response,
                        "suggested_questions": suggested_questions,
                        "agents_used": agents_used,
                        "error": getattr(locals(), 'error_info', None)
                    }
                    
                    background_tasks.add_task(
                        _save_conversation_in_background,
                        customer_id=current_customer.id, interaction_id=interaction.id, chat_thread_id=chat_thread.id,
                        session_id=session_id, request_data=request_data, final_result_data=final_result_data,
                        processing_time_sec=processing_time_sec
                    )

            return StreamingResponse(stream_generator(), media_type="text/event-stream")

    except Exception as e:
        ERROR_COUNT.labels(endpoint="workflow_stream").inc()
        logger.error(f"[REQUEST:{request_id}] Critical error in workflow stream: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error_code": "STREAMING_FAILED", "message": str(e), "retry_suggested": "Try again later"}
        )

# Background task (synchronous)
def _save_conversation_in_background(
    customer_id: int, interaction_id: uuid.UUID, chat_thread_id: uuid.UUID,
    session_id: str, request_data: CustomerWorkflowRequest, final_result_data: dict,
    processing_time_sec: float
):
    logger.info(f"[REQUEST:{REQUEST_ID.get()}] Background task started for interaction {interaction_id}")
    try:
        with SessionLocal() as db:
            full_answer = final_result_data.get("full_final_answer", "")
            suggested_questions = final_result_data.get("suggested_questions", [])
            agents_used = final_result_data.get("agents_used", [])
            processing_time_ms = int(processing_time_sec * 1000)

            chat_thread = db.query(CustomerChatThread).filter(CustomerChatThread.id == chat_thread_id).first()
            if not chat_thread:
                logger.error(f"Chat thread {chat_thread_id} not found")
                return

            max_seq = db.query(func.max(CustomerChatMessage.sequence_number)).filter(
                CustomerChatMessage.thread_id == chat_thread.id
            ).scalar() or 0

            user_msg = CustomerChatMessage(
                thread_id=chat_thread.id, sequence_number=max_seq + 1, speaker="customer",
                content=request_data.query, created_at=datetime.utcnow()
            )
            db.add(user_msg)
            db.flush()

            ai_msg = CustomerChatMessage(
                thread_id=chat_thread.id, sequence_number=max_seq + 2, speaker="assistant",
                content=full_answer, created_at=datetime.utcnow(),
                agent_used=agents_used[0] if agents_used else None,
                processing_time_ms=processing_time_ms,
                message_metadata={
                    "suggested_questions": suggested_questions,
                    "agents_used": agents_used,
                    "interaction_id": str(interaction_id)
                }
            )
            db.add(ai_msg)
            db.flush()

            interaction = db.query(CustomerInteraction).filter(CustomerInteraction.id == interaction_id).first()
            if interaction:
                workflow_metadata = {
                    "agents_used": agents_used, "session_id": session_id, "chat_thread_id": str(chat_thread.id),
                    "user_message_id": str(user_msg.id), "ai_message_id": str(ai_msg.id), "error": final_result_data.get("error")
                }
                interaction.agent_response = full_answer
                interaction.suggested_questions = suggested_questions
                interaction.processing_time_ms = processing_time_ms
                interaction.workflow_metadata = workflow_metadata
                interaction.completed_at = datetime.utcnow()

                current_metadata = chat_thread.conversation_metadata or {}
                if not isinstance(current_metadata, dict):
                    current_metadata = {}
                current_metadata.setdefault("context_data", {})
                current_metadata.setdefault("agent_memory", {})
                current_metadata.setdefault("conversation_turns", 0)
                current_metadata.setdefault("total_processing_time_ms", 0)

                current_metadata["context_data"].update({
                    "last_query": request_data.query,
                    "last_response_time": datetime.utcnow().isoformat(),
                    "agents_used": agents_used
                })
                current_metadata["conversation_turns"] += 1
                current_metadata["total_processing_time_ms"] += processing_time_ms
                # Safely update agent memory with list concatenation
                topics = current_metadata["agent_memory"].get("topics_discussed", [])
                # Make sure we're adding a list to a list
                new_topics = topics + (agents_used if isinstance(agents_used, list) else [])
                
                current_metadata["agent_memory"].update({
                    "topics_discussed": new_topics,
                    "user_preferences": current_metadata["agent_memory"].get("user_preferences", {}),
                    "conversation_summary": final_result_data.get("conversation_summary", "")
                })

                chat_thread.conversation_metadata = current_metadata
                chat_thread.last_message_at = datetime.utcnow()

                customer = db.query(Customer).filter(Customer.id == customer_id).first()
                if customer:
                    customer.last_activity_at = datetime.utcnow()
                    customer.total_interactions = (customer.total_interactions or 0) + 1

                db.commit()
                logger.info(f"[REQUEST:{REQUEST_ID.get()}] Background task completed for interaction {interaction_id}")

    except Exception as e:
        logger.error(f"[REQUEST:{REQUEST_ID.get()}] Error in background task: {str(e)}", exc_info=True)
@router.get("/health/db")
async def check_db_health(db: AsyncSession = Depends(get_db_session)):
    try:
        await db.execute(select(1))
        return {"status": "ok", "message": "Database connection healthy"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database connection failed: {str(e)}")
@router.get("/ping")
async def ping_customer_api():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat(), "message": "Customer API is working"}