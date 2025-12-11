from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Request, BackgroundTasks
from fastapi.security import HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger
import json
import uuid

from app.api.deps import (
    get_current_customer, get_current_active_customer, get_current_verified_customer,
    get_current_admin_customer, get_current_premium_customer, require_permissions,
    verify_customer_access, verify_session_access, verify_interaction_access,
    verify_chat_thread_access, rate_limit_workflow_requests, rate_limit_chat_requests,
    track_customer_activity, update_customer_last_activity
)
from app.core.security import (
    create_access_token, create_refresh_token, verify_token, hash_refresh_token,
    generate_password_reset_token, verify_password_reset_token,
    generate_email_verification_token, verify_email_verification_token,
    CustomerPermissions, get_customer_permissions
)
from app.core.config import settings
from app.db.session import get_db_session
from app.db.models.customer import (
    Customer, CustomerRole, CustomerStatus, CustomerInteractionType,
    CustomerChatThread, CustomerChatMessage,  # Added missing imports
    # Schemas
    CustomerRegister, CustomerLogin, CustomerLoginResponse, TokenRefresh,
    PasswordChange, PasswordReset, PasswordResetConfirm,
    CustomerResponse, CustomerUpdate, CustomerAdminUpdate,
    CustomerSessionResponse, CustomerInteractionResponse, CustomerInteractionUpdate,
    CustomerChatThreadResponse, CustomerChatThreadCreate, CustomerChatThreadUpdate,
    CustomerChatMessageResponse, CustomerChatMessageCreate,
    CustomerWorkflowRequest, CustomerWorkflowResponse, CustomerFeedback,
    CustomerAnalytics, CustomerList, CustomerStats,
    CustomerInteractionCreate  # Added the missing import
)
from app.crud.crud_customer import (
    # Customer operations
    get_customer, get_customer_by_email, get_customer_by_username, create_customer,
    authenticate_customer, update_customer_last_login, increment_customer_failed_login,
    update_customer, change_customer_password, verify_customer_email, admin_update_customer,
    get_customers_with_filters, get_customer_stats,
    # Session operations
    create_customer_session, get_customer_session, get_active_customer_sessions,
    update_customer_session_activity, expire_old_customer_sessions,
    # Interaction operations
    create_customer_interaction, get_customer_interaction, get_customer_interactions,
    update_customer_interaction, update_customer_interaction_feedback,
    # Chat operations
    create_customer_chat_thread, get_customer_chat_thread, get_customer_chat_threads,
    update_customer_chat_thread, archive_customer_chat_thread, delete_customer_chat_thread,
    create_customer_chat_message, get_chat_thread_messages, get_next_message_sequence_number,
    # Token operations
    create_customer_refresh_token, get_customer_refresh_token_by_hash,
    revoke_customer_refresh_token, revoke_all_customer_tokens
)
from app.api.deps import get_workflow_manager
from app.agents.workflow.customer_workflow import create_customer_workflow_session
router = APIRouter(prefix="/api/v1/customer", tags=["customer"])
security = HTTPBearer()

# Authentication endpoints
@router.post("/auth/register", response_model=CustomerResponse, status_code=status.HTTP_201_CREATED)
async def register_customer(
    customer_in: CustomerRegister,
    db: AsyncSession = Depends(get_db_session)
):
    """Register a new customer"""
    # Check if email already exists
    existing_customer = await get_customer_by_email(db, email=customer_in.email)
    if existing_customer:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Check if username already exists (if provided)
    if customer_in.username:
        try:
            existing_username = await get_customer_by_username(db, username=customer_in.username)
            if existing_username:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Username already taken"
                )
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username must be an integer"
            )
    
    # Create customer
    customer = await create_customer(db, customer_in)
    
    # TODO: Send email verification email
    # set verify default to True
    customer.is_verified = True
    
    return customer

@router.post("/auth/login", response_model=CustomerLoginResponse)
async def login_customer(
    login_data: CustomerLogin,
    request: Request,
    db: AsyncSession = Depends(get_db_session)
):
    """Authenticate customer and return tokens"""
    # Try authentication with username first (if provided), then email
    customer = None
    
    if login_data.username:
        # Try username-based authentication
        customer = await authenticate_customer(db, username=login_data.username, password=login_data.password)
    
    if not customer:
        # Try email-based authentication
        customer_by_email = await get_customer_by_email(db, login_data.email)
        if customer_by_email and customer_by_email.verify_password(login_data.password):
            customer = customer_by_email
    
    if not customer:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )
    
    # Check if account is locked
    if customer.is_locked:
        raise HTTPException(
            status_code=status.HTTP_423_LOCKED,
            detail="Account is temporarily locked due to too many failed login attempts"
        )
    
    # OPTION 1: Remove the is_active check completely (allow all registered users to login)
    # Comment out or remove these lines:
    # if not customer.is_active:
    #     raise HTTPException(
    #         status_code=status.HTTP_403_FORBIDDEN,
    #         detail="Account is not active"
    #     )
    
    # OPTION 2: Auto-activate customers on successful login (for demo/development)
    if not customer.is_active:
        # Auto-activate customer for demo purposes
        from app.crud.crud_customer import set_customer_active
        await set_customer_active(db, customer.id, True)
        customer.status = CustomerStatus.ACTIVE
        logger.info(f"Auto-activated customer {customer.id} on login")
    
    # OPTION 3: Only check for specific statuses that should block login
    # if customer.status in [CustomerStatus.SUSPENDED]:
    #     raise HTTPException(
    #         status_code=status.HTTP_403_FORBIDDEN,
    #         detail="Account is suspended"
    #     )
    
    # Get customer permissions
    permissions = get_customer_permissions(customer.role.value)
    
    # Create tokens
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        subject=str(customer.id),
        expires_delta=access_token_expires,
        scopes=permissions
    )
    
    refresh_token_expires = timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    refresh_token = create_refresh_token(
        subject=str(customer.id),
        expires_delta=refresh_token_expires
    )
    
    # Store refresh token
    refresh_token_hash = hash_refresh_token(refresh_token)
    await create_customer_refresh_token(
        db,
        customer_id=customer.id,
        token_hash=refresh_token_hash,
        expires_at=datetime.utcnow() + refresh_token_expires,
        device_info=login_data.device_info,
        ip_address=request.client.host,
        user_agent=request.headers.get("user-agent")
    )
    
    # Update last login
    await update_customer_last_login(db, customer, request.client.host)
    
    # Create session
    await create_customer_session(
        db,
        customer_id=customer.id,
        ip_address=request.client.host,
        user_agent=request.headers.get("user-agent"),
        device_info=login_data.device_info
    )
    
    # Ensure total_sessions and total_interactions are not None (for backward compatibility)
    if customer.total_sessions is None:
        customer.total_sessions = 0
    if customer.total_interactions is None:
        customer.total_interactions = 0
    await db.commit()
    await db.refresh(customer)
    
    return CustomerLoginResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=int(access_token_expires.total_seconds()),
        customer=customer
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
    interaction = await update_customer_interaction_feedback(
        db,
        interaction_id=interaction_id,
        rating=feedback.rating,
        feedback_text=feedback.feedback_text,
        was_helpful=feedback.was_helpful
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
async def get_chat_thread_messages_list(
    thread_id: uuid.UUID,
    skip: int = 0,
    limit: int = 100,
    current_customer: Customer = Depends(get_current_customer),
    db: AsyncSession = Depends(get_db_session),
    _: bool = Depends(verify_chat_thread_access)
):
    """Get messages for a chat thread"""
    messages = await get_chat_thread_messages(db, thread_id, skip=skip, limit=limit)
    return messages

# Workflow endpoint (similar to guest workflow but with authentication)
@router.post("/workflow", response_model=CustomerWorkflowResponse)
async def customer_workflow(
    request: CustomerWorkflowRequest,
    current_customer: Customer = Depends(get_current_customer),
    db: AsyncSession = Depends(get_db_session),
    _: bool = Depends(rate_limit_workflow_requests)
):
    """
    Process customer query through AI workflow with chat thread integration
    """
    try:
        # Import the customer workflow
        from app.agents.workflow.customer_workflow import CustomerWorkflow, create_customer_workflow_session
        
        # Step 1: Find or create chat thread for this session
        chat_thread = None
        session_id = request.session_id
        
        if session_id:
            # Try to find existing chat thread by session_id
            from sqlalchemy import select
            stmt = select(CustomerChatThread).where(
                CustomerChatThread.customer_id == current_customer.id,
                CustomerChatThread.conversation_metadata.op('->>')('session_key') == session_id,
                CustomerChatThread.is_deleted == False,
                CustomerChatThread.is_archived == False
            )
            result_thread = await db.execute(stmt)
            chat_thread = result_thread.scalar_one_or_none()
        
        # Create new chat thread if none exists
        if not chat_thread:
            session_id = session_id or await create_customer_workflow_session(
                current_customer.id,
                current_customer.role.value
            )
            
            # Create chat thread with session metadata
            thread_data = CustomerChatThreadCreate(
                customer_id=current_customer.id,
                title=request.query[:100] + "..." if len(request.query) > 100 else request.query,
                summary=None,
                primary_topic="workflow_conversation"
            )
            
            chat_thread = await create_customer_chat_thread(db, thread_data)
            
            # Update thread metadata with session info
            chat_thread.conversation_metadata = {
                "session_key": session_id,
                "session_type": "workflow",
                "created_by": "customer_workflow",
                "initial_query": request.query,
                "context_data": {},
                "agent_memory": {},
                "conversation_turns": 0,
                "total_processing_time_ms": 0,
                "status": "active"
            }
            await db.commit()
            await db.refresh(chat_thread)
        
        # Step 2: Add user message to chat thread
        user_sequence = await get_next_message_sequence_number(db, chat_thread.id)
        user_message_data = CustomerChatMessageCreate(
            thread_id=chat_thread.id,
            sequence_number=user_sequence,
            speaker="customer",
            content=request.query
        )
        user_message = await create_customer_chat_message(db, user_message_data)
        
        # Step 3: Create interaction record (linked to chat thread)
        interaction_data = CustomerInteractionCreate(
            customer_id=current_customer.id,
            original_query=request.query,
            interaction_type=CustomerInteractionType.GENERAL_SEARCH
        )
        interaction = await create_customer_interaction(db, interaction_data)
        
        # Step 4: Get conversation context from chat thread
        context_data = chat_thread.conversation_metadata.get("context_data", {}) if chat_thread.conversation_metadata else {}
        agent_memory = chat_thread.conversation_metadata.get("agent_memory", {}) if chat_thread.conversation_metadata else {}
        
        # Step 5: Execute workflow
        workflow_manager = CustomerWorkflow()
        start_time = datetime.utcnow()
        
        result = await workflow_manager.arun_simple_authenticated(
            query=request.query,
            customer_id=current_customer.id,
            customer_role=current_customer.role.value,
            session_id=session_id
        )
        
        processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        
        # Step 6: Add AI response to chat thread
        ai_sequence = await get_next_message_sequence_number(db, chat_thread.id)
        ai_message_data = CustomerChatMessageCreate(
            thread_id=chat_thread.id,
            sequence_number=ai_sequence,
            speaker="assistant",
            content=result.get("full_answer", "")
        )
        ai_message = await create_customer_chat_message(db, ai_message_data)
        
        # Update AI message with workflow metadata
        ai_message.agent_used = result.get("agents_used", [None])[0] if result.get("agents_used") else None
        ai_message.processing_time_ms = processing_time
        ai_message.message_metadata = {
            "suggested_questions": result.get("suggested_questions", []),
            "agents_used": result.get("agents_used", []),
            "interaction_id": str(interaction.id),
            "workflow_metadata": result.get("workflow_metadata", {}),
            "processing_time_ms": processing_time
        }
        
        # Step 7: Update interaction record
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
        
        # Step 8: Update chat thread metadata
        current_metadata = chat_thread.conversation_metadata or {}
        context_update = {
            "last_query": request.query,
            "last_response_time": datetime.utcnow().isoformat(),
            "agents_used": result.get("agents_used", []),
            "primary_topic": result.get("primary_topic", "general")
        }
        current_metadata["context_data"].update(context_update)
        current_metadata["conversation_turns"] = current_metadata.get("conversation_turns", 0) + 1
        current_metadata["total_processing_time_ms"] = current_metadata.get("total_processing_time_ms", 0) + processing_time
        
        # Update agent memory
        memory_update = {
            "topics_discussed": agent_memory.get("topics_discussed", []) + [result.get("primary_topic", "general")],
            "user_preferences": agent_memory.get("user_preferences", {}),
            "conversation_summary": result.get("conversation_summary", "")
        }
        current_metadata["agent_memory"].update(memory_update)
        
        chat_thread.conversation_metadata = current_metadata
        chat_thread.last_message_at = datetime.utcnow()
        
        # Step 9: Update customer activity
        current_customer.last_activity_at = datetime.utcnow()
        if current_customer.total_interactions is None:
            current_customer.total_interactions = 0
        current_customer.total_interactions += 1
        await db.commit()
        
        response = CustomerWorkflowResponse(
            agent_response=result.get("full_answer", ""),
            suggested_questions=result.get("suggested_questions", []),
            session_id=session_id,
            interaction_id=interaction.id,
            processing_time_ms=processing_time,
            agents_used=result.get("agents_used", [])
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error in customer workflow: {str(e)}", exc_info=True)
        
        # Update interaction with error if it was created
        if 'interaction' in locals():
            update_data = CustomerInteractionUpdate(
                agent_response="I apologize, but I encountered an error processing your request. Please try again.",
                workflow_metadata={"error": str(e)},
                completed_at=datetime.utcnow()
            )
            await update_customer_interaction(db, interaction, update_data)
        
        # Add error message to chat thread if it exists
        if 'chat_thread' in locals() and chat_thread:
            try:
                error_sequence = await get_next_message_sequence_number(db, chat_thread.id)
                error_message_data = CustomerChatMessageCreate(
                    thread_id=chat_thread.id,
                    sequence_number=error_sequence,
                    speaker="assistant",
                    content="I apologize, but I encountered an error processing your request. Please try again."
                )
                await create_customer_chat_message(db, error_message_data)
                await db.commit()
            except Exception as chat_error:
                logger.error(f"Error adding error message to chat: {chat_error}")
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing your request"
        )
        
    except Exception as e:
        logger.error(f"Error in customer workflow: {str(e)}", exc_info=True)
        
        # Update interaction with error
        if 'interaction' in locals():
            update_data = CustomerInteractionUpdate(
                agent_response="I apologize, but I encountered an error processing your request. Please try again.",
                workflow_metadata={"error": str(e)},
                completed_at=datetime.utcnow()
            )
            await update_customer_interaction(db, interaction, update_data)
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing your request"
        )

@router.post("/workflow/stream")
async def customer_workflow_stream(
    request: CustomerWorkflowRequest,
    current_customer: Customer = Depends(get_current_customer),
    db: AsyncSession = Depends(get_db_session),
    _: bool = Depends(rate_limit_workflow_requests)
):
    """
    Process customer query through AI workflow with streaming response
    """
    from fastapi.responses import StreamingResponse
    import json
    
    # Utility function for JSON serialization with datetime support
    def json_dumps_with_datetime(obj):
        return json.dumps(obj, default=str)

    async def generate_stream():
        interaction = None
        try:
            # Import the customer workflow
            from app.agents.workflow.customer_workflow import CustomerWorkflow, create_customer_workflow_session
            
            # Create interaction record
            interaction_data = CustomerInteractionCreate(
                customer_id=current_customer.id,  # Integer customer ID
                original_query=request.query,
                interaction_type=CustomerInteractionType.GENERAL_SEARCH
            )
            
            interaction = await create_customer_interaction(db, interaction_data)
            
            # Initialize workflow
            workflow_manager = CustomerWorkflow()
            
            # Create or use provided session
            session_id = request.session_id or await create_customer_workflow_session(
                current_customer.id,  # Integer customer ID
                current_customer.role.value
            )
            config = {"configurable": {"session_id": session_id}}
            
            config = {"configurable": {"thread_id": session_id}}
            start_time = datetime.utcnow()
            full_answer = ""
            
            # Stream workflow events
            async for event in workflow_manager.arun_streaming_authenticated(
                query=request.query,
                config=config,
                customer_id=current_customer.id,  # Integer customer ID
                customer_role=current_customer.role.value,
                interaction_id=interaction.id
            ):
                # Send event to client
                event_data = json_dumps_with_datetime(event) + "\n"
                yield f"data: {event_data}\n\n"
                
                # Collect full answer for database update
                if event.get("event") == "answer_chunk":
                    full_answer += event.get("data", "")
                elif event.get("event") == "final_result":
                    # Map 'full_final_answer' to 'full_answer' if present in event data
                    if "full_final_answer" in event.get("data", {}):
                        full_answer = event["data"]["full_final_answer"]
                    processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
                    
                    update_data = CustomerInteractionUpdate(
                        agent_response=full_answer,
                        suggested_questions=event.get("data", {}).get("suggested_questions", []),
                        processing_time_ms=processing_time,
                        workflow_metadata={
                            "agents_used": event.get("data", {}).get("node", []),
                            "session_id": session_id,
                            "customer_role": current_customer.role.value
                        },
                        completed_at=datetime.utcnow()
                    )
                    
                    await update_customer_interaction(db, interaction, update_data)
                    
                    # Update customer activity
                    current_customer.last_activity_at = datetime.utcnow()
                    current_customer.total_interactions += 1
                    await db.commit()
            
        except Exception as e:
            logger.error(f"Error in customer workflow stream: {str(e)}", exc_info=True)
            
            # Log the full traceback for debugging
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            
            error_event = {
                "event": "error", 
                "data": {"error": "An error occurred while processing your request"},
                "metadata": {"timestamp": datetime.utcnow().isoformat()}
            }
            yield f"data: {json_dumps_with_datetime(error_event)}\n\n"
            
            # Update interaction with error if it was created
            if interaction:
                try:
                    update_data = CustomerInteractionUpdate(
                        agent_response="I apologize, but I encountered an error processing your request.",
                        workflow_metadata={"error": str(e)},
                        completed_at=datetime.utcnow()
                    )
                    await update_customer_interaction(db, interaction, update_data)
                    await db.commit()
                except Exception as db_error:
                    logger.error(f"Error updating interaction after workflow error: {str(db_error)}")
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )

# Admin endpoints
@router.get("/admin/customers", response_model=CustomerList)
async def get_customers_admin(
    skip: int = 0,
    limit: int = 100,
    role: Optional[CustomerRole] = None,
    status: Optional[CustomerStatus] = None,
    search: Optional[str] = None,
    sort_by: str = "created_at",
    sort_order: str = "desc",
    current_customer: Customer = Depends(get_current_admin_customer),
    db: AsyncSession = Depends(get_db_session)
):
    """Get customers list (admin only)"""
    customers = await get_customers_with_filters(
        db, skip=skip, limit=limit, role=role, status=status, 
        search=search, sort_by=sort_by, sort_order=sort_order
    )
    
    # TODO: Get total count for pagination
    total = len(customers)  # Placeholder
    
    return CustomerList(
        customers=customers,
        total=total,
        page=skip // limit + 1,
        per_page=limit,
        total_pages=(total + limit - 1) // limit
    )

@router.get("/admin/customers/active", response_model=List[Dict[str, Any]])
async def get_active_customers(
    minutes: int = 15,
    current_customer: Customer = Depends(get_current_admin_customer),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Get a list of customers who have been active in the last X minutes
    Only administrators can access this endpoint
    """
    from sqlalchemy import select
    from datetime import datetime, timedelta
    
    # Calculate the cutoff time for activity
    cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)
    
    # Query for customers with activity after the cutoff time
    query = select(Customer).where(Customer.last_activity_at >= cutoff_time)
    result = await db.execute(query)
    active_customers = result.scalars().all()
    
    # Format the response
    return [
        {
            "id": customer.id,
            "email": customer.email,
            "username": customer.username,
            "first_name": customer.first_name,
            "last_name": customer.last_name,
            "last_login_at": customer.last_login_at,
            "last_activity_at": customer.last_activity_at,
            "role": customer.role.value,
            "status": customer.status.value
        }
        for customer in active_customers
    ]

@router.put("/admin/customers/{customer_id}", response_model=CustomerResponse)
async def update_customer_admin(
    customer_id: int,  # Changed from uuid.UUID to int
    customer_update: CustomerAdminUpdate,
    current_customer: Customer = Depends(get_current_admin_customer),
    db: AsyncSession = Depends(get_db_session)
):
    """Update customer (admin only)"""
    customer = await get_customer(db, customer_id)
    if not customer:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Customer not found"
        )
    
    updated_customer = await admin_update_customer(db, customer, customer_update)
    return updated_customer

@router.get("/admin/customers/{customer_id}/activity", response_model=Dict[str, Any])
async def check_customer_activity_status(
    customer_id: int,
    current_customer: Customer = Depends(get_current_admin_customer),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Check if a specific customer is currently active based on their last activity timestamp
    Only administrators can access this endpoint
    """
    from datetime import datetime, timedelta
    
    # Get the customer
    customer = await get_customer(db, customer_id)
    if not customer:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Customer not found"
        )
    
    # Check if the customer is active (has activity in the last 15 minutes)
    is_active = False
    if customer.last_activity_at:
        time_diff = datetime.utcnow() - customer.last_activity_at
        is_active = time_diff.total_seconds() < 900  # 15 minutes in seconds
    
    # Calculate time since last activity
    time_since_last_activity = None
    if customer.last_activity_at:
        time_since_last_activity = (datetime.utcnow() - customer.last_activity_at).total_seconds()
    
    # Calculate time since last login
    time_since_last_login = None
    if customer.last_login_at:
        time_since_last_login = (datetime.utcnow() - customer.last_login_at).total_seconds()
    
    return {
        "customer_id": customer.id,
        "email": customer.email,
        "username": customer.username,
        "is_active": is_active,
        "last_activity_at": customer.last_activity_at,
        "seconds_since_last_activity": time_since_last_activity,
        "last_login_at": customer.last_login_at,
        "seconds_since_last_login": time_since_last_login,
        "status": customer.status.value,
        "role": customer.role.value
    }

@router.put("/admin/customers/{customer_id}/activate", response_model=CustomerResponse)
async def activate_customer(
    customer_id: int,
    current_customer: Customer = Depends(get_current_admin_customer),
    db: AsyncSession = Depends(get_db_session)
):
    """Activate customer account (admin only)"""
    from app.crud.crud_customer import set_customer_active
    
    customer = await get_customer(db, customer_id)
    if not customer:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Customer not found"
        )
    
    success = await set_customer_active(db, customer_id, True)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to activate customer"
        )
    
    # Refresh customer data
    updated_customer = await get_customer(db, customer_id)
    logger.info(f"Customer {customer_id} activated by admin {current_customer.id}")
    
    return updated_customer

@router.put("/admin/customers/{customer_id}/deactivate", response_model=CustomerResponse)
async def deactivate_customer(
    customer_id: int,
    current_customer: Customer = Depends(get_current_admin_customer),
    db: AsyncSession = Depends(get_db_session)
):
    """Deactivate customer account (admin only)"""
    from app.crud.crud_customer import set_customer_active
    
    customer = await get_customer(db, customer_id)
    if not customer:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Customer not found"
        )
    
    # Prevent deactivating admin accounts
    if customer.role == CustomerRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot deactivate admin accounts"
        )
    
    success = await set_customer_active(db, customer_id, False)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to deactivate customer"
        )
    
    # Refresh customer data
    updated_customer = await get_customer(db, customer_id)
    logger.info(f"Customer {customer_id} deactivated by admin {current_customer.id}")
    
    return updated_customer

@router.get("/admin/analytics", response_model=CustomerAnalytics)
async def get_customer_analytics(
    current_customer: Customer = Depends(get_current_admin_customer),
    db: AsyncSession = Depends(get_db_session)
):
    """Get customer analytics (admin only)"""
    # TODO: Implement analytics calculation
    # This is a placeholder
    analytics = CustomerAnalytics(
        total_customers=0,
        active_customers=0,
        total_sessions=0,
        total_interactions=0,
        avg_interactions_per_customer=0.0,
        most_common_interaction_types=[],
        most_used_agents=[],
        avg_processing_time_ms=None,
        user_satisfaction_rating=None,
        customer_retention_rate=None
    )
    
    return analytics

@router.get("/me", response_model=CustomerResponse)
async def get_current_customer_info(
    current_customer: Customer = Depends(get_current_customer),  # Use simpler dependency to avoid async context issues
    db: AsyncSession = Depends(get_db_session)
):
    """
    Get current customer info and track activity
    This endpoint will automatically update the customer's last_activity_at timestamp
    """
    try:
        # Log customer data for debugging
        logger.debug(f"Customer data for ID {current_customer.id}: total_sessions={current_customer.total_sessions}, total_interactions={current_customer.total_interactions}")
        
        # Ensure numeric fields are not None
        total_sessions = current_customer.total_sessions if current_customer.total_sessions is not None else 0
        total_interactions = current_customer.total_interactions if current_customer.total_interactions is not None else 0
        
        # Create explicit response to ensure all required fields are present
        customer_response = CustomerResponse(
            id=current_customer.id,
            email=current_customer.email,
            username=current_customer.username,
            first_name=current_customer.first_name,
            last_name=current_customer.last_name,
            phone_number=current_customer.phone_number,
            preferred_language=current_customer.preferred_language or "vi",
            timezone=current_customer.timezone or "Asia/Ho_Chi_Minh",
            role=current_customer.role,
            status=current_customer.status,
            is_verified=current_customer.is_verified,
            email_verified=current_customer.email_verified,
            phone_verified=current_customer.phone_verified,
            created_at=current_customer.created_at,
            updated_at=current_customer.updated_at,
            last_login_at=current_customer.last_login_at,
            last_activity_at=current_customer.last_activity_at,
            total_sessions=total_sessions,
            total_interactions=total_interactions,
            full_name=current_customer.full_name,
            is_active=current_customer.is_active
        )
        logger.debug(f"Customer response created successfully for ID {current_customer.id}")
        
        # Manually track customer activity
        try:
            await update_customer_last_activity(db, current_customer.id)
            logger.debug(f"Activity tracked for customer ID={current_customer.id}")
        except Exception as activity_error:
            logger.warning(f"Failed to track activity for customer ID={current_customer.id}: {activity_error}")
        
        return customer_response
    except Exception as e:
        logger.error(f"Error creating customer response for ID {current_customer.id}: {str(e)}")
        logger.error(f"Customer object: {current_customer.__dict__}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error serializing customer data: {str(e)}"
        )
