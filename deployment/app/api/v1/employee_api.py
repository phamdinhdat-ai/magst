from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger
import json
import uuid
from functools import lru_cache
from contextvars import ContextVar
from prometheus_client import Counter, Histogram
from sqlalchemy import select, update, func
from tenacity import retry, stop_after_attempt, wait_exponential
from fastapi.security import HTTPBearer
from app.api.deps_employee import (
    get_current_employee, get_current_active_employee, get_current_verified_employee,
    get_current_admin_employee, get_current_manager_employee, require_employee_role,
    rate_limit_employee_workflow_requests, rate_limit_employee_chat_requests
)
from app.core.security import (
    create_access_token, create_refresh_token, verify_token, hash_refresh_token,
    generate_password_reset_token, verify_password_reset_token,
    generate_email_verification_token, verify_email_verification_token
)
from app.core.security import get_employee_permissions  
from app.core.config import settings
from app.db.session import get_db_session
from app.db.models.employee import (
    Employee, EmployeeRole, EmployeeStatus,
    # Schemas
    EmployeeCreate, EmployeeUpdate, EmployeeResponse
    , EmployeePasswordChange, EmployeeList, 
    # Chat schemas
    EmployeeChatThreadCreate, EmployeeChatThreadResponse, EmployeeChatThreadUpdate,
    EmployeeChatMessageCreate, EmployeeChatMessageResponse,
    EmployeeWorkflowRequest, EmployeeWorkflowResponse, EmployeeInteractionType, EmployeeInteractionCreate,
    
    EmployeeInteractionUpdate, EmployeeInteraction, EmployeeLogin, EmployeeRegister, 
    EmployeeLoginResponse, EmployeeChatThread, EmployeeChatMessage
)
from app.crud.crud_employee import (
    # Employee operations
    get_employee, get_employee_by_email, get_employee_by_username, create_employee,
    authenticate_employee, update_employee_last_login, update_employee, 
    change_employee_password, verify_employee_account, admin_update_employee,
    get_employees_with_filters, count_employees_with_filters, get_employee_stats,
    suspend_employee, activate_employee, deactivate_employee,
    get_employees_by_role, get_active_employees, get_recently_active_employees,
    get_employee_analytics, delete_employee, create_employee_refresh_token,
    # Chat operations
    create_employee_chat_thread, get_employee_chat_thread, get_employee_chat_threads,
    create_employee_chat_message, get_employee_chat_thread_messages,
    get_next_employee_message_sequence_number, find_or_create_employee_chat_thread_by_session,
    # Instance
    employee_crud,
    create_employee_interaction, update_employee_interaction, create_employee_session
)
from app.agents.workflow.employee_workflow import EmployeeWorkflow, create_employee_workflow_session
from app.agents.workflow.state import GraphState as AgentState
from app.api.queue_manager import employee_request_queue
import os
import asyncio
import json

async def _generate_error_stream(error_message: str):
    """Generate an error event stream"""
    error_event = {
        "event": "error",
        "data": {"error": error_message}
    }
    yield f"data: {json.dumps(error_event)}\n\n"
    yield "data: [DONE]\n\n"
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


load_dotenv()
SYNC_ENGINE = create_engine(settings.POSTGRES_DATABASE_URL.replace("postgresql+asyncpg", "postgresql"))
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=SYNC_ENGINE)

# Prometheus metrics
WORKFLOW_REQUESTS = Counter("employee_workflow_requests_total", "Total workflow requests", ["endpoint"])
WORKFLOW_LATENCY = Histogram("employee_workflow_latency_seconds", "Workflow latency", ["endpoint"])
ERROR_COUNT = Counter("employee_workflow_errors_total", "Total workflow errors", ["endpoint"])

# Request tracing
REQUEST_ID = ContextVar("request_id", default=None)

router = APIRouter(prefix="/api/v1/employee", tags=["employee"])
@lru_cache(maxsize=1)
def get_workflow_manager() -> EmployeeWorkflow:
    workflow = EmployeeWorkflow()
    # Start the queue workers with this workflow manager
    asyncio.create_task(employee_request_queue.start_workers(workflow))
    return workflow
# =====================================================================================
# Authentication Endpoints
# =====================================================================================
@router.post("/auth/register", response_model=EmployeeResponse, status_code=status.HTTP_201_CREATED)
async def register_employee(
    employee_in: EmployeeRegister,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db_session)
):  
    """Register a new employee"""
    # Check if email already exists
    existing_employee = await get_employee_by_email(db, email=employee_in.email)
    if existing_employee:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Employee with this email already exists"
        )
    existing_employee = await get_employee_by_username(db, username=employee_in.username)
    if existing_employee:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Employee with this username already exists"
        )
    # Create employee
    employee = await create_employee(db, employee_in=employee_in)
    employee.is_verified = True
    logger.info(f"Employee {employee.email} registered successfully with ID {employee.id}")
    # # Send verification email
    # background_tasks.add_task(
    #     generate_email_verification_token,employee.email
    # )
    
    logger.info(f"New employee registered: {employee.email}")
    return EmployeeResponse.from_orm(employee)

@router.post("/auth/login", response_model= EmployeeLoginResponse)
async def login_employee(
    login_data: EmployeeLogin,
    request: Request,
    db: AsyncSession = Depends(get_db_session)
):
    """Authenticate employee and return tokens"""
    # Try authentication with username first (if provided), then email
    employee = None

    if login_data.username:
        # Try username-based authentication
        employee = await authenticate_employee(db, email=login_data.email, password=login_data.password)

    if not employee:
        # Try email-based authentication 
        logger.info(f"Attempting login for employee by email: {login_data.email}")
        employee_by_email = await get_employee_by_email(db, login_data.email)
        logger.info(f"Attempting login for employee: {employee_by_email.email if employee_by_email else 'Not found'}")
        if employee_by_email and employee_by_email.verify_password(login_data.password):
            employee = employee_by_email
    logger.info(f"Employee found: {employee.email if employee else 'None'}")
    if not employee:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )
    
    # Check if account is locked
    if employee.is_locked:
        raise HTTPException(
            status_code=status.HTTP_423_LOCKED,
            detail="Account is temporarily locked due to too many failed login attempts"
        )
    if not employee.is_active: 
        # Auto-activate employee for demo purposes
        logger.info(f"Auto-activating employee {employee.email} on login")
        from app.crud.crud_employee import set_employee_active
        await set_employee_active(db, employee.email, True)
        employee.status = EmployeeStatus.ACTIVE
        logger.info(f"Auto-activated employee {employee.id} on login")

    # OPTION 3: Only check for specific statuses that should block login
    # if employee.status in [EmployeeStatus.SUSPENDED]:
    #     raise HTTPException(
    #         status_code=status.HTTP_403_FORBIDDEN,
    #         detail="Account is suspended"
    #     )
    # logger.info(f"Employee {employee.email} logged in with permissions: {permissions}")

    # Get employee permissions
    permissions = get_employee_permissions(employee.role.value)
    # logger.info(f"Employee {employee.email} logged in with permissions: {permissions}")
    # Create tokens
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        subject=str(employee.id),
        expires_delta=access_token_expires,
        scopes=permissions
    )
    
    refresh_token_expires = timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    refresh_token = create_refresh_token(
        subject=str(employee.id),
        expires_delta=refresh_token_expires
    )
    
    # Store refresh token
    refresh_token_hash = hash_refresh_token(refresh_token)
    await create_employee_refresh_token(
        db,
        employee_id=employee.id,
        token_hash=refresh_token_hash,
        expires_at=datetime.utcnow() + refresh_token_expires,
        device_info=login_data.device_info,
        ip_address=request.client.host,
        user_agent=request.headers.get("user-agent")
    )
    
    # Update last login
    await update_employee_last_login(db, employee)

    # Create session
    await create_employee_session(
        db,
        employee_id=employee.id,
        ip_address=request.client.host,
        user_agent=request.headers.get("user-agent"),
        device_info=login_data.device_info
    )
    
    # Ensure total_sessions and total_interactions are not None (for backward compatibility)
    if employee.total_sessions is None:
        employee.total_sessions = 0
    if employee.total_interactions is None:
        employee.total_interactions = 0
    await db.commit()
    await db.refresh(employee)

    return EmployeeLoginResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=int(access_token_expires.total_seconds()),
        employee=employee
    )
@router.post("/auth/change-password")
async def change_password(
    password_data: EmployeePasswordChange,
    current_employee: Employee = Depends(get_current_employee),
    db: AsyncSession = Depends(get_db_session)
):
    """Change employee password"""
    # Convert to the CRUD expected format
    from app.crud.crud_employee import EmployeePasswordUpdate
    password_update = EmployeePasswordUpdate(
        current_password=password_data.current_password,
        new_password=password_data.new_password
    )
    
    updated_employee = await change_employee_password(db, current_employee, password_update)
    
    if not updated_employee:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect"
        )
    
    logger.info(f"Employee {current_employee.email} changed password successfully")
    
    return {"message": "Password changed successfully"}

@router.post("/auth/logout")
async def logout_employee():
    """Logout employee (client-side token removal)"""
    return {"message": "Logged out successfully"}

# =====================================================================================
# Employee Profile Endpoints
# =====================================================================================

@router.get("/profile", response_model=EmployeeResponse)
async def get_employee_profile(
    current_employee: Employee = Depends(get_current_employee)
):
    """Get current employee profile"""
    return EmployeeResponse.from_orm(current_employee)

@router.put("/profile", response_model=EmployeeResponse)
async def update_employee_profile(
    employee_update: EmployeeUpdate,
    current_employee: Employee = Depends(get_current_employee),
    db: AsyncSession = Depends(get_db_session)
):
    """Update employee profile"""
    updated_employee = await update_employee(db, current_employee, employee_update)
    
    logger.info(f"Employee {current_employee.email} updated profile")
    
    return EmployeeResponse.from_orm(updated_employee)

@router.get("/profile/stats", response_model=Dict[str, Any])
async def get_employee_profile_stats(
    current_employee: Employee = Depends(get_current_employee),
    db: AsyncSession = Depends(get_db_session)
):
    """Get employee statistics"""
    stats = await get_employee_stats(db, current_employee.id)
    return stats

# =====================================================================================
# Employee Chat Thread Endpoints
# =====================================================================================

@router.post("/chat-threads", response_model=EmployeeChatThreadResponse)
async def create_chat_thread(
    thread_data: EmployeeChatThreadCreate,
    current_employee: Employee = Depends(get_current_employee),
    db: AsyncSession = Depends(get_db_session)
):
    """Create a new chat thread for the employee"""
    # Ensure thread is created for current employee
    thread_data.employee_id = current_employee.id
    thread = await create_employee_chat_thread(db, thread_data)
    return EmployeeChatThreadResponse.model_validate(thread)

@router.get("/chat-threads", response_model=List[EmployeeChatThreadResponse])
async def get_chat_threads(
    skip: int = 0,
    limit: int = 50,
    include_archived: bool = False,
    current_employee: Employee = Depends(get_current_employee),
    db: AsyncSession = Depends(get_db_session)
):
    """Get employee's chat threads"""
    threads = await get_employee_chat_threads(db, current_employee.id, skip, limit, include_archived)
    return [EmployeeChatThreadResponse.model_validate(thread) for thread in threads]

@router.get("/chat-threads/{thread_id}", response_model=EmployeeChatThreadResponse)
async def get_chat_thread(
    thread_id: uuid.UUID,
    current_employee: Employee = Depends(get_current_employee),
    db: AsyncSession = Depends(get_db_session)
):
    """Get a specific chat thread"""
    thread = await get_employee_chat_thread(db, thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Chat thread not found")
    
    # Ensure thread belongs to current employee
    if thread.employee_id != current_employee.id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return EmployeeChatThreadResponse.model_validate(thread)

@router.get("/chat-threads/{thread_id}/messages", response_model=List[EmployeeChatMessageResponse])
async def get_chat_thread_messages(
    thread_id: uuid.UUID,
    skip: int = 0,
    limit: int = 100,
    current_employee: Employee = Depends(get_current_employee),
    db: AsyncSession = Depends(get_db_session)
):
    """Get messages from a chat thread"""
    # Verify thread belongs to current employee
    thread = await get_employee_chat_thread(db, thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Chat thread not found")
    
    if thread.employee_id != current_employee.id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    messages = await get_employee_chat_thread_messages(db, thread_id, skip, limit)
    return [EmployeeChatMessageResponse.model_validate(msg) for msg in messages]

# =====================================================================================
# Employee Workflow Endpoints
# =====================================================================================
@router.post("/workflow", response_model=EmployeeWorkflowResponse)
async def employee_workflow(
    request: EmployeeWorkflowRequest,
    current_employee: Employee = Depends(get_current_employee),
    db: AsyncSession = Depends(get_db_session),
    workflow_manager: EmployeeWorkflow = Depends(get_workflow_manager),
    _: bool = Depends(rate_limit_employee_workflow_requests)
):
    request_id = str(uuid.uuid4())
    REQUEST_ID.set(request_id)
    logger.info(f"[REQUEST:{request_id}] Processing workflow request for employee {current_employee.id}")
    WORKFLOW_REQUESTS.labels(endpoint="workflow").inc()

    try:
        with WORKFLOW_LATENCY.labels(endpoint="workflow").time():
            # Find or create chat thread
            session_id = request.session_id or await create_employee_workflow_session(current_employee.id, current_employee.role.value)
            stmt = select(EmployeeChatThread).where(
                EmployeeChatThread.employee_id == current_employee.id,
                EmployeeChatThread.conversation_metadata.op('->>')('session_key') == session_id,
                EmployeeChatThread.is_deleted == False,
                EmployeeChatThread.is_archived == False
            )
            result_thread = await db.execute(stmt)
            chat_thread = result_thread.scalar_one_or_none()

            if not chat_thread:
                thread_data = EmployeeChatThreadCreate(
                    employee_id=current_employee.id,
                    title=request.query[:100] + "..." if len(request.query) > 100 else request.query,
                    summary=None,
                    primary_topic="workflow_conversation"
                )
                chat_thread = await create_employee_chat_thread(db, thread_data)
                chat_thread.conversation_metadata = {
                    "session_key": session_id, "session_type": "workflow", "created_by": "employee_workflow",
                    "initial_query": request.query, "context_data": {}, "agent_memory": {},
                    "conversation_turns": 0, "total_processing_time_ms": 0, "status": "active"
                }
                await db.commit()
                await db.refresh(chat_thread)

            # Add user message
            user_sequence = await get_next_employee_message_sequence_number(db, chat_thread.id)
            user_message_data = EmployeeChatMessageCreate(
                thread_id=chat_thread.id, sequence_number=user_sequence, speaker="employee", content=request.query
            )
            user_message = await create_employee_chat_message(db, user_message_data)

            # Create interaction record
            interaction_data = EmployeeInteractionCreate(
                employee_id=current_employee.id, original_query=request.query,
                interaction_type=EmployeeInteractionType.GENERAL_SEARCH
            )
            interaction = await create_employee_interaction(db, interaction_data)

            # Get chat history
            messages = await get_employee_chat_thread_messages(db, chat_thread.id, skip=0, limit=20)
            chat_history = [
                {"role": "user" if msg.speaker == "employee" else "assistant", "content": msg.content}
                for msg in messages
            ]
            if len(chat_history) > 20:
                summary_state = await workflow_manager._summarize_context_node(
                    AgentState(chat_history=chat_history, original_query=request.query)
                )
                chat_history = summary_state.get("chat_history", chat_history[-20:])

            # Prepare request data for the queue
            is_priority = current_employee.role in [EmployeeRole.MANAGER, EmployeeRole.ADMIN]
            start_time = datetime.utcnow()
            
            # Enqueue the request instead of directly processing
            request_data = {
                'query': request.query,
                'employee_id': str(current_employee.id),
                'employee_role': current_employee.role.value,
                'session_id': session_id,
                'prioritize': is_priority,
                'chat_history': chat_history,
                'streaming': False
            }
            
            try:
                # Add to queue with appropriate priority
                queue_id, future = await employee_request_queue.enqueue_request(
                    request_data, priority=is_priority
                )
                
                # If queue is busy, return immediate response with queue status
                if employee_request_queue.queue.qsize() > 3:  # More than 3 requests in queue
                    return EmployeeWorkflowResponse(
                        agent_response=f"Your request has been queued (position: {employee_request_queue.queue.qsize()}). Please wait a moment while we process previous requests.",
                        suggested_questions=["Check status", "Try again later"],
                        session_id=session_id,
                        thread_id=chat_thread.id,
                        processing_time_ms=0,
                        agents_used=["QueueManager"],
                        task_id=queue_id,
                        metadata={
                            "queue_status": "pending",
                            "queue_position": employee_request_queue.queue.qsize(),
                            "priority": "high" if is_priority else "normal"
                        }
                    )
                
                # Wait for processing to complete (with timeout)
                try:
                    result = await asyncio.wait_for(future, timeout=30.0)  # 30 second timeout
                except asyncio.TimeoutError:
                    # If timeout, return intermediate response but keep processing in background
                    return EmployeeWorkflowResponse(
                        agent_response="Your request is taking longer than expected. It will continue processing in the background.",
                        suggested_questions=["Check status later"],
                        session_id=session_id,
                        thread_id=chat_thread.id,
                        processing_time_ms=int((datetime.utcnow() - start_time).total_seconds() * 1000),
                        agents_used=["QueueManager"],
                        task_id=queue_id,
                        metadata={
                            "queue_status": "processing",
                            "priority": "high" if is_priority else "normal"
                        }
                    )
            
            except ValueError as e:
                # Queue is full
                logger.error(f"Queue error: {str(e)}")
                result = {
                    "full_answer": "Our system is currently experiencing high load. Please try again in a few moments.",
                    "suggested_questions": ["Try again later", "Simplify your query"],
                    "agents_used": ["QueueManager"]
                }
            except Exception as e:
                # Other queue errors
                logger.error(f"Unexpected queue error: {str(e)}")
                # Fall back to direct execution
                result = await workflow_manager.process_with_load_balancing(
                    query=request.query,
                    employee_id=str(current_employee.id),
                    employee_role=current_employee.role.value,
                    session_id=session_id,
                    prioritize=is_priority
                )

            # Process result from queue or direct execution
            if result.get("status") == "processing" or result.get("status") == "queued":
                return EmployeeWorkflowResponse(
                    agent_response="Request is being processed. Check task status with task_id.",
                    task_id=result.get("task_id"), 
                    session_id=session_id, 
                    interaction_id=interaction.id if 'interaction' in locals() else None,
                    processing_time_ms=0, 
                    agents_used=[],
                    metadata={"queue_status": result.get("status", "processing")}
                )

            processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)

            # Add AI response
            ai_sequence = await get_next_employee_message_sequence_number(db, chat_thread.id)
            ai_message_data = EmployeeChatMessageCreate(
                thread_id=chat_thread.id, 
                sequence_number=ai_sequence, 
                speaker="assistant",
                content=result.get("full_answer", "")
            )
            ai_message = await create_employee_chat_message(db, ai_message_data)
            ai_message.agent_used = result.get("agents_used", [None])[0] if result.get("agents_used") else None
            ai_message.processing_time_ms = processing_time
            
            # Add queue information to message metadata if available
            queue_metadata = {
                "queue_used": True,
                "queue_stats": employee_request_queue.get_queue_stats(),
                "processing_time_ms": processing_time
            }
            
            if not ai_message.message_metadata:
                ai_message.message_metadata = {}
            
            ai_message.message_metadata.update(queue_metadata)
            ai_message.message_metadata = {
                "suggested_questions": result.get("suggested_questions", []),
                "agents_used": result.get("agents_used", []),
                "interaction_id": str(interaction.id),
                "workflow_metadata": result.get("workflow_metadata", {}),
                "processing_time_ms": processing_time
            }

            # Update interaction
            update_data = EmployeeInteractionUpdate(
                agent_response=result.get("full_answer", ""),
                suggested_questions=result.get("suggested_questions", []),
                processing_time_ms=processing_time,
                workflow_metadata={
                    "agents_used": result.get("agents_used", []),
                    "session_id": session_id,
                    "employee_role": current_employee.role.value,
                    "chat_thread_id": str(chat_thread.id),
                    "user_message_id": str(user_message.id),
                    "ai_message_id": str(ai_message.id)
                },
                completed_at=datetime.utcnow()
            )
            await update_employee_interaction(db, interaction, update_data)

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
            stmt = update(EmployeeChatThread).where(
                EmployeeChatThread.id == chat_thread.id
            ).values(
                conversation_metadata=current_metadata,
                last_message_at=datetime.utcnow()
            )
            await db.execute(stmt)

            # Update employee activity
            current_employee.last_activity_at = datetime.utcnow()
            current_employee.total_interactions = (current_employee.total_interactions or 0) + 1
            await db.commit()

            return EmployeeWorkflowResponse(
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
            update_data = EmployeeInteractionUpdate(
                agent_response="Error processing request. Please try again.",
                workflow_metadata={"error": str(e)},
                completed_at=datetime.utcnow()
            )
            await update_employee_interaction(db, interaction, update_data)
        if 'chat_thread' in locals():
            error_sequence = await get_next_employee_message_sequence_number(db, chat_thread.id)
            error_message_data = EmployeeChatMessageCreate(
                thread_id=chat_thread.id, sequence_number=error_sequence, speaker="assistant",
                content="Error processing request. Please try again."
            )
            await create_employee_chat_message(db, error_message_data)
            await db.commit()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error_code": "WORKFLOW_FAILED", "message": str(e), "retry_suggested": "Try again later"}
        )



# @router.post("/workflow", response_model=EmployeeWorkflowResponse)
# async def employee_workflow_query(
#     request: EmployeeWorkflowRequest,
#     current_employee: Employee = Depends(get_current_active_employee),
#     db: AsyncSession = Depends(get_db_session),
#     workflow_manager: EmployeeWorkflow = Depends(get_workflow_manager),
#     _: bool = Depends(rate_limit_employee_workflow_requests)
# ):
#     """Execute employee workflow query with chat thread integration"""
#     try:
#         # Step 1: Find or create chat thread for this session
#         session_id = request.session_id or f"emp_workflow_{current_employee.id}_{uuid.uuid4().hex[:8]}"
        
#         # Find or create chat thread by session_id
#         chat_thread = await find_or_create_employee_chat_thread_by_session(
#             db, current_employee.id, session_id
#         )
        
#         # If it's a new thread, update the title with the query
#         if not chat_thread.conversation_metadata or chat_thread.conversation_metadata.get("conversation_turns", 0) == 0:
#             chat_thread.title = request.query[:100] + "..." if len(request.query) > 100 else request.query
#             chat_thread.primary_topic = "workflow_conversation"
            
#             # Ensure metadata is properly initialized
#             if not chat_thread.conversation_metadata:
#                 chat_thread.conversation_metadata = {}
            
#             chat_thread.conversation_metadata.update({
#                 "session_id": session_id,
#                 "session_type": "workflow",
#                 "created_by": "employee_workflow",
#                 "initial_query": request.query,
#                 "context_data": {},
#                 "agent_memory": {},
#                 "conversation_turns": 0,
#                 "total_processing_time_ms": 0,
#                 "status": "active"
#             })
#             await db.commit()
#             await db.refresh(chat_thread)
        
#         # Step 2: Add user message to chat thread
#         user_sequence = await get_next_employee_message_sequence_number(db, chat_thread.id)
#         user_message_data = EmployeeChatMessageCreate(
#             thread_id=chat_thread.id,
#             sequence_number=user_sequence,
#             speaker="employee",
#             content=request.query
#         )
#         user_message = await create_employee_chat_message(db, user_message_data)
        
#         # Step 3: Get conversation context from chat thread
#         context_data = chat_thread.conversation_metadata.get("context_data", {}) if chat_thread.conversation_metadata else {}
#         agent_memory = chat_thread.conversation_metadata.get("agent_memory", {}) if chat_thread.conversation_metadata else {}
        
#         # Step 4: Execute workflow with load balancing
#         start_time = datetime.utcnow()
        
#         # Use the new load balancing method with intelligent fallback
#         try:
#             logger.info(f"Processing employee query with load balancing for employee {current_employee.id}")
            
#             result = await workflow_manager.process_with_load_balancing(
#                 query=request.query,
#                 employee_id=str(current_employee.id),
#                 employee_role=current_employee.role.value,
#                 session_id=session_id,
#                 prioritize=current_employee.role in [EmployeeRole.MANAGER, EmployeeRole.ADMIN]
#             )
            
#             # Handle different result types from load balancing
#             if result.get("status") == "error":
#                 logger.error(f"Load balancing error: {result.get('message')}")
#                 raise Exception(result.get("message", "Load balancing failed"))
                
#             elif result.get("status") == "processing":
#                 # Task is queued and processing asynchronously
#                 logger.info(f"Employee query queued with task_id: {result.get('task_id')}")
#                 return EmployeeWorkflowResponse(
#                     agent_response="Your request is being processed by our distributed system. Please check back in a moment.",
#                     suggested_questions=["Check status", "Submit another query"],
#                     session_id=session_id,
#                     thread_id=chat_thread.id,
#                     processing_time_ms=int((datetime.utcnow() - start_time).total_seconds() * 1000),
#                     agents_used=["QueueManager"],
#                     task_id=result.get("task_id"),
#                     metadata={
#                         "processing_status": "queued",
#                         "load_balancing_used": True,
#                         "priority": "high" if current_employee.role in [EmployeeRole.MANAGER, EmployeeRole.ADMIN] else "normal"
#                     }
#                 )
            
#             # Extract successful workflow result data
#             full_response = result.get("full_answer", result.get("agent_response", ""))
#             suggested_questions = result.get("suggested_questions", [])
#             agents_used = result.get("agents_used", [])
#             processing_metadata = {
#                 "load_balancing_used": True,
#                 "forwarded_to_node": result.get("forwarded_to_node"),
#                 "queue_used": result.get("queue_used", False),
#                 "priority_level": result.get("priority_level")
#             }
            
#             logger.info(f"Load balancing successful for employee {current_employee.id}")
            
#         except Exception as workflow_error:
#             logger.warning(f"Load balancing failed, falling back to direct execution: {workflow_error}")
            
#             # Fallback to direct streaming execution if load balancing fails
#             config = {
#                 "configurable": {
#                     "thread_id": session_id
#                 }
#             }
            
#             # Collect all streaming events
#             full_response = ""
#             suggested_questions = []
#             agents_used = []
#             processing_metadata = {
#                 "load_balancing_used": False,
#                 "fallback_reason": str(workflow_error)
#             }
            
#             async for event in workflow_manager.arun_streaming(
#                 query=request.query,
#                 config=config,
#                 employee_id=str(current_employee.id),
#                 employee_role=current_employee.role.value
#             ):
#                 if event.get("event") == "answer_chunk":
#                     full_response += event.get("data", "")
#                 elif event.get("event") == "final_result":
#                     final_data = event.get("data", {})
#                     suggested_questions = final_data.get("suggested_questions", [])
#                     agents_used = final_data.get("agents_used", [])
        
#         processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        
#         # Step 5: Add AI response to chat thread
#         ai_sequence = await get_next_employee_message_sequence_number(db, chat_thread.id)
#         ai_message_data = EmployeeChatMessageCreate(
#             thread_id=chat_thread.id,
#             sequence_number=ai_sequence,
#             speaker="assistant",
#             content=full_response
#         )
#         ai_message = await create_employee_chat_message(db, ai_message_data)
        
#         # Update AI message with workflow metadata including load balancing info
#         ai_message.agent_used = agents_used[0] if agents_used else None
#         ai_message.processing_time_ms = processing_time
#         ai_message.message_metadata = {
#             "suggested_questions": suggested_questions,
#             "agents_used": agents_used,
#             "workflow_metadata": result.get("workflow_metadata", {}),
#             "processing_metadata": processing_metadata,
#             "processing_time_ms": processing_time,
#             "load_balancing_info": processing_metadata
#         }
        
#         # Step 6: Update chat thread metadata
#         current_metadata = chat_thread.conversation_metadata or {}
        
#         # Ensure required keys exist
#         if "context_data" not in current_metadata:
#             current_metadata["context_data"] = {}
#         if "agent_memory" not in current_metadata:
#             current_metadata["agent_memory"] = {}
        
#         context_update = {
#             "last_query": request.query,
#             "last_response_time": datetime.utcnow().isoformat(),
#             "agents_used": agents_used,
#             "primary_topic": result.get("primary_topic", "general")
#         }
#         current_metadata["context_data"].update(context_update)
#         current_metadata["conversation_turns"] = current_metadata.get("conversation_turns", 0) + 1
#         current_metadata["total_processing_time_ms"] = current_metadata.get("total_processing_time_ms", 0) + processing_time
        
#         # Update agent memory
#         memory_update = {
#             "topics_discussed": agent_memory.get("topics_discussed", []) + [result.get("primary_topic", "general")],
#             "user_preferences": agent_memory.get("user_preferences", {}),
#             "conversation_summary": result.get("conversation_summary", "")
#         }
#         current_metadata["agent_memory"].update(memory_update)
        
#         chat_thread.conversation_metadata = current_metadata
#         chat_thread.last_message_at = datetime.utcnow()
        
#         await db.commit()
        
#         logger.info(f"Employee {current_employee.email} executed workflow query: {request.query}")
        
#         response = EmployeeWorkflowResponse(
#             agent_response=full_response,
#             suggested_questions=suggested_questions,
#             session_id=session_id,
#             thread_id=chat_thread.id,
#             processing_time_ms=processing_time,
#             agents_used=agents_used,
#             metadata=processing_metadata  # Include load balancing metadata
#         )
        
#         return response
        
#     except Exception as e:
#         logger.error(f"Error in employee workflow: {str(e)}", exc_info=True)
        
#         # Add error message to chat thread if it exists
#         if 'chat_thread' in locals() and chat_thread:
#             try:
#                 error_sequence = await get_next_employee_message_sequence_number(db, chat_thread.id)
#                 error_message_data = EmployeeChatMessageCreate(
#                     thread_id=chat_thread.id,
#                     sequence_number=error_sequence,
#                     speaker="assistant",
#                     content="I apologize, but I encountered an error processing your request. Please try again."
#                 )
#                 await create_employee_chat_message(db, error_message_data)
#                 await db.commit()
#             except Exception as chat_error:
#                 logger.error(f"Error adding error message to chat: {chat_error}")
        
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="An error occurred while processing your request"
#         )


   
# Streaming workflow endpoint
@router.post("/workflow/stream")
async def employee_workflow_stream(
    request_data: EmployeeWorkflowRequest,
    background_tasks: BackgroundTasks,
    current_employee: Employee = Depends(get_current_active_employee),
    db: AsyncSession = Depends(get_db_session),
    workflow_manager: EmployeeWorkflow = Depends(get_workflow_manager),
    _: bool = Depends(rate_limit_employee_workflow_requests)
):
    request_id = str(uuid.uuid4())
    REQUEST_ID.set(request_id)
    logger.info(f"[REQUEST:{request_id}] Streaming workflow request for employee {current_employee.id}")
    WORKFLOW_REQUESTS.labels(endpoint="workflow_stream").inc()

    try:
        with WORKFLOW_LATENCY.labels(endpoint="workflow_stream").time():
            # Check if queue is full
            if employee_request_queue.queue.full():
                return StreamingResponse(
                    _generate_error_stream("Our system is currently at capacity. Please try again in a few moments."),
                    media_type="text/event-stream"
                )
            # Find or create chat thread
            session_id = request_data.session_id or await create_employee_workflow_session(current_employee.id, current_employee.role.value)
            stmt = select(EmployeeChatThread).where(
                EmployeeChatThread.employee_id == current_employee.id,
                EmployeeChatThread.conversation_metadata.op('->>')('session_key') == session_id,
                EmployeeChatThread.is_deleted == False
            )
            result = await db.execute(stmt)
            chat_thread = result.scalar_one_or_none()
            if not chat_thread:
                thread_data = EmployeeChatThreadCreate(employee_id=current_employee.id, title=request_data.query[:100])
                chat_thread = await create_employee_chat_thread(db, thread_data)
                chat_thread.conversation_metadata = {"session_key": session_id}
                await db.commit()
                await db.refresh(chat_thread)

            # Get chat history
            messages = await get_employee_chat_thread_messages(db, chat_thread.id, skip=0, limit=20)
            chat_history = [
                {"role": "user" if msg.speaker == "employee" else "assistant", "content": msg.content}
                for msg in messages
            ]
            if len(chat_history) > 20:
                summary_state = await workflow_manager._summarize_context_node(
                    AgentState(chat_history=chat_history, original_query=request_data.query)
                )
                chat_history = summary_state.get("chat_history", chat_history[-20:])

            # Create interaction record
            interaction = await create_employee_interaction(db, EmployeeInteractionCreate(
                employee_id=current_employee.id, original_query=request_data.query,
                interaction_type=EmployeeInteractionType.GENERAL_SEARCH
            ))
            await db.commit()
            await db.refresh(interaction)
            await db.refresh(chat_thread)
            
            # Check if we need to prioritize this employee
            is_priority = current_employee.role in [EmployeeRole.MANAGER, EmployeeRole.ADMIN]
            
            # Prepare request data for queue
            request_data_dict = {
                'query': request_data.query,
                'employee_id': str(current_employee.id),
                'employee_role': current_employee.role.value,
                'session_id': session_id,
                'prioritize': is_priority,
                'chat_history': chat_history,
                'config': {"thread_id": str(chat_thread.id)},
                'interaction_id': str(interaction.id),
                'streaming': True
            }

            # Stream generator
            async def stream_generator():
                final_result_data = {}
                processing_time_sec = 0.0
                start_time = datetime.utcnow()

                try:
                    # First yield a queuing status message
                    queue_size = employee_request_queue.queue.qsize()
                    if queue_size > 0:
                        queue_msg = {
                            "event": "queue_status",
                            "data": {
                                "position": queue_size,
                                "message": f"Your request is in queue position {queue_size}. Processing will begin shortly."
                            }
                        }
                        yield f"data: {json.dumps(queue_msg, default=str)}\n\n"
                    
                    # Enqueue the request with priority if applicable
                    try:
                        queue_id, future = await employee_request_queue.enqueue_request(
                            request_data_dict, priority=is_priority
                        )
                        
                        # Wait for the request to be processed and get the generator
                        stream_generator = await future
                        
                        # Stream directly from the generator returned by the queue
                        async for event in stream_generator:
                            if event.get("event") == "error":
                                ERROR_COUNT.labels(endpoint="workflow_stream").inc()
                                yield f"data: {json.dumps({'event': 'partial_error', 'data': event['data']}, default=str)}\n\n"
                            else:
                                yield f"data: {json.dumps(event, default=str)}\n\n"
                                
                            if event.get("event") == "final_result":
                                final_result_data = event.get("data", {})
                    except ValueError as e:
                        # Queue is full
                        yield f"data: {json.dumps({'event': 'error', 'data': {'error': f'Queue error: {str(e)}'}}, default=str)}\n\n"
                        logger.error(f"Queue error in streaming endpoint: {str(e)}")
                    except Exception as e:
                        # Other queue errors
                        yield f"data: {json.dumps({'event': 'error', 'data': {'error': f'Unexpected error: {str(e)}'}}, default=str)}\n\n"
                        logger.error(f"Unexpected error in queue processing: {str(e)}")
                except Exception as e:
                    ERROR_COUNT.labels(endpoint="workflow_stream").inc()
                    logger.error(f"[REQUEST:{request_id}] Error in streaming: {str(e)}", exc_info=True)
                    error_event = {
                        "event": "error",
                        "data": {"error_code": "STREAMING_FAILED", "message": str(e), "retry_suggested": "Try again later"}
                    }
                    yield f"data: {json.dumps(error_event)}\n\n"
                    final_result_data['error'] = str(e)
                finally:
                    processing_time_sec = (datetime.utcnow() - start_time).total_seconds()
                    background_tasks.add_task(
                        _save_conversation_in_background,
                        employee_id=current_employee.id, interaction_id=interaction.id, chat_thread_id=chat_thread.id,
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



# @router.post("/workflow/stream")
# async def employee_workflow_stream(
#     request_data: EmployeeWorkflowRequest,
#     background_tasks: BackgroundTasks,
#     current_employee: Employee = Depends(get_current_active_employee),
#     db: AsyncSession = Depends(get_db_session),
# ):
#     """Stream responses from the employee workflow"""
#     try:
#         # Create a dedicated workflow instance for this request
#         from app.agents.workflow.employee_workflow import EmployeeWorkflow
#         workflow_manager = EmployeeWorkflow()
#         logger.info(f"Created new workflow instance for employee {current_employee.id}")

#         # BƯỚC 1: CHUẨN BỊ NGỮ CẢNH (LOGIC QUAN TRỌNG TỪ HÀM GỐC)
#         logger.info(f"Session ID for employee {current_employee.id}: {request_data.session_id}")
#         session_id = request_data.session_id or f"emp_{current_employee.id}_{uuid.uuid4()}"
#         chat_thread = None
        
#         # Find or create chat thread
#         stmt = select(EmployeeChatThread).where(
#             EmployeeChatThread.employee_id == current_employee.id,
#             EmployeeChatThread.conversation_metadata.op('->>')('session_key') == session_id,
#             EmployeeChatThread.is_deleted == False
#         )
#     # check if chat thread already exists
#         logger.info(f"Checking for existing chat thread for employee {current_employee.id} with session_id {session_id}")
#         logger.info(f"Executing query: {stmt}")
#         logger.info(f"Query parameters: employee_id={current_employee.id}, session_key={session_id}")
#         result = await db.execute(stmt)
#         chat_thread = result.scalar_one_or_none()
            
#         logger.info(f"Found chat thread: {chat_thread.id if chat_thread else 'None'}")
#         if not chat_thread:
#             thread_data = EmployeeChatThreadCreate(
#                 employee_id=current_employee.id, 
#                 title=request_data.query[:100]
#             )
#             chat_thread = await create_employee_chat_thread(db, thread_data)
#             chat_thread.conversation_metadata = {"session_key": session_id}

#         # Get chat history
#         messages = await get_chat_thread_messages(thread_id=chat_thread.id, limit=20, skip=0, current_employee=current_employee, db=db)
#         chat_history = [
#             {"role": "user" if msg.speaker == "employee" else "assistant", "content": msg.content}
#             for msg in messages
#         ]
#         logger.info(f"Loaded {len(chat_history)} messages from chat thread {chat_thread.id} for employee {current_employee.id}: {chat_history}...")  # Log first 5 messages for brevity
#         # chat_history.reverse()  # Ensure chronological order
#         # Create interaction record
#         chat_history = chat_history[-20:]  # Limit to last 20 messages for performance
#         interaction = await create_employee_interaction(db, EmployeeInteractionCreate(
#             employee_id=current_employee.id,
#             original_query=request_data.query,
#             interaction_type=EmployeeInteractionType.GENERAL_SEARCH
#         ))
        
#         # Ensure proper connection cleanup after committing
#         await db.commit()
#         await db.refresh(interaction)
#         await db.refresh(chat_thread)
        
#         # Add explicit connection cleanup
#         from app.db.session import close_db_connections
#         background_tasks.add_task(close_db_connections)
        
#         # BƯỚC 2: XỬ LÝ YÊU CẦU VỚI WORKFLOW
#         async def stream_generator():
#             final_result_data = {}
#             processing_time_sec = 0.0
#             start_time = datetime.utcnow()
            
#             try:
#                 # Simple streaming with basic error handling
#                 logger.info(f"Starting streaming for employee {current_employee.id}, query: '{request_data.query[:50]}...'")
#                 config = {"thread_id": chat_thread.id}
#                 async for event in workflow_manager.arun_streaming_authenticated(
#                     query=request_data.query,
#                     config=config,
#                     employee_id=current_employee.id,
#                     employee_role=current_employee.role.value,
#                     interaction_id=interaction.id,
#                     chat_history=chat_history
#                 ):
                    
#                     yield f"data: {json.dumps(event, default=str)}\n\n"
#                     if event.get("event") == "final_result":
#                         final_result_data = event.get("data", {})
            
#             except Exception as e:
#                 logger.error(f"Error in streaming: {str(e)}", exc_info=True)
#                 error_event = {"event": "error", "data": {"error": str(e)}}
#                 yield f"data: {json.dumps(error_event)}\n\n"
#                 final_result_data['error'] = str(e)
            
#             finally:
#                 end_time = datetime.utcnow()
#                 processing_time_sec = (end_time - start_time).total_seconds()
                
#                 # Add explicit connection cleanup before background task
#                 from app.db.session import close_db_connections
#                 await close_db_connections()
                
#                 # Schedule background task for saving conversation
#                 background_tasks.add_task(
#                     _save_conversation_in_background,
#                     employee_id=current_employee.id,
#                     interaction_id=interaction.id,
#                     chat_thread_id=chat_thread.id,
#                     session_id=session_id,
#                     request_data=request_data,
#                     final_result_data=final_result_data,
#                     processing_time_sec=processing_time_sec
#                 )
        
#         return StreamingResponse(stream_generator(), media_type="text/event-stream")
    
#     except Exception as e:
#         # Ensure connections are cleaned up on error
#         try:
#             from app.db.session import close_db_connections
#             await close_db_connections()
#         except:
#             pass
#         logger.error(f"Critical error in workflow stream endpoint: {e}", exc_info=True)
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Failed to process request: {str(e)}"
#         )

@router.get("/admin/employees", response_model=EmployeeList)
async def get_employees_admin(
    skip: int = 0,
    limit: int = 100,
    role: Optional[EmployeeRole] = None,
    status: Optional[EmployeeStatus] = None,
    search: Optional[str] = None,
    sort_by: str = "created_at",
    sort_order: str = "desc",
    current_employee: Employee = Depends(get_current_manager_employee),
    db: AsyncSession = Depends(get_db_session)
):
    """Get employees list (Manager/Admin only)"""
    employees = await get_employees_with_filters(
        db, skip, limit, role, status, search, sort_by, sort_order
    )
    total = await count_employees_with_filters(db, role, status, search)
    
    return EmployeeList(
        total=total,
        items=[EmployeeResponse.from_orm(emp) for emp in employees]
    )

@router.get("/admin/employees/{employee_id}", response_model=EmployeeResponse)
async def get_employee_admin(
    employee_id: int,
    current_employee: Employee = Depends(get_current_manager_employee),
    db: AsyncSession = Depends(get_db_session)
):
    """Get specific employee details (Manager/Admin only)"""
    employee = await get_employee(db, employee_id)
    
    if not employee:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Employee not found"
        )
    
    return EmployeeResponse.from_orm(employee)

@router.put("/admin/employees/{employee_id}", response_model=EmployeeResponse)
async def update_employee_admin(
    employee_id: int,
    employee_update: EmployeeUpdate,
    current_employee: Employee = Depends(get_current_admin_employee),
    db: AsyncSession = Depends(get_db_session)
):
    """Update employee (Admin only)"""
    employee = await get_employee(db, employee_id)
    
    if not employee:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Employee not found"
        )
    
    # Convert to admin update format
    from app.crud.crud_employee import EmployeeAdminUpdate
    admin_update = EmployeeAdminUpdate(**employee_update.dict())
    
    updated_employee = await admin_update_employee(db, employee, admin_update)
    
    logger.info(f"Admin {current_employee.email} updated employee {employee.email}")
    
    return EmployeeResponse.from_orm(updated_employee)

@router.post("/admin/employees", response_model=EmployeeResponse, status_code=status.HTTP_201_CREATED)
async def create_employee_admin(
    employee_in: EmployeeCreate,
    current_employee: Employee = Depends(get_current_admin_employee),
    db: AsyncSession = Depends(get_db_session)
):
    """Create new employee (Admin only)"""
    # Check if email already exists
    existing_employee = await get_employee_by_email(db, email=employee_in.email)
    if existing_employee:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Employee with this email already exists"
        )
    
    # Check if username already exists (if provided)
    if employee_in.username:
        existing_username = await get_employee_by_username(db, username=employee_in.username)
        if existing_username:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Employee with this username already exists"
            )
    
    # Create employee
    employee = await create_employee(db, employee_in=employee_in)
    
    logger.info(f"Admin {current_employee.email} created new employee {employee.email}")
    
    return EmployeeResponse.from_orm(employee)

@router.post("/admin/employees/{employee_id}/suspend")
async def suspend_employee_admin(
    employee_id: int,
    current_employee: Employee = Depends(get_current_admin_employee),
    db: AsyncSession = Depends(get_db_session)
):
    """Suspend employee (Admin only)"""
    employee = await get_employee(db, employee_id)
    
    if not employee:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Employee not found"
        )
    
    if employee.id == current_employee.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot suspend your own account"
        )
    
    await suspend_employee(db, employee)
    
    logger.info(f"Admin {current_employee.email} suspended employee {employee.email}")
    
    return {"message": "Employee suspended successfully"}

@router.post("/admin/employees/{employee_id}/activate")
async def activate_employee_admin(
    employee_id: int,
    current_employee: Employee = Depends(get_current_admin_employee),
    db: AsyncSession = Depends(get_db_session)
):
    """Activate employee (Admin only)"""
    employee = await get_employee(db, employee_id)
    
    if not employee:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Employee not found"
        )
    
    await activate_employee(db, employee)
    
    logger.info(f"Admin {current_employee.email} activated employee {employee.email}")
    
    return {"message": "Employee activated successfully"}

@router.post("/admin/employees/{employee_id}/verify")
async def verify_employee_admin(
    employee_id: int,
    current_employee: Employee = Depends(get_current_admin_employee),
    db: AsyncSession = Depends(get_db_session)
):
    """Verify employee account (Admin only)"""
    employee = await get_employee(db, employee_id)
    
    if not employee:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Employee not found"
        )
    
    await verify_employee_account(db, employee)
    
    logger.info(f"Admin {current_employee.email} verified employee {employee.email}")
    
    return {"message": "Employee account verified successfully"}

@router.delete("/admin/employees/{employee_id}")
async def delete_employee_admin(
    employee_id: int,
    current_employee: Employee = Depends(get_current_admin_employee),
    db: AsyncSession = Depends(get_db_session)
):
    """Delete employee (Admin only - soft delete)"""
    employee = await get_employee(db, employee_id)
    
    if not employee:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Employee not found"
        )
    
    if employee.id == current_employee.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete your own account"
        )
    
    success = await delete_employee(db, employee_id)
    
    if success:
        logger.info(f"Admin {current_employee.email} deleted employee {employee.email}")
        return {"message": "Employee deleted successfully"}
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete employee"
        )

# =====================================================================================
# Analytics Endpoints
# =====================================================================================

@router.get("/admin/analytics", response_model=Dict[str, Any])
async def get_employee_analytics_admin(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    current_employee: Employee = Depends(get_current_admin_employee),
    db: AsyncSession = Depends(get_db_session)
):
    """Get employee analytics (Admin only)"""
    analytics = await get_employee_analytics(db, start_date, end_date)
    return analytics

@router.get("/admin/employees/active", response_model=List[EmployeeResponse])
async def get_active_employees_admin(
    skip: int = 0,
    limit: int = 100,
    current_employee: Employee = Depends(get_current_manager_employee),
    db: AsyncSession = Depends(get_db_session)
):
    """Get active employees (Manager/Admin only)"""
    employees = await get_active_employees(db, skip, limit)
    return [EmployeeResponse.from_orm(emp) for emp in employees]

@router.get("/admin/employees/recent", response_model=List[EmployeeResponse])
async def get_recently_active_employees_admin(
    hours: int = 24,
    limit: int = 50,
    current_employee: Employee = Depends(get_current_manager_employee),
    db: AsyncSession = Depends(get_db_session)
):
    """Get recently active employees (Manager/Admin only)"""
    employees = await get_recently_active_employees(db, hours, limit)
    return [EmployeeResponse.from_orm(emp) for emp in employees]

@router.get("/admin/employees/by-role/{role}", response_model=List[EmployeeResponse])
async def get_employees_by_role_admin(
    role: EmployeeRole,
    skip: int = 0,
    limit: int = 100,
    current_employee: Employee = Depends(get_current_manager_employee),
    db: AsyncSession = Depends(get_db_session)
):
    """Get employees by role (Manager/Admin only)"""
    employees = await get_employees_by_role(db, role, skip, limit)
    return [EmployeeResponse.from_orm(emp) for emp in employees]



# # Completely rewrite the background task to use synchronous SQLAlchemy
# def _save_conversation_in_background(
#     employee_id: int,
#     interaction_id: uuid.UUID,
#     chat_thread_id: uuid.UUID,
#     session_id: str,
#     request_data: EmployeeWorkflowRequest,
#     final_result_data: dict,
#     processing_time_sec: float
# ):
#     """
#     Synchronous implementation of the conversation saving logic
#     using SQLAlchemy synchronous API to avoid event loop issues
#     """
#     from sqlalchemy import create_engine
#     from sqlalchemy.orm import sessionmaker, Session
#     import os
    
#     logger.info(f"Background task started for interaction {interaction_id}")
    
#     # Create synchronous database engine and session
#     try:
#         # Get the database URL from environment or config
#         database_url = "postgresql+asyncpg://datpd:datpd@localhost:5432/gst_agents"
#         if database_url and database_url.startswith("postgresql+asyncpg"):
#             # Convert async URL to sync URL
#             database_url = database_url.replace("postgresql+asyncpg", "postgresql")
        
        
        
#         # Create engine and session
#         engine = create_engine(database_url)
#         SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        
#         # Use synchronous CRUD operations
#         with SessionLocal() as db:
#             # Extract data from workflow result
#             full_answer = final_result_data.get("full_final_answer", "")
#             logger.info(f"Full answer length: {len(full_answer)} characters")
#             suggested_questions = final_result_data.get("suggested_questions", [])
#             agents_used = final_result_data.get("agents_used", [])
#             processing_time_ms = int(processing_time_sec * 1000)
            
#             # Get chat thread (synchronous version)
#             from sqlalchemy.orm import joinedload
#             chat_thread = db.query(EmployeeChatThread).filter(
#                 EmployeeChatThread.id == chat_thread_id
#             ).first()
            
#             if not chat_thread:
#                 logger.error(f"Chat thread {chat_thread_id} not found in background task")
#                 return
            
#             # Synchronous version of sequence number generation
#             from sqlalchemy import func
#             max_seq = db.query(func.max(EmployeeChatMessage.sequence_number)).filter(
#                 EmployeeChatMessage.thread_id == chat_thread.id
#             ).scalar() or 0
            
#             # Create user message
#             user_msg = EmployeeChatMessage(
#                 thread_id=chat_thread.id,
#                 sequence_number=max_seq + 1,
#                 speaker="customer",
#                 content=request_data.query,
#                 created_at=datetime.utcnow()
#             )
#             db.add(user_msg)
#             db.flush()
            
#             # Create AI message
#             ai_msg = EmployeeChatMessage(
#                 thread_id=chat_thread.id,
#                 sequence_number=max_seq + 2,
#                 speaker="assistant",
#                 content=full_answer,
#                 created_at=datetime.utcnow(),
#                 agent_used=agents_used[0] if agents_used else None,
#                 processing_time_ms=processing_time_ms,
#                 message_metadata={
#                     "suggested_questions": suggested_questions,
#                     "agents_used": agents_used,
#                     "interaction_id": str(interaction_id)
#                 }
#             )
#             db.add(ai_msg)
#             db.flush()
            
#             # Update interaction
#             interaction = db.query(EmployeeInteraction).filter(
#                 EmployeeInteraction.id == interaction_id
#             ).first()
            
#             if interaction:
#                 workflow_metadata = {
#                     "agents_used": agents_used,
#                     "session_id": session_id,
#                     "chat_thread_id": str(chat_thread.id),
#                     "user_message_id": str(user_msg.id),
#                     "ai_message_id": str(ai_msg.id),
#                     "error": final_result_data.get("error")
#                 }
                
#                 interaction.agent_response = full_answer
#                 interaction.suggested_questions = suggested_questions
#                 interaction.processing_time_ms = processing_time_ms
#                 interaction.workflow_metadata = workflow_metadata
#                 interaction.completed_at = datetime.utcnow()
                
#                 # Update chat thread metadata
#                 current_metadata = chat_thread.conversation_metadata or {}
                
#                 if not isinstance(current_metadata, dict):
#                     current_metadata = {}
                
#                 if "context_data" not in current_metadata:
#                     current_metadata["context_data"] = {}
#                 if "agent_memory" not in current_metadata:
#                     current_metadata["agent_memory"] = {}
#                 if "conversation_turns" not in current_metadata:
#                     current_metadata["conversation_turns"] = 0
#                 if "total_processing_time_ms" not in current_metadata:
#                     current_metadata["total_processing_time_ms"] = 0
                
#                 # Update context
#                 current_metadata["context_data"].update({
#                     "last_query": request_data.query,
#                     "last_response_time": datetime.utcnow().isoformat(),
#                     "agents_used": agents_used
#                 })
#                 current_metadata["conversation_turns"] += 1
#                 current_metadata["total_processing_time_ms"] += processing_time_ms
                
#                 # Update agent memory
#                 agent_memory = current_metadata["agent_memory"]
#                 if "topics_discussed" not in agent_memory:
#                     agent_memory["topics_discussed"] = []
                
#                 if isinstance(agent_memory["topics_discussed"], list):
#                     agent_memory["topics_discussed"].extend(
#                         agents_used if isinstance(agents_used, list) else []
#                     )
                
#                 chat_thread.conversation_metadata = current_metadata
#                 chat_thread.last_message_at = datetime.utcnow()

#                 # Update employee activity
#                 employee = db.query(Employee).filter(Employee.id == employee_id).first()
#                 if employee:
#                     employee.last_activity_at = datetime.utcnow()
#                     employee.total_interactions = (employee.total_interactions or 0) + 1

#                 # Commit all changes
#                 db.commit()
#                 logger.info(f"Background task finished successfully for interaction {interaction_id}")
#             else:
#                 logger.error(f"Interaction {interaction_id} not found in background task")
                
#     except Exception as e:
#         logger.error(f"Error in background task for interaction {interaction_id}: {e}", exc_info=True)


def _save_conversation_in_background(
    employee_id: int, interaction_id: uuid.UUID, chat_thread_id: uuid.UUID,
    session_id: str, request_data: EmployeeWorkflowRequest, final_result_data: dict,
    processing_time_sec: float
):
    logger.info(f"[REQUEST:{REQUEST_ID.get()}] Background task started for interaction {interaction_id}")
    try:
        with SessionLocal() as db:
            full_answer = final_result_data.get("full_final_answer", "")
            suggested_questions = final_result_data.get("suggested_questions", [])
            agents_used = final_result_data.get("agents_used", [])
            processing_time_ms = int(processing_time_sec * 1000)

            chat_thread = db.query(EmployeeChatThread).filter(EmployeeChatThread.id == chat_thread_id).first()
            if not chat_thread:
                logger.error(f"Chat thread {chat_thread_id} not found")
                return

            max_seq = db.query(func.max(EmployeeChatMessage.sequence_number)).filter(
                EmployeeChatMessage.thread_id == chat_thread.id
            ).scalar() or 0

            user_msg = EmployeeChatMessage(
                thread_id=chat_thread.id, sequence_number=max_seq + 1, speaker="employee",
                content=request_data.query, created_at=datetime.utcnow()
            )
            db.add(user_msg)
            db.flush()

            ai_msg = EmployeeChatMessage(
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

            interaction = db.query(EmployeeInteraction).filter(EmployeeInteraction.id == interaction_id).first()
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
                current_metadata["agent_memory"].update({
                    "topics_discussed": current_metadata["agent_memory"].get("topics_discussed", []) + agents_used,
                    "user_preferences": current_metadata["agent_memory"].get("user_preferences", {}),
                    "conversation_summary": final_result_data.get("conversation_summary", "")
                })

                chat_thread.conversation_metadata = current_metadata
                chat_thread.last_message_at = datetime.utcnow()

                employee = db.query(Employee).filter(Employee.id == employee_id).first()
                if employee:
                    employee.last_activity_at = datetime.utcnow()
                    employee.total_interactions = (employee.total_interactions or 0) + 1

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
# =====================================================================================
# Health Check Endpoint
# =====================================================================================

@router.get("/health")
async def employee_health_check():
    """Employee service health check"""
    return {
        "status": "healthy",
        "service": "employee_api",
        "features": ["authentication", "profile_management", "workflow_execution", "admin_management"],
        "workflow_status": "operational",
        "timestamp": datetime.utcnow().isoformat()
    }

@router.post("/workflow/mock", response_model=EmployeeWorkflowResponse)
async def employee_workflow_mock(
    request: EmployeeWorkflowRequest,
    current_employee: Employee = Depends(get_current_active_employee),
    db: AsyncSession = Depends(get_db_session)
):
    """Mock employee workflow for testing purposes"""
    import time
    from datetime import datetime
    import asyncio
    # Simulate processing time
    await asyncio.sleep(0.5)
    
    # Create a simple mock response
    session_id = request.session_id or f"emp_mock_{current_employee.id}_{uuid.uuid4().hex[:8]}"
    
    # Create or find chat thread
    chat_thread = await find_or_create_employee_chat_thread_by_session(
        db, current_employee.id, session_id
    )
    
    # Add user message
    user_sequence = await get_next_employee_message_sequence_number(db, chat_thread.id)
    user_message_data = EmployeeChatMessageCreate(
        thread_id=chat_thread.id,
        sequence_number=user_sequence,
        speaker="employee",
        content=request.query
    )
    await create_employee_chat_message(db, user_message_data)
    
    # Create mock response
    mock_response = f"Hello {current_employee.full_name}! I received your query: '{request.query}'. This is a mock response for testing the employee workflow API. The workflow system is working properly and can handle your requests."
    
    # Add AI response
    ai_sequence = await get_next_employee_message_sequence_number(db, chat_thread.id)
    ai_message_data = EmployeeChatMessageCreate(
        thread_id=chat_thread.id,
        sequence_number=ai_sequence,
        speaker="assistant",
        content=mock_response
    )
    await create_employee_chat_message(db, ai_message_data)
    
    # Update chat thread
    chat_thread.last_message_at = datetime.utcnow()
    await db.commit()
    
    # Create response
    response = EmployeeWorkflowResponse(
        agent_response=mock_response,
        suggested_questions=[
            "How can I access employee resources?",
            "What are the company policies?",
            "How do I submit time off requests?"
        ],
        session_id=session_id,
        thread_id=chat_thread.id,
        processing_time_ms=500,
        agents_used=["MockAgent"]
    )
    
    logger.info(f"Employee {current_employee.email} used mock workflow: {request.query}")
    
    return response

# Import feedback model and crud function
from app.db.models.feedback import EmployeeFeedback
from app.crud.crud_feedback import update_feedback

@router.post("/interactions/{interaction_id}/feedback")
async def submit_interaction_feedback(
    interaction_id: uuid.UUID,
    feedback: EmployeeFeedback,
    current_employee: Employee = Depends(get_current_active_employee),
    db: AsyncSession = Depends(get_db_session)
):
    """Submit feedback for an employee interaction"""
    # Verify that the interaction belongs to this employee
    stmt = select(EmployeeInteraction).where(
        EmployeeInteraction.id == interaction_id,
        EmployeeInteraction.employee_id == current_employee.id
    )
    result = await db.execute(stmt)
    interaction = result.scalars().first()
    
    if not interaction:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Interaction not found or you don't have permission to access it"
        )
    
    # Update feedback using the unified feedback system
    try:
        await update_feedback(
            db=db,
            interaction_id=interaction_id,
            user_type='employee',
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
    
    logger.info(f"Employee {current_employee.email} submitted feedback for interaction {interaction_id}")
    return {"message": "Feedback submitted successfully"}
