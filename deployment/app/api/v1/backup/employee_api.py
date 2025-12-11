from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Request, BackgroundTasks
from fastapi.security import HTTPBearer
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger
import uuid
import json
import asyncio
import uuid
import json

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
from app.core.config import settings
from app.db.session import get_db_session
from app.db.models.employee import (
    Employee, EmployeeRole, EmployeeStatus,
    # Schemas
    EmployeeCreate, EmployeeUpdate, EmployeeResponse, EmployeeList,
    EmployeeLoginRequest, EmployeePasswordChange,
    # Chat schemas
    EmployeeChatThreadCreate, EmployeeChatThreadResponse, EmployeeChatThreadUpdate,
    EmployeeChatMessageCreate, EmployeeChatMessageResponse,
    EmployeeWorkflowRequest, EmployeeWorkflowResponse
)
from app.crud.crud_employee import (
    # Employee operations
    get_employee, get_employee_by_email, get_employee_by_username, create_employee,
    authenticate_employee, update_employee_last_login, update_employee, 
    change_employee_password, verify_employee_account, admin_update_employee,
    get_employees_with_filters, count_employees_with_filters, get_employee_stats,
    suspend_employee, activate_employee, deactivate_employee,
    get_employees_by_role, get_active_employees, get_recently_active_employees,
    get_employee_analytics, delete_employee,
    # Chat operations
    create_employee_chat_thread, get_employee_chat_thread, get_employee_chat_threads,
    create_employee_chat_message, get_employee_chat_thread_messages,
    get_next_employee_message_sequence_number, find_or_create_employee_chat_thread_by_session,
    # Instance
    employee_crud
)
from app.agents.workflow.employee_workflow import EmployeeWorkflow

router = APIRouter(prefix="/api/v1/employee", tags=["employee"])
security = HTTPBearer()

# Initialize employee workflow
employee_workflow = EmployeeWorkflow()

# =====================================================================================
# Authentication Endpoints
# =====================================================================================

@router.post("/auth/login", response_model=Dict[str, Any])
async def login_employee(
    login_data: EmployeeLoginRequest,
    request: Request,
    db: AsyncSession = Depends(get_db_session)
):
    """Authenticate employee and return tokens"""
    # Authenticate employee
    employee = await authenticate_employee(db, email=login_data.email, password=login_data.password)
    
    if not employee:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
    
    # Check if account is active
    if not employee.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is not active. Please contact administrator."
        )
    
    # Check if account is verified
    if not employee.is_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is not verified. Please check your email for verification link."
        )
    
    # Create tokens
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        subject=str(employee.id),
        expires_delta=access_token_expires
    )
    
    refresh_token_expires = timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    refresh_token = create_refresh_token(
        subject=str(employee.id),
        expires_delta=refresh_token_expires
    )
    
    # Update last login
    await update_employee_last_login(db, employee)
    
    logger.info(f"Employee {employee.email} logged in successfully")
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "expires_in": settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        "employee": EmployeeResponse.from_orm(employee)
    }

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
async def employee_workflow_query(
    request: EmployeeWorkflowRequest,
    current_employee: Employee = Depends(get_current_active_employee),
    db: AsyncSession = Depends(get_db_session),
    _: bool = Depends(rate_limit_employee_workflow_requests)
):
    """Execute employee workflow query with chat thread integration"""
    try:
        # Step 1: Find or create chat thread for this session
        session_id = request.session_id or f"emp_workflow_{current_employee.id}_{uuid.uuid4().hex[:8]}"
        
        # Find or create chat thread by session_id
        chat_thread = await find_or_create_employee_chat_thread_by_session(
            db, current_employee.id, session_id
        )
        
        # If it's a new thread, update the title with the query
        if not chat_thread.conversation_metadata or chat_thread.conversation_metadata.get("conversation_turns", 0) == 0:
            chat_thread.title = request.query[:100] + "..." if len(request.query) > 100 else request.query
            chat_thread.primary_topic = "workflow_conversation"
            
            # Ensure metadata is properly initialized
            if not chat_thread.conversation_metadata:
                chat_thread.conversation_metadata = {}
            
            chat_thread.conversation_metadata.update({
                "session_id": session_id,
                "session_type": "workflow",
                "created_by": "employee_workflow",
                "initial_query": request.query,
                "context_data": {},
                "agent_memory": {},
                "conversation_turns": 0,
                "total_processing_time_ms": 0,
                "status": "active"
            })
            await db.commit()
            await db.refresh(chat_thread)
        
        # Step 2: Add user message to chat thread
        user_sequence = await get_next_employee_message_sequence_number(db, chat_thread.id)
        user_message_data = EmployeeChatMessageCreate(
            thread_id=chat_thread.id,
            sequence_number=user_sequence,
            speaker="employee",
            content=request.query
        )
        user_message = await create_employee_chat_message(db, user_message_data)
        
        # Step 3: Get conversation context from chat thread
        context_data = chat_thread.conversation_metadata.get("context_data", {}) if chat_thread.conversation_metadata else {}
        agent_memory = chat_thread.conversation_metadata.get("agent_memory", {}) if chat_thread.conversation_metadata else {}
        
        # Step 4: Execute workflow
        start_time = datetime.utcnow()
        
        config = {
            "configurable": {
                "thread_id": session_id
            }
        }
        
        # Collect all streaming events
        full_response = ""
        suggested_questions = []
        agents_used = []
        metadata = {}
        
        async for event in employee_workflow.arun_streaming(
            query=request.query,
            config=config,
            employee_id=str(current_employee.id)
        ):
            if event.get("event") == "answer_chunk":
                full_response += event.get("data", "")
            elif event.get("event") == "final_result":
                final_data = event.get("data", {})
                suggested_questions = final_data.get("suggested_questions", [])
                agents_used = final_data.get("agents_used", [])
                metadata = final_data.get("metadata", {})
        
        processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        
        # Step 5: Add AI response to chat thread
        ai_sequence = await get_next_employee_message_sequence_number(db, chat_thread.id)
        ai_message_data = EmployeeChatMessageCreate(
            thread_id=chat_thread.id,
            sequence_number=ai_sequence,
            speaker="assistant",
            content=full_response
        )
        ai_message = await create_employee_chat_message(db, ai_message_data)
        
        # Update AI message with workflow metadata
        ai_message.agent_used = agents_used[0] if agents_used else None
        ai_message.processing_time_ms = processing_time
        ai_message.message_metadata = {
            "suggested_questions": suggested_questions,
            "agents_used": agents_used,
            "workflow_metadata": metadata,
            "processing_time_ms": processing_time
        }
        
        # Step 6: Update chat thread metadata
        current_metadata = chat_thread.conversation_metadata or {}
        
        # Ensure required keys exist
        if "context_data" not in current_metadata:
            current_metadata["context_data"] = {}
        if "agent_memory" not in current_metadata:
            current_metadata["agent_memory"] = {}
        
        context_update = {
            "last_query": request.query,
            "last_response_time": datetime.utcnow().isoformat(),
            "agents_used": agents_used,
            "primary_topic": metadata.get("primary_topic", "general")
        }
        current_metadata["context_data"].update(context_update)
        current_metadata["conversation_turns"] = current_metadata.get("conversation_turns", 0) + 1
        current_metadata["total_processing_time_ms"] = current_metadata.get("total_processing_time_ms", 0) + processing_time
        
        # Update agent memory
        memory_update = {
            "topics_discussed": agent_memory.get("topics_discussed", []) + [metadata.get("primary_topic", "general")],
            "user_preferences": agent_memory.get("user_preferences", {}),
            "conversation_summary": metadata.get("conversation_summary", "")
        }
        current_metadata["agent_memory"].update(memory_update)
        
        chat_thread.conversation_metadata = current_metadata
        chat_thread.last_message_at = datetime.utcnow()
        
        await db.commit()
        
        logger.info(f"Employee {current_employee.email} executed workflow query: {request.query}")
        
        response = EmployeeWorkflowResponse(
            agent_response=full_response,
            suggested_questions=suggested_questions,
            session_id=session_id,
            thread_id=chat_thread.id,
            processing_time_ms=processing_time,
            agents_used=agents_used
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error in employee workflow: {str(e)}", exc_info=True)
        
        # Add error message to chat thread if it exists
        if 'chat_thread' in locals() and chat_thread:
            try:
                error_sequence = await get_next_employee_message_sequence_number(db, chat_thread.id)
                error_message_data = EmployeeChatMessageCreate(
                    thread_id=chat_thread.id,
                    sequence_number=error_sequence,
                    speaker="assistant",
                    content="I apologize, but I encountered an error processing your request. Please try again."
                )
                await create_employee_chat_message(db, error_message_data)
                await db.commit()
            except Exception as chat_error:
                logger.error(f"Error adding error message to chat: {chat_error}")
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing your request"
        )

@router.post("/workflow/stream")
async def employee_workflow_stream(
    request: Dict[str, Any],
    current_employee: Employee = Depends(get_current_active_employee),
    db: AsyncSession = Depends(get_db_session),
    _: bool = Depends(rate_limit_employee_workflow_requests)
):
    """Stream employee workflow query results"""
    import json
    query = request.get("query", "").strip()
    
    if not query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query is required"
        )
    
    # Generate session ID for this workflow execution
    session_id = f"emp_stream_{current_employee.id}_{uuid.uuid4().hex[:8]}"
    
    config = {
        "configurable": {
            "thread_id": session_id
        }
    }
    
    async def generate_stream():
        """Generator function for streaming response"""
        try:
            # Send initial metadata
            yield f"data: {json.dumps({'event': 'start', 'data': {'session_id': session_id, 'employee_id': current_employee.id}})}\n\n"
            
            async for event in employee_workflow.arun_streaming(
                query=query,
                config=config,
                employee_id=str(current_employee.id)
            ):
                if event.get("event") == "answer_chunk":
                    full_response += event.get("data", "")
                elif event.get("event") == "final_result":
                    final_data = event.get("data", {})
                    suggested_questions = final_data.get("suggested_questions", [])
                    agents_used = final_data.get("agents_used", [])
                    metadata = final_data.get("metadata", {})
                # Stream each event as SSE
                yield f"data: {json.dumps(event)}\n\n"
            
            # Send completion event
            yield f"data: {json.dumps({'event': 'complete', 'data': {'timestamp': datetime.utcnow().isoformat()}})}\n\n"
            
        except Exception as e:
            logger.error(f"Employee workflow streaming error for {current_employee.email}: {str(e)}")
            error_event = {
                "event": "error",
                "data": {"error": str(e), "timestamp": datetime.utcnow().isoformat()}
            }
            yield f"data: {json.dumps(error_event)}\n\n"
    
    logger.info(f"Employee {current_employee.email} started streaming workflow query: {query}")
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )

# =====================================================================================
# Manager/Admin Endpoints
# =====================================================================================

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
