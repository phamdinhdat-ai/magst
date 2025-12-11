import uuid
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func, desc, and_, or_, asc, update
from sqlalchemy.orm import selectinload, joinedload
from passlib.context import CryptContext
from loguru import logger
from app.db.models.employee import (
    Employee, EmployeeRole, EmployeeStatus, 
    EmployeeChatThread, EmployeeChatMessage,
    EmployeeChatThreadCreate, EmployeeChatMessageCreate,
    EmployeeChatThreadUpdate, EmployeeChatMessageUpdate,
    EmployeeInteractionCreate, EmployeeInteractionUpdate, 
    EmployeeInteraction, EmployeeRefreshToken, EmployeeSession
)
from app.core.security import generate_session_id
from app.core.config import settings
# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Pydantic models for Employee CRUD operations
from pydantic import BaseModel, EmailStr, Field, validator

class EmployeeBase(BaseModel):
    email: EmailStr
    username: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    phone_number: Optional[str] = None
    role: EmployeeRole = EmployeeRole.EMPLOYEE
    status: EmployeeStatus = EmployeeStatus.PENDING_VERIFICATION

class EmployeeCreate(EmployeeBase):
    password: str = Field(..., min_length=8, description="Employee password")


class EmployeeUpdate(BaseModel):
    email: Optional[EmailStr] = None
    username: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    phone_number: Optional[str] = None
    role: Optional[EmployeeRole] = None
    status: Optional[EmployeeStatus] = None

class EmployeeAdminUpdate(EmployeeUpdate):
    """Admin-only update fields"""
    is_verified: Optional[bool] = None

class EmployeePasswordUpdate(BaseModel):
    current_password: str
    new_password: str = Field(..., min_length=8)
    
    @validator('new_password')
    def validate_new_password(cls, v):
        if len(v) < 8:
            raise ValueError('New password must be at least 8 characters long')
        return v

class EmployeeResponse(EmployeeBase):
    id: int
    is_verified: bool
    created_at: datetime
    updated_at: datetime
    last_login_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True

# =====================================================================================
# Helper Functions for Password Management
# =====================================================================================

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Generate password hash"""
    return pwd_context.hash(password)

# =====================================================================================
# Employee CRUD Operations
# =====================================================================================

async def create_employee(
    db: AsyncSession, 
    *, 
    employee_in: EmployeeCreate
) -> Employee:
    """Create a new employee with password hashing"""
    # Hash the password
    hashed_password = get_password_hash(employee_in.password)
    obj_in_data = employee_in.model_dump()
    password = obj_in_data.pop("password")
    # If email is not provided, generate one using username and UUID
    logger.info(f"Creating employee with email: {employee_in.email} and username: {employee_in.username}")
    email = employee_in.email
    username = employee_in.username
    if obj_in_data.get("email") is None:
        # Find the next available email (integer)
        result = await db.execute(
            select(func.max(Employee.email)).where(Employee.email.isnot(None))
        )
        max_email = result.scalar()
        obj_in_data["email"] = (max_email or 0) + 1

    # Create employee object
    db_obj = Employee(**obj_in_data)
    db_obj.set_password(password)
    logger.info(f"Setting hashed password for employee: {db_obj.email}")
    db.add(db_obj)
    await db.commit()
    await db.refresh(db_obj)
    return db_obj

async def get_employee(db: AsyncSession, employee_id: str) -> Optional[Employee]:
    """Get employee by ID"""
    try:
        print(f"Looking up employee with ID: {employee_id} (type: {type(employee_id).__name__})")
        
        # Convert string employee_id to integer for database lookup
        try:
            employee_id_int = int(employee_id)
        except (ValueError, TypeError):
            print(f"Invalid employee_id format: {employee_id}")
            return None
        
        # Direct lookup with the integer ID
        result = await db.execute(
            select(Employee)
            .filter(Employee.id == employee_id_int)
            .options(selectinload(Employee.documents))
        )
        employee = result.scalars().first()
        
        # If found, return it
        if employee:
            print(f"Found employee with ID: {employee.id}")
            return employee
        
        # Debug query to show all employees if not found
        debug_result = await db.execute(select(Employee.id, Employee.email).limit(5))
        samples = debug_result.fetchall()
        print(f"Employee sample records (first 5): {samples}")
        print(f"Employee with ID {employee_id_int} not found")
        return None
        
    except Exception as e:
        print(f"Error in get_employee: {str(e)}")
        return None

async def get_employee_by_email(db: AsyncSession, email: str) -> Optional[Employee]:
    """Get employee by email"""
    result = await db.execute(
        select(Employee).filter(Employee.email == email)
    )
    return result.scalars().first()

async def get_employee_by_username(db: AsyncSession, username: str) -> Optional[Employee]:
    """Get employee by username"""
    result = await db.execute(
        select(Employee).filter(Employee.username == username)
    )
    return result.scalars().first()

async def update_employee_interaction(
    db: AsyncSession, interaction: EmployeeInteraction, update_data: EmployeeInteractionUpdate
) -> EmployeeInteraction:
    """Update employee interaction"""
    update_dict = update_data.model_dump(exclude_unset=True)
    
    for field, value in update_dict.items():
        setattr(interaction, field, value)
    
    await db.commit()
    await db.refresh(interaction)
    return interaction

async def create_employee_interaction(
    db: AsyncSession, interaction_in: EmployeeInteractionCreate
) -> EmployeeInteraction:
    """Create employee interaction"""
    db_obj = EmployeeInteraction(**interaction_in.model_dump())
    db.add(db_obj)
    await db.commit()
    await db.refresh(db_obj)
    return db_obj



async def authenticate_employee(db: AsyncSession, email: str, password: str) -> Optional[Employee]:
    """Authenticate employee by email and password"""
    employee = await get_employee_by_email(db, email)
    logger.info(f"Authenticating employee: {email}")
    if not employee:
        return None
    # logger.info(f"Found employee: {employee.email} (ID: {employee.id}, Password Hash: {employee.hashed_password})")
    
    if not employee.verify_password(password):
        logger.warning(f"Authentication failed for employee: {email}")
        return None
    employee.is_verified = True  # Automatically verify on successful login
    logger.info("Employee authenticated successfully")
    return employee

async def update_employee(
    db: AsyncSession,
    employee: Employee,
    employee_update: EmployeeUpdate
) -> Employee:
    """Update employee profile"""
    update_data = employee_update.model_dump(exclude_unset=True)
    
    for field, value in update_data.items():
        setattr(employee, field, value)
    
    employee.updated_at = datetime.utcnow()
    
    await db.commit()
    await db.refresh(employee)
    return employee

async def admin_update_employee(
    db: AsyncSession,
    employee: Employee,
    admin_update: EmployeeAdminUpdate
) -> Employee:
    """Admin update with additional fields"""
    update_data = admin_update.model_dump(exclude_unset=True)
    
    for field, value in update_data.items():
        setattr(employee, field, value)
    
    employee.updated_at = datetime.utcnow()
    
    await db.commit()
    await db.refresh(employee)
    return employee

async def change_employee_password(
    db: AsyncSession,
    employee: Employee,
    password_update: EmployeePasswordUpdate
) -> Optional[Employee]:
    """Change employee password"""
    # Verify current password
    if not verify_password(password_update.current_password, employee.hashed_password):
        return None
    
    # Set new password
    employee.hashed_password = get_password_hash(password_update.new_password)
    employee.updated_at = datetime.utcnow()
    
    await db.commit()
    await db.refresh(employee)
    return employee

async def set_employee_active(db: AsyncSession, email: str, is_active: bool) -> bool:
    """Set employee active status"""
    result = await db.execute(
        select(Employee).where(Employee.email == email)
    )
    employee = result.scalar_one_or_none()
    if not employee:
        return False

    employee.status = EmployeeStatus.ACTIVE if is_active else EmployeeStatus.INACTIVE
    employee.updated_at = datetime.utcnow()
    employee.is_verified = True  # Automatically verify when activating

    await db.commit()
    return True



async def update_employee_last_login(
    db: AsyncSession, 
    employee: Employee
) -> Employee:
    """Update employee's last login timestamp"""
    employee.last_login_at = datetime.utcnow()
    
    await db.commit()
    await db.refresh(employee)
    return employee

async def verify_employee_account(db: AsyncSession, employee: Employee) -> Employee:
    """Mark employee account as verified"""
    employee.is_verified = True
    if employee.status == EmployeeStatus.PENDING_VERIFICATION:
        employee.status = EmployeeStatus.ACTIVE
    employee.updated_at = datetime.utcnow()
    
    await db.commit()
    await db.refresh(employee)
    return employee

async def suspend_employee(db: AsyncSession, employee: Employee) -> Employee:
    """Suspend employee account"""
    employee.status = EmployeeStatus.SUSPENDED
    employee.updated_at = datetime.utcnow()
    
    await db.commit()
    await db.refresh(employee)
    return employee

async def activate_employee(db: AsyncSession, employee: Employee) -> Employee:
    """Activate employee account"""
    employee.status = EmployeeStatus.ACTIVE
    employee.updated_at = datetime.utcnow()
    
    await db.commit()
    await db.refresh(employee)
    return employee

async def deactivate_employee(db: AsyncSession, employee: Employee) -> Employee:
    """Deactivate employee account"""
    employee.status = EmployeeStatus.INACTIVE
    employee.updated_at = datetime.utcnow()
    
    await db.commit()
    await db.refresh(employee)
    return employee

async def get_employees_with_filters(
    db: AsyncSession,
    skip: int = 0,
    limit: int = 100,
    role: Optional[EmployeeRole] = None,
    status: Optional[EmployeeStatus] = None,
    search: Optional[str] = None,
    sort_by: str = "created_at",
    sort_order: str = "desc"
) -> List[Employee]:
    """Get employees with filtering and pagination"""
    query = select(Employee)
    
    # Apply filters
    if role:
        query = query.filter(Employee.role == role)
    if status:
        query = query.filter(Employee.status == status)
    if search:
        search_filter = or_(
            Employee.email.ilike(f"%{search}%"),
            Employee.username.ilike(f"%{search}%"),
            Employee.first_name.ilike(f"%{search}%"),
            Employee.last_name.ilike(f"%{search}%")
        )
        query = query.filter(search_filter)
    
    # Apply sorting
    if hasattr(Employee, sort_by):
        sort_column = getattr(Employee, sort_by)
        if sort_order.lower() == "desc":
            query = query.order_by(desc(sort_column))
        else:
            query = query.order_by(asc(sort_column))
    
    # Apply pagination
    query = query.offset(skip).limit(limit)
    
    result = await db.execute(query)
    return result.scalars().all()

async def count_employees_with_filters(
    db: AsyncSession,
    role: Optional[EmployeeRole] = None,
    status: Optional[EmployeeStatus] = None,
    search: Optional[str] = None
) -> int:
    """Count employees with filters"""
    query = select(func.count(Employee.id))
    
    # Apply filters
    if role:
        query = query.filter(Employee.role == role)
    if status:
        query = query.filter(Employee.status == status)
    if search:
        search_filter = or_(
            Employee.email.ilike(f"%{search}%"),
            Employee.username.ilike(f"%{search}%"),
            Employee.first_name.ilike(f"%{search}%"),
            Employee.last_name.ilike(f"%{search}%")
        )
        query = query.filter(search_filter)
    
    result = await db.execute(query)
    return result.scalar() or 0

async def get_employee_stats(db: AsyncSession, employee_id: int) -> Dict[str, Any]:
    """Get employee statistics"""
    employee = await get_employee(db, employee_id)
    if not employee:
        return {}
    
    # Count documents
    document_count = len(employee.documents) if employee.documents else 0
    
    # Calculate account age
    account_age_days = (datetime.utcnow() - employee.created_at).days
    
    # Days since last login
    days_since_login = None
    if employee.last_login_at:
        days_since_login = (datetime.utcnow() - employee.last_login_at).days
    
    return {
        "employee_id": employee.id,
        "total_documents": document_count,
        "account_age_days": account_age_days,
        "days_since_last_login": days_since_login,
        "is_verified": employee.is_verified,
        "role": employee.role.value,
        "status": employee.status.value
    }

async def get_employees_by_role(
    db: AsyncSession,
    role: EmployeeRole,
    skip: int = 0,
    limit: int = 100
) -> List[Employee]:
    """Get employees by role"""
    result = await db.execute(
        select(Employee)
        .filter(Employee.role == role)
        .order_by(desc(Employee.created_at))
        .offset(skip)
        .limit(limit)
    )
    return result.scalars().all()

async def get_active_employees(
    db: AsyncSession,
    skip: int = 0,
    limit: int = 100
) -> List[Employee]:
    """Get active employees"""
    result = await db.execute(
        select(Employee)
        .filter(Employee.status == EmployeeStatus.ACTIVE)
        .order_by(desc(Employee.last_login_at))
        .offset(skip)
        .limit(limit)
    )
    return result.scalars().all()

async def get_recently_active_employees(
    db: AsyncSession,
    hours: int = 24,
    limit: int = 50
) -> List[Employee]:
    """Get employees active within specified hours"""
    since_time = datetime.utcnow() - timedelta(hours=hours)
    result = await db.execute(
        select(Employee)
        .filter(
            and_(
                Employee.last_login_at >= since_time,
                Employee.status == EmployeeStatus.ACTIVE
            )
        )
        .order_by(desc(Employee.last_login_at))
        .limit(limit)
    )
    return result.scalars().all()

async def delete_employee(db: AsyncSession, employee_id: int) -> bool:
    """Delete employee (soft delete by marking as inactive)"""
    employee = await get_employee(db, employee_id)
    if not employee:
        return False
    
    employee.status = EmployeeStatus.INACTIVE
    employee.updated_at = datetime.utcnow()
    
    await db.commit()
    return True

# =====================================================================================
# Analytics and Reporting
# =====================================================================================

async def get_employee_analytics(
    db: AsyncSession,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> Dict[str, Any]:
    """Get analytics data for employees"""
    if not start_date:
        start_date = datetime.utcnow() - timedelta(days=30)
    if not end_date:
        end_date = datetime.utcnow()
    
    # Total employees
    total_employees = await db.scalar(
        select(func.count(Employee.id))
        .filter(Employee.created_at.between(start_date, end_date))
    )
    
    # Active employees
    active_employees = await db.scalar(
        select(func.count(Employee.id))
        .filter(Employee.status == EmployeeStatus.ACTIVE)
    )
    
    # Employees by role
    roles_result = await db.execute(
        select(
            Employee.role,
            func.count(Employee.id).label('count')
        )
        .group_by(Employee.role)
        .order_by(desc(func.count(Employee.id)))
    )
    roles = [
        {"role": row.role.value, "count": row.count}
        for row in roles_result.fetchall()
    ]
    
    # Employees by status
    status_result = await db.execute(
        select(
            Employee.status,
            func.count(Employee.id).label('count')
        )
        .group_by(Employee.status)
        .order_by(desc(func.count(Employee.id)))
    )
    statuses = [
        {"status": row.status.value, "count": row.count}
        for row in status_result.fetchall()
    ]
    
    # Recently active (last 7 days)
    recently_active = await db.scalar(
        select(func.count(Employee.id))
        .filter(
            and_(
                Employee.last_login_at >= datetime.utcnow() - timedelta(days=7),
                Employee.status == EmployeeStatus.ACTIVE
            )
        )
    )
    
    # Verified employees
    verified_employees = await db.scalar(
        select(func.count(Employee.id))
        .filter(Employee.is_verified == True)
    )
    
    return {
        "total_employees": total_employees or 0,
        "active_employees": active_employees or 0,
        "verified_employees": verified_employees or 0,
        "recently_active_employees": recently_active or 0,
        "employees_by_role": roles,
        "employees_by_status": statuses,
        "period": {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat()
        }
    }

# =====================================================================================
# Employee Chat Thread CRUD Operations
# =====================================================================================

async def create_employee_session(
    db: AsyncSession, employee_id: int, **kwargs
) -> EmployeeSession:
    """Create new employee session"""
    session_data = {
        "employee_id": employee_id,
        "session_id": generate_session_id(),
        "expires_at": datetime.utcnow() + timedelta(hours=settings.SESSION_EXPIRE_HOURS),
        **kwargs
    }

    db_obj = EmployeeSession(**session_data)
    db.add(db_obj)
    await db.commit()
    await db.refresh(db_obj)
    return db_obj



async def create_employee_refresh_token(
    db: AsyncSession,
    employee_id: int,  # Changed from uuid.UUID to int to match the model
    token_hash: str,
    expires_at: datetime,
    device_info: Optional[str] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None
) -> EmployeeRefreshToken:
    """Create refresh token"""
    db_obj = EmployeeRefreshToken(
        employee_id=employee_id,
        token_hash=token_hash,
        expires_at=expires_at,
        device_info=device_info,
        ip_address=ip_address,
        user_agent=user_agent
    )
    
    db.add(db_obj)
    await db.commit()
    await db.refresh(db_obj)
    return db_obj


async def create_employee_chat_thread(
    db: AsyncSession, thread_in
) -> EmployeeChatThread:
    """Create employee chat thread"""
    
    db_obj = EmployeeChatThread(**thread_in.model_dump())
    db.add(db_obj)
    await db.commit()
    await db.refresh(db_obj)
    return db_obj

async def get_employee_chat_thread(
    db: AsyncSession, thread_id: uuid.UUID
) -> Optional[EmployeeChatThread]:
    """Get employee chat thread by ID"""
    
    result = await db.execute(
        select(EmployeeChatThread).where(EmployeeChatThread.id == thread_id)
    )
    return result.scalar_one_or_none()

async def get_employee_chat_threads(
    db: AsyncSession,
    employee_id: int,
    skip: int = 0,
    limit: int = 100,
    include_archived: bool = False
) -> List[EmployeeChatThread]:
    """Get employee chat threads"""
    
    query = select(EmployeeChatThread).where(
        and_(
            EmployeeChatThread.employee_id == employee_id,
            EmployeeChatThread.is_deleted == False
        )
    )
    
    if not include_archived:
        query = query.where(EmployeeChatThread.is_archived == False)
    
    query = query.order_by(desc(EmployeeChatThread.last_message_at)).offset(skip).limit(limit)
    
    result = await db.execute(query)
    return result.scalars().all()

async def update_employee_chat_thread(
    db: AsyncSession, thread, update_data
) -> EmployeeChatThread:
    """Update employee chat thread"""
    update_dict = update_data.model_dump(exclude_unset=True)
    
    for field, value in update_dict.items():
        setattr(thread, field, value)
    
    await db.commit()
    await db.refresh(thread)
    return thread

async def archive_employee_chat_thread(
    db: AsyncSession, thread_id: uuid.UUID
) -> Optional[EmployeeChatThread]:
    """Archive a chat thread"""
    thread = await get_employee_chat_thread(db, thread_id)
    if thread:
        thread.is_archived = True
        await db.commit()
        await db.refresh(thread)
    return thread

async def delete_employee_chat_thread(
    db: AsyncSession, thread_id: uuid.UUID
) -> Optional[EmployeeChatThread]:
    """Soft delete a chat thread"""
    thread = await get_employee_chat_thread(db, thread_id)
    if thread:
        thread.is_deleted = True
        await db.commit()
        await db.refresh(thread)
    return thread

# =====================================================================================
# Employee Chat Message CRUD Operations
# =====================================================================================

async def create_employee_chat_message(
    db: AsyncSession, message_in
) -> EmployeeChatMessage:
    """Create employee chat message"""
    
    db_obj = EmployeeChatMessage(**message_in.model_dump())
    db.add(db_obj)
    
    # Update thread message count and last message time
    thread = await get_employee_chat_thread(db, message_in.thread_id)
    if thread:
        thread.message_count += 1
        thread.last_message_at = datetime.utcnow()
    
    await db.commit()
    await db.refresh(db_obj)
    return db_obj

async def get_employee_chat_message(
    db: AsyncSession, message_id: uuid.UUID
) -> Optional[EmployeeChatMessage]:
    """Get employee chat message by ID"""
    
    result = await db.execute(
        select(EmployeeChatMessage).where(EmployeeChatMessage.id == message_id)
    )
    return result.scalar_one_or_none()

async def get_employee_chat_thread_messages(
    db: AsyncSession,
    thread_id: uuid.UUID,
    skip: int = 0,
    limit: int = 100
) -> List[EmployeeChatMessage]:
    """Get messages for a chat thread"""
    
    result = await db.execute(
        select(EmployeeChatMessage).where(
            and_(
                EmployeeChatMessage.thread_id == thread_id,
                EmployeeChatMessage.is_deleted == False
            )
        ).order_by(asc(EmployeeChatMessage.sequence_number)).offset(skip).limit(limit)
    )
    return result.scalars().all()

async def get_next_employee_message_sequence_number(
    db: AsyncSession, thread_id: uuid.UUID
) -> int:
    """Get the next sequence number for a thread"""
    
    result = await db.scalar(
        select(func.max(EmployeeChatMessage.sequence_number)).where(
            EmployeeChatMessage.thread_id == thread_id
        )
    )
    return (result or 0) + 1

async def update_employee_chat_message(
    db: AsyncSession, message, update_data
) -> EmployeeChatMessage:
    """Update employee chat message"""
    update_dict = update_data.model_dump(exclude_unset=True)
    
    for field, value in update_dict.items():
        setattr(message, field, value)
    
    await db.commit()
    await db.refresh(message)
    return message

async def find_or_create_employee_chat_thread_by_session(
    db: AsyncSession,
    employee_id: int,
    session_id: str
) -> EmployeeChatThread:
    """Find or create employee chat thread by session_id stored in conversation_metadata"""
    
    # Try to find existing thread by session_id in metadata
    result = await db.execute(
        select(EmployeeChatThread).where(
            and_(
                EmployeeChatThread.employee_id == employee_id,
                EmployeeChatThread.conversation_metadata.op('->>')('session_id') == session_id,
                EmployeeChatThread.is_deleted == False
            )
        )
    )
    
    thread = result.scalar_one_or_none()
    
    if not thread:
        # Create new thread
        thread_data = EmployeeChatThreadCreate(
            employee_id=employee_id,
            title=f"Session {session_id[:8]}",
            conversation_metadata={
                "session_id": session_id,
                "created_via": "workflow_api",
                "agent_memory": {},
                "context": {}
            }
        )
        
        thread = await create_employee_chat_thread(db, thread_data)
    
    return thread

# =====================================================================================
# CRUD Classes for structured access
# =====================================================================================

class EmployeeCrud:
    """CRUD operations for Employees"""
    
    async def create(self, db: AsyncSession, *, obj_in: EmployeeCreate) -> Employee:
        """Create a new employee"""
        return await create_employee(db, employee_in=obj_in)
    
    async def get(self, db: AsyncSession, employee_id: int) -> Optional[Employee]:
        """Get employee by ID"""
        return await get_employee(db, employee_id)
    
    async def get_by_email(self, db: AsyncSession, email: str) -> Optional[Employee]:
        """Get employee by email"""
        return await get_employee_by_email(db, email)
    
    async def get_by_username(self, db: AsyncSession, username: str) -> Optional[Employee]:
        """Get employee by username"""
        return await get_employee_by_username(db, username)
    
    async def authenticate(self, db: AsyncSession, email: str, password: str) -> Optional[Employee]:
        """Authenticate employee"""
        return await authenticate_employee(db, email, password)
    
    async def update(self, db: AsyncSession, employee: Employee, obj_in: EmployeeUpdate) -> Employee:
        """Update employee"""
        return await update_employee(db, employee, obj_in)
    
    async def admin_update(self, db: AsyncSession, employee: Employee, obj_in: EmployeeAdminUpdate) -> Employee:
        """Admin update employee"""
        return await admin_update_employee(db, employee, obj_in)
    
    async def change_password(
        self, 
        db: AsyncSession, 
        employee: Employee, 
        password_update: EmployeePasswordUpdate
    ) -> Optional[Employee]:
        """Change employee password"""
        return await change_employee_password(db, employee, password_update)
    
    async def update_last_login(self, db: AsyncSession, employee: Employee) -> Employee:
        """Update last login timestamp"""
        return await update_employee_last_login(db, employee)
    
    async def verify_account(self, db: AsyncSession, employee: Employee) -> Employee:
        """Verify employee account"""
        return await verify_employee_account(db, employee)
    
    async def suspend(self, db: AsyncSession, employee: Employee) -> Employee:
        """Suspend employee account"""
        return await suspend_employee(db, employee)
    
    async def activate(self, db: AsyncSession, employee: Employee) -> Employee:
        """Activate employee account"""
        return await activate_employee(db, employee)
    
    async def deactivate(self, db: AsyncSession, employee: Employee) -> Employee:
        """Deactivate employee account"""
        return await deactivate_employee(db, employee)
    
    async def get_multi(
        self,
        db: AsyncSession,
        skip: int = 0,
        limit: int = 100,
        role: Optional[EmployeeRole] = None,
        status: Optional[EmployeeStatus] = None,
        search: Optional[str] = None,
        sort_by: str = "created_at",
        sort_order: str = "desc"
    ) -> List[Employee]:
        """Get multiple employees with filters"""
        return await get_employees_with_filters(
            db, skip, limit, role, status, search, sort_by, sort_order
        )
    
    async def count(
        self,
        db: AsyncSession,
        role: Optional[EmployeeRole] = None,
        status: Optional[EmployeeStatus] = None,
        search: Optional[str] = None
    ) -> int:
        """Count employees with filters"""
        return await count_employees_with_filters(db, role, status, search)
    
    async def get_by_role(
        self, 
        db: AsyncSession, 
        role: EmployeeRole, 
        skip: int = 0, 
        limit: int = 100
    ) -> List[Employee]:
        """Get employees by role"""
        return await get_employees_by_role(db, role, skip, limit)
    
    async def get_active(self, db: AsyncSession, skip: int = 0, limit: int = 100) -> List[Employee]:
        """Get active employees"""
        return await get_active_employees(db, skip, limit)
    
    async def get_recently_active(self, db: AsyncSession, hours: int = 24, limit: int = 50) -> List[Employee]:
        """Get recently active employees"""
        return await get_recently_active_employees(db, hours, limit)
    
    async def get_stats(self, db: AsyncSession, employee_id: int) -> Dict[str, Any]:
        """Get employee statistics"""
        return await get_employee_stats(db, employee_id)
    
    async def get_analytics(
        self,
        db: AsyncSession,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get employee analytics"""
        return await get_employee_analytics(db, start_date, end_date)
    
    async def delete(self, db: AsyncSession, employee_id: int) -> bool:
        """Delete employee (soft delete)"""
        return await delete_employee(db, employee_id)


# =====================================================================================
# CRUD object instances for easy import
# =====================================================================================

# Create instance for easy import
employee_crud = EmployeeCrud()
