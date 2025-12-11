from typing import Optional, List
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from jose import JWTError
import uuid
import time
from datetime import datetime, timezone

from app.db.session import get_db_session
from app.core.security import verify_token, TokenData, CustomerPermissions
from app.crud.crud_customer import CustomerCrud, get_customer, update_customer_last_login
from app.db.models.customer import Customer, CustomerRole, CustomerStatus
from app.crud.crud_document import get_document
from app.crud.crud_guest import get_guest_session, get_or_create_guest, get_guest_by_session_id
from app.db.models.guest import GuestSession, GuestSessionStatus, Guest
from app.crud.crud_employee import get_employee
from app.db.models.employee import Employee, EmployeeRole, EmployeeStatus
from loguru import logger
from sqlalchemy import update
from app.agents.workflow.customer_workflow import CustomerWorkflow
from app.agents.workflow.guest_workflow import GuestWorkflow
# Security scheme
security = HTTPBearer()

# Rate limiting tracker
document_upload_tracker = {}
workflow_request_tracker = {}
chat_request_tracker = {}

# Activity tracking function
def get_workflow_manager() -> CustomerWorkflow:
    """
    Get the current customer workflow manager instance.
    This is used to access the workflow methods and properties.
    """
    from app.main import workflow_manager
    # Ensure we always return a valid workflow instance even if the main one isn't initialized
    if workflow_manager.customer_workflow is None:
        logger.warning("Main workflow manager not initialized, creating a new instance")
        return CustomerWorkflow()
    return workflow_manager.customer_workflow

def get_guest_workflow_manager() -> GuestWorkflow:
    """
    Get the current guest workflow manager instance.
    This is used to access the guest workflow methods and properties.
    """
    from app.main import workflow_manager
    return workflow_manager.guest_workflow


async def update_customer_last_activity(
    db: AsyncSession, 
    customer_id: int
) -> None:
    """
    Update the customer's last activity timestamp
    """
    try:
        await db.execute(
            update(Customer)
            .where(Customer.id == customer_id)
            .values(last_activity_at=datetime.now(timezone.utc))
        )
        await db.commit()
        logger.debug(f"Updated last_activity_at for customer {customer_id}")
    except Exception as e:
        logger.error(f"Error updating last_activity_at for customer {customer_id}: {str(e)}")
        await db.rollback()

async def get_current_customer(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db_session)
) -> Customer:
    """
    Get current authenticated customer from JWT token
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Verify the token
        token = credentials.credentials
        print(f"Verifying token: {token[:10]}...")
        token_data = verify_token(token, token_type="access")
        if token_data is None:
            print("Token verification failed - invalid token data")
            raise credentials_exception
            
        customer_id = token_data.customer_id
        if customer_id is None:
            print("Token verification failed - missing customer_id")
            raise credentials_exception
            
    except JWTError as e:
        print(f"JWT Error: {str(e)}")
        raise credentials_exception
    
    # Get customer from database
    try:
        # Convert customer_id to integer if it's a string
        customer_id_int = int(customer_id)
        print(f"Looking up customer with ID: {customer_id_int} (converted from: {customer_id})")
        customer = await get_customer(db, customer_id_int)
        
        if customer is None:
            print(f"No customer found with ID: {customer_id_int}")
            raise credentials_exception

        logger.info(f"Customer found: ID={customer.id}, email={customer.email}")
    except ValueError as e:
        # If conversion fails, raise authentication error
        logger.error(f"ValueError converting customer_id: {customer_id}, error: {str(e)}")
        raise credentials_exception
    
    # Check if customer is active
    if not customer.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Customer account is not active"
        )
    
    # Check if account is locked
    if customer.is_locked:
        raise HTTPException(
            status_code=status.HTTP_423_LOCKED,
            detail="Customer account is temporarily locked"
        )
    
    return customer

async def get_current_active_customer(
    current_customer: Customer = Depends(get_current_customer)
) -> Customer:
    """
    Get current active customer (alias for backward compatibility)
    """
    return current_customer

async def track_customer_activity(
    current_customer: Customer = Depends(get_current_customer),
    db: AsyncSession = Depends(get_db_session)
) -> Customer:
    """
    Track customer activity by updating last_activity_at timestamp.
    Use this dependency in endpoints where you want to track customer activity.
    """
    # Update last_activity timestamp
    await update_customer_last_activity(db, current_customer.id)
    logger.info(f"Activity tracked for customer ID={current_customer.id}")
    return current_customer

async def get_current_verified_customer(
    current_customer: Customer = Depends(get_current_customer)
) -> Customer:
    """
    Get current verified customer
    """
    if not current_customer.is_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Customer account is not verified"
        )
    return current_customer

async def get_current_admin_customer(
    current_customer: Customer = Depends(get_current_customer)
) -> Customer:
    """
    Get current customer with admin role
    """
    if current_customer.role != CustomerRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    return current_customer

async def get_current_premium_customer(
    current_customer: Customer = Depends(get_current_customer)
) -> Customer:
    """
    Get current customer with premium or higher role
    """
    if current_customer.role not in [
        CustomerRole.PREMIUM_CUSTOMER, 
        CustomerRole.VIP_CUSTOMER, 
        CustomerRole.ADMIN
    ]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Premium subscription required"
        )
    return current_customer

def require_permissions(required_permissions: List[str]):
    """
    Dependency factory to require specific permissions
    """
    async def check_permissions(
        credentials: HTTPAuthorizationCredentials = Depends(security),
        db: AsyncSession = Depends(get_db_session)
    ) -> Customer:
        # Get current customer
        customer = await get_current_customer(credentials, db)
        
        # Get customer permissions based on role
        from app.core.security import get_customer_permissions
        customer_permissions = get_customer_permissions(customer.role.value)
        
        # Check if customer has all required permissions
        missing_permissions = set(required_permissions) - set(customer_permissions)
        if missing_permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing required permissions: {', '.join(missing_permissions)}"
            )
        
        return customer
    
    return check_permissions

# Common permission checks
require_admin = require_permissions([CustomerPermissions.MANAGE_SYSTEM])
require_analytics_access = require_permissions([CustomerPermissions.READ_ANALYTICS])
require_customer_management = require_permissions([CustomerPermissions.READ_ALL_CUSTOMERS])

async def get_optional_customer(
    request: Request,
    db: AsyncSession = Depends(get_db_session)
) -> Optional[Customer]:
    """
    Get current customer if authenticated, otherwise return None.
    Useful for endpoints that work for both authenticated and guest users.
    """
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return None
    
    try:
        token = auth_header.split(" ")[1]
        token_data = verify_token(token, token_type="access")
        if token_data is None:
            return None
            
        customer_id = token_data.customer_id
        if customer_id is None:
            return None
        
        try:
            # Convert customer_id to integer if it's a string
            customer_id_int = int(customer_id)
            customer = await get_customer(db, customer_id_int)
            if customer and customer.is_active and not customer.is_locked:
                return customer
        except ValueError:
            # If conversion fails, return None
            pass
            
    except (JWTError, ValueError):
        pass
    
    return None

async def verify_customer_access(
    target_customer_id: int,
    current_customer: Customer = Depends(get_current_customer)
) -> None:
    """
    Verify that current customer can access target customer's data.
    Only allows access to own data unless admin.
    """
    if current_customer.role == CustomerRole.ADMIN:
        return None
    
    if current_customer.id != target_customer_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this customer's data"
        )
    
    return True

async def verify_session_access(
    session_id: uuid.UUID,
    current_customer: Customer = Depends(get_current_customer),
    db: AsyncSession = Depends(get_db_session)
) -> bool:
    """
    Verify that current customer can access the specified session.
    """
    from app.crud.crud_customer import customer_session_crud
    
    session = await customer_session_crud.get(db, id=session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    if current_customer.role != CustomerRole.ADMIN and session.customer_id != current_customer.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this session"
        )
    
    return True

async def verify_interaction_access(
    interaction_id: uuid.UUID,
    current_customer: Customer = Depends(get_current_customer),
    db: AsyncSession = Depends(get_db_session)
) -> bool:
    """
    Verify that current customer can access the specified interaction.
    """
    from app.crud.crud_customer import customer_interaction_crud
    
    interaction = await customer_interaction_crud.get(db, id=interaction_id)
    if not interaction:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Interaction not found"
        )
    
    if current_customer.role != CustomerRole.ADMIN and interaction.customer_id != current_customer.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this interaction"
        )
    
    return True

async def verify_chat_thread_access(
    thread_id: uuid.UUID,
    current_customer: Customer = Depends(get_current_customer),
    db: AsyncSession = Depends(get_db_session)
) -> bool:
    """
    Verify that current customer can access the specified chat thread.
    """
    from app.crud.crud_customer import CustomerChatThreadCrud

    thread = await CustomerChatThreadCrud.get(db, id=thread_id)
    if not thread:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat thread not found"
        )
    
    if current_customer.role != CustomerRole.ADMIN and thread.customer_id != current_customer.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this chat thread"
        )
    
    return True

# Rate limiting dependencies
class RateLimitError(HTTPException):
    def __init__(self, detail: str = "Rate limit exceeded"):
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=detail
        )

async def rate_limit_workflow_requests(
    current_customer: Customer = Depends(get_current_customer)
) -> None:
    """
    Rate limit workflow requests based on customer role
    """
    customer_id = str(current_customer.id)
    current_time = time.time()
    
    # Set limits based on customer role
    if current_customer.role == CustomerRole.ADMIN:
        max_requests = 200
        time_window = 3600  # 1 hour
    elif current_customer.role == CustomerRole.PREMIUM_CUSTOMER:
        max_requests = 100
        time_window = 3600  # 1 hour
    else:  # Basic customer
        max_requests = 20
        time_window = 3600  # 1 hour
    
    # Clean up old entries
    if customer_id in workflow_request_tracker:
        workflow_request_tracker[customer_id] = [
            timestamp for timestamp in workflow_request_tracker[customer_id]
            if current_time - timestamp < time_window
        ]
    else:
        workflow_request_tracker[customer_id] = []
    
    # Check if limit exceeded
    if len(workflow_request_tracker[customer_id]) >= max_requests:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Workflow request limit of {max_requests} requests per {time_window//60} minutes reached"
        )
    
    # Add new request timestamp
    workflow_request_tracker[customer_id].append(current_time)
    
    return None

async def rate_limit_chat_requests(
    current_customer: Customer = Depends(get_current_customer)
) -> None:
    """
    Rate limit chat requests based on customer role
    """
    customer_id = str(current_customer.id)
    current_time = time.time()
    
    # Set limits based on customer role
    if current_customer.role == CustomerRole.ADMIN:
        max_requests = 500
        time_window = 3600  # 1 hour
    elif current_customer.role == CustomerRole.PREMIUM_CUSTOMER:
        max_requests = 200
        time_window = 3600  # 1 hour
    else:  # Basic customer
        max_requests = 50
        time_window = 3600  # 1 hour
    
    # Clean up old entries
    if customer_id in chat_request_tracker:
        chat_request_tracker[customer_id] = [
            timestamp for timestamp in chat_request_tracker[customer_id]
            if current_time - timestamp < time_window
        ]
    else:
        chat_request_tracker[customer_id] = []
    
    # Check if limit exceeded
    if len(chat_request_tracker[customer_id]) >= max_requests:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Chat request limit of {max_requests} requests per {time_window//60} minutes reached"
        )
    
    # Add new request timestamp
    chat_request_tracker[customer_id].append(current_time)
    
    return True

# Document access verification
async def verify_document_access(
    document_id: int,
    current_customer: Customer = Depends(get_current_customer),
    db: AsyncSession = Depends(get_db_session)
) -> None:
    """Verify customer has access to document"""
    document = await get_document(db, document_id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    # Allow access if customer owns the document or if document is public
    if document.customer_id == current_customer.id or document.is_public:
        return None
    
    # Allow admin access to all documents
    if current_customer.role == CustomerRole.ADMIN:
        return None
        
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Access denied to this document"
    )

# Rate limiting for document uploads
async def rate_limit_document_uploads(
    request: Request,
    current_customer: Customer = Depends(get_current_customer)
) -> None:
    """Rate limit document uploads based on customer role"""
    customer_id = current_customer.id
    current_time = time.time()
    
    # Define limits based on role
    if current_customer.role == CustomerRole.ADMIN:
        max_uploads = 100
        time_window = 3600  # 1 hour
    elif current_customer.role == CustomerRole.VIP_CUSTOMER:
        max_uploads = 50
        time_window = 3600  # 1 hour
    elif current_customer.role == CustomerRole.PREMIUM_CUSTOMER:
        max_uploads = 20
        time_window = 3600  # 1 hour
    else:  # Basic customer
        max_uploads = 10
        time_window = 3600  # 1 hour
    
    # Clean up old entries
    if customer_id in document_upload_tracker:
        document_upload_tracker[customer_id] = [
            timestamp for timestamp in document_upload_tracker[customer_id]
            if current_time - timestamp < time_window
        ]
    else:
        document_upload_tracker[customer_id] = []
    
    # Check if limit exceeded
    if len(document_upload_tracker[customer_id]) >= max_uploads:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Upload limit of {max_uploads} documents per {time_window//60} minutes reached"
        )
    
    # Add new upload timestamp
    document_upload_tracker[customer_id].append(current_time)
    
    return True

# Security headers dependency
async def add_security_headers(request: Request):
    """
    Add security headers to response.
    This would typically be done at the middleware level.
    """
    return {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Content-Security-Policy": "default-src 'self'"
    }

# Guest session dependencies
async def get_current_guest_session(
    session_id: str,
    db: AsyncSession = Depends(get_db_session)
) -> GuestSession:
    """
    Get current guest session by session ID
    """
    guest_session = await get_guest_session(db, session_id)
    if not guest_session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Guest session not found"
        )
    
    # Check if session is active
    if guest_session.status != GuestSessionStatus.ACTIVE:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Guest session is not active"
        )
    
    # Check if session has expired
    if guest_session.expires_at and guest_session.expires_at < datetime.utcnow():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Guest session has expired"
        )
    
    return guest_session

async def get_current_guest(
    session_id: str,
    db: AsyncSession = Depends(get_db_session)
) -> Guest:
    """
    Get current guest by session ID, creating if needed
    """
    guest = await get_or_create_guest(db, session_id=session_id)
    return guest

async def get_current_active_guest(
    session_id: str,
    db: AsyncSession = Depends(get_db_session)
) -> Guest:
    """
    Get current active guest (alias for get_current_guest)
    """
    return await get_current_guest(session_id, db)

async def get_current_employee(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db_session)
) -> Employee:
    """
    Get the current employee from the access token.
    
    This dependency validates the employee's JWT token and returns the employee object.
    If token is invalid or employee not found, it raises an HTTP exception.
    """
    try:
        # Verify the token and extract payload
        token_data = verify_token(credentials.credentials)
        if token_data.user_type != "employee":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid user type in token"
            )
            
        # Get employee by ID
        employee = await get_employee(db, employee_id=token_data.user_id)
        if not employee:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Employee not found"
            )
            
        # Check if employee is active
        if employee.status != EmployeeStatus.ACTIVE:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Employee account is not active"
            )
            
        return employee
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )
