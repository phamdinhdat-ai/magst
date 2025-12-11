from typing import Optional
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from jose import JWTError
import uuid
from datetime import datetime
import time

from app.db.session import get_db_session
from app.core.security import verify_token, TokenData
from app.crud.crud_employee import get_employee
from app.db.models.employee import Employee, EmployeeRole, EmployeeStatus

# Security scheme
security = HTTPBearer()

# Rate limiting tracker for employees
employee_workflow_request_tracker = {}
employee_chat_request_tracker = {}

class EmployeeRateLimitError(HTTPException):
    def __init__(self, detail: str = "Rate limit exceeded"):
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=detail
        )

async def get_current_employee(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db_session)
) -> Employee:
    """
    Get current authenticated employee from JWT token
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Verify the token with detailed debugging
        token = credentials.credentials
        print(f"Employee: Verifying token: {token[:10]}...")
        token_data = verify_token(token, token_type="access")
        if token_data is None:
            print("Employee: Token verification failed - invalid token data")
            raise credentials_exception
            
        # Extract employee ID from token (it's stored as customer_id in the token structure)
        employee_id = token_data.customer_id
        if employee_id is None:
            print("Employee: Token verification failed - missing customer_id")
            raise credentials_exception
            
        print(f"Employee: Token verified successfully. employee_id in token: {employee_id} (type: {type(employee_id).__name__})")
            
    except JWTError as e:
        print(f"Employee: JWT Error: {str(e)}")
        raise credentials_exception
    
    # Get employee from database
    try:
        print(f"Employee: Looking up employee with ID: {employee_id}")
        
        # First, try to get the employee record
        employee = await get_employee(db, employee_id=employee_id)
        
        if employee is None:
            print(f"Employee: Not found for ID: {employee_id}")
            # Try debugging the database query
            from sqlalchemy import text
            query = text("SELECT * FROM employees WHERE id = :emp_id")
            result = await db.execute(query, {"emp_id": int(employee_id)})
            raw_data = result.fetchone()
            print(f"Employee: Raw database query result: {raw_data}")
            raise credentials_exception
        
        print(f"Employee: Found employee: ID={employee.id}, email={employee.email}, role={employee.role}")
    except Exception as e:
        print(f"Employee: Error getting employee with ID {employee_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}"
        )
    
    # Check if employee is active
    if employee.status != EmployeeStatus.ACTIVE:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Employee account is not active"
        )
    
    return employee

async def get_current_active_employee(
    current_employee: Employee = Depends(get_current_employee)
) -> Employee:
    """
    Get current active employee
    """
    return current_employee

async def get_current_verified_employee(
    current_employee: Employee = Depends(get_current_employee)
) -> Employee:
    """
    Get current verified employee
    """
    if not current_employee.is_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Employee account is not verified"
        )
    return current_employee

async def get_current_admin_employee(
    current_employee: Employee = Depends(get_current_employee)
) -> Employee:
    """
    Get current employee with admin role
    """
    if current_employee.role != EmployeeRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    return current_employee

async def get_current_manager_employee(
    current_employee: Employee = Depends(get_current_employee)
) -> Employee:
    """
    Get current employee with manager or admin role
    """
    if current_employee.role not in [EmployeeRole.MANAGER, EmployeeRole.ADMIN]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Manager or Admin permissions required"
        )
    return current_employee

def require_employee_role(*required_roles: EmployeeRole):
    """
    Dependency factory to require specific employee roles
    """
    def role_checker(current_employee: Employee = Depends(get_current_employee)):
        if current_employee.role not in required_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Required role: {' or '.join([role.value for role in required_roles])}"
            )
        return current_employee
    return role_checker

async def rate_limit_employee_workflow_requests(
    current_employee: Employee = Depends(get_current_employee)
) -> bool:
    """
    Rate limit workflow requests for employees based on role
    """
    employee_id = str(current_employee.id)
    current_time = time.time()
    
    # Set limits based on employee role
    if current_employee.role == EmployeeRole.ADMIN:
        max_requests = 500
        time_window = 3600  # 1 hour
    elif current_employee.role == EmployeeRole.MANAGER:
        max_requests = 200
        time_window = 3600  # 1 hour
    else:  # Basic employee
        max_requests = 100
        time_window = 3600  # 1 hour
    
    # Clean up old entries
    if employee_id in employee_workflow_request_tracker:
        employee_workflow_request_tracker[employee_id] = [
            timestamp for timestamp in employee_workflow_request_tracker[employee_id]
            if current_time - timestamp < time_window
        ]
    
    # Initialize if not exists
    if employee_id not in employee_workflow_request_tracker:
        employee_workflow_request_tracker[employee_id] = []
    
    # Check if limit exceeded
    if len(employee_workflow_request_tracker[employee_id]) >= max_requests:
        raise EmployeeRateLimitError(
            f"Workflow request rate limit exceeded. Maximum {max_requests} requests per hour for {current_employee.role.value} role."
        )
    
    # Add current request
    employee_workflow_request_tracker[employee_id].append(current_time)
    
    return True

async def rate_limit_employee_chat_requests(
    current_employee: Employee = Depends(get_current_employee)
) -> bool:
    """
    Rate limit chat requests for employees
    """
    employee_id = str(current_employee.id)
    current_time = time.time()
    
    # Set limits based on employee role
    if current_employee.role == EmployeeRole.ADMIN:
        max_requests = 1000
        time_window = 3600  # 1 hour
    elif current_employee.role == EmployeeRole.MANAGER:
        max_requests = 500
        time_window = 3600  # 1 hour
    else:  # Basic employee
        max_requests = 200
        time_window = 3600  # 1 hour
    
    # Clean up old entries
    if employee_id in employee_chat_request_tracker:
        employee_chat_request_tracker[employee_id] = [
            timestamp for timestamp in employee_chat_request_tracker[employee_id]
            if current_time - timestamp < time_window
        ]
    
    # Initialize if not exists
    if employee_id not in employee_chat_request_tracker:
        employee_chat_request_tracker[employee_id] = []
    
    # Check if limit exceeded
    if len(employee_chat_request_tracker[employee_id]) >= max_requests:
        raise EmployeeRateLimitError(
            f"Chat request rate limit exceeded. Maximum {max_requests} requests per hour for {current_employee.role.value} role."
        )
    
    # Add current request
    employee_chat_request_tracker[employee_id].append(current_time)
    
    return True
