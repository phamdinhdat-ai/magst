from datetime import datetime, timedelta
from typing import Any, Union, Optional, Dict
from jose import jwt, JWTError
from passlib.context import CryptContext
from pydantic import BaseModel
import secrets
import hashlib
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.core.config import settings

# HTTP Bearer security scheme
security = HTTPBearer(auto_error=False)

# Password context for hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str
    expires_in: int

class TokenData(BaseModel):
    customer_id: Optional[str] = None
    email: Optional[str] = None
    role: Optional[str] = None
    scopes: list[str] = []

def create_access_token(
    subject: Union[str, Any], 
    expires_delta: timedelta = None,
    scopes: list[str] = None
) -> str:
    """Create JWT access token"""
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )
    
    to_encode = {
        "exp": expire,
        "sub": str(subject),
        "type": "access",
        "scopes": scopes or []
    }
    
    encoded_jwt = jwt.encode(
        to_encode, 
        settings.SECRET_KEY, 
        algorithm=settings.ALGORITHM
    )
    return encoded_jwt

def create_refresh_token(
    subject: Union[str, Any],
    expires_delta: timedelta = None
) -> str:
    """Create JWT refresh token"""
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            days=settings.REFRESH_TOKEN_EXPIRE_DAYS
        )
    
    to_encode = {
        "exp": expire,
        "sub": str(subject),
        "type": "refresh",
        "jti": secrets.token_urlsafe(32)  # Unique token ID
    }
    
    encoded_jwt = jwt.encode(
        to_encode,
        settings.SECRET_KEY,
        algorithm=settings.ALGORITHM
    )
    return encoded_jwt

async def get_current_user_optional(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[Dict[str, Any]]:
    """
    Get the current user information from the authentication token (if available).
    This is an optional authentication function that doesn't raise exceptions
    if no token or invalid token is provided. Instead, it returns None.
    
    Args:
        request: FastAPI request object
        credentials: Optional HTTP Authorization credentials
        
    Returns:
        Dictionary with user information if authenticated, None otherwise
    """
    # No credentials provided - return None
    if not credentials or not credentials.credentials:
        return None
    
    try:
        # Verify the token
        token = credentials.credentials
        token_data = verify_token(token, token_type="access")
        
        # No valid token data - return None
        if token_data is None:
            return None
        
        # Determine the user type based on token information
        user_info = {
            "authenticated": True,
        }
        
        # Handle different types of users
        if hasattr(token_data, "customer_id") and token_data.customer_id:
            user_info["user_type"] = "customer"
            user_info["customer_id"] = int(token_data.customer_id)
        elif hasattr(token_data, "employee_id") and token_data.employee_id:
            user_info["user_type"] = "employee"
            user_info["employee_id"] = int(token_data.employee_id)
        elif hasattr(token_data, "guest_id") and token_data.guest_id:
            user_info["user_type"] = "guest"
            user_info["guest_id"] = int(token_data.guest_id)
        else:
            # If we can't determine the user type, default to None
            return None
            
        return user_info
    except JWTError:
        return None
    except Exception:
        return None

def verify_token(token: str, token_type: str = "access") -> Optional[TokenData]:
    """Verify and decode JWT token"""
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM]
        )
        
        # Check token type
        if payload.get("type") != token_type:
            print(f"Token type mismatch: expected {token_type}, got {payload.get('type')}")
            return None
        
        # For backward compatibility, check both customer_id and sub fields
        customer_id = payload.get("customer_id") or payload.get("sub")
        scopes: list[str] = payload.get("scopes", [])
        
        if customer_id is None:
            print("No customer_id or sub found in token")
            return None
        
        print(f"Token verification successful: ID={customer_id}, type={token_type}")
        token_data = TokenData(
            customer_id=customer_id,
            scopes=scopes
        )
        return token_data
        
    except JWTError:
        return None

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Generate password hash"""
    return pwd_context.hash(password)

def generate_password_reset_token(email: str) -> str:
    """Generate password reset token"""
    delta = timedelta(hours=settings.PASSWORD_RESET_TOKEN_EXPIRE_HOURS)
    expire = datetime.utcnow() + delta
    
    to_encode = {
        "exp": expire,
        "sub": email,
        "type": "password_reset"
    }
    
    return jwt.encode(
        to_encode,
        settings.SECRET_KEY,
        algorithm=settings.ALGORITHM
    )

def verify_password_reset_token(token: str) -> Optional[str]:
    """Verify password reset token and return email"""
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM]
        )
        
        if payload.get("type") != "password_reset":
            return None
            
        email: str = payload.get("sub")
        return email
        
    except JWTError:
        return None

def generate_email_verification_token(email: str) -> str:
    """Generate email verification token"""
    delta = timedelta(hours=settings.EMAIL_VERIFY_TOKEN_EXPIRE_HOURS)
    expire = datetime.utcnow() + delta
    
    to_encode = {
        "exp": expire,
        "sub": email,
        "type": "email_verification"
    }
    
    return jwt.encode(
        to_encode,
        settings.SECRET_KEY,
        algorithm=settings.ALGORITHM
    )

def verify_email_verification_token(token: str) -> Optional[str]:
    """Verify email verification token and return email"""
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM]
        )
        
        if payload.get("type") != "email_verification":
            return None
            
        email: str = payload.get("sub")
        return email
        
    except JWTError:
        return None

def hash_refresh_token(token: str) -> str:
    """Hash refresh token for storage"""
    return hashlib.sha256(token.encode()).hexdigest()

def generate_session_id() -> str:
    """Generate unique session ID"""
    return secrets.token_urlsafe(32)

# Permission/scope definitions
class CustomerPermissions:
    """Define customer permissions/scopes"""
    
    # Basic permissions
    READ_OWN_PROFILE = "customer:read_own_profile"
    UPDATE_OWN_PROFILE = "customer:update_own_profile"
    DELETE_OWN_ACCOUNT = "customer:delete_own_account"
    
    # Interaction permissions
    CREATE_INTERACTIONS = "customer:create_interactions"
    READ_OWN_INTERACTIONS = "customer:read_own_interactions"
    
    # Chat permissions
    CREATE_CHATS = "customer:create_chats"
    READ_OWN_CHATS = "customer:read_own_chats"
    UPDATE_OWN_CHATS = "customer:update_own_chats"
    DELETE_OWN_CHATS = "customer:delete_own_chats"
    
    # Premium permissions
    ADVANCED_SEARCH = "customer:advanced_search"
    PRIORITY_SUPPORT = "customer:priority_support"
    EXPORT_DATA = "customer:export_data"
    
    # Admin permissions
    READ_ALL_CUSTOMERS = "admin:read_all_customers"
    UPDATE_ALL_CUSTOMERS = "admin:update_all_customers"
    DELETE_CUSTOMERS = "admin:delete_customers"
    READ_ANALYTICS = "admin:read_analytics"
    MANAGE_SYSTEM = "admin:manage_system"

def get_customer_permissions(role: str) -> list[str]:
    """Get permissions for customer role"""
    permissions = [
        CustomerPermissions.READ_OWN_PROFILE,
        CustomerPermissions.UPDATE_OWN_PROFILE,
        CustomerPermissions.CREATE_INTERACTIONS,
        CustomerPermissions.READ_OWN_INTERACTIONS,
        CustomerPermissions.CREATE_CHATS,
        CustomerPermissions.READ_OWN_CHATS,
        CustomerPermissions.UPDATE_OWN_CHATS,
        CustomerPermissions.DELETE_OWN_CHATS,
    ]
    
    if role in ["premium_customer", "vip_customer"]:
        permissions.extend([
            CustomerPermissions.ADVANCED_SEARCH,
            CustomerPermissions.PRIORITY_SUPPORT,
            CustomerPermissions.EXPORT_DATA,
        ])
    
    if role == "vip_customer":
        permissions.extend([
            CustomerPermissions.DELETE_OWN_ACCOUNT,
        ])
    
    if role == "admin":
        permissions.extend([
            CustomerPermissions.READ_ALL_CUSTOMERS,
            CustomerPermissions.UPDATE_ALL_CUSTOMERS,
            CustomerPermissions.DELETE_CUSTOMERS,
            CustomerPermissions.READ_ANALYTICS,
            CustomerPermissions.MANAGE_SYSTEM,
        ])
    
    return permissions



class EmployeePermissions:
    """Define employee permissions/scopes"""
    
    # Basic permissions
    READ_OWN_PROFILE = "employee:read_own_profile"
    UPDATE_OWN_PROFILE = "employee:update_own_profile"
    
    # Interaction permissions
    MANAGE_CUSTOMER_INTERACTIONS = "employee:manage_customer_interactions"
    
    # Chat permissions
    MANAGE_CUSTOMER_CHATS = "employee:manage_customer_chats"  
    # Premium permissions
    ADVANCED_SEARCH = "employee:advanced_search"
    PRIORITY_SUPPORT = "employee:priority_support"
    EXPORT_DATA = "employee:export_data"
    
    # Admin permissions
    READ_ALL_EMPLOYEES = "admin:read_all_employees"
    UPDATE_ALL_EMPLOYEES = "admin:update_all_employees"
    DELETE_EMPLOYEES = "admin:delete_employees" 
    READ_ANALYTICS = "admin:read_analytics"
    MANAGE_SYSTEM = "admin:manage_system"
def get_employee_permissions(role: str) -> list[str]:
    """Get permissions for employee role"""
    permissions = [
        EmployeePermissions.READ_OWN_PROFILE,
        EmployeePermissions.UPDATE_OWN_PROFILE,
        EmployeePermissions.MANAGE_CUSTOMER_INTERACTIONS,
        EmployeePermissions.MANAGE_CUSTOMER_CHATS,
    ]
    
    if role in ["premium_employee", "vip_employee"]:
        permissions.extend([
            EmployeePermissions.ADVANCED_SEARCH,
            EmployeePermissions.PRIORITY_SUPPORT,
            EmployeePermissions.EXPORT_DATA,
        ])
    
    if role == "vip_employee":
        permissions.extend([
            EmployeePermissions.DELETE_EMPLOYEES,
        ])
    
    if role == "admin":
        permissions.extend([
            EmployeePermissions.READ_ALL_EMPLOYEES,
            EmployeePermissions.UPDATE_ALL_EMPLOYEES,
            EmployeePermissions.DELETE_EMPLOYEES,
            EmployeePermissions.READ_ANALYTICS,
            EmployeePermissions.MANAGE_SYSTEM,
        ])
    
    return permissions
