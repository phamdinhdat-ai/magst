from fastapi import Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
import uuid
from typing import Optional, Any

from app.db.models.customer import Customer
from app.db.models.employee import Employee
from app.db.models.guest import Guest
from app.api.schemas.user import UserType
from app.db.models.document import Document, OwnerType
from app.db.session import get_db_session
from app.api.deps import (
    get_current_customer, get_current_guest,
    get_current_admin_customer
)
from app.api.deps_employee import get_current_employee, get_current_active_employee

async def get_document_by_id(
    document_id: int,
    db: AsyncSession = Depends(get_db_session)
) -> Document:
    """
    Get document by ID - helper function
    """
    from app.crud.crud_document import get_document
    
    doc = await get_document(db, document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    return doc

async def verify_document_access_customer(
    document_id: int,
    current_customer: Customer = Depends(get_current_customer),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Check if current customer has access to the document
    """
    # Get document first
    doc = await get_document_by_id(document_id, db)
    
    # Admin customers can access any document
    if current_customer.is_admin:
        return None
        
    # Check if document belongs to current customer
    if doc.owner_type == OwnerType.CUSTOMER and doc.owner_id == current_customer.id:
        return None
        
    # Check if document is public
    if doc.is_public:
        return None
        
    # No access
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="You don't have permission to access this document"
    )

async def verify_document_access_employee(
    document_id: int,
    current_employee: Employee = Depends(get_current_employee),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Check if current employee has access to the document
    """
    # Get document first
    doc = await get_document_by_id(document_id, db)
    
    # Admin employees can access any document
    if current_employee.is_admin:
        return None
        
    # Check if document belongs to current employee
    if doc.owner_type == OwnerType.EMPLOYEE and doc.owner_id == current_employee.id:
        return None
    
    # Employees can access all customer documents
    if doc.owner_type == OwnerType.CUSTOMER:
        return None
        
    # Check if document is public
    if doc.is_public:
        return None
        
    # No access
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="You don't have permission to access this document"
    )

async def verify_document_access_guest(
    document_id: int,
    current_guest: Guest = Depends(get_current_guest),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Check if current guest has access to the document
    """
    # Get document first
    doc = await get_document_by_id(document_id, db)
    
    # Check if document belongs to current guest
    if doc.owner_type == OwnerType.GUEST and doc.owner_id == current_guest.id:
        return None
        
    # Check if document is public
    if doc.is_public:
        return None
        
    # No access
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="You don't have permission to access this document"
    )

async def verify_document_access(
    document_id: int, 
    current_user: Optional[Any] = None,
    db: AsyncSession = Depends(get_db_session)
):
    """Generic document access verification based on user type, returns True if access is allowed or raises HTTPException"""
    """
    Generic document access verification based on user type
    """
    # Get document first
    doc = await get_document_by_id(document_id, db)
    
    if not current_user:
        # If no user, only public documents are accessible
        if doc.is_public:
            return None
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required to access non-public documents"
        )
    
    # Check based on user type
    if isinstance(current_user, Customer):
        await verify_document_access_customer(document_id, current_user, db)
        return None
    elif isinstance(current_user, Employee):
        await verify_document_access_employee(document_id, current_user, db)
        return None
    elif isinstance(current_user, Guest):
        await verify_document_access_guest(document_id, current_user, db)
        return None
    
    # Fallback - no access
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="You don't have permission to access this document"
    )

# Rate limiting function for document uploads
async def rate_limit_document_uploads() -> None:
    """
    Rate limit document uploads
    Can be enhanced with real rate limiting logic
    """
    # Implement rate limiting logic here
    # For now, just return None to allow uploads
    return None
