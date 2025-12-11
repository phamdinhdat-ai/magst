"""
Document Queue Endpoints
This module provides endpoints for managing document processing request queues.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, List, Any, Optional, Union
import uuid
from loguru import logger

from app.api.deps import get_current_customer
from app.api.deps_employee import get_current_employee
from app.db.models.customer import Customer
from app.db.models.employee import Employee
from app.db.session import get_db_session
from app.api.v1.document_request_queue import document_request_queue, OwnerType

router = APIRouter(prefix="/api/v1/documents/queue", tags=["document-queue"])

@router.get("/status")
async def get_document_queue_status(
    current_user: Union[Customer, Employee] = Depends(get_current_customer),
):
    """Get current document request queue status and statistics"""
    queue_stats = document_request_queue.get_queue_stats()
    
    # For security reasons, hide some internal details
    public_stats = {
        "current_queue_size": queue_stats["current_queue_size"],
        "active_requests": queue_stats["active_requests"],
        "avg_processing_time_sec": queue_stats["avg_processing_time_sec"],
        "processed_bytes_total": queue_stats["processed_bytes_total"],
        "timestamp": queue_stats["timestamp"],
    }
    
    return public_stats

@router.get("/request/{request_id}")
async def get_document_request_status(
    request_id: str,
    current_user: Union[Customer, Employee] = Depends(get_current_customer),
):
    """Get status of a specific document processing request"""
    request_status = document_request_queue.get_request_status(request_id)
    
    if "error" in request_status:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=request_status["error"])
    
    # Determine the user type and ID
    user_type = None
    user_id = None
    
    if hasattr(current_user, 'role'):  # Customer has a role attribute
        user_type = OwnerType.CUSTOMER.value
        user_id = str(current_user.id)
    else:  # Employee
        user_type = OwnerType.EMPLOYEE.value
        user_id = str(current_user.id)
    
    # Verify that the user owns this request
    if user_id != request_status.get("owner_id") or user_type != request_status.get("owner_type"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied: Not your document request")
    
    return request_status

@router.post("/request/{request_id}/cancel")
async def cancel_document_request(
    request_id: str,
    current_user: Union[Customer, Employee] = Depends(get_current_customer),
):
    """Cancel a queued document processing request"""
    # Determine the user type and ID
    user_type = None
    user_id = None
    
    if hasattr(current_user, 'role'):  # Customer has a role attribute
        user_type = OwnerType.CUSTOMER.value
        user_id = str(current_user.id)
    else:  # Employee
        user_type = OwnerType.EMPLOYEE.value
        user_id = str(current_user.id)
        
    result = await document_request_queue.cancel_request(request_id, user_id, user_type)
    
    if result:
        return {"status": "success", "message": "Document processing request cancelled successfully"}
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unable to cancel request. It may already be processing, completed, or does not exist."
        )
