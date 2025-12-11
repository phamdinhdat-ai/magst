"""
Customer Queue Endpoints
This module provides endpoints for managing customer workflow request queues.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, List, Any, Optional
import uuid
from loguru import logger

from app.api.deps import get_current_customer, rate_limit_workflow_requests
from app.db.models.customer import Customer
from app.db.session import get_db_session
from app.api.v1.customer_request_queue import customer_request_queue

router = APIRouter(prefix="/api/v1/customer/queue", tags=["customer-queue"])

@router.get("/status")
async def get_queue_status(
    current_customer: Customer = Depends(get_current_customer),
):
    """Get current customer request queue status and statistics"""
    if current_customer.role.value not in ["CUSTOMER", "PREMIUM_CUSTOMER"]:
        raise HTTPException(status_code=403, detail="Access denied: Insufficient permissions")
        
    queue_stats = customer_request_queue.get_queue_stats()
    
    # For security reasons, hide some internal details
    public_stats = {
        "current_queue_size": queue_stats["current_queue_size"],
        "active_requests": queue_stats["active_requests"],
        "avg_processing_time_sec": queue_stats["avg_processing_time_sec"],
        "timestamp": queue_stats["timestamp"],
    }
    
    return public_stats

@router.get("/request/{request_id}")
async def get_request_status(
    request_id: str,
    current_customer: Customer = Depends(get_current_customer),
):
    """Get status of a specific customer request"""
    request_status = customer_request_queue.get_request_status(request_id)
    
    if "error" in request_status:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=request_status["error"])
    
    # Verify that the customer owns this request
    if str(current_customer.id) != request_status.get("customer_id"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied: Not your request")
    
    return request_status

@router.post("/request/{request_id}/cancel")
async def cancel_request(
    request_id: str,
    current_customer: Customer = Depends(get_current_customer),
):
    """Cancel a queued request"""
    result = await customer_request_queue.cancel_request(request_id, str(current_customer.id))
    
    if result:
        return {"status": "success", "message": "Request cancelled successfully"}
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unable to cancel request. It may already be processing, completed, or does not exist."
        )
