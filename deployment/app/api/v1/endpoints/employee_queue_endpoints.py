from fastapi import APIRouter, Depends, HTTPException, status
from app.api.deps_employee import get_current_employee
from app.api.queue_manager import employee_request_queue
from app.db.models.employee import Employee
from typing import Dict, Any
import time

router = APIRouter(prefix="/api/v1/employee/queue", tags=["employee_queue"])

@router.get("/status")
async def get_queue_status(
    current_employee: Employee = Depends(get_current_employee)
):
    """Get current queue statistics and status"""
    # Add admin/manager role check if needed for detailed stats
    is_admin = current_employee.role.value in ["admin", "manager"]
    
    stats = employee_request_queue.get_queue_stats()
    
    # Filter sensitive info for non-admin users
    if not is_admin:
        stats = {
            "current_queue_size": stats["current_queue_size"],
            "active_requests": stats["active_requests"],
            "estimated_wait_time": stats["current_queue_size"] * 5  # Rough estimate: 5 seconds per request
        }
    
    return {
        "queue_stats": stats,
        "timestamp": time.time(),
        "is_queue_full": employee_request_queue.queue.full()
    }

@router.get("/request/{request_id}")
async def get_request_status(
    request_id: str,
    current_employee: Employee = Depends(get_current_employee)
):
    """Get status of a specific request"""
    status = employee_request_queue.get_request_status(request_id)
    
    # Check if this is the employee's own request (if needed)
    # This would require storing employee_id in the request data
    
    return {
        "request_status": status,
        "timestamp": time.time()
    }
