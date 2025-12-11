# Imports for the feedback endpoint
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import uuid
from typing import Optional

from app.db.session import get_db_session
from app.db.models.employee import Employee, EmployeeInteraction
from app.db.models.feedback import EmployeeFeedback
from app.crud.crud_feedback import update_feedback
from app.api.deps_employee import get_current_active_employee
from app.agents.data_storages.response_storages import store_feedback

router = APIRouter(prefix="/api/v1/employee", tags=["employee"])

@router.post("/interactions/{interaction_id}/feedback")
async def submit_interaction_feedback(
    interaction_id: uuid.UUID,
    feedback: EmployeeFeedback,
    current_employee: Employee = Depends(get_current_active_employee),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Submit feedback for an employee interaction.
    
    This endpoint handles both database updates and additional CSV storage for feedback data.
    It supports session_id and message_id tracking for better analytics.
    """
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
    
    # Get session_id from the interaction if available
    session_id = str(interaction.session_id) if interaction.session_id else None
    
    # Get message_id from the feedback if available
    message_id = feedback.message_id
    
    # Update feedback using the unified feedback system
    try:
        # Update in the database
        await update_feedback(
            db=db,
            interaction_id=interaction_id,
            user_type='employee',
            rating=feedback.rating,
            feedback_text=feedback.feedback_text,
            was_helpful=feedback.was_helpful,
            feedback_type=feedback.feedback_type.value if feedback.feedback_type else None
        )
        
        # Also store in data storage system with session and message tracking
        await store_feedback(
            interaction_id=str(interaction_id),
            user_type='employee',
            feedback_type=feedback.feedback_type.value if feedback.feedback_type else "unknown",
            rating=feedback.rating,
            feedback_text=feedback.feedback_text,
            was_helpful=feedback.was_helpful,
            session_id=session_id,
            message_id=message_id
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit feedback: {str(e)}"
        )
    
    return {
        "message": "Feedback submitted successfully",
        "status": "success", 
        "interaction_id": str(interaction_id),
        "session_id": session_id,
        "message_id": message_id
    }
