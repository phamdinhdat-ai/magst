
from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any, Optional, Union, List
import uuid
from datetime import datetime

from app.db.session import get_db_session
from app.db.models.feedback import BaseFeedback, FeedbackType
from app.crud.crud_feedback import update_feedback, create_feedback
from app.core.security import get_current_user_optional
from app.api.deps import get_current_customer, get_current_employee

router = APIRouter(prefix="/api/v1/feedback", tags=["feedback"])

@router.post("/", response_model=Dict[str, Any])
async def submit_generic_feedback(
    feedback_data: BaseFeedback,
    request: Request,
    user_type: str = "anonymous",  # Can be 'employee', 'customer', 'guest', or 'anonymous'
    db: AsyncSession = Depends(get_db_session),
    current_user: Optional[Union[Dict, None]] = Depends(get_current_user_optional)
):
    """
    Universal feedback endpoint that can be used for any user type.
    
    This endpoint now stores feedback associated with the specific user ID
    similar to how documents are stored. It tries to detect the user from 
    authentication token first, then falls back to anonymous if not authenticated.
    
    Args:
        feedback_data: The feedback data including interaction_id and feedback type
        request: FastAPI request object for extracting user information
        user_type: The type of user submitting feedback
        current_user: The current authenticated user (optional)
    """
    # Determine user ID and type from authentication if available
    user_id = None
    detected_user_type = user_type
    
    # If we have an authenticated user, use their information
    if current_user:
        print(f"Authenticated user detected: {current_user}")
        
        if "customer_id" in current_user and current_user["customer_id"]:
            user_id = current_user["customer_id"]
            detected_user_type = "customer"
        elif "employee_id" in current_user and current_user["employee_id"]:
            user_id = current_user["employee_id"]
            detected_user_type = "employee"
        elif "guest_id" in current_user and current_user["guest_id"]:
            user_id = current_user["guest_id"]
            detected_user_type = "guest"
    else:
        print("No authenticated user detected, using anonymous mode")
    
    # Log the incoming data for debugging
    print(f"Received feedback: {feedback_data.dict()}, user_type: {detected_user_type}, user_id: {user_id}")
    
    if detected_user_type not in ["employee", "customer", "guest", "anonymous"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid user type. Must be 'employee', 'customer', 'guest', or 'anonymous'."
        )
    
    # Always ensure CSV storage works first
    csv_success = False
    try:
        print(f"Anonymous feedback received for interaction {feedback_data.interaction_id}")
        print(f"Rating: {feedback_data.rating}, Was helpful: {feedback_data.was_helpful}, Feedback type: {feedback_data.feedback_type.value if feedback_data.feedback_type else None}")
        print(f"Feedback text: {feedback_data.feedback_text}")
        
        # First store in CSV feedback storage to ensure at least one storage method works
        try:
            from app.agents.data_storages.response_storages import store_feedback
            # Extract session_id from feedback data if available
            session_id = getattr(feedback_data, 'session_id', None)
            await store_feedback(
                interaction_id=str(feedback_data.interaction_id),
                user_id=str(user_id) if user_id else "anonymous",
                user_type=detected_user_type,
                feedback_type=feedback_data.feedback_type.value if feedback_data.feedback_type else "unknown",
                rating=feedback_data.rating,
                feedback_text=feedback_data.feedback_text,
                was_helpful=feedback_data.was_helpful,
                message_id=feedback_data.message_id,
                session_id=session_id  # Add session_id parameter
            )
            csv_success = True
            print("Successfully stored feedback in CSV")
        except Exception as e:
            # Log but continue with database storage attempt
            print(f"Warning: Could not store feedback in CSV: {str(e)}")
        
        # Now try to create a new feedback entry in the database
        try:
            # Extract session_id from feedback data if available
            session_id = getattr(feedback_data, 'session_id', None)
            
            # Check if feedback already exists to prevent duplicates
            from app.crud.crud_feedback import check_existing_feedback
            
            try:
                # Convert interaction_id to UUID if it's a string (same as in the Pydantic model)
                interaction_id = feedback_data.interaction_id
                if isinstance(interaction_id, str):
                    try:
                        import uuid
                        interaction_id = uuid.UUID(interaction_id)
                    except ValueError:
                        # If it's not a valid UUID format, create a new UUID based on the string
                        interaction_id = uuid.uuid5(uuid.NAMESPACE_URL, f"feedback:{interaction_id}")
                
                print(f"Checking for existing feedback with converted interaction_id: {interaction_id}")
                
                existing_feedback = await check_existing_feedback(
                    db=db,
                    interaction_id=interaction_id,
                    user_id=user_id,
                    user_type=detected_user_type,
                    session_id=session_id,
                    message_id=feedback_data.message_id
                )
                
                if existing_feedback:
                    print(f"Feedback already exists for interaction {interaction_id} from user {user_id or 'anonymous'}")
                    return {
                        "message": "Feedback has already been submitted for this interaction",
                        "status": "duplicate",
                        "feedback_id": None,
                        "interaction_id": str(interaction_id),
                        "user_id": user_id,
                        "user_type": detected_user_type,
                        "storage": {
                            "database": False,
                            "csv": csv_success
                        }
                    }
            except Exception as check_error:
                print(f"Error checking for existing feedback: {str(check_error)}")
                # Rollback any failed transaction before continuing
                try:
                    await db.rollback()
                except:
                    pass
                
            # Create new feedback entry with fresh transaction
            feedback_id = await create_feedback(
                db=db,
                interaction_id=interaction_id,  # Use the converted UUID
                user_id=user_id,
                user_type=detected_user_type,
                rating=feedback_data.rating,
                feedback_text=feedback_data.feedback_text,
                was_helpful=feedback_data.was_helpful,
                feedback_type=feedback_data.feedback_type.value if feedback_data.feedback_type else None,
                message_id=feedback_data.message_id,
                session_id=session_id,  # Add session_id parameter
                feedback_metadata={
                    "ip_address": request.client.host if request.client else None,
                    "user_agent": request.headers.get("user-agent", ""),
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            print(f"Successfully stored feedback in database with ID: {feedback_id}")
            db_success = True
                    
        except Exception as db_e:
            print(f"Database feedback storage failed: {str(db_e)}")
            feedback_id = uuid.uuid4()  # Generate a new ID since DB storage failed
            db_success = False
        
        # Determine status based on what succeeded
        if db_success and csv_success:
            status_msg = "success"
            message = "Feedback submitted successfully to both database and CSV"
        elif db_success:
            status_msg = "partial_success"
            message = "Feedback submitted to database, but CSV storage failed"
        elif csv_success:
            status_msg = "partial_success"
            message = "Feedback submitted to CSV, but database storage failed (table might not exist)"
        else:
            # Both failed but we're still returning a 200 response since this isn't critical
            status_msg = "error"
            message = "Failed to store feedback, but acknowledging receipt"
            
        return {
            "message": message,
            "status": status_msg,
            "feedback_id": str(feedback_id),
            "interaction_id": str(feedback_data.interaction_id),
            "user_id": user_id,
            "user_type": detected_user_type,
            "storage": {
                "database": db_success,
                "csv": csv_success
            }
        }
    
    except ValueError as e:
        # Handle validation errors
        print(f"Validation error in feedback submission: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        # Handle unexpected errors
        print(f"Unexpected error in feedback submission: {str(e)}")
        import traceback
        print(traceback.format_exc())
        
        # If we reach here, something went very wrong
        # Return a more user-friendly error response
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing your feedback. Please try again later."
        )


@router.get("/customer", response_model=List[Dict[str, Any]])
async def get_customer_feedback(
    db: AsyncSession = Depends(get_db_session),
    current_customer = Depends(get_current_customer),
    skip: int = 0,
    limit: int = 20
):
    """
    Get all feedback submitted by the current authenticated customer.
    Similar to how documents are retrieved, this endpoint allows a customer
    to see their feedback history.
    """
    try:
        from app.crud.crud_feedback import get_user_feedback
        feedback_list = await get_user_feedback(
            db=db,
            user_id=current_customer.id,
            user_type="customer",
            skip=skip,
            limit=limit
        )
        
        return feedback_list
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve customer feedback: {str(e)}"
        )


@router.get("/employee", response_model=List[Dict[str, Any]])
async def get_employee_feedback(
    db: AsyncSession = Depends(get_db_session),
    current_employee = Depends(get_current_employee),
    skip: int = 0,
    limit: int = 20
):
    """
    Get all feedback submitted by the current authenticated employee.
    Similar to how documents are retrieved, this endpoint allows an employee
    to see their feedback history.
    """
    try:
        from app.crud.crud_feedback import get_user_feedback
        feedback_list = await get_user_feedback(
            db=db,
            user_id=current_employee.id,
            user_type="employee",
            skip=skip,
            limit=limit
        )
        
        return feedback_list
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve employee feedback: {str(e)}"
        )


@router.get("/admin/all", response_model=List[Dict[str, Any]])
async def get_all_feedback(
    db: AsyncSession = Depends(get_db_session),
    # Add admin dependency here when available
    # current_admin = Depends(get_admin_user),
    skip: int = 0,
    limit: int = 50,
    user_type: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None
):
    """
    Admin endpoint to get all feedback across the system.
    Can be filtered by user_type and date range.
    
    This endpoint should be protected by admin authentication.
    """
    try:
        from app.crud.crud_feedback import get_all_feedback
        
        # Convert string dates to datetime objects if provided
        from_datetime = None
        to_datetime = None
        
        if from_date:
            from_datetime = datetime.fromisoformat(from_date)
        if to_date:
            to_datetime = datetime.fromisoformat(to_date)
        
        feedback_list = await get_all_feedback(
            db=db,
            skip=skip,
            limit=limit,
            user_type=user_type,
            from_date=from_datetime,
            to_date=to_datetime
        )
        
        return feedback_list
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve feedback: {str(e)}"
        )
