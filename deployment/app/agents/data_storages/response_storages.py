import pandas as pd 
from app.core.config import get_settings
from app.agents.workflow.state import GraphState as AgentState
from loguru import logger
import os
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from uuid import UUID
import json
from datetime import datetime
from app.db.session import get_db_session
from app.crud.crud_feedback import update_feedback
from app.db.models.employee import EmployeeInteraction
from app.db.models.customer import CustomerInteraction
from app.db.models.guest import GuestInteraction

settings = get_settings()

# Keep the CSV file path for backward compatibility
data_dir = f"{settings.DATA_STORAGE_DIR}/response_storages.csv"
feedback_dir = f"{settings.DATA_STORAGE_DIR}/feedback_storages.csv"

# Helper function to store feedback in CSV as fallback
def _store_feedback_csv(feedback_data: dict) -> None:
    """Store feedback in CSV file as fallback when database operations fail."""
    # Convert to DataFrame
    df = pd.DataFrame([feedback_data])
    
    # Check if file exists
    if not os.path.exists(feedback_dir):
        os.makedirs(os.path.dirname(feedback_dir), exist_ok=True)
        df.to_csv(feedback_dir, index=False)
    else:
        df.to_csv(feedback_dir, mode='a', header=False, index=False)  # Append to existing file
        
    logger.info(f"Feedback stored in CSV fallback storage")


def store_response(state: AgentState) -> None:
    """
    Lưu phản hồi của agent vào file CSV.
    """
    # Chuyển đổi state thành DataFrame
    df = pd.DataFrame([state])
    
    # Kiểm tra nếu file đã tồn tại
    if not os.path.exists(data_dir):
        os.makedirs(os.path.dirname(data_dir), exist_ok=True)
        df.to_csv(data_dir, index=False)
    else:
        df.to_csv(data_dir, mode='a', header=False, index=False)  # Append to existing file
        
        
def load_responses() -> pd.DataFrame:
    """
    Tải tất cả phản hồi đã lưu từ file CSV.
    """
    try:
        df = pd.read_csv(data_dir)
        return df
    except FileNotFoundError:
        return pd.DataFrame()  # Trả về DataFrame rỗng nếu file không tồn tại
    except Exception as e:
        logger.error(f"Error loading responses: {e}")
        return pd.DataFrame()  # Trả về DataFrame rỗng nếu có lỗi khác xảy ra
    
async def store_feedback(interaction_id: str, user_type: str, feedback_type: str, 
                       rating: int = None, feedback_text: str = None, 
                       was_helpful: bool = None, session_id: str = None,
                       message_id: str = None, user_id: str = None) -> None:
    """
    Store user feedback in the database with CSV fallback.
    
    Args:
        interaction_id: The UUID of the interaction
        user_type: Type of user ('employee', 'customer', 'guest')
        feedback_type: Type of feedback ('like', 'dislike', 'neutral')
        rating: Optional numeric rating (1-5)
        feedback_text: Optional text feedback
        was_helpful: Optional boolean indicating if response was helpful
        session_id: Optional session identifier to associate feedback with a specific user session
        message_id: Optional message identifier to associate feedback with a specific message
        user_id: Optional user ID to associate feedback with a specific user
    """
    # Convert interaction_id to UUID if it's a string
    interaction_uuid = None
    if isinstance(interaction_id, str):
        try:
            interaction_uuid = UUID(interaction_id)
        except ValueError:
            logger.error(f"Invalid UUID format for interaction_id: {interaction_id}")
            # We'll still store in CSV even if the UUID is invalid
    else:
        interaction_uuid = interaction_id
            
    # For logging and CSV fallback, create the feedback data dictionary
    feedback_data = {
        'interaction_id': str(interaction_id),
        'session_id': session_id,
        'message_id': message_id,
        'user_id': user_id,
        'user_type': user_type,
        'feedback_type': feedback_type,
        'rating': rating,
        'feedback_text': feedback_text,
        'was_helpful': was_helpful,
        'timestamp': datetime.now().isoformat()
    }
    
    # Log the feedback being stored
    logger.info(f"Storing feedback for {user_type} interaction {interaction_id}: {feedback_type}")
    
    db_success = False
    
    if interaction_uuid:
        try:
            # Create an async database session
            async_session = get_db_session()
            db = await anext(async_session)
            
            try:
                # Use the crud function to update feedback in the database
                await update_feedback(
                    db=db, 
                    interaction_id=interaction_uuid,
                    user_type=user_type,
                    rating=rating,
                    feedback_text=feedback_text,
                    was_helpful=was_helpful,
                    feedback_type=feedback_type
                )
                
                # Store message_id in metadata if provided
                if message_id:
                    # Determine which model to use based on user_type
                    model = None
                    if user_type == 'employee':
                        model = EmployeeInteraction
                    elif user_type == 'customer':
                        model = CustomerInteraction
                    elif user_type == 'guest':
                        model = GuestInteraction
                        
                    if model:
                        # Get the interaction
                        stmt = select(model).where(model.id == interaction_uuid)
                        result = await db.execute(stmt)
                        interaction = result.scalars().first()
                        
                        if interaction:
                            # Update metadata with message_id
                            if not interaction.workflow_metadata:
                                interaction.workflow_metadata = {}
                            interaction.workflow_metadata["message_id"] = message_id
                            
                # Commit the changes
                await db.commit()
                db_success = True
                logger.info(f"Feedback successfully stored in database for {user_type} interaction {interaction_id}")
                
            except Exception as e:
                await db.rollback()
                logger.error(f"Error storing feedback in database: {e}")
                # We'll fall back to CSV storage below
            finally:
                await db.close()
                
        except Exception as e:
            logger.error(f"Database session error: {e}")
    
    # Fall back to CSV storage if database operation failed
    if not db_success:
        _store_feedback_csv(feedback_data)
            
    # Log additional session and message information
    if session_id:
        logger.debug(f"Associated with session: {session_id}")
    if message_id:
        logger.debug(f"Associated with message: {message_id}")


async def load_feedback(session_id: str = None, interaction_id: str = None, message_id: str = None) -> pd.DataFrame:
    """
    Load feedback from the CSV storage with optional filtering.
    
    Args:
        session_id: Optional session identifier to filter feedback
        interaction_id: Optional interaction identifier to filter feedback
        message_id: Optional message identifier to filter feedback
        
    Returns:
        pd.DataFrame: DataFrame containing the feedback data, filtered if parameters are provided
    """
    feedback_dir = f"{settings.DATA_STORAGE_DIR}/feedback_storages.csv"
    
    try:
        df = pd.read_csv(feedback_dir)
        
        # Apply filters if parameters are provided
        if session_id:
            df = df[df['session_id'] == session_id]
        if interaction_id:
            df = df[df['interaction_id'] == interaction_id]
        if message_id:
            df = df[df['message_id'] == message_id]
            
        return df
    except FileNotFoundError:
        logger.warning("Feedback storage file not found")
        return pd.DataFrame()  # Return empty DataFrame if file doesn't exist
    except Exception as e:
        logger.error(f"Error loading feedback: {e}")
        return pd.DataFrame()  # Return empty DataFrame if there's another error