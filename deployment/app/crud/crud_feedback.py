from typing import Optional, List, Dict, Any, Union
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, insert, desc, or_, and_, text
from uuid import UUID
import uuid
from datetime import datetime

from app.db.models.employee import EmployeeInteraction
from app.db.models.customer import CustomerInteraction
from app.db.models.guest import GuestInteraction
from app.db.models.feedback import FeedbackModel

async def update_feedback(
    db: AsyncSession,
    interaction_id: UUID,
    user_type: str,
    rating: Optional[int] = None,
    feedback_text: Optional[str] = None,
    was_helpful: Optional[bool] = None,
    feedback_type: Optional[str] = None
):
    """
    Update feedback for an interaction. Works for all user types.
    
    Args:
        db: AsyncSession - Database session
        interaction_id: UUID - Interaction ID
        user_type: str - Type of user ('employee', 'customer', 'guest')
        rating: Optional[int] - Rating from 1 to 5
        feedback_text: Optional[str] - Text feedback
        was_helpful: Optional[bool] - Whether the response was helpful
        feedback_type: Optional[str] - Simple feedback type ('like', 'dislike', 'neutral')
        
    Returns:
        The updated interaction object
    """
    if user_type == 'employee':
        stmt = select(EmployeeInteraction).where(EmployeeInteraction.id == interaction_id)
        result = await db.execute(stmt)
        interaction = result.scalars().first()
        
        if not interaction:
            raise ValueError(f"Employee interaction {interaction_id} not found")
            
        # Update feedback fields
        if rating is not None:
            interaction.user_feedback_rating = rating
        if feedback_text is not None:
            interaction.user_feedback_text = feedback_text
        if was_helpful is not None:
            interaction.was_helpful = was_helpful
        # Store feedback_type in the workflow_metadata if it exists
        if feedback_type is not None:
            if not interaction.workflow_metadata:
                interaction.workflow_metadata = {}
            interaction.workflow_metadata["feedback_type"] = feedback_type
            
    elif user_type == 'customer':
        # Customer interaction update logic
        stmt = select(CustomerInteraction).where(CustomerInteraction.id == interaction_id)
        result = await db.execute(stmt)
        interaction = result.scalars().first()
        
        if not interaction:
            raise ValueError(f"Customer interaction {interaction_id} not found")
            
        # Update feedback fields
        if rating is not None:
            interaction.user_feedback_rating = rating
        if feedback_text is not None:
            interaction.user_feedback_text = feedback_text
        if was_helpful is not None:
            interaction.was_helpful = was_helpful
        # Store feedback_type in the workflow_metadata if it exists
        if feedback_type is not None:
            if not interaction.workflow_metadata:
                interaction.workflow_metadata = {}
            interaction.workflow_metadata["feedback_type"] = feedback_type
            
    elif user_type == 'guest':
        # Guest interaction update logic
        stmt = select(GuestInteraction).where(GuestInteraction.id == interaction_id)
        result = await db.execute(stmt)
        interaction = result.scalars().first()
        
        if not interaction:
            raise ValueError(f"Guest interaction {interaction_id} not found")
            
        # Update feedback fields
        if rating is not None:
            interaction.user_feedback_rating = rating
        if feedback_text is not None:
            interaction.user_feedback_text = feedback_text
        if was_helpful is not None:
            interaction.was_helpful = was_helpful
        # Store feedback_type in the workflow_metadata if it exists
        if feedback_type is not None:
            if not interaction.workflow_metadata:
                interaction.workflow_metadata = {}
            interaction.workflow_metadata["feedback_type"] = feedback_type
    elif user_type == 'anonymous':
        # For anonymous feedback, we don't have a specific table
        # Just log the feedback and return a dictionary
        print(f"Anonymous feedback received for interaction {interaction_id}")
        print(f"Rating: {rating}, Was helpful: {was_helpful}, Feedback type: {feedback_type}")
        print(f"Feedback text: {feedback_text}")
        
        # Since we don't have a real interaction to update,
        # create a simple dict to represent what we received
        interaction = {
            "id": interaction_id,
            "user_type": "anonymous",
            "user_feedback_rating": rating,
            "user_feedback_text": feedback_text,
            "was_helpful": was_helpful,
            "feedback_type": feedback_type
        }
        return interaction
    else:
        raise ValueError(f"Invalid user type: {user_type}")
    
    # Commit changes for database interactions
    try:
        await db.commit()
        await db.refresh(interaction)
    except Exception as e:
        # If the interaction doesn't exist, we can create it or just return what we have
        print(f"Error updating feedback: {e}")
        
    return interaction


async def check_existing_feedback(
    db: AsyncSession,
    interaction_id: UUID,
    user_id: Optional[int] = None,
    user_type: str = "anonymous",
    session_id: Optional[str] = None,
    message_id: Optional[str] = None
) -> bool:
    """
    Check if feedback already exists for this interaction from this user.
    
    Args:
        db: AsyncSession - Database session
        interaction_id: UUID - Interaction ID
        user_id: Optional[int] - User ID if available
        user_type: str - Type of user ('employee', 'customer', 'guest', 'anonymous')
        session_id: Optional[str] - Session ID for guest users
        message_id: Optional[str] - Message ID if available
        
    Returns:
        bool: True if feedback already exists, False otherwise
    """
    try:
        # Build the query conditions
        conditions = [FeedbackModel.interaction_id == interaction_id]
        print(f"Checking for duplicate feedback with interaction_id: {interaction_id}")
        
        # Add user-specific conditions
        if user_type == "customer" and user_id:
            conditions.append(FeedbackModel.customer_id == user_id)
            print(f"Added customer condition: customer_id = {user_id}")
        elif user_type == "employee" and user_id:
            conditions.append(FeedbackModel.employee_id == user_id)
            print(f"Added employee condition: employee_id = {user_id}")
        elif user_type == "guest" and user_id:
            conditions.append(FeedbackModel.guest_id == user_id)
            print(f"Added guest condition: guest_id = {user_id}")
        else:
            # For anonymous users, use session_id or message_id as identifiers
            session_conditions = []
            print(f"Checking anonymous user with session_id: {session_id}, message_id: {message_id}")
            
            if session_id:
                # Only check metadata since session_id column doesn't exist in DB
                session_conditions.append(
                    FeedbackModel.feedback_metadata.op('->>')('session_id') == session_id
                )
                print(f"Added session conditions for session_id: {session_id}")
            
            if message_id:
                session_conditions.append(FeedbackModel.message_id == message_id)
                print(f"Added message condition: message_id = {message_id}")
            
            # Add session conditions if we have any
            if session_conditions:
                conditions.append(or_(*session_conditions))
        
        print(f"Final query conditions: {len(conditions)} conditions")
        
        # Execute the query
        stmt = select(FeedbackModel).where(and_(*conditions))
        result = await db.execute(stmt)
        existing_feedback = result.scalars().first()
        
        if existing_feedback:
            print(f"Found existing feedback for interaction {interaction_id}: {existing_feedback.id}")
            return True
        else:
            print(f"No existing feedback found for interaction {interaction_id}")
            return False
        
    except Exception as e:
        print(f"Error checking existing feedback: {str(e)}")
        import traceback
        print(traceback.format_exc())
        
        # If we can't check reliably, allow the feedback to prevent blocking legitimate feedback
        # But log the issue for debugging
        return False


async def create_feedback(
    db: AsyncSession,
    interaction_id: UUID,
    user_type: str,
    user_id: Optional[int] = None,
    rating: Optional[int] = None,
    feedback_text: Optional[str] = None,
    was_helpful: Optional[bool] = None,
    feedback_type: Optional[str] = None,
    message_id: Optional[str] = None,
    session_id: Optional[str] = None,
    feedback_metadata: Optional[Dict[str, Any]] = None
) -> UUID:
    """
    Create a new feedback entry in the database.
    
    This function stores feedback in a dedicated feedback table with user association,
    similar to how documents are stored.
    
    Args:
        db: AsyncSession - Database session
        interaction_id: UUID - Interaction ID
        user_type: str - Type of user ('employee', 'customer', 'guest', 'anonymous')
        user_id: Optional[int] - User ID if available
        rating: Optional[int] - Rating from 1 to 5
        feedback_text: Optional[str] - Text feedback
        was_helpful: Optional[bool] - Whether the response was helpful
        feedback_type: Optional[str] - Simple feedback type ('like', 'dislike', 'neutral')
        message_id: Optional[str] - ID of the specific message receiving feedback
        session_id: Optional[str] - ID of the session for tracking conversations
        feedback_metadata: Optional[Dict] - Additional metadata to store
        
    Returns:
        UUID of the created feedback entry
    """
    # First, update the interaction as before (if it exists)
    try:
        await update_feedback(
            db=db,
            interaction_id=interaction_id,
            user_type=user_type,
            rating=rating,
            feedback_text=feedback_text,
            was_helpful=was_helpful,
            feedback_type=feedback_type
        )
    except Exception as e:
        print(f"Warning: Could not update interaction: {e}")
    
    # Now create a dedicated feedback entry
    feedback_id = uuid.uuid4()
    
    # Store session_id in both column and metadata for compatibility
    metadata_dict = feedback_metadata or {}
    if session_id:
        metadata_dict["session_id"] = session_id
    
    # Create values dictionary for insert
    values = {
        "id": feedback_id,
        "interaction_id": interaction_id,
        "user_type": user_type,
        "rating": rating,
        "feedback_text": feedback_text,
        "was_helpful": was_helpful,
        "feedback_type": feedback_type,
        "message_id": message_id,
        # Note: session_id column doesn't exist in DB, only store in metadata
        "created_at": datetime.utcnow(),
        "feedback_metadata": metadata_dict
    }
    
    # Add user ID to the correct field based on user_type
    if user_id is not None:
        if user_type == "customer":
            values["customer_id"] = user_id
        elif user_type == "employee":
            values["employee_id"] = user_id
        elif user_type == "guest":
            values["guest_id"] = user_id
    
    # Insert into the feedback table
    try:
        print(f"Inserting feedback with values: {values}")
        # Convert metadata dict to JSON string for PostgreSQL
        values_copy = values.copy()
        if 'feedback_metadata' in values_copy and isinstance(values_copy['feedback_metadata'], dict):
            import json
            values_copy['feedback_metadata'] = json.dumps(values_copy['feedback_metadata'])
        
        stmt = insert(FeedbackModel).values(**values_copy)
        await db.execute(stmt)
        await db.commit()
        print(f"Successfully inserted feedback with ID: {feedback_id}")
        return feedback_id
    except Exception as e:
        print(f"Error inserting feedback: {str(e)}")
        await db.rollback()
        
        # Handle the case where the table doesn't exist yet
        if "relation" in str(e) and "does not exist" in str(e):
            print("Feedback table doesn't exist. Returning feedback ID without database storage.")
            # Just return the ID without database storage - the API will handle CSV fallback
            return feedback_id
            
        # Handle SQLAlchemy text() expression errors
        if "Textual SQL expression" in str(e) and "should be explicitly declared as text" in str(e):
            print("SQLAlchemy text() expression error detected - falling back to direct SQL insertion")
            # We'll skip to the fallback direct SQL method below
        
        # Handle the case where a column doesn't exist (like session_id)
        elif "column" in str(e) and "does not exist" in str(e):
            print(f"Column error detected: {str(e)}")
            print("Removing problematic columns and trying again...")
            
            # Remove the session_id column from values if that's the issue
            if "session_id" in str(e) and "session_id" in values:
                # Move session_id to metadata
                if "feedback_metadata" not in values:
                    values["feedback_metadata"] = {}
                values["feedback_metadata"]["session_id"] = values.pop("session_id")
                print(f"Moved session_id to metadata: {values}")
            
            # Try again with modified values
            try:
                stmt = insert(FeedbackModel).values(**values)
                await db.execute(stmt)
                await db.commit()
                print(f"Successfully inserted feedback with modified values, ID: {feedback_id}")
                return feedback_id
            except Exception as retry_error:
                print(f"Error on retry: {str(retry_error)}")
                await db.rollback()
            
        # Try a more direct approach if the ORM insert fails for other reasons
        try:
            print("Trying fallback direct SQL insertion...")
            # Using raw SQL to avoid SQLAlchemy ORM issues
            # First check if the table exists
            check_sql = text("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'feedback')")
            result = await db.execute(check_sql)
            table_exists = result.scalar()
            
            if not table_exists:
                print("Feedback table confirmed not to exist. Skipping database storage.")
                return feedback_id
            
            # Get actual columns in the feedback table
            columns_sql = text("SELECT column_name FROM information_schema.columns WHERE table_name = 'feedback'")
            result = await db.execute(columns_sql)
            actual_columns = [row[0] for row in result.fetchall()]
            print(f"Actual columns in feedback table: {actual_columns}")
            
            # Filter values to only include existing columns and convert metadata to JSON
            safe_values = {}
            for k, v in values.items():
                if k in actual_columns:
                    if k == 'feedback_metadata' and isinstance(v, dict):
                        # Convert metadata dict to JSON string for PostgreSQL
                        import json
                        safe_values[k] = json.dumps(v)
                        print(f"Converting metadata to JSON: {safe_values[k]}")
                    else:
                        safe_values[k] = v
            
            # Construct dynamic SQL based on actual columns
            columns_str = ", ".join(safe_values.keys())
            placeholders_str = ", ".join([f":{k}" for k in safe_values.keys()])
            
            # If table exists, try direct insert with only existing columns
            raw_sql = text(f"""
            INSERT INTO feedback (
                {columns_str}
            ) VALUES (
                {placeholders_str}
            )
            """)
            await db.execute(raw_sql, safe_values)
            await db.commit()
            print(f"Successfully inserted feedback with fallback method, ID: {feedback_id}")
            return feedback_id
        except Exception as e2:
            print(f"Even fallback insertion failed: {str(e2)}")
            await db.rollback()
            # Return the ID anyway - the API will handle CSV fallback
            return feedback_id


async def get_user_feedback(
    db: AsyncSession,
    user_id: int,
    user_type: str,
    skip: int = 0,
    limit: int = 20
) -> List[Dict[str, Any]]:
    """
    Get all feedback submitted by a specific user.
    
    Args:
        db: AsyncSession - Database session
        user_id: int - User ID
        user_type: str - Type of user ('employee', 'customer', 'guest')
        skip: int - Number of records to skip (for pagination)
        limit: int - Maximum number of records to return
        
    Returns:
        List of feedback entries
    """
    if user_type not in ["employee", "customer", "guest"]:
        raise ValueError(f"Invalid user type: {user_type}")
    
    # Create query based on user type
    if user_type == "customer":
        stmt = select(FeedbackModel).where(FeedbackModel.customer_id == user_id)
    elif user_type == "employee":
        stmt = select(FeedbackModel).where(FeedbackModel.employee_id == user_id)
    elif user_type == "guest":
        stmt = select(FeedbackModel).where(FeedbackModel.guest_id == user_id)
    
    # Add order, pagination
    stmt = stmt.order_by(desc(FeedbackModel.created_at)).offset(skip).limit(limit)
    
    result = await db.execute(stmt)
    feedback_entries = result.scalars().all()
    
    # Convert to list of dictionaries
    return [feedback_entry.to_dict() for feedback_entry in feedback_entries]


async def get_feedback_by_interaction(
    db: AsyncSession,
    interaction_id: UUID
) -> List[Dict[str, Any]]:
    """
    Get all feedback for a specific interaction.
    
    Args:
        db: AsyncSession - Database session
        interaction_id: UUID - Interaction ID
        
    Returns:
        List of feedback entries for this interaction
    """
    stmt = select(FeedbackModel).where(FeedbackModel.interaction_id == interaction_id)
    result = await db.execute(stmt)
    feedback_entries = result.scalars().all()
    
    return [feedback_entry.to_dict() for feedback_entry in feedback_entries]


async def get_all_feedback(
    db: AsyncSession,
    skip: int = 0,
    limit: int = 50,
    user_type: Optional[str] = None,
    from_date: Optional[datetime] = None,
    to_date: Optional[datetime] = None
) -> List[Dict[str, Any]]:
    """
    Get all feedback with optional filtering.
    
    Args:
        db: AsyncSession - Database session
        skip: int - Number of records to skip (for pagination)
        limit: int - Maximum number of records to return
        user_type: Optional[str] - Filter by user type ('employee', 'customer', 'guest')
        from_date: Optional[datetime] - Filter by feedback created after this date
        to_date: Optional[datetime] - Filter by feedback created before this date
        
    Returns:
        List of feedback entries matching the filters
    """
    query = select(FeedbackModel)
    
    # Apply filters
    if user_type:
        query = query.where(FeedbackModel.user_type == user_type)
    
    if from_date:
        query = query.where(FeedbackModel.created_at >= from_date)
        
    if to_date:
        query = query.where(FeedbackModel.created_at <= to_date)
    
    # Add pagination and order by most recent first
    query = query.order_by(desc(FeedbackModel.created_at)).offset(skip).limit(limit)
    
    result = await db.execute(query)
    feedback_entries = result.scalars().all()
    
    # Convert to list of dictionaries
    return [feedback_entry.to_dict() for feedback_entry in feedback_entries]
