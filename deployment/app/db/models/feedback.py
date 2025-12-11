import uuid
from datetime import datetime
from typing import Optional, Union, Dict, Any
from pydantic import BaseModel, Field, validator, root_validator
from enum import Enum
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Boolean, func, JSON, Enum as SQLAEnum, UUID
from sqlalchemy.orm import relationship

from app.db.base_class import Base

class FeedbackType(str, Enum):
    LIKE = "like"
    DISLIKE = "dislike"
    NEUTRAL = "neutral"

# SQLAlchemy Model for Feedback
class FeedbackModel(Base):
    """
    Database model for storing user feedback, similar to how documents are stored.
    Links feedback to specific users by ID.
    """
    __tablename__ = "feedback"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    interaction_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    
    # User identification - only one of these should be set
    user_type = Column(String(50), nullable=False, index=True)
    customer_id = Column(Integer, ForeignKey("customers.id", ondelete="CASCADE"), nullable=True, index=True)
    employee_id = Column(Integer, ForeignKey("employees.id", ondelete="CASCADE"), nullable=True, index=True)
    guest_id = Column(Integer, ForeignKey("guests.id", ondelete="CASCADE"), nullable=True, index=True)
    
    # Feedback content
    rating = Column(Integer, nullable=True)  # 1-5 stars
    feedback_text = Column(Text, nullable=True)  # Optional text feedback
    was_helpful = Column(Boolean, nullable=True)  # True/False/None
    feedback_type = Column(String(20), nullable=True)  # 'like', 'dislike', 'neutral'
    message_id = Column(String(100), nullable=True)  # Optional specific message ID
    session_id = Column(String(100), nullable=True)  # Optional session ID for tracking conversations
    
    # Metadata and timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    feedback_metadata = Column(JSON, nullable=True)  # For additional metadata
    
    # Relationships - using fully qualified string references to avoid circular imports
    customer = relationship("app.db.models.customer.Customer", back_populates="feedback", foreign_keys=[customer_id])
    employee = relationship("app.db.models.employee.Employee", back_populates="feedback", foreign_keys=[employee_id])
    guest = relationship("app.db.models.guest.Guest", back_populates="feedback", foreign_keys=[guest_id])
    
    def __repr__(self):
        return f"<Feedback(id={self.id}, user_type={self.user_type})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the model to a dictionary for API responses"""
        return {
            "id": str(self.id),
            "interaction_id": str(self.interaction_id),
            "user_type": self.user_type,
            "customer_id": self.customer_id,
            "employee_id": self.employee_id,
            "guest_id": self.guest_id,
            "rating": self.rating,
            "feedback_text": self.feedback_text,
            "was_helpful": self.was_helpful,
            "feedback_type": self.feedback_type,
            "message_id": self.message_id,
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "feedback_metadata": getattr(self, 'feedback_metadata', getattr(self, 'metadata', {})) or {}
        }
    
# Pydantic Schemas for API validation
class BaseFeedback(BaseModel):
    """Base schema for all user feedback"""
    interaction_id: Union[uuid.UUID, str] = Field(..., description="Interaction ID as UUID or string")
    rating: Optional[int] = Field(None, ge=1, le=5, description="Rating from 1 to 5")
    feedback_text: Optional[str] = Field(None, description="Text feedback")
    was_helpful: Optional[bool] = Field(None, description="Whether the response was helpful")
    feedback_type: Optional[Union[FeedbackType, str]] = Field(None, description="Simple feedback type (like/dislike)")
    message_id: Optional[str] = Field(None, description="Optional ID of the specific message receiving feedback")
    session_id: Optional[str] = Field(None, description="Optional session ID for tracking conversations")
    
    @root_validator(pre=True)
    def convert_interaction_id(cls, values):
        """Convert string interaction_id to UUID if needed"""
        if 'interaction_id' in values and isinstance(values['interaction_id'], str):
            try:
                values['interaction_id'] = uuid.UUID(values['interaction_id'])
            except ValueError:
                # If it's not a valid UUID format, create a new UUID based on the string
                # This allows using session-based IDs from the frontend
                values['interaction_id'] = uuid.uuid5(uuid.NAMESPACE_URL, f"feedback:{values['interaction_id']}")
        return values

    @validator('rating')
    def validate_rating(cls, v):
        if v is not None and (v < 1 or v > 5):
            raise ValueError('Rating must be between 1 and 5')
        return v
        
    @validator('feedback_type')
    def validate_feedback_type(cls, v):
        if v is None:
            return None
        if isinstance(v, FeedbackType):
            return v
        if v in ['like', 'dislike', 'neutral']:
            return FeedbackType(v)
        raise ValueError('Feedback type must be like, dislike, or neutral')

# Define response model
class FeedbackResponse(BaseModel):
    id: uuid.UUID
    interaction_id: uuid.UUID
    user_type: str
    customer_id: Optional[int] = None
    employee_id: Optional[int] = None
    guest_id: Optional[int] = None
    rating: Optional[int] = None
    feedback_text: Optional[str] = None
    was_helpful: Optional[bool] = None
    feedback_type: Optional[str] = None
    message_id: Optional[str] = None
    session_id: Optional[str] = None
    created_at: datetime
    feedback_metadata: Optional[Dict[str, Any]] = None
    
    class Config:
        from_attributes = True

class EmployeeFeedback(BaseFeedback):
    """Schema for employee feedback"""
    pass

class CustomerFeedback(BaseFeedback):
    """Schema for customer feedback"""
    pass

class GuestFeedback(BaseFeedback):
    """Schema for guest feedback"""
    pass
