import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy import Column, String, DateTime, ForeignKey, Text, Integer, Boolean, func, JSON, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
from pydantic import BaseModel, Field, validator

# Import base class and common enums
from app.db.base_class import Base
from app.db.models.base_models import GuestSessionStatus, GuestInteractionType

# Enums are now imported from base_models.py

# SQLAlchemy Models
class GuestSession(Base):
    """
    Model for tracking guest user sessions.
    Guests don't have persistent accounts but we track their session for better UX.
    """
    __tablename__ = "guest_sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(String(255), unique=True, index=True, nullable=False)  # Generated session ID
    ip_address = Column(String(45), nullable=True)  # IPv4/IPv6 support
    user_agent = Column(Text, nullable=True)  # Browser/device info for analytics
    
    # Session tracking
    status = Column(Enum(GuestSessionStatus), default=GuestSessionStatus.ACTIVE, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_activity_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=True)  # Optional session expiry
    
    # Analytics and preferences
    total_interactions = Column(Integer, default=0)
    preferred_language = Column(String(10), default="vi", nullable=True)  # ISO language code
    session_metadata = Column(JSON, nullable=True)  # Store additional session data
    
    # Relationships
    interactions = relationship(
        "GuestInteraction", 
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="GuestInteraction.created_at"
    )
    
    chat_threads = relationship(
        "GuestChatThread",
        back_populates="session", 
        cascade="all, delete-orphan"
    )
    
    def __repr__(self):
        return f"<GuestSession(id={self.id}, session_id='{self.session_id}', status='{self.status}')>"

# Guest model definition
class Guest(Base):
    """
    Model for guest users.
    """
    __tablename__ = "guests"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(255), unique=True, index=True, nullable=False)
    
    # Session tracking
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    
    # Preferences
    preferred_language = Column(String(10), default="vi", nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_activity_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationship with documents
    # Using string-based relationship definition to avoid circular imports
    documents = relationship(
        "Document",
        back_populates="guest", 
        cascade="all, delete-orphan",
        foreign_keys="Document.guest_id"
    )
    
    # Relationship with feedback - using string-based definition to avoid circular imports
    feedback = relationship(
        "app.db.models.feedback.FeedbackModel",
        back_populates="guest",
        foreign_keys="app.db.models.feedback.FeedbackModel.guest_id",
        cascade="all, delete-orphan"
    )
    
    def __repr__(self):
        return f"<Guest(id={self.id}, session_id='{self.session_id}')>"

class GuestInteraction(Base):
    """
    Model for tracking individual interactions by guest users.
    Each query/response in the guest workflow creates an interaction record.
    """
    __tablename__ = "guest_interactions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("guest_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Interaction details
    interaction_type = Column(Enum(GuestInteractionType), nullable=False, index=True)
    original_query = Column(Text, nullable=False)
    rewritten_query = Column(Text, nullable=True)
    
    # Agent workflow data
    classified_agent = Column(String(100), nullable=True)  # Which agent handled the query
    agent_response = Column(Text, nullable=True)
    suggested_questions = Column(JSON, nullable=True)  # Array of suggested follow-up questions
    
    # Metadata
    processing_time_ms = Column(Integer, nullable=True)  # How long the query took to process
    iteration_count = Column(Integer, default=1)  # Number of agent iterations
    workflow_metadata = Column(JSON, nullable=True)  # Store workflow state and agent thinks
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Quality tracking
    user_feedback_rating = Column(Integer, nullable=True)  # 1-5 rating if provided
    user_feedback_text = Column(Text, nullable=True)
    was_helpful = Column(Boolean, nullable=True)
    
    # Relationships
    session = relationship("GuestSession", back_populates="interactions")
    
    def __repr__(self):
        return f"<GuestInteraction(id={self.id}, type='{self.interaction_type}', agent='{self.classified_agent}')>"

class GuestChatThread(Base):
    """
    Model for guest chat conversations.
    Similar to regular chat threads but for guest users.
    """
    __tablename__ = "guest_chat_threads"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("guest_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Chat details
    title = Column(String(255), nullable=True)
    summary = Column(Text, nullable=True)  # AI-generated summary of the conversation
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_message_at = Column(DateTime(timezone=True), nullable=True)
    
    # Chat metadata
    message_count = Column(Integer, default=0)
    primary_topic = Column(String(100), nullable=True)  # Main topic discussed
    conversation_metadata = Column(JSON, nullable=True)
    
    # Relationships
    session = relationship("GuestSession", back_populates="chat_threads")
    messages = relationship(
        "GuestChatMessage",
        back_populates="thread",
        cascade="all, delete-orphan",
        order_by="GuestChatMessage.sequence_number"
    )
    
    def __repr__(self):
        return f"<GuestChatThread(id={self.id}, title='{self.title}', message_count={self.message_count})>"
    
    @property
    def chat_list_title(self):
        """Generate a display title for the chat thread"""
        if self.title:
            return self.title
        if self.created_at:
            return f"Guest Chat {self.created_at.strftime('%Y-%m-%d %H:%M')}"
        return f"Guest Chat {str(self.id)[:8]}"

class GuestChatMessage(Base):
    """
    Individual messages within a guest chat thread.
    """
    __tablename__ = "guest_chat_messages"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    thread_id = Column(UUID(as_uuid=True), ForeignKey("guest_chat_threads.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Message details
    sequence_number = Column(Integer, nullable=False, index=True)  # Order within thread
    speaker = Column(String(20), nullable=False)  # "guest" or "assistant"
    content = Column(Text, nullable=False)
    
    # Message metadata
    agent_used = Column(String(100), nullable=True)  # Which agent generated this response
    processing_time_ms = Column(Integer, nullable=True)
    message_metadata = Column(JSON, nullable=True)  # Store additional data
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    thread = relationship("GuestChatThread", back_populates="messages")
    
    def __repr__(self):
        return f"<GuestChatMessage(id={self.id}, speaker='{self.speaker}', seq={self.sequence_number})>"

# Pydantic Schemas for API
class GuestSessionBase(BaseModel):
    """Base schema for guest sessions"""
    session_id: str = Field(..., description="Unique session identifier")
    preferred_language: Optional[str] = Field("vi", description="Preferred language code")
    session_metadata: Optional[Dict[str, Any]] = Field(None, description="Additional session data")

class GuestSessionCreate(GuestSessionBase):
    """Schema for creating a new guest session"""
    ip_address: Optional[str] = Field(None, description="Client IP address")
    user_agent: Optional[str] = Field(None, description="User agent string")

class GuestSessionUpdate(BaseModel):
    """Schema for updating guest session"""
    preferred_language: Optional[str] = None
    session_metadata: Optional[Dict[str, Any]] = None
    last_activity_at: Optional[datetime] = None

class GuestSessionResponse(GuestSessionBase):
    """Schema for guest session responses"""
    id: uuid.UUID
    status: GuestSessionStatus
    created_at: datetime
    last_activity_at: datetime
    expires_at: Optional[datetime]
    total_interactions: int
    
    class Config:
        from_attributes = True

class GuestInteractionBase(BaseModel):
    """Base schema for guest interactions"""
    original_query: str = Field(..., description="The original user query")
    interaction_type: Optional[GuestInteractionType] = Field(None, description="Type of interaction")

class GuestInteractionCreate(GuestInteractionBase):
    """Schema for creating a guest interaction"""
    session_id: uuid.UUID = Field(..., description="Associated session ID")

class GuestInteractionUpdate(BaseModel):
    """Schema for updating guest interaction"""
    rewritten_query: Optional[str] = None
    classified_agent: Optional[str] = None
    agent_response: Optional[str] = None
    suggested_questions: Optional[List[str]] = None
    processing_time_ms: Optional[int] = None
    iteration_count: Optional[int] = None
    workflow_metadata: Optional[Dict[str, Any]] = None
    completed_at: Optional[datetime] = None
    user_feedback_rating: Optional[int] = Field(None, ge=1, le=5, description="Rating from 1 to 5")
    user_feedback_text: Optional[str] = None
    was_helpful: Optional[bool] = None

class GuestInteractionResponse(GuestInteractionBase):
    """Schema for guest interaction responses"""
    id: uuid.UUID
    session_id: uuid.UUID
    rewritten_query: Optional[str]
    classified_agent: Optional[str]
    agent_response: Optional[str]
    suggested_questions: Optional[List[str]]
    processing_time_ms: Optional[int]
    iteration_count: int
    workflow_metadata: Optional[Dict[str, Any]]
    created_at: datetime
    completed_at: Optional[datetime]
    user_feedback_rating: Optional[int]
    user_feedback_text: Optional[str]
    was_helpful: Optional[bool]
    
    class Config:
        from_attributes = True

class GuestChatThreadBase(BaseModel):
    """Base schema for guest chat threads"""
    title: Optional[str] = Field(None, description="Chat thread title")
    summary: Optional[str] = Field(None, description="Chat summary")
    primary_topic: Optional[str] = Field(None, description="Main topic of conversation")

class GuestChatThreadCreate(GuestChatThreadBase):
    """Schema for creating a guest chat thread"""
    session_id: uuid.UUID = Field(..., description="Associated session ID")

class GuestChatThreadUpdate(GuestChatThreadBase):
    """Schema for updating a guest chat thread"""
    last_message_at: Optional[datetime] = None
    message_count: Optional[int] = None
    conversation_metadata: Optional[Dict[str, Any]] = None

class GuestChatThreadResponse(GuestChatThreadBase):
    """Schema for guest chat thread responses"""
    id: uuid.UUID
    session_id: uuid.UUID
    created_at: datetime
    last_message_at: Optional[datetime]
    message_count: int
    conversation_metadata: Optional[Dict[str, Any]]
    chat_list_title: str
    
    class Config:
        from_attributes = True

class GuestChatMessageBase(BaseModel):
    """Base schema for guest chat messages"""
    speaker: str = Field(..., description="Message speaker: 'guest' or 'assistant'")
    content: str = Field(..., description="Message content")

class GuestChatMessageCreate(GuestChatMessageBase):
    """Schema for creating a guest chat message"""
    thread_id: uuid.UUID = Field(..., description="Associated thread ID")
    sequence_number: int = Field(..., description="Message order in thread")

class GuestChatMessageUpdate(BaseModel):
    """Schema for updating a guest chat message"""
    agent_used: Optional[str] = None
    processing_time_ms: Optional[int] = None
    message_metadata: Optional[Dict[str, Any]] = None

class GuestChatMessageResponse(GuestChatMessageBase):
    """Schema for guest chat message responses"""
    id: uuid.UUID
    thread_id: uuid.UUID
    sequence_number: int
    agent_used: Optional[str]
    processing_time_ms: Optional[int]
    message_metadata: Optional[Dict[str, Any]]
    created_at: datetime
    
    class Config:
        from_attributes = True

# Workflow-specific schemas
class GuestWorkflowRequest(BaseModel):
    """Schema for guest workflow requests"""
    query: str = Field(..., description="User query")
    session_id: Optional[str] = Field(None, description="Session ID for continuity")
    preferred_language: Optional[str] = Field("vi", description="Preferred response language")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata like priority flags")
    chat_history: Optional[List[Dict[str, str]]] = Field(None, description="Chat history for context")

class GuestWorkflowStreamEvent(BaseModel):
    """Schema for guest workflow streaming events"""
    event: str = Field(..., description="Event type")
    data: Any = Field(..., description="Event data")
    timestamp: Optional[datetime] = Field(None, description="Event timestamp")

class GuestWorkflowResponse(BaseModel):
    """Schema for guest workflow final response"""
    agent_response: str = Field(..., description="Final agent response")
    suggested_questions: List[str] = Field(default_factory=list, description="Suggested follow-up questions")
    session_id: str = Field(..., description="Session ID")
    interaction_id: uuid.UUID = Field(..., description="Interaction record ID")
    processing_time_ms: Optional[int] = Field(None, description="Total processing time")
    agents_used: List[str] = Field(default_factory=list, description="List of agents that processed the query")

class GuestFeedback(BaseModel):
    """Schema for guest feedback"""
    interaction_id: uuid.UUID = Field(..., description="Interaction ID")
    rating: Optional[int] = Field(None, ge=1, le=5, description="Rating from 1 to 5")
    feedback_text: Optional[str] = Field(None, description="Text feedback")
    was_helpful: Optional[bool] = Field(None, description="Whether the response was helpful")
    
    @validator('rating')
    def validate_rating(cls, v):
        if v is not None and (v < 1 or v > 5):
            raise ValueError('Rating must be between 1 and 5')
        return v

# Analytics schemas
class GuestAnalytics(BaseModel):
    """Schema for guest usage analytics"""
    total_sessions: int = Field(..., description="Total number of guest sessions")
    active_sessions: int = Field(..., description="Currently active sessions")
    total_interactions: int = Field(..., description="Total interactions")
    avg_interactions_per_session: float = Field(..., description="Average interactions per session")
    most_common_interaction_types: List[Dict[str, Any]] = Field(..., description="Most common interaction types")
    most_used_agents: List[Dict[str, Any]] = Field(..., description="Most frequently used agents")
    avg_processing_time_ms: Optional[float] = Field(None, description="Average processing time")
    user_satisfaction_rating: Optional[float] = Field(None, description="Average user satisfaction rating")

