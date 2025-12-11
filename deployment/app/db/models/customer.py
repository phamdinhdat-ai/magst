import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy import Column, String, DateTime, ForeignKey, Text, Integer, Boolean, func, JSON, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
from pydantic import BaseModel, Field, validator, EmailStr
import enum
from passlib.context import CryptContext

# Import base class
from app.db.base_class import Base

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class CustomerRole(str, enum.Enum):
    """Customer roles for authorization"""
    CUSTOMER = "customer"
    PREMIUM_CUSTOMER = "premium_customer"
    VIP_CUSTOMER = "vip_customer"
    ADMIN = "admin"

class CustomerStatus(str, enum.Enum):
    """Customer account status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING_VERIFICATION = "pending_verification"

class CustomerSessionStatus(str, enum.Enum):
    """Status of a customer session"""
    ACTIVE = "active"
    EXPIRED = "expired"
    COMPLETED = "completed"
    
class CustomerInteractionType(str, enum.Enum):
    """Types of interactions a customer can have"""
    COMPANY_INFO = "company_info"
    PRODUCT_INQUIRY = "product_inquiry" 
    MEDICAL_QUESTION = "medical_question"
    DRUG_INQUIRY = "drug_inquiry"
    GENETIC_QUESTION = "genetic_question"
    GENERAL_SEARCH = "general_search"
    SUPPORT = "support"
    ACCOUNT_MANAGEMENT = "account_management"
    BILLING_INQUIRY = "billing_inquiry"

# SQLAlchemy Models
class Customer(Base):
    """
    Model for authenticated customer accounts.
    Customers have persistent accounts with authentication and authorization.
    """
    __tablename__ = "customers"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(Integer, unique=True, index=True, nullable=False)  # Integer username as code_id
    hashed_password = Column(String(255), nullable=False)
    
    # Customer profile
    first_name = Column(String(100), nullable=True)
    last_name = Column(String(100), nullable=True)
    phone_number = Column(String(20), nullable=True)
    date_of_birth = Column(DateTime(timezone=True), nullable=True)
    
    # Account management
    role = Column(Enum(CustomerRole), default=CustomerRole.CUSTOMER, nullable=False)
    status = Column(Enum(CustomerStatus), default=CustomerStatus.PENDING_VERIFICATION, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    email_verified = Column(Boolean, default=False, nullable=False)
    phone_verified = Column(Boolean, default=False, nullable=False)
    
    # Security
    failed_login_attempts = Column(Integer, default=0)
    locked_until = Column(DateTime(timezone=True), nullable=True)
    last_password_change = Column(DateTime(timezone=True), nullable=True)
    must_change_password = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    last_login_at = Column(DateTime(timezone=True), nullable=True)
    last_activity_at = Column(DateTime(timezone=True), nullable=True)
    
    # Preferences and metadata
    preferred_language = Column(String(10), default="vi", nullable=True)
    timezone = Column(String(50), default="Asia/Ho_Chi_Minh", nullable=True)
    customer_metadata = Column(JSON, nullable=True)
    
    # Analytics
    total_sessions = Column(Integer, default=0)
    total_interactions = Column(Integer, default=0)
    
    # Relationships
    sessions = relationship(
        "CustomerSession", 
        back_populates="customer",
        cascade="all, delete-orphan"
    )
    
    interactions = relationship(
        "CustomerInteraction",
        back_populates="customer",
        cascade="all, delete-orphan"
    )
    
    chat_threads = relationship(
        "CustomerChatThread",
        back_populates="customer", 
        cascade="all, delete-orphan"
    )
    
    refresh_tokens = relationship(
        "CustomerRefreshToken",
        back_populates="customer",
        cascade="all, delete-orphan"
    )
    
    documents = relationship(
        "Document",
        back_populates="customer",
        cascade="all, delete-orphan"
    )
    
    # Using string-based relationship definition to avoid circular imports
    feedback = relationship(
        "app.db.models.feedback.FeedbackModel",
        back_populates="customer",
        foreign_keys="app.db.models.feedback.FeedbackModel.customer_id",
        cascade="all, delete-orphan"
    )
    
    def __repr__(self):
        return f"<Customer(id={self.id}, email='{self.email}', role='{self.role}')>"
    
    def verify_password(self, password: str) -> bool:
        """Verify password against hash"""
        return pwd_context.verify(password, self.hashed_password)
    
    def set_password(self, password: str):
        """Set password hash"""
        self.hashed_password = pwd_context.hash(password)
        self.last_password_change = datetime.utcnow()
    
    @property
    def full_name(self):
        """Get customer's full name"""
        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"
        return self.first_name or self.last_name or self.username or self.email
    
    @property
    def is_active(self):
        """Check if customer account is active"""
        return self.status == CustomerStatus.ACTIVE
    
    @property
    def is_locked(self):
        """Check if account is locked"""
        if self.locked_until:
            return datetime.utcnow() < self.locked_until
        return False

class CustomerRefreshToken(Base):
    """
    Model for storing customer refresh tokens for JWT authentication.
    """
    __tablename__ = "customer_refresh_tokens"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    customer_id = Column(Integer, ForeignKey("customers.id", ondelete="CASCADE"), nullable=False, index=True)
    token_hash = Column(String(255), unique=True, index=True, nullable=False)
    
    # Token metadata
    device_info = Column(String(255), nullable=True)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=False)
    last_used_at = Column(DateTime(timezone=True), nullable=True)
    
    # Security
    is_revoked = Column(Boolean, default=False)
    revoked_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    customer = relationship("Customer", back_populates="refresh_tokens")
    
    def __repr__(self):
        return f"<CustomerRefreshToken(id={self.id}, customer_id={self.customer_id}, revoked={self.is_revoked})>"

class CustomerSession(Base):
    """
    Model for tracking customer user sessions.
    """
    __tablename__ = "customer_sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    customer_id = Column(Integer, ForeignKey("customers.id", ondelete="CASCADE"), nullable=False, index=True)
    session_id = Column(String(255), unique=True, index=True, nullable=False)
    
    # Session info
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    device_info = Column(String(255), nullable=True)
    
    # Session tracking
    status = Column(Enum(CustomerSessionStatus), default=CustomerSessionStatus.ACTIVE, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_activity_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=True)
    ended_at = Column(DateTime(timezone=True), nullable=True)
    
    # Analytics
    total_interactions = Column(Integer, default=0)
    session_metadata = Column(JSON, nullable=True)
    
    # Relationships
    customer = relationship("Customer", back_populates="sessions")
    interactions = relationship(
        "CustomerInteraction", 
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="CustomerInteraction.created_at"
    )
    
    def __repr__(self):
        return f"<CustomerSession(id={self.id}, customer_id={self.customer_id}, status='{self.status}')>"

class CustomerInteraction(Base):
    """
    Model for tracking individual interactions by customer users.
    """
    __tablename__ = "customer_interactions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    customer_id = Column(Integer, ForeignKey("customers.id", ondelete="CASCADE"), nullable=False, index=True)
    session_id = Column(UUID(as_uuid=True), ForeignKey("customer_sessions.id", ondelete="CASCADE"), nullable=True, index=True)
    
    # Interaction details
    interaction_type = Column(Enum(CustomerInteractionType), nullable=False, index=True)
    original_query = Column(Text, nullable=False)
    rewritten_query = Column(Text, nullable=True)
    
    # Agent workflow data
    classified_agent = Column(String(100), nullable=True)
    agent_response = Column(Text, nullable=True)
    suggested_questions = Column(JSON, nullable=True)
    
    # Metadata
    processing_time_ms = Column(Integer, nullable=True)
    iteration_count = Column(Integer, default=1)
    workflow_metadata = Column(JSON, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Quality tracking
    user_feedback_rating = Column(Integer, nullable=True)
    user_feedback_text = Column(Text, nullable=True)
    was_helpful = Column(Boolean, nullable=True)
    
    # Relationships
    customer = relationship("Customer", back_populates="interactions")
    session = relationship("CustomerSession", back_populates="interactions")
    
    def __repr__(self):
        return f"<CustomerInteraction(id={self.id}, type='{self.interaction_type}', customer_id={self.customer_id})>"

class CustomerChatThread(Base):
    """
    Model for customer chat conversations.
    """
    __tablename__ = "customer_chat_threads"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    customer_id = Column(Integer, ForeignKey("customers.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Chat details
    title = Column(String(255), nullable=True)
    summary = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_message_at = Column(DateTime(timezone=True), nullable=True)
    
    # Chat metadata
    message_count = Column(Integer, default=0)
    primary_topic = Column(String(100), nullable=True)
    conversation_metadata = Column(JSON, nullable=True)
    
    # Privacy settings
    is_archived = Column(Boolean, default=False)
    is_deleted = Column(Boolean, default=False)
    
    # Relationships
    customer = relationship("Customer", back_populates="chat_threads")
    messages = relationship(
        "CustomerChatMessage",
        back_populates="thread",
        cascade="all, delete-orphan",
        order_by="CustomerChatMessage.sequence_number"
    )
    
    def __repr__(self):
        return f"<CustomerChatThread(id={self.id}, customer_id={self.customer_id}, message_count={self.message_count})>"
    
    @property
    def chat_list_title(self):
        """Generate a display title for the chat thread"""
        if self.title:
            return self.title
        if self.created_at:
            return f"Chat {self.created_at.strftime('%Y-%m-%d %H:%M')}"
        return f"Chat {str(self.id)[:8]}"

class CustomerChatMessage(Base):
    """
    Individual messages within a customer chat thread.
    """
    __tablename__ = "customer_chat_messages"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    thread_id = Column(UUID(as_uuid=True), ForeignKey("customer_chat_threads.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Message details
    sequence_number = Column(Integer, nullable=False, index=True)
    speaker = Column(String(20), nullable=False)  # "customer" or "assistant"
    content = Column(Text, nullable=False)
    
    # Message metadata
    agent_used = Column(String(100), nullable=True)
    processing_time_ms = Column(Integer, nullable=True)
    message_metadata = Column(JSON, nullable=True)
    
    # Privacy
    is_deleted = Column(Boolean, default=False)
    deleted_at = Column(DateTime(timezone=True), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    thread = relationship("CustomerChatThread", back_populates="messages")
    
    def __repr__(self):
        return f"<CustomerChatMessage(id={self.id}, speaker='{self.speaker}', seq={self.sequence_number})>"

# Pydantic Schemas for API

# Authentication schemas
class CustomerRegister(BaseModel):
    """Schema for customer registration"""
    email: EmailStr = Field(..., description="Customer email address")
    password: str = Field(..., min_length=8, description="Customer password (min 8 characters)")
    username: int = Field(..., description="Customer ID as integer username")
    first_name: Optional[str] = Field(None, max_length=100, description="First name")
    last_name: Optional[str] = Field(None, max_length=100, description="Last name")
    phone_number: Optional[str] = Field(None, max_length=20, description="Phone number")
    preferred_language: Optional[str] = Field("vi", description="Preferred language code")
    
    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        return v

class CustomerLogin(BaseModel):
    """Schema for customer login"""
    email: EmailStr = Field(..., description="Customer email address")
    password: str = Field(..., description="Customer password")
    username: Optional[int] = Field(None, description="Customer ID as integer username")
    remember_me: Optional[bool] = Field(False, description="Remember login session")
    device_info: Optional[str] = Field(None, description="Device information")
    # Note: Login is still primarily by email and password
    # Username (as integer) can be added if needed for specific authentication flows

class CustomerLoginResponse(BaseModel):
    """Schema for login response"""
    access_token: str = Field(..., description="JWT access token")
    refresh_token: str = Field(..., description="JWT refresh token")
    token_type: str = Field("bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration time in seconds")
    customer: "CustomerResponse"

class TokenRefresh(BaseModel):
    """Schema for token refresh"""
    refresh_token: str = Field(..., description="Refresh token")

class PasswordChange(BaseModel):
    """Schema for password change"""
    current_password: str = Field(..., description="Current password")
    new_password: str = Field(..., min_length=8, description="New password")
    
    @validator('new_password')
    def validate_new_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        return v

class PasswordReset(BaseModel):
    """Schema for password reset request"""
    email: EmailStr = Field(..., description="Customer email address")

class PasswordResetConfirm(BaseModel):
    """Schema for password reset confirmation"""
    token: str = Field(..., description="Reset token")
    new_password: str = Field(..., min_length=8, description="New password")

# Customer profile schemas
class CustomerBase(BaseModel):
    """Base schema for customers"""
    email: EmailStr = Field(..., description="Customer email address")
    username: Optional[int] = Field(None, description="Customer ID as integer username")
    first_name: Optional[str] = Field(None, description="First name")
    last_name: Optional[str] = Field(None, description="Last name")
    phone_number: Optional[str] = Field(None, description="Phone number")
    preferred_language: Optional[str] = Field("vi", description="Preferred language code")
    timezone: Optional[str] = Field("Asia/Ho_Chi_Minh", description="Customer timezone")

class CustomerCreate(CustomerBase):
    """Schema for creating a customer"""
    password: str = Field(..., min_length=8, description="Customer password")

class CustomerUpdate(BaseModel):
    """Schema for updating customer profile"""
    username: Optional[int] = Field(None, description="Customer ID as integer username")
    first_name: Optional[str] = Field(None, max_length=100)
    last_name: Optional[str] = Field(None, max_length=100)
    phone_number: Optional[str] = Field(None, max_length=20)
    preferred_language: Optional[str] = None
    timezone: Optional[str] = None
    customer_metadata: Optional[Dict[str, Any]] = None

class CustomerAdminUpdate(CustomerUpdate):
    """Schema for admin updates to customer"""
    role: Optional[CustomerRole] = None
    status: Optional[CustomerStatus] = None
    is_verified: Optional[bool] = None
    email_verified: Optional[bool] = None
    phone_verified: Optional[bool] = None

class CustomerResponse(CustomerBase):
    """Schema for customer responses"""
    id: int  # Changed from uuid.UUID to int to match SQLAlchemy model
    role: CustomerRole
    status: CustomerStatus
    is_verified: bool
    email_verified: bool
    phone_verified: bool
    created_at: datetime
    updated_at: datetime
    last_login_at: Optional[datetime]
    last_activity_at: Optional[datetime]
    total_sessions: int
    total_interactions: int
    full_name: str
    is_active: bool
    
    class Config:
        from_attributes = True

# Session schemas
class CustomerSessionBase(BaseModel):
    """Base schema for customer sessions"""
    session_metadata: Optional[Dict[str, Any]] = Field(None, description="Additional session data")

class CustomerSessionCreate(CustomerSessionBase):
    """Schema for creating a customer session"""
    customer_id: int = Field(..., description="Customer ID")  # Changed from uuid.UUID to int
    ip_address: Optional[str] = Field(None, description="Client IP address")
    user_agent: Optional[str] = Field(None, description="User agent string")
    device_info: Optional[str] = Field(None, description="Device information")

class CustomerSessionUpdate(BaseModel):
    """Schema for updating customer session"""
    session_metadata: Optional[Dict[str, Any]] = None
    last_activity_at: Optional[datetime] = None

class CustomerSessionResponse(CustomerSessionBase):
    """Schema for customer session responses"""
    id: uuid.UUID
    customer_id: int  # Changed from uuid.UUID to int
    session_id: str
    status: CustomerSessionStatus
    created_at: datetime
    last_activity_at: datetime
    expires_at: Optional[datetime]
    total_interactions: int
    
    class Config:
        from_attributes = True

# Interaction schemas
class CustomerInteractionBase(BaseModel):
    """Base schema for customer interactions"""
    original_query: str = Field(..., description="The original user query")
    interaction_type: Optional[CustomerInteractionType] = Field(None, description="Type of interaction")

class CustomerInteractionCreate(CustomerInteractionBase):
    """Schema for creating a customer interaction"""
    customer_id: int = Field(..., description="Associated customer ID")  # Changed from uuid.UUID to int
    session_id: Optional[uuid.UUID] = Field(None, description="Associated session ID")

class CustomerInteractionUpdate(BaseModel):
    """Schema for updating customer interaction"""
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

class CustomerInteractionResponse(CustomerInteractionBase):
    """Schema for customer interaction responses"""
    id: uuid.UUID
    customer_id: int  # Changed from uuid.UUID to int
    session_id: Optional[uuid.UUID]
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

# Chat thread schemas
class CustomerChatThreadBase(BaseModel):
    """Base schema for customer chat threads"""
    title: Optional[str] = Field(None, description="Chat thread title")
    summary: Optional[str] = Field(None, description="Chat summary")
    primary_topic: Optional[str] = Field(None, description="Main topic of conversation")

class CustomerChatThreadCreate(CustomerChatThreadBase):
    """Schema for creating a customer chat thread"""
    customer_id: int = Field(..., description="Associated customer ID")  # Changed from uuid.UUID to int

class CustomerChatThreadUpdate(CustomerChatThreadBase):
    """Schema for updating a customer chat thread"""
    last_message_at: Optional[datetime] = None
    message_count: Optional[int] = None
    conversation_metadata: Optional[Dict[str, Any]] = None
    is_archived: Optional[bool] = None

class CustomerChatThreadResponse(CustomerChatThreadBase):
    """Schema for customer chat thread responses"""
    id: uuid.UUID
    customer_id: int  # Changed from uuid.UUID to int
    created_at: datetime
    last_message_at: Optional[datetime]
    message_count: int
    conversation_metadata: Optional[Dict[str, Any]]
    is_archived: bool
    is_deleted: bool
    chat_list_title: str
    
    class Config:
        from_attributes = True

# Chat message schemas
class CustomerChatMessageBase(BaseModel):
    """Base schema for customer chat messages"""
    speaker: str = Field(..., description="Message speaker: 'customer' or 'assistant'")
    content: str = Field(..., description="Message content")

class CustomerChatMessageCreate(CustomerChatMessageBase):
    """Schema for creating a customer chat message"""
    thread_id: uuid.UUID = Field(..., description="Associated thread ID")
    sequence_number: int = Field(..., description="Message order in thread")

class CustomerChatMessageUpdate(BaseModel):
    """Schema for updating a customer chat message"""
    agent_used: Optional[str] = None
    processing_time_ms: Optional[int] = None
    message_metadata: Optional[Dict[str, Any]] = None

class CustomerChatMessageResponse(CustomerChatMessageBase):
    """Schema for customer chat message responses"""
    id: uuid.UUID
    thread_id: uuid.UUID
    sequence_number: int
    agent_used: Optional[str]
    processing_time_ms: Optional[int]
    message_metadata: Optional[Dict[str, Any]]
    is_deleted: bool
    created_at: datetime
    
    class Config:
        from_attributes = True

# Workflow-specific schemas
class CustomerWorkflowRequest(BaseModel):
    """Schema for customer workflow requests"""
    query: str = Field(..., description="User query")
    session_id: Optional[str] = Field(None, description="Session ID for continuity")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")

class CustomerWorkflowStreamEvent(BaseModel):
    """Schema for customer workflow streaming events"""
    event: str = Field(..., description="Event type")
    data: Any = Field(..., description="Event data")
    timestamp: Optional[datetime] = Field(None, description="Event timestamp")

class CustomerWorkflowResponse(BaseModel):
    """Schema for customer workflow final response"""
    agent_response: str = Field(..., description="Final agent response")
    suggested_questions: List[str] = Field(default_factory=list, description="Suggested follow-up questions")
    session_id: str = Field(..., description="Session ID")
    interaction_id: uuid.UUID = Field(..., description="Interaction record ID")
    processing_time_ms: Optional[int] = Field(None, description="Total processing time")
    agents_used: List[str] = Field(default_factory=list, description="List of agents that processed the query")

class CustomerFeedback(BaseModel):
    """Schema for customer feedback"""
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
class CustomerAnalytics(BaseModel):
    """Schema for customer usage analytics"""
    total_customers: int = Field(..., description="Total number of customers")
    active_customers: int = Field(..., description="Currently active customers")
    total_sessions: int = Field(..., description="Total customer sessions")
    total_interactions: int = Field(..., description="Total interactions")
    avg_interactions_per_customer: float = Field(..., description="Average interactions per customer")
    most_common_interaction_types: List[Dict[str, Any]] = Field(..., description="Most common interaction types")
    most_used_agents: List[Dict[str, Any]] = Field(..., description="Most frequently used agents")
    avg_processing_time_ms: Optional[float] = Field(None, description="Average processing time")
    user_satisfaction_rating: Optional[float] = Field(None, description="Average user satisfaction rating")
    customer_retention_rate: Optional[float] = Field(None, description="Customer retention rate")

# Admin schemas
class CustomerList(BaseModel):
    """Schema for customer list with pagination"""
    customers: List[CustomerResponse]
    total: int
    page: int
    per_page: int
    total_pages: int

class CustomerStats(BaseModel):
    """Schema for individual customer statistics"""
    customer_id: int  # Changed from uuid.UUID to int
    total_sessions: int
    total_interactions: int
    avg_session_duration_minutes: Optional[float]
    most_used_interaction_type: Optional[str]
    last_activity_at: Optional[datetime]
    satisfaction_rating: Optional[float]
    total_feedback_count: int

# Update forward references
CustomerLoginResponse.model_rebuild()



