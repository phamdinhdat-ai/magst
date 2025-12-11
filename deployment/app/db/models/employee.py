import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy import Column, String, DateTime, ForeignKey, Text, Integer, Boolean, func, JSON, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
from pydantic import BaseModel, Field, validator, EmailStr
import enum

# Import for document relationship
from app.db.models.document import Document
from passlib.context import CryptContext

# Import base class
from app.db.base_class import Base
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class EmployeeRole(str, enum.Enum):
    """Employee roles for authorization"""
    EMPLOYEE = "employee"
    MANAGER = "manager"
    ADMIN = "admin"

class EmployeeStatus(str, enum.Enum):
    """Employee account status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING_VERIFICATION = "pending_verification"

class EmployeeSessionStatus(str, enum.Enum):
    """Status of an employee session"""
    ACTIVE = "active"
    EXPIRED = "expired"
    COMPLETED = "completed"

class EmployeeInteractionUpdate(BaseModel):
    """Schema for updating employee interaction"""
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

class EmployeeInteractionType(str, enum.Enum):
    """Types of interactions an employee can have"""
    COMPANY_INFO = "company_info"
    PRODUCT_INQUIRY = "product_inquiry" 
    MEDICAL_QUESTION = "medical_question"
    DRUG_INQUIRY = "drug_inquiry"
    GENETIC_QUESTION = "genetic_question"
    GENERAL_SEARCH = "general_search"
    SUPPORT = "support"
    ACCOUNT_MANAGEMENT = "account_management"
    BILLING_INQUIRY = "billing_inquiry"

class EmployeeInteractionBase(BaseModel):
    """Base schema for employee interactions"""
    original_query: str = Field(..., description="The original user query")
    interaction_type: Optional[EmployeeInteractionType] = Field(None, description="Type of interaction")

class EmployeeInteractionCreate(EmployeeInteractionBase):
    """Schema for creating an employee interaction"""
    employee_id: int = Field(..., description="Associated employee ID")  # Changed from uuid.UUID to int
    session_id: Optional[uuid.UUID] = Field(None, description="Associated session ID")


    
class EmployeeInteraction(Base):
    """
    Model for tracking individual interactions by employees.
    """
    __tablename__ = "employee_interactions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    employee_id = Column(Integer, ForeignKey("employees.id", ondelete="CASCADE"), nullable=False, index=True)
    session_id = Column(UUID(as_uuid=True), ForeignKey("employee_sessions.id", ondelete="CASCADE"), nullable=True, index=True)

    # Interaction details
    interaction_type = Column(Enum(EmployeeInteractionType), nullable=False, index=True)
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
    employee = relationship("Employee", back_populates="interactions")
    session = relationship("EmployeeSession", back_populates="interactions")

    def __repr__(self):
        return f"<EmployeeInteraction(id={self.id}, type='{self.interaction_type}', employee_id={self.employee_id})>"






class Employee(Base):
    """
    Model for authenticated employee accounts.
    Employees have persistent accounts with authentication and authorization.
    """
    __tablename__ = "employees"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True, nullable=False)  # String username
    hashed_password = Column(String(255), nullable=False)

    # Employee profile
    first_name = Column(String(100), nullable=True)
    last_name = Column(String(100), nullable=True)
    phone_number = Column(String(20), nullable=True)
    date_of_birth = Column(DateTime(timezone=True), nullable=True)
    
    # Account management
    role = Column(Enum(EmployeeRole), default=EmployeeRole.EMPLOYEE, nullable=False)
    status = Column(Enum(EmployeeStatus), default=EmployeeStatus.PENDING_VERIFICATION, nullable=False)
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
        "EmployeeSession", 
        back_populates="employee",
        cascade="all, delete-orphan"
    )
    
    interactions = relationship(
        "EmployeeInteraction",
        back_populates="employee",
        cascade="all, delete-orphan"
    )
    
    chat_threads = relationship(
        "EmployeeChatThread",
        back_populates="employee", 
        cascade="all, delete-orphan"
    )
    
    refresh_tokens = relationship(
        "EmployeeRefreshToken",
        back_populates="employee",
        cascade="all, delete-orphan"
    )
    
    documents = relationship(
        "Document",
        back_populates="employee",
        cascade="all, delete-orphan"
    )
    
    # Using string-based relationship definition to avoid circular imports
    feedback = relationship(
        "app.db.models.feedback.FeedbackModel",
        back_populates="employee",
        foreign_keys="app.db.models.feedback.FeedbackModel.employee_id",
        cascade="all, delete-orphan"
    )
    
    def __repr__(self):
        return f"<Employee(id={self.id}, email='{self.email}', role='{self.role}')>"
    
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
        """Check if employee account is active"""
        return self.status == EmployeeStatus.ACTIVE

    @property
    def is_locked(self):
        """Check if account is locked"""
        if self.locked_until:
            return datetime.utcnow() < self.locked_until
        return False



class EmployeeRefreshToken(Base):
    """
    Model for storing employee refresh tokens for JWT authentication.
    """
    __tablename__ = "employee_refresh_tokens"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    employee_id = Column(Integer, ForeignKey("employees.id", ondelete="CASCADE"), nullable=False, index=True)
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
    employee = relationship("Employee", back_populates="refresh_tokens")

    def __repr__(self):
        return f"<EmployeeRefreshToken(id={self.id}, employee_id={self.employee_id}, revoked={self.is_revoked})>"

class EmployeeSession(Base):
    """
    Model for tracking employee user sessions.
    """
    __tablename__ = "employee_sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    employee_id = Column(Integer, ForeignKey("employees.id", ondelete="CASCADE"), nullable=False, index=True)
    session_id = Column(String(255), unique=True, index=True, nullable=False)
    
    # Session info
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    device_info = Column(String(255), nullable=True)
    
    # Session tracking
    status = Column(Enum(EmployeeSessionStatus), default=EmployeeSessionStatus.ACTIVE, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_activity_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=True)
    ended_at = Column(DateTime(timezone=True), nullable=True)
    
    # Analytics
    total_interactions = Column(Integer, default=0)
    session_metadata = Column(JSON, nullable=True)
    
    # Relationships
    employee = relationship("Employee", back_populates="sessions")
    interactions = relationship(
        "EmployeeInteraction",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="EmployeeInteraction.created_at"
    )
    
    def __repr__(self):
        return f"<EmployeeSession(id={self.id}, employee_id={self.employee_id}, status='{self.status}')>"


class EmployeeChatThread(Base):
    """
    Model for employee chat conversations.
    """
    __tablename__ = "employee_chat_threads"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    employee_id = Column(Integer, ForeignKey("employees.id", ondelete="CASCADE"), nullable=False, index=True)
    
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
    employee = relationship("Employee", back_populates="chat_threads")
    messages = relationship(
        "EmployeeChatMessage",
        back_populates="thread",
        cascade="all, delete-orphan",
        order_by="EmployeeChatMessage.sequence_number"
    )
    
    def __repr__(self):
        return f"<EmployeeChatThread(id={self.id}, employee_id={self.employee_id}, message_count={self.message_count})>"
    
    @property
    def chat_list_title(self):
        """Generate a display title for the chat thread"""
        if self.title:
            return self.title
        if self.created_at:
            return f"Chat {self.created_at.strftime('%Y-%m-%d %H:%M')}"
        return f"Chat {str(self.id)[:8]}"


class EmployeeChatMessage(Base):
    """
    Individual messages within an employee chat thread.
    """
    __tablename__ = "employee_chat_messages"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    thread_id = Column(UUID(as_uuid=True), ForeignKey("employee_chat_threads.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Message details
    sequence_number = Column(Integer, nullable=False, index=True)
    speaker = Column(String(20), nullable=False)  # "employee" or "assistant"
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
    thread = relationship("EmployeeChatThread", back_populates="messages")
    
    def __repr__(self):
        return f"<EmployeeChatMessage(id={self.id}, speaker='{self.speaker}', seq={self.sequence_number})>"


# =====================================================================================
# Pydantic Models for API
# ====================================================================================

class EmployeeBase(BaseModel):
    """Base schema for employees"""
    email: EmailStr = Field(..., description="Employee email address")
    username: Optional[str] = Field(None, description="Employee's username")
    first_name: Optional[str] = Field(None, description="First name")
    last_name: Optional[str] = Field(None, description="Last name")
    phone_number: Optional[str] = Field(None, description="Phone number")
    preferred_language: Optional[str] = Field("vi", description="Preferred language code")
    timezone: Optional[str] = Field("Asia/Ho_Chi_Minh", description="Employee timezone")





class EmployeeResponse(EmployeeBase):
    """Schema for employee responses"""
    id: int  # Changed from uuid.UUID to int to match SQLAlchemy model
    role: EmployeeRole
    status: EmployeeStatus
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


class EmployeeRegister(BaseModel):
    """Schema for employee registration"""
    email: EmailStr = Field(..., description="Employee email address")
    password: str = Field(..., min_length=8, description="Employee password (min 8 characters)")
    username: Optional[str] = Field(None, description="Employee's username")
    first_name: Optional[str] = Field(None, max_length=100, description="First name")
    last_name: Optional[str] = Field(None, max_length=100, description="Last name")
    phone_number: Optional[str] = Field(None, max_length=20, description="Phone number")
    preferred_language: Optional[str] = Field("vi", description="Preferred language code")
    
    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        return v


class EmployeeLogin(BaseModel):
    """Schema for employee login"""
    email: EmailStr = Field(..., description="Employee email address")
    password: str = Field(..., description="Employee password")
    username: Optional[str] = Field(None, description="Employee's username")
    remember_me: Optional[bool] = Field(False, description="Remember login session")
    device_info: Optional[str] = Field(None, description="Device information")
    # Note: Login is still primarily by email and password
    # Username (as integer) can be added if needed for specific authentication flows

class EmployeeLoginResponse(BaseModel):
    """Schema for login response"""
    access_token: str = Field(..., description="JWT access token")
    refresh_token: str = Field(..., description="JWT refresh token")
    token_type: str = Field("bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration time in seconds")
    employee: "EmployeeResponse"



class EmployeeCreate(EmployeeBase):
    password: str = Field(..., min_length=8)
    
    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        return v

class EmployeeUpdate(BaseModel):
    """Schema for updating employee profile"""
    username: Optional[str] = Field(None, description="Employee's username")
    first_name: Optional[str] = Field(None, max_length=100)
    last_name: Optional[str] = Field(None, max_length=100)
    phone_number: Optional[str] = Field(None, max_length=20)
    preferred_language: Optional[str] = None
    timezone: Optional[str] = None
    employee_metadata: Optional[Dict[str, Any]] = None

# Session schemas
class EmployeeSessionBase(BaseModel):
    """Base schema for employee sessions"""
    session_metadata: Optional[Dict[str, Any]] = Field(None, description="Additional session data")

class EmployeeSessionCreate(EmployeeSessionBase):
    """Schema for creating a employee session"""
    employee_id: int = Field(..., description="Employee ID")  # Changed from uuid.UUID to int
    ip_address: Optional[str] = Field(None, description="Client IP address")
    user_agent: Optional[str] = Field(None, description="User agent string")
    device_info: Optional[str] = Field(None, description="Device information")

class EmployeeSessionUpdate(BaseModel):
    """Schema for updating employee session"""
    session_metadata: Optional[Dict[str, Any]] = None
    last_activity_at: Optional[datetime] = None

class EmployeeSessionResponse(EmployeeSessionBase):
    """Schema for employee session responses"""
    id: uuid.UUID
    employee_id: int  # Changed from uuid.UUID to int
    session_id: str
    status: EmployeeSessionStatus
    created_at: datetime
    last_activity_at: datetime
    expires_at: Optional[datetime]
    total_interactions: int
    
    class Config:
        from_attributes = True

class EmployeePasswordChange(BaseModel):
    current_password: str
    new_password: str = Field(..., min_length=8)
    
    @validator('new_password')
    def validate_new_password(cls, v):
        if len(v) < 8:
            raise ValueError('New password must be at least 8 characters long')
        return v


# =====================================================================================
# Employee Chat Models - Pydantic Schemas
# =====================================================================================

# Chat thread schemas
class EmployeeChatThreadBase(BaseModel):
    """Base schema for employee chat threads"""
    title: Optional[str] = Field(None, description="Chat thread title")
    summary: Optional[str] = Field(None, description="Chat summary")
    primary_topic: Optional[str] = Field(None, description="Main topic of conversation")

class EmployeeChatThreadCreate(EmployeeChatThreadBase):
    """Schema for creating an employee chat thread"""
    employee_id: int = Field(..., description="Associated employee ID")

class EmployeeChatThreadUpdate(EmployeeChatThreadBase):
    """Schema for updating an employee chat thread"""
    last_message_at: Optional[datetime] = None
    message_count: Optional[int] = None
    conversation_metadata: Optional[Dict[str, Any]] = None
    is_archived: Optional[bool] = None

class EmployeeChatThreadResponse(EmployeeChatThreadBase):
    """Schema for employee chat thread responses"""
    id: uuid.UUID
    employee_id: int
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
class EmployeeChatMessageBase(BaseModel):
    """Base schema for employee chat messages"""
    speaker: str = Field(..., description="Message speaker: 'employee' or 'assistant'")
    content: str = Field(..., description="Message content")

class EmployeeChatMessageCreate(EmployeeChatMessageBase):
    """Schema for creating an employee chat message"""
    thread_id: uuid.UUID = Field(..., description="Associated thread ID")
    sequence_number: int = Field(..., description="Message order in thread")

class EmployeeChatMessageUpdate(BaseModel):
    """Schema for updating an employee chat message"""
    agent_used: Optional[str] = None
    processing_time_ms: Optional[int] = None
    message_metadata: Optional[Dict[str, Any]] = None

class EmployeeChatMessageResponse(EmployeeChatMessageBase):
    """Schema for employee chat message responses"""
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
class EmployeeWorkflowRequest(BaseModel):
    """Schema for employee workflow requests"""
    query: str = Field(..., description="User query")
    session_id: Optional[str] = Field(None, description="Session ID for continuity")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")

class EmployeeWorkflowResponse(BaseModel):
    """Schema for employee workflow final response"""
    agent_response: str = Field(..., description="Final agent response")
    suggested_questions: List[str] = Field(default_factory=list, description="Suggested follow-up questions")
    session_id: str = Field(..., description="Session ID")
    thread_id: uuid.UUID = Field(..., description="Chat thread ID")
    processing_time_ms: Optional[int] = Field(None, description="Total processing time")
    agents_used: List[str] = Field(default_factory=list, description="List of agents that processed the query")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata about the response")

class CustomerWorkflowStreamEvent(BaseModel):
    """Schema for customer workflow streaming events"""
    event: str = Field(..., description="Event type")
    data: Any = Field(..., description="Event data")
    timestamp: Optional[datetime] = Field(None, description="Event timestamp")


# Admin schemas
class EmployeeList(BaseModel):
    """Schema for employee list with pagination"""
    employees: List[EmployeeResponse]
    total: int
    page: int
    per_page: int
    total_pages: int
    
