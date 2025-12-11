import uuid
from datetime import datetime
from typing import Optional, List
from sqlalchemy import Column, String, DateTime, ForeignKey, Text, Integer, Boolean, func, JSON, Enum
from sqlalchemy.orm import relationship
from pydantic import BaseModel, Field, validator
import enum

from app.db.base_class import Base

class DocumentType(str, enum.Enum):
    """Types of documents that can be uploaded"""
    PDF = "pdf"
    CSV = "csv"
    DOC = "doc"
    DOCX = "docx"
    TXT = "txt"
    XLS = "xls"
    XLSX = "xlsx"
    PNG = "png"
    JPG = "jpg"
    JPEG = "jpeg"
    OTHER = "other"

class DocumentStatus(str, enum.Enum):
    """Status of a document"""
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PENDING = "pending"

class OwnerType(str, enum.Enum):
    """Types of document owners"""
    CUSTOMER = "customer"
    EMPLOYEE = "employee"
    GUEST = "guest"

class Document(Base):
    """
    Model for document storage information.
    """
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    temp_file_path = Column(String(500), nullable=True)  # Path to temporary file
    file_size = Column(Integer, nullable=False)  # in bytes
    file_type = Column(Enum(DocumentType), nullable=False)
    mime_type = Column(String(100), nullable=True)
    
    # Document metadata
    title = Column(String(255), nullable=True)
    description = Column(Text, nullable=True)
    status = Column(Enum(DocumentStatus), default=DocumentStatus.PENDING, nullable=False)
    processing_error = Column(Text, nullable=True)
    doc_metadata = Column(JSON, nullable=True)  # For storing extracted metadata
    is_processed = Column(Boolean, default=False)  # Flag to indicate if file moved from temp to permanent
    
    # Owner information
    owner_type = Column(Enum(OwnerType), default=OwnerType.CUSTOMER, nullable=False)
    owner_id = Column(Integer, nullable=False, index=True)
    customer_id = Column(Integer, ForeignKey("customers.id", ondelete="CASCADE"), nullable=True, index=True)
    employee_id = Column(Integer, ForeignKey("employees.id", ondelete="CASCADE"), nullable=True, index=True)
    guest_id = Column(Integer, ForeignKey("guests.id", ondelete="CASCADE"), nullable=True, index=True)
    is_public = Column(Boolean, default=False, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    processed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships - using string references to avoid circular imports
    customer = relationship("Customer", back_populates="documents", foreign_keys=[customer_id])
    employee = relationship("Employee", back_populates="documents", foreign_keys=[employee_id])
    guest = relationship("Guest", back_populates="documents", foreign_keys=[guest_id])
    
    def __repr__(self):
        return f"<Document(id={self.id}, filename={self.filename}, type={self.file_type}, owner_type={self.owner_type})>"
        
    @classmethod
    async def search_documents(cls, db, search_text, owner_type=None, owner_id=None, file_type=None, status=None, skip=0, limit=20):
        """
        Search for documents based on text in title, description or metadata.
        
        Args:
            db: Database session
            search_text: Text to search for in title, description or metadata
            owner_type: Optional filter by owner type
            owner_id: Optional filter by owner ID
            file_type: Optional filter by file type
            status: Optional filter by document status
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of matching Document objects
        """
        from sqlalchemy import or_, and_, String
        
        # Use the search_documents function from crud_document.py
        from app.crud.crud_document import search_documents
        return await search_documents(db, search_text, owner_type, owner_id, file_type, status, skip, limit)

# Add relationship to Customer class - this will be patched in the customer model

# Pydantic schemas for API
class DocumentBase(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    is_public: bool = False

class DocumentCreate(DocumentBase):
    owner_type: OwnerType = OwnerType.CUSTOMER

class DocumentUpdate(DocumentBase):
    status: Optional[DocumentStatus] = None
    doc_metadata: Optional[dict] = None

class DocumentResponse(DocumentBase):
    id: int
    filename: str
    original_filename: str
    file_size: int
    file_type: DocumentType
    mime_type: Optional[str]
    status: DocumentStatus
    owner_type: OwnerType
    owner_id: int
    customer_id: Optional[int] = None
    employee_id: Optional[int] = None
    guest_id: Optional[int] = None
    created_at: datetime
    updated_at: datetime
    processed_at: Optional[datetime] = None
    is_processed: bool
    # doc_metadata: Optional[dict] = None
    # metadata: Optional[dict] = None
    
    class Config:
        from_attributes = True

class DocumentList(BaseModel):
    total: int
    items: List[DocumentResponse]
