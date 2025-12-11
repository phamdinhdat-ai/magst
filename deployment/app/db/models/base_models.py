"""
Base model definitions shared between models to avoid circular imports
"""
import enum

class OwnerType(str, enum.Enum):
    """Types of document owners"""
    CUSTOMER = "customer"
    EMPLOYEE = "employee"
    GUEST = "guest"

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

class GuestSessionStatus(str, enum.Enum):
    """Status of a guest session"""
    ACTIVE = "active"
    EXPIRED = "expired"
    COMPLETED = "completed"
    
class GuestInteractionType(str, enum.Enum):
    """Types of interactions a guest can have"""
    COMPANY_INFO = "company_info"
    PRODUCT_INQUIRY = "product_inquiry" 
    MEDICAL_QUESTION = "medical_question"
    DRUG_INQUIRY = "drug_inquiry"
    GENETIC_QUESTION = "genetic_question"
    GENERAL_SEARCH = "general_search"
    SUPPORT = "support"
