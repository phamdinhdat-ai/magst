from typing import Optional, List, Dict, Any, Union
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_, desc, asc, text, update, String
from sqlalchemy.orm import selectinload, joinedload
import os
import uuid
import shutil
from loguru import logger
import asyncio

# Setup logger
import asyncio

from app.db.models.document import Document, DocumentType, DocumentStatus, OwnerType
from app.db.models.document import DocumentCreate, DocumentUpdate
from app.core.config import settings


# Document CRUD operations

async def get_document(db: AsyncSession, document_id: int) -> Optional[Document]:
    """Get document by ID"""
    result = await db.execute(
        select(Document).where(Document.id == document_id)
    )
    return result.scalar_one_or_none()

async def get_document_by_filename(db: AsyncSession, filename: str) -> Optional[Document]:
    """Get document by filename"""
    result = await db.execute(
        select(Document).where(Document.filename == filename)
    )
    return result.scalar_one_or_none()

async def get_customer_documents(
    db: AsyncSession,
    customer_id: int,
    skip: int = 0,
    limit: int = 100,
    file_type: Optional[DocumentType] = None,
    status: Optional[DocumentStatus] = None
) -> List[Document]:
    """Get documents for a specific customer with optional filters"""
    query = select(Document).where(
        and_(
            Document.owner_type == DocumentType.CUSTOMER,
            Document.owner_id == customer_id,
            Document.customer_id == customer_id
        )
    )
    
    # Apply filters if provided
    if file_type:
        query = query.where(Document.file_type == file_type)
    if status:
        query = query.where(Document.status == status)
        
    # Apply pagination
    query = query.order_by(desc(Document.created_at)).offset(skip).limit(limit)
    
    result = await db.execute(query)
    return result.scalars().all()

async def get_employee_documents(
    db: AsyncSession,
    employee_id: int,
    skip: int = 0,
    limit: int = 100,
    file_type: Optional[DocumentType] = None,
    status: Optional[DocumentStatus] = None
) -> List[Document]:
    """Get documents for a specific employee with optional filters"""
    query = select(Document).where(
        and_(
            Document.owner_type == DocumentType.EMPLOYEE,
            Document.owner_id == employee_id,
            Document.employee_id == employee_id
        )
    )
    
    # Apply filters if provided
    if file_type:
        query = query.where(Document.file_type == file_type)
    if status:
        query = query.where(Document.status == status)
        
    # Apply pagination
    query = query.order_by(desc(Document.created_at)).offset(skip).limit(limit)
    
    result = await db.execute(query)
    return result.scalars().all()

async def get_guest_documents(
    db: AsyncSession,
    guest_id: int,
    skip: int = 0,
    limit: int = 100,
    file_type: Optional[DocumentType] = None,
    status: Optional[DocumentStatus] = None
) -> List[Document]:
    """Get documents for a specific guest with optional filters"""
    query = select(Document).where(
        and_(
            Document.owner_type == DocumentType.GUEST,
            Document.owner_id == guest_id,
            Document.guest_id == guest_id
        )
    )
    
    # Apply filters if provided
    if file_type:
        query = query.where(Document.file_type == file_type)
    if status:
        query = query.where(Document.status == status)
        
    # Apply pagination
    query = query.order_by(desc(Document.created_at)).offset(skip).limit(limit)
    
    result = await db.execute(query)
    return result.scalars().all()

async def get_user_documents(
    db: AsyncSession,
    owner_type: OwnerType,
    owner_id: int,
    skip: int = 0,
    limit: int = 100,
    file_type: Optional[DocumentType] = None,
    status: Optional[DocumentStatus] = None
) -> List[Document]:
    """Get documents for any user type with optional filters"""
    query = select(Document).where(
        and_(
            Document.owner_type == owner_type,
            Document.owner_id == owner_id
        )
    )
    
    # Apply filters if provided
    if file_type:
        query = query.where(Document.file_type == file_type)
    if status:
        query = query.where(Document.status == status)
        
    # Apply pagination
    query = query.order_by(desc(Document.created_at)).offset(skip).limit(limit)
    
    result = await db.execute(query)
    return result.scalars().all()

async def create_document(
    db: AsyncSession,
    owner_id: int,
    owner_type: OwnerType,
    file_data: Dict[str, Any],
    document_in: DocumentCreate
) -> Document:
    """Create new document record with temporary file storage"""
    # Generate a unique filename to store on disk
    original_filename = file_data['filename']
    file_ext = os.path.splitext(original_filename)[1].lower()
    file_name = os.path.splitext(original_filename)[0]
    unique_filename =  f"{owner_type.value}_{owner_id}_{file_name}{file_ext}"
    
    # Determine document type based on extension
    file_type = _determine_file_type(file_ext)
    
    # Calculate temporary file path
    temp_dir = os.path.join(settings.TEMP_UPLOAD_DIR, f"{owner_type.value}_{owner_id}")
    logger.info(f"Creating temporary directory for {owner_type.value} ID {owner_id}: {temp_dir}")
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, unique_filename)
    
    # Calculate permanent file path (will be used after processing)
    final_dir = os.path.join(settings.UPLOAD_DIR, f"{owner_type.value}_{owner_id}")
    os.makedirs(final_dir, exist_ok=True)
    # check if the file already exists not storing it in the final directory
    if os.path.exists(os.path.join(final_dir, unique_filename)):
        logger.warning(f"File {unique_filename} already exists for {owner_type.value} ID {owner_id}. Not storing again.")
        # Use the existing file path
        file_path = os.path.join(final_dir, unique_filename)
    else:
        # Store the file in the final directory
        logger.info(f"Storing new file {unique_filename} for {owner_type.value} ID {owner_id}")
        # Move the file to the final directory
        shutil.move(temp_file_path, os.path.join(final_dir, unique_filename))
    file_path = os.path.join(final_dir, unique_filename)
    logger.info(f"Creating document record for {owner_type.value} ID {owner_id} with file {file_path}")
    
    # Create document record with owner info based on type
    db_obj = Document(
        owner_type=owner_type,
        owner_id=owner_id,
        filename=unique_filename,
        original_filename=original_filename,
        file_path=file_path,
        temp_file_path=temp_file_path,
        file_size=file_data['size'],
        file_type=file_type,
        mime_type=file_data.get('content_type'),
        title=document_in.title or original_filename,
        description=document_in.description,
        is_public=document_in.is_public,
        is_processed=False
    )
    
    # Set the specific ID based on owner type
    if owner_type == OwnerType.CUSTOMER:
        db_obj.customer_id = owner_id
    elif owner_type == OwnerType.EMPLOYEE:
        db_obj.employee_id = owner_id
    elif owner_type == OwnerType.GUEST:
        db_obj.guest_id = owner_id
    
    db.add(db_obj)
    await db.commit()
    await db.refresh(db_obj)
    return db_obj

async def update_document(
    db: AsyncSession,
    document: Document,
    document_update: DocumentUpdate
) -> Document:
    """Update document metadata"""
    update_data = document_update.dict(exclude_unset=True)
    
    for key, value in update_data.items():
        setattr(document, key, value)
        
    await db.commit()
    await db.refresh(document)
    return document

async def delete_document(
    db: AsyncSession,
    document_id: int
) -> bool:
    """Delete document record and file if it exists"""
    document = await get_document(db, document_id)
    if not document:
        return False
    
    # Delete file from disk if it exists
    if document.file_path and os.path.exists(document.file_path):
        try:
            os.remove(document.file_path)
        except OSError:
            logger.error(f"Error deleting file for document {document_id}")
            # Log error but continue with DB deletion
            pass
            
    # Delete any temporary file if it exists
    if document.temp_file_path and os.path.exists(document.temp_file_path):
        try:
            os.remove(document.temp_file_path)
        except OSError:
            logger.warning(f"Error deleting temp file for document {document_id}")
            pass
    
    # Delete document from vector store
    try:
        from app.utils.embeddings import delete_document_embeddings
        
        # Get owner type and id
        owner_type = document.owner_type.value if hasattr(document.owner_type, 'value') else str(document.owner_type)
        owner_id = document.owner_id
        
        # Create background task to delete embeddings
        asyncio.create_task(
            delete_document_embeddings(document.id, owner_type, owner_id)
        )
    except ImportError:
        # If embeddings module is not available, just log
        logger.warning(f"Embeddings module not available, skipping vector store cleanup for document {document_id}")
    
    await db.delete(document)
    await db.commit()
    return True

async def update_document_status(
    db: AsyncSession,
    document_id: int,
    status: DocumentStatus,
    error_message: Optional[str] = None
) -> Optional[Document]:
    """Update document processing status"""
    document = await get_document(db, document_id)
    if not document:
        return None
        
    document.status = status
    if status == DocumentStatus.COMPLETED:
        document.processed_at = datetime.utcnow()
    elif status == DocumentStatus.FAILED and error_message:
        document.processing_error = error_message
        
    await db.commit()
    await db.refresh(document)
    return document

async def count_customer_documents(
    db: AsyncSession,
    customer_id: int
) -> int:
    """Count total documents for a customer"""
    result = await db.execute(
        select(func.count()).where(
            and_(
                Document.owner_type == OwnerType.CUSTOMER,
                Document.owner_id == customer_id
            )
        )
    )
    return result.scalar_one()

async def count_employee_documents(
    db: AsyncSession,
    employee_id: int
) -> int:
    """Count total documents for an employee"""
    result = await db.execute(
        select(func.count()).where(
            and_(
                Document.owner_type == OwnerType.EMPLOYEE,
                Document.owner_id == employee_id
            )
        )
    )
    return result.scalar_one()

async def count_guest_documents(
    db: AsyncSession,
    guest_id: int
) -> int:
    """Count total documents for a guest"""
    result = await db.execute(
        select(func.count()).where(
            and_(
                Document.owner_type == OwnerType.GUEST,
                Document.owner_id == guest_id
            )
        )
    )
    return result.scalar_one()

async def count_user_documents(
    db: AsyncSession,
    owner_type: OwnerType,
    owner_id: int
) -> int:
    """Count total documents for any user type"""
    result = await db.execute(
        select(func.count()).where(
            and_(
                Document.owner_type == owner_type,
                Document.owner_id == owner_id
            )
        )
    )
    return result.scalar_one()

async def get_public_documents(
    db: AsyncSession,
    skip: int = 0,
    limit: int = 100
) -> List[Document]:
    """Get all public documents"""
    query = select(Document).where(Document.is_public == True)
    query = query.order_by(desc(Document.created_at)).offset(skip).limit(limit)
    
    result = await db.execute(query)
    return result.scalars().all()

async def process_and_move_document(
    db: AsyncSession,
    document_id: int,
    process_metadata: Optional[Dict[str, Any]] = None
) -> Optional[Document]:
    """Process a document and move it from temporary to permanent storage"""
    document = await get_document(db, document_id)
    if not document:
        return None
        
    if not document.temp_file_path or not os.path.exists(document.temp_file_path):
        # Update status to failed if temp file doesn't exist
        document.status = DocumentStatus.FAILED
        document.processing_error = "Temporary file not found"
        await db.commit()
        return document
    
    try:
        # Extract text from document for search indexing
        from app.utils.document_processor import extract_text_from_document, update_document_metadata_with_text, process_document_for_rag
        
        # Get file type
        file_type = document.file_type.value if hasattr(document.file_type, 'value') else str(document.file_type)
        
        # Extract text
        extracted_text = extract_text_from_document(
            document.temp_file_path, 
            file_type
        )
        
        # Update metadata with extracted text
        metadata = process_metadata or {}
        if extracted_text:
            document.doc_metadata = update_document_metadata_with_text(metadata, extracted_text)
        
        # Process for RAG in background
        if extracted_text:
            # Get owner type and id
            owner_type = document.owner_type.value if hasattr(document.owner_type, 'value') else str(document.owner_type)
            owner_id = document.owner_id
            
            # Start RAG processing task
            asyncio.create_task(
                process_document_for_rag(
                    document_id=document.id,
                    file_path=document.temp_file_path,
                    file_type=file_type,
                    metadata=document.doc_metadata or {},
                    owner_type=owner_type,
                    owner_id=owner_id
                )
            )
        
        # Ensure the directory for the permanent file exists
        os.makedirs(os.path.dirname(document.file_path), exist_ok=True)
        logger.info(f"Moving document {document_id} from temporary to permanent storage")
        
        # Move the file from temporary to permanent location
        shutil.move(document.temp_file_path, document.file_path) 
        
        # Update document record
        document.is_processed = True
        document.status = DocumentStatus.COMPLETED
        document.processed_at = datetime.utcnow()
        
        # Update document metadata if provided
        if process_metadata:
            if document.doc_metadata:
                document.doc_metadata.update(process_metadata)
            else:
                document.doc_metadata = process_metadata
        
        await db.commit()
        await db.refresh(document)
        return document
        
    except Exception as e:
        # Update status to failed if there's an error during processing
        document.status = DocumentStatus.FAILED
        document.processing_error = str(e)
        await db.commit()
        return document

async def cleanup_temp_files(db: AsyncSession) -> int:
    """Clean up temporary files that are older than the expiry time"""
    expiry_time = datetime.utcnow() - timedelta(seconds=settings.TEMP_FILE_EXPIRY)
    
    # Find documents with temp files that haven't been processed
    # and are older than the expiry time
    query = select(Document).where(
        and_(
            Document.is_processed == False,
            Document.temp_file_path != None,
            Document.created_at < expiry_time
        )
    )
    
    result = await db.execute(query)
    documents = result.scalars().all()
    
    count = 0
    for doc in documents:
        try:
            # Delete the temporary file if it exists
            if doc.temp_file_path and os.path.exists(doc.temp_file_path):
                os.remove(doc.temp_file_path)
                
            # Update document status
            doc.status = DocumentStatus.FAILED
            doc.processing_error = "Temporary file expired and was cleaned up"
            count += 1
        except Exception as e:
            # Log error but continue with other files
            doc.processing_error = f"Error during cleanup: {str(e)}"
    
    # Commit all changes at once
    await db.commit()
    return count

async def get_documents_by_status(
    db: AsyncSession,
    status: DocumentStatus,
    limit: int = 100
) -> List[Document]:
    """Get documents with a specific status"""
    query = select(Document).where(Document.status == status)
    query = query.order_by(asc(Document.created_at)).limit(limit)
    
    result = await db.execute(query)
    return result.scalars().all()

def _determine_file_type(file_ext: str) -> DocumentType:
    """Helper function to determine document type based on file extension"""
    ext_map = {
        '.pdf': DocumentType.PDF,
        '.csv': DocumentType.CSV,
        '.doc': DocumentType.DOC,
        '.docx': DocumentType.DOCX,
        '.txt': DocumentType.TXT,
        '.xls': DocumentType.XLS,
        '.xlsx': DocumentType.XLSX,
        '.png': DocumentType.PNG,
        '.jpg': DocumentType.JPG,
        '.jpeg': DocumentType.JPEG,
    }
    return ext_map.get(file_ext.lower(), DocumentType.OTHER)

# CRUD object for easy import
class DocumentCrud:
    @staticmethod
    async def get(db: AsyncSession, id: int) -> Optional[Document]:
        return await get_document(db, id)
    
    @staticmethod
    async def create(db: AsyncSession, owner_id: int, owner_type: OwnerType, file_data: Dict[str, Any], obj_in: DocumentCreate) -> Document:
        return await create_document(db, owner_id, owner_type, file_data, obj_in)
    
    @staticmethod
    async def update(db: AsyncSession, db_obj: Document, obj_in: DocumentUpdate) -> Document:
        return await update_document(db, db_obj, obj_in)
    
    @staticmethod
    async def delete(db: AsyncSession, id: int) -> bool:
        return await delete_document(db, id)
    
    @staticmethod
    async def get_customer_multi(db: AsyncSession, customer_id: int, skip: int = 0, limit: int = 100) -> List[Document]:
        return await get_customer_documents(db, customer_id, skip, limit)
    
    @staticmethod
    async def get_employee_multi(db: AsyncSession, employee_id: int, skip: int = 0, limit: int = 100) -> List[Document]:
        return await get_employee_documents(db, employee_id, skip, limit)
    
    @staticmethod
    async def get_guest_multi(db: AsyncSession, guest_id: int, skip: int = 0, limit: int = 100) -> List[Document]:
        return await get_guest_documents(db, guest_id, skip, limit)
    
    @staticmethod
    async def get_user_multi(db: AsyncSession, owner_type: OwnerType, owner_id: int, skip: int = 0, limit: int = 100) -> List[Document]:
        return await get_user_documents(db, owner_type, owner_id, skip, limit)
    
    @staticmethod
    async def search(
        db: AsyncSession,
        search_text: str,
        owner_type: Optional[OwnerType] = None,
        owner_id: Optional[int] = None,
        file_type: Optional[DocumentType] = None,
        status: Optional[DocumentStatus] = None,
        skip: int = 0,
        limit: int = 20
    ) -> List[Document]:
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
        query = select(Document).where(
            or_(
                Document.title.ilike(f"%{search_text}%"),
                Document.description.ilike(f"%{search_text}%"),
                func.cast(Document.doc_metadata, String).ilike(f"%{search_text}%")
            )
        )
        
        # Apply additional filters
        if owner_type:
            query = query.where(Document.owner_type == owner_type)
        
        if owner_id:
            query = query.where(Document.owner_id == owner_id)
            
        if file_type:
            query = query.where(Document.file_type == file_type)
            
        if status:
            query = query.where(Document.status == status)
            
        # Apply pagination
        query = query.order_by(desc(Document.created_at)).offset(skip).limit(limit)
        
        result = await db.execute(query)
        return result.scalars().all()

# Create an instance for easy import
document_crud = DocumentCrud()
