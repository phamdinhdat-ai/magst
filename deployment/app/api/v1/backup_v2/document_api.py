from datetime import datetime
from typing import List, Optional, Union, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, File, UploadFile, Form, BackgroundTasks, Query
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
import os
import shutil
from pathlib import Path

from app.db.models.customer import Customer
from app.db.models.employee import Employee
from app.db.models.guest import Guest

from app.api.deps import (
    get_current_customer, get_current_active_customer, get_current_admin_customer,
    get_current_guest, get_current_active_guest
)
from app.api.deps_employee import (
    get_current_employee, get_current_active_employee, get_current_admin_employee
)
from app.api.deps_document import (
    verify_document_access, verify_document_access_customer,
    verify_document_access_employee, verify_document_access_guest,
    rate_limit_document_uploads
)
from app.db.session import get_db_session
from app.db.models.document import (
    Document, DocumentType, DocumentStatus, OwnerType,
    DocumentCreate, DocumentUpdate, DocumentResponse, DocumentList
)
from app.crud.crud_document import (
    get_document, get_customer_documents, get_employee_documents, get_guest_documents, get_user_documents,
    create_document, update_document, delete_document, 
    count_customer_documents, count_employee_documents, count_guest_documents, count_user_documents,
    update_document_status, get_public_documents, process_and_move_document,
    cleanup_temp_files, get_documents_by_status
)
from app.core.config import settings

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])

# Document search endpoint
@router.get("/search", response_model=DocumentList)
async def search_document_api(
    search_text: str,
    owner_type: Optional[OwnerType] = None,
    owner_id: Optional[int] = None,
    file_type: Optional[DocumentType] = None,
    status: Optional[DocumentStatus] = None,
    skip: int = 0,
    limit: int = 20,
    db: AsyncSession = Depends(get_db_session),
    current_user: Union[Customer, Employee] = Depends(get_current_active_employee)
):
    """
    Search for documents based on title, description or metadata content.
    Only authenticated employees can use this endpoint.
    """
    documents = await Document.search_documents(
        db=db, 
        search_text=search_text,
        owner_type=owner_type,
        owner_id=owner_id,
        file_type=file_type,
        status=status,
        skip=skip,
        limit=limit
    )
    
    total = await count_user_documents(db, owner_type, owner_id) if owner_type and owner_id else 0
    
    return DocumentList(
        total=total,
        items=documents
    )

# Vector search endpoint
@router.post("/vector-search", response_model=Dict[str, Any])
async def vector_search(
    query: str,
    owner_type: Optional[str] = Query(None),
    owner_id: Optional[int] = Query(None),
    document_id: Optional[int] = Query(None),
    limit: int = Query(5),
    db: AsyncSession = Depends(get_db_session),
    current_user: Union[Customer, Employee] = Depends(get_current_active_employee)
):
    """
    Search for document content using vector similarity.
    
    This endpoint allows searching for document content based on semantic similarity
    rather than keyword matching. It uses embeddings to find the most relevant
    document chunks for a given query.
    
    Only authenticated employees can use this endpoint.
    """
    try:
        from app.utils.document_processor import query_document_vectors
        
        # Execute vector search
        results = await query_document_vectors(
            query=query,
            owner_type=owner_type,
            owner_id=owner_id,
            document_id=document_id,
            limit=limit
        )
        
        return {
            "query": query,
            "results": results,
            "total_results": len(results)
        }
        
    except ImportError:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Vector search is not available on this server."
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during vector search: {str(e)}"
        )

# Customer-specific document search endpoint
@router.get("/customer/search", response_model=DocumentList)
async def search_customer_documents(
    search_text: str,
    file_type: Optional[DocumentType] = None,
    status: Optional[DocumentStatus] = None,
    skip: int = 0,
    limit: int = 20,
    db: AsyncSession = Depends(get_db_session),
    current_customer: Customer = Depends(get_current_active_customer)
):
    """
    Search for documents belonging to the authenticated customer.
    """
    documents = await Document.search_documents(
        db=db, 
        search_text=search_text,
        owner_type=OwnerType.CUSTOMER,
        owner_id=current_customer.id,
        file_type=file_type,
        status=status,
        skip=skip,
        limit=limit
    )
    
    total = await count_customer_documents(db, current_customer.id)
    
    return DocumentList(
        total=total,
        items=documents
    )

# Employee-specific document search endpoint
@router.get("/employee/search", response_model=DocumentList)
async def search_employee_documents(
    search_text: str,
    file_type: Optional[DocumentType] = None,
    status: Optional[DocumentStatus] = None,
    skip: int = 0,
    limit: int = 20,
    db: AsyncSession = Depends(get_db_session),
    current_employee: Employee = Depends(get_current_active_employee)
):
    """
    Search for documents belonging to the authenticated employee.
    """
    documents = await Document.search_documents(
        db=db, 
        search_text=search_text,
        owner_type=OwnerType.EMPLOYEE,
        owner_id=current_employee.id,
        file_type=file_type,
        status=status,
        skip=skip,
        limit=limit
    )
    
    total = await count_employee_documents(db, current_employee.id)
    
    return DocumentList(
        total=total,
        items=documents
    )

# Document upload endpoint
@router.post("/customer/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_customer_document(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    is_public: bool = Form(False),
    process_immediately: bool = Form(False),
    background_tasks: BackgroundTasks = None,
    current_customer: Customer = Depends(get_current_active_customer),
    db: AsyncSession = Depends(get_db_session),
    _: bool = Depends(rate_limit_document_uploads)
):
    """
    Upload a document file as a customer
    """
    # Validate file size
    content = await file.read()
    size = len(content)
    
    if size > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Max size is {settings.MAX_UPLOAD_SIZE / (1024 * 1024)}MB"
        )
    
    # Prepare file data
    file_data = {
        "filename": file.filename,
        "content_type": file.content_type,
        "size": size
    }
    
    # Prepare document metadata
    doc_in = DocumentCreate(
        title=title or file.filename,
        description=description,
        is_public=is_public,
        owner_type=OwnerType.CUSTOMER
    )
    
    # Create document in database (with temp file path)
    doc = await create_document(db, current_customer.id, OwnerType.CUSTOMER, file_data, doc_in)
    
    # Save the file to temporary location
    temp_dir = Path(os.path.dirname(doc.temp_file_path))
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    with open(doc.temp_file_path, "wb") as f:
        # Reset file pointer to beginning
        await file.seek(0)
        # Write in chunks to handle large files
        while chunk := await file.read(1024 * 1024):  # Read 1MB at a time
            f.write(chunk)
    
    # If process_immediately flag is set, move from temp to permanent storage immediately
    if process_immediately:
        await process_and_move_document(db, doc.id)
    elif background_tasks is not None:
        # Process in background if background_tasks is provided
        background_tasks.add_task(process_and_move_document, db, doc.id)
        await update_document_status(db, doc.id, DocumentStatus.PROCESSING)
    else:
        # Just update status to show it's ready for processing (manual or scheduled processing later)
        await update_document_status(db, doc.id, DocumentStatus.PROCESSING)
    
    return doc

@router.post("/employee/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_employee_document(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    is_public: bool = Form(False),
    process_immediately: bool = Form(False),
    background_tasks: BackgroundTasks = None,
    current_employee: Employee = Depends(get_current_active_employee),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Upload a document file as an employee
    """
    # Validate file size
    content = await file.read()
    size = len(content)
    
    if size > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Max size is {settings.MAX_UPLOAD_SIZE / (1024 * 1024)}MB"
        )
    
    # Prepare file data
    file_data = {
        "filename": file.filename,
        "content_type": file.content_type,
        "size": size
    }
    
    # Prepare document metadata
    doc_in = DocumentCreate(
        title=title or file.filename,
        description=description,
        is_public=is_public,
        owner_type=OwnerType.EMPLOYEE
    )
    
    # Create document in database (with temp file path)
    doc = await create_document(db, current_employee.id, OwnerType.EMPLOYEE, file_data, doc_in)
    
    # Save the file to temporary location
    temp_dir = Path(os.path.dirname(doc.temp_file_path))
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    with open(doc.temp_file_path, "wb") as f:
        # Reset file pointer to beginning
        await file.seek(0)
        # Write in chunks to handle large files
        while chunk := await file.read(1024 * 1024):  # Read 1MB at a time
            f.write(chunk)
    
    # If process_immediately flag is set, move from temp to permanent storage immediately
    if process_immediately:
        await process_and_move_document(db, doc.id)
    elif background_tasks is not None:
        # Process in background if background_tasks is provided
        background_tasks.add_task(process_and_move_document, db, doc.id)
        await update_document_status(db, doc.id, DocumentStatus.PROCESSING)
    else:
        # Just update status to show it's ready for processing (manual or scheduled processing later)
        await update_document_status(db, doc.id, DocumentStatus.PROCESSING)
    
    return doc

@router.post("/guest/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_guest_document(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    is_public: bool = Form(False),
    process_immediately: bool = Form(False),
    background_tasks: BackgroundTasks = None,
    current_guest: Guest = Depends(get_current_active_guest),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Upload a document file as a guest
    """
    # Validate file size
    content = await file.read()
    size = len(content)
    
    if size > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Max size is {settings.MAX_UPLOAD_SIZE / (1024 * 1024)}MB"
        )
    
    # Prepare file data
    file_data = {
        "filename": file.filename,
        "content_type": file.content_type,
        "size": size
    }
    
    # Prepare document metadata
    doc_in = DocumentCreate(
        title=title or file.filename,
        description=description,
        is_public=is_public,
        owner_type=OwnerType.GUEST
    )
    
    # Create document in database (with temp file path)
    doc = await create_document(db, current_guest.id, OwnerType.GUEST, file_data, doc_in)
    
    # Save the file to temporary location
    temp_dir = Path(os.path.dirname(doc.temp_file_path))
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    with open(doc.temp_file_path, "wb") as f:
        # Reset file pointer to beginning
        await file.seek(0)
        # Write in chunks to handle large files
        while chunk := await file.read(1024 * 1024):  # Read 1MB at a time
            f.write(chunk)
    
    # If process_immediately flag is set, move from temp to permanent storage immediately
    if process_immediately:
        await process_and_move_document(db, doc.id)
    elif background_tasks is not None:
        # Process in background if background_tasks is provided
        background_tasks.add_task(process_and_move_document, db, doc.id)
        await update_document_status(db, doc.id, DocumentStatus.PROCESSING)
    else:
        # Just update status to show it's ready for processing (manual or scheduled processing later)
        await update_document_status(db, doc.id, DocumentStatus.PROCESSING)
    
    return doc

# Get document list for current customer
@router.get("/customer", response_model=DocumentList)
async def get_customer_documents_list(
    skip: int = 0,
    limit: int = 100,
    file_type: Optional[DocumentType] = None,
    status: Optional[DocumentStatus] = None,
    current_customer: Customer = Depends(get_current_customer),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Retrieve all documents for the current customer
    """
    docs = await get_customer_documents(db, current_customer.id, skip, limit, file_type, status)
    total = await count_customer_documents(db, current_customer.id)
    
    return {
        "total": total,
        "items": docs
    }

@router.get("/employee", response_model=DocumentList)
async def get_employee_documents_list(
    skip: int = 0,
    limit: int = 100,
    file_type: Optional[DocumentType] = None,
    status: Optional[DocumentStatus] = None,
    current_employee: Employee = Depends(get_current_employee),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Retrieve all documents for the current employee
    """
    docs = await get_employee_documents(db, current_employee.id, skip, limit, file_type, status)
    total = await count_employee_documents(db, current_employee.id)
    
    return {
        "total": total,
        "items": docs
    }

@router.get("/guest", response_model=DocumentList)
async def get_guest_documents_list(
    skip: int = 0,
    limit: int = 100,
    file_type: Optional[DocumentType] = None,
    status: Optional[DocumentStatus] = None,
    current_guest: Guest = Depends(get_current_guest),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Retrieve all documents for the current guest
    """
    docs = await get_guest_documents(db, current_guest.id, skip, limit, file_type, status)
    total = await count_guest_documents(db, current_guest.id)
    
    return {
        "total": total,
        "items": docs
    }

# Get a specific document
@router.get("/customer/{document_id}", response_model=DocumentResponse)
async def get_customer_document(
    document_id: int,
    current_customer: Customer = Depends(get_current_customer),
    db: AsyncSession = Depends(get_db_session),
    _: bool = Depends(verify_document_access_customer)
):
    """
    Get a specific document by ID (for customers)
    """
    doc = await get_document(db, document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    return doc

@router.get("/employee/{document_id}", response_model=DocumentResponse)
async def get_employee_document(
    document_id: int,
    current_employee: Employee = Depends(get_current_employee),
    db: AsyncSession = Depends(get_db_session),
    _: bool = Depends(verify_document_access_employee)
):
    """
    Get a specific document by ID (for employees)
    """
    doc = await get_document(db, document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    return doc

@router.get("/guest/{document_id}", response_model=DocumentResponse)
async def get_guest_document(
    document_id: int,
    current_guest: Guest = Depends(get_current_guest),
    db: AsyncSession = Depends(get_db_session),
    _: bool = Depends(verify_document_access_guest)
):
    """
    Get a specific document by ID (for guests)
    """
    doc = await get_document(db, document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    return doc

# Download document
@router.get("/{document_id}/download", response_model=None)
async def download_document(
    document_id: int,
    current_customer: Customer = Depends(get_current_customer),
    db: AsyncSession = Depends(get_db_session),
    _: bool = Depends(verify_document_access)
):
    """
    Download a document file (from either temporary or permanent location)
    """
    doc = await get_document(db, document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    # First try permanent location if processed
    if doc.is_processed and os.path.exists(doc.file_path):
        return FileResponse(
            path=doc.file_path,
            filename=doc.original_filename,
            media_type=doc.mime_type or "application/octet-stream"
        )
    
    # Otherwise try temporary location
    elif doc.temp_file_path and os.path.exists(doc.temp_file_path):
        return FileResponse(
            path=doc.temp_file_path,
            filename=doc.original_filename,
            media_type=doc.mime_type or "application/octet-stream"
        )
    
    # Neither file exists
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found on server (neither in permanent nor temporary storage)"
        )

# Update document metadata
@router.put("/{document_id}", response_model=DocumentResponse)
async def update_document_by_id(
    document_id: int,
    update_data: DocumentUpdate,
    current_customer: Customer = Depends(get_current_customer),
    db: AsyncSession = Depends(get_db_session),
    _: bool = Depends(verify_document_access)
):
    """
    Update document metadata
    """
    doc = await get_document(db, document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    updated_doc = await update_document(db, doc, update_data)
    return updated_doc

# Delete document
@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT, response_model=None)
async def delete_document_by_id(
    document_id: int,
    current_customer: Customer = Depends(get_current_customer),
    db: AsyncSession = Depends(get_db_session),
    _: bool = Depends(verify_document_access)
):
    """
    Delete a document
    """
    result = await delete_document(db, document_id)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    return None

# Public documents endpoint
@router.get("/public/list", response_model=DocumentList)
async def list_public_documents(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db_session)
):
    """
    List all public documents
    """
    docs = await get_public_documents(db, skip, limit)
    
    # Count total public documents
    result = await db.execute(
        select(func.count()).where(Document.is_public == True)
    )
    total = result.scalar_one()
    
    return {
        "total": total,
        "items": docs
    }

# Process document (move from temp to permanent storage)
@router.post("/{document_id}/process", response_model=DocumentResponse)
async def process_document(
    document_id: int,
    doc_metadata: Optional[dict] = None,
    current_customer: Customer = Depends(get_current_customer),
    db: AsyncSession = Depends(get_db_session),
    _: bool = Depends(verify_document_access)
):
    """
    Process a document and move it from temporary to permanent storage
    """
    # Process the document
    processed_doc = await process_and_move_document(db, document_id, doc_metadata)
    
    if not processed_doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
        
    if processed_doc.status == DocumentStatus.FAILED:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process document: {processed_doc.processing_error}"
        )
    
    return processed_doc

# Cleanup temporary files (admin only)
@router.post("/admin/cleanup-temp", dependencies=[Depends(get_current_admin_customer)])
async def admin_cleanup_temp_files(
    db: AsyncSession = Depends(get_db_session)
):
    """
    Clean up expired temporary files (admin only)
    """
    count = await cleanup_temp_files(db)
    
    return {
        "message": f"Cleaned up {count} expired temporary files",
        "cleaned_count": count
    }
