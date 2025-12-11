"""
Document-enhanced workflow API endpoints
Enables RAG-based responses leveraging user-uploaded documents
"""
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional, Dict, Any, Union
import logging
import asyncio

from app.db.session import get_db_session
from app.api.deps import get_current_guest, get_current_customer
from app.api.deps_employee import get_current_employee
from app.utils.document_processor import query_document_vectors
from app.db.models.document import OwnerType

# Setup logger
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/rag", tags=["rag"])

# Guest RAG workflow endpoint
@router.post("/guest/query", response_model=Dict[str, Any])
async def rag_guest_query(
    query: str,
    session_id: str,
    document_id: Optional[int] = None,
    limit: int = Query(3, gt=0, le=10),
    db: AsyncSession = Depends(get_db_session),
    guest_id: int = Depends(get_current_guest)
):
    """
    Enhanced workflow query using RAG for guest users.
    This endpoint allows querying documents using semantic search.
    """
    try:
        # Search for relevant document chunks
        results = await query_document_vectors(
            query=query,
            owner_type=OwnerType.GUEST.value,
            owner_id=guest_id,
            document_id=document_id,
            limit=limit
        )
        
        # Include public documents if guest has no documents or few results
        if len(results) < limit:
            public_results = await query_document_vectors(
                query=query,
                # Public documents are marked with is_public=True
                # They are typically owned by employees
                limit=limit - len(results)
            )
            
            # Add public results to list
            results.extend(public_results)
        
        return {
            "query": query,
            "session_id": session_id,
            "results": results,
            "document_count": len(set([r["metadata"].get("document_id") for r in results])),
            "total_results": len(results)
        }
        
    except Exception as e:
        logger.error(f"Error in RAG guest query: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing RAG query: {str(e)}"
        )

# Customer RAG workflow endpoint
@router.post("/customer/query", response_model=Dict[str, Any])
async def rag_customer_query(
    query: str,
    session_id: Optional[str] = None,
    document_id: Optional[int] = None,
    limit: int = Query(5, gt=0, le=10),
    include_public: bool = Query(True),
    db: AsyncSession = Depends(get_db_session),
    customer_id: int = Depends(get_current_customer)
):
    """
    Enhanced workflow query using RAG for customer users.
    This endpoint allows querying documents using semantic search.
    Customers can search their own documents and optionally public documents.
    """
    try:
        # Search for relevant document chunks in customer's documents
        results = await query_document_vectors(
            query=query,
            owner_type=OwnerType.CUSTOMER.value,
            owner_id=customer_id,
            document_id=document_id,
            limit=limit
        )
        
        # Include public documents if requested and if we have room for more results
        if include_public and len(results) < limit:
            public_results = await query_document_vectors(
                query=query,
                # Public documents are marked with is_public=True
                limit=limit - len(results)
            )
            
            # Add public results to list
            results.extend(public_results)
        
        return {
            "query": query,
            "session_id": session_id,
            "results": results,
            "document_count": len(set([r["metadata"].get("document_id") for r in results])),
            "total_results": len(results)
        }
        
    except Exception as e:
        logger.error(f"Error in RAG customer query: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing RAG query: {str(e)}"
        )

# Employee RAG workflow endpoint  
@router.post("/employee/query", response_model=Dict[str, Any])
async def rag_employee_query(
    query: str,
    session_id: Optional[str] = None,
    document_id: Optional[int] = None,
    owner_type: Optional[str] = None,
    owner_id: Optional[int] = None,
    limit: int = Query(5, gt=0, le=10),
    db: AsyncSession = Depends(get_db_session),
    employee_id: int = Depends(get_current_employee)
):
    """
    Enhanced workflow query using RAG for employee users.
    This endpoint allows querying documents using semantic search.
    Employees can search specific documents, owner documents, or all documents.
    """
    try:
        # Search for relevant document chunks based on parameters
        results = await query_document_vectors(
            query=query,
            owner_type=owner_type,
            owner_id=owner_id,
            document_id=document_id,
            limit=limit
        )
        
        return {
            "query": query,
            "session_id": session_id,
            "results": results,
            "document_count": len(set([r["metadata"].get("document_id") for r in results])),
            "total_results": len(results)
        }
        
    except Exception as e:
        logger.error(f"Error in RAG employee query: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing RAG query: {str(e)}"
        )
