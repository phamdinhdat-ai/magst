"""
Enhanced health endpoint implementations for the GenStory API server.

This module provides robust health check endpoints that report on:
- Overall server status
- Available workflows and features
- System information
- Configuration status
"""

import os
import platform
import psutil
import time
import logging
from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Any, List, Optional
from sqlalchemy import text

from app.core.config import settings
from app.db.session import AsyncSessionLocal

# Configure logger
logger = logging.getLogger(__name__)

health_router = APIRouter()

@health_router.get("/health", summary="Get server health status")
async def health() -> Dict[str, Any]:
    """
    Get detailed server health status.
    
    Returns:
        Dict with health information including server status, 
        available features, and system information.
    """
    # Check for enabled workflows
    available_features = []
    
    # Use getattr with default value True to handle missing settings
    if getattr(settings, "ENABLE_GUEST_WORKFLOW", True):
        available_features.append("guest_workflow")
    if getattr(settings, "ENABLE_CUSTOMER_WORKFLOW", True):
        available_features.append("customer_workflow")
    if getattr(settings, "ENABLE_EMPLOYEE_WORKFLOW", True):
        available_features.append("employee_workflow")
    if getattr(settings, "ENABLE_DOCUMENT_WORKFLOW", True):
        available_features.append("document_workflow")
    
    # Basic system metrics
    process = psutil.Process(os.getpid())
    
    return {
        "status": "healthy",
        "service": "genstory-api",
        "timestamp": time.time(),
        "features": available_features,
        "system": {
            "version": getattr(settings, "API_VERSION", "1.0.0"),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "memory_usage": {
                "percent": process.memory_percent(),
                "rss": process.memory_info().rss / (1024 * 1024),  # MB
            },
            "cpu_percent": process.cpu_percent(),
        }
    }

@health_router.get("/", summary="API information and status")
async def root() -> Dict[str, Any]:
    """
    API root endpoint providing information about the server and available endpoints.
    
    Returns:
        Dict with API information including name, version, description,
        and available endpoints.
    """
    # Build a map of available endpoints
    endpoints = {
        "guest": {
            "workflow": "/api/v1/guest/workflow",
            "streaming": "/api/v1/guest/workflow/stream",
            "health": "/api/v1/guest/ping",
        } if getattr(settings, "ENABLE_GUEST_WORKFLOW", True) else {"status": "disabled"},
        
        "customer": {
            "workflow": "/api/v1/customer/workflow",
            "streaming": "/api/v1/customer/workflow/stream",
            "health": "/api/v1/customer/ping",
        } if getattr(settings, "ENABLE_CUSTOMER_WORKFLOW", True) else {"status": "disabled"},
        
        "employee": {
            "workflow": "/api/v1/employee/workflow",
            "streaming": "/api/v1/employee/workflow/stream", 
            "health": "/api/v1/employee/ping",
        } if getattr(settings, "ENABLE_EMPLOYEE_WORKFLOW", True) else {"status": "disabled"},
        
        "document": {
            "upload": "/api/v1/document/upload",
            "query": "/api/v1/document/query",
            "health": "/api/v1/document/ping",
        } if getattr(settings, "ENABLE_DOCUMENT_WORKFLOW", True) else {"status": "disabled"},
        
        "health": {
            "root": "/health",
            "api": "/api/v1/health"
        }
    }
    
    return {
        "name": "GenStory API",
        "version": getattr(settings, "API_VERSION", "1.0.0"),
        "description": "GenStory API for pharmacogenomics and health information workflows",
        "documentation": "/docs",
        "endpoints": endpoints
    }

@health_router.get("/api/v1/health", summary="API v1 health status")
async def api_health() -> Dict[str, Any]:
    """
    Check the health of API v1.
    
    Returns:
        Dict with health information for API v1.
    """
    # Try to connect to the database
    db_healthy = True
    try:
        # Create a session
        async_session = AsyncSessionLocal()
        async with async_session as session:
            # Execute a simple query
            await session.execute(text("SELECT 1"))
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        db_healthy = False
    
    return {
        "status": "healthy",
        "api_version": getattr(settings, "API_VERSION", "1.0.0"),
        "database": "healthy" if db_healthy else "unhealthy",
        "timestamp": time.time(),
    }

# Individual workflow health endpoints

@health_router.get("/api/v1/guest/ping", summary="Guest workflow health check")
async def guest_health() -> Dict[str, Any]:
    """
    Check the health of the guest workflow.
    
    Returns:
        Dict with health information for the guest workflow.
    """
    if not getattr(settings, "ENABLE_GUEST_WORKFLOW", True):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Guest workflow is disabled"
        )
    
    return {
        "status": "healthy",
        "workflow": "guest",
        "timestamp": time.time()
    }

@health_router.get("/api/v1/customer/ping", summary="Customer workflow health check")
async def customer_health() -> Dict[str, Any]:
    """
    Check the health of the customer workflow.
    
    Returns:
        Dict with health information for the customer workflow.
    """
    if not getattr(settings, "ENABLE_CUSTOMER_WORKFLOW", True):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Customer workflow is disabled"
        )
    
    return {
        "status": "healthy",
        "workflow": "customer",
        "timestamp": time.time()
    }

@health_router.get("/api/v1/employee/ping", summary="Employee workflow health check")
async def employee_health() -> Dict[str, Any]:
    """
    Check the health of the employee workflow.
    
    Returns:
        Dict with health information for the employee workflow.
    """
    if not getattr(settings, "ENABLE_EMPLOYEE_WORKFLOW", True):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Employee workflow is disabled"
        )
    
    return {
        "status": "healthy",
        "workflow": "employee",
        "timestamp": time.time()
    }

@health_router.get("/api/v1/document/ping", summary="Document workflow health check")
async def document_health() -> Dict[str, Any]:
    """
    Check the health of the document workflow.
    
    Returns:
        Dict with health information for the document workflow.
    """
    if not getattr(settings, "ENABLE_DOCUMENT_WORKFLOW", True):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Document workflow is disabled"
        )
    
    return {
        "status": "healthy",
        "workflow": "document",
        "timestamp": time.time()
    }

# Export the router to be included in the main app
def get_health_router() -> APIRouter:
    """
    Get the health router for inclusion in the main app.
    
    Returns:
        APIRouter: The health router.
    """
    return health_router
