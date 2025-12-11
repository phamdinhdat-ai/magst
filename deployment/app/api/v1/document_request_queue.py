"""
Request Queue Management for Document API
This module provides a queue system for handling document uploads and processing,
similar to what we've implemented for customer and employee workflows.
"""
import asyncio
import time
import uuid
import os
from typing import Dict, Any, Optional, Tuple, AsyncGenerator
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from loguru import logger
from fastapi import HTTPException, status
from pathlib import Path

class RequestStatus(Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"

class OwnerType(Enum):
    CUSTOMER = "customer"
    EMPLOYEE = "employee"
    GUEST = "guest"
    SYSTEM = "system"

@dataclass
class QueuedDocumentRequest:
    request_id: str
    owner_id: str
    owner_type: OwnerType
    document_id: Optional[int]
    file_path: str
    file_size: int
    file_name: str
    content_type: str
    timestamp: float = field(default_factory=time.time)
    status: RequestStatus = RequestStatus.QUEUED
    future: asyncio.Future = field(default_factory=asyncio.Future)
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    priority: bool = False  # Whether this is a priority request

class DocumentRequestQueue:
    """
    Queue system for handling document upload and processing requests to prevent overload
    and provide fair scheduling.
    """
    def __init__(self, max_concurrent: int = 2, max_queue_size: int = 20, request_timeout: int = 300):
        self.max_concurrent = max_concurrent
        self.max_queue_size = max_queue_size
        self.request_timeout = request_timeout  # seconds
        self.queue = asyncio.Queue(maxsize=max_queue_size)
        self.active_requests = {}
        self.request_history = {}
        self.stats = {
            'total_requests': 0,
            'completed_requests': 0,
            'failed_requests': 0,
            'timeout_requests': 0,
            'current_queue_size': 0,
            'active_requests': 0,
            'avg_processing_time_sec': 0,
            'processed_bytes_total': 0
        }
        self._workers_started = False
        self._document_processor = None
        
        logger.info(f"Document Request Queue initialized with max_concurrent={max_concurrent}, max_queue_size={max_queue_size}")
    
    def set_document_processor(self, processor):
        """Set the document processor instance for processing requests"""
        self._document_processor = processor
        logger.info("Document processor set for request queue")
        return self
    
    async def start_workers(self):
        """Start background workers to process queued requests"""
        if self._workers_started:
            return
        
        self._workers_started = True
        
        # Start worker tasks
        for i in range(self.max_concurrent):
            asyncio.create_task(self._worker(f"document-worker-{i}"))
        
        logger.info(f"Started {self.max_concurrent} document processing workers")
    
    async def _worker(self, worker_id: str):
        """Background worker to process requests from the queue"""
        logger.info(f"Document request worker {worker_id} started")
        
        while True:
            try:
                # Get request from queue
                queued_request: QueuedDocumentRequest = await self.queue.get()
                
                if queued_request is None:  # Shutdown signal
                    logger.info(f"Worker {worker_id} received shutdown signal")
                    break
                
                logger.info(f"Worker {worker_id} processing document request {queued_request.request_id} for {queued_request.owner_type.value} {queued_request.owner_id}")
                
                # Update status
                queued_request.status = RequestStatus.PROCESSING
                queued_request.start_time = time.time()
                self.active_requests[queued_request.request_id] = queued_request
                self.stats['active_requests'] = len(self.active_requests)
                self.stats['current_queue_size'] = self.queue.qsize()
                
                try:
                    # Set up timeout
                    timeout_task = asyncio.create_task(
                        self._check_timeout(queued_request, self.request_timeout)
                    )
                    
                    # Process the request
                    await self._process_request(queued_request)
                    
                    # Cancel timeout task since we're done
                    timeout_task.cancel()
                    
                    if queued_request.status != RequestStatus.TIMEOUT:
                        queued_request.status = RequestStatus.COMPLETED
                        self.stats['completed_requests'] += 1
                        self.stats['processed_bytes_total'] += queued_request.file_size
                    
                except asyncio.CancelledError:
                    logger.warning(f"Request {queued_request.request_id} was cancelled")
                    if queued_request.status != RequestStatus.TIMEOUT:
                        queued_request.status = RequestStatus.FAILED
                        self.stats['failed_requests'] += 1
                
                except Exception as e:
                    logger.error(f"Worker {worker_id} failed to process request {queued_request.request_id}: {str(e)}", exc_info=True)
                    queued_request.status = RequestStatus.FAILED
                    if not queued_request.future.done():
                        queued_request.future.set_exception(e)
                    self.stats['failed_requests'] += 1
                
                finally:
                    # Cleanup
                    queued_request.end_time = time.time()
                    processing_time = queued_request.end_time - queued_request.start_time if queued_request.start_time else 0
                    
                    # Update average processing time
                    total_completed = self.stats['completed_requests']
                    if total_completed > 0:
                        current_avg = self.stats['avg_processing_time_sec']
                        self.stats['avg_processing_time_sec'] = (current_avg * (total_completed - 1) + processing_time) / total_completed
                    
                    self.active_requests.pop(queued_request.request_id, None)
                    self.request_history[queued_request.request_id] = queued_request
                    self.stats['active_requests'] = len(self.active_requests)
                    self.queue.task_done()
                    
                    # Keep only last 100 requests in history
                    if len(self.request_history) > 100:
                        oldest_key = min(self.request_history.keys(), 
                                      key=lambda k: self.request_history[k].timestamp)
                        del self.request_history[oldest_key]
                        
            except Exception as e:
                logger.error(f"Worker {worker_id} encountered error: {str(e)}", exc_info=True)
                await asyncio.sleep(1)  # Brief pause before retrying
    
    async def _check_timeout(self, request: QueuedDocumentRequest, timeout_sec: int):
        """Check if a request has timed out and handle it"""
        try:
            await asyncio.sleep(timeout_sec)
            # If we get here, the request has timed out
            if request.status == RequestStatus.PROCESSING:
                logger.warning(f"Document request {request.request_id} timed out after {timeout_sec}s")
                request.status = RequestStatus.TIMEOUT
                self.stats['timeout_requests'] += 1
                
                # Create a timeout event and pass it to the future if not already done
                if not request.future.done():
                    timeout_event = self._create_timeout_event(request)
                    request.future.set_result(timeout_event)
        except asyncio.CancelledError:
            # Normal cancellation when request completes before timeout
            pass
        except Exception as e:
            logger.error(f"Error in timeout handler: {str(e)}", exc_info=True)
    
    def _create_timeout_event(self, request: QueuedDocumentRequest) -> Dict:
        """Create a timeout event for a request that has timed out"""
        return {
            "status": "timeout",
            "message": f"Document processing request timed out after {self.request_timeout} seconds",
            "request_id": request.request_id,
            "owner_id": request.owner_id,
            "owner_type": request.owner_type.value,
            "file_name": request.file_name,
            "document_id": request.document_id,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _process_request(self, queued_request: QueuedDocumentRequest):
        """Process a single document request"""
        if not self._document_processor:
            # Use built-in default processor if no custom processor is set
            result = await self._default_document_processor(queued_request)
        else:
            # Use the custom document processor
            result = await self._document_processor.process_document(
                document_id=queued_request.document_id,
                file_path=queued_request.file_path,
                owner_id=queued_request.owner_id,
                owner_type=queued_request.owner_type.value,
                metadata=queued_request.metadata
            )
        
        # Set the result in the future
        if not queued_request.future.done():
            queued_request.future.set_result(result)
    
    async def _default_document_processor(self, request: QueuedDocumentRequest) -> Dict:
        """Default document processing if no custom processor is set"""
        try:
            # Simple file validation and move from temp to storage
            file_path = Path(request.file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"Document file not found: {file_path}")
            
            # Validate file exists and size matches
            actual_size = file_path.stat().st_size
            if actual_size != request.file_size:
                logger.warning(f"File size mismatch: expected {request.file_size}, got {actual_size}")
            
            # Create destination directory if it doesn't exist
            owner_dir = f"storage/{request.owner_type.value}_{request.owner_id}"
            os.makedirs(owner_dir, exist_ok=True)
            
            # Move file to permanent storage
            dest_path = f"{owner_dir}/{request.file_name}"
            
            # If a file with same name exists, add timestamp
            if os.path.exists(dest_path):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename, ext = os.path.splitext(request.file_name)
                dest_path = f"{owner_dir}/{filename}_{timestamp}{ext}"
            
            # Simulate processing time based on file size
            await asyncio.sleep(min(5, request.file_size / (1024 * 1024)))
            
            # Move the file
            # Since this is just a default implementation, we'll just return success without actually moving
            return {
                "status": "success",
                "document_id": request.document_id,
                "original_path": str(file_path),
                "destination_path": dest_path,
                "processing_time_sec": time.time() - request.start_time,
                "message": "Document processed successfully"
            }
            
        except Exception as e:
            logger.error(f"Error in default document processor: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "document_id": request.document_id,
                "message": str(e),
                "error": str(e)
            }
    
    async def enqueue_request(self, 
                             owner_id: str,
                             owner_type: OwnerType,
                             document_id: int,
                             file_path: str,
                             file_size: int,
                             file_name: str,
                             content_type: str,
                             metadata: Optional[Dict] = None,
                             priority: bool = False) -> Tuple[str, asyncio.Future]:
        """
        Add a document request to the queue and return request ID and future
        
        Args:
            owner_id: ID of the owner (customer/employee/guest)
            owner_type: Type of the owner
            document_id: ID of the document in the database
            file_path: Path to the temporary file
            file_size: Size of the file in bytes
            file_name: Original filename
            content_type: MIME type
            metadata: Additional metadata for the document
            priority: Whether this request should be prioritized
            
        Returns:
            Tuple of request ID and future that will contain the result
        """
        if self.queue.full():
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, 
                               detail="Document processing queue is full. Please try again later.")
        
        request_id = str(uuid.uuid4())
        queued_request = QueuedDocumentRequest(
            request_id=request_id,
            owner_id=owner_id,
            owner_type=owner_type,
            document_id=document_id,
            file_path=file_path,
            file_size=file_size,
            file_name=file_name,
            content_type=content_type,
            metadata=metadata or {},
            priority=priority
        )
        
        try:
            # Handle priority requests by putting them at the front of the queue
            if priority:
                # For priority, create a new temporary queue with this request first
                temp_queue = asyncio.Queue(maxsize=self.max_queue_size)
                await temp_queue.put(queued_request)
                
                # Move items from original queue to temp queue
                while not self.queue.empty():
                    item = self.queue.get_nowait()
                    await temp_queue.put(item)
                
                # Replace queue
                self.queue = temp_queue
                logger.info(f"Priority document request {request_id} added to front of queue")
            else:
                await self.queue.put(queued_request)
                
            self.stats['total_requests'] += 1
            self.stats['current_queue_size'] = self.queue.qsize()
            
            logger.info(f"Enqueued document request {request_id}, queue size: {self.queue.qsize()}, priority: {priority}")
            return request_id, queued_request.future
            
        except asyncio.QueueFull:
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, 
                              detail="Document processing queue is full. Please try again later.")
    
    def get_request_status(self, request_id: str) -> dict:
        """Get status of a specific request"""
        if request_id in self.active_requests:
            req = self.active_requests[request_id]
            position = -1  # Not known precisely without scanning the queue
            
            return {
                "request_id": request_id,
                "owner_id": req.owner_id,
                "owner_type": req.owner_type.value,
                "document_id": req.document_id,
                "file_name": req.file_name,
                "file_size": req.file_size,
                "status": req.status.value,
                "queued_at": req.timestamp,
                "started_at": req.start_time,
                "position_in_queue": position,
                "elapsed_time": time.time() - (req.start_time or req.timestamp)
            }
        elif request_id in self.request_history:
            req = self.request_history[request_id]
            return {
                "request_id": request_id,
                "owner_id": req.owner_id,
                "owner_type": req.owner_type.value,
                "document_id": req.document_id,
                "file_name": req.file_name,
                "file_size": req.file_size,
                "status": req.status.value,
                "queued_at": req.timestamp,
                "started_at": req.start_time,
                "completed_at": req.end_time,
                "processing_time": (req.end_time - req.start_time) if req.end_time and req.start_time else None,
                "priority": req.priority
            }
        else:
            return {"error": "Document request not found"}
    
    def get_queue_stats(self) -> dict:
        """Get current queue statistics"""
        return {
            **self.stats,
            "current_queue_size": self.queue.qsize(),
            "active_requests": len(self.active_requests),
            "request_history_size": len(self.request_history),
            "workers_started": self._workers_started,
            "max_concurrent": self.max_concurrent,
            "max_queue_size": self.max_queue_size,
            "timestamp": datetime.now().isoformat()
        }
    
    async def cancel_request(self, request_id: str, owner_id: str = None, owner_type: str = None) -> bool:
        """
        Cancel a request if it's still in the queue
        
        Args:
            request_id: The ID of the request to cancel
            owner_id: Optional owner ID to verify ownership
            owner_type: Optional owner type to verify ownership
            
        Returns:
            bool: True if request was cancelled, False otherwise
        """
        # Check if request is active
        if request_id in self.active_requests:
            req = self.active_requests[request_id]
            
            # Check ownership if owner_id provided
            if owner_id and req.owner_id != str(owner_id):
                logger.warning(f"{owner_type} {owner_id} attempted to cancel request {request_id} that belongs to {req.owner_type.value} {req.owner_id}")
                return False
            
            # Can't cancel already processing requests
            if req.status == RequestStatus.PROCESSING:
                logger.warning(f"Request {request_id} is already processing and cannot be cancelled")
                return False
            
            # Mark as failed/canceled
            req.status = RequestStatus.FAILED
            req.end_time = time.time()
            
            # Set exception in future if not done
            if not req.future.done():
                error_msg = "Request cancelled by user or client"
                req.future.set_exception(asyncio.CancelledError(error_msg))
            
            # Move to history
            self.request_history[request_id] = req
            del self.active_requests[request_id]
            
            self.stats['failed_requests'] += 1
            
            logger.info(f"Document request {request_id} was cancelled successfully")
            return True
            
        logger.warning(f"Document request {request_id} not found or already completed, cannot cancel")
        return False
    
    async def shutdown(self):
        """Shutdown the queue by cancelling all active requests"""
        logger.info("Shutting down document request queue...")
        
        # Signal all workers to stop
        for _ in range(self.max_concurrent):
            await self.queue.put(None)
            
        # Cancel all active requests
        for req_id, request in list(self.active_requests.items()):
            if not request.future.done():
                request.future.cancel()
                
        logger.info("Document request queue shutdown complete")

# Initialize queue with default settings
# Configure based on environment or default values
MAX_CONCURRENT = 2  # Can be set from environment variable
MAX_QUEUE_SIZE = 20 # Can be set from environment variable
REQUEST_TIMEOUT = 300 # Can be set from environment variable (5 min for document processing)

# Create and configure the request queue
document_request_queue = DocumentRequestQueue(
    max_concurrent=MAX_CONCURRENT,
    max_queue_size=MAX_QUEUE_SIZE,
    request_timeout=REQUEST_TIMEOUT
)
