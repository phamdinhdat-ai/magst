"""
Request Queue Management for Guest API
This module provides a queue system for handling guest requests,
similar to the one in main.py but adapted for the guest workflow.
"""
import asyncio
import time
import uuid
from typing import Dict, Any, Optional, Tuple, AsyncGenerator
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from loguru import logger
from fastapi import HTTPException

class RequestStatus(Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"

@dataclass
class QueuedGuestRequest:
    request_id: str
    guest_id: str
    query: str
    session_id: Optional[str]
    chat_history: Optional[list]
    timestamp: float = field(default_factory=time.time)
    status: RequestStatus = RequestStatus.QUEUED
    future: asyncio.Future = field(default_factory=asyncio.Future)
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    priority: bool = False  # Whether this is a priority request

class GuestRequestQueue:
    """
    Queue system for handling guest requests to prevent overload
    and provide fair scheduling.
    """
    def __init__(self, max_concurrent: int = 3, max_queue_size: int = 50, request_timeout: int = 60):
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
            'active_requests': 0
        }
        self._workers_started = False
        self._guest_workflow = None
        
        logger.info(f"Guest Request Queue initialized with max_concurrent={max_concurrent}, max_queue_size={max_queue_size}")
    
    def set_workflow(self, workflow):
        """Set the guest workflow instance for processing requests"""
        self._guest_workflow = workflow
        logger.info("Guest workflow set for request queue")
    
    async def start_workers(self):
        """Start background workers to process queued requests"""
        if self._workers_started or not self._guest_workflow:
            return
        
        self._workers_started = True
        
        # Start worker tasks
        for i in range(self.max_concurrent):
            asyncio.create_task(self._worker(f"worker-{i}"))
        
        logger.info(f"Started {self.max_concurrent} guest request processing workers")
    
    async def _worker(self, worker_id: str):
        """Background worker to process requests from the queue"""
        logger.info(f"Guest request worker {worker_id} started")
        
        while True:
            try:
                # Get request from queue
                queued_request: QueuedGuestRequest = await self.queue.get()
                
                if queued_request is None:  # Shutdown signal
                    logger.info(f"Worker {worker_id} received shutdown signal")
                    break
                
                logger.info(f"Worker {worker_id} processing request {queued_request.request_id} for guest {queued_request.guest_id}")
                
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
    
    async def _check_timeout(self, request: QueuedGuestRequest, timeout_sec: int):
        """Check if a request has timed out and handle it"""
        try:
            await asyncio.sleep(timeout_sec)
            # If we get here, the request has timed out
            if request.status == RequestStatus.PROCESSING:
                logger.warning(f"Request {request.request_id} timed out after {timeout_sec}s")
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
    
    def _create_timeout_event(self, request: QueuedGuestRequest) -> AsyncGenerator:
        """Create a timeout event generator for a request that has timed out"""
        async def timeout_generator():
            # Yield a timeout notification event
            yield {
                "event": "timeout",
                "data": {
                    "message": f"Request timed out after {self.request_timeout} seconds",
                    "request_id": request.request_id,
                    "guest_id": request.guest_id,
                    "query": request.query[:50] + "..." if len(request.query) > 50 else request.query
                },
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                }
            }
            
            # Generate a simple fallback response
            yield {
                "event": "answer_chunk",
                "data": "I'm sorry, but I'm currently experiencing high demand and couldn't process your request in time. Please try again in a moment.",
                "metadata": {
                    "is_fallback": True,
                    "timestamp": datetime.now().isoformat(),
                }
            }
            
            # Final event
            yield {
                "event": "workflow_complete",
                "data": {
                    "status": "timeout",
                    "request_id": request.request_id,
                    "guest_id": request.guest_id,
                    "processing_time_ms": int((time.time() - request.start_time) * 1000) if request.start_time else 0,
                },
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                }
            }
        
        return timeout_generator()
    
    async def _process_request(self, queued_request: QueuedGuestRequest):
        """Process a single guest request through the workflow"""
        if not self._guest_workflow:
            raise ValueError("Guest workflow not set. Cannot process request.")
        
        config = {"configurable": {"thread_id": queued_request.session_id}}
        
        # Get the streaming generator from workflow
        results_generator = self._guest_workflow.astreaming_workflow(
            queued_request.query, 
            config,
            guest_id=queued_request.guest_id,
            chat_history=queued_request.chat_history
        )
        
        # Set the result in the future
        queued_request.future.set_result(results_generator)
    
    async def enqueue_request(self, 
                             query: str, 
                             guest_id: str,
                             session_id: Optional[str] = None,
                             chat_history: Optional[list] = None,
                             priority: bool = False) -> Tuple[str, asyncio.Future]:
        """
        Add a request to the queue and return request ID and future
        
        Args:
            query: The user's query string
            guest_id: ID of the guest user
            session_id: Current session ID 
            chat_history: Optional chat history for context
            priority: Whether this request should be prioritized
            
        Returns:
            Tuple of request ID and future that will contain the result generator
        """
        if self.queue.full():
            raise HTTPException(status_code=429, detail="Request queue is full. Please try again later.")
        
        request_id = str(uuid.uuid4())
        queued_request = QueuedGuestRequest(
            request_id=request_id,
            guest_id=guest_id,
            query=query,
            session_id=session_id,
            chat_history=chat_history,
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
                logger.info(f"Priority request {request_id} added to front of queue")
            else:
                await self.queue.put(queued_request)
            
            self.stats['total_requests'] += 1
            self.stats['current_queue_size'] = self.queue.qsize()
            
            logger.info(f"Enqueued guest request {request_id}, queue size: {self.queue.qsize()}, priority: {priority}")
            return request_id, queued_request.future
            
        except asyncio.QueueFull:
            raise HTTPException(status_code=429, detail="Request queue is full. Please try again later.")
    
    def get_request_status(self, request_id: str) -> dict:
        """Get status of a specific request"""
        if request_id in self.active_requests:
            req = self.active_requests[request_id]
            position = -1  # Not known precisely without scanning the queue
            
            return {
                "request_id": request_id,
                "guest_id": req.guest_id,
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
                "guest_id": req.guest_id,
                "status": req.status.value,
                "queued_at": req.timestamp,
                "started_at": req.start_time,
                "completed_at": req.end_time,
                "processing_time": (req.end_time - req.start_time) if req.end_time and req.start_time else None
            }
        else:
            return {"error": "Request not found"}
    
    def get_queue_stats(self) -> dict:
        """Get current queue statistics"""
        return {
            **self.stats,
            "current_queue_size": self.queue.qsize(),
            "active_requests": len(self.active_requests),
            "timestamp": datetime.now().isoformat()
        }
    
    async def cancel_request(self, request_id: str) -> bool:
        """
        Cancel a request if it's still in the queue or not yet processing
        
        Args:
            request_id: The ID of the request to cancel
            
        Returns:
            bool: True if request was cancelled, False otherwise
        """
        # Check if request is active
        if request_id in self.active_requests:
            req = self.active_requests[request_id]
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
            
            logger.info(f"Request {request_id} was cancelled successfully")
            return True
            
        logger.warning(f"Request {request_id} not found or already completed, cannot cancel")
        return False
    
    async def shutdown(self):
        """Shutdown the queue by cancelling all active requests"""
        logger.info("Shutting down guest request queue...")
        
        # Signal all workers to stop
        for _ in range(self.max_concurrent):
            await self.queue.put(None)
            
        # Cancel all active requests
        for req_id, request in list(self.active_requests.items()):
            if not request.future.done():
                request.future.cancel()
                
        logger.info("Guest request queue shutdown complete")
