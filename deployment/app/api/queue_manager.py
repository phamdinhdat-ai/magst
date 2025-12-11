import asyncio
import time
import uuid
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, AsyncGenerator, Callable, Awaitable

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("employee_queue_manager")

class RequestStatus(Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class QueuedRequest:
    request_id: str
    request_data: dict
    timestamp: float = field(default_factory=time.time)
    status: RequestStatus = RequestStatus.QUEUED
    future: asyncio.Future = field(default_factory=asyncio.Future)
    start_time: Optional[float] = None
    end_time: Optional[float] = None

class WorkflowRequestQueue:
    def __init__(self, max_concurrent: int = 3, max_queue_size: int = 50):
        self.max_concurrent = max_concurrent
        self.max_queue_size = max_queue_size
        self.queue = asyncio.Queue(maxsize=max_queue_size)
        self.active_requests = {}
        self.request_history = {}
        self.stats = {
            'total_requests': 0,
            'completed_requests': 0,
            'failed_requests': 0,
            'current_queue_size': 0,
            'active_requests': 0
        }
        self.workflow_manager = None
        self._workers_started = False
    
    async def start_workers(self, workflow_manager):
        """Start background workers to process queued requests"""
        if self._workers_started:
            return
        
        self.workflow_manager = workflow_manager
        self._workers_started = True
        
        # Start worker tasks
        for i in range(self.max_concurrent):
            asyncio.create_task(self._worker(f"employee-worker-{i}"))
        
        logger.info(f"Started {self.max_concurrent} employee request processing workers")
    
    async def _worker(self, worker_id: str):
        """Background worker to process requests from the queue"""
        logger.info(f"Employee request worker {worker_id} started")
        
        while True:
            try:
                # Get request from queue
                queued_request: QueuedRequest = await self.queue.get()
                
                if queued_request is None:
                    # Handle shutdown signal
                    logger.info(f"Worker {worker_id} received shutdown signal")
                    break
                
                logger.info(f"Worker {worker_id} processing request {queued_request.request_id}")
                
                # Update status and tracking
                queued_request.status = RequestStatus.PROCESSING
                queued_request.start_time = time.time()
                self.active_requests[queued_request.request_id] = queued_request
                self.stats['active_requests'] += 1
                self.stats['current_queue_size'] = self.queue.qsize()
                
                # Process the request
                try:
                    await self._process_request(queued_request)
                    queued_request.status = RequestStatus.COMPLETED
                    self.stats['completed_requests'] += 1
                except Exception as e:
                    logger.error(f"Error processing request {queued_request.request_id}: {str(e)}", exc_info=True)
                    queued_request.status = RequestStatus.FAILED
                    self.stats['failed_requests'] += 1
                    # Set exception in future to propagate error to client
                    if not queued_request.future.done():
                        queued_request.future.set_exception(e)
                
                # Cleanup and record keeping
                queued_request.end_time = time.time()
                self.request_history[queued_request.request_id] = queued_request
                self.stats['active_requests'] -= 1
                if queued_request.request_id in self.active_requests:
                    del self.active_requests[queued_request.request_id]
                
                # Mark task as done
                self.queue.task_done()
                
            except Exception as e:
                logger.error(f"Unexpected error in worker {worker_id}: {str(e)}", exc_info=True)
                # Continue processing other requests even if one fails
                continue
    
    async def _process_request(self, queued_request: QueuedRequest):
        """Process a single employee workflow request"""
        request_data = queued_request.request_data
        
        if not self.workflow_manager:
            raise ValueError("Workflow manager not initialized")
        
        # Extract request parameters
        query = request_data.get('query')
        employee_id = request_data.get('employee_id')
        employee_role = request_data.get('employee_role')
        session_id = request_data.get('session_id')
        config = request_data.get('config', {})
        chat_history = request_data.get('chat_history', [])
        interaction_id = request_data.get('interaction_id')
        
        # Process with the appropriate workflow method
        if request_data.get('streaming', False):
            # Streaming requests get a generator that will be consumed by the client
            stream_generator = self.workflow_manager.astreaming_workflow(
                query=query,
                config=config,
                employee_id=employee_id,
                employee_role=employee_role,
                interaction_id=interaction_id,
                chat_history=chat_history
            )
            
            # Set the generator as the result
            if not queued_request.future.done():
                queued_request.future.set_result(stream_generator)
        else:
            # Non-streaming requests get the complete response
            result = await self.workflow_manager.process_with_load_balancing(
                query=query,
                employee_id=employee_id,
                employee_role=employee_role,
                session_id=session_id,
                prioritize=request_data.get('prioritize', False)
            )
            
            # Set the result in the future
            if not queued_request.future.done():
                queued_request.future.set_result(result)
    
    async def enqueue_request(self, request_data: dict, priority: bool = False) -> tuple[str, asyncio.Future]:
        """Add a request to the queue and return request ID and future"""
        if self.queue.full():
            raise ValueError("Request queue is full. Please try again later.")
        
        request_id = str(uuid.uuid4())
        queued_request = QueuedRequest(
            request_id=request_id,
            request_data=request_data
        )
        
        try:
            # Use put_nowait for priority requests to avoid potential deadlocks
            if priority:
                # For priority, we'd ideally have a priority queue, but as a simple implementation
                # we can just put it at the front by creating a new temporary queue
                temp_queue = asyncio.Queue(maxsize=self.max_queue_size)
                await temp_queue.put(queued_request)
                
                # Move items from original queue to temp queue
                while not self.queue.empty():
                    item = self.queue.get_nowait()
                    await temp_queue.put(item)
                
                # Replace queue
                self.queue = temp_queue
            else:
                await self.queue.put(queued_request)
                
            self.stats['total_requests'] += 1
            self.stats['current_queue_size'] = self.queue.qsize()
            
            logger.info(f"Enqueued request {request_id}, queue size: {self.queue.qsize()}, priority: {priority}")
            return request_id, queued_request.future
            
        except asyncio.QueueFull:
            raise ValueError("Request queue is full. Please try again later.")
    
    def get_request_status(self, request_id: str) -> dict:
        """Get status of a specific request"""
        if request_id in self.active_requests:
            req = self.active_requests[request_id]
            return {
                "request_id": request_id,
                "status": req.status.value,
                "queued_at": req.timestamp,
                "started_at": req.start_time,
                "position_in_queue": None  # Would require more complex queue structure to implement
            }
        elif request_id in self.request_history:
            req = self.request_history[request_id]
            return {
                "request_id": request_id,
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
            "active_requests": len(self.active_requests)
        }

# Create a global instance of the request queue
employee_request_queue = WorkflowRequestQueue(max_concurrent=3, max_queue_size=50)
