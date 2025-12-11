"""
Queue Manager for handling asynchronous task processing.
This provides a robust queue system that allows distributing workload across workers.
"""

import asyncio
import time
from typing import Dict, Any, Callable, Awaitable, List, Optional, Union
from loguru import logger
from pydantic import BaseModel
import uuid
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import functools
import signal
import os

class TaskPriority:
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3

class TaskStatus:
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class QueueTask(BaseModel):
    id: str
    function_name: str
    args: List[Any] = []
    kwargs: Dict[str, Any] = {}
    priority: int = TaskPriority.NORMAL
    status: str = TaskStatus.PENDING
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3

class QueueManager:
    """
    Manages task queues with priority support and distributed processing.
    """
    # Priority levels - also accessible as instance attributes
    PRIORITY_LOW = TaskPriority.LOW
    PRIORITY_NORMAL = TaskPriority.NORMAL
    PRIORITY_HIGH = TaskPriority.HIGH
    PRIORITY_CRITICAL = TaskPriority.CRITICAL
    
    def __init__(self, 
                 max_workers: int = None, 
                 max_queue_size: int = 1000,
                 health_check_interval: int = 60):
        # Default to CPU count for workers
        self.max_workers = max_workers or max(1, multiprocessing.cpu_count() - 1)
        self.max_queue_size = max_queue_size
        self.health_check_interval = health_check_interval
        
        # Task storage
        self.tasks: Dict[str, QueueTask] = {}
        self.priority_queues: Dict[int, asyncio.Queue] = {
            TaskPriority.LOW: asyncio.Queue(maxsize=max_queue_size),
            TaskPriority.NORMAL: asyncio.Queue(maxsize=max_queue_size),
            TaskPriority.HIGH: asyncio.Queue(maxsize=max_queue_size),
            TaskPriority.CRITICAL: asyncio.Queue(maxsize=max_queue_size)
        }
        
        # Processing pools
        self.process_pool = ProcessPoolExecutor(max_workers=self.max_workers)
        self.task_functions: Dict[str, Callable] = {}
        
        # Statistics
        self.stats = {
            "tasks_queued": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "tasks_processing": 0,
            "queue_lengths": {
                TaskPriority.LOW: 0,
                TaskPriority.NORMAL: 0, 
                TaskPriority.HIGH: 0,
                TaskPriority.CRITICAL: 0
            },
            "avg_processing_time": 0,
            "total_processing_time": 0
        }
        
        # Background tasks
        self.workers: List[asyncio.Task] = []
        self.is_running = False
        self.health_check_task = None
        
        logger.info(f"Queue Manager initialized with {self.max_workers} workers")

    def register_task(self, name: str, func: Callable) -> None:
        """Register a task function that can be executed by the queue"""
        self.task_functions[name] = func
        logger.info(f"Registered task function: {name}")

    async def enqueue(self, 
               function_name: str, 
               args: List[Any] = None, 
               kwargs: Dict[str, Any] = None,
               priority: int = TaskPriority.NORMAL) -> str:
        """Add a task to the queue with the given priority"""
        if function_name not in self.task_functions:
            raise ValueError(f"Task function '{function_name}' is not registered")
            
        if not self.is_running:
            raise RuntimeError("Queue manager is not running")
            
        if self.priority_queues[priority].full():
            raise RuntimeError(f"Queue at priority {priority} is full")
            
        task_id = str(uuid.uuid4())
        task = QueueTask(
            id=task_id,
            function_name=function_name,
            args=args or [],
            kwargs=kwargs or {},
            priority=priority,
            created_at=time.time()
        )
        
        self.tasks[task_id] = task
        await self.priority_queues[priority].put(task_id)
        
        self.stats["tasks_queued"] += 1
        self.stats["queue_lengths"][priority] += 1
        
        logger.debug(f"Enqueued task {task_id} with priority {priority}")
        return task_id

    async def add_task(self, 
               task_func: str, 
               args: Dict[str, Any] = None,
               priority: int = None) -> str:
        """
        Add a task to the queue (alias for enqueue with slightly different parameters)
        
        Args:
            task_func: The function name to execute
            args: Dictionary of arguments to pass to the function
            priority: Priority level (use class constants like PRIORITY_HIGH)
            
        Returns:
            task_id: ID of the created task
        """
        # Convert args dict to proper args and kwargs
        if args is None:
            args = {}
        
        # Use normal priority by default
        if priority is None:
            priority = self.PRIORITY_NORMAL
            
        # Call the main enqueue method
        return await self.enqueue(
            function_name=task_func,
            args=[],
            kwargs=args,
            priority=priority
        )
    
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get the current status of a task"""
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")
            
        task = self.tasks[task_id]
        return {
            "id": task.id,
            "status": task.status,
            "created_at": task.created_at,
            "started_at": task.started_at,
            "completed_at": task.completed_at,
            "result": task.result,
            "error": task.error
        }

    async def cancel_task(self, task_id: str) -> bool:
        """Try to cancel a pending task"""
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")
            
        task = self.tasks[task_id]
        if task.status == TaskStatus.PENDING:
            task.status = TaskStatus.CANCELLED
            logger.info(f"Cancelled task {task_id}")
            return True
        else:
            logger.warning(f"Cannot cancel task {task_id} with status {task.status}")
            return False

    async def start(self) -> None:
        """Start the queue manager and worker processes"""
        if self.is_running:
            return
            
        self.is_running = True
        
        # Start worker tasks for each priority level
        for priority in self.priority_queues.keys():
            for _ in range(max(1, self.max_workers // len(self.priority_queues))):
                worker = asyncio.create_task(self._worker(priority))
                self.workers.append(worker)
        
        # Start health check task
        self.health_check_task = asyncio.create_task(self._health_check())
        
        logger.info(f"Started {len(self.workers)} worker tasks")

    async def stop(self) -> None:
        """Gracefully stop the queue manager"""
        if not self.is_running:
            return
            
        self.is_running = False
        
        # Cancel all worker tasks
        for worker in self.workers:
            worker.cancel()
            
        # Cancel health check task
        if self.health_check_task:
            self.health_check_task.cancel()
            
        # Wait for all tasks to complete
        await asyncio.gather(*self.workers, return_exceptions=True)
        
        # Shutdown process pool
        
    def is_active(self) -> bool:
        """Check if the queue manager is currently running"""
        return self.is_running
        self.process_pool.shutdown(wait=True)
        
        logger.info("Queue manager stopped")

    async def _worker(self, priority: int) -> None:
        """Worker coroutine that processes tasks from a specific priority queue"""
        queue = self.priority_queues[priority]
        
        while self.is_running:
            try:
                # Get next task from queue
                task_id = await queue.get()
                task = self.tasks.get(task_id)
                
                if not task or task.status == TaskStatus.CANCELLED:
                    queue.task_done()
                    continue
                    
                # Update task and stats
                task.status = TaskStatus.PROCESSING
                task.started_at = time.time()
                self.stats["tasks_processing"] += 1
                self.stats["queue_lengths"][priority] -= 1
                
                # Execute task
                function = self.task_functions[task.function_name]
                try:
                    # If the function is async, await it, otherwise run in process pool
                    if asyncio.iscoroutinefunction(function):
                        result = await function(*task.args, **task.kwargs)
                    else:
                        loop = asyncio.get_running_loop()
                        wrapped = functools.partial(function, *task.args, **task.kwargs)
                        result = await loop.run_in_executor(self.process_pool, wrapped)
                        
                    # Update task with success
                    task.status = TaskStatus.COMPLETED
                    task.result = result
                    task.completed_at = time.time()
                    
                    # Update statistics
                    processing_time = task.completed_at - task.started_at
                    self.stats["tasks_completed"] += 1
                    self.stats["total_processing_time"] += processing_time
                    self.stats["avg_processing_time"] = (
                        self.stats["total_processing_time"] / self.stats["tasks_completed"]
                    )
                    
                    logger.debug(f"Task {task_id} completed in {processing_time:.2f}s")
                    
                except Exception as e:
                    # Update task with failure
                    task.error = str(e)
                    task.retry_count += 1
                    
                    if task.retry_count <= task.max_retries:
                        # Re-queue the task
                        task.status = TaskStatus.PENDING
                        await queue.put(task_id)
                        logger.warning(f"Task {task_id} failed, retrying ({task.retry_count}/{task.max_retries}): {e}")
                    else:
                        task.status = TaskStatus.FAILED
                        task.completed_at = time.time()
                        self.stats["tasks_failed"] += 1
                        logger.error(f"Task {task_id} failed after {task.retry_count} attempts: {e}")
                
                finally:
                    self.stats["tasks_processing"] -= 1
                    queue.task_done()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker error: {e}")
                await asyncio.sleep(1)  # Avoid tight loop on repeated errors

    async def _health_check(self) -> None:
        """Periodically check queue health and log stats"""
        while self.is_running:
            try:
                await asyncio.sleep(self.health_check_interval)
                
                total_queued = sum(self.stats["queue_lengths"].values())
                logger.info(
                    f"Queue health: {total_queued} queued, "
                    f"{self.stats['tasks_processing']} processing, "
                    f"{self.stats['tasks_completed']} completed, "
                    f"{self.stats['tasks_failed']} failed, "
                    f"avg processing time: {self.stats['avg_processing_time']:.2f}s"
                )
                
                # Check for bottlenecks
                if total_queued > self.max_queue_size * 0.8:
                    logger.warning(f"Queue is nearing capacity: {total_queued}/{self.max_queue_size}")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics about the queue"""
        return self.stats

    async def wait_for_task(self, task_id: str, timeout: float = 60.0, polling_interval: float = 0.2) -> Optional[Dict[str, Any]]:
        """
        Wait for a task to complete with timeout
        
        Args:
            task_id: ID of the task to wait for
            timeout: Maximum time to wait in seconds
            polling_interval: How frequently to check task status
            
        Returns:
            Task result or None if timed out
        """
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")
            
        start_time = time.time()
        while time.time() - start_time < timeout:
            task = self.tasks[task_id]
            if task.status == TaskStatus.COMPLETED:
                return task.result
            elif task.status in [TaskStatus.FAILED, TaskStatus.CANCELLED]:
                logger.error(f"Task {task_id} ended with status {task.status}: {task.error}")
                return {"status": "error", "message": str(task.error), "task_id": task_id}
                
            # Wait before checking again
            await asyncio.sleep(polling_interval)
            
        # If we get here, we timed out
        logger.warning(f"Timed out waiting for task {task_id}")
        return None
    
    async def get_best_worker_node(self) -> Optional[str]:
        """
        Get the best worker node for executing a task.
        
        This is a placeholder implementation that simply returns None.
        In a real system, this could use heuristics or load information
        to select an appropriate worker node.
        
        Returns:
            worker_id: ID of the selected worker, or None if no suitable worker found
        """
        return None

# Global queue manager instance
queue_manager = None

def init_queue_manager(max_workers: int = None, max_queue_size: int = 1000) -> QueueManager:
    """Initialize the global queue manager"""
    global queue_manager
    if queue_manager is None:
        queue_manager = QueueManager(max_workers=max_workers, max_queue_size=max_queue_size)
    return queue_manager

def get_queue_manager() -> QueueManager:
    """Get the global queue manager instance"""
    if queue_manager is None:
        raise RuntimeError("Queue manager not initialized")
    return queue_manager

# Process pool management for multiprocessing tasks
class ProcessManager:
    """
    Manages a pool of processes for CPU-intensive tasks.
    Uses the multiprocessing module to distribute work across cores.
    """
    def __init__(self, 
                 num_processes: int = None, 
                 init_func: Callable = None,
                 init_args: tuple = None):
        self.num_processes = num_processes or max(1, multiprocessing.cpu_count() - 1)
        self.init_func = init_func
        self.init_args = init_args or ()
        self.pool = None
        logger.info(f"Process Manager initialized with {self.num_processes} processes")
        
    def start(self):
        """Start the process pool"""
        if self.pool is not None:
            return
            
        # Set up process start method
        multiprocessing.set_start_method('spawn', force=True)
        
        # Create process pool
        self.pool = ProcessPoolExecutor(
            max_workers=self.num_processes, 
            initializer=self.init_func,
            initargs=self.init_args
        )
        logger.info(f"Process pool started with {self.num_processes} workers")
        
    def stop(self):
        """Stop the process pool"""
        if self.pool is None:
            return
            
        self.pool.shutdown(wait=True)
        self.pool = None
        logger.info("Process pool stopped")
        
    async def execute(self, func, *args, **kwargs):
        """Execute a function in the process pool"""
        if self.pool is None:
            raise RuntimeError("Process pool not started")
            
        loop = asyncio.get_running_loop()
        wrapped = functools.partial(func, *args, **kwargs)
        return await loop.run_in_executor(self.pool, wrapped)

# Global process manager instance
process_manager = None

def init_process_manager(num_processes: int = None, 
                         init_func: Callable = None, 
                         init_args: tuple = None) -> ProcessManager:
    """Initialize the global process manager"""
    global process_manager
    if process_manager is None:
        process_manager = ProcessManager(
            num_processes=num_processes,
            init_func=init_func,
            init_args=init_args
        )
    return process_manager

def get_process_manager() -> ProcessManager:
    """Get the global process manager instance"""
    if process_manager is None:
        raise RuntimeError("Process manager not initialized")
    return process_manager
