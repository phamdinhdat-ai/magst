#!/usr/bin/env python3
"""
Test Guest Request Queue
-----------------------
This script tests the guest request queue system in isolation to ensure it's working properly.
It can help identify issues with the queue system separate from workflow execution.

Usage:
    python test_guest_request_queue.py [--parallel N] [--delay SECONDS]

Options:
    --parallel  Number of parallel requests to test (default: 3)
    --delay     Delay between requests in seconds (default: 1)
    --timeout   Maximum time to wait for request completion in seconds (default: 60)
"""

import os
import sys
import json
import asyncio
import argparse
import uuid
import time
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path
import traceback
from loguru import logger

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Configure logging
os.makedirs("logs", exist_ok=True)
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("logs/test_queue_{time}.log", level="DEBUG", rotation="10 MB")

# Parse command line arguments
parser = argparse.ArgumentParser(description="Test Guest Request Queue")
parser.add_argument("--parallel", type=int, default=3, help="Number of parallel requests to test")
parser.add_argument("--delay", type=float, default=1, help="Delay between requests in seconds")
parser.add_argument("--timeout", type=int, default=60, help="Maximum time to wait for completion (seconds)")
args = parser.parse_args()

# Create a mock workflow for testing
class MockGuestWorkflow:
    """Mock implementation of GuestWorkflow for testing the queue"""
    
    def __init__(self, delay: float = 2.0):
        self.delay = delay
        logger.info(f"Initialized MockGuestWorkflow with delay={delay}")
    
    async def arun(self, query: str, config: Dict[str, Any], guest_id: str) -> str:
        """Mock implementation of arun method"""
        logger.info(f"MockGuestWorkflow.arun called with query: {query}, guest_id: {guest_id}")
        await asyncio.sleep(self.delay)
        return f"Mock response for query: {query}"
    
    async def arun_streaming(self, query: str, config: Dict[str, Any], guest_id: str):
        """Mock implementation of arun_streaming method"""
        logger.info(f"MockGuestWorkflow.arun_streaming called with query: {query}, guest_id: {guest_id}")
        
        # Yield some mock events
        yield {"event": "start", "data": {"status": "processing"}}
        
        chunks = [
            "This is ",
            "a mock ",
            "response ",
            "for query: ",
            query
        ]
        
        for chunk in chunks:
            await asyncio.sleep(self.delay / len(chunks))
            yield {"event": "answer_chunk", "data": chunk}
        
        await asyncio.sleep(self.delay / 2)
        yield {
            "event": "final_result", 
            "data": {
                "full_final_answer": f"This is a mock response for query: {query}"
            }
        }


# Import the guest request queue if available
try:
    from app.api.v1.guest_request_queue import GuestRequestQueue
    QUEUE_AVAILABLE = True
except ImportError:
    logger.warning("GuestRequestQueue not available, trying to create a mock implementation")
    QUEUE_AVAILABLE = False

# Mock implementation of the guest request queue
class MockGuestRequestQueue:
    """Mock implementation of GuestRequestQueue for testing"""
    
    def __init__(self):
        self.workflow = MockGuestWorkflow()
        self.requests = {}
        self.results = {}
        self.processing = set()
        
        # Create a task queue
        self.queue = asyncio.Queue()
        
        # Start worker tasks
        self.workers = [asyncio.create_task(self._worker()) for _ in range(2)]
        
        logger.info("MockGuestRequestQueue initialized")
    
    async def _worker(self):
        """Worker task that processes requests from the queue"""
        while True:
            request_id, request_data = await self.queue.get()
            
            try:
                self.processing.add(request_id)
                self.requests[request_id]["status"] = "processing"
                
                # Process the request
                query = request_data.get("query", "")
                guest_id = request_data.get("guest_id", "")
                session_id = request_data.get("session_id", "")
                
                config = {"configurable": {"thread_id": session_id}}
                
                result = await self.workflow.arun(query, config, guest_id)
                
                # Store the result
                self.results[request_id] = result
                self.requests[request_id]["status"] = "completed"
                
            except Exception as e:
                logger.error(f"Error processing request {request_id}: {e}")
                self.requests[request_id]["status"] = "failed"
                self.requests[request_id]["error"] = str(e)
            finally:
                self.processing.remove(request_id)
                self.queue.task_done()
    
    async def enqueue_request(self, request_data: Dict[str, Any]) -> str:
        """Enqueue a request for processing"""
        request_id = str(uuid.uuid4())
        
        self.requests[request_id] = {
            "data": request_data,
            "status": "queued",
            "timestamp": datetime.now().isoformat()
        }
        
        # Add the request to the queue
        await self.queue.put((request_id, request_data))
        
        return request_id
    
    def get_request_status(self, request_id: str) -> Dict[str, Any]:
        """Get the status of a request"""
        if request_id not in self.requests:
            return {"status": "not_found"}
        
        return self.requests[request_id]
    
    def get_result(self, request_id: str) -> Any:
        """Get the result of a completed request"""
        return self.results.get(request_id)
    
    def get_status(self) -> Dict[str, Any]:
        """Get the overall status of the queue"""
        queued = sum(1 for req in self.requests.values() if req["status"] == "queued")
        processing = len(self.processing)
        completed = sum(1 for req in self.requests.values() if req["status"] == "completed")
        failed = sum(1 for req in self.requests.values() if req["status"] == "failed")
        
        return {
            "queue_size": self.queue.qsize(),
            "queued": queued,
            "processing": processing,
            "completed": completed,
            "failed": failed,
            "total": len(self.requests)
        }


async def test_queue(queue, num_requests: int = 3, delay: float = 1.0, timeout: int = 60) -> None:
    """Test the queue with multiple requests"""
    print(f"\nTesting queue with {num_requests} parallel requests (delay: {delay}s)")
    print("=" * 80)
    
    # Generate test requests
    request_ids = []
    for i in range(num_requests):
        # Create a test request
        guest_id = f"test_guest_{uuid.uuid4()}"
        session_id = f"test_session_{uuid.uuid4()}"
        query = f"Test query {i+1}: What is genetic testing?"
        
        request_data = {
            "guest_id": guest_id,
            "session_id": session_id,
            "query": query
        }
        
        # Enqueue the request
        print(f"\nEnqueuing request {i+1}/{num_requests}: {query[:40]}...")
        request_id = await queue.enqueue_request(request_data)
        request_ids.append(request_id)
        print(f"Request ID: {request_id}")
        
        # Print queue status
        status = queue.get_status()
        print(f"Queue status: {status}")
        
        # Add a delay between requests if specified
        if i < num_requests - 1 and delay > 0:
            await asyncio.sleep(delay)
    
    # Monitor the queue until all requests are completed or timeout
    print("\nMonitoring queue for completion...")
    start_time = time.time()
    pending_requests = set(request_ids)
    
    while pending_requests and time.time() - start_time < timeout:
        # Check the status of all pending requests
        for request_id in list(pending_requests):
            status = queue.get_request_status(request_id)
            
            if status["status"] == "completed":
                print(f"\nRequest {request_id} completed")
                pending_requests.remove(request_id)
                
                # Get and print the result
                result = queue.get_result(request_id)
                print(f"Result: {result[:100]}...")
                
            elif status["status"] == "failed":
                print(f"\nRequest {request_id} failed: {status.get('error', 'Unknown error')}")
                pending_requests.remove(request_id)
        
        # Print current status
        if pending_requests:
            queue_status = queue.get_status()
            print(f"\rQueue: {queue_status['queue_size']} queued, {queue_status['processing']} processing, "
                  f"{queue_status['completed']} completed, {queue_status['failed']} failed", 
                  end="", flush=True)
            
            await asyncio.sleep(1)
    
    # Check for requests that didn't complete within the timeout
    if pending_requests:
        print(f"\n\n[WARNING] {len(pending_requests)} requests did not complete within the timeout")
        for request_id in pending_requests:
            status = queue.get_request_status(request_id)
            print(f"Request {request_id}: {status['status']}")
    
    # Print final queue status
    final_status = queue.get_status()
    print(f"\n\nFinal queue status: {final_status}")
    print("=" * 80)


async def run_test_with_actual_queue() -> None:
    """Run the test with the actual GuestRequestQueue implementation"""
    from app.api.v1.guest_request_queue import GuestRequestQueue
    
    try:
        # Try to import the GuestWorkflow
        from app.agents.workflow.guest_workflow import GuestWorkflow
        
        # Create the queue
        queue = GuestRequestQueue()
        print("Created actual GuestRequestQueue")
        
        # Run the test
        await test_queue(queue, args.parallel, args.delay, args.timeout)
        
    except Exception as e:
        logger.error(f"Error testing actual queue: {e}")
        logger.error(traceback.format_exc())
        print(f"\n[ERROR] Failed to test actual queue: {e}")


async def run_test_with_mock_queue() -> None:
    """Run the test with a mock queue implementation"""
    try:
        # Create a mock queue
        queue = MockGuestRequestQueue()
        print("Created mock GuestRequestQueue")
        
        # Run the test
        await test_queue(queue, args.parallel, args.delay, args.timeout)
        
    except Exception as e:
        logger.error(f"Error testing mock queue: {e}")
        logger.error(traceback.format_exc())
        print(f"\n[ERROR] Failed to test mock queue: {e}")


async def main() -> None:
    """Main entry point for the test script"""
    print("\nGuest Request Queue Test Tool")
    print("=" * 80)
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Parameters: {args.parallel} parallel requests, {args.delay}s delay, {args.timeout}s timeout")
    
    if QUEUE_AVAILABLE:
        print("\nActual GuestRequestQueue implementation found.")
        await run_test_with_actual_queue()
    else:
        print("\nActual GuestRequestQueue not available, using mock implementation.")
        await run_test_with_mock_queue()
    
    print("\nTest completed.")
    print(f"Finished: {datetime.now().isoformat()}")


if __name__ == "__main__":
    asyncio.run(main())
