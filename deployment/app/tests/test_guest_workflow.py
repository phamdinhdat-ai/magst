#!/usr/bin/env python3
"""
Guest Workflow Test Script
--------------------------
This script tests the guest workflow execution and request queue functionality.
It can be used to diagnose issues with the workflow execution.

Usage:
    python test_guest_workflow.py [--simple] [--stream] [--queue-test]
    
    --simple: Test the simple (non-streaming) workflow
    --stream: Test the streaming workflow
    --queue-test: Test the request queue system with multiple concurrent requests
    --all: Run all tests
"""

import os
import sys
import json
import time
import asyncio
import argparse
import uuid
import random
from datetime import datetime
from typing import List, Dict, Any, Optional
import httpx
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("test_guest_workflow_{time}.log", level="DEBUG", rotation="10 MB")

# Set up rich console for pretty output
console = Console()

# Test configuration
BASE_URL = "http://localhost:8000"  # Update as needed
GUEST_API_URL = f"{BASE_URL}/api/v1/guest"
DEFAULT_TIMEOUT = 30.0  # Seconds
MAX_CONCURRENT_REQUESTS = 5
TEST_QUERIES = [
    "Tell me about GeneStory",
    "What genetic testing services do you offer?",
    "How can I learn about my genetic ancestry?",
    "What is pharmacogenomics?",
    "How can genetics help with medication?",
    "Tell me about DNA testing",
    "What are the benefits of genetic counseling?",
    "How do you protect genetic data privacy?",
    "What can genetic testing tell me about health risks?",
    "How accurate is your genetic testing?"
]

# Helper functions
async def make_simple_request(query: str, session_id: Optional[str] = None) -> Dict[str, Any]:
    """Make a simple (non-streaming) request to the guest workflow API"""
    url = f"{GUEST_API_URL}/chat/simple"
    payload = {
        "query": query,
        "session_id": session_id or str(uuid.uuid4()),
        "preferred_language": "en"
    }
    
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        try:
            console.print(f"[bold blue]Sending simple request:[/bold blue] {query}")
            response = await client.post(url, json=payload)
            response.raise_for_status()
            result = response.json()
            return result
        except httpx.HTTPStatusError as e:
            console.print(f"[bold red]HTTP Error:[/bold red] {e.response.status_code} - {e.response.text}")
            logger.error(f"HTTP Error: {e.response.status_code} - {e.response.text}")
            return {"error": str(e), "status_code": e.response.status_code}
        except httpx.RequestError as e:
            console.print(f"[bold red]Request Error:[/bold red] {str(e)}")
            logger.error(f"Request Error: {str(e)}")
            return {"error": str(e)}
        except Exception as e:
            console.print(f"[bold red]Unexpected Error:[/bold red] {str(e)}")
            logger.error(f"Unexpected Error: {str(e)}", exc_info=True)
            return {"error": str(e)}

async def stream_workflow_response(query: str, session_id: Optional[str] = None) -> Dict[str, Any]:
    """Make a streaming request to the guest workflow API and process the events"""
    url = f"{GUEST_API_URL}/chat"
    payload = {
        "query": query,
        "session_id": session_id or str(uuid.uuid4()),
        "preferred_language": "en"
    }
    
    collected_data = {
        "events": [],
        "full_response": "",
        "suggested_questions": [],
        "agents_used": [],
        "processing_time_ms": None,
        "errors": []
    }
    
    console.print(f"[bold blue]Sending streaming request:[/bold blue] {query}")
    
    # Using httpx for streaming
    async with httpx.AsyncClient(timeout=None) as client:
        try:
            # Start streaming request
            async with client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                
                # Process the stream of events
                async for line in response.aiter_lines():
                    if not line.strip() or not line.startswith("data:"):
                        continue
                    
                    # Extract the JSON data
                    try:
                        json_str = line[5:].strip()  # Remove "data: " prefix
                        event_data = json.loads(json_str)
                        
                        # Track the event
                        collected_data["events"].append(event_data)
                        
                        # Process different event types
                        event_type = event_data.get("event")
                        
                        if event_type == "queued":
                            console.print(f"  [yellow]Request queued at position: {event_data.get('data', {}).get('queue_position', 'unknown')}[/yellow]")
                        
                        elif event_type == "processing":
                            console.print(f"  [yellow]Processing started[/yellow]")
                            
                        elif event_type == "node_start":
                            node_name = event_data.get("data", {}).get("node")
                            if node_name and node_name not in collected_data["agents_used"]:
                                collected_data["agents_used"].append(node_name)
                                console.print(f"  [cyan]Using agent: {node_name}[/cyan]")
                                
                        elif event_type == "answer_chunk":
                            chunk_text = event_data.get("data", "")
                            # Only print if it's a new chunk (to avoid duplication)
                            if chunk_text and chunk_text != collected_data["full_response"]:
                                console.print(f"  [green]Received answer chunk: {len(chunk_text)} chars[/green]")
                                collected_data["full_response"] = chunk_text
                                
                        elif event_type == "final_result":
                            final_data = event_data.get("data", {})
                            suggested_questions = final_data.get("suggested_questions", [])
                            if suggested_questions:
                                collected_data["suggested_questions"] = suggested_questions
                                
                            full_answer = final_data.get("full_final_answer", "")
                            if full_answer:
                                collected_data["full_response"] = full_answer
                                
                        elif event_type == "workflow_complete":
                            processing_time = event_data.get("data", {}).get("processing_time_ms")
                            if processing_time:
                                collected_data["processing_time_ms"] = processing_time
                                console.print(f"  [bold green]Workflow completed in {processing_time} ms[/bold green]")
                                
                        elif event_type == "error":
                            error_message = event_data.get("data", {}).get("error", "Unknown error")
                            collected_data["errors"].append(error_message)
                            console.print(f"  [bold red]Error: {error_message}[/bold red]")
                            
                        else:
                            # Generic event handling
                            console.print(f"  [blue]Event: {event_type}[/blue]")
                            
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON decode error: {e} - Line: {line[:100]}")
                        collected_data["errors"].append(f"JSON decode error: {str(e)}")
                        
        except httpx.HTTPStatusError as e:
            console.print(f"[bold red]HTTP Error:[/bold red] {e.response.status_code} - {e.response.text}")
            logger.error(f"HTTP Error: {e.response.status_code} - {e.response.text}")
            collected_data["errors"].append(f"HTTP Error: {e.response.status_code} - {e.response.text}")
            
        except httpx.RequestError as e:
            console.print(f"[bold red]Request Error:[/bold red] {str(e)}")
            logger.error(f"Request Error: {str(e)}")
            collected_data["errors"].append(f"Request Error: {str(e)}")
            
        except Exception as e:
            console.print(f"[bold red]Unexpected Error:[/bold red] {str(e)}")
            logger.error(f"Unexpected Error: {str(e)}", exc_info=True)
            collected_data["errors"].append(f"Unexpected Error: {str(e)}")
            
    return collected_data

async def get_queue_status() -> Dict[str, Any]:
    """Get the current status of the request queue"""
    url = f"{GUEST_API_URL}/queue/status"
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.get(url)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            console.print(f"[bold red]HTTP Error getting queue status:[/bold red] {e.response.status_code}")
            logger.error(f"HTTP Error getting queue status: {e.response.status_code} - {e.response.text}")
            return {"error": str(e), "status_code": e.response.status_code}
        except Exception as e:
            console.print(f"[bold red]Error getting queue status:[/bold red] {str(e)}")
            logger.error(f"Error getting queue status: {str(e)}")
            return {"error": str(e)}

async def get_health_status() -> Dict[str, Any]:
    """Get the current health status of the API"""
    url = f"{GUEST_API_URL}/health"
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.get(url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            console.print(f"[bold red]Error getting health status:[/bold red] {str(e)}")
            logger.error(f"Error getting health status: {str(e)}")
            return {"error": str(e)}

async def test_queue_load() -> None:
    """Test the request queue with multiple concurrent requests"""
    # Get current queue status
    console.print("\n[bold]Testing queue load handling...[/bold]")
    
    initial_status = await get_queue_status()
    if "error" in initial_status:
        console.print("[bold red]Could not get queue status, aborting queue test[/bold red]")
        return
    
    console.print(Panel.fit(
        "\n".join([f"{k}: {v}" for k, v in initial_status.get("queue_stats", {}).items()]),
        title="Initial Queue Status"
    ))
    
    # Create multiple concurrent requests
    console.print(f"Sending {MAX_CONCURRENT_REQUESTS} concurrent requests...")
    
    session_id = str(uuid.uuid4())
    
    # Use random queries from the test set
    queries = random.sample(TEST_QUERIES, min(MAX_CONCURRENT_REQUESTS, len(TEST_QUERIES)))
    
    # Show a progress spinner
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task("[green]Processing requests...", total=None)
        
        # Send all requests concurrently
        tasks = []
        for i, query in enumerate(queries):
            tasks.append(make_simple_request(f"[Concurrent Test {i+1}] {query}", session_id))
            
        # Wait for all to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Mark task as complete
        progress.update(task, completed=True)
    
    # Check results
    success_count = 0
    error_count = 0
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            console.print(f"[red]Request {i+1} failed: {str(result)}[/red]")
            error_count += 1
        elif "error" in result:
            console.print(f"[red]Request {i+1} failed: {result['error']}[/red]")
            error_count += 1
        else:
            console.print(f"[green]Request {i+1} succeeded[/green]")
            success_count += 1
    
    console.print(f"\n[bold]Queue Test Results:[/bold] {success_count} successful, {error_count} failed")
    
    # Get final queue status
    final_status = await get_queue_status()
    console.print(Panel.fit(
        "\n".join([f"{k}: {v}" for k, v in final_status.get("queue_stats", {}).items()]),
        title="Final Queue Status"
    ))

async def run_simple_test() -> None:
    """Run a test of the simple workflow endpoint"""
    console.print("\n[bold]Testing simple workflow endpoint...[/bold]")
    
    session_id = str(uuid.uuid4())
    result = await make_simple_request("Test request for the guest workflow system", session_id)
    
    if "error" in result:
        console.print("[bold red]Simple test failed[/bold red]")
    else:
        console.print("[bold green]Simple test succeeded[/bold green]")
        console.print(Panel.fit(
            f"Response: {result.get('agent_response', '')[:200]}...\n\n" +
            f"Processing time: {result.get('processing_time_ms')} ms\n" +
            f"Agents used: {', '.join(result.get('agents_used', []))}\n" +
            f"Suggested questions: {len(result.get('suggested_questions', []))}"
        ))

async def run_stream_test() -> None:
    """Run a test of the streaming workflow endpoint"""
    console.print("\n[bold]Testing streaming workflow endpoint...[/bold]")
    
    session_id = str(uuid.uuid4())
    result = await stream_workflow_response("Test streaming request for the guest workflow system", session_id)
    
    if result["errors"]:
        console.print("[bold red]Stream test failed[/bold red]")
        for error in result["errors"]:
            console.print(f"[red]Error: {error}[/red]")
    else:
        console.print("[bold green]Stream test succeeded[/bold green]")
        console.print(Panel.fit(
            f"Response: {result.get('full_response', '')[:200]}...\n\n" +
            f"Processing time: {result.get('processing_time_ms')} ms\n" +
            f"Agents used: {', '.join(result.get('agents_used', []))}\n" +
            f"Suggested questions: {len(result.get('suggested_questions', []))}\n" +
            f"Total events: {len(result.get('events', []))}"
        ))

async def check_system_health() -> None:
    """Check the overall system health"""
    console.print("\n[bold]Checking system health...[/bold]")
    
    result = await get_health_status()
    
    if "error" in result:
        console.print("[bold red]Health check failed[/bold red]")
    else:
        status = result.get("status", "unknown")
        color = "green" if status == "healthy" else "red"
        console.print(f"[bold {color}]System status: {status}[/bold {color}]")
        
        # Display queue health if available
        queue_health = result.get("queue_health", {})
        if queue_health:
            console.print(Panel.fit(
                "\n".join([f"{k}: {v}" for k, v in queue_health.items()]),
                title="Queue Health"
            ))

async def main() -> None:
    """Main entry point for the test script"""
    parser = argparse.ArgumentParser(description="Test the guest workflow system")
    parser.add_argument("--simple", action="store_true", help="Test simple workflow endpoint")
    parser.add_argument("--stream", action="store_true", help="Test streaming workflow endpoint")
    parser.add_argument("--queue-test", action="store_true", help="Test request queue with concurrent requests")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    args = parser.parse_args()
    
    # Default to all tests if none specified
    run_all = args.all or not (args.simple or args.stream or args.queue_test)
    
    console.print(Panel.fit(
        f"Guest Workflow Test\nAPI URL: {GUEST_API_URL}\nStarted: {datetime.now().isoformat()}", 
        title="Test Configuration"
    ))
    
    # Always check system health first
    await check_system_health()
    
    if args.simple or run_all:
        await run_simple_test()
        
    if args.stream or run_all:
        await run_stream_test()
        
    if args.queue_test or run_all:
        await test_queue_load()
    
    console.print("\n[bold]All tests completed[/bold]")

if __name__ == "__main__":
    asyncio.run(main())
