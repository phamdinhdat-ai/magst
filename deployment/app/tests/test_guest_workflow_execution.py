#!/usr/bin/env python3
"""
Test Guest Workflow Execution
----------------------------
This script provides a comprehensive testing environment for the guest workflow.
It offers different testing modes and configuration options to help diagnose issues.

Usage:
    python test_guest_workflow_execution.py --mode [simple|stream|component] --query "Your test query"

Options:
    --mode      Testing mode: simple (default), stream, component
    --query     Test query to use (defaults to predefined test queries)
    --verbose   Enable verbose output
    --debug     Enable debug mode with more detailed logging
    --agent     Test a specific agent (use with component mode)
"""

import os
import sys
import json
import asyncio
import argparse
import uuid
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
import traceback
from loguru import logger

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Configure logging
os.makedirs("logs", exist_ok=True)
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("logs/test_workflow_{time}.log", level="DEBUG", rotation="10 MB")

# Parse command line arguments
parser = argparse.ArgumentParser(description="Test Guest Workflow Execution")
parser.add_argument("--mode", choices=["simple", "stream", "component"], default="simple",
                  help="Testing mode: simple, stream, component")
parser.add_argument("--query", type=str, help="Test query to use")
parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
parser.add_argument("--debug", action="store_true", help="Enable debug mode")
parser.add_argument("--agent", type=str, help="Test a specific agent (use with component mode)")
args = parser.parse_args()

# Set logging level based on debug flag
if args.debug:
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")

# Default test queries
DEFAULT_TEST_QUERIES = [
    "Tell me about GeneStory",
    "What genetic testing services do you offer?",
    "What is pharmacogenomics?",
]

# Use provided query or default
TEST_QUERIES = [args.query] if args.query else DEFAULT_TEST_QUERIES

try:
    # Import required modules
    from app.agents.workflow.guest_workflow import GuestWorkflow, AgentState
    from app.agents.workflow.initalize import llm_instance, llm_reasoning
    from app.core.config import get_settings
    
    # Try to import the guest request queue if available
    try:
        from app.api.v1.guest_request_queue import GuestRequestQueue
        QUEUE_AVAILABLE = True
    except ImportError:
        logger.warning("GuestRequestQueue not available, queue tests will be skipped")
        QUEUE_AVAILABLE = False
    
    # Initialize settings
    settings = get_settings()
    WORKFLOW_AVAILABLE = True
    
except ImportError as e:
    logger.error(f"Failed to import workflow modules: {e}")
    logger.error(traceback.format_exc())
    WORKFLOW_AVAILABLE = False


class WorkflowTester:
    """Class to handle different types of workflow tests"""
    
    def __init__(self, verbose: bool = False, debug: bool = False):
        self.verbose = verbose
        self.debug = debug
        self.workflow = None
        self.queue = None
        
        # Initialize the workflow
        if WORKFLOW_AVAILABLE:
            try:
                self.workflow = GuestWorkflow()
                logger.info("GuestWorkflow initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize GuestWorkflow: {e}")
                logger.error(traceback.format_exc())
        
        # Initialize the queue if available
        if QUEUE_AVAILABLE:
            try:
                self.queue = GuestRequestQueue()
                logger.info("GuestRequestQueue initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize GuestRequestQueue: {e}")
                logger.error(traceback.format_exc())
    
    async def test_simple_execution(self, query: str) -> None:
        """Test direct workflow execution without streaming"""
        if not self.workflow:
            logger.error("Workflow not available, cannot run simple execution test")
            return
        
        print(f"\n{'='*80}\nTesting Simple Execution\n{'='*80}")
        print(f"Query: {query}")
        
        try:
            # Generate a test guest ID and session ID
            guest_id = f"test_{uuid.uuid4()}"
            session_id = f"session_{uuid.uuid4()}"
            config = {"configurable": {"thread_id": session_id}}
            
            # Execute the workflow
            start_time = datetime.now()
            
            # Use the arun method for non-streaming execution
            result = await self.workflow.arun(query, config, guest_id=guest_id)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Print results
            print(f"\nExecution completed in {execution_time:.2f} seconds")
            print(f"\nResponse:")
            print(f"{'-'*80}\n{result}\n{'-'*80}")
            
            # Add debug info if requested
            if self.debug:
                print("\nDebug Information:")
                print(f"  Guest ID: {guest_id}")
                print(f"  Session ID: {session_id}")
                
            return result
            
        except Exception as e:
            logger.error(f"Simple execution test failed: {e}")
            logger.error(traceback.format_exc())
            print(f"\n[ERROR] Simple execution test failed: {e}")
            if self.debug:
                print(traceback.format_exc())
    
    async def test_streaming_execution(self, query: str) -> None:
        """Test streaming workflow execution"""
        if not self.workflow:
            logger.error("Workflow not available, cannot run streaming execution test")
            return
        
        print(f"\n{'='*80}\nTesting Streaming Execution\n{'='*80}")
        print(f"Query: {query}")
        
        try:
            # Generate a test guest ID and session ID
            guest_id = f"test_{uuid.uuid4()}"
            session_id = f"session_{uuid.uuid4()}"
            config = {"configurable": {"thread_id": session_id}}
            
            # Execute the workflow with streaming
            start_time = datetime.now()
            
            # Collect response chunks
            chunks = []
            full_response = ""
            
            print("\nStreaming response:")
            print("-" * 80)
            
            async for event in self.workflow.arun_streaming(query, config, guest_id=guest_id):
                event_type = event.get("event")
                
                if self.verbose:
                    print(f"Event: {event_type}")
                
                if event_type == "answer_chunk":
                    chunk = event.get("data", "")
                    chunks.append(chunk)
                    if self.verbose:
                        print(chunk, end="", flush=True)
                    else:
                        # Print a dot to show progress
                        print(".", end="", flush=True)
                        
                elif event_type == "final_result":
                    final_data = event.get("data", {})
                    if isinstance(final_data, dict):
                        full_response = final_data.get("full_final_answer", "")
                    else:
                        full_response = str(final_data)
                    
                elif event_type == "error":
                    error_data = event.get("data", {})
                    print(f"\n[ERROR] {error_data.get('error', 'Unknown error')}")
                    
                elif event_type == "agent_thinking":
                    if self.verbose:
                        thinking = event.get("data", {}).get("thinking", "")
                        print(f"\n[Agent thinking]: {thinking[:100]}...")
                    else:
                        print("T", end="", flush=True)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Print results
            print(f"\n\nStreaming completed in {execution_time:.2f} seconds")
            print(f"Total chunks: {len(chunks)}")
            
            print(f"\nFinal Response:")
            print(f"{'-'*80}\n{full_response}\n{'-'*80}")
            
            # Add debug info if requested
            if self.debug:
                print("\nDebug Information:")
                print(f"  Guest ID: {guest_id}")
                print(f"  Session ID: {session_id}")
                
            return full_response
            
        except Exception as e:
            logger.error(f"Streaming execution test failed: {e}")
            logger.error(traceback.format_exc())
            print(f"\n[ERROR] Streaming execution test failed: {e}")
            if self.debug:
                print(traceback.format_exc())
    
    async def test_queue_execution(self, query: str) -> None:
        """Test workflow execution through the queue system"""
        if not self.queue:
            logger.error("Queue not available, cannot run queue execution test")
            return
        
        print(f"\n{'='*80}\nTesting Queue Execution\n{'='*80}")
        print(f"Query: {query}")
        
        try:
            # Generate a test guest ID and session ID
            guest_id = f"test_{uuid.uuid4()}"
            session_id = f"session_{uuid.uuid4()}"
            
            # Create the request data
            request_data = {
                "session_id": session_id,
                "query": query,
                "guest_id": guest_id
            }
            
            # Enqueue the request
            print("\nEnqueuing request...")
            start_time = datetime.now()
            request_id = await self.queue.enqueue_request(request_data)
            
            print(f"Request enqueued with ID: {request_id}")
            print(f"Queue status: {self.queue.get_status()}")
            
            # Monitor the queue for status updates
            max_wait_time = 60  # Maximum wait time in seconds
            start_wait = time.time()
            completed = False
            
            print("\nMonitoring queue...")
            while time.time() - start_wait < max_wait_time:
                status = self.queue.get_request_status(request_id)
                
                if status["status"] == "completed":
                    completed = True
                    print(f"\nRequest completed in {time.time() - start_wait:.2f} seconds")
                    break
                elif status["status"] == "failed":
                    print(f"\n[ERROR] Request failed: {status.get('error', 'Unknown error')}")
                    break
                elif status["status"] == "processing":
                    print("P", end="", flush=True)
                else:
                    print(".", end="", flush=True)
                
                await asyncio.sleep(0.5)
            
            if not completed:
                print("\n[WARNING] Request did not complete within the maximum wait time")
            
            # Get the final result
            result = self.queue.get_result(request_id)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Print results
            print(f"\nExecution through queue completed in {execution_time:.2f} seconds")
            
            if result:
                print(f"\nResponse:")
                print(f"{'-'*80}\n{result}\n{'-'*80}")
            else:
                print("\n[ERROR] No result returned from queue")
            
            # Add debug info if requested
            if self.debug:
                print("\nDebug Information:")
                print(f"  Request ID: {request_id}")
                print(f"  Guest ID: {guest_id}")
                print(f"  Session ID: {session_id}")
                print(f"  Queue status: {self.queue.get_status()}")
                
            return result
            
        except Exception as e:
            logger.error(f"Queue execution test failed: {e}")
            logger.error(traceback.format_exc())
            print(f"\n[ERROR] Queue execution test failed: {e}")
            if self.debug:
                print(traceback.format_exc())
    
    async def test_component(self, component_name: str, query: str) -> None:
        """Test a specific component of the workflow system"""
        if not self.workflow:
            logger.error("Workflow not available, cannot run component test")
            return
        
        print(f"\n{'='*80}\nTesting Component: {component_name}\n{'='*80}")
        
        if component_name == "llm":
            await self._test_llm()
        elif component_name == "agents":
            await self._test_agents()
        elif component_name == "graph":
            await self._test_graph()
        elif component_name == "cache":
            await self._test_cache()
        elif component_name.startswith("agent:"):
            agent_name = component_name.split(":", 1)[1]
            await self._test_specific_agent(agent_name, query)
        else:
            print(f"Unknown component: {component_name}")
    
    async def _test_llm(self) -> None:
        """Test the LLM instances"""
        print("\nTesting LLM instances...")
        
        try:
            if llm_instance:
                print(f"Main LLM: {type(llm_instance).__name__}")
                print(f"Model name: {getattr(llm_instance, 'model_name', 'Unknown')}")
                
                # Test a simple completion
                print("\nTesting LLM completion...")
                test_prompt = "Hello, can you tell me about genetic testing?"
                
                try:
                    start_time = time.time()
                    response = await llm_instance.agenerate([{"text": test_prompt}])
                    completion_time = time.time() - start_time
                    
                    print(f"LLM completion successful in {completion_time:.2f} seconds")
                    if self.verbose:
                        print(f"\nPrompt: {test_prompt}")
                        print(f"Response: {response.generations[0][0].text[:500]}...")
                    
                except Exception as e:
                    logger.error(f"LLM completion failed: {e}")
                    print(f"[ERROR] LLM completion failed: {e}")
                    if self.debug:
                        print(traceback.format_exc())
                
                # Test streaming if available
                if hasattr(llm_instance, "astream"):
                    print("\nTesting LLM streaming...")
                    try:
                        start_time = time.time()
                        stream = await llm_instance.astream(test_prompt)
                        chunks = []
                        
                        if self.verbose:
                            print("Streaming response:")
                        
                        async for chunk in stream:
                            chunks.append(chunk)
                            if self.verbose:
                                print(chunk, end="", flush=True)
                            else:
                                print(".", end="", flush=True)
                        
                        streaming_time = time.time() - start_time
                        print(f"\nLLM streaming successful in {streaming_time:.2f} seconds")
                        print(f"Received {len(chunks)} chunks")
                        
                    except Exception as e:
                        logger.error(f"LLM streaming failed: {e}")
                        print(f"\n[ERROR] LLM streaming failed: {e}")
                        if self.debug:
                            print(traceback.format_exc())
            else:
                print("[ERROR] llm_instance not available")
            
            # Test reasoning LLM if available
            if llm_reasoning:
                print(f"\nReasoning LLM: {type(llm_reasoning).__name__}")
                print(f"Model name: {getattr(llm_reasoning, 'model_name', 'Unknown')}")
            else:
                print("\n[ERROR] llm_reasoning not available")
                
        except Exception as e:
            logger.error(f"LLM test failed: {e}")
            print(f"[ERROR] LLM test failed: {e}")
            if self.debug:
                print(traceback.format_exc())
    
    async def _test_agents(self) -> None:
        """Test all agents in the workflow"""
        print("\nTesting all agents in workflow...")
        
        try:
            agents = self.workflow.agents
            print(f"Total agents: {len(agents)}")
            
            for agent_name, agent in agents.items():
                print(f"\n- Agent: {agent_name}")
                print(f"  Type: {type(agent).__name__}")
                
                # Check if the agent has required methods
                has_execute = hasattr(agent, "execute")
                has_aexecute = hasattr(agent, "aexecute")
                
                print(f"  Has execute method: {has_execute}")
                print(f"  Has aexecute method: {has_aexecute}")
                
                if self.verbose:
                    # Try to get the agent's description or tools
                    if hasattr(agent, "description"):
                        print(f"  Description: {agent.description}")
                    
                    if hasattr(agent, "tools") and agent.tools:
                        print(f"  Tools: {', '.join(str(tool) for tool in agent.tools)}")
            
        except Exception as e:
            logger.error(f"Agents test failed: {e}")
            print(f"[ERROR] Agents test failed: {e}")
            if self.debug:
                print(traceback.format_exc())
    
    async def _test_graph(self) -> None:
        """Test the workflow graph"""
        print("\nTesting workflow graph...")
        
        try:
            graph = self.workflow.graph
            
            if not hasattr(graph, 'driver'):
                print("[ERROR] Graph is not properly compiled")
                return
            
            print("Graph structure verification:")
            
            # Get all nodes and edges in the graph
            nodes = list(graph.driver.graph.nodes)
            edges = list(graph.driver.graph.edges)
            
            print(f"Total nodes: {len(nodes)}")
            for node in nodes:
                print(f"- Node: {node.name}")
            
            print(f"\nTotal edges: {len(edges)}")
            for edge in edges:
                print(f"- Edge: {edge.source} -> {edge.target}")
            
            # Check for important routing methods
            print("\nVerifying routing methods:")
            routing_methods = [
                "_route_after_entry",
                "_route_after_reflection",
            ]
            
            for method_name in routing_methods:
                if hasattr(self.workflow, method_name):
                    print(f"- {method_name}: Found")
                else:
                    print(f"- {method_name}: [ERROR] Not found")
            
        except Exception as e:
            logger.error(f"Graph test failed: {e}")
            print(f"[ERROR] Graph test failed: {e}")
            if self.debug:
                print(traceback.format_exc())
    
    async def _test_cache(self) -> None:
        """Test the cache manager"""
        print("\nTesting cache manager...")
        
        try:
            cache_manager = self.workflow.cache_manager
            
            if cache_manager:
                print(f"Cache manager: {type(cache_manager).__name__}")
                print(f"Cache active: {cache_manager.is_active()}")
                
                if hasattr(cache_manager, "get_statistics"):
                    stats = cache_manager.get_statistics()
                    print("\nCache statistics:")
                    for key, value in stats.items():
                        print(f"- {key}: {value}")
            else:
                print("[ERROR] Cache manager not available")
                
        except Exception as e:
            logger.error(f"Cache test failed: {e}")
            print(f"[ERROR] Cache test failed: {e}")
            if self.debug:
                print(traceback.format_exc())
    
    async def _test_specific_agent(self, agent_name: str, query: str) -> None:
        """Test a specific agent in isolation"""
        print(f"\nTesting specific agent: {agent_name}")
        print(f"Query: {query}")
        
        try:
            agent = self.workflow.agents.get(agent_name)
            
            if not agent:
                print(f"[ERROR] Agent '{agent_name}' not found in workflow")
                print(f"Available agents: {', '.join(self.workflow.agents.keys())}")
                return
            
            # Create a minimal state for testing
            state = AgentState(
                original_query=query,
                rewritten_query=query,  # Assume no rewrite needed for direct test
                chat_history=[],
                iteration_count=0,
                user_role="guest",
                guest_id=f"test_{uuid.uuid4()}",
                session_id=f"session_{uuid.uuid4()}",
                timestamp=datetime.now().isoformat()
            )
            
            # Execute the agent
            print("\nExecuting agent...")
            start_time = time.time()
            
            result_state = await agent.aexecute(state)
            
            execution_time = time.time() - start_time
            print(f"Agent execution completed in {execution_time:.2f} seconds")
            
            # Print results
            if 'agent_response' in result_state:
                print(f"\nAgent response:")
                print(f"{'-'*80}\n{result_state['agent_response'][:1000]}")
                if len(result_state['agent_response']) > 1000:
                    print(f"...[truncated, total length: {len(result_state['agent_response'])} chars]")
                print(f"{'-'*80}")
            
            # Print other relevant state changes
            if self.verbose:
                print("\nState changes:")
                for key, value in result_state.items():
                    if key not in ['agent_response', 'original_query', 'chat_history', 'rewritten_query']:
                        if isinstance(value, str) and len(value) > 100:
                            print(f"- {key}: {value[:100]}...")
                        else:
                            print(f"- {key}: {value}")
            
        except Exception as e:
            logger.error(f"Agent '{agent_name}' test failed: {e}")
            print(f"[ERROR] Agent '{agent_name}' test failed: {e}")
            if self.debug:
                print(traceback.format_exc())


async def main() -> None:
    """Main entry point for the test script"""
    print("\nGuest Workflow Execution Test Tool")
    print("="*80)
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Mode: {args.mode}")
    
    if not WORKFLOW_AVAILABLE:
        print("\n[ERROR] Workflow modules are not available. Cannot run tests.")
        return
    
    # Create the workflow tester
    tester = WorkflowTester(verbose=args.verbose, debug=args.debug)
    
    # Run tests based on selected mode
    if args.mode == "simple":
        for query in TEST_QUERIES:
            await tester.test_simple_execution(query)
    
    elif args.mode == "stream":
        for query in TEST_QUERIES:
            await tester.test_streaming_execution(query)
            
            # Also test queue if available
            if QUEUE_AVAILABLE:
                await tester.test_queue_execution(query)
    
    elif args.mode == "component":
        if args.agent:
            component = f"agent:{args.agent}"
        else:
            # Test all components
            components = ["llm", "agents", "graph", "cache"]
            
            for component in components:
                await tester.test_component(component, TEST_QUERIES[0])
                
    print("\nAll tests completed.")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(main())
