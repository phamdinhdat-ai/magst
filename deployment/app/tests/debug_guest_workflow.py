#!/usr/bin/env python3
"""
Debug Guest Workflow Test Script
-------------------------------
This script directly tests the GuestWorkflow class without going through the API.
It's useful for diagnosing issues with the workflow execution itself.

Usage:
    python debug_guest_workflow.py
"""

import os
import sys
import json
import asyncio
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
import traceback
from loguru import logger

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("debug_guest_workflow_{time}.log", level="DEBUG", rotation="10 MB")

try:
    # Import the GuestWorkflow class
    from app.agents.workflow.guest_workflow import GuestWorkflow, AgentState
    from app.agents.workflow.initalize import llm_instance, llm_reasoning
    from app.core.config import get_settings
    
    # Initialize settings
    settings = get_settings()
    WORKFLOW_AVAILABLE = True
    
except ImportError as e:
    logger.error(f"Failed to import workflow modules: {e}")
    WORKFLOW_AVAILABLE = False

TEST_QUERIES = [
    "Tell me about GeneStory",
    "What genetic testing services do you offer?",
    "What is pharmacogenomics?",
]

async def test_direct_workflow_execution(query: str) -> None:
    """Test directly executing the workflow without going through the API"""
    if not WORKFLOW_AVAILABLE:
        logger.error("Workflow modules not available, cannot run test")
        return
    
    try:
        logger.info(f"Creating GuestWorkflow instance for query: {query}")
        workflow = GuestWorkflow()
        
        # Create a session and interaction
        guest_id = f"test_{uuid.uuid4()}"
        session_id = f"session_{uuid.uuid4()}"
        config = {"configurable": {"thread_id": session_id}}
        
        logger.info("Testing arun_streaming method...")
        start_time = datetime.now()
        
        # Test streaming execution
        response_events = []
        full_response = ""
        try:
            async for event in workflow.arun_streaming(query, config, guest_id=guest_id):
                response_events.append(event)
                event_type = event.get("event")
                
                if event_type == "answer_chunk":
                    full_response = event.get("data", "")
                elif event_type == "final_result":
                    final_data = event.get("data", {})
                    full_response = final_data.get("full_final_answer", full_response)
        except Exception as e:
            logger.error(f"Error during workflow streaming execution: {e}")
            logger.error(traceback.format_exc())
            raise
        
        execution_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Workflow execution completed in {execution_time:.2f} seconds")
        
        # Print results
        print("\n" + "="*80)
        print(f"QUERY: {query}")
        print("-"*80)
        print(f"RESPONSE: {full_response[:1000]}...")
        if len(full_response) > 1000:
            print(f"[Note: Response truncated, total length: {len(full_response)} chars]")
        print("-"*80)
        print(f"EXECUTION TIME: {execution_time:.2f} seconds")
        print(f"TOTAL EVENTS: {len(response_events)}")
        
        # Check for event types
        event_types = {}
        for event in response_events:
            event_type = event.get("event", "unknown")
            event_types[event_type] = event_types.get(event_type, 0) + 1
        
        print("\nEVENT TYPES:")
        for event_type, count in event_types.items():
            print(f"  {event_type}: {count}")
        
        # Check for errors
        errors = [event for event in response_events if event.get("event") == "error"]
        if errors:
            print("\nERRORS DETECTED:")
            for error in errors:
                print(f"  {error.get('data', {}).get('error', 'Unknown error')}")
        
        print("="*80 + "\n")
        
    except Exception as e:
        logger.error(f"Failed to execute workflow: {e}")
        logger.error(traceback.format_exc())
        print(f"\n[ERROR] Workflow execution failed: {e}")

async def test_workflow_components() -> None:
    """Test individual components of the workflow to identify issues"""
    if not WORKFLOW_AVAILABLE:
        logger.error("Workflow modules not available, cannot run component test")
        return
    
    try:
        workflow = GuestWorkflow()
        
        print("\nTESTING WORKFLOW COMPONENTS")
        print("="*80)
        
        # 1. Test LLM instances
        print("\n1. Testing LLM instances...")
        try:
            if llm_instance:
                print(f"  Main LLM: {type(llm_instance).__name__}, Model: {getattr(llm_instance, 'model_name', 'Unknown')}")
                
                # Test a simple completion
                try:
                    response = await llm_instance.agenerate([{"text": "Hello, how are you?"}])
                    print(f"  LLM test successful")
                except Exception as e:
                    print(f"  [ERROR] LLM test failed: {e}")
            else:
                print("  [ERROR] llm_instance not available")
                
            if llm_reasoning:
                print(f"  Reasoning LLM: {type(llm_reasoning).__name__}, Model: {getattr(llm_reasoning, 'model_name', 'Unknown')}")
            else:
                print("  [ERROR] llm_reasoning not available")
                
        except Exception as e:
            print(f"  [ERROR] LLM test failed: {e}")
        
        # 2. Test agent initialization
        print("\n2. Testing agent initialization...")
        try:
            agents = workflow.agents
            print(f"  Total agents: {len(agents)}")
            for agent_name, agent in agents.items():
                print(f"  - {agent_name}: {type(agent).__name__}")
        except Exception as e:
            print(f"  [ERROR] Agent initialization test failed: {e}")
        
        # 3. Test graph compilation
        print("\n3. Testing graph compilation...")
        try:
            graph = workflow.graph
            if graph:
                print(f"  Graph successfully compiled")
            else:
                print(f"  [ERROR] Graph compilation failed")
        except Exception as e:
            print(f"  [ERROR] Graph compilation test failed: {e}")
        
        # 4. Test cache manager
        print("\n4. Testing cache manager...")
        try:
            cache_manager = workflow.cache_manager
            if cache_manager:
                print(f"  Cache manager: {type(cache_manager).__name__}")
                print(f"  Cache active: {cache_manager.is_active()}")
            else:
                print(f"  [ERROR] Cache manager not available")
        except Exception as e:
            print(f"  [ERROR] Cache manager test failed: {e}")
        
        print("\nCOMPONENT TESTS COMPLETED")
        print("="*80)
        
    except Exception as e:
        logger.error(f"Component testing failed: {e}")
        logger.error(traceback.format_exc())
        print(f"\n[ERROR] Component testing failed: {e}")

async def test_entry_agent(query: str) -> None:
    """Test the entry agent directly"""
    if not WORKFLOW_AVAILABLE:
        logger.error("Workflow modules not available, cannot run entry agent test")
        return
    
    try:
        workflow = GuestWorkflow()
        entry_agent = workflow.agents.get("entry")
        
        if not entry_agent:
            print("[ERROR] Entry agent not found in workflow")
            return
        
        print("\nTESTING ENTRY AGENT")
        print("="*80)
        print(f"Query: {query}")
        
        # Create a minimal state for testing
        state = AgentState(
            original_query=query,
            chat_history=[],
            iteration_count=0,
            user_role="guest",
            guest_id=f"test_{uuid.uuid4()}",
            session_id=f"session_{uuid.uuid4()}",
            timestamp=datetime.now().isoformat()
        )
        
        # Execute the entry agent
        try:
            result_state = await entry_agent.aexecute(state)
            
            print("\nEntry Agent Output:")
            print(f"  Agent response: {result_state.get('agent_response', '')[:500]}...")
            print(f"  Classified agent: {result_state.get('classified_agent', 'None')}")
            print(f"  Needs rewrite: {result_state.get('needs_rewrite', False)}")
            
            # Additional state inspection
            for key, value in result_state.items():
                if key not in ['agent_response', 'classified_agent', 'needs_rewrite', 'original_query', 'chat_history']:
                    print(f"  {key}: {value}")
            
        except Exception as e:
            print(f"[ERROR] Entry agent execution failed: {e}")
            logger.error(f"Entry agent execution failed: {e}")
            logger.error(traceback.format_exc())
    
    except Exception as e:
        logger.error(f"Entry agent test failed: {e}")
        logger.error(traceback.format_exc())
        print(f"\n[ERROR] Entry agent test failed: {e}")

async def debug_workflow_graph() -> None:
    """Debug the workflow graph structure"""
    if not WORKFLOW_AVAILABLE:
        logger.error("Workflow modules not available, cannot debug graph")
        return
    
    try:
        workflow = GuestWorkflow()
        graph = workflow.graph
        
        print("\nDEBUGGING WORKFLOW GRAPH")
        print("="*80)
        
        # Check if the graph is properly compiled
        if not hasattr(graph, 'driver'):
            print("[ERROR] Graph is not properly compiled")
            return
        
        print("Graph structure verification:")
        print("  Entry point defined: Yes")
        
        # Check for nodes in the graph
        nodes = ["entry", "rewriter", "specialist_agent", "reflection", "supervisor"]
        for node in nodes:
            try:
                # Check if the node exists in the graph's schema
                node_exists = any(node == n.name for n in graph.driver.graph.nodes)
                print(f"  Node '{node}' exists: {node_exists}")
            except Exception as e:
                print(f"  [ERROR] Failed to verify node '{node}': {e}")
        
        print("\nWorkflow routing functions:")
        routing_funcs = [
            "_route_after_entry",
            "_route_after_reflection",
        ]
        
        for func_name in routing_funcs:
            if hasattr(workflow, func_name):
                print(f"  {func_name}: Found")
            else:
                print(f"  {func_name}: [ERROR] Not found")
        
        print("\nGraph analysis complete")
        
    except Exception as e:
        logger.error(f"Graph debugging failed: {e}")
        logger.error(traceback.format_exc())
        print(f"\n[ERROR] Graph debugging failed: {e}")

async def test_specific_agent(agent_name: str, query: str) -> None:
    """Test a specific agent in isolation"""
    if not WORKFLOW_AVAILABLE:
        logger.error("Workflow modules not available, cannot run agent test")
        return
    
    try:
        workflow = GuestWorkflow()
        agent = workflow.agents.get(agent_name)
        
        if not agent:
            print(f"[ERROR] Agent '{agent_name}' not found in workflow")
            return
        
        print(f"\nTESTING AGENT: {agent_name}")
        print("="*80)
        print(f"Query: {query}")
        
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
        try:
            start_time = datetime.now()
            result_state = await agent.aexecute(state)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            print(f"\n{agent_name} Output:")
            print(f"  Execution time: {execution_time:.2f} seconds")
            
            # Print the agent's response
            if 'agent_response' in result_state:
                print(f"  Agent response: {result_state['agent_response'][:500]}...")
                if len(result_state['agent_response']) > 500:
                    print(f"  [Note: Response truncated, total length: {len(result_state['agent_response'])} chars]")
            
            # Print other relevant state changes
            for key, value in result_state.items():
                if key not in ['agent_response', 'original_query', 'chat_history', 'rewritten_query']:
                    if isinstance(value, str) and len(value) > 100:
                        print(f"  {key}: {value[:100]}...")
                    else:
                        print(f"  {key}: {value}")
            
        except Exception as e:
            print(f"[ERROR] Agent '{agent_name}' execution failed: {e}")
            logger.error(f"Agent '{agent_name}' execution failed: {e}")
            logger.error(traceback.format_exc())
    
    except Exception as e:
        logger.error(f"Agent test failed: {e}")
        logger.error(traceback.format_exc())
        print(f"\n[ERROR] Agent test failed: {e}")

async def main() -> None:
    """Main entry point for the debug script"""
    print("\nGuest Workflow Debugging Tool")
    print("="*80)
    print("Started:", datetime.now().isoformat())
    
    if not WORKFLOW_AVAILABLE:
        print("\n[ERROR] Workflow modules are not available. Cannot run tests.")
        return
    
    # 1. First, run component tests to verify basic functionality
    await test_workflow_components()
    
    # 2. Debug the workflow graph structure
    await debug_workflow_graph()
    
    # 3. Test the entry agent with a simple query
    await test_entry_agent("Tell me about GeneStory")
    
    # 4. Test some other key agents
    await test_specific_agent("GuestAgent", "Tell me about GeneStory")
    await test_specific_agent("CompanyAgent", "What is GeneStory?")
    
    # 5. Finally, do a full workflow execution test
    print("\nRunning full workflow execution test...")
    for query in TEST_QUERIES:
        await test_direct_workflow_execution(query)
    
    print("\nAll debug tests completed.")
    print("="*80)
    print("Review the output above and the log file for details.")

if __name__ == "__main__":
    asyncio.run(main())
