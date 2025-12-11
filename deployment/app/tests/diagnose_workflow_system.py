#!/usr/bin/env python3
"""
Workflow System Diagnostics
--------------------------
This script runs a series of diagnostic checks on the workflow system
to help identify common issues that might cause workflow execution failures.

Usage:
    python diagnose_workflow_system.py [--verbose]

Options:
    --verbose    Enable verbose output with more details
"""

import os
import sys
import importlib
import asyncio
import argparse
import traceback
import inspect
import pkgutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from loguru import logger

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Configure logging
os.makedirs("logs", exist_ok=True)
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("logs/diagnose_workflow_{time}.log", level="DEBUG", rotation="10 MB")

# Parse command line arguments
parser = argparse.ArgumentParser(description="Diagnose Workflow System")
parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
args = parser.parse_args()


class WorkflowDiagnostic:
    """Runs diagnostic tests on the workflow system"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results = {}
        self.errors = {}
        self.warnings = {}
        self.module_errors = {}
    
    def log_result(self, category: str, test: str, result: bool, message: str = "") -> None:
        """Log a test result"""
        if category not in self.results:
            self.results[category] = {}
        
        self.results[category][test] = {
            "success": result,
            "message": message
        }
        
        if result:
            status = "✅ PASS"
        else:
            status = "❌ FAIL"
        
        if message:
            print(f"{status} - {category} - {test}: {message}")
        else:
            print(f"{status} - {category} - {test}")
    
    def log_error(self, category: str, test: str, error: Exception) -> None:
        """Log an error during a test"""
        if category not in self.errors:
            self.errors[category] = {}
        
        self.errors[category][test] = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc()
        }
        
        print(f"❌ ERROR - {category} - {test}: {type(error).__name__}: {error}")
        if self.verbose:
            print(traceback.format_exc())
    
    def log_warning(self, category: str, test: str, message: str) -> None:
        """Log a warning during a test"""
        if category not in self.warnings:
            self.warnings[category] = {}
        
        self.warnings[category][test] = message
        
        print(f"⚠️ WARNING - {category} - {test}: {message}")
    
    def print_section_header(self, title: str) -> None:
        """Print a section header"""
        print(f"\n{'=' * 80}")
        print(f"  {title}")
        print(f"{'=' * 80}")
    
    def print_summary(self) -> None:
        """Print a summary of all test results"""
        self.print_section_header("DIAGNOSTIC SUMMARY")
        
        # Count total passes and failures
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        
        for category, tests in self.results.items():
            for test, result in tests.items():
                total_tests += 1
                if result["success"]:
                    passed_tests += 1
                else:
                    failed_tests += 1
        
        print(f"\nTotal tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Errors: {sum(len(tests) for tests in self.errors.values())}")
        print(f"Warnings: {sum(len(tests) for tests in self.warnings.values())}")
        
        # Print failed tests
        if failed_tests > 0:
            print("\nFailed tests:")
            for category, tests in self.results.items():
                for test, result in tests.items():
                    if not result["success"]:
                        print(f"- {category} - {test}: {result['message']}")
        
        # Print errors
        if self.errors:
            print("\nErrors:")
            for category, tests in self.errors.items():
                for test, error in tests.items():
                    print(f"- {category} - {test}: {error['error_type']}: {error['error_message']}")
        
        # Print warnings
        if self.warnings:
            print("\nWarnings:")
            for category, tests in self.warnings.items():
                for test, message in tests.items():
                    print(f"- {category} - {test}: {message}")
        
        # Print module import errors
        if self.module_errors:
            print("\nModule import errors:")
            for module, error in self.module_errors.items():
                print(f"- {module}: {error}")
    
    async def run_all_diagnostics(self) -> None:
        """Run all diagnostic tests"""
        self.print_section_header("WORKFLOW SYSTEM DIAGNOSTICS")
        print(f"Started: {datetime.now().isoformat()}")
        
        # 1. Check environment and dependencies
        await self.check_environment()
        
        # 2. Check core modules
        await self.check_core_modules()
        
        # 3. Check workflow modules
        await self.check_workflow_modules()
        
        # 4. Check LLM configuration
        await self.check_llm_configuration()
        
        # 5. Check agent components
        await self.check_agent_components()
        
        # 6. Check workflow graph
        await self.check_workflow_graph()
        
        # 7. Check API endpoints
        await self.check_api_endpoints()
        
        # Print summary
        self.print_summary()
    
    async def check_environment(self) -> None:
        """Check environment and dependencies"""
        self.print_section_header("CHECKING ENVIRONMENT")
        
        # Check Python version
        try:
            python_version = sys.version
            min_version = (3, 8)
            current_version = tuple(map(int, python_version.split('.')[0:2]))
            
            version_ok = current_version >= min_version
            self.log_result("Environment", "Python Version", version_ok, 
                           f"Python {python_version} (minimum required: {min_version[0]}.{min_version[1]})")
        except Exception as e:
            self.log_error("Environment", "Python Version", e)
        
        # Check for required packages
        required_packages = [
            "fastapi", "langchain", "langchain_community", "loguru", 
            "pydantic", "sqlalchemy", "uvicorn"
        ]
        
        for package in required_packages:
            try:
                # Try to import the package
                importlib.import_module(package)
                self.log_result("Dependencies", package, True)
            except ImportError as e:
                self.log_result("Dependencies", package, False, f"Could not import {package}")
                self.log_error("Dependencies", package, e)
        
        # Check if we're in a virtual environment
        in_venv = sys.prefix != sys.base_prefix
        self.log_result("Environment", "Virtual Environment", in_venv, 
                       "Running in a virtual environment" if in_venv else "Not running in a virtual environment")
        
        # Check for environment variables
        important_env_vars = [
            "OPENAI_API_KEY", "VLLM_MODEL_PATH", "MODEL_ENDPOINT", 
            "LANGCHAIN_TRACING_V2", "LANGCHAIN_ENDPOINT"
        ]
        
        for var in important_env_vars:
            exists = var in os.environ
            value = os.environ.get(var, "Not set")
            
            # Hide sensitive values like API keys
            if exists and "API_KEY" in var:
                masked_value = value[:4] + "..." + value[-4:] if len(value) > 8 else "***"
                self.log_result("Environment Variables", var, True, f"Set ({masked_value})")
            else:
                if exists:
                    self.log_result("Environment Variables", var, True, "Set")
                else:
                    # Some env vars are optional, so just log a warning
                    self.log_warning("Environment Variables", var, "Not set")
    
    async def safe_import(self, module_name: str) -> Tuple[bool, Any, Optional[str]]:
        """Safely import a module and return success status, module object, and error message"""
        try:
            module = importlib.import_module(module_name)
            return True, module, None
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            self.module_errors[module_name] = error_msg
            return False, None, error_msg
    
    async def check_core_modules(self) -> None:
        """Check core modules"""
        self.print_section_header("CHECKING CORE MODULES")
        
        core_modules = [
            "app.core.config",
            "app.core.queue_manager",
            "app.db.session",
            "app.api.deps"
        ]
        
        for module_name in core_modules:
            success, module, error = await self.safe_import(module_name)
            self.log_result("Core Modules", module_name, success, error if error else "")
            
            if success and self.verbose:
                # Try to get some info about the module
                try:
                    classes = [name for name, obj in inspect.getmembers(module, inspect.isclass) 
                              if obj.__module__ == module.__name__]
                    functions = [name for name, obj in inspect.getmembers(module, inspect.isfunction)
                                if obj.__module__ == module.__name__]
                    
                    if classes:
                        print(f"  Classes in {module_name}: {', '.join(classes)}")
                    if functions:
                        print(f"  Functions in {module_name}: {', '.join(functions)}")
                except Exception:
                    pass
    
    async def check_workflow_modules(self) -> None:
        """Check workflow modules"""
        self.print_section_header("CHECKING WORKFLOW MODULES")
        
        workflow_modules = [
            "app.agents.workflow.guest_workflow",
            "app.agents.workflow.initalize",
            "app.agents.workflow.base_workflow",
            "app.api.v1.guest_request_queue"
        ]
        
        for module_name in workflow_modules:
            success, module, error = await self.safe_import(module_name)
            self.log_result("Workflow Modules", module_name, success, error if error else "")
            
            if success:
                # Check for specific classes in the module
                if module_name == "app.agents.workflow.guest_workflow":
                    has_guest_workflow = hasattr(module, "GuestWorkflow")
                    self.log_result("Workflow Classes", "GuestWorkflow", has_guest_workflow,
                                   "Class found" if has_guest_workflow else "Class not found")
                
                elif module_name == "app.api.v1.guest_request_queue":
                    has_queue = hasattr(module, "GuestRequestQueue")
                    self.log_result("Workflow Classes", "GuestRequestQueue", has_queue,
                                   "Class found" if has_queue else "Class not found")
                
                # If verbose, show module contents
                if self.verbose:
                    try:
                        classes = [name for name, obj in inspect.getmembers(module, inspect.isclass) 
                                  if obj.__module__ == module.__name__]
                        
                        if classes:
                            print(f"  Classes in {module_name}: {', '.join(classes)}")
                    except Exception:
                        pass
    
    async def check_llm_configuration(self) -> None:
        """Check LLM configuration"""
        self.print_section_header("CHECKING LLM CONFIGURATION")
        
        # Try to import the initialize module
        success, init_module, error = await self.safe_import("app.agents.workflow.initalize")
        
        if not success:
            self.log_result("LLM Configuration", "Initialize Module", False, error)
            return
        
        # Check if LLM instances are defined
        has_llm_instance = hasattr(init_module, "llm_instance")
        has_llm_reasoning = hasattr(init_module, "llm_reasoning")
        
        self.log_result("LLM Configuration", "llm_instance", has_llm_instance,
                       "Defined" if has_llm_instance else "Not defined")
        self.log_result("LLM Configuration", "llm_reasoning", has_llm_reasoning,
                       "Defined" if has_llm_reasoning else "Not defined")
        
        # Check LLM instance details if available
        if has_llm_instance and init_module.llm_instance:
            llm = init_module.llm_instance
            llm_type = type(llm).__name__
            model_name = getattr(llm, "model_name", "Unknown")
            
            self.log_result("LLM Details", "Type", True, llm_type)
            self.log_result("LLM Details", "Model Name", True, model_name)
            
            # Check for specific LLM types
            is_vllm = "vllm" in llm_type.lower()
            if is_vllm:
                # Check if the class has the necessary streaming methods
                has_astream = hasattr(llm, "astream")
                self.log_result("VLLMOpenAI", "astream Method", has_astream,
                               "Method found" if has_astream else "Method not found")
    
    async def check_agent_components(self) -> None:
        """Check agent components"""
        self.print_section_header("CHECKING AGENT COMPONENTS")
        
        # Try to import the guest workflow
        success, workflow_module, error = await self.safe_import("app.agents.workflow.guest_workflow")
        
        if not success:
            self.log_result("Agent Components", "Guest Workflow Module", False, error)
            return
        
        # Check for the GuestWorkflow class
        has_guest_workflow = hasattr(workflow_module, "GuestWorkflow")
        
        if not has_guest_workflow:
            self.log_result("Agent Components", "GuestWorkflow Class", False, "Class not found")
            return
        
        # Create an instance of GuestWorkflow
        try:
            workflow = workflow_module.GuestWorkflow()
            self.log_result("Agent Components", "GuestWorkflow Instance", True, "Successfully created")
            
            # Check if the workflow has the necessary attributes
            has_agents = hasattr(workflow, "agents") and workflow.agents
            has_graph = hasattr(workflow, "graph") and workflow.graph
            has_cache = hasattr(workflow, "cache_manager")
            
            self.log_result("Workflow Attributes", "agents", has_agents,
                           f"Found {len(workflow.agents) if has_agents else 0} agents")
            self.log_result("Workflow Attributes", "graph", has_graph,
                           "Graph defined" if has_graph else "Graph not defined")
            self.log_result("Workflow Attributes", "cache_manager", has_cache,
                           "Cache manager defined" if has_cache else "Cache manager not defined")
            
            # Check individual agents
            if has_agents:
                for agent_name, agent in workflow.agents.items():
                    agent_type = type(agent).__name__
                    has_execute = hasattr(agent, "execute")
                    has_aexecute = hasattr(agent, "aexecute")
                    
                    if has_execute and has_aexecute:
                        self.log_result("Agents", agent_name, True, f"Type: {agent_type}")
                    else:
                        missing = []
                        if not has_execute:
                            missing.append("execute")
                        if not has_aexecute:
                            missing.append("aexecute")
                        
                        self.log_result("Agents", agent_name, False, 
                                       f"Type: {agent_type}, Missing methods: {', '.join(missing)}")
        
        except Exception as e:
            self.log_result("Agent Components", "GuestWorkflow Instance", False, "Failed to create")
            self.log_error("Agent Components", "GuestWorkflow Instance", e)
    
    async def check_workflow_graph(self) -> None:
        """Check workflow graph"""
        self.print_section_header("CHECKING WORKFLOW GRAPH")
        
        # Try to import the guest workflow
        success, workflow_module, error = await self.safe_import("app.agents.workflow.guest_workflow")
        
        if not success:
            self.log_result("Workflow Graph", "Guest Workflow Module", False, error)
            return
        
        # Check for the GuestWorkflow class
        has_guest_workflow = hasattr(workflow_module, "GuestWorkflow")
        
        if not has_guest_workflow:
            self.log_result("Workflow Graph", "GuestWorkflow Class", False, "Class not found")
            return
        
        # Create an instance of GuestWorkflow
        try:
            workflow = workflow_module.GuestWorkflow()
            graph = getattr(workflow, "graph", None)
            
            if not graph:
                self.log_result("Workflow Graph", "Graph Object", False, "Graph not defined")
                return
            
            # Check if the graph is properly compiled
            has_driver = hasattr(graph, "driver")
            self.log_result("Workflow Graph", "Driver", has_driver,
                          "Graph driver found" if has_driver else "Graph driver not found")
            
            if has_driver:
                # Check for nodes and edges
                try:
                    nodes = list(graph.driver.graph.nodes)
                    edges = list(graph.driver.graph.edges)
                    
                    self.log_result("Graph Structure", "Nodes", len(nodes) > 0,
                                   f"Found {len(nodes)} nodes")
                    self.log_result("Graph Structure", "Edges", len(edges) > 0,
                                   f"Found {len(edges)} edges")
                    
                    if self.verbose and nodes:
                        print("  Graph nodes:")
                        for node in nodes:
                            print(f"  - {node.name}")
                    
                    if self.verbose and edges:
                        print("  Graph edges:")
                        for edge in edges:
                            print(f"  - {edge.source} -> {edge.target}")
                    
                    # Check for specific nodes that should exist
                    important_nodes = ["entry", "rewriter", "reflection", "supervisor"]
                    for node_name in important_nodes:
                        node_exists = any(node.name == node_name for node in nodes)
                        self.log_result("Graph Nodes", node_name, node_exists,
                                       "Node exists" if node_exists else "Node not found")
                
                except Exception as e:
                    self.log_error("Graph Structure", "Nodes and Edges", e)
            
            # Check for important routing methods
            routing_methods = [
                "_route_after_entry",
                "_route_after_reflection",
            ]
            
            for method_name in routing_methods:
                has_method = hasattr(workflow, method_name)
                self.log_result("Routing Methods", method_name, has_method,
                              "Method found" if has_method else "Method not found")
        
        except Exception as e:
            self.log_result("Workflow Graph", "Graph Check", False, "Failed to check graph")
            self.log_error("Workflow Graph", "Graph Check", e)
    
    async def check_api_endpoints(self) -> None:
        """Check API endpoints"""
        self.print_section_header("CHECKING API ENDPOINTS")
        
        # Try to import the guest API
        success, guest_api_module, error = await self.safe_import("app.api.v1.guest_api")
        
        if not success:
            self.log_result("API Endpoints", "Guest API Module", False, error)
            return
        
        # Check for router object
        has_router = hasattr(guest_api_module, "router")
        self.log_result("API Endpoints", "Guest API Router", has_router,
                       "Router found" if has_router else "Router not found")
        
        # Check if request queue is used in the guest API
        try:
            with open(guest_api_module.__file__, "r") as f:
                content = f.read()
                uses_queue = "GuestRequestQueue" in content
                self.log_result("API Implementation", "Uses Request Queue", uses_queue,
                               "Request queue is used" if uses_queue else "Request queue not found in API")
                
                has_chat_endpoint = "/chat" in content
                self.log_result("API Endpoints", "Chat Endpoint", has_chat_endpoint,
                               "Chat endpoint found" if has_chat_endpoint else "Chat endpoint not found")
                
                has_streaming = "StreamingResponse" in content or "EventSourceResponse" in content
                self.log_result("API Features", "Streaming Support", has_streaming,
                               "Streaming support found" if has_streaming else "Streaming support not found")
        except Exception as e:
            self.log_error("API Implementation", "Check API File", e)
        
        # Try to import the guest request queue
        success, queue_module, error = await self.safe_import("app.api.v1.guest_request_queue")
        
        if success:
            # Check for queue class
            has_queue_class = hasattr(queue_module, "GuestRequestQueue")
            self.log_result("Request Queue", "GuestRequestQueue Class", has_queue_class,
                           "Class found" if has_queue_class else "Class not found")
            
            if has_queue_class:
                # Create an instance of the queue to check its methods
                try:
                    queue = queue_module.GuestRequestQueue()
                    
                    has_enqueue = hasattr(queue, "enqueue_request")
                    has_get_status = hasattr(queue, "get_status")
                    has_get_result = hasattr(queue, "get_result")
                    
                    self.log_result("Queue Methods", "enqueue_request", has_enqueue,
                                   "Method found" if has_enqueue else "Method not found")
                    self.log_result("Queue Methods", "get_status", has_get_status,
                                   "Method found" if has_get_status else "Method not found")
                    self.log_result("Queue Methods", "get_result", has_get_result,
                                   "Method found" if has_get_result else "Method not found")
                    
                    # Check if the queue has worker tasks
                    has_workers = hasattr(queue, "workers") and queue.workers
                    worker_count = len(queue.workers) if has_workers else 0
                    
                    self.log_result("Queue Implementation", "Worker Tasks", has_workers,
                                   f"Found {worker_count} workers" if has_workers else "No workers found")
                    
                except Exception as e:
                    self.log_result("Request Queue", "Queue Instance", False, "Failed to create")
                    self.log_error("Request Queue", "Queue Instance", e)
        else:
            self.log_result("Request Queue", "Queue Module", False, error)


async def main() -> None:
    """Main entry point for the diagnostic script"""
    diagnostics = WorkflowDiagnostic(verbose=args.verbose)
    await diagnostics.run_all_diagnostics()


if __name__ == "__main__":
    asyncio.run(main())
