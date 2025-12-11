# app/agents/workflow/employee_workflow.py

import asyncio
import sys
import time
import traceback
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, AsyncGenerator

from loguru import logger
from pathlib import Path

# LangGraph Imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver

# Agent and State Imports
from app.agents.workflow.state import GraphState as AgentState
from app.agents.workflow.initalize import llm_instance
from app.agents.stores.triage_guardrail_agent import TriageGuardrailAgent
from app.agents.stores.synthesizer_agent import SynthesizerAgent
from app.agents.stores.naive_agent import NaiveAgent # To be used as Fallback
# Import specialist agents for EMPLOYEES (NO CustomerAgent for security)
from app.agents.stores.employee_agent import EmployeeAgent
from app.agents.stores.company_agent import CompanyAgent
from app.agents.stores.product_agent import ProductAgent
from app.agents.stores.medical_agent import MedicalAgent
from app.agents.stores.drug_agent import DrugAgent
from app.agents.stores.genetic_agent import GeneticAgent
# A new, lightweight final agent
from app.agents.stores.final_answer_agent import FinalAnswerAgent 
from app.agents.stores.question_generator_agent import QuestionGeneratorAgent 
from app.agents.stores.cache_manager import CacheManager
from app.agents.stores.history_cache import HistoryCache
from app.agents.data_storages.response_storages import store_response


class EmployeeWorkflow:
    """
    An optimized, powerful, and secure workflow for employees.
    - Uses a TriageRouterAgent for intelligent, low-latency planning.
    - Follows adaptive paths based on query complexity.
    - Ensures no access to customer-specific agents or data.
    - Includes robust verification and fallback mechanisms.
    """
    def __init__(self):
        self.agents = self._initialize_agents()
        self.graph = self._build_and_compile_graph()
        self.cache_manager = CacheManager()  # Initialize cache manager
        self.history_cache = HistoryCache()  # Initialize enhanced memory cache
        
        log_path = Path("app/logs/log_workflows/employee_workflow_optimized.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(log_path, rotation="10 MB", level="DEBUG", backtrace=True, diagnose=True)
        logger.info("Optimized Employee Workflow initialized successfully.")

    def _initialize_agents(self) -> Dict[str, Any]:
        """
        Initializes agents for the SECURE Employee Workflow.
        *** DOES NOT INCLUDE CustomerAgent for security. ***
        """
        logger.info("Initializing agents for the optimized employee workflow...")
        
        # Use a reasoning-focused LLM for the critical triage step
        # Use a standard, potentially faster LLM for other tasks
        standard_llm = llm_instance

        return {
            "TriageGuardrailAgent": TriageGuardrailAgent(llm=standard_llm),
            "SynthesizerAgent": SynthesizerAgent(llm=standard_llm),
            "FinalAnswerAgent": FinalAnswerAgent(llm=standard_llm),
            "QuestionGeneratorAgent": QuestionGeneratorAgent(llm=standard_llm),
            "FallbackAgent": NaiveAgent(llm=standard_llm), # NaiveAgent serves as our fallback
            "DirectAnswerAgent": NaiveAgent(llm=standard_llm), # Can also use a simple LLM chain
            # All Specialist Agents available to employees (NO CustomerAgent)
            "EmployeeAgent": EmployeeAgent(llm=standard_llm),
            "CompanyAgent": CompanyAgent(llm=standard_llm, default_tool_names=["company_retriever_tool"]),
            "ProductAgent": ProductAgent(llm=standard_llm, default_tool_names=["product_retriever_tool"]),
            "MedicalAgent": MedicalAgent(llm=standard_llm, default_tool_names=["medical_retriever_tool"]),
            "DrugAgent": DrugAgent(llm=standard_llm, default_tool_names=["drug_retriever_tool"]),
            "GeneticAgent": GeneticAgent(llm=standard_llm, default_tool_names=["genetic_retriever_tool"]),
        }

    # --- Core Nodes of the Graph ---

    async def _triage_node(self, state: AgentState) -> AgentState:
        """1. The entry point that runs the planning agent for employees."""
        logger.info("--- (1) Executing Employee Triage Guardrail Node ---")
        logger.debug(f"Employee triage input state keys: {list(state.keys())}")
        
        agent = self.agents["TriageGuardrailAgent"]
        # Ensure the workflow type is set for context
        state['workflow_type'] = 'employee'
        
        try:
            result_state = await agent.aexecute(state)
            
            # Check for toxic content detection
            if result_state.get("next_step") == "toxic_content_block":
                logger.warning(f"Toxic content detected in employee query: {result_state.get('toxicity_reason', 'Unknown reason')}")
                return result_state
            
            # Debug logging to see what the triage agent returned
            logger.debug(f"Employee triage agent returned: next_step='{result_state.get('next_step')}', "
                        f"classified_agent='{result_state.get('classified_agent')}', "
                        f"should_re_execute={result_state.get('should_re_execute', False)}")
            
            # Validate and fix critical fields
            next_step = result_state.get("next_step")
            classified_agent = result_state.get("classified_agent")
            
            # Security check: Ensure the classified agent is in the allowed list for employees
            if classified_agent and classified_agent not in self.agents:
                logger.error(f"SECURITY/CONFIG ERROR: Triage selected agent '{classified_agent}' which is not available in EmployeeWorkflow.")
                result_state["classified_agent"] = "DirectAnswerAgent"
                result_state["next_step"] = "direct_answer"
                result_state["error_message"] = f"Access Denied: Agent '{classified_agent}' is not configured for employee use."
                return result_state
            
            if not next_step:
                logger.warning("Employee triage agent didn't set next_step! Attempting to infer from classified_agent...")
                if classified_agent == "DirectAnswerAgent":
                    result_state["next_step"] = "direct_answer"
                    logger.info("Set next_step to 'direct_answer' based on DirectAnswerAgent classification")
                elif classified_agent and classified_agent in self.agents:
                    result_state["next_step"] = "specialist_agent"
                    logger.info(f"Set next_step to 'specialist_agent' based on classified_agent: {classified_agent}")
                else:
                    result_state["next_step"] = "direct_answer"  # Safe fallback
                    logger.warning("Could not infer next_step, defaulting to 'direct_answer'")
            
            # Ensure classified_agent is set
            if not classified_agent:
                logger.warning("No classified_agent set by triage! Setting to DirectAnswerAgent as fallback.")
                result_state["classified_agent"] = "DirectAnswerAgent"
                if not next_step:
                    result_state["next_step"] = "direct_answer"
            
            logger.info(f"Final employee triage result: next_step='{result_state.get('next_step')}', "
                       f"classified_agent='{result_state.get('classified_agent')}'")
            
            return result_state
            
        except Exception as e:
            logger.error(f"Error in employee triage node: {e}", exc_info=True)
            # Return state with safe defaults
            state["next_step"] = "direct_answer"
            state["classified_agent"] = "DirectAnswerAgent"
            state["agent_response"] = f"I apologize, but I encountered an issue processing your request. Let me try to help you directly."
            return state

    async def _direct_answer_node(self, state: AgentState) -> AgentState:
        """2a. FAST PATH: For simple employee queries."""
        logger.info("--- (2a) Executing Employee Direct Answer Node (Fast Path) ---")
        agent = self.agents["DirectAnswerAgent"]
        # The Triage agent already classified this, so we just execute.
        return await agent.aexecute(state)

    async def _run_specialist_node(self, state: AgentState) -> AgentState:
        """2b. STANDARD PATH: Executes a specialist agent for an employee."""
        logger.info("--- (2b) Executing Employee Specialist Node ---")
        agent_name = state.get("classified_agent")
        
        # Security check: Ensure the classified agent is in the allowed list for employees
        if not agent_name or agent_name not in self.agents:
            logger.error(f"SECURITY/CONFIG ERROR: Triage selected agent '{agent_name}' which is not available in EmployeeWorkflow.")
            state['error_message'] = f"Access Denied: Agent '{agent_name}' is not configured for employee use."
            return state
        
        logger.info(f"Routing to employee specialist: {agent_name}")
        agent = self.agents[agent_name]
        return await agent.aexecute(state)

    async def _plan_executor_node(self, state: AgentState) -> AgentState:
        """2c. MULTI-AGENT PATH: For complex employee queries."""
        logger.info("--- (2c) Executing Employee Plan Executor Node (Multi-Agent Path) ---")
        logger.warning("Multi-agent plan execution is complex. Simulating with a single agent for now.")
        
        # In a full implementation, you would loop through a `state['plan']` list.
        # For this example, we just run the primary classified agent.
        result_state = await self._run_specialist_node(state)
        
        # Here you would aggregate results from all agents in the plan.
        # We'll just pass the current context forward.
        return result_state

    async def _verification_node(self, state: AgentState) -> AgentState:
        """3. Lightweight check on the specialist's output for an employee query."""
        logger.info("--- (3) Executing Employee Verification Node ---")
        agent_response = state.get("agent_response", "").lower()
        error_message = state.get("error_message")
        
        if error_message or not agent_response or "tôi không thể" in agent_response:
            logger.warning(f"Employee Verification FAILED. Reason: '{error_message or 'Unhelpful response'}'")
            state["is_final_answer"] = False
        else:
            logger.info("Employee Verification PASSED.")
            state["is_final_answer"] = True
        return state

    async def _fallback_agent_node(self, state: AgentState) -> AgentState:
        """4. Safety net for failed employee queries."""
        logger.warning("--- (4) Executing Employee Fallback Agent Node ---")
        agent = self.agents["FallbackAgent"]
        return await agent.aexecute(state)

    async def _synthesizer_node(self, state: AgentState) -> AgentState:
        """5. Combines results for a multi-agent employee plan."""
        logger.info("--- (5) Executing Employee Synthesizer Node ---")
        agent = self.agents["SynthesizerAgent"]
        return await agent.aexecute(state)

    async def _final_answer_node(self, state: AgentState) -> AgentState:
        """6. Final answer synthesis and streaming for an employee-facing response."""
        logger.info("--- (6) Executing Employee Final Answer Node ---")
        agent = self.agents["FinalAnswerAgent"]
        # The FinalAnswerAgent now handles comprehensive information synthesis and streaming internally
        return await agent.aexecute(state)
        
    async def store_feedback(self, interaction_id: str, feedback_type: str, rating: int = None, 
                           feedback_text: str = None, was_helpful: bool = None,
                           session_id: str = None, message_id: str = None) -> bool:
        """
        Store user feedback for an employee interaction
        
        Args:
            interaction_id: The UUID of the interaction
            feedback_type: The type of feedback ('like', 'dislike', 'neutral')
            rating: Optional numeric rating (1-5)
            feedback_text: Optional text feedback
            was_helpful: Optional boolean indicating if response was helpful
            session_id: Optional session identifier to associate feedback with a specific user session
            message_id: Optional message identifier to associate feedback with a specific message
            
        Returns:
            bool: True if feedback was successfully stored
        """
        try:
            # Log the feedback
            logger.info(f"Employee feedback received for {interaction_id}: {feedback_type}")
            
            # Store feedback in response storage if applicable
            if hasattr(store_response, 'store_feedback'):
                await store_response.store_feedback(
                    interaction_id=interaction_id,
                    user_type='employee',
                    feedback_type=feedback_type,
                    rating=rating,
                    feedback_text=feedback_text,
                    was_helpful=was_helpful,
                    session_id=session_id,
                    message_id=message_id
                )
            
            # The actual database update happens in the API layer
            return True
        except Exception as e:
            logger.error(f"Error storing employee feedback: {str(e)}")
            return False

    # --- Routing Logic ---

    def _route_from_triage(self, state: AgentState) -> str:
        """Directs the workflow based on the TriageGuardrailAgent's simplified output for employees."""
        # Extract simplified fields from the new guardrail structure
        next_step = state.get("next_step")
        classified_agent = state.get("classified_agent")
        need_analysis = state.get("need_analysis", False)
        is_toxic = state.get("is_toxic", False)
        
        # Comprehensive debug logging
        logger.info(f"Employee Triage Guardrail Router Decision: Routing to '{next_step}'")
        logger.debug(f"Simplified employee routing state - next_step: '{next_step}', classified_agent: '{classified_agent}', "
                    f"need_analysis: {need_analysis}, is_toxic: {is_toxic}")
        logger.debug(f"Employee state keys available: {list(state.keys())}")
        
        # Handle toxic content blocking first
        if is_toxic or next_step == "toxic_content_block":
            logger.warning("Routing toxic content directly to final answer for safety response")
            return "final_answer"
        
        # Handle need_analysis = True (requires clarification/context)
        if need_analysis:
            logger.info("Query needs analysis - routing to specialist agent with analysis context")
            return "specialist_agent"
        
        # Handle None or empty next_step
        if next_step is None or next_step == "":
            logger.warning(f"Employee next_step is None or empty! Attempting recovery based on classified_agent...")
            
            # Try to recover based on classified agent
            if classified_agent == "DirectAnswerAgent":
                logger.info("Found DirectAnswerAgent, routing to direct_answer")
                return "direct_answer"
            elif classified_agent and classified_agent in self.agents:
                logger.info(f"Found classified_agent '{classified_agent}', routing to specialist_agent")
                return "specialist_agent"
            else:
                logger.error("Cannot recover routing decision. Ending workflow.")
                return "END"
        
        # Handle standard routing based on next_step from simplified guardrail
        if next_step == "specialist_agent":
            logger.info(f"Routing to specialist agent: {classified_agent}")
            return "specialist_agent"
        elif next_step == "direct_answer":
            logger.info("Routing to direct answer for simple query")
            return "direct_answer"
        elif next_step == "need_analysis":
            logger.info("Query requires analysis - routing to specialist with analysis context")
            return "specialist_agent"
        
        # Legacy support for old routing paths (remove after full migration)
        elif next_step in ["multi_agent_plan", "plan_executor"]:
            logger.info("Multi-agent plan requested - routing to plan executor")
            return "plan_executor"
        elif next_step == "clarify_question":
            logger.info("Clarification question generated - ending workflow")
            return "END"
        
        # Final fallback: if no next_step but we have a classified_agent, use specialist
        if classified_agent and classified_agent in self.agents:
            logger.warning(f"Unrecognized next_step '{next_step}' but found classified_agent '{classified_agent}'. Routing to specialist_agent.")
            return "specialist_agent"
            
        logger.error(f"Invalid next_step from Employee Triage: '{next_step}'. All recovery attempts failed. Defaulting to END.")
        return "END"

    def _route_after_verification(self, state: AgentState) -> str:
        """Routes to the end or to a fallback based on answer quality."""
        if state.get("is_final_answer"):
            return "final_answer"
        else:
            return "fallback_agent"

    # --- Graph Construction ---

    def _build_and_compile_graph(self) -> StateGraph:
        """
        Builds and compiles the powerful, adaptive LangGraph workflow for employees.
        """
        workflow = StateGraph(AgentState)

        # Add all nodes to the graph
        workflow.add_node("triage_node", self._triage_node)
        workflow.add_node("direct_answer", self._direct_answer_node)
        workflow.add_node("specialist_agent", self._run_specialist_node)
        workflow.add_node("plan_executor", self._plan_executor_node)
        workflow.add_node("verification_node", self._verification_node)
        workflow.add_node("fallback_agent", self._fallback_agent_node)
        workflow.add_node("synthesizer_node", self._synthesizer_node)
        workflow.add_node("final_answer", self._final_answer_node)

        # Set the entry point
        workflow.set_entry_point("triage_node")

        # Define the main routing from the Triage node
        workflow.add_conditional_edges(
            "triage_node",
            self._route_from_triage,
            {
                "direct_answer": "direct_answer",
                "specialist_agent": "specialist_agent",
                "plan_executor": "plan_executor",
                "final_answer": "final_answer",  # For toxic content
                "END": END
            }
        )

        # Define path for the Fast Path
        workflow.add_edge("direct_answer", "final_answer")

        # Define path for the Standard Specialist Path
        workflow.add_edge("specialist_agent", "verification_node")
        workflow.add_conditional_edges(
            "verification_node",
            self._route_after_verification,
            {
                "final_answer": "final_answer",
                "fallback_agent": "fallback_agent"
            }
        )
        workflow.add_edge("fallback_agent", "final_answer")

        # Define path for the Multi-Agent Path
        workflow.add_edge("plan_executor", "synthesizer_node")
        workflow.add_edge("synthesizer_node", "final_answer")

        # Define the final exit point
        workflow.add_edge("final_answer", END)

        # Compile the graph
        logger.info("Compiling the optimized employee workflow graph.")
        return workflow.compile(checkpointer=InMemorySaver())

    # --- Execution Methods ---

    async def arun_streaming(self, query: str, chat_history: list = None) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Executes the optimized workflow and streams events for employees.
        """
        logger.info(f"--- Starting New Employee Workflow Run for Query: '{query}' ---")
        initial_state = AgentState(
            original_query=query,
            chat_history=chat_history or [],
            user_role="employee"  # Hardcode user_role for security
        )
        
        # Configuration for the stream
        config = {"configurable": {
            "thread_id": str(uuid.uuid4())
        }}

        async for event in self.graph.astream_events(initial_state, config=config, version="v1"):
            kind = event.get("event")
            
            if kind == "on_chain_start":
                node_name = event.get("name")
                yield {"event": "node_start", "data": {"node": node_name}}
            
            elif kind == "on_chain_stream":
                chunk = event.get("data", {}).get("chunk", {})
                if isinstance(chunk, dict) and "agent_response" in chunk:
                    logger.info(f"Chunking chunk from node: {event.get('name')} - {chunk.get('agent_response', '')[:50]}...")
                    yield {
                        "event": "final_stream_chunk",  # Changed to final_stream_chunk for frontend compatibility
                        "data": chunk.get("agent_response", "")
                    }
            
            elif kind == "on_chain_end" and event.get("name") == "final_answer":
                final_state = event.get("data", {}).get("output", {})
                final_result = {
                    "full_final_answer": final_state.get("agent_response", ""),
                    "suggested_questions": final_state.get("suggested_questions", [])
                }
                yield {"event": "final_result", "data": final_result}
            
            elif kind == "on_chain_error":
                yield {"event": "error", "data": {"error_message": str(event.get('data'))}}

    async def arun_streaming_authenticated(
        self, 
        query: str, 
        config: Dict, 
        employee_id: str,  # Employee ID as string for consistency
        employee_role: str = "employee",
        interaction_id: Optional[uuid.UUID] = None,
        chat_history: Optional[list] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Executes the optimized workflow for an authenticated employee, providing a rich,
        streaming experience while ensuring security (no access to customer data).
        """
        # 1. --- Input Validation and Setup ---
        if not isinstance(query, str) or not query.strip():
            logger.error("Invalid query provided to authenticated employee stream.")
            yield {"event": "error", "data": {"error": "Query cannot be empty."}}
            return
        
        # Ensure config is a dict for thread_id access
        if not isinstance(config, dict):
            config = {}
        if "configurable" not in config:
            config["configurable"] = {}
        
        start_time = time.time()
        interaction_id_str = str(interaction_id if interaction_id else uuid.uuid4())
        thread_id = config["configurable"].setdefault("thread_id", f"employee_auth_stream_{interaction_id_str}")
        config["configurable"]["thread_id"] = thread_id

        # 2. --- Cache Check for Employee ---
        if self.cache_manager.is_active():
            try:
                # Use a consistent key format for employees
                cache_key = self.cache_manager.create_cache_key(query, chat_history or [], context="employee_workflow")
                cached_result = await self.cache_manager.get(cache_key)
                
                if cached_result:
                    logger.info(f"[CACHE] HIT for authenticated employee query (employee: {employee_id}): {query[:50]}...")
                    # Stream the cached result directly
                    yield {"event": "final_result", "data": {**cached_result, "metadata": {"cache_hit": True}}}
                    return
            except Exception as cache_error:
                logger.error(f"[CACHE] Error during employee cache check: {cache_error}", exc_info=True)

        # 2.5. --- Enhanced Memory Management Integration ---
        session_context = None
        try:
            # Get or create session for the employee
            session_id = f"employee_{employee_id}_{thread_id}"
            
            # Get session context from enhanced memory (using current query for context)
            session_context = await self.cache_manager.get_session_context(session_id, query)
            if session_context:
                logger.info(f"[MEMORY] Retrieved session context for employee {employee_id}: {len(session_context)} chars")
            
            # Add current message to conversation history
            await self.history_cache.add_message(
                session_id=session_id,
                role="user",
                content=query,
                metadata={
                    "employee_id": str(employee_id),
                    "employee_role": employee_role,
                    "interaction_id": interaction_id_str,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
        except Exception as memory_error:
            logger.error(f"[MEMORY] Error during employee memory management: {memory_error}", exc_info=True)

        logger.info(f"Starting OPTIMIZED authenticated stream for employee {employee_id}")
        
        # 3. --- Initial State Creation ---
        initial_state = AgentState(
            original_query=query,
            iteration_count=0,
            chat_history=chat_history or [],
            user_role="employee",  # Hardcode user_role for security
            employee_id=str(employee_id),
            employee_role=employee_role,
            interaction_id=interaction_id_str,
            session_id=thread_id,
            timestamp=datetime.utcnow().isoformat()
        )
        
        # 4. --- Graph Streaming Execution ---
        try:
            # First send a starting message to confirm the stream is working
            yield {
                "event": "stream_start",
                "data": {"message": "Processing your query..."},
                "metadata": {
                    "employee_id": str(employee_id),
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
            
            full_answer = ""
            async for event in self.graph.astream_events(initial_state, config=config, version="v1"):
                kind = event["event"]
                node_name = event.get("name", "unknown")
                
                if kind == "on_chain_start":
                    yield {
                        "event": "node_start",
                        "data": {"node": node_name},
                        "metadata": {"employee_id": str(employee_id)}
                    }
                
                elif kind == "on_chain_stream":
                    chunk = event["data"]["chunk"]
                    logger.debug(f"Processing chunk from node: {node_name} - {chunk[:50]}...")
                    if isinstance(chunk, dict) and "agent_response" in chunk:
                        response = chunk.get("agent_response", "")
                        if response:
                            
                            logger.debug(f"Streaming checkpoint response: {response[:50]}...")
                            yield {
                                "event": "response_checkpoint",  # Special event type for checkpoints
                                "data": response,  # Send the whole response as a checkpoint
                                "metadata": {
                                    "employee_id": str(employee_id),
                                    "timestamp": datetime.utcnow().isoformat()
                                }
                            }
                            full_answer = response  # Update our tracking of the full answer
                    
                    # Also handle direct text chunks from the LLM
                    elif isinstance(chunk, str) and chunk.strip():
                        logger.debug(f"Streaming raw text chunk: {chunk[:50]}...")
                        yield {
                            "event": "token_chunk",  # Consistent event type for token-level streaming
                            "data": chunk,  # Send the raw chunk
                            "metadata": {
                                "employee_id": str(employee_id),
                                "timestamp": datetime.utcnow().isoformat()
                            }
                        }
                        
                elif kind == "on_chat_model_stream":
                    # logger.debug(f"Processing on_chat_model_stream event from: {node_name}")
                    chunk_text = event.get("data", {}).get("chunk", "")
                    if chunk_text and hasattr(chunk_text, 'content'):
                        chunk_content = chunk_text.content
                        if chunk_content:
                            # Only stream final answer chunks from the final_answer node
                            if node_name == "final_answer":
                                yield {
                                    "event": "token_chunk",  # Use token_chunk for real streaming
                                    "data": chunk_content,
                                    "metadata": {
                                        "employee_id": str(employee_id),
                                        "node": node_name,
                                        "is_final_answer": True,
                                        "timestamp": datetime.utcnow().isoformat()
                                    }
                                }
                            # For non-final nodes, DON'T stream individual chunks to avoid streaming intermediate processing
                
                # Final result is triggered by the 'final_answer' node
                elif kind == "on_chain_end" and node_name == "final_answer":
                    final_state = event["data"]["output"]
                    final_answer = final_state.get("agent_response", "")
                    
                    # --- Enhanced Memory Management: Store Response ---
                    try:
                        session_id = f"employee_{employee_id}_{thread_id}"
                        
                        # Store assistant response in conversation history
                        await self.history_cache.add_message(
                            session_id=session_id,
                            role="assistant",
                            content=final_answer,
                            metadata={
                                "employee_id": str(employee_id),
                                "employee_role": employee_role,
                                "interaction_id": interaction_id_str,
                                "processing_time": time.time() - start_time,
                                "timestamp": datetime.utcnow().isoformat(),
                                "agents_used": list(final_state.get("agent_thinks", {}).keys())
                            }
                        )
                        
                        # Get enhanced session insights
                        session_summary = await self.cache_manager.get_session_summary(session_id)
                        session_insights = await self.cache_manager.get_session_insights(session_id)
                        
                    except Exception as memory_error:
                        logger.error(f"[MEMORY] Error storing employee response in memory: {memory_error}", exc_info=True)
                        session_summary = None
                        session_insights = None
                    
                    final_result_data = {
                        "suggested_questions": final_state.get("suggested_questions", []),
                        "full_final_answer": final_answer,
                        "status": "success",
                        "agents_used": list(final_state.get("agent_thinks", {}).keys()),
                        "interaction_id": interaction_id_str,
                        "employee_id": str(employee_id),
                        "metadata": {
                            "cache_hit": False,
                            "processing_mode": "optimized_authenticated_employee_workflow",
                            "user_role": employee_role,
                            "is_private": True,
                            "processing_time": time.time() - start_time,
                            "timestamp": datetime.utcnow().isoformat(),
                            "session_context": session_context,
                            "session_summary": session_summary,
                            "session_insights": session_insights,
                            "has_enhanced_memory": True
                        }
                    }

                    # Cache the successful result
                    if self.cache_manager.is_active():
                        # Use the same key as the check at the beginning
                        cache_key = self.cache_manager.create_cache_key(query, chat_history or [], context="employee_workflow")
                        asyncio.create_task(self.cache_manager.set(cache_key, final_result_data, ttl=1800))
                        logger.info(f"Cached authenticated result for employee {employee_id}")

                    # Store response in database
                    try:
                        store_response(final_state)
                    except Exception as store_error:
                        logger.error(f"Error storing employee final response: {store_error}", exc_info=True)
                    
                    yield {"event": "final_result", "data": final_result_data}

                elif kind == "on_chain_error":
                    error_data = event.get("data", {})
                    error_message = str(error_data.get("error", "Unknown error"))
                    logger.error(f"Chain error in employee node {node_name} for employee {employee_id}: {error_message}")
                    
                    # Check for LLM connection errors and provide better messages
                    user_message = error_message
                    if "Connection error" in error_message:
                        user_message = "There was an issue connecting to the AI service. Please try again."
                    
                    yield {
                        "event": "error",
                        "data": {
                            "error": user_message, 
                            "technical_error": error_message,
                            "node": node_name
                        },
                        "metadata": {"employee_id": str(employee_id)}
                    }
                    
                    # Send a fallback response so the user isn't left hanging
                    yield {
                        "event": "final_stream_chunk",  # Changed to final_stream_chunk for frontend compatibility
                        "data": "I apologize, but I'm having trouble processing your request at the moment. Please try again shortly.",
                        "metadata": {
                            "employee_id": str(employee_id),
                            "timestamp": datetime.utcnow().isoformat(),
                            "is_fallback": True
                        }
                    }
        
        except Exception as stream_error:
            error_msg = f"Internal streaming error for employee {employee_id}: {stream_error}"
            logger.error(error_msg, exc_info=True)
            
            # Send a user-friendly error message
            yield {
                "event": "error", 
                "data": {
                    "error": "Sorry, there was an unexpected error processing your request.",
                    "technical_error": error_msg
                },
                "metadata": {"employee_id": str(employee_id)}
            }
            
            # Also send a fallback response
            yield {
                "event": "final_stream_chunk",  # Changed to final_stream_chunk for frontend compatibility
                "data": "I apologize for the inconvenience. Our system encountered an issue while processing your request. Please try again in a moment.",
                "metadata": {
                    "employee_id": str(employee_id),
                    "timestamp": datetime.utcnow().isoformat(),
                    "is_fallback": True
                }
            }

    async def _direct_answer_node(self, state: AgentState) -> AgentState:
        """2a. FAST PATH: For simple employee queries."""
        logger.info("--- (2a) Executing Employee Direct Answer Node ---")
        agent = self.agents["DirectAnswerAgent"]
        return await agent.aexecute(state)

    async def _run_specialist_node(self, state: AgentState) -> AgentState:
        """2b. STANDARD PATH: Executes a specialist agent for an employee."""
        logger.info("--- (2b) Executing Employee Specialist Node ---")
        agent_name = state.get("classified_agent")
        
        # Security check: Ensure the classified agent is in the allowed list for employees
        if not agent_name or agent_name not in self.agents:
            logger.error(f"SECURITY/CONFIG ERROR: Triage selected agent '{agent_name}' which is not available in EmployeeWorkflow.")
            state['error_message'] = f"Access Denied: Agent '{agent_name}' is not configured for employee use."
            return state
        
        logger.info(f"Routing to employee specialist: {agent_name}")
        agent = self.agents[agent_name]
        return await agent.aexecute(state)

    async def _plan_executor_node(self, state: AgentState) -> AgentState:
        """2c. MULTI-AGENT PATH: For complex employee queries."""
        logger.info("--- (2c) Executing Employee Plan Executor Node ---")
        # Placeholder for multi-step logic, same as customer workflow
        return await self._run_specialist_node(state)

    async def _verification_node(self, state: AgentState) -> AgentState:
        """3. Lightweight check on the specialist's output for an employee query."""
        logger.info("--- (3) Executing Employee Verification Node ---")
        agent_response = state.get("agent_response", "").lower()
        error_message = state.get("error_message")
        
        if error_message or not agent_response or "tôi không thể" in agent_response:
            logger.warning(f"Employee Verification FAILED. Reason: '{error_message or 'Unhelpful response'}'")
            state["is_final_answer"] = False
        else:
            logger.info("Employee Verification PASSED.")
            state["is_final_answer"] = True
        return state

    async def _fallback_agent_node(self, state: AgentState) -> AgentState:
        """4. Safety net for failed employee queries."""
        logger.warning("--- (4) Executing Employee Fallback Agent Node ---")
        agent = self.agents["FallbackAgent"]
        return await agent.aexecute(state)

    async def _synthesizer_node(self, state: AgentState) -> AgentState:
        """5. Combines results for a multi-agent employee plan."""
        logger.info("--- (5) Executing Employee Synthesizer Node ---")
        agent = self.agents["SynthesizerAgent"]
        return await agent.aexecute(state)

    async def _final_answer_node(self, state: AgentState) -> AgentState:
        """6. Final polishing for an employee-facing answer."""
        logger.info("--- (6) Executing Employee Final Answer Node ---")
        agent = self.agents["FinalAnswerAgent"]
        return await agent.aexecute(state)

    # --- Routing Logic (Identical structure to CustomerWorkflow) ---

    def _route_from_triage(self, state: AgentState) -> str:
        """Directs the workflow based on the TriageGuardrailAgent's simplified output for employees."""
        # Extract simplified fields from the new guardrail structure
        next_step = state.get("next_step")
        classified_agent = state.get("classified_agent")
        need_analysis = state.get("need_analysis", False)
        is_toxic = state.get("is_toxic", False)
        
        logger.info(f"Employee Triage Router Decision: Routing to '{next_step}'")
        logger.debug(f"Simplified employee routing - next_step: '{next_step}', agent: '{classified_agent}', "
                    f"need_analysis: {need_analysis}, is_toxic: {is_toxic}")
        
        # Handle toxic content blocking first
        if is_toxic or next_step == "toxic_content_block":
            logger.warning("Routing toxic content directly to final answer for safety response")
            return "final_answer"
        
        # Handle need_analysis = True (requires clarification/context)
        if need_analysis:
            logger.info("Query needs analysis - routing to specialist agent with analysis context")
            return "specialist_agent"
        
        # Handle standard routing based on next_step from simplified guardrail
        if next_step == "specialist_agent":
            logger.info(f"Routing to specialist agent: {classified_agent}")
            return "specialist_agent"
        elif next_step == "direct_answer":
            logger.info("Routing to direct answer for simple query")
            return "direct_answer"
        elif next_step == "need_analysis":
            logger.info("Query requires analysis - routing to specialist with analysis context")
            return "specialist_agent"
        
        # Legacy support for old routing paths
        elif next_step in ["multi_agent_plan", "plan_executor"]:
            logger.info("Multi-agent plan requested - routing to plan executor")
            return "plan_executor"
        elif next_step == "clarify_question":
            logger.info("Clarification question generated - routing to final answer")
            return "final_answer"
        
        # Final fallback: if no next_step but we have a classified_agent, use specialist
        if classified_agent and classified_agent in self.agents:
            logger.warning(f"Unrecognized next_step '{next_step}' but found classified_agent '{classified_agent}'. Routing to specialist_agent.")
            return "specialist_agent"
            
        logger.error(f"Invalid next_step from Employee Triage: '{next_step}'. All recovery attempts failed. Defaulting to END.")
        return "END"

    def _route_after_verification(self, state: AgentState) -> str:
        return "final_answer" if state.get("is_final_answer") else "fallback_agent"

    # --- Graph Construction (Identical structure to CustomerWorkflow) ---

    def _build_and_compile_graph(self) -> StateGraph:
        """Builds and compiles the optimized graph for the Employee Workflow."""
        workflow = StateGraph(AgentState)
        workflow.add_node("triage_node", self._triage_node)
        workflow.add_node("direct_answer", self._direct_answer_node)
        workflow.add_node("specialist_agent", self._run_specialist_node)
        workflow.add_node("plan_executor", self._plan_executor_node)
        workflow.add_node("verification_node", self._verification_node)
        workflow.add_node("fallback_agent", self._fallback_agent_node)
        workflow.add_node("synthesizer_node", self._synthesizer_node)
        workflow.add_node("final_answer", self._final_answer_node)
        workflow.set_entry_point("triage_node")
        workflow.add_conditional_edges("triage_node", self._route_from_triage)
        workflow.add_edge("direct_answer", "final_answer")
        workflow.add_edge("specialist_agent", "verification_node")
        workflow.add_conditional_edges("verification_node", self._route_after_verification)
        workflow.add_edge("fallback_agent", "final_answer")
        workflow.add_edge("plan_executor", "synthesizer_node")
        workflow.add_edge("synthesizer_node", "final_answer")
        workflow.add_edge("final_answer", END)
        logger.info("Compiling the optimized Employee workflow graph.")
        return workflow.compile(checkpointer=InMemorySaver())

    # --- Execution Method ---

    # async def arun_streaming_authenticated(
    #     self, 
    #     query: str, 
    #     config: Dict, 
    #     employee_id: str,
    #     employee_role: str = "employee",
    #     chat_history: Optional[list] = None
    # ) -> AsyncGenerator[Dict[str, Any], None]:
    #     """
    #     Executes the optimized workflow for an authenticated employee,
    #     providing a rich, streaming experience.
    #     """
    #     if not isinstance(config, dict): config = {}
    #     if "configurable" not in config: config["configurable"] = {}
    #     config["configurable"].setdefault("thread_id", f"employee_stream_{employee_id}_{time.time()}")
        
    #     logger.info(f"--- Starting OPTIMIZED Employee Workflow for Employee {employee_id} ---")
        
    #     # Caching logic specific to employees
    #     if self.cache_manager.is_active():
    #         cache_key = self.cache_manager.create_cache_key(query, chat_history or [], context="employee_workflow")
    #         cached_result = await self.cache_manager.get(cache_key)
    #         if cached_result:
    #             logger.info(f"[CACHE] HIT for employee query: {query[:50]}...")
    #             yield {"event": "final_result", "data": {**cached_result, "metadata": {"cache_hit": True}}}
    #             return
        
    #     initial_state = AgentState(
    #         original_query=query,
    #         chat_history=chat_history or [],
    #         employee_id=employee_id,
    #         employee_role=employee_role,
    #         user_role="employee", # Hardcode user_role for security
    #     )
        
    #     # Stream events from the new graph
    #     async for event in self.graph.astream_events(initial_state, config=config, version="v1"):
    #         kind = event.get("event")
    #         node_name = event.get("name")
            
    #         if kind == "on_chain_start":
    #             yield {"event": "node_start", "data": {"node": node_name}}
            
    #         elif kind == "on_chain_stream":
    #             chunk = event.get("data", {}).get("chunk", {})
    #             if isinstance(chunk, dict) and "agent_response" in chunk:
    #                 yield {"event": "final_stream_chunk", "data": chunk.get("agent_response", "")}  # Changed to final_stream_chunk for frontend compatibility
            
    #         elif kind == "on_chain_end" and node_name == "final_answer":
    #             final_state = event.get("data", {}).get("output", {})
    #             final_result = {
    #                 "full_final_answer": final_state.get("agent_response", ""),
    #                 "suggested_questions": final_state.get("suggested_questions", []),
    #                 "employee_id": final_state.get("employee_id")
    #             }
                
    #             # Store and cache the result
    #             store_response(final_state)
    #             if self.cache_manager.is_active():
    #                 cache_key = self.cache_manager.create_cache_key(query, chat_history or [], context="employee_workflow")
    #                 asyncio.create_task(self.cache_manager.set(cache_key, final_result, ttl=1800))

    #             yield {"event": "final_result", "data": final_result}
            
    #         elif kind == "on_chain_error":
    #             yield {"event": "error", "data": {"error_message": str(event.get('data')), "node": node_name}}
    
    def _calculate_processing_time(self, initial_state: AgentState) -> float:
        """Calculate total processing time for the workflow"""
        start_time_str = initial_state.get("timestamp")
        if not start_time_str:
            return 0.0
        try:
            start_time = float(start_time_str)
            end_time = time.time()
            return end_time - start_time
        except Exception as e:
            logger.error(f"Error calculating processing time: {e}")
            return 0.0
            
    async def astreaming_workflow(self, query: str, config: Dict = None, 
                               employee_id: str = None, employee_role: str = "employee",
                               interaction_id: Optional[uuid.UUID] = None,
                               chat_history: Optional[list] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Direct streaming workflow that uses the LangGraph astream_events directly.
        Focuses on streaming events from the final_answer node and returns final results.
        
        Args:
            query: The employee's query text
            config: Configuration dictionary for the workflow
            employee_id: ID of the employee
            employee_role: Role of the employee (default: "employee") 
            interaction_id: Optional UUID for tracking the interaction
            chat_history: Optional chat history for context
            
        Yields:
            Dict[str, Any]: Streaming events or final result
        """
        logger.info(f"Starting direct graph streaming for employee query: {query[:100]}...")
        
        # Initialize tracking variables
        final_answer = ""
        suggested_questions = []
        
        # Generate IDs if not provided
        if employee_id is None:
            employee_id = f"employee_{uuid.uuid4().hex[:8]}"
            
        if interaction_id is None:
            interaction_id = uuid.uuid4()
        
        # Ensure config is properly structured
        if not isinstance(config, dict):
            config = {}
        if "configurable" not in config:
            config["configurable"] = {}
        
        thread_id = config["configurable"].setdefault("thread_id", f"employee_stream_{interaction_id}")
        
        # Prepare initial state
        initial_state = AgentState(
            original_query=query,
            iteration_count=0,
            chat_history=chat_history or [],
            user_role=employee_role,
            employee_id=employee_id,
            interaction_id=str(interaction_id),
            session_id=thread_id,
            timestamp=str(time.time())  # Using time format expected by employee workflow
        )
        
        try:
            # Stream events DIRECTLY from the graph (not using other arun methods)
            async for event in self.graph.astream_events(initial_state, config=config, version="v1"):
                # Focus on the 'kind' of event
                kind = event.get("event")
                if not kind:
                    continue
                
                # Check for node name in metadata
                metadata = event.get("metadata", {})
                node_name = metadata.get("langgraph_node", "unknown")
                
                # STEP 1: Check if we're in a streaming event kind
                if kind in ["on_chat_model_stream", "on_chain_stream", "token_chunk"]:
                    # If in streaming event for final_answer node, yield streaming tokens
                    if node_name == "final_answer":
                        token = None
                        
                        # Extract token from different streaming event formats
                        if kind == "on_chat_model_stream":
                            chunk = event.get("data", {}).get("chunk")
                            if chunk and hasattr(chunk, "content"):
                                token = chunk.content
                                yield {
                                "event": "token_chunk",
                                "data": token,
                                "node": "final_answer"
                            }
                            final_answer += token
                        elif kind == "on_chain_stream":
                            chunk = event.get("data", {}).get("chunk")
                            if chunk and hasattr(chunk, "content") and isinstance(chunk.content, str):
                                token = chunk.content
                        elif kind == "token_chunk":
                            token = event.get("data", "")
                        
                        # If we have a token, yield it as a streaming chunk
                        # if token:
                        #     final_answer += token
                        #     yield {
                        #         "event": "streaming",
                        #         "data": token,
                        #         "node": "final_answer"
                        #     }
                
                # Collect agent_response from final_answer node completion
                elif kind == "on_chain_end" and node_name == "final_answer":
                    output = event.get("data", {}).get("output", {})
                    
                    # Handle different output formats
                    if hasattr(output, "content"):
                        final_answer = output.content
                    elif isinstance(output, dict) and "agent_response" in output:
                        final_answer = output["agent_response"]
                
                # Collect suggested_questions from question_generator node
                elif kind == "on_chain_end" and node_name == "question_generator":
                    output = event.get("data", {}).get("output", {})
                    if isinstance(output, dict) and "suggested_questions" in output:
                        suggested_questions = output["suggested_questions"]
            
            # FINAL STEP: Return the final output state with both required fields
            final_state = {
                "agent_response": final_answer,
                "suggested_questions": suggested_questions or []
            }
            
            yield {
                "event": "final_result",
                "data": final_state
            }
                    
        except Exception as e:
            logger.error(f"Error in direct graph streaming workflow: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            
            # Yield error state with basic information
            yield {
                "event": "error",
                "data": {
                    "message": f"Error processing your request: {str(e)}",
                    "agent_response": "I'm sorry, I encountered an error while processing your request.",
                    "suggested_questions": ["Can you try a simpler question?"]
                }
            }

    async def arun_simple(
        self, 
        query: str, 
        employee_id: str,
        employee_role: str = "employee",
        session_id: Optional[str] = None,
        chat_history: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Run enhanced workflow for employee and return final result with sentiment analysis support.
        """
        if chat_history is None:
            chat_history = []
        start_time = time.time()
        logger.info(f"[EMPLOYEE_SIMPLE] Starting enhanced simple workflow for employee {employee_id} with query: {query[:100]}...")
        
        try:
            # Check cache first for faster responses
            if self.cache_manager.is_active():
                try:
                    # Create employee-specific cache key
                    employee_query = self._create_employee_cache_query(employee_id, employee_role, query)
                    cache_key = self.cache_manager.create_cache_key(employee_query, chat_history, context="employee_workflow")
                    logger.debug(f"[CACHE] Checking enhanced cache with key: {cache_key}")
                    
                    cached_result = await self._check_and_return_cached_result(cache_key, query, employee_id, employee_role, session_id, chat_history)
                    if cached_result:
                        cached_result["metadata"]["processing_time"] = time.time() - start_time
                        return cached_result
                        
                except Exception as cache_error:
                    logger.error(f"[CACHE] Error checking enhanced cache: {str(cache_error)}\n{traceback.format_exc()}")

            # If cache miss or error, proceed with enhanced execution
            logger.info(f"[EMPLOYEE_SIMPLE] Proceeding with enhanced workflow execution")
            if not session_id:
                session_id = f"enhanced_employee_{employee_id}_{uuid.uuid4().hex}"
            config = {"configurable": {"thread_id": session_id}}
            
            # Reuse the enhanced streaming logic to collect the final result
            final_result = {}
            full_answer = ""
            error_occurred = False
            
            try:
                async for event in self.arun_streaming(query, config, employee_id, employee_role, chat_history):
                    try:
                        event_type = event.get("event")
                        logger.debug(f"[EMPLOYEE_SIMPLE] Processing enhanced event: {event_type}")
                        
                        if event_type == "final_stream_chunk":
                            chunk_data = event.get("data", "")
                            full_answer += chunk_data
                            logger.debug(f"[EMPLOYEE_SIMPLE] Accumulated final stream chunk: {len(chunk_data)} chars")
                            
                        elif event_type == "sentiment_analysis_result":
                            # Log sentiment analysis results for simple execution
                            sentiment_data = event.get("data", {})
                            logger.info(f"[EMPLOYEE_SIMPLE] Sentiment Analysis - Intent: {sentiment_data.get('user_intent')}, "
                                       f"Re-execute: {sentiment_data.get('should_re_execute')}")
                            
                        elif event_type == "re_execution_complete":
                            logger.info(f"[EMPLOYEE_SIMPLE] Re-execution completed for employee query")
                            
                        elif event_type == "final_result":
                            final_result = event.get("data", {}).copy()
                            final_result["full_answer"] = full_answer
                            final_result["employee_id"] = employee_id
                            final_result["session_id"] = session_id
                            processing_time = time.time() - start_time
                            logger.info(f"[EMPLOYEE_SIMPLE] Enhanced final result ready, processing time: {processing_time:.3f}s")
                            
                            # Update metadata
                            if "metadata" not in final_result:
                                final_result["metadata"] = {}
                            final_result["metadata"]["processing_time"] = processing_time
                            
                            # Cache the enhanced successful result
                            if self.cache_manager.is_active() and final_result.get("status") != "error":
                                try:
                                    employee_query = self._create_employee_cache_query(employee_id, employee_role, query)
                                    cache_key = self.cache_manager.create_cache_key(employee_query, chat_history, context="employee_workflow")
                                    asyncio.create_task(self._cache_result(cache_key, final_result, query))
                                    logger.info(f"[CACHE] Initiated caching for enhanced employee simple result: {query[:50]}...")
                                except Exception as cache_store_error:
                                    logger.error(f"[CACHE] Error storing enhanced result in cache: {str(cache_store_error)}\n{traceback.format_exc()}")
                            break
                            
                        elif event_type == "error":
                            error_data = event.get("data", {})
                            error_msg = error_data.get("error", "Unknown error")
                            logger.error(f"[EMPLOYEE_SIMPLE] Enhanced error event received: {error_msg}")
                            final_result = {
                                "status": "error",
                                "error": error_msg,
                                "details": error_data.get("details", ""),
                                "employee_id": employee_id,
                                "session_id": session_id,
                                "processing_time": time.time() - start_time,
                                "metadata": {
                                    "cache_hit": False,
                                    "processing_mode": "enhanced_employee_simple_workflow",
                                    "employee_role": employee_role,
                                    "is_employee_query": True,
                                    "timestamp": datetime.utcnow().isoformat(),
                                    "has_sentiment_analysis": False
                                }
                            }
                            error_occurred = True
                            break
                            
                    except Exception as event_error:
                        logger.error(f"[EMPLOYEE_SIMPLE] Error processing enhanced event {event.get('event', 'unknown')}: {str(event_error)}\n{traceback.format_exc()}")
                        continue
                        
            except Exception as stream_error:
                logger.error(f"[EMPLOYEE_SIMPLE] Error in enhanced streaming workflow: {str(stream_error)}\n{traceback.format_exc()}")
                final_result = {
                    "status": "error",
                    "error": "Enhanced streaming workflow error",
                    "details": str(stream_error),
                    "employee_id": employee_id,
                    "session_id": session_id,
                    "processing_time": time.time() - start_time,
                    "metadata": {
                        "cache_hit": False,
                        "processing_mode": "enhanced_employee_simple_workflow",
                        "employee_role": employee_role,
                        "is_employee_query": True,
                        "timestamp": datetime.utcnow().isoformat(),
                        "has_sentiment_analysis": False
                    }
                }
            
            # Return result or default error result
            if not final_result:
                final_result = {
                    "status": "error",
                    "error": "No result received from enhanced workflow",
                    "employee_id": employee_id,
                    "session_id": session_id,
                    "processing_time": time.time() - start_time,
                    "metadata": {
                        "cache_hit": False,
                        "processing_mode": "enhanced_employee_simple_workflow",
                        "employee_role": employee_role,
                        "is_employee_query": True,
                        "timestamp": datetime.utcnow().isoformat(),
                        "has_sentiment_analysis": False
                    }
                }
            return final_result
            
        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"[EMPLOYEE_SIMPLE] Critical error in enhanced employee simple workflow: {str(e)}\n{traceback.format_exc()}")
            return {
                "status": "error",
                "error": "Critical enhanced workflow error",
                "details": str(e),
                "employee_id": employee_id,
                "session_id": session_id,
                "processing_time": total_time,
                "metadata": {
                    "cache_hit": False,
                    "processing_mode": "enhanced_employee_simple_workflow",
                    "employee_role": employee_role,
                    "is_employee_query": True,
                    "timestamp": datetime.utcnow().isoformat(),
                    "has_sentiment_analysis": False
                }
            }

    async def arun_streaming_authenticated(
        self, 
        query: str, 
        config: Dict, 
        employee_id: int,  # Explicitly typed as int (from API layer)
        employee_role: str = "employee",
        interaction_id: Optional[uuid.UUID] = None,
        chat_history: Optional[list] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Run enhanced workflow for authenticated employee and stream events with sentiment analysis.
        """
        # Convert employee_id to string for internal consistency within the workflow
        employee_id_str = str(employee_id)
        
        try:
            # Validate inputs to catch errors early
            if not isinstance(query, str) or not query.strip():
                logger.error("Invalid query: empty or not a string")
                yield {"event": "error", "data": {"error": "Invalid query"}}
                return
                
            if not config or not isinstance(config, dict):
                logger.error(f"Invalid config: {config}")
                config = {"configurable": {}}
            
            # Check cache first for faster responses
            if self.cache_manager.is_active():
                try:
                    # Create employee-specific cache key
                    employee_query = self._create_employee_cache_query(employee_id_str, employee_role, query)
                    cache_key = self.cache_manager.create_cache_key(employee_query, chat_history or [], context="employee_workflow")
                    logger.debug(f"[CACHE] Checking enhanced authenticated streaming cache with key: {cache_key}")
                    
                    cached_result = await self.cache_manager.get(cache_key)
                    if cached_result:
                        logger.info(f"[CACHE] Cache HIT for enhanced authenticated employee streaming query: {query[:50]}...")
                        # Stream from cache with sentiment data
                        async for event in self._stream_from_cache(cached_result, employee_id_str, employee_role, config.get("configurable", {}).get("thread_id"), str(interaction_id) if interaction_id else None):
                            yield event
                        return
                        
                except Exception as cache_error:
                    logger.error(f"[CACHE] Error checking enhanced cache for authenticated streaming: {str(cache_error)}\n{traceback.format_exc()}")

            # Log detailed information for debugging
            logger.info(f"Starting enhanced authenticated workflow stream for employee {employee_id}")
            logger.debug(f"Query: '{query}', Config: {config}, Role: {employee_role}")
            
            # Setup enhanced initial state with sentiment fields
            initial_state = AgentState(
                original_query=query,
                iteration_count=0,
                chat_history=chat_history if chat_history else [],
                user_role="employee",
                employee_id=employee_id_str,
                employee_role=employee_role or "employee",
                interaction_id=str(interaction_id) if interaction_id else None,
                session_id=config.get("configurable", {}).get("thread_id", ""),
                timestamp=str(time.time()),
                # Initialize sentiment-related fields
                needs_re_execution=False,
                sentiment_analysis={},
                is_re_execution=False,
                was_re_executed=False
            )
            
            # Ensure we have a valid graph
            if not hasattr(self, 'graph') or self.graph is None:
                logger.error("Enhanced workflow graph not initialized")
                yield {"event": "error", "data": {"error": "Enhanced workflow not properly initialized"}}
                return
            
            # Stream events from the enhanced graph
            async for event in self._stream_events_from_graph(initial_state, config, employee_id_str, employee_role):
                
                # Handle final result storage and caching
                if event.get("event") == "final_result":
                    final_data = event.get("data", {})
                    try:
                        store_response(final_data)
                    except Exception as store_error:
                        logger.error(f"Error storing enhanced response: {store_error}")
                    
                    # Cache the enhanced successful result
                    if self.cache_manager.is_active() and final_data.get("status") != "error":
                        try:
                            employee_query = self._create_employee_cache_query(employee_id_str, employee_role, query)
                            cache_key = self.cache_manager.create_cache_key(employee_query, chat_history or [], context="employee_workflow")
                            asyncio.create_task(self._cache_result(cache_key, final_data, query))
                            logger.info(f"[CACHE] Initiated caching for enhanced authenticated employee streaming result: {query[:50]}...")
                        except Exception as cache_error:
                            logger.error(f"[CACHE] Error initiating enhanced cache storage: {str(cache_error)}\n{traceback.format_exc()}")
                
                yield event
                
        except Exception as e:
            logger.exception(f"Critical error in enhanced workflow streaming: {str(e)}")
            yield {
                "event": "error",
                "data": {"error": f"Enhanced workflow error: {str(e)}"}
            }

    async def process_with_load_balancing(
        self, 
        query: str, 
        employee_id: str,
        employee_role: str = "employee",
        session_id: Optional[str] = None,
        prioritize: bool = False
    ) -> Dict[str, Any]:
        """
        Process an enhanced employee query with load balancing support and sentiment analysis.
        """
        try:
            # Try to import load balancer and queue manager
            try:
                from app.core.load_balancer import get_load_balancer
                from app.core.queue_manager import get_queue_manager
                load_balancer = get_load_balancer()
                queue_manager = get_queue_manager()
            except ImportError:
                load_balancer = None
                queue_manager = None
                logger.warning("Load balancer or queue manager not available - processing enhanced workflow locally")
            
            # Create session ID if not provided
            if not session_id:
                session_id = f"enhanced_employee_{employee_id}_{uuid.uuid4().hex}"
            
            # Check if we should process this request on another node
            if load_balancer and load_balancer.is_active():
                local_load = await load_balancer.get_local_load()
                
                if local_load.get('cpu_percent', 0) > 70 or local_load.get('memory_percent', 0) > 80:
                    logger.info(f"Local system load is high: {local_load}. Trying to find another node for enhanced processing.")
                    
                    best_node = await load_balancer.get_best_node(request_type="enhanced_employee_query")
                    if best_node and best_node['id'] != load_balancer.get_node_id():
                        logger.info(f"Forwarding enhanced request to node: {best_node['id']}")
                        
                        response = await load_balancer.forward_request(
                            node=best_node,
                            endpoint="/api/v1/employee/chat",
                            method="POST",
                            payload={
                                "query": query,
                                "employee_id": employee_id,
                                "employee_role": employee_role,
                                "session_id": session_id,
                                "enhanced_workflow": True  # Flag for enhanced processing
                            }
                        )
                        
                        if response and 'result' in response:
                            return response['result']
                            
                        logger.warning(f"Failed to get valid response from enhanced node {best_node['id']}")
            
            # Enhanced queue processing
            if queue_manager and queue_manager.is_active():
                from app.core.queue_manager import TaskPriority
                
                priority = TaskPriority.HIGH if (
                    prioritize or 
                    employee_role in ("manager", "director", "executive")
                ) else TaskPriority.NORMAL
                
                logger.info(f"Queueing enhanced employee query with priority {priority}")
                
                task_id = await queue_manager.add_task(
                    task_func="app.tasks.workflow_tasks.process_enhanced_employee_query_task",  # Enhanced task
                    args={
                        "query": query,
                        "employee_id": employee_id,
                        "employee_role": employee_role,
                        "session_id": session_id,
                        "workflow_instance_id": id(self),
                        "enhanced_processing": True
                    },
                    priority=priority
                )
                
                result = await queue_manager.wait_for_task(task_id, timeout=120, polling_interval=0.5)
                
                if result:
                    return result
                    
                status = await queue_manager.get_task_status(task_id)
                if status == "RUNNING":
                    return {
                        "status": "processing",
                        "task_id": task_id,
                        "message": "Your enhanced request is still being processed. Please check back later.",
                        "enhanced_processing": True
                    }
                
                logger.error(f"Enhanced employee task {task_id} failed with status {status}")
                return {
                    "status": "error",
                    "message": "Failed to process your enhanced request. Please try again."
                }
            
            # Direct enhanced processing
            logger.info("Processing enhanced employee query directly without queue or load balancing")
            return await self.arun_simple(query, employee_id, employee_role, session_id)
            
        except Exception as e:
            logger.exception(f"Error in enhanced load-balanced processing: {e}")
            return {
                "status": "error",
                "message": "An error occurred while processing your enhanced request.",
                "error": str(e)
            }


# Enhanced utility functions
# Employee-specific utility functions
async def create_employee_workflow_session(
    employee_id: str,
    employee_role: str = "employee"
) -> str:
    """
    Create a new workflow session for employee.
    Args:
        employee_id: The ID of the employee
        employee_role: The role of the employee (default: "employee")
    Returns:
        str: A unique session ID for the employee workflow
    """
    # Use uuid4 for better uniqueness
    session_id = f"employee_{employee_id}_{uuid.uuid4().hex}"
    logger.info(f"Created employee workflow session: {session_id}")
    return session_id
