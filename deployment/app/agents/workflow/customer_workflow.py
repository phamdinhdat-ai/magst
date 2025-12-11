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
# Import all specialist agents
from app.agents.stores.company_agent import CompanyAgent
from app.agents.stores.customer_agent import CustomerAgent
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


class CustomerWorkflow:
    """
    An optimized, powerful, and resilient agentic workflow for authenticated customers.
    - Uses a TriageRouterAgent for intelligent, low-latency planning.
    - Follows adaptive paths based on query complexity.
    - Includes robust verification and fallback mechanisms.
    - Generates suggested questions after providing the final answer to enhance user engagement.
    """
    def __init__(self):
        self.agents = self._initialize_agents()
        self.graph = self._build_and_compile_graph()
        self.cache_manager = CacheManager()  # Initialize cache manager
        self.history_cache = HistoryCache()  # Initialize enhanced memory cache
        
        log_path = Path("app/logs/log_workflows/customer_workflow_optimized.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(log_path, rotation="10 MB", level="DEBUG", backtrace=True, diagnose=True)
        logger.info("Optimized Customer Workflow initialized successfully.")

    def _initialize_agents(self) -> Dict[str, Any]:
        """Initializes all agent instances required for the workflow."""
        logger.info("Initializing all agents for the new workflow...")
        llm = llm_instance

        return {
            "TriageGuardrailAgent": TriageGuardrailAgent(llm=llm),
            "SynthesizerAgent": SynthesizerAgent(llm=llm),
            "FinalAnswerAgent": FinalAnswerAgent(llm=llm),
            "QuestionGeneratorAgent": QuestionGeneratorAgent(llm=llm),
            "FallbackAgent": NaiveAgent(llm=llm, default_tool_names=["searchweb_tool","company_retriever_mcp_tool"]), # NaiveAgent serves as our fallback
            "DirectAnswerAgent": NaiveAgent(llm=llm, default_tool_names=["searchweb_tool"]), # Can also use a simple LLM chain
            # All Specialist Agents
            "CompanyAgent": CompanyAgent(llm=llm, default_tool_names=["company_retriever_mcp_tool"]),
            "CustomerAgent": CustomerAgent(llm=llm),
            "ProductAgent": ProductAgent(llm=llm, default_tool_names=["product_retriever_mcp_tool"]),
            "MedicalAgent": MedicalAgent(llm=llm, default_tool_names=["medical_retriever_mcp_tool"]),
            "DrugAgent": DrugAgent(llm=llm, default_tool_names=["drug_retriever_mcp_tool"]),
            "GeneticAgent": GeneticAgent(llm=llm, default_tool_names=["genetic_retriever_mcp_tool"]),
        }

    # --- Core Nodes of the Graph ---

    async def _triage_node(self, state: AgentState) -> AgentState:
        """1. The entry point that runs the planning agent."""
        logger.info("--- (1) Executing Customer Triage Guardrail Node ---")
        logger.debug(f"Customer triage input state keys: {list(state.keys())}")
        
        agent = self.agents["TriageGuardrailAgent"]
        # Ensure the workflow type is set for context
        state['workflow_type'] = 'customer'
        
        try:
            result_state = await agent.aexecute(state)
            
            # Check for toxic content detection
            if result_state.get("next_step") == "toxic_content_block":
                logger.warning(f"Toxic content detected in customer query: {result_state.get('toxicity_reason', 'Unknown reason')}")
                return result_state
            
            # Debug logging to see what the triage agent returned
            logger.debug(f"Customer triage agent returned: next_step='{result_state.get('next_step')}', "
                        f"classified_agent='{result_state.get('classified_agent')}', "
                        f"should_re_execute={result_state.get('should_re_execute', False)}")
            
            # Validate and fix critical fields
            next_step = result_state.get("next_step")
            classified_agent = result_state.get("classified_agent")
            
            if not next_step:
                logger.warning("Customer triage agent didn't set next_step! Attempting to infer from classified_agent...")
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
            
            logger.info(f"Final customer triage result: next_step='{result_state.get('next_step')}', "
                       f"classified_agent='{result_state.get('classified_agent')}'")
            
            return result_state
            
        except Exception as e:
            logger.error(f"Error in customer triage node: {e}", exc_info=True)
            # Return state with safe defaults
            state["next_step"] = "direct_answer"
            state["classified_agent"] = "DirectAnswerAgent"
            state["agent_response"] = f"I apologize, but I encountered an issue processing your request. Let me try to help you directly."
            return state

    async def _direct_answer_node(self, state: AgentState) -> AgentState:
        """2a. FAST PATH: For simple queries that don't need specialist tools."""
        logger.info("--- (2a) Executing Direct Answer Node (Fast Path) ---")
        agent = self.agents["DirectAnswerAgent"]
        # The Triage agent already classified this, so we just execute.
        return await agent.aexecute(state)

    async def _run_specialist_node(self, state: AgentState) -> AgentState:
        """2b. STANDARD PATH: Executes a single, chosen specialist agent."""
        logger.info("--- (2b) Executing Specialist Node ---")
        agent_name = state.get("classified_agent")
        if not agent_name or agent_name not in self.agents:
            logger.error(f"Invalid or missing agent name in state: '{agent_name}'. Routing to fallback.")
            state['error_message'] = f"Triage agent selected an invalid specialist: {agent_name}"
            return state
        
        logger.info(f"Routing to specialist: {agent_name}")
        agent = self.agents[agent_name]
        return await agent.aexecute(state)

    async def _plan_executor_node(self, state: AgentState) -> AgentState:
        """2c. MULTI-AGENT PATH: Executes a sequence of specialist agents."""
        # This is a placeholder for a more complex implementation.
        # For now, we'll simulate it by running the one classified agent
        # and then noting that a multi-plan would run more.
        logger.info("--- (2c) Executing Plan Executor Node (Multi-Agent Path) ---")
        logger.warning("Multi-agent plan execution is complex. Simulating with a single agent for now.")
        
        # In a full implementation, you would loop through a `state['plan']` list.
        # For this example, we just run the primary classified agent.
        result_state = await self._run_specialist_node(state)
        
        # Here you would aggregate results from all agents in the plan.
        # We'll just pass the current context forward.
        return result_state

    async def _verification_node(self, state: AgentState) -> AgentState:
        """3. A lightweight check on the specialist's output."""
        logger.info("--- (3) Executing Verification Node ---")
        agent_response = state.get("agent_response", "").lower()
        error_message = state.get("error_message")
        
        if error_message or not agent_response or "tôi không thể" in agent_response:
            logger.warning("Verification FAILED. Answer is unhelpful or an error occurred.")
            state["is_final_answer"] = False
        else:
            logger.info("Verification PASSED.")
            state["is_final_answer"] = True
        return state

    async def _fallback_agent_node(self, state: AgentState) -> AgentState:
        """4. The safety net when verification fails."""
        logger.warning("--- (4) Executing Fallback Agent Node ---")
        agent = self.agents["FallbackAgent"]
        return await agent.aexecute(state)

    async def _synthesizer_node(self, state: AgentState) -> AgentState:
        """5. The node that combines results from a multi-agent plan."""
        logger.info("--- (5) Executing Synthesizer Node ---")
        agent = self.agents["SynthesizerAgent"]
        return await agent.aexecute(state)

    async def _final_answer_node(self, state: AgentState) -> AgentState:
        """6. Final answer synthesis and streaming for customer-facing response."""
        logger.info("--- (6) Executing Customer Final Answer Node ---")
        agent = self.agents["FinalAnswerAgent"]
        # The FinalAnswerAgent now handles comprehensive information synthesis and streaming internally
        return await agent.aexecute(state)

    async def _question_generator_node(self, state: AgentState) -> AgentState:
        """7. Generate suggested questions based on the final answer."""
        logger.info("--- (7) Executing Customer Question Generator Node ---")
        agent = self.agents["QuestionGeneratorAgent"]
        return await agent.aexecute(state)

    # --- Routing Logic ---

    def _route_from_triage(self, state: AgentState) -> str:
        """Directs the workflow based on the TriageGuardrailAgent's simplified output."""
        # Extract simplified fields from the new guardrail structure
        next_step = state.get("next_step")
        classified_agent = state.get("classified_agent")
        need_analysis = state.get("need_analysis", False)
        is_toxic = state.get("is_toxic", False)
        
        # Comprehensive debug logging
        logger.info(f"Customer Triage Guardrail Router Decision: Routing to '{next_step}'")
        logger.debug(f"Simplified routing state - next_step: '{next_step}', classified_agent: '{classified_agent}', "
                    f"need_analysis: {need_analysis}, is_toxic: {is_toxic}")
        logger.debug(f"Customer state keys available: {list(state.keys())}")
        
        # Handle toxic content blocking first
        if is_toxic or next_step == "toxic_content_block":
            logger.warning("Routing toxic content directly to final answer for safety response")
            return "final_answer"
        
        # Handle need_analysis = True (requires clarification/context)
        if need_analysis:
            logger.info("Query needs analysis - routing to question generator for clarification")
            # For now, route to specialist agent with additional context gathering
            # In future, could route to a dedicated analysis/clarification node
            return "specialist_agent"
        
        # Handle None or empty next_step
        if next_step is None or next_step == "":
            logger.warning(f"next_step is None or empty! Attempting recovery based on classified_agent...")
            
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
            # If we need to ask a question, we can end the flow directly.
            # The agent_response is already set by the Triage agent.
            logger.info("Clarification question generated - ending workflow")
            return "END"
        
        # Final fallback: if no next_step but we have a classified_agent, use specialist
        if classified_agent and classified_agent in self.agents:
            logger.warning(f"Unrecognized next_step '{next_step}' but found classified_agent '{classified_agent}'. Routing to specialist_agent.")
            return "specialist_agent"
            
        logger.error(f"Invalid next_step from Triage: '{next_step}'. All recovery attempts failed. Defaulting to END.")
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
        Builds and compiles the powerful, adaptive LangGraph workflow.
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
        workflow.add_node("question_generator", self._question_generator_node)

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

        # Define the path from final_answer to question_generator and then END
        workflow.add_edge("final_answer", "question_generator")
        workflow.add_edge("question_generator", END)

        # Compile the graph
        logger.info("Compiling the optimized workflow graph.")
        return workflow.compile(checkpointer=InMemorySaver())

    # --- Execution Method (for streaming) ---

    async def arun_streaming(self, query: str, chat_history: list = None) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Executes the optimized workflow and streams events.
        """
        logger.info(f"--- Starting New Workflow Run for Query: '{query}' ---")
        initial_state = AgentState(
            original_query=query,
            chat_history=chat_history or []
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
                    yield {
                        "event": "answer_chunk",
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
        customer_id: str,
        customer_role: str = "customer",
        interaction_id: Optional[uuid.UUID] = None,
        chat_history: Optional[list] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Enhanced streaming workflow for authenticated customers with improved event handling.
        """
        if not isinstance(query, str) or not query.strip():
            logger.error("Invalid customer query provided: empty or not a string")
            yield {
                "event": "error", 
                "data": {"error": "Query cannot be empty."},
                "metadata": {"timestamp": datetime.utcnow().isoformat()}
            }
            return
            
        logger.info(f"Starting customer workflow for query: {query[:100]}... (Customer: {customer_id})")
        
        # Ensure config is properly structured
        if not isinstance(config, dict):
            config = {}
        if "configurable" not in config:
            config["configurable"] = {}
        
        start_time = time.time()
        
        # Generate IDs if not provided
        interaction_id_str = str(interaction_id if interaction_id else uuid.uuid4())
        thread_id = config["configurable"].setdefault("thread_id", f"customer_stream_{interaction_id_str}")
        config["configurable"]["thread_id"] = thread_id
        
        # Check cache first for faster responses
        if self.cache_manager.is_active():
            try:
                # Validate and prepare chat history first for the cache key
                validated_chat_history = []
                if chat_history:
                    try:
                        for message in chat_history:
                            if isinstance(message, dict) and "role" in message and "content" in message:
                                if message["role"] in ["user", "assistant", "system"]:
                                    validated_chat_history.append(message)
                    except Exception as e:
                        logger.warning(f"Error validating chat history for cache: {e}")
                        validated_chat_history = []
                
                cache_key = self.cache_manager.create_cache_key(
                    query.strip().lower(), 
                    validated_chat_history[-5:] if validated_chat_history else [],
                    context="customer_workflow"
                )
                logger.debug(f"Checking cache with key: {cache_key}")
                cached_result = await self.cache_manager.get(cache_key)
                
                if cached_result:
                    logger.info(f"Cache HIT for streaming customer query: {query[:50]}...")
                    
                    # Send stream start notification
                    yield {
                        "event": "stream_start",
                        "data": {"message": "Processing your query (from cache)..."},
                        "metadata": {
                            "customer_id": customer_id,
                            "timestamp": datetime.utcnow().isoformat(),
                            "cache_hit": True
                        }
                    }
                    
                    # Stream cached response with natural chunking
                    agent_response = cached_result.get("agent_response", "")
                    if agent_response:
                        import re
                        sentences = re.split(r'(?<=[.!?])\s+', agent_response)
                        response_chunk = ''
                        for sentence in sentences:
                            response_chunk += sentence + " "
                            yield {
                                "event": "final_stream_chunk",  # Changed to final_stream_chunk for frontend compatibility
                                "data": response_chunk,
                                "metadata": {
                                    "customer_id": customer_id,
                                    "timestamp": datetime.utcnow().isoformat(),
                                    "cache_hit": True
                                }
                            }
                            await asyncio.sleep(min(0.05 * (len(sentence) / 20), 0.2))
                    
                    # Send final result
                    final_result = cached_result.copy()
                    final_result["customer_id"] = customer_id
                    final_result["session_id"] = thread_id
                    if "metadata" not in final_result:
                        final_result["metadata"] = {}
                    
                    final_result["metadata"]["cache_hit"] = True
                    final_result["metadata"]["processing_time"] = 0.1
                    
                    yield {
                        "event": "final_result",
                        "data": final_result,
                        "metadata": {
                            "customer_id": customer_id,
                            "interaction_id": interaction_id_str,
                            "timestamp": datetime.utcnow().isoformat(),
                            "cache_hit": True
                        }
                    }
                    return
                else:
                    logger.debug("No cache hit, proceeding with workflow execution")
            except Exception as e:
                logger.error(f"Error during cache check: {str(e)}", exc_info=True)
        
        # Validate and prepare chat history
        validated_chat_history = []
        if chat_history:
            try:
                for message in chat_history:
                    if isinstance(message, dict) and "role" in message and "content" in message:
                        if message["role"] in ["user", "assistant", "system"]:
                            validated_chat_history.append(message)
                        else:
                            logger.warning(f"Invalid role in chat history: {message['role']}")
                    else:
                        logger.warning(f"Invalid message format in chat history: {message}")
                
                logger.info(f"Using chat history with {len(validated_chat_history)} messages")
            except Exception as history_error:
                logger.error(f"Error processing chat history: {str(history_error)}")
                validated_chat_history = []

        # --- Enhanced Memory Management Integration ---
        session_context = None
        try:
            # Get or create session for the customer
            session_id = f"customer_{customer_id}_{thread_id}"
            
            # Get session context from enhanced memory (using current query for context)
            session_context = await self.cache_manager.get_session_context(session_id, query)
            if session_context:
                logger.info(f"[MEMORY] Retrieved session context for customer {customer_id}: {len(session_context)} chars")
            
            # Add current message to conversation history
            await self.history_cache.add_message(
                session_id=session_id,
                role="user",
                content=query,
                metadata={
                    "customer_id": str(customer_id),
                    "customer_role": customer_role,
                    "interaction_id": interaction_id_str,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
        except Exception as memory_error:
            logger.error(f"[MEMORY] Error during customer memory management: {memory_error}", exc_info=True)
        
        # Create initial state
        initial_state = AgentState(
            original_query=query,
            iteration_count=0,
            chat_history=validated_chat_history,
            user_role=customer_role,
            customer_id=str(customer_id),
            interaction_id=interaction_id_str,
            session_id=thread_id,
            timestamp=datetime.utcnow().isoformat()
        )
        
        # Graph streaming execution
        try:
            logger.info("Starting customer workflow execution...")
            
            # Send stream start message
            yield {
                "event": "stream_start",
                "data": {"message": "Processing your query..."},
                "metadata": {
                    "customer_id": customer_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
            
            # Track full answer for incremental updates
            full_answer = ""
            
            # Stream events from LangGraph
            async for event in self.graph.astream_events(initial_state, config=config, version="v1"):
                try:
                    kind = event.get("event")
                    # node_name = event.get("name", "unknown")
                    node_name = event.get("metadata", {}).get("langgraph_node", "unknown")

                    # logger.debug(f"📡 Processing event: {kind} from node: {node_name}")
                    if not kind:
                        logger.warning(f"Event missing 'event' field: {event}")
                        continue
                    
                    # logger.debug(f"📡 Processing event: {kind} from node: {node_name}")
                    
                    # Handle chain start events
                    if kind == "on_chain_start":
                        logger.info(f"Node started: {node_name}")
                        yield {
                            "event": "node_start",
                            "data": {
                                "node": node_name,
                                "customer_id": customer_id
                            },
                            "metadata": {
                                "timestamp": datetime.utcnow().isoformat()
                            }
                        }
                    
                    # Handle streaming chunks from the LLM
                    elif kind == "on_chain_stream":
                        chunk = event.get("data", {}).get("chunk")
                        # logger.debug(f"🔥 Processing on_chain_stream event from: {node_name} | Chunk : {chunk}")
                        if not chunk:
                            continue
                            
                        # Process when we get agent_response dict
                        if isinstance(chunk.content, str):
                            response = chunk.content
                            if response:
                                # ALWAYS SEND CHUNKS - don't try to be smart with comparison
                                # Simply use this as a checkpoint for the whole response so far
                                # logger.debug(f"Streaming checkpoint response: {response[:50]}...")
                                yield {
                                    "event": "response_checkpoint",  # Special event type for checkpoints
                                    "data": response,  # Send the whole response as a checkpoint
                                    "metadata": {
                                        "customer_id": customer_id,
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
                                    "customer_id": customer_id,
                                    "timestamp": datetime.utcnow().isoformat()
                                }
                            }
                    
                    # Handle direct LLM streaming events
                    elif kind == "on_chat_model_stream":
                        # logger.debug(f"🔥 Processing on_chat_model_stream event from: {node_name}")
                        chunk_text = event.get("data", {}).get("chunk", "")
                        if chunk_text and hasattr(chunk_text, 'content'):
                            chunk_content = chunk_text.content
                            if chunk_content:
                                # Only stream final answer chunks from the final_answer node
                                if node_name == "final_answer":
                                    full_answer += chunk_content
                                    # logger.info(f"🚀 STREAMING: Final answer LLM chunk ({len(chunk_content)} chars): {chunk_content[:50]}...")
                                    yield {
                                        "event": "token_chunk",  # Use token_chunk for real streaming
                                        "data": chunk_content,
                                        "metadata": {
                                            "customer_id": customer_id,
                                            "node": node_name,
                                            "is_final_answer": True,
                                            "timestamp": datetime.utcnow().isoformat()
                                        }
                                    }
                            else:
                                logger.debug(f"🔍 Empty chunk_content from {node_name}")
                        else:
                            logger.debug(f"🔍 No content in chunk from {node_name}: {chunk_text}")
                    
                    # Handle node completion events
                    elif kind == "on_chain_end":
                        logger.info(f"Processing on_chain_end for node: {node_name}")
                        
                        # Stream intermediate responses from non-final nodes only
                        if node_name != "final_answer" and "agent_response" in event.get("data", {}).get("output", {}):
                            response = event.get("data", {}).get("output", {}).get("agent_response", "")
                            if response and response != full_answer:
                                logger.debug(f"Intermediate response from {node_name}: {response[:50]}...")
                                # Don't stream intermediate responses to avoid confusion
                                # Just track them for logging purposes
                        
                        # Special handling for final answer node
                        if node_name == "final_answer":
                            try:
                                final_state = event.get("data", {}).get("output", {})
                                logger.debug(f"Final state: {final_state}")
                                
                                # final_answer = final_state.get("agent_response", "")
                                final_answer = final_state.content if final_state and hasattr(final_state, 'content') else final_state.get("agent_response", "")

                                # --- Enhanced Memory Management: Store Response ---
                                try:
                                    session_id = f"customer_{customer_id}_{thread_id}"
                                    
                                    # Store assistant response in conversation history
                                    await self.history_cache.add_message(
                                        session_id=session_id,
                                        role="assistant",
                                        content=final_answer,
                                        metadata={
                                            "customer_id": str(customer_id),
                                            "customer_role": customer_role,
                                            "interaction_id": interaction_id_str,
                                            "processing_time": time.time() - start_time,
                                            "timestamp": datetime.utcnow().isoformat(),
                                            "agents_used": [node_name if hasattr(final_state, "content") else final_state.get("classified_agent", "")]
                                        }
                                    )
                                    
                                    # Get enhanced session insights
                                    session_summary = await self.cache_manager.get_session_summary(session_id)
                                    session_insights = await self.cache_manager.get_session_insights(session_id)
                                    
                                except Exception as memory_error:
                                    logger.error(f"[MEMORY] Error storing customer response in memory: {memory_error}", exc_info=True)
                                    session_summary = None
                                    session_insights = None
                                
                                # Send final chunk if different from what we've been streaming
                                if final_answer and final_answer != full_answer:
                                    logger.info("Final answer differs from streamed content, sending final complete chunk")
                                    full_answer = final_answer
                                    # Send final complete chunk directly with all content
                                    yield {
                                        "event": "final_stream_chunk",
                                        "data": final_answer,  # Send complete final answer
                                        "metadata": {
                                            "customer_id": customer_id,
                                            "timestamp": datetime.utcnow().isoformat(),
                                            "is_final_chunk": True,
                                            "is_complete": True
                                        }
                                    }
                                
                                # Prepare final result data
                                final_result_data = {
                                    "suggested_questions": final_state.get("suggested_questions", []),
                                    "full_final_answer": final_answer,
                                    "agents_used": list(final_state.get("agent_thinks", {}).keys()),
                                    "processing_time": self._calculate_processing_time(initial_state),
                                    "status": "success",
                                    "agent_response": final_answer,
                                    "interaction_id": interaction_id_str,
                                    "customer_id": customer_id,
                                    "metadata": {
                                        "cache_hit": False,
                                        "processing_mode": "streaming_workflow",
                                        "user_role": customer_role,
                                        "is_authenticated": True,
                                        "session_context": session_context,
                                        "session_summary": session_summary,
                                        "session_insights": session_insights,
                                        "has_enhanced_memory": True
                                    }
                                }
                                
                                # Store response for analytics
                                try:
                                    store_response(final_state)
                                except Exception as store_error:
                                    logger.error(f"Error storing response: {store_error}")
                                
                                # Cache the successful result
                                if self.cache_manager.is_active():
                                    try:
                                        cache_key = self.cache_manager.create_cache_key(
                                            query.strip().lower(), 
                                            validated_chat_history[-5:] if validated_chat_history else [], 
                                            context="customer_workflow"
                                        )
                                        
                                        cacheable_result = {
                                            "suggested_questions": final_result_data.get("suggested_questions", []),
                                            "agent_response": final_result_data.get("agent_response", ""),
                                            "full_final_answer": final_result_data.get("full_final_answer", ""),
                                            "agents_used": final_result_data.get("agents_used", []),
                                            "metadata": {
                                                "cache_hit": False,
                                                "processing_time": time.time() - start_time,
                                                "cached_at": datetime.utcnow().isoformat()
                                            }
                                        }
                                        
                                        asyncio.create_task(
                                            self.cache_manager.set(cache_key, cacheable_result, ttl=1800)
                                        )
                                        logger.info(f"Cached customer streaming result for query: {query[:50]}...")
                                    except Exception as cache_error:
                                        logger.error(f"Error caching result: {str(cache_error)}", exc_info=True)
                                
                                # Final answer is already streamed by FinalAnswerAgent
                                logger.debug(f"Final result data prepared: {final_result_data}")
                                final_answer = final_result_data.get("agent_response", "")
                                if final_answer:
                                    logger.info("Final answer already streamed by FinalAnswerAgent")
                                
                                # Send final complete event with all data
                                yield {
                                    "event": "final_complete",
                                    "data": {
                                        "content": final_answer,
                                        "suggested_questions": final_result_data.get("suggested_questions", []),
                                        "agents_used": final_result_data.get("agents_used", []),
                                        "agent_response": final_result_data.get("agent_response", ""),
                                        "full_final_answer": final_result_data.get("full_final_answer", "")
                                    },
                                    "metadata": {
                                        "customer_id": customer_id,
                                        "interaction_id": interaction_id_str,
                                        "timestamp": datetime.utcnow().isoformat()
                                    }
                                }
                                
                                # Send stream end event
                                yield {
                                    "event": "stream_end",
                                    "data": {
                                        "status": "complete",
                                        "message": "Stream completed successfully",
                                        "processing_time_sec": time.time() - start_time
                                    },
                                    "metadata": {
                                        "customer_id": customer_id,
                                        "timestamp": datetime.utcnow().isoformat()
                                    }
                                }
                            except Exception as final_processing_error:
                                logger.error(f"Error in final processing: {str(final_processing_error)}", exc_info=True)
                                yield {
                                    "event": "error",
                                    "data": {
                                        "error": f"Error in final processing: {str(final_processing_error)}",
                                        "node": "final_answer"
                                    },
                                    "metadata": {
                                        "customer_id": customer_id,
                                        "timestamp": datetime.utcnow().isoformat()
                                    }
                                }
                        
                        # Special handling for the question generator node
                        elif node_name == "question_generator":
                            try:
                                question_state = event.get("data", {}).get("output", {})
                                suggested_questions = question_state.get("suggested_questions", [])
                                
                                if suggested_questions:
                                    logger.info(f"Sending {len(suggested_questions)} suggested questions for customer")
                                    # Send suggested questions as a separate event
                                    yield {
                                        "event": "suggested_questions",
                                        "data": {
                                            "questions": suggested_questions,
                                            "count": len(suggested_questions)
                                        },
                                        "metadata": {
                                            "customer_id": customer_id,
                                            "timestamp": datetime.utcnow().isoformat()
                                        }
                                    }
                                else:
                                    logger.warning("No suggested questions generated for customer")
                                    
                            except Exception as question_error:
                                logger.error(f"Error processing question generator node: {question_error}")
                    
                    # Handle chain errors
                    elif kind == "on_chain_error":
                        error_msg = str(event.get("data", {}).get("error", "Unknown error"))
                        node_name = event.get("name", "unknown")
                        logger.error(f"Error in node {node_name}: {error_msg}")
                        
                        # Provide user-friendly error message
                        user_message = error_msg
                        if "Connection error" in error_msg:
                            user_message = "There was an issue connecting to the AI service. Please try again."
                        elif "timeout" in error_msg.lower():
                            user_message = "The request took too long to process. Please try a simpler query."
                        
                        yield {
                            "event": "error",
                            "data": {
                                "error": user_message,
                                "technical_error": error_msg,
                                "node": node_name
                            },
                            "metadata": {
                                "customer_id": customer_id,
                                "timestamp": datetime.utcnow().isoformat()
                            }
                        }
                        
                        # Send fallback response
                        fallback_message = "I apologize, but I encountered an issue processing your request. Please try again or rephrase your question."
                        yield {
                            "event": "answer_chunk",
                            "data": fallback_message,
                            "metadata": {
                                "customer_id": customer_id,
                                "is_fallback": True,
                                "timestamp": datetime.utcnow().isoformat()
                            }
                        }
                    
                except Exception as event_error:
                    logger.error(f"Error processing event: {str(event_error)}", exc_info=True)
                    
                    error_msg = str(event_error)
                    if "Connection error" in error_msg or "ConnectTimeout" in error_msg:
                        user_message = "Apologies, there was a temporary issue connecting to the AI service. Please try again."
                    elif "timeout" in error_msg.lower():
                        user_message = "Your request is taking longer than expected. Please try a simpler question."
                    else:
                        user_message = "I encountered an issue processing your request. Please try again."
                        
                    yield {
                        "event": "error",
                        "data": {
                            "error": user_message,
                            "technical_error": f"Error processing event: {str(event_error)}",
                            "node": "event_processor"
                        },
                        "metadata": {
                            "customer_id": customer_id,
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    }
                    
                    # Provide fallback response
                    fallback_message = "I'm having trouble processing your request right now. Please try again in a moment."
                    yield {
                        "event": "answer_chunk",
                        "data": fallback_message,
                        "metadata": {
                            "customer_id": customer_id,
                            "timestamp": datetime.utcnow().isoformat(),
                            "is_fallback": True
                        }
                    }
                    
                    # Ensure stream end event
                    yield {
                        "event": "stream_end",
                        "data": {"status": "error", "message": "Stream ended with errors"},
                        "metadata": {
                            "customer_id": customer_id,
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    }
                    
        except Exception as workflow_error:
            error_msg = str(workflow_error)
            logger.error(f"Customer workflow execution failed: {error_msg}", exc_info=True)
            
            # Create user-friendly error message
            user_message = "Sorry, there was an issue processing your request."
            if "timeout" in error_msg.lower():
                user_message = "Your request took too long to process. Please try a shorter or simpler question."
            elif "connection" in error_msg.lower():
                user_message = "There was an issue connecting to the AI service. Please try again in a moment."
            
            yield {
                "event": "error",
                "data": {
                    "error": user_message,
                    "technical_error": f"Workflow execution failed: {error_msg}",
                    "node": "workflow"
                },
                "metadata": {
                    "customer_id": customer_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
            
            # Always provide fallback response
            yield {
                "event": "answer_chunk",
                "data": "I apologize for the inconvenience. I'm having trouble processing your request right now. Please try again with a simpler question, or try again later.",
                "metadata": {
                    "customer_id": customer_id,
                    "is_fallback": True,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }

    def _calculate_processing_time(self, initial_state: AgentState) -> float:
        """Calculate total processing time for the workflow"""
        start_time = initial_state.get("timestamp")
        if not start_time:
            return 0.0
            
        try:
            start_dt = datetime.fromisoformat(start_time)
            end_dt = datetime.utcnow()
            return (end_dt - start_dt).total_seconds()
        except Exception as e:
            logger.error(f"Error calculating processing time: {e}")
            return 0.0
            
    async def astreaming_workflow(self, query: str, config: Dict = None, 
                               customer_id: str = None, customer_role: str = "customer",
                               interaction_id: Optional[uuid.UUID] = None,
                               chat_history: Optional[list] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Direct streaming workflow that uses the LangGraph astream_events directly.
        Focuses on streaming events from the final_answer node and returns final results.
        
        Args:
            query: The customer's query text
            config: Configuration dictionary for the workflow
            customer_id: ID of the customer
            customer_role: Role of the customer (default: "customer")
            interaction_id: Optional UUID for tracking the interaction
            chat_history: Optional chat history for context
            
        Yields:
            Dict[str, Any]: Streaming events or final result
        """
        logger.info(f"Starting direct graph streaming for customer query: {query[:100]}...")
        
        # Initialize tracking variables
        final_answer = ""
        suggested_questions = []
        
        # Generate IDs if not provided
        if customer_id is None:
            customer_id = f"customer_{uuid.uuid4().hex[:8]}"
            
        if interaction_id is None:
            interaction_id = uuid.uuid4()
        
        # Ensure config is properly structured
        if not isinstance(config, dict):
            config = {}
        if "configurable" not in config:
            config["configurable"] = {}
        
        thread_id = config["configurable"].setdefault("thread_id", f"customer_stream_{interaction_id}")
        
        # Prepare initial state
        initial_state = AgentState(
            original_query=query,
            iteration_count=0,
            chat_history=chat_history or [],
            user_role=customer_role,
            customer_id=customer_id,
            interaction_id=str(interaction_id),
            session_id=thread_id,
            timestamp=datetime.utcnow().isoformat()
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
 
 
async def create_customer_workflow_session(
    customer_id: int,
    customer_role: str = "customer"
) -> str:
    """Create a new enhanced workflow session for customer with sentiment analysis support."""
    session_id = f"enhanced_customer_{customer_id}_{datetime.utcnow().timestamp()}"
    logger.info(f"Created enhanced customer workflow session: {session_id}")
    return session_id

def validate_customer_access(customer_role: str, requested_feature: str) -> bool:
    """Validate if customer role has access to requested feature (enhanced version)."""
    access_matrix = {
        "customer": ["basic_search", "company_info", "product_info", "sentiment_analysis"],
        "premium_customer": ["basic_search", "company_info", "product_info", "advanced_search", "priority_support", "sentiment_analysis", "query_re_execution"],
        "vip_customer": ["basic_search", "company_info", "product_info", "advanced_search", "priority_support", "personal_consultant", "sentiment_analysis", "query_re_execution", "priority_re_execution"],
        "admin": ["all_features"]
    }
    
    allowed_features = access_matrix.get(customer_role, [])
    return requested_feature in allowed_features or "all_features" in allowed_features
 
     
# Example usage
if __name__ == '__main__':
    async def main():
        workflow = CustomerWorkflow()
        
        test_queries = [
            "Hello, How do you do, me?", # Should take the fast path
        #     "What are the main side effects of Paracetamol?", # Should take the specialist path
        #     "Tell me about that thing I asked about before.", # Should trigger clarification
        #     "Compare your company's stock performance to its return policy." # Should take the multi-agent path
        ]
        
        for q in test_queries:
            print(f"\n{'='*20} TESTING QUERY: {q} {'='*20}")
            full_response = ""
            config = {"configurable": {"thread_id": "12345"}}
            async for event in workflow.arun_streaming_authenticated(q, config=config, customer_id=1, customer_role="customer"):
                if event['event'] == 'node_start':
                    print(f"  -> Entering Node: {event['data']['node']}")
                elif event['event'] == 'token_chunk':
                    new_part = event['data'].replace(full_response, "", 1)
                    print(new_part, end="", flush=True)
                    full_response = event['data']
                elif event['event'] == 'final_result':
                    print("\n---")
                    print(f"Suggested Questions: {event['data'].get('suggested_questions', [])}")
            print(f"\n{'='*60}\n")
            
    asyncio.run(main())