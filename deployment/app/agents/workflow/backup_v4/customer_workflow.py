import asyncio
import sys
import time
import traceback
from typing import Dict, Any, Optional, AsyncGenerator, List, Union
import uuid
from datetime import datetime

from loguru import logger
from pathlib import Path

# --- LangGraph Imports ---
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver

# --- Existing imports from your workflow ---
from app.agents.workflow.state import GraphState as AgentState
from app.agents.workflow.initalize import llm_instance, agent_config
from app.agents.factory.factory_tools import TOOL_FACTORY
from app.agents.stores.entry_agent import EntryAgent
from app.agents.stores.company_agent import CompanyAgent
from app.agents.stores.customer_agent import CustomerAgent
from app.agents.stores.product_agent import ProductAgent
from app.agents.stores.visual_agent import VisualAgent
from app.agents.stores.naive_agent import NaiveAgent
from app.agents.stores.rewriter_agent import RewriterAgent
from app.agents.stores.medical_agent import MedicalAgent
from app.agents.stores.genetic_agent import GeneticAgent
from app.agents.stores.drug_agent import DrugAgent
from app.agents.stores.reflection_agent import ReflectionAgent
from app.agents.stores.supervisor_agent import SupervisorAgent
from app.agents.stores.question_generator_agent import QuestionGeneratorAgent
from app.agents.stores.cache_manager import CacheManager
from app.agents.stores.summary_agent import SummaryAgent
from app.agents.data_storages.response_storages import store_response

# --- NEW IMPORT: SentimentAnalysisAgent ---
from app.agents.stores.sentiment_analysis_agent import SentimentAnalysisAgent

class CustomerWorkflow:
    """
    Enhanced workflow with sentiment analysis integration for better user experience.
    Analyzes user satisfaction and can re-execute previous queries when needed.
    """
    def __init__(self, max_iterations: int = 3):
        # Using a default of 3 iterations balances thoroughness with avoiding excessive recursion
        self.max_iterations = max_iterations
        self.agents = self._initialize_agents()
        self.graph = self._build_and_compile_graph()
        self.chat_history_threshold = 6
        self.cache_manager = CacheManager()
        
        logger.add(Path("app/logs/log_workflows/customer_workflow.log"), rotation="10 MB", level="DEBUG", backtrace=True, diagnose=True)
        logger.info("Customer Workflow with Sentiment Analysis initialized.")

    def _initialize_agents(self) -> Dict[str, Any]:
        """Initialize all agent instances including the new SentimentAnalysisAgent."""
        logger.info("Initializing all agent instances with sentiment analysis...")
        llm = llm_instance

        # Core workflow agents
        entry_agent = EntryAgent(llm=llm)
        rewriter_agent = RewriterAgent(llm=llm)
        reflection_agent = ReflectionAgent(llm=llm, default_tool_names=[])
        supervisor_agent = SupervisorAgent(llm=llm)
        question_generator = QuestionGeneratorAgent(llm=llm)
        summary_agent = SummaryAgent(llm=llm)
        
        # NEW: Add SentimentAnalysisAgent
        sentiment_analysis_agent = SentimentAnalysisAgent(llm=llm)

        # Specialist agents
        company_agent = CompanyAgent(llm=llm, default_tool_names=["company_retriever_tool"])
        customer_agent = CustomerAgent(llm=llm)
        product_agent = ProductAgent(llm=llm, default_tool_names=["product_retriever_tool"])
        medical_agent = MedicalAgent(llm=llm, default_tool_names=["medical_retriever_tool"])
        drug_agent = DrugAgent(llm=llm, default_tool_names=["drug_retriever_tool"])
        genetic_agent = GeneticAgent(llm=llm, default_tool_names=["genetic_retriever_tool"])
        visual_agent = VisualAgent(llm=llm, default_tool_names=["image_analyzer"])
        naive_agent = NaiveAgent(llm=llm, default_tool_names=[""])

        return {
            "entry": entry_agent, 
            "rewriter": rewriter_agent,
            "sentiment_analyzer": sentiment_analysis_agent,  # NEW
            "reflection": reflection_agent, 
            "supervisor": supervisor_agent,
            "question_generator": question_generator, 
            "CompanyAgent": company_agent,
            "CustomerAgent": customer_agent, 
            "ProductAgent": product_agent,
            "MedicalAgent": medical_agent, 
            "DrugAgent": drug_agent,
            "GeneticAgent": genetic_agent, 
            "VisualAgent": visual_agent, 
            "NaiveAgent": naive_agent,
            "SummaryAgent": summary_agent   
        }

    async def _sentiment_analysis_node(self, state: AgentState) -> AgentState:
        """
        Node that runs sentiment analysis to understand user intent and satisfaction.
        This determines if we should re-execute previous queries or continue normally.
        """
        logger.info("--- Running Sentiment Analysis ---")
        sentiment_agent = self.agents["sentiment_analyzer"]
        logger.info(f"Chat history length: {len(state.get('chat_history', []))} messages")
        try:
            # Execute sentiment analysis
            analyzed_state = await sentiment_agent.aexecute(state)
            logger.info("Sentiment analysis completed successfully.")
            # Log sentiment analysis results
            sentiment_result = analyzed_state.get("sentiment_analysis", {})
            logger.info(f"Sentiment Analysis - Intent: {sentiment_result.get('user_intent')}, "
                       f"Re-execute: {sentiment_result.get('should_re_execute')}, "
                       f"Similarity: {sentiment_result.get('similarity_score')}")
            
            return analyzed_state
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            # Set default values if sentiment analysis fails
            state["sentiment_analysis"] = {
                "user_intent": "neutral",
                "should_re_execute": False,
                "reasoning": f"Sentiment analysis failed: {str(e)}"
            }
            state["needs_re_execution"] = False
            return state

    async def _re_execution_handler_node(self, state: AgentState) -> AgentState:
        """
        Special node that handles re-execution of previous queries when user is dissatisfied.
        This runs the same specialist agent that handled the previous query.
        """
        logger.info("--- Handling Query Re-execution ---")
        
        try:
            # Get the query to re-execute and the reason
            re_execution_query = state.get("re_execution_query", "")
            
            re_execution_reason = state.get("re_execution_reason", "User requested re-execution")
            
            if not re_execution_query:
                logger.warning("No query found for re-execution")
                last_user_query = state.get("chat_history", [])[1] if state.get("chat_history") else None
                content_query = last_user_query["content"] if last_user_query else "No previous query found"
                logger.info(f"Using last user query for re-execution: {content_query}")
                # return state
                re_execution_query = content_query
            
            logger.info(f"Re-executing query: {re_execution_query[:100]}...")
            logger.info(f"Reason: {re_execution_reason}")
            
            # Temporarily replace the current query with the previous one for re-execution
            original_current_query = state.get("original_query", "")
            state["original_query"] = re_execution_query
            state["rewritten_query"] = re_execution_query
            
            # Add context about this being a re-execution
            state["is_re_execution"] = True
            state["re_execution_context"] = {
                "original_current_query": original_current_query,
                "re_execution_reason": re_execution_reason,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Run the specialist agent with the re-execution query
            result_state = await self._run_agent(state)
            
            # Add re-execution metadata to the response
            if "agent_response" in result_state:
                re_execution_note = f"\n\n[Câu trả lời được cập nhật dựa trên phản hồi của bạn: {re_execution_reason}]"
                result_state["agent_response"] += re_execution_note
            
            # Restore the original current query for context
            result_state["original_query"] = original_current_query
            result_state["was_re_executed"] = True
            
            logger.info("Re-execution completed successfully")
            return result_state
            
        except Exception as e:
            logger.error(f"Error in re-execution handler: {e}")
            # If re-execution fails, continue with normal flow
            state["re_execution_error"] = str(e)
            return state

    async def _summarize_context_node(self, state: AgentState) -> AgentState:
        """Node that runs the SummaryAgent to condense chat history."""
        logger.info("--- Condensing long chat history... ---")
        summary_agent = self.agents["SummaryAgent"]
        summarized_state = await summary_agent.aexecute(state)
        summarized_state['rewritten_query'] = state['rewritten_query']
        return summarized_state
    
    def _should_analyze_sentiment(self, state: AgentState) -> str:
        """
        NEW ROUTING: Determines if sentiment analysis should be performed.
        Only analyze sentiment if there's sufficient chat history.
        """
        logger.info("--- ROUTING: CHECKING IF SENTIMENT ANALYSIS IS NEEDED ---")
        
        chat_history = state.get("chat_history", [])
        
        # Only perform sentiment analysis if we have at least 2 interactions
        if len(chat_history) >= 2:
            logger.info("Chat history sufficient for sentiment analysis. Proceeding to sentiment analysis.")
            return "sentiment_analysis_node"
        else:
            logger.info("Insufficient chat history for sentiment analysis. Proceeding to context summarization.")
            return "summarize_context"
    
    def _should_summarize(self, state: AgentState) -> str:
        """Checks if the context is too long and needs summarization."""
        logger.info("--- ROUTING: CHECKING CONTEXT LENGTH ---")
        history_len = len(state.get("chat_history", []))
        if history_len > self.chat_history_threshold:
            logger.info(f"Context too long ({history_len} messages). Routing to summarizer.")
            return "summarize_context"
        logger.info("Context length is OK. Proceeding to specialist agent.")
        return "specialist_agent"
    
    def _route_after_sentiment_analysis(self, state: AgentState) -> str:
        """
        NEW ROUTING: Decides next step after sentiment analysis.
        If user is dissatisfied and query is similar, re-execute previous query.
        Otherwise, continue with normal flow.
        """
        logger.info("--- ROUTING AFTER SENTIMENT ANALYSIS ---")
        
        sentiment_result = state.get("sentiment_analysis", {})
        should_re_execute = sentiment_result.get("should_re_execute", False)
        user_intent = sentiment_result.get("user_intent", "neutral")
        
        if should_re_execute:
            logger.info(f"User intent: {user_intent}. Re-executing previous query.")
            return "re_execute_query"
        else:
            logger.info(f"User intent: {user_intent}. Continuing with normal flow.")
            return "check_context_length"
    
    def _route_after_context_guardrail(self, state: AgentState) -> str:
        """Decides the next step after the ContextGuardrail runs."""
        logger.info("--- ROUTING AFTER CONTEXT GUARDRAIL ---")
        if state.get("is_context_relevant") is True:
            logger.info("Guardrail check PASSED. Proceeding to specialist agent.")
            return "continue_to_specialist"
        
        logger.warning(f"Guardrail check FAILED. Reason: {state.get('relevance_reason')}. Skipping to supervisor.")
        return "go_to_supervisor"
    
    async def _run_agent(self, state: AgentState) -> AgentState:
        """Node that executes specialist agents with customer context."""
        time_start = time.time()
        agent_name = state.get("classified_agent", "NaiveAgent")
        agent_to_run = self.agents.get(agent_name)
        
        if not agent_to_run:
            state["error_message"] = f"Agent {agent_name} not found"
            state["classified_agent"] = "NaiveAgent"
            agent_to_run = self.agents["NaiveAgent"]
        
        logger.info(f"--- Running Specialist Agent: {agent_name} for Customer ---")
        
        # Add customer-specific context to state
        customer_id = state.get("customer_id")
        customer_role = state.get("customer_role", "customer")
        
        # Enhanced state with customer information
        enhanced_state = state.copy()
        enhanced_state["user_context"] = {
            "customer_id": customer_id,
            "customer_role": customer_role,
            "is_authenticated": True,
            "access_level": self._get_access_level(customer_role),
            "is_re_execution": state.get("is_re_execution", False)
        }
        
        # Database connection management (keeping your existing logic)
        db_health = None
        before_metrics = {}
        try:
            from app.core.db_health_checker import get_db_health_checker, init_db_health_checker
            db_health = get_db_health_checker()
            if not db_health:
                try:
                    logger.info("DB health checker not initialized. Attempting to initialize...")
                    db_health = init_db_health_checker(check_interval=60)
                    await db_health.start()
                except Exception as init_error:
                    logger.warning(f"Could not initialize DB health checker: {init_error}")
            
            if db_health:
                before_metrics = await db_health.get_pool_stats()
                logger.debug(f"DB pool before agent {agent_name}: {before_metrics}")
        except Exception as e:
            logger.debug(f"DB health checker not available: {e}")
        
        try:
            logger.info(f"Executing agent {agent_name} with enhanced state")
            result_state = await agent_to_run.aexecute(enhanced_state)
            logger.info("Agent execution completed successfully.")
            
            # Always clean up connections
            from app.db.session import close_db_connections
            await close_db_connections()
            
            # Database metrics check (keeping your existing logic)
            if db_health:
                try:
                    after_metrics = await db_health.get_pool_stats()
                    logger.debug(f"DB pool after agent {agent_name}: {after_metrics}")
                    
                    if after_metrics.get('checked_out', 0) > before_metrics.get('checked_out', 0):
                        logger.warning(f"Detected potential connection leak in agent {agent_name}. Attempting cleanup.")
                        from app.db.session import close_db_connections
                        await close_db_connections()
                except Exception as metric_error:
                    logger.warning(f"Error getting DB metrics: {metric_error}")
                    from app.db.session import close_db_connections
                    await close_db_connections()
            else:
                from app.db.session import close_db_connections
                await close_db_connections()
                
        except Exception as e:
            logger.error(f"Error executing agent {agent_name}: {e}")
            try:
                from app.db.session import close_db_connections
                await close_db_connections()
            except:
                logger.warning("Failed to clean up DB connections after agent execution.")
            raise
    
        # Preserve customer context and sentiment analysis results
        preserved_keys = [
            'original_query', 'rewritten_query', 'chat_history', 'session_id', 'user_role', 
            'iteration_count', 'agent_thinks', 'customer_id', 'customer_role', 'interaction_id',
            'sentiment_analysis', 'needs_re_execution', 're_execution_query', 'is_re_execution',
            're_execution_context', 'was_re_executed'
        ]
        
        for key in preserved_keys:
            if key in state and key not in result_state:
                result_state[key] = state[key]
        
        # Update agent thinks
        agent_thinks = state.get("agent_thinks", {})
        agent_thinks[agent_name] = result_state.get("agent_response")
        result_state["agent_thinks"] = agent_thinks
        
        logger.info(f"Agent {agent_name} completed with response: {result_state.get('agent_response', '')[:100]}...")
        time_end = time.time()
        logger.info(f"Agent {agent_name} execution took {time_end - time_start:.2f} seconds.")

        # Cleanup connections if agent has cleanup method
        if hasattr(agent_to_run, 'cleanup_connections') and callable(agent_to_run.cleanup_connections):
            await agent_to_run.cleanup_connections()
            
        return result_state

    def _get_access_level(self, customer_role: str) -> str:
        """Determine access level based on customer role."""
        role_access_map = {
            "customer": "basic",
            "premium_customer": "premium", 
            "vip_customer": "vip",
            "admin": "admin"
        }
        return role_access_map.get(customer_role, "basic")

    async def _final_processing_node(self, state: AgentState) -> AgentState:
        """
        Enhanced final processing that includes sentiment analysis results in the output.
        """
        logger.info("--- Running Final Processing (Question Generation + Sentiment Summary) ---")
        question_generator = self.agents["question_generator"]
        
        # Run question generator
        final_state = await question_generator.aexecute(state)
        
        # Add sentiment analysis summary to final state if available
        sentiment_result = state.get("sentiment_analysis")
        if sentiment_result:
            final_state["sentiment_summary"] = {
                "user_intent": sentiment_result.get("user_intent"),
                "was_re_executed": state.get("was_re_executed", False),
                "confidence_level": sentiment_result.get("confidence_level"),
                "similarity_score": sentiment_result.get("similarity_score", 0.0)
            }
        
        return final_state

    def _build_and_compile_graph(self) -> StateGraph:
        """
        Enhanced graph that includes sentiment analysis in the workflow.
        """
        workflow = StateGraph(AgentState)
        
        # --- Define Nodes ---
        workflow.add_node("entry", self.agents["entry"].aexecute)
        workflow.add_node("rewriter", self.agents["rewriter"].aexecute)
        workflow.add_node("sentiment_analysis_node", self._sentiment_analysis_node)  # NEW
        workflow.add_node("re_execute_query", self._re_execution_handler_node)  # NEW
        workflow.add_node("initial_summary", self._summarize_context_node)
        workflow.add_node("specialist_agent", self._run_agent)
        workflow.add_node("reflection", self.agents["reflection"].aexecute)
        workflow.add_node("supervisor", self.agents["supervisor"].astream_execute)
        workflow.add_node("final_processing", self._final_processing_node)
        
        # --- Enhanced Flow with Sentiment Analysis ---
        workflow.set_entry_point("entry")
        
        # 1. Entry -> Rewrite (optional) -> Sentiment Analysis or Summary
        workflow.add_conditional_edges("entry", self._route_after_entry, {
            "rewriter": "rewriter",
            "sentiment_check": "sentiment_analysis_node"  # NEW path
        })
        workflow.add_conditional_edges("rewriter", self._should_analyze_sentiment, {
            "sentiment_analysis_node": "sentiment_analysis_node",
            "summarize_context": "initial_summary"
        })
        
        # 2. NEW: Sentiment Analysis -> Re-execute or Continue
        workflow.add_conditional_edges("sentiment_analysis_node", self._route_after_sentiment_analysis, {
            "re_execute_query": "re_execute_query",
            "check_context_length": "initial_summary"
        })
        
        # 3. NEW: Re-execution -> Reflection (skip normal specialist flow)
        workflow.add_edge("re_execute_query", "reflection")
        
        # 4. Normal flow: Summary -> Specialist -> Reflection
        workflow.add_conditional_edges("initial_summary", self._should_summarize, {
            "summarize_context": "initial_summary",
            "specialist_agent": "specialist_agent"
        })
        workflow.add_edge("specialist_agent", "reflection")
        
        # 5. Reflection -> Supervisor or Loop back
        workflow.add_conditional_edges("reflection", self._route_after_reflection, {
            "supervisor": "supervisor",
            "specialist_agent": "specialist_agent",
        })

        # 6. Final processing and end
        workflow.add_edge("supervisor", "final_processing")
        workflow.add_edge("final_processing", END)

        # Set recursion limit to prevent GraphRecursionError
        recursion_limit = max(50, self.max_iterations * 5)  # Allow enough recursion for max iterations
        logger.info(f"Compiling workflow graph with recursion_limit={recursion_limit}")
        return workflow.compile(checkpointer=InMemorySaver())

    # --- Updated Routing Logic ---
    def _route_after_entry(self, state: AgentState) -> str:
        """Enhanced routing that considers sentiment analysis."""
        logger.info("--- ROUTING AFTER ENTRY ---")
        
        # Check if we should do sentiment analysis first
        chat_history = state.get("chat_history", [])
        if len(chat_history) >= 2:
            logger.info("Sufficient chat history for sentiment analysis.")
            return "sentiment_check"
        
        # Original logic for rewriting
        if state.get("needs_rewrite", False):
            logger.info("Decision: Needs rewrite.")
            return "rewriter"
            
        logger.info("Decision: Proceeding to initial context preparation.")
        return "sentiment_check"

    def _route_after_reflection(self, state: AgentState) -> str:
        """Decides the next step after the Reflection agent runs."""
        logger.info("--- ROUTING AFTER REFLECTION ---")
        
        # Increment and store iteration count
        iteration_count = state.get("iteration_count", 0) + 1
        state["iteration_count"] = iteration_count
        logger.info(f"Reflection iteration {iteration_count}/{self.max_iterations}")

        # Check for termination conditions
        # 1. Max iterations reached - highest priority
        if iteration_count >= self.max_iterations:
            logger.warning(f"Max iterations ({self.max_iterations}) reached. Forcing finalization.")
            # Set flags to ensure we break out of the graph
            state["is_final_answer"] = True
            state["force_terminate"] = True
            return "supervisor"
            
        # 2. Final answer determined by reflection
        if state.get("is_final_answer", False):
            logger.info("Reflection determined the answer is final.")
            return "supervisor"

        # 3. Follow-up with another agent suggested
        followup_agent = state.get("suggest_agent_followups")
        if followup_agent and followup_agent in self.agents:
            logger.info(f"Reflection suggests follow-up with: {followup_agent}")
            state["classified_agent"] = followup_agent
            return "specialist_agent"
            
        logger.info("No more follow-ups. Finalizing with supervisor.")
        return "supervisor"
        
    # --- Enhanced Streaming Methods (keeping your existing logic but with sentiment context) ---
    
    async def arun_streaming(self, query: str, config: Dict = None, customer_id: str = None, user_role: str = "customer", chat_history: list = None) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Enhanced streaming with sentiment analysis integration.
        
        Args:
            query: The user's query to process
            config: Configuration dict for the workflow execution
            customer_id: Customer ID for tracking, if available
            user_role: Role of the user (customer, guest)
            chat_history: Previous conversation history
            
        Returns:
            AsyncGenerator yielding streaming responses
        """
        if chat_history is None:
            chat_history = []
            
        logger.info(f"Starting enhanced workflow with sentiment analysis for {'customer' if customer_id else 'guest'} query: {query[:100]}...")
        start_time = time.time()
        
        # Set graph recursion limit to prevent infinite loops
        if config is None:
            config = {}
            
        # Ensure recursion_limit is set to a reasonable value based on max_iterations
        # This prevents GraphRecursionError during complex workflows
        recommended_limit = max(50, self.max_iterations * 10)
        config.setdefault("recursion_limit", recommended_limit)
        logger.info(f"Setting graph recursion limit to {config.get('recursion_limit')} (max_iterations={self.max_iterations})")
        
        try:
            if self.cache_manager.is_active():
                cache_context = {"customer_id": customer_id} if customer_id and user_role == "customer" else {}
                cache_key = self.cache_manager.create_cache_key(query, chat_history)
                
                logger.debug(f"Checking cache with key: {cache_key}")
                cached_result = await self.cache_manager.get(cache_key)
                
                if cached_result:
                    logger.info(f"Cache HIT for streaming customer query: {query[:50]}...")
                    # Your existing cache streaming logic...
                    if not customer_id:
                        customer_id = f"customer_{uuid.uuid4().hex[:8]}"
                    
                    interaction_id = str(uuid.uuid4())
                    
                    agent_response = cached_result.get("agent_response", "")
                    if agent_response:
                        chunk_size = 50
                        for i in range(0, len(agent_response), chunk_size):
                            chunk = agent_response[i:i+chunk_size]
                            try:
                                yield {
                                    "event": "answer_chunk",
                                    "data": chunk,
                                    "metadata": {
                                        "customer_id": customer_id,
                                        "timestamp": datetime.utcnow().isoformat(),
                                        "cache_hit": True
                                    }
                                }
                                await asyncio.sleep(0.01)
                            except Exception as chunk_error:
                                logger.error(f"Error yielding chunk: {chunk_error}")
                                continue
                    
                    final_result = cached_result.copy()
                    final_result["customer_id"] = customer_id
                    final_result["session_id"] = config.get("configurable", {}).get("thread_id")
                    if "metadata" not in final_result:
                        final_result["metadata"] = {}
                    final_result["metadata"].update({
                        "cache_hit": True,
                        "processing_time": 0.1,
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    
                    yield {
                        "event": "final_result",
                        "data": final_result,
                        "metadata": {
                            "customer_id": customer_id,
                            "interaction_id": interaction_id,
                            "timestamp": datetime.utcnow().isoformat(),
                            "cache_hit": True
                        }
                    }
                    return
            
            # Enhanced initial state with sentiment context
            initial_state = AgentState(
                original_query=query, 
                iteration_count=0, 
                chat_history=chat_history, 
                customer_id=customer_id, 
                user_role=user_role,
                timestamp=datetime.utcnow().isoformat(),
                # Initialize sentiment-related fields
                needs_re_execution=False,
                sentiment_analysis={},
                is_re_execution=False
            )
            
            logger.info("Starting enhanced workflow execution with sentiment analysis...")
            
            # Stream events with sentiment analysis information
            try:
                async for event in self.graph.astream_events(initial_state, config=config, version="v1"):
                    try:
                        kind = event.get("event")
                        node_name = event.get("name", "unknown")
                        
                        # Add sentiment analysis events
                        if kind == "on_chain_start" and node_name == "sentiment_analysis":
                            yield {
                                "event": "sentiment_analysis_start",
                                "data": {
                                    "message": "Analyzing user satisfaction...",
                                    "timestamp": datetime.utcnow().isoformat()
                                },
                                "metadata": {
                                    "node": node_name,
                                    "customer_id": customer_id
                                }
                            }
                        
                        elif kind == "on_chain_end" and node_name == "sentiment_analysis":
                            sentiment_data = event.get("data", {}).get("output", {})
                            sentiment_result = sentiment_data.get("sentiment_analysis", {})
                            
                            yield {
                                "event": "sentiment_analysis_result",
                                "data": {
                                    "user_intent": sentiment_result.get("user_intent"),
                                    "should_re_execute": sentiment_result.get("should_re_execute"),
                                    "confidence_level": sentiment_result.get("confidence_level"),
                                    "reasoning": sentiment_result.get("reasoning", "")[:200] + "..." if len(sentiment_result.get("reasoning", "")) > 200 else sentiment_result.get("reasoning", "")
                                },
                                "metadata": {
                                    "node": node_name,
                                    "timestamp": datetime.utcnow().isoformat()
                                }
                            }
                        
                        elif kind == "on_chain_start" and node_name == "re_execute_query":
                            yield {
                                "event": "re_execution_start",
                                "data": {
                                    "message": "Re-processing your previous query based on your feedback...",
                                    "timestamp": datetime.utcnow().isoformat()
                                },
                                "metadata": {
                                    "node": node_name,
                                    "customer_id": customer_id
                                }
                            }
                        
                        # Your existing event handling logic
                        elif kind == "on_chain_stream":
                            chunk = event.get("data", {}).get("chunk", {})
                            if isinstance(chunk, dict) and "agent_response" in chunk:
                                try:
                                    yield {
                                        "event": "answer_chunk",
                                        "data": chunk.get("agent_response", ""),
                                        "metadata": {
                                            "node": node_name,
                                            "timestamp": datetime.utcnow().isoformat(),
                                            "is_re_execution": chunk.get("is_re_execution", False)
                                        }
                                    }
                                except Exception as chunk_error:
                                    logger.error(f"Error processing answer chunk: {chunk_error}")
                                    continue
                        
                        elif kind == "on_chain_end" and node_name == "final_processing":
                            try:
                                final_state = event.get("data", {}).get("output", {})
                                if not final_state:
                                    logger.error("Final state is empty or invalid")
                                    continue
                                    
                                final_result_data = {
                                    "suggested_questions": final_state.get("suggested_questions", []),
                                    "full_final_answer": final_state.get("agent_response", ""),
                                    "agent_response": final_state.get("agent_response", ""),
                                    "status": "success",
                                    "sentiment_summary": final_state.get("sentiment_summary", {}),  # NEW
                                    "was_re_executed": final_state.get("was_re_executed", False),   # NEW
                                    "metadata": {
                                        "cache_hit": False,
                                        "processing_mode": "enhanced_streaming_workflow",
                                        "user_role": user_role,
                                        "is_private": bool(customer_id and user_role == "customer"),
                                        "processing_time": time.time() - start_time,
                                        "timestamp": datetime.utcnow().isoformat(),
                                        "has_sentiment_analysis": bool(final_state.get("sentiment_summary"))
                                    }
                                }
                                
                                # Cache the successful result for future requests
                                if self.cache_manager.is_active():
                                    try:
                                        cache_context = {"customer_id": customer_id} if customer_id and user_role == "customer" else {}
                                        cache_key = self.cache_manager.create_cache_key(query, chat_history)
                                        
                                        # Create a cacheable version without customer-specific data
                                        cacheable_result = final_result_data.copy()
                                        
                                        # Store in cache asynchronously to not block the response
                                        asyncio.create_task(self.cache_manager.set(
                                            cache_key, 
                                            cacheable_result, 
                                            ttl=1800  # 30 minutes TTL
                                        ))
                                        logger.info(f"Cached enhanced streaming result for {'private ' if cache_context else ''}customer query: {query[:50]}...")
                                    except Exception as cache_error:
                                        logger.error(f"Error caching result: {cache_error}")
                                
                                yield {
                                    "event": "final_result",
                                    "data": final_result_data,
                                    "metadata": {
                                        "node": node_name,
                                        "timestamp": datetime.utcnow().isoformat()
                                    }
                                }
                            except Exception as final_processing_error:
                                logger.error(f"Error in final processing: {final_processing_error}")
                                logger.error(traceback.format_exc())
                                yield {
                                    "event": "error",
                                    "data": {
                                        "error": "Error in final processing",
                                        "details": str(final_processing_error)
                                    },
                                    "metadata": {
                                        "node": node_name,
                                        "timestamp": datetime.utcnow().isoformat()
                                    }
                                }
                        
                        elif kind == "on_chain_start":
                            # Enhanced node start events with sentiment context
                            try:
                                yield {
                                    "event": "node_start",
                                    "data": {
                                        "node": node_name,
                                        "timestamp": datetime.utcnow().isoformat()
                                    },
                                    "metadata": {
                                        "customer_id": customer_id,
                                        "user_role": user_role
                                    }
                                }
                            except Exception as node_start_error:
                                logger.error(f"Error in node start event: {node_start_error}")
                                continue
                        
                        elif kind == "on_chain_error":
                            # Error events
                            error_data = event.get("data", {})
                            error_msg = str(error_data.get("error", "Unknown error"))
                            logger.error(f"Error in node {node_name}: {error_msg}")
                            
                            yield {
                                "event": "error",
                                "data": {
                                    "error": error_msg,
                                    "node": node_name
                                },
                                "metadata": {
                                    "customer_id": customer_id,
                                    "timestamp": datetime.utcnow().isoformat()
                                }
                            }
                            
                    except Exception as event_error:
                        logger.error(f"Error processing event: {event_error}")
                        logger.error(traceback.format_exc())
                        continue
                        
            except Exception as stream_error:
                logger.error(f"Error in enhanced workflow streaming: {stream_error}")
                logger.error(traceback.format_exc())
                yield {
                    "event": "error",
                    "data": {
                        "error": "Error in enhanced workflow execution",
                        "details": str(stream_error)
                    },
                    "metadata": {
                        "timestamp": datetime.utcnow().isoformat()
                    }
                }
                
        except Exception as e:
            # Check specifically for GraphRecursionError or similar recursion issues
            error_type = type(e).__name__
            error_str = str(e).lower()
            
            if "recursion" in error_str or "recursion" in error_type.lower():
                logger.error(f"GraphRecursionError detected: {error_type} - {e}")
                logger.error(f"Workflow exceeded recursion limit with max_iterations={self.max_iterations}")
                logger.error(traceback.format_exc())
                
                # Return a more specific error for recursion issues
                yield {
                    "event": "error",
                    "data": {
                        "error": "Workflow complexity limit exceeded",
                        "details": "The request requires more processing steps than allowed. Please try with a simpler query or contact support.",
                        "technical_details": f"{error_type}: {str(e)}"
                    },
                    "metadata": {
                        "timestamp": datetime.utcnow().isoformat(),
                        "error_type": "recursion_limit_exceeded"
                    }
                }
            else:
                # Handle other exceptions normally
                logger.error(f"Critical error in enhanced workflow: {e}")
                logger.error(traceback.format_exc())
                yield {
                    "event": "error",
                    "data": {
                        "error": "Critical error in enhanced workflow execution",
                        "details": str(e)
                    },
                    "metadata": {
                        "timestamp": datetime.utcnow().isoformat()
                    }
                }

    async def arun_streaming_authenticated(
        self, 
        query: str, 
        config: Dict, 
        customer_id: int,
        customer_role: str = "customer",
        interaction_id: Optional[uuid.UUID] = None,
        chat_history: Optional[list] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Enhanced authenticated streaming with sentiment analysis integration.
        """
        try:
            if not isinstance(query, str) or not query.strip():
                logger.error("Invalid query: empty or not a string")
                yield {"event": "error", "data": {"error": "Invalid query"}}
                return
                
            if not config or not isinstance(config, dict):
                logger.error(f"Invalid config: {config}")
                config = {"configurable": {}}
                
            start_time = time.time()
            
            # Enhanced cache checking with sentiment context
            if self.cache_manager.is_active():
                try:
                    customer_query = f"[CUSTOMER:{customer_id}:{customer_role}] {query}"
                    cache_key = self.cache_manager.create_cache_key(customer_query, chat_history or [])
                    logger.debug(f"[CACHE] Checking enhanced authenticated streaming cache with key: {cache_key}")
                    
                    cached_result = await self.cache_manager.get(cache_key)
                    
                    if cached_result:
                        logger.info(f"[CACHE] Cache HIT for enhanced authenticated streaming query: {query[:50]}...")
                        
                        try:
                            interaction_id_str = str(interaction_id) if interaction_id else str(uuid.uuid4())
                            
                            # Stream cached response
                            agent_response = cached_result.get("agent_response", "")
                            if agent_response:
                                logger.debug(f"[CACHE] Streaming {len(agent_response)} characters from cache")
                                agent_response = agent_response.split(" ")
                                response_chunk = ''
                                for i, chunk in enumerate(agent_response):
                                    response_chunk += chunk + ' '
                                    yield {
                                        "event": "answer_chunk",
                                        "data": response_chunk,
                                        "metadata": {
                                            "customer_id": str(customer_id),
                                            "customer_role": customer_role,
                                            "timestamp": datetime.utcnow().isoformat(),
                                            "cache_hit": True,
                                            "chunk_index": i 
                                        }
                                    }
                                    await asyncio.sleep(0.01)
                            
                            # Enhanced final result with sentiment data
                            final_result = cached_result.copy()
                            final_result["customer_id"] = str(customer_id)
                            final_result["session_id"] = config.get("configurable", {}).get("thread_id")
                            final_result["interaction_id"] = interaction_id_str
                            
                            if "metadata" not in final_result:
                                final_result["metadata"] = {}
                            final_result["metadata"].update({
                                "cache_hit": True,
                                "processing_time": 0.1,
                                "cached_response": True,
                                "processing_mode": "enhanced_authenticated_streaming_workflow",
                                "has_sentiment_analysis": "sentiment_summary" in final_result
                            })
                            
                            yield {
                                "event": "final_result",
                                "data": final_result,
                                "metadata": {
                                    "customer_id": str(customer_id),
                                    "customer_role": customer_role,
                                    "interaction_id": interaction_id_str,
                                    "timestamp": datetime.utcnow().isoformat(),
                                    "cache_hit": True
                                }
                            }
                            logger.info(f"[CACHE] Successfully streamed cached enhanced result")
                            return
                            
                        except Exception as cache_stream_error:
                            logger.error(f"[CACHE] Error streaming cached result: {str(cache_stream_error)}\n{traceback.format_exc()}")
                    else:
                        logger.debug(f"[CACHE] Cache MISS for enhanced authenticated streaming query: {query[:50]}...")
                        
                except Exception as cache_error:
                    logger.error(f"[CACHE] Error checking cache for enhanced streaming: {str(cache_error)}\n{traceback.format_exc()}")
            
            logger.info(f"Starting enhanced authenticated workflow stream for customer {customer_id}")
            logger.debug(f"Query: '{query}', Config: {config}, Role: {customer_role}")
            
            # Enhanced initial state with sentiment analysis fields
            initial_state = AgentState(
                original_query=query,
                iteration_count=0,
                chat_history=chat_history if chat_history else [],
                user_role="customer",
                customer_id=str(customer_id) if customer_id is not None else "",
                customer_role=customer_role or "customer",
                interaction_id=str(interaction_id) if interaction_id else None,
                session_id=config.get("configurable", {}).get("thread_id", ""),
                timestamp=datetime.utcnow().isoformat(),
                # Initialize sentiment-related fields
                needs_re_execution=False,
                sentiment_analysis={},
                is_re_execution=False,
                was_re_executed=False
            )
            
            if not hasattr(self, 'graph') or self.graph is None:
                logger.error("Enhanced workflow graph not initialized")
                yield {"event": "error", "data": {"error": "Enhanced workflow not properly initialized"}}
                return
                
            # Enhanced streaming with sentiment analysis events
            try:
                async for event in self.graph.astream_events(initial_state, config=config, version="v1"):
                    kind = event["event"]
                    node_name = event.get("name", "unknown")
                    
                    # Sentiment analysis specific events
                    if kind == "on_chain_start" and node_name == "sentiment_analysis":
                        yield {
                            "event": "sentiment_analysis_start",
                            "data": {
                                "message": "Analyzing your satisfaction with previous responses...",
                                "timestamp": datetime.utcnow().isoformat()
                            },
                            "metadata": {
                                "customer_id": str(customer_id),
                                "customer_role": customer_role,
                                "node": node_name
                            }
                        }
                    
                    elif kind == "on_chain_end" and node_name == "sentiment_analysis":
                        sentiment_data = event.get("data", {}).get("output", {})
                        sentiment_result = sentiment_data.get("sentiment_analysis", {})
                        
                        yield {
                            "event": "sentiment_analysis_result",
                            "data": {
                                "user_intent": sentiment_result.get("user_intent", "neutral"),
                                "should_re_execute": sentiment_result.get("should_re_execute", False),
                                "confidence_level": sentiment_result.get("confidence_level", "low"),
                                "similarity_score": sentiment_result.get("similarity_score", 0.0),
                                "reasoning_summary": sentiment_result.get("reasoning", "")[:150] + "..." if len(sentiment_result.get("reasoning", "")) > 150 else sentiment_result.get("reasoning", "")
                            },
                            "metadata": {
                                "customer_id": str(customer_id),
                                "customer_role": customer_role,
                                "timestamp": datetime.utcnow().isoformat()
                            }
                        }
                    
                    elif kind == "on_chain_start" and node_name == "re_execute_query":
                        yield {
                            "event": "re_execution_start",
                            "data": {
                                "message": "Based on your feedback, I'm re-processing your previous query with improvements...",
                                "timestamp": datetime.utcnow().isoformat()
                            },
                            "metadata": {
                                "customer_id": str(customer_id),
                                "customer_role": customer_role,
                                "node": node_name
                            }
                        }
                    
                    elif kind == "on_chain_end" and node_name == "re_execute_query":
                        yield {
                            "event": "re_execution_complete",
                            "data": {
                                "message": "Query re-processing completed based on your feedback.",
                                "timestamp": datetime.utcnow().isoformat()
                            },
                            "metadata": {
                                "customer_id": str(customer_id),
                                "customer_role": customer_role
                            }
                        }
                    
                    # Standard streaming events
                    elif kind == "on_chain_stream":
                        chunk = event["data"]["chunk"]
                        if isinstance(chunk, dict) and "agent_response" in chunk:
                            yield {
                                "event": "answer_chunk",
                                "data": chunk.get("agent_response", ""),
                                "metadata": {
                                    "customer_id": str(customer_id),
                                    "customer_role": customer_role,
                                    "timestamp": datetime.utcnow().isoformat(),
                                    "is_re_execution": chunk.get("is_re_execution", False)
                                }
                            }
                    
                    elif kind == "on_chain_end" and node_name == "final_processing":
                        final_state = event["data"]["output"]
                        
                        # Enhanced final result with sentiment analysis data
                        final_result_data = {
                            "suggested_questions": final_state.get("suggested_questions", []),
                            "full_final_answer": final_state.get("agent_response", ""),
                            "agent_response": final_state.get("agent_response", ""),
                            "status": "success",
                            "sentiment_summary": final_state.get("sentiment_summary", {}),
                            "was_re_executed": final_state.get("was_re_executed", False),
                            "agents_used": list(final_state.get("agent_thinks", {}).keys()),
                            "interaction_id": str(interaction_id) if interaction_id else None,
                            "processing_time": self._calculate_processing_time(initial_state),
                            "metadata": {
                                "cache_hit": False,
                                "processing_mode": "enhanced_authenticated_streaming_workflow",
                                "user_role": "customer",
                                "is_private": bool(customer_id),
                                "processing_time": time.time() - start_time,
                                "timestamp": datetime.utcnow().isoformat(),
                                "has_sentiment_analysis": bool(final_state.get("sentiment_summary"))
                            }
                        }
                        
                        # Cache the successful result
                        if self.cache_manager.is_active():
                            try:
                                cache_key = self.cache_manager.create_cache_key(customer_query, chat_history)
                                cacheable_result = final_result_data.copy()
                                cacheable_result["metadata"].pop("customer_id", None)
                                
                                asyncio.create_task(self.cache_manager.set(
                                    cache_key, 
                                    cacheable_result, 
                                    ttl=1800  # 30 minutes TTL
                                ))
                                logger.info(f"Cached enhanced authenticated streaming result: {query[:50]}...")
                            except Exception as cache_error:
                                logger.error(f"Error caching enhanced result: {cache_error}")
                        
                        # Store response
                        try:
                            store_response(final_state)
                        except Exception as store_error:
                            logger.error(f"Error storing enhanced response: {store_error}")
                        
                        yield {
                            "event": "final_result",
                            "data": final_result_data,
                            "metadata": {
                                "node": node_name,
                                "timestamp": datetime.utcnow().isoformat()
                            }
                        }
                    
                    elif kind == "on_chain_start":
                        yield {
                            "event": "node_start",
                            "data": {
                                "node": event["name"],
                                "customer_id": str(customer_id)
                            }
                        }
                    
                    elif kind == "on_chain_error":
                        error_data = event.get("data", {})
                        error_message = str(error_data.get("error", "Unknown error"))
                        logger.error(f"Enhanced chain error in node {event['name']}: {error_message}")
                        
                        yield {
                            "event": "error",
                            "data": {
                                "error": error_message,
                                "node": event["name"]
                            },
                            "metadata": {
                                "customer_id": str(customer_id),
                                "timestamp": datetime.utcnow().isoformat()
                            }
                        }
                        
            except Exception as stream_error:
                error_msg = f"Internal enhanced streaming error: {stream_error}"
                logger.exception(error_msg)
                yield {
                    "event": "error",
                    "data": {"error": error_msg},
                    "metadata": {"timestamp": datetime.utcnow().isoformat()}
                }
                
        except Exception as e:
            logger.exception(f"Critical error in enhanced workflow streaming: {str(e)}")
            yield {
                "event": "error",
                "data": {"error": f"Enhanced workflow error: {str(e)}"}
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

    async def arun_simple_authenticated(
        self, 
        query: str, 
        customer_id: str, 
        user_role: str = "customer",
        chat_history: Optional[list] = None,
        config: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Enhanced simple authenticated execution with sentiment analysis support.
        """
        if not config:
            config = {"configurable": {"thread_id": f"customer_{customer_id}_{int(datetime.utcnow().timestamp())}"}}
        
        # Enhanced cache checking
        if self.cache_manager.is_active():
            cache_context = {"customer_id": customer_id, "user_role": user_role}
            cache_key = self.cache_manager.create_cache_key(query, chat_history or [], context=cache_context)
            cached_result = await self.cache_manager.get(cache_key)
            
            if cached_result:
                logger.info(f"Cache HIT for enhanced authenticated customer query: {query[:50]}...")
                if "metadata" in cached_result:
                    cached_result["metadata"]["cache_hit"] = True
                    cached_result["metadata"]["processing_time"] = 0.1
                    cached_result["metadata"]["has_sentiment_analysis"] = "sentiment_summary" in cached_result
                return cached_result
        
        # Execute enhanced workflow
        final_result = {}
        full_answer = ""
        
        async for event in self.arun_streaming_authenticated(
            query=query,
            config=config,
            customer_id=customer_id,
            user_role=user_role,
            chat_history=chat_history or []
        ):
            if event.get("event") == "answer_chunk":
                full_answer += event["data"]
            elif event.get("event") == "final_result":
                final_result = event["data"]
                final_result["full_answer"] = full_answer
                
                # Cache the enhanced result
                if self.cache_manager.is_active() and final_result.get("status") != "error":
                    cacheable_result = final_result.copy()
                    
                    asyncio.create_task(self.cache_manager.set(
                        cache_key, 
                        cacheable_result, 
                        ttl=1800  # 30 minutes TTL
                    ))
                    logger.info(f"Cached enhanced authenticated customer query result: {query[:50]}...")
                
                break
        
        return final_result

    # Keep all your existing methods (document processing, load balancing, etc.)
    # with the same logic but enhanced logging for sentiment context

    async def arun_document_processing(
        self,
        query: str,
        document_id: int,
        user_type: str,
        user_id: uuid.UUID,
        session_id: Optional[str] = None,
        chat_history: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Enhanced document processing with sentiment analysis support.
        (Keeping your existing implementation with enhanced logging)
        """
        if not session_id:
            session_id = f"doc_session_{document_id}_{datetime.utcnow().timestamp()}"
        
        config = {
            "configurable": {
                "thread_id": session_id,
                "document_id": document_id,
                "retrieval_mode": "document_specific"
            }
        }
        
        # Enhanced initial state
        initial_state = AgentState(
            original_query=query,
            customer_id=str(user_id) if user_type == "customer" else None,
            employee_id=int(user_id) if user_type == "employee" else None,
            guest_id=str(user_id) if user_type == "guest" else None,
            user_role=user_type,
            session_id=session_id,
            chat_history=[],
            rewritten_query="",
            intents="",
            agent_response="",
            agent_thinks={},
            reflection_feedback="",
            is_final_answer=False,
            needs_rewrite=False,
            retry_count=0,
            suggested_questions=[],
            task_assigned=[],
            # Initialize sentiment fields
            needs_re_execution=False,
            sentiment_analysis={},
            is_re_execution=False
        )
        
        logger.info(f"Starting enhanced document processing for {user_type} {user_id}, document {document_id}")
        
        document_context = {
            "document_id": document_id,
            "context_type": "document_processing"
        }
        initial_state["contexts"] = [document_context]
        
        # Enhanced cache checking for document queries
        if query and self.cache_manager.is_active():
            cache_context = {
                "document_id": document_id,
                "user_type": user_type,
                "user_id": str(user_id)
            }
            cache_key = self.cache_manager.create_cache_key(f"doc_query:{query}", chat_history or [])
            
            cached_result = await self.cache_manager.get(cache_key)
            if cached_result:
                logger.info(f"Cache HIT for enhanced document query (doc_id={document_id}): {query[:50]}...")
                if "metadata" in cached_result:
                    cached_result["metadata"]["cache_hit"] = True
                    cached_result["metadata"]["processing_time"] = 0.1
                    cached_result["metadata"]["has_sentiment_analysis"] = "sentiment_summary" in cached_result
                return cached_result
        
        # Process the document with enhanced workflow
        final_result = {}
        full_answer = ""
        
        async for event in self.graph.astream_events(initial_state, config=config, version="v1"):
            kind = event["event"]
            
            if kind == "on_chain_stream":
                chunk = event["data"]["chunk"]
                if isinstance(chunk, dict) and "agent_response" in chunk:
                    current_chunk = chunk.get("agent_response", "")
                    full_answer += current_chunk
            
            elif kind == "on_chain_end":
                node_name = event["name"]
                if node_name == "final_processing":
                    final_state = event["data"]["output"]
                    
                    # Enhanced response with sentiment data
                    final_result = {
                        "document_id": document_id,
                        "user_type": user_type,
                        "user_id": str(user_id),
                        "processed_content": full_answer,
                        "suggested_questions": final_state.get("suggested_questions", []),
                        "agents_used": list(final_state.get("agent_thinks", {}).keys()),
                        "processing_time": self._calculate_processing_time(initial_state) if "timestamp" in initial_state else 0.0,
                        "sentiment_summary": final_state.get("sentiment_summary", {}),
                        "was_re_executed": final_state.get("was_re_executed", False),
                        "metadata": {
                            "cache_hit": False,
                            "document_processed": True,
                            "document_id": document_id,
                            "has_sentiment_analysis": bool(final_state.get("sentiment_summary"))
                        }
                    }
                    
                    # Cache enhanced result
                    if query and self.cache_manager.is_active() and final_result.get("status") != "error":
                        cacheable_result = final_result.copy()
                        
                        asyncio.create_task(self.cache_manager.set(
                            cache_key, 
                            cacheable_result, 
                            ttl=3600  # 1 hour TTL for document queries
                        ))
                        logger.info(f"Cached enhanced document query result (doc_id={document_id}): {query[:50]}...")
                    
                    break
        
        return final_result
    
    async def process_with_load_balancing(
        self, 
        query: str, 
        customer_id: int,
        customer_role: str = "customer",
        session_id: Optional[str] = None,
        prioritize: bool = False
    ) -> Dict[str, Any]:
        """
        Enhanced load balancing with sentiment analysis context.
        (Keeping your existing implementation with enhanced logging)
        """
        try:
            # Import load balancer and queue manager
            try:
                from app.core.load_balancer import get_load_balancer
                from app.core.queue_manager import get_queue_manager
                load_balancer = get_load_balancer()
                queue_manager = get_queue_manager()
            except ImportError:
                load_balancer = None
                queue_manager = None
                logger.warning("Load balancer or queue manager not available - processing with enhanced workflow locally")
            
            if not session_id:
                session_id = f"customer_{customer_id}_{datetime.utcnow().timestamp()}"
                
            # Enhanced load balancing logic
            if load_balancer and load_balancer.is_active():
                local_load = await load_balancer.get_local_load()
                
                if local_load.get('cpu_percent', 0) > 70 or local_load.get('memory_percent', 0) > 80:
                    logger.info(f"Local system load is high: {local_load}. Trying to find another node for enhanced processing.")
                    
                    request_type = "premium_query" if customer_role in ("premium_customer", "vip_customer") else "standard_query"
                    best_node = await load_balancer.get_best_node(request_type=request_type)
                    
                    if best_node and best_node['id'] != load_balancer.get_node_id():
                        logger.info(f"Forwarding enhanced request to node: {best_node['id']}")
                        
                        response = await load_balancer.forward_request(
                            node=best_node,
                            endpoint="/api/v1/customer/chat",
                            method="POST",
                            payload={
                                "query": query,
                                "customer_id": customer_id,
                                "customer_role": customer_role,
                                "session_id": session_id,
                                "enhanced_workflow": True  # Flag for enhanced processing
                            }
                        )
                        
                        if response and 'result' in response:
                            return response['result']
                            
                        logger.warning(f"Failed to get valid response from node {best_node['id']}")
            
            # Enhanced queue processing
            if queue_manager and queue_manager.is_active():
                from app.core.queue_manager import TaskPriority
                
                priority = TaskPriority.HIGH if (
                    prioritize or 
                    customer_role in ("premium_customer", "vip_customer")
                ) else TaskPriority.NORMAL
                
                logger.info(f"Queueing enhanced customer query with priority {priority}")
                
                task_id = await queue_manager.add_task(
                    task_func="app.tasks.workflow_tasks.process_enhanced_query_task",  # Enhanced task
                    args={
                        "query": query,
                        "customer_id": customer_id,
                        "customer_role": customer_role,
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
                
                logger.error(f"Enhanced task {task_id} failed with status {status}")
                return {
                    "status": "error",
                    "message": "Failed to process your enhanced request. Please try again."
                }
            
            # Direct enhanced processing
            logger.info("Processing enhanced query directly without queue or load balancing")
            return await self.arun_simple_authenticated(query, customer_id, customer_role, session_id)
            
        except Exception as e:
            logger.exception(f"Error in enhanced load-balanced processing: {e}")
            return {
                "status": "error",
                "message": "An error occurred while processing your enhanced request.",
                "error": str(e)
            }


# Enhanced utility functions
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

# Enhanced test function with sentiment analysis scenarios
async def test_sentiment_integration():
    """Test function to demonstrate sentiment analysis integration."""
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    logger.info("====== TESTING ENHANCED WORKFLOW WITH SENTIMENT ANALYSIS ======")
    
    workflow_manager = CustomerWorkflow()
    
    # Test scenarios
    test_scenarios = [
        {
            "name": "Positive Feedback - New Question",
            "query": "Cảm ơn, bây giờ cho tôi biết về thuốc Aspirin",
            "chat_history": [
                {"role": "user", "content": "Gen BRCA1 là gì?"},
                {"role": "assistant", "content": "Gen BRCA1 là gen ức chế khối u liên quan đến ung thư vú và buồng trứng..."},
                {"role": "user", "content": "Cảm ơn, bây giờ cho tôi biết về thuốc Aspirin"}
            ],
            "expected_intent": "positive"
        },
        {
            "name": "Negative Feedback - Re-execution Request",
            "query": "Tôi không hiểu rõ câu trả lời trước, hỏi lại về gen BRCA1 đi",
            "chat_history": [
                {"role": "user", "content": "Gen BRCA1 là gì?"},
                {"role": "assistant", "content": "Gen BRCA1 là gen ức chế khối u..."},
                {"role": "user", "content": "Tôi không hiểu rõ câu trả lời trước, hỏi lại về gen BRCA1 đi"}
            ],
            "expected_intent": "negative",
            "expected_re_execution": True
        },
        {
            "name": "Similar Query - Neutral Intent",
            "query": "Gen BRCA1 có chức năng gì khác?",
            "chat_history": [
                {"role": "user", "content": "Gen BRCA1 là gì?"},
                {"role": "assistant", "content": "Gen BRCA1 là gen ức chế khối u..."},
                {"role": "user", "content": "Gen BRCA1 có chức năng gì khác?"}
            ],
            "expected_intent": "neutral"
        }
    ]
    
    for i, scenario in enumerate(test_scenarios, 1):
        logger.info(f"\n{'='*20} TEST SCENARIO {i}: {scenario['name']} {'='*20}")
        
        session_id = f"test_enhanced_session_{i}_{datetime.utcnow().timestamp()}"
        config = {"configurable": {"thread_id": session_id}}
        
        logger.info(f"🚀 EXECUTING QUERY: '{scenario['query']}'")
        logger.info(f"📚 CHAT HISTORY: {len(scenario['chat_history'])} messages")
        
        # Collect all events
        events_collected = []
        full_answer = ""
        final_data = {}
        sentiment_data = {}
        
        try:
            async for event in workflow_manager.arun_streaming(
                query=scenario['query'], 
                config=config, 
                customer_id=f"test_customer_{i}", 
                user_role="customer",
                chat_history=scenario['chat_history']
            ):
                events_collected.append(event)
                
                if event["event"] == "sentiment_analysis_result":
                    sentiment_data = event["data"]
                    logger.info(f"🎭 SENTIMENT ANALYSIS:")
                    logger.info(f"   Intent: {sentiment_data.get('user_intent')}")
                    logger.info(f"   Should Re-execute: {sentiment_data.get('should_re_execute')}")
                    logger.info(f"   Confidence: {sentiment_data.get('confidence_level')}")
                    logger.info(f"   Similarity Score: {sentiment_data.get('similarity_score')}")
                    logger.info(f"   Reasoning: {sentiment_data.get('reasoning_summary')}")
                
                elif event["event"] == "re_execution_start":
                    logger.info(f"🔄 RE-EXECUTION STARTED: {event['data']['message']}")
                
                elif event["event"] == "re_execution_complete":
                    logger.info(f"✅ RE-EXECUTION COMPLETED: {event['data']['message']}")
                
                elif event["event"] == "answer_chunk":
                    chunk_data = event["data"]
                    new_part = chunk_data.replace(full_answer, "", 1)
                    print(new_part, end="", flush=True)
                    full_answer = chunk_data
                
                elif event["event"] == "final_result":
                    final_data = event["data"]
        
            # Verify results
            print(f"\n\n🔍 VERIFICATION:")
            print(f"Expected Intent: {scenario['expected_intent']}")
            print(f"Actual Intent: {sentiment_data.get('user_intent', 'N/A')}")
            print(f"Expected Re-execution: {scenario.get('expected_re_execution', False)}")
            print(f"Actual Re-execution: {sentiment_data.get('should_re_execute', False)}")
            print(f"Was Re-executed: {final_data.get('was_re_executed', False)}")
            print(f"Sentiment Summary: {final_data.get('sentiment_summary', {})}")
            print(f"Suggested Questions: {len(final_data.get('suggested_questions', []))}")
            
            # Validation
            intent_match = sentiment_data.get('user_intent') == scenario['expected_intent']
            re_execution_match = sentiment_data.get('should_re_execute', False) == scenario.get('expected_re_execution', False)
            
            status = "✅ PASSED" if (intent_match and re_execution_match) else "❌ FAILED"
            print(f"Test Status: {status}")
            
        except Exception as e:
            logger.error(f"❌ TEST FAILED with error: {e}")
            logger.error(traceback.format_exc())
        
        print(f"{'='*80}\n")
    
    logger.info("====== ENHANCED WORKFLOW TESTING COMPLETED ======")

# ==============================================================================
# === MAIN EXECUTION WITH ENHANCED TESTING
# ==============================================================================
if __name__ == "__main__":
    async def main():
        # Choose test mode
        import sys
        if len(sys.argv) > 1 and sys.argv[1] == "--test-sentiment":
            await test_sentiment_integration()
        else:
            # Regular workflow test
            logger.remove()
            logger.add(sys.stdout, level="INFO")
            logger.info("====== INITIALIZING ENHANCED STREAMING WORKFLOW ======")
            
            workflow_manager = CustomerWorkflow()
            
            session_id = f"test_session_{asyncio.Task.current_task().get_name()}"
            config = {"configurable": {"thread_id": session_id}}
            
            # Test with sentiment analysis scenario
            query = "Tôi không hiểu rõ câu trả lời về gen BRCA1, bạn có thể giải thích lại không?"
            chat_history = [
                {"role": "user", "content": "Gen BRCA1 là gì?"},
                {"role": "assistant", "content": "Gen BRCA1 là gen ức chế khối u liên quan đến ung thư vú."},
            ]
            
            logger.info("-" * 80)
            logger.info(f"🚀 EXECUTING ENHANCED QUERY: '{query}'")
            logger.info(f"📚 CHAT HISTORY: {len(chat_history)} previous messages")
            
            full_answer = ""
            final_data = {}
            sentiment_detected = False

            # Execute enhanced workflow
            async for event in workflow_manager.arun_streaming(
                query, 
                config, 
                customer_id="test_customer_789", 
                user_role="customer",
                chat_history=chat_history
            ):
                if event["event"] == "sentiment_analysis_result":
                    sentiment_detected = True
                    sentiment_data = event["data"]
                    logger.info(f"🎭 SENTIMENT DETECTED:")
                    logger.info(f"   User Intent: {sentiment_data.get('user_intent')}")
                    logger.info(f"   Should Re-execute: {sentiment_data.get('should_re_execute')}")
                    logger.info(f"   Confidence: {sentiment_data.get('confidence_level')}")
                
                elif event["event"] == "re_execution_start":
                    logger.info(f"🔄 RE-EXECUTION: {event['data']['message']}")
                
                elif event["event"] == "answer_chunk":
                    chunk_data = event["data"]
                    new_part = chunk_data.replace(full_answer, "", 1)
                    print(new_part, end="", flush=True)
                    full_answer = chunk_data
                
                elif event["event"] == "final_result":
                    final_data = event["data"]

            print("\n\n" + "="*20 + " ENHANCED FINAL RESULT " + "="*20)
            print(f"Query: {query}")
            print(f"Sentiment Analysis Performed: {sentiment_detected}")
            print(f"Was Re-executed: {final_data.get('was_re_executed', False)}")
            print(f"Sentiment Summary: {final_data.get('sentiment_summary', {})}")
            print(f"Full Final Answer: {full_answer[:200]}...")
            print(f"Suggested Questions: {final_data.get('suggested_questions', 'N/A')}")
            print("=" * 64 + "\n")

    try:
        asyncio.run(main())
    finally:
        TOOL_FACTORY.cleanup_singletons()