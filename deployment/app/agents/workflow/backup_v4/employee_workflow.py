# app/agents/workflow/employee_workflow.py (Enhanced with Sentiment Analysis)

import asyncio
import sys
import uuid
import time
import traceback
from typing import Dict, Any, Optional, AsyncGenerator, List
from datetime import datetime, timedelta
from loguru import logger
from pathlib import Path

# --- LangGraph Imports ---
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver

# --- Import base components ---
sys.path.append(str(Path(__file__).parent.parent))

# --- Import all agent CLASSES ---
from app.agents.workflow.state import GraphState as AgentState
from app.agents.workflow.initalize import llm_instance, agent_config, llm_reasoning
from app.agents.factory.factory_tools import TOOL_FACTORY
from app.agents.stores.entry_agent import EntryAgent
from app.agents.stores.company_agent import CompanyAgent
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
from app.agents.stores.employee_agent import EmployeeAgent
from app.agents.stores.cache_manager import CacheManager
from app.agents.data_storages.response_storages import store_response
from app.agents.stores.summary_agent import SummaryAgent

# --- NEW IMPORT: SentimentAnalysisAgent for Employee workflow ---
from app.agents.stores.sentiment_analysis_agent import SentimentAnalysisAgent

class EmployeeWorkflow:
    """
    Enhanced workflow with sentiment analysis integration for better employee experience.
    Analyzes employee satisfaction and can re-execute previous queries when needed.
    Separated from customer data to ensure security.
    """
    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.agents = self._initialize_agents()
        self.graph = self._build_and_compile_graph()
        self.chat_history_threshold = 6
        self.cache_manager = CacheManager()
        
        logger.add(Path("app/logs/log_workflows/employee_workflow.log"), rotation="10 MB", level="DEBUG", backtrace=True, diagnose=True)
        logger.info("Enhanced Employee Workflow with Sentiment Analysis initialized.")

    def _create_employee_cache_query(self, employee_id: str, employee_role: str, query: str) -> str:
        """Helper to create a standardized employee-specific cache query string."""
        return f"[EMPLOYEE:{employee_id}:{employee_role}] {query}"

    def _initialize_agents(self) -> Dict[str, Any]:
        """
        Initialize agents for SECURE Employee Workflow with sentiment analysis.
        *** DOES NOT INCLUDE CustomerAgent for security. ***
        """
        logger.info("Initializing agents for SECURE Employee Workflow with sentiment analysis...")
        llm = llm_instance

        # Core workflow agents
        entry_agent = EntryAgent(llm=llm_reasoning)
        rewriter_agent = RewriterAgent(llm=llm)
        reflection_agent = ReflectionAgent(llm=llm, default_tool_names=[])
        supervisor_agent = SupervisorAgent(llm=llm)
        question_generator = QuestionGeneratorAgent(llm=llm)
        summary_agent = SummaryAgent(llm=llm)
        
        # NEW: Add SentimentAnalysisAgent for employee workflow
        sentiment_analysis_agent = SentimentAnalysisAgent(llm=llm)

        # Specialist agents allowed for employees
        employee_agent = EmployeeAgent(llm=llm, default_tool_names=[])
        company_agent = CompanyAgent(llm=llm, default_tool_names=["company_retriever_tool"])
        product_agent = ProductAgent(llm=llm, default_tool_names=["product_retriever_tool"])
        medical_agent = MedicalAgent(llm=llm, default_tool_names=["medical_retriever_tool"])
        drug_agent = DrugAgent(llm=llm, default_tool_names=["drug_retriever_tool"])
        genetic_agent = GeneticAgent(llm=llm, default_tool_names=["genetic_retriever_tool"])
        visual_agent = VisualAgent(llm=llm, default_tool_names=["image_analyzer"])
        naive_agent = NaiveAgent(llm=llm, default_tool_names=["searchweb_tool"])

        return {
            # Core workflow agents
            "entry": entry_agent,
            "rewriter": rewriter_agent,
            "sentiment_analyzer": sentiment_analysis_agent,  # NEW
            "reflection": reflection_agent,
            "supervisor": supervisor_agent,
            "question_generator": question_generator,
            "SummaryAgent": summary_agent,
            # Specialist agents for employees
            "EmployeeAgent": employee_agent,
            "CompanyAgent": company_agent,
            "ProductAgent": product_agent,
            "MedicalAgent": medical_agent,
            "DrugAgent": drug_agent,
            "GeneticAgent": genetic_agent,
            "VisualAgent": visual_agent,
            "NaiveAgent": naive_agent,
        }

    async def _sentiment_analysis_node(self, state: AgentState) -> AgentState:
        """
        Node that runs sentiment analysis to understand employee intent and satisfaction.
        This determines if we should re-execute previous queries or continue normally.
        """
        logger.info("--- Running Employee Sentiment Analysis ---")
        sentiment_agent = self.agents["sentiment_analyzer"]
        logger.info(f"Employee chat history length: {len(state.get('chat_history', []))} messages")
        
        try:
            # Execute sentiment analysis
            analyzed_state = await sentiment_agent.aexecute(state)
            logger.info("Employee sentiment analysis completed successfully.")
            
            # Log sentiment analysis results
            sentiment_result = analyzed_state.get("sentiment_analysis", {})
            logger.info(f"Employee Sentiment Analysis - Intent: {sentiment_result.get('user_intent')}, "
                       f"Re-execute: {sentiment_result.get('should_re_execute')}, "
                       f"Similarity: {sentiment_result.get('similarity_score')}")
            
            return analyzed_state
            
        except Exception as e:
            logger.error(f"Error in employee sentiment analysis: {e}")
            # Set default values if sentiment analysis fails
            state["sentiment_analysis"] = {
                "user_intent": "neutral",
                "should_re_execute": False,
                "reasoning": f"Employee sentiment analysis failed: {str(e)}"
            }
            state["needs_re_execution"] = False
            return state

    async def _re_execution_handler_node(self, state: AgentState) -> AgentState:
        """
        Special node that handles re-execution of previous queries when employee is dissatisfied.
        This runs the same specialist agent that handled the previous query.
        """
        logger.info("--- Handling Employee Query Re-execution ---")
        
        try:
            # Get the query to re-execute and the reason
            re_execution_query = state.get("re_execution_query", "")
            re_execution_reason = state.get("re_execution_reason", "Employee requested re-execution")
            
            if not re_execution_query:
                logger.warning("No query found for employee re-execution")
                last_user_query = state.get("chat_history", [])[1] if state.get("chat_history") else None
                content_query = last_user_query["content"] if last_user_query else "No previous query found"
                logger.info(f"Using last employee query for re-execution: {content_query}")
                re_execution_query = content_query
            
            logger.info(f"Re-executing employee query: {re_execution_query[:100]}...")
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
                re_execution_note = f"\n\n[Answer updated based on your feedback: {re_execution_reason}]"
                result_state["agent_response"] += re_execution_note
            
            # Restore the original current query for context
            result_state["original_query"] = original_current_query
            result_state["was_re_executed"] = True
            
            logger.info("Employee re-execution completed successfully")
            return result_state
            
        except Exception as e:
            logger.error(f"Error in employee re-execution handler: {e}")
            # If re-execution fails, continue with normal flow
            state["re_execution_error"] = str(e)
            return state

    async def _summarize_context_node(self, state: AgentState) -> AgentState:
        """Node that runs the SummaryAgent to condense chat history."""
        logger.info("--- Condensing long employee chat history... ---")
        summary_agent = self.agents["SummaryAgent"]
        summarized_state = await summary_agent.aexecute(state)
        summarized_state['rewritten_query'] = state['rewritten_query']
        return summarized_state

    def _should_analyze_sentiment(self, state: AgentState) -> str:
        """
        NEW ROUTING: Determines if sentiment analysis should be performed.
        Only analyze sentiment if there's sufficient chat history.
        """
        logger.info("--- ROUTING: CHECKING IF EMPLOYEE SENTIMENT ANALYSIS IS NEEDED ---")
        
        chat_history = state.get("chat_history", [])
        
        # Only perform sentiment analysis if we have at least 2 interactions
        if len(chat_history) >= 2:
            logger.info("Employee chat history sufficient for sentiment analysis. Proceeding to sentiment analysis.")
            return "sentiment_analysis_node"
        else:
            logger.info("Insufficient employee chat history for sentiment analysis. Proceeding to context summarization.")
            return "summarize_context"

    def _should_summarize(self, state: AgentState) -> str:
        """Checks if the context is too long and needs summarization."""
        logger.info("--- ROUTING: CHECKING EMPLOYEE CONTEXT LENGTH ---")
        history_len = len(state.get("chat_history", []))
        if history_len > self.chat_history_threshold:
            logger.info(f"Employee context too long ({history_len} messages). Routing to summarizer.")
            return "summarize_context"
        logger.info("Employee context length is OK. Proceeding to specialist agent.")
        return "specialist_agent"

    def _route_after_sentiment_analysis(self, state: AgentState) -> str:
        """
        NEW ROUTING: Decides next step after sentiment analysis.
        If employee is dissatisfied and query is similar, re-execute previous query.
        Otherwise, continue with normal flow.
        """
        logger.info("--- ROUTING AFTER EMPLOYEE SENTIMENT ANALYSIS ---")
        
        sentiment_result = state.get("sentiment_analysis", {})
        should_re_execute = sentiment_result.get("should_re_execute", False)
        user_intent = sentiment_result.get("user_intent", "neutral")
        
        if should_re_execute:
            logger.info(f"Employee intent: {user_intent}. Re-executing previous query.")
            return "re_execute_query"
        else:
            logger.info(f"Employee intent: {user_intent}. Continuing with normal flow.")
            return "check_context_length"

    async def _run_agent(self, state: AgentState) -> AgentState:
        """Node that executes specialist agents with employee context."""
        time_start = time.time()
        agent_name = state.get("classified_agent")
        
        # Security check: ensure agent is available for employees
        if not agent_name or agent_name not in self.agents:
            state['error_message'] = f"Access Denied or Invalid Agent: The requested agent '{agent_name}' is not available in employee workflow."
            state["classified_agent"] = "NaiveAgent"
            agent_name = "NaiveAgent"
        
        agent_to_run = self.agents[agent_name]
        logger.info(f"--- Running Specialist Agent: {agent_name} for Employee ---")
        
        # Add employee-specific context to state
        employee_id = state.get("employee_id")
        employee_role = state.get("employee_role", "employee")
        
        # Enhanced state with employee information
        enhanced_state = state.copy()
        enhanced_state["user_context"] = {
            "employee_id": employee_id,
            "employee_role": employee_role,
            "is_authenticated": True,
            "access_level": self._get_employee_access_level(employee_role),
            "is_re_execution": state.get("is_re_execution", False)
        }
        
        # Database connection management
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
                logger.debug(f"DB pool before employee agent {agent_name}: {before_metrics}")
        except Exception as e:
            logger.debug(f"DB health checker not available: {e}")
        
        try:
            logger.info(f"Executing employee agent {agent_name} with enhanced state")
            result_state = await agent_to_run.aexecute(enhanced_state)
            logger.info("Employee agent execution completed successfully.")
            
            # Always clean up connections
            from app.db.session import close_db_connections
            await close_db_connections()
            
        except Exception as e:
            logger.error(f"Error executing employee agent {agent_name}: {e}")
            try:
                from app.db.session import close_db_connections
                await close_db_connections()
            except:
                logger.warning("Failed to clean up DB connections after employee agent execution.")
            raise
        
        # Preserve employee context and sentiment analysis results
        preserved_keys = [
            'original_query', 'rewritten_query', 'chat_history', 'session_id', 'user_role', 
            'iteration_count', 'agent_thinks', 'employee_id', 'employee_role', 'interaction_id',
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
        
        logger.info(f"Employee agent {agent_name} completed with response: {result_state.get('agent_response', '')[:100]}...")
        time_end = time.time()
        logger.info(f"Employee agent {agent_name} execution took {time_end - time_start:.2f} seconds.")

        # Cleanup connections if agent has cleanup method
        if hasattr(agent_to_run, 'cleanup_connections') and callable(agent_to_run.cleanup_connections):
            await agent_to_run.cleanup_connections()
            
        return result_state

    def _get_employee_access_level(self, employee_role: str) -> str:
        """Determine access level based on employee role."""
        role_access_map = {
            "employee": "standard",
            "senior_employee": "advanced",
            "manager": "manager",
            "director": "director",
            "executive": "executive",
            "admin": "admin"
        }
        return role_access_map.get(employee_role, "standard")

    async def _final_processing_node(self, state: AgentState) -> AgentState:
        """
        Enhanced final processing that includes sentiment analysis results in the output.
        """
        logger.info("--- Running Employee Final Processing (Question Generation + Sentiment Summary) ---")
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
        Enhanced graph that includes sentiment analysis in the employee workflow.
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
        workflow.add_conditional_edges("reflection", self._route_after_reflection_with_loop, {
            "supervisor": "supervisor",
            "specialist_agent": "specialist_agent",
        })

        # 6. Final processing and end
        workflow.add_edge("supervisor", "final_processing")
        workflow.add_edge("final_processing", END)

        return workflow.compile(checkpointer=InMemorySaver())

    def _route_after_entry(self, state: AgentState) -> str:
        """Enhanced routing that considers sentiment analysis."""
        logger.info("--- ROUTING AFTER EMPLOYEE ENTRY ---")
        
        # Check if we should do sentiment analysis first
        chat_history = state.get("chat_history", [])
        if len(chat_history) >= 2:
            logger.info("Sufficient employee chat history for sentiment analysis.")
            return "sentiment_check"
        
        # Original logic for rewriting
        if state.get("needs_rewrite", False):
            logger.info("Employee query needs rewrite.")
            return "rewriter"
        
        # Check if agent is available for employees
        agent_name = state.get("classified_agent")
        if agent_name in self.agents:
            logger.info("Proceeding directly to employee specialist agent.")
            return "specialist_agent"
        
        # Default to NaiveAgent if classified agent not available
        state["classified_agent"] = "NaiveAgent"
        return "specialist_agent"

    def _route_after_reflection_with_loop(self, state: AgentState) -> str:
        """Routing logic with loop for employee workflow."""
        logger.info("--- ROUTING AFTER EMPLOYEE REFLECTION ---")
        iteration_count = state.get("iteration_count", 0) + 1
        state["iteration_count"] = iteration_count

        if state.get("error_message"):
            logger.warning("Error message detected, ending workflow.")
            return "supervisor"
        
        if iteration_count >= self.max_iterations:
            logger.warning(f"Max iterations ({self.max_iterations}) reached for employee. Finalizing.")
            return "supervisor"
        
        if state.get("is_final_answer", False):
            logger.info("Employee reflection determined the answer is final.")
            return "supervisor"

        followup_agent = state.get("suggest_agent_followups")
        if followup_agent and followup_agent in self.agents:
            logger.info(f"Employee reflection suggests follow-up with: {followup_agent}")
            state["classified_agent"] = followup_agent
            return "specialist_agent"
            
        logger.info("No more follow-ups for employee. Finalizing with supervisor.")
        return "supervisor"

    # --- Enhanced Caching Methods ---
    
    async def _cache_result(self, cache_key: str, result: Dict[str, Any], query: str, ttl: int = 1800) -> None:
        """
        Helper method to cache a result with proper error handling and logging.
        """
        try:
            if not self.cache_manager.is_active():
                return
            start_time = time.time()
            logger.debug(f"[CACHE] Attempting to cache employee result for key: {cache_key}")
            
            # Create a safe copy of the result to avoid modifying the original
            cacheable_result = result.copy()
            # Remove sensitive or session-specific data
            cacheable_result.pop("session_id", None)
            cacheable_result.pop("interaction_id", None)
            cacheable_result.pop("employee_id", None)
            
            # Add cache metadata
            if "metadata" not in cacheable_result:
                cacheable_result["metadata"] = {}
            cacheable_result["metadata"]["cached_at"] = datetime.utcnow().isoformat()
            
            await self.cache_manager.set(cache_key, cacheable_result, ttl=ttl)
            duration = time.time() - start_time
            logger.info(f"[CACHE] Successfully cached employee result for query: {query[:50]}... (duration: {duration:.3f}s)")
            
        except Exception as e:
            logger.error(f"[CACHE] Failed to cache employee result: {str(e)}\n{traceback.format_exc()}")

    async def _check_and_return_cached_result(self, cache_key: str, query: str, employee_id: str, employee_role: str, session_id: Optional[str], chat_history: List) -> Optional[Dict[str, Any]]:
        """Checks cache and returns formatted cached result if found."""
        try:
            cached_result = await self.cache_manager.get(cache_key)
            if cached_result:
                logger.info(f"[CACHE] Cache HIT for employee query: {query[:50]}...")
                # Update result with current session info
                result = cached_result.copy()
                result["employee_id"] = employee_id
                result["session_id"] = session_id
                # Update metadata with cache info
                if "metadata" not in result:
                    result["metadata"] = {}
                result["metadata"].update({
                    "cache_hit": True,
                    "cached_response": True,
                    "has_sentiment_analysis": "sentiment_summary" in result
                })
                return result
            else:
                logger.debug(f"[CACHE] Cache MISS for employee query: {query[:50]}...")
        except Exception as cache_error:
            logger.error(f"[CACHE] Error checking cache: {str(cache_error)}\n{traceback.format_exc()}")
        return None

    async def _stream_from_cache(self, cached_result: Dict[str, Any], employee_id: str, employee_role: str, session_id: Optional[str], interaction_id: Optional[str] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """Streams events from a cached result with sentiment data."""
        try:
            interaction_id_str = interaction_id or str(uuid.uuid4())
            
            # Yield answer chunks if available
            agent_response = cached_result.get("agent_response", "")
            if agent_response:
                logger.debug(f"[CACHE] Streaming {len(agent_response)} characters from employee cache")
                agent_response_words = agent_response.split(" ")
                response_chunk = ''
                total_chunks = len(agent_response_words)
                
                for i, word in enumerate(agent_response_words):
                    response_chunk += word + ' '
                    yield {
                        "event": "answer_chunk",
                        "data": response_chunk.strip(),
                        "metadata": {
                            "employee_id": employee_id,
                            "employee_role": employee_role,
                            "timestamp": datetime.utcnow().isoformat(),
                            "cache_hit": True,
                            "chunk_index": i,
                            "total_chunks": total_chunks,
                            "has_sentiment_analysis": "sentiment_summary" in cached_result
                        }
                    }
                    await asyncio.sleep(0.01)
            
            # Yield final result with sentiment data
            final_result = cached_result.copy()
            final_result["employee_id"] = employee_id
            final_result["session_id"] = session_id
            final_result["interaction_id"] = interaction_id_str
            
            if "metadata" not in final_result:
                final_result["metadata"] = {}
            
            final_result["metadata"]["cache_hit"] = True
            final_result["metadata"]["cached_response"] = True
            if "processing_time" not in final_result["metadata"]:
                final_result["metadata"]["processing_time"] = 0.01

            yield {
                "event": "final_result",
                "data": final_result,
                "metadata": {
                    "employee_id": employee_id,
                    "employee_role": employee_role,
                    "interaction_id": interaction_id_str,
                    "timestamp": datetime.utcnow().isoformat(),
                    "cache_hit": True,
                    "has_sentiment_analysis": "sentiment_summary" in final_result
                }
            }
            logger.info(f"[CACHE] Successfully streamed cached employee result")
            
        except Exception as cache_stream_error:
            logger.error(f"[CACHE] Error streaming cached employee result: {str(cache_stream_error)}\n{traceback.format_exc()}")
            yield {
                "event": "error",
                "data": {
                    "error": "Error streaming cached employee result",
                    "details": str(cache_stream_error)
                },
                "metadata": {
                    "employee_id": employee_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }

    async def _stream_events_from_graph(self, initial_state: AgentState, config: Dict, employee_id: str, employee_role: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Core logic to stream events from the LangGraph workflow with sentiment analysis."""
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
                                "message": "Analyzing employee satisfaction...",
                                "timestamp": datetime.utcnow().isoformat()
                            },
                            "metadata": {
                                "node": node_name,
                                "employee_id": employee_id,
                                "employee_role": employee_role
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
                                "employee_id": employee_id,
                                "employee_role": employee_role
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
                                "employee_id": employee_id,
                                "employee_role": employee_role
                            }
                        }
                    
                    elif kind == "on_chain_stream":
                        # Handle streaming chunks from SupervisorAgent
                        chunk = event.get("data", {}).get("chunk")
                        if isinstance(chunk, dict) and "agent_response" in chunk:
                            agent_response = chunk.get("agent_response", "")
                            logger.debug(f"[EMPLOYEE_STREAMING] Yielding answer chunk: {len(agent_response)} chars")
                            yield {
                                "event": "answer_chunk",
                                "data": agent_response,
                                "metadata": {
                                    "employee_id": employee_id,
                                    "employee_role": employee_role,
                                    "timestamp": datetime.utcnow().isoformat(),
                                    "cache_hit": False,
                                    "is_re_execution": chunk.get("is_re_execution", False)
                                }
                            }
                    
                    elif kind == "on_chain_end":
                        # Handle node completion events
                        logger.debug(f"[EMPLOYEE_STREAMING] Node completed: {node_name}")
                        if node_name == "final_processing":
                            # Final processing node completed - yield final result
                            try:
                                final_state = event.get("data", {}).get("output", {})
                                processing_time = time.time() - float(initial_state.get("timestamp", time.time()))
                                
                                # Enhanced final result with sentiment analysis data
                                final_result_data = {
                                    "suggested_questions": final_state.get("suggested_questions", []),
                                    "agent_response": final_state.get("agent_response", ""),
                                    "full_final_answer": final_state.get("agent_response", ""),
                                    "agents_used": list(final_state.get("agent_thinks", {}).keys()),
                                    "processing_time": processing_time,
                                    "status": "success",
                                    "sentiment_summary": final_state.get("sentiment_summary", {}),  # NEW
                                    "was_re_executed": final_state.get("was_re_executed", False),   # NEW
                                    "metadata": {
                                        "cache_hit": False,
                                        "processing_mode": "enhanced_employee_streaming_workflow",
                                        "employee_role": employee_role,
                                        "is_employee_query": True,
                                        "timestamp": datetime.utcnow().isoformat(),
                                        "has_sentiment_analysis": bool(final_state.get("sentiment_summary"))
                                    }
                                }
                                
                                logger.info(f"[EMPLOYEE_STREAMING] Final result ready with sentiment analysis, processing time: {processing_time:.3f}s")
                                yield {
                                    "event": "final_result",
                                    "data": final_result_data,
                                    "metadata": {
                                        "employee_id": employee_id,
                                        "employee_role": employee_role,
                                        "timestamp": datetime.utcnow().isoformat(),
                                        "processing_time": processing_time,
                                        "has_sentiment_analysis": bool(final_state.get("sentiment_summary"))
                                    }
                                }
                                
                            except Exception as final_processing_error:
                                logger.error(f"[EMPLOYEE_STREAMING] Error processing final result: {str(final_processing_error)}\n{traceback.format_exc()}")
                                yield {
                                    "event": "error",
                                    "data": {
                                        "error": "Error processing final result",
                                        "details": str(final_processing_error),
                                        "node": "final_processing"
                                    },
                                    "metadata": {
                                        "employee_id": employee_id,
                                        "timestamp": datetime.utcnow().isoformat()
                                    }
                                }
                    
                    elif kind == "on_chain_start":
                        # Node start events
                        logger.debug(f"[EMPLOYEE_STREAMING] Node started: {node_name}")
                        yield {
                            "event": "node_start",
                            "data": {
                                "node": node_name,
                                "employee_id": employee_id
                            },
                            "metadata": {
                                "timestamp": datetime.utcnow().isoformat(),
                                "employee_role": employee_role
                            }
                        }
                    
                    elif kind == "on_chain_error":
                        # Error handling
                        error_data = event.get("data", {})
                        error_msg = str(error_data.get("error", "Unknown error"))
                        logger.error(f"[EMPLOYEE_STREAMING] Chain error in node {node_name}: {error_msg}")
                        yield {
                            "event": "error",
                            "data": {
                                "error": error_msg,
                                "node": node_name
                            },
                            "metadata": {
                                "employee_id": employee_id,
                                "employee_role": employee_role,
                                "timestamp": datetime.utcnow().isoformat()
                            }
                        }
                        
                except Exception as event_error:
                    logger.error(f"[EMPLOYEE_STREAMING] Error processing event {event.get('event', 'unknown')}: {str(event_error)}\n{traceback.format_exc()}")
                    yield {
                        "event": "error",
                        "data": {
                            "error": "Error processing workflow event",
                            "details": str(event_error)
                        },
                        "metadata": {
                            "employee_id": employee_id,
                            "employee_role": employee_role,
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    }
        except Exception as stream_error:
            logger.error(f"[EMPLOYEE_STREAMING] Error in graph stream: {str(stream_error)}\n{traceback.format_exc()}")
            yield {
                "event": "error",
                "data": {
                    "error": "Workflow streaming error",
                    "details": str(stream_error)
                },
                "metadata": {
                    "employee_id": employee_id,
                    "employee_role": employee_role,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }

    # --- Enhanced Streaming Methods with Sentiment Analysis ---

    async def arun_streaming(self, query: str, config: Dict, employee_id: str, employee_role: str = "employee", chat_history: Optional[list] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Run enhanced workflow for employee and stream events with comprehensive caching and sentiment analysis.
        """
        if chat_history is None:
            chat_history = []
        start_time = time.time()
        logger.info(f"[EMPLOYEE_STREAMING] Starting enhanced workflow for employee {employee_id} with query: {query[:100]}...")
        
        try:
            # Check cache first for faster responses
            if self.cache_manager.is_active():
                try:
                    # Create employee-specific cache key
                    employee_query = self._create_employee_cache_query(employee_id, employee_role, query)
                    cache_key = self.cache_manager.create_cache_key(employee_query, chat_history)
                    logger.debug(f"[CACHE] Checking enhanced cache with key: {cache_key}")
                    
                    cached_result = await self.cache_manager.get(cache_key)
                    if cached_result:
                        cache_time = time.time() - start_time
                        logger.info(f"[CACHE] Cache HIT for enhanced employee streaming query: {query[:50]}... (retrieved in {cache_time:.3f}s)")
                        # Stream from cache with sentiment data
                        async for event in self._stream_from_cache(cached_result, employee_id, employee_role, config.get("configurable", {}).get("thread_id"), None):
                            if event.get("event") == "final_result" and "metadata" in event.get("data", {}):
                                event["data"]["metadata"]["processing_time"] = cache_time
                                event["metadata"]["processing_time"] = cache_time
                            yield event
                        return
                        
                except Exception as cache_error:
                    logger.error(f"[CACHE] Error checking enhanced cache: {str(cache_error)}\n{traceback.format_exc()}")

            # If cache miss or error, proceed with enhanced execution
            logger.info(f"[EMPLOYEE_STREAMING] Proceeding with enhanced workflow execution")
            
            # Enhanced initial state with sentiment analysis fields
            initial_state = AgentState(
                original_query=query,
                iteration_count=0,
                chat_history=chat_history,
                employee_id=employee_id,
                employee_role=employee_role,
                user_role="employee",
                session_id=config.get("configurable", {}).get("thread_id"),
                timestamp=str(time.time()),
                # Initialize sentiment-related fields
                needs_re_execution=False,
                sentiment_analysis={},
                is_re_execution=False,
                was_re_executed=False
            )
            
            logger.debug(f"[EMPLOYEE_STREAMING] Enhanced initial state created, starting graph stream")
            
            # Stream events from the enhanced graph
            async for event in self._stream_events_from_graph(initial_state, config, employee_id, employee_role):
                
                # Cache successful final results with sentiment data
                if event.get("event") == "final_result":
                    final_data = event.get("data", {})
                    if self.cache_manager.is_active() and final_data.get("status") != "error":
                        try:
                            employee_query = self._create_employee_cache_query(employee_id, employee_role, query)
                            cache_key = self.cache_manager.create_cache_key(employee_query, chat_history)
                            # Cache asynchronously to avoid blocking the stream
                            asyncio.create_task(self._cache_result(cache_key, final_data, query))
                            logger.info(f"[CACHE] Initiated caching for enhanced employee streaming result: {query[:50]}...")
                        except Exception as cache_store_error:
                            logger.error(f"[CACHE] Error storing enhanced result in cache: {str(cache_store_error)}\n{traceback.format_exc()}")
                
                yield event
                
        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"[EMPLOYEE_STREAMING] Critical error in enhanced employee streaming workflow: {str(e)}\n{traceback.format_exc()}")
            yield {
                "event": "error",
                "data": {
                    "error": "Critical enhanced workflow error",
                    "details": str(e)
                },
                "metadata": {
                    "employee_id": employee_id,
                    "employee_role": employee_role,
                    "timestamp": datetime.utcnow().isoformat(),
                    "processing_time": total_time
                }
            }

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
                    cache_key = self.cache_manager.create_cache_key(employee_query, chat_history)
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
                        
                        if event_type == "answer_chunk":
                            chunk_data = event.get("data", "")
                            full_answer += chunk_data
                            logger.debug(f"[EMPLOYEE_SIMPLE] Accumulated answer chunk: {len(chunk_data)} chars")
                            
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
                                    cache_key = self.cache_manager.create_cache_key(employee_query, chat_history)
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
                    cache_key = self.cache_manager.create_cache_key(employee_query, chat_history or [])
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
                            cache_key = self.cache_manager.create_cache_key(employee_query, chat_history or [])
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
