import asyncio
import sys
import uuid
from typing import Dict, Any, Optional, AsyncGenerator, List
from datetime import datetime, timedelta

from loguru import logger
from pathlib import Path

# --- LangGraph Imports ---
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver

# --- Import base components ---
sys.path.append(str(Path(__file__).parent.parent))
from app.agents.workflow.state import GraphState as AgentState  # Sử dụng AgentState đã định nghĩa
from app.agents.workflow.initalize import llm_instance, agent_config
from app.agents.factory.factory_tools import TOOL_FACTORY  # Import factory tools   
from app.agents.stores.entry_agent import EntryAgent
from app.agents.stores.company_agent import CompanyAgent
from app.agents.stores.product_agent import ProductAgent
# Giả định NaiveAgent, RewriterAgent cũng đã được tối ưu
from app.agents.stores.naive_agent import NaiveAgent
from app.agents.stores.rewriter_agent import RewriterAgent
from app.agents.stores.medical_agent import MedicalAgent
from app.agents.stores.genetic_agent import GeneticAgent
from app.agents.stores.drug_agent import DrugAgent
from app.agents.stores.reflection_agent import ReflectionAgent
from app.agents.stores.supervisor_agent import SupervisorAgent
from app.agents.stores.summary_agent import SummaryAgent
from app.agents.stores.question_generator_agent import QuestionGeneratorAgent
from app.agents.stores.guest_agent import GuestAgent
from app.agents.stores.cache_manager import CacheManager
from app.agents.data_storages.response_storages import store_response
import time
class GuestWorkflow:
    """
    Workflow được thiết kế cho người dùng vãng lai (khách).
    Tập trung vào việc cung cấp thông tin chung và giới thiệu.
    Không truy cập vào dữ liệu của khách hàng hoặc nhân viên.
    """
    def __init__(self, max_iterations: int = 2): # Reduced for faster guest responses
        self.max_iterations = max_iterations
        self.agents = self._initialize_agents()
        self.graph = self._build_and_compile_graph()
        
        # Initialize cache manager for workflow caching
        self.cache_manager = CacheManager()
        
        logger.add(Path("app/logs/log_workflows/guest_workflow.log"), rotation="10 MB", level="DEBUG", backtrace=True, diagnose=True)
        logger.info("Guest Workflow initialized with caching support.")

    def _initialize_agents(self) -> Dict[str, Any]:
        """
        Khởi tạo các agent dành riêng cho người dùng vãng lai.
        *** KHÔNG BAO GỒM CustomerAgent và EmployeeAgent. ***
        """
        logger.info("Initializing agents for GUEST Workflow...")
        llm = llm_instance

        return {
            # Các node điều khiển chung
            "entry": EntryAgent(llm=llm),
            "rewriter": RewriterAgent(llm=llm_instance),
            "reflection": ReflectionAgent(llm=llm_instance, default_tool_names=[]),
            "supervisor": SupervisorAgent(llm=llm),
            "question_generator": QuestionGeneratorAgent(llm=llm_instance),

            # Các agent chuyên môn được phép cho khách
            "CompanyAgent": CompanyAgent(llm=llm_instance, default_tool_names=["company_retriever_tool"]),
            "ProductAgent": ProductAgent(llm=llm_instance, default_tool_names=["product_retriever_tool"]),
            "MedicalAgent": MedicalAgent(llm=llm_instance, default_tool_names=["medical_retriever_tool"]),
            "DrugAgent": DrugAgent(llm=llm_instance, default_tool_names=["drug_retriever_tool"]),
            "GeneticAgent": GeneticAgent(llm=llm_instance, default_tool_names=["genetic_retriever_tool"]),
            "NaiveAgent": NaiveAgent(llm=llm, default_tool_names=[]),
            "GuestAgent": GuestAgent(llm=llm_instance, default_tool_names=["company_retriever_tool"]),
            "SummaryAgent": SummaryAgent(llm=llm_instance),

            
        }

    async def _summarize_context_node(self, state: AgentState) -> AgentState:
        """Node that runs the SummaryAgent to condense chat history."""
        logger.info("--- Condensing long chat history... ---")
        summary_agent = self.agents["SummaryAgent"]
        summarized_state = await summary_agent.aexecute(state)
        # Ensure the original query is preserved
        summarized_state['rewritten_query'] = state['rewritten_query']
        return summarized_state 
    
    def _should_summarize(self, state: AgentState) -> str:
        """Checks if the context is too long and needs summarization."""
        logger.info("--- ROUTING: CHECKING CONTEXT LENGTH ---")
        history_len = len(state.get("chat_history", []))
        if history_len > self.chat_history_threshold:
            logger.info(f"Context too long ({history_len} messages). Routing to summarizer.")
            return "summarize_context"
        logger.info("Context length is OK. Proceeding to guardrail.")
        return "context_guardrail"
    
    async def _run_agent(self, state: AgentState) -> AgentState:
        """Node thực thi chung với giám sát kết nối cơ sở dữ liệu."""
        agent_name = state.get("classified_agent")
        if not agent_name or agent_name not in self.agents:
            state['error_message'] = f"Access Denied or Invalid Agent: The requested agent '{agent_name}' is not available in this workflow."
            return state
            
        agent_to_run = self.agents[agent_name]
        logger.info(f"--- Running Specialist Agent: {agent_name} ---")
        
        # Add guest-specific context to state
        guest_id = state.get("guest_id")
        
        # Enhance state with guest information
        enhanced_state = state.copy()
        enhanced_state["user_context"] = {
            "guest_id": guest_id,
            "is_authenticated": False,
            "access_level": "public"
        }
        
        # Get database metrics before executing agent
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
        start_time = time.time()
        result_state = await agent_to_run.aexecute(enhanced_state)
        end_time = time.time()
        logger.info(f"Agent {agent_name} execution time: {end_time - start_time:.2f} seconds")
        # Check database metrics after executing agent
        
        try:
            if db_health:
                try:
                    after_metrics = await db_health.get_pool_stats()
                    logger.debug(f"DB pool after agent {agent_name}: {after_metrics}")
                    
                    # If connections increased, attempt cleanup
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
            logger.error(f"Error handling DB connections: {e}")
            try:
                from app.db.session import close_db_connections
                await close_db_connections()
            except:
                pass
        
        
        preserved_keys = [
            'original_query', 'rewritten_query', 'chat_history',
            'session_id', 'user_role', 'iteration_count', 'agent_thinks',
            'guest_id', 'interaction_id'
        ]
        for key in preserved_keys:
            if key in state and key not in result_state:
                result_state[key] = state[key]
                
        agent_thinks = result_state.get("agent_thinks", {})
        agent_thinks[agent_name] = result_state.get("agent_response")
        result_state["agent_thinks"] = agent_thinks
        
        # Clean up any database connections that might have been left open
        if hasattr(agent_to_run, 'cleanup_connections') and callable(agent_to_run.cleanup_connections):
            await agent_to_run.cleanup_connections()
            
        return result_state

    async def _final_processing_node(self, state: AgentState) -> AgentState:
        """
        Node xử lý cuối cùng, chạy QuestionGenerator.
        Node này chạy SAU KHI supervisor đã stream xong.
        """
        logger.info("--- Running Final Processing (Question Generation) ---")
        question_generator = self.agents["question_generator"]
        # Chạy question generator và cập nhật state
        final_state = await question_generator.aexecute(state)
        return final_state

    def _build_and_compile_graph(self) -> StateGraph:
        """Builds and compiles the streamlined state graph."""
        workflow = StateGraph(AgentState)
        
        # --- Define Nodes ---
        workflow.add_node("entry", self.agents["entry"].aexecute)
        workflow.add_node("rewriter", self.agents["rewriter"].aexecute)
        workflow.add_node("initial_summary", self._summarize_context_node)
        workflow.add_node("specialist_agent", self._run_agent)
        workflow.add_node("reflection", self.agents["reflection"].aexecute)
        workflow.add_node("supervisor", self.agents["supervisor"].astream_execute)
        workflow.add_node("final_processing", self._final_processing_node)
        
        # --- Define Edges for the Streamlined Flow ---
        workflow.set_entry_point("entry")
        
        # 1. Entry -> Rewrite (optional) -> Initial Summary
        workflow.add_conditional_edges("entry", self._route_after_entry, {
            "rewriter": "rewriter",
            "summarize": "initial_summary"
        })
        workflow.add_edge("rewriter", "initial_summary")
        
        # 2. After initial prep, go directly to the specialist
        workflow.add_edge("initial_summary", "specialist_agent")
        
        # 3. After specialist, always reflect
        workflow.add_edge("specialist_agent", "reflection")
        
        # 4. After reflection, go to the supervisor or loop back
        workflow.add_conditional_edges("reflection", self._route_after_reflection, {
            "supervisor": "supervisor",
            "specialist_agent": "specialist_agent",
        })

        # 5. After supervisor streams, do final processing
        workflow.add_edge("supervisor", "final_processing")
        workflow.add_edge("final_processing", END)

        return workflow.compile(checkpointer=InMemorySaver())


    def _route_after_entry(self, state: AgentState) -> str:
        """Decides whether to rewrite the query or proceed to summarization."""
        logger.info("--- ROUTING AFTER ENTRY ---")
        if state.get("needs_rewrite", False):
            logger.info("Decision: Needs rewrite.")
            return "rewriter"
        logger.info("Decision: Proceeding to initial context preparation.")
        return "summarize"

    def _route_after_reflection(self, state: AgentState) -> str:
        """Decides the next step after the Reflection agent runs."""
        logger.info("--- ROUTING AFTER REFLECTION ---")
        iteration_count = state.get("iteration_count", 0) + 1
        state["iteration_count"] = iteration_count

        # Check for termination conditions
        if iteration_count >= self.max_iterations:
            logger.warning(f"Max iterations ({self.max_iterations}) reached. Finalizing with supervisor.")
            return "supervisor"
        if state.get("is_final_answer", False):
            logger.info("Reflection determined the answer is final. Proceeding to supervisor.")
            return "supervisor"

        # Check for follow-up
        followup_agent = state.get("suggest_agent_followups")
        if followup_agent and followup_agent in self.agents:
            logger.info(f"Reflection suggests follow-up with: {followup_agent}")
            state["classified_agent"] = followup_agent
            return "specialist_agent"
            
        logger.info("No more follow-ups. Finalizing with supervisor.")
        return "supervisor"

    def _route_after_reflection(self, state: AgentState) -> str:
        """Decides the next step after the Reflection agent runs."""
        logger.info("--- ROUTING AFTER REFLECTION ---")
        iteration_count = state.get("iteration_count", 0) + 1
        state["iteration_count"] = iteration_count

        if iteration_count >= self.max_iterations:
            logger.warning(f"Max iterations ({self.max_iterations}) reached. Finalizing.")
            return "supervisor"
        if state.get("is_final_answer", False):
            logger.info("Reflection determined the answer is final.")
            return "supervisor"

        followup_agent = state.get("suggest_agent_followups")
        if followup_agent and followup_agent in self.agents:
            logger.info(f"Reflection suggests follow-up with: {followup_agent}")
            state["classified_agent"] = followup_agent
            return "specialist_agent"
            
        logger.info("No more follow-ups. Finalizing with supervisor.")
        return "supervisor"
    
    
    async def arun_streaming(
        self, 
        query: str, 
        config: Dict,  
        guest_id: Optional[str] = None,
        chat_history: Optional[list] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Chạy workflow cho khách và stream các sự kiện.
        Includes intelligent caching to improve response times.
        """
        logger.info(f"Starting guest workflow for query: {query[:100]}...")
        
        # Check cache first for faster responses
        if self.cache_manager.is_active():
            try:
                cache_key = self.cache_manager.create_cache_key(query, chat_history)
                logger.debug(f"Checking cache with key: {cache_key}")
                cached_result = await self.cache_manager.get(cache_key)
                
                if cached_result:
                    logger.info(f"Cache HIT for streaming guest query: {query[:50]}...")
                    # Simulate streaming events from cached result
                    if not guest_id:
                        guest_id = f"guest_{uuid.uuid4()}"
                    
                    interaction_id = str(uuid.uuid4())
                    
                    # Yield answer chunks if available
                    agent_response = cached_result.get("agent_response", "")
                    if agent_response:
                        agent_response = agent_response.split(" ")
                        response_chunk = ''
                        for word in agent_response:
                            response_chunk += word + " "
                            yield {
                                "event": "answer_chunk",
                                "data": response_chunk,
                                "metadata": {
                                    "guest_id": guest_id,
                                    "timestamp": datetime.utcnow().isoformat(),
                                    "cache_hit": True
                                }
                            }
                            # Small delay to simulate streaming
                            await asyncio.sleep(0.01)
                    
                    # Yield final result
                    final_result = cached_result.copy()
                    final_result["guest_id"] = guest_id
                    final_result["session_id"] = config.get("configurable", {}).get("thread_id")
                    if "metadata" in final_result:
                        final_result["metadata"]["cache_hit"] = True
                        final_result["metadata"]["processing_time"] = 0.1
                    
                    yield {
                        "event": "final_result",
                        "data": final_result,
                        "metadata": {
                            "guest_id": guest_id,
                            "interaction_id": interaction_id,
                            "timestamp": datetime.utcnow().isoformat(),
                            "cache_hit": True
                        }
                    }
                    return
                else:
                    logger.debug("No cache hit, proceeding with workflow execution")
            except Exception as e:
                logger.error(f"Error during cache check: {str(e)}", exc_info=True)
                # Continue with normal execution if cache check fails
        
        if not guest_id:
            guest_id = f"guest_{uuid.uuid4()}"
            
        interaction_id = str(uuid.uuid4())
            
        initial_state = AgentState(
            original_query=query,
            iteration_count=0,
            chat_history=chat_history if chat_history else [],
            user_role="guest", 
            guest_id=guest_id,
            interaction_id=interaction_id,
            session_id=config.get("configurable", {}).get("thread_id"),
            timestamp=datetime.utcnow().isoformat()
        )
        
        try:
            logger.info("Starting workflow execution...")
            async for event in self.graph.astream_events(initial_state, config=config, version="v1"):
                try:
                    kind = event.get("event")
                    if not kind:
                        logger.warning(f"Event missing 'event' field: {event}")
                        continue
                        
                    # logger.debug(f"Processing event: {kind}")
                    
                    if kind == "on_chain_stream":
                        chunk = event.get("data", {}).get("chunk")
                        if not chunk:
                            logger.warning("on_chain_stream event missing chunk data")
                            continue
                            
                        if isinstance(chunk, dict) and "agent_response" in chunk:
                            response = chunk.get("agent_response", "")
                            # logger.debug(f"Yielding answer chunk: {response[:100]}...")
                            yield {
                                "event": "answer_chunk",
                                "data": response,
                                "metadata": {
                                    "guest_id": guest_id,
                                    "timestamp": datetime.utcnow().isoformat()
                                }
                            }
                    
                    elif kind == "on_chain_end":
                        node_name = event.get("name", "")
                        # logger.info(f"Processing on_chain_end for node: {node_name}")
                        
                        if node_name == "final_processing":
                            try:
                                final_state = event.get("data", {}).get("output", {})
                                logger.debug(f"Final state: {final_state}")
                                
                                final_result_data = {
                                    "suggested_questions": final_state.get("suggested_questions", []),
                                    "full_final_answer": final_state.get("agent_response", ""),
                                    "agents_used": list(final_state.get("agent_thinks", {}).keys()),
                                    "processing_time": self._calculate_processing_time(initial_state),
                                    "status": "success",
                                    "agent_response": final_state.get("agent_response", ""),
                                    "metadata": {
                                        "cache_hit": False,
                                        "processing_mode": "streaming_workflow"
                                    }
                                }
                                try:
                                    store_response(final_state)
                                except Exception as store_error:
                                    logger.error(f"Error storing response: {store_error}")
                                    continue
                                # Cache the successful result for future requests
                                if self.cache_manager.is_active():
                                    try:
                                        cache_key = self.cache_manager.create_cache_key(query, chat_history)
                                        # Create a cacheable version without session-specific data
                                        cacheable_result = final_result_data.copy()
                                        
                                        # Cache asynchronously to avoid blocking the stream
                                        asyncio.create_task(
                                            self.cache_manager.set(cache_key, cacheable_result, ttl=1800)
                                        )
                                        logger.info(f"Cached streaming result for query: {query[:50]}...")
                                    except Exception as cache_error:
                                        logger.error(f"Error caching result: {str(cache_error)}", exc_info=True)
                                
                                yield {
                                    "event": "final_result",
                                    "data": final_result_data,
                                    "metadata": {
                                        "guest_id": guest_id,
                                        "interaction_id": interaction_id,
                                        "timestamp": datetime.utcnow().isoformat()
                                    }
                                }
                            except Exception as final_processing_error:
                                logger.error(f"Error in final processing: {str(final_processing_error)}", exc_info=True)
                                yield {
                                    "event": "error",
                                    "data": {
                                        "error": f"Error in final processing: {str(final_processing_error)}",
                                        "node": "final_processing"
                                    },
                                    "metadata": {
                                        "guest_id": guest_id,
                                        "timestamp": datetime.utcnow().isoformat()
                                    }
                                }
                    
                    elif kind == "on_chain_start":
                        node_name = event.get("name", "unknown")
                        logger.info(f"Node started: {node_name}")
                        yield {
                            "event": "node_start",
                            "data": {
                                "node": node_name,
                                "guest_id": guest_id
                            }
                        }
                        
                    elif kind == "on_chain_error":
                        error_msg = str(event.get("data", {}).get("error", "Unknown error"))
                        node_name = event.get("name", "unknown")
                        logger.error(f"Error in node {node_name}: {error_msg}")
                        yield {
                            "event": "error",
                            "data": {
                                "error": error_msg,
                                "node": node_name
                            },
                            "metadata": {
                                "guest_id": guest_id,
                                "timestamp": datetime.utcnow().isoformat()
                            }
                        }
                    
                  

                except Exception as event_error:
                    logger.error(f"Error processing event: {str(event_error)}", exc_info=True)
                    yield {
                        "event": "error",
                        "data": {
                            "error": f"Error processing event: {str(event_error)}",
                            "node": "event_processor"
                        },
                        "metadata": {
                            "guest_id": guest_id,
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    }
                    
        except Exception as workflow_error:
            logger.error(f"Workflow execution failed: {str(workflow_error)}", exc_info=True)
            yield {
                "event": "error",
                "data": {
                    "error": f"Workflow execution failed: {str(workflow_error)}",
                    "node": "workflow"
                },
                "metadata": {
                    "guest_id": guest_id,
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

    async def arun_simple(
        self, 
        query: str, 
        guest_id: Optional[str] = None,
        session_id: Optional[str] = None,
        chat_history: Optional[list] = None
    ) -> Dict[str, Any]:
        # Create or get guest session if not provided
        if not session_id or not guest_id:
            guest_id, session_id = await self.cache_manager.create_guest_session(guest_id, session_id)
        
        # If chat history not provided, try to fetch from cache
        if not chat_history:
            chat_history = await self.cache_manager.get_chat_history(session_id)
        """
        Run workflow for guest and return final result.
        Includes intelligent caching to improve response times.
        
        Args:
            query: Guest query
            guest_id: Optional guest identifier
            session_id: Optional session ID
            chat_history: Optional chat history
            
        Returns:
            Final workflow result
        """
        logger.info(f"Chat history length: {len(chat_history) if chat_history else 0}")
        # Check cache first for faster responses
        if self.cache_manager.is_active():
            cache_key = self.cache_manager.create_cache_key(query, chat_history)
            cached_result = await self.cache_manager.get(cache_key)
            
            if cached_result:
                logger.info(f"Cache HIT for simple guest query: {query[:50]}...")
                # Update session-specific data
                cached_result["session_id"] = session_id or cached_result.get("session_id")
                cached_result["guest_id"] = guest_id or cached_result.get("guest_id")
                if "metadata" in cached_result:
                    cached_result["metadata"]["cache_hit"] = True
                    cached_result["metadata"]["processing_time"] = 0.1
                return cached_result
        if not guest_id:
            guest_id = f"guest_{uuid.uuid4()}"
            
        if not session_id:
            session_id = f"guest_{guest_id}_{datetime.utcnow().timestamp()}"
        
        config = {"configurable": {"thread_id": session_id}}
        
        final_result = {}
        full_answer = ""
        
        async for event in self.arun_streaming(query, config, guest_id, chat_history):
            if event.get("event") == "answer_chunk":
                full_answer += event["data"]
            elif event.get("event") == "final_result":
                final_result = event["data"]
                final_result["full_answer"] = full_answer
                break
        
        # Cache the successful result for future requests
        if self.cache_manager.is_active() and final_result.get("status") == "success":
            cache_key = self.cache_manager.create_cache_key(query, chat_history)
            # Create a cacheable version without session-specific data
            cacheable_result = final_result.copy()
            cacheable_result["session_id"] = None
            cacheable_result["guest_id"] = None
            if "metadata" in cacheable_result:
                cacheable_result["metadata"]["cache_hit"] = False
            
            # Cache asynchronously to avoid blocking the response
            asyncio.create_task(self.cache_manager.set(cache_key, cacheable_result, ttl=1800))  # 30 min TTL
            logger.info(f"Cached simple result for query: {query[:50]}...")
        
        return final_result

    async def process_with_load_balancing(
        self, 
        query: str, 
        guest_id: Optional[str] = None,
        session_id: Optional[str] = None,
        chat_history: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Process a guest query with load balancing support.
        
        This method will:
        1. Check if the request should be handled locally or remotely
        2. If remote, forward to another node via load balancer
        3. If local, either process directly or queue for processing
        

        
        Args:
            query: Guest query text
            guest_id: Optional guest identifier
            session_id: Optional session ID for tracking
            chat_history: Optional chat history
            
        Returns:
            Result dictionary with answer and metadata
        """
        try:
            if not guest_id:
                guest_id = f"guest_{uuid.uuid4()}"
                
            # Try to import load balancer and queue manager
            try:
                from app.core.load_balancer import get_load_balancer
                from app.core.queue_manager import get_queue_manager
                load_balancer = get_load_balancer()
                queue_manager = get_queue_manager()
            except ImportError:
                load_balancer = None
                queue_manager = None
                logger.warning("Load balancer or queue manager not available - processing locally")
            
            # Create session ID if not provided
            if not session_id:
                session_id = f"guest_{guest_id}_{datetime.utcnow().timestamp()}"
                
            # Check if we should process this request on another node
            if load_balancer and load_balancer.is_active():
                # Get system load
                local_load = await load_balancer.get_local_load()
                
                # If local load is high, try to find another node
                if local_load.get('cpu_percent', 0) > 70 or local_load.get('memory_percent', 0) > 80:
                    logger.info(f"Local system load is high: {local_load}. Trying to find another node.")
                    
                    # Get best node for guest requests
                    best_node = await load_balancer.get_best_node(request_type="guest_query")
                    
                    if best_node and best_node['id'] != load_balancer.get_node_id():
                        logger.info(f"Forwarding request to node: {best_node['id']}")
                        
                        # Forward request to the selected node
                        response = await load_balancer.forward_request(
                            node=best_node,
                            endpoint="/api/v1/guest/chat",
                            method="POST",
                            payload={
                                "query": query,
                                "guest_id": guest_id,
                                "session_id": session_id,
                                "chat_history": chat_history
                            }
                        )
                        
                        if response and 'result' in response:
                            return response['result']
                            
                        logger.warning(f"Failed to get valid response from node {best_node['id']}")
            
            # If we reach here, process locally (either directly or via queue)
            if queue_manager and queue_manager.is_active():
                # Import priority constants
                from app.core.queue_manager import TaskPriority
                
                # Guest requests are always lowest priority
                priority = TaskPriority.LOW
                
                logger.info(f"Queueing guest query with priority {priority}")
                
                # Add task to queue
                task_id = await queue_manager.add_task(
                    task_func="app.tasks.workflow_tasks.process_guest_query_task",
                    args={
                        "query": query,
                        "guest_id": guest_id,
                        "session_id": session_id,
                        "chat_history": chat_history,
                        "workflow_instance_id": id(self)  # Pass instance ID to identify the workflow
                    },
                    priority=priority
                )
                
                # Wait for task completion with timeout
                result = await queue_manager.wait_for_task(
                    task_id, 
                    timeout=90,  # 1.5 minute timeout (shorter for guests)
                    polling_interval=0.5
                )
                
                if result:
                    return result
                    
                # If waiting timed out but task is still running, return the task_id
                status = await queue_manager.get_task_status(task_id)
                if status == "RUNNING":
                    return {
                        "status": "processing",
                        "task_id": task_id,
                        "message": "Your request is being processed. Please check back soon."
                    }
                
                # If we reach here, something went wrong with the task
                logger.error(f"Task {task_id} failed with status {status}")
                return {
                    "status": "error",
                    "message": "Unable to process your request at this time. Please try again later."
                }
            
            # If no queue manager or load balancer, process directly
            logger.info("Processing query directly without queue or load balancing")
            return await self.arun_simple(query, guest_id, session_id, chat_history)
            
        except Exception as e:
            logger.exception(f"Error in load-balanced processing: {e}")
            return {
                "status": "error",
                "message": "An error occurred while processing your request.",
                "error": str(e)
            }

    async def process_fast_mode(
        self, 
        query: str, 
        guest_id: Optional[str] = None,
        session_id: Optional[str] = None,
        chat_history: Optional[list] = None
    ) -> Dict[str, Any]:
        # Create or get guest session for chat history if not provided
        if not session_id or not guest_id:
            guest_id, session_id = await self.cache_manager.create_guest_session(guest_id, session_id)
        
        # If chat history not provided, try to fetch from cache
        if not chat_history and session_id:
            try:
                chat_history = await self.cache_manager.get_chat_history(session_id)
                logger.info(f"Retrieved {len(chat_history) if chat_history else 0} messages from chat history for session {session_id}")
            except Exception as e:
                logger.error(f"Failed to retrieve chat history: {e}")
                chat_history = []
        """
        Fast processing mode for guest queries - actually runs the workflow.
        Includes intelligent caching to improve response times.
        """
        start_time = datetime.utcnow()
        
        # Check cache first for faster responses
        if self.cache_manager.is_active():
            cache_key = self.cache_manager.create_cache_key(query, chat_history)
            cached_result = await self.cache_manager.get(cache_key)
            
            if cached_result:
                logger.info(f"Cache HIT for guest query: {query[:50]}...")
                # Update metadata with cache info
                cached_result["metadata"]["cache_hit"] = True
                cached_result["metadata"]["processing_time"] = 0.1  # Minimal cache retrieval time
                cached_result["session_id"] = session_id or cached_result.get("session_id")
                cached_result["guest_id"] = guest_id or cached_result.get("guest_id")
                return cached_result
        
        try:
            if not guest_id:
                guest_id = f"guest_{uuid.uuid4().hex[:8]}"
                
            if not session_id:
                session_id = f"fast_{guest_id}_{int(start_time.timestamp())}"
            
            logger.info(f"Processing guest query in FAST MODE: {query[:50]}...")
            
            # Use the actual workflow graph - but with a proper initial state
            initial_state = AgentState(
                original_query=query,
                iteration_count=0,
                chat_history=chat_history if chat_history else [],
                user_role="guest", 
                guest_id=guest_id,
                interaction_id=str(uuid.uuid4()),
                session_id=session_id,
                timestamp=start_time.isoformat(),
                max_iterations=2,  # Reduced for speed
                next_agent="entry"  # Start with entry agent
            )
            
            config = {"configurable": {"thread_id": session_id}}
            
            # Process with timeout using the actual graph
            try:
                logger.info("Running workflow graph...")
                result = await asyncio.wait_for(
                    self.graph.ainvoke(initial_state, config=config),
                    timeout=40  # 8 second timeout (optimized for speed)
                )
                logger.info(f"Workflow completed: {type(result)}")
                
                # Extract meaningful response from the result
                end_time = datetime.utcnow()
                processing_time = (end_time - start_time).total_seconds()
                
                # Get the final response from the workflow result
                agent_response = result.get("agent_response", "")
                if not agent_response:
                    # Try to get response from other fields
                    agent_response = result.get("final_response", "")
                if not agent_response:
                    # Fallback to constructing a response based on the query
                    agent_response = self._generate_contextual_response(query)
                
                suggested_questions = result.get("suggested_questions", [])
                if not suggested_questions:
                    suggested_questions = self._generate_suggested_questions(query)
                
                final_result = {
                    "status": "success",
                    "agent_response": agent_response,
                    "suggested_questions": suggested_questions,
                    "session_id": session_id,
                    "guest_id": guest_id,
                    "chat_history": result.get("chat_history", chat_history or []),
                    "metadata": {
                        "processing_time": processing_time,
                        "processing_mode": "fast_workflow",
                        "load_balancing_used": False,
                        "queue_processing": False,
                        "iteration_count": result.get("iteration_count", 0),
                        "final_agent": result.get("current_agent", "unknown"),
                        "agents_used": list(result.get("agent_thinks", {}).keys()) if result.get("agent_thinks") else ["entry"],
                        "cache_hit": False
                    }
                }
                
                # Cache the successful result for future requests
                if self.cache_manager.is_active():
                    cache_key = self.cache_manager.create_cache_key(query, chat_history)
                    # Create a cacheable version without session-specific data
                    cacheable_result = final_result.copy()
                    cacheable_result["session_id"] = None  # Remove session-specific data for better cache reuse
                    cacheable_result["guest_id"] = None
                    
                    # Cache asynchronously to avoid blocking the response
                    asyncio.create_task(self.cache_manager.set(cache_key, cacheable_result, ttl=1800))  # 30 min TTL
                    logger.info(f"Cached result for query: {query[:50]}...")
                
                logger.info(f"Fast mode processing completed in {processing_time:.2f}s with response: {agent_response[:100]}...")
                
                # Store conversation in chat history
                try:
                    if session_id:
                        # Add conversation turn to history cache
                        user_metadata = {"timestamp": start_time.isoformat()}
                        assistant_metadata = {
                            "processing_time": processing_time,
                            "agents_used": final_result["metadata"]["agents_used"],
                            "suggested_questions": suggested_questions
                        }
                        
                        await self.cache_manager.add_conversation_turn(
                            session_id=session_id,
                            user_message=query,
                            assistant_response=agent_response,
                            user_metadata=user_metadata,
                            assistant_metadata=assistant_metadata
                        )
                        logger.info(f"Stored conversation in chat history for session {session_id}")
                except Exception as e:
                    logger.error(f"Failed to store conversation in chat history: {e}")
                
                return final_result
                
            except asyncio.TimeoutError:
                logger.warning(f"Fast mode processing timed out for query: {query[:50]}")
                timeout_response = self._generate_timeout_response(query, session_id, guest_id)
                
                # Store timeout response in chat history
                try:
                    if session_id:
                        user_metadata = {"timestamp": start_time.isoformat()}
                        assistant_metadata = {"error": "timeout", "processing_time": 40.0}
                        
                        await self.cache_manager.add_conversation_turn(
                            session_id=session_id,
                            user_message=query,
                            assistant_response=timeout_response["agent_response"],
                            user_metadata=user_metadata,
                            assistant_metadata=assistant_metadata
                        )
                except Exception as e:
                    logger.error(f"Failed to store timeout response in chat history: {e}")
                
                return timeout_response
            
        except Exception as e:
            logger.error(f"Error in fast mode processing: {e}")
            end_time = datetime.utcnow()
            processing_time = (end_time - start_time).total_seconds()
            
            error_response = {
                "status": "error",
                "agent_response": self._generate_contextual_response(query),
                "suggested_questions": self._generate_suggested_questions(query),
                "session_id": session_id,
                "guest_id": guest_id,
                "metadata": {
                    "processing_time": processing_time,
                    "processing_mode": "fast_error_fallback",
                    "error": str(e)
                }
            }
            
            # Store error response in chat history
            try:
                if session_id:
                    user_metadata = {"timestamp": start_time.isoformat()}
                    assistant_metadata = {"error": str(e), "processing_time": processing_time}
                    
                    await self.cache_manager.add_conversation_turn(
                        session_id=session_id,
                        user_message=query,
                        assistant_response=error_response["agent_response"],
                        user_metadata=user_metadata,
                        assistant_metadata=assistant_metadata
                    )
            except Exception as ex:
                logger.error(f"Failed to store error response in chat history: {ex}")
                
            return error_response

    def _generate_contextual_response(self, query: str) -> str:
        """Generate a contextual response based on the query content"""
        query_lower = query.lower()
        
        if "genstory" in query_lower:
            return """GeneStory is a leading genetic testing and personalized medicine company that helps people understand their genetic makeup and health risks. We provide comprehensive DNA analysis, genetic counseling, and personalized health recommendations based on your unique genetic profile."""
        
        elif any(word in query_lower for word in ["genetic testing", "dna test", "genetics"]):
            return """Genetic testing analyzes your DNA to identify changes in genes, chromosomes, or proteins. At GeneStory, we offer comprehensive genetic testing that can reveal information about your health risks, drug responses, and inherited conditions. Our tests are processed in certified laboratories with detailed reports and genetic counseling support."""
        
        elif any(word in query_lower for word in ["how", "work", "process"]):
            return """Our genetic testing process is simple: 1) Order your test kit online, 2) Collect a saliva sample at home, 3) Send it to our certified lab, 4) Receive your comprehensive genetic report with personalized insights, and 5) Schedule optional genetic counseling to discuss your results."""
        
        elif any(word in query_lower for word in ["service", "offer", "provide"]):
            return """GeneStory offers a range of genetic testing services including: health predisposition testing, pharmacogenomics (drug response testing), carrier screening, ancestry analysis, and trait testing. We also provide genetic counseling services to help you understand and act on your results."""
        
        elif any(word in query_lower for word in ["price", "cost", "pricing"]):
            return """Our genetic testing packages range from basic health screening to comprehensive genome analysis. Prices start at $99 for basic trait testing up to $499 for our complete health and wellness package. We also accept HSA/FSA payments and offer payment plans."""
        
        elif any(word in query_lower for word in ["genomic", "medicine", "personalized"]):
            return """Genomic medicine uses your genetic information to guide healthcare decisions. GeneStory's personalized medicine approach analyzes your genetic variants to provide insights about disease risks, optimal medications, and lifestyle recommendations tailored specifically to your genetic profile."""
        
        else:
            return f"""Thank you for your question about "{query}". GeneStory is here to help you understand genetics and personalized medicine. We offer comprehensive genetic testing, detailed health insights, and expert genetic counseling to help you make informed decisions about your health."""

    def _generate_suggested_questions(self, query: str) -> List[str]:
        """Generate relevant suggested questions based on the user's query"""
        query_lower = query.lower()
        
        if "genstory" in query_lower:
            return [
                "What genetic tests does GeneStory offer?",
                "How accurate are GeneStory's genetic tests?",
                "How long does it take to get results?",
                "Do you provide genetic counseling?"
            ]
        
        elif any(word in query_lower for word in ["genetic testing", "dna test"]):
            return [
                "What can genetic testing tell me about my health?",
                "How is genetic testing performed?",
                "What's the difference between different types of genetic tests?",
                "Is genetic testing covered by insurance?"
            ]
        
        elif any(word in query_lower for word in ["price", "cost"]):
            return [
                "What genetic testing packages do you offer?",
                "Do you accept insurance or HSA/FSA?",
                "Are there payment plans available?",
                "What's included in each testing package?"
            ]
        
        else:
            return [
                "What is genetic testing?",
                "How does GeneStory work?",
                "What genetic tests do you offer?",
                "How much does genetic testing cost?",
                "How accurate are your genetic tests?"
            ]

    def _generate_timeout_response(self, query: str, session_id: str, guest_id: str) -> Dict[str, Any]:
        """Generate a response when processing times out"""
        return {
            "status": "timeout",
            "agent_response": self._generate_contextual_response(query),
            "suggested_questions": self._generate_suggested_questions(query),
            "session_id": session_id,
            "guest_id": guest_id,
            "metadata": {
                "processing_time": 15.0,
                "processing_mode": "fast_timeout_fallback",
                "error": "timeout"
            }
        }


# Guest-specific utility functions
async def create_guest_workflow_session(
    guest_id: Optional[str] = None
) -> tuple:
    """
    Create a new workflow session for guest using the session-only history cache.
    This ensures guest data is only stored in memory and not persisted long-term.
    """
    # Import here to avoid circular imports
    from app.agents.stores.history_cache import get_history_cache
    
    # Get the session-only history cache instance
    history_cache = get_history_cache()
    
    if not guest_id:
        guest_id = f"guest_{uuid.uuid4().hex[:8]}"
    
    # Create a new session in the cache
    guest_id, session_id = await history_cache.create_session(
        guest_id=guest_id,
        metadata={
            "created_at": datetime.utcnow().isoformat(),
            "session_type": "guest_workflow",
            "is_temporary": True  # Flag indicating this is a temporary session
        }
    )
    
    logger.info(f"Created temporary guest workflow session: {session_id} for guest {guest_id}")
    return guest_id, session_id


# ==============================================================================
# === TEST EXECUTION
# ==============================================================================
if __name__ == "__main__":
    async def main():
        logger.remove(); logger.add(sys.stdout, level="INFO")
        logger.info("====== TESTING GUEST WORKFLOW WITH LOAD BALANCING ======")
        
        workflow_manager = GuestWorkflow()
        
        # Test the process_with_load_balancing method
        query = "What products does GeneStory offer?"
        
        logger.info("-" * 80)
        logger.info(f"🚀 EXECUTING GUEST QUERY WITH LOAD BALANCING: '{query}'")
        
        result = await workflow_manager.process_with_load_balancing(query=query)
        
        logger.info(f"Result: {result}")

    try:
        asyncio.run(main())
    except Exception as e:
        logger.exception(f"Error running main: {e}")
    finally:
        try:
            TOOL_FACTORY.cleanup_singletons()
        except:
            pass