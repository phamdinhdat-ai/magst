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
# **QUAN TRỌNG**: Không import CustomerAgent và các tool liên quan đến khách hàng
from app.agents.workflow.state import GraphState as AgentState  # Sử dụng AgentState đã định nghĩa
from app.agents.workflow.initalize import llm_instance, agent_config, llm_reasoning  # Import phiên bản
from app.agents.factory.factory_tools import TOOL_FACTORY  # Import factory tools
from app.agents.stores.entry_agent import EntryAgent
from app.agents.stores.company_agent import CompanyAgent
from app.agents.stores.product_agent import ProductAgent
from app.agents.stores.visual_agent import VisualAgent
# Giả định NaiveAgent, RewriterAgent cũng đã được tối ưu
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
class EmployeeWorkflow:
    """
    Workflow dành riêng cho nhân viên, được tách biệt hoàn toàn
    khỏi dữ liệu và các agent của khách hàng để đảm bảo bảo mật.
    """
    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.agents = self._initialize_agents()
        self.graph = self._build_and_compile_graph()
        
        # Initialize cache manager for performance optimization
        self.cache_manager = CacheManager()
        
        logger.add(Path("app/logs/log_workflows/employee_workflow.log"), rotation="10 MB", level="DEBUG", backtrace=True, diagnose=True)

        logger.info("Secure Employee Workflow initialized with caching support.")
        
    async def _cache_result(self, cache_key: str, result: Dict[str, Any], query: str) -> None:
        """
        Helper method to cache a result with proper error handling and logging.
        
        Args:
            cache_key: The cache key to use
            result: The result to cache (must be JSON-serializable)
            query: The original query for logging purposes
        """
        try:
            logger.debug(f"[CACHE] Attempting to cache result for key: {cache_key}")
            await self.cache_manager.set(
                cache_key, 
                result, 
                ttl=1800  # 30 minutes TTL
            )
            logger.info(f"[CACHE] Successfully cached result for query: {query[:50]}...")
            logger.debug(f"[CACHE] Cached data keys: {list(result.keys())}")
        except Exception as e:
            logger.error(f"[CACHE] Failed to cache result: {str(e)}", exc_info=True)

    def _initialize_agents(self) -> Dict[str, Any]:
        """
        Khởi tạo các agent dành riêng cho nhân viên.
        *** KHÔNG BAO GỒM CustomerAgent. ***
        """
        logger.info("Initializing agents for SECURE Employee Workflow...")
        llm = llm_instance

        return {
            # Các node điều khiển chung
            "entry": EntryAgent(llm=llm_reasoning),
            "rewriter": RewriterAgent(llm=llm),
            "reflection": ReflectionAgent(llm=llm, default_tool_names=[]),
            "supervisor": SupervisorAgent(llm=llm),
            "question_generator": QuestionGeneratorAgent(llm=llm),
            
            # Các agent chuyên môn được phép cho nhân viên
            "EmployeeAgent": EmployeeAgent(llm=llm, default_tool_names=[]), # Agent chính
            "CompanyAgent": CompanyAgent(llm=llm, default_tool_names=["company_retriever_tool"]),
            "ProductAgent": ProductAgent(llm=llm, default_tool_names=["product_retriever_tool"]),
            "MedicalAgent": MedicalAgent(llm=llm, default_tool_names=["medical_retriever_tool"]),
            "DrugAgent": DrugAgent(llm=llm, default_tool_names=["drug_retriever_tool"]),
            "GeneticAgent": GeneticAgent(llm=llm, default_tool_names=["genetic_retriever_tool"]),
            "VisualAgent": VisualAgent(llm=llm, default_tool_names=["image_analyzer"]),
            "NaiveAgent": NaiveAgent(llm=llm, default_tool_names=["searchweb_tool"]),

        }



    async def _run_agent(self, state: AgentState) -> AgentState:
        """Node thực thi chung với giám sát kết nối cơ sở dữ liệu."""
        agent_name = state.get("classified_agent")
        # Logic này tự động an toàn: nếu EntryAgent có lỡ phân loại nhầm thành
        # 'CustomerAgent', nó sẽ không tìm thấy trong self.agents và báo lỗi.
        if not agent_name or agent_name not in self.agents:
            state['error_message'] = f"Access Denied or Invalid Agent: The requested agent '{agent_name}' is not available in this workflow."
            return state
            
        agent_to_run = self.agents[agent_name]
        logger.info(f"--- Running Specialist Agent: {agent_name} ---")
        
        # Add employee-specific context to state
        employee_id = state.get("employee_id")
        employee_role = state.get("employee_role", "employee")
        
        # Enhance state with employee information
        enhanced_state = state.copy()
        enhanced_state["user_context"] = {
            "employee_id": employee_id,
            "employee_role": employee_role,
            "is_authenticated": True
        }
        
        # Get database metrics before executing agent
        db_health = None
        before_metrics = {}
        try:
            from app.core.db_health_checker import get_db_health_checker, init_db_health_checker
            db_health = get_db_health_checker()
            if not db_health:
                # Try to initialize the health checker if it's not available
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
        
        result_state = await agent_to_run.aexecute(enhanced_state)
        
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
                    # Always try to clean up connections
                    from app.db.session import close_db_connections
                    await close_db_connections()
            else:
                # Even without metrics, try to clean up connections
                from app.db.session import close_db_connections
                await close_db_connections()
        except Exception as e:
            logger.error(f"Error handling DB connections: {e}")
            # As a last resort, try to clean up
            try:
                from app.db.session import close_db_connections
                await close_db_connections()
            except:
                pass
        
        preserved_keys = [
            'original_query', 'rewritten_query', 'chat_history', 
            'employee_id', 'session_id', 'user_role', 'employee_role',
            'iteration_count', 'agent_thinks', 'interaction_id'
        ]
        for key in preserved_keys:
            if key in state and key not in result_state:
                result_state[key] = state[key]

        agent_thinks = result_state.get("agent_thinks", {})
        agent_thinks[agent_name] = result_state.get("agent_response")
        result_state["agent_thinks"] = agent_thinks

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

    def _build_and_compile_graph(self) -> AgentState:
        """Xây dựng và biên dịch graph."""
        workflow = StateGraph(AgentState)
        workflow.add_node("entry", self.agents["entry"].aexecute)
        workflow.add_node("rewriter", self.agents["rewriter"].aexecute)
        workflow.add_node("specialist_agent", self._run_agent)
        workflow.add_node("reflection", self.agents["reflection"].aexecute)
        workflow.add_node("supervisor", self.agents["supervisor"].astream_execute) 
        workflow.add_node("final_processing", self._final_processing_node)
        workflow.set_entry_point("entry")
        workflow.add_conditional_edges("entry", self._route_after_entry)
        workflow.add_edge("rewriter", "entry")
        workflow.add_edge("specialist_agent", "reflection")
        workflow.add_conditional_edges("reflection", self._route_after_reflection_with_loop)
        workflow.add_edge("supervisor", "final_processing")
        workflow.add_edge("final_processing", END)
        return workflow.compile(checkpointer=InMemorySaver())

    def _route_after_entry(self, state: AgentState) -> str:
        """Routing logic (Tái sử dụng 100%)."""
        if state.get("needs_rewrite", False): return "rewriter"
        agent_name = state.get("classified_agent")
        if agent_name in self.agents: return "specialist_agent"
        state["classified_agent"] = "NaiveAgent"
        return "specialist_agent"

    def _route_after_reflection_with_loop(self, state: AgentState) -> str:
        """Routing logic với vòng lặp (Tái sử dụng 100%)."""
        iteration_count = state.get("iteration_count", 0) + 1
        state["iteration_count"] = iteration_count
        if state.get("error_message"): return END
        if iteration_count >= self.max_iterations: return "supervisor"
        if state.get("is_final_answer", False): return "supervisor"
        followup_agent = state.get("suggest_agent_followups")
        if followup_agent and followup_agent in self.agents:
            state["classified_agent"] = followup_agent
            return "specialist_agent"
        return "supervisor"
    
    
    async def arun_streaming(self, query: str, config: Dict, employee_id: str, employee_role: str = "employee", chat_history: Optional[list] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Run workflow for employee and stream events with comprehensive caching and error handling.
        Includes intelligent caching to improve response times.
        """
        if chat_history is None:
            chat_history = []
            
        start_time = time.time()
        logger.info(f"[EMPLOYEE_STREAMING] Starting workflow for employee {employee_id} with query: {query[:100]}...")
        
        try:
            # Check cache first for faster responses
            if self.cache_manager.is_active():
                try:
                    # Create employee-specific cache key by modifying the query
                    employee_query = f"[EMPLOYEE:{employee_id}:{employee_role}] {query}"
                    cache_key = self.cache_manager.create_cache_key(
                        employee_query, 
                        chat_history
                    )
                    logger.debug(f"[CACHE] Checking cache with key: {cache_key}")
                    
                    cached_result = await self.cache_manager.get(cache_key)
                    
                    if cached_result:
                        cache_time = time.time() - start_time
                        logger.info(f"[CACHE] Cache HIT for employee streaming query: {query[:50]}... (retrieved in {cache_time:.3f}s)")
                        
                        # Simulate streaming events from cached result
                        interaction_id = str(uuid.uuid4())
                        
                        try:
                            # Yield answer chunks if available
                            agent_response = cached_result.get("agent_response", "")
                            if agent_response:
                                # Split response into chunks for streaming effect
                                agent_response = agent_response.split("")
                                total_chunks = len(agent_response)
                                
                                logger.debug(f"[CACHE] Streaming {total_chunks} chunks from cached response")

                                for i, word in enumerate(agent_response):
                                    response_chunk = response_chunk + ' ' + word
                                        
                                    yield {
                                        "event": "answer_chunk",
                                        "data": response_chunk,
                                        "metadata": {
                                            "employee_id": employee_id,
                                            "employee_role": employee_role,
                                            "timestamp": datetime.utcnow().isoformat(),
                                            "cache_hit": True,
                                            "chunk_index": i,
                                            "total_chunks": total_chunks
                                        }
                                    }
                                    # Small delay to simulate streaming
                                    await asyncio.sleep(0.01)
                            
                            # Yield final result
                            final_result = cached_result.copy()
                            final_result["employee_id"] = employee_id
                            final_result["session_id"] = config.get("configurable", {}).get("thread_id")
                            
                            if "metadata" not in final_result:
                                final_result["metadata"] = {}
                            final_result["metadata"].update({
                                "cache_hit": True,
                                "processing_time": cache_time,
                                "cached_response": True
                            })
                            
                            yield {
                                "event": "final_result",
                                "data": final_result,
                                "metadata": {
                                    "employee_id": employee_id,
                                    "employee_role": employee_role,
                                    "interaction_id": interaction_id,
                                    "timestamp": datetime.utcnow().isoformat(),
                                    "cache_hit": True,
                                    "processing_time": cache_time
                                }
                            }
                            return
                            
                        except Exception as cache_stream_error:
                            logger.error(f"[CACHE] Error streaming cached result: {str(cache_stream_error)}\n{traceback.format_exc()}")
                            # Continue with normal execution if cache streaming fails
                            
                    else:
                        logger.debug(f"[CACHE] Cache MISS for employee streaming query: {query[:50]}...")
                        
                except Exception as cache_error:
                    logger.error(f"[CACHE] Error checking cache: {str(cache_error)}\n{traceback.format_exc()}")
                    # Continue with normal execution if cache check fails
            
            # If cache miss or error, proceed with normal execution
            logger.info(f"[EMPLOYEE_STREAMING] Proceeding with normal workflow execution")
            
            initial_state = AgentState(
                original_query=query,
                iteration_count=0,
                chat_history=chat_history,
                employee_id=employee_id,
                employee_role=employee_role,
                user_role="employee",
                session_id=config.get("configurable", {}).get("thread_id"),
                timestamp=datetime.utcnow().isoformat()
            )
            
            logger.debug(f"[EMPLOYEE_STREAMING] Initial state created, starting graph stream")
            
            try:
                async for event in self.graph.astream_events(initial_state, config=config, version="v1"):
                    try:
                        kind = event.get("event")
                        logger.debug(f"[EMPLOYEE_STREAMING] Processing event: {kind}")
                        
                        if kind == "on_chain_stream":
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
                                        "cache_hit": False
                                    }
                                }
                        
                        elif kind == "on_chain_end":
                            # Handle node completion events
                            node_name = event.get("name")
                            logger.debug(f"[EMPLOYEE_STREAMING] Node completed: {node_name}")
                            
                            if node_name == "final_processing":
                                # Final processing node completed - yield final result
                                try:
                                    final_state = event.get("data", {}).get("output", {})
                                    processing_time = time.time() - start_time
                                    
                                    final_result_data = {
                                        "suggested_questions": final_state.get("suggested_questions", []),
                                        "agent_response": final_state.get("agent_response", ""),
                                        "full_final_answer": final_state.get("agent_response", ""),
                                        "agents_used": list(final_state.get("agent_thinks", {}).keys()),
                                        "processing_time": processing_time,
                                        "status": "success",
                                        "metadata": {
                                            "cache_hit": False,
                                            "processing_mode": "streaming_workflow",
                                            "employee_role": employee_role,
                                            "is_employee_query": True,
                                            "timestamp": datetime.utcnow().isoformat()
                                        }
                                    }
                                    
                                    logger.info(f"[EMPLOYEE_STREAMING] Final result ready, processing time: {processing_time:.3f}s")
                                    
                                    yield {
                                        "event": "final_result",
                                        "data": final_result_data,
                                        "metadata": {
                                            "employee_id": employee_id,
                                            "employee_role": employee_role,
                                            "timestamp": datetime.utcnow().isoformat(),
                                            "processing_time": processing_time
                                        }
                                    }
                                    
                                    # Cache the successful result for future requests
                                    if self.cache_manager.is_active():
                                        try:
                                            # Use same cache key generation as in the beginning
                                            employee_query = f"[EMPLOYEE:{employee_id}:{employee_role}] {query}"
                                            cache_key = self.cache_manager.create_cache_key(
                                                employee_query, 
                                                chat_history
                                            )
                                            logger.debug(f"[CACHE] Storing result with key: {cache_key}")
                                            
                                            # Create a cacheable version without session-specific data
                                            cacheable_result = final_result_data.copy()
                                            cacheable_result.pop("employee_id", None)
                                            cacheable_result.pop("session_id", None)
                                            
                                            # Add cache metadata
                                            cacheable_result["cached_at"] = datetime.utcnow().isoformat()
                                            
                                            # Cache asynchronously to avoid blocking the stream
                                            asyncio.create_task(
                                                self._cache_result(cache_key, cacheable_result, query)
                                            )
                                            logger.info(f"[CACHE] Initiated caching for employee streaming result: {query[:50]}...")
                                            
                                        except Exception as cache_store_error:
                                            logger.error(f"[CACHE] Error storing result in cache: {str(cache_store_error)}\n{traceback.format_exc()}")
                                            
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
                            node_name = event.get("name")
                            logger.debug(f"[EMPLOYEE_STREAMING] Node started: {node_name}")
                            
                            yield {
                                "event": "node_start",
                                "data": {
                                    "node": node_name,
                                    "employee_id": employee_id
                                },
                                "metadata": {
                                    "timestamp": datetime.utcnow().isoformat()
                                }
                            }
                        
                        elif kind == "on_chain_error":
                            # Error handling
                            error_data = event.get("data", {})
                            error_msg = str(error_data.get("error", "Unknown error"))
                            node_name = event.get("name", "unknown")
                            
                            logger.error(f"[EMPLOYEE_STREAMING] Chain error in node {node_name}: {error_msg}")
                            
                            yield {
                                "event": "error",
                                "data": {
                                    "error": error_msg,
                                    "node": node_name
                                },
                                "metadata": {
                                    "employee_id": employee_id,
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
                        "timestamp": datetime.utcnow().isoformat()
                    }
                }
                
        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"[EMPLOYEE_STREAMING] Critical error in employee streaming workflow: {str(e)}\n{traceback.format_exc()}")
            yield {
                "event": "error",
                "data": {
                    "error": "Critical workflow error",
                    "details": str(e)
                },
                "metadata": {
                    "employee_id": employee_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "processing_time": total_time
                }
            }

    async def _cache_result(self, cache_key: str, result: Dict[str, Any], query: str, ttl: int = 1800) -> None:
        """
        Helper method to cache a result with proper error handling and logging.
        
        Args:
            cache_key: The cache key to use
            result: The result to cache (must be JSON-serializable)
            query: The original query for logging purposes
            ttl: Time to live in seconds (default 30 minutes)
        """
        try:
            if not self.cache_manager.is_active():
                return
                
            start_time = time.time()
            logger.debug(f"[CACHE] Attempting to cache result for key: {cache_key}")
            
            # Create a safe copy of the result to avoid modifying the original
            cacheable_result = result.copy()
            
            # Remove any sensitive or session-specific data
            cacheable_result.pop("session_id", None)
            cacheable_result.pop("interaction_id", None)
            cacheable_result.pop("employee_id", None)
            
            # Add cache metadata
            if "metadata" not in cacheable_result:
                cacheable_result["metadata"] = {}
                
            cacheable_result["metadata"]["cached_at"] = datetime.utcnow().isoformat()
            
            await self.cache_manager.set(cache_key, cacheable_result, ttl=ttl)
            
            duration = time.time() - start_time
            logger.info(
                f"[CACHE] Successfully cached employee result for query: {query[:50]}... "
                f"(key: {cache_key}, size: {len(str(cacheable_result))} bytes, "
                f"duration: {duration:.3f}s, ttl: {ttl}s)"
            )
            
        except Exception as e:
            logger.error(
                f"[CACHE] Failed to cache employee result: {str(e)}\n"
                f"Cache key: {cache_key}\n"
                f"Error details: {traceback.format_exc()}"
            )

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
        employee_id: str,
        employee_role: str = "employee",
        session_id: Optional[str] = None,
        chat_history: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Run workflow for employee and return final result with comprehensive caching and error handling.
        Includes intelligent caching to improve response times.
        
        Args:
            query: Employee query
            employee_id: Employee ID
            employee_role: Employee role
            session_id: Optional session ID
            chat_history: Optional chat history
            
        Returns:
            Final workflow result
        """
        if chat_history is None:
            chat_history = []
            
        start_time = time.time()
        logger.info(f"[EMPLOYEE_SIMPLE] Starting simple workflow for employee {employee_id} with query: {query[:100]}...")
        
        try:
            # Check cache first for faster responses
            if self.cache_manager.is_active():
                try:
                    # Create employee-specific cache key by modifying the query
                    employee_query = f"[EMPLOYEE:{employee_id}:{employee_role}] {query}"
                    cache_key = self.cache_manager.create_cache_key(
                        employee_query, 
                        chat_history
                    )
                    logger.debug(f"[CACHE] Checking cache with key: {cache_key}")
                    
                    cached_result = await self.cache_manager.get(cache_key)
                    
                    if cached_result:
                        cache_time = time.time() - start_time
                        logger.info(f"[CACHE] Cache HIT for employee simple query: {query[:50]}... (retrieved in {cache_time:.3f}s)")
                        
                        # Update result with current session info
                        result = cached_result.copy()
                        result["employee_id"] = employee_id
                        result["session_id"] = session_id
                        
                        # Update metadata with cache info
                        if "metadata" not in result:
                            result["metadata"] = {}
                        result["metadata"].update({
                            "cache_hit": True,
                            "processing_time": cache_time,
                            "cached_response": True
                        })
                        
                        return result
                        
                    else:
                        logger.debug(f"[CACHE] Cache MISS for employee simple query: {query[:50]}...")
                        
                except Exception as cache_error:
                    logger.error(f"[CACHE] Error checking cache: {str(cache_error)}\n{traceback.format_exc()}")
                    # Continue with normal execution if cache check fails
            
            # If cache miss or error, proceed with normal execution
            logger.info(f"[EMPLOYEE_SIMPLE] Proceeding with normal workflow execution")
            
            if not session_id:
                session_id = f"employee_{employee_id}_{datetime.utcnow().timestamp()}"
            
            config = {"configurable": {"thread_id": session_id}}
            
            final_result = {}
            full_answer = ""
            error_occurred = False
            
            try:
                async for event in self.arun_streaming(query, config, employee_id, employee_role, chat_history):
                    try:
                        event_type = event.get("event")
                        logger.debug(f"[EMPLOYEE_SIMPLE] Processing event: {event_type}")
                        
                        if event_type == "answer_chunk":
                            chunk_data = event.get("data", "")
                            full_answer += chunk_data
                            logger.debug(f"[EMPLOYEE_SIMPLE] Accumulated answer chunk: {len(chunk_data)} chars")
                            
                        elif event_type == "final_result":
                            final_result = event.get("data", {})
                            final_result["full_answer"] = full_answer
                            final_result["employee_id"] = employee_id
                            final_result["session_id"] = session_id
                            
                            processing_time = time.time() - start_time
                            logger.info(f"[EMPLOYEE_SIMPLE] Final result ready, processing time: {processing_time:.3f}s")
                            
                            # Update metadata
                            if "metadata" not in final_result:
                                final_result["metadata"] = {}
                            final_result["metadata"]["processing_time"] = processing_time
                            
                            # Cache the successful result for future requests
                            if self.cache_manager.is_active() and final_result.get("status") != "error":
                                try:
                                    # Use same cache key as in the beginning
                                    employee_query = f"[EMPLOYEE:{employee_id}:{employee_role}] {query}"
                                    cache_key = self.cache_manager.create_cache_key(
                                        employee_query, 
                                        chat_history
                                    )
                                    
                                    # Cache asynchronously to avoid blocking the response
                                    asyncio.create_task(
                                        self._cache_result(cache_key, final_result, query)
                                    )
                                    logger.info(f"[CACHE] Initiated caching for employee simple result: {query[:50]}...")
                                    
                                except Exception as cache_store_error:
                                    logger.error(f"[CACHE] Error storing result in cache: {str(cache_store_error)}\n{traceback.format_exc()}")
                            
                            break
                            
                        elif event_type == "error":
                            error_data = event.get("data", {})
                            error_msg = error_data.get("error", "Unknown error")
                            logger.error(f"[EMPLOYEE_SIMPLE] Error event received: {error_msg}")
                            
                            final_result = {
                                "status": "error",
                                "error": error_msg,
                                "details": error_data.get("details", ""),
                                "employee_id": employee_id,
                                "session_id": session_id,
                                "processing_time": time.time() - start_time,
                                "metadata": {
                                    "cache_hit": False,
                                    "processing_mode": "simple_workflow",
                                    "employee_role": employee_role,
                                    "is_employee_query": True,
                                    "timestamp": datetime.utcnow().isoformat()
                                }
                            }
                            error_occurred = True
                            break
                            
                    except Exception as event_error:
                        logger.error(f"[EMPLOYEE_SIMPLE] Error processing event {event.get('event', 'unknown')}: {str(event_error)}\n{traceback.format_exc()}")
                        continue
                        
            except Exception as stream_error:
                logger.error(f"[EMPLOYEE_SIMPLE] Error in streaming workflow: {str(stream_error)}\n{traceback.format_exc()}")
                final_result = {
                    "status": "error",
                    "error": "Streaming workflow error",
                    "details": str(stream_error),
                    "employee_id": employee_id,
                    "session_id": session_id,
                    "processing_time": time.time() - start_time,
                    "metadata": {
                        "cache_hit": False,
                        "processing_mode": "simple_workflow",
                        "employee_role": employee_role,
                        "is_employee_query": True,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                }
            
            # Return result or default error result
            if not final_result:
                final_result = {
                    "status": "error",
                    "error": "No result received from workflow",
                    "employee_id": employee_id,
                    "session_id": session_id,
                    "processing_time": time.time() - start_time,
                    "metadata": {
                        "cache_hit": False,
                        "processing_mode": "simple_workflow",
                        "employee_role": employee_role,
                        "is_employee_query": True,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                }
            
            return final_result
            
        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"[EMPLOYEE_SIMPLE] Critical error in employee simple workflow: {str(e)}\n{traceback.format_exc()}")
            
            return {
                "status": "error",
                "error": "Critical workflow error",
                "details": str(e),
                "employee_id": employee_id,
                "session_id": session_id,
                "processing_time": total_time,
                "metadata": {
                    "cache_hit": False,
                    "processing_mode": "simple_workflow",
                    "employee_role": employee_role,
                    "is_employee_query": True,
                    "timestamp": datetime.utcnow().isoformat()
                }
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
        Process an employee query with load balancing support.
        
        This method will:
        1. Check if the request should be handled locally or remotely
        2. If remote, forward to another node via load balancer
        3. If local, either process directly or queue for processing
        
        Args:
            query: Employee query text
            employee_id: Employee ID
            employee_role: Employee role
            session_id: Optional session ID for tracking
            prioritize: Whether this request should be prioritized in the queue
            
        Returns:
            Result dictionary with answer and metadata
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
                logger.warning("Load balancer or queue manager not available - processing locally")
            
            # Create session ID if not provided
            if not session_id:
                session_id = f"employee_{employee_id}_{datetime.utcnow().timestamp()}"
                
            # Check if we should process this request on another node
            if load_balancer and load_balancer.is_active():
                # Get system load
                local_load = await load_balancer.get_local_load()
                
                # If local load is high, try to find another node
                if local_load.get('cpu_percent', 0) > 70 or local_load.get('memory_percent', 0) > 80:
                    logger.info(f"Local system load is high: {local_load}. Trying to find another node.")
                    
                    # Get best node for employee requests
                    best_node = await load_balancer.get_best_node(request_type="employee_query")
                    
                    if best_node and best_node['id'] != load_balancer.get_node_id():
                        logger.info(f"Forwarding request to node: {best_node['id']}")
                        
                        # Forward request to the selected node
                        response = await load_balancer.forward_request(
                            node=best_node,
                            endpoint="/api/v1/employee/chat",
                            method="POST",
                            payload={
                                "query": query,
                                "employee_id": employee_id,
                                "employee_role": employee_role,
                                "session_id": session_id
                            }
                        )
                        
                        if response and 'result' in response:
                            return response['result']
                            
                        logger.warning(f"Failed to get valid response from node {best_node['id']}")
            
            # If we reach here, process locally (either directly or via queue)
            if queue_manager and queue_manager.is_active():
                # Import priority constants
                from app.core.queue_manager import TaskPriority
                
                # Determine priority based on employee role and prioritize flag
                priority = TaskPriority.HIGH if (
                    prioritize or 
                    employee_role in ("manager", "director", "executive")
                ) else TaskPriority.NORMAL
                
                logger.info(f"Queueing employee query with priority {priority}")
                
                # Add task to queue
                task_id = await queue_manager.add_task(
                    task_func="app.tasks.workflow_tasks.process_employee_query_task",
                    args={
                        "query": query,
                        "employee_id": employee_id,
                        "employee_role": employee_role,
                        "session_id": session_id,
                        "workflow_instance_id": id(self)  # Pass instance ID to identify the workflow
                    },
                    priority=priority
                )
                
                # Wait for task completion with timeout
                result = await queue_manager.wait_for_task(
                    task_id, 
                    timeout=120,  # 2 minute timeout
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
                        "message": "Your request is still being processed. Please check back later."
                    }
                
                # If we reach here, something went wrong with the task
                logger.error(f"Task {task_id} failed with status {status}")
                return {
                    "status": "error",
                    "message": "Failed to process your request. Please try again."
                }
            
            # If no queue manager or load balancer, process directly
            logger.info("Processing query directly without queue or load balancing")
            return await self.arun_simple(query, employee_id, employee_role, session_id)
            
        except Exception as e:
            logger.exception(f"Error in load-balanced processing: {e}")
            return {
                "status": "error",
                "message": "An error occurred while processing your request.",
                "error": str(e)
            }
            
            
    async def arun_streaming_authenticated(
        self, 
        query: str, 
        config: Dict, 
        employee_id: int,  # Explicitly typed as int
        employee_role: str = "employee",
        interaction_id: Optional[uuid.UUID] = None,
        chat_history: Optional[list] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Run workflow for authenticated employee and stream events.
        Includes intelligent caching to improve response times.
        
        Args:
            query: Employee query
            config: LangGraph configuration
            employee_id: Authenticated employee ID (integer)
            employee_role: Employee role (employee, manager, admin)
            interaction_id: Optional interaction ID for tracking
            chat_history: Optional chat history
        """
        try:
            # Validate inputs to catch errors early
            if not isinstance(query, str) or not query.strip():
                logger.error("Invalid query: empty or not a string")
                yield {
                    "event": "error", 
                    "data": {"error": "Invalid query"}
                }
                return
                
            if not config or not isinstance(config, dict):
                logger.error(f"Invalid config: {config}")
                config = {"configurable": {}}
            
            # Check cache first for faster responses
            if self.cache_manager.is_active():
                try:
                    # Create employee-specific cache key by modifying the query
                    employee_query = f"[EMPLOYEE:{employee_id}:{employee_role}] {query}"
                    cache_key = self.cache_manager.create_cache_key(
                        employee_query, 
                        chat_history or []
                    )
                    logger.debug(f"[CACHE] Checking authenticated streaming cache with key: {cache_key}")
                    
                    cached_result = await self.cache_manager.get(cache_key)
                    
                    if cached_result:
                        logger.info(f"[CACHE] Cache HIT for authenticated employee streaming query: {query[:50]}...")
                        
                        try:
                            # Simulate streaming events from cached result
                            interaction_id_str = str(interaction_id) if interaction_id else str(uuid.uuid4())
                            
                            # Yield answer chunks if available
                            agent_response = cached_result.get("agent_response", "")
                            if agent_response:
                                logger.debug(f"[CACHE] Streaming {len(agent_response)} characters from cache")
                                # Split response into chunks for streaming effect
                                agent_response = agent_response.split(" ")
                                response_chunk = ''
                                for i, chunk in enumerate(agent_response):
                                    response_chunk += chunk + ' '
                                    yield {
                                        "event": "answer_chunk",
                                        "data": response_chunk,
                                        "metadata": {
                                            "employee_id": str(employee_id),
                                            "employee_role": employee_role,
                                            "timestamp": datetime.utcnow().isoformat(),
                                            "cache_hit": True,
                                            "chunk_index": i
                                        }
                                    }
                                    # Small delay to simulate streaming
                                    await asyncio.sleep(0.01)
                            
                            # Yield final result
                            final_result = cached_result.copy()
                            final_result["employee_id"] = str(employee_id)
                            final_result["session_id"] = config.get("configurable", {}).get("thread_id")
                            final_result["interaction_id"] = interaction_id_str
                            
                            if "metadata" not in final_result:
                                final_result["metadata"] = {}
                            final_result["metadata"].update({
                                "cache_hit": True,
                                "processing_time": 0.1,
                                "cached_response": True,
                                "processing_mode": "authenticated_streaming_workflow"
                            })
                            
                            yield {
                                "event": "final_result",
                                "data": final_result,
                                "metadata": {
                                    "employee_id": str(employee_id),
                                    "employee_role": employee_role,
                                    "interaction_id": interaction_id_str,
                                    "timestamp": datetime.utcnow().isoformat(),
                                    "cache_hit": True
                                }
                            }
                            logger.info(f"[CACHE] Successfully streamed cached result for authenticated employee query")
                            return
                            
                        except Exception as cache_stream_error:
                            logger.error(f"[CACHE] Error streaming cached result: {str(cache_stream_error)}\n{traceback.format_exc()}")
                            # Continue with normal execution if cache streaming fails
                    else:
                        logger.debug(f"[CACHE] Cache MISS for authenticated employee streaming query: {query[:50]}...")
                        
                except Exception as cache_error:
                    logger.error(f"[CACHE] Error checking cache for authenticated streaming: {str(cache_error)}\n{traceback.format_exc()}")
                    # Continue with normal execution if cache check fails
                
            # Log detailed information for debugging
            logger.info(f"Starting authenticated workflow stream for employee {employee_id}")
            logger.debug(f"Query: '{query}', Config: {config}, Role: {employee_role}")
            # Setup initial state with defensive conversions
            initial_state = AgentState(
                original_query=query,
                iteration_count=0,
                chat_history=chat_history if chat_history else [],
                user_role="employee",
                employee_id=str(employee_id) if employee_id is not None else "",  # Safe conversion
                employee_role=employee_role or "employee",  # Default if None
                interaction_id=str(interaction_id) if interaction_id else None,
                session_id=config.get("configurable", {}).get("thread_id", ""),
                timestamp=datetime.utcnow().isoformat()
            )
            
            # Ensure we have a valid graph
            if not hasattr(self, 'graph') or self.graph is None:
                logger.error("Workflow graph not initialized")
                yield {
                    "event": "error",
                    "data": {"error": "Workflow not properly initialized"}
                }
                return
                
            # Stream events with proper error handling
            try:
                async for event in self.graph.astream_events(initial_state, config=config, version="v1"):
                    # CHECK: NODE NAME is supervisor_agent
                    if not event or not isinstance(event, dict):
                        logger.error(f"Invalid event received: {event}")
                        yield {
                            "event": "error",
                            "data": {"error": "Invalid event format"}
                        }
                        continue
                    # logger.debug(f"[EMPLOYEE_STREAMING] Received event: {event}")
                    if "event" not in event or "data" not in event:
                        logger.error(f"Event missing required fields: {event}")
                        yield {
                            "event": "error",
                            "data": {"error": "Event missing required fields"}
                        }
                        continue
                    # Process event based on type
                    node_name = event.get("langgraph_node", "unknown")
                    if not node_name:
                        logger.error(f"Event missing node name: {event}")
                        yield {
                            "event": "error",
                            "data": {"error": "Event missing node name"}
                        }
                        continue


                    # Handle different event types

                    kind = event["event"]
                    node_name = event.get("langgraph_node", "unknown")
                    if kind == "on_chain_stream":
                        # Stream chunks from SupervisorAgent
                        chunk = event["data"]["chunk"]
                        if isinstance(chunk, dict) and "agent_response" in chunk:
                            yield {
                                "event": "answer_chunk",
                                "data": chunk.get("agent_response", ""),
                                "metadata": {
                                    "employee_id": str(employee_id),
                                    "employee_role": employee_role,
                                    "timestamp": datetime.utcnow().isoformat()
                                }
                            }
                    
                    elif kind == "on_chain_end":
                        node_name = event["name"]
                        if node_name == "final_processing":
                            # Final result with suggested questions
                            final_state = event["data"]["output"]
                            final_result_data = {
                                "suggested_questions": final_state.get("suggested_questions", []),
                                "agent_response": final_state.get("agent_response", ""),
                                "full_final_answer": final_state.get("agent_response", ""),
                                "agents_used": list(final_state.get("agent_thinks", {}).keys()),
                                "interaction_id": str(interaction_id) if interaction_id else None,
                                "processing_time": self._calculate_processing_time(initial_state),
                                "status": "success",
                                "metadata": {
                                    "cache_hit": False,
                                    "processing_mode": "authenticated_streaming_workflow",
                                    "employee_role": employee_role,
                                    "is_employee_query": True
                                }
                            }
                            try:
                                store_response(final_state)
                            except Exception as store_error:
                                logger.error(f"Error storing response: {store_error}")
                                continue
                            yield {
                                "event": "final_result",
                                "data": final_result_data,
                                "metadata": {
                                    "employee_id": str(employee_id),
                                    "employee_role": employee_role,
                                    "timestamp": datetime.utcnow().isoformat()
                                }
                            }
                            
                            # Cache the successful result for future requests
                            if self.cache_manager.is_active():
                                try:
                                    # Use same cache key generation as in the beginning
                                    employee_query = f"[EMPLOYEE:{employee_id}:{employee_role}] {query}"
                                    cache_key = self.cache_manager.create_cache_key(
                                        employee_query, 
                                        chat_history or []
                                    )
                                    
                                    # Cache asynchronously using the helper method
                                    asyncio.create_task(
                                        self._cache_result(cache_key, final_result_data, query)
                                    )
                                    logger.info(f"[CACHE] Initiated caching for authenticated employee streaming result: {query[:50]}...")
                                    
                                except Exception as cache_error:
                                    logger.error(f"[CACHE] Error initiating cache storage: {str(cache_error)}\n{traceback.format_exc()}")


                    elif kind == "on_chain_start":
                        # Node start events
                        yield {
                            "event": "node_start",
                            "data": {
                                "node": event["name"],
                                "employee_id": str(employee_id)
                            }
                        }
                    
                    elif kind == "on_chain_error":
                        # Error handling
                        error_data = event.get("data", {})
                        error_message = str(error_data.get("error", "Unknown error"))
                        logger.error(f"Chain error in node {event['name']}: {error_message}")
                        
                        yield {
                            "event": "error",
                            "data": {
                                "error": error_message,
                                "node": event["name"]
                            },
                            "metadata": {
                                "employee_id": str(employee_id),
                                "timestamp": datetime.utcnow().isoformat()
                            }
                        }
            except Exception as stream_error:
                # Handle internal streaming errors
                error_msg = f"Internal streaming error: {stream_error}"
                logger.exception(error_msg)
                yield {
                    "event": "error",
                    "data": {"error": error_msg},
                    "metadata": {"timestamp": datetime.utcnow().isoformat()}
                }
                
        except Exception as e:
            logger.exception(f"Critical error in workflow streaming: {str(e)}")
            yield {
                "event": "error",
                "data": {"error": f"Workflow error: {str(e)}"}
            }


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
    session_id = f"employee_{employee_id}_{datetime.utcnow().timestamp()}"
    logger.info(f"Created employee workflow session: {session_id}")
    return session_id