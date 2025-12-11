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

# --- Các import khác giữ nguyên ---
# ... (imports from a_customer_workflow_optimized.py)
from app.agents.workflow.state import GraphState as AgentState  # Sử dụng AgentState đã định nghĩa
from app.agents.workflow.initalize import llm_instance, agent_config  # Import phiên bản
from app.agents.factory.factory_tools import TOOL_FACTORY  # Import factory tools   
from app.agents.stores.entry_agent import EntryAgent
from app.agents.stores.company_agent import CompanyAgent
from app.agents.stores.customer_agent import CustomerAgent
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
from app.agents.stores.cache_manager import CacheManager

class CustomerWorkflow:
    """
    Workflow được tối ưu cho streaming và thực thi song song các tác vụ cuối.
    """
    def __init__(self, max_iterations: int = 3):
        self.max_iterations = max_iterations
        self.agents = self._initialize_agents()
        self.graph = self._build_and_compile_graph()
        
        # Initialize cache manager for workflow caching
        self.cache_manager = CacheManager()
        
        logger.add(Path("app/logs/log_workflows/customer_workflow.log"), rotation="10 MB", level="DEBUG", backtrace=True, diagnose=True)
        logger.info("Customer Workflow (Streaming & Concurrent) initialized with caching support.")

    def _initialize_agents(self) -> Dict[str, Any]:
        """Khởi tạo tất cả các instance agent (giữ nguyên logic)."""
        logger.info("Initializing all agent instances...")
        llm = llm_instance

        # Các agent node xử lý
        entry_agent = EntryAgent(llm=llm)
        rewriter_agent = RewriterAgent(llm=llm)
        reflection_agent = ReflectionAgent(llm=llm, default_tool_names=["summary_tool"])
        supervisor_agent = SupervisorAgent(llm=llm) # Supervisor không cần tool
        question_generator = QuestionGeneratorAgent(llm=llm)

        # Các agent chuyên môn
        company_agent = CompanyAgent(llm=llm, default_tool_names=["company_retriever_tool"])
        customer_agent = CustomerAgent(llm=llm)
        product_agent = ProductAgent(llm=llm, default_tool_names=["product_retriever_tool"])
        medical_agent = MedicalAgent(llm=llm, default_tool_names=["medical_retriever_tool"])
        drug_agent = DrugAgent(llm=llm, default_tool_names=["drug_retriever_tool"])
        genetic_agent = GeneticAgent(llm=llm, default_tool_names=["genetic_retriever_tool"])
        visual_agent = VisualAgent(llm=llm, default_tool_names=["image_analyzer"])
        naive_agent = NaiveAgent(llm=llm, default_tool_names=[""])

        return {
            "entry": entry_agent, "rewriter": rewriter_agent,
            "reflection": reflection_agent, "supervisor": supervisor_agent,
            "question_generator": question_generator, "CompanyAgent": company_agent,
            "CustomerAgent": customer_agent, "ProductAgent": product_agent,
            "MedicalAgent": medical_agent, "DrugAgent": drug_agent,
            "GeneticAgent": genetic_agent, "VisualAgent": visual_agent, "NaiveAgent": naive_agent,
        }
        # ---

    async def _run_agent(self, state: AgentState) -> AgentState:
        """Node thực thi chung cho các agent chuyên môn với customer context."""
        agent_name = state.get("classified_agent")
        if not agent_name or agent_name not in self.agents:
            state["error_message"] = f"Agent {agent_name} not found"
            state["classified_agent"] = "NaiveAgent"  # Fallback to naive agent
            agent_name = "NaiveAgent"
        
        agent_to_run = self.agents[agent_name]
        logger.info(f"--- Running Specialist Agent: {agent_name} for Customer ---")
        
        # Add customer-specific context to state
        customer_id = state.get("customer_id")
        customer_role = state.get("customer_role", "customer")
        
        # Enhance state with customer information
        enhanced_state = state.copy()
        enhanced_state["user_context"] = {
            "customer_id": customer_id,
            "customer_role": customer_role,
            "is_authenticated": True,
            "access_level": self._get_access_level(customer_role)
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
        
        try:
            # Execute the agent and ensure we don't carry any DB connections between agents
            logger.info(f"Executing agent {agent_name} with enhanced state: {enhanced_state}")
            result_state = await agent_to_run.aexecute(enhanced_state)
            logger.info("Agent execution completed successfully.")
            # Always explicitly clean up connections after agent execution
            from app.db.session import close_db_connections
            await close_db_connections()
            
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
                    logger.warning("Failed to clean up DB connections after agent execution.")
        except Exception as e:
            # Ensure we clean up even on errors
            logger.error(f"Error executing agent {agent_name}: {e}")
            try:
                from app.db.session import close_db_connections
                await close_db_connections()
            except:
                logger.warning("Failed to clean up DB connections after agent execution.")
                pass
            raise logger.error(f"Failed to execute agent {agent_name}: {e}")
    
        # Preserve customer context
        preserved_keys = [
            'original_query', 'rewritten_query', 'chat_history',
            'session_id', 'user_role', 'iteration_count', 'agent_thinks',
            'customer_id', 'customer_role', 'interaction_id'
        ]
        for key in preserved_keys:
            if key in state and key not in result_state:
                result_state[key] = state[key]
        logger.info(f"Agent {agent_name} preserved state: {result_state}")
        agent_thinks = state.get("agent_thinks", {})
        agent_thinks[agent_name] = result_state.get("agent_response")
        result_state["agent_thinks"] = agent_thinks
        logger.info(f"Agent {agent_name} completed with response: {result_state.get('agent_response', '')[:100]}...")
        
        # Clean up any database connections that might have been left open
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
        Node xử lý cuối cùng, chạy QuestionGenerator.
        Node này chạy SAU KHI supervisor đã stream xong.
        """
        logger.info("--- Running Final Processing (Question Generation) ---")
        question_generator = self.agents["question_generator"]
        # Chạy question generator và cập nhật state
        final_state = await question_generator.aexecute(state)
        return final_state

    def _build_and_compile_graph(self) -> AgentState:
        """Xây dựng và biên dịch graph, hỗ trợ streaming."""
        workflow = StateGraph(AgentState)
        
        # --- Định nghĩa các Node ---
        workflow.add_node("entry", self.agents["entry"].aexecute)
        workflow.add_node("rewriter", self.agents["rewriter"].aexecute)
        workflow.add_node("specialist_agent", self._run_agent)
        workflow.add_node("reflection", self.agents["reflection"].aexecute)
        # **THAY ĐỔI QUAN TRỌNG**: Node supervisor giờ trỏ đến `.astream_execute`
        workflow.add_node("supervisor", self.agents["supervisor"].astream_execute)
        # Node mới để xử lý sau khi stream
        workflow.add_node("final_processing", self._final_processing_node)
        
        # --- Định nghĩa các cạnh (Edges) ---
        workflow.set_entry_point("entry")
        workflow.add_conditional_edges("entry", self._route_after_entry)
        workflow.add_edge("rewriter", "entry")
        workflow.add_edge("specialist_agent", "reflection")
        workflow.add_conditional_edges("reflection", self._route_after_reflection)
        
        # **THAY ĐỔI QUAN TRỌNG**: Sau khi supervisor stream xong, đi đến final_processing
        workflow.add_edge("supervisor", "final_processing")
        workflow.add_edge("final_processing", END)

        memory = InMemorySaver()
        return workflow.compile(checkpointer=memory)
    # --- Routing Logic (giữ nguyên) ---
    def _route_after_entry(self, state: AgentState) -> str:
        logger.info("--- ROUTING AFTER ENTRY ---")
        if state.get("needs_rewrite", False): return "rewriter"
        agent_name = state.get("classified_agent")
        if agent_name in self.agents: return "specialist_agent"
        state["classified_agent"] = "NaiveAgent"
        return "specialist_agent"


    def _route_after_reflection(self, state: AgentState) -> str:
        logger.info("--- ROUTING AFTER REFLECTION ---")
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
        # ---
    
    # --- **PHƯƠNG THỨC THỰC THI CÔNG KHAI MỚI** ---
    async def arun_streaming(self, query: str, config: Dict, customer_id: str = None, user_role: str = "customer", chat_history: list = None) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Run workflow and stream events to the client.
        Includes answer chunks and final information.
        Implements intelligent caching for improved response times.
        """
        if chat_history is None:
            chat_history = []
            
        logger.info(f"Starting workflow execution for {'customer' if customer_id else 'guest'} query: {query[:100]}...")
        start_time = time.time()
        
        try:
            # Check cache first for faster responses
            if self.cache_manager.is_active():
                # For private customer conversations, include customer_id in cache key
                cache_context = {"customer_id": customer_id} if customer_id and user_role == "customer" else {}
                cache_key = self.cache_manager.create_cache_key(
                    query, 
                    chat_history,
                    context=cache_context
                )
                
                logger.debug(f"Checking cache with key: {cache_key}")
                cached_result = await self.cache_manager.get(cache_key)
                
                if cached_result:
                    logger.info(f"Cache HIT for streaming customer query: {query[:50]}...")
                    # Simulate streaming events from cached result
                    if not customer_id:
                        customer_id = f"customer_{uuid.uuid4().hex[:8]}"
                    
                    interaction_id = str(uuid.uuid4())
                    
                    # Yield answer chunks if available
                    agent_response = cached_result.get("agent_response", "")
                    if agent_response:
                        # Split response into chunks for streaming effect
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
                                # Small delay to simulate streaming
                                await asyncio.sleep(0.01)
                            except Exception as chunk_error:
                                logger.error(f"Error yielding chunk: {chunk_error}")
                                continue
                    
                    # Yield final result
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
            
            # If cache miss, proceed with normal execution
            initial_state = AgentState(
                original_query=query, 
                iteration_count=0, 
                chat_history=chat_history, 
                customer_id=customer_id, 
                user_role=user_role,
                timestamp=datetime.utcnow().isoformat()
            )
            
            logger.info("Starting workflow execution...")
            
            # Use astream_events for detailed event information
            try:
                async for event in self.graph.astream_events(initial_state, config=config, version="v1"):
                    try:
                        kind = event.get("event")
                        node_name = event.get("name", "unknown")
                        
                        if kind == "on_chain_stream":
                            # Event when a node is streaming (SupervisorAgent)
                            chunk = event.get("data", {}).get("chunk", {})
                            if isinstance(chunk, dict) and "agent_response" in chunk:
                                try:
                                    yield {
                                        "event": "answer_chunk",
                                        "data": chunk.get("agent_response", ""),
                                        "metadata": {
                                            "node": node_name,
                                            "timestamp": datetime.utcnow().isoformat()
                                        }
                                    }
                                except Exception as chunk_error:
                                    logger.error(f"Error processing answer chunk: {chunk_error}")
                                    continue
                        
                        elif kind == "on_chain_end":
                            # Event when a node completes
                            if node_name == "final_processing":
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
                                        "metadata": {
                                            "cache_hit": False,
                                            "processing_mode": "streaming_workflow",
                                            "user_role": user_role,
                                            "is_private": bool(customer_id and user_role == "customer"),
                                            "processing_time": time.time() - start_time,
                                            "timestamp": datetime.utcnow().isoformat()
                                        }
                                    }
                                    
                                    # Cache the successful result for future requests
                                    if self.cache_manager.is_active():
                                        try:
                                            cache_context = {"customer_id": customer_id} if customer_id and user_role == "customer" else {}
                                            cache_key = self.cache_manager.create_cache_key(
                                                query, 
                                                chat_history,
                                                context=cache_context
                                            )
                                            
                                            # Create a cacheable version without customer-specific data
                                            cacheable_result = final_result_data.copy()
                                            
                                            # Store in cache asynchronously to not block the response
                                            asyncio.create_task(self.cache_manager.set(
                                                cache_key, 
                                                cacheable_result, 
                                                ttl=1800  # 30 minutes TTL
                                            ))
                                            logger.info(f"Cached streaming result for {'private ' if cache_context else ''}customer query: {query[:50]}...")
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
                            # Node start events
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
                logger.error(f"Error in workflow streaming: {stream_error}")
                logger.error(traceback.format_exc())
                yield {
                    "event": "error",
                    "data": {
                        "error": "Error in workflow execution",
                        "details": str(stream_error)
                    },
                    "metadata": {
                        "timestamp": datetime.utcnow().isoformat()
                    }
                }
                
        except Exception as e:
            logger.error(f"Critical error in workflow: {e}")
            logger.error(traceback.format_exc())
            yield {
                "event": "error",
                "data": {
                    "error": "Critical error in workflow execution",
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
        customer_id: int,  # Explicitly typed as int
        customer_role: str = "customer",
        interaction_id: Optional[uuid.UUID] = None,
        chat_history: Optional[list] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Run workflow for authenticated customer and stream events.
        
        Args:
            query: Customer query
            config: LangGraph configuration
            customer_id: Authenticated customer ID (integer)
            customer_role: Customer role (customer, premium_customer, vip_customer, admin)
            interaction_id: Optional interaction ID for tracking
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
            start_time = time.time()
            # Check cache first for faster responses
            if self.cache_manager.is_active():
                try:
                    # Create customer-specific cache key by modifying the query
                    customer_query = f"[CUSTOMER:{customer_id}:{customer_role}] {query}"
                    cache_key = self.cache_manager.create_cache_key(
                        customer_query, 
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
                                agent_response = agent_response.split()
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
                                    # Small delay to simulate streaming
                                    await asyncio.sleep(0.01)
                            
                            # Yield final result
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
                                "processing_mode": "authenticated_streaming_workflow"
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
                            logger.info(f"[CACHE] Successfully streamed cached result for authenticated employee query")
                            return
                            
                        except Exception as cache_stream_error:
                            logger.error(f"[CACHE] Error streaming cached result: {str(cache_stream_error)}\n{traceback.format_exc()}")
                            # Continue with normal execution if cache streaming fails
                    else:
                        logger.debug(f"[CACHE] Cache MISS for authenticated employee streaming query: {query[:50]}...")
                        
                except Exception as cache_error:
                    logger.error(f"[CACHE] Error checking cache for authenticated streaming: {str(cache_error)}\n{traceback.format_exc()}")
            # Log detailed information for debugging
            logger.info(f"Starting authenticated workflow stream for customer {customer_id}")
            logger.debug(f"Query: '{query}', Config: {config}, Role: {customer_role}")
            
            # Setup initial state with defensive conversions
            initial_state = AgentState(
                original_query=query,
                iteration_count=0,
                chat_history=chat_history if chat_history else [],
                user_role="customer",
                customer_id=str(customer_id) if customer_id is not None else "",  # Safe conversion
                customer_role=customer_role or "customer",  # Default if None
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
                    kind = event["event"]
                    
                    if kind == "on_chain_stream":
                        # Stream chunks from SupervisorAgent
                        chunk = event["data"]["chunk"]
                        if isinstance(chunk, dict) and "agent_response" in chunk:
                            yield {
                                "event": "answer_chunk",
                                "data": chunk.get("agent_response", ""),
                                "metadata": {
                                    "customer_id": str(customer_id),
                                    "customer_role": customer_role,
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
                                        "full_final_answer": final_state.get("agent_response", ""),
                                        "agent_response": final_state.get("agent_response", ""),
                                        "status": "success",
                                        "metadata": {
                                            "cache_hit": False,
                                            "processing_mode": "streaming_workflow",
                                            "user_role": 'customer',
                                            "is_private": bool(customer_id),
                                            "processing_time": time.time() - start_time,
                                            "timestamp": datetime.utcnow().isoformat()
                                        }
                                    }
                            logger.info(f"Final Output from {node_name}: {final_state}")
                            yield {
                                "event": "final_result",
                                "data": {
                                    "suggested_questions": final_state.get("suggested_questions", []),
                                    "full_final_answer": final_state.get("agent_response", ""),
                                    "agents_used": list(final_state.get("agent_thinks", {}).keys()),
                                    "interaction_id": str(interaction_id) if interaction_id else None,
                                    "processing_time": self._calculate_processing_time(initial_state)
                                },
                                "metadata": {
                                    "customer_id": str(customer_id),
                                    "customer_role": customer_role,
                                    "timestamp": datetime.utcnow().isoformat()
                                }
                            }# Cache the successful result for future requests
                            if self.cache_manager.is_active():
                                try:
                                    cache_context = {"customer_id": customer_id} if customer_id is not None else {}
                                    cache_key = self.cache_manager.create_cache_key(
                                        query, 
                                        chat_history,
                                    )
                                    
                                    # Create a cacheable version without customer-specific data
                                    cacheable_result = final_result_data.copy()
                                    cacheable_result["metadata"].pop("customer_id", None)
                                    # Store in cache asynchronously to not block the response
                                    asyncio.create_task(self.cache_manager.set(
                                        cache_key, 
                                        cacheable_result, 
                                        ttl=1800  # 30 minutes TTL
                                    ))
                                    logger.info(f"Cached streaming result for {'private ' if cache_context else ''}customer query: {query[:50]}...")
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
                        # logger.info(f"Node {node_name} completed successfully.")
                        # logger.debug(f"Final state: {final_state}")
                    elif kind == "on_chain_start":
                        # Node start events
                        yield {
                            "event": "node_start",
                            "data": {
                                "node": event["name"],
                                "customer_id": str(customer_id)
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
                                "customer_id": str(customer_id),
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
        Chạy workflow và trả về kết quả cuối cùng (không streaming).
        Phương thức này đã được xác thực nên luôn có customer_id.
        Includes intelligent caching to improve response times.
        """
        # Initialize config if not provided
        if not config:
            config = {"configurable": {"thread_id": f"customer_{customer_id}_{int(datetime.utcnow().timestamp())}"}}
        
        # Check cache first for faster responses
        if self.cache_manager.is_active():
            # Always include customer_id in cache key for authenticated endpoints
            cache_context = {"customer_id": customer_id, "user_role": user_role}
            cache_key = self.cache_manager.create_cache_key(
                query, 
                chat_history or [],
                context=cache_context
            )
            cached_result = await self.cache_manager.get(cache_key)
            
            if cached_result:
                logger.info(f"Cache HIT for authenticated customer query: {query[:50]}...")
                # Update metadata with cache info
                if "metadata" in cached_result:
                    cached_result["metadata"]["cache_hit"] = True
                    cached_result["metadata"]["processing_time"] = 0.1
                return cached_result
        
        # If cache miss, proceed with normal execution
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
                
                # Cache the successful result for future requests
                if self.cache_manager.is_active() and final_result.get("status") != "error":
                    # Prepare cacheable result (exclude any sensitive data)
                    cacheable_result = final_result.copy()
                    
                    # Store in cache asynchronously to not block the response
                    asyncio.create_task(self.cache_manager.set(
                        cache_key, 
                        cacheable_result, 
                        ttl=1800  # 30 minutes TTL
                    ))
                    logger.info(f"Cached authenticated customer query result: {query[:50]}...")
                
                break
        
        return final_result

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
        Process a document following the RAG workflow with caching support:
        1. Check cache for existing results
        2. If cache miss, process document:
           - Document uploaded and stored in folder
           - Document preprocessed and chunked
           - Chunks stored in vector database
           - Content retrieved to answer queries
        
        Args:
            query: User query about document content
            document_id: ID of the document to process
            user_type: Type of user (customer, employee, guest)
            user_id: ID of the user
            session_id: Optional session ID
            chat_history: Optional chat history for context
            
        Returns:
            Document processing result with answers and insights
        """
        if not session_id:
            session_id = f"doc_session_{document_id}_{datetime.utcnow().timestamp()}"
        
        # Configure document processing parameters
        config = {
            "configurable": {
                "thread_id": session_id,
                "document_id": document_id,
                "retrieval_mode": "document_specific"  # Focus retrieval on this document
            }
        }
        
        # Create initial state with document context using available AgentState fields
        initial_state = AgentState(
            original_query=query,
            # Map user type to appropriate state field
            customer_id=str(user_id) if user_type == "customer" else None,
            employee_id=int(user_id) if user_type == "employee" else None,
            guest_id=str(user_id) if user_type == "guest" else None,
            user_role=user_type,
            session_id=session_id,
            chat_history=[],
            # Default values for required fields
            rewritten_query="",
            intents="",
            agent_response="",
            agent_thinks={},
            reflection_feedback="",
            is_final_answer=False,
            needs_rewrite=False,
            retry_count=0,
            suggested_questions=[],
            task_assigned=[]
        )
        
        logger.info(f"Starting document processing for {user_type} {user_id}, document {document_id}")
        
        # Context for document processing
        document_context = {
            "document_id": document_id,
            "context_type": "document_processing"
        }
        
        # Add document context to state
        initial_state["contexts"] = [document_context]
        
        # Check cache first for document queries
        if query and self.cache_manager.is_active():
            # Create a cache key that includes document_id and user context
            cache_context = {
                "document_id": document_id,
                "user_type": user_type,
                "user_id": str(user_id)
            }
            cache_key = self.cache_manager.create_cache_key(
                f"doc_query:{query}",
                chat_history or [],
            )
            
            # Try to get from cache
            cached_result = await self.cache_manager.get(cache_key)
            if cached_result:
                logger.info(f"Cache HIT for document query (doc_id={document_id}): {query[:50]}...")
                if "metadata" in cached_result:
                    cached_result["metadata"]["cache_hit"] = True
                    cached_result["metadata"]["processing_time"] = 0.1
                return cached_result
        
        # Process the document if not in cache
        final_result = {}
        full_answer = ""
        
        async for event in self.graph.astream_events(initial_state, config=config, version="v1"):
            kind = event["event"]
            
            if kind == "on_chain_stream":
                # Stream chunks from processing
                chunk = event["data"]["chunk"]
                if isinstance(chunk, dict) and "agent_response" in chunk:
                    current_chunk = chunk.get("agent_response", "")
                    full_answer += current_chunk
            
            elif kind == "on_chain_end":
                node_name = event["name"]
                if node_name == "final_processing":
                    # Final result with document insights
                    final_state = event["data"]["output"]
                    
                    # Create response with document insights
                    final_result = {
                        "document_id": document_id,
                        "user_type": user_type,
                        "user_id": str(user_id),
                        "processed_content": full_answer,
                        "suggested_questions": final_state.get("suggested_questions", []),
                        "agents_used": list(final_state.get("agent_thinks", {}).keys()),
                        "processing_time": self._calculate_processing_time(initial_state) if "timestamp" in initial_state else 0.0,
                        "metadata": {
                            "cache_hit": False,
                            "document_processed": True,
                            "document_id": document_id
                        }
                    }
                    
                    # Cache the successful result if this was a query
                    if query and self.cache_manager.is_active() and final_result.get("status") != "error":
                        # Prepare cacheable result (exclude any sensitive data)
                        cacheable_result = final_result.copy()
                        
                        # Store in cache asynchronously to not block the response
                        asyncio.create_task(self.cache_manager.set(
                            cache_key, 
                            cacheable_result, 
                            ttl=3600  # 1 hour TTL for document queries
                        ))
                        logger.info(f"Cached document query result (doc_id={document_id}): {query[:50]}...")
                    
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
        Process a customer query with load balancing support.
        
        This method will:
        1. Check if the request should be handled locally or remotely
        2. If remote, forward to another node via load balancer
        3. If local, either process directly or queue for processing
        
        Args:
            query: Customer query text
            customer_id: Customer ID
            customer_role: Customer role (e.g., "customer", "premium_customer")
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
                session_id = f"customer_{customer_id}_{datetime.utcnow().timestamp()}"
                
            # Check if we should process this request on another node
            if load_balancer and load_balancer.is_active():
                # Get system load
                local_load = await load_balancer.get_local_load()
                
                # If local load is high, try to find another node
                if local_load.get('cpu_percent', 0) > 70 or local_load.get('memory_percent', 0) > 80:
                    logger.info(f"Local system load is high: {local_load}. Trying to find another node.")
                    
                    # Get best node for this request type
                    request_type = "premium_query" if customer_role in ("premium_customer", "vip_customer") else "standard_query"
                    best_node = await load_balancer.get_best_node(request_type=request_type)
                    
                    if best_node and best_node['id'] != load_balancer.get_node_id():
                        logger.info(f"Forwarding request to node: {best_node['id']}")
                        
                        # Forward request to the selected node
                        response = await load_balancer.forward_request(
                            node=best_node,
                            endpoint="/api/v1/customer/chat",
                            method="POST",
                            payload={
                                "query": query,
                                "customer_id": customer_id,
                                "customer_role": customer_role,
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
                
                # Determine priority based on customer role and prioritize flag
                priority = TaskPriority.HIGH if (
                    prioritize or 
                    customer_role in ("premium_customer", "vip_customer")
                ) else TaskPriority.NORMAL
                
                logger.info(f"Queueing customer query with priority {priority}")
                
                # Add task to queue
                task_id = await queue_manager.add_task(
                    task_func="app.tasks.workflow_tasks.process_query_task",
                    args={
                        "query": query,
                        "customer_id": customer_id,
                        "customer_role": customer_role,
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
            return await self.arun_simple_authenticated(query, customer_id, customer_role, session_id)
            
        except Exception as e:
            logger.exception(f"Error in load-balanced processing: {e}")
            return {
                "status": "error",
                "message": "An error occurred while processing your request.",
                "error": str(e)
            }
        

# Customer-specific utility functions
async def create_customer_workflow_session(
    customer_id: int,  # Changed from uuid.UUID to int
    customer_role: str = "customer"
) -> str:
    """Create a new workflow session for customer."""
    session_id = f"customer_{customer_id}_{datetime.utcnow().timestamp()}"
    logger.info(f"Created customer workflow session: {session_id}")
    return session_id

def validate_customer_access(customer_role: str, requested_feature: str) -> bool:
    """Validate if customer role has access to requested feature."""
    access_matrix = {
        "customer": ["basic_search", "company_info", "product_info"],
        "premium_customer": ["basic_search", "company_info", "product_info", "advanced_search", "priority_support"],
        "vip_customer": ["basic_search", "company_info", "product_info", "advanced_search", "priority_support", "personal_consultant"],
        "admin": ["all_features"]
    }
    
    allowed_features = access_matrix.get(customer_role, [])
    return requested_feature in allowed_features or "all_features" in allowed_features


# ==============================================================================
# === TEST EXECUTION
# ==============================================================================
if __name__ == "__main__":
    async def main():
        logger.remove()
        logger.add(sys.stdout, level="INFO")
        logger.info("====== INITIALIZING STREAMING/CONCURRENT WORKFLOW ======")
        
        workflow_manager = CustomerWorkflow()
        
        session_id = f"test_session_{asyncio.Task.current_task().get_name()}"
        config = {"configurable": {"thread_id": session_id}}
        
        query = "So sánh Aspirin và Paracetamol, và cho biết gen BRCA1 có liên quan không."
        
        logger.info("-" * 80)
        logger.info(f"🚀 EXECUTING QUERY: '{query}'")
        
        full_answer = ""
        final_data = {}

        # Mô phỏng client nhận các sự kiện
        async for event in workflow_manager.arun_streaming(query, config, customer_id="789122254025", user_role="customer"):
            if event["event"] == "answer_chunk":
                chunk_data = event["data"]
                # Giả lập việc hiển thị chunk cho người dùng
                # In ra phần mới của câu trả lời
                new_part = chunk_data.replace(full_answer, "", 1)
                print(new_part, end="", flush=True)
                full_answer = chunk_data
            
            elif event["event"] == "final_result":
                final_data = event["data"]

        print("\n\n" + "="*20 + " FINAL RESULT " + "="*20)
        print(f"Query: {query}")
        print(f"Full Final Answer (reconstructed): {full_answer}")
        print(f"Suggested Questions: {final_data.get('suggested_questions', 'N/A')}")
        print("=" * 54 + "\n")

    try:
        asyncio.run(main())
    finally:
        TOOL_FACTORY.cleanup_singletons()