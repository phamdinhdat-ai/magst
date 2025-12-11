import asyncio
import sys
from typing import Dict, Any, Optional, AsyncGenerator
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

class CustomerWorkflow:
    """
    Workflow được tối ưu cho streaming và thực thi song song các tác vụ cuối.
    """
    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.agents = self._initialize_agents()
        self.graph = self._build_and_compile_graph()
        logger.info("Customer Workflow (Streaming & Concurrent) initialized.")

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
        naive_agent = NaiveAgent(llm=llm, default_tool_names=["searchweb_tool"])

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
            state['error_message'] = f"Router specified an invalid agent: {agent_name}"
            return state
        
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
        
        result_state = await agent_to_run.aexecute(enhanced_state)
        
        # Preserve customer context
        preserved_keys = [
            'original_query', 'rewritten_query', 'chat_history',
            'session_id', 'user_role', 'iteration_count', 'agent_thinks',
            'customer_id', 'customer_role', 'interaction_id'
        ]
        for key in preserved_keys:
            if key in state:
                result_state[key] = state[key]
        
        agent_thinks = state.get("agent_thinks", {})
        agent_thinks[agent_name] = result_state.get("agent_response")
        result_state["agent_thinks"] = agent_thinks
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
        return workflow.compile(checkpointer=memory)

    # --- Routing Logic (giữ nguyên) ---
    def _route_after_entry(self, state: AgentState) -> str:
        # ... (logic từ phiên bản trước không đổi)
        # ---
        logger.info("--- ROUTING AFTER ENTRY ---")
        if state.get("needs_rewrite", False): return "rewriter"
        agent_name = state.get("classified_agent")
        if agent_name in self.agents: return "specialist_agent"
        state["classified_agent"] = "NaiveAgent"
        return "specialist_agent"
        # ---

    def _route_after_reflection(self, state: AgentState) -> str:
        # ... (logic từ phiênbản trước không đổi)
        # ---
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
    async def arun_streaming(self, query: str, config: Dict, customer_id: str = None, user_role: str="customer", chat_history: list = []) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Chạy workflow và stream các sự kiện về cho client.
        Bao gồm các chunk của câu trả lời và các thông tin cuối cùng.
        """
        initial_state = AgentState(original_query=query, iteration_count=0, chat_history=[], customer_id=customer_id, user_role=user_role)
        
        # Sử dụng astream_events để có thông tin chi tiết về các sự kiện
        async for event in self.graph.astream_events(initial_state, config=config, version="v1"):
            kind = event["event"]
            
            if kind == "on_chain_stream":
                # Sự kiện này xảy ra khi một node đang stream (chính là SupervisorAgent)
                chunk = event["data"]["chunk"]
                if isinstance(chunk, dict) and "agent_response" in chunk:
                    # Yield một sự kiện "answer_chunk"
                    yield {
                        "event": "answer_chunk",
                        "data": chunk.get("agent_response", "")
                    }
            
            elif kind == "on_chain_end":
                # Sự kiện này xảy ra khi một node kết thúc
                node_name = event["name"]
                if node_name == "final_processing":
                    # Khi node cuối cùng kết thúc, chúng ta có các câu hỏi gợi ý
                    final_state = event["data"]["output"]
                    yield {
                        "event": "final_result",
                        "data": {
                            "suggested_questions": final_state.get("suggested_questions", []),
                            "full_final_answer": final_state.get("agent_response", ""),
                            # Thêm các thông tin debug khác nếu muốn
                            # "agent_thinks": final_state.get("agent_thinks") 
                        }
                    }
            
            elif kind == "on_chain_start":
                # Có thể yield các sự kiện về việc node nào đang bắt đầu chạy
                 yield {
                     "event": "node_start",
                     "data": {"node": event["name"]}
                 }

    async def arun_streaming_authenticated(
        self, 
        query: str, 
        config: Dict, 
        customer_id: int,  # Explicitly typed as int
        customer_role: str = "customer",
        interaction_id: Optional[uuid.UUID] = None
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
        initial_state = AgentState(
            original_query=query,
            iteration_count=0,
            chat_history=[],
            user_role="customer",
            customer_id=str(customer_id),  # Convert to string since AgentState expects string
            customer_role=customer_role,
            interaction_id=str(interaction_id) if interaction_id else None,
            session_id=config.get("configurable", {}).get("thread_id"),
            timestamp=datetime.utcnow().isoformat()
        )
        
        logger.info(f"Starting authenticated customer workflow for customer {customer_id} with role {customer_role}")
        
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
                    }
            
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
                yield {
                    "event": "error",
                    "data": {
                        "error": str(event.get("data", {}).get("error", "Unknown error")),
                        "node": event["name"]
                    },
                    "metadata": {
                        "customer_id": str(customer_id),
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

    async def arun_simple_authenticated(
        self, 
        query: str, 
        customer_id: int,
        customer_role: str = "customer",
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run workflow for authenticated customer and return final result.
        
        Args:
            query: Customer query
            customer_id: Authenticated customer ID
            customer_role: Customer role
            session_id: Optional session ID
            
        Returns:
            Final workflow result
        """
        if not session_id:
            session_id = f"customer_{customer_id}_{datetime.utcnow().timestamp()}"
        
        config = {"configurable": {"thread_id": session_id}}
        
        final_result = {}
        full_answer = ""
        
        async for event in self.arun_streaming_authenticated(query, config, customer_id, customer_role):
            if event.get("event") == "answer_chunk":
                full_answer += event["data"]
            elif event.get("event") == "final_result":
                final_result = event["data"]
                final_result["full_answer"] = full_answer
                break
        
        return final_result

    async def arun_document_processing(
        self,
        query: str,
        document_id: int,
        user_type: str,
        user_id: uuid.UUID,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a document following the RAG workflow:
        1. Document uploaded and stored in folder
        2. Document preprocessed and chunked
        3. Chunks stored in vector database
        4. Content retrieved to answer queries
        
        Args:
            query: User query about document content
            document_id: ID of the document to process
            user_type: Type of user (customer, employee, guest)
            user_id: ID of the user
            session_id: Optional session ID
            
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
        
        # Process the document
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
                        "processing_time": self._calculate_processing_time(initial_state) if "timestamp" in initial_state else 0.0
                    }
                    break
        
        return final_result
        

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