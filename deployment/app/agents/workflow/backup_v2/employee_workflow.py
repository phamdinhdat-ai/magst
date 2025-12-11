import asyncio
import sys
from typing import Dict, Any, Optional, AsyncGenerator

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
from app.agents.workflow.initalize import llm_instance, agent_config  # Import phiên bản
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
class EmployeeWorkflow:
    """
    Workflow dành riêng cho nhân viên, được tách biệt hoàn toàn
    khỏi dữ liệu và các agent của khách hàng để đảm bảo bảo mật.
    """
    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.agents = self._initialize_agents()
        self.graph = self._build_and_compile_graph()
        logger.info("Secure Employee Workflow initialized.")

    def _initialize_agents(self) -> Dict[str, Any]:
        """
        Khởi tạo các agent dành riêng cho nhân viên.
        *** KHÔNG BAO GỒM CustomerAgent. ***
        """
        logger.info("Initializing agents for SECURE Employee Workflow...")
        llm = llm_instance

        return {
            # Các node điều khiển chung
            "entry": EntryAgent(llm=llm),
            "rewriter": RewriterAgent(llm=llm),
            "reflection": ReflectionAgent(llm=llm, default_tool_names=["summary_tool"]),
            "supervisor": SupervisorAgent(llm=llm),
            "question_generator": QuestionGeneratorAgent(llm=llm),
            
            # Các agent chuyên môn được phép cho nhân viên
            "EmployeeAgent": EmployeeAgent(llm=llm), # Agent chính
            "CompanyAgent": CompanyAgent(llm=llm, default_tool_names=["company_retriever_tool"]),
            "ProductAgent": ProductAgent(llm=llm, default_tool_names=["product_retriever_tool"]),
            "MedicalAgent": MedicalAgent(llm=llm, default_tool_names=["medical_retriever_tool"]),
            "DrugAgent": DrugAgent(llm=llm, default_tool_names=["drug_retriever_tool"]),
            "GeneticAgent": GeneticAgent(llm=llm, default_tool_names=["genetic_retriever_tool"]),
            "VisualAgent": VisualAgent(llm=llm, default_tool_names=["image_analyzer"]),
            "NaiveAgent": NaiveAgent(llm=llm, default_tool_names=["searchweb_tool"]),

            # **CustomerAgent đã được loại bỏ khỏi danh sách này**
        }

    # ==============================================================================
    # === CÁC PHƯƠNG THỨC CÒN LẠI ĐƯỢC TÁI SỬ DỤNG HOÀN TOÀN ===
    # === KHÔNG CẦN THAY ĐỔI GÌ Ở _run_agent, _build_and_compile_graph, ROUTING ===
    # ==============================================================================

    async def _run_agent(self, state: AgentState) -> AgentState:
        """Node thực thi chung (Tái sử dụng 100%)."""
        agent_name = state.get("classified_agent")
        # Logic này tự động an toàn: nếu EntryAgent có lỡ phân loại nhầm thành
        # 'CustomerAgent', nó sẽ không tìm thấy trong self.agents và báo lỗi.
        if not agent_name or agent_name not in self.agents:
            state['error_message'] = f"Access Denied or Invalid Agent: The requested agent '{agent_name}' is not available in this workflow."
            return state
            
        agent_to_run = self.agents[agent_name]
        logger.info(f"--- Running Specialist Agent: {agent_name} ---")
        
        result_state = await agent_to_run.aexecute(state)
        
        preserved_keys = [
            'original_query', 'rewritten_query', 'chat_history', 
            'employee_id', 'session_id', 'user_role', # Chỉ có employee_id
            'iteration_count', 'agent_thinks'
        ]
        for key in preserved_keys:
            if key in state:
                result_state[key] = state[key]

        agent_thinks = result_state.get("agent_thinks", {})
        agent_thinks[agent_name] = result_state.get("agent_response")
        result_state["agent_thinks"] = agent_thinks

        return result_state

    def _build_and_compile_graph(self) -> AgentState:
        """Xây dựng và biên dịch graph (Tái sử dụng 100%)."""
        workflow = StateGraph(AgentState)
        workflow.add_node("entry", self.agents["entry"].aexecute)
        workflow.add_node("rewriter", self.agents["rewriter"].aexecute)
        workflow.add_node("specialist_agent", self._run_agent)
        workflow.add_node("reflection", self.agents["reflection"].aexecute)
        workflow.add_node("supervisor", self.agents["supervisor"].astream_execute) 
        workflow.add_node("question_generator", self.agents["question_generator"].aexecute)
        workflow.set_entry_point("entry")
        workflow.add_conditional_edges("entry", self._route_after_entry)
        workflow.add_edge("rewriter", "entry")
        workflow.add_edge("specialist_agent", "reflection")
        workflow.add_conditional_edges("reflection", self._route_after_reflection_with_loop)
        workflow.add_edge("supervisor", "question_generator")
        workflow.add_edge("question_generator", END)
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
    
    # --- Public Execution Method ---
    async def arun_streaming(self, query: str, config: Dict, employee_id: str) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Chạy workflow cho nhân viên.
        *** KHÔNG CÒN tham số `other_context` để truyền customer_id. ***
        """
        initial_state = AgentState(
            original_query=query,
            iteration_count=0,
            chat_history=[],
            employee_id=employee_id,
            user_role="employee",
            session_id=config.get("configurable", {}).get("thread_id")
        )
        
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



if __name__ == "__main__":
    async def main():
        logger.remove(); logger.add(sys.stdout, level="INFO")
        logger.info("====== INITIALIZING SECURE EMPLOYEE WORKFLOW ======")
        
        workflow_manager = EmployeeWorkflow()
        
        session_id = "test_secure_employee_789"
        config = {"configurable": {"thread_id": session_id}}
        
        # Kịch bản 1: Nhân viên hỏi về chính sách
        query1 = "Chính sách làm việc từ xa của công ty như thế nào?"
        employee_id1 = "EMP-001"
        
        logger.info("-" * 80); logger.info(f"🚀 EXECUTING QUERY FOR EMPLOYEE '{employee_id1}': '{query1}'")
        async for event in workflow_manager.arun_streaming(query1, config, employee_id=employee_id1):
             # Logic xử lý event để hiển thị...
            if event["event"] == "answer_chunk":
                chunk_data = event["data"]
                # Giả lập việc hiển thị chunk cho người dùng
                # In ra phần mới của câu trả lời
                new_part = chunk_data.replace(full_answer, "", 1)
                print(new_part, end="", flush=True)
                full_answer = chunk_data
            
            elif event["event"] == "final_result":
                final_data = event["data"]

        # Kịch bản 2: Nhân viên cố gắng hỏi về khách hàng (Sẽ thất bại một cách an toàn)
        query2 = "Thông tin của khách hàng CUST-007 là gì?"
        employee_id2 = "EMP-002"
        
        logger.info("\n" + "-" * 80); logger.info(f"🚀 ATTEMPTING TO ACCESS CUSTOMER DATA: '{query2}'")
        async for event in workflow_manager.arun_streaming(query2, config, employee_id=employee_id2):
            # ... xử lý event ...
            # Trong trường hợp này, EntryAgent có thể sẽ phân loại là "CustomerAgent".
            # Khi đó, node `_run_agent` sẽ không tìm thấy "CustomerAgent" trong `self.agents`
            # và sẽ trả về lỗi "Access Denied or Invalid Agent".
            if event.get("event") == "on_chain_end" and event.get("name") == "_run_agent":
                output_state = event.get("data", {}).get("output")
                if output_state and output_state.get("error_message"):
                    logger.error(f"Workflow stopped as expected: {output_state.get('error_message')}")

    # Chạy hàm main để thực thi các kịch bản
    asyncio.run(main()) 