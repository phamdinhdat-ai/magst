import asyncio
import sys
from typing import Dict, Any, Optional, AsyncGenerator

from loguru import logger
from pathlib import Path

# --- LangGraph Imports ---
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver

# --- Import base components ---
sys.path.append(str(Path(__file__).parent.parent)) # Sử dụng AgentState đã định nghĩa
from app.agents.workflow.initalize import llm_instance, agent_config  # Import phiên bản
from app.agents.factory.factory_tools import TOOL_FACTORY  # Import factory tools
from app.agents.workflow.state import GraphState as AgentState  # Sử dụng AgentState đã định nghĩa
from app.agents.workflow.initalize import llm_instance, agent_config  # Import phiên bản
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
from app.agents.stores.question_generator_agent import QuestionGeneratorAgent

class GuestWorkflow:
    """
    Workflow được thiết kế cho người dùng vãng lai (khách).
    Tập trung vào việc cung cấp thông tin chung và giới thiệu.
    Không truy cập vào dữ liệu của khách hàng hoặc nhân viên.
    """
    def __init__(self, max_iterations: int = 4): # Có thể giảm số lần lặp cho khách
        self.max_iterations = max_iterations
        self.agents = self._initialize_agents()
        self.graph = self._build_and_compile_graph()
        logger.info("Guest Workflow initialized.")

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
            "rewriter": RewriterAgent(llm=llm),
            "reflection": ReflectionAgent(llm=llm, default_tool_names=["summary_tool"]),
            "supervisor": SupervisorAgent(llm=llm),
            "question_generator": QuestionGeneratorAgent(llm=llm),
            
            # Các agent chuyên môn được phép cho khách
            "CompanyAgent": CompanyAgent(llm=llm, default_tool_names=["company_retriever_tool"]),
            "ProductAgent": ProductAgent(llm=llm, default_tool_names=["product_retriever_tool"]),
            "MedicalAgent": MedicalAgent(llm=llm, default_tool_names=["medical_retriever_tool"]),
            "DrugAgent": DrugAgent(llm=llm, default_tool_names=["drug_retriever_tool"]),
            "GeneticAgent": GeneticAgent(llm=llm, default_tool_names=["genetic_retriever_tool"]),
            "NaiveAgent": NaiveAgent(llm=llm, default_tool_names=[]),

            # ** CustomerAgent và EmployeeAgent đã được loại bỏ **
        }

    # ==============================================================================
    # === CÁC PHƯƠNG THỨC CÒN LẠI ĐƯỢC TÁI SỬ DỤNG HOÀN TOÀN ===
    # === KHÔNG CẦN THAY ĐỔI GÌ Ở _run_agent, _build_and_compile_graph, ROUTING ===
    # ==============================================================================

    async def _run_agent(self, state: AgentState) -> AgentState:
        """Node thực thi chung (Tái sử dụng 100%)."""
        agent_name = state.get("classified_agent")
        if not agent_name or agent_name not in self.agents:
            state['error_message'] = f"Access Denied or Invalid Agent: The requested agent '{agent_name}' is not available in this workflow."
            return state
        agent_to_run = self.agents[agent_name]
        logger.info(f"--- Running Specialist Agent: {agent_name} ---")
        result_state = await agent_to_run.aexecute(state)
        preserved_keys = [
            'original_query', 'rewritten_query', 'chat_history',
            'session_id', 'user_role', 'iteration_count', 'agent_thinks'
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
    async def arun_streaming(self, query: str, config: Dict) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Chạy workflow cho khách và stream các sự kiện.
        *** KHÔNG nhận customer_id hay employee_id. ***
        """
        initial_state = AgentState(
            original_query=query,
            iteration_count=0,
            chat_history=[],
            user_role="guest", # Vai trò là khách
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

# ==============================================================================
# === TEST EXECUTION
# ==============================================================================
if __name__ == "__main__":
    async def main():
        logger.remove(); logger.add(sys.stdout, level="INFO")
        logger.info("====== INITIALIZING GUEST WORKFLOW FOR TESTING ======")
        
        workflow_manager = GuestWorkflow()
        
        session_id = "test_guest_session_101"
        config = {"configurable": {"thread_id": session_id}}
        
        # --- Kịch bản 1: Khách hỏi về thông tin công ty ---
        query1 = "GeneStory có trụ sở ở đâu?"
        
        logger.info("-" * 80); logger.info(f"🚀 EXECUTING GUEST QUERY: '{query1}'")
        async for event in workflow_manager.arun_streaming(query1, config):
            # Logic xử lý event để hiển thị...
             if event.get("event") == "on_chain_start":
                print(f"\n[Workflow] -> Running node: {event['name']}")
             elif event.get("event") == "on_chain_stream":
                chunk = event.get("data", {}).get("chunk", {})
                if isinstance(chunk, AgentState):
                    print(chunk.get("agent_response", ""), end="", flush=True)

        # --- Kịch bản 2: Khách hỏi một câu hỏi kiến thức y khoa chung ---
        query2 = "Bệnh tiểu đường là gì?"
        
        logger.info("\n" + "-" * 80); logger.info(f"🚀 EXECUTING GUEST QUERY: '{query2}'")
        async for event in workflow_manager.arun_streaming(query2, config):
            if event["event"] == "answer_chunk":
                chunk_data = event["data"]
                # Giả lập việc hiển thị chunk cho người dùng
                # In ra phần mới của câu trả lời
                new_part = chunk_data.replace(full_answer, "", 1)
                print(new_part, end="", flush=True)
                full_answer = chunk_data
            
            elif event["event"] == "final_result":
                final_data = event["data"]

    asyncio.run(main())