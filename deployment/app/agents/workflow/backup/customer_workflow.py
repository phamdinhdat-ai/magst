import asyncio
import sys
from typing import Dict, Any, Optional, AsyncIterator
from loguru import logger
from pathlib import Path

# --- LangGraph Imports ---
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver

# --- Import base components ---
# Đảm bảo sys.path chỉ được thêm một lần và đúng cách
current_dir = Path(__file__).parent
sys.path.append(str(current_dir.parent))

from app.agents.workflow.state import GraphState as AgentState  # Sử dụng AgentState đã định nghĩa
from app.agents.workflow.initalize import llm_instance, agent_config
# Import factory, nó sẽ quản lý việc tạo tool
from app.agents.factory.factory_tools import TOOL_FACTORY, ToolFactory
# --- Import all agent CLASSES (không phải instance) ---


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
    Quản lý và thực thi luồng công việc multi-agent cho chatbot khách hàng.
    Kiến trúc này sử dụng ToolFactory để quản lý tool và các agent được tối ưu
    để hoạt động bất đồng bộ.
    """
    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        # --- 1. Khởi tạo tất cả các agent ---
        # Agent được khởi tạo một lần và tái sử dụng.
        self.agents = self._initialize_agents()
        
        # --- 2. Xây dựng và biên dịch graph ---
        self.graph = self._build_and_compile_graph()
        
        logger.info("Customer Workflow initialized successfully with an async-native graph.")

    def _initialize_agents(self) -> Dict[str, Any]:
        """Khởi tạo tất cả các instance agent và lưu vào một dictionary."""
        logger.info("Initializing all agent instances...")
        llm = llm_instance # Lấy llm một lần

        # Các agent node xử lý (không cần tool)
        entry_agent = EntryAgent(llm=llm)
        rewriter_agent = RewriterAgent(llm=llm)
        reflection_agent = ReflectionAgent(llm=llm, default_tool_names=["summary_tool"])
        supervisor_agent = SupervisorAgent(llm=llm)
        question_generator = QuestionGeneratorAgent(llm=llm)

        # Các agent chuyên môn (sẽ lấy tool từ factory)
        # Chúng ta chỉ cần truyền cấu hình `default_tool_names` nếu cần
        company_agent = CompanyAgent(llm=llm, default_tool_names=["company_retriever_tool"])
        customer_agent = CustomerAgent(llm=llm) # Tool động sẽ được lấy tự động
        product_agent = ProductAgent(llm=llm, default_tool_names=["product_retriever_tool"])
        medical_agent = MedicalAgent(llm=llm, default_tool_names=["medical_retriever_tool"])
        drug_agent = DrugAgent(llm=llm, default_tool_names=["drug_retriever_tool"])
        genetic_agent = GeneticAgent(llm=llm, default_tool_names=["genetic_retriever_tool"])
        visual_agent = VisualAgent(llm=llm, default_tool_names=["image_analyzer"])
        naive_agent = NaiveAgent(llm=llm, default_tool_names=["searchweb_tool"])

        return {
            # Map tên node trong graph với instance agent tương ứng
            "entry": entry_agent,
            "rewriter": rewriter_agent,
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
        }

    async def _run_agent(self, state: AgentState) -> AgentState:
        """
        Một node thực thi chung. Nó sẽ xem `classified_agent` trong state
        và gọi đến agent tương ứng đã được khởi tạo.
        """
        agent_name = state.get("classified_agent")
        if not agent_name or agent_name not in self.agents:
            logger.error(f"Invalid or missing agent name in state: '{agent_name}'. Routing to error handler.")
            state['error_message'] = f"Router specified an invalid agent: {agent_name}"
            return state
            
        agent_to_run = self.agents[agent_name]
        logger.info(f"--- Running Specialist Agent: {agent_name} ---")
        
        # Chạy agent và cập nhật state
        result_state = await agent_to_run.aexecute(state)
        
        # Ghi lại "suy nghĩ" của agent vào state
        agent_thinks = state.get("agent_thinks", {})
        agent_thinks[agent_name] = result_state.get("agent_response")
        result_state["agent_thinks"] = agent_thinks

        return result_state

    def _build_and_compile_graph(self) -> StateGraph:
        """Xây dựng và biên dịch graph langgraph."""
        workflow = StateGraph(AgentState)
        
        # --- Định nghĩa các Node ---
        # Các node này là các hàm hoặc phương thức bất đồng bộ
        workflow.add_node("entry", self.agents["entry"].aexecute)
        workflow.add_node("rewriter", self.agents["rewriter"].aexecute)
        # Node chung để chạy các agent chuyên môn
        workflow.add_node("specialist_agent", self._run_agent)
        workflow.add_node("reflection", self.agents["reflection"].aexecute)
        workflow.add_node("supervisor", self.agents["supervisor"].aexecute)
        workflow.add_node("question_generator", self.agents["question_generator"].aexecute)
        
        # --- Định nghĩa các cạnh (Edges) ---
        workflow.set_entry_point("entry")
        
        workflow.add_conditional_edges("entry", self._route_after_entry)
        
        # Sau khi viết lại, quay lại node entry để phân loại lại
        workflow.add_edge("rewriter", "entry")
        
        # Sau khi agent chuyên môn chạy xong, đi đến bước phản ánh
        workflow.add_edge("specialist_agent", "reflection")
        
        workflow.add_conditional_edges("reflection", self._route_after_reflection)
        
        # Sau khi supervisor hoàn thành, có thể đi đến bước tạo câu hỏi gợi ý
        workflow.add_edge("supervisor", "question_generator")
        
        # Node cuối cùng
        workflow.add_edge("question_generator", END)

        # Biên dịch graph
        memory = InMemorySaver()
        return workflow.compile(checkpointer=memory)

    # --- Routing Logic ---
    def _route_after_entry(self, state: AgentState) -> str:
        """Quyết định nhánh đi tiếp theo sau EntryAgent."""
        logger.info("--- ROUTING AFTER ENTRY ---")
        if state.get("needs_rewrite", False):
            logger.info("Decision: -> rewriter")
            return "rewriter"
        
        agent_name = state.get("classified_agent")
        if agent_name in self.agents:
            logger.info(f"Decision: -> specialist_agent (to run {agent_name})")
            return "specialist_agent"
        
        logger.warning(f"Unknown agent '{agent_name}', defaulting to specialist_agent with NaiveAgent.")
        state["classified_agent"] = "NaiveAgent" # Fallback an toàn
        return "specialist_agent"

    def _route_after_reflection(self, state: AgentState) -> str:
        """Quyết định có cần chạy lại, chạy tiếp, hay kết thúc."""
        logger.info("--- ROUTING AFTER REFLECTION ---")
        iteration_count = state.get("iteration_count", 0) + 1
        state["iteration_count"] = iteration_count

        if state.get("error_message"):
            logger.error(f"Error detected in reflection. Terminating. Error: {state['error_message']}")
            return END
        
        if iteration_count >= self.max_iterations:
            logger.warning(f"Max iterations ({self.max_iterations}) reached. Moving to supervisor.")
            return "supervisor"
            
        if state.get("is_final_answer", False):
            logger.info("Decision: Answer is final -> supervisor")
            return "supervisor"
        
        followup_agent = state.get("suggest_agent_followups")
        if followup_agent and followup_agent in self.agents:
            logger.info(f"Decision: Reflection suggests followup -> specialist_agent (to run {followup_agent})")
            state["classified_agent"] = followup_agent
            return "specialist_agent"

        logger.info("Decision: No clear next step -> supervisor")
        return "supervisor"

    # --- Public Execution Method ---
    async def arun(self, query: str, config: Dict) -> "AsyncIterator[AgentState]":
        """Chạy workflow và stream các state cập nhật."""
        initial_state = AgentState(
            original_query=query,
            iteration_count=0,
            chat_history=[],
        )
        # Sử dụng astream để có thể theo dõi các bước
        async for output in self.graph.astream(initial_state, config=config):
            # `output` sẽ là một dict, với key là tên node và value là state sau khi node đó chạy
            for key, value in output.items():
                logger.info(f"--- Node '{key}' finished ---")
                yield value # Trả về state cập nhật sau mỗi bước

# ==============================================================================
# === TEST EXECUTION
# ==============================================================================
if __name__ == "__main__":
    async def main():
        logger.remove()
        logger.add(sys.stdout, level="INFO")
        logger.info("====== INITIALIZING OPTIMIZED CUSTOMER WORKFLOW ======")
        
        workflow_manager = CustomerWorkflow()
        
        session_id = "test_session_123"
        config = {"configurable": {"thread_id": session_id}}
        
        queries = [
            "Công ty GeneStory làm về lĩnh vực gì?",
            "Tôi là khách hàng có mã số CUST-007, xem giúp tôi báo cáo gần nhất.",
            "Ảnh này là biểu đồ gì vậy?", # Cần có state['image_path']
            "Thuốc paracetamol có tác dụng phụ gì không?"
        ]
        
        for query in queries:
            logger.info("-" * 80)
            logger.info(f"🚀 EXECUTING QUERY: '{query}'")
            final_state = None
            async for state in workflow_manager.arun(query, config):
                final_state = state

            print("\n" + "="*20 + " FINAL RESULT " + "="*20)
            print(f"Query: {query}")
            print(f"Final Answer: {final_state.get('agent_response', 'N/A')}")
            print(f"Suggested Questions: {final_state.get('suggested_questions', 'N/A')}")
            if final_state.get('error_message'):
                print(f"Error: {final_state.get('error_message')}")
            print("=" * 54 + "\n")
            
    try:
        asyncio.run(main())
    finally:
        # Quan trọng: Dọn dẹp các tool singleton khi ứng dụng kết thúc
        TOOL_FACTORY.cleanup_singletons()