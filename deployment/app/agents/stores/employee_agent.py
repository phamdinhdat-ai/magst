import sys
import asyncio
from typing import List

from loguru import logger
from pathlib import Path

# --- LangChain Core & Community Imports ---
from langchain_core.language_models.chat_models import BaseChatModel

# --- Local/App Imports ---
sys.path.append(str(Path(__file__).parent.parent.parent))
# Sửa các import này cho đúng với cấu trúc dự án của bạn
from app.agents.stores.base_agent import Agent, AgentState  # Sử dụng AgentState đã định nghĩa
from app.agents.workflow.initalize import llm_instance, agent_config  # Import phiên bản
# --- Prerequisites ---
# Để code này hoạt động, bạn cần đảm bảo:
# 1. Trong `tool_factory_optimized.py`:
#    - Đã có logic `get_dynamic_tool` cho "employee_retriever_tool".
# 2. Trong `base_agent_optimized.py` (hoặc config chung):
#    - DYNAMIC_TOOLS đã được định nghĩa:
#      DYNAMIC_TOOLS = {"EmployeeAgent": "employee_retriever_tool", ...}

class EmployeeAgent(Agent):
    """
    Agent chuyên xử lý các câu hỏi liên quan đến một nhân viên cụ thể.
    Nó kế thừa toàn bộ logic từ lớp cha và không cần logic ghi đè phức tạp.
    """
    def __init__(self, llm: BaseChatModel, default_tool_names: List[str] = None, **kwargs):
        """
        Khởi tạo EmployeeAgent.

        Args:
            llm (BaseChatModel): Language model sẽ được sử dụng.
            default_tool_names (List[str], optional): Tên các tool mặc định luôn chạy.
        """
        agent_name = "EmployeeAgent"
        system_prompt = agent_config['employee_agent']['description']
        
        # Gọi __init__ của lớp cha.
        super().__init__(
            llm=llm,
            agent_name=agent_name,
            system_prompt=system_prompt,
            default_tool_names=default_tool_names or [],
            **kwargs
        )
        logger.info(f"'{self.agent_name}' initialized. It will request tools dynamically from the factory.")

    # KHÔNG CẦN GHI ĐÈ _get_tools_to_run.
    # Logic trong lớp `Agent` cha sẽ tự động:
    # 1. Thấy `self.agent_name` là "EmployeeAgent".
    # 2. Nếu `intent` là "retrieve", nó sẽ tra trong `DYNAMIC_TOOLS`.
    # 3. Yêu cầu "employee_retriever_tool" từ factory, truyền `state` vào để factory
    #    có thể lấy `employee_id`.

if __name__ == "__main__":
    async def main():
        # --- Setup ---
        llm = llm_instance
        
        # Khởi tạo agent rất đơn giản.
        # Chúng ta có thể quyết định rằng tool retriever là một tool mặc định
        # để nó luôn chạy khi agent này được gọi, cung cấp context nhân viên.
        employee_agent = EmployeeAgent(
            llm=llm,
            default_tool_names=["employee_retriever_tool"]
        )
        
        # --- Test Case 1: Có employee_id và intent phù hợp ---
        print("--- Test Case 1: Retrieve employee info ---")
        state_with_id = AgentState(
            original_query="Xem lại mục tiêu hiệu suất quý trước của tôi.",
            employee_id="12345",
            intents=["retrieve"], # Kích hoạt việc tìm kiếm tool
            user_role="employee",
            chat_history=[]
        )

        final_state = None
        async for partial_state in employee_agent.astream_execute(state_with_id):
            print(f"Streaming response: ...{partial_state.get('agent_response', '')[-30:]}", end="\r")
            final_state = partial_state
        
        print("\n\n--- Final Result for Test Case 1 ---")
        if final_state.get('error_message'):
            print(f"Error: {final_state['error_message']}")
        else:
            print(f"Final Answer: {final_state.get('agent_response')}")
        print(f"Tools Selected & Run (from context): {list(final_state.get('contexts', {}).keys())}")
        
        # --- Test Case 2: Intent không liên quan đến retrieve ---
        print("\n--- Test Case 2: Intent is 'searchweb' ---")
        state_search = AgentState(
            original_query="Tìm kiếm thông tin về các khóa học Python nâng cao.",
            employee_id="12345", # Vẫn có ID nhân viên
            intents=["searchweb"], # Nhưng intent không phải là retrieve
            user_role="employee",
            chat_history=[]
        )
        
        # Logic trong base class sẽ:
        # 1. Thấy 'employee_retriever_tool' trong default_tool_names -> CHỌN
        # 2. Thấy intent 'searchweb' -> CHỌN 'searchweb_tool' từ map.
        # 3. Chạy cả hai tool.
        async for final_state_2 in employee_agent.astream_execute(state_search):
            pass # Lấy state cuối
            
        print("\n--- Final Result for Test Case 2 ---")
        print(f"Tools Selected & Run (from context): {list(final_state_2.get('contexts', {}).keys())}")

    # Chạy kịch bản test
    asyncio.run(main())