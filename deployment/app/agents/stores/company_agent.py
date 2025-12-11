import asyncio
import sys
from typing import List
from pathlib import Path

from loguru import logger

# --- LangChain Core & Community Imports ---
from langchain_core.language_models.chat_models import BaseChatModel

# --- Local/App Imports ---
# Đảm bảo các import này trỏ đến các file đã được tối ưu
sys.path.append(str(Path(__file__).parent.parent.parent))
from app.agents.stores.base_agent import Agent, AgentState  # Sử dụng AgentState đã định nghĩa
from app.agents.workflow.initalize import llm_instance, agent_config  # Import phiên bản
from app.agents.factory.tools.base import BaseAgentTool



class CompanyAgent(Agent):
    """
    Agent chuyên xử lý các câu hỏi về thông tin công ty.
    Nó kế thừa toàn bộ logic từ lớp Agent cơ sở và chỉ cần được cấu hình đúng cách.
    """
    def __init__(self, llm: BaseChatModel, default_tool_names: List[str] = None, **kwargs):
        """
        Khởi tạo CompanyAgent.

        Args:
            llm (BaseChatModel): Language model sẽ được sử dụng.
            default_tool_names (List[str], optional): Danh sách tên các tool MẶC ĐỊNH
                luôn chạy cho agent này, bất kể intent là gì.
        """
        # Lấy system prompt từ file cấu hình
        agent_name = "CompanyAgent"
        system_prompt = agent_config["company_agent"]["description"]
        
        # Gọi __init__ của lớp cha (Agent) với các thông tin đã được cấu hình.
        # Toàn bộ logic phức tạp đã nằm trong lớp cha.
        super().__init__(
            llm=llm, 
            agent_name=agent_name, 
            system_prompt=system_prompt, 
            default_tool_names=default_tool_names or [],
            **kwargs
        )
        logger.info(f"'{self.agent_name}' initialized successfully.")
    
    # KHÔNG CẦN GHI ĐÈ BẤT KỲ PHƯƠNG THỨC NÀO.
    # Lớp `Agent` cơ sở đã xử lý tất cả:
    # - _get_tools_to_run: Sẽ tự động lấy tool mặc định và tool theo intent từ factory.
    # - astream_execute: Sẽ chạy tool, xây dựng prompt và gọi LLM.


# ==============================================================================
# === TEST EXECUTION
# ==============================================================================
if __name__ == '__main__':
    async def main():
        # --- Setup ---
        # 1. Lấy một instance của LLM
        llm = llm_instance

        # 2. Tạo instance của CompanyAgent
        # Sử dụng `default_tool_names` để đảm bảo CompanyRetrieverTool luôn được chạy
        # khi agent này được gọi, cung cấp context nền tảng quan trọng.
        company_agent = CompanyAgent(
            llm=llm,
            default_tool_names=["company_retriever_tool"]
        )

        # 3. Tạo một trạng thái (state) giả để thực thi
        # Quan trọng: Cung cấp `intents` mà EntryAgent sẽ tạo ra.
        test_state = AgentState(
            original_query="Địa chỉ và thông tin liên hệ của công ty là gì?",
            rewritten_query="Địa chỉ và thông tin liên hệ của công ty là gì?",
            intents=["retrieve"], # Giả lập đầu ra từ EntryAgent
            chat_history=[]
        )

        # --- Execution (Sử dụng luồng async) ---
        print(f"--- Executing {company_agent.agent_name} ---")
        # Gọi `aexecute` vì đây là phương thức chính, bất đồng bộ.
        # `execute` chỉ là trình bao (wrapper) cho mục đích test đơn giản.
        final_state = await company_agent.aexecute(test_state)

        # --- Output ---
        print("\n--- Final State ---")
        if final_state.get('error_message'):
            print(f"Error: {final_state['error_message']}")
        else:
            print(f"Agent Response: {final_state.get('agent_response')}")
        
        if final_state.get('contexts'):
            print("\n--- Contexts Used by Agent ---")
            for tool_name, context in final_state['contexts'].items():
                print(f"[{tool_name}]:\n{context[:300]}...\n")

    # Chạy kịch bản test bất đồng bộ
    asyncio.run(main())