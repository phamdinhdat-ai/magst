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


class DrugAgent(Agent):
    """
    Agent chuyên xử lý các câu hỏi về thuốc, sử dụng các công cụ truy xuất thông tin tĩnh.
    Đây là một ví dụ về một agent "đơn giản" không cần logic động phức tạp.
    """
    def __init__(self, llm: BaseChatModel, default_tool_names: List[str] = None, **kwargs):
        """
        Khởi tạo DrugAgent.
        Nó kế thừa toàn bộ logic từ lớp cha và chỉ cần được cấu hình đúng cách.

        Args:
            llm (BaseChatModel): Language model sẽ được sử dụng.
            default_tool_names (List[str], optional): Tên các tool mặc định luôn chạy.
        """
        agent_name = "DrugAgent"
        system_prompt = agent_config['drug_agent']['description']
        
        # Gọi __init__ của lớp cha.
        # Agent này không cần biết về cách tool được tạo ra.
        super().__init__(
            llm=llm, 
            agent_name=agent_name, 
            system_prompt=system_prompt, 
            default_tool_names=default_tool_names or [],
            **kwargs
        )
        logger.info(f"'{self.agent_name}' initialized. It will request tools from the factory.")
    
    # KHÔNG CẦN CÁC PHƯƠƠNG THỨC GHI ĐÈ.
    # Lớp `Agent` cơ sở đã xử lý tất cả.


if __name__ == "__main__":
    async def main():
        # --- Setup ---
        llm = llm_instance
        
        # Khởi tạo agent.
        # Chúng ta quyết định rằng đối với DrugAgent, 'drug_retriever_tool'
        # là một tool mặc định, luôn cần chạy để có context tốt nhất.
        drug_agent = DrugAgent(
            llm=llm, 
            default_tool_names=["drug_retriever_tool"]
        )
        
        # Tạo state mẫu
        state = AgentState(
            original_query="Cơ chế hoạt động của Aspirin là gì?",
            rewritten_query="Aspirin mechanism of action",
            intents=["retrieve"], # Giả lập đầu ra từ EntryAgent
            user_role="doctor",
            chat_history=[]
        )

        # --- Execution ---
        print(f"--- Executing {drug_agent.agent_name} ---")
        final_state = None
        # Logic trong base class sẽ:
        # 1. Thấy 'drug_retriever_tool' trong default_tool_names.
        # 2. Thấy intent 'retrieve' và cũng có thể lấy 'drug_retriever_tool' từ map.
        # 3. Yêu cầu factory cung cấp tool này (chỉ 1 lần do dùng set).
        # 4. Chạy tool, xây dựng prompt, và stream câu trả lời.
        async for partial_state in drug_agent.astream_execute(state):
            print(f"Streaming response: ...{partial_state.get('agent_response', '')[-30:]}", end="\r")
            final_state = partial_state
        
        print("\n\n--- Final Result ---")
        if final_state.get('error_message'):
            print(f"Error: {final_state['error_message']}")
        else:
            print(f"Final Answer: {final_state.get('agent_response')}")
        print(f"\nContexts from tools: {list(final_state.get('contexts', {}).keys())}")

    # Chạy kịch bản test
    asyncio.run(main())