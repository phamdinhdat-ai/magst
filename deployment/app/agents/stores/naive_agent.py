import sys
import asyncio
from typing import List

from loguru import logger
from pathlib import Path

# --- LangChain Core & Community Imports ---
from langchain_core.language_models.chat_models import BaseChatModel

# --- Local/App Imports ---
sys.path.append(str(Path(__file__).parent.parent.parent))
from app.agents.stores.base_agent import Agent, AgentState  # Sử dụng AgentState đã định nghĩa
from app.agents.workflow.initalize import llm_instance, agent_config  # Import phiên bản
from app.agents.factory.tools.base import BaseAgentTool
from app.agents.factory.factory_tools import TOOL_FACTORY  # Import factory tools
# --- Prerequisites ---

class NaiveAgent(Agent):
    """
    Agent chuyên xử lý các câu hỏi từ người dùng vãng lai (khách).
    Thường xử lý các câu hỏi chung, tư vấn về sản phẩm, và tìm kiếm web.
    """
    def __init__(self, llm: BaseChatModel, default_tool_names: List[str] = None, **kwargs):
        """
        Khởi tạo NaiveAgent.

        Args:
            llm (BaseChatModel): Language model sẽ được sử dụng.
            default_tool_names (List[str], optional): Tên các tool mặc định luôn chạy.
        """
        agent_name = "NaiveAgent"
        system_prompt = agent_config['naive_agent']['description']
        # Gọi __init__ của lớp cha.
        # Toàn bộ logic phức tạp đã nằm ở lớp cha và ToolFactory.
        super().__init__(
            llm=llm, 
            agent_name=agent_name, 
            system_prompt=system_prompt, 
            default_tool_names=default_tool_names or [],
            **kwargs
        )
        logger.info(f"'{self.agent_name}' initialized. It will request tools from the factory.")
  
if __name__ == "__main__":  
        
    async def main():
        # --- Setup ---
        llm = llm_instance
        
        # Khởi tạo agent.
        # Với NaiveAgent, chúng ta có thể quyết định rằng không có tool mặc định nào.
        # Tool sẽ được chọn hoàn toàn dựa trên `intent` từ EntryAgent.
        naive_agent = NaiveAgent(llm=llm)
        
        # --- Test Case 1: Intent là 'searchweb' ---
        print("--- Test Case 1: Intent is 'searchweb' ---")
        state_search = AgentState(
            original_query="Giá cổ phiếu của Apple hôm nay là bao nhiêu?",
            intents=["searchweb"], # Giả lập đầu ra từ EntryAgent
            user_role="guest",
            chat_history=[]
        )

        final_state = None
        # Logic trong base class sẽ:
        # 1. Thấy intent 'searchweb'.
        # 2. Tra trong INTENT_TO_TOOL_MAP và tìm thấy 'searchweb_tool'.
        # 3. Yêu cầu factory cung cấp 'searchweb_tool'.
        # 4. Chạy tool và trả về kết quả.
        async for partial_state in naive_agent.astream_execute(state_search):
            print(f"Streaming response: ...{partial_state.get('agent_response', '')[-30:]}", end="\r")
            final_state = partial_state
        
        print("\n\n--- Final Result for Test Case 1 ---")
        if final_state.get('error_message'):
            print(f"Error: {final_state['error_message']}")
        else:
            print(f"Final Answer: {final_state.get('agent_response')}")
        print(f"\nContexts from tools: {list(final_state.get('contexts', {}).keys())}")

        # --- Test Case 2: Intent là 'chitchat' ---
        print("\n\n--- Test Case 2: Intent is 'chitchat' ---")
        state_chitchat = AgentState(
            original_query="Chào bạn, bạn có khỏe không?",
            intents=["chitchat"],
            user_role="guest",
            chat_history=[]
        )

        final_state_2 = None
        # Logic trong base class sẽ không tìm thấy tool nào cho intent 'chitchat'
        # và sẽ trả lời trực tiếp mà không dùng tool.
        async for partial_state_2 in naive_agent.astream_execute(state_chitchat):
            print(f"Streaming response: ...{partial_state_2.get('agent_response', '')[-30:]}", end="\r")
            final_state_2 = partial_state_2
            
        print("\n\n--- Final Result for Test Case 2 ---")
        print(f"Final Answer: {final_state_2.get('agent_response')}")
        print(f"\nContexts from tools: {list(final_state_2.get('contexts', {}).keys())}") # Sẽ là danh sách rỗng

    # Chạy kịch bản test
    asyncio.run(main())