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
from app.agents.factory.tools.base import BaseAgentTool
from app.agents.factory.factory_tools import TOOL_FACTORY  # Import factory tools

class ProductAgent(Agent):
    """
    Agent chuyên xử lý các câu hỏi về sản phẩm và dịch vụ của công ty.
    """
    def __init__(self, llm: BaseChatModel, default_tool_names: List[str] = None, **kwargs):
        """
        Khởi tạo ProductAgent.

        Args:
            llm (BaseChatModel): Language model sẽ được sử dụng.
            default_tool_names (List[str], optional): Tên các tool mặc định luôn chạy.
        """
        agent_name = "ProductAgent"
        system_prompt = agent_config['product_agent']['description']
        super().__init__(
            llm=llm, 
            agent_name=agent_name, 
            system_prompt=system_prompt, 
            default_tool_names=default_tool_names or [],
            **kwargs
        )
        logger.info(f"'{self.agent_name}' initialized. It will request tools from the factory.")
    
    # KHÔNG CẦN CÁC PHƯƠNG THỨC GHI ĐÈ.
    # Lớp `Agent` cơ sở đã xử lý tất cả.

if __name__ == "__main__":
    async def main():
        # --- Setup ---
        llm = llm_instance
        
        # Khởi tạo agent.
        # Đặt 'product_retriever_tool' làm tool mặc định để agent này
        # luôn có context về sản phẩm khi trả lời.
        product_agent = ProductAgent(
            llm=llm, 
            default_tool_names=["product_retriever_tool"]
        )
        
        # Tạo state mẫu
        state = AgentState(
            original_query="Các gói xét nghiệm di truyền của GeneStory có gì khác nhau?",
            rewritten_query="compare GeneStory genetic testing packages",
            intents=["retrieve", "product"], # Giả lập đầu ra từ EntryAgent
            user_role="guest",
            chat_history=[]
        )

        # --- Execution ---
        print(f"--- Executing {product_agent.agent_name} ---")
        final_state = None
        # Lớp cha sẽ tự động lấy 'product_retriever_tool' từ factory
        # và chạy nó để cung cấp context cho LLM.
        async for partial_state in product_agent.astream_execute(state):
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