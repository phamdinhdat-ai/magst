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
from app.agents.retrievers.genetic_retriever import GeneticRetrieverTool
from app.agents.factory.factory_tools import TOOL_FACTORY
# --- Prerequisites ---
# Đảm bảo các cấu hình trong factory và base_agent đã được thực hiện

class GeneticAgent(Agent):
    """
    Agent chuyên xử lý các câu hỏi về di truyền, sử dụng một retriever singleton mạnh mẽ.
    """
    def __init__(self, llm: BaseChatModel, default_tool_names: List[str] = None, **kwargs):
        agent_name = "GeneticAgent"
        system_prompt = agent_config['genetic_agent']['description']
        super().__init__(
            llm=llm, 
            agent_name=agent_name, 
            system_prompt=system_prompt, 
            default_tool_names=default_tool_names or [],
            **kwargs
        )
        logger.info(f"'{self.agent_name}' initialized. It will request tools from the factory.")
    
    # KHÔNG CẦN BẤT KỲ LOGIC GHI ĐÈ NÀO

if __name__ == "__main__":
    async def main():
        # --- Setup ---
        llm = llm_instance
        
        # Khởi tạo agent. Đặt 'genetic_retriever_tool' làm tool mặc định.
        # Khi agent được gọi, nó sẽ yêu cầu tool này từ factory.
        # Factory sẽ tạo ra (nếu chưa có) và trả về instance singleton.
        genetic_agent = GeneticAgent(
            llm=llm, 
            default_tool_names=["genetic_retriever_tool"]
        )
        
        # Tạo state mẫu
        state = AgentState(
            original_query="Đột biến gen BRCA1 là gì?",
            intents=["retrieve", "genetic"],
            user_role="doctor",
            chat_history=[]
        )

        # --- Execution ---
        print(f"--- Executing {genetic_agent.agent_name} ---")
        final_state = await genetic_agent.aexecute(state)
        
        print("\n\n--- Final Result ---")
        print(f"Final Answer: {final_state.get('agent_response')}")
        print(f"\nContexts from tools: {list(final_state.get('contexts', {}).keys())}")

    try:
        asyncio.run(main())
    finally:
        # Quan trọng: Dọn dẹp tài nguyên khi test kết thúc
        TOOL_FACTORY.cleanup_singletons()