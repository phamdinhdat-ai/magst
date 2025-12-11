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
# --- Prerequisites ---
# Để code này hoạt động, bạn cần đảm bảo:
# 1. Trong `tool_factory_optimized.py`:
#    - Đã import `MedicalRetrieverTool`.
#    - Đã đăng ký blueprint cho nó:
#      'medical_retriever_tool': lambda: MedicalRetrieverTool(collection_name="medical_docs", ...),
#
# 2. Trong `base_agent_optimized.py` (hoặc config chung):
#    - Đã cập nhật INTENT_TO_TOOL_MAP để bao gồm tool này nếu cần, ví dụ:
#      "retrieve": ["medical_retriever_tool", ...],
#
# 3. Trong file định nghĩa `MedicalRetrieverTool`:
#    - Đã thêm thuộc tính `intents`:
#      intents: Set[str] = {"retrieve", "medical"}

class MedicalAgent(Agent):
    """
    Agent chuyên xử lý các câu hỏi về y khoa, bệnh lý, và triệu chứng.
    """
    def __init__(self, llm: BaseChatModel, default_tool_names: List[str] = None, **kwargs):
        """
        Khởi tạo MedicalAgent.

        Args:
            llm (BaseChatModel): Language model sẽ được sử dụng.
            default_tool_names (List[str], optional): Tên các tool mặc định luôn chạy.
        """
        agent_name = "MedicalAgent"
        system_prompt = agent_config['medical_agent']['description']
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
        # Chúng ta quyết định 'medical_retriever_tool' là tool mặc định,
        # vì agent này luôn cần truy cập kho kiến thức y khoa của nó.
        medical_agent = MedicalAgent(
            llm=llm, 
            default_tool_names=["medical_retriever_tool"]
        )
        
        # Tạo state mẫu
        state = AgentState(
            original_query="Triệu chứng của bệnh tiểu đường type 2 là gì?",
            rewritten_query="symptoms of type 2 diabetes",
            intents=["retrieve", "medical"], # Giả lập đầu ra từ EntryAgent
            user_role="guest",
            chat_history=[]
        )

        # --- Execution ---
        print(f"--- Executing {medical_agent.agent_name} ---")
        final_state = None
        async for partial_state in medical_agent.astream_execute(state):
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