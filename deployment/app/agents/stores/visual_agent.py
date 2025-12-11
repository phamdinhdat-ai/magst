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
from app.agents.factory.tools.image_analysis_tool import ImageAnalysisTool  # Import tool phân tích
from app.agents.factory.factory_tools import TOOL_FACTORY  # Import factory tools

class VisualAgent(Agent):
    """
    Agent chuyên xử lý các truy vấn liên quan đến hình ảnh.
    Nó sử dụng ImageAnalysisTool để hiểu nội dung ảnh và trả lời câu hỏi.
    """
    def __init__(self, llm: BaseChatModel, default_tool_names: List[str] = None, **kwargs):
        """
        Khởi tạo VisualAgent.

        Args:
            llm (BaseChatModel): Language model sẽ được sử dụng.
            default_tool_names (List[str], optional): Tên các tool mặc định luôn chạy.
        """
        agent_name = "VisualAgent"
        # Prompt này hướng dẫn agent cách sử dụng mô tả ảnh từ tool
        system_prompt = agent_config['visual_agent']['description']
        
        super().__init__(
            llm=llm, 
            agent_name=agent_name, 
            system_prompt=system_prompt, 
            default_tool_names=default_tool_names or [],
            **kwargs
        )
        
        logger.info(f"'{self.agent_name}' initialized. It will use visual analysis tools.")
    
    # KHÔNG CẦN CÁC PHƯƠNG THỨC GHI ĐÈ.
    # Lớp `Agent` cơ sở sẽ tự động:
    # 1. Thấy intent là 'visual'.
    # 2. Yêu cầu 'image_analyzer' từ ToolFactory.
    # 3. Chạy tool với state hiện tại (trong đó phải có 'image_path').
    # 4. Đưa context (mô tả ảnh) vào prompt và gọi LLM.


if __name__ == "__main__":
    async def main():
        # --- Setup ---
        # Tạo một file ảnh giả để test
        from PIL import Image, ImageDraw, ImageFont
        
        img = Image.new('RGB', (600, 300), color = (20, 20, 80))
        d = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 30)
        except IOError:
            font = ImageFont.load_default()
        d.text((10,10), "GeneStory Analysis\nChart: Gene Expression Levels\nGene X: High\nGene Y: Low", fill=(200, 200, 200), font=font)
        
        test_image_path = Path("/home/datpd1/genstory/multi-agent-app/agentic-gst-chatbot/backend/app/storage/Screenshot from 2025-05-28 15-40-42.png")

        img.save(test_image_path)
        print(f"Created a test image at: {test_image_path.resolve()}")
        
        llm = llm_instance
        
        # Khởi tạo agent.
        # Đặt 'image_analysis_tool' làm tool mặc định để nó luôn chạy khi VisualAgent được gọi.
        visual_agent = VisualAgent(
            llm=llm, 
            default_tool_names=["image_analysis_tool"]
        )
        
        # Tạo state mẫu. Rất quan trọng là phải có `image_path`.
        state = AgentState(
            original_query="Hay mo ta noi dung hinh anh nay.",
            intents=["visual"], # Giả lập đầu ra từ EntryAgent
            image_path=str(test_image_path), # Cung cấp đường dẫn ảnh
            user_role="researcher",
            chat_history=[]
        )

        # --- Execution ---
        print(f"\n--- Executing {visual_agent.agent_name} ---")
        final_state = await visual_agent.aexecute(state)
        
        print("\n--- Final Result ---")
        if final_state.get('error_message'):
            print(f"Error: {final_state['error_message']}")
        else:
            print(f"Agent Response: {final_state.get('agent_response')}")
        
        print("\n--- Contexts Used by Agent ---")
        # Context sẽ chứa mô tả ảnh từ ImageAnalysisTool
        print(final_state.get('contexts'))
        
        # Dọn dẹp file test
        test_image_path.unlink()

    # Chạy kịch bản test
    asyncio.run(main())