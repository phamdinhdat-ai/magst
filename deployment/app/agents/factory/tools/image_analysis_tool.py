import sys
import base64
import asyncio
from typing import Set, Dict, Any, Optional
from pathlib import Path

from loguru import logger
from pydantic import Field

# --- LangChain Core & Community Imports ---
# Chúng ta cần một ChatModel có khả năng xử lý Vision
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel

# --- Local/App Imports ---
sys.path.append(str(Path(__file__).parent.parent.parent))
# Giả định BaseAgentTool nằm ở factory/tools/base.py
from app.agents.factory.tools.base import BaseAgentTool
from app.agents.workflow.state import GraphState as AgentState  # Sử dụng AgentState đã định nghĩa
from app.agents.workflow.initalize import llm_instance # Dùng llm_instance làm fallback

# --- Helper Function ---
def image_to_base64(image_path: Path) -> Optional[str]:
    """Chuyển đổi file ảnh sang chuỗi base64."""
    if not image_path.is_file():
        logger.error(f"Image file not found at path: {image_path}")
        return None
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Failed to read or encode image file {image_path}: {e}")
        return None

# --- Tool Implementation ---

class ImageAnalysisTool(BaseAgentTool):
    """
    Một công cụ phân tích hình ảnh để tạo ra mô tả (caption).
    Nó nhận đầu vào là đường dẫn file ảnh và một câu truy vấn,
    sau đó sử dụng một mô hình Vision-LLM để mô tả nội dung ảnh.
    """
    # --- Cấu hình Tool ---
    name: str = "image_analyzer"
    description: str = "Analyzes an image to provide a detailed description of its content, relevant to a user's query."
    # Tool này được kích hoạt bởi intent 'visual' hoặc 'retrieve' khi có ảnh
    intents: Set[str] = {"visual", "retrieve"}
    is_async: bool = True # Đánh dấu là tool bất đồng bộ

    # --- Cấu hình nội bộ ---
    # Sử dụng một mô hình Vision riêng, hoặc fallback về llm mặc định
    vision_llm: BaseChatModel = Field(
        description="A vision-enabled language model, like Llava.",
        default_factory=lambda: ChatOllama(model="llava:latest") # Cấu hình mô hình Vision của bạn ở đây
    )

    def run (self, state: AgentState) -> str:  
        """
        Thực thi đồng bộ (không khuyến khích, chỉ để tương thích).
        Chạy phiên bản async và đợi kết quả.
        """
        logger.warning("Synchronous run is deprecated. Use async _arun instead.")
        return self._run(state)
    
    def arun(self, state: AgentState) -> str:
        """
        Thực thi bất đồng bộ.
        """
        logger.warning("Asynchronous run is deprecated. Use _arun instead.")
        return self._arun(state)
    
    def _run(self, state: AgentState) -> str:
        """
        Thực thi đồng bộ (không khuyến khích, chỉ để tương thích).
        Chạy phiên bản async và đợi kết quả.
        """
        # Đây là một anti-pattern nhỏ nhưng cần thiết để tương thích với BaseTool
        # Trong thực tế, chúng ta sẽ luôn gọi _arun
        return asyncio.run(self._arun(state))

    async def _arun(self, state: AgentState) -> str:
        """
        Thực thi phân tích ảnh một cách bất đồng bộ.
        """
        # 1. Lấy thông tin từ state
        # Giả định state có key `image_path`
        image_path_str = state.get("image_path")
        query = state.get("rewritten_query") or state.get("original_query", "Describe this image in detail.")
        
        if not image_path_str:
            return "Lỗi: Không tìm thấy đường dẫn file ảnh (`image_path`) trong state."
        
        image_path = Path(image_path_str)
        
        # 2. Chuyển đổi ảnh sang base64
        base64_image = image_to_base64(image_path)
        if not base64_image:
            return f"Lỗi: Không thể đọc hoặc xử lý file ảnh tại '{image_path_str}'."

        logger.info(f"Analyzing image '{image_path.name}' with query: '{query}'")

        # 3. Xây dựng prompt cho mô hình Vision
        prompt_messages = [
            SystemMessage(content="Bạn là một mô hình AI có khả năng phân tích hình ảnh và tạo mô tả chi tiết về nội dung của chúng."),
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": f"User query: '{query}'.\n\nBased on this query, please provide a detailed caption for the following image:"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            )
        ]
        
        # 4. Gọi LLM và trả về kết quả
        try:
            response = await self.vision_llm.ainvoke(prompt_messages)
            caption = response.content
            logger.info(f"Successfully generated caption for image '{image_path.name}'.")
            return f"Description of the image '{image_path.name}':\n{caption}"
        except Exception as e:
            logger.error(f"Failed to invoke vision model for image analysis: {e}")
            return "Lỗi: Đã có sự cố khi mô hình AI phân tích hình ảnh."

# ==============================================================================
# === TEST EXECUTION
# ==============================================================================
if __name__ == '__main__':
    async def main():
        # --- Setup ---
        # Tạo một file ảnh giả để test
        from PIL import Image, ImageDraw, ImageFont
        
        # Tạo ảnh
        img = Image.new('RGB', (600, 300))
        d = ImageDraw.Draw(img)
        try:
            # Cố gắng tải font, nếu không được thì dùng font mặc định
            font = ImageFont.truetype("arial.ttf", 40)
        except IOError:
            font = ImageFont.load_default()
        d.text((10,10), "GeneStory Lab Report\n\nPatient ID: 12345\nGene: BRCA1 (Positive)", fill=(255,255,0), font=font)

        test_image_path = Path("/home/datpd1/genstory/multi-agent-app/agentic-gst-chatbot/backend/app/storage/Screenshot from 2025-05-28 15-40-42.png")
        # img.save(test_image_path)
        print(f"Created a test image at: {test_image_path.resolve()}")

        # Khởi tạo tool
        # Đảm bảo bạn đã chạy `ollama pull llava`
        try:
            image_tool = ImageAnalysisTool()
        except Exception as e:
            print(f"\nCould not initialize ImageAnalysisTool. Is Ollama running and is 'llava' model available? Error: {e}")
            return

        # --- Test Case 1: Phân tích ảnh báo cáo gen ---
        print("\n--- Test Case 1: Analyze a genetic report image ---")
        state = AgentState(
            original_query="Tóm tắt nội dung chính trong ảnh báo cáo này.",
            image_path=str(test_image_path)
        )
        
        # Chạy tool
        result = await image_tool._arun(state)
        
        print("\n--- Analysis Result ---")
        print(result)

        # Dọn dẹp file test
        test_image_path.unlink()
        
    asyncio.run(main())