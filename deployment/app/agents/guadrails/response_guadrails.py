from typing import Dict, Any
from loguru import logger

# --- Local/App Imports ---
from app.agents.workflow.initalize import llm_instance
from app.agents.workflow.state import GraphState as AgentState
from app.agents.stores.base_agent import BaseAgentNode
from langchain_core.language_models.chat_models import BaseChatModel

# --- LangChain Core Imports ---
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# --- Pydantic Model cho Output có cấu trúc ---
# Model này giúp LLM trả về một quyết định rõ ràng và một câu trả lời đã được sửa lỗi.
class ResponseCoherencyDecision(BaseModel):
    """
    Cấu trúc dữ liệu cho quyết định về sự mạch lạc của câu trả lời.
    """
    is_coherent: bool = Field(..., description="True nếu câu trả lời là một phản hồi hợp lý và trực tiếp cho câu hỏi. False nếu nó lạc đề hoặc không liên quan.")
    reason: str = Field(..., description="Giải thích ngắn gọn tại sao câu trả lời mạch lạc hoặc không.")
    corrected_response: str = Field(..., description="Nếu is_coherent=false, hãy cung cấp một câu trả lời ngắn gọn, phù hợp và đúng trọng tâm hơn. Nếu is_coherent=true, hãy trả lại chính câu trả lời ứng viên.")

# --- Prompt Template ---
# Prompt này yêu cầu LLM đóng vai một chuyên gia QA (Quality Assurance)
COHERENCY_PROMPT_TEMPLATE = """
Bạn là một chuyên gia đảm bảo chất lượng AI. Nhiệm vụ của bạn là đánh giá xem "Câu trả lời ứng viên" của một AI có phải là một phản hồi trực tiếp, hợp lý và đúng trọng tâm cho "Câu hỏi gần nhất của người dùng" hay không.

### Phân tích các ví dụ sau:

**VÍ DỤ TỐT (Mạch lạc):**
-   **Câu hỏi gần nhất của người dùng:** "Aspirin là gì?"
-   **Câu trả lời ứng viên:** "Aspirin (axit acetylsalicylic) là một loại thuốc dùng để giảm đau, hạ sốt và chống viêm. Nó cũng được sử dụng với liều lượng thấp để giúp ngăn ngừa các cơn đau tim."
-   **Phân tích của bạn:** Câu trả lời này hoàn toàn mạch lạc, định nghĩa và giải thích rõ về Aspirin.

**VÍ DỤ TỒI (Không mạch lạc):**
-   **Câu hỏi gần nhất của người dùng:** "chào bạn"
-   **Câu trả lời ứng viên:** "Rất tiếc, hiện tại tôi không có thông tin nào về người có tên Phạm Đình Đạt trong hồ sơ của GeneStory. Để tôi có thể hỗ trợ bạn tốt hơn, bạn có thể cung cấp thêm thông tin không?"
-   **Phân tích của bạn:** Câu trả lời này hoàn toàn không mạch lạc. Nó trả lời một vấn đề cũ không liên quan thay vì chào lại người dùng. Một câu trả lời đúng phải là một lời chào đơn giản.

---
### Yêu cầu thực tế:

**Câu hỏi gần nhất của người dùng:**
{query}

**Câu trả lời ứng viên (do một agent khác tạo ra):**
{candidate_response}

**Hãy đánh giá và trả về một đối tượng JSON theo đúng cấu trúc yêu cầu.**
"""

class ResponseGuardrail(BaseAgentNode):

    def __init__(self, llm: BaseChatModel):
        super().__init__(agent_name="ResponseGuardrail")
        
        prompt = ChatPromptTemplate.from_template(COHERENCY_PROMPT_TEMPLATE)
        self.chain = prompt | llm.with_structured_output(ResponseCoherencyDecision)
        
        logger.info(f"Node '{self.agent_name}' initialized successfully.")

    async def aexecute(self, state: AgentState) -> AgentState:
        """
        Thực thi node guardrail cho câu trả lời.
        """
        state = self._prepare_execution(state)
        logger.info(f"--- Executing: {self.agent_name} ---")

        try:
            query = state.get("rewritten_query") or state.get("original_query", "")
            candidate_response = state.get("agent_response", "")

            # Nếu không có câu hỏi hoặc không có câu trả lời ứng viên, bỏ qua bước kiểm tra
            if not query or not candidate_response:
                logger.info("Skipping ResponseGuardrail due to empty query or response.")
                return state

            logger.info(f"Checking response coherency. Query: '{query}', Candidate Response: '{candidate_response[:150]}...'")

            # Gọi LLM để đánh giá
            decision: ResponseCoherencyDecision = await self.chain.ainvoke({
                "query": query,
                "candidate_response": candidate_response
            })
            
            logger.info(f"Response Guardrail decision: is_coherent={decision.is_coherent}, Reason='{decision.reason}'")

            # --- LOGIC QUAN TRỌNG: SỬA LỖI CÂU TRẢ LỜI ---
            if not decision.is_coherent:
                logger.warning("Response was NOT coherent. Correcting the response.")
                # Ghi đè câu trả lời lạc đề bằng câu trả lời đã được sửa lỗi
                state['agent_response'] = decision.corrected_response
                state['agent_thinks'].update({"ResponseGuardrail": decision.reason})
                # Thêm một cờ để theo dõi (tùy chọn)
                state['was_response_corrected'] = True
            else:
                logger.info("Response is coherent. Passing through.")
                state['was_response_corrected'] = False

        except Exception as e:
            logger.error(f"An exception occurred in {self.agent_name}: {e}", exc_info=True)
            state = self._handle_execution_error(e, state)

        return state

# Tạo instance để sử dụng trong graph
response_guardrail_node = ResponseGuardrail(llm=llm_instance)