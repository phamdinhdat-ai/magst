import sys
import asyncio
from typing import List

from loguru import logger
from pathlib import Path
from pydantic import BaseModel, Field

# --- LangChain Core & Community Imports ---
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# --- Local/App Imports ---
sys.path.append(str(Path(__file__).parent.parent.parent))
# Sửa các import này cho đúng với cấu trúc dự án của bạn
from app.agents.stores.base_agent import BaseAgentNode, AgentState  # Sử dụng AgentState đã định nghĩa
from app.agents.workflow.initalize import llm_instance, agent_config  # Import phiên bản
from app.agents.factory.tools.base import BaseAgentTool
# --- Pydantic Model for Structured Output ---
class SuggestedQuestionsOutput(BaseModel):
    """Định nghĩa cấu trúc đầu ra cho các câu hỏi gợi ý."""
    suggested_questions: List[str] = Field(
        default_factory=list,
        description="Danh sách các câu hỏi gợi ý từ 3 đến 4 câu"
    )

class QuestionGeneratorAgent(BaseAgentNode):
    """
    Một node xử lý sau, có nhiệm vụ tạo ra các câu hỏi gợi ý dựa trên
    toàn bộ ngữ cảnh của cuộc hội thoại vừa kết thúc.
    """
    def __init__(self, llm: BaseChatModel, **kwargs):
        """Khởi tạo QuestionGeneratorAgent."""
        agent_name = "QuestionGeneratorAgent"
        super().__init__(agent_name=agent_name)
        
        self.llm = llm
        self.system_prompt = agent_config['question_agent']['description']

        logger.info(f"'{self.agent_name}' initialized.")

    def _get_default_suggestions(self) -> List[str]:
        """Trả về danh sách các câu hỏi mặc định khi có lỗi."""
        logger.warning("Falling back to default suggested questions.")
        return [
            "Công ty GeneStory có những sản phẩm nào?",
            "Làm thế nào để liên hệ với bộ phận hỗ trợ?",
            "So sánh các gói xét nghiệm di truyền.",
            "Cho tôi biết thêm về nguy cơ bệnh tim mạch.",
        ]

    async def aexecute(self, state: AgentState) -> AgentState:
        """
        Thực thi logic tạo câu hỏi gợi ý một cách bất đồng bộ.
        """
        state = self._prepare_execution(state)
        
        # Thu thập toàn bộ ngữ cảnh từ state
        query = state.get('original_query', '')
        final_answer = state.get('agent_response', '')
        history = state.get('chat_history', [])
        # Ghép các context đã dùng thành một chuỗi
        contexts_str = "\n".join(state.get('contexts', {}).values())

        # Nếu không có thông tin gì, trả về câu hỏi mặc định
        if not final_answer and not history:
            logger.info("Not enough context to generate questions. Using defaults.")
            state['suggested_questions'] = self._get_default_suggestions()
            return state

        # Chuyển đổi lịch sử chat
        history_messages: List[BaseMessage] = []
        for q, a in history[-3:]: # Lấy 3 cặp hội thoại gần nhất
            history_messages.append(HumanMessage(content=q))
            history_messages.append(AIMessage(content=a))

        # Xây dựng prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", 
             "Dựa trên cuộc hội thoại trên và thông tin bổ sung dưới đây, hãy tạo ra 3-4 câu hỏi tiếp theo thật sự hữu ích và có liên quan mà người dùng có thể muốn hỏi.\n"
             "Câu hỏi nên ngắn gọn, dễ hiểu và gợi mở các chủ đề liên quan nhưng chưa được khám phá sâu.\n"
             "### Truy vấn cuối cùng của người dùng:\n{query}\n\n"
             "### Câu trả lời cuối cùng của trợ lý:\n{final_answer}\n\n"
             "### Thông tin nền tảng đã được sử dụng:\n{contexts}"
            )
        ])
        
        chain = prompt | self.llm.with_structured_output(SuggestedQuestionsOutput)
        
        logger.info(f"Invoking LLM for '{self.agent_name}'...")
        try:
            response = await chain.ainvoke({
                "chat_history": history_messages,
                "query": query,
                "final_answer": final_answer,
                "contexts": contexts_str
            })
           
            if len (response.suggested_questions) == 0:
                logger.warning("No questions generated, using default suggestions.")
                state['suggested_questions'] = self._get_default_suggestions()
            else:
                # Giới hạn số lượng câu hỏi gợi ý
                state['suggested_questions'] = response.suggested_questions[:4]
            logger.info(f"Generated suggested questions: {response.suggested_questions}")
        except Exception as e:
            logger.error(f"Failed to generate suggested questions: {e}")
            state = self._handle_execution_error(e, state)
            # Quan trọng: Luôn cung cấp câu hỏi mặc định khi có lỗi
            state['suggested_questions'] = self._get_default_suggestions()

        return state

if __name__ == "__main__":
    async def main():
        # --- Setup ---
        llm = llm_instance
        # Khởi tạo agent rất đơn giản, không cần tool
        question_agent = QuestionGeneratorAgent(llm=llm)
        
        # --- Test Case 1: Có đầy đủ context ---
        print("--- Test Case 1: Full context ---")
        state_full = AgentState(
            original_query="Đột biến gen BRCA1 là gì?",
            chat_history=[("Chào bạn", "Chào bạn, tôi có thể giúp gì?")],
            contexts={
                "genetic_retriever_tool": "Gen BRCA1 là một gen ức chế khối u. Đột biến ở gen này làm tăng nguy cơ ung thư vú và buồng trứng."
            },
            agent_response="Đột biến ở gen BRCA1 làm tăng nguy cơ mắc ung thư vú và buồng trứng. Bạn có muốn tìm hiểu thêm về các biện pháp phòng ngừa không?"
        )
        
        final_state = await question_agent.aexecute(state_full)
        
        print("\n--- Final Result for Test Case 1 ---")
        if final_state.get('error_message'):
            print(f"Error: {final_state['error_message']}")
        else:
            print(f"Suggested Questions: {final_state.get('suggested_questions')}")

        # --- Test Case 2: Context rỗng ---
        print("\n--- Test Case 2: Empty context ---")
        state_empty = AgentState(
            original_query="",
            chat_history=[],
        )
        final_state_2 = await question_agent.aexecute(state_empty)
        print("\n--- Final Result for Test Case 2 ---")
        print(f"Suggested Questions: {final_state_2.get('suggested_questions')}")

    # Chạy kịch bản test
    asyncio.run(main())