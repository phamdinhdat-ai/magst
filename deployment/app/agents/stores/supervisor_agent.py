import sys
import asyncio
from typing import List, Dict, AsyncGenerator, Optional, Tuple

from loguru import logger
from pathlib import Path

# --- LangChain Core & Community Imports ---
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# --- Local/App Imports ---
sys.path.append(str(Path(__file__).parent.parent.parent))
from app.agents.stores.base_agent import BaseAgentNode, AgentState  # Sử dụng AgentState đã định nghĩa
from app.agents.workflow.initalize import llm_instance, agent_config  # Import phiên bản
from app.agents.factory.tools.base import BaseAgentTool
from app.agents.factory.factory_tools import TOOL_FACTORY  # Import factory tools
class SupervisorAgent(BaseAgentNode):
    """
    Agent cuối cùng, có khả năng stream câu trả lời tổng hợp.
    """
    def __init__(self, llm: BaseChatModel, history_k: int = 5, **kwargs):
        agent_name = "SupervisorAgent"
        system_prompt = agent_config['supervisor_agent_v3']['description']
        super().__init__(agent_name=agent_name)
        self.llm = llm
        self.system_prompt = system_prompt
        self.history_k = history_k  # Số lượng lịch sử chat sẽ được sử dụng
        logger.info(f"'{self.agent_name}' initialized for streaming final responses.")

    async def aexecute(self, state: AgentState) -> AgentState:
        """
        Thực thi và trả về state cuối cùng sau khi stream kết thúc.
        """
        final_state = state
        async for partial_state in self.astream_execute(state):
            final_state = partial_state
        return final_state

    # === THÊM PHƯƠNG THỨC STREAMING MỚI ===
    async def astream_execute(self, state: AgentState) -> AsyncGenerator[AgentState, None]:
        """
        Thực thi logic tổng hợp và stream câu trả lời cuối cùng.
        """
        state = self._prepare_execution(state)
        
        try:
            # --- 1. Thu thập và định dạng tất cả thông tin có sẵn (Logic giữ nguyên) ---
            query = state.get("rewritten_query") or state.get("original_query", "")
            
            
            all_info_parts = []
            contexts = state.get("contexts", {})
            # logger.debug(f"Contexts prepared for final response: {contexts}")
            reflection_feedback = state.get("reflection_feedback", {})
            if contexts:
                context_str = "\n".join([f"- {content}" for content in contexts.values()])
                all_info_parts.append(f"### Key Information Gathered:\n{context_str}")
            agent_thinks = state.get("agent_thinks", {})
            if agent_thinks:
                think_str = "\n".join([f"- Agent '{name}' thought: {str(think)}" for name, think in agent_thinks.items()])
                all_info_parts.append(f"### Intermediate Agent Thoughts:\n{think_str}")
            # logger.debug(f"Agent thoughts prepared for final response: {agent_thinks}")
            full_context_str = "\n\n".join(all_info_parts)
            if not full_context_str:
                full_context_str = "No specific information was gathered."
            logger.debug(f"Full context for final response: {full_context_str}")
            history_messages = self._format_chat_history(state.get("chat_history", []))
            logger.debug(f"Formatted history messages: {history_messages}")
            logger.info(f"Preparing final response for query: {query}")
            intents = state.get("intents", [])
            # --- 2. Xây dựng prompt cuối cùng (Logic giữ nguyên) ---
            prompt = ChatPromptTemplate.from_messages([
                ("system", self.system_prompt),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("human",
                "Dựa trên câu hỏi gốc của tôi và tất cả thông tin mà hệ thống của bạn đã thu thập bên dưới, vui lòng tạo ra câu trả lời cuối cùng đã được chỉnh sửa hoàn chỉnh.\n\n"
                "**Câu hỏi của tôi:**\n{user_query}\n\n"
                "**Mục đích của câu hỏi này là: {intents} **\n"
                "**Tất cả thông tin có sẵn để bạn tổng hợp:**\n{full_context}\n\n"
                "**Tổng hợp các suy nghĩ của các agent khác:**\n{agent_thoughts}\n\n"
                "**Đánh giá phản hồi của các agents :**\n{reflection_feedback}\n\n"
                "**Hãy đưa ra câu trả lời chính xác cho câu hỏi của tôi.**")
            ])

            chain = prompt | self.llm
            logger.info(f"Streaming final response from '{self.agent_name}'...")

            # --- 3. Thực thi và STREAM kết quả ---
            full_response = ""
            async for chunk in chain.astream({
                "chat_history": history_messages,
                "user_query": query,
                "intents": intents,
                "contexts": contexts, 
                "full_context": full_context_str,
                "reflection_feedback": reflection_feedback,
                "agent_thoughts": state.get("agent_thinks", {})
            }):
                if hasattr(chunk, 'content'):
                    full_response += chunk.content
                    state["agent_response"] = full_response
                    yield state # Yield state cập nhật với mỗi chunk mới
            logger.info(f"Final response from '{self.agent_name}': {full_response}...")  # Log đầu 100 ký tự của câu trả lời cuối cùng
            logger.info("Finished streaming final response.")
            state['is_final_answer'] = True
            yield state # Yield state cuối cùng với cờ is_final_answer

        except Exception as e:
            state = self._handle_execution_error(e, state)
            logger.error(f"Error during streaming execution: {e}")
            state['is_final_answer'] = True
            yield state

    def _format_chat_history(self, history: List[Tuple[str, str]]) -> List[BaseMessage]:
        """Chuyển đổi lịch sử chat từ tuple sang đối tượng message của LangChain."""
        messages = []
        logger.debug(f"show chat history: {history}")
        for item in history[-self.history_k:]: # Lấy k cặp hội thoại gần nhất
            if item['role'] == 'user':
                messages.append(HumanMessage(content=item['content']))
            elif item['role'] == 'assistant':
                messages.append(AIMessage(content=item['content']))
            else:
                logger.warning(f"Unknown role in chat history: {item['role']}. Skipping this message.")
        logger.debug(f"Formatted chat history: {messages}")
        return messages
    
if __name__ == "__main__":
    async def main():
        # --- Setup ---
        llm = llm_instance
        supervisor_agent = SupervisorAgent(llm=llm)
        
        # --- Test Case ---
        print("--- Testing SupervisorAgent with Streaming ---")
        state_to_finalize = AgentState(
            original_query="So sánh Aspirin và Paracetamol.",
            contexts={
                "drug_retriever": "Aspirin is an NSAID for pain, inflammation... Paracetamol is for pain and fever..."
            }
        )
        
        full_answer = ""
        final_state = None

        # Mô phỏng client nhận các chunk từ stream
        async for partial_state in supervisor_agent.astream_execute(state_to_finalize):
            new_part = partial_state.get("agent_response", "").replace(full_answer, "", 1)
            print(new_part, end="", flush=True)
            full_answer = partial_state.get("agent_response", "")
            final_state = partial_state
        
        print("\n\n--- Final State from Test ---")
        if final_state:
            print(f"Is Final Answer: {final_state.get('is_final_answer')}")
            print(f"Full Reconstructed Answer: {full_answer}")
        
    # Chạy kịch bản test
    asyncio.run(main())