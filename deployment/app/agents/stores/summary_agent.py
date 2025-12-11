import sys
import asyncio
from typing import List, Dict, Optional

from loguru import logger
from pathlib import Path
from pydantic import BaseModel, Field

# --- LangChain Core & Community Imports ---
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

# --- Local/App Imports ---
sys.path.append(str(Path(__file__).parent.parent.parent))
# Use a flexible AgentState that can be imported
from app.agents.stores.base_agent import BaseAgentNode, AgentState
from app.agents.workflow.initalize import llm_instance, agent_config

# --- Pydantic Model for Structured Output ---
class HistoryCondensationOutput(BaseModel):
    """Defines the structured output for the history condensation task."""
    condensed_summary: str = Field(
        description="A single, concise paragraph summarizing the key points of the previous conversation from the assistant's perspective. It must capture all essential information needed to understand the user's most recent query."
    )

class SummaryAgent(BaseAgentNode):
    """
    An agent that condenses long chat histories to prevent token overflow.
    It replaces a lengthy history with a concise summary, preserving the most
    recent user query.
    """
    def __init__(self, llm: BaseChatModel, history_threshold: int = 6, **kwargs):
        """
        Initializes the SummaryAgent.

        Args:
            llm: The language model instance.
            history_threshold: The number of messages (not pairs) after which to trigger condensation.
        """
        agent_name = "SummaryAgent"
        super().__init__(agent_name=agent_name)
        
        self.llm = llm
        self.history_threshold = history_threshold
        self.system_prompt = agent_config.get(agent_name.lower(), {}).get(
            'description', 
            "You are a helpful AI assistant that excels at summarizing conversations. "
            "Your task is to condense a chat history into a concise summary."
        )
        
        logger.info(f"'{self.agent_name}' initialized with a history threshold of {self.history_threshold} messages.")

    async def aexecute(self, state: AgentState) -> AgentState:
        """
        Conditionally executes the summarization logic if the chat history is too long.
        """
        state = self._prepare_execution(state)
        history: List[BaseMessage] = state.get('chat_history', [])
        
        # 1. Conditional Execution: Only run if history is long enough.
        if len(history) <= self.history_threshold:
            logger.info(f"Chat history length ({len(history)}) is within threshold ({self.history_threshold}). Skipping summarization.")
            return state

        logger.warning(f"Chat history length ({len(history)}) exceeds threshold. Condensing...")

        # 2. Separate the most recent query from the history to be summarized.
        # if not history or not isinstance(history[-1], HumanMessage):
        #     logger.warning("History is empty or does not end with a user message. Skipping.")
        #     return state

        last_user_query = history[-1]
        history_to_summarize = history[:-1]
        logger.info(f"History: {history_to_summarize[:3]}")
        # Convert messages to a readable string format for the prompt
        history_str = "\n".join([f"Role: {message['role']}\nAssistant: {message['content']}" for message in history_to_summarize]) # Lấy 5 cặp hội thoại gần nhất
        logger.info(f"Last user query: {last_user_query}")
        logger.info(f"History to summarize: {history_str[:200]}...")  # Log first 200 chars for brevity
        # Also consider other contexts for a more informed summary
        contexts: Dict = state.get('contexts', {})
        # contexts_str = "\n\n".join([f"--- Context from {tool_name} ---\n{content}" for tool_name, content in contexts.items()])
        
        # 3. Build a more focused prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", 
            "Hãy tóm tắt phần 'Lịch sử cuộc trò chuyện' dưới đây thành một đoạn ngắn gọn, súc tích. "
            "Bản tóm tắt này sẽ thay thế lịch sử cũ, vì vậy nó phải bao gồm tất cả các thông tin, số liệu và quyết định quan trọng. "
            "Bản tóm tắt cũng cần cung cấp đầy đủ ngữ cảnh cần thiết để hiểu được 'Truy vấn gần nhất của người dùng'. "
            "Nếu có, hãy tích hợp cả thông tin liên quan từ phần 'Ngữ cảnh bổ sung'.\n\n"
            "**Lịch sử cuộc trò chuyện cần tóm tắt:**\n{history}\n\n"
            # "**Ngữ cảnh bổ sung:**\n{contexts}\n\n"
            "**Truy vấn gần nhất của người dùng:**\n{last_query}"
            )

        ])
        
        # Using structured output is highly recommended for reliability
        chain = prompt | self.llm.with_structured_output(HistoryCondensationOutput)
        
        try:
            # 4. Invoke the LLM
            response: HistoryCondensationOutput = await chain.ainvoke({
                "history": history_str,
                # "contexts": contexts_str,
                "last_query": last_user_query
            })
            
            summary_text = response.condensed_summary
            
            # 5. Modify the state: Replace old history with the new, condensed version
            new_history = [{"role": "assistant", "content": summary_text}, last_user_query]
            state['chat_history'] = new_history
            logger.success(f"Successfully condensed chat history. New history : {state['chat_history']}")
            logger.success(f"Successfully condensed chat history from {len(history)} to {len(new_history)} messages.")

        except Exception as e:
            logger.error(f"Error during history condensation: {e}")
            state = self._handle_execution_error(e, state)
            logger.error("Failed to condense chat history. The original history will be kept.")
            # Do not modify the history on error to avoid data loss

        return state

if __name__ == "__main__":
    async def main():
        # --- Setup ---
        llm = llm_instance
        # Trigger summary if more than 4 messages (2 pairs of Q&A)
        summary_agent = SummaryAgent(llm=llm, history_threshold=4)
        
        # --- Test Case 1: Long history that should be summarized ---
        print("--- Test Case 1: Long history (should trigger summarization) ---")
        long_history = [
            HumanMessage(content="What is Aspirin used for?"),
            AIMessage(content="Aspirin is commonly used for pain relief, fever reduction, and as an anti-inflammatory drug. It's also used in low doses to prevent heart attacks."),
            HumanMessage(content="What about the BRCA1 gene?"),
            AIMessage(content="The BRCA1 gene is a tumor suppressor gene. Mutations in this gene significantly increase the risk of breast and ovarian cancer."),
            # This is the last user query that must be preserved
            HumanMessage(content="Can you compare their relevance to a 45-year-old male's health check-up?")
        ]
        
        initial_state = AgentState(
            chat_history=long_history,
            contexts={"some_tool": "The patient has a family history of heart disease."},
        )
        
        print(f"Original history length: {len(initial_state['chat_history'])}")
        
        final_state = await summary_agent.aexecute(initial_state)
        
        print("\n--- Result ---")
        print(f"New history length: {len(final_state['chat_history'])}")
        
        new_history = final_state['chat_history']
        assert len(new_history) == 2
        
        summary_message = new_history[0]
        last_query_preserved = new_history[1]
        
        print(f"\n[1] Condensed Summary Message (AI): {summary_message.content}")
        print(f"\n[2] Preserved User Query (Human): {last_query_preserved.content}")

        assert isinstance(summary_message, AIMessage)
        assert "Aspirin" in summary_message.content and "BRCA1" in summary_message.content
        assert last_query_preserved.content == "Can you compare their relevance to a 45-year-old male's health check-up?"

        # --- Test Case 2: Short history that should be skipped ---
        print("\n\n--- Test Case 2: Short history (should be skipped) ---")
        short_history = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there! How can I help?")
        ]
        
        short_state = AgentState(chat_history=short_history)
        print(f"Original history length: {len(short_state['chat_history'])}")
        
        final_short_state = await summary_agent.aexecute(short_state)
        
        print("\n--- Result ---")
        print(f"New history length: {len(final_short_state['chat_history'])}")
        assert final_short_state['chat_history'] == short_history

    asyncio.run(main())