import sys
import asyncio
from typing import List, Optional

from loguru import logger
from pathlib import Path
from pydantic import BaseModel, Field

# --- LangChain Core & Community Imports ---
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# --- Local/App Imports ---
sys.path.append(str(Path(__file__).parent.parent.parent))
from app.agents.stores.base_agent import Agent, AgentState  # Sử dụng AgentState đã định nghĩa
from app.agents.workflow.initalize import llm_instance, agent_config  # Import phiên bản
from app.agents.factory.tools.base import BaseAgentTool
from app.agents.factory.factory_tools import TOOL_FACTORY  # Import factory tools
import json 
# --- Pydantic Model for Structured Output ---
class ReflectionOutput(BaseModel):
    """Định nghĩa cấu trúc đầu ra cho ReflectionAgent."""
    
    is_final_answer: bool = Field(
        ...,
        description="True if the response is complete and accurate, False if it needs revision or more information."
    )
    suggested_agent_for_followup: Optional[str] = Field(
        None,
        description="If the answer is not final, suggest the name of another specialized agent that could improve the answer. E.g., 'DrugAgent', 'GeneticAgent'."
    )
    # quality_score bị loại bỏ để LLM tập trung vào quyết định và phản hồi
    # thay vì một con số trừu tượng.

class ReflectionAgent(Agent):
    """
    Một agent chuyên đánh giá chất lượng câu trả lời của agent khác.
    Nó quyết định xem câu trả lời đã đủ tốt chưa hoặc có cần chuyển cho agent khác không.
    """
    def __init__(self, llm: BaseChatModel, **kwargs):
        agent_name = "ReflectionAgent"
        system_prompt = agent_config['reflection_agent_v2']['description']
        # Gọi __init__ của lớp cha. 
        # Chúng ta sẽ cấu hình tool mặc định khi khởi tạo instance.
        super().__init__(
            llm=llm,
            agent_name=agent_name,
            system_prompt=system_prompt,
            **kwargs
        )
        logger.info(f"'{self.agent_name}' initialized.")

    async def aexecute(self, state: AgentState) -> AgentState:
        """
        Thực thi logic phản ánh một cách bất đồng bộ.
        """
        state = self._prepare_execution(state)
        
        # Thu thập thông tin cần thiết từ state
        query = state.get("rewritten_query") or state.get("original_query", "")
        agent_response = state.get("agent_response", "Không có phản hồi nào từ trợ lý trước đó.")
        contexts = state.get("contexts", {})
        chat_history = state.get("chat_history", [])
        
        # --- Tóm tắt Contexts (sử dụng cơ chế Tool của lớp cha) ---
        # Chúng ta sẽ chạy 'SummaryTool' trên mỗi context.
        # Logic này có thể được đơn giản hóa nếu SummaryTool có thể nhận một dict.
        # Ở đây, chúng ta sẽ chuẩn bị một "query" đặc biệt cho tool tóm tắt.
        # summary_query = json.dumps(contexts)
        # tools_to_run = self._get_tools_to_run(state)
        # summarized_contexts = await self._arun_tools_in_parallel(tools_to_run, summary_query)
        # state["tool_summaries"] = summarized_contexts

        # Ghép các context đã tóm tắt thành một chuỗi
        contexts_str = "\n".join(contexts.values())
        workflow_name = "Đây là cuộc trò chuyện của Khách với Trợ lý"
        
        if state.get("customer_id"):
            workflow_name = f"Đây là cuộc trò chuyện của Khách hàng {state['customer_id']} với Trợ lý"
        elif state.get("employee_id"):
            workflow_name = f"Đây là cuộc trò chuyện của Nhân viên {state['employee_id']} với Trợ lý"
        logger.info(f"Workflow Name: {workflow_name}")
        
        classified_agent = state.get("classified_agent", "Unknown")
        # Chuyển đổi lịch sử chat - handle both formats
        history_messages = []
        if chat_history:
            # Check if chat history is in dict format with 'role' and 'content' keys
            if isinstance(chat_history[0], dict) and 'role' in chat_history[0] and 'content' in chat_history[0]:
                for msg in chat_history[-self.history_k:]:
                    if msg['role'] == 'user':
                        history_messages.append(HumanMessage(content=msg['content']))
                    elif msg['role'] == 'assistant':
                        history_messages.append(AIMessage(content=msg['content']))
            # Otherwise assume it's in the old pair format [user_msg, ai_msg]
            else:
                try:
                    history_messages = [
                        msg for pair in chat_history[-self.history_k:] 
                        for msg in (HumanMessage(content=pair[0]), AIMessage(content=pair[1]))
                    ]
                except (IndexError, KeyError):
                    # If any issue with the format, skip adding history
                    logger.warning("Could not parse chat history format, skipping history")
                    history_messages = []
    
        # Xây dựng prompt để phản ánh
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human",
            "Vui lòng xem xét tương tác sau đây:\n\n"
            "**Truy vấn của người dùng:**\n{query}\n\n"
            "**Thông tin được trợ lý sử dụng:**\n{contexts}\n\n"
            "**Câu trả lời cuối cùng của trợ lý: {classified_agent}:**\n{agent_response}\n\n"
            "Dựa trên tất cả những thông tin trên, hãy đưa ra phản hồi phản biện của bạn."
            )
        ])

        
        chain = prompt | self.llm.with_structured_output(ReflectionOutput)
        logger.info(f"Invoking LLM for '{self.agent_name}' to reflect on the answer...")
        
        try:
            response = await chain.ainvoke({
                "chat_history": history_messages,
                "query": query,
                "contexts": contexts_str,
                "agent_response": agent_response,
                "classified_agent": classified_agent
                
            })
            
            # Cập nhật state với kết quả phản ánh
            
            state['is_final_answer'] = response.is_final_answer
            state['suggest_agent_followups'] = response.suggested_agent_for_followup
            logger.info("suggested_agent_followups: " + str(state['suggest_agent_followups']))
            # logger.info(f"Reflection complete. Final Answer: {response.is_final_answer}. Feedback: {response.feedback}")
            if response.is_final_answer:
                state['reflection_feedback'] = "Câu trả lời đã đầy đủ và chính xác."
            else:
                state['reflection_feedback'] = "Câu trả lời cần cải thiện hoặc bổ sung thông tin."
        except Exception as e:
            logger.error(f"Error during reflection: {e}")
            state = self._handle_execution_error(e, state)
            # Luôn đặt trạng thái an toàn khi có lỗi: yêu cầu làm lại
            state['reflection_feedback'] = "An error occurred during reflection. Assuming the answer is not final."
            state['is_final_answer'] = False
            state['suggest_agent_followups'] = None # Không gợi ý agent nào
        
        return state

if __name__ == "__main__":
    async def main():
        # --- Setup ---
        llm = llm_instance
        # ReflectionAgent cần SummaryTool. Chúng ta đặt nó làm tool mặc định.
        reflection_agent = ReflectionAgent(
            llm=llm,
            default_tool_names=["summary_tool"]
        )
        
        # --- Test Case 1: Câu trả lời tốt ---
        print("--- Test Case 1: Good Answer ---")
        state_good = AgentState(
            original_query="Đột biến gen BRCA1 là gì?",
            contexts={
                "genetic_retriever_tool": "Gen BRCA1 là gen ức chế khối u. Đột biến làm tăng nguy cơ ung thư vú và buồng trứng."
            },
            agent_response="Đột biến ở gen BRCA1 làm tăng nguy cơ mắc ung thư vú và buồng trứng.",
            chat_history=[]
        )
        
        final_state_1 = await reflection_agent.aexecute(state_good)
        
        print("\n--- Final Result for Test Case 1 ---")
        print(f"Is Final Answer: {final_state_1.get('is_final_answer')}")
        print(f"Feedback: {final_state_1.get('reflection_feedback')}")
        print(f"Suggested Follow-up: {final_state_1.get('suggest_agent_followups')}")

        # --- Test Case 2: Câu trả lời chưa đầy đủ ---
        print("\n--- Test Case 2: Incomplete Answer ---")
        state_incomplete = AgentState(
            original_query="Thuốc Aspirin dùng để làm gì và liều dùng cho người lớn?",
            contexts={
                "drug_retriever": "Aspirin là một loại thuốc chống viêm không steroid (NSAID), được sử dụng để giảm đau, hạ sốt và chống viêm."
            },
            agent_response="Aspirin được sử dụng để giảm đau và hạ sốt.", # Thiếu thông tin về liều dùng
            chat_history=[]
        )
        final_state_2 = await reflection_agent.aexecute(state_incomplete)

        print("\n--- Final Result for Test Case 2 ---")
        print(f"Is Final Answer: {final_state_2.get('is_final_answer')}")
        print(f"Feedback: {final_state_2.get('reflection_feedback')}")
        print(f"Suggested Follow-up: {final_state_2.get('suggest_agent_followups')}")
        
    # Chạy kịch bản test
    asyncio.run(main())