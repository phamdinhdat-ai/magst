import sys
import asyncio
from typing import List, AsyncGenerator, Dict, Any

from loguru import logger
from pathlib import Path
from pydantic import BaseModel, Field

# --- Imports from previous context ---
# Đảm bảo các import này trỏ đến các file đã được tối ưu
sys.path.append(str(Path(__file__).parent.parent.parent))

from app.agents.stores.base_agent import Agent, AgentState , BaseAgentNode
from app.agents.workflow.initalize import llm_instance, agent_config

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage


class EntryAgentOutput(BaseModel):
    """
    Defines the structured output for the EntryAgent.
    """
    intents: List[str] = Field(..., description="List of intents identified in the query.")
    classified_agent: str = Field(..., description="The name of the agent identified to handle the query.")
    needs_rewrite: bool = Field(False, description="Whether the query needs to be rewritten for clarity.")
    rewritten_query: str = Field("", description="The rewritten query if needs_rewrite is True, otherwise the original query.")
    
    class Config:
        """Configuration for Pydantic model"""
        arbitrary_types_allowed = True
    
    def dict(self, **kwargs):
        """Override dict method to make serializable"""
        base_dict = super().dict(**kwargs)
        return base_dict
    
    def to_json(self):
        """Convert to JSON-serializable dict"""
        return self.dict()

# --- Agent Implementation ---

class EntryAgent(BaseAgentNode):
    """
    The first node in the graph. It acts as a planner and router.
    - Classifies user intent.
    - Routes to the appropriate specialist agent.
    - Determines if the query needs rewriting.
    It does NOT use tools to answer the user; its sole purpose is to populate the state for the next step.
    """
    
    def __init__(self, llm: BaseChatModel):
        # Tối ưu #3: __init__ đơn giản hơn, không cần tools
        agent_name = "EntryAgent"
        super().__init__(agent_name=agent_name)
        
        self.llm = llm
        # Lấy prompt từ config, bao gồm danh sách các agent có sẵn để LLM lựa chọn
        self.system_prompt = agent_config['entry_agent_prompt']['description']
        
        logger.info(f"'{self.agent_name}' initialized.")

    async def aexecute(self, state: AgentState) -> AgentState:
        """
        Asynchronously executes the planning and routing logic.
        """
        state = self._prepare_execution(state)
        
        query = state.get('original_query', '')
        history = state.get('chat_history', [])
        workflow_name = "Đây là cuộc trò chuyện của Khách với Trợ lý"
        
        if state.get("customer_id"):
            workflow_name = f"Đây là cuộc trò chuyện của Khách hàng {state['customer_id']} với Trợ lý"
        elif state.get("employee_id"):
            workflow_name = f"Đây là cuộc trò chuyện của Nhân viên {state['employee_id']} với Trợ lý"
        logger.info(f"Workflow Name: {workflow_name}")
        
        
        # Chỉ lấy 3 cặp hội thoại gần nhất
        history_messages = []
        for q, a in history[-3:]:
            history_messages.append(HumanMessage(content=q))
            history_messages.append(AIMessage(content=a))

        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            # MessagesPlaceholder(variable_name="history", optional=True),
            ("human", "Thông tin cuộc trò chuyện: {workflow}\nTruy vấn: {query}")
        ])
        
        # Sử dụng structured_output để đảm bảo LLM trả về JSON theo định dạng mong muốn
        try:
            chain = prompt | self.llm.with_structured_output(EntryAgentOutput)
            
            logger.info(f"Invoking LLM for '{self.agent_name}' to classify and route the query.")
            
            response: EntryAgentOutput = await chain.ainvoke({
                "query": query,
                # "history": history_messages,
                "workflow": workflow_name
            }, config={"callbacks_manager": None})
            
            # Tối ưu #4: Cập nhật state một cách an toàn và rõ ràng
            state['intents'] = [response.intents[0]] if response.intents else []
            state['classified_agent'] = response.classified_agent
            state['needs_rewrite'] = response.needs_rewrite
            # Nếu cần viết lại, sử dụng câu đã viết lại. Nếu không, dùng câu gốc.
            state['rewritten_query'] = response.rewritten_query if response.needs_rewrite else query

            logger.info(f"Routing decision: intents={response.intents}, agent='{response.classified_agent}', needs_rewrite={response.needs_rewrite}")

        except Exception as e:
            # Tối ưu #5: Xử lý lỗi mạnh mẽ hơn
            logger.error(f"Error during LLM execution for {self.agent_name}: {e}")
            state = self._handle_execution_error(e, state)
            # Luôn đảm bảo workflow có thể tiếp tục bằng cách định tuyến đến agent mặc định
            state['intents'] = ["chitchat"]
            state['classified_agent'] = "NaiveAgent" # Fallback agent
            state['needs_rewrite'] = False
            state['rewritten_query'] = query

        return state


if __name__ == "__main__":
    async def main():
        import json
        # --- Setup ---
        # EntryAgent không cần tools, chỉ cần LLM
        entry_agent = EntryAgent(llm=llm_instance)
        test_cases  = json.loads(Path("backend/app/tests/data/entry_test.json").read_text())
        for case in test_cases:
            print(f"Testing query: {case['query']}")
            # state = AgentState(
            #     original_query=case['query'],
            #     chat_history=[],
            #     user_role="guest",
            # )
            # result_state = await entry_agent.aexecute(state)
            # print(f"Result State: {result_state}")
            # print(f"Classified Agent: {result_state.get('classified_agent')}")
            # print(f"Intents: {result_state.get('intents')}")
            # print(f"Needs Rewrite: {result_state.get('needs_rewrite')}")
            # print(f"Rewritten Query: {result_state.get('rewritten_query')}\n")
        # --- Test Case 1: Simple routing ---
       

    # Chạy kịch bản test
    asyncio.run(main())