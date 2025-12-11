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
# Để code này hoạt động, bạn cần đảm bảo các tool mà GuestAgent có thể dùng
# (ví dụ: 'searchweb_tool', 'product_retriever_tool') đã được đăng ký
# trong ToolFactory

class RewriterAgentOutput(BaseModel):
    rewritten_query: str = Field(..., description="Câu hỏi đã được viết lại để phù hợp với ngữ cảnh và mục tiêu của người dùng.")
    

class RewriterAgent(Agent):
    """
    Agent chuyên viết lại câu hỏi để phù hợp với ngữ cảnh và mục tiêu của người dùng.
    Nó sẽ nhận đầu vào là câu hỏi gốc và các thông tin bổ sung từ state
    """
    def __init__(self, llm: BaseChatModel, default_tool_names: List[str] = None, **kwargs):
        """
        Khởi tạo RewriterAgent.

        Args:
            llm (BaseChatModel): Language model sẽ được sử dụng.
            default_tool_names (List[str], optional): Tên các tool mặc định luôn chạy.
        """
        agent_name = "RewriterAgent"
        system_prompt = agent_config['rewriter_agent']['description']
        # Gọi __init__ của lớp cha.
        # Toàn bộ logic phức tạp đã nằm ở lớp cha và ToolFactory.
        super().__init__(
            llm=llm, 
            agent_name=agent_name, 
            system_prompt=system_prompt, 
            default_tool_names=default_tool_names or [],
            **kwargs
        )
   
        logger.info(f"'{self.agent_name}' initialized. It will request tools from the factory.")
    
    async def aexecute(self, state):
        query = state.get("rewritten_query") or state.get("original_query", "")
        chat_history = state.get("chat_history", [])
        
        history_messages: List[BaseMessage] = [
            msg for pair in chat_history[-self.history_k:] for msg in (HumanMessage(content=pair[0]), AIMessage(content=pair[1]))
        ]

        # Xây dựng prompt để phản ánh
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "Rewrite the following query to better fit the history and goals of the user: {query}"
            )
        ])
        chain = prompt | self.llm.with_structured_output(RewriterAgentOutput)
        logger.info(f"Invoking LLM for '{self.agent_name}' to reflect on the answer...")
        try:
            result = await chain.ainvoke({
                "query": query,
                "chat_history": history_messages
            })
            logger.info(f"LLM response: {result}")
            # Cập nhật state với câu hỏi đã viết lại
            state["rewritten_query"] = result.rewritten_query
            state["agent_response"] = f"Rewritten query: {result.rewritten_query}"
        except Exception as e:
            logger.error(f"Error invoking LLM in {self.agent_name}: {e}")
            state["agent_response"] = "An error occurred while rewriting the query."    
            state["error_message"] = str(e)
        return state
    # Lớp `Agent` cơ sở đã xử lý tất cả.


if __name__ == "__main__":
    llm = llm_instance
    default_tools = []
    rewriter_agent = RewriterAgent(llm, default_tools)

    state = AgentState(
        original_query="Genetic predisposition to disease",
        chat_history=[("What is the drug?", "The drug is used for...")],
        customer_id="",
        intents=["retrieve"],
        contexts = {
            "drug_retriever": "This drug is used to treat genetic disorders.",
            "genetic_retriever": "Genetic predisposition refers to the increased likelihood of developing a particular disease based on a person's genetic makeup."
        },
        agent_response="The drug is used to treat genetic disorders. Genetic predisposition refers to the increased likelihood of developing a particular disease based on a person's genetic makeup.",
        
        # rewritten_query="Drug mechanism of action"
    )
    result = rewriter_agent.execute(state)
    print(result)