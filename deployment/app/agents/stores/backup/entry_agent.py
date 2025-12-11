import os 
import sys
import re
import os
import sys
import json
import time
import chromadb
from typing import Optional, TypedDict, Literal, List, Tuple, Dict, Any, Callable
from loguru import logger
import asyncio
from typing import List
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
# --- LangChain Core & Community Imports ---
from langchain_community.chat_models.ollama import ChatOllama
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool, BaseTool
from app.agents.factory.tools.base import BaseAgentTool
from pydantic import BaseModel, Field
from langchain_tavily import TavilySearch
import re
# --- Tool Imports ---
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from dotenv import load_dotenv
from app.agents.workflow.state import GraphState
from app.agents.workflow.initalize import llm_instance, settings, agent_config
from app.agents.stores.base_agent import BaseAgentNode, Agent
from app.agents.factory.tools.search_tool import SearchTool
from app.agents.factory.tools.summary_tool import SummaryTool
# --- Load Environment Variables ---

class EntryAgentOutput(BaseModel):
    intents: List[str] = Field(..., description="Danh sách các ý định của người dùng được xác định từ truy vấn.")
    classified_agent: str = Field(..., description="Tên của tác nhân đã được xác định để xử lý truy vấn.")
    needs_rewrite: bool = Field(..., description="Biến xác định xem truy vấn có cần được viết lại hay không.")
    

class EntryAgent(Agent):
    def __init__(self, llm: BaseChatModel, default_tools: List[BaseAgentTool], vectorstore: Optional[Chroma] = None):
        # logger.debug("Company Agent Description: {}".format(agent_config['company_agent']['description']))
        system_msg = agent_config['entry_agent']['description']
        super().__init__(llm, "EntryAgent", system_msg, default_tools, vectorstore)

    def get_dynamic_tools(self, state: GraphState) -> List[BaseTool]:
        tools = super().get_dynamic_tools(state)
        if self.retriever:
            tools.append(self.retriever)
        return tools
    
    def get_tool_contexts(self, state: GraphState) -> Dict[str, Any]:
        """Get the contexts for the tools available to this agent."""
        tool_contexts = super().get_tool_contexts(state)
        return tool_contexts
    
    def _prepare_execution(self, state: GraphState, critical_error_check: str = "Query processing failed") -> Dict[str, Any]:
        return super()._prepare_execution(state, critical_error_check=critical_error_check)
    
    
    def execute(self, state):
        partial_state = self._prepare_execution(state, critical_error_check="Query processing failed")
        if not isinstance(partial_state, dict):
            return {**state, "error_message": "Failed to prepare execution state."}  # Skip execution if critical error check fails

        query = state.get('original_query','    ')
        chat_history = state.get("chat_history", [])
        history_messages = [HumanMessage(content=q) if i % 2 == 0 else AIMessage(content=a) for i, (q, a) in enumerate(chat_history[-3:])]
        
        logger.info(f"Chat history for {self.agent_name}: {history_messages}")
        

        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("human", "Truy vấn của người dùng: {query}"),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("human", "Hay xác định các ý định của người dùng từ truy vấn này và phân loại tác nhân phù hợp để xử lý truy vấn. "
                          "- Nếu truy vấn cần được viết lại, hãy đặt needs_rewrite thành True và cung cấp câu hỏi đã được viết lại trong trường rewritten_query.\n"
                          "- Nếu không cần viết lại, hãy đặt needs_rewrite thành False và để trường rewritten_query trống.\n"
                          "- Trả lời dưới dạng JSON với các trường intents, classified_agent, needs_rewrite")
            ]
        )
        chain = prompt_template | self.llm.with_structured_output(EntryAgentOutput)
        logger.debug(f"Messages for {self.agent_name}: {query}")
        try:
            
            response = chain.invoke({
                "query": query,
                "chat_history": history_messages,
            })
            logger.info(f"LLM response for {self.agent_name}: {response}")
            response_dict = response.model_dump()
            partial_state.update(response_dict)
            return {**state, **partial_state}
            
        except Exception as e:
            logger.error(f"Error during LLM execution for {self.agent_name}: {e}")
            partial_state = self._handle_execution_error(e, partial_state)
            partial_state["error_message"] = str(e)  # Add error message to the state
            partial_state["intents"] = []  # Ensure intents is always a list
            partial_state["classified_agent"] = "NaiveAgent"  # Default agent if classification fails
            partial_state["needs_rewrite"] = False  # Default rewrite flag
        if "rewritten_query" not in partial_state:
            partial_state["rewritten_query"] = query
        return {**state, **partial_state}

    async def async_execute(self, state: GraphState) -> Dict[str, Any]:
        """Asynchronously execute the agent's logic."""
        partial_state = self._prepare_execution(state, critical_error_check="Query processing failed")
        if not isinstance(partial_state, dict):
            return {}
        
        query = state.get("rewritten_query", state.get("original_query", ""))
        chat_history = state.get("chat_history", [])
        history_messages = [HumanMessage(content=q) if i % 2 == 0 else AIMessage(content=a) for i, (q, a) in enumerate(chat_history[-3:])]
        
        logger.info(f"Chat history for {self.agent_name}: {history_messages}")
        contexts = self.get_tool_contexts(state)
        
        # Prepare the prompt template
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                {"human": "Truy vấn của người dùng: {query}"},
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("human", "Dưới đây là các kết quả từ các công cụ đã truy xuất dựa trên câu hỏi của người dùng. "
                          "- Ưu tiên: Tài liệu công ty (nguồn chính thống và đáng tin cậy nhất)\n"
                          "- Thứ hai: Tài liệu từ các nguồn khác (nếu có)\n"
                          "- Trả lời ngắn gọn, súc tích, không dài dòng."),
                MessagesPlaceholder(variable_name="contexts", optional=True),
                ("human", "Dựa trên các kết quả trên, hãy trả lời câu hỏi của người dùng một cách ngắn gọn và súc tích nhất có thể."),
            ]
        )
        # Prepare the messages for the prompt
        messages = prompt_template.format_messages(
            query=query,
            chat_history=history_messages,
            contexts=contexts
        )
        # Log the messages for debugging
        logger.debug(f"Messages for {self.agent_name}: {messages}")
        # Execute the LLM with the prepared messages
        try:
            response = await self.llm.ainvoke(messages)
            logger.info(f"LLM response for {self.agent_name}: {response}")
            partial_state["response"] = response.content
        except Exception as e:
            logger.error(f"Error during LLM execution for {self.agent_name}: {e}")
            partial_state = self._handle_execution_error(e, partial_state)
        
        return partial_state
    
    
    
    

if __name__ == "__main__":
    llm = llm_instance
    default_tools = []
    entry_agent = EntryAgent(llm, default_tools)

    state = GraphState(
        original_query="Tôi muốn xem báo cáo gen của mình",
        chat_history=[("Paracetamol là gì?", "Thuốc này được sử dụng để...")],
        customer_id="",
        intents=["retrieve"],
        contexts = {
            "drug_retriever": "Thuốc này được sử dụng để điều trị các rối loạn di truyền.",
            "genetic_retriever": "Xu hướng di truyền đề cập đến khả năng cao hơn trong việc phát triển một bệnh cụ thể dựa trên cấu trúc di truyền của một người."
        },
        agent_response="Thuốc này được sử dụng để điều trị các rối loạn di truyền. Xu hướng di truyền đề cập đến khả năng cao hơn trong việc phát triển một bệnh cụ thể dựa trên cấu trúc di truyền của một người.",
        
        # rewritten_query="Drug mechanism of action"
    )
    result = entry_agent.execute(state)
    print(result)