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



class RewriterAgentOutput(BaseModel):
    rewritten_query: str = Field(..., description="Câu hỏi đã được viết lại để phù hợp với ngữ cảnh và mục tiêu của người dùng.")
    


class RewriterAgent(Agent):
    def __init__(self, llm: BaseChatModel, default_tools: List[BaseAgentTool], vectorstore: Optional[Chroma] = None):
        # logger.debug("Company Agent Description: {}".format(agent_config['company_agent']['description']))
        system_msg = agent_config['rewrite_agent']['description']
        super().__init__(llm, "RewriterAgent", system_msg, default_tools, vectorstore)

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
            return {}
        
        
        query = state.get("rewritten_query", state.get("original_query", ""))
        chat_history = state.get("chat_history", [])
        history_messages = [HumanMessage(content=q) if i % 2 == 0 else AIMessage(content=a) for i, (q, a) in enumerate(chat_history[-3:])]
        
        logger.info(f"Chat history for {self.agent_name}: {history_messages}")
        
        
        
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("human", "Truy vấn của người dùng: {query}"),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("human", "Hay viet lại câu hỏi của người dùng một cách ngắn gọn và súc tích nhất có thể, "),
            ]
        )
        chain = prompt_template | self.llm.with_structured_output(RewriterAgentOutput)
        logger.debug(f"Messages for {self.agent_name}: {query}")
        try:
            current_partial_state = self._prepare_execution(state, critical_error_check="critical")
            if not current_partial_state:
                return {**state, "error_message": "Failed to prepare execution state."}  # Skip execution if critical error check fails
            
            response = chain.invoke({
                "query": query,
                "chat_history": history_messages,
            })
            logger.info(f"LLM response for {self.agent_name}: {response}")
            response_dict = response.model_dump()
            partial_state["rewritten_query"] = response_dict.get("rewritten_query", "")
        except Exception as e:
            logger.error(f"Error during LLM execution for {self.agent_name}: {e}")
            partial_state = self._handle_execution_error(e, partial_state)
        if "rewritten_query" not in partial_state:
            partial_state["rewritten_query"] = query
        return partial_state
    
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
    rewriter_agent = RewriterAgent(llm, default_tools)

    state = GraphState(
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