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
from pydantic import BaseModel, Field as PydanticField
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
from app.agents.retrievers.genetic_retriever import GeneticRetrieverTool
# --- Load Environment Variables ---



class GeneticAgent(Agent):
    def __init__(self, llm: BaseChatModel, default_tools: List[BaseAgentTool], vectorstore: Optional[Chroma] = None):
        # logger.debug("Company Agent Description: {}".format(agent_config['company_agent']['description']))
        system_msg = agent_config['genetic_agent']['description']
        super().__init__(llm, "GeneticAgent", system_msg, default_tools, vectorstore)

        self.retriever = GeneticRetrieverTool(collection_name="genetic_docs", watch_directory='app/agents/retrievers/storages/genetics')
        self.vectorstore = self.retriever._vector_store
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
        return super().execute(state)
    
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
        partial_state["contexts"] = contexts
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
            
        return {**state, **partial_state}  # Merge the original state with the partial state
    
    
    
    

if __name__ == "__main__":
    llm = llm_instance
    default_tools = [GeneticRetrieverTool()]
    genetic_agent = GeneticAgent(llm, default_tools)

    state = GraphState(
        original_query="What is the genetic marker associated with this condition?",
        chat_history=[("What is the condition?", "The condition is characterized by...")],
        customer_id="789122254025",
        intents=["retrieve"],
        # rewritten_query="Genetic marker of action"
    )
    result = genetic_agent.execute(state)
    print(result)