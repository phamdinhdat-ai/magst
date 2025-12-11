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
from app.agents.factory.tools.search_tool import SearchTool
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
from app.agents.retrievers.customer_retriever import CustomerRetrieverTool

# --- Load Environment Variables ---
load_dotenv()


class CustomerAgent(Agent):
    def __init__(self, llm: BaseChatModel, default_tools: List[BaseAgentTool], vectorstore: Optional[Chroma] = None):
        # logger.debug("Company Agent Description: {}".format(agent_config['company_agent']['description']))
        system_msg = agent_config['customer_agent']['description']
        super().__init__(llm, "CompanyAgent", system_msg, default_tools, vectorstore)
        self.retriever = None
        self.vectorstore = vectorstore
        
    def get_dynamic_tools(self, state: GraphState) -> List[BaseTool]:
        self.retriever = CustomerRetrieverTool(customer_id=state['customer_id'], watch_directory='app/agents/retrievers/storages/customers')
        self.vectorstore = self.retriever._vector_store
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
        self.system_prompt = self.system_prompt.format(
            customer_id=state['customer_id'],
            db_customer_id=state['customer_id'])
        return super().execute(state)
    
    


if __name__ == "__main__":
    
    
    llm = llm_instance
    default_tools = [SearchTool()]
    customer_agent = CustomerAgent(llm, default_tools)
    
    state = GraphState(
        original_query="What is the customer's mission?",
        chat_history=[("What is our mission?", "Our mission is to...")],
        customer_id="789122254025",
        # rewritten_query="Company mission"
    )
    result = customer_agent.execute(state)
    print(result)