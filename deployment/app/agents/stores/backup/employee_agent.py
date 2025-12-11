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
from app.agents.retrievers.employee_retriever import EmployeeRetrieverTool

# --- Load Environment Variables ---
load_dotenv()


class EmployeeAgent(Agent):
    def __init__(self, llm: BaseChatModel, default_tools: List[BaseAgentTool], vectorstore: Optional[Chroma] = None):
        # logger.debug("Company Agent Description: {}".format(agent_config['company_agent']['description']))
        system_msg = agent_config['employee_agent']['description']
        super().__init__(llm, "EmployeeAgent", system_msg, default_tools, vectorstore)
        self.retriever = None
        self.vectorstore = vectorstore
        
    def get_dynamic_tools(self, state: GraphState) -> List[BaseTool]:
        self.retriever = EmployeeRetrieverTool(employee_id=state['employee_id'], watch_directory='app/agents/retrievers/storages/employees')
        self.vectorstore = self.retriever.vector_store
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
    
    


if __name__ == "__main__":
    
    
    llm = llm_instance
    default_tools = [SearchTool()]
    employee_agent = EmployeeAgent(llm, default_tools)
    # Example state for the employee agent
    state = GraphState(
        original_query="What is the employee's mission?",
        chat_history=[("What is our mission?", "Our mission is to...")],
        employee_id="12345",
        # rewritten_query="Company mission"
    )
    result = employee_agent.execute(state)
    print(result)