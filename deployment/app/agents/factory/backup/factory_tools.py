# --- Tool Factory Class ---
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
from pydantic import BaseModel, Field as PydanticField
from langchain_tavily import TavilySearch
import re
# --- Tool Imports ---
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from app.services.database_tools import PostgresProductTool
from app.services.database_tools import postgres_product_db
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from app.agents.tools.base import BaseAgentTool
from app.agents.tools.sql_query_tool import SQLDatabaseTool
from app.agents.tools.search_tool import SearchTool 
from app.agents.tools.summary_tool import SummaryTool
from app.agents.retrievers.product_retriever import ProductRetrieverTool
from app.agents.base import extract_clean_json, GraphState, agent_config, llm_instance

        
class ToolFactory:
    def __init__(self, state: GraphState):
        self.tools = {
            "product": ProductRetrieverTool(watch_directory='app/agents/retrievers/storages/products'),
            "searchweb": SearchTool(n_results=3),
            "summary": SummaryTool(llm=llm_instance),
        }
        self.state = state
        
    def get_tool(self, tool_name: str) -> Optional[BaseAgentTool]:
        return self.tools.get(tool_name, None)

    def add_tool(self, tool_name: str, tool: BaseAgentTool) -> None:
        self.tools[tool_name] = tool
        
    def update_tool(self, tool_name: str, new_tool: BaseAgentTool) -> None:
        if tool_name in self.tools:
            self.tools[tool_name] = new_tool
        else:
            raise ValueError(f"Tool '{tool_name}' not found in the factory.")
    
 