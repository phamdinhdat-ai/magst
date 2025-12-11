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
from app.services.database_tools import PostgresProductTool
from app.services.database_tools import postgres_product_db
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from app.agents.tools.base import BaseAgentTool


class SQLDatabaseTool(BaseAgentTool):
    name: str = "ProductDatabaseSQLQuery"
    description: str = "Executes a SQL SELECT query against the product database."
    db: PostgresProductTool
    
    def __init__(self, db: PostgresProductTool):
        super().__init__(db=db)
    
    def _run(self, sql_query: str) -> List[str]:
        """Synchronous SQL query execution."""
        return self.run_sql_query(sql_query)

    async def _arun(self, sql_query: str) -> List[str]:
        """Asynchronous SQL query execution."""
        return self.run_sql_query(sql_query)

    def run_sql_query(self, sql_query: str) -> List[str]:
        """Execute SQL query and return results."""
        if not sql_query.strip().upper().startswith("SELECT"):
            return ["Error: Only SELECT queries allowed."]
        try:
            result = self.db.run(sql_query)
            if not result or result == "No results found.":
                return ["No results found."]
            
            return result if isinstance(result, list) else [str(result)]
        except Exception as e:
            return [f"Error: {e}"]