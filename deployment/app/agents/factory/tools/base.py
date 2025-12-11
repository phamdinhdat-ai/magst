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
from pydantic import  Field as PydanticField
import re
# --- Tool Imports ---
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from abc import ABC, abstractmethod
from typing import List, Any, Optional
from langchain_core.tools import BaseTool
from pydantic import Field


class BaseAgentTool(BaseTool):
    """Base class for all agent tools with proper LangChain compatibility."""
    
    name: str = Field(...)
    description: str = Field(...)
    
    class Config:
        arbitrary_types_allowed = True
    
    @abstractmethod
    def _run(self, *args, **kwargs) -> Any:
        """Synchronous tool execution - must be implemented by subclasses."""
        pass
    
    @abstractmethod
    async def _arun(self, *args, **kwargs) -> Any:
        """Asynchronous tool execution - must be implemented by subclasses."""
        pass
    
    def cleanup(self) -> None:
        """Optional cleanup method to be called after tool execution."""
        # Default implementation does nothing, can be overridden by subclasses
        pass
    # Provide run/arun methods for backward compatibility
    def run(self, *args, **kwargs) -> Any:
        """Public synchronous execution method."""
        return self._run(*args, **kwargs)
    
    async def arun(self, *args, **kwargs) -> Any:
        """Public asynchronous execution method."""
        return await self._arun(*args, **kwargs)

