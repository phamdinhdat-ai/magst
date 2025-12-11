import re
import os
import sys
import json
import time
import chromadb
from typing import Optional, TypedDict, Literal, List, Tuple, Dict, Any, Callable
from loguru import logger
import asyncio
from pydantic import Field

from langchain_tavily import TavilySearch
import re
# --- Tool Imports ---
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from app.agents.factory.tools.base import BaseAgentTool
from app.agents.factory.tools.get_datetime_tool import get_time
from langchain_community.utilities import GoogleSerperAPIWrapper

os.environ["SERPER_API_KEY"] = "4cd9269c26dfebf5c28267585779d8a63a534bee"
search = GoogleSerperAPIWrapper(hl='vie' , gl='vn')
# result = search.run("Genestory Company")
# To install: pip install tavily-python
os.environ["TAVILY_API_KEY"] = "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"
from tavily import TavilyClient



class SearchTool(BaseAgentTool):
    # Define Pydantic fields
    n_results: int = Field(default=3, description="Number of search results to return")
    search_engine: TavilyClient = Field(default=None, description="Tavily search engine instance")
    serper_engine: GoogleSerperAPIWrapper = Field(default=None, description="Google Serper search engine instance")
    
    def __init__(self, n_results: int = 3, **kwargs):
        # Initialize tool with proper name and description
        name = "WebSearch"
        description = "This is a web search tool. It searches the web for the latest information."
        
        super().__init__(
            name=name, 
            description=description,
            n_results=n_results,
            **kwargs
        )
        
        # Initialize search engines after parent initialization
        # self.search_engine = TavilySearch(
        #     max_results=self.n_results,
        #     topic="general",
        #     search_depth="advanced",
        # )
        self.search_engine = TavilyClient()
        self.serper_engine = GoogleSerperAPIWrapper(hl='vie', gl='vn', k=5)
    
    def _run(self, query: str) -> List[str]:
        """Synchronous search execution for LangChain compatibility."""
        date_time = get_time("Asia/Ho_Chi_Minh", include_date=True).replace("Thông tin thời gian hiện tại Việt Nam: ", "")
        return self.search_serper_sync(query + f" {date_time}")

    async def _arun(self, query: str) -> List[str]:
        """Asynchronous search execution for LangChain compatibility."""
        date_time = get_time("Asia/Ho_Chi_Minh", include_date=True).replace("Thông tin thời gian hiện tại Việt Nam: ", "")
        return self.tavily_search(query + f" {date_time}")
    
    async def arun_impl(self, query: str) -> List[str]:
        # Implement the asynchronous search logic here
        try:
            search_results = await self.search_serper(query)
            if not search_results:
                return ["No results found."]

            logger.info(f"Search results: {len(search_results)}")
            return search_results
        except Exception as e:
            return [f"Error: {e}"]
        
    def tavily_search(self, query: str) -> List[str]:
        """Perform a search using Tavily."""
        try:
            results = self.search_engine.search(
                    query=query,
                    max_results=self.n_results,
                    chunks_per_source=4,
                    country="vietnam",
                    include_domains=["https://genestory.ai/home/","https://vnexpress.net/","https://www.vinmec.com/vie/", "https://www.accuweather.com/"],
                    exclude_domains=[""]
                )
            if not results:
                return ["No results found."]
            output_search = []
            for result in results['results']:
                title = result.get("title", "No title")
                url = result.get("url", "No URL")
                content = result.get("content", "No content")
                output_search.append(f"Title: {title}\nURL: {url}\nContent: {content}")
            return output_search
        except Exception as e:
            return [f"Error: {e}"]
        
    async def search_serper(self, query: str) -> List[str]:
        try:
            search_results = await self.serper_engine.aresults(query)
            results = search_results.get("organic", [])
            if not results:
                return ["No results found."]
            output_search = []
            for result in results:
                logger.info(f"Serper Search Result: {result}")
                logger.info(f"Serper Search Result type: {type(result)}")
                title = result.get("title", "No title")
                url = result.get("link", "No URL")
                content = result.get("snippet", "No content")
                output_search.append(f"Title: {title}\nURL: {url}\nContent: {content}")
                logger.info(f"Serper Search Result: {title} - {url} - {content}")
            return output_search
        except Exception as e:
            return [f"Error: {e}"]
    
    def search_serper_sync(self, query: str) -> List[str]:
        try:
            search_results = self.serper_engine.results(query)
            results = search_results.get("organic", [])
            if not results:
                return ["No results found."]
            output_search = []
            for result in results:
                title = result.get("title", "No title")
                url = result.get("link", "No URL")
                content = result.get("snippet", "No content")
                output_search.append(f"Title: {title}\nURL: {url}\nContent: {content}")
            return output_search
        except Exception as e:
            return [f"Error: {e}"]