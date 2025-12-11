from datetime import datetime
from zoneinfo import ZoneInfo

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

def get_time(timezone: str, include_date: bool = True) -> str:

    try:
        now = datetime.now(ZoneInfo(timezone))
        if include_date:
            return f"Thông tin thời gian hiện tại ở Việt Nam: {now.strftime('%Y-%m-%d %H:%M:%S')}"
        else:
            return f"Thông tin thời gian hiện tại ở Việt Nam: {now.strftime('%H:%M')}"
    except Exception as e:
        return f"Error: {e}"


class GetDateTimeTool(BaseAgentTool):
    """A tool to get the current date and time in a specified timezone."""
    
    timezone: str = Field(
        default="Asia/Ho_Chi_Minh",
        description="The IANA timezone identifier (e.g. 'Asia/Ho_Chi_Minh')."
    )
    include_date: bool = Field(
        default=False,
        description="If True, returns full datetime. Otherwise, just time."
    )
    
    def __init__(self, **kwargs):
        name = "GetDateTime"
        description = "This tool returns the current date and time in a specified timezone."
        super().__init__(name=name, description=description, **kwargs)
    
    def _run(self) -> str:
        """Synchronous run method."""
        return get_time(self.timezone, self.include_date)
    
    async def _arun(self, query: str) -> str:
        """Asynchronous run method."""
        return await asyncio.to_thread(get_time, self.timezone, self.include_date)