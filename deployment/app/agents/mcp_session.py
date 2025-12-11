
import asyncio
from mcp.client.sse import sse_client
from mcp.client.session import ClientSession   
import time
from typing import AsyncGenerator, AsyncContextManager
from contextlib import asynccontextmanager
from loguru import logger 
from app.core.config import settings
@asynccontextmanager
async def get_session():
    """Context manager for MCP sessions with automatic cleanup"""
    session = None
    active_sessions = []
    try:
        async with sse_client(url=settings.MCP_SERVER_URL) as streams:
            async with ClientSession(read_stream=streams[0], write_stream=streams[1]) as session:
                await session.initialize()
                active_sessions.append(session)
                logger.info(f"Session initialized. Active sessions: {len(active_sessions)}")
                yield session
    except Exception as e:
        logger.error(f"Session error: {e}")
        raise
    finally:
        if session and session in active_sessions:
            active_sessions.remove(session)
            logger.info(f"Session cleaned up. Active sessions: {len(active_sessions)}")



