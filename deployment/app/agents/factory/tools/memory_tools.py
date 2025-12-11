import re
import os
import sys
import json
import time
from typing import Optional, TypedDict, Literal, List, Tuple, Dict, Any, Callable, AsyncGenerator, AsyncContextManager
from loguru import logger
import asyncio
from datetime import datetime
from mem0 import Memory
from pathlib import Path
from contextlib import asynccontextmanager

# --- LangChain Core & Community Imports ---
from pydantic import Field as PydanticField
from langchain_core.tools import BaseTool
from pydantic import Field

# --- MCP Imports ---
from mcp.client.sse import sse_client
from mcp.client.session import ClientSession

# Add parent path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))


class MCPMemoryToolBase(BaseTool):
    """Base class for all memory tools with both mem0 and MCP integration."""
    
    name: str = Field(...)
    description: str = Field(...)
    memory_instance: Optional[Memory] = Field(default=None, exclude=True)
    mcp_server_url: str = Field(default="http://localhost:8000/sse", exclude=True)
    use_mcp: bool = Field(default=False, exclude=True)
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, **data):
        super().__init__(**data)
        if not self.use_mcp and self.memory_instance is None:
            self.memory_instance = self._initialize_memory()
    
    def _initialize_memory(self) -> Memory:
        """Initialize mem0 memory with Ollama configuration."""
        os.environ["OPENAI_API_KEY"] = "EMPTY"
        
        config = {
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "collection_name": "agent_memory",
                    "path": "vector_stores_data",
                }
            },
            "llm": {
                "provider": "ollama",
                "config": {
                    "model": "deepseek-r1:1.5b",
                    "temperature": 0.1,
                    "max_tokens": 2000,
                    "ollama_base_url": "http://localhost:11434",
                },
            },
            "embedder": {
                "provider": "ollama",
                "config": {
                    "model": "mxbai-embed-large:latest",
                    "embedding_dims": 1024
                }
            }
        }
        
        return Memory.from_config(config)
    
    @asynccontextmanager
    async def get_mcp_session(self) -> AsyncGenerator[ClientSession, None]:
        """Context manager for MCP sessions with automatic cleanup"""
        session = None
        try:
            async with sse_client(url=self.mcp_server_url) as streams:
                async with ClientSession(read_stream=streams[0], write_stream=streams[1]) as session:
                    await session.initialize()
                    logger.info("MCP session initialized")
                    yield session
        except Exception as e:
            logger.error(f"MCP session error: {e}")
            raise
        finally:
            if session:
                logger.info("MCP session cleaned up")
    
    def _run(self, *args, **kwargs) -> Any:
        """Synchronous tool execution - runs async version in event loop."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, we need to run in a separate thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self._arun(*args, **kwargs))
                    return future.result()
            else:
                return loop.run_until_complete(self._arun(*args, **kwargs))
        except RuntimeError:
            # No event loop, create one
            return asyncio.run(self._arun(*args, **kwargs))
    
    async def _arun(self, *args, **kwargs) -> Any:
        """Asynchronous tool execution - must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _arun method")


class ConversationMemoryTool(MCPMemoryToolBase):
    """Tool for storing and retrieving conversation memories with MCP support."""
    
    name: str = "conversation_memory"
    description: str = """Store and retrieve conversation memories for users and agents.
    Supports both mem0 local storage and MCP server integration."""
    
    async def _arun(
        self,
        action: Literal["store", "retrieve", "search"] = "store",
        messages: Optional[List[Dict[str, str]]] = None,
        user_id: str = "",
        agent_id: str = "",
        session_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        query: Optional[str] = None,
        limit: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute conversation memory operations.
        
        Args:
            action: Type of operation - "store", "retrieve", or "search"
            messages: List of conversation messages for storing
            user_id: User identifier
            agent_id: Agent identifier
            session_id: Session identifier for MCP
            metadata: Additional metadata for the conversation
            query: Search query for finding memories
            limit: Maximum number of results to return
            filters: Additional filters for search
        
        Returns:
            Dict containing operation results
        """
        try:
            if self.use_mcp:
                return await self._handle_mcp_operation(action, messages, user_id, agent_id, session_id, metadata, query, limit, filters)
            else:
                return await self._handle_mem0_operation(action, messages, user_id, agent_id, metadata, query, limit, filters)
                
        except Exception as e:
            logger.error(f"Error in conversation memory tool: {str(e)}")
            return {"error": str(e)}
    
    async def _handle_mcp_operation(
        self,
        action: str,
        messages: Optional[List[Dict[str, str]]],
        user_id: str,
        agent_id: str,
        session_id: str,
        metadata: Optional[Dict[str, Any]],
        query: Optional[str],
        limit: int,
        filters: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Handle MCP-based memory operations."""
        try:
            async with self.get_mcp_session() as session:
                if action == "store":
                    # Convert messages to input/output format for MCP
                    if messages and len(messages) >= 2:
                        # Assume the last message is the output, previous ones are input context
                        input_text = " ".join([msg.get("content", "") for msg in messages[:-1]])
                        output_text = messages[-1].get("content", "")
                        
                        result = await session.call_tool(
                            "memory_saver",
                            arguments={
                                "input": input_text,
                                "output": output_text,
                                "session_id": session_id or f"{user_id}_{agent_id}",
                                "metadata": json.dumps(metadata) if metadata else "{}"
                            }
                        )
                        return {
                            "success": True,
                            "result": result.content[0].text if result.content else "Stored successfully",
                            "session_id": session_id,
                            "user_id": user_id,
                            "agent_id": agent_id
                        }
                    else:
                        return {"error": "Need at least 2 messages for MCP storage"}
                
                elif action in ["retrieve", "search"]:
                    search_query = query or f"user:{user_id} agent:{agent_id}"
                    result = await session.call_tool(
                        "memory_loader",
                        arguments={
                            "query": search_query,
                            "session_id": session_id or f"{user_id}_{agent_id}",
                            "limit": limit
                        }
                    )
                    return {
                        "success": True,
                        "results": result.content[0].text if result.content else "No results found",
                        "query": search_query,
                        "session_id": session_id,
                        "user_id": user_id,
                        "agent_id": agent_id
                    }
                
                else:
                    return {"error": f"Unknown action: {action}"}
                    
        except Exception as e:
            logger.error(f"Error in MCP operation: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _handle_mem0_operation(
        self,
        action: str,
        messages: Optional[List[Dict[str, str]]],
        user_id: str,
        agent_id: str,
        metadata: Optional[Dict[str, Any]],
        query: Optional[str],
        limit: int,
        filters: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Handle mem0-based memory operations."""
        if action == "store":
            return await self._store_conversation_mem0(messages, user_id, agent_id, metadata)
        elif action == "retrieve":
            return await self._retrieve_memories_mem0(user_id, agent_id)
        elif action == "search":
            return await self._search_memories_mem0(query, user_id, agent_id, limit, filters)
        else:
            return {"error": f"Unknown action: {action}"}
    
    async def _store_conversation_mem0(
        self, 
        messages: List[Dict[str, str]], 
        user_id: str, 
        agent_id: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Store a conversation in mem0 memory."""
        if not messages or not user_id or not agent_id:
            return {"error": "Missing required parameters: messages, user_id, agent_id"}
        
        if metadata is None:
            metadata = {}
        
        if "timestamp" not in metadata:
            metadata["timestamp"] = datetime.now().isoformat()
        
        try:
            start_time = time.time()
            result = self.memory_instance.add(
                messages=messages,
                user_id=user_id,
                agent_id=agent_id,
                metadata=metadata,
                infer=False
            )
            processing_time = time.time() - start_time
            
            return {
                "success": True,
                "result": result,
                "processing_time": processing_time,
                "messages_count": len(messages),
                "user_id": user_id,
                "agent_id": agent_id
            }
            
        except Exception as e:
            logger.error(f"Error storing conversation: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _retrieve_memories_mem0(
        self, 
        user_id: str, 
        agent_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Retrieve all memories for a user from mem0."""
        if not user_id:
            return {"error": "user_id is required"}
        
        try:
            start_time = time.time()
            
            if agent_id:
                memories = self.memory_instance.get_all(user_id=user_id, agent_id=agent_id)
            else:
                memories = self.memory_instance.get_all(user_id=user_id)
            
            processing_time = time.time() - start_time
            
            return {
                "success": True,
                "memories": memories,
                "count": len(memories),
                "processing_time": processing_time,
                "user_id": user_id,
                "agent_id": agent_id
            }
            
        except Exception as e:
            logger.error(f"Error retrieving memories: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _search_memories_mem0(
        self,
        query: str,
        user_id: str,
        agent_id: Optional[str] = None,
        limit: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Search memories using semantic search in mem0."""
        if not query or not user_id:
            return {"error": "query and user_id are required"}
        
        try:
            start_time = time.time()
            
            search_params = {
                "query": query,
                "user_id": user_id,
                "limit": limit,
                "threshold": 0.3
            }
            
            if agent_id:
                search_params["agent_id"] = agent_id
            
            if filters:
                search_params["filters"] = filters
            
            results = self.memory_instance.search(**search_params)
            processing_time = time.time() - start_time
            
            if isinstance(results, dict) and "results" in results:
                search_results = results["results"]
            else:
                search_results = results
            
            return {
                "success": True,
                "results": search_results,
                "count": len(search_results),
                "processing_time": processing_time,
                "query": query,
                "user_id": user_id,
                "agent_id": agent_id
            }
            
        except Exception as e:
            logger.error(f"Error searching memories: {str(e)}")
            return {"success": False, "error": str(e)}


class ContextMemoryTool(MCPMemoryToolBase):
    """Tool for managing contextual memories with MCP support."""
    
    name: str = "context_memory"
    description: str = """Store and retrieve contextual facts and preferences about users.
    Supports both mem0 local storage and MCP server integration."""
    
    async def _arun(
        self,
        action: Literal["store_fact", "retrieve_facts", "search_context"] = "store_fact",
        fact: Optional[str] = None,
        user_id: str = "",
        agent_id: str = "",
        session_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        query: Optional[str] = None,
        limit: int = 10,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute context memory operations."""
        try:
            if self.use_mcp:
                return await self._handle_mcp_context_operation(action, fact, user_id, agent_id, session_id, metadata, query, limit)
            else:
                return await self._handle_mem0_context_operation(action, fact, user_id, agent_id, metadata, query, limit)
                
        except Exception as e:
            logger.error(f"Error in context memory tool: {str(e)}")
            return {"error": str(e)}
    
    async def _handle_mcp_context_operation(
        self,
        action: str,
        fact: Optional[str],
        user_id: str,
        agent_id: str,
        session_id: str,
        metadata: Optional[Dict[str, Any]],
        query: Optional[str],
        limit: int
    ) -> Dict[str, Any]:
        """Handle MCP-based context operations."""
        try:
            async with self.get_mcp_session() as session:
                if action == "store_fact":
                    if not fact:
                        return {"error": "fact is required for storage"}
                    
                    result = await session.call_tool(
                        "memory_saver",
                        arguments={
                            "input": f"Context about {user_id}",
                            "output": fact,
                            "session_id": session_id or f"context_{user_id}_{agent_id}",
                            "metadata": json.dumps({**(metadata or {}), "type": "fact"})
                        }
                    )
                    return {
                        "success": True,
                        "result": result.content[0].text if result.content else "Fact stored successfully",
                        "fact": fact,
                        "user_id": user_id,
                        "agent_id": agent_id
                    }
                
                elif action in ["retrieve_facts", "search_context"]:
                    search_query = query or f"context user:{user_id}"
                    result = await session.call_tool(
                        "memory_loader",
                        arguments={
                            "query": search_query,
                            "session_id": session_id or f"context_{user_id}_{agent_id}",
                            "limit": limit
                        }
                    )
                    return {
                        "success": True,
                        "results": result.content[0].text if result.content else "No facts found",
                        "query": search_query,
                        "user_id": user_id,
                        "agent_id": agent_id
                    }
                
        except Exception as e:
            logger.error(f"Error in MCP context operation: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _handle_mem0_context_operation(
        self,
        action: str,
        fact: Optional[str],
        user_id: str,
        agent_id: str,
        metadata: Optional[Dict[str, Any]],
        query: Optional[str],
        limit: int
    ) -> Dict[str, Any]:
        """Handle mem0-based context operations."""
        if action == "store_fact":
            return await self._store_fact_mem0(fact, user_id, agent_id, metadata)
        elif action == "retrieve_facts":
            return await self._retrieve_facts_mem0(user_id, agent_id)
        elif action == "search_context":
            return await self._search_context_mem0(query, user_id, agent_id, limit)
        else:
            return {"error": f"Unknown action: {action}"}
    
    async def _store_fact_mem0(
        self,
        fact: str,
        user_id: str,
        agent_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Store a contextual fact in mem0."""
        if not fact or not user_id or not agent_id:
            return {"error": "Missing required parameters: fact, user_id, agent_id"}
        
        if metadata is None:
            metadata = {"type": "fact", "timestamp": datetime.now().isoformat()}
        
        try:
            result = self.memory_instance.add(
                fact,
                user_id=user_id,
                agent_id=agent_id,
                metadata=metadata,
                infer=False
            )
            
            return {
                "success": True,
                "result": result,
                "fact": fact,
                "user_id": user_id,
                "agent_id": agent_id
            }
            
        except Exception as e:
            logger.error(f"Error storing fact: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _retrieve_facts_mem0(
        self,
        user_id: str,
        agent_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Retrieve all contextual facts from mem0."""
        if not user_id:
            return {"error": "user_id is required"}
        
        try:
            if agent_id:
                facts = self.memory_instance.get_all(user_id=user_id, agent_id=agent_id)
            else:
                facts = self.memory_instance.get_all(user_id=user_id)
            
            return {
                "success": True,
                "facts": facts,
                "count": len(facts),
                "user_id": user_id,
                "agent_id": agent_id
            }
            
        except Exception as e:
            logger.error(f"Error retrieving facts: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _search_context_mem0(
        self,
        query: str,
        user_id: str,
        agent_id: Optional[str] = None,
        limit: int = 10
    ) -> Dict[str, Any]:
        """Search for relevant contextual information in mem0."""
        if not query or not user_id:
            return {"error": "query and user_id are required"}
        
        try:
            search_params = {
                "query": query,
                "user_id": user_id,
                "limit": limit,
                "threshold": 0.2
            }
            
            if agent_id:
                search_params["agent_id"] = agent_id
            
            results = self.memory_instance.search(**search_params)
            
            if isinstance(results, dict) and "results" in results:
                search_results = results["results"]
            else:
                search_results = results
            
            return {
                "success": True,
                "results": search_results,
                "count": len(search_results),
                "query": query,
                "user_id": user_id,
                "agent_id": agent_id
            }
            
        except Exception as e:
            logger.error(f"Error searching context: {str(e)}")
            return {"success": False, "error": str(e)}


# Usage example and factory functions
def create_mcp_conversation_tool(mcp_server_url: str = "http://localhost:8000/sse") -> ConversationMemoryTool:
    """Create a ConversationMemoryTool configured for MCP."""
    return ConversationMemoryTool(use_mcp=True, mcp_server_url=mcp_server_url)

def create_mem0_conversation_tool() -> ConversationMemoryTool:
    """Create a ConversationMemoryTool configured for mem0."""
    return ConversationMemoryTool(use_mcp=False)

def create_mcp_context_tool(mcp_server_url: str = "http://localhost:8000/sse") -> ContextMemoryTool:
    """Create a ContextMemoryTool configured for MCP."""
    return ContextMemoryTool(use_mcp=True, mcp_server_url=mcp_server_url)

def create_mem0_context_tool() -> ContextMemoryTool:
    """Create a ContextMemoryTool configured for mem0."""
    return ContextMemoryTool(use_mcp=False)


# Example usage
async def example_usage():
    """Example of how to use the integrated memory tools."""
    
    # Create MCP-based tools
    mcp_conv_tool = create_mcp_conversation_tool("http://localhost:8000/sse")
    mcp_context_tool = create_mcp_context_tool("http://localhost:8000/sse")
    
    # Store a conversation via MCP
    messages = [
        {"role": "user", "content": "What is machine learning?"},
        {"role": "assistant", "content": "Machine learning is a subset of AI that enables computers to learn from data."}
    ]
    
    result = await mcp_conv_tool._arun(
        action="store",
        messages=messages,
        user_id="user123",
        agent_id="agent456",
        session_id="session789"
    )
    print("MCP Store Result:", result)
    
    # Search memories via MCP
    search_result = await mcp_conv_tool._arun(
        action="search",
        query="machine learning",
        user_id="user123",
        agent_id="agent456",
        session_id="session789"
    )
    print("MCP Search Result:", search_result)
    
    # Store a fact via MCP
    fact_result = await mcp_context_tool._arun(
        action="store_fact",
        fact="User prefers technical explanations with examples",
        user_id="user123",
        agent_id="agent456",
        session_id="context_session"
    )
    print("MCP Fact Store Result:", fact_result)


if __name__ == "__main__":
    asyncio.run(example_usage())