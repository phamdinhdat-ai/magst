

from typing import Any, Optional, Dict
import chromadb
from langchain_chroma import Chroma
from loguru import logger
import re 
from app.agents.factory.mcp_retriever_factory import MCPRetrieverFactory, MCP_RETRIEVER_MANAGER


class AgentFactory:
    """
    Enhanced agent factory with integrated MCP retriever support.
    Provides centralized configuration and creation of all agent tools.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.mcp_config = config.get('mcp', {})
        self.mcp_factory = MCPRetrieverFactory()
        self.mcp_manager = MCP_RETRIEVER_MANAGER
        
    def create_mcp_retriever(self, retriever_type: str, **override_kwargs) -> Any:
        """Create MCP retriever with factory configuration"""
        
        # Get MCP server configuration
        mcp_server_url = self.mcp_config.get('server_url', 'http://localhost:50051/sse')
        
        # Get retriever-specific configuration
        retriever_config = self.mcp_config.get('retrievers', {}).get(retriever_type, {})
        
        # Merge configurations
        kwargs = {
            'mcp_server_url': mcp_server_url,
            **retriever_config,
            **override_kwargs  # Allow runtime overrides
        }
        
        return self.mcp_factory.create_retriever(retriever_type, **kwargs)
    
    def create_and_register_mcp_retriever(self, name: str, retriever_type: str, **override_kwargs) -> Any:
        """Create and register an MCP retriever with the manager."""
        
        # Get MCP server configuration
        mcp_server_url = self.mcp_config.get('server_url', 'http://localhost:50051/sse')
        
        # Get retriever-specific configuration
        retriever_config = self.mcp_config.get('retrievers', {}).get(retriever_type, {})
        
        # Merge configurations
        kwargs = {
            'mcp_server_url': mcp_server_url,
            **retriever_config,
            **override_kwargs
        }
        
        return self.mcp_manager.create_and_register(name, retriever_type, **kwargs)
    
    def create_tools(self) -> Dict[str, Any]:
        """Create all configured tools including MCP retrievers"""
        tools = {}

        mcp_retrievers = self.config.get('mcp', {}).get('retrievers', {})
        
        for retriever_name, retriever_config in mcp_retrievers.items():
            if retriever_config.get('enabled', False):
                try:
                    # Register with manager for lifecycle management
                    tool = self.create_and_register_mcp_retriever(
                        name=retriever_name,
                        retriever_type=retriever_config['type'],
                        **retriever_config.get('params', {})
                    )
                    if tool:
                        tools[f"{retriever_name}_mcp"] = tool
                        logger.info(f"Created and registered MCP retriever: {retriever_name}")
                except Exception as e:
                    logger.error(f"Failed to create MCP retriever {retriever_name}: {e}")
        
        return tools
    
    def get_available_mcp_types(self) -> Dict[str, str]:
        """Get available MCP retriever types."""
        return self.mcp_factory.get_available_types()
    
    def create_batch_mcp_retrievers(self, retriever_specs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Create multiple MCP retrievers at once."""
        return self.mcp_factory.create_batch_retrievers(retriever_specs)
    
    def cleanup_mcp_resources(self):
        """Cleanup all MCP resources."""
        self.mcp_manager.cleanup_all()
        logger.info("All MCP resources cleaned up")