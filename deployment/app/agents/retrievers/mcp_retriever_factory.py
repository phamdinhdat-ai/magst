
from typing import Dict, Any, Optional
from pathlib import Path

from app.agents.retrievers.company_mcp_retriever import CompanyRetrieverMCPClient, create_company_retriever_client
from app.agents.retrievers.drug_mcp_retriever import DrugRetrieverMCPClient, create_drug_retriever_client

class MCPRetrieverFactory:
    """Factory for creating MCP retriever clients"""
    
    @staticmethod
    def create_company_retriever(
        mcp_server_url: str,
        watch_directory: str,
        collection_name: str = "company_docs",
        **kwargs
    ) -> CompanyRetrieverMCPClient:
        """Create a company document retriever MCP client"""
        return create_company_retriever_client(
            mcp_server_url=mcp_server_url,
            watch_directory=watch_directory,
            collection_name=collection_name
        )
    
    @staticmethod
    def create_drug_retriever(
        mcp_server_url: str,
        watch_directory: str,
        collection_name: str = "drug_docs",
        **kwargs
    ) -> DrugRetrieverMCPClient:
        """Create a drug document retriever MCP client"""
        return create_drug_retriever_client(
            mcp_server_url=mcp_server_url,
            watch_directory=watch_directory,
            collection_name=collection_name
        )
    
    @staticmethod
    def create_retriever(
        retriever_type: str,
        mcp_server_url: str,
        watch_directory: str,
        collection_name: str = None,
        **kwargs
    ):
        """Generic retriever factory method"""
        
        retriever_map = {
            "company": MCPRetrieverFactory.create_company_retriever,
            "drug": MCPRetrieverFactory.create_drug_retriever,
        }
        
        if retriever_type not in retriever_map:
            raise ValueError(f"Unknown retriever type: {retriever_type}. Available: {list(retriever_map.keys())}")
        
        # Set default collection names if not provided
        default_collections = {
            "company": "company_docs",
            "drug": "drug_docs"
        }
        
        if collection_name is None:
            collection_name = default_collections.get(retriever_type, f"{retriever_type}_docs")
        
        return retriever_map[retriever_type](
            mcp_server_url=mcp_server_url,
            watch_directory=watch_directory,
            collection_name=collection_name,
            **kwargs
        )