from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union

class VectorSearchResult(BaseModel):
    """Schema for vector search results"""
    text: str = Field(..., description="Text content of the chunk")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata about the chunk")
    similarity: float = Field(..., description="Similarity score to query")

class VectorSearchRequest(BaseModel):
    """Schema for vector search requests"""
    query: str = Field(..., description="Query text to search for")
    owner_type: Optional[str] = Field(None, description="Optional filter by owner type")
    owner_id: Optional[int] = Field(None, description="Optional filter by owner ID")
    document_id: Optional[int] = Field(None, description="Optional filter by document ID")
    limit: int = Field(5, description="Maximum number of results to return")

class VectorSearchResponse(BaseModel):
    """Schema for vector search responses"""
    query: str = Field(..., description="Original query")
    results: List[VectorSearchResult] = Field(default_factory=list, description="Search results")
    total_results: int = Field(0, description="Total number of results")
