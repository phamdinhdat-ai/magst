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
# --- Simplified Drug Retriever Tool ---
import os
import sys
import time
from typing import Optional, List, Dict, Any
from loguru import logger
from pydantic import Field, BaseModel
import numpy as np

class RetrievedDocument(BaseModel):
    """Model for a retrieved document with enhanced metadata for reranking"""
    content: str
    source: str
    retrieval_score: float
    relevance_score: float = 0.0
    drug_name: str = ""
    category: str = ""
    _content_hash: Optional[str] = None
    _processed_tokens: Optional[List[str]] = None
    
    def calculate_advanced_relevance(self, query: str) -> float:
        """Enhanced relevance scoring with multiple factors"""
        query_lower = query.lower()
        content_lower = self.content.lower()
        
        # 1. Exact phrase matching (highest weight)
        exact_match_score = 0.0
        if query_lower in content_lower:
            exact_match_score = 1.0
        
        # 2. Drug name matching
        drug_match_score = 0.0
        query_terms = set(query_lower.split())
        if self.drug_name and any(term in self.drug_name.lower() for term in query_terms):
            drug_match_score = 0.8
        
        # 3. Term overlap with position weighting
        query_terms = query_lower.split()
        content_terms = content_lower.split()
        overlap_score = 0.0
        
        for q_term in query_terms:
            for i, c_term in enumerate(content_terms):
                if q_term in c_term or c_term in q_term:
                    # Earlier positions get higher weight
                    position_weight = 1.0 / (i + 1) if i < 20 else 0.1
                    overlap_score += position_weight
        
        overlap_score = min(overlap_score / max(1, len(query_terms)), 1.0)
        
        # 4. Content length penalty (prefer concise, relevant content)
        length_penalty = 1.0 - min(len(self.content) / 1000, 0.3)
        
        # 5. Source reliability score
        source_score = 0.9 if 'CPIC' in self.source else 0.7 if 'FDA' in self.source else 0.5
        
        # Combine all scores with weights
        self.relevance_score = (
            exact_match_score * 0.3 +
            drug_match_score * 0.25 +
            overlap_score * 0.2 +
            self.retrieval_score * 0.15 +
            length_penalty * 0.05 +
            source_score * 0.05
        )
        
        return self.relevance_score
    
    def get_content_hash(self) -> str:
        """Get cached content hash for deduplication"""
        if self._content_hash is None:
            import hashlib
            self._content_hash = hashlib.md5(self.content.encode()).hexdigest()
        return self._content_hash
    
    def get_processed_tokens(self) -> List[str]:
        """Get cached processed tokens for faster text analysis"""
        if self._processed_tokens is None:
            # Simple tokenization - can be enhanced with better tokenizers
            self._processed_tokens = re.findall(r'\b\w+\b', self.content.lower())
        return self._processed_tokens
    
    def calculate_fast_relevance(self, query_tokens: List[str]) -> float:
        """Fast relevance calculation using pre-processed tokens"""
        content_tokens = set(self.get_processed_tokens())
        query_token_set = set(query_tokens)
        
        # Jaccard similarity
        intersection = len(content_tokens & query_token_set)
        union = len(content_tokens | query_token_set)
        
        if union == 0:
            return 0.0
        
        jaccard_score = intersection / union
        
        # Combine with retrieval score
        self.relevance_score = (jaccard_score * 0.6) + (self.retrieval_score * 0.4)
        return self.relevance_score
