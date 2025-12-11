import os
import json
import sys
import time
import hashlib
import threading
import re
import gc
import torch
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple, Union
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from collections import defaultdict
import weakref
from functools import wraps
from loguru import logger
from pydantic import Field, BaseModel, PrivateAttr
import chromadb
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from sqlalchemy.orm import Session
from sqlalchemy import text, or_, and_, func
import pandas as pd

# --- LangChain/Community Imports ---
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, JSONLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# Your existing imports
from app.agents.workflow.initalize import llm_instance, settings, agent_config
from app.agents.factory.tools.base import BaseAgentTool
from app.agents.factory.tools.search_tool import SearchTool
from app.utils.document_processor import DocumentCustomConverter, markdown_splitter, remove_image_tags
from app.db.base_class import Base  # Assuming you have this
from app.db.models.product import ProductModel # Your product model

# Text2SQL Integration
class Text2SQLParser:
    """Converts natural language queries to SQL for product database"""
    
    def __init__(self, llm_instance):
        self.llm = llm_instance
        self.schema_info = self._get_schema_info()
    
    def _get_schema_info(self) -> str:
        """Get database schema information for SQL generation"""
        return """
        Table: products
        Columns:
        - id (Integer, Primary Key): Product ID from CSV
        - name (String): Product name in Vietnamese
        - type (String): Product type/category
        - price (String): Formatted price string
        - price_numeric (Numeric): Numeric price for calculations
        - subject (String): Target audience (adult, kid, etc.)
        - working_time (Integer): Processing time in days
        - technology (String): Technology used
        - summary (Text): Detailed Vietnamese description
        - feature (Text): Vietnamese features
        - feature_en (Text): English features
        - product_index (String): Unique product identifier
        - is_active (String): Active status
        - created_at (DateTime): Creation timestamp
        - updated_at (DateTime): Last update timestamp
        
        Common queries:
        - Find products by price range
        - Search by product type
        - Filter by target audience (subject)
        - Find by technology
        - Search by processing time
        """
    
    def parse_query_to_sql(self, natural_query: str) -> Optional[str]:
        """Convert natural language to SQL query"""
        prompt = f"""
        Convert the following natural language query to SQL for the products table.
        
        Database Schema:
        {self.schema_info}
        
        Natural Language Query: "{natural_query}"
        
        Rules:
        1. Use only SELECT statements
        2. Always include WHERE is_active = 'true'
        3. Use ILIKE for case-insensitive text matching
        4. For price queries, use price_numeric column
        5. Limit results to 20 unless specified
        6. Use proper PostgreSQL syntax
        
        Return only the SQL query, no explanations:
        """
        
        try:
            response = self.llm.invoke(prompt)
            sql_query = response.content.strip()
            
            # Basic SQL injection prevention
            if self._is_safe_sql(sql_query):
                return sql_query
            else:
                logger.warning(f"Potentially unsafe SQL query: {sql_query}")
                return None
                
        except Exception as e:
            logger.error(f"Error generating SQL: {e}")
            return None
    
    def _is_safe_sql(self, sql: str) -> bool:
        """Basic SQL injection prevention"""
        dangerous_keywords = [
            'DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER', 'CREATE', 
            'TRUNCATE', 'EXEC', 'EXECUTE', '--', ';'
        ]
        
        sql_upper = sql.upper()
        for keyword in dangerous_keywords:
            if keyword in sql_upper:
                return False
        return True

class HybridProductResult(BaseModel):
    """Combined result from both vector and SQL searches"""
    content: str
    source: str
    result_type: str  # 'vector', 'sql', 'hybrid'
    relevance_score: float
    product_data: Optional[Dict[str, Any]] = None
    sql_query: Optional[str] = None

class EnhancedProductRetrieverTool(BaseAgentTool):
    """Enhanced product retriever with both vector search and Text2SQL capabilities"""
    
    name: str = "enhanced_product_retriever"
    description: str = "Retrieves product information using both vector search and SQL queries."
    
    # Core Configuration
    collection_name: str = Field(default="genestory_products")
    watch_directory: Path = Field(description="Directory to watch for documents.")
    db_session: Session = Field(description="Database session for SQL queries")
    
    # Vector search components (from original)
    _vector_store: Chroma = PrivateAttr(default=None)
    _embeddings: HuggingFaceEmbeddings = PrivateAttr(default=None)
    _text_splitter: RecursiveCharacterTextSplitter = PrivateAttr(default=None)
    _final_retriever: ContextualCompressionRetriever = PrivateAttr(default=None)
    
    # Text2SQL components
    _text2sql_parser: Text2SQLParser = PrivateAttr(default=None)
    
    # Performance components
    _query_cache: Dict[str, Any] = PrivateAttr(default_factory=dict)
    _thread_pool: ThreadPoolExecutor = PrivateAttr(default=None)
    _is_initialized: bool = PrivateAttr(default=False)
    
    def __init__(self, watch_directory: str, db_session: Session, **kwargs):
        super().__init__(**kwargs)
        self.watch_directory = Path(watch_directory).resolve()
        self.db_session = db_session
        self._initialize_all()
    
    def _initialize_all(self):
        """Initialize both vector and SQL components"""
        logger.info("Initializing Enhanced Product Retriever...")
        
        # Initialize Text2SQL parser
        self._text2sql_parser = Text2SQLParser(llm_instance)
        
        # Initialize vector components (simplified from original)
        self._initialize_vector_components()
        
        # Initialize thread pool
        self._thread_pool = ThreadPoolExecutor(max_workers=4)
        
        self._is_initialized = True
        logger.info("Enhanced Product Retriever initialized successfully.")
    
    def _initialize_vector_components(self):
        """Initialize vector search components"""
        try:
            # Initialize embeddings
            model_kwargs = {'device': 'cpu'}  # Start with CPU
            encode_kwargs = {'normalize_embeddings': True}
            
            self._embeddings = HuggingFaceEmbeddings(
                model_name=settings.HF_EMBEDDING_MODEL,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
            
            # Initialize vector store
            persistent_client = chromadb.PersistentClient(
                path=str(Path(settings.VECTOR_STORE_BASE_DIR))
            )
            self._vector_store = Chroma(
                client=persistent_client,
                collection_name=self.collection_name,
                embedding_function=self._embeddings
            )
            
            # Initialize text splitter
            self._text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1024, 
                chunk_overlap=256
            )
            
            # Build retriever pipeline
            self._build_retriever_pipeline()
            
        except Exception as e:
            logger.error(f"Error initializing vector components: {e}")
            self._vector_store = None
    
    def _build_retriever_pipeline(self):
        """Build the vector retriever pipeline"""
        try:
            if not self._vector_store:
                return
                
            base_retriever = self._vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            
            # Create reranker
            model = HuggingFaceCrossEncoder(
                model_name=settings.HF_RERANKER_MODEL,
                model_kwargs={'device': 'cpu'}
            )
            compressor = CrossEncoderReranker(model=model, top_n=3)
            
            self._final_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=base_retriever
            )
            
        except Exception as e:
            logger.error(f"Error building retriever pipeline: {e}")
            self._final_retriever = None
    
    def _detect_query_type(self, query: str) -> str:
        """Detect if query is better suited for SQL or vector search"""
        
        # SQL indicators
        sql_indicators = [
            'giá', 'price', 'cost', 'chi phí',
            'loại', 'type', 'category',
            'thời gian', 'working time', 'processing time',
            'công nghệ', 'technology',
            'đối tượng', 'subject', 'target',
            'so sánh', 'compare', 'comparison',
            'rẻ nhất', 'cheapest', 'expensive',
            'nhanh nhất', 'fastest', 'slowest'
        ]
        
        # Vector search indicators
        vector_indicators = [
            'mô tả', 'description', 'thông tin',
            'tính năng', 'feature', 'benefit',
            'ưu điểm', 'advantage', 'disadvantage',
            'chi tiết', 'detail', 'specific',
            'tư vấn', 'advice', 'recommend'
        ]
        
        query_lower = query.lower()
        
        sql_score = sum(1 for indicator in sql_indicators if indicator in query_lower)
        vector_score = sum(1 for indicator in vector_indicators if indicator in query_lower)
        
        if sql_score > vector_score:
            return 'sql'
        elif vector_score > sql_score:
            return 'vector'
        else:
            return 'hybrid'  # Use both
    
    async def _sql_search(self, query: str) -> List[HybridProductResult]:
        """Perform SQL-based search"""
        try:
            sql_query = self._text2sql_parser.parse_query_to_sql(query)
            if not sql_query:
                return []
            
            logger.info(f"Executing SQL query: {sql_query}")
            
            # Execute SQL query
            result = self.db_session.execute(text(sql_query))
            rows = result.fetchall()
            
            results = []
            for row in rows:
                # Convert row to dict
                row_dict = dict(row._mapping)
                
                # Create content summary
                content = f"""
                Product: {row_dict.get('name', 'N/A')}
                Type: {row_dict.get('type', 'N/A')}
                Price: {row_dict.get('price', 'N/A')}
                Subject: {row_dict.get('subject', 'N/A')}
                Technology: {row_dict.get('technology', 'N/A')}
                Working Time: {row_dict.get('working_time', 'N/A')} days
                Summary: {row_dict.get('summary', 'N/A')[:200]}...
                """
                
                result_obj = HybridProductResult(
                    content=content.strip(),
                    source=f"Database - Product ID: {row_dict.get('id')}",
                    result_type='sql',
                    relevance_score=0.9,  # High relevance for exact SQL matches
                    product_data=row_dict,
                    sql_query=sql_query
                )
                results.append(result_obj)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in SQL search: {e}")
            return []
    
    async def _vector_search(self, query: str) -> List[HybridProductResult]:
        """Perform vector-based search"""
        try:
            if not self._final_retriever:
                return []
            
            compressed_docs = await self._final_retriever.ainvoke(query)
            
            results = []
            for doc in compressed_docs:
                result_obj = HybridProductResult(
                    content=doc.page_content,
                    source=doc.metadata.get('source', 'unknown'),
                    result_type='vector',
                    relevance_score=0.7,  # Base relevance for vector search
                    product_data=None
                )
                results.append(result_obj)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            return []
    
    def _run(self, query: str) -> str:
        """Synchronous version"""
        return asyncio.run(self._arun(query))
    
    async def _arun(self, query: str) -> str:
        """Main retrieval method combining SQL and vector search"""
        if not self._is_initialized:
            return "Tool not initialized properly."
        
        logger.info(f"Processing query: '{query}'")
        
        # Detect query type
        query_type = self._detect_query_type(query)
        logger.info(f"Detected query type: {query_type}")
        
        all_results = []
        
        try:
            if query_type in ['sql', 'hybrid']:
                # Perform SQL search
                sql_results = await self._sql_search(query)
                all_results.extend(sql_results)
                logger.info(f"SQL search returned {len(sql_results)} results")
            
            if query_type in ['vector', 'hybrid'] and len(all_results) < 3:
                # Perform vector search if we need more results
                vector_results = await self._vector_search(query)
                all_results.extend(vector_results)
                logger.info(f"Vector search returned {len(vector_results)} results")
            
            # Sort by relevance score
            all_results.sort(key=lambda x: x.relevance_score, reverse=True)
            
            # Format output
            if not all_results:
                return "Không tìm thấy thông tin sản phẩm phù hợp với truy vấn của bạn."
            
            # Take top 5 results
            top_results = all_results[:5]
            
            formatted_results = []
            for i, result in enumerate(top_results, 1):
                formatted_result = f"--- Kết quả {i} ({result.result_type.upper()}) ---\n"
                formatted_result += f"Source: {result.source}\n"
                if result.sql_query:
                    formatted_result += f"SQL Query: {result.sql_query}\n"
                formatted_result += f"Content:\n{result.content}\n"
                formatted_results.append(formatted_result)
            
            return "\n".join(formatted_results)
            
        except Exception as e:
            logger.error(f"Error in _arun: {e}")
            return f"Đã xảy ra lỗi khi xử lý truy vấn: {str(e)}"
    
    async def get_product_by_id(self, product_id: int) -> Optional[Dict[str, Any]]:
        """Get specific product by ID"""
        try:
            product = self.db_session.query(ProductModel).filter(
                ProductModel.id == product_id,
                ProductModel.is_active == 'true'
            ).first()
            
            return product.to_dict() if product else None
            
        except Exception as e:
            logger.error(f"Error getting product by ID: {e}")
            return None
    
    async def get_products_by_price_range(self, min_price: float, max_price: float) -> List[Dict[str, Any]]:
        """Get products within price range"""
        try:
            products = self.db_session.query(ProductModel).filter(
                ProductModel.price_numeric.between(min_price, max_price),
                ProductModel.is_active == 'true'
            ).limit(10).all()
            
            return [product.to_dict() for product in products]
            
        except Exception as e:
            logger.error(f"Error getting products by price range: {e}")
            return []
    
    async def get_products_by_type(self, product_type: str) -> List[Dict[str, Any]]:
        """Get products by type"""
        try:
            products = self.db_session.query(ProductModel).filter(
                ProductModel.type.ilike(f'%{product_type}%'),
                ProductModel.is_active == 'true'
            ).limit(10).all()
            
            return [product.to_dict() for product in products]
            
        except Exception as e:
            logger.error(f"Error getting products by type: {e}")
            return []
    
    def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up Enhanced Product Retriever...")
        
        if self._thread_pool:
            self._thread_pool.shutdown(wait=False)
        
        # Clean up GPU resources
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Usage example
async def example_usage():
    """Example of how to use the enhanced retriever"""
    
    # Assuming you have a database session
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    
    # Create database session (replace with your actual database URL)
    engine = create_engine("postgresql://user:password@localhost/dbname")
    SessionLocal = sessionmaker(bind=engine)
    db_session = SessionLocal()
    
    try:
        # Initialize the enhanced retriever
        retriever = EnhancedProductRetrieverTool(
            watch_directory="./product_data",
            db_session=db_session
        )
        
        # Test queries
        test_queries = [
            "Tìm sản phẩm có giá dưới 3 triệu",  # Should trigger SQL
            "Thông tin chi tiết về GeneMap Adult",  # Should trigger vector
            "So sánh các gói xét nghiệm gen",  # Should trigger hybrid
            "Sản phẩm nào phù hợp với trẻ em?",  # Should trigger SQL
            "Tính năng của các gói premium"  # Should trigger vector
        ]
        
        for query in test_queries:
            print(f"\n{'='*50}")
            print(f"Query: {query}")
            print(f"{'='*50}")
            
            result = await retriever._arun(query)
            print(result)
            
            # Small delay between queries
            await asyncio.sleep(1)
    
    finally:
        db_session.close()
        retriever.cleanup()

if __name__ == "__main__":
    asyncio.run(example_usage())