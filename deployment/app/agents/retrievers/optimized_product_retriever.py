"""
Optimized ProductRetrieverTool that combines database queries with vector search
for enhanced product information retrieval.
"""

import re
import time
import hashlib
import threading
import asyncio
import chromadb
from typing import List, Dict, Any, Optional, Set, Tuple, Union
from enum import Enum
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from functools import wraps
import gc

from loguru import logger
from pydantic import Field, BaseModel, PrivateAttr
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, text
from sqlalchemy.orm import selectinload

# Database imports
from app.db.session import AsyncSessionLocal
from app.db.models.product import ProductModel
from app.crud.product import product_crud

# Vector search imports
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# Core imports
from app.agents.workflow.initalize import llm_instance, settings, agent_config
from app.agents.factory.tools.base import BaseAgentTool

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class QueryType(Enum):
    """Types of queries the system can handle"""
    SIMPLE_SEARCH = "simple_search"      # "thông tin về genemap adult"
    EXCLUSION = "exclusion"              # "các gói khác ngoài genemap adult" 
    COMPARISON = "comparison"            # "so sánh genemap adult và kid"
    LISTING = "listing"                  # "tất cả các gói dịch vụ"
    PRICE_QUERY = "price_query"          # "giá của genemap adult"
    FEATURE_QUERY = "feature_query"      # "tính năng của genemap"
    TYPE_QUERY = "type_query"            # "các sản phẩm chính"

@dataclass
class QueryIntent:
    """Represents the parsed intent of a user query"""
    query_type: QueryType
    main_terms: List[str]
    exclusion_terms: List[str]
    comparison_terms: List[str]
    context_terms: List[str]
    confidence: float
    original_query: str

@dataclass
class ProductSearchResult:
    """Enhanced product search result"""
    product: ProductModel
    relevance_score: float
    match_type: str  # 'exact', 'partial', 'fuzzy', 'vector'
    matched_fields: List[str]
    snippet: str

class PerformanceCache:
    """High-performance cache with TTL"""
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        self.cache = {}
        self.timestamps = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self.cache:
                return None
            
            if time.time() - self.timestamps[key] > self.ttl_seconds:
                del self.cache[key]
                del self.timestamps[key]
                return None
            
            return self.cache[key]
    
    def set(self, key: str, value: Any) -> None:
        with self._lock:
            if len(self.cache) >= self.max_size:
                oldest_key = min(self.timestamps.keys(), key=lambda k: self.timestamps[k])
                del self.cache[oldest_key]
                del self.timestamps[oldest_key]
            
            self.cache[key] = value
            self.timestamps[key] = time.time()

class QueryIntentAnalyzer:
    """Analyzes user queries to understand intent"""
    
    def __init__(self):
        self.exclusion_patterns = [
            r'(?:khác\s+)?(?:ngoài|trừ|loại\s+trừ)\s+([^?]+)',
            r'(?:không\s+phải|không\s+bao\s+gồm)\s+([^?]+)',
            r'(?:khác\s+với|khác\s+so\s+với)\s+([^?]+)',
        ]
        
        self.listing_patterns = [
            r'(?:tất\s+cả|toàn\s+bộ|danh\s+sách)\s+(?:các\s+)?(.+)',
            r'(?:có\s+)?(?:những|các)\s+(.+?)\s+(?:nào|gì)',
            r'(?:liệt\s+kê|kể\s+ra)\s+(.+)',
            r'(?:tất\s+cả|toàn\s+bộ)\s+(?:sản\s+phẩm|dịch\s+vụ|gói)',
            r'(?:danh\s+sách|list)\s*(?:của\s+)?(?:genestory|công\s+ty)',
        ]
        
        self.price_patterns = [
            r'(?:giá|chi\s+phí|phí|tiền)\s+(?:của\s+)?(.+)',
            r'(.+)\s+(?:giá|bao\s+nhiêu|chi\s+phí)',
            r'(?:bao\s+nhiêu\s+tiền|có\s+giá)\s+(.+)',
        ]
        
        self.comparison_patterns = [
            r'(?:so\s+sánh|khác\s+biệt)\s+(.+?)\s+(?:và|với)\s+(.+)',
            r'(.+?)\s+(?:khác\s+gì|giống\s+gì)\s+(.+)',
        ]
        
        self.feature_patterns = [
            r'(?:tính\s+năng|chức\s+năng|đặc\s+điểm)\s+(?:của\s+)?(.+)',
            r'(.+)\s+(?:có\s+gì|bao\s+gồm\s+gì|làm\s+được\s+gì)',
        ]
    
    def analyze(self, query: str) -> QueryIntent:
        """Analyze query and return intent"""
        query_lower = query.lower().strip()
        
        # Check for exclusion patterns
        for pattern in self.exclusion_patterns:
            match = re.search(pattern, query_lower)
            if match:
                exclusion_terms = self._extract_terms(match.group(1))
                main_terms = self._extract_main_terms(query_lower, exclusion_terms)
                return QueryIntent(
                    query_type=QueryType.EXCLUSION,
                    main_terms=main_terms,
                    exclusion_terms=exclusion_terms,
                    comparison_terms=[],
                    context_terms=[],
                    confidence=0.9,
                    original_query=query
                )
        
        # Check for price patterns
        for pattern in self.price_patterns:
            match = re.search(pattern, query_lower)
            if match:
                main_terms = self._extract_terms(match.group(1))
                return QueryIntent(
                    query_type=QueryType.PRICE_QUERY,
                    main_terms=main_terms,
                    exclusion_terms=[],
                    comparison_terms=[],
                    context_terms=['giá', 'chi phí', 'tiền'],
                    confidence=0.9,
                    original_query=query
                )
        
        # Check for comparison patterns
        for pattern in self.comparison_patterns:
            match = re.search(pattern, query_lower)
            if match:
                term1 = self._extract_terms(match.group(1))
                term2 = self._extract_terms(match.group(2))
                return QueryIntent(
                    query_type=QueryType.COMPARISON,
                    main_terms=term1 + term2,
                    exclusion_terms=[],
                    comparison_terms=term1 + term2,
                    context_terms=[],
                    confidence=0.8,
                    original_query=query
                )
        
        # Check for listing patterns
        for pattern in self.listing_patterns:
            match = re.search(pattern, query_lower)
            if match:
                main_terms = self._extract_terms(match.group(1))
                return QueryIntent(
                    query_type=QueryType.LISTING,
                    main_terms=main_terms,
                    exclusion_terms=[],
                    comparison_terms=[],
                    context_terms=[],
                    confidence=0.8,
                    original_query=query
                )
        
        # Check for feature patterns
        for pattern in self.feature_patterns:
            match = re.search(pattern, query_lower)
            if match:
                main_terms = self._extract_terms(match.group(1))
                return QueryIntent(
                    query_type=QueryType.FEATURE_QUERY,
                    main_terms=main_terms,
                    exclusion_terms=[],
                    comparison_terms=[],
                    context_terms=['tính năng', 'chức năng', 'đặc điểm'],
                    confidence=0.8,
                    original_query=query
                )
        
        # Default to simple search
        main_terms = self._extract_terms(query_lower)
        return QueryIntent(
            query_type=QueryType.SIMPLE_SEARCH,
            main_terms=main_terms,
            exclusion_terms=[],
            comparison_terms=[],
            context_terms=[],
            confidence=0.7,
            original_query=query
        )
    
    def _extract_terms(self, text: str) -> List[str]:
        """Extract meaningful terms from text"""
        # Remove common Vietnamese stopwords
        stopwords = {
            'của', 'và', 'có', 'là', 'được', 'cho', 'từ', 'với', 'về', 'trong',
            'các', 'những', 'này', 'đó', 'gì', 'như', 'để', 'hay', 'hoặc'
        }
        
        # Extract words
        words = re.findall(r'\b\w+\b', text.lower())
        return [word for word in words if word not in stopwords and len(word) > 2]
    
    def _extract_main_terms(self, query: str, exclusion_terms: List[str]) -> List[str]:
        """Extract main terms excluding the exclusion terms"""
        all_terms = self._extract_terms(query)
        return [term for term in all_terms if term not in exclusion_terms]

class OptimizedProductRetrieverTool(BaseAgentTool):
    """Optimized product retriever combining database and vector search"""
    
    name: str = "optimized_product_retriever"
    description: str = "Advanced product information retrieval with database and vector search"
    
    # Configuration
    use_database: bool = Field(default=True, description="Enable database search")
    use_vector_search: bool = Field(default=True, description="Enable vector search") 
    cache_enabled: bool = Field(default=True, description="Enable caching")
    
    # Private attributes
    _query_cache: PerformanceCache = PrivateAttr(default=None)
    _intent_analyzer: QueryIntentAnalyzer = PrivateAttr(default=None)
    _vector_store: Optional[Chroma] = PrivateAttr(default=None)
    _embeddings: Optional[HuggingFaceEmbeddings] = PrivateAttr(default=None)
    _final_retriever: Optional[ContextualCompressionRetriever] = PrivateAttr(default=None)
    _is_initialized: bool = PrivateAttr(default=False)
    _gpu_lock: threading.RLock = PrivateAttr(default=None)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all components"""
        logger.info("Initializing OptimizedProductRetrieverTool...")
        
        # Initialize caches
        if self.cache_enabled:
            self._query_cache = PerformanceCache(max_size=500, ttl_seconds=300)
        
        # Initialize intent analyzer
        self._intent_analyzer = QueryIntentAnalyzer()
        
        # Initialize vector search if enabled
        if self.use_vector_search:
            self._initialize_vector_search()
        
        # Thread safety
        self._gpu_lock = threading.RLock()
        
        self._is_initialized = True
        logger.info("OptimizedProductRetrieverTool initialized successfully")
    
    def _initialize_vector_search(self):
        """Initialize vector search components"""
        try:
            logger.info("Initializing vector search components...")
            
            # Initialize embeddings
            model_kwargs = {'device': 'cpu'}  # Start with CPU
            encode_kwargs = {'normalize_embeddings': True}
            
            self._embeddings = HuggingFaceEmbeddings(
                model_name=getattr(settings, 'HF_EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2'),
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
            
            # Initialize vector store for product documents
            vector_store_path = Path(getattr(settings, 'VECTOR_STORE_BASE_DIR', './vector_stores_data'))
            vector_store_path.mkdir(parents=True, exist_ok=True)
            
            import chromadb
            persistent_client = chromadb.PersistentClient(path=str(vector_store_path))
            self._vector_store = Chroma(
                client=persistent_client,
                collection_name="products_optimized",
                embedding_function=self._embeddings
            )
            
            # Build retriever pipeline
            self._build_retriever_pipeline()
            
            logger.info("Vector search components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector search: {e}")
            self.use_vector_search = False
    
    def _build_retriever_pipeline(self):
        """Build the retriever pipeline with reranking"""
        try:
            base_retriever = self._vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 10}
            )
            
            # Create reranker
            model_kwargs = {'device': 'cpu'}
            reranker_model = HuggingFaceCrossEncoder(
                model_name=getattr(settings, 'HF_RERANKER_MODEL', 'cross-encoder/ms-marco-MiniLM-L-6-v2'),
                model_kwargs=model_kwargs
            )
            compressor = CrossEncoderReranker(model=reranker_model, top_n=5)

            self._final_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=base_retriever
            )
            
            logger.info("Retriever pipeline built successfully")
            
        except Exception as e:
            logger.error(f"Failed to build retriever pipeline: {e}")
            self._final_retriever = None

    async def _database_search(self, intent: QueryIntent) -> List[ProductSearchResult]:
        """Search products using database queries"""
        if not self.use_database:
            return []
        
        try:
            async with AsyncSessionLocal() as db:
                results = []
                
                if intent.query_type == QueryType.LISTING:
                    # Get all products - increase limit for comprehensive listing
                    products = await product_crud.get_all(db, limit=100)
                    for product in products:
                        results.append(ProductSearchResult(
                            product=product,
                            relevance_score=0.8,
                            match_type='listing',
                            matched_fields=['all'],
                            snippet=f"{product.name} - {product.type} - {product.price}"
                        ))
                
                elif intent.query_type == QueryType.EXCLUSION:
                    # Get products excluding certain terms
                    all_products = await product_crud.get_all(db, limit=50)
                    excluded_products = set()
                    
                    # Find products to exclude
                    for term in intent.exclusion_terms:
                        excluded = await self._find_products_by_term(db, term)
                        excluded_products.update(p.id for p in excluded)
                    
                    # Return products not in exclusion set
                    for product in all_products:
                        if product.id not in excluded_products:
                            results.append(ProductSearchResult(
                                product=product,
                                relevance_score=0.7,
                                match_type='exclusion',
                                matched_fields=['name'],
                                snippet=f"{product.name} - {product.type}"
                            ))
                
                elif intent.query_type == QueryType.PRICE_QUERY:
                    # Search by product name and return with price focus
                    products = await self._find_products_by_terms(db, intent.main_terms)
                    for product in products:
                        snippet = f"{product.name} - Giá: {product.price}"
                        results.append(ProductSearchResult(
                            product=product,
                            relevance_score=0.9,
                            match_type='price',
                            matched_fields=['name', 'price'],
                            snippet=snippet
                        ))
                
                elif intent.query_type == QueryType.COMPARISON:
                    # Find products for comparison
                    products = await self._find_products_by_terms(db, intent.comparison_terms)
                    for product in products:
                        snippet = f"{product.name} - {product.type} - {product.price}"
                        results.append(ProductSearchResult(
                            product=product,
                            relevance_score=0.8,
                            match_type='comparison',
                            matched_fields=['name', 'type'],
                            snippet=snippet
                        ))
                
                else:  # SIMPLE_SEARCH, FEATURE_QUERY, TYPE_QUERY
                    products = await self._find_products_by_terms(db, intent.main_terms)
                    for product in products:
                        relevance = self._calculate_relevance(product, intent.main_terms)
                        snippet = self._create_snippet(product, intent)
                        results.append(ProductSearchResult(
                            product=product,
                            relevance_score=relevance,
                            match_type='search',
                            matched_fields=self._get_matched_fields(product, intent.main_terms),
                            snippet=snippet
                        ))
                
                return sorted(results, key=lambda x: x.relevance_score, reverse=True)
                
        except Exception as e:
            logger.error(f"Database search failed: {e}")
            return []
    
    async def _find_products_by_terms(self, db: AsyncSession, terms: List[str]) -> List[ProductModel]:
        """Find products matching any of the given terms"""
        if not terms:
            return []
        
        # Create search conditions
        conditions = []
        for term in terms:
            term_pattern = f"%{term}%"
            conditions.extend([
                ProductModel.name.ilike(term_pattern),
                ProductModel.feature.ilike(term_pattern),
                ProductModel.feature_en.ilike(term_pattern),
                ProductModel.summary.ilike(term_pattern),
                ProductModel.product_index.ilike(term_pattern)
            ])
        
        # Execute query
        result = await db.execute(
            select(ProductModel)
            .where(and_(
                or_(*conditions),
                ProductModel.is_active == "true"
            ))
            .limit(20)
        )
        
        return result.scalars().all()
    
    async def _find_products_by_term(self, db: AsyncSession, term: str) -> List[ProductModel]:
        """Find products matching a single term"""
        term_pattern = f"%{term}%"
        result = await db.execute(
            select(ProductModel)
            .where(and_(
                or_(
                    ProductModel.name.ilike(term_pattern),
                    ProductModel.product_index.ilike(term_pattern)
                ),
                ProductModel.is_active == "true"
            ))
        )
        return result.scalars().all()
    
    def _calculate_relevance(self, product: ProductModel, terms: List[str]) -> float:
        """Calculate relevance score for a product"""
        score = 0.0
        total_terms = len(terms)
        
        if total_terms == 0:
            return 0.5
        
        text_fields = [
            (product.name or "", 3.0),           # Name has highest weight
            (product.product_index or "", 2.5), # Product index high weight
            (product.feature or "", 1.5),       # Features medium weight
            (product.summary or "", 1.0),       # Summary normal weight
            (product.type or "", 1.2)           # Type medium weight
        ]
        
        for term in terms:
            term_lower = term.lower()
            for text, weight in text_fields:
                if term_lower in text.lower():
                    score += weight
        
        # Normalize by number of terms
        return min(score / total_terms, 1.0)
    
    def _get_matched_fields(self, product: ProductModel, terms: List[str]) -> List[str]:
        """Get list of fields that matched the search terms"""
        matched = []
        
        field_map = {
            'name': product.name or "",
            'feature': product.feature or "",
            'summary': product.summary or "",
            'type': product.type or "",
            'product_index': product.product_index or ""
        }
        
        for field, text in field_map.items():
            for term in terms:
                if term.lower() in text.lower():
                    matched.append(field)
                    break
        
        return matched
    
    def _create_snippet(self, product: ProductModel, intent: QueryIntent) -> str:
        """Create a relevant snippet for the product"""
        if intent.query_type == QueryType.PRICE_QUERY:
            return f"{product.name} - Giá: {product.price} - {product.type}"
        elif intent.query_type == QueryType.FEATURE_QUERY:
            features = product.feature_en or product.feature or "Chưa có thông tin tính năng"
            return f"{product.name} - Tính năng: {features[:200]}..."
        else:
            summary = product.summary or product.feature or "Chưa có thông tin chi tiết"
            return f"{product.name} - {product.type} - {summary[:200]}..."

    async def _vector_search(self, intent: QueryIntent) -> List[ProductSearchResult]:
        """Search using vector similarity"""
        if not self.use_vector_search or not self._final_retriever:
            return []
        
        try:
            query = intent.original_query
            compressed_docs = await self._final_retriever.ainvoke(query)
            
            results = []
            for doc in compressed_docs:
                # Try to map back to product if possible
                source = doc.metadata.get('source', '')
                snippet = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                
                # Create a pseudo-product result for vector matches
                # In a real implementation, you'd want to link this back to actual products
                results.append(ProductSearchResult(
                    product=None,  # Vector result doesn't have direct product mapping
                    relevance_score=0.6,  # Default vector score
                    match_type='vector',
                    matched_fields=['content'],
                    snippet=snippet
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    def _merge_results(self, db_results: List[ProductSearchResult], 
                      vector_results: List[ProductSearchResult]) -> List[ProductSearchResult]:
        """Merge and deduplicate results from different sources"""
        # For now, prioritize database results since they're more structured
        # In the future, could implement more sophisticated merging
        
        all_results = db_results + vector_results
        
        # Remove duplicates based on product ID (for db results) or content (for vector results)
        seen_products = set()
        seen_content = set()
        merged_results = []
        
        for result in all_results:
            if result.product:
                if result.product.id not in seen_products:
                    seen_products.add(result.product.id)
                    merged_results.append(result)
            else:
                content_hash = hashlib.md5(result.snippet.encode()).hexdigest()
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    merged_results.append(result)
        
        # Sort by relevance score
        return sorted(merged_results, key=lambda x: x.relevance_score, reverse=True)

    def _format_results(self, results: List[ProductSearchResult], intent: QueryIntent) -> str:
        """Format results into a readable response"""
        if not results:
            return "Không tìm thấy thông tin sản phẩm phù hợp với yêu cầu của bạn."
        
        # Limit results based on query type
        if intent.query_type == QueryType.LISTING:
            top_results = results[:20]  # Show more results for listing queries
        else:
            top_results = results[:5]   # Standard limit for other queries
        
        if intent.query_type == QueryType.LISTING:
            response = "Danh sách các sản phẩm/dịch vụ hiện có:\n\n"
            
            # Group products by type for better organization
            product_groups = {}
            for result in top_results:
                if result.product:
                    product_type = result.product.type or "Khác"
                    if product_type not in product_groups:
                        product_groups[product_type] = []
                    product_groups[product_type].append(result.product)
            
            # Display grouped products
            for product_type, products in product_groups.items():
                response += f"📋 **{product_type.title()}:**\n"
                for i, product in enumerate(products, 1):
                    response += f"   {i}. {product.name}\n"
                    if product.price:
                        response += f"      💰 Giá: {product.price}\n"
                    if product.summary:
                        summary = product.summary[:100] + "..." if len(product.summary) > 100 else product.summary
                        response += f"      📝 {summary}\n"
                    response += "\n"
                response += "\n"
        
        elif intent.query_type == QueryType.EXCLUSION:
            response = f"Các sản phẩm khác (ngoài {', '.join(intent.exclusion_terms)}):\n\n"
            for i, result in enumerate(top_results, 1):
                if result.product:
                    response += f"{i}. {result.product.name} - {result.product.type}\n"
                    if result.product.summary:
                        response += f"   {result.product.summary[:150]}...\n"
                    response += "\n"
        
        elif intent.query_type == QueryType.COMPARISON:
            response = "Thông tin so sánh các sản phẩm:\n\n"
            for i, result in enumerate(top_results, 1):
                if result.product:
                    response += f"{i}. {result.product.name}\n"
                    response += f"   Loại: {result.product.type}\n"
                    response += f"   Giá: {result.product.price}\n"
                    if result.product.working_time:
                        response += f"   Thời gian xử lý: {result.product.working_time} ngày\n"
                    response += "\n"
        
        else:
            # Standard format for other query types
            response = ""
            for i, result in enumerate(top_results, 1):
                if result.product:
                    response += f"🔹 {result.product.name}\n"
                    response += f"Loại: {result.product.type}\n"
                    if result.product.price:
                        response += f"Giá: {result.product.price}\n"
                    if intent.query_type == QueryType.FEATURE_QUERY and result.product.feature:
                        response += f"Tính năng: {result.product.feature}\n"
                    elif result.product.summary:
                        response += f"Mô tả: {result.product.summary[:200]}...\n"
                    response += "\n"
                else:
                    response += f"🔹 {result.snippet}\n\n"
        
        return response.strip()

    async def _arun(self, query: str) -> str:
        """Main async run method"""
        start_time = time.time()
        
        # Check cache first
        if self.cache_enabled and self._query_cache:
            cache_key = hashlib.md5(query.encode()).hexdigest()
            cached_result = self._query_cache.get(cache_key)
            if cached_result:
                logger.info(f"Cache hit for query: '{query}'")
                return cached_result
        
        try:
            # Analyze query intent
            intent = self._intent_analyzer.analyze(query)
            logger.info(f"Query intent: {intent.query_type.value}, terms: {intent.main_terms}")
            
            # Perform searches
            db_results = await self._database_search(intent)
            vector_results = await self._vector_search(intent) if self.use_vector_search else []
            
            # Merge and format results
            merged_results = self._merge_results(db_results, vector_results)
            response = self._format_results(merged_results, intent)
            
            # Cache the result
            if self.cache_enabled and self._query_cache:
                self._query_cache.set(cache_key, response)
            
            processing_time = time.time() - start_time
            logger.info(f"Query processed in {processing_time:.2f}s: '{query}' -> {len(merged_results)} results")
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query '{query}': {e}")
            return f"Xin lỗi, đã xảy ra lỗi khi tìm kiếm thông tin: {str(e)}"

    def _run(self, query: str) -> str:
        """Synchronous run method"""
        return asyncio.run(self._arun(query))

    def clear_cache(self):
        """Clear the query cache"""
        if self._query_cache:
            self._query_cache.cache.clear()
            self._query_cache.timestamps.clear()
            logger.info("Query cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics"""
        stats = {
            "initialized": self._is_initialized,
            "database_enabled": self.use_database,
            "vector_search_enabled": self.use_vector_search,
            "cache_enabled": self.cache_enabled
        }
        
        if self._query_cache:
            stats["cache_size"] = len(self._query_cache.cache)
            stats["cache_max_size"] = self._query_cache.max_size
        
        return stats
