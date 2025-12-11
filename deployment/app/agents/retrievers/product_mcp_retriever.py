import asyncio
import os
import json
import time
import hashlib
import threading
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple, Union
from enum import Enum
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

from loguru import logger
from pydantic import BaseModel, Field as PydanticField
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# --- LangChain Imports for Document Processing ---
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, JSONLoader, TextLoader

# --- MCP Client Imports ---
from mcp.client.sse import sse_client
from mcp.client.session import ClientSession

# --- Database Imports ---
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, text
from app.db.session import AsyncSessionLocal
from app.db.models.product import ProductModel
from app.crud.product import product_crud

# --- Base Agent Tool Import ---
from app.agents.factory.tools.base import BaseAgentTool
from app.core.config import settings

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
    product: Optional[ProductModel]
    relevance_score: float
    match_type: str  # 'exact', 'partial', 'fuzzy', 'vector', 'mcp'
    matched_fields: List[str]
    snippet: str
    source: str  # 'database', 'mcp', 'vector'


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


class DocumentWatcher(FileSystemEventHandler):
    """File system watcher that triggers document uploads to MCP server."""
    
    def __init__(self, retriever_tool: 'EnhancedProductRetrieverMCPClient'):
        self.tool = retriever_tool

    def on_created(self, event):
        if not event.is_directory:
            logger.info(f"[Watcher] New file detected: {event.src_path}")
            time.sleep(1)  # Small delay to ensure file is fully written
            asyncio.create_task(self.tool._process_file_if_needed(Path(event.src_path)))

    def on_modified(self, event):
        if not event.is_directory and "_registry.json" not in event.src_path:
            logger.info(f"[Watcher] File modified: {event.src_path}")
            time.sleep(1)
            asyncio.create_task(self.tool._process_file_if_needed(Path(event.src_path)))
            
    def on_deleted(self, event):
        if not event.is_directory:
            logger.info(f"[Watcher] File deleted: {event.src_path}")
            asyncio.create_task(self.tool._remove_from_registry(Path(event.src_path)))


class ProductRetrieverInput(BaseModel):
    """Input schema for product document retrieval."""
    query: str = PydanticField(description="Search query for finding relevant product information")
    collection_name: str = PydanticField(description="Name of the document collection to search in", default="")
    max_results: int = PydanticField(default=5, description="Maximum number of results to return")


class EnhancedProductRetrieverMCPClient(BaseAgentTool):
    """Enhanced MCP client with intelligent query analysis and intent-based search."""
    
    name: str = "enhanced_product_retriever_mcp"
    description: str = "Intelligent product retrieval with MCP server integration, query intent analysis, and multi-source search"
    args_schema: type[BaseModel] = ProductRetrieverInput
    
    # Configuration
    mcp_server_url: str = PydanticField(description="URL of the MCP server")
    watch_directory: Path = PydanticField(description="Directory to watch for document changes")
    default_collection: str = PydanticField(default="product_docs", description="Default collection name")
    enable_database_sync: bool = PydanticField(default=True, description="Enable syncing with product database")
    enable_cache: bool = PydanticField(default=True, description="Enable query caching")
    
    # Internal components
    _document_registry: Dict[str, str] = {}
    _observer: Optional[Observer] = None
    _text_splitter: Optional[RecursiveCharacterTextSplitter] = None
    _thread_pool: Optional[ThreadPoolExecutor] = None
    _intent_analyzer: Optional[QueryIntentAnalyzer] = None
    _query_cache: Optional[PerformanceCache] = None
    _is_initialized: bool = False
    
    def __init__(self, 
                 mcp_server_url: str,
                 watch_directory: str,
                 default_collection: str = "product_docs",
                 enable_database_sync: bool = True,
                 enable_cache: bool = True,
                 **kwargs):
        """
        Initialize the enhanced MCP client tool.
        
        Args:
            mcp_server_url: URL of the MCP server (e.g., "http://localhost:50051/sse")
            watch_directory: Local directory to watch for document changes
            default_collection: Default collection name for documents
            enable_database_sync: Whether to sync with product database
            enable_cache: Whether to enable query caching
        """
        super().__init__(
            mcp_server_url=mcp_server_url,
            watch_directory=Path(watch_directory).resolve(),
            default_collection=default_collection,
            enable_database_sync=enable_database_sync,
            enable_cache=enable_cache,
            **kwargs
        )
        
        if not self._is_initialized:
            asyncio.create_task(self._initialize_all())
    
    async def _initialize_all(self):
        """Initialize all components."""
        logger.info(f"Initializing EnhancedProductRetrieverMCPClient...")
        
        # Ensure watch directory exists
        self.watch_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize query intent analyzer
        self._intent_analyzer = QueryIntentAnalyzer()
        
        # Initialize cache if enabled
        if self.enable_cache:
            self._query_cache = PerformanceCache(max_size=500, ttl_seconds=300)
        
        # Initialize document processing components
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.OVERLAP_SIZE,
            is_separator_regex=True
        )
        
        # Initialize thread pool for parallel processing
        self._thread_pool = ThreadPoolExecutor(
            max_workers=4,
            thread_name_prefix="Enhanced_mcp_client"
        )
        
        # Load document registry
        await self._load_document_registry()
        
        # Test MCP server connection
        if await self._test_server_connection():
            logger.info("MCP server connection successful")
        else:
            logger.warning("MCP server connection failed - some features may not work")
        
        # Sync with database if enabled
        if self.enable_database_sync:
            await self._sync_database_products()
        
        # Scan and process existing files
        await self._scan_and_process_all_files()
        
        # Start document watcher
        self._start_document_watcher()
        
        self._is_initialized = True
        logger.info("EnhancedProductRetrieverMCPClient initialized successfully")
    
    async def _test_server_connection(self) -> bool:
        """Test connection to MCP server."""
        try:
            async with sse_client(url=self.mcp_server_url) as streams:
                async with ClientSession(read_stream=streams[0], write_stream=streams[1]) as session:
                    await session.initialize()
                    logger.debug("MCP server connection test successful")
                    return True
        except Exception as e:
            logger.error(f"MCP server connection test failed: {e}")
            return False

    # [Previous methods remain the same: _load_document_registry, _save_document_registry, etc.]
    @property
    def _registry_path(self) -> Path:
        """Path to the document registry file."""
        return self.watch_directory / f"{self.default_collection}_mcp_registry.json"
    
    async def _load_document_registry(self):
        """Load the document registry from disk."""
        if self._registry_path.exists():
            try:
                with open(self._registry_path, 'r', encoding='utf-8') as f:
                    self._document_registry = json.load(f)
                logger.info(f"Loaded {len(self._document_registry)} entries from registry")
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Failed to load registry: {e}. Starting fresh.")
                self._document_registry = {}
        else:
            self._document_registry = {}
    
    async def _save_document_registry(self):
        """Save the document registry to disk."""
        try:
            with open(self._registry_path, 'w', encoding='utf-8') as f:
                json.dump(self._document_registry, f, indent=2, ensure_ascii=False)
            logger.debug(f"Registry saved with {len(self._document_registry)} entries")
        except IOError as e:
            logger.error(f"Failed to save registry: {e}")
    
    def _get_file_hash(self, file_path: Path) -> Optional[str]:
        """Calculate MD5 hash of a file."""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except IOError as e:
            logger.error(f"Could not read file for hashing: {file_path}. Error: {e}")
            return None
    
    def _load_and_split_file(self, file_path: Path) -> List[Document]:
        """Load and split a file into document chunks."""
        loader_map = {
            '.pdf': PyPDFLoader,
            '.csv': CSVLoader,
            '.json': JSONLoader,
            '.txt': TextLoader
        }
        
        loader_class = loader_map.get(file_path.suffix.lower())
        if not loader_class:
            logger.warning(f"Unsupported file type: {file_path.suffix}")
            return []
        
        try:
            loader = loader_class(str(file_path))
            raw_docs = loader.load()
            
            # Split documents into chunks
            split_docs = self._text_splitter.split_documents(raw_docs)
            
            # Add source metadata
            for doc in split_docs:
                doc.metadata['source'] = file_path.name
                doc.metadata['file_path'] = str(file_path)
                doc.metadata['timestamp'] = time.time()
                doc.metadata['document_type'] = 'product_info'
            
            logger.info(f"Loaded and split {file_path.name} into {len(split_docs)} chunks")
            return split_docs
            
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            return []

    async def _upload_documents_to_server(self, documents: List[Document], collection_name: str) -> bool:
        """Upload document chunks to the MCP server."""
        if not documents:
            return False
        
        try:
            # Convert documents to list of strings (content only)
            doc_contents = [doc.page_content for doc in documents]
            doc_metadata = [doc.metadata for doc in documents]
            
            async with sse_client(url=self.mcp_server_url) as streams:
                async with ClientSession(read_stream=streams[0], write_stream=streams[1]) as session:
                    await session.initialize()
                    
                    # Call the ingest_documents tool on the server
                    response = await session.call_tool(
                        "ingest_documents",
                        arguments={
                            "documents": doc_contents,
                            "metadata": doc_metadata,
                            "collection_name": collection_name
                        }
                    )
                    
                    if response and response.content:
                        result_text = response.content[0].text
                        logger.info(f"Server response: {result_text}")
                        return "successfully" in result_text.lower()
                    else:
                        logger.error("No response from server")
                        return False
                        
        except Exception as e:
            logger.error(f"Error uploading documents to server: {e}")
            return False
    
    async def _sync_database_products(self):
        """Sync product database information to vector store."""
        if not self.enable_database_sync:
            return
        
        try:
            logger.info("Syncing product database to vector store...")
            async with AsyncSessionLocal() as db:
                # Get all active products
                result = await db.execute(
                    select(ProductModel).where(ProductModel.is_active == "true")
                )
                products = result.scalars().all()
                
                if not products:
                    logger.info("No active products found in database")
                    return
                
                # Convert products to documents
                documents = []
                for product in products:
                    content = self._create_product_document(product)
                    metadata = {
                        'source': 'database',
                        'product_id': str(product.id),
                        'product_name': product.name,
                        'product_type': product.type or "",
                        'timestamp': time.time(),
                        'document_type': 'product_database'
                    }
                    
                    doc = Document(page_content=content, metadata=metadata)
                    documents.append(doc)
                
                # Upload to MCP server
                success = await self._upload_documents_to_server(documents, self.default_collection)
                if success:
                    logger.info(f"Successfully synced {len(documents)} products to vector store")
                else:
                    logger.error("Failed to sync products to vector store")
                    
        except Exception as e:
            logger.error(f"Error syncing database products: {e}")
    
    def _create_product_document(self, product: ProductModel) -> str:
        """Create a searchable document from a product record."""
        content_parts = []
        
        # Product name and basic info
        if product.name:
            content_parts.append(f"Tên sản phẩm: {product.name}")
        
        if product.type:
            content_parts.append(f"Loại sản phẩm: {product.type}")
        
        if product.price:
            content_parts.append(f"Giá: {product.price}")
        
        if product.working_time:
            content_parts.append(f"Thời gian xử lý: {product.working_time} ngày")
        
        # Product features and descriptions
        if product.feature:
            content_parts.append(f"Tính năng: {product.feature}")
        
        if product.feature_en:
            content_parts.append(f"Features (EN): {product.feature_en}")
        
        if product.summary:
            content_parts.append(f"Mô tả: {product.summary}")
        
        if product.product_index:
            content_parts.append(f"Chỉ số sản phẩm: {product.product_index}")
        
        # Additional details
        if product.subject:
            content_parts.append(f"Đối tượng: {product.subject}")
        
        if product.technology:
            content_parts.append(f"Công nghệ: {product.technology}")
        
        return "\n".join(content_parts)

    # [Document processing methods remain the same]
    async def _process_file_if_needed(self, file_path: Path):
        """Process a file if it has changed since last processing."""
        if "_registry.json" in file_path.name or "_mcp_registry.json" in file_path.name:
            return
        
        current_hash = self._get_file_hash(file_path)
        if not current_hash:
            return
        
        stored_hash = self._document_registry.get(str(file_path))
        if current_hash != stored_hash:
            logger.info(f"Change detected for '{file_path.name}'. Processing...")
            
            # Load and split the file
            documents = self._load_and_split_file(file_path)
            
            if documents:
                # Upload to server
                success = await self._upload_documents_to_server(documents, self.default_collection)
                
                if success:
                    # Update registry
                    self._document_registry[str(file_path)] = current_hash
                    await self._save_document_registry()
                    logger.info(f"Successfully processed and uploaded '{file_path.name}'")
                else:
                    logger.error(f"Failed to upload '{file_path.name}' to server")
            else:
                logger.warning(f"No documents extracted from '{file_path.name}'")
    
    async def _scan_and_process_all_files(self):
        """Scan directory and process all supported files."""
        logger.info(f"Scanning directory: {self.watch_directory}")
        
        current_files = set()
        for file_path in self.watch_directory.rglob('*'):
            if file_path.is_file() and not file_path.name.endswith('_registry.json'):
                current_files.add(str(file_path))
                await self._process_file_if_needed(file_path)
        
        # Clean up registry for deleted files
        registered_files = set(self._document_registry.keys())
        deleted_files = registered_files - current_files
        for file_path_str in deleted_files:
            await self._remove_from_registry(Path(file_path_str))
        
        await self._save_document_registry()
    
    async def _remove_from_registry(self, file_path: Path):
        """Remove a file from the registry."""
        path_str = str(file_path)
        if path_str in self._document_registry:
            logger.info(f"Removing '{file_path.name}' from registry")
            del self._document_registry[path_str]
            await self._save_document_registry()
    
    def _start_document_watcher(self):
        """Start watching the directory for file changes."""
        if self._observer:
            return
        
        self._observer = Observer()
        event_handler = DocumentWatcher(self)
        self._observer.schedule(event_handler, str(self.watch_directory), recursive=True)
        
        watcher_thread = threading.Thread(target=self._observer.start, daemon=True)
        watcher_thread.start()
        logger.info(f"Started document watcher on '{self.watch_directory}'")

    # === NEW ENHANCED SEARCH METHODS ===
    
    async def _database_search_with_intent(self, intent: QueryIntent) -> List[ProductSearchResult]:
        """Search products using database queries with intent analysis"""
        if not self.enable_database_sync:
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
                            snippet=f"{product.name} - {product.type} - {product.price}",
                            source='database'
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
                                snippet=f"{product.name} - {product.type}",
                                source='database'
                            ))
                
                elif intent.query_type == QueryType.PRICE_QUERY:
                    # Search by product name and return with price focus
                    products = await self._find_products_by_terms(db, intent.main_terms)
                    for product in products:
                        snippet = f"{product.name} - Giá: {product.price}"
                        if product.working_time:
                            snippet += f" - Thời gian xử lý: {product.working_time} ngày"
                        results.append(ProductSearchResult(
                            product=product,
                            relevance_score=0.9,
                            match_type='price',
                            matched_fields=['name', 'price'],
                            snippet=snippet,
                            source='database'
                        ))
                
                elif intent.query_type == QueryType.COMPARISON:
                    # Find products for comparison
                    products = await self._find_products_by_terms(db, intent.comparison_terms)
                    for product in products:
                        snippet = f"{product.name} - {product.type} - {product.price}"
                        if product.working_time:
                            snippet += f" - {product.working_time} ngày"
                        results.append(ProductSearchResult(
                            product=product,
                            relevance_score=0.8,
                            match_type='comparison',
                            matched_fields=['name', 'type'],
                            snippet=snippet,
                            source='database'
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
                            snippet=snippet,
                            source='database'
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
                ProductModel.product_index.ilike(term_pattern),
                ProductModel.type.ilike(term_pattern)
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

    async def _retrieve_from_mcp_server(self, intent: QueryIntent, collection_name: str, k: int = 5) -> List[ProductSearchResult]:
        """Retrieve documents from MCP server with intent-aware query enhancement"""
        try:
            # Enhance query based on intent
            enhanced_query = self._enhance_query_for_intent(intent)
            
            async with sse_client(url=self.mcp_server_url) as streams:
                async with ClientSession(read_stream=streams[0], write_stream=streams[1]) as session:
                    await session.initialize()
                    
                    # Call the retrieve_vectorstore tool on the server
                    response = await session.call_tool(
                        "retrieve_vectorstore_with_reranker",
                        arguments={
                            "query": enhanced_query,
                            "collection": collection_name,
                            "initial_k": settings.TOP_K_RETRIEVE,
                            "final_k": settings.TOP_K_RERANK
                        }
                    )
                    
                    if response and response.content:
                        result_text = response.content[0].text
                        logger.info(f"Retrieved MCP documents for query: '{enhanced_query[:50]}...'")
                        
                        # Parse MCP response into structured results
                        return self._parse_mcp_response(result_text, intent)
                    else:
                        return []
                        
        except ConnectionRefusedError:
            error_msg = f"Connection refused. Is the MCP server running at {self.mcp_server_url}?"
            logger.error(error_msg)
            return []
        except Exception as e:
            error_msg = f"Error retrieving documents from MCP server: {e}"
            logger.error(error_msg)
            return []
    
    def _enhance_query_for_intent(self, intent: QueryIntent) -> str:
        """Enhance the query based on the detected intent"""
        base_query = intent.original_query
        
        if intent.query_type == QueryType.PRICE_QUERY:
            # Add price-related terms to improve retrieval
            return f"{base_query} giá chi phí tiền bao nhiêu"
        elif intent.query_type == QueryType.FEATURE_QUERY:
            # Add feature-related terms
            return f"{base_query} tính năng chức năng đặc điểm"
        elif intent.query_type == QueryType.COMPARISON:
            # Enhance for comparison
            return f"so sánh {base_query} khác biệt giống nhau"
        elif intent.query_type == QueryType.LISTING:
            # Broaden for listing
            return f"danh sách tất cả {' '.join(intent.main_terms)} sản phẩm dịch vụ"
        else:
            return base_query
    
    def _parse_mcp_response(self, response_text: str, intent: QueryIntent) -> List[ProductSearchResult]:
        """Parse MCP server response into structured results"""
        results = []
        
        # Simple parsing - in practice, you might have more structured responses
        lines = response_text.split('\n')
        current_result = ""
        
        for line in lines:
            line = line.strip()
            if line:
                current_result += line + " "
                if len(current_result) > 200:  # Create chunks
                    results.append(ProductSearchResult(
                        product=None,  # MCP results don't map to products directly
                        relevance_score=0.6,
                        match_type='mcp',
                        matched_fields=['content'],
                        snippet=current_result.strip()[:300] + "...",
                        source='mcp'
                    ))
                    current_result = ""
        
        # Add any remaining content
        if current_result.strip():
            results.append(ProductSearchResult(
                product=None,
                relevance_score=0.6,
                match_type='mcp',
                matched_fields=['content'],
                snippet=current_result.strip()[:300] + "...",
                source='mcp'
            ))
        
        return results
    
    def _merge_results(self, db_results: List[ProductSearchResult], 
                      mcp_results: List[ProductSearchResult]) -> List[ProductSearchResult]:
        """Merge and deduplicate results from different sources"""
        # Prioritize database results since they're more structured
        all_results = db_results + mcp_results
        
        # Remove duplicates and sort by relevance
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
        
        return sorted(merged_results, key=lambda x: x.relevance_score, reverse=True)
    
    def _format_results_with_intent(self, results: List[ProductSearchResult], intent: QueryIntent) -> str:
        """Format results into a readable response based on query intent"""
        if not results:
            return self._get_no_results_message(intent)
        
        # Limit results based on query type
        if intent.query_type == QueryType.LISTING:
            top_results = results[:20]
        else:
            top_results = results[:5]
        
        if intent.query_type == QueryType.LISTING:
            return self._format_listing_results(top_results)
        elif intent.query_type == QueryType.EXCLUSION:
            return self._format_exclusion_results(top_results, intent)
        elif intent.query_type == QueryType.COMPARISON:
            return self._format_comparison_results(top_results)
        elif intent.query_type == QueryType.PRICE_QUERY:
            return self._format_price_results(top_results)
        elif intent.query_type == QueryType.FEATURE_QUERY:
            return self._format_feature_results(top_results)
        else:
            return self._format_standard_results(top_results)
    
    def _get_no_results_message(self, intent: QueryIntent) -> str:
        """Get appropriate no results message based on intent"""
        if intent.query_type == QueryType.LISTING:
            return "Hiện tại chưa có thông tin về danh sách sản phẩm/dịch vụ."
        elif intent.query_type == QueryType.PRICE_QUERY:
            return f"Không tìm thấy thông tin giá cho '{intent.original_query}'."
        elif intent.query_type == QueryType.EXCLUSION:
            return f"Không tìm thấy sản phẩm nào khác ngoài {', '.join(intent.exclusion_terms)}."
        else:
            return "Không tìm thấy thông tin sản phẩm phù hợp với yêu cầu của bạn."
    
    def _format_listing_results(self, results: List[ProductSearchResult]) -> str:
        """Format results for listing queries"""
        response = "Danh sách các sản phẩm/dịch vụ hiện có:\n\n"
        
        # Group products by type
        product_groups = {}
        mcp_results = []
        
        for result in results:
            if result.product:
                product_type = result.product.type or "Khác"
                if product_type not in product_groups:
                    product_groups[product_type] = []
                product_groups[product_type].append(result.product)
            else:
                mcp_results.append(result)
        
        # Display grouped products
        for product_type, products in product_groups.items():
            response += f"**{product_type.title()}:**\n"
            for i, product in enumerate(products, 1):
                response += f"   {i}. {product.name}\n"
                if product.price:
                    response += f"      Giá: {product.price}\n"
                if product.summary:
                    summary = product.summary[:100] + "..." if len(product.summary) > 100 else product.summary
                    response += f"      {summary}\n"
                response += "\n"
            response += "\n"
        
        # Add MCP results if any
        if mcp_results:
            response += "**Thông tin bổ sung:**\n"
            for result in mcp_results:
                response += f"• {result.snippet}\n\n"
        
        return response.strip()
    
    def _format_exclusion_results(self, results: List[ProductSearchResult], intent: QueryIntent) -> str:
        """Format results for exclusion queries"""
        response = f"Các sản phẩm khác (ngoài {', '.join(intent.exclusion_terms)}):\n\n"
        for i, result in enumerate(results, 1):
            if result.product:
                response += f"{i}. {result.product.name} - {result.product.type}\n"
                if result.product.price:
                    response += f"   Giá: {result.product.price}\n"
                if result.product.summary:
                    response += f"   {result.product.summary[:150]}...\n"
                response += "\n"
            else:
                response += f"{i}. {result.snippet}\n\n"
        return response.strip()
    
    def _format_comparison_results(self, results: List[ProductSearchResult]) -> str:
        """Format results for comparison queries"""
        response = "Thông tin so sánh các sản phẩm:\n\n"
        for i, result in enumerate(results, 1):
            if result.product:
                response += f"{i}. {result.product.name}\n"
                response += f"   Loại: {result.product.type}\n"
                response += f"   Giá: {result.product.price}\n"
                if result.product.working_time:
                    response += f"   Thời gian xử lý: {result.product.working_time} ngày\n"
                if result.product.feature:
                    response += f"   Tính năng: {result.product.feature[:100]}...\n"
                response += "\n"
            else:
                response += f"{i}. {result.snippet}\n\n"
        return response.strip()
    
    def _format_price_results(self, results: List[ProductSearchResult]) -> str:
        """Format results for price queries"""
        response = "Thông tin giá sản phẩm:\n\n"
        for i, result in enumerate(results, 1):
            if result.product:
                response += f"{i}. {result.product.name}\n"
                response += f"   Giá: {result.product.price}\n"
                if result.product.working_time:
                    response += f"   Thời gian xử lý: {result.product.working_time} ngày\n"
                response += "\n"
            else:
                response += f"{i}. {result.snippet}\n\n"
        return response.strip()
    
    def _format_feature_results(self, results: List[ProductSearchResult]) -> str:
        """Format results for feature queries"""
        response = "Thông tin tính năng sản phẩm:\n\n"
        for i, result in enumerate(results, 1):
            if result.product:
                response += f"{i}. {result.product.name}\n"
                if result.product.feature:
                    response += f"   Tính năng: {result.product.feature}\n"
                if result.product.feature_en:
                    response += f"   Features (EN): {result.product.feature_en}\n"
                if result.product.summary:
                    response += f"   Mô tả: {result.product.summary[:200]}...\n"
                response += "\n"
            else:
                response += f"{i}. {result.snippet}\n\n"
        return response.strip()
    
    def _format_standard_results(self, results: List[ProductSearchResult]) -> str:
        """Format results for standard queries"""
        response = ""
        for i, result in enumerate(results, 1):
            if result.product:
                response += f"{i}. {result.product.name}\n"
                response += f"Loại: {result.product.type}\n"
                if result.product.price:
                    response += f"Giá: {result.product.price}\n"
                if result.product.summary:
                    response += f"Mô tả: {result.product.summary[:200]}...\n"
                response += "\n"
            else:
                response += f"{i}. {result.snippet}\n\n"
        return response.strip()

    # === MAIN RETRIEVAL METHOD ===
    
    def _run(self, query: str, collection_name: str = None, max_results: int = 5) -> str:
        """Synchronous document retrieval."""
        collection = collection_name or self.default_collection
        return asyncio.run(self._arun(query, collection, max_results))
    
    async def _arun(self, query: str, collection_name: str = None, max_results: int = 5) -> str:
        """Enhanced asynchronous document retrieval with intent analysis."""
        if not self._is_initialized:
            return "Error: Enhanced product retriever client is not initialized."
        
        collection = collection_name or self.default_collection
        start_time = time.time()
        
        # Check cache first
        if self.enable_cache and self._query_cache:
            cache_key = hashlib.md5(f"{query}_{collection}_{max_results}".encode()).hexdigest()
            cached_result = self._query_cache.get(cache_key)
            if cached_result:
                logger.info(f"Cache hit for query: '{query}'")
                return cached_result
        
        try:
            # Analyze query intent
            intent = self._intent_analyzer.analyze(query)
            logger.info(f"Query intent: {intent.query_type.value}, terms: {intent.main_terms}")
            
            # Perform searches based on intent
            db_results = await self._database_search_with_intent(intent)
            mcp_results = await self._retrieve_from_mcp_server(intent, collection, max_results)
            
            # Merge results
            merged_results = self._merge_results(db_results, mcp_results)
            
            # Format response based on intent
            response = self._format_results_with_intent(merged_results, intent)
            
            # Cache the result
            if self.enable_cache and self._query_cache:
                self._query_cache.set(cache_key, response)
            
            retrieval_time = time.time() - start_time
            logger.info(f"Enhanced query processed in {retrieval_time:.2f}s: '{query}' -> {len(merged_results)} results")
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing enhanced query '{query}': {e}")
            # Fallback to basic database search
            fallback_result = await self._database_fallback_search(query, max_results)
            return fallback_result

    async def _database_fallback_search(self, query: str, max_results: int = 5) -> str:
        """Fallback to basic database search if other methods fail."""
        if not self.enable_database_sync:
            return "Database search not enabled and MCP server unavailable."
        
        try:
            async with AsyncSessionLocal() as db:
                query_lower = query.lower()
                
                # Create search conditions
                conditions = []
                for word in query_lower.split():
                    word_pattern = f"%{word}%"
                    conditions.extend([
                        ProductModel.name.ilike(word_pattern),
                        ProductModel.feature.ilike(word_pattern),
                        ProductModel.feature_en.ilike(word_pattern),
                        ProductModel.summary.ilike(word_pattern),
                        ProductModel.product_index.ilike(word_pattern),
                        ProductModel.type.ilike(word_pattern)
                    ])
                
                # Execute query
                result = await db.execute(
                    select(ProductModel)
                    .where(and_(
                        or_(*conditions),
                        ProductModel.is_active == "true"
                    ))
                    .limit(max_results)
                )
                
                products = result.scalars().all()
                
                if not products:
                    return f"Không tìm thấy sản phẩm nào phù hợp với truy vấn: {query}"
                
                # Format results
                response_parts = [f"Tìm thấy {len(products)} sản phẩm phù hợp:\n"]
                for i, product in enumerate(products, 1):
                    response_parts.append(f"{i}. {product.name}")
                    if product.type:
                        response_parts.append(f"   Loại: {product.type}")
                    if product.price:
                        response_parts.append(f"   Giá: {product.price}")
                    if product.summary:
                        summary = product.summary[:200] + "..." if len(product.summary) > 200 else product.summary
                        response_parts.append(f"   Mô tả: {summary}")
                    response_parts.append("")
                
                return "\n".join(response_parts)
                
        except Exception as e:
            logger.error(f"Database fallback search failed: {e}")
            return f"Lỗi khi tìm kiếm: {str(e)}"
    
    # === UTILITY METHODS ===
    
    def clear_cache(self):
        """Clear the query cache"""
        if self._query_cache:
            self._query_cache.cache.clear()
            self._query_cache.timestamps.clear()
            logger.info("Query cache cleared")
    
    def get_enhanced_stats(self) -> Dict[str, Any]:
        """Get enhanced statistics"""
        stats = {
            "initialized": self._is_initialized,
            "mcp_server_url": self.mcp_server_url,
            "watch_directory": str(self.watch_directory),
            "default_collection": self.default_collection,
            "database_sync_enabled": self.enable_database_sync,
            "cache_enabled": self.enable_cache,
            "tracked_files": len(self._document_registry),
        }
        
        if self._query_cache:
            stats["cache_size"] = len(self._query_cache.cache)
            stats["cache_max_size"] = self._query_cache.max_size
        
        return stats

    # === EXISTING METHODS REMAIN THE SAME ===
    
    async def upload_file(self, file_path: str, collection_name: str = None) -> str:
        """Manually upload a specific file to the server."""
        collection = collection_name or self.default_collection
        path = Path(file_path)
        
        if not path.exists():
            return f"Error: File not found: {file_path}"
        
        documents = self._load_and_split_file(path)
        if not documents:
            return f"Error: Could not process file: {file_path}"
        
        success = await self._upload_documents_to_server(documents, collection)
        if success:
            # Update registry
            file_hash = self._get_file_hash(path)
            if file_hash:
                self._document_registry[str(path)] = file_hash
                await self._save_document_registry()
            return f"Successfully uploaded {len(documents)} document chunks from {file_path}"
        else:
            return f"Failed to upload file: {file_path}"
    
    async def upload_text_documents(self, texts: List[str], collection_name: str = None, source_name: str = "manual_upload") -> str:
        """Manually upload text documents to the server."""
        collection = collection_name or self.default_collection
        
        if not texts:
            return "Error: No texts provided"
        
        # Create documents with metadata
        documents = []
        for i, text in enumerate(texts):
            metadata = {
                'source': source_name,
                'chunk_id': i,
                'timestamp': time.time(),
                'document_type': 'manual_upload'
            }
            doc = Document(page_content=text, metadata=metadata)
            documents.append(doc)
        
        success = await self._upload_documents_to_server(documents, collection)
        if success:
            return f"Successfully uploaded {len(texts)} text documents to collection '{collection}'"
        else:
            return f"Failed to upload text documents to collection '{collection}'"
    
    async def resync_database(self) -> str:
        """Manually trigger database resync."""
        if not self.enable_database_sync:
            return "Database sync is not enabled"
        
        try:
            await self._sync_database_products()
            return "Database resync completed successfully"
        except Exception as e:
            return f"Database resync failed: {str(e)}"
    
    def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up EnhancedProductRetrieverMCPClient resources...")
        
        if self._observer and self._observer.is_alive():
            self._observer.stop()
            self._observer.join()
            logger.info("Document watcher stopped")
        
        if self._thread_pool:
            self._thread_pool.shutdown(wait=False)
            logger.info("Thread pool shut down")
    
    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.cleanup()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


# === FACTORY FUNCTIONS ===

def create_enhanced_product_retriever_client(
    mcp_server_url: str, 
    watch_directory: str,
    collection_name: str = "product_docs",
    enable_database_sync: bool = True,
    enable_cache: bool = True
) -> EnhancedProductRetrieverMCPClient:
    """
    Factory function to create an EnhancedProductRetrieverMCPClient instance.
    
    Args:
        mcp_server_url: URL of the MCP server (e.g., "http://localhost:50051/sse")
        watch_directory: Local directory to watch for documents
        collection_name: Collection name for the vector database
        enable_database_sync: Whether to sync with product database
        enable_cache: Whether to enable query caching
    
    Returns:
        Configured EnhancedProductRetrieverMCPClient instance
    """
    return EnhancedProductRetrieverMCPClient(
        mcp_server_url=mcp_server_url,
        watch_directory=watch_directory,
        default_collection=collection_name,
        enable_database_sync=enable_database_sync,
        enable_cache=enable_cache
    )


# === EXAMPLE USAGE ===
if __name__ == "__main__":
    async def main():
        # Configuration
        SERVER_URL = "http://192.168.1.60:50051/sse"
        WATCH_DIR = "app/agents/retrievers/storages/products"
        COLLECTION = "product_knowledge"
        
        # Create the enhanced client
        client = create_enhanced_product_retriever_client(
            mcp_server_url=SERVER_URL,
            watch_directory=WATCH_DIR,
            collection_name=COLLECTION,
            enable_database_sync=True,
            enable_cache=True
        )
        
        # Wait for initialization
        await asyncio.sleep(3)
        
        # Test different query types
        test_queries = [
            "genemap adult giá bao nhiêu",           # Price query
            "tính năng của genemap adult",            # Feature query
            "so sánh genemap adult và genemap kid",   # Comparison query  
            "tất cả các sản phẩm dịch vụ",          # Listing query
            "các gói khác ngoài genemap adult",      # Exclusion query
            "thông tin về genemap adult"             # Simple search
        ]
        
        for query in test_queries:
            print(f"\n{'='*50}")
            print(f"Query: {query}")
            print('='*50)
            result = await client._arun(query)
            print(f"Result:\n{result}")
        
        # Test enhanced statistics
        stats = client.get_enhanced_stats()
        print(f"\nEnhanced Stats:\n{json.dumps(stats, indent=2)}")
        
        # Cleanup
        client.cleanup()
    
    asyncio.run(main())