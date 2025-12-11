import os
import json
import sys
import time
import hashlib
import re
import threading
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
from functools import lru_cache, wraps
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import asyncio
from collections import defaultdict
import weakref
import gc

from loguru import logger
from pydantic import Field, BaseModel, PrivateAttr
import chromadb
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# --- LangChain/Community Imports ---
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, JSONLoader, TextLoader
import asyncio
from app.agents.workflow.initalize import llm_instance, settings, agent_config
from app.agents.factory.tools.base import BaseAgentTool
sys.path.append(str(Path(__file__).parent.parent.parent))
from app.core.config import get_settings
# --- Imports for the Contextual Compression Pattern ---
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.embeddings import HuggingFaceEmbeddings

from app.utils.document_processor import markdown_splitter, remove_image_tags, DocumentCustomConverter
# Import torch for GPU memory management if available
TORCH_AVAILABLE = False
try:
    import torch
    TORCH_AVAILABLE = True
    logger.info("PyTorch is available for GPU memory management in genetic retriever")
except ImportError:
    logger.warning("PyTorch not available - some GPU memory management features will be disabled in genetic retriever")

# GPU Memory management constants
GPU_TTL_SECONDS = 300  # Time to live for GPU resources
GPU_MEMORY_PRESSURE_THRESHOLD = 0.8  # Threshold for releasing GPU resources

def gpu_management_decorator(func):
    """Decorator to manage GPU resources for retrieval methods."""
    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        try:
            # Make sure GPU components are initialized if needed
            await self.manually_ensure_gpu_ready()
            
            # Execute the original function
            result = await func(self, *args, **kwargs)
            return result
        except Exception as e:
            logger.error(f"Error in GPU-managed function {func.__name__}: {e}")
            raise
        finally:
            # Check memory pressure after function runs
            self._check_memory_pressure()
            
    return wrapper
class DocumentWatcher(FileSystemEventHandler):
    """File system watcher that triggers document loading and reloading."""
    def __init__(self, retriever_tool: 'GeneticRetrieverTool'):
        self.tool = retriever_tool

    def on_created(self, event):
        if not event.is_directory:
            logger.info(f"[Watcher] New file detected: {event.src_path}")
            time.sleep(1)
            self.tool.process_file_if_needed(Path(event.src_path))

    def on_modified(self, event):
        if not event.is_directory and "_registry.json" not in event.src_path:
            logger.info(f"[Watcher] File modified: {event.src_path}")
            time.sleep(1)
            self.tool.process_file_if_needed(Path(event.src_path))
            
    def on_deleted(self, event):
        if not event.is_directory:
            logger.info(f"[Watcher] File deleted: {event.src_path}")
            self.tool.remove_document_by_path(Path(event.src_path))


class PerformanceCache:
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
            
            # Check TTL
            if time.time() - self.timestamps[key] > self.ttl_seconds:
                del self.cache[key]
                del self.timestamps[key]
                return None
            
            return self.cache[key]
    
    def set(self, key: str, value: Any) -> None:
        with self._lock:
            # Evict oldest entries if cache is full
            if len(self.cache) >= self.max_size:
                oldest_key = min(self.timestamps.keys(), key=lambda k: self.timestamps[k])
                del self.cache[oldest_key]
                del self.timestamps[oldest_key]
            
            self.cache[key] = value
            self.timestamps[key] = time.time()
    
    def clear(self) -> None:
        with self._lock:
            self.cache.clear()
            self.timestamps.clear()

class RetrievedGeneticDocument(BaseModel):
    """Enhanced genetic document with performance optimizations"""
    content: str
    source: str
    relevance_score: float
    topic: str = ""
    category: str = ""
    _cached_hash: Optional[str] = None
    _processed_tokens: Optional[List[str]] = None
    
    @lru_cache(maxsize=128)
    def get_content_hash(self) -> str:
        """Cached content hash for deduplication"""
        if self._cached_hash is None:
            self._cached_hash = hashlib.md5(self.content.encode()).hexdigest()
        return self._cached_hash
    
    def get_processed_tokens(self) -> List[str]:
        """Get cached processed tokens for faster genetic text analysis"""
        if self._processed_tokens is None:
            # Enhanced tokenization for genetic/medical terms
            self._processed_tokens = re.findall(r'\b\w+\b', self.content.lower())
        return self._processed_tokens
    
    def calculate_genetic_relevance(self, query_tokens: List[str]) -> float:
        """Calculate relevance with genetic-specific scoring"""
        content_tokens = set(self.get_processed_tokens())
        query_token_set = set(query_tokens)
        
        # Genetic-specific term weights
        genetic_terms = {'dna', 'rna', 'protein', 'mutation', 'variant', 'allele', 'chromosome', 'genome'}
        medical_terms = {'disease', 'syndrome', 'disorder', 'treatment', 'therapy', 'drug', 'medication'}
        
        # Calculate weighted intersection
        intersection_score = 0.0
        for token in query_token_set & content_tokens:
            if token in genetic_terms:
                intersection_score += 2.0  # Higher weight for genetic terms
            elif token in medical_terms:
                intersection_score += 1.5  # Medium weight for medical terms
            else:
                intersection_score += 1.0  # Normal weight
        
        # Normalize by query length
        if len(query_token_set) > 0:
            normalized_score = intersection_score / len(query_token_set)
        else:
            normalized_score = 0.0
        
        # Combine with retrieval score (use relevance_score as fallback)
        retrieval_score = getattr(self, 'retrieval_score', self.relevance_score)
        self.relevance_score = (normalized_score * 0.7) + (retrieval_score * 0.3)
        return self.relevance_score


class GeneticRetrieverTool(BaseAgentTool):
    """
    An advanced retriever for genetic and biomedical information that automatically
    ingests and updates documents, and uses a hybrid search approach.
    """
    name: str = "genetic_retriever"
    description: str = "Retrieves and reranks genetic and biomedical information from a self-updating knowledge base."

    # --- Core Configuration (Pydantic Fields) ---
    collection_name: str = Field(description="Name for the ChromaDB collection.")
    watch_directory: Path = Field(description="Directory to watch for new/updated documents.")
    
    # --- Internal Components (Private Attributes) ---
    _vector_store: Chroma = PrivateAttr(default=None)
    _bm25_retriever: BM25Retriever = PrivateAttr(default=None)
    _embeddings: OllamaEmbeddings = PrivateAttr(default=None)
    _text_splitter: RecursiveCharacterTextSplitter = PrivateAttr(default=None)
    _observer: Observer = PrivateAttr(default=None)
    _document_registry: Dict[str, str] = PrivateAttr(default_factory=dict)
    _is_initialized: bool = PrivateAttr(default=False)
    
    # --- Performance Optimizations ---
    _query_cache: PerformanceCache = PrivateAttr(default=None)
    _document_cache: PerformanceCache = PrivateAttr(default=None)
    _embedding_cache: PerformanceCache = PrivateAttr(default=None)
    _thread_pool: ThreadPoolExecutor = PrivateAttr(default=None)
    _batch_queue: List[Tuple[str, asyncio.Future]] = PrivateAttr(default_factory=list)
    _batch_lock: threading.Lock = PrivateAttr(default=None)
    _last_bm25_update: float = PrivateAttr(default=0.0)
    _bm25_update_threshold: float = PrivateAttr(default=60.0)  # seconds
    
    # --- GPU Management ---
    _gpu_components_active: bool = PrivateAttr(default=False)
    _gpu_last_used: float = PrivateAttr(default=0.0)
    _gpu_ttl_seconds: int = PrivateAttr(default=GPU_TTL_SECONDS)
    _gpu_lock: threading.RLock = PrivateAttr(default=None)
    _final_retriever: Any = PrivateAttr(default=None)
    _was_pickled: bool = PrivateAttr(default=False)


    def __init__(self, watch_directory: str, collection_name: str = "genetic_docs", **kwargs):
        # Sửa lỗi Pydantic bằng cách truyền tất cả các trường vào super().__init__
        init_kwargs = kwargs.copy()
        init_kwargs['watch_directory'] = Path(watch_directory).resolve()
        init_kwargs['collection_name'] = self._sanitize_collection_name(collection_name)
        
        super().__init__(**init_kwargs)
        
        # Gán giá trị mặc định cho PrivateAttr
        self._vector_store: Optional[Chroma] = None
        self._bm25_retriever: Optional[BM25Retriever] = None
        self._embeddings: Optional[OllamaEmbeddings] = None
        self._text_splitter: Optional[RecursiveCharacterTextSplitter] = None
        self._observer: Optional[Observer] = None
        self._document_registry: Dict[str, str] = {}
        self._is_initialized: bool = False
        
        # Initialize performance components
        self._query_cache = PerformanceCache(max_size=500, ttl_seconds=300)
        self._document_cache = PerformanceCache(max_size=1000, ttl_seconds=600)
        self._embedding_cache = PerformanceCache(max_size=200, ttl_seconds=1800)
        self._thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="genetic")
        self._batch_queue = []
        self._batch_lock = threading.Lock()
        self._last_bm25_update = 0.0
        self._bm25_update_threshold = 60.0
        
        # GPU management attributes
        self._gpu_components_active = False
        self._gpu_last_used = 0.0
        self._gpu_ttl_seconds = GPU_TTL_SECONDS
        self._gpu_lock = threading.RLock()
        self._final_retriever = None
        self._was_pickled = False
        
        if not self._is_initialized:
            self._initialize_all()

    def _sanitize_collection_name(self, name: str) -> str:
        sanitized = re.sub(r'[^a-zA-Z0-9_.-]', '_', name)
        return sanitized[:63]

    def _initialize_all(self):
        logger.info(f"Initializing GeneticRetrieverTool for collection '{self.collection_name}'...")
        self.watch_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize with CPU components first for faster startup
        # GPU components will be initialized lazily when needed
        self._initialize_core_components(use_gpu=False)
        self._load_document_registry()
        
        # Only build a minimal CPU-based pipeline initially
        # The GPU pipeline will be built when needed
        self._build_retriever_pipeline(force_gpu=False)
        
        # Initialize document handling
        self._scan_and_process_all_files()
        self._start_document_watcher()
        
        self._is_initialized = True
        logger.info("GeneticRetrieverTool initialized successfully with CPU components. GPU components will be loaded when needed.")

    def _initialize_core_components(self, use_gpu: bool = False):
        """Initialize core components with optional GPU support."""
        logger.info(f"Initializing core components for genetic retriever (GPU: {use_gpu})")
        
        # Initialize embeddings based on GPU availability
        if use_gpu and TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                model_kwargs = {'device': 'cuda'}
                encode_kwargs = {'normalize_embeddings': True}
                self._embeddings = HuggingFaceEmbeddings(
                    model_name=settings.HF_EMBEDDING_MODEL, 
                    model_kwargs=model_kwargs, 
                    encode_kwargs=encode_kwargs
                )
                self._gpu_components_active = True
                self._gpu_last_used = time.time()
                logger.info(f"Using GPU for embeddings (model: {settings.HF_EMBEDDING_MODEL})")
            except Exception as e:
                logger.error(f"Failed to initialize GPU embeddings: {e}. Falling back to CPU.")
                model_kwargs = {'device': 'cpu'}
                encode_kwargs = {'normalize_embeddings': True}
                self._embeddings = HuggingFaceEmbeddings(
                    model_name=settings.HF_EMBEDDING_MODEL, 
                    model_kwargs=model_kwargs, 
                    encode_kwargs=encode_kwargs
                )
                self._gpu_components_active = False
        else:
            # CPU mode - either by choice or due to lack of GPU
            model_kwargs = {'device': 'cpu'}
            encode_kwargs = {'normalize_embeddings': True}
            self._embeddings = HuggingFaceEmbeddings(
                model_name=settings.HF_EMBEDDING_MODEL, 
                model_kwargs=model_kwargs, 
                encode_kwargs=encode_kwargs
            )
            self._gpu_components_active = False
            logger.info(f"Using CPU for embeddings (model: {settings.HF_EMBEDDING_MODEL})")
        
        # Initialize vector store
        persistent_client = chromadb.PersistentClient(path=str(Path(settings.VECTOR_STORE_BASE_DIR)))
        self._vector_store = Chroma(
            client=persistent_client,
            collection_name=self.collection_name,
            embedding_function=self._embeddings,
        )
        
        # Initialize text splitter
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=256,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Track GPU status if using GPU
        if use_gpu and TORCH_AVAILABLE and torch.cuda.is_available():
            self._gpu_components_active = True
            self._gpu_last_used = time.time()
            logger.info(f"Initialized core components on GPU for genetic retriever")
        else:
            self._gpu_components_active = False
            
        logger.info("Core components initialized.")

    @property
    def _registry_path(self) -> Path:
        return self.watch_directory / f"{self.collection_name}_registry.json"

    def _build_retriever_pipeline(self, force_gpu: bool = True):
        """Builds the final ContextualCompressionRetriever pipeline with optional GPU acceleration."""
        logger.info(f"Building retriever pipeline (Vector Search -> Reranker) with force_gpu={force_gpu}...")
        
        # Check if we should use GPU
        use_gpu = force_gpu and TORCH_AVAILABLE and torch.cuda.is_available()
        
        # A. Create the base retriever directly from the vector store.
        # This is the first stage: get a broad set of potentially relevant docs.
        base_retriever = self._vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 7},  # Get top 7 results from vector search
            return_source_documents=True
        )
        
        try:
            # B. Create the reranker model and compressor.
            # This is the second stage: accurately re-score the candidates.
            if use_gpu:
                model_kwargs = {'device': 'cuda'}
                logger.info("Using GPU for cross-encoder reranking")
            else:
                model_kwargs = {'device': 'cpu'}
                logger.info("Using CPU for cross-encoder reranking")
                
            model = HuggingFaceCrossEncoder(model_name=settings.HF_RERANKER_MODEL, model_kwargs=model_kwargs)
            compressor = CrossEncoderReranker(model=model, top_n=3)
            
            # C. Create the final compression retriever.
            self._final_retriever = ContextualCompressionRetriever(
                base_compressor=compressor, 
                base_retriever=base_retriever
            )
            
            # Update GPU status
            if use_gpu:
                self._gpu_components_active = True
                self._gpu_last_used = time.time()
                logger.info("Retriever pipeline built successfully on GPU")
            else:
                logger.info("Retriever pipeline built successfully on CPU")
                
        except Exception as e:
            logger.error(f"Error building retriever pipeline: {e}. Retrieval will not work.")
            self._final_retriever = None
            
    def _run(self, query: str) -> str:
        """Synchronously retrieves and reranks documents for the customer."""
        try:
            # Ensure GPU components are ready if needed
            self._ensure_gpu_components()
            
            if not self._final_retriever:
                logger.info("GeneticRetriever is not initialized or retriever pipeline is not built.")
                return "Không tìm thấy kết quả phù hợp với truy vấn của bạn"
            
            try:
                # Perform retrieval
                compressed_docs = self._final_retriever.invoke(query)
                
                # Check if we got any results
                if not compressed_docs:
                    logger.warning(f"No relevant documents found for query: '{query}'")
                    return "Không có kết quả phù hợp từ dữ liệu của bạn với truy vấn của bạn"
                
                logger.info(f"Retrieved {len(compressed_docs)} documents for query: '{query}'")
                
                # Format results
                return "\n\n".join(
                    f"Source: {doc.metadata.get('source', 'unknown')}\nContent: {doc.page_content}"
                    for doc in compressed_docs
                )
            except Exception as e:
                logger.error(f"Error during retrieval: {e}")
                return "Công cụ GeneticRetriever gặp lỗi trong quá trình truy xuất dữ liệu. Vui lòng thử lại sau."
        finally:
            # Check memory pressure and release resources if needed
            self._check_memory_pressure()
            
            # Update last used time to prevent premature release
            if getattr(self, '_gpu_components_active', False):
                self._gpu_last_used = time.time()
        

    @gpu_management_decorator
    async def _arun(self, query: str) -> str:
        """Asynchronously retrieves and reranks documents for the customer."""
        try:
            if not self._final_retriever:
                logger.info("GeneticRetriever is not initialized or retriever pipeline is not built.")
                return "Công cụ GeneticRetriever gặp lỗi. Vui lòng thử lại sau."
            try:
                # Use the async invoke method for the final retriever
                compressed_docs = await self._final_retriever.ainvoke(query)
            except Exception as e:
                logger.error(f"Error during async retrieval: {e}")
                return "Công cụ GeneticRetriever gặp lỗi trong quá trình truy xuất dữ liệu. Vui lòng thử lại sau."
    
            if not compressed_docs:
                logger.warning(f"No relevant documents found for query: '{query}'")
                return "Không có kết quả phù hợp từ dữ liệu của bạn với truy vấn của bạn"
            logger.info(f"Retrieved {len(compressed_docs)} documents for query: '{query}'")
            return "\n\n".join(
                f"Source: {doc.metadata.get('source', 'unknown')}\nContent: {doc.page_content}"
                for doc in compressed_docs
            )
        finally:
            # Check memory pressure and release resources if needed
            self._check_memory_pressure()
            
            # Update last used time to prevent premature release
            if getattr(self, '_gpu_components_active', False):
                self._gpu_last_used = time.time()
        
        
    def run(self, query: str) -> str:
        """Synchronous entry point for running the retriever with GPU management."""
        if not self._is_initialized:
            self._initialize_all()
        # Ensure GPU components are ready if needed
        self._ensure_gpu_components()
        return self._run(query)
        
    async def arun(self, query: str) -> str:
        """Asynchronous entry point for running the retriever with GPU management."""
        if not self._is_initialized:
            self._initialize_all()
        logger.info(f"Asynchronously running genetic retriever for query: '{query}'")
        return await self._arun(query)
    def _load_document_registry(self):
        if self._registry_path.exists():
            try:
                with open(self._registry_path, 'r', encoding='utf-8') as f:
                    self._document_registry = json.load(f)
                logger.info(f"Loaded {len(self._document_registry)} entries from registry.")
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Failed to load registry: {e}. Starting fresh.")
                self._document_registry = {}

    def _save_document_registry(self):
        try:
            with open(self._registry_path, 'w', encoding='utf-8') as f:
                json.dump(self._document_registry, f, indent=2)
            logger.debug(f"Registry saved with {len(self._document_registry)} entries.")
        except IOError as e:
            logger.error(f"Failed to save registry: {e}")

    def _scan_and_process_all_files(self):
        logger.info(f"Performing initial scan of directory: {self.watch_directory}")
        current_files = set()
        for file_path in self.watch_directory.rglob('*'):
            if file_path.is_file():
                current_files.add(str(file_path))
                self.process_file_if_needed(file_path)
        
        registered_files = set(self._document_registry.keys())
        deleted_files = registered_files - current_files
        for file_path_str in deleted_files:
            self.remove_document_by_path(Path(file_path_str))
        self._save_document_registry()

    def process_file_if_needed(self, file_path: Path):
        if "_registry.json" in file_path.name: return
        current_hash = self._get_file_hash(file_path)
        if not current_hash: return
        if current_hash != self._document_registry.get(str(file_path)):
            logger.info(f"Change detected for '{file_path.name}'. Processing...")
            self._reload_document(file_path)

    def _get_file_hash(self, file_path: Path) -> Optional[str]:
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except IOError as e:
            logger.error(f"Could not read file for hashing: {file_path}. Error: {e}")
            return None

    def _start_document_watcher(self):
        if self._observer: return
        self._observer = Observer()
        event_handler = DocumentWatcher(self)
        self._observer.schedule(event_handler, str(self.watch_directory), recursive=True)
        watcher_thread = threading.Thread(target=self._observer.start, daemon=True)
        watcher_thread.start()
        logger.info(f"Started document watcher on '{self.watch_directory}'.")

    def stop_document_watcher(self):
        if self._observer and self._observer.is_alive():
            self._observer.stop()
            self._observer.join()
            logger.info("Document watcher stopped.")
            self._observer = None

    def _reload_document(self, file_path: Path):
        self._delete_documents_by_source(file_path.name)
        documents = self._load_and_split_file(file_path)
        logger.info(f"Reloaded {len(documents)} documents from '{file_path.name}'")
        if documents:
            logger.info(f"Reloading {len(documents)} documents from '{file_path.name}'...")
            self._add_documents_to_store(documents)
            new_hash = self._get_file_hash(file_path)
            if new_hash:
                self._document_registry[str(file_path)] = new_hash
                self._save_document_registry()
                logger.info(f"Successfully reloaded and registered '{file_path.name}'.")

    def remove_document_by_path(self, file_path: Path):
        path_str = str(file_path)
        if path_str in self._document_registry:
            logger.info(f"Removing document '{file_path.name}' from knowledge base.")
            self._delete_documents_by_source(file_path.name)
            del self._document_registry[path_str]
            self._save_document_registry()
        
    def _load_and_split_file(self, file_path: Path) -> List[Document]:
        loader_map = {'.pdf': DocumentCustomConverter, '.csv': CSVLoader, '.json': JSONLoader, '.txt': TextLoader}
        loader_class = loader_map.get(file_path.suffix.lower())
        if not loader_class: return []
        try:
            loader = loader_class(str(file_path))
            raw_docs = loader.load()
            # logger.info(f"Loaded {len(raw_docs)} documents from {file_path.name} with data: {raw_docs[0][:100]}...")  # Log first 100 characters of the first document
            if "pdf" in file_path.suffix.lower():
                logger.info(f"Processing PDF file: {file_path.name}")
                # split markdown file
                cleaned_text = remove_image_tags(raw_docs[0])
                logger.info(f"Cleaned text length: {len(cleaned_text)} characters")
                try: 
                    raw_docs = markdown_splitter(cleaned_text)
                except Exception as e:
                    logger.error(f"Error splitting markdown from PDF file {file_path.name}: {e}")
                logger.info(f"Split {len(raw_docs)} sections from PDF file: {file_path.name}")
            try:   
                split_docs = self._text_splitter.split_documents(raw_docs)
                for doc in split_docs:
                    doc.metadata['source'] = file_path.name
                return split_docs
            except Exception as e:
                logger.error(f"Error splitting documents from {file_path.name}: {e}")
                return []
        except Exception: return []

    def _add_documents_to_store(self, docs: List[Document]):
        if not docs: return
        self._vector_store.add_documents(docs)
        # self._update_bm25_retriever()

    def _delete_documents_by_source(self, source_filename: str):
        try:
            existing_ids = self._vector_store.get(where={"source": source_filename})['ids']
            if existing_ids:
                logger.info(f"Deleting {len(existing_ids)} old chunks for source '{source_filename}'...")
                self._vector_store.delete(ids=existing_ids)
                self._update_bm25_retriever()
        except Exception as e:
            logger.error(f"Failed to delete documents for source '{source_filename}': {e}")
            
    def _update_bm25_retriever(self, force: bool = False):
        """Update BM25 retriever with throttling to avoid excessive rebuilds"""
        current_time = time.time()
        
        # Throttle updates unless forced
        if not force and (current_time - self._last_bm25_update) < self._bm25_update_threshold:
            return
            
        try:
            # Check cache first
            cache_key = f"bm25_docs_genetic"
            cached_docs = self._document_cache.get(cache_key)
            
            if cached_docs is None:
                all_docs = self._vector_store.get(include=["metadatas", "documents"])
                if all_docs and all_docs['documents']:
                    cached_docs = [Document(page_content=doc, metadata=meta)
                                 for doc, meta in zip(all_docs['documents'], all_docs['metadatas'])]
                    self._document_cache.set(cache_key, cached_docs)
                else:
                    cached_docs = []
            
            if cached_docs:
                self._bm25_retriever = BM25Retriever.from_documents(cached_docs)
                self._bm25_retriever.k = 10
                self._last_bm25_update = current_time
                logger.debug(f"BM25 retriever updated with {len(cached_docs)} documents")
            else:
                self._bm25_retriever = None
                
        except Exception as e:
            logger.error(f"Failed to update BM25 retriever: {e}")
    
    def _get_vector_results(self, query: str) -> List[Tuple[Document, float]]:
        """Get vector search results with timeout"""
        try:
            return self._vector_store.similarity_search_with_relevance_scores(query, k=5)
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    def _get_bm25_results(self, query: str) -> List[Document]:
        """Get BM25 search results with timeout"""
        try:
            if self._bm25_retriever:
                return self._bm25_retriever.get_relevant_documents(query)
            return []
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []

    def retrieve_documents(self, query: str, use_cache: bool = True) -> List[str]:
        """Retrieve documents with performance optimizations"""
        if not self._is_initialized:
            return ["Error: Genetic retriever is not initialized."]
        
        start_time = time.time()
        
        # Check cache first
        cache_key = f"genetic_query_{hashlib.md5(query.encode()).hexdigest()}"
        if use_cache:
            cached_result = self._query_cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for query: {query[:50]}...")
                return cached_result
        
        try:
            # Parallel search execution
            future_vector = self._thread_pool.submit(self._get_vector_results, query)
            future_bm25 = self._thread_pool.submit(self._get_bm25_results, query)
            
            # Collect results with timeout
            vector_results = future_vector.result(timeout=10)
            bm25_results = future_bm25.result(timeout=10)
            
            # Process results with deduplication by content hash
            hybrid_results = {}
            seen_hashes = set()
            
            # Process vector results
            for doc, score in vector_results:
                genetic_doc = RetrievedGeneticDocument(
                    content=doc.page_content,
                    source=doc.metadata.get("source", "unknown"),
                    relevance_score=score
                )
                content_hash = genetic_doc.get_content_hash()
                if content_hash not in seen_hashes:
                    hybrid_results[content_hash] = genetic_doc
                    seen_hashes.add(content_hash)
            logger.info(f"Vector search results: {vector_results}")
            
            # Process BM25 results
            for doc in bm25_results:
                genetic_doc = RetrievedGeneticDocument(
                    content=doc.page_content,
                    source=doc.metadata.get("source", "unknown"),
                    relevance_score=0.5
                )
                content_hash = genetic_doc.get_content_hash()
                if content_hash not in seen_hashes:
                    hybrid_results[content_hash] = genetic_doc
                    seen_hashes.add(content_hash)
            
            # Enhanced genetic-specific relevance scoring
            query_tokens = re.findall(r'\b\w+\b', query.lower())
            for genetic_doc in hybrid_results.values():
                genetic_doc.calculate_genetic_relevance(query_tokens)
            
            # Sort by relevance score
            sorted_docs = sorted(hybrid_results.values(), key=lambda x: x.relevance_score, reverse=True)
            
            if not sorted_docs:
                result = [f"Không tìm thấy thông tin di truyền phù hợp cho: '{query}'"]
            else:
                result = [f"[{doc.source}] {doc.content}" for doc in sorted_docs[:10]]
            
            # Cache the result
            if use_cache:
                self._query_cache.set(cache_key, result)
            
            processing_time = time.time() - start_time
            logger.info(f"Genetic retrieval completed in {processing_time:.3f}s for query: {query[:50]}...")
            logger.info(f"Result: {result}")    
            return result
            
        except Exception as e:
            logger.error(f"Error in genetic document retrieval: {e}")
            return [f"Error retrieving genetic documents: {str(e)}"]

    @gpu_management_decorator
    async def batch_retrieve_documents(self, queries: List[str]) -> List[List[str]]:
        """Process multiple queries in parallel with error handling and GPU optimization."""
        if not queries:
            return []
        
        if not self._is_initialized:
            return [["Error: Genetic retriever is not initialized."] for _ in queries]
        
        start_time = time.time()
        logger.info(f"Starting batch retrieval for {len(queries)} queries")
        
        # Ensure GPU components are ready for batch processing
        await self.manually_ensure_gpu_ready()
        
        async def process_single_query(query: str) -> List[str]:
            try:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(self._thread_pool, self.retrieve_documents, query)
            except Exception as e:
                logger.error(f"Error processing query '{query[:50]}...': {e}")
                return [f"Error processing query: {str(e)}"]
        
        try:
            # Process all queries concurrently
            tasks = [process_single_query(query) for query in queries]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions in results
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Exception in batch query {i}: {result}")
                    processed_results.append([f"Error: {str(result)}"])
                else:
                    processed_results.append(result)
            
            processing_time = time.time() - start_time
            logger.info(f"Batch retrieval completed in {processing_time:.2f}s for {len(queries)} queries")
            
            return processed_results
        finally:
            # Update last used time to prevent premature release
            if getattr(self, '_gpu_components_active', False):
                self._gpu_last_used = time.time()
            
            # Check memory pressure after batch processing
            self._check_memory_pressure()
    
    def clear_caches(self) -> None:
        """Clear all caches"""
        self._query_cache.clear()
        self._document_cache.clear()
        self._embedding_cache.clear()
        logger.info("All genetic retriever caches cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring"""
        return {
            "query_cache_size": len(self._query_cache.cache),
            "document_cache_size": len(self._document_cache.cache),
            "embedding_cache_size": len(self._embedding_cache.cache),
            "query_cache_max": self._query_cache.max_size,
            "document_cache_max": self._document_cache.max_size,
            "embedding_cache_max": self._embedding_cache.max_size,
            "last_bm25_update": self._last_bm25_update,
            "bm25_update_threshold": self._bm25_update_threshold
        }
    
    # These methods are already defined above
    
    def cleanup(self) -> None:
        """Clean up resources"""
        try:
            if self._thread_pool:
                self._thread_pool.shutdown(wait=True)
                logger.info("Thread pool shut down")
            
            self.clear_caches()
            
            if self._observer:
                self.stop_document_watcher()
                
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    # def _run(self, query: str) -> str:
    #     results = self.retrieve_documents(query)
    #     return "\n\n".join(results)

    # async def _arun(self, query: str) -> str:
    #     loop = asyncio.get_event_loop()
    #     return await loop.run_in_executor(None, self._run, query)
    
    def __getstate__(self):
        """Custom pickle method to handle unpicklable attributes."""
        state = self.__dict__.copy()
        # Don't pickle these objects
        for key in ['_thread_pool', '_observer', '_vector_store', '_bm25_retriever', 
                   '_final_retriever', '_embeddings', '_gpu_lock']:
            if key in state:
                state[key] = None
                
        # Mark that this object was pickled
        state['_private_attributes']['_was_pickled'] = True
        return state
    
    def __setstate__(self, state):
        """Custom unpickle method to restore state."""
        self.__dict__.update(state)
        # Reset lock
        self._gpu_lock = threading.RLock()

    def _release_gpu_resources(self):
        """Releases GPU resources to free memory when not in use."""
        with self._gpu_lock:
            if not self._gpu_components_active:
                return
            
            logger.info("Releasing GPU resources for genetic retriever...")
            
            # Release cross encoder and embeddings model
            if hasattr(self, '_final_retriever') and self._final_retriever:
                if hasattr(self._final_retriever.base_compressor, 'model'):
                    self._final_retriever.base_compressor.model = None
            
            # Recreate with CPU components
            self._initialize_core_components(use_gpu=False)
            self._build_retriever_pipeline(force_gpu=False)
            
            # Force garbage collection
            if TORCH_AVAILABLE:
                try:
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()
                except Exception as e:
                    logger.error(f"Error clearing GPU memory: {e}")
            
            self._gpu_components_active = False
            logger.info("GPU resources released for genetic retriever.")
    
    def _ensure_gpu_components(self):
        """Ensures that GPU components are loaded when needed."""
        with self._gpu_lock:
            current_time = time.time()
            
            # Check if components are active and not timed out
            if (getattr(self, '_gpu_components_active', False) and 
                getattr(self, '_final_retriever', None) is not None and 
                (current_time - getattr(self, '_gpu_last_used', 0)) < getattr(self, '_gpu_ttl_seconds', 300)):
                # Update last used time
                self._gpu_last_used = current_time
                return True
            
            # If we get here, we need to initialize or reinitialize GPU components
            if not TORCH_AVAILABLE or not torch.cuda.is_available():
                logger.warning("GPU requested but not available. Using CPU instead.")
                return False
            
            logger.info("Loading GPU components for genetic retriever...")
            self._initialize_core_components(use_gpu=True)
            self._build_retriever_pipeline(force_gpu=True)
            self._gpu_components_active = True
            self._gpu_last_used = time.time()
            logger.info("GPU components loaded for genetic retriever.")
            return True
    
    def _check_memory_pressure(self):
        """Checks GPU memory pressure and releases resources if needed."""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return
            
        current_time = time.time()
        time_since_used = current_time - self._gpu_last_used
        
        # Release if idle for too long
        if time_since_used > self._gpu_ttl_seconds:
            logger.info(f"GPU components idle for {time_since_used:.1f}s, releasing resources.")
            self._release_gpu_resources()
            return
            
        # Check memory pressure
        try:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
                memory_reserved = torch.cuda.memory_reserved() / (1024 ** 3)  # GB
                
                # If memory pressure is high, release resources
                if memory_allocated > GPU_MEMORY_PRESSURE_THRESHOLD * memory_reserved:
                    logger.warning(f"High GPU memory pressure detected: {memory_allocated:.2f}GB/{memory_reserved:.2f}GB")
                    self._release_gpu_resources()
        except Exception as e:
            logger.error(f"Error checking GPU memory pressure: {e}")
    
    def get_gpu_memory_stats(self):
        """Get current GPU memory stats."""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return {"error": "CUDA not available"}
            
        try:
            stats = {
                "active": self._gpu_components_active,
                "last_used_seconds_ago": time.time() - self._gpu_last_used,
                "allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024,
                "reserved_mb": torch.cuda.memory_reserved() / 1024 / 1024,
                "max_mb": torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
            }
            return stats
        except Exception as e:
            return {"error": str(e)}
            
    async def manually_ensure_gpu_ready(self):
        """
        Manually ensure GPU resources are initialized.
        Use this before expected high-volume query periods.
        """
        logger.info("Manually ensuring GPU components are ready for genetic retriever")
        return self._ensure_gpu_components()
        
    async def manually_release_gpu(self):
        """
        Manually release GPU resources.
        Use this after periods of inactivity to free up GPU memory.
        """
        logger.info("Manually releasing GPU resources for genetic retriever")
        self._release_gpu_resources()
        return {"status": "success", "message": "GPU resources released successfully for genetic retriever"}
    
    def __del__(self):
        self.cleanup()


# Example usage
if __name__ == '__main__':
    async def main():
        DATA_DIR = Path("./genetic_data_test")
        DATA_DIR.mkdir(exist_ok=True, parents=True)
        logger.info(f"Using data directory: {DATA_DIR.resolve()}")

        # Create test files
        with open(DATA_DIR / "brca1_gene.txt", "w", encoding="utf-8") as f:
            f.write("Gen BRCA1 là một gen ức chế khối u. Đột biến ở gen này làm tăng nguy cơ ung thư vú và buồng trứng.")
        with open(DATA_DIR / "intelligence_types.csv", "w", encoding="utf-8") as f:
            f.write("intelligence_type,description\n")
            f.write("Thông minh logic-toán học,Khả năng suy luận, giải quyết vấn đề và tính toán.\n")

        # Initialize the tool
        genetic_tool = GeneticRetrieverTool(watch_directory=str(DATA_DIR), collection_name="genetic_test_collection")
        
        # Test initial retrieval
        query = "nguy cơ ung thư từ gen BRCA1"
        print(f"\n--- Testing query: '{query}' ---")
        print(await genetic_tool._arun(query))

        # Test another query
        query2 = "trí thông minh logic"
        print(f"\n--- Testing query: '{query2}' ---")
        print(await genetic_tool._arun(query2))
        
        # Test auto-update (new file)
        print("\n--- Testing auto-update. Adding a new file... ---")
        time.sleep(2)
        with open(DATA_DIR / "mash_system.json", "w", encoding="utf-8") as f:
            json.dump({"systemName": "MASH", "description": "Hệ thống phân tích dữ liệu y sinh của GeneStory."}, f, ensure_ascii=False)
        time.sleep(5)

        query3 = "hệ thống MASH"
        print(f"\n--- Testing query after new file: '{query3}' ---")
        print(await genetic_tool._arun(query3))

    asyncio.run(main())