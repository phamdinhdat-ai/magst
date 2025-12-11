import os
import json
import sys
import time
import hashlib
import re
import gc
import threading
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
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
# --- Tool Imports ---
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.embeddings import HuggingFaceEmbeddings

from app.utils.document_processor import markdown_splitter, remove_image_tags, DocumentCustomConverter




try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. GPU memory management disabled.")
    
    
def gpu_management_decorator(func):
    """Decorator to manage GPU resources for retrieval methods."""
    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        # Check if object was pickled/unpickled
        if hasattr(self, '_was_pickled') and getattr(self, '_was_pickled', False):
            logger.info("Detected unpickled instance - reinitializing core components")
            try:
                # Reinitialize core components if needed
                if not hasattr(self, '_vector_store') or self._vector_store is None:
                    # Need to reinitialize everything
                    self._initialize_all()
            except Exception as e:
                logger.error(f"Failed to reinitialize after unpickling: {e}")
        
        # Try to ensure GPU components are available
        try:
            if not hasattr(self, '_ensure_gpu_components') or not self._ensure_gpu_components():
                logger.warning("Failed to ensure GPU components availability")
                # Try to fall back to CPU if possible
                if not hasattr(self, '_final_retriever') or self._final_retriever is None:
                    try:
                        # Initialize with CPU as fallback
                        self._initialize_core_components(use_gpu=False)
                        self._build_retriever_pipeline(force_gpu=False)
                    except Exception as e:
                        logger.error(f"CPU fallback initialization failed: {e}")
                        return "Retrieval tool temporarily unavailable. Please try again in a moment."
        except Exception as gpu_err:
            logger.error(f"Error ensuring GPU components: {gpu_err}")
            # Continue with potentially limited functionality
        
        try:
            # Run the retrieval function
            result = await func(self, *args, **kwargs)
            return result
        except Exception as e:
            logger.error(f"Error in retrieval function: {str(e)}")
            return f"Error retrieving information: {str(e)}"
        finally:
            # Update last used time for GPU components
            if hasattr(self, '_gpu_components_active') and getattr(self, '_gpu_components_active', False):
                self._gpu_last_used = time.time()
            
            # Check memory pressure
            try:
                if hasattr(self, '_check_memory_pressure'):
                    self._check_memory_pressure()
            except Exception as e:
                logger.error(f"Error checking memory pressure: {e}")
            
    return wrapper    
class DocumentWatcher(FileSystemEventHandler):
    """File system watcher that triggers document loading and reloading."""
    def __init__(self, retriever_tool: 'MedicalRetrieverTool'):
        self.tool = retriever_tool

    def on_created(self, event):
        if not event.is_directory:
            logger.info(f"New file detected: {event.src_path}")
            # Add delay to ensure file is fully written
            time.sleep(1)
            self.tool._process_file_if_needed(Path(event.src_path))

    def on_modified(self, event):
        if not event.is_directory and "_registry.json" not in event.src_path:
            logger.info(f"File modified: {event.src_path}")
            time.sleep(1)
            self.tool._process_file_if_needed(Path(event.src_path))
            
    def on_deleted(self, event):
        if not event.is_directory:
            logger.info(f"File deleted: {event.src_path}")
            self.tool._remove_document_by_path(Path(event.src_path))

class PerformanceCache:
    """High-performance cache for medical retriever with TTL and memory management"""
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.timestamps = {}
        self.lock = threading.RLock()
    
    def get(self, key: str):
        with self.lock:
            if key not in self.cache:
                return None
            
            # Check TTL
            if time.time() - self.timestamps[key] > self.ttl_seconds:
                del self.cache[key]
                del self.timestamps[key]
                return None
            
            return self.cache[key]
    
    def set(self, key: str, value):
        with self.lock:
            # Evict oldest entries if cache is full
            if len(self.cache) >= self.max_size:
                oldest_key = min(self.timestamps.keys(), key=lambda k: self.timestamps[k])
                del self.cache[oldest_key]
                del self.timestamps[oldest_key]
            
            self.cache[key] = value
            self.timestamps[key] = time.time()
    
    def clear(self):
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()

class RetrievedMedicalDocument(BaseModel):
    """Enhanced medical document with performance optimizations"""
    content: str
    source: str
    relevance_score: float
    condition_name: str = ""
    category: str = ""
    risk_level: str = ""
    _cached_hash: Optional[str] = None
    _processed_tokens: Optional[List[str]] = None
    
    def get_content_hash(self) -> str:
        """Cached content hash for deduplication"""
        if self._cached_hash is None:
            self._cached_hash = hashlib.md5(self.content.encode()).hexdigest()
        return self._cached_hash
    
    def get_processed_tokens(self) -> List[str]:
        """Get cached processed tokens for faster medical text analysis"""
        if self._processed_tokens is None:
            # Enhanced tokenization for medical terms
            self._processed_tokens = re.findall(r'\b\w+\b', self.content.lower())
        return self._processed_tokens
    
    def calculate_medical_relevance(self, query_tokens: List[str]) -> float:
        """Calculate relevance with medical-specific scoring"""
        content_tokens = set(self.get_processed_tokens())
        query_token_set = set(query_tokens)
        
        # Medical-specific term weights
        medical_terms = {'disease', 'symptom', 'treatment', 'diagnosis', 'medicine', 'drug', 'therapy', 'condition', 'syndrome', 'disorder'}
        clinical_terms = {'patient', 'clinical', 'hospital', 'doctor', 'physician', 'nurse', 'medical', 'health', 'care', 'examination'}
        anatomy_terms = {'heart', 'lung', 'brain', 'liver', 'kidney', 'blood', 'bone', 'muscle', 'nerve', 'organ'}
        
        # Calculate weighted intersection
        intersection_score = 0.0
        for token in query_token_set & content_tokens:
            if token in medical_terms:
                intersection_score += 2.5  # Highest weight for medical terms
            elif token in clinical_terms:
                intersection_score += 2.0  # High weight for clinical terms
            elif token in anatomy_terms:
                intersection_score += 1.8  # High weight for anatomy terms
            else:
                intersection_score += 1.0  # Normal weight
        
        # Normalize by query length
        if len(query_token_set) > 0:
            normalized_score = intersection_score / len(query_token_set)
        else:
            normalized_score = 0.0
        
        # Combine with retrieval score
        retrieval_score = getattr(self, 'retrieval_score', self.relevance_score)
        self.relevance_score = (normalized_score * 0.7) + (retrieval_score * 0.3)
        return self.relevance_score


class MedicalRetrieverTool(BaseAgentTool):
    """
    Một công cụ tìm kiếm thông tin y tế tiên tiến, tự động thu thập và cập nhật tài liệu từ một thư mục cụ thể, 
    lưu trữ chúng trong cơ sở dữ liệu vector 
    và sử dụng phương pháp tìm kiếm kết hợp để có kết quả có độ liên quan cao.
    """
    name: str = "medical_retriever"
    description: str = "Retrieves and reranks medical information from a self-updating knowledge base."

    # --- Core Configuration (Pydantic Fields) ---
    collection_name: str = Field(description="Name for the ChromaDB collection.")
    watch_directory: Path = Field(description="Directory to watch for new/updated documents.")
    
    # --- Internal Components (Private Attributes for Pydantic v2) ---
    _vector_store: Chroma = PrivateAttr(default=None)
    _bm25_retriever: BM25Retriever = PrivateAttr(default=None)
    _embeddings: OllamaEmbeddings = PrivateAttr(default=None)
    _text_splitter: RecursiveCharacterTextSplitter = PrivateAttr(default=None)
    _observer: Observer = PrivateAttr(default=None)
    _document_registry: Dict[str, str] = PrivateAttr(default_factory=dict)
    _is_initialized: bool = PrivateAttr(default=False)
    
    # --- Performance Caches ---
    _query_cache: PerformanceCache = PrivateAttr(default=None)
    _document_cache: PerformanceCache = PrivateAttr(default=None)
    _embedding_cache: PerformanceCache = PrivateAttr(default=None)
    
    # --- Parallel Processing ---
    _thread_pool: ThreadPoolExecutor = PrivateAttr(default=None)
    _batch_queue: List = PrivateAttr(default_factory=list)
    _batch_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)
    
    # --- BM25 Throttling ---
    _last_bm25_update: float = PrivateAttr(default=0.0)
    _bm25_update_threshold: float = PrivateAttr(default=60.0)


    # GPU Memory Management
    _gpu_components_active: bool = PrivateAttr(default=False)
    _gpu_last_used: float = PrivateAttr(default=0.0)
    _gpu_ttl_seconds: float = PrivateAttr(default=300.0)  # 5 minutes before release
    _gpu_lock: threading.RLock = PrivateAttr(default=None)  # Instance lock for GPU operations
    

    def __init__(self, watch_directory: str, collection_name: str = "medical_docs", reranker_device: str = 'cuda', **kwargs):
        # Sửa lỗi Pydantic bằng cách truyền tất cả các trường vào super().__init__
        init_kwargs = kwargs.copy()
        init_kwargs['watch_directory'] = Path(watch_directory).resolve()
        init_kwargs['collection_name'] = self._sanitize_collection_name(collection_name)
        
        super().__init__(**init_kwargs)
        self._vector_store = None
        self._bm25_retriever = None
        self._embeddings = None
        self._text_splitter = None
        self._observer = None
        self._document_registry = {}
        self._is_initialized = False
        
        # Initialize performance components
        self._query_cache = PerformanceCache(max_size=500, ttl_seconds=300)
        self._document_cache = PerformanceCache(max_size=1000, ttl_seconds=600)
        self._embedding_cache = PerformanceCache(max_size=200, ttl_seconds=1800)
        self._thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="medical")
        self._batch_queue = []
        self._batch_lock = threading.Lock()
        self._last_bm25_update = 0.0
        self._bm25_update_threshold = 60.0
        self._reranker_device = reranker_device
        if not self._is_initialized:
            self._initialize_all()

    def _sanitize_collection_name(self, name: str) -> str:
        sanitized = re.sub(r'[^a-zA-Z0-9_.-]', '_', name)
        return sanitized[:63]

    def _initialize_all(self):
        """Orchestrates the full initialization process."""
        logger.info(f"Initializing MedicalRetrieverTool for collection '{self.collection_name}'...")
        self.watch_directory.mkdir(parents=True, exist_ok=True)
        self._gpu_lock = threading.RLock()
        
        # Initialize performance caches
        self._query_cache = PerformanceCache(max_size=500, ttl_seconds=300)  # 5 min TTL
        self._document_cache = PerformanceCache(max_size=1000, ttl_seconds=600)  # 10 min TTL
        self._embedding_cache = PerformanceCache(max_size=200, ttl_seconds=1800)  # 30 min TTL
        
        # Initialize thread pool for parallel processing
        self._thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="medical_retriever")
        
        self._initialize_core_components()
        self._final_retriever = None
        self._gpu_components_active = False
        self._scan_and_process_all_files()
        self._start_document_watcher()
        
        self._is_initialized = True
        logger.info("MedicalRetrieverTool initialized successfully.")

    def _initialize_core_components(self, use_gpu: bool = False):
        """Initialize core components with GPU option."""
        device = 'cuda' if use_gpu else 'cpu'
        logger.info(f"Initializing embedding model with device: {device}")
        
        # self._embeddings = OllamaEmbeddings(model=settings.EMBEDDING_MODEL, base_url=settings.OLLAMA_BASE_URL)
        model_kwargs = {'device': device}
        encode_kwargs = {'normalize_embeddings': True}
        
        self._embeddings = HuggingFaceEmbeddings(
            model_name=settings.HF_EMBEDDING_MODEL, 
            model_kwargs=model_kwargs, 
            encode_kwargs=encode_kwargs
        )
        
        persistent_client = chromadb.PersistentClient(path=str(Path(settings.VECTOR_STORE_BASE_DIR)))
        self._vector_store = Chroma(
            client=persistent_client, 
            collection_name=self.collection_name, 
            embedding_function=self._embeddings
        )
        
        self._text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=256)
        
        # Track GPU status if using GPU
        if use_gpu:
            self._gpu_components_active = True
            self._gpu_last_used = time.time()
            logger.info(f"Initialized core components on GPU")
    @property
    def _registry_path(self) -> Path:
        return self.watch_directory / f"{self.collection_name}_registry.json"
    def _build_retriever_pipeline(self, force_gpu: bool = True):
        """Builds the final ContextualCompressionRetriever pipeline with GPU management."""
        logger.info("Building retriever pipeline with GPU support (Vector Search -> Reranker)...")
        
        # A. Create the base retriever directly from the vector store.
        # This is the first stage: get a broad set of potentially relevant docs.
        base_retriever = self._vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 7},  # Get top 7 results from vector search
            return_source_documents=True
        )
        
        # Release any existing GPU resources before building new ones
        self._release_gpu_resources()
        
        try:
            # B. Create the reranker model and compressor.
            # This is the second stage: accurately re-score the candidates.
            device = 'cuda' if force_gpu and TORCH_AVAILABLE else 'cpu'
            logger.info(f"Initializing cross-encoder reranker on {device}")
            
            model_kwargs = {'device': device}
            model = HuggingFaceCrossEncoder(model_name=settings.HF_RERANKER_MODEL, model_kwargs=model_kwargs)
            compressor = CrossEncoderReranker(model=model, top_n=3)
            
            # C. Create the final compression retriever.
            self._final_retriever = ContextualCompressionRetriever(
                base_compressor=compressor, 
                base_retriever=base_retriever
            )
            
            # Set GPU active status
            if device == 'cuda':
                self._gpu_components_active = True
                self._gpu_last_used = time.time()
                
            logger.info(f"Retriever pipeline built successfully on {device}.")
            
        except Exception as e:
            logger.error(f"Error building retriever pipeline: {e}. Retrieval may not work properly.")
            self._final_retriever = None
            self._gpu_components_active = False
            
    def _run(self, query: str) -> str:
        """Synchronously retrieves and reranks documents for the medical domain."""
        try:
            if not self._final_retriever:
                return "Khong tim thay Ket qua Phu hop ve truy van cua ban"
            
            compressed_docs = self._final_retriever.invoke(query)
    
            if not compressed_docs:
                return "Khong tim thay Ket qua Phu hop ve truy van cua ban"
    
            return "\n\n".join(
                f"Source: {doc.metadata.get('source', 'unknown')}\nContent: {doc.page_content}"
                for doc in compressed_docs
            )
        finally:
            # Check memory pressure and release resources if needed
            self._check_memory_pressure()
            
            # Update last used time to prevent premature release
            if self._gpu_components_active:
                self._gpu_last_used = time.time()

    @gpu_management_decorator
    async def _arun(self, query: str) -> str:
        """Asynchronously retrieves and reranks documents for the medical with GPU management."""
        if not self._final_retriever:
            return "Công cụ không hoạt động. Không tìm thấy kết quả phù hợp với truy vấn của bạn."

        try:
            start_time = time.time()
            compressed_docs = await self._final_retriever.ainvoke(query)
            retrieval_time = time.time() - start_time
            
            logger.info(f"Retrieved {len(compressed_docs)} documents for query: '{query}' in {retrieval_time:.2f}s")
            
            if not compressed_docs:
                logger.warning(f"No relevant documents found for query: '{query}'")
                return "Không tìm thấy kết quả phù hợp với truy vấn của bạn."
            
            return "\n\n".join(
                f"Source: {doc.metadata.get('source', 'unknown')}\nContent: {doc.page_content}"
                for doc in compressed_docs
            )
        except Exception as e:
            logger.error(f"Error retrieving documents for query: '{query}'. Error: {e}")
            return f"Đã xảy ra lỗi khi tìm kiếm thông tin: {str(e)}"
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
        """Scans the directory and processes all supported files on startup."""
        logger.info(f"Performing initial scan of directory: {self.watch_directory}")
        current_files = set()
        for file_path in self.watch_directory.rglob('*'):
            logger.debug(f"Found file: {file_path}")
            if file_path.is_file():
                current_files.add(str(file_path))
                logger.debug(f"Found file in current scan: {file_path}")
                
                self._process_file_if_needed(file_path)
        
        # Check for deleted files that are still in the registry
        registered_files = set(self._document_registry.keys())
        deleted_files = registered_files - current_files
        for file_path_str in deleted_files:
            self.remove_document_by_path(Path(file_path_str))

        self._save_document_registry()

    def _process_file_if_needed(self, file_path: Path):
        """Checks a file against the registry and loads/reloads it if needed."""
        if "_registry.json" in file_path.name: return

        current_hash = self._get_file_hash(file_path)
        if not current_hash: return

        stored_hash = self._document_registry.get(str(file_path))
        if current_hash != stored_hash:
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

    def _stop_document_watcher(self):
        if self._observer and self._observer.is_alive():
            self._observer.stop()
            self._observer.join()
            logger.info("Document watcher stopped.")

    def _reload_document(self, file_path: Path):
        """Deletes old versions and loads the new version of a document."""
        # 1. Delete existing documents from this source
        try: 
            logger.info(f"Delete existing documents from this source")
            self._delete_documents_by_source(file_path.name)
            logger.info("Load, split, and embed the new version")
            # 2. Load, split, and embed the new version
            documents = self._load_and_split_file(file_path)
            logger.info(f"Loaded {len(documents)} document chunks from {file_path.name}")
            if documents:
                logger.info(f"Adding {len(documents)} document chunks to the knowledge base.")
                self._add_documents_to_store(documents)
                
                # 3. Update the registry with the new hash
                new_hash = self._get_file_hash(file_path)
                if new_hash:
                    self._document_registry[str(file_path)] = new_hash
                    self._save_document_registry()
                    logger.info(f"Successfully reloaded and registered '{file_path.name}'.")
        except Exception as e:
            logger.error(f"Error reloading document '{file_path.name}': {e}")

    def _remove_document_by_path(self, file_path: Path):
        """Removes a document's vectors and its entry from the registry."""
        path_str = str(file_path)
        if path_str in self._document_registry:
            logger.info(f"Removing document '{file_path.name}' from knowledge base.")
            self._delete_documents_by_source(file_path.name)
            del self._document_registry[path_str]
            self._save_document_registry()
        
    def _load_and_split_file(self, file_path: Path) -> List[Document]:
        loader_map = {'.pdf': DocumentCustomConverter, '.csv': CSVLoader, '.json': JSONLoader, '.txt': TextLoader}
        loader_class = loader_map.get(file_path.suffix.lower())
        logger.info(f"Type of extension: {file_path.suffix.lower()}")
        logger.info(f"Loading file: {file_path.name} with loader: {loader_class.__name__ if loader_class else 'Unknown'}")
        try:
            try:
                logger.info(f"Loader class: {loader_class}")
                
                loader = loader_class(str(file_path))
                raw_docs = loader.load()
                logger.info(f"Loaded {len(raw_docs)} raw documents from {file_path.name}")
                
            except Exception as e:
                logger.error(f"Error initializing loader for {file_path.name}: {e}")
                return []

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
                
            split_docs = self._text_splitter.split_documents(raw_docs)
            for doc in split_docs:
                doc.metadata['source'] = file_path.name
            return split_docs
        except Exception: return []

    def _add_documents_to_store(self, docs: List[Document]):
        if not docs: return
        logger.info(f"Adding {len(docs)} document chunks to the knowledge base.")
        self._vector_store.add_documents(docs)


    def _delete_documents_by_source(self, source_filename: str):
        try:
            existing_ids = self._vector_store.get(where={"source": source_filename})['ids']
            if existing_ids:
                logger.info(f"Deleting {len(existing_ids)} old chunks for source '{source_filename}'...")
                self._vector_store.delete(ids=existing_ids)

        except Exception as e:
            logger.error(f"Failed to delete documents for source '{source_filename}': {e}")
            
    def _update_bm25_retriever(self, force: bool = False):
        """Update BM25 retriever with throttling and caching"""
        current_time = time.time()
        if not force and (current_time - self._last_bm25_update) < self._bm25_update_threshold:
            return
        
        try:
            # Check cache first
            cache_key = "bm25_docs"
            cached_docs = self._document_cache.get(cache_key)
            
            if cached_docs is None:
                all_docs = self._vector_store.get(include=["metadatas", "documents"])
                if all_docs and all_docs['documents']:
                    cached_docs = [Document(page_content=doc, metadata=meta)
                                   for doc, meta in zip(all_docs['documents'], all_docs['metadatas'])]
                    # Cache the documents
                    self._document_cache.set(cache_key, cached_docs)
                else:
                    cached_docs = []
            
            if cached_docs:
                self._bm25_retriever = BM25Retriever.from_documents(cached_docs)
                self._bm25_retriever.k = 10
                logger.info(f"BM25 retriever updated with {len(cached_docs)} documents.")
            else:
                self._bm25_retriever = None
            
            self._last_bm25_update = current_time
            
        except Exception as e:
            logger.error(f"Failed to update BM25 retriever: {e}")
            self._bm25_retriever = None

    def _get_vector_results(self, query: str) -> List[Tuple[Document, float]]:
        """Get vector search results with error handling."""
        try:
            return self._vector_store.similarity_search_with_relevance_scores(query, k=5)
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    def _get_bm25_results(self, query: str) -> List[Document]:
        """Get BM25 search results with error handling."""
        try:
            if self._bm25_retriever:
                return self._bm25_retriever.get_relevant_documents(query)
            return []
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []

    def retrieve_documents(self, query: str, use_cache: bool = True) -> List[str]:
        """Enhanced hybrid search with caching, parallel processing, and medical-specific scoring."""
        if not self._is_initialized:
            return ["Error: Medical retriever is not initialized."]
        
        start_time = time.time()
        
        # Check cache first
        cache_key = f"query:{hashlib.md5(query.encode()).hexdigest()}"
        if use_cache:
            cached_result = self._query_cache.get(cache_key)
            if cached_result is not None:
                logger.info(f"Cache hit for query: '{query}'")
                return cached_result
        
        logger.info(f"Retrieving medical documents for query: '{query}'")
        
        # Parallel search execution
        future_vector = self._thread_pool.submit(self._get_vector_results, query)
        future_bm25 = self._thread_pool.submit(self._get_bm25_results, query)
        
        try:
            # Wait for results with timeout
            vector_results = future_vector.result(timeout=10)
            bm25_results = future_bm25.result(timeout=10)
        except Exception as e:
            logger.error(f"Search timeout or error: {e}")
            vector_results = []
            bm25_results = []
        
        # Process results with deduplication
        hybrid_results = {}
        seen_hashes = set()
        
        # Process vector results
        for doc, score in vector_results:
            medical_doc = RetrievedMedicalDocument(
                content=doc.page_content,
                source=doc.metadata.get("source", "unknown"),
                relevance_score=score
            )
            
            content_hash = medical_doc.get_content_hash()
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                hybrid_results[doc.page_content] = medical_doc
        
        # Process BM25 results
        for doc in bm25_results:
            if doc.page_content not in hybrid_results:
                medical_doc = RetrievedMedicalDocument(
                    content=doc.page_content,
                    source=doc.metadata.get("source", "unknown"),
                    relevance_score=0.5  # Default BM25 score
                )
                
                content_hash = medical_doc.get_content_hash()
                if content_hash not in seen_hashes:
                    seen_hashes.add(content_hash)
                    hybrid_results[doc.page_content] = medical_doc
        
        # Apply medical-specific relevance scoring
        query_tokens = query.lower().split()
        for doc in hybrid_results.values():
            doc.calculate_medical_relevance(query_tokens)
        
        # Sort by relevance score
        sorted_docs = sorted(hybrid_results.values(), key=lambda x: x.relevance_score, reverse=True)
        
        # Format output
        if not sorted_docs:
            result = [f"No medical information found for: '{query}'"]
        else:
            top_results = sorted_docs[:5]  # Return top 5
            result = [
                f"Source: {doc.source}\nContent: {doc.content}"
                for doc in top_results
            ]
        
        # Cache the result
        if use_cache:
            self._query_cache.set(cache_key, result)
        
        processing_time = time.time() - start_time
        logger.info(f"Retrieved {len(result)} medical results in {processing_time:.2f}s for query: '{query}'")
        
        return result
    
    async def batch_retrieve(self, queries: List[str]) -> List[List[str]]:
        """Process multiple queries in parallel with batch optimization."""
        if not self._is_initialized:
            raise RuntimeError("MedicalRetrieverTool is not initialized.")
        
        start_time = time.time()
        logger.info(f"Processing batch of {len(queries)} medical queries")
        
        # Process queries in parallel
        tasks = []
        for query in queries:
            task = asyncio.create_task(
                asyncio.get_event_loop().run_in_executor(
                    self._thread_pool, self.retrieve_documents, query
                )
            )
            tasks.append(task)
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Error processing query '{queries[i]}': {result}")
                    processed_results.append([f"Error processing query: {result}"])
                else:
                    processed_results.append(result)
            
            processing_time = time.time() - start_time
            logger.info(f"Batch processed {len(queries)} medical queries in {processing_time:.2f}s")
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            return [[f"Batch processing error: {e}"] for _ in queries]
    def _release_gpu_resources(self):
        """Release GPU resources to free memory."""
        # Safely check if GPU components are active
        if not getattr(self, '_gpu_components_active', False):
            return
            
        logger.info("Releasing GPU resources...")
        
        # Delete the final retriever which holds GPU resources
        if hasattr(self, '_final_retriever') and self._final_retriever is not None:
            try:
                self._final_retriever = None
                logger.debug("Final retriever released")
            except Exception as e:
                logger.error(f"Error releasing final retriever: {e}")
                
        # Attempt to reinitialize with CPU if needed
        try:
            if hasattr(self, '_initialize_core_components'):
                # Reinitialize with CPU for lighter footprint
                if hasattr(self, '_vector_store') and self._vector_store is not None:
                    self._vector_store = None
                self._initialize_core_components(use_gpu=False)
                logger.debug("Reinitialized with CPU components")
        except Exception as e:
            logger.error(f"Error reinitializing with CPU: {e}")
        
        # Run garbage collection
        try:
            gc.collect()
        except Exception as e:
            logger.error(f"Error during garbage collection: {e}")
        
        # If using PyTorch, clear CUDA cache
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                logger.info("CUDA memory cache cleared")
            except Exception as e:
                logger.error(f"Error clearing CUDA memory: {e}")
        
        self._gpu_components_active = False
        logger.info("GPU resources released successfully")
    
    def _ensure_gpu_components(self):
        """Ensure GPU components are initialized when needed."""
        # Ensure lock exists
        if not hasattr(self, '_gpu_lock') or self._gpu_lock is None:
            self._gpu_lock = threading.RLock()
            
        # Use lock to prevent race conditions
        with self._gpu_lock:
            current_time = time.time()
            
            # Check if components are active and not timed out
            if (getattr(self, '_gpu_components_active', False) and 
                getattr(self, '_final_retriever', None) is not None and 
                (current_time - getattr(self, '_gpu_last_used', 0)) < getattr(self, '_gpu_ttl_seconds', 300)):
                # Update last used time
                self._gpu_last_used = current_time
                return True
            
            # If components timed out or not initialized, release and rebuild
            logger.info("GPU components inactive or timed out. Reinitializing...")
            self._release_gpu_resources()
            
            try:
                # Initialize the embeddings with GPU first
                self._initialize_core_components(use_gpu=True)
                
                # Then build the retriever pipeline with GPU
                self._build_retriever_pipeline(force_gpu=True)
                
                return getattr(self, '_gpu_components_active', False)
            except Exception as e:
                logger.error(f"Failed to initialize GPU components: {e}")
                return False
    def clear_caches(self):
        """Clear all performance caches."""
        if self._query_cache:
            self._query_cache.clear()
        if self._document_cache:
            self._document_cache.clear()
        if self._embedding_cache:
            self._embedding_cache.clear()
        logger.info("All medical retriever caches cleared")
    
    def get_cache_stats(self) -> Dict[str, Dict[str, int]]:
        """Get statistics about cache usage."""
        stats = {}
        
        if self._query_cache:
            stats['query_cache'] = {
                'size': len(self._query_cache.cache),
                'max_size': self._query_cache.max_size,
                'ttl_seconds': self._query_cache.ttl_seconds
            }
        
        if self._document_cache:
            stats['document_cache'] = {
                'size': len(self._document_cache.cache),
                'max_size': self._document_cache.max_size,
                'ttl_seconds': self._document_cache.ttl_seconds
            }
        
        if self._embedding_cache:
            stats['embedding_cache'] = {
                'size': len(self._embedding_cache.cache),
                'max_size': self._embedding_cache.max_size,
                'ttl_seconds': self._embedding_cache.ttl_seconds
            }
        
        return stats
    def _check_memory_pressure(self):
        """Check GPU memory pressure and release resources if needed."""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return
            
        try:
            # Check current memory usage
            memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
            memory_reserved = torch.cuda.memory_reserved() / (1024 ** 3)  # GB
            
            # If memory pressure is high, release resources
            if memory_allocated > 0.8 * memory_reserved:
                logger.warning(f"High GPU memory pressure detected: {memory_allocated:.2f}GB/{memory_reserved:.2f}GB")
                self._release_gpu_resources()
        except Exception as e:
            logger.error(f"Error checking memory pressure: {e}")
            
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
            
    def cleanup(self):
        """Clean up resources including thread pool and document watcher."""
        logger.info("Cleaning up medical retriever resources...")
        self._release_gpu_resources()
        self._stop_document_watcher()
        
        if self._thread_pool:
            self._thread_pool.shutdown(wait=False)

    async def manually_ensure_gpu_ready(self):
        """
        Manually ensure GPU resources are initialized.
        Use this before expected high-volume query periods.
        """
        logger.info("Manually ensuring GPU components are ready")
        return self._ensure_gpu_components()
        
    async def manually_release_gpu(self):
        """
        Manually release GPU resources.
        Use this after periods of inactivity to free up GPU memory.
        """
        logger.info("Manually releasing GPU resources")
        self._release_gpu_resources()
        return {"status": "success", "message": "GPU resources released successfully"}
    
    def __getstate__(self):
        """
        Custom method to prepare object for pickling.
        Excludes thread locks, GPU components and other unpicklable objects.
        """
        state = self.__dict__.copy()
        
        # Remove unpicklable attributes
        for attr in ['_gpu_lock', '_thread_pool', '_observer', '_embeddings', 
                     '_final_retriever', '_vector_store']:
            if attr in state:
                state[attr] = None
                
        # Make sure we know this was pickled
        state['_was_pickled'] = True
        logger.debug("medicalRetrieverTool prepared for pickling")
        return state
    
    def __setstate__(self, state):
        """Restore object after unpickling and reinitialize necessary components."""
        self.__dict__.update(state)
        
        # Create a new lock if needed
        if not hasattr(self, '_gpu_lock') or self._gpu_lock is None:
            self._gpu_lock = threading.RLock()
            
        # Flag that we need to reinitialize if used
        self._gpu_components_active = False
        logger.debug("medicalRetrieverTool unpickled, will reinitialize components when needed")
        
    def __del__(self):
        """Cleanup resources on object destruction."""
        try:
            # Check if initialization was completed
            if not hasattr(self, '_is_initialized') or not self._is_initialized:
                return
                
            # Release GPU resources if active
            if hasattr(self, '_gpu_components_active') and getattr(self, '_gpu_components_active', False):
                try:
                    if hasattr(self, '_release_gpu_resources'):
                        self._release_gpu_resources()
                except Exception as gpu_err:
                    logger.error(f"Error releasing GPU resources: {gpu_err}")
            
            # Shutdown thread pool if exists
            if hasattr(self, '_thread_pool') and getattr(self, '_thread_pool', None):
                try:
                    self._thread_pool.shutdown(wait=False)
                    logger.debug("Thread pool shut down")
                except Exception as tp_err:
                    logger.error(f"Error shutting down thread pool: {tp_err}")
            
            # Stop document watcher if exists
            if hasattr(self, '_observer') and getattr(self, '_observer', None):
                try:
                    self._stop_document_watcher()
                    logger.debug("Document watcher stopped")
                except Exception as obs_err:
                    logger.error(f"Error stopping document watcher: {obs_err}")
                    
        except Exception as e:
            logger.error(f"General error during cleanup: {str(e)}")
# Example usage
if __name__ == '__main__':
    async def main():
        DATA_DIR = Path("./medical_data_test")
        DATA_DIR.mkdir(exist_ok=True)
        logger.info(f"Using data directory: {DATA_DIR.resolve()}")

        # Create test files
        with open(DATA_DIR / "diabetes.txt", "w", encoding="utf-8") as f:
            f.write("Bệnh tiểu đường (diabetes) là một bệnh rối loạn chuyển hóa. Triệu chứng bao gồm khát nước và đi tiểu nhiều.")
        with open(DATA_DIR / "heart_disease.csv", "w", encoding="utf-8") as f:
            f.write("condition_name,risk_factor,recommendation\n")
            f.write("Bệnh tim mạch,Hút thuốc lá,Ngừng hút thuốc ngay lập tức\n")

        # Initialize the tool
        medical_tool = MedicalRetrieverTool(watch_directory=str(DATA_DIR), collection_name="medical_test_collection")
        
        # Test initial retrieval
        query = "triệu chứng bệnh tiểu đường"
        print(f"\n--- Testing query: '{query}' ---")
        print(await medical_tool._arun(query))

        # Test auto-update (new file)
        print("\n--- Testing auto-update. Adding a new file... ---")
        time.sleep(2)
        with open(DATA_DIR / "hypertension.txt", "w", encoding="utf-8") as f:
            f.write("Tăng huyết áp là tình trạng áp lực máu lên thành động mạch cao hơn mức bình thường.")
        time.sleep(5) # Wait for watcher
        
        query2 = "tăng huyết áp là gì"
        print(f"\n--- Testing query after new file: '{query2}' ---")
        print(await medical_tool._arun(query2))
        
        # Test auto-update (delete file)
        print("\n--- Testing auto-update. Deleting a file... ---")
        time.sleep(2)
        os.remove(DATA_DIR / "diabetes.txt")
        time.sleep(5) # Wait for watcher

        print(f"\n--- Re-testing original query after delete: '{query}' ---")
        print(await medical_tool._arun(query)) # Should find less relevant or no results

        medical_tool.stop_document_watcher()

    asyncio.run(main())