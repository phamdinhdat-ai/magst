import os
import json
import sys
import time
import hashlib
import re
import threading
import asyncio
import gc
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError
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

# --- Import for GPU memory management ---
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. GPU memory management disabled.")
import asyncio
from app.agents.workflow.initalize import llm_instance, settings, agent_config
from app.agents.factory.tools.base import BaseAgentTool
from app.utils.document_processor import  DocumentCustomConverter, markdown_splitter, remove_image_tags
# --- Imports for the Contextual Compression Pattern ---
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.embeddings import HuggingFaceEmbeddings, HuggingFaceInferenceAPIEmbeddings, SelfHostedHuggingFaceEmbeddings,HypotheticalDocumentEmbedder
from langchain_openai.embeddings import OpenAIEmbeddings

class DocumentWatcher(FileSystemEventHandler):
    """File system watcher that triggers document loading and reloading."""
    def __init__(self, retriever_tool: 'CompanyRetrieverTool'):
        self.tool = retriever_tool

    def on_created(self, event):
        if not event.is_directory:
            logger.info(f"[Watcher] New file detected: {event.src_path}")
            time.sleep(1)
            self.tool._process_file_if_needed(Path(event.src_path))

    def on_modified(self, event):
        if not event.is_directory and "_registry.json" not in event.src_path:
            logger.info(f"[Watcher] File modified: {event.src_path}")
            time.sleep(1)
            self.tool._process_file_if_needed(Path(event.src_path))
            
    def on_deleted(self, event):
        if not event.is_directory:
            logger.info(f"[Watcher] File deleted: {event.src_path}")
            self.tool._remove_document_by_path(Path(event.src_path))


class PerformanceCache:
    """Thread-safe cache with TTL support for performance optimization."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        self.cache = {}
        self.timestamps = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.lock = threading.Lock()
    
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
            if len(self.cache) >= self.max_size and key not in self.cache:
                oldest_key = min(self.timestamps.keys(), key=lambda k: self.timestamps[k])
                del self.cache[oldest_key]
                del self.timestamps[oldest_key]
            
            self.cache[key] = value
            self.timestamps[key] = time.time()
    
    def clear(self):
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()


class RetrievedCompanyDocument(BaseModel):
    """Represents a retrieved company document with metadata and company-specific scoring."""
    content: str
    source: str
    relevance_score: float
    topic: str = ""
    category: str = ""
    
    # Cached attributes for performance
    _content_hash: Optional[str] = None
    _processed_tokens: Optional[List[str]] = None
    
    def get_content_hash(self) -> str:
        """Get cached content hash for deduplication."""
        if self._content_hash is None:
            self._content_hash = hashlib.md5(self.content.encode()).hexdigest()
        return self._content_hash
    
    def get_processed_tokens(self) -> List[str]:
        """Get cached processed tokens for efficient analysis."""
        if self._processed_tokens is None:
            # Enhanced tokenization for company terms
            self._processed_tokens = re.findall(r'\b\w+\b', self.content.lower())
        return self._processed_tokens
    
    def calculate_company_relevance(self, query_tokens: List[str]) -> float:
        """Calculate relevance with company-specific scoring"""
        content_tokens = set(self.get_processed_tokens())
        query_token_set = set(query_tokens)
        
        # Company-specific term weights
        business_terms = {'company', 'business', 'corporation', 'enterprise', 'organization', 'firm', 'industry', 'market', 'revenue', 'profit'}
        financial_terms = {'financial', 'finance', 'investment', 'capital', 'funding', 'budget', 'cost', 'income', 'expense', 'asset'}
        management_terms = {'management', 'executive', 'leadership', 'strategy', 'operations', 'performance', 'growth', 'development', 'planning', 'decision'}
        
        # Calculate weighted intersection
        intersection_score = 0.0
        for token in query_token_set & content_tokens:
            if token in business_terms:
                intersection_score += 2.5  # Highest weight for business terms
            elif token in financial_terms:
                intersection_score += 2.0  # High weight for financial terms
            elif token in management_terms:
                intersection_score += 1.8  # High weight for management terms
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
        
        # Try to ensure components are available (GPU or CPU fallback)
        try:
            if not hasattr(self, '_ensure_gpu_components') or not self._ensure_gpu_components():
                logger.warning("Failed to ensure components availability with GPU/CPU fallback")
                return "Retrieval tool temporarily unavailable. Please try again in a moment."
        except Exception as gpu_err:
            logger.error(f"Error ensuring components: {gpu_err}")
            return "Retrieval tool temporarily unavailable due to initialization error."
        
        try:
            # Run the retrieval function
            result = await func(self, *args, **kwargs)
            return result
        except Exception as e:
            logger.error(f"Error in retrieval function: {str(e)}")
            return f"Error retrieving information: {str(e)}"
        finally:
            # Update last used time for GPU components (if active)
            if hasattr(self, '_gpu_components_active') and getattr(self, '_gpu_components_active', False):
                self._gpu_last_used = time.time()
            
            # Check memory pressure
            try:
                if hasattr(self, '_check_memory_pressure'):
                    self._check_memory_pressure()
            except Exception as e:
                logger.error(f"Error checking memory pressure: {e}")
            
    return wrapper

class CompanyRetrieverTool(BaseAgentTool):
    """
    An advanced retriever for company information that automatically ingests
    and updates documents from a specified directory, stores them in a
    vector database, and uses a hybrid search approach for high-relevance results.
    """
    name: str = "company_retriever"
    description: str = "Retrieves and reranks company information from a self-updating knowledge base."

    # --- Core Configuration (Pydantic Fields) ---
    collection_name: str = Field(description="Name for the ChromaDB collection.")
    watch_directory: Path = Field(description="Directory to watch for new/updated documents.")
    
    # --- Internal Components (Private Attributes for Pydantic v2) ---
    _vector_store: Chroma = PrivateAttr(default=None)
    _bm25_retriever: BM25Retriever = PrivateAttr(default=None)
    _embeddings: OllamaEmbeddings = PrivateAttr(default=None)
    _text_splitter: RecursiveCharacterTextSplitter = PrivateAttr(default=None)
    _observer: Observer = PrivateAttr(default=None) # type: ignore
    _document_registry: Dict[str, str] = PrivateAttr(default_factory=dict)
    _is_initialized: bool = PrivateAttr(default=False)
    
    # --- Performance Optimization Components ---
    _query_cache: PerformanceCache = PrivateAttr(default=None)
    _document_cache: PerformanceCache = PrivateAttr(default=None)
    _embedding_cache: PerformanceCache = PrivateAttr(default=None)
    _thread_pool: ThreadPoolExecutor = PrivateAttr(default=None)
    _last_bm25_update: float = PrivateAttr(default=0.0)
    _bm25_update_threshold: float = PrivateAttr(default=60.0)  # 60 seconds
    
    # --- GPU Memory Management ---
    _gpu_components_active: bool = PrivateAttr(default=False)
    _gpu_last_used: float = PrivateAttr(default=0.0)
    _gpu_ttl_seconds: float = PrivateAttr(default=300.0)  # 5 minutes before release
    _gpu_lock: threading.RLock = PrivateAttr(default=None)  # Instance lock for GPU operations
    _final_retriever: ContextualCompressionRetriever = PrivateAttr(default=None)


    def __init__(self, watch_directory: str, collection_name: str = "company_docs", **kwargs):
        # Sửa lỗi Pydantic bằng cách truyền tất cả các trường vào super().__init__
        init_kwargs = kwargs.copy()
        init_kwargs['watch_directory'] = Path(watch_directory).resolve()
        init_kwargs['collection_name'] = self._sanitize_collection_name(collection_name)
        
        super().__init__(**init_kwargs)
        
        if not self._is_initialized:
            self._initialize_all()

    def _sanitize_collection_name(self, name: str) -> str:
        sanitized = re.sub(r'[^a-zA-Z0-9_.-]', '_', name)
        return sanitized[:63]

    def _initialize_all(self):
        """Orchestrates the full initialization process."""
        logger.info(f"Initializing CompanyRetrieverTool for collection '{self.collection_name}'...")
        self.watch_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize GPU lock first
        self._gpu_lock = threading.RLock()
        
        # Initialize performance caches
        self._query_cache = PerformanceCache(max_size=500, ttl_seconds=300)  # 5 min TTL
        self._document_cache = PerformanceCache(max_size=1000, ttl_seconds=600)  # 10 min TTL
        self._embedding_cache = PerformanceCache(max_size=200, ttl_seconds=1800)  # 30 min TTL
        
        # Initialize thread pool for parallel processing
        self._thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="company_retriever")
        
        # Initialize core components without GPU first
        self._initialize_core_components(use_gpu=False)
        self._load_document_registry()
        
        # Don't initialize GPU pipeline yet - do it on demand
        self._final_retriever = None
        self._gpu_components_active = False
        
        self._scan_and_process_all_files()
        self._start_document_watcher()
        
        self._is_initialized = True
        logger.info("CompanyRetrieverTool initialized successfully (GPU components deferred).")

    def _build_retriever_pipeline(self, force_gpu: bool = True):
        """Builds the final ContextualCompressionRetriever pipeline with GPU management."""
        logger.info("Building retriever pipeline with GPU support (Vector Search -> Reranker)...")
        
        # A. Create the base retriever directly from the vector store.
        # This is the first stage: get a broad set of potentially relevant docs.
        base_retriever = self._vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10},  # Get top 10 results from vector search
            return_source_documents=True
        )
        
        # Release any existing GPU resources before building new ones
        self._release_gpu_resources()
        
        # Try GPU first, then fallback to CPU if there are CUDA memory issues
        # devices_to_try = ['cpu']
        devices_to_try = ['cuda', 'cpu'] if force_gpu and TORCH_AVAILABLE else ['cpu']
        
        for device in devices_to_try:
            try:
                # B. Create the reranker model and compressor.
                # This is the second stage: accurately re-score the candidates.
                logger.info(f"Attempting to initialize cross-encoder reranker on {device}")
                
                # Check if the Vietnamese reranker model exists
                model_path = settings.HF_RERANKER_MODEL
                if not os.path.exists(model_path):
                    logger.warning(f"Vietnamese reranker model not found at '{model_path}', falling back to standard model")
                    model_path = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Standard reranker model
                
                model_kwargs = {'device': device}
                model = HuggingFaceCrossEncoder(model_name=model_path, model_kwargs=model_kwargs)
                compressor = CrossEncoderReranker(model=model, top_n=5)
                
                # C. Create the final compression retriever.
                self._final_retriever = ContextualCompressionRetriever(
                    base_compressor=compressor, 
                    base_retriever=base_retriever
                )
                
                # Set GPU active status
                if device == 'cuda':
                    self._gpu_components_active = True
                    self._gpu_last_used = time.time()
                else:
                    self._gpu_components_active = False
                    
                logger.info(f"Retriever pipeline built successfully on {device} using model: {model_path}")
                return  # Success, exit the loop
                
            except Exception as e:
                error_msg = str(e)
                if device == 'cuda' and ('CUDA out of memory' in error_msg or 'out of memory' in error_msg.lower()):
                    logger.warning(f"CUDA out of memory during pipeline building: {e}. Trying CPU fallback...")
                    # Clear CUDA cache before trying CPU
                    if TORCH_AVAILABLE and torch.cuda.is_available():
                        try:
                            torch.cuda.empty_cache()
                            gc.collect()
                        except Exception:
                            pass
                    continue  # Try next device (CPU)
                else:
                    logger.error(f"Error building retriever pipeline on {device}: {e}")
                    if device == devices_to_try[-1]:  # Last device to try
                        # If all devices failed, fall back to base retriever without reranking
                        logger.warning(f"All devices failed. Falling back to base retriever without reranking.")
                        self._final_retriever = base_retriever
                        self._gpu_components_active = False
                        return
                    continue
        
        # If we get here, something went wrong
        logger.error("Failed to build retriever pipeline on any device.")
        self._final_retriever = base_retriever  # Fallback to base retriever
        self._gpu_components_active = False
    
    def _run(self, query: str) -> str:
        """Synchronously retrieves and reranks documents for the customer."""
        try:
            if not self._final_retriever:
                return "Khong tim thay Ket qua Phu hop ve truy van cua ban"
            start_time = time.time()
            compressed_docs = self._final_retriever.invoke(query)
            retrieval_time = time.time() - start_time

            logger.info(f"Retrieved {len(compressed_docs)} documents for query: '{query}' in {retrieval_time:.2f}s")

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
        """Asynchronously retrieves and reranks documents for the customer with GPU management."""
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

    

    def _initialize_core_components(self, use_gpu: bool = False):
        """Initialize core components with GPU option and CUDA fallback."""
        # Try GPU first if requested, then fallback to CPU on CUDA memory errors
        devices_to_try = ['cuda', 'cpu'] if use_gpu and TORCH_AVAILABLE else ['cpu']
        
        for device in devices_to_try:
            try:
                logger.info(f"Attempting to initialize embedding model with device: {device}")
                
                model_kwargs = {'device': device}
                encode_kwargs = {'normalize_embeddings': True}
                self._embeddings = HuggingFaceEmbeddings(model_name=settings.HF_EMBEDDING_MODEL, 
                                                       model_kwargs=model_kwargs, 
                                                       encode_kwargs=encode_kwargs)
                persistent_client = chromadb.PersistentClient(path=str(Path(settings.VECTOR_STORE_BASE_DIR)))
                self._vector_store = Chroma(
                    client=persistent_client, 
                    collection_name=self.collection_name, 
                    embedding_function=self._embeddings
                )
                
                self._text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=256)
                
                # Track GPU status if using GPU
                if device == 'cuda':
                    self._gpu_components_active = True
                    self._gpu_last_used = time.time()
                    logger.info(f"Initialized core components on GPU")
                else:
                    self._gpu_components_active = False
                    logger.info(f"Initialized core components on CPU")
                
                return  # Success, exit the loop
                
            except Exception as e:
                error_msg = str(e)
                if device == 'cuda' and ('CUDA out of memory' in error_msg or 'out of memory' in error_msg.lower()):
                    logger.warning(f"CUDA out of memory during core component initialization: {e}. Trying CPU fallback...")
                    # Clear CUDA cache before trying CPU
                    if TORCH_AVAILABLE and torch.cuda.is_available():
                        try:
                            torch.cuda.empty_cache()
                            gc.collect()
                        except Exception:
                            pass
                    continue  # Try next device (CPU)
                else:
                    logger.error(f"Error initializing core components on {device}: {e}")
                    if device == devices_to_try[-1]:  # Last device to try
                        raise e  # Re-raise the exception if all devices failed
                    continue
        
        # If we get here, all devices failed
        raise Exception("Failed to initialize core components on any device.")

    @property
    def _registry_path(self) -> Path:
        return self.watch_directory / f"{self.collection_name}_registry.json"

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
        
        registered_files = set(self._document_registry.keys())
        deleted_files = registered_files - current_files
        for file_path_str in deleted_files:
            self._remove_document_by_path(Path(file_path_str))

        self._save_document_registry()
    def _remove_document_by_path(self, file_path: Path):
        path_str = str(file_path)
        if path_str in self._document_registry:
            self._delete_documents_by_source(file_path.name)
            del self._document_registry[path_str]
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

    def _reload_document(self, file_path: Path):
        """Deletes old versions and loads the new version of a document."""
        self._delete_documents_by_source(file_path.name)
        documents = self._load_and_split_file(file_path)
        if documents:
            self._add_documents_to_store(documents)
            new_hash = self._get_file_hash(file_path)
            if new_hash:
                self._document_registry[str(file_path)] = new_hash
                self._save_document_registry()
                logger.info(f"Successfully reloaded and registered '{file_path.name}'.")

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
            split_docs = self._text_splitter.split_documents(raw_docs)
            for doc in split_docs:
                doc.metadata['source'] = file_path.name
            return split_docs
        except Exception: return []

    def _add_documents_to_store(self, docs: List[Document]):
        if not docs: return
        logger.info(f"Adding {len(docs)} document chunks to the knowledge base.")
        self._vector_store.add_documents(docs)
        self._update_bm25_retriever()

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
        """Update BM25 retriever with throttling and caching."""
        current_time = time.time()
        
        # Throttle updates unless forced
        if not force and (current_time - self._last_bm25_update) < self._bm25_update_threshold:
            return
        
        try:
            # Check document cache first
            cache_key = "bm25_documents"
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
                self._bm25_retriever.k = 5
                logger.info(f"BM25 retriever updated with {len(cached_docs)} documents (cached: {cached_docs is not None}).")
            else:
                self._bm25_retriever = None
                
            self._last_bm25_update = current_time
            
        except Exception as e:
            logger.error(f"Failed to update BM25 retriever: {e}")
    
    def _parallel_vector_search(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """Perform vector search in parallel with timeout."""
        try:
            future = self._thread_pool.submit(
                self._vector_store.similarity_search_with_relevance_scores, query, k
            )
            return future.result(timeout=10)  # 10 second timeout
        except TimeoutError:
            logger.warning(f"Vector search timed out for query: {query[:50]}...")
            return []
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    def _parallel_bm25_search(self, query: str) -> List[Document]:
        """Perform BM25 search in parallel with timeout."""
        if not self._bm25_retriever:
            return []
        
        try:
            future = self._thread_pool.submit(
                self._bm25_retriever.get_relevant_documents, query
            )
            return future.result(timeout=10)  # 10 second timeout
        except TimeoutError:
            logger.warning(f"BM25 search timed out for query: {query[:50]}...")
            return []
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []

    def retrieve_documents(self, query: str, use_cache: bool = True) -> List[str]:
        """Enhanced retrieval with caching, parallel processing, and company-specific scoring."""
        if not self._is_initialized:
            return ["Error: Company retriever is not initialized."]
        
        start_time = time.time()
        
        # Check cache first
        cache_key = f"query_{hashlib.md5(query.encode()).hexdigest()}"
        if use_cache:
            cached_result = self._query_cache.get(cache_key)
            if cached_result is not None:
                logger.info(f"Cache hit for query: {query[:50]}... (took {time.time() - start_time:.3f}s)")
                return cached_result
        
        # Update BM25 retriever (throttled)
        self._update_bm25_retriever()
        
        # Parallel search execution
        vector_future = self._thread_pool.submit(self._parallel_vector_search, query, 10)
        bm25_future = self._thread_pool.submit(self._parallel_bm25_search, query)
        
        try:
            # Get results from parallel searches
            vector_results = vector_future.result(timeout=12)
            bm25_results = bm25_future.result(timeout=12)
            
            # Process and deduplicate results
            hybrid_results = {}
            seen_hashes = set()
            
            # Process vector results
            for doc, score in vector_results:
                company_doc = RetrievedCompanyDocument(
                    content=doc.page_content,
                    source=doc.metadata.get("source", "unknown"),
                    relevance_score=score
                )
                
                content_hash = company_doc.get_content_hash()
                if content_hash not in seen_hashes:
                    seen_hashes.add(content_hash)
                    hybrid_results[content_hash] = company_doc
            
            # Process BM25 results
            for doc in bm25_results:
                company_doc = RetrievedCompanyDocument(
                    content=doc.page_content,
                    source=doc.metadata.get("source", "unknown"),
                    relevance_score=0.5
                )
                
                content_hash = company_doc.get_content_hash()
                if content_hash not in seen_hashes:
                    seen_hashes.add(content_hash)
                    hybrid_results[content_hash] = company_doc
            
            # Apply company-specific relevance scoring
            query_tokens = re.findall(r'\b\w+\b', query.lower())
            for company_doc in hybrid_results.values():
                company_doc.calculate_company_relevance(query_tokens)
            
            # Sort and format results
            sorted_docs = sorted(hybrid_results.values(), key=lambda x: x.relevance_score, reverse=True)
            
            if not sorted_docs:
                result = [f"No relevant company information found for: '{query}'"]
            else:
                result = [f"Source: {doc.source}\nContent: {doc.content}" for doc in sorted_docs[:5]]
            
            # Cache the result
            if use_cache:
                self._query_cache.set(cache_key, result)
            
            processing_time = time.time() - start_time
            logger.info(f"Company retrieval completed for query: {query[:50]}... "
                       f"(took {processing_time:.3f}s, found {len(sorted_docs)} docs)")
            
            return result
            
        except Exception as e:
            logger.error(f"Error during company document retrieval: {e}")
            return [f"Error retrieving company documents: {str(e)}"]

    # def _run(self, query: str) -> str:
    #     results = self.retrieve_documents(query)
    #     return "\n\n---\n\n".join(results)

    # async def _arun(self, query: str) -> str:
    #     loop = asyncio.get_event_loop()
    #     return await loop.run_in_executor(None, self._run, query)

    def run(self, query: str) -> str:
        """Synchronous entry point for running the retriever."""
        return self._run(query)
    
    async def arun(self, query: str) -> str:
        """Asynchronous entry point for running the retriever."""
        logger.info(f"Asynchronously running customer retriever for query: '{query}'")
        return await self._arun(query)  
    
    async def batch_retrieve(self, queries: List[str]) -> List[List[str]]:
        """Process multiple queries in parallel with async support."""
        if not self._is_initialized:
            return [["Error: Company retriever is not initialized."] for _ in queries]
        
        start_time = time.time()
        logger.info(f"Starting batch retrieval for {len(queries)} queries...")
        
        async def process_query(query: str) -> List[str]:
            try:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(self._thread_pool, self.retrieve_documents, query)
            except Exception as e:
                logger.error(f"Error processing query '{query[:50]}...': {e}")
                return [f"Error processing query: {str(e)}"]
        
        # Process all queries concurrently
        tasks = [process_query(query) for query in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Exception in batch query {i}: {result}")
                processed_results.append([f"Error: {str(result)}"])
            else:
                processed_results.append(result)
        
        processing_time = time.time() - start_time
        logger.info(f"Batch retrieval completed in {processing_time:.3f}s for {len(queries)} queries")
        
        return processed_results
    
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
        """Ensure GPU components are initialized when needed with CUDA fallback."""
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
                
            # Check if we have a working CPU fallback retriever
            if (getattr(self, '_final_retriever', None) is not None and 
                not getattr(self, '_gpu_components_active', False)):
                # Already have a working CPU retriever
                return True
            
            # If components timed out or not initialized, release and rebuild
            logger.info("GPU components inactive or timed out. Reinitializing...")
            self._release_gpu_resources()
            
            # Try GPU first, then fallback to CPU
            for try_gpu in [True, False]:
                try:
                    # Initialize the embeddings 
                    self._initialize_core_components(use_gpu=try_gpu)
                    
                    # Then build the retriever pipeline
                    self._build_retriever_pipeline(force_gpu=try_gpu)
                    
                    device_type = "GPU" if try_gpu else "CPU"
                    logger.info(f"Successfully initialized components on {device_type}")
                    return True
                    
                except Exception as e:
                    error_msg = str(e)
                    if try_gpu and ('CUDA out of memory' in error_msg or 'out of memory' in error_msg.lower()):
                        logger.warning(f"CUDA out of memory during component initialization: {e}. Trying CPU fallback...")
                        # Clear CUDA cache before trying CPU
                        if TORCH_AVAILABLE and torch.cuda.is_available():
                            try:
                                torch.cuda.empty_cache()
                                gc.collect()
                            except Exception:
                                pass
                        continue  # Try CPU
                    else:
                        logger.error(f"Failed to initialize components: {e}")
                        if not try_gpu:  # Already on CPU, this is the final fallback
                            return False
                        continue  # Try CPU if this was GPU attempt
            
            # All attempts failed
            logger.error("Failed to initialize components on any device")
            return False

    def clear_caches(self):
        """Clear all performance caches."""
        logger.info("Clearing company retriever caches...")
        if self._query_cache:
            self._query_cache.clear()
        if self._document_cache:
            self._document_cache.clear()
        if self._embedding_cache:
            self._embedding_cache.clear()
        logger.info("Company retriever caches cleared.")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring."""
        stats = {
            "query_cache": {
                "size": len(self._query_cache.cache) if self._query_cache else 0,
                "max_size": self._query_cache.max_size if self._query_cache else 0,
                "ttl_seconds": self._query_cache.ttl_seconds if self._query_cache else 0
            },
            "document_cache": {
                "size": len(self._document_cache.cache) if self._document_cache else 0,
                "max_size": self._document_cache.max_size if self._document_cache else 0,
                "ttl_seconds": self._document_cache.ttl_seconds if self._document_cache else 0
            },
            "embedding_cache": {
                "size": len(self._embedding_cache.cache) if self._embedding_cache else 0,
                "max_size": self._embedding_cache.max_size if self._embedding_cache else 0,
                "ttl_seconds": self._embedding_cache.ttl_seconds if self._embedding_cache else 0
            }
        }
        return stats
    
    # GPU management decorator moved to module level

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
        logger.info("Cleaning up company retriever resources...")
        self._release_gpu_resources()
        self._stop_document_watcher()
        
        if self._thread_pool:
            self._thread_pool.shutdown(wait=False)
    
    def _stop_document_watcher(self):
        """Stop the document watcher."""
        if self._observer and self._observer.is_alive():
            self._observer.stop()
            self._observer.join()
            logger.info("Document watcher stopped.")
    
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
    
    
    
    async def mcp_retrieve_docs(self, query):
        pass
    
    
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
        logger.debug("CompanyRetrieverTool prepared for pickling")
        return state
    
    def __setstate__(self, state):
        """Restore object after unpickling and reinitialize necessary components."""
        self.__dict__.update(state)
        
        # Create a new lock if needed
        if not hasattr(self, '_gpu_lock') or self._gpu_lock is None:
            self._gpu_lock = threading.RLock()
            
        # Flag that we need to reinitialize if used
        self._gpu_components_active = False
        logger.debug("CompanyRetrieverTool unpickled, will reinitialize components when needed")
        
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
            # Don't re-raise the exception to avoid crashes during garbage collection
# Example usage
if __name__ == '__main__':
    async def main():
        DATA_DIR = Path("./company_data_test")
        DATA_DIR.mkdir(exist_ok=True)
        logger.info(f"Using data directory: {DATA_DIR.resolve()}")

        # Create test files
        with open(DATA_DIR / "contact.txt", "w", encoding="utf-8") as f:
            f.write("Thông tin liên hệ GeneStory. Địa chỉ: Tầng 7, Tòa nhà HL, Hà Nội. Email: contact@genestory.asia")
        with open(DATA_DIR / "projects.csv", "w", encoding="utf-8") as f:
            f.write("project_name,description,status\n")
            f.write("1000 Hệ gen người Việt,Giải mã 1000 hệ gen để xây dựng dữ liệu tham chiếu,Completed\n")

        # Initialize the tool
        company_tool = CompanyRetrieverTool(watch_directory=str(DATA_DIR), collection_name="company_test_collection")
        
        # Test initial retrieval
        query = "địa chỉ của genestory"
        print(f"\n--- Testing query: '{query}' ---")
        print(await company_tool._arun(query))

        # Test auto-update (new file)
        print("\n--- Testing auto-update. Adding a new file... ---")
        time.sleep(2)
        with open(DATA_DIR / "policy.pdf", "wb") as f:
            # Creating a dummy PDF is complex, we'll use a text file with .pdf extension for this test
            # In a real scenario, you would place an actual PDF file here.
            pass # Create an empty file to trigger the watcher
        with open(DATA_DIR / "policy.txt", "w", encoding="utf-8") as f:
             f.write("Chính sách bảo mật: GeneStory cam kết bảo vệ thông tin cá nhân của khách hàng.")
        os.rename(DATA_DIR / "policy.txt", DATA_DIR / "policy.pdf") # Rename to trigger as a PDF

        time.sleep(5) # Wait for watcher
        
        query2 = "chính sách bảo mật"
        print(f"\n--- Testing query after new file: '{query2}' ---")
        print(await company_tool._arun(query2))
        
        # Test auto-update (delete file)
        print("\n--- Testing auto-update. Deleting a file... ---")
        time.sleep(2)
        os.remove(DATA_DIR / "contact.txt")
        time.sleep(5) # Wait for watcher

        print(f"\n--- Re-testing original query after delete: '{query}' ---")
        print(await company_tool._arun(query))

        company_tool.cleanup()

    asyncio.run(main())