import os
import json
import sys
import time
import hashlib
import re
import threading
import asyncio
import gc
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from functools import wraps

from loguru import logger
from pydantic import Field, BaseModel, PrivateAttr
import chromadb
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Import torch for GPU memory management if available
TORCH_AVAILABLE = False
try:
    import torch
    TORCH_AVAILABLE = True
    logger.info("PyTorch is available for GPU memory management")
except ImportError:
    logger.warning("PyTorch not available - some GPU memory management features will be disabled")

# GPU Memory management constants
GPU_TTL_SECONDS = 300  # Time to live for GPU resources
GPU_MEMORY_PRESSURE_THRESHOLD = 0.8  # Threshold for releasing GPU resources

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
from app.agents.factory.tools.search_tool import SearchTool
from app.utils.document_processor import markdown_splitter, remove_image_tags, DocumentCustomConverter
# --- Imports for the Contextual Compression Pattern ---
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.embeddings import HuggingFaceEmbeddings


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
    def __init__(self, retriever_tool: 'DrugRetrieverTool'):
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


class RetrievedDrugDocument(BaseModel):
    """Represents a retrieved drug document with metadata and drug-specific scoring."""
    content: str
    source: str
    relevance_score: float
    drug_name: str = ""
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
            # Enhanced tokenization for drug terms
            self._processed_tokens = re.findall(r'\b\w+\b', self.content.lower())
        return self._processed_tokens
    
    def calculate_drug_relevance(self, query_tokens: List[str]) -> float:
        """Calculate relevance with drug-specific scoring"""
        content_tokens = set(self.get_processed_tokens())
        query_token_set = set(query_tokens)
        
        # Drug-specific term weights
        drug_terms = {'drug', 'medication', 'medicine', 'pharmaceutical', 'tablet', 'capsule', 'injection', 'dosage', 'prescription', 'treatment'}
        medical_terms = {'disease', 'symptom', 'condition', 'therapy', 'clinical', 'patient', 'doctor', 'hospital', 'health', 'care'}
        chemical_terms = {'compound', 'molecule', 'chemical', 'formula', 'structure', 'synthesis', 'reaction', 'element', 'acid', 'base'}
        
        # Calculate weighted intersection
        intersection_score = 0.0
        for token in query_token_set & content_tokens:
            if token in drug_terms:
                intersection_score += 2.5  # Highest weight for drug terms
            elif token in medical_terms:
                intersection_score += 2.0  # High weight for medical terms
            elif token in chemical_terms:
                intersection_score += 1.8  # High weight for chemical terms
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


class DrugRetrieverTool(BaseAgentTool):
    """
    An advanced retriever for drug information that automatically ingests
    and updates documents, and uses a hybrid search approach.
    """
    name: str = "drug_retriever"
    description: str = "Retrieves and reranks drug information from a self-updating knowledge base."

    # --- Core Configuration (Pydantic Fields) ---
    collection_name: str = Field(description="Name for the ChromaDB collection.")
    watch_directory: Path = Field(description="Directory to watch for new/updated documents.")
    
    # --- Internal Components (Private Attributes) ---
    _vector_store: Optional[Chroma] = PrivateAttr(default=None)
    _bm25_retriever: Optional[BM25Retriever] = PrivateAttr(default=None)
    _embeddings: Optional[Any] = PrivateAttr(default=None)
    _text_splitter: Optional[RecursiveCharacterTextSplitter] = PrivateAttr(default=None)
    _observer: Observer = PrivateAttr(default=None)
    _document_registry: Dict[str, str] = PrivateAttr(default_factory=dict)
    _is_initialized: bool = PrivateAttr(default=False)
    
    # --- Performance Caches ---
    _query_cache: Optional[PerformanceCache] = PrivateAttr(default=None)
    _document_cache: Optional[PerformanceCache] = PrivateAttr(default=None)
    _embedding_cache: Optional[PerformanceCache] = PrivateAttr(default=None)
    
    # --- Parallel Processing ---
    _thread_pool: Optional[ThreadPoolExecutor] = PrivateAttr(default=None)
    _batch_queue: List = PrivateAttr(default_factory=list)
    _batch_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)
    
    # --- BM25 Throttling ---
    _last_bm25_update: float = PrivateAttr(default=0.0)
    _bm25_update_threshold: float = PrivateAttr(default=60.0)
    
    # --- GPU Management ---
    _gpu_components_active: bool = PrivateAttr(default=False)
    _gpu_last_used: float = PrivateAttr(default=0.0)
    _gpu_ttl_seconds: int = PrivateAttr(default=GPU_TTL_SECONDS)
    _gpu_lock: threading.RLock = PrivateAttr(default=None)
    _final_retriever: Any = PrivateAttr(default=None)
    _was_pickled: bool = PrivateAttr(default=False)


    def __init__(self, watch_directory: str, collection_name: str = "drug_docs", **kwargs):
        # Sửa lỗi Pydantic bằng cách truyền tất cả các trường vào super().__init__
        init_kwargs = kwargs.copy()
        init_kwargs['watch_directory'] = Path(watch_directory).resolve()
        init_kwargs['collection_name'] = self._sanitize_collection_name(collection_name)
        
        super().__init__(**init_kwargs)
        
        # Gán giá trị mặc định cho PrivateAttr
        self._vector_store = None
        self._bm25_retriever = None
        self._embeddings = None
        self._text_splitter = None
        self._observer: Optional[Observer] = None
        self._document_registry = {}
        self._is_initialized = False
        
        # GPU management attributes
        self._gpu_components_active = False
        self._gpu_last_used = 0.0
        self._gpu_ttl_seconds = GPU_TTL_SECONDS
        self._gpu_lock = threading.RLock()
        self._final_retriever = None
        self._was_pickled = False
        
        if not self._is_initialized:
            self._initialize_all()
            
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

    def _sanitize_collection_name(self, name: str) -> str:
        sanitized = re.sub(r'[^a-zA-Z0-9_.-]', '_', name)
        return sanitized[:63]

    def _initialize_all(self):
        logger.info(f"Initializing DrugRetrieverTool for collection '{self.collection_name}'...")
        self.watch_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize performance caches
        self._query_cache = PerformanceCache(max_size=500, ttl_seconds=300)
        self._document_cache = PerformanceCache(max_size=1000, ttl_seconds=600)
        self._embedding_cache = PerformanceCache(max_size=200, ttl_seconds=1800)
        
        # Initialize thread pool
        self._thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="drug_retriever")
        
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
        logger.info("DrugRetrieverTool initialized successfully with CPU components. GPU components will be loaded when needed.")

    def _initialize_core_components(self, use_gpu: bool = False):
        """Initialize core components with optional GPU support."""
        logger.info(f"Initializing core components for drug retriever (GPU: {use_gpu})")
        
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
            embedding_function=self._embeddings
        )
        
        # Initialize text splitter
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024, chunk_overlap=256
        )
        
        # Track GPU status if using GPU
        if use_gpu and TORCH_AVAILABLE and torch.cuda.is_available():
            self._gpu_components_active = True
            self._gpu_last_used = time.time()
            logger.info(f"Initialized core components on GPU for drug retriever")
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
                logger.info("DrugRetriever is not initialized or retriever pipeline is not built.")
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
                return "Công cụ DrugRetriever gặp lỗi trong quá trình truy xuất dữ liệu. Vui lòng thử lại sau."
        finally:
            # Check memory pressure and release resources if needed
            self._check_memory_pressure()
            
            # Update last used time to prevent premature release
            if getattr(self, '_gpu_components_active', False):
                self._gpu_last_used = time.time()
        

    async def _arun(self, query: str) -> str:
        """Asynchronously retrieves and reranks documents for the customer."""
        try:
            if not self._final_retriever:
                logger.info("DrugRetriever is not initialized or retriever pipeline is not built.")
                return "Công cụ DrugRetriever gặp lỗi. Vui lòng thử lại sau."
            try:
                # Use the async invoke method for the final retriever
                compressed_docs = await self._final_retriever.ainvoke(query)
            except Exception as e:
                logger.error(f"Error during async retrieval: {e}")
                return "Công cụ DrugRetriever gặp lỗi trong quá trình truy xuất dữ liệu. Vui lòng thử lại sau."
    
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
                self._process_file_if_needed(file_path)
        
        registered_files = set(self._document_registry.keys())
        deleted_files = registered_files - current_files
        for file_path_str in deleted_files:
            self._remove_document_by_path(Path(file_path_str))
        self._save_document_registry()

    def _process_file_if_needed(self, file_path: Path):
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

    def _stop_document_watcher(self):
        if self._observer and self._observer.is_alive():
            self._observer.stop()
            self._observer.join()
            logger.info("Document watcher stopped.")
            self._observer = None

    def _reload_document(self, file_path: Path):
        try:
            self._delete_documents_by_source(file_path.name)
            documents = self._load_and_split_file(file_path)
            if documents:
                self._add_documents_to_store(documents)
                new_hash = self._get_file_hash(file_path)
                if new_hash:
                    self._document_registry[str(file_path)] = new_hash
                    self._save_document_registry()
                    logger.info(f"Successfully reloaded and registered '{file_path.name}'.")
        except Exception as e:
            logger.error(f"Failed to reload document '{file_path.name}': {e}")

    def _remove_document_by_path(self, file_path: Path):
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
        """Update BM25 retriever with throttling and caching."""
        current_time = time.time()
        
        # Check if update is needed (throttling)
        if not force and (current_time - self._last_bm25_update) < self._bm25_update_threshold:
            return
        
        try:
            # Check document cache first
            cache_key = "bm25_documents"
            cached_docs = self._document_cache.get(cache_key) if self._document_cache else None
            
            if cached_docs is None:
                all_docs = self._vector_store.get(include=["metadatas", "documents"])
                if all_docs and all_docs['documents']:
                    docs_for_bm25 = [Document(page_content=doc, metadata=meta)
                                     for doc, meta in zip(all_docs['documents'], all_docs['metadatas'])]
                    # Cache the documents
                    if self._document_cache:
                        self._document_cache.set(cache_key, docs_for_bm25)
                else:
                    docs_for_bm25 = []
            else:
                docs_for_bm25 = cached_docs
            
            if docs_for_bm25:
                self._bm25_retriever = BM25Retriever.from_documents(docs_for_bm25)
                self._bm25_retriever.k = 5
                logger.debug(f"BM25 retriever updated with {len(docs_for_bm25)} documents")
            else:
                self._bm25_retriever = None
                logger.warning("No documents available for BM25 retriever")
            
            self._last_bm25_update = current_time
            
        except Exception as e:
            logger.error(f"Failed to update BM25 retriever: {e}")

    def _parallel_vector_search(self, query: str, k: int = 5) -> List[Tuple]:
        """Perform vector search in parallel with timeout."""
        try:
            future = self._thread_pool.submit(
                self._vector_store.similarity_search_with_relevance_scores, query, k
            )
            return future.result(timeout=10.0)
        except TimeoutError:
            logger.warning(f"Vector search timed out for query: {query[:50]}...")
            return []
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    def _parallel_bm25_search(self, query: str) -> List:
        """Perform BM25 search in parallel with timeout."""
        if not self._bm25_retriever:
            return []
        
        try:
            future = self._thread_pool.submit(
                self._bm25_retriever.get_relevant_documents, query
            )
            return future.result(timeout=10.0)
        except TimeoutError:
            logger.warning(f"BM25 search timed out for query: {query[:50]}...")
            return []
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []

    def retrieve_documents(self, query: str, use_cache: bool = True) -> List[str]:
        """Enhanced retrieve with caching, parallel processing, and drug-specific scoring."""
        if not self._is_initialized:
            return ["Error: Drug retriever is not initialized."]
        
        start_time = time.time()
        
        # Check cache first
        cache_key = f"query_{hashlib.md5(query.encode()).hexdigest()}"
        if use_cache and self._query_cache:
            cached_result = self._query_cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for query: {query[:50]}...")
                return cached_result
        
        # Update BM25 retriever (throttled)
        self._update_bm25_retriever()
        
        # Parallel search execution
        vector_future = self._thread_pool.submit(self._parallel_vector_search, query, 5)
        bm25_future = self._thread_pool.submit(self._parallel_bm25_search, query)
        
        try:
            vector_results = vector_future.result(timeout=12.0)
            bm25_results = bm25_future.result(timeout=12.0)
        except TimeoutError:
            logger.warning(f"Search operations timed out for query: {query[:50]}...")
            vector_results, bm25_results = [], []
        
        # Process results with deduplication
        hybrid_results = {}
        seen_hashes = set()
        
        # Process vector results
        for doc, score in vector_results:
            drug_doc = RetrievedDrugDocument(
                content=doc.page_content,
                source=doc.metadata.get("source", "unknown"),
                relevance_score=score,
                drug_name=doc.metadata.get("drug_name", ""),
                category=doc.metadata.get("category", "")
            )
            
            content_hash = drug_doc.get_content_hash()
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                hybrid_results[doc.page_content] = drug_doc
        
        # Process BM25 results
        for doc in bm25_results:
            if doc.page_content not in hybrid_results:
                drug_doc = RetrievedDrugDocument(
                    content=doc.page_content,
                    source=doc.metadata.get("source", "unknown"),
                    relevance_score=0.5,
                    drug_name=doc.metadata.get("drug_name", ""),
                    category=doc.metadata.get("category", "")
                )
                
                content_hash = drug_doc.get_content_hash()
                if content_hash not in seen_hashes:
                    seen_hashes.add(content_hash)
                    hybrid_results[doc.page_content] = drug_doc
        
        # Apply drug-specific relevance scoring
        query_tokens = re.findall(r'\b\w+\b', query.lower())
        for drug_doc in hybrid_results.values():
            drug_doc.calculate_drug_relevance(query_tokens)
        
        # Sort by enhanced relevance score
        sorted_docs = sorted(hybrid_results.values(), key=lambda x: x.relevance_score, reverse=True)
        
        # Format results
        if not sorted_docs:
            result = [f"No relevant drug information found for: '{query}'"]
        else:
            result = [f"Source: {doc.source}\nContent: {doc.content}" for doc in sorted_docs[:5]]
        
        # Cache the result
        if use_cache and self._query_cache:
            self._query_cache.set(cache_key, result)
        
        processing_time = time.time() - start_time
        logger.info(f"Drug retrieval completed in {processing_time:.2f}s for query: {query[:50]}...")
        
        return result
    def run(self, query: str) -> str:
        """Synchronous entry point for running the retriever."""
        return self._run(query)
    @gpu_management_decorator
    async def arun(self, query: str) -> str:
        """Asynchronous entry point for running the retriever."""
        logger.info(f"Asynchronously running drug retriever for query: '{query}'")
        return await self._arun(query)
    
    @gpu_management_decorator
    async def batch_retrieve(self, queries: List[str]) -> List[List[str]]:
        """Process multiple queries in parallel with error handling and GPU optimization."""
        if not queries:
            return []
        
        if not self._is_initialized:
            return [["Error: Drug retriever is not initialized."] for _ in queries]
        
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
            
            # Handle any exceptions in results
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
        
        return processed_results
    
    def clear_caches(self):
        """Clear all performance caches."""
        if self._query_cache:
            self._query_cache.clear()
        if self._document_cache:
            self._document_cache.clear()
        if self._embedding_cache:
            self._embedding_cache.clear()
        logger.info("All caches cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring."""
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
    
    def _release_gpu_resources(self):
        """Release GPU resources to free memory."""
        # Safely check if GPU components are active
        if not getattr(self, '_gpu_components_active', False):
            return
            
        logger.info("Releasing GPU resources for drug retriever...")
        
        # Delete the final retriever which holds GPU resources
        if hasattr(self, '_final_retriever') and self._final_retriever is not None:
            try:
                self._final_retriever = None
                logger.debug("Final retriever released for drug retriever")
            except Exception as e:
                logger.error(f"Error releasing final retriever for drug retriever: {e}")
                
        # Attempt to reinitialize with CPU if needed
        try:
            if hasattr(self, '_initialize_core_components'):
                # Reinitialize with CPU for lighter footprint
                if hasattr(self, '_vector_store') and self._vector_store is not None:
                    self._vector_store = None
                self._initialize_core_components(use_gpu=False)
                logger.debug("Reinitialized with CPU components for drug retriever")
        except Exception as e:
            logger.error(f"Error reinitializing with CPU for drug retriever: {e}")
        
        # Run garbage collection
        try:
            gc.collect()
        except Exception as e:
            logger.error(f"Error during garbage collection for drug retriever: {e}")
        
        # If using PyTorch, clear CUDA cache
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                logger.info("CUDA memory cache cleared for drug retriever")
            except Exception as e:
                logger.error(f"Error clearing CUDA cache: {e}")
                
        # Update status
        self._gpu_components_active = False
        logger.info("GPU resources released successfully for drug retriever")
    
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
            logger.info("GPU components inactive or timed out for drug retriever. Reinitializing...")
            self._release_gpu_resources()
            
            try:
                # Initialize the embeddings with GPU first
                self._initialize_core_components(use_gpu=True)
                
                # Then build the retriever pipeline with GPU
                self._build_retriever_pipeline(force_gpu=True)
                
                return getattr(self, '_gpu_components_active', False)
            except Exception as e:
                logger.error(f"Failed to initialize GPU components for drug retriever: {e}")
                return False
    
    def _check_memory_pressure(self):
        """Check GPU memory pressure and release resources if needed."""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return
            
        try:
            # Check current memory usage
            memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
            memory_reserved = torch.cuda.memory_reserved() / (1024 ** 3)  # GB
            
            # If memory pressure is high, release resources
            if memory_allocated > GPU_MEMORY_PRESSURE_THRESHOLD * memory_reserved:
                logger.warning(f"High GPU memory pressure detected for drug retriever: {memory_allocated:.2f}GB/{memory_reserved:.2f}GB")
                self._release_gpu_resources()
        except Exception as e:
            logger.error(f"Error checking memory pressure for drug retriever: {e}")
            
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
        """A dedicated method to stop the watcher and clean up resources."""
        logger.info("Cleaning up resources for drug retriever...")
        
        # Release GPU resources first
        try:
            self._release_gpu_resources()
        except Exception as e:
            logger.error(f"Error releasing GPU resources: {e}")
            
        # Stop document watcher
        self._stop_document_watcher()
    
    def __del__(self):
        """Cleanup resources on object destruction."""
        try:
            # Shutdown thread pool
            if hasattr(self, '_thread_pool') and self._thread_pool:
                self._thread_pool.shutdown(wait=False)
            
            # Stop document watcher
            if hasattr(self, '_observer') and self._observer:
                self._stop_document_watcher()
                
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            
    async def manually_ensure_gpu_ready(self):
        """
        Manually ensure GPU resources are initialized.
        Use this before expected high-volume query periods.
        """
        logger.info("Manually ensuring GPU components are ready for drug retriever")
        return self._ensure_gpu_components()
        
    async def manually_release_gpu(self):
        """
        Manually release GPU resources.
        Use this after periods of inactivity to free up GPU memory.
        """
        logger.info("Manually releasing GPU resources for drug retriever")
        self._release_gpu_resources()
        return {"status": "success", "message": "GPU resources released successfully for drug retriever"}


# Example usage
if __name__ == '__main__':
    async def main():
        DATA_DIR = Path("./drug_data_test")
        DATA_DIR.mkdir(exist_ok=True, parents=True)
        logger.info(f"Using data directory: {DATA_DIR.resolve()}")

        # Create test files
        with open(DATA_DIR / "warfarin_cpic.txt", "w", encoding="utf-8") as f:
            f.write("Hướng dẫn CPIC cho Warfarin. Liều lượng nên được điều chỉnh dựa trên kiểu gen VKORC1 và CYP2C9.")
        with open(DATA_DIR / "fda_drug_list.csv", "w", encoding="utf-8") as f:
            f.write("drug_name,approval_date\n")
            f.write("Aspirin,1988-03-04\n")
            f.write("Ibuprofen,1974-05-13\n")

        # Initialize the tool
        drug_tool = DrugRetrieverTool(watch_directory=str(DATA_DIR), collection_name="drug_test_collection")
        
        # Test initial retrieval
        query = "liều lượng warfarin"
        print(f"\n--- Testing query: '{query}' ---")
        print(await drug_tool._arun(query))

        # Test another query
        query2 = "thuốc aspirin"
        print(f"\n--- Testing query: '{query2}' ---")
        print(await drug_tool._arun(query2))
        
        # Test auto-update (new file)
        print("\n--- Testing auto-update. Adding a new file... ---")
        time.sleep(2)
        with open(DATA_DIR / "paracetamol_info.json", "w", encoding="utf-8") as f:
            json.dump({"drug_name": "Paracetamol", "usage": "Giảm đau, hạ sốt."}, f, ensure_ascii=False)
        time.sleep(5)

        query3 = "công dụng paracetamol"
        print(f"\n--- Testing query after new file: '{query3}' ---")
        print(await drug_tool._arun(query3))

    asyncio.run(main())