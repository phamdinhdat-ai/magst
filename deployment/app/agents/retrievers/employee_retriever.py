import os
import json
import sys
import time
import hashlib
import re
import threading
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
from functools import lru_cache
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
# --- LangChain Imports ---
import gc 

# --- LangChain/Community Imports ---
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, JSONLoader, TextLoader
import asyncio
from app.agents.workflow.initalize import llm_instance, settings, agent_config
from app.utils.document_processor import markdown_splitter, remove_image_tags, DocumentCustomConverter
from app.agents.factory.tools.base import BaseAgentTool
# --- Imports for the Contextual Compression Pattern ---
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.embeddings import HuggingFaceEmbeddings
# GPU memory management constants
GPU_INACTIVITY_TIMEOUT = 300  # 5 minutes
GPU_MEMORY_PRESSURE_THRESHOLD = 0.9  # 90% GPU memory usage
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. GPU memory management disabled.")
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
logger.add("employee_workflow.log", rotation="10 MB")

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
class EmployeeDocumentWatcher(FileSystemEventHandler):
    """File system watcher that triggers document loading for a specific employee."""
    def __init__(self, retriever_tool: 'EmployeeRetrieverTool'):
        self.tool = retriever_tool
        self.employee_id = retriever_tool.employee_id

    def _is_relevant_file(self, file_path_str: str) -> bool:
        """Checks if the file belongs to the employee this watcher is responsible for."""
        filename = os.path.basename(file_path_str)
        # File phải chứa ID của nhân viên để được xử lý
        return f"employee_{self.employee_id}" in filename or filename.startswith(f"{self.employee_id}_")

    def on_created(self, event):
        if not event.is_directory and self._is_relevant_file(event.src_path):
            logger.info(f"[Watcher] New file for employee {self.employee_id}: {event.src_path}")
            time.sleep(1)
            self.tool._process_file_if_needed(Path(event.src_path))

    def on_modified(self, event):
        if not event.is_directory and "_registry.json" not in event.src_path and self._is_relevant_file(event.src_path):
            logger.info(f"[Watcher] File modified for employee {self.employee_id}: {event.src_path}")
            time.sleep(1)
            self.tool._process_file_if_needed(Path(event.src_path))
            
    def on_deleted(self, event):
        if not event.is_directory and self._is_relevant_file(event.src_path):
            logger.info(f"[Watcher] File deleted for employee {self.employee_id}: {event.src_path}")
            self.tool._remove_document_by_path(Path(event.src_path))


class PerformanceCache:
    """High-performance cache for employee retriever with TTL and memory management"""
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

class RetrievedEmployeeDocument(BaseModel):
    content: str
    source: str
    relevance_score: float
    _cached_hash: Optional[str] = None
    
    # Employee-specific term dictionaries for enhanced relevance scoring
    _employee_terms = {
        'performance', 'evaluation', 'review', 'rating', 'assessment', 'feedback',
        'salary', 'wage', 'compensation', 'benefits', 'bonus', 'promotion',
        'contract', 'employment', 'position', 'role', 'responsibility', 'duty',
        'training', 'development', 'skill', 'competency', 'certification',
        'attendance', 'leave', 'vacation', 'sick', 'overtime', 'schedule'
    }
    
    _hr_terms = {
        'human resources', 'hr', 'personnel', 'staff', 'workforce', 'team',
        'department', 'manager', 'supervisor', 'colleague', 'subordinate',
        'policy', 'procedure', 'guideline', 'regulation', 'compliance'
    }
    
    _professional_terms = {
        'project', 'task', 'assignment', 'deadline', 'milestone', 'deliverable',
        'meeting', 'presentation', 'report', 'documentation', 'communication',
        'collaboration', 'leadership', 'teamwork', 'initiative', 'innovation'
    }
    
    @lru_cache(maxsize=128)
    def get_content_hash(self) -> str:
        """Cached content hash for deduplication"""
        if self._cached_hash is None:
            self._cached_hash = hashlib.md5(self.content.encode()).hexdigest()
        return self._cached_hash
    
    def calculate_employee_relevance(self, query_tokens: set) -> float:
        """Calculate employee-specific relevance score based on term matching"""
        content_tokens = set(re.findall(r'\b\w+\b', self.content.lower()))
        
        # Calculate weighted scores
        employee_score = len(query_tokens.intersection(self._employee_terms.intersection(content_tokens))) * 2.0
        hr_score = len(query_tokens.intersection(self._hr_terms.intersection(content_tokens))) * 1.8
        professional_score = len(query_tokens.intersection(self._professional_terms.intersection(content_tokens))) * 1.5
        general_score = len(query_tokens.intersection(content_tokens - self._employee_terms - self._hr_terms - self._professional_terms))
        
        total_score = employee_score + hr_score + professional_score + general_score
        max_possible = len(query_tokens) * 2.0  # Assuming all terms are employee terms
        
        return min(total_score / max_possible if max_possible > 0 else 0.0, 1.0)

class EmployeeRetrieverTool(BaseAgentTool):
    """
    A dynamic and secure retriever for a specific employee's data. It creates a
    dedicated, isolated knowledge base for the employee, which automatically
    updates from a specified directory.
    """
    name: str = "employee_retriever"
    description: str = "Retrieves and reranks a specific employee's data from their isolated, self-updating knowledge base."

    # --- Core Configuration (Pydantic Fields) ---
    employee_id: str = Field(description="The unique identifier for the employee.")
    collection_name: str = Field(description="The isolated ChromaDB collection name for this employee.")
    watch_directory: Path = Field(description="Directory to watch for this employee's documents.")
    
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

    # GPU Memory Management
    _gpu_components_active: bool = PrivateAttr(default=False)
    _gpu_last_used: float = PrivateAttr(default=0.0)
    _gpu_ttl_seconds: float = PrivateAttr(default=300.0)  # 5 minutes before release
    _gpu_lock: threading.RLock = PrivateAttr(default=None)  # Instance lock for GPU operations
    

    def __init__(self, employee_id: str, watch_directory: str, **kwargs):
        if not employee_id:
            raise ValueError("employee_id cannot be empty.")
            
        safe_employee_id = re.sub(r'[^a-zA-Z0-9_.-]', '_', employee_id)
        collection_name = f"employee_{safe_employee_id}_data"
        
        # Sửa lỗi Pydantic bằng cách truyền tất cả các trường vào super().__init__
        init_kwargs = kwargs.copy()
        init_kwargs['employee_id'] = employee_id
        init_kwargs['watch_directory'] = Path(watch_directory).resolve()
        init_kwargs['collection_name'] = self._sanitize_collection_name(collection_name)
        init_kwargs['name'] = f"employee_retriever_{safe_employee_id}"
        init_kwargs['description'] = f"Retrieves data for employee {employee_id}"
        
        super().__init__(**init_kwargs)
        
        # Gán giá trị mặc định cho PrivateAttr
        self._vector_store: Optional[Chroma] = None
        self._bm25_retriever: Optional[BM25Retriever] = None
        self._embeddings: Optional[OllamaEmbeddings] = None
        self._text_splitter: Optional[RecursiveCharacterTextSplitter] = None
        self._observer: Optional[Observer] = None
        self._document_registry: Dict[str, str] = {}
        self._is_initialized: bool = False
        
        
        self._thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix=f"employee_{safe_employee_id}")
        
        # Initialize GPU management attributes
        self._gpu_lock = threading.RLock()
        self._gpu_components_active = False
        self._gpu_last_used = time.time()
        self._gpu_ttl_seconds = 300.0  # 5 minutes
        # Initialize performance components
        self._query_cache = PerformanceCache(max_size=500, ttl_seconds=300)
        self._document_cache = PerformanceCache(max_size=1000, ttl_seconds=600)
        self._embedding_cache = PerformanceCache(max_size=200, ttl_seconds=1800)
        self._thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix=f"emp_{employee_id}")
        self._batch_queue = []
        self._batch_lock = threading.Lock()
        self._last_bm25_update = 0.0
        self._bm25_update_threshold = 60.0
        
        if not self._is_initialized:
            self._initialize_all()

    def _sanitize_collection_name(self, name: str) -> str:
        sanitized = re.sub(r'[^a-zA-Z0-9_.-]', '_', name)
        return sanitized[:63]
    def __getstate__(self):
        """Custom serialization method to handle thread locks."""
        state = self.__dict__.copy()
        
        # Remove unpicklable attributes
        for attr in ['_gpu_lock', '_thread_pool', '_observer', '_embeddings', 
                     '_final_retriever', '_vector_store']:
            if attr in state:
                state[attr] = None
                
        # Make sure we know this was pickled
        state['_was_pickled'] = True
        logger.debug("employeeRetrieverTool prepared for pickling")
        return state
    
    def __setstate__(self, state):
        """Custom deserialization method to restore thread locks."""
        self.__dict__.update(state)
        
        # Create a new lock if needed
        if not hasattr(self, '_gpu_lock') or self._gpu_lock is None:
            self._gpu_lock = threading.RLock()
            
        # Flag that we need to reinitialize if used
        self._gpu_components_active = False
        logger.debug("employeeRetrieverTool unpickled, will reinitialize components when needed")

    def _initialize_all(self):
        logger.info(f"Initializing retriever for employee '{self.employee_id}' (Collection: '{self.collection_name}')...")
        self.watch_directory.mkdir(parents=True, exist_ok=True)
        
        self._initialize_core_components()
        self._load_document_registry()
        self._scan_and_process_all_files()
        # self._build_retriever_pipeline()
        # Don't initialize GPU pipeline yet - do it on demand
        self._final_retriever = None
        self._gpu_components_active = False
        self._start_document_watcher()
        
        self._is_initialized = True
        logger.info(f"Retriever for employee '{self.employee_id}' initialized successfully.")

    def _initialize_core_components(self, use_gpu: bool = False):
        """Initialize core components with GPU option."""
        device = 'cuda' if use_gpu and TORCH_AVAILABLE else 'cpu'
        logger.info(f"Initializing embedding model with device: {device} for employee '{self.employee_id}'")

        # self._embeddings = OllamaEmbeddings(model=settings.EMBEDDING_MODEL, base_url=settings.OLLAMA_BASE_URL)
        model_kwargs = {'device': device}
        encode_kwargs = {'normalize_embeddings': True}
        self._embeddings = HuggingFaceEmbeddings(model_name=settings.HF_EMBEDDING_MODEL, 
                                               model_kwargs=model_kwargs, 
                                               encode_kwargs=encode_kwargs)
        persistent_client = chromadb.PersistentClient(path=str(Path(settings.VECTOR_STORE_BASE_DIR)))
        self._vector_store = Chroma(client=persistent_client, collection_name=self.collection_name, embedding_function=self._embeddings)
        self._text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=256)
        
        # Track GPU status if using GPU
        if use_gpu and TORCH_AVAILABLE:
            self._gpu_components_active = True
            self._gpu_last_used = time.time()
            logger.info(f"Initialized core components on GPU for employee '{self.employee_id}'")
        else:
            self._gpu_components_active = False

    @property
    def _registry_path(self) -> Path:
        return self.watch_directory / f"{self.collection_name}_registry.json"

    def _build_retriever_pipeline(self, force_gpu: bool = True):
        """Builds the final ContextualCompressionRetriever pipeline with GPU management."""
        logger.info(f"Building retriever pipeline for employee '{self.employee_id}' (Vector Search -> Reranker)...")

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
            logger.info(f"Initializing cross-encoder reranker on {device} for employee '{self.employee_id}'")

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

            logger.info(f"Retriever pipeline for employee '{self.employee_id}' built successfully on {device}.")

        except Exception as e:
            logger.error(f"Error building retriever pipeline for employee '{self.employee_id}': {e}. Retrieval may not work properly.")
            self._final_retriever = None
            self._gpu_components_active = False
    
    def _run(self, query: str) -> str:
        """Synchronously retrieves and reranks documents for the employee."""
        try:
            if not self._final_retriever:
                return f"Error: Retriever pipeline for employee {self.employee_id} is not built. Check for initialization errors."

            compressed_docs = self._final_retriever.invoke(query)
            
            if not compressed_docs:
                return f"No relevant information found for employee '{self.employee_id}' with query: '{query}'"

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

    async def _arun(self, query: str) -> str:
        """Asynchronously retrieves and reranks documents for the employee."""
        try:
            if not self._final_retriever:
                return f"Error: Retriever pipeline for employee {self.employee_id} is not built. Check for initialization errors."

            compressed_docs = await self._final_retriever.ainvoke(query)

            if not compressed_docs:
                logger.warning(f"No relevant information found for employee '{self.employee_id}' with query: '{query}'")
                return f"No relevant information found for employee '{self.employee_id}' with query: '{query}'"
            logger.info(f"Retrieved {len(compressed_docs)} documents for employee '{self.employee_id}' with query: '{query}'")
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
    def _release_gpu_resources(self):
        """Release GPU resources to free memory."""
        # Safely check if GPU components are active
        if not getattr(self, '_gpu_components_active', False):
            return
            
        logger.info(f"Releasing GPU resources for employee '{self.employee_id}'...")
        
        # Delete the final retriever which holds GPU resources
        if hasattr(self, '_final_retriever') and self._final_retriever is not None:
            try:
                self._final_retriever = None
                logger.debug(f"Final retriever released for employee '{self.employee_id}'")
            except Exception as e:
                logger.error(f"Error releasing final retriever for employee '{self.employee_id}': {e}")
                
        # Attempt to reinitialize with CPU if needed
        try:
            if hasattr(self, '_initialize_core_components'):
                # Reinitialize with CPU for lighter footprint
                if hasattr(self, '_vector_store') and self._vector_store is not None:
                    self._vector_store = None
                self._initialize_core_components(use_gpu=False)
                logger.debug(f"Reinitialized with CPU components for employee '{self.employee_id}'")
        except Exception as e:
            logger.error(f"Error reinitializing with CPU for employee '{self.employee_id}': {e}")
        
        # Run garbage collection
        try:
            gc.collect()
        except Exception as e:
            logger.error(f"Error during garbage collection for employee '{self.employee_id}': {e}")
        
        # If using PyTorch, clear CUDA cache
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                logger.info(f"CUDA memory cache cleared for employee '{self.employee_id}'")
            except Exception as e:
                logger.error(f"Error clearing CUDA memory for employee '{self.employee_id}': {e}")
        
        self._gpu_components_active = False
        logger.info(f"GPU resources released successfully for employee '{self.employee_id}'")
    
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
            logger.info(f"GPU components inactive or timed out for employee '{self.employee_id}'. Reinitializing...")
            self._release_gpu_resources()
            
            try:
                # Initialize the embeddings with GPU first
                self._initialize_core_components(use_gpu=True)
                
                # Then build the retriever pipeline with GPU
                self._build_retriever_pipeline(force_gpu=True)
                
                return getattr(self, '_gpu_components_active', False)
            except Exception as e:
                logger.error(f"Failed to initialize GPU components for employee '{self.employee_id}': {e}")
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
                logger.warning(f"High GPU memory pressure detected for employee '{self.employee_id}': {memory_allocated:.2f}GB/{memory_reserved:.2f}GB")
                self._release_gpu_resources()
        except Exception as e:
            logger.error(f"Error checking memory pressure for employee '{self.employee_id}': {e}")
            
    def _load_document_registry(self):
        if self._registry_path.exists():
            try:
                with open(self._registry_path, 'r', encoding='utf-8') as f:
                    self._document_registry = json.load(f)
                logger.info(f"Loaded {len(self._document_registry)} entries from registry for employee {self.employee_id}.")
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Failed to load registry: {e}. Starting fresh.")
                self._document_registry = {}

    def _save_document_registry(self):
        try:
            with open(self._registry_path, 'w', encoding='utf-8') as f:
                json.dump(self._document_registry, f, indent=2)
            logger.debug(f"Registry for employee {self.employee_id} saved.")
        except IOError as e:
            logger.error(f"Failed to save registry: {e}")

    def _is_relevant_file(self, file_path: Path) -> bool:
        """Checks if the file belongs to the employee this tool instance is for."""
        filename = file_path.name
        return f"employee_{self.employee_id}" in filename or filename.startswith(f"{self.employee_id}_")

    def _scan_and_process_all_files(self):
        logger.info(f"Performing initial scan for employee '{self.employee_id}' in '{self.watch_directory}'")
        current_employee_files = set()
        for file_path in self.watch_directory.rglob('*'):
            if file_path.is_file() and self._is_relevant_file(file_path):
                current_employee_files.add(str(file_path))
                self._process_file_if_needed(file_path)
        
        registered_files = set(self._document_registry.keys())
        deleted_files = registered_files - current_employee_files
        for file_path_str in deleted_files:
            self._remove_document_by_path(Path(file_path_str))
        self._save_document_registry()

    def _process_file_if_needed(self, file_path: Path):
        if "_registry.json" in file_path.name: return
        current_hash = self._get_file_hash(file_path)
        if not current_hash: return
        if current_hash != self._document_registry.get(str(file_path)):
            logger.info(f"Change detected for '{file_path.name}'. Processing for employee {self.employee_id}...")
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
        event_handler = EmployeeDocumentWatcher(self)
        self._observer.schedule(event_handler, str(self.watch_directory), recursive=True)
        watcher_thread = threading.Thread(target=self._observer.start, daemon=True)
        watcher_thread.start()
        logger.info(f"Started document watcher for employee '{self.employee_id}' on '{self.watch_directory}'.")

    def stop_document_watcher(self):
        if self._observer and self._observer.is_alive():
            self._observer.stop()
            self._observer.join()
            logger.info(f"Document watcher for employee '{self.employee_id}' stopped.")
            self._observer = None

    def _reload_document(self, file_path: Path):
        self._delete_documents_by_source(file_path.name)
        documents = self._load_and_split_file(file_path)
        if documents:
            self._add_documents_to_store(documents)
            new_hash = self._get_file_hash(file_path)
            if new_hash:
                self._document_registry[str(file_path)] = new_hash
                self._save_document_registry()
                logger.info(f"Successfully reloaded '{file_path.name}' for employee {self.employee_id}.")

    def _remove_document_by_path(self, file_path: Path):
        path_str = str(file_path)
        if path_str in self._document_registry:
            logger.info(f"Removing document '{file_path.name}' for employee {self.employee_id}.")
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
                logger.error(f"Error splitting documents from file {file_path.name}: {e}")
                return []
        except Exception: return []

    def _add_documents_to_store(self, docs: List[Document]):
        if not docs: return
        self._vector_store.add_documents(docs)
        self._update_bm25_retriever()

    def _delete_documents_by_source(self, source_filename: str):
        try:
            existing_ids = self._vector_store.get(where={"source": source_filename})['ids']
            if existing_ids:
                logger.info(f"Deleting {len(existing_ids)} old chunks for source '{source_filename}' from employee {self.employee_id}'s collection...")
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
            cache_key = f"bm25_docs_{self.employee_id}"
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
                logger.debug(f"BM25 retriever updated for employee {self.employee_id} with {len(cached_docs)} documents")
            else:
                self._bm25_retriever = None
                
        except Exception as e:
            logger.error(f"Failed to update BM25 retriever for employee {self.employee_id}: {e}")

    def retrieve_documents(self, query: str, use_cache: bool = True) -> List[str]:
        if not self._is_initialized: 
            return [f"Error: Retriever for employee {self.employee_id} is not initialized."]
        
        # Check cache first
        if use_cache:
            cache_key = f"query_{hashlib.md5(query.encode()).hexdigest()}"
            cached_result = self._query_cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for query: {query[:50]}...")
                return cached_result
        
        start_time = time.time()
        
        # Prepare query tokens for employee-specific relevance scoring
        query_tokens = set(re.findall(r'\b\w+\b', query.lower()))
        
        # Parallel hybrid search using dedicated methods
        vector_results = self._parallel_vector_search(query)
        bm25_results = self._parallel_bm25_search(query)
        
        # Collect and deduplicate results using content hash
        hybrid_results = {}
        
        # Process vector search results
        for doc, score in vector_results:
            content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
            if content_hash not in hybrid_results:
                retrieved_doc = RetrievedEmployeeDocument(
                    content=doc.page_content, 
                    source=doc.metadata.get("source", "unknown"), 
                    relevance_score=score
                )
                # Apply employee-specific relevance scoring
                employee_relevance = retrieved_doc.calculate_employee_relevance(query_tokens)
                retrieved_doc.relevance_score = (score * 0.7) + (employee_relevance * 0.3)
                hybrid_results[content_hash] = retrieved_doc
        
        # Process BM25 results
        for doc in bm25_results:
            content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
            if content_hash not in hybrid_results:
                retrieved_doc = RetrievedEmployeeDocument(
                    content=doc.page_content, 
                    source=doc.metadata.get("source", "unknown"), 
                    relevance_score=0.5
                )
                # Apply employee-specific relevance scoring
                employee_relevance = retrieved_doc.calculate_employee_relevance(query_tokens)
                retrieved_doc.relevance_score = (0.5 * 0.7) + (employee_relevance * 0.3)
                hybrid_results[content_hash] = retrieved_doc
            else:
                # Boost score for documents found in both searches
                existing_doc = hybrid_results[content_hash]
                existing_doc.relevance_score = min(existing_doc.relevance_score * 1.2, 1.0)
        
        # Sort by enhanced relevance score
        sorted_docs = sorted(hybrid_results.values(), key=lambda x: x.relevance_score, reverse=True)
        
        if not sorted_docs:
            result = [f"Không tìm thấy thông tin nhân viên phù hợp cho: '{query}'"]
        else:
            result = [f"Source: {doc.source}\nContent: {doc.content}" for doc in sorted_docs[:5]]
        
        # Cache the result
        if use_cache:
            self._query_cache.set(cache_key, result)
        
        processing_time = time.time() - start_time
        logger.debug(f"Query processed in {processing_time:.3f}s for employee {self.employee_id}")
        
        return result
    
    def _parallel_vector_search(self, query: str) -> List[Tuple[Any, float]]:
        """Parallel vector search with timeout and error handling"""
        try:
            future = self._thread_pool.submit(self._get_vector_results, query)
            return future.result(timeout=10.0)
        except Exception as e:
            logger.error(f"Parallel vector search failed for employee {self.employee_id}: {e}")
            return []
    
    def _parallel_bm25_search(self, query: str) -> List[Any]:
        """Parallel BM25 search with timeout and error handling"""
        try:
            future = self._thread_pool.submit(self._get_bm25_results, query)
            return future.result(timeout=10.0)
        except Exception as e:
            logger.error(f"Parallel BM25 search failed for employee {self.employee_id}: {e}")
            return []
    
    def _get_vector_results(self, query: str) -> List[Tuple[Any, float]]:
        """Get vector search results with caching"""
        try:
            return self._vector_store.similarity_search_with_relevance_scores(query, k=5)
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return []
    
    def _get_bm25_results(self, query: str) -> List[Any]:
        """Get BM25 search results"""
        try:
            if self._bm25_retriever:
                return self._bm25_retriever.get_relevant_documents(query)
            return []
        except Exception as e:
            logger.error(f"BM25 search error: {e}")
            return []
    def run(self, query: str) -> str:
        """Synchronous entry point for running the retriever."""
        try:
            # Ensure GPU components are initialized if needed
            if self._ensure_gpu_components():
                result = self._run(query)
            else:
                result = f"Error: Failed to initialize retriever for customer {self.customer_id}"
            return result
        finally:
            # Check memory pressure and release resources if needed
            self._check_memory_pressure()
            
            # Update last used time to prevent premature release
            if getattr(self, '_gpu_components_active', False):
                self._gpu_last_used = time.time()
    
    @gpu_management_decorator
    async def arun(self, query: str) -> str:
        """Asynchronous entry point for running the retriever with GPU memory management."""
        logger.info(f"Asynchronously running customer retriever for query: '{query}'")
        return await self._arun(query)   
    def cleanup(self):
        """Clean up resources including thread pool, document watcher, and GPU resources."""
        logger.info(f"Cleaning up resources for customer retriever '{self.customer_id}'...")
        
        # Release GPU resources first
        try:
            self._release_gpu_resources()
        except Exception as e:
            logger.error(f"Error releasing GPU resources: {e}")
        
        # Stop document watcher
        try:
            if self._observer and self._observer.is_alive(): 
                self._observer.stop()
                self._observer.join(timeout=2)
                logger.info(f"Document watcher stopped for customer '{self.customer_id}'")
        except Exception as e:
            logger.error(f"Error stopping document watcher: {e}")
        
        # Shutdown thread pool
        try:
            if self._thread_pool: 
                self._thread_pool.shutdown(wait=False)
                logger.info(f"Thread pool shut down for customer '{self.customer_id}'")
        except Exception as e:
            logger.error(f"Error shutting down thread pool: {e}")
    
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
                except Exception as e:
                    logger.error(f"Error releasing GPU resources: {e}")
            
            self.cleanup()
        except Exception as e:
            logger.error(f"Error during cleanup in __del__: {e}")
            # Don't re-raise the exception to avoid crashes during garbage collection

    
    async def retrieve_documents_batch(self, queries: List[str]) -> List[List[str]]:
        """Process multiple queries concurrently with enhanced error handling"""
        if not queries:
            return []
        
        logger.info(f"Processing batch of {len(queries)} queries for employee {self.employee_id}")
        start_time = time.time()
        
        # Create semaphore to limit concurrent operations
        semaphore = asyncio.Semaphore(4)  # Limit to 4 concurrent queries
        
        async def process_single_query(query: str) -> List[str]:
            async with semaphore:
                try:
                    return await self._async_retrieve_single(query)
                except Exception as e:
                    logger.error(f"Error processing query '{query[:50]}...': {e}")
                    return [f"Error processing query: {str(e)}"]
        
        # Process all queries concurrently
        tasks = [process_single_query(query) for query in queries]
        results = await asyncio.gather(*tasks)
        
        processing_time = time.time() - start_time
        logger.info(f"Batch processing completed in {processing_time:.3f}s for employee {self.employee_id}")
        
        return results
    
    async def batch_retrieve(self, queries: List[str]) -> List[List[str]]:
        """Legacy method - redirects to retrieve_documents_batch"""
        return await self.retrieve_documents_batch(queries)
    
    async def _async_retrieve_single(self, query: str) -> List[str]:
        """Async wrapper for single query retrieval"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._thread_pool, self.retrieve_documents, query)
    
    def clear_caches(self):
        """Clear all caches to free memory"""
        logger.info(f"Clearing caches for employee {self.employee_id}")
        self._query_cache.clear()
        self._document_cache.clear()
        self._embedding_cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring"""
        return {
            "employee_id": self.employee_id,
            "query_cache_size": len(self._query_cache.cache),
            "document_cache_size": len(self._document_cache.cache),
            "embedding_cache_size": len(self._embedding_cache.cache),
            "last_bm25_update": self._last_bm25_update,
            "is_initialized": self._is_initialized
        }
    
    def cleanup(self):
        """A dedicated method to stop the watcher and clean up resources."""
        logger.info(f"Cleaning up resources for employee retriever '{self.employee_id}'...")
        self.stop_document_watcher()
        
        # Clear caches
        self.clear_caches()
        
        # Shutdown thread pool
        if self._thread_pool:
            self._thread_pool.shutdown(wait=True)
            logger.debug(f"Thread pool shut down for employee {self.employee_id}")
    
    def __del__(self):
        self.cleanup()

# Example usage
if __name__ == '__main__':
    async def main():
        DATA_DIR = Path("./employee_data_test")
        DATA_DIR.mkdir(exist_ok=True, parents=True)
        logger.info(f"Using data directory: {DATA_DIR.resolve()}")

        # Create test files for two different employees
        EMP_A_ID = "nv_001"
        EMP_B_ID = "nv_002"

        with open(DATA_DIR / f"{EMP_A_ID}_contract.txt", "w", encoding="utf-8") as f:
            f.write("Hợp đồng lao động cho nhân viên NV_001. Vị trí: Kỹ sư phần mềm. Mức lương: 30,000,000 VND.")
        with open(DATA_DIR / f"{EMP_B_ID}_performance_review.csv", "w", encoding="utf-8") as f:
            f.write("quarter,rating,comment\n")
            f.write("Q1,Exceeds Expectations,Hoàn thành xuất sắc dự án X\n")
        with open(DATA_DIR / "general_policy.txt", "w", encoding="utf-8") as f:
            f.write("This file should be ignored by both retrievers.")

        # Initialize a retriever ONLY for Employee A
        print("\n" + "="*20 + f" INITIALIZING FOR EMPLOYEE {EMP_A_ID} " + "="*20)
        retriever_A = EmployeeRetrieverTool(employee_id=EMP_A_ID, watch_directory=str(DATA_DIR))
        
        # Initialize a retriever ONLY for Employee B
        print("\n" + "="*20 + f" INITIALIZING FOR EMPLOYEE {EMP_B_ID} " + "="*20)
        retriever_B = EmployeeRetrieverTool(employee_id=EMP_B_ID, watch_directory=str(DATA_DIR))

        # Test retrieval for Employee A
        query_A = "mức lương của nhân viên 001"
        print(f"\n--- Testing for Employee A with query: '{query_A}' ---")
        results_A = await retriever_A._arun(query_A)
        print("Results for A:", results_A)
        assert "30,000,000" in results_A
        assert "dự án X" not in results_A

        # Test retrieval for Employee B
        query_B = "đánh giá hiệu suất của nhân viên 002"
        print(f"\n--- Testing for Employee B with query: '{query_B}' ---")
        results_B = await retriever_B._arun(query_B)
        print("Results for B:", results_B)
        assert "dự án X" in results_B
        assert "30,000,000" not in results_B

        try:
            # Test auto-update (modify Employee B's file)
            print("\n--- Testing auto-update. Modifying Employee B's file... ---")
            time.sleep(2)
            with open(DATA_DIR / f"{EMP_B_ID}_performance_review.csv", "a", encoding="utf-8") as f:
                f.write("Q2,Needs Improvement,Cần cải thiện kỹ năng giao tiếp\n")
            time.sleep(5)

            query_B2 = "kỹ năng giao tiếp"
            print(f"\n--- Re-testing for Employee B after update: '{query_B2}' ---")
            results_B2 = await retriever_B._arun(query_B2)
            print("Updated Results for B:", results_B2)
            assert "giao tiếp" in results_B2
            
            # Test batch processing
            print("\n--- Testing batch processing ---")
            batch_queries = [
                "mức lương nhân viên 001",
                "đánh giá hiệu suất nhân viên 002",
                "kỹ năng giao tiếp"
            ]
            batch_results = await retriever_A.retrieve_documents_batch(batch_queries)
            print(f"Batch processing returned {len(batch_results)} results")
            
            # Test cache statistics
            print("\n--- Cache Statistics ---")
            stats_A = retriever_A.get_cache_stats()
            stats_B = retriever_B.get_cache_stats()
            print(f"Employee A cache stats: {stats_A}")
            print(f"Employee B cache stats: {stats_B}")
            
        finally:
            # Always cleanup resources
            print("\n--- Cleaning up resources ---")
            retriever_A.cleanup()
            retriever_B.cleanup()

    asyncio.run(main())