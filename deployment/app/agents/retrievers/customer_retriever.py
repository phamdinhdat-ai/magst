import os
import json
import sys
import time
import hashlib
import re
import threading
import gc
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from functools import wraps

from loguru import logger
from pydantic import Field, BaseModel, PrivateAttr
import chromadb

# --- Import for GPU memory management ---
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. GPU memory management disabled.")
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# --- LangChain/Community Imports ---
from langchain_core.retrievers import BaseRetriever
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
# BM25Retriever is no longer needed
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, JSONLoader, TextLoader
import asyncio

# --- Imports for the Contextual Compression Pattern ---
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from app.utils.document_processor import  DocumentCustomConverter, markdown_splitter, remove_image_tags


from app.agents.workflow.initalize import llm_instance, settings, agent_config
from app.agents.factory.tools.base import BaseAgentTool

# GPU memory management constants
GPU_INACTIVITY_TIMEOUT = 300  # 5 minutes
GPU_MEMORY_PRESSURE_THRESHOLD = 0.9  # 90% GPU memory usage

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

# PerformanceCache and CustomerDocumentWatcher remain unchanged
class PerformanceCache:
    """Thread-safe cache with TTL and size limits for performance optimization."""
    def __init__(self, ttl_seconds: int = 300, max_size: int = 1000):
        self.ttl_seconds = ttl_seconds; self.max_size = max_size; self.cache = {}; self.timestamps = {}; self.lock = threading.RLock()
    def get(self, key: str):
        with self.lock:
            if key not in self.cache: return None
            if time.time() - self.timestamps[key] > self.ttl_seconds:
                del self.cache[key]; del self.timestamps[key]; return None
            return self.cache[key]
    def set(self, key: str, value):
        with self.lock:
            if len(self.cache) >= self.max_size and key not in self.cache:
                oldest_key = min(self.timestamps.keys(), key=lambda k: self.timestamps[k])
                del self.cache[oldest_key]; del self.timestamps[oldest_key]
            self.cache[key] = value; self.timestamps[key] = time.time()

class CustomerDocumentWatcher(FileSystemEventHandler):
    """File system watcher that triggers document loading for a specific customer."""
    def __init__(self, retriever_tool: 'CustomerRetrieverTool'):
        self.tool = retriever_tool; self.customer_id = retriever_tool.customer_id
    def _is_relevant_file(self, file_path_str: str) -> bool:
        filename = os.path.basename(file_path_str)
        return f"customer_{self.customer_id}" in filename or filename.startswith(f"{self.customer_id}_")
    def on_created(self, event):
        if not event.is_directory and self._is_relevant_file(event.src_path) and "registry" not in event.src_path:
            time.sleep(1); self.tool._process_file_if_needed(Path(event.src_path))
    def on_modified(self, event):
        if not event.is_directory and "registry" not in event.src_path and self._is_relevant_file(event.src_path):
            time.sleep(1); self.tool._process_file_if_needed(Path(event.src_path))
    def on_deleted(self, event):
        if not event.is_directory and self._is_relevant_file(event.src_path) and "registry" not in event.src_path:
            self.tool._remove_document_by_path(Path(event.src_path))



class CustomerRetrieverTool(BaseAgentTool):
    """
    A dynamic, secure retriever for customer data. It uses a LangChain
    ContextualCompressionRetriever with a vector store base and a cross-encoder
    reranker to find the most relevant information.
    
    Includes GPU memory optimization with lazy loading and automatic cleanup.
    """
    name: str = "customer_retriever"
    description: str = "Retrieves and reranks a specific customer's data from their isolated, self-updating knowledge base using vector search."

    # Core Configuration
    customer_id: str
    collection_name: str
    watch_directory: Path
    
    # Internal components
    _vector_store: Any = PrivateAttr(default=None)
    _embeddings: Any = PrivateAttr(default=None)
    _text_splitter: Any = PrivateAttr(default=None)
    _final_retriever: Any = PrivateAttr(default=None)

    _observer: Observer = PrivateAttr(default=None) # type: ignore
    _document_registry: Dict[str, str] = PrivateAttr(default_factory=dict)
    _is_initialized: bool = PrivateAttr(default=False)
    _thread_pool: ThreadPoolExecutor = PrivateAttr(default=None)
    
    # GPU Memory Management
    _gpu_components_active: bool = PrivateAttr(default=False)
    _gpu_last_used: float = PrivateAttr(default=0.0)
    _gpu_ttl_seconds: float = PrivateAttr(default=300.0)  # 5 minutes before release
    _gpu_lock: threading.RLock = PrivateAttr(default=None)  # Instance lock for GPU operations
    
    def __init__(self, customer_id: str = "123", watch_directory: str = "./customer_data", **kwargs):
        safe_customer_id = re.sub(r'[^a-zA-Z0-9_.-]', '_', customer_id)
        collection_name = f"customer_{safe_customer_id}_data"
        
        init_kwargs = kwargs.copy()
        init_kwargs['customer_id'] = customer_id
        init_kwargs['watch_directory'] = Path(watch_directory).resolve()
        init_kwargs['collection_name'] = self._sanitize_collection_name(collection_name)
        init_kwargs['name'] = f"customer_retriever_{safe_customer_id}"
        init_kwargs['description'] = f"Retrieves data for customer {customer_id}"
        
        super().__init__(**init_kwargs)
        
        # Initialize private attributes
        self._vector_store = None
        self._embeddings = None
        self._text_splitter = None
        self._final_retriever = None
        self._observer = None
        self._document_registry = {}
        self._is_initialized = False
        self._thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix=f"customer_{safe_customer_id}")
        
        # Initialize GPU management attributes
        self._gpu_lock = threading.RLock()
        self._gpu_components_active = False
        self._gpu_last_used = time.time()
        self._gpu_ttl_seconds = 300.0  # 5 minutes
        
        if not self._is_initialized:
            self._initialize_all()
            
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
        logger.debug("CustomerRetrieverTool prepared for pickling")
        return state
    
    def __setstate__(self, state):
        """Custom deserialization method to restore thread locks."""
        self.__dict__.update(state)
        
        # Create a new lock if needed
        if not hasattr(self, '_gpu_lock') or self._gpu_lock is None:
            self._gpu_lock = threading.RLock()
            
        # Flag that we need to reinitialize if used
        self._gpu_components_active = False
        logger.debug("CustomerRetrieverTool unpickled, will reinitialize components when needed")

    def _initialize_all(self):
        """Initializes all components and sets up the final retriever pipeline."""
        logger.info(f"Initializing retriever for customer '{self.customer_id}'...")
        self.watch_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize core components without GPU first - more efficient
        self._initialize_core_components(use_gpu=False)
        self._load_document_registry()
        
        # Don't initialize GPU pipeline yet - do it on demand
        self._final_retriever = None
        self._gpu_components_active = False
        
        self._scan_and_process_all_files() # This populates the vector store
        self._start_document_watcher()
        
        self._is_initialized = True
        logger.info(f"Retriever for customer '{self.customer_id}' initialized successfully (GPU components deferred).")

    def _build_retriever_pipeline(self, force_gpu: bool = True):
        """Builds the final ContextualCompressionRetriever pipeline with GPU management."""
        logger.info(f"Building retriever pipeline for customer '{self.customer_id}' (Vector Search -> Reranker)...")
        
        # A. Create the base retriever directly from the vector store.
        # This is the first stage: get a broad set of potentially relevant docs.
        base_retriever = self._vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 7},  # Get top 7 results from vector search
            return_source_documents=True
        )
        
        # Release any existing GPU resources before building new ones
        self._release_gpu_resources()
        
        # Try GPU first, then fallback to CPU if there are CUDA memory issues
        devices_to_try = ['cuda', 'cpu'] if force_gpu and TORCH_AVAILABLE else ['cpu']
        
        for device in devices_to_try:
            try:
                # B. Create the reranker model and compressor.
                # This is the second stage: accurately re-score the candidates.
                logger.info(f"Attempting to initialize cross-encoder reranker on {device} for customer '{self.customer_id}'")
                
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
                else:
                    self._gpu_components_active = False
                    
                logger.info(f"Retriever pipeline for customer '{self.customer_id}' built successfully on {device}.")
                return  # Success, exit the loop
                
            except Exception as e:
                error_msg = str(e)
                if device == 'cuda' and ('CUDA out of memory' in error_msg or 'out of memory' in error_msg.lower()):
                    logger.warning(f"CUDA out of memory for customer '{self.customer_id}': {e}. Trying CPU fallback...")
                    # Clear CUDA cache before trying CPU
                    if TORCH_AVAILABLE and torch.cuda.is_available():
                        try:
                            torch.cuda.empty_cache()
                            gc.collect()
                        except Exception:
                            pass
                    continue  # Try next device (CPU)
                else:
                    logger.error(f"Error building retriever pipeline on {device} for customer '{self.customer_id}': {e}")
                    if device == devices_to_try[-1]:  # Last device to try
                        # If all devices failed, fall back to base retriever without reranking
                        logger.warning(f"All devices failed for customer '{self.customer_id}'. Falling back to base retriever without reranking.")
                        self._final_retriever = base_retriever
                        self._gpu_components_active = False
                        return
                    continue
        
        # If we get here, something went wrong
        logger.error(f"Failed to build retriever pipeline for customer '{self.customer_id}' on any device.")
        self._final_retriever = base_retriever  # Fallback to base retriever
        self._gpu_components_active = False

    def _run(self, query: str) -> str:
        """Synchronously retrieves and reranks documents for the customer."""
        try:
            if not self._final_retriever:
                return f"Error: Retriever pipeline for customer {self.customer_id} is not built. Check for initialization errors."
            start_time = time.time()
            compressed_docs = self._final_retriever.invoke(query)
            logger.info(f"Retrieved {len(compressed_docs)} documents for customer '{self.customer_id}' with query: '{query}' in {time.time() - start_time:.2f}s")
            if not compressed_docs:
                return f"No relevant information found for customer '{self.customer_id}' with query: '{query}'"
            
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
        """Asynchronously retrieves and reranks documents for the customer."""
        try:
            if not self._final_retriever:
                return f"Error: Retriever pipeline for customer {self.customer_id} is not built. Check for initialization errors."
            try:
                compressed_docs = await self._final_retriever.ainvoke(query)
                
                logger.info(f"Database: {self._vector_store}")
                # logger.info(f"Similar docs: {self._vector_store.similar_docs}")
                logger.info(f"{self._final_retriever}")
                logger.info(f"Compressed docs: {compressed_docs}")
            except Exception as e:
                logger.error(f"Error retrieving documents for customer '{self.customer_id}' with query: '{query}': {e}")
                return f"Error retrieving documents for customer '{self.customer_id}' with query: '{query}': {e}"
            # If no documents found, return a message

            logger.info(f"Retrieved {len(compressed_docs)} documents for customer '{self.customer_id}' with query: '{query}' in collection '{self.collection_name}'")
            if not compressed_docs:
                logger.warning(f"No relevant information found for customer '{self.customer_id}' with query: '{query}'")
                return f"No relevant information found for customer '{self.customer_id}' with query: '{query}'"
            logger.info(f"Retrieved {len(compressed_docs)} documents for customer '{self.customer_id}' with query: '{query}'")
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

    # --- BM25 related methods have been removed ---

    def _initialize_core_components(self, use_gpu: bool = False):
        """Initialize core components with GPU option and CUDA fallback."""
        # Try GPU first if requested, then fallback to CPU on CUDA memory errors
        devices_to_try = ['cuda', 'cpu'] if use_gpu and TORCH_AVAILABLE else ['cpu']
        
        for device in devices_to_try:
            try:
                logger.info(f"Attempting to initialize embedding model with device: {device} for customer '{self.customer_id}'")
                
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
                if device == 'cuda':
                    self._gpu_components_active = True
                    self._gpu_last_used = time.time()
                    logger.info(f"Initialized core components on GPU for customer '{self.customer_id}'")
                else:
                    self._gpu_components_active = False
                    logger.info(f"Initialized core components on CPU for customer '{self.customer_id}'")
                
                return  # Success, exit the loop
                
            except Exception as e:
                error_msg = str(e)
                if device == 'cuda' and ('CUDA out of memory' in error_msg or 'out of memory' in error_msg.lower()):
                    logger.warning(f"CUDA out of memory during core component initialization for customer '{self.customer_id}': {e}. Trying CPU fallback...")
                    # Clear CUDA cache before trying CPU
                    if TORCH_AVAILABLE and torch.cuda.is_available():
                        try:
                            torch.cuda.empty_cache()
                            gc.collect()
                        except Exception:
                            pass
                    continue  # Try next device (CPU)
                else:
                    logger.error(f"Error initializing core components on {device} for customer '{self.customer_id}': {e}")
                    if device == devices_to_try[-1]:  # Last device to try
                        raise e  # Re-raise the exception if all devices failed
                    continue
        
        # If we get here, all devices failed
        raise Exception(f"Failed to initialize core components for customer '{self.customer_id}' on any device.")

    def _sanitize_collection_name(self, name: str) -> str:
        return re.sub(r'[^a-zA-Z0-9_.-]', '_', name)[:63]

    @property
    def _registry_path(self) -> Path:
        return self.watch_directory / f"{self.collection_name}_registry.json"

    def _load_document_registry(self):
        if self._registry_path.exists():
            try:
                with open(self._registry_path, 'r', encoding='utf-8') as f: self._document_registry = json.load(f)
            except Exception: self._document_registry = {}

    def _save_document_registry(self):
        try:
            with open(self._registry_path, 'w', encoding='utf-8') as f: json.dump(self._document_registry, f, indent=2)
        except Exception as e: logger.error(f"Failed to save registry: {e}")

    def _is_relevant_file(self, file_path: Path) -> bool:
        filename = file_path.name
        return f"customer_{self.customer_id}" in filename or filename.startswith(f"{self.customer_id}_")

    def _scan_and_process_all_files(self):
        current_customer_files = set()
        if self.watch_directory.exists():
            for file_path in self.watch_directory.rglob('*'):
                if file_path.is_file() and self._is_relevant_file(file_path) and "_registry.json" not in file_path.name:
                    current_customer_files.add(str(file_path))
                    self._process_file_if_needed(file_path)
        registered_files = set(self._document_registry.keys())
        deleted_files = registered_files - current_customer_files
        for file_path_str in deleted_files:
            self._remove_document_by_path(Path(file_path_str))
        self._save_document_registry()

    def _process_file_if_needed(self, file_path: Path):
        if "_registry.json" in file_path.name: return
        current_hash = self._get_file_hash(file_path)
        if not current_hash: return
        if current_hash != self._document_registry.get(str(file_path)):
            self._reload_document(file_path)

    def _get_file_hash(self, file_path: Path) -> Optional[str]:
        try:
            with open(file_path, 'rb') as f: return hashlib.md5(f.read()).hexdigest()
        except Exception: return None

    def _start_document_watcher(self):
        
        if self._observer: return
        self._observer = Observer()
        self._observer.schedule(CustomerDocumentWatcher(self), str(self.watch_directory), recursive=True)
        self._observer.start()

    def _reload_document(self, file_path: Path):

        self._delete_documents_by_source(file_path.name)
        documents = self._load_and_split_file(file_path)
        if documents:
            self._add_documents_to_store(documents)
            new_hash = self._get_file_hash(file_path)
            if new_hash:
                self._document_registry[str(file_path)] = new_hash
                self._save_document_registry()

    def _remove_document_by_path(self, file_path: Path):
        path_str = str(file_path)
        if path_str in self._document_registry:
            self._delete_documents_by_source(file_path.name)
            del self._document_registry[path_str]
            self._save_document_registry()
    
    def _release_gpu_resources(self):
        """Release GPU resources to free memory."""
        # Safely check if GPU components are active
        if not getattr(self, '_gpu_components_active', False):
            return
            
        logger.info(f"Releasing GPU resources for customer '{self.customer_id}'...")
        
        # Delete the final retriever which holds GPU resources
        if hasattr(self, '_final_retriever') and self._final_retriever is not None:
            try:
                self._final_retriever = None
                logger.debug(f"Final retriever released for customer '{self.customer_id}'")
            except Exception as e:
                logger.error(f"Error releasing final retriever for customer '{self.customer_id}': {e}")
                
        # Attempt to reinitialize with CPU if needed
        try:
            if hasattr(self, '_initialize_core_components'):
                # Reinitialize with CPU for lighter footprint
                if hasattr(self, '_vector_store') and self._vector_store is not None:
                    self._vector_store = None
                self._initialize_core_components(use_gpu=False)
                logger.debug(f"Reinitialized with CPU components for customer '{self.customer_id}'")
        except Exception as e:
            logger.error(f"Error reinitializing with CPU for customer '{self.customer_id}': {e}")
        
        # Run garbage collection
        try:
            gc.collect()
        except Exception as e:
            logger.error(f"Error during garbage collection for customer '{self.customer_id}': {e}")
        
        # If using PyTorch, clear CUDA cache
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                logger.info(f"CUDA memory cache cleared for customer '{self.customer_id}'")
            except Exception as e:
                logger.error(f"Error clearing CUDA memory for customer '{self.customer_id}': {e}")
        
        self._gpu_components_active = False
        logger.info(f"GPU resources released successfully for customer '{self.customer_id}'")
    
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
            logger.info(f"GPU components inactive or timed out for customer '{self.customer_id}'. Reinitializing...")
            self._release_gpu_resources()
            
            # Try GPU first, then fallback to CPU
            for try_gpu in [True, False]:
                try:
                    # Initialize the embeddings 
                    self._initialize_core_components(use_gpu=try_gpu)
                    
                    # Then build the retriever pipeline
                    self._build_retriever_pipeline(force_gpu=try_gpu)
                    
                    device_type = "GPU" if try_gpu else "CPU"
                    logger.info(f"Successfully initialized components on {device_type} for customer '{self.customer_id}'")
                    return True
                    
                except Exception as e:
                    error_msg = str(e)
                    if try_gpu and ('CUDA out of memory' in error_msg or 'out of memory' in error_msg.lower()):
                        logger.warning(f"CUDA out of memory during component initialization for customer '{self.customer_id}': {e}. Trying CPU fallback...")
                        # Clear CUDA cache before trying CPU
                        if TORCH_AVAILABLE and torch.cuda.is_available():
                            try:
                                torch.cuda.empty_cache()
                                gc.collect()
                            except Exception:
                                pass
                        continue  # Try CPU
                    else:
                        logger.error(f"Failed to initialize components for customer '{self.customer_id}': {e}")
                        if not try_gpu:  # Already on CPU, this is the final fallback
                            return False
                        continue  # Try CPU if this was GPU attempt
            
            # All attempts failed
            logger.error(f"Failed to initialize components on any device for customer '{self.customer_id}'")
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
                logger.warning(f"High GPU memory pressure detected for customer '{self.customer_id}': {memory_allocated:.2f}GB/{memory_reserved:.2f}GB")
                self._release_gpu_resources()
        except Exception as e:
            logger.error(f"Error checking memory pressure for customer '{self.customer_id}': {e}")
            
    def _load_and_split_file(self, file_path: Path) -> List[Document]:
        loader_map = {'.pdf': DocumentCustomConverter, '.csv': CSVLoader, '.json': JSONLoader, '.txt': TextLoader}
        loader_class = loader_map.get(file_path.suffix.lower())
        if not loader_class: 
            logger.warning(f"No loader found for file type: {file_path.suffix}")
            return []
        try:
            logger.info(f"Loading file: {file_path.name} with {loader_class.__name__}")
            loader = loader_class(str(file_path))
            raw_docs = loader.load()
            logger.info(f"Loaded {len(raw_docs)} documents from {file_path.name}")
            
            if raw_docs and len(raw_docs) > 0:
                # Handle case where loader returns strings instead of Document objects
                first_doc = raw_docs[0]
                if isinstance(first_doc, str):
                    logger.info(f"Loader returned strings, converting to Document objects")
                    from langchain_core.documents import Document
                    raw_docs = [Document(page_content=doc, metadata={"source": file_path.name}) for doc in raw_docs]
                    first_doc = raw_docs[0]
                
                logger.info(f"First document preview: {first_doc.page_content[:100]}...")
            else:
                logger.warning(f"No documents loaded from {file_path.name}")
                return []
                
            if "pdf" in file_path.suffix.lower():
                logger.info(f"Processing PDF file: {file_path.name}")
                # split markdown file - get the text content from the first document
                first_doc_content = raw_docs[0].page_content if hasattr(raw_docs[0], 'page_content') else str(raw_docs[0])
                cleaned_text = remove_image_tags(first_doc_content)
                logger.info(f"Cleaned text length: {len(cleaned_text)} characters")
                try: 
                    raw_docs = markdown_splitter(cleaned_text)
                except Exception as e:
                    logger.error(f"Error splitting markdown from PDF file {file_path.name}: {e}")
                logger.info(f"Split {len(raw_docs)} sections from PDF file: {file_path.name}")
            split_docs = self._text_splitter.split_documents(raw_docs)
            for doc in split_docs:
                doc.metadata['source'] = file_path.name; doc.metadata['customer_id'] = self.customer_id
            logger.info(f"Successfully processed {len(split_docs)} document chunks from {file_path.name}")
            return split_docs
        except Exception as e:
            logger.error(f"Error loading and splitting file {file_path.name}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []

    
    
    def _add_documents_to_store(self, docs: List[Document]):
        if not docs: return
        self._vector_store.add_documents(docs)
        # No need to update BM25 anymore

    def _delete_documents_by_source(self, source_filename: str):
        try:
            existing_docs = self._vector_store.get(where={"source": source_filename})
            if existing_docs and existing_docs['ids']:
                self._vector_store.delete(ids=existing_docs['ids'])
        except Exception: pass

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

    
    
if __name__ == '__main__':
    logger.remove(); logger.add(sys.stderr, level="INFO")
    
    async def main():
        DATA_DIR = Path("./customer_data_test")
        DATA_DIR.mkdir(exist_ok=True, parents=True)
        logger.info(f"Using data directory: {DATA_DIR.resolve()}")

        CUST_A_ID = "cust_123"
        CUST_B_ID = "cust_456"

        with open(DATA_DIR / f"{CUST_A_ID}_report.txt", "w", encoding="utf-8") as f:
            f.write("Báo cáo sức khỏe cho khách hàng 123. Nguy cơ tiểu đường loại 2 ở mức trung bình. Không có dấu hiệu của bệnh tim mạch.")
        with open(DATA_DIR / f"{CUST_B_ID}_genetic_results.csv", "w", encoding="utf-8") as f:
            f.write("gene,result\nBRCA1,Negative\nCFTR,Positive")

        print("\n" + "="*20 + f" INITIALIZING FOR CUSTOMER {CUST_A_ID} " + "="*20)
        retriever_A = CustomerRetrieverTool(customer_id=CUST_A_ID, watch_directory=str(DATA_DIR))
        
        print("\n" + "="*20 + f" INITIALIZING FOR CUSTOMER {CUST_B_ID} " + "="*20)
        retriever_B = CustomerRetrieverTool(customer_id=CUST_B_ID, watch_directory=str(DATA_DIR))

        try:
            query_A = "thông tin bệnh tim mạch"
            print(f"\n--- Testing for Customer A with query: '{query_A}' ---")
            results_A = await retriever_A._arun(query_A)
            print("Results for A:\n", results_A)
            assert "tim mạch" in results_A and "BRCA1" not in results_A

            query_B = "kết quả gen CFTR"
            print(f"\n--- Testing for Customer B with query: '{query_B}' ---")
            results_B = await retriever_B._arun(query_B)
            print("Results for B:\n", results_B)
            assert "CFTR" in results_B and "tiểu đường" not in results_B
            
            print("\n--- Testing auto-update. Modifying Customer A's file... ---")
            time.sleep(2)
            with open(DATA_DIR / f"{CUST_A_ID}_report.txt", "a", encoding="utf-8") as f: f.write("\nCập nhật: Có nguy cơ dị ứng với đậu phộng.")
            time.sleep(5) # Give watcher time to process the change
            
            query_A2 = "dị ứng đậu phộng"
            print(f"\n--- Re-testing for Customer A after update: '{query_A2}' ---")
            results_A2 = await retriever_A._arun(query_A2)
            print("Updated Results for A:\n", results_A2)
            assert "đậu phộng" in results_A2
            
        finally:
            print("\n--- Cleaning up resources ---")
            retriever_A.cleanup(); retriever_B.cleanup()
            import shutil
            if DATA_DIR.exists(): shutil.rmtree(DATA_DIR)
            db_dir = Path(settings.VECTOR_STORE_BASE_DIR)
            if db_dir.exists(): shutil.rmtree(db_dir)
            print("Test directories cleaned up.")

    # To run this, you need to install:
    # pip install langchain-community chromadb loguru pydantic watchdog ollama torch sentence-transformers
    # The `sentence-transformers` package is required by `HuggingFaceCrossEncoder`.
    asyncio.run(main())