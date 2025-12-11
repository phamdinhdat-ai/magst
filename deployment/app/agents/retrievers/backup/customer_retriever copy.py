import os
import json
import sys
import time
import hashlib
import re
import threading
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from loguru import logger
from pydantic import Field, BaseModel, PrivateAttr
import chromadb
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# --- LangChain/Community Imports ---
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, JSONLoader, TextLoader
import asyncio
from app.agents.workflow.initalize import llm_instance, settings, agent_config
from app.agents.factory.tools.base import BaseAgentTool


class PerformanceCache:
    """Thread-safe cache with TTL and size limits for performance optimization."""
    
    def __init__(self, ttl_seconds: int = 300, max_size: int = 1000):
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
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
            # Evict oldest entries if at max size
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


class CustomerDocumentWatcher(FileSystemEventHandler):
    """File system watcher that triggers document loading for a specific customer."""
    def __init__(self, retriever_tool: 'CustomerRetrieverTool'):
        self.tool = retriever_tool
        self.customer_id = retriever_tool.customer_id

    def _is_relevant_file(self, file_path_str: str) -> bool:
        """Checks if the file belongs to the customer this watcher is responsible for."""
        filename = os.path.basename(file_path_str)
        # File phải chứa ID của khách hàng để được xử lý
        return f"customer_{self.customer_id}" in filename or filename.startswith(f"{self.customer_id}_")

    def on_created(self, event):
        if not event.is_directory and self._is_relevant_file(event.src_path) and "registry" not in event.src_path:
            logger.info(f"[Watcher] New file for customer {self.customer_id}: {event.src_path}")
            time.sleep(1)
            self.tool._process_file_if_needed(Path(event.src_path))

    def on_modified(self, event):
        if not event.is_directory and "registry" not in event.src_path and self._is_relevant_file(event.src_path):
            logger.info(f"[Watcher] File modified for customer {self.customer_id}: {event.src_path}")
            time.sleep(1)
            self.tool._process_file_if_needed(Path(event.src_path))
            
    def on_deleted(self, event):
        if not event.is_directory and self._is_relevant_file(event.src_path) and "registry" not in event.src_path:
            logger.info(f"[Watcher] File deleted for customer {self.customer_id}: {event.src_path}")
            self.tool._remove_document_by_path(Path(event.src_path))


class RetrievedCustomerDocument(BaseModel):
    content: str
    source: str
    relevance_score: float
    
    # Private attributes for caching
    _content_hash: Optional[str] = PrivateAttr(default=None)
    _processed_tokens: Optional[List[str]] = PrivateAttr(default=None)
    
    # Customer-specific term dictionaries for relevance scoring
    _customer_terms = {
        'personal': ['name', 'address', 'phone', 'email', 'contact', 'personal', 'profile', 'information'],
        'service': ['service', 'product', 'order', 'purchase', 'transaction', 'payment', 'billing', 'account'],
        'support': ['support', 'help', 'issue', 'problem', 'question', 'request', 'ticket', 'complaint'],
        'preference': ['preference', 'setting', 'option', 'choice', 'configuration', 'customization']
    }
    
    def get_content_hash(self) -> str:
        """Get cached content hash for deduplication."""
        if self._content_hash is None:
            self._content_hash = hashlib.md5(self.content.encode()).hexdigest()
        return self._content_hash
    
    def get_processed_tokens(self) -> List[str]:
        """Get cached processed tokens for efficient analysis."""
        if self._processed_tokens is None:
            # Enhanced tokenization for customer-specific terms
            tokens = re.findall(r'\b\w+\b', self.content.lower())
            # Filter out common stop words and keep meaningful terms
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
            self._processed_tokens = [token for token in tokens if len(token) > 2 and token not in stop_words]
        return self._processed_tokens
    
    def calculate_customer_relevance(self, query_tokens: List[str]) -> None:
        """Calculate customer-specific weighted relevance score."""
        doc_tokens = self.get_processed_tokens()
        doc_token_set = set(doc_tokens)
        query_token_set = set(query_tokens)
        
        # Calculate weighted token intersection
        weighted_score = 0.0
        total_matches = 0
        
        for token in query_token_set.intersection(doc_token_set):
            weight = 1.0  # Default weight
            
            # Apply customer-specific term weights
            if token in self._customer_terms['personal']:
                weight = 2.5  # High weight for personal information
            elif token in self._customer_terms['service']:
                weight = 2.0  # High weight for service-related terms
            elif token in self._customer_terms['support']:
                weight = 1.8  # Medium-high weight for support terms
            elif token in self._customer_terms['preference']:
                weight = 1.5  # Medium weight for preference terms
            
            weighted_score += weight
            total_matches += 1
        
        if total_matches > 0:
            # Combine with existing relevance score
            token_relevance = weighted_score / len(query_token_set) if query_token_set else 0
            self.relevance_score = (self.relevance_score * 0.7) + (token_relevance * 0.3)
        
        # Boost score for exact phrase matches
        query_text = ' '.join(query_tokens)
        if query_text in self.content.lower():
            self.relevance_score *= 1.2

class CustomerRetrieverTool(BaseAgentTool):
    """
    A dynamic and secure retriever for a specific customer's data. It creates a
    dedicated, isolated knowledge base for the customer, which automatically
    updates from a specified directory.
    """
    name: str = "customer_retriever"
    description: str = "Retrieves and reranks a specific customer's data from their isolated, self-updating knowledge base."

    # --- Core Configuration (Pydantic Fields) ---
    customer_id: str = Field(description="The unique identifier for the customer.")
    collection_name: str = Field(description="The isolated ChromaDB collection name for this customer.")
    watch_directory: Path = Field(description="Directory to watch for this customer's documents.")
    
    # --- Internal Components (Private Attributes) ---
    _vector_store: Chroma = PrivateAttr(default=None)
    _bm25_retriever: BM25Retriever = PrivateAttr(default=None)
    _embeddings: OllamaEmbeddings = PrivateAttr(default=None)
    _text_splitter: RecursiveCharacterTextSplitter = PrivateAttr(default=None)
    _observer: Observer = PrivateAttr(default=None)
    _document_registry: Dict[str, str] = PrivateAttr(default_factory=dict)
    _is_initialized: bool = PrivateAttr(default=False)
    
    # --- Performance Optimization Components ---
    _query_cache: PerformanceCache = PrivateAttr(default=None)
    _document_cache: PerformanceCache = PrivateAttr(default=None)
    _embedding_cache: PerformanceCache = PrivateAttr(default=None)
    _thread_pool: ThreadPoolExecutor = PrivateAttr(default=None)
    _last_bm25_update: float = PrivateAttr(default=0.0)
    _bm25_update_threshold: float = PrivateAttr(default=60.0)  # 60 seconds

    def __init__(self, customer_id: str = "123", watch_directory: str = "./customer_data", **kwargs):
        
            
        safe_customer_id = re.sub(r'[^a-zA-Z0-9_.-]', '_', customer_id)
        collection_name = f"customer_{safe_customer_id}_data"
        
        init_kwargs = kwargs.copy()
        init_kwargs['customer_id'] = customer_id
        init_kwargs['watch_directory'] = Path(watch_directory).resolve()
        init_kwargs['collection_name'] = self._sanitize_collection_name(collection_name)
        init_kwargs['name'] = f"customer_retriever_{safe_customer_id}"
        init_kwargs['description'] = f"Retrieves data for customer {customer_id}"
        
        # Gọi super().__init__ trước
        super().__init__(**init_kwargs)
        
        # **Initialize private attributes explicitly**
        self._vector_store: Optional[Chroma] = None
        self._bm25_retriever: Optional[BM25Retriever] = None
        self._embeddings: Optional[OllamaEmbeddings] = None
        self._text_splitter: Optional[RecursiveCharacterTextSplitter] = None
        self._observer: Optional[Observer] = None
        self._document_registry: Dict[str, str] = {}
        self._is_initialized: bool = False
        
        # Initialize performance optimization components
        self._query_cache = PerformanceCache(ttl_seconds=300, max_size=500)  # 5 min TTL
        self._document_cache = PerformanceCache(ttl_seconds=600, max_size=1000)  # 10 min TTL
        self._embedding_cache = PerformanceCache(ttl_seconds=1800, max_size=200)  # 30 min TTL
        self._thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix=f"customer_{safe_customer_id}")
        self._last_bm25_update = 0.0
        self._bm25_update_threshold = 60.0
        
        if not self._is_initialized:
            self._initialize_all()

    def _sanitize_collection_name(self, name: str) -> str:
        sanitized = re.sub(r'[^a-zA-Z0-9_.-]', '_', name)
        return sanitized[:63]

    def _initialize_all(self):
        logger.info(f"Initializing retriever for customer '{self.customer_id}' (Collection: '{self.collection_name}')...")
        self.watch_directory.mkdir(parents=True, exist_ok=True)
        
        self._initialize_core_components()
        self._load_document_registry()
        self._scan_and_process_all_files()
        self._start_document_watcher()
        
        self._is_initialized = True
        logger.info(f"Retriever for customer '{self.customer_id}' initialized successfully.")

    def _initialize_core_components(self):
        self._embeddings = OllamaEmbeddings(
            model=settings.EMBEDDING_MODEL, base_url=settings.OLLAMA_BASE_URL
        )
        persistent_client = chromadb.PersistentClient(path=str(Path(settings.VECTOR_STORE_BASE_DIR)))
        self._vector_store = Chroma(
            client=persistent_client, collection_name=self.collection_name, embedding_function=self._embeddings
        )
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        logger.info("Core components initialized.")

    @property
    def _registry_path(self) -> Path:
        # Mỗi khách hàng có một file registry riêng trong cùng thư mục để tránh xung đột
        return self.watch_directory / f"{self.collection_name}_registry.json"

    def _load_document_registry(self):
        if self._registry_path.exists():
            try:
                with open(self._registry_path, 'r', encoding='utf-8') as f:
                    self._document_registry = json.load(f)
                logger.info(f"Loaded {len(self._document_registry)} entries from registry for customer {self.customer_id}.")
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Failed to load registry: {e}. Starting fresh.")
                self._document_registry = {}

    def _save_document_registry(self):
        try:
            with open(self._registry_path, 'w', encoding='utf-8') as f:
                json.dump(self._document_registry, f, indent=2)
            logger.debug(f"Registry for customer {self.customer_id} saved.")
        except IOError as e:
            logger.error(f"Failed to save registry: {e}")

    def _is_relevant_file(self, file_path: Path) -> bool:
        """Checks if the file belongs to the customer this tool instance is for."""
        filename = file_path.name
        return f"customer_{self.customer_id}" in filename or filename.startswith(f"{self.customer_id}_")

    def _scan_and_process_all_files(self):
        logger.info(f"Performing initial scan for customer '{self.customer_id}' in '{self.watch_directory}'")
        current_customer_files = set()
        for file_path in self.watch_directory.rglob('*'):
            if file_path.is_file() and self._is_relevant_file(file_path):
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
            logger.info(f"Change detected for '{file_path.name}'. Processing for customer {self.customer_id}...")
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
        event_handler = CustomerDocumentWatcher(self)
        self._observer.schedule(event_handler, str(self.watch_directory), recursive=True)
        watcher_thread = threading.Thread(target=self._observer.start, daemon=True)
        watcher_thread.start()
        logger.info(f"Started document watcher for customer '{self.customer_id}' on '{self.watch_directory}'.")

    def stop_document_watcher(self):
        if self._observer and self._observer.is_alive():
            self._observer.stop()
            self._observer.join()
            logger.info(f"Document watcher for customer '{self.customer_id}' stopped.")
            self._observer = None # Đặt lại observer sau khi stop

    # Thêm hàm cleanup để quản lý tài nguyên
    def cleanup(self):
        """A dedicated method to stop the watcher and clean up resources."""
        logger.info(f"Cleaning up resources for customer retriever '{self.customer_id}'...")
        self.stop_document_watcher()
    
    
        
        
    def _reload_document(self, file_path: Path):
        self._delete_documents_by_source(file_path.name)
        documents = self._load_and_split_file(file_path)
        if documents:
            self._add_documents_to_store(documents)
            new_hash = self._get_file_hash(file_path)
            if new_hash:
                self._document_registry[str(file_path)] = new_hash
                self._save_document_registry()
                logger.info(f"Successfully reloaded '{file_path.name}' for customer {self.customer_id}.")

    def _remove_document_by_path(self, file_path: Path):
        path_str = str(file_path)
        if path_str in self._document_registry:
            logger.info(f"Removing document '{file_path.name}' for customer {self.customer_id}.")
            self._delete_documents_by_source(file_path.name)
            del self._document_registry[path_str]
            self._save_document_registry()
        
    def _load_and_split_file(self, file_path: Path) -> List[Document]:
        ext = file_path.suffix.lower()
        loader_map = {'.pdf': PyPDFLoader, '.csv': CSVLoader, '.json': JSONLoader, '.txt': TextLoader}
        loader_class = loader_map.get(ext)
        if not loader_class: return []
        try:
            if ext == '.json':
                # JSONLoader expects a file path, not a string
                loader = loader_class(file_path, jq_schema='$.[]')
            elif ext == '.csv':
                loader = loader_class(file_path, autodetect_encoding=True)
            elif ext == '.txt':
                loader = loader_class(file_path)
            else:
                loader = loader_class(file_path)
            raw_docs = loader.load()
            split_docs = self._text_splitter.split_documents(raw_docs)
            for doc in split_docs:
                doc.metadata['source'] = file_path.name
                doc.metadata['customer_id'] = self.customer_id
            logger.info(f"Loaded and split '{file_path.name}' into {len(split_docs)} chunks.")
            return split_docs
        except Exception as e:
            logger.error(f"Failed to load/split file {file_path}: {e}")
            return []

    def _add_documents_to_store(self, docs: List[Document]):
        if not docs: return
        self._vector_store.add_documents(docs)
        self._update_bm25_retriever()

    def _delete_documents_by_source(self, source_filename: str):
        try:
            existing_ids = self._vector_store.get(where={"source": source_filename})['ids']
            if existing_ids:
                logger.info(f"Deleting {len(existing_ids)} old chunks for source '{source_filename}' from customer {self.customer_id}'s collection...")
                self._vector_store.delete(ids=existing_ids)
                self._update_bm25_retriever()
        except Exception as e:
            logger.error(f"Failed to delete documents for source '{source_filename}': {e}")
            
    def _update_bm25_retriever(self, force: bool = False):
        """Update BM25 retriever with throttling and caching to avoid frequent recomputation."""
        current_time = time.time()
        
        # Skip update if not enough time has passed (unless forced)
        if not force and (current_time - self._last_bm25_update) < self._bm25_update_threshold:
            return
        
        try:
            # Check document cache first
            cache_key = f"bm25_docs_{self.customer_id}"
            cached_docs = self._document_cache.get(cache_key)
            
            if cached_docs is None:
                # Retrieve documents from vector store
                all_docs = self._vector_store.get(include=["metadatas", "documents"])
                if all_docs and all_docs['documents']:
                    cached_docs = [Document(page_content=doc, metadata=meta)
                                  for doc, meta in zip(all_docs['documents'], all_docs['metadatas'])]
                    # Cache the documents
                    self._document_cache.set(cache_key, cached_docs)
                else:
                    cached_docs = []
            
            # Update BM25 retriever
            if cached_docs:
                self._bm25_retriever = BM25Retriever.from_documents(cached_docs)
                self._bm25_retriever.k = 5
                logger.debug(f"BM25 retriever updated for customer {self.customer_id} with {len(cached_docs)} documents")
            else:
                self._bm25_retriever = None
                logger.debug(f"BM25 retriever cleared for customer {self.customer_id} (no documents)")
            
            self._last_bm25_update = current_time
            
        except Exception as e:
            logger.error(f"Failed to update BM25 retriever for customer {self.customer_id}: {e}")

    def _parallel_vector_search(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """Perform vector search in parallel with timeout."""
        try:
            future = self._thread_pool.submit(
                self._vector_store.similarity_search_with_relevance_scores, query, k
            )
            return future.result(timeout=10)  # 10 second timeout
        except TimeoutError:
            logger.warning(f"Vector search timed out for customer {self.customer_id} query: {query[:50]}...")
            return []
        except Exception as e:
            logger.error(f"Vector search failed for customer {self.customer_id}: {e}")
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
            logger.warning(f"BM25 search timed out for customer {self.customer_id} query: {query[:50]}...")
            return []
        except Exception as e:
            logger.error(f"BM25 search failed for customer {self.customer_id}: {e}")
            return []

    def retrieve_documents(self, query: str, use_cache: bool = True) -> List[str]:
        """Enhanced retrieval with caching, parallel processing, and customer-specific scoring."""
        if not self._is_initialized:
            return [f"Error: Retriever for customer {self.customer_id} is not initialized."]
        
        start_time = time.time()
        
        # Check cache first
        cache_key = f"query_{self.customer_id}_{hashlib.md5(query.encode()).hexdigest()}"
        if use_cache:
            cached_result = self._query_cache.get(cache_key)
            if cached_result is not None:
                logger.info(f"Cache hit for customer {self.customer_id} query: {query[:50]}... (took {time.time() - start_time:.3f}s)")
                return cached_result
        
        # Update BM25 retriever (throttled)
        self._update_bm25_retriever()
        
        # Parallel search execution
        vector_future = self._thread_pool.submit(self._parallel_vector_search, query, 5)
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
                customer_doc = RetrievedCustomerDocument(
                    content=doc.page_content,
                    source=doc.metadata.get("source", "unknown"),
                    relevance_score=score
                )
                logger.debug(f"Processing vector result for customer {self.customer_id}: {customer_doc.source} (score: {score})")
                content_hash = customer_doc.get_content_hash()
                if content_hash not in seen_hashes:
                    seen_hashes.add(content_hash)
                    hybrid_results[content_hash] = customer_doc
            
            # Process BM25 results
            for doc in bm25_results:
                customer_doc = RetrievedCustomerDocument(
                    content=doc.page_content,
                    source=doc.metadata.get("source", "unknown"),
                    relevance_score=0.5
                )
                
                content_hash = customer_doc.get_content_hash()
                if content_hash not in seen_hashes:
                    seen_hashes.add(content_hash)
                    hybrid_results[content_hash] = customer_doc
            
            # Apply customer-specific relevance scoring
            query_tokens = re.findall(r'\b\w+\b', query.lower())
            for customer_doc in hybrid_results.values():
                customer_doc.calculate_customer_relevance(query_tokens)
            
            # Sort and format results
            sorted_docs = sorted(hybrid_results.values(), key=lambda x: x.relevance_score, reverse=True)
            
            if not sorted_docs:
                result = [f"No relevant information found for customer '{self.customer_id}' with query: '{query}'"]
            else:
                result = [f"Source: {doc.source}\nContent: {doc.content}" for doc in sorted_docs[:5]]
            
            # Cache the result
            if use_cache:
                self._query_cache.set(cache_key, result)
            
            processing_time = time.time() - start_time
            logger.info(f"Customer {self.customer_id} retrieval completed for query: {query[:50]}... "
                       f"(took {processing_time:.3f}s, found {len(sorted_docs)} docs)")
            
            return result
            
        except Exception as e:
            logger.error(f"Error during customer {self.customer_id} document retrieval: {e}")
            return [f"Error retrieving documents for customer {self.customer_id}: {str(e)}"]
    
    async def retrieve_documents_batch(self, queries: List[str], use_cache: bool = True) -> Dict[str, List[str]]:
        """Process multiple queries in parallel with async support."""
        if not self._is_initialized:
            error_msg = f"Error: Retriever for customer {self.customer_id} is not initialized."
            return {query: [error_msg] for query in queries}
        
        start_time = time.time()
        logger.info(f"Starting batch processing for customer {self.customer_id} with {len(queries)} queries")
        
        async def process_query(query: str) -> Tuple[str, List[str]]:
            try:
                # Run in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self._thread_pool, self.retrieve_documents, query, use_cache
                )
                return query, result
            except Exception as e:
                logger.error(f"Error processing query '{query}' for customer {self.customer_id}: {e}")
                return query, [f"Error processing query: {str(e)}"]
        
        # Process all queries concurrently
        tasks = [process_query(query) for query in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Format results
        batch_results = {}
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Batch processing exception for customer {self.customer_id}: {result}")
                continue
            query, docs = result
            batch_results[query] = docs
        
        processing_time = time.time() - start_time
        logger.info(f"Batch processing completed for customer {self.customer_id}: "
                   f"{len(batch_results)} queries in {processing_time:.3f}s")
        
        return batch_results
    
    def clear_caches(self):
        """Clear all performance caches."""
        if self._query_cache:
            self._query_cache.clear()
        if self._document_cache:
            self._document_cache.clear()
        if self._embedding_cache:
            self._embedding_cache.clear()
        logger.info(f"All caches cleared for customer {self.customer_id}")
    
    def get_cache_stats(self) -> Dict[str, Dict[str, int]]:
        """Get statistics for all caches."""
        return {
            "query_cache": self._query_cache.get_stats() if self._query_cache else {},
            "document_cache": self._document_cache.get_stats() if self._document_cache else {},
            "embedding_cache": self._embedding_cache.get_stats() if self._embedding_cache else {}
        }
    
    def cleanup(self):
        """Clean up resources including thread pool and caches."""
        try:
            # Stop file watcher
            if self._observer and self._observer.is_alive():
                self._observer.stop()
                self._observer.join(timeout=5)
            
            # Shutdown thread pool
            if self._thread_pool:
                self._thread_pool.shutdown(wait=True, timeout=10)
            
            # Clear caches
            self.clear_caches()
            
            logger.info(f"Customer retriever {self.customer_id} cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during customer {self.customer_id} cleanup: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except Exception:
            pass  # Ignore errors during destruction

    def _run(self, query: str) -> str:
        results = self.retrieve_documents(query)
        return "\n\n".join(results)

    async def _arun(self, query: str) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._run, query)

# Example usage
if __name__ == '__main__':
    async def main():
        DATA_DIR = Path("./customer_data_test")
        DATA_DIR.mkdir(exist_ok=True, parents=True)
        logger.info(f"Using data directory: {DATA_DIR.resolve()}")

        # Create test files for two different customers
        CUST_A_ID = "cust_123"
        CUST_B_ID = "cust_456"

        with open(DATA_DIR / f"{CUST_A_ID}_report.txt", "w", encoding="utf-8") as f:
            f.write("Báo cáo sức khỏe cho khách hàng 123. Nguy cơ tiểu đường loại 2 ở mức trung bình.")
        with open(DATA_DIR / f"{CUST_B_ID}_genetic_results.csv", "w", encoding="utf-8") as f:
            f.write("gene,result\n")
            f.write("BRCA1,Negative\n")
        with open(DATA_DIR / "some_other_file.txt", "w", encoding="utf-8") as f:
            f.write("This file should be ignored by both retrievers.")


        # Initialize a retriever ONLY for Customer A
        print("\n" + "="*20 + f" INITIALIZING FOR CUSTOMER {CUST_A_ID} " + "="*20)
        retriever_A = CustomerRetrieverTool(customer_id=CUST_A_ID, watch_directory=str(DATA_DIR))
        
        # Initialize a retriever ONLY for Customer B
        print("\n" + "="*20 + f" INITIALIZING FOR CUSTOMER {CUST_B_ID} " + "="*20)
        retriever_B = CustomerRetrieverTool(customer_id=CUST_B_ID, watch_directory=str(DATA_DIR))

        try:
            # Test retrieval for Customer A
            query_A = "nguy cơ tiểu đường"
            print(f"\n--- Testing for Customer A with query: '{query_A}' ---")
            results_A = await retriever_A._arun(query_A)
            print("Results for A:", results_A)
            assert "tiểu đường" in results_A # Should find the result
            assert "BRCA1" not in results_A # Should NOT find B's data

            # Test retrieval for Customer B
            query_B = "kết quả gen BRCA1"
            print(f"\n--- Testing for Customer B with query: '{query_B}' ---")
            results_B = await retriever_B._arun(query_B)
            print("Results for B:", results_B)
            assert "BRCA1" in results_B # Should find the result
            assert "tiểu đường" not in results_B # Should NOT find A's data
            
            # Test auto-update (modify Customer A's file)
            print("\n--- Testing auto-update. Modifying Customer A's file... ---")
            time.sleep(2)
            with open(DATA_DIR / f"{CUST_A_ID}_report.txt", "a", encoding="utf-8") as f:
                f.write("\nCập nhật: Có nguy cơ dị ứng với đậu phộng.")
            time.sleep(5)
            
            # Re-test after update
            query_A2 = "dị ứng đậu phộng"
            print(f"\n--- Re-testing for Customer A after update: '{query_A2}' ---")
            results_A2 = await retriever_A._arun(query_A2)
            print("Updated Results for A:", results_A2)
            assert "đậu phộng" in results_A2
            
            # Test cache stats
            print("\n--- Cache Statistics ---")
            stats_A = retriever_A.get_cache_stats()
            stats_B = retriever_B.get_cache_stats()
            print(f"Customer A cache stats: {stats_A}")
            print(f"Customer B cache stats: {stats_B}")
            
        finally:
            # Always cleanup resources
            print("\n--- Cleaning up resources ---")
            retriever_A.cleanup()
            retriever_B.cleanup()

    asyncio.run(main())