# --- Enhanced Dynamic Customer Retriever Tool ---
import os
import sys
import time
import hashlib
import json
import pandas as pd
import csv
from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from loguru import logger
from pydantic import Field, BaseModel
import numpy as np
import re
from pathlib import Path

# --- File System Watching ---
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# --- Document Processing ---
from pypdf import PdfReader

# --- Vector Store and Embedding Imports ---
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Tool Imports ---
sys.path.append(str(Path(__file__).parent.parent.parent))
# from app.agents.tools.base import BaseAgentTool
# from app.core.config import get_settings

from agents.tools.base import BaseAgentTool
from core.config import get_settings

logger = logger.bind(name="customer_retriever")

settings = get_settings()

class CustomerDocumentWatcher(FileSystemEventHandler):
    """File system watcher for automatic customer document loading"""
    
    def __init__(self, retriever):
        self.retriever = retriever
        self.supported_extensions = {'.txt', '.pdf', '.csv', '.json'}
        
    def on_created(self, event):
        if not event.is_directory:
            file_path = event.src_path
            if any(file_path.endswith(ext) for ext in self.supported_extensions):
                if self._is_customer_file(file_path):
                    logger.info(f"New customer file detected: {file_path}")
                    self.retriever.load_new_document(file_path)
                
    def on_modified(self, event):
        if not event.is_directory:
            file_path = event.src_path
            if any(file_path.endswith(ext) for ext in self.supported_extensions) and "registry" not in file_path:
                if self._is_customer_file(file_path):
                    logger.info(f"Customer file modified: {file_path}")
                    self.retriever.reload_document(file_path)
    
    def _is_customer_file(self, file_path: str) -> bool:
        """Check if the file is a customer-specific file"""
        filename = os.path.basename(file_path)
        return (f"customer_{self.retriever.customer_id}" in filename or
                filename.startswith(f"{self.retriever.customer_id}_"))

class CustomerData(BaseModel):
    data_type: str
    data_id: str
    name: str
    category: str = ""
    expert_advice: str = ""
    introduction: str = ""
    conclusion: str = ""
    full_context: str = ""

class RetrievedCustomerDocument(BaseModel):

    customer_id : str = Field(..., description="Customer ID for specific data retrieval")
    content: str = Field(..., description="Content of the retrieved document")
    source: str = Field(..., description="Source of the document")
    retrieval_score: float = Field(..., description="Initial retrieval score from the vector store")
    data_type: str = Field(..., description="Type of data (e.g., trait, condition)")
    data_id: str = Field(..., description="Unique identifier for the data")
    data_name: str = Field(default="", description="Name of the data item")
    data_category: str = Field(default="", description="Category of the data item")
    
    def calculate_customer_relevance(self, query: str, customer_id: str = "") -> float:
        """Calculate relevance score based on customer-specific query"""
        # Simple relevance calculation based on content similarity
        query_lower = query.lower()
        content_lower = self.content.lower()
        
        # Basic scoring based on keyword matches
        query_words = set(query_lower.split())
        content_words = set(content_lower.split())
        
        if not query_words:
            return self.retrieval_score
        
        # Calculate word overlap
        overlap = len(query_words.intersection(content_words))
        overlap_score = overlap / len(query_words)
        
        # Combine with original retrieval score
        combined_score = (self.retrieval_score * 0.7) + (overlap_score * 0.3)
        
        # Update the retrieval score
        self.retrieval_score = min(1.0, combined_score)
        return self.retrieval_score

class CustomerRetrieverTool(BaseAgentTool):

    # Define Pydantic fields
    customer_id: str = Field(..., description="Customer ID for specific data retrieval")
    collection_name: str = Field(default="", description="Collection name for the retriever")
    use_bm25: bool = Field(default=True, description="Whether to use BM25 retriever")
    is_initialized: bool = Field(default=False, description="Initialization status")
    vector_store: Any = Field(default=None, description="Vector store instance")
    bm25_retriever: Any = Field(default=None, description="BM25 retriever instance")
    embeddings: Any = Field(default=None, description="Embeddings model")
    customer_docs: Dict[str, CustomerData] = Field(default_factory=dict, description="Customer-specific documents and traits")
    customers_storage_path: str = Field(default="", description="Path to customer storage directory")
    
   # New fields for enhanced functionality
    watch_directory: str = Field(default="", description="Directory to watch for new documents")
    document_registry: Dict[str, str] = Field(default_factory=dict, description="Registry of loaded documents with hashes")
    observer: Any = Field(default=None, description="File system observer")
    text_splitter: Any = Field(default=None, description="Text splitter for large documents")
    auto_scan_enabled: bool = Field(default=True, description="Whether automatic scanning is enabled")
    supported_formats: List[str] = Field(default_factory=lambda: ['.txt', '.pdf', '.csv', '.json'], description="Supported document formats")
    files_loaded: List[str] = Field(default_factory=list, description="List of loaded document files")
    last_scan_time: float = Field(default_factory=lambda: time.time(), description="Timestamp of the last scan for new documents")  
    name: str = Field(default="customer_retriever", description="Tool name")
    description: str = Field(default="Retrieve customer-specific medical information and health data", 
                           description="Tool description")

    def __init__(self, customer_id: str, use_bm25: bool = True, auto_scan_enabled: bool = True, watch_directory: str = None, **kwargs):
        # Set up customer-specific collection name
        safe_customer_id = customer_id.replace("-", "_").replace(" ", "_")
        collection_name = f"customer_{safe_customer_id}_data"
        
        # Set customers storage path and watch directory
        customers_storage_path = os.path.join(os.path.dirname(__file__), 'storages', 'customers')
        watch_directory = watch_directory or customers_storage_path  # Watch the customers directory

        # Initialize tool with customer-specific name and description
        name = f"customer_retriever_{safe_customer_id}"
        description = f"Dynamic customer retrieval tool for medical information and health data for customer {customer_id} with multi-format support"
        
        super().__init__(
            name=name,
            description=description,
            customer_id=customer_id,
            collection_name=collection_name,
            use_bm25=use_bm25,
            customers_storage_path=customers_storage_path,
            watch_directory=watch_directory,
            auto_scan_enabled=auto_scan_enabled,
            **kwargs
        )
        
        # Initialize text splitter for document processing
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=100,
            length_function=len,
        )
        
        # Initialize timestamp
        self.last_scan_time = time.time()
        self.load_document_registry()
        # Initialize components and load customer data
        self._initialize_components()
        
        
        # Start dynamic file watching if enabled
        if self.auto_scan_enabled:
            self._start_document_watcher()
            self.scan_for_new_documents()
    
    
    
    def _initialize_components(self):
        """Initialize vector store and other components"""
        try:
            # Setup persistent client for ChromaDB
            persistent_client = chromadb.PersistentClient(path=settings.VECTOR_STORE_BASE_DIR)
            logger.info(f"ChromaDB client initialized for customer {self.customer_id}")
            
            # Setup embeddings model
            self.embeddings = OllamaEmbeddings(
                model=settings.EMBEDDING_MODEL,
                base_url=settings.OLLAMA_BASE_URL
            )
            
            # Ensure collection exists
            try:
                collection = persistent_client.get_or_create_collection(self.collection_name)
                logger.info(f"Using customer collection: {self.collection_name}")
            except Exception as e:
                logger.error(f"Error with customer collection: {e}")
                raise
            
            # Initialize vector store
            self.vector_store = Chroma(
                client=persistent_client,
                collection_name=self.collection_name,
                embedding_function=self.embeddings
            )
            
            # Setup BM25 retriever if requested
            if self.use_bm25:
                try:
                    # Get all documents for BM25 indexing
                    all_docs = self.vector_store.get()
                    if all_docs and 'documents' in all_docs and all_docs['documents']:
                        documents = [
                            Document(
                                page_content=doc_text,
                                metadata=all_docs.get('metadatas', [{}])[i] if i < len(all_docs.get('metadatas', [])) else {}
                            )
                            for i, doc_text in enumerate(all_docs['documents'])
                        ]
                        
                        self.bm25_retriever = BM25Retriever.from_documents(documents)
                        logger.info(f"BM25 retriever initialized with {len(documents)} customer documents")
                    else:
                        logger.warning("No customer documents available for BM25 indexing")
                        self.bm25_retriever = None
                except Exception as e:
                    logger.warning(f"BM25 initialization failed: {e}")
                    self.bm25_retriever = None
            
            # Mark initialization as successful
            self.is_initialized = True
            logger.info(f"Customer retriever initialized successfully for customer {self.customer_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize customer retriever: {e}")
            self.is_initialized = False
            self.vector_store = None
            self.bm25_retriever = None

    def _load_customer_data(self):
        """Load customer-specific data from the appropriate customer file"""
        if not self.is_initialized:
            logger.warning("Cannot load customer data: Retriever not initialized")
            return
        # check if not have documents in collection
        if self.vector_store.get().get('documents', []) and self.vector_store.get().get('metadatas', []):
            logger.info(f"Customer {self.customer_id} already has documents in the vector store")
            return
        
        
        try:
            logger.info(f"Checking existing documents for customer {self.customer_id}")
            self._parse_customer_docs_from_vectorstore()
        except Exception as e:
            logger.warning(f"Error checking existing documents: {e}")
        
        # Try to find customer-specific files
        customer_files = self._find_customer_files()
        
        if not customer_files:
            logger.warning(f"No customer files found for customer {self.customer_id}")
            return
        else: 
            logger.info(f"Found {len(customer_files)} customer files for customer {self.customer_id}")
        
        # Load documents from customer files
        for file_path in customer_files:
            try:
                self.load_new_document(file_path)
                logger.info(f"Loaded customer file: {file_path}")
            except Exception as e:
                logger.error(f"Failed to load customer file {file_path}: {e}")

    def _find_customer_files(self) -> List[str]:
        """Find all files belonging to this customer"""
        customer_files = []
        
        if not os.path.exists(self.customers_storage_path):
            logger.warning(f"Customer storage path does not exist: {self.customers_storage_path}")
            return customer_files
        
        for filename in os.listdir(self.customers_storage_path):
            file_path = os.path.join(self.customers_storage_path, filename)
            
            # Check if file belongs to this customer
            if (f"customer_{self.customer_id}" in filename or 
                filename.startswith(f"customer_{self.customer_id}_")):
                
                # Check if it's a supported format
                if any(filename.endswith(ext) for ext in self.supported_formats):
                    customer_files.append(file_path)
        
        return customer_files

    def _parse_customer_docs_from_vectorstore(self):
        try:
            all_docs = self.vector_store.get()
            if not all_docs or 'documents' not in all_docs:
                return
            
            documents = all_docs['documents']
            metadatas = all_docs.get('metadatas', [])
            
            for i, doc_text in enumerate(documents):
                metadata = metadatas[i] if i < len(metadatas) else {}
                
                # Extract data information from metadata or content
                data_id = metadata.get('data_id', f"data_{i}")
                
                if data_id not in self.customer_docs:
                    docs = CustomerData(
                        data_type=metadata.get('data_type', 'docs'),
                        data_id=data_id,
                        name=metadata.get('name', metadata.get('data_name', '')),
                        category=metadata.get('category', metadata.get('data_category', '')),
                        expert_advice='',
                        introduction='',
                        conclusion='',
                        full_context=doc_text
                    )
                    self.customer_docs[data_id] = docs
            
            logger.info(f"Loaded {len(self.customer_docs)} customer data items from vector store for customer {self.customer_id}")
        
        except Exception as e:
            logger.warning(f"Error parsing customer data from vector store: {e}")

    def load_new_document(self, file_path: str):
        """Load a new document into the vector store"""
        if not self.is_initialized:
            logger.warning("Cannot load document: Retriever not initialized")
            return
        
        try:
            # Get file hash for registry
            file_hash = self._get_file_hash(file_path)
            filename = os.path.basename(file_path)
            
            # Check if already loaded
            if filename in self.document_registry and self.document_registry[filename] == file_hash:
                logger.info(f"Document {filename} already loaded with same content")
                return
            else: 
                logger.info(f"Loading new document: {filename}")
            # Load content based on file type
                documents = self._load_document_by_type(file_path)
                self.vector_store.add_documents(documents)
                self.document_registry[filename] = file_hash
                self.store_load_document_registry()

            if not documents:
                logger.warning(f"No content extracted from {file_path}")
                return

            # Add to vector store
            logger.info(f"Added {len(documents)} chunks from {filename} to vector store")
            
            # Update registry
            
            # Refresh BM25 retriever if enabled
            if self.use_bm25:
                self._refresh_bm25_retriever()
            
            # Parse customer documents to update the local registry
            self._parse_customer_docs_from_vectorstore()
            
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {e}")

    def _load_document_by_type(self, file_path: str) -> List[Document]:
        """Load document content based on file type"""
        filename = os.path.basename(file_path)
        file_ext = os.path.splitext(filename)[1].lower()
        
        try:
            if file_ext == '.txt':
                return self._load_txt_document(file_path)
            elif file_ext == '.csv':
                return self._load_csv_document(file_path)
            elif file_ext == '.json':
                return self._load_json_document(file_path)
            elif file_ext == '.pdf':
                return self._load_pdf_document(file_path)
            else:
                logger.warning(f"Unsupported file type: {file_ext}")
                return []
        except Exception as e:
            logger.error(f"Error loading {file_ext} document {file_path}: {e}")
            return []

    def _load_txt_document(self, file_path: str) -> List[Document]:
        """Load text document"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split into chunks
        chunks = self.text_splitter.split_text(content)
        
        documents = []
        for i, chunk in enumerate(chunks):
            metadata = {
                'source': file_path,
                'chunk_index': i,
                'customer_id': self.customer_id,
                'file_type': 'txt',
                'data_type': 'document',
                'data_id': f"txt_{os.path.basename(file_path)}_{i}",
                'data_name': os.path.basename(file_path)
            }
            documents.append(Document(page_content=chunk, metadata=metadata))
        
        return documents

    def _load_csv_document(self, file_path: str) -> List[Document]:
        """Load CSV document"""
        try:
            df = pd.read_csv(file_path)
            documents = []
            
            for index, row in df.iterrows():
                # Convert row to text
                row_text = ' | '.join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                
                metadata = {
                    'source': file_path,
                    'row_index': index,
                    'customer_id': self.customer_id,
                    'file_type': 'csv',
                    'data_type': 'tabular',
                    'data_id': f"csv_{os.path.basename(file_path)}_{index}",
                    'data_name': f"Row {index + 1}"
                }
                
                documents.append(Document(page_content=row_text, metadata=metadata))
            
            return documents
        except Exception as e:
            logger.error(f"Error loading CSV {file_path}: {e}")
            return []

    def _load_json_document(self, file_path: str) -> List[Document]:
        """Load JSON document"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert JSON to text
            if isinstance(data, dict):
                content = json.dumps(data, indent=2, ensure_ascii=False)
            elif isinstance(data, list):
                content = '\n'.join([json.dumps(item, ensure_ascii=False) for item in data])
            else:
                content = str(data)
            
            # Split into chunks
            chunks = self.text_splitter.split_text(content)
            
            documents = []
            for i, chunk in enumerate(chunks):
                metadata = {
                    'source': file_path,
                    'chunk_index': i,
                    'customer_id': self.customer_id,
                    'file_type': 'json',
                    'data_type': 'document',
                    'data_id': f"json_{os.path.basename(file_path)}_{i}",
                    'data_name': f"JSON Chunk {i+1}"
                }
                documents.append(Document(page_content=chunk, metadata=metadata))
            
            return documents
        except Exception as e:
            logger.error(f"Error loading JSON {file_path}: {e}")
            return []

    def _load_pdf_document(self, file_path: str) -> List[Document]:
        """Load PDF document"""
        try:
            reader = PdfReader(file_path)
            content = ""
            for page in reader.pages:
                content += page.extract_text() + "\n"
            
            # Split into chunks
            chunks = self.text_splitter.split_text(content)
            
            documents = []
            for i, chunk in enumerate(chunks):
                metadata = {
                    'source': file_path,
                    'chunk_index': i,
                    'customer_id': self.customer_id,
                    'file_type': 'pdf',
                    'data_type': 'document',
                    'data_id': f"pdf_{os.path.basename(file_path)}_{i}",
                    'data_name': f"PDF Page Chunk {i+1}"
                }
                documents.append(Document(page_content=chunk, metadata=metadata))
            
            return documents
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {e}")
            return []

    def _get_file_hash(self, file_path: str) -> str:
        """Get file hash for change detection"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            logger.error(f"Error getting file hash for {file_path}: {e}")
            return ""

    def _refresh_bm25_retriever(self):
        """Refresh BM25 retriever with updated documents"""
        if not self.use_bm25:
            return
        
        try:
            all_docs = self.vector_store.get()
            if all_docs and 'documents' in all_docs and all_docs['documents']:
                documents = [
                    Document(
                        page_content=doc_text,
                        metadata=all_docs.get('metadatas', [{}])[i] if i < len(all_docs.get('metadatas', [])) else {}
                    )
                    for i, doc_text in enumerate(all_docs['documents'])
                ]
                
                self.bm25_retriever = BM25Retriever.from_documents(documents)
                logger.info(f"BM25 retriever refreshed with {len(documents)} documents")
            else:
                self.bm25_retriever = None
        except Exception as e:
            logger.warning(f"Error refreshing BM25 retriever: {e}")

    def _start_document_watcher(self):
        """Start file system watcher for automatic document loading"""
        if not os.path.exists(self.watch_directory):
            logger.warning(f"Watch directory does not exist: {self.watch_directory}")
            return
        
        try:
            self.observer = Observer()
            event_handler = CustomerDocumentWatcher(self)
            self.observer.schedule(event_handler, self.watch_directory, recursive=False)
            self.observer.start()
            logger.info(f"Started watching directory: {self.watch_directory}")
        except Exception as e:
            logger.error(f"Error starting document watcher: {e}")
            self.observer = None

    def store_load_document_registry(self):
        """Store the document registry to a file"""
        if not self.document_registry:
            logger.info("No documents to store in registry")
            return
        
        try:
            if os.path.exists(f"{self.watch_directory}/registry") is False:
                os.makedirs(f"{self.watch_directory}/registry", exist_ok=True)
            registry_path = os.path.join(f"{self.watch_directory}/registry", f"{self.collection_name}.json")
            with open(registry_path, 'w', encoding='utf-8') as f:
                json.dump(self.document_registry, f, indent=2)
            logger.info(f"Document registry stored at {registry_path}")
        except Exception as e:
            logger.error(f"Error storing document registry: {e}")

    def load_document_registry(self):
        """Load the document registry from a file"""
        try:
            if os.path.exists(f"{self.watch_directory}/registry") is False:
                os.makedirs(f"{self.watch_directory}/registry", exist_ok=True)
            registry_path = os.path.join(f"{self.watch_directory}/registry", f"{self.collection_name}.json")
            if os.path.exists(registry_path):
                with open(registry_path, 'r', encoding='utf-8') as f:
                    self.document_registry = json.load(f)
                logger.info(f"Document registry loaded from {registry_path}")
            else:
                logger.warning(f"Document registry file not found: {registry_path}")
        except Exception as e:
            logger.error(f"Error loading document registry: {e}")
            
    def check_file_in_registry(self, file_path: str) -> bool:
        """Check if a file is already in the document registry"""
        filename = os.path.basename(file_path)
        return filename in self.document_registry and self.document_registry[filename] == self._get_file_hash(file_path)

    def reload_document(self, file_path: str):
        """Reload a modified document"""
        filename = os.path.basename(file_path)
        logger.info(f"Reloading modified document: {filename}")
        
        # Remove from registry to force reload
        if filename in self.document_registry:
            del self.document_registry[filename]
        
        # Load the updated document
        self.load_new_document(file_path)

    
    def get_document_hash(self, file_path: str) -> str:
        """Calculate hash of document for change detection"""
        try:
            with open(file_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            return file_hash
        except Exception as e:
            logger.error(f"Error calculating hash for {file_path}: {e}")
            return ""
    def scan_for_new_documents(self):
        """Manually scan for new customer documents"""
        if not os.path.exists(self.watch_directory):
            logger.warning(f"Customer storage path does not exist: {self.watch_directory}")
            return
        
        new_files = []

        for root, dirs, files in os.walk(self.watch_directory):
            for file in files:
                file_path = os.path.join(root, file)
                logger.debug(f"Scanning file: {file}")
                current_hash = self.get_document_hash(file_path)
                stored_hash = self.document_registry.get(file_path)
                # Check if file belongs to this customer
                if (f"customer_{self.customer_id}" in file or
                    file.startswith(f"{self.customer_id}_")):
                    logger.debug(f"File belongs to customer {self.customer_id}: {file}")
                    # Check if it's a supported format
                    if any(file.endswith(ext) for ext in self.supported_formats) and "registry" not in file_path:

                        # Check if not already loaded
                        if current_hash != stored_hash or file_path not in self.document_registry:
                            new_files.append(file_path)
                            self.document_registry[file_path] = current_hash
            
        # Load new files
        self.store_load_document_registry()
        
        for file_path in new_files:
            try:
                self.load_new_document(file_path)
                logger.info(f"Loaded new customer file: {os.path.basename(file_path)}")
            except Exception as e:
                logger.error(f"Failed to load new file {file_path}: {e}")
        
        self.last_scan_time = time.time()
        return len(new_files)

    def get_load_status(self) -> Dict[str, Any]:
        """Get detailed load status information"""
        try:
            # Get vector store stats
            all_docs = self.vector_store.get() if self.vector_store else None
            doc_count = len(all_docs['documents']) if all_docs and 'documents' in all_docs else 0
            
            # Get file stats
            total_files = 0
            loaded_files = len(self.document_registry)
            
            if os.path.exists(self.watch_directory):
                for filename in os.listdir(self.watch_directory):
                    if (f"customer_{self.customer_id}" in filename or
                        filename.startswith(f"{self.customer_id}_")):
                        if any(filename.endswith(ext) for ext in self.supported_formats):
                            total_files += 1
            
            status = {
                'customer_id': self.customer_id,
                'is_initialized': self.is_initialized,
                'collection_name': self.collection_name,
                'storage_path': self.customers_storage_path,
                'watch_directory': self.watch_directory,
                'total_documents': doc_count,
                'total_contents': len(self.customer_docs),
                'files_in_registry': loaded_files,
                'total_customer_files': total_files,
                'auto_scan_enabled': self.auto_scan_enabled,
                'supported_formats': self.supported_formats,
                'last_scan_time': datetime.fromtimestamp(self.last_scan_time).isoformat(),
                'bm25_enabled': self.bm25_retriever is not None,
                'watcher_active': self.observer is not None and self.observer.is_alive() if self.observer else False
            }
            
            return status
        except Exception as e:
            logger.error(f"Error getting load status: {e}")
            return {'error': str(e), 'customer_id': self.customer_id}

    def get_customer_files(self, file_filter: str = "") -> Dict[str, List[str]]:
        """Get list of customer files, optionally filtered"""
        files_by_type = {}
        
        if not os.path.exists(self.customers_storage_path):
            return files_by_type
        
        for filename in os.listdir(self.customers_storage_path):
            if (f"customer_{self.customer_id}" in filename or 
                filename.startswith(f"{self.customer_id}_")):
                
                # Apply filter if provided
                if file_filter and file_filter.lower() not in filename.lower():
                    continue
                
                file_ext = os.path.splitext(filename)[1].lower()
                if file_ext in self.supported_formats:
                    if file_ext not in files_by_type:
                        files_by_type[file_ext] = []
                    files_by_type[file_ext].append(filename)
        
        return files_by_type

    def _run(self, query: str, k: int = 5, search_type: str = "hybrid") -> str:
        """Execute the customer retrieval with dynamic reranking"""
        if not self.is_initialized:
            return f"❌ Customer retriever not initialized for customer {self.customer_id}"
        
        if not self.vector_store:
            return f"❌ Vector store not available for customer {self.customer_id}"
        
        try:
            # Get retrieved documents using hybrid approach
            retrieved_docs = []
            
            # Vector similarity search
            try:
                vector_docs = self.vector_store.similarity_search_with_score(query, k=min(k*2, 10))
                for doc, score in vector_docs:
                    retrieved_doc = RetrievedCustomerDocument(
                        content=doc.page_content,
                        source=doc.metadata.get('source', 'unknown'),
                        retrieval_score=float(score),
                        data_type=doc.metadata.get('data_type', 'customer_docs'),
                        data_id=doc.metadata.get('data_id', ''),
                        customer_id=self.customer_id
                    )
                    retrieved_docs.append(retrieved_doc)
            except Exception as e:
                logger.warning(f"Vector search failed: {e}")
            
            # BM25 search if available
            if self.bm25_retriever and search_type in ["hybrid", "bm25"]:
                try:
                    bm25_docs = self.bm25_retriever.get_relevant_documents(query)[:k]
                    for doc in bm25_docs:
                        retrieved_doc = RetrievedCustomerDocument(
                            content=doc.page_content,
                            source=doc.metadata.get('source', 'unknown'),
                            retrieval_score=0.8,  # Default BM25 score
                            data_type=doc.metadata.get('data_type', 'customer_docs'),  
                            data_id=doc.metadata.get('data_id', ''),
                            customer_id=self.customer_id
                        )
                        # Avoid duplicates
                        if not any(rd.content == retrieved_doc.content for rd in retrieved_docs):
                            retrieved_docs.append(retrieved_doc)
                except Exception as e:
                    logger.warning(f"BM25 search failed: {e}")
            
            if not retrieved_docs:
                return f"ℹ️ No relevant medical information found for customer {self.customer_id} with query: '{query}'"
            
            # Calculate customer-specific relevance scores
            for doc in retrieved_docs:
                doc.calculate_customer_relevance(query, self.customer_id)
            
            # Sort by relevance score (descending)
            retrieved_docs.sort(key=lambda x: x.retrieval_score, reverse=True)
            
            # Take top k results
            top_docs = retrieved_docs[:k]
            
            # Format response with customer context
            response_parts = [f"🏥 **Customer {self.customer_id} Medical Information**\n"]
            response_parts.append(f"📋 **Query:** {query}\n")
            
            for i, doc in enumerate(top_docs, 1):
                response_parts.append(f"**Result {i}** (Relevance: {doc.retrieval_score:.3f})")
                if doc.data_name:
                    response_parts.append(f"🔍 **Data:** {doc.data_name}")
                if doc.data_category:
                    response_parts.append(f"📂 **Category:** {doc.data_category}")
                response_parts.append(f"📄 **Source:** {os.path.basename(doc.source)}")
                response_parts.append(f"📝 **Content:** {doc.content[:800]}...")
                response_parts.append("")
            
            # Add summary statistics
            response_parts.append(f"📊 **Summary:** Found {len(top_docs)} relevant results from {len(self.customer_docs)} customer documents")
            
            return "\n".join(response_parts)
            
        except Exception as e:
            logger.error(f"Error during customer retrieval: {e}")
            return f"❌ Error retrieving medical information for customer {self.customer_id}: {str(e)}"

    async def _arun(self, query: str, k: int = 5, search_type: str = "hybrid") -> str:
        """Asynchronously execute the customer retrieval with dynamic reranking"""
        # This is a simple async wrapper. For a true async implementation,
        # the underlying I/O operations (vector store, file access) should be async.
        return self._run(query, k, search_type)
    
    def get_customer_medical_profile(self) -> Dict[str, Any]:
        """Get a summary of the customer's medical profile"""
        if not self.is_initialized:
            return {"error": "Retriever not initialized"}

        profile = {
            "customer_id": self.customer_id,
            "total_documents": len(self.document_registry),
            "total_data_items": len(self.customer_docs),
            "data_summary": []
        }

        for data_id, data in self.customer_docs.items():
            profile["data_summary"].append({
                "data_id": data.data_id,
                "name": data.name,
                "type": data.data_type,
                "category": data.category
            })
        return profile

    def search_customer_conditions(self, condition_query: str) -> List[Dict[str, Any]]:
        """Search for specific conditions within the customer's data"""
        if not self.is_initialized:
            return [{"error": "Retriever not initialized"}]

        results = []
        for doc in self.customer_docs.values():
            if condition_query.lower() in doc.full_context.lower():
                results.append({
                    "data_id": doc.data_id,
                    "name": doc.name,
                    "match_context": doc.full_context[:200] + "..."
                })
        return results
    
    def cleanup(self):
        """Clean up resources"""
        try:
            # Stop file watcher
            if self.observer and self.observer.is_alive():
                self.observer.stop()
                self.observer.join(timeout=5)
                logger.info("File watcher stopped")
            
            # Clear references
            self.vector_store = None
            self.bm25_retriever = None
            self.embeddings = None
            self.customer_docs.clear()
            self.document_registry.clear()
            
            logger.info(f"Customer retriever cleanup completed for customer {self.customer_id}")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


# if __name__ == "__main__":
#     # Example usage
#     retriever = CustomerRetrieverTool(customer_id="12345", use_bm25=True, auto_scan_enabled=True)
#     print(retriever.get_load_status())
    
#     # Perform a sample retrieval
#     query_result = retriever._run(query="What are the symptoms of diabetes?", k=5, search_type="hybrid")
#     print(query_result)
    
#     # Cleanup resources
#     retriever.cleanup()