# --- Enhanced Product Retriever Tool ---
import os
import json
import sys
import time
import csv
from datetime import datetime
import hashlib
import pandas as pd
from pypdf import PdfReader
import threading
from typing import Optional, List, Dict, Any
from loguru import logger
from pydantic import Field, BaseModel
import numpy as np
from loguru import logger
from pydantic import Field, BaseModel
import numpy as np
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
# --- Vector Store and Embedding Imports ---
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

# --- Tool Imports ---
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
# from app.agents.tools.base import BaseAgentTool
# from app.core.config import get_settings


from agents.tools.base import BaseAgentTool
from core.config import get_settings
settings = get_settings()










class DocumentWatcher(FileSystemEventHandler):
    """File system watcher for automatic document loading"""
    
    def __init__(self, retriever):
        self.retriever = retriever
        self.supported_extensions = {'.txt', '.pdf', '.csv', '.json'}
        
    def on_created(self, event):
        if not event.is_directory:
            file_path = event.src_path
            if any(file_path.endswith(ext) for ext in self.supported_extensions):
                logger.info(f"New document detected: {file_path}")
                self.retriever.load_new_document(file_path)
                
    def on_modified(self, event):
        if not event.is_directory:
            file_path = event.src_path
            if any(file_path.endswith(ext) for ext in self.supported_extensions) and "registry" not in file_path:
                logger.info(f"Document modified: {file_path}")
                self.retriever.reload_document(file_path)


class RetrievedProductDocument(BaseModel):
    """Model for a retrieved product document with enhanced metadata for reranking"""
    content: str = Field(description="The full text content of the retrieved document")
    source: str = Field(description="Source identifier of the document")
    retrieval_score: float = Field(description="Raw similarity score from the retrieval system")
    relevance_score: float = Field(default=0.0, description="Calculated relevance score after reranking")
    product_name: str = Field(default="", description="Name of the product if identified")
    feature_type: str = Field(default="", description="Type of product feature or aspect")
    price_info: str = Field(default="", description="Price information if available")
    category: str = Field(default="", description="Product category classification")


    
    def calculate_advanced_relevance(self, query: str) -> float:
        """Enhanced relevance scoring with multiple factors for product content"""
        query_lower = query.lower()
        content_lower = self.content.lower()
        
        # 1. Exact phrase matching (highest weight)
        exact_match_score = 0.0
        if query_lower in content_lower:
            exact_match_score = 1.0
        
        # 2. Product name relevance matching
        product_match_score = 0.0
        query_terms = set(query_lower.split())
        if self.product_name and any(term in self.product_name.lower() for term in query_terms):
            product_match_score = 0.9
        
        # 3. Feature and price relevance
        feature_match_score = 0.0
        price_keywords = ['giá', 'price', 'cost', 'tiền', 'vnđ', 'đồng', 'phí', 'fee']
        if any(keyword in query_lower for keyword in price_keywords) and self.price_info:
            feature_match_score = 0.8
        
        # 4. Product feature matching
        product_keywords = ['genemap', 'adult', 'kid', 'trẻ em', 'người lớn', 'test', 'xét nghiệm']
        if any(keyword in query_lower for keyword in product_keywords):
            if any(keyword in content_lower for keyword in product_keywords):
                feature_match_score = max(feature_match_score, 0.7)
        
        # 5. Content semantic similarity (term overlap)
        query_words = set(query_lower.split())
        content_words = set(content_lower.split())
        if query_words and content_words:
            overlap = len(query_words.intersection(content_words))
            semantic_score = min(overlap / len(query_words), 1.0) * 0.6
        else:
            semantic_score = 0.0
        
        # 6. Length penalty for very short or very long content
        length_penalty = 1.0
        if len(self.content) < 20:
            length_penalty = 0.7
        elif len(self.content) > 1000:
            length_penalty = 0.9
        
        # Combine scores with weights
        final_score = (
            exact_match_score * 0.35 +
            product_match_score * 0.25 +
            feature_match_score * 0.20 +
            semantic_score * 0.15 +
            (1.0 - exact_match_score) * 0.05  # Small boost for variety
        ) * length_penalty
        
        return min(final_score, 1.0)

class ProductRetrieverTool(BaseAgentTool):
    """Enhanced Product Retriever with advanced reranking and Vietnamese support"""
    
    name: str = "product_retriever"
    description: str = "Advanced product information retrieval with reranking for GeneMap products and services"
    
    # Core configuration
    collection_name: str = Field(default="genestory_products")
    collection_type: str = "product"
    use_bm25: bool = Field(default=True)
    
    # Retrieval parameters
    max_results: int = Field(default=5)
    similarity_threshold: float = Field(default=0.1)
    rerank_threshold: float = Field(default=0.3)
    
    # Internal components
    vector_store: Optional[Chroma] = None
    bm25_retriever: Optional[BM25Retriever] = None
    embeddings: Optional[OllamaEmbeddings] = None
    is_initialized: bool = False


     # New fields for enhanced functionality
    watch_directory: str = Field(default="", description="Directory to watch for new documents")
    document_registry: Dict[str, str] = Field(default_factory=dict, description="Registry of loaded documents with hashes")
    observer: Any = Field(default=None, description="File system observer")
    text_splitter: Any = Field(default=None, description="Text splitter for large documents")
    auto_scan_enabled: bool = Field(default=True, description="Whether automatic scanning is enabled")
    supported_formats: List[str] = Field(default_factory=lambda: ['.txt', '.pdf', '.csv', '.json'], description="Supported document formats")
    files_loaded: List[str] = Field(default_factory=list, description="List of loaded document files")
    
    def __init__(self, collection_name: str = "product_docs", use_bm25: bool = True, collection_type: str = "product", 
                 watch_directory: str = None, auto_scan_enabled: bool = True, **kwargs):
        
        safe_collection_name = collection_name.replace(" ", "_").replace("-", "_")
        if len(safe_collection_name) > 63:
            safe_collection_name = safe_collection_name[:63]  # ChromaDB has a length limit
        
        # Set default watch directory
        if not watch_directory:
            watch_directory = os.path.join(os.path.dirname(__file__), 'data_raw')
        
        # Initialize tool with proper name and description
        name = f"product_retriever"
        description = f"Retrieve GeneStory product information with multi-format support and auto-scanning."

        super().__init__(
            name=name, 
            description=description,
            collection_name=safe_collection_name,
            watch_directory=watch_directory,
            use_bm25=use_bm25,
            collection_type=collection_type,
            auto_scan_enabled=auto_scan_enabled,
            **kwargs
        )
        self._initialize_components()

        self.load_document_registry()
        if self.auto_scan_enabled:
            logger.info(f"Auto-scanning enabled, starting document watcher for directory: {self.watch_directory}")
            self._start_document_watcher()
            # self.scan_for_new_documents()
            
            
    
    
    def _initialize_components(self) -> None:
        """Initialize ChromaDB, embeddings, and BM25 components"""
        try:
            # Initialize ChromaDB client
            persistent_client = chromadb.PersistentClient(path=settings.VECTOR_STORE_BASE_DIR)
            logger.info("ChromaDB client initialized successfully")
            
            # Initialize embedding model
            self.embeddings = OllamaEmbeddings(
                model="mxbai-embed-large",
                base_url="http://localhost:11434"
            )
            
            # Ensure collection exists
            try:
                collection = persistent_client.get_or_create_collection(self.collection_name)
                logger.info(f"Using collection: {self.collection_name}")
            except Exception as e:
                logger.error(f"Error with collection: {e}")
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
                    if all_docs['documents']:
                        documents = [Document(page_content=doc, metadata=meta) 
                                   for doc, meta in zip(all_docs['documents'], all_docs['metadatas'])]
                        self.bm25_retriever = BM25Retriever.from_documents(documents)
                        self.bm25_retriever.k = 10
                        logger.info(f"BM25 retriever initialized with {len(documents)} documents")
                    else:
                        logger.warning("No documents available for BM25 indexing")
                except Exception as e:
                    logger.warning(f"Failed to initialize BM25 retriever: {e}")
                    self.bm25_retriever = None
            
            self.is_initialized = True
            logger.info(f"Product retriever initialized successfully with collection '{self.collection_name}'")
            
            # Auto-load documents if vector store is empty
            self._auto_load_documents()
            
        except Exception as e:
            logger.error(f"Failed to initialize product retriever: {e}")
            self.is_initialized = False
            raise
    def _start_document_watcher(self):
        """Start the file system watcher for automatic document loading"""
        try:
            if not os.path.exists(self.watch_directory):
                logger.info(f"Creating watch directory: {self.watch_directory}")
                os.makedirs(self.watch_directory, exist_ok=True)
            files_in_watch_dir = os.listdir(self.watch_directory)
            if not files_in_watch_dir:
                logger.info(f"No files found in watch directory: {self.watch_directory}")   
            else:
                logger.info(f"Files found in watch directory: {files_in_watch_dir}")
                # store file names in the registry
                self.files_loaded.extend(files_in_watch_dir)
            # Initialize file system observer
            self.observer = Observer()
            event_handler = DocumentWatcher(self)
            self.observer.schedule(event_handler, self.watch_directory, recursive=True)
            self.observer.start()
            
            logger.info(f"Started document watcher for directory: {self.watch_directory}")
        except Exception as e:
            logger.error(f"Failed to start document watcher: {e}")
            self.observer = None

    def stop_document_watcher(self):
        """Stop the file system watcher"""
        try:
            if self.observer:
                self.observer.stop()
                self.observer.join()
                logger.info("Document watcher stopped")
        except Exception as e:
            logger.error(f"Error stopping document watcher: {e}")
    
    def get_document_hash(self, file_path: str) -> str:
        """Calculate hash of document for change detection"""
        try:
            with open(file_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            return file_hash
        except Exception as e:
            logger.error(f"Error calculating hash for {file_path}: {e}")
            return ""
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
    def _auto_load_documents(self) -> None:
        """Automatically scan and load documents from multiple formats"""
        # self.scan_and_load_all_documents()
        
        try:
            # Check if vector store has documents
            if self.vector_store:
                all_docs = self.vector_store.get()
                logger.info(f"Checking vector store contents for collection '{self.collection_name}'")
                if not all_docs or not all_docs.get('documents'):
                    logger.info("Vector store is empty, scanning for documents in multiple formats")
                    self.scan_and_load_all_documents()
                else:
                    logger.info(f"Vector store already has {len(all_docs['documents'])} documents")
                    # Still scan for new documents
                    if self.auto_scan_enabled:
                        self.scan_for_new_documents()
        except Exception as e:
            logger.warning(f"Could not check vector store contents: {e}")
            # Try to load anyway
            self.scan_and_load_all_documents()

    def _get_file_hash(self, file_path: str) -> str:
        """Get file hash for change detection"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            logger.error(f"Error getting file hash for {file_path}: {e}")
            return ""

    def load_pdf_document(self, pdf_path: str) -> List[Document]:
        """Load documents from PDF file"""
        documents = []
        try:
            reader = PdfReader(pdf_path)
            full_text = ""
            
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                full_text += f"\n\nPage {page_num + 1}:\n{text}"
            
            if full_text.strip():
                # Split large PDFs into chunks
                chunks = self.text_splitter.split_text(full_text)
                
                for i, chunk in enumerate(chunks):
                    if chunk.strip():
                        # Parse for company information
                        parsed = self.parse_product_document(chunk)
                        
                        doc = Document(
                            page_content=chunk.strip(),
                            metadata={
                                'source': os.path.basename(pdf_path),
                                'file_type': 'pdf',
                                'chunk_id': i,
                                'topic': parsed.get('topic', ''),
                                'category': parsed.get('category', ''),
                                'priority_level': parsed.get('priority_level', ''),
                                'loaded_at': datetime.now().isoformat()
                            }
                        )
                        documents.append(doc)
            
            logger.info(f"Loaded {len(documents)} chunks from PDF: {pdf_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading PDF {pdf_path}: {e}")
            return []
    
    def load_csv_document(self, csv_path: str) -> List[Document]:
        """Load documents from CSV file"""
        # """An other way to load CSV files using CSVLoader"""
        # if not os.path.exists(csv_path):
        #     logger.error(f"CSV file not found: {csv_path}")
        #     return []
        # try:
        #     loader = CSVLoader(file_path=csv_path)
        #     documents = loader.load()
        #     logger.info(f"Loaded {len(documents)} documents from CSV: {csv_path}")
        #     return documents

        # except Exception as e:
        #     logger.error(f"Error loading CSV {csv_path}: {e}")
        #     return []

        documents = []
        try:
            df = pd.read_csv(csv_path)
            
            for index, row in df.iterrows():
                # Convert row to text format
                content_parts = []
                for col, value in row.items():
                    if pd.notna(value):
                        content_parts.append(f"{col}: {value}")
                
                content = "\n".join(content_parts)
                
                if content.strip():
                    # Parse for company information
                    parsed = self.parse_product_document(content)
                    
                    doc = Document(
                        page_content=content,
                        metadata={
                            'source': os.path.basename(csv_path),
                            'file_type': 'csv',
                            'row_id': index,
                            'topic': parsed.get('topic', ''),
                            'category': parsed.get('category', ''),
                            'priority_level': parsed.get('priority_level', ''),
                            'loaded_at': datetime.now().isoformat()
                        }
                    )
                    documents.append(doc)
            
            logger.info(f"Loaded {len(documents)} rows from CSV: {csv_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading CSV {csv_path}: {e}")
            return []
    
    def load_json_document(self, json_path: str) -> List[Document]:
        """Load documents from JSON file"""
        documents = []
        import json 
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            def extract_text_from_json(obj, prefix=""):
                """Recursively extract text from JSON object"""
                texts = []
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        new_prefix = f"{prefix}.{key}" if prefix else key
                        texts.extend(extract_text_from_json(value, new_prefix))
                elif isinstance(obj, list):
                    for i, item in enumerate(obj):
                        new_prefix = f"{prefix}[{i}]"
                        texts.extend(extract_text_from_json(item, new_prefix))
                else:
                    texts.append(f"{prefix}: {obj}")
                return texts
            
            if isinstance(data, list):
                # Handle JSON array
                for i, item in enumerate(data):
                    texts = extract_text_from_json(item)
                    content = "\n".join(texts)
                    
                    if content.strip():
                        parsed = self.parse_product_document(content)
                        
                        doc = Document(
                            page_content=content,
                            metadata={
                                'source': os.path.basename(json_path),
                                'file_type': 'json',
                                'item_id': i,
                                'topic': parsed.get('topic', ''),
                                'category': parsed.get('category', ''),
                                'priority_level': parsed.get('priority_level', ''),
                                'loaded_at': datetime.now().isoformat()
                            }
                        )
                        documents.append(doc)
            else:
                # Handle single JSON object
                texts = extract_text_from_json(data)
                content = "\n".join(texts)
                
                if content.strip():
                    parsed = self.parse_product_document(content)
                    
                    doc = Document(
                        page_content=content,
                        metadata={
                            'source': os.path.basename(json_path),
                            'file_type': 'json',
                            'topic': parsed.get('topic', ''),
                            'category': parsed.get('category', ''),
                            'priority_level': parsed.get('priority_level', ''),
                            'loaded_at': datetime.now().isoformat()
                        }
                    )
                    documents.append(doc)
            
            logger.info(f"Loaded {len(documents)} items from JSON: {json_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading JSON {json_path}: {e}")
            return []
    
    def load_document_by_format(self, file_path: str) -> List[Document]:
        """Load document based on file format"""
        _, ext = os.path.splitext(file_path.lower())
        
        if ext == '.txt':
            return self.load_documents_from_txt(file_path)
        elif ext == '.pdf':
            return self.load_pdf_document(file_path)
        elif ext == '.csv':
            return self.load_csv_document(file_path)
        elif ext == '.json':
            return self.load_json_document(file_path)
        else:
            logger.warning(f"Unsupported file format: {ext}")
            return []
    
    def scan_and_load_all_documents(self):
        """Scan directory and load all supported documents"""
        if not os.path.exists(self.watch_directory):
            logger.warning(f"Watch directory does not exist: {self.watch_directory}")
            return
        
        total_loaded = 0
        for root, dirs, files in os.walk(self.watch_directory):
            logger.info(f"Scanning directory: {root}")
            logger.info(f"Found {len(files)} files in {root}")
            for file in files:
                file_path = os.path.join(root, file)
                _, ext = os.path.splitext(file.lower())
                
                if ext in self.supported_formats:
                    documents = self.load_document_by_format(file_path)
                    if documents:
                        self.add_documents(documents)
                        total_loaded += len(documents)
                        
                        # Update document registry
                        file_hash = self.get_document_hash(file_path)
                        self.document_registry[file_path] = file_hash
        
        logger.info(f"Scanned and loaded {total_loaded} documents from {self.watch_directory}")
    
    def scan_for_new_documents(self):
        """Scan for new or modified documents"""
        if not os.path.exists(self.watch_directory):
            return
        logger.info("looking for new or modified documents in the watch directory: {}".format(self.watch_directory))
        # check document registry
        logger.info(f"Current document registry: {self.document_registry}")
        new_docs_count = 0
        for root, dirs, files in os.walk(self.watch_directory):
            for file in files:
                file_path = os.path.join(root, file)
                _, ext = os.path.splitext(file.lower())
                
                if ext in self.supported_formats:
                    current_hash = self.get_document_hash(file_path)
                    stored_hash = self.document_registry.get(file_path)
                    
                    
                    if file_path not in self.document_registry:
                        logger.debug(f"Processing file: {file_path}, current hash: {current_hash}, stored hash: {stored_hash}")
                        # New or modified document
                        documents = self.load_document_by_format(file_path)
                        if documents:
                            self.add_documents(documents)
                            new_docs_count += len(documents)
                            self.document_registry[file_path] = current_hash
                            logger.info(f"Loaded new/modified document: {file_path}")
                
        self.store_load_document_registry()
        logger.info(f"Document registry updated with {len(self.document_registry)} entries")
        if new_docs_count > 0:
            logger.info(f"Found and loaded {new_docs_count} new/modified documents")
            

    def load_new_document(self, file_path: str):
        """Load a single new document (called by file watcher)"""
        file_hash = self._get_file_hash(file_path)
        filename = os.path.join(self.watch_directory, os.path.basename(file_path))
        logger.info(f"Loading new document: {filename} with hash {file_hash}")
        logger.info(f"Current document registry: {self.document_registry.keys()}")
        try:
            # Get file hash for registry
            file_hash = self.get_document_hash(file_path)
            filename = os.path.basename(file_path)
            
            # Check if already loaded
            if filename in self.document_registry and self.document_registry[filename] == file_hash:
                logger.info(f"Document {filename} already loaded with same content")
                return
            else: 
                logger.info(f"Loading new document: {filename}")
            # Load content based on file type
                documents = self.load_document_by_format(file_path)
                self.vector_store.add_documents(documents)
                self.document_registry[filename] = file_hash
                self.store_load_document_registry()
                
            # Load content based on file type
                
        except Exception as e:
            logger.error(f"Error auto-loading document {file_path}: {e}")
    
    def reload_document(self, file_path: str):
        """Reload a modified document (called by file watcher)"""
        try:
            # For now, we'll just load it as new
            # In a more sophisticated implementation, we might remove old versions first
            self.load_new_document(file_path)
        except Exception as e:
            logger.error(f"Error reloading document {file_path}: {e}")
    
    def parse_product_document(self, content: str) -> Dict[str, Any]:
        """Parse product document content and extract structured information"""
        lines = content.strip().split('\n')
        
        # Extract metadata
        product_name = ""
        feature_type = ""
        price_info = ""
        category = "General"
        
        # Look for product names
        for line in lines[:5]:  # Check first few lines
            if "GeneMap" in line:
                product_name = line.strip('#').strip()
                break
        
        # Look for feature information
        if "chỉ số" in content.lower() or "indicator" in content.lower():
            feature_type = "Features"
        elif "giá" in content.lower() or "vnđ" in content.lower():
            feature_type = "Pricing"
        elif "dịch vụ" in content.lower() or "service" in content.lower():
            feature_type = "Services"
        
        # Extract price information
        for line in lines:
            if "vnđ" in line.lower() or "giá" in line.lower():
                price_info = line.strip()
                break
        
        # Determine category
        if "adult" in content.lower() or "người lớn" in content.lower():
            category = "Adult Products"
        elif "kid" in content.lower() or "trẻ em" in content.lower():
            category = "Kid Products"
        elif "tư vấn" in content.lower() or "consult" in content.lower():
            category = "Consultation Services"
        elif "genemark" in content.lower():
            category = "GeneMark Services"
        
        return {
            'product_name': product_name,
            'feature_type': feature_type,
            'price_info': price_info,
            'category': category,
            'content_length': len(content),
            'source': 'product_docs.txt'
        }

    def load_documents_from_txt(self, txt_file_path: str) -> List[Document]:
        """Load documents from the company_docs.txt file"""
        documents = []
        
        try:
            with open(txt_file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Split by double newlines to separate company information entries
            entries = content.strip().split('\n\n')
            
            for i, entry in enumerate(entries):
                if entry.strip():
                    # Parse the entry to extract topic for metadata
                    parsed = self.parse_product_document(entry)
                    
                    doc = Document(
                        page_content=entry.strip(),
                        metadata={
                            'source': 'product_docs.txt',
                            'topic': parsed.get('topic', ''),
                            'category': parsed.get('category', ''),
                            'priority_level': parsed.get('priority_level', ''),
                            'entry_id': i
                        }
                    )
                    documents.append(doc)
            
            logger.info(f"Loaded {len(documents)} documents from {txt_file_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading documents from {txt_file_path}: {e}")
            return []


    def _run(self, query: str) -> List[str]:
        """Synchronous document retrieval for LangChain compatibility."""
        return self.retrieve_documents(query)
    
    async def _arun(self, query: str) -> List[str]:
        """Asynchronous document retrieval for LangChain compatibility."""
        # For simplicity, we're using the synchronous version
        return self.retrieve_documents(query)
    
    def run(self, query: str, state: Optional[Dict[str, Any]] = None) -> List[str]:
        """Main run method for the retrieval tool."""
        return self.retrieve_documents(query)
    
    async def arun(self, query: str, state: Optional[Dict[str, Any]] = None) -> List[str]:
        """Async run method (using sync implementation for simplicity)."""
        return self.retrieve_documents(query)

    def add_documents(self, documents: List[Document]) -> bool:
        """Add documents to the vector store"""
        if not documents:
            logger.warning("No documents to add")
            return False
        
        try:
            # Add to vector store
            self.vector_store.add_documents(documents)
            logger.info(f"Added {len(documents)} documents to product collection")
            
            # Reinitialize BM25 retriever with new documents
            if self.use_bm25:
                try:
                    all_docs = self.vector_store.get()
                    if all_docs['documents']:
                        documents_for_bm25 = [Document(page_content=doc, metadata=meta) 
                                            for doc, meta in zip(all_docs['documents'], all_docs['metadatas'])]
                        self.bm25_retriever = BM25Retriever.from_documents(documents_for_bm25)
                        self.bm25_retriever.k = 10
                        logger.info(f"Updated BM25 retriever with {len(documents_for_bm25)} documents")
                except Exception as e:
                    logger.warning(f"Failed to update BM25 retriever: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return False

    def initialize_from_txt(self) -> bool:
        """Initialize the retriever by loading documents from txt file"""
        logger.info("Initializing product retriever from txt file...")
        
        # Load documents
        documents = self.load_documents_from_txt()
        if not documents:
            logger.error("No documents loaded from txt file")
            return False
        
        # Add documents to vector store
        success = self.add_documents(documents)
        if success:
            logger.info(f"Successfully initialized product retriever with {len(documents)} documents")
        
        return success

    def get_relevant_documents_with_reranking(self, query: str, k: int = 8) -> List[RetrievedProductDocument]:
        """Get documents using hybrid search with advanced reranking"""
        all_docs = []
        initial_k = max(k * 2, 15)  # Get more docs initially for better reranking
        
        try:
            # 1. Vector similarity search
            vector_docs = self.vector_store.similarity_search_with_score((query, initial_k))
            logger.info(f"Vector search returned {len(vector_docs)} documents for query: '{query}'")
            logger.debug(f"Vector search results: {vector_docs[0]}")
            for doc, score in vector_docs:
                 
                all_docs.append((doc, score, "vector"))
                logger.info(f"Vector search found document: {doc.metadata.get('source', 'unknown')} with score {score:.2f}")
            
            # 2. BM25 search if available
            if self.bm25_retriever:
                bm25_docs = self.bm25_retriever.get_relevant_documents(query)[:initial_k]
                
                for doc in bm25_docs:
                    all_docs.append((doc, 0.7, "bm25"))  # Default BM25 score
            
            # 3. Convert to RetrievedProductDocument objects
            retrieved_docs = []
            for doc, score, method in all_docs:
                retrieved_doc = RetrievedProductDocument(
                    content=doc.page_content,
                    source=doc.metadata.get('source', 'unknown'),
                    retrieval_score=float(score),
                    product_name=doc.metadata.get('product_name', ''),
                    feature_type=doc.metadata.get('feature_type', ''),
                    price_info=doc.metadata.get('price_info', ''),
                    category=doc.metadata.get('category', 'General')
                )
                retrieved_docs.append(retrieved_doc)
            
            # 4. Advanced reranking
            reranked_docs = self.rerank_documents(retrieved_docs, query)
            
            return reranked_docs[:k]
            
        except Exception as e:
            logger.error(f"Error in document retrieval: {e}")
            return []

    def rerank_documents(self, docs: List[RetrievedProductDocument], query: str) -> List[RetrievedProductDocument]:
        """Advanced reranking with product-specific scoring"""
        if not docs:
            return []
        
        # Calculate relevance scores
        for doc in docs:
            doc.relevance_score = doc.calculate_advanced_relevance(query)
        
        # Sort by relevance score
        docs.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Filter by threshold
        filtered_docs = [doc for doc in docs if doc.relevance_score >= self.rerank_threshold]
        
        # If too few docs pass threshold, include top results anyway
        if len(filtered_docs) < 3 and docs:
            filtered_docs = docs[:max(3, len(filtered_docs))]
        
        # Diversity filtering - avoid too many docs from same category
        final_docs = []
        category_count = {}
        
        for doc in filtered_docs:
            category = doc.category
            if category_count.get(category, 0) < 2:  # Max 2 per category
                final_docs.append(doc)
                category_count[category] = category_count.get(category, 0) + 1
        
        logger.info(f"Reranked {len(docs)} documents to {len(final_docs)} high-quality results")
        return final_docs

    def retrieve_documents(self, query: str) -> List[str]:
        """Enhanced document retrieval with advanced reranking for better product context."""
        if not hasattr(self, 'vector_store') or self.vector_store is None:
            logger.warning("Vector store not available. Cannot retrieve documents.")
            return [f"Error: {self.collection_type.capitalize()} document retrieval is unavailable."]
        
        try:
            time_start = time.time()
            
            # Use enhanced retrieval with reranking
            reranked_docs = self.get_relevant_documents_with_reranking(query, k=8)
            
            if not reranked_docs:
                logger.warning(f"No relevant {self.collection_type} documents found for query: '{query}'")
                return [f"Không tìm thấy thông tin {self.collection_type} phù hợp cho câu hỏi: '{query}'"]
            
            # Format results with enhanced metadata
            formatted_results = []
            for i, doc in enumerate(reranked_docs, 1):
                # Create formatted result with metadata
                result_prefix = f"📦 PRODUCT {doc.category}"
                if doc.product_name:
                    result_prefix += f" [{doc.product_name}]"
                if doc.feature_type:
                    result_prefix += f" ({doc.feature_type})"
                
                formatted_content = f"{result_prefix}: {doc.content}"
                logger.info(f"Retrieved {self.collection_type} document {i}: {doc.product_name} - {doc.category} - Score: {doc.relevance_score:.2f} content length: {len(doc.content)}")
                formatted_results.append(formatted_content)
            
            # Log retrieval performance
            time_elapsed = time.time() - time_start
            logger.info(f"Enhanced {self.collection_type} document retrieval completed in {time_elapsed:.2f} seconds with {len(formatted_results)} reranked results.")
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error during {self.collection_type} document retrieval: {e}")
            return [f"Lỗi khi tìm kiếm thông tin {self.collection_type}: {str(e)}"]

    async def aretrieve_documents(self, query: str) -> List[str]:
        """Asynchronous document retrieval for LangChain compatibility."""
        return self.retrieve_documents(query)

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the product collection"""
        try:
            if not hasattr(self.vector_store, '_collection'):
                return {"error": "Vector store not available"}
            
            all_docs = self.vector_store.get()
            if not all_docs['metadatas']:
                return {"total_documents": 0}
            
            # Basic stats
            total_docs = len(all_docs['metadatas'])
            
            # Extract categories and products
            categories = set()
            products = set()
            feature_types = set()
            
            for metadata in all_docs['metadatas']:
                if metadata.get('category'):
                    categories.add(metadata['category'])
                if metadata.get('product_name'):
                    products.add(metadata['product_name'])
                if metadata.get('feature_type'):
                    feature_types.add(metadata['feature_type'])
            
            return {
                "total_documents": total_docs,
                "unique_products": len(products),
                "unique_categories": len(categories),
                "unique_feature_types": len(feature_types),
                "categories": sorted(list(categories)),
                "products": sorted(list(products)),
                "feature_types": sorted(list(feature_types))
            }
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {"error": str(e)}

    def search_by_category(self, category: str, limit: int = 5) -> List[str]:
        """Search documents by category"""
        try:
            all_docs = self.vector_store.get()
            if not all_docs['documents']:
                return []
            
            filtered_results = []
            for doc, meta in zip(all_docs['documents'], all_docs['metadatas']):
                if meta.get('category', '').lower() == category.lower():
                    product_name = meta.get('product_name', '')
                    prefix = f"📦 PRODUCT {category}"
                    if product_name:
                        prefix += f" [{product_name}]"
                    formatted_content = f"{prefix}: {doc}"
                    filtered_results.append(formatted_content)
                    
                    if len(filtered_results) >= limit:
                        break
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error in category search: {e}")
            return []

    async def _arun(self, query: str) -> str:
        """Async implementation for LangChain compatibility"""
        results = self.retrieve_documents(query)
        return "\n\n".join(results)

    def _run(self, query: str) -> str:
        """Sync implementation for LangChain compatibility"""
        results = self.retrieve_documents(query)
        return "\n\n".join(results)
