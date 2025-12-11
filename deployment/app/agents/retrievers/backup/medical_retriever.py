# --- Enhanced Medical Retriever Tool ---
import os
import sys
import time
import json
import hashlib
from datetime import datetime
from typing import Optional, List, Dict, Any
from loguru import logger
from pydantic import Field, BaseModel
import numpy as np

# --- File Watching Imports ---
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# --- Document Loading Imports ---
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logger.warning("pandas not available - CSV support will be limited")

try:
    from pypdf import PdfReader
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logger.warning("pypdf not available - PDF support disabled")

# --- Vector Store and Embedding Imports ---
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Tool Imports ---
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
# from app.agents.tools.base import BaseAgentTool
# from app.core.config import get_settings
from agents.tools.base import BaseAgentTool
from core.config import get_settings
settings = get_settings()

class DocumentWatcher(FileSystemEventHandler):
    """File system event handler for monitoring document changes"""
    
    def __init__(self, retriever_tool):
        """Initialize the document watcher with a retriever tool reference"""
        self.retriever_tool = retriever_tool
        self.supported_extensions = {'.txt', '.pdf', '.csv', '.json'}
        logger.info(f"Document watcher initialized for {len(self.supported_extensions)} file types")
    
    def on_created(self, event):
        """Handle file creation events"""
        if not event.is_directory:
            file_path = event.src_path
            if any(file_path.endswith(ext) for ext in self.supported_extensions):
                logger.info(f"New document detected: {file_path}")
                self.retriever_tool.load_new_document(file_path)
    
    def on_modified(self, event):
        """Handle file modification events"""
        if not event.is_directory:
            file_path = event.src_path
            if any(file_path.endswith(ext) for ext in self.supported_extensions) and "registry" not in file_path:
                logger.info(f"Document modified: {file_path}")
                self.retriever_tool.reload_document(file_path)
    
    def on_deleted(self, event):
        """Handle file deletion events"""
        if not event.is_directory:
            file_path = event.src_path
            if any(file_path.endswith(ext) for ext in self.supported_extensions):
                logger.info(f"Document deleted: {file_path}")
                # Optional: Handle document deletion if needed
                if hasattr(self.retriever_tool, 'remove_document'):
                    self.retriever_tool.remove_document(file_path)
class RetrievedMedicalDocument(BaseModel):
    """Model for a retrieved medical document with enhanced metadata for reranking"""
    content: str
    source: str
    retrieval_score: float
    relevance_score: float = 0.0
    condition_name: str = ""
    category: str = ""
    risk_level: str = ""
    metadata: Dict[str, Any] = {}
    
    def calculate_advanced_relevance(self, query: str) -> float:
        """Enhanced relevance scoring with multiple factors for medical content"""
        query_lower = query.lower()
        content_lower = self.content.lower()
        
        # 1. Exact phrase matching (highest weight)
        exact_match_score = 0.0
        if query_lower in content_lower:
            exact_match_score = 1.0
        
        # 2. Medical condition matching
        condition_match_score = 0.0
        query_terms = set(query_lower.split())
        if self.condition_name and any(term in self.condition_name.lower() for term in query_terms):
            condition_match_score = 0.8
        
        # 3. Category relevance (specialized medical categories)
        category_match_score = 0.0
        if self.category:
            category_lower = self.category.lower()
            for term in query_terms:
                if term in category_lower:
                    category_match_score = 0.6
                    break
        
        # 4. Term overlap with position weighting
        query_terms = query_lower.split()
        content_terms = content_lower.split()
        overlap_score = 0.0
        
        for q_term in query_terms:
            for i, c_term in enumerate(content_terms):
                if q_term in c_term or c_term in q_term:
                    # Earlier positions get higher weight
                    position_weight = 1.0 / (i + 1) if i < 20 else 0.1
                    overlap_score += position_weight
        
        overlap_score = min(overlap_score / max(1, len(query_terms)), 1.0)
        
        # 5. Risk level priority (higher risk gets more attention)
        risk_score = 0.0
        if self.risk_level:
            risk_lower = self.risk_level.lower()
            if 'cao' in risk_lower or 'high' in risk_lower:
                risk_score = 0.9
            elif 'trung bình' in risk_lower or 'medium' in risk_lower:
                risk_score = 0.6
            elif 'thấp' in risk_lower or 'low' in risk_lower:
                risk_score = 0.3
        
        # 6. Content length penalty (prefer concise, relevant content)
        length_penalty = 1.0 - min(len(self.content) / 2000, 0.3)
        
        # Combine all scores with weights
        self.relevance_score = (
            exact_match_score * 0.25 +
            condition_match_score * 0.25 +
            category_match_score * 0.15 +
            overlap_score * 0.15 +
            risk_score * 0.1 +
            self.retrieval_score * 0.05 +
            length_penalty * 0.05
        )
        
        return self.relevance_score

class MedicalRetrieverTool(BaseAgentTool):
    """Enhanced medical retrieval tool with dynamic loading, directory watching, and advanced reranking"""
    
    # Define Pydantic fields
    collection_name: str = Field(..., description="Collection name for the retriever")
    use_bm25: bool = Field(default=True, description="Whether to use BM25 retriever")
    collection_type: str = Field(default="medical", description="Type of collection")
    is_initialized: bool = Field(default=False, description="Initialization status")
    vector_store: Any = Field(default=None, description="Vector store instance")
    bm25_retriever: Any = Field(default=None, description="BM25 retriever instance")
    embeddings: Any = Field(default=None, description="Embeddings model")
    
    # Enhanced fields for file watching and document management
    watch_directory: str = Field(default="", description="Directory to watch for new documents")
    document_registry: Dict[str, str] = Field(default_factory=dict, description="Registry of loaded documents with hashes")
    observer: Any = Field(default=None, description="File system observer")
    text_splitter: Any = Field(default=None, description="Text splitter for large documents")
    auto_scan_enabled: bool = Field(default=True, description="Whether automatic scanning is enabled")
    supported_formats: List[str] = Field(default_factory=lambda: ['.txt', '.pdf', '.csv', '.json'], description="Supported document formats")
    name: str = Field(default="medical_retriever", description="Tool name")
    description: str = Field(default="Retrieve medical information from the medical knowledge base with reranking for improved clinical relevance.", 
                            description="Tool description")
                            
    def __init__(self, 
                 collection_name: str = "medical_data", 
                 use_bm25: bool = True, 
                 collection_type: str = "medical",
                 watch_directory: str = None,
                 auto_scan_enabled: bool = True,
                 **kwargs):
        # Sanitize collection name for ChromaDB compatibility
        safe_collection_name = collection_name.replace(" ", "_").replace("-", "_")
        if len(safe_collection_name) > 63:
            safe_collection_name = safe_collection_name[:63]  # ChromaDB has a length limit
        
        # Set default watch directory if not provided
        if not watch_directory:
            watch_directory = os.path.join(os.path.dirname(__file__), 'storages', 'medicines')
        
        # Initialize tool with proper name and description
        name = f"medical_retriever"
        description = f"Advanced medical information retrieval with dynamic data loading and reranking for improved clinical relevance"
        
        super().__init__(
            name=name, 
            description=description,
            collection_name=safe_collection_name,
            use_bm25=use_bm25,
            collection_type=collection_type,
            watch_directory=watch_directory,
            auto_scan_enabled=auto_scan_enabled,
            **kwargs
        )
        
        # Initialize text splitter for large documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            length_function=len,
        )
        
        # Initialize the tool components
        self._initialize_components()
        
        # Auto-load documents from multiple formats
        self._auto_load_documents()
        self.load_document_registry()
        # Start document watching if enabled
        if self.auto_scan_enabled:
            self._start_document_watcher()
            self.scan_for_new_documents()
    
    def _initialize_components(self):
        """Initialize vector store and other components with simplified error handling"""
        try:
            # Setup persistent client for ChromaDB
            persistent_client = chromadb.PersistentClient(path=settings.VECTOR_STORE_BASE_DIR)
            logger.info(f"ChromaDB client initialized successfully")
            
            # Setup embeddings model
            self.embeddings = OllamaEmbeddings(
                model=settings.EMBEDDING_MODEL, 
                base_url=settings.OLLAMA_BASE_URL
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
                    if all_docs and 'documents' in all_docs and all_docs['documents']:
                        documents = [
                            Document(
                                page_content=doc_text, 
                                metadata=all_docs.get('metadatas', [{}])[i] if i < len(all_docs.get('metadatas', [])) else {}
                            )
                            for i, doc_text in enumerate(all_docs['documents'])
                        ]
                        
                        self.bm25_retriever = BM25Retriever.from_documents(documents)
                        logger.info(f"BM25 retriever initialized with {len(documents)} documents")
                    else:
                        logger.warning("No documents available for BM25 indexing")
                        self.bm25_retriever = None
                except Exception as e:
                    logger.warning(f"BM25 initialization failed: {e}")
                    self.bm25_retriever = None
            
            # Mark initialization as successful
            self.is_initialized = True
            logger.info(f"Medical retriever initialized successfully with collection '{self.collection_name}'")
            
        except Exception as e:
            logger.error(f"Failed to initialize medical retriever: {e}")
            self.is_initialized = False
            self.vector_store = None
            self.bm25_retriever = None
    
    def _auto_load_documents(self):
        """Automatically load documents from multiple formats if vector store is empty"""
        try:
            if hasattr(self.vector_store, '_collection'):
                current_count = self.vector_store._collection.count()
                if current_count == 0:
                    logger.info("Vector store is empty, auto-loading medical documents...")
                    self.scan_and_load_all_documents()
                else:
                    logger.info(f"Vector store already has {current_count} documents")
        except Exception as e:
            logger.warning(f"Could not check vector store count: {e}")
            # Try to load anyway
            self.scan_and_load_all_documents()
    
    def _start_document_watcher(self) -> None:
        """Initialize and start the document watcher to monitor for file changes"""
        if not self.watch_directory:
            logger.warning("No watch directory specified, skipping document watcher")
            return
            
        if not os.path.exists(self.watch_directory):
            logger.warning(f"Watch directory not found: {self.watch_directory}")
            return
            
        try:
            # Create a document watcher instance
            event_handler = DocumentWatcher(self)
            
            # Initialize observer
            self.observer = Observer()
            self.observer.schedule(event_handler, self.watch_directory, recursive=True)
            self.observer.start()
            
            logger.info(f"Started watching directory for changes: {self.watch_directory}")
        except Exception as e:
            logger.error(f"Error starting document watcher: {e}")
    
    def stop_document_watcher(self) -> None:
        """Stop the document watcher if it's running"""
        if hasattr(self, 'observer') and self.observer:
            try:
                self.observer.stop()
                self.observer.join()
                logger.info("Document watcher stopped")
            except Exception as e:
                logger.error(f"Error stopping document watcher: {e}")
    
    def get_document_hash(self, file_path: str) -> str:
        """Calculate hash of document for change detection"""
        try:
            with open(file_path, 'rb') as f:
                file_bytes = f.read()
                return hashlib.md5(file_bytes).hexdigest()
        except Exception as e:
            logger.error(f"Error calculating document hash for {file_path}: {e}")
            return ""
    
    def is_document_changed(self, file_path: str) -> bool:
        """Check if document has changed since last loading"""
        if file_path not in self.document_registry:
            return True
        
        current_hash = self.get_document_hash(file_path)
        return current_hash != self.document_registry.get(file_path, "")
    
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
            # self.document_registry = {}

    def register_document(self, file_path: str) -> None:
        """Register document in the registry with current hash"""
        self.document_registry[file_path] = self.get_document_hash(file_path)
    
    def scan_and_load_all_documents(self) -> None:
        """Scan watch directory and load all supported documents"""
        if not self.watch_directory or not os.path.exists(self.watch_directory):
            logger.warning(f"Watch directory not found or not set: {self.watch_directory}")
            return
        
        try:
            loaded_count = 0
            for root, _, files in os.walk(self.watch_directory):
                for file in files:
                    file_path = os.path.join(root, file)
                    
                    # Check if file has a supported extension
                    if any(file_path.endswith(ext) for ext in self.supported_formats):
                        # Check if file was already loaded or has changed
                        current_hash = self.get_document_hash(file_path)
                        if file_path not in self.document_registry or self.document_registry[file_path] != current_hash:
                            logger.info(f"Loading document: {file_path}")
                            documents = self.load_document_by_format(file_path)
                            if documents:
                                self.add_documents(documents)
                                self.document_registry[file_path] = current_hash
                                loaded_count += 1
            
            # Also check for legacy txt file
            legacy_path = Path(__file__).parent / "storages" / "medicines" / "medical_docs.txt"
            if os.path.exists(legacy_path):
                current_hash = self.get_document_hash(str(legacy_path))
                if str(legacy_path) not in self.document_registry or self.document_registry[str(legacy_path)] != current_hash:
                    logger.info(f"Loading legacy medical document: {legacy_path}")
                    documents = self.load_documents_from_txt(str(legacy_path))
                    if documents:
                        self.add_documents(documents)
                        self.document_registry[str(legacy_path)] = current_hash
                        loaded_count += 1
            
            logger.info(f"Scanned and loaded {loaded_count} documents from {self.watch_directory}")
        
        except Exception as e:
            logger.error(f"Error scanning and loading documents: {e}")
    
    def load_new_document(self, file_path: str) -> None:
        """Load a single new document (called by file watcher)"""
        if not os.path.exists(file_path):
            logger.warning(f"Document not found: {file_path}")
            return
        
        try:
            # c# Get file hash for registry
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

            if not documents:
                logger.warning(f"No content extracted from {file_path}")
                return
        except Exception as e:
            logger.error(f"Error loading new document {file_path}: {e}")
    
    def reload_document(self, file_path: str) -> None:
        """Reload a modified document (called by file watcher)"""
        if not os.path.exists(file_path):
            logger.warning(f"Document not found: {file_path}")
            return
        
        try:
            # Calculate new hash
            current_hash = self.get_document_hash(file_path)
            old_hash = self.document_registry.get(file_path, "")
            
            # Only reload if hash changed
            if current_hash != old_hash:
                logger.info(f"Document changed, reloading: {file_path}")
                documents = self.load_document_by_format(file_path)
                if documents:
                    # Remove old entries with same source
                    if hasattr(self.vector_store, '_collection'):
                        self.vector_store._collection.delete(
                            where={"source": os.path.basename(file_path)}
                        )
                    # Add new documents
                    self.add_documents(documents)
                    self.document_registry[file_path] = current_hash
                    logger.info(f"Successfully reloaded document: {file_path}")
            else:
                logger.info(f"Document unchanged, skipping reload: {file_path}")
        except Exception as e:
            logger.error(f"Error reloading document {file_path}: {e}")
    
    def remove_document(self, file_path: str) -> None:
        """Remove document from registry when deleted"""
        if file_path in self.document_registry:
            self.document_registry.pop(file_path)
            logger.info(f"Removed document from registry: {file_path}")
            # Note: Documents are still in the vector store
            # Advanced implementation would remove vectors as well
    
    def load_document_by_format(self, file_path: str) -> List[Document]:
        """Load document based on file format"""
        _, ext = os.path.splitext(file_path.lower())
        
        if ext == '.txt':
            return self.load_documents_from_txt(file_path)
        elif ext == '.pdf':
            return self.load_documents_from_pdf(file_path)
        elif ext == '.csv':
            return self.load_documents_from_csv(file_path)
        elif ext == '.json':
            return self.load_documents_from_json(file_path)
        else:
            logger.warning(f"Unsupported file format: {ext}")
            return []
    
    def load_documents_from_txt(self, txt_file_path: str) -> List[Document]:
        """Load documents from the medical_docs.txt file"""
        documents = []
        
        try:
            with open(txt_file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Split by double newlines to separate medical entries
            entries = content.strip().split('\n\n')
            
            for i, entry in enumerate(entries):
                if entry.strip():
                    # Parse medical document content
                    parsed_metadata = self.parse_medical_document(entry)
                    
                    doc = Document(
                        page_content=entry.strip(),
                        metadata={
                            'source': os.path.basename(txt_file_path),
                            'doc_type': 'txt',
                            'entry_id': i,
                            'condition_name': parsed_metadata.get('condition_name', ''),
                            'category': parsed_metadata.get('category', ''),
                            'risk_level': parsed_metadata.get('risk_level', ''),
                            'creation_date': datetime.now().isoformat()
                        }
                    )
                    documents.append(doc)
            
            logger.info(f"Loaded {len(documents)} documents from {txt_file_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading documents from {txt_file_path}: {e}")
            return []

    def load_documents_from_pdf(self, pdf_path: str) -> List[Document]:
        """Load documents from PDF file"""
        if not PDF_AVAILABLE:
            logger.warning("PDF support not available (pypdf not installed)")
            return []
        
        documents = []
        try:
            reader = PdfReader(pdf_path)
            full_text = ""
            
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    full_text += f"\n\nPage {page_num + 1}:\n{text}"
            
            if full_text.strip():
                # Split large PDFs into chunks
                chunks = self.text_splitter.split_text(full_text)
                
                for i, chunk in enumerate(chunks):
                    if chunk.strip():
                        # Parse for medical information
                        parsed = self.parse_medical_document(chunk)
                        
                        doc = Document(
                            page_content=chunk.strip(),
                            metadata={
                                'source': os.path.basename(pdf_path),
                                'file_type': 'pdf',
                                'page': i,
                                'condition_name': parsed.get('condition_name', ''),
                                'category': parsed.get('category', ''),
                                'risk_level': parsed.get('risk_level', ''),
                                'creation_date': datetime.now().isoformat()
                            }
                        )
                        documents.append(doc)
            
            logger.info(f"Loaded {len(documents)} chunks from PDF: {pdf_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading PDF {pdf_path}: {e}")
            return []

    def load_documents_from_csv(self, csv_path: str) -> List[Document]:
        """Load documents from CSV file"""
        if not PANDAS_AVAILABLE:
            logger.warning("CSV support limited (pandas not installed)")
            return []
            
        documents = []
        try:
            df = pd.read_csv(csv_path)
            
            for index, row in df.iterrows():
                # Handle different CSV formats
                content = ""
                metadata = {
                    'source': os.path.basename(csv_path),
                    'row': index,
                    'doc_type': 'csv',
                    'creation_date': datetime.now().isoformat()
                }
                
                # Try to extract content and metadata from common columns
                if 'content' in df.columns:
                    content = row['content']
                elif 'text' in df.columns:
                    content = row['text']
                elif 'description' in df.columns:
                    content = row['description']
                else:
                    # Concatenate all columns as content
                    content = " | ".join([f"{col}: {val}" for col, val in row.items()])
                
                # Extract medical metadata
                if 'condition_name' in df.columns:
                    metadata['condition_name'] = row['condition_name']
                if 'category' in df.columns:
                    metadata['category'] = row['category']
                if 'risk_level' in df.columns:
                    metadata['risk_level'] = row['risk_level']
                
                if content.strip():
                    doc = Document(page_content=content.strip(), metadata=metadata)
                    documents.append(doc)
            
            logger.info(f"Loaded {len(documents)} rows from CSV: {csv_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading CSV {csv_path}: {e}")
            return []
        
    def scan_for_new_documents(self):
        """Scan for new or modified documents"""
        if not os.path.exists(self.watch_directory):
            return
        
        new_docs_count = 0
        for root, dirs, files in os.walk(self.watch_directory):
            for file in files:
                file_path = os.path.join(root, file)
                _, ext = os.path.splitext(file.lower())
                
                if ext in self.supported_formats and "registry" not in file_path:
                    current_hash = self.get_document_hash(file_path)
                    stored_hash = self.document_registry.get(file_path)
                    
                    if current_hash != stored_hash or file_path not in self.document_registry:
                        # New or modified document
                        documents = self.load_document_by_format(file_path)
                        if documents:
                            self.add_documents(documents)
                            new_docs_count += len(documents)
                            self.document_registry[file_path] = current_hash
                            logger.info(f"Loaded new/modified document: {file_path}")
        self.store_load_document_registry()
        if new_docs_count > 0:
            logger.info(f"Found and loaded {new_docs_count} new/modified documents")
            self._start_document_watcher()

    def load_documents_from_json(self, json_path: str) -> List[Document]:
        """Load documents from JSON file"""
        documents = []
        try:
            with open(json_path, 'r', encoding='utf-8') as file:
                json_data = json.load(file)
            
            # Process as list of records
            if isinstance(json_data, list):
                for i, item in enumerate(json_data):
                    if isinstance(item, dict):
                        # Extract content field or convert whole item to string
                        content = ""
                        if 'content' in item:
                            content = item['content']
                        elif 'text' in item:
                            content = item['text']
                        else:
                            # Convert item to formatted string
                            content = json.dumps(item, ensure_ascii=False, indent=2)
                        
                        # Extract medical metadata
                        metadata = {
                            'source': os.path.basename(json_path),
                            'doc_type': 'json',
                            'item_index': i,
                            'creation_date': datetime.now().isoformat()
                        }
                        
                        if 'condition_name' in item:
                            metadata['condition_name'] = item['condition_name']
                        if 'category' in item:
                            metadata['category'] = item['category']
                        if 'risk_level' in item:
                            metadata['risk_level'] = item['risk_level']
                        
                        if content.strip():
                            doc = Document(page_content=content.strip(), metadata=metadata)
                            documents.append(doc)
            
            # Process as single object with array fields
            elif isinstance(json_data, dict):
                # Look for array fields that might contain medical data
                for key, value in json_data.items():
                    if isinstance(value, list) and value and isinstance(value[0], dict):
                        for i, item in enumerate(value):
                            # Extract content or convert item to string
                            content = json.dumps(item, ensure_ascii=False, indent=2)
                            
                            metadata = {
                                'source': os.path.basename(json_path),
                                'doc_type': 'json',
                                'array_field': key,
                                'item_index': i,
                                'creation_date': datetime.now().isoformat()
                            }
                            
                            # Extract medical metadata
                            if 'condition_name' in item:
                                metadata['condition_name'] = item['condition_name']
                            if 'category' in item:
                                metadata['category'] = item['category']
                            if 'risk_level' in item:
                                metadata['risk_level'] = item['risk_level']
                            
                            if content.strip():
                                doc = Document(page_content=content.strip(), metadata=metadata)
                                documents.append(doc)
                
                # If no arrays were found, treat the whole object as one document
                if not documents:
                    content = json.dumps(json_data, ensure_ascii=False, indent=2)
                    doc = Document(
                        page_content=content.strip(),
                        metadata={
                            'source': os.path.basename(json_path),
                            'doc_type': 'json',
                            'creation_date': datetime.now().isoformat()
                        }
                    )
                    documents.append(doc)
            
            logger.info(f"Loaded {len(documents)} documents from JSON: {json_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading JSON {json_path}: {e}")
            return []

    def parse_medical_document(self, content: str) -> Dict[str, str]:
        """Parse medical document to extract structured information"""
        lines = content.split('\n')
        parsed = {
            'condition_name': '',
            'category': '',
            'trait': '',
            'expert_advice': '',
            'introduction': '',
            'conclusion': '',
            'risk_level': ''
        }
        
        for line in lines:
            line = line.strip()
            if line.startswith('Ten:') or line.startswith('Name:'):
                parsed['condition_name'] = line.replace('Ten:', '').replace('Name:', '').strip()
            elif line.startswith('Trait:'):
                parsed['trait'] = line.replace('Trait:', '').strip()
            elif line.startswith('Thuoc Nhom:'):
                parsed['category'] = line.replace('Thuoc Nhom:', '').strip()
            elif line.startswith('Goi y chuyen gia:'):
                parsed['expert_advice'] = line.replace('Goi y chuyen gia:', '').strip()
            elif line.startswith('Gioi thieu:'):
                parsed['introduction'] = line.replace('Gioi thieu:', '').strip()
            elif line.startswith('Ket luan:'):
                parsed['conclusion'] = line.replace('Ket luan:', '').strip()
        
        # Extract risk level from conclusion
        if parsed['conclusion']:
            conclusion_lower = parsed['conclusion'].lower()
            if 'cao' in conclusion_lower or 'high' in conclusion_lower:
                parsed['risk_level'] = 'cao'
            elif 'thấp' in conclusion_lower or 'low' in conclusion_lower:
                parsed['risk_level'] = 'thấp'
            elif 'trung bình' in conclusion_lower or 'bình thường' in conclusion_lower:
                parsed['risk_level'] = 'bình thường'
        
        return parsed
    
    def _run(self, query: str) -> List[str]:
        """Execute the medical retriever with the provided query"""
        return self.retrieve_documents(query)
        
    async def _arun(self, query: str) -> List[str]:
        """Execute the medical retriever asynchronously with the provided query"""
        return self.retrieve_documents(query)  # For now, just call the sync version
    
    # Retrieval parameters
    max_results: int = Field(default=5, description="Maximum number of results to return")
    similarity_threshold: float = Field(default=0.1, description="Minimum similarity threshold")
    rerank_threshold: float = Field(default=0.3, description="Minimum relevance score after reranking")
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats"""
        return self.supported_formats.copy()

    def set_watch_directory(self, new_directory: str):
        """Set a new watch directory and restart monitoring"""
        # Stop current watcher if running
        if hasattr(self, 'observer') and self.observer:
            self.stop_document_watcher()
        
        # Update directory and restart if it exists
        self.watch_directory = new_directory
        if os.path.exists(new_directory):
            self._start_document_watcher()
            logger.info(f"Watch directory changed to: {new_directory}")
        else:
            logger.warning(f"New watch directory does not exist: {new_directory}")

    def enable_auto_scan(self):
        """Enable automatic document scanning"""
        self.auto_scan_enabled = True
        if self.watch_directory and os.path.exists(self.watch_directory):
            self._start_document_watcher()

    def disable_auto_scan(self):
        """Disable automatic document scanning"""
        self.auto_scan_enabled = False
        self.stop_document_watcher()

    def manual_scan(self):
        """Manually scan and load documents"""
        self.scan_and_load_all_documents()

    def get_document_registry(self) -> Dict[str, str]:
        """Get the document registry with file hashes"""
        return self.document_registry.copy()

    def clear_document_registry(self):
        """Clear the document registry"""
        self.document_registry.clear()
        logger.info("Document registry cleared")

    def force_reload_all(self):
        """Force reload all documents by clearing registry first"""
        self.clear_document_registry()
        self.scan_and_load_all_documents()

    def get_load_status(self) -> Dict[str, Any]:
        """Get comprehensive loading status"""
        status = {
            "is_initialized": self.is_initialized,
            "watch_directory": self.watch_directory,
            "auto_scan_enabled": self.auto_scan_enabled,
            "documents_registered": len(self.document_registry),
            "supported_formats": self.get_supported_formats(),
            "collection_name": self.collection_name,
            "use_bm25": self.use_bm25,
            "observer_running": hasattr(self, 'observer') and self.observer and self.observer.is_alive() if hasattr(self, 'observer') else False
        }
        
        # Add vector store info if available
        if hasattr(self, 'vector_store') and self.vector_store:
            try:
                status["document_count"] = len(self.vector_store.get()['documents']) if self.vector_store.get()['documents'] else 0
            except:
                status["document_count"] = 0
        
        return status

    def __del__(self):
        """Cleanup when object is deleted"""
        try:
            self.cleanup()
        except:
            pass

    def cleanup(self):
        """Clean up resources"""
        try:
            self.stop_document_watcher()
            logger.info("Medical retriever cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def rerank_documents(self, documents: List[RetrievedMedicalDocument], query: str) -> List[RetrievedMedicalDocument]:
        """Enhanced reranking with multiple scoring factors for medical content"""
        if not documents:
            return documents
        
        # Parse documents for structured information
        for doc in documents:
            parsed = self.parse_medical_document(doc.content)
            doc.condition_name = parsed['condition_name']
            doc.category = parsed['category']
            doc.risk_level = parsed['risk_level']
            
            # Calculate advanced relevance score
            doc.calculate_advanced_relevance(query)
        
        # Sort by relevance score (highest first)
        reranked = sorted(documents, key=lambda x: x.relevance_score, reverse=True)
        
        # Apply diversity filtering to avoid too many similar results
        filtered_results = []
        seen_conditions = set()
        
        for doc in reranked:
            # Add document if it's about a new condition or has very high relevance
            if doc.condition_name not in seen_conditions or doc.relevance_score > 0.8:
                filtered_results.append(doc)
                seen_conditions.add(doc.condition_name)
                
                # Limit to top 10 results for better quality
                if len(filtered_results) >= 10:
                    break
        
        logger.info(f"Reranked {len(documents)} documents to {len(filtered_results)} high-quality results")
        return filtered_results

    def get_relevant_documents_with_reranking(self, query: str, k: int = 10) -> List[RetrievedMedicalDocument]:
        """Get relevant documents with enhanced reranking for medical content"""
        if not self.is_initialized:
            logger.error("Cannot retrieve documents: Retriever not initialized.")
            return []
            
        retrieved_docs = []
        
        try:
            # Step 1: Retrieve more candidates for better reranking
            initial_k = min(k * 2, 20)  # Get more candidates initially
            
            # BM25 retrieval
            if self.use_bm25 and self.bm25_retriever is not None:
                bm25_docs = self.bm25_retriever.get_relevant_documents(query)[:initial_k]
                for i, doc in enumerate(bm25_docs):
                    retrieved_doc = RetrievedMedicalDocument(
                        content=doc.page_content,
                        source=doc.metadata.get('source', 'unknown'),
                        retrieval_score=1.0 - (i * 0.1),  # Simple scoring
                        metadata=doc.metadata
                    )
                    retrieved_docs.append(retrieved_doc)
            
            # Vector search
            vector_docs = self.vector_store.similarity_search_with_score(query, k=initial_k)
            for doc, score in vector_docs:
                # Skip duplicates
                if any(d.content == doc.page_content for d in retrieved_docs):
                    continue
                
                # Normalize score (ChromaDB returns distance, lower is better)
                normalized_score = max(0.0, 1.0 - min(score, 2.0) / 2.0)
                
                retrieved_doc = RetrievedMedicalDocument(
                    content=doc.page_content,
                    source=doc.metadata.get('source', 'unknown'),
                    retrieval_score=normalized_score,
                    metadata=doc.metadata
                )
                retrieved_docs.append(retrieved_doc)
            
            # Step 2: Apply enhanced reranking
            reranked_docs = self.rerank_documents(retrieved_docs, query)
            
            # Step 3: Return top k results
            return reranked_docs[:k]
            
        except Exception as e:
            logger.error(f"Error in enhanced document retrieval: {e}")
            return []

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the loaded medical documents"""
        try:
            if not self.vector_store:
                return {"error": "Vector store not initialized"}
            
            all_docs = self.vector_store.get()
            if not all_docs or not all_docs.get('documents'):
                return {"total_documents": 0, "message": "No documents loaded"}
            
            documents = all_docs['documents']
            metadatas = all_docs.get('metadatas', [])
            
            # Count unique conditions and categories
            unique_conditions = set()
            unique_categories = set()
            risk_levels = {'cao': 0, 'thấp': 0, 'bình thường': 0, 'unknown': 0}
            
            for i, doc in enumerate(documents):
                parsed = self.parse_medical_document(doc)
                condition_name = parsed.get('condition_name', '')
                category = parsed.get('category', '')
                risk_level = parsed.get('risk_level', 'unknown')
                
                if condition_name:
                    unique_conditions.add(condition_name)
                if category:
                    unique_categories.add(category)
                if risk_level in risk_levels:
                    risk_levels[risk_level] += 1
                else:
                    risk_levels['unknown'] += 1
            
            return {
                "total_documents": len(documents),
                "unique_conditions": len(unique_conditions),
                "unique_categories": len(unique_categories),
                "condition_names": list(unique_conditions)[:10],  # First 10 for brevity
                "categories": list(unique_categories),
                "risk_distribution": risk_levels,
                "collection_name": self.collection_name,
                "watch_directory": self.watch_directory,
                "registry_size": len(self.document_registry)
            }
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {"error": str(e)}

    def add_documents(self, documents: List[Document]) -> bool:
        """Add documents to the vector store with enhanced error handling"""
        if not self.is_initialized:
            logger.error("Cannot add documents: Retriever not initialized.")
            return False
        
        if not documents:
            logger.warning("No documents provided to add")
            return False
        
        try:
            # Add to vector store
            self.vector_store.add_documents(documents)
            
            # Update BM25 if available and we have new documents
            if self.use_bm25:
                try:
                    all_docs = self.vector_store.get()
                    if all_docs and all_docs.get('documents'):
                        documents_for_bm25 = [
                            Document(
                                page_content=doc_text,
                                metadata=all_docs.get('metadatas', [{}])[i] if i < len(all_docs.get('metadatas', [])) else {}
                            )
                            for i, doc_text in enumerate(all_docs['documents'])
                        ]
                        self.bm25_retriever = BM25Retriever.from_documents(documents_for_bm25)
                        self.bm25_retriever.k = 10
                        logger.debug(f"Updated BM25 retriever with {len(documents_for_bm25)} documents")
                except Exception as e:
                    logger.warning(f"Failed to update BM25 retriever: {e}")
            
            logger.info(f"Added {len(documents)} documents to medical collection")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            return False

    def retrieve_documents(self, query: str) -> List[str]:
        """Enhanced document retrieval with advanced reranking for better medical context."""
        if not hasattr(self, 'vector_store') or self.vector_store is None:
            logger.warning("Vector store not available. Cannot retrieve documents.")
            return [f"Error: {self.collection_type.capitalize()} document retrieval is unavailable."]
        
        try:
            time_start = time.time()
            
            # Use enhanced retrieval with reranking
            reranked_docs = self.get_relevant_documents_with_reranking(query, k=8)
            
            if not reranked_docs:
                return [f"No relevant medical documents found for query: {query}"]
            
            # Format results with enhanced relevance information
            formatted_results = []
            
            for i, doc in enumerate(reranked_docs):
                result = f"""
                            📋 **Medical Document {i+1}** (Relevance: {doc.relevance_score:.2f})
                            🏥 **Condition**: {doc.condition_name or 'General Medical Info'}
                            📁 **Category**: {doc.category or 'Uncategorized'}
                            ⚠️ **Risk Level**: {doc.risk_level or 'Not specified'}
                            📄 **Source**: {doc.source}

                            **Content**:
                            {doc.content[:800]}{'...' if len(doc.content) > 800 else ''}

                            ---
                            """
                formatted_results.append(result.strip())
            
            logger.info(f"Enhanced medical document retrieval completed in {time.time() - time_start:.2f} seconds with {len(formatted_results)} reranked results.")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error during enhanced document retrieval: {e}", exc_info=True)
            return [f"Error retrieving documents: {e}"]

    def initialize_from_txt(self, txt_file_path: str = None) -> bool:
        """Initialize the retriever with documents from txt file"""
        if not txt_file_path:
            # Default path to medical_docs.txt
            txt_file_path = os.path.join(
                os.path.dirname(__file__), 
                'storages', 
                'medicines',
                'medical_docs.txt'
            )
        
        if not os.path.exists(txt_file_path):
            logger.error(f"Medical documents file not found: {txt_file_path}")
            return False
        
        # Load documents from txt file
        documents = self.load_documents_from_txt(txt_file_path)
        
        if documents:
            # Add documents to the vector store
            success = self.add_documents(documents)
            if success:
                logger.info(f"Successfully initialized medical retriever with {len(documents)} documents")
                return True
            else:
                logger.error("Failed to add documents to vector store")
                return False
        else:
            logger.warning("No documents loaded from txt file")
            return False



# if __name__ == "__main__":
#     # Example usage
#     retriever = MedicalRetrieverTool(
#         collection_name="medical_docs",
#         watch_directory="app/agents/retrievers/storages/medicines",
#         use_bm25=True,
#         auto_scan_enabled=True
#     )
    
#     # Initialize from txt file
#     retriever.initialize_from_txt("app/agents/retrievers/storages/medicines/medical_docs.txt")
    
#     # Perform a sample retrieval
#     query_result = retriever.run(query="What are the symptoms of diabetes?")
#     print(query_result)
    
#     # Cleanup resources
#     retriever.cleanup()