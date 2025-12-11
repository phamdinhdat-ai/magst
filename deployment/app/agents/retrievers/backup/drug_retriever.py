# --- Enhanced Drug Retriever Tool ---
import os
import sys
import time
import json
import hashlib
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path
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
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logger.warning("PyPDF2 not available - PDF support disabled")

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
settings = get_settings()

class DocumentWatcher(FileSystemEventHandler):
    """File system event handler for monitoring document changes"""
    
    def __init__(self, retriever_tool):
        self.retriever_tool = retriever_tool
        self.supported_extensions = {'.txt', '.pdf', '.csv', '.json'}
        super().__init__()
    
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
    
    def _is_supported_file(self, file_path: str) -> bool:
        """Check if file is supported for processing"""
        return any(file_path.lower().endswith(ext) for ext in self.supported_extensions)

class RetrievedDocument(BaseModel):
    """Model for a retrieved document with enhanced metadata for reranking"""
    content: str
    source: str
    retrieval_score: float
    relevance_score: float = 0.0
    drug_name: str = ""
    category: str = ""
    
    def calculate_advanced_relevance(self, query: str) -> float:
        """Enhanced relevance scoring with multiple factors"""
        query_lower = query.lower()
        content_lower = self.content.lower()
        
        # 1. Exact phrase matching (highest weight)
        exact_match_score = 0.0
        if query_lower in content_lower:
            exact_match_score = 1.0
        
        # 2. Drug name matching
        drug_match_score = 0.0
        query_terms = set(query_lower.split())
        if self.drug_name and any(term in self.drug_name.lower() for term in query_terms):
            drug_match_score = 0.8
        
        # 3. Term overlap with position weighting
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
        
        # 4. Content length penalty (prefer concise, relevant content)
        length_penalty = 1.0 - min(len(self.content) / 1000, 0.3)
        
        # 5. Source reliability score
        source_score = 0.9 if 'CPIC' in self.source else 0.7 if 'FDA' in self.source else 0.5
        
        # Combine all scores with weights
        self.relevance_score = (
            exact_match_score * 0.3 +
            drug_match_score * 0.25 +
            overlap_score * 0.2 +
            self.retrieval_score * 0.15 +
            length_penalty * 0.05 +
            source_score * 0.05
        )
        
        return self.relevance_score

class DrugRetrieverTool(BaseAgentTool):
    """Enhanced drug retrieval tool with multi-format support, auto-scanning, and file watching"""
    # Define Pydantic fields
    collection_name: str = Field(..., description="Collection name for the retriever")
    use_bm25: bool = Field(default=True, description="Whether to use BM25 retriever")
    collection_type: str = Field(default="drug", description="Type of collection")
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
    
    def __init__(self, 
                 collection_name: str = "drug_data", 
                 use_bm25: bool = True, 
                 collection_type: str = "drug",
                 watch_directory: str = None,
                 auto_scan_enabled: bool = True,
                 **kwargs):
        # Sanitize collection name for ChromaDB compatibility
        safe_collection_name = collection_name.replace(" ", "_").replace("-", "_")
        if len(safe_collection_name) > 63:
            safe_collection_name = safe_collection_name[:63]  # ChromaDB has a length limit
        
        # Set default watch directory if not provided
        if not watch_directory:
            watch_directory = os.path.join(os.path.dirname(__file__), 'storages', 'drugs')
        
        # Initialize tool with proper name and description
        name = f"drug_retriever"
        description = f"Enhanced drug information retrieval with multi-format support, auto-scanning, and real-time monitoring."
        
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
        # self._auto_load_documents()
        self.load_document_registry()
        # Start document watching if enabled
        if self.auto_scan_enabled:
            self._start_document_watcher()
            # self._auto_load_documents()
            self.scan_for_new_documents()
            # self._start_document_watcher()

    def __del__(self):
        """Cleanup resources when object is destroyed"""
        self.cleanup()
    
    
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
    
    
    def cleanup(self):
        """Clean up resources, especially file watcher"""
        if hasattr(self, 'observer') and self.observer:
            try:
                self.observer.stop()
                self.observer.join(timeout=1.0)
                logger.info("Document watcher stopped")
            except Exception as e:
                logger.warning(f"Error stopping document watcher: {e}")
            finally:
                self.observer = None
        logger.info("DrugRetrieverTool cleanup completed")
    
    def _start_document_watcher(self):
        """Start file system watcher for automatic document loading"""
        try:
            if not os.path.exists(self.watch_directory):
                logger.info(f"Creating watch directory: {self.watch_directory}")
                os.makedirs(self.watch_directory, exist_ok=True)
            
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
        if hasattr(self, 'observer') and self.observer:
            try:
                self.observer.stop()
                self.observer.join(timeout=1.0)
                self.observer = None
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
            logger.info(f"Drug retriever initialized successfully with collection '{self.collection_name}'")
            
        except Exception as e:
            logger.error(f"Failed to initialize drug retriever: {e}")
            self.is_initialized = False
            self.vector_store = None
            self.bm25_retriever = None
    
    def _auto_load_documents(self):
        """Automatically scan and load documents from multiple formats"""
        try:
            # Check if vector store has documents
            if self.vector_store:
                all_docs = self.vector_store.get()
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
    
    def scan_and_load_all_documents(self):
        """Scan directory and load all supported documents"""
        if not os.path.exists(self.watch_directory):
            logger.warning(f"Watch directory does not exist: {self.watch_directory}")
            # Try to create it
            try:
                os.makedirs(self.watch_directory, exist_ok=True)
                logger.info(f"Created watch directory: {self.watch_directory}")
            except Exception as e:
                logger.error(f"Failed to create watch directory: {e}")
            return
        
        total_loaded = 0
        for root, dirs, files in os.walk(self.watch_directory):
            logger.info(f"Scanning directory: {root}")
            for file in files:
                file_path = os.path.join(root, file)
                _, ext = os.path.splitext(file.lower())
                
                if ext in self.supported_formats:
                    # Check if file is already processed with the same hash
                    file_hash = self.get_document_hash(file_path)
                    if file_path in self.document_registry and self.document_registry[file_path] == file_hash:
                        logger.debug(f"Skipping already processed file: {file_path}")
                        continue
                        
                    documents = self.load_document_by_format(file_path)
                    if documents:
                        self.add_documents(documents)
                        self.document_registry[file_path] = file_hash
                        total_loaded += len(documents)
                        logger.info(f"Loaded {len(documents)} documents from {file_path}")
        
        logger.info(f"Scanned and loaded {total_loaded} documents from {self.watch_directory}")
    
    def scan_for_new_documents(self):
        """Scan for new or modified documents"""
        if not os.path.exists(self.watch_directory):
            logger.warning(f"Watch directory does not exist: {self.watch_directory}")
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
                        # File is new or modified
                        documents = self.load_document_by_format(file_path)
                        if documents:
                            self.add_documents(documents)
                            self.document_registry[file_path] = current_hash
                            new_docs_count += len(documents)
                            logger.info(f"Loaded new/modified file: {file_path} ({len(documents)} documents)")
        self.store_load_document_registry()
        
        if new_docs_count > 0:
            logger.info(f"Found and loaded {new_docs_count} new/modified documents")
        return new_docs_count
    
    def load_new_document(self, file_path: str):
        """Load a single new document (called by file watcher)"""
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

            if not documents:
                logger.warning(f"No content extracted from {file_path}")
                return
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
            
    def load_document_by_format(self, file_path: str) -> List[Document]:
        """Load document based on file format"""
        _, ext = os.path.splitext(file_path.lower())
        
        if ext == '.txt':
            return self.load_documents_from_txt(file_path)
        elif ext == '.pdf' and PDF_AVAILABLE:
            return self.load_pdf_document(file_path)
        elif ext == '.csv' and PANDAS_AVAILABLE:
            return self.load_csv_document(file_path)
        elif ext == '.json':
            return self.load_json_document(file_path)
        else:
            logger.warning(f"Unsupported file format: {ext}")
            return []
    
    def load_pdf_document(self, pdf_path: str) -> List[Document]:
        """Load documents from PDF file"""
        if not PDF_AVAILABLE:
            logger.error("PyPDF2 not available for PDF processing")
            return []
        
        documents = []
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                full_text = ""
                
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    if text:
                        full_text += f"\nPage {page_num + 1}:\n{text}"
                
                if full_text.strip():
                    # Split into drug entries if needed
                    chunks = self.text_splitter.split_text(full_text)
                    
                    for i, chunk in enumerate(chunks):
                        if chunk.strip():
                            parsed = self.parse_drug_document(chunk)
                            doc = Document(
                                page_content=chunk,
                                metadata={
                                    'source': f"PDF: {os.path.basename(pdf_path)}",
                                    'page': i,
                                    'drug_name': parsed.get('drug_name', ''),
                                    'category': parsed.get('category', ''),
                                    'loaded_at': datetime.now().isoformat()
                                }
                            )
                            documents.append(doc)
            
            logger.info(f"Loaded {len(documents)} documents from PDF: {pdf_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading PDF {pdf_path}: {e}")
            return []
    
    def load_csv_document(self, csv_path: str) -> List[Document]:
        """Load documents from CSV file"""
        if not PANDAS_AVAILABLE:
            logger.error("pandas not available for CSV processing")
            return []
        
        documents = []
        try:
            df = pd.read_csv(csv_path, encoding='utf-8')
            
            for index, row in df.iterrows():
                # Convert row to drug document format
                content_parts = []
                drug_name = ""
                category = ""
                
                for col, value in row.items():
                    if pd.notna(value):
                        col_clean = str(col).strip()
                        value_clean = str(value).strip()
                        
                        # Map common column names to drug format
                        if col_clean.lower() in ['drug', 'drug_name', 'medicine', 'thuoc', 'loai_thuoc']:
                            content_parts.append(f"Loai Thuoc: {value_clean}")
                            drug_name = value_clean
                        elif col_clean.lower() in ['target', 'patient', 'doi_tuong']:
                            content_parts.append(f"Doi Tuong: {value_clean}")
                        elif col_clean.lower() in ['description', 'desc', 'mo_ta']:
                            content_parts.append(f"Mo ta: {value_clean}")
                        elif col_clean.lower() in ['category', 'class', 'phan_loai']:
                            content_parts.append(f"Phan Loai: {value_clean}")
                            category = value_clean
                        elif col_clean.lower() in ['source', 'nguon']:
                            content_parts.append(f"Nguon: {value_clean}")
                        elif col_clean.lower() in ['recommendation', 'advice', 'khuyen_cao']:
                            content_parts.append(f"Khuyen cao: {value_clean}")
                        else:
                            content_parts.append(f"{col_clean}: {value_clean}")
                
                if content_parts:
                    content = "\n".join(content_parts)
                    doc = Document(
                        page_content=content,
                        metadata={
                            'source': f"CSV: {os.path.basename(csv_path)}",
                            'row_id': index,
                            'drug_name': drug_name,
                            'category': category,
                            'loaded_at': datetime.now().isoformat()
                        }
                    )
                    documents.append(doc)
            
            logger.info(f"Loaded {len(documents)} documents from CSV: {csv_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading CSV {csv_path}: {e}")
            return []
    
    def load_json_document(self, json_path: str) -> List[Document]:
        """Load documents from JSON file"""
        documents = []
        try:
            with open(json_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            # Handle different JSON structures
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict):
                # Try common keys for arrays
                items = data.get('drugs', data.get('medications', data.get('data', [data])))
                if not isinstance(items, list):
                    items = [items]
            else:
                items = [data]
            
            for i, item in enumerate(items):
                if isinstance(item, dict):
                    # Convert dict to drug document format
                    content_parts = []
                    drug_name = ""
                    category = ""
                    
                    for key, value in item.items():
                        if value:
                            key_clean = str(key).strip()
                            value_clean = str(value).strip()
                            
                            # Map JSON keys to drug format
                            if key_clean.lower() in ['drug', 'drug_name', 'medicine']:
                                content_parts.append(f"Loai Thuoc: {value_clean}")
                                drug_name = value_clean
                            elif key_clean.lower() in ['target', 'patient']:
                                content_parts.append(f"Doi Tuong: {value_clean}")
                            elif key_clean.lower() in ['description', 'desc']:
                                content_parts.append(f"Mo ta: {value_clean}")
                            elif key_clean.lower() in ['category', 'class']:
                                content_parts.append(f"Phan Loai: {value_clean}")
                                category = value_clean
                            elif key_clean.lower() in ['source']:
                                content_parts.append(f"Nguon: {value_clean}")
                            elif key_clean.lower() in ['recommendation', 'advice']:
                                content_parts.append(f"Khuyen cao: {value_clean}")
                            else:
                                content_parts.append(f"{key_clean}: {value_clean}")
                    
                    if content_parts:
                        content = "\n".join(content_parts)
                        doc = Document(
                            page_content=content,
                            metadata={
                                'source': f"JSON: {os.path.basename(json_path)}",
                                'item_id': i,
                                'drug_name': drug_name,
                                'category': category,
                                'loaded_at': datetime.now().isoformat()
                            }
                        )
                        documents.append(doc)
                else:
                    # Handle non-dict items
                    content = str(item)
                    if content.strip():
                        parsed = self.parse_drug_document(content)
                        doc = Document(
                            page_content=content,
                            metadata={
                                'source': f"JSON: {os.path.basename(json_path)}",
                                'item_id': i,
                                'drug_name': parsed.get('drug_name', ''),
                                'category': parsed.get('category', ''),
                                'loaded_at': datetime.now().isoformat()
                            }
                        )
                        documents.append(doc)
            
            logger.info(f"Loaded {len(documents)} documents from JSON: {json_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading JSON {json_path}: {e}")
            return []
    
    def _split_drug_content(self, content: str) -> List[str]:
        """Split content into individual drug entries"""
        # Split by double newlines (common separator)
        entries = content.strip().split('\n\n')
        
        # Further split if entries are too large
        final_entries = []
        for entry in entries:
            if len(entry) > 1000:  # If entry is too large, try to split further
                # Try splitting by single newlines and group by drug name patterns
                lines = entry.split('\n')
                current_entry = []
                
                for line in lines:
                    if line.strip().startswith('Loai Thuoc:') and current_entry:
                        # Start of new drug entry
                        final_entries.append('\n'.join(current_entry))
                        current_entry = [line]
                    else:
                        current_entry.append(line)
                
                if current_entry:
                    final_entries.append('\n'.join(current_entry))
            else:
                final_entries.append(entry)
        
        return [entry for entry in final_entries if entry.strip()]
    
    def load_documents_from_txt(self, txt_file_path: str) -> List[Document]:
        """Load documents from the drug_docs.txt file"""
        documents = []
        
        try:
            with open(txt_file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Split by double newlines to separate drug entries
            entries = self._split_drug_content(content)
            
            for i, entry in enumerate(entries):
                if entry.strip():
                    # Parse the entry to extract drug name for metadata
                    parsed = self.parse_drug_document(entry)
                    
                    doc = Document(
                        page_content=entry.strip(),
                        metadata={
                            'source': os.path.basename(txt_file_path),
                            'drug_name': parsed.get('drug_name', ''),
                            'category': parsed.get('category', ''),
                            'entry_id': i,
                            'loaded_at': datetime.now().isoformat()
                        }
                    )
                    documents.append(doc)
            
            logger.info(f"Loaded {len(documents)} documents from {txt_file_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading documents from {txt_file_path}: {e}")
            return []

    def initialize_from_txt(self, txt_file_path: str = None):
        """Initialize the retriever with documents from txt file (legacy method)"""
        if not txt_file_path:
            # Default path to drug_docs.txt
            txt_file_path = os.path.join(
                self.watch_directory, 
                'drug_docs.txt'
            )
        
        if not os.path.exists(txt_file_path):
            logger.warning(f"Drug documents file not found: {txt_file_path}")
            # Try scanning for all documents instead
            self.scan_and_load_all_documents()
            return
        
        # Load documents from txt file
        documents = self.load_documents_from_txt(txt_file_path)
        
        if documents:
            # Add documents to the vector store
            self.add_documents(documents)
            # Update registry
            file_hash = self.get_document_hash(txt_file_path)
            self.document_registry[txt_file_path] = file_hash
            logger.info(f"Successfully initialized drug retriever with {len(documents)} documents from txt file")
        else:
            logger.warning("No documents loaded from txt file, trying full scan")
            self.scan_and_load_all_documents()
    
    def parse_drug_document(self, content: str) -> Dict[str, str]:
        """Parse drug document to extract structured information"""
        lines = content.split('\n')
        parsed = {
            'drug_name': '',
            'target': '',
            'description': '',
            'category': '',
            'source': '',
            'recommendation': ''
        }
        
        for line in lines:
            line = line.strip()
            if line.startswith('Loai Thuoc:'):
                parsed['drug_name'] = line.replace('Loai Thuoc:', '').strip()
            elif line.startswith('Doi Tuong:'):
                parsed['target'] = line.replace('Doi Tuong:', '').strip()
            elif line.startswith('Mo ta:'):
                parsed['description'] = line.replace('Mo ta:', '').strip()
            elif line.startswith('Phan Loai:'):
                parsed['category'] = line.replace('Phan Loai:', '').strip()
            elif line.startswith('Nguon:'):
                parsed['source'] = line.replace('Nguon:', '').strip()
            elif line.startswith('Khuyen cao:'):
                parsed['recommendation'] = line.replace('Khuyen cao:', '').strip()
        
        return parsed

    def _run(self, query: str) -> List[str]:
        """Synchronous document retrieval for LangChain compatibility."""
        return self.retrieve_documents(query)
    
    async def _arun(self, query: str) -> List[str]:
        """Asynchronous document retrieval for LangChain compatibility."""
        # For simplicity, we're using the synchronous version
        return self.retrieve_documents(query)
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store with enhanced BM25 handling."""
        if not self.is_initialized:
            logger.error("Cannot add documents: Retriever not initialized.")
            return
        
        try:
            # Add to vector store
            self.vector_store.add_documents(documents)
            
            # Reinitialize BM25 with all documents for better performance
            if self.use_bm25:
                try:
                    # Get all documents for BM25 reinitialization
                    all_docs = self.vector_store.get()
                    if all_docs and 'documents' in all_docs and all_docs['documents']:
                        docs_for_bm25 = [
                            Document(
                                page_content=doc_text, 
                                metadata=all_docs.get('metadatas', [{}])[i] if i < len(all_docs.get('metadatas', [])) else {}
                            )
                            for i, doc_text in enumerate(all_docs['documents'])
                        ]
                        self.bm25_retriever = BM25Retriever.from_documents(docs_for_bm25)
                        logger.info(f"BM25 retriever reinitialized with {len(docs_for_bm25)} documents")
                except Exception as e:
                    logger.warning(f"BM25 reinitialization failed: {e}")
                
            logger.info(f"Added {len(documents)} documents to drug collection")
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")

    def run(self, query: str, state: Optional[Dict[str, Any]] = None) -> List[str]:
        """Main run method for the retrieval tool."""
        return self.retrieve_documents(query)
    
    async def arun(self, query: str, state: Optional[Dict[str, Any]] = None) -> List[str]:
        """Async run method (using sync implementation for simplicity)."""
        return self.retrieve_documents(query)
    
    def retrieve_documents(self, query: str) -> List[str]:
        """Enhanced document retrieval with advanced reranking for better context."""
        if not hasattr(self, 'vector_store') or self.vector_store is None:
            logger.warning("Vector store not available. Cannot retrieve documents.")
            return [f"Error: {self.collection_type.capitalize()} document retrieval is unavailable."]
        
        try:
            time_start = time.time()
            
            # Use enhanced retrieval with reranking
            reranked_docs = self.get_relevant_documents_with_reranking(query, k=4)
            
            if not reranked_docs:
                logger.warning(f"No relevant {self.collection_type} documents found for query: '{query}'")
                return [f"No relevant {self.collection_type} information found for your query."]
            
            # Format results with enhanced relevance information
            formatted_results = []
            
            for i, doc in enumerate(reranked_docs):
                # Enhanced relevance indicators
                if doc.relevance_score > 0.8:
                    relevance_indicator = "🟢 HIGH"
                    doc_type = "Highly Relevant"
                elif doc.relevance_score > 0.6:
                    relevance_indicator = "🟡 MEDIUM"
                    doc_type = "Moderately Relevant"
                else:
                    relevance_indicator = "🔵 LOW"
                    doc_type = "Potentially Relevant"
                
                # Add drug name if available
                drug_info = f" [{doc.drug_name}]" if doc.drug_name else ""
                category_info = f" ({doc.category})" if doc.category else ""
                
                formatted_result = (
                    f"{relevance_indicator} {doc_type}{drug_info}{category_info}: "
                    f"{doc.content} "
                    f"(Score: {doc.relevance_score:.2f}, Source: {doc.source})"
                )
                formatted_results.append(formatted_result)
            
            logger.info(f"Enhanced document retrieval completed in {time.time() - time_start:.2f} seconds with {len(formatted_results)} reranked results.")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error during enhanced document retrieval: {e}", exc_info=True)
            return [f"Error retrieving documents: {e}"]

    async def aretrieve_documents(self, query: str) -> List[str]:
        """Async version with enhanced reranking for better context retrieval."""
        if not hasattr(self, 'vector_store') or self.vector_store is None:
            logger.warning("Vector store not available. Cannot retrieve documents.")
            return [f"Error: {self.collection_type.capitalize()} document retrieval is unavailable."]
        
        try:
            time_start = time.time()
        
            reranked_docs = self.get_relevant_documents_with_reranking(query, k=8)
            
            if not reranked_docs:
                logger.warning(f"No relevant {self.collection_type} documents found for async query: '{query}'")
                return [f"No relevant {self.collection_type} information found for your query."]
            
            # Format results with enhanced relevance information
            formatted_results = []
            
            for i, doc in enumerate(reranked_docs):
                # Enhanced relevance indicators
                if doc.relevance_score > 0.8:
                    relevance_indicator = "🟢 HIGH"
                    doc_type = "Highly Relevant"
                elif doc.relevance_score > 0.6:
                    relevance_indicator = "🟡 MEDIUM" 
                    doc_type = "Moderately Relevant"
                else:
                    relevance_indicator = "🔵 LOW"
                    doc_type = "Potentially Relevant"
                
                # Add drug name if available
                drug_info = f" [{doc.drug_name}]" if doc.drug_name else ""
                category_info = f" ({doc.category})" if doc.category else ""
                
                formatted_result = (
                    f"{relevance_indicator} {doc_type}{drug_info}{category_info}: "
                    f"{doc.content} "
                    f"(Score: {doc.relevance_score:.2f}, Source: {doc.source})"
                )
                formatted_results.append(formatted_result)
            
            logger.info(f"Async enhanced document retrieval completed in {time.time() - time_start:.2f} seconds with {len(formatted_results)} reranked results.")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error during async enhanced document retrieval: {e}", exc_info=True)
            return [f"Error retrieving documents: {e}"]
            
    def rerank_documents(self, documents: List[RetrievedDocument], query: str) -> List[RetrievedDocument]:
        """Enhanced reranking with multiple scoring factors"""
        if not documents:
            return documents
        
        # Parse documents for structured information
        for doc in documents:
            parsed = self.parse_drug_document(doc.content)
            doc.drug_name = parsed['drug_name']
            doc.category = parsed['category']
            
            # Calculate advanced relevance score
            doc.calculate_advanced_relevance(query)
        
        # Sort by relevance score (highest first)
        reranked = sorted(documents, key=lambda x: x.relevance_score, reverse=True)
        
        # Apply diversity filtering to avoid too many similar results
        filtered_results = []
        seen_drugs = set()
        
        for doc in reranked:
            # Add document if it's about a new drug or has very high relevance
            if doc.drug_name not in seen_drugs or doc.relevance_score > 0.8:
                filtered_results.append(doc)
                if doc.drug_name:
                    seen_drugs.add(doc.drug_name)
                
                # Limit to top 8 results for better quality
                if len(filtered_results) >= 8:
                    break
        
        logger.info(f"Reranked {len(documents)} documents to {len(filtered_results)} high-quality results")
        return filtered_results

    def get_relevant_documents_with_reranking(self, query: str, k: int = 10) -> List[RetrievedDocument]:
        """Get relevant documents with enhanced reranking"""
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
                    score = 0.95 - (i * 0.02)  # More gradual score decrease
                    parsed = self.parse_drug_document(doc.page_content)
                    
                    retrieved_docs.append(
                        RetrievedDocument(
                            content=doc.page_content,
                            source=doc.metadata.get('source', 'BM25'),
                            retrieval_score=score,
                            drug_name=parsed.get('drug_name', ''),
                            category=parsed.get('category', '')
                        )
                    )
            
            # Vector search
            vector_docs = self.vector_store.similarity_search_with_score(query, k=initial_k)
            for doc, score in vector_docs:
                # Skip duplicates
                if any(d.content == doc.page_content for d in retrieved_docs):
                    continue
                
                # Normalize score (ChromaDB returns distance, lower is better)
                normalized_score = max(0.0, 1.0 - min(score, 2.0) / 2.0)
                parsed = self.parse_drug_document(doc.page_content)
                
                retrieved_docs.append(
                    RetrievedDocument(
                        content=doc.page_content,
                        source=doc.metadata.get('source', 'Vector'),
                        retrieval_score=normalized_score,
                        drug_name=parsed.get('drug_name', ''),
                        category=parsed.get('category', '')
                    )
                )
            
            # Step 2: Apply enhanced reranking
            reranked_docs = self.rerank_documents(retrieved_docs, query)
            
            # Step 3: Return top k results
            return reranked_docs[:k]
            
        except Exception as e:
            logger.error(f"Error in enhanced document retrieval: {e}")
            return []

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the loaded documents"""
        try:
            if not self.vector_store:
                return {"error": "Vector store not initialized"}
            
            all_docs = self.vector_store.get()
            if not all_docs or not all_docs.get('documents'):
                return {"total_documents": 0, "unique_drugs": 0}
            
            documents = all_docs['documents']
            metadatas = all_docs.get('metadatas', [])
            
            # Count unique drugs and categories
            unique_drugs = set()
            unique_categories = set()
            file_types = {}
            
            for i, doc in enumerate(documents):
                parsed = self.parse_drug_document(doc)
                if parsed.get('drug_name'):
                    unique_drugs.add(parsed['drug_name'])
                if parsed.get('category'):
                    unique_categories.add(parsed['category'])
                
                # Count file types from metadata
                metadata = metadatas[i] if i < len(metadatas) else {}
                source = metadata.get('source', 'unknown')
                file_type = 'txt'
                if ':' in source:
                    file_type = source.split(':')[0].lower()
                file_types[file_type] = file_types.get(file_type, 0) + 1
            
            return {
                "total_documents": len(documents),
                "unique_drugs": len(unique_drugs),
                "drug_names": list(unique_drugs)[:10],  # Show first 10
                "categories": list(unique_categories),
                "file_type_distribution": file_types,
                "supported_formats": self.supported_formats,
                "auto_scan_enabled": self.auto_scan_enabled,
                "watch_directory": self.watch_directory,
                "documents_in_registry": len(self.document_registry)
            }
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {"error": str(e)}
    
    def search_by_drug_name(self, drug_name: str, k: int = 5) -> List[str]:
        """Search specifically by drug name for targeted results"""
        query = f"Loai Thuoc: {drug_name}"
        return self.retrieve_documents(query)
    
    def get_drug_categories(self) -> List[str]:
        """Get all available drug categories"""
        try:
            if not self.vector_store:
                return []
            
            all_docs = self.vector_store.get()
            if not all_docs or not all_docs.get('documents'):
                return []
            
            categories = set()
            for doc in all_docs['documents']:
                parsed = self.parse_drug_document(doc)
                if parsed.get('category'):
                    categories.add(parsed['category'])
            
            return list(categories)
            
        except Exception as e:
            logger.error(f"Error getting categories: {e}")
            return []
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported document formats"""
        return self.supported_formats.copy()
    
    def set_watch_directory(self, new_directory: str):
        """Change the watch directory and restart monitoring"""
        # Stop current watcher
        self.stop_document_watcher()
        
        # Update directory
        self.watch_directory = new_directory
        
        # Restart watcher if auto-scan is enabled
        if self.auto_scan_enabled:
            self._start_document_watcher()
        
        logger.info(f"Watch directory changed to: {new_directory}")
    
    def enable_auto_scan(self):
        """Enable automatic document scanning"""
        if not self.auto_scan_enabled:
            self.auto_scan_enabled = True
            self._start_document_watcher()
            logger.info("Auto-scan enabled")
    
    def disable_auto_scan(self):
        """Disable automatic document scanning"""
        if self.auto_scan_enabled:
            self.auto_scan_enabled = False
            self.stop_document_watcher()
            logger.info("Auto-scan disabled")
    
    def manual_scan(self):
        """Manually trigger a scan for new documents"""
        logger.info("Starting manual document scan...")
        return self.scan_for_new_documents()
    
    def get_document_registry(self) -> Dict[str, str]:
        """Get the current document registry (file paths and hashes)"""
        return self.document_registry.copy()
    
    def clear_document_registry(self):
        """Clear the document registry (will cause all documents to be reloaded on next scan)"""
        self.document_registry.clear()
        logger.info("Document registry cleared")
    
    def force_reload_all(self):
        """Force reload all documents from the watch directory"""
        self.clear_document_registry()
        self.scan_and_load_all_documents()
        
    def get_load_status(self) -> Dict[str, Any]:
        """Get current loading status and statistics"""
        try:
            vector_docs = 0
            if self.vector_store:
                all_docs = self.vector_store.get()
                if all_docs and all_docs.get('documents'):
                    vector_docs = len(all_docs['documents'])
            
            return {
                "is_initialized": self.is_initialized,
                "auto_scan_enabled": self.auto_scan_enabled,
                "watch_directory": self.watch_directory,
                "supported_formats": self.supported_formats,
                "documents_in_registry": len(self.document_registry),
                "documents_in_vector_store": vector_docs,
                "watcher_active": self.observer is not None and self.observer.is_alive() if self.observer else False
            }
        except Exception as e:
            logger.error(f"Error getting load status: {e}")
            return {"error": str(e)}


# if __name__ == "__main__":
#     # Example usage
#     retriever = DrugRetrieverTool(
#         collection_name="drug_docs",
#         watch_directory="app/agents/retrievers/storages/drug",
#         use_bm25=True,
#         auto_scan_enabled=True
#     )
    
#     # Initialize from txt file
#     retriever.initialize_from_txt("app/agents/retrievers/storages/drug/drug_docs.txt")
    
#     # Perform a sample retrieval
#     query_result = retriever.run(query="What are the symptoms of diabetes?")
#     print(query_result)
    
#     # Cleanup resources
#     retriever.cleanup()