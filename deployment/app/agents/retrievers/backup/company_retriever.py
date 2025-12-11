# --- Enhanced Company Retriever Tool ---
import os
import sys
import time
import json
import csv
import hashlib
import threading
from datetime import datetime
from typing import Optional, List, Dict, Any
from loguru import logger
from pydantic import Field, BaseModel
import numpy as np
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# --- Document Processing Imports ---
import pandas as pd
from pypdf import PdfReader
from langchain_community.document_loaders import TextLoader, CSVLoader, PyPDFLoader, JSONLoader

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
            

class RetrievedCompanyDocument(BaseModel):
    """Model for a retrieved company document with enhanced metadata for reranking"""
    content: str
    source: str
    retrieval_score: float
    relevance_score: float = 0.0
    topic: str = ""
    category: str = ""
    priority_level: str = ""
    
    def calculate_advanced_relevance(self, query: str) -> float:
        """Enhanced relevance scoring with multiple factors for company content"""
        query_lower = query.lower()
        content_lower = self.content.lower()
        
        # 1. Exact phrase matching (highest weight)
        exact_match_score = 0.0
        if query_lower in content_lower:
            exact_match_score = 1.0
        
        # 2. Topic relevance matching
        topic_match_score = 0.0
        query_terms = set(query_lower.split())
        if self.topic and any(term in self.topic.lower() for term in query_terms):
            topic_match_score = 0.8
        
        # 3. Category relevance (business categories)
        category_match_score = 0.0
        if self.category:
            category_lower = self.category.lower()
            for term in query_terms:
                if term in category_lower:
                    category_match_score = 0.6
                    break
        
        # 4. Business priority scoring
        priority_score = 0.0
        if 'contact' in query_lower or 'liên hệ' in query_lower:
            if 'địa chỉ' in self.topic.lower() or 'liên hệ' in self.topic.lower():
                priority_score = 0.9
        elif 'price' in query_lower or 'giá' in query_lower or 'cost' in query_lower:
            if 'sản phẩm' in self.topic.lower() or 'dịch vụ' in self.topic.lower():
                priority_score = 0.9
        elif 'account' in query_lower or 'tài khoản' in query_lower:
            if 'tài khoản' in self.topic.lower():
                priority_score = 0.9
        
        # 5. Term overlap with position weighting
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
        
        # 6. Content length penalty (prefer concise, relevant content)
        length_penalty = 1.0 - min(len(self.content) / 2000, 0.3)
        
        # Combine all scores with weights
        self.relevance_score = (
            exact_match_score * 0.25 +
            topic_match_score * 0.25 +
            category_match_score * 0.15 +
            priority_score * 0.15 +
            overlap_score * 0.1 +
            self.retrieval_score * 0.05 +
            length_penalty * 0.05
        )
        return self.relevance_score

class CompanyRetrieverTool(BaseAgentTool):
    """Enhanced company retrieval tool with multi-format support and automatic document scanning"""
    # Define Pydantic fields
    collection_name: str = Field(..., description="Collection name for the retriever")
    use_bm25: bool = Field(default=True, description="Whether to use BM25 retriever")
    collection_type: str = Field(default="company", description="Type of collection")
    is_initialized: bool = Field(default=False, description="Initialization status")
    vector_store: Any = Field(default=None, description="Vector store instance")
    bm25_retriever: Any = Field(default=None, description="BM25 retriever instance")
    embeddings: Any = Field(default=None, description="Embeddings model")
    
    # New fields for enhanced functionality
    watch_directory: str = Field(default="", description="Directory to watch for new documents")
    document_registry: Dict[str, str] = Field(default_factory=dict, description="Registry of loaded documents with hashes")
    observer: Any = Field(default=None, description="File system observer")
    text_splitter: Any = Field(default=None, description="Text splitter for large documents")
    auto_scan_enabled: bool = Field(default=True, description="Whether automatic scanning is enabled")
    supported_formats: List[str] = Field(default_factory=lambda: ['.txt', '.pdf', '.csv', '.json'], description="Supported document formats")
    files_loaded: List[str] = Field(default_factory=list, description="List of loaded document files")
    
    
    
    def __init__(self, collection_name: str = "company_data", use_bm25: bool = True, collection_type: str = "company", 
                 watch_directory: str = None, auto_scan_enabled: bool = True, **kwargs):
        # Sanitize collection name for ChromaDB compatibility
        safe_collection_name = collection_name.replace(" ", "_").replace("-", "_")
        if len(safe_collection_name) > 63:
            safe_collection_name = safe_collection_name[:63]  # ChromaDB has a length limit
        
        # Set default watch directory
        if not watch_directory:
            watch_directory = os.path.join(os.path.dirname(__file__), 'data_raw')
        
        # Initialize tool with proper name and description
        name = f"company_retriever"
        description = f"Retrieve GeneStory company information with multi-format support and auto-scanning."
        
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
            chunk_overlap=200,
            length_function=len,
        )
        
        # Initialize the tool components
        self._initialize_components()
        self.load_document_registry()
        
        # Auto-load documents from multiple formats
        # self.auto_load_documents()

        # Start document watching if enabled
        if self.auto_scan_enabled:
            logger.info(f"Auto-scanning enabled, starting document watcher for directory: {self.watch_directory}")
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
            logger.info(f"Company retriever initialized successfully with collection '{self.collection_name}'")
            
        except Exception as e:
            logger.error(f"Failed to initialize company retriever: {e}")
            self.is_initialized = False
            self.vector_store = None
            self.bm25_retriever = None
    
    def auto_load_documents(self):
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
                logger.info(f"Files loaded from watch directory: {self.files_loaded}")
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
            if not os.path.exists(f"{self.watch_directory}/registry"):
                logger.info(f"Creating registry directory: {self.watch_directory}/registry")
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
                        parsed = self.parse_company_document(chunk)
                        
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
                    parsed = self.parse_company_document(content)
                    
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
                        parsed = self.parse_company_document(content)
                        
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
                    parsed = self.parse_company_document(content)
                    
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
        
        new_docs_count = 0
        for root, dirs, files in os.walk(self.watch_directory):
            for file in files:
                file_path = os.path.join(root, file)
                _, ext = os.path.splitext(file.lower())
                
                if ext in self.supported_formats:
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

    
    def load_new_document(self, file_path: str):
        """Load a single new document (called by file watcher)"""
        file_hash = self.get_document_hash(file_path)
        filename = os.path.basename(file_path)
        try:
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
    
    def parse_company_document(self, content: str) -> Dict[str, str]:
        """Parse company document to extract structured information"""
        lines = content.split('\n')
        parsed = {
            'topic': '',
            'summary': '',
            'description': '',
            'category': '',
            'embedded_link': '',
            'priority_level': ''
        }
        
        current_section = ''
        for line in lines:
            line = line.strip()
            if line.startswith('Topic:') or line.startswith('opic:'):  # Handle potential OCR error
                parsed['topic'] = line.replace('Topic:', '').replace('opic:', '').strip()
                current_section = 'topic'
            elif line.startswith('Summary:'):
                parsed['summary'] = line.replace('Summary:', '').strip()
                current_section = 'summary'
            elif line.startswith('Description:'):
                parsed['description'] = line.replace('Description:', '').strip()
                current_section = 'description'
            elif line.startswith('Embedded Link:'):
                parsed['embedded_link'] = line.replace('Embedded Link:', '').strip()
                current_section = 'link'
            elif current_section == 'description' and line and not line.startswith('Embedded Link:'):
                # Continue adding to description if we're in that section
                if parsed['description']:
                    parsed['description'] += ' ' + line
                else:
                    parsed['description'] = line
        
        # Determine category based on topic
        topic_lower = parsed['topic'].lower()
        if 'địa chỉ' in topic_lower or 'liên hệ' in topic_lower:
            parsed['category'] = 'Contact Information'
            parsed['priority_level'] = 'high'
        elif 'tài khoản' in topic_lower:
            parsed['category'] = 'Account Management'
            parsed['priority_level'] = 'high'
        elif 'sản phẩm' in topic_lower or 'dịch vụ' in topic_lower:
            parsed['category'] = 'Products & Services'
            parsed['priority_level'] = 'high'
        elif 'nhân sự' in topic_lower:
            parsed['category'] = 'Personnel & Leadership'
            parsed['priority_level'] = 'medium'
        elif 'dự án' in topic_lower or 'nghiên cứu' in topic_lower:
            parsed['category'] = 'Research & Projects'
            parsed['priority_level'] = 'medium'
        elif 'thanh toán' in topic_lower:
            parsed['category'] = 'Payment & Billing'
            parsed['priority_level'] = 'high'
        elif 'bảo mật' in topic_lower:
            parsed['category'] = 'Security & Privacy'
            parsed['priority_level'] = 'medium'
        elif 'giới thiệu' in topic_lower:
            parsed['category'] = 'Company Information'
            parsed['priority_level'] = 'medium'
        else:
            parsed['category'] = 'General Information'
            parsed['priority_level'] = 'low'
        
        return parsed

    def rerank_documents(self, documents: List[RetrievedCompanyDocument], query: str) -> List[RetrievedCompanyDocument]:
        """Enhanced reranking with multiple scoring factors for company content"""
        if not documents:
            return documents
        
        # Parse documents for structured information
        for doc in documents:
            parsed = self.parse_company_document(doc.content)
            doc.topic = parsed['topic']
            doc.category = parsed['category']
            doc.priority_level = parsed['priority_level']
            
            # Calculate advanced relevance score
            doc.calculate_advanced_relevance(query)
        
        # Sort by relevance score (highest first)
        reranked = sorted(documents, key=lambda x: x.relevance_score, reverse=True)
        
        # Apply diversity filtering to avoid too many similar results
        filtered_results = []
        seen_topics = set()
        
        for doc in reranked:
            # Add document if it's about a new topic or has very high relevance
            if doc.topic not in seen_topics or doc.relevance_score > 0.8:
                filtered_results.append(doc)
                seen_topics.add(doc.topic)
                
                # Limit to top 10 results for better quality
                if len(filtered_results) >= 10:
                    break
        
        logger.info(f"Reranked {len(documents)} documents to {len(filtered_results)} high-quality results")
        return filtered_results

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
                    parsed = self.parse_company_document(entry)
                    
                    doc = Document(
                        page_content=entry.strip(),
                        metadata={
                            'source': 'company_docs.txt',
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

    def initialize_from_txt(self, txt_file_path: str = None):
        """Initialize the retriever with documents from txt file (legacy method)"""
        if not txt_file_path:
            # Default path to company_docs.txt
            txt_file_path = os.path.join(
                self.watch_directory, 
                'company_docs.txt'
            )
        
        if not os.path.exists(txt_file_path):
            logger.warning(f"Company documents file not found: {txt_file_path}")
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
            logger.info(f"Successfully initialized company retriever with {len(documents)} documents from txt file")
        else:
            logger.warning("No documents loaded from txt file, trying full scan")
            self.scan_and_load_all_documents()

    def get_relevant_documents_with_reranking(self, query: str, k: int = 10) -> List[RetrievedCompanyDocument]:
        """Get relevant documents with enhanced reranking for company content"""
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
                    parsed = self.parse_company_document(doc.page_content)
                    
                    retrieved_docs.append(
                        RetrievedCompanyDocument(
                            content=doc.page_content,
                            source=doc.metadata.get('source', 'BM25'),
                            retrieval_score=score,
                            topic=parsed.get('topic', ''),
                            category=parsed.get('category', ''),
                            priority_level=parsed.get('priority_level', '')
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
                parsed = self.parse_company_document(doc.page_content)
                
                retrieved_docs.append(
                    RetrievedCompanyDocument(
                        content=doc.page_content,
                        source=doc.metadata.get('source', 'Vector'),
                        retrieval_score=normalized_score,
                        topic=parsed.get('topic', ''),
                        category=parsed.get('category', ''),
                        priority_level=parsed.get('priority_level', '')
                    )
                )
            
            # Step 2: Apply enhanced reranking
            reranked_docs = self.rerank_documents(retrieved_docs, query)
            
            # Step 3: Return top k results
            return reranked_docs[:k]
            
        except Exception as e:
            logger.error(f"Error in enhanced document retrieval: {e}")
            return []

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
                        all_documents = [
                            Document(
                                page_content=doc_text, 
                                metadata=all_docs.get('metadatas', [{}])[i] if i < len(all_docs.get('metadatas', [])) else {}
                            )
                            for i, doc_text in enumerate(all_docs['documents'])
                        ]
                        
                        self.bm25_retriever = BM25Retriever.from_documents(all_documents)
                        logger.info(f"Reinitialized BM25 with {len(all_documents)} total documents")
                except Exception as e:
                    logger.warning(f"BM25 reinitialization failed: {e}")
                
            logger.info(f"Added {len(documents)} documents to company collection")
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")

    def run(self, query: str, state: Optional[Dict[str, Any]] = None) -> List[str]:
        """Main run method for the retrieval tool."""
        return self.retrieve_documents(query)
    
    async def arun(self, query: str, state: Optional[Dict[str, Any]] = None) -> List[str]:
        """Async run method (using sync implementation for simplicity)."""
        return self.retrieve_documents(query)
    
    def retrieve_documents(self, query: str) -> List[str]:
        """Enhanced document retrieval with advanced reranking for better company context."""
        if not hasattr(self, 'vector_store') or self.vector_store is None:
            logger.warning("Vector store not available. Cannot retrieve documents.")
            return [f"Error: {self.collection_type.capitalize()} document retrieval is unavailable."]
        
        try:
            time_start = time.time()
            
            # Use enhanced retrieval with reranking
            reranked_docs = self.get_relevant_documents_with_reranking(query, k=8)
            
            if not reranked_docs:
                logger.warning(f"No relevant {self.collection_type} documents found for query: '{query}'")
                return [f"No relevant {self.collection_type} information found for your query."]
            
            # Format results with enhanced relevance information
            formatted_results = []
            
            for i, doc in enumerate(reranked_docs):
                # Enhanced relevance indicators
                if doc.relevance_score > 0.8:
                    relevance_indicator = "🔥 URGENT"
                    doc_type = "Critical Information"
                elif doc.relevance_score > 0.6:
                    relevance_indicator = "⭐ HIGH"
                    doc_type = "Important Information"
                else:
                    relevance_indicator = "📝 INFO"
                    doc_type = "General Information"
                
                # Add topic if available
                topic_info = f" [{doc.topic}]" if doc.topic else ""
                category_info = f" ({doc.category})" if doc.category else ""
                priority_info = f" - {doc.priority_level} priority" if doc.priority_level else ""
                
                formatted_result = (
                    f"{relevance_indicator} {doc_type}{topic_info}{category_info}{priority_info}: "
                    f"{doc.content} "
                    f"(Score: {doc.relevance_score:.2f}, Source: {doc.source})"
                )
                formatted_results.append(formatted_result)
            
            logger.info(f"Enhanced company document retrieval completed in {time.time() - time_start:.2f} seconds with {len(formatted_results)} reranked results.")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error during enhanced document retrieval: {e}", exc_info=True)
            return [f"Error retrieving documents: {e}"]

    async def aretrieve_documents(self, query: str) -> List[str]:
        """Async version with enhanced reranking for better company context retrieval."""
        if not hasattr(self, 'vector_store') or self.vector_store is None:
            logger.warning("Vector store not available. Cannot retrieve documents.")
            return [f"Error: {self.collection_type.capitalize()} document retrieval is unavailable."]
        
        try:
            time_start = time.time()
            
            # For async, we'll use the synchronous enhanced reranking for now
            # as most vector stores don't have full async support for complex operations
            reranked_docs = self.get_relevant_documents_with_reranking(query, k=8)
            
            if not reranked_docs:
                logger.warning(f"No relevant {self.collection_type} documents found for async query: '{query}'")
                return [f"No relevant {self.collection_type} information found for your query."]
            
            # Format results with enhanced relevance information
            formatted_results = []
            
            for i, doc in enumerate(reranked_docs):
                # Enhanced relevance indicators
                if doc.relevance_score > 0.8:
                    relevance_indicator = "🔥 URGENT"
                    doc_type = "Critical Information"
                elif doc.relevance_score > 0.6:
                    relevance_indicator = "⭐ HIGH" 
                    doc_type = "Important Information"
                else:
                    relevance_indicator = "📝 INFO"
                    doc_type = "General Information"
                
                # Add topic if available
                topic_info = f" [{doc.topic}]" if doc.topic else ""
                category_info = f" ({doc.category})" if doc.category else ""
                priority_info = f" - {doc.priority_level} priority" if doc.priority_level else ""
                
                formatted_result = (
                    f"{relevance_indicator} {doc_type}{topic_info}{category_info}{priority_info}: "
                    f"{doc.content} "
                    f"(Score: {doc.relevance_score:.2f}, Source: {doc.source})"
                )
                formatted_results.append(formatted_result)
            
            logger.info(f"Async enhanced company document retrieval completed in {time.time() - time_start:.2f} seconds with {len(formatted_results)} reranked results.")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error during async enhanced document retrieval: {e}", exc_info=True)
            return [f"Error retrieving documents: {e}"]

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the loaded company documents"""
        try:
            if not self.vector_store:
                return {"error": "Vector store not initialized"}
            
            all_docs = self.vector_store.get()
            if not all_docs or not all_docs.get('documents'):
                return {"total_documents": 0, "unique_topics": 0}
            
            documents = all_docs['documents']
            metadatas = all_docs.get('metadatas', [])
            
            # Count unique topics and categories
            unique_topics = set()
            unique_categories = set()
            priority_levels = {'high': 0, 'medium': 0, 'low': 0}
            file_types = {}
            
            for i, doc in enumerate(documents):
                parsed = self.parse_company_document(doc)
                if parsed.get('topic'):
                    unique_topics.add(parsed['topic'])
                if parsed.get('category'):
                    unique_categories.add(parsed['category'])
                if parsed.get('priority_level'):
                    priority_level = parsed['priority_level']
                    if priority_level in priority_levels:
                        priority_levels[priority_level] += 1
                
                # Count file types from metadata
                metadata = metadatas[i] if i < len(metadatas) else {}
                file_type = metadata.get('file_type', 'unknown')
                file_types[file_type] = file_types.get(file_type, 0) + 1
            
            return {
                "total_documents": len(documents),
                "unique_topics": len(unique_topics),
                "unique_categories": len(unique_categories),
                "topic_names": list(unique_topics)[:10],  # Show first 10
                "categories": list(unique_categories),
                "priority_distribution": priority_levels,
                "file_type_distribution": file_types,
                "supported_formats": self.supported_formats,
                "auto_scan_enabled": self.auto_scan_enabled,
                "watch_directory": self.watch_directory,
                "documents_in_registry": len(self.document_registry)
            }
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {"error": str(e)}
    
    def search_by_topic(self, topic: str, k: int = 5) -> List[str]:
        """Search specifically by company topic for targeted results"""
        query = f"Topic: {topic}"
        return self.retrieve_documents(query)
    
    def search_by_category(self, category: str, k: int = 5) -> List[str]:
        """Search by company category for related information"""
        query = f"Category: {category}"
        return self.retrieve_documents(query)
    
    def get_company_categories(self) -> List[str]:
        """Get all available company information categories"""
        try:
            if not self.vector_store:
                return []
            
            all_docs = self.vector_store.get()
            if not all_docs or not all_docs.get('documents'):
                return []
            
            categories = set()
            for doc in all_docs['documents']:
                parsed = self.parse_company_document(doc)
                if parsed.get('category'):
                    categories.add(parsed['category'])
            
            return list(categories)
            
        except Exception as e:
            logger.error(f"Error getting categories: {e}")
            return []
    
    def get_high_priority_topics(self) -> List[str]:
        """Get topics with high priority levels"""
        try:
            if not self.vector_store:
                return []
            
            all_docs = self.vector_store.get()
            if not all_docs or not all_docs.get('documents'):
                return []
            
            high_priority_topics = []
            for doc in all_docs['documents']:
                parsed = self.parse_company_document(doc)
                if parsed.get('priority_level') == 'high' and parsed.get('topic'):
                    high_priority_topics.append(parsed['topic'])
            
            return list(set(high_priority_topics))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Error getting high priority topics: {e}")
            return []
    
    def get_contact_information(self) -> List[str]:
        """Get contact and address information specifically"""
        query = "địa chỉ liên hệ contact address phone email"
        return self.retrieve_documents(query)
    
    def get_product_information(self) -> List[str]:
        """Get product and service information specifically"""
        query = "sản phẩm dịch vụ giá product service price GeneMap"
        return self.retrieve_documents(query)

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
        self.scan_for_new_documents()
    
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
    
    def __del__(self):
        """Cleanup method to stop file watcher when object is destroyed"""
        try:
            self.stop_document_watcher()
        except:
            pass
    
    def cleanup(self):
        """Explicit cleanup method for proper resource management"""
        self.stop_document_watcher()
        logger.info("CompanyRetrieverTool cleanup completed")


if __name__ == "__main__":
    # Example usage
    retriever = CompanyRetrieverTool(
        watch_directory="company_docs",
        use_bm25=True,
        auto_scan_enabled=True
    )
    
    # Initialize from txt file
    retriever.initialize_from_txt("company_docs/company_docs.txt")
    
    # Perform a sample retrieval
    query_result = retriever.run(query="What are the symptoms of diabetes?")
    print(query_result)
    
    # Cleanup resources
    retriever.cleanup()