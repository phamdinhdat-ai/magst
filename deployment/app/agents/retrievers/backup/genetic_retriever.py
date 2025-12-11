# --- Enhanced Genetic Retriever Tool ---
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
from app.agents.tools.base import BaseAgentTool
from app.core.config import get_settings

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
                logger.info(f"New genetic document detected: {file_path}")
                self.retriever_tool.load_new_document(file_path)
    
    def on_modified(self, event):
        """Handle file modification events"""
        if not event.is_directory:
            file_path = event.src_path
            if any(file_path.endswith(ext) for ext in self.supported_extensions) and "registry" not in file_path:
                logger.info(f"Genetic document modified: {file_path}")
                self.retriever_tool.reload_document(file_path)
    
    def _is_supported_file(self, file_path: str) -> bool:
        """Check if file is supported for processing"""
        return any(file_path.lower().endswith(ext) for ext in self.supported_extensions)




class RetrievedGeneticDocument(BaseModel):
    """Model for a retrieved genetic document with enhanced metadata for reranking"""
    content: str
    source: str
    retrieval_score: float
    relevance_score: float = 0.0
    topic: str = ""
    intelligence_type: str = ""
    research_area: str = ""
    category: str = ""
    
    def calculate_advanced_relevance(self, query: str) -> float:
        """Enhanced relevance scoring with multiple factors for genetic content"""
        query_lower = query.lower()
        content_lower = self.content.lower()
        
        # 1. Exact phrase matching (highest weight)
        exact_match_score = 0.0
        if query_lower in content_lower:
            exact_match_score = 1.0
        
        # 2. Intelligence type relevance matching
        intelligence_match_score = 0.0
        intelligence_keywords = ['trí thông minh', 'intelligence', 'iq', 'eq', 'thông minh ngôn ngữ', 
                               'thông minh logic', 'thông minh âm nhạc', 'thông minh không gian']
        if any(keyword in query_lower for keyword in intelligence_keywords):
            if self.intelligence_type or any(keyword in content_lower for keyword in intelligence_keywords):
                intelligence_match_score = 0.9
        
        # 3. Genetic analysis relevance
        genetic_match_score = 0.0
        genetic_keywords = ['gen', 'gene', 'di truyền', 'genetic', 'dna', 'giải mã gen', 'genestory']
        if any(keyword in query_lower for keyword in genetic_keywords):
            if any(keyword in content_lower for keyword in genetic_keywords):
                genetic_match_score = 0.8
        
        # 4. Research and technical relevance
        research_match_score = 0.0
        research_keywords = ['nghiên cứu', 'research', 'dữ liệu', 'data', 'hệ thống', 'system', 'mash']
        if any(keyword in query_lower for keyword in research_keywords):
            if self.research_area or any(keyword in content_lower for keyword in research_keywords):
                research_match_score = 0.7
        
        # 5. Health and medical relevance
        health_match_score = 0.0
        health_keywords = ['sức khỏe', 'health', 'bệnh', 'disease', 'y tế', 'medical', 'thuốc', 'medicine']
        if any(keyword in query_lower for keyword in health_keywords):
            if any(keyword in content_lower for keyword in health_keywords):
                health_match_score = 0.6
        
        # 6. Content semantic similarity (term overlap)
        query_words = set(query_lower.split())
        content_words = set(content_lower.split())
        if query_words and content_words:
            overlap = len(query_words.intersection(content_words))
            semantic_score = min(overlap / len(query_words), 1.0) * 0.5
        else:
            semantic_score = 0.0
        
        # 7. Length penalty for very short or very long content
        length_penalty = 1.0
        if len(self.content) < 30:
            length_penalty = 0.7
        elif len(self.content) > 2000:
            length_penalty = 0.9
        
        # Combine scores with weights
        final_score = (
            exact_match_score * 0.30 +
            intelligence_match_score * 0.25 +
            genetic_match_score * 0.20 +
            research_match_score * 0.15 +
            health_match_score * 0.10 +
            semantic_score * 0.10 +
            (1.0 - exact_match_score) * 0.05  # Small boost for variety
        ) * length_penalty
        
        self.relevance_score = min(final_score, 1.0)
        return self.relevance_score


class GeneticRetrieverTool(BaseAgentTool):
    """Enhanced Genetic Retriever with dynamic data loading, file watching, and advanced reranking"""
    
    # Define Pydantic fields
    collection_name: str = Field(..., description="Collection name for the retriever")
    use_bm25: bool = Field(default=True, description="Whether to use BM25 retriever")
    collection_type: str = Field(default="genetic", description="Type of collection")
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
    name: str = Field(default="genetic_retriever", description="Tool name")
    description: str = Field(default="Retrieves information about genetics, intelligence types, and related research.", 
                            description="Tool description")
                            
    def _run(self, query: str) -> List[str]:
        """Execute the genetic retriever with the provided query"""
        return self.retrieve_documents(query)
        
    async def _arun(self, query: str) -> List[str]:
        """Execute the genetic retriever asynchronously with the provided query"""
        return self.retrieve_documents(query)  # For now, just call the sync version
    
    # Retrieval parameters
    max_results: int = Field(default=5, description="Maximum number of results to return")
    similarity_threshold: float = Field(default=0.1, description="Minimum similarity threshold")
    rerank_threshold: float = Field(default=0.3, description="Minimum relevance score after reranking")
    
    def __init__(self, 
                 collection_name: str = "genetic_data", 
                 use_bm25: bool = True, 
                 collection_type: str = "genetic",
                 watch_directory: str = None,
                 auto_scan_enabled: bool = True,
                 **kwargs):
        # Sanitize collection name for ChromaDB compatibility
        safe_collection_name = collection_name.replace(" ", "_").replace("-", "_")
        if len(safe_collection_name) > 63:
            safe_collection_name = safe_collection_name[:63]  # ChromaDB has a length limit
        
        # Set default watch directory if not provided
        if not watch_directory:
            watch_directory = os.path.join(os.path.dirname(__file__), 'storages', 'genetics')
        
        # Initialize tool with proper name and description
        name = f"genetic_retriever"
        description = f"Advanced genetic information retrieval with dynamic data loading and reranking for intelligence types, genetic analysis, and biomedical research"
        
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
            self.scan_for_new_documents()
            
    def _initialize_components(self) -> None:
        """Initialize ChromaDB, embeddings, and BM25 components"""
        try:
            # Initialize ChromaDB client
            persistent_client = chromadb.PersistentClient(path=settings.VECTOR_STORE_BASE_DIR)
            logger.info("ChromaDB client initialized successfully")
            
            # Initialize embedding model
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
            logger.info(f"Genetic retriever initialized successfully with collection '{self.collection_name}'")
            
        except Exception as e:
            logger.error(f"Failed to initialize genetic retriever: {e}")
            self.is_initialized = False
            raise
    
    def _auto_load_documents(self) -> None:
        """Automatically load documents from multiple formats if vector store is empty"""
        try:
            if hasattr(self.vector_store, '_collection'):
                current_count = self.vector_store._collection.count()
                if current_count == 0:
                    logger.info("Vector store is empty, auto-loading genetic documents...")
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
                file_hash = hashlib.md5(f.read()).hexdigest()
            return file_hash
        except Exception as e:
            logger.error(f"Error calculating hash for {file_path}: {e}")
            return ""
    
    def load_pdf_document(self, pdf_path: str) -> List[Document]:
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
                    full_text += text + "\n"
            
            if full_text.strip():
                chunks = self.text_splitter.split_text(full_text)
                for i, chunk in enumerate(chunks):
                    metadata = self.parse_genetic_document(chunk)
                    metadata['source'] = pdf_path
                    metadata['page'] = i
                    metadata['doc_type'] = 'pdf'
                    doc = Document(page_content=chunk, metadata=metadata)
                    documents.append(doc)
            
            logger.info(f"Loaded {len(documents)} chunks from PDF: {pdf_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading PDF {pdf_path}: {e}")
            return []
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

    def load_csv_document(self, csv_path: str) -> List[Document]:
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
                    'source': csv_path,
                    'row': index,
                    'doc_type': 'csv'
                }
                
                # Try to extract content and metadata from common columns
                if 'content' in df.columns:
                    content = row['content']
                elif 'text' in df.columns:
                    content = row['text']
                else:
                    # If no content column, combine all columns
                    content = ' '.join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                
                # Extract metadata from common columns
                for meta_field in ['topic', 'category', 'intelligence_type']:
                    if meta_field in df.columns:
                        metadata[meta_field] = row[meta_field] if pd.notna(row[meta_field]) else ""
                
                if content.strip():
                    # Parse for additional metadata
                    parsed_meta = self.parse_genetic_document(content)
                    metadata.update(parsed_meta)
                    
                    doc = Document(page_content=content, metadata=metadata)
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
                result = []
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        if isinstance(value, (dict, list)):
                            result.extend(extract_text_from_json(value, f"{prefix}{key}."))
                        else:
                            result.append(f"{prefix}{key}: {value}")
                elif isinstance(obj, list):
                    for i, item in enumerate(obj):
                        if isinstance(item, (dict, list)):
                            result.extend(extract_text_from_json(item, f"{prefix}[{i}]."))
                        else:
                            result.append(f"{prefix}[{i}: {item}")
                return result
            
            if isinstance(data, list):
                # Process list of objects
                for i, item in enumerate(data):
                    if isinstance(item, dict):
                        content = "\n".join(extract_text_from_json(item))
                        metadata = {
                            'source': json_path,
                            'index': i,
                            'doc_type': 'json'
                        }
                        
                        # Extract common metadata fields
                        for key in ['topic', 'category', 'intelligence_type', 'research_area']:
                            if key in item:
                                metadata[key] = item[key]
                        
                        # Parse for additional metadata
                        parsed_meta = self.parse_genetic_document(content)
                        metadata.update(parsed_meta)
                        
                        doc = Document(page_content=content, metadata=metadata)
                        documents.append(doc)
            else:
                # Process single object
                content = "\n".join(extract_text_from_json(data))
                metadata = {
                    'source': json_path,
                    'doc_type': 'json'
                }
                
                # Parse for additional metadata
                parsed_meta = self.parse_genetic_document(content)
                metadata.update(parsed_meta)
                
                doc = Document(page_content=content, metadata=metadata)
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
            for file in files:
                file_path = os.path.join(root, file)
                _, ext = os.path.splitext(file_path.lower())
                
                if ext in self.supported_formats and "registry" not in file_path:
                    # Check if file was already loaded
                    current_hash = self.get_document_hash(file_path)
                    if file_path not in self.document_registry or self.document_registry[file_path] != current_hash:
                        logger.info(f"Loading document: {file_path}")
                        documents = self.load_document_by_format(file_path)
                        if documents:
                            self.add_documents(documents)
                            self.document_registry[file_path] = current_hash
                            total_loaded += 1
        
        # Also check for legacy txt file
        legacy_path = Path(__file__).parent / "data_raw" / "genetic_docs.txt"
        if os.path.exists(legacy_path):
            current_hash = self.get_document_hash(legacy_path)
            if str(legacy_path) not in self.document_registry or self.document_registry[str(legacy_path)] != current_hash:
                logger.info(f"Loading legacy genetic document: {legacy_path}")
                documents = self.load_documents_from_txt(legacy_path)
                if documents:
                    self.add_documents(documents)
                    self.document_registry[str(legacy_path)] = current_hash
                    total_loaded += 1
        
        logger.info(f"Scanned and loaded {total_loaded} documents from {self.watch_directory}")
    
    def load_new_document(self, file_path: str):
        """Load a single new document (called by file watcher)"""
        try:
            current_hash = self.get_document_hash(file_path)
            documents = self.load_document_by_format(file_path)
            if documents:
                self.add_documents(documents)
                self.document_registry[file_path] = current_hash
                logger.info(f"Successfully loaded new document: {file_path}")
        except Exception as e:
            logger.error(f"Error loading new document {file_path}: {e}")
    
    def reload_document(self, file_path: str):
        """Reload a modified document (called by file watcher)"""
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
                    self.vector_store._collection.delete(
                        where={"source": file_path}
                    )
                    # Add new documents
                    self.add_documents(documents)
                    self.document_registry[file_path] = current_hash
                    logger.info(f"Successfully reloaded document: {file_path}")
            else:
                logger.info(f"Document unchanged, skipping reload: {file_path}")
        except Exception as e:
            logger.error(f"Error reloading document {file_path}: {e}")

    def parse_genetic_document(self, content: str) -> Dict[str, Any]:
        """Parse genetic document content and extract structured information"""
        lines = content.strip().split('\n')
        
        # Extract metadata
        topic = ""
        intelligence_type = ""
        research_area = ""
        category = "General"
        
        # Look for topics in the first few lines
        for line in lines[:5]:
            if line.strip().startswith('##') and not line.strip().startswith('###'):
                topic = line.strip('#').strip()
                break
        
        # Determine intelligence type
        intelligence_keywords = {
            'ngôn ngữ': 'Language Intelligence',
            'logic': 'Logical Intelligence', 
            'toán học': 'Mathematical Intelligence',
            'âm nhạc': 'Musical Intelligence',
            'không gian': 'Spatial Intelligence',
            'thể chất': 'Bodily Intelligence',
            'cá nhân': 'Intrapersonal Intelligence',
            'xã hội': 'Interpersonal Intelligence'
        }
        
        for keyword, intel_type in intelligence_keywords.items():
            if keyword in content.lower():
                intelligence_type = intel_type
                break
        
        # Determine research area
        if 'mash' in content.lower() or 'hệ thống' in content.lower():
            research_area = "Biomedical Data Systems"
        elif 'gen' in content.lower() or 'genetic' in content.lower():
            research_area = "Genetic Analysis"
        elif 'trí thông minh' in content.lower() or 'intelligence' in content.lower():
            research_area = "Intelligence Research"
        elif 'y tế' in content.lower() or 'medical' in content.lower():
            research_area = "Medical Research"
        
        # Determine category
        if 'trí thông minh' in content.lower() or 'intelligence' in content.lower():
            category = "Intelligence & Genetics"
        elif 'hệ thống' in content.lower() or 'system' in content.lower():
            category = "Data Systems"
        elif 'nghiên cứu' in content.lower() or 'research' in content.lower():
            category = "Research & Development"
        elif 'y tế' in content.lower() or 'sức khỏe' in content.lower():
            category = "Health & Medicine"
        elif 'gen' in content.lower() or 'genetic' in content.lower():
            category = "Genetic Analysis"
        
        return {
            'topic': topic,
            'intelligence_type': intelligence_type,
            'research_area': research_area,
            'category': category,
            'content_length': len(content),
            'source': 'genetic_docs.txt'
        }

    
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
    
    def get_document_hash(self, file_path: str) -> str:
        """Calculate hash of document for change detection"""
        try:
            with open(file_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            return file_hash
        except Exception as e:
            logger.error(f"Error calculating hash for {file_path}: {e}")
            return ""
    
    def load_pdf_document(self, pdf_path: str) -> List[Document]:
        """Load documents from PDF file"""
        documents = []
        try:
            reader = (pdf_path)
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
                    parsed = self.parse_genetic_document(content)
                    
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
    
    def load_documents_from_txt(self, txt_file_path: str = None) -> List[Document]:
        """Load and parse genetic documents from text file"""
        if not txt_file_path:
            data_file = Path(__file__).parent / "data_raw" / "genetic_docs.txt"
            txt_file_path = str(data_file)
        
        if not os.path.exists(txt_file_path):
            logger.error(f"Genetic data file not found: {txt_file_path}")
            return []
        
        try:
            with open(txt_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split by double ## markers for sections
            sections = content.split('##')
            documents = []
            
            for i, section in enumerate(sections):
                section = section.strip()
                if len(section) < 20:  # Skip very short sections
                    continue
                
                # Parse document metadata
                metadata = self.parse_genetic_document(section)
                metadata['doc_id'] = f"genetic_doc_{i}"
                metadata['section_number'] = i
                metadata['source'] = txt_file_path
                
                # Create document
                doc = Document(
                    page_content=section,
                    metadata=metadata
                )
                documents.append(doc)
            
            logger.info(f"Loaded {len(documents)} documents from {txt_file_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading genetic documents: {e}")
            return []

    def add_documents(self, documents: List[Document]) -> bool:
        """Add documents to the vector store"""
        if not documents:
            logger.warning("No documents to add")
            return False
        
        try:
            # Add to vector store
            self.vector_store.add_documents(documents)
            logger.info(f"Added {len(documents)} documents to genetic collection")
            
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

    def initialize_from_txt(self, txt_file_path: str = None) -> bool:
        """Initialize the retriever by loading documents from txt file"""
        logger.info("Initializing genetic retriever from txt file...")
        
        # Load documents
        documents = self.load_documents_from_txt(txt_file_path)
        if not documents:
            logger.error("No documents loaded from txt file")
            return False
        
        # Add documents to vector store
        success = self.add_documents(documents)
        if success:
            logger.info(f"Successfully initialized genetic retriever with {len(documents)} documents")
        
        return success

    def get_relevant_documents_with_reranking(self, query: str, k: int = 8) -> List[RetrievedGeneticDocument]:
        """Get documents using hybrid search with advanced reranking"""
        all_docs = []
        initial_k = max(k * 2, 15)  # Get more docs initially for better reranking
        
        try:
            # 1. Vector similarity search
            vector_docs = self.vector_store.similarity_search_with_score(query, k=initial_k)
            for doc, score in vector_docs:
                all_docs.append((doc, score, "vector"))
            
            # 2. BM25 search if available
            if self.bm25_retriever:
                bm25_docs = self.bm25_retriever.get_relevant_documents(query)[:initial_k]
                for doc in bm25_docs:
                    all_docs.append((doc, 0.7, "bm25"))  # Default BM25 score
            
            # 3. Convert to RetrievedGeneticDocument objects
            retrieved_docs = []
            for doc, score, method in all_docs:
                retrieved_doc = RetrievedGeneticDocument(
                    content=doc.page_content,
                    source=doc.metadata.get('source', 'unknown'),
                    retrieval_score=float(score),
                    topic=doc.metadata.get('topic', ''),
                    intelligence_type=doc.metadata.get('intelligence_type', ''),
                    research_area=doc.metadata.get('research_area', ''),
                    category=doc.metadata.get('category', 'General')
                )
                retrieved_docs.append(retrieved_doc)
            
            # 4. Advanced reranking
            reranked_docs = self.rerank_documents(retrieved_docs, query)
            
            return reranked_docs[:k]
            
        except Exception as e:
            logger.error(f"Error in document retrieval: {e}")
            return []

    def rerank_documents(self, docs: List[RetrievedGeneticDocument], query: str) -> List[RetrievedGeneticDocument]:
        """Advanced reranking with genetic-specific scoring"""
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
        """Enhanced document retrieval with advanced reranking for better genetic context."""
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
                result_prefix = f"🧬 GENETIC {doc.category}"
                if doc.topic:
                    result_prefix += f" [{doc.topic[:50]}...]" if len(doc.topic) > 50 else f" [{doc.topic}]"
                if doc.intelligence_type:
                    result_prefix += f" ({doc.intelligence_type})"
                elif doc.research_area:
                    result_prefix += f" ({doc.research_area})"
                
                formatted_content = f"{result_prefix}: {doc.content}"
                formatted_results.append(formatted_content)
            
            # Log retrieval performance
            time_elapsed = time.time() - time_start
            logger.info(f"Enhanced {self.collection_type} document retrieval completed in {time_elapsed:.2f} seconds with {len(formatted_results)} reranked results.")
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error during {self.collection_type} document retrieval: {e}")
            return [f"Lỗi khi tìm kiếm thông tin {self.collection_type}: {str(e)}"]

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the genetic collection"""
        try:
            if not hasattr(self.vector_store, '_collection'):
                return {"error": "Vector store not available"}
            
            all_docs = self.vector_store.get()
            if not all_docs['metadatas']:
                return {"total_documents": 0}
            
            # Basic stats
            total_docs = len(all_docs['metadatas'])
            
            # Extract categories and intelligence types
            categories = set()
            intelligence_types = set()
            research_areas = set()
            topics = set()
            
            for metadata in all_docs['metadatas']:
                if metadata.get('category'):
                    categories.add(metadata['category'])
                if metadata.get('intelligence_type'):
                    intelligence_types.add(metadata['intelligence_type'])
                if metadata.get('research_area'):
                    research_areas.add(metadata['research_area'])
                if metadata.get('topic'):
                    topics.add(metadata['topic'])
            
            return {
                "total_documents": total_docs,
                "unique_topics": len(topics),
                "unique_categories": len(categories),
                "unique_intelligence_types": len(intelligence_types),
                "unique_research_areas": len(research_areas),
                "categories": sorted(list(categories)),
                "intelligence_types": sorted(list(intelligence_types)),
                "research_areas": sorted(list(research_areas)),
                "sample_topics": sorted(list(topics))[:10]  # Show first 10 topics
            }
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {"error": str(e)}

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
            logger.info("Auto scanning enabled")
    
    def disable_auto_scan(self):
        """Disable automatic document scanning"""
        if self.auto_scan_enabled:
            self.auto_scan_enabled = False
            self.stop_document_watcher()
            logger.info("Auto scanning disabled")
    
    def manual_scan(self):
        """Manually trigger a scan for new documents"""
        logger.info("Starting manual genetic document scan...")
        self.scan_and_load_all_documents()
    
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
            if not hasattr(self.vector_store, '_collection'):
                return {"status": "not_initialized"}
                
            document_count = self.vector_store._collection.count()
            return {
                "status": "initialized" if self.is_initialized else "initializing",
                "document_count": document_count,
                "document_registry_size": len(self.document_registry),
                "auto_scan_enabled": self.auto_scan_enabled,
                "watch_directory": self.watch_directory,
                "file_watcher_active": self.observer is not None
            }
        except Exception as e:
            logger.error(f"Error getting load status: {e}")
            return {"status": "error", "error": str(e)}
    
    def __del__(self):
        """Cleanup method to stop file watcher when object is destroyed"""
        try:
            self.stop_document_watcher()
        except:
            pass
    
    def cleanup(self):
        """Explicit cleanup method for proper resource management"""
        self.stop_document_watcher()
        logger.info("GeneticRetrieverTool cleanup completed")