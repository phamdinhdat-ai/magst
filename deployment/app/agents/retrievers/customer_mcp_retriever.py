# customer_retriever_mcp_client.py
import asyncio
import os
import json
import time
import hashlib
import threading
import re
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor

from loguru import logger
from pydantic import BaseModel, Field as PydanticField
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# --- LangChain Imports for Document Processing ---
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, JSONLoader, TextLoader, PyMuPDFLoader

# --- MCP Client Imports ---
from mcp.client.sse import sse_client
from mcp.client.session import ClientSession

# --- Base Agent Tool Import ---
from app.agents.factory.tools.base import BaseAgentTool  # Adjust import path as needed
from app.utils.document_processor import DocumentCustomConverter, markdown_splitter, remove_image_tags
from app.core.config import settings

class CustomerDocumentWatcher(FileSystemEventHandler):
    """File system watcher that triggers document uploads for a specific customer."""
    
    def __init__(self, retriever_tool: 'CustomerRetrieverMCPClient'):
        self.tool = retriever_tool
        self.customer_id = retriever_tool.customer_id

    def _is_relevant_file(self, file_path_str: str) -> bool:
        """Check if file is relevant to this customer."""
        filename = os.path.basename(file_path_str)
        return f"customer_{self.customer_id}" in filename or filename.startswith(f"{self.customer_id}_")

    def on_created(self, event):
        if not event.is_directory  and "_registry.json" not in event.src_path:
            logger.info(f"[Customer Watcher] New file detected for customer {self.customer_id}: {event.src_path}")
            time.sleep(1)  # Small delay to ensure file is fully written
            asyncio.create_task(self.tool._process_file_if_needed(Path(event.src_path)))

    def on_modified(self, event):
        if not event.is_directory and "_registry.json" not in event.src_path :
            logger.info(f"[Customer Watcher] File modified for customer {self.customer_id}: {event.src_path}")
            time.sleep(1)
            asyncio.create_task(self.tool._process_file_if_needed(Path(event.src_path)))
            
    def on_deleted(self, event):
        if not event.is_directory and "_registry.json" not in event.src_path:
            logger.info(f"[Customer Watcher] File deleted for customer {self.customer_id}: {event.src_path}")
            asyncio.create_task(self.tool._remove_from_registry(Path(event.src_path)))


class CustomerRetrieverInput(BaseModel):
    """Input schema for customer document retrieval."""
    query: str = PydanticField(description="Search query for finding relevant customer documents")
    collection_name: str = PydanticField(description="Name of the document collection to search in", default=None)
    max_results: int = PydanticField(default=5, description="Maximum number of results to return")


class CustomerRetrieverMCPClient(BaseAgentTool):
    """
    MCP client tool that connects to a remote MCP server for customer-specific document
    retrieval and ingestion. Watches a local directory for customer files and automatically
    uploads new/modified documents to the remote vector database with customer isolation.
    """
    
    name: str = "customer_retriever_mcp"
    args_schema: type[BaseModel] = CustomerRetrieverInput
    
    # Configuration
    mcp_server_url: str = PydanticField(description="URL of the MCP server")
    watch_directory: Path = PydanticField(description="Directory to watch for document changes")
    customer_id: str = PydanticField(description="Customer ID for document isolation")
    collection_name: str = PydanticField(description="Collection name for this customer")
    
    # Internal components
    _document_registry: Dict[str, str] = {}
    _observer: Optional[Observer] = None
    _text_splitter: Optional[RecursiveCharacterTextSplitter] = None
    _thread_pool: Optional[ThreadPoolExecutor] = None
    _is_initialized: bool = False
    
    def __init__(self, 
                 mcp_server_url: str,
                 watch_directory: str,
                 customer_id: str = "123",
                 **kwargs):
        """
        Initialize the customer MCP client tool.
        
        Args:
            mcp_server_url: URL of the MCP server (e.g., "http://localhost:50051/sse")
            watch_directory: Local directory to watch for document changes
            customer_id: Customer ID for document isolation
        """
        # Sanitize customer ID for use in collection names
        safe_customer_id = re.sub(r'[^a-zA-Z0-9_.-]', '_', customer_id)
        collection_name = f"customer_{safe_customer_id}_data"
        
        # Remove conflicting kwargs if they exist
        kwargs.pop('description', None)
        kwargs.pop('name', None)
        
        super().__init__(
            mcp_server_url=mcp_server_url,
            watch_directory=Path(watch_directory).resolve(),
            customer_id=customer_id,
            collection_name=self._sanitize_collection_name(collection_name),
            name=f"customer_retriever_mcp_{safe_customer_id}",
            description=f"Retrieves and manages documents for customer {customer_id}",
            **kwargs
        )
        
        if not self._is_initialized:
            asyncio.create_task(self._initialize_all())
    
    def _sanitize_collection_name(self, name: str) -> str:
        """Sanitize collection name for database compatibility."""
        return re.sub(r'[^a-zA-Z0-9_.-]', '_', name)[:63]
    
    async def _initialize_all(self):
        """Initialize all components."""
        logger.info(f"Initializing CustomerRetrieverMCPClient for customer '{self.customer_id}'...")
        
        # Ensure watch directory exists
        self.watch_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize document processing components
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.OVERLAP_SIZE,
            is_separator_regex=True
        )
        
        # Initialize thread pool for parallel processing
        self._thread_pool = ThreadPoolExecutor(
            max_workers=4,
            thread_name_prefix=f"customer_mcp_{self.customer_id}"
        )
        
        # Load document registry
        await self._load_document_registry()
        
        # Test MCP server connection
        if await self._test_server_connection():
            logger.info(f"MCP server connection successful for customer '{self.customer_id}'")
        else:
            logger.warning(f"MCP server connection failed for customer '{self.customer_id}' - some features may not work")
        
        # Scan and process existing files
        await self._scan_and_process_all_files()
        
        # Start document watcher
        self._start_document_watcher()
        
        self._is_initialized = True
        logger.info(f"CustomerRetrieverMCPClient initialized successfully for customer '{self.customer_id}'")
    
    async def _test_server_connection(self) -> bool:
        """Test connection to MCP server."""
        try:
            async with sse_client(url=self.mcp_server_url) as streams:
                async with ClientSession(read_stream=streams[0], write_stream=streams[1]) as session:
                    await session.initialize()
                    logger.debug(f"MCP server connection test successful for customer '{self.customer_id}'")
                    return True
        except Exception as e:
            logger.error(f"MCP server connection test failed for customer '{self.customer_id}': {e}")
            return False
    
    @property
    def _registry_path(self) -> Path:
        """Path to the document registry file."""
        return self.watch_directory / f"{self.collection_name}_mcp_registry.json"
    
    async def _load_document_registry(self):
        """Load the document registry from disk."""
        if self._registry_path.exists():
            try:
                with open(self._registry_path, 'r', encoding='utf-8') as f:
                    self._document_registry = json.load(f)
                logger.info(f"Loaded {len(self._document_registry)} entries from registry for customer '{self.customer_id}'")
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Failed to load registry for customer '{self.customer_id}': {e}. Starting fresh.")
                self._document_registry = {}
        else:
            self._document_registry = {}
    
    async def _save_document_registry(self):
        """Save the document registry to disk."""
        try:
            with open(self._registry_path, 'w', encoding='utf-8') as f:
                json.dump(self._document_registry, f, indent=2, ensure_ascii=False)
            logger.debug(f"Registry saved with {len(self._document_registry)} entries for customer '{self.customer_id}'")
        except IOError as e:
            logger.error(f"Failed to save registry for customer '{self.customer_id}': {e}")
    
    def _get_file_hash(self, file_path: Path) -> Optional[str]:
        """Calculate MD5 hash of a file."""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except IOError as e:
            logger.error(f"Could not read file for hashing: {file_path}. Error: {e}")
            return None
    
    # def _is_relevant_file(self, file_path: Path) -> bool:
    #     """Check if file is relevant to this customer."""
    #     filename = file_path.name
    #     return f"customer_{self.customer_id}" in filename or filename.startswith(f"{self.customer_id}_")
    
    def _load_and_split_file(self, file_path: Path) -> List[Document]:
        """Load and split a file into document chunks with customer-specific processing."""
        loader_map = {
            '.pdf': PyMuPDFLoader,
            '.csv': CSVLoader,
            '.json': JSONLoader,
            '.txt': TextLoader
        }
        
        loader_class = loader_map.get(file_path.suffix.lower())
        if not loader_class:
            logger.warning(f"Unsupported file type: {file_path.suffix}")
            return []
        
        try:
            logger.info(f"Loading file: {file_path.name} with {loader_class.__name__} for customer '{self.customer_id}'")
            loader = loader_class(str(file_path))
            raw_docs = loader.load()
            logger.info(f"Loaded {len(raw_docs)} documents from {file_path.name}")
            
            if raw_docs and len(raw_docs) > 0:
                # Handle case where loader returns strings instead of Document objects
                first_doc = raw_docs[0]
                if isinstance(first_doc, str):
                    logger.info(f"Loader returned strings, converting to Document objects")
                    raw_docs = [Document(page_content=doc, metadata={"source": file_path.name}) for doc in raw_docs]
                    first_doc = raw_docs[0]
                
                logger.info(f"First document preview: {first_doc.page_content[:100]}...")
            else:
                logger.warning(f"No documents loaded from {file_path.name}")
                return []
                
            # Special processing for PDF files
            # if "pdf" in file_path.suffix.lower():
            #     logger.info(f"Processing PDF file: {file_path.name}")
            #     first_doc_content = raw_docs[0].page_content if hasattr(raw_docs[0], 'page_content') else str(raw_docs[0])
            #     cleaned_text = remove_image_tags(first_doc_content)
            #     logger.info(f"Cleaned text length: {len(cleaned_text)} characters")
            #     try: 
            #         raw_docs = markdown_splitter(cleaned_text)
            #     except Exception as e:
            #         logger.error(f"Error splitting markdown from PDF file {file_path.name}: {e}")
            #     logger.info(f"Split {len(raw_docs)} sections from PDF file: {file_path.name}")
            
            # Split documents into chunks
            split_docs = self._text_splitter.split_documents(raw_docs)
            
            # Add source metadata with customer information
            for doc in split_docs:
                doc.metadata['source'] = file_path.name
                doc.metadata['file_path'] = str(file_path)
                doc.metadata['customer_id'] = self.customer_id
                doc.metadata['timestamp'] = time.time()
            
            logger.info(f"Successfully processed {len(split_docs)} document chunks from {file_path.name} for customer '{self.customer_id}'")
            return split_docs
            
        except Exception as e:
            logger.error(f"Error loading file {file_path} for customer '{self.customer_id}': {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []
    
    async def _upload_documents_to_server(self, documents: List[Document], collection_name: str) -> bool:
        """Upload document chunks to the MCP server."""
        if not documents:
            return False
        
        try:
            # Convert documents to the format expected by the server
            doc_contents = [doc.page_content for doc in documents]
            doc_metadata = [doc.metadata for doc in documents]
            
            async with sse_client(url=self.mcp_server_url) as streams:
                async with ClientSession(read_stream=streams[0], write_stream=streams[1]) as session:
                    await session.initialize()
                    
                    # Call the ingest_documents tool on the server
                    response = await session.call_tool(
                        "ingest_documents",
                        arguments={
                            "documents": doc_contents,
                            "metadatas": doc_metadata,
                            "collection_name": collection_name
                        }
                    )
                    
                    if response and response.content:
                        result_text = response.content[0].text
                        logger.info(f"Server response for customer '{self.customer_id}': {result_text}")
                        return "successfully" in result_text.lower()
                    else:
                        logger.error(f"No response from server for customer '{self.customer_id}'")
                        return False
                        
        except Exception as e:
            logger.error(f"Error uploading documents to server for customer '{self.customer_id}': {e}")
            return False
    
    async def _process_file_if_needed(self, file_path: Path):
        """Process a file if it has changed since last processing."""
        if "_registry.json" in file_path.name or "_mcp_registry.json" in file_path.name:
            return
        
        # Check if file is relevant to this customer
        # if not self._is_relevant_file(file_path):
        #     logger.debug(f"File {file_path.name} is not relevant to customer '{self.customer_id}', skipping")
        #     return
        
        current_hash = self._get_file_hash(file_path)
        if not current_hash:
            return
        
        stored_hash = self._document_registry.get(str(file_path))
        if current_hash != stored_hash:
            logger.info(f"Change detected for '{file_path.name}' for customer '{self.customer_id}'. Processing...")
            
            # Load and split the file
            documents = self._load_and_split_file(file_path)
            
            if documents:
                # Upload to server
                success = await self._upload_documents_to_server(documents, self.collection_name)
                
                if success:
                    # Update registry
                    self._document_registry[str(file_path)] = current_hash
                    await self._save_document_registry()
                    logger.info(f"Successfully processed and uploaded '{file_path.name}' for customer '{self.customer_id}'")
                else:
                    logger.error(f"Failed to upload '{file_path.name}' to server for customer '{self.customer_id}'")
            else:
                logger.warning(f"No documents extracted from '{file_path.name}' for customer '{self.customer_id}'")
    
    async def _scan_and_process_all_files(self):
        """Scan directory and process all relevant files for this customer."""
        logger.info(f"Scanning directory: {self.watch_directory} for customer '{self.customer_id}' files")
        
        current_files = set()
        for file_path in self.watch_directory.rglob('*'):
            logger.debug(f"Found file: {file_path}")
            if (file_path.is_file() and 
                not file_path.name.endswith('_registry.json')):
                current_files.add(str(file_path))
                logger.debug(f"Processing file: {file_path} for customer '{self.customer_id}'")
                await self._process_file_if_needed(file_path)
        
        # Clean up registry for deleted files
        registered_files = set(self._document_registry.keys())
        deleted_files = registered_files - current_files
        for file_path_str in deleted_files:
            await self._remove_from_registry(Path(file_path_str))
        
        await self._save_document_registry()
    
    async def _remove_from_registry(self, file_path: Path):
        """Remove a file from the registry."""
        path_str = str(file_path)
        if path_str in self._document_registry:
            logger.info(f"Removing '{file_path.name}' from registry for customer '{self.customer_id}'")
            del self._document_registry[path_str]
            await self._save_document_registry()
    
    def _start_document_watcher(self):
        """Start watching the directory for file changes."""
        if self._observer:
            return
        
        self._observer = Observer()
        event_handler = CustomerDocumentWatcher(self)
        self._observer.schedule(event_handler, str(self.watch_directory), recursive=True)
        
        watcher_thread = threading.Thread(target=self._observer.start, daemon=True)
        watcher_thread.start()
        logger.info(f"Started document watcher on '{self.watch_directory}' for customer '{self.customer_id}'")
    
    async def _retrieve_from_server(self, query: str, collection_name: str, k: int = 5) -> str:
        """Retrieve documents from the MCP server."""
        try:
            async with sse_client(url=self.mcp_server_url) as streams:
                async with ClientSession(read_stream=streams[0], write_stream=streams[1]) as session:
                    await session.initialize()
                    
                    # Call the retrieve_vectorstore tool on the server
                    response = await session.call_tool(
                        "retrieve_vectorstore_with_reranker",
                        arguments={
                            "query": query,
                            "collection": collection_name,
                            "initial_k": settings.TOP_K_RETRIEVE,
                            "final_k": settings.TOP_K_RERANK
                        }
                    )
                    
                    if response and response.content:
                        result_text = response.content[0].text
                        logger.info(f"Retrieved documents for customer '{self.customer_id}' with query: '{query[:50]}...'")
                        return result_text
                    else:
                        return f"No response from server for customer '{self.customer_id}'"
                        
        except ConnectionRefusedError:
            error_msg = f"Connection refused. Is the MCP server running at {self.mcp_server_url}?"
            logger.error(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"Error retrieving documents from server for customer '{self.customer_id}': {e}"
            logger.error(error_msg)
            return error_msg
    
    def _run(self, query: str, collection_name: str = None, max_results: int = 5) -> str:
        """Synchronous document retrieval."""
        collection = collection_name or self.collection_name
        return asyncio.run(self._arun(query, collection, max_results))
    
    async def _arun(self, query: str, collection_name: str = None, max_results: int = 5) -> str:
        """Asynchronous document retrieval."""
        if not self._is_initialized:
            return f"Error: Customer retriever client for customer '{self.customer_id}' is not initialized."
        
        collection = collection_name or self.collection_name
        logger.info(f"Retrieving documents for customer '{self.customer_id}' with query: '{query}' from collection: '{collection}'")
        
        start_time = time.time()
        result = await self._retrieve_from_server(query, collection, max_results)
        retrieval_time = time.time() - start_time
        
        logger.info(f"Document retrieval completed for customer '{self.customer_id}' in {retrieval_time:.2f}s")
        return result
    
    # Public methods for manual operations
    async def upload_file(self, file_path: str, collection_name: str = None) -> str:
        """Manually upload a specific file to the server."""
        collection = collection_name or self.collection_name
        path = Path(file_path)
        
        if not path.exists():
            return f"Error: File not found: {file_path}"
        
        # Check if file is relevant to this customer
        # if not self._is_relevant_file(path):
        #     return f"Error: File {file_path} is not relevant to customer '{self.customer_id}'"
        
        documents = self._load_and_split_file(path)
        if not documents:
            return f"Error: Could not process file: {file_path}"
        
        success = await self._upload_documents_to_server(documents, collection)
        if success:
            # Update registry
            file_hash = self._get_file_hash(path)
            if file_hash:
                self._document_registry[str(path)] = file_hash
                await self._save_document_registry()
            return f"Successfully uploaded {len(documents)} document chunks from {file_path} for customer '{self.customer_id}'"
        else:
            return f"Failed to upload file: {file_path} for customer '{self.customer_id}'"
    
    async def upload_text_documents(self, texts: List[str], collection_name: str = None, source_name: str = "manual_upload") -> str:
        """Manually upload text documents to the server."""
        collection = collection_name or self.collection_name
        
        if not texts:
            return "Error: No texts provided"
        
        # Create Document objects with customer metadata
        documents = []
        for i, text in enumerate(texts):
            doc = Document(
                page_content=text,
                metadata={
                    'source': f"{source_name}_{i}",
                    'customer_id': self.customer_id,
                    'timestamp': time.time()
                }
            )
            documents.append(doc)
        
        success = await self._upload_documents_to_server(documents, collection)
        if success:
            return f"Successfully uploaded {len(documents)} text documents to collection '{collection}' for customer '{self.customer_id}'"
        else:
            return f"Failed to upload text documents to collection '{collection}' for customer '{self.customer_id}'"
    
    async def get_status(self) -> Dict[str, Any]:
        """Get the current status of the client."""
        status = {
            "initialized": self._is_initialized,
            "customer_id": self.customer_id,
            "mcp_server_url": self.mcp_server_url,
            "watch_directory": str(self.watch_directory),
            "collection_name": self.collection_name,
            "tracked_files": len(self._document_registry),
            "server_connected": await self._test_server_connection()
        }
        return status
    
    def cleanup(self):
        """Clean up resources."""
        customer_id = getattr(self, 'customer_id', 'unknown')
        logger.info(f"Cleaning up CustomerRetrieverMCPClient resources for customer '{customer_id}'...")
        
        if hasattr(self, '_observer') and self._observer and self._observer.is_alive():
            self._observer.stop()
            self._observer.join()
            logger.info(f"Document watcher stopped for customer '{customer_id}'")
        
        if hasattr(self, '_thread_pool') and self._thread_pool:
            self._thread_pool.shutdown(wait=False)
            logger.info(f"Thread pool shut down for customer '{customer_id}'")
    
    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.cleanup()
        except Exception as e:
            # Use getattr to safely access customer_id in case initialization failed
            customer_id = getattr(self, 'customer_id', 'unknown')
            logger.error(f"Error during cleanup for customer '{customer_id}': {e}")


# Example usage and factory functions
def create_customer_retriever_client(mcp_server_url: str, 
                                    watch_directory: str,
                                    customer_id: str) -> CustomerRetrieverMCPClient:
    """
    Factory function to create a CustomerRetrieverMCPClient instance.
    
    Args:
        mcp_server_url: URL of the MCP server (e.g., "http://localhost:50051/sse")
        watch_directory: Local directory to watch for documents
        customer_id: Customer ID for document isolation
    
    Returns:
        Configured CustomerRetrieverMCPClient instance
    """
    return CustomerRetrieverMCPClient(
        mcp_server_url=mcp_server_url,
        watch_directory=watch_directory,
        customer_id=customer_id
    )


# Example usage
if __name__ == "__main__":
    async def main():
        # Configuration
        SERVER_URL = "http://192.168.1.60:50051/sse"
        WATCH_DIR = "app/uploaded_files/documents/customer_21"
        CUSTOMER_ID = "21"
        
        # Create the client
        client = create_customer_retriever_client(
            mcp_server_url=SERVER_URL,
            watch_directory=WATCH_DIR,
            customer_id=CUSTOMER_ID
        )
        
        # Wait for initialization
        await asyncio.sleep(2)
        
        # Test retrieval
        query = "thông tin bệnh tim mạch"
        print(f"Query: {query}")
        result = await client.arun(query)
        print(f"Result: {result}")
        
        # Test manual upload
        test_texts = [
            f"Customer {CUSTOMER_ID} health report: Risk of diabetes type 2 at moderate level.",
            f"Customer {CUSTOMER_ID} genetic results: BRCA1 negative, CFTR positive.",
        ]
        upload_result = await client.upload_text_documents(test_texts)
        print(f"Upload result: {upload_result}")
        
        # Test query again
        result2 = await client.arun("genetic results CFTR")
        print(f"Result after upload: {result2}")
        
        # Get status
        status = await client.get_status()
        print(f"Status: {json.dumps(status, indent=2)}")
        
        # Cleanup
        client.cleanup()
    
    asyncio.run(main())