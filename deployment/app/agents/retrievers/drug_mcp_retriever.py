import asyncio
import os
import json
import time
import hashlib
import threading
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
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, JSONLoader, TextLoader

# --- MCP Client Imports ---
from mcp.client.sse import sse_client
from mcp.client.session import ClientSession

# --- Base Agent Tool Import ---
from app.agents.factory.tools.base import BaseAgentTool  # Adjust import path as needed
from app.utils.document_processor import markdown_splitter, remove_image_tags, DocumentCustomConverter
from app.core.config import settings

class DocumentWatcher(FileSystemEventHandler):
    """File system watcher that triggers document uploads to MCP server."""
    
    def __init__(self, retriever_tool: 'DrugRetrieverMCPClient'):
        self.tool = retriever_tool

    def on_created(self, event):
        if not event.is_directory:
            logger.info(f"[Watcher] New file detected: {event.src_path}")
            time.sleep(1)  # Small delay to ensure file is fully written
            asyncio.create_task(self.tool._process_file_if_needed(Path(event.src_path)))

    def on_modified(self, event):
        if not event.is_directory and "_registry.json" not in event.src_path:
            logger.info(f"[Watcher] File modified: {event.src_path}")
            time.sleep(1)
            asyncio.create_task(self.tool._process_file_if_needed(Path(event.src_path)))
            
    def on_deleted(self, event):
        if not event.is_directory:
            logger.info(f"[Watcher] File deleted: {event.src_path}")
            # Note: For deletion, we'd need a delete tool on the server side
            # For now, we'll just update our local registry
            asyncio.create_task(self.tool._remove_from_registry(Path(event.src_path)))


class DrugRetrieverInput(BaseModel):
    """Input schema for drug document retrieval."""
    query: str = PydanticField(description="Search query for finding relevant drug documents")
    collection_name: str = PydanticField(description="Name of the document collection to search in")
    max_results: int = PydanticField(default=5, description="Maximum number of results to return")


class DrugRetrieverMCPClient(BaseAgentTool):
    """Tool to retrieve drug documents from a remote vector database via MCP server."""
    
    name: str = "drug_retriever_mcp"
    description: str = "Retrieves drug documents from remote vector database and manages document uploads"
    args_schema: type[BaseModel] = DrugRetrieverInput
    
    # Configuration
    mcp_server_url: str = PydanticField(description="URL of the MCP server")
    watch_directory: Path = PydanticField(description="Directory to watch for document changes")
    default_collection: str = PydanticField(default="drug_docs", description="Default collection name")
    
    # Internal components
    _document_registry: Dict[str, str] = {}
    _observer: Optional[Observer] = None
    _text_splitter: Optional[RecursiveCharacterTextSplitter] = None
    _thread_pool: Optional[ThreadPoolExecutor] = None
    _is_initialized: bool = False
    
    def __init__(self, 
                 mcp_server_url: str,
                 watch_directory: str,
                 default_collection: str = "drug_docs",
                 **kwargs):
        """
        Initialize the MCP client tool.
        
        Args:
            mcp_server_url: URL of the MCP server (e.g., "http://localhost:50051/sse")
            watch_directory: Local directory to watch for document changes
            default_collection: Default collection name for documents
        """
        super().__init__(
            mcp_server_url=mcp_server_url,
            watch_directory=Path(watch_directory).resolve(),
            default_collection=default_collection,
            **kwargs
        )
        
        if not self._is_initialized:
            asyncio.create_task(self._initialize_all())
    
    async def _initialize_all(self):
        """Initialize all components."""
        logger.info(f"Initializing DrugRetrieverMCPClient...")
        
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
            thread_name_prefix="Drug_mcp_client"
        )
        
        # Load document registry
        await self._load_document_registry()
        
        # Test MCP server connection
        if await self._test_server_connection():
            logger.info("MCP server connection successful")
        else:
            logger.warning("MCP server connection failed - some features may not work")
        
        # Scan and process existing files
        await self._scan_and_process_all_files()
        
        # Start document watcher
        self._start_document_watcher()
        
        self._is_initialized = True
        logger.info("DrugRetrieverMCPClient initialized successfully")
    
    async def _test_server_connection(self) -> bool:
        """Test connection to MCP server."""
        try:
            async with sse_client(url=self.mcp_server_url) as streams:
                async with ClientSession(read_stream=streams[0], write_stream=streams[1]) as session:
                    await session.initialize()
                    logger.debug("MCP server connection test successful")
                    return True
        except Exception as e:
            logger.error(f"MCP server connection test failed: {e}")
            return False
    
    @property
    def _registry_path(self) -> Path:
        """Path to the document registry file."""
        return self.watch_directory / f"{self.default_collection}_mcp_registry.json"
    
    async def _load_document_registry(self):
        """Load the document registry from disk."""
        if self._registry_path.exists():
            try:
                with open(self._registry_path, 'r', encoding='utf-8') as f:
                    self._document_registry = json.load(f)
                logger.info(f"Loaded {len(self._document_registry)} entries from registry")
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Failed to load registry: {e}. Starting fresh.")
                self._document_registry = {}
        else:
            self._document_registry = {}
    
    async def _save_document_registry(self):
        """Save the document registry to disk."""
        try:
            with open(self._registry_path, 'w', encoding='utf-8') as f:
                json.dump(self._document_registry, f, indent=2, ensure_ascii=False)
            logger.debug(f"Registry saved with {len(self._document_registry)} entries")
        except IOError as e:
            logger.error(f"Failed to save registry: {e}")
    
    def _get_file_hash(self, file_path: Path) -> Optional[str]:
        """Calculate MD5 hash of a file."""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except IOError as e:
            logger.error(f"Could not read file for hashing: {file_path}. Error: {e}")
            return None
    
    def _load_and_split_file(self, file_path: Path) -> List[Document]:
        """Load and split a file into document chunks."""
        loader_map = {
            '.pdf': PyPDFLoader,
            '.csv': CSVLoader,
            '.json': JSONLoader,
            '.txt': TextLoader
        }
        
        loader_class = loader_map.get(file_path.suffix.lower())
        if not loader_class:
            logger.warning(f"Unsupported file type: {file_path.suffix}")
            return []
        
        try:
            loader = loader_class(str(file_path))
            raw_docs = loader.load()
            
            # Split documents into chunks
            split_docs = self._text_splitter.split_documents(raw_docs)
            
            # Add source metadata
            for doc in split_docs:
                doc.metadata['source'] = file_path.name
                doc.metadata['file_path'] = str(file_path)
                doc.metadata['timestamp'] = time.time()
            
            logger.info(f"Loaded and split {file_path.name} into {len(split_docs)} chunks")
            return split_docs
            
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            return []

    async def _upload_documents_to_server(self, documents: List[Document], collection_name: str) -> bool:
        """Upload document chunks to the MCP server."""
        if not documents:
            return False
        
        try:
            # Convert documents to list of strings (content only)
            doc_contents = [doc.page_content for doc in documents]
            doc_metadata = [doc.metadata for doc in documents]
            
            # print(doc_contents[:10])
            async with sse_client(url=self.mcp_server_url) as streams:
                async with ClientSession(read_stream=streams[0], write_stream=streams[1]) as session:
                    await session.initialize()
                    
                    # Call the ingest_documents tool on the server
                    response = await session.call_tool(
                        "ingest_documents",
                        arguments={
                            "documents": doc_contents,
                            "metadata": doc_metadata,
                            "collection_name": collection_name
                        }
                    )
                    
                    if response and response.content:
                        result_text = response.content[0].text
                        logger.info(f"Server response: {result_text}")
                        return "successfully" in result_text.lower()
                    else:
                        logger.error("No response from server")
                        return False
                        
        except Exception as e:
            logger.error(f"Error uploading documents to server: {e}")
            return False
    
    async def _process_file_if_needed(self, file_path: Path):
        """Process a file if it has changed since last processing."""
        if "_registry.json" in file_path.name or "_mcp_registry.json" in file_path.name:
            return
        
        current_hash = self._get_file_hash(file_path)
        if not current_hash:
            return
        
        stored_hash = self._document_registry.get(str(file_path))
        if current_hash != stored_hash:
            logger.info(f"Change detected for '{file_path.name}'. Processing...")
            
            # Load and split the file
            documents = self._load_and_split_file(file_path)
            
            if documents:
                # Upload to server
                success = await self._upload_documents_to_server(documents, self.default_collection)
                
                if success:
                    # Update registry
                    self._document_registry[str(file_path)] = current_hash
                    await self._save_document_registry()
                    logger.info(f"Successfully processed and uploaded '{file_path.name}'")
                else:
                    logger.error(f"Failed to upload '{file_path.name}' to server")
            else:
                logger.warning(f"No documents extracted from '{file_path.name}'")
    
    async def _scan_and_process_all_files(self):
        """Scan directory and process all supported files."""
        logger.info(f"Scanning directory: {self.watch_directory}")
        
        current_files = set()
        for file_path in self.watch_directory.rglob('*'):
            if file_path.is_file() and not file_path.name.endswith('_registry.json'):
                current_files.add(str(file_path))
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
            logger.info(f"Removing '{file_path.name}' from registry")
            del self._document_registry[path_str]
            await self._save_document_registry()
    
    def _start_document_watcher(self):
        """Start watching the directory for file changes."""
        if self._observer:
            return
        
        self._observer = Observer()
        event_handler = DocumentWatcher(self)
        self._observer.schedule(event_handler, str(self.watch_directory), recursive=True)
        
        watcher_thread = threading.Thread(target=self._observer.start, daemon=True)
        watcher_thread.start()
        logger.info(f"Started document watcher on '{self.watch_directory}'")
    
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
                        logger.info(f"Retrieved documents for query: '{query[:50]}...'")
                        return result_text
                    else:
                        return "No response from server"
                        
        except ConnectionRefusedError:
            error_msg = f"Connection refused. Is the MCP server running at {self.mcp_server_url}?"
            logger.error(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"Error retrieving documents from server: {e}"
            logger.error(error_msg)
            return error_msg
    
    def _run(self, query: str, collection_name: str = None, max_results: int = 5) -> str:
        """Synchronous document retrieval."""
        collection = collection_name or self.default_collection
        return asyncio.run(self._arun(query, collection, max_results))
    
    async def _arun(self, query: str, collection_name: str = None, max_results: int = 5) -> str:
        """Asynchronous document retrieval."""
        if not self._is_initialized:
            return "Error: Drug retriever client is not initialized."
        
        collection = collection_name or self.default_collection
        logger.info(f"Retrieving documents for query: '{query}' from collection: '{collection}'")
        
        start_time = time.time()
        result = await self._retrieve_from_server(query, collection, max_results)
        retrieval_time = time.time() - start_time
        
        logger.info(f"Document retrieval completed in {retrieval_time:.2f}s")
        return result
    
    # Public methods for manual operations
    async def upload_file(self, file_path: str, collection_name: str = None) -> str:
        """Manually upload a specific file to the server."""
        collection = collection_name or self.default_collection
        path = Path(file_path)
        
        if not path.exists():
            return f"Error: File not found: {file_path}"
        
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
            return f"Successfully uploaded {len(documents)} document chunks from {file_path}"
        else:
            return f"Failed to upload file: {file_path}"
    
    async def upload_text_documents(self, texts: List[str], collection_name: str = None, source_name: str = "manual_upload") -> str:
        """Manually upload text documents to the server."""
        collection = collection_name or self.default_collection
        
        if not texts:
            return "Error: No texts provided"
        
        success = await self._upload_documents_to_server(texts, collection)  # texts is already a list of strings
        if success:
            return f"Successfully uploaded {len(texts)} text documents to collection '{collection}'"
        else:
            return f"Failed to upload text documents to collection '{collection}'"
    
    async def get_status(self) -> Dict[str, Any]:
        """Get the current status of the client."""
        status = {
            "initialized": self._is_initialized,
            "mcp_server_url": self.mcp_server_url,
            "watch_directory": str(self.watch_directory),
            "default_collection": self.default_collection,
            "tracked_files": len(self._document_registry),
            "server_connected": await self._test_server_connection()
        }
        return status
    
    def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up DrugRetrieverMCPClient resources...")
        
        if self._observer and self._observer.is_alive():
            self._observer.stop()
            self._observer.join()
            logger.info("Document watcher stopped")
        
        if self._thread_pool:
            self._thread_pool.shutdown(wait=False)
            logger.info("Thread pool shut down")
    
    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.cleanup()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


# Example usage and factory functions
def create_drug_retriever_client(mcp_server_url: str, 
                                watch_directory: str,
                                collection_name: str = "drug_docs") -> DrugRetrieverMCPClient:
    """
    Factory function to create a DrugRetrieverMCPClient instance.
    
    Args:
        mcp_server_url: URL of the MCP server (e.g., "http://localhost:50051/sse")
        watch_directory: Local directory to watch for documents
        collection_name: Collection name for the vector database
    
    Returns:
        Configured DrugRetrieverMCPClient instance
    """
    return DrugRetrieverMCPClient(
        mcp_server_url=mcp_server_url,
        watch_directory=watch_directory,
        default_collection=collection_name
    )


# Example usage
if __name__ == "__main__":
    async def main():
        # Configuration
        SERVER_URL = "http://192.168.1.60:50051/sse"
        WATCH_DIR = "app/agents/retrievers/storages/drugs"
        COLLECTION = "drug_knowledge"
        
        # Create the client
        client = create_drug_retriever_client(
            mcp_server_url=SERVER_URL,
            watch_directory=WATCH_DIR,
            collection_name=COLLECTION
        )
        
        # Wait for initialization
        await asyncio.sleep(2)
        
        # Test retrieval
        query = "warfarin dosage"
        print(f"Query: {query}")
        result = await client._arun(query)
        print(f"Result: {result}")
        
        # Test manual upload
        test_texts = [
            "Warfarin is an anticoagulant medication used to prevent blood clots.",
            "Dosage should be adjusted based on INR values and patient response.",
        ]
        upload_result = await client.upload_text_documents(test_texts, COLLECTION)
        print(f"Upload result: {upload_result}")
        
        # Test query again
        result2 = await client._arun("anticoagulant medication")
        print(f"Result after upload: {result2}")
        
        # Get status
        status = await client.get_status()
        print(f"Status: {json.dumps(status, indent=2)}")
        
        # Cleanup
        client.cleanup()
    
    asyncio.run(main())