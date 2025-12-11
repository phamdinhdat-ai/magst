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
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, JSONLoader, TextLoader, PyMuPDFLoader

# --- MCP Client Imports ---
from mcp.client.sse import sse_client
from mcp.client.session import ClientSession

# --- Base Agent Tool Import ---
from app.agents.factory.tools.base import BaseAgentTool  # Adjust import path as needed
from app.utils.document_processor import markdown_splitter, remove_image_tags, DocumentCustomConverter
from app.core.config import settings

class DocumentWatcher(FileSystemEventHandler):
    """File system watcher that triggers document uploads to MCP server."""
    
    def __init__(self, retriever_tool: 'MedicalRetrieverMCPClient'):
        self.tool = retriever_tool

    def on_created(self, event):
        if not event.is_directory:
            logger.info(f"[Medical Watcher] New file detected: {event.src_path}")
            time.sleep(1)  # Small delay to ensure file is fully written
            asyncio.create_task(self.tool._process_file_if_needed(Path(event.src_path)))

    def on_modified(self, event):
        if not event.is_directory and "_registry.json" not in event.src_path:
            logger.info(f"[Medical Watcher] File modified: {event.src_path}")
            time.sleep(1)
            asyncio.create_task(self.tool._process_file_if_needed(Path(event.src_path)))
            
    def on_deleted(self, event):
        if not event.is_directory:
            logger.info(f"[Medical Watcher] File deleted: {event.src_path}")
            # Note: For deletion, we'd need a delete tool on the server side
            # For now, we'll just update our local registry
            asyncio.create_task(self.tool._remove_from_registry(Path(event.src_path)))


class MedicalRetrieverInput(BaseModel):
    """Input schema for medical document retrieval."""
    query: str = PydanticField(description="Search query for finding relevant medical documents and healthcare information")
    collection_name: str = PydanticField(description="Name of the medical document collection to search in")
    max_results: int = PydanticField(default=5, description="Maximum number of results to return")


class MedicalRetrieverMCPClient(BaseAgentTool):
    """
    Tool to retrieve medical documents from a remote vector database via MCP server.
    Specialized for medical and healthcare information retrieval with Vietnamese language support.
    """
    
    name: str = "medical_retriever_mcp"
    description: str = "Retrieves medical and healthcare documents from remote vector database and manages medical document uploads"
    args_schema: type[BaseModel] = MedicalRetrieverInput
    
    # Configuration
    mcp_server_url: str = PydanticField(description="URL of the MCP server")
    watch_directory: Path = PydanticField(description="Directory to watch for medical document changes")
    default_collection: str = PydanticField(default="medical_docs", description="Default collection name for medical documents")
    
    # Internal components
    _document_registry: Dict[str, str] = {}
    _observer: Optional[Observer] = None
    _text_splitter: Optional[RecursiveCharacterTextSplitter] = None
    _thread_pool: Optional[ThreadPoolExecutor] = None
    _is_initialized: bool = False
    
    def __init__(self, 
                 mcp_server_url: str,
                 watch_directory: str,
                 default_collection: str = "medical_docs",
                 **kwargs):
        """
        Initialize the Medical MCP client tool.
        
        Args:
            mcp_server_url: URL of the MCP server (e.g., "http://localhost:50051/sse")
            watch_directory: Local directory to watch for medical document changes
            default_collection: Default collection name for medical documents
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
        logger.info(f"Initializing MedicalRetrieverMCPClient...")
        
        # Ensure watch directory exists
        self.watch_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize document processing components optimized for medical content
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.OVERLAP_SIZE,
            is_separator_regex=True,
            # Medical-specific separators including Vietnamese medical terms
            separators=[
                "\n\n", "\n", 
                ". ", " ", "",
                # Vietnamese medical section markers
                "Triệu chứng:", "Chẩn đoán:", "Điều trị:", "Nguyên nhân:",
                "Symptoms:", "Diagnosis:", "Treatment:", "Causes:"
            ]
        )
        
        # Initialize thread pool for parallel processing
        self._thread_pool = ThreadPoolExecutor(
            max_workers=settings.MAX_WORKERS,
            thread_name_prefix="Medical_mcp_client"
        )
        
        # Load document registry
        await self._load_document_registry()
        
        # Test MCP server connection
        if await self._test_server_connection():
            logger.info("Medical MCP server connection successful")
        else:
            logger.warning("Medical MCP server connection failed - some features may not work")
        
        # Scan and process existing files
        await self._scan_and_process_all_files()
        
        # Start document watcher
        self._start_document_watcher()
        
        self._is_initialized = True
        logger.info("MedicalRetrieverMCPClient initialized successfully")
    
    async def _test_server_connection(self) -> bool:
        """Test connection to Medical MCP server."""
        try:
            async with sse_client(url=self.mcp_server_url) as streams:
                async with ClientSession(read_stream=streams[0], write_stream=streams[1]) as session:
                    await session.initialize()
                    logger.debug("Medical MCP server connection test successful")
                    return True
        except Exception as e:
            logger.error(f"Medical MCP server connection test failed: {e}")
            return False
    
    @property
    def _registry_path(self) -> Path:
        """Path to the medical document registry file."""
        return self.watch_directory / f"{self.default_collection}_mcp_registry.json"
    
    async def _load_document_registry(self):
        """Load the medical document registry from disk."""
        if self._registry_path.exists():
            try:
                with open(self._registry_path, 'r', encoding='utf-8') as f:
                    self._document_registry = json.load(f)
                logger.info(f"Loaded {len(self._document_registry)} medical document entries from registry")
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Failed to load medical registry: {e}. Starting fresh.")
                self._document_registry = {}
        else:
            self._document_registry = {}
    
    async def _save_document_registry(self):
        """Save the medical document registry to disk."""
        try:
            with open(self._registry_path, 'w', encoding='utf-8') as f:
                json.dump(self._document_registry, f, indent=2, ensure_ascii=False)
            logger.debug(f"Medical registry saved with {len(self._document_registry)} entries")
        except IOError as e:
            logger.error(f"Failed to save medical registry: {e}")
    
    def _get_file_hash(self, file_path: Path) -> Optional[str]:
        """Calculate MD5 hash of a medical document file."""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except IOError as e:
            logger.error(f"Could not read medical file for hashing: {file_path}. Error: {e}")
            return None
    
    def _load_and_split_file(self, file_path: Path) -> List[Document]:
        """Load and split a medical document file into document chunks."""
        loader_map = {
            '.pdf': PyMuPDFLoader,  # Use custom converter for medical PDFs
            '.csv': CSVLoader,
            '.json': JSONLoader,
            '.txt': TextLoader
        }
        
        loader_class = loader_map.get(file_path.suffix.lower())
        if not loader_class:
            logger.warning(f"Unsupported medical file type: {file_path.suffix}")
            return []
        
        try:
            loader = loader_class(str(file_path))
            raw_docs = loader.load()
            
            # # Special processing for medical PDFs
            # if "pdf" in file_path.suffix.lower():
            #     logger.info(f"Processing medical PDF file: {file_path.name}")
            #     # Clean and split markdown content from PDF
            #     cleaned_text = remove_image_tags(raw_docs[0])
            #     logger.info(f"Cleaned medical text length: {len(cleaned_text)} characters")
            #     try: 
            #         raw_docs = markdown_splitter(cleaned_text)
            #     except Exception as e:
            #         logger.error(f"Error splitting markdown from medical PDF file {file_path.name}: {e}")
            #     logger.info(f"Split {len(raw_docs)} sections from medical PDF file: {file_path.name}")
            
            # Split documents into chunks
            split_docs = self._text_splitter.split_documents(raw_docs)
            
            # Add source metadata with medical-specific fields
            for doc in split_docs:
                doc.metadata['source'] = file_path.name
                doc.metadata['file_path'] = str(file_path)
                doc.metadata['timestamp'] = time.time()
                doc.metadata['document_type'] = 'medical'
                doc.metadata['collection'] = self.default_collection
                
                # Extract medical-specific metadata if possible
                content_lower = doc.page_content.lower()
                
                # Detect medical categories (Vietnamese and English)
                medical_categories = {
                    'disease': ['bệnh', 'disease', 'disorder', 'syndrome'],
                    'symptom': ['triệu chứng', 'symptom', 'sign'],
                    'treatment': ['điều trị', 'treatment', 'therapy', 'medicine'],
                    'diagnosis': ['chẩn đoán', 'diagnosis', 'examination'],
                    'prevention': ['phòng ngừa', 'prevention', 'prophylaxis']
                }
                
                detected_categories = []
                for category, keywords in medical_categories.items():
                    if any(keyword in content_lower for keyword in keywords):
                        detected_categories.append(category)
                
                if detected_categories:
                    doc.metadata['medical_categories'] = detected_categories
            
            logger.info(f"Loaded and split medical file {file_path.name} into {len(split_docs)} chunks")
            return split_docs
            
        except Exception as e:
            logger.error(f"Error loading medical file {file_path}: {e}")
            return []

    async def _upload_documents_to_server(self, documents: List[Document], collection_name: str) -> bool:
        """Upload medical document chunks to the MCP server."""
        if not documents:
            return False
        
        try:
            # Convert documents to list of strings (content only)
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
                            "metadata": doc_metadata,
                            "collection_name": collection_name
                        }
                    )
                    
                    if response and response.content:
                        result_text = response.content[0].text
                        logger.info(f"Medical server response: {result_text}")
                        return "successfully" in result_text.lower()
                    else:
                        logger.error("No response from medical MCP server")
                        return False
                        
        except Exception as e:
            logger.error(f"Error uploading medical documents to server: {e}")
            return False
    
    async def _process_file_if_needed(self, file_path: Path):
        """Process a medical file if it has changed since last processing."""
        if "_registry.json" in file_path.name or "_mcp_registry.json" in file_path.name:
            return
        
        current_hash = self._get_file_hash(file_path)
        if not current_hash:
            return
        
        stored_hash = self._document_registry.get(str(file_path))
        if current_hash != stored_hash:
            logger.info(f"Change detected for medical file '{file_path.name}'. Processing...")
            
            # Load and split the medical file
            documents = self._load_and_split_file(file_path)
            
            if documents:
                # Upload to server
                success = await self._upload_documents_to_server(documents, self.default_collection)
                
                if success:
                    # Update registry
                    self._document_registry[str(file_path)] = current_hash
                    await self._save_document_registry()
                    logger.info(f"Successfully processed and uploaded medical file '{file_path.name}'")
                else:
                    logger.error(f"Failed to upload medical file '{file_path.name}' to server")
            else:
                logger.warning(f"No medical documents extracted from '{file_path.name}'")
    
    async def _scan_and_process_all_files(self):
        """Scan directory and process all supported medical files."""
        logger.info(f"Scanning medical directory: {self.watch_directory}")
        
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
        """Remove a medical file from the registry."""
        path_str = str(file_path)
        if path_str in self._document_registry:
            logger.info(f"Removing medical file '{file_path.name}' from registry")
            del self._document_registry[path_str]
            await self._save_document_registry()
    
    def _start_document_watcher(self):
        """Start watching the directory for medical file changes."""
        if self._observer:
            return
        
        self._observer = Observer()
        event_handler = DocumentWatcher(self)
        self._observer.schedule(event_handler, str(self.watch_directory), recursive=True)
        
        watcher_thread = threading.Thread(target=self._observer.start, daemon=True)
        watcher_thread.start()
        logger.info(f"Started medical document watcher on '{self.watch_directory}'")
    
    async def _retrieve_from_server(self, query: str, collection_name: str, k: int = 5) -> str:
        """Retrieve medical documents from the MCP server."""
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
                        logger.info(f"Retrieved medical documents for query: '{query[:50]}...'")
                        return result_text
                    else:
                        return "No response from medical MCP server"
                        
        except ConnectionRefusedError:
            error_msg = f"Connection refused. Is the medical MCP server running at {self.mcp_server_url}?"
            logger.error(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"Error retrieving medical documents from server: {e}"
            logger.error(error_msg)
            return error_msg
    
    def _run(self, query: str, collection_name: str = None, max_results: int = 5) -> str:
        """Synchronous medical document retrieval."""
        collection = collection_name or self.default_collection
        return asyncio.run(self._arun(query, collection, max_results))
    
    async def _arun(self, query: str, collection_name: str = None, max_results: int = 5) -> str:
        """Asynchronous medical document retrieval."""
        if not self._is_initialized:
            return "Error: Medical retriever client is not initialized."
        
        collection = collection_name or self.default_collection
        logger.info(f"Retrieving medical documents for query: '{query}' from collection: '{collection}'")
        
        start_time = time.time()
        result = await self._retrieve_from_server(query, collection, max_results)
        retrieval_time = time.time() - start_time
        
        logger.info(f"Medical document retrieval completed in {retrieval_time:.2f}s")
        return result
    
    # Public methods for manual operations
    async def upload_file(self, file_path: str, collection_name: str = None) -> str:
        """Manually upload a specific medical file to the server."""
        collection = collection_name or self.default_collection
        path = Path(file_path)
        
        if not path.exists():
            return f"Error: Medical file not found: {file_path}"
        
        documents = self._load_and_split_file(path)
        if not documents:
            return f"Error: Could not process medical file: {file_path}"
        
        success = await self._upload_documents_to_server(documents, collection)
        if success:
            # Update registry
            file_hash = self._get_file_hash(path)
            if file_hash:
                self._document_registry[str(path)] = file_hash
                await self._save_document_registry()
            return f"Successfully uploaded {len(documents)} medical document chunks from {file_path}"
        else:
            return f"Failed to upload medical file: {file_path}"
    
    async def upload_text_documents(self, texts: List[str], collection_name: str = None, source_name: str = "manual_upload") -> str:
        """Manually upload medical text documents to the server."""
        collection = collection_name or self.default_collection
        
        if not texts:
            return "Error: No medical texts provided"
        
        # Create Document objects with medical metadata
        documents = []
        for i, text in enumerate(texts):
            doc = Document(
                page_content=text,
                metadata={
                    'source': f"{source_name}_{i}",
                    'timestamp': time.time(),
                    'document_type': 'medical',
                    'collection': collection,
                    'manual_upload': True,
                    'language': 'vietnamese' if any(char in text for char in 'àáảãạăắằẳẵặâấầẩẫậđèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ') else 'english'
                }
            )
            documents.append(doc)
        
        success = await self._upload_documents_to_server(documents, collection)
        if success:
            return f"Successfully uploaded {len(texts)} medical text documents to collection '{collection}'"
        else:
            return f"Failed to upload medical text documents to collection '{collection}'"
    
    async def upload_medical_case(self, 
                                patient_info: str, 
                                symptoms: str, 
                                diagnosis: str, 
                                treatment: str,
                                collection_name: str = None) -> str:
        """Upload a structured medical case to the server."""
        collection = collection_name or self.default_collection
        
        # Create a structured medical case document
        case_content = f"""
        Thông tin bệnh nhân / Patient Information:
        {patient_info}
        
        Triệu chứng / Symptoms:
        {symptoms}
        
        Chẩn đoán / Diagnosis:
        {diagnosis}
        
        Điều trị / Treatment:
        {treatment}
        """
        
        doc = Document(
            page_content=case_content.strip(),
            metadata={
                'source': f'medical_case_{int(time.time())}',
                'timestamp': time.time(),
                'document_type': 'medical_case',
                'collection': collection,
                'medical_categories': ['diagnosis', 'treatment', 'symptom'],
                'structured_case': True
            }
        )
        
        success = await self._upload_documents_to_server([doc], collection)
        if success:
            return f"Successfully uploaded medical case to collection '{collection}'"
        else:
            return f"Failed to upload medical case to collection '{collection}'"
    
    async def get_status(self) -> Dict[str, Any]:
        """Get the current status of the medical client."""
        status = {
            "initialized": self._is_initialized,
            "mcp_server_url": self.mcp_server_url,
            "watch_directory": str(self.watch_directory),
            "default_collection": self.default_collection,
            "tracked_medical_files": len(self._document_registry),
            "server_connected": await self._test_server_connection(),
            "document_type": "medical_healthcare",
            "supported_languages": ["vietnamese", "english"]
        }
        return status
    
    def cleanup(self):
        """Clean up medical retriever resources."""
        logger.info("Cleaning up MedicalRetrieverMCPClient resources...")
        
        if self._observer and self._observer.is_alive():
            self._observer.stop()
            self._observer.join()
            logger.info("Medical document watcher stopped")
        
        if self._thread_pool:
            self._thread_pool.shutdown(wait=False)
            logger.info("Medical thread pool shut down")
    
    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.cleanup()
        except Exception as e:
            logger.error(f"Error during medical retriever cleanup: {e}")


# Factory functions for easy instantiation
def create_medical_retriever_client(mcp_server_url: str, 
                                   watch_directory: str,
                                   collection_name: str = "medical_docs") -> MedicalRetrieverMCPClient:
    """
    Factory function to create a MedicalRetrieverMCPClient instance.
    
    Args:
        mcp_server_url: URL of the MCP server (e.g., "http://localhost:50051/sse")
        watch_directory: Local directory to watch for medical documents
        collection_name: Collection name for the medical vector database
    
    Returns:
        Configured MedicalRetrieverMCPClient instance
    """
    return MedicalRetrieverMCPClient(
        mcp_server_url=mcp_server_url,
        watch_directory=watch_directory,
        default_collection=collection_name
    )


# Example usage
if __name__ == "__main__":
    async def main():
        # Configuration
        SERVER_URL = "http://192.168.1.60:50051/sse"
        WATCH_DIR = "app/agents/retrievers/storages/medical"
        COLLECTION = "medical_knowledge"
        
        # Create the medical client
        client = create_medical_retriever_client(
            mcp_server_url=SERVER_URL,
            watch_directory=WATCH_DIR,
            collection_name=COLLECTION
        )
        
        # Wait for initialization
        await asyncio.sleep(2)
        
        # Test retrieval - Vietnamese medical query
        query = "triệu chứng bệnh tiểu đường"
        print(f"Query: {query}")
        result = await client._arun(query)
        print(f"Result: {result}")
        
        # Test manual upload of medical texts
        test_medical_texts = [
            "Bệnh tiểu đường (diabetes) là một bệnh rối loạn chuyển hóa. Triệu chứng bao gồm khát nước và đi tiểu nhiều.",
            "Tăng huyết áp là tình trạng áp lực máu lên thành động mạch cao hơn mức bình thường. Có thể gây đột quỵ và bệnh tim.",
            "Hypertension is a condition where blood pressure in the arteries is persistently elevated. It can lead to stroke and heart disease.",
            "COVID-19 symptoms include fever, dry cough, and difficulty breathing. Severe cases may require hospitalization."
        ]
        upload_result = await client.upload_text_documents(test_medical_texts, COLLECTION)
        print(f"Upload result: {upload_result}")
        
        # Test structured medical case upload
        case_result = await client.upload_medical_case(
            patient_info="Nam, 45 tuổi, có tiền sử hút thuốc",
            symptoms="Đau ngực, khó thở khi gắng sức, mệt mỏi",
            diagnosis="Bệnh mạch vành, thiếu máu cơ tim",
            treatment="Thuốc chống đông máu, thay đổi lối sống, theo dõi định kỳ"
        )
        print(f"Medical case upload: {case_result}")
        
        # Test queries after upload
        result2 = await client._arun("bệnh tiểu đường triệu chứng")
        print(f"Vietnamese query result: {result2}")
        
        result3 = await client._arun("hypertension symptoms treatment")
        print(f"English query result: {result3}")
        
        # Get status
        status = await client.get_status()
        print(f"Status: {json.dumps(status, indent=2, ensure_ascii=False)}")
        
        # Cleanup
        client.cleanup()
    
    asyncio.run(main())