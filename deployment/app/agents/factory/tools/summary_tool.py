# --- Tool Factory Class ---
import re
import os
import sys
import json
import time
import chromadb
from typing import Optional, List, Tuple, Dict, Any, Callable
from langchain_core.documents import Document
from loguru import logger
import asyncio
from pydantic import Field
from langchain_community.vectorstores import Chroma
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.summarize.chain import load_summarize_chain
try:
    from langchain_ollama import OllamaEmbeddings
except ImportError:
    from langchain_community.embeddings import OllamaEmbeddings
# --- LangChain Core & Community Imports ---
import re
from pydantic import Field, PrivateAttr
# --- Tool Imports ---
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from app.agents.workflow.state import GraphState as AgentState
from app.agents.factory.tools.base import BaseAgentTool
from app.core.config import get_settings
from app.agents.workflow.initalize import llm_instance
settings = get_settings()



class SummaryTool(BaseAgentTool):
    """Tool for summarizing text using a language model."""
    # Define Pydantic fields with proper type annotations
    name: str = Field(default="TextSummarizer", description="Name of the summarization tool")
    description: str = Field(default="This tool summarizes the provided text using a language model.", description="Description of what the tool does")
    llm: BaseChatModel = Field(default=None, description="Language model instance for summarization")
    collection_name: str = Field(default=None, description="Name of the collection in the vector store")
    _vector_store: Chroma = PrivateAttr(default=None)
    _embeddings: OllamaEmbeddings = PrivateAttr(default=None)

    def __init__(self, llm: BaseChatModel, collection_name: str, **kwargs):
        # Initialize with proper field values
        kwargs.setdefault('name', 'TextSummarizer')
        kwargs.setdefault('description', 'This tool summarizes the provided text using a language model.')
        kwargs['llm'] = llm
        kwargs['collection_name'] = collection_name

        super().__init__(**kwargs)
        self._vector_store: Optional[Chroma] = None
        self._embeddings: Optional[OllamaEmbeddings] = None
        self._initialized = False
        
        if not self._initialized:
            self._initialize_vector_store()
            self._initialized = True


    def _initialize_vector_store(self) -> None:
        """Initialize the vector store and embeddings."""
        logger.info(f"Collection name for SummaryTool: {self.collection_name}")
        self._embeddings = OllamaEmbeddings(model=settings.EMBEDDING_MODEL, base_url=settings.OLLAMA_BASE_URL)
        persistent_client = chromadb.PersistentClient(path=str(Path(settings.VECTOR_STORE_BASE_DIR)))
        self._vector_store = Chroma(client=persistent_client, collection_name=self.collection_name, embedding_function=self._embeddings)

    def _run(self, text: str) -> str:
        """Synchronous summarization execution for LangChain compatibility."""
        return self.summarize_text(text)
    
    async def _arun(self, text: str) -> str:
        """Asynchronous summarization execution for LangChain compatibility."""
        return await self.arun_impl(text)
    
    async def arun_impl(self, text: str) -> str:
        """Asynchronous implementation of the summarization logic."""
        logger.info(f"Running SummaryTool with text of length {len(text)} characters")
        return self.get_vectorstore_summary()  # Note: making this sync for now, can be made async later

    def summarize_text(self, text: str) -> str:
        """Summarize the provided text using the language model."""
        if not text or not isinstance(text, str):
            return "Error: Invalid input text."
        
        # Prepare the prompt for summarization
        system_prompt = """
        Bạn là một trợ lý Genee rất hiệu quả và nhanh nhẹn. Bạn có khả năng tổng hợp dữ liệu một cách chính xác và đơn giản.
        Dựa theo thông tin mà nguời dùng cung cấp, hãy tóm tắt lượng thông tin đó ngắn ngọn và đảm bảo các thông tin quan trọng được đề cập.
        Hãy tóm tắt và mô tả theo dạng items dưới đây: 
        I. <tên mục>
        1. <thong tin 1>
        2. <thong tin 2>
        3. <thong tin 3>
        ...
        II. <tên mục khác nếu có>
        1. <thong tin 1>
        2. <thong tin 2>
        3. <thong tin 3>
        ...
        Nếu không có thông tin nào, hãy trả lời "Không có thông tin nào để tóm tắt."
        #LUU Ý: 
        - Hãy đảm bảo rằng bạn chỉ tóm tắt các thông tin quan trọng và liên quan nhất.
        - Tóm tắt không quá 100 từ.
        - Trả lời ngắn gọn, súc tích và dễ hiểu.
        - Trả lời bằng tiếng Việt.
        """
        
        query = """Dựa trên văn bản sau, hãy tóm tắt các thông tin quan trọng:
        
        {text}
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", query)
        ])
        # logger.info(f"Prompt for summarization: {prompt}")
        # logger.info(f"summarize text : {text[:100]}...")  # Log first 100 characters for debugging
        # Use the language model to generate a summary
        try:
            chain = prompt | self.llm
            response = chain.invoke({
                "text": text
            })
            summary = response.content
            return summary if summary else "Tóm tắt không thành công. Vui lòng thử lại."
        except Exception as e:
            logger.error(f"Error during summarization: {e}")
            return f"Error during summarization: {str(e)}"

    def get_all_data_from_vectorstore(self) -> List[str]:
        """Retrieve all documents from the vector store."""
        if not self._vector_store:
            logger.warning("No vector store available to retrieve documents.")
            return []
        logger.info("Retrieving all documents from the vector store...")
        try:
            all_documents = self._vector_store.get(include=["metadatas", "documents"], limit=100)

            logger.debug(f"Retrieved {len(all_documents)} documents from vector store.")
            logger.debug(f"Sample documents: {all_documents['documents'][:3] if len(all_documents['documents']) > 3 else all_documents['documents']}")
            if not all_documents:
                logger.warning("No documents found in vector store.")
                return []
            return all_documents
        except Exception as e:
            logger.error(f"Error retrieving documents from vector store: {str(e)}")
            return []
        
    def get_vectorstore_summary(self) -> str:
        """Generate a summary of all documents in the vector store."""
        all_docs = self.get_all_data_from_vectorstore()
        logger.info(f"Retrieved {len(all_docs)} documents from vector store.")
        if not all_docs:
            logger.warning("No documents found in the vector store.")
            return "No documents found in the vector store."
        
        # Join all document texts into a single string for summarization
        logger.info(f"Combining {len(all_docs['documents'])} documents for summarization...")
        combined_text = "\n\n".join(all_docs['documents']).strip()
        if len(combined_text) == 0:
            return "No content available to summarize from the vector store."   
        if len(combined_text) > 10000:
            summary_text = ''
            for i in range(0, len(combined_text), 10000):
                chunk = combined_text[i:i+10000]
                summary_chunk = self.summarize_text(chunk)
                summary_text += summary_chunk + "\n\n"
                logger.info(f"Processed chunk {i//10000 + 1}: {len(chunk)} characters")
            logger.info(f"Combined text length exceeded 10000 characters, summarized in chunks.")
            return summary_text.strip()
            
        # Log the length of the combined text
        logger.info(f"Combined text length for summarization: {len(combined_text)} characters")
        # Use the summarize_text method to generate a summary
        return self.summarize_text(combined_text)
    
    def get_summary(self, text: str) -> str:
        """Public method to get a summary of the provided text."""
        if len(text) == 0:
            return "No text provided for summarization."
        if len(text) > 12000:
            text = text[:12000]
        return self.summarize_text(text)

    def run(self, text: str) -> str:
        """Run the summarization tool synchronously."""
        logger.info(f"Running SummaryTool with text of length {len(text)} characters")
        return self.summarize_text(text)
    
    async def arun(self, text: str) -> str:
        """Run the summarization tool asynchronously."""
        logger.info(f"Running SummaryTool asynchronously with text of length {len(text)} characters")
        return await self._arun(text)



    async def summarize_chain(self, text: str) -> str:
        """Asynchronous method to summarize text using a chain."""
        logger.info(f"Running asynchronous summarization chain with text of length {len(text)} characters")
        if not self._vector_store:
            logger.warning("Vector store not initialized, cannot run async summarization chain.")
            return "Vector store not initialized."
        
        # Load the summarize chain
        all_documents = self._vector_store.get(include=["metadatas", "documents"], limit=100)
        logger.info(f"Retrieved {len(all_documents['documents'])} documents from vector store for summarization.")
        if not all_documents or 'documents' not in all_documents or len(all_documents['documents']) == 0:
            logger.warning("No documents found in vector store for summarization.")
            return "Khong có tài liệu nào để tóm tắt từ kho dữ liệu."
        all_chunks = [Document(page_content=doc) for doc, metadata in zip(all_documents['documents'], all_documents['metadatas']) if doc and metadata]
        if not all_chunks:
            logger.warning("No valid document chunks found for summarization.")
            return "Khong có đoạn tài liệu hợp lệ nào để tóm tắt."

        try:
            chain = load_summarize_chain(self.llm, chain_type="map_reduce")
            summary = await chain.arun(all_chunks)
            return summary
        except Exception as e:
            logger.error(f"Error during asynchronous summarization: {e}")
            return f"Loi khi tóm tắt: {str(e)}"
if __name__ == "__main__":
    print("🔧 Testing SummaryTool...")
    
    # Example usage of the SummaryTool
    llm = llm_instance
    print(f"✓ LLM loaded: {type(llm)}")
    
    # Try to create embeddings with proper parameters
    try:
        embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")
        print("✓ OllamaEmbeddings initialized successfully")
    except Exception as e:
        print(f"⚠️ Warning: Could not initialize OllamaEmbeddings: {e}")
        embeddings = None
    
    # Create vectorstore only if embeddings are available and vectorstore path exists
    vectorstore = None
    if embeddings and hasattr(settings, 'vectorstore_path') and settings.vectorstore_path:
        try:
            vectorstore = Chroma(
                collection_name="example_collection",
                embedding_function=embeddings,
                persist_directory=settings.vectorstore_path
            )
            print("✓ Chroma vectorstore initialized successfully")
        except Exception as e:
            print(f"⚠️ Warning: Could not initialize Chroma vectorstore: {e}")
            vectorstore = None
    else:
        print("⚠️ Skipping vectorstore initialization (no embeddings or path)")
    
    # Create the summary tool
    try:
        summary_tool = SummaryTool(llm=llm, vectorstore=vectorstore)
        print("✓ SummaryTool initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize SummaryTool: {e}")
        sys.exit(1)
    
    # Example text to summarize (Vietnamese text for testing)
    example_text = """
    GeneStory là một công ty công nghệ sinh học hàng đầu tại Việt Nam, chuyên cung cấp các dịch vụ xét nghiệm di truyền và phân tích gen.
    Công ty được thành lập với sứ mệnh mang lại những giải pháp y tế cá nhân hóa dựa trên thông tin di truyền của từng cá nhân.
    Các sản phẩm chính của GeneStory bao gồm: xét nghiệm di truyền để dự đoán nguy cơ mắc bệnh, xét nghiệm dược lý di truyền để tối ưu hóa việc sử dụng thuốc, và xét nghiệm di truyền về dinh dưỡng và thể thao.
    Công ty sử dụng công nghệ tiên tiến và có đội ngũ chuyên gia giàu kinh nghiệm trong lĩnh vực sinh học phân tử và di truyền học.
    """
    
    print("\n📄 Testing text summarization...")
    print(f"Input text length: {len(example_text)} characters")
    
    # Run the summarization
    try:
        summary = summary_tool.summarize_text(example_text)
        print("✓ Summarization completed successfully!")
        print("\n📋 Summary Result:")
        print("=" * 50)
        print(summary)
        print("=" * 50)
    except Exception as e:
        print(f"✗ Summarization failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test vectorstore summary if available
    if vectorstore:
        print("\n📚 Testing vectorstore summary...")
        try:
            vectorstore_summary = summary_tool.get_vectorstore_summary()
            print("✓ Vectorstore summary completed!")
            print("\n📋 Vectorstore Summary:")
            print("=" * 50)
            print(vectorstore_summary)
            print("=" * 50)
        except Exception as e:
            print(f"⚠️ Vectorstore summary failed: {e}")
    else:
        print("\n⚠️ Skipping vectorstore summary test (no vectorstore available)")
    
    print("\n🎉 SummaryTool testing completed!")

