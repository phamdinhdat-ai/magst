import time
import uuid
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector
from mcp.server.fastmcp import FastMCP
import os
from dotenv import load_dotenv
from os.path import join, dirname
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.memory import VectorStoreRetrieverMemory
from fastapi import FastAPI
from fastapi.responses import JSONResponse

# --- 1. SETUP ---

env_path = join(dirname(__file__), '.env')
load_dotenv(env_path)
DEVICE = os.getenv("DEVICE", "cpu")

# Initialize FastAPI for health checks
app = FastAPI()

@app.get("/health")
async def health_check():
    return JSONResponse({"status": "healthy", "service": "mcp-server"})

mcp = FastMCP("PostgreSQL-VDB", port=int(os.getenv("MCP_PORT", 50051)), host=os.getenv("MCP_HOST", "0.0.0.0"))

print("--- Initializing Embeddings Model (this may take a moment) ---")
print("Database URL: ", os.getenv("DB_URL"))
db_engine = create_async_engine(os.getenv("DB_URL"), echo=False, future=True)
embeddings = HuggingFaceEmbeddings(
    model_name=os.getenv("EMBEDDING_MODEL", "AITeamVN/Vietnamese_Embedding_v2"),
    model_kwargs={'device': DEVICE}, # or 'cpu'
    encode_kwargs={'normalize_embeddings': True}
)

reranker_device = DEVICE  # Change to "cuda:0" if you have GPU
reranker_model_kwargs = {'device': reranker_device}
reranker_model_path = os.getenv("RERANKING_MODEL", "AITeamVN/Vietnamese_Reranker")
reranker_model = HuggingFaceCrossEncoder(
    model_name=reranker_model_path, 
    model_kwargs=reranker_model_kwargs
)
#
print("Loading Reranker Model From Path:", reranker_model_path)
print("--- Embeddings and Reranker Models Initialized ---\n:", os.getenv("EMBEDDING_MODEL"))



AsyncSessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=db_engine,
    class_=AsyncSession,
    expire_on_commit=False
)


@mcp.tool(title="Retrieve data from vectordatabase with reranking")
async def retrieve_vectorstore_with_reranker(
    query: str, 
    collection: str, 
    initial_k: int = 20, 
    final_k: int = 5,
    use_reranker: bool = True
) -> str:
    """
    Retrieve data from vectordatabase given a query and collection name.
    Uses reranker to improve result quality.
    
    Args:
        query: Search query
        collection: Collection name in the database
        initial_k: Number of documents to retrieve from vector search (default: 20)
        final_k: Number of top documents to return after reranking (default: 5)
        use_reranker: Whether to use reranker or just return vector search results
    """
    print(f"Received query for collection '{collection}': '{query}'")
    start_time = time.time()
    
    try:
        # Initialize vectorstore
        vectorstore = PGVector(
            embeddings=embeddings,
            collection_name=collection,
            connection=db_engine,
            use_jsonb=True,
        )
        
        if not use_reranker:
            # Simple vector search without reranking
            results = await vectorstore.asimilarity_search(query, k=final_k)
            output = ""
            for i, doc in enumerate(results, 1):
                content = doc.page_content.replace("\n", " ")
                output += f"{i}. Content: {content}\n\n"
            
            retrieval_time = time.time() - start_time
            print(f"Vector search completed in {retrieval_time:.4f}s")
            return output if output else "No relevant documents found."
        
        # Create base retriever with higher k for reranking
        base_retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={'k': initial_k, 'lambda_mult': 0.25},
            return_source_documents=True
        )
        
        # Create compressor with reranker
        compressor = CrossEncoderReranker(
            model=reranker_model, 
            top_n=final_k
        )
        
        # Create the final compression retriever
        final_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )
        
        # Retrieve and rerank documents
        compressed_docs = await final_retriever.ainvoke(query)
        
        # Format output
        output = ""
        for i, doc in enumerate(compressed_docs, 1):
            content = doc.page_content.replace("\n", " ")
            # Include relevance score if available
            score_info = ""
            if hasattr(doc, 'metadata') and 'relevance_score' in doc.metadata:
                score_info = f" (Score: {doc.metadata['relevance_score']:.4f})"
            
            output += f"{i}. Content{score_info}: {content}\n\n"
        
        total_time = time.time() - start_time
        print(f"Retrieval with reranking completed in {total_time:.4f}s")
        
        return output if output else "No relevant documents found."
        
    except Exception as e:
        error_msg = f"Error during retrieval: {str(e)}"
        print(error_msg)
        return error_msg


@mcp.tool(title="Retrieve data from vectordatabase (simple)")
async def retrieve_vectorstore(query: str, collection: str) -> str:
    """
    Simple retrieval from vectordatabase without reranking (for backward compatibility).
    """
    return await retrieve_vectorstore_with_reranker(
        query=query, 
        collection=collection, 
        use_reranker=False
    )


# Optional: Tool to configure reranker settings
@mcp.tool(title="Configure retrieval settings")
async def configure_retrieval_settings(
    collection: str,
    initial_k: int = 20,
    final_k: int = 5,
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
) -> str:
    """
    Configure retrieval settings for a collection.
    
    Args:
        collection: Collection name
        initial_k: Number of documents for initial vector search
        final_k: Number of documents after reranking
        reranker_model: Reranker model to use
    """
    try:
        # Store settings (you might want to implement persistent storage)
        settings = {
            "collection": collection,
            "initial_k": initial_k,
            "final_k": final_k,
            "reranker_model": reranker_model
        }
        
        return f"Settings configured for collection '{collection}': {settings}"
        
    except Exception as e:
        return f"Error configuring settings: {str(e)}"
# --- NEW TOOL ADDED HERE ---



@mcp.tool(title="Ingest documents into a collection")
async def ingest_documents(documents: list[str], collection_name: str) -> str:
    if not documents:
        return "Error: No documents provided to ingest."

    print(f"Received request to ingest {len(documents)} documents into collection '{collection_name}'...")
    
    try:
        # 1. Create a PGVector instance configured for the target collection
        vectorstore = PGVector(
            embeddings=embeddings,
            collection_name=collection_name,
            connection=db_engine,
            use_jsonb=True,
        )

        # 2. Convert the list of strings to LangChain Document objects
        docs_to_add = [Document(page_content=doc) for doc in documents]
        
        # 3. Generate a unique ID for each document (best practice)
        ids = [str(uuid.uuid4()) for _ in docs_to_add]

        # 4. Add the documents to the vector store asynchronously
        await vectorstore.aadd_documents(docs_to_add, ids=ids)
        
        success_message = f"Successfully ingested {len(documents)} documents into collection '{collection_name}'."
        print(success_message)
        return success_message
        
    except Exception as e:
        error_message = f"Failed to ingest documents into collection '{collection_name}': {e}"
        print(error_message)
        return error_message


# This tool is just for demonstration from the original file, you can keep or remove it
@mcp.tool(title="File to documents")
async def file_to_documents(file_path: str, collection: str) -> str:
    """Convert a file to documents and add them to the vectordatabase."""
    with open(file_path, "r") as file:
        content = file.read()
    # Split content into individual documents (for simplicity, by newlines)
    docs = content.split("\n")
    return await ingest_documents(docs, collection_name=collection)




@mcp.tool(title="Memory Saver")
async def memory_saver(input: str, output: str, session_id: str) -> str:
    """
    Save a memory context.
    """
    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name=session_id,
        connection=db_engine,
        use_jsonb=True,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    memory = VectorStoreRetrieverMemory(retriever=retriever)
    await memory.asave_context({"input": input}, {"output": output})
    return "Memory context saved."

@mcp.tool(title="Memory Loader")
async def memory_loader(query: str, session_id: str) -> str:
    """
    Load memory contexts relevant to the query.
    """
    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name=session_id,
        connection=db_engine,
        use_jsonb=True,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    memory = VectorStoreRetrieverMemory(retriever=retriever)
    docs = await memory.aload_memory_variables({"input": query})
    return str(docs)






if __name__ == "__main__":
   mcp.run(transport="sse")