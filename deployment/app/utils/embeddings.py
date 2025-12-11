"""
Embeddings utilities for the RAG system
This module handles vector embeddings for document chunks
"""
import os
import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Union
import aiohttp
import asyncio
import time
from pathlib import Path
from datetime import datetime

# Setup logger
logger = logging.getLogger(__name__)

# Embeddings settings
EMBEDDING_DIMENSION = 1536  # For OpenAI ada-002 compatibility
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDINGS_API_URL = os.getenv("EMBEDDINGS_API_URL", "http://localhost:11434/api/embeddings")

async def get_embedding(text: str, model: str = DEFAULT_EMBEDDING_MODEL) -> List[float]:
    """
    Get embeddings for text using the configured embeddings API
    
    Args:
        text: Text to embed
        model: Embedding model to use
        
    Returns:
        Vector embedding as list of floats
    """
    if not text or text.strip() == "":
        # Return zero vector for empty text
        return [0.0] * EMBEDDING_DIMENSION
        
    try:
        # First try Ollama API if available
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    EMBEDDINGS_API_URL,
                    json={"model": model, "prompt": text},
                    timeout=30
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        if "embedding" in result:
                            return result["embedding"]
            except Exception as e:
                logger.warning(f"Error using Ollama embeddings API: {str(e)}")
        
        # Fall back to SentenceTransformers if available
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(model)
            embedding = model.encode(text)
            return embedding.tolist()
        except ImportError:
            logger.warning("SentenceTransformers not installed")
        
        # Last resort - return random embedding for testing (not for production)
        logger.warning("Using random embeddings as fallback (for testing only)")
        import random
        return [random.uniform(-1, 1) for _ in range(EMBEDDING_DIMENSION)]
        
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        # Return zero vector for errors
        return [0.0] * EMBEDDING_DIMENSION

def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    if not vec_a or not vec_b:
        return 0.0
        
    vec_a = np.array(vec_a)
    vec_b = np.array(vec_b)
    
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
        
    return dot_product / (norm_a * norm_b)

async def process_document_embeddings(document_path: str, chunks: List[str]) -> bool:
    """
    Process document chunks to generate and store embeddings
    
    Args:
        document_path: Path to document vector store directory
        chunks: List of text chunks to process
        
    Returns:
        Success status
    """
    try:
        # Create embeddings directory
        embeddings_dir = os.path.join(document_path, "embeddings")
        os.makedirs(embeddings_dir, exist_ok=True)
        
        # Process each chunk
        for i, chunk in enumerate(chunks):
            # Load chunk data
            chunk_file = os.path.join(document_path, f"chunk_{i}.json")
            if not os.path.exists(chunk_file):
                continue
                
            with open(chunk_file, "r", encoding="utf-8") as f:
                chunk_data = json.load(f)
            
            # Generate embedding
            embedding = await get_embedding(chunk)
            
            # Save embedding
            embedding_file = os.path.join(embeddings_dir, f"chunk_{i}.json")
            with open(embedding_file, "w", encoding="utf-8") as f:
                json.dump({
                    "chunk_id": f"{chunk_data['metadata']['document_id']}_{i}",
                    "embedding": embedding,
                    "metadata": chunk_data["metadata"],
                    "created_at": str(datetime.now())
                }, f)
        
        # Update index file with embeddings info
        index_file = os.path.join(document_path, "index.json")
        if os.path.exists(index_file):
            with open(index_file, "r", encoding="utf-8") as f:
                index_data = json.load(f)
                
            index_data["embeddings_processed"] = True
            index_data["embeddings_count"] = len(chunks)
            index_data["embeddings_updated_at"] = str(datetime.now())
            
            with open(index_file, "w", encoding="utf-8") as f:
                json.dump(index_data, f, ensure_ascii=False)
        
        return True
    except Exception as e:
        logger.error(f"Error processing document embeddings: {str(e)}")
        return False

async def search_embeddings(
    query: str, 
    document_dirs: List[str],
    limit: int = 5
) -> List[Dict[str, Any]]:
    """
    Search for relevant chunks using vector similarity
    
    Args:
        query: Query text
        document_dirs: List of document directories to search in
        limit: Maximum number of results
        
    Returns:
        List of relevant chunks with metadata and similarity scores
    """
    try:
        # Generate embedding for query
        query_embedding = await get_embedding(query)
        
        # Find all embedding files across the provided document directories
        all_results = []
        
        for doc_dir in document_dirs:
            embeddings_dir = os.path.join(doc_dir, "embeddings")
            if not os.path.exists(embeddings_dir):
                continue
                
            # Load all embeddings from this document
            for filename in os.listdir(embeddings_dir):
                if not filename.endswith(".json"):
                    continue
                    
                embedding_file = os.path.join(embeddings_dir, filename)
                with open(embedding_file, "r", encoding="utf-8") as f:
                    embedding_data = json.load(f)
                
                # Calculate similarity
                chunk_embedding = embedding_data.get("embedding", [])
                similarity = cosine_similarity(query_embedding, chunk_embedding)
                
                # Get the original chunk text
                chunk_id = embedding_data.get("metadata", {}).get("chunk_index", 0)
                chunk_file = os.path.join(doc_dir, f"chunk_{chunk_id}.json")
                
                if os.path.exists(chunk_file):
                    with open(chunk_file, "r", encoding="utf-8") as f:
                        chunk_data = json.load(f)
                        
                    # Add result with similarity score
                    all_results.append({
                        "text": chunk_data.get("text", ""),
                        "metadata": embedding_data.get("metadata", {}),
                        "similarity": similarity
                    })
        
        # Sort by similarity and return top results
        all_results.sort(key=lambda x: x["similarity"], reverse=True)
        return all_results[:limit]
        
    except Exception as e:
        logger.error(f"Error searching embeddings: {str(e)}")
        return []

async def delete_document_embeddings(document_id: int, owner_type: str, owner_id: int) -> bool:
    """
    Delete document embeddings when a document is deleted
    
    Args:
        document_id: ID of document to delete
        owner_type: Type of document owner
        owner_id: ID of document owner
        
    Returns:
        Success status
    """
    from app.utils.document_processor import get_document_vector_path
    import shutil
    
    try:
        vector_path = get_document_vector_path(document_id, owner_type, owner_id)
        if os.path.exists(vector_path):
            shutil.rmtree(vector_path)
        return True
    except Exception as e:
        logger.error(f"Error deleting document embeddings: {str(e)}")
        return False
