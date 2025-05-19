"""
Vector store utilities for the Datasheet Analyzer.
"""
from typing import List, Dict, Any, Optional
from pathlib import Path
import chromadb
from chromadb.config import Settings
from langchain.vectorstores import Chroma
from ..config.settings import VECTOR_STORE_DIR, VECTOR_STORE_CONFIG

class VectorStore:
    """
    A wrapper class for the vector store.
    """
    def __init__(self, embedding_function):
        """
        Initialize the vector store.
        
        Args:
            embedding_function: Function to use for embeddings
        """
        self.embedding_function = embedding_function
        self.vector_store = None
        self.collection_name = VECTOR_STORE_CONFIG["collection_name"]
    
    def create_store(self, chunks: List[Dict[str, Any]]):
        """
        Create a new vector store with the given chunks.
        
        Args:
            chunks: List of chunks with metadata
        """
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [{"chunk_id": chunk["chunk_id"], "source": chunk["source"]} 
                    for chunk in chunks]
        
        self.vector_store = Chroma.from_texts(
            texts=texts,
            embedding=self.embedding_function,
            metadatas=metadatas,
            persist_directory=str(VECTOR_STORE_DIR / self.collection_name)
        )
    
    def similarity_search(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        """
        Perform similarity search on the vector store.
        
        Args:
            query: Query text
            k: Number of results to return
            
        Returns:
            List of similar documents with metadata
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
        
        results = self.vector_store.similarity_search_with_score(query, k=k)
        return [
            {
                "text": doc.page_content,
                "metadata": doc.metadata,
                "score": score
            }
            for doc, score in results
        ]
    
    def clear_store(self):
        """
        Clear the vector store.
        """
        if self.vector_store:
            self.vector_store.delete_collection()
            self.vector_store = None 