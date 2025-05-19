"""
Embedding model implementation for the Datasheet Analyzer.
"""
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from ..config.settings import EMBEDDING_MODEL

class EmbeddingModel:
    """
    A wrapper class for the embedding model.
    """
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Name of the model to use for embeddings
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.langchain_embeddings = HuggingFaceEmbeddings(model_name=model_name)
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings
        """
        return self.model.encode(texts).tolist()
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        return self.model.encode(text).tolist()
    
    def get_langchain_embeddings(self) -> HuggingFaceEmbeddings:
        """
        Get the LangChain embeddings object.
        
        Returns:
            LangChain embeddings object
        """
        return self.langchain_embeddings 