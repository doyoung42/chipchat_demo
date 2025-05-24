"""
Embedding model implementation for the Datasheet Analyzer.
"""
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from ..config.settings import EMBEDDING_MODEL
import logging

logger = logging.getLogger(__name__)

class EmbeddingModel:
    """
    A wrapper class for the embedding model.
    """
    def __init__(self, model_name: str = EMBEDDING_MODEL, hf_token: Optional[str] = None):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Name of the model to use for embeddings
            hf_token: HuggingFace API token
        """
        try:
            self.model_name = model_name
            self.hf_token = hf_token
            
            # 모델 초기화
            logger.info(f"Initializing embedding model: {model_name}")
            self.model = SentenceTransformer(model_name, token=hf_token)
            
            # LangChain 임베딩 초기화
            self.langchain_embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'token': hf_token}
            )
            logger.info("Embedding model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing embedding model: {str(e)}")
            raise
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings
        """
        try:
            return self.model.encode(texts).tolist()
        except Exception as e:
            logger.error(f"Error getting embeddings: {str(e)}")
            raise
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        try:
            return self.model.encode(text).tolist()
        except Exception as e:
            logger.error(f"Error getting embedding: {str(e)}")
            raise
    
    def get_langchain_embeddings(self) -> HuggingFaceEmbeddings:
        """
        Get the LangChain embeddings object.
        
        Returns:
            LangChain embeddings object
        """
        return self.langchain_embeddings 