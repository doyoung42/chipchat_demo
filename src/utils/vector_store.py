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
    
    def similarity_search(self, query: str, k: int = 4, threshold: float = 0.0, use_mmr: bool = False, diversity: float = 0.3) -> List[Dict[str, Any]]:
        """
        Perform similarity search on the vector store.
        
        Args:
            query: Query text
            k: Number of results to return
            threshold: Minimum similarity score threshold (0.0 to 1.0)
            use_mmr: Whether to use MMR for diverse results
            diversity: Diversity parameter for MMR (0.0 to 1.0)
            
        Returns:
            List of similar documents with metadata
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
        
        # 검색 방법 선택 (MMR 또는 일반 유사도 검색)
        if use_mmr:
            # MMR 검색 (다양한 결과)
            # Chroma에서는 max_marginal_relevance_search_with_score가 아닌 max_marginal_relevance_search를 사용
            docs = self.vector_store.max_marginal_relevance_search(
                query, 
                k=k,
                fetch_k=k*2,  # 더 많은 후보를 가져와서 다양성을 높임
                lambda_mult=diversity  # 다양성 파라미터 (0: 다양성 최대, 1: 유사도 최대)
            )
            
            # 일반 검색과 형식을 맞추기 위해 임의의 스코어 부여
            results = [(doc, 0.5) for doc in docs]  # 임의의 중간 스코어 부여
        else:
            # 일반 유사도 검색
            results = self.vector_store.similarity_search_with_score(query, k=k)
        
        # 결과를 스코어 기준으로 필터링 및 정렬
        filtered_results = []
        for doc, score in results:
            # 스코어 정규화 (0~1 범위로 변환, 높을수록 더 유사)
            if use_mmr:
                # MMR은 이미 정렬된 결과이므로 높은 스코어 부여
                normalized_score = 0.8 - (0.1 * filtered_results.count(doc))  # 순서에 따라 약간 차등
            else:
                normalized_score = 1.0 - (score / 2.0)  # 원래 스코어가 거리 기반이므로 변환
            
            # 임계값 이상인 결과만 포함
            if normalized_score >= threshold:
                filtered_results.append({
                    "text": doc.page_content,
                    "metadata": doc.metadata,
                    "score": normalized_score
                })
        
        # 스코어 기준 내림차순 정렬
        filtered_results.sort(key=lambda x: x["score"], reverse=True)
        
        return filtered_results
    
    def clear_store(self):
        """
        Clear the vector store.
        """
        if self.vector_store:
            self.vector_store.delete_collection()
            self.vector_store = None 