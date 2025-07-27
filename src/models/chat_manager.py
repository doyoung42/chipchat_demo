import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
from langchain.schema import Document
from langchain_community.vectorstores import FAISS

from .llm_manager import LLMManager
from .vectorstore_manager import VectorstoreManager

class ChatManager:
    def __init__(self, provider: str = "openai", model_name: str = None):
        """채팅 관리자 초기화
        
        Args:
            provider: LLM provider ("openai" or "claude")
            model_name: Specific model name (optional)
        """
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize LLM Manager
        self.llm_manager = LLMManager(provider=provider, model_name=model_name)
        
        # Initialize Vectorstore Manager
        self.vectorstore_manager = VectorstoreManager()
        
        # Chat history
        self.chat_history: List[Dict[str, str]] = []
        
        self.logger.info(f"ChatManager initialized with {provider} ({model_name or 'default model'})")
    
    def load_prompt_template(self, template_path: str) -> Dict:
        """프롬프트 템플릿 로드"""
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading prompt template: {str(e)}")
            # Return default template
            return {
                "pre": "당신은 전자 부품 데이터시트에 대해 응답하는 전문 도우미입니다.",
                "post": "검색된 정보를 바탕으로 명확하고 간결하게 답변해주세요."
            }
    
    def save_prompt_template(self, template: Dict, template_path: str):
        """프롬프트 템플릿 저장"""
        try:
            # Ensure directory exists
            Path(template_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(template_path, 'w', encoding='utf-8') as f:
                json.dump(template, f, indent=2, ensure_ascii=False)
                
            self.logger.info(f"Prompt template saved to {template_path}")
        except Exception as e:
            self.logger.error(f"Error saving prompt template: {str(e)}")
    
    def get_chat_response(
        self,
        query: str,
        vectorstore: FAISS,
        pre_prompt: str = "",
        post_prompt: str = "",
        k: int = 5,
        filters: Dict[str, Any] = None,
        include_metadata: bool = False
    ) -> Dict[str, Any]:
        """챗봇 응답 생성
        
        Args:
            query: User query
            vectorstore: FAISS vectorstore
            pre_prompt: System prompt prefix
            post_prompt: System prompt suffix
            k: Number of documents to retrieve
            filters: Metadata filters for search
            include_metadata: Whether to include source metadata in response
            
        Returns:
            Dictionary containing response and metadata
        """
        try:
            # Search for relevant documents with optional filters
            if filters:
                docs = self.vectorstore_manager.search_with_filters(
                    vectorstore, query, filters=filters, k=k
                )
            else:
                docs = vectorstore.similarity_search(query, k=k)
            
            # Construct context from retrieved documents
            context_parts = []
            source_metadata = []
            
            for i, doc in enumerate(docs):
                context_parts.append(f"[문서 {i+1}]\n{doc.page_content}")
                
                if include_metadata:
                    metadata = {
                        'source': doc.metadata.get('filename', 'Unknown'),
                        'component': doc.metadata.get('component_name', 'Unknown'),
                        'manufacturer': doc.metadata.get('manufacturer', 'Unknown'),
                        'category': doc.metadata.get('category', 'Unknown'),
                        'maker_pn': doc.metadata.get('maker_pn', ''),
                        'part_number': doc.metadata.get('part_number', '')
                    }
                    source_metadata.append(metadata)
            
            context = "\n\n".join(context_parts)
            
            # Generate response using LLM Manager
            response = self.llm_manager.get_chat_response(
                query=query,
                context=context,
                pre_prompt=pre_prompt,
                post_prompt=post_prompt
            )
            
            # Add to chat history
            self.chat_history.append({
                'query': query,
                'response': response,
                'context_docs': len(docs)
            })
            
            # Prepare result
            result = {
                'response': response,
                'sources_found': len(docs),
                'query': query
            }
            
            if include_metadata:
                result['source_metadata'] = source_metadata
                result['retrieved_docs'] = [doc.page_content for doc in docs]
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating chat response: {str(e)}")
            return {
                'response': f"응답 생성 중 오류가 발생했습니다: {str(e)}",
                'sources_found': 0,
                'query': query,
                'error': str(e)
            }
    
    def test_retrieval(
        self,
        query: str,
        vectorstore: FAISS,
        k: int = 5,
        threshold: float = 0.7,
        filters: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """검색 테스트 수행
        
        Args:
            query: Test query
            vectorstore: FAISS vectorstore
            k: Number of results to retrieve
            threshold: Similarity threshold
            filters: Metadata filters
            
        Returns:
            List of test results with scores and metadata
        """
        try:
            # Perform similarity search with scores
            if filters:
                # For filtered search, we need to use a workaround since FAISS doesn't support native filtering
                docs = self.vectorstore_manager.search_with_filters(
                    vectorstore, query, filters=filters, k=k*2
                )
                # Convert to format with scores (approximate)
                results = [(doc, 0.8) for doc in docs]  # Placeholder score
            else:
                results = vectorstore.similarity_search_with_score(query, k=k)
            
            # Filter by threshold and format results
            test_results = []
            for doc, score in results:
                if score >= threshold:
                    result = {
                        'content': doc.page_content,
                        'score': float(score),
                        'metadata': dict(doc.metadata),
                        'passes_threshold': True
                    }
                    test_results.append(result)
                elif len(test_results) < k:  # Include some below-threshold results for comparison
                    result = {
                        'content': doc.page_content,
                        'score': float(score),
                        'metadata': dict(doc.metadata),
                        'passes_threshold': False
                    }
                    test_results.append(result)
            
            return test_results[:k]
            
        except Exception as e:
            self.logger.error(f"Error in retrieval test: {str(e)}")
            return [{
                'content': f"검색 테스트 중 오류 발생: {str(e)}",
                'score': 0.0,
                'metadata': {},
                'passes_threshold': False,
                'error': str(e)
            }]
    
    def get_chat_history(self) -> List[Dict[str, str]]:
        """채팅 히스토리 반환"""
        return self.chat_history.copy()
    
    def clear_chat_history(self):
        """채팅 히스토리 초기화"""
        self.chat_history.clear()
        self.logger.info("Chat history cleared")
    
    def get_available_filters(self, vectorstore: FAISS) -> Dict[str, List[str]]:
        """사용 가능한 필터 옵션 반환
        
        Args:
            vectorstore: FAISS vectorstore
            
        Returns:
            Dictionary of available filter values for each metadata key
        """
        try:
            # Sample some documents to determine available filter options
            sample_docs = vectorstore.similarity_search("sample", k=50)
            
            filters = {}
            for doc in sample_docs:
                for key, value in doc.metadata.items():
                    if isinstance(value, (str, int, float)) and value:
                        if key not in filters:
                            filters[key] = set()
                        filters[key].add(str(value))
            
            # Convert sets to sorted lists
            return {key: sorted(list(values)) for key, values in filters.items()}
            
        except Exception as e:
            self.logger.error(f"Error getting available filters: {str(e)}")
            return {}
    
    def test_llm_connection(self) -> bool:
        """LLM 연결 테스트"""
        return self.llm_manager.test_connection()
    
    def switch_llm_provider(self, provider: str, model_name: str = None):
        """LLM 제공자 변경
        
        Args:
            provider: New LLM provider
            model_name: New model name (optional)
        """
        try:
            self.llm_manager = LLMManager(provider=provider, model_name=model_name)
            self.logger.info(f"Switched to {provider} ({model_name or 'default model'})")
        except Exception as e:
            self.logger.error(f"Error switching LLM provider: {str(e)}")
            raise 