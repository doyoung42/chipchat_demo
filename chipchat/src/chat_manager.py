import json
from pathlib import Path
from typing import List, Dict, Optional
import openai
from langchain.vectorstores import FAISS

class ChatManager:
    def __init__(self, api_key: str):
        """챗봇 관리자 초기화"""
        openai.api_key = api_key
        
    def load_prompt_template(self, template_path: str) -> Dict:
        """프롬프트 템플릿 로드"""
        with open(template_path, 'r') as f:
            return json.load(f)
    
    def save_prompt_template(self, template: Dict, template_path: str):
        """프롬프트 템플릿 저장"""
        with open(template_path, 'w') as f:
            json.dump(template, f, indent=2)
    
    def get_chat_response(
        self,
        query: str,
        vectorstore: FAISS,
        pre_prompt: str,
        post_prompt: str,
        k: int = 3
    ) -> str:
        """챗봇 응답 생성"""
        # 관련 문서 검색
        docs = vectorstore.similarity_search(query, k=k)
        
        # 컨텍스트 구성
        context = "\n".join([doc.page_content for doc in docs])
        
        # 프롬프트 구성
        prompt = f"{pre_prompt}\n\nQuestion: {query}\n\nContext: {context}\n\n{post_prompt}"
        
        # LLM 응답 생성
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.choices[0].message.content
    
    def test_retrieval(
        self,
        query: str,
        vectorstore: FAISS,
        k: int = 3,
        threshold: float = 0.7
    ) -> List[tuple]:
        """검색 테스트 수행"""
        results = vectorstore.similarity_search_with_score(query, k=k)
        return [(doc, score) for doc, score in results if score >= threshold] 