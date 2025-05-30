import json
import requests
from typing import Dict, List
from faiss import FAISS

class LLMManager:
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        """LLM 관리자 초기화"""
        self.api_key = api_key
        self.model = model
        self.api_url = "https://api.openai.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def _call_llm(self, prompt: str) -> str:
        """LLM API 호출"""
        data = {
            "model": self.model,  # Use the selected model
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7
        }
        
        response = requests.post(self.api_url, headers=self.headers, json=data)
        response.raise_for_status()
        
        return response.json()["choices"][0]["message"]["content"]
    
    def get_chat_response(self, query: str, vectorstore: FAISS, pre_prompt: str, post_prompt: str, k: int = 3) -> str:
        """챗봇 응답 생성"""
        # 벡터 스토어에서 관련 문서 검색
        docs = vectorstore.similarity_search(query, k=k)
        
        # 검색된 문서들을 컨텍스트로 구성
        context = "\n".join([doc.page_content for doc in docs])
        
        # 프롬프트 구성
        prompt = f"""{pre_prompt}

        컨텍스트:
        {context}

        질문: {query}

        {post_prompt}"""
        
        # LLM을 통해 응답 생성
        response = self._call_llm(prompt)
        return response 