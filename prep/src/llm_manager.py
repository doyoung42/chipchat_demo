import json
import requests
from typing import Dict, List, Optional

class LLMManager:
    def __init__(self, api_key: str, provider: str = "openai", claude_api_key: Optional[str] = None):
        """LLM 관리자 초기화
        
        Args:
            api_key: OpenAI API 키
            provider: LLM 제공자 ("openai" 또는 "claude")
            claude_api_key: Claude API 키 (provider가 "claude"일 때 필요)
        """
        self.provider = provider.lower()
        
        # OpenAI 설정
        self.openai_api_key = api_key
        self.openai_api_url = "https://api.openai.com/v1/chat/completions"
        self.openai_headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Claude 설정
        self.claude_api_key = claude_api_key
        self.claude_api_url = "https://api.anthropic.com/v1/messages"
        self.claude_headers = {
            "x-api-key": claude_api_key if claude_api_key else "",
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
    
    def _call_llm(self, prompt: str) -> str:
        """LLM API 호출"""
        if self.provider == "openai":
            return self._call_openai(prompt)
        elif self.provider == "claude":
            return self._call_claude(prompt)
        else:
            raise ValueError(f"지원하지 않는 LLM 제공자: {self.provider}")
    
    def _call_openai(self, prompt: str) -> str:
        """OpenAI API 호출"""
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7
        }
        
        response = requests.post(self.openai_api_url, headers=self.openai_headers, json=data)
        response.raise_for_status()
        
        return response.json()["choices"][0]["message"]["content"]
    
    def _call_claude(self, prompt: str) -> str:
        """Claude API 호출"""
        if not self.claude_api_key:
            raise ValueError("Claude API 키가 설정되지 않았습니다.")
            
        data = {
            "model": "claude-3-sonnet-20240229",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1000,
            "temperature": 0.7
        }
        
        response = requests.post(self.claude_api_url, headers=self.claude_headers, json=data)
        response.raise_for_status()
        
        return response.json()["content"][0]["text"]
    
    def check_page_usefulness(self, page_text: str) -> bool:
        """페이지의 유용성을 LLM을 통해 판단"""
        prompt = f"""다음 텍스트가 데이터시트나 기술 문서에서 유용한 정보를 포함하고 있는지 판단해주세요.
        유용한 정보란 제품 사양, 기술적 특성, 성능 지표 등을 의미합니다.
        목차, 저작권 정보, 빈 페이지 등은 유용하지 않은 것으로 간주합니다.
        
        텍스트:
        {page_text}
        
        유용한 정보를 포함하고 있다면 'yes', 그렇지 않다면 'no'로만 답변해주세요."""
        
        response = self._call_llm(prompt)
        return response.strip().lower() == 'yes'
    
    def extract_features(self, pages: List[str]) -> Dict:
        """유용한 페이지들에서 특징을 추출하여 JSON 형태로 반환"""
        combined_text = "\n".join(pages)
        
        prompt = """다음 텍스트에서 중요한 특징들을 추출하여 JSON 형태로 정리해주세요.
        다음 형식으로 출력해주세요:
        {
            "product_name": "제품명",
            "key_features": ["특징1", "특징2", ...],
            "specifications": {
                "spec1": "값1",
                "spec2": "값2",
                ...
            },
            "applications": ["응용1", "응용2", ...],
            "notes": "추가 참고사항"
        }
        
        텍스트:
        {combined_text}"""
        
        response = self._call_llm(prompt)
        return response 