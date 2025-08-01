import json
import os
import logging
from typing import Dict, List, Any, Optional
import requests
from pathlib import Path

# PromptManager import (try-except for backwards compatibility)
try:
    from ..utils.prompt_manager import get_prompt_manager
    PROMPT_MANAGER_AVAILABLE = True
except ImportError:
    PROMPT_MANAGER_AVAILABLE = False

class LLMManager:
    def __init__(self, provider: str = "openai", model_name: str = None):
        """Initialize LLM Manager with multi-provider support
        
        Args:
            provider: LLM provider ("openai" or "claude")
            model_name: Specific model name (optional)
        """
        self.provider = provider
        self.model_name = model_name
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load API keys from environment variables or streamlit secrets
        self.api_keys = self._load_api_keys()
        
        # Set default models if not specified
        if not self.model_name:
            if provider == "openai":
                self.model_name = "gpt-4o-mini"
            elif provider == "claude":
                self.model_name = "claude-3-sonnet-20240229"
        
        # Configure API settings
        self._configure_api()
        
        # Initialize prompt manager if available
        if PROMPT_MANAGER_AVAILABLE:
            try:
                self.prompt_manager = get_prompt_manager()
                self.logger.info("PromptManager initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize PromptManager: {e}")
                self.prompt_manager = None
        else:
            self.prompt_manager = None
            self.logger.info("PromptManager not available, using default prompts")
    
    def _load_api_keys(self) -> Dict[str, str]:
        """Load API keys from various sources"""
        keys = {}
        
        # Try loading from environment variables first
        keys['openai'] = os.environ.get('OPENAI_API_KEY', '')
        keys['anthropic'] = os.environ.get('ANTHROPIC_API_KEY', '')
        keys['huggingface'] = os.environ.get('HF_TOKEN', '')
        
        # Try loading from streamlit secrets if available
        try:
            import streamlit as st
            if hasattr(st, 'secrets'):
                keys['openai'] = keys['openai'] or st.secrets.get("openai_api_key", "")
                keys['anthropic'] = keys['anthropic'] or st.secrets.get("anthropic_api_key", "")
                keys['huggingface'] = keys['huggingface'] or st.secrets.get("hf_token", "")
        except ImportError:
            pass
        
        return keys
    
    def _configure_api(self):
        """Configure API settings based on provider"""
        if self.provider == "openai":
            if not self.api_keys['openai']:
                raise ValueError("OpenAI API key not found")
            
            self.api_url = "https://api.openai.com/v1/chat/completions"
            self.headers = {
                "Authorization": f"Bearer {self.api_keys['openai']}",
                "Content-Type": "application/json"
            }
            
        elif self.provider == "claude":
            if not self.api_keys['anthropic']:
                raise ValueError("Anthropic API key not found")
            
            self.api_url = "https://api.anthropic.com/v1/messages"
            self.headers = {
                "x-api-key": self.api_keys['anthropic'],
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            }
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _call_llm(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2000) -> str:
        """Call LLM API with provider-specific formatting
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated response text
        """
        try:
            if self.provider == "openai":
                data = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
                
                response = requests.post(self.api_url, headers=self.headers, json=data)
                response.raise_for_status()
                
                return response.json()["choices"][0]["message"]["content"]
                
            elif self.provider == "claude":
                data = {
                    "model": self.model_name,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "messages": [{"role": "user", "content": prompt}]
                }
                
                response = requests.post(self.api_url, headers=self.headers, json=data)
                response.raise_for_status()
                
                return response.json()["content"][0]["text"]
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed: {str(e)}")
            return "죄송합니다. LLM 서비스에 일시적인 문제가 발생했습니다."
        except Exception as e:
            self.logger.error(f"Unexpected error: {str(e)}")
            return "응답 생성 중 오류가 발생했습니다."
    
    def get_chat_response(self, query: str, context: str = "", pre_prompt: str = "", post_prompt: str = "") -> str:
        """Generate chat response with context
        
        Args:
            query: User query
            context: Retrieved context from vector search
            pre_prompt: System prompt prefix
            post_prompt: System prompt suffix
            
        Returns:
            Generated response
        """
        # Default prompts if not provided - use PromptManager if available
        if not pre_prompt or not post_prompt:
            if self.prompt_manager:
                system_prompts = self.prompt_manager.get_system_prompts("default")
                if not pre_prompt:
                    pre_prompt = system_prompts.get("pre_prompt", "당신은 전자 부품 데이터시트에 대해 응답하는 전문 도우미입니다.")
                if not post_prompt:
                    post_prompt = system_prompts.get("post_prompt", "검색된 정보를 바탕으로 명확하게 답변해주세요.")
            else:
                # Fallback to hardcoded prompts if PromptManager not available
                if not pre_prompt:
                    pre_prompt = "당신은 전자 부품 데이터시트에 대해 응답하는 전문 도우미입니다. 제공된 컨텍스트 정보를 기반으로 질문에 정확하고 상세하게 답변하세요."
                if not post_prompt:
                    post_prompt = "검색된 정보를 바탕으로 명확하고 간결하게 답변해주세요. 정보가 불충분하다면 그 점을 명시하세요."
        
        # Construct prompt
        if context:
            prompt = f"""{pre_prompt}

컨텍스트 정보:
{context}

질문: {query}

{post_prompt}"""
        else:
            prompt = f"""{pre_prompt}

질문: {query}

{post_prompt}"""
        
        return self._call_llm(prompt)
    
    def extract_metadata(self, content: str, target_part_number: str = None) -> Dict[str, str]:
        """Extract metadata from datasheet content
        
        Args:
            content: Datasheet content
            target_part_number: Specific part number to focus on
            
        Returns:
            Dictionary of extracted metadata
        """
        part_specific_instruction = ""
        if target_part_number:
            part_specific_instruction = f"""
중요: 이 데이터시트에는 여러 부품 정보가 포함될 수 있습니다.
특정 부품 번호 {target_part_number}에 대한 정보만 추출하세요.
"""
        
        prompt = f"""다음 데이터시트 내용에서 핵심 메타데이터를 추출하세요.
{part_specific_instruction}

다음 JSON 형식으로 응답하세요:
{{
    "component_name": "부품명",
    "manufacturer": "제조사",
    "key_specifications": "주요 사양 요약",
    "applications": "주요 응용 분야"
}}

데이터시트 내용:
{content}"""
        
        response = self._call_llm(prompt)
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            self.logger.warning("Failed to parse metadata JSON, returning default")
            return {
                "component_name": "Unknown",
                "manufacturer": "Unknown", 
                "key_specifications": "정보 추출 실패",
                "applications": "정보 추출 실패"
            }
    
    def test_connection(self) -> bool:
        """Test LLM API connection
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            test_response = self._call_llm("Hello", temperature=0.1, max_tokens=10)
            return len(test_response) > 0
        except Exception as e:
            self.logger.error(f"Connection test failed: {str(e)}")
            return False 