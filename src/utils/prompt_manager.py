"""
프롬프트 템플릿 관리 모듈
./prompts 폴더의 JSON 파일들을 로드하여 프롬프트를 제공합니다.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional


class PromptManager:
    """프롬프트 템플릿 로더 및 관리자"""
    
    def __init__(self, prompts_dir: str = "./prompts"):
        """
        Args:
            prompts_dir: 프롬프트 템플릿이 저장된 디렉토리 경로
        """
        self.prompts_dir = Path(prompts_dir)
        self.logger = logging.getLogger(__name__)
        
        # 프롬프트 캐시
        self._classification_cache = None
        self._system_prompts_cache = None
        
        # 프롬프트 디렉토리 존재 확인
        if not self.prompts_dir.exists():
            self.logger.warning(f"Prompts directory not found: {self.prompts_dir}")
            self.prompts_dir.mkdir(parents=True, exist_ok=True)
            self._create_default_templates()
    
    def _create_default_templates(self):
        """기본 프롬프트 템플릿 파일들을 생성합니다."""
        self.logger.info("Creating default prompt templates...")
        
        # classification_prompt.json
        classification_template = {
            "classification_prompt": "Analyze this user query and classify it into one or more categories:\n\nQuery: \"{query}\"\n\nCategories:\n1. COMPONENT_LIST: User wants to list/find components with specific functionality\n   Examples: \"what components do voltage regulation\", \"list all power converters\"\n   \n2. TECHNICAL_DETAIL: User wants detailed technical information about specific components\n   Examples: \"W25Q32JV electrical characteristics\", \"voltage specifications of LM324\"\n   \n3. PDF_UPLOAD: User wants to upload/add new PDF datasheets\n   Examples: \"add new datasheet\", \"upload PDF\", \"process new component\"\n   \n4. HYBRID: Combination of above (e.g., find components then get details)\n\nReturn ONLY the primary category name (COMPONENT_LIST, TECHNICAL_DETAIL, PDF_UPLOAD, or HYBRID).\nIf multiple categories apply, return HYBRID.",
            "fallback_classification": "HYBRID",
            "valid_classifications": ["COMPONENT_LIST", "TECHNICAL_DETAIL", "PDF_UPLOAD", "HYBRID"],
            "description": "Query classification prompt for LangGraph agent tool selection"
        }
        
        # system_prompts.json
        system_prompts_template = {
            "default": {
                "pre_prompt": "당신은 전자 부품 데이터시트에 대해 응답하는 전문 도우미입니다. 제공된 컨텍스트 정보를 기반으로 질문에 정확하고 상세하게 답변하세요.",
                "post_prompt": "검색된 정보를 바탕으로 명확하고 간결하게 답변해주세요. 정보가 불충분하다면 그 점을 명시하세요.",
                "description": "Default system prompts for general component queries"
            },
            "english": {
                "pre_prompt": "You are an expert assistant that answers questions about electronic component datasheets. Please provide accurate and detailed responses based on the provided context information.",
                "post_prompt": "Please provide a clear and concise answer based on the retrieved information. If the information is insufficient, please specify that.",
                "description": "English system prompts for international users"
            }
        }
        
        # 파일 저장
        self._save_json_template("classification_prompt.json", classification_template)
        self._save_json_template("system_prompts.json", system_prompts_template)
    
    def _save_json_template(self, filename: str, template: Dict[str, Any]):
        """JSON 템플릿을 파일에 저장합니다."""
        try:
            filepath = self.prompts_dir / filename
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(template, f, indent=4, ensure_ascii=False)
            self.logger.info(f"Saved template: {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save template {filename}: {e}")
    
    def _load_json_template(self, filename: str) -> Dict[str, Any]:
        """JSON 템플릿을 파일에서 로드합니다."""
        try:
            filepath = self.prompts_dir / filename
            if not filepath.exists():
                self.logger.warning(f"Template file not found: {filepath}")
                return {}
            
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load template {filename}: {e}")
            return {}
    
    def get_classification_prompt(self, query: str) -> str:
        """쿼리 분류 프롬프트를 반환합니다."""
        if self._classification_cache is None:
            self._classification_cache = self._load_json_template("classification_prompt.json")
        
        template = self._classification_cache.get("classification_prompt", "")
        if not template:
            # 폴백 프롬프트
            return f"Classify this query: '{query}'. Return one of: COMPONENT_LIST, TECHNICAL_DETAIL, PDF_UPLOAD, HYBRID"
        
        return template.format(query=query)
    
    def get_valid_classifications(self) -> list:
        """유효한 분류 목록을 반환합니다."""
        if self._classification_cache is None:
            self._classification_cache = self._load_json_template("classification_prompt.json")
        
        return self._classification_cache.get("valid_classifications", ["HYBRID"])
    
    def get_fallback_classification(self) -> str:
        """기본 분류를 반환합니다."""
        if self._classification_cache is None:
            self._classification_cache = self._load_json_template("classification_prompt.json")
        
        return self._classification_cache.get("fallback_classification", "HYBRID")
    
    def get_system_prompts(self, prompt_type: str = "default") -> Dict[str, str]:
        """시스템 프롬프트를 반환합니다."""
        if self._system_prompts_cache is None:
            self._system_prompts_cache = self._load_json_template("system_prompts.json")
        
        prompts = self._system_prompts_cache.get(prompt_type, {})
        if not prompts:
            # 기본 프롬프트로 폴백
            prompts = self._system_prompts_cache.get("default", {
                "pre_prompt": "당신은 전자 부품 데이터시트에 대해 응답하는 전문 도우미입니다.",
                "post_prompt": "검색된 정보를 바탕으로 명확하게 답변해주세요."
            })
        
        return {
            "pre_prompt": prompts.get("pre_prompt", ""),
            "post_prompt": prompts.get("post_prompt", "")
        }
    
    def get_response_generation_prompt(self) -> str:
        """응답 생성 프롬프트를 반환합니다."""
        if self._system_prompts_cache is None:
            self._system_prompts_cache = self._load_json_template("system_prompts.json")
        
        return self._system_prompts_cache.get("response_generation", {}).get(
            "prompt", 
            "Generate a comprehensive response by combining the tool results intelligently."
        )
    
    def reload_templates(self):
        """템플릿 캐시를 클리어하고 다시 로드합니다."""
        self._classification_cache = None
        self._system_prompts_cache = None
        self.logger.info("Prompt templates cache cleared")


# 싱글톤 인스턴스
_prompt_manager_instance = None

def get_prompt_manager() -> PromptManager:
    """싱글톤 PromptManager 인스턴스 반환"""
    global _prompt_manager_instance
    
    if _prompt_manager_instance is None:
        _prompt_manager_instance = PromptManager()
    
    return _prompt_manager_instance 