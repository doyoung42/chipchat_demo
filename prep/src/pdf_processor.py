import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from pypdf import PdfReader
from .llm_manager import LLMManager

class PDFProcessor:
    def __init__(self, api_key: str, provider: str = "openai", claude_api_key: Optional[str] = None):
        """PDF 처리기 초기화
        
        Args:
            api_key: OpenAI API 키
            provider: LLM 제공자 ("openai" 또는 "claude")
            claude_api_key: Claude API 키 (provider가 "claude"일 때 필요)
        """
        self.llm_manager = LLMManager(api_key, provider, claude_api_key)
        
    def extract_text_from_pdf(self, pdf_path: str) -> List[str]:
        """PDF 파일에서 텍스트를 추출하여 페이지별로 리스트로 반환"""
        reader = PdfReader(pdf_path)
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text())
        return pages
    
    def process_pdf(self, pdf_path: str) -> Tuple[Dict, List[int]]:
        """PDF 파일을 처리하여 특징을 추출하고 유용한 페이지 번호를 반환"""
        # PDF 텍스트 추출
        pages = self.extract_text_from_pdf(pdf_path)
        
        # 유용한 페이지 필터링
        useful_pages = []
        useful_page_numbers = []
        for i, page in enumerate(pages):
            if self.llm_manager.check_page_usefulness(page):
                useful_pages.append(page)
                useful_page_numbers.append(i + 1)
        
        # 특징 추출
        if useful_pages:
            features_json = self.llm_manager.extract_features(useful_pages)
            features = json.loads(features_json)
            return features, useful_page_numbers
        else:
            return {}, [] 