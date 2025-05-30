import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from pypdf import PdfReader
from .llm_manager import LLMManager

class PDFProcessor:
    def __init__(self, api_key: str, provider: str = "openai", claude_api_key: Optional[str] = None, chunk_size: int = 3):
        """PDF 처리기 초기화
        
        Args:
            api_key: OpenAI API 키
            provider: LLM 제공자 ("openai" 또는 "claude")
            claude_api_key: Claude API 키 (provider가 "claude"일 때 필요)
            chunk_size: 한 번에 처리할 PDF 페이지 수
        """
        self.llm_manager = LLMManager(api_key, provider, claude_api_key)
        self.chunk_size = chunk_size
        
    def extract_text_from_pdf(self, pdf_path: str) -> List[str]:
        """PDF 파일에서 텍스트를 추출하여 페이지별로 리스트로 반환"""
        reader = PdfReader(pdf_path)
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text())
        return pages
    
    def process_pdf(self, pdf_path: str) -> Tuple[Dict[str, Any], List[int]]:
        """PDF 파일을 처리하여 특징을 추출하고 유용한 페이지 번호를 반환
        
        1) PDF를 청크 단위로 나누어 각 청크의 유용성을 평가하고 필요한 내용 추출
        2) 필요한 페이지들만 모아서 최종 분석 수행
        """
        # PDF 텍스트 추출
        pages = self.extract_text_from_pdf(pdf_path)
        
        # 청크 단위로 유용한 페이지 필터링 (divide and conquer)
        useful_pages = []
        useful_page_numbers = []
        useful_content = []
        
        # 청크 단위로 나누어 처리
        for i in range(0, len(pages), self.chunk_size):
            chunk = pages[i:i+self.chunk_size]
            chunk_pages = list(range(i+1, min(i+self.chunk_size+1, len(pages)+1)))
            
            # 청크의 유용성 평가 및 주요 내용 추출
            is_useful, page_info = self.llm_manager.analyze_chunk(chunk, chunk_pages)
            
            if is_useful:
                for page_num, info in page_info.items():
                    if info["is_useful"]:
                        page_idx = int(page_num) - 1
                        useful_pages.append(pages[page_idx])
                        useful_page_numbers.append(int(page_num))
                        useful_content.append({
                            "page": int(page_num),
                            "content": info["content"]
                        })
        
        # 유용한 페이지가 있을 경우 최종 특징 추출
        if useful_pages:
            if len(useful_pages) <= 10:  # 유용한 페이지 수가 적은 경우 전체를 한 번에 처리
                features_json = self.llm_manager.extract_features(useful_pages)
                features = json.loads(features_json)
            else:
                # 유용한 콘텐츠만 모아서 처리
                collected_content = "\n\n".join([f"Page {item['page']}: {item['content']}" for item in useful_content])
                features_json = self.llm_manager.extract_features_from_content(collected_content)
                features = json.loads(features_json)
            
            return features, useful_page_numbers
        else:
            return {}, [] 