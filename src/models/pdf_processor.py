"""
Independent PDF processor for ChipChat
Based on prep module implementation with identical prompts and processing logic
"""
import io
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

try:
    from pypdf import PdfReader
    PDF_LIBRARY = "pypdf"
except ImportError:
    PDF_LIBRARY = None

from .llm_manager import LLMManager

class PDFProcessor:
    """독립적인 PDF 처리기 (prep 모듈과 동일한 로직)"""
    
    def __init__(self, llm_manager: LLMManager):
        """
        Args:
            llm_manager: LLM 매니저 인스턴스
        """
        self.llm_manager = llm_manager
        
        if PDF_LIBRARY is None:
            raise ImportError("PDF 처리를 위해 pypdf가 필요합니다.")
        
        # Default parameters (based on prep module defaults)
        self.params = {
            'pages_per_chunk': 3,
            'categories': [
                "Product Summary",
                "Electrical Characteristics",
                "Application Circuits",
                "Mechanical Characteristics",
                "Reliability and Environmental Conditions",
                "Packaging Information"
            ]
        }
    
    def process_pdf(self, pdf_content: bytes, filename: str, target_part_number: Optional[str] = None) -> Dict[str, Any]:
        """PDF를 처리하여 구조화된 JSON 반환 (prep 모듈과 동일한 2단계 방식)"""
        try:
            # Create PdfReader from bytes
            pdf_file = io.BytesIO(pdf_content)
            reader = PdfReader(pdf_file)
            total_pages = len(reader.pages)
            
            # Stage 1: Process pages in chunks for usefulness and category classification
            page_analysis_results = self._process_pages_in_chunks(reader, total_pages)
            
            # Filter for useful pages and their content
            useful_pages = []
            page_summaries = []
            category_contents = {category: [] for category in self.params['categories']}
            
            # Collect useful pages and categorize content
            for page_num, info in page_analysis_results.items():
                if info.get('is_useful', False):
                    page_num_int = int(page_num)
                    useful_pages.append(page_num_int)
                    
                    # Add to page summaries
                    page_summaries.append({
                        'page_number': page_num_int,
                        'is_useful': True,
                        'categories': info.get('categories', []),
                        'content': info.get('content', '')
                    })
                    
                    # Add content to respective categories
                    for category in info.get('categories', []):
                        if category in category_contents:
                            category_contents[category].append(info.get('content', ''))
            
            # Stage 2: Create semantic chunks for each category
            category_chunks = self._create_category_chunks(category_contents, target_part_number)
            
            # Extract metadata from all useful content
            all_useful_content = "\n\n".join([info.get('content', '') for info in page_summaries])
            metadata = self._extract_metadata(all_useful_content, target_part_number)
            
            # Create final result structure
            result = {
                "filename": filename,
                "metadata": metadata,
                "page_summaries": page_summaries,
                "category_chunks": category_chunks,
                "processing_info": {
                    "processed_at": datetime.now().isoformat(),
                    "total_pages": total_pages,
                    "total_chunks": sum(len(chunks) for chunks in category_chunks.values()),
                    "pdf_library": PDF_LIBRARY,
                    "useful_pages": len(useful_pages)
                }
            }
            
            return result
            
        except Exception as e:
            raise Exception(f"PDF 처리 실패: {str(e)}")
    
    def _process_pages_in_chunks(self, reader: PdfReader, total_pages: int) -> Dict[str, Any]:
        """Process PDF pages in chunks for usefulness and category classification (prep 모듈과 동일)"""
        all_page_analysis = {}
        
        # Process pages in chunks
        for chunk_start in range(0, total_pages, self.params['pages_per_chunk']):
            chunk_end = min(chunk_start + self.params['pages_per_chunk'], total_pages)
            
            # Extract text from chunk
            chunk_pages = []
            chunk_page_numbers = []
            for page_num in range(chunk_start, chunk_end):
                page = reader.pages[page_num]
                chunk_pages.append(page.extract_text())
                chunk_page_numbers.append(page_num + 1)
            
            # Analyze chunk for usefulness and categories using LLM
            page_analysis = self._analyze_pages_with_categories(chunk_pages, chunk_page_numbers)
            
            # Add to overall results
            all_page_analysis.update(page_analysis)
        
        return all_page_analysis
    
    def _analyze_pages_with_categories(self, pages: List[str], page_numbers: List[int]) -> Dict[str, Any]:
        """Analyze pages with categories using identical prompt from prep module"""
        # Combine chunk contents (including page numbers)
        combined_text = ""
        for i, page in enumerate(pages):
            combined_text += f"=== PAGE {page_numbers[i]} ===\n{page}\n\n"
        
        prompt = f"""The following text contains multiple pages extracted from a datasheet PDF.
        For each page, please evaluate:
        
        1. Whether the page contains useful information (product specifications, technical characteristics, performance metrics, etc.)
        2. If useful, identify which category(s) the page belongs to from the following options:
            - Product Summary
            - Electrical Characteristics
            - Application Circuits
            - Mechanical Characteristics
            - Reliability and Environmental Conditions
            - Packaging Information
        3. Extract the most important information from that page (exact text, not summarized)
        
        IMPORTANT: Pay special attention to summary pages that often appear at the beginning of datasheets. These pages typically contain critical overview information, key features, and condensed specifications. These summary pages SHOULD BE MARKED AS USEFUL and categorized as "Product Summary".
        
        Non-useful content: copyright notices, blank pages, detailed indexes, etc. Note that table of contents pages might be useful if they contain overview information along with the contents.
        
        Text:
        {combined_text}
        
        Please output in the following JSON format:
        {{
            "page_analysis": {{
                "page_number1": {{
                    "is_useful": true or false,
                    "categories": ["Category1", "Category2"],
                    "content": "Important information here if useful (exact text)"
                }},
                "page_number2": {{
                    "is_useful": true or false,
                    "categories": ["Category1"],
                    "content": "Important information here if useful (exact text)"
                }},
                ...
            }}
        }}
        
        Please use the actual page numbers from the document.
        """
        
        try:
            response = self.llm_manager._call_llm(prompt)
            result = json.loads(response)
            page_analysis = result.get('page_analysis', {})
            return page_analysis
        except (json.JSONDecodeError, Exception):
            # Return empty result if parsing fails
            return {}
    
    def _create_category_chunks(self, category_contents: Dict[str, List[str]], target_part_number: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """Create semantic chunks for each category using identical logic from prep module"""
        category_chunks = {}
        
        for category, contents in category_contents.items():
            if not contents:
                continue
            
            combined_content = "\n\n".join(contents)
            chunks = self._create_semantic_chunks_for_category(category, combined_content, target_part_number)
            
            if chunks:
                # Convert to format with metadata
                chunk_objects = []
                for i, chunk_content in enumerate(chunks):
                    chunk_obj = {
                        'content': chunk_content,
                        'page_number': 0,  # Will be set by vectorstore manager
                        'chunk_index': i,
                        'maker_pn': target_part_number or '',
                        'part_number': target_part_number or '',
                        'grade': ''  # Will be filled by LLM if available
                    }
                    chunk_objects.append(chunk_obj)
                
                category_chunks[category] = chunk_objects
        
        return category_chunks
    
    def _create_semantic_chunks_for_category(self, category_name: str, category_content: str, target_part_number: Optional[str] = None) -> List[str]:
        """Create semantic chunks for category using identical prompt from prep module"""
        category_descriptions = {
            "Product Summary": "Overview, key features, and general description of the product",
            "Electrical Characteristics": "Voltage, current, power specifications, timing characteristics, and other electrical parameters",
            "Application Circuits": "Example circuits, application notes, and implementation examples",
            "Mechanical Characteristics": "Physical dimensions, form factor, mounting information, and mechanical specifications",
            "Reliability and Environmental Conditions": "Operating conditions, temperature ranges, reliability tests, and environmental specifications",
            "Packaging Information": "Package types, ordering information, marking details, and shipment specifications"
        }
        
        description = category_descriptions.get(category_name, "")
        
        part_specific_instruction = ""
        if target_part_number:
            part_specific_instruction = f"""
        IMPORTANT: This datasheet may contain information about multiple components.
        Focus ONLY on information related to part number: {target_part_number}.
        When creating chunks, ONLY include information relevant to this specific part number.
        """
        
        prompt = f"""You are an expert in analyzing electronic component datasheets.
        Your task is to create semantic chunks from content related to "{category_name}".
        
        {description}
        {part_specific_instruction}
        
        Please divide the following text into 3-5 meaningful chunks that would be useful for vector search.
        Each chunk should be self-contained and meaningful on its own.
        Create chunks with some semantic overlap for better retrieval.
        
        Text related to {category_name}:
        {category_content}
        
        Please output in the following JSON format:
        {{
            "chunks": [
                "Chunk 1 content...",
                "Chunk 2 content...",
                "Chunk 3 content..."
            ]
        }}
        """
        
        try:
            response = self.llm_manager._call_llm(prompt)
            result = json.loads(response)
            chunks = result.get('chunks', [])
            return chunks
        except (json.JSONDecodeError, Exception):
            # Return empty list if parsing fails
            return []
    
    def _extract_metadata(self, all_useful_content: str, target_part_number: Optional[str] = None) -> Dict[str, str]:
        """Extract metadata using identical prompt from prep module"""
        part_specific_instruction = ""
        if target_part_number:
            part_specific_instruction = f"""
        IMPORTANT: This datasheet contains information about multiple components. 
        Focus ONLY on the component with part number: {target_part_number}.
        Extract information ONLY related to this specific part number.
        """
            
        prompt = f"""You are an expert in analyzing electronic component datasheets. 
        Your task is to extract key metadata from the provided datasheet content.
        {part_specific_instruction}
        
        Please extract the following information:
        - Component name/model
        - Manufacturer
        - Key specifications (brief)
        - Main applications
        
        Text:
        {all_useful_content}
        
        Please output in the following JSON format:
        {{
            "component_name": "string",
            "manufacturer": "string",
            "key_specifications": "string",
            "applications": "string"
        }}
        """
        
        try:
            response = self.llm_manager._call_llm(prompt)
            result = json.loads(response)
            return result
        except (json.JSONDecodeError, Exception):
            # Return default values if parsing fails
            return {
                "component_name": "Unknown",
                "manufacturer": "Unknown",
                "key_specifications": "정보 추출 실패",
                "applications": "정보 추출 실패"
            } 