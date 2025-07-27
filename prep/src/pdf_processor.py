import os
import json
import logging
import csv
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set, Optional
from pypdf import PdfReader, PdfWriter
from .llm_manager import LLMManager

class PDFProcessor:
    def __init__(self, llm_manager: LLMManager):
        """Initialize PDF Processor
        
        Args:
            llm_manager: LLM Manager instance for text analysis
        """
        self.llm_manager = llm_manager
        
        # Load parameters from param.json
        param_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'misc', 'param.json')
        with open(param_path, 'r') as f:
            self.params = json.load(f)['pdf_processing']
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Create necessary directories
        for folder in self.params['folders'].values():
            os.makedirs(folder, exist_ok=True)
            
        # Load part number mapping if available
        self.part_mapping = {}
        self._load_part_mapping()
    
    def _load_part_mapping(self):
        """Load part number mapping from CSV file if it exists"""
        # 먼저 기본 csv_path 설정
        csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'misc', 'part_mapping.csv')
        
        # PDF 폴더에서 pdf_filenames.csv 찾기 (Google Drive)
        pdf_folder = self.params['folders']['pdf_folder']
        pdf_filenames_csv = os.path.join(pdf_folder, 'pdf_filenames.csv')
        
        if os.path.exists(pdf_filenames_csv):
            csv_path = pdf_filenames_csv
            self.logger.info(f"Using PDF filenames CSV from datasheet folder: {pdf_filenames_csv}")
        
        if os.path.exists(csv_path):
            try:
                with open(csv_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        filename_key = None
                        part_number = None
                        grade = None
                        spec = None
                        
                        # PDF 폴더에서의 csv 파일 포맷 처리
                        if 'filename' in row:
                            filename_key = os.path.splitext(row['filename'])[0]
                            
                        # Google Drive에서의 pdf_filenames.csv 형식 처리
                        if not filename_key and any(col in row for col in ['file', 'pdf']):
                            # 파일명 컬럼 찾기
                            for col in ['file', 'pdf', 'filename']:
                                if col in row and row[col]:
                                    filename_key = os.path.splitext(row[col])[0]
                                    break
                        
                        # maker pn 컬럼 찾기
                        for col in ['maker pn', 'maker_pn', 'part_number', 'partnumber']:
                            if col in row and row[col]:
                                part_number = row[col]
                                break
                        
                        # grade 컬럼 찾기
                        for col in ['fake_grade', 'grade']:
                            if col in row and row[col]:
                                grade = row[col]
                                break
                        
                        # spec 컬럼 찾기
                        for col in ['final_fake_code', 'spec', 'specification']:
                            if col in row and row[col]:
                                spec = row[col]
                                break
                        
                        if filename_key and part_number:
                            self.part_mapping[filename_key] = {
                                'part_number': part_number,
                                'grade': grade,
                                'spec': spec
                            }
                
                self.logger.info(f"Loaded {len(self.part_mapping)} part number mappings from {csv_path}")
            except Exception as e:
                self.logger.error(f"Error loading part mapping CSV: {str(e)}")
        else:
            self.logger.warning(f"Part mapping CSV file not found at {csv_path}")
    
    def process_pdf_folder(self):
        """Process all PDFs in the configured folder"""
        pdf_folder = self.params['folders']['pdf_folder']
        pdf_files = list(Path(pdf_folder).glob("*.pdf"))
        total_files = len(pdf_files)
        
        self.logger.info(f"Found {total_files} PDF files to process")
        
        for idx, pdf_path in enumerate(pdf_files, 1):
            self.logger.info(f"Processing file {idx}/{total_files}: {pdf_path.name}")
            self.process_single_pdf(pdf_path)
    
    def process_single_pdf(self, pdf_path: Path):
        """Process a single PDF file using the improved two-stage approach
        
        Args:
            pdf_path: Path to the PDF file
        """
        try:
            # Read PDF
            reader = PdfReader(pdf_path)
            total_pages = len(reader.pages)
            self.logger.info(f"PDF has {total_pages} pages")
            
            # Get target part number if available
            target_part_number = self._get_target_part_number(pdf_path)
            if target_part_number:
                self.logger.info(f"Processing for specific part number: {target_part_number}")
            
            # Stage 1: Process pages in chunks for usefulness and category classification
            self.logger.info("Stage 1: Analyzing pages for usefulness and categories")
            page_analysis_results = self._process_pages_in_chunks(reader, total_pages)
            
            # Filter for useful pages and their content
            useful_pages = []
            page_summaries = []
            category_contents = {
                "Product Summary": [],
                "Electrical Characteristics": [],
                "Application Circuits": [],
                "Mechanical Characteristics": [],
                "Reliability and Environmental Conditions": [],
                "Packaging Information": []
            }
            
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
            
            # Save filtered PDF if requested
            if self.params['output_formats']['save_filtered_pdf']:
                self._save_filtered_pdf(pdf_path, reader, useful_pages)
            
            # Stage 2: Create semantic chunks for each category
            self.logger.info("Stage 2: Creating semantic chunks for each category")
            category_chunks = self._create_category_chunks(category_contents, target_part_number)
            
            # Extract metadata from all useful content
            all_useful_content = "\n\n".join([info.get('content', '') for info in page_summaries])
            metadata = self.llm_manager.extract_metadata(all_useful_content, target_part_number)
            
            # Save final results
            self._save_results(pdf_path, metadata, page_summaries, category_chunks)
            
            self.logger.info(f"Successfully processed {pdf_path.name}")
            
        except Exception as e:
            self.logger.error(f"Error processing {pdf_path.name}: {str(e)}")
    
    def _get_target_part_number(self, pdf_path: Path) -> Optional[str]:
        """Get target part number from mapping if available
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Target part number or None if not found
        """
        # Try to find a match in the part mapping
        filename = pdf_path.stem
        if filename in self.part_mapping:
            if isinstance(self.part_mapping[filename], dict):
                return self.part_mapping[filename]['part_number']
            else:
                return self.part_mapping[filename]
        return None
    
    def _get_part_metadata(self, pdf_path: Path) -> Dict[str, str]:
        """Get part metadata (grade and spec) from mapping if available
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary with grade and spec information
        """
        filename = pdf_path.stem
        result = {'grade': None, 'spec': None}
        
        if filename in self.part_mapping and isinstance(self.part_mapping[filename], dict):
            mapping = self.part_mapping[filename]
            if 'grade' in mapping:
                result['grade'] = mapping['grade']
            if 'spec' in mapping:
                result['spec'] = mapping['spec']
        
        return result
    
    def _process_pages_in_chunks(self, reader: PdfReader, total_pages: int) -> Dict[str, Any]:
        """Process PDF pages in chunks for usefulness and category classification
        
        Args:
            reader: PDF reader object
            total_pages: Total number of pages in the PDF
            
        Returns:
            Dictionary of page analysis results
        """
        all_page_analysis = {}
        
        # Process pages in chunks
        for chunk_start in range(0, total_pages, self.params['pages_per_chunk']):
            chunk_end = min(chunk_start + self.params['pages_per_chunk'], total_pages)
            self.logger.info(f"Processing pages {chunk_start+1} to {chunk_end}")
            
            # Extract text from chunk
            chunk_pages = []
            chunk_page_numbers = []
            for page_num in range(chunk_start, chunk_end):
                page = reader.pages[page_num]
                chunk_pages.append(page.extract_text())
                chunk_page_numbers.append(page_num + 1)
            
            # Analyze chunk for usefulness and categories
            page_analysis = self.llm_manager.analyze_pages_with_categories(chunk_pages, chunk_page_numbers)
            
            # Add to overall results
            all_page_analysis.update(page_analysis)
        
        return all_page_analysis
    
    def _create_category_chunks(self, category_contents: Dict[str, List[str]], target_part_number: Optional[str] = None) -> Dict[str, List[str]]:
        """Create semantic chunks for each category
        
        Args:
            category_contents: Dictionary of category contents
            target_part_number: Specific part number to focus on (optional)
            
        Returns:
            Dictionary of category chunks
        """
        category_chunks = {}
        
        for category, contents in category_contents.items():
            if not contents:
                continue
                
            self.logger.info(f"Creating chunks for category: {category}")
            combined_content = "\n\n".join(contents)
            
            # Create semantic chunks for this category
            chunks = self.llm_manager.create_semantic_chunks_for_category(category, combined_content, target_part_number)
            
            if chunks:
                category_chunks[category] = chunks
        
        return category_chunks
    
    def _save_filtered_pdf(self, pdf_path: Path, reader: PdfReader, useful_pages: List[int]):
        """Save filtered PDF with only useful pages
        
        Args:
            pdf_path: Original PDF path
            reader: PDF reader object
            useful_pages: List of useful page numbers
        """
        writer = PdfWriter()
        
        for page_num in useful_pages:
            writer.add_page(reader.pages[page_num - 1])
        
        filtered_pdf_path = Path(self.params['folders']['pre_json_folder']) / f"{pdf_path.stem}_filtered.pdf"
        with open(filtered_pdf_path, 'wb') as f:
            writer.write(f)
        
        self.logger.info(f"Saved filtered PDF: {filtered_pdf_path}")
    
    def _save_results(self, pdf_path: Path, metadata: Dict[str, str], 
                     page_summaries: List[Dict[str, Any]], category_chunks: Dict[str, List[str]]):
        """Save processing results
        
        Args:
            pdf_path: Path to the PDF file
            metadata: Extracted metadata
            page_summaries: List of page summaries
            category_chunks: Dictionary of category chunks
        """
        # Get additional part metadata (grade and spec)
        part_metadata = self._get_part_metadata(pdf_path)
        
        # Add grade and spec to metadata if available
        if part_metadata['grade']:
            metadata['grade'] = part_metadata['grade']
        if part_metadata['spec']:
            metadata['spec'] = part_metadata['spec']
        
        # Prepare final result with filename at the top level
        final_result = {
            "filename": pdf_path.name,
            "metadata": metadata,
            "page_summaries": page_summaries,
            "category_chunks": category_chunks
        }
        
        # Save intermediate results (page summaries)
        summary_path = Path(self.params['folders']['pre_json_folder']) / f"{pdf_path.stem}_summaries.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(final_result, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved page summaries: {summary_path}")
        
        # Save final results based on output format settings
        if self.params['output_formats']['save_summary_only']:
            result_path = Path(self.params['folders']['result_json_folder']) / f"{pdf_path.stem}_R1.json"
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(final_result, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Saved final result: {result_path}")
        
        if self.params['output_formats']['save_combined']:
            # Here you could implement additional combined output format if needed
            # For now, just use the same format as summary_only
            result_path = Path(self.params['folders']['result_json_folder']) / f"{pdf_path.stem}_combined.json"
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(final_result, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Saved combined result: {result_path}") 