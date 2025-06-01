import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
from PyPDF2 import PdfReader, PdfWriter
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
        """Process a single PDF file
        
        Args:
            pdf_path: Path to the PDF file
        """
        try:
            # Read PDF
            reader = PdfReader(pdf_path)
            total_pages = len(reader.pages)
            self.logger.info(f"PDF has {total_pages} pages")
            
            # Process pages in chunks
            useful_pages = []
            page_summaries = []
            
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
                
                # Analyze chunk
                has_useful_pages, page_info = self.llm_manager.analyze_chunk(chunk_pages, chunk_page_numbers)
                
                if has_useful_pages:
                    for page_num, info in page_info.items():
                        if info['is_useful']:
                            useful_pages.append(int(page_num))
                            page_summaries.append({
                                'page_number': int(page_num),
                                'content': info['content']
                            })
            
            # Save intermediate results
            self._save_intermediate_results(pdf_path, useful_pages, page_summaries)
            
            # Generate final summaries
            self._generate_final_summaries(pdf_path, useful_pages, page_summaries)
            
        except Exception as e:
            self.logger.error(f"Error processing {pdf_path.name}: {str(e)}")
    
    def _save_intermediate_results(self, pdf_path: Path, useful_pages: List[int], page_summaries: List[Dict]):
        """Save intermediate processing results
        
        Args:
            pdf_path: Original PDF path
            useful_pages: List of useful page numbers
            page_summaries: List of page summaries
        """
        # Save filtered PDF
        if self.params['output_formats']['save_filtered_pdf']:
            writer = PdfWriter()
            reader = PdfReader(pdf_path)
            
            for page_num in useful_pages:
                writer.add_page(reader.pages[page_num - 1])
            
            filtered_pdf_path = Path(self.params['folders']['pre_json_folder']) / f"{pdf_path.stem}_filtered.pdf"
            with open(filtered_pdf_path, 'wb') as f:
                writer.write(f)
            
            self.logger.info(f"Saved filtered PDF: {filtered_pdf_path}")
        
        # Save page summaries
        summary_path = Path(self.params['folders']['pre_json_folder']) / f"{pdf_path.stem}_summaries.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump({
                'useful_pages': useful_pages,
                'page_summaries': page_summaries
            }, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved page summaries: {summary_path}")
    
    def _generate_final_summaries(self, pdf_path: Path, useful_pages: List[int], page_summaries: List[Dict]):
        """Generate final summaries using different approaches
        
        Args:
            pdf_path: Original PDF path
            useful_pages: List of useful page numbers
            page_summaries: List of page summaries
        """
        # Combine all summaries
        combined_text = "\n".join([summary['content'] for summary in page_summaries])
        
        # Generate summaries using different approaches
        if self.params['output_formats']['save_summary_only']:
            summary_result = self.llm_manager.extract_features_from_content(combined_text)
            result_path = Path(self.params['folders']['result_json_folder']) / f"{pdf_path.stem}_R1.json"
            with open(result_path, 'w', encoding='utf-8') as f:
                f.write(summary_result)
            self.logger.info(f"Saved summary-only result: {result_path}")
        
        if self.params['output_formats']['save_combined']:
            # Here you would implement the combined approach (filtered PDF + summaries)
            # This would require additional logic to combine both sources
            pass 