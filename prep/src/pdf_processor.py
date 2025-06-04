import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set
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
            category_chunks = self._create_category_chunks(category_contents)
            
            # Extract metadata from all useful content
            all_useful_content = "\n\n".join([info.get('content', '') for info in page_summaries])
            metadata = self.llm_manager.extract_metadata(all_useful_content)
            
            # Save final results
            self._save_results(pdf_path, metadata, page_summaries, category_chunks)
            
            self.logger.info(f"Successfully processed {pdf_path.name}")
            
        except Exception as e:
            self.logger.error(f"Error processing {pdf_path.name}: {str(e)}")
    
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
    
    def _create_category_chunks(self, category_contents: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Create semantic chunks for each category
        
        Args:
            category_contents: Dictionary of category contents
            
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
            chunks = self.llm_manager.create_semantic_chunks_for_category(category, combined_content)
            
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
            pdf_path: Original PDF path
            metadata: Extracted metadata
            page_summaries: List of page summaries
            category_chunks: Dictionary of category chunks
        """
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