"""
PDF processing utilities for the Datasheet Analyzer.
"""
from typing import List, Dict, Any
import pypdf
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ..config.settings import CHUNK_SIZE, CHUNK_OVERLAP

class PDFProcessor:
    """
    A class for processing PDF files.
    """
    def __init__(self):
        """
        Initialize the PDF processor.
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )
    
    def extract_text(self, pdf_path: Path) -> str:
        """
        Extract text from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text
        """
        with open(pdf_path, 'rb') as file:
            reader = pypdf.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        return self.text_splitter.split_text(text)
    
    def process_pdf(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """
        Process a PDF file and return chunks with metadata.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of chunks with metadata
        """
        text = self.extract_text(pdf_path)
        chunks = self.split_text(text)
        
        # Add metadata to chunks
        chunks_with_metadata = []
        for i, chunk in enumerate(chunks):
            chunks_with_metadata.append({
                "text": chunk,
                "chunk_id": i,
                "source": str(pdf_path),
            })
        
        return chunks_with_metadata 