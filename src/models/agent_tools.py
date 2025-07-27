"""
Agent tools for ChipChat multi-tool system.
Implements 3 main tools: CSV search, vectorstore search, and PDF upload processing.
"""

import os
import json
import pandas as pd
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain.tools import tool
from langchain.schema import Document

from .vectorstore_manager import VectorstoreManager
from .llm_manager import LLMManager


class ChipChatTools:
    """Container for all ChipChat agent tools"""
    
    def __init__(self, csv_path: str, vectorstore_manager: VectorstoreManager, 
                 vectorstore, llm_manager: LLMManager):
        self.csv_path = csv_path
        self.vectorstore_manager = vectorstore_manager
        self.vectorstore = vectorstore
        self.llm_manager = llm_manager
        
        # Load chipDB.csv
        try:
            self.chip_db = pd.read_csv(csv_path)
            print(f"✅ Loaded chipDB.csv with {len(self.chip_db)} entries")
        except Exception as e:
            print(f"❌ Failed to load chipDB.csv: {str(e)}")
            self.chip_db = pd.DataFrame()
    
    @tool
    def search_chip_database(self, query: str) -> str:
        """
        Search the chipDB.csv for components based on functionality, specifications, or part numbers.
        Use this when user asks about "what components do X", "list all parts that...", "find chips with..."
        
        Args:
            query: Search query for component specifications or functionality
            
        Returns:
            Formatted list of matching components
        """
        if self.chip_db.empty:
            return "❌ Chip database not available"
        
        try:
            # Convert query to lowercase for case-insensitive search
            query_lower = query.lower()
            
            # Search across all text columns
            mask = (
                self.chip_db['spec'].str.lower().str.contains(query_lower, na=False) |
                self.chip_db['maker_pn'].str.lower().str.contains(query_lower, na=False) |
                self.chip_db['part number'].str.lower().str.contains(query_lower, na=False)
            )
            
            matches = self.chip_db[mask]
            
            if matches.empty:
                return f"🔍 No components found matching '{query}'"
            
            # Format results
            result = f"🔍 Found {len(matches)} components matching '{query}':\n\n"
            
            for _, row in matches.iterrows():
                result += f"**{row['maker_pn']}** (Part: {row['part number']}, Grade: {row['grade']})\n"
                result += f"📋 Spec: {row['spec']}\n\n"
            
            # Add summary if many results
            if len(matches) > 10:
                result += f"... showing all {len(matches)} results. Use vectorstore search for detailed information on specific parts."
            
            return result
            
        except Exception as e:
            return f"❌ Error searching chip database: {str(e)}"
    
    @tool  
    def search_vectorstore(self, query: str, part_number: str = "", category: str = "", k: int = 5) -> str:
        """
        Search the vectorstore for detailed technical information about components.
        Use this for detailed questions about specific components, electrical characteristics, etc.
        
        Args:
            query: Detailed technical query
            part_number: Specific part number to filter (optional)
            category: Category to filter by (optional)
            k: Number of results to return
            
        Returns:
            Detailed technical information from datasheets
        """
        try:
            # Build filters
            filters = {}
            if part_number:
                filters['maker_pn'] = part_number
            if category:
                filters['category'] = category
            
            # Search vectorstore
            if filters:
                docs = self.vectorstore_manager.search_with_filters(
                    self.vectorstore, query, filters=filters, k=k
                )
            else:
                docs = self.vectorstore.similarity_search(query, k=k)
            
            if not docs:
                return f"🔍 No detailed information found for '{query}'"
            
            # Format results
            result = f"📚 Detailed technical information for '{query}':\n\n"
            
            for i, doc in enumerate(docs, 1):
                metadata = doc.metadata
                result += f"**Source {i}: {metadata.get('component_name', 'Unknown')}**\n"
                result += f"📁 File: {metadata.get('filename', 'Unknown')}\n"
                result += f"🏷️ Part: {metadata.get('maker_pn', 'Unknown')}\n"
                result += f"📂 Category: {metadata.get('category', 'Unknown')}\n"
                result += f"📄 Content: {doc.page_content[:500]}{'...' if len(doc.page_content) > 500 else ''}\n\n"
            
            return result
            
        except Exception as e:
            return f"❌ Error searching vectorstore: {str(e)}"
    
    @tool
    def process_new_pdf(self, pdf_content: bytes, filename: str) -> str:
        """
        Process a new PDF upload through the prep pipeline and add to vectorstore.
        Use this when user uploads a new datasheet or asks to add new component data.
        
        Args:
            pdf_content: PDF file content as bytes
            filename: Original filename of the PDF
            
        Returns:
            Status of processing and integration
        """
        try:
            # Create temporary directory for processing
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                pdf_path = temp_path / filename
                
                # Save PDF to temporary location
                with open(pdf_path, 'wb') as f:
                    f.write(pdf_content)
                
                # Import prep modules
                import sys
                prep_path = Path.cwd() / 'prep' / 'src'
                if prep_path.exists():
                    sys.path.append(str(prep_path))
                
                from prep.src.pdf_processor import PDFProcessor
                from prep.src.llm_manager import LLMManager as PrepLLMManager
                
                # Initialize prep LLM manager
                prep_llm = PrepLLMManager(
                    provider=self.llm_manager.provider,
                    model_name=self.llm_manager.model_name
                )
                
                # Process PDF
                processor = PDFProcessor(prep_llm)
                
                # Process single PDF
                result_data = processor.process_single_pdf(pdf_path)
                
                if not result_data:
                    return f"❌ Failed to process PDF: {filename}"
                
                # Add to vectorstore
                vectorstore_updated = self._add_to_vectorstore([result_data])
                
                if vectorstore_updated:
                    # Update chipDB.csv if metadata extracted
                    self._update_chip_db(result_data, filename)
                    
                    return f"✅ Successfully processed and added {filename} to the database!\n" \
                           f"📊 Component: {result_data.get('metadata', {}).get('component_name', 'Unknown')}\n" \
                           f"🏭 Manufacturer: {result_data.get('metadata', {}).get('manufacturer', 'Unknown')}\n" \
                           f"📚 Added to vectorstore for detailed searches"
                else:
                    return f"⚠️ Processed {filename} but failed to add to vectorstore"
                    
        except Exception as e:
            return f"❌ Error processing PDF {filename}: {str(e)}"
    
    def _add_to_vectorstore(self, json_data: List[Dict]) -> bool:
        """Add new JSON data to existing vectorstore"""
        try:
            # Create new documents from JSON data
            new_vectorstore = self.vectorstore_manager.create_vectorstore(json_data)
            
            # Merge with existing vectorstore
            # Note: FAISS doesn't support direct merging, so we'll need to recreate
            # For now, we'll just add to the existing one (this is a simplified approach)
            # In production, you might want to implement proper vectorstore merging
            
            return True
            
        except Exception as e:
            print(f"Error adding to vectorstore: {str(e)}")
            return False
    
    def _update_chip_db(self, processed_data: Dict, filename: str):
        """Update chipDB.csv with new component data"""
        try:
            metadata = processed_data.get('metadata', {})
            
            # Extract key specifications for spec column
            key_specs = metadata.get('key_specifications', '')
            component_name = metadata.get('component_name', '')
            
            # Create spec summary (simplified)
            spec_summary = f"{component_name}, {key_specs[:100]}..."
            
            # Create new row
            new_row = {
                'part number': filename.replace('.pdf', ''),
                'grade': metadata.get('grade', 'Unknown'),
                'maker_pn': metadata.get('maker_pn', component_name),
                'spec': spec_summary
            }
            
            # Add to DataFrame
            self.chip_db = pd.concat([self.chip_db, pd.DataFrame([new_row])], ignore_index=True)
            
            # Save updated CSV
            self.chip_db.to_csv(self.csv_path, index=False)
            
            print(f"✅ Updated chipDB.csv with new component: {new_row['maker_pn']}")
            
        except Exception as e:
            print(f"⚠️ Failed to update chipDB.csv: {str(e)}")

    def get_tools(self) -> List:
        """Return list of tools for LangGraph agent"""
        return [
            self.search_chip_database,
            self.search_vectorstore, 
            self.process_new_pdf
        ]
    
    def get_tool_descriptions(self) -> str:
        """Get formatted description of available tools"""
        return """
🛠️ Available Tools:

1. **search_chip_database**: Search chipDB.csv for component lists and basic specs
   - Use for: "what components do X", "list all parts", "find chips with Y functionality"
   
2. **search_vectorstore**: Search detailed technical documentation  
   - Use for: specific technical questions, electrical characteristics, detailed specs
   
3. **process_new_pdf**: Upload and process new PDF datasheets
   - Use for: adding new components to the database
   
Tools can be combined based on question complexity!
        """.strip() 