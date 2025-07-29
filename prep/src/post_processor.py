import os
import json
import pandas as pd
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Set
import logging

class PostProcessor:
    def __init__(self, base_folder: Optional[str] = None):
        """Initialize Post Processor
        
        Args:
            base_folder: Base folder path for processing (if None, reads from param.json)
        """
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Set base folder paths
        if base_folder is None:
            # Read from param.json
            try:
                param_path = Path(__file__).parent.parent / 'misc' / 'param.json'
                with open(param_path, 'r') as f:
                    params = json.load(f)
                
                # Use result_json_folder as base for most operations
                result_json_folder = params['pdf_processing']['folders']['result_json_folder']
                pdf_folder = params['pdf_processing']['folders']['pdf_folder']
                
                self.base_folder = Path(result_json_folder).parent  # Get parent directory
                self.pdf_folder = Path(pdf_folder)
                self.json_folder = Path(result_json_folder)
                
            except Exception as e:
                self.logger.warning(f"Failed to read param.json: {e}, using default paths")
                # Fallback to default relative paths
                self.base_folder = Path.cwd()  # Assume running from prep folder
                self.pdf_folder = self.base_folder / 'datasheets'
                self.json_folder = self.base_folder / 'prep_json'
        else:
            self.base_folder = Path(base_folder)
            self.pdf_folder = self.base_folder / 'datasheets'
            self.json_folder = self.base_folder / 'prep_json'
            
        # Define folder paths
        self.pre_json_folder = self.json_folder / 'pre_json'
        self.merged_json_folder = self.json_folder / 'merged_json'
        self.modified_json_folder = self.json_folder / 'modified_json'
        self.modified_json_2_folder = self.json_folder / 'modified_json_2'
        self.final_json_folder = self.json_folder / 'final_json'
        self.not_processed_folder = self.pdf_folder / 'notprocessed'
        
        # CSV file path
        self.csv_file = self.pdf_folder / 'pdf_filenames.csv'
        
        # Create necessary directories
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories if they don't exist"""
        folders = [
            self.merged_json_folder,
            self.modified_json_folder,
            self.modified_json_2_folder,
            self.final_json_folder,
            self.not_processed_folder
        ]
        
        for folder in folders:
            folder.mkdir(parents=True, exist_ok=True)
            
    def run_complete_postprocessing(self):
        """Run the complete post-processing pipeline"""
        self.logger.info("Starting complete post-processing pipeline")
        
        # Step 1: JSON Merge
        self.logger.info("Step 1: Merging JSON files with CSV metadata")
        self.merge_json_with_csv()
        
        # Step 2: Field Normalization
        self.logger.info("Step 2: Normalizing field names")
        self.normalize_field_names()
        
        # Step 3: Dataset Validation and Cleanup
        self.logger.info("Step 3: Validating dataset and cleaning up")
        self.validate_and_cleanup_dataset()
        
        self.logger.info("Complete post-processing pipeline finished")
    
    def merge_json_with_csv(self):
        """Merge JSON files with CSV metadata (json_merge.ipynb logic)"""
        try:
            # Read CSV file
            if not self.csv_file.exists():
                self.logger.error(f"CSV file not found: {self.csv_file}")
                return
                
            pdf_df = pd.read_csv(self.csv_file)
            
            # Ensure necessary columns exist
            required_csv_columns = ['final_fake_code', 'fake_grade', 'maker pn']
            if not all(col in pdf_df.columns for col in required_csv_columns):
                self.logger.error(f"CSV file must contain columns: {required_csv_columns}")
                return
                
            # Create lookup dictionary
            csv_lookup = pdf_df.set_index('final_fake_code')[['fake_grade', 'maker pn']].to_dict(orient='index')
            
            # Find JSON files to process
            json_files = list(self.json_folder.glob('*_combined.json'))
            
            if not json_files:
                self.logger.warning(f"No JSON files ending with '_combined.json' found in {self.json_folder}")
                return
                
            processed_count = 0
            for json_file in json_files:
                try:
                    # Extract fake_code from filename
                    file_stem = json_file.stem
                    fake_code = file_stem.replace('_combined', '')
                    
                    if fake_code in csv_lookup:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        # Add CSV metadata
                        if 'metadata' not in data:
                            data['metadata'] = {}
                            
                        csv_values = csv_lookup[fake_code]
                        data['metadata']['fake_grade'] = csv_values['fake_grade']
                        data['metadata']['maker_pn'] = csv_values['maker pn']
                        
                        # Save merged JSON
                        output_filename = f"{fake_code}.json"
                        output_path = self.merged_json_folder / output_filename
                        
                        with open(output_path, 'w', encoding='utf-8') as f:
                            json.dump(data, f, indent=2, ensure_ascii=False)
                            
                        processed_count += 1
                        self.logger.info(f"Merged and saved: {output_path}")
                    else:
                        self.logger.warning(f"Skipping {json_file}: '{fake_code}' not found in CSV")
                        
                except Exception as e:
                    self.logger.error(f"Error processing {json_file}: {e}")
                    
            self.logger.info(f"JSON merge completed. Processed {processed_count} files")
            
        except Exception as e:
            self.logger.error(f"Error in merge_json_with_csv: {e}")
    
    def normalize_field_names(self):
        """Normalize field names in JSON files (modify part number.ipynb logic)"""
        try:
            # Process files from merged_json folder
            source_folder = self.merged_json_folder
            target_folder = self.modified_json_folder
            
            json_files = list(source_folder.glob('*.json'))
            
            if not json_files:
                self.logger.warning(f"No JSON files found in {source_folder}")
                return
                
            processed_count = 0
            for json_file in json_files:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Normalize field names at root level
                    if "filename" in data:
                        data["part number"] = data.pop("filename").replace(".pdf", "")
                    
                    if "fake_grade" in data:
                        data["grade"] = data.pop("fake_grade")
                    
                    # Normalize field names in metadata
                    if "metadata" in data and isinstance(data["metadata"], dict):
                        if "fake_grade" in data["metadata"]:
                            data["metadata"]["grade"] = data["metadata"].pop("fake_grade")
                    
                    # Normalize field names in page_summaries
                    if "page_summaries" in data and isinstance(data["page_summaries"], list):
                        for page_summary in data["page_summaries"]:
                            if isinstance(page_summary, dict):
                                if "filename" in page_summary:
                                    page_summary["part number"] = page_summary.pop("filename").replace(".pdf", "")
                                if "fake_grade" in page_summary:
                                    page_summary["grade"] = page_summary.pop("fake_grade")
                    
                    # Save normalized JSON
                    output_path = target_folder / json_file.name
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    
                    processed_count += 1
                    self.logger.info(f"Normalized and saved: {output_path}")
                    
                except Exception as e:
                    self.logger.error(f"Error processing {json_file}: {e}")
                    
            self.logger.info(f"Field normalization completed. Processed {processed_count} files")
            
        except Exception as e:
            self.logger.error(f"Error in normalize_field_names: {e}")
    
    def normalize_field_names_text_replacement(self):
        """Normalize field names using text replacement (alternative method)"""
        try:
            source_folder = self.modified_json_folder
            target_folder = self.modified_json_2_folder
            
            json_files = list(source_folder.glob('*.json'))
            
            if not json_files:
                self.logger.warning(f"No JSON files found in {source_folder}")
                return
                
            processed_count = 0
            for json_file in json_files:
                try:
                    # Read file as text
                    with open(json_file, 'r', encoding='utf-8') as f:
                        file_content = f.read()
                    
                    # Text replacement
                    modified_content = file_content.replace('"fake_grade"', '"grade"')
                    modified_content = modified_content.replace('"filename"', '"part number"')
                    
                    # Save modified file
                    output_path = target_folder / json_file.name
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(modified_content)
                    
                    processed_count += 1
                    self.logger.info(f"Text normalized and saved: {output_path}")
                    
                except Exception as e:
                    self.logger.error(f"Error processing {json_file}: {e}")
                    
            self.logger.info(f"Text normalization completed. Processed {processed_count} files")
            
        except Exception as e:
            self.logger.error(f"Error in normalize_field_names_text_replacement: {e}")
    
    def validate_and_cleanup_dataset(self):
        """Validate dataset and cleanup unprocessed files (datasetPostprep.ipynb logic)"""
        try:
            # Compare PDF and JSON files
            pdf_files = {f.stem for f in self.pdf_folder.glob("*.pdf")}
            json_files = {f.stem for f in self.modified_json_2_folder.glob("*.json")}
            
            # Find common and unique files
            common_files = pdf_files.intersection(json_files)
            pdf_only_files = pdf_files - json_files
            json_only_files = json_files - pdf_files
            
            # Log results
            self.logger.info(f"Files in both PDF and JSON: {len(common_files)}")
            self.logger.info(f"PDF only files: {len(pdf_only_files)}")
            self.logger.info(f"JSON only files: {len(json_only_files)}")
            
            # Move unprocessed PDFs to notprocessed folder
            moved_count = 0
            for file_base_name in pdf_only_files:
                source_path = self.pdf_folder / f"{file_base_name}.pdf"
                destination_path = self.not_processed_folder / f"{file_base_name}.pdf"
                
                try:
                    shutil.move(str(source_path), str(destination_path))
                    moved_count += 1
                    self.logger.info(f"Moved {file_base_name}.pdf to notprocessed folder")
                except Exception as e:
                    self.logger.error(f"Error moving {file_base_name}.pdf: {e}")
            
            self.logger.info(f"Moved {moved_count} unprocessed PDF files")
            
            # Restore original filenames for unprocessed files
            self._restore_original_filenames()
            
            # Create final dataset
            self._create_final_dataset()
            
        except Exception as e:
            self.logger.error(f"Error in validate_and_cleanup_dataset: {e}")
    
    def _restore_original_filenames(self):
        """Restore original filenames for files in notprocessed folder"""
        try:
            if not self.csv_file.exists():
                self.logger.warning(f"CSV file not found: {self.csv_file}")
                return
                
            # Read CSV file
            df = pd.read_csv(self.csv_file)
            
            # Check required columns
            if 'final_fake_code' not in df.columns or 'PDF_Filename' not in df.columns:
                self.logger.error("CSV file must contain 'final_fake_code' and 'PDF_Filename' columns")
                return
            
            # Get files in notprocessed folder
            not_processed_files = list(self.not_processed_folder.glob("*.pdf"))
            
            renamed_count = 0
            for pdf_file in not_processed_files:
                current_code = pdf_file.stem
                
                # Find matching row in CSV
                matching_row = df[df['final_fake_code'] == current_code]
                
                if not matching_row.empty:
                    original_filename = matching_row['PDF_Filename'].values[0]
                    new_filepath = self.not_processed_folder / original_filename
                    
                    try:
                        pdf_file.rename(new_filepath)
                        renamed_count += 1
                        self.logger.info(f"Renamed {pdf_file.name} to {original_filename}")
                    except Exception as e:
                        self.logger.error(f"Error renaming {pdf_file.name}: {e}")
                else:
                    self.logger.warning(f"No matching original filename found for {current_code}")
            
            self.logger.info(f"Restored original filenames for {renamed_count} files")
            
        except Exception as e:
            self.logger.error(f"Error in _restore_original_filenames: {e}")
    
    def _create_final_dataset(self):
        """Create final cleaned dataset"""
        try:
            source_folder = self.modified_json_2_folder
            target_folder = self.final_json_folder
            
            json_files = list(source_folder.glob('*.json'))
            
            if not json_files:
                self.logger.warning(f"No JSON files found in {source_folder}")
                return
            
            processed_count = 0
            for json_file in json_files:
                try:
                    # Copy file to final folder
                    target_path = target_folder / json_file.name
                    shutil.copy2(str(json_file), str(target_path))
                    processed_count += 1
                    
                except Exception as e:
                    self.logger.error(f"Error copying {json_file}: {e}")
            
            self.logger.info(f"Created final dataset with {processed_count} files in {target_folder}")
            
        except Exception as e:
            self.logger.error(f"Error in _create_final_dataset: {e}")
    
    def get_processing_summary(self) -> Dict:
        """Get summary of processing results"""
        try:
            summary = {
                'original_json_files': len(list(self.json_folder.glob('*_combined.json'))),
                'merged_json_files': len(list(self.merged_json_folder.glob('*.json'))),
                'normalized_json_files': len(list(self.modified_json_folder.glob('*.json'))),
                'final_json_files': len(list(self.final_json_folder.glob('*.json'))),
                'unprocessed_pdf_files': len(list(self.not_processed_folder.glob('*.pdf'))),
                'processed_pdf_files': len(list(self.pdf_folder.glob('*.pdf')))
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting processing summary: {e}")
            return {} 