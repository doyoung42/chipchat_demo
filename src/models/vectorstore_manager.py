import json
import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
import faiss

class VectorstoreManager:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """벡터 스토어 관리자 초기화"""
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load HuggingFace token from various sources
        hf_token = self._load_hf_token()
        
        # Initialize embeddings
        embedding_kwargs = {'model_name': model_name}
        if hf_token:
            embedding_kwargs['huggingface_api_token'] = hf_token
            
        # Try to use GPU if available, fallback to CPU
        try:
            embedding_kwargs['model_kwargs'] = {'device': 'cuda'}
            self.embeddings = HuggingFaceEmbeddings(**embedding_kwargs)
            self.use_gpu = True
            self.logger.info("Using GPU for embeddings")
            
            # Configure FAISS to use GPU
            self.res = faiss.StandardGpuResources()
        except Exception as e:
            self.logger.warning(f"GPU not available, using CPU: {str(e)}")
            embedding_kwargs['model_kwargs'] = {'device': 'cpu'}
            self.embeddings = HuggingFaceEmbeddings(**embedding_kwargs)
            self.use_gpu = False
        
        self.model_name = model_name
    
    def _load_hf_token(self) -> Optional[str]:
        """Load HuggingFace token from various sources"""
        # Try environment variable first
        hf_token = os.environ.get('HF_TOKEN', '')
        
        if not hf_token:
            # Try streamlit secrets
            try:
                import streamlit as st
                if hasattr(st, 'secrets'):
                    hf_token = st.secrets.get("hf_token", "")
            except ImportError:
                pass
        
        return hf_token if hf_token else None
    
    def load_json_files(self, folder_path: str) -> List[Dict]:
        """JSON 파일들을 로드하여 리스트로 반환"""
        json_files = Path(folder_path).glob("*.json")
        json_data = []
        
        for f in json_files:
            try:
                data = json.loads(f.read_text(encoding='utf-8'))
                json_data.append(data)
            except Exception as e:
                self.logger.error(f"Error loading {f.name}: {str(e)}")
                
        return json_data
    
    def create_vectorstore(self, json_data: List[Dict]) -> FAISS:
        """JSON 데이터로부터 벡터 스토어 생성
        
        새로운 JSON 구조 지원:
        {
            "filename": "example.pdf",
            "metadata": { ... },
            "page_summaries": [ ... ],
            "category_chunks": { ... }
        }
        """
        documents = []
        
        for data in json_data:
            filename = data.get('filename', 'unknown.pdf')
            metadata_base = {
                'filename': filename,
                'component_name': data.get('metadata', {}).get('component_name', ''),
                'manufacturer': data.get('metadata', {}).get('manufacturer', ''),
                'source': 'datasheet'
            }
            
            # Add global part information if available
            if 'part number' in data:
                metadata_base['part_number'] = data['part number']
            if data.get('metadata', {}).get('maker_pn'):
                metadata_base['maker_pn'] = data['metadata']['maker_pn']
            if data.get('metadata', {}).get('grade'):
                metadata_base['grade'] = data['metadata']['grade']
            
            # 1. Process category chunks (primary data source)
            category_chunks = data.get('category_chunks', {})
            for category, chunks in category_chunks.items():
                for i, chunk in enumerate(chunks):
                    # Handle both string and dict format chunks
                    if isinstance(chunk, dict):
                        # New format: chunk is a dict with content and metadata
                        chunk_content = chunk.get('content', '')
                        chunk_maker_pn = chunk.get('maker_pn', '')
                        chunk_part_number = chunk.get('part number', '')
                        chunk_grade = chunk.get('grade', '')
                    else:
                        # Old format: chunk is just a string
                        chunk_content = chunk
                        chunk_maker_pn = data.get('metadata', {}).get('maker_pn', '')
                        chunk_part_number = data.get('part number', '')
                        chunk_grade = data.get('metadata', {}).get('grade', '')
                    
                    # Create document with metadata
                    chunk_metadata = metadata_base.copy()
                    chunk_metadata.update({
                        'category': category,
                        'chunk_index': i,
                        'content_type': 'category_chunk',
                        'maker_pn': chunk_maker_pn,
                        'part_number': chunk_part_number,
                        'grade': chunk_grade
                    })
                    
                    if chunk_content.strip():  # Only add non-empty chunks
                        doc = Document(page_content=chunk_content, metadata=chunk_metadata)
                        documents.append(doc)
            
            # 2. Process page summaries as additional chunks
            for summary in data.get('page_summaries', []):
                if summary.get('is_useful', False):
                    page_content = summary.get('content', '')
                    if page_content.strip():
                        # Create document with metadata
                        page_metadata = metadata_base.copy()
                        page_metadata.update({
                            'page_number': summary.get('page_number', 0),
                            'categories': summary.get('categories', []),
                            'content_type': 'page_summary',
                            'maker_pn': summary.get('maker_pn', ''),
                            'part_number': summary.get('part number', ''),
                            'grade': summary.get('grade', '')
                        })
                        
                        doc = Document(page_content=page_content, metadata=page_metadata)
                        documents.append(doc)
        
        # Create vectorstore from documents
        if documents:
            self.logger.info(f"Creating vectorstore with {len(documents)} documents")
            vectorstore = FAISS.from_documents(documents, self.embeddings)
            
            # Move index to GPU if available
            if self.use_gpu and hasattr(vectorstore, 'index'):
                try:
                    vectorstore.index = faiss.index_cpu_to_gpu(self.res, 0, vectorstore.index)
                    self.logger.info("Moved vectorstore index to GPU")
                except Exception as e:
                    self.logger.warning(f"Failed to move index to GPU: {str(e)}")
            
            return vectorstore
        else:
            # Create empty vectorstore if no documents
            self.logger.warning("No documents found, creating empty vectorstore")
            return FAISS.from_texts(["No data available"], self.embeddings)
    
    def save_vectorstore(self, vectorstore: FAISS, path: str):
        """벡터 스토어를 파일로 저장"""
        try:
            # Move index to CPU before saving if using GPU
            if self.use_gpu and hasattr(vectorstore, 'index'):
                vectorstore.index = faiss.index_gpu_to_cpu(vectorstore.index)
            
            # Ensure directory exists
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            
            vectorstore.save_local(path)
            self.logger.info(f"Vectorstore saved to {path}")
            
        except Exception as e:
            self.logger.error(f"Error saving vectorstore: {str(e)}")
            raise
    
    def load_vectorstore(self, path: str) -> FAISS:
        """저장된 벡터 스토어를 로드"""
        try:
            vectorstore = FAISS.load_local(
                path, 
                self.embeddings,
                allow_dangerous_deserialization=True  # Required for FAISS loading
            )
            
            # Move index to GPU after loading if available
            if self.use_gpu and hasattr(vectorstore, 'index'):
                try:
                    vectorstore.index = faiss.index_cpu_to_gpu(self.res, 0, vectorstore.index)
                    self.logger.info("Moved loaded vectorstore index to GPU")
                except Exception as e:
                    self.logger.warning(f"Failed to move loaded index to GPU: {str(e)}")
            
            self.logger.info(f"Vectorstore loaded from {path}")
            return vectorstore
            
        except Exception as e:
            self.logger.error(f"Error loading vectorstore from {path}: {str(e)}")
            raise
    
    def search_with_filters(self, vectorstore: FAISS, query: str, 
                           filters: Dict[str, Any] = None, k: int = 5) -> List[Document]:
        """필터를 적용한 벡터 검색
        
        Args:
            vectorstore: FAISS vectorstore
            query: Search query
            filters: Metadata filters (e.g., {'maker_pn': 'LM324', 'category': 'Electrical Characteristics'})
            k: Number of results to return
            
        Returns:
            List of matching documents
        """
        try:
            if filters:
                # FAISS doesn't support native filtering, so we'll search more results and filter manually
                search_k = min(k * 10, 100)  # Search more to account for filtering
                docs = vectorstore.similarity_search(query, k=search_k)
                
                # Apply filters manually
                filtered_docs = []
                for doc in docs:
                    match = True
                    for filter_key, filter_value in filters.items():
                        doc_value = doc.metadata.get(filter_key, '')
                        if isinstance(filter_value, list):
                            if doc_value not in filter_value:
                                match = False
                                break
                        else:
                            if doc_value != filter_value:
                                match = False
                                break
                    
                    if match:
                        filtered_docs.append(doc)
                        if len(filtered_docs) >= k:
                            break
                
                return filtered_docs[:k]
            else:
                return vectorstore.similarity_search(query, k=k)
                
        except Exception as e:
            self.logger.error(f"Error in filtered search: {str(e)}")
            # Fallback to regular search
            return vectorstore.similarity_search(query, k=k)
    
    def get_vectorstore_info(self, vectorstore: FAISS) -> Dict[str, Any]:
        """벡터 스토어 정보 반환"""
        try:
            # Get basic info
            info = {
                'total_documents': vectorstore.index.ntotal if hasattr(vectorstore, 'index') else 0,
                'embedding_model': self.model_name,
                'device': 'GPU' if self.use_gpu else 'CPU'
            }
            
            # Get sample metadata keys if documents exist
            if info['total_documents'] > 0:
                try:
                    sample_docs = vectorstore.similarity_search("test", k=1)
                    if sample_docs:
                        info['available_metadata_keys'] = list(sample_docs[0].metadata.keys())
                except:
                    pass
            
            return info
            
        except Exception as e:
            self.logger.error(f"Error getting vectorstore info: {str(e)}")
            return {'error': str(e)} 