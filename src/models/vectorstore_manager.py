import json
import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from ..config.settings import PATHS

class VectorstoreManager:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """벡터 스토어 관리자 초기화 (CPU 전용)
        
        Args:
            model_name: 사용할 임베딩 모델명
        """
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load HuggingFace token from various sources
        hf_token = self._load_hf_token()
        
        # Use vectorstore path from settings (automatically detects environment)
        self.vectorstore_folder = str(PATHS['vectorstore'])
        self.logger.info(f"✅ 벡터스토어 경로 설정: {self.vectorstore_folder}")
        
        # Create vectorstore folder if it doesn't exist
        os.makedirs(self.vectorstore_folder, exist_ok=True)
        
        # Initialize embeddings (CPU only for Google Colab compatibility)
        embedding_kwargs = {'model_name': model_name}
        # Note: HuggingFaceEmbeddings now uses environment variables (HF_TOKEN, HUGGINGFACE_API_KEY)
        # instead of direct token parameter
        
        # Always use CPU for stability and compatibility
        embedding_kwargs['model_kwargs'] = {'device': 'cpu'}
        self.embeddings = HuggingFaceEmbeddings(**embedding_kwargs)
        self.logger.info("✅ 임베딩 모델 초기화 완료 (device: cpu)")
        
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
        
        if not hf_token:
            # Try loading from key.json files (prep and main)
            key_paths = [
                # Check prep folder first
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'prep', 'misc', 'key.json'),
                # Check current project root
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'misc', 'key.json')
            ]
            
            for key_path in key_paths:
                try:
                    if os.path.exists(key_path):
                        with open(key_path, 'r') as f:
                            keys = json.load(f)
                        hf_token = keys.get('huggingface_api_key', '')
                        if hf_token:
                            self.logger.info(f"✅ HuggingFace 토큰을 {key_path}에서 로드함")
                            break
                except Exception as e:
                    self.logger.warning(f"키 파일 로드 실패 {key_path}: {str(e)}")
        
        # Set environment variable if token found
        if hf_token:
            os.environ['HF_TOKEN'] = hf_token
            os.environ['HUGGINGFACE_API_KEY'] = hf_token
            self.logger.info("✅ HuggingFace 토큰 환경변수 설정 완료")
        
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
            self.logger.info(f"📄 총 {len(documents)}개 문서로 벡터스토어 생성 중... (CPU 모드)")
            vectorstore = FAISS.from_documents(documents, self.embeddings)
            
            # FAISS index is always CPU for CPU-only mode
            self.logger.info("✅ 벡터스토어 생성 완료 (CPU 모드)")
            
            return vectorstore
        else:
            # Create empty vectorstore if no documents
            self.logger.warning("⚠️ 문서가 없어 빈 벡터스토어 생성")
            return FAISS.from_texts(["No data available"], self.embeddings)
    
    def save_vectorstore(self, vectorstore: FAISS, name: str):
        """벡터 스토어를 파일로 저장"""
        try:
            path = os.path.join(self.vectorstore_folder, name)
            
            # FAISS index is always CPU for CPU-only mode
            self.logger.info("💾 벡터스토어 저장 중 (CPU 모드)")
            
            # Ensure directory exists
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            
            vectorstore.save_local(path)
            self.logger.info(f"✅ 벡터스토어 저장 완료: {path}")
            
        except Exception as e:
            self.logger.error(f"❌ 벡터스토어 저장 실패: {str(e)}")
            raise
    
    def load_vectorstore(self, name: str) -> FAISS:
        """저장된 벡터 스토어를 로드"""
        try:
            path = os.path.join(self.vectorstore_folder, name)
            self.logger.info(f"벡터스토어 로드 중: {path}")
            vectorstore = FAISS.load_local(
                path, 
                self.embeddings,
                allow_dangerous_deserialization=True  # Required for FAISS loading
            )
            
            # FAISS index is always CPU for CPU-only mode
            self.logger.info("✅ 벡터스토어 로드 완료 (CPU 모드)")
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
                'device': 'CPU'  # Always CPU for CPU-only mode
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