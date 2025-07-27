import json
import os
from pathlib import Path
from typing import List, Dict, Any
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

class VectorstoreManager:
    def __init__(self):
        """벡터 스토어 관리자 초기화 (CPU 전용)"""
        # Load API keys from key.json
        key_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'misc', 'key.json')
        with open(key_path, 'r') as f:
            keys = json.load(f)
        
        # Set HuggingFace API key as environment variable (multiple formats for compatibility)
        if keys.get('huggingface_api_key'):
            # Set both common environment variable names
            os.environ["HUGGINGFACE_API_KEY"] = keys.get('huggingface_api_key')
            os.environ["HF_TOKEN"] = keys.get('huggingface_api_key')
            print("✅ HuggingFace 토큰 설정 완료")
        
        # Load parameters from param.json
        param_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'misc', 'param.json')
        with open(param_path, 'r') as f:
            params = json.load(f)
        
        self.params = params['vectorstore']
        
        # Create vectorstore folder if it doesn't exist
        os.makedirs(self.params['folders']['vectorstore_folder'], exist_ok=True)
        
        # Initialize embeddings (CPU only for Google Colab compatibility)
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.params['model_name'],
                model_kwargs={'device': 'cpu'}
            )
            print("✅ 임베딩 모델 초기화 완료 (device: cpu)")
        except Exception as e:
            print(f"❌ 임베딩 모델 초기화 실패: {str(e)}")
            raise
        
    def load_json_files(self, folder_path: str) -> List[Dict]:
        """JSON 파일들을 로드하여 리스트로 반환"""
        json_files = Path(folder_path).glob("*.json")
        return [json.loads(f.read_text(encoding='utf-8')) for f in json_files]
    
    def create_vectorstore(self, json_data: List[Dict]) -> FAISS:
        """JSON 데이터로부터 벡터 스토어 생성
        
        새로운 JSON 구조:
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
                    
                    doc = Document(page_content=chunk_content, metadata=chunk_metadata)
                    documents.append(doc)
            
            # 2. Process page summaries as additional chunks
            for summary in data.get('page_summaries', []):
                if summary.get('is_useful', False):
                    page_content = summary.get('content', '')
                    if page_content:
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
        
        # Create vectorstore from documents (CPU only)
        if documents:
            print(f"📄 총 {len(documents)}개 문서로 벡터스토어 생성 중... (CPU 모드)")
            vectorstore = FAISS.from_documents(documents, self.embeddings)
            print("✅ 벡터스토어 생성 완료")
            return vectorstore
        else:
            # Create empty vectorstore if no documents
            print("⚠️ 문서가 없어 빈 벡터스토어 생성")
            return FAISS.from_texts(["No data available"], self.embeddings)
    
    def save_vectorstore(self, vectorstore: FAISS, name: str):
        """벡터 스토어를 파일로 저장"""
        path = os.path.join(self.params['folders']['vectorstore_folder'], name)
        vectorstore.save_local(path)
        print(f"✅ 벡터스토어 저장 완료: {path}")
    
    def load_vectorstore(self, name: str) -> FAISS:
        """저장된 벡터 스토어를 로드"""
        path = os.path.join(self.params['folders']['vectorstore_folder'], name)
        vectorstore = FAISS.load_local(path, self.embeddings)
        print(f"✅ 벡터스토어 로드 완료: {path}")
        return vectorstore 