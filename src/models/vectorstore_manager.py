import json
from pathlib import Path
from typing import List, Dict
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss

class VectorstoreManager:
    def __init__(self, hf_token: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """벡터 스토어 관리자 초기화"""
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            huggingface_api_token=hf_token,
            model_kwargs={'device': 'cuda'}  # Use GPU for embeddings
        )
        
        # Configure FAISS to use GPU
        self.res = faiss.StandardGpuResources()  # Use GPU resources
        
    def load_json_files(self, folder_path: str) -> List[Dict]:
        """JSON 파일들을 로드하여 리스트로 반환"""
        json_files = Path(folder_path).glob("*.json")
        return [json.loads(f.read_text()) for f in json_files]
    
    def create_vectorstore(self, json_data: List[Dict]) -> FAISS:
        """JSON 데이터로부터 벡터 스토어 생성"""
        # JSON 데이터를 텍스트로 변환
        texts = []
        for data in json_data:
            text = f"Product: {data.get('product_name', '')}\n"
            text += f"Features: {', '.join(data.get('key_features', []))}\n"
            text += f"Specifications: {json.dumps(data.get('specifications', {}), indent=2)}\n"
            text += f"Applications: {', '.join(data.get('applications', []))}\n"
            text += f"Notes: {data.get('notes', '')}\n"
            texts.append(text)
        
        # 텍스트 분할
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.split_text("\n".join(texts))
        
        # 벡터 스토어 생성 (GPU 사용)
        vectorstore = FAISS.from_texts(texts, self.embeddings)
        
        # Move index to GPU
        if hasattr(vectorstore, 'index'):
            vectorstore.index = faiss.index_cpu_to_gpu(self.res, 0, vectorstore.index)
        
        return vectorstore
    
    def save_vectorstore(self, vectorstore: FAISS, path: str):
        """벡터 스토어를 파일로 저장"""
        # Move index to CPU before saving
        if hasattr(vectorstore, 'index'):
            vectorstore.index = faiss.index_gpu_to_cpu(vectorstore.index)
            
        vectorstore.save_local(path)
    
    def load_vectorstore(self, path: str) -> FAISS:
        """저장된 벡터 스토어를 로드"""
        vectorstore = FAISS.load_local(path, self.embeddings)
        
        # Move index to GPU after loading
        if hasattr(vectorstore, 'index'):
            vectorstore.index = faiss.index_cpu_to_gpu(self.res, 0, vectorstore.index)
            
        return vectorstore 