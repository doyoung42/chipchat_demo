"""
Session-based vectorstore management for ChipChat
Manages separate vectorstores for each user session
"""
import streamlit as st
from typing import Dict, List, Any, Optional
from pathlib import Path
import tempfile
import shutil

from langchain_community.vectorstores import FAISS
from langchain.schema import Document

from .vectorstore_manager import VectorstoreManager
from .pdf_processor import PDFProcessor

class SessionVectorstoreManager:
    """세션별 벡터스토어 관리자"""
    
    def __init__(self, base_vectorstore_manager: VectorstoreManager):
        """
        Args:
            base_vectorstore_manager: 기본 벡터스토어 매니저
        """
        self.base_manager = base_vectorstore_manager
        
        # 세션 상태 초기화
        if 'session_vectorstore' not in st.session_state:
            st.session_state.session_vectorstore = None
        
        if 'uploaded_documents' not in st.session_state:
            st.session_state.uploaded_documents = []
        
        if 'session_json_data' not in st.session_state:
            st.session_state.session_json_data = []
    
    def get_combined_vectorstore(self, base_vectorstore):
        """기본 벡터스토어와 세션 벡터스토어를 결합"""
        if st.session_state.session_vectorstore is None:
            # 세션 벡터스토어가 없으면 기본 벡터스토어만 반환
            return base_vectorstore
        
        # 두 벡터스토어를 결합 (기본적으로 세션 벡터스토어를 우선 검색)
        return st.session_state.session_vectorstore
    
    def add_pdf_to_session(self, pdf_content: bytes, filename: str, 
                          pdf_processor: PDFProcessor) -> Dict[str, Any]:
        """PDF를 세션 벡터스토어에 추가"""
        try:
            # PDF 처리
            processed_data = pdf_processor.process_pdf(pdf_content, filename)
            
            # 세션 JSON 데이터에 추가
            st.session_state.session_json_data.append(processed_data)
            
            # 업로드된 문서 목록에 추가
            doc_info = {
                'filename': filename,
                'component_name': processed_data['metadata'].get('component_name', 'Unknown'),
                'manufacturer': processed_data['metadata'].get('manufacturer', 'Unknown'),
                'uploaded_at': processed_data['processing_info']['processed_at'],
                'total_pages': processed_data['processing_info']['total_pages'],
                'total_chunks': processed_data['processing_info']['total_chunks']
            }
            st.session_state.uploaded_documents.append(doc_info)
            
            # 세션 벡터스토어 업데이트
            self._update_session_vectorstore()
            
            return {
                'success': True,
                'document_info': doc_info,
                'processed_data': processed_data
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _update_session_vectorstore(self):
        """세션 JSON 데이터로 벡터스토어 업데이트"""
        try:
            if st.session_state.session_json_data:
                # 새 벡터스토어 생성
                new_vectorstore = self.base_manager.create_vectorstore(
                    st.session_state.session_json_data
                )
                st.session_state.session_vectorstore = new_vectorstore
        except Exception as e:
            print(f"세션 벡터스토어 업데이트 실패: {str(e)}")
    
    def search_session_vectorstore(self, query: str, k: int = 5) -> List[Document]:
        """세션 벡터스토어에서 검색"""
        if st.session_state.session_vectorstore is None:
            return []
        
        try:
            return st.session_state.session_vectorstore.similarity_search(query, k=k)
        except Exception as e:
            print(f"세션 벡터스토어 검색 실패: {str(e)}")
            return []
    
    def search_combined(self, base_vectorstore, query: str, k: int = 5) -> List[Document]:
        """기본 + 세션 벡터스토어에서 통합 검색"""
        results = []
        
        # 세션 벡터스토어에서 먼저 검색
        session_results = self.search_session_vectorstore(query, k//2)
        results.extend(session_results)
        
        # 기본 벡터스토어에서 검색
        try:
            base_results = base_vectorstore.similarity_search(query, k - len(session_results))
            results.extend(base_results)
        except Exception as e:
            print(f"기본 벡터스토어 검색 실패: {str(e)}")
        
        return results[:k]
    
    def get_session_info(self) -> Dict[str, Any]:
        """세션 정보 반환"""
        return {
            'uploaded_documents_count': len(st.session_state.uploaded_documents),
            'uploaded_documents': st.session_state.uploaded_documents,
            'has_session_vectorstore': st.session_state.session_vectorstore is not None,
            'total_session_chunks': sum(
                data['processing_info']['total_chunks'] 
                for data in st.session_state.session_json_data
            ) if st.session_state.session_json_data else 0
        }
    
    def clear_session(self):
        """세션 데이터 클리어"""
        st.session_state.session_vectorstore = None
        st.session_state.uploaded_documents = []
        st.session_state.session_json_data = []
    
    def remove_document(self, filename: str) -> bool:
        """특정 문서 제거"""
        try:
            # 업로드된 문서 목록에서 제거
            st.session_state.uploaded_documents = [
                doc for doc in st.session_state.uploaded_documents 
                if doc['filename'] != filename
            ]
            
            # JSON 데이터에서 제거
            st.session_state.session_json_data = [
                data for data in st.session_state.session_json_data 
                if data['filename'] != filename
            ]
            
            # 벡터스토어 재생성
            if st.session_state.session_json_data:
                self._update_session_vectorstore()
            else:
                st.session_state.session_vectorstore = None
            
            return True
            
        except Exception as e:
            print(f"문서 제거 실패: {str(e)}")
            return False 