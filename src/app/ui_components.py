"""
UI components for the ChipChat Streamlit app
"""
import streamlit as st
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

def show_system_status(paths: Dict[str, str]):
    """시스템 상태를 표시하는 UI 컴포넌트"""
    with st.expander("📊 시스템 상태", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # 환경 감지
            try:
                from google.colab import drive
                env = "COLAB"
            except ImportError:
                env = "LOCAL"
            st.metric("환경", env)
        
        with col2:
            # ChipDB 확인
            chipdb_path = Path(paths['chipdb_path'])
            if chipdb_path.exists():
                try:
                    df = pd.read_csv(chipdb_path)
                    st.metric("ChipDB 부품", len(df))
                except:
                    st.metric("ChipDB", "❌ 읽기 실패")
            else:
                st.metric("ChipDB", "❌ 없음")
        
        with col3:
            # Vectorstore 확인
            vs_path = Path(paths['vectorstore_path'])
            if vs_path.exists():
                files = list(vs_path.glob("**/*.faiss")) + list(vs_path.glob("**/*.pkl"))
                st.metric("Vectorstore 파일", len(files))
            else:
                st.metric("Vectorstore", "❌ 없음")

def show_performance_metrics(logger):
    """성능 메트릭을 표시하는 UI 컴포넌트"""
    if hasattr(logger, 'get_performance_summary'):
        perf_summary = logger.get_performance_summary()
        if perf_summary:
            with st.expander("⏱️ 성능 요약", expanded=False):
                for op, stats in perf_summary.items():
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(f"{op}", f"{stats['success_count']}회")
                    with col2:
                        st.metric("평균", f"{stats['avg_time']:.1f}초")
                    with col3:
                        st.metric("최대", f"{stats['max_time']:.1f}초")

def show_llm_settings():
    """LLM 설정 UI 컴포넌트"""
    st.subheader("🤖 LLM 설정")
    
    provider = st.selectbox(
        "LLM 제공자",
        ["openai", "claude"],
        key="llm_provider"
    )
    
    if provider == "openai":
        model_options = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
    else:
        model_options = ["claude-3-sonnet-20240229", "claude-3-haiku-20240307"]
    
    model_name = st.selectbox(
        "모델",
        model_options,
        key="llm_model"
    )
    
    return provider, model_name

def show_chat_controls():
    """채팅 컨트롤 UI 컴포넌트"""
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("🔄 대화 초기화", use_container_width=True):
            st.session_state.messages = []
            st.success("대화가 초기화되었습니다.")
            st.rerun()
    
    with col2:
        if st.button("📋 에이전트 정보", use_container_width=True):
            return True
    
    return False

def show_agent_info(agent):
    """에이전트 정보를 표시하는 UI 컴포넌트"""
    with st.info("ℹ️"):
        st.markdown(agent.get_agent_info())

def init_chat_container():
    """채팅 컨테이너 초기화"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # 채팅 히스토리 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def add_chat_message(role: str, content: str):
    """채팅 메시지 추가"""
    st.session_state.messages.append({"role": role, "content": content})
    with st.chat_message(role):
        st.markdown(content)

def show_pdf_upload():
    """PDF 업로드 UI 컴포넌트"""
    st.subheader("📄 PDF 업로드")
    
    uploaded_file = st.file_uploader(
        "데이터시트 PDF 업로드",
        type=['pdf'],
        help="새로운 부품 데이터시트를 업로드하면 자동으로 처리되어 검색에 추가됩니다."
    )
    
    return uploaded_file

def show_session_documents():
    """세션에 업로드된 문서 목록 표시"""
    if hasattr(st.session_state, 'uploaded_documents') and st.session_state.uploaded_documents:
        with st.expander("📁 업로드된 문서", expanded=False):
            for i, doc in enumerate(st.session_state.uploaded_documents):
                col1, col2, col3 = st.columns([3, 2, 1])
                
                with col1:
                    st.text(f"📄 {doc['filename']}")
                    st.caption(f"{doc['component_name']} ({doc['manufacturer']})")
                
                with col2:
                    st.text(f"📄 {doc['total_pages']}페이지")
                    st.caption(f"📦 {doc['total_chunks']}청크")
                
                with col3:
                    if st.button("🗑️", key=f"remove_doc_{i}", help="문서 제거"):
                        return doc['filename']  # 제거할 파일명 반환
            
            # 세션 클리어 버튼
            if st.button("🗑️ 모든 문서 삭제", type="secondary"):
                return "clear_all"
    
    return None

def show_upload_status(status_message: str, status_type: str = "info"):
    """업로드 상태 표시"""
    if status_type == "success":
        st.success(status_message)
    elif status_type == "error":
        st.error(status_message)
    elif status_type == "warning":
        st.warning(status_message)
    else:
        st.info(status_message) 