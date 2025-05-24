"""
Main Streamlit application for the Datasheet Analyzer.
"""
import streamlit as st
from pathlib import Path
import json
from typing import Dict, Any, Optional
import tempfile
import os
import sys
import yaml
import logging
import traceback
from logging.handlers import RotatingFileHandler
import platform
import pandas as pd
import numpy as np
import pydantic
import torch

assert pd.__version__ >= "2.2.2", "pandas 버전이 너무 낮습니다. 구글 코랩 기본 버전을 사용하세요."
assert np.__version__ >= "1.26.0", "numpy 버전이 너무 낮습니다. 구글 코랩 기본 버전을 사용하세요."
assert int(pydantic.VERSION.split(".")[0]) >= 2, "pydantic 버전이 너무 낮습니다. 구글 코랩 기본 버전을 사용하세요."
assert int(torch.__version__.split(".")[0]) >= 2, "torch 버전이 너무 낮습니다. 구글 코랩 기본 버전을 사용하세요."

# Python 경로 설정
current_dir = Path(__file__).parent.parent.parent
sys.path.append(str(current_dir))
sys.path.append(str(current_dir / 'src'))

# 로그 디렉토리 설정
if platform.system() == 'Linux' and 'google.colab' in sys.modules:
    # Google Colab 환경
    LOG_DIR = Path('/tmp/streamlit_logs')
else:
    # 로컬 환경
    LOG_DIR = Path(__file__).parent.parent.parent / "logs"

LOG_DIR.mkdir(exist_ok=True, parents=True)
LOG_FILE = LOG_DIR / "streamlit.log"

# 로깅 설정
def setup_logging():
    """Setup logging configuration."""
    try:
        # 기존 핸들러 제거
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        # 로깅 설정
        logging.basicConfig(
            level=logging.DEBUG,  # 더 자세한 로깅을 위해 DEBUG 레벨로 설정
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                # 콘솔 출력
                logging.StreamHandler(),
                # 파일 출력 (최대 10MB, 최대 5개 파일 유지)
                RotatingFileHandler(
                    LOG_FILE,
                    maxBytes=10*1024*1024,  # 10MB
                    backupCount=5,
                    encoding='utf-8'
                )
            ]
        )
        
        # Streamlit 로거 설정
        streamlit_logger = logging.getLogger('streamlit')
        streamlit_logger.setLevel(logging.DEBUG)
        
        # 로깅 설정 완료 메시지
        logging.info(f"Logging setup completed. Log file: {LOG_FILE}")
        logging.info(f"Current working directory: {os.getcwd()}")
        logging.info(f"Python path: {sys.path}")
        
    except Exception as e:
        print(f"Error setting up logging: {str(e)}")
        print(traceback.format_exc())

# 로깅 설정 실행
setup_logging()
logger = logging.getLogger(__name__)

# Import after path setup
from src.config.settings import STREAMLIT_CONFIG, UPLOAD_DIR
from src.config.token_manager import TokenManager
from src.models.embedding import EmbeddingModel
from src.models.llm import LLMModel
from src.utils.pdf_processor import PDFProcessor
from src.utils.vector_store import VectorStore

def initialize_session_state():
    """Initialize session state with error handling."""
    try:
        logger.info("Initializing session state...")
        
        # 토큰 관리자 초기화
        if "token_manager" not in st.session_state:
            st.session_state.token_manager = TokenManager()
            logger.info("Token manager initialized")
        
        # PDF 프로세서 초기화
        if "pdf_processor" not in st.session_state:
            st.session_state.pdf_processor = PDFProcessor()
            logger.info("PDF processor initialized")
        
        # 임베딩 모델 초기화
        if "embedding_model" not in st.session_state:
            try:
                hf_token = st.session_state.token_manager.get_token('huggingface')
                st.session_state.embedding_model = EmbeddingModel(hf_token=hf_token)
                logger.info("Embedding model initialized")
            except Exception as e:
                logger.error(f"Error initializing embedding model: {str(e)}")
                st.error("임베딩 모델 초기화 중 오류가 발생했습니다. HuggingFace 토큰을 확인해주세요.")
                return False
        
        # 벡터 스토어 초기화
        if "vector_store" not in st.session_state:
            try:
                st.session_state.vector_store = VectorStore(
                    st.session_state.embedding_model.get_langchain_embeddings()
                )
                logger.info("Vector store initialized")
            except Exception as e:
                logger.error(f"Error initializing vector store: {str(e)}")
                st.error("벡터 스토어 초기화 중 오류가 발생했습니다.")
                return False
        
        # LLM 모델 상태 초기화
        if "llm_model" not in st.session_state:
            st.session_state.llm_model = None
            logger.info("LLM model state initialized")
        
        # 기타 상태 초기화
        if "current_pdf" not in st.session_state:
            st.session_state.current_pdf = None
        
        if "chunks" not in st.session_state:
            st.session_state.chunks = None
        
        if "search_results" not in st.session_state:
            st.session_state.search_results = None
        
        if "llm_response" not in st.session_state:
            st.session_state.llm_response = None
            
        logger.info("Session state initialization completed")
        return True
    except Exception as e:
        logger.error(f"Error initializing session state: {str(e)}")
        logger.error(traceback.format_exc())
        st.error(f"초기화 중 오류가 발생했습니다: {str(e)}")
        return False

def load_user_config():
    """Load user configuration from yaml file."""
    config_path = Path("user_config.yaml")
    if config_path.exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}

def save_user_config(config: Dict[str, Any]):
    """Save user configuration to yaml file."""
    config_path = Path("user_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)

def initialize_page():
    """Initialize the Streamlit page configuration."""
    st.set_page_config(**STREAMLIT_CONFIG)
    st.title("Datasheet Analyzer")

def setup_sidebar():
    """Setup the sidebar with controls."""
    with st.sidebar:
        st.header("Settings")
        
        # API 토큰 설정
        st.subheader("API Tokens")
        
        # HuggingFace 토큰
        hf_token = st.text_input(
            "HuggingFace API Token",
            type="password",
            value=st.session_state.token_manager.get_token('huggingface') or "",
            key="hf_token"
        )
        if hf_token:
            st.session_state.token_manager.set_token('huggingface', hf_token)
        
        # LLM Model Selection
        model_type = st.selectbox(
            "Select LLM Model",
            ["gpt4", "claude"],
            index=0,
            key="model_type"
        )
        
        # LLM API 토큰
        if model_type == "gpt4":
            api_key = st.text_input(
                "OpenAI API Key",
                type="password",
                value=st.session_state.token_manager.get_token('openai') or "",
                key="openai_token"
            )
            if api_key:
                st.session_state.token_manager.set_token('openai', api_key)
        else:  # claude
            api_key = st.text_input(
                "Anthropic API Key",
                type="password",
                value=st.session_state.token_manager.get_token('anthropic') or "",
                key="anthropic_token"
            )
            if api_key:
                st.session_state.token_manager.set_token('anthropic', api_key)
        
        # Apply settings
        if st.button("Apply Settings"):
            if api_key:
                try:
                    st.session_state.llm_model = LLMModel(model_type, api_key)
                    st.success("Settings applied successfully!")
                    logger.info(f"LLM model initialized: {model_type}")
                except Exception as e:
                    logger.error(f"Error initializing LLM model: {str(e)}")
                    st.error("LLM 모델 초기화 중 오류가 발생했습니다. API 토큰을 확인해주세요.")
            else:
                st.error("Please enter an API key")

def process_pdf():
    """Process the uploaded PDF file."""
    try:
        if st.session_state.current_pdf:
            logger.info(f"Processing PDF: {st.session_state.current_pdf}")
            chunks = st.session_state.pdf_processor.process_pdf(st.session_state.current_pdf)
            st.session_state.chunks = chunks
            logger.info(f"Created {len(chunks)} chunks from PDF")
            
            st.session_state.vector_store.create_store(chunks)
            logger.info("Vector store created successfully")
            return True
        return False
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        logger.error(traceback.format_exc())
        st.error(f"PDF 처리 중 오류가 발생했습니다: {str(e)}")
        return False

def display_chunks_and_search():
    """Display chunks and search functionality."""
    with st.expander("PDF Processing Settings", expanded=True):
        # Embedding model parameters
        chunk_size = st.slider("Chunk Size", 100, 2000, 500)
        chunk_overlap = st.slider("Chunk Overlap", 0, 500, 50)
        
        # Search settings
        search_text = st.text_input("Search Text")
        num_results = st.slider("Number of Results", 1, 10, 5)
        
        if search_text:
            results = st.session_state.vector_store.similarity_search(
                search_text,
                k=num_results
            )
            st.session_state.search_results = results
            
            st.subheader("Search Results")
            for i, result in enumerate(results, 1):
                st.text_area(f"Result {i}", result["text"], height=100)
    
    with st.expander("All Chunks", expanded=False):
        if st.session_state.chunks:
            for i, chunk in enumerate(st.session_state.chunks, 1):
                st.text_area(f"Chunk {i}", chunk["text"], height=100)

def display_llm_prompting():
    """Display LLM prompting interface."""
    st.header("LLM Prompting")
    
    # System prompt
    system_prompt = st.text_area(
        "System Prompt",
        """You are an expert in analyzing electronic component datasheets. 
Your task is to extract and organize key specifications and information from the provided datasheet.
Focus on technical details, specifications, and important characteristics of the component.
Maintain accuracy and use the original terminology from the datasheet.""",
        height=150
    )
    
    # RAG settings
    use_rag = st.checkbox("Use RAG (Vector Store Search Results)", value=True)
    
    # Output format prompt
    output_format = st.text_area(
        "Output Format Instructions",
        """Please provide the analysis in the following JSON format:
{
    "component_name": "string",
    "specifications": {
        "voltage": "string",
        "current": "string",
        "package": "string",
        "temperature_range": "string"
    },
    "key_features": ["string"],
    "applications": ["string"],
    "notes": "string"
}""",
        height=150
    )
    
    # Generate button
    if st.button("Generate Analysis"):
        if not st.session_state.llm_model:
            st.error("Please configure LLM settings in the sidebar")
            return
            
        # Prepare context
        if use_rag and st.session_state.search_results:
            context = "\n\n".join([r["text"] for r in st.session_state.search_results])
        else:
            context = "\n\n".join([c["text"] for c in st.session_state.chunks])
            
        # Prepare full prompt
        full_prompt = f"""System: {system_prompt}

Context from datasheet:
{context}

Output Format Instructions:
{output_format}"""
        
        # Get LLM response
        response = st.session_state.llm_model.get_model().invoke(full_prompt)
        st.session_state.llm_response = response.content
        
        # Display response
        st.subheader("Analysis Results")
        st.json(st.session_state.llm_response)

def get_available_pdfs():
    """Get list of available PDF files from uploads directory."""
    try:
        if not UPLOAD_DIR.exists():
            UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created uploads directory: {UPLOAD_DIR}")
        
        pdf_files = list(UPLOAD_DIR.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files in uploads directory")
        return pdf_files
    except Exception as e:
        logger.error(f"Error getting available PDFs: {str(e)}")
        logger.error(traceback.format_exc())
        return []

def main():
    """Main application function."""
    try:
        logger.info("Starting application...")
        initialize_session_state()
        initialize_page()
        setup_sidebar()
        
        # PDF 파일 선택 섹션
        st.header("PDF 파일 선택")
        
        # 기존 PDF 파일 목록 표시
        available_pdfs = get_available_pdfs()
        if available_pdfs:
            st.subheader("업로드된 PDF 파일")
            selected_pdf = st.selectbox(
                "기존 PDF 파일 선택",
                options=available_pdfs,
                format_func=lambda x: x.name,
                key="existing_pdf"
            )
            
            if selected_pdf:
                st.session_state.current_pdf = selected_pdf
                logger.info(f"Selected existing PDF: {selected_pdf}")
        
        # 새 PDF 파일 업로드
        st.subheader("새 PDF 파일 업로드")
        uploaded_file = st.file_uploader("PDF 파일 업로드", type=["pdf"])
        if uploaded_file:
            logger.info("New PDF file uploaded")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                # 업로드된 파일을 uploads 디렉토리로 복사
                upload_path = UPLOAD_DIR / uploaded_file.name
                with open(upload_path, 'wb') as f:
                    f.write(uploaded_file.getvalue())
                st.session_state.current_pdf = upload_path
                logger.info(f"PDF saved to: {upload_path}")
        
        if st.session_state.current_pdf:
            if process_pdf():
                # Create two columns for the main layout
                col1, col2 = st.columns(2)
                
                with col1:
                    st.header("PDF Analysis")
                    display_chunks_and_search()
                
                with col2:
                    display_llm_prompting()
        
        logger.info("Application running successfully")
    except Exception as e:
        logger.error(f"Error in main application: {str(e)}")
        logger.error(traceback.format_exc())
        st.error(f"애플리케이션 실행 중 오류가 발생했습니다: {str(e)}")

if __name__ == "__main__":
    main() 