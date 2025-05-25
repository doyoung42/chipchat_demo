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

# 인코딩 설정 추가
os.environ['PYTHONIOENCODING'] = 'utf-8'
if sys.stdout.encoding != 'utf-8':
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
    sys.stderr = open(sys.stderr.fileno(), mode='w', encoding='utf-8', buffering=1)

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
            
        # PDF 분석 상태 플래그 추가
        if "pdf_analysis_complete" not in st.session_state:
            st.session_state.pdf_analysis_complete = False
            
        # PDF 분석 자동 시작 플래그 추가
        if "auto_start_analysis" not in st.session_state:
            st.session_state.auto_start_analysis = False
            
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
        # LLM 모델/토큰
        st.subheader("API Tokens & LLM Model")
        hf_token = st.text_input("HuggingFace API Token", type="password", value=st.session_state.token_manager.get_token('huggingface') or "", key="hf_token")
        if hf_token:
            st.session_state.token_manager.set_token('huggingface', hf_token)
        openai_token = st.text_input("OpenAI API Key", type="password", value=st.session_state.token_manager.get_token('openai') or "", key="openai_token")
        if openai_token:
            st.session_state.token_manager.set_token('openai', openai_token)
        anthropic_token = st.text_input("Anthropic API Key", type="password", value=st.session_state.token_manager.get_token('anthropic') or "", key="anthropic_token")
        if anthropic_token:
            st.session_state.token_manager.set_token('anthropic', anthropic_token)
            
        # 임베딩 모델 선택
        st.subheader("임베딩 모델 설정")
        embedding_models = [
            "sentence-transformers/paraphrase-MiniLM-L3-v2",
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
        ]
        selected_model = st.selectbox(
            "임베딩 모델 선택",
            embedding_models,
            index=0,
            key="embedding_model_name"
        )
        
        if st.button("임베딩 모델 적용"):
            try:
                hf_token = st.session_state.token_manager.get_token('huggingface')
                st.session_state.embedding_model = EmbeddingModel(model_name=selected_model, hf_token=hf_token)
                st.session_state.vector_store = VectorStore(
                    st.session_state.embedding_model.get_langchain_embeddings()
                )
                st.success(f"{selected_model} 모델 적용 완료!")
            except Exception as e:
                st.error(f"임베딩 모델 초기화 오류: {str(e)}")
            
        # 사용 가능한 모델만 선택지로 노출
        available_models = []
        if openai_token:
            available_models += ["gpt-4o-2024-08-06"]
        if anthropic_token:
            available_models += ["claude-3-7-sonnet-20250219"]
        model_type = st.selectbox("Select LLM Model", available_models, key="model_type") if available_models else None
        if model_type and (openai_token or anthropic_token):
            if st.button("LLM 모델 적용"):
                try:
                    if "gpt" in model_type:
                        st.session_state.llm_model = LLMModel("gpt4", openai_token)
                    elif "claude" in model_type:
                        st.session_state.llm_model = LLMModel("claude", anthropic_token)
                    st.success(f"{model_type} 모델 적용 완료!")
                except Exception as e:
                    st.error(f"LLM 모델 초기화 오류: {str(e)}")
        # PDF 파일 선택
        st.subheader("PDF 파일 선택")
        available_pdfs = get_available_pdfs()
        selected_pdf = st.selectbox("기존 PDF 파일 선택", options=available_pdfs, format_func=lambda x: x.name if hasattr(x, 'name') else str(x), key="existing_pdf") if available_pdfs else None
        if selected_pdf:
            if st.button("PDF 선택 완료"):
                st.session_state.current_pdf = selected_pdf
                st.session_state.chunks = None  # 새 파일 선택 시 청킹 초기화
                st.session_state.search_results = None
                st.session_state.pdf_analysis_complete = False  # 분석 상태 초기화
                st.session_state.auto_start_analysis = True  # 자동 분석 시작 플래그 설정
                st.success(f"{selected_pdf.name if hasattr(selected_pdf, 'name') else str(selected_pdf)} 선택 완료!")
                st.rerun()  # 페이지 리로드하여 자동 분석 시작

def process_pdf():
    """Process the uploaded PDF file."""
    try:
        if st.session_state.current_pdf:
            logger.info(f"Processing PDF: {st.session_state.current_pdf}")
            
            # 프로그레스 바 추가
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # 텍스트 추출
            status_text.text("PDF에서 텍스트 추출 중...")
            progress_bar.progress(20)
            
            chunks = st.session_state.pdf_processor.process_pdf(st.session_state.current_pdf)
            st.session_state.chunks = chunks
            logger.info(f"Created {len(chunks)} chunks from PDF")
            
            # 벡터 스토어 생성
            status_text.text("벡터 스토어 생성 중...")
            progress_bar.progress(60)
            
            st.session_state.vector_store.create_store(chunks)
            logger.info("Vector store created successfully")
            
            # 완료
            progress_bar.progress(100)
            status_text.text("PDF 처리 완료!")
            
            # 분석 완료 플래그 설정
            st.session_state.pdf_analysis_complete = True
            
            # 페이지 리로드하여 다음 화면으로 자동 전환
            st.rerun()
            
            return True
        return False
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        logger.error(traceback.format_exc())
        st.error(f"PDF 처리 중 오류가 발생했습니다: {str(e)}")
        st.session_state.pdf_analysis_complete = False
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
    
    # System prompt (높이 절반으로 줄임)
    system_prompt = st.text_area(
        "System Prompt",
        """You are an expert in analyzing electronic component datasheets. 
Your task is to extract and organize key specifications and information from the provided datasheet.
Focus on technical details, specifications, and important characteristics of the component.
Maintain accuracy and use the original terminology from the datasheet.""",
        height=200,  # 기존 400에서 200으로 줄임
        key="system_prompt"
    )
    
    # RAG 설정
    use_rag = st.checkbox("Use RAG (Vector Store Search Results)", value=True, help="체크하면 검색 결과만 사용, 체크 해제하면 전체 PDF 내용 사용", key="use_rag")
    
    # RAG 설정에 따른 설명 및 추가 입력 필드
    if use_rag:
        st.info("RAG 모드: 왼쪽 검색 결과를 LLM에 전달합니다. 검색 결과가 없으면 전체 PDF 내용이 사용됩니다.")
    else:
        st.info("전체 PDF 내용이 LLM에 전달됩니다.")
    
    # Output format prompt (높이 2배로 늘림)
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
        height=400,  # 기존 200에서 400으로 늘림
        key="output_format"
    )
    
    # Generate 버튼
    if st.button("Generate Analysis"):
        if not st.session_state.llm_model:
            st.error("Please configure LLM settings in the sidebar")
            return
            
        # 프로그레스 바 추가
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text("분석 준비 중...")
        
        try:
            # 컨텍스트 준비
            context = ""
            
            # RAG 모드 사용 시
            if use_rag:
                # 왼쪽 검색 결과가 있는지 확인
                if st.session_state.get("search_results") and len(st.session_state.search_results) > 0:
                    context = "\n\n".join([r["text"] for r in st.session_state.search_results])
                    status_text.text(f"RAG 모드: 검색 결과 {len(st.session_state.search_results)}개 사용")
                # 없으면 전체 PDF 사용
                else:
                    context = "\n\n".join([c["text"] for c in st.session_state.chunks])
                    status_text.text("검색 결과가 없어 전체 PDF 내용 사용")
            # RAG 모드 미사용 시
            else:
                context = "\n\n".join([c["text"] for c in st.session_state.chunks])
                status_text.text("전체 PDF 내용 사용")
            
            progress_bar.progress(30)
            status_text.text("LLM에 요청 전송 중...")
                
            # Prepare full prompt
            full_prompt = f"""System: {system_prompt}

Context from datasheet:
{context}

Output Format Instructions:
{output_format}

IMPORTANT: Return only valid JSON without any markdown code blocks or backticks."""
            
            progress_bar.progress(50)
            
            # Get LLM response with timeout handling
            response = st.session_state.llm_model.get_model().invoke(full_prompt)
            response_content = response.content
            
            # 유니코드 문자 처리를 위한 인코딩 처리 추가
            response_content = st.session_state.llm_model.process_response(response_content)
            
            # Claude가 반환한 응답에서 코드 블록 마크다운 제거
            response_content = response_content.replace("```json", "").replace("```", "").strip()
            
            st.session_state.llm_response = response_content
            
            progress_bar.progress(100)
            status_text.text("분석 완료!")
            
            # Display response
            st.subheader("Analysis Results")
            try:
                st.json(st.session_state.llm_response)
            except Exception as json_error:
                logger.error(f"JSON 파싱 오류: {str(json_error)}")
                st.error(f"JSON 형식이 올바르지 않습니다. 원본 응답을 표시합니다.")
                st.text_area("Raw Response", st.session_state.llm_response, height=400)
            
        except Exception as e:
            logger.error(f"Error generating LLM response: {str(e)}")
            logger.error(traceback.format_exc())
            st.error(f"LLM 응답 생성 중 오류가 발생했습니다: {str(e)}")
            progress_bar.empty()
            status_text.empty()

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
        
        if st.session_state.get("current_pdf"):
            # PDF 분석(청킹)
            if st.session_state.chunks is None:
                # 자동 분석 시작 플래그 확인
                if st.session_state.get("auto_start_analysis", False):
                    # 자동 분석 플래그 초기화
                    st.session_state.auto_start_analysis = False
                    # 자동으로 PDF 분석 시작
                    process_pdf()
                else:
                    # 수동 분석 버튼 표시
                    if st.button("PDF 분석 시작"):
                        process_pdf()
            else:
                # 2컬럼 레이아웃 설정
                col1, col2 = st.columns(2)
                
                # 왼쪽 컬럼: PDF Analysis & Retrieval
                with col1:
                    st.header("PDF Analysis & Retrieval")
                    
                    # 검색 파라미터 설정
                    st.subheader("검색 설정")
                    with st.form("search_form"):
                        search_text = st.text_input("Retrieval (예: voltage, 특징, 적용분야 등)", key="search_text")
                        
                        # 검색 파라미터 상세화
                        st.write("검색 파라미터 설정")
                        num_results = st.slider("검색 결과 개수", 1, 20, 5, key="num_results")
                        search_threshold = st.slider("유사도 임계값", 0.0, 1.0, 0.7, step=0.05, key="search_threshold")
                        search_method = st.selectbox(
                            "검색 방법",
                            ["Similarity", "MMR (Maximum Marginal Relevance)"],
                            index=0,
                            key="search_method"
                        )
                        
                        if search_method == "MMR (Maximum Marginal Relevance)":
                            diversity = st.slider("다양성 (높을수록 다양한 결과)", 0.0, 1.0, 0.3, step=0.05, key="diversity")
                        else:
                            diversity = 0.3  # 기본값
                        
                        submitted = st.form_submit_button("검색")
                    
                    if submitted and search_text:
                        # 검색 방법에 따라 파라미터 설정
                        use_mmr = search_method == "MMR (Maximum Marginal Relevance)"
                        
                        # 검색 실행
                        results = st.session_state.vector_store.similarity_search(
                            search_text, 
                            k=num_results,
                            threshold=search_threshold,
                            use_mmr=use_mmr,
                            diversity=diversity
                        )
                        st.session_state.search_results = results
                    
                    # 검색 결과 표시
                    if st.session_state.get("search_results"):
                        st.subheader("Search Results")
                        for i, result in enumerate(st.session_state.search_results, 1):
                            st.text_area(f"Result {i}", result["text"], height=100)
                    
                    # 전체 청크 보기
                    with st.expander("All Chunks", expanded=False):
                        for i, chunk in enumerate(st.session_state.chunks, 1):
                            st.text_area(f"Chunk {i}", chunk["text"], height=100)
                
                # 오른쪽 컬럼: LLM Prompting
                with col2:
                    st.header("LLM Prompting")
                    
                    # System prompt (높이 절반으로 줄임)
                    system_prompt = st.text_area(
                        "System Prompt",
                        """You are an expert in analyzing electronic component datasheets. 
Your task is to extract and organize key specifications and information from the provided datasheet.
Focus on technical details, specifications, and important characteristics of the component.
Maintain accuracy and use the original terminology from the datasheet.""",
                        height=200,  # 기존 400에서 200으로 줄임
                        key="system_prompt"
                    )
                    
                    # RAG 설정
                    use_rag = st.checkbox("Use RAG (Vector Store Search Results)", value=True, help="체크하면 검색 결과만 사용, 체크 해제하면 전체 PDF 내용 사용", key="use_rag")
                    
                    # RAG 설정에 따른 설명
                    if use_rag:
                        st.info("RAG 모드: 왼쪽 검색 결과를 LLM에 전달합니다. 검색 결과가 없으면 전체 PDF 내용이 사용됩니다.")
                    else:
                        st.info("전체 PDF 내용이 LLM에 전달됩니다.")
                    
                    # Output format prompt (높이 2배로 늘림)
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
                        height=400,  # 기존 200에서 400으로 늘림
                        key="output_format"
                    )
                    
                    # Generate 버튼
                    if st.button("Generate Analysis"):
                        if not st.session_state.llm_model:
                            st.error("Please configure LLM settings in the sidebar")
                        else:
                            # 프로그레스 바 추가
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            status_text.text("분석 준비 중...")
                            
                            try:
                                # 컨텍스트 준비
                                context = ""
                                
                                # RAG 모드 사용 시
                                if use_rag:
                                    # 왼쪽 검색 결과가 있는지 확인
                                    if st.session_state.get("search_results") and len(st.session_state.search_results) > 0:
                                        context = "\n\n".join([r["text"] for r in st.session_state.search_results])
                                        status_text.text(f"RAG 모드: 검색 결과 {len(st.session_state.search_results)}개 사용")
                                    # 없으면 전체 PDF 사용
                                    else:
                                        context = "\n\n".join([c["text"] for c in st.session_state.chunks])
                                        status_text.text("검색 결과가 없어 전체 PDF 내용 사용")
                                # RAG 모드 미사용 시
                                else:
                                    context = "\n\n".join([c["text"] for c in st.session_state.chunks])
                                    status_text.text("전체 PDF 내용 사용")
                                
                                progress_bar.progress(30)
                                status_text.text("LLM에 요청 전송 중...")
                                
                                full_prompt = f"""System: {system_prompt}\n\nContext from datasheet:\n{context}\n\nOutput Format Instructions:\n{output_format}\n\nIMPORTANT: Return only valid JSON without any markdown code blocks or backticks."""
                                
                                progress_bar.progress(50)
                                
                                # LLM 응답 생성
                                response = st.session_state.llm_model.get_model().invoke(full_prompt)
                                response_content = response.content
                                
                                # 유니코드 문자 처리를 위한 인코딩 처리 추가
                                response_content = st.session_state.llm_model.process_response(response_content)
                                
                                # Claude가 반환한 응답에서 코드 블록 마크다운 제거
                                response_content = response_content.replace("```json", "").replace("```", "").strip()
                                
                                st.session_state.llm_response = response_content
                                
                                progress_bar.progress(100)
                                status_text.text("분석 완료!")
                                
                                # 결과 표시
                                st.subheader("Analysis Results")
                                try:
                                    st.json(st.session_state.llm_response)
                                except Exception as json_error:
                                    logger.error(f"JSON 파싱 오류: {str(json_error)}")
                                    st.error(f"JSON 형식이 올바르지 않습니다. 원본 응답을 표시합니다.")
                                    st.text_area("Raw Response", st.session_state.llm_response, height=400)
                                
                            except Exception as e:
                                logger.error(f"Error generating LLM response: {str(e)}")
                                logger.error(traceback.format_exc())
                                st.error(f"LLM 응답 생성 중 오류가 발생했습니다: {str(e)}")
                                progress_bar.empty()
                                status_text.empty()
        else:
            st.info("사이드바에서 PDF 파일을 선택하고 'PDF 선택 완료'를 눌러주세요.")
        
        logger.info("Application running successfully")
    except Exception as e:
        logger.error(f"Error in main application: {str(e)}")
        logger.error(traceback.format_exc())
        st.error(f"애플리케이션 실행 중 오류가 발생했습니다: {str(e)}")

if __name__ == "__main__":
    main() 