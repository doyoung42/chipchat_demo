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

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.settings import STREAMLIT_CONFIG, UPLOAD_DIR
from models.embedding import EmbeddingModel
from models.llm import LLMModel
from utils.pdf_processor import PDFProcessor
from utils.vector_store import VectorStore

def initialize_session_state():
    """Initialize session state with error handling."""
    try:
        logger.info("Initializing session state...")
        if "pdf_processor" not in st.session_state:
            st.session_state.pdf_processor = PDFProcessor()
            logger.info("PDF processor initialized")
        
        if "embedding_model" not in st.session_state:
            st.session_state.embedding_model = EmbeddingModel()
            logger.info("Embedding model initialized")
        
        if "vector_store" not in st.session_state:
            st.session_state.vector_store = VectorStore(
                st.session_state.embedding_model.get_langchain_embeddings()
            )
            logger.info("Vector store initialized")
        
        if "llm_model" not in st.session_state:
            st.session_state.llm_model = None
            logger.info("LLM model state initialized")
        
        if "current_pdf" not in st.session_state:
            st.session_state.current_pdf = None
        
        if "chunks" not in st.session_state:
            st.session_state.chunks = None
        
        if "search_results" not in st.session_state:
            st.session_state.search_results = None
        
        if "llm_response" not in st.session_state:
            st.session_state.llm_response = None
            
        logger.info("Session state initialization completed")
    except Exception as e:
        logger.error(f"Error initializing session state: {str(e)}")
        logger.error(traceback.format_exc())
        st.error(f"초기화 중 오류가 발생했습니다: {str(e)}")

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
        
        # Load saved config
        user_config = load_user_config()
        
        # LLM Model Selection
        model_type = st.selectbox(
            "Select LLM Model",
            ["gpt4", "claude"],
            index=0,
            key="model_type"
        )
        
        # API Key Input
        api_key = st.text_input(
            f"Enter {model_type.upper()} API Key",
            type="password",
            value=user_config.get("api_key", ""),
            key="api_key"
        )
        
        # Save settings
        if st.button("Save Settings"):
            save_user_config({
                "model_type": model_type,
                "api_key": api_key
            })
            st.success("Settings saved!")
        
        # Apply settings
        if st.button("Apply Settings"):
            if api_key:
                st.session_state.llm_model = LLMModel(model_type, api_key)
                st.success("Settings applied successfully!")
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