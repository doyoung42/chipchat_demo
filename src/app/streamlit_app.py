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

# Add src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config.settings import STREAMLIT_CONFIG, PDF_VIEWER_CONFIG
from src.models.embedding import EmbeddingModel
from src.models.llm import LLMModel
from src.utils.pdf_processor import PDFProcessor
from src.utils.vector_store import VectorStore

# Initialize session state
if "pdf_processor" not in st.session_state:
    st.session_state.pdf_processor = PDFProcessor()
if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = EmbeddingModel()
if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStore(
        st.session_state.embedding_model.get_langchain_embeddings()
    )
if "llm_model" not in st.session_state:
    st.session_state.llm_model = None
if "current_pdf" not in st.session_state:
    st.session_state.current_pdf = None
if "specifications" not in st.session_state:
    st.session_state.specifications = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def initialize_page():
    """Initialize the Streamlit page configuration."""
    st.set_page_config(**STREAMLIT_CONFIG)
    st.title("Datasheet Analyzer")

def setup_sidebar():
    """Setup the sidebar with controls."""
    with st.sidebar:
        st.header("Controls")
        
        # PDF Upload
        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                st.session_state.current_pdf = Path(tmp_file.name)
        
        # LLM Model Selection
        model_type = st.selectbox(
            "Select LLM Model",
            ["gpt4", "claude"],
            index=0
        )
        
        # API Key Input
        api_key = st.text_input(
            f"Enter {model_type.upper()} API Key",
            type="password"
        )
        
        # Apply and Reset buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Apply"):
                if api_key:
                    st.session_state.llm_model = LLMModel(model_type, api_key)
                    st.success("API key applied successfully!")
                else:
                    st.error("Please enter an API key")
        
        with col2:
            if st.button("Reset"):
                st.session_state.llm_model = None
                st.session_state.specifications = None
                st.session_state.chat_history = []
                st.experimental_rerun()

def process_pdf():
    """Process the uploaded PDF file."""
    if st.session_state.current_pdf:
        chunks = st.session_state.pdf_processor.process_pdf(st.session_state.current_pdf)
        st.session_state.vector_store.create_store(chunks)
        return True
    return False

def display_pdf():
    """Display the PDF viewer."""
    if st.session_state.current_pdf:
        st.components.v1.iframe(
            str(st.session_state.current_pdf),
            height=PDF_VIEWER_CONFIG["page_height"],
            scrolling=True
        )

def extract_specifications():
    """Extract specifications from the PDF."""
    if not st.session_state.llm_model:
        st.error("Please select a model and enter API key")
        return
    
    # Create prompt for specification extraction
    prompt = """
    Please analyze the following datasheet and extract the following specifications:
    1. Function/Purpose
    2. Operating Voltage/Current Range
    3. Package Type
    4. Pin Descriptions
    5. JEDEC Compliance
    6. Key Diagrams
    7. Other Important Specifications
    
    Please maintain the original terminology and model numbers as they appear in the datasheet.
    Format the output in markdown.
    """
    
    # Get relevant chunks
    chunks = st.session_state.vector_store.similarity_search(prompt, k=5)
    context = "\n\n".join([chunk["text"] for chunk in chunks])
    
    # Generate specifications
    response = st.session_state.llm_model.get_model().invoke(
        f"{prompt}\n\nContext:\n{context}"
    )
    
    st.session_state.specifications = response.content

def display_specifications():
    """Display the extracted specifications."""
    if st.session_state.specifications:
        st.markdown(st.session_state.specifications)
        
        # Add feedback controls
        st.text_area("Improvement Suggestions", key="feedback")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Regenerate"):
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": st.session_state.feedback
                })
                extract_specifications()
        
        with col2:
            if st.button("Reset Chain"):
                st.session_state.chat_history = []
                st.session_state.specifications = None
        
        with col3:
            if st.button("Save Specifications"):
                if st.session_state.current_pdf:
                    output_file = st.session_state.current_pdf.with_suffix(".json")
                    with open(output_file, "w") as f:
                        json.dump({
                            "specifications": st.session_state.specifications,
                            "chat_history": st.session_state.chat_history
                        }, f, indent=2)
                    st.success(f"Specifications saved to {output_file}")

def main():
    """Main application function."""
    initialize_page()
    setup_sidebar()
    
    # Create two columns for the main layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("PDF Viewer")
        display_pdf()
    
    with col2:
        st.header("Specifications")
        if process_pdf():
            extract_specifications()
        display_specifications()

if __name__ == "__main__":
    main() 