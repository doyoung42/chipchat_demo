"""
Configuration settings for the Datasheet Analyzer application.
"""
from pathlib import Path
from typing import Dict, Any
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"

# Create directories if they don't exist
UPLOAD_DIR.mkdir(exist_ok=True)
VECTOR_STORE_DIR.mkdir(exist_ok=True)

# Model settings
EMBEDDING_MODEL = "sentence-transformers/paraphrase-MiniLM-L3-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# LLM settings
LLM_CONFIG: Dict[str, Any] = {
    "gpt4": {
        "model_name": "gpt-4-turbo-preview",
        "temperature": 0.1,
        "max_tokens": 2000,
    },
    "claude": {
        "model_name": "claude-3-sonnet-20240229",
        "temperature": 0.1,
        "max_tokens": 2000,
    }
}

# Vector store settings
VECTOR_STORE_CONFIG = {
    "collection_name": "datasheet_chunks",
    "embedding_function": None,  # Will be set at runtime
}

# Streamlit settings
STREAMLIT_CONFIG = {
    "page_title": "Datasheet Analyzer",
    "page_icon": "🔍",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}

# PDF viewer settings
PDF_VIEWER_CONFIG = {
    "zoom": 1.0,
    "page_width": 800,
    "page_height": 1000,
} 