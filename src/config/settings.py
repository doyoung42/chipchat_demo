"""
Configuration settings for the ChipChat application.
Supports both local and Google Colab environments.
"""
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def detect_environment() -> str:
    """Detect current environment (local or colab)"""
    try:
        import google.colab
        return "colab"
    except ImportError:
        return "local"

def get_base_paths() -> Dict[str, Path]:
    """Get base paths based on environment"""
    env = detect_environment()
    
    if env == "colab":
        # Google Colab paths
        base_dir = Path('/content/drive/MyDrive')
        return {
            'base': base_dir,
            'vectorstore': base_dir / 'vectorstore',
            'json_data': base_dir / 'prep_json',
            'prompt_templates': base_dir / 'prompt_templates',
            'uploads': base_dir / 'uploads'
        }
    else:
        # Local environment paths
        base_dir = Path(__file__).parent.parent.parent
        return {
            'base': base_dir,
            'vectorstore': base_dir / 'vectorstore',
            'json_data': base_dir / 'prep' / 'prep_json',
            'prompt_templates': base_dir / 'prompt_templates',
            'uploads': base_dir / 'uploads'
        }

# Get paths based on environment
PATHS = get_base_paths()

# Create directories if they don't exist
for path in PATHS.values():
    if path != PATHS['base']:
        path.mkdir(parents=True, exist_ok=True)

# Model settings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# LLM Configuration
LLM_CONFIG: Dict[str, Any] = {
    "openai": {
        "models": {
            "gpt-4o-mini": {
                "model_name": "gpt-4o-mini",
                "temperature": 0.1,
                "max_tokens": 2000,
                "description": "Fast and cost-effective model"
            },
            "gpt-4o": {
                "model_name": "gpt-4o",
                "temperature": 0.1,
                "max_tokens": 2000,
                "description": "Most capable model"
            },
            "gpt-3.5-turbo": {
                "model_name": "gpt-3.5-turbo",
                "temperature": 0.1,
                "max_tokens": 2000,
                "description": "Legacy model"
            }
        },
        "default_model": "gpt-4o-mini",
        "api_url": "https://api.openai.com/v1/chat/completions"
    },
    "claude": {
        "models": {
            "claude-3-sonnet-20240229": {
                "model_name": "claude-3-sonnet-20240229",
                "temperature": 0.1,
                "max_tokens": 2000,
                "description": "Balanced performance and speed"
            },
            "claude-3-haiku-20240307": {
                "model_name": "claude-3-haiku-20240307",
                "temperature": 0.1,
                "max_tokens": 2000,
                "description": "Fastest Claude model"
            },
            "claude-3-opus-20240229": {
                "model_name": "claude-3-opus-20240229",
                "temperature": 0.1,
                "max_tokens": 2000,
                "description": "Most capable Claude model"
            }
        },
        "default_model": "claude-3-sonnet-20240229",
        "api_url": "https://api.anthropic.com/v1/messages"
    }
}

# Vector store settings
VECTOR_STORE_CONFIG = {
    "collection_name": "datasheet_chunks",
    "embedding_model": EMBEDDING_MODEL,
    "chunk_size": CHUNK_SIZE,
    "chunk_overlap": CHUNK_OVERLAP,
    "use_gpu": True,  # Will fallback to CPU if GPU not available
}

# Streamlit settings
STREAMLIT_CONFIG = {
    "page_title": "ChipChat - 데이터시트 챗봇",
    "page_icon": "💬",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}

# PDF processing settings (for reference)
PDF_PROCESSING_CONFIG = {
    "pages_per_chunk": 3,
    "categories": [
        "Product Summary",
        "Electrical Characteristics", 
        "Application Circuits",
        "Mechanical Characteristics",
        "Reliability and Environmental Conditions",
        "Packaging Information"
    ]
}

# Default prompt templates
DEFAULT_PROMPT_TEMPLATES = {
    "korean": {
        "pre": "당신은 전자 부품 데이터시트에 대해 응답하는 전문 도우미입니다. 제공된 컨텍스트 정보를 기반으로 질문에 정확하고 상세하게 답변하세요.",
        "post": "검색된 정보를 바탕으로 명확하고 간결하게 답변해주세요. 정보가 불충분하다면 그 점을 명시하세요."
    },
    "english": {
        "pre": "You are an expert assistant that answers questions about electronic component datasheets. Please provide accurate and detailed responses based on the provided context information.",
        "post": "Please provide a clear and concise answer based on the retrieved information. If the information is insufficient, please specify that."
    },
    "technical": {
        "pre": "당신은 전자공학 전문가입니다. 데이터시트의 기술적 세부사항을 정확히 분석하고 전문적인 답변을 제공하세요.",
        "post": "기술적 정확성을 우선시하여 답변하고, 관련 수치나 사양이 있다면 구체적으로 명시하세요."
    }
}

# API Keys configuration
def get_api_keys() -> Dict[str, Optional[str]]:
    """Get API keys from environment variables or streamlit secrets"""
    api_keys = {
        'openai': os.environ.get('OPENAI_API_KEY'),
        'anthropic': os.environ.get('ANTHROPIC_API_KEY'), 
        'huggingface': os.environ.get('HF_TOKEN')
    }
    
    # Try to load from streamlit secrets if available
    try:
        import streamlit as st
        if hasattr(st, 'secrets'):
            api_keys['openai'] = api_keys['openai'] or st.secrets.get("openai_api_key")
            api_keys['anthropic'] = api_keys['anthropic'] or st.secrets.get("anthropic_api_key")
            api_keys['huggingface'] = api_keys['huggingface'] or st.secrets.get("hf_token")
    except ImportError:
        pass
    
    return api_keys

# Environment info
ENVIRONMENT = detect_environment()

# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "default",
            "stream": "ext://sys.stdout"
        }
    },
    "root": {
        "level": "INFO",
        "handlers": ["console"]
    }
}

# Performance settings
PERFORMANCE_CONFIG = {
    "max_search_results": 50,
    "default_search_results": 5,
    "similarity_threshold": 0.7,
    "max_context_length": 4000,
    "enable_caching": True
}

# Feature flags
FEATURE_FLAGS = {
    "enable_metadata_display": True,
    "enable_source_tracking": True,
    "enable_chat_history": True,
    "enable_prompt_customization": True,
    "enable_multi_llm": True,
    "enable_filtering": True
} 