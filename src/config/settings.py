"""
Configuration settings for the ChipChat application.
Supports both local and Google Colab environments.
"""
import os
import json
from pathlib import Path
from typing import Optional, Dict

def get_project_settings():
    """프로젝트 설정을 config.json에서 읽어오는 함수"""
    try:
        from ..utils.config_manager import get_config_manager
        config = get_config_manager()
        
        # 현재 환경의 경로들 가져오기
        paths = config.get_paths()
        
        return {
            'BASE_DIR': Path(paths.get('base_path', '.')),
            'PREP_JSON_FOLDER': Path(paths.get('prep_json_folder', './prep/prep_json')),
            'VECTORSTORE_FOLDER': Path(paths.get('vectorstore_folder', './prep/vectorstore')), 
            'PROMPT_TEMPLATES_FOLDER': Path(paths.get('prompt_templates_folder', './prompt_templates')),
            'MODEL_CACHE_FOLDER': Path(paths.get('model_cache_folder', './hf_model_cache')),
            'LOGS_FOLDER': Path(paths.get('logs_folder', './logs')),
            'EMBEDDING_MODEL': config.get_embedding_model(),
            'SUPPORTED_MODELS': config.get_supported_models(),
            'USE_GOOGLE_DRIVE': config.get_environment() == 'google_drive'
        }
    except Exception as e:
        # config.json 읽기 실패 시 기본값 반환
        print(f"Warning: Failed to read config.json: {e}")
        return get_default_settings()

def get_default_settings():
    """기본 설정값 반환 (config.json 읽기 실패 시 사용)"""
    # 환경 감지
    try:
        from google.colab import drive
        # Google Colab 환경
        base_dir = Path('/content/drive/MyDrive')
        use_google_drive = True
    except ImportError:
        # 로컬 환경
        base_dir = Path('.')
        use_google_drive = False
    
    return {
        'BASE_DIR': base_dir,
        'PREP_JSON_FOLDER': base_dir / 'prep' / 'prep_json',
        'VECTORSTORE_FOLDER': base_dir / 'prep' / 'vectorstore',
        'PROMPT_TEMPLATES_FOLDER': base_dir / 'prompts',
        'MODEL_CACHE_FOLDER': base_dir / 'hf_model_cache',
        'LOGS_FOLDER': base_dir / ('chipchat_logs' if use_google_drive else 'logs'),
        'EMBEDDING_MODEL': 'sentence-transformers/all-MiniLM-L6-v2',
        'SUPPORTED_MODELS': {
            'openai': ['gpt-4o-mini', 'gpt-4o', 'gpt-3.5-turbo'],
            'claude': ['claude-3-sonnet', 'claude-3-haiku', 'claude-3-opus']
        },
        'USE_GOOGLE_DRIVE': use_google_drive
    }

# 프로젝트 설정 로드
SETTINGS = get_project_settings()

# 개별 설정값들을 모듈 레벨에서 접근 가능하도록 export
BASE_DIR = SETTINGS['BASE_DIR']
PREP_JSON_FOLDER = SETTINGS['PREP_JSON_FOLDER']
VECTORSTORE_FOLDER = SETTINGS['VECTORSTORE_FOLDER']
PROMPT_TEMPLATES_FOLDER = SETTINGS['PROMPT_TEMPLATES_FOLDER']
MODEL_CACHE_FOLDER = SETTINGS['MODEL_CACHE_FOLDER']
LOGS_FOLDER = SETTINGS['LOGS_FOLDER']
EMBEDDING_MODEL = SETTINGS['EMBEDDING_MODEL']
SUPPORTED_MODELS = SETTINGS['SUPPORTED_MODELS']
USE_GOOGLE_DRIVE = SETTINGS['USE_GOOGLE_DRIVE']

# PATHS 딕셔너리 (기존 코드 호환성을 위해)
PATHS = {
    'base': BASE_DIR,
    'prep_json': PREP_JSON_FOLDER,
    'vectorstore': VECTORSTORE_FOLDER,
    'prompt_templates': PROMPT_TEMPLATES_FOLDER,
    'model_cache': MODEL_CACHE_FOLDER,
    'logs': LOGS_FOLDER
}

# Model settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# LLM Configuration
LLM_CONFIG = {
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
            "claude-3-sonnet": {
                "model_name": "claude-3-sonnet",
                "temperature": 0.1,
                "max_tokens": 2000,
                "description": "Balanced performance and speed"
            },
            "claude-3-haiku": {
                "model_name": "claude-3-haiku",
                "temperature": 0.1,
                "max_tokens": 2000,
                "description": "Fastest Claude model"
            },
            "claude-3-opus": {
                "model_name": "claude-3-opus",
                "temperature": 0.1,
                "max_tokens": 2000,
                "description": "Most capable Claude model"
            }
        },
        "default_model": "claude-3-sonnet",
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
    "default": {
        "pre": "You are a professional assistant specializing in electronic component datasheets. Provide accurate and detailed responses based on the provided context information. Always maintain technical accuracy and cite specific information from the datasheets when available.",
        "post": "Provide a clear and concise answer based on the retrieved information. If the information is insufficient or unclear, explicitly state what information is missing and suggest what additional details might be needed."
    },
    "english": {
        "pre": "You are an expert assistant that answers questions about electronic component datasheets. Please provide accurate and detailed responses based on the provided context information.",
        "post": "Please provide a clear and concise answer based on the retrieved information. If the information is insufficient, please specify that."
    },
    "technical": {
        "pre": "You are an electronics engineering expert specializing in component analysis. Analyze datasheet technical specifications with precision and provide comprehensive technical responses. Focus on electrical characteristics, performance parameters, and implementation considerations.",
        "post": "Prioritize technical accuracy in your response. Include specific numerical values, specifications, and technical details when available. If presenting electrical characteristics, include units and operating conditions."
    }
}

# API Keys configuration
def get_api_keys() -> Dict[str, Optional[str]]:
    """Get API keys from all available sources: environment variables, streamlit secrets, and tokens.json"""
    api_keys = {
        'openai': None,
        'anthropic': None, 
        'huggingface': None
    }
    
    # 1. Try environment variables first
    api_keys['openai'] = os.environ.get('OPENAI_API_KEY')
    api_keys['anthropic'] = os.environ.get('ANTHROPIC_API_KEY')
    api_keys['huggingface'] = os.environ.get('HF_TOKEN')
    
    # 2. Try to load from streamlit secrets if available
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and st.secrets:
            api_keys['openai'] = api_keys['openai'] or st.secrets.get("openai_api_key")
            api_keys['anthropic'] = api_keys['anthropic'] or st.secrets.get("anthropic_api_key")
            api_keys['huggingface'] = api_keys['huggingface'] or st.secrets.get("hf_token")
    except ImportError:
        pass
    except Exception as e:
        print(f"Warning: Failed to load streamlit secrets: {e}")
    
    # 3. Try to load from TokenManager (tokens.json) as fallback
    try:
        from .token_manager import TokenManager
        token_manager = TokenManager()
        
        api_keys['openai'] = api_keys['openai'] or token_manager.get_token('openai')
        api_keys['anthropic'] = api_keys['anthropic'] or token_manager.get_token('anthropic')
        api_keys['huggingface'] = api_keys['huggingface'] or token_manager.get_token('huggingface')
        
        print("✅ TokenManager에서 API 키 로드 시도 완료")
        
    except Exception as e:
        print(f"Warning: Failed to load from TokenManager: {e}")
    
    # Log API key status (without revealing the keys)
    status = {
        'openai': '✅' if api_keys['openai'] else '❌',
        'anthropic': '✅' if api_keys['anthropic'] else '❌', 
        'huggingface': '✅' if api_keys['huggingface'] else '❌'
    }
    print(f"🔑 API 키 상태: OpenAI {status['openai']}, Claude {status['anthropic']}, HuggingFace {status['huggingface']}")
    
    return api_keys

# Environment info
ENVIRONMENT = SETTINGS['USE_GOOGLE_DRIVE'] and 'colab' or 'local'

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