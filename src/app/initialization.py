"""
Streamlit 앱 초기화 모듈
"""

import os
import streamlit as st
from pathlib import Path
from typing import Dict, Any

def initialize_app_config():
    """앱 설정 초기화"""
    try:
        from ..utils.config_manager import get_config_manager
        config = get_config_manager()
        
        # 현재 환경 가져오기
        use_google_drive = config.get_environment() == 'google_drive'
        paths = config.get_paths()
        
        # Streamlit 환경변수 설정
        os.environ['USE_GOOGLE_DRIVE'] = str(use_google_drive)
        os.environ['VECTORSTORE_PATH'] = paths.get('vectorstore_folder', './vectorstore')
        os.environ['JSON_FOLDER_PATH'] = paths.get('prep_json_folder', './prep/prep_json')
        os.environ['PROMPT_TEMPLATES_PATH'] = paths.get('prompt_templates_folder', './prompt_templates')
        os.environ['MODEL_CACHE_DIR'] = paths.get('model_cache_folder', './hf_model_cache')
        
        # 필요한 디렉토리 생성
        config.create_directories()
        
        return {
            'environment': config.get_environment(),
            'paths': paths,
            'embedding_model': config.get_embedding_model(),
            'supported_models': config.get_supported_models()
        }
        
    except Exception as e:
        st.error(f"설정 초기화 실패: {e}")
        # 폴백 설정
        return initialize_fallback_config()

def initialize_fallback_config():
    """config.json 읽기 실패 시 폴백 설정"""
    # 환경 감지
    try:
        from google.colab import drive
        # Google Colab 환경
        use_google_drive = True
        base_path = Path('/content/drive/MyDrive')
        paths = {
            'base_path': str(base_path),
            'prep_json_folder': str(base_path / 'prep_json'),
            'vectorstore_folder': str(base_path / 'vectorstore'),
            'prompt_templates_folder': str(base_path / 'prompt_templates'),
            'model_cache_folder': str(base_path / 'hf_model_cache'),
            'logs_folder': str(base_path / 'chipchat_logs')
        }
    except ImportError:
        # 로컬 환경
        use_google_drive = False
        base_path = Path('.')
        paths = {
            'base_path': str(base_path),
            'prep_json_folder': str(base_path / 'prep_json'),
            'vectorstore_folder': str(base_path / 'vectorstore'),
            'prompt_templates_folder': str(base_path / 'prompt_templates'),
            'model_cache_folder': str(base_path / 'hf_model_cache'),
            'logs_folder': str(base_path / 'logs')
        }
    
    # 환경변수 설정
    os.environ['USE_GOOGLE_DRIVE'] = str(use_google_drive)
    os.environ['VECTORSTORE_PATH'] = paths['vectorstore_folder']
    os.environ['JSON_FOLDER_PATH'] = paths['prep_json_folder']
    os.environ['PROMPT_TEMPLATES_PATH'] = paths['prompt_templates_folder']
    os.environ['MODEL_CACHE_DIR'] = paths['model_cache_folder']
    
    # 필요한 디렉토리 생성
    for key, path_str in paths.items():
        if key != 'base_path':
            Path(path_str).mkdir(parents=True, exist_ok=True)
    
    return {
        'environment': 'google_drive' if use_google_drive else 'local',
        'paths': paths,
        'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
        'supported_models': {
            'openai': ['gpt-4o-mini', 'gpt-4o', 'gpt-3.5-turbo'],
            'claude': ['claude-3-sonnet', 'claude-3-haiku', 'claude-3-opus']
        }
    }


def setup_paths():
    """경로 설정을 초기화합니다."""
    try:
        from ..utils.config_manager import get_config_manager
        config = get_config_manager()
        paths = config.get_paths()
        
        # ChipDB 경로 추가
        paths['chipdb_path'] = str(Path(paths.get('prep_json_folder', './prep/prep_json')) / 'chipDB.csv')
        
        return paths
    except Exception as e:
        st.error(f"경로 설정 실패: {e}")
        # 폴백 경로
        return {
            'vectorstore_folder': './prep/vectorstore',
            'prep_json_folder': './prep/prep_json',
            'chipdb_path': './prep/prep_json/chipDB.csv'
        }


def load_api_keys():
    """API 키를 로드합니다."""
    try:
        from ..config.settings import get_api_keys
        return get_api_keys()
    except Exception as e:
        st.warning(f"API 키 로드 실패: {e}")
        return {
            'openai': os.environ.get('OPENAI_API_KEY'),
            'anthropic': os.environ.get('ANTHROPIC_API_KEY'),
            'huggingface': os.environ.get('HF_TOKEN')
        }


def initialize_managers(provider: str, model_name: str):
    """매니저들을 초기화합니다."""
    try:
        from ..models.chat_manager import ChatManager
        from ..models.vectorstore_manager import VectorstoreManager
        
        # ChatManager 초기화
        chat_manager = ChatManager(provider=provider, model_name=model_name)
        
        # VectorstoreManager 초기화
        vectorstore_manager = VectorstoreManager()
        
        # Vectorstore 로드
        paths = setup_paths()
        vectorstore_path = Path(paths['vectorstore_folder'])
        
        if vectorstore_path.exists() and any(vectorstore_path.iterdir()):
            # 설정에서 vectorstore 이름 가져오기
            from ..utils.config_manager import ConfigManager
            config = ConfigManager()
            vectorstore_name = config.get_vectorstore_name()
            vectorstore = vectorstore_manager.load_vectorstore(vectorstore_name)
        else:
            st.warning("벡터스토어가 없습니다. prep_json 폴더에서 생성하겠습니다.")
            json_folder = paths['prep_json_folder']
            if Path(json_folder).exists():
                vectorstore = vectorstore_manager.create_vectorstore_from_json(
                    json_folder, str(vectorstore_path)
                )
            else:
                vectorstore = None
        
        return chat_manager, vectorstore_manager, vectorstore, None
        
    except Exception as e:
        return None, None, None, str(e)


def initialize_agent(chat_manager, vectorstore_manager, vectorstore, chipdb_path: str):
    """에이전트를 초기화합니다."""
    try:
        from ..models.langgraph_agent import ChipChatAgent
        
        if not Path(chipdb_path).exists():
            return None, f"ChipDB 파일을 찾을 수 없습니다: {chipdb_path}"
        
        agent = ChipChatAgent(
            csv_path=chipdb_path,
            vectorstore_manager=vectorstore_manager,
            vectorstore=vectorstore,
            llm_manager=chat_manager.llm_manager
        )
        
        return agent, None
        
    except Exception as e:
        return None, str(e) 