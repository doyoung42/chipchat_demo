"""
Initialization functions for the ChipChat app
"""
import os
import streamlit as st
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import time

# 로깅 시스템 임포트
try:
    from src.utils.logger import get_logger
    USE_ADVANCED_LOGGING = True
    logger = get_logger()
except ImportError:
    USE_ADVANCED_LOGGING = False
    import logging
    logger = logging.getLogger(__name__)

def setup_paths() -> Dict[str, str]:
    """환경에 따른 경로 설정"""
    # 환경 감지
    try:
        from google.colab import drive
        env = "colab"
    except ImportError:
        env = "local"
    
    if env == "colab":
        # Google Colab 환경
        base_path = Path('/content/drive/MyDrive')
        paths = {
            'vectorstore_path': str(base_path / 'vectorstore'),
            'json_folder_path': str(base_path / 'prep_json'),
            'prompt_templates_path': str(base_path / 'prompt_templates'),
            'chipdb_path': str(base_path / 'prep_json' / 'chipDB.csv')
        }
    else:
        # 로컬 환경
        base_path = Path.cwd()
        paths = {
            'vectorstore_path': str(base_path / 'vectorstore'),
            'json_folder_path': str(base_path / 'prep' / 'prep_json'),
            'prompt_templates_path': str(base_path / 'prompt_templates'),
            'chipdb_path': str(base_path / 'prep' / 'prep_json' / 'chipDB.csv')
        }
    
    # 환경 변수로도 설정
    for key, value in paths.items():
        env_key = key.upper()
        if env_key in os.environ:
            paths[key] = os.environ[env_key]
    
    return paths

def load_api_keys() -> Dict[str, str]:
    """API 키 로드"""
    api_keys = {}
    
    # 환경 변수에서 먼저 시도
    api_keys['openai'] = os.environ.get('OPENAI_API_KEY', '')
    api_keys['anthropic'] = os.environ.get('ANTHROPIC_API_KEY', '')
    api_keys['huggingface'] = os.environ.get('HF_TOKEN', '')
    
    # Streamlit secrets에서 시도
    if hasattr(st, 'secrets'):
        api_keys['openai'] = api_keys['openai'] or st.secrets.get("openai_api_key", "")
        api_keys['anthropic'] = api_keys['anthropic'] or st.secrets.get("anthropic_api_key", "")
        api_keys['huggingface'] = api_keys['huggingface'] or st.secrets.get("hf_token", "")
    
    return api_keys

@st.cache_resource
def initialize_managers(provider: str = "openai", model_name: Optional[str] = None) -> Tuple[Any, Any, Any, Optional[str]]:
    """매니저들 초기화 (캐싱됨)"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # 로깅 데코레이터 사용
    if USE_ADVANCED_LOGGING and hasattr(logger, 'measure_time'):
        measure_time = logger.measure_time
    else:
        def measure_time(name):
            def decorator(func):
                def wrapper(*args, **kwargs):
                    start = time.time()
                    result = func(*args, **kwargs)
                    elapsed = time.time() - start
                    logger.info(f"{name} 완료: {elapsed:.2f}초")
                    return result
                return wrapper
            return decorator
    
    try:
        # ChatManager 초기화
        status_text.info("🔄 1/3 ChatManager 초기화 중...")
        progress_bar.progress(10)
        
        @measure_time("ChatManager 초기화")
        def init_chat_manager():
            from src.models.chat_manager import ChatManager
            return ChatManager(provider=provider, model_name=model_name)
        
        chat_manager = init_chat_manager()
        progress_bar.progress(30)
        
        # VectorstoreManager 초기화
        status_text.info("🔄 2/3 VectorstoreManager 초기화 중... (캐시 확인 중)")
        
        @measure_time("VectorstoreManager 초기화")
        def init_vectorstore_manager():
            from src.models.vectorstore_manager import VectorstoreManager
            return VectorstoreManager()
        
        vectorstore_manager = init_vectorstore_manager()
        progress_bar.progress(60)
        
        # Vectorstore 로드
        status_text.info("🔄 3/3 Vectorstore 로드 중...")
        
        @measure_time("Vectorstore 로드")
        def load_vs():
            paths = st.session_state.get('paths', setup_paths())
            vectorstore_path = paths['vectorstore_path']
            
            if Path(vectorstore_path).exists():
                return vectorstore_manager.load_vectorstore(vectorstore_path)
            else:
                # JSON 파일에서 생성 시도
                json_folder = paths['json_folder_path']
                if Path(json_folder).exists():
                    json_data = vectorstore_manager.load_json_files(json_folder)
                    if json_data:
                        vectorstore = vectorstore_manager.create_vectorstore(json_data)
                        Path(vectorstore_path).parent.mkdir(parents=True, exist_ok=True)
                        vectorstore_manager.save_vectorstore(vectorstore, vectorstore_path)
                        return vectorstore
                
                raise ValueError("벡터스토어나 JSON 파일을 찾을 수 없습니다.")
        
        vectorstore = load_vs()
        progress_bar.progress(100)
        
        status_text.success("✅ 모든 구성 요소 초기화 완료!")
        time.sleep(1)
        
        # UI 정리
        progress_bar.empty()
        status_text.empty()
        
        return chat_manager, vectorstore_manager, vectorstore, None
        
    except Exception as e:
        logger.error(f"매니저 초기화 실패: {str(e)}", extra={"error": str(e)})
        progress_bar.empty()
        status_text.error(f"❌ 초기화 실패: {str(e)}")
        return None, None, None, str(e)

@st.cache_resource
def initialize_agent(_chat_manager, _vectorstore_manager, _vectorstore, chipdb_path: str) -> Tuple[Any, Optional[str]]:
    """LangGraph 에이전트 초기화"""
    try:
        if not Path(chipdb_path).exists():
            return None, f"chipDB.csv not found at {chipdb_path}"
        
        from src.models.langgraph_agent import ChipChatAgent
        agent = ChipChatAgent(
            csv_path=chipdb_path,
            vectorstore_manager=_vectorstore_manager,
            vectorstore=_vectorstore,
            llm_manager=_chat_manager.llm_manager
        )
        
        logger.info("LangGraph 에이전트 초기화 완료")
        return agent, None
        
    except Exception as e:
        logger.error(f"에이전트 초기화 실패: {str(e)}")
        return None, str(e) 