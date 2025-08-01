"""
최적화된 로딩 모듈
Streamlit 캐싱을 활용하여 초기 로딩 시간을 단축합니다.
"""

import streamlit as st
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple


logger = logging.getLogger(__name__)


@st.cache_resource
def get_cached_vectorstore_manager():
    """VectorstoreManager를 캐시된 형태로 반환합니다."""
    try:
        from ..models.vectorstore_manager import VectorstoreManager
        logger.info("VectorstoreManager 캐시 생성")
        return VectorstoreManager()
    except Exception as e:
        logger.error(f"VectorstoreManager 캐시 생성 실패: {e}")
        return None


@st.cache_resource
def get_cached_chat_manager(provider: str, model_name: str):
    """ChatManager를 캐시된 형태로 반환합니다."""
    try:
        from ..models.chat_manager import ChatManager
        logger.info(f"ChatManager 캐시 생성: {provider}/{model_name}")
        return ChatManager(provider=provider, model_name=model_name)
    except Exception as e:
        logger.error(f"ChatManager 캐시 생성 실패: {e}")
        return None


@st.cache_data
def load_chipdb_cached(csv_path: str) -> pd.DataFrame:
    """ChipDB CSV 파일을 캐시된 형태로 로드합니다."""
    try:
        if not Path(csv_path).exists():
            logger.warning(f"ChipDB 파일이 없습니다: {csv_path}")
            return pd.DataFrame()
        
        df = pd.read_csv(csv_path)
        logger.info(f"ChipDB 캐시 로드 완료: {len(df)} 항목")
        return df
    except Exception as e:
        logger.error(f"ChipDB 캐시 로드 실패: {e}")
        return pd.DataFrame()


@st.cache_resource
def get_cached_vectorstore(vectorstore_path: str, vectorstore_manager=None):
    """Vectorstore를 캐시된 형태로 로드합니다."""
    try:
        if vectorstore_manager is None:
            vectorstore_manager = get_cached_vectorstore_manager()
        
        if not vectorstore_manager:
            return None
            
        vectorstore_path_obj = Path(vectorstore_path)
        
        if vectorstore_path_obj.exists() and any(vectorstore_path_obj.iterdir()):
            vectorstore = vectorstore_manager.load_vectorstore(str(vectorstore_path_obj))
            logger.info(f"Vectorstore 캐시 로드 완료: {vectorstore_path}")
            return vectorstore
        else:
            logger.warning(f"Vectorstore가 없습니다: {vectorstore_path}")
            return None
            
    except Exception as e:
        logger.error(f"Vectorstore 캐시 로드 실패: {e}")
        return None


@st.cache_resource
def get_cached_langgraph_agent(csv_path: str, vectorstore_manager, vectorstore, chat_manager):
    """LangGraph Agent를 캐시된 형태로 생성합니다."""
    try:
        from ..models.langgraph_agent import ChipChatAgent
        
        if not Path(csv_path).exists():
            logger.error(f"ChipDB 파일이 없습니다: {csv_path}")
            return None
        
        agent = ChipChatAgent(
            csv_path=csv_path,
            vectorstore_manager=vectorstore_manager,
            vectorstore=vectorstore,
            llm_manager=chat_manager.llm_manager
        )
        
        logger.info("LangGraph Agent 캐시 생성 완료")
        return agent
        
    except Exception as e:
        logger.error(f"LangGraph Agent 캐시 생성 실패: {e}")
        return None


@st.cache_data
def get_cached_paths():
    """경로 설정을 캐시된 형태로 반환합니다."""
    try:
        from ..utils.config_manager import get_config_manager
        config = get_config_manager()
        paths = config.get_paths()
        
        # ChipDB 경로 추가
        paths['chipdb_path'] = str(Path(paths.get('prep_json_folder', './prep_json')) / 'chipDB.csv')
        
        logger.info("경로 설정 캐시 생성 완료")
        return paths
    except Exception as e:
        logger.error(f"경로 설정 캐시 생성 실패: {e}")
        # 폴백 경로
        return {
            'vectorstore_folder': './vectorstore',
            'prep_json_folder': './prep_json',
            'chipdb_path': './prep_json/chipDB.csv'
        }


@st.cache_data
def get_cached_api_keys():
    """API 키를 캐시된 형태로 반환합니다."""
    try:
        from ..config.settings import get_api_keys
        api_keys = get_api_keys()
        logger.info("API 키 캐시 생성 완료")
        return api_keys
    except Exception as e:
        logger.warning(f"API 키 캐시 생성 실패: {e}")
        import os
        return {
            'openai': os.environ.get('OPENAI_API_KEY'),
            'anthropic': os.environ.get('ANTHROPIC_API_KEY'),
            'huggingface': os.environ.get('HF_TOKEN')
        }


def initialize_optimized_managers(provider: str, model_name: str) -> Tuple[Any, Any, Any, Optional[str]]:
    """최적화된 매니저 초기화 함수"""
    try:
        # 캐시된 매니저들 로드
        chat_manager = get_cached_chat_manager(provider, model_name)
        vectorstore_manager = get_cached_vectorstore_manager()
        
        if not chat_manager:
            return None, None, None, "ChatManager 초기화 실패"
        
        if not vectorstore_manager:
            return None, None, None, "VectorstoreManager 초기화 실패"
        
        # 경로 설정
        paths = get_cached_paths()
        vectorstore_path = paths['vectorstore_folder']
        
        # 캐시된 Vectorstore 로드
        vectorstore = get_cached_vectorstore(vectorstore_path, vectorstore_manager)
        
        # Vectorstore가 없는 경우 생성 시도
        if vectorstore is None:
            st.warning("🔄 벡터스토어가 없습니다. prep_json 폴더에서 생성하겠습니다...")
            json_folder = paths['prep_json_folder']
            if Path(json_folder).exists():
                try:
                    # 캐시를 사용하지 않는 일회성 생성
                    vectorstore = vectorstore_manager.create_vectorstore_from_json(
                        json_folder, str(vectorstore_path)
                    )
                    # 생성 후 캐시 무효화하여 다음에는 캐시된 버전 사용
                    get_cached_vectorstore.clear()
                    vectorstore = get_cached_vectorstore(vectorstore_path, vectorstore_manager)
                except Exception as e:
                    logger.error(f"Vectorstore 생성 실패: {e}")
                    vectorstore = None
        
        return chat_manager, vectorstore_manager, vectorstore, None
        
    except Exception as e:
        return None, None, None, str(e)


def initialize_optimized_agent(chat_manager, vectorstore_manager, vectorstore, chipdb_path: str) -> Tuple[Any, Optional[str]]:
    """최적화된 에이전트 초기화 함수"""
    try:
        # 캐시된 Agent 로드
        agent = get_cached_langgraph_agent(chipdb_path, vectorstore_manager, vectorstore, chat_manager)
        
        if not agent:
            return None, f"Agent 초기화 실패: ChipDB 파일 확인 필요 ({chipdb_path})"
        
        return agent, None
        
    except Exception as e:
        return None, str(e)


def clear_all_caches():
    """모든 캐시를 지웁니다."""
    try:
        get_cached_chat_manager.clear()
        get_cached_vectorstore_manager.clear()
        load_chipdb_cached.clear()
        get_cached_vectorstore.clear()
        get_cached_langgraph_agent.clear()
        get_cached_paths.clear()
        get_cached_api_keys.clear()
        
        logger.info("모든 캐시가 지워졌습니다")
        return True
    except Exception as e:
        logger.error(f"캐시 지우기 실패: {e}")
        return False


def clear_model_caches(provider: str, model_name: str):
    """특정 모델과 관련된 캐시만 지웁니다."""
    try:
        # ChatManager 캐시 지우기 (모델별로 캐시됨)
        get_cached_chat_manager.clear()
        
        # Agent도 ChatManager에 의존하므로 지우기
        get_cached_langgraph_agent.clear()
        
        logger.info(f"모델 캐시 지움: {provider}/{model_name}")
        return True
    except Exception as e:
        logger.error(f"모델 캐시 지우기 실패: {e}")
        return False


# 캐시 상태 정보 제공
def get_cache_info() -> Dict[str, Any]:
    """현재 캐시 상태 정보를 반환합니다."""
    cache_info = {
        "cached_functions": [
            "get_cached_chat_manager",
            "get_cached_vectorstore_manager", 
            "load_chipdb_cached",
            "get_cached_vectorstore",
            "get_cached_langgraph_agent",
            "get_cached_paths",
            "get_cached_api_keys"
        ],
        "description": "Streamlit @st.cache_resource 및 @st.cache_data를 사용한 캐싱 시스템"
    }
    
    return cache_info 