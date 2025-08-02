"""
단순화된 초기화 모듈
복잡한 캐싱 없이 직접적이고 명확한 초기화를 제공합니다.
"""

import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, Any

logger = logging.getLogger(__name__)


class SimpleInitializer:
    """단순하고 명확한 초기화를 위한 클래스"""
    
    def __init__(self):
        self.initialized_components = {}
    
    def check_prerequisites(self, paths: Dict[str, str]) -> Tuple[bool, list]:
        """필수 파일들이 존재하는지 확인"""
        missing_files = []
        
        # ChipDB 확인
        chipdb_path = Path(paths.get('chipdb_path', './prep/prep_json/chipDB.csv'))
        if not chipdb_path.exists():
            missing_files.append(f"ChipDB 파일: {chipdb_path}")
        
        # Vectorstore 확인
        vectorstore_path = Path(paths.get('vectorstore_folder', './prep/vectorstore'))
        if not vectorstore_path.exists() or not any(vectorstore_path.iterdir()):
            missing_files.append(f"Vectorstore: {vectorstore_path}")
        
        return len(missing_files) == 0, missing_files
    
    def check_api_keys(self, api_keys: Dict[str, str]) -> bool:
        """API 키가 설정되어 있는지 확인"""
        return any(api_keys.values())
    
    def initialize_chat_manager(self, provider: str, model_name: str):
        """ChatManager 초기화"""
        try:
            from ..models.chat_manager import ChatManager
            
            logger.info(f"ChatManager 초기화 시작: {provider}/{model_name}")
            chat_manager = ChatManager(provider=provider, model_name=model_name)
            
            # 연결 테스트
            if not chat_manager.test_llm_connection():
                raise Exception(f"LLM 연결 테스트 실패: {provider}/{model_name}")
            
            logger.info(f"ChatManager 초기화 완료: {provider}/{model_name}")
            self.initialized_components['chat_manager'] = chat_manager
            return chat_manager, None
            
        except Exception as e:
            error_msg = f"ChatManager 초기화 실패: {str(e)}"
            logger.error(error_msg)
            return None, error_msg
    
    def initialize_vectorstore_manager(self):
        """VectorstoreManager 초기화"""
        try:
            from ..models.vectorstore_manager import VectorstoreManager
            
            logger.info("VectorstoreManager 초기화 시작")
            vectorstore_manager = VectorstoreManager()
            
            logger.info("VectorstoreManager 초기화 완료")
            self.initialized_components['vectorstore_manager'] = vectorstore_manager
            return vectorstore_manager, None
            
        except Exception as e:
            error_msg = f"VectorstoreManager 초기화 실패: {str(e)}"
            logger.error(error_msg)
            return None, error_msg
    
    def load_vectorstore(self, vectorstore_manager, vectorstore_path: str):
        """Vectorstore 로드"""
        try:
            # 설정에서 vectorstore 이름 가져오기
            from ..utils.config_manager import ConfigManager
            config = ConfigManager()
            vectorstore_name = config.get_vectorstore_name()
            logger.info(f"Vectorstore 로드 시작: {vectorstore_path} (name: {vectorstore_name})")
            vectorstore = vectorstore_manager.load_vectorstore(vectorstore_name)
            
            if not vectorstore:
                raise Exception("Vectorstore 로드 결과가 None입니다")
            
            logger.info(f"Vectorstore 로드 완료: {vectorstore_path}")
            self.initialized_components['vectorstore'] = vectorstore
            return vectorstore, None
            
        except Exception as e:
            error_msg = f"Vectorstore 로드 실패: {str(e)}"
            logger.error(error_msg)
            return None, error_msg
    
    def initialize_agent(self, chat_manager, vectorstore_manager, vectorstore, chipdb_path: str):
        """LangGraph Agent 초기화"""
        try:
            from ..models.langgraph_agent import ChipChatAgent
            
            logger.info(f"Agent 초기화 시작: {chipdb_path}")
            
            if not Path(chipdb_path).exists():
                raise Exception(f"ChipDB 파일이 없습니다: {chipdb_path}")
            
            agent = ChipChatAgent(
                csv_path=chipdb_path,
                vectorstore_manager=vectorstore_manager,
                vectorstore=vectorstore,
                llm_manager=chat_manager.llm_manager
            )
            
            logger.info("Agent 초기화 완료")
            self.initialized_components['agent'] = agent
            return agent, None
            
        except Exception as e:
            error_msg = f"Agent 초기화 실패: {str(e)}"
            logger.error(error_msg)
            return None, error_msg
    
    def initialize_all(self, provider: str, model_name: str, paths: Dict[str, str], api_keys: Dict[str, str]) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        """모든 컴포넌트를 순차적으로 초기화"""
        
        # 1. 필수 조건 확인
        prerequisites_ok, missing_files = self.check_prerequisites(paths)
        if not prerequisites_ok:
            return False, {}, f"필수 파일 누락: {', '.join(missing_files)}"
        
        if not self.check_api_keys(api_keys):
            return False, {}, "API 키가 설정되지 않았습니다"
        
        # 2. ChatManager 초기화
        chat_manager, error = self.initialize_chat_manager(provider, model_name)
        if error:
            return False, {}, error
        
        # 3. VectorstoreManager 초기화
        vectorstore_manager, error = self.initialize_vectorstore_manager()
        if error:
            return False, {}, error
        
        # 4. Vectorstore 로드
        vectorstore, error = self.load_vectorstore(vectorstore_manager, paths['vectorstore_folder'])
        if error:
            return False, {}, error
        
        # 5. Agent 초기화
        agent, error = self.initialize_agent(chat_manager, vectorstore_manager, vectorstore, paths['chipdb_path'])
        if error:
            return False, {}, error
        
        # 모든 컴포넌트 반환
        components = {
            'chat_manager': chat_manager,
            'vectorstore_manager': vectorstore_manager,
            'vectorstore': vectorstore,
            'agent': agent
        }
        
        logger.info("모든 컴포넌트 초기화 완료")
        return True, components, None
    
    def get_status_summary(self) -> Dict[str, str]:
        """초기화된 컴포넌트들의 상태 요약"""
        status = {}
        for name, component in self.initialized_components.items():
            status[name] = "초기화됨" if component else "초기화 실패"
        return status


def create_simple_initializer() -> SimpleInitializer:
    """SimpleInitializer 인스턴스 생성"""
    return SimpleInitializer() 