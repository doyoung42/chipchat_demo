import os
import json
from pathlib import Path
from typing import Dict, Any, Optional

class ConfigManager:
    """Main 프로젝트의 설정을 관리하는 클래스"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Args:
            config_path: config.json 파일의 경로 (None이면 프로젝트 루트에서 찾음)
        """
        if config_path is None:
            # 프로젝트 루트 디렉토리에서 config.json 찾기
            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent  # src/utils에서 프로젝트 루트로
            self.config_path = project_root / 'config.json'
        else:
            self.config_path = Path(config_path)
        
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """config.json 파일을 로드"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file: {e}")
    
    def save_config(self):
        """현재 설정을 config.json에 저장"""
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=4, ensure_ascii=False)
    
    def get_environment(self) -> str:
        """현재 환경 반환 ('google_drive' 또는 'local')"""
        use_google_drive = self.config.get('environment', {}).get('use_google_drive', False)
        return 'google_drive' if use_google_drive else 'local'
    
    def set_environment(self, use_google_drive: bool):
        """환경 설정 변경"""
        if 'environment' not in self.config:
            self.config['environment'] = {}
        self.config['environment']['use_google_drive'] = use_google_drive
        self.save_config()
    
    def get_paths(self) -> Dict[str, str]:
        """현재 환경에 맞는 경로들 반환"""
        env = self.get_environment()
        return self.config.get('paths', {}).get(env, {})
    
    def get_path(self, path_key: str) -> str:
        """특정 경로 반환"""
        paths = self.get_paths()
        return paths.get(path_key, f"./{path_key}")
    
    def get_model_config(self) -> Dict[str, Any]:
        """모델 설정 반환"""
        return self.config.get('models', {})
    
    def get_embedding_model(self) -> str:
        """임베딩 모델명 반환"""
        return self.get_model_config().get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
    
    def get_supported_models(self) -> Dict[str, list]:
        """지원되는 LLM 모델들 반환"""
        return self.get_model_config().get('supported_llm', {})
    
    def create_directories(self):
        """필요한 디렉토리들을 생성"""
        paths = self.get_paths()
        for key, path_str in paths.items():
            if key != 'base_path':  # base_path는 제외
                path = Path(path_str)
                path.mkdir(parents=True, exist_ok=True)
    
    def __getitem__(self, key):
        """딕셔너리 스타일 접근 지원"""
        return self.config[key]
    
    def __setitem__(self, key, value):
        """딕셔너리 스타일 설정 지원"""
        self.config[key] = value


# 싱글톤 인스턴스
_config_manager_instance = None

def get_config_manager() -> ConfigManager:
    """싱글톤 ConfigManager 인스턴스 반환"""
    global _config_manager_instance
    
    if _config_manager_instance is None:
        _config_manager_instance = ConfigManager()
    
    return _config_manager_instance 