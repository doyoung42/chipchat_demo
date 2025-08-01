"""
사용자 설정 관리 모듈
LLM 모델 선택 등의 사용자 설정을 JSON 파일로 저장하고 관리합니다.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional


class UserSettingsManager:
    """사용자 설정 관리자"""
    
    def __init__(self, settings_file: str = "user_settings.json"):
        """
        Args:
            settings_file: 설정 파일 경로
        """
        self.settings_file = Path(settings_file)
        self.logger = logging.getLogger(__name__)
        
        # 기본 설정값
        self.default_settings = {
            "llm": {
                "provider": "claude",
                "model_name": "claude-3-sonnet-20240229",
                "description": "Default LLM settings"
            },
            "ui": {
                "show_agent_info": False,
                "show_performance_metrics": True,
                "prompt_template": "default"
            },
            "advanced": {
                "temperature": 0.7,
                "max_tokens": 2000,
                "k_documents": 5
            },
            "version": "1.0",
            "created_at": None,
            "updated_at": None
        }
        
        # 지원되는 모델 목록
        self.supported_models = {
            "openai": [
                {"name": "gpt-4o-mini", "display": "GPT-4o Mini (빠르고 경제적)"},
                {"name": "gpt-4o", "display": "GPT-4o (최신 고성능)"},
                {"name": "gpt-3.5-turbo", "display": "GPT-3.5 Turbo (안정적)"}
            ],
            "claude": [
                {"name": "claude-3-sonnet-20240229", "display": "Claude 3 Sonnet (추천)"},
                {"name": "claude-3-haiku-20240307", "display": "Claude 3 Haiku (빠름)"},
                {"name": "claude-3-opus-20240229", "display": "Claude 3 Opus (최고 성능)"}
            ]
        }
    
    def settings_exist(self) -> bool:
        """설정 파일이 존재하는지 확인"""
        return self.settings_file.exists()
    
    def load_settings(self) -> Dict[str, Any]:
        """설정을 로드합니다. 파일이 없으면 기본 설정을 반환합니다."""
        try:
            if self.settings_file.exists():
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                
                # 기본 설정과 병합 (새로운 키가 추가된 경우 대비)
                merged_settings = self._merge_settings(self.default_settings.copy(), settings)
                self.logger.info(f"Settings loaded from {self.settings_file}")
                return merged_settings
            else:
                self.logger.info("Settings file not found, using defaults")
                return self.default_settings.copy()
                
        except Exception as e:
            self.logger.error(f"Failed to load settings: {e}")
            return self.default_settings.copy()
    
    def save_settings(self, settings: Dict[str, Any]) -> bool:
        """설정을 파일에 저장합니다."""
        try:
            # 타임스탬프 업데이트
            import datetime
            settings["updated_at"] = datetime.datetime.now().isoformat()
            if not settings.get("created_at"):
                settings["created_at"] = settings["updated_at"]
            
            # 설정 검증
            validated_settings = self._validate_settings(settings)
            
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(validated_settings, f, indent=4, ensure_ascii=False)
            
            self.logger.info(f"Settings saved to {self.settings_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save settings: {e}")
            return False
    
    def _merge_settings(self, default: Dict, user: Dict) -> Dict:
        """기본 설정과 사용자 설정을 병합합니다."""
        for key, value in user.items():
            if key in default:
                if isinstance(value, dict) and isinstance(default[key], dict):
                    default[key] = self._merge_settings(default[key], value)
                else:
                    default[key] = value
            else:
                default[key] = value
        return default
    
    def _validate_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """설정값을 검증하고 수정합니다."""
        # LLM 설정 검증
        llm_settings = settings.get("llm", {})
        provider = llm_settings.get("provider", "claude")
        model_name = llm_settings.get("model_name", "")
        
        # 지원되는 모델인지 확인
        if provider in self.supported_models:
            supported_model_names = [m["name"] for m in self.supported_models[provider]]
            if model_name not in supported_model_names:
                # 기본 모델로 설정
                settings["llm"]["model_name"] = self.supported_models[provider][0]["name"]
                self.logger.warning(f"Invalid model {model_name}, using default")
        else:
            # 기본 provider로 설정
            settings["llm"]["provider"] = "claude"
            settings["llm"]["model_name"] = "claude-3-sonnet-20240229"
            self.logger.warning(f"Invalid provider {provider}, using default")
        
        # 고급 설정 검증
        advanced = settings.get("advanced", {})
        if "temperature" in advanced:
            advanced["temperature"] = max(0.0, min(2.0, float(advanced["temperature"])))
        if "max_tokens" in advanced:
            advanced["max_tokens"] = max(100, min(4000, int(advanced["max_tokens"])))
        if "k_documents" in advanced:
            advanced["k_documents"] = max(1, min(20, int(advanced["k_documents"])))
        
        return settings
    
    def get_llm_config(self, settings: Dict[str, Any] = None) -> Dict[str, str]:
        """LLM 설정을 반환합니다."""
        if settings is None:
            settings = self.load_settings()
        
        llm_config = settings.get("llm", {})
        return {
            "provider": llm_config.get("provider", "claude"),
            "model_name": llm_config.get("model_name", "claude-3-sonnet-20240229")
        }
    
    def get_supported_models(self) -> Dict[str, list]:
        """지원되는 모델 목록을 반환합니다."""
        return self.supported_models
    
    def update_llm_settings(self, provider: str, model_name: str) -> bool:
        """LLM 설정을 업데이트합니다."""
        settings = self.load_settings()
        settings["llm"]["provider"] = provider
        settings["llm"]["model_name"] = model_name
        return self.save_settings(settings)
    
    def reset_to_defaults(self) -> bool:
        """설정을 기본값으로 초기화합니다."""
        try:
            if self.settings_file.exists():
                self.settings_file.unlink()
                self.logger.info("Settings file deleted, will use defaults")
            return True
        except Exception as e:
            self.logger.error(f"Failed to reset settings: {e}")
            return False
    
    def export_settings(self, export_path: str) -> bool:
        """설정을 다른 파일로 내보냅니다."""
        try:
            settings = self.load_settings()
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=4, ensure_ascii=False)
            self.logger.info(f"Settings exported to {export_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to export settings: {e}")
            return False
    
    def import_settings(self, import_path: str) -> bool:
        """다른 파일에서 설정을 가져옵니다."""
        try:
            with open(import_path, 'r', encoding='utf-8') as f:
                settings = json.load(f)
            return self.save_settings(settings)
        except Exception as e:
            self.logger.error(f"Failed to import settings: {e}")
            return False


# 싱글톤 인스턴스
_user_settings_manager_instance = None

def get_user_settings_manager() -> UserSettingsManager:
    """싱글톤 UserSettingsManager 인스턴스 반환"""
    global _user_settings_manager_instance
    
    if _user_settings_manager_instance is None:
        _user_settings_manager_instance = UserSettingsManager()
    
    return _user_settings_manager_instance 