import os
import shutil
import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import hashlib
import zipfile

from .logger import get_logger

class HFModelCache:
    """HuggingFace 모델을 Google Drive에 캐싱하는 시스템"""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Args:
            cache_dir: 모델 캐시 디렉토리 (None이면 환경변수나 환경 감지로 자동 설정)
        """
        if cache_dir is None:
            # 환경변수에서 캐시 디렉토리 확인
            cache_dir = os.environ.get('MODEL_CACHE_DIR')
            
            if cache_dir is None:
                # 환경 감지로 기본값 설정
                try:
                    from google.colab import drive
                    # Google Colab 환경
                    cache_dir = "/content/drive/MyDrive/hf_model_cache"
                except ImportError:
                    # 로컬 환경
                    cache_dir = "./hf_model_cache"
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = get_logger()
        self.cache_metadata_file = self.cache_dir / "cache_metadata.json"
        self.cache_metadata = self._load_cache_metadata()
        
        # HuggingFace 캐시 디렉토리 설정
        self.hf_cache_dir = Path.home() / '.cache' / 'huggingface'
        
        self.logger.info("HF 모델 캐시 시스템 초기화", extra={
            "cache_dir": str(self.cache_dir),
            "hf_cache_dir": str(self.hf_cache_dir)
        })
    
    def _load_cache_metadata(self) -> Dict[str, Any]:
        """캐시 메타데이터 로드"""
        if self.cache_metadata_file.exists():
            try:
                with open(self.cache_metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"캐시 메타데이터 로드 실패: {e}")
                return {}
        return {}
    
    def _save_cache_metadata(self):
        """캐시 메타데이터 저장"""
        try:
            with open(self.cache_metadata_file, 'w') as f:
                json.dump(self.cache_metadata, f, indent=2)
        except Exception as e:
            self.logger.error(f"캐시 메타데이터 저장 실패: {e}")
    
    def _get_model_hash(self, model_name: str) -> str:
        """모델 이름의 해시값 생성"""
        return hashlib.md5(model_name.encode()).hexdigest()[:8]
    
    @get_logger().measure_time("모델 캐시 확인")
    def is_model_cached(self, model_name: str) -> bool:
        """모델이 캐시되어 있는지 확인"""
        model_hash = self._get_model_hash(model_name)
        
        if model_name in self.cache_metadata:
            cache_info = self.cache_metadata[model_name]
            cache_file = self.cache_dir / cache_info['filename']
            
            if cache_file.exists():
                self.logger.info(f"모델 캐시 발견: {model_name}", extra={
                    "cache_file": str(cache_file),
                    "cached_at": cache_info.get('cached_at', 'unknown')
                })
                return True
        
        return False
    
    @get_logger().measure_time("모델 Google Drive 저장")
    def save_model_to_cache(self, model_name: str, local_model_path: Optional[Path] = None):
        """로컬 HF 캐시의 모델을 Google Drive에 저장"""
        try:
            model_hash = self._get_model_hash(model_name)
            cache_filename = f"{model_name.replace('/', '_')}_{model_hash}.zip"
            cache_file_path = self.cache_dir / cache_filename
            
            # HuggingFace 캐시에서 모델 찾기
            if local_model_path is None:
                # transformers 캐시 디렉토리 확인
                transformers_cache = self.hf_cache_dir / 'hub'
                if not transformers_cache.exists():
                    self.logger.error(f"HF 캐시 디렉토리를 찾을 수 없음: {transformers_cache}")
                    return False
                
                # 모델 관련 파일들 찾기
                model_files = []
                model_name_safe = model_name.replace('/', '--')
                
                for item in transformers_cache.iterdir():
                    if model_name_safe in str(item):
                        model_files.append(item)
                
                if not model_files:
                    self.logger.error(f"로컬 캐시에서 모델을 찾을 수 없음: {model_name}")
                    return False
            else:
                model_files = [local_model_path]
            
            self.logger.info(f"모델 파일 압축 중: {len(model_files)}개 항목")
            
            # ZIP 파일로 압축
            with zipfile.ZipFile(cache_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in model_files:
                    if file_path.is_file():
                        arcname = file_path.name
                        zipf.write(file_path, arcname)
                        self.logger.debug(f"압축 추가: {arcname}")
                    elif file_path.is_dir():
                        for root, dirs, files in os.walk(file_path):
                            for file in files:
                                file_path_full = Path(root) / file
                                arcname = str(file_path_full.relative_to(file_path.parent))
                                zipf.write(file_path_full, arcname)
            
            # 메타데이터 업데이트
            self.cache_metadata[model_name] = {
                'filename': cache_filename,
                'cached_at': datetime.now().isoformat(),
                'size_bytes': cache_file_path.stat().st_size,
                'model_hash': model_hash
            }
            self._save_cache_metadata()
            
            self.logger.info(f"모델 캐시 저장 완료: {model_name}", extra={
                "cache_file": str(cache_file_path),
                "size_mb": f"{cache_file_path.stat().st_size / 1024 / 1024:.2f}"
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"모델 캐시 저장 실패: {e}")
            return False
    
    @get_logger().measure_time("모델 Google Drive 로드")
    def load_model_from_cache(self, model_name: str) -> bool:
        """Google Drive에서 모델을 로컬 HF 캐시로 복원"""
        try:
            if model_name not in self.cache_metadata:
                self.logger.error(f"캐시 메타데이터에 모델이 없음: {model_name}")
                return False
            
            cache_info = self.cache_metadata[model_name]
            cache_file_path = self.cache_dir / cache_info['filename']
            
            if not cache_file_path.exists():
                self.logger.error(f"캐시 파일이 존재하지 않음: {cache_file_path}")
                return False
            
            # HuggingFace 캐시 디렉토리 준비
            transformers_cache = self.hf_cache_dir / 'hub'
            transformers_cache.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"캐시 파일 압축 해제 중: {cache_file_path}")
            
            # ZIP 파일 압축 해제
            with zipfile.ZipFile(cache_file_path, 'r') as zipf:
                zipf.extractall(transformers_cache)
            
            self.logger.info(f"모델 캐시 로드 완료: {model_name}", extra={
                "cache_file": str(cache_file_path),
                "target_dir": str(transformers_cache)
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"모델 캐시 로드 실패: {e}")
            return False
    
    def get_cache_info(self) -> Dict[str, Any]:
        """캐시 정보 반환"""
        total_size = 0
        model_count = len(self.cache_metadata)
        
        for model_info in self.cache_metadata.values():
            if 'size_bytes' in model_info:
                total_size += model_info['size_bytes']
        
        return {
            'cache_dir': str(self.cache_dir),
            'model_count': model_count,
            'total_size_mb': total_size / 1024 / 1024,
            'models': list(self.cache_metadata.keys())
        }
    
    def clear_cache(self, model_name: Optional[str] = None):
        """캐시 삭제"""
        try:
            if model_name:
                # 특정 모델만 삭제
                if model_name in self.cache_metadata:
                    cache_info = self.cache_metadata[model_name]
                    cache_file = self.cache_dir / cache_info['filename']
                    
                    if cache_file.exists():
                        cache_file.unlink()
                    
                    del self.cache_metadata[model_name]
                    self._save_cache_metadata()
                    
                    self.logger.info(f"모델 캐시 삭제: {model_name}")
            else:
                # 전체 캐시 삭제
                for cache_file in self.cache_dir.glob("*.zip"):
                    cache_file.unlink()
                
                self.cache_metadata.clear()
                self._save_cache_metadata()
                
                self.logger.info("전체 모델 캐시 삭제 완료")
                
        except Exception as e:
            self.logger.error(f"캐시 삭제 실패: {e}")

# 전역 캐시 인스턴스
_model_cache_instance = None

def get_model_cache(cache_dir: Optional[str] = None) -> HFModelCache:
    """싱글톤 모델 캐시 인스턴스 반환"""
    global _model_cache_instance
    
    if _model_cache_instance is None:
        if cache_dir is None:
            # config.json에서 캐시 디렉토리 읽어오기
            try:
                from .config_manager import get_config_manager
                config = get_config_manager()
                cache_dir = config.get_path('model_cache_folder')
            except Exception:
                # config.json 읽기 실패 시 환경 감지로 폴백
                try:
                    from google.colab import drive
                    # Google Colab 환경
                    cache_dir = "/content/drive/MyDrive/hf_model_cache"
                except ImportError:
                    # 로컬 환경
                    cache_dir = "./hf_model_cache"
        
        _model_cache_instance = HFModelCache(cache_dir)
    
    return _model_cache_instance 