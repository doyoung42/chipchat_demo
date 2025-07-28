import logging
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import functools
import traceback

class GoogleDriveLogger:
    """Google Drive와 연동되는 로깅 시스템"""
    
    def __init__(self, log_dir: str = "/content/drive/MyDrive/chipchat_logs"):
        """
        Args:
            log_dir: Google Drive 내 로그 저장 경로
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 현재 세션의 로그 파일 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"chipchat_log_{timestamp}.json"
        self.performance_log = self.log_dir / f"performance_log_{timestamp}.json"
        
        # 로그 데이터 초기화
        self.logs = []
        self.performance_data = {}
        
        # 기본 로거 설정
        self.logger = logging.getLogger("ChipChat")
        self.logger.setLevel(logging.DEBUG)
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # 초기 로그
        self.info("Google Drive 로깅 시스템 초기화 완료", extra={"log_file": str(self.log_file)})
    
    def _save_log(self, level: str, message: str, extra: Optional[Dict[str, Any]] = None):
        """로그를 메모리와 파일에 저장"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message,
            "extra": extra or {}
        }
        
        self.logs.append(log_entry)
        
        # 즉시 파일에 저장 (append 모드)
        try:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(self.logs, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"로그 파일 저장 실패: {e}")
    
    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """디버그 로그"""
        self.logger.debug(message)
        self._save_log("DEBUG", message, extra)
    
    def info(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """정보 로그"""
        self.logger.info(message)
        self._save_log("INFO", message, extra)
    
    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """경고 로그"""
        self.logger.warning(message)
        self._save_log("WARNING", message, extra)
    
    def error(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """에러 로그"""
        self.logger.error(message)
        self._save_log("ERROR", message, extra)
    
    def critical(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """심각한 에러 로그"""
        self.logger.critical(message)
        self._save_log("CRITICAL", message, extra)
    
    def measure_time(self, operation_name: str):
        """데코레이터: 함수 실행 시간 측정"""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                self.info(f"{operation_name} 시작")
                
                try:
                    result = func(*args, **kwargs)
                    elapsed_time = time.time() - start_time
                    
                    # 성능 데이터 저장
                    if operation_name not in self.performance_data:
                        self.performance_data[operation_name] = []
                    
                    self.performance_data[operation_name].append({
                        "timestamp": datetime.now().isoformat(),
                        "elapsed_time": elapsed_time,
                        "status": "success"
                    })
                    
                    self.info(f"{operation_name} 완료", extra={
                        "elapsed_time": f"{elapsed_time:.2f}초",
                        "status": "success"
                    })
                    
                    # 성능 로그 저장
                    self._save_performance_log()
                    
                    return result
                    
                except Exception as e:
                    elapsed_time = time.time() - start_time
                    error_info = {
                        "error": str(e),
                        "traceback": traceback.format_exc()
                    }
                    
                    # 성능 데이터 저장 (실패)
                    if operation_name not in self.performance_data:
                        self.performance_data[operation_name] = []
                    
                    self.performance_data[operation_name].append({
                        "timestamp": datetime.now().isoformat(),
                        "elapsed_time": elapsed_time,
                        "status": "failed",
                        "error": str(e)
                    })
                    
                    self.error(f"{operation_name} 실패", extra={
                        "elapsed_time": f"{elapsed_time:.2f}초",
                        "error_info": error_info
                    })
                    
                    # 성능 로그 저장
                    self._save_performance_log()
                    
                    raise
            
            return wrapper
        return decorator
    
    def _save_performance_log(self):
        """성능 데이터를 파일에 저장"""
        try:
            with open(self.performance_log, 'w', encoding='utf-8') as f:
                json.dump(self.performance_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"성능 로그 파일 저장 실패: {e}")
    
    def log_system_info(self):
        """시스템 정보 로그"""
        import platform
        import sys
        
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            cuda_device = torch.cuda.get_device_name(0) if cuda_available else "N/A"
        except:
            cuda_available = False
            cuda_device = "N/A"
        
        system_info = {
            "platform": platform.platform(),
            "python_version": sys.version,
            "cuda_available": cuda_available,
            "cuda_device": cuda_device
        }
        
        self.info("시스템 정보", extra=system_info)
        return system_info
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """성능 요약 정보 반환"""
        summary = {}
        
        for operation, records in self.performance_data.items():
            if records:
                times = [r['elapsed_time'] for r in records if r['status'] == 'success']
                if times:
                    summary[operation] = {
                        "count": len(records),
                        "success_count": len(times),
                        "fail_count": len(records) - len(times),
                        "avg_time": sum(times) / len(times),
                        "min_time": min(times),
                        "max_time": max(times),
                        "total_time": sum(times)
                    }
        
        return summary

# 글로벌 로거 인스턴스
_logger_instance = None

def get_logger(log_dir: Optional[str] = None) -> GoogleDriveLogger:
    """싱글톤 로거 인스턴스 반환"""
    global _logger_instance
    
    if _logger_instance is None:
        if log_dir is None:
            # 환경 감지
            try:
                from google.colab import drive
                log_dir = "/content/drive/MyDrive/chipchat_logs"
            except ImportError:
                log_dir = "./logs"
        
        _logger_instance = GoogleDriveLogger(log_dir)
    
    return _logger_instance 