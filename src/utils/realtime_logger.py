import streamlit as st
import logging
import json
import time
import threading
import queue
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from .logger import GoogleDriveLogger

class StreamlitRealtimeLogger:
    """Streamlit과 연동되는 실시간 로깅 시스템"""
    
    def __init__(self, base_logger: Optional[GoogleDriveLogger] = None):
        """
        Args:
            base_logger: 기본 GoogleDriveLogger (있으면 사용)
        """
        self.base_logger = base_logger
        self.log_queue = queue.Queue()
        self.is_active = True
        
        # Streamlit 세션 상태에 로그 저장
        if 'realtime_logs' not in st.session_state:
            st.session_state.realtime_logs = []
        
        if 'log_container' not in st.session_state:
            st.session_state.log_container = None
    
    def log(self, level: str, message: str, extra: Optional[Dict[str, Any]] = None):
        """로그 메시지 추가"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message,
            "extra": extra or {}
        }
        
        # 기본 로거에도 저장
        if self.base_logger:
            getattr(self.base_logger, level.lower())(message, extra)
        
        # 실시간 로그에 추가
        st.session_state.realtime_logs.append(log_entry)
        
        # 최근 100개만 유지
        if len(st.session_state.realtime_logs) > 100:
            st.session_state.realtime_logs = st.session_state.realtime_logs[-100:]
    
    def info(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """정보 로그"""
        self.log("INFO", message, extra)
    
    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """경고 로그"""
        self.log("WARNING", message, extra)
    
    def error(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """에러 로그"""
        self.log("ERROR", message, extra)
    
    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """디버그 로그"""
        self.log("DEBUG", message, extra)
    
    def show_logs_sidebar(self, max_logs: int = 20):
        """사이드바에 실시간 로그 표시"""
        with st.sidebar:
            with st.expander("📋 실시간 로그", expanded=False):
                if st.session_state.realtime_logs:
                    # 최근 로그부터 표시
                    recent_logs = st.session_state.realtime_logs[-max_logs:]
                    recent_logs.reverse()
                    
                    for log_entry in recent_logs:
                        timestamp = log_entry['timestamp'][:19]  # Remove microseconds
                        level = log_entry['level']
                        message = log_entry['message']
                        
                        # 레벨별 색상
                        if level == "ERROR":
                            st.error(f"🔴 {timestamp} - {message}")
                        elif level == "WARNING":
                            st.warning(f"🟡 {timestamp} - {message}")
                        elif level == "INFO":
                            st.info(f"🔵 {timestamp} - {message}")
                        else:
                            st.text(f"⚪ {timestamp} - {message}")
                else:
                    st.text("아직 로그가 없습니다.")
                
                # 로그 클리어 버튼
                if st.button("🗑️ 로그 클리어", key="clear_logs"):
                    st.session_state.realtime_logs = []
                    st.rerun()
    
    def show_logs_main(self, title: str = "📋 시스템 로그"):
        """메인 화면에 로그 표시"""
        with st.expander(title, expanded=False):
            if st.session_state.realtime_logs:
                # 최근 로그 표시
                for log_entry in reversed(st.session_state.realtime_logs[-50:]):
                    timestamp = log_entry['timestamp'][:19]
                    level = log_entry['level']
                    message = log_entry['message']
                    extra = log_entry.get('extra', {})
                    
                    # 로그 엔트리 표시
                    col1, col2, col3 = st.columns([2, 1, 4])
                    
                    with col1:
                        st.text(timestamp)
                    with col2:
                        if level == "ERROR":
                            st.error(level, icon="🔴")
                        elif level == "WARNING":
                            st.warning(level, icon="🟡") 
                        elif level == "INFO":
                            st.info(level, icon="🔵")
                        else:
                            st.text(level)
                    with col3:
                        st.text(message)
                        if extra:
                            st.json(extra, expanded=False)
            else:
                st.info("아직 로그가 없습니다.")
    
    def measure_time(self, operation_name: str):
        """시간 측정 데코레이터 (실시간 로그 포함)"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                self.info(f"🔄 {operation_name} 시작")
                
                try:
                    result = func(*args, **kwargs)
                    elapsed_time = time.time() - start_time
                    
                    self.info(f"✅ {operation_name} 완료", extra={
                        "elapsed_time": f"{elapsed_time:.2f}초",
                        "status": "success"
                    })
                    
                    # 기본 로거에도 성능 데이터 저장
                    if self.base_logger and hasattr(self.base_logger, 'measure_time'):
                        if operation_name not in self.base_logger.performance_data:
                            self.base_logger.performance_data[operation_name] = []
                        
                        self.base_logger.performance_data[operation_name].append({
                            "timestamp": datetime.now().isoformat(),
                            "elapsed_time": elapsed_time,
                            "status": "success"
                        })
                        self.base_logger._save_performance_log()
                    
                    return result
                    
                except Exception as e:
                    elapsed_time = time.time() - start_time
                    self.error(f"❌ {operation_name} 실패", extra={
                        "elapsed_time": f"{elapsed_time:.2f}초",
                        "error": str(e)
                    })
                    
                    # 기본 로거에도 실패 데이터 저장
                    if self.base_logger and hasattr(self.base_logger, 'measure_time'):
                        if operation_name not in self.base_logger.performance_data:
                            self.base_logger.performance_data[operation_name] = []
                        
                        self.base_logger.performance_data[operation_name].append({
                            "timestamp": datetime.now().isoformat(),
                            "elapsed_time": elapsed_time,
                            "status": "failed",
                            "error": str(e)
                        })
                        self.base_logger._save_performance_log()
                    
                    raise
            
            return wrapper
        return decorator

# 글로벌 실시간 로거 인스턴스
_realtime_logger_instance = None

def get_realtime_logger(base_logger: Optional[GoogleDriveLogger] = None) -> StreamlitRealtimeLogger:
    """실시간 로거 인스턴스 반환"""
    global _realtime_logger_instance
    
    if _realtime_logger_instance is None:
        if base_logger is None:
            try:
                from .logger import get_logger
                base_logger = get_logger()
            except:
                pass
        
        _realtime_logger_instance = StreamlitRealtimeLogger(base_logger)
    
    return _realtime_logger_instance 