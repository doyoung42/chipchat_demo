"""
ChipChat - Streamlit App
간소화된 LangGraph 기반 멀티턴 챗봇
"""
import streamlit as st
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import modules
from src.app.ui_components import (
    show_system_status, show_performance_metrics, show_llm_settings,
    show_chat_controls, show_agent_info, init_chat_container, add_chat_message
)
from src.app.initialization import (
    setup_paths, load_api_keys, initialize_managers, initialize_agent
)

# 로깅 시스템
try:
    from src.utils.logger import get_logger
    logger = get_logger()
    logger.info("ChipChat 앱 시작")
except:
    import logging
    logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="ChipChat - 데이터시트 챗봇",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """메인 앱 함수"""
    st.title("💬 ChipChat - 데이터시트 챗봇")
    st.caption("🤖 LangGraph 기반 멀티에이전트 시스템")
    
    # 초기 설정
    if 'initialized' not in st.session_state:
        with st.spinner("🔧 시스템 초기화 중..."):
            # 경로 설정
            st.session_state.paths = setup_paths()
            st.session_state.api_keys = load_api_keys()
            st.session_state.initialized = True
            logger.info("초기 설정 완료")
    
    # 사이드바 설정
    with st.sidebar:
        st.header("⚙️ 설정")
        
        # LLM 설정
        provider, model_name = show_llm_settings()
        
        # LLM 변경 감지
        if 'current_provider' not in st.session_state:
            st.session_state.current_provider = provider
            st.session_state.current_model = model_name
        
        if (provider != st.session_state.current_provider or 
            model_name != st.session_state.current_model):
            # 변경 시 재초기화 필요
            for key in ['managers_initialized', 'chat_manager', 'vectorstore_manager', 
                       'vectorstore', 'agent']:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.current_provider = provider
            st.session_state.current_model = model_name
            st.rerun()
        
        st.divider()
        
        # 시스템 상태
        show_system_status(st.session_state.paths)
        
        # 성능 메트릭
        show_performance_metrics(logger)
        
        st.divider()
        
        # 채팅 컨트롤
        show_agent_info_flag = show_chat_controls()
    
    # API 키 확인
    if not any(st.session_state.api_keys.values()):
        st.error("🚨 API 키가 필요합니다!")
        st.markdown("""
        Colab 노트북의 3단계에서 API 키를 입력해주세요:
        - OpenAI API Key
        - Claude API Key  
        - HuggingFace Token
        """)
        return
    
    # 매니저 초기화
    if 'managers_initialized' not in st.session_state:
        chat_manager, vectorstore_manager, vectorstore, error = initialize_managers(
            provider=provider, 
            model_name=model_name
        )
        
        if error:
            st.error(f"❌ 초기화 실패: {error}")
            st.stop()
            return
        
        # 에이전트 초기화
        agent, agent_error = initialize_agent(
            chat_manager, vectorstore_manager, vectorstore,
            st.session_state.paths['chipdb_path']
        )
        
        if agent_error:
            st.error(f"❌ 에이전트 초기화 실패: {agent_error}")
            st.stop()
            return
        
        # 세션 상태에 저장
        st.session_state.chat_manager = chat_manager
        st.session_state.vectorstore_manager = vectorstore_manager
        st.session_state.vectorstore = vectorstore
        st.session_state.agent = agent
        st.session_state.managers_initialized = True
        
        logger.info("모든 매니저 초기화 완료")
    
    # 에이전트 정보 표시
    if show_agent_info_flag:
        show_agent_info(st.session_state.agent)
    
    # 채팅 인터페이스
    st.markdown("---")
    
    # 채팅 컨테이너 초기화
    init_chat_container()
    
    # 사용자 입력
    if prompt := st.chat_input("질문을 입력하세요 (예: 전압 변환기 기능을 하는 부품들을 알려줘)"):
        # 사용자 메시지 추가
        add_chat_message("user", prompt)
        
        # 에이전트 응답 생성
        with st.chat_message("assistant"):
            with st.spinner("🤔 생각 중..."):
                try:
                    # 파일 업로드 처리 (추후 구현)
                    uploaded_file = None
                    
                    # 에이전트 실행
                    response = st.session_state.agent.process_query(prompt, uploaded_file)
                    
                    # 응답 표시
                    st.markdown(response)
                    
                    # 히스토리에 추가
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response
                    })
                    
                    # 로깅
                    logger.info("쿼리 처리 완료", extra={
                        "query": prompt[:100],
                        "response_length": len(response)
                    })
                    
                except Exception as e:
                    error_msg = f"❌ 오류가 발생했습니다: {str(e)}"
                    st.error(error_msg)
                    logger.error(f"쿼리 처리 중 오류: {str(e)}", extra={"query": prompt})

if __name__ == "__main__":
    main() 