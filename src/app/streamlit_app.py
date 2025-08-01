"""
ChipChat - Streamlit App (페이지 분리 버전)
설정 페이지와 챗 페이지로 분리된 LangGraph 기반 멀티턴 챗봇
"""
import streamlit as st
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import modules
from src.utils.user_settings_manager import get_user_settings_manager
from src.app.initialization import setup_paths, load_api_keys, initialize_managers, initialize_agent
from src.utils.optimized_loaders import (
    initialize_optimized_managers, initialize_optimized_agent, 
    get_cached_paths, get_cached_api_keys, clear_all_caches, clear_model_caches
)

# 실시간 로깅 시스템
try:
    from src.utils.realtime_logger import get_realtime_logger
    from src.utils.logger import get_logger
    base_logger = get_logger()
    logger = get_realtime_logger(base_logger)
    logger.info("ChipChat 앱 시작 (실시간 로깅 활성화)")
except:
    import logging
    logger = logging.getLogger(__name__)
    logger.info("기본 로깅 시스템 사용")

# Page configuration
st.set_page_config(
    page_title="ChipChat - 데이터시트 챗봇",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="collapsed"  # 사이드바 기본 숨김
)


def show_settings_page():
    """설정 페이지를 표시합니다."""
    st.title("⚙️ ChipChat 설정")
    st.caption("🔧 LLM 모델 선택 및 기본 설정")
    
    settings_manager = get_user_settings_manager()
    current_settings = settings_manager.load_settings()
    
    # API 키 상태 확인
    api_keys = load_api_keys()
    
    st.markdown("---")
    
    # API 키 상태 표시
    st.subheader("🔑 API 키 상태")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        openai_status = "✅ 설정됨" if api_keys.get('openai') else "❌ 미설정"
        st.metric("OpenAI", openai_status)
    
    with col2:
        claude_status = "✅ 설정됨" if api_keys.get('anthropic') else "❌ 미설정"
        st.metric("Claude", claude_status)
    
    with col3:
        hf_status = "✅ 설정됨" if api_keys.get('huggingface') else "❌ 미설정"
        st.metric("HuggingFace", hf_status)
    
    if not any(api_keys.values()):
        st.error("🚨 최소 하나 이상의 API 키가 필요합니다!")
        st.markdown("""
        **API 키 설정 방법:**
        1. main.ipynb의 3단계에서 API 키를 입력하세요
        2. 또는 환경변수로 설정하세요:
           - `OPENAI_API_KEY`
           - `ANTHROPIC_API_KEY` 
           - `HF_TOKEN`
        """)
        return
    
    st.markdown("---")
    
    # LLM 모델 설정
    st.subheader("🤖 LLM 모델 설정")
    
    supported_models = settings_manager.get_supported_models()
    current_llm = current_settings.get("llm", {})
    
    # Provider 선택
    provider_options = []
    if api_keys.get('openai'):
        provider_options.append("openai")
    if api_keys.get('anthropic'):
        provider_options.append("claude")
    
    if not provider_options:
        st.error("사용 가능한 LLM 모델이 없습니다. API 키를 먼저 설정해주세요.")
        return
    
    current_provider = current_llm.get("provider", "claude")
    if current_provider not in provider_options:
        current_provider = provider_options[0]
    
    selected_provider = st.selectbox(
        "🏢 LLM Provider 선택",
        options=provider_options,
        index=provider_options.index(current_provider) if current_provider in provider_options else 0,
        format_func=lambda x: "OpenAI" if x == "openai" else "Claude"
    )
    
    # Model 선택
    available_models = supported_models.get(selected_provider, [])
    current_model = current_llm.get("model_name", "")
    
    model_names = [m["name"] for m in available_models]
    model_displays = [m["display"] for m in available_models]
    
    if current_model in model_names:
        current_index = model_names.index(current_model)
    else:
        current_index = 0
    
    selected_model = st.selectbox(
        "🎯 모델 선택",
        options=model_names,
        index=current_index,
        format_func=lambda x: model_displays[model_names.index(x)] if x in model_names else x
    )
    
    st.markdown("---")
    
    # 고급 설정
    with st.expander("🔧 고급 설정", expanded=False):
        advanced_settings = current_settings.get("advanced", {})
        
        temperature = st.slider(
            "Temperature (창의성)",
            min_value=0.0,
            max_value=2.0,
            value=advanced_settings.get("temperature", 0.7),
            step=0.1,
            help="높을수록 더 창의적이지만 불안정할 수 있습니다"
        )
        
        max_tokens = st.number_input(
            "Max Tokens (최대 토큰 수)",
            min_value=100,
            max_value=4000,
            value=advanced_settings.get("max_tokens", 2000),
            step=100,
            help="응답의 최대 길이를 제한합니다"
        )
        
        k_documents = st.number_input(
            "검색할 문서 수",
            min_value=1,
            max_value=20,
            value=advanced_settings.get("k_documents", 5),
            step=1,
            help="벡터 검색에서 가져올 문서의 수"
        )
        
        # 캐시 관리 섹션
        st.markdown("---")
        st.subheader("🧹 캐시 관리")
        st.markdown("성능 향상을 위해 모델과 데이터를 캐시합니다. 문제가 발생하면 캐시를 정리해보세요.")
        
        col_cache1, col_cache2 = st.columns(2)
        
        with col_cache1:
            if st.button("🗑️ 전체 캐시 정리", help="모든 캐시를 지웁니다"):
                with st.spinner("🔄 전체 캐시 정리 중..."):
                    if clear_all_caches():
                        st.success("✅ 모든 캐시가 정리되었습니다!")
                        logger.info("All caches cleared by user")
                    else:
                        st.error("❌ 캐시 정리에 실패했습니다.")
        
        with col_cache2:
            from src.utils.optimized_loaders import get_cache_info
            cache_info = get_cache_info()
            
            with st.popover("ℹ️ 캐시 정보"):
                st.markdown("**캐시된 함수들:**")
                for func_name in cache_info["cached_functions"]:
                    st.markdown(f"• `{func_name}`")
                
                st.markdown("**설명:**")
                st.markdown(cache_info["description"])
    
    st.markdown("---")
    
    # 설정 저장 버튼
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("💾 설정 저장", type="primary", use_container_width=True):
            # 모델 변경 여부 확인
            model_changed = (
                selected_provider != current_llm.get("provider") or 
                selected_model != current_llm.get("model_name")
            )
            
            # 설정 업데이트
            new_settings = current_settings.copy()
            new_settings["llm"]["provider"] = selected_provider
            new_settings["llm"]["model_name"] = selected_model
            new_settings["advanced"]["temperature"] = temperature
            new_settings["advanced"]["max_tokens"] = max_tokens
            new_settings["advanced"]["k_documents"] = k_documents
            
            if settings_manager.save_settings(new_settings):
                st.success("✅ 설정이 저장되었습니다!")
                
                # 모델이 변경된 경우 관련 캐시 정리
                if model_changed:
                    with st.spinner("🔄 모델 캐시 정리 중..."):
                        if clear_model_caches(selected_provider, selected_model):
                            st.success("🧹 모델 캐시가 정리되었습니다!")
                            logger.info(f"Model caches cleared for: {selected_provider}/{selected_model}")
                        else:
                            st.warning("⚠️ 모델 캐시 정리에 실패했습니다.")
                
                st.session_state.settings_changed = True
                logger.info(f"Settings saved: {selected_provider}/{selected_model}")
            else:
                st.error("❌ 설정 저장에 실패했습니다.")
    
    with col2:
        if st.button("🚀 챗봇 시작", use_container_width=True):
            st.session_state.page = "chat"
            st.session_state.settings_applied = False
            st.rerun()
    
    with col3:
        if st.button("🔄 기본값 복원", use_container_width=True):
            if settings_manager.reset_to_defaults():
                st.success("✅ 기본 설정으로 복원되었습니다!")
                st.rerun()
            else:
                st.error("❌ 설정 복원에 실패했습니다.")
    
    # 현재 설정 정보 표시
    st.markdown("---")
    
    with st.expander("📊 현재 설정 정보", expanded=False):
        st.json({
            "LLM Provider": selected_provider,
            "Model": selected_model,
            "Temperature": temperature,
            "Max Tokens": max_tokens,
            "K Documents": k_documents,
            "Created": current_settings.get("created_at"),
            "Updated": current_settings.get("updated_at")
        })


def show_chat_page():
    """챗 페이지를 표시합니다."""
    # 상단 설정 버튼
    col1, col2, col3 = st.columns([1, 6, 1])
    with col1:
        if st.button("⚙️ 설정", help="설정 페이지로 이동"):
            st.session_state.page = "settings"
            st.rerun()
    
    with col2:
        st.title("💬 ChipChat - 데이터시트 챗봇")
    
    with col3:
        if st.button("🔄 새로고침", help="챗 세션 초기화"):
            # 챗 관련 세션 상태 초기화
            for key in ['messages', 'chat_manager', 'vectorstore_manager', 'vectorstore', 'agent']:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.settings_applied = False
            st.rerun()
    
    st.caption("🤖 LangGraph 기반 멀티에이전트 시스템")
    
    # 설정 로드 및 시스템 초기화 (무한루프 방지)
    if not st.session_state.get('settings_applied', False):
        # 초기화 시도 횟수 추적
        if "init_attempts" not in st.session_state:
            st.session_state.init_attempts = 0
        
        # 최대 3회 시도 후 설정 페이지로 이동
        if st.session_state.init_attempts >= 3:
            st.error("❌ 시스템 초기화에 반복적으로 실패했습니다.")
            st.markdown("**가능한 해결 방법:**")
            st.markdown("1. API 키가 올바르게 설정되었는지 확인")
            st.markdown("2. 필요한 파일들(chipDB.csv, vectorstore 등)이 존재하는지 확인")
            st.markdown("3. 설정을 다시 확인하고 저장")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("⚙️ 설정 페이지로 이동", type="primary"):
                    st.session_state.page = "settings"
                    st.session_state.init_attempts = 0  # 카운터 리셋
                    st.session_state.settings_applied = False
                    st.rerun()
            
            with col2:
                if st.button("🔄 다시 시도"):
                    st.session_state.init_attempts = 0  # 카운터 리셋
                    st.session_state.settings_applied = False
                    st.rerun()
            return
        
        st.session_state.init_attempts += 1
        
        with st.spinner(f"🔧 시스템 초기화 중... (시도 {st.session_state.init_attempts}/3)"):
            try:
                settings_manager = get_user_settings_manager()
                settings = settings_manager.load_settings()
                llm_config = settings_manager.get_llm_config(settings)
                
                # 캐시된 경로 및 API 키 설정 (최적화)
                st.session_state.paths = get_cached_paths()
                st.session_state.api_keys = get_cached_api_keys()
                
                # LLM 설정 표시
                st.info(f"🤖 사용 중인 모델: {llm_config['provider'].upper()} - {llm_config['model_name']}")
                
                # 최적화된 매니저 초기화 (캐싱 활용)
                chat_manager, vectorstore_manager, vectorstore, error = initialize_optimized_managers(
                    provider=llm_config['provider'], 
                    model_name=llm_config['model_name']
                )
                
                if error:
                    st.error(f"❌ 매니저 초기화 실패: {error}")
                    if st.session_state.init_attempts >= 3:
                        return  # 최대 시도 횟수 도달 시 위의 오류 처리로 이동
                    else:
                        st.warning(f"🔄 {3 - st.session_state.init_attempts}회 더 시도합니다...")
                        time.sleep(1)  # 잠시 대기
                        st.rerun()
                        return
                
                # 최적화된 에이전트 초기화 (캐싱 활용)
                agent, agent_error = initialize_optimized_agent(
                    chat_manager, vectorstore_manager, vectorstore,
                    st.session_state.paths['chipdb_path']
                )
                
                if agent_error:
                    st.error(f"❌ 에이전트 초기화 실패: {agent_error}")
                    if st.session_state.init_attempts >= 3:
                        return  # 최대 시도 횟수 도달 시 위의 오류 처리로 이동
                    else:
                        st.warning(f"🔄 {3 - st.session_state.init_attempts}회 더 시도합니다...")
                        time.sleep(1)  # 잠시 대기
                        st.rerun()
                        return
                
                # 세션 상태에 저장
                st.session_state.chat_manager = chat_manager
                st.session_state.vectorstore_manager = vectorstore_manager
                st.session_state.vectorstore = vectorstore
                st.session_state.agent = agent
                st.session_state.settings_applied = True
                st.session_state.init_attempts = 0  # 성공 시 카운터 리셋
                
                logger.info("챗 페이지 초기화 완료")
                
            except Exception as e:
                st.error(f"❌ 초기화 중 예상치 못한 오류: {str(e)}")
                logger.error(f"챗 페이지 초기화 오류: {str(e)}")
                
                if st.session_state.init_attempts >= 3:
                    return  # 최대 시도 횟수 도달 시 위의 오류 처리로 이동
                else:
                    st.warning(f"🔄 {3 - st.session_state.init_attempts}회 더 시도합니다...")
                    time.sleep(1)  # 잠시 대기
                    st.rerun()
                    return
    
    # API 키 확인
    if not any(st.session_state.api_keys.values()):
        st.error("🚨 API 키가 필요합니다!")
        st.markdown("설정 페이지에서 API 키를 확인해주세요.")
        if st.button("⚙️ 설정 페이지로 이동"):
            st.session_state.page = "settings"
            st.rerun()
        return
    
    # PDF 업로드 섹션
    st.markdown("### 📄 PDF 업로드")
    
    uploaded_file = st.file_uploader(
        "새로운 데이터시트 PDF를 업로드하세요",
        type=['pdf'],
        help="업로드된 PDF는 자동으로 처리되어 벡터스토어에 추가됩니다."
    )
    
    # PDF 처리
    if uploaded_file is not None:
        logger.info(f"PDF 업로드됨: {uploaded_file.name}")
        
        # 중복 체크
        existing_files = [doc['filename'] for doc in st.session_state.get('uploaded_documents', [])]
        if uploaded_file.name in existing_files:
            st.warning(f"⚠️ '{uploaded_file.name}' 파일이 이미 업로드되어 있습니다.")
        else:
            # PDF 처리
            with st.spinner(f"📄 {uploaded_file.name} 처리 중..."):
                try:
                    # PDF 내용 읽기
                    pdf_content = uploaded_file.read()
                    
                    # 에이전트를 통해 PDF 처리
                    if hasattr(st.session_state, 'agent') and st.session_state.agent:
                        logger.info(f"PDF 처리 시작: {uploaded_file.name}")
                        
                        # process_new_pdf 도구 직접 호출
                        result = st.session_state.agent.tools.process_new_pdf(pdf_content, uploaded_file.name)
                        
                        if "✅ Successfully processed" in result:
                            st.success(result)
                            logger.info(f"PDF 처리 완료: {uploaded_file.name}")
                        else:
                            st.error(result)
                            logger.error(f"PDF 처리 실패: {uploaded_file.name}")
                    else:
                        st.error("❌ 에이전트가 초기화되지 않았습니다.")
                        logger.error("PDF 처리 실패: 에이전트 없음")
                        
                except Exception as e:
                    error_msg = f"❌ PDF 처리 중 오류: {str(e)}"
                    st.error(error_msg)
                    logger.error(f"PDF 처리 오류: {str(e)}")
    
    # 채팅 인터페이스
    st.markdown("---")
    st.markdown("### 💬 질의응답")
    
    # 채팅 히스토리 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # 채팅 히스토리 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # 사용자 입력
    if prompt := st.chat_input("질문을 입력하세요 (예: 전압 변환기 기능을 하는 부품들을 알려줘)"):
        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # 에이전트 응답 생성
        with st.chat_message("assistant"):
            with st.spinner("🤔 생각 중..."):
                try:
                    # 에이전트 실행
                    response = st.session_state.agent.process_query(prompt, None)
                    
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
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_msg
                    })
                    logger.error(f"쿼리 처리 중 오류: {str(e)}", extra={"query": prompt})


def main():
    """메인 앱 함수"""
    # 무한루프 방지를 위한 재시도 카운터
    if "rerun_counter" not in st.session_state:
        st.session_state.rerun_counter = 0
    
    # 과도한 재시도 방지 (최대 3회)
    if st.session_state.rerun_counter > 3:
        st.error("❌ 페이지 초기화에 문제가 발생했습니다. 페이지를 새로고침해주세요.")
        st.markdown("**문제 해결 방법:**")
        st.markdown("1. 브라우저 새로고침 (F5 또는 Ctrl+R)")
        st.markdown("2. 브라우저 캐시 삭제")
        st.markdown("3. 다른 브라우저에서 시도")
        st.stop()
        return
    
    # 사용자 설정 매니저 초기화
    try:
        settings_manager = get_user_settings_manager()
    except Exception as e:
        st.error(f"❌ 설정 관리자 초기화 실패: {e}")
        st.markdown("user_settings.json 파일에 문제가 있을 수 있습니다.")
        if st.button("🔄 설정 파일 초기화"):
            try:
                from pathlib import Path
                Path("user_settings.json").unlink(missing_ok=True)
                st.success("✅ 설정 파일이 초기화되었습니다. 페이지를 새로고침해주세요.")
            except Exception as e2:
                st.error(f"❌ 설정 파일 초기화 실패: {e2}")
        st.stop()
        return
    
    # 페이지 상태 초기화 (무한루프 방지)
    if "page" not in st.session_state:
        # 설정 파일이 있으면 챗 페이지로, 없으면 설정 페이지로
        try:
            if settings_manager.settings_exist():
                st.session_state.page = "chat"
            else:
                st.session_state.page = "settings"
        except Exception as e:
            logger.error(f"페이지 상태 초기화 실패: {e}")
            st.session_state.page = "settings"  # 안전한 기본값
    
    # 페이지 상태 유효성 검증
    valid_pages = ["settings", "chat"]
    if st.session_state.page not in valid_pages:
        logger.warning(f"잘못된 페이지 상태: {st.session_state.page}")
        st.session_state.page = "settings"
        # rerun_counter 증가 후 재시도
        st.session_state.rerun_counter += 1
        st.rerun()
        return
    
    # 정상적인 페이지 라우팅 시 카운터 리셋
    if st.session_state.rerun_counter > 0:
        st.session_state.rerun_counter = 0
    
    # 페이지 라우팅
    try:
        if st.session_state.page == "settings":
            show_settings_page()
        elif st.session_state.page == "chat":
            show_chat_page()
    except Exception as e:
        st.error(f"❌ 페이지 렌더링 중 오류 발생: {str(e)}")
        logger.error(f"페이지 렌더링 오류: {str(e)}")
        
        # 오류 발생 시 설정 페이지로 안전하게 이동
        if st.button("🔧 설정 페이지로 이동"):
            st.session_state.page = "settings"
            # 문제가 있는 세션 상태들 정리
            problematic_keys = ['chat_manager', 'vectorstore_manager', 'vectorstore', 'agent', 'settings_applied']
            for key in problematic_keys:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()


if __name__ == "__main__":
    main() 