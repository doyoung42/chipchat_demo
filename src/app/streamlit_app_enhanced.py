import streamlit as st
import json
import os
import time
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

# Import from the new directory structure
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.chat_manager import ChatManager
from src.models.vectorstore_manager import VectorstoreManager
from src.models.langgraph_agent import ChipChatAgent

# Page configuration
st.set_page_config(
    page_title="ChipChat Enhanced - 개선된 버전",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

def detect_environment():
    """환경 감지 (로컬 vs Colab)"""
    try:
        from google.colab import drive
        return "colab"
    except ImportError:
        return "local"

def setup_paths():
    """환경에 따른 경로 설정"""
    env = detect_environment()
    
    if env == "colab":
        # Google Colab 환경
        try:
            from google.colab import drive
            drive.mount('/content/drive')
        except:
            pass
        
        base_path = Path('/content/drive/MyDrive')
        return {
            'vectorstore_path': str(base_path / 'vectorstore'),
            'json_folder_path': str(base_path / 'prep_json'),
            'prompt_templates_path': str(base_path / 'prompt_templates'),
            'chipdb_path': str(base_path / 'prep_json' / 'chipDB.csv')
        }
    else:
        # 로컬 환경
        base_path = Path.cwd()
        return {
            'vectorstore_path': str(base_path / 'vectorstore'),
            'json_folder_path': str(base_path / 'prep' / 'prep_json'),
            'prompt_templates_path': str(base_path / 'prompt_templates'),
            'chipdb_path': str(base_path / 'prep' / 'prep_json' / 'chipDB.csv')
        }

def load_api_keys():
    """API 키 로드"""
    api_keys = {}
    
    # Try environment variables first
    api_keys['openai'] = os.environ.get('OPENAI_API_KEY', '')
    api_keys['anthropic'] = os.environ.get('ANTHROPIC_API_KEY', '')
    api_keys['huggingface'] = os.environ.get('HF_TOKEN', '')
    
    # Try streamlit secrets
    if hasattr(st, 'secrets'):
        api_keys['openai'] = api_keys['openai'] or st.secrets.get("openai_api_key", "")
        api_keys['anthropic'] = api_keys['anthropic'] or st.secrets.get("anthropic_api_key", "")
        api_keys['huggingface'] = api_keys['huggingface'] or st.secrets.get("hf_token", "")
    
    return api_keys

@st.cache_resource
def initialize_managers_with_progress():
    """매니저들 초기화 (진행 상황 표시)"""
    progress_container = st.empty()
    status_container = st.empty()
    
    try:
        # 1. ChatManager 초기화
        progress_container.progress(10)
        status_container.info("🔄 1/5 ChatManager 초기화 중...")
        time.sleep(0.5)  # UI 업데이트를 위한 대기
        
        from src.models.chat_manager import ChatManager
        chat_manager = ChatManager(provider="openai")
        
        progress_container.progress(30)
        status_container.success("✅ 1/5 ChatManager 초기화 완료")
        time.sleep(0.5)
        
        # 2. VectorstoreManager 초기화 (가장 오래 걸리는 부분)
        progress_container.progress(40)
        status_container.info("🔄 2/5 VectorstoreManager 초기화 중... (임베딩 모델 다운로드 중, 시간이 오래 걸릴 수 있습니다)")
        
        start_time = time.time()
        from src.models.vectorstore_manager import VectorstoreManager
        vectorstore_manager = VectorstoreManager()
        elapsed = time.time() - start_time
        
        progress_container.progress(80)
        status_container.success(f"✅ 2/5 VectorstoreManager 초기화 완료 ({elapsed:.1f}초)")
        time.sleep(0.5)
        
        progress_container.progress(100)
        status_container.success("✅ 모든 매니저 초기화 완료!")
        
        # 상태 표시 제거
        time.sleep(2)
        progress_container.empty()
        status_container.empty()
        
        return chat_manager, vectorstore_manager, None
        
    except Exception as e:
        progress_container.empty()
        status_container.error(f"❌ 매니저 초기화 실패: {str(e)}")
        return None, None, str(e)

@st.cache_resource
def load_vectorstore_with_progress(_vectorstore_manager, vectorstore_path):
    """벡터스토어 로드 (진행 상황 표시)"""
    progress_container = st.empty()
    status_container = st.empty()
    
    try:
        progress_container.progress(20)
        status_container.info("🔄 3/5 벡터스토어 확인 중...")
        
        if Path(vectorstore_path).exists():
            progress_container.progress(50)
            status_container.info("🔄 3/5 벡터스토어 로딩 중...")
            
            vectorstore = _vectorstore_manager.load_vectorstore(vectorstore_path)
            
            progress_container.progress(100)
            status_container.success("✅ 3/5 벡터스토어 로드 완료!")
            
        else:
            # Try loading from JSON files and creating vectorstore
            progress_container.progress(30)
            status_container.info("🔄 3/5 JSON 파일에서 벡터스토어 생성 중...")
            
            json_folder = st.session_state.paths['json_folder_path']
            if Path(json_folder).exists():
                json_data = _vectorstore_manager.load_json_files(json_folder)
                if json_data:
                    progress_container.progress(70)
                    status_container.info("🔄 3/5 벡터스토어 생성 중...")
                    
                    vectorstore = _vectorstore_manager.create_vectorstore(json_data)
                    # Save for future use
                    Path(vectorstore_path).parent.mkdir(parents=True, exist_ok=True)
                    _vectorstore_manager.save_vectorstore(vectorstore, vectorstore_path)
                    
                    progress_container.progress(100)
                    status_container.success("✅ 3/5 벡터스토어 생성 완료!")
                else:
                    raise Exception("JSON 데이터가 비어있습니다")
            else:
                raise Exception("JSON 폴더를 찾을 수 없습니다")
        
        time.sleep(1)
        progress_container.empty()
        status_container.empty()
        
        return vectorstore, None
        
    except Exception as e:
        progress_container.empty()
        status_container.error(f"❌ 벡터스토어 로드 실패: {str(e)}")
        return None, str(e)

@st.cache_resource
def initialize_agent_with_progress(_chat_manager, _vectorstore_manager, _vectorstore, chipdb_path):
    """에이전트 초기화 (진행 상황 표시)"""
    progress_container = st.empty()
    status_container = st.empty()
    
    try:
        progress_container.progress(20)
        status_container.info("🔄 4/5 ChipDB.csv 확인 중...")
        
        if not Path(chipdb_path).exists():
            raise Exception(f"chipDB.csv not found at {chipdb_path}")
        
        progress_container.progress(50)
        status_container.info("🔄 4/5 LangGraph 에이전트 초기화 중...")
        
        from src.models.langgraph_agent import ChipChatAgent
        agent = ChipChatAgent(
            csv_path=chipdb_path,
            vectorstore_manager=_vectorstore_manager,
            vectorstore=_vectorstore,
            llm_manager=_chat_manager.llm_manager
        )
        
        progress_container.progress(100)
        status_container.success("✅ 4/5 LangGraph 에이전트 초기화 완료!")
        
        time.sleep(1)
        progress_container.empty()
        status_container.empty()
        
        return agent, None
        
    except Exception as e:
        progress_container.empty()
        status_container.error(f"❌ 에이전트 초기화 실패: {str(e)}")
        return None, str(e)

def main():
    st.title("💬 ChipChat Enhanced - 개선된 버전")
    st.markdown("**로딩 진행 상황을 상세히 표시하는 개선된 버전입니다**")
    
    # Loading status indicator
    if 'initialization_complete' not in st.session_state:
        st.session_state.initialization_complete = False
    
    # Initialize session state
    if 'paths' not in st.session_state:
        st.session_state.paths = setup_paths()
    
    if 'api_keys' not in st.session_state:
        st.session_state.api_keys = load_api_keys()
    
    # Check API keys
    api_keys_available = any(st.session_state.api_keys.values())
    if not api_keys_available:
        st.error("🚨 API 키가 설정되지 않았습니다!")
        st.markdown("""
        다음 중 하나의 방법으로 API 키를 설정해주세요:
        1. **환경 변수**: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `HF_TOKEN`
        2. **Streamlit Secrets**: `.streamlit/secrets.toml` 파일
        3. **Colab**: 노트북의 3단계에서 API 키 입력
        """)
        return
    
    # Show loading indicator if not initialized
    if not st.session_state.initialization_complete:
        st.info("🔄 시스템 초기화 중... 아래 진행 상황을 확인하세요.")
        
        # Overall progress
        overall_progress = st.progress(0)
        overall_status = st.empty()
        
        overall_status.info("🚀 초기화 시작...")
        overall_progress.progress(5)
        
        # Initialize managers
        overall_status.info("🔄 1-2/5 매니저 초기화 중...")
        chat_manager, vectorstore_manager, init_error = initialize_managers_with_progress()
        
        if init_error:
            st.error(f"초기화 실패: {init_error}")
            return
        
        overall_progress.progress(40)
        
        # Load vectorstore
        overall_status.info("🔄 3/5 벡터스토어 로딩 중...")
        vectorstore, vs_error = load_vectorstore_with_progress(vectorstore_manager, st.session_state.paths['vectorstore_path'])
        
        if vs_error:
            st.error(f"벡터스토어 로드 실패: {vs_error}")
            st.info("prep 모듈을 먼저 실행하여 데이터시트를 전처리하고 벡터스토어를 생성해주세요.")
            return
        
        overall_progress.progress(70)
        
        # Initialize agent
        overall_status.info("🔄 4/5 AI 에이전트 초기화 중...")
        agent, agent_error = initialize_agent_with_progress(
            chat_manager, vectorstore_manager, vectorstore, 
            st.session_state.paths['chipdb_path']
        )
        
        overall_progress.progress(90)
        
        # Finalize
        overall_status.info("🔄 5/5 마무리 중...")
        overall_progress.progress(100)
        
        # Store in session state
        st.session_state.chat_manager = chat_manager
        st.session_state.vectorstore_manager = vectorstore_manager
        st.session_state.vectorstore = vectorstore
        st.session_state.agent = agent
        st.session_state.initialization_complete = True
        
        overall_status.success("✅ 모든 초기화 완료! 앱을 사용할 수 있습니다.")
        time.sleep(1)
        overall_progress.empty()
        overall_status.empty()
        
        st.rerun()  # Refresh to show the main interface
    
    # Main interface (only shown after initialization)
    if st.session_state.initialization_complete:
        chat_manager = st.session_state.chat_manager
        vectorstore_manager = st.session_state.vectorstore_manager
        vectorstore = st.session_state.vectorstore
        agent = st.session_state.agent
        
        # Sidebar configuration
        with st.sidebar:
            st.header("⚙️ 설정")
            
            # System status
            st.subheader("📊 시스템 상태")
            st.success("✅ 모든 구성 요소 로드됨")
            
            # LLM Provider selection
            provider = st.selectbox(
                "LLM 제공자",
                ["openai", "claude"],
                help="사용할 LLM 서비스를 선택하세요"
            )
            
            # Model selection
            if provider == "openai":
                model_options = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
            else:
                model_options = ["claude-3-sonnet-20240229", "claude-3-haiku-20240307", "claude-3-opus-20240229"]
            
            model_name = st.selectbox("모델", model_options)
            
            # Mode selection
            mode = st.selectbox(
                "모드 선택",
                ["🤖 AI Agent", "💬 Chat", "🔍 Retrieval Test", "📊 Vectorstore Info"]
            )
            
            # Reset button
            if st.button("🔄 시스템 재초기화"):
                for key in ['initialization_complete', 'chat_manager', 'vectorstore_manager', 'vectorstore', 'agent']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
        
        # Main content based on mode
        if mode == "🤖 AI Agent":
            if agent:
                render_agent_mode(agent)
            else:
                st.error("에이전트를 사용할 수 없습니다. 시스템을 재초기화해보세요.")
        elif mode == "💬 Chat":
            render_chat_mode(chat_manager, vectorstore)
        elif mode == "🔍 Retrieval Test":
            render_retrieval_test_mode(chat_manager, vectorstore)
        elif mode == "📊 Vectorstore Info":
            render_vectorstore_info_mode(vectorstore_manager, vectorstore)

def render_agent_mode(agent):
    """AI 에이전트 모드 렌더링"""
    st.header("🤖 AI Agent Mode")
    st.markdown("**스마트 에이전트가 질문을 분석하고 자동으로 최적의 도구를 선택합니다**")
    
    # Query input
    user_query = st.text_area(
        "질문을 입력하세요",
        placeholder="예: 전압 변환기 기능을 하는 모든 부품들을 알려줘",
        height=100
    )
    
    if st.button("🚀 에이전트 실행", type="primary") and user_query:
        with st.spinner("🤖 에이전트가 처리 중..."):
            response = agent.process_query(user_query)
        
        st.markdown("### 🤖 에이전트 응답")
        st.markdown(response)

def render_chat_mode(chat_manager, vectorstore):
    """채팅 모드 렌더링"""
    st.header("💬 채팅")
    
    user_input = st.text_input("질문을 입력하세요", placeholder="예: W25Q32JV의 전기적 특성은?")
    
    if user_input:
        with st.spinner("응답 생성 중..."):
            result = chat_manager.get_chat_response(
                query=user_input,
                vectorstore=vectorstore,
                k=5
            )
        
        st.markdown("### 💡 답변")
        st.markdown(result['response'])

def render_retrieval_test_mode(chat_manager, vectorstore):
    """검색 테스트 모드 렌더링"""
    st.header("🔍 검색 테스트")
    
    test_query = st.text_area("테스트 쿼리", placeholder="검색하고 싶은 내용을 입력하세요")
    k = st.slider("검색할 문서 수", 1, 20, 5)
    
    if test_query:
        with st.spinner("검색 중..."):
            results = chat_manager.test_retrieval(
                query=test_query,
                vectorstore=vectorstore,
                k=k
            )
        
        st.markdown("### 📋 검색 결과")
        for i, result in enumerate(results):
            with st.expander(f"결과 {i+1} (유사도: {result['score']:.3f})"):
                st.markdown("**내용:**")
                st.markdown(result['content'])
                
                st.markdown("**메타데이터:**")
                metadata_df = pd.DataFrame([result['metadata']]).T
                metadata_df.columns = ['값']
                st.dataframe(metadata_df)

def render_vectorstore_info_mode(vectorstore_manager, vectorstore):
    """벡터스토어 정보 모드 렌더링"""
    st.header("📊 벡터스토어 정보")
    
    info = vectorstore_manager.get_vectorstore_info(vectorstore)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("총 문서 수", info.get('total_documents', 0))
    
    with col2:
        st.metric("임베딩 모델", info.get('embedding_model', 'Unknown'))
    
    with col3:
        st.metric("디바이스", info.get('device', 'Unknown'))

if __name__ == "__main__":
    main() 