import streamlit as st
import json
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

# Import from the new directory structure
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

# Page configuration
st.set_page_config(
    page_title="ChipChat - 데이터시트 챗봇",
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

@st.cache_data
def load_chipdb_only(chipdb_path):
    """ChipDB.csv만 로드 (경량 모드용)"""
    try:
        if Path(chipdb_path).exists():
            df = pd.read_csv(chipdb_path)
            return df, None
        else:
            return None, f"ChipDB.csv not found at {chipdb_path}"
    except Exception as e:
        return None, str(e)

@st.cache_data
def load_json_data_only(json_folder_path):
    """JSON 데이터만 로드 (경량 모드용)"""
    try:
        json_files = list(Path(json_folder_path).glob("*.json"))
        json_data = []
        
        for f in json_files:
            try:
                data = json.loads(f.read_text(encoding='utf-8'))
                json_data.append(data)
            except Exception as e:
                continue
        
        return json_data, None
    except Exception as e:
        return [], str(e)

def search_chipdb_simple(df, query, max_results=20):
    """ChipDB에서 간단한 텍스트 검색"""
    if df is None or df.empty:
        return []
    
    query_lower = query.lower()
    
    # 여러 컬럼에서 검색
    mask = (
        df['spec'].str.lower().str.contains(query_lower, na=False) |
        df['maker_pn'].str.lower().str.contains(query_lower, na=False) |
        df['part number'].str.lower().str.contains(query_lower, na=False)
    )
    
    matches = df[mask].head(max_results)
    return matches.to_dict('records')

def search_json_simple(json_data, query):
    """JSON 데이터에서 간단한 텍스트 검색"""
    if not json_data:
        return []
    
    query_lower = query.lower()
    results = []
    
    for data in json_data:
        filename = data.get('filename', 'unknown.pdf')
        
        # Category chunks에서 검색
        category_chunks = data.get('category_chunks', {})
        for category, chunks in category_chunks.items():
            for i, chunk in enumerate(chunks):
                if isinstance(chunk, dict):
                    content = chunk.get('content', '')
                else:
                    content = chunk
                
                if query_lower in content.lower():
                    results.append({
                        'content': content[:300] + "..." if len(content) > 300 else content,
                        'source': filename,
                        'category': category,
                        'component': data.get('metadata', {}).get('component_name', 'Unknown'),
                        'manufacturer': data.get('metadata', {}).get('manufacturer', 'Unknown'),
                        'score': content.lower().count(query_lower)
                    })
    
    # 점수순으로 정렬
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:10]

# 기존 함수들 (전체 모드용)
@st.cache_resource
def initialize_managers_full():
    """매니저들 초기화 (전체 모드용, 시간 측정)"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.info("🔄 1/3 ChatManager 초기화 중...")
        progress_bar.progress(10)
        
        start_time = time.time()
        from src.models.chat_manager import ChatManager
        chat_manager = ChatManager(provider="openai")
        elapsed = time.time() - start_time
        
        status_text.success(f"✅ 1/3 ChatManager 초기화 완료 ({elapsed:.1f}초)")
        progress_bar.progress(30)
        time.sleep(0.5)
        
        status_text.info("🔄 2/3 VectorstoreManager 초기화 중... (임베딩 모델 다운로드, 시간이 오래 걸릴 수 있습니다)")
        
        start_time = time.time()
        from src.models.vectorstore_manager import VectorstoreManager
        vectorstore_manager = VectorstoreManager()
        elapsed = time.time() - start_time
        
        status_text.success(f"✅ 2/3 VectorstoreManager 초기화 완료 ({elapsed:.1f}초)")
        progress_bar.progress(90)
        time.sleep(0.5)
        
        status_text.success("✅ 3/3 모든 매니저 초기화 완료!")
        progress_bar.progress(100)
        time.sleep(1)
        
        # UI 정리
        progress_bar.empty()
        status_text.empty()
        
        return chat_manager, vectorstore_manager, None
        
    except Exception as e:
        progress_bar.empty()
        status_text.error(f"❌ 매니저 초기화 실패: {str(e)}")
        return None, None, str(e)

@st.cache_resource
def load_vectorstore(_vectorstore_manager, vectorstore_path):
    """벡터스토어 로드 (전체 모드용)"""
    try:
        if Path(vectorstore_path).exists():
            return _vectorstore_manager.load_vectorstore(vectorstore_path), None
        else:
            # Try loading from JSON files and creating vectorstore
            json_folder = st.session_state.paths['json_folder_path']
            if Path(json_folder).exists():
                json_data = _vectorstore_manager.load_json_files(json_folder)
                if json_data:
                    vectorstore = _vectorstore_manager.create_vectorstore(json_data)
                    # Save for future use
                    Path(vectorstore_path).parent.mkdir(parents=True, exist_ok=True)
                    _vectorstore_manager.save_vectorstore(vectorstore, vectorstore_path)
                    return vectorstore, None
            
            return None, "벡터스토어나 JSON 파일을 찾을 수 없습니다."
    except Exception as e:
        return None, str(e)

@st.cache_resource
def initialize_agent(_chat_manager, _vectorstore_manager, _vectorstore, chipdb_path):
    """에이전트 초기화 (전체 모드용)"""
    try:
        if not Path(chipdb_path).exists():
            return None, f"chipDB.csv not found at {chipdb_path}"
        
        from src.models.langgraph_agent import ChipChatAgent
        agent = ChipChatAgent(
            csv_path=chipdb_path,
            vectorstore_manager=_vectorstore_manager,
            vectorstore=_vectorstore,
            llm_manager=_chat_manager.llm_manager
        )
        return agent, None
    except Exception as e:
        return None, str(e)

def main():
    st.title("💬 ChipChat - 데이터시트 챗봇")
    
    # Initialize session state
    if 'paths' not in st.session_state:
        st.session_state.paths = setup_paths()
    
    if 'api_keys' not in st.session_state:
        st.session_state.api_keys = load_api_keys()
    
    # 모드 선택을 제일 먼저
    st.markdown("### 🎯 모드 선택")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("⚡ 경량 모드 (빠른 시작)", type="primary", use_container_width=True):
            st.session_state.app_mode = "lightweight"
            st.rerun()
    
    with col2:
        if st.button("🤖 전체 모드 (AI 기능 포함)", use_container_width=True):
            st.session_state.app_mode = "full"
            st.rerun()
    
    # 모드별 설명
    if 'app_mode' not in st.session_state:
        st.info("👆 위에서 사용하고 싶은 모드를 선택해주세요")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **⚡ 경량 모드**
            - 🚀 빠른 시작 (30초 이내)
            - 📊 ChipDB.csv 검색
            - 📚 JSON 문서 텍스트 검색
            - ❌ AI 임베딩/에이전트 없음
            """)
        
        with col2:
            st.markdown("""
            **🤖 전체 모드**
            - 🧠 AI 임베딩 기반 검색
            - 🤖 LangGraph 에이전트
            - 📄 PDF 업로드 처리
            - ⏱️ 초기 로딩 시간 필요 (1-3분)
            """)
        
        # 시스템 정보
        st.markdown("---")
        st.markdown("### 📊 시스템 상태")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            env = detect_environment()
            st.metric("환경", env.upper())
        
        with col2:
            # ChipDB 확인
            chipdb, chipdb_error = load_chipdb_only(st.session_state.paths['chipdb_path'])
            if chipdb is not None:
                st.metric("ChipDB 부품 수", len(chipdb))
            else:
                st.metric("ChipDB", "❌ 없음")
        
        with col3:
            # JSON 파일 확인
            json_data, json_error = load_json_data_only(st.session_state.paths['json_folder_path'])
            if json_data:
                st.metric("JSON 문서 수", len(json_data))
            else:
                st.metric("JSON 문서", "❌ 없음")
        
        return
    
    # API 키 확인
    api_keys_available = any(st.session_state.api_keys.values())
    if not api_keys_available and st.session_state.app_mode == "full":
        st.error("🚨 전체 모드에는 API 키가 필요합니다!")
        st.markdown("""
        다음 중 하나의 방법으로 API 키를 설정해주세요:
        1. **환경 변수**: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `HF_TOKEN`
        2. **Streamlit Secrets**: `.streamlit/secrets.toml` 파일
        3. **Colab**: 노트북의 3단계에서 API 키 입력
        """)
        if st.button("⚡ 경량 모드로 전환"):
            st.session_state.app_mode = "lightweight"
            st.rerun()
        return
    
    # 선택된 모드에 따라 다른 인터페이스 표시
    if st.session_state.app_mode == "lightweight":
        render_lightweight_mode()
    else:
        render_full_mode()

def render_lightweight_mode():
    """경량 모드 렌더링"""
    st.markdown("### ⚡ 경량 모드 활성화")
    st.info("💡 임베딩 모델 없이 기본 텍스트 검색을 제공합니다. 빠른 로딩과 기본 검색이 가능합니다.")
    
    # 모드 변경 버튼
    if st.button("🤖 전체 모드로 전환"):
        st.session_state.app_mode = "full"
        st.rerun()
    
    # 데이터 로드
    chipdb, chipdb_error = load_chipdb_only(st.session_state.paths['chipdb_path'])
    json_data, json_error = load_json_data_only(st.session_state.paths['json_folder_path'])
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ 경량 모드 설정")
        
        max_chipdb_results = st.slider("ChipDB 최대 결과 수", 5, 50, 20)
        max_doc_results = st.slider("문서 최대 결과 수", 3, 20, 10)
        
        # 상태 표시
        st.subheader("📊 데이터 상태")
        if chipdb is not None:
            st.success(f"✅ ChipDB: {len(chipdb)} 부품")
        else:
            st.error(f"❌ ChipDB: {chipdb_error}")
        
        if json_data:
            st.success(f"✅ JSON: {len(json_data)} 문서")
        else:
            st.error(f"❌ JSON: {json_error or 'No files found'}")
    
    # 메인 검색 인터페이스
    st.header("🔍 검색")
    
    # 예시 버튼들
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**⚡ 부품 검색:**")
        if st.button("전압 변환기", key="ex1"):
            st.session_state.search_query = "voltage converter"
        if st.button("로직 게이트", key="ex2"):
            st.session_state.search_query = "logic gate"
    
    with col2:
        st.markdown("**🔧 특정 부품:**")
        if st.button("LM324", key="ex3"):
            st.session_state.search_query = "LM324"
        if st.button("W25Q32JV", key="ex4"):
            st.session_state.search_query = "W25Q32JV"
    
    with col3:
        st.markdown("**📊 사양 검색:**")
        if st.button("3.3V", key="ex5"):
            st.session_state.search_query = "3.3V"
        if st.button("I2C", key="ex6"):
            st.session_state.search_query = "I2C"
    
    # 검색 입력
    default_query = st.session_state.get('search_query', '')
    if default_query:
        del st.session_state.search_query
    
    user_query = st.text_input(
        "검색어를 입력하세요",
        value=default_query,
        placeholder="예: voltage converter, LM324, 3.3V, I2C 등"
    )
    
    if user_query:
        with st.spinner("🔍 검색 중..."):
            # Search ChipDB
            chipdb_results = []
            if chipdb is not None:
                chipdb_results = search_chipdb_simple(chipdb, user_query, max_chipdb_results)
            
            # Search JSON data
            json_results = []
            if json_data:
                json_results = search_json_simple(json_data, user_query)
        
        # Display results
        st.markdown("### 🎯 검색 결과")
        
        response_parts = []
        
        if chipdb_results:
            response_parts.append(f"📊 **{len(chipdb_results)}개의 관련 부품:**")
            for result in chipdb_results[:5]:
                response_parts.append(f"• **{result['maker_pn']}** (Part: {result['part number']}, Grade: {result['grade']})")
                response_parts.append(f"  📋 {result['spec']}")
        
        if json_results:
            response_parts.append(f"\n📚 **{len(json_results)}개의 관련 문서:**")
            for result in json_results[:3]:
                response_parts.append(f"🔸 **{result['component']}** ({result['manufacturer']})")
                response_parts.append(f"📁 {result['category']}: {result['content']}")
        
        if response_parts:
            st.markdown("\n".join(response_parts))
        else:
            st.warning(f"❌ '{user_query}'에 대한 정보를 찾을 수 없습니다.")
        
        # 상세 결과를 탭으로 표시
        if chipdb_results or json_results:
            tab1, tab2 = st.tabs(["📊 ChipDB 결과", "📚 문서 결과"])
            
            with tab1:
                if chipdb_results:
                    df_results = pd.DataFrame(chipdb_results)
                    st.dataframe(df_results, use_container_width=True)
                else:
                    st.info("ChipDB에서 결과를 찾을 수 없습니다.")
            
            with tab2:
                if json_results:
                    for i, result in enumerate(json_results):
                        with st.expander(f"📄 {result['component']} - {result['category']}"):
                            st.markdown(f"**내용:** {result['content']}")
                            st.markdown(f"**소스:** {result['source']}")
                            st.markdown(f"**제조사:** {result['manufacturer']}")
                else:
                    st.info("문서에서 결과를 찾을 수 없습니다.")

def render_full_mode():
    """전체 모드 렌더링"""
    st.markdown("### 🤖 전체 모드 활성화")
    st.info("🧠 AI 임베딩 기반 검색과 LangGraph 에이전트를 사용합니다. 초기 로딩 시간이 필요합니다.")
    
    # 모드 변경 버튼
    if st.button("⚡ 경량 모드로 전환"):
        st.session_state.app_mode = "lightweight"
        st.rerun()
    
    # 초기화 상태 확인
    if 'full_mode_initialized' not in st.session_state:
        st.session_state.full_mode_initialized = False
    
    if not st.session_state.full_mode_initialized:
        st.warning("🔄 전체 모드 구성 요소를 초기화해야 합니다.")
        
        if st.button("🚀 전체 모드 초기화 시작", type="primary"):
            with st.spinner("초기화 중..."):
                # Initialize managers
                chat_manager, vectorstore_manager, init_error = initialize_managers_full()
                
                if init_error:
                    st.error(f"매니저 초기화 실패: {init_error}")
                    return
                
                # Load vectorstore
                vectorstore, vs_error = load_vectorstore(vectorstore_manager, st.session_state.paths['vectorstore_path'])
                
                if vs_error:
                    st.error(f"벡터스토어 로드 실패: {vs_error}")
                    st.info("prep 모듈을 먼저 실행하여 데이터시트를 전처리하고 벡터스토어를 생성해주세요.")
                    return
                
                # Initialize agent
                agent, agent_error = initialize_agent(
                    chat_manager, vectorstore_manager, vectorstore, 
                    st.session_state.paths['chipdb_path']
                )
                
                # Store in session state
                st.session_state.chat_manager = chat_manager
                st.session_state.vectorstore_manager = vectorstore_manager
                st.session_state.vectorstore = vectorstore
                st.session_state.agent = agent
                st.session_state.full_mode_initialized = True
                
                st.success("✅ 전체 모드 초기화 완료!")
                st.rerun()
        
        return
    
    # 전체 모드 인터페이스
    chat_manager = st.session_state.chat_manager
    vectorstore_manager = st.session_state.vectorstore_manager
    vectorstore = st.session_state.vectorstore
    agent = st.session_state.agent
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ 전체 모드 설정")
        
        # 시스템 상태
        st.subheader("📊 시스템 상태")
        st.success("✅ 모든 구성 요소 로드됨")
        
        # LLM 설정
        provider = st.selectbox("LLM 제공자", ["openai", "claude"])
        
        if provider == "openai":
            model_options = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
        else:
            model_options = ["claude-3-sonnet-20240229", "claude-3-haiku-20240307"]
        
        model_name = st.selectbox("모델", model_options)
        
        # 모드 선택
        mode = st.selectbox(
            "기능 선택",
            ["🤖 AI Agent", "💬 Chat", "🔍 Retrieval Test", "📊 Info"]
        )
        
        # 재초기화 버튼
        if st.button("🔄 재초기화"):
            for key in ['full_mode_initialized', 'chat_manager', 'vectorstore_manager', 'vectorstore', 'agent']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    
    # 선택된 기능에 따라 인터페이스 표시
    if mode == "🤖 AI Agent":
        if agent:
            st.header("🤖 AI Agent")
            
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
        else:
            st.error("에이전트를 사용할 수 없습니다.")
    
    elif mode == "💬 Chat":
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
    
    elif mode == "🔍 Retrieval Test":
        st.header("🔍 검색 테스트")
        
        test_query = st.text_area("테스트 쿼리")
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
                    st.markdown(result['content'])
    
    elif mode == "📊 Info":
        st.header("📊 시스템 정보")
        
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