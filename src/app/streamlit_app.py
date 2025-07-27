import streamlit as st
import json
import os
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

@st.cache_resource
def initialize_managers():
    """매니저들 초기화 (캐시됨)"""
    try:
        # Initialize with OpenAI as default
        chat_manager = ChatManager(provider="openai")
        vectorstore_manager = VectorstoreManager()
        return chat_manager, vectorstore_manager, None
    except Exception as e:
        return None, None, str(e)

@st.cache_resource
def load_vectorstore(_vectorstore_manager, vectorstore_path):
    """벡터스토어 로드 (캐시됨)"""
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
    """에이전트 초기화 (캐시됨)"""
    try:
        if not Path(chipdb_path).exists():
            return None, f"chipDB.csv not found at {chipdb_path}"
        
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
    st.markdown("**AI 에이전트가 자동으로 최적의 도구를 선택하여 질문에 답변합니다**")
    
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
    
    # Initialize managers
    chat_manager, vectorstore_manager, init_error = initialize_managers()
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
    if agent_error:
        st.error(f"에이전트 초기화 실패: {agent_error}")
        st.info("chipDB.csv 파일을 먼저 생성해주세요.")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ 설정")
        
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
        
        # Update chat manager if provider changed
        if hasattr(st.session_state, 'current_provider') and st.session_state.current_provider != (provider, model_name):
            try:
                chat_manager.switch_llm_provider(provider, model_name)
                st.session_state.current_provider = (provider, model_name)
                st.success(f"✅ {provider} ({model_name})로 변경되었습니다")
            except Exception as e:
                st.error(f"LLM 제공자 변경 실패: {str(e)}")
        else:
            st.session_state.current_provider = (provider, model_name)
        
        # Mode selection
        mode = st.selectbox(
            "모드 선택",
            ["🤖 AI Agent", "💬 Chat", "🔍 Retrieval Test", "📊 Vectorstore Info"]
        )
        
        # Show agent info if agent mode selected
        if mode == "🤖 AI Agent" and agent:
            with st.expander("🤖 에이전트 정보"):
                st.markdown("**스마트 질문 분류**")
                st.markdown("• 부품 목록 질문")
                st.markdown("• 기술적 세부사항 질문") 
                st.markdown("• PDF 업로드 요청")
                st.markdown("• 복합 질문")
        
        # Retrieval parameters (for non-agent modes)
        if mode != "🤖 AI Agent":
            st.subheader("검색 설정")
            k = st.slider("검색할 문서 수", 1, 20, 5)
            
            if mode == "🔍 Retrieval Test":
                threshold = st.slider("유사도 임계값", 0.0, 1.0, 0.7, 0.05)
            
            # Filters
            st.subheader("필터 설정")
            available_filters = chat_manager.get_available_filters(vectorstore)
            
            selected_filters = {}
            for filter_key, filter_values in available_filters.items():
                if filter_key in ['maker_pn', 'category', 'manufacturer', 'grade']:
                    selected_value = st.selectbox(
                        f"{filter_key}",
                        ["전체"] + filter_values,
                        key=f"filter_{filter_key}"
                    )
                    if selected_value != "전체":
                        selected_filters[filter_key] = selected_value
    
    # Main content based on mode
    if mode == "🤖 AI Agent":
        if agent:
            render_agent_mode(agent)
        else:
            st.error("에이전트를 사용할 수 없습니다. chipDB.csv를 확인해주세요.")
    elif mode == "💬 Chat":
        render_chat_mode(chat_manager, vectorstore, k, selected_filters)
    elif mode == "🔍 Retrieval Test":
        render_retrieval_test_mode(chat_manager, vectorstore, k, threshold, selected_filters)
    elif mode == "📊 Vectorstore Info":
        render_vectorstore_info_mode(vectorstore_manager, vectorstore)

def render_agent_mode(agent):
    """AI 에이전트 모드 렌더링"""
    st.header("🤖 AI Agent Mode")
    st.markdown("**스마트 에이전트가 질문을 분석하고 자동으로 최적의 도구를 선택합니다**")
    
    # Agent info
    with st.expander("📋 에이전트 기능 설명"):
        st.markdown(agent.get_agent_info())
    
    # Query examples
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🔍 부품 검색 예시:**")
        if st.button("전압 변환기 찾기", key="example1"):
            st.session_state.example_query = "전압 변환기 기능을 하는 모든 부품들을 알려줘"
        if st.button("로직 게이트 찾기", key="example2"):
            st.session_state.example_query = "로직 게이트 기능 부품들 리스트업해줘"
    
    with col2:
        st.markdown("**📚 기술 정보 예시:**")
        if st.button("W25Q32JV 특성", key="example3"):
            st.session_state.example_query = "W25Q32JV의 전기적 특성은?"
        if st.button("LM324 스펙", key="example4"):
            st.session_state.example_query = "LM324의 동작 전압과 온도 범위는?"
    
    with col3:
        st.markdown("**🔄 복합 질문 예시:**")
        if st.button("메모리 칩 추천", key="example5"):
            st.session_state.example_query = "32Mbit 플래시 메모리 칩을 찾고 상세 스펙도 알려줘"
        if st.button("파워 컨버터 비교", key="example6"):
            st.session_state.example_query = "3.3V 파워 컨버터들을 찾고 각각의 특징을 비교해줘"
    
    # File upload for PDF processing
    st.markdown("### 📄 새 데이터시트 업로드")
    uploaded_file = st.file_uploader(
        "PDF 데이터시트를 업로드하세요",
        type=['pdf'],
        help="새로운 부품의 데이터시트를 업로드하면 자동으로 처리되어 데이터베이스에 추가됩니다"
    )
    
    # Main query input
    st.markdown("### 💬 질문하기")
    
    # Use example query if selected
    default_query = st.session_state.get('example_query', '')
    if default_query:
        del st.session_state.example_query
    
    user_query = st.text_area(
        "질문을 입력하세요",
        value=default_query,
        placeholder="예: 전압 변환기 기능을 하는 모든 부품들을 알려줘",
        height=100
    )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        process_query = st.button("🚀 에이전트 실행", type="primary")
    with col2:
        show_process = st.checkbox("처리 과정 표시", value=False)
    
    # Process query or file upload
    if process_query and (user_query or uploaded_file):
        with st.spinner("🤖 에이전트가 처리 중..."):
            if show_process:
                st.markdown("**🔄 처리 단계:**")
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("1/4 질문 분석 중...")
                progress_bar.progress(25)
                
                status_text.text("2/4 도구 선택 중...")
                progress_bar.progress(50)
                
                status_text.text("3/4 데이터 검색 중...")
                progress_bar.progress(75)
                
                status_text.text("4/4 응답 생성 중...")
                progress_bar.progress(100)
            
            # Process through agent
            response = agent.process_query(user_query, uploaded_file)
            
            if show_process:
                status_text.text("✅ 완료!")
        
        # Display response
        st.markdown("### 🤖 에이전트 응답")
        st.markdown(response)
        
        # Add to chat history
        if 'agent_history' not in st.session_state:
            st.session_state.agent_history = []
        
        st.session_state.agent_history.append({
            'query': user_query or f"파일 업로드: {uploaded_file.name if uploaded_file else ''}",
            'response': response
        })
    
    # Show chat history
    if 'agent_history' in st.session_state and st.session_state.agent_history:
        st.markdown("### 💭 대화 기록")
        for i, entry in enumerate(reversed(st.session_state.agent_history[-5:]), 1):
            with st.expander(f"질문 {i}: {entry['query'][:50]}..."):
                st.markdown(f"**질문:** {entry['query']}")
                st.markdown(f"**응답:** {entry['response']}")

def render_chat_mode(chat_manager, vectorstore, k, filters):
    """채팅 모드 렌더링"""
    st.header("💬 채팅")
    
    # Prompt template management
    templates_folder = Path(st.session_state.paths['prompt_templates_path'])
    templates_folder.mkdir(parents=True, exist_ok=True)
    
    # Load or create default template
    default_template = {
        "pre": "당신은 전자 부품 데이터시트에 대해 응답하는 전문 도우미입니다. 제공된 컨텍스트 정보를 기반으로 질문에 정확하고 상세하게 답변하세요.",
        "post": "검색된 정보를 바탕으로 명확하고 간결하게 답변해주세요. 정보가 불충분하다면 그 점을 명시하세요."
    }
    
    template_file = templates_folder / "default_template.json"
    if not template_file.exists():
        chat_manager.save_prompt_template(default_template, str(template_file))
    
    # Template selection
    template_files = list(templates_folder.glob("*.json"))
    template_names = [f.stem for f in template_files]
    
    if template_names:
        selected_template = st.selectbox("프롬프트 템플릿", template_names)
        template = chat_manager.load_prompt_template(str(templates_folder / f"{selected_template}.json"))
    else:
        template = default_template
    
    # Prompt customization
    with st.expander("🛠️ 프롬프트 커스터마이징"):
        pre_prompt = st.text_area("시스템 프롬프트 (앞부분)", template["pre"], height=100)
        post_prompt = st.text_area("시스템 프롬프트 (뒷부분)", template["post"], height=100)
    
    # Chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        user_input = st.text_input("질문을 입력하세요", placeholder="예: W25Q32JV의 전기적 특성은?")
    
    with col2:
        include_metadata = st.checkbox("메타데이터 포함", value=False)
    
    if user_input:
        with st.spinner("응답 생성 중..."):
            result = chat_manager.get_chat_response(
                query=user_input,
                vectorstore=vectorstore,
                pre_prompt=pre_prompt,
                post_prompt=post_prompt,
                k=k,
                filters=filters if filters else None,
                include_metadata=include_metadata
            )
        
        # Display response
        st.markdown("### 💡 답변")
        st.markdown(result['response'])
        
        # Display metadata if requested
        if include_metadata and result.get('source_metadata'):
            st.markdown("### 📚 참조된 소스")
            for i, metadata in enumerate(result['source_metadata']):
                with st.expander(f"소스 {i+1}: {metadata['component']}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**파일명**: {metadata['source']}")
                        st.write(f"**제조사**: {metadata['manufacturer']}")
                        st.write(f"**카테고리**: {metadata['category']}")
                    with col2:
                        st.write(f"**부품번호**: {metadata['maker_pn']}")
                        st.write(f"**Part Number**: {metadata['part_number']}")
        
        st.info(f"📊 총 {result['sources_found']}개의 관련 문서를 찾았습니다.")

def render_retrieval_test_mode(chat_manager, vectorstore, k, threshold, filters):
    """검색 테스트 모드 렌더링"""
    st.header("🔍 검색 테스트")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        test_query = st.text_area("테스트 쿼리", placeholder="검색하고 싶은 내용을 입력하세요")
    
    with col2:
        st.metric("검색 문서 수", k)
        st.metric("유사도 임계값", f"{threshold:.2f}")
        if filters:
            st.write("**활성 필터:**")
            for key, value in filters.items():
                st.write(f"• {key}: {value}")
    
    if test_query:
        with st.spinner("검색 중..."):
            results = chat_manager.test_retrieval(
                query=test_query,
                vectorstore=vectorstore,
                k=k,
                threshold=threshold,
                filters=filters if filters else None
            )
        
        st.markdown("### 📋 검색 결과")
        
        for i, result in enumerate(results):
            score_color = "🟢" if result['passes_threshold'] else "🟡"
            
            with st.expander(f"{score_color} 결과 {i+1} (유사도: {result['score']:.3f})"):
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
    
    # ChipDB info if available
    if Path(st.session_state.paths['chipdb_path']).exists():
        try:
            chipdb = pd.read_csv(st.session_state.paths['chipdb_path'])
            st.markdown("### 📊 ChipDB 정보")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("총 부품 수", len(chipdb))
            with col2:
                st.metric("제조사 수", chipdb['maker_pn'].nunique())
            with col3:
                st.metric("등급 종류", chipdb['grade'].nunique())
            
            # Show sample data
            st.markdown("### 🔍 ChipDB 샘플 데이터")
            st.dataframe(chipdb.head(10))
            
        except Exception as e:
            st.error(f"ChipDB 로드 실패: {str(e)}")
    
    # Available metadata keys
    if 'available_metadata_keys' in info:
        st.markdown("### 🏷️ 사용 가능한 메타데이터 키")
        cols = st.columns(3)
        for i, key in enumerate(info['available_metadata_keys']):
            with cols[i % 3]:
                st.code(key)

if __name__ == "__main__":
    main() 