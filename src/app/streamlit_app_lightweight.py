import streamlit as st
import json
import os
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

# Page configuration
st.set_page_config(
    page_title="ChipChat Lite - 경량화 버전",
    page_icon="⚡",
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
            'json_folder_path': str(base_path / 'prep_json'),
            'chipdb_path': str(base_path / 'prep_json' / 'chipDB.csv')
        }
    else:
        # 로컬 환경
        base_path = Path.cwd()
        return {
            'json_folder_path': str(base_path / 'prep' / 'prep_json'),
            'chipdb_path': str(base_path / 'prep' / 'prep_json' / 'chipDB.csv')
        }

@st.cache_data
def load_chipdb(chipdb_path):
    """ChipDB.csv 로드 (캐시됨)"""
    try:
        if Path(chipdb_path).exists():
            df = pd.read_csv(chipdb_path)
            return df, None
        else:
            return None, f"ChipDB.csv not found at {chipdb_path}"
    except Exception as e:
        return None, str(e)

@st.cache_data
def load_json_data(json_folder_path):
    """JSON 파일들 로드 (캐시됨)"""
    try:
        json_files = list(Path(json_folder_path).glob("*.json"))
        json_data = []
        
        for f in json_files:
            try:
                data = json.loads(f.read_text(encoding='utf-8'))
                json_data.append(data)
            except Exception as e:
                st.warning(f"Failed to load {f.name}: {str(e)}")
        
        return json_data, None
    except Exception as e:
        return [], str(e)

def search_chipdb(df, query, max_results=20):
    """ChipDB에서 텍스트 검색"""
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

def search_json_data(json_data, query):
    """JSON 데이터에서 텍스트 검색"""
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
                        'content': content[:500] + "..." if len(content) > 500 else content,
                        'source': filename,
                        'category': category,
                        'component': data.get('metadata', {}).get('component_name', 'Unknown'),
                        'manufacturer': data.get('metadata', {}).get('manufacturer', 'Unknown'),
                        'score': content.lower().count(query_lower)  # 단순 점수
                    })
        
        # Page summaries에서 검색
        for summary in data.get('page_summaries', []):
            if summary.get('is_useful', False):
                content = summary.get('content', '')
                if query_lower in content.lower():
                    results.append({
                        'content': content[:500] + "..." if len(content) > 500 else content,
                        'source': filename,
                        'category': 'Page Summary',
                        'component': data.get('metadata', {}).get('component_name', 'Unknown'),
                        'manufacturer': data.get('metadata', {}).get('manufacturer', 'Unknown'),
                        'score': content.lower().count(query_lower)
                    })
    
    # 점수순으로 정렬
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:10]  # 상위 10개만 반환

def simple_llm_response(query, search_results, chipdb_results):
    """LLM API 없이 간단한 응답 생성"""
    response_parts = []
    
    if chipdb_results:
        response_parts.append(f"📊 **{len(chipdb_results)}개의 관련 부품을 찾았습니다:**\n")
        for result in chipdb_results[:5]:  # 상위 5개만 표시
            response_parts.append(f"• **{result['maker_pn']}** (Part: {result['part number']}, Grade: {result['grade']})")
            response_parts.append(f"  📋 {result['spec']}\n")
    
    if search_results:
        response_parts.append(f"\n📚 **{len(search_results)}개의 관련 문서를 찾았습니다:**\n")
        for result in search_results[:3]:  # 상위 3개만 표시
            response_parts.append(f"🔸 **{result['component']}** ({result['manufacturer']})")
            response_parts.append(f"📁 카테고리: {result['category']}")
            response_parts.append(f"📄 내용: {result['content'][:200]}...\n")
    
    if not response_parts:
        return f"❌ '{query}'에 대한 정보를 찾을 수 없습니다. 다른 검색어를 시도해보세요."
    
    return "\n".join(response_parts)

def main():
    st.title("⚡ ChipChat Lite - 경량화 버전")
    st.markdown("**빠른 로딩과 기본 검색 기능을 제공합니다 (AI 임베딩 없음)**")
    
    # 경고 메시지
    st.info("💡 이 버전은 임베딩 모델 없이 단순 텍스트 검색만 제공합니다. 더 정확한 결과를 위해서는 원본 앱을 사용하세요.")
    
    # Initialize session state
    if 'paths' not in st.session_state:
        st.session_state.paths = setup_paths()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ 설정")
        
        # Search settings
        st.subheader("검색 설정")
        max_chipdb_results = st.slider("ChipDB 최대 결과 수", 5, 50, 20)
        max_doc_results = st.slider("문서 최대 결과 수", 3, 20, 10)
        
        # Status display
        st.subheader("📊 상태")
        
        # Check ChipDB
        chipdb, chipdb_error = load_chipdb(st.session_state.paths['chipdb_path'])
        if chipdb is not None:
            st.success(f"✅ ChipDB: {len(chipdb)} 부품")
        else:
            st.error(f"❌ ChipDB: {chipdb_error}")
        
        # Check JSON data
        json_data, json_error = load_json_data(st.session_state.paths['json_folder_path'])
        if json_data:
            st.success(f"✅ JSON: {len(json_data)} 파일")
        else:
            st.error(f"❌ JSON: {json_error or 'No files found'}")
    
    # Main search interface
    st.header("🔍 검색")
    
    # Query examples
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
    
    # Search input
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
                chipdb_results = search_chipdb(chipdb, user_query, max_chipdb_results)
            
            # Search JSON data
            json_results = []
            if json_data:
                json_results = search_json_data(json_data, user_query)
            
            # Generate response
            response = simple_llm_response(user_query, json_results, chipdb_results)
        
        # Display results
        st.markdown("### 🎯 검색 결과")
        st.markdown(response)
        
        # Detailed results in tabs
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
                            col1, col2 = st.columns([2, 1])
                            with col1:
                                st.markdown(f"**내용:**\n{result['content']}")
                            with col2:
                                st.markdown(f"**소스:** {result['source']}")
                                st.markdown(f"**제조사:** {result['manufacturer']}")
                                st.markdown(f"**점수:** {result['score']}")
                else:
                    st.info("문서에서 결과를 찾을 수 없습니다.")
    
    # System status
    st.markdown("---")
    st.markdown("### 📈 시스템 상태")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        env = detect_environment()
        st.metric("환경", env.upper())
    
    with col2:
        if chipdb is not None:
            st.metric("로드된 부품", len(chipdb))
        else:
            st.metric("로드된 부품", "0")
    
    with col3:
        st.metric("로드된 문서", len(json_data) if json_data else 0)
    
    # Help section
    with st.expander("💡 도움말"):
        st.markdown("""
        **경량화 버전 특징:**
        - ⚡ 빠른 로딩 (임베딩 모델 없음)
        - 🔍 기본 텍스트 검색
        - 📊 ChipDB.csv 검색
        - 📚 JSON 문서 검색
        
        **검색 팁:**
        - 정확한 부품 번호 사용 (예: LM324, W25Q32JV)
        - 기능 키워드 사용 (예: voltage, converter, logic)
        - 사양 검색 (예: 3.3V, I2C, SPI)
        
        **한계:**
        - AI 기반 의미론적 검색 없음
        - 자연어 질문 처리 제한적
        - 벡터 기반 유사도 검색 없음
        """)

if __name__ == "__main__":
    main() 