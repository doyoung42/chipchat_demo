import streamlit as st
import json
import os
import time
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="ChipChat Debug - 로딩 진단",
    page_icon="🔧",
    layout="wide"
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
            st.success("✅ Google Drive 마운트 성공")
        except Exception as e:
            st.error(f"❌ Google Drive 마운트 실패: {str(e)}")
        
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

def test_api_keys():
    """API 키 테스트"""
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

def test_file_access(paths):
    """파일 접근성 테스트"""
    results = {}
    
    # ChipDB.csv 확인
    chipdb_path = Path(paths['chipdb_path'])
    if chipdb_path.exists():
        try:
            df = pd.read_csv(chipdb_path)
            results['chipdb'] = f"✅ {len(df)} rows loaded"
        except Exception as e:
            results['chipdb'] = f"❌ Load failed: {str(e)}"
    else:
        results['chipdb'] = f"❌ File not found: {chipdb_path}"
    
    # Vectorstore 확인
    vs_path = Path(paths['vectorstore_path'])
    if vs_path.exists():
        vs_files = list(vs_path.glob("**/*"))
        results['vectorstore'] = f"✅ {len(vs_files)} files found"
    else:
        results['vectorstore'] = f"❌ Directory not found: {vs_path}"
    
    # JSON 폴더 확인
    json_path = Path(paths['json_folder_path'])
    if json_path.exists():
        json_files = list(json_path.glob("*.json"))
        results['json_files'] = f"✅ {len(json_files)} JSON files found"
    else:
        results['json_files'] = f"❌ Directory not found: {json_path}"
    
    return results

def test_import_components():
    """컴포넌트 import 테스트"""
    results = {}
    
    try:
        import sys
        sys.path.append(str(Path(__file__).parent.parent.parent))
        results['sys_path'] = "✅ Path added"
    except Exception as e:
        results['sys_path'] = f"❌ {str(e)}"
    
    try:
        from src.models.chat_manager import ChatManager
        results['chat_manager'] = "✅ Import successful"
    except Exception as e:
        results['chat_manager'] = f"❌ {str(e)}"
    
    try:
        from src.models.vectorstore_manager import VectorstoreManager
        results['vectorstore_manager'] = "✅ Import successful"
    except Exception as e:
        results['vectorstore_manager'] = f"❌ {str(e)}"
    
    try:
        from src.models.langgraph_agent import ChipChatAgent
        results['langgraph_agent'] = "✅ Import successful"
    except Exception as e:
        results['langgraph_agent'] = f"❌ {str(e)}"
    
    return results

def test_embedding_model():
    """임베딩 모델 로딩 테스트 (가장 의심되는 부분)"""
    try:
        st.info("🔄 임베딩 모델 테스트 시작...")
        start_time = time.time()
        
        # HuggingFace token 설정
        hf_token = os.environ.get('HF_TOKEN', '')
        if hf_token:
            os.environ['HUGGINGFACE_API_KEY'] = hf_token
        
        from langchain_huggingface import HuggingFaceEmbeddings
        
        st.info("⏳ HuggingFaceEmbeddings 초기화 중... (이 단계에서 멈출 가능성이 높음)")
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        elapsed = time.time() - start_time
        st.success(f"✅ 임베딩 모델 로딩 성공! ({elapsed:.1f}초)")
        
        # 간단한 테스트
        st.info("🧪 임베딩 테스트 실행 중...")
        test_embedding = embeddings.embed_query("test")
        st.success(f"✅ 임베딩 테스트 성공! 벡터 차원: {len(test_embedding)}")
        
        return True, f"성공 ({elapsed:.1f}초)"
        
    except Exception as e:
        elapsed = time.time() - start_time
        st.error(f"❌ 임베딩 모델 로딩 실패 ({elapsed:.1f}초): {str(e)}")
        return False, str(e)

def main():
    st.title("🔧 ChipChat 로딩 진단 도구")
    st.markdown("**10분 이상 로딩되는 문제를 진단합니다**")
    
    # 환경 정보
    env = detect_environment()
    st.info(f"🖥️ 감지된 환경: {env.upper()}")
    
    # 단계별 진단
    st.header("📋 단계별 진단")
    
    # 1단계: 경로 설정
    with st.expander("1️⃣ 경로 설정 테스트", expanded=True):
        try:
            paths = setup_paths()
            st.success("✅ 경로 설정 성공")
            for key, path in paths.items():
                st.code(f"{key}: {path}")
        except Exception as e:
            st.error(f"❌ 경로 설정 실패: {str(e)}")
            paths = {}
    
    # 2단계: API 키 확인
    with st.expander("2️⃣ API 키 확인"):
        api_keys = test_api_keys()
        for provider, key in api_keys.items():
            if key:
                st.success(f"✅ {provider}: 설정됨 ({'*' * (len(key)-8) + key[-8:]})")
            else:
                st.warning(f"⚠️ {provider}: 설정되지 않음")
    
    # 3단계: 파일 접근성
    with st.expander("3️⃣ 파일 접근성 테스트"):
        if paths:
            file_results = test_file_access(paths)
            for test_name, result in file_results.items():
                if "✅" in result:
                    st.success(f"{test_name}: {result}")
                else:
                    st.error(f"{test_name}: {result}")
        else:
            st.error("경로 설정이 실패하여 파일 테스트를 건너뜁니다")
    
    # 4단계: Import 테스트
    with st.expander("4️⃣ 컴포넌트 Import 테스트"):
        import_results = test_import_components()
        for component, result in import_results.items():
            if "✅" in result:
                st.success(f"{component}: {result}")
            else:
                st.error(f"{component}: {result}")
    
    # 5단계: 임베딩 모델 테스트 (가장 중요!)
    with st.expander("5️⃣ 임베딩 모델 로딩 테스트 (가장 의심되는 부분)", expanded=True):
        st.warning("⚠️ 이 단계에서 10분 이상 멈출 가능성이 높습니다!")
        
        if st.button("🔥 임베딩 모델 테스트 시작 (위험!)"):
            test_embedding_model()
    
    # 대안 제시
    st.header("🛠️ 문제 해결 방안")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("즉시 시도할 수 있는 방법")
        st.markdown("""
        1. **런타임 재시작**: 런타임 → 세션 재시작
        2. **GPU 사용**: 런타임 → 런타임 유형 변경 → GPU
        3. **메모리 확인**: 좌측 사이드바에서 RAM 사용량 확인
        4. **네트워크 확인**: 다른 웹사이트 접속 테스트
        """)
    
    with col2:
        st.subheader("경량화 버전 실행")
        if st.button("🚀 경량화 버전으로 실행"):
            st.info("경량화 버전을 준비 중입니다...")
            # 여기에 경량화 버전 로직 추가
    
    # 시스템 정보
    st.header("🖥️ 시스템 정보")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        try:
            import psutil
            st.metric("RAM 사용률", f"{psutil.virtual_memory().percent}%")
        except:
            st.metric("RAM 사용률", "측정 불가")
    
    with col2:
        try:
            import torch
            st.metric("CUDA 사용 가능", "Yes" if torch.cuda.is_available() else "No")
        except:
            st.metric("CUDA 사용 가능", "PyTorch 없음")
    
    with col3:
        st.metric("환경", env.upper())

if __name__ == "__main__":
    main() 