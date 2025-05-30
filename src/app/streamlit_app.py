import streamlit as st
import json
import os
from pathlib import Path
from typing import List, Dict
import openai
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from google.colab import drive
import pandas as pd

# Import from the new directory structure
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

# Google Drive 마운트
drive.mount('/content/drive')

# 환경 변수에서 경로 가져오기
vectorstore_path = os.environ.get('VECTORSTORE_PATH', '/content/drive/MyDrive/vectorstore')
json_folder_path = os.environ.get('JSON_FOLDER_PATH', '/content/drive/MyDrive/prep_json')
prompt_templates_path = os.environ.get('PROMPT_TEMPLATES_PATH', '/content/drive/MyDrive/prompt_templates')

# OpenAI API 키 설정 및 HuggingFace 토큰 설정
openai.api_key = st.secrets["openai_api_key"]
hf_token = st.secrets["hf_token"]

# 세션 상태 초기화
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'retrieval_params' not in st.session_state:
    st.session_state.retrieval_params = {
        'k': 3,
        'threshold': 0.7
    }

def load_vectorstore():
    """벡터 스토어 로드"""
    if not Path(vectorstore_path).exists():
        st.error(f"벡터 스토어 경로({vectorstore_path})가 존재하지 않습니다.")
        return None
    
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            huggingface_api_token=hf_token
        )
        vectorstore = FAISS.load_local(vectorstore_path, embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"벡터 스토어 로드 중 오류가 발생했습니다: {str(e)}")
        return None

def main():
    st.title("ChipChat - 데이터시트 챗봇")
    
    # 사이드바 - 모드 선택
    mode = st.sidebar.selectbox(
        "Select Mode",
        ["Retrieval Test", "Chat"]
    )
    
    # 앱 시작 시 벡터 스토어 로드
    if st.session_state.vectorstore is None:
        with st.spinner("벡터 스토어를 로드하는 중..."):
            st.session_state.vectorstore = load_vectorstore()
        
        if st.session_state.vectorstore is None:
            st.error("벡터 스토어를 로드할 수 없습니다. prep 모듈을 먼저 실행하여 벡터 스토어를 생성해주세요.")
            return
        else:
            st.success("벡터 스토어가 성공적으로 로드되었습니다.")
    
    if mode == "Retrieval Test":
        st.header("Retrieval Test")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Parameters")
            k = st.slider("Number of results (k)", 1, 10, st.session_state.retrieval_params['k'])
            threshold = st.slider("Similarity threshold", 0.0, 1.0, st.session_state.retrieval_params['threshold'])
            
            if st.button("Update Parameters"):
                st.session_state.retrieval_params = {'k': k, 'threshold': threshold}
                st.success("Parameters updated!")
        
        with col2:
            st.subheader("Test Query")
            query = st.text_area("Enter your test query")
            
            if query and st.session_state.vectorstore:
                results = st.session_state.vectorstore.similarity_search_with_score(
                    query,
                    k=k
                )
                
                for doc, score in results:
                    if score >= threshold:
                        st.write(f"Score: {score:.2f}")
                        st.write(doc.page_content)
                        st.write("---")
    
    else:  # Chat mode
        st.header("Chat")
        
        # System prompt templates
        templates_folder = Path(prompt_templates_path)
        os.makedirs(templates_folder, exist_ok=True)
        
        # Load or create default template
        default_template = {
            "pre": "You are a helpful assistant that answers questions about electronic components based on their datasheets.",
            "post": "Please provide a clear and concise answer based on the retrieved information."
        }
        
        template_file = templates_folder / "default_template.json"
        if not template_file.exists():
            with open(template_file, 'w') as f:
                json.dump(default_template, f, indent=2)
        
        # Load templates
        templates = [f.stem for f in templates_folder.glob("*.json")]
        selected_template = st.selectbox("Select prompt template", templates)
        
        with open(templates_folder / f"{selected_template}.json") as f:
            template = json.load(f)
        
        # Customize prompts
        pre_prompt = st.text_area("System prompt (pre)", template["pre"])
        post_prompt = st.text_area("System prompt (post)", template["post"])
        
        # Chat interface
        user_input = st.text_input("Your question")
        
        if user_input and st.session_state.vectorstore:
            # Retrieve relevant documents
            docs = st.session_state.vectorstore.similarity_search(
                user_input,
                k=st.session_state.retrieval_params['k']
            )
            
            # Construct prompt
            context = "\n".join([doc.page_content for doc in docs])
            prompt = f"{pre_prompt}\n\nQuestion: {user_input}\n\nContext: {context}\n\n{post_prompt}"
            
            # Get response from LLM
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )
            
            st.write("Answer:", response.choices[0].message.content)

if __name__ == "__main__":
    main() 