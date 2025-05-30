import streamlit as st
import json
import os
from pathlib import Path
from typing import List, Dict
import openai
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from google.colab import drive
import pandas as pd

# Google Drive 마운트
drive.mount('/content/drive')

# OpenAI API 키 설정
openai.api_key = st.secrets["openai_api_key"]

# 세션 상태 초기화
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'retrieval_params' not in st.session_state:
    st.session_state.retrieval_params = {
        'k': 3,
        'threshold': 0.7
    }

def load_json_files(folder_path: str) -> List[Dict]:
    """JSON 파일들을 로드하여 리스트로 반환"""
    json_files = Path(folder_path).glob("*.json")
    return [json.loads(f.read_text()) for f in json_files]

def create_vectorstore(json_data: List[Dict]) -> FAISS:
    """JSON 데이터로부터 벡터 스토어 생성"""
    # JSON 데이터를 텍스트로 변환
    texts = []
    for data in json_data:
        text = f"Product: {data.get('product_name', '')}\n"
        text += f"Features: {', '.join(data.get('key_features', []))}\n"
        text += f"Specifications: {json.dumps(data.get('specifications', {}), indent=2)}\n"
        text += f"Applications: {', '.join(data.get('applications', []))}\n"
        text += f"Notes: {data.get('notes', '')}\n"
        texts.append(text)
    
    # 텍스트 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_text("\n".join(texts))
    
    # 벡터 스토어 생성
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts, embeddings)
    
    return vectorstore

def main():
    st.title("ChipChat - 데이터시트 챗봇")
    
    # 사이드바 - 모드 선택
    mode = st.sidebar.selectbox(
        "Select Mode",
        ["Vectorstore Creation", "Retrieval Test", "Chat"]
    )
    
    if mode == "Vectorstore Creation":
        st.header("Vectorstore Creation")
        
        json_folder = "/content/drive/MyDrive/processed_json"
        vectorstore_path = "/content/drive/MyDrive/vectorstore"
        
        if st.button("Create Vectorstore"):
            if not os.path.exists(vectorstore_path):
                json_data = load_json_files(json_folder)
                vectorstore = create_vectorstore(json_data)
                vectorstore.save_local(vectorstore_path)
                st.success("Vectorstore created successfully!")
            else:
                st.info("Vectorstore already exists!")
    
    elif mode == "Retrieval Test":
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
        templates_folder = "/content/drive/MyDrive/prompt_templates"
        os.makedirs(templates_folder, exist_ok=True)
        
        # Load or create default template
        default_template = {
            "pre": "You are a helpful assistant that answers questions about electronic components based on their datasheets.",
            "post": "Please provide a clear and concise answer based on the retrieved information."
        }
        
        template_file = Path(templates_folder) / "default_template.json"
        if not template_file.exists():
            with open(template_file, 'w') as f:
                json.dump(default_template, f, indent=2)
        
        # Load templates
        templates = [f.stem for f in Path(templates_folder).glob("*.json")]
        selected_template = st.selectbox("Select prompt template", templates)
        
        with open(Path(templates_folder) / f"{selected_template}.json") as f:
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