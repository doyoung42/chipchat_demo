# ChipChat - AI Agent 기반 데이터시트 챗봇

🤖 **LangGraph 기반 AI 에이전트가 자동으로 최적의 도구를 선택하여 답변하는 스마트 챗봇**

## ✨ 주요 기능

- 🤖 **AI Agent 시스템**: 질문 분석 후 자동 도구 선택
- 🔧 **3가지 Tool**: ChipDB 검색, 벡터스토어 검색, PDF 처리  
- 📄 **실시간 PDF 업로드**: 새 데이터시트 자동 처리
- 🎯 **다중 LLM 지원**: OpenAI, Claude 모델 선택

## 🚀 빠른 시작

### 📊 1단계: 데이터 준비 (전처리)
**Google Colab에서 바로 실행:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/doyoung42/chipchat_demo/blob/main/prep/prep_main.ipynb)

- PDF 데이터시트를 업로드하고 벡터 데이터베이스로 변환
- **산출물 위치:**
  - **Google Drive:** `/content/drive/MyDrive/prep/prep_json/`, `/content/drive/MyDrive/prep/vectorstore/`
  - **로컬:** `./prep/prep_json/`, `./prep/vectorstore/`

### 💬 2단계: AI 챗봇 실행
**Google Colab에서 바로 실행:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/doyoung42/chipchat_demo/blob/main/main.ipynb)

- 전처리된 데이터로 AI 에이전트 챗봇 사용
- **데이터 참조 경로:**
  - **Google Drive:** `/content/drive/MyDrive/prep_json/`, `/content/drive/MyDrive/vectorstore/`
  - **로컬:** `./prep_json/`, `./vectorstore/`

## 💬 사용 예시

**부품 목록 검색**: "전압 변환기 기능을 하는 부품들을 알려줘" → ChipDB 자동 검색  
**기술적 세부사항**: "W25Q32JV의 전기적 특성은?" → 벡터스토어 검색  
**복합 질문**: "32Mbit 플래시 메모리를 찾고 상세 스펙도 알려줘" → 도구 자동 조합  
**PDF 업로드**: 새 데이터시트 업로드 → 자동 처리 후 즉시 검색 가능

## 🛠️ 로컬 환경 설정

```bash
# 패키지 설치  
pip install -r requirements.txt

# 환경 변수 설정
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"

# Streamlit 실행
streamlit run src/app/streamlit_app.py
```

## 📂 폴더 구조

```
chipchat_demo/
├── prep/                    # 전처리 모듈 (독립)
│   ├── prep_main.ipynb     # 전처리 노트북
│   ├── datasheets/         # PDF 입력 파일
│   ├── prep_json/          # 처리된 JSON 
│   └── vectorstore/        # 생성된 벡터스토어
├── src/                    # 메인 앱 소스코드
│   ├── models/             # AI Agent & LLM
│   └── app/                # Streamlit UI
├── main.ipynb              # 메인 실행 노트북
├── prep_json/              # 메인앱용 JSON (prep 산출물 복사)
├── vectorstore/            # 메인앱용 벡터스토어
└── config.json             # 통합 설정 파일
```

## 📚 더 자세한 정보

- [전처리 사용법](./prep/README.md)
- **지원 LLM**: OpenAI (gpt-4o-mini, gpt-4o), Claude (sonnet, haiku, opus)

