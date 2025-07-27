# ChipChat v2.0 - AI Agent 기반 데이터시트 챗봇

🤖 **LangGraph 기반 AI 에이전트가 질문을 분석하고 자동으로 최적의 도구를 선택하여 답변하는 스마트 챗봇**

## ✨ 주요 기능

- 🤖 **AI Agent 시스템**: LangGraph 기반 스마트 에이전트
- 🔧 **3가지 Tool 자동 선택**: chipDB 검색, 벡터스토어 검색, PDF 처리  
- 📊 **ChipDB.csv 연동**: 부품 사양 요약 데이터베이스
- 📄 **실시간 PDF 업로드**: 새 데이터시트 자동 처리 및 통합
- 🎯 **다중 LLM 지원**: OpenAI와 Claude 모델 선택 가능
- 🔍 **스마트 검색**: 부품번호, 제조사, 카테고리별 필터링

## 🏗️ 시스템 구조

![시스템 구조](./docs/system_architecture.md)

AI 에이전트가 질문을 분석하여 자동으로 적절한 도구를 선택:
- 📊 **ChipDB 검색**: 부품 목록 및 기본 사양
- 📚 **벡터스토어 검색**: 상세 기술 문서  
- 📄 **PDF 처리**: 새 데이터시트 통합

## 🚀 빠른 시작 (Google Colab 권장)

### 1단계: 데이터 전처리
PDF 데이터시트를 벡터 데이터베이스로 변환합니다.

1. **Google Colab에서 `prep/prep_main.ipynb` 실행**
   - Google Drive 연동
   - PDF 파일 업로드 (`/content/drive/MyDrive/datasheets/`)
   - 자동으로 JSON 및 벡터스토어 생성
   - chipDB.csv 생성

2. **처리 결과**
   - `/content/drive/MyDrive/prep_json/`: 처리된 JSON 파일들
   - `/content/drive/MyDrive/vectorstore/`: FAISS 벡터스토어
   - `/content/drive/MyDrive/prep_json/chipDB.csv`: 부품 사양 요약

### 2단계: AI 챗봇 실행  
전처리된 데이터로 AI 에이전트 챗봇을 사용합니다.

1. **Google Colab에서 `main.ipynb` 실행**
   - Google Drive 자동 연동
   - API 키 설정 (OpenAI 또는 Claude)
   - 벡터스토어 및 chipDB 자동 로드
   - Streamlit 서버 실행

2. **웹 인터페이스 접속**
   - 노트북에서 제공하는 URL 클릭
   - 🤖 AI Agent 모드 선택
   - 질문 입력 또는 PDF 업로드

## 💬 사용 예시

### 부품 목록 검색
```
질문: "전압 변환기 기능을 하는 모든 부품들을 알려줘"
→ chipDB.csv에서 전압 변환기 부품들 자동 검색
```

### 기술적 세부사항
```  
질문: "W25Q32JV의 전기적 특성과 동작 온도 범위는?"
→ 벡터스토어에서 상세 기술 문서 검색
```

### 복합 질문
```
질문: "32Mbit 플래시 메모리 칩을 찾고 각각의 상세 스펙도 알려줘"  
→ chipDB 검색 + 벡터스토어 검색 자동 조합
```

### PDF 업로드
```
새 데이터시트 PDF 업로드
→ 자동으로 prep 파이프라인 처리 → 즉시 검색 가능
```

## 🛠️ 로컬 환경 설정 (선택사항)

```bash
# 저장소 클론
git clone https://github.com/yourusername/chipchat.git
cd chipchat

# 패키지 설치  
pip install -r requirements.txt

# 환경 변수 설정
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"  
export HF_TOKEN="your-token"

# Streamlit 실행
streamlit run src/app/streamlit_app.py
```

## 📂 프로젝트 구조

```
chipchat/
├── prep/                    # 전처리 모듈 (독립)
│   ├── prep_main.ipynb     # 전처리 노트북
│   ├── src/                # 전처리 소스코드
│   └── prep_json/          # 처리된 데이터
│       └── chipDB.csv      # 부품 사양 요약
├── src/                    # 메인 앱 소스코드
│   ├── models/             # AI 모델 및 에이전트
│   │   ├── langgraph_agent.py  # LangGraph 에이전트
│   │   ├── agent_tools.py      # 3가지 도구
│   │   ├── llm_manager.py      # LLM 관리
│   │   └── vectorstore_manager.py
│   ├── app/
│   │   └── streamlit_app.py    # 웹 인터페이스
│   └── config/
├── main.ipynb              # 메인 실행 노트북
└── docs/
    └── system_architecture.md
```

## 🔧 지원되는 LLM 모델

- **OpenAI**: gpt-4o-mini, gpt-4o, gpt-3.5-turbo
- **Claude**: claude-3-sonnet, claude-3-haiku, claude-3-opus

## ⚠️ 주의사항

- **API 키 보안**: API 키는 절대 공개 저장소에 커밋하지 마세요
- **Colab 세션**: Google Colab 세션 시간 제한에 주의하세요  
- **메모리 사용량**: 대용량 PDF 처리 시 메모리 사용량 확인

## 📚 상세 문서

- [전처리 도구 사용법](./prep/README.md)
- [시스템 구조 상세](./docs/system_architecture.md)

## 📄 라이선스

MIT License 