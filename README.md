# ChipChat - 데이터시트 기반 챗봇

데이터시트 PDF를 분석하고 질의응답할 수 있는 챗봇 시스템입니다.

## 주요 기능

- PDF 데이터시트 기반 질의응답
- 다중 LLM 지원 (OpenAI, Claude)
- RAG(Retrieval-Augmented Generation) 기반 응답 생성
- Streamlit 기반 웹 인터페이스

## 시작하기

1. 저장소 클론:
```bash
git clone https://github.com/yourusername/chipchat.git
cd chipchat
```

2. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

## 사용 방법

### 1단계: PDF 전처리
먼저 PDF 데이터시트를 처리해야 합니다. 자세한 내용은 [전처리 도구 문서](./prep/README.md)를 참조하세요.

### 2단계: 챗봇 사용
전처리된 데이터를 기반으로 챗봇을 사용합니다:

1. Google Colab에서 `main.ipynb` 파일을 열어 실행합니다.
2. 안내에 따라 Google Drive를 연동하고 API 키를 설정합니다.
3. Streamlit 인터페이스를 통해 데이터시트에 대한 질의응답을 수행합니다.

## 주의사항

- API 키는 절대 공개 저장소에 커밋하지 마세요
- 대용량 PDF 파일 처리 시 메모리 사용량에 주의하세요

## 라이선스

MIT License 