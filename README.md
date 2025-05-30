# Chipchat

데이터시트를 자동으로 분석하고 질의응답할 수 있는 시스템입니다.

## 프로젝트 구조

이 프로젝트는 두 개의 모듈로 구성되어 있습니다:

1. **PDF 전처리 모듈** (`prep/`)
   - PDF 데이터시트를 분석하여 JSON 형태로 변환
   - Google Colab 환경에서 실행 가능
   - 자세한 내용은 `prep/README.md` 참조

2. **챗봇 모듈** (루트 디렉토리)
   - 전처리된 JSON 데이터를 기반으로 질의응답 수행
   - Streamlit 기반의 웹 인터페이스
   - Google Colab에서 실행 가능

## 사용 방법

### 1단계: PDF 전처리

먼저 `prep` 모듈을 사용하여 PDF 데이터시트를 JSON 형식으로 변환합니다:

1. Google Colab에서 `prep/main.ipynb` 파일을 열어 실행합니다.
2. 안내에 따라 PDF 파일을 업로드하고 전처리를 진행합니다.
3. 처리된 JSON 파일은 Google Drive에 저장됩니다.

### 2단계: 챗봇 사용

전처리된 JSON 파일을 기반으로 챗봇을 사용합니다:

1. Google Colab에서 루트 디렉토리의 `main.ipynb` 파일을 열어 실행합니다.
2. 안내에 따라 Google Drive를 연동하고 API 키를 설정합니다.
3. Streamlit 인터페이스를 통해 데이터시트에 대한 질의응답을 수행합니다.

## 주요 기능

- PDF 데이터시트 업로드 및 분석
- 주요 사양 자동 추출
- RAG 기반 정보 검색
- JSON 형식의 구조화된 출력
- 데이터시트에 대한 자연어 질의응답

## 라이선스

MIT License 