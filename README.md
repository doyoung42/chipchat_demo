# AutoDataSheet

데이터시트를 자동으로 분석하고 질의응답할 수 있는 시스템입니다.

## 프로젝트 구조

이 프로젝트는 두 개의 독립적인 모듈로 구성되어 있습니다:

1. **PDF 전처리 모듈** (`prep/`)
   - PDF 데이터시트를 분석하여 JSON 형태로 변환
   - Google Colab 환경에서 실행 가능
   - 자세한 내용은 `prep/README.md` 참조

2. **챗봇 모듈** (`chipchat/`)
   - 전처리된 JSON 데이터를 기반으로 질의응답 수행
   - Streamlit 기반의 웹 인터페이스
   - 자세한 내용은 `chipchat/README.md` 참조

각 모듈은 독립적으로 실행 가능하며, Google Drive를 통해 데이터를 주고받습니다.

## 주요 기능

- PDF 데이터시트 업로드 및 분석
- 주요 사양 자동 추출
- RAG 기반 정보 검색
- JSON 형식의 구조화된 출력

## 라이선스

MIT License 