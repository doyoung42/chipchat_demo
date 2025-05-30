# ChipChat - 데이터시트 챗봇

이 모듈은 전처리된 JSON 데이터를 기반으로 데이터시트에 대한 질의응답을 수행하는 Streamlit 기반 챗봇입니다.

## 기능

1. Vectorstore 생성
   - 전처리된 JSON 파일들을 기반으로 FAISS 벡터 스토어 생성
   - Google Drive 연동으로 벡터 스토어 저장 및 로드

2. Retrieval 테스트
   - 검색 파라미터 설정 (k, threshold)
   - 실시간 검색 결과 확인

3. 챗봇 인터페이스
   - 시스템 프롬프트 템플릿 관리
   - 사용자 질문 입력 및 응답 생성
   - 다양한 LLM 모델 지원

## 사용 방법

1. Google Colab에서 `main.ipynb` 파일을 실행하여 Streamlit 서버를 시작합니다.
2. Streamlit 인터페이스에서 다음 순서로 작업을 수행합니다:
   - Vectorstore Creation: JSON 데이터로 벡터 스토어 생성
   - Retrieval Test: 검색 파라미터 최적화
   - Chat: 실제 챗봇 사용

## 의존성

- Python 3.8 이상
- requirements.txt에 명시된 패키지들

## 설치

```bash
pip install -r requirements.txt
```

## 환경 설정

1. OpenAI API 키 설정
2. Google Drive 연동 설정
3. 프롬프트 템플릿 관리 