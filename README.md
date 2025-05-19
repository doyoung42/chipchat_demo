# Datasheet Analyzer

전자 부품 데이터시트를 자동으로 분석하고 주요 사양을 추출하는 도구입니다.

## 주요 기능

- PDF 데이터시트 업로드 및 뷰어
- 주요 사양 자동 추출
- RAG 기반 정보 검색
- 다중 LLM 모델 지원 (GPT-4, Claude Sonnet 3.7)
- 마크다운 형식의 사양 문서 생성

## 설치 방법

1. 저장소 클론
```bash
git clone https://github.com/yourusername/Datasheet_analyzer.git
cd Datasheet_analyzer
```

2. 의존성 설치
```bash
pip install -r requirements.txt
```

## 사용 방법

1. Google Colab에서 `main.ipynb` 실행
2. Streamlit 앱 실행
```bash
streamlit run src/app/streamlit_app.py
```

## 프로젝트 구조

- `src/`: 소스 코드
  - `config/`: 설정 파일
  - `models/`: 임베딩 및 LLM 모델
  - `utils/`: 유틸리티 함수
  - `app/`: Streamlit 애플리케이션
- `notebooks/`: 예제 노트북

## 라이선스

MIT License 