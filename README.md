# Datasheet Analyzer

전자 부품 데이터시트를 자동으로 분석하고 주요 사양을 추출하는 도구입니다.

## 주요 기능

- PDF 데이터시트 업로드 및 분석
- 주요 사양 자동 추출
- RAG 기반 정보 검색
- 다중 LLM 모델 지원 (GPT-4, Claude Sonnet 3.7)
- JSON 형식의 구조화된 출력

## 설치 방법

### 1. 저장소 클론
```bash
git clone https://github.com/yourusername/Datasheet_analyzer.git
cd Datasheet_analyzer
```

### 2. 환경 설정

#### Conda 환경 사용 (권장)
```bash
# Conda 환경 생성 및 활성화
conda env create -f env.yml
conda activate datasheet_analyzer
```

#### pip 사용
```bash
pip install -r requirements.txt
```

## 사용 방법

### Google Colab에서 실행
1. `main.ipynb` 노트북 열기
2. GPU 런타임 설정 (T4 권장)
3. 노트북 실행

### 로컬에서 실행
```bash
# Conda 환경 활성화 (Conda 사용 시)
conda activate datasheet_analyzer

# 애플리케이션 실행
python main.py
```

## 프로젝트 구조

- `src/`: 소스 코드
  - `config/`: 설정 파일
  - `models/`: 임베딩 및 LLM 모델
  - `utils/`: 유틸리티 함수
  - `app/`: Streamlit 애플리케이션
- `notebooks/`: 예제 노트북
- `main.py`: 로컬 실행 스크립트
- `env.yml`: Conda 환경 설정
- `requirements.txt`: pip 의존성

## 라이선스

MIT License 