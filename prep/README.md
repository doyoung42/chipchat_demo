# ChipChat PDF 전처리 도구

PDF 데이터시트를 분석하여 유용한 정보를 추출하고 벡터 데이터베이스로 변환하는 전처리 도구입니다.

## 주요 기능

- PDF 데이터시트에서 주요 정보 추출
- 다중 LLM 지원 (OpenAI, Claude)
- FAISS 벡터 스토어 생성
- Google Colab 지원

## 설치 방법

1. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

## 사용 방법

### 1. Google Colab에서 실행

1. `prep_main.ipynb` 파일을 Google Colab에서 열기
2. 각 셀을 순서대로 실행:
   - Google Drive 연동
   - 라이브러리 설치
   - API 키 설정 (노트북에서 자동으로 `key.json` 생성)
   - PDF 처리 설정 (노트북에서 자동으로 `param.json` 생성)
   - PDF 파일 처리
   - 벡터 스토어 생성

### 2. 설정 파일 참고

노트북 실행 시 자동으로 생성되는 설정 파일들의 구조:

1. API 키 (`misc/key.json`):
   - `openai_api_key`: OpenAI API 키
   - `anthropic_api_key`: Anthropic API 키
   - `huggingface_token`: HuggingFace 토큰

2. 처리 설정 (`misc/param.json`):
   - `pages_per_chunk`: 한 번에 처리할 페이지 수
   - `chunk_overlap`: 청크 간 중복 페이지 수
   - 출력 형식 설정
   - 폴더 경로 설정

## 출력 파일

처리 과정에서 생성되는 파일들:

1. 중간 결과물:
   - `pre_json_folder/*_filtered.pdf`: 필터링된 PDF
   - `pre_json_folder/*_summaries.json`: 페이지별 요약

2. 최종 결과물:
   - `result_json_folder/*_R1.json`: 최종 처리된 JSON
   - `vectorstore_folder/datasheet_vectors`: FAISS 벡터 스토어

## 주의사항

- API 키는 절대 공개 저장소에 커밋하지 마세요
- 대용량 PDF 파일 처리 시 메모리 사용량에 주의하세요
- Google Colab 세션 시간 제한에 주의하세요
