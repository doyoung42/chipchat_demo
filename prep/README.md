# ChipChat 전처리 도구

PDF 데이터시트를 벡터 데이터베이스로 변환하는 전처리 도구입니다.

## 🚀 빠른 시작

**Google Colab에서 바로 실행:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/doyoung42/chipchat_demo/blob/main/prep/prep_main.ipynb)

**노트북 실행 순서:** 각 셀을 순서대로 실행하면 자동으로 처리됩니다.

## 📁 데이터셋 준비

### Google Drive 환경 (Colab 권장)
1. PDF 파일 업로드: `/content/drive/MyDrive/prep/datasheets/`
2. 산출물 저장 위치: 
   - 처리된 JSON: `/content/drive/MyDrive/prep/prep_json/`
   - 벡터스토어: `/content/drive/MyDrive/prep/vectorstore/`

### 로컬 환경
1. PDF 파일 위치: `./prep/datasheets/`
2. 산출물 저장 위치:
   - 처리된 JSON: `./prep/prep_json/`
   - 벡터스토어: `./prep/vectorstore/`

## 🛠️ 환경 설정

### Google Colab
- 노트북에서 자동으로 필요한 라이브러리 설치
- Google Drive 자동 마운트

### 로컬 환경
```bash
pip install -r requirements.txt
```

## 🔧 API 키 설정

노트북 실행 시 다음 API 키 입력:
- **OpenAI API Key** (필수 중 하나)
- **Claude API Key** (필수 중 하나) 
- **HuggingFace Token** (선택)

## 📊 주요 기능

- **PDF 자동 분석**: 유용한 페이지만 필터링
- **카테고리별 분류**: 6가지 기술 카테고리로 자동 분류
- **메타데이터 강화**: 부품번호, 등급, 사양 정보 모든 청크에 자동 추가
- **벡터스토어 생성**: FAISS 기반 고성능 검색 데이터베이스

## ⚠️ 주의사항

- API 키 보안: 공개 저장소에 커밋 금지
- Colab 세션 시간 제한 주의
- 대용량 PDF 처리 시 메모리 사용량 확인
