# PDF 전처리 모듈

이 모듈은 PDF 데이터시트를 전처리하여 JSON 형태로 변환하는 과정을 수행합니다.

## 기능

1. Google Drive에서 PDF 파일 읽기
2. PDF 페이지별 유용성 판단
3. 유용한 페이지 추출
4. 특징 추출 및 JSON 변환

## 사용 방법

1. Google Colab에서 `main.ipynb` 파일을 엽니다.
2. Google Drive를 마운트합니다.
3. PDF 파일을 Google Drive의 `datasheets` 폴더에 업로드합니다.
4. 노트북의 셀을 순서대로 실행합니다.
5. 처리된 JSON 파일은 Google Drive의 `processed_json` 폴더에 저장됩니다.

## 의존성

- Python 3.8 이상
- requirements.txt에 명시된 패키지들

## 설치

```bash
pip install -r requirements.txt
``` 