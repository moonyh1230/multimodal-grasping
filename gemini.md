# Gemini 컨텍스트: 로봇 그리핑 및 이미지 분할 프로젝트

## 프로젝트 개요

이 프로젝트는 컴퓨터 비전 기술을 사용하여 이미지 내에서 객체를 분할(Segmentation)하고, 로봇이 해당 객체를 잡을 수 있는 지점(Grasping Point)을 추론하는 딥러닝 모델을 개발합니다.

## 주요 기술 스택

- **언어**: Python
- **프레임워크**: PyTorch (추정)

## 주요 명령어

- **의존성 설치**:
  ```bash
  pip install -r requirements.txt
  ```

- **모델 학습 (Joint Training)**:
  ```bash
  python scripts/train_joint.py
  ```

- **추론 (Inference)**:
  ```bash
  python scripts/inference.py
  ```

- **테스트**:
  (추후 추가)

## 디렉토리 구조

- `data/`: 데이터셋 및 데이터 로더 관련 파일
- `models/`: 모델 아키텍처 (Segmentation 백본, Grasping 헤드 등)
- `scripts/`: 학습(`train_joint.py`), 추론(`inference.py`) 스크립트
- `checkpoints/`: 학습된 모델 가중치 저장 위치
- `utils/`: 손실 함수, 평가지표 등 유틸리티 함수
