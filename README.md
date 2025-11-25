# YOLOv3 from Scratch - 어린이 보호구역 위험 탐지

> NIPA 어린이 보호구역 위험 탐지를 위한 YOLOv3 모델 학습 프로젝트
>
> YOLOv3 논문을 기반으로 PyTorch로 처음부터 구현한 Object Detection 프로젝트입니다.

## 🚀 빠른 시작

### 1. 설치
```bash
git clone https://github.com/s0ooo0k/NIPA_YOLOV3_PROJECT.git
cd myyolo
pip install -r requirements.txt
```

### 2. 데이터 준비
```bash
# 1) data/images/와 data/labels/에 데이터 넣기
# 2) data/custom.yaml 수정 (클래스 개수와 이름)
# 3) train.txt, val.txt 자동 생성
python -m my_yolo.prepare_data --config data/custom.yaml
```

### 3. 학습 시작
```bash
python -m my_yolo.train
```

## 📁 프로젝트 구조

```
myyolo/
├── my_yolo/              # YOLOv3 구현 코드
│   ├── backbone.py       # Darknet-53 백본
│   ├── neck.py           # FPN (Feature Pyramid Network)
│   ├── head.py           # Detection Head
│   ├── model.py          # YOLOv3 전체 모델
│   ├── loss.py           # YOLOv3 Loss 함수
│   ├── dataset.py        # Dataset 클래스
│   ├── dataloader.py     # DataLoader 헬퍼
│   ├── prepare_data.py   # 데이터 준비 스크립트
│   └── train.py          # 학습 스크립트
│
├── data/
│   ├── custom.yaml          # 데이터셋 설정 (25 클래스)
│   ├── dataset_template.yaml # 범용 템플릿 (80 클래스)
│   ├── images/              # 이미지 (git에 포함 안 됨)
│   ├── labels/              # 라벨 (git에 포함 안 됨)
│   ├── train.txt            # prepare_data.py가 생성
│   └── val.txt              # prepare_data.py가 생성
│
└── requirements.txt
```

## 📊 데이터셋 형식

### YOLO 포맷 라벨
각 이미지의 라벨 파일 (`.txt`):
```
class_id x_center y_center width height
class_id x_center y_center width height
...
```
- 모든 값은 0~1로 normalized
- x_center, y_center: 이미지 너비/높이 대비 중심 좌표
- width, height: 이미지 너비/높이 대비 박스 크기

### 데이터셋 설정 (custom.yaml)
```yaml
train: ./train.txt
val: ./val.txt
nc: 25  # 클래스 개수
names: ["child", "adult", "bus", ...]  # 클래스 이름
```

## 🎯 학습 파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `--epochs` | 100 | 학습 에폭 수 |
| `--batch-size` | 16 | 배치 크기 |
| `--lr` | 0.001 | Learning rate |
| `--img-size` | 416 | 입력 이미지 크기 |
| `--num-classes` | 25 | 클래스 개수 |
| `--scheduler` | cosine | LR scheduler (cosine/step/none) |
| `--checkpoint-dir` | ./checkpoints | Checkpoint 저장 경로 |

전체 옵션:
```bash
python -m my_yolo.train --help
```

## 🏗️ 모델 아키텍처

### YOLOv3 구조
```
Input (416x416x3)
    ↓
[Darknet-53 Backbone]
    ├─→ 52x52x256  (작은 객체)
    ├─→ 26x26x512  (중간 객체)
    └─→ 13x13x1024 (큰 객체)
    ↓
[FPN Neck]
    ├─→ 52x52x128
    ├─→ 26x26x256
    └─→ 13x13x512
    ↓
[Detection Head]
    ├─→ 52x52x90  (3 anchors × 30)
    ├─→ 26x26x90  (3 anchors × 30)
    └─→ 13x13x90  (3 anchors × 30)
```

- **Backbone**: Darknet-53 (53 conv layers)
- **Neck**: Feature Pyramid Network (FPN)
- **Head**: 3 scales × 3 anchors per scale
- **Loss**: Bounding Box + Objectness + Classification

## 💾 체크포인트

학습 중 자동 저장:
- `best.pt`: 최고 성능 모델
- `last.pt`: 마지막 에폭 모델
- `checkpoint_epoch_N.pt`: N 에폭마다 저장

학습 재개:
```bash
python -m my_yolo.train --resume ./checkpoints/last.pt
```


