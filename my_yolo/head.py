import torch
import torch.nn as nn
from .backbone import ConvBlock


class YOLOHead(nn.Module):
    """
    YOLOv3 Detection Head
    - 각 스케일의 feature map에서 bounding box와 class를 예측하는 레이어

    예측 구조:
    - 각 grid cell마다 3개의 anchor box 사용
    - 각 anchor box는 다음을 예측:
      * (tx, ty): bounding box의 중심 좌표 offset
      * (tw, th): bounding box의 너비/높이
      * objectness: 객체가 있을 확률
      * class probabilities: 각 클래스별 확률 (25개)

    출력 채널 수 계산:
    - num_anchors * (5 + num_classes)
    - 3 * (5 + 25) = 90 channels
    """

    def __init__(self, in_channels, num_classes=25):
        """
        Args:
            in_channels: 입력 feature map의 채널 수
                - 13x13 스케일: 512
                - 26x26 스케일: 256
                - 52x52 스케일: 128
            num_classes: 클래스 개수 (기본값 25)
        """
        super(YOLOHead, self).__init__()
        # 각 스케일마다 3개의 anchor box
        self.num_classes = num_classes
        self.num_anchors = 3  

        # 최종 예측 전 feature 처리
        # 3x3 conv로 spatial information 추가 학습
        self.conv = ConvBlock(
            in_channels=in_channels,
            out_channels=in_channels * 2,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # 최종 detection layer
        # 1x1 conv로 각 anchor의 예측값 출력
        self.detection = nn.Conv2d(
            in_channels=in_channels * 2,
            out_channels=self.num_anchors * (5 + num_classes),
            kernel_size=1,
            stride=1,
            padding=0
        )

    def forward(self, x):
        """
        Forward pass

        Args:
            x: feature map [B, C, H, W]
                - C: in_channels (512, 256, or 128)
                - H, W: grid size (13, 26, or 52)

        Returns:
            prediction: [B, num_anchors, H, W, 5 + num_classes]
                - num_anchors: 3
                - 5: (tx, ty, tw, th, objectness)
                - num_classes: 25
        """
        batch_size = x.size(0)
        grid_size = x.size(2)  # H = W (정사각형 grid)

        # Feature 처리
        x = self.conv(x)  # [B, C*2, H, W]

        # Detection 예측
        prediction = self.detection(x)  # [B, 90, H, W]

        # Reshape: [B, 90, H, W] → [B, 3, H, W, 30]
        # 각 anchor별로 30개 값 (5 + 25)을 분리
        prediction = prediction.view(
            batch_size,
            self.num_anchors,
            5 + self.num_classes,
            grid_size,
            grid_size
        ).permute(0, 1, 3, 4, 2).contiguous()
        # 최종 shape: [B, 3, H, W, 30]

        return prediction


class YOLOv3Head(nn.Module):
    """
    YOLOv3의 전체 Detection Head 모음

    세 개의 서로 다른 스케일에서 detection 수행:
    - Large scale (13x13): 큰 객체 탐지
    - Medium scale (26x26): 중간 크기 객체 탐지
    - Small scale (52x52): 작은 객체 탐지

    각 스케일마다 독립적인 detection head 사용
    """

    def __init__(self, num_classes=25):
        super(YOLOv3Head, self).__init__()

        # 큰 객체 탐지용 head (13x13 grid)
        # 입력: 512 channels
        self.head_large = YOLOHead(in_channels=512, num_classes=num_classes)

        # 중간 객체 탐지용 head (26x26 grid)
        # 입력: 256 channels
        self.head_medium = YOLOHead(in_channels=256, num_classes=num_classes)

        # 작은 객체 탐지용 head (52x52 grid)
        # 입력: 128 channels
        self.head_small = YOLOHead(in_channels=128, num_classes=num_classes)

    def forward(self, out_large, out_medium, out_small):
        """
        Forward pass

        Args:
            out_large: [B, 512, 13, 13]  - Neck의 큰 객체용 feature
            out_medium: [B, 256, 26, 26] - Neck의 중간 객체용 feature
            out_small: [B, 128, 52, 52]  - Neck의 작은 객체용 feature

        Returns:
            pred_large: [B, 3, 13, 13, 30]  - 13x13 grid의 예측
            pred_medium: [B, 3, 26, 26, 30] - 26x26 grid의 예측
            pred_small: [B, 3, 52, 52, 30]  - 52x52 grid의 예측

            각 예측의 마지막 차원 (30):
                - [0:2]: tx, ty (중심 좌표 offset)
                - [2:4]: tw, th (너비/높이)
                - [4]: objectness (객체 존재 확률)
                - [5:30]: class probabilities (25개 클래스)
        """
        pred_large = self.head_large(out_large)    # [B, 3, 13, 13, 30]
        pred_medium = self.head_medium(out_medium)  # [B, 3, 26, 26, 30]
        pred_small = self.head_small(out_small)    # [B, 3, 52, 52, 30]

        return pred_large, pred_medium, pred_small
