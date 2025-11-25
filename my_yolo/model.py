"""
YOLOv3 전체 모델
논문: YOLOv3: An Incremental Improvement
"""

import torch
import torch.nn as nn
from .backbone import Darknet53
from .neck import YOLONeck
from .head import YOLOv3Head


class YOLOv3(nn.Module):
    """
    YOLOv3 전체 모델

    구조:
    1. Backbone (Darknet-53): 이미지에서 3개 스케일의 feature map 추출
    2. Neck (FPN): 멀티스케일 feature fusion으로 detection용 feature 생성
    3. Head: 각 스케일에서 bounding box와 class 예측

    멀티스케일 detection:
    - 13x13 grid: 큰 객체 탐지 (각 grid cell의 receptive field가 큼)
    - 26x26 grid: 중간 크기 객체 탐지
    - 52x52 grid: 작은 객체 탐지 (각 grid cell의 receptive field가 작음)

    각 grid cell마다 3개의 anchor box를 사용하여 예측 수행
    """

    def __init__(self, num_classes=25):
        """
        Args:
            num_classes: 탐지할 클래스 개수 (기본값 25)
        """
        super(YOLOv3, self).__init__()

        self.num_classes = num_classes

        # Backbone: Darknet-53
        # 입력 이미지에서 3개 스케일의 feature map 추출
        self.backbone = Darknet53()

        # Neck: Feature Pyramid Network
        # 3개 스케일의 feature를 융합하여 detection용 feature 생성
        self.neck = YOLONeck()

        # Head: Detection layers
        # 각 스케일에서 bounding box와 class 예측
        self.head = YOLOv3Head(num_classes=num_classes)

        # 가중치 초기화
        self._initialize_weights()

    def _initialize_weights(self):
        """
        모델 가중치 초기화

        Conv layer:
        - weight: Kaiming Normal 초기화 (He initialization)
        - bias: 0으로 초기화

        BatchNorm layer:
        - weight (gamma): 1로 초기화
        - bias (beta): 0으로 초기화
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Conv2d 가중치 초기화
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                # BatchNorm 초기화
                nn.init.constant_(m.weight, 1)  # gamma = 1
                nn.init.constant_(m.bias, 0)    # beta = 0

    def forward(self, x):
        """
        Forward pass

        Args:
            x: 입력 이미지 [B, 3, H, W]
               - H, W는 32의 배수여야 함 (일반적으로 416x416 또는 608x608)

        Returns:
            predictions: 3개 스케일의 예측 결과를 담은 tuple
                - pred_large: [B, 3, 13, 13, 30]  - 큰 객체 예측
                - pred_medium: [B, 3, 26, 26, 30] - 중간 객체 예측
                - pred_small: [B, 3, 52, 52, 30]  - 작은 객체 예측

            각 예측의 마지막 차원 (30개 값):
                [0:2] - tx, ty: bounding box 중심 좌표의 offset
                [2:4] - tw, th: bounding box의 너비/높이 (anchor 대비)
                [4] - objectness: 해당 위치에 객체가 있을 확률
                [5:30] - class_probs: 25개 클래스 각각에 대한 확률
        """

        # 1. Backbone: Feature Extraction
        # 입력 이미지에서 3개 스케일의 feature map 추출
        route_1, route_2, route_3 = self.backbone(x)
        # route_1: [B, 256, 52, 52]   - 작은 객체용
        # route_2: [B, 512, 26, 26]   - 중간 객체용
        # route_3: [B, 1024, 13, 13]  - 큰 객체용

        # 2. Neck: Feature Pyramid Network
        # 멀티스케일 feature fusion
        out_large, out_medium, out_small = self.neck(route_1, route_2, route_3)
        # out_large: [B, 512, 13, 13]  - 큰 객체 detection용
        # out_medium: [B, 256, 26, 26] - 중간 객체 detection용
        # out_small: [B, 128, 52, 52]  - 작은 객체 detection용

        # 3. Head: Detection
        # 각 스케일에서 bounding box와 class 예측
        pred_large, pred_medium, pred_small = self.head(out_large, out_medium, out_small)
        # pred_large: [B, 3, 13, 13, 30]
        # pred_medium: [B, 3, 26, 26, 30]
        # pred_small: [B, 3, 52, 52, 30]

        return pred_large, pred_medium, pred_small

    def get_num_params(self):
        """
        모델의 전체 파라미터 수 반환

        Returns:
            total_params: 전체 파라미터 수
            trainable_params: 학습 가능한 파라미터 수
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params


# YOLOv3 모델 생성 함수
def create_yolov3(num_classes=25, pretrained=False):
    """
    YOLOv3 모델 생성 헬퍼 함수

    Args:
        num_classes: 탐지할 클래스 개수
        
    Returns:
        model: YOLOv3 모델 인스턴스
    """
    model = YOLOv3(num_classes=num_classes)
    return model
