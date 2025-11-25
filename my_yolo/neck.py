import torch
import torch.nn as nn
from .backbone import ConvBlock


class YOLONeck(nn.Module):
    """
    YOLOv3의 Neck 부분 - Feature Pyramid Network (FPN)
    - 백본에서 나온 3개의 서로 다른 스케일 feature map을 받음
    - top-down 방식으로 정보를 융합하여 멀티스케일 detection을 위한 feature를 생성

    동작 방식:
    1. 가장 작은 feature map (13x13, 큰 객체용)부터 처리
    2. 이를 upsample하여 중간 feature map (26x26)과 결합
    3. 다시 upsample하여 큰 feature map (52x52, 작은 객체용)과 결합

    각 스케일에서:
    - 5개의 conv 레이어로 feature 정제 (1x1-3x3-1x1-3x3-1x1 패턴)
    - 상위 레벨의 semantic 정보와 하위 레벨의 detail 정보를 결합
    """

    def __init__(self):
        super(YOLONeck, self).__init__()

        # 1. 첫 번째 스케일 처리 (13x13, 가장 깊은 feature, 큰 객체 탐지용)
        # 입력: route_3 [B, 1024, 13, 13]
        # 출력: detection용 feature [B, 512, 13, 13]

        self.conv_set_1 = nn.Sequential(
            ConvBlock(1024, 512, kernel_size=1, stride=1, padding=0),  # 채널 축소
            ConvBlock(512, 1024, kernel_size=3, stride=1, padding=1),  # 공간 정보 처리
            ConvBlock(1024, 512, kernel_size=1, stride=1, padding=0),  # 채널 축소
            ConvBlock(512, 1024, kernel_size=3, stride=1, padding=1),  # 공간 정보 처리
            ConvBlock(1024, 512, kernel_size=1, stride=1, padding=0),  # 최종 채널 축소
        )

        # 두 번째 스케일로 전달하기 위한 처리
        # 채널을 256으로 줄이고 upsample (13x13 → 26x26)
        self.conv_1_to_2 = ConvBlock(512, 256, kernel_size=1, stride=1, padding=0)
        self.upsample_1 = nn.Upsample(scale_factor=2, mode='nearest')

        # 2. 두 번째 스케일 처리 (26x26, 중간 객체 탐지용)
        # 입력: upsample된 feature (256) + route_2 (512) = 768 channels
        # 출력: detection용 feature [B, 256, 26, 26]

        self.conv_set_2 = nn.Sequential(
            ConvBlock(768, 256, kernel_size=1, stride=1, padding=0),   # 768 = 256(upsample) + 512(route_2)
            ConvBlock(256, 512, kernel_size=3, stride=1, padding=1),
            ConvBlock(512, 256, kernel_size=1, stride=1, padding=0),
            ConvBlock(256, 512, kernel_size=3, stride=1, padding=1),
            ConvBlock(512, 256, kernel_size=1, stride=1, padding=0),
        )

        # 세 번째 스케일로 전달하기 위한 처리
        # 채널을 128로 줄이고 upsample (26x26 → 52x52)
        self.conv_2_to_3 = ConvBlock(256, 128, kernel_size=1, stride=1, padding=0)
        self.upsample_2 = nn.Upsample(scale_factor=2, mode='nearest')

        # 3. 세 번째 스케일 처리 (52x52, 작은 객체 탐지용)
        # 입력: upsample된 feature (128) + route_1 (256) = 384 channels
        # 출력: detection용 feature [B, 128, 52, 52]

        self.conv_set_3 = nn.Sequential(
            ConvBlock(384, 128, kernel_size=1, stride=1, padding=0),   # 384 = 128(upsample) + 256(route_1)
            ConvBlock(128, 256, kernel_size=3, stride=1, padding=1),
            ConvBlock(256, 128, kernel_size=1, stride=1, padding=0),
            ConvBlock(128, 256, kernel_size=3, stride=1, padding=1),
            ConvBlock(256, 128, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, route_1, route_2, route_3):
        """
        Forward pass

        Args:
            route_1: [B, 256, 52, 52]   - 백본의 작은 객체용 feature map
            route_2: [B, 512, 26, 26]   - 백본의 중간 객체용 feature map
            route_3: [B, 1024, 13, 13]  - 백본의 큰 객체용 feature map

        Returns:
            out_large: [B, 512, 13, 13]  - 큰 객체 detection용 (13x13 grid)
            out_medium: [B, 256, 26, 26] - 중간 객체 detection용 (26x26 grid)
            out_small: [B, 128, 52, 52]  - 작은 객체 detection용 (52x52 grid)
        """

        # 1. 첫 번째 스케일: 13x13 (큰 객체용)
        # route_3를 5개 conv로 처리
        out_large = self.conv_set_1(route_3)  # [B, 512, 13, 13]

        # 다음 스케일로 전달: 채널 축소 후 upsample
        x = self.conv_1_to_2(out_large)       # [B, 256, 13, 13]
        x = self.upsample_1(x)                # [B, 256, 26, 26]

        # 2. 두 번째 스케일: 26x26 (중간 객체용)
        # upsample된 feature와 route_2를 channel 방향으로 concatenate
        x = torch.cat([x, route_2], dim=1)    # [B, 768, 26, 26] = 256 + 512
        out_medium = self.conv_set_2(x)       # [B, 256, 26, 26]

        # 다음 스케일로 전달: 채널 축소 후 upsample
        x = self.conv_2_to_3(out_medium)      # [B, 128, 26, 26]
        x = self.upsample_2(x)                # [B, 128, 52, 52]

        # 세 번째 스케일: 52x52 (작은 객체용)
        # upsample된 feature와 route_1을 channel 방향으로 concatenate
        x = torch.cat([x, route_1], dim=1)    # [B, 384, 52, 52] = 128 + 256

        # 5개 conv로 처리
        out_small = self.conv_set_3(x)        # [B, 128, 52, 52]

        return out_large, out_medium, out_small
