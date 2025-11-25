import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    기본 컨볼루션 블록: Conv2d → BatchNorm2d → LeakyReLU

    YOLOv3의 모든 컨볼루션 레이어는 이 구조를 따름
    - BatchNorm 사용 시 Conv의 bias는 False로 설정 (BN이 bias 역할을 대신함)
    - LeakyReLU의 negative slope는 0.1 사용 (논문 기준)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False  
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class ResidualBlock(nn.Module):
    """
    Darknet의 Residual Block
    - 학습을 돕는 잔차 연결(Residual Connection) 구현
    - conv1(1*1)과 coonv2(3*3)을 사용
    - forward 메서드에서 입력 텐서 + 두 컨볼루션 결과
    - 채널 감소 → 처리 → 채널 복원의 흐름으로 계산량 절감
    - Skip connection으로 gradient flow 개선

    구조:
    1. 1x1 convolution으로 채널을 절반으로 줄임 (차원 축소)
    2. 3x3 convolution으로 원래 채널 수로 복원
    3. Skip connection (입력을 출력에 더함)
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()

        # 1x1 conv: 채널 수를 절반으로 줄임 
        self.conv1 = ConvBlock(
            in_channels=channels,
            out_channels=channels // 2,
            kernel_size=1,
            stride=1,
            padding=0
        )

        # 3x3 conv: 원래 채널 수로 복원
        self.conv2 = ConvBlock(
            in_channels=channels // 2,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

    def forward(self, x):
        # skip connection
        residual = x  

        x = self.conv1(x)
        x = self.conv2(x)

        x = x + residual  
        return x


class Darknet53(nn.Module):
    """
    Darknet-53 백본 네트워크
    - 53개 컨볼루션 레이어로 구성된 네트워크

    네트워크 구조 (논문 Table 1 기준):
    1. Conv 3x3, 32, stride=1           → [416, 416, 32]
    2. Conv 3x3, 64, stride=2           → [208, 208, 64]
       + Residual x1
    3. Conv 3x3, 128, stride=2          → [104, 104, 128]
       + Residual x2
    4. Conv 3x3, 256, stride=2          → [52, 52, 256]  ← route 1 (작은 객체 탐지용)
       + Residual x8
    5. Conv 3x3, 512, stride=2          → [26, 26, 512]  ← route 2 (중간 객체 탐지용)
       + Residual x8
    6. Conv 3x3, 1024, stride=2         → [13, 13, 1024] ← route 3 (큰 객체 탐지용)
       + Residual x4

    출력:
    - route_1: 52x52 feature map (작은 객체 탐지)
    - route_2: 26x26 feature map (중간 객체 탐지)
    - route_3: 13x13 feature map (큰 객체 탐지)
    """
    def __init__(self):
        super(Darknet53, self).__init__()

        # 초기 컨볼루션: 3채널 입력 → 32채널
        self.conv1 = ConvBlock(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # Stage 1: 64 channels, 1x residual block
        # Downsampling: 416x416 → 208x208
        self.conv2 = ConvBlock(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1
        )
        self.residual_block1 = self._make_residual_blocks(64, num_blocks=1)

        # Stage 2: 128 channels, 2x residual blocks
        # Downsampling: 208x208 → 104x104
        self.conv3 = ConvBlock(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=2,
            padding=1
        )
        self.residual_block2 = self._make_residual_blocks(128, num_blocks=2)

        # Stage 3: 256 channels, 8x residual blocks
        # Downsampling: 104x104 → 52x52
        # 이 stage의 출력이 route_1 (작은 객체 탐지용)
        self.conv4 = ConvBlock(
            in_channels=128,
            out_channels=256,
            kernel_size=3,
            stride=2,
            padding=1
        )
        self.residual_block3 = self._make_residual_blocks(256, num_blocks=8)

        # Stage 4: 512 channels, 8x residual blocks
        # Downsampling: 52x52 → 26x26
        # 이 stage의 출력이 route_2 (중간 객체 탐지용)
        self.conv5 = ConvBlock(
            in_channels=256,
            out_channels=512,
            kernel_size=3,
            stride=2,
            padding=1
        )
        self.residual_block4 = self._make_residual_blocks(512, num_blocks=8)

        # Stage 5: 1024 channels, 4x residual blocks
        # Downsampling: 26x26 → 13x13
        # 이 stage의 출력이 route_3 (큰 객체 탐지용)
        self.conv6 = ConvBlock(
            in_channels=512,
            out_channels=1024,
            kernel_size=3,
            stride=2,
            padding=1
        )
        self.residual_block5 = self._make_residual_blocks(1024, num_blocks=4)

    def _make_residual_blocks(self, channels, num_blocks):
        """
        여러 개의 Residual Block을 Sequential로 묶어서 반환

        Args:
            channels: 입출력 채널 수
            num_blocks: 반복할 residual block 개수
        """
        layers = []
        for _ in range(num_blocks):
            layers.append(ResidualBlock(channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: 입력 이미지 텐서 [batch_size, 3, 416, 416]

        Returns:
            route_1: [batch_size, 256, 52, 52]  - 작은 객체용
            route_2: [batch_size, 512, 26, 26]  - 중간 객체용
            route_3: [batch_size, 1024, 13, 13] - 큰 객체용
        """
        # Stage 1
        x = self.conv1(x)         # [B, 32, 416, 416]
        x = self.conv2(x)         # [B, 64, 208, 208]
        x = self.residual_block1(x)

        # Stage 2
        x = self.conv3(x)         # [B, 128, 104, 104]
        x = self.residual_block2(x)

        # Stage 3 - route_1 출력
        x = self.conv4(x)         # [B, 256, 52, 52]
        x = self.residual_block3(x)
        route_1 = x               # 작은 객체 탐지용 feature map

        # Stage 4 - route_2 출력
        x = self.conv5(x)         # [B, 512, 26, 26]
        x = self.residual_block4(x)
        route_2 = x               # 중간 객체 탐지용 feature map

        # Stage 5 - route_3 출력
        x = self.conv6(x)         # [B, 1024, 13, 13]
        x = self.residual_block5(x)
        route_3 = x               # 큰 객체 탐지용 feature map

        return route_1, route_2, route_3
