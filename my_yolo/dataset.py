import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class YOLODataset(Dataset):
    """
    YOLOv3용 Dataset 클래스
    - YOLO 포맷의 라벨 파일을 파싱하고 이미지를 로딩

    라벨 포맷: class_id x_center y_center width height (모두 normalized 0~1)
    """

    def __init__(
        self,
        img_paths_file,
        img_size=416,
        augment=False,
        multiscale=False
    ):
        """
        Args:
            img_paths_file: 이미지 경로가 담긴 파일 (train.txt, val.txt)
            img_size: 입력 이미지 크기 (기본값 416)
            augment: augmentation 적용 여부
            multiscale: multi-scale training 여부
        """
        with open(img_paths_file, 'r') as f:
            self.img_paths = [line.strip() for line in f.readlines()]

        self.img_size = img_size
        self.augment = augment
        self.multiscale = multiscale
        self.min_size = self.img_size - 3 * 32  # 320
        self.max_size = self.img_size + 3 * 32  # 512
        self.batch_count = 0

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        """
        Returns:
            img: 전처리된 이미지 [3, img_size, img_size]
            targets: 라벨 텐서 [num_objects, 5]
                     각 행: [class_id, x_center, y_center, width, height]
                     (x, y, w, h는 normalized 좌표)
        """
        # 이미지 로드
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')

        # 라벨 로드
        label_path = img_path.replace('/images/', '/labels/').replace('.jpg', '.txt')
        targets = None

        if os.path.exists(label_path):
            # YOLO 포맷 라벨 파싱: class x_center y_center width height
            boxes = np.loadtxt(label_path).reshape(-1, 5)
            targets = torch.from_numpy(boxes).float()

        # Augmentation (선택적)
        if self.augment:
            img, targets = self._augment(img, targets)

        # 이미지 전처리
        img, targets = self._preprocess(img, targets)

        return img, targets

    def _preprocess(self, img, targets):
        """
        이미지 리사이즈 및 정규화
        """
        # 원본 크기 저장
        w, h = img.size

        # 비율 유지하며 리사이즈 (letterbox)
        img, pad = self._resize_with_pad(img, self.img_size)

        # 라벨 좌표 조정 (padding 고려)
        if targets is not None and len(targets) > 0:
            # 패딩 적용
            pad_left, pad_top = pad
            targets[:, 1] = (targets[:, 1] * w + pad_left) / self.img_size  # x
            targets[:, 2] = (targets[:, 2] * h + pad_top) / self.img_size   # y
            targets[:, 3] = (targets[:, 3] * w) / self.img_size             # w
            targets[:, 4] = (targets[:, 4] * h) / self.img_size             # h

        # PIL Image -> Tensor 변환 및 정규화
        img = transforms.ToTensor()(img)  # [0, 1]로 자동 정규화

        return img, targets

    def _resize_with_pad(self, img, target_size):
        """
        비율을 유지하면서 리사이즈하고 패딩 추가 (letterbox)

        Returns:
            resized_img: 리사이즈된 이미지
            pad: (pad_left, pad_top) 패딩 크기
        """
        w, h = img.size

        # 비율 계산
        scale = min(target_size / w, target_size / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # 리사이즈
        img_resized = img.resize((new_w, new_h), Image.BILINEAR)

        # 새 이미지 생성 (회색 패딩)
        new_img = Image.new('RGB', (target_size, target_size), (128, 128, 128))

        # 중앙에 배치
        pad_left = (target_size - new_w) // 2
        pad_top = (target_size - new_h) // 2
        new_img.paste(img_resized, (pad_left, pad_top))

        return new_img, (pad_left, pad_top)

    def _augment(self, img, targets):
        """
        Data augmentation
        - Horizontal flip
        """
        # 간단한 horizontal flip만 구현
        if np.random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if targets is not None and len(targets) > 0:
                # x 좌표 flip
                targets[:, 1] = 1 - targets[:, 1]

        return img, targets

    def collate_fn(self, batch):
        """
        배치 생성을 위한 collate function

        서로 다른 개수의 객체를 가진 이미지들을 배치로 묶기 위해
        targets에 batch_idx를 추가

        Args:
            batch: [(img, targets), (img, targets), ...]

        Returns:
            imgs: [batch_size, 3, img_size, img_size]
            targets: [total_objects, 6]
                     각 행: [batch_idx, class_id, x, y, w, h]
        """
        imgs = []
        targets = []

        for batch_idx, (img, target) in enumerate(batch):
            imgs.append(img)

            if target is not None and len(target) > 0:
                # batch_idx 추가
                batch_indices = torch.full((target.size(0), 1), batch_idx)
                target = torch.cat([batch_indices, target], dim=1)
                targets.append(target)

        imgs = torch.stack(imgs, 0)

        if len(targets) > 0:
            targets = torch.cat(targets, 0)
        else:
            targets = torch.zeros((0, 6))

        return imgs, targets


# YOLOv3 Anchor boxes (COCO dataset 기준, 416x416 이미지용)
# 각 스케일마다 3개의 anchor 사용
ANCHORS = [
    # 52x52 grid (작은 객체용)
    [(10, 13), (16, 30), (33, 23)],
    # 26x26 grid (중간 객체용)
    [(30, 61), (62, 45), (59, 119)],
    # 13x13 grid (큰 객체용)
    [(116, 90), (156, 198), (373, 326)]
]


def get_anchors():
    """
    YOLOv3 anchor boxes 반환

    Returns:
        anchors: [3, 3, 2] shape의 numpy array
                 [스케일, anchor_idx, (w, h)]
    """
    return np.array(ANCHORS)
