import torch
from torch.utils.data import DataLoader
from .dataset import YOLODataset


def create_dataloaders(
    train_path,
    val_path,
    batch_size=16,
    img_size=416,
    num_workers=4,
    augment=True
):
    """
    Train/Val DataLoader 생성

    Args:
        train_path: train.txt 경로
        val_path: val.txt 경로
        batch_size: 배치 크기
        img_size: 입력 이미지 크기
        num_workers: 데이터 로딩 워커 수
        augment: train 데이터에 augmentation 적용 여부

    Returns:
        train_loader: Train DataLoader
        val_loader: Validation DataLoader
    """
# Train Dataset & DataLoader
    train_dataset = YOLODataset(
        img_paths_file=train_path,
        img_size=img_size,
        augment=augment,
        multiscale=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=train_dataset.collate_fn,
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )

    # Validation Dataset & DataLoader
    val_dataset = YOLODataset(
        img_paths_file=val_path,
        img_size=img_size,
        augment=False,
        multiscale=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers, # train과 같은 num_workers 사용
        pin_memory=True,
        collate_fn=val_dataset.collate_fn,
        drop_last=False,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )

    return train_loader, val_loader