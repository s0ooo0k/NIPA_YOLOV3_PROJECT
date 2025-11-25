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
        drop_last=True  
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
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=val_dataset.collate_fn,
        drop_last=False
    )

    return train_loader, val_loader


def test_dataloader(data_path, batch_size=4, img_size=416):
    """
    DataLoader 테스트 함수

    Args:
        data_path: 데이터 경로 (train.txt 또는 val.txt)
        batch_size: 배치 크기
        img_size: 이미지 크기
    """
    print(f"\n{'='*60}")
    print(f"Testing DataLoader: {data_path}")
    print(f"{'='*60}\n")

    # Dataset & DataLoader 생성
    dataset = YOLODataset(
        img_paths_file=data_path,
        img_size=img_size,
        augment=False
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=dataset.collate_fn
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Number of batches: {len(dataloader)}\n")

    # 첫 번째 배치 확인
    imgs, targets = next(iter(dataloader))

    print(f"Batch images shape: {imgs.shape}")
    print(f"  - Expected: [batch_size={batch_size}, 3, {img_size}, {img_size}]")
    print(f"\nBatch targets shape: {targets.shape}")
    print(f"  - Format: [total_objects, 6]")
    print(f"  - Columns: [batch_idx, class_id, x_center, y_center, width, height]")

    # 배치별 객체 수 확인
    print(f"\nObjects per image in batch:")
    for i in range(batch_size):
        num_objs = (targets[:, 0] == i).sum().item()
        print(f"  Image {i}: {num_objs} objects")

    # 샘플 타겟 출력
    if len(targets) > 0:
        print(f"\nFirst 5 targets:")
        print(targets[:5])

    print(f"\nImage value range: [{imgs.min():.3f}, {imgs.max():.3f}]")
    print(f"Expected range: [0.0, 1.0]")

    print(f"\n{'='*60}\n")

    return imgs, targets


if __name__ == "__main__":
    import sys
    import os

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    # 데이터 경로
    train_path = "/home/sookidayo/safety-yolo/data/train.txt"
    val_path = "/home/sookidayo/safety-yolo/data/val.txt"

    # Train DataLoader 테스트
    print("\n" + "="*60)
    print("TRAIN DATALOADER TEST")
    print("="*60)
    test_dataloader(train_path, batch_size=4, img_size=416)

    # Val DataLoader 테스트
    print("\n" + "="*60)
    print("VALIDATION DATALOADER TEST")
    print("="*60)
    test_dataloader(val_path, batch_size=4, img_size=416)

    # Train/Val DataLoader 생성 테스트
    print("\n" + "="*60)
    print("CREATE TRAIN/VAL DATALOADERS")
    print("="*60)
    train_loader, val_loader = create_dataloaders(
        train_path=train_path,
        val_path=val_path,
        batch_size=8,
        img_size=416,
        num_workers=0,
        augment=True
    )
    print(f"\nTrain loader: {len(train_loader)} batches")
    print(f"Val loader: {len(val_loader)} batches")
    print("\n✓ DataLoader creation successful!\n")
