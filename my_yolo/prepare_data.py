"""
데이터셋 준비 스크립트
- images, labels 폴더 스캔
- train.txt, val.txt 생성 (상대 경로)
- 데이터 검증
"""

import os
import yaml
import argparse
import random
from pathlib import Path
from tqdm import tqdm


def load_config(config_path):
    """
    YAML 설정 파일 로드

    Args:
        config_path: YAML 파일 경로

    Returns:
        config: 설정 dict
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def scan_dataset(data_dir, extensions=['.jpg', '.jpeg', '.png']):
    """
    데이터셋 스캔 및 검증

    Args:
        data_dir: 데이터 디렉토리 (images, labels 폴더 포함)
        extensions: 이미지 확장자 리스트

    Returns:
        valid_images: 검증된 이미지 경로 리스트
    """
    data_dir = Path(data_dir)
    images_dir = data_dir / 'images'
    labels_dir = data_dir / 'labels'

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    print(f"\nScanning dataset in: {data_dir}")
    print(f"  Images dir: {images_dir}")
    print(f"  Labels dir: {labels_dir}\n")

    # 이미지 파일 찾기
    image_files = []
    for ext in extensions:
        image_files.extend(images_dir.glob(f'*{ext}'))

    print(f"Found {len(image_files)} image files")

    # 이미지-라벨 매칭 검증
    valid_images = []
    missing_labels = []

    for img_path in tqdm(image_files, desc="Validating dataset"):
        # 대응하는 라벨 파일 경로
        label_path = labels_dir / (img_path.stem + '.txt')

        if label_path.exists():
            # 라벨 파일이 비어있는지 확인 (선택적)
            if label_path.stat().st_size > 0:
                valid_images.append(img_path)
            else:
                # 비어있어도 추가 (객체가 없는 이미지일 수 있음)
                valid_images.append(img_path)
        else:
            missing_labels.append(img_path.name)

    print(f"\nValidation Results:")
    print(f"  Valid pairs: {len(valid_images)}")
    print(f"  Missing labels: {len(missing_labels)}")

    if missing_labels and len(missing_labels) <= 10:
        print(f"  Missing label files:")
        for name in missing_labels:
            print(f"    - {name}")

    return valid_images


def split_dataset(image_paths, split_ratio=0.8, seed=42):
    """
    데이터셋을 train/val로 분할

    Args:
        image_paths: 이미지 경로 리스트
        split_ratio: train 비율 (기본 0.8)
        seed: random seed

    Returns:
        train_images, val_images: 분할된 이미지 경로 리스트
    """
    random.seed(seed)
    random.shuffle(image_paths)

    split_idx = int(len(image_paths) * split_ratio)
    train_images = image_paths[:split_idx]
    val_images = image_paths[split_idx:]

    print(f"\nDataset Split (ratio={split_ratio}):")
    print(f"  Train: {len(train_images)} images")
    print(f"  Val:   {len(val_images)} images")

    return train_images, val_images


def save_paths(image_paths, output_file, data_dir, use_relative=True):
    """
    이미지 경로를 파일로 저장

    Args:
        image_paths: 이미지 경로 리스트
        output_file: 출력 파일 경로
        data_dir: 데이터 디렉토리 (상대 경로 계산용)
        use_relative: 상대 경로 사용 여부
    """
    output_file = Path(output_file)
    data_dir = Path(data_dir)

    with open(output_file, 'w') as f:
        for img_path in image_paths:
            if use_relative:
                # 상대 경로 계산 (프로젝트 루트 기준 = data_dir의 부모)
                try:
                    rel_path = img_path.relative_to(data_dir.parent)
                    path_str = str(rel_path)
                except ValueError:
                    # 상대 경로 계산 실패 시 절대 경로 사용
                    path_str = str(img_path.absolute())
            else:
                path_str = str(img_path.absolute())

            f.write(path_str + '\n')

    print(f"  Saved: {output_file} ({len(image_paths)} paths)")


def prepare_dataset(config_path, split_ratio=0.8, seed=42, use_relative=True):
    """
    데이터셋 준비 메인 함수

    Args:
        config_path: YAML 설정 파일 경로
        split_ratio: train/val 분할 비율
        seed: random seed
        use_relative: 상대 경로 사용 여부
    """
    print("="*60)
    print("YOLOv3 Dataset Preparation")
    print("="*60)

    # YAML 설정 로드
    config = load_config(config_path)
    print(f"\nConfig loaded: {config_path}")
    print(f"  Classes: {config['nc']}")
    print(f"  Class names: {config['names'][:3]}... (total {len(config['names'])})")

    # 데이터 디렉토리 (config 파일과 같은 위치)
    config_path = Path(config_path)
    data_dir = config_path.parent

    # 데이터셋 스캔 및 검증
    valid_images = scan_dataset(data_dir)

    if len(valid_images) == 0:
        raise ValueError("No valid image-label pairs found!")

    # Train/Val 분할
    train_images, val_images = split_dataset(valid_images, split_ratio, seed)

    # 경로 저장
    print(f"\nSaving paths (relative={use_relative}):")
    train_file = data_dir / 'train.txt'
    val_file = data_dir / 'val.txt'

    save_paths(train_images, train_file, data_dir, use_relative)
    save_paths(val_images, val_file, data_dir, use_relative)

    print("\n" + "="*60)
    print("Dataset preparation completed!")
    print("="*60)
    print(f"\nGenerated files:")
    print(f"  - {train_file}")
    print(f"  - {val_file}")
    print(f"\nYou can now start training with:")
    print(f"  python -m my_yolo.train --train-path {train_file} --val-path {val_file}")
    print("")


def parse_args():
    """커맨드 라인 인자 파싱"""
    parser = argparse.ArgumentParser(description='Prepare YOLOv3 dataset')

    parser.add_argument(
        '--config',
        type=str,
        default='./data/custom.yaml',
        help='Path to dataset config YAML file'
    )
    parser.add_argument(
        '--split-ratio',
        type=float,
        default=0.8,
        help='Train/val split ratio (default: 0.8)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--absolute-path',
        action='store_true',
        help='Use absolute paths instead of relative paths'
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    prepare_dataset(
        config_path=args.config,
        split_ratio=args.split_ratio,
        seed=args.seed,
        use_relative=not args.absolute_path
    )
