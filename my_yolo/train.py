import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from tqdm import tqdm
import argparse
from pathlib import Path

from .model import YOLOv3
from .loss import YOLOv3Loss
from .dataset import get_anchors
from .dataloader import create_dataloaders


class Trainer:
    """
    YOLOv3 Trainer 클래스
    """

    def __init__(self, args):
        """
        Args:
            args: argparse arguments
        """
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"\n{'='*60}")
        print(f"YOLOv3 Training")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Num classes: {args.num_classes}")
        print(f"Batch size: {args.batch_size}")
        print(f"Image size: {args.img_size}")
        print(f"Epochs: {args.epochs}")
        print(f"Learning rate: {args.lr}")
        print(f"{'='*60}\n")

        # 모델 초기화
        self.model = YOLOv3(num_classes=args.num_classes).to(self.device)
        print(f"Model created successfully!")

        # 파라미터 수 출력
        total_params, trainable_params = self.model.get_num_params()
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}\n")

        # Loss 함수
        anchors = get_anchors()
        self.criterion = YOLOv3Loss(
            anchors=anchors,
            num_classes=args.num_classes,
            img_size=args.img_size,
            ignore_thresh=0.5
        )

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        # Learning Rate Scheduler
        if args.scheduler == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=args.epochs,
                eta_min=args.lr * 0.01
            )
        elif args.scheduler == 'step':
            self.scheduler = StepLR(
                self.optimizer,
                step_size=args.step_size,
                gamma=0.1
            )
        else:
            self.scheduler = None

        # DataLoader
        print("Creating dataloaders...")
        self.train_loader, self.val_loader = create_dataloaders(
            train_path=args.train_path,
            val_path=args.val_path,
            batch_size=args.batch_size,
            img_size=args.img_size,
            num_workers=args.num_workers,
            augment=args.augment
        )
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}\n")

        # Checkpoint 디렉토리
        self.checkpoint_dir = Path(args.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Best loss 기록
        self.best_val_loss = float('inf')

        # Resume from checkpoint
        self.start_epoch = 0
        if args.resume:
            self.load_checkpoint(args.resume)

    def train_epoch(self, epoch):
        """
        한 에폭 학습

        Args:
            epoch: 현재 에폭 번호

        Returns:
            평균 loss dict
        """
        self.model.train()

        total_loss = 0.0
        total_box_loss = 0.0
        total_obj_loss = 0.0
        total_cls_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.args.epochs} [Train]")

        for batch_idx, (imgs, targets) in enumerate(pbar):
            imgs = imgs.to(self.device)
            targets = targets.to(self.device)

            # Forward
            predictions = self.model(imgs)

            # Loss 계산
            loss, loss_components = self.criterion(predictions, targets)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Loss 누적
            total_loss += loss_components['total_loss']
            total_box_loss += loss_components['box_loss']
            total_obj_loss += loss_components['obj_loss']
            total_cls_loss += loss_components['cls_loss']

            # Progress bar 업데이트
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'box': f"{loss_components['box_loss']:.4f}",
                'obj': f"{loss_components['obj_loss']:.4f}",
                'cls': f"{loss_components['cls_loss']:.4f}"
            })

        # 평균 loss
        num_batches = len(self.train_loader)
        avg_losses = {
            'total_loss': total_loss / num_batches,
            'box_loss': total_box_loss / num_batches,
            'obj_loss': total_obj_loss / num_batches,
            'cls_loss': total_cls_loss / num_batches
        }

        return avg_losses

    @torch.no_grad()
    def validate(self, epoch):
        """
        Validation

        Args:
            epoch: 현재 에폭 번호

        Returns:
            평균 loss dict
        """
        self.model.eval()

        total_loss = 0.0
        total_box_loss = 0.0
        total_obj_loss = 0.0
        total_cls_loss = 0.0

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch}/{self.args.epochs} [Val]")

        for imgs, targets in pbar:
            imgs = imgs.to(self.device)
            targets = targets.to(self.device)

            # Forward
            predictions = self.model(imgs)

            # Loss 계산
            loss, loss_components = self.criterion(predictions, targets)

            # Loss 누적
            total_loss += loss_components['total_loss']
            total_box_loss += loss_components['box_loss']
            total_obj_loss += loss_components['obj_loss']
            total_cls_loss += loss_components['cls_loss']

            # Progress bar 업데이트
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'box': f"{loss_components['box_loss']:.4f}",
                'obj': f"{loss_components['obj_loss']:.4f}",
                'cls': f"{loss_components['cls_loss']:.4f}"
            })

        # 평균 loss
        num_batches = len(self.val_loader)
        avg_losses = {
            'total_loss': total_loss / num_batches,
            'box_loss': total_box_loss / num_batches,
            'obj_loss': total_obj_loss / num_batches,
            'cls_loss': total_cls_loss / num_batches
        }

        return avg_losses

    def train(self):
        """
        전체 학습 루프
        """
        print("Starting training...\n")

        for epoch in range(self.start_epoch + 1, self.args.epochs + 1):
            # Train
            train_losses = self.train_epoch(epoch)

            # Validate
            val_losses = self.validate(epoch)

            # Learning rate 업데이트
            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
            else:
                current_lr = self.args.lr

            # 결과 출력
            print(f"\nEpoch {epoch}/{self.args.epochs}")
            print(f"  LR: {current_lr:.6f}")
            print(f"  Train Loss: {train_losses['total_loss']:.4f} "
                  f"(box: {train_losses['box_loss']:.4f}, "
                  f"obj: {train_losses['obj_loss']:.4f}, "
                  f"cls: {train_losses['cls_loss']:.4f})")
            print(f"  Val Loss:   {val_losses['total_loss']:.4f} "
                  f"(box: {val_losses['box_loss']:.4f}, "
                  f"obj: {val_losses['obj_loss']:.4f}, "
                  f"cls: {val_losses['cls_loss']:.4f})")

            # Checkpoint 저장
            if epoch % self.args.save_interval == 0:
                self.save_checkpoint(epoch, val_losses['total_loss'], is_best=False)

            # Best model 저장
            if val_losses['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_losses['total_loss']
                self.save_checkpoint(epoch, val_losses['total_loss'], is_best=True)
                print(f"  -> Best model saved! (val_loss: {self.best_val_loss:.4f})")

            print("")

        print("Training completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")

    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """
        Checkpoint 저장

        Args:
            epoch: 현재 에폭
            val_loss: validation loss
            is_best: best model 여부
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # Best model
        if is_best:
            best_path = self.checkpoint_dir / 'best.pt'
            torch.save(checkpoint, best_path)

        # Regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)

        # Last checkpoint (항상 덮어쓰기)
        last_path = self.checkpoint_dir / 'last.pt'
        torch.save(checkpoint, last_path)

    def load_checkpoint(self, checkpoint_path):
        """
        Checkpoint 로드

        Args:
            checkpoint_path: checkpoint 파일 경로
        """
        print(f"Loading checkpoint from {checkpoint_path}...")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.start_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        print(f"Checkpoint loaded! Resuming from epoch {self.start_epoch}\n")


def parse_args():
    """
    커맨드 라인 인자 파싱
    """
    parser = argparse.ArgumentParser(description='YOLOv3 Training')

    # Data
    parser.add_argument('--train-path', type=str, default='data/train.txt',
                        help='Path to train.txt')
    parser.add_argument('--val-path', type=str, default='data/val.txt',
                        help='Path to val.txt')
    parser.add_argument('--num-classes', type=int, default=25,
                        help='Number of classes')

    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--img-size', type=int, default=416,
                        help='Input image size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='Weight decay')

    # Scheduler
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'step', 'none'],
                        help='Learning rate scheduler')
    parser.add_argument('--step-size', type=int, default=30,
                        help='Step size for StepLR scheduler')

    # Augmentation
    parser.add_argument('--augment', action='store_true', default=True,
                        help='Use data augmentation')

    # DataLoader
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of dataloader workers')

    # Checkpoint
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--save-interval', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default='',
                        help='Resume from checkpoint')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # 학습 시작
    trainer = Trainer(args)
    trainer.train()
