import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class YOLOv3Loss(nn.Module):
    """
    YOLOv3 Loss 함수 - 3가지
    1. Bounding Box Loss: 좌표 예측 오차 (x, y, w, h)
    2. Objectness Loss: 객체 존재 여부 예측 오차
    3. Classification Loss: 클래스 예측 오차

    각 스케일마다 독립적으로 loss 계산
    """

    def __init__(self, anchors, num_classes=25, img_size=416, ignore_thresh=0.5):
        """
        Args:
            anchors: Anchor boxes  [3, 3, 2] - [스케일, anchor_idx, (w, h)]
            num_classes: 클래스 개수
            img_size: 입력 이미지 크기
            ignore_thresh: IoU threshold (이 값보다 크면 negative sample로 간주하지 않음)
        """
        super(YOLOv3Loss, self).__init__()

        self.anchors = anchors
        self.num_classes = num_classes
        self.img_size = img_size
        self.ignore_thresh = ignore_thresh

        # Loss 가중치 (논문 기준)
        self.lambda_coord = 5.0    # 좌표 loss 가중치
        self.lambda_obj = 1.0      # objectness loss 가중치
        self.lambda_noobj = 0.5    # no-objectness loss 가중치
        self.lambda_class = 1.0    # classification loss 가중치

        # BCE Loss
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.mse_loss = nn.MSELoss(reduction='none')

    def forward(self, predictions, targets):
        """
        Args:
            predictions: 3개 스케일의 예측 결과 tuple
                - pred_large: [B, 3, 13, 13, 30]
                - pred_medium: [B, 3, 26, 26, 30]
                - pred_small: [B, 3, 52, 52, 30]
            targets: [total_objects, 6]
                     각 행: [batch_idx, class_id, x, y, w, h] (normalized)

        Returns:
            total_loss: 전체 loss
            loss_components: loss 구성 요소 dict
        """
        device = predictions[0].device
        batch_size = predictions[0].size(0)

        # 각 스케일별 grid size
        grid_sizes = [13, 26, 52]

        # 전체 loss 초기화
        total_loss = torch.tensor(0.0, device=device)
        total_box_loss = torch.tensor(0.0, device=device)
        total_obj_loss = torch.tensor(0.0, device=device)
        total_cls_loss = torch.tensor(0.0, device=device)

        # 각 스케일마다 loss 계산
        for scale_idx, (pred, grid_size) in enumerate(zip(predictions, grid_sizes)):
            # Anchor 가져오기
            anchors = torch.tensor(self.anchors[scale_idx], device=device).float()

            # Loss 계산
            loss, box_loss, obj_loss, cls_loss = self._compute_loss_for_scale(
                pred, targets, anchors, grid_size, batch_size
            )

            total_loss += loss
            total_box_loss += box_loss
            total_obj_loss += obj_loss
            total_cls_loss += cls_loss

        loss_components = {
            'total_loss': total_loss.item(),
            'box_loss': total_box_loss.item(),
            'obj_loss': total_obj_loss.item(),
            'cls_loss': total_cls_loss.item()
        }

        return total_loss, loss_components

    def _compute_loss_for_scale(self, pred, targets, anchors, grid_size, batch_size):
        """
        특정 스케일에서 loss 계산

        Args:
            pred: [B, 3, H, W, 30] - 예측값
            targets: [total_objects, 6] - Ground truth
            anchors: [3, 2] - 해당 스케일의 anchor boxes
            grid_size: H (= W)
            batch_size: B

        Returns:
            loss: 해당 스케일의 total loss
            box_loss, obj_loss, cls_loss: 각 구성 요소
        """
        device = pred.device
        num_anchors = anchors.size(0)

        # 예측값 분리
        # pred: [B, 3, H, W, 30]
        pred_xy = pred[..., 0:2]      # [B, 3, H, W, 2] - 중심 좌표 offset (sigmoid 전)
        pred_wh = pred[..., 2:4]      # [B, 3, H, W, 2] - 너비/높이 (exp 전)
        pred_obj = pred[..., 4:5]     # [B, 3, H, W, 1] - objectness (sigmoid 전)
        pred_cls = pred[..., 5:]      # [B, 3, H, W, num_classes] - class probs (sigmoid 전)

        # Grid 좌표 생성
        # grid_y, grid_x: [H, W]
        grid_y, grid_x = torch.meshgrid(
            torch.arange(grid_size, device=device),
            torch.arange(grid_size, device=device),
            indexing='ij'
        )
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()  # [H, W, 2]

        # Anchor를 grid 스케일로 정규화
        # anchors: [3, 2] -> [1, 3, 1, 1, 2]
        scaled_anchors = anchors / grid_size
        scaled_anchors = scaled_anchors.view(1, num_anchors, 1, 1, 2)

        # Target 텐서 초기화
        obj_mask = torch.zeros(batch_size, num_anchors, grid_size, grid_size, 1, device=device)
        noobj_mask = torch.ones(batch_size, num_anchors, grid_size, grid_size, 1, device=device)
        tx = torch.zeros(batch_size, num_anchors, grid_size, grid_size, 1, device=device)
        ty = torch.zeros(batch_size, num_anchors, grid_size, grid_size, 1, device=device)
        tw = torch.zeros(batch_size, num_anchors, grid_size, grid_size, 1, device=device)
        th = torch.zeros(batch_size, num_anchors, grid_size, grid_size, 1, device=device)
        tcls = torch.zeros(batch_size, num_anchors, grid_size, grid_size, self.num_classes, device=device)

        # Target이 있는 경우에만 처리
        if targets.size(0) > 0:
            # Targets 분리
            batch_idx = targets[:, 0].long()
            target_cls = targets[:, 1].long()
            target_x = targets[:, 2] * grid_size
            target_y = targets[:, 3] * grid_size
            target_w = targets[:, 4] * grid_size
            target_h = targets[:, 5] * grid_size

            # Grid cell 인덱스
            gx = target_x.long()
            gy = target_y.long()

            # Grid cell 내 offset
            gx_offset = target_x - gx.float()
            gy_offset = target_y - gy.float()

            # 각 target마다 best anchor 찾기
            target_wh = torch.stack([target_w, target_h], dim=1)  # [num_targets, 2]

            for t_idx in range(targets.size(0)):
                b = batch_idx[t_idx]
                gx_i = gx[t_idx]
                gy_i = gy[t_idx]

                # Grid 범위 체크
                if gx_i >= grid_size or gy_i >= grid_size or gx_i < 0 or gy_i < 0:
                    continue

                # Best anchor 찾기 (IoU 최대)
                gt_box = torch.zeros(4, device=device)
                gt_box[2:] = target_wh[t_idx]

                ious = []
                for anchor in scaled_anchors[0, :, 0, 0, :]:
                    anchor_box = torch.zeros(4, device=device)
                    anchor_box[2:] = anchor
                    iou = self._bbox_iou(gt_box.unsqueeze(0), anchor_box.unsqueeze(0), xywh=True)
                    ious.append(iou)

                best_anchor = torch.tensor(ious, device=device).argmax()

                # Mask 설정
                obj_mask[b, best_anchor, gy_i, gx_i, 0] = 1
                noobj_mask[b, best_anchor, gy_i, gx_i, 0] = 0

                # Target 값 설정
                tx[b, best_anchor, gy_i, gx_i, 0] = gx_offset[t_idx]
                ty[b, best_anchor, gy_i, gx_i, 0] = gy_offset[t_idx]
                tw[b, best_anchor, gy_i, gx_i, 0] = torch.log(target_w[t_idx] / scaled_anchors[0, best_anchor, 0, 0, 0] + 1e-16)
                th[b, best_anchor, gy_i, gx_i, 0] = torch.log(target_h[t_idx] / scaled_anchors[0, best_anchor, 0, 0, 1] + 1e-16)
                tcls[b, best_anchor, gy_i, gx_i, target_cls[t_idx]] = 1

        # Loss 계산
        # 1. Bounding Box Loss (obj_mask가 1인 위치만)
        # BCE Loss와 MSE Loss를 섞어서 사용
        loss_x = self.bce_loss(pred_xy[..., 0:1], tx)
        loss_y = self.bce_loss(pred_xy[..., 1:2], ty)
        loss_w = self.mse_loss(pred_wh[..., 0:1], tw)
        loss_h = self.mse_loss(pred_wh[..., 1:2], th)

        box_loss = obj_mask * (loss_x + loss_y + loss_w + loss_h)
        box_loss = box_loss.sum() * self.lambda_coord

        # 2. Objectness Loss
        loss_obj = self.bce_loss(pred_obj, obj_mask)
        loss_obj = (obj_mask * loss_obj).sum() * self.lambda_obj

        loss_noobj = self.bce_loss(pred_obj, obj_mask)
        loss_noobj = (noobj_mask * loss_noobj).sum() * self.lambda_noobj

        obj_loss = loss_obj + loss_noobj

        # 3. Classification Loss (obj_mask가 1인 위치만)
        cls_loss = self.bce_loss(pred_cls, tcls)
        cls_loss = (obj_mask * cls_loss.sum(dim=-1, keepdim=True)).sum() * self.lambda_class

        # Total loss
        total_loss = box_loss + obj_loss + cls_loss

        return total_loss, box_loss, obj_loss, cls_loss

    def _bbox_iou(self, box1, box2, xywh=True, eps=1e-9):
        """
        IoU 계산

        Args:
            box1: [N, 4] - (x, y, w, h) 또는 (x1, y1, x2, y2)
            box2: [M, 4]
            xywh: True면 (x, y, w, h) 포맷, False면 (x1, y1, x2, y2) 포맷

        Returns:
            iou: [N, M] - IoU 행렬
        """
        if xywh:
            # (x, y, w, h) -> (x1, y1, x2, y2)
            b1_x1 = box1[:, 0] - box1[:, 2] / 2
            b1_y1 = box1[:, 1] - box1[:, 3] / 2
            b1_x2 = box1[:, 0] + box1[:, 2] / 2
            b1_y2 = box1[:, 1] + box1[:, 3] / 2

            b2_x1 = box2[:, 0] - box2[:, 2] / 2
            b2_y1 = box2[:, 1] - box2[:, 3] / 2
            b2_x2 = box2[:, 0] + box2[:, 2] / 2
            b2_y2 = box2[:, 1] + box2[:, 3] / 2
        else:
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

        # Intersection area
        inter_x1 = torch.max(b1_x1.unsqueeze(1), b2_x1)
        inter_y1 = torch.max(b1_y1.unsqueeze(1), b2_y1)
        inter_x2 = torch.min(b1_x2.unsqueeze(1), b2_x2)
        inter_y2 = torch.min(b1_y2.unsqueeze(1), b2_y2)

        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h

        # Union area
        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        union_area = b1_area.unsqueeze(1) + b2_area - inter_area

        # IoU
        iou = inter_area / (union_area + eps)

        return iou
