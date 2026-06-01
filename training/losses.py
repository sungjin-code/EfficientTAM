"""Losses for EfficientTAM training (focal + dice + IoU L1 + object-score BCE).

The mask head can emit either a single mask (K=1) or a multimask candidate set (K=3,
when `_use_multimask` returns True). For multimask outputs we pick the candidate with
the lowest combined focal+dice loss and backprop only that channel — matches SAM/SAM2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F


def sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Focal loss averaged over spatial dims, matching SAM2 training.

    Args:
        logits: [B, K, H, W]
        targets: [B, 1, H, W] in {0, 1}; broadcast across K candidates

    Returns:
        [B, K] per-candidate loss.
    """
    targets = targets.expand_as(logits).float()
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.flatten(2).mean(-1)


def dice_loss(
    logits: torch.Tensor, targets: torch.Tensor, eps: float = 1.0
) -> torch.Tensor:
    """Soft Dice loss per-(batch, channel). Shapes match focal."""
    targets = targets.expand_as(logits).float()
    probs = torch.sigmoid(logits)
    probs_flat = probs.flatten(2)
    tgt_flat = targets.flatten(2)
    inter = (probs_flat * tgt_flat).sum(-1)
    denom = probs_flat.sum(-1) + tgt_flat.sum(-1)
    return 1 - (2 * inter + eps) / (denom + eps)


def iou_l1_loss(
    iou_pred: torch.Tensor,
    pred_mask_logits: torch.Tensor,
    gt_mask: torch.Tensor,
) -> torch.Tensor:
    """L1 between predicted IoU and the true IoU of (pred>0) vs gt.

    Args:
        iou_pred: [B, K]
        pred_mask_logits: [B, K, H, W]
        gt_mask: [B, 1, H, W]
    """
    with torch.no_grad():
        pred_bin = (pred_mask_logits > 0).float()
        tgt = gt_mask.expand_as(pred_bin).float()
        inter = (pred_bin * tgt).flatten(2).sum(-1)
        union = ((pred_bin + tgt) > 0).float().flatten(2).sum(-1)
        true_iou = inter / union.clamp(min=1.0)
    return (iou_pred - true_iou).abs()


def obj_score_bce(obj_logits: torch.Tensor, has_object: torch.Tensor) -> torch.Tensor:
    """BCE over object-presence head. obj_logits: [B, 1], has_object: [B] in {0,1}."""
    target = has_object.float().view_as(obj_logits)
    return F.binary_cross_entropy_with_logits(
        obj_logits, target, reduction="none"
    ).mean(-1)


@dataclass
class LossWeights:
    focal: float = 20.0
    dice: float = 1.0
    iou: float = 1.0
    obj: float = 1.0
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0


class MultiStepLoss(torch.nn.Module):
    """Combined per-frame loss with best-of-K mask selection."""

    def __init__(self, weights: LossWeights | None = None):
        super().__init__()
        self.w = weights or LossWeights()

    def forward(
        self,
        mask_logits: torch.Tensor | list[torch.Tensor],
        ious: torch.Tensor | list[torch.Tensor],
        object_score_logits: torch.Tensor | list[torch.Tensor],
        gt_mask: torch.Tensor,  # [B, 1, H, W] full-res GT
        has_object: torch.Tensor,  # [B] in {0,1}
    ) -> Tuple[torch.Tensor, dict]:
        if isinstance(mask_logits, list):
            assert isinstance(ious, list)
            assert isinstance(object_score_logits, list)
            total_loss = gt_mask.new_zeros(())
            metric_sums: dict[str, torch.Tensor] = {}
            for masks_i, ious_i, obj_i in zip(
                mask_logits, ious, object_score_logits
            ):
                loss_i, metrics_i = self._forward_one(
                    masks_i, ious_i, obj_i, gt_mask, has_object
                )
                total_loss = total_loss + loss_i
                for k, v in metrics_i.items():
                    metric_sums[k] = metric_sums.get(k, gt_mask.new_zeros(())) + v
            return total_loss, metric_sums

        return self._forward_one(
            mask_logits, ious, object_score_logits, gt_mask, has_object
        )

    def _forward_one(
        self,
        mask_logits: torch.Tensor,  # [B, K, H, W] high-res mask logits
        ious: torch.Tensor,  # [B, K]
        object_score_logits: torch.Tensor,  # [B, 1]
        gt_mask: torch.Tensor,
        has_object: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        B, K = mask_logits.shape[:2]
        focal = sigmoid_focal_loss(
            mask_logits,
            gt_mask,
            alpha=self.w.focal_alpha,
            gamma=self.w.focal_gamma,
        )  # [B, K]
        dice = dice_loss(mask_logits, gt_mask)  # [B, K]
        iou_l1 = iou_l1_loss(ious, mask_logits, gt_mask)  # [B, K]

        # Mask out frames where the object is absent — only the obj_score head
        # should be supervised on those frames; pushing empty-mask predictions
        # via focal/dice would conflict with the obj-score signal.
        present = has_object.float().view(B, 1)  # [B, 1]

        # Best-of-K selection on (focal + dice). Detach idx so gradients flow
        # only through the chosen channel.
        combined = self.w.focal * focal + self.w.dice * dice  # [B, K]
        if K > 1:
            best_idx = combined.argmin(dim=1, keepdim=True).detach()  # [B, 1]
            focal_sel = focal.gather(1, best_idx).squeeze(1)
            dice_sel = dice.gather(1, best_idx).squeeze(1)
            iou_sel = iou_l1.gather(1, best_idx).squeeze(1)
        else:
            focal_sel = focal.squeeze(1)
            dice_sel = dice.squeeze(1)
            iou_sel = iou_l1.squeeze(1)

        focal_sel = focal_sel * present.squeeze(1)
        dice_sel = dice_sel * present.squeeze(1)
        iou_sel = iou_sel * present.squeeze(1)

        obj_bce = obj_score_bce(object_score_logits, has_object)  # [B]

        loss = (
            self.w.focal * focal_sel.mean()
            + self.w.dice * dice_sel.mean()
            + self.w.iou * iou_sel.mean()
            + self.w.obj * obj_bce.mean()
        )
        metrics = {
            "loss": loss.detach(),
            "focal": focal_sel.mean().detach(),
            "dice": dice_sel.mean().detach(),
            "iou_l1": iou_sel.mean().detach(),
            "obj_bce": obj_bce.mean().detach(),
        }
        return loss, metrics
