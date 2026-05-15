"""Prompt sampling for EfficientTAM training.

Coordinate convention matches `EfficientTAMTransforms.transform_coords` and
`PromptEncoder._embed_points`: coordinates are in the **input image pixel space**
(0..image_size). They are NOT pre-normalized to [0,1].
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class PromptSample:
    """One frame's prompt — at most one of points/boxes/mask is populated."""

    point_coords: Optional[torch.Tensor] = None  # [N, 2] (x, y) in pixels
    point_labels: Optional[torch.Tensor] = None  # [N] (1 pos, 0 neg, -1 padding)
    box: Optional[torch.Tensor] = None  # [4] (x0, y0, x1, y1) in pixels
    mask: Optional[torch.Tensor] = None  # [1, H, W] low-res or full-res


def _sample_point_in_mask(
    mask_np: np.ndarray, positive: bool, rng: random.Random
) -> tuple[int, int]:
    """Pick a (x, y) pixel inside (positive) or outside (negative) the binary mask."""
    target = (mask_np > 0) if positive else (mask_np == 0)
    ys, xs = np.where(target)
    if len(xs) == 0:
        # Fallback: any pixel inside the mask if we needed negative but mask covers everything
        ys, xs = (
            np.where(mask_np > 0) if not positive else np.where(np.ones_like(mask_np))
        )
    idx = rng.randrange(len(xs))
    return int(xs[idx]), int(ys[idx])


def _box_from_mask(
    mask_np: np.ndarray, jitter_pct: float = 0.0, rng: Optional[random.Random] = None
) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask_np > 0)
    if len(xs) == 0:
        return 0, 0, 0, 0
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    if jitter_pct > 0 and rng is not None:
        w, h = x1 - x0, y1 - y0
        jx = int(w * jitter_pct)
        jy = int(h * jitter_pct)
        x0 += rng.randint(-jx, jx)
        y0 += rng.randint(-jy, jy)
        x1 += rng.randint(-jx, jx)
        y1 += rng.randint(-jy, jy)
        x0 = max(0, min(x0, mask_np.shape[1] - 1))
        x1 = max(x0 + 1, min(x1, mask_np.shape[1]))
        y0 = max(0, min(y0, mask_np.shape[0] - 1))
        y1 = max(y0 + 1, min(y1, mask_np.shape[0]))
    return x0, y0, x1, y1


class PromptSampler:
    """Sample point/box prompts from a binary GT mask, and correction clicks from errors.

    `mode`:
        - "point": single positive point
        - "box":   tight bounding box (optionally jittered)
        - "mixed": random choice per call (60% point, 40% box)
    """

    def __init__(
        self,
        mode: str = "mixed",
        max_neg_points: int = 0,
        box_jitter_pct: float = 0.05,
        seed: Optional[int] = None,
    ):
        assert mode in ("point", "box", "mixed")
        self.mode = mode
        self.max_neg_points = max_neg_points
        self.box_jitter_pct = box_jitter_pct
        self.rng = random.Random(seed)

    def _pick_mode(self) -> str:
        if self.mode != "mixed":
            return self.mode
        return "point" if self.rng.random() < 0.6 else "box"

    def sample(self, gt_mask: torch.Tensor) -> PromptSample:
        """Sample a single-prompt initial input from a binary GT mask [1, H, W] or [H, W]."""
        if gt_mask.dim() == 3:
            gt_mask = gt_mask.squeeze(0)
        mask_np = (gt_mask > 0).cpu().numpy().astype(np.uint8)
        if mask_np.sum() == 0:
            # Empty mask: emit a dummy negative point at (0, 0) — loss masks it out via has_object=0.
            return PromptSample(
                point_coords=torch.zeros(1, 2),
                point_labels=torch.tensor([-1], dtype=torch.int32),
            )

        mode = self._pick_mode()
        if mode == "point":
            x, y = _sample_point_in_mask(mask_np, positive=True, rng=self.rng)
            coords = [[x, y]]
            labels = [1]
            for _ in range(self.rng.randint(0, self.max_neg_points)):
                nx, ny = _sample_point_in_mask(mask_np, positive=False, rng=self.rng)
                coords.append([nx, ny])
                labels.append(0)
            return PromptSample(
                point_coords=torch.tensor(coords, dtype=torch.float32),
                point_labels=torch.tensor(labels, dtype=torch.int32),
            )

        # box
        x0, y0, x1, y1 = _box_from_mask(mask_np, self.box_jitter_pct, self.rng)
        # SAM/EfficientTAM encode a box as two corner points with labels 2, 3.
        coords = [[x0, y0], [x1, y1]]
        labels = [2, 3]
        return PromptSample(
            point_coords=torch.tensor(coords, dtype=torch.float32),
            point_labels=torch.tensor(labels, dtype=torch.int32),
        )

    def sample_correction(
        self,
        gt_mask: torch.Tensor,  # [1, H, W] or [H, W], binary
        pred_logits: torch.Tensor,  # [1, H, W] or [H, W], high-res mask logits
    ) -> PromptSample:
        """Pick the largest connected error region; click its centroid.

        Label = 1 if the error is a false negative (object pixel missed),
                0 if it is a false positive (background pixel labeled object).
        """
        if gt_mask.dim() == 3:
            gt_mask = gt_mask.squeeze(0)
        if pred_logits.dim() == 3:
            pred_logits = pred_logits.squeeze(0)
        gt_np = (gt_mask > 0).cpu().numpy().astype(np.uint8)
        pred_np = (pred_logits > 0).cpu().numpy().astype(np.uint8)
        fn = (gt_np == 1) & (pred_np == 0)
        fp = (gt_np == 0) & (pred_np == 1)
        # Pick whichever error region has more pixels — that's where the model is wrongest.
        if fn.sum() >= fp.sum():
            err, label = fn, 1
        else:
            err, label = fp, 0
        if err.sum() == 0:
            # Perfect prediction — sample a redundant positive point.
            x, y = _sample_point_in_mask(gt_np, positive=True, rng=self.rng)
            label = 1
        else:
            ys, xs = np.where(err)
            idx = self.rng.randrange(len(xs))
            x, y = int(xs[idx]), int(ys[idx])
        return PromptSample(
            point_coords=torch.tensor([[x, y]], dtype=torch.float32),
            point_labels=torch.tensor([label], dtype=torch.int32),
        )


def to_point_inputs(
    sample: PromptSample,
    batch_size: int = 1,
    device: torch.device | str = "cpu",
) -> Optional[dict]:
    """Convert one PromptSample into the `point_inputs` dict consumed by `_track_step`.

    Returns None if the sample carries only a mask (caller passes that as `mask_inputs`).
    """
    if sample.point_coords is None:
        return None
    coords = sample.point_coords.to(device).unsqueeze(0)  # [1, N, 2]
    labels = sample.point_labels.to(device).unsqueeze(0)  # [1, N]
    if batch_size > 1:
        coords = coords.expand(batch_size, -1, -1).contiguous()
        labels = labels.expand(batch_size, -1).contiguous()
    return {"point_coords": coords, "point_labels": labels}
