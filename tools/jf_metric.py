"""DAVIS-style J&F metrics.

Vendored from the davis2017-evaluation reference implementation
(BSD-3-Clause; Pont-Tuset et al., 2017) so we don't pull a heavy dependency
just for two metrics.

- J = Jaccard / region similarity = IoU(pred, gt)
- F = boundary F-measure between contour pixels of pred and gt, computed
      with a disk-shaped dilation tolerance proportional to the image diagonal.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation


def _to_bool(mask: np.ndarray) -> np.ndarray:
    return mask.astype(bool)


def db_eval_iou(gt: np.ndarray, pred: np.ndarray) -> float:
    gt = _to_bool(gt)
    pred = _to_bool(pred)
    inter = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()
    if union == 0:
        return 1.0 if not gt.any() and not pred.any() else 0.0
    return float(inter) / float(union)


def _seg2bmap(seg: np.ndarray) -> np.ndarray:
    """Boundary map: pixels of `seg` that differ from any 4-neighbor."""
    h, w = seg.shape
    e = np.zeros_like(seg, dtype=bool)
    e[:, :-1] |= seg[:, :-1] != seg[:, 1:]
    e[:-1, :] |= seg[:-1, :] != seg[1:, :]
    return np.logical_and(e, seg)


def db_eval_boundary(
    gt: np.ndarray, pred: np.ndarray, bound_th: float = 0.008
) -> float:
    """F-measure over the boundary pixels, with `bound_th * diag` dilation tolerance."""
    gt = _to_bool(gt)
    pred = _to_bool(pred)
    if not gt.any() and not pred.any():
        return 1.0
    if not gt.any() or not pred.any():
        return 0.0

    h, w = gt.shape
    bound_pix = int(np.ceil(bound_th * np.sqrt(h * h + w * w)))
    pred_b = _seg2bmap(pred)
    gt_b = _seg2bmap(gt)

    struct = np.ones((3, 3), dtype=bool)
    pred_dil = binary_dilation(pred_b, structure=struct, iterations=bound_pix)
    gt_dil = binary_dilation(gt_b, structure=struct, iterations=bound_pix)

    pred_match = np.logical_and(pred_b, gt_dil).sum()
    gt_match = np.logical_and(gt_b, pred_dil).sum()
    n_pred = pred_b.sum()
    n_gt = gt_b.sum()

    precision = pred_match / n_pred if n_pred > 0 else 0.0
    recall = gt_match / n_gt if n_gt > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def compute_sequence(pred_seq: np.ndarray, gt_seq: np.ndarray) -> dict:
    """Per-frame J/F averaged over a sequence.

    Args:
        pred_seq: [T, H, W] bool/{0,1}
        gt_seq:   [T, H, W] bool/{0,1}
    """
    assert pred_seq.shape == gt_seq.shape
    T = pred_seq.shape[0]
    js = np.zeros(T)
    fs = np.zeros(T)
    for t in range(T):
        js[t] = db_eval_iou(gt_seq[t], pred_seq[t])
        fs[t] = db_eval_boundary(gt_seq[t], pred_seq[t])
    return {
        "J_per_frame": js,
        "F_per_frame": fs,
        "J_mean": float(js.mean()),
        "F_mean": float(fs.mean()),
        "JF_mean": float(0.5 * (js.mean() + fs.mean())),
    }
