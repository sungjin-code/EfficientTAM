"""Validate promptable image segmentation on an SA-23-style image/mask root."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from efficient_track_anything.build_efficienttam import build_efficienttam
from efficient_track_anything.efficienttam_image_predictor import (
    EfficientTAMImagePredictor,
)
from efficient_track_anything.utils.misc import mask_to_box
from training.data.image_dataset import _index_images


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_binary_mask(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr > 0


def _box_from_mask(mask: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(mask.astype(bool))[None, None]
    box = mask_to_box(tensor)[0, 0].numpy().astype(np.float32)
    return box


def _iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(pred, gt).sum()) / float(union)


def evaluate_image_root(
    predictor: EfficientTAMImagePredictor,
    root: Path,
    max_images: int | None = None,
) -> dict:
    entries = _index_images(root)
    if max_images is not None:
        entries = entries[:max_images]
    per_mask: list[dict] = []

    for image_path, mask_paths in entries:
        image = np.array(Image.open(image_path).convert("RGB"))
        predictor.set_image(image)
        for mask_path in mask_paths:
            gt = _load_binary_mask(mask_path)
            if not gt.any():
                continue
            masks, iou_pred, _ = predictor.predict(
                box=_box_from_mask(gt),
                multimask_output=False,
            )
            pred = masks[0] > 0
            score = _iou(pred, gt)
            per_mask.append(
                {
                    "image": str(image_path),
                    "mask": str(mask_path),
                    "iou": score,
                    "predicted_iou": float(np.asarray(iou_pred).reshape(-1)[0]),
                }
            )

    miou = float(np.mean([m["iou"] for m in per_mask])) if per_mask else 0.0
    return {"mIoU": miou, "n_masks": len(per_mask), "per_mask": per_mask}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    device = pick_device()
    print(f"[validate_image_miou] device={device}")
    model = build_efficienttam(
        config_file=args.config,
        ckpt_path=args.ckpt,
        device=str(device),
        mode="eval",
    )
    predictor = EfficientTAMImagePredictor(model)
    results = evaluate_image_root(predictor, Path(args.root), args.max_images)
    print(f"[validate_image_miou] mIoU={results['mIoU']:.4f} n={results['n_masks']}")
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"[validate_image_miou] wrote {out}")


if __name__ == "__main__":
    main()
