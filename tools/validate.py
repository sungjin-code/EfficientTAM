"""Validate EfficientTAM on a DAVIS-style VOS validation set.

For each video, prompts frame 0 with the GT mask and propagates through the
whole clip; computes J&F for the predicted track.

Usage:
    python -m tools.validate \\
        --config configs/efficienttam/efficienttam_s.yaml \\
        --ckpt checkpoints/efficienttam_s.pt \\
        --val-root /path/to/DAVIS17/val
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from efficient_track_anything.build_efficienttam import (
    build_efficienttam_video_predictor,
)
from tools.jf_metric import compute_sequence


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_palette_mask(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr


def _binary_obj_mask(palette: np.ndarray, obj_id: int) -> np.ndarray:
    return (palette == obj_id).astype(np.uint8)


def _has_video_dirs(jpeg_root: Path, ann_root: Path) -> bool:
    if not jpeg_root.is_dir() or not ann_root.is_dir():
        return False
    for video_dir in jpeg_root.iterdir():
        if video_dir.is_dir() and (ann_root / video_dir.name).is_dir():
            return True
    return False


def _resolve_standard_layout(val_root: Path) -> tuple[Path, Path] | None:
    candidates = [
        # DAVIS trainval zip extracts as JPEGImages/480p and Annotations/480p.
        (val_root / "JPEGImages" / "480p", val_root / "Annotations" / "480p"),
        (val_root / "JPEGImages", val_root / "Annotations"),
    ]
    for jpeg_root, ann_root in candidates:
        if _has_video_dirs(jpeg_root, ann_root):
            return jpeg_root, ann_root
    return None


def _resolve_sav_layout(val_root: Path) -> tuple[Path, Path] | None:
    jpeg_root = val_root / "JPEGImages_24fps"
    ann_root = val_root / "Annotations_6fps"
    if _has_video_dirs(jpeg_root, ann_root):
        return jpeg_root, ann_root
    return None


def has_supported_vos_layout(val_root: Path) -> bool:
    return (
        _resolve_standard_layout(val_root) is not None
        or _resolve_sav_layout(val_root) is not None
    )


def evaluate(predictor, val_root: Path, max_videos: int | None) -> dict:
    sav_layout = _resolve_sav_layout(val_root)
    if sav_layout is not None:
        return _evaluate_sav_layout(predictor, sav_layout[0], sav_layout[1], max_videos)

    standard_layout = _resolve_standard_layout(val_root)
    if standard_layout is None:
        raise RuntimeError(
            f"Unsupported VOS layout at {val_root}. Expected DAVIS-style "
            "JPEGImages/Annotations, DAVIS JPEGImages/480p/Annotations/480p, "
            "or SA-V JPEGImages_24fps/Annotations_6fps."
        )
    jpeg_root, ann_root = standard_layout
    videos = sorted([d.name for d in jpeg_root.iterdir() if d.is_dir()])
    if max_videos is not None:
        videos = videos[:max_videos]

    per_video: dict[str, dict] = {}
    j_all: list[float] = []
    f_all: list[float] = []

    for video in videos:
        frames_dir = jpeg_root / video
        ann_dir = ann_root / video
        frames = sorted(frames_dir.glob("*.jpg"))
        if not frames:
            continue
        first_ann = _load_palette_mask(ann_dir / (frames[0].stem + ".png"))
        obj_ids = [int(i) for i in np.unique(first_ann) if i != 0]
        if not obj_ids:
            continue

        state = predictor.init_state(str(frames_dir))
        H, W = first_ann.shape

        # Seed every object with its GT mask on frame 0.
        for obj_id in obj_ids:
            gt0 = _binary_obj_mask(first_ann, obj_id)
            predictor.add_new_mask(
                state, frame_idx=0, obj_id=obj_id, mask=torch.from_numpy(gt0).bool()
            )

        # Propagate; collect per-object high-res masks per frame.
        pred_per_obj: dict[int, dict[int, np.ndarray]] = {oid: {} for oid in obj_ids}
        for frame_idx, out_obj_ids, mask_logits in predictor.propagate_in_video(state):
            for i, oid in enumerate(out_obj_ids):
                m = (mask_logits[i] > 0).cpu().numpy().astype(np.uint8)
                if m.ndim == 3:
                    m = m[0]
                pred_per_obj[oid][frame_idx] = m

        # Score per object (mean across objects then across frames).
        video_J: list[float] = []
        video_F: list[float] = []
        for oid in obj_ids:
            gt_seq = []
            pr_seq = []
            for frame_idx, fp in enumerate(frames):
                gt_palette = _load_palette_mask(ann_dir / (fp.stem + ".png"))
                gt_bin = _binary_obj_mask(gt_palette, oid)
                pr = pred_per_obj[oid].get(frame_idx)
                if pr is None:
                    pr = np.zeros_like(gt_bin)
                if pr.shape != gt_bin.shape:
                    pr_t = torch.from_numpy(pr).float()[None, None]
                    pr_t = F.interpolate(pr_t, size=gt_bin.shape, mode="nearest")
                    pr = pr_t[0, 0].numpy().astype(np.uint8)
                gt_seq.append(gt_bin)
                pr_seq.append(pr)
            res = compute_sequence(np.stack(pr_seq), np.stack(gt_seq))
            video_J.append(res["J_mean"])
            video_F.append(res["F_mean"])

        vJ = float(np.mean(video_J))
        vF = float(np.mean(video_F))
        per_video[video] = {"J": vJ, "F": vF, "JF": 0.5 * (vJ + vF)}
        j_all.extend(video_J)
        f_all.extend(video_F)
        print(f"  {video}: J={vJ:.3f} F={vF:.3f} JF={0.5 * (vJ + vF):.3f}")

    J_mean = float(np.mean(j_all)) if j_all else 0.0
    F_mean = float(np.mean(f_all)) if f_all else 0.0
    return {
        "per_video": per_video,
        "J_mean": J_mean,
        "F_mean": F_mean,
        "JF_mean": 0.5 * (J_mean + F_mean),
        "n_videos": len(per_video),
    }


def _evaluate_sav_layout(
    predictor,
    jpeg_root: Path,
    ann_root: Path,
    max_videos: int | None,
) -> dict:
    split_files = sorted(jpeg_root.parent.glob("sav_*.txt"))
    if split_files:
        videos = [
            line.strip()
            for line in split_files[0].read_text().splitlines()
            if line.strip()
        ]
    else:
        videos = sorted([d.name for d in jpeg_root.iterdir() if d.is_dir()])
    videos = [v for v in videos if (jpeg_root / v).is_dir() and (ann_root / v).is_dir()]
    if max_videos is not None:
        videos = videos[:max_videos]

    per_video: dict[str, dict] = {}
    j_all: list[float] = []
    f_all: list[float] = []

    for video in videos:
        frames_dir = jpeg_root / video
        video_ann_dir = ann_root / video
        frames = sorted(frames_dir.glob("*.jpg"))
        if not frames:
            continue
        frame_lookup = {fp.stem: idx for idx, fp in enumerate(frames)}
        obj_dirs = sorted([d for d in video_ann_dir.iterdir() if d.is_dir()])
        if not obj_dirs:
            continue

        state = predictor.init_state(str(frames_dir))
        eval_masks_by_obj: dict[int, list[Path]] = {}
        for obj_idx, obj_dir in enumerate(obj_dirs, start=1):
            masks = sorted(obj_dir.glob("*.png"))
            masks = [m for m in masks if m.stem in frame_lookup]
            if not masks:
                continue
            first_mask = masks[0]
            eval_masks_by_obj[obj_idx] = masks[1:-1] if len(masks) > 2 else masks
            predictor.add_new_mask(
                state,
                frame_idx=frame_lookup[first_mask.stem],
                obj_id=obj_idx,
                mask=torch.from_numpy(_load_palette_mask(first_mask) > 0),
            )

        if not eval_masks_by_obj:
            continue

        pred_per_obj: dict[int, dict[int, np.ndarray]] = {
            oid: {} for oid in eval_masks_by_obj
        }
        for frame_idx, out_obj_ids, mask_logits in predictor.propagate_in_video(state):
            for i, oid in enumerate(out_obj_ids):
                if int(oid) not in pred_per_obj:
                    continue
                m = (mask_logits[i] > 0).cpu().numpy().astype(np.uint8)
                if m.ndim == 3:
                    m = m[0]
                pred_per_obj[int(oid)][frame_idx] = m

        video_J: list[float] = []
        video_F: list[float] = []
        for oid, mask_paths in eval_masks_by_obj.items():
            gt_seq = []
            pr_seq = []
            for mask_path in mask_paths:
                frame_idx = frame_lookup[mask_path.stem]
                gt_bin = (_load_palette_mask(mask_path) > 0).astype(np.uint8)
                pr = pred_per_obj[oid].get(frame_idx)
                if pr is None:
                    pr = np.zeros_like(gt_bin)
                if pr.shape != gt_bin.shape:
                    pr_t = torch.from_numpy(pr).float()[None, None]
                    pr_t = F.interpolate(pr_t, size=gt_bin.shape, mode="nearest")
                    pr = pr_t[0, 0].numpy().astype(np.uint8)
                gt_seq.append(gt_bin)
                pr_seq.append(pr)
            res = compute_sequence(np.stack(pr_seq), np.stack(gt_seq))
            video_J.append(res["J_mean"])
            video_F.append(res["F_mean"])

        if not video_J:
            continue
        vJ = float(np.mean(video_J))
        vF = float(np.mean(video_F))
        per_video[video] = {"J": vJ, "F": vF, "JF": 0.5 * (vJ + vF)}
        j_all.extend(video_J)
        f_all.extend(video_F)
        print(f"  {video}: J={vJ:.3f} F={vF:.3f} JF={0.5 * (vJ + vF):.3f}")

    J_mean = float(np.mean(j_all)) if j_all else 0.0
    F_mean = float(np.mean(f_all)) if f_all else 0.0
    return {
        "per_video": per_video,
        "J_mean": J_mean,
        "F_mean": F_mean,
        "JF_mean": 0.5 * (J_mean + F_mean),
        "n_videos": len(per_video),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--val-root", required=True)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    device = pick_device()
    print(f"[validate] device={device}")
    predictor = build_efficienttam_video_predictor(
        config_file=args.config,
        ckpt_path=args.ckpt,
        device=str(device),
        mode="eval",
    )

    results = evaluate(predictor, Path(args.val_root), args.max_videos)
    print("=" * 60)
    print(
        f"Overall: J={results['J_mean']:.3f}  F={results['F_mean']:.3f}  J&F={results['JF_mean']:.3f}  "
        f"(n={results['n_videos']})"
    )
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(results, indent=2))
        print(f"[validate] wrote {args.output_json}")


if __name__ == "__main__":
    main()
