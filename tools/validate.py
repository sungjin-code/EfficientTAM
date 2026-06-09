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
        (val_root / "valid" / "JPEGImages", val_root / "valid" / "Annotations"),
        (val_root / "val" / "JPEGImages", val_root / "val" / "Annotations"),
        (val_root / "test" / "JPEGImages", val_root / "test" / "Annotations"),
    ]
    for jpeg_root, ann_root in candidates:
        if _has_video_dirs(jpeg_root, ann_root):
            return jpeg_root, ann_root
    return None


def _resolve_sav_layout(val_root: Path) -> tuple[Path, Path] | None:
    candidates = [
        (val_root / "JPEGImages_24fps", val_root / "Annotations_6fps"),
        (
            val_root / "valid" / "JPEGImages_24fps",
            val_root / "valid" / "Annotations_6fps",
        ),
        (val_root / "val" / "JPEGImages_24fps", val_root / "val" / "Annotations_6fps"),
        (
            val_root / "test" / "JPEGImages_24fps",
            val_root / "test" / "Annotations_6fps",
        ),
    ]
    for jpeg_root, ann_root in candidates:
        if _has_video_dirs(jpeg_root, ann_root):
            return jpeg_root, ann_root
    return None


def has_supported_vos_layout(val_root: Path) -> bool:
    return (
        _resolve_standard_layout(val_root) is not None
        or _resolve_sav_layout(val_root) is not None
    )


def _shard_videos(videos: list[str], num_shards: int, shard_idx: int) -> list[str]:
    """Take every `num_shards`-th video starting at `shard_idx`.

    Strided slicing (not contiguous blocks) keeps each shard's workload balanced
    even when videos are ordered by length, and is stable regardless of how many
    videos exist. `num_shards <= 1` returns the list unchanged.
    """
    if num_shards <= 1:
        return videos
    if not 0 <= shard_idx < num_shards:
        raise ValueError(f"shard_idx={shard_idx} out of range for num_shards={num_shards}")
    return videos[shard_idx::num_shards]


def _load_cache(cache_path: Path | None) -> dict[str, dict]:
    """Load a per-video resume cache, or {} if absent/corrupt."""
    if cache_path is None or not Path(cache_path).is_file():
        return {}
    try:
        data = json.loads(Path(cache_path).read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_cache(cache_path: Path | None, per_video: dict[str, dict]) -> None:
    """Atomically persist per-video results so a crash resumes cleanly."""
    if cache_path is None:
        return
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    tmp.write_text(json.dumps(per_video, indent=2))
    tmp.replace(cache_path)


def _aggregate_per_video(per_video: dict[str, dict]) -> dict:
    """Aggregate cached per-video J/F into overall means.

    Matches the original object-micro-average: each object contributes equally,
    so videos are weighted by their object count (`n_objects`). Older entries
    without that field fall back to weight 1.
    """
    j_sum = f_sum = 0.0
    total = 0
    for v in per_video.values():
        n = int(v.get("n_objects", 1))
        j_sum += float(v["J"]) * n
        f_sum += float(v["F"]) * n
        total += n
    J_mean = j_sum / total if total else 0.0
    F_mean = f_sum / total if total else 0.0
    return {
        "per_video": per_video,
        "J_mean": J_mean,
        "F_mean": F_mean,
        "JF_mean": 0.5 * (J_mean + F_mean),
        "n_videos": len(per_video),
    }


class _DebugAccum:
    """Accumulates per-frame J split by whether the GT object is present.

    Separates the two inflation causes:
      - empty-GT frames score a free 1.0 (object absent, prediction also empty),
      - genuine high scores on present frames (e.g. train/eval overlap).
    A large gap between J_all and J_present points at empty-frame inflation; both
    being high points at memorization / data leakage instead.
    """

    def __init__(self) -> None:
        self.n_total = 0
        self.n_empty = 0
        self._j_present: list[float] = []
        self._f_present: list[float] = []

    def add(
        self,
        j_per_frame: np.ndarray,
        f_per_frame: np.ndarray,
        gt_empty: list[bool],
    ) -> None:
        em = np.asarray(gt_empty, dtype=bool)
        self.n_total += int(em.size)
        self.n_empty += int(em.sum())
        if (~em).any():
            self._j_present.append(float(j_per_frame[~em].mean()))
            self._f_present.append(float(f_per_frame[~em].mean()))

    def summary(self, j_all: float, f_all: float) -> str:
        frac = 100.0 * self.n_empty / self.n_total if self.n_total else 0.0
        j_present = f"{np.mean(self._j_present):.3f}" if self._j_present else "n/a"
        f_present = f"{np.mean(self._f_present):.3f}" if self._f_present else "n/a"
        return (
            f"[debug] frames={self.n_total} empty_gt={self.n_empty} ({frac:.1f}%) "
            f"J_all={j_all:.3f} J_present={j_present} "
            f"F_all={f_all:.3f} F_present={f_present}"
        )


def evaluate(
    predictor,
    val_root: Path,
    max_videos: int | None,
    cache_path: Path | None = None,
    offload_video_to_cpu: bool = True,
    offload_state_to_cpu: bool = True,
    debug: bool = False,
    num_shards: int = 1,
    shard_idx: int = 0,
    ann_root_override: Path | None = None,
) -> dict:
    sav_layout = _resolve_sav_layout(val_root)
    if sav_layout is not None:
        return _evaluate_sav_layout(
            predictor,
            sav_layout[0],
            sav_layout[1],
            max_videos,
            cache_path,
            offload_video_to_cpu,
            offload_state_to_cpu,
            debug,
            num_shards,
            shard_idx,
        )

    standard_layout = _resolve_standard_layout(val_root)
    if standard_layout is None:
        raise RuntimeError(
            f"Unsupported VOS layout at {val_root}. Expected DAVIS-style "
            "JPEGImages/Annotations, DAVIS JPEGImages/480p/Annotations/480p, "
            "or SA-V JPEGImages_24fps/Annotations_6fps."
        )
    jpeg_root, ann_root = standard_layout
    if ann_root_override is not None:
        # Some benchmarks ship the input frames and the dense GT in separate
        # trees (e.g. MOSEv2 valid: frames under valid/JPEGImages, but the
        # submission-format valid/Annotations holds only the first frame, while
        # the dense masks live at <root>/<video>/*.png). Point GT lookup at the
        # override; frames still come from the resolved jpeg_root.
        ann_root = ann_root_override
        if not _has_video_dirs(jpeg_root, ann_root):
            print(
                f"[validate] WARNING: --ann-root {ann_root} has no video dirs "
                f"matching frames in {jpeg_root}; scoring will likely skip all videos."
            )
    videos = sorted([d.name for d in jpeg_root.iterdir() if d.is_dir()])
    if max_videos is not None:
        videos = videos[:max_videos]
    videos = _shard_videos(videos, num_shards, shard_idx)

    # Resume: pre-load any per-video results from a prior (interrupted) run.
    per_video: dict[str, dict] = _load_cache(cache_path)

    for video in videos:
        if video in per_video:
            print(f"  {video}: cached, skipping")
            continue
        frames_dir = jpeg_root / video
        ann_dir = ann_root / video
        frames = sorted(frames_dir.glob("*.jpg"))
        if not frames:
            continue
        annotated_frames = [
            (idx, fp, ann_dir / (fp.stem + ".png"))
            for idx, fp in enumerate(frames)
            if (ann_dir / (fp.stem + ".png")).is_file()
        ]
        if not annotated_frames:
            continue
        # Score every annotated frame except the GT-seeded first one (and the last,
        # per DAVIS convention). If only the first frame is annotated, there is
        # nothing to score locally — this is the case for MOSE/LVOS/YTVOS val,
        # whose dense GT is held out for server submission. Skip rather than
        # emit a bogus ~1.0 from "scoring" the seed against itself.
        scored_frames = (
            annotated_frames[1:-1]
            if len(annotated_frames) > 2
            else annotated_frames[1:]
        )
        if not scored_frames:
            print(
                f"  {video}: only the seed frame is annotated "
                f"(annotated={len(annotated_frames)}, jpg={len(frames)}); "
                "no frames to score locally — skipping. This benchmark likely "
                "needs server submission (held-out GT)."
            )
            continue
        first_frame_idx, _, first_ann_path = annotated_frames[0]
        first_ann = _load_palette_mask(first_ann_path)
        obj_ids = [int(i) for i in np.unique(first_ann) if i != 0]
        if not obj_ids:
            continue

        state = predictor.init_state(
            str(frames_dir),
            offload_video_to_cpu=offload_video_to_cpu,
            offload_state_to_cpu=offload_state_to_cpu,
        )
        H, W = first_ann.shape

        # Seed every object with its GT mask on the first annotated frame.
        for obj_id in obj_ids:
            gt0 = _binary_obj_mask(first_ann, obj_id)
            predictor.add_new_mask(
                state,
                frame_idx=first_frame_idx,
                obj_id=obj_id,
                mask=torch.from_numpy(gt0).bool(),
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
        # `scored_frames` was computed above (seed frame excluded).
        video_J: list[float] = []
        video_F: list[float] = []
        dbg = _DebugAccum() if debug else None
        for oid in obj_ids:
            gt_seq = []
            pr_seq = []
            gt_empty = []
            for frame_idx, _, ann_path in scored_frames:
                gt_palette = _load_palette_mask(ann_path)
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
                gt_empty.append(not gt_bin.any())
            res = compute_sequence(np.stack(pr_seq), np.stack(gt_seq))
            video_J.append(res["J_mean"])
            video_F.append(res["F_mean"])
            if dbg is not None:
                dbg.add(res["J_per_frame"], res["F_per_frame"], gt_empty)

        vJ = float(np.mean(video_J))
        vF = float(np.mean(video_F))
        per_video[video] = {
            "J": vJ,
            "F": vF,
            "JF": 0.5 * (vJ + vF),
            "n_objects": len(video_J),
        }
        _save_cache(cache_path, per_video)
        print(f"  {video}: J={vJ:.3f} F={vF:.3f} JF={0.5 * (vJ + vF):.3f}")
        if dbg is not None:
            print(
                f"    {dbg.summary(vJ, vF)} "
                f"annotated_frames={len(annotated_frames)} jpg_frames={len(frames)} "
                f"scored_frames={len(scored_frames)}"
            )

    return _aggregate_per_video(per_video)


def _evaluate_sav_layout(
    predictor,
    jpeg_root: Path,
    ann_root: Path,
    max_videos: int | None,
    cache_path: Path | None = None,
    offload_video_to_cpu: bool = True,
    offload_state_to_cpu: bool = True,
    debug: bool = False,
    num_shards: int = 1,
    shard_idx: int = 0,
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
    videos = _shard_videos(videos, num_shards, shard_idx)

    per_video: dict[str, dict] = _load_cache(cache_path)

    for video in videos:
        if video in per_video:
            print(f"  {video}: cached, skipping")
            continue
        frames_dir = jpeg_root / video
        video_ann_dir = ann_root / video
        frames = sorted(frames_dir.glob("*.jpg"))
        if not frames:
            continue
        frame_lookup = {fp.stem: idx for idx, fp in enumerate(frames)}
        obj_dirs = sorted([d for d in video_ann_dir.iterdir() if d.is_dir()])
        if not obj_dirs:
            continue

        state = predictor.init_state(
            str(frames_dir),
            offload_video_to_cpu=offload_video_to_cpu,
            offload_state_to_cpu=offload_state_to_cpu,
        )
        eval_masks_by_obj: dict[int, list[Path]] = {}
        raw_mask_total = 0
        for obj_idx, obj_dir in enumerate(obj_dirs, start=1):
            masks = sorted(obj_dir.glob("*.png"))
            masks = [m for m in masks if m.stem in frame_lookup]
            if not masks:
                continue
            raw_mask_total += len(masks)
            first_mask = masks[0]
            # Exclude the GT-seeded first mask (and the last, DAVIS convention).
            # If only the seed exists for this object, there is nothing to score.
            scored = masks[1:-1] if len(masks) > 2 else masks[1:]
            if not scored:
                continue
            eval_masks_by_obj[obj_idx] = scored
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
        dbg = _DebugAccum() if debug else None
        for oid, mask_paths in eval_masks_by_obj.items():
            gt_seq = []
            pr_seq = []
            gt_empty = []
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
                gt_empty.append(not gt_bin.any())
            res = compute_sequence(np.stack(pr_seq), np.stack(gt_seq))
            video_J.append(res["J_mean"])
            video_F.append(res["F_mean"])
            if dbg is not None:
                dbg.add(res["J_per_frame"], res["F_per_frame"], gt_empty)

        if not video_J:
            continue
        vJ = float(np.mean(video_J))
        vF = float(np.mean(video_F))
        per_video[video] = {
            "J": vJ,
            "F": vF,
            "JF": 0.5 * (vJ + vF),
            "n_objects": len(video_J),
        }
        _save_cache(cache_path, per_video)
        print(f"  {video}: J={vJ:.3f} F={vF:.3f} JF={0.5 * (vJ + vF):.3f}")
        if dbg is not None:
            print(
                f"    {dbg.summary(vJ, vF)} "
                f"raw_masks={raw_mask_total} jpg_frames={len(frames)} "
                f"scored_masks={sum(len(v) for v in eval_masks_by_obj.values())}"
            )

    return _aggregate_per_video(per_video)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--val-root", required=True)
    parser.add_argument(
        "--ann-root",
        default=None,
        help="Override the directory holding per-video GT masks (<ann-root>/"
        "<video>/*.png), when dense GT lives in a different tree than the frames. "
        "Frames are still read from --val-root's JPEGImages. Use for MOSEv2 valid, "
        "whose valid/Annotations holds only the seed frame while the dense masks "
        "sit at <MOSE>/<video>/*.png.",
    )
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument(
        "--cache-json",
        default=None,
        help="Per-video resume cache; completed videos are skipped on re-run.",
    )
    parser.add_argument(
        "--no-offload",
        action="store_true",
        help="Keep all video frames and tracking state on GPU (faster, but OOMs "
        "on long videos / shared GPUs). Offloading to CPU is on by default.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print per-video diagnostics: empty-GT frame ratio and J restricted "
        "to frames where the object is present (J_present). A large J_all vs "
        "J_present gap means empty frames inflate the score; both high means "
        "likely train/eval overlap.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Split the video list into this many strided shards so multiple "
        "processes (one per GPU) can evaluate disjoint subsets in parallel. "
        "Give each shard its own --cache-json, then merge with tools.merge_vos_shards.",
    )
    parser.add_argument(
        "--shard-idx",
        type=int,
        default=0,
        help="Which shard this process evaluates (0..num_shards-1).",
    )
    args = parser.parse_args()

    device = pick_device()
    print(f"[validate] device={device}")
    predictor = build_efficienttam_video_predictor(
        config_file=args.config,
        ckpt_path=args.ckpt,
        device=str(device),
        mode="eval",
    )

    cache_path = Path(args.cache_json) if args.cache_json else None
    results = evaluate(
        predictor,
        Path(args.val_root),
        args.max_videos,
        cache_path,
        offload_video_to_cpu=not args.no_offload,
        offload_state_to_cpu=not args.no_offload,
        debug=args.debug,
        num_shards=args.num_shards,
        shard_idx=args.shard_idx,
        ann_root_override=Path(args.ann_root) if args.ann_root else None,
    )
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
