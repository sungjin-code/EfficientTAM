"""Convert raw SA-V mp4/json files into the Stage-2 video folder layout.

Expected raw SA-V files include mp4 videos and per-video annotation JSON files
such as `{video_id}_manual.json` and `{video_id}_auto.json`. The output matches
`training.data.video_dataset.VideoSegmentationDataset`:

    {output_dir}/JPEGImages/{video_id}/{frame_idx:05d}.jpg
    {output_dir}/Annotations/{video_id}/{frame_idx:05d}.png
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
from typing import Iterable

# Pin per-process math/codec threads to 1 so the worker-pool size is the only
# source of parallelism. Without this, each worker's OpenCV/BLAS spawns its own
# threads and the pool saturates every core, defeating the worker cap. Must be
# set before cv2/numpy import so BLAS/OpenMP read them.
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import cv2
import numpy as np
from PIL import Image

cv2.setNumThreads(1)

try:
    from pycocotools import mask as mask_utils
except ImportError as exc:
    raise SystemExit(
        "pycocotools is required for SA-V RLE decoding. "
        "Install it with `pip install pycocotools`."
    ) from exc


def _find_video(raw_root: Path, video_id: str) -> Path | None:
    candidates = list(raw_root.rglob(f"{video_id}.mp4"))
    if candidates:
        return candidates[0]
    candidates = list(raw_root.rglob(f"{video_id}*.mp4"))
    return candidates[0] if candidates else None


def _annotation_files(raw_root: Path, annotation_kind: str) -> list[Path]:
    patterns = {
        "manual": ["*_manual.json"],
        "auto": ["*_auto.json"],
        "both": ["*_manual.json", "*_auto.json"],
    }[annotation_kind]
    out: list[Path] = []
    for pattern in patterns:
        out.extend(raw_root.rglob(pattern))
    return sorted(out)


def _as_frame_rles(value) -> list:
    if value is None:
        return []
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return value
    return []


def _decode_rle(rle) -> np.ndarray | None:
    if not rle:
        return None
    decoded = mask_utils.decode(rle)
    if decoded.ndim == 3:
        decoded = decoded[..., 0]
    return decoded.astype(bool)


def _collect_masks(
    annotation_paths: Iterable[Path],
) -> tuple[str, dict[int, np.ndarray]]:
    video_id = ""
    object_map: dict[int, int] = {}
    frames: dict[int, np.ndarray] = {}

    for ann_path in annotation_paths:
        with ann_path.open("r") as f:
            ann = json.load(f)
        fallback_video_id = ann_path.stem.removesuffix("_manual").removesuffix("_auto")
        video_id = video_id or ann.get("video_id") or fallback_video_id
        masklets = ann.get("masklet", [])
        masklet_ids = ann.get("masklet_id") or list(
            range(len(masklets[0]) if masklets else 0)
        )
        if isinstance(masklet_ids, int):
            masklet_ids = [masklet_ids]

        for frame_idx, frame_value in enumerate(masklets):
            rles = _as_frame_rles(frame_value)
            if not rles:
                continue
            canvas = frames.get(frame_idx)
            for local_idx, rle in enumerate(rles):
                if local_idx >= len(masklet_ids):
                    continue
                raw_obj_id = int(masklet_ids[local_idx])
                if raw_obj_id not in object_map:
                    object_map[raw_obj_id] = len(object_map) + 1
                mask = _decode_rle(rle)
                if mask is None:
                    continue
                if canvas is None:
                    canvas = np.zeros(mask.shape, dtype=np.uint16)
                canvas[mask] = object_map[raw_obj_id]
            if canvas is not None:
                frames[frame_idx] = canvas

    if not video_id:
        raise RuntimeError("Could not infer video_id from SA-V annotations")
    return video_id, frames


def _write_frames(video_path: Path, frame_indices: list[int], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame_bgr = cap.read()
        if not ok:
            continue
        cv2.imwrite(str(out_dir / f"{frame_idx:05d}.jpg"), frame_bgr)
    cap.release()


def _worker_init() -> None:
    # Re-apply for the 'spawn' start method, where the module-level call does
    # not run in the child. Harmless under 'fork'.
    cv2.setNumThreads(1)


def _process_one(task: tuple) -> tuple[str, str]:
    """Worker: convert one video. Returns (video_id, status)."""
    video_id, ann_paths_str, raw_root_str, out_root_str, annotation_stride = task
    raw_root = Path(raw_root_str)
    out_root = Path(out_root_str)

    ann_dir = out_root / "Annotations" / video_id
    # Resume: Annotations dir exists and is non-empty → already done
    if ann_dir.exists() and any(ann_dir.iterdir()):
        return video_id, "skipped"

    ann_paths = [Path(p) for p in ann_paths_str]
    video_id_actual, masks_by_frame = _collect_masks(ann_paths)
    if not masks_by_frame:
        return video_id, "no_masks"

    video_path = _find_video(raw_root, video_id)
    if video_path is None:
        return video_id, "no_mp4"

    frame_indices = sorted(masks_by_frame)
    video_frame_indices = [idx * annotation_stride for idx in frame_indices]

    ann_dir.mkdir(parents=True, exist_ok=True)
    _write_frames(video_path, video_frame_indices, out_root / "JPEGImages" / video_id)
    for frame_idx, video_frame_idx in zip(frame_indices, video_frame_indices):
        Image.fromarray(masks_by_frame[frame_idx]).save(
            ann_dir / f"{video_frame_idx:05d}.png"
        )

    return video_id_actual, str(len(frame_indices))


def convert_sav(
    input_dir: str | Path,
    output_dir: str | Path,
    annotation_kind: str = "both",
    max_videos: int | None = None,
    annotation_stride: int = 4,
    num_workers: int = 0,
    cpu_fraction: float = 0.6,
) -> None:
    raw_root = Path(input_dir)
    out_root = Path(output_dir)
    ann_paths = _annotation_files(raw_root, annotation_kind)
    grouped: dict[str, list[Path]] = {}
    for path in ann_paths:
        stem = path.stem
        video_id = stem.removesuffix("_manual").removesuffix("_auto")
        grouped.setdefault(video_id, []).append(path)

    if not grouped:
        raise RuntimeError(f"No SA-V annotation JSON files found under {raw_root}")

    if num_workers > 0:
        workers = num_workers
    else:
        _getaffinity = getattr(os, "sched_getaffinity", None)
        available = len(_getaffinity(0)) if _getaffinity else (os.cpu_count() or 1)
        workers = max(1, int(available * cpu_fraction))
    tasks = [
        (vid, [str(p) for p in paths], str(raw_root), str(out_root), annotation_stride)
        for vid, paths in grouped.items()
    ]
    if max_videos is not None:
        tasks = tasks[:max_videos]

    total = len(tasks)
    print(f"[prepare_sav] {total} videos to process, {workers} workers", flush=True)

    done = skipped = errors = 0
    with mp.Pool(workers, initializer=_worker_init) as pool:
        for video_id, status in pool.imap_unordered(_process_one, tasks, chunksize=4):
            if status == "skipped":
                skipped += 1
                print(f"[prepare_sav] already done: {video_id} [{done + skipped + errors}/{total}]", flush=True)
            elif status in ("no_masks", "no_mp4"):
                errors += 1
                print(f"[prepare_sav] skipping {video_id}: {status}", flush=True)
            else:
                done += 1
                print(
                    f"[prepare_sav] converted {video_id}: {status} annotated frames "
                    f"(stride={annotation_stride}) [{done + skipped + errors}/{total}]",
                    flush=True,
                )

    print(
        f"[prepare_sav] done — converted {done}, already done {skipped}, "
        f"skipped (no data) {errors}, total {total}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert raw SA-V data for Stage 2")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--annotation-kind",
        choices=("manual", "auto", "both"),
        default="both",
    )
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument(
        "--annotation-stride",
        type=int,
        default=4,
        help="Video-frame stride between SA-V annotations. Official SA-V is 6 fps annotations on 24 fps videos, so the default is 4.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Absolute number of parallel worker processes. "
        "0 = derive from --cpu-fraction (default).",
    )
    parser.add_argument(
        "--cpu-fraction",
        type=float,
        default=0.6,
        help="Fraction of available CPU cores to use when --workers is 0. "
        "Default 0.6 (60%%).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_sav(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        annotation_kind=args.annotation_kind,
        max_videos=args.max_videos,
        annotation_stride=args.annotation_stride,
        num_workers=args.workers,
        cpu_fraction=args.cpu_fraction,
    )


if __name__ == "__main__":
    main()
