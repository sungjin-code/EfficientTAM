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
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image

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
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame_bgr = cap.read()
        if not ok:
            continue
        cv2.imwrite(str(out_dir / f"{frame_idx:05d}.jpg"), frame_bgr)
    cap.release()


def convert_sav(
    input_dir: str | Path,
    output_dir: str | Path,
    annotation_kind: str = "both",
    max_videos: int | None = None,
    annotation_stride: int = 4,
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

    for count, paths in enumerate(grouped.values(), start=1):
        if max_videos is not None and count > max_videos:
            break
        video_id, masks_by_frame = _collect_masks(paths)
        if not masks_by_frame:
            print(f"[prepare_sav] skipping {video_id}: no decoded masks")
            continue
        video_path = _find_video(raw_root, video_id)
        if video_path is None:
            print(f"[prepare_sav] skipping {video_id}: mp4 not found")
            continue

        frame_indices = sorted(masks_by_frame)
        video_frame_indices = [idx * annotation_stride for idx in frame_indices]
        image_dir = out_root / "JPEGImages" / video_id
        ann_dir = out_root / "Annotations" / video_id
        ann_dir.mkdir(parents=True, exist_ok=True)
        _write_frames(video_path, video_frame_indices, image_dir)
        for frame_idx, video_frame_idx in zip(frame_indices, video_frame_indices):
            Image.fromarray(masks_by_frame[frame_idx]).save(
                ann_dir / f"{video_frame_idx:05d}.png"
            )
        print(
            f"[prepare_sav] converted {video_id}: {len(frame_indices)} annotated frames "
            f"(stride={annotation_stride})"
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_sav(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        annotation_kind=args.annotation_kind,
        max_videos=args.max_videos,
        annotation_stride=args.annotation_stride,
    )


if __name__ == "__main__":
    main()
