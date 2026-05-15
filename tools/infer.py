"""Batch (headless) inference for EfficientTAM.

Modes:
    --mode image   single image + prompt JSON -> PNG mask
    --mode video   directory of JPEG frames + frame-0 prompt JSON -> directory of PNG masks

Prompt JSON schema (image mode):
    {"points": [[x, y], ...], "labels": [1, 0, ...]}    or
    {"box": [x0, y0, x1, y1]}                            or
    {"mask": "/path/to/binary_mask.png"}

Prompt JSON schema (video mode):
    {"objects": [
        {"obj_id": 1, "frame_idx": 0, "points": [[x,y]], "labels": [1]},
        {"obj_id": 2, "frame_idx": 0, "box":   [x0,y0,x1,y1]},
        {"obj_id": 3, "frame_idx": 0, "mask":  "/path/to/m.png"}
    ]}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from efficient_track_anything.build_efficienttam import (
    build_efficienttam,
    build_efficienttam_video_predictor,
)
from efficient_track_anything.efficienttam_image_predictor import (
    EfficientTAMImagePredictor,
)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)


def run_image(args: argparse.Namespace, device: torch.device) -> None:
    model = build_efficienttam(args.config, ckpt_path=args.ckpt, device=str(device))
    predictor = EfficientTAMImagePredictor(model)

    img = np.array(Image.open(args.input).convert("RGB"))
    predictor.set_image(img)

    prompt = json.loads(Path(args.prompt).read_text())
    point_coords = (
        np.array(prompt["points"], dtype=np.float32) if "points" in prompt else None
    )
    point_labels = (
        np.array(prompt["labels"], dtype=np.int32) if "labels" in prompt else None
    )
    box = np.array(prompt["box"], dtype=np.float32) if "box" in prompt else None
    mask_in = None
    if "mask" in prompt:
        mask_in = np.array(Image.open(prompt["mask"]).convert("L")) > 0

    masks, iou_pred, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=box,
        mask_input=mask_in[None] if mask_in is not None else None,
        multimask_output=True,
    )
    best = int(np.argmax(iou_pred))
    _save_mask(masks[best], Path(args.output))
    print(f"[infer/image] best_iou={float(iou_pred[best]):.3f} -> {args.output}")


def run_video(args: argparse.Namespace, device: torch.device) -> None:
    predictor = build_efficienttam_video_predictor(
        config_file=args.config,
        ckpt_path=args.ckpt,
        device=str(device),
        mode="eval",
    )
    state = predictor.init_state(args.input)
    prompt = json.loads(Path(args.prompt).read_text())

    for obj in prompt["objects"]:
        obj_id = int(obj["obj_id"])
        frame_idx = int(obj.get("frame_idx", 0))
        if "mask" in obj:
            m = np.array(Image.open(obj["mask"]).convert("L")) > 0
            predictor.add_new_mask(
                state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                mask=torch.from_numpy(m).bool(),
            )
        else:
            points = obj.get("points")
            labels = obj.get("labels")
            box = obj.get("box")
            predictor.add_new_points_or_box(
                state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                points=points,
                labels=labels,
                box=box,
            )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
        for i, oid in enumerate(obj_ids):
            m = (mask_logits[i] > 0).cpu().numpy().astype(np.uint8)
            if m.ndim == 3:
                m = m[0]
            _save_mask(m, out_dir / f"obj{oid:02d}_frame{frame_idx:05d}.png")
    print(f"[infer/video] wrote masks to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("image", "video"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument(
        "--input", required=True, help="Image path or video frame directory."
    )
    parser.add_argument("--prompt", required=True, help="Prompt JSON file.")
    parser.add_argument(
        "--output",
        required=True,
        help="Output PNG (image mode) or directory (video mode).",
    )
    args = parser.parse_args()

    device = pick_device()
    print(f"[infer] mode={args.mode} device={device}")
    if args.mode == "image":
        run_image(args, device)
    else:
        run_video(args, device)


if __name__ == "__main__":
    main()
