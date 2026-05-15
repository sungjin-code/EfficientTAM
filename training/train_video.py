"""Stage 2: video finetuning entry point.

Usage:
    python -m training.train_video \\
        --config training/configs/train_video_s.yaml \\
        --data-root /path/to/sav_or_davis_root \\
        --output-dir runs/video_s \\
        --init-from runs/image_s/image_final.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from efficient_track_anything.build_efficienttam import build_efficienttam
from training.data.prompts import PromptSampler
from training.data.video_dataset import VideoSegmentationDataset, collate_video_batch
from training.engine import (
    TrainState,
    load_checkpoint,
    save_checkpoint,
    train_one_epoch_video,
)
from training.losses import LossWeights, MultiStepLoss
from training.optim import WarmupCosineSchedule, build_optimizer


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--resume", default=None)
    p.add_argument(
        "--init-from", default=None, help="Stage-1 checkpoint to seed weights from."
    )
    p.add_argument("--overfit-one-batch", action="store_true")
    p.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Stop after N optimizer steps (overrides epoch budget).",
    )
    p.add_argument(
        "--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train_cfg = cfg["train"]
    model_cfg = cfg["model"]

    device = pick_device()
    print(f"[train_video] device={device}")

    overrides = list(model_cfg.get("overrides", []))
    overrides.append("++model.compile_image_encoder=false")

    model = build_efficienttam(
        config_file=model_cfg["config_file"],
        ckpt_path=None,
        device=str(device),
        mode="train",
        hydra_overrides_extra=overrides,
        apply_postprocessing=False,
    )
    if args.init_from is not None:
        load_checkpoint(model, args.init_from, weights_only_into_model=True)

    prompt_sampler = PromptSampler(
        mode=train_cfg.get("prompt_mode", "mixed"),
        max_neg_points=train_cfg.get("max_neg_points", 0),
        box_jitter_pct=train_cfg.get("box_jitter_pct", 0.05),
        seed=train_cfg.get("seed", 0),
    )

    dataset = VideoSegmentationDataset(
        root=args.data_root,
        image_size=train_cfg["image_size"],
        clip_len=train_cfg["clip_len"],
        prompt_sampler=prompt_sampler,
        correction_prob=train_cfg.get("correction_prob", 0.5),
        max_correction_frames=train_cfg.get("max_correction_frames", 2),
        stride_choices=tuple(train_cfg.get("stride_choices", [1, 2, 3])),
        scale_range=tuple(train_cfg.get("scale_range", [0.7, 1.0])),
        hflip_prob=train_cfg.get("hflip_prob", 0.5),
        brightness=train_cfg.get("brightness", 0.1),
        contrast=train_cfg.get("contrast", 0.03),
        saturation=train_cfg.get("saturation", 0.03),
        grayscale_prob=train_cfg.get("grayscale_prob", 0.05),
        affine_degree=train_cfg.get("affine_degree", 25.0),
        affine_shear=train_cfg.get("affine_shear", 20.0),
        objects_per_clip=train_cfg.get("objects_per_clip", 1),
        seed=train_cfg.get("seed", 0),
    )
    loader = DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 2),
        collate_fn=collate_video_batch,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
    )

    steps_per_epoch = max(1, len(loader))
    total_steps = train_cfg["epochs"] * steps_per_epoch

    optimizer = build_optimizer(
        model,
        backbone_lr=train_cfg["backbone_lr"],
        head_lr=train_cfg["head_lr"],
        layerwise_decay=train_cfg.get("layerwise_decay", 0.8),
        weight_decay=train_cfg.get("weight_decay", 0.1),
    )
    scheduler = WarmupCosineSchedule(
        optimizer,
        total_steps=total_steps,
        warmup_pct=train_cfg.get("warmup_pct", 0.05),
    )

    loss_fn = MultiStepLoss(
        LossWeights(
            focal=train_cfg.get("focal_w", 20.0),
            dice=train_cfg.get("dice_w", 1.0),
            iou=train_cfg.get("iou_w", 1.0),
            obj=train_cfg.get("obj_w", 1.0),
        )
    )

    state = TrainState()
    if args.resume is not None:
        load_checkpoint(
            model, args.resume, optimizer=optimizer, scheduler=scheduler, state=state
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(state.epoch, train_cfg["epochs"]):
        if args.max_steps is not None and state.step >= args.max_steps:
            break
        state.epoch = epoch
        train_one_epoch_video(
            model=model,
            loader=loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            prompt_sampler=prompt_sampler,
            device=device,
            state=state,
            grad_clip=train_cfg.get("grad_clip", 0.1),
            log_every=train_cfg.get("log_every", 5),
            overfit_one_batch=args.overfit_one_batch,
            max_steps=args.max_steps,
            precision=args.precision,
        )
        ckpt_path = out_dir / f"video_epoch_{epoch:04d}.pt"
        save_checkpoint(model, optimizer, scheduler, str(ckpt_path), state)
        print(f"[train_video] saved {ckpt_path}")

    final_path = out_dir / "video_final.pt"
    save_checkpoint(model, optimizer, scheduler, str(final_path), state)
    print(f"[train_video] done. final={final_path}")


if __name__ == "__main__":
    main()
