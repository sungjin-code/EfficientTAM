"""Stage 1: image pretraining entry point.

Usage:
    python -m training.train_image \\
        --config training/train_image_s \\
        --data-root /path/to/sa1b_style_root \\
        --output-dir runs/image_s \\
        train.batch_size=4 train.epochs=10   # optional Hydra-style overrides

`--config` is a Hydra config name resolved under
`efficient_track_anything/configs/` (the package's Hydra config module). The
config has a `model` block (pointing at an inference config under
`efficient_track_anything/configs/`) and a `train` block with all
hyperparameters. Any positional CLI args are treated as Hydra-style overrides
(e.g. `train.batch_size=8`).
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

import torch
from hydra import compose
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

import efficient_track_anything  # noqa: F401  (initializes Hydra config module)
from efficient_track_anything.build_efficienttam import build_efficienttam
from training.data.image_dataset import ImageSegmentationDataset, collate_image_batch
from training.data.prompts import PromptSampler
from training.engine import (
    TrainState,
    load_checkpoint,
    save_checkpoint,
    train_one_epoch_image,
)
from training.losses import LossWeights, MultiStepLoss
from training.optim import WarmupCosineSchedule, build_optimizer
from training.wandb_utils import load_dotenv, make_logger


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    # Load `.env` before reading argparse defaults so env-derived fallbacks
    # for paths (data-root, init-from, resume, output-dir) are visible.
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        required=True,
        help="Hydra config name under efficient_track_anything/configs "
        "(e.g. 'training/train_image_s').",
    )
    p.add_argument(
        "--data-root",
        default=os.environ.get("DATA_ROOT_IMAGE") or os.environ.get("DATA_ROOT"),
        help="Image dataset root. Defaults to $DATA_ROOT_IMAGE / $DATA_ROOT from `.env`.",
    )
    p.add_argument(
        "--output-dir",
        default=os.environ.get("OUTPUT_DIR_IMAGE") or os.environ.get("OUTPUT_DIR"),
        help="Where to write checkpoints. Defaults to $OUTPUT_DIR_IMAGE / $OUTPUT_DIR from `.env`.",
    )
    p.add_argument(
        "--resume",
        default=os.environ.get("RESUME_IMAGE") or os.environ.get("RESUME"),
        help="Path to a checkpoint to resume from. Defaults to $RESUME_IMAGE / $RESUME from `.env`.",
    )
    p.add_argument(
        "--init-from",
        default=os.environ.get("INIT_FROM_IMAGE") or os.environ.get("INIT_FROM"),
        help="Path to a checkpoint to seed weights only. Defaults to $INIT_FROM_IMAGE / $INIT_FROM from `.env`.",
    )
    p.add_argument("--overfit-one-batch", action="store_true")
    p.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Stop after N optimizer steps (overrides epoch budget). Paper uses 300k.",
    )
    p.add_argument(
        "--precision",
        choices=("auto", "bf16", "fp16", "fp32"),
        default="auto",
        help="Mixed-precision mode. `auto` picks bf16 on CUDA, fp16 on MPS, fp32 on CPU.",
    )
    p.add_argument(
        "overrides",
        nargs="*",
        help="Hydra-style overrides, e.g. train.batch_size=8.",
    )
    return p.parse_args()


def _load_cfg(config_name: str, overrides: list[str]) -> dict:
    # Strip .yaml suffix / leading path noise so users can pass either
    # 'training/train_image_s' or 'training/train_image_s.yaml'.
    name = config_name
    if name.endswith(".yaml"):
        name = name[: -len(".yaml")]
    cfg = compose(config_name=name, overrides=overrides)
    return OmegaConf.to_container(cfg, resolve=True)


def main() -> None:
    args = parse_args()
    missing = [
        name
        for name, val in (("--data-root", args.data_root), ("--output-dir", args.output_dir))
        if not val
    ]
    if missing:
        raise SystemExit(
            "[train_image] missing required path(s): "
            + ", ".join(missing)
            + " — pass on the CLI or set DATA_ROOT / OUTPUT_DIR in `.env`."
        )
    cfg = _load_cfg(args.config, args.overrides)
    train_cfg = cfg["train"]
    model_cfg = cfg["model"]

    device = pick_device()
    print(f"[train_image] device={device}")

    overrides = list(model_cfg.get("overrides", []))
    # Training never wants compile (graph recompilation on shape changes).
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

    dataset = ImageSegmentationDataset(
        root=args.data_root,
        image_size=train_cfg["image_size"],
        prompt_sampler=prompt_sampler,
        scale_range=tuple(train_cfg.get("scale_range", [0.5, 1.0])),
        hflip_prob=train_cfg.get("hflip_prob", 0.5),
        brightness=train_cfg.get("brightness", 0.1),
        contrast=train_cfg.get("contrast", 0.03),
        saturation=train_cfg.get("saturation", 0.03),
        grayscale_prob=train_cfg.get("grayscale_prob", 0.05),
        affine_degree=train_cfg.get("affine_degree", 25.0),
        affine_shear=train_cfg.get("affine_shear", 20.0),
        objects_per_image=train_cfg.get("objects_per_image", 1),
    )
    loader = DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 4),
        collate_fn=collate_image_batch,
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

    logger = make_logger()
    logger.init(
        wandb_cfg=cfg.get("wandb"),
        run_config={
            "stage": "image",
            "config_name": args.config,
            "config_overrides": args.overrides,
            "output_dir": str(out_dir),
            "device": str(device),
            "precision": args.precision,
            "max_steps": args.max_steps,
            "model": model_cfg,
            "train": train_cfg,
        },
        default_run_name=f"image_{out_dir.name}",
    )

    latest_path = out_dir / "image_latest.pt"
    last_epoch_result: dict = {}
    exit_status = "ok"
    try:
        for epoch in range(state.epoch, train_cfg["epochs"]):
            if args.max_steps is not None and state.step >= args.max_steps:
                break
            state.epoch = epoch
            last_epoch_result = train_one_epoch_image(
                model=model,
                loader=loader,
                optimizer=optimizer,
                scheduler=scheduler,
                loss_fn=loss_fn,
                prompt_sampler=prompt_sampler,
                device=device,
                state=state,
                grad_clip=train_cfg.get("grad_clip", 0.1),
                log_every=train_cfg.get("log_every", 20),
                overfit_one_batch=args.overfit_one_batch,
                max_steps=args.max_steps,
                precision=args.precision,
                logger=logger,
            )
            ckpt_path = out_dir / f"image_epoch_{epoch:04d}.pt"
            save_checkpoint(model, optimizer, scheduler, str(ckpt_path), state)
            save_checkpoint(model, optimizer, scheduler, str(latest_path), state)
            print(f"[train_image] saved {ckpt_path}")
            logger.log({"checkpoint/epoch": epoch}, step=state.step)
    except KeyboardInterrupt:
        exit_status = "interrupted"
        print("[train_image] KeyboardInterrupt — saving emergency checkpoint")
    except Exception:
        exit_status = "crashed"
        print("[train_image] FATAL exception — saving emergency checkpoint", file=sys.stderr)
        traceback.print_exc()
    finally:
        if exit_status != "ok":
            try:
                emergency = out_dir / "image_interrupt.pt"
                save_checkpoint(model, optimizer, scheduler, str(emergency), state)
                save_checkpoint(model, optimizer, scheduler, str(latest_path), state)
                print(f"[train_image] emergency checkpoint saved at {emergency}")
            except Exception:
                print("[train_image] emergency save FAILED", file=sys.stderr)
                traceback.print_exc()

    if exit_status == "ok":
        final_path = out_dir / "image_final.pt"
        save_checkpoint(model, optimizer, scheduler, str(final_path), state)
        save_checkpoint(model, optimizer, scheduler, str(latest_path), state)
        print(f"[train_image] done. final={final_path}")
    else:
        final_path = latest_path

    logger.summary(
        {
            "result/status": exit_status,
            "result/final_checkpoint": str(final_path),
            "result/steps": state.step,
            "result/epochs": state.epoch + 1,
            **{f"result/{k}": v for k, v in last_epoch_result.items()},
        }
    )
    logger.finish()
    if exit_status == "crashed":
        sys.exit(1)


if __name__ == "__main__":
    main()
