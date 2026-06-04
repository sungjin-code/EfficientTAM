"""Stage 2: video finetuning entry point.

Usage:
    python -m training.train_video \\
        --config training/train_video_s \\
        --data-root /path/to/sav_or_davis_root \\
        --output-dir runs/video_s \\
        --init-from runs/image_s/image_final.pt \\
        train.batch_size=2   # optional Hydra-style overrides

`--config` is a Hydra config name under `efficient_track_anything/configs/`.
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import argparse
import os
import sys
import traceback
from pathlib import Path

import torch
from hydra import compose
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import efficient_track_anything  # noqa: F401  (initializes Hydra config module)
from efficient_track_anything.build_efficienttam import build_efficienttam
from training.distributed import broadcast_model, cleanup_distributed, init_distributed
from training.data.prompts import PromptSampler
from training.data.image_dataset import ImageSegmentationDataset
from training.data.video_dataset import (
    MixedVideoImageDataset,
    VideoSegmentationDataset,
    collate_video_batch,
)
from training.engine import (
    TrainState,
    load_checkpoint,
    save_checkpoint,
    save_training_artifact,
    train_one_epoch_video,
)
from training.losses import LossWeights, MultiStepLoss
from training.optim import build_optimizer, build_scheduler
from training.wandb_utils import load_dotenv, make_logger


def pick_device(local_rank: int = 0) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        required=True,
        help="Hydra config name under efficient_track_anything/configs "
        "(e.g. 'training/train_video_s').",
    )
    p.add_argument(
        "--data-root",
        default=os.environ.get("DATA_ROOT_VIDEO") or os.environ.get("DATA_ROOT"),
        help="Video dataset root. Defaults to $DATA_ROOT_VIDEO / $DATA_ROOT from `.env`.",
    )
    p.add_argument(
        "--image-data-root",
        default=os.environ.get("DATA_ROOT_IMAGE"),
        help="SA-1B-style image dataset root for Stage-2 10% image mixing.",
    )
    p.add_argument(
        "--output-dir",
        default=os.environ.get("OUTPUT_DIR_VIDEO") or os.environ.get("OUTPUT_DIR"),
        help="Where to write checkpoints. Defaults to $OUTPUT_DIR_VIDEO / $OUTPUT_DIR from `.env`.",
    )
    p.add_argument(
        "--resume",
        default=os.environ.get("RESUME_VIDEO") or os.environ.get("RESUME"),
        help="Path to a checkpoint to resume from. Defaults to $RESUME_VIDEO / $RESUME from `.env`.",
    )
    p.add_argument(
        "--init-from",
        default=os.environ.get("INIT_FROM_VIDEO") or os.environ.get("INIT_FROM"),
        help="Stage-1 checkpoint to seed weights from. Defaults to $INIT_FROM_VIDEO / $INIT_FROM from `.env`.",
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
    p.add_argument(
        "overrides",
        nargs="*",
        help="Hydra-style overrides, e.g. train.batch_size=4.",
    )
    return p.parse_args()


def _load_cfg(config_name: str, overrides: list[str]) -> dict:
    name = config_name
    if name.endswith(".yaml"):
        name = name[: -len(".yaml")]
    if name.startswith("training/configs/"):
        name = "configs/training/" + name[len("training/configs/") :]
    elif name.startswith("training/"):
        name = "configs/" + name
    cfg = compose(config_name=name, overrides=overrides)
    return OmegaConf.to_container(cfg, resolve=True)


def _drop_default_image_encoder_pretrain(overrides: list[str]) -> list[str]:
    return [o for o in overrides if "model.image_encoder.weights_path" not in o]


def main() -> None:
    dist_info = init_distributed()
    args = parse_args()
    missing = [
        name
        for name, val in (
            ("--data-root", args.data_root),
            ("--output-dir", args.output_dir),
        )
        if not val
    ]
    if missing:
        raise SystemExit(
            "[train_video] missing required path(s): "
            + ", ".join(missing)
            + " — pass on the CLI or set DATA_ROOT / OUTPUT_DIR in `.env`."
        )
    cfg = _load_cfg(args.config, args.overrides)
    train_cfg = cfg["train"]
    model_cfg = cfg["model"]
    effective_max_steps = args.max_steps
    if effective_max_steps is None:
        effective_max_steps = train_cfg.get("max_steps")

    device = pick_device(dist_info.local_rank)
    if dist_info.is_main:
        print(
            f"[train_video] device={device} distributed={dist_info.enabled} "
            f"world_size={dist_info.world_size}"
        )
    torch.manual_seed(train_cfg.get("seed", 0))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(train_cfg.get("seed", 0))

    overrides = list(model_cfg.get("overrides", []))
    if args.init_from is not None or args.resume is not None:
        overrides = _drop_default_image_encoder_pretrain(overrides)
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
    broadcast_model(model, dist_info)

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
    image_mix_prob = float(train_cfg.get("image_mix_prob", 0.0))
    if image_mix_prob > 0:
        if not args.image_data_root:
            raise SystemExit(
                "[train_video] train.image_mix_prob > 0 requires "
                "--image-data-root or DATA_ROOT_IMAGE."
            )
        image_dataset = ImageSegmentationDataset(
            root=args.image_data_root,
            image_size=train_cfg["image_size"],
            prompt_sampler=prompt_sampler,
            scale_range=tuple(train_cfg.get("image_scale_range", [0.5, 1.0])),
            hflip_prob=train_cfg.get("hflip_prob", 0.5),
            brightness=train_cfg.get("brightness", 0.1),
            contrast=train_cfg.get("contrast", 0.03),
            saturation=train_cfg.get("saturation", 0.03),
            grayscale_prob=train_cfg.get("grayscale_prob", 0.05),
            affine_degree=train_cfg.get("affine_degree", 25.0),
            affine_shear=train_cfg.get("affine_shear", 20.0),
            objects_per_image=train_cfg.get("objects_per_clip", 1),
        )
        dataset = MixedVideoImageDataset(
            video_dataset=dataset,
            image_dataset=image_dataset,
            image_mix_prob=image_mix_prob,
            seed=train_cfg.get("seed", 0),
        )
    sampler = (
        DistributedSampler(dataset, shuffle=True, drop_last=True)
        if dist_info.enabled
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=train_cfg.get("num_workers", 2),
        collate_fn=collate_video_batch,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
    )

    accumulation_steps = max(1, int(train_cfg.get("accumulation_steps", 1)))
    steps_per_epoch = max(1, len(loader))
    optimizer_steps_per_epoch = max(
        1, (steps_per_epoch + accumulation_steps - 1) // accumulation_steps
    )
    total_steps = effective_max_steps or (
        train_cfg["epochs"] * optimizer_steps_per_epoch
    )
    micro_global_batch_size = train_cfg["batch_size"] * dist_info.world_size
    effective_global_batch_size = micro_global_batch_size * accumulation_steps

    optimizer = build_optimizer(
        model,
        backbone_lr=train_cfg["backbone_lr"],
        head_lr=train_cfg["head_lr"],
        layerwise_decay=train_cfg.get("layerwise_decay", 0.8),
        weight_decay=train_cfg.get("weight_decay", 0.1),
    )
    scheduler = build_scheduler(optimizer, train_cfg, total_steps=total_steps)

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
    if dist_info.is_main:
        logger.init(
            wandb_cfg=cfg.get("wandb"),
            run_config={
                "stage": "video",
                "config_name": args.config,
                "config_overrides": args.overrides,
                "output_dir": str(out_dir),
                "device": str(device),
                "precision": args.precision,
                "max_steps": effective_max_steps,
                "init_from": args.init_from,
                "distributed": dist_info.enabled,
                "world_size": dist_info.world_size,
                "per_gpu_batch_size": train_cfg["batch_size"],
                "micro_global_batch_size": micro_global_batch_size,
                "accumulation_steps": accumulation_steps,
                "global_batch_size": effective_global_batch_size,
                "model": model_cfg,
                "train": train_cfg,
            },
            default_run_name=f"video_{out_dir.name}",
        )

    latest_path = out_dir / "video_latest.pt"
    artifact_path = out_dir / "video_training_artifact.json"
    last_epoch_result: dict = {}
    exit_status = "ok"
    try:
        for epoch in range(state.epoch, train_cfg["epochs"]):
            if effective_max_steps is not None and state.step >= effective_max_steps:
                break
            if sampler is not None:
                sampler.set_epoch(epoch)
            state.epoch = epoch
            last_epoch_result = train_one_epoch_video(
                model=model,
                loader=loader,
                optimizer=optimizer,
                scheduler=scheduler,
                loss_fn=loss_fn,
                prompt_sampler=prompt_sampler,
                device=device,
                state=state,
                grad_clip=train_cfg.get("grad_clip", 0.1),
                accumulation_steps=accumulation_steps,
                log_every=train_cfg.get("log_every", 5),
                overfit_one_batch=args.overfit_one_batch,
                max_steps=effective_max_steps,
                precision=args.precision,
                logger=logger,
                num_correction_points_per_frame=train_cfg.get(
                    "num_correction_points_per_frame", 1
                ),
                add_correction_frames_as_cond=train_cfg.get(
                    "add_correction_frames_as_cond", False
                ),
            )
            if dist_info.is_main:
                ckpt_path = out_dir / f"video_epoch_{epoch:04d}.pt"
                save_checkpoint(model, optimizer, scheduler, str(ckpt_path), state)
                save_checkpoint(model, optimizer, scheduler, str(latest_path), state)
                print(f"[train_video] saved {ckpt_path}")
                save_training_artifact(
                    artifact_path,
                    stage="video",
                    status="running",
                    state=state,
                    total_steps=total_steps,
                    output_dir=out_dir,
                    latest_checkpoint=latest_path,
                    epoch_checkpoint=ckpt_path,
                    metrics=last_epoch_result,
                    extra={
                        "config_name": args.config,
                        "data_root": args.data_root,
                        "image_data_root": args.image_data_root,
                        "init_from": args.init_from,
                        "precision": args.precision,
                    },
                )
                logger.log({"checkpoint/epoch": epoch}, step=state.step)
    except KeyboardInterrupt:
        exit_status = "interrupted"
        print("[train_video] KeyboardInterrupt — saving emergency checkpoint")
    except Exception:
        exit_status = "crashed"
        print(
            "[train_video] FATAL exception — saving emergency checkpoint",
            file=sys.stderr,
        )
        traceback.print_exc()
    finally:
        if exit_status != "ok":
            if dist_info.is_main:
                try:
                    emergency = out_dir / "video_interrupt.pt"
                    save_checkpoint(model, optimizer, scheduler, str(emergency), state)
                    save_checkpoint(
                        model, optimizer, scheduler, str(latest_path), state
                    )
                    print(f"[train_video] emergency checkpoint saved at {emergency}")
                    save_training_artifact(
                        artifact_path,
                        stage="video",
                        status=exit_status,
                        state=state,
                        total_steps=total_steps,
                        output_dir=out_dir,
                        latest_checkpoint=latest_path,
                        interrupt_checkpoint=emergency,
                        metrics=last_epoch_result,
                        extra={
                            "config_name": args.config,
                            "data_root": args.data_root,
                            "image_data_root": args.image_data_root,
                            "init_from": args.init_from,
                            "precision": args.precision,
                        },
                    )
                except Exception:
                    print("[train_video] emergency save FAILED", file=sys.stderr)
                    traceback.print_exc()

    if exit_status == "ok":
        final_path = out_dir / "video_final.pt"
        if dist_info.is_main:
            save_checkpoint(model, optimizer, scheduler, str(final_path), state)
            save_checkpoint(model, optimizer, scheduler, str(latest_path), state)
            print(f"[train_video] done. final={final_path}")
            save_training_artifact(
                artifact_path,
                stage="video",
                status=exit_status,
                state=state,
                total_steps=total_steps,
                output_dir=out_dir,
                latest_checkpoint=latest_path,
                final_checkpoint=final_path,
                metrics=last_epoch_result,
                extra={
                    "config_name": args.config,
                    "data_root": args.data_root,
                    "image_data_root": args.image_data_root,
                    "init_from": args.init_from,
                    "precision": args.precision,
                },
            )
    else:
        final_path = latest_path

    if dist_info.is_main:
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
    cleanup_distributed(dist_info)
    if exit_status == "crashed":
        sys.exit(1)


if __name__ == "__main__":
    main()
