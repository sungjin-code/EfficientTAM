"""Training engine: forward_clip + train_one_epoch_{image,video}.

The clip forward runs all T frames through the image encoder in one batch (for
throughput), then iterates per-frame calling `_track_step` directly so we get
the full sam_outputs tuple (which includes object_score_logits — the public
`track_step` drops it during training). Memory bookkeeping mirrors the
inference predictor.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, Optional

import torch
import torch.nn.functional as F

from training.data.prompts import PromptSampler
from training.losses import MultiStepLoss
from training.optim import WarmupCosineSchedule, clip_grad_norm
from training.wandb_utils import WandbLogger


@dataclass
class ClipOutput:
    total_loss: torch.Tensor
    per_frame_metrics: list[dict]
    last_pred_logits: torch.Tensor  # [B, 1, H, W] high-res, useful for inspection


def _make_point_inputs(coords: torch.Tensor, labels: torch.Tensor) -> Optional[dict]:
    """Wrap (coords, labels) into the dict shape `_track_step` expects.

    Returns None if every label in the batch is the padding sentinel (-1).
    """
    if (labels >= 0).any():
        return {"point_coords": coords, "point_labels": labels}
    return None


def _store_frame(output_dict: dict, t: int, is_cond: bool, current_out: dict) -> None:
    bucket = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
    output_dict[bucket][t] = current_out


def forward_clip(
    model: torch.nn.Module,
    frames: torch.Tensor,  # [B, T, 3, H, W] (normalized)
    gt_masks: torch.Tensor,  # [B, T, 1, H, W] (binary)
    has_object: torch.Tensor,  # [B, T]
    frame0_point_inputs: Optional[
        dict
    ],  # {"point_coords":[B,N,2], "point_labels":[B,N]}
    correction_frames: list[
        list[int]
    ],  # per-clip list of frame indices to add correction click
    prompt_sampler: PromptSampler,
    loss_fn: MultiStepLoss,
    run_mem_encoder: bool,
) -> ClipOutput:
    B, T = frames.shape[:2]
    device = frames.device
    H = frames.shape[-2]

    # 1. Image encoder over all T*B frames in one batch.
    flat_frames = frames.flatten(0, 1)  # [B*T, 3, H, W]
    backbone_out = model.forward_image(flat_frames)
    _, vision_feats, vision_pos, feat_sizes = model._prepare_backbone_features(
        backbone_out
    )
    # vision_feats / vision_pos: list of [HW, B*T, C]
    last_feat = vision_feats[-1]  # [HW, B*T, C]
    last_pos = vision_pos[-1]  # [HW, B*T, C]
    HW = last_feat.shape[0]
    C = last_feat.shape[-1]

    # 2. Per-frame loop. Keep an output_dict that mirrors the inference predictor.
    output_dict: dict = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
    total_loss = frames.new_zeros(())
    per_frame: list[dict] = []
    last_high_res: Optional[torch.Tensor] = None

    for t in range(T):
        # Slice this frame's features out of the flat batch.
        # Flat indexing: when we flattened [B, T, ...] the layout was sample-major
        # i.e. flat[b*T + t]. We want the t-th frame for every sample.
        idx = torch.arange(B, device=device) * T + t
        feats_t = [vf[:, idx, :] for vf in vision_feats]
        pos_t = [vp[:, idx, :] for vp in vision_pos]

        # Decide prompt for this frame.
        is_init = t == 0
        point_inputs: Optional[dict] = None
        if is_init and frame0_point_inputs is not None:
            point_inputs = _make_point_inputs(
                frame0_point_inputs["point_coords"].to(device),
                frame0_point_inputs["point_labels"].to(device),
            )
        elif last_high_res is not None and any(t in cf for cf in correction_frames):
            # Sample one correction click per clip-in-batch (independent per sample).
            coord_list = []
            label_list = []
            for b in range(B):
                if t in correction_frames[b] and has_object[b, t] > 0:
                    sample = prompt_sampler.sample_correction(
                        gt_masks[b, t].float(),
                        last_high_res[b].detach(),
                    )
                    coord_list.append(sample.point_coords)
                    label_list.append(sample.point_labels)
                else:
                    coord_list.append(torch.zeros(1, 2))
                    label_list.append(torch.tensor([-1], dtype=torch.int32))
            coords = torch.stack(coord_list, dim=0).to(device)
            labels = torch.stack(label_list, dim=0).to(device)
            point_inputs = _make_point_inputs(coords, labels)

        # Forward through the track step. _track_step gives us the full sam_outputs.
        current_out, sam_outputs, _, _ = model._track_step(
            frame_idx=t,
            is_init_cond_frame=is_init,
            current_vision_feats=feats_t,
            current_vision_pos_embeds=pos_t,
            feat_sizes=feat_sizes,
            point_inputs=point_inputs,
            mask_inputs=None,
            output_dict=output_dict,
            num_frames=T,
            track_in_reverse=False,
            prev_sam_mask_logits=None,
        )
        (
            low_res_multimasks,
            high_res_multimasks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        ) = sam_outputs

        # Per-frame loss — supervise at high-res (image_size) as in SAM/SAM2.
        # `high_res_multimasks` is bilinear-upsampled from the H/4 head to (image_size, image_size).
        gt_full = gt_masks[:, t].float()
        frame_loss, metrics = loss_fn(
            mask_logits=high_res_multimasks,
            ious=ious,
            object_score_logits=object_score_logits,
            gt_mask=gt_full,
            has_object=has_object[:, t],
        )
        total_loss = total_loss + frame_loss
        per_frame.append(metrics)
        last_high_res = high_res_masks

        # Stash high-res masks + memory features into output_dict for downstream frames.
        current_out["pred_masks"] = low_res_masks
        current_out["pred_masks_high_res"] = high_res_masks
        current_out["obj_ptr"] = obj_ptr
        current_out["object_score_logits"] = object_score_logits

        is_correction = (point_inputs is not None) and not is_init
        is_cond = is_init or is_correction

        # Run memory encoder so future frames can attend (video stage only).
        if run_mem_encoder and T > 1:
            model._encode_memory_in_output(
                current_vision_feats=feats_t,
                feat_sizes=feat_sizes,
                point_inputs=point_inputs,
                run_mem_encoder=True,
                high_res_masks=high_res_masks,
                object_score_logits=object_score_logits,
                current_out=current_out,
            )

        _store_frame(output_dict, t, is_cond, current_out)

    total_loss = total_loss / T
    return ClipOutput(
        total_loss=total_loss,
        per_frame_metrics=per_frame,
        last_pred_logits=last_high_res,
    )


@dataclass
class TrainState:
    step: int = 0
    epoch: int = 0


def _autocast(device: torch.device, precision: str = "auto"):
    """Return an autocast context.

    `precision` may be `"auto"`, `"bf16"`, `"fp16"`, or `"fp32"`. The MPS path
    can be numerically unstable for some operations — use `"fp32"` if you hit
    NaNs.
    """
    if precision == "fp32":
        return torch.autocast(
            device_type=device.type if device.type != "mps" else "cpu", enabled=False
        )
    dtype_for = {"bf16": torch.bfloat16, "fp16": torch.float16}
    if precision in dtype_for:
        return torch.autocast(device_type=device.type, dtype=dtype_for[precision])
    # auto
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if device.type == "mps":
        return torch.autocast(device_type="mps", dtype=torch.float16)
    return torch.autocast(device_type="cpu", enabled=False)


def train_one_epoch_image(
    model: torch.nn.Module,
    loader: Iterable[dict],
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineSchedule,
    loss_fn: MultiStepLoss,
    prompt_sampler: PromptSampler,
    device: torch.device,
    state: TrainState,
    grad_clip: float = 0.1,
    log_every: int = 20,
    overfit_one_batch: bool = False,
    max_steps: int | None = None,
    precision: str = "auto",
    logger: Optional[WandbLogger] = None,
) -> dict:
    """One pass over the image loader. Each batch is a T=1 clip."""
    model.train()
    fixed_batch = None
    running = {"loss": 0.0, "n": 0}
    t_start = time.time()
    loss = torch.zeros((), device=device)
    for batch in loader:
        if max_steps is not None and state.step >= max_steps:
            break
        if overfit_one_batch:
            if fixed_batch is None:
                fixed_batch = {
                    k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }
                fixed_batch["point_inputs"] = {
                    k: v.to(device) for k, v in batch["point_inputs"].items()
                }
            batch = fixed_batch

        images = batch["image"].to(device)  # [N, 3, H, W]
        gt_masks = batch["gt_mask"].to(device)  # [N, 1, H, W]
        has_object = batch["has_object"].to(device)  # [N]
        point_inputs = {k: v.to(device) for k, v in batch["point_inputs"].items()}

        frames = images.unsqueeze(1)  # [N, 1, 3, H, W]
        gt = gt_masks.unsqueeze(1)  # [N, 1, 1, H, W]
        ho = has_object.unsqueeze(1)  # [N, 1]

        with _autocast(device, precision):
            out = forward_clip(
                model=model,
                frames=frames,
                gt_masks=gt,
                has_object=ho,
                frame0_point_inputs=point_inputs,
                correction_frames=[[] for _ in range(frames.shape[0])],
                prompt_sampler=prompt_sampler,
                loss_fn=loss_fn,
                run_mem_encoder=False,
            )
            loss = out.total_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            clip_grad_norm(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        state.step += 1

        step_loss = float(loss.detach().item())
        running["loss"] += step_loss
        running["n"] += 1
        if logger is not None and logger.enabled:
            logger.log(
                {
                    "train/loss": step_loss,
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/epoch": state.epoch,
                },
                step=state.step,
            )
        if state.step % log_every == 0:
            avg = running["loss"] / max(1, running["n"])
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t_start
            ips = running["n"] / max(elapsed, 1e-3)
            print(
                f"[image] step={state.step} epoch={state.epoch} loss={avg:.4f} "
                f"lr={lr:.2e} ips={ips:.2f}"
            )
            if logger is not None and logger.enabled:
                logger.log(
                    {"train/loss_avg": avg, "train/ips": ips}, step=state.step
                )
            running = {"loss": 0.0, "n": 0}
            t_start = time.time()
    return {"final_loss": float(loss.detach().item())}


def train_one_epoch_video(
    model: torch.nn.Module,
    loader: Iterable[dict],
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineSchedule,
    loss_fn: MultiStepLoss,
    prompt_sampler: PromptSampler,
    device: torch.device,
    state: TrainState,
    grad_clip: float = 0.1,
    log_every: int = 5,
    overfit_one_batch: bool = False,
    max_steps: int | None = None,
    precision: str = "auto",
    logger: Optional[WandbLogger] = None,
) -> dict:
    model.train()
    fixed_batch = None
    running = {"loss": 0.0, "n": 0}
    t_start = time.time()
    loss = torch.zeros((), device=device)
    for batch in loader:
        if max_steps is not None and state.step >= max_steps:
            break
        if overfit_one_batch:
            if fixed_batch is None:
                fixed_batch = {
                    k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }
                fixed_batch["frame0_point_inputs"] = {
                    k: v.to(device) for k, v in batch["frame0_point_inputs"].items()
                }
            batch = fixed_batch

        frames = batch["frames"].to(device)  # [B, T, 3, H, W]
        gt = batch["gt_masks"].to(device)  # [B, T, 1, H, W]
        ho = batch["has_object"].to(device)  # [B, T]
        point_inputs = {
            k: v.to(device) for k, v in batch["frame0_point_inputs"].items()
        }
        correction = batch["correction_frames"]

        with _autocast(device, precision):
            out = forward_clip(
                model=model,
                frames=frames,
                gt_masks=gt,
                has_object=ho,
                frame0_point_inputs=point_inputs,
                correction_frames=correction,
                prompt_sampler=prompt_sampler,
                loss_fn=loss_fn,
                run_mem_encoder=True,
            )
            loss = out.total_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            clip_grad_norm(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        state.step += 1

        step_loss = float(loss.detach().item())
        running["loss"] += step_loss
        running["n"] += 1
        if logger is not None and logger.enabled:
            logger.log(
                {
                    "train/loss": step_loss,
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/epoch": state.epoch,
                },
                step=state.step,
            )
        if state.step % log_every == 0:
            avg = running["loss"] / max(1, running["n"])
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t_start
            cps = running["n"] / max(elapsed, 1e-3)
            print(
                f"[video] step={state.step} epoch={state.epoch} loss={avg:.4f} "
                f"lr={lr:.2e} clips/s={cps:.2f}"
            )
            if logger is not None and logger.enabled:
                logger.log(
                    {"train/loss_avg": avg, "train/clips_per_s": cps},
                    step=state.step,
                )
            running = {"loss": 0.0, "n": 0}
            t_start = time.time()
    return {"final_loss": float(loss.detach().item())}


def save_checkpoint(
    model: torch.nn.Module, optimizer, scheduler, path: str, state: TrainState
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "step": state.step,
            "epoch": state.epoch,
        },
        path,
    )


def load_checkpoint(
    model: torch.nn.Module,
    path: str,
    *,
    weights_only_into_model: bool = False,
    optimizer=None,
    scheduler=None,
    state: Optional[TrainState] = None,
) -> None:
    # `weights_only=True` rejects pickled tensors with unknown classes — safe for
    # our own training checkpoints which only contain torch state dicts and ints.
    try:
        sd = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        # Older PyTorch / mixed payloads: fall back, but only for trusted local files.
        sd = torch.load(path, map_location="cpu", weights_only=False)

    missing, unexpected = model.load_state_dict(sd["model"], strict=False)
    total = sum(1 for _ in model.state_dict())
    loaded = total - len(missing)
    print(f"[load_checkpoint] loaded {loaded}/{total} params from {path}")
    if missing:
        # Group missing keys by top-level module so the summary is readable.
        groups: dict[str, int] = {}
        for k in missing:
            head = k.split(".", 1)[0]
            groups[head] = groups.get(head, 0) + 1
        summary = ", ".join(f"{h}={n}" for h, n in sorted(groups.items()))
        print(f"[load_checkpoint] missing ({len(missing)}): {summary}")
    if unexpected:
        print(
            f"[load_checkpoint] unexpected ({len(unexpected)}): {unexpected[:3]}{'...' if len(unexpected) > 3 else ''}"
        )
    if weights_only_into_model:
        return
    if optimizer is not None and "optimizer" in sd and sd["optimizer"] is not None:
        optimizer.load_state_dict(sd["optimizer"])
    if scheduler is not None and "scheduler" in sd and sd["scheduler"] is not None:
        scheduler.load_state_dict(sd["scheduler"])
    if state is not None:
        state.step = sd.get("step", 0)
        state.epoch = sd.get("epoch", 0)
