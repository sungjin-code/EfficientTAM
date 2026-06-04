"""Training engine: forward_clip + train_one_epoch_{image,video}.

The clip forward runs all T frames through the image encoder in one batch (for
throughput), then iterates per-frame calling `_track_step` directly so we get
the full sam_outputs tuple (which includes object_score_logits — the public
`track_step` drops it during training). Memory bookkeeping mirrors the
inference predictor.
"""

from __future__ import annotations

import json
import math
import os
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Optional

import torch

from efficient_track_anything.utils.misc import concat_points
from training.data.prompts import PromptSampler
from training.distributed import average_gradients, unwrap_model
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


def _sample_correction_inputs(
    prompt_sampler: PromptSampler,
    gt_masks_t: torch.Tensor,
    pred_logits: torch.Tensor,
    has_object_t: torch.Tensor,
    correction_frames: list[list[int]],
    frame_idx: int,
    device: torch.device,
) -> Optional[dict]:
    coord_list = []
    label_list = []
    for b in range(gt_masks_t.shape[0]):
        if frame_idx in correction_frames[b] and has_object_t[b] > 0:
            sample = prompt_sampler.sample_correction(
                gt_masks_t[b].float(),
                pred_logits[b].detach(),
            )
            coord_list.append(sample.point_coords)
            label_list.append(sample.point_labels)
        else:
            coord_list.append(torch.zeros(1, 2))
            label_list.append(torch.tensor([-1], dtype=torch.int32))
    coords = torch.stack(coord_list, dim=0).to(device)
    labels = torch.stack(label_list, dim=0).to(device)
    return _make_point_inputs(coords, labels)


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
    num_correction_points_per_frame: int = 1,
    add_correction_frames_as_cond: bool = False,
) -> ClipOutput:
    model = unwrap_model(model)
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

        # Forward through the track step. _track_step gives us the full sam_outputs.
        current_out, sam_outputs, high_res_features, pix_feat = model._track_step(
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
        multistep_masks = [high_res_multimasks]
        multistep_ious = [ious]
        multistep_obj_logits = [object_score_logits]

        is_correction = any(t in cf for cf in correction_frames)
        if is_correction and num_correction_points_per_frame > 0:
            for _ in range(num_correction_points_per_frame):
                new_point_inputs = _sample_correction_inputs(
                    prompt_sampler=prompt_sampler,
                    gt_masks_t=gt_masks[:, t],
                    pred_logits=high_res_masks,
                    has_object_t=has_object[:, t],
                    correction_frames=correction_frames,
                    frame_idx=t,
                    device=device,
                )
                if new_point_inputs is None:
                    continue
                point_inputs = concat_points(
                    point_inputs,
                    new_point_inputs["point_coords"],
                    new_point_inputs["point_labels"],
                )
                multimask_output = model._use_multimask(is_init, point_inputs)
                sam_outputs = model._forward_sam_heads(
                    backbone_features=pix_feat,
                    point_inputs=point_inputs,
                    mask_inputs=low_res_masks,
                    high_res_features=high_res_features,
                    multimask_output=multimask_output,
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
                multistep_masks.append(high_res_multimasks)
                multistep_ious.append(ious)
                multistep_obj_logits.append(object_score_logits)

        # Per-frame loss — supervise at high-res (image_size) as in SAM/SAM2.
        # `high_res_multimasks` is bilinear-upsampled from the H/4 head to (image_size, image_size).
        gt_full = gt_masks[:, t].float()
        frame_loss, metrics = loss_fn(
            mask_logits=multistep_masks,
            ious=multistep_ious,
            object_score_logits=multistep_obj_logits,
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

        is_cond = is_init or (add_correction_frames_as_cond and is_correction)

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
    # Adaptive image-encoder forward chunk size (samples processed per forward).
    # 0 = uninitialized (set to the full batch on first use). `auto_chunk_max`
    # caps re-growth at the largest size that previously OOMed.
    auto_chunk: int = 0
    auto_chunk_max: int = 0


def _aggregate_clip_metrics(per_frame: list[dict]) -> dict:
    """Average per-frame loss components (returned by MultiStepLoss) into scalars."""
    if not per_frame:
        return {}
    keys = list(per_frame[0].keys())
    out: dict[str, float] = {}
    for k in keys:
        vals = [float(m[k].detach().item()) for m in per_frame]
        out[k] = sum(vals) / len(vals)
    return out


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


def _scale_gradients(model: torch.nn.Module, scale: float) -> None:
    if scale == 1.0:
        return
    for param in model.parameters():
        if param.grad is not None:
            param.grad.mul_(scale)


def _optimizer_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineSchedule,
    state: TrainState,
    grad_clip: float,
    accumulation_steps: int,
    pending_micro_batches: int,
) -> None:
    if pending_micro_batches <= 0:
        return
    # Backward divides by the target accumulation count. For a partial epoch
    # tail, rescale so the step averages the microbatches that actually ran.
    _scale_gradients(model, accumulation_steps / pending_micro_batches)
    average_gradients(model)
    if grad_clip > 0:
        clip_grad_norm(model.parameters(), grad_clip)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    state.step += 1


def _maybe_grow_chunk(state: TrainState, device: torch.device, n: int) -> None:
    """Grow the forward chunk when this process's peak memory leaves headroom.

    Uses the caching allocator's peak (reset each batch) rather than free memory
    between batches, which would over-report headroom. Capped at `auto_chunk_max`
    so we never climb back to a size that already OOMed.
    """
    if device.type != "cuda":
        return
    chunk = state.auto_chunk if state.auto_chunk > 0 else n
    cap = min(state.auto_chunk_max if state.auto_chunk_max > 0 else n, n)
    if chunk >= cap:
        return
    total = torch.cuda.get_device_properties(device).total_memory
    peak = torch.cuda.max_memory_reserved(device)
    torch.cuda.reset_peak_memory_stats(device)
    if total > 0 and peak / total < 0.70:
        new = min(cap, chunk + max(1, chunk // 4))
        if new > chunk:
            state.auto_chunk = new
            print(
                f"[image] peak mem {peak / total:.0%} of GPU — growing forward "
                f"chunk {chunk} -> {new}"
            )


def _image_forward_backward_adaptive(
    *,
    model: torch.nn.Module,
    frames: torch.Tensor,
    gt: torch.Tensor,
    ho: torch.Tensor,
    point_inputs: dict,
    prompt_sampler: PromptSampler,
    loss_fn: MultiStepLoss,
    precision: str,
    device: torch.device,
    accumulation_steps: int,
    state: TrainState,
    optimizer: torch.optim.Optimizer,
) -> tuple[float, dict, bool]:
    """Forward+backward one image batch in memory-adaptive chunks.

    Splits the N independent T=1 samples into sub-batches, shrinking on CUDA OOM
    and growing when memory is plentiful. Each chunk's loss is weighted by
    (chunk / N) so the accumulated gradient equals the full-batch mean — making
    the no-OOM path (single chunk = N) identical to a plain full-batch step.

    Returns (step_loss, components, oom_reset). When `oom_reset` is True an OOM
    occurred: grads were cleared and the batch skipped, so the caller must reset
    its accumulation window. `step_loss` may be NaN for the caller's NaN guard.

    Note: under DDP each chunk triggers a gradient all-reduce (one per chunk).
    With world_size==1 this is a no-op; multi-GPU users pay extra syncs only
    while the chunk size is below the batch size.
    """
    n = frames.shape[0]
    if device.type != "cuda":
        chunk = n
    else:
        if state.auto_chunk <= 0:
            state.auto_chunk = n
        chunk = max(1, min(state.auto_chunk, n))

    step_loss = 0.0
    comp_sums: dict[str, float] = {}
    try:
        for i in range(0, n, chunk):
            cur = min(chunk, n - i)
            sl = slice(i, i + cur)
            with _autocast(device, precision):
                out = forward_clip(
                    model=model,
                    frames=frames[sl],
                    gt_masks=gt[sl],
                    has_object=ho[sl],
                    frame0_point_inputs={k: v[sl] for k, v in point_inputs.items()},
                    correction_frames=[[] for _ in range(cur)],
                    prompt_sampler=prompt_sampler,
                    loss_fn=loss_fn,
                    run_mem_encoder=False,
                )
                loss = out.total_loss
            chunk_loss = float(loss.detach().item())
            if not math.isfinite(chunk_loss):
                return float("nan"), {}, False
            weight = cur / n
            (loss * weight / accumulation_steps).backward()
            step_loss += chunk_loss * weight
            for k, v in _aggregate_clip_metrics(out.per_frame_metrics).items():
                comp_sums[k] = comp_sums.get(k, 0.0) + v * weight
    except RuntimeError as exc:
        # `torch.cuda.OutOfMemoryError` subclasses RuntimeError on modern torch;
        # older versions raise a plain RuntimeError with this message. Re-raise
        # anything that is not an OOM so real bugs still surface.
        if "out of memory" not in str(exc).lower():
            raise
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        new_chunk = max(1, chunk // 2)
        if new_chunk >= chunk:
            raise  # cannot fit even a single sample
        state.auto_chunk = new_chunk
        state.auto_chunk_max = chunk  # never re-grow to a size that OOMed
        print(
            f"[image] CUDA OOM at forward chunk={chunk} — shrinking to "
            f"{new_chunk} and skipping this batch"
        )
        return 0.0, {}, True

    _maybe_grow_chunk(state, device, n)
    return step_loss, comp_sums, False


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
    accumulation_steps: int = 1,
    log_every: int = 20,
    overfit_one_batch: bool = False,
    max_steps: int | None = None,
    precision: str = "auto",
    logger: Optional[WandbLogger] = None,
) -> dict:
    """One pass over the image loader. Each batch is a T=1 clip."""
    model.train()
    accumulation_steps = max(1, int(accumulation_steps))
    fixed_batch = None
    running: dict[str, float] = {"loss": 0.0, "n": 0.0}
    component_keys = ("focal", "dice", "iou_l1", "obj_bce")
    for k in component_keys:
        running[k] = 0.0
    t_start = time.time()
    last_step_loss = 0.0
    last_components: dict[str, float] = {}
    consecutive_nan = 0
    MAX_CONSECUTIVE_NAN = 50
    pending_micro_batches = 0
    optimizer.zero_grad(set_to_none=True)
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

        # Forward+backward with adaptive memory chunking (shrinks on CUDA OOM,
        # grows when memory is plentiful). Backward happens inside.
        step_loss, last_components, oom_reset = _image_forward_backward_adaptive(
            model=model,
            frames=frames,
            gt=gt,
            ho=ho,
            point_inputs=point_inputs,
            prompt_sampler=prompt_sampler,
            loss_fn=loss_fn,
            precision=precision,
            device=device,
            accumulation_steps=accumulation_steps,
            state=state,
            optimizer=optimizer,
        )
        if oom_reset:
            # OOM cleared grads mid-window; reset the accumulation window cleanly.
            pending_micro_batches = 0
            running = {"loss": 0.0, "n": 0.0}
            for k in component_keys:
                running[k] = 0.0
            continue

        # NaN/Inf guard — a poisoned step corrupts every param via optimizer.step.
        if not math.isfinite(step_loss):
            consecutive_nan += 1
            print(
                f"[image] WARNING non-finite loss={step_loss} at step={state.step} "
                f"epoch={state.epoch} — skipping optimizer step "
                f"({consecutive_nan}/{MAX_CONSECUTIVE_NAN} consecutive)"
            )
            optimizer.zero_grad(set_to_none=True)
            pending_micro_batches = 0
            running = {"loss": 0.0, "n": 0.0}
            for k in component_keys:
                running[k] = 0.0
            if logger is not None and logger.enabled:
                logger.log({"train/nan_skip": 1.0}, step=state.step)
            if consecutive_nan >= MAX_CONSECUTIVE_NAN:
                raise RuntimeError(
                    f"[image] aborting: {consecutive_nan} consecutive non-finite losses. "
                    "Check data, learning rate, or precision (try --precision fp32)."
                )
            continue
        consecutive_nan = 0

        last_step_loss = step_loss
        pending_micro_batches += 1
        running["loss"] += step_loss
        running["n"] += 1
        for k in component_keys:
            running[k] += last_components.get(k, 0.0)

        if pending_micro_batches < accumulation_steps:
            continue

        _optimizer_step(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            state=state,
            grad_clip=grad_clip,
            accumulation_steps=accumulation_steps,
            pending_micro_batches=pending_micro_batches,
        )
        pending_micro_batches = 0

        n = max(1, running["n"])
        avg = running["loss"] / n
        comp_avgs = {k: running[k] / n for k in component_keys}
        if logger is not None and logger.enabled:
            payload = {
                "train/loss": avg,
                "train/lr": optimizer.param_groups[0]["lr"],
                "train/epoch": state.epoch,
                "train/accumulation_steps": accumulation_steps,
            }
            for k, v in comp_avgs.items():
                payload[f"train/{k}"] = v
            logger.log(payload, step=state.step)
        if state.step % log_every == 0:
            n = max(1, running["n"])
            avg = running["loss"] / n
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t_start
            ips = running["n"] / max(elapsed, 1e-3)
            comp_avgs = {k: running[k] / n for k in component_keys}
            comp_str = " ".join(f"{k}={v:.4f}" for k, v in comp_avgs.items())
            print(
                f"[image] step={state.step} epoch={state.epoch} loss={avg:.4f} "
                f"{comp_str} lr={lr:.2e} accum={accumulation_steps} "
                f"micro_batches/s={ips:.2f}"
            )
            if logger is not None and logger.enabled:
                payload = {"train/loss_avg": avg, "train/micro_batches_per_s": ips}
                for k, v in comp_avgs.items():
                    payload[f"train/{k}_avg"] = v
                logger.log(payload, step=state.step)
            running = {"loss": 0.0, "n": 0.0}
            for k in component_keys:
                running[k] = 0.0
            t_start = time.time()
    if pending_micro_batches > 0 and (max_steps is None or state.step < max_steps):
        _optimizer_step(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            state=state,
            grad_clip=grad_clip,
            accumulation_steps=accumulation_steps,
            pending_micro_batches=pending_micro_batches,
        )
    final = {"final_loss": last_step_loss}
    for k, v in last_components.items():
        final[f"final_{k}"] = v
    return final


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
    accumulation_steps: int = 1,
    log_every: int = 5,
    overfit_one_batch: bool = False,
    max_steps: int | None = None,
    precision: str = "auto",
    logger: Optional[WandbLogger] = None,
    num_correction_points_per_frame: int = 1,
    add_correction_frames_as_cond: bool = False,
) -> dict:
    model.train()
    accumulation_steps = max(1, int(accumulation_steps))
    fixed_batch = None
    running: dict[str, float] = {"loss": 0.0, "n": 0.0}
    component_keys = ("focal", "dice", "iou_l1", "obj_bce")
    for k in component_keys:
        running[k] = 0.0
    t_start = time.time()
    loss = torch.zeros((), device=device)
    last_components: dict[str, float] = {}
    consecutive_nan = 0
    MAX_CONSECUTIVE_NAN = 50
    pending_micro_batches = 0
    optimizer.zero_grad(set_to_none=True)
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
                num_correction_points_per_frame=num_correction_points_per_frame,
                add_correction_frames_as_cond=add_correction_frames_as_cond,
            )
            loss = out.total_loss

        step_loss = float(loss.detach().item())
        if not math.isfinite(step_loss):
            consecutive_nan += 1
            print(
                f"[video] WARNING non-finite loss={step_loss} at step={state.step} "
                f"epoch={state.epoch} — skipping optimizer step "
                f"({consecutive_nan}/{MAX_CONSECUTIVE_NAN} consecutive)"
            )
            optimizer.zero_grad(set_to_none=True)
            pending_micro_batches = 0
            running = {"loss": 0.0, "n": 0.0}
            for k in component_keys:
                running[k] = 0.0
            if logger is not None and logger.enabled:
                logger.log({"train/nan_skip": 1.0}, step=state.step)
            if consecutive_nan >= MAX_CONSECUTIVE_NAN:
                raise RuntimeError(
                    f"[video] aborting: {consecutive_nan} consecutive non-finite losses. "
                    "Check data, learning rate, or precision (try --precision fp32)."
                )
            continue
        consecutive_nan = 0

        (loss / accumulation_steps).backward()
        pending_micro_batches += 1
        running["loss"] += step_loss
        running["n"] += 1
        last_components = _aggregate_clip_metrics(out.per_frame_metrics)
        for k in component_keys:
            running[k] += last_components.get(k, 0.0)

        if pending_micro_batches < accumulation_steps:
            continue

        _optimizer_step(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            state=state,
            grad_clip=grad_clip,
            accumulation_steps=accumulation_steps,
            pending_micro_batches=pending_micro_batches,
        )
        pending_micro_batches = 0

        n = max(1, running["n"])
        avg = running["loss"] / n
        comp_avgs = {k: running[k] / n for k in component_keys}
        if logger is not None and logger.enabled:
            payload = {
                "train/loss": avg,
                "train/lr": optimizer.param_groups[0]["lr"],
                "train/epoch": state.epoch,
                "train/accumulation_steps": accumulation_steps,
            }
            for k, v in comp_avgs.items():
                payload[f"train/{k}"] = v
            logger.log(payload, step=state.step)
        if state.step % log_every == 0:
            n = max(1, running["n"])
            avg = running["loss"] / n
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t_start
            cps = running["n"] / max(elapsed, 1e-3)
            comp_avgs = {k: running[k] / n for k in component_keys}
            comp_str = " ".join(f"{k}={v:.4f}" for k, v in comp_avgs.items())
            print(
                f"[video] step={state.step} epoch={state.epoch} loss={avg:.4f} "
                f"{comp_str} lr={lr:.2e} accum={accumulation_steps} "
                f"micro_batches/s={cps:.2f}"
            )
            if logger is not None and logger.enabled:
                payload = {"train/loss_avg": avg, "train/micro_batches_per_s": cps}
                for k, v in comp_avgs.items():
                    payload[f"train/{k}_avg"] = v
                logger.log(payload, step=state.step)
            running = {"loss": 0.0, "n": 0.0}
            for k in component_keys:
                running[k] = 0.0
            t_start = time.time()
    if pending_micro_batches > 0 and (max_steps is None or state.step < max_steps):
        _optimizer_step(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            state=state,
            grad_clip=grad_clip,
            accumulation_steps=accumulation_steps,
            pending_micro_batches=pending_micro_batches,
        )
    final = {"final_loss": float(loss.detach().item())}
    for k, v in last_components.items():
        final[f"final_{k}"] = v
    return final


def save_checkpoint(
    model: torch.nn.Module, optimizer, scheduler, path: str, state: TrainState
) -> None:
    """Atomic save: write to `<path>.tmp` first, then rename.

    A kill / disk-full / oom during torch.save would otherwise leave a
    truncated .pt that breaks `--resume`.
    """
    model = unwrap_model(model)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "step": state.step,
        "epoch": state.epoch,
    }
    tmp_path = f"{path}.tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def _file_artifact(path: str | os.PathLike | None) -> dict:
    if path is None:
        return {"path": None, "exists": False, "size_bytes": 0}
    path_str = str(path)
    exists = os.path.isfile(path_str)
    return {
        "path": path_str,
        "exists": exists,
        "size_bytes": os.path.getsize(path_str) if exists else 0,
    }


def save_training_artifact(
    path: str | os.PathLike,
    *,
    stage: str,
    status: str,
    state: TrainState,
    total_steps: int,
    output_dir: str | os.PathLike,
    latest_checkpoint: str | os.PathLike,
    final_checkpoint: str | os.PathLike | None = None,
    epoch_checkpoint: str | os.PathLike | None = None,
    interrupt_checkpoint: str | os.PathLike | None = None,
    metrics: Optional[dict] = None,
    extra: Optional[dict] = None,
) -> None:
    """Write a machine-readable training status artifact atomically."""
    payload = {
        "stage": stage,
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "step": int(state.step),
        "epoch": int(state.epoch),
        "total_steps": int(total_steps),
        "completed_expected_steps": int(state.step) >= int(total_steps),
        "output_dir": str(output_dir),
        "checkpoints": {
            "latest": _file_artifact(latest_checkpoint),
            "final": _file_artifact(final_checkpoint),
            "epoch": _file_artifact(epoch_checkpoint),
            "interrupt": _file_artifact(interrupt_checkpoint),
        },
        "metrics": metrics or {},
    }
    if extra:
        payload["extra"] = extra
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp_path, path)


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
    # Fall back to weights_only=False ONLY on the specific "unsupported global"
    # error that older PyTorch / SAM2-style payloads raise; any other failure
    # (truncation, IO error, ...) propagates so we don't silently `pickle.load`
    # an untrusted file.
    try:
        sd = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:
        msg = str(e).lower()
        is_unsupported_global = (
            "unsupported global" in msg or "weights_only" in msg or "globals" in msg
        )
        if not is_unsupported_global:
            raise
        print(
            f"[load_checkpoint] weights_only=True rejected {path} "
            f"({type(e).__name__}); retrying with weights_only=False. "
            "Only do this for checkpoints you trust."
        )
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
