"""Optimizer, parameter groups, and LR schedule for EfficientTAM training.

- AdamW with weight decay 0.1 (no decay on bias / norm / position-embedding buffers).
- Layer-wise LR decay on the image encoder backbone (depth-indexed).
- Linear warmup -> cosine to zero.
"""

from __future__ import annotations

import math
import re
from typing import Iterable

import torch


def _is_no_decay(name: str, param: torch.nn.Parameter) -> bool:
    if param.ndim <= 1:
        return True
    lower = name.lower()
    return any(
        k in lower
        for k in ("bias", "norm", "pos_embed", "tpos_enc", "no_obj", "no_mem")
    )


_BLOCK_PATTERN = re.compile(r"image_encoder\..*?(?:blocks|layers)\.(\d+)\.")


def _backbone_depth(name: str) -> int | None:
    """Return the block index of a parameter inside the image encoder, else None."""
    if not name.startswith("image_encoder."):
        return None
    m = _BLOCK_PATTERN.search(name)
    if m is None:
        return -1  # patch embed / pre-block params; treat as layer 0
    return int(m.group(1))


def build_param_groups(
    model: torch.nn.Module,
    backbone_lr: float,
    head_lr: float,
    layerwise_decay: float = 0.8,
    weight_decay: float = 0.1,
    num_layers_hint: int = 24,
    verbose: bool = True,
) -> list[dict]:
    """Split parameters into AdamW groups with layer-wise LR decay on the encoder.

    Backbone block depth is discovered by regex on parameter names. Parameters in
    `image_encoder` that don't match the block pattern (e.g. `patch_embed`,
    `pos_embed`) are treated as depth -1 and receive the deepest decay factor.
    Set `verbose=True` to print the resulting depth distribution at startup.
    """
    groups: dict[tuple, dict] = {}
    # Discover actual max depth so the decay scale is correct
    max_depth = -1
    for name, _ in model.named_parameters():
        d = _backbone_depth(name)
        if d is not None and d > max_depth:
            max_depth = d
    if max_depth < 0:
        max_depth = num_layers_hint

    depth_counts: dict[str, int] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        no_decay = _is_no_decay(name, param)
        wd = 0.0 if no_decay else weight_decay
        depth = _backbone_depth(name)
        if depth is not None:
            # Deeper layers get higher LR; shallower (closer to input) decayed more.
            scale = layerwise_decay ** (max_depth - max(depth, 0))
            lr = backbone_lr * scale
            tag = ("backbone", depth, wd)
            depth_counts[f"backbone_d{depth}"] = (
                depth_counts.get(f"backbone_d{depth}", 0) + param.numel()
            )
        else:
            lr = head_lr
            tag = ("head", 0, wd)
            depth_counts["head"] = depth_counts.get("head", 0) + param.numel()
        groups.setdefault(
            tag,
            {
                "params": [],
                "lr": lr,
                "weight_decay": wd,
                "name": f"{tag[0]}_d{tag[1]}_wd{wd}",
            },
        )
        groups[tag]["params"].append(param)

    if verbose:
        n_groups = len(groups)
        total_params = sum(depth_counts.values())
        print(
            f"[optim] {n_groups} param groups, {total_params / 1e6:.2f}M trainable params"
        )
        print(
            f"[optim] backbone max_depth={max_depth} layerwise_decay={layerwise_decay}"
        )
        head_p = depth_counts.get("head", 0)
        bb_p = total_params - head_p
        print(
            f"[optim] backbone params: {bb_p / 1e6:.2f}M  head params: {head_p / 1e6:.2f}M"
        )
        if max_depth >= 0:
            min_scale = layerwise_decay**max_depth
            print(
                f"[optim] backbone LR range: {backbone_lr * min_scale:.2e} (depth 0) -> {backbone_lr:.2e} (depth {max_depth})"
            )
    return list(groups.values())


def build_optimizer(
    model: torch.nn.Module,
    backbone_lr: float,
    head_lr: float,
    layerwise_decay: float = 0.8,
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.999),
) -> torch.optim.Optimizer:
    groups = build_param_groups(
        model, backbone_lr, head_lr, layerwise_decay, weight_decay
    )
    return torch.optim.AdamW(groups, lr=head_lr, betas=betas, weight_decay=weight_decay)


class WarmupCosineSchedule:
    """Per-step LR multiplier. Apply via `scheduler.step(optimizer)` after `optimizer.step()`.

    Each param group's `initial_lr` is captured on construction so we can scale it
    uniformly without flattening the layer-wise LRs assembled in `build_param_groups`.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        warmup_pct: float = 0.05,
    ):
        self.optimizer = optimizer
        self.total_steps = max(1, total_steps)
        self.warmup_steps = max(1, int(self.total_steps * warmup_pct))
        self.step_num = 0
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]

    def _factor(self) -> float:
        if self.step_num < self.warmup_steps:
            return self.step_num / self.warmup_steps
        progress = (self.step_num - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps
        )
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    def step(self) -> float:
        self.step_num += 1
        f = self._factor()
        for g, base in zip(self.optimizer.param_groups, self.base_lrs):
            g["lr"] = base * f
        return f

    def state_dict(self) -> dict:
        return {"step_num": self.step_num, "base_lrs": self.base_lrs}

    def load_state_dict(self, sd: dict) -> None:
        self.step_num = sd["step_num"]
        self.base_lrs = sd["base_lrs"]


def clip_grad_norm(
    params: Iterable[torch.nn.Parameter], max_norm: float = 0.1
) -> torch.Tensor:
    return torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm)
