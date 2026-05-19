"""Small distributed-training helpers for torchrun-based launches."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class DistInfo:
    enabled: bool
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def init_distributed() -> DistInfo:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return DistInfo(enabled=False)

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ["WORLD_SIZE"])
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend)
    return DistInfo(
        enabled=True,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
    )


def cleanup_distributed(info: DistInfo) -> None:
    if info.enabled and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def broadcast_model(model: torch.nn.Module, info: DistInfo) -> None:
    if not info.enabled or not dist.is_initialized():
        return
    for tensor in model.state_dict().values():
        if torch.is_tensor(tensor):
            dist.broadcast(tensor, src=0)


def average_gradients(model: torch.nn.Module) -> None:
    if not dist.is_available() or not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    for param in model.parameters():
        if param.grad is None:
            continue
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        param.grad.div_(world_size)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model
