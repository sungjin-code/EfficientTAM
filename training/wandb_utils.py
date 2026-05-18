"""Optional Weights & Biases logging.

Training stays fully runnable when `wandb` (or `python-dotenv`) is not
installed, or when no API key is configured. In those cases all logger
methods are no-ops.

Resolution order for credentials and run metadata:
    1. Process env (already exported).
    2. `.env` file at repo root, if `python-dotenv` is available.
    3. `wandb` block in the training YAML.
Set `WANDB_MODE=disabled` (or omit `WANDB_API_KEY`) to skip entirely.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore
    except ImportError:
        return
    # Walk up from this file to find a .env at repo root.
    for parent in [Path.cwd(), *Path(__file__).resolve().parents]:
        candidate = parent / ".env"
        if candidate.is_file():
            load_dotenv(candidate, override=False)
            return


class WandbLogger:
    """Thin wrapper around wandb.run. Becomes a no-op if wandb is unavailable."""

    def __init__(self) -> None:
        self._run = None
        self._wandb = None

    @property
    def enabled(self) -> bool:
        return self._run is not None

    def init(
        self,
        wandb_cfg: Optional[dict],
        run_config: dict,
        default_run_name: str,
    ) -> None:
        _load_dotenv()
        wandb_cfg = wandb_cfg or {}
        if not wandb_cfg.get("enabled", True):
            print("[wandb] disabled via config")
            return
        if os.environ.get("WANDB_MODE", "").lower() == "disabled":
            print("[wandb] disabled via WANDB_MODE=disabled")
            return
        if not os.environ.get("WANDB_API_KEY"):
            print("[wandb] WANDB_API_KEY not set — skipping wandb logging")
            return
        try:
            import wandb  # type: ignore
        except ImportError:
            print("[wandb] wandb not installed — skipping wandb logging")
            return

        self._wandb = wandb
        self._run = wandb.init(
            project=wandb_cfg.get("project")
            or os.environ.get("WANDB_PROJECT", "efficient-tam"),
            entity=wandb_cfg.get("entity") or os.environ.get("WANDB_ENTITY"),
            name=wandb_cfg.get("run_name") or default_run_name,
            tags=wandb_cfg.get("tags"),
            notes=wandb_cfg.get("notes"),
            config=run_config,
        )
        print(f"[wandb] logging to {self._run.url}")

    def log(self, metrics: dict, step: Optional[int] = None) -> None:
        if self._run is None:
            return
        self._run.log(metrics, step=step)

    def save(self, path: str) -> None:
        if self._run is None:
            return
        try:
            self._wandb.save(path, policy="now")
        except Exception as e:
            print(f"[wandb] save failed for {path}: {e}")

    def finish(self) -> None:
        if self._run is None:
            return
        self._run.finish()
        self._run = None


def make_logger() -> WandbLogger:
    return WandbLogger()


def log_metrics(logger: Optional[WandbLogger], metrics: dict, step: int) -> None:
    if logger is not None:
        logger.log(metrics, step=step)


__all__ = ["WandbLogger", "make_logger", "log_metrics"]
