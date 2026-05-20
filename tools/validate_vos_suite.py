"""Run DAVIS-style J/F validation across the paper VOS benchmark roots."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from efficient_track_anything.build_efficienttam import (
    build_efficienttam_video_predictor,
)
from tools.validate import evaluate, has_supported_vos_layout, pick_device


BENCHMARK_ENV = {
    "mose": "VAL_ROOT_MOSE",
    "davis": "VAL_ROOT_DAVIS",
    "lvos": "VAL_ROOT_LVOS",
    "sav": "VAL_ROOT_SAV",
    "ytvos": "VAL_ROOT_YTVOS",
}


def _resolve_roots(args: argparse.Namespace) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for name, env_name in BENCHMARK_ENV.items():
        value = getattr(args, name) or os.environ.get(env_name)
        if value:
            roots[name] = Path(value)
    return roots


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--max-videos", type=int, default=None)
    for name, env_name in BENCHMARK_ENV.items():
        parser.add_argument(
            f"--{name}",
            default=None,
            help=f"Validation root for {name.upper()} (or ${env_name}).",
        )
    args = parser.parse_args()

    roots = _resolve_roots(args)
    if not roots:
        raise SystemExit(
            "No VOS validation roots configured. Set one or more of "
            + ", ".join(f"${v}" for v in BENCHMARK_ENV.values())
            + "."
        )

    device = pick_device()
    print(f"[validate_vos_suite] device={device}")
    predictor = build_efficienttam_video_predictor(
        config_file=args.config,
        ckpt_path=args.ckpt,
        device=str(device),
        mode="eval",
    )

    results: dict[str, dict] = {}
    for name, root in roots.items():
        if not has_supported_vos_layout(root):
            results[name] = {
                "status": "skipped",
                "reason": "missing supported VOS layout",
                "root": str(root),
            }
            print(f"[validate_vos_suite] skipping {name}: unsupported layout at {root}")
            continue
        print(f"[validate_vos_suite] evaluating {name} at {root}")
        res = evaluate(predictor, root, args.max_videos)
        res["status"] = "ok"
        res["root"] = str(root)
        results[name] = res

    ok = {k: v for k, v in results.items() if v.get("status") == "ok"}
    if ok:
        mean_jf = sum(float(v["JF_mean"]) for v in ok.values()) / len(ok)
    else:
        mean_jf = 0.0
    summary = {
        "benchmarks": results,
        "mean_JF_over_evaluated_benchmarks": mean_jf,
        "n_evaluated_benchmarks": len(ok),
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"[validate_vos_suite] wrote {out}")


if __name__ == "__main__":
    main()
