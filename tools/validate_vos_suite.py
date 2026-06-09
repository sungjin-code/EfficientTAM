"""Run VOS validation across the paper benchmark roots."""

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

# Optional per-benchmark override for the dense-GT directory, when frames and
# dense annotations live in separate trees (e.g. MOSEv2 valid: frames under
# valid/JPEGImages, dense masks at <MOSE>/<video>/*.png while valid/Annotations
# holds only the seed frame). GT masks are read from <ann-root>/<video>/*.png.
BENCHMARK_ANN_ENV = {
    "mose": "VAL_ANN_ROOT_MOSE",
    "davis": "VAL_ANN_ROOT_DAVIS",
    "lvos": "VAL_ANN_ROOT_LVOS",
    "sav": "VAL_ANN_ROOT_SAV",
    "ytvos": "VAL_ANN_ROOT_YTVOS",
}

BENCHMARK_PRIMARY_METRIC = {
    "mose": "J&F",
    "davis": "J&F",
    "lvos": "J&F",
    "sav": "J&F",
    "ytvos": "G",
}


def _resolve_roots(args: argparse.Namespace) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for name, env_name in BENCHMARK_ENV.items():
        value = getattr(args, name) or os.environ.get(env_name)
        if value:
            roots[name] = Path(value)
    return roots


def _resolve_ann_roots(args: argparse.Namespace) -> dict[str, Path]:
    ann_roots: dict[str, Path] = {}
    for name, env_name in BENCHMARK_ANN_ENV.items():
        value = getattr(args, f"{name}_ann") or os.environ.get(env_name)
        if value:
            ann_roots[name] = Path(value)
    return ann_roots


def _apply_paper_metric_name(name: str, result: dict) -> dict:
    primary_metric = BENCHMARK_PRIMARY_METRIC[name]
    result["primary_metric"] = primary_metric

    if primary_metric == "G":
        result["G_mean"] = result.pop("JF_mean")
        for video_result in result.get("per_video", {}).values():
            video_result["G"] = video_result.pop("JF")
        result["primary_score"] = result["G_mean"]
    else:
        result["primary_score"] = result["JF_mean"]
    return result


def _write_summary(out: Path, results: dict[str, dict]) -> dict:
    """Persist the running summary so finished benchmarks survive a later crash."""
    ok = {k: v for k, v in results.items() if v.get("status") == "ok"}
    mean_score = (
        sum(float(v["primary_score"]) for v in ok.values()) / len(ok) if ok else 0.0
    )
    summary = {
        "benchmarks": results,
        "mean_primary_score_over_evaluated_benchmarks": mean_score,
        "n_evaluated_benchmarks": len(ok),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(summary, indent=2))
    tmp.replace(out)
    return summary


def _load_prior_benchmarks(out: Path) -> dict[str, dict]:
    """Load completed benchmarks from a previous (interrupted) suite run."""
    if not out.is_file():
        return {}
    try:
        data = json.loads(out.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    benchmarks = data.get("benchmarks") if isinstance(data, dict) else None
    return benchmarks if isinstance(benchmarks, dict) else {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable image-encoder torch.compile (skips slow max-autotune; "
        "useful for short/smoke validation runs).",
    )
    parser.add_argument(
        "--no-offload",
        action="store_true",
        help="Keep all video frames and tracking state on GPU. Offloading to "
        "CPU is on by default; it avoids OOM on long videos / shared GPUs at a "
        "small FPS cost.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print per-video diagnostics (empty-GT frame ratio and J_present) "
        "to help distinguish empty-frame inflation from train/eval overlap.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Split each benchmark's video list into this many strided shards so "
        "one process per GPU can run disjoint subsets in parallel. Give each shard "
        "its own --output-json, then combine them with tools.merge_vos_shards.",
    )
    parser.add_argument(
        "--shard-idx",
        type=int,
        default=0,
        help="Which shard this process evaluates (0..num_shards-1).",
    )
    for name, env_name in BENCHMARK_ENV.items():
        parser.add_argument(
            f"--{name}",
            default=None,
            help=f"Validation root for {name.upper()} (or ${env_name}).",
        )
    for name, env_name in BENCHMARK_ANN_ENV.items():
        parser.add_argument(
            f"--{name}-ann",
            default=None,
            help=f"Override dense-GT dir for {name.upper()} (<dir>/<video>/*.png; "
            f"or ${env_name}). Use when frames and dense masks live in separate "
            "trees, e.g. MOSEv2 valid.",
        )
    args = parser.parse_args()

    roots = _resolve_roots(args)
    ann_roots = _resolve_ann_roots(args)
    if not roots:
        raise SystemExit(
            "No VOS validation roots configured. Set one or more of "
            + ", ".join(f"${v}" for v in BENCHMARK_ENV.values())
            + "."
        )

    out = Path(args.output_json)
    # Resume: reuse benchmarks already completed in a prior run.
    prior = _load_prior_benchmarks(out)
    results: dict[str, dict] = {}
    pending: list[tuple[str, Path]] = []
    for name, root in roots.items():
        if prior.get(name, {}).get("status") == "ok":
            results[name] = prior[name]
            print(f"[validate_vos_suite] {name}: already evaluated, skipping")
        elif not has_supported_vos_layout(root):
            results[name] = {
                "status": "skipped",
                "reason": "missing supported VOS layout",
                "root": str(root),
            }
            print(f"[validate_vos_suite] skipping {name}: unsupported layout at {root}")
        else:
            pending.append((name, root))

    if not pending:
        # Nothing left to evaluate — just (re)write the summary; skip model load.
        _write_summary(out, results)
        print(f"[validate_vos_suite] nothing to evaluate; wrote {out}")
        return

    device = pick_device()
    print(f"[validate_vos_suite] device={device}")
    overrides = ["++model.compile_image_encoder=false"] if args.no_compile else []
    predictor = build_efficienttam_video_predictor(
        config_file=args.config,
        ckpt_path=args.ckpt,
        device=str(device),
        mode="eval",
        hydra_overrides_extra=overrides,
    )

    # Tag per-shard scratch files so parallel shards sharing EVAL_DIR don't collide.
    shard_suffix = f"_s{args.shard_idx}" if args.num_shards > 1 else ""
    for name, root in pending:
        print(f"[validate_vos_suite] evaluating {name} at {root}")
        # Per-video resume cache; removed once the benchmark fully completes.
        cache_path = out.parent / f".{name}_partial{shard_suffix}.json"
        res = evaluate(
            predictor,
            root,
            args.max_videos,
            cache_path=cache_path,
            offload_video_to_cpu=not args.no_offload,
            offload_state_to_cpu=not args.no_offload,
            debug=args.debug,
            num_shards=args.num_shards,
            shard_idx=args.shard_idx,
            ann_root_override=ann_roots.get(name),
        )
        if res.get("n_videos", 0) == 0:
            # No video had frames to score — typically MOSE/LVOS/YTVOS val, whose
            # dense GT is held out for server submission. Don't report a bogus 0.0.
            results[name] = {
                "status": "skipped",
                "reason": "no locally-scorable frames (held-out GT; "
                "submit predictions to the benchmark server)",
                "root": str(root),
            }
            print(
                f"[validate_vos_suite] {name}: no locally-scorable frames "
                "(held-out GT) — skipping; needs server submission."
            )
        else:
            res = _apply_paper_metric_name(name, res)
            res["status"] = "ok"
            res["root"] = str(root)
            results[name] = res
        # Commit progress after each benchmark so a later crash resumes here.
        _write_summary(out, results)
        cache_path.unlink(missing_ok=True)

    _write_summary(out, results)
    print(f"[validate_vos_suite] wrote {out}")


if __name__ == "__main__":
    main()
