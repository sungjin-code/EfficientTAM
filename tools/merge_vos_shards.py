"""Merge per-shard VOS suite summaries into a single combined summary.

When `validate_vos_suite` is run with `--num-shards N` (one process per GPU, each
with its own `--output-json`), every shard scores a disjoint, strided subset of
each benchmark's videos. This tool unions the per-video results back together and
re-aggregates the overall J&F (or G) so the combined numbers match an unsharded
run.

Usage:
    python -m tools.merge_vos_shards \\
        --output-json eval/vos_suite.json \\
        eval/vos_suite.shard0.json eval/vos_suite.shard1.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.validate import _aggregate_per_video
from tools.validate_vos_suite import _apply_paper_metric_name, _write_summary


def _normalize_per_video(per_video: dict[str, dict]) -> dict[str, dict]:
    """Re-key each entry to the raw {J, F, JF, n_objects} schema.

    A shard whose benchmark used the "G" primary metric (YTVOS) has already had
    "JF" renamed to "G"; rebuild a clean JF entry so re-aggregation and the
    paper-metric rename below stay uniform across benchmarks.
    """
    out: dict[str, dict] = {}
    for video, v in per_video.items():
        j = float(v["J"])
        f = float(v["F"])
        out[video] = {
            "J": j,
            "F": f,
            "JF": 0.5 * (j + f),
            "n_objects": int(v.get("n_objects", 1)),
        }
    return out


def merge(shard_files: list[Path]) -> dict[str, dict]:
    """Combine each benchmark's per-video results across all shard summaries."""
    # Per benchmark: union of per-video results, plus a fallback non-"ok" entry.
    union: dict[str, dict[str, dict]] = {}
    fallback: dict[str, dict] = {}
    order: list[str] = []

    for path in shard_files:
        data = json.loads(path.read_text())
        benchmarks = data.get("benchmarks", {})
        for name, entry in benchmarks.items():
            if name not in union:
                union[name] = {}
                order.append(name)
            if entry.get("status") == "ok":
                union[name].update(_normalize_per_video(entry.get("per_video", {})))
            else:
                # Keep the first non-ok entry (e.g. skipped: held-out GT) as a
                # fallback in case no shard ever scored this benchmark.
                fallback.setdefault(name, entry)

    results: dict[str, dict] = {}
    for name in order:
        per_video = union[name]
        if per_video:
            res = _aggregate_per_video(per_video)
            res = _apply_paper_metric_name(name, res)
            res["status"] = "ok"
            if name in fallback and "root" in fallback[name]:
                res["root"] = fallback[name]["root"]
            results[name] = res
        else:
            results[name] = fallback.get(
                name, {"status": "skipped", "reason": "no results in any shard"}
            )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", required=True, help="Combined summary path.")
    parser.add_argument(
        "shard_files",
        nargs="+",
        help="Per-shard summary JSONs produced by validate_vos_suite --num-shards.",
    )
    args = parser.parse_args()

    shard_files = [Path(p) for p in args.shard_files]
    missing = [str(p) for p in shard_files if not p.is_file()]
    if missing:
        raise SystemExit("Missing shard summaries: " + ", ".join(missing))

    results = merge(shard_files)
    out = Path(args.output_json)
    summary = _write_summary(out, results)
    print("=" * 60)
    for name, res in results.items():
        if res.get("status") == "ok":
            metric = res.get("primary_metric", "J&F")
            print(
                f"{name}: {metric}={res['primary_score']:.3f} "
                f"(n_videos={res.get('n_videos', 0)})"
            )
        else:
            print(f"{name}: {res.get('status')} ({res.get('reason', '')})")
    print(
        f"mean primary score over {summary['n_evaluated_benchmarks']} benchmarks: "
        f"{summary['mean_primary_score_over_evaluated_benchmarks']:.3f}"
    )
    print(f"[merge_vos_shards] wrote {out}")


if __name__ == "__main__":
    main()
