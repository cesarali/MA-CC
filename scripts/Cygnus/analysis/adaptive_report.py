"""Compare a fixed-draw finalize with an adaptive-stopping finalize of the same study.

    python scripts/Cygnus/analysis/adaptive_report.py <fixed-output-dir> <adaptive-output-dir> [--markdown out.md]

Both arguments are finalizer output directories (each with ``tables/`` and
``analysis_manifest.json``). The report covers the information estimates
table: how many draws each interval used, how far the early-stopped endpoints
sit from the full-draw ones (in units of the full-draw width), that the point
estimates and p-values are unchanged, and the stage wall times.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

TABLE = "study_information_estimates"


def _stages(manifest: Path) -> dict[str, float]:
    data = json.loads(manifest.read_text())
    return {s["stage"]: float(s["elapsed_seconds"]) for s in data["performance"]["stages"]}


def _load(output: Path) -> pd.DataFrame:
    for name in (TABLE, "information_estimates", "study_information"):
        path = output / "tables" / f"{name}.parquet"
        if path.is_file():
            return pd.read_parquet(path)
    candidates = sorted((output / "tables").glob("*information*estimates*.parquet"))
    if not candidates:
        raise SystemExit(f"no information estimates table under {output / 'tables'}")
    return pd.read_parquet(candidates[0])


def compare(fixed_dir: Path, adaptive_dir: Path) -> dict:
    fixed, adaptive = _load(fixed_dir), _load(adaptive_dir)
    keys = [c for c in ("cell_id", "metric", "estimator_variant", "conditioning_json") if c in fixed.columns]
    merged = fixed.merge(adaptive, on=keys, suffixes=("_fixed", "_adaptive"), validate="one_to_one")
    width = merged["ci_high_fixed"] - merged["ci_low_fixed"]
    ok = width.notna() & (width > 0)
    low_shift = ((merged["ci_low_adaptive"] - merged["ci_low_fixed"]).abs() / width)[ok]
    high_shift = ((merged["ci_high_adaptive"] - merged["ci_high_fixed"]).abs() / width)[ok]
    draws = merged["bootstrap_resamples_adaptive"]
    requested = int(merged["bootstrap_resamples_fixed"].max())
    same_estimate = np.isclose(merged["estimate_fixed"], merged["estimate_adaptive"], rtol=0, atol=0, equal_nan=True)
    same_p = np.isclose(merged["p_value_fixed"], merged["p_value_adaptive"], rtol=0, atol=0, equal_nan=True)
    per_metric = (merged.assign(low_shift=(merged["ci_low_adaptive"] - merged["ci_low_fixed"]).abs() / width,
                                high_shift=(merged["ci_high_adaptive"] - merged["ci_high_fixed"]).abs() / width)
                  .groupby("metric").agg(rows=("metric", "size"), draws_mean=("bootstrap_resamples_adaptive", "mean"),
                                         draws_min=("bootstrap_resamples_adaptive", "min"),
                                         low_shift_max=("low_shift", "max"), high_shift_max=("high_shift", "max"))
                  .reset_index())
    stages_fixed, stages_adaptive = _stages(fixed_dir / "analysis_manifest.json"), _stages(adaptive_dir / "analysis_manifest.json")
    return {
        "rows": int(len(merged)), "requested_draws": requested,
        "draws": {"mean": float(draws.mean()), "median": float(draws.median()), "min": int(draws.min()), "max": int(draws.max()),
                  "share_stopped_early": float((draws < requested).mean()),
                  "share_of_requested_work": float(draws.sum() / (requested * len(draws)))},
        "endpoint_shift_in_widths": {"low_median": float(low_shift.median()), "low_p95": float(low_shift.quantile(0.95)),
                                     "low_max": float(low_shift.max()), "high_median": float(high_shift.median()),
                                     "high_p95": float(high_shift.quantile(0.95)), "high_max": float(high_shift.max()),
                                     "intervals_compared": int(ok.sum())},
        "point_estimates_identical": bool(same_estimate.all()), "p_values_identical": bool(same_p.all()),
        "stage_seconds": {stage: {"fixed": stages_fixed.get(stage), "adaptive": stages_adaptive.get(stage)}
                          for stage in ("information_resampling", "derived_study_aggregates", "blackboard_calibration")
                          if stage in stages_fixed or stage in stages_adaptive},
        "total_seconds": {"fixed": sum(stages_fixed.values()), "adaptive": sum(stages_adaptive.values())},
        "per_metric": per_metric.to_dict("records"),
    }


def markdown(report: dict) -> str:
    lines = ["# Adaptive resampling vs fixed draws", ""]
    d, e = report["draws"], report["endpoint_shift_in_widths"]
    lines += [f"Rows compared: {report['rows']} (requested {report['requested_draws']} draws each).",
              f"Draws used: mean {d['mean']:.0f}, median {d['median']:.0f}, min {d['min']}, max {d['max']}; "
              f"{100 * d['share_stopped_early']:.0f} % of intervals stopped early; {100 * d['share_of_requested_work']:.0f} % of the requested work done.",
              f"Endpoint shift (in units of the full-draw width, {e['intervals_compared']} intervals): "
              f"low median {e['low_median']:.3f} / p95 {e['low_p95']:.3f} / max {e['low_max']:.3f}; "
              f"high median {e['high_median']:.3f} / p95 {e['high_p95']:.3f} / max {e['high_max']:.3f}.",
              f"Point estimates identical: {report['point_estimates_identical']}; p-values identical: {report['p_values_identical']}.", ""]
    lines += ["| stage | fixed s | adaptive s |", "|---|---|---|"]
    for stage, seconds in report["stage_seconds"].items():
        lines.append(f"| {stage} | {seconds['fixed'] or math.nan:.1f} | {seconds['adaptive'] or math.nan:.1f} |")
    lines.append(f"| total | {report['total_seconds']['fixed']:.1f} | {report['total_seconds']['adaptive']:.1f} |")
    lines += ["", "| metric | rows | draws mean | draws min | max low shift | max high shift |", "|---|---|---|---|---|---|"]
    for row in report["per_metric"]:
        lines.append(f"| {row['metric']} | {row['rows']} | {row['draws_mean']:.0f} | {row['draws_min']} | {row['low_shift_max']:.3f} | {row['high_shift_max']:.3f} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("fixed", type=Path)
    parser.add_argument("adaptive", type=Path)
    parser.add_argument("--markdown", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)
    report = compare(args.fixed, args.adaptive)
    if args.markdown:
        args.markdown.write_text(markdown(report))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, default=float) + "\n")
    print(markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
