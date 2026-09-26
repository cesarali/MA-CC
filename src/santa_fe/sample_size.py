"""Reference-data subsampling calibration."""
from __future__ import annotations

from dataclasses import replace
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .bootstrap import bootstrap_cell, metrics
from .config import Cell, Config
from .measurements import STATISTICS, transitions
from .nulls import permutation_test
from .runner import run


def _subset(groups: dict[int, pd.DataFrame], ids: np.ndarray) -> pd.DataFrame:
    return pd.concat([groups[int(i)] for i in ids], ignore_index=True)


def calibrate(config: Config) -> pd.DataFrame:
    settings = config.sample_size
    root = config.results_dir / "sample_size"
    root.mkdir(parents=True, exist_ok=True)
    reference_n = int(settings.get("reference_episodes", 1000))
    repetitions = int(settings.get("repetitions", 200))
    counts = settings.get("episode_counts", [10, 20, 30, 50, 75, 100])
    confidence = float(config.bootstrap.get("confidence_level", .95))
    n_boot = int(settings.get("n_bootstrap", 100))
    n_perm = int(settings.get("n_permutations", 100))
    rows = []
    for cell in tqdm(config.cells, desc="sample-size cells"):
        ref_config = replace(config, cells=(cell,), episodes=reference_n, seed=config.seed + 700_000)
        reference, _ = run(ref_config)
        reference.to_parquet(root / f"reference_cell_{cell.cell_id}.parquet", index=False)
        ref_metrics = metrics(reference, config.bins, config.coordinate)
        null_cell = Cell(cell.cell_id, replace(cell.params, budget_fraction=0.0))
        null_config = replace(config, cells=(null_cell,), episodes=reference_n, seed=config.seed + 900_000)
        null_reference, _ = run(null_config)
        null_reference.to_parquet(root / f"no_control_cell_{cell.cell_id}.parquet", index=False)
        ref_groups = {int(s): d for s, d in reference.groupby("seed")}
        null_groups = {int(s): d for s, d in null_reference.groupby("seed")}
        ref_ids, null_ids = list(ref_groups), list(null_groups)
        rng = np.random.default_rng(config.seed + cell.cell_id + 500_000)
        for n in tqdm(counts, desc=f"sample sizes cell {cell.cell_id}", leave=False):
            if n > reference_n:
                raise ValueError("sample size cannot exceed reference episodes")
            for rep in tqdm(range(repetitions), desc=f"replicates n={n}", leave=False):
                sample = _subset(ref_groups, [ref_ids[i] for i in rng.choice(len(ref_ids), n, replace=False)])
                values = metrics(sample, config.bins, config.coordinate)
                ci = bootstrap_cell(sample, config.bins, config.coordinate, n_boot, confidence,
                                    config.seed + cell.cell_id * 1_000_003 + n * 1000 + rep, progress=False)
                ci_by_metric = ci.set_index("metric")
                null_summary, _ = permutation_test(transitions(sample), config.bins, config.coordinate,
                                                   n_perm, config.seed + rep + n * 1000, progress=False)
                p_values = dict(zip(null_summary.statistic, null_summary.p_value))
                sham = _subset(null_groups, [null_ids[i] for i in rng.choice(len(null_ids), n, replace=False)]).copy()
                sham["controller_effective_U"] = sham["controller_U"]  # policy decision without posted control
                sham_summary, _ = permutation_test(transitions(sham), config.bins, config.coordinate,
                                                   n_perm, config.seed + rep + n * 2000, progress=False)
                sham_p = dict(zip(sham_summary.statistic, sham_summary.p_value))
                for metric, value in values.items():
                    interval = ci_by_metric.loc[metric]
                    truth = ref_metrics[metric]
                    rows.append({"cell_id": cell.cell_id, "n_episodes": n, "repetition": rep,
                                 "metric": metric, "estimate": value, "reference": truth,
                                 "bias": value - truth, "ci_width": interval.ci_high - interval.ci_low,
                                 "ci_covers_reference": bool(interval.ci_low <= truth <= interval.ci_high) if pd.notna(truth) and pd.notna(interval.ci_low) and pd.notna(interval.ci_high) else None,
                                 "detected": bool(p_values[metric] < .05) if metric in STATISTICS and pd.notna(p_values[metric]) else None,
                                 "false_positive": bool(sham_p[metric] < .05) if metric in STATISTICS[1:] and pd.notna(sham_p[metric]) else None})
    details = pd.DataFrame(rows)
    details.to_parquet(root / "sample_size_repetitions.parquet", index=False)
    summary = details.groupby(["cell_id", "n_episodes", "metric"], as_index=False).agg(
        bias=("bias", "mean"), variance=("estimate", "var"), mean_ci_width=("ci_width", "mean"),
        ci_coverage=("ci_covers_reference", "mean"), detection_probability=("detected", "mean"),
        false_positive_rate=("false_positive", "mean"))
    summary.to_csv(root / "sample_size_summary.csv", index=False)
    return summary
