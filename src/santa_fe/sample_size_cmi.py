"""Slurm-ready sample-size calibration using the shared round CMI/null engine."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
from multiprocessing import get_context
import os
from pathlib import Path
import shlex

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mas_cc.games.hidden_bench.imitation_round_feedback.analysis import round_information_analysis
from .config import load_config
from .llm_parallel import adapt_trajectories, _summarize_nulls
from .runner import run


def _root(config) -> Path:
    return config.results_dir / "sample_size_cmi"


def _reference(config, cell_id: int) -> Path:
    return _root(config) / f"reference_cell_{cell_id}.parquet"


def _check_recipe(config) -> None:
    root = _root(config)
    root.mkdir(parents=True, exist_ok=True)
    recipe = root / "analysis_recipe.yaml"
    if recipe.is_file() and recipe.read_bytes() != config.path.read_bytes():
        raise ValueError("sample-size result root belongs to a different config")
    recipe.write_bytes(config.path.read_bytes())


def prepare_reference(config, cell_id: int) -> dict:
    _check_recipe(config)
    root = _root(config)
    path = _reference(config, cell_id)
    if path.is_file():
        existing = pd.read_parquet(path)
        if existing.seed.nunique() == int(config.sample_size["reference_episodes"]):
            return {"cell_id": cell_id, "status": "already_complete"}
    cell = config.cells[cell_id]
    reference_config = replace(config, cells=(cell,), episodes=int(config.sample_size["reference_episodes"]),
                               seed=config.seed + 700_000)
    rounds, _ = run(reference_config)
    temporary = path.with_suffix(".tmp.parquet")
    rounds.to_parquet(temporary, index=False)
    os.replace(temporary, path)
    return {"cell_id": cell_id, "episodes": rounds.seed.nunique(), "path": str(path)}


_REP_CONTEXT = None


def _init_replicates(ids, groups, statistic, n_boot, n_perm, confidence, seed, cell_id, n):
    global _REP_CONTEXT
    _REP_CONTEXT = (ids, groups, statistic, n_boot, n_perm, confidence, seed, cell_id, n)


def _replicate(rep):
    ids, groups, statistic, n_boot, n_perm, confidence, seed, cell_id, n = _REP_CONTEXT
    replicate_seed = seed + cell_id * 1_000_003 + n * 10_000 + rep
    rng = np.random.default_rng(replicate_seed)
    chosen = rng.choice(ids, n, replace=False)
    events = [event for episode_id in chosen for event in groups[episode_id]]
    estimates, nulls = round_information_analysis(events, statistics=(statistic,),
        bootstrap_resamples=n_boot, null_permutations=n_perm, confidence=confidence,
        seed=replicate_seed)
    row = _summarize_nulls(estimates, nulls)[0]
    return {"cell_id": cell_id, "n_episodes": n, "repetition": rep, "statistic": statistic,
            "estimate": row["estimate"], "null_mean": row["null_mean"],
            "estimate_minus_null": row["estimate_minus_null"], "null_p_value": row["null_p_value"],
            "ci_width": row["bootstrap_ci_high"] - row["bootstrap_ci_low"],
            "detected": bool(row["null_p_value"] < .05), "n_rounds": row["n_rounds"]}


def run_task(config, task_id: int) -> dict:
    _check_recipe(config)
    counts = config.sample_size["episode_counts"]
    cell_id, index = divmod(task_id, len(counts))
    if cell_id >= len(config.cells):
        raise ValueError("sample-size task ID outside array")
    n = int(counts[index])
    output = _root(config) / f"cell_{cell_id}_n_{n}.parquet"
    if output.is_file() and len(pd.read_parquet(output)) == int(config.sample_size["repetitions"]):
        return {"task_id": task_id, "status": "already_complete"}
    path = _reference(config, cell_id)
    if not path.is_file():
        raise FileNotFoundError(f"prepare reference first: {path}")
    rounds = pd.read_parquet(path)
    events = adapt_trajectories(rounds, bins=config.bins)
    groups = {}
    for event in events:
        groups.setdefault(event.episode_id, []).append(event)
    ids = list(groups)
    if n > len(ids):
        raise ValueError("episode count exceeds reference episodes")
    settings = config.sample_size
    context = (ids, groups, settings.get("statistic", "round_target_actuation_cmi"),
               int(settings["n_bootstrap"]), int(settings["n_permutations"]), .95,
               config.seed, cell_id, n)
    repetitions = int(settings["repetitions"])
    cpus = min(config.processes, int(os.environ.get("SLURM_CPUS_PER_TASK", config.processes)))
    if cpus == 1:
        _init_replicates(*context)
        rows = list(map(_replicate, range(repetitions)))
    else:
        with ProcessPoolExecutor(max_workers=cpus, mp_context=get_context("spawn"),
                                 initializer=_init_replicates, initargs=context) as pool:
            rows = list(pool.map(_replicate, range(repetitions), chunksize=1))
    table = pd.DataFrame(rows)
    table["rho"] = config.cells[cell_id].params.rho
    temporary = output.with_suffix(".tmp.parquet")
    table.to_parquet(temporary, index=False)
    os.replace(temporary, output)
    return {"task_id": task_id, "cell_id": cell_id, "n_episodes": n,
            "repetitions": len(table), "path": str(output)}


def aggregate(config) -> dict:
    _check_recipe(config)
    root = _root(config)
    paths = [root / f"cell_{cell.cell_id}_n_{n}.parquet" for cell in config.cells
             for n in config.sample_size["episode_counts"]]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete sample-size tasks: {missing}")
    details = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    expected = len(paths) * int(config.sample_size["repetitions"])
    if len(details) != expected:
        raise ValueError(f"expected {expected} sample-size repetitions, got {len(details)}")
    details.to_parquet(root / "sample_size_cmi_repetitions.parquet", index=False)
    details.to_csv(root / "sample_size_cmi_repetitions.csv", index=False)
    summary = details.groupby(["cell_id", "rho", "n_episodes"], as_index=False).agg(
        estimate_mean=("estimate", "mean"), estimator_variance=("estimate", "var"),
        corrected_mean=("estimate_minus_null", "mean"), corrected_variance=("estimate_minus_null", "var"),
        p_value_median=("null_p_value", "median"), p_value_q025=("null_p_value", lambda x: x.quantile(.025)),
        p_value_q975=("null_p_value", lambda x: x.quantile(.975)),
        mean_ci_width=("ci_width", "mean"), detection_probability=("detected", "mean"))
    summary.to_csv(root / "sample_size_cmi_summary.csv", index=False)
    for rho, rows in summary.groupby("rho"):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(rows.n_episodes, rows.detection_probability, marker="o")
        ax.axvline(50, linestyle="--", color="gray")
        ax.set(xlabel="episodes per cell", ylabel="P(null p<0.05)", ylim=(0, 1), title=f"CMI detection; rho={rho:g}")
        fig.tight_layout()
        fig.savefig(root / f"detection_rho_{rho:g}.png", dpi=150)
        plt.close(fig)
    (root / "analysis_recipe.yaml").write_bytes(config.path.read_bytes())
    return {"repetitions": len(details), "summary_rows": len(summary), "path": str(root)}


def plan(config, python: str, repository_root: str) -> Path:
    _check_recipe(config)
    root = _root(config)
    (root / "logs").mkdir(exist_ok=True)
    cells = len(config.cells)
    tasks = cells * len(config.sample_size["episode_counts"])
    cfg = shlex.quote(str(config.path))
    py = shlex.quote(python)
    repo = shlex.quote(repository_root)
    logs = shlex.quote(str(root / "logs"))
    base = f"cd {repo} && export PYTHONPATH={repo}/src MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 && {py} -m santa_fe.sample_size_cmi"
    ref = shlex.quote(base + f" prepare-reference --config {cfg} --cell-id $SLURM_ARRAY_TASK_ID")
    task = shlex.quote(base + f" run-task --config {cfg} --task-id $SLURM_ARRAY_TASK_ID")
    finish = shlex.quote(base + f" aggregate --config {cfg}")
    lines = ["#!/usr/bin/env bash", "set -euo pipefail",
             f"ref_job=$(sbatch --parsable --job-name=sf-cmi-ref --array=0-{cells-1}%2 --cpus-per-task=4 --mem=8G --time=02:00:00 --output={logs}/ref-%A_%a.out --wrap={ref})",
             f"size_job=$(sbatch --parsable --job-name=sf-cmi-size --dependency=afterok:$ref_job --array=0-{tasks-1}%8 --cpus-per-task=4 --mem=12G --time=12:00:00 --output={logs}/size-%A_%a.out --wrap={task})",
             f"final_job=$(sbatch --parsable --job-name=sf-cmi-final --dependency=afterok:$size_job --cpus-per-task=2 --mem=8G --time=01:00:00 --output={logs}/final-%j.out --wrap={finish})",
             'printf "reference=%s sample_size=%s final=%s\\n" "$ref_job" "$size_job" "$final_job"', ""]
    path = root / "submit_sample_size_cygnus.sh"
    path.write_text("\n".join(lines))
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare-reference", "run-task", "aggregate", "plan-cygnus"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--cell-id", type=int)
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--python")
    parser.add_argument("--repository-root")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.operation == "prepare-reference":
        result = prepare_reference(config, args.cell_id if args.cell_id is not None else int(os.environ["SLURM_ARRAY_TASK_ID"]))
    elif args.operation == "run-task":
        result = run_task(config, args.task_id if args.task_id is not None else int(os.environ["SLURM_ARRAY_TASK_ID"]))
    elif args.operation == "aggregate":
        result = aggregate(config)
    else:
        if not args.python or not args.repository_root:
            parser.error("plan-cygnus needs --python and --repository-root")
        result = {"script": str(plan(config, args.python, args.repository_root))}
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
