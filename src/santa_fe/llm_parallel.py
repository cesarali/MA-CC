"""Apply the established MA-CC round information engine to Santa Fe trajectories.

This is a post-hoc adapter: no simulation or provider calls occur here.
"""
from __future__ import annotations

import argparse
import json
import hashlib
import math
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from mas_cc.games.hidden_bench.imitation.controller import ADVOCATE_TARGET, NO_OP
from mas_cc.games.hidden_bench.imitation_round_feedback.analysis import (
    adapt_round_record,
    round_information_analysis,
)

from .config import Config, load_config
from .measurements import bin01

DEFAULT_STATISTICS = (
    "round_sensing_mi",
    "round_target_sensing_mi",
    "round_sensor_action_mi",
    "round_population_actuation_cmi",
    "round_target_actuation_cmi",
    "round_truth_actuation_cmi",
    "round_kappa_target_actuation_cmi",
    "round_phi_target_actuation_cmi",
    "round_epistemic_target_actuation_cmi",
)

def _count(share: float, population: int) -> int:
    count = round(float(share) * population)
    if not 0 <= count <= population or abs(float(share) - count / population) > 1e-8:
        raise ValueError("round share does not correspond to an integer population count")
    return count


def adapt_trajectories(rounds: pd.DataFrame, *, bins: int) -> list:
    """Build pre-action round events; only rounds with a successor are eligible."""
    needed = {"cell_id", "seed", "round", "N", "budget", "controller_target",
              "target_share", "truth_share", "controller_observed_target_share",
              "controller_sensed_messages", "controller_effective_U", "controller_p_act",
              "kappa_mean_coverage", "kappa_population_coverage"}
    missing = needed - set(rounds)
    if missing:
        raise ValueError(f"synthetic trajectories are missing fields: {sorted(missing)}")
    events = []
    for (cell, seed), episode in rounds.groupby(["cell_id", "seed"], sort=False):
        episode = episode.sort_values("round")
        rows = episode.to_dict("records")
        for index in range(1, len(rows)):
            current = rows[index]
            future = rows[index + 1] if index + 1 < len(rows) else None
            time = int(current["round"])
            if future is not None and int(future["round"]) != time + 1:
                raise ValueError(f"nonconsecutive rounds in cell {cell}, episode {seed}")
            n = int(current["N"])
            if future is not None and int(future["N"]) != n:
                raise ValueError("population size changed within an episode")
            target_now = _count(current["target_share"], n)
            truth_now = _count(current["truth_share"], n)
            # The last round remains eligible for sensing MI only. Its after-state
            # placeholder is never used by an actuation statistic because U=None.
            target_next = _count(future["target_share"], n) if future is not None else target_now
            truth_next = _count(future["truth_share"], n) if future is not None else truth_now
            sensed = int(current["controller_sensed_messages"])
            sensed_target = _count(current["controller_observed_target_share"], sensed)
            controlled = int(current["budget"]) > 0 and future is not None
            action = (ADVOCATE_TARGET if int(current["controller_effective_U"]) else NO_OP) if controlled else None
            probability = float(current["controller_p_act"]) if controlled else None
            k_mean = int(bin01([current["kappa_mean_coverage"]], bins)[0])
            k_pop = int(bin01([current["kappa_population_coverage"]], bins)[0])
            record = {
                "round_index": time,
                "has_successor": future is not None,
                "episode_id": str(int(seed)),
                "occupation_counts_before": [target_now, n - target_now],
                "occupation_counts_after": [target_next, n - target_next],
                "target_count_before": target_now,
                "target_count_after": target_next,
                "truth_count_before": truth_now,
                "truth_count_after": truth_next,
                "sensor_count_vector": [sensed_target, sensed - sensed_target],
                "sensor_target_count": sensed_target,
                "controller_action": action,
                "controller_advocate_probability": probability,
                # Synthetic fact-coverage coordinates; they are not the LLM game's phi/kappa semantics.
                "conditioning_kappa_bin": k_mean,
                "conditioning_phi_bin": k_pop,
                "conditioning_epistemic_state": [k_mean, k_pop],
            }
            events.append(adapt_round_record(record, cell_id=str(int(cell))))
    return events


def _summarize_nulls(estimate_rows: list[dict], null_rows: list[dict]) -> list[dict]:
    by_stat: dict[str, list[float]] = {}
    for row in null_rows:
        by_stat.setdefault(row["statistic"], []).append(float(row["estimate"]))
    for row in estimate_rows:
        values = [v for v in by_stat.get(row["statistic"], []) if math.isfinite(v)]
        observed = float(row["estimate"])
        row["null_p_value"] = ((1 + sum(v >= observed for v in values)) / (1 + len(values))) if values else math.nan
        row["null_q025"] = float(np.quantile(values, .025)) if values else math.nan
        row["null_q975"] = float(np.quantile(values, .975)) if values else math.nan
        row["null_q95"] = float(np.quantile(values, .95)) if values else math.nan
        row["null_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else math.nan
        row["null_z_score"] = ((observed - float(row["null_mean"])) / row["null_std"]
                               if math.isfinite(row["null_std"]) and row["null_std"] > 0 else math.nan)
        row["estimate_minus_null"] = observed - float(row["null_mean"]) if values else math.nan
    return estimate_rows


def _plot_by_time(estimates: pd.DataFrame, path: Path) -> None:
    plot_dir = path / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    per_round = estimates.loc[estimates.scope == "per_round"]
    for (cell, name), group in tqdm(per_round.groupby(["cell_id", "statistic"]), desc="information plots"):
        group = group.sort_values("round")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(group["round"], group["estimate"], marker="o", markersize=2, label="observed")
        if group["null_mean"].notna().any():
            ax.plot(group["round"], group["null_mean"], linestyle="--", label="null mean")
        if group["bootstrap_ci_low"].notna().any():
            ax.fill_between(group["round"].to_numpy(dtype=float),
                            group["bootstrap_ci_low"].to_numpy(dtype=float),
                            group["bootstrap_ci_high"].to_numpy(dtype=float), alpha=.2, label="episode bootstrap CI")
        ax.set(xlabel="round t (action at t, outcome at t+1)", ylabel="information (bits)",
               title=f"Cell {cell}: {name}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(plot_dir / f"cell_{cell}_{name}.png", dpi=150)
        plt.close(fig)


def _report(estimates: pd.DataFrame, config: Config, root: Path) -> None:
    confidence = float(config.raw.get("analysis", {}).get("llm_parallel", {}).get("confidence", .95))
    lines = ["# Santa Fe information by round", "",
             "This analysis uses the same MA-CC round-information estimator as the LLM studies.",
             "Actuation CMI is predictive transfer information, not a causal effect.",
             "", "## Mapping", "",
             "- State: target/truth vote counts after synthetic round `t`.",
             "- Sensor: target votes in the board messages sensed when round `t` closes.",
             "- Action: recommendation posted after round `t`, affecting the next board.",
             "- Outcome: vote counts after round `t+1`.",
             "- Extra synthetic conditioning: binned mean and population fact coverage.",
             "- Zero-budget cells have no physical controller action, so actuation information is unavailable there.",
             "- Round 0 has no sensor record. Sensing MI uses rounds 1 through R; transfer CMI uses rounds 1 through R−1.",
             "", "## Cell-wide estimates", ""]
    pooled = estimates.loc[estimates.scope == "pooled"].sort_values(["cell_id", "statistic"])
    lines += [f"| cell | statistic | estimate (bits) | bootstrap {confidence:.0%} CI | null mean | p | episodes | rounds |",
              "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |"]
    for row in pooled.itertuples():
        ci = f"[{row.bootstrap_ci_low:.3f}, {row.bootstrap_ci_high:.3f}]" if pd.notna(row.bootstrap_ci_low) else "—"
        null = f"{row.null_mean:.3f}" if pd.notna(row.null_mean) else "—"
        p = f"{row.null_p_value:.3f}" if pd.notna(row.null_p_value) else "—"
        lines.append(f"| {row.cell_id} | {row.statistic} | {row.estimate:.3f} | {ci} | {null} | {p} | {row.n_episodes} | {row.n_rounds} |")
    lines += ["", "## Time series", "",
              "Every eligible transition round is in `round_information_estimates.parquet` and `.csv`.",
              "The `plots/` directory shows each statistic by round with its null mean and episode bootstrap interval.",
              "Support diagnostics accompany every estimate; sparse round cells may have weak action overlap.",
              "With few episodes, per-round direct-counting MI can have substantial finite-sample bias; compare the null mean and support before interpreting peaks.", ""]
    focus = ("round_target_sensing_mi", "round_sensor_action_mi",
             "round_target_actuation_cmi", "round_kappa_target_actuation_cmi")
    for cell, subset in estimates.loc[estimates.scope == "per_round"].groupby("cell_id"):
        lines += [f"### Cell {cell}: bits at each round", "",
                  "| round | sensing MI | sensor→action MI | target transfer CMI | coverage-conditioned CMI | transfer-null p |",
                  "| ---: | ---: | ---: | ---: | ---: | ---: |"]
        for time, rows in subset.groupby("round"):
            by_name = rows.set_index("statistic")
            def value(name: str, column: str = "estimate") -> str:
                if name not in by_name.index:
                    return "—"
                x = by_name.loc[name, column]
                return f"{x:.3f}" if pd.notna(x) else "—"
            lines.append(f"| {int(time)} | {value(focus[0])} | {value(focus[1])} | "
                         f"{value(focus[2])} | {value(focus[3])} | {value(focus[2], 'null_p_value')} |")
        lines.append("")
    (root / "round_information_report.md").write_text("\n".join(lines), encoding="utf-8")


def _analyze_group(payload: tuple) -> tuple[list[dict], list[dict]]:
    cell, scope, time, rows, statistics, bootstrap, permutations, confidence, seed = payload
    group_seed = seed + int(cell) * 1_000_003 + (0 if time is None else int(time))
    estimate_rows, null_rows = round_information_analysis(
        rows, statistics=statistics, bootstrap_resamples=bootstrap,
        null_permutations=permutations, confidence=confidence, seed=group_seed)
    estimates = [{"cell_id": int(cell), "scope": scope, "round": time, **row}
                 for row in _summarize_nulls(estimate_rows, null_rows)]
    nulls = [{"cell_id": int(cell), "scope": scope, "round": time, **row} for row in null_rows]
    return estimates, nulls


def analyze_existing(config: Config) -> dict:
    source = config.results_dir / "trajectories" / "round_trajectories.parquet"
    if not source.is_file():
        raise FileNotFoundError(f"saved round trajectories required: {source}")
    settings = config.raw.get("analysis", {}).get("llm_parallel", {})
    if not isinstance(settings, dict):
        raise ValueError("analysis.llm_parallel must be a mapping")
    statistics = tuple(settings.get("statistics", DEFAULT_STATISTICS))
    bootstrap = int(settings.get("bootstrap_resamples", 100))
    permutations = int(settings.get("null_permutations", 100))
    confidence = float(settings.get("confidence", .95))
    seed = int(settings.get("seed", config.seed))
    processes = int(settings.get("processes", 1))
    if processes < 1 or bootstrap < 0 or permutations < 0 or not 0 < confidence < 1:
        raise ValueError("invalid llm_parallel resampling settings")
    rounds = pd.read_parquet(source)
    root = config.results_dir / "information" / "llm_parallel"
    root.mkdir(parents=True, exist_ok=True)
    estimates, nulls = [], []
    transition_rounds: set[int] = set()
    sensing_rounds: set[int] = set()
    cells = 0
    for cell, cell_rounds in tqdm(rounds.groupby("cell_id"), desc="information cells"):
        events = adapt_trajectories(cell_rounds, bins=config.bins)
        if not events:
            continue
        cells += 1
        round_groups: dict[int, list] = {}
        for event in events:
            round_groups.setdefault(event.round_index, []).append(event)
            sensing_rounds.add(event.round_index)
            if event.event["has_successor"]:
                transition_rounds.add(event.round_index)
        scopes = [("pooled", None, events), *[("per_round", time, rows) for time, rows in sorted(round_groups.items())]]
        payloads = [(cell, scope, time, rows, statistics, bootstrap, permutations, confidence, seed)
                    for scope, time, rows in scopes]
        if processes == 1:
            iterator = map(_analyze_group, payloads)
            for estimate_rows, null_rows in tqdm(iterator, total=len(payloads), desc=f"information cell {cell}", leave=False):
                estimates.extend(estimate_rows)
                nulls.extend(null_rows)
        else:
            with ProcessPoolExecutor(max_workers=processes, mp_context=get_context("spawn")) as pool:
                iterator = pool.map(_analyze_group, payloads, chunksize=1)
                for estimate_rows, null_rows in tqdm(iterator, total=len(payloads), desc=f"information cell {cell}", leave=False):
                    estimates.extend(estimate_rows)
                    nulls.extend(null_rows)
    if not cells:
        raise ValueError("no eligible round transitions found")
    estimate_frame = pd.DataFrame(estimates)
    null_frame = pd.DataFrame(nulls)
    estimate_frame.to_parquet(root / "round_information_estimates.parquet", index=False)
    estimate_frame.to_csv(root / "round_information_estimates.csv", index=False)
    null_frame.to_parquet(root / "round_information_nulls.parquet", index=False)
    _plot_by_time(estimate_frame, root)
    _report(estimate_frame, config, root)
    metadata = {"source_trajectories": str(source), "statistics": statistics,
                "bootstrap_resamples": bootstrap, "null_permutations": permutations,
                "confidence": confidence, "seed": seed, "bins": config.bins, "processes": processes,
                "engine": "mas_cc.games.hidden_bench.imitation_round_feedback.analysis.round_information_analysis"}
    metadata["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    (root / "analysis_config.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (root / "analysis_recipe.yaml").write_bytes(config.path.read_bytes())
    return {"results_dir": str(root), "estimates": len(estimate_frame), "null_draws": len(null_frame),
            "cells": cells, "transition_rounds": sorted(transition_rounds),
            "sensing_rounds": sorted(sensing_rounds)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    result = analyze_existing(load_config(args.config))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
