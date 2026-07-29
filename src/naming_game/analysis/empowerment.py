"""End-to-end empowerment analysis over stored Parquet trajectories."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .estimators import Estimate, conditional_mutual_information, mutual_information
from .metrics import GROUP_COLUMNS, add_efficiency, summarize_episode_metrics
from .surrogates import circular_shift_trajectories, shuffle_episode_labels, swap_labels_half

STRATA_COLUMNS = [column for column in GROUP_COLUMNS if column != "committee_policy"]


@dataclass(frozen=True)
class AnalysisConfig:
    horizons_population_rounds: tuple[int, ...] = (1, 3, 5, 10)
    bootstrap_resamples: int = 1000
    null_permutations: int = 1000
    confidence: float = 0.95
    seed: int = 1

    def __post_init__(self) -> None:
        if any(value < 1 for value in self.horizons_population_rounds):
            raise ValueError("Analysis horizons must be positive.")
        if self.bootstrap_resamples < 0 or self.null_permutations < 0:
            raise ValueError("Resample counts cannot be negative.")
        if not 0 < self.confidence < 1:
            raise ValueError("confidence must lie in (0, 1).")


def _estimate_fields(estimate: Estimate) -> dict[str, Any]:
    return asdict(estimate)


def _bootstrap_interval(
    ids: np.ndarray,
    estimator: Callable[[np.ndarray], float],
    *,
    resamples: int,
    confidence: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    if resamples == 0 or len(ids) == 0:
        return math.nan, math.nan
    values = np.asarray(
        [estimator(rng.choice(ids, size=len(ids), replace=True)) for _ in range(resamples)],
        dtype=float,
    )
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return math.nan, math.nan
    alpha = (1 - confidence) / 2
    return float(np.quantile(values, alpha)), float(np.quantile(values, 1 - alpha))


def estimate_terminal(
    episodes: pd.DataFrame, config: AnalysisConfig, *, resolved_only: bool = False
) -> pd.DataFrame:
    source = episodes[episodes["final_convention"].isin(("A", "B"))] if resolved_only else episodes
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(config.seed + int(resolved_only))
    for keys, group in source.groupby(STRATA_COLUMNS, dropna=False, sort=True):
        policy_levels = tuple(sorted(group["committee_policy"].dropna().unique()))
        outcome_levels = ("A", "B") if resolved_only else ("A", "B", "unresolved")

        def evaluate(frame: pd.DataFrame) -> Estimate:
            return mutual_information(
                frame["committee_policy"].tolist(),
                frame["final_convention"].tolist(),
                x_levels=policy_levels,
                y_levels=outcome_levels,
            )

        estimate = evaluate(group)
        indexed = group.set_index("episode_id", drop=False)

        def sampled(sampled_ids: np.ndarray) -> float:
            sampled_frame = pd.concat([indexed.loc[[episode_id]] for episode_id in sampled_ids], ignore_index=True)
            return evaluate(sampled_frame).jeffreys

        low, high = _bootstrap_interval(
            group["episode_id"].to_numpy(), sampled,
            resamples=config.bootstrap_resamples, confidence=config.confidence, rng=rng,
        )
        row = dict(zip(STRATA_COLUMNS, keys, strict=True))
        row.update(
            statistic="terminal_resolved_only" if resolved_only else "terminal_all",
            horizon_population_rounds=None,
            ci_low=low,
            ci_high=high,
            **_estimate_fields(estimate),
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _lag_pairs(group: pd.DataFrame, lag: int, state_column: str) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for _, episode in group.groupby("episode_id", sort=False):
        ordered = episode.sort_values("interaction_index")
        pair = ordered[["episode_id", "committee_policy", state_column]].copy()
        pair["future"] = pair[state_column].shift(-lag)
        parts.append(pair.dropna(subset=[state_column, "future"]))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def estimate_lagged(
    interactions: pd.DataFrame,
    config: AnalysisConfig,
    *,
    state_column: str = "macrostate_binary",
    state_levels: tuple[Any, ...] = (0.0, 1.0),
    statistic: str = "lagged_binary",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(config.seed + 100)
    for keys, group in interactions.groupby(STRATA_COLUMNS, dropna=False, sort=True):
        n_agents = int(group["N"].iloc[0])
        policy_levels = tuple(sorted(group["committee_policy"].dropna().unique()))
        episode_ids = group["episode_id"].drop_duplicates().to_numpy()
        episode_frames = {episode_id: frame for episode_id, frame in group.groupby("episode_id", sort=False)}
        for horizon in config.horizons_population_rounds:
            lag = horizon * n_agents

            def evaluate(frame: pd.DataFrame) -> Estimate:
                pairs = _lag_pairs(frame, lag, state_column)
                if pairs.empty:
                    return Estimate(math.nan, math.nan, math.nan, 0)
                return conditional_mutual_information(
                    pairs["committee_policy"].tolist(), pairs["future"].tolist(),
                    pairs[state_column].tolist(), x_levels=policy_levels,
                    y_levels=state_levels, z_levels=state_levels,
                )

            estimate = evaluate(group)

            def sampled(sampled_ids: np.ndarray) -> float:
                sample = pd.concat([episode_frames[episode_id] for episode_id in sampled_ids], ignore_index=True)
                # Assign unique IDs so duplicate bootstrap draws remain independent trajectories.
                sizes = [len(episode_frames[episode_id]) for episode_id in sampled_ids]
                sample["episode_id"] = np.repeat(np.arange(len(sampled_ids)), sizes)
                return evaluate(sample).jeffreys

            low, high = _bootstrap_interval(
                episode_ids, sampled, resamples=config.bootstrap_resamples,
                confidence=config.confidence, rng=rng,
            )
            row = dict(zip(STRATA_COLUMNS, keys, strict=True))
            row.update(
                statistic=statistic, horizon_population_rounds=horizon,
                ci_low=low, ci_high=high, **_estimate_fields(estimate),
            )
            rows.append(row)
    return pd.DataFrame(rows)


def estimate_nulls(
    interactions: pd.DataFrame, episodes: pd.DataFrame, config: AnalysisConfig
) -> pd.DataFrame:
    if config.null_permutations == 0:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    shuffle_strata = [
        "regime", "committee_size", "pulse_rounds", "provider", "model",
        "prompt_version", "initial_condition",
    ]
    for permutation in range(config.null_permutations):
        seed = config.seed + 10_000 + permutation
        shuffled = shuffle_episode_labels(episodes, strata=shuffle_strata, seed=seed)
        mapping = shuffled.set_index("episode_id")["committee_policy"]
        shuffled_interactions = interactions.copy()
        shuffled_interactions["committee_policy"] = shuffled_interactions["episode_id"].map(mapping)
        terminal = estimate_terminal(shuffled, AnalysisConfig(config.horizons_population_rounds, 0, 0, config.confidence, seed))
        null_config = AnalysisConfig(config.horizons_population_rounds, 0, 0, config.confidence, seed)
        lagged = pd.concat(
            [
                estimate_lagged(shuffled_interactions, null_config),
                estimate_lagged(
                    shuffled_interactions, null_config, state_column="macrostate_three",
                    state_levels=("B_dominant", "mixed", "A_dominant"), statistic="lagged_three",
                ),
            ],
            ignore_index=True,
        )
        for frame, null_type in ((terminal, "episode_label_shuffle_null"), (lagged, "episode_label_shuffle_null")):
            for record in frame.to_dict("records"):
                rows.append({**record, "null_type": null_type, "permutation": permutation})
        shifted = circular_shift_trajectories(interactions, seed=seed)
        shifted_lagged = estimate_lagged(shifted, null_config)
        for record in shifted_lagged.to_dict("records"):
            rows.append({**record, "null_type": "circular_shift_null", "permutation": permutation})
    return pd.DataFrame(rows)


def _make_plots(
    interactions: pd.DataFrame,
    metrics: pd.DataFrame,
    estimates: pd.DataFrame,
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    terminal = estimates[estimates["statistic"] == "terminal_all"]
    lagged = estimates[estimates["statistic"] == "lagged_binary"]
    terminal = terminal.assign(committee_fraction=terminal["committee_size"] / terminal["N"])
    metrics = metrics.assign(committee_fraction=metrics["committee_size"] / metrics["N"])

    def simple_plot(frame: pd.DataFrame, x: str, y: str, filename: str, ylabel: str) -> None:
        fig, ax = plt.subplots(figsize=(7, 4))
        if not frame.empty:
            grouped = frame.groupby(x, dropna=False)[y].mean().sort_index()
            ax.plot(grouped.index, grouped.values, marker="o")
        ax.set(xlabel=x.replace("_", " "), ylabel=ylabel)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=160)
        plt.close(fig)

    simple_plot(terminal, "committee_fraction", "jeffreys", "terminal_empowerment.png", "terminal empowerment (bits)")
    simple_plot(metrics, "committee_fraction", "takeover_probability", "takeover_probability.png", "takeover probability")
    simple_plot(lagged, "horizon_population_rounds", "jeffreys", "lagged_empowerment.png", "lagged empowerment (bits)")
    pulse = interactions[interactions["regime"] == "pulse"]
    fig, ax = plt.subplots(figsize=(7, 4))
    for keys, group in pulse.groupby(["initial_condition", "committee_policy", "pulse_rounds"], dropna=False):
        trajectory = group.groupby("population_round")["rolling_share_A"].mean()
        ax.plot(trajectory.index, trajectory.values, label=" / ".join(map(str, keys)))
    ax.set(xlabel="population round", ylabel="mean rolling share A")
    if not pulse.empty:
        ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "pulse_trajectories.png", dpi=160)
    plt.close(fig)

    fig, first_axis = plt.subplots(figsize=(7, 4))
    recovery = metrics.groupby("committee_fraction")["mean_recovery_time_interactions"].mean()
    efficiency = terminal.groupby("committee_fraction")["efficiency"].mean()
    first_axis.plot(recovery.index, recovery.values, marker="o", color="tab:blue")
    first_axis.set(xlabel="committee fraction", ylabel="mean recovery time", )
    second_axis = first_axis.twinx()
    second_axis.plot(efficiency.index, efficiency.values, marker="s", color="tab:orange")
    second_axis.set_ylabel("empowerment / action")
    fig.tight_layout()
    fig.savefig(output_dir / "recovery_efficiency.png", dpi=160)
    plt.close(fig)


def analyze_histories(
    history_dir: str | Path,
    output_dir: str | Path,
    config: AnalysisConfig | None = None,
    column_mapping: dict[str, str] | None = None,
) -> dict[str, str | int]:
    settings = config or AnalysisConfig()
    source = Path(history_dir)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    interactions = pd.read_parquet(source / "interactions.parquet")
    episodes = pd.read_parquet(source / "episodes.parquet")
    if column_mapping:
        interactions = interactions.rename(columns=column_mapping)
        episodes = episodes.rename(columns=column_mapping)
    terminal_all = estimate_terminal(episodes, settings)
    terminal_resolved = estimate_terminal(episodes, settings, resolved_only=True)
    lagged_binary = estimate_lagged(interactions, settings)
    lagged_three = estimate_lagged(
        interactions,
        settings,
        state_column="macrostate_three",
        state_levels=("B_dominant", "mixed", "A_dominant"),
        statistic="lagged_three",
    )
    estimates = pd.concat(
        [terminal_all, terminal_resolved, lagged_binary, lagged_three], ignore_index=True
    )
    metrics = summarize_episode_metrics(episodes)
    estimates = add_efficiency(metrics, estimates)
    nulls = estimate_nulls(interactions, episodes, settings)
    swapped_interactions, swapped_episodes = swap_labels_half(interactions, episodes, seed=settings.seed)
    swapped_terminal = estimate_terminal(
        swapped_episodes,
        AnalysisConfig(settings.horizons_population_rounds, 0, 0, settings.confidence, settings.seed),
    )
    invariance = terminal_all.merge(swapped_terminal, on=STRATA_COLUMNS + ["statistic"], suffixes=("_original", "_swapped"), how="outer")
    invariance["absolute_difference"] = (invariance["jeffreys_original"] - invariance["jeffreys_swapped"]).abs()
    invariance["invariant_within_tolerance"] = invariance["absolute_difference"] <= 1e-9
    estimates.to_parquet(destination / "empowerment_estimates.parquet", index=False)
    metrics.to_parquet(destination / "episode_metrics.parquet", index=False)
    nulls.to_parquet(destination / "null_results.parquet", index=False)
    invariance.to_parquet(destination / "label_swap_invariance.parquet", index=False)
    baseline = terminal_all[terminal_all["committee_size"] == 0].copy()
    if not baseline.empty and not nulls.empty:
        null_baseline = (
            nulls[
                (nulls["committee_size"] == 0)
                & (nulls["statistic"] == "terminal_all")
                & (nulls["null_type"] == "episode_label_shuffle_null")
            ]
            .groupby(STRATA_COLUMNS, dropna=False)["jeffreys"]
            .quantile(0.95)
            .rename("shuffle_95pct")
            .reset_index()
        )
        baseline = baseline.merge(null_baseline, on=STRATA_COLUMNS, how="left")
        baseline["near_zero_baseline"] = baseline["jeffreys"] <= baseline["shuffle_95pct"].fillna(0.05).clip(lower=0.05)
    baseline.to_parquet(destination / "no_committee_baseline.parquet", index=False)
    _make_plots(interactions, metrics, estimates, destination / "plots")
    (destination / "analysis_config.json").write_text(json.dumps(asdict(settings), indent=2), encoding="utf-8")
    return {
        "estimates": len(estimates), "metric_rows": len(metrics), "null_rows": len(nulls),
        "output_dir": str(destination),
    }


__all__ = ["AnalysisConfig", "analyze_histories", "estimate_lagged", "estimate_nulls", "estimate_terminal"]
