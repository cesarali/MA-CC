"""End-to-end empowerment analysis over stored Parquet trajectories."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .estimators import (
    Estimate,
    conditional_mutual_information_from_counts,
    mutual_information_from_counts,
)
from .metrics import GROUP_COLUMNS, add_efficiency, summarize_episode_metrics
from .reporting import (
    attach_null_summary,
    collect_warnings,
    make_experiment_summary,
    make_pulse_summary,
    normalize_histories,
    write_summary_markdown,
)
from .surrogates import circular_shift_trajectories, shuffle_episode_labels, swap_labels_half

STRATA_COLUMNS = [column for column in GROUP_COLUMNS if column != "committee_policy"]


@dataclass(frozen=True)
class AnalysisConfig:
    horizons_population_rounds: tuple[int, ...] = (1, 3, 5, 10)
    bootstrap_resamples: int = 1000
    null_permutations: int = 1000
    confidence: float = 0.95
    seed: int = 1
    minimum_episodes_per_policy: int = 5
    normal_reporting_episodes: int = 10

    def __post_init__(self) -> None:
        if any(value < 1 for value in self.horizons_population_rounds):
            raise ValueError("Analysis horizons must be positive.")
        if self.bootstrap_resamples < 0 or self.null_permutations < 0:
            raise ValueError("Resample counts cannot be negative.")
        if not 0 < self.confidence < 1:
            raise ValueError("confidence must lie in (0, 1).")
        if self.minimum_episodes_per_policy < 1:
            raise ValueError("minimum_episodes_per_policy must be positive.")
        if self.normal_reporting_episodes < self.minimum_episodes_per_policy:
            raise ValueError(
                "normal_reporting_episodes cannot be below the estimation minimum."
            )


def _estimate_fields(estimate: Estimate) -> dict[str, Any]:
    return asdict(estimate)


def _estimation_status(
    group: pd.DataFrame, config: AnalysisConfig
) -> tuple[str, str | None]:
    counts = group.groupby("committee_policy", dropna=False)["episode_id"].nunique()
    if len(counts) < 2:
        return "non_estimable", "only one committee policy is present"
    minimum = int(counts.min())
    if minimum < config.minimum_episodes_per_policy:
        return (
            "non_estimable",
            f"at least one policy has fewer than {config.minimum_episodes_per_policy} completed episodes",
        )
    if minimum < config.normal_reporting_episodes:
        return "exploratory", "exploratory and highly noisy: 5-9 episodes per policy"
    return "estimable", None


def _unavailable_estimate(observations: int) -> Estimate:
    return Estimate(math.nan, math.nan, math.nan, observations)


def _sum_episode_counts(
    counts: dict[Any, np.ndarray], ids: np.ndarray, shape: tuple[int, ...]
) -> np.ndarray:
    total = np.zeros(shape, dtype=float)
    for episode_id in ids:
        total += counts[episode_id]
    return total


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
        estimate_status, status_reason = _estimation_status(group, config)
        policy_levels = tuple(sorted(group["committee_policy"].dropna().unique()))
        outcome_levels = ("A", "B") if resolved_only else ("A", "B", "unresolved")
        policy_index = {value: index for index, value in enumerate(policy_levels)}
        outcome_index = {value: index for index, value in enumerate(outcome_levels)}
        shape = (len(policy_levels), len(outcome_levels))
        episode_counts: dict[Any, np.ndarray] = {}
        for record in group[["episode_id", "committee_policy", "final_convention"]].to_dict("records"):
            counts = np.zeros(shape, dtype=float)
            policy = record["committee_policy"]
            outcome = record["final_convention"]
            if policy in policy_index and outcome in outcome_index:
                counts[policy_index[policy], outcome_index[outcome]] += 1
            episode_counts[record["episode_id"]] = counts
        episode_ids = group["episode_id"].to_numpy()
        complete_counts = _sum_episode_counts(episode_counts, episode_ids, shape)

        estimate = (
            mutual_information_from_counts(complete_counts)
            if estimate_status != "non_estimable"
            else _unavailable_estimate(len(group))
        )

        def sampled(sampled_ids: np.ndarray) -> float:
            return mutual_information_from_counts(
                _sum_episode_counts(episode_counts, sampled_ids, shape)
            ).jeffreys

        low, high = (
            _bootstrap_interval(
                episode_ids, sampled,
                resamples=config.bootstrap_resamples, confidence=config.confidence, rng=rng,
            )
            if estimate_status != "non_estimable"
            else (math.nan, math.nan)
        )
        row = dict(zip(STRATA_COLUMNS, keys, strict=True))
        row.update(
            statistic="terminal_resolved_only" if resolved_only else "terminal_all",
            horizon_population_rounds=None,
            ci_low=low,
            ci_high=high,
            estimate_status=estimate_status,
            status_reason=status_reason,
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
        estimate_status, status_reason = _estimation_status(group, config)
        n_agents = int(group["N"].iloc[0])
        policy_levels = tuple(sorted(group["committee_policy"].dropna().unique()))
        episode_ids = group["episode_id"].drop_duplicates().to_numpy()
        episode_frames = {episode_id: frame for episode_id, frame in group.groupby("episode_id", sort=False)}
        for horizon in config.horizons_population_rounds:
            lag = horizon * n_agents

            episode_pairs = {
                episode_id: _lag_pairs(frame, lag, state_column)
                for episode_id, frame in episode_frames.items()
            }
            policy_index = {value: index for index, value in enumerate(policy_levels)}
            state_index = {value: index for index, value in enumerate(state_levels)}
            shape = (len(policy_levels), len(state_levels), len(state_levels))
            episode_counts: dict[Any, np.ndarray] = {}
            for episode_id, pairs in episode_pairs.items():
                counts = np.zeros(shape, dtype=float)
                for record in pairs[["committee_policy", state_column, "future"]].to_dict("records"):
                    policy = record["committee_policy"]
                    current = record[state_column]
                    future = record["future"]
                    if policy in policy_index and current in state_index and future in state_index:
                        counts[
                            policy_index[policy], state_index[current], state_index[future]
                        ] += 1
                episode_counts[episode_id] = counts
            complete_counts = _sum_episode_counts(episode_counts, episode_ids, shape)
            estimate = (
                conditional_mutual_information_from_counts(complete_counts)
                if estimate_status != "non_estimable"
                else _unavailable_estimate(int(complete_counts.sum()))
            )
            row_status = estimate_status
            row_reason = status_reason
            if estimate_status != "non_estimable" and not math.isfinite(estimate.jeffreys):
                row_status = "non_estimable"
                row_reason = "no usable lagged state pairs are present"

            def sampled(sampled_ids: np.ndarray) -> float:
                return conditional_mutual_information_from_counts(
                    _sum_episode_counts(episode_counts, sampled_ids, shape)
                ).jeffreys

            low, high = (
                _bootstrap_interval(
                    episode_ids, sampled, resamples=config.bootstrap_resamples,
                    confidence=config.confidence, rng=rng,
                )
                if row_status != "non_estimable"
                else (math.nan, math.nan)
            )
            row = dict(zip(STRATA_COLUMNS, keys, strict=True))
            row.update(
                statistic=statistic, horizon_population_rounds=horizon,
                ci_low=low, ci_high=high,
                estimate_status=row_status, status_reason=row_reason,
                **_estimate_fields(estimate),
            )
            rows.append(row)
    return pd.DataFrame(rows)


def estimate_nulls(
    interactions: pd.DataFrame, episodes: pd.DataFrame, config: AnalysisConfig
) -> pd.DataFrame:
    if config.null_permutations == 0:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    compact_columns = list(
        dict.fromkeys(
            STRATA_COLUMNS
            + [
                "episode_id",
                "committee_policy",
                "interaction_index",
                "macrostate_binary",
                "macrostate_three",
            ]
        )
    )
    compact_interactions = interactions[compact_columns].copy()
    shuffle_strata = [
        "regime", "committee_size", "pulse_rounds", "provider", "model",
        "prompt_version", "initial_condition",
    ]
    for permutation in range(config.null_permutations):
        seed = config.seed + 10_000 + permutation
        shuffled = shuffle_episode_labels(episodes, strata=shuffle_strata, seed=seed)
        mapping = shuffled.set_index("episode_id")["committee_policy"]
        shuffled_interactions = compact_interactions.copy()
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
        shifted = circular_shift_trajectories(compact_interactions, seed=seed)
        shifted_lagged = estimate_lagged(shifted, null_config)
        for record in shifted_lagged.to_dict("records"):
            rows.append({**record, "null_type": "circular_shift_null", "permutation": permutation})
    return pd.DataFrame(rows)


def analyze_histories(
    history_dir: str | Path,
    output_dir: str | Path,
    config: AnalysisConfig | None = None,
    column_mapping: dict[str, str] | None = None,
) -> dict[str, str | int | None]:
    settings = config or AnalysisConfig()
    source = Path(history_dir)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    interactions = pd.read_parquet(source / "interactions.parquet")
    episodes = pd.read_parquet(source / "episodes.parquet")
    if column_mapping:
        interactions = interactions.rename(columns=column_mapping)
        episodes = episodes.rename(columns=column_mapping)
    interactions, episodes, normalization_warnings = normalize_histories(
        interactions, episodes
    )
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
    metrics = summarize_episode_metrics(
        episodes,
        confidence=settings.confidence,
        bootstrap_resamples=settings.bootstrap_resamples,
        seed=settings.seed,
    )
    estimates = add_efficiency(metrics, estimates)
    nulls = estimate_nulls(interactions, episodes, settings)
    estimates = attach_null_summary(estimates, nulls)
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
    warnings = normalization_warnings + collect_warnings(episodes, estimates, nulls)
    write_summary_markdown(
        episodes, metrics, estimates, warnings, destination / "summary.md"
    )
    plots_dir = destination / "plots"
    make_experiment_summary(
        interactions,
        metrics,
        estimates,
        plots_dir / "experiment_summary.png",
        bootstrap_resamples=settings.bootstrap_resamples,
        confidence=settings.confidence,
        seed=settings.seed,
    )
    pulse_plot = make_pulse_summary(
        interactions,
        metrics,
        plots_dir / "pulse_summary.png",
        bootstrap_resamples=settings.bootstrap_resamples,
        confidence=settings.confidence,
        seed=settings.seed,
    )
    (destination / "analysis_config.json").write_text(json.dumps(asdict(settings), indent=2), encoding="utf-8")
    return {
        "estimates": len(estimates), "metric_rows": len(metrics), "null_rows": len(nulls),
        "output_dir": str(destination),
        "summary": str(destination / "summary.md"),
        "experiment_summary_plot": str(plots_dir / "experiment_summary.png"),
        "pulse_summary_plot": str(plots_dir / "pulse_summary.png") if pulse_plot else None,
        "warnings": len(warnings),
    }


__all__ = ["AnalysisConfig", "analyze_histories", "estimate_lagged", "estimate_nulls", "estimate_terminal"]
