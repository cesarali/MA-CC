"""End-to-end empowerment analysis over a completed grid directory.

Same estimation shape as the legacy `naming_game/analysis/empowerment.py`
(episode-level contingency tables, bootstrap resampled over `episode_id`,
permutation/circular-shift nulls), rewritten against this module's own tidy
schema (`reader.GridData`) instead of the legacy's bespoke Parquet columns.
"""

from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .estimators import (
    Estimate,
    conditional_mutual_information_from_counts,
    mutual_information_from_counts,
)
from .reader import GridData, read_grid
from .surrogates import circular_shift_macrostate, shuffle_condition_labels, swap_condition_and_outcome_labels

_UNAVAILABLE = Estimate(math.nan, math.nan, math.nan, 0)


def _bootstrap_interval(
    episode_ids: np.ndarray,
    estimator: Callable[[np.ndarray], float],
    *,
    resamples: int,
    confidence: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    if resamples == 0 or len(episode_ids) == 0:
        return math.nan, math.nan
    values = np.asarray(
        [estimator(rng.choice(episode_ids, size=len(episode_ids), replace=True)) for _ in range(resamples)],
        dtype=float,
    )
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return math.nan, math.nan
    alpha = (1 - confidence) / 2
    return float(np.quantile(values, alpha)), float(np.quantile(values, 1 - alpha))


def estimate_terminal_mi(
    episodes: pd.DataFrame,
    condition_column: str,
    *,
    outcome_column: str = "terminal_outcome",
    bootstrap_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 1,
) -> dict[str, Any]:
    """I(condition; terminal outcome), one contingency-table cell per episode."""

    if episodes.empty:
        return {
            "statistic": "terminal", "condition_column": condition_column,
            "outcome_column": outcome_column, "horizon": None,
            "ci_low": math.nan, "ci_high": math.nan, "episodes": 0, **asdict(_UNAVAILABLE),
        }
    condition_levels = tuple(sorted(episodes[condition_column].dropna().unique()))
    outcome_levels = tuple(sorted(episodes[outcome_column].dropna().unique()))
    condition_index = {value: index for index, value in enumerate(condition_levels)}
    outcome_index = {value: index for index, value in enumerate(outcome_levels)}
    shape = (len(condition_levels), len(outcome_levels))

    episode_counts: dict[str, np.ndarray] = {}
    for record in episodes[["episode_id", condition_column, outcome_column]].to_dict("records"):
        counts = np.zeros(shape, dtype=float)
        condition, outcome = record[condition_column], record[outcome_column]
        if condition in condition_index and outcome in outcome_index:
            counts[condition_index[condition], outcome_index[outcome]] = 1.0
        episode_counts[record["episode_id"]] = counts
    episode_ids = np.array(list(episode_counts))

    def totals(ids: np.ndarray) -> np.ndarray:
        total = np.zeros(shape, dtype=float)
        for episode_id in ids:
            total += episode_counts[episode_id]
        return total

    estimate = mutual_information_from_counts(totals(episode_ids))
    rng = np.random.default_rng(seed)
    low, high = _bootstrap_interval(
        episode_ids,
        lambda ids: mutual_information_from_counts(totals(ids)).jeffreys,
        resamples=bootstrap_resamples, confidence=confidence, rng=rng,
    )
    return {
        "statistic": "terminal", "condition_column": condition_column,
        "outcome_column": outcome_column, "horizon": None,
        "ci_low": low, "ci_high": high, "episodes": len(episode_ids), **asdict(estimate),
    }


def estimate_lagged_mi(
    rounds: pd.DataFrame,
    condition_column: str,
    horizon: int,
    *,
    state_column: str = "macrostate",
    bootstrap_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 1,
) -> dict[str, Any]:
    """I(condition; future macrostate | current macrostate) at one horizon (in rounds)."""

    if rounds.empty:
        return {
            "statistic": "lagged", "condition_column": condition_column,
            "outcome_column": state_column, "horizon": horizon,
            "ci_low": math.nan, "ci_high": math.nan, "episodes": 0, **asdict(_UNAVAILABLE),
        }
    parts: list[pd.DataFrame] = []
    for episode_id, group in rounds.groupby("episode_id", sort=False):
        ordered = group.sort_values("round_index")
        pair = ordered[["episode_id", condition_column, state_column]].copy()
        pair["future"] = pair[state_column].shift(-horizon)
        parts.append(pair.dropna(subset=[state_column, "future"]))
    pairs = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    condition_levels = tuple(sorted(rounds[condition_column].dropna().unique()))
    state_levels = tuple(sorted(rounds[state_column].dropna().unique()))
    condition_index = {value: index for index, value in enumerate(condition_levels)}
    state_index = {value: index for index, value in enumerate(state_levels)}
    shape = (len(condition_levels), len(state_levels), len(state_levels))

    episode_counts: dict[str, np.ndarray] = {}
    if not pairs.empty:
        for episode_id, group in pairs.groupby("episode_id", sort=False):
            counts = np.zeros(shape, dtype=float)
            for record in group[[condition_column, state_column, "future"]].to_dict("records"):
                condition, current, future = record[condition_column], record[state_column], record["future"]
                if condition in condition_index and current in state_index and future in state_index:
                    counts[condition_index[condition], state_index[current], state_index[future]] += 1
            episode_counts[episode_id] = counts
    episode_ids = np.array(list(episode_counts))

    def totals(ids: np.ndarray) -> np.ndarray:
        total = np.zeros(shape, dtype=float)
        for episode_id in ids:
            total += episode_counts[episode_id]
        return total

    estimate = (
        conditional_mutual_information_from_counts(totals(episode_ids))
        if len(episode_ids) else _UNAVAILABLE
    )
    rng = np.random.default_rng(seed + 100)
    low, high = _bootstrap_interval(
        episode_ids,
        lambda ids: conditional_mutual_information_from_counts(totals(ids)).jeffreys,
        resamples=bootstrap_resamples, confidence=confidence, rng=rng,
    )
    return {
        "statistic": "lagged", "condition_column": condition_column,
        "outcome_column": state_column, "horizon": horizon,
        "ci_low": low, "ci_high": high, "episodes": len(episode_ids), **asdict(estimate),
    }


def estimate_nulls(
    data: GridData,
    condition_column: str,
    horizons: tuple[int, ...],
    *,
    permutations: int,
    seed: int = 1,
) -> pd.DataFrame:
    """Label-shuffle null for the terminal statistic, circular-shift null for lagged ones."""

    if permutations == 0:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for permutation in range(permutations):
        draw_seed = seed + 10_000 + permutation
        shuffled_episodes = shuffle_condition_labels(data.episodes, condition_column, seed=draw_seed)
        terminal = estimate_terminal_mi(
            shuffled_episodes, condition_column, bootstrap_resamples=0, seed=draw_seed,
        )
        rows.append({**terminal, "null_type": "condition_label_shuffle", "permutation": permutation})

        shifted_rounds = circular_shift_macrostate(data.rounds, seed=draw_seed)
        for horizon in horizons:
            lagged = estimate_lagged_mi(
                shifted_rounds, condition_column, horizon, bootstrap_resamples=0, seed=draw_seed,
            )
            rows.append({**lagged, "null_type": "circular_shift", "permutation": permutation})
    return pd.DataFrame(rows)


def label_swap_invariance(
    episodes: pd.DataFrame,
    condition_column: str,
    outcome_column: str = "terminal_outcome",
    *,
    seed: int = 1,
) -> pd.DataFrame | None:
    """A/B relabeling invariance check - only meaningful for binary condition/outcome."""

    try:
        swapped = swap_condition_and_outcome_labels(episodes, condition_column, outcome_column, seed=seed)
    except ValueError:
        return None
    original = estimate_terminal_mi(episodes, condition_column, outcome_column=outcome_column, bootstrap_resamples=0)
    after = estimate_terminal_mi(swapped, condition_column, outcome_column=outcome_column, bootstrap_resamples=0)
    difference = abs(original["jeffreys"] - after["jeffreys"])
    return pd.DataFrame([{
        "condition_column": condition_column, "outcome_column": outcome_column,
        "jeffreys_original": original["jeffreys"], "jeffreys_swapped": after["jeffreys"],
        "absolute_difference": difference, "invariant_within_tolerance": difference <= 1e-9,
    }])


def analyze_grid(
    grid_dir: str | Path,
    output_dir: str | Path,
    *,
    condition_column: str | None = None,
    horizons: tuple[int, ...] = (1,),
    bootstrap_resamples: int = 1000,
    null_permutations: int = 1000,
    confidence: float = 0.95,
    seed: int = 1,
) -> dict[str, Any]:
    """Read a grid directory, estimate MI, and write CSV artifacts under `output_dir`."""

    data = read_grid(grid_dir)
    if condition_column is None:
        if len(data.condition_columns) != 1:
            raise ValueError(
                "condition_column must be given explicitly when the grid swept more (or "
                f"fewer) than one axis; swept axes were: {data.condition_columns}"
            )
        condition_column = data.condition_columns[0]
    elif condition_column not in data.condition_columns:
        raise ValueError(
            f"{condition_column!r} was not a swept grid axis; available: {data.condition_columns}"
        )

    terminal = estimate_terminal_mi(
        data.episodes, condition_column,
        bootstrap_resamples=bootstrap_resamples, confidence=confidence, seed=seed,
    )
    lagged = [
        estimate_lagged_mi(
            data.rounds, condition_column, horizon,
            bootstrap_resamples=bootstrap_resamples, confidence=confidence, seed=seed,
        )
        for horizon in horizons
    ]
    estimates = pd.DataFrame([terminal, *lagged])
    nulls = estimate_nulls(data, condition_column, horizons, permutations=null_permutations, seed=seed)
    invariance = label_swap_invariance(data.episodes, condition_column, seed=seed)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    estimates.to_csv(destination / "mi_estimates.csv", index=False)
    nulls.to_csv(destination / "null_results.csv", index=False)
    if invariance is not None:
        invariance.to_csv(destination / "label_swap_invariance.csv", index=False)

    return {
        "condition_column": condition_column,
        "episodes": len(data.episodes),
        "terminal_mi_jeffreys": terminal["jeffreys"],
        "lagged_mi_jeffreys": {str(row["horizon"]): row["jeffreys"] for row in lagged},
        "null_rows": len(nulls),
        "label_swap_invariance_checked": invariance is not None,
        "output_dir": str(destination),
    }


__all__ = [
    "analyze_grid",
    "estimate_lagged_mi",
    "estimate_nulls",
    "estimate_terminal_mi",
    "label_swap_invariance",
]
