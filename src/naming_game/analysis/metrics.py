"""Complementary episode-level metrics for empowerment experiments."""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pandas as pd

GROUP_COLUMNS = [
    "regime",
    "N",
    "committee_size",
    "initial_condition",
    "provider",
    "model",
    "prompt_version",
    "pulse_rounds",
    "committee_policy",
    "attack_direction",
]


def _wilson_interval(successes: int, total: int, confidence: float) -> tuple[float, float]:
    if total <= 0:
        return math.nan, math.nan
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    probability = successes / total
    denominator = 1 + z * z / total
    center = (probability + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            probability * (1 - probability) / total + z * z / (4 * total * total)
        )
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def _probability_fields(
    prefix: str, values: pd.Series, confidence: float
) -> dict[str, float]:
    clean = values.dropna().astype(bool)
    if clean.empty:
        return {
            f"{prefix}_probability": math.nan,
            f"{prefix}_ci_low": math.nan,
            f"{prefix}_ci_high": math.nan,
        }
    successes = int(clean.sum())
    low, high = _wilson_interval(successes, len(clean), confidence)
    return {
        f"{prefix}_probability": successes / len(clean),
        f"{prefix}_ci_low": low,
        f"{prefix}_ci_high": high,
    }


def _bootstrap_median(
    values: pd.Series,
    *,
    resamples: int,
    confidence: float,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    clean = values.dropna().astype(float).to_numpy()
    if len(clean) == 0:
        return math.nan, math.nan, math.nan
    median = float(np.median(clean))
    if resamples == 0:
        return median, math.nan, math.nan
    samples = np.asarray(
        [float(np.median(rng.choice(clean, size=len(clean), replace=True))) for _ in range(resamples)]
    )
    alpha = (1 - confidence) / 2
    return median, float(np.quantile(samples, alpha)), float(np.quantile(samples, 1 - alpha))


def summarize_episode_metrics(
    episodes: pd.DataFrame,
    *,
    confidence: float = 0.95,
    bootstrap_resamples: int = 1000,
    seed: int = 1,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(seed + 500)
    for keys, group in episodes.groupby(GROUP_COLUMNS, dropna=False, sort=True):
        row = dict(zip(GROUP_COLUMNS, keys, strict=True))
        terminal = group["final_convention"]
        actions = float(group["total_committee_actions"].mean())
        terminal_takeover = group["terminal_takeover"].astype(bool)
        ever_crossed = group["ever_crossed"].astype(bool)
        incumbent_survives = group["incumbent_survives"].astype(bool)
        unresolved = terminal == "unresolved"
        consensus = group["stopping_interaction"].notna()
        permanent_flip = group["permanent_flip"].astype(bool)
        is_pulse = str(row["regime"]) == "pulse"
        recovery = (~group["recovery_censored"].astype(bool)) if is_pulse else pd.Series(dtype=bool)
        median_recovery, recovery_low, recovery_high = _bootstrap_median(
            group["recovery_time_population_rounds"] if is_pulse else pd.Series(dtype=float),
            resamples=bootstrap_resamples,
            confidence=confidence,
            rng=rng,
        )
        row.update(
            episodes=len(group),
            takeover_probability=float(ever_crossed.mean()),
            terminal_probability_A=float((terminal == "A").mean()),
            terminal_probability_B=float((terminal == "B").mean()),
            terminal_probability_unresolved=float((terminal == "unresolved").mean()),
            permanent_flip_probability=float(permanent_flip.mean()),
            consensus_probability=float(consensus.mean()),
            mean_consensus_time_interactions=float(group["stopping_interaction"].mean()),
            mean_consensus_time_population_rounds=float(group["stopping_population_round"].mean()),
            mean_peak_displacement=float(group["peak_displacement"].mean()),
            mean_time_to_peak_interactions=float(group["time_to_peak_interactions"].mean()),
            mean_recovery_time_interactions=float(group["recovery_time_interactions"].mean()),
            median_recovery_time_population_rounds=median_recovery,
            median_recovery_time_population_rounds_ci_low=recovery_low,
            median_recovery_time_population_rounds_ci_high=recovery_high,
            recovery_censoring_probability=float(group["recovery_censored"].mean()),
            mean_committee_actions=actions,
            mean_terminal_share_A=float(group["terminal_share_A"].mean()),
            mean_post_consensus_persistence=float(group["post_consensus_persistence"].mean()),
        )
        row.update(_probability_fields("terminal_takeover", terminal_takeover, confidence))
        row.update(_probability_fields("ever_crossed", ever_crossed, confidence))
        row.update(_probability_fields("incumbent_survival", incumbent_survives, confidence))
        row.update(_probability_fields("unresolved", unresolved, confidence))
        row.update(_probability_fields("terminal_A", terminal == "A", confidence))
        row.update(_probability_fields("terminal_B", terminal == "B", confidence))
        row.update(_probability_fields("consensus", consensus, confidence))
        row.update(_probability_fields("permanent_flip", permanent_flip, confidence))
        row.update(_probability_fields("recovery", recovery, confidence))
        rows.append(row)
    return pd.DataFrame(rows)


def add_efficiency(metrics: pd.DataFrame, terminal: pd.DataFrame) -> pd.DataFrame:
    if terminal.empty:
        return terminal.copy()
    action_groups = [column for column in GROUP_COLUMNS if column != "committee_policy"]
    weighted = metrics.assign(
        weighted_actions=metrics["mean_committee_actions"] * metrics["episodes"]
    )
    actions = weighted.groupby(action_groups, dropna=False).agg(
        weighted_actions=("weighted_actions", "sum"),
        channel_episodes=("episodes", "sum"),
    ).reset_index()
    actions["expected_committee_actions"] = (
        actions["weighted_actions"] / actions["channel_episodes"]
    )
    actions = actions.drop(columns=["weighted_actions", "channel_episodes"])
    result = terminal.merge(actions, on=action_groups, how="left")
    result["efficiency"] = result.apply(
        lambda row: row["jeffreys"] / row["expected_committee_actions"]
        if row["expected_committee_actions"] > 0
        else math.nan,
        axis=1,
    )
    return result


__all__ = ["GROUP_COLUMNS", "add_efficiency", "summarize_episode_metrics"]
