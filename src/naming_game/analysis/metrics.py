"""Complementary episode-level metrics for empowerment experiments."""

from __future__ import annotations

import math

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
]


def summarize_episode_metrics(episodes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in episodes.groupby(GROUP_COLUMNS, dropna=False, sort=True):
        row = dict(zip(GROUP_COLUMNS, keys, strict=True))
        terminal = group["final_convention"]
        actions = float(group["total_committee_actions"].mean())
        row.update(
            episodes=len(group),
            takeover_probability=float(group["takeover"].mean()),
            terminal_probability_A=float((terminal == "A").mean()),
            terminal_probability_B=float((terminal == "B").mean()),
            terminal_probability_unresolved=float((terminal == "unresolved").mean()),
            permanent_flip_probability=float(group["permanent_flip"].mean()),
            consensus_probability=float(group["stopping_interaction"].notna().mean()),
            mean_consensus_time_interactions=float(group["stopping_interaction"].mean()),
            mean_peak_displacement=float(group["peak_displacement"].mean()),
            mean_time_to_peak_interactions=float(group["time_to_peak_interactions"].mean()),
            mean_recovery_time_interactions=float(group["recovery_time_interactions"].mean()),
            recovery_censoring_probability=float(group["recovery_censored"].mean()),
            mean_committee_actions=actions,
            mean_terminal_share_A=float(group["terminal_share_A"].mean()),
            mean_post_consensus_persistence=float(group["post_consensus_persistence"].mean()),
        )
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
