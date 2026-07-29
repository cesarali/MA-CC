"""Episode-preserving surrogate transformations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def shuffle_episode_labels(
    episodes: pd.DataFrame,
    *,
    strata: list[str],
    label: str = "committee_policy",
    seed: int = 1,
) -> pd.DataFrame:
    result = episodes.copy()
    rng = np.random.default_rng(seed)
    for _, indexes in result.groupby(strata, dropna=False).groups.items():
        values = result.loc[indexes, label].to_numpy(copy=True)
        rng.shuffle(values)
        result.loc[indexes, label] = values
    return result


def circular_shift_trajectories(
    interactions: pd.DataFrame,
    *,
    state_column: str = "macrostate_binary",
    seed: int = 1,
) -> pd.DataFrame:
    result = interactions.copy()
    rng = np.random.default_rng(seed)
    for _, indexes in result.groupby("episode_id", sort=False).groups.items():
        ordered = list(indexes)
        if len(ordered) < 2:
            continue
        offset = int(rng.integers(1, len(ordered)))
        result.loc[ordered, state_column] = np.roll(
            result.loc[ordered, state_column].to_numpy(), offset
        )
    return result


def swap_labels_half(
    interactions: pd.DataFrame, episodes: pd.DataFrame, *, seed: int = 1
) -> tuple[pd.DataFrame, pd.DataFrame]:
    trajectory = interactions.copy()
    summaries = episodes.copy()
    rng = np.random.default_rng(seed)
    selected: set[object] = set()
    balance_columns = [
        column for column in ("regime", "committee_size", "committee_policy")
        if column in summaries
    ]
    grouped = (
        summaries.groupby(balance_columns, dropna=False, sort=False)
        if balance_columns
        else [(None, summaries)]
    )
    for _, group in grouped:
        ids = group["episode_id"].drop_duplicates().to_numpy()
        if len(ids) >= 2:
            selected.update(rng.choice(ids, size=len(ids) // 2, replace=False))

    def swap_scalar(value: object) -> object:
        mapping = {
            "A": "B", "B": "A", "resolved_A": "resolved_B", "resolved_B": "resolved_A",
            "always_A": "always_B", "always_B": "always_A", "consensus_A": "consensus_B",
            "consensus_B": "consensus_A", "A_dominant": "B_dominant", "B_dominant": "A_dominant",
        }
        return mapping.get(value, value)

    trajectory_mask = trajectory["episode_id"].isin(selected)
    summary_mask = summaries["episode_id"].isin(selected)
    for frame, mask in ((trajectory, trajectory_mask), (summaries, summary_mask)):
        for column in ("output_i", "output_j", "resolved_state", "terminal_outcome", "final_convention", "incumbent", "alternative", "initial_condition", "committee_policy", "macrostate_three"):
            if column in frame:
                frame.loc[mask, column] = frame.loc[mask, column].map(swap_scalar)
        for column in ("rolling_share_A", "terminal_share_A"):
            if column in frame:
                frame.loc[mask, column] = 1.0 - frame.loc[mask, column].astype(float)
        if "macrostate_binary" in frame:
            values = frame.loc[mask, "macrostate_binary"]
            frame.loc[mask, "macrostate_binary"] = values.map(lambda value: 1 - value if pd.notna(value) else value)
    return trajectory, summaries


__all__ = ["circular_shift_trajectories", "shuffle_episode_labels", "swap_labels_half"]
