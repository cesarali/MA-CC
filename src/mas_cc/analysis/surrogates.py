"""Episode-preserving surrogate transformations (the MI null models).

Same algorithmic shapes as `src/naming_game/analysis/surrogates.py`, adapted
to this module's own tidy schema (`reader.py`) rather than the legacy's
bespoke Parquet column names - see `docs/legacy_empowerment_implementation.md`
section 8 for the originals these mirror.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def shuffle_condition_labels(
    episodes: pd.DataFrame,
    condition_column: str,
    *,
    strata: list[str] | None = None,
    seed: int = 1,
) -> pd.DataFrame:
    """Permute `condition_column` across episodes within `strata` groups.

    With `strata=None`, permutes across every episode in the frame. This is
    the label-permutation null: what MI would appear if the condition had no
    real effect but episode outcomes were otherwise unchanged.
    """

    result = episodes.copy()
    rng = np.random.default_rng(seed)
    groups = (
        result.groupby(strata, dropna=False).groups if strata else {(): result.index}
    )
    for _, indexes in groups.items():
        values = result.loc[indexes, condition_column].to_numpy(copy=True)
        rng.shuffle(values)
        result.loc[indexes, condition_column] = values
    return result


def circular_shift_macrostate(
    rounds: pd.DataFrame,
    *,
    episode_column: str = "episode_id",
    state_column: str = "macrostate",
    seed: int = 1,
) -> pd.DataFrame:
    """Roll each episode's macrostate sequence by a random offset.

    Destroys the true condition-to-timing alignment while preserving each
    episode's own autocorrelation structure - a temporal null for lagged
    conditional-MI estimates.
    """

    result = rounds.copy()
    rng = np.random.default_rng(seed)
    for _, indexes in result.groupby(episode_column, sort=False).groups.items():
        ordered = list(indexes)
        if len(ordered) < 2:
            continue
        offset = int(rng.integers(1, len(ordered)))
        result.loc[ordered, state_column] = np.roll(
            result.loc[ordered, state_column].to_numpy(), offset
        )
    return result


def swap_condition_and_outcome_labels(
    episodes: pd.DataFrame,
    condition_column: str,
    outcome_column: str,
    *,
    seed: int = 1,
) -> pd.DataFrame:
    """Relabel both columns' two values for a balanced random half of episodes.

    Only meaningful when both columns are binary: a real MI estimate must
    come out (nearly) identical before and after, since the two labels are
    arbitrary. Raises `ValueError` if either column does not have exactly
    two distinct values.
    """

    condition_levels = sorted(episodes[condition_column].dropna().unique())
    outcome_levels = sorted(episodes[outcome_column].dropna().unique())
    if len(condition_levels) != 2 or len(outcome_levels) != 2:
        raise ValueError(
            "swap_condition_and_outcome_labels requires exactly two levels in "
            f"both {condition_column!r} and {outcome_column!r}"
        )
    condition_swap = dict(zip(condition_levels, reversed(condition_levels)))
    outcome_swap = dict(zip(outcome_levels, reversed(outcome_levels)))

    result = episodes.copy()
    rng = np.random.default_rng(seed)
    ids = result.index.to_numpy()
    selected = rng.choice(ids, size=len(ids) // 2, replace=False)
    mask = result.index.isin(selected)
    result.loc[mask, condition_column] = result.loc[mask, condition_column].map(condition_swap)
    result.loc[mask, outcome_column] = result.loc[mask, outcome_column].map(outcome_swap)
    return result


__all__ = [
    "circular_shift_macrostate",
    "shuffle_condition_labels",
    "swap_condition_and_outcome_labels",
]
