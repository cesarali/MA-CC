import math

import pandas as pd
import pytest

from naming_game.analysis.estimators import (
    conditional_mutual_information,
    mutual_information,
)
from naming_game.analysis.surrogates import (
    circular_shift_trajectories,
    shuffle_episode_labels,
    swap_labels_half,
)


def test_direct_counting_known_channels():
    perfect = mutual_information([0, 0, 1, 1], [0, 0, 1, 1], x_levels=(0, 1), y_levels=(0, 1))
    independent = mutual_information(
        [0, 0, 1, 1], [0, 1, 0, 1], x_levels=(0, 1), y_levels=(0, 1)
    )
    constant = mutual_information([0, 0, 0], [0, 1, 1], x_levels=(0, 1), y_levels=(0, 1))
    assert perfect.unsmoothed == pytest.approx(1.0)
    assert independent.unsmoothed == pytest.approx(0.0)
    assert constant.unsmoothed == pytest.approx(0.0)
    assert 0 < perfect.jeffreys < perfect.unsmoothed
    assert math.isfinite(constant.jeffreys)


def test_conditional_channel_and_miller_madow_are_reported():
    estimate = conditional_mutual_information(
        [0, 0, 1, 1] * 2,
        [0, 0, 1, 1] * 2,
        [0] * 4 + [1] * 4,
        x_levels=(0, 1),
        y_levels=(0, 1),
        z_levels=(0, 1),
    )
    assert estimate.unsmoothed == pytest.approx(1.0)
    assert math.isfinite(estimate.miller_madow)
    assert estimate.observations == 8


def test_surrogates_preserve_episode_units_and_are_deterministic():
    episodes = pd.DataFrame(
        {
            "episode_id": ["a", "b", "c", "d"],
            "regime": ["pulse"] * 4,
            "committee_size": [1] * 4,
            "committee_policy": ["on", "off", "on", "off"],
        }
    )
    first = shuffle_episode_labels(
        episodes, strata=["regime", "committee_size"], seed=7
    )
    second = shuffle_episode_labels(
        episodes, strata=["regime", "committee_size"], seed=7
    )
    assert first.equals(second)
    assert sorted(first.committee_policy) == sorted(episodes.committee_policy)

    trajectory = pd.DataFrame(
        {"episode_id": ["a"] * 4, "macrostate_binary": [0, 0, 1, 1]}
    )
    shifted = circular_shift_trajectories(trajectory, seed=2)
    assert sorted(shifted.macrostate_binary) == [0, 0, 1, 1]
    assert not shifted.macrostate_binary.equals(trajectory.macrostate_binary)


def test_half_episode_label_swap_preserves_a_symmetric_channel():
    episodes = pd.DataFrame(
        {
            "episode_id": [f"e{index}" for index in range(8)],
            "committee_policy": ["always_A"] * 4 + ["always_B"] * 4,
            "final_convention": ["A"] * 4 + ["B"] * 4,
            "terminal_share_A": [1.0] * 4 + [0.0] * 4,
        }
    )
    interactions = pd.DataFrame(
        {
            "episode_id": episodes.episode_id,
            "output_i": episodes.final_convention,
            "output_j": episodes.final_convention,
            "terminal_outcome": episodes.final_convention,
            "rolling_share_A": episodes.terminal_share_A,
            "macrostate_binary": [1] * 4 + [0] * 4,
        }
    )
    _, swapped = swap_labels_half(interactions, episodes, seed=4)
    before = mutual_information(
        episodes.committee_policy.tolist(), episodes.final_convention.tolist()
    )
    after = mutual_information(
        swapped.committee_policy.tolist(), swapped.final_convention.tolist()
    )
    assert after.jeffreys == pytest.approx(before.jeffreys)
