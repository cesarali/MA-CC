"""Checkpoint family: pooled branch metrics and gathered parent bootstraps reproduce the serial, per-parent code."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd

from mas_cc.analysis import checkpoint_ensemble as ce

_spec = importlib.util.spec_from_file_location(
    "cp_fixtures", Path(__file__).with_name("test_blackboard_checkpoint_phase2.py"))
fixtures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fixtures)


def _reference_parent_bootstrap(values, *, resamples, confidence, seed):
    """The pre-pool implementation, kept verbatim as the oracle."""
    if values.empty or resamples <= 0:
        return math.nan, math.nan
    parents = values["parent_id"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(resamples):
        sampled = rng.choice(parents, size=len(parents), replace=True)
        draws.append(float(np.mean([
            values.loc[values["parent_id"] == parent, "paired_difference"].mean()
            for parent in sampled
        ])))
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(draws, alpha)), float(np.quantile(draws, 1 - alpha))


def _reference_activation_draws(frame, parents, *, resamples, seed):
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(resamples):
        sampled = rng.choice(parents, size=len(parents), replace=True)
        values = [value for parent in sampled
                  for value in frame.loc[frame["parent_id"] == parent, "contribution"].tolist()]
        draws.append(float(np.mean(values)))
    return draws


def _sensing_rounds():
    rounds = fixtures._rounds(parents=5, horizons=3)
    rng = np.random.default_rng(4)
    sensing = rounds["branch_policy"].isin(("sensing_truth", "sensing_false"))
    rounds.loc[sensing, "controller_advocate_probability"] = rng.uniform(0.2, 0.8, size=int(sensing.sum()))
    rounds.loc[sensing, "controller_sampled_U"] = rng.integers(0, 2, size=int(sensing.sum()))
    # The relational round adapter needs these two on continuation rows.
    rounds["round_index"] = rounds["absolute_round"]
    rounds["correct_answer"] = "T"
    return rounds


def test_parent_bootstrap_matches_the_per_parent_lookup_bit_for_bit():
    rng = np.random.default_rng(1)
    values = pd.DataFrame({
        "parent_id": [f"p{i}" for i in range(9)],
        "paired_difference": rng.normal(size=9) / 7,
    })
    for seed in (1, 7, 12):
        assert ce._parent_bootstrap(values, resamples=50, confidence=0.9, seed=seed) == \
            _reference_parent_bootstrap(values, resamples=50, confidence=0.9, seed=seed)


def test_activation_response_matches_the_per_parent_lookup_bit_for_bit():
    rounds = _sensing_rounds()
    new = ce.sensing_activation_response(rounds, bootstrap_resamples=40, confidence=0.95, seed=3)
    assert not new.empty and new["eligible_rounds"].gt(0).all()
    # Rebuild the contributions exactly as the function does, then run the old draw loop on them.
    keys = ["q", "rho", "branch_policy", "posting_budget"]
    sensing = rounds[rounds["branch_policy"].isin(("sensing_truth", "sensing_false"))]
    for group_index, (coordinates, group) in enumerate(sensing.groupby(keys, dropna=False, sort=True)):
        contributions = []
        for record in group.to_dict(orient="records"):
            p, u, target = record["controller_advocate_probability"], int(record["controller_sampled_U"]), str(record["controller_target_semantic_id"])
            change = (ce._count(record, target) - ce._count_before(record, target)) / int(record["N"])
            contributions.append({"parent_id": str(record["parent_id"]),
                                  "contribution": (u / p - (1 - u) / (1 - p)) * change})
        frame = pd.DataFrame(contributions)
        parents = frame["parent_id"].drop_duplicates().to_numpy()
        draws = _reference_activation_draws(frame, parents, resamples=40, seed=3 + group_index)
        row = new.iloc[group_index]
        alpha = (1.0 - 0.95) / 2.0  # the function's own alpha arithmetic, not the literal 0.025
        assert (row["ci_low"], row["ci_high"]) == (float(np.quantile(draws, alpha)), float(np.quantile(draws, 1.0 - alpha)))


def test_pooled_branch_round_metrics_equal_serial():
    rounds = _sensing_rounds()
    kwargs = dict(bootstrap_resamples=30, null_permutations=20, confidence=0.95, seed=11)
    serial_metrics, serial_nulls = ce.branch_round_metrics(rounds, workers=1, **kwargs)
    pooled_metrics, pooled_nulls = ce.branch_round_metrics(rounds, workers=3, **kwargs)
    assert not serial_metrics.empty
    pd.testing.assert_frame_equal(serial_metrics, pooled_metrics, check_exact=True)
    pd.testing.assert_frame_equal(serial_nulls, pooled_nulls, check_exact=True)
    assert set(serial_metrics["metric"]) >= {"T_pi", "chi", "eta_IR", "controller_action_entropy"}
    assert serial_metrics["n_parents"].eq(5).all()
