"""The fast engine reproduces the row engine exactly: bits bootstraps, policy and sensor nulls, diagnostic bootstraps."""

from __future__ import annotations

import importlib.util
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from mas_cc.games.hidden_bench.imitation.controller import ADVOCATE_TARGET, NO_OP
from mas_cc.games.hidden_bench.imitation_round_feedback import analysis

_spec = importlib.util.spec_from_file_location(
    "rf_fixtures", Path(__file__).with_name("test_hidden_bench_imitation_round_feedback_analysis.py"))
fixtures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fixtures)


def _rows(seed: int = 5, episodes: int = 7, rounds: int = 9):
    rng = np.random.default_rng(seed)
    rows = []
    for episode in range(episodes):
        for index in range(rounds):
            before = tuple(int(v) for v in rng.multinomial(6, [0.4, 0.4, 0.2]))
            after = tuple(int(v) for v in rng.multinomial(6, [0.5, 0.3, 0.2]))
            sensor = tuple(int(v) for v in rng.multinomial(3, [0.5, 0.3, 0.2]))
            action = None if rng.random() < 0.1 else (ADVOCATE_TARGET if rng.random() < 0.6 else NO_OP)
            rows.append(fixtures._row(episode=f"ep-{episode}", index=index, action=action,
                                      before=before, after=after, sensor=sensor, probability=float(rng.uniform(0.2, 0.9))))
    return rows


def _rich_rows(seed: int = 11, episodes: int = 6, rounds: int = 8):
    """Rows carrying every field the diagnostics read: share deltas, scalar sensor, memory states."""
    rng = np.random.default_rng(seed)
    out = []
    for row in _rows(seed=seed, episodes=episodes, rounds=rounds):
        extra = {
            "delta_p_ctrl": float(rng.choice([-0.25, 0.0, 0.25, 0.5])),
            "sensor_target_count": None if row.Y_k is None else int(row.Y_k[0]),
            "conditioning_memory_state": [int(v) for v in rng.integers(0, 3, size=3)],
            "conditioning_epistemic_state": [int(rng.integers(0, 2)), int(rng.integers(0, 2))],
            "conditioning_phi_bin": int(rng.integers(0, 3)),
            "conditioning_susceptible_bin": int(rng.integers(0, 3)),
            "conditioning_kappa_bin": int(rng.integers(0, 2)),
        }
        out.append(replace(row, event={**dict(row.event), **extra}))
    return out


def _run(monkeypatch, engine: str, rows, statistics, **kwargs):
    monkeypatch.setattr(analysis, "_INFORMATION_ENGINE", engine)
    return analysis.round_information_analysis(rows, statistics=statistics, **kwargs)


def _assert_identical(a, b):
    assert len(a) == len(b)
    for left, right in zip(a, b, strict=True):
        assert set(left) == set(right)
        for key in left:
            x, y = left[key], right[key]
            if isinstance(x, float) and isinstance(y, float) and math.isnan(x) and math.isnan(y):
                continue
            assert x == y, (key, x, y)


BITS = [
    "round_sensing_mi", "round_target_sensing_mi", "round_sensor_action_mi",
    "round_population_actuation_cmi", "round_target_actuation_cmi", "round_truth_actuation_cmi",
]


def test_fast_engine_matches_row_engine_on_estimates_and_nulls(monkeypatch):
    rows = _rows()
    slow = _run(monkeypatch, "rows", rows, BITS, bootstrap_resamples=60, null_permutations=40, seed=3)
    fast = _run(monkeypatch, "fast", rows, BITS, bootstrap_resamples=60, null_permutations=40, seed=3)
    _assert_identical(slow[0], fast[0])
    _assert_identical(slow[1], fast[1])
    names = {row["statistic"] for row in fast[0]} if fast[0] and "statistic" in fast[0][0] else set()
    assert len(fast[0]) == len(BITS) or names, "every requested statistic must be present"


def test_fast_engine_handles_degenerate_inputs(monkeypatch):
    rows = [r for r in _rows(seed=1, episodes=1, rounds=3)]
    for engine in ("rows", "fast"):
        out = _run(monkeypatch, engine, rows, BITS, bootstrap_resamples=5, null_permutations=3, seed=1)
        assert isinstance(out[0], list)
    slow = _run(monkeypatch, "rows", rows, BITS, bootstrap_resamples=5, null_permutations=3, seed=1)
    fast = _run(monkeypatch, "fast", rows, BITS, bootstrap_resamples=5, null_permutations=3, seed=1)
    _assert_identical(slow[0], fast[0])


def test_policy_null_draws_match_scalar_stream():
    rows = _rows(seed=9, episodes=3, rounds=6)
    eligible = [row for row in rows if row.U_k in {ADVOCATE_TARGET, NO_OP}]
    fast = analysis._BitsPolicyNull("round_target_actuation_cmi", eligible).values(25, seed=77)
    slow = analysis.policy_resampling_null("round_target_actuation_cmi", eligible, permutations=25, seed=77)
    assert fast == slow


def test_unknown_statistic_still_rejected():
    with pytest.raises(ValueError):
        analysis._bits_sequences("round_nonsense", [])


def test_every_statistic_matches_the_row_engine_on_rich_rows(monkeypatch):
    rows = _rich_rows()
    names = list(analysis.ROUND_ANALYSIS_STATISTICS)
    slow = _run(monkeypatch, "rows", rows, names, bootstrap_resamples=40, null_permutations=30, seed=5)
    fast = _run(monkeypatch, "fast", rows, names, bootstrap_resamples=40, null_permutations=30, seed=5)
    assert {row["statistic"] for row in slow[0]} == set(names), "the rich rows must make every statistic eligible"
    _assert_identical(slow[0], fast[0])
    _assert_identical(slow[1], fast[1])
    assert {row["null_type"] for row in fast[1]} == {"policy_conditional_randomization", "sensor_permutation"}


def test_diagnostic_bootstrap_matches_row_draws_one_by_one():
    rows = [row for row in _rich_rows(seed=3) if row.U_k in {ADVOCATE_TARGET, NO_OP}]
    names = [name for name in analysis.ROUND_ANALYSIS_STATISTICS if name not in analysis._BITS_STATISTICS]
    for name in names:
        fast = analysis._DiagnosticBootstrap(name, rows)
        rng = np.random.default_rng(21)
        expected = [analysis._diagnostic_for(name, draw) for draw in analysis.bootstrap_episode_rows(rows, resamples=15, seed=21)]
        got = [fast.draw(rng) for _ in range(15)]
        for a, b in zip(expected, got, strict=True):
            assert (math.isnan(a) and math.isnan(b)) or a == b, (name, a, b)


def test_sensor_permutation_null_matches_row_engine(monkeypatch):
    rows = _rich_rows(seed=8)
    for name in ("round_sensing_mi", "round_target_sensing_mi", "round_sensor_action_mi"):
        slow = _run(monkeypatch, "rows", rows, [name], bootstrap_resamples=0, null_permutations=20, seed=2)
        fast = _run(monkeypatch, "fast", rows, [name], bootstrap_resamples=0, null_permutations=20, seed=2)
        assert [r["estimate"] for r in slow[1]] == [r["estimate"] for r in fast[1]], name


def test_sum_helpers_match_the_row_path_arithmetic():
    rng = np.random.default_rng(0)
    values = rng.normal(size=257) * 1e3 + rng.normal(size=257)
    assert analysis._builtin_sum(values) == sum(values.tolist())
    total = 0.0
    for value in values.tolist():
        total += value
    assert analysis._running_sum(values) == total
    assert math.copysign(1.0, analysis._running_sum(np.array([-0.0]))) == 1.0
    rank = rng.integers(0, 4, size=257)
    sums, counts = analysis._group_sums(rank, values, 5)
    for group in range(5):
        assert sums[group] == sum(values[rank == group].tolist()) and counts[group] == int((rank == group).sum())
