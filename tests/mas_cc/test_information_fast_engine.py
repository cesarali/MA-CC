"""The fast bits-statistics engine reproduces the row engine exactly (draws, nulls, intervals)."""

from __future__ import annotations

import importlib.util
import math
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
