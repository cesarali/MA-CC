"""Single-affinity bootstrap on index arrays reproduces the row path's draws and intervals exactly."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np

from mas_cc.analysis import single_affinity as sa
from mas_cc.analysis.draw_components import _DrawComponents
from mas_cc.games.hidden_bench.imitation_round_feedback import analysis as round_analysis

_spec = importlib.util.spec_from_file_location("sa_fixtures", Path(__file__).with_name("test_single_affinity_consistency.py"))
fixtures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fixtures)


def _same(a, b):
    return (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b)) or a == b


def _data(seed=37, episodes=60, rounds=3):
    return fixtures.simulate_single_affinity(fixtures._parameters(N=6, q_c=3, b=4), episodes=episodes, rounds=rounds, n0=2, seed=seed)


def test_fast_engine_reproduces_the_row_engine_intervals(monkeypatch):
    rows, micro = _data()
    monkeypatch.setattr(round_analysis, "_INFORMATION_ENGINE", "rows")
    slow = sa.single_affinity_analysis(rows, micro, bootstrap_resamples=80, confidence=0.9, seed=101)
    monkeypatch.setattr(round_analysis, "_INFORMATION_ENGINE", "fast")
    assert sa._fast_draws(rows, *sa._by_episode(rows, micro)[:1], sorted(sa._by_episode(rows, micro)[0], key=str)) is not None
    fast = sa.single_affinity_analysis(rows, micro, bootstrap_resamples=80, confidence=0.9, seed=101)
    assert set(slow) == set(fast)
    for key in slow:
        assert _same(slow[key], fast[key]), (key, slow[key], fast[key])
    assert any(math.isfinite(fast[f"{name}_ci_low"]) for name in ("eta_ir", "eta_th", "controlled_current_horizon"))


def test_scalars_match_point_estimate_draw_by_draw():
    rows, micro = _data(seed=5, episodes=30)
    grouped_rounds, grouped_micro = sa._by_episode(rows, micro)
    keys = sorted(grouped_rounds, key=str)
    micro_counts = {key: sa.controlled_transition_counts(grouped_micro.get(key, ())) for key in keys}
    eligible = sa.controlled_rows(rows)
    N, q_c = sa.population_size(eligible), sa.sensor_sample_size(eligible)
    components = _DrawComponents(rows, filter_controlled=True)
    position = {id(row): i for i, row in enumerate(rows)}
    members = {key: np.array([position[id(row)] for row in grouped_rounds[key]]) for key in keys}
    S = sa.sensor_kernel(N, q_c)
    rng = np.random.default_rng(9)
    for _ in range(25):
        selected = [keys[index] for index in rng.integers(0, len(keys), len(keys))]
        counts = {name: sum(micro_counts[key][name] for key in selected) for name in micro_counts[keys[0]]}
        expected = sa.point_estimate([row for key in selected for row in grouped_rounds[key]], _micro_counts=counts, _micro_count=0)
        got = components.single_affinity_scalars(np.concatenate([members[key] for key in selected]), N=N, S=S, micro_counts=counts)
        for name in sa._SCALARS:
            assert _same(float(expected.get(name, math.nan)), float(got[name])), (name, expected.get(name), got[name])


def test_row_path_is_kept_when_the_cell_mixes_sensor_sizes(monkeypatch):
    rows, micro = _data(seed=3, episodes=8)
    monkeypatch.setattr(round_analysis, "_INFORMATION_ENGINE", "fast")
    rows[0].event.pop("sensor_sample_size", None)
    grouped_rounds, _ = sa._by_episode(rows, micro)
    assert sa._fast_draws(rows, grouped_rounds, sorted(grouped_rounds, key=str)) is None
