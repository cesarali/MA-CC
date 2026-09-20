"""Adaptive bootstrap stopping: off is byte-identical, on stops early within tolerance, settings hash only when on."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from mas_cc.analysis.adaptive import AdaptiveResampling, IntervalStopper, coerce
from mas_cc.games.hidden_bench.imitation_round_feedback import analysis
from mas_cc.studies.aggregation import _resampling

_spec = importlib.util.spec_from_file_location("fe", Path(__file__).with_name("test_information_fast_engine.py"))
fe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fe)

STATS = ["round_sensing_mi", "round_target_actuation_cmi", "round_controller_action_entropy_given_population",
         "round_target_signed_actuation", "round_sensor_mae"]


def test_config_parsing_and_coercion():
    assert coerce(None) is None and coerce({}) is None and coerce({"enabled": False}) is None
    config = coerce({"enabled": True, "min_resamples": 50, "check_every": 10, "tolerance": 0.05})
    assert config == AdaptiveResampling(True, 50, 10, 0.05)
    with pytest.raises(ValueError):
        AdaptiveResampling.from_mapping({"enabled": True, "min_resamples": 1})
    with pytest.raises(ValueError):
        AdaptiveResampling.from_mapping({"enabled": True, "tolerance": 1.5})


def test_resampling_settings_carry_adaptive_only_when_enabled():
    plain = _resampling({"resampling": {"bootstrap_resamples": 10}})
    off = _resampling({"resampling": {"bootstrap_resamples": 10, "adaptive": {"enabled": False}}})
    assert plain == off and "adaptive" not in plain
    on = _resampling({"resampling": {"bootstrap_resamples": 10, "adaptive": {"enabled": True, "check_every": 5}}})
    assert on["adaptive"] == {"enabled": True, "min_resamples": 200, "check_every": 5, "tolerance": 0.02}


def test_stopper_waits_for_min_then_stops_when_endpoints_settle():
    stopper = IntervalStopper(AdaptiveResampling(True, min_resamples=20, check_every=10, tolerance=0.05), alpha=0.025)
    rng = np.random.default_rng(0)
    values: list[float] = []
    stopped = None
    for draw in range(1, 2001):
        values.append(float(rng.normal()))
        if stopper.should_stop(draw, values):
            stopped = draw
            break
    assert stopped is not None and stopped >= 30 and stopped % 10 == 0
    assert stopper.stopped_at == stopped
    # never stops on the very first check, never before min_resamples
    fresh = IntervalStopper(AdaptiveResampling(True, min_resamples=20, check_every=10, tolerance=0.5), alpha=0.025)
    assert not fresh.should_stop(19, [0.0] * 19) and not fresh.should_stop(20, [0.0, 1.0] * 10)
    assert fresh.should_stop(30, [0.0, 1.0] * 15)  # identical interval → moved (0, 0)


def test_disabled_config_is_byte_identical_to_no_config(monkeypatch):
    rows = fe._rich_rows()
    for engine in ("rows", "fast"):
        plain = fe._run(monkeypatch, engine, rows, STATS, bootstrap_resamples=60, null_permutations=10, seed=4)
        off = fe._run(monkeypatch, engine, rows, STATS, bootstrap_resamples=60, null_permutations=10, seed=4,
                      adaptive={"enabled": False})
        fe._assert_identical(plain[0], off[0])
        assert all("bootstrap_draws" not in row for row in off[0])


@pytest.mark.parametrize("engine", ["rows", "fast"])
def test_enabled_stops_early_and_stays_within_tolerance(monkeypatch, engine):
    rows = fe._rich_rows(seed=2, episodes=12, rounds=10)
    config = {"enabled": True, "min_resamples": 100, "check_every": 50, "tolerance": 0.05}
    full = fe._run(monkeypatch, engine, rows, STATS, bootstrap_resamples=800, null_permutations=5, seed=4)
    early = fe._run(monkeypatch, engine, rows, STATS, bootstrap_resamples=800, null_permutations=5, seed=4, adaptive=config)
    assert [r["statistic"] for r in full[0]] == [r["statistic"] for r in early[0]]
    stopped_somewhere = False
    for reference, adaptive in zip(full[0], early[0], strict=True):
        draws = adaptive["bootstrap_draws"]
        assert 100 <= draws <= 800 and (draws == 800 or (draws - 100) % 50 == 0)
        stopped_somewhere |= draws < 800
        width = reference["bootstrap_ci_high"] - reference["bootstrap_ci_low"]
        if math.isfinite(width) and width > 0:
            # early-stopped endpoints sit close to the full-draw ones (loose: 25 % of the width)
            assert abs(adaptive["bootstrap_ci_low"] - reference["bootstrap_ci_low"]) <= 0.25 * width + 1e-12
            assert abs(adaptive["bootstrap_ci_high"] - reference["bootstrap_ci_high"]) <= 0.25 * width + 1e-12
        # nulls are never adaptive
        a, b = adaptive["null_mean"], reference["null_mean"]
        assert (math.isnan(a) and math.isnan(b)) or a == b
    assert stopped_somewhere, "no statistic stopped early on 800 draws"
    fe._assert_identical(early[1], full[1])
