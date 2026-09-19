"""Derived aggregates: the index-array draw components and the fast policy null reproduce the row path exactly."""

from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mas_cc.games.hidden_bench.imitation_round_feedback import analysis as round_analysis
from mas_cc.studies import derived_aggregation as D
from mas_cc.studies.weighted_summaries import PairedBootstrap

_spec = importlib.util.spec_from_file_location("da_fixtures", Path(__file__).with_name("test_derived_study_aggregation.py"))
fixtures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fixtures)


def _events(paired: bool):
    events = fixtures._cell_events("truth-rho1", 1, episodes=6) + fixtures._cell_events("false-rho2", 2, episodes=6)
    if paired:
        events = [replace(e, event={**e.event, "physical_initial_state_hash": e.episode_id.rsplit("/", 1)[-1]})
                  for e in events]
    return events


def _config(paired: bool):
    config = fixtures._recipe()
    if paired:
        config["derived_study_aggregates"]["bootstrap"] = {"unit": "shared_initialization_block"}
    return config


SETTINGS = {"bootstrap_resamples": 40, "null_permutations": 25, "confidence": 0.9, "seed": 7}


@pytest.mark.parametrize("paired", [False, True])
def test_fast_engine_reproduces_every_output_frame(monkeypatch, paired):
    events, config = _events(paired), _config(paired)
    monkeypatch.setattr(round_analysis, "_INFORMATION_ENGINE", "rows")
    slow = D.derive_study_control_aggregates(events, fixtures._cells(), config, SETTINGS, "hash")
    monkeypatch.setattr(round_analysis, "_INFORMATION_ENGINE", "fast")
    fast = D.derive_study_control_aggregates(events, fixtures._cells(), config, SETTINGS, "hash")
    for name in ("study_metrics", "state_local_metrics", "stability", "state_local_reconstruction"):
        left, right = getattr(slow, name), getattr(fast, name)
        assert left.shape == right.shape, name
        if not left.empty:
            pd.testing.assert_frame_equal(left, right, check_exact=True)
    assert not slow.study_metrics.empty and not slow.state_local_metrics.empty
    assert slow.study_metrics["bootstrap_resamples"].eq(40).all()


@pytest.mark.parametrize("paired", [False, True])
def test_draw_components_match_row_components_draw_by_draw(paired):
    events = _events(paired)
    cell_rows = [e for e in events if str(e.cell_id) == "truth-rho1"]
    eligible = D.controlled_rows(cell_rows)
    plan = None
    if paired:
        plan = PairedBootstrap(pd.DataFrame([{**e.event, "cell_id": str(e.cell_id), "episode_id": str(e.episode_id)} for e in events]),
                               resamples=30, seed=3)
    fast = D._DrawComponents(eligible)
    reference = list(plan.event_draws(eligible)) if plan is not None else list(
        round_analysis.bootstrap_episode_rows(eligible, resamples=30, seed=11))
    orders = list(D._draw_orders(eligible, bootstrap_plan=plan, resamples=30, seed=11))
    assert len(orders) == len(reference) == 30
    for order, draw in zip(orders, reference, strict=True):
        assert [id(eligible[i]) for i in order] == [id(row) for row in draw], "draw order differs"
        expected = D._components(draw)
        got = fast.value(order)
        for key in expected:
            a, b = expected[key], got[key]
            assert (np.isnan(a) and np.isnan(b)) or a == b, (key, a, b)


def test_fast_policy_null_is_the_scalar_stream():
    eligible = D.controlled_rows(_events(False))
    assert D._policy_null(eligible, permutations=15, seed=5) == round_analysis.policy_resampling_null(
        D.TARGET_CMI, eligible, permutations=15, seed=5)
    assert D._policy_null([], permutations=15, seed=5) == ()
