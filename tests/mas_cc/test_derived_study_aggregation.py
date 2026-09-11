from __future__ import annotations

import math
from dataclasses import replace

import pandas as pd
import pytest

from mas_cc.games.hidden_bench.imitation.controller import ADVOCATE_TARGET, NO_OP
from mas_cc.games.hidden_bench.imitation_round_feedback.analysis import RoundEvent
from mas_cc.studies.derived_aggregation import derive_study_control_aggregates


def _event(
    cell: str,
    episode: int,
    round_index: int,
    *,
    before: int,
    after: int,
    advocate: bool,
    probability: float = 0.5,
) -> RoundEvent:
    return RoundEvent(
        cell_id=cell,
        episode_id=f"{cell}/episode-{episode}",
        round_index=round_index,
        event={
            "episode_id": f"episode-{episode}",
            "N": 4,
            "occupation_counts_before": [before, 4 - before],
            "occupation_counts_after": [after, 4 - after],
            "target_count_before": before,
            "target_count_after": after,
            "truth_count_before": before,
            "truth_count_after": after,
            "controller_action": ADVOCATE_TARGET if advocate else NO_OP,
            "controller_advocate_probability": probability,
            "delta_p_ctrl": (after - before) / 4,
            "delta_m_ctrl": (after - before) / 2,
        },
    )


def _cell_events(cell: str, strength: int, *, episodes: int = 4) -> list[RoundEvent]:
    rows = []
    for episode in range(episodes):
        rows.extend(
            [
                _event(cell, episode, 0, before=1, after=1 + strength, advocate=True),
                _event(cell, episode, 1, before=1, after=1, advocate=False),
                _event(cell, episode, 2, before=2, after=min(4, 2 + strength), advocate=True),
                _event(cell, episode, 3, before=2, after=2, advocate=False),
            ]
        )
    return rows


def _cells() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "study_id": "study",
                "cell_id": "truth-rho1",
                "task_id": "task",
                "population_size": 4,
                "social_group_size": 2,
                "epistemic_persistence": 0.7,
                "intervention_budget": 3,
                "target_semantics": "truth",
                "completed_episodes": 4,
                "expected_episodes": 4,
            },
            {
                "study_id": "study",
                "cell_id": "false-rho2",
                "task_id": "task",
                "population_size": 4,
                "social_group_size": 2,
                "epistemic_persistence": 1.0,
                "intervention_budget": 3,
                "target_semantics": False,
                "completed_episodes": 4,
                "expected_episodes": 4,
            },
        ]
    )


def _recipe(*, state_local: bool = True) -> dict:
    return {
        "state_local_x_bins": 2,
        "derived_study_aggregates": {
            "enabled": True,
            "metrics": [
                "round_target_actuation_cmi",
                "susceptibility_occupancy_weighted",
                "round_target_information_fraction",
                "eta_ir",
            ],
            "groupings": [
                {
                    "name": "all",
                    "group_by": ["intervention_budget"],
                    "marginalize": ["epistemic_persistence", "target_semantics"],
                    "weighting": "balanced_cell",
                }
            ],
            "state_local": {
                "enabled": state_local,
                "groupings": [
                    {
                        "name": "all-x",
                        "group_by": ["intervention_budget"],
                        "marginalize": ["epistemic_persistence", "target_semantics"],
                    }
                ],
            },
        },
    }


def _run(events, cells=None, *, bootstrap=8, null=9):
    return derive_study_control_aggregates(
        events,
        _cells() if cells is None else cells,
        _recipe(),
        {
            "bootstrap_resamples": bootstrap,
            "null_permutations": null,
            "confidence": 0.95,
            "seed": 17,
        },
        "hash",
    )


def test_balanced_cells_ignore_observation_counts_and_ratios_use_components():
    events = _cell_events("truth-rho1", 1, episodes=2) + _cell_events(
        "false-rho2", 2, episodes=8
    )
    cells = _cells()
    cells.loc[cells.cell_id == "truth-rho1", ["completed_episodes", "expected_episodes"]] = 2
    cells.loc[cells.cell_id == "false-rho2", ["completed_episodes", "expected_episodes"]] = 8
    result = _run(events, cells)
    table = result.study_metrics.set_index("metric")

    # The aggregate is the equal-cell mean, not the observation-weighted mean.
    from mas_cc.studies.derived_aggregation import _components

    first = _components([row for row in events if row.cell_id == "truth-rho1"])
    second = _components([row for row in events if row.cell_id == "false-rho2"])
    assert table.loc["round_target_actuation_cmi", "estimate"] == pytest.approx(
        (first["T"] + second["T"]) / 2
    )
    assert table.loc["round_target_information_fraction", "estimate"] == pytest.approx(
        (first["T"] + second["T"]) / (first["H"] + second["H"])
    )
    assert table.loc["eta_ir", "estimate"] == pytest.approx(
        (first["ir_numerator"] + second["ir_numerator"])
        / (first["ir_denominator"] + second["ir_denominator"])
    )


def test_aggregate_p_value_comes_from_aggregate_null_draws():
    result = _run(_cell_events("truth-rho1", 1) + _cell_events("false-rho2", 2))
    transfer = result.study_metrics.query("metric == 'round_target_actuation_cmi'").iloc[0]
    assert transfer.n_null_draws == 9
    assert 1 / 10 <= transfer.permutation_p_value <= 1
    assert math.isfinite(transfer.null_mean_bits)
    assert "p_value" not in result.study_metrics.columns


def test_state_local_uses_observation_weights_and_target_coordinates():
    events = _cell_events("truth-rho1", 1, episodes=2) + _cell_events(
        "false-rho2", 2, episodes=6
    )
    result = _run(events)
    local = result.state_local_metrics
    assert set(local["aggregation_weight"]) == {"n_observations"}
    assert local["descriptive_only"].all()
    assert set(local["n_target_semantics_contributing"]) == {2}
    assert local["target_semantics"].isna().all()
    assert (local["n_observations"] > 0).all()
    assert (local["bootstrap_resamples"] == 8).all()
    assert local["ci_low"].notna().all()
    assert local["ci_high"].notna().all()
    assert set(local["n_rho_contributing"]) == {2}
    assert set(local["rho_values_json"]) == {'["0.7", "1.0"]'}


def test_order_and_scheduler_labels_do_not_change_results():
    events = _cell_events("truth-rho1", 1) + _cell_events("false-rho2", 2)
    first = _run(events).study_metrics.sort_values("metric").reset_index(drop=True)
    renamed = {
        "truth-rho1": "shard-99-truth",
        "false-rho2": "shard-01-false",
    }
    shuffled = [
        replace(row, cell_id=renamed[row.cell_id], episode_id=row.episode_id.replace(row.cell_id, renamed[row.cell_id]))
        for row in reversed(events)
    ]
    cells = _cells().iloc[::-1].copy()
    cells["cell_id"] = cells["cell_id"].map(renamed)
    second = _run(shuffled, cells).study_metrics.sort_values("metric").reset_index(drop=True)
    pd.testing.assert_series_equal(first["estimate"], second["estimate"], check_names=False)


def test_truth_false_labels_do_not_change_target_coordinate_pooling():
    events = _cell_events("truth-rho1", 1) + _cell_events("false-rho2", 1)
    first = _run(events).study_metrics.sort_values("metric").reset_index(drop=True)
    swapped = _cells()
    swapped["target_semantics"] = swapped["target_semantics"].map(
        {"truth": False, False: "truth"}
    )
    second = _run(events, swapped).study_metrics.sort_values("metric").reset_index(
        drop=True
    )
    pd.testing.assert_series_equal(
        first["estimate"], second["estimate"], check_names=False
    )


def test_unsupported_cell_reduces_coverage_without_zero_fill():
    supported = _cell_events("truth-rho1", 1)
    unsupported = [
        replace(row, event={**row.event, "controller_action": NO_OP})
        for row in _cell_events("false-rho2", 2)
    ]
    result = _run(supported + unsupported)
    row = result.study_metrics.query("metric == 'round_target_actuation_cmi'").iloc[0]
    assert row.n_contributing_cells == 1
    assert row.n_expected_cells == 2
    assert row.cell_coverage_fraction == pytest.approx(0.5)
    assert row.support_status == "limited"


def test_bootstrap_recomputes_ratio_and_physical_input_is_unchanged():
    events = _cell_events("truth-rho1", 1) + _cell_events("false-rho2", 2)
    before = [(row.cell_id, row.episode_id, dict(row.event)) for row in events]
    result = _run(events, bootstrap=12)
    eta_if = result.study_metrics.query(
        "metric == 'round_target_information_fraction'"
    ).iloc[0]
    assert eta_if.bootstrap_resamples == 12
    assert math.isfinite(eta_if.ci_low)
    assert before == [(row.cell_id, row.episode_id, dict(row.event)) for row in events]


def test_unaccounted_scientific_dimension_is_rejected():
    cells = _cells()
    cells["model"] = ["a", "b"]
    with pytest.raises(ValueError, match="silently mix"):
        _run(_cell_events("truth-rho1", 1) + _cell_events("false-rho2", 2), cells)


def test_state_local_grouping_cannot_silently_mix_a_scientific_dimension():
    cells = _cells()
    cells["model"] = ["a", "b"]
    recipe = _recipe()
    recipe["derived_study_aggregates"]["groupings"][0]["marginalize"].append("model")
    with pytest.raises(ValueError, match="state-local aggregation would silently mix"):
        derive_study_control_aggregates(
            _cell_events("truth-rho1", 1) + _cell_events("false-rho2", 2),
            cells,
            recipe,
            {
                "bootstrap_resamples": 2,
                "null_permutations": 2,
                "confidence": 0.95,
                "seed": 17,
            },
            "hash",
        )


def test_balanced_state_local_semantics_requires_both_supported_arms():
    unsupported = [
        replace(row, event={**row.event, "controller_action": NO_OP})
        for row in _cell_events("false-rho2", 2)
    ]
    recipe = _recipe()
    recipe["derived_study_aggregates"]["state_local"]["weighting"] = "balanced_cell"
    result = derive_study_control_aggregates(
        _cell_events("truth-rho1", 1) + unsupported,
        _cells(),
        recipe,
        {
            "bootstrap_resamples": 2,
            "null_permutations": 2,
            "confidence": 0.95,
            "seed": 17,
        },
        "hash",
    )
    assert set(result.state_local_metrics["support_status"]) == {"unsupported"}
    assert set(result.state_local_metrics["n_target_semantics_contributing"]) == {1}


def test_state_local_reconstruction_is_exported_as_a_diagnostic():
    result = _run(
        _cell_events("truth-rho1", 1) + _cell_events("false-rho2", 2)
    )
    diagnostic = result.state_local_reconstruction
    assert len(diagnostic) == 1
    assert diagnostic.iloc[0].diagnostic_only
    assert diagnostic.iloc[0].n_state_bins == 2
    assert math.isfinite(diagnostic.iloc[0].state_local_reconstruction)
    assert math.isfinite(diagnostic.iloc[0].whole_cell_aggregate)
    assert diagnostic.iloc[0].difference == pytest.approx(
        diagnostic.iloc[0].state_local_reconstruction
        - diagnostic.iloc[0].whole_cell_aggregate
    )
