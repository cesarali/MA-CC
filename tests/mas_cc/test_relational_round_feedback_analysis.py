"""The relational adapter into the shared round-feedback estimators.

Nothing here re-tests the estimators themselves - those have their own
fixtures next door. What is worth pinning down is the seam: that a relational
round record resolves into the same `RoundEvent` shape, and that the
memory-aware conditioning genuinely *conditions* rather than being carried
along and ignored.
"""

from __future__ import annotations

import json
import math

import pytest

from mas_cc.games.hidden_bench.imitation.controller import ADVOCATE_TARGET, NO_OP
from mas_cc.games.hidden_bench.imitation_round_feedback.analysis import (
    MAIN_ESTIMATOR_VARIANT,
    round_information_analysis,
)
from mas_cc.games.relational_reasoning.imitation_round_feedback.analysis import (
    adapt_relational_round_record,
    analyze_relational_imitation_round_feedback,
    controller_action_summary,
    read_relational_round_records,
)

OPTIONS = ["SOUTHWEST", "WEST", "EAST"]
CORRECT = "WEST"
TARGET = "EAST"  # index 2, the pinned wrong target of the pilot


def _record(
    *,
    episode: str,
    index: int,
    action: str | None,
    before=(14, 5, 5),
    after=(14, 5, 5),
    strata_before=(12, 12, 0),
    kappa=0.25,
    phi=0.0,
    probability=0.5,
):
    total = sum(before)
    return {
        "record_type": "relational_imitation_round_feedback",
        "episode_id": episode,
        "round_index": index,
        "task_id": "task_0001",
        "possible_answers": list(OPTIONS),
        "correct_answer": CORRECT,
        "analysis_target": None if action is None else TARGET,
        "occupation_counts_before": list(before),
        "occupation_counts_after": list(after),
        "controller_action": action,
        "controller_advocate_probability": None if action is None else probability,
        "sensor_count_vector": None if action is None else [6, 3, 3],
        "sensor_target_share": None if action is None else 0.25,
        "sensor_sample_size": None if action is None else 12,
        "controlled_position_count": 12 if action == ADVOCATE_TARGET else 0,
        "knowledge_stratum_counts_before": list(strata_before),
        "mean_supporting_fact_coverage_before": kappa,
        "mean_supporting_fact_coverage": kappa,
        "full_proof_agent_share_before": phi,
        "full_proof_agent_share": phi,
        "delta_m_ctrl": (after[2] - before[2]) * 1.5 / total,
        "delta_m_truth": (after[1] - before[1]) * 1.5 / total,
        "delta_m_order": (max(after) - max(before)) * 1.5 / total,
        "delta_H_vote": 0.0,
    }


def test_adapter_resolves_target_and_truth_counts_from_the_option_vector():
    row = adapt_relational_round_record(
        _record(episode="e0", index=0, action=ADVOCATE_TARGET, before=(14, 5, 5), after=(10, 5, 9))
    )
    # `EAST` is index 2 and `WEST` index 1: the pipeline's scalar channels have
    # to come from the label, never from a positional assumption.
    assert (row.target_before, row.target_after) == (5, 9)
    assert (row.truth_before, row.truth_after) == (5, 5)
    assert row.memory_state == (12, 12, 0)
    assert row.event["delta_p_ctrl"] == pytest.approx(4 / 24)


def test_an_uncontrolled_record_falls_back_to_the_correct_answer():
    row = adapt_relational_round_record(_record(episode="e0", index=0, action=None))
    assert row.target_before == row.truth_before
    assert row.U_k is None


def test_epistemic_bins_are_coarse_and_bounded():
    row = adapt_relational_round_record(
        _record(episode="e0", index=0, action=NO_OP, kappa=1.0, phi=0.4), epistemic_bins=4
    )
    # kappa = 1.0 must land in the last bin rather than off the end.
    assert row.epistemic_state == (3, 1)


def test_memory_conditioning_recovers_an_effect_the_plain_cmi_averages_away():
    """The whole reason `E_k` is in the conditioning state.

    Two memory regimes push the target the opposite way. Marginally, given
    `n_Z,k`, the action then says nothing at all about `n_Z,k+1`; the effect
    only exists once `E_k` is conditioned on. If the memory statistic were
    silently reusing the plain conditioning, both numbers would agree.
    """

    rows = []
    for repetition in range(6):
        for strata, up in (((12, 12, 0), True), ((0, 12, 12), False)):
            for action, moved in ((ADVOCATE_TARGET, up), (NO_OP, not up)):
                after = (13, 5, 6) if moved else (15, 5, 4)
                rows.append(
                    adapt_relational_round_record(
                        _record(
                            episode=f"e{repetition}-{strata[0]}",
                            index=len(rows),
                            action=action,
                            before=(14, 5, 5),
                            after=after,
                            strata_before=strata,
                        )
                    )
                )

    estimates, _ = round_information_analysis(
        rows,
        statistics=["round_target_actuation_cmi", "round_memory_target_actuation_cmi"],
        bootstrap_resamples=0,
        null_permutations=0,
    )
    values = {row["statistic"]: row["estimate"] for row in estimates}
    assert values["round_target_actuation_cmi"] == pytest.approx(0.0, abs=1e-9)
    assert values["round_memory_target_actuation_cmi"] == pytest.approx(1.0, abs=1e-9)

    support = {row["statistic"]: row for row in estimates}
    # Each statistic reports the sparsity of its OWN conditioning state.
    assert support["round_target_actuation_cmi"]["round_conditioning_state_count"] == 1
    assert support["round_memory_target_actuation_cmi"]["round_conditioning_state_count"] == 2


def test_a_constant_memory_state_leaves_the_estimate_untouched():
    """Conditioning on a constant must add exactly nothing.

    This is the mock-provider case: no fact ever moves, so `E_k` never moves,
    and the memory-aware number has to collapse onto the plain one instead of
    inventing structure.
    """

    rows = [
        adapt_relational_round_record(
            _record(
                episode=f"e{index % 4}",
                index=index,
                action=ADVOCATE_TARGET if index % 3 else NO_OP,
                after=(13, 5, 6) if index % 2 else (15, 5, 4),
            )
        )
        for index in range(24)
    ]
    estimates, _ = round_information_analysis(
        rows,
        statistics=["round_target_actuation_cmi", "round_memory_target_actuation_cmi"],
        bootstrap_resamples=0,
        null_permutations=0,
    )
    values = {row["statistic"]: row["estimate"] for row in estimates}
    assert values["round_memory_target_actuation_cmi"] == pytest.approx(
        values["round_target_actuation_cmi"]
    )


def test_budget_and_sensor_bookkeeping_is_reported_per_action():
    rows = [
        adapt_relational_round_record(
            _record(episode="e0", index=index, action=ADVOCATE_TARGET if index < 7 else NO_OP)
        )
        for index in range(10)
    ]
    summary = controller_action_summary(rows)
    assert summary["advocate_rounds"] == 7
    assert summary["no_op_rounds"] == 3
    assert summary["mean_controlled_positions_on_advocate"] == pytest.approx(12.0)
    assert summary["max_controlled_positions_on_no_op"] == 0
    assert summary["mean_sensor_sample_size"] == pytest.approx(12.0)


def test_analysis_writes_the_report_and_flags_memory_support(tmp_path):
    episodes = tmp_path / "cells" / "cell-0000" / "data" / "episodes" / "cell-0000-0000"
    episodes.mkdir(parents=True)
    (tmp_path / "cells" / "cell-0000" / "overrides.json").write_text("{}", encoding="utf-8")
    (episodes / "round_trajectory.jsonl").write_text(
        "\n".join(
            json.dumps(
                _record(
                    episode="cell-0000-0000",
                    index=index,
                    action=ADVOCATE_TARGET if index % 2 else NO_OP,
                    after=(13, 5, 6) if index % 3 else (15, 5, 4),
                )
            )
            for index in range(10)
        )
        + "\n",
        encoding="utf-8",
    )

    assert len(read_relational_round_records(tmp_path)) == 10
    summary = analyze_relational_imitation_round_feedback(
        tmp_path, tmp_path / "analysis", bootstrap_resamples=5, null_permutations=5
    )
    assert summary["n_rounds"] == 10
    assert summary["n_cells"] == 1
    for name in (
        "round_information_estimates.csv",
        "round_information_nulls.csv",
        "round_support_diagnostics.csv",
        "controller_action_summary.csv",
        "episode_epistemic_regime.csv",
        "round_epistemic_trajectory.csv",
    ):
        assert (tmp_path / "analysis" / name).is_file()
    reported = {row["statistic"] for row in summary["memory_conditioning_support"]}
    assert reported == {
        "round_memory_target_actuation_cmi",
        "round_epistemic_target_actuation_cmi",
    }
    assert all(
        math.isfinite(float(row["estimate"]))
        for row in summary["memory_conditioning_support"]
    )
