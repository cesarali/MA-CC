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
    COARSE_BIN_EDGES,
    COARSE_BIN_LABELS,
    EPISTEMIC_CONDITIONING_VARIABLES,
    adapt_relational_round_record,
    analyze_relational_imitation_round_feedback,
    coarse_bin,
    controller_action_summary,
    epistemic_conditioning_values,
    read_relational_round_records,
)

EPISTEMIC_CONDITIONING_STATISTICS = tuple(
    f"round_{name}_target_actuation_cmi" for name in EPISTEMIC_CONDITIONING_VARIABLES
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
        "round_phi_target_actuation_cmi",
        "round_susceptible_target_actuation_cmi",
        "round_kappa_target_actuation_cmi",
    }
    assert summary["coarse_bins"]["labels"] == ["low", "medium", "high"]
    assert summary["coarse_bins"]["conditioned_separately"] is True
    assert all(
        math.isfinite(float(row["estimate"]))
        for row in summary["memory_conditioning_support"]
    )


# ---------------------------------------------------------------------------
# Coarse-grained scalar epistemic conditioning
# ---------------------------------------------------------------------------


def test_the_three_bins_are_exactly_the_documented_half_open_intervals():
    assert COARSE_BIN_EDGES == (0.0, 1 / 3, 2 / 3, 1.0)
    assert COARSE_BIN_LABELS == ("low", "medium", "high")
    # The two interior edges belong to the bin ABOVE them, and 1.0 must land in
    # `high` rather than off the end - the case a multiply-and-truncate binner
    # gets wrong depending on how `value * 3` rounds.
    assert coarse_bin(0.0) == 0
    assert coarse_bin(0.3333) == 0
    assert coarse_bin(1 / 3) == 1
    assert coarse_bin(0.5) == 1
    assert coarse_bin(0.6666) == 1
    assert coarse_bin(2 / 3) == 2
    assert coarse_bin(1.0) == 2
    assert coarse_bin(None) is None


def test_phi_is_the_last_stratum_over_the_population():
    """`phi = n_L / N`, read off `E_k` rather than recomputed."""

    values = epistemic_conditioning_values(
        _record(episode="e0", index=0, action=NO_OP, strata_before=(4, 14, 6))
    )
    assert values["phi"] == pytest.approx(6 / 24)


def test_phi_falls_back_to_the_recorded_share_without_a_histogram():
    record = _record(episode="e0", index=0, action=NO_OP, phi=0.375)
    del record["knowledge_stratum_counts_before"]
    assert epistemic_conditioning_values(record)["phi"] == pytest.approx(0.375)


def test_susceptible_is_one_minus_phi():
    values = epistemic_conditioning_values(
        _record(episode="e0", index=0, action=NO_OP, strata_before=(4, 14, 6))
    )
    assert values["susceptible"] == pytest.approx(1.0 - values["phi"])


def test_kappa_is_the_already_recorded_coverage_and_is_not_recomputed():
    # Deliberately inconsistent with the strata: if kappa were being derived
    # from the histogram this would come out 0.5, not 0.8125.
    values = epistemic_conditioning_values(
        _record(episode="e0", index=0, action=NO_OP, strata_before=(12, 12, 0), kappa=0.8125)
    )
    assert values["kappa"] == pytest.approx(0.8125)


def test_each_scalar_variable_reaches_the_record_under_its_own_key():
    row = adapt_relational_round_record(
        _record(
            episode="e0", index=0, action=NO_OP, strata_before=(4, 14, 6), kappa=0.7
        )
    )
    assert row.event["phi_before"] == pytest.approx(6 / 24)
    assert row.event["susceptible_before"] == pytest.approx(18 / 24)
    assert row.event["kappa_before"] == pytest.approx(0.7)
    assert row.event["conditioning_phi_bin"] == 0  # 0.25  -> low
    assert row.event["conditioning_susceptible_bin"] == 2  # 0.75 -> high
    assert row.event["conditioning_kappa_bin"] == 2  # 0.70 -> high


def _epistemic_rows():
    """Rows whose target response flips with the coarse epistemic regime.

    Marginally, given `n_Z,k`, the action says nothing; the effect exists only
    once the regime is conditioned on. Both regimes are chosen so that phi, s
    and kappa each separate them.
    """

    rows = []
    for repetition in range(6):
        for strata, kappa, up in (((12, 12, 0), 0.25, True), ((2, 2, 20), 0.9, False)):
            for action, moved in ((ADVOCATE_TARGET, up), (NO_OP, not up)):
                rows.append(
                    adapt_relational_round_record(
                        _record(
                            episode=f"e{repetition}-{strata[0]}",
                            index=len(rows),
                            action=action,
                            before=(14, 5, 5),
                            after=(13, 5, 6) if moved else (15, 5, 4),
                            strata_before=strata,
                            kappa=kappa,
                        )
                    )
                )
    return rows


@pytest.mark.parametrize(
    "statistic",
    [
        "round_phi_target_actuation_cmi",
        "round_susceptible_target_actuation_cmi",
        "round_kappa_target_actuation_cmi",
    ],
)
def test_each_coarse_conditioning_recovers_the_effect_the_plain_cmi_averages_away(
    statistic,
):
    estimates, nulls = round_information_analysis(
        _epistemic_rows(),
        statistics=["round_target_actuation_cmi", statistic],
        bootstrap_resamples=4,
        null_permutations=4,
    )
    values = {row["statistic"]: row for row in estimates}
    assert values["round_target_actuation_cmi"]["estimate"] == pytest.approx(0.0, abs=1e-9)
    assert values[statistic]["estimate"] == pytest.approx(1.0, abs=1e-9)

    row = values[statistic]
    # Every reporting field the pipeline already produces has to come along.
    assert row["units"] == "bits"
    assert row["estimator_variant"] == "direct_counting"
    assert row["null_type"] == "policy_conditional_randomization"
    assert row["bootstrap_unit"] == "episode"
    assert math.isfinite(row["bootstrap_ci_low"]) and math.isfinite(row["bootstrap_ci_high"])
    assert row["conditional_action_entropy_bits"] >= row["estimate"]
    assert row["entropy_bound_satisfied"] is True
    assert row["round_conditioning_state_count"] == 2
    assert 0.0 <= row["round_singleton_fraction"] <= 1.0
    assert row["round_dual_action_state_fraction"] == pytest.approx(1.0)
    assert row["round_dual_action_event_fraction"] == pytest.approx(1.0)
    assert any(item["statistic"] == statistic for item in nulls)


def test_the_new_statistics_go_through_the_shared_cmi_implementation(monkeypatch):
    """No second estimator: patching the shared one must blank all three."""

    import mas_cc.games.hidden_bench.imitation_round_feedback.analysis as pipeline

    calls: list[int] = []

    def _spy(x, y, z, **kwargs):
        calls.append(len(x))
        return pipeline.Estimate(-1.0, -1.0, -1.0, len(x))

    monkeypatch.setattr(pipeline, "conditional_mutual_information", _spy)
    estimates, _ = round_information_analysis(
        _epistemic_rows(),
        statistics=list(EPISTEMIC_CONDITIONING_STATISTICS),
        bootstrap_resamples=0,
        null_permutations=0,
    )
    assert len(calls) == len(EPISTEMIC_CONDITIONING_STATISTICS)
    assert {row["estimate"] for row in estimates} == {-1.0}


def test_the_matched_signed_response_removes_an_epistemic_confound():
    """What conditioning the signed response on `phi` is actually for.

    The target drifts up in the low-phi regime and down in the high-phi one,
    whatever the controller does, and ADVOCATE happens to be concentrated in
    the first. The unmatched difference reads that drift as a controller
    effect; matched inside a phi bin - the same state the CMI conditions on -
    it correctly reads zero. `n_Z,k` is identical in both regimes, so matching
    on the opinion state alone cannot remove this.
    """

    rows = []
    for regime, (strata, after, advocates) in enumerate(
        (((12, 12, 0), (13, 5, 6), 5), ((2, 2, 20), (15, 5, 4), 1))
    ):
        for index in range(6):
            rows.append(
                adapt_relational_round_record(
                    _record(
                        episode=f"e{regime}",
                        index=index,
                        action=ADVOCATE_TARGET if index < advocates else NO_OP,
                        before=(14, 5, 5),
                        after=after,
                        strata_before=strata,
                    )
                )
            )

    estimates, _ = round_information_analysis(
        rows,
        statistics=[
            "round_target_signed_response_share",
            "round_target_signed_actuation",
            "round_phi_target_signed_response",
        ],
        bootstrap_resamples=0,
        null_permutations=0,
    )
    values = {row["statistic"]: row for row in estimates}
    # E[dp | ADVOCATE] = +4/144 and E[dp | NO_OP] = -4/144, so the unmatched
    # difference reports a spurious +8/144 of pure drift.
    assert values["round_target_signed_response_share"]["estimate"] == pytest.approx(
        8 / 144
    )
    # Matched on `n_Z,k` alone, the confound survives: it is constant at 5.
    assert values["round_target_signed_actuation"]["estimate"] != pytest.approx(
        0.0, abs=1e-9
    )
    assert values["round_phi_target_signed_response"]["estimate"] == pytest.approx(
        0.0, abs=1e-12
    )
    # Named coordinate, not just "dimensionless": this is a target-FRACTION
    # response, and the magnetization response beside it differs by K/(K-1).
    assert (
        values["round_phi_target_signed_response"]["units"]
        == "target_fraction_per_cycle"
    )
    # And it reports the sparsity of the conditioning it was matched on.
    assert values["round_phi_target_signed_response"]["round_conditioning_state_count"] == 2


def test_phi_and_susceptible_agree_when_the_binning_is_a_relabelling():
    """Expected, not an error: CMI is invariant under relabelling of `z`."""

    rows = _epistemic_rows()
    estimates, _ = round_information_analysis(
        rows,
        statistics=[
            "round_phi_target_actuation_cmi",
            "round_susceptible_target_actuation_cmi",
        ],
        bootstrap_resamples=0,
        null_permutations=0,
    )
    values = {row["statistic"]: row["estimate"] for row in estimates}
    assert values["round_phi_target_actuation_cmi"] == pytest.approx(
        values["round_susceptible_target_actuation_cmi"]
    )


def test_the_pre_existing_statistics_are_untouched_by_the_new_conditionings():
    """Adding conditioning variables must not move anything already reported."""

    rows = _epistemic_rows()
    historical = [
        "round_target_actuation_cmi",
        "round_truth_actuation_cmi",
        "round_order_actuation_cmi",
        "round_population_actuation_cmi",
        "round_controller_action_entropy",
        "round_target_signed_actuation",
        "round_memory_target_actuation_cmi",
        "round_epistemic_target_actuation_cmi",
    ]
    alone, _ = round_information_analysis(
        rows, statistics=historical, bootstrap_resamples=8, null_permutations=8, seed=7
    )
    together, _ = round_information_analysis(
        rows,
        statistics=[*historical, *EPISTEMIC_CONDITIONING_STATISTICS],
        bootstrap_resamples=8,
        null_permutations=8,
        seed=7,
    )
    kept = {row["statistic"]: row for row in together if row["statistic"] in historical}
    for row in alone:
        assert kept[row["statistic"]] == row
