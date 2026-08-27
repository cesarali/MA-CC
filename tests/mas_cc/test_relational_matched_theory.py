"""The matched finite-N controlled q-voter reference, and its seam.

Two things are being pinned down here and they are different in kind.

The first is that the exact theory is *exact*: kernels normalise, the
information functional respects its own bound, and the two independent routes
to the `q = 1` response - a closed form and a composition of round kernels -
agree to floating point. Those are checkable without any data, and they are
the reason this reference can be quoted beside an estimate without a
confidence interval of its own.

The second is that plugging it in changed nothing that was already there. The
existing MI/CMI numbers, their bootstrap, their nulls and their support
diagnostics must come out bit-identical with the theory layer switched on,
because the theory is meant to be a second column beside the empirical result,
never an influence on it.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from mas_cc.games.hidden_bench.imitation.controller import ADVOCATE_TARGET, NO_OP
from mas_cc.games.relational_reasoning.imitation_round_feedback.analysis import (
    _controlled,
    _signed_response,
    adapt_relational_round_record,
    analyze_relational_imitation_round_feedback,
    empirical_policy_curve,
    empirical_round_occupancy,
    finite_horizon_occupancy,
    matched_qvoter_comparison,
    matched_qvoter_parameters_for,
    matched_qvoter_state_curves,
)
from mas_cc.games.relational_reasoning.imitation_round_feedback.matched_qvoter import (
    TheoryParameters,
    advocacy_probability_curve,
    advocate_round_kernel,
    binary_entropy_bits,
    classical_reference,
    kernel_mean_response,
    local_transfer_entropy,
    microscopic_kernels,
    q1_mean_response,
    sensor_law,
    theory_parameters_from_record,
)

OPTIONS = ["SOUTHWEST", "WEST", "EAST"]
CORRECT = "WEST"
TARGET = "EAST"

PARAMETERS = [
    TheoryParameters(N=24, q=1, q_c=12, b=12, beta=4.0, theta=0.5),
    TheoryParameters(N=12, q=2, q_c=6, b=3, beta=2.0, theta=0.4),
    TheoryParameters(N=8, q=3, q_c=8, b=8, beta=8.0, theta=0.6),
    TheoryParameters(N=10, q=1, q_c=4, b=0, beta=1.0, theta=0.5),
]


# --------------------------------------------------------------------------
# 1-2. Sensing and policy
# --------------------------------------------------------------------------


@pytest.mark.parametrize("parameters", PARAMETERS)
def test_the_hypergeometric_sensor_is_a_distribution_at_every_state(parameters):
    for n in range(parameters.N + 1):
        law = sensor_law(parameters.N, n, parameters.q_c)
        assert law.sum() == pytest.approx(1.0)
        assert (law >= 0.0).all()


def test_the_sensor_cannot_report_more_on_target_than_exist():
    # Three supporters in a population of ten cannot produce a sample of five
    # containing four of them, and the combinatorics must say so exactly.
    law = sensor_law(10, 3, 5)
    assert law[4] == 0.0
    assert law[5] == 0.0
    assert law[3] > 0.0


@pytest.mark.parametrize("parameters", PARAMETERS)
def test_advocacy_probability_is_a_probability_at_every_state(parameters):
    advocacy = advocacy_probability_curve(parameters)
    assert advocacy.shape == (parameters.N + 1,)
    assert ((advocacy >= 0.0) & (advocacy <= 1.0)).all()


def test_advocacy_falls_as_the_target_spreads():
    # The controller advocates when the target looks under-represented, so a_n
    # must be monotonically decreasing in n. This is what makes the policy
    # comparison a calibration check rather than a fit.
    advocacy = advocacy_probability_curve(PARAMETERS[0])
    assert (np.diff(advocacy) <= 1e-12).all()


def test_a_zero_beta_controller_ignores_what_it_sensed():
    flat = advocacy_probability_curve(
        TheoryParameters(N=10, q=1, q_c=5, b=2, beta=0.0, theta=0.5)
    )
    assert flat == pytest.approx(np.full(11, 0.5))


# --------------------------------------------------------------------------
# 3-5. Kernels
# --------------------------------------------------------------------------


@pytest.mark.parametrize("parameters", PARAMETERS)
def test_microscopic_kernels_are_row_stochastic(parameters):
    K0, K1 = microscopic_kernels(parameters)
    assert K0.sum(axis=1) == pytest.approx(np.ones(parameters.N + 1))
    assert K1.sum(axis=1) == pytest.approx(np.ones(parameters.N + 1))
    assert (K0 >= -1e-15).all()
    assert (K1 >= -1e-15).all()


@pytest.mark.parametrize("parameters", PARAMETERS)
def test_a_controlled_update_never_moves_an_agent_off_the_target(parameters):
    _, K1 = microscopic_kernels(parameters)
    assert np.tril(K1, -1) == pytest.approx(np.zeros_like(K1))


def test_the_controlled_update_needs_one_fewer_ordinary_supporter():
    # K1 differs from K0 in exactly the documented way: the controller occupies
    # one of the q social slots, so q-1 ordinary supporters suffice.
    parameters = TheoryParameters(N=12, q=2, q_c=6, b=3, beta=2.0, theta=0.4)
    K0, K1 = microscopic_kernels(parameters)
    N, q, n = parameters.N, parameters.q, 5
    assert K0[n, n + 1] == pytest.approx(
        (N - n) / N * math.comb(n, q) / math.comb(N - 1, q)
    )
    assert K1[n, n + 1] == pytest.approx(
        (N - n) / N * math.comb(n, q - 1) / math.comb(N - 1, q - 1)
    )
    assert K1[n, n + 1] > K0[n, n + 1]


@pytest.mark.parametrize("parameters", PARAMETERS)
def test_round_kernels_are_row_stochastic(parameters):
    reference = classical_reference(parameters)
    ones = np.ones(parameters.N + 1)
    assert reference.R0.sum(axis=1) == pytest.approx(ones)
    assert reference.R1.sum(axis=1) == pytest.approx(ones)
    assert (reference.R1 >= -1e-12).all()


def test_the_advocate_round_uses_exactly_b_control_not_bernoulli_c():
    """The one modelling choice that a mean-preserving shortcut would hide.

    Drawing `b` positions without replacement and flipping each position
    independently with probability `c = b/N` agree on the expected number of
    controlled positions and disagree on everything else. Since the whole
    quantity being measured is an information flow - a variance-like object -
    the two must not be allowed to be confused, so this pins the difference
    rather than merely asserting the recursion runs.
    """

    parameters = TheoryParameters(N=8, q=2, q_c=4, b=3, beta=3.0, theta=0.5)
    K0, K1 = microscopic_kernels(parameters)
    exact = advocate_round_kernel(K0, K1, N=parameters.N, b=parameters.b)
    c = parameters.actuation_fraction
    bernoulli = np.linalg.matrix_power((1.0 - c) * K0 + c * K1, parameters.N)
    assert exact.sum(axis=1) == pytest.approx(np.ones(parameters.N + 1))
    assert not np.allclose(exact, bernoulli, atol=1e-9)


def test_a_full_budget_advocate_round_is_all_controlled_positions():
    # b = N leaves the recursion no ordinary positions to schedule, so it must
    # collapse to K1^N exactly - a closed-form corner the recursion has to hit.
    parameters = TheoryParameters(N=6, q=2, q_c=3, b=6, beta=2.0, theta=0.5)
    K0, K1 = microscopic_kernels(parameters)
    exact = advocate_round_kernel(K0, K1, N=6, b=6)
    assert exact == pytest.approx(np.linalg.matrix_power(K1, 6))


def test_a_zero_budget_advocate_round_is_an_ordinary_round():
    parameters = TheoryParameters(N=6, q=2, q_c=3, b=0, beta=2.0, theta=0.5)
    reference = classical_reference(parameters)
    assert reference.R1 == pytest.approx(reference.R0)


# --------------------------------------------------------------------------
# 6-9. The information functional
# --------------------------------------------------------------------------


@pytest.mark.parametrize("parameters", PARAMETERS)
def test_local_transfer_entropy_is_non_negative(parameters):
    assert (classical_reference(parameters).local_te >= -1e-12).all()


@pytest.mark.parametrize("parameters", PARAMETERS)
def test_local_transfer_entropy_respects_the_action_entropy_ceiling(parameters):
    reference = classical_reference(parameters)
    assert (reference.local_te <= reference.entropy_ceiling() + 1e-9).all()


def test_a_deterministic_controller_transmits_nothing():
    # a_n -> {0, 1}: the action never varies, so there is no channel to read
    # regardless of how different the two kernels are.
    parameters = TheoryParameters(N=8, q=2, q_c=8, b=4, beta=1e6, theta=0.5)
    reference = classical_reference(parameters)
    saturated = [
        n for n, a in enumerate(reference.advocacy) if a < 1e-12 or a > 1 - 1e-12
    ]
    assert saturated
    for n in saturated:
        assert reference.local_te[n] == pytest.approx(0.0, abs=1e-12)


def test_identical_kernels_transmit_nothing_however_random_the_action():
    # b = 0 makes R1 == R0 while a_n stays strictly interior, so the action is
    # maximally uncertain and still carries no information about the outcome.
    parameters = TheoryParameters(N=10, q=1, q_c=4, b=0, beta=1.0, theta=0.5)
    reference = classical_reference(parameters)
    assert (reference.advocacy > 0.01).all() and (reference.advocacy < 0.99).all()
    assert reference.local_te == pytest.approx(np.zeros(11), abs=1e-12)


# --------------------------------------------------------------------------
# 10. Response
# --------------------------------------------------------------------------


def test_the_q1_closed_form_agrees_with_the_exact_round_kernels():
    """Two independent routes to the same number.

    `q1_mean_response` is a one-line closed form; `kernel_mean_response` runs
    through the hypergeometric sensor, the microscopic kernels, `K0^N` and the
    exactly-`b` recursion. They agree only if the whole round-kernel
    composition is right, which makes this the single most informative check
    in the file.
    """

    for parameters in (
        TheoryParameters(N=24, q=1, q_c=12, b=12, beta=4.0, theta=0.5),
        TheoryParameters(N=10, q=1, q_c=4, b=3, beta=1.0, theta=0.5),
        TheoryParameters(N=7, q=1, q_c=7, b=7, beta=2.0, theta=0.5),
    ):
        reference = classical_reference(parameters)
        closed = np.array(
            [
                q1_mean_response(x, N=parameters.N, b=parameters.b)
                for x in reference.shares
            ]
        )
        assert reference.mean_response == pytest.approx(closed, abs=1e-12)


def test_the_q1_no_op_round_has_no_drift():
    # At q = 1 an ordinary round is a martingale in x, which is what makes the
    # whole ADVOCATE separation attributable to the controlled positions.
    reference = classical_reference(
        TheoryParameters(N=24, q=1, q_c=12, b=12, beta=4.0, theta=0.5)
    )
    assert reference.R0 @ reference.shares == pytest.approx(
        reference.shares, abs=1e-12
    )


def test_the_response_vanishes_where_everyone_already_holds_the_target():
    reference = classical_reference(PARAMETERS[0])
    assert reference.mean_response[-1] == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------
# 11. Occupancy weighting
# --------------------------------------------------------------------------


def test_occupancy_weighted_te_is_the_plain_weighted_sum():
    reference = classical_reference(PARAMETERS[1])
    rng = np.random.default_rng(0)
    occupancy = rng.random(reference.parameters.N + 1)
    occupancy /= occupancy.sum()
    expected = sum(
        float(occupancy[n]) * float(reference.local_te[n])
        for n in range(reference.parameters.N + 1)
    )
    assert reference.occupancy_weighted_te(occupancy) == pytest.approx(expected)


def test_the_finite_horizon_weighting_averages_rounds_not_observations():
    """`(1/K) sum_k P_k`, per section 8.2 - not one pooled histogram.

    The two coincide only when every round index carries the same number of
    episodes. Episodes can stop early, so the distinction is real and the
    round-averaged form is the one the plan specifies.
    """

    rows = [
        _event(episode="a", index=0, action=ADVOCATE_TARGET, before=(0, 20, 4)),
        _event(episode="a", index=1, action=ADVOCATE_TARGET, before=(0, 20, 4)),
        _event(episode="b", index=0, action=NO_OP, before=(0, 16, 8)),
    ]
    occupancy = finite_horizon_occupancy(rows, 24)
    assert occupancy.sum() == pytest.approx(1.0)
    # Round 0 is half at n=4 and half at n=8; round 1 is entirely at n=4.
    assert occupancy[4] == pytest.approx(0.75)
    assert occupancy[8] == pytest.approx(0.25)


def test_the_entropy_ceiling_is_the_binary_entropy():
    assert binary_entropy_bits(0.5) == pytest.approx(1.0)
    assert binary_entropy_bits(0.0) == 0.0
    assert binary_entropy_bits(1.0) == 0.0
    # A saturated controller has no decision entropy to spend, which is the
    # mechanism behind `T_qv = 0` at the saturated states above.
    assert binary_entropy_bits(1e-9) < 1e-6


# --------------------------------------------------------------------------
# The seam into the analysis
# --------------------------------------------------------------------------


def _record(
    *,
    episode: str,
    index: int,
    action: str | None,
    before=(0, 20, 4),
    after=(0, 20, 4),
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
        "analysis_target": TARGET,
        "occupation_counts_before": list(before),
        "occupation_counts_after": list(after),
        "controller_action": action,
        "controller_advocate_probability": probability,
        # The six matched-protocol parameters, exactly as the runtime writes
        # them onto every round record.
        "N": total,
        "social_group_size": 1,
        "sensor_sample_size": 12,
        "intervention_budget": 12,
        "controller_beta": 4.0,
        "controller_threshold": 0.5,
        "sensor_count_vector": [0, 8, 4],
        "sensor_target_share": 4 / 12,
        "controlled_position_count": 12 if action == ADVOCATE_TARGET else 0,
        "knowledge_stratum_counts_before": [12, 12, 0],
        "mean_supporting_fact_coverage_before": 0.25,
        "mean_supporting_fact_coverage": 0.25,
        "full_proof_agent_share_before": 0.0,
        "full_proof_agent_share": 0.0,
        "delta_m_ctrl": (after[2] - before[2]) * 1.5 / total,
        "delta_m_truth": (after[1] - before[1]) * 1.5 / total,
        "delta_m_order": (max(after) - max(before)) * 1.5 / total,
        "delta_H_vote": 0.0,
    }


def _event(**kwargs):
    return adapt_relational_round_record(_record(**kwargs), cell_id="cell")


def test_the_parameters_are_read_off_the_round_record():
    parameters = theory_parameters_from_record(_record(
        episode="a", index=0, action=ADVOCATE_TARGET
    ))
    assert parameters == TheoryParameters(
        N=24, q=1, q_c=12, b=12, beta=4.0, theta=0.5
    )
    assert parameters.sensing_fraction == pytest.approx(0.5)
    assert parameters.actuation_fraction == pytest.approx(0.5)


def test_a_run_without_a_controller_has_no_matched_protocol():
    record = _record(episode="a", index=0, action=None)
    record["controller_beta"] = None
    record["sensor_sample_size"] = None
    assert theory_parameters_from_record(record) is None
    parameters, reason = matched_qvoter_parameters_for(
        [adapt_relational_round_record(record, cell_id="cell")]
    )
    assert parameters is None
    assert "controller" in reason


def test_cells_with_different_protocols_refuse_a_single_reference():
    """Section 2.3: pooling incompatible tuples would invent a process.

    Two cells run at different budgets have two different classical
    references. Averaging them into one would compare the run against a
    controller nobody configured, so the pooled slice must decline instead.
    """

    first = _record(episode="a", index=0, action=ADVOCATE_TARGET)
    second = _record(episode="b", index=0, action=ADVOCATE_TARGET)
    second["intervention_budget"] = 6
    parameters, reason = matched_qvoter_parameters_for(
        [
            adapt_relational_round_record(first, cell_id="cell"),
            adapt_relational_round_record(second, cell_id="cell"),
        ]
    )
    assert parameters is None
    assert "distinct" in reason


def test_the_round_occupancy_keeps_one_distribution_per_round():
    """`P_k`, not a pooled histogram - the classical self-occupancy starts at `P_0`.

    Propagating the classical chain from a pooled average would start it
    somewhere the run never began, which is exactly the occupancy confound the
    secondary quantity exists to expose.
    """

    rows = [
        _event(episode="a", index=0, action=ADVOCATE_TARGET, before=(0, 20, 4)),
        _event(episode="b", index=0, action=NO_OP, before=(0, 16, 8)),
        _event(episode="a", index=1, action=NO_OP, before=(0, 12, 12)),
    ]
    occupancy = empirical_round_occupancy(rows, 24)
    assert occupancy.shape == (2, 25)
    assert occupancy.sum(axis=1) == pytest.approx(np.ones(2))
    assert occupancy[0, 4] == pytest.approx(0.5)
    assert occupancy[0, 8] == pytest.approx(0.5)
    assert occupancy[1, 12] == pytest.approx(1.0)


def test_the_empirical_policy_curve_counts_actions_at_each_state():
    rows = [
        _event(episode="a", index=0, action=ADVOCATE_TARGET, before=(0, 20, 4)),
        _event(episode="a", index=1, action=NO_OP, before=(0, 20, 4)),
        _event(episode="b", index=0, action=ADVOCATE_TARGET, before=(0, 16, 8)),
    ]
    advocate, no_op = empirical_policy_curve(rows, 24)
    assert advocate[4] == 1 and no_op[4] == 1
    assert advocate[8] == 1 and no_op[8] == 0


def test_the_empirical_response_aggregate_is_the_pipelines_signed_response():
    """The theory column is weighted exactly like the estimate beside it.

    `_signed_response` is the repository's definition of "what the action did",
    dual-action states only, weighted by slice size. The matched comparison
    reuses that weighting rather than inventing a second one, and this pins
    the two together so a later change to either is caught.
    """

    rows = [
        _event(episode="a", index=0, action=ADVOCATE_TARGET,
               before=(0, 20, 4), after=(0, 18, 6)),
        _event(episode="a", index=1, action=NO_OP,
               before=(0, 20, 4), after=(0, 20, 4)),
        _event(episode="b", index=0, action=ADVOCATE_TARGET,
               before=(0, 16, 8), after=(0, 15, 9)),
        _event(episode="b", index=1, action=NO_OP,
               before=(0, 16, 8), after=(0, 17, 7)),
    ]
    comparison, _ = matched_qvoter_comparison(
        rows, [], cell_id="cell", bootstrap_resamples=0, confidence=0.95, seed=1
    )
    expected = _signed_response(
        _controlled(rows),
        state=lambda row: row.target_before,
        delta=lambda row: float(row.event["delta_p_ctrl"]),
    )
    assert comparison["delta_mu_empirical"] == pytest.approx(expected)


def test_the_exact_theory_curve_is_never_bootstrapped():
    """Section 16: resampling moves the occupancy, never the local theory.

    The point of the whole construction is that half the comparison carries no
    sampling uncertainty. If an episode resample could change `T_qv(n)`, the
    residual interval would be double-counting noise that is not there.
    """

    rows = [
        _event(episode="a", index=0, action=ADVOCATE_TARGET, before=(0, 20, 4)),
        _event(episode="b", index=0, action=NO_OP, before=(0, 16, 8)),
    ]
    reference = classical_reference(matched_qvoter_parameters_for(rows)[0])
    before = reference.local_te.copy()
    subsets = ([rows[0]], [rows[1]], rows, rows + rows)
    occupancies = {
        tuple(np.round(finite_horizon_occupancy(subset, 24), 12))
        for subset in subsets
    }
    # The occupancy genuinely varies across resamples ...
    assert len(occupancies) > 1
    # ... while the exact local curve does not move at all.
    assert classical_reference(reference.parameters).local_te == pytest.approx(before)


def test_an_open_loop_run_is_marked_non_identifiable_but_still_compared():
    """Section 18: zero TE by construction is not zero behavioral control.

    The older open-loop studies always advocate, so `H(U | n) = 0` and the CMI
    is zero for a reason that has nothing to do with whether the controller
    steered anything. The comparison still runs - the classical response
    reference remains meaningful - but the TE line is flagged.
    """

    rows = [
        _event(episode="a", index=index, action=ADVOCATE_TARGET)
        for index in range(4)
    ]
    comparison, curves = matched_qvoter_comparison(
        rows, [], cell_id="cell", bootstrap_resamples=0, confidence=0.95, seed=1
    )
    assert comparison["theory_applicable"] is True
    assert comparison["te_comparison_identifiable"] is False
    assert comparison["number_of_actions_observed"] == 1
    assert comparison["theory_interpretation"] == "degenerate_no_action_variation"
    # The classical side is unaffected by the empirical degeneracy.
    assert math.isfinite(comparison["theory_te_emp_occ_bits"])
    assert any(math.isfinite(row["delta_mu_theory"]) for row in curves)


def test_the_state_curves_cover_every_state_and_carry_their_support():
    rows = [
        _event(episode="a", index=0, action=ADVOCATE_TARGET, before=(0, 20, 4)),
        _event(episode="a", index=1, action=NO_OP, before=(0, 20, 4)),
    ]
    reference = classical_reference(matched_qvoter_parameters_for(rows)[0])
    curves = matched_qvoter_state_curves(rows, reference, cell_id="cell")
    assert len(curves) == 25
    assert [row["n_target"] for row in curves] == list(range(25))
    visited = next(row for row in curves if row["n_target"] == 4)
    assert visited["observations"] == 2
    assert visited["dual_action_state"] is True
    assert visited["a_n_empirical"] == pytest.approx(0.5)
    # The theory exists at every state; the empirical curve does not, and an
    # unvisited state must say so rather than reporting a zero.
    unvisited = next(row for row in curves if row["n_target"] == 17)
    assert unvisited["observations"] == 0
    assert math.isnan(unvisited["a_n_empirical"])
    assert math.isfinite(unvisited["a_n_theory"])
    assert math.isfinite(unvisited["local_te_theory_bits"])


def test_the_q1_closed_form_column_matches_the_kernel_column():
    rows = [_event(episode="a", index=0, action=ADVOCATE_TARGET)]
    reference = classical_reference(matched_qvoter_parameters_for(rows)[0])
    curves = matched_qvoter_state_curves(rows, reference, cell_id="cell")
    for row in curves:
        assert row["delta_mu_theory_q1_closed_form"] == pytest.approx(
            row["delta_mu_theory"], abs=1e-12
        )


# --------------------------------------------------------------------------
# 13-14. Backward compatibility
# --------------------------------------------------------------------------


def _write_run(tmp_path, *, seed=7):
    rng = np.random.default_rng(seed)
    directory = tmp_path / "cell-0000"
    directory.mkdir(parents=True)
    (directory / "overrides.json").write_text("{}", encoding="utf-8")
    lines = []
    for episode in range(6):
        target = 4
        for index in range(5):
            action = ADVOCATE_TARGET if rng.random() < 0.6 else NO_OP
            moved = int(rng.random() < (0.5 if action == ADVOCATE_TARGET else 0.15))
            after_target = min(24, target + moved)
            lines.append(
                json.dumps(
                    _record(
                        episode=f"e{episode}",
                        index=index,
                        action=action,
                        before=(0, 24 - target, target),
                        after=(0, 24 - after_target, after_target),
                    )
                )
            )
            target = after_target
    (directory / "round_trajectory.jsonl").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return tmp_path


def test_the_existing_mi_outputs_are_identical_with_and_without_theory(tmp_path):
    """Section 19.13: the theory layer is additive, not perturbative.

    The empirical estimates, their bootstrap intervals, their nulls and their
    support diagnostics must be bit-identical whether or not the classical
    reference was computed. If they were not, the theory would be influencing
    the number it exists to be compared against.
    """

    run = _write_run(tmp_path / "run")
    import pandas as pd

    frames = {}
    for enabled in (False, True):
        destination = tmp_path / f"analysis-{enabled}"
        analyze_relational_imitation_round_feedback(
            run,
            destination,
            bootstrap_resamples=25,
            null_permutations=25,
            seed=3,
            theory_comparison_enabled=enabled,
        )
        frames[enabled] = pd.read_csv(destination / "round_information_estimates.csv")
        if enabled:
            assert (destination / "single_affinity_theory_comparison.csv").is_file()
        else:
            assert not (destination / "single_affinity_theory_comparison.csv").exists()
    pd.testing.assert_frame_equal(frames[False], frames[True])


def test_matched_qvoter_is_opt_in_and_uses_a_separate_namespace(tmp_path):
    run = _write_run(tmp_path / "run")
    destination = tmp_path / "analysis"
    summary = analyze_relational_imitation_round_feedback(
        run,
        destination,
        bootstrap_resamples=0,
        null_permutations=0,
        theoretical_reference="matched_qvoter_null",
    )
    assert (destination / "matched_qvoter_null.csv").is_file()
    assert (destination / "matched_qvoter_null_state_curves.csv").is_file()
    assert not (destination / "single_affinity_theory_comparison.csv").exists()
    assert all(
        row["reference"] == "matched_qvoter_classical_null"
        for row in summary["matched_qvoter_classical_null"]
    )


def test_the_analysis_writes_revised_reference_or_unavailable_by_default(tmp_path):
    """The default never substitutes the matched q-voter for revised theory."""

    run = _write_run(tmp_path / "run")
    destination = tmp_path / "analysis"
    summary = analyze_relational_imitation_round_feedback(
        run, destination, bootstrap_resamples=25, null_permutations=25, seed=3
    )

    table = destination / "single_affinity_theory_comparison.csv"
    assert table.is_file()
    assert not (destination / "theory_comparison.csv").exists()
    assert not (destination / "matched_qvoter_null.csv").exists()
    report = (destination / "round_information_estimates.md").read_text()
    assert "# Round feedback information estimates" in report
    frame = pd.read_csv(table)
    assert set(frame["reference"]) == {"single_affinity_revised"}
    assert set(frame["theory_module"]) == {
        "mas_cc.games.relational_reasoning.imitation_round_feedback.theory_revised"
    }
    assert not frame["available"].any()
    assert summary["theoretical_reference"] == "single_affinity_revised"


def test_the_report_declines_rather_than_inventing_a_reference(tmp_path):
    """A run with no controller parameters says why it was skipped."""

    directory = tmp_path / "run" / "cell-0000"
    directory.mkdir(parents=True)
    (directory / "overrides.json").write_text("{}", encoding="utf-8")
    lines = []
    for episode in range(3):
        for index in range(4):
            record = _record(
                episode=f"e{episode}",
                index=index,
                action=ADVOCATE_TARGET if index % 2 else NO_OP,
            )
            record["controller_beta"] = None
            lines.append(json.dumps(record))
    (directory / "round_trajectory.jsonl").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    destination = tmp_path / "analysis"
    summary = analyze_relational_imitation_round_feedback(
        tmp_path / "run",
        destination,
        bootstrap_resamples=10,
        null_permutations=10,
        seed=1,
    )
    entry = summary["single_affinity_theory_comparison"][0]
    assert entry["available"] is False
    assert "identify a finite h" in entry["reason"]
    assert entry["reference"] == "single_affinity_revised"
