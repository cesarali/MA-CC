"""One canonical single-affinity treatment across theory, simulation, estimation.

These tests exist to stop the three layers from drifting apart:

* `theory_revised.py` - exact, deterministic, provider-free;
* a simulator that draws from the *same* controlled kernel the theory states;
* `mas_cc.analysis.single_affinity` - the empirical estimators.

The point is that a quantity called `chi` means the same thing in all three,
in the same units, before anyone compares numbers.  Most of the failures these
guard against are unit slips, not algebra slips: a magnetization response where
a fraction response belongs, bits where nats belong, a factored expectation
where a state-by-state sum belongs.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from mas_cc.analysis import single_affinity as sa
from mas_cc.games.hidden_bench.imitation_round_feedback.analysis import (
    ROUND_SINGLE_AFFINITY_STATISTICS,
    round_information_analysis,
)
from mas_cc.games.relational_reasoning.imitation_round_feedback import (
    theory_revised as theory,
)

ADVOCATE = "ADVOCATE_Z"
NO_OP = "NO_OP"


# ---------------------------------------------------------------------------
# helpers: a simulator that IS the theory's kernel, not a second copy of it
# ---------------------------------------------------------------------------


def _round_event(
    *,
    cell_id="cell",
    episode_id="ep",
    round_index=0,
    action,
    before,
    after,
    N,
    K=3,
    q_c=2,
    sensor_target_count=None,
    protocol=None,
):
    """A `RoundEvent`-shaped double carrying only what the estimators read."""

    others_before = [0] * (K - 1)
    others_after = [0] * (K - 1)
    others_before[0] = N - before
    others_after[0] = N - after
    event = {
        "N": N,
        "K": K,
        "episode_id": episode_id,
        "round_index": round_index,
        "controller_action": action,
        "target_count_before": before,
        "target_count_after": after,
        "delta_p_ctrl": (after - before) / N,
        # aligned magnetization: m = (K p - 1)/(K - 1), so dm = dp * K/(K-1)
        "delta_m_ctrl": (after - before) / N * K / (K - 1),
        "sensor_sample_size": q_c,
        "sensor_target_count": sensor_target_count,
        "possible_answers": [f"opt{i}" for i in range(K)],
        "analysis_target": "opt0",
    }
    if protocol is not None:
        # what `theory_parameters_from_record` needs to rebuild the protocol
        event.update(
            {
                "social_group_size": 1,
                "intervention_budget": protocol.b,
                "controller_beta": protocol.beta,
                "controller_threshold": protocol.theta,
            }
        )
    return SimpleNamespace(
        cell_id=cell_id,
        episode_id=episode_id,
        round_index=round_index,
        event=event,
        U_k=action,
        target_before=before,
        target_after=after,
        N_k=tuple([before, *others_before]),
        N_k1=tuple([after, *others_after]),
        Y_k=None if sensor_target_count is None else (sensor_target_count, 0, 0),
        target_index=0,
        sensor_target_count=sensor_target_count,
    )


def simulate_single_affinity(parameters, *, episodes, rounds, n0, seed=0):
    """Draw trajectories from the theory's own controlled kernel.

    One controlled *opportunity* is: pick a uniformly random agent; a
    non-target agent flips to the target with probability `gamma*sigma(h)`, a
    target agent flips away with probability `gamma*(1-sigma(h))`.  Summed over
    agents that is exactly `K(n+1|n)=gamma (N-n)/N sigma(h)` and
    `K(n-1|n)=gamma n/N (1-sigma(h))`, so the simulator does not restate the
    kernel - it realises it.  ADVOCATE takes `b` such opportunities, NO_OP
    takes none, which is `Q1=K^b` and `Q0=I`.
    """

    rng = np.random.default_rng(seed)
    N, q_c, b = parameters.N, parameters.q_c, parameters.b
    p_h, gamma = parameters.p_h, parameters.gamma
    pi1 = theory.policy_advocacy_vector(q_c, parameters.beta, parameters.theta)
    rounds_out: list[SimpleNamespace] = []
    micro_out: list[dict] = []
    for episode in range(episodes):
        n = int(n0)
        episode_id = f"ep{episode:04d}"
        for k in range(rounds):
            y = int(rng.hypergeometric(n, N - n, q_c))
            advocate = bool(rng.random() < pi1[y])
            before = n
            if advocate:
                for slot in range(b):
                    is_target = rng.random() < n / N
                    if is_target:
                        flipped = rng.random() < gamma * (1.0 - p_h)
                        n -= int(flipped)
                    else:
                        flipped = rng.random() < gamma * p_h
                        n += int(flipped)
                    micro_out.append(
                        {
                            "cell_id": "cell",
                            "episode_id": episode_id,
                            "round_index": k,
                            "micro_slot_index": slot,
                            "controlled_slot": True,
                            "round_controller_action": ADVOCATE,
                            "analysis_target": "opt0",
                            "focal_opinion_before": "opt0" if is_target else "opt1",
                            "focal_opinion_after": (
                                ("opt1" if flipped else "opt0")
                                if is_target
                                else ("opt0" if flipped else "opt1")
                            ),
                        }
                    )
            rounds_out.append(
                _round_event(
                    episode_id=episode_id,
                    round_index=k,
                    action=ADVOCATE if advocate else NO_OP,
                    before=before,
                    after=n,
                    N=N,
                    q_c=q_c,
                    sensor_target_count=y,
                    protocol=parameters,
                )
            )
    return rounds_out, micro_out


def _parameters(**overrides):
    base = dict(N=8, q_c=4, b=6, beta=6.0, theta=0.5, h=1.2, gamma=0.6)
    base.update(overrides)
    return theory.TheoryParameters(**base)


# ---------------------------------------------------------------------------
# 12.1 pure theory invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"N": 6, "q_c": 3, "b": 2, "h": -0.8, "gamma": 0.3},
        {"N": 10, "q_c": 10, "b": 1, "beta": 2.0, "theta": 0.25, "h": 3.0, "gamma": 0.9},
        {"N": 12, "q_c": 2, "b": 12, "beta": 12.0, "theta": 0.75, "h": 0.0, "gamma": 1.0},
    ],
)
def test_theory_invariants_hold_on_a_parameter_grid(overrides):
    parameters = _parameters(**overrides)
    reference = theory.single_affinity_reference(parameters)
    N = parameters.N

    assert np.allclose(reference.S.sum(axis=1), 1.0)
    assert np.allclose(reference.K.sum(axis=1), 1.0)
    assert np.allclose(reference.Q1.sum(axis=1), 1.0)
    assert np.allclose(reference.Q0, np.eye(N + 1))
    assert np.allclose(
        reference.Q1, np.linalg.matrix_power(reference.K, parameters.b)
    )
    # the closed form and the kernel must agree state by state
    assert np.allclose(
        theory.susceptibility_curve(parameters),
        theory.kernel_mean_response(reference.Q0, reference.Q1, N),
        atol=1e-12,
    )
    assert np.all(reference.pinsker_bound <= reference.T_pi + 1e-12)
    assert np.all(reference.T_pi <= reference.entropy_ceiling() + 1e-12)
    finite = np.isfinite(reference.eta_IR)
    assert np.all(reference.eta_IR[finite] >= -1e-12)
    assert np.all(reference.eta_IR[finite] <= 1.0 + 1e-12)


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"N": 6, "q_c": 3, "b": 2, "h": -0.8, "gamma": 0.3},
        {"N": 10, "q_c": 10, "b": 1, "beta": 2.0, "theta": 0.25, "h": 3.0, "gamma": 0.9},
    ],
)
@pytest.mark.parametrize("x0", [0.1, 0.5, 0.9])
def test_one_cycle_closes_the_path_identity_and_bounds_eta_th(overrides, x0):
    parameters = _parameters(**overrides)
    reference = theory.single_affinity_reference(parameters)
    p0 = theory.binomial_ensemble(parameters.N, x0)
    cycle = reference.one_cycle(p0)

    # the decomposition and the directly computed path KL are the same number
    assert cycle.Sigma_direct_KL_nats == pytest.approx(cycle.Sigma_nats, abs=1e-9)
    assert cycle.Sigma_nats == pytest.approx(
        cycle.delta_S_sys_nats
        + parameters.h * cycle.J_c
        + cycle.I_sens_nats,
        abs=1e-9,
    )
    assert cycle.C_th_nats == pytest.approx(
        cycle.Sigma_nats - cycle.delta_S_sys_nats, abs=1e-9
    )
    assert cycle.C_th_nats == pytest.approx(
        parameters.h * cycle.J_c + cycle.I_sens_nats, abs=1e-9
    )
    if cycle.eta_th_has_bounded_interpretation:
        assert 0.0 <= cycle.eta_th <= 1.0


# ---------------------------------------------------------------------------
# 12.2 calibration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("h", [-2.0, -0.5, 0.0, 1.3, 3.5])
@pytest.mark.parametrize("gamma", [0.05, 0.4, 0.95])
def test_calibration_recovers_known_h_and_gamma(h, gamma):
    p_plus = gamma * theory.logistic(h)
    p_minus = gamma * theory.logistic(-h)
    calibration = theory.calibrate_affinity_compliance(p_plus, p_minus)
    assert calibration.h_eff == pytest.approx(h, abs=1e-12)
    assert calibration.gamma_eff == pytest.approx(gamma, abs=1e-12)


def test_reported_study_calibration_example():
    """The worked example in the revised report, to four significant figures."""

    calibration = theory.calibrate_affinity_compliance_from_counts(
        plus_transitions=208, plus_eligible=572,
        minus_transitions=4, minus_eligible=508,
    )
    assert calibration.h_eff == pytest.approx(3.83, abs=0.01)
    assert calibration.gamma_eff == pytest.approx(0.372, abs=0.001)


def test_empirical_and_theory_calibration_agree_on_one_transition_table():
    micro = (
        [
            {
                "controlled_slot": True,
                "round_controller_action": ADVOCATE,
                "analysis_target": "opt0",
                "focal_opinion_before": "opt1",
                "focal_opinion_after": "opt0" if index < 208 else "opt1",
            }
            for index in range(572)
        ]
        + [
            {
                "controlled_slot": True,
                "round_controller_action": ADVOCATE,
                "analysis_target": "opt0",
                "focal_opinion_before": "opt0",
                "focal_opinion_after": "opt1" if index < 4 else "opt0",
            }
            for index in range(508)
        ]
    )
    empirical = sa.affinity_compliance(micro)
    exact = theory.calibrate_affinity_compliance_from_counts(
        plus_transitions=208, plus_eligible=572,
        minus_transitions=4, minus_eligible=508,
    )
    assert empirical["affinity_valid"]
    assert empirical["effective_affinity"] == pytest.approx(exact.h_eff)
    assert empirical["kinetic_compliance"] == pytest.approx(exact.gamma_eff)


def test_a_direction_that_never_fired_is_not_smoothed_into_an_affinity():
    micro = [
        {
            "controlled_slot": True,
            "round_controller_action": ADVOCATE,
            "analysis_target": "opt0",
            "focal_opinion_before": "opt1",
            "focal_opinion_after": "opt0",
        },
        {
            "controlled_slot": True,
            "round_controller_action": ADVOCATE,
            "analysis_target": "opt0",
            "focal_opinion_before": "opt0",
            "focal_opinion_after": "opt0",
        },
    ]
    result = sa.affinity_compliance(micro)
    assert result["affinity_valid"] is False
    assert math.isnan(result["effective_affinity"])
    # the counts stay visible so the reason is auditable
    assert result["minus_eligible"] == 1 and result["minus_transitions"] == 0


# ---------------------------------------------------------------------------
# 12.3 susceptibility units
# ---------------------------------------------------------------------------


def test_susceptibility_is_a_fraction_response_and_magnetization_is_K_over_K_minus_1():
    N, K = 6, 3
    rows = []
    # state n=2: ADVOCATE moves +1, NO_OP moves 0  -> chi_x = 1/6
    # state n=4: ADVOCATE moves +2, NO_OP moves -1 -> chi_x = 3/6
    for index, (before, advocate_after, no_op_after) in enumerate(
        ((2, 3, 2), (4, 6, 3))
    ):
        rows.append(
            _round_event(
                episode_id=f"ep{index}", round_index=0, action=ADVOCATE,
                before=before, after=advocate_after, N=N, K=K,
            )
        )
        rows.append(
            _round_event(
                episode_id=f"ep{index}", round_index=1, action=NO_OP,
                before=before, after=no_op_after, N=N, K=K,
            )
        )
    table = sa.state_response_table(rows)
    assert table[2]["chi"] == pytest.approx(1 / 6)
    assert table[4]["chi"] == pytest.approx(3 / 6)

    estimates, _ = round_information_analysis(
        rows,
        statistics=list(ROUND_SINGLE_AFFINITY_STATISTICS[:1])
        + ["round_target_signed_actuation"],
        bootstrap_resamples=0, null_permutations=0,
    )
    by_metric = {row["statistic"]: row["estimate"] for row in estimates}
    chi_x = by_metric["round_target_susceptibility"]
    chi_m = by_metric["round_target_signed_actuation"]
    # identical rows, identical state matching -> exactly the K/(K-1) factor
    assert chi_m == pytest.approx(K / (K - 1) * chi_x)
    assert chi_m == pytest.approx(1.5 * chi_x)


# ---------------------------------------------------------------------------
# 12.4 eta_ir
# ---------------------------------------------------------------------------


def test_eta_ir_is_an_occupancy_ratio_of_sums_not_a_mean_of_ratios():
    """Two states with very different occupancy must not count equally."""

    N = 4
    rows = []
    index = 0

    def add(before, action, after, copies):
        nonlocal index
        for _ in range(copies):
            rows.append(
                _round_event(
                    episode_id=f"ep{index:04d}", round_index=0, action=action,
                    before=before, after=after, N=N,
                )
            )
            index += 1

    add(1, ADVOCATE, 2, 40)
    add(1, NO_OP, 1, 40)
    add(3, ADVOCATE, 3, 1)
    add(3, NO_OP, 2, 1)

    result = sa.eta_ir(rows)
    occupancy = sa.pooled_occupancy(rows)
    table = sa.state_response_table(rows)
    expected = sum(
        occupancy[state]
        * 2.0
        * table[state]["a"]
        * (1.0 - table[state]["a"])
        * table[state]["chi"] ** 2
        / math.log(2.0)
        for state in table
    )
    assert result["eta_ir_pinsker_numerator_bits"] == pytest.approx(expected)
    assert result["eta_ir"] == pytest.approx(
        expected / result["eta_ir_denominator_T_bits"]
    )
    assert result["eta_ir_valid"]
    assert result["eta_ir_identified_occupancy_mass"] == pytest.approx(1.0)


def test_eta_ir_numerator_uses_fraction_units_so_magnetization_would_inflate_it():
    parameters = _parameters(N=6, q_c=3, b=4)
    rows, _ = simulate_single_affinity(
        parameters, episodes=400, rounds=4, n0=2, seed=11
    )
    result = sa.eta_ir(rows)
    table = sa.state_response_table(rows)
    occupancy = sa.pooled_occupancy(rows)
    K = 3
    inflated = sum(
        occupancy[state]
        * 2.0
        * table[state]["a"]
        * (1.0 - table[state]["a"])
        * (table[state]["chi"] * K / (K - 1)) ** 2
        / math.log(2.0)
        for state in table
        if table[state]["identified"]
    )
    assert inflated == pytest.approx(
        (K / (K - 1)) ** 2 * result["eta_ir_pinsker_numerator_bits"]
    )
    assert result["eta_ir_pinsker_numerator_bits"] < inflated


def test_eta_ir_never_exceeds_one_on_data_drawn_from_the_theory_kernel():
    parameters = _parameters(N=6, q_c=3, b=4)
    rows, _ = simulate_single_affinity(
        parameters, episodes=800, rounds=5, n0=2, seed=5
    )
    result = sa.eta_ir(rows)
    assert result["eta_ir_valid"]
    assert 0.0 <= result["eta_ir"] <= 1.0 + 1e-9


def test_empirical_state_susceptibility_converges_to_the_exact_chi():
    parameters = _parameters(N=6, q_c=3, b=4)
    exact = theory.susceptibility_curve(parameters)
    rows, _ = simulate_single_affinity(
        parameters, episodes=4000, rounds=4, n0=3, seed=7
    )
    table = sa.state_response_table(rows)
    checked = 0
    for state, item in table.items():
        if not item["identified"] or item["observations"] < 400:
            continue
        assert item["chi"] == pytest.approx(exact[state], abs=0.04)
        checked += 1
    assert checked >= 2


# ---------------------------------------------------------------------------
# 12.5 sensing
# ---------------------------------------------------------------------------


def test_target_sensing_information_matches_the_exact_channel_on_the_same_occupancy():
    parameters = _parameters(N=6, q_c=3, b=4)
    rows, _ = simulate_single_affinity(
        parameters, episodes=300, rounds=3, n0=2, seed=3
    )
    result = sa.target_sensing_information(rows)
    assert result["target_sensing_valid"]

    S = theory.sensor_kernel(parameters.N, parameters.q_c)
    expected = 0.0
    for occupancy in sa.round_occupancy(rows).values():
        vector = np.zeros(parameters.N + 1)
        for n, weight in occupancy.items():
            vector[n] = weight
        value, _ = theory.sensing_information_nats(vector / vector.sum(), S)
        expected += value
    assert result["target_sensing_information_horizon_nats"] == pytest.approx(expected)
    assert result["target_sensing_information_nats"] == pytest.approx(
        expected / result["target_sensing_information_rounds"]
    )


def test_horizon_sensing_is_a_sum_over_rounds_not_a_pooled_stationary_estimate():
    parameters = _parameters(N=6, q_c=3, b=4)
    rows, _ = simulate_single_affinity(
        parameters, episodes=300, rounds=4, n0=1, seed=9
    )
    result = sa.target_sensing_information(rows)
    assert result["target_sensing_information_rounds"] == 4

    pooled = sa.pooled_occupancy(rows)
    vector = np.zeros(parameters.N + 1)
    for n, weight in pooled.items():
        vector[n] = weight
    S = theory.sensor_kernel(parameters.N, parameters.q_c)
    pooled_value, _ = theory.sensing_information_nats(vector, S)
    # a transient run does not have one stationary occupancy, so 4x the pooled
    # value is a different number from the honest round-by-round sum
    assert result["target_sensing_information_horizon_nats"] != pytest.approx(
        4 * pooled_value, rel=1e-6
    )


def test_direct_counting_target_sensing_mi_is_a_separate_named_statistic():
    parameters = _parameters(N=6, q_c=3, b=4)
    rows, _ = simulate_single_affinity(
        parameters, episodes=600, rounds=3, n0=2, seed=13
    )
    estimates, _ = round_information_analysis(
        rows,
        statistics=["round_target_sensing_mi"],
        bootstrap_resamples=0, null_permutations=0,
    )
    row = estimates[0]
    assert row["statistic"] == "round_target_sensing_mi"
    assert row["units"] == "bits"
    # bits from counting vs nats from the exact channel: same channel, and the
    # counting estimate converges from above at these sample sizes
    exact_nats = sa.target_sensing_information(rows)
    assert row["estimate"] / math.log2(math.e) == pytest.approx(
        exact_nats["target_sensing_information_nats"], abs=0.15
    )


# ---------------------------------------------------------------------------
# 12.6 controlled current
# ---------------------------------------------------------------------------


def test_controlled_current_is_N_sum_p_a_chi():
    N = 4
    rows = []
    index = 0

    def add(before, action, after, copies):
        nonlocal index
        for _ in range(copies):
            rows.append(
                _round_event(
                    episode_id=f"ep{index:04d}", round_index=0, action=action,
                    before=before, after=after, N=N,
                )
            )
            index += 1

    add(1, ADVOCATE, 2, 30)
    add(1, NO_OP, 1, 10)
    add(3, ADVOCATE, 3, 10)
    add(3, NO_OP, 2, 30)

    table = sa.state_response_table(rows)
    occupancy = sa.pooled_occupancy(rows)
    expected = N * sum(
        occupancy[state] * table[state]["a"] * table[state]["chi"] for state in table
    )
    result = sa.controlled_current(rows)
    assert result["controlled_current_horizon"] == pytest.approx(expected)
    assert result["controlled_current"] == pytest.approx(expected)


def test_controlled_current_does_not_factor_the_expectation():
    """`a_n` and `chi_n` are deliberately anti-correlated across the two states.

    `N * mean(a) * mean(chi)` is then a materially different number, so this
    would fail loudly if the implementation ever factored the expectation.
    """

    N = 4
    rows = []
    index = 0

    def add(before, action, after, copies):
        nonlocal index
        for _ in range(copies):
            rows.append(
                _round_event(
                    episode_id=f"ep{index:04d}", round_index=0, action=action,
                    before=before, after=after, N=N,
                )
            )
            index += 1

    # state 1: a = 0.9, chi = +1/4   state 3: a = 0.1, chi = -1/4
    add(1, ADVOCATE, 2, 45)
    add(1, NO_OP, 1, 5)
    add(3, ADVOCATE, 2, 5)
    add(3, NO_OP, 3, 45)

    table = sa.state_response_table(rows)
    occupancy = sa.pooled_occupancy(rows)
    exact = N * sum(
        occupancy[state] * table[state]["a"] * table[state]["chi"] for state in table
    )
    factored = (
        N
        * sum(occupancy[state] * table[state]["a"] for state in table)
        * sum(occupancy[state] * table[state]["chi"] for state in table)
    )
    assert exact != pytest.approx(factored)
    assert sa.controlled_current(rows)["controlled_current_horizon"] == pytest.approx(
        exact
    )


def test_controlled_current_converges_to_the_exact_theory_current():
    parameters = _parameters(N=6, q_c=3, b=4)
    reference = theory.single_affinity_reference(parameters)
    rows, _ = simulate_single_affinity(
        parameters, episodes=3000, rounds=3, n0=2, seed=17
    )
    result = sa.controlled_current(rows)
    expected = 0.0
    for occupancy in sa.round_occupancy(rows).values():
        vector = np.zeros(parameters.N + 1)
        for n, weight in occupancy.items():
            vector[n] = weight
        expected += reference.current(vector / vector.sum())
    assert result["controlled_current_horizon"] == pytest.approx(expected, abs=0.25)


def test_cell_current_is_not_the_thermodynamic_current():
    """The terminal episode difference mixes in everything the controller did not do."""

    parameters = _parameters(N=6, q_c=3, b=4)
    rows, _ = simulate_single_affinity(
        parameters, episodes=200, rounds=4, n0=1, seed=23
    )
    by_episode: dict[str, list] = {}
    for row in rows:
        by_episode.setdefault(row.episode_id, []).append(row)
    terminal = float(
        np.mean(
            [
                sorted(group, key=lambda item: item.round_index)[-1].target_after
                - sorted(group, key=lambda item: item.round_index)[0].target_before
                for group in by_episode.values()
            ]
        )
    )
    response_based = sa.controlled_current(rows)["controlled_current_horizon"]
    # both are finite and meaningful, but they are different observables
    assert math.isfinite(terminal) and math.isfinite(response_based)
    assert terminal != pytest.approx(response_based, rel=1e-6)


# ---------------------------------------------------------------------------
# 12.7 end-to-end eta_th against the exact theory
# ---------------------------------------------------------------------------


def test_eta_th_end_to_end_recovers_the_exact_single_affinity_values():
    parameters = _parameters(N=6, q_c=3, b=5, h=1.5, gamma=0.5)
    reference = theory.single_affinity_reference(parameters)
    rows, micro = simulate_single_affinity(
        parameters, episodes=3000, rounds=3, n0=2, seed=29
    )
    estimate = sa.point_estimate(rows, micro)

    assert estimate["affinity_valid"]
    assert estimate["effective_affinity"] == pytest.approx(parameters.h, abs=0.08)
    assert estimate["kinetic_compliance"] == pytest.approx(parameters.gamma, abs=0.03)

    # exact theory evaluated on the SAME empirical occupancies the estimator saw
    exact_current = 0.0
    exact_sensing = 0.0
    S = theory.sensor_kernel(parameters.N, parameters.q_c)
    for occupancy in sa.round_occupancy(rows).values():
        vector = np.zeros(parameters.N + 1)
        for n, weight in occupancy.items():
            vector[n] = weight
        vector /= vector.sum()
        exact_current += reference.current(vector)
        exact_sensing += theory.sensing_information_nats(vector, S)[0]

    assert estimate["controlled_current_horizon"] == pytest.approx(
        exact_current, abs=0.25
    )
    assert estimate["target_sensing_information_horizon_nats"] == pytest.approx(
        exact_sensing, abs=1e-9
    )

    exact_eta, exact_C, bounded = theory.thermodynamic_efficiency(
        h=parameters.h, J_c=exact_current, I_sens_nats=exact_sensing
    )
    assert bounded and estimate["eta_th_valid"]
    assert estimate["thermodynamic_control_expenditure_nats"] == pytest.approx(
        exact_C, abs=0.5
    )
    assert estimate["eta_th"] == pytest.approx(exact_eta, abs=0.05)
    assert 0.0 <= estimate["eta_th"] <= 1.0


def test_eta_th_is_a_ratio_of_accumulated_terms_and_flags_invalid_cases():
    ratio_of_sums = sa.eta_th_from_components(
        h=2.0, current_horizon=3.0, sensing_horizon=1.0
    )
    assert ratio_of_sums["affinity_weighted_current_nats"] == pytest.approx(6.0)
    assert ratio_of_sums["thermodynamic_control_expenditure_nats"] == pytest.approx(7.0)
    assert ratio_of_sums["eta_th"] == pytest.approx(6.0 / 7.0)
    assert ratio_of_sums["eta_th_valid"]

    # controller pushing against its own affinity: signed, not clipped
    against = sa.eta_th_from_components(
        h=2.0, current_horizon=-3.0, sensing_horizon=1.0
    )
    assert against["affinity_weighted_current_nats"] == pytest.approx(-6.0)
    assert against["eta_th_target_directed"] is False
    assert against["eta_th_valid"] is False
    assert math.isnan(against["eta_th"])
    assert against["eta_th_signed"] == pytest.approx(-6.0 / -5.0)


def test_eta_th_is_undefined_without_an_identified_affinity():
    parameters = _parameters(N=6, q_c=3, b=4)
    rows, _ = simulate_single_affinity(
        parameters, episodes=100, rounds=3, n0=2, seed=31
    )
    estimate = sa.point_estimate(rows, ())
    assert estimate["affinity_valid"] is False
    assert math.isnan(estimate["eta_th"])
    assert estimate["eta_th_valid"] is False
    # the current itself is still identified and reported
    assert estimate["controlled_current_valid"]


# ---------------------------------------------------------------------------
# 12.8 bootstrap
# ---------------------------------------------------------------------------


def test_bootstrap_resamples_whole_episodes_and_is_deterministic():
    parameters = _parameters(N=6, q_c=3, b=4)
    rows, micro = simulate_single_affinity(
        parameters, episodes=60, rounds=3, n0=2, seed=37
    )
    first = sa.single_affinity_analysis(
        rows, micro, bootstrap_resamples=64, confidence=0.9, seed=101
    )
    second = sa.single_affinity_analysis(
        rows, micro, bootstrap_resamples=64, confidence=0.9, seed=101
    )
    different = sa.single_affinity_analysis(
        rows, micro, bootstrap_resamples=64, confidence=0.9, seed=202
    )
    for name in ("eta_ir", "eta_th", "controlled_current_horizon"):
        assert first[f"{name}_ci_low"] == second[f"{name}_ci_low"]
        assert first[f"{name}_ci_high"] == second[f"{name}_ci_high"]
        assert first[f"{name}_ci_low"] <= first[name] <= first[f"{name}_ci_high"]
    assert first["eta_ir_ci_low"] != different["eta_ir_ci_low"]


def test_bootstrap_keeps_a_round_with_its_own_micro_slots():
    """Resampling rounds independently of their slots would break `h`."""

    parameters = _parameters(N=6, q_c=3, b=4)
    rows, micro = simulate_single_affinity(
        parameters, episodes=40, rounds=3, n0=2, seed=41
    )
    keys = {sa.episode_key(row) for row in rows}
    micro_keys = {(str(row["cell_id"]), str(row["episode_id"])) for row in micro}
    assert micro_keys <= keys


def test_provenance_travels_with_every_estimate():
    parameters = _parameters(N=6, q_c=3, b=4)
    rows, micro = simulate_single_affinity(
        parameters, episodes=20, rounds=2, n0=2, seed=43
    )
    result = sa.single_affinity_analysis(rows, micro, bootstrap_resamples=0)
    assert result["theory_semantics_version"] == "single_affinity_v1"
    assert result["response_coordinate"] == "target_fraction"
    assert result["sensing_log_base"] == "e"
    assert result["actuation_information_log_base"] == "2"
    assert result["eta_th_aggregation"] == "finite_horizon_ratio_of_sums"
    assert result["bootstrap_unit"] == "episode"


# ---------------------------------------------------------------------------
# 13 acceptance: the side-by-side comparison table
# ---------------------------------------------------------------------------


def test_theory_comparison_puts_all_six_quantities_side_by_side():
    parameters = _parameters(N=6, q_c=3, b=5, h=1.5, gamma=0.5)
    rows, micro = simulate_single_affinity(
        parameters, episodes=2000, rounds=3, n0=2, seed=47
    )
    comparison = sa.theory_comparison(rows, micro)
    assert comparison["available"], comparison.get("reason")
    by_quantity = {row["quantity"]: row for row in comparison["rows"]}
    assert set(by_quantity) == {"chi", "T_pi", "eta_IR", "J_c", "I_sens", "eta_th"}
    assert by_quantity["chi"]["units"] == "target_fraction_per_cycle"
    assert by_quantity["T_pi"]["units"] == "bits"
    assert by_quantity["I_sens"]["units"] == "nats_per_horizon"
    assert all(
        row["reference"] == "single_affinity_revised"
        for row in comparison["rows"]
    )

    # data drawn from the kernel itself must land close to the exact reference
    for name, tolerance in (
        ("chi", 0.03),
        ("J_c", 0.3),
        ("I_sens", 1e-9),
        ("eta_th", 0.05),
        ("eta_IR", 0.15),
    ):
        row = by_quantity[name]
        assert row["empirical"] == pytest.approx(
            row["single_affinity_theory"], abs=tolerance
        ), name
        assert row["residual"] == pytest.approx(
            row["empirical"] - row["single_affinity_theory"]
        )


def test_theory_comparison_refuses_rather_than_inventing_a_reference():
    parameters = _parameters(N=6, q_c=3, b=4)
    rows, _ = simulate_single_affinity(
        parameters, episodes=50, rounds=2, n0=2, seed=53
    )
    # no controlled micro-slots -> no identified h -> no theory column
    refusal = sa.theory_comparison(rows, ())
    assert refusal["available"] is False
    assert "finite h" in refusal["reason"]


def test_the_derived_chi_summary_equals_the_primary_state_matched_estimator():
    """One response, two tables, one number.

    The primary `round_target_susceptibility` weights each identified state by
    its own sample size; the derived summary weights by occupancy renormalised
    over the identified states.  On controlled rows those are the same weights,
    so the study's primary table and its derived table cannot disagree about
    how strongly the controller moved the population.
    """

    parameters = _parameters(N=6, q_c=3, b=4)
    rows, _ = simulate_single_affinity(
        parameters, episodes=500, rounds=3, n0=2, seed=59
    )
    estimates, _ = round_information_analysis(
        rows,
        statistics=["round_target_susceptibility"],
        bootstrap_resamples=0, null_permutations=0,
    )
    primary = estimates[0]["estimate"]
    derived = sa.susceptibility_summary(rows)["susceptibility_occupancy_weighted"]
    assert derived == pytest.approx(primary)
