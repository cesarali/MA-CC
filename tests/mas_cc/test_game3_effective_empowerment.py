"""The acceptance checks from the Game 3 empowerment extension.

Reference: Song et al., *Estimating the Empowerment of Language Model Agents*,
arXiv:2509.22504, and `docs/tdd/architecture/06082026_game3_empowerment_extension.md`.

Most of these are structural: a closed form that must agree with a brute-force
enumeration to machine precision, or a quantity that must be *identically* zero.
Those are the load-bearing ones - a value 5% off is a judgement call, an exact
zero that is not zero is not.
"""

from __future__ import annotations

import numpy as np
import pytest

from mas_cc.config import GameConfig
from mas_cc.games.synthetic import exact
from mas_cc.games.synthetic.controlled_markov.game import (
    SyntheticControlledMarkovGame,
    control_kernels,
)
from mas_cc.games.synthetic.effective_empowerment import (
    EPISODE_FIXED,
    PER_ROUND,
    EmpowermentSpec,
    analyse,
)
from mas_cc.games.synthetic.markov.game import initial_distribution

EXACT = 1e-12


def controlled_config(**options) -> GameConfig:
    return GameConfig(
        type="synthetic_controlled_markov",
        population_size=options.pop("population_size", 6),
        horizon=options.pop("horizon", 60),
        options={
            "actions": ["Q", "M"],
            "coupling": "ring",
            "epsilon": 0.05,
            "controlled_agents": [0, 1, 2],
            **options,
        },
    )


def report_for(config: GameConfig):
    return SyntheticControlledMarkovGame().effective_empowerment(config)


# -- check 1: the KL form against a brute-force joint ----------------------


@pytest.mark.parametrize("horizon", [1, 2, 3, 5])
def test_the_kl_closed_form_agrees_with_a_brute_force_joint(horizon):
    """``E_h(s)`` averaged over ``d`` must be the CMI of ``p(u, s, s')`` exactly.

    The KL form of the document's section 2 and the enumerated joint are two
    different routes to ``I(U_t; S_{t+h} | S_t)``. They are allowed to disagree
    only at floating-point noise; anything larger means the policy weighting or
    the kernel composition is wrong, and every number downstream inherits it.
    """

    game = SyntheticControlledMarkovGame()
    config = controlled_config(control_strength=0.3)
    rules = game.rules(config)
    kernels = control_kernels(rules, game.control(config))
    design = game.control_policy(config)
    report, _ = report_for(config)

    averaged = exact.policy_averaged_kernel(kernels, design)
    tail = np.linalg.matrix_power(averaged, horizon - 1)
    # p(u, s, s') = pi(u) d(s) [T_u T-bar^{h-1}]_{ss'}, axes (X=U, Z=S_t, Y=S_{t+h}).
    joint = design[:, None, None] * report.state_distribution[None, :, None] * (kernels @ tail)
    brute_force = exact.conditional_mutual_information(joint)

    assert report.microstate_curve[horizon - 1] == pytest.approx(brute_force, abs=1e-12)


# -- check 2: the marginalisation identity --------------------------------


def test_marginalising_the_action_recovers_the_policy_averaged_kernel():
    """``sum_u pi(u|s) [T_u T-bar^{h-1}] = [T-bar^h]``, numerically.

    Cheap, and it catches kernel-construction errors immediately: if the family
    of kernels does not mix back to the kernel the state distribution was
    propagated under, the KL divergence is being taken against the wrong
    reference and the empowerment is not a mutual information at all.
    """

    report, _ = report_for(controlled_config(control_strength=0.4))
    assert report.marginalisation_error < 1e-12

    truth = SyntheticControlledMarkovGame().ground_truth(
        controlled_config(control_strength=0.4)
    )
    assert truth.value("empowerment_marginalisation_error") < 1e-12


def test_the_policy_average_is_a_mixture_and_not_a_product_kernel():
    """Why the mixture has to be built explicitly rather than per agent.

    A single ``u`` pushes every controlled agent toward the *same* target, so
    averaging over ``u`` correlates them. Substituting the mean target into the
    per-agent push probability gives a product kernel, which is a different and
    wrong chain - and the difference is exactly the correlation the controller
    induces.
    """

    from mas_cc.games.synthetic.markov.game import transition_matrix

    game = SyntheticControlledMarkovGame()
    config = controlled_config(control_strength=0.6)
    rules = game.rules(config)
    kernels = control_kernels(rules, game.control(config))
    mixture = exact.policy_averaged_kernel(kernels, game.control_policy(config))

    # The tempting shortcut: one kernel with each controlled agent pushed toward
    # the mean target, 0.5. Linear in the per-agent formula, but not in the joint.
    product = transition_matrix(
        rules,
        tuple((0.6, 0.5) if index in (0, 1, 2) else None for index in range(6)),
    )
    assert np.abs(mixture - product).max() > 1e-3


# -- check 3: the episode-fixed decay -------------------------------------


@pytest.mark.parametrize("horizon", [1, 2, 5, 10])
def test_the_episode_fixed_conditional_mutual_information_decays_in_the_round(horizon):
    """``I^3a_h(t)`` must decrease in ``t``, and that decay is the contamination.

    With ``u`` drawn once and held, the state at round ``t`` is causally
    downstream of it, so conditioning on ``S_t`` blocks more and more of the
    path as the episode runs. A flat curve would mean the kernel is not being
    applied from round 0, which is the failure this check exists to catch.
    """

    report, _ = report_for(controlled_config(control_strength=0.3))
    curve = report.episode_fixed[report.episode_fixed_horizons.index(horizon)]

    assert report.episode_fixed_is_monotone(horizon)
    assert curve[0] > curve[-1], "the decay must be real, not a rounding effect"


def test_the_gap_between_the_two_variants_is_reported_as_a_number():
    """``Delta_h(t) = E_h^{3b} - I^3a_h(t)``, and it grows as the episode runs.

    This is the whole point of keeping both variants: the difference between
    the paper's quantity and the episode-fixed design stops being an argument.
    """

    report, _ = report_for(controlled_config(control_strength=0.3))
    gap = report.gap(1)
    assert gap[0] > 0.0
    assert gap[-1] > gap[0]

    truth = SyntheticControlledMarkovGame().ground_truth(
        controlled_config(control_strength=0.3)
    )
    first_round = str(report.episode_fixed_rounds[0])
    assert truth.value("empowerment_gap", ("1", first_round)) == pytest.approx(
        gap[0], abs=EXACT
    )


# -- check 4: the null, no control input ----------------------------------


def test_an_alphabet_that_never_pushes_has_exactly_zero_empowerment():
    """Game 1's situation, expressed in the machinery Game 3 has.

    The document's null is "no control input, so E = 0 exactly at every
    horizon". A control alphabet whose every entry is the do-nothing value is
    that game: the kernels are identical, so every KL divergence is against
    itself. Structural, not small.
    """

    truth = SyntheticControlledMarkovGame().ground_truth(
        controlled_config(control_alphabet=[None, None], control_strength=0.8)
    )
    assert truth.value("effective_empowerment") == pytest.approx(0.0, abs=EXACT)
    horizons = int(truth.value("effective_empowerment_horizon"))
    assert all(
        truth.value("empowerment_at_horizon", (str(h),)) == pytest.approx(0.0, abs=EXACT)
        for h in range(1, horizons + 1)
    )


# -- check 5: the absorbing-chain trap ------------------------------------


def test_the_stationary_distribution_is_the_trap_the_document_warns_about():
    """``stationary`` and ``visitation`` must give different answers.

    A noiseless ring with a push toward Q and a do-nothing alternative is
    absorbing at all-Q, where the two kernels agree and empowerment is exactly
    zero. Averaging over the stationary law therefore reports ``E ~ 0`` - true,
    trivial, and about the wrong states. If the two ever agree on this game,
    the visitation distribution is being computed wrong.
    """

    config = controlled_config(
        epsilon=0.0, control_alphabet=["Q", None], control_strength=0.2, horizon=40
    )
    truth = SyntheticControlledMarkovGame().ground_truth(config)

    visitation = truth.value("effective_empowerment")
    stationary = truth.value("stationary_effective_empowerment")
    assert stationary == pytest.approx(0.0, abs=1e-9)
    assert visitation > 1e-3
    assert visitation != pytest.approx(stationary, abs=1e-6)


def test_the_declared_state_distribution_changes_the_answer():
    """Which average was taken is a config parameter because it moves the number."""

    values = {}
    for kind in ("visitation", "discounted_visitation", "stationary"):
        config = controlled_config(
            epsilon=0.0,
            control_alphabet=["Q", None],
            control_strength=0.2,
            horizon=40,
            empowerment={"state_distribution": kind},
        )
        values[kind] = SyntheticControlledMarkovGame().ground_truth(config).value(
            "effective_empowerment"
        )
    assert len(set(round(value, 9) for value in values.values())) == 3


# -- check 6: truncation --------------------------------------------------


@pytest.mark.parametrize("tolerance", [1e-2, 1e-3, 1e-4])
def test_the_logged_truncation_residual_sits_below_the_declared_tolerance(tolerance):
    """``H`` comes from the tolerance, and ``gamma^H`` is reported rather than assumed."""

    config = controlled_config(
        control_strength=0.3, empowerment={"horizon_tolerance": tolerance}
    )
    truth = SyntheticControlledMarkovGame().ground_truth(config)
    residual = truth.value("effective_empowerment_truncation_residual")
    horizon = truth.value("effective_empowerment_horizon")

    assert residual <= tolerance
    assert residual == pytest.approx(0.9**horizon, abs=EXACT)


def test_the_curve_is_the_primary_output_and_the_scalar_follows_from_it():
    """``E = sum_h (1-gamma) gamma^{h-1} E_h``, recomputed from the reported curve.

    The scalar is recoverable from the curve; the curve is not recoverable from
    the scalar, which is why both are shipped and the curve is the one called
    primary.
    """

    report, spec = report_for(controlled_config(control_strength=0.3))
    assert report.scalar == pytest.approx(
        float(spec.discount_weights @ report.microstate_curve), abs=EXACT
    )


# -- check 7: the zero-control limit --------------------------------------


def test_zero_control_strength_gives_an_identically_zero_empowerment():
    """``control_strength: 0`` makes ``T_u = T-bar`` for every u, so every KL is zero."""

    truth = SyntheticControlledMarkovGame().ground_truth(
        controlled_config(control_strength=0.0)
    )
    assert truth.value("effective_empowerment") == pytest.approx(0.0, abs=EXACT)
    assert truth.value("effective_empowerment_macrostate") == pytest.approx(0.0, abs=EXACT)
    assert truth.value("empowerment_gap", ("1", "1")) == pytest.approx(0.0, abs=EXACT)


def test_effective_empowerment_increases_with_control_strength():
    """The positive control: more authority is a larger number, monotonically."""

    values = [
        SyntheticControlledMarkovGame()
        .ground_truth(controlled_config(control_strength=strength))
        .value("effective_empowerment")
        for strength in (0.0, 0.1, 0.25, 0.5, 0.8)
    ]
    assert values == sorted(values)
    assert values[-1] > values[0]


# -- the per-state profile and the coarse-graining gap --------------------


def test_the_macrostate_profile_reconstructs_the_scalar_it_was_lumped_from():
    """The profile is a decomposition of the average, not a separate quantity.

    ``sum_m d(m) E(m) = E`` by construction, so a profile that does not sum back
    is a lumping bug. Reporting the profile is free - the terms are already
    computed - and it is what makes the average interpretable.
    """

    report, _ = report_for(controlled_config(control_strength=0.3))
    reconstructed = report.macrostate_occupancy @ report.discounted_macrostate_profile
    assert reconstructed == pytest.approx(report.scalar, abs=1e-10)


def test_coarse_graining_is_measured_rather_than_bounded():
    """Both representations on the same run, with no inequality assumed.

    There is no general ordering between ``I(U;S'|S)`` and ``I(U;M'|M)``:
    coarse-graining the *conditioning* variable can raise a conditional mutual
    information as easily as lower it. So this asserts that both are computed
    and that the gap is their difference - not that the gap has a sign.
    """

    truth = SyntheticControlledMarkovGame().ground_truth(
        controlled_config(control_strength=0.3)
    )
    micro = truth.value("effective_empowerment")
    macro = truth.value("effective_empowerment_macrostate")
    assert truth.value("coarse_graining_gap") == pytest.approx(micro - macro, abs=EXACT)
    assert macro > 0.0


def test_dropping_the_macrostate_representation_drops_its_quantities():
    """``state_representation`` is a declared choice, so it has to have an effect."""

    config = controlled_config(
        control_strength=0.3, empowerment={"state_representation": ["microstate"]}
    )
    truth = SyntheticControlledMarkovGame().ground_truth(config)
    assert truth.value("effective_empowerment") > 0.0
    with pytest.raises(KeyError):
        truth.value("effective_empowerment_macrostate")


# -- the control mode, in the simulation ----------------------------------


def test_the_control_mode_is_not_only_a_label_on_the_closed_forms():
    """``per_round`` has to change the dynamics, or the config knob is a lie."""

    game = SyntheticControlledMarkovGame()
    fixed = game.simulate(controlled_config(control_strength=0.5), (5, 7))
    resampled = game.simulate(
        controlled_config(control_strength=0.5, control_mode=PER_ROUND), (5, 7)
    )
    assert not np.array_equal(fixed.actions, resampled.actions)
    # u alternating between the two targets leaves the population near an even
    # split; u pinned to one of them drags it toward that target.
    assert abs(resampled.actions.mean() - 0.5) < abs(fixed.actions.mean() - 0.5)


def test_episode_fixed_is_the_default_so_older_configs_replay_unchanged():
    """The mode was added after these configs were written; they must not move."""

    game = SyntheticControlledMarkovGame()
    assert game.empowerment_spec(controlled_config()).control_mode == EPISODE_FIXED
    explicit = game.simulate(
        controlled_config(control_strength=0.5, control_mode=EPISODE_FIXED), (5,)
    )
    implicit = game.simulate(controlled_config(control_strength=0.5), (5,))
    assert np.array_equal(explicit.actions, implicit.actions)


def test_per_round_control_uses_the_mixture_kernel_for_its_derived_quantities():
    """A mixture of product kernels is not a product kernel, and the chain knows it.

    Under ``per_round`` the exact dynamics must run on the policy-averaged
    kernel. Reusing the per-agent product form would silently drop the
    correlation a shared control value induces between the agents it pushes.
    """

    game = SyntheticControlledMarkovGame()
    config = controlled_config(control_strength=0.5, control_mode=PER_ROUND)
    kernels = control_kernels(game.rules(config), game.control(config))
    expected = exact.policy_averaged_kernel(kernels, game.control_policy(config))

    assert game.chain_control(config) is None
    assert np.abs(game.exact_transition(config) - expected).max() < EXACT
    assert game.exact_transition(controlled_config(control_strength=0.5)) is None


# -- the config surface ---------------------------------------------------


def test_the_control_alphabet_and_control_targets_are_two_spellings_of_one_thing():
    game = SyntheticControlledMarkovGame()
    labels = game.control(controlled_config(control_alphabet=["Q", "M"])).targets
    indices = game.control(controlled_config(control_targets=[0, 1])).targets
    assert labels == indices == (0, 1)
    assert game.control(controlled_config(control_alphabet=["M", None])).targets == (1, -1)

    with pytest.raises(ValueError, match="exactly one"):
        game.control(controlled_config(control_alphabet=["Q"], control_targets=[0]))
    with pytest.raises(ValueError, match="not one of the game's actions"):
        game.control(controlled_config(control_alphabet=["Q", "Z"]))


def test_a_state_dependent_policy_is_refused_until_the_simple_case_is_verified():
    """Declared in the document as the next step, not as available today.

    The closed forms already accept a ``pi(u|s)`` table; what is missing is a
    config syntax for one. Refused with that reason rather than silently
    treated as uniform.
    """

    with pytest.raises(ValueError, match="state_dependent"):
        EmpowermentSpec.from_config({"policy": "state_dependent"}, 60)
    with pytest.raises(ValueError, match="control_mode"):
        EmpowermentSpec.from_config({"control_mode": "every_other_round"}, 60)
    with pytest.raises(ValueError, match="state_distribution"):
        EmpowermentSpec.from_config({"empowerment": {"state_distribution": "uniform"}}, 60)


def test_the_analysis_window_is_declared_rather_than_incidental():
    """Two windows over an unmixed chain must not give the same answer."""

    early = SyntheticControlledMarkovGame().ground_truth(
        controlled_config(
            epsilon=0.0,
            control_alphabet=["Q", None],
            control_strength=0.2,
            empowerment={"analysis_window": [1, 5]},
        )
    )
    late = SyntheticControlledMarkovGame().ground_truth(
        controlled_config(
            epsilon=0.0,
            control_alphabet=["Q", None],
            control_strength=0.2,
            empowerment={"analysis_window": [30, 60]},
        )
    )
    assert early.value("effective_empowerment") != pytest.approx(
        late.value("effective_empowerment"), abs=1e-6
    )


# -- the algebra underneath ------------------------------------------------


def test_the_horizon_truncation_solves_the_inequality_it_claims_to():
    for gamma in (0.5, 0.9, 0.99):
        for tolerance in (1e-2, 1e-3, 1e-6):
            horizon = exact.geometric_truncation(gamma, tolerance)
            assert gamma**horizon <= tolerance
            assert horizon == 1 or gamma ** (horizon - 1) > tolerance
    with pytest.raises(ValueError, match="discount"):
        exact.geometric_truncation(1.0, 1e-3)


def test_the_row_wise_kl_divergence_matches_its_definition():
    rows = np.array([[0.5, 0.5], [1.0, 0.0]])
    reference = np.array([[0.25, 0.75], [0.5, 0.5]])
    expected = np.array(
        [0.5 * np.log2(2.0) + 0.5 * np.log2(2 / 3), np.log2(2.0)]
    )
    assert exact.kl_divergence_rows(rows, reference) == pytest.approx(expected, abs=EXACT)
    # A distribution against itself is a structural zero, not a small number.
    assert exact.kl_divergence_rows(rows, rows) == pytest.approx([0.0, 0.0], abs=EXACT)


def test_the_visitation_distribution_is_the_window_average_it_claims_to_be():
    game = SyntheticControlledMarkovGame()
    config = controlled_config(control_strength=0.3)
    rules = game.rules(config)
    kernels = control_kernels(rules, game.control(config))
    averaged = exact.policy_averaged_kernel(kernels, game.control_policy(config))
    initial = initial_distribution(rules)

    window = (2, 6)
    manual = sum(
        exact.propagate(initial, averaged, round_index)
        for round_index in range(window[0], window[1] + 1)
    )
    manual = manual / manual.sum()
    assert exact.visitation_distribution(initial, averaged, window) == pytest.approx(
        manual, abs=1e-12
    )


def test_a_state_independent_policy_broadcasts_to_the_full_table():
    weights = exact.policy_weights(np.array([0.25, 0.75]), 2, 4)
    assert weights.shape == (2, 4)
    assert weights.sum(axis=0) == pytest.approx(np.ones(4), abs=EXACT)
    assert weights[0] == pytest.approx(np.full(4, 0.25), abs=EXACT)
    with pytest.raises(ValueError, match="pi"):
        exact.policy_weights(np.zeros((3, 3)), 2, 4)


def test_analyse_refuses_a_state_dependent_policy_because_3a_cannot_use_one():
    """Variant 3a draws ``u`` before any state exists, so ``pi(u|s)`` is undefined for it."""

    kernels = np.stack([np.eye(2), np.eye(2)])
    spec = EmpowermentSpec.from_config({}, 4)
    with pytest.raises(ValueError, match="before any state exists"):
        analyse(
            kernels,
            np.array([0.5, 0.5]),
            np.array([[0], [1]]),
            spec,
            policy=np.full((2, 2), 0.5),
        )
