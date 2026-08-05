"""Games 2 and 3, and the family-2 empowerment answer key.

The load-bearing tests here are the ones asserting an **exact zero**. A value
that is 5% off is hard to call a bug; a conditional mutual information that
must be identically zero and isn't is not ambiguous. Those are the checks that
catch direction errors and off-by-one round alignment - the classic silent
failures - so they are asserted at machine precision rather than with a
tolerance.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from mas_cc.analysis.estimators import conditional_mutual_information
from mas_cc.config import GameConfig, load_run_config
from mas_cc.games import create_game
from mas_cc.games.synthetic import compare_modes, create_synthetic_provider_registry
from mas_cc.games.synthetic import exact
from mas_cc.games.synthetic.bernoulli.game import SyntheticBernoulliGame
from mas_cc.games.synthetic.controlled_markov.game import SyntheticControlledMarkovGame
from mas_cc.games.synthetic.empowerment import (
    AnalysisSpec,
    SweepCell,
    SweepSpec,
    binning_matrix,
    sweep_from_values,
    sweep_ground_truth,
)
from mas_cc.games.synthetic.markov.game import SyntheticMarkovGame
from mas_cc.games.synthetic.runtime import run_synthetic_game_sync
from mas_cc.llm_runtime.providers import create_llm_provider

EXACT = 1e-12


def markov_config(**options) -> GameConfig:
    return GameConfig(
        type="synthetic_markov",
        population_size=options.pop("population_size", 3),
        horizon=options.pop("horizon", 400),
        options={"actions": ["Q", "M"], **options},
    )


def bernoulli_config(**options) -> GameConfig:
    return GameConfig(
        type="synthetic_bernoulli",
        population_size=options.pop("population_size", 5),
        horizon=options.pop("horizon", 200),
        options={"actions": ["Q", "M"], **options},
    )


# -- Game 2: the structural zeros -----------------------------------------


def test_chain_coupling_gives_an_exactly_zero_conditional_mutual_information():
    """On 1 -> 2 -> 3, agent 2 screens agent 0 off from agent 2's future.

    The data-processing configuration. Both halves matter: the conditional
    value must be *identically* zero, and the unconditional chain of MIs must
    strictly decrease along the path.
    """

    truth = SyntheticMarkovGame().ground_truth(
        markov_config(population_size=3, coupling="chain", epsilon=0.1)
    )
    screened = truth.value(
        "conditional_transfer_entropy", ("agent-000", "agent-002", "agent-001")
    )
    assert screened == pytest.approx(0.0, abs=EXACT)

    near = truth.value("mutual_information", ("agent-000", "agent-001"))
    far = truth.value("mutual_information", ("agent-000", "agent-002"))
    assert 0.0 < far < near, "the data-processing inequality must hold along the chain"


def test_asymmetric_coupling_makes_transfer_entropy_directional():
    """Agent 1 follows agent 0 and nothing follows agent 1.

    The reverse direction must be exactly zero. An off-by-one in round
    alignment, or a swapped source/target, turns that zero positive - which is
    the whole reason this configuration exists.
    """

    truth = SyntheticMarkovGame().ground_truth(
        markov_config(population_size=3, coupling=[0, 0, 2], epsilon=0.15)
    )
    forward = truth.value("transfer_entropy", ("agent-000", "agent-001"))
    reverse = truth.value("transfer_entropy", ("agent-001", "agent-000"))
    assert forward > 0.05
    assert reverse == pytest.approx(0.0, abs=EXACT)


def test_uncoupled_agents_have_no_transfer_entropy_at_all():
    truth = SyntheticMarkovGame().ground_truth(
        markov_config(population_size=4, coupling="self", epsilon=0.2)
    )
    values = [q.value for q in truth.quantities if q.name == "transfer_entropy"]
    assert values and max(abs(value) for value in values) < 1e-12


def test_lumpability_fails_exactly_where_the_theory_says_it_should():
    """Exchangeable populations lump; asymmetric coupling does not.

    Reported rather than assumed, because when it is False the shortcut of
    computing on the lumped macrostate chain gives a *different and wrong*
    answer, and only microstate propagation is correct.
    """

    exchangeable = SyntheticMarkovGame().ground_truth(
        markov_config(population_size=3, coupling="self", epsilon=0.2)
    )
    assert exchangeable.value("macrostate_is_lumpable") == 1.0

    asymmetric = SyntheticMarkovGame().ground_truth(
        markov_config(population_size=3, coupling=[0, 0, 2], epsilon=0.15)
    )
    assert asymmetric.value("macrostate_is_lumpable") == 0.0


def test_estimated_transfer_entropy_converges_on_the_exact_chain_value():
    """The claim the whole game exists to support, at a sample size where it holds."""

    game = SyntheticMarkovGame()
    config = markov_config(population_size=3, horizon=40_000, coupling=[0, 0, 2], epsilon=0.15)
    truth = game.ground_truth(config)
    actions = game.simulate(config, (17,)).actions[0]

    for source, target in ((0, 1), (1, 0)):
        estimate = conditional_mutual_information(
            actions[:-1, source], actions[1:, target], actions[:-1, target],
            x_levels=(0, 1), y_levels=(0, 1), z_levels=(0, 1),
        )
        expected = truth.value(
            "transfer_entropy", (f"agent-{source:03d}", f"agent-{target:03d}")
        )
        assert estimate.miller_madow == pytest.approx(expected, abs=0.01)


# -- Game 3: the positive control -----------------------------------------


def controlled_config(**options) -> GameConfig:
    return GameConfig(
        type="synthetic_controlled_markov",
        population_size=options.pop("population_size", 6),
        horizon=options.pop("horizon", 60),
        options={"actions": ["Q", "M"], "coupling": "ring", "epsilon": 0.05, **options},
    )


def test_control_mutual_information_is_zero_without_a_controller():
    """Zero strength is genuinely no channel, not merely a small one."""

    truth = SyntheticControlledMarkovGame().ground_truth(
        controlled_config(control_strength=0.0)
    )
    assert truth.value("control_design_mutual_information") == pytest.approx(0.0, abs=EXACT)
    assert truth.value("control_capacity") == pytest.approx(0.0, abs=EXACT)


def test_control_mutual_information_increases_with_strength():
    game = SyntheticControlledMarkovGame()
    values = [
        game.ground_truth(controlled_config(control_strength=strength)).value(
            "control_design_mutual_information"
        )
        for strength in (0.0, 0.05, 0.1, 0.25, 0.5, 1.0)
    ]
    assert values == sorted(values)
    assert values[-1] > 0.9  # two control values, so the ceiling is 1 bit


def test_capacity_bounds_the_design_mutual_information_and_coarse_graining_loses():
    """Two orderings that must always hold, and say different things.

    Capacity is the best any input law could do, so it bounds the design value
    - a gap there means the grid was chosen badly, not that the controller is
    weak. And the microstate ceiling bounds the macrostate one, because
    coarse-graining can only discard.
    """

    truth = SyntheticControlledMarkovGame().ground_truth(
        controlled_config(control_strength=0.3)
    )
    design = truth.value("control_design_mutual_information")
    capacity = truth.value("control_capacity")
    micro = truth.value("control_microstate_capacity")
    assert design <= capacity + EXACT
    assert capacity <= micro + EXACT


def test_a_disabled_controller_replays_the_uncontrolled_game_exactly():
    """Control draws come from their own stream, so adding one is not a re-seed."""

    plain = SyntheticMarkovGame().simulate(
        markov_config(population_size=6, horizon=50, coupling="ring", epsilon=0.05), (5,)
    )
    disabled = SyntheticControlledMarkovGame().simulate(
        controlled_config(horizon=50, control_strength=0.0), (5,)
    )
    assert np.array_equal(plain.actions, disabled.actions)


# -- both new games through the real pipeline -----------------------------


@pytest.mark.parametrize(
    "config_path",
    ["configs/runs/synthetic_markov.yaml", "configs/runs/synthetic_controlled_markov.yaml"],
)
def test_fidelity_and_speed_modes_agree_bit_for_bit(config_path):
    config = load_run_config(config_path, environment={})
    config = replace(config, game=replace(config.game, horizon=40))
    game = create_game(config.game)
    provider = create_llm_provider(
        config.llm_provider, registry=create_synthetic_provider_registry()
    )
    try:
        result = run_synthetic_game_sync(game, config, provider)
    finally:
        provider.close()
    parity = compare_modes(game, config.game, config.execution.seed, result.action_series())
    assert parity.identical, f"{parity.mismatched_cells} mismatched cells"


# -- the empowerment answer key -------------------------------------------


def test_game_one_terminal_empowerment_is_exactly_zero_for_odd_populations():
    """The cheapest and most diagnostic check in the whole harness.

    With an odd population there are no ties, the terminal marginal is uniform
    for every epsilon, and the terminal mutual information is a structural zero
    for *any* sweep. If an estimator's shuffle null does not cover this, the
    null construction is broken and every downstream significance claim is
    unreliable.
    """

    game = SyntheticBernoulliGame()
    for size in (3, 5, 7, 9):
        sweep = sweep_from_values(
            bernoulli_config(population_size=size), "epsilon", [0.05, 0.15, 0.3, 0.45],
            repetitions=50,
        )
        truth = sweep_ground_truth(game, sweep, AnalysisSpec(horizons=(1,)))
        assert truth.value("terminal_mutual_information") == pytest.approx(0.0, abs=EXACT)


@pytest.mark.parametrize("size", [4, 6, 8])
def test_even_populations_reproduce_the_tie_break_artifact_exactly(size):
    """Nonzero empowerment produced entirely by a tie-break convention.

    `reader.py` resolves an exact tie with `idxmax`, so every tie lands on the
    first declared action. With even N ties have positive probability and the
    outcome marginal stops being symmetric, so the pipeline reports empowerment
    where none exists. Run this deliberately: it must reproduce *precisely* the
    spurious value and no more, or the contamination is somewhere else.
    """

    grid = [0.05, 0.15, 0.3, 0.45]
    sweep = sweep_from_values(
        bernoulli_config(population_size=size), "epsilon", grid, repetitions=50
    )
    truth = sweep_ground_truth(SyntheticBernoulliGame(), sweep, AnalysisSpec(horizons=(1,)))

    def entropy(p: float) -> float:
        return 0.0 if p in (0.0, 1.0) else -p * math.log2(p) - (1 - p) * math.log2(1 - p)

    probabilities = [
        0.5 + 0.5 * math.comb(size, size // 2) * (e * (1 - e)) ** (size // 2) for e in grid
    ]
    mean = sum(probabilities) / len(probabilities)
    expected = entropy(mean) - sum(entropy(p) for p in probabilities) / len(probabilities)

    assert truth.value("terminal_mutual_information") == pytest.approx(expected, abs=1e-12)
    assert expected > 0.0, "the artifact is real, not a rounding effect"


def test_game_one_lagged_conditional_mutual_information_is_flat_in_the_horizon():
    """The diagnostic is the shape, not the number.

    Rounds are i.i.d. given the condition, so the lagged CMI must be *flat* in
    h at a height predicted in advance. Any slope is an artifact - windowing,
    an episode-boundary edge effect, or a null leaking into the estimate.
    """

    sweep = sweep_from_values(
        bernoulli_config(population_size=6), "epsilon", [0.05, 0.15, 0.3, 0.45],
        repetitions=50,
    )
    horizons = (1, 2, 3, 5, 10, 20)
    truth = sweep_ground_truth(
        SyntheticBernoulliGame(), sweep,
        AnalysisSpec(horizons=horizons, macrostate_bins=4),
    )
    values = [
        truth.value("lagged_conditional_mutual_information", (str(h),)) for h in horizons
    ]
    assert max(values) - min(values) < 1e-12
    assert values[0] > 0.0


def test_the_lagged_identity_from_the_ground_truth_document_holds():
    """I(C; S_{t+h} | S_t) = I(C;S) - I(S_t; S_{t+h}) for i.i.d. rounds.

    S_{t+h} is conditionally independent of S_t given C, but not
    unconditionally independent of it - they share C as a common cause. Checked
    against a directly-computed I(S_t;S_{t+h}) rather than restating the same
    algebra the implementation uses.
    """

    game = SyntheticBernoulliGame()
    grid = [0.05, 0.15, 0.3, 0.45]
    base = bernoulli_config(population_size=6)
    sweep = sweep_from_values(base, "epsilon", grid, repetitions=50)
    truth = sweep_ground_truth(
        game, sweep, AnalysisSpec(horizons=(1,), macrostate_bins=4)
    )

    design = sweep.design_distribution
    binner = binning_matrix(6, 4)
    laws = np.stack(
        [game.exact_dynamics(cell.config).macrostate_law(1) @ binner for cell in sweep.cells]
    )
    # p(k, k') = sum_c p(c) p(k|c) p(k'|c) - independent given c, coupled through it.
    pair = sum(design[i] * np.outer(laws[i], laws[i]) for i in range(len(design)))
    temporal = exact.mutual_information(pair)

    lagged = truth.value("lagged_conditional_mutual_information", ("1",))
    condition_state = truth.value("condition_state_mutual_information")
    assert lagged == pytest.approx(condition_state - temporal, abs=1e-10)


def test_sweeping_population_size_without_binning_is_refused():
    """The support artifact from the ground-truth document's section 3.4.

    A raw action share has support {0, 1/N, ..., 1}: a different alphabet per
    condition. The condition then becomes recoverable from a single observation
    and I(C;S) collapses onto H(C) - maximal, and entirely an artifact. Refused
    outright rather than warned about, because it invalidates every
    population-size result silently.
    """

    sweep = sweep_from_values(
        bernoulli_config(), "population_size", [4, 6, 10], repetitions=20
    )
    with pytest.raises(ValueError, match="support"):
        sweep_ground_truth(SyntheticBernoulliGame(), sweep, AnalysisSpec(horizons=(1,)))

    # Binned onto a common grid, the same sweep is well posed.
    truth = sweep_ground_truth(
        SyntheticBernoulliGame(), sweep, AnalysisSpec(horizons=(1,), macrostate_bins=4)
    )
    assert truth.value("macrostate_alphabet_size") == 4.0


def test_binning_puts_every_population_size_on_one_common_grid():
    for size in (4, 6, 10):
        matrix = binning_matrix(size, 4)
        assert matrix.shape == (size + 1, 4)
        # Every macrostate lands in exactly one bin, and the extremes separate.
        assert np.allclose(matrix.sum(axis=1), 1.0)
        assert matrix[0].argmax() == 0
        assert matrix[-1].argmax() == 3


def test_design_distribution_follows_repetitions_not_an_assumption():
    """Unequal repetitions per cell change p(c), and therefore the answer."""

    base = bernoulli_config(population_size=4)
    cells = (
        SweepCell("a", replace(base, options={**dict(base.options), "epsilon": 0.05}), 90),
        SweepCell("b", replace(base, options={**dict(base.options), "epsilon": 0.45}), 10),
    )
    sweep = SweepSpec("epsilon", cells)
    assert sweep.design_distribution == pytest.approx([0.9, 0.1])
    assert sweep.episode_count == 100

    balanced = sweep_from_values(base, "epsilon", [0.05, 0.45], repetitions=50)
    assert balanced.design_distribution == pytest.approx([0.5, 0.5])
    # The same cells with a different design give a different mutual information.
    unequal = sweep_ground_truth(SyntheticBernoulliGame(), sweep, AnalysisSpec())
    equal = sweep_ground_truth(SyntheticBernoulliGame(), balanced, AnalysisSpec())
    assert unequal.value("terminal_mutual_information") != pytest.approx(
        equal.value("terminal_mutual_information"), abs=1e-9
    )


def test_plugin_bias_uses_episodes_and_the_documented_degrees_of_freedom():
    """dof/(2 N ln2), with N the number of EPISODES rather than rounds."""

    assert exact.plugin_bias_bits(110, 100) == pytest.approx(
        110 / (2 * 100 * math.log(2)), abs=1e-12
    )
    sweep = sweep_from_values(
        bernoulli_config(population_size=10), "epsilon", [0.1, 0.4], repetitions=50
    )
    truth = sweep_ground_truth(
        SyntheticBernoulliGame(), sweep, AnalysisSpec(horizons=(1,))
    )
    # The worked case from the document: |M| = 11, |C| = 2, 100 episodes.
    assert truth.value("effective_sample_size") == 100.0
    assert truth.value("macrostate_alphabet_size") == 11.0
    assert truth.value("lagged_plugin_bias", ("1",)) == pytest.approx(
        11 * 1 * 10 / (2 * 100 * math.log(2)), abs=1e-9
    )
    # ~0.79 bits of pure bias - enough to swamp any real effect, which is the
    # point of shipping the number next to the truth.
    assert truth.value("lagged_plugin_bias", ("1",)) > 0.75


def test_binning_the_macrostate_collapses_the_bias():
    """Why binning to 3-5 levels is almost certainly not optional."""

    sweep = sweep_from_values(
        bernoulli_config(population_size=10), "epsilon", [0.1, 0.4], repetitions=50
    )
    unbinned = sweep_ground_truth(
        SyntheticBernoulliGame(), sweep, AnalysisSpec(horizons=(1,))
    ).value("lagged_plugin_bias", ("1",))
    binned = sweep_ground_truth(
        SyntheticBernoulliGame(), sweep, AnalysisSpec(horizons=(1,), macrostate_bins=3)
    ).value("lagged_plugin_bias", ("1",))
    assert binned < unbinned / 5


# -- exact algebra --------------------------------------------------------


def test_blahut_arimoto_reproduces_textbook_capacities():
    for crossover in (0.0, 0.1, 0.25, 0.5):
        channel = np.array([[1 - crossover, crossover], [crossover, 1 - crossover]])
        capacity, _ = exact.blahut_arimoto(channel)
        entropy = (
            0.0
            if crossover in (0.0, 0.5) and crossover != 0.5
            else exact.entropy(np.array([crossover, 1 - crossover]))
        )
        assert capacity == pytest.approx(1.0 - entropy, abs=1e-9)
    # Z-channel, capacity log2(5/4).
    capacity, _ = exact.blahut_arimoto(np.array([[1.0, 0.0], [0.5, 0.5]]))
    assert capacity == pytest.approx(math.log2(1.25), abs=1e-9)


def test_stationary_distribution_is_actually_stationary():
    matrix = np.array([[0.9, 0.1], [0.2, 0.8]])
    law = exact.stationary_distribution(matrix)
    assert law == pytest.approx([2 / 3, 1 / 3], abs=1e-12)
    assert law @ matrix == pytest.approx(law, abs=1e-12)


def test_transition_matrix_rows_are_probability_distributions():
    from mas_cc.games.synthetic.markov.game import MarkovSpec, transition_matrix

    spec = MarkovSpec.from_config(
        markov_config(population_size=4, coupling="ring", epsilon=0.12)
    )
    matrix = transition_matrix(spec)
    assert matrix.shape == (16, 16)
    assert matrix.sum(axis=1) == pytest.approx(np.ones(16), abs=1e-12)


def test_population_size_is_capped_so_the_chain_stays_exact():
    with pytest.raises(ValueError, match="1024"):
        SyntheticMarkovGame().rules(markov_config(population_size=11))
