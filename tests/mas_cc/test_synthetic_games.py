"""Checks on the synthetic games themselves.

A note on what this file is and is not. The synthetic games are *not* tests -
they are a rehearsal of the workflow with the answer key in hand, run the way a
real experiment is run. This file is the thin layer underneath that: it pins
down the things that would silently invalidate the rehearsal if they broke -
the closed forms, the determinism, the fidelity/speed equivalence, and the fact
that the agent really does read the prompt.

The most load-bearing test here is
`test_parity_detects_a_deliberately_misaligned_game`. A parity check that
cannot fail proves nothing, so that one breaks the pipeline on purpose and
insists the check notices.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from mas_cc.config import GameConfig, load_run_config
from mas_cc.games import create_default_game_registry, create_game, game_metrics
from mas_cc.games.protocols import Observation
from mas_cc.games.synthetic import (
    SyntheticGame,
    SyntheticPromptError,
    compare_modes,
    create_synthetic_provider_registry,
    pairwise_estimates,
    read_action_series,
    run_synthetic_game_sync,
    with_truth,
)
from mas_cc.games.synthetic.analysis import series_to_indices
from mas_cc.games.synthetic.bernoulli import SyntheticBernoulliGame, binary_entropy
from mas_cc.games.synthetic.provider import decide, read_observation
from mas_cc.llm_runtime.providers import CompletionRequest, create_llm_provider
from mas_cc.llm_runtime.messages import Message, MessageRole

FIDELITY_CONFIG = "configs/runs/synthetic_bernoulli_fidelity.yaml"
NULL_CONFIG = "configs/runs/synthetic_bernoulli_null.yaml"


def _config(path: str = FIDELITY_CONFIG):
    return load_run_config(path, environment={})


def _small(config, *, rounds: int = 25, population: int = 3, epsilon: float = 0.2):
    """A config small enough to run a full-pipeline episode inside a test."""

    options = {key: value for key, value in config.game.options.items() if key != "epsilons"}
    options["epsilon"] = epsilon
    return replace(
        config,
        game=replace(
            config.game, population_size=population, horizon=rounds, options=options
        ),
    )


# -- the closed forms -----------------------------------------------------


def test_ground_truth_hits_both_anchors_exactly():
    """eps = 0.5 gives exactly zero bits; eps = 0 gives exactly one."""

    game = SyntheticBernoulliGame()
    base = _config().game
    pair = ("agent-000", "agent-001")

    independent = game.ground_truth(replace(base, options={"actions": ["Q", "M"], "epsilon": 0.5}))
    assert independent.value("mutual_information", pair) == 0.0

    identical = game.ground_truth(replace(base, options={"actions": ["Q", "M"], "epsilon": 0.0}))
    assert identical.value("mutual_information", pair) == 1.0


def test_ground_truth_is_per_pair_when_agents_differ():
    """Asymmetric noise means each pair has its own truth, not one shared number."""

    game = SyntheticBernoulliGame()
    config = replace(
        _config().game,
        population_size=3,
        options={"actions": ["Q", "M"], "epsilons": [0.0, 0.1, 0.4]},
    )
    truth = game.ground_truth(config)

    def expected(left: float, right: float) -> float:
        return 1.0 - binary_entropy(left * (1 - right) + right * (1 - left))

    assert truth.value("mutual_information", ("agent-000", "agent-001")) == pytest.approx(
        expected(0.0, 0.1)
    )
    assert truth.value("mutual_information", ("agent-001", "agent-002")) == pytest.approx(
        expected(0.1, 0.4)
    )
    # The three pairs really are different; a single shared value would pass a
    # weaker assertion than this one.
    values = {
        round(quantity.value, 12)
        for quantity in truth.quantities
        if quantity.name == "mutual_information"
    }
    assert len(values) == 3


def test_ground_truth_follows_the_config_it_was_given():
    """Change epsilon and the answer key changes with it - no stale expectations."""

    game = SyntheticBernoulliGame()
    base = _config().game
    pair = ("agent-000", "agent-001")
    values = [
        game.ground_truth(
            replace(base, options={"actions": ["Q", "M"], "epsilon": epsilon})
        ).value("mutual_information", pair)
        for epsilon in (0.0, 0.1, 0.25, 0.5)
    ]
    assert values == sorted(values, reverse=True)


def test_unanimity_probability_matches_the_sampled_rate():
    """A closed form that exercises the transition, not just the estimator."""

    game = SyntheticBernoulliGame()
    config = replace(
        _config().game, population_size=4, horizon=4000,
        options={"actions": ["Q", "M"], "epsilon": 0.25},
    )
    expected = game.ground_truth(config).value("unanimity_probability")
    episodes = game.simulate(config, range(8))
    observed = float(
        np.mean(
            (episodes.actions.min(axis=2) == episodes.actions.max(axis=2)).astype(float)
        )
    )
    assert observed == pytest.approx(expected, abs=0.01)


# -- determinism and speed mode -------------------------------------------


def test_simulation_is_reproducible_and_seed_dependent():
    game = SyntheticBernoulliGame()
    config = _small(_config()).game
    first = game.simulate(config, (11, 12))
    again = game.simulate(config, (11, 12))
    assert np.array_equal(first.actions, again.actions)
    assert not np.array_equal(first.actions[0], first.actions[1])


def test_marginals_are_uniform_by_construction():
    """Every agent's own action is a fair coin whatever its noise level.

    This is what makes I(A_i;A_j) = 1 - H(q) exact rather than approximate, so
    it is worth checking independently of the mutual information itself.
    """

    game = SyntheticBernoulliGame()
    config = replace(
        _config().game, population_size=4, horizon=5000,
        options={"actions": ["Q", "M"], "epsilons": [0.0, 0.1, 0.35, 0.5]},
    )
    episodes = game.simulate(config, (7,))
    rates = episodes.actions[0].mean(axis=0)
    assert rates == pytest.approx(np.full(4, 0.5), abs=0.03)


def test_estimator_converges_on_the_closed_form():
    """The rehearsal's headline claim, at a sample size where it should hold."""

    game = SyntheticBernoulliGame()
    config = replace(
        _config().game, population_size=3, horizon=20000,
        options={"actions": ["Q", "M"], "epsilon": 0.15},
    )
    episodes = game.simulate(config, (3,))
    truth = game.ground_truth(config)
    frame = with_truth(
        pairwise_estimates(episodes.actions[0], ["agent-000", "agent-001", "agent-002"], 2),
        truth,
    )
    assert frame["gap_unsmoothed"].abs().max() < 0.01


# -- the agent really reads the prompt ------------------------------------


def _request(content: str) -> CompletionRequest:
    return CompletionRequest(messages=(Message(MessageRole.USER, content),))


def test_agent_decodes_the_observation_out_of_the_compiled_prompt():
    game = SyntheticBernoulliGame()
    config = _small(_config()).game
    state = game.initialize(config, seed=5)
    observations = game.construct_observations(
        state, game.select_participants(state, config, __import__("random").Random(0)), config
    )
    request = game.build_decision_requests(state, observations, config)[0]
    compiled = request.prompt.compile()

    payload = read_observation(CompletionRequest(messages=compiled.messages))
    assert payload["signal"] in ("Q", "M")
    assert payload["round"] == 1

    answer = decide(CompletionRequest(messages=compiled.messages))
    expected = payload["signal"] if not payload["flip"] else ("M" if payload["signal"] == "Q" else "Q")
    assert answer == expected


def test_agent_refuses_a_prompt_that_does_not_carry_the_observation():
    """The failure mode this design exists to catch, made explicit.

    If prompt construction ever stops rendering the payload, the agent must
    fail loudly rather than fall back to a default - a silent default would
    turn a broken prompt into a plausible-looking mutual information.
    """

    with pytest.raises(SyntheticPromptError, match="no SYNTHETIC-OBSERVATION"):
        decide(_request("Choose Q or M."))


def test_agent_rejects_an_unknown_decoding_policy():
    with pytest.raises(SyntheticPromptError, match="unknown decoding policy"):
        decide(
            _request(
                'SYNTHETIC-OBSERVATION-V1 {"policy":"not_a_policy","round":1,'
                '"actions":["Q","M"],"signal":"Q","flip":false}'
            )
        )


# -- fidelity mode and the parity check -----------------------------------


def _run_fidelity(config) -> dict[str, list[str]]:
    game = create_game(config.game)
    provider = create_llm_provider(
        config.llm_provider, registry=create_synthetic_provider_registry()
    )
    try:
        result = run_synthetic_game_sync(game, config, provider)
    finally:
        provider.close()
    return result.action_series()


def test_fidelity_and_speed_modes_agree_bit_for_bit():
    """The check that makes speed mode a trustworthy proxy for sweeps."""

    config = _small(_config())
    game = create_game(config.game)
    recorded = _run_fidelity(config)
    parity = compare_modes(game, config.game, config.execution.seed, recorded)
    assert parity.identical
    assert parity.mismatched_cells == 0
    assert parity.fidelity_estimate == pytest.approx(parity.speed_estimate)


def test_parity_detects_a_deliberately_misaligned_game(monkeypatch):
    """A parity check that cannot fail proves nothing, so make it fail.

    The injected bug is the classic silent one: the observation is built from
    the *previous* round's coins. Nothing raises, every action is individually
    legal, and the population statistics still look reasonable - only the
    trajectory comparison notices.
    """

    config = _small(_config())
    game = create_game(config.game)
    honest = game.construct_observations

    def _off_by_one(state, participants, game_config):
        shifted = replace(state, turn=max(0, state.turn - 1))
        return tuple(
            Observation(
                agent_id=observation.agent_id,
                interaction_id=observation.interaction_id,
                participants=observation.participants,
                visible_state={
                    **dict(observation.visible_state),
                    "round": state.turn + 1,
                },
            )
            for observation in honest(shifted, participants, game_config)
        )

    monkeypatch.setattr(type(game), "construct_observations", staticmethod(_off_by_one))
    recorded = _run_fidelity(config)
    monkeypatch.undo()

    parity = compare_modes(create_game(config.game), config.game, config.execution.seed, recorded)
    assert not parity.identical
    assert parity.mismatched_cells > 0
    assert parity.first_mismatch is not None


# -- the artifact path ----------------------------------------------------


def test_recorded_artifacts_round_trip_to_the_same_trajectory(tmp_path):
    """What the recorder wrote must be what the run actually did.

    Reads back `metrics/streaming.csv` rather than the in-memory result,
    because that file is what a run pulled off the cluster is analysed from.
    """

    from mas_cc.cli.synthetic import _run_fidelity_episode

    config = _small(_config())
    game = create_game(config.game)
    result, destination = _run_fidelity_episode(config, game, tmp_path / "run")

    assert (destination / "ground_truth.json").is_file()
    series = read_action_series(destination)
    assert series == result.action_series()

    parity = compare_modes(game, config.game, config.execution.seed, series)
    assert parity.identical


def test_ground_truth_artifact_is_written_before_the_episode_can_fail(tmp_path):
    """The answer key survives a failed run, which is when it is most wanted."""

    from mas_cc.cli.synthetic import _run_fidelity_episode

    config = _small(_config())
    game = create_game(config.game)

    def _explode(*args, **kwargs):
        raise RuntimeError("injected failure")

    game.apply_transition = _explode  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="injected failure"):
        _run_fidelity_episode(config, game, tmp_path / "run")
    assert (tmp_path / "run" / "ground_truth.json").is_file()


# -- framework wiring -----------------------------------------------------


def test_registry_exposes_the_synthetic_game():
    registry = create_default_game_registry()
    assert "synthetic_bernoulli" in registry.names()
    assert isinstance(registry.create(_config().game), SyntheticGame)


def test_metrics_are_discovered_from_a_nested_game_package():
    """`game_metrics` resolves from the game's package, not its game_type string."""

    metrics, to_round_view = game_metrics(create_game(_config().game))
    assert to_round_view is not None
    assert "agent_current_action" in {metric.name for metric in metrics}

    # The games that live at games/<game_type>/ must still resolve unchanged.
    naming = create_game(
        load_run_config("configs/runs/naming_convention_smoke_test_v3.yaml", environment={}).game
    )
    naming_metrics, naming_view = game_metrics(naming)
    assert naming_view is not None and naming_metrics


def test_a_synthetic_game_without_an_answer_key_cannot_be_constructed():
    """The ABC is the enforcement: a missing ground_truth fails at construction."""

    class Incomplete(SyntheticGame):
        pass

    with pytest.raises(TypeError, match="abstract"):
        Incomplete()  # type: ignore[abstract]


def test_series_to_indices_rejects_an_action_outside_the_alphabet():
    with pytest.raises(ValueError, match="outside the declared alphabet"):
        series_to_indices({"agent-000": ["Q", "X"]}, ("Q", "M"))


def test_binary_entropy_takes_the_limits_at_the_endpoints():
    """H(0) = H(1) = 0 rather than nan; eps = 0 is one of the anchor configs."""

    assert binary_entropy(0.0) == 0.0
    assert binary_entropy(1.0) == 0.0
    assert binary_entropy(0.5) == pytest.approx(1.0)
    assert not math.isnan(binary_entropy(0.0))


def test_invalid_configs_are_rejected_with_a_useful_message():
    game = SyntheticBernoulliGame()
    base = _config().game
    with pytest.raises(ValueError, match="epsilon"):
        game.rules(replace(base, options={"actions": ["Q", "M"], "epsilon": 1.5}))
    with pytest.raises(ValueError, match="two distinct action labels"):
        game.rules(replace(base, options={"actions": ["Q"], "epsilon": 0.1}))
    with pytest.raises(ValueError, match="one entry per agent"):
        game.rules(
            replace(base, population_size=4, options={"actions": ["Q", "M"], "epsilons": [0.1, 0.2]})
        )
