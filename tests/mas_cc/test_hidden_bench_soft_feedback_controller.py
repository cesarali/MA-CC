"""Acceptance tests for the stochastic `soft_target` feedback controller.

Covers `docs/tdd/architecture/11082026_soft_feedback_controller_implementation.md`.
The controller exists for one reason: `threshold_target` puts nearly every event
in a conditioning slice that only ever saw one action, and
`I(U_t; n_Z(t+1) | n_Z(t))` cannot be estimated from slices with no contrast.
So the tests that matter most here are the two at the bottom - action overlap
*within* a `Z_t` slice, and `threshold_target` staying bit-for-bit what it was.
"""

from __future__ import annotations

import asyncio
import json
import random
from collections import defaultdict
from dataclasses import replace

import pytest

from mas_cc.config import load_run_config
from mas_cc.control import create_control
from mas_cc.control.registry import create_default_control_registry
from mas_cc.games import create_game
from mas_cc.games.hidden_bench.data import DEFAULT_CORPUS_ROOT, normalized_text
from mas_cc.games.hidden_bench.imitation import run_hidden_bench_imitation_game
from mas_cc.games.hidden_bench.imitation.controller import (
    ADVOCATE_TARGET,
    FIXED_ADVOCACY_TEMPLATE_ID,
    FORBIDDEN_MESSAGE_TERMS,
    NO_OP,
    SoftTargetControl,
    ThresholdTargetControl,
    advocacy_probability,
    check_frames,
    fixed_advocacy_message,
)
from mas_cc.games.hidden_bench.imitation.game import HiddenBenchImitationGame
from mas_cc.llm_runtime.exceptions import ConfigurationError
from mas_cc.llm_runtime.providers import create_llm_provider
from mas_cc.llm_runtime.providers.adapters.mock import MockLLMProvider

pytestmark = pytest.mark.skipif(
    not (DEFAULT_CORPUS_ROOT / "canonical" / "tasks.json").exists(),
    reason="HiddenBench corpus is not present",
)

CLASSICAL_CONFIG = "configs/runs/hidden_bench/hidden_bench_imitation_classical.yaml"
REASONING_CONFIG = "configs/runs/hidden_bench/hidden_bench_imitation_reasoning_mock.yaml"
RANDOM_TARGET_CONFIG = (
    "configs/runs/hidden_bench/hidden_bench_imitation_classical_control_10.yaml"
)


def _config(path: str, *, horizon: int | None = None, **options):
    config = load_run_config(path, environment={})
    if options or horizon is not None:
        config = replace(
            config,
            game=replace(
                config.game,
                horizon=config.game.horizon if horizon is None else horizon,
                options={**dict(config.game.options), **options},
            ),
        )
    return config


def _with_control(config, mechanism: str, **options):
    return replace(
        config,
        control=replace(
            config.control,
            mechanism=mechanism,
            options={
                "target": "correct",
                "sensor_sample_size": 2,
                "threshold": 0.5,
                **options,
            },
        ),
    )


def _run(config, *, provider=None, control=None):
    game = create_game(config.game)
    provider = provider or create_llm_provider(config.llm_provider)
    return asyncio.run(
        run_hidden_bench_imitation_game(game, config, provider, control=control)
    )


def _events(result):
    return [item.transition.event for item in result.interactions]


# --------------------------------------------------------------------------
# The policy itself
# --------------------------------------------------------------------------


def test_advocacy_probability_decreases_with_sampled_target_support():
    low, mid, high = (
        advocacy_probability(share, threshold=0.5, beta=4.0)
        for share in (0.0, 0.5, 1.0)
    )
    assert low > mid > high
    assert 0.0 < high and low < 1.0


def test_advocacy_probability_is_one_half_at_the_threshold():
    for threshold in (0.0, 0.25, 0.5, 0.75, 1.0):
        for beta in (0.5, 4.0, 50.0):
            assert advocacy_probability(
                threshold, threshold=threshold, beta=beta
            ) == pytest.approx(0.5)


def test_large_beta_approaches_the_hard_threshold_and_never_overflows():
    """`beta` interpolates to `threshold_target`; huge values saturate, not raise."""

    assert advocacy_probability(0.0, threshold=0.5, beta=1e4) == pytest.approx(1.0)
    assert advocacy_probability(1.0, threshold=0.5, beta=1e4) == pytest.approx(0.0)


def test_select_action_uses_the_supplied_rng_and_reports_its_probability():
    control = SoftTargetControl.from_options(
        {"target": "correct", "sensor_sample_size": 2, "threshold": 0.5, "beta": 4.0}
    )
    expected = advocacy_probability(0.5, threshold=0.5, beta=4.0)

    class _Draw:
        def __init__(self, value: float) -> None:
            self.value = value

        def random(self) -> float:
            return self.value

    below, probability = control.select_action(0.5, _Draw(expected - 1e-9))
    above, _ = control.select_action(0.5, _Draw(expected + 1e-9))
    assert (below, above) == (ADVOCATE_TARGET, NO_OP)
    assert probability == pytest.approx(expected)


def test_threshold_policy_reports_a_degenerate_probability():
    """The deterministic mechanism logs the same quantity, at 1.0 or 0.0."""

    control = ThresholdTargetControl.from_options({"target": "correct"})
    assert control.select_action(0.0, random.Random(0)) == (ADVOCATE_TARGET, 1.0)
    assert control.select_action(1.0, random.Random(0)) == (NO_OP, 0.0)


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


def test_soft_target_is_registered_and_selectable_through_run_configuration():
    assert "soft_target" in create_default_control_registry().names()
    config = _with_control(_config(CLASSICAL_CONFIG), "soft_target", beta=4.0)
    control = create_control(config.control)
    assert isinstance(control, SoftTargetControl)
    assert (control.policy, control.beta, control.threshold) == ("soft_target", 4.0, 0.5)


def test_beta_defaults_are_explicit_and_invalid_values_are_refused():
    assert SoftTargetControl.from_options({"target": "correct"}).beta == 4.0
    for bad in (0, -1.0, True, "4.0", None):
        with pytest.raises(ConfigurationError):
            SoftTargetControl.from_options({"target": "correct", "beta": bad})


def test_target_resolution_failure_is_never_silently_defaulted():
    control = SoftTargetControl.from_options({"target": 99, "sensor_sample_size": 1})
    state = HiddenBenchImitationGame().initialize(_config(CLASSICAL_CONFIG).game, 5)
    with pytest.raises(ValueError, match="outside"):
        control.interaction_signal(
            agent_id=state.agents[0].agent_id,
            interaction_index=1,
            state=state,
            rng=random.Random(0),
        )


def test_random_incorrect_target_is_task_agnostic_reproducible_and_episode_fixed():
    control = SoftTargetControl.from_options(
        {"target": "random_incorrect", "sensor_sample_size": 1}
    )
    game = HiddenBenchImitationGame()
    config = _config(RANDOM_TARGET_CONFIG)
    targets_by_seed = {}
    for seed in range(20):
        state = game.initialize(config.game, seed)
        first = control._resolved_target(state)
        second = control._resolved_target(state)
        assert first == second
        assert first in state.possible_answers
        assert first != state.correct_answer
        targets_by_seed[seed] = first

    # The task used by this fixture has multiple wrong answers, so a reasonable
    # seed range must exercise more than one randomly selected direction.
    assert len(set(targets_by_seed.values())) > 1


def test_random_incorrect_is_accepted_through_run_configuration():
    config = _with_control(
        _config(RANDOM_TARGET_CONFIG), "soft_target", target="random_incorrect"
    )
    control = create_control(config.control)
    assert isinstance(control, SoftTargetControl)
    assert control.target == "random_incorrect"


def test_sensor_sample_size_cannot_exceed_the_population():
    control = SoftTargetControl.from_options({"target": "correct", "sensor_sample_size": 99})
    state = HiddenBenchImitationGame().initialize(_config(CLASSICAL_CONFIG).game, 5)
    with pytest.raises(ValueError, match="population size"):
        control.interaction_signal(
            agent_id=state.agents[0].agent_id,
            interaction_index=1,
            state=state,
            rng=random.Random(0),
        )


# --------------------------------------------------------------------------
# The fixed language realization of ADVOCATE_Z
# --------------------------------------------------------------------------


def test_fixed_advocacy_message_advocates_without_asserting_anything_new():
    message = fixed_advocacy_message("North Hill")
    assert "North Hill" in message
    assert message.rstrip().endswith("I'm voting **North Hill**.")
    lowered = message.lower()
    assert all(term not in lowered for term in FORBIDDEN_MESSAGE_TERMS)
    # No claim of private or newly acquired evidence, and nothing task-specific.
    assert all(
        phrase not in lowered
        for phrase in ("i just learned", "i have information", "i know that", "evidence shows")
    )
    assert fixed_advocacy_message("North Hill") == message, "must be deterministic"
    assert not check_frames()


def test_soft_target_defaults_to_the_fixed_template_and_keeps_the_others_available():
    assert SoftTargetControl.from_options({"target": "correct"}).template_version == 3
    assert ThresholdTargetControl.from_options({"target": "correct"}).template_version == 1
    state = HiddenBenchImitationGame().initialize(_config(CLASSICAL_CONFIG).game, 5)
    control = SoftTargetControl.from_options(
        # Sampling the whole population, a threshold of 1.0, and a saturating
        # beta make advocacy certain, so the message is the only thing tested.
        {
            "target": "correct",
            "sensor_sample_size": len(state.agents),
            "threshold": 1.0,
            "beta": 1e4,
        }
    )
    signal = control.interaction_signal(
        agent_id=state.agents[0].agent_id,
        interaction_index=1,
        state=state,
        rng=random.Random(0),
    )
    assert signal.action == ADVOCATE_TARGET
    assert signal.message == fixed_advocacy_message(state.correct_answer)
    assert signal.metadata["message_template_version"] == 3
    assert signal.metadata["message_template_id"] == FIXED_ADVOCACY_TEMPLATE_ID
    assert signal.metadata["message_fact_index"] is None


def test_reasoning_advocacy_shapes_the_prompt_but_never_writes_the_vote():
    base = _config(REASONING_CONFIG, horizon=1)
    wrong = "East Town"
    config = _with_control(
        replace(
            base,
            game=replace(
                base.game,
                options={
                    **dict(base.game.options),
                    "initialization": {
                        "mode": "explicit",
                        "initial_votes": [wrong] * 4,
                        "initial_distribution": None,
                    },
                },
            ),
            llm_provider=replace(
                base.llm_provider,
                options={"response": '{"vote": "East Town", "rationale": "I am not moved"}'},
            ),
        ),
        "soft_target",
        sensor_sample_size=4,
        threshold=1.0,
        beta=50.0,
    )
    result = _run(config, control=create_control(config.control))
    event = _events(result)[0]
    assert event["controller_action"] == ADVOCATE_TARGET
    assert event["controller_target"] == "West City"
    assert event["controller_message"] == fixed_advocacy_message("West City")
    assert event["controller_template_version"] == 3
    assert event["controller_template_id"] == FIXED_ADVOCACY_TEMPLATE_ID
    # The focal agent went through the ordinary LLM decision path and kept its
    # own answer.  A controller that could overwrite the vote would make the
    # actuation channel a tautology.
    assert event["focal_opinion_after"] == wrong
    # No HiddenBench fact was injected: the controller is never given the task's
    # unshared information, so advocacy is social pressure, not new evidence.
    assert event["disclosed_hidden_facts_this_event"] == [
        False for _ in result.initial_state.hidden_information
    ]
    assert all(
        normalized_text(fact) not in normalized_text(event["controller_message"])
        for fact in result.initial_state.hidden_information
    )


# --------------------------------------------------------------------------
# Classical mode, reproducibility, logging
# --------------------------------------------------------------------------


def test_classical_mode_is_provider_free_and_still_mixes_both_actions():
    def forbidden(_request):
        raise AssertionError("classical mode called the provider")

    config = _with_control(_config(CLASSICAL_CONFIG, horizon=60), "soft_target", beta=4.0)
    provider = MockLLMProvider(config.llm_provider, response_factory=forbidden)
    result = _run(config, provider=provider, control=create_control(config.control))
    events = _events(result)
    assert result.logical_decisions == 0
    assert {event["controller_action"] for event in events} == {ADVOCATE_TARGET, NO_OP}
    # The existing classical actuator, unchanged: advocacy tilts the transition
    # weight toward the target rather than assigning it.
    advocated = [event for event in events if event["controller_action"] == ADVOCATE_TARGET]
    assert any(event["classical_control_weight"] > 0 for event in advocated)
    assert all(
        event["classical_control_weight"] == 0
        for event in events
        if event["controller_action"] == NO_OP
    )


def test_same_seed_and_config_reproduce_the_controller_action_sequence():
    config = _with_control(_config(CLASSICAL_CONFIG, horizon=40), "soft_target", beta=4.0)
    first = _run(config, control=create_control(config.control))
    second = _run(config, control=create_control(config.control))
    assert [event["controller_action"] for event in _events(first)] == [
        event["controller_action"] for event in _events(second)
    ]
    assert json.dumps(first.to_dict(), sort_keys=True) == json.dumps(
        second.to_dict(), sort_keys=True
    )

    other_seed = replace(config, execution=replace(config.execution, seed=config.execution.seed + 1))
    assert [event["controller_action"] for event in _events(first)] != [
        event["controller_action"] for event in _events(_run(other_seed, control=create_control(other_seed.control)))
    ], "a different seed must be able to produce a different action sequence"


def test_each_event_logs_the_realized_action_and_the_probability_behind_it():
    config = _with_control(_config(CLASSICAL_CONFIG, horizon=40), "soft_target", beta=4.0)
    for event in _events(_run(config, control=create_control(config.control))):
        assert event["controller_policy"] == "soft_target"
        assert event["controller_threshold"] == 0.5
        assert event["controller_beta"] == 4.0
        expected = advocacy_probability(
            event["sensor_target_share"], threshold=0.5, beta=4.0
        )
        assert event["controller_advocacy_probability"] == pytest.approx(expected)
        assert 0.0 < event["controller_advocacy_probability"] < 1.0
        # Everything the information analysis reads is still there.
        for field in (
            "N", "population_state_before", "population_state_after", "sensor_count_vector",
            "sensor_sample_size", "controller_action", "controller_target", "episode_id",
            "interaction_index", "focal_opinion_before", "focal_opinion_after",
        ):
            assert event[field] is not None


def test_soft_control_gives_the_target_slices_the_action_overlap_cmi_needs():
    """The reason this mechanism exists, stated as a test.

    `target_actuation_cmi` conditions on `Z_t`, the headcount on the target.
    A slice that only ever saw one action contributes nothing, so the fraction
    of events sitting in single-action slices is the ceiling on how much of the
    data the estimator can use.
    """

    def slices(mechanism: str, **options) -> dict[int, set[str]]:
        config = _with_control(_config(CLASSICAL_CONFIG, horizon=400), mechanism, **options)
        by_slice: dict[int, list[str]] = defaultdict(list)
        for event in _events(_run(config, control=create_control(config.control))):
            target = event["controller_target"]
            by_slice[event["occupation_counts_before"][target]].append(
                event["controller_action"]
            )
        return by_slice

    def collapsed_event_fraction(by_slice: dict[int, list[str]]) -> float:
        total = sum(len(actions) for actions in by_slice.values())
        return sum(
            len(actions) for actions in by_slice.values() if len(set(actions)) < 2
        ) / total

    hard = slices("threshold_target")
    soft = slices("soft_target", beta=4.0)
    assert hard.keys() == soft.keys(), "both mechanisms must visit the same Z_t slices"

    # The deterministic policy is saturated at both ends: with everyone off the
    # target it always advocates, with the target dominant it never does.
    assert any(len(set(actions)) < 2 for actions in hard.values())
    assert all(len(set(actions)) == 2 for actions in soft.values())
    assert collapsed_event_fraction(soft) == 0.0
    assert collapsed_event_fraction(hard) > 0.25


# --------------------------------------------------------------------------
# Regression
# --------------------------------------------------------------------------


def test_threshold_target_behaviour_is_unchanged():
    config = _with_control(_config(CLASSICAL_CONFIG, horizon=40), "threshold_target")
    events = _events(_run(config, control=create_control(config.control)))
    for event in events:
        assert event["controller_policy"] == "threshold_target"
        assert event["controller_beta"] is None
        # Deterministic: the action is exactly the threshold comparison.
        advocated = event["controller_action"] == ADVOCATE_TARGET
        assert advocated == (event["sensor_target_share"] < 0.5)
        assert event["controller_advocacy_probability"] == (1.0 if advocated else 0.0)
    assert {event["controller_action"] for event in events} == {ADVOCATE_TARGET, NO_OP}


def test_threshold_target_still_rejects_a_foreign_policy_name():
    with pytest.raises(ConfigurationError):
        ThresholdTargetControl.from_options({"target": "correct", "policy": "soft_target"})
    with pytest.raises(ConfigurationError):
        SoftTargetControl.from_options({"target": "correct", "policy": "threshold_target"})
