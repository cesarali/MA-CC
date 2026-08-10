"""Acceptance tests for matched reasoning/classical HiddenBench imitation."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace

import pytest

from mas_cc.config import load_run_config
from mas_cc.control import NoneControl, create_control
from mas_cc.games import create_game
from mas_cc.games.hidden_bench.data import DEFAULT_CORPUS_ROOT
from mas_cc.games.hidden_bench.imitation import (
    HiddenBenchImitationGame,
    run_hidden_bench_imitation_game,
)
from mas_cc.games.hidden_bench.imitation.metrics import population_observables
from mas_cc.games.registry import create_default_game_registry
from mas_cc.llm_runtime.providers import create_llm_provider
from mas_cc.llm_runtime.providers.adapters.mock import MockLLMProvider

pytestmark = pytest.mark.skipif(
    not (DEFAULT_CORPUS_ROOT / "canonical" / "tasks.json").exists(),
    reason="HiddenBench corpus is not present",
)

CLASSICAL_CONFIG = "configs/runs/hidden_bench/hidden_bench_imitation_classical.yaml"
REASONING_CONFIG = "configs/runs/hidden_bench/hidden_bench_imitation_reasoning_mock.yaml"


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


def _run(config, *, provider=None, control=None):
    game = create_game(config.game)
    provider = provider or create_llm_provider(config.llm_provider)
    return asyncio.run(
        run_hidden_bench_imitation_game(
            game, config, provider, control=control
        )
    )


def test_game_is_registered_and_shipped_configs_load():
    assert "hidden_bench_imitation" in create_default_game_registry().names()
    assert _config(CLASSICAL_CONFIG).game.type == "hidden_bench_imitation"
    assert _config(REASONING_CONFIG).game.type == "hidden_bench_imitation"


def test_horizon_is_the_only_step_limit():
    config = _config(CLASSICAL_CONFIG, horizon=3)
    assert len(_run(config, control=NoneControl()).interactions) == 3

    legacy = replace(
        config,
        game=replace(
            config.game,
            options={**dict(config.game.options), "interactions": 1},
        ),
    )
    with pytest.raises(ValueError, match="use game.horizon"):
        HiddenBenchImitationGame().rules(legacy.game)


def test_order_parameters_are_correct_for_uniform_three_and_four_option_states():
    three = population_observables(["A", "B", "C"], ["A", "B", "C"], "A", "B")
    four = population_observables(
        ["A", "B", "C", "D"], ["A", "B", "C", "D"], "A", "D"
    )
    for value in (three, four):
        assert value["m_truth"] == pytest.approx(0.0)
        assert value["m_ctrl"] == pytest.approx(0.0)
        assert value["m_order"] == pytest.approx(0.0)
        assert value["H_vote"] == pytest.approx(1.0)


def test_classical_mode_is_provider_free_deterministic_and_one_focal_per_jump():
    calls = 0

    def forbidden(_request):
        nonlocal calls
        calls += 1
        raise AssertionError("classical mode called the provider")

    config = _config(CLASSICAL_CONFIG, horizon=12)
    provider = MockLLMProvider(config.llm_provider, response_factory=forbidden)
    first = _run(config, provider=provider, control=create_control(config.control))
    second = _run(
        config,
        provider=MockLLMProvider(config.llm_provider, response_factory=forbidden),
        control=create_control(config.control),
    )
    assert calls == 0
    assert first.logical_decisions == 0
    assert first.initial_state.initial_votes == (
        "West City", "East Town", "North Hill", "East Town"
    )
    assert json.dumps(first.to_dict(), sort_keys=True) == json.dumps(
        second.to_dict(), sort_keys=True
    )
    for interaction in first.interactions:
        event = interaction.transition.event
        before = event["population_state_before"]
        after = event["population_state_after"]
        assert sum(left != right for left, right in zip(before, after)) == 1
        assert event["classical_source_opinion"] != event["classical_destination_opinion"]
        assert event["classical_candidate_channels"]


def test_classical_supports_a_four_option_task_end_to_end():
    options = (
        "Deep Jungle Site (A)",
        "Riverbank Clearing (B)",
        "Hilltop Overlook (C)",
        "Plateau Camp (D)",
    )
    config = _config(
        CLASSICAL_CONFIG,
        task_id="scientists_animal_base_decision",
        horizon=5,
        initialization={"mode": "explicit", "initial_votes": list(options)},
    )
    result = _run(config, control=NoneControl())
    assert len(result.final_state.possible_answers) == 4
    assert len(result.interactions) == 5
    assert all(item.transition.event["K"] == 4 for item in result.interactions)


def test_exact_replication_initializes_the_frozen_scaled_population():
    config = _config(
        CLASSICAL_CONFIG,
        task_set="expanded",
        assignment_scheme="exact_replication",
        n_agents=32,
        horizon=1,
        initialization={
            "mode": "distribution",
            "initial_votes": None,
            "initial_distribution": {"West City": 1.0},
        },
    )
    config = replace(config, game=replace(config.game, population_size=32))
    state = HiddenBenchImitationGame().initialize(config.game, config.execution.seed)
    assert len(state.agents) == 32
    assert state.data["assignment"]


def test_reasoning_initializes_every_agent_before_events_and_only_focal_vote_changes():
    config = _config(REASONING_CONFIG, horizon=3)
    result = _run(config)
    assert len(result.initial_decisions) == config.game.population_size
    assert all(agent.committed_action is not None for agent in result.initial_state.agents)
    assert result.initial_state.turn == 0
    for interaction in result.interactions:
        event = interaction.transition.event
        before = event["population_state_before"]
        after = event["population_state_after"]
        changed = [index for index, pair in enumerate(zip(before, after)) if pair[0] != pair[1]]
        assert len(changed) <= 1
        if changed:
            assert str(result.initial_state.agents[changed[0]].agent_id) == event["focal_agent_id"]
    assert result.logical_decisions == result.initial_state.data["rules"]["n_agents"] + 3
    assert result.logical_decisions == create_game(config.game).call_plan(config.game).provider_requests.lower


def test_private_observation_contains_only_the_agents_assigned_evidence():
    config = _config(REASONING_CONFIG, horizon=1)
    game = HiddenBenchImitationGame()
    state = game.initialize(config.game, config.execution.seed)
    focal = state.agents[0].agent_id
    visible = game._visible_state(state, focal)
    own = set(state.hidden_bench_agent(focal).presented_information)
    assert set(visible["presented_information"]) == own
    all_other_private = {
        fact
        for agent in state.agents[1:]
        for fact in agent.private_information
        if fact not in own
    }
    rendered = json.dumps(visible)
    assert all(fact not in rendered for fact in all_other_private)
    assert "correct_answer" not in rendered


def test_controller_measurement_drives_both_actions_and_logs_feedback_tuple():
    config = _config(CLASSICAL_CONFIG, horizon=20)
    result = _run(config, control=create_control(config.control))
    actions = {item.transition.event["controller_action"] for item in result.interactions}
    assert actions == {"NO_OP", "ADVOCATE_Z"}
    for item in result.interactions:
        event = item.transition.event
        assert sum(event["sensor_count_vector"].values()) == event["sensor_sample_size"]
        assert len(event["population_state_before"]) == event["N"]
        assert len(event["population_state_after"]) == event["N"]
        assert event["controller_policy"] == "threshold_target"


def test_reasoning_controller_advocacy_never_forces_the_vote():
    base = _config(REASONING_CONFIG)
    wrong = "East Town"
    config = replace(
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
        control=replace(
            base.control,
            mechanism="threshold_target",
            options={"target": "correct", "sensor_sample_size": 1, "threshold": 0.5},
        ),
        llm_provider=replace(
            base.llm_provider,
            options={"response": '{"vote": "East Town", "rationale": "I reject it"}'},
        ),
    )
    config = replace(config, game=replace(config.game, horizon=1))
    result = _run(config, control=create_control(config.control))
    event = result.interactions[0].transition.event
    assert event["controller_action"] == "ADVOCATE_Z"
    assert event["controller_target"] == "West City"
    assert event["focal_opinion_after"] == wrong


def test_none_control_is_exactly_the_uncontrolled_path():
    config = _config(CLASSICAL_CONFIG, horizon=8)
    config = replace(config, control=replace(config.control, mechanism="none", options={}))
    without_argument = _run(config)
    explicit_none = _run(config, control=NoneControl())
    assert json.dumps(without_argument.to_dict(), sort_keys=True) == json.dumps(
        explicit_none.to_dict(), sort_keys=True
    )
    assert all(
        item.transition.event["controller_enabled"] is False
        for item in without_argument.interactions
    )
