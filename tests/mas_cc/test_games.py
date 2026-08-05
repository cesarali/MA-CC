from dataclasses import replace

import pytest

from mas_cc.config import LLMProviderConfig, load_run_config
from mas_cc.core import AgentId
from mas_cc.games import Game, create_default_game_registry, create_game, run_game_sync
from mas_cc.llm_runtime.providers import OfflinePricingSource, create_llm_provider
from mas_cc.planning import static_game_preflight


def _config():
    return load_run_config("configs/runs/toy_game_smoke_test.yaml", environment={})


def test_default_registry_constructs_a_generic_game_lazily():
    config = _config()
    registry = create_default_game_registry()
    assert registry.names() == (
        "naming_convention",
        "synthetic_bernoulli",
        "synthetic_controlled_markov",
        "synthetic_markov",
        "toy_coordination",
    )
    game = registry.create(config.game)
    assert isinstance(game, Game)
    assert game.spec.game_type == "toy_coordination"
    with pytest.raises(ValueError, match="already registered"):
        registry.register("toy_coordination", lambda: game)


def test_initial_state_is_immutable_and_transition_is_pure():
    config = _config()
    game = create_game(config.game)
    initial = game.initialize(config.game, config.execution.seed)
    rng = __import__("random").Random(7)
    participants = game.select_participants(initial, config.game, rng)
    observations = game.construct_observations(initial, participants, config.game)
    requests = game.build_decision_requests(initial, observations, config.game)
    actions = tuple(game.parse_action(request, "A") for request in requests)
    assert all(
        game.validate_action(initial, request, action, config.game).is_valid
        for request, action in zip(requests, actions, strict=True)
    )

    transition = game.apply_transition(initial, participants, actions, config.game)
    assert initial.turn == 0
    assert all(agent.score == 0 for agent in initial.agents)
    assert transition.next_state.turn == 1
    assert transition.matched is True
    assert all(
        transition.next_state.agent(agent_id).score == 1 for agent_id in participants
    )
    with pytest.raises(TypeError):
        initial.data["matches"] = 99
    with pytest.raises(TypeError):
        initial.agent(AgentId("agent-000")).attributes["available_actions"][0] = "B"


def test_action_validation_rejects_values_outside_game_rules():
    config = _config()
    game = create_game(config.game)
    state = game.initialize(config.game, config.execution.seed)
    participants = (AgentId("agent-000"), AgentId("agent-001"))
    request = game.build_decision_requests(
        state, game.construct_observations(state, participants, config.game), config.game
    )[0]
    action = game.parse_action(request, "choose A")
    result = game.validate_action(state, request, action, config.game)
    assert not result.is_valid
    assert result.issues[0].field == "action.value"


def test_call_plan_is_provider_independent_and_bounds_prompt_memory():
    config = _config()
    game = create_game(config.game)
    plan = game.call_plan(config.game)
    changed_provider = replace(
        config,
        llm_provider=LLMProviderConfig(type="openai", model="gpt-4o-mini"),
    )
    assert game.call_plan(changed_provider.game) == plan
    assert plan.interactions.fixed == config.game.horizon
    assert plan.provider_requests.lower == 6
    assert plan.provider_requests.expected == 6
    assert plan.provider_requests.maximum == 6
    stage = plan.decision_stages[0]
    assert stage.requests_per_interaction == 2
    assert len(stage.representative_prompt.bound_prompt.block("visible_memory").value) == 0
    assert (
        len(stage.maximum_prompt.bound_prompt.block("visible_memory").value)
        == config.game.horizon - 1
    )
    assert "pricing" not in plan.to_dict()


def test_game_plan_composes_with_phase_4_pricing():
    config = _config()
    game = create_game(config.game)
    plan = game.call_plan(config.game)
    quote = OfflinePricingSource().fetch(
        config.llm_provider.type, config.llm_provider.model
    )
    estimate = static_game_preflight(
        plan,
        config.prompt,
        config.llm_provider,
        pricing_quote=quote,
        assumed_output_tokens=1,
    )
    assert estimate.provider_requests.to_dict() == {
        "lower": 6,
        "expected": 6,
        "conservative": 6,
    }
    assert estimate.input_tokens.conservative > estimate.input_tokens.expected
    assert estimate.costs.expected is not None
    assert estimate.costs.expected.amount == 0
    assert estimate.launch_status == "permitted"


def test_generic_runner_is_reproducible_and_records_complete_decisions():
    config = _config()

    def once():
        game = create_game(config.game)
        provider = create_llm_provider(config.llm_provider)
        try:
            return run_game_sync(game, config, provider)
        finally:
            provider.close()

    first = once()
    second = once()
    assert first.to_dict() == second.to_dict()
    assert first.final_state.terminated
    assert first.termination_reason == "finite_horizon_reached"
    assert len(first.interactions) == config.game.horizon
    assert all(len(interaction.decisions) == 2 for interaction in first.interactions)
    first_decision = first.interactions[0].decisions[0].to_dict()
    assert first_decision["decision_request"]["observation"]
    assert first_decision["completion_request"]["messages"]
    assert first_decision["response"]["content"] == "A"
    assert first_decision["action"]["value"] == "A"
