import asyncio
import json
from dataclasses import replace

import pytest

from mas_cc.config import load_run_config
from mas_cc.control import ForcedActionControl
from mas_cc.core import AgentId
from mas_cc.games import Action, Game, create_game
from mas_cc.games.naming_convention import (
    InvalidConventionResponse,
    NamingConventionGame,
    parse_convention_response,
    run_naming_convention_game_sync,
)
from mas_cc.llm_providers import CompletionResponse, ProviderCapabilities, ProviderUsage
from mas_cc.prompts import RegexTokenCounter


def _config(*, population_size=6, horizon=12, memory_size=3):
    config = load_run_config(
        "configs/runs/naming_convention_smoke_test.yaml", environment={}
    )
    game = replace(
        config.game,
        population_size=population_size,
        horizon=horizon,
        options={**dict(config.game.options), "memory_size": memory_size},
    )
    return replace(config, game=game)


class ScriptedProvider:
    name = "mock"
    model = "scripted-convention-v1"
    capabilities = ProviderCapabilities(supports_seed=True, reports_usage=True)

    def __init__(self, script=None, delays=None):
        self.script = script or {}
        self.delays = delays or {}
        self.calls = {}
        self.finished = []

    async def complete(self, request):
        agent = str(request.metadata["agent_id"])
        attempt = self.calls.get(agent, 0)
        self.calls[agent] = attempt + 1
        await asyncio.sleep(self.delays.get(agent, 0))
        values = self.script.get(agent, ('{"value":"Q","reason":"default"}',))
        content = values[min(attempt, len(values) - 1)]
        self.finished.append(agent)
        return CompletionResponse(
            content=content,
            provider=self.name,
            model=self.model,
            usage=ProviderUsage(10, 4, 14),
            finish_reason="stop",
            request_id=f"{agent}-{attempt + 1}",
        )

    def close(self):
        pass


def _fixed_pair(game, state, config, first="agent-000", second="agent-001"):
    pair = (AgentId(first), AgentId(second))
    observations = game.construct_observations(state, pair, config.game)
    requests = game.build_decision_requests(state, observations, config.game)
    return pair, requests


def test_registry_exposes_dedicated_repeated_convention_game():
    config = _config()
    game = create_game(config.game)
    assert isinstance(game, NamingConventionGame)
    assert isinstance(game, Game)
    rules = game.rules(config.game)
    assert rules.actions == ("Q", "M")
    assert rules.success_payoff == 100
    assert rules.failure_payoff == -50
    assert rules.stop_on_convergence is False


@pytest.mark.parametrize(
    ("values", "expected_payoff", "success"),
    [(('Q', 'Q'), 100, True), (('Q', 'M'), -50, False)],
)
def test_pure_transition_payoff_perspective_and_selected_pair_only(
    values, expected_payoff, success
):
    config = _config(population_size=3, horizon=2)
    game = create_game(config.game)
    state = game.initialize(config.game, 1)
    pair = (AgentId("agent-000"), AgentId("agent-001"))
    actions = (
        Action(pair[0], values[0], "pair_decision"),
        Action(pair[1], values[1], "pair_decision"),
    )
    transition = game.apply_transition(state, pair, actions, config.game)
    assert state.global_interaction_index == 0
    assert all(len(agent.private_history) == 0 for agent in state.agents)
    assert transition.next_state.global_interaction_index == 1
    assert transition.payoff == expected_payoff
    assert transition.success is success
    first = transition.next_state.convention_agent(pair[0]).private_history[-1]
    second = transition.next_state.convention_agent(pair[1]).private_history[-1]
    assert (first.own_action, first.partner_action) == values
    assert (second.own_action, second.partner_action) == values[::-1]
    assert first.payoff == second.payoff == expected_payoff
    assert len(
        transition.next_state.convention_agent(AgentId("agent-002")).private_history
    ) == 0


def test_invalid_action_cannot_enter_pure_transition():
    config = _config(population_size=2, horizon=1)
    game = create_game(config.game)
    state = game.initialize(config.game, 1)
    pair = (AgentId("agent-000"), AgentId("agent-001"))
    with pytest.raises(ValueError, match="invalid convention action"):
        game.apply_transition(
            state,
            pair,
            (
                Action(pair[0], "Q", "pair_decision"),
                Action(pair[1], "invented", "pair_decision"),
            ),
            config.game,
        )


def test_memory_is_bounded_in_prompt_but_complete_for_evaluator():
    config = _config(population_size=2, horizon=5, memory_size=2)
    game = create_game(config.game)
    state = game.initialize(config.game, 1)
    pair = (AgentId("agent-000"), AgentId("agent-001"))
    for index in range(4):
        values = ("Q", "Q") if index != 1 else ("Q", "M")
        state = game.apply_transition(
            state,
            pair,
            (
                Action(pair[0], values[0], "pair_decision"),
                Action(pair[1], values[1], "pair_decision"),
            ),
            config.game,
        ).next_state
    _, requests = _fixed_pair(game, state, config)
    assert len(state.convention_agent(pair[0]).private_history) == 4
    assert len(requests[0].visible_memory) == 2
    assert requests[0].visible_score == 200
    assert requests[0].local_round == 3


def test_empty_memory_prompt_is_private_and_uses_anonymous_local_roles():
    config = _config(population_size=6, horizon=1)
    game = create_game(config.game)
    state = game.initialize(config.game, 1)
    pair, requests = _fixed_pair(game, state, config)
    prompts = [request.prompt.compile(RegexTokenCounter()) for request in requests]
    for prompt in prompts:
        text = "\n".join(message.content for message in prompt.messages).lower()
        assert "player 1" in text and "player 2" in text
        assert "no past rounds" in text
        for forbidden in (
            "agent-",
            "population size",
            "global interaction",
            "consensus",
            "committee",
            "committed",
        ):
            assert forbidden not in text
        assert str(pair[0]).lower() not in text
        assert str(pair[1]).lower() not in text


def test_action_orders_are_seeded_independent_and_preserve_legal_set():
    config = _config(population_size=2, horizon=4)
    game = create_game(config.game)
    state = game.initialize(config.game, 1)
    pair = (AgentId("agent-000"), AgentId("agent-001"))
    observed = []
    for _ in range(4):
        first = game.construct_observations(state, pair, config.game)
        second = game.construct_observations(state, pair, config.game)
        assert first == second
        orders = tuple(tuple(item.visible_state["presented_actions"]) for item in first)
        assert all(set(order) == {"Q", "M"} for order in orders)
        observed.append(orders)
        state = game.apply_transition(
            state,
            pair,
            (
                Action(pair[0], "Q", "pair_decision"),
                Action(pair[1], "Q", "pair_decision"),
            ),
            config.game,
        ).next_state
    assert any(first != second for first, second in observed)


@pytest.mark.parametrize(
    ("content", "mode"),
    [
        ('{"value":"Q","reason":"coordination"}', "strict_json_reason_v1"),
        ("{'value': 'M', 'reason': 'memory'}", "python_object_reason_v1"),
        ("{'value': 'Q'; 'reason': 'paper'}", "paper_semicolon_reason_v1"),
    ],
)
def test_answer_first_parser_modes(content, mode):
    parsed = parse_convention_response(content, ("Q", "M"))
    assert parsed.value in {"Q", "M"}
    assert parsed.reason
    assert parsed.parser_mode == mode


@pytest.mark.parametrize(
    "content",
    ["I would choose Q because it worked.", '{"value":"Q"}', '{"value":"X","reason":"Q"}'],
)
def test_parser_never_infers_an_action_from_free_form_reasoning(content):
    with pytest.raises(ValueError):
        parse_convention_response(content, ("Q", "M"))


def test_only_invalid_focal_decision_retries_without_state_mutation():
    config = _config(population_size=2, horizon=1)
    provider = ScriptedProvider(
        {
            "agent-000": ("malformed", '{"value":"Q","reason":"retry"}'),
            "agent-001": ('{"value":"Q","reason":"valid"}',),
        }
    )
    game = create_game(config.game)
    result = run_naming_convention_game_sync(game, config, provider)
    assert provider.calls == {"agent-000": 2, "agent-001": 1}
    assert result.logical_decisions == 2
    assert result.validation_attempts == 3
    by_agent = {
        str(decision.request.agent_id): decision.validation_attempts
        for decision in result.interactions[0].decisions
    }
    assert by_agent == {"agent-000": 2, "agent-001": 1}
    assert result.final_state.global_interaction_index == 1
    assert all(len(agent.private_history) == 1 for agent in result.final_state.agents)


def test_retry_exhaustion_raises_typed_error_and_never_transitions():
    config = _config(population_size=2, horizon=1)
    provider = ScriptedProvider(
        {"agent-000": ("bad",), "agent-001": ("also bad",)}
    )
    game = create_game(config.game)
    initial = game.initialize(config.game, config.execution.seed)
    with pytest.raises(InvalidConventionResponse, match="validation attempts"):
        run_naming_convention_game_sync(game, config, provider)
    assert initial.global_interaction_index == 0
    assert all(len(agent.private_history) == 0 for agent in initial.agents)
    assert provider.calls == {"agent-000": 3, "agent-001": 3}


def test_completion_order_does_not_change_simultaneous_transition():
    config = _config(population_size=2, horizon=1)
    script = {
        "agent-000": ('{"value":"Q","reason":"fixed by id"}',),
        "agent-001": ('{"value":"M","reason":"fixed by id"}',),
    }
    slow_first = ScriptedProvider(script, {"agent-000": 0.02, "agent-001": 0})
    slow_second = ScriptedProvider(script, {"agent-000": 0, "agent-001": 0.02})
    first = run_naming_convention_game_sync(create_game(config.game), config, slow_first)
    second = run_naming_convention_game_sync(create_game(config.game), config, slow_second)
    assert slow_first.finished != slow_second.finished
    normalize = lambda result: {
        str(decision.request.agent_id): decision.action.value
        for decision in result.interactions[0].decisions
    }
    assert normalize(first) == normalize(second) == {
        "agent-000": "Q",
        "agent-001": "M",
    }
    assert first.interactions[0].transition.payoff == -50
    assert second.interactions[0].transition.payoff == -50
    assert all(
        not decision.request.visible_memory
        for decision in first.interactions[0].decisions
    )


def test_forced_control_skips_the_provider_and_is_audited():
    config = _config(population_size=2, horizon=3)
    control = ForcedActionControl(
        agent_ids=frozenset({AgentId("agent-000")}), forced_value="Q", until_interaction=2,
    )
    provider = ScriptedProvider(
        {
            "agent-000": ('{"value":"M","reason":"free choice"}',),
            "agent-001": ('{"value":"Q","reason":"match"}',) * 3,
        }
    )
    game = create_game(config.game)
    result = run_naming_convention_game_sync(game, config, provider, control=control)

    # agent-000 is forced for interactions 1-2 (never calls the provider) and
    # free on interaction 3 (exactly one real call).
    assert provider.calls == {"agent-000": 1, "agent-001": 3}
    assert len(result.interactions) == 3
    for interaction in result.interactions[:2]:
        forced_decision = next(
            d for d in interaction.decisions if d.request.agent_id == AgentId("agent-000")
        )
        free_decision = next(
            d for d in interaction.decisions if d.request.agent_id == AgentId("agent-001")
        )
        assert forced_decision.forced is True
        assert forced_decision.action.value == "Q"
        assert free_decision.forced is False
        # A forced decision has no provider attempts, and to_dict() must not
        # crash on that (regression guard for the empty-attempts branch).
        forced_payload = forced_decision.to_dict()
        assert forced_payload["validation"]["forced_decision"] is True
        assert forced_payload["validation"]["attempts"] == []
        assert forced_payload["parsed_action"] == "Q"

    last_interaction = result.interactions[-1]
    last_agent_000 = next(
        d for d in last_interaction.decisions if d.request.agent_id == AgentId("agent-000")
    )
    assert last_agent_000.forced is False
    assert last_agent_000.action.value == "M"

    assert result.to_dict()["counters"]["forced_decisions"] == 2


def test_call_plan_is_stage_retry_memory_and_provider_independent():
    config = _config(population_size=6, horizon=12, memory_size=3)
    game = create_game(config.game)
    plan = game.call_plan(config.game)
    assert plan.logical_decisions.to_dict() == {
        "lower": 24,
        "expected": 24,
        "maximum": 24,
    }
    assert plan.provider_requests.to_dict() == {
        "lower": 24,
        "expected": 24,
        "maximum": 72,
    }
    stage = plan.decision_stages[0]
    assert stage.name == "pair_decision"
    assert stage.concurrency_within_stage == 2
    assert stage.state_barrier_after_stage
    assert stage.retry_bound == 2
    assert {
        len(scenario.bound_prompt.block("visible_memory").value)
        for scenario in stage.prompt_scenarios
    } == {0, 1, 3}
    serialized = json.dumps(plan.to_dict(), sort_keys=True)
    assert '"provider_prices_included": false' in serialized
    assert '"pricing"' not in serialized
