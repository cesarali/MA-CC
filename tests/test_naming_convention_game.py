import asyncio
import json

import pytest

from naming_game.api_client import MockAsyncLLMClient
from naming_game.local_model_types import ChoiceScore, ConstrainedDecisionResponse
from naming_game.models import LLMResponse, TokenUsage
from naming_game.naming_convention_game import (
    ConventionAgent,
    ConventionGameConfig,
    InvalidConventionResponse,
    NamingConventionGame,
    build_convention_messages,
    parse_convention_decision,
)


def always(action):
    return lambda messages: json.dumps({"value": action, "reason": "coordinate"})


def test_prompt_uses_only_bounded_private_history_and_local_score():
    agent = ConventionAgent(agent_id=7)
    agent.remember(
        interaction_index=11,
        own_action="Q",
        partner_action="M",
        payoff=-50,
        partner_id=2,
    )
    agent.remember(
        interaction_index=19,
        own_action="M",
        partner_action="M",
        payoff=100,
        partner_id=4,
    )
    messages = build_convention_messages(
        agent=agent,
        action_order=("M", "Q"),
        memory_size=1,
        success_reward=100,
        failure_payoff=-50,
    )
    prompt = messages[0]["content"]
    assert '"Player 1": "M"' in prompt
    assert '"Player 1": "Q"' not in prompt
    assert "current score of Player 1 is 100" in prompt
    assert "now round 2" in prompt
    assert "interaction_index" not in prompt
    assert "agent_id" not in prompt.lower()
    assert "population" not in prompt.lower()
    assert "Player 2 for 100 rounds" in prompt


def test_parser_accepts_json_and_original_python_mapping_shape():
    assert parse_convention_decision('{"value":"Q","reason":"x"}', ("Q", "M")) == (
        "Q",
        "x",
    )
    original_shape = "{'value': 'M'; 'reason': 'x'}".replace(";", ",")
    assert parse_convention_decision(original_shape, ("Q", "M")) == ("M", "x")
    with pytest.raises(ValueError, match="configured actions"):
        parse_convention_decision('{"value":"A"}', ("Q", "M"))


def test_pair_choices_are_simultaneous_and_history_is_updated_after_both_calls():
    histories_seen = []

    def observer(messages):
        histories_seen.append("history of choices" in messages[0]["content"])

    client = MockAsyncLLMClient(
        artificial_latency=0.001,
        response_factory=always("Q"),
        request_observer=observer,
    )
    game = NamingConventionGame(
        client=client,
        config=ConventionGameConfig(num_agents=2, actions=("Q", "M")),
        seed=3,
    )
    result = asyncio.run(game.run(1, stop_on_convergence=False))
    record = result.interactions[0]
    assert histories_seen == [False, False]
    assert client.max_active_requests == 2
    assert record.success is True
    assert record.payoff == 100
    assert all(len(agent.history) == 1 for agent in game.agents.values())
    assert all(agent.score == 100 for agent in game.agents.values())


def test_existing_default_mock_client_supports_convention_prompts():
    client = MockAsyncLLMClient(artificial_latency=0, seed=9)
    game = NamingConventionGame(
        client=client,
        config=ConventionGameConfig(num_agents=2, actions=("Q", "M")),
        seed=9,
    )
    result = asyncio.run(game.run(1, stop_on_convergence=False))
    assert len(result.interactions) == 1
    assert client.stats["actual_calls"] == 2
    assert {
        result.interactions[0].player_1_action,
        result.interactions[0].player_2_action,
    } <= {"Q", "M"}


def test_run_stops_after_reference_three_n_success_window():
    client = MockAsyncLLMClient(
        artificial_latency=0,
        response_factory=always("Q"),
    )
    game = NamingConventionGame(
        client=client,
        config=ConventionGameConfig(num_agents=2, actions=("Q", "M")),
        seed=5,
    )
    result = asyncio.run(game.run(50))
    assert result.converged is True
    assert result.convention == "Q"
    assert len(result.interactions) == 6
    assert result.convergence_interaction_index == 6
    assert client.stats["actual_calls"] == 12


def test_each_run_uses_a_fresh_stage_convergence_window():
    client = MockAsyncLLMClient(artificial_latency=0, response_factory=always("Q"))
    game = NamingConventionGame(
        client=client,
        config=ConventionGameConfig(num_agents=2, actions=("Q", "M")),
    )
    first = asyncio.run(game.run(6))
    second = asyncio.run(game.run(1))
    assert first.converged is True
    assert second.converged is False
    assert second.start_interaction_index == 7
    assert second.end_interaction_index == 7


def test_population_round_callback_reports_completed_rounds():
    client = MockAsyncLLMClient(artificial_latency=0, response_factory=always("Q"))
    game = NamingConventionGame(
        client=client,
        config=ConventionGameConfig(num_agents=2, actions=("Q", "M")),
    )
    completed_rounds = []
    result = asyncio.run(
        game.run(
            5,
            stop_on_convergence=False,
            population_round_callback=completed_rounds.append,
        )
    )
    assert len(result.interactions) == 5
    assert completed_rounds == [1, 2]


def test_committed_agents_skip_api_requests():
    client = MockAsyncLLMClient(artificial_latency=0, response_factory=always("Q"))
    game = NamingConventionGame(
        client=client,
        config=ConventionGameConfig(num_agents=2, actions=("Q", "M")),
        seed=2,
    )
    selected = game.introduce_committed_minority(size=2, action="M", mode="swap")
    result = asyncio.run(game.run(1, stop_on_convergence=False))
    assert selected == (1, 2)
    assert client.stats["actual_calls"] == 0
    assert result.interactions[0].player_1_action == "M"
    assert result.interactions[0].player_2_action == "M"


def test_consensus_initialization_is_memory_only_not_a_tracked_interaction():
    game = NamingConventionGame(
        client=MockAsyncLLMClient(artificial_latency=0, response_factory=always("Q")),
        config=ConventionGameConfig(
            num_agents=3, actions=("Q", "M"), memory_size=5
        ),
    )
    game.seed_consensus_history("Q")
    assert game.interactions == []
    assert all(len(agent.history) == 5 for agent in game.agents.values())
    assert all(agent.score == 500 for agent in game.agents.values())


def test_invalid_responses_are_retried_but_never_silently_repaired():
    client = MockAsyncLLMClient(
        artificial_latency=0,
        response_factory=lambda messages: '{"value":"not-in-pool"}',
    )
    game = NamingConventionGame(
        client=client,
        config=ConventionGameConfig(
            num_agents=2,
            actions=("Q", "M"),
            invalid_response_retries=1,
        ),
    )
    with pytest.raises(InvalidConventionResponse):
        asyncio.run(game.run(1))
    assert client.stats["actual_calls"] == 4
    assert game.interactions == []
    assert all(not agent.history for agent in game.agents.values())


@pytest.mark.parametrize(
    ("output_format", "content", "reason"),
    (
        ("json_reason", '{"value":"Q","reason":"json"}', "json"),
        ("choice_reason", "Q\nReason: text", "text"),
        ("choice_only", "Q", None),
    ),
)
def test_ordinary_clients_use_strict_generated_text_for_all_formats(
    output_format, content, reason
):
    client = MockAsyncLLMClient(
        artificial_latency=0, response_factory=lambda messages: content
    )
    game = NamingConventionGame(
        client=client,
        config=ConventionGameConfig(
            num_agents=2,
            actions=("Q", "M"),
            decision_output_format=output_format,
        ),
        seed=4,
    )
    record = asyncio.run(game.run(1, stop_on_convergence=False)).interactions[0]
    assert record.player_1_decision.reason == reason
    assert record.player_2_decision.reason == reason
    assert record.player_1_decision.constrained_scores is None
    assert client.stats["actual_calls"] == 2


class CombinedConventionClient:
    model = "fake/combined"
    provider_name = "not-gemma"
    concurrency = 4

    def __init__(self, *, malformed_reason=False):
        self.calls = []
        self.complete_calls = 0
        self.active = 0
        self.maximum = 0
        self.malformed_reason = malformed_reason

    async def complete_decision(self, messages, *, choices, **kwargs):
        self.active += 1
        self.maximum = max(self.maximum, self.active)
        await asyncio.sleep(0.005)
        self.active -= 1
        self.calls.append((messages, tuple(choices), kwargs))
        scores = tuple(
            ChoiceScore(choice, (index + 1,), 0.0, 0.5)
            for index, choice in enumerate(choices)
        )
        selected = choices[0]
        reason = None if kwargs["output_format"] == "choice_only" else "bad B"
        valid = None if kwargs["output_format"] == "choice_only" else not self.malformed_reason
        if self.malformed_reason:
            reason = None
        content = selected if reason is None else f"{selected}\nReason: {reason}"
        return ConstrainedDecisionResponse(
            selected,
            scores,
            content,
            reason,
            valid,
            kwargs["output_format"],
            self.model,
            0.005,
            TokenUsage(4, 1, 5),
            kwargs["choice_temperature"],
            kwargs["selection_policy"],
        )

    async def complete(self, messages, **kwargs):
        self.complete_calls += 1
        return LLMResponse("unused", self.model, 0.0)

    def close(self):
        pass


@pytest.mark.parametrize("output_format", ("choice_reason", "choice_only"))
def test_combined_capability_is_provider_agnostic_concurrent_and_preserves_order_temperature(
    output_format,
):
    client = CombinedConventionClient()
    game = NamingConventionGame(
        client=client,
        config=ConventionGameConfig(
            num_agents=2,
            actions=("Q", "M"),
            decision_output_format=output_format,
            choice_temperature=0.5,
        ),
        seed=12,
    )
    record = asyncio.run(game.run(1, stop_on_convergence=False)).interactions[0]
    assert client.complete_calls == 0
    assert len(client.calls) == 2
    assert client.maximum == 2
    for _, choices, kwargs in client.calls:
        assert choices in (("Q", "M"), ("M", "Q"))
        assert kwargs["choice_temperature"] == 0.5
    assert tuple(score.choice for score in record.player_1_decision.constrained_scores) == record.player_1_decision.action_order
    assert tuple(score.choice for score in record.player_2_decision.constrained_scores) == record.player_2_decision.action_order
    assert record.player_1_action == record.player_1_decision.action_order[0]
    assert record.player_2_action == record.player_2_decision.action_order[0]
    assert all(
        "Reason:" not in entry.own_action and "Reason:" not in entry.partner_action
        for agent in game.agents.values()
        for entry in agent.history
    )


def test_malformed_combined_reasons_do_not_change_state_or_payoff():
    client = CombinedConventionClient(malformed_reason=True)
    game = NamingConventionGame(
        client=client,
        config=ConventionGameConfig(
            num_agents=2,
            actions=("Q", "M"),
            decision_output_format="choice_reason",
        ),
        seed=6,
    )
    record = asyncio.run(game.run(1, stop_on_convergence=False)).interactions[0]
    assert record.player_1_decision.reason is None
    assert record.player_1_decision.reason_valid is False
    assert record.player_1_action == record.player_1_decision.action_order[0]
    assert record.payoff in (100, -50)


def test_forced_decisions_have_no_rationale_and_make_no_model_request():
    class AlwaysForced:
        def action_for(self, agent_id, interaction_index):
            return "Q"

        def window_active(self, interaction_index):
            return True

    client = CombinedConventionClient()
    game = NamingConventionGame(
        client=client,
        config=ConventionGameConfig(
            num_agents=2,
            actions=("Q", "M"),
            decision_output_format="choice_reason",
        ),
        intervention=AlwaysForced(),
    )
    record = asyncio.run(game.run(1, stop_on_convergence=False)).interactions[0]
    assert not client.calls and client.complete_calls == 0
    assert record.player_1_decision.reason is None
    assert record.player_2_decision.reason is None
