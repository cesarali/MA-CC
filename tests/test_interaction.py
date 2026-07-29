import asyncio
import json

import pytest

from naming_game.api_client import MockAsyncLLMClient
from naming_game.interaction import basic_naming_update, execute_pair_interaction
from naming_game.models import AgentSnapshot, normalize_inventory


def run(coroutine):
    return asyncio.run(coroutine)


@pytest.mark.parametrize(
    "value",
    [{"A"}, {"B"}, {"A", "B"}, "{A}", "{B}", "{A, B}"],
)
def test_only_three_inventories_are_valid(value):
    assert normalize_inventory(value) in {
        frozenset({"A"}),
        frozenset({"B"}),
        frozenset({"A", "B"}),
    }


@pytest.mark.parametrize("value", [set(), {"C"}, {"A", "C"}, {"A", "B", "C"}])
def test_invalid_inventories_are_rejected(value):
    with pytest.raises(ValueError, match="Inventory"):
        normalize_inventory(value)


def test_success_update_collapses_both_inventories():
    speaker, listener, success = basic_naming_update(
        frozenset({"A", "B"}), frozenset({"A"}), "A"
    )
    assert success is True
    assert speaker == listener == frozenset({"A"})


def test_failure_update_changes_only_listener():
    speaker, listener, success = basic_naming_update(
        frozenset({"A"}), frozenset({"B"}), "A"
    )
    assert success is False
    assert speaker == frozenset({"A"})
    assert listener == frozenset({"A", "B"})


def test_malformed_json_is_logged_and_repaired_locally():
    client = MockAsyncLLMClient(
        artificial_latency=0,
        response_factory=lambda messages: "not-json",
    )
    result = run(
        execute_pair_interaction(
            client=client,
            speaker=AgentSnapshot(0, frozenset({"A"})),
            listener=AgentSnapshot(1, frozenset({"B"})),
            interaction_index=1,
            round_index=None,
            pair_index=None,
            interaction_kind="basic",
            choice_seed=1,
            temperature=0,
            max_tokens_speaker=20,
            max_tokens_listener=20,
        )
    )
    assert result.selected_name == "A"
    assert result.speaker_response_valid is False
    assert result.listener_response_valid is False
    assert "repaired" in result.speaker_validation_error
    assert "engine truth" in result.listener_validation_error
    assert result.listener_after == frozenset({"A", "B"})


def test_listener_disagreement_is_invalid_and_engine_remains_authoritative():
    def responses(messages):
        action = messages[-1]["content"]
        if "speaker_basic" in action:
            return json.dumps({"selected_name": "A"})
        return json.dumps({"already_known": True})

    result = run(
        execute_pair_interaction(
            client=MockAsyncLLMClient(artificial_latency=0, response_factory=responses),
            speaker=AgentSnapshot(0, frozenset({"A"})),
            listener=AgentSnapshot(1, frozenset({"B"})),
            interaction_index=1,
            round_index=None,
            pair_index=None,
            interaction_kind="basic",
            choice_seed=1,
            temperature=0,
            max_tokens_speaker=20,
            max_tokens_listener=20,
        )
    )
    assert result.listener_reported_known is True
    assert result.engine_already_known is False
    assert result.listener_response_valid is False
    assert result.naming_success is False
    assert result.listener_after == frozenset({"A", "B"})


def test_each_interaction_makes_exactly_two_logical_api_calls():
    client = MockAsyncLLMClient(artificial_latency=0)
    run(
        execute_pair_interaction(
            client=client,
            speaker=AgentSnapshot(0, frozenset({"A"})),
            listener=AgentSnapshot(1, frozenset({"B"})),
            interaction_index=1,
            round_index=None,
            pair_index=None,
            interaction_kind="basic",
            choice_seed=1,
            temperature=0,
            max_tokens_speaker=20,
            max_tokens_listener=20,
        )
    )
    assert client.stats["actual_calls"] == 2
    assert client.stats["successful_calls"] == 2

