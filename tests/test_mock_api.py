import asyncio

from naming_game.agent import Agent, create_agents
from naming_game.api_client import AsyncLLMClient, MockAsyncLLMClient
from naming_game.sequential_game import SequentialNamingGame


def test_agents_never_share_mutable_histories():
    agents = create_agents(5, 1)
    assert len({id(agent.history) for agent in agents}) == len(agents)
    agents[0].history.append({"private": True})
    assert all(not agent.history for agent in agents[1:])


def test_independent_replicates_get_fresh_instances_and_no_provider_sessions():
    first = create_agents(5, 1)
    second = create_agents(5, 1)
    assert {agent.instance_id for agent in first}.isdisjoint(
        {agent.instance_id for agent in second}
    )
    assert all(agent.provider_session_id is None for agent in first + second)


def test_session_ids_are_unique_when_explicitly_used():
    first = Agent(0, frozenset({"A"}), provider_session_id="rep1-agent0")
    second = Agent(1, frozenset({"B"}), provider_session_id="rep1-agent1")
    assert first.provider_session_id != second.provider_session_id


def test_shared_clients_do_not_have_message_history(monkeypatch):
    mock = MockAsyncLLMClient()
    real = AsyncLLMClient(
        model="model",
        api_key="fake-key",
        base_url="https://example.invalid",
    )
    try:
        for client in (mock, real):
            assert not hasattr(client, "messages")
            assert not hasattr(client, "history")
            assert not hasattr(client, "conversations")
    finally:
        real.close()


def test_evaluator_history_is_never_inserted_into_later_prompts():
    prompts = []
    client = MockAsyncLLMClient(
        artificial_latency=0,
        request_observer=lambda messages: prompts.append(messages[-1]["content"]),
    )
    agents = create_agents(5, 2)
    asyncio.run(
        SequentialNamingGame(agents=agents, client=client, seed=2).run(3)
    )
    assert any(agent.history for agent in agents)
    assert all("counterpart_id" not in prompt for prompt in prompts)
    assert all("interaction_index" not in prompt for prompt in prompts)
    assert all("history" not in prompt.lower() for prompt in prompts)

