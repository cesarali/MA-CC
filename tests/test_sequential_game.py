import asyncio

from naming_game.agent import create_agents
from naming_game.api_client import MockAsyncLLMClient
from naming_game.sequential_game import SequentialNamingGame


def run_game(num_agents=5, interactions=10, seed=7, latency=0):
    agents = create_agents(num_agents, seed)
    client = MockAsyncLLMClient(
        concurrency=20, artificial_latency=latency, seed=seed
    )
    result = asyncio.run(
        SequentialNamingGame(agents=agents, client=client, seed=seed).run(interactions)
    )
    return agents, client, result


def test_exact_interaction_budget_and_no_internal_concurrency():
    _, client, result = run_game(interactions=12, latency=0.001)
    assert len(result.interactions) == 12
    assert len(result.states) == 12
    assert client.stats["actual_calls"] == 24
    assert client.max_active_requests == 1


def test_updates_are_immediate_and_each_next_pair_sees_live_state():
    seed = 3
    initial = create_agents(6, seed)
    expected_state = {agent.agent_id: agent.inventory for agent in initial}
    _, _, result = run_game(num_agents=6, interactions=20, seed=seed)
    for interaction in result.interactions:
        assert interaction.speaker_before == expected_state[interaction.speaker_id]
        assert interaction.listener_before == expected_state[interaction.listener_id]
        expected_state[interaction.speaker_id] = interaction.speaker_after
        expected_state[interaction.listener_id] = interaction.listener_after


def test_seeded_ordered_pair_sampling_is_reproducible():
    first = run_game(num_agents=7, interactions=15, seed=99)[2]
    second = run_game(num_agents=7, interactions=15, seed=99)[2]
    first_pairs = [(item.speaker_id, item.listener_id) for item in first.interactions]
    second_pairs = [(item.speaker_id, item.listener_id) for item in second.interactions]
    assert first_pairs == second_pairs


def test_only_participants_receive_private_history_entries():
    agents, _, result = run_game(num_agents=5, interactions=1, seed=4)
    participants = {
        result.interactions[0].speaker_id,
        result.interactions[0].listener_id,
    }
    for agent in agents:
        assert len(agent.history) == (1 if agent.agent_id in participants else 0)

