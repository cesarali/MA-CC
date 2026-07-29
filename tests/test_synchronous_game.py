import asyncio
import json

from naming_game.agent import create_agents
from naming_game.api_client import MockAsyncLLMClient
from naming_game.synchronous_game import SynchronousParallelNamingGame


def run_game(num_agents=5, rounds=4, seed=7, latency=0):
    agents = create_agents(num_agents, seed)
    client = MockAsyncLLMClient(
        concurrency=20, artificial_latency=latency, seed=seed
    )
    result = asyncio.run(
        SynchronousParallelNamingGame(agents=agents, client=client, seed=seed).run(rounds)
    )
    return agents, client, result


def test_pairs_are_disjoint_and_odd_population_has_one_idle_agent():
    _, _, result = run_game(num_agents=5, rounds=5)
    for round_record in result.rounds:
        round_pairs = [
            item
            for item in result.interactions
            if item.round_index == round_record.round_index
        ]
        participants = [
            agent_id
            for item in round_pairs
            for agent_id in (item.speaker_id, item.listener_id)
        ]
        assert len(participants) == len(set(participants)) == 4
        assert round_record.idle_agent_id not in participants
        assert round_record.parallel_pairs == 2


def test_all_pairs_read_one_start_of_round_snapshot():
    seed = 11
    initial_agents = create_agents(6, seed)
    initial = {agent.agent_id: agent.inventory for agent in initial_agents}
    _, _, result = run_game(num_agents=6, rounds=1, seed=seed)
    for interaction in result.interactions:
        assert interaction.speaker_before == initial[interaction.speaker_id]
        assert interaction.listener_before == initial[interaction.listener_id]


def test_no_live_update_is_visible_before_round_barrier():
    agents = create_agents(4, 1)
    before = [agent.inventory for agent in agents]
    listeners_started = asyncio.Event()
    release = asyncio.Event()
    listener_count = 0
    default_client = MockAsyncLLMClient(artificial_latency=0, seed=1)

    async def delayed_response(messages):
        nonlocal listener_count
        if "listener_basic" in messages[-1]["content"]:
            listener_count += 1
            if listener_count == 2:
                listeners_started.set()
            await release.wait()
        return default_client._default_response(messages)

    client = MockAsyncLLMClient(
        artificial_latency=0,
        seed=1,
        response_factory=delayed_response,
    )

    async def scenario():
        task = asyncio.create_task(
            SynchronousParallelNamingGame(agents=agents, client=client, seed=1).run(1)
        )
        await asyncio.wait_for(listeners_started.wait(), timeout=1)
        assert [agent.inventory for agent in agents] == before
        assert all(not agent.history for agent in agents)
        release.set()
        await task

    asyncio.run(scenario())


def test_completion_order_does_not_change_seeded_result():
    def latency_forward(messages):
        user = messages[-1]["content"]
        agent_id = int(next(line for line in user.splitlines() if line.startswith("AGENT_ID:")).split(":")[1])
        return agent_id * 0.0002

    def latency_reverse(messages):
        user = messages[-1]["content"]
        agent_id = int(next(line for line in user.splitlines() if line.startswith("AGENT_ID:")).split(":")[1])
        return (10 - agent_id) * 0.0002

    async def execute(latency):
        agents = create_agents(8, 23)
        client = MockAsyncLLMClient(seed=23, artificial_latency=latency)
        result = await SynchronousParallelNamingGame(
            agents=agents, client=client, seed=23
        ).run(5)
        return (
            [agent.inventory for agent in agents],
            [
                (item.speaker_id, item.listener_id, item.selected_name)
                for item in result.interactions
            ],
        )

    assert asyncio.run(execute(latency_forward)) == asyncio.run(execute(latency_reverse))


def test_seeded_pairing_is_reproducible():
    first = run_game(num_agents=8, rounds=5, seed=71)[2]
    second = run_game(num_agents=8, rounds=5, seed=71)[2]
    assert [
        (item.round_index, item.speaker_id, item.listener_id)
        for item in first.interactions
    ] == [
        (item.round_index, item.speaker_id, item.listener_id)
        for item in second.interactions
    ]


def test_pair_prompts_contain_only_pair_local_state():
    observed = []

    def observer(messages):
        observed.append(messages[-1]["content"])

    agents = create_agents(6, 5)
    client = MockAsyncLLMClient(
        artificial_latency=0, seed=5, request_observer=observer
    )
    asyncio.run(
        SynchronousParallelNamingGame(agents=agents, client=client, seed=5).run(1)
    )
    assert len(observed) == 6
    for prompt in observed:
        assert prompt.count("AGENT_ID:") == 1
        assert prompt.count("INVENTORY_JSON:") == 1
        assert "population" not in prompt.lower()
        assert "history" not in prompt.lower()
        assert "round" not in prompt.lower()

