"""Round-synchronous Naming Game with concurrent disjoint pairs."""

from __future__ import annotations

import asyncio
import random
import time

from .agent import Agent
from .api_client import LLMClient
from .interaction import execute_pair_interaction
from .models import (
    AgentSnapshot,
    ConfigurationError,
    GameResult,
    RoundRecord,
    has_consensus,
    population_counts,
)
from .reasoning_game import ReasoningTask


class SynchronousParallelNamingGame:
    """Apply disjoint pair results at one simultaneous end-of-round barrier."""

    def __init__(
        self,
        *,
        agents: list[Agent],
        client: LLMClient,
        seed: int,
        reasoning_fraction: float = 0.0,
        reasoning_task: ReasoningTask | None = None,
        temperature: float = 0.0,
        max_tokens_speaker: int = 20,
        max_tokens_listener: int = 20,
    ) -> None:
        if len(agents) < 2:
            raise ConfigurationError("A Naming Game requires at least two agents.")
        if len({agent.agent_id for agent in agents}) != len(agents):
            raise ConfigurationError("Agent IDs must be unique within a trajectory.")
        if not 0.0 <= reasoning_fraction <= 1.0:
            raise ConfigurationError("reasoning_fraction must be between 0 and 1.")
        if reasoning_fraction > 0 and reasoning_task is None:
            raise ConfigurationError(
                "reasoning_fraction > 0 requires an explicit reasoning-task specification."
            )
        self.agents = agents
        self.client = client
        self.rng = random.Random(seed)
        self.reasoning_fraction = reasoning_fraction
        self.reasoning_task = reasoning_task
        self.temperature = temperature
        self.max_tokens_speaker = max_tokens_speaker
        self.max_tokens_listener = max_tokens_listener

    def _interaction_kind(self) -> str:
        if self.reasoning_fraction == 0:
            return "basic"
        if self.reasoning_fraction == 1:
            return "reasoning"
        return "reasoning" if self.rng.random() < self.reasoning_fraction else "basic"

    async def run(self, rounds: int) -> GameResult:
        if rounds < 0:
            raise ConfigurationError("rounds cannot be negative.")
        initial_counts = population_counts([agent.inventory for agent in self.agents])
        interactions = []
        round_records: list[RoundRecord] = []
        consensus_index: int | None = None
        completed_interactions = 0
        agent_by_id = {agent.agent_id: agent for agent in self.agents}
        started = time.perf_counter()

        for round_index in range(1, rounds + 1):
            round_started = time.perf_counter()
            # Every snapshot exists before any pair coroutine is created.
            snapshot: dict[int, AgentSnapshot] = {
                agent.agent_id: agent.snapshot(
                    evidence=(
                        self.reasoning_task.evidence_for(agent.agent_id)
                        if self.reasoning_task is not None
                        else None
                    )
                )
                for agent in self.agents
            }
            shuffled_ids = list(snapshot)
            self.rng.shuffle(shuffled_ids)
            idle_agent_id = shuffled_ids[-1] if len(shuffled_ids) % 2 else None
            paired_ids = shuffled_ids[: len(shuffled_ids) - (len(shuffled_ids) % 2)]
            pairs = list(zip(paired_ids[::2], paired_ids[1::2], strict=True))

            coroutines = []
            for pair_index, (speaker_id, listener_id) in enumerate(pairs, start=1):
                interaction_index = completed_interactions + pair_index
                choice_seed = self.rng.getrandbits(64)
                kind = self._interaction_kind()
                # Each task receives only its two immutable pair snapshots.
                coroutines.append(
                    execute_pair_interaction(
                        client=self.client,
                        speaker=snapshot[speaker_id],
                        listener=snapshot[listener_id],
                        interaction_index=interaction_index,
                        round_index=round_index,
                        pair_index=pair_index,
                        interaction_kind=kind,  # type: ignore[arg-type]
                        choice_seed=choice_seed,
                        temperature=self.temperature,
                        max_tokens_speaker=self.max_tokens_speaker,
                        max_tokens_listener=self.max_tokens_listener,
                        reasoning_task=self.reasoning_task,
                    )
                )

            pair_results = await asyncio.gather(*coroutines)

            # This is the update barrier: no live agent was mutated while any
            # pair task was running. Pair-result ordering therefore cannot leak.
            for result in pair_results:
                speaker = agent_by_id[result.speaker_id]
                listener = agent_by_id[result.listener_id]
                speaker.set_inventory(result.speaker_after)
                listener.set_inventory(result.listener_after)
                speaker.record(
                    {
                        "interaction_index": result.interaction_index,
                        "round_index": round_index,
                        "role": "speaker",
                        "counterpart_id": listener.agent_id,
                        "selected_name": result.selected_name,
                        "inventory_after": sorted(result.speaker_after),
                    }
                )
                listener.record(
                    {
                        "interaction_index": result.interaction_index,
                        "round_index": round_index,
                        "role": "listener",
                        "counterpart_id": speaker.agent_id,
                        "received_name": result.selected_name,
                        "inventory_after": sorted(result.listener_after),
                    }
                )
            interactions.extend(pair_results)
            completed_interactions += len(pair_results)

            counts = population_counts([agent.inventory for agent in self.agents])
            consensus = has_consensus([agent.inventory for agent in self.agents])
            if consensus and consensus_index is None:
                consensus_index = completed_interactions
            round_records.append(
                RoundRecord(
                    round_index=round_index,
                    interactions_completed=completed_interactions,
                    parallel_pairs=len(pairs),
                    idle_agent_id=idle_agent_id,
                    round_wall_seconds=time.perf_counter() - round_started,
                    slowest_pair_seconds=max(
                        (result.pair_wall_seconds for result in pair_results), default=0.0
                    ),
                    count_a=counts["A"],
                    count_b=counts["B"],
                    count_ab=counts["AB"],
                    consensus=consensus,
                )
            )

        wall_seconds = time.perf_counter() - started
        final_counts = population_counts([agent.inventory for agent in self.agents])
        return GameResult(
            interactions=interactions,
            states=[],
            rounds=round_records,
            initial_counts=initial_counts,
            final_counts=final_counts,
            wall_seconds=wall_seconds,
            consensus_reached=has_consensus(
                [agent.inventory for agent in self.agents]
            ),
            consensus_interaction_index=consensus_index,
            trajectory_concurrency=1,
        )
