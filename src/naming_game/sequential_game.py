"""Canonical random-sequential asynchronous Naming Game."""

from __future__ import annotations

import random
import time

from .agent import Agent
from .api_client import LLMClient
from .interaction import execute_pair_interaction
from .models import (
    ConfigurationError,
    GameResult,
    StateRecord,
    has_consensus,
    population_counts,
)
from .reasoning_game import ReasoningTask


class SequentialNamingGame:
    """Run dependent interactions one at a time within one trajectory."""

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

    async def run(self, num_interactions: int) -> GameResult:
        if num_interactions < 0:
            raise ConfigurationError("num_interactions cannot be negative.")
        initial_counts = population_counts([agent.inventory for agent in self.agents])
        interactions = []
        states: list[StateRecord] = []
        consensus_index: int | None = None
        started = time.perf_counter()

        for interaction_index in range(1, num_interactions + 1):
            speaker, listener = self.rng.sample(self.agents, 2)
            choice_seed = self.rng.getrandbits(64)
            kind = self._interaction_kind()
            result = await execute_pair_interaction(
                client=self.client,
                speaker=speaker.snapshot(
                    evidence=(
                        self.reasoning_task.evidence_for(speaker.agent_id)
                        if self.reasoning_task is not None
                        else None
                    )
                ),
                listener=listener.snapshot(
                    evidence=(
                        self.reasoning_task.evidence_for(listener.agent_id)
                        if self.reasoning_task is not None
                        else None
                    )
                ),
                interaction_index=interaction_index,
                round_index=None,
                pair_index=None,
                interaction_kind=kind,  # type: ignore[arg-type]
                choice_seed=choice_seed,
                temperature=self.temperature,
                max_tokens_speaker=self.max_tokens_speaker,
                max_tokens_listener=self.max_tokens_listener,
                reasoning_task=self.reasoning_task,
            )

            # The mutation happens before the next pair is sampled.
            speaker.set_inventory(result.speaker_after)
            listener.set_inventory(result.listener_after)
            event = {
                "interaction_index": interaction_index,
                "role": "speaker",
                "counterpart_id": listener.agent_id,
                "selected_name": result.selected_name,
                "inventory_after": sorted(result.speaker_after),
            }
            speaker.record(event)
            listener.record(
                {
                    "interaction_index": interaction_index,
                    "role": "listener",
                    "counterpart_id": speaker.agent_id,
                    "received_name": result.selected_name,
                    "inventory_after": sorted(result.listener_after),
                }
            )
            interactions.append(result)

            counts = population_counts([agent.inventory for agent in self.agents])
            consensus = has_consensus([agent.inventory for agent in self.agents])
            if consensus and consensus_index is None:
                consensus_index = interaction_index
            states.append(
                StateRecord(
                    interaction_index=interaction_index,
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
            states=states,
            rounds=[],
            initial_counts=initial_counts,
            final_counts=final_counts,
            wall_seconds=wall_seconds,
            consensus_reached=has_consensus(
                [agent.inventory for agent in self.agents]
            ),
            consensus_interaction_index=consensus_index,
            trajectory_concurrency=1,
        )

