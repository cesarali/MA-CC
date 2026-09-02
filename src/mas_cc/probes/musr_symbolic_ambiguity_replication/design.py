"""Additional-only call plan for the frozen symbolic-ambiguity replication."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from mas_cc.core import Seed
from mas_cc.games.relational_reasoning.data import RelationalTask
from mas_cc.probes.musr_private_redistribution.design import option_mapping
from mas_cc.probes.musr_prompt_solvability.design import CallSpec, packet


@dataclass(frozen=True, slots=True)
class ReplicationCallSpec(CallSpec):
    latent_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            **CallSpec.to_dict(self),
            "replicate_id": self.repetition,
            "latent_ids": list(self.latent_ids),
        }


def _latent_ids(task: RelationalTask, evidence_ids: tuple[str, ...]) -> tuple[str, ...]:
    selected = set(evidence_ids)
    return tuple(
        latent_id
        for latent_id, cards in (task.supporting_fact_groups or {}).items()
        if selected.intersection(cards)
    )


def additional_call_plan(
    tasks: Mapping[str, RelationalTask],
    *,
    private_start: int,
    private_count: int,
    endpoint_start: int,
    endpoint_count: int,
    seed: int,
) -> tuple[ReplicationCallSpec, ...]:
    """Create only unused repetition IDs while preserving the original seed scheme."""

    root = Seed(seed)
    specs: list[ReplicationCallSpec] = []
    for task_id, task in tasks.items():
        for repetition in range(endpoint_start, endpoint_start + endpoint_count):
            stream = root.derive(f"{task_id}:endpoint:{repetition}")
            mapping = option_mapping(task, stream, repetition)
            provider_seed = int(stream.derive("provider"))
            for condition, evidence in (("Zero", ()), ("F9", packet(task, 1))):
                specs.append(
                    ReplicationCallSpec(
                        call_id=f"{task_id}:{condition}:{repetition:02d}",
                        phase="behavioral_replication",
                        task_id=task_id,
                        prompt_variant="P2",
                        packet_variant=condition,
                        condition=condition.lower(),
                        repetition=repetition,
                        evidence_ids=tuple(evidence),
                        option_mapping=mapping,
                        provider_seed=provider_seed,
                        latent_ids=_latent_ids(task, tuple(evidence)),
                    )
                )
        for agent_index, agent in enumerate(task.agent_ids, 1):
            for repetition in range(private_start, private_start + private_count):
                stream = root.derive(f"{task_id}:private:{agent_index}:{repetition}")
                specs.append(
                    ReplicationCallSpec(
                        call_id=f"{task_id}:Private:{agent_index:03d}:{repetition:02d}",
                        phase="behavioral_replication",
                        task_id=task_id,
                        prompt_variant="P2",
                        packet_variant="Private",
                        condition="private",
                        repetition=repetition,
                        evidence_ids=task.known_facts(agent),
                        option_mapping=option_mapping(task, stream, repetition),
                        provider_seed=int(stream.derive("provider")),
                        agent_id=agent_index,
                        latent_ids=_latent_ids(task, task.known_facts(agent)),
                    )
                )
    if len({spec.call_id for spec in specs}) != len(specs):
        raise ValueError("replication call IDs are not unique")
    return tuple(specs)


__all__ = ["ReplicationCallSpec", "additional_call_plan"]
