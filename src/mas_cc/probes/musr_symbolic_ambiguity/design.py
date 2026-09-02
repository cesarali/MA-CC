"""Frozen private packets and Zero/Private/F9 behavioral call plan."""

from __future__ import annotations

from collections.abc import Mapping

from mas_cc.core import Seed
from mas_cc.games.relational_reasoning.data import RelationalTask
from mas_cc.probes.musr_private_redistribution.design import option_mapping
from mas_cc.probes.musr_prompt_solvability.design import CallSpec, packet


def call_plan(
    tasks: Mapping[str, RelationalTask],
    *,
    private_repetitions: int,
    endpoint_repetitions: int,
    seed: int,
) -> tuple[CallSpec, ...]:
    root = Seed(seed)
    specs: list[CallSpec] = []
    for task_id, task in tasks.items():
        for repetition in range(endpoint_repetitions):
            stream = root.derive(f"{task_id}:endpoint:{repetition}")
            mapping = option_mapping(task, stream, repetition)
            provider_seed = int(stream.derive("provider"))
            for condition, evidence in (("Zero", ()), ("F9", packet(task, 1))):
                specs.append(
                    CallSpec(
                        call_id=f"{task_id}:{condition}:{repetition:02d}",
                        phase="behavioral_validation",
                        task_id=task_id,
                        prompt_variant="P2",
                        packet_variant=condition,
                        condition=condition.lower(),
                        repetition=repetition,
                        evidence_ids=tuple(evidence),
                        option_mapping=mapping,
                        provider_seed=provider_seed,
                    )
                )
        for agent_index, agent in enumerate(task.agent_ids, 1):
            for repetition in range(private_repetitions):
                stream = root.derive(f"{task_id}:private:{agent_index}:{repetition}")
                specs.append(
                    CallSpec(
                        call_id=f"{task_id}:Private:{agent_index:03d}:{repetition:02d}",
                        phase="behavioral_validation",
                        task_id=task_id,
                        prompt_variant="P2",
                        packet_variant="Private",
                        condition="private",
                        repetition=repetition,
                        evidence_ids=task.known_facts(agent),
                        option_mapping=option_mapping(task, stream, repetition),
                        provider_seed=int(stream.derive("provider")),
                        agent_id=agent_index,
                    )
                )
    if len({spec.call_id for spec in specs}) != len(specs):
        raise ValueError("behavioral call IDs are not unique")
    return tuple(specs)


__all__ = ["call_plan"]
