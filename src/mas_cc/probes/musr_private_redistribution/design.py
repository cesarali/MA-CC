"""Behavioral call plan for Zero/NAT/R2/R3/R4/F9."""

from __future__ import annotations
from collections.abc import Mapping, Sequence
from mas_cc.core import Seed
from mas_cc.games.relational_reasoning.data import RelationalTask
from mas_cc.probes.musr_prompt_solvability.design import CallSpec, packet
from .assignment import build_assignment

REGIMES = ("Zero", "NAT", "R2", "R3", "R4", "F9")


def option_mapping(
    task: RelationalTask, stream: Seed, rotation: int = 0
) -> dict[str, str]:
    options = list(task.semantic_answers)
    stream.create_random().shuffle(options)
    options = options[rotation % 3 :] + options[: rotation % 3]
    return dict(zip("ABC", options, strict=True))


def assignments(
    tasks: Mapping[str, RelationalTask], seed: int
) -> dict[tuple[str, str], dict]:
    return {
        (task_id, regime): build_assignment(task, regime, seed)
        for task_id, task in tasks.items()
        for regime in ("NAT", "R2", "R3", "R4")
    }


def call_plan(
    tasks: Mapping[str, RelationalTask],
    assignment_map: Mapping[tuple[str, str], Mapping],
    private_repetitions: int,
    endpoint_repetitions: int,
    seed: int,
) -> tuple[CallSpec, ...]:
    root = Seed(seed)
    specs = []
    for task_id, task in tasks.items():
        for repetition in range(endpoint_repetitions):
            stream = root.derive(f"{task_id}:endpoint:{repetition}")
            mapping = option_mapping(task, stream, repetition)
            provider_seed = int(stream.derive("provider"))
            for regime, evidence in (("Zero", ()), ("F9", packet(task, 1))):
                specs.append(
                    CallSpec(
                        f"{task_id}:{regime}:{repetition:02d}",
                        "behavioral",
                        task_id,
                        "P2",
                        regime,
                        regime.lower(),
                        repetition,
                        tuple(evidence),
                        mapping,
                        provider_seed,
                    )
                )
        for agent_index, agent in enumerate(task.agent_ids, 1):
            for repetition in range(private_repetitions):
                stream = root.derive(f"{task_id}:private:{agent_index}:{repetition}")
                mapping = option_mapping(task, stream, repetition)
                provider_seed = int(stream.derive("provider"))
                for regime in ("NAT", "R2", "R3", "R4"):
                    evidence = tuple(
                        assignment_map[(task_id, regime)]["agent_assignments"][agent]
                    )
                    specs.append(
                        CallSpec(
                            f"{task_id}:{regime}:{agent_index:03d}:{repetition:02d}",
                            "behavioral",
                            task_id,
                            "P2",
                            regime,
                            regime.lower(),
                            repetition,
                            evidence,
                            mapping,
                            provider_seed,
                            agent_index,
                        )
                    )
    if len(specs) != 492 or len({s.call_id for s in specs}) != 492:
        raise ValueError("redistribution plan must have 492 unique calls")
    return tuple(specs)


__all__ = ["REGIMES", "assignments", "call_plan", "option_mapping"]
