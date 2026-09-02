"""Deterministic staged call design for prompt and packet calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from mas_cc.core import Seed
from mas_cc.games.relational_reasoning.data import RelationalTask
from mas_cc.musr_team_allocation_generator.io_utils import sha256_object

PROMPTS = ("P0", "P1", "P2", "P3")
PACKETS = ("F9", "F18", "F27")


@dataclass(frozen=True, slots=True)
class CallSpec:
    call_id: str
    phase: str
    task_id: str
    prompt_variant: str
    packet_variant: str
    condition: str
    repetition: int
    evidence_ids: tuple[str, ...]
    option_mapping: Mapping[str, str]
    provider_seed: int
    agent_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "phase": self.phase,
            "task_id": self.task_id,
            "prompt_variant": self.prompt_variant,
            "packet_variant": self.packet_variant,
            "condition": self.condition,
            "repetition": self.repetition,
            "agent_id": self.agent_id,
            "evidence_ids": list(self.evidence_ids),
            "evidence_sha256": sha256_object(list(self.evidence_ids)),
            "semantic_option_mapping": dict(self.option_mapping),
            "provider_seed": self.provider_seed,
        }


def packet(task: RelationalTask, branches: int) -> tuple[str, ...]:
    groups = task.supporting_fact_groups or {}
    selected = []
    for latent in sorted(groups):
        selected.extend(sorted(groups[latent])[:branches])
    return tuple(card for card in task.fact_order if card in set(selected))


def packet_definitions(
    tasks: Mapping[str, RelationalTask],
) -> dict[str, dict[str, list[str]]]:
    return {
        task_id: {
            "F9": list(packet(task, 1)),
            "F18": list(packet(task, 2)),
            "F27": list(packet(task, 3)),
        }
        for task_id, task in tasks.items()
    }


def mapping(task: RelationalTask, seed: Seed) -> dict[str, str]:
    options = list(task.semantic_answers)
    seed.create_random().shuffle(options)
    return dict(zip("ABC", options, strict=True))


def phase_a(
    tasks: Mapping[str, RelationalTask],
    task_ids: Sequence[str],
    repetitions: int,
    seed: int,
) -> tuple[CallSpec, ...]:
    root = Seed(seed)
    specs = []
    for task_id in task_ids:
        task = tasks[task_id]
        evidence = packet(task, 1)
        for repetition in range(repetitions):
            matched = root.derive(f"phase-a:{task_id}:{repetition}")
            option_mapping = mapping(task, matched.derive("options"))
            provider_seed = int(matched.derive("provider"))
            order = list(PROMPTS)
            matched.derive("dispatch").create_random().shuffle(order)
            for variant in order:
                specs.append(
                    CallSpec(
                        f"A:{task_id}:{repetition:02d}:{variant}",
                        "prompt_ablation",
                        task_id,
                        variant,
                        "F9",
                        "full",
                        repetition,
                        evidence,
                        option_mapping,
                        provider_seed,
                    )
                )
    return tuple(specs)


def phase_b(
    tasks: Mapping[str, RelationalTask],
    task_ids: Sequence[str],
    selected_prompt: str,
    repetitions: int,
    seed: int,
) -> tuple[CallSpec, ...]:
    root = Seed(seed)
    specs = []
    for task_id in task_ids:
        task = tasks[task_id]
        for repetition in range(repetitions):
            matched = root.derive(f"phase-b:{task_id}:{repetition}")
            option_mapping = mapping(task, matched.derive("options"))
            provider_seed = int(matched.derive("provider"))
            order = list(PACKETS)
            matched.derive("dispatch").create_random().shuffle(order)
            for name in order:
                branches = {"F9": 1, "F18": 2, "F27": 3}[name]
                specs.append(
                    CallSpec(
                        f"B:{task_id}:{repetition:02d}:{name}",
                        "full_profile_ablation",
                        task_id,
                        selected_prompt,
                        name,
                        "full",
                        repetition,
                        packet(task, branches),
                        option_mapping,
                        provider_seed,
                    )
                )
    return tuple(specs)


def phase_c(
    tasks: Mapping[str, RelationalTask],
    task_ids: Sequence[str],
    selected_prompt: str,
    selected_packet: str,
    repetitions: int,
    population_size: int,
    seed: int,
) -> tuple[CallSpec, ...]:
    root = Seed(seed)
    specs = []
    branches = {"F9": 1, "F18": 2, "F27": 3}[selected_packet]
    for task_id in task_ids:
        task = tasks[task_id]
        for repetition in range(repetitions):
            for condition, evidence in (("zero", ()), ("full", packet(task, branches))):
                key = f"phase-c:{task_id}:{condition}:{repetition}"
                stream = root.derive(key)
                specs.append(
                    CallSpec(
                        f"C:{task_id}:{condition}:{repetition:02d}",
                        "heldout_validation",
                        task_id,
                        selected_prompt,
                        selected_packet if condition == "full" else "none",
                        condition,
                        repetition,
                        tuple(evidence),
                        mapping(task, stream.derive("options")),
                        int(stream.derive("provider")),
                    )
                )
            for agent in range(1, population_size + 1):
                key = f"phase-c:{task_id}:private:{agent}:{repetition}"
                stream = root.derive(key)
                specs.append(
                    CallSpec(
                        f"C:{task_id}:private:{agent:03d}:{repetition:02d}",
                        "heldout_validation",
                        task_id,
                        selected_prompt,
                        "natural",
                        "private",
                        repetition,
                        task.known_facts(f"agent_{agent:03d}"),
                        mapping(task, stream.derive("options")),
                        int(stream.derive("provider")),
                        agent,
                    )
                )
    return tuple(specs)


__all__ = [
    "CallSpec",
    "PACKETS",
    "PROMPTS",
    "packet",
    "packet_definitions",
    "phase_a",
    "phase_b",
    "phase_c",
]
