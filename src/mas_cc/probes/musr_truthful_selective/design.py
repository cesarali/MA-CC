"""Deterministic local behavioral conditions for truthful selective disclosure."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from mas_cc.core import Seed
from mas_cc.games.relational_reasoning.data import RelationalTask
from mas_cc.probes.musr_prompt_solvability.design import mapping

from .config import TruthfulSelectiveConfig


@dataclass(frozen=True, slots=True)
class CallSpec:
    call_id: str
    task_id: str
    condition: str
    repetition: int
    evidence_ids: tuple[str, ...]
    option_mapping: Mapping[str, str]
    provider_seed: int
    agent_id: str | None = None
    budget: int | None = None
    subset_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        from mas_cc.musr_team_allocation_generator.io_utils import sha256_object

        return {
            "call_id": self.call_id,
            "task_id": self.task_id,
            "condition": self.condition,
            "repetition": self.repetition,
            "evidence_ids": list(self.evidence_ids),
            "evidence_sha256": sha256_object(list(self.evidence_ids)),
            "semantic_option_mapping": dict(self.option_mapping),
            "provider_seed": self.provider_seed,
            "agent_id": self.agent_id,
            "budget": self.budget,
            "subset_id": self.subset_id,
        }


def _ids(path: Path) -> tuple[str, ...]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list) and value and isinstance(value[0], Mapping):
        return tuple(str(row["fact_id"]) for row in value)
    return tuple(str(item) for item in value)


def _append(
    specs: list[CallSpec],
    config: TruthfulSelectiveConfig,
    task: RelationalTask,
    condition: str,
    evidence: Sequence[str],
    repetitions: int,
    *,
    agent_id: str | None = None,
    budget: int | None = None,
    subset_id: str | None = None,
) -> None:
    root = Seed(config.seed)
    for repetition in range(repetitions):
        identity = ":".join(
            value
            for value in (
                task.task_id,
                condition,
                agent_id,
                subset_id,
                f"{repetition:03d}",
            )
            if value is not None
        )
        stream = root.derive(identity)
        specs.append(
            CallSpec(
                identity,
                task.task_id,
                condition,
                repetition,
                tuple(evidence),
                mapping(task, stream.derive("options")),
                int(stream.derive("provider")),
                agent_id,
                budget,
                subset_id,
            )
        )


def call_plan(
    config: TruthfulSelectiveConfig,
    tasks: Mapping[str, RelationalTask],
    task_root: Path,
) -> tuple[CallSpec, ...]:
    specs: list[CallSpec] = []
    for task_id, task in sorted(tasks.items()):
        root = task_root / task_id
        _append(specs, config, task, "ZERO", (), config.zero_repetitions)
        for agent_id in task.agent_ids:
            _append(
                specs,
                config,
                task,
                "PRIVATE",
                task.known_facts(agent_id),
                config.private_repetitions,
                agent_id=agent_id,
            )
        decisive = _ids(root / "facts/decisive_facts.json")
        _append(specs, config, task, "DECISIVE", decisive, config.decisive_repetitions)
        _append(specs, config, task, "FULL", task.fact_order, config.full_repetitions)
        pool = task.controller_reportable_fact_ids
        for budget in config.symbolic.controller_budgets:
            controller = _ids(root / f"controller/selected_C{budget}.json")
            _append(
                specs,
                config,
                task,
                f"C{budget}",
                controller,
                config.controller_repetitions,
                budget=budget,
            )
            _append(
                specs,
                config,
                task,
                f"C{budget}+D",
                (*controller, *decisive),
                config.mixed_repetitions,
                budget=budget,
            )
        for budget in (6, 12, 24):
            rng = random.Random(f"{config.seed}:{task_id}:alternatives:{budget}")
            seen: set[tuple[str, ...]] = set()
            attempts = 0
            while len(seen) < config.alternative_subsets_per_budget and attempts < 1000:
                attempts += 1
                seen.add(tuple(sorted(rng.sample(pool, budget))))
            for subset_index, subset in enumerate(sorted(seen), 1):
                _append(
                    specs,
                    config,
                    task,
                    f"C{budget}_ALT",
                    subset,
                    config.alternative_subset_repetitions,
                    budget=budget,
                    subset_id=f"alt_{subset_index:02d}",
                )
    if len(specs) > config.behavioral_calls:
        raise RuntimeError("behavioral call plan exceeds the configured ceiling")
    if len({spec.call_id for spec in specs}) != len(specs):
        raise RuntimeError("behavioral call IDs are not unique")
    return tuple(specs)


__all__ = ["CallSpec", "call_plan"]
