"""Frozen 10,818-request design for isolated truthful-selective OSS evaluation."""

from __future__ import annotations

import itertools
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from mas_cc.core import Seed
from mas_cc.games.relational_reasoning.data import RelationalTask, load_musr_team_allocation_task
from mas_cc.musr_team_allocation_generator.ambiguity import TeamAllocationCompletionIndex
from mas_cc.musr_team_allocation_generator.io_utils import sha256_object
from mas_cc.musr_team_allocation_generator.symbolic_facts import CanonicalFact

from .isolated_config import IsolatedOSSConfig


@dataclass(frozen=True, slots=True)
class IsolatedRequest:
    request_id: str
    paired_unit_id: str
    task_id: str
    candidate_id: int
    task_artifact_sha256: str
    assignment_sha256: str
    condition: str
    budget: int | None
    agent_id: str | None
    answer_order_id: str
    option_mapping: Mapping[str, str]
    private_fact_ids: tuple[str, ...]
    report_fact_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    random_replicate: int | None
    random_seed: int | None
    unique_visible_fact_count: int
    private_report_overlap_count: int
    latent_coverage_count: int
    predicate_family_count: int
    compatible_worlds_prefix_only: int | None
    compatible_worlds_private_plus_reports: int | None
    compatible_world_reduction_prefix_only: int | None
    compatible_world_reduction_private_plus_reports: int | None
    report_character_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "paired_unit_id": self.paired_unit_id,
            "task_id": self.task_id,
            "candidate_id": self.candidate_id,
            "task_artifact_sha256": self.task_artifact_sha256,
            "assignment_sha256": self.assignment_sha256,
            "condition": self.condition,
            "budget": self.budget,
            "agent_id": self.agent_id,
            "answer_order_id": self.answer_order_id,
            "semantic_option_mapping": dict(self.option_mapping),
            "private_fact_ids": list(self.private_fact_ids),
            "report_fact_ids": list(self.report_fact_ids),
            "evidence_ids": list(self.evidence_ids),
            "random_replicate": self.random_replicate,
            "random_seed": self.random_seed,
            "delivered_report_count": len(self.report_fact_ids),
            "private_report_overlap_count": self.private_report_overlap_count,
            "unique_visible_fact_count": self.unique_visible_fact_count,
            "latent_coverage_count": self.latent_coverage_count,
            "predicate_family_count": self.predicate_family_count,
            "compatible_worlds_prefix_only": self.compatible_worlds_prefix_only,
            "compatible_worlds_private_plus_reports": self.compatible_worlds_private_plus_reports,
            "compatible_world_reduction_prefix_only": self.compatible_world_reduction_prefix_only,
            "compatible_world_reduction_private_plus_reports": self.compatible_world_reduction_private_plus_reports,
            "report_character_count": self.report_character_count,
        }


def load_selected_tasks(config: IsolatedOSSConfig) -> dict[str, RelationalTask]:
    tasks = {
        task_id: load_musr_team_allocation_task(
            config.calibration_root / "tasks",
            task_id,
            population_size=config.assignment_population,
        )
        for task_id in sorted(config.expected_tasks)
    }
    for task_id, task in tasks.items():
        source = json.loads(
            (config.calibration_root / f"tasks/{task_id}/task.json").read_text(
                encoding="utf-8"
            )
        )
        if int(source["candidate_id"]) != config.expected_tasks[task_id]:
            raise RuntimeError(f"{task_id} candidate does not match frozen selection")
        if task.population_size != 24 or len(task.agent_ids) != 24:
            raise RuntimeError(f"{task_id} does not have the approved N=24 assignment")
    return tasks


def answer_orders(task: RelationalTask) -> tuple[tuple[str, Mapping[str, str]], ...]:
    return tuple(
        (
            "order_" + "".join(str(task.semantic_answers.index(value)) for value in permutation),
            dict(zip("ABC", permutation, strict=True)),
        )
        for permutation in itertools.permutations(task.semantic_answers)
    )


def random_permutations(
    config: IsolatedOSSConfig, task: RelationalTask
) -> tuple[tuple[int, tuple[str, ...]], ...]:
    pool = tuple(task.controller_reportable_fact_ids)
    rows = []
    for replicate in range(config.random_replicates):
        seed = int(Seed(config.seed).derive(f"random-pool:{task.task_id}:{replicate}"))
        values = list(pool)
        random.Random(seed).shuffle(values)
        rows.append((seed, tuple(values)))
    return tuple(rows)


def _fact_map(root: Path) -> dict[str, CanonicalFact]:
    return {
        str(row["fact_id"]): CanonicalFact.from_dict(row)
        for row in json.loads((root / "facts/all_true_facts.json").read_text(encoding="utf-8"))
    }


def _request(
    *,
    task: RelationalTask,
    candidate: int,
    condition: str,
    budget: int | None,
    agent_id: str | None,
    order_id: str,
    mapping: Mapping[str, str],
    private: Sequence[str],
    reports: Sequence[str],
    replicate: int | None,
    random_seed: int | None,
    facts: Mapping[str, CanonicalFact],
    completion_index: TeamAllocationCompletionIndex,
    task_artifact_sha256: str,
    assignment_sha256: str,
) -> IsolatedRequest:
    private_ids = tuple(private)
    report_ids = tuple(reports)
    evidence = tuple(dict.fromkeys((*private_ids, *report_ids)))
    selected_facts = tuple(facts[fact_id] for fact_id in evidence)
    report_facts = tuple(facts[fact_id] for fact_id in report_ids)
    latent = {
        int(value)
        for fact in selected_facts
        for value in fact.provenance.get("latent_indices", ())
    }
    families = {(fact.kind, fact.operator) for fact in selected_facts}
    pair = f"{task.task_id}:{agent_id or 'task'}:{order_id}:{budget or 0}"
    identity = {
        "task": task.task_id,
        "candidate": candidate,
        "condition": condition,
        "budget": budget,
        "agent": agent_id,
        "order": order_id,
        "replicate": replicate,
        "private": private_ids,
        "reports": report_ids,
        "mapping": dict(mapping),
    }
    baseline_worlds = completion_index.metrics_for_facts(()).valid_completion_count
    prefix_worlds = (
        None
        if not report_facts
        else completion_index.metrics_for_facts(report_facts).valid_completion_count
    )
    combined_worlds = (
        None
        if not selected_facts
        else completion_index.metrics_for_facts(selected_facts).valid_completion_count
    )
    canonical = task.controller_report_texts or {}
    return IsolatedRequest(
        request_id="iso_" + sha256_object(identity)[:24],
        paired_unit_id=pair,
        task_id=task.task_id,
        candidate_id=candidate,
        task_artifact_sha256=task_artifact_sha256,
        assignment_sha256=assignment_sha256,
        condition=condition,
        budget=budget,
        agent_id=agent_id,
        answer_order_id=order_id,
        option_mapping=dict(mapping),
        private_fact_ids=private_ids,
        report_fact_ids=report_ids,
        evidence_ids=evidence,
        random_replicate=replicate,
        random_seed=random_seed,
        unique_visible_fact_count=len(evidence),
        private_report_overlap_count=len(set(private_ids) & set(report_ids)),
        latent_coverage_count=len(latent),
        predicate_family_count=len(families),
        compatible_worlds_prefix_only=prefix_worlds,
        compatible_worlds_private_plus_reports=combined_worlds,
        compatible_world_reduction_prefix_only=(None if prefix_worlds is None else baseline_worlds - prefix_worlds),
        compatible_world_reduction_private_plus_reports=(None if combined_worlds is None else baseline_worlds - combined_worlds),
        report_character_count=sum(len(str(canonical[fact_id])) for fact_id in report_ids),
    )


def build_manifest(
    config: IsolatedOSSConfig,
    tasks: Mapping[str, RelationalTask],
) -> tuple[IsolatedRequest, ...]:
    requests: list[IsolatedRequest] = []
    completion_index = TeamAllocationCompletionIndex()
    for task_id, task in sorted(tasks.items()):
        root = config.calibration_root / "tasks" / task_id
        facts = _fact_map(root)
        from mas_cc.musr_team_allocation_generator.io_utils import sha256_file

        task_artifact_sha256 = sha256_file(root / "task.json")
        assignment_sha256 = sha256_file(root / "private/N24_assignment.json")
        ranked = tuple(
            str(row["fact_id"])
            for row in json.loads(
                (root / "controller/ranked_fact_pool.json").read_text(encoding="utf-8")
            )
        )
        random_rows = random_permutations(config, task)
        orders = answer_orders(task)
        for order_id, mapping in orders:
            requests.append(
                _request(
                    task=task,
                    candidate=config.expected_tasks[task_id],
                    condition="full",
                    budget=None,
                    agent_id=None,
                    order_id=order_id,
                    mapping=mapping,
                    private=task.fact_order,
                    reports=(),
                    replicate=None,
                    random_seed=None,
                    facts=facts,
                    completion_index=completion_index,
                    task_artifact_sha256=task_artifact_sha256,
                    assignment_sha256=assignment_sha256,
                )
            )
            for agent_id in task.agent_ids:
                private = task.known_facts(agent_id)
                requests.append(
                    _request(
                        task=task,
                        candidate=config.expected_tasks[task_id],
                        condition="private",
                        budget=None,
                        agent_id=agent_id,
                        order_id=order_id,
                        mapping=mapping,
                        private=private,
                        reports=(),
                        replicate=None,
                        random_seed=None,
                        facts=facts,
                        completion_index=completion_index,
                        task_artifact_sha256=task_artifact_sha256,
                        assignment_sha256=assignment_sha256,
                    )
                )
                for budget in config.budgets:
                    requests.append(
                        _request(
                            task=task,
                            candidate=config.expected_tasks[task_id],
                            condition="strategic",
                            budget=budget,
                            agent_id=agent_id,
                            order_id=order_id,
                            mapping=mapping,
                            private=private,
                            reports=ranked[:budget],
                            replicate=None,
                            random_seed=None,
                            facts=facts,
                            completion_index=completion_index,
                            task_artifact_sha256=task_artifact_sha256,
                            assignment_sha256=assignment_sha256,
                        )
                    )
                    for replicate, (random_seed, permutation) in enumerate(random_rows):
                        requests.append(
                            _request(
                                task=task,
                                candidate=config.expected_tasks[task_id],
                                condition="random",
                                budget=budget,
                                agent_id=agent_id,
                                order_id=order_id,
                                mapping=mapping,
                                private=private,
                                reports=permutation[:budget],
                                replicate=replicate,
                                random_seed=random_seed,
                                facts=facts,
                                completion_index=completion_index,
                                task_artifact_sha256=task_artifact_sha256,
                                assignment_sha256=assignment_sha256,
                            )
                        )
    if len(requests) != config.logical_calls or len({row.request_id for row in requests}) != len(requests):
        raise RuntimeError("isolated evaluation manifest count or identity mismatch")
    return tuple(requests)


def smoke_ids(requests: Sequence[IsolatedRequest]) -> tuple[str, ...]:
    selected = []
    for condition in ("full", "private", "strategic", "random"):
        selected.extend(
            row.request_id
            for row in requests
            if row.task_id == "task_001"
            and row.answer_order_id == "order_012"
            and row.condition == condition
            and (row.agent_id in {None, "agent_001"})
            and (row.budget in {None, 3})
            and (row.random_replicate in {None, 0})
        )
    return tuple(dict.fromkeys(selected))


__all__ = [
    "IsolatedRequest",
    "answer_orders",
    "build_manifest",
    "load_selected_tasks",
    "random_permutations",
    "smoke_ids",
]
