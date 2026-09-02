"""Typed records used by the native Team Allocation generator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "musr_team_allocation_native_v1"
TASK_FAMILY = "musr_team_allocation"
MUSR_REPOSITORY = "https://github.com/Zayne-sprague/MuSR"
MUSR_COMMIT = "b1f4d4168a9cfc6760e8b74d728e4516023dfaa5"
PROMPT_VERSION = "musr_team_allocation_evidence_v1"


@dataclass(frozen=True, slots=True)
class TaskSpec:
    name: str
    skill: str

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "skill": self.skill}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TaskSpec":
        return cls(name=str(value["name"]), skill=str(value["skill"]))


@dataclass(frozen=True, slots=True)
class CandidateAllocation:
    singleton: str
    pair: tuple[str, str]

    def to_dict(self, tasks: Sequence[TaskSpec]) -> dict[str, Any]:
        return {
            "singleton": self.singleton,
            "pair": list(self.pair),
            "assignment": {
                tasks[0].name: [self.singleton],
                tasks[1].name: list(self.pair),
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CandidateAllocation":
        pair = value.get("pair")
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(
                "candidate allocation pair must contain exactly two people"
            )
        return cls(singleton=str(value["singleton"]), pair=(str(pair[0]), str(pair[1])))


@dataclass(frozen=True, slots=True)
class LatentProblem:
    people: tuple[str, str, str]
    tasks: tuple[TaskSpec, TaskSpec]
    skill_matrix: Mapping[str, tuple[int, int]]
    cooperation_matrix: Mapping[str, int]
    candidate_allocations: tuple[
        CandidateAllocation, CandidateAllocation, CandidateAllocation
    ]
    candidate_scores: tuple[int, int, int]
    gold_index: int
    margin_to_second_best: int

    @property
    def gold_allocation(self) -> CandidateAllocation:
        return self.candidate_allocations[self.gold_index]

    def to_dict(self) -> dict[str, Any]:
        return {
            "people": list(self.people),
            "tasks": [task.to_dict() for task in self.tasks],
            "skills": [task.skill for task in self.tasks],
            "skill_matrix": {
                person: list(values) for person, values in self.skill_matrix.items()
            },
            "cooperation_matrix": dict(self.cooperation_matrix),
            "candidate_allocations": [
                allocation.to_dict(self.tasks)
                for allocation in self.candidate_allocations
            ],
            "candidate_scores": list(self.candidate_scores),
            "gold_index": self.gold_index,
            "gold_allocation": self.gold_allocation.to_dict(self.tasks),
            "margin_to_second_best": self.margin_to_second_best,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LatentProblem":
        people = value.get("people")
        tasks = value.get("tasks")
        scores = value.get("candidate_scores")
        allocations = value.get("candidate_allocations")
        skills = value.get("skill_matrix")
        cooperation = value.get("cooperation_matrix")
        if not isinstance(people, list) or len(people) != 3:
            raise ValueError("latent people must contain exactly three entries")
        if not isinstance(tasks, list) or len(tasks) != 2:
            raise ValueError("latent tasks must contain exactly two entries")
        if not isinstance(scores, list) or len(scores) != 3:
            raise ValueError(
                "latent candidate_scores must contain exactly three entries"
            )
        if not isinstance(allocations, list) or len(allocations) != 3:
            raise ValueError(
                "latent candidate_allocations must contain exactly three entries"
            )
        if not isinstance(skills, Mapping) or not isinstance(cooperation, Mapping):
            raise ValueError("latent matrices must be objects")
        skill_matrix: dict[str, tuple[int, int]] = {}
        for person, raw_values in skills.items():
            if not isinstance(raw_values, list) or len(raw_values) != 2:
                raise ValueError("every skill row must contain exactly two entries")
            skill_matrix[str(person)] = (int(raw_values[0]), int(raw_values[1]))
        return cls(
            people=(str(people[0]), str(people[1]), str(people[2])),
            tasks=(TaskSpec.from_dict(tasks[0]), TaskSpec.from_dict(tasks[1])),
            skill_matrix=skill_matrix,
            cooperation_matrix={
                str(key): int(item) for key, item in cooperation.items()
            },
            candidate_allocations=tuple(
                CandidateAllocation.from_dict(item) for item in allocations
            ),  # type: ignore[arg-type]
            candidate_scores=(int(scores[0]), int(scores[1]), int(scores[2])),
            gold_index=int(value["gold_index"]),
            margin_to_second_best=int(value["margin_to_second_best"]),
        )


@dataclass(frozen=True, slots=True)
class LatentFact:
    fact_id: str
    kind: str
    value: int
    people: tuple[str, ...]
    task_index: int | None
    hidden_claim: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "kind": self.kind,
            "value": self.value,
            "people": list(self.people),
            "task_index": self.task_index,
            "hidden_claim": self.hidden_claim,
        }


@dataclass(frozen=True, slots=True)
class EvidenceCard:
    evidence_id: str
    latent_fact_id: str
    branch_id: str
    statements: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "latent_fact_id": self.latent_fact_id,
            "branch_id": self.branch_id,
            "text": list(self.statements),
        }


@dataclass(frozen=True, slots=True)
class FrozenTask:
    payload: Mapping[str, Any]

    @property
    def task_id(self) -> str:
        return str(self.payload["task_id"])

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)
