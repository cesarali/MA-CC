"""Canonical, machine-checkable facts derived from Team Allocation worlds.

Unlike an evidence card, a :class:`CanonicalFact` is one logical proposition.
Its truth set over the finite hidden-world support is therefore exact and can be
used for posterior conditioning without asking a language model.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Mapping, Sequence

from .latent_problem import latent_facts
from .schemas import LatentProblem


@dataclass(frozen=True, slots=True)
class CanonicalFact:
    fact_id: str
    kind: str
    left_index: int
    operator: str
    right_index: int | None
    threshold: int | None
    canonical_text: str
    provenance: Mapping[str, Any]

    def holds(self, vector: Sequence[int]) -> bool:
        left = int(vector[self.left_index])
        if self.operator == "eq_value":
            return left == int(self.threshold)
        if self.operator == "ge_threshold":
            return left >= int(self.threshold)
        if self.operator == "le_threshold":
            return left <= int(self.threshold)
        right = int(vector[int(self.right_index)])
        if self.operator == "le":
            return left <= right
        if self.operator == "eq":
            return left == right
        if self.operator == "ge":
            return left >= right
        raise ValueError(f"unknown canonical fact operator {self.operator!r}")

    @property
    def logical_signature(self) -> tuple[Any, ...]:
        return (
            self.operator,
            self.left_index,
            self.right_index,
            self.threshold,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "kind": self.kind,
            "predicate": {
                "operator": self.operator,
                "left_index": self.left_index,
                "right_index": self.right_index,
                "threshold": self.threshold,
            },
            "canonical_fact_text": self.canonical_text,
            "provenance": dict(self.provenance),
            "logical_signature": list(self.logical_signature),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CanonicalFact":
        predicate = value.get("predicate")
        if not isinstance(predicate, Mapping):
            raise ValueError("canonical fact predicate must be an object")
        provenance = value.get("provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError("canonical fact provenance must be an object")
        right = predicate.get("right_index")
        threshold = predicate.get("threshold")
        return cls(
            fact_id=str(value["fact_id"]),
            kind=str(value["kind"]),
            left_index=int(predicate["left_index"]),
            operator=str(predicate["operator"]),
            right_index=None if right is None else int(right),
            threshold=None if threshold is None else int(threshold),
            canonical_text=str(value["canonical_fact_text"]),
            provenance=dict(provenance),
        )


def _label(problem: LatentProblem, index: int) -> str:
    fact = latent_facts(problem)[index]
    if fact.kind == "skill":
        assert fact.task_index is not None
        return f"{fact.people[0]}'s skill for {problem.tasks[fact.task_index].name}"
    return f"the cooperation of {fact.people[0]} and {fact.people[1]}"


def _provenance(problem: LatentProblem, indices: Sequence[int]) -> dict[str, Any]:
    source = latent_facts(problem)
    return {
        "latent_indices": list(indices),
        "latent_fact_ids": [source[index].fact_id for index in indices],
        "derivation": "deterministic predicate over the exact hidden latent values",
    }


def canonical_fact_catalog(problem: LatentProblem) -> tuple[CanonicalFact, ...]:
    """Return the fixed non-tautological proposition catalog for one world.

    Skill values are compared only with other skill values and cooperation
    values only with other cooperation values. Inclusive comparisons are kept
    because ``x <= y`` and ``x == y`` have different truth sets and meanings;
    semantic duplicates are rejected below by exhaustive truth-table identity.
    """

    facts: list[CanonicalFact] = []
    for index in range(9):
        label = _label(problem, index)
        kind = "skill_threshold" if index < 6 else "cooperation_threshold"
        facts.extend(
            CanonicalFact(
                f"cf_x{index:02d}_eq_{value}",
                "skill_exact" if index < 6 else "cooperation_exact",
                index,
                "eq_value",
                None,
                value,
                f"{label.capitalize()} is {_level_word(index, value)}.",
                _provenance(problem, (index,)),
            )
            for value in (1, 2, 3)
        )
        facts.extend(
            (
                CanonicalFact(
                    f"cf_x{index:02d}_ge_2",
                    kind,
                    index,
                    "ge_threshold",
                    None,
                    2,
                    f"{label.capitalize()} is at least moderate.",
                    _provenance(problem, (index,)),
                ),
                CanonicalFact(
                    f"cf_x{index:02d}_le_2",
                    kind,
                    index,
                    "le_threshold",
                    None,
                    2,
                    f"{label.capitalize()} is at most moderate.",
                    _provenance(problem, (index,)),
                ),
            )
        )
    for domain, kind in (
        ((0, 1, 2, 3, 4, 5), "skill_comparison"),
        ((6, 7, 8), "cooperation_comparison"),
    ):
        for left, right in combinations(domain, 2):
            left_label, right_label = _label(problem, left), _label(problem, right)
            for operator, phrase, suffix in (
                ("le", "is no stronger than", "le"),
                ("eq", "is at the same level as", "eq"),
                ("ge", "is at least as strong as", "ge"),
            ):
                facts.append(
                    CanonicalFact(
                        f"cf_x{left:02d}_{suffix}_x{right:02d}",
                        kind,
                        left,
                        operator,
                        right,
                        None,
                        f"{left_label.capitalize()} {phrase} {right_label}.",
                        _provenance(problem, (left, right)),
                    )
                )
    signatures = [fact.logical_signature for fact in facts]
    if len(signatures) != len(set(signatures)):
        raise RuntimeError("canonical fact catalog contains duplicate propositions")
    return tuple(facts)


def true_canonical_facts(problem: LatentProblem) -> tuple[CanonicalFact, ...]:
    from .latent_problem import latent_values

    vector = latent_values(problem)
    return tuple(fact for fact in canonical_fact_catalog(problem) if fact.holds(vector))


def render_canonical_equality_evidence(
    problem: LatentProblem, fact: CanonicalFact
) -> tuple[str, str]:
    """Render direct categorical equality without disclosing the category itself."""

    if fact.operator != "eq" or fact.right_index is None:
        raise ValueError("canonical equality rendering requires operator 'eq'")
    left = _label(problem, fact.left_index)
    right = _label(problem, fact.right_index)
    if fact.kind == "skill_comparison":
        return (
            f"An independent assessment compared {left} with {right} using one shared categorical proficiency rubric.",
            f"That assessment placed {left} and {right} in the same proficiency category.",
        )
    if fact.kind == "cooperation_comparison":
        return (
            f"An independent teamwork assessment compared {left} with {right} using one shared categorical rubric.",
            f"That assessment placed {left} and {right} in the same category.",
        )
    raise ValueError(f"unsupported equality fact kind {fact.kind!r}")


def _level_word(index: int, value: int) -> str:
    if index < 6:
        return {1: "limited", 2: "moderate", 3: "strong"}[value]
    return {1: "poor", 2: "adequate", 3: "very strong"}[value]


__all__ = [
    "CanonicalFact",
    "canonical_fact_catalog",
    "render_canonical_equality_evidence",
    "true_canonical_facts",
]
