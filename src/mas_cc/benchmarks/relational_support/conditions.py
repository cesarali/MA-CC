"""Evidence conditions: which of the L supporting facts a prompt is allowed to show.

One task yields several prompts that differ in **exactly one** respect - the
subset of supporting facts included.  Distractors, the question, the option
labels and the option order are held identical across every condition of a task,
so a difference in accuracy between conditions is attributable to the evidence
and to nothing else.

For ``0 < k < L`` there is more than one subset of size ``k``, and they are not
interchangeable: in a chain ``A->B->C`` the first link and the second link carry
different amounts of information about the endpoints.  So subsets are enumerated
rather than prefixed, and the analysis reports them separately before pooling.
"""

from __future__ import annotations

import itertools
import random
from collections.abc import Sequence
from dataclasses import dataclass

FULL = "full"
PARTIAL = "partial"
ZERO = "zero"


@dataclass(frozen=True, slots=True)
class EvidenceCondition:
    """One prompt's worth of evidence policy for a single task."""

    condition_id: str
    condition: str
    k: int
    shown_supporting_fact_ids: tuple[str, ...]
    omitted_supporting_fact_ids: tuple[str, ...]

    @property
    def shown_signature(self) -> str:
        return "+".join(self.shown_supporting_fact_ids) or "none"

    def to_dict(self) -> dict[str, object]:
        return {
            "condition_id": self.condition_id,
            "condition": self.condition,
            "num_supporting_facts_shown": self.k,
            "supporting_fact_ids_shown": list(self.shown_supporting_fact_ids),
            "supporting_fact_ids_omitted": list(self.omitted_supporting_fact_ids),
        }


def build_evidence_conditions(
    supporting_fact_ids: Sequence[str],
    *,
    task_seed: int,
    include_zero: bool = True,
    max_subsets_per_k: int = 4,
) -> tuple[EvidenceCondition, ...]:
    """Every evidence condition for one task, in ascending ``k`` order.

    ``k = 0`` and ``k = L`` are single conditions.  Intermediate ``k`` enumerate
    all ``C(L, k)`` subsets when there are at most ``max_subsets_per_k`` of them,
    and otherwise a deterministic seeded sample of that many - so ``L = 4`` stays
    affordable without silently always testing the same prefix.
    """

    support = tuple(supporting_fact_ids)
    depth = len(support)
    if depth < 1:
        raise ValueError("a task must have at least one supporting fact")
    if max_subsets_per_k < 1:
        raise ValueError("max_subsets_per_k must be at least 1")

    conditions: list[EvidenceCondition] = []
    if include_zero:
        conditions.append(
            EvidenceCondition(
                condition_id="k0_none",
                condition=ZERO,
                k=0,
                shown_supporting_fact_ids=(),
                omitted_supporting_fact_ids=support,
            )
        )
    for k in range(1, depth):
        subsets = list(itertools.combinations(support, k))
        if len(subsets) > max_subsets_per_k:
            rng = random.Random(f"{task_seed}|evidence|k={k}")
            subsets = sorted(rng.sample(subsets, max_subsets_per_k))
        for subset in subsets:
            shown = tuple(subset)
            conditions.append(
                EvidenceCondition(
                    condition_id=f"k{k}_{'+'.join(shown)}",
                    condition=PARTIAL,
                    k=k,
                    shown_supporting_fact_ids=shown,
                    omitted_supporting_fact_ids=tuple(
                        fact_id for fact_id in support if fact_id not in set(shown)
                    ),
                )
            )
    conditions.append(
        EvidenceCondition(
            condition_id=f"k{depth}_{'+'.join(support)}",
            condition=FULL,
            k=depth,
            shown_supporting_fact_ids=support,
            omitted_supporting_fact_ids=(),
        )
    )
    return tuple(conditions)
