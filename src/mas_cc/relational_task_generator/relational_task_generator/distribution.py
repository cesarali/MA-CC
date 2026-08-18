"""Fact-assignment utilities for distributed-information tasks."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Sequence


def _agent_ids(population_size: int) -> List[str]:
    if population_size < 1:
        raise ValueError("population_size must be >= 1")
    width = max(3, len(str(population_size)))
    return [f"agent_{i:0{width}d}" for i in range(1, population_size + 1)]


def validate_distribution_parameters(
    *,
    population_size: int,
    reasoning_depth: int,
    support_redundancy: int,
    distractor_redundancy: int,
    no_single_agent_solution: bool,
) -> None:
    """Reject impossible assignment configurations before sampling."""
    if population_size < 1:
        raise ValueError("population_size must be >= 1")
    if reasoning_depth < 1:
        raise ValueError("reasoning_depth must be >= 1")
    if not 1 <= support_redundancy <= population_size:
        raise ValueError(
            "support_redundancy must be between 1 and population_size"
        )
    if not 1 <= distractor_redundancy <= population_size:
        raise ValueError(
            "distractor_redundancy must be between 1 and population_size"
        )

    if no_single_agent_solution:
        if reasoning_depth == 1:
            raise ValueError(
                "no_single_agent_solution=true is impossible for reasoning_depth=1: "
                "any agent who receives the sole supporting fact can solve the task."
            )
        # Each agent can receive at most L-1 of the L supporting facts.
        total_required_slots = reasoning_depth * support_redundancy
        total_safe_capacity = population_size * (reasoning_depth - 1)
        if total_required_slots > total_safe_capacity:
            raise ValueError(
                "Requested support redundancy is incompatible with "
                "no_single_agent_solution=true. Need "
                f"L*r <= N*(L-1), but got {reasoning_depth}*{support_redundancy} "
                f"> {population_size}*{reasoning_depth - 1}."
            )


def distribute_facts(
    *,
    supporting_fact_ids: Sequence[str],
    distractor_fact_ids: Sequence[str],
    population_size: int,
    support_redundancy: int,
    distractor_redundancy: int,
    no_single_agent_solution: bool,
    rng: random.Random,
) -> Dict[str, Dict[str, List[str]]]:
    """Assign fact IDs to agents reproducibly.

    Every supporting fact is assigned to exactly ``support_redundancy`` distinct
    agents.  Every distractor is assigned to exactly ``distractor_redundancy``
    distinct agents.

    Under ``no_single_agent_solution=True``, each agent is given at most L-1 of
    the L supporting facts.  The constructive assignment keeps support loads
    balanced, which makes every feasible v1 configuration straightforward to
    realize.
    """
    L = len(supporting_fact_ids)
    validate_distribution_parameters(
        population_size=population_size,
        reasoning_depth=L,
        support_redundancy=support_redundancy,
        distractor_redundancy=distractor_redundancy,
        no_single_agent_solution=no_single_agent_solution,
    )

    agent_ids = _agent_ids(population_size)
    assignments: Dict[str, List[str]] = {agent_id: [] for agent_id in agent_ids}

    if no_single_agent_solution:
        support_load = {agent_id: 0 for agent_id in agent_ids}
        capacity = L - 1
        # Randomize fact order while preserving final output order later.
        support_order = list(supporting_fact_ids)
        rng.shuffle(support_order)
        for fact_id in support_order:
            eligible = [a for a in agent_ids if support_load[a] < capacity]
            if len(eligible) < support_redundancy:
                raise RuntimeError(
                    "Internal assignment failure despite a feasible configuration."
                )
            # Balance support loads; random tie-breaking preserves seed dependence.
            tie_break = {a: rng.random() for a in eligible}
            eligible.sort(key=lambda a: (support_load[a], tie_break[a]))
            chosen = eligible[:support_redundancy]
            for agent_id in chosen:
                assignments[agent_id].append(fact_id)
                support_load[agent_id] += 1
    else:
        for fact_id in supporting_fact_ids:
            chosen = rng.sample(agent_ids, k=support_redundancy)
            for agent_id in chosen:
                assignments[agent_id].append(fact_id)

    for fact_id in distractor_fact_ids:
        chosen = rng.sample(agent_ids, k=distractor_redundancy)
        for agent_id in chosen:
            assignments[agent_id].append(fact_id)

    # Canonicalize fact order by numeric fact index where possible.
    def fact_key(fid: str) -> tuple[int, str]:
        try:
            return (int(fid.lstrip("f")), fid)
        except ValueError:
            return (10**9, fid)

    return {
        agent_id: {"fact_ids": sorted(fact_ids, key=fact_key)}
        for agent_id, fact_ids in assignments.items()
    }


def fact_recipient_counts(
    agents: Mapping[str, Mapping[str, Iterable[str]]]
) -> Dict[str, int]:
    """Count how many agents receive each fact ID."""
    counts: Dict[str, int] = defaultdict(int)
    for payload in agents.values():
        for fact_id in set(payload.get("fact_ids", [])):
            counts[fact_id] += 1
    return dict(counts)
