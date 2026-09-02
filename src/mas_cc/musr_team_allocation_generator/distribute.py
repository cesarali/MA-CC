"""Balanced distribution of coherent evidence cards over a population."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Sequence

from .schemas import EvidenceCard, LatentProblem
from .validate import agent_can_certify_unique_allocation


def _target_assignments(card_count: int, population_size: int, redundancy: int) -> int:
    return card_count * redundancy


def distribute_evidence(
    cards: Sequence[EvidenceCard],
    problem: LatentProblem,
    *,
    population_size: int,
    redundancy: int,
    rng: random.Random,
    min_cards_per_agent: int = 1,
    max_attempts: int = 1_000,
) -> dict[str, list[str]]:
    if population_size < 2:
        raise ValueError("population_size must be at least 2")
    if not 1 <= redundancy < population_size:
        raise ValueError("redundancy must satisfy 1 <= redundancy < population_size")
    if not cards:
        raise ValueError("at least one evidence card is required")
    if _target_assignments(len(cards), population_size, redundancy) < (
        population_size * min_cards_per_agent
    ):
        raise ValueError(
            "too few card assignments to give every agent the requested minimum"
        )

    card_to_fact = {card.evidence_id: card.latent_fact_id for card in cards}
    all_fact_ids = frozenset(card_to_fact.values())
    for _ in range(max_attempts):
        holdings = {str(agent): [] for agent in range(population_size)}
        loads = Counter({str(agent): 0 for agent in range(population_size)})
        fact_loads: dict[str, Counter[str]] = defaultdict(Counter)
        shuffled = list(cards)
        rng.shuffle(shuffled)
        for card in shuffled:
            ranked = list(holdings)
            rng.shuffle(ranked)
            ranked.sort(
                key=lambda agent: (fact_loads[agent][card.latent_fact_id], loads[agent])
            )
            for agent in ranked[:redundancy]:
                holdings[agent].append(card.evidence_id)
                loads[agent] += 1
                fact_loads[agent][card.latent_fact_id] += 1
        held_fact_sets = {
            agent: {card_to_fact[evidence_id] for evidence_id in evidence_ids}
            for agent, evidence_ids in holdings.items()
        }
        if min(loads.values()) < min_cards_per_agent:
            continue
        if any(facts == all_fact_ids for facts in held_fact_sets.values()):
            continue
        if any(
            agent_can_certify_unique_allocation(problem, facts)[0]
            for facts in held_fact_sets.values()
        ):
            continue
        return {agent: sorted(evidence_ids) for agent, evidence_ids in holdings.items()}
    raise RuntimeError(
        "could not distribute evidence while preserving the no-single-agent structural guard"
    )
