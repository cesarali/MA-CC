"""Exact design of truthful selective-disclosure Team Allocation tasks."""

from __future__ import annotations

import random
import statistics
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from mas_cc.core import Seed

from .ambiguity import PrivateViewMetrics, TeamAllocationCompletionIndex
from .latent_problem import latent_values
from .schemas import LatentProblem
from .symbolic_facts import CanonicalFact, true_canonical_facts


def _profile(metrics: PrivateViewMetrics) -> dict[str, Any]:
    return {
        "posterior_vector": list(metrics.probabilities),
        "M": metrics.max_predictability,
        "Hbar": metrics.normalized_entropy,
        "compatible_worlds": metrics.valid_completion_count,
        "valid_probability_mass": metrics.valid_probability_mass,
    }


@dataclass(frozen=True, slots=True)
class SelectiveThresholds:
    zero_max_probability: float = 0.45
    zero_min_entropy: float = 0.90
    private_max_probability: float = 0.45
    private_min_entropy: float = 0.90
    minimum_controller_facts: int = 24
    controller_budgets: tuple[int, ...] = (3, 6, 12, 24)
    controller_min_lift: float = 1e-12
    controller_max_false_probability: float = 0.70
    decisive_min_truth_probability: float = 0.80
    subset_positive_lift_fraction: float = 0.70
    subset_samples: int = 64
    population_size: int = 24
    private_assignments_tested: int = 32
    private_facts_per_agent: int = 1
    minimum_private_holders: int = 1


@dataclass(frozen=True, slots=True)
class SelectiveTaskDesign:
    problem: LatentProblem
    false_target_index: int
    facts: tuple[CanonicalFact, ...]
    controller_facts: tuple[CanonicalFact, ...]
    decisive_facts: tuple[CanonicalFact, ...]
    neutral_facts: tuple[CanonicalFact, ...]
    selected_controller: Mapping[int, tuple[CanonicalFact, ...]]
    private_assignment: tuple[tuple[CanonicalFact, ...], ...]
    profiles: Mapping[str, Any]
    robustness: Mapping[str, Any]
    individual_controller_audit: tuple[Mapping[str, Any], ...]

    @property
    def gold_target(self) -> str:
        return f"ALLOCATION_{self.problem.gold_index}"

    @property
    def false_target(self) -> str:
        return f"ALLOCATION_{self.false_target_index}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "gold_target": self.gold_target,
            "false_target": self.false_target,
            "all_true_facts": [fact.to_dict() for fact in self.facts],
            "controller_reportable_fact_ids": [
                fact.fact_id for fact in self.controller_facts
            ],
            "decisive_fact_ids": [fact.fact_id for fact in self.decisive_facts],
            "neutral_fact_ids": [fact.fact_id for fact in self.neutral_facts],
            "selected_controller": {
                str(budget): [fact.fact_id for fact in facts]
                for budget, facts in self.selected_controller.items()
            },
            "private_assignment": [
                [fact.fact_id for fact in packet] for packet in self.private_assignment
            ],
            "profiles": dict(self.profiles),
            "robustness": dict(self.robustness),
            "individual_controller_audit": list(self.individual_controller_audit),
        }


def _greedy_decisive(
    index: TeamAllocationCompletionIndex,
    candidates: Sequence[CanonicalFact],
    gold: int,
) -> tuple[CanonicalFact, ...] | None:
    chosen: list[CanonicalFact] = []
    remaining = list(candidates)
    while remaining:
        ranked = []
        for fact in remaining:
            metrics = index.metrics_for_facts((*chosen, fact))
            ranked.append(
                (
                    metrics.probabilities[gold],
                    -metrics.valid_completion_count,
                    fact.fact_id,
                    fact,
                    metrics,
                )
            )
        _, _, _, selected, metrics = max(ranked, key=lambda row: row[:3])
        chosen.append(selected)
        remaining.remove(selected)
        if metrics.probabilities[gold] == 1.0:
            return tuple(chosen)
    return None


def _private_eligible(
    facts: Sequence[CanonicalFact],
    index: TeamAllocationCompletionIndex,
    thresholds: SelectiveThresholds,
) -> tuple[CanonicalFact, ...]:
    return tuple(
        fact
        for fact in facts
        if (metrics := index.metrics_for_facts((fact,))).max_predictability
        <= thresholds.private_max_probability
        and metrics.normalized_entropy >= thresholds.private_min_entropy
    )


def _private_assignment(
    eligible: Sequence[CanonicalFact],
    index: TeamAllocationCompletionIndex,
    thresholds: SelectiveThresholds,
    seed: int,
    required_facts: Sequence[CanonicalFact],
) -> tuple[tuple[CanonicalFact, ...], ...] | None:
    required = thresholds.population_size * thresholds.private_facts_per_agent
    if (
        not eligible
        or len(required_facts) * thresholds.minimum_private_holders > required
    ):
        return None
    if thresholds.private_facts_per_agent == 1:
        rows = list(eligible)
        random.Random(f"{seed}:private").shuffle(rows)
        slots = [
            fact
            for fact in required_facts
            for _ in range(thresholds.minimum_private_holders)
        ]
        cursor = 0
        while len(slots) < required:
            slots.append(rows[cursor % len(rows)])
            cursor += 1
        return tuple((fact,) for fact in slots)
    for attempt in range(thresholds.private_assignments_tested):
        rng = random.Random(f"{seed}:private:{attempt}")
        rows = list(eligible)
        rng.shuffle(rows)
        slots = [
            fact
            for fact in required_facts
            for _ in range(thresholds.minimum_private_holders)
        ]
        while len(slots) < required:
            slots.append(rng.choice(rows))
        rng.shuffle(slots)
        assignment = tuple(
            tuple(slots[start : start + thresholds.private_facts_per_agent])
            for start in range(0, required, thresholds.private_facts_per_agent)
        )
        if len(assignment) == thresholds.population_size and all(
            index.metrics_for_facts(packet).max_predictability
            <= thresholds.private_max_probability
            and index.metrics_for_facts(packet).normalized_entropy
            >= thresholds.private_min_entropy
            for packet in assignment
        ):
            return assignment
    return None


def _robustness(
    index: TeamAllocationCompletionIndex,
    pool: Sequence[CanonicalFact],
    false_target: int,
    truth: int,
    zero_false: float,
    thresholds: SelectiveThresholds,
    seed: int,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for budget in thresholds.controller_budgets:
        rng = random.Random(f"{seed}:controller-subsets:{budget}")
        subsets: set[tuple[int, ...]] = set()
        attempts = 0
        while (
            len(subsets) < thresholds.subset_samples
            and attempts < thresholds.subset_samples * 50
        ):
            attempts += 1
            subsets.add(tuple(sorted(rng.sample(range(len(pool)), budget))))
        rows = [
            index.metrics_for_facts(tuple(pool[position] for position in subset))
            for subset in sorted(subsets)
        ]
        false_values = [row.probabilities[false_target] for row in rows]
        output[str(budget)] = {
            "subsets_tested": len(rows),
            "p_false_values": false_values,
            "mean_p_false": statistics.fmean(false_values),
            "median_p_false": statistics.median(false_values),
            "min_p_false": min(false_values),
            "max_p_false": max(false_values),
            "std_p_false": statistics.pstdev(false_values),
            "mean_Hbar": statistics.fmean(row.normalized_entropy for row in rows),
            "fraction_positive_false_target_lift": statistics.fmean(
                value > zero_false for value in false_values
            ),
            "fraction_eliminate_truth": statistics.fmean(
                row.probabilities[truth] == 0 for row in rows
            ),
        }
    return output


def build_selective_design(
    problem: LatentProblem,
    index: TeamAllocationCompletionIndex,
    thresholds: SelectiveThresholds,
    *,
    seed: int,
    false_target_index: int | None = None,
    evaluate_robustness: bool = True,
) -> SelectiveTaskDesign:
    """Build one design or raise ``ValueError`` with its first failed gate."""

    if problem.candidate_scores.count(max(problem.candidate_scores)) != 1:
        raise ValueError("unique_gold")
    truth = problem.gold_index
    zero = index.metrics_for_facts(())
    if (
        zero.max_predictability > thresholds.zero_max_probability
        or zero.normalized_entropy < thresholds.zero_min_entropy
    ):
        raise ValueError("zero_ambiguity")
    facts = true_canonical_facts(problem)
    full_profile = index.metrics_for_facts(facts)
    if full_profile.probabilities[truth] != 1.0:
        raise ValueError("full_recovery")
    private_eligible = _private_eligible(facts, index, thresholds)
    if not private_eligible:
        raise ValueError("private_ambiguity")
    decisive = _greedy_decisive(index, private_eligible, truth)
    if decisive is None:
        raise ValueError("decisive_recovery")
    private = _private_assignment(private_eligible, index, thresholds, seed, decisive)
    if private is None:
        raise ValueError("private_ambiguity")
    decisive_profile = index.metrics_for_facts(decisive)
    if (
        decisive_profile.probabilities[truth]
        < thresholds.decisive_min_truth_probability
    ):
        raise ValueError("decisive_recovery")

    targets = (
        (false_target_index,)
        if false_target_index is not None
        else tuple(index for index in range(3) if index != truth)
    )
    chosen: (
        tuple[
            int,
            tuple[CanonicalFact, ...],
            Mapping[int, tuple[CanonicalFact, ...]],
            Mapping[str, Any],
            tuple[Mapping[str, Any], ...],
        ]
        | None
    ) = None
    decisive_ids = {fact.fact_id for fact in decisive}
    world_vector = latent_values(problem)
    worlds = index.worlds
    target_failures: list[str] = []
    for target in targets:
        if target is None or target == truth:
            continue
        eligible: list[tuple[CanonicalFact, Mapping[str, Any]]] = []
        for fact in facts:
            if fact.fact_id in decisive_ids:
                continue
            metrics = index.metrics_for_facts((fact,))
            if (
                metrics.probabilities[target] - zero.probabilities[target]
                >= thresholds.controller_min_lift
                and metrics.probabilities[target] > 0
                and metrics.probabilities[truth] > 0
                and metrics.probabilities[target] < 1
            ):
                eligible.append(
                    (
                        fact,
                        {
                            "fact_id": fact.fact_id,
                            "canonical_fact_text": fact.canonical_text,
                            "exact_provenance": dict(fact.provenance),
                            **_profile(metrics),
                            "compatible_allocations": [
                                f"ALLOCATION_{idx}"
                                for idx, probability in enumerate(metrics.probabilities)
                                if probability > 0
                            ],
                        },
                    )
                )
        target_worlds = [
            position
            for position, (_, winner, _) in enumerate(worlds)
            if winner == target
        ]
        if not eligible or not target_worlds:
            target_failures.append("individual_controller_viability")
            continue
        witness = max(
            target_worlds,
            key=lambda position: sum(
                fact.holds(worlds[position][0]) for fact, _ in eligible
            ),
        )
        pool = tuple(fact for fact, _ in eligible if fact.holds(worlds[witness][0]))
        if len(pool) < thresholds.minimum_controller_facts:
            target_failures.append("controller_fact_pool")
            continue
        ranked_pool = tuple(
            sorted(
                pool,
                key=lambda fact: (
                    -index.metrics_for_facts((fact,)).probabilities[target],
                    fact.fact_id,
                ),
            )
        )
        selected: dict[int, tuple[CanonicalFact, ...]] = {}
        current: list[CanonicalFact] = []
        remaining = list(ranked_pool)
        valid = True
        failed_budget = 0
        for position in range(max(thresholds.controller_budgets)):
            candidates = []
            for fact in remaining:
                metrics = index.metrics_for_facts((*current, fact))
                p_false = metrics.probabilities[target]
                if (
                    p_false <= thresholds.controller_max_false_probability
                    and metrics.probabilities[truth] > 0
                    and p_false - zero.probabilities[target]
                    >= thresholds.controller_min_lift
                ):
                    candidates.append(
                        (
                            p_false,
                            metrics.probabilities[truth],
                            metrics.valid_completion_count,
                            fact.fact_id,
                            fact,
                        )
                    )
            if not candidates:
                valid = False
                failed_budget = next(
                    (
                        budget
                        for budget in thresholds.controller_budgets
                        if budget >= position + 1
                    ),
                    max(thresholds.controller_budgets),
                )
                break
            fact = max(candidates, key=lambda row: row[:4])[-1]
            current.append(fact)
            remaining.remove(fact)
            if position + 1 in thresholds.controller_budgets:
                selected[position + 1] = tuple(current)
        if not valid:
            target_failures.append(f"controller_C{failed_budget}_profile")
            continue
        robustness: Mapping[str, Any] = {}
        if evaluate_robustness:
            robustness = _robustness(
                index,
                ranked_pool,
                target,
                truth,
                zero.probabilities[target],
                thresholds,
                seed,
            )
            if any(
                row["fraction_eliminate_truth"] != 0
                or row["fraction_positive_false_target_lift"]
                < thresholds.subset_positive_lift_fraction
                for row in robustness.values()
            ):
                target_failures.append("controller_subset_robustness")
                continue
        if any(
            index.metrics_for_facts((*packet, *decisive)).probabilities[truth] != 1.0
            for packet in selected.values()
        ):
            target_failures.append("mixed_recovery")
            continue
        audit_by_id = {str(row["fact_id"]): row for _, row in eligible}
        audit = tuple(audit_by_id[fact.fact_id] for fact in ranked_pool)
        chosen = (target, ranked_pool, selected, robustness, audit)
        break
    if chosen is None:
        priority = {
            "individual_controller_viability": 0,
            "controller_fact_pool": 1,
            "controller_C3_profile": 2,
            "controller_C6_profile": 3,
            "controller_C12_profile": 4,
            "controller_C24_profile": 5,
            "controller_subset_robustness": 6,
            "mixed_recovery": 7,
        }
        raise ValueError(
            max(target_failures, key=lambda reason: priority[reason])
            if target_failures
            else "individual_controller_viability"
        )
    target, controller, selected, robustness, audit = chosen
    selected_ids = {fact.fact_id for fact in controller} | decisive_ids
    neutral = tuple(fact for fact in facts if fact.fact_id not in selected_ids)
    profiles: dict[str, Any] = {
        "ZERO": _profile(zero),
        "PRIVATE": [
            {
                "agent_id": f"agent_{position + 1:03d}",
                "private_fact_ids": [fact.fact_id for fact in packet],
                **_profile(index.metrics_for_facts(packet)),
            }
            for position, packet in enumerate(private)
        ],
        "DECISIVE": _profile(decisive_profile),
        "FULL": _profile(full_profile),
    }
    for budget, packet in selected.items():
        profiles[f"CONTROLLER_b{budget:02d}"] = _profile(
            index.metrics_for_facts(packet)
        )
        profiles[f"CONTROLLER_b{budget:02d}+DECISIVE"] = _profile(
            index.metrics_for_facts((*packet, *decisive))
        )
    return SelectiveTaskDesign(
        problem,
        target,
        facts,
        controller,
        decisive,
        neutral,
        selected,
        private,
        profiles,
        robustness,
        audit,
    )


def scan_selective_worlds(
    vectors: Sequence[Sequence[int]],
    index: TeamAllocationCompletionIndex,
    thresholds: SelectiveThresholds,
    *,
    seed: int,
    limit: int | None = None,
) -> tuple[tuple[SelectiveTaskDesign, ...], Mapping[str, int]]:
    passes: list[SelectiveTaskDesign] = []
    failures: Counter[str] = Counter()
    for candidate, vector in enumerate(vectors, 1):
        try:
            design = build_selective_design(
                __import__(
                    "mas_cc.musr_team_allocation_generator.latent_problem",
                    fromlist=["problem_from_latent_values"],
                ).problem_from_latent_values(vector),
                index,
                thresholds,
                seed=int(Seed(seed).derive(f"candidate:{candidate}")),
            )
        except ValueError as exc:
            failures[str(exc)] += 1
            continue
        passes.append(design)
        if limit is not None and len(passes) >= limit:
            break
    return tuple(passes), dict(failures)


__all__ = [
    "SelectiveTaskDesign",
    "SelectiveThresholds",
    "build_selective_design",
    "scan_selective_worlds",
]
