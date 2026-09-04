"""Controller-fact diversity, redundancy, and marginal posterior audits."""

from __future__ import annotations

import json
import itertools
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from mas_cc.musr_team_allocation_generator.ambiguity import (
    TeamAllocationCompletionIndex,
)
from mas_cc.musr_team_allocation_generator.io_utils import write_json_atomic
from mas_cc.musr_team_allocation_generator.symbolic_facts import CanonicalFact
from mas_cc.musr_team_allocation_generator.latent_problem import LATENT_VALUE_SUPPORT

_LOGICAL_WORLDS = tuple(itertools.product(LATENT_VALUE_SUPPORT, repeat=9))


def _logical_mask(fact: CanonicalFact) -> int:
    return sum(
        1 << position
        for position, vector in enumerate(_LOGICAL_WORLDS)
        if fact.holds(vector)
    )


def diversity_aware_ranking(
    facts: Sequence[CanonicalFact],
    *,
    false_target: int,
    truth: int,
    max_probability: float = 0.70,
    minimum_lift: float = 1e-12,
    maximum_length: int = 12,
    beam_width: int = 1_000,
) -> tuple[tuple[CanonicalFact, ...], tuple[dict[str, Any], ...]]:
    """Beam-search genuinely new, positive-marginal truthful information."""

    index = TeamAllocationCompletionIndex()
    all_logical = (1 << len(_LOGICAL_WORLDS)) - 1
    initial = index.metrics_for_facts(())
    masks = {fact.fact_id: _logical_mask(fact) for fact in facts}
    # selected indices, logical mask, profile, latent coverage, family coverage,
    # marginal deltas, diagnostic rows
    states = [((), all_logical, initial, frozenset(), frozenset(), (), ())]
    best = states[0]
    for _depth in range(maximum_length):
        expanded: dict[int, tuple[Any, ...]] = {}
        for selected, logical_mask, previous, covered, families, deltas, rows in states:
            for position, fact in enumerate(facts):
                if position in selected:
                    continue
                new_logical = logical_mask & masks[fact.fact_id]
                if new_logical == logical_mask:
                    continue
                profile = index.metrics_for_facts(
                    tuple(facts[index] for index in (*selected, position))
                )
                delta = (
                    profile.probabilities[false_target]
                    - previous.probabilities[false_target]
                )
                if (
                    profile.valid_completion_count == previous.valid_completion_count
                    or delta <= minimum_lift
                    or profile.probabilities[truth] <= 0
                    or profile.probabilities[false_target] > max_probability
                ):
                    continue
                latents = frozenset(
                    int(value) for value in fact.provenance.get("latent_indices", ())
                )
                family = _source_family(fact)
                diagnostic = {
                    "rank": len(selected) + 1,
                    "fact_id": fact.fact_id,
                    "new_latent_indices": sorted(latents - covered),
                    "new_predicate_family": family not in families,
                    "p_false_before": previous.probabilities[false_target],
                    "p_false_after": profile.probabilities[false_target],
                    "delta_p_false": delta,
                    "p_truth_after": profile.probabilities[truth],
                    "entropy_before": previous.normalized_entropy,
                    "entropy_after": profile.normalized_entropy,
                    "marginal_entropy_reduction": previous.normalized_entropy
                    - profile.normalized_entropy,
                    "compatible_worlds_before": previous.valid_completion_count,
                    "compatible_worlds_after": profile.valid_completion_count,
                    "logically_redundant": False,
                    "posterior_redundant": False,
                }
                state = (
                    (*selected, position),
                    new_logical,
                    profile,
                    covered | latents,
                    families | {family},
                    (*deltas, delta),
                    (*rows, diagnostic),
                )
                score = (
                    len(state[0]),
                    len(state[3]),
                    len(state[4]),
                    min(state[5]),
                    profile.probabilities[false_target],
                    profile.probabilities[truth],
                    tuple(-index for index in state[0]),
                )
                old = expanded.get(new_logical)
                if old is None or score > old[0]:
                    expanded[new_logical] = (score, state)
        if not expanded:
            break
        states = [
            value[1]
            for value in sorted(
                expanded.values(), key=lambda value: value[0], reverse=True
            )[:beam_width]
        ]
        best = max(
            states,
            key=lambda state: (
                len(state[0]),
                len(state[3]),
                len(state[4]),
                min(state[5]),
                state[2].probabilities[false_target],
                tuple(-index for index in state[0]),
            ),
        )
    return tuple(facts[index] for index in best[0]), tuple(best[6])


def _source_family(fact: CanonicalFact) -> str:
    indices = tuple(int(value) for value in fact.provenance.get("latent_indices", ()))
    domain = (
        "skill" if indices and all(index < 6 for index in indices) else "cooperation"
    )
    return f"{domain}:{fact.kind}:{fact.operator}"


def _implication_pairs(facts: Sequence[CanonicalFact]) -> list[dict[str, Any]]:
    rows = []
    for left in facts:
        for right in facts:
            if left.fact_id >= right.fact_id:
                continue
            same_left = left.left_index == right.left_index
            pair = None
            if same_left and {left.operator, right.operator} == {
                "eq_value",
                "ge_threshold",
            }:
                exact = left if left.operator == "eq_value" else right
                bound = right if left.operator == "eq_value" else left
                if int(exact.threshold) >= int(bound.threshold):
                    pair = f"{exact.fact_id} implies {bound.fact_id}"
            elif same_left and {left.operator, right.operator} == {
                "eq_value",
                "le_threshold",
            }:
                exact = left if left.operator == "eq_value" else right
                bound = right if left.operator == "eq_value" else left
                if int(exact.threshold) <= int(bound.threshold):
                    pair = f"{exact.fact_id} implies {bound.fact_id}"
            elif (
                left.left_index == right.left_index
                and left.right_index == right.right_index
                and {left.operator, right.operator} in ({"eq", "ge"}, {"eq", "le"})
            ):
                exact = left if left.operator == "eq" else right
                weak = right if left.operator == "eq" else left
                pair = f"{exact.fact_id} implies {weak.fact_id}"
            if pair:
                rows.append(
                    {"left": left.fact_id, "right": right.fact_id, "relationship": pair}
                )
    return rows


def build_diversity_audit(task_root: Path) -> dict[str, Any]:
    fact_rows = json.loads(
        (task_root / "facts/all_true_facts.json").read_text(encoding="utf-8")
    )
    facts = {str(row["fact_id"]): CanonicalFact.from_dict(row) for row in fact_rows}
    ranked = json.loads(
        (task_root / "controller/ranked_fact_pool.json").read_text(encoding="utf-8")
    )
    pool = [facts[str(row["fact_id"])] for row in ranked]
    task = json.loads((task_root / "task.json").read_text(encoding="utf-8"))
    false_index = int(str(task["false_target"]).rsplit("_", 1)[1])
    index = TeamAllocationCompletionIndex()
    prior = index.metrics_for_facts(()).probabilities[false_index]
    running: list[CanonicalFact] = []
    marginal = []
    previous = prior
    previous_entropy = index.metrics_for_facts(()).normalized_entropy
    for rank, fact in enumerate(pool, 1):
        running.append(fact)
        profile = index.metrics_for_facts(running)
        current = profile.probabilities[false_index]
        marginal.append(
            {
                "rank": rank,
                "fact_id": fact.fact_id,
                "p_false_before": previous,
                "p_false_after": current,
                "delta_p_false": current - previous,
                "entropy_before": previous_entropy,
                "entropy_after": profile.normalized_entropy,
                "marginal_entropy_reduction": previous_entropy
                - profile.normalized_entropy,
                "compatible_worlds": profile.valid_completion_count,
            }
        )
        previous = current
        previous_entropy = profile.normalized_entropy
    latent_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    proposition_counts: Counter[tuple[Any, ...]] = Counter()
    for fact in pool:
        latent_counts.update(
            str(index) for index in fact.provenance.get("latent_indices", ())
        )
        family_counts[_source_family(fact)] += 1
        proposition_counts[fact.logical_signature] += 1
    grouped: dict[tuple[int, ...], list[str]] = defaultdict(list)
    for fact in pool:
        grouped[
            tuple(int(value) for value in fact.provenance.get("latent_indices", ()))
        ].append(fact.fact_id)
    implication = _implication_pairs(pool)
    improved, improved_curve = diversity_aware_ranking(
        pool,
        false_target=false_index,
        truth=int(str(task["gold_target"]).rsplit("_", 1)[1]),
    )
    audit = {
        "task_id": task_root.name,
        "controller_fact_ids": len(pool),
        "distinct_logical_signatures": len(proposition_counts),
        "duplicate_logical_signatures": sum(
            count - 1 for count in proposition_counts.values()
        ),
        "distinct_latent_indices": len(latent_counts),
        "latent_index_counts": dict(sorted(latent_counts.items())),
        "relation_family_counts": dict(sorted(family_counts.items())),
        "source_groups_with_multiple_facts": {
            "|".join(map(str, key)): values
            for key, values in grouped.items()
            if len(values) > 1
        },
        "implication_or_subsumption_pairs": implication,
        "implication_pair_count": len(implication),
        "marginal_controller_order": marginal,
        "diversity_aware_informative_order": [fact.fact_id for fact in improved],
        "diversity_aware_curve": list(improved_curve),
        "effective_informative_additions": len(improved),
    }
    write_json_atomic(task_root / "controller/diversity_audit.json", audit)
    return audit


def apply_diversity_ranking(
    task_root: Path,
    *,
    budgets: Sequence[int] = (3, 6, 9, 12, 24),
) -> dict[str, Any]:
    """Freeze an informative prefix and recompute its exact diagnostic profiles."""

    audit = build_diversity_audit(task_root)
    informative_order = tuple(
        str(value) for value in audit["diversity_aware_informative_order"]
    )
    ranked_path = task_root / "controller/ranked_fact_pool.json"
    original = json.loads(ranked_path.read_text(encoding="utf-8"))
    by_id = {str(row["fact_id"]): row for row in original}
    tail = tuple(
        str(row["fact_id"])
        for row in original
        if str(row["fact_id"]) not in set(informative_order)
    )
    complete = (*informative_order, *tail)
    if max(budgets) > len(complete):
        raise ValueError(f"{task_root.name} has fewer facts than requested budget")
    write_json_atomic(
        ranked_path,
        [
            {
                **by_id[fact_id],
                "rank": rank,
                "score": float(len(complete) - rank + 1),
                "ranking_method": "diversity_positive_marginal_beam_v1",
            }
            for rank, fact_id in enumerate(complete, 1)
        ],
    )
    facts = {
        str(row["fact_id"]): CanonicalFact.from_dict(row)
        for row in json.loads(
            (task_root / "facts/all_true_facts.json").read_text(encoding="utf-8")
        )
    }
    task = json.loads((task_root / "task.json").read_text(encoding="utf-8"))
    false_target = int(str(task["false_target"]).rsplit("_", 1)[1])
    truth = int(str(task["gold_target"]).rsplit("_", 1)[1])
    index = TeamAllocationCompletionIndex()
    profiles = {}
    for budget in budgets:
        selected = complete[:budget]
        write_json_atomic(
            task_root / f"controller/selected_C{budget}.json", list(selected)
        )
        metrics = index.metrics_for_facts(tuple(facts[fact_id] for fact_id in selected))
        profiles[f"CONTROLLER_b{budget:02d}"] = {
            "posterior_vector": list(metrics.probabilities),
            "p_false": metrics.probabilities[false_target],
            "p_truth": metrics.probabilities[truth],
            "M": metrics.max_predictability,
            "Hbar": metrics.normalized_entropy,
            "compatible_worlds": metrics.valid_completion_count,
            "informative_prefix_exhausted": budget > len(informative_order),
            "informative_facts_in_prefix": min(budget, len(informative_order)),
        }
    write_json_atomic(
        task_root / "symbolic/controller_profiles_diversity_reranked.json", profiles
    )
    return {
        "task_id": task_root.name,
        "budgets": list(budgets),
        "informative_order_length": len(informative_order),
        "profiles": profiles,
    }


__all__ = [
    "apply_diversity_ranking",
    "build_diversity_audit",
    "diversity_aware_ranking",
]
