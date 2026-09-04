"""Provider-free candidate scan and frozen symbolic artifact writer."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from mas_cc.core import Seed
from mas_cc.musr_team_allocation_generator.ambiguity import (
    TeamAllocationCompletionIndex,
)
from mas_cc.musr_team_allocation_generator.io_utils import (
    sha256_object,
    write_json_atomic,
)
from mas_cc.musr_team_allocation_generator.latent_problem import (
    LATENT_VALUE_PRIOR,
    LATENT_VALUE_SUPPORT,
    problem_from_latent_values,
)
from mas_cc.musr_team_allocation_generator.selective_design import (
    SelectiveTaskDesign,
    build_selective_design,
)
from mas_cc.musr_team_allocation_generator.symbolic_facts import true_canonical_facts

from .config import TruthfulSelectiveConfig


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _balanced_accept(
    accepted: Sequence[SelectiveTaskDesign], count: int
) -> tuple[SelectiveTaskDesign, ...]:
    selected: list[SelectiveTaskDesign] = []
    pair_counts: Counter[tuple[int, int]] = Counter()
    gold_counts: Counter[int] = Counter()
    target_counts: Counter[int] = Counter()
    for design in accepted:
        pair = (design.problem.gold_index, design.false_target_index)
        prospective = (
            pair_counts[pair],
            gold_counts[pair[0]],
            target_counts[pair[1]],
        )
        alternatives = [
            (pair_counts[(g, z)], gold_counts[g], target_counts[z])
            for g in range(3)
            for z in range(3)
            if g != z
        ]
        if prospective > min(alternatives):
            continue
        selected.append(design)
        pair_counts[pair] += 1
        gold_counts[pair[0]] += 1
        target_counts[pair[1]] += 1
        if len(selected) == count:
            return tuple(selected)
    return tuple(selected)


def _fact_payload(fact: Any, role: str) -> dict[str, Any]:
    return {**fact.to_dict(), "role": role}


def write_design_artifacts(
    root: Path,
    design: SelectiveTaskDesign,
    *,
    task_id: str,
    candidate_id: int,
    seed: int,
) -> dict[str, Any]:
    task_root = root / task_id
    controller = {fact.fact_id for fact in design.controller_facts}
    decisive = {fact.fact_id for fact in design.decisive_facts}
    selected_order = tuple(
        fact.fact_id
        for fact in design.selected_controller[max(design.selected_controller)]
    )
    remaining_order = tuple(
        fact.fact_id
        for fact in design.controller_facts
        if fact.fact_id not in set(selected_order)
    )
    authoritative_order = (*selected_order, *remaining_order)
    all_rows = [
        _fact_payload(
            fact,
            "controller-compatible"
            if fact.fact_id in controller
            else "decisive"
            if fact.fact_id in decisive
            else "neutral",
        )
        for fact in design.facts
    ]
    files: dict[str, Any] = {
        "hidden_world.json": design.problem.to_dict(),
        "facts/all_true_facts.json": all_rows,
        "facts/controller_reportable_facts.json": [
            row for row in all_rows if row["role"] == "controller-compatible"
        ],
        "facts/decisive_facts.json": [
            row for row in all_rows if row["role"] == "decisive"
        ],
        "facts/neutral_facts.json": [
            row for row in all_rows if row["role"] == "neutral"
        ],
        "controller/ranked_fact_pool.json": [
            {
                "rank": rank,
                "fact_id": fact_id,
                "score": float(len(authoritative_order) - rank + 1),
                "strategy_class": "target-compatible",
            }
            for rank, fact_id in enumerate(authoritative_order, 1)
        ],
        "private/N24_assignment.json": {
            "schema_version": "musr_truthful_selective_private_v1",
            "population_size": 24,
            "agent_assignments": {
                f"agent_{index + 1:03d}": [fact.fact_id for fact in packet]
                for index, packet in enumerate(design.private_assignment)
            },
            "profiles": design.profiles["PRIVATE"],
        },
        "symbolic/zero_profile.json": design.profiles["ZERO"],
        "symbolic/private_profiles.json": design.profiles["PRIVATE"],
        "symbolic/controller_profiles.json": {
            key: value
            for key, value in design.profiles.items()
            if key.startswith("CONTROLLER_b") and "+" not in key
        },
        "symbolic/decisive_profiles.json": {"DECISIVE": design.profiles["DECISIVE"]},
        "symbolic/mixed_profiles.json": {
            key: value for key, value in design.profiles.items() if "+DECISIVE" in key
        },
        "symbolic/full_profile.json": design.profiles["FULL"],
        "symbolic/robustness_by_subset.json": design.robustness,
        "symbolic/individual_controller_fact_audit.json": list(
            design.individual_controller_audit
        ),
    }
    for budget, facts in design.selected_controller.items():
        files[f"controller/selected_C{budget}.json"] = [fact.fact_id for fact in facts]
    for relative, payload in files.items():
        write_json_atomic(task_root / relative, payload)
    task = {
        "schema_version": "musr_team_allocation_selective_v1",
        "task_id": task_id,
        "candidate_id": candidate_id,
        "seed": seed,
        "gold_target": design.gold_target,
        "false_target": design.false_target,
        "controller_eligible_fact_count": len(design.controller_facts),
        "decisive_fact_count": len(design.decisive_facts),
        "artifact_files": sorted(files),
    }
    task["task_hash"] = sha256_object(task)
    write_json_atomic(task_root / "task.json", task)
    return task


def repair_controller_ranking(task_root: Path) -> None:
    """Make the ranked pool's first 24 facts equal the frozen calibrated C24 order."""

    ranked_path = task_root / "controller/ranked_fact_pool.json"
    ranked = json.loads(ranked_path.read_text(encoding="utf-8"))
    c24 = tuple(
        str(value)
        for value in json.loads(
            (task_root / "controller/selected_C24.json").read_text(encoding="utf-8")
        )
    )
    by_id = {str(row["fact_id"]): row for row in ranked}
    tail = tuple(
        str(row["fact_id"]) for row in ranked if str(row["fact_id"]) not in set(c24)
    )
    order = (*c24, *tail)
    repaired = [
        {
            **by_id[fact_id],
            "rank": rank,
            "score": float(len(order) - rank + 1),
        }
        for rank, fact_id in enumerate(order, 1)
    ]
    write_json_atomic(ranked_path, repaired)


def run_symbolic_scan(config: TruthfulSelectiveConfig, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    rng = Seed(config.seed).derive("truthful-selective-candidates").create_random()
    weights = tuple(weight for _, weight in LATENT_VALUE_PRIOR)
    index = TeamAllocationCompletionIndex()
    failures: Counter[str] = Counter()
    accepted: list[tuple[int, SelectiveTaskDesign]] = []
    gate_order = (
        "unique_gold",
        "zero_ambiguity",
        "private_ambiguity",
        "decisive_recovery",
        "controller_fact_pool",
        "individual_controller_viability",
        "controller_C3_profile",
        "controller_C6_profile",
        "controller_C12_profile",
        "controller_C24_profile",
        "controller_subset_robustness",
        "mixed_recovery",
        "full_recovery",
    )
    candidate_rows: list[dict[str, Any]] = []
    for candidate_id in range(1, config.candidate_worlds + 1):
        vector = tuple(rng.choices(LATENT_VALUE_SUPPORT, weights=weights, k=9))
        problem = problem_from_latent_values(vector)
        if problem.candidate_scores.count(max(problem.candidate_scores)) != 1:
            failures["unique_gold"] += 1
            candidate_rows.append(
                {
                    "candidate_id": candidate_id,
                    "latent_values": "|".join(map(str, vector)),
                    "gold_index": -1,
                    "score_margin": problem.margin_to_second_best,
                    "passed": False,
                    "failure_reason": "unique_gold",
                }
            )
            continue
        true_fact_count = len(true_canonical_facts(problem))
        try:
            design = build_selective_design(
                problem,
                index,
                config.symbolic,
                seed=int(Seed(config.seed).derive(f"candidate:{candidate_id}")),
                evaluate_robustness=False,
            )
        except ValueError as exc:
            reason = str(exc)
            failures[reason] += 1
            candidate_rows.append(
                {
                    "candidate_id": candidate_id,
                    "latent_values": "|".join(map(str, vector)),
                    "gold_index": problem.gold_index,
                    "score_margin": problem.margin_to_second_best,
                    "true_fact_count": true_fact_count,
                    "passed": False,
                    "failure_reason": reason,
                }
            )
            continue
        try:
            robust_design = build_selective_design(
                problem,
                index,
                config.symbolic,
                seed=int(Seed(config.seed).derive(f"candidate:{candidate_id}")),
                false_target_index=design.false_target_index,
                evaluate_robustness=True,
            )
        except ValueError as exc:
            reason = str(exc)
            failures[reason] += 1
            candidate_rows.append(
                {
                    "candidate_id": candidate_id,
                    "latent_values": "|".join(map(str, vector)),
                    "gold_index": problem.gold_index,
                    "false_target_index": design.false_target_index,
                    "score_margin": problem.margin_to_second_best,
                    "controller_fact_count": len(design.controller_facts),
                    "decisive_fact_count": len(design.decisive_facts),
                    "passed": False,
                    "failure_reason": reason,
                }
            )
            continue
        accepted.append((candidate_id, robust_design))
        candidate_rows.append(
            {
                "candidate_id": candidate_id,
                "latent_values": "|".join(map(str, vector)),
                "gold_index": problem.gold_index,
                "false_target_index": robust_design.false_target_index,
                "score_margin": problem.margin_to_second_best,
                "true_fact_count": true_fact_count,
                "controller_fact_count": len(robust_design.controller_facts),
                "decisive_fact_count": len(robust_design.decisive_facts),
                "passed": True,
                "failure_reason": "",
            }
        )
    balanced = _balanced_accept(
        [design for _, design in accepted], config.development_tasks
    )
    if len(balanced) != config.development_tasks:
        balanced = tuple(design for _, design in accepted[: config.development_tasks])
    if len(balanced) != config.development_tasks:
        raise RuntimeError("symbolic scan did not produce enough development tasks")
    selected: list[dict[str, Any]] = []
    for index_position, design in enumerate(balanced, 1):
        candidate_id = next(candidate for candidate, item in accepted if item is design)
        selected.append(
            write_design_artifacts(
                output.parent / "tasks",
                design,
                task_id=f"task_{index_position:03d}",
                candidate_id=candidate_id,
                seed=int(Seed(config.seed).derive(f"task:{index_position}")),
            )
        )
    _write_csv(output / "candidate_worlds.csv", candidate_rows)
    failure_order = {
        "unique_gold": 0,
        "zero_ambiguity": 1,
        "private_ambiguity": 2,
        "decisive_recovery": 3,
        "controller_fact_pool": 4,
        "individual_controller_viability": 5,
        "controller_C3_profile": 6,
        "controller_C6_profile": 7,
        "controller_C12_profile": 8,
        "controller_C24_profile": 9,
        "controller_subset_robustness": 10,
        "mixed_recovery": 11,
        "full_recovery": 12,
    }
    gate_passes = {
        gate: sum(
            row.get("passed") is True
            or failure_order.get(str(row.get("failure_reason")), -1) > position
            for row in candidate_rows
        )
        for position, gate in enumerate(gate_order)
    }
    funnel = [
        {
            "gate": gate,
            "passed": gate_passes[gate],
            "failed_at_gate": failures.get(gate, 0),
            "pass_fraction_all": gate_passes[gate] / config.candidate_worlds,
            "conditional_pass_fraction": (
                gate_passes[gate] / config.candidate_worlds
                if position == 0
                else gate_passes[gate] / gate_passes[gate_order[position - 1]]
                if gate_passes[gate_order[position - 1]]
                else 0.0
            ),
        }
        for position, gate in enumerate(gate_order)
    ]
    summary = {
        "schema_version": 1,
        "candidate_worlds_scanned": config.candidate_worlds,
        "symbolic_pass_count": len(accepted),
        "symbolic_pass_rate": len(accepted) / config.candidate_worlds,
        "failure_reasons": dict(sorted(failures.items())),
        "cumulative_gate_passes": gate_passes,
        "gate_funnel": funnel,
        "gate_order": list(gate_order),
        "full_profile_candidates_evaluated": gate_passes["zero_ambiguity"],
        "selected_tasks": selected,
        "thresholds": config.to_dict()["symbolic_thresholds"],
    }
    summary["fingerprint_sha256"] = sha256_object(summary)
    write_json_atomic(output / "scan_summary.json", summary)
    return summary


__all__ = ["run_symbolic_scan", "write_design_artifacts"]
