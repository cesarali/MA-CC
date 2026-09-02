"""Deterministic symbolic scan and frozen construction-rule selection."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mas_cc.core import Seed
from mas_cc.musr_team_allocation_generator.ambiguity import (
    PrivateViewMetrics,
    TeamAllocationCompletionIndex,
    choose_private_views,
)
from mas_cc.musr_team_allocation_generator.io_utils import sha256_object, write_json_atomic
from mas_cc.musr_team_allocation_generator.latent_problem import (
    LATENT_VALUE_PRIOR,
    LATENT_VALUE_SUPPORT,
    problem_from_latent_values,
)

from .config import SymbolicAmbiguityConfig


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def summarize_metrics(rows: Sequence[PrivateViewMetrics]) -> dict[str, float]:
    predictability = [row.max_predictability for row in rows]
    entropy = [row.normalized_entropy for row in rows]
    return {
        "mean_M": statistics.fmean(predictability),
        "median_M": statistics.median(predictability),
        "max_M": max(predictability),
        "p95_M": _percentile(predictability, 0.95),
        "mean_Hbar": statistics.fmean(entropy),
        "min_Hbar": min(entropy),
    }


def _feasible(
    problem: Any,
    index: TeamAllocationCompletionIndex,
    config: SymbolicAmbiguityConfig,
    *,
    breadth: int,
    threshold: float,
    seed: int,
    attempts: int = 2,
) -> bool:
    try:
        choose_private_views(
            problem,
            index,
            breadth=breadth,
            population_size=config.population_size,
            max_predictability=threshold,
            min_normalized_entropy=config.min_normalized_entropy,
            seed=seed,
            minimum_holders=config.minimum_holders,
            max_attempts=attempts,
        )
        return True
    except ValueError:
        return False


class _ParquetSink:
    def __init__(self, path: Path) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        self.pa = pa
        self.pq = pq
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.unlink()
        self.writer = None

    def write(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        table = self.pa.Table.from_pylist(rows)
        if self.writer is None:
            self.writer = self.pq.ParquetWriter(
                self.path, table.schema, compression="zstd"
            )
        self.writer.write_table(table)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


def _select_rule(
    candidate_rows: Sequence[Mapping[str, Any]],
    config: SymbolicAmbiguityConfig,
) -> dict[str, Any]:
    per_gold = config.final_tasks // 3
    for criterion, threshold in (
        ("preferred", config.preferred_max_predictability),
        ("fallback", config.fallback_max_predictability),
    ):
        for breadth in sorted(config.private_breadth_candidates, reverse=True):
            for margin in sorted(config.margin_candidates, reverse=True):
                key = f"margin_{margin}_k_{breadth}_{criterion}_assignment_feasible"
                unique: dict[tuple[int, ...], Mapping[str, Any]] = {}
                for row in candidate_rows:
                    if row.get(key):
                        unique.setdefault(tuple(json.loads(str(row["latent_values"]))), row)
                counts = Counter(int(row["gold_index"]) for row in unique.values())
                if all(counts[index] >= per_gold for index in range(3)):
                    return {
                        "criterion": criterion,
                        "max_predictability": threshold,
                        "min_normalized_entropy": config.min_normalized_entropy,
                        "private_breadth": breadth,
                        "min_score_margin": margin,
                        "balance_rule": f"exactly {per_gold} tasks per ALLOCATION ID",
                        "selection_rule": (
                            "preferred before fallback; largest feasible breadth; "
                            "then largest feasible score margin"
                        ),
                        "available_by_gold": {
                            f"ALLOCATION_{index}": counts[index] for index in range(3)
                        },
                    }
    raise RuntimeError("symbolic scan cannot supply a balanced final task set")


def _select_worlds(
    rows: Sequence[Mapping[str, Any]],
    rule: Mapping[str, Any],
    config: SymbolicAmbiguityConfig,
    indexes: Mapping[int, TeamAllocationCompletionIndex],
) -> list[dict[str, Any]]:
    margin = int(rule["min_score_margin"])
    breadth = int(rule["private_breadth"])
    criterion = str(rule["criterion"])
    threshold = float(rule["max_predictability"])
    key = f"margin_{margin}_k_{breadth}_{criterion}_assignment_feasible"
    per_gold = config.final_tasks // 3
    selected: list[dict[str, Any]] = []
    seen: set[tuple[int, ...]] = set()
    for gold in range(3):
        candidates = [row for row in rows if bool(row.get(key)) and int(row["gold_index"]) == gold]
        candidates.sort(key=lambda row: sha256_object([config.seed, row["candidate_id"], row["latent_values"]]))
        for row in candidates:
            vector = tuple(int(value) for value in json.loads(str(row["latent_values"])))
            if vector in seen:
                continue
            problem = problem_from_latent_values(vector)
            try:
                views = choose_private_views(
                    problem,
                    indexes[margin],
                    breadth=breadth,
                    population_size=config.population_size,
                    max_predictability=threshold,
                    min_normalized_entropy=config.min_normalized_entropy,
                    seed=int(Seed(config.seed).derive(f"assignment:{row['candidate_id']}")),
                    minimum_holders=config.minimum_holders,
                    max_attempts=256,
                )
            except ValueError:
                continue
            selected.append(
                {
                    "task_id": f"task_{len(selected) + 1:03d}",
                    "candidate_id": int(row["candidate_id"]),
                    "latent_values": list(vector),
                    "candidate_scores": list(problem.candidate_scores),
                    "gold_index": problem.gold_index,
                    "gold_answer": f"ALLOCATION_{problem.gold_index}",
                    "score_margin": problem.margin_to_second_best,
                    "private_views": [view.to_dict() for view in views],
                }
            )
            seen.add(vector)
            if sum(item["gold_index"] == gold for item in selected) == per_gold:
                break
        if sum(item["gold_index"] == gold for item in selected) != per_gold:
            raise RuntimeError(f"could not materialize {per_gold} unique gold-{gold} worlds")
    return selected


def run_symbolic_scan(
    config: SymbolicAmbiguityConfig,
    output: Path,
) -> dict[str, Any]:
    """Scan, archive every sampled world, and freeze a balanced selection."""

    output.mkdir(parents=True, exist_ok=True)
    indexes = {
        margin: TeamAllocationCompletionIndex(min_score_margin=margin)
        for margin in config.margin_candidates
    }
    rng = Seed(config.seed).derive("symbolic-candidate-worlds").create_random()
    weights = tuple(weight for _, weight in LATENT_VALUE_PRIOR)
    candidate_rows: list[dict[str, Any]] = []
    subset_sink = _ParquetSink(output / "subset_metrics.parquet")
    subset_batch: list[dict[str, Any]] = []
    margin_one = indexes[min(config.margin_candidates)]
    try:
        for candidate_id in range(1, config.candidate_worlds + 1):
            vector = tuple(rng.choices(LATENT_VALUE_SUPPORT, weights=weights, k=9))
            problem = problem_from_latent_values(vector)
            scores = problem.candidate_scores
            unique = scores.count(max(scores)) == 1
            row: dict[str, Any] = {
                "candidate_id": candidate_id,
                "latent_values": json.dumps(vector, separators=(",", ":")),
                "candidate_scores": json.dumps(scores, separators=(",", ":")),
                "gold_index": problem.gold_index if unique else -1,
                "gold_answer": f"ALLOCATION_{problem.gold_index}" if unique else "TIE",
                "gold_score": max(scores),
                "second_best_score": sorted(scores, reverse=True)[1],
                "score_margin": problem.margin_to_second_best,
                "unique_optimum": unique,
            }
            for breadth in config.private_breadth_candidates:
                retained = margin_one.scan(problem, breadth)
                for metrics in retained:
                    subset_batch.append(
                        {
                            "candidate_id": candidate_id,
                            "completion_min_score_margin": min(config.margin_candidates),
                            "k": breadth,
                            "visible_indices": json.dumps(metrics.visible_indices),
                            "visible_values": json.dumps(metrics.visible_values),
                            "p_allocation_0": metrics.probabilities[0],
                            "p_allocation_1": metrics.probabilities[1],
                            "p_allocation_2": metrics.probabilities[2],
                            "M": metrics.max_predictability,
                            "Hbar": metrics.normalized_entropy,
                            "valid_completion_count": metrics.valid_completion_count,
                            "invalid_completion_count": metrics.invalid_completion_count,
                        }
                    )
                if len(subset_batch) >= 25_000:
                    subset_sink.write(subset_batch)
                    subset_batch = []
            for margin, index in indexes.items():
                for breadth in config.private_breadth_candidates:
                    metrics = index.scan(problem, breadth)
                    summary = summarize_metrics(metrics)
                    prefix = f"margin_{margin}_k_{breadth}"
                    row.update({f"{prefix}_{key}": value for key, value in summary.items()})
                    for criterion, threshold in (
                        ("preferred", config.preferred_max_predictability),
                        ("fallback", config.fallback_max_predictability),
                    ):
                        eligible = [
                            item
                            for item in metrics
                            if item.max_predictability <= threshold
                            and item.normalized_entropy >= config.min_normalized_entropy
                        ]
                        row[f"{prefix}_{criterion}_eligible_subsets"] = len(eligible)
                        row[f"{prefix}_{criterion}_assignment_feasible"] = bool(
                            unique
                            and problem.margin_to_second_best >= margin
                            and _feasible(
                                problem,
                                index,
                                config,
                                breadth=breadth,
                                threshold=threshold,
                                seed=int(Seed(config.seed).derive(f"scan:{candidate_id}:{margin}:{breadth}:{criterion}")),
                            )
                        )
            candidate_rows.append(row)
        subset_sink.write(subset_batch)
    finally:
        subset_sink.close()

    rule = _select_rule(candidate_rows, config)
    selected = _select_worlds(candidate_rows, rule, config, indexes)
    _write_csv(output / "candidate_worlds.csv", candidate_rows)

    acceptance_rows: list[dict[str, Any]] = []
    for margin in config.margin_candidates:
        for breadth in config.private_breadth_candidates:
            for criterion, threshold in (
                ("preferred", config.preferred_max_predictability),
                ("fallback", config.fallback_max_predictability),
            ):
                key = f"margin_{margin}_k_{breadth}_{criterion}_assignment_feasible"
                accepted = [row for row in candidate_rows if row[key]]
                eligible = [
                    row
                    for row in candidate_rows
                    if row["unique_optimum"] and int(row["score_margin"]) >= margin
                ]
                acceptance_rows.append(
                    {
                        "min_score_margin": margin,
                        "k": breadth,
                        "criterion": criterion,
                        "max_predictability_threshold": threshold,
                        "entropy_threshold": config.min_normalized_entropy,
                        "candidate_worlds": config.candidate_worlds,
                        "structurally_valid_worlds": len(eligible),
                        "accepted_worlds": len(accepted),
                        "acceptance_rate_all": len(accepted) / config.candidate_worlds,
                        "acceptance_rate_structurally_valid": len(accepted) / len(eligible) if eligible else 0,
                        "gold_0": sum(int(row["gold_index"]) == 0 for row in accepted),
                        "gold_1": sum(int(row["gold_index"]) == 1 for row in accepted),
                        "gold_2": sum(int(row["gold_index"]) == 2 for row in accepted),
                    }
                )
    _write_csv(output / "acceptance_summary.csv", acceptance_rows)

    ambiguity_rows = []
    for breadth in config.private_breadth_candidates:
        prefix = f"margin_1_k_{breadth}"
        valid = [row for row in candidate_rows if row["unique_optimum"]]
        ambiguity_rows.append(
            {
                "k": breadth,
                "candidate_worlds": len(valid),
                "pass_M_le_0_45": sum(row[f"{prefix}_preferred_assignment_feasible"] for row in valid),
                "pass_rate": sum(row[f"{prefix}_preferred_assignment_feasible"] for row in valid) / len(valid),
                "median_worst_case_M": statistics.median(float(row[f"{prefix}_max_M"]) for row in valid),
                "median_mean_Hbar": statistics.median(float(row[f"{prefix}_mean_Hbar"]) for row in valid),
            }
        )
    _write_csv(output / "ambiguity_by_k.csv", ambiguity_rows)
    _write_csv(output / "margin_ambiguity_tradeoff.csv", acceptance_rows)
    frozen = {
        "schema_version": 1,
        "latent_support": list(LATENT_VALUE_SUPPORT),
        "latent_prior": {str(key): value for key, value in LATENT_VALUE_PRIOR},
        "completion_weighting": (
            "independent generator prior conditioned on visible values, unique optimum, "
            "and the selected minimum score margin; tied/sub-margin completions excluded"
        ),
        "candidate_worlds_scanned": config.candidate_worlds,
        "construction_rule": rule,
        "selected_worlds": selected,
    }
    frozen["fingerprint_sha256"] = sha256_object(frozen)
    write_json_atomic(output / "frozen_selection.json", frozen)
    return frozen


__all__ = ["run_symbolic_scan", "summarize_metrics"]
