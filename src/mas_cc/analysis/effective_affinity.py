"""Effective directional affinity and kinetic compliance from microscopic slots.

This is the reusable form of the established Study-05
``estimate_effective_affinity.py`` analysis.  It preserves that analysis's
binary target/non-target coarse graining, raw point estimates, Jeffreys
stabilization inside sparse bootstrap draws, and whole-episode resampling.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def _jeffreys(successes: int, exposures: int) -> float:
    return (successes + 0.5) / (exposures + 1.0)


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n_plus = sum(int(row["n_plus"]) for row in rows)
    k_plus = sum(int(row["k_plus"]) for row in rows)
    n_minus = sum(int(row["n_minus"]) for row in rows)
    k_minus = sum(int(row["k_minus"]) for row in rows)
    p_plus = k_plus / n_plus if n_plus else math.nan
    p_minus = k_minus / n_minus if n_minus else math.nan
    affinity = (
        math.nan
        if not math.isfinite(p_plus) or not math.isfinite(p_minus)
        else math.inf if p_plus <= 0 or p_minus <= 0 else math.log(p_plus / p_minus)
    )
    p_plus_j = _jeffreys(k_plus, n_plus)
    p_minus_j = _jeffreys(k_minus, n_minus)
    return {
        "n_plus": n_plus,
        "k_plus": k_plus,
        "p_plus": p_plus,
        "n_minus": n_minus,
        "k_minus": k_minus,
        "p_minus": p_minus,
        "effective_affinity": affinity,
        "effective_affinity_jeffreys": math.log(p_plus_j / p_minus_j),
        "kinetic_compliance": p_plus + p_minus,
        "kinetic_compliance_jeffreys": p_plus_j + p_minus_j,
    }


def _episode_counts(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        action = str(row.get("round_controller_action", row.get("controller_action", "")))
        if action not in {"ADVOCATE_Z", "ADVOCATE_TARGET"} or not bool(row.get("controlled_slot")):
            continue
        target = row.get(
            "analysis_target", row.get("controller_target", row.get("round_controller_target"))
        )
        before = row.get("focal_opinion_before")
        after = row.get("focal_opinion_after")
        if target is None or before is None or after is None:
            continue
        grouped[(str(row.get("cell_id", "run")), str(row.get("episode_id", "episode")))].append(row)
    result = []
    for (cell_id, episode_id), group in grouped.items():
        n_plus = k_plus = n_minus = k_minus = 0
        for row in group:
            target = str(
                row.get(
                    "analysis_target",
                    row.get("controller_target", row.get("round_controller_target")),
                )
            )
            before_target = str(row["focal_opinion_before"]) == target
            after_target = str(row["focal_opinion_after"]) == target
            if before_target:
                n_minus += 1
                k_minus += int(not after_target)
            else:
                n_plus += 1
                k_plus += int(after_target)
        result.append(
            {
                "cell_id": cell_id,
                "episode_id": episode_id,
                "n_plus": n_plus,
                "k_plus": k_plus,
                "n_minus": n_minus,
                "k_minus": k_minus,
            }
        )
    return result


def effective_affinity_analysis(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 1,
) -> list[dict[str, Any]]:
    """Return per-cell affinity/compliance summaries."""

    episodes = _episode_counts(rows)
    if not episodes:
        return []
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in episodes:
        groups[str(row["cell_id"])].append(row)
    alpha = (1.0 - confidence) / 2.0
    output: list[dict[str, Any]] = []
    for group_index, (cell_id, group) in enumerate(sorted(groups.items())):
        rng = random.Random(seed + group_index)
        strata: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in group:
            strata["cell"].append(row)
        draws: dict[str, list[float]] = {
            "effective_affinity": [], "kinetic_compliance": []
        }
        for _ in range(bootstrap_resamples):
            sampled = [
                stratum[rng.randrange(len(stratum))]
                for stratum in strata.values()
                for _ in range(len(stratum))
            ]
            summary = _summary(sampled)
            draws["effective_affinity"].append(summary["effective_affinity_jeffreys"])
            draws["kinetic_compliance"].append(summary["kinetic_compliance_jeffreys"])
        summary = _summary(group)
        for metric in ("effective_affinity", "kinetic_compliance"):
            values = draws[metric]
            summary[f"{metric}_ci_low"] = (
                math.nan if not values else float(np.quantile(values, alpha))
            )
            summary[f"{metric}_ci_high"] = (
                math.nan if not values else float(np.quantile(values, 1.0 - alpha))
            )
        output.append(
            {
                "cell_id": cell_id,
                "n_episodes": len(group),
                "n_observations": summary["n_plus"] + summary["n_minus"],
                **summary,
            }
        )
    return output


__all__ = ["effective_affinity_analysis"]
