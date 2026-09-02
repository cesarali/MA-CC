"""Exact private-view ambiguity for MuSR Team Allocation worlds."""

from __future__ import annotations

import itertools
import math
import random
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .latent_problem import (
    LATENT_VALUE_PRIOR,
    LATENT_VALUE_SUPPORT,
    latent_values,
    problem_from_latent_values,
)
from .schemas import LatentProblem


@dataclass(frozen=True, slots=True)
class PrivateViewMetrics:
    visible_indices: tuple[int, ...]
    visible_values: tuple[int, ...]
    probabilities: tuple[float, float, float]
    max_predictability: float
    normalized_entropy: float
    valid_completion_count: int
    invalid_completion_count: int
    valid_probability_mass: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "visible_indices": list(self.visible_indices),
            "visible_values": list(self.visible_values),
            "p_allocation_0": self.probabilities[0],
            "p_allocation_1": self.probabilities[1],
            "p_allocation_2": self.probabilities[2],
            "max_predictability": self.max_predictability,
            "normalized_entropy": self.normalized_entropy,
            "valid_completion_count": self.valid_completion_count,
            "invalid_completion_count": self.invalid_completion_count,
            "valid_probability_mass": self.valid_probability_mass,
        }


def _normalized_entropy(probabilities: Sequence[float]) -> float:
    entropy = -sum(value * math.log(value) for value in probabilities if value > 0)
    result = entropy / math.log(len(probabilities))
    return min(1.0, max(0.0, result))


def exact_private_view_metrics(
    *,
    latent_count: int,
    visible_indices: Sequence[int],
    visible_values: Sequence[int],
    support: Sequence[int],
    score_function: Callable[[tuple[int, ...]], Sequence[float]],
    priors: Sequence[Mapping[int, float]] | Mapping[int, float] | None = None,
    min_score_margin: float = 1,
) -> PrivateViewMetrics:
    """Enumerate completions, excluding tied/sub-margin worlds before weighting.

    ``priors`` may be one common discrete prior or one prior per latent index.
    The posterior is the supplied independent prior conditioned on the visible
    values and on a unique optimum satisfying ``min_score_margin``.
    """

    indices = tuple(int(value) for value in visible_indices)
    values = tuple(int(value) for value in visible_values)
    support = tuple(int(value) for value in support)
    if len(indices) != len(values) or len(set(indices)) != len(indices):
        raise ValueError("visible indices and values must be aligned and unique")
    if any(index < 0 or index >= latent_count for index in indices):
        raise ValueError("visible index is outside the latent vector")
    if not support or any(value not in support for value in values):
        raise ValueError("visible values must belong to the latent support")
    if priors is None:
        common = {value: 1.0 / len(support) for value in support}
        prior_rows = tuple(common for _ in range(latent_count))
    elif isinstance(priors, Mapping):
        prior_rows = tuple(priors for _ in range(latent_count))
    else:
        prior_rows = tuple(priors)
        if len(prior_rows) != latent_count:
            raise ValueError("one prior mapping is required per latent index")
    for row in prior_rows:
        if set(row) != set(support) or any(float(row[value]) < 0 for value in support):
            raise ValueError("each prior must assign nonnegative weight to the support")
        if sum(float(row[value]) for value in support) <= 0:
            raise ValueError("each prior must have positive total mass")

    fixed = dict(zip(indices, values, strict=True))
    unknown = tuple(index for index in range(latent_count) if index not in fixed)
    winner_mass = [0.0, 0.0, 0.0]
    valid_count = 0
    invalid_count = 0
    valid_mass = 0.0
    for completion in itertools.product(support, repeat=len(unknown)):
        vector = [0] * latent_count
        vector_weights = 1.0
        cursor = 0
        for index in range(latent_count):
            value = fixed[index] if index in fixed else completion[cursor]
            cursor += int(index not in fixed)
            vector[index] = value
            if index not in fixed:
                vector_weights *= float(prior_rows[index][value])
        scores = tuple(float(value) for value in score_function(tuple(vector)))
        if len(scores) != 3:
            raise ValueError("Team Allocation score_function must return three scores")
        ordered = sorted(scores, reverse=True)
        winners = [index for index, value in enumerate(scores) if value == ordered[0]]
        if len(winners) != 1 or ordered[0] - ordered[1] < min_score_margin:
            invalid_count += 1
            continue
        valid_count += 1
        valid_mass += vector_weights
        winner_mass[winners[0]] += vector_weights
    if valid_mass <= 0:
        raise ValueError("partial view has no valid completions")
    probabilities = tuple(value / valid_mass for value in winner_mass)
    return PrivateViewMetrics(
        visible_indices=indices,
        visible_values=values,
        probabilities=probabilities,  # type: ignore[arg-type]
        max_predictability=max(probabilities),
        normalized_entropy=_normalized_entropy(probabilities),
        valid_completion_count=valid_count,
        invalid_completion_count=invalid_count,
        valid_probability_mass=valid_mass,
    )


class TeamAllocationCompletionIndex:
    """Cached exact posterior tables under the generator's latent prior."""

    def __init__(
        self,
        *,
        min_score_margin: int = 1,
        prior: Mapping[int, float] | None = None,
    ) -> None:
        if min_score_margin < 1:
            raise ValueError("min_score_margin must be at least one")
        self.min_score_margin = min_score_margin
        self.support = LATENT_VALUE_SUPPORT
        self.prior = dict(prior or LATENT_VALUE_PRIOR)
        self._worlds: list[tuple[tuple[int, ...], int, float]] = []
        self._invalid_worlds = 0
        for vector in itertools.product(self.support, repeat=9):
            problem = problem_from_latent_values(vector)
            scores = problem.candidate_scores
            top = max(scores)
            winners = [index for index, score in enumerate(scores) if score == top]
            if len(winners) != 1 or problem.margin_to_second_best < min_score_margin:
                self._invalid_worlds += 1
                continue
            weight = math.prod(float(self.prior[value]) for value in vector)
            self._worlds.append((vector, winners[0], weight))
        self._lookups: dict[int, dict[tuple[tuple[int, ...], tuple[int, ...]], PrivateViewMetrics]] = {}

    @property
    def valid_world_count(self) -> int:
        return len(self._worlds)

    @property
    def invalid_world_count(self) -> int:
        return self._invalid_worlds

    def _build(self, k: int) -> None:
        if k in self._lookups:
            return
        if not 0 <= k <= 9:
            raise ValueError("private breadth must be between zero and nine")
        tables: dict[
            tuple[int, ...],
            dict[tuple[int, ...], tuple[list[float], int, float]],
        ] = {}
        for indices in itertools.combinations(range(9), k):
            tables[indices] = defaultdict(lambda: ([0.0, 0.0, 0.0], 0, 0.0))
        for vector, winner, weight in self._worlds:
            for indices, table in tables.items():
                key = tuple(vector[index] for index in indices)
                masses, count, total = table[key]
                masses[winner] += weight
                table[key] = (masses, count + 1, total + weight)
        total_completions = len(self.support) ** (9 - k)
        lookup: dict[tuple[tuple[int, ...], tuple[int, ...]], PrivateViewMetrics] = {}
        for indices, table in tables.items():
            for values, (masses, valid_count, total) in table.items():
                probabilities = tuple(value / total for value in masses)
                visible_mass = math.prod(
                    float(self.prior[value]) for value in values
                )
                lookup[(indices, values)] = PrivateViewMetrics(
                    visible_indices=indices,
                    visible_values=values,
                    probabilities=probabilities,  # type: ignore[arg-type]
                    max_predictability=max(probabilities),
                    normalized_entropy=_normalized_entropy(probabilities),
                    valid_completion_count=valid_count,
                    invalid_completion_count=total_completions - valid_count,
                    valid_probability_mass=total / visible_mass,
                )
        self._lookups[k] = lookup

    def metrics(
        self, vector: Sequence[int], visible_indices: Sequence[int]
    ) -> PrivateViewMetrics:
        indices = tuple(sorted(int(index) for index in visible_indices))
        values = tuple(int(vector[index]) for index in indices)
        self._build(len(indices))
        try:
            return self._lookups[len(indices)][(indices, values)]
        except KeyError as exc:
            raise ValueError("partial view has no valid completions") from exc

    def scan(self, problem: LatentProblem, k: int) -> tuple[PrivateViewMetrics, ...]:
        vector = latent_values(problem)
        return tuple(
            self.metrics(vector, indices)
            for indices in itertools.combinations(range(9), k)
        )


def choose_private_views(
    problem: LatentProblem,
    index: TeamAllocationCompletionIndex,
    *,
    breadth: int,
    population_size: int,
    max_predictability: float,
    min_normalized_entropy: float,
    seed: int,
    minimum_holders: int = 2,
    max_attempts: int = 256,
) -> tuple[PrivateViewMetrics, ...]:
    """Choose deterministic threshold-passing views with collective coverage."""

    if population_size * breadth < 9 * minimum_holders:
        raise ValueError("assignment cannot satisfy the requested holder minimum")
    eligible = [
        row
        for row in index.scan(problem, breadth)
        if row.max_predictability <= max_predictability
        and row.normalized_entropy >= min_normalized_entropy
    ]
    if not eligible:
        raise ValueError("world has no ambiguity-qualified private views")
    for attempt in range(max_attempts):
        rng = random.Random(f"{seed}:{attempt}")
        holder_counts = Counter({latent: 0 for latent in range(9)})
        chosen: list[PrivateViewMetrics] = []
        for agent in range(population_size):
            ranked = list(eligible)
            rng.shuffle(ranked)
            ranked.sort(
                key=lambda row: (
                    -sum(
                        max(0, minimum_holders - holder_counts[index])
                        for index in row.visible_indices
                    ),
                    sum(holder_counts[index] for index in row.visible_indices),
                    sum(
                        len(set(row.visible_indices) & set(old.visible_indices))
                        for old in chosen
                    ),
                    row.visible_indices,
                )
            )
            selected = ranked[0]
            chosen.append(selected)
            holder_counts.update(selected.visible_indices)
        if min(holder_counts.values()) >= minimum_holders:
            return tuple(chosen)
    raise ValueError("no collectively complete ambiguity-qualified assignment found")


__all__ = [
    "PrivateViewMetrics",
    "TeamAllocationCompletionIndex",
    "choose_private_views",
    "exact_private_view_metrics",
]
