"""Provider-free focal-conditioned embedded jump chain.

The kernel keeps the Irisarri multi-opinion geometry explicit: one uniformly
sampled focal agent leaves opinion A for one B != A, and the occupation vector
changes by ``-e_A + e_B``.  The configured forward/reverse rates are attached
to each unordered option pair.  This is a discrete embedded chain; event index,
not Gillespie physical time, is the v1 clock.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

from mas_cc.control import InteractionControlSignal

from .controller import ADVOCATE_TARGET
from .state import ImitationRules


@dataclass(frozen=True, slots=True)
class ClassicalJump:
    source: str
    destination: str
    reaction_id: str
    transition_weight: float
    total_weight: float
    base_weight: float
    control_weight: float
    candidate_channels: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class ClassicalSocialContext:
    """Sampled q-peer context reserved for a future explicit q-voter kernel.

    The current linear Irisarri kernel intentionally ignores this object.  It
    is nevertheless a typed boundary now, so extending the transition law does
    not require redesigning the scheduler or persisted event schema.
    """

    peer_ids: tuple[str, ...]
    peer_opinions: tuple[str, ...]
    replaced_peer_slot: int | None


def _interaction_factor(
    destination_count: int, population_size: int, rules: ImitationRules
) -> float:
    if rules.interaction_factor == "constant":
        return 1.0
    return (destination_count + rules.interaction_offset) / population_size


def _channel(
    source: str,
    destination: str,
    options: Sequence[str],
    source_count: int,
    destination_count: int,
    population_size: int,
    rules: ImitationRules,
) -> tuple[str, float, float]:
    source_index = options.index(source)
    destination_index = options.index(destination)
    low, high = sorted((source_index, destination_index))
    forward = source_index == low
    rate_constant = rules.forward_rate if forward else rules.reverse_rate
    direction = "forward" if forward else "reverse"
    factor = _interaction_factor(destination_count, population_size, rules)
    reaction_id = f"pair-{low}-{high}:{direction}"
    return reaction_id, rate_constant * source_count * factor, factor


def sample_jump(
    votes: Sequence[str],
    focal_index: int,
    options: Sequence[str],
    rules: ImitationRules,
    rng: random.Random,
    signal: InteractionControlSignal | None = None,
    social_context: ClassicalSocialContext | None = None,
) -> ClassicalJump:
    """Sample B for one selected focal A; v1 deliberately ignores q-context."""

    _ = social_context

    source = votes[focal_index]
    counts = Counter(votes)
    candidates: list[dict[str, Any]] = []
    for destination in options:
        if destination == source:
            continue
        reaction_id, base, factor = _channel(
            source,
            destination,
            options,
            counts[source],
            counts.get(destination, 0),
            len(votes),
            rules,
        )
        control = 0.0
        if (
            signal is not None
            and signal.action == ADVOCATE_TARGET
            and signal.target == destination
        ):
            control = rules.control_strength * counts[source]
        candidates.append(
            {
                "reaction_id": reaction_id,
                "source": source,
                "destination": destination,
                "base_weight": base,
                "control_weight": control,
                "transition_weight": base + control,
                "interaction_factor": factor,
            }
        )
    total = sum(float(item["transition_weight"]) for item in candidates)
    if total <= 0:
        raise ValueError("classical transition weights have zero total support")
    draw = rng.random() * total
    cumulative = 0.0
    chosen = candidates[-1]
    for candidate in candidates:
        cumulative += float(candidate["transition_weight"])
        if draw < cumulative:
            chosen = candidate
            break
    reaction_id = str(chosen["reaction_id"])
    if float(chosen["control_weight"]) > 0:
        reaction_id = f"control+{reaction_id}"
    return ClassicalJump(
        source=source,
        destination=str(chosen["destination"]),
        reaction_id=reaction_id,
        transition_weight=float(chosen["transition_weight"]),
        total_weight=total,
        base_weight=float(chosen["base_weight"]),
        control_weight=float(chosen["control_weight"]),
        candidate_channels=tuple(dict(item) for item in candidates),
    )
