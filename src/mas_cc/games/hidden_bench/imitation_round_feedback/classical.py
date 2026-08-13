"""Provider-free strict-unanimity q-voter kernel for round feedback."""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True, slots=True)
class RoundClassicalTransition:
    """One deterministic response to an already-sampled social context."""

    source: str
    destination: str
    controlled_slot: bool
    effective_input_opinions: tuple[str, ...]
    unanimous_opinion: str | None
    changed: bool


def analytical_switch_probability(
    *,
    population_state: Sequence[str],
    focal_opinion: str,
    destination: str,
    social_group_size: int,
    controlled_slot: bool,
    controller_target: str | None = None,
) -> float:
    """Return the K0/K1 switch probability conditional on the focal opinion.

    This is the microscopic probability after the focal has been selected.
    Multiplying by the source occupation share gives the corresponding
    mesoscopic transition probability in the kernel-choice brief.
    """

    population = tuple(str(opinion) for opinion in population_state)
    n_agents = len(population)
    q = social_group_size
    if n_agents < 2:
        raise ValueError("population_state must contain at least two opinions")
    if not 1 <= q <= n_agents - 1:
        raise ValueError("social_group_size must be between 1 and N - 1")
    if focal_opinion not in population:
        raise ValueError("focal_opinion must occur in population_state")
    if destination == focal_opinion:
        return 0.0

    counts = Counter(population)
    if controlled_slot:
        if controller_target is None:
            raise ValueError("a controlled slot requires a controller target")
        if destination != controller_target:
            return 0.0
        # A switching focal is non-target, so every target holder is available
        # among the N-1 potential ordinary peers.
        return math.comb(counts[controller_target], q - 1) / math.comb(
            n_agents - 1, q - 1
        )
    return math.comb(counts[destination], q) / math.comb(n_agents - 1, q)


def analytical_mesoscopic_transition_probability(
    *,
    population_state: Sequence[str],
    source: str,
    destination: str,
    social_group_size: int,
    controlled_slot: bool,
    controller_target: str | None = None,
) -> float:
    """Return K0/K1 including the probability of selecting a source focal."""

    population = tuple(str(opinion) for opinion in population_state)
    if not population:
        raise ValueError("population_state must not be empty")
    source_count = Counter(population)[source]
    if source_count == 0:
        return 0.0
    conditional = analytical_switch_probability(
        population_state=population,
        focal_opinion=source,
        destination=destination,
        social_group_size=social_group_size,
        controlled_slot=controlled_slot,
        controller_target=controller_target,
    )
    return source_count / len(population) * conditional


def classical_transition(
    *,
    population_state: Sequence[str],
    focal_agent_id: str,
    focal_opinion: str,
    peer_agent_ids: Sequence[str],
    peer_opinions: Sequence[str],
    controlled_slot: bool,
    controller_target: str | None,
    replaced_peer_slot: int | None,
    rng: random.Random,
) -> RoundClassicalTransition:
    """Apply the same strict-unanimity response in K0 and K1.

    The scheduler has already sampled ``q`` distinct ordinary peers. An
    ordinary update uses all q opinions. A controlled update replaces exactly
    one sampled peer by the controller target, leaving q-1 ordinary opinions.
    The focal changes only when every effective input is unanimous on a value
    different from its current opinion.
    """

    _ = focal_agent_id, rng  # Sampling belongs to the runtime, not this rule.
    population = tuple(str(opinion) for opinion in population_state)
    peers = tuple(str(opinion) for opinion in peer_opinions)
    if len(peer_agent_ids) != len(peers):
        raise ValueError("peer_agent_ids and peer_opinions must have equal length")
    if not peers:
        raise ValueError("the unanimity kernel requires at least one sampled peer")
    if focal_opinion not in population:
        raise ValueError("focal_opinion must occur in population_state")

    effective = list(peers)
    if controlled_slot:
        if controller_target is None:
            raise ValueError("a controlled slot requires a controller target")
        if replaced_peer_slot is None or not 0 <= replaced_peer_slot < len(peers):
            raise ValueError("a controlled slot requires a valid replaced_peer_slot")
        effective[replaced_peer_slot] = str(controller_target)
    elif replaced_peer_slot is not None:
        raise ValueError("an ordinary update cannot replace a peer slot")

    unanimous = effective[0] if len(set(effective)) == 1 else None
    destination = (
        unanimous
        if unanimous is not None and unanimous != focal_opinion
        else str(focal_opinion)
    )
    return RoundClassicalTransition(
        source=str(focal_opinion),
        destination=destination,
        controlled_slot=controlled_slot,
        effective_input_opinions=tuple(effective),
        unanimous_opinion=unanimous,
        changed=destination != focal_opinion,
    )


__all__ = [
    "RoundClassicalTransition",
    "analytical_mesoscopic_transition_probability",
    "analytical_switch_probability",
    "classical_transition",
]
