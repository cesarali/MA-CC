"""Strict-unanimity K0/K1 tests for the round-feedback classical kernel."""

from __future__ import annotations

import math
import random

import pytest

from mas_cc.games.hidden_bench.imitation_round_feedback.classical import (
    analytical_mesoscopic_transition_probability,
    analytical_switch_probability,
    classical_transition,
)


def _transition(
    peer_opinions,
    *,
    focal="A",
    controlled=False,
    target="B",
    replaced_slot=None,
):
    population = [focal, *peer_opinions, "B", "C"]
    return classical_transition(
        population_state=population,
        focal_agent_id="focal",
        focal_opinion=focal,
        peer_agent_ids=[f"peer-{index}" for index in range(len(peer_opinions))],
        peer_opinions=peer_opinions,
        controlled_slot=controlled,
        controller_target=target if controlled else None,
        replaced_peer_slot=replaced_slot,
        rng=random.Random(1),
    )


def test_k0_switches_only_under_unanimous_ordinary_peers():
    switch = _transition(["B", "B"])
    disagree = _transition(["B", "C"])
    same_as_focal = _transition(["A", "A"])

    assert switch.destination == "B"
    assert switch.effective_input_opinions == ("B", "B")
    assert switch.unanimous_opinion == "B"
    assert switch.changed is True
    assert disagree.destination == "A"
    assert disagree.unanimous_opinion is None
    assert disagree.changed is False
    assert same_as_focal.destination == "A"
    assert same_as_focal.unanimous_opinion == "A"
    assert same_as_focal.changed is False


def test_k1_switches_non_target_to_target_iff_remaining_peers_support_target():
    # Slot 1 is displaced: B, C, B becomes B, B(controller), B.
    switch = _transition(["B", "C", "B"], controlled=True, replaced_slot=1)
    # One non-target ordinary input survives the replacement.
    disagree = _transition(["B", "C", "C"], controlled=True, replaced_slot=1)
    target_focal = _transition(
        ["B", "C", "B"], focal="B", controlled=True, replaced_slot=1
    )

    assert switch.effective_input_opinions == ("B", "B", "B")
    assert switch.destination == "B"
    assert switch.changed is True
    assert disagree.effective_input_opinions == ("B", "B", "C")
    assert disagree.destination == "A"
    assert disagree.changed is False
    assert target_focal.destination == "B"
    assert target_focal.changed is False


def test_q_one_controlled_always_converts_non_target_and_ordinary_copies_peer():
    controlled = _transition(["C"], controlled=True, replaced_slot=0)
    ordinary = _transition(["C"])
    already_target = _transition(
        ["C"], focal="B", controlled=True, replaced_slot=0
    )

    assert controlled.effective_input_opinions == ("B",)
    assert controlled.destination == "B"
    assert ordinary.destination == "C"
    assert already_target.destination == "B"
    assert analytical_switch_probability(
        population_state=["A", "B", "B", "C"],
        focal_opinion="A",
        destination="B",
        social_group_size=1,
        controlled_slot=True,
        controller_target="B",
    ) == 1.0


def test_controller_can_introduce_a_target_with_zero_population_occupancy():
    transition = classical_transition(
        population_state=["A", "A", "C"],
        focal_agent_id="0",
        focal_opinion="A",
        peer_agent_ids=["1"],
        peer_opinions=["A"],
        controlled_slot=True,
        controller_target="B",
        replaced_peer_slot=0,
        rng=random.Random(1),
    )

    assert transition.effective_input_opinions == ("B",)
    assert transition.destination == "B"
    assert analytical_switch_probability(
        population_state=["A", "A", "C"],
        focal_opinion="A",
        destination="B",
        social_group_size=1,
        controlled_slot=True,
        controller_target="B",
    ) == 1.0


def test_analytical_k0_and_k1_probabilities_match_the_brief():
    population = ["A", "A", "B", "B", "B", "B", "C", "C"]
    ordinary = analytical_mesoscopic_transition_probability(
        population_state=population,
        source="A",
        destination="B",
        social_group_size=2,
        controlled_slot=False,
    )
    controlled = analytical_mesoscopic_transition_probability(
        population_state=population,
        source="A",
        destination="B",
        social_group_size=3,
        controlled_slot=True,
        controller_target="B",
    )
    assert ordinary == pytest.approx((2 / 8) * math.comb(4, 2) / math.comb(7, 2))
    assert controlled == pytest.approx((2 / 8) * math.comb(4, 2) / math.comb(7, 2))
    assert analytical_mesoscopic_transition_probability(
        population_state=population,
        source="A",
        destination="C",
        social_group_size=3,
        controlled_slot=True,
        controller_target="B",
    ) == 0.0


@pytest.mark.parametrize("controlled", [False, True])
def test_empirical_mesoscopic_frequencies_match_analytical_kernels(controlled):
    population = ["A", "A", "B", "B", "B", "B", "C", "C"]
    q = 3 if controlled else 2
    expected = analytical_mesoscopic_transition_probability(
        population_state=population,
        source="A",
        destination="B",
        social_group_size=q,
        controlled_slot=controlled,
        controller_target="B" if controlled else None,
    )
    rng = random.Random(20260813 + int(controlled))
    transitions = 0
    trials = 60_000
    for _ in range(trials):
        focal_index, *peer_indices = rng.sample(range(len(population)), q + 1)
        focal = population[focal_index]
        peer_opinions = [population[index] for index in peer_indices]
        replaced = rng.randrange(q) if controlled else None
        transition = classical_transition(
            population_state=population,
            focal_agent_id=str(focal_index),
            focal_opinion=focal,
            peer_agent_ids=[str(index) for index in peer_indices],
            peer_opinions=peer_opinions,
            controlled_slot=controlled,
            controller_target="B" if controlled else None,
            replaced_peer_slot=replaced,
            rng=rng,
        )
        transitions += int(focal == "A" and transition.destination == "B")
    assert transitions / trials == pytest.approx(expected, abs=0.005)
