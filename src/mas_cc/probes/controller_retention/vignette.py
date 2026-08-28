"""Turn one frozen vignette into a real relational-game prompt.

This is the only place the probe touches prompt text, and it writes **none**.
Every string a model sees comes from the production relational-reasoning
modules:

* :func:`~...prompts.build_relational_ballot_prompt` assembles the whole prompt;
* :func:`~...prompts.render_social_sources` renders the visible participants;
* :func:`~...prompts.epistemic_framing` supplies the ``naive`` / ``vigilant``
  block;
* :func:`~...prompts.shuffled_option_letters` permutes the options;
* :func:`~...prompts.render_own_fact` renders the focal agent's own knowledge;
* :func:`~...prompts.agent_label` / :func:`~...prompts.control_label` name the
  participants;
* :func:`~...prompts.parse_relational_ballot` reads the answer back.

The one thing the probe builds itself is the **list of social-source records** -
the small dictionaries the renderer consumes.  The game's own
``build_social_sources`` takes a single ``replaced_peer_slot``, so it cannot
express the probe-only ``two_slots`` arm where two slots are controlled.
Rather than widen the production runtime for a diagnostic, this module emits
records in exactly that function's shape and hands them to the same renderer.
See :func:`social_sources` for the field-by-field correspondence.

A note on identity under ``two_slots``: the real controller is a single
ordinary-looking participant, ``Agent N+1``.  When the probe controls several
slots it numbers them ``N+1, N+2, ...`` - still ordinary-looking participants
that happen to agree, never labelled "controller", "expert" or "system".  Two
slots sharing one label would read as one participant speaking twice, which is
not a thing the population can produce either.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from mas_cc.core import Seed
from mas_cc.games.relational_reasoning.data import RelationalTask
from mas_cc.games.relational_reasoning.imitation_round_feedback.prompts import (
    RelationalBallotPrompt,
    agent_label,
    build_relational_ballot_prompt,
    control_label,
    parse_relational_ballot,
    render_control_reason,
    render_own_fact,
    shuffled_option_letters,
)
from mas_cc.games.relational_reasoning.imitation_round_feedback.runtime import (
    CONTROL_SOURCE_ID,
)

from .design import ARMS_BY_Q, NO_OP, ONE_SLOT, TWO_SLOTS, Vignette

FOCAL_AGENT_INDEX = 1
"""The focal agent is ``Agent 1``; the visible peers are numbered after it.

The number is presentation only - the frozen tasks number their agents from one
and :func:`agent_label` turns an id into that number - but pinning it keeps the
focal agent's own identity block identical across every prompt in a pair.
"""


def option_letters(task: RelationalTask, vignette: Vignette) -> dict[str, str]:
    """This vignette's ``letter -> relation`` map, from the frozen seed.

    Both halves of a matched pair use the same permutation: if the controlled
    prompt called the target "C" and its NO_OP twin called it "A", part of the
    measured difference would be presentation rather than intervention.
    """

    rng = Seed(vignette.option_permutation_seed).create_random()
    return shuffled_option_letters(task.semantic_answers, rng)


def social_sources(
    task: RelationalTask, vignette: Vignette, arm: str
) -> tuple[dict[str, Any], ...]:
    """The visible participants, in slot order, in the game's record shape.

    Each record carries the same six fields
    ``build_social_sources`` produces - ``slot``, ``source_id``, ``source_type``,
    ``label``, ``vote``, ``reason``, ``shared_fact_id``, ``shared_fact_text`` -
    because those are what ``render_social_source`` reads.  ``reason`` is stored
    and never rendered, exactly as in the game.

    Under ``controlled`` the first ``controller_slots`` positions are the
    controller's.  Under ``NO_OP`` those same positions are filled by the
    vignette's pre-drawn replacement peers, so both prompts show ``q``
    participants and differ only in who occupies those slots and what they say.
    """

    if arm not in ARMS_BY_Q[vignette.q]:
        raise ValueError(f"arm {arm!r} is not supported for q={vignette.q}")
    population = task.population_size
    records: list[dict[str, Any]] = []
    controller_slots = vignette.controller_slots(arm)
    for index, vote in enumerate(vignette.ordinary_peer_votes):
        if index < controller_slots:
            records.append(
                {
                    "slot": index,
                    "source_id": (
                        CONTROL_SOURCE_ID
                        if index == 0
                        else f"{CONTROL_SOURCE_ID}-{index + 1}"
                    ),
                    "source_type": "control",
                    "label": control_label(population + index),
                    "vote": vignette.controller_target_semantic,
                    "reason": render_control_reason(
                        vignette.controller_target_semantic
                    ),
                    "shared_fact_id": None,
                    "shared_fact_text": None,
                }
            )
        else:
            records.append(_peer(index, _peer_number(population, index), vote))
    return tuple(records)


def _peer_number(population: int, slot: int) -> int:
    """A stable participant number for one visible peer slot.

    Peers are numbered from ``2`` upward - ``Agent 1`` is the focal agent - and
    never collide with the controller's ``N+1`` and above.
    """

    number = FOCAL_AGENT_INDEX + 1 + slot
    if number > population:
        raise ValueError(
            f"q is too large for a population of {population}: slot {slot} would "
            "need a participant number the task does not have"
        )
    return number


def _peer(slot: int, number: int, vote: str) -> dict[str, Any]:
    """One ordinary peer.  It exposes no fact: peers in this probe are a
    *background*, and letting them cite evidence would add a second uncontrolled
    information channel next to the controller's."""

    return {
        "slot": slot,
        "source_id": f"agent_{number:03d}",
        "source_type": "ordinary",
        "label": agent_label(f"agent_{number:03d}"),
        "vote": vote,
        "reason": None,
        "shared_fact_id": None,
        "shared_fact_text": None,
    }


def known_fact_lines(task: RelationalTask, vignette: Vignette) -> tuple[str, ...]:
    """The focal agent's own facts, rendered by the production renderer."""

    return tuple(
        render_own_fact(fact_id, task.fact_text(fact_id))
        for fact_id in vignette.known_fact_ids
    )


def build_prompt(
    task: RelationalTask, vignette: Vignette, arm: str
) -> RelationalBallotPrompt:
    """One fully bound prompt: the probe's only unit of provider work.

    ``social_context=True`` marks this as a focal update with other participants
    present, which is what a ``q>=1`` local decision is.  It is deliberately not
    the round-zero framing, which would assert to the model that nobody has
    stated a position yet.
    """

    letters = option_letters(task, vignette)
    return build_relational_ballot_prompt(
        identity=agent_label(f"agent_{FOCAL_AGENT_INDEX:03d}"),
        question=task.question,
        option_letters=letters,
        known_facts=known_fact_lines(task, vignette),
        fact_ids=vignette.known_fact_ids,
        current_vote=_letter_for(letters, vignette.initial_vote_semantic),
        social_sources=social_sources(task, vignette, arm),
        vote_visibility="public",
        receiver_epistemic_disposition=vignette.receiver_epistemic_disposition,
        social_context=True,
    )


def _letter_for(letters: Mapping[str, str], relation: str) -> str:
    """The focal agent's standing vote, restated in this call's letter space."""

    for letter, value in letters.items():
        if value == relation:
            return letter
    raise ValueError(f"relation {relation!r} is not among this call's options")


def parse_vote(
    task: RelationalTask, vignette: Vignette, response: str
) -> tuple[str | None, dict[str, Any]]:
    """Read one model response back into a *semantic* relation.

    The letter never leaves this function, exactly as in
    ``RelationalImitationRoundFeedbackGame.parse_action``: everything saved and
    compared downstream is the compass relation the letter stood for here.
    """

    letters = option_letters(task, vignette)
    ballot = parse_relational_ballot(response, tuple(letters), letters)
    vote = None if ballot.vote is None else letters.get(ballot.vote, ballot.vote)
    return vote, {
        "raw_vote": ballot.raw_vote,
        "presented_letter": ballot.vote,
        "reason": ballot.reason,
        "shared_fact_id": ballot.shared_fact_id,
        "raw_shared_fact_id": ballot.raw_shared_fact_id,
        "shared_fact_present": ballot.shared_fact_present,
    }


def validate_response(task: RelationalTask, vignette: Vignette, response: str) -> None:
    """Apply the same ballot response contract used by the production game."""

    compiled = build_prompt(task, vignette, NO_OP).compile()
    compiled.response_contract.validate(response).raise_for_errors(
        context="controller-retention response contract"
    )


def rendered_blocks(
    task: RelationalTask, vignette: Vignette, arm: str
) -> dict[str, str]:
    """Every rendered block of one prompt, for the pair-identity preflight."""

    compiled = build_prompt(task, vignette, arm).compile()
    return {block.name: block.content for block in compiled.blocks}


def visible_slot_count(
    task: RelationalTask, vignette: Vignette, arm: str
) -> int:
    return len(social_sources(task, vignette, arm))


__all__ = [
    "FOCAL_AGENT_INDEX",
    "NO_OP",
    "ONE_SLOT",
    "TWO_SLOTS",
    "build_prompt",
    "known_fact_lines",
    "option_letters",
    "parse_vote",
    "rendered_blocks",
    "social_sources",
    "validate_response",
    "visible_slot_count",
]
