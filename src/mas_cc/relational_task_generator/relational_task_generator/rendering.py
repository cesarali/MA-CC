"""Deterministic natural-language rendering for symbolic spatial facts.

Version 1 intentionally uses one canonical template per relation.  The rendering
API is table-driven so additional deterministic paraphrase sets can be added later
without changing the symbolic task representation.
"""

from __future__ import annotations

from typing import Mapping


RELATION_PHRASES: Mapping[str, str] = {
    "NORTH": "north of",
    "NORTHEAST": "northeast of",
    "EAST": "east of",
    "SOUTHEAST": "southeast of",
    "SOUTH": "south of",
    "SOUTHWEST": "southwest of",
    "WEST": "west of",
    "NORTHWEST": "northwest of",
}


def render_fact(subject: str, relation: str, object_: str) -> str:
    """Render one symbolic fact using the canonical v1 language template."""
    try:
        phrase = RELATION_PHRASES[relation]
    except KeyError as exc:
        raise ValueError(f"Unknown relation: {relation}") from exc
    return f"{subject} is {phrase} {object_}."


def render_question(subject: str, reference: str) -> str:
    """Render the task query."""
    return f"Where is {subject} relative to {reference}?"
