"""An *independent* re-implementation of the frozen v1 spatial semantics.

This table is deliberately **not** imported from the generator.  Its whole job
is to check the generator's work: that the designated supporting facts really do
determine the stored answer, and - the claim the benchmark actually rests on -
that any strict subset of them really does *not*.  A checker that shares code
with the thing it checks cannot catch a wrong table, so this one is typed out
again from the schema documented in the generator README (§1) and pinned by a
test that replays every frozen example task through it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

RELATION_VECTORS: Mapping[str, tuple[int, int]] = {
    "NORTH": (0, 1),
    "NORTHEAST": (1, 1),
    "EAST": (1, 0),
    "SOUTHEAST": (1, -1),
    "SOUTH": (0, -1),
    "SOUTHWEST": (-1, -1),
    "WEST": (-1, 0),
    "NORTHWEST": (-1, 1),
}
"""``position(subject) - position(object) == RELATION_VECTORS[relation]``."""


def relation_from_delta(dx: int, dy: int) -> str | None:
    """The compass sector of a displacement, or ``None`` for the zero vector."""

    if dx == 0 and dy == 0:
        return None
    sign_x = (dx > 0) - (dx < 0)
    sign_y = (dy > 0) - (dy < 0)
    for name, (vx, vy) in RELATION_VECTORS.items():
        if (vx, vy) == (sign_x, sign_y):
            return name
    raise AssertionError(f"unreachable sign pair {(sign_x, sign_y)!r}")


def resolve_displacement(
    facts: Sequence[Any], subject: str, reference: str
) -> tuple[int, int] | None:
    """Net ``position(subject) - position(reference)`` implied by ``facts``.

    Returns ``None`` when the two entities are not connected by the given
    constraints - i.e. when the facts leave the query **underdetermined**, which
    is exactly the property a partial-evidence condition must have.

    ``facts`` are anything exposing ``.subject``, ``.relation`` and ``.object``
    (``RelationalFact`` does).  Each fact is traversed in both directions; the
    reverse direction negates the vector.
    """

    if subject == reference:
        return (0, 0)
    adjacency: dict[str, list[tuple[str, tuple[int, int]]]] = {}
    for fact in facts:
        # position(subject) - position(object) == vector, so stepping *from* the
        # object *to* the subject adds +vector, and the reverse step subtracts it.
        vector = RELATION_VECTORS[fact.relation]
        adjacency.setdefault(fact.object, []).append((fact.subject, vector))
        adjacency.setdefault(fact.subject, []).append(
            (fact.object, (-vector[0], -vector[1]))
        )
    if subject not in adjacency or reference not in adjacency:
        return None
    # Breadth-first accumulation from the reference; the world is consistent by
    # construction and independently re-validated by the generator, so the first
    # path found is the displacement.
    offsets: dict[str, tuple[int, int]] = {reference: (0, 0)}
    frontier = [reference]
    while frontier:
        current = frontier.pop()
        base = offsets[current]
        for neighbour, (dx, dy) in adjacency.get(current, ()):
            if neighbour in offsets:
                continue
            offsets[neighbour] = (base[0] + dx, base[1] + dy)
            frontier.append(neighbour)
    return offsets.get(subject)


def determined_relation(
    facts: Sequence[Any], subject: str, reference: str
) -> str | None:
    """The compass answer these facts force, or ``None`` if they force none."""

    displacement = resolve_displacement(facts, subject, reference)
    if displacement is None:
        return None
    return relation_from_delta(*displacement)


def feasible_options(
    shown_facts: Sequence[Any],
    omitted_facts: Sequence[Any],
    option_relations: Sequence[str],
    subject: str,
    reference: str,
) -> tuple[str, ...]:
    """Which displayed options are still reachable given only the shown facts.

    A partial-evidence condition leaves the query *displacement* undetermined -
    that is what :func:`determined_relation` returning ``None`` means, and the
    validator enforces it.  It does **not** follow that all the displayed
    options remain possible.  Each omitted supporting fact is one unit step, so
    the reachable set is the image of the shown constraints under every
    assignment of compass directions to the missing links, and a displayed
    relation outside that image can be eliminated without any further evidence.

    When this returns a single option the item is answerable by elimination, and
    its accuracy belongs in a different bucket from a genuinely 3-way choice.
    """

    import itertools

    if not omitted_facts:
        determined = determined_relation(shown_facts, subject, reference)
        return tuple(option for option in option_relations if option == determined)

    template = type(omitted_facts[0])
    reachable: set[str] = set()
    for assignment in itertools.product(RELATION_VECTORS, repeat=len(omitted_facts)):
        hypotheses = [
            template(
                fact_id=f"hypothesis_{index}",
                subject=fact.subject,
                relation=relation,
                object=fact.object,
                role="supporting",
                text="",
            )
            for index, (fact, relation) in enumerate(zip(omitted_facts, assignment))
        ]
        displacement = resolve_displacement(
            [*shown_facts, *hypotheses], subject, reference
        )
        if displacement is None:
            continue
        relation = relation_from_delta(*displacement)
        if relation is not None:
            reachable.add(relation)
    return tuple(option for option in option_relations if option in reachable)
