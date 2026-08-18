"""Validation logic for generated relational reasoning datasets."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple

from distribution import fact_recipient_counts
from generator import (
    RELATIONS,
    RELATION_VECTORS,
    canonical_json_bytes,
    dataset_fingerprint,
    generate_task,
    relation_from_delta,
    task_fingerprint,
)
from rendering import render_fact, render_question


Coord = Tuple[int, int]


def _fact_adjacency(facts: Sequence[Mapping[str, object]]) -> Dict[str, List[Tuple[str, Coord, str]]]:
    """Adjacency carrying coordinate increments.

    A fact ``subject REL object`` encodes pos(subject)-pos(object)=v.
    Therefore from subject -> object the increment is -v, and from object ->
    subject the increment is +v.
    """
    adjacency: Dict[str, List[Tuple[str, Coord, str]]] = {}
    for fact in facts:
        subject = str(fact["subject"])
        object_ = str(fact["object"])
        relation = str(fact["relation"])
        fid = str(fact["id"])
        dx, dy = RELATION_VECTORS[relation]
        adjacency.setdefault(subject, []).append((object_, (-dx, -dy), fid))
        adjacency.setdefault(object_, []).append((subject, (dx, dy), fid))
    return adjacency


def _check_constraint_consistency(
    facts: Sequence[Mapping[str, object]]
) -> Tuple[List[str], Dict[str, Coord]]:
    errors: List[str] = []
    adjacency = _fact_adjacency(facts)
    inferred: Dict[str, Coord] = {}

    for start in adjacency:
        if start in inferred:
            continue
        inferred[start] = (0, 0)
        queue = deque([start])
        while queue:
            node = queue.popleft()
            x, y = inferred[node]
            for other, (dx, dy), fid in adjacency.get(node, []):
                proposal = (x + dx, y + dy)
                if other not in inferred:
                    inferred[other] = proposal
                    queue.append(other)
                elif inferred[other] != proposal:
                    errors.append(
                        f"Constraint inconsistency at fact {fid}: {other} would be "
                        f"both {inferred[other]} and {proposal}."
                    )
    return errors, inferred


def _solve_relation_from_facts(
    facts: Sequence[Mapping[str, object]], subject: str, reference: str
) -> str | None:
    adjacency = _fact_adjacency(facts)
    if reference not in adjacency or subject not in adjacency:
        return None
    positions: Dict[str, Coord] = {reference: (0, 0)}
    queue = deque([reference])
    while queue:
        node = queue.popleft()
        x, y = positions[node]
        if node == subject:
            break
        for other, (dx, dy), _ in adjacency.get(node, []):
            proposal = (x + dx, y + dy)
            if other not in positions:
                positions[other] = proposal
                queue.append(other)
            elif positions[other] != proposal:
                return None
    if subject not in positions:
        return None
    dx, dy = positions[subject]
    if dx == 0 and dy == 0:
        return None
    return relation_from_delta(dx, dy)


def _shortest_path_length(
    facts: Sequence[Mapping[str, object]], subject: str, reference: str
) -> int | None:
    graph: Dict[str, Set[str]] = {}
    for fact in facts:
        a, b = str(fact["subject"]), str(fact["object"])
        graph.setdefault(a, set()).add(b)
        graph.setdefault(b, set()).add(a)
    if subject == reference:
        return 0
    if subject not in graph or reference not in graph:
        return None
    queue = deque([(subject, 0)])
    seen = {subject}
    while queue:
        node, dist = queue.popleft()
        for other in graph[node]:
            if other == reference:
                return dist + 1
            if other not in seen:
                seen.add(other)
                queue.append((other, dist + 1))
    return None


def validate_task(task: Mapping[str, object]) -> List[str]:
    """Return a list of validation errors; an empty list means valid."""
    errors: List[str] = []

    try:
        world = task["world"]  # type: ignore[index]
        query = task["query"]  # type: ignore[index]
        answer = task["answer"]  # type: ignore[index]
        agents = task["agents"]  # type: ignore[index]
        rendered = task["rendered"]  # type: ignore[index]
        generation = task["generation"]  # type: ignore[index]
    except Exception as exc:
        return [f"Missing top-level task structure: {exc}"]

    if task.get("schema_version") != "spatial_relational_task_v1":
        errors.append("Unexpected or missing schema_version")

    # ----- Entities and exact coordinates -----
    try:
        entity_rows = world["entities"]
        facts = world["facts"]
    except Exception as exc:
        return errors + [f"Malformed world block: {exc}"]

    entity_coords: Dict[str, Coord] = {}
    for entity in entity_rows:
        name = entity.get("name")
        coords = entity.get("coordinates", {})
        if not isinstance(name, str) or not name:
            errors.append(f"Invalid entity name: {name!r}")
            continue
        if name in entity_coords:
            errors.append(f"Duplicate entity name: {name}")
            continue
        x, y = coords.get("x"), coords.get("y")
        if not isinstance(x, int) or not isinstance(y, int):
            errors.append(f"Entity {name} must have integer x/y coordinates")
            continue
        entity_coords[name] = (x, y)

    if len(set(entity_coords.values())) != len(entity_coords):
        errors.append("Two or more world entities occupy the same stored coordinate")

    fact_by_id: Dict[str, Mapping[str, object]] = {}
    for fact in facts:
        fid = fact.get("id")
        if not isinstance(fid, str) or not fid:
            errors.append(f"Invalid fact id: {fid!r}")
            continue
        if fid in fact_by_id:
            errors.append(f"Duplicate fact id: {fid}")
            continue
        fact_by_id[fid] = fact
        relation = fact.get("relation")
        subject = fact.get("subject")
        object_ = fact.get("object")
        role = fact.get("role")
        if relation not in RELATIONS:
            errors.append(f"Fact {fid} has unknown relation {relation!r}")
            continue
        if role not in {"supporting", "distractor"}:
            errors.append(f"Fact {fid} has invalid role {role!r}")
        if subject not in entity_coords or object_ not in entity_coords:
            errors.append(f"Fact {fid} references an unknown entity")
            continue
        sx, sy = entity_coords[str(subject)]
        ox, oy = entity_coords[str(object_)]
        expected = RELATION_VECTORS[str(relation)]
        actual = (sx - ox, sy - oy)
        if actual != expected:
            errors.append(
                f"Fact {fid} conflicts with stored coordinates: expected delta "
                f"{expected}, got {actual}"
            )

    # Validate the fact constraints independently of the stored coordinates.
    valid_relation_facts = [
        f for f in facts if f.get("relation") in RELATIONS and f.get("id") in fact_by_id
    ]
    consistency_errors, _ = _check_constraint_consistency(valid_relation_facts)
    errors.extend(consistency_errors)

    # ----- Query and support chain -----
    subject = query.get("subject")
    reference = query.get("reference")
    depth = query.get("reasoning_depth")
    support_ids = query.get("supporting_fact_ids")
    if subject not in entity_coords or reference not in entity_coords:
        errors.append("Query references unknown entities")
    if subject == reference:
        errors.append("Query subject and reference must be distinct")
    if not isinstance(depth, int) or depth not in {1, 2, 3, 4}:
        errors.append("Query reasoning_depth must be one of 1, 2, 3, 4")
    if not isinstance(support_ids, list) or not support_ids:
        errors.append("supporting_fact_ids must be a non-empty list")
        support_ids = []
    if isinstance(depth, int) and len(support_ids) != depth:
        errors.append(
            f"supporting_fact_ids length {len(support_ids)} != reasoning_depth {depth}"
        )

    support_facts: List[Mapping[str, object]] = []
    for fid in support_ids:
        fact = fact_by_id.get(fid)
        if fact is None:
            errors.append(f"Supporting fact {fid} does not exist")
        elif fact.get("role") != "supporting":
            errors.append(f"Supporting fact {fid} is not marked role=supporting")
        else:
            support_facts.append(fact)

    all_role_support = {fid for fid, fact in fact_by_id.items() if fact.get("role") == "supporting"}
    if set(support_ids) != all_role_support:
        errors.append(
            "The set of role=supporting facts must exactly equal supporting_fact_ids"
        )

    if isinstance(subject, str) and isinstance(reference, str) and support_facts:
        inferred_answer = _solve_relation_from_facts(support_facts, subject, reference)
        if inferred_answer is None:
            errors.append("Designated supporting facts do not uniquely connect the query entities")
        elif inferred_answer != answer.get("correct_relation"):
            errors.append(
                f"Supporting facts imply {inferred_answer}, not stored answer "
                f"{answer.get('correct_relation')}"
            )
        shortest = _shortest_path_length(support_facts, subject, reference)
        if isinstance(depth, int) and shortest != depth:
            errors.append(
                f"Supporting graph shortest path is {shortest}, expected exactly L={depth}"
            )

    # Distractors are deliberately placed in disjoint entity sets in v1.  This
    # gives a strong, easy-to-audit irrelevance guarantee.
    support_entities: Set[str] = set()
    for fact in support_facts:
        support_entities.add(str(fact["subject"]))
        support_entities.add(str(fact["object"]))
    for fid, fact in fact_by_id.items():
        if fact.get("role") == "distractor":
            if str(fact.get("subject")) in support_entities or str(fact.get("object")) in support_entities:
                errors.append(
                    f"Distractor {fid} touches the query-support component and is not strictly irrelevant"
                )

    # Stored answer must also follow from the exact world coordinates.
    if isinstance(subject, str) and isinstance(reference, str):
        if subject in entity_coords and reference in entity_coords:
            sx, sy = entity_coords[subject]
            rx, ry = entity_coords[reference]
            if (sx, sy) == (rx, ry):
                errors.append("Queried entities occupy the same position")
            else:
                world_answer = relation_from_delta(sx - rx, sy - ry)
                if world_answer != answer.get("correct_relation"):
                    errors.append(
                        f"Stored correct_relation {answer.get('correct_relation')} does not "
                        f"follow from world coordinates ({world_answer})"
                    )

    # ----- Multiple-choice presentation -----
    correct_relation = answer.get("correct_relation")
    options = answer.get("options")
    correct_option = answer.get("correct_option")
    if correct_relation not in RELATIONS:
        errors.append("correct_relation is not a valid compass relation")
    if not isinstance(options, list):
        errors.append("answer.options must be a list")
        options = []
    labels: List[str] = []
    option_relations: List[str] = []
    for option in options:
        label = option.get("label") if isinstance(option, Mapping) else None
        relation = option.get("relation") if isinstance(option, Mapping) else None
        if not isinstance(label, str):
            errors.append(f"Invalid option label: {label!r}")
        else:
            labels.append(label)
        if relation not in RELATIONS:
            errors.append(f"Invalid option relation: {relation!r}")
        else:
            option_relations.append(str(relation))
    if len(option_relations) != len(set(option_relations)):
        errors.append("Answer options are not all distinct")
    if option_relations.count(str(correct_relation)) != 1:
        errors.append("Exactly one answer option must equal correct_relation")
    if correct_option not in labels:
        errors.append("correct_option does not name an existing option label")
    else:
        matching = [o for o in options if o.get("label") == correct_option]
        if len(matching) != 1 or matching[0].get("relation") != correct_relation:
            errors.append("correct_option label does not point to correct_relation")

    # ----- Distribution across agents -----
    population_size = generation.get("population_size")
    support_redundancy = generation.get("support_redundancy")
    distractor_redundancy = generation.get("distractor_redundancy")
    no_single = generation.get("no_single_agent_solution")
    if not isinstance(agents, Mapping):
        errors.append("agents must be a mapping")
        agents = {}
    if isinstance(population_size, int) and len(agents) != population_size:
        errors.append(
            f"Agent count {len(agents)} does not match population_size {population_size}"
        )

    all_fact_ids = set(fact_by_id)
    support_set = set(support_ids)
    for agent_id, payload in agents.items():
        ids = payload.get("fact_ids", []) if isinstance(payload, Mapping) else []
        if len(ids) != len(set(ids)):
            errors.append(f"Agent {agent_id} contains duplicate fact IDs")
        unknown = set(ids) - all_fact_ids
        if unknown:
            errors.append(f"Agent {agent_id} references unknown facts: {sorted(unknown)}")
        if no_single is True and support_set and support_set.issubset(set(ids)):
            errors.append(
                f"Agent {agent_id} possesses the complete supporting set despite "
                "no_single_agent_solution=true"
            )

    counts = fact_recipient_counts(agents)
    for fid in support_set:
        if counts.get(fid, 0) == 0:
            errors.append(f"Population does not collectively possess supporting fact {fid}")
        if isinstance(support_redundancy, int) and counts.get(fid, 0) != support_redundancy:
            errors.append(
                f"Supporting fact {fid} has redundancy {counts.get(fid, 0)}, "
                f"expected {support_redundancy}"
            )
    for fid, fact in fact_by_id.items():
        if fact.get("role") == "distractor":
            if counts.get(fid, 0) == 0:
                errors.append(f"Distractor fact {fid} is assigned to no agent")
            if isinstance(distractor_redundancy, int) and counts.get(fid, 0) != distractor_redundancy:
                errors.append(
                    f"Distractor fact {fid} has redundancy {counts.get(fid, 0)}, "
                    f"expected {distractor_redundancy}"
                )

    # ----- Generation metadata consistency -----
    for key in (
        "reasoning_depth",
        "population_size",
        "support_redundancy",
        "distractors",
        "distractor_redundancy",
        "num_options",
        "no_single_agent_solution",
    ):
        if key == "reasoning_depth" and generation.get(key) != depth:
            errors.append("generation.reasoning_depth disagrees with query.reasoning_depth")
        if key == "num_options" and isinstance(options, list) and generation.get(key) != len(options):
            errors.append("generation.num_options disagrees with answer.options length")
    distractor_count = sum(1 for f in fact_by_id.values() if f.get("role") == "distractor")
    if generation.get("distractors") != distractor_count:
        errors.append("generation.distractors disagrees with world fact roles")

    # ----- Deterministic language rendering -----
    rendered_facts = rendered.get("facts", {}) if isinstance(rendered, Mapping) else {}
    for fid, fact in fact_by_id.items():
        expected_text = render_fact(str(fact["subject"]), str(fact["relation"]), str(fact["object"]))
        if rendered_facts.get(fid) != expected_text:
            errors.append(f"Rendered text for {fid} is not the canonical deterministic rendering")
    if isinstance(subject, str) and isinstance(reference, str):
        expected_q = render_question(subject, reference)
        if rendered.get("question") != expected_q:
            errors.append("Rendered question is not canonical")
    expected_chain = [rendered_facts.get(fid) for fid in support_ids]
    if rendered.get("reasoning_chain") != expected_chain:
        errors.append("rendered.reasoning_chain does not match supporting_fact_ids order")

    return errors


def validate_dataset_directory(
    dataset_dir: Path, *, check_reproducibility: bool = True
) -> List[str]:
    """Validate a generated dataset directory, including seed reproducibility."""
    dataset_dir = Path(dataset_dir)
    errors: List[str] = []
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.exists():
        return [f"Missing manifest.json in {dataset_dir}"]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"Could not parse manifest.json: {exc}"]

    task_files = manifest.get("task_files", [])
    if not isinstance(task_files, list) or not task_files:
        return ["manifest.task_files must be a non-empty list"]

    actual_task_files = sorted(p.name for p in dataset_dir.glob("task_*.json"))
    if sorted(task_files) != actual_task_files:
        errors.append(
            "Task files on disk do not exactly match manifest.task_files"
        )

    tasks: List[dict] = []
    for filename in task_files:
        path = dataset_dir / filename
        if not path.exists():
            errors.append(f"Missing task file: {filename}")
            continue
        try:
            task = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"Could not parse {filename}: {exc}")
            continue
        tasks.append(task)
        for error in validate_task(task):
            errors.append(f"{filename}: {error}")

        expected_hash = manifest.get("task_fingerprints_sha256", {}).get(filename)
        actual_hash = task_fingerprint(task)
        if expected_hash != actual_hash:
            errors.append(f"{filename}: task fingerprint does not match manifest")

        if check_reproducibility:
            gen = task.get("generation", {})
            try:
                regenerated = generate_task(
                    task_id=task["task_id"],
                    task_seed=task["seed"],
                    dataset_seed=gen["dataset_seed"],
                    task_index=gen["task_index"],
                    population_size=gen["population_size"],
                    reasoning_depth=gen["reasoning_depth"],
                    support_redundancy=gen["support_redundancy"],
                    distractors=gen["distractors"],
                    distractor_redundancy=gen["distractor_redundancy"],
                    num_options=gen["num_options"],
                    no_single_agent_solution=gen["no_single_agent_solution"],
                )
                if canonical_json_bytes(regenerated) != canonical_json_bytes(task):
                    errors.append(
                        f"{filename}: regeneration from the stored seed/config is not identical"
                    )
            except Exception as exc:
                errors.append(f"{filename}: reproducibility regeneration failed: {exc}")

    if len(tasks) != manifest.get("num_tasks"):
        errors.append(
            f"Parsed {len(tasks)} tasks but manifest.num_tasks={manifest.get('num_tasks')}"
        )
    if tasks:
        actual_dataset_hash = dataset_fingerprint(tasks)
        if actual_dataset_hash != manifest.get("dataset_fingerprint_sha256"):
            errors.append("Dataset fingerprint does not match manifest")

    # Cross-check each task against the dataset-level config.
    config = manifest.get("config", {})
    dataset_seed = manifest.get("dataset_seed")
    for task in tasks:
        gen = task.get("generation", {})
        if gen.get("dataset_seed") != dataset_seed:
            errors.append(f"{task.get('task_id')}: dataset_seed disagrees with manifest")
        for key, value in config.items():
            if gen.get(key) != value:
                errors.append(
                    f"{task.get('task_id')}: generation.{key}={gen.get(key)!r} "
                    f"!= manifest config {value!r}"
                )

    return errors
