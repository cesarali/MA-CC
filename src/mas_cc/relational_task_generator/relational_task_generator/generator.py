"""Core generator for exact synthetic spatial relational reasoning tasks.

No LLMs, APIs, network calls, or external dependencies are used.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

from distribution import distribute_facts, validate_distribution_parameters
from rendering import render_fact, render_question


RELATION_VECTORS: Mapping[str, Tuple[int, int]] = {
    "NORTH": (0, 1),
    "NORTHEAST": (1, 1),
    "EAST": (1, 0),
    "SOUTHEAST": (1, -1),
    "SOUTH": (0, -1),
    "SOUTHWEST": (-1, -1),
    "WEST": (-1, 0),
    "NORTHWEST": (-1, 1),
}
RELATIONS: Tuple[str, ...] = tuple(RELATION_VECTORS)
OPTION_LABELS: Tuple[str, ...] = tuple("ABCDEFGH")

# Neutral, invented entity names.  Keeping a fixed table makes rendering fully
# deterministic and avoids external name-generation dependencies.
ENTITY_NAMES: Tuple[str, ...] = (
    "Lumo", "Kavi", "Tero", "Navi", "Selo", "Mira", "Daro", "Pavi",
    "Renu", "Vela", "Kiro", "Zani", "Faro", "Nelo", "Bira", "Tavi",
    "Jora", "Ceno", "Wira", "Havi", "Ralo", "Yani", "Demi", "Sora",
    "Pelo", "Gavi", "Lira", "Nori", "Viko", "Meno", "Cali", "Zora",
    "Rivo", "Feni", "Tala", "Belo", "Kora", "Javi", "Weno", "Hira",
    "Pira", "Savi", "Delo", "Nira", "Vani", "Romi", "Kelo", "Zavi",
    "Maro", "Fira", "Teni", "Bavi", "Leni", "Caro", "Yaro", "Jeni",
    "Garo", "Peni", "Haro", "Wali", "Seni", "Davi", "Naro", "Vero",
)


def relation_from_delta(dx: int, dy: int) -> str:
    """Map a non-zero exact displacement to one of eight qualitative directions."""
    if dx == 0 and dy == 0:
        raise ValueError("Coincident entities have no compass relation")
    sx = 0 if dx == 0 else (1 if dx > 0 else -1)
    sy = 0 if dy == 0 else (1 if dy > 0 else -1)
    reverse = {vector: relation for relation, vector in RELATION_VECTORS.items()}
    return reverse[(sx, sy)]


def stable_child_seed(*parts: object) -> int:
    """Derive a deterministic 64-bit seed from arbitrary values."""
    text = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(text).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _choose_entity_names(rng: random.Random, count: int) -> List[str]:
    if count > len(ENTITY_NAMES):
        raise ValueError(
            f"Need {count} entity names but v1 only defines {len(ENTITY_NAMES)}"
        )
    return rng.sample(list(ENTITY_NAMES), k=count)


def _generate_walk_relations(
    *, rng: random.Random, edge_count: int, anchor: Tuple[int, int]
) -> Tuple[List[str], List[Tuple[int, int]]]:
    """Generate a self-avoiding relation chain ending at ``anchor``.

    For relations r_i describing entity_i relative to entity_{i+1}, coordinates
    satisfy x_i = x_{i+1} + vector(r_i).  A self-avoiding walk guarantees that
    query endpoints are distinct and removes degenerate intermediate overlaps.
    """
    if edge_count < 1:
        raise ValueError("edge_count must be >= 1")

    for _ in range(1000):
        relations = [rng.choice(RELATIONS) for _ in range(edge_count)]
        coords: List[Tuple[int, int]] = [(0, 0)] * (edge_count + 1)
        coords[-1] = anchor
        used = {anchor}
        valid = True
        for i in range(edge_count - 1, -1, -1):
            dx, dy = RELATION_VECTORS[relations[i]]
            nxt = coords[i + 1]
            current = (nxt[0] + dx, nxt[1] + dy)
            if current in used:
                valid = False
                break
            coords[i] = current
            used.add(current)
        if valid:
            return relations, coords
    raise RuntimeError("Could not generate a self-avoiding spatial chain")


def _entity_records(names: Sequence[str], coords: Sequence[Tuple[int, int]]) -> List[dict]:
    return [
        {
            "name": name,
            "coordinates": {"x": int(coord[0]), "y": int(coord[1])},
        }
        for name, coord in zip(names, coords)
    ]


def _build_task_once(
    *,
    task_id: str,
    task_seed: int,
    dataset_seed: int,
    task_index: int,
    population_size: int,
    reasoning_depth: int,
    support_redundancy: int,
    distractors: int,
    distractor_redundancy: int,
    num_options: int,
    no_single_agent_solution: bool,
    attempt: int,
) -> dict:
    rng = random.Random(stable_child_seed(task_seed, "attempt", attempt))

    support_entity_count = reasoning_depth + 1
    distractor_entity_count = distractors + 1 if distractors > 0 else 0
    names = _choose_entity_names(
        rng, support_entity_count + distractor_entity_count
    )
    support_names = names[:support_entity_count]
    distractor_names = names[support_entity_count:]

    support_relations, support_coords = _generate_walk_relations(
        rng=rng, edge_count=reasoning_depth, anchor=(0, 0)
    )

    facts: List[dict] = []
    supporting_fact_ids: List[str] = []
    for i, relation in enumerate(support_relations, start=1):
        fact_id = f"f{i}"
        supporting_fact_ids.append(fact_id)
        facts.append(
            {
                "id": fact_id,
                "subject": support_names[i - 1],
                "relation": relation,
                "object": support_names[i],
                "role": "supporting",
            }
        )

    entities = _entity_records(support_names, support_coords)

    distractor_fact_ids: List[str] = []
    if distractors > 0:
        # A disconnected component makes distractors provably irrelevant to the
        # query chain while still forming a coherent symbolic world.
        distractor_relations, distractor_coords = _generate_walk_relations(
            rng=rng, edge_count=distractors, anchor=(1000, 1000)
        )
        entities.extend(_entity_records(distractor_names, distractor_coords))
        for j, relation in enumerate(distractor_relations, start=1):
            fact_index = reasoning_depth + j
            fact_id = f"f{fact_index}"
            distractor_fact_ids.append(fact_id)
            facts.append(
                {
                    "id": fact_id,
                    "subject": distractor_names[j - 1],
                    "relation": relation,
                    "object": distractor_names[j],
                    "role": "distractor",
                }
            )

    subject = support_names[0]
    reference = support_names[-1]
    sx, sy = support_coords[0]
    rx, ry = support_coords[-1]
    correct_relation = relation_from_delta(sx - rx, sy - ry)

    wrong_relations = [r for r in RELATIONS if r != correct_relation]
    selected_wrong = rng.sample(wrong_relations, k=num_options - 1)
    option_relations = [correct_relation, *selected_wrong]
    rng.shuffle(option_relations)
    options = [
        {"label": OPTION_LABELS[i], "relation": relation}
        for i, relation in enumerate(option_relations)
    ]
    correct_option = next(
        option["label"] for option in options if option["relation"] == correct_relation
    )

    agents = distribute_facts(
        supporting_fact_ids=supporting_fact_ids,
        distractor_fact_ids=distractor_fact_ids,
        population_size=population_size,
        support_redundancy=support_redundancy,
        distractor_redundancy=distractor_redundancy,
        no_single_agent_solution=no_single_agent_solution,
        rng=rng,
    )

    rendered_facts = {
        fact["id"]: render_fact(fact["subject"], fact["relation"], fact["object"])
        for fact in facts
    }

    task = {
        "schema_version": "spatial_relational_task_v1",
        "task_id": task_id,
        "seed": task_seed,
        "generation": {
            "dataset_seed": dataset_seed,
            "task_index": task_index,
            "population_size": population_size,
            "reasoning_depth": reasoning_depth,
            "support_redundancy": support_redundancy,
            "distractors": distractors,
            "distractor_redundancy": distractor_redundancy,
            "num_options": num_options,
            "no_single_agent_solution": no_single_agent_solution,
        },
        "world": {
            "coordinate_convention": (
                "For each fact (subject, relation, object), "
                "position(subject)-position(object) equals the unit vector of relation."
            ),
            "entities": entities,
            "facts": facts,
        },
        "query": {
            "subject": subject,
            "reference": reference,
            "reasoning_depth": reasoning_depth,
            "supporting_fact_ids": supporting_fact_ids,
        },
        "answer": {
            "correct_relation": correct_relation,
            "options": options,
            "correct_option": correct_option,
        },
        "agents": agents,
        "rendered": {
            "question": render_question(subject, reference),
            "facts": rendered_facts,
            "reasoning_chain": [rendered_facts[fid] for fid in supporting_fact_ids],
        },
    }
    return task


def validate_generation_parameters(
    *,
    population_size: int,
    reasoning_depth: int,
    support_redundancy: int,
    distractors: int,
    distractor_redundancy: int,
    num_options: int,
    no_single_agent_solution: bool,
) -> None:
    if reasoning_depth not in {1, 2, 3, 4}:
        raise ValueError("reasoning_depth must be one of 1, 2, 3, 4 in v1")
    if distractors < 0:
        raise ValueError("distractors must be >= 0")
    if not 2 <= num_options <= len(RELATIONS):
        raise ValueError("num_options must be between 2 and 8")
    validate_distribution_parameters(
        population_size=population_size,
        reasoning_depth=reasoning_depth,
        support_redundancy=support_redundancy,
        distractor_redundancy=distractor_redundancy,
        no_single_agent_solution=no_single_agent_solution,
    )


def generate_task(
    *,
    task_id: str,
    task_seed: int,
    dataset_seed: int,
    task_index: int,
    population_size: int = 24,
    reasoning_depth: int = 2,
    support_redundancy: int = 6,
    distractors: int = 4,
    distractor_redundancy: int = 1,
    num_options: int = 3,
    no_single_agent_solution: bool = False,
    max_attempts: int = 100,
) -> dict:
    """Generate one task and accept it only after automatic validation."""
    validate_generation_parameters(
        population_size=population_size,
        reasoning_depth=reasoning_depth,
        support_redundancy=support_redundancy,
        distractors=distractors,
        distractor_redundancy=distractor_redundancy,
        num_options=num_options,
        no_single_agent_solution=no_single_agent_solution,
    )

    # Local import avoids a module-level circular import: validation optionally
    # regenerates tasks when checking reproducibility.
    from validation import validate_task

    last_errors: List[str] = []
    for attempt in range(max_attempts):
        task = _build_task_once(
            task_id=task_id,
            task_seed=task_seed,
            dataset_seed=dataset_seed,
            task_index=task_index,
            population_size=population_size,
            reasoning_depth=reasoning_depth,
            support_redundancy=support_redundancy,
            distractors=distractors,
            distractor_redundancy=distractor_redundancy,
            num_options=num_options,
            no_single_agent_solution=no_single_agent_solution,
            attempt=attempt,
        )
        errors = validate_task(task)
        if not errors:
            return task
        last_errors = errors
    raise RuntimeError(
        f"Failed to generate a valid task after {max_attempts} attempts. "
        f"Last validation errors: {last_errors}"
    )


def canonical_json_bytes(data: object) -> bytes:
    """Stable representation used for fingerprints and reproducibility checks."""
    return json.dumps(
        data, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def task_fingerprint(task: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_json_bytes(task)).hexdigest()


def dataset_fingerprint(tasks: Sequence[Mapping[str, object]]) -> str:
    digest = hashlib.sha256()
    for task in tasks:
        digest.update(canonical_json_bytes(task))
        digest.update(b"\n")
    return digest.hexdigest()


def generate_dataset_in_memory(
    *,
    num_tasks: int,
    population_size: int = 24,
    reasoning_depth: int = 2,
    support_redundancy: int = 6,
    distractors: int = 4,
    distractor_redundancy: int = 1,
    num_options: int = 3,
    seed: int = 42,
    no_single_agent_solution: bool = False,
) -> List[dict]:
    if num_tasks < 1:
        raise ValueError("num_tasks must be >= 1")
    validate_generation_parameters(
        population_size=population_size,
        reasoning_depth=reasoning_depth,
        support_redundancy=support_redundancy,
        distractors=distractors,
        distractor_redundancy=distractor_redundancy,
        num_options=num_options,
        no_single_agent_solution=no_single_agent_solution,
    )

    width = max(4, len(str(num_tasks)))
    tasks: List[dict] = []
    for task_index in range(1, num_tasks + 1):
        task_id = f"task_{task_index:0{width}d}"
        task_seed = stable_child_seed(seed, "task", task_index)
        tasks.append(
            generate_task(
                task_id=task_id,
                task_seed=task_seed,
                dataset_seed=seed,
                task_index=task_index,
                population_size=population_size,
                reasoning_depth=reasoning_depth,
                support_redundancy=support_redundancy,
                distractors=distractors,
                distractor_redundancy=distractor_redundancy,
                num_options=num_options,
                no_single_agent_solution=no_single_agent_solution,
            )
        )
    return tasks


def _write_json(path: Path, data: object) -> None:
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def write_inspection_markdown(
    tasks: Sequence[Mapping[str, object]], output_path: Path, num_examples: int = 3
) -> None:
    """Write a compact human-readable audit view for a few tasks."""
    lines: List[str] = [
        "# Human-readable example inspection",
        "",
        "These are deterministic renderings of the symbolic JSON tasks. Exact coordinates",
        "remain available in each JSON file for auditing but are not repeated here; a",
        "downstream game should normally expose agents only to the facts listed in their",
        "`fact_ids`.",
        "",
    ]

    for task in tasks[:num_examples]:
        task_id = str(task["task_id"])
        world = task["world"]  # type: ignore[index]
        query = task["query"]  # type: ignore[index]
        answer = task["answer"]  # type: ignore[index]
        agents = task["agents"]  # type: ignore[index]
        rendered = task["rendered"]  # type: ignore[index]
        facts = world["facts"]  # type: ignore[index]
        support_ids = set(query["supporting_fact_ids"])  # type: ignore[index]
        fact_text = rendered["facts"]  # type: ignore[index]

        lines.extend([f"## {task_id}", "", "### Task", ""])
        lines.append(f"Seed: `{task['seed']}`")
        lines.append("")
        lines.append("### Supporting facts")
        lines.append("")
        for fid in query["supporting_fact_ids"]:  # type: ignore[index]
            lines.append(f"- `{fid}` — {fact_text[fid]}")
        lines.append("")
        lines.append("### Distractors")
        lines.append("")
        distractor_rows = [f for f in facts if f["id"] not in support_ids]
        if distractor_rows:
            for fact in distractor_rows:
                lines.append(f"- `{fact['id']}` — {fact_text[fact['id']]}")
        else:
            lines.append("- None")
        lines.append("")
        lines.append("### Distribution of facts across agents")
        lines.append("")
        for agent_id, payload in agents.items():
            ids = ", ".join(payload["fact_ids"]) if payload["fact_ids"] else "—"
            lines.append(f"- `{agent_id}`: {ids}")
        lines.append("")
        lines.append("### Question")
        lines.append("")
        lines.append(str(rendered["question"]))
        lines.append("")
        lines.append("### Reasoning chain")
        lines.append("")
        for idx, sentence in enumerate(rendered["reasoning_chain"], start=1):
            lines.append(f"{idx}. {sentence}")
        lines.append("")
        lines.append("### Correct answer")
        lines.append("")
        lines.append(
            f"`{answer['correct_relation']}` (option `{answer['correct_option']}`)"
        )
        lines.append("")
        lines.append("---")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_dataset(
    *,
    output_dir: Path,
    num_tasks: int,
    population_size: int = 24,
    reasoning_depth: int = 2,
    support_redundancy: int = 6,
    distractors: int = 4,
    distractor_redundancy: int = 1,
    num_options: int = 3,
    seed: int = 42,
    no_single_agent_solution: bool = False,
    overwrite: bool = False,
    inspection_examples: int = 3,
) -> dict:
    """Generate, validate, and atomically-ish write a frozen dataset folder."""
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory {output_dir} is not empty. Use --overwrite to replace it."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Avoid stale task files when overwriting.
    if overwrite:
        for old in output_dir.glob("task_*.json"):
            old.unlink()
        for old_name in ("manifest.json", "INSPECTION.md"):
            old = output_dir / old_name
            if old.exists():
                old.unlink()

    tasks = generate_dataset_in_memory(
        num_tasks=num_tasks,
        population_size=population_size,
        reasoning_depth=reasoning_depth,
        support_redundancy=support_redundancy,
        distractors=distractors,
        distractor_redundancy=distractor_redundancy,
        num_options=num_options,
        seed=seed,
        no_single_agent_solution=no_single_agent_solution,
    )

    task_files: List[str] = []
    task_hashes: Dict[str, str] = {}
    for task in tasks:
        filename = f"{task['task_id']}.json"
        task_files.append(filename)
        _write_json(output_dir / filename, task)
        task_hashes[filename] = task_fingerprint(task)

    manifest = {
        "schema_version": "spatial_relational_dataset_v1",
        "dataset_seed": seed,
        "num_tasks": num_tasks,
        "config": {
            "population_size": population_size,
            "reasoning_depth": reasoning_depth,
            "support_redundancy": support_redundancy,
            "distractors": distractors,
            "distractor_redundancy": distractor_redundancy,
            "num_options": num_options,
            "no_single_agent_solution": no_single_agent_solution,
        },
        "task_files": task_files,
        "task_fingerprints_sha256": task_hashes,
        "dataset_fingerprint_sha256": dataset_fingerprint(tasks),
    }
    _write_json(output_dir / "manifest.json", manifest)
    write_inspection_markdown(
        tasks, output_dir / "INSPECTION.md", num_examples=min(inspection_examples, num_tasks)
    )

    # Final independent validation before the caller considers the dataset saved.
    from validation import validate_dataset_directory

    errors = validate_dataset_directory(output_dir, check_reproducibility=True)
    if errors:
        raise RuntimeError(
            "Generated dataset failed final validation: " + " | ".join(errors[:10])
        )
    return manifest
