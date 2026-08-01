from __future__ import annotations

import hashlib
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ID = "YuxuanLi1225/HiddenBench"
SOURCE_FILENAME = "benchmark.json"


class PipelineError(RuntimeError):
    pass


class ValidationError(PipelineError):
    pass


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any, *, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise PipelineError(
            f"{path} already exists. Use --overwrite to replace it."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*parts: Any) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16)


def normalized_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def deduplicate_texts(texts: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for text in texts:
        text = text.strip()
        key = normalized_text(text)
        if key and key not in seen:
            seen.add(key)
            result.append(text)
    return result


def parse_json_from_text(text: str) -> Any:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate)
        candidate = re.sub(r"\s*```$", "", candidate)

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        starts = [i for i in (candidate.find("{"), candidate.find("[")) if i >= 0]
        if not starts:
            raise ValidationError("LLM response contains no JSON.")
        start = min(starts)
        end = max(candidate.rfind("}"), candidate.rfind("]"))
        if end < start:
            raise ValidationError("LLM response contains incomplete JSON.")
        try:
            return json.loads(candidate[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValidationError(f"Could not parse LLM JSON: {exc}") from exc


REQUIRED_SOURCE_FIELDS = {
    "id",
    "name",
    "description",
    "shared_information",
    "hidden_information",
    "possible_answers",
    "correct_answer",
}


def validate_source_task(task: Mapping[str, Any]) -> None:
    missing = REQUIRED_SOURCE_FIELDS - set(task)
    if missing:
        raise ValidationError(
            f"Task {task.get('id', '<unknown>')} is missing {sorted(missing)}."
        )
    for field in ("shared_information", "hidden_information", "possible_answers"):
        value = task[field]
        if not isinstance(value, list) or not value:
            raise ValidationError(
                f"Task {task['id']} field {field} must be a non-empty list."
            )
        if not all(isinstance(item, str) and item.strip() for item in value):
            raise ValidationError(
                f"Task {task['id']} field {field} contains invalid text."
            )
    if task["correct_answer"] not in task["possible_answers"]:
        raise ValidationError(
            f"Task {task['id']} correct answer is not an available option."
        )


def load_source_tasks(path: Path) -> list[dict[str, Any]]:
    value = read_json(path)
    if not isinstance(value, list):
        raise ValidationError("The raw benchmark must be a JSON list.")
    tasks = [dict(item) for item in value]
    for task in tasks:
        validate_source_task(task)
    ids = [int(task["id"]) for task in tasks]
    if len(ids) != len(set(ids)):
        raise ValidationError("Duplicate task IDs in source benchmark.")
    return tasks


def download_source(
    data_root: Path,
    *,
    revision: str = "main",
    overwrite: bool = False,
) -> tuple[Path, dict[str, Any]]:
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise PipelineError(
            "Use the project's MA-CC conda environment."
        ) from exc

    source_dir = data_root / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    destination = source_dir / SOURCE_FILENAME

    if not destination.exists() or overwrite:
        cached = Path(
            hf_hub_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                filename=SOURCE_FILENAME,
                revision=revision,
            )
        )
        shutil.copy2(cached, destination)

    tasks = load_source_tasks(destination)
    resolved_sha = None
    try:
        resolved_sha = HfApi().dataset_info(REPO_ID, revision=revision).sha
    except Exception:
        pass

    metadata = {
        "repo_id": REPO_ID,
        "requested_revision": revision,
        "resolved_revision_sha": resolved_sha,
        "source_filename": SOURCE_FILENAME,
        "number_of_tasks": len(tasks),
        "sha256": sha256_file(destination),
        "note": (
            "The file is preserved from the authors' Hugging Face repository. "
            "The repository calls its only split 'train', but the rows are used "
            "as benchmark/evaluation tasks."
        ),
    }
    write_json(
        source_dir / "source_metadata.json",
        metadata,
        overwrite=True,
    )
    return destination, metadata


COUNT_REWRITES = [
    (re.compile(r"\byou and the other three\b", re.I), "you and the others"),
    (re.compile(r"\byou and three other\b", re.I), "you and other"),
    (
        re.compile(r"\byou will discuss with three other participants\b", re.I),
        "you will discuss with other participants",
    ),
    (
        re.compile(r"\byou will discuss with four other participants\b", re.I),
        "you will discuss with other participants",
    ),
    (re.compile(r"\bthree-person\b", re.I), "multi-person"),
    (re.compile(r"\bfour-person\b", re.I), "multi-person"),
    (re.compile(r"\bteam of four\b", re.I), "team"),
    (re.compile(r"\bgroup of four\b", re.I), "group"),
]


def neutralize_population_wording(description: str) -> tuple[str, list[dict[str, str]]]:
    current = description
    changes: list[dict[str, str]] = []
    for pattern, replacement in COUNT_REWRITES:
        while True:
            match = pattern.search(current)
            if match is None:
                break
            before = match.group(0)
            current = current[:match.start()] + replacement + current[match.end():]
            changes.append({"before": before, "after": replacement})
    return current, changes


def canonicalize_task(task: Mapping[str, Any]) -> dict[str, Any]:
    description, changes = neutralize_population_wording(str(task["description"]))
    hidden = list(task["hidden_information"])
    return {
        "task_id": int(task["id"]),
        "name": task["name"],
        "source_description": task["description"],
        "scenario_description": description,
        "population_wording_changes": changes,
        "shared_information": list(task["shared_information"]),
        "hidden_information": [
            {"evidence_type": index, "source_text": text}
            for index, text in enumerate(hidden)
        ],
        "possible_answers": list(task["possible_answers"]),
        "correct_answer": task["correct_answer"],
        "rationale": task.get("rationale"),
        "source_base_agent_count": len(hidden),
    }


def canonicalize_tasks(tasks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [canonicalize_task(task) for task in tasks]


def extract_tasks_payload(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Load a raw benchmark list, a canonical payload, or a scaled payload.
    """
    value = read_json(path)
    if isinstance(value, list):
        return [canonicalize_task(task) for task in value], {
            "kind": "source",
            "path": str(path),
        }
    if not isinstance(value, Mapping) or not isinstance(value.get("tasks"), list):
        raise ValidationError(
            "Expected a raw task list or an object containing a `tasks` list."
        )
    return [dict(task) for task in value["tasks"]], dict(value.get("metadata", {}))


def source_hidden_texts(task: Mapping[str, Any]) -> list[str]:
    values = task.get("source_hidden_information", task.get("hidden_information", []))
    result: list[str] = []
    for value in values:
        if isinstance(value, Mapping):
            result.append(str(value["source_text"]))
        else:
            result.append(str(value))
    return result


def task_id(task: Mapping[str, Any]) -> int:
    return int(task.get("task_id", task.get("id")))


def select_tasks(
    tasks: Sequence[Mapping[str, Any]], task_ids: Sequence[int] | None
) -> list[dict[str, Any]]:
    if not task_ids:
        return [dict(task) for task in tasks]
    wanted = set(task_ids)
    selected = [dict(task) for task in tasks if task_id(task) in wanted]
    missing = wanted - {task_id(task) for task in selected}
    if missing:
        raise ValidationError(f"Unknown task IDs: {sorted(missing)}")
    return selected


def balanced_type_assignment(
    num_agents: int,
    num_types: int,
    *,
    seed: int,
) -> list[int]:
    if num_agents < num_types:
        raise ValidationError(
            f"Exact/paraphrase replication requires N >= C. Got N={num_agents}, C={num_types}."
        )
    labels = list(range(num_types))
    while len(labels) < num_agents:
        labels.append((len(labels) - num_types) % num_types)
    rng = random.Random(seed)
    rng.shuffle(labels)
    return labels


def allocate_factor_components(
    components: Sequence[Mapping[str, Any]],
    num_agents: int,
    *,
    seed: int,
) -> tuple[list[list[Mapping[str, Any]]], dict[str, Any]]:
    """
    Allocate semantic factors to agents.

    Every component is assigned at least once. When N < M, agents receive multiple
    components. The greedy rule tries not to place two components from the same
    evidence type in one agent. When N > M, components are replicated evenly.
    """
    if num_agents <= 0:
        raise ValidationError("Factor allocation requires at least one agent.")
    if not components:
        raise ValidationError("No factor components were supplied.")

    rng = random.Random(seed)
    base = [dict(component) for component in components]
    rng.shuffle(base)

    allocation: list[list[Mapping[str, Any]]] = [[] for _ in range(num_agents)]
    held_types: list[Counter[int]] = [Counter() for _ in range(num_agents)]

    def choose_agent(evidence_type: int) -> int:
        without_type = [
            i for i in range(num_agents) if held_types[i][evidence_type] == 0
        ]
        candidates = without_type or list(range(num_agents))
        minimum = min(len(allocation[i]) for i in candidates)
        candidates = [i for i in candidates if len(allocation[i]) == minimum]
        return rng.choice(candidates)

    for component in base:
        evidence_type = int(component["evidence_type"])
        i = choose_agent(evidence_type)
        allocation[i].append(component)
        held_types[i][evidence_type] += 1

    # Extra agents should not remain empty. Replicate components as evenly as possible.
    empties = [i for i, packet in enumerate(allocation) if not packet]
    replica_index = 0
    for i in empties:
        component = dict(base[replica_index % len(base)])
        component["replica"] = True
        allocation[i].append(component)
        held_types[i][int(component["evidence_type"])] += 1
        replica_index += 1

    component_counts = Counter(
        str(component["component_id"])
        for packet in allocation
        for component in packet
    )

    # Diagnose whether an agent received all components of a latent type.
    total_by_type: dict[int, set[str]] = defaultdict(set)
    for component in base:
        total_by_type[int(component["evidence_type"])].add(
            str(component["component_id"])
        )

    complete_type_packets: list[dict[str, int]] = []
    for i, packet in enumerate(allocation):
        by_type: dict[int, set[str]] = defaultdict(set)
        for component in packet:
            by_type[int(component["evidence_type"])].add(
                str(component["component_id"])
            )
        for evidence_type, ids in by_type.items():
            if ids >= total_by_type[evidence_type]:
                complete_type_packets.append(
                    {"agent_id": i, "evidence_type": evidence_type}
                )

    diagnostics = {
        "num_components": len(base),
        "component_replication_counts": dict(component_counts),
        "agents_receiving_complete_evidence_type": complete_type_packets,
        "warning": (
            "Some agents receive all components of an evidence type. "
            "Run the information-sufficiency audit before using this condition."
            if complete_type_packets
            else None
        ),
    }
    return allocation, diagnostics


def normalize_vote(value: Any, possible_answers: Sequence[str]) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    for answer in possible_answers:
        if normalized_text(text) == normalized_text(answer):
            return answer
    # Conservative substring fallback for verbose model outputs.
    matches = [
        answer
        for answer in possible_answers
        if normalized_text(answer) in normalized_text(text)
    ]
    return matches[0] if len(matches) == 1 else None
