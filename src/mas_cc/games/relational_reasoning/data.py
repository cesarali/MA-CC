"""Loading and validation of **frozen** synthetic relational reasoning tasks.

A task file is an exact symbolic world produced once, offline, by
``src/mas_cc/relational_task_generator/``.  This module reads such a file and
refuses anything it does not fully understand.  It deliberately does **not**:

* run or import the generator (that folder is not a Python package, and an
  experiment must never depend on regeneration);
* re-render natural language (the frozen ``rendered`` block *is* the
  generator's deterministic rendering, and copying the relation-to-phrase table
  here would create a second source of truth that could silently drift);
* redistribute facts across agents (``agents[*].fact_ids`` is the initial
  knowledge state ``K_i(0)`` and is preserved verbatim);
* repair a malformed file.  Every check below raises
  :class:`RelationalTaskError` instead of patching the data.

The accepted schema is ``spatial_relational_task_v1``; see the generator's
README §7 and the checked-in ``examples/`` for the authoritative shape.
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = "spatial_relational_task_v1"
"""The one task schema this loader accepts."""

MUSR_TASK_FAMILY = "musr_team_allocation"
MUSR_SCHEMA_VERSION = "musr_team_allocation_native_v1"
MUSR_DISTRIBUTION_SCHEMA_VERSION = "musr_team_allocation_distribution_v1"
MUSR_INITIAL_INFORMATION_SCHEMA_VERSION = "musr_initial_information_v1"

SUPPORTING = "supporting"
DISTRACTOR = "distractor"
FACT_ROLES = (SUPPORTING, DISTRACTOR)

NO_FACT = "none"
"""The reserved ``shared_fact_id`` meaning "I am exposing no evidence"."""

DEFAULT_TASK_DATASET_DIR = (
    Path(__file__).resolve().parents[2]
    / "relational_task_generator"
    / "relational_task_generator"
    / "examples"
)
"""The dataset checked in next to the generator.

Note the doubled directory name: ``src/mas_cc/relational_task_generator/`` is
the drop location and ``relational_task_generator/`` inside it is the
generator's own self-contained folder.  Configs may still point
``game.options.task_dataset_dir`` anywhere else.
"""


class RelationalTaskError(ValueError):
    """A task file is missing, unreadable, or does not satisfy the v1 schema."""


@dataclass(frozen=True, slots=True)
class RelationalFact:
    """One symbolic spatial constraint plus its frozen natural-language form."""

    fact_id: str
    subject: str
    relation: str
    object: str
    role: str
    text: str

    @property
    def is_supporting(self) -> bool:
        return self.role == SUPPORTING

    def to_dict(self) -> dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "subject": self.subject,
            "relation": self.relation,
            "object": self.object,
            "role": self.role,
            "text": self.text,
        }


@dataclass(frozen=True, slots=True)
class RelationalTask:
    """One frozen task, in the shape the game actually needs.

    The **vote alphabet is semantic**: the compass relations, not the frozen
    ``A``/``B``/``C`` labels.  A label is a presentation detail, and one shared
    by the whole population would be its own attractor - agents could converge
    on "B" for reasons that have nothing to do with what B means, and the run
    would record consensus that is really position bias.  The frozen labels are
    kept for provenance (``option_labels``, ``correct_option``); each LLM call
    gets its own freshly shuffled letter->relation map instead, see
    ``game.RelationalImitationRoundFeedbackGame.option_letters``.
    """

    task_id: str
    seed: int
    population_size: int
    reasoning_depth: int
    question: str
    fact_order: tuple[str, ...]
    facts: Mapping[str, RelationalFact]
    supporting_fact_ids: tuple[str, ...]
    distractor_fact_ids: tuple[str, ...]
    option_labels: tuple[str, ...]
    option_relations: Mapping[str, str]
    correct_option: str
    correct_relation: str
    agent_ids: tuple[str, ...]
    agent_fact_ids: Mapping[str, tuple[str, ...]]
    reasoning_chain: tuple[str, ...]
    source_path: str
    task_family: str = "spatial_relational"
    answer_display_texts: Mapping[str, str] | None = None
    supporting_fact_groups: Mapping[str, tuple[str, ...]] | None = None

    def fact(self, fact_id: str) -> RelationalFact:
        try:
            return self.facts[fact_id]
        except KeyError as exc:
            raise RelationalTaskError(
                f"task {self.task_id!r} has no fact {fact_id!r}"
            ) from exc

    def fact_text(self, fact_id: str) -> str:
        """The generator's deterministic rendering, verbatim."""

        return self.fact(fact_id).text

    def known_facts(self, agent_id: str) -> tuple[str, ...]:
        """``K_i(0)`` for one agent, exactly as frozen in the file."""

        try:
            return self.agent_fact_ids[agent_id]
        except KeyError as exc:
            raise RelationalTaskError(
                f"task {self.task_id!r} has no agent {agent_id!r}"
            ) from exc

    @property
    def semantic_answers(self) -> tuple[str, ...]:
        """The vote alphabet: the compass relations, in frozen label order.

        Label order is a stable per-task ordering that is not the presentation
        order any agent sees, so it serves purely as the canonical index for
        occupation counts and metric columns.
        """

        return tuple(self.option_relations[label] for label in self.option_labels)

    def to_dict(self) -> dict[str, Any]:
        """The task projection carried in ``GameState.data['task']``."""

        projection = {
            "task_id": self.task_id,
            "task_seed": self.seed,
            "schema_version": (
                MUSR_SCHEMA_VERSION
                if self.task_family == MUSR_TASK_FAMILY
                else SCHEMA_VERSION
            ),
            "source_path": self.source_path,
            "question": self.question,
            "reasoning_depth": self.reasoning_depth,
            "population_size": self.population_size,
            # The alphabet the population state actually lives in.
            "possible_answers": list(self.semantic_answers),
            "correct_answer": self.correct_relation,
            # Frozen presentation labels, kept for provenance only: nothing in
            # the dynamics reads them.
            "option_labels": list(self.option_labels),
            "option_relations": dict(self.option_relations),
            "correct_option": self.correct_option,
            "correct_relation": self.correct_relation,
            "fact_order": list(self.fact_order),
            "facts": {key: value.to_dict() for key, value in self.facts.items()},
            "supporting_fact_ids": list(self.supporting_fact_ids),
            "distractor_fact_ids": list(self.distractor_fact_ids),
            "agent_fact_ids": {
                agent: list(ids) for agent, ids in self.agent_fact_ids.items()
            },
            "reasoning_chain": list(self.reasoning_chain),
        }
        # MuSR adds these fields, but their empty defaults must not perturb the
        # historical spatial projection: paired initialization artifacts use
        # this exact projection as part of their frozen compatibility hash.
        if self.task_family != "spatial_relational":
            projection["task_family"] = self.task_family
        if self.answer_display_texts:
            projection["answer_display_texts"] = dict(self.answer_display_texts)
        if self.supporting_fact_groups:
            projection["supporting_fact_groups"] = {
                key: list(values)
                for key, values in self.supporting_fact_groups.items()
            }
        return projection


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def list_relational_task_ids(dataset_dir: str | Path) -> tuple[str, ...]:
    """Every ``task_*.json`` stem in a dataset directory, in sorted order."""

    root = Path(dataset_dir)
    if not root.is_dir():
        raise RelationalTaskError(f"task dataset directory does not exist: {root}")
    return tuple(sorted(path.stem for path in root.glob("task_*.json")))


def load_relational_task(
    dataset_dir: str | Path,
    task_id: str | None = None,
    *,
    population_size: int | None = None,
) -> RelationalTask:
    """Read one frozen task and validate it against the v1 schema.

    ``task_id`` may be given with or without the ``.json`` suffix; omitting it
    selects the first task in the directory, which is only meant for smoke runs.
    ``population_size``, when given, must equal the task's own agent count -
    this is the check that stops a 24-agent hidden-profile task from silently
    running with 6 agents and two thirds of the evidence discarded.
    """

    root = Path(dataset_dir)
    if not root.is_dir():
        raise RelationalTaskError(f"task dataset directory does not exist: {root}")
    if task_id is None:
        available = list_relational_task_ids(root)
        if not available:
            raise RelationalTaskError(f"no task_*.json files under {root}")
        stem = available[0]
    else:
        stem = str(task_id)[:-5] if str(task_id).endswith(".json") else str(task_id)
    path = root / f"{stem}.json"
    if not path.is_file():
        available = ", ".join(list_relational_task_ids(root)) or "<none>"
        raise RelationalTaskError(
            f"task {stem!r} does not exist under {root}; available: {available}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RelationalTaskError(f"task file {path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise RelationalTaskError(f"task file {path} must contain a JSON object")
    return _build_task(payload, path, population_size)


def _sha256_object(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_musr_team_allocation_task(
    dataset_dir: str | Path,
    task_id: str,
    *,
    population_size: int,
    initial_information_path: str | Path | None = None,
    initial_information_sha256: str | None = None,
) -> RelationalTask:
    """Adapt a validated MuSR task and its distribution to this game."""

    root = Path(dataset_dir) / str(task_id)
    base_path = root / "base_task.json"
    distribution_path = root / f"distribution_N{population_size}.json"
    assignment_path = (
        None if initial_information_path is None else Path(initial_information_path)
    )
    for path in (base_path, assignment_path or distribution_path):
        if not path.is_file():
            raise RelationalTaskError(f"required MuSR task file does not exist: {path}")
    try:
        base = json.loads(base_path.read_text(encoding="utf-8"))
        assignment_source = assignment_path or distribution_path
        distribution = json.loads(assignment_source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RelationalTaskError(f"MuSR task JSON is invalid: {exc}") from exc
    if not isinstance(base, Mapping) or not isinstance(distribution, Mapping):
        raise RelationalTaskError("MuSR base task and distribution must be objects")
    if base.get("schema_version") != MUSR_SCHEMA_VERSION:
        raise RelationalTaskError(
            f"{base_path} must use schema {MUSR_SCHEMA_VERSION!r}"
        )
    if base.get("task_family") != MUSR_TASK_FAMILY:
        raise RelationalTaskError(f"{base_path} is not a MuSR Team Allocation task")
    expected_assignment_schema = (
        MUSR_INITIAL_INFORMATION_SCHEMA_VERSION
        if assignment_path is not None
        else MUSR_DISTRIBUTION_SCHEMA_VERSION
    )
    if distribution.get("schema_version") != expected_assignment_schema:
        raise RelationalTaskError(
            f"{assignment_source} must use schema {expected_assignment_schema!r}"
        )
    if str(base.get("task_id")) != str(task_id) or str(
        distribution.get("task_id")
    ) != str(task_id):
        raise RelationalTaskError("MuSR base/assignment task IDs do not match")
    semantic_hash = str(base.get("semantic_world_sha256", ""))
    expected_hash = _sha256_object(
        {key: value for key, value in base.items() if key != "semantic_world_sha256"}
    )
    if semantic_hash != expected_hash:
        raise RelationalTaskError("MuSR base task semantic_world_sha256 does not match")
    if distribution.get("semantic_world_sha256") != semantic_hash:
        raise RelationalTaskError("MuSR assignment does not match the base task")
    if assignment_path is None:
        fingerprint = distribution.get("fingerprint_sha256")
        if fingerprint != _sha256_object(
            {
                key: value
                for key, value in distribution.items()
                if key != "fingerprint_sha256"
            }
        ):
            raise RelationalTaskError(
                "MuSR distribution fingerprint_sha256 does not match"
            )
    else:
        if not initial_information_sha256:
            raise RelationalTaskError(
                "MuSR initial-information artifact requires an expected file SHA-256"
            )
        actual_file_hash = hashlib.sha256(assignment_path.read_bytes()).hexdigest()
        if actual_file_hash != initial_information_sha256:
            raise RelationalTaskError(
                "MuSR initial-information artifact file SHA-256 does not match"
            )
        assignment_hash = distribution.get("assignment_sha256")
        if assignment_hash != _sha256_object(
            {
                key: value
                for key, value in distribution.items()
                if key != "assignment_sha256"
            }
        ):
            raise RelationalTaskError(
                "MuSR initial-information assignment_sha256 does not match"
            )
    if int(distribution.get("population_size", -1)) != population_size:
        raise RelationalTaskError("MuSR distribution population size does not match")
    if (
        assignment_path is None
        and int(distribution.get("no_single_agent_violations", -1)) != 0
    ):
        raise RelationalTaskError(
            "MuSR distribution violates no-single-agent constraint"
        )

    facts: dict[str, RelationalFact] = {}
    groups: dict[str, list[str]] = {}
    order: list[str] = []
    for raw in _sequence(base.get("evidence", ()), "evidence", base_path):
        item = _mapping(raw, "evidence[]", base_path)
        evidence_id = str(_require(item, "evidence_id", base_path))
        latent_id = str(_require(item, "latent_fact_id", base_path))
        statements = _sequence(
            _require(item, "text", base_path), "evidence[].text", base_path
        )
        text = " ".join(str(statement).strip() for statement in statements).strip()
        if not text or evidence_id in facts:
            raise RelationalTaskError(
                "MuSR evidence IDs must be unique with non-empty text"
            )
        facts[evidence_id] = RelationalFact(
            fact_id=evidence_id,
            subject=latent_id,
            relation="EVIDENCE_FOR",
            object=latent_id,
            role=SUPPORTING,
            text=text,
        )
        order.append(evidence_id)
        groups.setdefault(latent_id, []).append(evidence_id)

    if assignment_path is not None:
        profile = tuple(
            str(item)
            for item in _sequence(
                distribution.get("evidence_profile_ids", ()),
                "evidence_profile_ids",
                assignment_source,
            )
        )
        if len(profile) != 9 or len(set(profile)) != 9:
            raise RelationalTaskError(
                "MuSR F9 initial-information profile must contain nine unique cards"
            )
        unknown_profile = set(profile) - set(facts)
        if unknown_profile:
            raise RelationalTaskError(
                f"MuSR initial-information profile has unknown evidence: "
                f"{sorted(unknown_profile)}"
            )
        card_latent = {
            evidence_id: latent_id
            for latent_id, evidence_ids in groups.items()
            for evidence_id in evidence_ids
        }
        if len({card_latent[evidence_id] for evidence_id in profile}) != 9:
            raise RelationalTaskError(
                "MuSR F9 initial-information profile must cover nine latent values"
            )
        selected = set(profile)
        order = [evidence_id for evidence_id in order if evidence_id in selected]
        facts = {evidence_id: facts[evidence_id] for evidence_id in order}
        groups = {
            latent_id: [
                evidence_id for evidence_id in evidence_ids if evidence_id in selected
            ]
            for latent_id, evidence_ids in groups.items()
            if any(evidence_id in selected for evidence_id in evidence_ids)
        }

    options_raw = _sequence(base.get("options", ()), "options", base_path)
    if len(options_raw) != 3:
        raise RelationalTaskError(
            "MuSR Team Allocation task must contain three options"
        )
    option_ids: list[str] = []
    display: dict[str, str] = {}
    for raw in options_raw:
        item = _mapping(raw, "options[]", base_path)
        option_id = str(_require(item, "id", base_path))
        text = str(_require(item, "display_text", base_path)).strip()
        if option_id in display or not text:
            raise RelationalTaskError(
                "MuSR option IDs must be unique with non-empty display text"
            )
        option_ids.append(option_id)
        display[option_id] = text
    gold = str(base.get("gold_answer", ""))
    if gold not in display:
        raise RelationalTaskError("MuSR gold_answer is not among the options")

    assignment_key = (
        "agent_assignments" if assignment_path is not None else "agent_evidence_ids"
    )
    assignments = _mapping(
        distribution.get(assignment_key), assignment_key, assignment_source
    )
    agent_fact_ids: dict[str, tuple[str, ...]] = {}

    def agent_number(value: Any) -> int:
        text = str(value)
        return (
            int(text.rsplit("_", 1)[-1]) - 1 if text.startswith("agent_") else int(text)
        )

    for raw_agent_id, raw_ids in sorted(
        assignments.items(), key=lambda pair: agent_number(pair[0])
    ):
        ids = tuple(
            str(item)
            for item in _sequence(raw_ids, f"{assignment_key}[]", assignment_source)
        )
        unknown = set(ids) - set(facts)
        if unknown:
            raise RelationalTaskError(
                f"MuSR agent references unknown evidence: {sorted(unknown)}"
            )
        if assignment_path is not None and len(ids) != 1:
            raise RelationalTaskError(
                "MuSR F9 initial-information assignment requires one card per agent"
            )
        agent_fact_ids[f"agent_{agent_number(raw_agent_id) + 1:03d}"] = tuple(
            evidence_id for evidence_id in order if evidence_id in set(ids)
        )
    if len(agent_fact_ids) != population_size:
        raise RelationalTaskError("MuSR distribution agent count does not match")
    if {item for ids in agent_fact_ids.values() for item in ids} != set(order):
        raise RelationalTaskError("MuSR population evidence union is incomplete")
    if assignment_path is not None:
        holder_counts = {
            evidence_id: sum(evidence_id in ids for ids in agent_fact_ids.values())
            for evidence_id in order
        }
        if sorted(holder_counts.values()) != [2, 2, 2, 3, 3, 3, 3, 3, 3]:
            raise RelationalTaskError(
                "MuSR F9 initial-information holder counts must be six 3s and three 2s"
            )
        declared_counts = distribution.get("card_holder_counts")
        if (
            declared_counts is not None
            and {
                str(key): int(value)
                for key, value in _mapping(
                    declared_counts, "card_holder_counts", assignment_source
                ).items()
            }
            != holder_counts
        ):
            raise RelationalTaskError(
                "MuSR initial-information card_holder_counts do not match assignment"
            )

    scenario = str(base.get("scenario", "")).strip()
    question = str(base.get("question", "")).strip()
    if not scenario or not question:
        raise RelationalTaskError("MuSR scenario and question must be non-empty")
    return RelationalTask(
        task_id=str(task_id),
        seed=int(base.get("generation", {}).get("task_seed", 0)),
        population_size=population_size,
        reasoning_depth=len(groups),
        question=f"SCENARIO\n{scenario}\n\nQUESTION\n{question}",
        fact_order=tuple(order),
        facts=facts,
        supporting_fact_ids=tuple(order),
        distractor_fact_ids=(),
        option_labels=tuple(chr(ord("A") + index) for index in range(len(option_ids))),
        option_relations={
            chr(ord("A") + index): option_id
            for index, option_id in enumerate(option_ids)
        },
        correct_option=chr(ord("A") + option_ids.index(gold)),
        correct_relation=gold,
        agent_ids=tuple(agent_fact_ids),
        agent_fact_ids=agent_fact_ids,
        reasoning_chain=(),
        source_path=f"{base_path}|{assignment_source}",
        task_family=MUSR_TASK_FAMILY,
        answer_display_texts=display,
        supporting_fact_groups={key: tuple(values) for key, values in groups.items()},
    )


def _require(payload: Mapping[str, Any], key: str, path: Path) -> Any:
    if key not in payload:
        raise RelationalTaskError(f"task file {path} is missing {key!r}")
    return payload[key]


def _mapping(value: Any, name: str, path: Path) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RelationalTaskError(f"task file {path}: {name} must be an object")
    return value


def _sequence(value: Any, name: str, path: Path) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise RelationalTaskError(f"task file {path}: {name} must be a list")
    return value


def _build_task(
    payload: Mapping[str, Any], path: Path, population_size: int | None
) -> RelationalTask:
    schema = str(payload.get("schema_version", ""))
    if schema != SCHEMA_VERSION:
        raise RelationalTaskError(
            f"task file {path} has schema_version {schema!r}; this game reads "
            f"{SCHEMA_VERSION!r} only"
        )
    task_id = str(_require(payload, "task_id", path))
    world = _mapping(_require(payload, "world", path), "world", path)
    query = _mapping(_require(payload, "query", path), "query", path)
    answer = _mapping(_require(payload, "answer", path), "answer", path)
    agents = _mapping(_require(payload, "agents", path), "agents", path)
    rendered = _mapping(_require(payload, "rendered", path), "rendered", path)
    generation = _mapping(payload.get("generation", {}), "generation", path)
    rendered_facts = _mapping(rendered.get("facts", {}), "rendered.facts", path)

    facts: dict[str, RelationalFact] = {}
    order: list[str] = []
    for entry in _sequence(world.get("facts", ()), "world.facts", path):
        item = _mapping(entry, "world.facts[]", path)
        for field in ("id", "subject", "relation", "object", "role"):
            if field not in item:
                raise RelationalTaskError(
                    f"task file {path}: a world fact is missing {field!r}"
                )
        fact_id = str(item["id"])
        if fact_id in facts:
            raise RelationalTaskError(
                f"task file {path}: duplicate fact id {fact_id!r}"
            )
        role = str(item["role"])
        if role not in FACT_ROLES:
            raise RelationalTaskError(
                f"task file {path}: fact {fact_id!r} has unknown role {role!r}"
            )
        if fact_id not in rendered_facts:
            raise RelationalTaskError(
                f"task file {path}: fact {fact_id!r} has no rendered text"
            )
        text = str(rendered_facts[fact_id]).strip()
        if not text:
            raise RelationalTaskError(
                f"task file {path}: fact {fact_id!r} has empty rendered text"
            )
        facts[fact_id] = RelationalFact(
            fact_id=fact_id,
            subject=str(item["subject"]),
            relation=str(item["relation"]),
            object=str(item["object"]),
            role=role,
            text=text,
        )
        order.append(fact_id)
    if not facts:
        raise RelationalTaskError(f"task file {path}: world.facts is empty")
    unrendered = sorted(set(rendered_facts) - set(facts))
    if unrendered:
        raise RelationalTaskError(
            f"task file {path}: rendered.facts names unknown fact(s) {unrendered}"
        )

    supporting = tuple(
        str(item)
        for item in _sequence(
            _require(query, "supporting_fact_ids", path),
            "query.supporting_fact_ids",
            path,
        )
    )
    if not supporting:
        raise RelationalTaskError(
            f"task file {path}: query.supporting_fact_ids is empty"
        )
    for fact_id in supporting:
        if fact_id not in facts:
            raise RelationalTaskError(
                f"task file {path}: supporting fact {fact_id!r} does not exist"
            )
        if not facts[fact_id].is_supporting:
            raise RelationalTaskError(
                f"task file {path}: fact {fact_id!r} is listed as supporting but has "
                f"role {facts[fact_id].role!r}"
            )
    declared_supporting = tuple(
        fact_id for fact_id in order if facts[fact_id].is_supporting
    )
    if set(declared_supporting) != set(supporting):
        raise RelationalTaskError(
            f"task file {path}: query.supporting_fact_ids {sorted(supporting)} does not "
            f"match the facts marked supporting {sorted(declared_supporting)}"
        )
    distractors = tuple(
        fact_id for fact_id in order if not facts[fact_id].is_supporting
    )

    labels: list[str] = []
    relations: dict[str, str] = {}
    for entry in _sequence(_require(answer, "options", path), "answer.options", path):
        item = _mapping(entry, "answer.options[]", path)
        if "label" not in item or "relation" not in item:
            raise RelationalTaskError(
                f"task file {path}: an answer option is missing label or relation"
            )
        label = str(item["label"])
        if label in relations:
            raise RelationalTaskError(
                f"task file {path}: duplicate answer option label {label!r}"
            )
        relation = str(item["relation"])
        if relation in relations.values():
            raise RelationalTaskError(
                f"task file {path}: duplicate answer option relation {relation!r}"
            )
        labels.append(label)
        relations[label] = relation
    if len(labels) < 2:
        raise RelationalTaskError(
            f"task file {path}: answer.options needs at least two options"
        )
    correct_option = str(_require(answer, "correct_option", path))
    correct_relation = str(_require(answer, "correct_relation", path))
    if correct_option not in relations:
        raise RelationalTaskError(
            f"task file {path}: correct_option {correct_option!r} is not among the options"
        )
    if relations[correct_option] != correct_relation:
        raise RelationalTaskError(
            f"task file {path}: correct_option {correct_option!r} maps to "
            f"{relations[correct_option]!r} but correct_relation is {correct_relation!r}"
        )

    assignment: dict[str, tuple[str, ...]] = {}
    for agent_id in sorted(agents):
        entry = _mapping(agents[agent_id], f"agents.{agent_id}", path)
        ids = tuple(
            str(item)
            for item in _sequence(
                entry.get("fact_ids", ()), f"agents.{agent_id}.fact_ids", path
            )
        )
        unknown = sorted(set(ids) - set(facts))
        if unknown:
            raise RelationalTaskError(
                f"task file {path}: agent {agent_id!r} references unknown fact(s) {unknown}"
            )
        if len(set(ids)) != len(ids):
            raise RelationalTaskError(
                f"task file {path}: agent {agent_id!r} lists a fact twice"
            )
        assignment[agent_id] = tuple(sorted(ids, key=order.index))
    if not assignment:
        raise RelationalTaskError(f"task file {path}: agents is empty")

    pooled = {fact_id for ids in assignment.values() for fact_id in ids}
    missing = sorted(set(supporting) - pooled)
    if missing:
        raise RelationalTaskError(
            f"task file {path}: the population does not collectively hold supporting "
            f"fact(s) {missing}; the task is unsolvable as distributed"
        )

    declared_population = generation.get("population_size", len(assignment))
    if int(declared_population) != len(assignment):
        raise RelationalTaskError(
            f"task file {path}: generation.population_size is {declared_population} but "
            f"{len(assignment)} agents are assigned"
        )
    if population_size is not None and int(population_size) != len(assignment):
        raise RelationalTaskError(
            f"game.population_size is {population_size} but task {task_id!r} distributes "
            f"its facts over {len(assignment)} agents; they must match exactly"
        )

    question = str(rendered.get("question", "")).strip()
    if not question:
        raise RelationalTaskError(f"task file {path}: rendered.question is missing")

    return RelationalTask(
        task_id=task_id,
        seed=int(payload.get("seed", 0)),
        population_size=len(assignment),
        reasoning_depth=int(query.get("reasoning_depth", len(supporting))),
        question=question,
        fact_order=tuple(order),
        facts=facts,
        supporting_fact_ids=tuple(
            fact_id for fact_id in order if fact_id in set(supporting)
        ),
        distractor_fact_ids=distractors,
        option_labels=tuple(labels),
        option_relations=relations,
        correct_option=correct_option,
        correct_relation=correct_relation,
        agent_ids=tuple(assignment),
        agent_fact_ids=assignment,
        reasoning_chain=tuple(
            str(item)
            for item in _sequence(
                rendered.get("reasoning_chain", ()), "rendered.reasoning_chain", path
            )
        ),
        source_path=str(path),
    )


__all__ = [
    "DEFAULT_TASK_DATASET_DIR",
    "DISTRACTOR",
    "FACT_ROLES",
    "MUSR_DISTRIBUTION_SCHEMA_VERSION",
    "MUSR_INITIAL_INFORMATION_SCHEMA_VERSION",
    "MUSR_SCHEMA_VERSION",
    "MUSR_TASK_FAMILY",
    "NO_FACT",
    "SCHEMA_VERSION",
    "SUPPORTING",
    "RelationalFact",
    "RelationalTask",
    "RelationalTaskError",
    "list_relational_task_ids",
    "load_musr_team_allocation_task",
    "load_relational_task",
]
