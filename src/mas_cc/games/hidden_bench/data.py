"""Corpus loading and information assignment for the HiddenBench games (§3, §4).

**The corpus is not regenerated here.** It was downloaded and normalized by
`scripts/local_llms/hiddenbench_population_pipeline/`, which is the user's own
prior work; this module reads what that pipeline produced. See
`docs/hidden_bench/data_provenance.md` for exactly what was found where.

Two things the brief guessed at and this module corrects:

1. **Scheme names.** The brief's §4 taxonomy (`bijective` / `redundant` /
   `factorized` / `padded` / `decoy`) was a prior guess. The authoritative names
   are the pipeline's own `--method` values: `exact_replication`,
   `paraphrased_replication`, `factorized_evidence`. Those are the names used
   here. `bijective` is kept as an explicit alias for the paper's N == C
   baseline, and `padded` / `decoy` are implemented because they are genuine
   group-size controls the pipeline does not cover - they are marked as
   *mas_cc-local* schemes so nobody mistakes them for pipeline output.
2. **Where the allocation lives.** The pipeline writes the realized per-agent
   allocation into its scaled task files rather than leaving it to run time. For
   `exact_replication` the allocation rule is short, deterministic and
   LLM-free, so it is reimplemented here (`_balanced_type_assignment`) to make
   arbitrary N available without regenerating data files - and
   `tests/games/hidden_bench/test_data.py` asserts it reproduces the
   checked-in `N_32.json` agent-for-agent, so the two can never drift.
   `paraphrased_replication` and `factorized_evidence` depend on verified
   LLM-generated annotations and are therefore *only* read from prebuilt files;
   they are never synthesized here.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from mas_cc.core import AgentId

from .schemas import AgentInfoSet, HiddenBenchDataError, HiddenProfileTask

# Resolved relative to this file so it works from any working directory.
_REPO_ROOT = Path(__file__).resolve().parents[4]

DEFAULT_CORPUS_ROOT = (
    _REPO_ROOT / "scripts" / "local_llms" / "hiddenbench_population_pipeline" / "data" / "hiddenbench"
)
"""Where the pipeline actually put the corpus.

The brief's §2 layout proposed `data/hidden_bench/`; that directory does not
exist and inventing it would mean either copying 65 tasks to a second location
or silently diverging from the pipeline that maintains them. Overridable per
run with `game.options.corpus_root`.
"""

PIPELINE_SCHEMES = ("exact_replication", "paraphrased_replication", "factorized_evidence")
"""Scaling methods owned by `prepare_hiddenbench.py` - names are authoritative."""

LOCAL_SCHEMES = ("bijective", "padded", "decoy")
"""Schemes this module implements directly; not produced by the pipeline."""

SCHEMES = PIPELINE_SCHEMES + LOCAL_SCHEMES

PIPELINE_BASE_SEED = 1729
"""`prepare_hiddenbench.py --seed` default; reproducing it reproduces N_32."""


# --------------------------------------------------------------------------
# The pipeline's own primitives, reimplemented byte-for-byte.
# --------------------------------------------------------------------------


def stable_seed(*parts: Any) -> int:
    """`hiddenbench_common.stable_seed` - a content-addressed allocation seed.

    Reimplemented rather than imported because `scripts/` is not an importable
    package; the equality is pinned by test, not by hope.
    """

    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16)


def _balanced_type_assignment(n_agents: int, n_types: int, *, seed: int) -> list[int]:
    """`hiddenbench_common.balanced_type_assignment`.

    Deals evidence types round-robin so the counts differ by at most one, then
    shuffles, so which *agent index* holds which type carries no information.
    """

    if n_agents < n_types:
        raise HiddenBenchDataError(
            f"exact/paraphrased replication requires n_agents >= n_evidence_types; "
            f"got N={n_agents}, C={n_types}"
        )
    labels = list(range(n_types))
    while len(labels) < n_agents:
        labels.append((len(labels) - n_types) % n_types)
    random.Random(seed).shuffle(labels)
    return labels


def normalized_text(text: str) -> str:
    """`hiddenbench_common.normalized_text` - whitespace/case-insensitive key."""

    return re.sub(r"\s+", " ", text.strip().lower())


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def _read_json(path: Path) -> Any:
    if not path.exists():
        raise HiddenBenchDataError(
            f"HiddenBench corpus file not found: {path}\n"
            "The corpus is produced by scripts/local_llms/hiddenbench_population_pipeline/ - "
            "see docs/hidden_bench/data_provenance.md. It is not regenerated automatically."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_task(record: Mapping[str, Any]) -> HiddenProfileTask:
    hidden = record["hidden_information"]
    if not isinstance(hidden, Sequence) or isinstance(hidden, (str, bytes)):
        raise HiddenBenchDataError(f"task {record.get('name')!r}: hidden_information must be a list")
    texts: list[str] = []
    for index, item in enumerate(hidden):
        if not isinstance(item, Mapping):
            raise HiddenBenchDataError(
                f"task {record.get('name')!r}: hidden_information[{index}] must be an object"
            )
        if int(item["evidence_type"]) != index:
            # The index-is-the-type convention is load-bearing everywhere else.
            raise HiddenBenchDataError(
                f"task {record.get('name')!r}: hidden_information[{index}] declares "
                f"evidence_type {item['evidence_type']}, breaking the index==type convention"
            )
        texts.append(str(item["source_text"]))
    return HiddenProfileTask(
        task_id=int(record["task_id"]),
        name=str(record["name"]),
        source_description=str(record["source_description"]),
        scenario_description=str(record["scenario_description"]),
        shared_information=tuple(str(item) for item in record["shared_information"]),
        hidden_information=tuple(texts),
        possible_answers=tuple(str(item) for item in record["possible_answers"]),
        correct_answer=str(record["correct_answer"]),
        n_agents_native=int(record["source_base_agent_count"]),
        source="canonical",
        rationale=record.get("rationale"),
        validation_stats=record.get("validation_stats"),
    )


def _scaled_task(record: Mapping[str, Any], method: str) -> tuple[HiddenProfileTask, tuple[Mapping[str, Any], ...]]:
    """A prebuilt scaled task plus its frozen per-agent allocation."""

    population = record["population"]
    task = HiddenProfileTask(
        task_id=int(record["task_id"]),
        name=str(record["name"]),
        source_description=str(record["source_description"]),
        scenario_description=str(record["scenario_description"]),
        shared_information=tuple(str(item) for item in record["shared_information"]),
        hidden_information=tuple(str(item) for item in record["source_hidden_information"]),
        possible_answers=tuple(str(item) for item in record["possible_answers"]),
        correct_answer=str(record["correct_answer"]),
        n_agents_native=int(population["source_base_agent_count"]),
        source=f"scaled:{method}",
        rationale=record.get("rationale"),
        population_instruction=str(record["population_instruction"]),
    )
    return task, tuple(record["agents"])


@lru_cache(maxsize=8)
def _load_canonical(corpus_root: str) -> tuple[HiddenProfileTask, ...]:
    payload = _read_json(Path(corpus_root) / "canonical" / "tasks.json")
    return tuple(_canonical_task(item) for item in payload["tasks"])


@lru_cache(maxsize=8)
def _load_scaled(corpus_root: str, method: str, n_agents: int) -> tuple[tuple[HiddenProfileTask, tuple[Mapping[str, Any], ...]], ...]:
    path = Path(corpus_root) / "scaled" / method / f"N_{n_agents}.json"
    if not path.exists():
        raise HiddenBenchDataError(
            f"no prebuilt {method} population for N={n_agents} at {path}.\n"
            "Build it with:\n"
            f"  cd scripts/local_llms/hiddenbench_population_pipeline && "
            f"python scripts/prepare_hiddenbench.py --agents {n_agents} --method {method} "
            "--data-root data/hiddenbench"
            + ("" if method == "exact_replication" else "\n  (also needs --annotations; see that directory's README §6-7)")
        )
    payload = _read_json(path)
    return tuple(_scaled_task(item, method) for item in payload["tasks"])


@dataclass(frozen=True, slots=True)
class TaskSet:
    """The tasks one run may draw from, plus where they came from."""

    tasks: tuple[HiddenProfileTask, ...]
    corpus_root: str
    task_set: str
    prebuilt_allocations: Mapping[int, tuple[Mapping[str, Any], ...]]

    def by_name(self, name: str) -> HiddenProfileTask:
        for task in self.tasks:
            if task.name == name:
                return task
        raise HiddenBenchDataError(
            f"unknown task {name!r}; the {self.task_set} set has "
            f"{len(self.tasks)} tasks including {', '.join(t.name for t in self.tasks[:3])}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "corpus_root": self.corpus_root,
            "task_set": self.task_set,
            "task_count": len(self.tasks),
            "task_names": [task.name for task in self.tasks],
        }


def load_task_set(
    task_set: str = "vanilla",
    *,
    corpus_root: str | Path | None = None,
    scheme: str = "bijective",
    n_agents: int | None = None,
) -> TaskSet:
    """Load the corpus for one run.

    `task_set: vanilla` reads `canonical/tasks.json` - the 65 upstream tasks.
    `task_set: expanded` reads `scaled/<scheme>/N_<n_agents>.json` when such a
    file exists, so a prebuilt (and audited) population is always preferred over
    one derived at run time.
    """

    root = str(Path(corpus_root) if corpus_root is not None else DEFAULT_CORPUS_ROOT)
    if task_set == "vanilla":
        return TaskSet(_load_canonical(root), root, task_set, {})
    if task_set != "expanded":
        raise HiddenBenchDataError(f"game.options.task_set must be 'vanilla' or 'expanded', got {task_set!r}")
    if n_agents is None:
        raise HiddenBenchDataError("task_set 'expanded' requires n_agents")
    if scheme in LOCAL_SCHEMES:
        # These schemes have no pipeline output by construction; they are
        # derived from the canonical tasks at assignment time.
        return TaskSet(_load_canonical(root), root, task_set, {})
    try:
        loaded = _load_scaled(root, scheme, n_agents)
    except HiddenBenchDataError:
        if scheme != "exact_replication":
            raise
        # Deterministic and LLM-free: deriving it is equivalent to the file
        # (pinned by test), so a missing N is not a reason to fail.
        return TaskSet(_load_canonical(root), root, task_set, {})
    return TaskSet(
        tuple(task for task, _ in loaded),
        root,
        task_set,
        {task.task_id: agents for task, agents in loaded},
    )


# --------------------------------------------------------------------------
# §4 - information assignment
# --------------------------------------------------------------------------


def _agent_ids(n_agents: int) -> tuple[AgentId, ...]:
    return tuple(AgentId(f"agent-{index:03d}") for index in range(n_agents))


def _from_prebuilt(
    task: HiddenProfileTask, allocation: Sequence[Mapping[str, Any]], n_agents: int
) -> dict[AgentId, AgentInfoSet]:
    if len(allocation) != n_agents:
        raise HiddenBenchDataError(
            f"task {task.name!r}: prebuilt allocation has {len(allocation)} agents, run wants {n_agents}"
        )
    ids = _agent_ids(n_agents)
    return {
        ids[int(record["agent_id"])]: AgentInfoSet(
            shared=task.shared_information,
            private=tuple(str(item) for item in record["private_information"]),
            evidence_types=tuple(int(item) for item in record.get("source_hidden_indices", ())),
            transformation=str(record.get("transformation", "identity")),
        )
        for record in allocation
    }


def assign(
    task: HiddenProfileTask,
    n_agents: int,
    scheme: str,
    rng: random.Random,
    *,
    profile: str = "hidden",
    prebuilt: Sequence[Mapping[str, Any]] | None = None,
) -> dict[AgentId, AgentInfoSet]:
    """Split `I` into `{agent: I_i}` (§4).

    Under `profile: full` every agent receives the whole of `Iu` and `scheme` is
    ignored - that is the paper's Full Profile ceiling condition (§1.2), and it
    is a *profile*, not a separate game, precisely so `Y_full` and `Y_post` come
    out of the same code path.

    `rng` is the episode RNG. It is only consulted by schemes that genuinely
    have a free choice left (`padded`, `decoy`); `exact_replication` derives its
    own content-addressed seed so that an assignment is reproducible from the
    task id and N alone, exactly as the pipeline does it.
    """

    if profile == "full":
        every = AgentInfoSet(
            shared=task.shared_information,
            private=task.unshared_information,
            evidence_types=tuple(range(len(task.unshared_information))),
            transformation="full_profile",
        )
        return dict.fromkeys(_agent_ids(n_agents), every)
    if profile != "hidden":
        raise HiddenBenchDataError(f"game.options.profile must be 'hidden' or 'full', got {profile!r}")
    if scheme not in SCHEMES:
        raise HiddenBenchDataError(f"unknown assignment_scheme {scheme!r}; available: {', '.join(SCHEMES)}")
    if n_agents < 2:
        raise HiddenBenchDataError("hidden_bench needs at least two agents")

    if prebuilt is not None:
        return _from_prebuilt(task, prebuilt, n_agents)

    ids = _agent_ids(n_agents)
    hidden = task.unshared_information
    shared = task.shared_information

    if scheme in {"bijective", "exact_replication"}:
        if scheme == "bijective" and n_agents != task.n_agents_native:
            raise HiddenBenchDataError(
                f"scheme 'bijective' is the paper's N == C baseline and needs "
                f"n_agents == {task.n_agents_native} for task {task.name!r}, got {n_agents}. "
                "Use 'exact_replication' to scale it."
            )
        seed = stable_seed(PIPELINE_BASE_SEED, task.task_id, n_agents, "exact_replication")
        labels = _balanced_type_assignment(n_agents, len(hidden), seed=seed)
        return {
            ids[index]: AgentInfoSet(shared, (hidden[label],), (label,), "identity")
            for index, label in enumerate(labels)
        }

    if scheme in {"paraphrased_replication", "factorized_evidence"}:
        raise HiddenBenchDataError(
            f"scheme {scheme!r} is only available from a prebuilt population file - it depends on "
            "verified LLM-generated annotations and is never synthesized at run time. Build it with "
            "scripts/local_llms/hiddenbench_population_pipeline/scripts/prepare_hiddenbench.py "
            f"--agents {n_agents} --method {scheme} --annotations ..."
        )

    # ---- mas_cc-local group-size controls -------------------------------
    if scheme == "padded":
        # The first C agents carry the bijection; every extra agent gets shared
        # information only. Isolates "more agents" from "more distributed
        # information" - without it, N and |Iu| move together and no group-size
        # result can be attributed to either.
        if n_agents < len(hidden):
            raise HiddenBenchDataError(
                f"scheme 'padded' needs n_agents >= {len(hidden)} for task {task.name!r}"
            )
        holders = list(range(n_agents))
        rng.shuffle(holders)
        carrying = {agent_index: label for label, agent_index in enumerate(holders[: len(hidden)])}
        return {
            ids[index]: (
                AgentInfoSet(shared, (hidden[carrying[index]],), (carrying[index],), "identity")
                if index in carrying
                else AgentInfoSet(shared, (), (), "padding")
            )
            for index in range(n_agents)
        }

    # scheme == "decoy"
    if n_agents < len(hidden):
        raise HiddenBenchDataError(f"scheme 'decoy' needs n_agents >= {len(hidden)} for task {task.name!r}")
    holders = list(range(n_agents))
    rng.shuffle(holders)
    carrying = {agent_index: label for label, agent_index in enumerate(holders[: len(hidden)])}
    # A decoy-supporting filler is a *shared* fact restated as if privately
    # held: it adds pooling noise without adding any new proposition, so a
    # drop in accuracy cannot be blamed on new misinformation.
    fillers = shared or ("You have no additional private information.",)
    return {
        ids[index]: (
            AgentInfoSet(shared, (hidden[carrying[index]],), (carrying[index],), "identity")
            if index in carrying
            else AgentInfoSet(shared, (fillers[index % len(fillers)],), (), "decoy")
        )
        for index in range(n_agents)
    }


def assert_union_invariant(task: HiddenProfileTask, assignment: Mapping[AgentId, AgentInfoSet]) -> None:
    """§4's invariant: the union of every agent's private information is `Iu`.

    Checked on normalized text, because `paraphrased_replication` restates each
    fact and `factorized_evidence` splits it - under those schemes the union
    reconstructs `Iu` semantically rather than literally, so this function
    verifies *coverage of evidence types* there and literal set equality under
    the identity schemes.
    """

    transformations = {info.transformation for info in assignment.values()}
    if transformations <= {"identity", "padding", "full_profile"}:
        pooled = {normalized_text(item) for info in assignment.values() for item in info.private}
        expected = {normalized_text(item) for item in task.unshared_information}
        if pooled != expected:
            missing = expected - pooled
            raise HiddenBenchDataError(
                f"task {task.name!r}: pooled private information does not equal Iu; "
                f"{len(missing)} item(s) unreachable by any agent"
            )
        return
    covered = {kind for info in assignment.values() for kind in info.evidence_types}
    expected_types = set(range(len(task.unshared_information)))
    if not expected_types <= covered:
        raise HiddenBenchDataError(
            f"task {task.name!r}: evidence types {sorted(expected_types - covered)} are held by no agent"
        )


# --------------------------------------------------------------------------
# §7 - disclosure detection
# --------------------------------------------------------------------------

_STOPWORDS = frozenset(
    """a an and are as at be been but by for from has have in into is it its of on or that the
    their there these this to was were will with you your""".split()
)


def _keywords(text: str) -> frozenset[str]:
    words = re.findall(r"[a-z0-9]+", normalized_text(text))
    return frozenset(word for word in words if len(word) > 3 and word not in _STOPWORDS)


def disclosed_facts(
    hidden: Sequence[str], messages: Iterable[str], *, threshold: float = 0.6
) -> tuple[bool, ...]:
    """Which hidden facts have been surfaced in `messages`, one flag per fact.

    **A lower bound, by construction.** Detection is normalized keyword overlap:
    a fact counts as disclosed once some single message contains at least
    `threshold` of its content words. A faithful paraphrase that shares few
    content words will be missed, so every number derived from this reads "at
    least this much was surfaced", never "exactly". An LLM-judge variant is the
    obvious follow-up and is deliberately not v1 - it would make the central
    diagnostic of the benchmark depend on a second, unaudited model.
    """

    corpus = [_keywords(message) for message in messages]
    result = []
    for fact in hidden:
        keys = _keywords(fact)
        if not keys:
            result.append(False)
            continue
        result.append(any(len(keys & seen) / len(keys) >= threshold for seen in corpus))
    return tuple(result)
