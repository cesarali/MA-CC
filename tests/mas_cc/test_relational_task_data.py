"""The frozen-task contract: what the game accepts, and what it refuses.

A malformed task is never repaired.  Every case below asserts that the loader
raises rather than quietly producing a runnable-looking episode on data that
would make its numbers meaningless.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mas_cc.games.relational_reasoning.data import (
    DEFAULT_TASK_DATASET_DIR,
    SCHEMA_VERSION,
    RelationalTaskError,
    list_relational_task_ids,
    load_relational_task,
)

pytestmark = pytest.mark.skipif(
    not (DEFAULT_TASK_DATASET_DIR / "task_0001.json").exists(),
    reason="the relational example dataset is not present",
)


def _payload(task_id: str = "task_0001") -> dict:
    return json.loads((DEFAULT_TASK_DATASET_DIR / f"{task_id}.json").read_text())


def _write(tmp_path: Path, payload: dict, name: str = "task_0001") -> Path:
    (tmp_path / f"{name}.json").write_text(json.dumps(payload))
    return tmp_path


# ---- loading -----------------------------------------------------------


def test_the_shipped_dataset_loads_and_every_task_validates():
    ids = list_relational_task_ids(DEFAULT_TASK_DATASET_DIR)
    assert len(ids) == 20
    for task_id in ids:
        task = load_relational_task(DEFAULT_TASK_DATASET_DIR, task_id)
        assert task.task_id == task_id
        assert task.population_size == len(task.agent_fact_ids) == 24
        assert task.correct_option in task.option_labels


def test_a_loaded_task_carries_the_exact_symbolic_and_rendered_content():
    task = load_relational_task(DEFAULT_TASK_DATASET_DIR, "task_0001", population_size=24)

    assert task.question == "Where is Bavi relative to Ralo?"
    assert task.supporting_fact_ids == ("f1", "f2")
    assert task.distractor_fact_ids == ("f3", "f4", "f5", "f6")
    assert task.option_relations == {"A": "NORTHEAST", "B": "SOUTHWEST", "C": "NORTH"}
    assert task.correct_option == "C"
    assert task.correct_relation == "NORTH"
    # The rendering is the generator's own, read out of the file rather than
    # recomputed here - there is exactly one source of truth for it.
    assert task.fact_text("f2") == "Zora is northwest of Ralo."
    assert task.fact("f2").relation == "NORTHWEST"
    assert task.reasoning_chain == (
        "Bavi is northeast of Zora.",
        "Zora is northwest of Ralo.",
    )


def test_the_initial_assignment_is_preserved_verbatim():
    task = load_relational_task(DEFAULT_TASK_DATASET_DIR, "task_0001")
    raw = _payload()["agents"]

    assert set(task.agent_fact_ids) == set(raw)
    for agent_id, entry in raw.items():
        assert set(task.known_facts(agent_id)) == set(entry["fact_ids"])
    # No agent holds the whole proof: this is a hidden-profile task.
    assert not any(
        set(task.supporting_fact_ids) <= set(ids)
        for ids in task.agent_fact_ids.values()
    )


def test_the_union_of_the_population_covers_every_supporting_fact():
    task = load_relational_task(DEFAULT_TASK_DATASET_DIR, "task_0001")
    pooled = {fact for ids in task.agent_fact_ids.values() for fact in ids}

    assert set(task.supporting_fact_ids) <= pooled


def test_omitting_the_task_id_takes_the_first_task_in_order():
    assert load_relational_task(DEFAULT_TASK_DATASET_DIR).task_id == "task_0001"


def test_a_json_suffix_on_the_task_id_is_accepted():
    assert load_relational_task(DEFAULT_TASK_DATASET_DIR, "task_0002.json").task_id == (
        "task_0002"
    )


# ---- refusals ----------------------------------------------------------


def test_a_missing_directory_or_task_is_reported_rather_than_guessed(tmp_path):
    with pytest.raises(RelationalTaskError, match="does not exist"):
        load_relational_task(tmp_path / "nowhere")
    with pytest.raises(RelationalTaskError, match="available"):
        load_relational_task(DEFAULT_TASK_DATASET_DIR, "task_9999")


def test_a_population_size_mismatch_is_refused(tmp_path):
    with pytest.raises(RelationalTaskError, match="population_size"):
        load_relational_task(DEFAULT_TASK_DATASET_DIR, "task_0001", population_size=6)


def test_an_unknown_schema_version_is_refused(tmp_path):
    payload = _payload()
    payload["schema_version"] = "spatial_relational_task_v2"

    with pytest.raises(RelationalTaskError, match="schema_version"):
        load_relational_task(_write(tmp_path, payload))
    assert SCHEMA_VERSION == "spatial_relational_task_v1"


def test_a_fact_referenced_by_an_agent_but_absent_from_the_world_is_refused(tmp_path):
    payload = _payload()
    payload["agents"]["agent_001"]["fact_ids"] = ["f99"]

    with pytest.raises(RelationalTaskError, match="unknown fact"):
        load_relational_task(_write(tmp_path, payload))


def test_a_supporting_fact_nobody_holds_is_refused(tmp_path):
    payload = _payload()
    for entry in payload["agents"].values():
        entry["fact_ids"] = [item for item in entry["fact_ids"] if item != "f1"]

    with pytest.raises(RelationalTaskError, match="does not collectively hold"):
        load_relational_task(_write(tmp_path, payload))


def test_an_unrendered_fact_is_refused(tmp_path):
    payload = _payload()
    payload["rendered"]["facts"].pop("f2")

    with pytest.raises(RelationalTaskError, match="no rendered text"):
        load_relational_task(_write(tmp_path, payload))


def test_a_correct_option_outside_the_option_set_is_refused(tmp_path):
    payload = _payload()
    payload["answer"]["correct_option"] = "Z"

    with pytest.raises(RelationalTaskError, match="not among the options"):
        load_relational_task(_write(tmp_path, payload))


def test_a_correct_option_disagreeing_with_the_correct_relation_is_refused(tmp_path):
    payload = _payload()
    payload["answer"]["correct_option"] = "A"

    with pytest.raises(RelationalTaskError, match="correct_relation"):
        load_relational_task(_write(tmp_path, payload))


def test_a_supporting_list_disagreeing_with_the_fact_roles_is_refused(tmp_path):
    payload = _payload()
    payload["query"]["supporting_fact_ids"] = ["f1"]

    with pytest.raises(RelationalTaskError, match="does not"):
        load_relational_task(_write(tmp_path, payload))


def test_a_supporting_id_pointing_at_a_distractor_is_refused(tmp_path):
    payload = _payload()
    payload["query"]["supporting_fact_ids"] = ["f1", "f3"]

    with pytest.raises(RelationalTaskError, match="role"):
        load_relational_task(_write(tmp_path, payload))


def test_a_declared_population_size_that_disagrees_with_the_agents_is_refused(tmp_path):
    payload = _payload()
    payload["generation"]["population_size"] = 25

    with pytest.raises(RelationalTaskError, match="generation.population_size"):
        load_relational_task(_write(tmp_path, payload))


def test_duplicate_answer_options_are_refused(tmp_path):
    payload = _payload()
    payload["answer"]["options"][0]["relation"] = payload["answer"]["options"][1][
        "relation"
    ]

    with pytest.raises(RelationalTaskError, match="duplicate answer option"):
        load_relational_task(_write(tmp_path, payload))


# ---- the matched r-variant datasets --------------------------------------

MATCHED_ROOT = DEFAULT_TASK_DATASET_DIR.parent / "datasets"
REDUNDANCIES = (1, 3, 6, 12)
MATCHED_DIRS = {r: MATCHED_ROOT / f"pop24_L2_r{r:02d}" for r in REDUNDANCIES}

matched = pytest.mark.skipif(
    not all(path.is_dir() for path in MATCHED_DIRS.values()),
    reason="the matched r-variant datasets are not present",
)

# Everything that defines the *problem*, as opposed to who was told what.
WORLD_KEYS = ("schema_version", "task_id", "seed", "world", "query", "answer", "rendered")


def _raw(r: int, index: int) -> dict:
    return json.loads((MATCHED_DIRS[r] / f"task_{index:04d}.json").read_text())


@matched
def test_each_matched_dataset_holds_ten_loadable_24_agent_tasks():
    for r, path in MATCHED_DIRS.items():
        ids = list_relational_task_ids(path)
        assert len(ids) == 10, r
        for task_id in ids:
            task = load_relational_task(path, task_id, population_size=24)
            assert task.population_size == 24
            assert task.reasoning_depth == len(task.supporting_fact_ids) == 2
            assert len(task.option_labels) == 3
            assert len(task.distractor_fact_ids) == 4


@matched
@pytest.mark.parametrize("index", range(1, 11))
def test_changing_r_changes_only_the_information_allocation(index):
    """The matched-design invariant, checked byte for byte.

    If `r` moved the world, the question or the answer as well as the
    allocation, then a difference between two `r` cells could be a difference
    between two *problems*, and the whole comparison would be meaningless.
    """

    base = _raw(REDUNDANCIES[0], index)
    for r in REDUNDANCIES[1:]:
        other = _raw(r, index)
        for key in WORLD_KEYS:
            assert json.dumps(other[key], sort_keys=True) == json.dumps(
                base[key], sort_keys=True
            ), f"{key} differs at r={r}"
        # The only generation parameter that may move is the one being swept.
        differing = {
            key
            for key in base["generation"]
            if base["generation"][key] != other["generation"][key]
        }
        assert differing == {"support_redundancy"}
        assert other["generation"]["support_redundancy"] == r
        # ...and the allocation really did change.
        assert other["agents"] != base["agents"]


@matched
@pytest.mark.parametrize("index", range(1, 11))
def test_every_supporting_fact_has_exactly_r_holders(index):
    for r in REDUNDANCIES:
        task = load_relational_task(MATCHED_DIRS[r], f"task_{index:04d}", population_size=24)
        for fact_id in task.supporting_fact_ids:
            holders = sum(
                1 for ids in task.agent_fact_ids.values() if fact_id in ids
            )
            assert holders == r, (index, r, fact_id)


@matched
@pytest.mark.parametrize("index", range(1, 11))
def test_no_agent_can_solve_alone_at_any_redundancy(index):
    """`no_single_agent_solution` has to hold at r = 12 too, where each fact is
    held by half the population - that is the tightest feasible point of
    `L*r <= N*(L-1)` for N=24, L=2."""

    for r in REDUNDANCIES:
        task = load_relational_task(MATCHED_DIRS[r], f"task_{index:04d}", population_size=24)
        support = set(task.supporting_fact_ids)
        assert not any(support <= set(ids) for ids in task.agent_fact_ids.values())
        pooled = {f for ids in task.agent_fact_ids.values() for f in ids}
        assert support <= pooled
