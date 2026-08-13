"""Corpus and assignment properties (brief §9.2).

The load-time assertions in `schemas.py` run on all 65 tasks, and the §4 union
invariant is checked for every (scheme, n_agents) pair the games can produce.
"""

from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path

import pytest

from mas_cc.games.hidden_bench.data import (
    DEFAULT_CORPUS_ROOT,
    PIPELINE_BASE_SEED,
    _balanced_type_assignment,
    assign,
    assert_union_invariant,
    disclosed_facts,
    load_task_set,
    stable_seed,
)
from mas_cc.games.hidden_bench.schemas import AgentInfoSet, HiddenBenchDataError, HiddenProfileTask

pytestmark = pytest.mark.skipif(
    not (DEFAULT_CORPUS_ROOT / "canonical" / "tasks.json").exists(),
    reason="HiddenBench corpus not present; see docs/hidden_bench/data_provenance.md",
)


@pytest.fixture(scope="module")
def tasks() -> tuple[HiddenProfileTask, ...]:
    return load_task_set("vanilla").tasks


def test_corpus_loads_all_65_tasks_and_every_assertion_passes(tasks):
    """Constructing a `HiddenProfileTask` *is* the §3.2 assertion suite."""

    assert len(tasks) == 65
    for task in tasks:
        assert task.correct_answer in task.possible_answers
        assert len(task.possible_answers) >= 3
        assert len(task.hidden_information) == task.n_agents_native
        assert len(set(task.hidden_information)) == len(task.hidden_information)
        assert not set(task.shared_information) & set(task.hidden_information)


def test_task_ids_and_names_are_unique(tasks):
    assert len({task.task_id for task in tasks}) == len(tasks)
    assert len({task.name for task in tasks}) == len(tasks)


@pytest.mark.parametrize("field", ["correct_answer", "possible_answers", "hidden_information"])
def test_malformed_task_is_rejected_loudly(tasks, field):
    """A broken task must raise, never be silently dropped or repaired."""

    base = tasks[0]
    fields = {
        "task_id": base.task_id,
        "name": base.name,
        "source_description": base.source_description,
        "scenario_description": base.scenario_description,
        "shared_information": base.shared_information,
        "hidden_information": base.hidden_information,
        "possible_answers": base.possible_answers,
        "correct_answer": base.correct_answer,
        "n_agents_native": base.n_agents_native,
        "source": "test",
    }
    fields[field] = {
        "correct_answer": "Not An Option",
        "possible_answers": ("Only", "Two"),
        "hidden_information": base.hidden_information[:1],
    }[field]
    with pytest.raises(HiddenBenchDataError):
        HiddenProfileTask(**fields)


def test_shared_and_hidden_overlap_is_rejected(tasks):
    base = tasks[0]
    with pytest.raises(HiddenBenchDataError, match="both shared_information"):
        HiddenProfileTask(
            task_id=base.task_id,
            name=base.name,
            source_description=base.source_description,
            scenario_description=base.scenario_description,
            shared_information=(*base.shared_information, base.hidden_information[0]),
            hidden_information=base.hidden_information,
            possible_answers=base.possible_answers,
            correct_answer=base.correct_answer,
            n_agents_native=base.n_agents_native,
            source="test",
        )


# --------------------------------------------------------------------------
# §4 - the union invariant, for every scheme the games can select
# --------------------------------------------------------------------------


@pytest.mark.parametrize("scheme", ["exact_replication", "padded", "decoy"])
@pytest.mark.parametrize("n_agents", [4, 5, 7, 12])
def test_union_of_private_information_reconstructs_iu(tasks, scheme, n_agents):
    """Every scheme, every N: no hidden fact may be unreachable by pooling."""

    exercised = 0
    for task in tasks:
        if n_agents < len(task.hidden_information):
            continue
        assignment = assign(task, n_agents, scheme, random.Random(11))
        assert len(assignment) == n_agents
        assert_union_invariant(task, assignment)
        exercised += 1
    assert exercised, f"{scheme} at N={n_agents} exercised no task"


def test_bijective_is_one_item_per_agent_and_only_at_native_size(tasks):
    task = tasks[0]
    assignment = assign(task, task.n_agents_native, "bijective", random.Random(0))
    assert all(len(info.private) == 1 for info in assignment.values())
    assert_union_invariant(task, assignment)
    with pytest.raises(HiddenBenchDataError, match="N == C baseline"):
        assign(task, task.n_agents_native + 2, "bijective", random.Random(0))


def test_padded_gives_extra_agents_no_private_information(tasks):
    """The group-size control: more agents, exactly the same information."""

    task = tasks[0]
    n = task.n_agents_native + 3
    assignment = assign(task, n, "padded", random.Random(5))
    empty = [info for info in assignment.values() if not info.private]
    assert len(empty) == 3
    assert all(info.transformation == "padding" for info in empty)
    assert_union_invariant(task, assignment)


def test_full_profile_gives_every_agent_all_of_iu(tasks):
    """§1.2's ceiling condition: `scheme` is ignored, everyone holds everything."""

    task = tasks[0]
    assignment = assign(task, 6, "bijective", random.Random(0), profile="full")
    assert len(assignment) == 6
    for info in assignment.values():
        assert info.private == task.unshared_information
        assert info.transformation == "full_profile"


def test_annotation_dependent_schemes_refuse_to_invent_evidence(tasks):
    """Paraphrase/factorization need prepared, verified annotation allocations."""

    for scheme in ("paraphrased_replication", "factorized_evidence"):
        with pytest.raises(HiddenBenchDataError, match="prepared population allocation"):
            assign(tasks[0], 8, scheme, random.Random(0))


def test_unknown_scheme_names_the_available_ones(tasks):
    with pytest.raises(HiddenBenchDataError, match="exact_replication"):
        assign(tasks[0], 4, "redundant", random.Random(0))


def test_union_invariant_catches_a_missing_fact(tasks):
    """The invariant must actually fail when a fact is unreachable."""

    task = tasks[0]
    crippled = {
        agent_id: AgentInfoSet(info.shared, (), (), "identity")
        for agent_id, info in assign(task, 4, "bijective", random.Random(0)).items()
    }
    with pytest.raises(HiddenBenchDataError, match="unreachable"):
        assert_union_invariant(task, crippled)


# --------------------------------------------------------------------------
# The reimplemented pipeline allocation must not drift from the checked-in file
# --------------------------------------------------------------------------

_SCALED = DEFAULT_CORPUS_ROOT / "scaled" / "exact_replication" / "N_32.json"


def test_producer_and_consumer_agree_on_where_hidden_bench_data_lives():
    """The pipeline writes the corpus and annotations under the shared data root.

    The two constants are declared in different trees (the pipeline's `scripts/`
    is not an importable package), so nothing but this test stops them drifting
    back apart the way they had, with the corpus buried under `scripts/`.
    """

    pipeline_scripts = (
        DEFAULT_CORPUS_ROOT.parents[1]
        / "scripts"
        / "local_llms"
        / "hiddenbench_population_pipeline"
        / "scripts"
    )
    spec = importlib.util.spec_from_file_location(
        "hiddenbench_common_for_test", pipeline_scripts / "hiddenbench_common.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.DEFAULT_DATA_ROOT == DEFAULT_CORPUS_ROOT
    assert module.DEFAULT_ANNOTATIONS_DIR == DEFAULT_CORPUS_ROOT / "annotations"


@pytest.mark.skipif(not _SCALED.exists(), reason="prebuilt N_32 population not present")
def test_derived_exact_replication_matches_the_pipelines_own_output(tasks):
    """`data.py` re-derives `prepare_hiddenbench.py`'s allocation exactly.

    This is what licenses deriving arbitrary N in-process instead of shipping a
    file per N. If the pipeline's rule ever changes, this fails rather than the
    two silently disagreeing about who knows what.
    """

    prebuilt = {
        record["task_id"]: record
        for record in json.loads(Path(_SCALED).read_text(encoding="utf-8"))["tasks"]
    }
    assert len(prebuilt) == 65
    for task in tasks:
        seed = stable_seed(PIPELINE_BASE_SEED, task.task_id, 32, "exact_replication")
        derived = _balanced_type_assignment(32, len(task.hidden_information), seed=seed)
        expected = [int(agent["evidence_type"]) for agent in prebuilt[task.task_id]["agents"]]
        assert derived == expected, f"allocation drift on task {task.name!r}"


@pytest.mark.skipif(not _SCALED.exists(), reason="prebuilt N_32 population not present")
def test_prebuilt_population_is_preferred_over_derivation():
    """When a scaled file exists it is used, allocation and all."""

    task_set = load_task_set("expanded", scheme="exact_replication", n_agents=32)
    assert len(task_set.prebuilt_allocations) == 65
    task = task_set.by_name("evacuation_west_city")
    assignment = assign(
        task,
        32,
        "exact_replication",
        random.Random(0),
        prebuilt=task_set.prebuilt_allocations[task.task_id],
    )
    assert len(assignment) == 32
    assert_union_invariant(task, assignment)


def test_missing_population_error_names_the_command_that_builds_it():
    with pytest.raises(HiddenBenchDataError, match="prepare_hiddenbench.py"):
        load_task_set("expanded", scheme="factorized_evidence", n_agents=32)


def _write_paraphrase_preparation_fixture(root: Path, *, include_annotations: bool) -> None:
    def task(task_id: int, name: str) -> dict:
        return {
            "task_id": task_id,
            "name": name,
            "source_description": f"Source scenario {task_id}",
            "scenario_description": f"Scenario {task_id}",
            "population_wording_changes": [],
            "shared_information": [f"Shared {task_id}"],
            "hidden_information": [
                {"evidence_type": 0, "source_text": f"Hidden {task_id}-0"},
                {"evidence_type": 1, "source_text": f"Hidden {task_id}-1"},
            ],
            "possible_answers": ["A", "B", "C"],
            "correct_answer": "C",
            "rationale": None,
            "source_base_agent_count": 2,
        }

    tasks = [task(7, "portable_task_one"), task(8, "portable_task_two")]
    canonical = root / "canonical" / "tasks.json"
    canonical.parent.mkdir(parents=True)
    canonical.write_text(json.dumps({"metadata": {}, "tasks": tasks}), encoding="utf-8")
    if not include_annotations:
        return

    annotations = {"tasks": {}}
    for item in tasks:
        evidence = {}
        for evidence_type, source in enumerate(item["hidden_information"]):
            evidence[str(evidence_type)] = {
                "source_text": source["source_text"],
                "variants": [
                    {
                        "variant_id": f"{item['task_id']}-{evidence_type}-{index}",
                        "text": f"Paraphrase {item['task_id']}-{evidence_type}-{index}",
                        "accepted": True,
                    }
                    for index in range(3)
                ],
            }
        annotations["tasks"][str(item["task_id"])] = {
            "name": item["name"],
            "evidence_types": evidence,
        }
    path = root / "annotations" / "paraphrases.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(annotations), encoding="utf-8")


def test_auto_preparation_builds_and_appends_only_from_existing_paraphrases(tmp_path):
    _write_paraphrase_preparation_fixture(tmp_path, include_annotations=True)

    first = load_task_set(
        "expanded",
        corpus_root=tmp_path,
        scheme="paraphrased_replication",
        n_agents=6,
        requested_task="portable_task_one",
        auto_prepare_paraphrases=True,
    )
    second = load_task_set(
        "expanded",
        corpus_root=tmp_path,
        scheme="paraphrased_replication",
        n_agents=6,
        requested_task="8",
        auto_prepare_paraphrases=True,
    )

    population_path = tmp_path / "scaled" / "paraphrased_replication" / "N_6.json"
    payload = json.loads(population_path.read_text(encoding="utf-8"))
    assert first.by_name("portable_task_one").source == "scaled:paraphrased_replication"
    assert second.by_name("8").name == "portable_task_two"
    assert [record["task_id"] for record in payload["tasks"]] == [7, 8]
    assert all(len(record["agents"]) == 6 for record in payload["tasks"])
    assert all(
        len({agent["variant_id"] for agent in record["agents"]}) == 6
        for record in payload["tasks"]
    )

    # Once the requested task is ready, loading it does not require the source
    # annotation file to remain present.
    (tmp_path / "annotations" / "paraphrases.json").unlink()
    reused = load_task_set(
        "expanded",
        corpus_root=tmp_path,
        scheme="paraphrased_replication",
        n_agents=6,
        requested_task="portable_task_two",
        auto_prepare_paraphrases=True,
    )
    assert reused.by_name("portable_task_two").task_id == 8


def test_auto_preparation_fails_clearly_when_source_paraphrases_are_absent(
    tmp_path,
):
    _write_paraphrase_preparation_fixture(tmp_path, include_annotations=False)

    with pytest.raises(HiddenBenchDataError, match="No paraphrase annotations exist"):
        load_task_set(
            "expanded",
            corpus_root=tmp_path,
            scheme="paraphrased_replication",
            n_agents=6,
            requested_task="portable_task_one",
            auto_prepare_paraphrases=True,
        )


# --------------------------------------------------------------------------
# Disclosure detection
# --------------------------------------------------------------------------


def test_disclosure_detects_a_restated_fact_and_ignores_unrelated_text(tasks):
    task = tasks[0]
    fact = task.hidden_information[0]
    assert disclosed_facts(task.hidden_information, [fact])[0] is True
    assert not any(disclosed_facts(task.hidden_information, ["I think we should hurry."]))


def test_disclosure_is_case_and_whitespace_insensitive(tasks):
    task = tasks[0]
    noisy = f"  {task.hidden_information[1].upper()}  "
    assert disclosed_facts(task.hidden_information, [noisy])[1] is True


def test_disclosure_is_documented_as_a_lower_bound(tasks):
    """A heavy paraphrase is missed - the docstring says so, so does this."""

    task = tasks[0]
    paraphrase = "The route north is impassable now."
    assert not any(disclosed_facts(task.hidden_information, [paraphrase]))
