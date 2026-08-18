"""The benchmark's own guarantees, checked the way the benchmark checks tasks.

The point of this suite is that a *silently wrong* benchmark is worse than a
broken one: it would report a full-versus-partial gap that measures prompt
construction rather than reasoning.  So the tests below concentrate on the two
places where that could happen - the independent geometry, and the leak checks -
and deliberately inject a leak to confirm the validator refuses it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from mas_cc.benchmarks.relational_support import (
    build_evidence_conditions,
    load_benchmark_config,
    parse_answer,
    render_prompt,
    validate_condition_prompts,
)
from mas_cc.benchmarks.relational_support.analysis import (
    headline_l2,
    summarize,
    wilson_interval,
    write_summary,
)
from mas_cc.benchmarks.relational_support.conditions import FULL, PARTIAL, ZERO
from mas_cc.benchmarks.relational_support.geometry import determined_relation
from mas_cc.benchmarks.relational_support.prompting import presentation_order
from mas_cc.benchmarks.relational_support.presentation import (
    ALL_POSITIONS,
    FROZEN,
    build_presentations,
    position_balance,
)
from mas_cc.benchmarks.relational_support.runner import build_benchmark, run_benchmark
from mas_cc.benchmarks.relational_support.tasks import load_benchmark_task, load_benchmark_tasks
from mas_cc.benchmarks.relational_support.validation import PromptValidationError
from mas_cc.games.relational_reasoning.data import DEFAULT_TASK_DATASET_DIR, list_relational_task_ids

pytestmark = pytest.mark.skipif(
    not (DEFAULT_TASK_DATASET_DIR / "task_0001.json").exists(),
    reason="the relational example dataset is not present",
)

MOCK_CONFIG = Path("configs/benchmarks/relational_support/L2_mock_preflight.yaml")


def _example_tasks():
    return load_benchmark_tasks(DEFAULT_TASK_DATASET_DIR)


# ---------------------------------------------------------------------------
# The independent geometry must agree with the generator on every frozen task.
# ---------------------------------------------------------------------------


def test_geometry_reproduces_the_stored_answer_of_every_example_task():
    tasks = _example_tasks()
    assert tasks, "expected the checked-in example dataset to contain tasks"
    for task in tasks:
        supporting = [task.fact(fact_id) for fact_id in task.supporting_fact_ids]
        implied = determined_relation(
            supporting, task.question_subject, task.question_reference
        )
        assert implied == task.correct_relation, task.task_id


def test_distractors_alone_never_determine_the_answer():
    for task in _example_tasks():
        distractors = [task.fact(fact_id) for fact_id in task.distractor_fact_ids]
        assert (
            determined_relation(distractors, task.question_subject, task.question_reference)
            is None
        ), task.task_id


def test_every_strict_subset_of_the_support_leaves_the_query_undetermined():
    for task in _example_tasks():
        for condition in build_evidence_conditions(
            task.supporting_fact_ids, task_seed=task.seed
        ):
            shown = [task.fact(fact_id) for fact_id in condition.shown_supporting_fact_ids]
            shown += [task.fact(fact_id) for fact_id in task.distractor_fact_ids]
            implied = determined_relation(
                shown, task.question_subject, task.question_reference
            )
            if condition.condition == FULL:
                assert implied == task.correct_relation
            else:
                assert implied is None, f"{task.task_id}/{condition.condition_id}"


# ---------------------------------------------------------------------------
# Evidence conditions
# ---------------------------------------------------------------------------


def test_l2_yields_zero_two_singletons_and_full():
    conditions = build_evidence_conditions(("f1", "f2"), task_seed=1)
    assert [c.condition_id for c in conditions] == ["k0_none", "k1_f1", "k1_f2", "k2_f1+f2"]
    assert [c.condition for c in conditions] == [ZERO, PARTIAL, PARTIAL, FULL]
    assert conditions[1].omitted_supporting_fact_ids == ("f2",)


def test_intermediate_k_is_capped_deterministically_not_prefixed():
    support = ("f1", "f2", "f3", "f4")
    first = build_evidence_conditions(support, task_seed=7, max_subsets_per_k=2)
    second = build_evidence_conditions(support, task_seed=7, max_subsets_per_k=2)
    assert [c.condition_id for c in first] == [c.condition_id for c in second]
    for k in (1, 2, 3):
        assert len([c for c in first if c.k == k]) == 2
    # Not always the same prefix: at least one sampled subset omits f1.
    assert any("f1" not in c.shown_supporting_fact_ids for c in first if c.k == 2)


def test_zero_condition_can_be_switched_off():
    conditions = build_evidence_conditions(("f1", "f2"), task_seed=1, include_zero=False)
    assert all(c.k > 0 for c in conditions)


# ---------------------------------------------------------------------------
# Prompts: conditions differ by deletion and by nothing else
# ---------------------------------------------------------------------------


def _prompts_for(task, mode=ALL_POSITIONS):
    order = presentation_order(task)
    conditions = build_evidence_conditions(task.supporting_fact_ids, task_seed=task.seed)
    presentations = build_presentations(task, seed=11, mode=mode)
    return [
        render_prompt(task, condition, presentation, order=order)
        for condition in conditions
        for presentation in presentations
    ]


def test_conditions_differ_only_by_deleted_fact_lines():
    task = load_benchmark_task(DEFAULT_TASK_DATASET_DIR, "task_0001")
    prompts = _prompts_for(task)
    full = next(p for p in prompts if p.condition.condition == FULL)
    full_lines = full.facts_block.splitlines()
    for prompt in prompts:
        assert prompt.question == full.question
        lines = prompt.facts_block.splitlines()
        assert lines == [line for line in full_lines if line in lines]
        assert set(lines) <= set(full_lines)
        if prompt.presentation.permutation_id == full.presentation.permutation_id:
            assert prompt.options_block == full.options_block


def test_prompts_never_expose_fact_identifiers_or_roles():
    task = load_benchmark_task(DEFAULT_TASK_DATASET_DIR, "task_0001")
    for prompt in _prompts_for(task):
        blob = f"{prompt.system}\n{prompt.user}"
        for fact_id in task.fact_order:
            assert f"{fact_id}:" not in blob
            assert f"[{fact_id}]" not in blob
        assert "supporting" not in blob.lower().replace("supported", "")
        assert "distractor" not in blob.lower()


def test_all_example_tasks_pass_validation():
    for task in _example_tasks():
        validate_condition_prompts(task, _prompts_for(task))


# ---------------------------------------------------------------------------
# The validator has to actually catch a leak
# ---------------------------------------------------------------------------


def test_validator_rejects_a_partial_prompt_that_smuggles_the_omitted_fact():
    task = load_benchmark_task(DEFAULT_TASK_DATASET_DIR, "task_0001")
    prompts = _prompts_for(task)
    partial = next(p for p in prompts if p.condition.condition == PARTIAL)
    omitted = task.fact(partial.condition.omitted_supporting_fact_ids[0])
    leaked = type(partial)(
        system=partial.system,
        user=partial.user.replace("QUESTION", f"- {omitted.text}\n\nQUESTION"),
        task_id=partial.task_id,
        condition=partial.condition,
        presentation=partial.presentation,
        shown_fact_ids=partial.shown_fact_ids,
        facts_block=partial.facts_block,
        options_block=partial.options_block,
        question=partial.question,
    )
    replaced = [leaked if p is partial else p for p in prompts]
    with pytest.raises(PromptValidationError, match="omitted_support_text_absent"):
        validate_condition_prompts(task, replaced)


def test_validator_rejects_a_partial_prompt_that_still_determines_the_answer():
    task = load_benchmark_task(DEFAULT_TASK_DATASET_DIR, "task_0001")
    order = presentation_order(task)
    conditions = build_evidence_conditions(task.supporting_fact_ids, task_seed=task.seed)
    partial = next(c for c in conditions if c.condition == PARTIAL)
    # Claim k = 1 while rendering the complete chain: the symbolic solver sees a
    # determined query in a condition that declares itself partial.
    mislabelled = type(partial)(
        condition_id=partial.condition_id,
        condition=PARTIAL,
        k=1,
        shown_supporting_fact_ids=task.supporting_fact_ids,
        omitted_supporting_fact_ids=(),
    )
    presentations = build_presentations(task, seed=11)
    prompts = [
        render_prompt(task, c, presentation, order=order)
        for c in conditions
        if c is not partial
        for presentation in presentations
    ]
    prompts.extend(
        render_prompt(task, mislabelled, presentation, order=order)
        for presentation in presentations
    )
    with pytest.raises(PromptValidationError, match="partial_conditions_underdetermined"):
        validate_condition_prompts(task, prompts)


def test_validator_rejects_reordered_options():
    task = load_benchmark_task(DEFAULT_TASK_DATASET_DIR, "task_0001")
    prompts = _prompts_for(task)
    first = prompts[0]
    scrambled = type(first)(
        system=first.system,
        user=first.user,
        task_id=first.task_id,
        condition=first.condition,
        presentation=first.presentation,
        shown_fact_ids=first.shown_fact_ids,
        facts_block=first.facts_block,
        options_block="\n".join(reversed(first.options_block.splitlines())),
        question=first.question,
    )
    with pytest.raises(PromptValidationError, match="options_block_identical"):
        validate_condition_prompts(task, [scrambled, *prompts[1:]])


# ---------------------------------------------------------------------------
# Answer parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("ANSWER: B", "B"),
        ("Reasoning...\n\nANSWER: c", "C"),
        ("I first said ANSWER: A but on reflection\nANSWER: B", "B"),
        ("**ANSWER:** C", "C"),
        ("answer - a", "A"),
        ("B", "B"),
        ("It could be anything.", None),
        ("", None),
        ("ANSWER: Z", None),
    ],
)
def test_parse_answer(content, expected):
    assert parse_answer(content, ("A", "B", "C")) == expected


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def test_wilson_interval_brackets_the_point_estimate():
    low, high = wilson_interval(20, 20)
    assert low > 0.8 and high == 1.0
    low, high = wilson_interval(0, 20)
    assert low == 0.0 and high < 0.2
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_headline_reports_single_fact_conditions_before_pooling():
    rows = [
        {"parameter_condition": "L2_D4_O3", "reasoning_depth": 2, "distractors": 4,
         "num_options": 3, "condition": "full", "supporting_fact_ids_shown": "f1+f2",
         "correct": True, "parse_ok": True, "error": ""},
        {"parameter_condition": "L2_D4_O3", "reasoning_depth": 2, "distractors": 4,
         "num_options": 3, "condition": "partial", "supporting_fact_ids_shown": "f1",
         "correct": True, "parse_ok": True, "error": ""},
        {"parameter_condition": "L2_D4_O3", "reasoning_depth": 2, "distractors": 4,
         "num_options": 3, "condition": "partial", "supporting_fact_ids_shown": "f2",
         "correct": False, "parse_ok": True, "error": ""},
        {"parameter_condition": "L2_D4_O3", "reasoning_depth": 2, "distractors": 4,
         "num_options": 3, "condition": "zero", "supporting_fact_ids_shown": "none",
         "correct": False, "parse_ok": True, "error": ""},
    ]
    entry = headline_l2(rows)[0]
    assert entry["accuracy_full"] == 1.0
    assert entry["accuracy_partial"] == 0.5
    assert entry["accuracy_zero"] == 0.0
    assert entry["full_minus_partial"] == 0.5
    assert entry["accuracy_per_single_fact"]["f1"]["accuracy"] == 1.0
    assert entry["accuracy_per_single_fact"]["f2"]["accuracy"] == 0.0


def test_provider_errors_are_excluded_rather_than_scored_as_wrong():
    rows = [
        {"parameter_condition": "c", "reasoning_depth": 2, "distractors": 0, "num_options": 3,
         "condition": "full", "supporting_fact_ids_shown": "f1+f2", "correct": True,
         "parse_ok": True, "error": ""},
        {"parameter_condition": "c", "reasoning_depth": 2, "distractors": 0, "num_options": 3,
         "condition": "full", "supporting_fact_ids_shown": "f1+f2", "correct": False,
         "parse_ok": False, "error": "timeout: upstream"},
    ]
    report = summarize(rows)
    assert report["rows_with_provider_error"] == 1
    assert report["accuracy_by_k"][0]["num_tasks"] == 1
    assert report["accuracy_by_k"][0]["accuracy"] == 1.0


# ---------------------------------------------------------------------------
# End to end, on the mock provider
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not MOCK_CONFIG.exists(), reason="mock benchmark config is not present")
def test_mock_run_produces_validated_rows_and_summary(tmp_path):
    config = load_benchmark_config(MOCK_CONFIG)
    destination = tmp_path / "run"
    plan, rows = run_benchmark(config, destination, verify_reproducibility=False)

    assert plan.valid
    assert len(rows) == plan.request_count == 72
    assert all(row["error"] == "" for row in rows)
    assert all(row["parse_ok"] for row in rows)
    # The mock always answers "A"; a row is correct exactly when A is the answer,
    # which is the strongest available check that scoring is not self-fulfilling.
    assert all(row["correct"] == (row["correct_option"] == "A") for row in rows)

    written = [json.loads(line) for line in (destination / "rows.jsonl").read_text().splitlines()]
    assert len(written) == 72
    assert (destination / "validation_report.json").exists()
    assert json.loads((destination / "validation_report.json").read_text())["all_passed"]

    summary_dir = write_summary(destination)
    assert (summary_dir / "summary.md").exists()
    assert (summary_dir / "headline_l2.csv").exists()


@pytest.mark.skipif(not MOCK_CONFIG.exists(), reason="mock benchmark config is not present")
def test_datasets_are_regenerated_byte_identically_from_their_seeds(tmp_path):
    config = load_benchmark_config(MOCK_CONFIG)
    plan = build_benchmark(config, tmp_path / "preflight", verify_reproducibility=True)
    assert plan.reproducibility
    assert all("VALID" in output for output in plan.reproducibility.values())
    for label, manifest in plan.manifests.items():
        assert manifest["dataset_fingerprint_sha256"]
        dataset_dir = tmp_path / "preflight" / "datasets" / label
        assert len(list_relational_task_ids(dataset_dir)) == config.tasks_per_condition


def test_config_rejects_unknown_keys_and_out_of_range_parameters(tmp_path):
    base = yaml.safe_load(MOCK_CONFIG.read_text())

    stray = tmp_path / "stray.yaml"
    stray.write_text(yaml.safe_dump({**base, "control": {"mechanism": "none"}}))
    with pytest.raises(ValueError, match="unknown top-level keys"):
        load_benchmark_config(stray)

    deep = tmp_path / "deep.yaml"
    deep.write_text(yaml.safe_dump({**base, "grid": {**base["grid"], "reasoning_depth": [5]}}))
    with pytest.raises(ValueError, match="outside the generator's supported 1..4"):
        load_benchmark_config(deep)


def test_l1_drops_the_impossible_no_single_agent_requirement(tmp_path):
    base = yaml.safe_load(MOCK_CONFIG.read_text())
    base["grid"]["reasoning_depth"] = [1, 2]
    path = tmp_path / "l1.yaml"
    path.write_text(yaml.safe_dump(base))
    conditions = load_benchmark_config(path).conditions()
    by_depth = {c.reasoning_depth: c for c in conditions}
    assert by_depth[1].no_single_agent_solution is False
    assert by_depth[2].no_single_agent_solution is True


# ---------------------------------------------------------------------------
# Option-position control
# ---------------------------------------------------------------------------


def test_all_positions_places_the_correct_relation_at_every_label_exactly_once():
    for task in _example_tasks():
        presentations = build_presentations(task, seed=11, mode=ALL_POSITIONS)
        assert len(presentations) == len(task.option_labels)
        assert position_balance(presentations) == {label: 1 for label in task.option_labels}
        for presentation in presentations:
            position = presentation.correct_display_position
            assert presentation.relation_by_label[position] == task.correct_relation


def test_permutations_preserve_the_relation_set_and_never_duplicate_one():
    for task in _example_tasks():
        expected = frozenset(task.option_relations.values())
        for presentation in build_presentations(task, seed=11, mode=ALL_POSITIONS):
            shown = list(presentation.relation_by_label.values())
            assert frozenset(shown) == expected
            assert len(set(shown)) == len(shown)


def test_frozen_mode_reproduces_the_generators_own_assignment():
    task = load_benchmark_task(DEFAULT_TASK_DATASET_DIR, "task_0001")
    (presentation,) = build_presentations(task, seed=11, mode=FROZEN)
    assert presentation.correct_display_position == task.correct_option
    assert dict(presentation.relation_by_label) == dict(task.option_relations)


def test_presentations_are_deterministic_in_the_seed():
    task = load_benchmark_task(DEFAULT_TASK_DATASET_DIR, "task_0001")
    first = build_presentations(task, seed=11, mode=ALL_POSITIONS)
    again = build_presentations(task, seed=11, mode=ALL_POSITIONS)
    other = build_presentations(task, seed=12, mode=ALL_POSITIONS)
    assert [p.displayed_order for p in first] == [p.displayed_order for p in again]
    assert [p.correct_display_position for p in first] == [
        p.correct_display_position for p in other
    ]


def test_facts_and_question_are_byte_identical_across_permutations():
    task = load_benchmark_task(DEFAULT_TASK_DATASET_DIR, "task_0001")
    prompts = _prompts_for(task)
    by_condition = {}
    for prompt in prompts:
        by_condition.setdefault(prompt.condition.condition_id, []).append(prompt)
    for group in by_condition.values():
        assert len({p.facts_block for p in group}) == 1
        assert len({p.question for p in group}) == 1
        # ...while the options are exactly what does move.
        assert len({p.options_block for p in group}) == len(group)


def test_validator_rejects_an_unbalanced_position_design():
    task = load_benchmark_task(DEFAULT_TASK_DATASET_DIR, "task_0001")
    prompts = [p for p in _prompts_for(task) if p.presentation.correct_display_position != "C"]
    with pytest.raises(PromptValidationError, match="correct_position_balanced_across_labels"):
        validate_condition_prompts(task, prompts)


def test_validator_rejects_a_presentation_whose_declared_position_is_wrong():
    task = load_benchmark_task(DEFAULT_TASK_DATASET_DIR, "task_0001")
    prompts = _prompts_for(task)
    first = prompts[0]
    lying = type(first.presentation)(
        permutation_id=first.presentation.permutation_id,
        correct_display_position=next(
            label for label in task.option_labels
            if label != first.presentation.correct_display_position
        ),
        labels=first.presentation.labels,
        relation_by_label=first.presentation.relation_by_label,
        correct_relation=task.correct_relation,
    )
    replaced = type(first)(
        system=first.system,
        user=first.user,
        task_id=first.task_id,
        condition=first.condition,
        presentation=lying,
        shown_fact_ids=first.shown_fact_ids,
        facts_block=first.facts_block,
        options_block=first.options_block,
        question=first.question,
    )
    with pytest.raises(PromptValidationError, match="correct_relation_sits_at_its_declared"):
        validate_condition_prompts(task, [replaced, *prompts[1:]])


def test_scoring_is_semantic_not_positional():
    """A model answering the same letter every time must not score above chance."""

    task = load_benchmark_task(DEFAULT_TASK_DATASET_DIR, "task_0001")
    prompts = _prompts_for(task)
    full = [p for p in prompts if p.condition.condition == FULL]
    correct_when_always_a = [
        p.presentation.relation_by_label["A"] == task.correct_relation for p in full
    ]
    # Exactly one of the three placements puts the answer at A.
    assert sum(correct_when_always_a) == 1
    assert len(correct_when_always_a) == 3


def test_position_tables_split_accuracy_and_expose_a_letter_habit():
    from mas_cc.benchmarks.relational_support.analysis import (
        accuracy_by_correct_position,
        predicted_position_distribution,
    )

    rows = []
    for position in ("A", "B", "C"):
        for index in range(4):
            rows.append(
                {
                    "parameter_condition": "L2_D4_O3",
                    "reasoning_depth": 2,
                    "distractors": 4,
                    "num_options": 3,
                    "condition": "zero",
                    "num_supporting_facts_shown": 0,
                    "correct_display_position": position,
                    # An "always A" model: correct only where A is the answer.
                    "prediction": "A",
                    "predicted_relation": "NORTH",
                    "correct": position == "A",
                    "parse_ok": True,
                    "error": "",
                }
            )
    by_position = {
        entry["correct_display_position"]: entry["accuracy"]
        for entry in accuracy_by_correct_position(rows)
    }
    assert by_position == {"A": 1.0, "B": 0.0, "C": 0.0}
    distribution = predicted_position_distribution(rows)
    assert distribution[0]["prediction"] == "A"
    assert distribution[0]["share"] == 1.0


# ---------------------------------------------------------------------------
# Option feasibility: reported, never enforced
# ---------------------------------------------------------------------------


def test_full_evidence_always_leaves_exactly_the_correct_option_feasible():
    from mas_cc.benchmarks.relational_support.geometry import feasible_options

    for task in _example_tasks():
        for prompt in _prompts_for(task):
            if prompt.condition.condition != FULL:
                continue
            surviving = feasible_options(
                [task.fact(f) for f in prompt.shown_fact_ids],
                [],
                list(prompt.presentation.relation_by_label.values()),
                task.question_subject,
                task.question_reference,
            )
            assert surviving == (task.correct_relation,)


def test_partial_feasible_set_always_contains_the_truth_and_never_exceeds_the_options():
    from mas_cc.benchmarks.relational_support.geometry import feasible_options

    for task in _example_tasks():
        for prompt in _prompts_for(task):
            if prompt.condition.condition != PARTIAL:
                continue
            options = list(prompt.presentation.relation_by_label.values())
            surviving = feasible_options(
                [task.fact(f) for f in prompt.shown_fact_ids],
                [task.fact(f) for f in prompt.condition.omitted_supporting_fact_ids],
                options,
                task.question_subject,
                task.question_reference,
            )
            # The true answer is reachable by definition, and elimination can
            # only ever shrink the displayed menu.
            assert task.correct_relation in surviving
            assert set(surviving) <= set(options)
            assert 1 <= len(surviving) <= len(options)


def test_a_task_answerable_by_elimination_is_diagnosed_not_rejected():
    """The property is real and common; aborting the run over it would be wrong."""

    from mas_cc.benchmarks.relational_support.validation import (
        summarize_diagnostics,
        task_diagnostics,
    )

    per_task = {}
    for task in _example_tasks():
        validate_condition_prompts(task, _prompts_for(task))  # must not raise
        per_task[task.task_id] = task_diagnostics(task, _prompts_for(task))
    pooled = summarize_diagnostics(per_task)
    assert pooled["partial_prompts"] > 0
    assert 0.0 <= pooled["partial_share_answerable_by_elimination"] <= 1.0
    # The eliminator ceiling is a real chance floor and must beat naive guessing.
    assert pooled["partial_perfect_eliminator_accuracy"] >= 1 / 3
