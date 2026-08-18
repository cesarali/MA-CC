"""Everything that must be true of a prompt *before* a model is charged for it.

The benchmark's conclusion is a comparison between conditions, so its validity
rests entirely on the conditions differing in the intended way and in no other
way.  These checks establish that mechanically, per task, on the exact strings
that will be sent.  They run in ``preflight`` with no provider at all, and again
at the start of every real run; a failure aborts before the first request.

The load-bearing one is ``partial_conditions_underdetermined``.  It is not a
string search - it re-solves the shown constraints with the independent geometry
in :mod:`.geometry` and asserts the query endpoints are not connected.  A
partial condition that *did* determine the answer would make ``A_k < A_L`` a
measurement of nothing, and no amount of checking for leaked text would catch a
leak that arrived through a distractor's geometry rather than through its words.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .conditions import EvidenceCondition
from .conditions import PARTIAL
from .geometry import determined_relation, feasible_options
from .prompting import BenchmarkPrompt

FORBIDDEN_MARKERS = (
    "correct_option",
    "correct_relation",
    "reasoning_chain",
    "supporting_fact_ids",
    "coordinate_convention",
    "schema_version",
    "distractor",
)
"""Field names from the frozen task JSON.  If any of these reaches a prompt, a
serialiser leaked the record instead of rendering it."""


class PromptValidationError(AssertionError):
    """A constructed prompt set is not fit to be sent to a model."""


@dataclass(frozen=True, slots=True)
class CheckResult:
    name: str
    passed: bool
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"check": self.name, "passed": self.passed, "detail": self.detail}


def validate_condition_prompts(
    task: Any,
    prompts: Sequence[BenchmarkPrompt],
    *,
    raise_on_failure: bool = True,
) -> tuple[CheckResult, ...]:
    """Run every pre-flight check over one task's complete condition set."""

    results: list[CheckResult] = []
    support = set(task.supporting_fact_ids)
    distractors = set(task.distractor_fact_ids)
    depth = len(task.supporting_fact_ids)

    results.append(
        CheckResult(
            "reasoning_depth_matches_support_count",
            task.reasoning_depth == depth,
            f"reasoning_depth={task.reasoning_depth} support={depth}",
        )
    )

    # --- the evidence filter is exactly the manipulation, and only that ------
    filter_ok, filter_detail = True, ""
    for prompt in prompts:
        shown_support = tuple(
            fact_id for fact_id in prompt.shown_fact_ids if fact_id in support
        )
        if set(shown_support) != set(prompt.condition.shown_supporting_fact_ids):
            filter_ok = False
            filter_detail = (
                f"{prompt.condition.condition_id}: rendered {shown_support} "
                f"but condition declares {prompt.condition.shown_supporting_fact_ids}"
            )
            break
    results.append(CheckResult("shown_support_matches_condition", filter_ok, filter_detail))

    # One full prompt per option permutation, each showing the whole chain.
    full = [p for p in prompts if p.condition.k == depth]
    permutation_ids = {p.presentation.permutation_id for p in prompts}
    results.append(
        CheckResult(
            "full_condition_shows_every_supporting_fact",
            len(full) == len(permutation_ids)
            and all(
                set(prompt.condition.shown_supporting_fact_ids) == support
                and len([f for f in prompt.shown_fact_ids if f in support]) == depth
                for prompt in full
            ),
            f"{len(full)} full prompt(s) over {len(permutation_ids)} permutation(s)",
        )
    )

    distractor_sets = {
        tuple(f for f in prompt.shown_fact_ids if f in distractors) for prompt in prompts
    }
    results.append(
        CheckResult(
            "distractor_condition_constant_across_conditions",
            len(distractor_sets) == 1
            and (not distractors or set(next(iter(distractor_sets))) == distractors),
            f"{len(distractor_sets)} distinct distractor set(s), "
            f"{len(distractors)} distractor(s) in task",
        )
    )

    # --- an omitted supporting fact must not arrive by another route ---------
    text_ok, text_detail = True, ""
    pair_ok, pair_detail = True, ""
    for prompt in prompts:
        blob = f"{prompt.system}\n{prompt.user}"
        for omitted in prompt.condition.omitted_supporting_fact_ids:
            fact = task.fact(omitted)
            if fact.text in blob:
                text_ok, text_detail = False, f"{prompt.condition.condition_id}: {fact.text!r}"
                break
            endpoints = {fact.subject, fact.object}
            for shown in prompt.shown_fact_ids:
                other = task.fact(shown)
                if {other.subject, other.object} == endpoints:
                    pair_ok = False
                    pair_detail = (
                        f"{prompt.condition.condition_id}: shown fact {shown} relates "
                        f"the same pair as omitted {omitted}"
                    )
                    break
            if not pair_ok:
                break
        if not (text_ok and pair_ok):
            break
    results.append(CheckResult("omitted_support_text_absent", text_ok, text_detail))
    results.append(CheckResult("omitted_support_pair_not_re_derivable", pair_ok, pair_detail))

    # --- the symbolic claim the whole benchmark rests on ---------------------
    partial_ok, partial_detail = True, ""
    full_ok, full_detail = True, ""
    for prompt in prompts:
        shown_facts = [task.fact(fact_id) for fact_id in prompt.shown_fact_ids]
        implied = determined_relation(shown_facts, task.question_subject, task.question_reference)
        if prompt.condition.k == depth:
            if implied != task.correct_relation:
                full_ok = False
                full_detail = f"full condition implies {implied!r}, stored {task.correct_relation!r}"
        elif implied is not None:
            partial_ok = False
            partial_detail = (
                f"{prompt.condition.condition_id} already determines {implied!r}"
            )
    results.append(
        CheckResult("full_condition_implies_stored_answer", full_ok, full_detail)
    )
    results.append(
        CheckResult("partial_conditions_underdetermined", partial_ok, partial_detail)
    )

    # --- question and options are held fixed across matched conditions -------
    #
    # "Matched" now means *within one option permutation*.  Across permutations
    # the option block is supposed to differ - that is the manipulation - while
    # the facts and the question must not move by a single byte.
    by_permutation: dict[str, list[BenchmarkPrompt]] = {}
    for prompt in prompts:
        by_permutation.setdefault(prompt.presentation.permutation_id, []).append(prompt)

    options_ok = all(
        len({prompt.options_block for prompt in group}) == 1
        for group in by_permutation.values()
    )
    results.append(
        CheckResult(
            "options_block_identical_across_conditions_within_permutation",
            options_ok,
            f"{len(by_permutation)} permutation(s)",
        )
    )
    results.append(
        CheckResult(
            "question_identical_across_conditions",
            len({prompt.question for prompt in prompts}) == 1,
            "",
        )
    )

    # --- the permutation moves the options and nothing else ------------------
    by_condition: dict[str, list[BenchmarkPrompt]] = {}
    for prompt in prompts:
        by_condition.setdefault(prompt.condition.condition_id, []).append(prompt)

    facts_ok, facts_detail = True, ""
    for condition_id, group in by_condition.items():
        if len({prompt.facts_block for prompt in group}) != 1:
            facts_ok, facts_detail = False, condition_id
            break
    results.append(
        CheckResult("facts_block_byte_identical_across_permutations", facts_ok, facts_detail)
    )

    relation_sets = {
        frozenset(prompt.presentation.relation_by_label.values()) for prompt in prompts
    }
    results.append(
        CheckResult(
            "option_relation_set_invariant_across_permutations",
            len(relation_sets) == 1
            and next(iter(relation_sets)) == frozenset(task.option_relations.values()),
            f"{len(relation_sets)} distinct relation set(s)",
        )
    )

    placement_ok, placement_detail = True, ""
    for prompt in prompts:
        presentation = prompt.presentation
        displayed = presentation.relation_by_label.get(presentation.correct_display_position)
        if displayed != task.correct_relation:
            placement_ok = False
            placement_detail = (
                f"{presentation.permutation_id}: label "
                f"{presentation.correct_display_position} carries {displayed!r}"
            )
            break
        if f"{presentation.correct_display_position}) {task.correct_relation}" not in prompt.options_block:
            placement_ok = False
            placement_detail = f"{presentation.permutation_id}: option line does not match"
            break
    results.append(
        CheckResult("correct_relation_sits_at_its_declared_position", placement_ok, placement_detail)
    )

    if len(by_permutation) > 1:
        counts: dict[str, int] = {}
        for prompt in prompts:
            key = prompt.presentation.correct_display_position
            counts[key] = counts.get(key, 0) + 1
        balanced = set(counts) == set(task.option_labels) and len(set(counts.values())) == 1
        results.append(
            CheckResult(
                "correct_position_balanced_across_labels",
                balanced,
                f"placement counts {dict(sorted(counts.items()))}",
            )
        )

    # Each label must carry a different relation within a prompt, or two options
    # would be indistinguishable and the item unanswerable.
    distinct_ok = all(
        len(set(prompt.presentation.relation_by_label.values()))
        == len(prompt.presentation.labels)
        for prompt in prompts
    )
    results.append(CheckResult("option_relations_distinct_within_prompt", distinct_ok, ""))

    # --- no ground truth, no record fields, no coordinates ------------------
    marker_ok, marker_detail = True, ""
    for prompt in prompts:
        blob = f"{prompt.system}\n{prompt.user}".lower()
        hit = next((m for m in FORBIDDEN_MARKERS if m in blob), None)
        if hit is not None:
            marker_ok, marker_detail = False, f"{prompt.condition.condition_id}: {hit!r}"
            break
    results.append(CheckResult("no_task_record_fields_in_prompt", marker_ok, marker_detail))

    answer_ok, answer_detail = True, ""
    for prompt in prompts:
        # The correct relation appears in CAPS exactly once - as its own option
        # line - and nowhere else; fact sentences render relations in lower case.
        # Matched with upper-case boundaries because NORTH is a prefix of both
        # NORTHEAST and NORTHWEST, and a plain substring count would read a
        # sibling option as a second occurrence of the answer.
        pattern = re.compile(rf"(?<![A-Z]){re.escape(task.correct_relation)}(?![A-Z])")
        occurrences = len(pattern.findall(prompt.user))
        in_options = len(pattern.findall(prompt.options_block))
        if occurrences != 1 or in_options != 1:
            answer_ok = False
            answer_detail = (
                f"{prompt.condition.condition_id}: correct relation appears "
                f"{occurrences}x in prompt, {in_options}x in options"
            )
            break
    results.append(CheckResult("correct_answer_only_present_as_an_option", answer_ok, answer_detail))

    coord_ok = all(not any(ch.isdigit() for ch in prompt.facts_block) for prompt in prompts)
    results.append(CheckResult("no_coordinates_in_facts_block", coord_ok, ""))

    structure_ok, structure_detail = True, ""
    for prompt in prompts:
        expected = [f"- {task.fact_text(fact_id)}" for fact_id in prompt.shown_fact_ids]
        actual = prompt.facts_block.splitlines()
        if expected and actual != expected:
            structure_ok, structure_detail = False, prompt.condition.condition_id
            break
    results.append(CheckResult("facts_block_is_exactly_the_shown_facts", structure_ok, structure_detail))

    failures = [result for result in results if not result.passed]
    if failures and raise_on_failure:
        rendered = "\n".join(f"  - {r.name}: {r.detail}" for r in failures)
        raise PromptValidationError(f"task {task.task_id} failed prompt validation:\n{rendered}")
    return tuple(results)


def summarize_checks(
    per_task: Mapping[str, Sequence[CheckResult]]
) -> dict[str, Any]:
    """Collapse per-task check results into one report object."""

    names: list[str] = []
    for results in per_task.values():
        for result in results:
            if result.name not in names:
                names.append(result.name)
    checks = []
    for name in names:
        failing = [
            task_id
            for task_id, results in per_task.items()
            if any(r.name == name and not r.passed for r in results)
        ]
        checks.append(
            {
                "check": name,
                "tasks_checked": len(per_task),
                "tasks_failed": len(failing),
                "failing_task_ids": failing[:10],
                "passed": not failing,
            }
        )
    return {
        "tasks_checked": len(per_task),
        "all_passed": all(entry["passed"] for entry in checks),
        "checks": checks,
    }


def task_diagnostics(task: Any, prompts: Sequence[BenchmarkPrompt]) -> dict[str, Any]:
    """Measured properties of a task that are reported, never enforced.

    How many displayed options survive elimination on the shown facts alone is a
    property of the frozen world and its option set.  A partial condition where
    only one option is reachable is answerable *without* the missing link - real,
    common, and not something this benchmark may repair by editing the world.
    Aborting a run over it would be wrong; leaving it out of the report would be
    worse, because it is what separates "one fact is uninformative" from "one
    fact plus the option list is nearly sufficient".
    """

    counts: dict[int, int] = {}
    for prompt in prompts:
        if prompt.condition.condition != PARTIAL:
            continue
        surviving = feasible_options(
            [task.fact(fact_id) for fact_id in prompt.shown_fact_ids],
            [task.fact(fact_id) for fact_id in prompt.condition.omitted_supporting_fact_ids],
            list(prompt.presentation.relation_by_label.values()),
            task.question_subject,
            task.question_reference,
        )
        counts[len(surviving)] = counts.get(len(surviving), 0) + 1
    return {"partial_feasible_option_counts": counts}


def summarize_diagnostics(per_task: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Pool per-task diagnostics into the numbers worth reading before a run."""

    totals: dict[int, int] = {}
    for diagnostics in per_task.values():
        for size, count in diagnostics.get("partial_feasible_option_counts", {}).items():
            totals[int(size)] = totals.get(int(size), 0) + count
    overall = sum(totals.values())
    # A perfect eliminator guesses uniformly among the options it cannot rule
    # out, which is the honest chance floor for a partial-evidence item.
    ceiling = (
        sum(count / size for size, count in totals.items() if size) / overall
        if overall
        else None
    )
    return {
        "partial_prompts": overall,
        "partial_feasible_option_counts": {str(k): totals[k] for k in sorted(totals)},
        "partial_share_answerable_by_elimination": (
            totals.get(1, 0) / overall if overall else None
        ),
        "partial_perfect_eliminator_accuracy": ceiling,
    }
