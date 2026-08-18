"""Build the items, prove they are sound, then (optionally) ask one model.

The run is split so that everything falsifiable happens before anything is paid
for.  :func:`build_benchmark` generates the datasets, expands the evidence
conditions, renders every prompt and runs the full validator; it makes no
network call and is what ``benchmark relational-support preflight`` executes.
:func:`run_benchmark` does that first, refuses to continue if any check failed or
if the plan exceeds ``limits.max_requests``, and only then opens a provider.

One item = one request.  There is no conversation, no retry-with-history and no
state carried between items, so the items are independent by construction and
the accuracy estimates are not autocorrelated through a shared context.
"""

from __future__ import annotations

import asyncio
import csv
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mas_cc.llm_runtime.messages import Message, MessageRole
from mas_cc.llm_runtime.providers import CompletionRequest, ProviderError, create_llm_provider

from .conditions import build_evidence_conditions
from .config import BenchmarkConfig, ParameterCondition
from .dataset import generate_condition_dataset, verify_dataset_reproducible
from .geometry import feasible_options
from .prompting import (
    PROMPT_FAMILY,
    PROMPT_VERSION,
    BenchmarkPrompt,
    parse_answer,
    presentation_order,
)
from .prompting import render_prompt
from .presentation import OptionPresentation, build_presentations
from .tasks import BenchmarkTask, load_benchmark_tasks
from .validation import (
    summarize_checks,
    summarize_diagnostics,
    task_diagnostics,
    validate_condition_prompts,
)

ROW_FIELDS = (
    "model",
    "provider",
    "parameter_condition",
    "task_id",
    "seed",
    "dataset_seed",
    "reasoning_depth",
    "distractors",
    "num_options",
    "population_size",
    "support_redundancy",
    "distractor_redundancy",
    "no_single_agent_solution",
    "condition",
    "condition_id",
    "num_supporting_facts_shown",
    "supporting_fact_ids_shown",
    "supporting_fact_ids_omitted",
    "num_facts_shown",
    "num_feasible_options",
    "feasible_options",
    "permutation_id",
    "correct_display_position",
    "displayed_option_order",
    "prediction",
    "predicted_relation",
    "correct_option",
    "stored_correct_option",
    "correct_relation",
    "correct",
    "parse_ok",
    "finish_reason",
    "input_tokens",
    "output_tokens",
    "latency_seconds",
    "error",
)


@dataclass(frozen=True, slots=True)
class BenchmarkItem:
    """One prompt plus everything the results row needs about its provenance."""

    condition: ParameterCondition
    task: BenchmarkTask
    prompt: BenchmarkPrompt
    feasible_options: tuple[str, ...] = ()

    @property
    def row_key(self) -> str:
        return (
            f"{self.condition.label}|{self.task.task_id}"
            f"|{self.prompt.condition.condition_id}|{self.prompt.presentation.permutation_id}"
        )

    def base_row(self) -> dict[str, Any]:
        evidence = self.prompt.condition
        presentation = self.prompt.presentation
        return {
            "parameter_condition": self.condition.label,
            "task_id": self.task.task_id,
            "seed": self.task.seed,
            "dataset_seed": self.condition.dataset_seed,
            "reasoning_depth": self.condition.reasoning_depth,
            "distractors": self.condition.distractors,
            "num_options": self.condition.num_options,
            "population_size": self.condition.population_size,
            "support_redundancy": self.condition.support_redundancy,
            "distractor_redundancy": self.condition.distractor_redundancy,
            "no_single_agent_solution": self.condition.no_single_agent_solution,
            "condition": evidence.condition,
            "condition_id": evidence.condition_id,
            "num_supporting_facts_shown": evidence.k,
            "supporting_fact_ids_shown": "+".join(evidence.shown_supporting_fact_ids) or "none",
            "supporting_fact_ids_omitted": "+".join(evidence.omitted_supporting_fact_ids) or "none",
            "num_facts_shown": len(self.prompt.shown_fact_ids),
            # How many displayed options survive elimination on the shown facts
            # alone. 1 means the item is answerable without the missing link,
            # which is a different question from a genuine 3-way choice.
            "num_feasible_options": len(self.feasible_options),
            "feasible_options": "|".join(self.feasible_options),
            "permutation_id": presentation.permutation_id,
            "correct_display_position": presentation.correct_display_position,
            "displayed_option_order": "|".join(presentation.displayed_order),
            # `correct_option` is the label the correct relation carries *in this
            # prompt*; `stored_correct_option` is the generator's frozen letter,
            # kept only for provenance. Scoring uses neither - it compares
            # relations.
            "correct_option": presentation.correct_display_position,
            "stored_correct_option": self.task.correct_option,
            "correct_relation": self.task.correct_relation,
        }


@dataclass(slots=True)
class BenchmarkPlan:
    """The complete, validated set of items - the thing a run executes."""

    config: BenchmarkConfig
    items: list[BenchmarkItem] = field(default_factory=list)
    manifests: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)
    reproducibility: dict[str, str] = field(default_factory=dict)

    @property
    def request_count(self) -> int:
        return len(self.items)

    @property
    def valid(self) -> bool:
        return bool(self.validation.get("all_passed"))


def build_benchmark(
    config: BenchmarkConfig,
    output_dir: Path,
    *,
    verify_reproducibility: bool = True,
) -> BenchmarkPlan:
    """Generate datasets, render every prompt, and validate the lot. No network."""

    output_dir = Path(output_dir)
    (output_dir / "datasets").mkdir(parents=True, exist_ok=True)
    plan = BenchmarkPlan(config=config)
    per_task_checks: dict[str, Any] = {}
    per_task_diagnostics: dict[str, Any] = {}

    for condition in config.conditions():
        dataset_dir = output_dir / "datasets" / condition.label
        plan.manifests[condition.label] = generate_condition_dataset(condition, dataset_dir)
        if verify_reproducibility:
            plan.reproducibility[condition.label] = verify_dataset_reproducible(dataset_dir)
        for task in load_benchmark_tasks(dataset_dir):
            evidence_conditions = build_evidence_conditions(
                task.supporting_fact_ids,
                task_seed=task.seed,
                include_zero=config.include_zero_condition,
                max_subsets_per_k=config.max_subsets_per_k,
            )
            order = presentation_order(task)
            presentations = build_presentations(
                task, seed=config.seed, mode=config.presentation_mode
            )
            prompts = [
                render_prompt(task, evidence, presentation, order=order)
                for evidence in evidence_conditions
                for presentation in presentations
            ]
            key = f"{condition.label}/{task.task_id}"
            per_task_checks[key] = validate_condition_prompts(
                task, prompts, raise_on_failure=False
            )
            per_task_diagnostics[key] = task_diagnostics(task, prompts)
            plan.items.extend(
                BenchmarkItem(
                    condition=condition,
                    task=task,
                    prompt=prompt,
                    feasible_options=feasible_options(
                        [task.fact(f) for f in prompt.shown_fact_ids],
                        [task.fact(f) for f in prompt.condition.omitted_supporting_fact_ids],
                        list(prompt.presentation.relation_by_label.values()),
                        task.question_subject,
                        task.question_reference,
                    ),
                )
                for prompt in prompts
            )

    plan.validation = summarize_checks(per_task_checks)
    plan.validation["diagnostics"] = summarize_diagnostics(per_task_diagnostics)
    _write_artifacts(plan, output_dir)
    return plan


def _write_artifacts(plan: BenchmarkPlan, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "validation_report.json").write_text(
        json.dumps(plan.validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "plan.json").write_text(
        json.dumps(
            {
                "config": plan.config.to_dict(),
                "parameter_conditions": [c.to_dict() for c in plan.config.conditions()],
                "request_count": plan.request_count,
                "prompt_family": PROMPT_FAMILY,
                "prompt_version": PROMPT_VERSION,
                "dataset_fingerprints": {
                    label: manifest.get("dataset_fingerprint_sha256")
                    for label, manifest in plan.manifests.items()
                },
                "reproducibility_check": plan.reproducibility,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    examples_dir = output_dir / "prompts"
    examples_dir.mkdir(parents=True, exist_ok=True)
    # Complete condition sets for the first few tasks of each grid cell, so that
    # full / partial / zero for the *same* task can be diffed line by line - the
    # only reading that shows the manipulation is a deletion and nothing else.
    quota = {label: plan.config.prompt_example_tasks for label in plan.manifests}
    kept: dict[str, set[str]] = {label: set() for label in plan.manifests}
    for item in plan.items:
        label, task_id = item.condition.label, item.task.task_id
        chosen = kept.setdefault(label, set())
        if task_id not in chosen:
            if len(chosen) >= quota.get(label, 1):
                continue
            chosen.add(task_id)
        name = (
            f"{label}__{task_id}__{item.prompt.condition.condition_id}"
            f"__{item.prompt.presentation.permutation_id}.md"
        )
        (examples_dir / name).write_text(item.prompt.to_markdown(), encoding="utf-8")


async def _complete_item(
    provider: Any,
    item: BenchmarkItem,
    config: BenchmarkConfig,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    row = item.base_row()
    row.update(
        {
            "model": config.provider.model,
            "provider": config.provider.type,
            "prediction": "",
            "predicted_relation": "",
            "correct": False,
            "parse_ok": False,
            "finish_reason": "",
            "input_tokens": None,
            "output_tokens": None,
            "latency_seconds": None,
            "error": "",
        }
    )
    request = CompletionRequest(
        messages=(
            Message(MessageRole.SYSTEM, item.prompt.system),
            Message(MessageRole.USER, item.prompt.user),
        ),
        temperature=config.provider.temperature,
        max_output_tokens=config.provider.max_output_tokens,
        seed=config.seed,
        metadata={
            "prompt_family": PROMPT_FAMILY,
            "prompt_version": PROMPT_VERSION,
            "benchmark_item": item.row_key,
        },
    )
    started = time.perf_counter()
    async with semaphore:
        try:
            response = await provider.complete(request)
        except ProviderError as exc:
            row["error"] = f"{exc.code}: {exc}"
            row["latency_seconds"] = round(time.perf_counter() - started, 3)
            return row
    prediction = parse_answer(response.content, item.task.option_labels)
    predicted_relation = item.prompt.presentation.relation_for(prediction)
    row.update(
        {
            "prediction": prediction or "",
            "predicted_relation": predicted_relation or "",
            "parse_ok": prediction is not None,
            # Correctness is semantic: the letter is a presentation accident.
            "correct": predicted_relation == item.task.correct_relation,
            "finish_reason": response.finish_reason or "",
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "latency_seconds": round(response.latency_seconds, 3),
        }
    )
    return row


async def run_benchmark_async(
    config: BenchmarkConfig,
    output_dir: Path,
    *,
    verify_reproducibility: bool = True,
    provider: Any | None = None,
    progress: bool = True,
) -> tuple[BenchmarkPlan, list[dict[str, Any]]]:
    """Validate, refuse on any failure, then execute every item exactly once."""

    output_dir = Path(output_dir)
    plan = build_benchmark(config, output_dir, verify_reproducibility=verify_reproducibility)
    if not plan.valid:
        failing = [c["check"] for c in plan.validation["checks"] if not c["passed"]]
        raise RuntimeError(
            "refusing to send prompts: pre-flight validation failed for "
            f"{failing}; see {output_dir / 'validation_report.json'}"
        )
    if plan.request_count > config.max_requests:
        raise RuntimeError(
            f"plan needs {plan.request_count} requests but limits.max_requests is "
            f"{config.max_requests}; raise the limit deliberately or shrink the grid"
        )

    owned = provider is None
    active = provider if provider is not None else create_llm_provider(config.provider)
    semaphore = asyncio.Semaphore(max(1, config.provider.request_concurrency))

    # A crash-safe journal. Rows land here the moment they return, in completion
    # order, so a run that dies at request 900 of 960 keeps the 900 already paid
    # for instead of discarding them. The ordered rows.jsonl and rows.csv are
    # rewritten from the in-memory results at the end.
    journal_path = output_dir / "rows.partial.jsonl"
    total = len(plan.items)
    started_at = time.perf_counter()
    state = {"done": 0, "correct": 0, "errors": 0}
    lock = asyncio.Lock()

    async def _tracked(item: BenchmarkItem) -> dict[str, Any]:
        row = await _complete_item(active, item, config, semaphore)
        async with lock:
            state["done"] += 1
            state["correct"] += bool(row.get("correct"))
            state["errors"] += bool(row.get("error"))
            with journal_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
            if progress:
                _emit_progress(state, total, started_at, row)
        return row

    journal_path.write_text("", encoding="utf-8")
    try:
        rows = await asyncio.gather(*(_tracked(item) for item in plan.items))
    finally:
        if owned:
            active.close()
    ordered = list(rows)
    _write_rows(ordered, output_dir)
    journal_path.unlink(missing_ok=True)
    return plan, ordered


def _emit_progress(
    state: dict[str, int], total: int, started_at: float, row: dict[str, Any]
) -> None:
    """One line per completed item on stderr: enough to watch, cheap to log."""

    done = state["done"]
    elapsed = time.perf_counter() - started_at
    rate = done / elapsed if elapsed > 0 else 0.0
    remaining = (total - done) / rate if rate > 0 else 0.0
    accuracy = state["correct"] / done if done else 0.0
    outcome = (
        "ERR"
        if row.get("error")
        else ("ok " if row.get("correct") else "MISS")
    )
    print(
        f"[{done:>4}/{total}] {done / total:5.1%} "
        f"acc={accuracy:.3f} err={state['errors']} "
        f"eta={remaining / 60:5.1f}m  "
        f"{outcome} {row.get('parameter_condition')} {row.get('task_id')} "
        f"{row.get('condition_id')}/{row.get('permutation_id')} "
        f"-> {row.get('prediction') or '?'}={row.get('predicted_relation') or '?'}",
        file=sys.stderr,
        flush=True,
    )


def run_benchmark(
    config: BenchmarkConfig,
    output_dir: Path,
    *,
    verify_reproducibility: bool = True,
    provider: Any | None = None,
    progress: bool = True,
) -> tuple[BenchmarkPlan, list[dict[str, Any]]]:
    return asyncio.run(
        run_benchmark_async(
            config,
            output_dir,
            verify_reproducibility=verify_reproducibility,
            provider=provider,
            progress=progress,
        )
    )


def _write_rows(rows: Sequence[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({key: row.get(key) for key in ROW_FIELDS}, sort_keys=True) + "\n")
    with (output_dir / "rows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ROW_FIELDS), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in ROW_FIELDS})
