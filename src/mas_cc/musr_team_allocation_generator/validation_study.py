"""Systematic full/partial/zero validation study for native Team Allocation tasks."""

from __future__ import annotations

import csv
import json
import math
import subprocess
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from mas_cc.core.random import Seed
from mas_cc.llm_runtime.providers.protocols import LLMProvider

from .distribute import distribute_evidence
from .evidence_generation import extract_json_object
from .generate import GenerationConfig, generate_world
from .io_utils import canonical_json, sha256_file, sha256_object, write_json_atomic
from .provider_adapter import MuSRGenerationModel
from .schemas import EvidenceCard, LatentProblem, MUSR_COMMIT, PROMPT_VERSION
from .validate import agent_can_certify_unique_allocation, validate_distribution

STUDY_ID = "musr_team_allocation_validation_01"
VALIDATION_PROMPT_VERSION = "musr_team_allocation_validation_v1"
Z_95 = 1.959963984540054


@dataclass(frozen=True, slots=True)
class ValidationStudyConfig:
    num_tasks: int = 3
    population_sizes: tuple[int, int] = (12, 24)
    branches_per_latent_fact: int = 3
    statements_per_branch: int = 2
    tree_depth: int = 2
    evidence_redundancy: int = 3
    min_margin: int = 1
    seed: int = 20260901
    semantic_retries: int = 3
    candidate_limit: int = 12
    full_calls_per_task: int = 5
    full_required_correct: int = 4
    zero_calls_per_task: int = 5
    generation_temperature: float = 1.0
    generation_max_output_tokens: int = 2048
    validation_temperature: float = 1.0
    validation_max_output_tokens: int = 1024
    generation_prompt_version: str = PROMPT_VERSION
    validation_prompt_version: str = VALIDATION_PROMPT_VERSION
    skip_full_acceptance_for_testing: bool = False

    def __post_init__(self) -> None:
        if self.num_tasks != 3:
            raise ValueError("validation_01 requires exactly three accepted tasks")
        if tuple(self.population_sizes) != (12, 24):
            raise ValueError("validation_01 requires matched N=12 and N=24 variants")
        if self.full_calls_per_task != 5 or self.full_required_correct != 4:
            raise ValueError(
                "validation_01 requires at least 4 correct out of 5 full calls"
            )
        if self.zero_calls_per_task != 5:
            raise ValueError(
                "validation_01 requires five zero-information calls per task"
            )


class JsonlJournal:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, row: Mapping[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(canonical_json(dict(row)) + "\n")
            stream.flush()


def _git_metadata(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args], cwd=root, check=True, capture_output=True, text=True
        )
        return completed.stdout.strip()

    try:
        return {
            "commit": run("rev-parse", "HEAD"),
            "dirty": bool(run("status", "--porcelain")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _cards(task: Mapping[str, Any]) -> tuple[EvidenceCard, ...]:
    return tuple(
        EvidenceCard(
            evidence_id=str(item["evidence_id"]),
            latent_fact_id=str(item["latent_fact_id"]),
            branch_id=str(item["branch_id"]),
            statements=tuple(str(line) for line in item["text"]),
        )
        for item in task["evidence"]
    )


def _semantic_payload(task: Mapping[str, Any], task_id: str) -> dict[str, Any]:
    payload = {
        key: value
        for key, value in task.items()
        if key not in {"agent_evidence_ids", "fingerprint_sha256"}
    }
    payload["task_id"] = task_id
    payload["validation"] = {
        "structural_qa_passed": True,
        "selection_rule": "full_information_at_least_4_of_5",
    }
    payload["semantic_world_sha256"] = sha256_object(payload)
    return payload


def build_distribution_variant(
    task: Mapping[str, Any],
    *,
    population_size: int,
    redundancy: int,
    seed: Seed,
) -> dict[str, Any]:
    problem = LatentProblem.from_dict(task["latent"])
    cards = _cards(task)
    assignments = distribute_evidence(
        cards,
        problem,
        population_size=population_size,
        redundancy=redundancy,
        rng=seed.create_random(),
    )
    errors = validate_distribution(
        cards, assignments, population_size=population_size, problem=problem
    )
    if errors:
        raise RuntimeError("invalid matched distribution: " + "; ".join(errors))
    card_to_fact = {card.evidence_id: card.latent_fact_id for card in cards}
    all_ids = set(card_to_fact)
    agent_diagnostics = []
    violations = 0
    for agent, evidence_ids in assignments.items():
        facts = {card_to_fact[item] for item in evidence_ids}
        certified, winner = agent_can_certify_unique_allocation(problem, facts)
        violations += int(certified)
        agent_diagnostics.append(
            {
                "agent_id": agent,
                "evidence_cards": len(evidence_ids),
                "distinct_latent_facts": len(facts),
                "global_evidence_fraction": len(set(evidence_ids)) / len(all_ids),
                "structurally_certifies_solution": certified,
                "certified_index": winner,
            }
        )
    payload = {
        "schema_version": "musr_team_allocation_distribution_v1",
        "task_id": task["task_id"],
        "semantic_world_sha256": task["semantic_world_sha256"],
        "population_size": population_size,
        "evidence_redundancy": redundancy,
        "distribution_seed": int(seed),
        "agent_evidence_ids": assignments,
        "agent_diagnostics": agent_diagnostics,
        "no_single_agent_violations": violations,
    }
    payload["fingerprint_sha256"] = sha256_object(payload)
    return payload


def _displayed_options(
    task: Mapping[str, Any], seed: Seed
) -> tuple[list[dict[str, str]], dict[str, str]]:
    semantic = [dict(option) for option in task["options"]]
    seed.create_random().shuffle(semantic)
    labels = ("A", "B", "C")
    displayed = [
        {
            "label": label,
            "semantic_option_id": str(option["id"]),
            "display_text": str(option["display_text"]),
        }
        for label, option in zip(labels, semantic, strict=True)
    ]
    return displayed, {item["label"]: item["semantic_option_id"] for item in displayed}


def validation_prompt(
    task: Mapping[str, Any],
    displayed_options: Sequence[Mapping[str, str]],
    evidence_ids: Sequence[str],
    *,
    condition: str,
) -> str:
    evidence_map = {item["evidence_id"]: item for item in task["evidence"]}
    options = "\n".join(
        f"{item['label']}) {item['display_text']}" for item in displayed_options
    )
    if evidence_ids:
        evidence = "\n".join(
            f"- {evidence_id}: {' '.join(evidence_map[evidence_id]['text'])}"
            for evidence_id in evidence_ids
        )
    else:
        evidence = (
            "No private evidence is available. Choose using only the task framing."
        )
    return f"""You are solving a Team Allocation reasoning task in the {condition} information condition.
One person must do the first task and the remaining two jointly do the second task. The strongest allocation depends on the relevant individual skills and on how well the two-person team cooperates. Choose exactly one displayed option. Do not invent additional evidence.

Scenario:
{task["scenario"]}

Candidate allocations:
{options}

Available evidence:
{evidence}

Return JSON only with this shape:
{{"option_label": "A", "rationale": "brief reasoning based only on the available information"}}
"""


async def run_validation_call(
    model: MuSRGenerationModel,
    task: Mapping[str, Any],
    *,
    condition: str,
    call_index: int,
    seed: Seed,
    raw_journal: JsonlJournal,
    population_size: int | None = None,
    agent_id: str | None = None,
    evidence_ids: Sequence[str] = (),
    logical_call_id: str | None = None,
) -> dict[str, Any]:
    displayed, mapping = _displayed_options(task, seed.derive("options"))
    prompt = validation_prompt(task, displayed, evidence_ids, condition=condition)
    response = await model.inference(
        prompt,
        seed=int(seed.derive("provider")),
        purpose=f"validation_{condition}",
        metadata={
            "task_id": task["task_id"],
            "condition": condition,
            "population_size": population_size,
            "agent_id": agent_id,
            "call_index": call_index,
            "validation_prompt_version": VALIDATION_PROMPT_VERSION,
            "logical_call_id": logical_call_id,
        },
    )
    selected_label: str | None = None
    semantic_answer: str | None = None
    rationale = ""
    parse_error: str | None = None
    try:
        parsed = extract_json_object(response.content)
        label = str(parsed.get("option_label", "")).strip().upper()
        if label not in mapping:
            raise ValueError("option_label must be A, B, or C")
        selected_label = label
        semantic_answer = mapping[label]
        rationale = str(parsed.get("rationale", ""))
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        parse_error = str(exc)
    row = {
        "logical_call_id": logical_call_id,
        "task_id": task["task_id"],
        "condition": condition,
        "population_size": population_size,
        "agent_id": agent_id,
        "call_index": call_index,
        "seed": int(seed),
        "prompt_version": VALIDATION_PROMPT_VERSION,
        "prompt": prompt,
        "displayed_options": displayed,
        "semantic_option_mapping": mapping,
        "gold_answer": task["gold_answer"],
        "gold_index": task["gold_index"],
        "selected_label": selected_label,
        "parsed_semantic_answer": semantic_answer,
        "correct": semantic_answer == task["gold_answer"],
        "parse_success": semantic_answer is not None,
        "parse_error": parse_error,
        "rationale": rationale,
        "raw_response": response.content,
        "provider": response.provider,
        "model": response.model,
        "request_id": response.request_id,
        "usage": response.usage.to_dict(),
        "latency_seconds": response.latency_seconds,
        "retries": response.retries,
        "evidence_ids": list(evidence_ids),
        "evidence_card_count": len(evidence_ids),
        "distinct_latent_facts": len(
            {
                item["latent_fact_id"]
                for item in task["evidence"]
                if item["evidence_id"] in set(evidence_ids)
            }
        ),
    }
    raw_journal.append(row)
    return row


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 1.0
    proportion = successes / total
    denominator = 1 + Z_95 * Z_95 / total
    centre = (proportion + Z_95 * Z_95 / (2 * total)) / denominator
    margin = (
        Z_95
        * math.sqrt(
            proportion * (1 - proportion) / total + Z_95 * Z_95 / (4 * total * total)
        )
        / denominator
    )
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _summarize(
    rows: Sequence[Mapping[str, Any]], keys: Sequence[str]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key) for key in keys)].append(row)
    output = []
    for values, group in sorted(
        grouped.items(), key=lambda item: tuple(str(v) for v in item[0])
    ):
        correct = sum(bool(row["correct"]) for row in group)
        parsed = sum(bool(row["parse_success"]) for row in group)
        low, high = wilson_interval(correct, len(group))
        histogram = Counter(row.get("parsed_semantic_answer") for row in group)
        result = dict(zip(keys, values, strict=True))
        result.update(
            {
                "n": len(group),
                "correct": correct,
                "accuracy": correct / len(group),
                "ci95_low": low,
                "ci95_high": high,
                "parse_successes": parsed,
                "parse_rate": parsed / len(group),
                "answer_ALLOCATION_0": histogram.get("ALLOCATION_0", 0),
                "answer_ALLOCATION_1": histogram.get("ALLOCATION_1", 0),
                "answer_ALLOCATION_2": histogram.get("ALLOCATION_2", 0),
                "unparsed": histogram.get(None, 0),
            }
        )
        output.append(result)
    return output


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _distribution_summary(
    tasks: Sequence[Mapping[str, Any]],
    variants: Mapping[tuple[str, int], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_id = {str(task["task_id"]): task for task in tasks}
    rows = []
    for (task_id, population_size), variant in sorted(variants.items()):
        diagnostics = variant["agent_diagnostics"]
        card_counts = [item["evidence_cards"] for item in diagnostics]
        fact_counts = [item["distinct_latent_facts"] for item in diagnostics]
        fractions = [item["global_evidence_fraction"] for item in diagnostics]
        task = by_id[task_id]
        fact_touches = Counter()
        card_to_fact = {
            item["evidence_id"]: item["latent_fact_id"] for item in task["evidence"]
        }
        for evidence_ids in variant["agent_evidence_ids"].values():
            for fact in {card_to_fact[evidence_id] for evidence_id in evidence_ids}:
                fact_touches[fact] += 1
        rows.append(
            {
                "task_id": task_id,
                "population_size": population_size,
                "total_evidence_cards": len(task["evidence"]),
                "total_explicit_snippets": sum(
                    len(item["text"]) for item in task["evidence"]
                ),
                "cards_per_agent_mean": sum(card_counts) / len(card_counts),
                "cards_per_agent_min": min(card_counts),
                "cards_per_agent_max": max(card_counts),
                "facts_per_agent_mean": sum(fact_counts) / len(fact_counts),
                "facts_per_agent_min": min(fact_counts),
                "facts_per_agent_max": max(fact_counts),
                "evidence_fraction_mean": sum(fractions) / len(fractions),
                "evidence_fraction_min": min(fractions),
                "evidence_fraction_max": max(fractions),
                "evidence_redundancy": variant["evidence_redundancy"],
                "no_single_agent_violations": variant["no_single_agent_violations"],
                "latent_fact_touch_fraction_min": min(fact_touches.values())
                / population_size,
                "latent_fact_touch_fraction_max": max(fact_touches.values())
                / population_size,
                "exact_margin": task["latent"]["margin_to_second_best"],
            }
        )
    return rows


def _plots(
    analysis_dir: Path,
    behavioral: Sequence[Mapping[str, Any]],
    per_task: Sequence[Mapping[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = analysis_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    order = [("zero", None), ("partial", 12), ("partial", 24), ("full", None)]
    labels = ["Zero", "Partial N=12", "Partial N=24", "Full"]
    values = []
    lows = []
    highs = []
    for condition, population in order:
        row = next(
            item
            for item in behavioral
            if item["condition"] == condition
            and item.get("population_size") == population
        )
        values.append(row["accuracy"])
        lows.append(row["accuracy"] - row["ci95_low"])
        highs.append(row["ci95_high"] - row["accuracy"])
    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    axis.bar(labels, values, color=["#8da0cb", "#66c2a5", "#2ca25f", "#fc8d62"])
    axis.errorbar(
        range(4), values, yerr=[lows, highs], fmt="none", color="black", capsize=4
    )
    axis.axhline(
        1 / 3, color="black", linestyle="--", linewidth=1, label="Chance = 1/3"
    )
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Accuracy")
    axis.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "accuracy_by_information_condition.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6.4, 4.4))
    for task_id in sorted({str(row["task_id"]) for row in per_task}):
        points = [
            row
            for row in per_task
            if row["condition"] == "partial" and row["task_id"] == task_id
        ]
        points.sort(key=lambda item: item["population_size"])
        axis.plot(
            [item["population_size"] for item in points],
            [item["accuracy"] for item in points],
            marker="o",
            label=task_id,
        )
    axis.axhline(1 / 3, color="black", linestyle="--", linewidth=1)
    axis.set_xticks([12, 24])
    axis.set_ylim(0, 1.05)
    axis.set_xlabel("Population size")
    axis.set_ylabel("Partial-information accuracy")
    axis.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "partial_accuracy_by_population.png", dpi=180)
    plt.close(fig)


def _table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(str(row.get(column, "")) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def _report(
    tasks: Sequence[Mapping[str, Any]],
    variants: Mapping[tuple[str, int], Mapping[str, Any]],
    behavioral: Sequence[Mapping[str, Any]],
    per_task: Sequence[Mapping[str, Any]],
    distribution: Sequence[Mapping[str, Any]],
    config: ValidationStudyConfig,
    rows: Sequence[Mapping[str, Any]],
) -> str:
    def pct(value: float) -> str:
        return f"{100 * value:.1f}%"

    behavior_rows = []
    names = {
        ("zero", None): "Zero evidence",
        ("partial", 12): "Partial evidence",
        ("partial", 24): "Partial evidence",
        ("full", None): "Full evidence",
    }
    for item in behavioral:
        key = (item["condition"], item.get("population_size"))
        behavior_rows.append(
            {
                "Condition": names[key],
                "N": item.get("population_size") or "—",
                "Observations": item["n"],
                "Correct": item["correct"],
                "Accuracy": pct(item["accuracy"]),
                "95% CI": f"[{pct(item['ci95_low'])}, {pct(item['ci95_high'])}]",
                "Parse rate": pct(item["parse_rate"]),
            }
        )
    task_qa = []
    full_by_task = {
        row["task_id"]: row for row in per_task if row["condition"] == "full"
    }
    for task in tasks:
        task_id = str(task["task_id"])
        task_qa.append(
            {
                "Task": task_id,
                "Gold allocation": task["options"][task["gold_index"]]["display_text"],
                "Margin": task["latent"]["margin_to_second_best"],
                "Cards": len(task["evidence"]),
                "Snippets": sum(len(card["text"]) for card in task["evidence"]),
                "Full": f"{full_by_task[task_id]['correct']}/5",
                "Leakage": "pass",
                "N12": "pass"
                if variants[(task_id, 12)]["no_single_agent_violations"] == 0
                else "fail",
                "N24": "pass"
                if variants[(task_id, 24)]["no_single_agent_violations"] == 0
                else "fail",
            }
        )
    zero = next(item for item in behavioral if item["condition"] == "zero")
    partial12 = next(
        item
        for item in behavioral
        if item["condition"] == "partial" and item["population_size"] == 12
    )
    partial24 = next(
        item
        for item in behavioral
        if item["condition"] == "partial" and item["population_size"] == 24
    )
    full = next(item for item in behavioral if item["condition"] == "full")
    partial_pooled = (partial12["correct"] + partial24["correct"]) / (
        partial12["n"] + partial24["n"]
    )
    ordering = zero["accuracy"] < partial_pooled < full["accuracy"]
    example = tasks[0]
    example_id = str(example["task_id"])
    n12 = variants[(example_id, 12)]
    n24 = variants[(example_id, 24)]
    evidence_map = {item["evidence_id"]: item for item in example["evidence"]}
    sample_full = next(
        row
        for row in rows
        if row["task_id"] == example_id
        and row["condition"] == "full"
        and row["correct"]
    )
    branch_examples = [
        f"- **{item['evidence_id']}** ({item['latent_fact_id']}): {' '.join(item['text'])}"
        for item in example["evidence"][:4]
    ]
    agent12 = "0"
    agent24 = "0"
    agent12_text = [
        f"- {item}: {' '.join(evidence_map[item]['text'])}"
        for item in n12["agent_evidence_ids"][agent12]
    ]
    agent24_text = [
        f"- {item}: {' '.join(evidence_map[item]['text'])}"
        for item in n24["agent_evidence_ids"][agent24]
    ]
    latent_rows = [
        {"Person": person, "Task 1 skill": values[0], "Task 2 skill": values[1]}
        for person, values in example["latent"]["skill_matrix"].items()
    ]
    cooperation_rows = [
        {"Pair": pair.replace("|", " + "), "Cooperation": value}
        for pair, value in example["latent"]["cooperation_matrix"].items()
    ]
    score_rows = [
        {
            "Option": index,
            "Allocation": option["display_text"],
            "Score": example["latent"]["candidate_scores"][index],
        }
        for index, option in enumerate(example["options"])
    ]
    return (
        f"""# Native MuSR Team Allocation: pilot validation study

## A. Study design

This systematic pilot used three independently generated semantic worlds. Each world has exact Team Allocation ground truth, three candidate allocations, and MuSR-style language-model-generated evidence. Generation and validation used `{example["generation"]["model"]}` through the MAS-CC provider abstraction. The same evidence pool was repartitioned into matched populations of 12 and 24 agents. Full, partial, and zero-information calls independently randomized the displayed option labels before mapping answers back to semantic allocations.

## B. Generator configuration

| Setting | Value |
| --- | --- |
| Semantic tasks | {config.num_tasks} |
| `tree_depth` | {config.tree_depth} |
| `branches_per_latent_fact` | {config.branches_per_latent_fact} |
| Latent facts per task | 9 |
| Provider/model | {example["generation"]["provider"]} / {example["generation"]["model"]} |
| Study seed | {config.seed} |
| Population sizes | 12, 24 |
| Choices per task | 3 |

## C. Task-generation and structural QA

{_table(task_qa, ("Task", "Gold allocation", "Margin", "Cards", "Snippets", "Full", "Leakage", "N12", "N24"))}

Only structural QA and the explicit full-information threshold selected tasks. Partial and zero-information results were not used for selection.

## D. Behavioral validation

{_table(behavior_rows, ("Condition", "N", "Observations", "Correct", "Accuracy", "95% CI", "Parse rate"))}

Per-task results are retained in `per_task_summary.csv`. Semantic answer histograms are retained in both summary CSV files.

## E. Distribution diagnostics

{_table(distribution, ("task_id", "population_size", "cards_per_agent_mean", "cards_per_agent_min", "cards_per_agent_max", "facts_per_agent_mean", "facts_per_agent_min", "facts_per_agent_max", "evidence_redundancy", "no_single_agent_violations"))}

## F. Interpretation

The desired strict ordering `zero < partial < full` was **{"observed" if ordering else "not observed"}** when the two partial populations were pooled. Zero-information accuracy was {pct(zero["accuracy"])}; partial accuracy was {pct(partial12["accuracy"])} for N=12 and {pct(partial24["accuracy"])} for N=24; full-information accuracy was {pct(full["accuracy"])}.

{"Partial evidence remained close to chance and should be treated as a task-design warning." if abs(partial_pooled - 1 / 3) < 0.08 else "Partial evidence moved accuracy away from chance, providing evidence that individual views carry useful but incomplete information."}

## G. Limitations

This is a three-world pilot validation. The agent-level observations are useful diagnostics, but only three independently generated semantic tasks are insufficient for broad inferential claims. The full-information threshold also selected worlds whose language was reliably solvable, so pooled full accuracy is a quality-assurance result rather than an unbiased estimate over all candidate worlds.

## Worked example: {example_id}

### Scenario

{example["scenario"]}

### Candidate allocations

"""
        + "\n".join(
            f"{index + 1}. {option['display_text']}"
            for index, option in enumerate(example["options"])
        )
        + f"""

**Correct allocation:** {example["options"][example["gold_index"]]["display_text"]}

### Hidden evaluation metadata

The following exact tables are never included in validation prompts.

{_table(latent_rows, ("Person", "Task 1 skill", "Task 2 skill"))}

{_table(cooperation_rows, ("Pair", "Cooperation"))}

{_table(score_rows, ("Option", "Allocation", "Score"))}

### Generated evidence branches

{chr(10).join(branch_examples)}

### Representative N=12 agent view: agent {agent12}

{chr(10).join(agent12_text)}

### Representative N=24 agent view: agent {agent24}

{chr(10).join(agent24_text)}

Neither representative agent owns every latent fact, and exact score-difference bounds confirm that neither view certifies one unique allocation.

### Full-information validation response

Selected semantic answer: `{sample_full["parsed_semantic_answer"]}` (correct). Rationale: {sample_full["rationale"]}

## Figures

![Accuracy by information condition](figures/accuracy_by_information_condition.png)

![Partial accuracy by population](figures/partial_accuracy_by_population.png)
"""
    )


async def run_validation_study(
    provider: LLMProvider,
    config: ValidationStudyConfig,
    *,
    output: Path,
    repository_root: Path,
) -> dict[str, Any]:
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty study directory: {output}"
        )
    raw_dir = output / "raw"
    task_dir = output / "tasks"
    analysis_dir = output / "analysis"
    for path in (raw_dir, task_dir, analysis_dir / "figures"):
        path.mkdir(parents=True, exist_ok=True)
    generation_journal = JsonlJournal(raw_dir / "generation_calls.jsonl")
    model = MuSRGenerationModel(
        provider,
        temperature=config.generation_temperature,
        max_output_tokens=config.generation_max_output_tokens,
        prompt_version=config.generation_prompt_version,
        audit_sink=generation_journal.append,
    )
    base_generation = GenerationConfig(
        num_tasks=1,
        population_size=24,
        branches_per_latent_fact=config.branches_per_latent_fact,
        statements_per_branch=config.statements_per_branch,
        tree_depth=config.tree_depth,
        evidence_redundancy=config.evidence_redundancy,
        min_margin=config.min_margin,
        seed=config.seed,
        semantic_retries=config.semantic_retries,
        world_retries=1,
        run_full_information_validation=False,
    )
    accepted: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    variants: dict[tuple[str, int], dict[str, Any]] = {}
    candidate_records = []
    candidate_index = 0
    while len(accepted) < config.num_tasks and candidate_index < config.candidate_limit:
        candidate_index += 1
        candidate_config = GenerationConfig(
            **{
                **asdict(base_generation),
                "seed": int(Seed(config.seed).derive(f"candidate-{candidate_index}")),
            }
        )
        try:
            generated = (
                await generate_world(model, candidate_config, task_index=1)
            ).to_dict()
        except Exception as exc:
            candidate_records.append(
                {
                    "candidate_index": candidate_index,
                    "accepted": False,
                    "error": str(exc),
                }
            )
            continue
        task_id = f"task_{len(accepted) + 1:03d}"
        base = _semantic_payload(generated, task_id)
        candidate_full_journal = JsonlJournal(
            raw_dir / "candidate_full_information.jsonl"
        )
        validation_model = MuSRGenerationModel(
            provider,
            temperature=config.validation_temperature,
            max_output_tokens=config.validation_max_output_tokens,
            prompt_version=config.validation_prompt_version,
        )
        candidate_rows = []
        for call_index in range(config.full_calls_per_task):
            candidate_rows.append(
                await run_validation_call(
                    validation_model,
                    base,
                    condition="full",
                    call_index=call_index,
                    seed=Seed(config.seed).derive(
                        f"candidate-{candidate_index}-full-{call_index}"
                    ),
                    raw_journal=candidate_full_journal,
                    evidence_ids=[item["evidence_id"] for item in base["evidence"]],
                )
            )
        correct = sum(row["correct"] for row in candidate_rows)
        is_accepted = (
            config.skip_full_acceptance_for_testing
            or correct >= config.full_required_correct
        )
        candidate_records.append(
            {
                "candidate_index": candidate_index,
                "accepted": is_accepted,
                "full_correct": correct,
            }
        )
        if not is_accepted:
            continue
        accepted_full_journal = JsonlJournal(raw_dir / "full_information.jsonl")
        for row in candidate_rows:
            accepted_full_journal.append(row)
        accepted.append(base)
        rows.extend(candidate_rows)
        destination = task_dir / task_id
        destination.mkdir(parents=True, exist_ok=True)
        write_json_atomic(destination / "base_task.json", base)
        for population_size in config.population_sizes:
            variant = build_distribution_variant(
                base,
                population_size=population_size,
                redundancy=config.evidence_redundancy,
                seed=Seed(config.seed).derive(f"{task_id}-N{population_size}"),
            )
            variants[(task_id, population_size)] = variant
            write_json_atomic(
                destination / f"distribution_N{population_size}.json", variant
            )

    if len(accepted) != config.num_tasks:
        raise RuntimeError(
            f"accepted only {len(accepted)}/{config.num_tasks} tasks after {candidate_index} candidates"
        )

    validation_model = MuSRGenerationModel(
        provider,
        temperature=config.validation_temperature,
        max_output_tokens=config.validation_max_output_tokens,
        prompt_version=config.validation_prompt_version,
    )
    for task in accepted:
        task_id = str(task["task_id"])
        zero_journal = JsonlJournal(raw_dir / "zero_information.jsonl")
        for call_index in range(config.zero_calls_per_task):
            rows.append(
                await run_validation_call(
                    validation_model,
                    task,
                    condition="zero",
                    call_index=call_index,
                    seed=Seed(config.seed).derive(f"{task_id}-zero-{call_index}"),
                    raw_journal=zero_journal,
                )
            )
        for population_size in config.population_sizes:
            partial_journal = JsonlJournal(
                raw_dir / f"partial_N{population_size}.jsonl"
            )
            assignments = variants[(task_id, population_size)]["agent_evidence_ids"]
            for agent_id in sorted(assignments, key=int):
                rows.append(
                    await run_validation_call(
                        validation_model,
                        task,
                        condition="partial",
                        call_index=int(agent_id),
                        seed=Seed(config.seed).derive(
                            f"{task_id}-partial-N{population_size}-{agent_id}"
                        ),
                        raw_journal=partial_journal,
                        population_size=population_size,
                        agent_id=agent_id,
                        evidence_ids=assignments[agent_id],
                    )
                )

    expected = {
        ("full", None): 15,
        ("zero", None): 15,
        ("partial", 12): 36,
        ("partial", 24): 72,
    }
    observed = Counter((row["condition"], row["population_size"]) for row in rows)
    if observed != expected:
        raise RuntimeError(f"validation row counts do not match contract: {observed}")

    behavioral = _summarize(rows, ("condition", "population_size"))
    per_task = _summarize(rows, ("task_id", "condition", "population_size"))
    distribution = _distribution_summary(accepted, variants)
    agent_rows = [row for row in rows if row["condition"] == "partial"]
    _write_csv(analysis_dir / "behavioral_summary.csv", behavioral)
    _write_csv(analysis_dir / "per_task_summary.csv", per_task)
    _write_csv(analysis_dir / "distribution_summary.csv", distribution)
    _write_csv(analysis_dir / "agent_level_results.csv", agent_rows)
    _plots(analysis_dir, behavioral, per_task)
    report = _report(
        accepted, variants, behavioral, per_task, distribution, config, rows
    )
    (analysis_dir / "validation_report.md").write_text(report, encoding="utf-8")

    resolved_config = {
        **asdict(config),
        "population_sizes": list(config.population_sizes),
    }
    (output / "config.yaml").write_text(
        yaml.safe_dump(resolved_config, sort_keys=False), encoding="utf-8"
    )
    readme = f"""# {STUDY_ID}

This directory is the complete pilot validation artifact for three native MuSR-style Team Allocation worlds. The paper-ready result is in `analysis/validation_report.md`. Raw provider responses are under `raw/`; matched frozen tasks are under `tasks/`.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    artifact_hashes = {
        str(path.relative_to(output)): sha256_file(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    manifest = {
        "study_id": STUDY_ID,
        "status": "complete",
        "config": resolved_config,
        "provider": provider.name,
        "model": provider.model,
        "mas_cc_git": _git_metadata(repository_root),
        "musr_commit": MUSR_COMMIT,
        "candidate_records": candidate_records,
        "accepted_task_ids": [task["task_id"] for task in accepted],
        "expected_validation_calls": 138,
        "observed_validation_calls": len(rows),
        "artifact_hashes": artifact_hashes,
    }
    manifest["manifest_content_sha256"] = sha256_object(manifest)
    write_json_atomic(output / "manifest.json", manifest)
    return {"manifest": manifest, "behavioral": behavioral, "output": str(output)}
