"""Add validation-only model arms to an existing Team Allocation pilot."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mas_cc.core.random import Seed
from mas_cc.llm_runtime.providers.protocols import LLMProvider

from .io_utils import sha256_file, sha256_object, write_json_atomic
from .provider_adapter import MuSRGenerationModel
from .validation_study import (
    JsonlJournal,
    VALIDATION_PROMPT_VERSION,
    _summarize,
    _table,
    _write_csv,
    run_validation_call,
)


def _slug(value: str) -> str:
    return "".join(
        character if character.isalnum() else "_" for character in value
    ).strip("_")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _logical_call_id(
    task_id: str,
    condition: str,
    call_index: int,
    population_size: int | None,
    agent_id: str | None,
) -> str:
    return "|".join(
        (
            task_id,
            condition,
            str(population_size) if population_size is not None else "none",
            agent_id or "none",
            str(call_index),
        )
    )


def _model_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    summary = _summarize(rows, ("validation_model", "condition", "population_size"))
    for row in summary:
        row["model"] = row.pop("validation_model")
    return summary


def _combined_report_section(
    original_model: str,
    additional_model: str,
    summary: Sequence[Mapping[str, Any]],
) -> str:
    rows = []
    for item in summary:
        rows.append(
            {
                "Validation model": item["model"],
                "Condition": item["condition"],
                "N": item.get("population_size") or "—",
                "Observations": item["n"],
                "Correct": item["correct"],
                "Accuracy": f"{100 * item['accuracy']:.1f}%",
                "95% CI": f"[{100 * item['ci95_low']:.1f}%, {100 * item['ci95_high']:.1f}%]",
                "Parse rate": f"{100 * item['parse_rate']:.1f}%",
            }
        )
    return f"""## H. Validation-model comparison

The frozen tasks and evidence distributions were not regenerated. The original validation arm used `{original_model}`. The same full, zero, and every-agent partial prompts were then rerun with `{additional_model}`, with independently seeded option permutations. This separates evidence generation from validation-model behavior.

{_table(rows, ("Validation model", "Condition", "N", "Observations", "Correct", "Accuracy", "95% CI", "Parse rate"))}

These are descriptive model-specific results from only three semantic worlds. Differences between the two validation models must not be interpreted as broad model rankings.
"""


def _comparison_plot(
    path: Path, summary: Sequence[Mapping[str, Any]], models: Sequence[str]
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = [("zero", None), ("partial", 12), ("partial", 24), ("full", None)]
    labels = ["Zero", "Partial N=12", "Partial N=24", "Full"]
    width = 0.36
    x = list(range(len(order)))
    figure, axis = plt.subplots(figsize=(8.2, 4.6))
    for model_index, model in enumerate(models):
        values = []
        for condition, population in order:
            row = next(
                item
                for item in summary
                if item["model"] == model
                and item["condition"] == condition
                and item.get("population_size") == population
            )
            values.append(row["accuracy"])
        positions = [value + (model_index - 0.5) * width for value in x]
        axis.bar(positions, values, width=width, label=model)
    axis.axhline(
        1 / 3, color="black", linestyle="--", linewidth=1, label="Chance = 1/3"
    )
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Accuracy")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


async def add_validation_model(
    provider: LLMProvider,
    *,
    study_dir: Path,
    seed: int,
    temperature: float = 1.0,
    max_output_tokens: int = 1024,
) -> dict[str, Any]:
    root = study_dir.resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"completed study manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise RuntimeError("base validation study is not complete")
    tasks = [
        json.loads(
            (root / "tasks" / task_id / "base_task.json").read_text(encoding="utf-8")
        )
        for task_id in manifest["accepted_task_ids"]
    ]
    variants = {
        (task["task_id"], population_size): json.loads(
            (
                root
                / "tasks"
                / task["task_id"]
                / f"distribution_N{population_size}.json"
            ).read_text(encoding="utf-8")
        )
        for task in tasks
        for population_size in (12, 24)
    }
    model_slug = _slug(provider.model)
    output_path = root / "raw" / f"validation_{model_slug}.jsonl"
    journal = JsonlJournal(output_path)
    existing = _read_jsonl(output_path)
    completed = {str(row["logical_call_id"]) for row in existing}
    model = MuSRGenerationModel(
        provider,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        prompt_version=VALIDATION_PROMPT_VERSION,
    )
    rows = list(existing)
    for task in tasks:
        task_id = str(task["task_id"])
        plans: list[tuple[str, int, int | None, str | None, Sequence[str]]] = []
        full_ids = [item["evidence_id"] for item in task["evidence"]]
        plans.extend(("full", index, None, None, full_ids) for index in range(5))
        plans.extend(("zero", index, None, None, ()) for index in range(5))
        for population_size in (12, 24):
            assignments = variants[(task_id, population_size)]["agent_evidence_ids"]
            plans.extend(
                (
                    "partial",
                    int(agent_id),
                    population_size,
                    agent_id,
                    evidence_ids,
                )
                for agent_id, evidence_ids in sorted(
                    assignments.items(), key=lambda item: int(item[0])
                )
            )
        for condition, call_index, population_size, agent_id, evidence_ids in plans:
            call_id = _logical_call_id(
                task_id, condition, call_index, population_size, agent_id
            )
            if call_id in completed:
                continue
            row = await run_validation_call(
                model,
                task,
                condition=condition,
                call_index=call_index,
                seed=Seed(seed).derive(f"{provider.model}|{call_id}"),
                raw_journal=journal,
                population_size=population_size,
                agent_id=agent_id,
                evidence_ids=evidence_ids,
                logical_call_id=call_id,
            )
            row["validation_model"] = provider.model
            # Replace the just-written row so the journal also records the explicit comparison arm.
            rows.append(row)
            completed.add(call_id)

    if len(rows) != 138:
        raise RuntimeError(f"additional model produced {len(rows)}/138 validation rows")
    counts = Counter((row["condition"], row["population_size"]) for row in rows)
    expected = {
        ("full", None): 15,
        ("zero", None): 15,
        ("partial", 12): 36,
        ("partial", 24): 72,
    }
    if counts != expected:
        raise RuntimeError(f"additional-model counts do not match contract: {counts}")

    original_rows = []
    for filename in (
        "full_information.jsonl",
        "zero_information.jsonl",
        "partial_N12.jsonl",
        "partial_N24.jsonl",
    ):
        original_rows.extend(_read_jsonl(root / "raw" / filename))
    original_model = str(manifest["model"])
    for row in original_rows:
        row["validation_model"] = original_model
    for row in rows:
        row["validation_model"] = provider.model
    all_rows = original_rows + rows
    combined = _model_summary(all_rows)
    per_task = _summarize(
        all_rows, ("validation_model", "task_id", "condition", "population_size")
    )
    analysis = root / "analysis"
    _write_csv(analysis / "behavioral_summary_by_model.csv", combined)
    _write_csv(analysis / "per_task_summary_by_model.csv", per_task)
    _comparison_plot(
        analysis / "figures" / "accuracy_by_validation_model.png",
        combined,
        (original_model, provider.model),
    )
    report_path = analysis / "validation_report.md"
    report = report_path.read_text(encoding="utf-8")
    marker = "\n## H. Validation-model comparison\n"
    if marker in report:
        report = report.split(marker, 1)[0].rstrip() + "\n"
    report += "\n" + _combined_report_section(original_model, provider.model, combined)
    report += (
        "\n![Accuracy by validation model](figures/accuracy_by_validation_model.png)\n"
    )
    report_path.write_text(report, encoding="utf-8")

    manifest.setdefault("additional_validation_models", {})[provider.model] = {
        "provider": provider.name,
        "model": provider.model,
        "temperature_requested": temperature,
        "max_output_tokens": max_output_tokens,
        "prompt_version": VALIDATION_PROMPT_VERSION,
        "seed": seed,
        "validation_calls": 138,
        "raw_file": str(output_path.relative_to(root)),
    }
    manifest["artifact_hashes"] = {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    manifest.pop("manifest_content_sha256", None)
    manifest["manifest_content_sha256"] = sha256_object(manifest)
    write_json_atomic(manifest_path, manifest)
    return {"rows": rows, "summary": combined, "output": str(root)}
