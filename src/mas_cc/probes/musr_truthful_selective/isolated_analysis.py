"""Missingness-safe aggregation for isolated truthful-selective OSS records."""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def request_table(
    root: Path, manifest: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    rows = []
    for request in manifest:
        checkpoint = root / "checkpoints" / f"{request['request_id']}.json"
        terminal = (
            json.loads(checkpoint.read_text(encoding="utf-8"))
            if checkpoint.is_file()
            else {}
        )
        parsed = terminal.get("parsed") or {}
        rows.append(
            {
                **dict(request),
                "status": terminal.get("status", "missing"),
                "parse_success": parsed.get("parse_success"),
                "semantic_answer": parsed.get("semantic_answer"),
                "gold_selected": parsed.get("gold_selected"),
                "false_target_selected": parsed.get("false_target_selected"),
                "other_selected": parsed.get("other_selected"),
                "provider_attempts": len(terminal.get("attempts", ())),
                "transport_retries": sum(
                    int((attempt.get("response") or {}).get("retries") or 0)
                    for attempt in terminal.get("attempts", ())
                ),
                "input_tokens": sum(
                    int(
                        ((attempt.get("response") or {}).get("usage") or {}).get(
                            "input_tokens"
                        )
                        or 0
                    )
                    for attempt in terminal.get("attempts", ())
                ),
                "output_tokens": sum(
                    int(
                        ((attempt.get("response") or {}).get("usage") or {}).get(
                            "output_tokens"
                        )
                        or 0
                    )
                    for attempt in terminal.get("attempts", ())
                ),
            }
        )
    return rows


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def aggregate(root: Path) -> dict[str, Any]:
    manifest = [
        json.loads(line)
        for line in (root / "preparation/evaluation_manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    rows = request_table(root, manifest)
    write_csv(root / "analysis/request_table.csv", rows)
    valid = [
        row
        for row in rows
        if row["status"] == "completed" and row["parse_success"] is True
    ]
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in valid:
        grouped[(row["task_id"], row["condition"], row.get("budget"))].append(row)
    summaries = []
    for (task, condition, budget), group in sorted(
        grouped.items(), key=lambda item: tuple(str(x) for x in item[0])
    ):
        summaries.append(
            {
                "task_id": task,
                "condition": condition,
                "budget": budget,
                "valid_n": len(group),
                "planned_n": sum(
                    r["task_id"] == task
                    and r["condition"] == condition
                    and r.get("budget") == budget
                    for r in rows
                ),
                "p_truth": _mean([float(r["gold_selected"]) for r in group]),
                "p_false": _mean([float(r["false_target_selected"]) for r in group]),
                "p_other": _mean([float(r["other_selected"]) for r in group]),
            }
        )
    write_csv(root / "analysis/per_task_summary.csv", summaries)

    by_request = {row["request_id"]: row for row in rows}
    private = {
        (row["task_id"], row["agent_id"], row["answer_order_id"]): row
        for row in valid
        if row["condition"] == "private"
    }
    paired = []
    for key, baseline in private.items():
        task, agent, order = key
        for budget in (3, 6, 9, 12):
            strategic = next(
                (
                    r
                    for r in valid
                    if r["task_id"] == task
                    and r["agent_id"] == agent
                    and r["answer_order_id"] == order
                    and r["condition"] == "strategic"
                    and r["budget"] == budget
                ),
                None,
            )
            randoms = [
                r
                for r in valid
                if r["task_id"] == task
                and r["agent_id"] == agent
                and r["answer_order_id"] == order
                and r["condition"] == "random"
                and r["budget"] == budget
            ]
            complete = strategic is not None and len(randoms) == 5
            paired.append(
                {
                    "task_id": task,
                    "agent_id": agent,
                    "answer_order_id": order,
                    "budget": budget,
                    "complete_pair": complete,
                    "random_valid_n": len(randoms),
                    "strategic_minus_private_false": None
                    if strategic is None
                    else float(strategic["false_target_selected"])
                    - float(baseline["false_target_selected"]),
                    "strategic_minus_random_false": None
                    if not complete
                    else float(strategic["false_target_selected"])
                    - statistics.fmean(
                        float(r["false_target_selected"]) for r in randoms
                    ),
                    "strategic_minus_private_truth": None
                    if strategic is None
                    else float(strategic["gold_selected"])
                    - float(baseline["gold_selected"]),
                    "strategic_minus_random_truth": None
                    if not complete
                    else float(strategic["gold_selected"])
                    - statistics.fmean(float(r["gold_selected"]) for r in randoms),
                }
            )
    write_csv(root / "analysis/paired_comparisons.csv", paired)
    diagnostics = [
        {
            "planned": len(rows),
            "completed_valid": len(valid),
            "invalid": sum(row["status"] == "invalid" for row in rows),
            "provider_failed": sum(row["status"] == "failed" for row in rows),
            "missing": sum(row["status"] == "missing" for row in rows),
            "complete_pairs": sum(row["complete_pair"] for row in paired),
            "incomplete_pairs": sum(not row["complete_pair"] for row in paired),
        }
    ]
    write_csv(root / "analysis/missingness_retry_diagnostics.csv", diagnostics)

    task_effects = []
    for task in sorted({row["task_id"] for row in paired}):
        group = [
            row for row in paired if row["task_id"] == task and row["complete_pair"]
        ]
        for budget in (3, 6, 9, 12):
            subset = [row for row in group if row["budget"] == budget]
            task_effects.append(
                {
                    "task_id": task,
                    "budget": budget,
                    "matched_units": len(subset),
                    "delta_false_vs_private": _mean(
                        [float(row["strategic_minus_private_false"]) for row in subset]
                    ),
                    "delta_false_vs_random": _mean(
                        [float(row["strategic_minus_random_false"]) for row in subset]
                    ),
                    "delta_truth_vs_private": _mean(
                        [float(row["strategic_minus_private_truth"]) for row in subset]
                    ),
                    "delta_truth_vs_random": _mean(
                        [float(row["strategic_minus_random_truth"]) for row in subset]
                    ),
                }
            )
    write_csv(root / "analysis/paired_effects_by_task.csv", task_effects)
    aggregate_rows = []
    for budget in (3, 6, 9, 12):
        group = [
            row
            for row in task_effects
            if row["budget"] == budget and row["delta_false_vs_random"] is not None
        ]
        aggregate_rows.append(
            {
                "budget": budget,
                "tasks": len(group),
                "weighting": "equal_task_primary",
                "delta_false_vs_private": _mean(
                    [float(row["delta_false_vs_private"]) for row in group]
                ),
                "delta_false_vs_random": _mean(
                    [float(row["delta_false_vs_random"]) for row in group]
                ),
                "delta_truth_vs_private": _mean(
                    [float(row["delta_truth_vs_private"]) for row in group]
                ),
                "delta_truth_vs_random": _mean(
                    [float(row["delta_truth_vs_random"]) for row in group]
                ),
            }
        )
    write_csv(root / "analysis/aggregate_summary.csv", aggregate_rows)
    _plots(root, summaries, task_effects)
    report = f"""# MuSR truthful-selective isolated OSS evaluation

Primary aggregation uses equal task weights. Calls are repeated measurements within three frozen tasks, not independent tasks.

- Planned requests: {len(rows)}
- Valid completed requests: {len(valid)}
- Missing/failed/invalid requests remain excluded from answer denominators.
- Complete matched units: {diagnostics[0]["complete_pairs"]}
- Incomplete matched units: {diagnostics[0]["incomplete_pairs"]}

See `request_table.csv`, `per_task_summary.csv`, `paired_comparisons.csv`, `paired_effects_by_task.csv`, and `aggregate_summary.csv`.
"""
    report_path = root / "analysis/isolated_oss_report.md"
    report_path.write_text(report, encoding="utf-8")
    return {"report": str(report_path), "diagnostics": diagnostics[0]}


def _plots(
    root: Path,
    summaries: Sequence[Mapping[str, Any]],
    effects: Sequence[Mapping[str, Any]],
) -> None:
    if not summaries:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = root / "analysis/figures"
    output.mkdir(parents=True, exist_ok=True)
    pooled: dict[tuple[str, Any], list[Mapping[str, Any]]] = defaultdict(list)
    for row in summaries:
        pooled[(str(row["condition"]), row.get("budget"))].append(row)
    conditions = sorted(pooled, key=lambda key: (key[0], int(key[1] or 0)))
    labels = [
        f"{condition}{'' if budget in {None, ''} else f'-{budget}'}"
        for condition, budget in conditions
    ]
    truth = [
        statistics.fmean(float(row["p_truth"]) for row in pooled[key])
        for key in conditions
    ]
    false = [
        statistics.fmean(float(row["p_false"]) for row in pooled[key])
        for key in conditions
    ]
    other = [
        statistics.fmean(float(row["p_other"]) for row in pooled[key])
        for key in conditions
    ]
    fig, ax = plt.subplots(figsize=(11, 5))
    x = range(len(labels))
    ax.plot(x, truth, marker="o", label="truth")
    ax.plot(x, false, marker="o", label="false target")
    ax.plot(x, other, marker="o", label="other")
    ax.set_xticks(list(x), labels, rotation=45, ha="right")
    ax.set(ylabel="Equal-task-weight frequency", ylim=(0, 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "choice_frequency_by_condition.png", dpi=180)
    plt.close(fig)
    for metric, filename, ylabel in (
        (
            "delta_false_vs_private",
            "strategic_false_shift_vs_private.png",
            "Strategic minus private false choice",
        ),
        (
            "delta_false_vs_random",
            "strategic_false_shift_vs_random.png",
            "Strategic minus random false choice",
        ),
        (
            "delta_truth_vs_private",
            "strategic_truth_shift_vs_private.png",
            "Strategic minus private truth choice",
        ),
    ):
        fig, ax = plt.subplots(figsize=(7, 4))
        for task in sorted({str(row["task_id"]) for row in effects}):
            rows = sorted(
                (row for row in effects if row["task_id"] == task),
                key=lambda row: int(row["budget"]),
            )
            ax.plot(
                [int(row["budget"]) for row in rows],
                [
                    float(row[metric]) if row[metric] is not None else float("nan")
                    for row in rows
                ],
                marker="o",
                label=task,
            )
        ax.axhline(0, color="black", linestyle="--")
        ax.set(xlabel="Report budget", ylabel=ylabel)
        if effects:
            ax.legend()
        fig.tight_layout()
        fig.savefig(output / filename, dpi=180)
        plt.close(fig)


__all__ = ["aggregate", "request_table", "write_csv"]
