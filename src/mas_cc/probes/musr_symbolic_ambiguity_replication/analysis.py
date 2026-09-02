"""Scientific tables, figures, and report for the frozen replication."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mas_cc.probes.musr_prompt_solvability.execution import read
from mas_cc.probes.musr_symbolic_ambiguity.analysis import (
    _correlation,
    observation_rows,
    summarize,
    terminal,
    write_csv,
)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _rank(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    output = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        average = (cursor + 1 + end) / 2
        for index in order[cursor:end]:
            output[index] = average
        cursor = end
    return output


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    return _correlation(_rank(xs), _rank(ys))


def _markdown_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    return "\n".join(
        [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
            *(
                "| " + " | ".join(str(row.get(column, "")) for column in columns) + " |"
                for row in rows
            ),
        ]
    )


def _load_observations(
    journal_path: Path, task_dir: Path, source_label: str
) -> list[dict[str, Any]]:
    journal = read(journal_path)
    raw = terminal(journal_path)
    rows = observation_rows(raw, task_dir)
    terminal_by_id = {str(row["call_id"]): row for row in raw}
    attempts: dict[str, int] = defaultdict(int)
    failed_ids: set[str] = set()
    evidence_to_latent: dict[tuple[str, str], str] = {}
    for path in sorted(task_dir.glob("task_*/base_task.json")):
        task_id = path.parent.name
        base = json.loads(path.read_text(encoding="utf-8"))
        for evidence in base["evidence"]:
            evidence_to_latent[(task_id, str(evidence["evidence_id"]))] = str(
                evidence["latent_fact_id"]
            )
    for event in journal:
        call_id = str(event.get("call_id", ""))
        if event.get("event") == "request_started":
            attempts[call_id] += 1
        elif event.get("event") == "call_failed":
            failed_ids.add(call_id)
    for row in rows:
        call_id = str(row["call_id"])
        event = terminal_by_id[call_id]
        evidence_ids = tuple(event.get("evidence_ids") or ())
        latent_ids = event.get("latent_ids") or list(
            dict.fromkeys(
                evidence_to_latent[(str(row["task_id"]), str(evidence_id))]
                for evidence_id in evidence_ids
            )
        )
        response = event.get("response") or {}
        row.update(
            {
                "observation_source": source_label,
                "replicate_id": event.get("replicate_id", row["repetition"]),
                "latent_ids": "|".join(str(value) for value in latent_ids),
                "requested_seed": event.get("provider_seed"),
                "provider": event.get("provider"),
                "model": event.get("model"),
                "request_id": event.get("request_id"),
                "provider_metadata": json.dumps(response, sort_keys=True),
                "raw_response": event.get("raw_response"),
                "parsed_response": json.dumps(
                    {
                        "parse_success": event.get("parse_success"),
                        "selected_letter": event.get("selected_letter"),
                        "semantic_answer": event.get("parsed_semantic_answer"),
                        "reason": event.get("reason"),
                        "shared_fact_id": event.get("shared_fact_id"),
                        "parse_error": event.get("parse_error"),
                    },
                    sort_keys=True,
                ),
                "attempt_count": attempts[call_id],
                "retry_status": (
                    "transport_failure_retried"
                    if call_id in failed_ids
                    else "provider_internal_retry"
                    if int(event.get("transport_retries") or 0) > 0
                    else "completed_parse_failure_not_retried"
                    if event.get("parse_success") is not True
                    else "none"
                ),
            }
        )
    return rows


def load_source_observations(source_root: Path) -> list[dict[str, Any]]:
    return _load_observations(
        source_root / "behavioral_validation/raw_calls.jsonl",
        source_root / "accepted_tasks",
        "calibration_01",
    )


def load_new_observations(root: Path, source_root: Path) -> list[dict[str, Any]]:
    return _load_observations(
        root / "behavioral_validation/new_raw_calls.jsonl",
        source_root / "accepted_tasks",
        "replication_01",
    )


def existing_task_diagnostic(source_root: Path) -> list[dict[str, Any]]:
    observations = load_source_observations(source_root)
    summaries = {
        (row["task_id"], row["condition"]): row
        for row in summarize(observations, ("task_id", "condition"))
    }
    private = summarize(
        [row for row in observations if row["condition"] == "Private"],
        ("task_id", "agent_id"),
    )
    agents: dict[str, list[float]] = defaultdict(list)
    for row in private:
        agents[str(row["task_id"])].append(float(row["truth_rate"]))
    frozen = json.loads(
        (source_root / "symbolic_scan/frozen_selection.json").read_text(
            encoding="utf-8"
        )
    )
    selected = {str(row["task_id"]): row for row in frozen["selected_worlds"]}
    output = []
    for task_id in sorted(selected):
        item = selected[task_id]
        values = agents[task_id]
        output.append(
            {
                "task_id": task_id,
                "gold": item["gold_answer"],
                "margin": item["score_margin"],
                "zero_truth_rate": summaries[(task_id, "Zero")]["truth_rate"],
                "private_truth_rate": summaries[(task_id, "Private")]["truth_rate"],
                "full_truth_rate": summaries[(task_id, "F9")]["truth_rate"],
                "private_agent_mean": statistics.fmean(values),
                "private_agent_min": min(values),
                "private_agent_max": max(values),
                "worst_private_M": max(
                    float(row["max_predictability"]) for row in item["private_views"]
                ),
            }
        )
    return output


def _final_task_rows(
    observations: Sequence[Mapping[str, Any]], source_root: Path
) -> list[dict[str, Any]]:
    summaries = {
        (row["task_id"], row["condition"]): row
        for row in summarize(observations, ("task_id", "condition"))
    }
    private = summarize(
        [row for row in observations if row["condition"] == "Private"],
        ("task_id", "agent_id"),
    )
    agents: dict[str, list[float]] = defaultdict(list)
    for row in private:
        agents[str(row["task_id"])].append(float(row["truth_rate"]))
    frozen = json.loads(
        (source_root / "symbolic_scan/frozen_selection.json").read_text(
            encoding="utf-8"
        )
    )
    output = []
    for item in frozen["selected_worlds"]:
        task_id = str(item["task_id"])
        values = agents[task_id]
        row: dict[str, Any] = {
            "task_id": task_id,
            "gold": item["gold_answer"],
            "margin": item["score_margin"],
            "worst_private_M": max(
                float(view["max_predictability"]) for view in item["private_views"]
            ),
            "private_agent_mean": statistics.fmean(values),
            "private_agent_min": min(values),
            "private_agent_max": max(values),
        }
        for condition, label in (
            ("Zero", "zero"),
            ("Private", "private"),
            ("F9", "full"),
        ):
            summary = summaries[(task_id, condition)]
            row[f"{label}_n"] = summary["n"]
            row[f"{label}_truth"] = summary["truth"]
            row[f"{label}_truth_rate"] = summary["truth_rate"]
            row[f"{label}_ci95_low"] = summary["ci95_low"]
            row[f"{label}_ci95_high"] = summary["ci95_high"]
        output.append(row)
    return output


def _private_view_rows(
    observations: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    summaries = summarize(
        [row for row in observations if row["condition"] == "Private"],
        ("task_id", "agent_id", "symbolic_M", "symbolic_Hbar"),
    )
    return [
        {
            "task_id": row["task_id"],
            "agent_id": row["agent_id"],
            "symbolic_M": row["symbolic_M"],
            "symbolic_Hbar": row["symbolic_Hbar"],
            "n": row["n"],
            "truth": row["truth"],
            "empirical_truth_rate": row["truth_rate"],
            "ci95_low": row["ci95_low"],
            "ci95_high": row["ci95_high"],
        }
        for row in summaries
    ]


def _associations(private_views: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    accuracy = [float(row["empirical_truth_rate"]) for row in private_views]
    output = []
    for metric in ("symbolic_M", "symbolic_Hbar"):
        values = [float(row[metric]) for row in private_views]
        output.append(
            {
                "symbolic_metric": metric,
                "empirical_metric": "private_truth_rate_over_6_repetitions",
                "n_views": len(private_views),
                "pearson_r": _correlation(values, accuracy),
                "spearman_rho": _spearman(values, accuracy),
                "interpretation": "descriptive_only",
            }
        )
    return output


def _figures(
    root: Path,
    pooled: Sequence[Mapping[str, Any]],
    task_rows: Sequence[Mapping[str, Any]],
    private_views: Sequence[Mapping[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = root / "analysis/figures"
    output.mkdir(parents=True, exist_ok=True)
    order = ("Zero", "Private", "F9")
    selected = [
        next(row for row in pooled if row["condition"] == condition)
        for condition in order
    ]
    values = [float(row["truth_rate"]) for row in selected]
    errors = [
        [
            value - float(row["ci95_low"])
            for value, row in zip(values, selected, strict=True)
        ],
        [
            float(row["ci95_high"]) - value
            for value, row in zip(values, selected, strict=True)
        ],
    ]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(("Zero", "Private", "Full F9"), values)
    ax.errorbar(range(3), values, yerr=errors, fmt="none", color="black", capsize=4)
    ax.axhline(1 / 3, color="black", linestyle="--", label="chance = 1/3")
    ax.set(ylabel="Truth-selection rate", ylim=(0, 1.05))
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "final_zero_private_full.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.24
    xs = list(range(len(task_rows)))
    for offset, (key, label) in zip(
        (-width, 0, width),
        (
            ("zero_truth_rate", "Zero"),
            ("private_truth_rate", "Private"),
            ("full_truth_rate", "Full"),
        ),
        strict=True,
    ):
        ax.bar(
            [x + offset for x in xs],
            [float(row[key]) for row in task_rows],
            width,
            label=label,
        )
    ax.axhline(1 / 3, color="black", linestyle="--")
    ax.set_xticks(xs, [str(row["task_id"]) for row in task_rows])
    ax.set(ylabel="Truth-selection rate", ylim=(0, 1.05))
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "per_task_zero_private_full.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(
        [float(row["empirical_truth_rate"]) for row in private_views],
        bins=[x / 12 for x in range(13)],
    )
    ax.axvline(1 / 3, color="black", linestyle="--")
    ax.set(xlabel="Private task-agent truth rate", ylabel="Number of task-agent views")
    fig.tight_layout()
    fig.savefig(output / "private_agent_view_distribution.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(
        [float(row["symbolic_M"]) for row in private_views],
        [float(row["empirical_truth_rate"]) for row in private_views],
        alpha=0.7,
    )
    ax.axhline(1 / 3, color="black", linestyle="--")
    ax.set(
        xlabel="Symbolic predictability $M_I$",
        ylabel="Empirical truth rate over 6 repetitions",
        ylim=(-0.02, 1.02),
    )
    fig.tight_layout()
    fig.savefig(output / "symbolic_vs_empirical.png", dpi=180)
    plt.close(fig)


def _recommendation(
    config: Any,
    pooled: Mapping[str, Mapping[str, Any]],
    task_rows: Sequence[Mapping[str, Any]],
) -> tuple[bool, str, list[str]]:
    zero = float(pooled["Zero"]["truth_rate"])
    private = float(pooled["Private"]["truth_rate"])
    full = float(pooled["F9"]["truth_rate"])
    strict = (
        zero <= config.zero_max_truth_rate
        and private <= config.private_max_truth_rate
        and full >= config.full_min_truth_rate
    )
    pathologies = [
        str(row["task_id"])
        for row in task_rows
        if float(row["full_truth_rate"]) < config.task_pathology_full_below
    ]
    if strict and not pathologies:
        return True, "PASS", pathologies
    if (
        zero <= config.zero_max_truth_rate
        and private <= config.borderline_private_max
        and full >= config.full_min_truth_rate
        and full - private >= config.minimum_full_private_separation
        and not pathologies
    ):
        return False, "BORDERLINE PASS / ACCEPTABLE FOR BLACKBOARD PILOT", pathologies
    return False, "FAIL", pathologies


def build_outputs(root: Path, source_root: Path, config: Any) -> dict[str, Any]:
    existing = load_source_observations(source_root)
    new = load_new_observations(root, source_root)
    observations = existing + new
    if len(existing) != 336:
        raise RuntimeError(f"expected 336 source observations, found {len(existing)}")
    if len(new) != 336:
        raise RuntimeError(f"expected 336 replication observations, found {len(new)}")
    if len({row["call_id"] for row in observations}) != 672:
        raise RuntimeError("combined call identities are not 672 unique values")
    pooled = summarize(observations, ("condition",))
    by_condition = {str(row["condition"]): row for row in pooled}
    expected_counts = {"Zero": 120, "Private": 432, "F9": 120}
    if {key: int(by_condition[key]["n"]) for key in expected_counts} != expected_counts:
        raise RuntimeError("combined condition counts do not match 120/432/120")
    task_rows = _final_task_rows(observations, source_root)
    private_views = _private_view_rows(observations)
    if len(private_views) != 72 or any(int(row["n"]) != 6 for row in private_views):
        raise RuntimeError(
            "private view table must contain 72 views with six repetitions each"
        )
    associations = _associations(private_views)
    strict, recommendation, pathologies = _recommendation(
        config, by_condition, task_rows
    )
    separations = {
        "full_minus_private": float(by_condition["F9"]["truth_rate"])
        - float(by_condition["Private"]["truth_rate"]),
        "private_minus_zero": float(by_condition["Private"]["truth_rate"])
        - float(by_condition["Zero"]["truth_rate"]),
        "full_minus_zero": float(by_condition["F9"]["truth_rate"])
        - float(by_condition["Zero"]["truth_rate"]),
    }

    write_csv(root / "behavioral_validation/new_observation_level_results.csv", new)
    write_csv(
        root / "behavioral_validation/combined_observation_level_results.csv",
        observations,
    )
    write_csv(root / "behavioral_validation/pooled_summary.csv", pooled)
    write_csv(root / "behavioral_validation/per_task_summary.csv", task_rows)
    write_csv(
        root / "behavioral_validation/per_task_agent_private_summary.csv", private_views
    )
    write_csv(root / "analysis/tables/final_zero_private_full.csv", pooled)
    write_csv(root / "analysis/tables/task_heterogeneity.csv", task_rows)
    write_csv(root / "analysis/tables/agent_view_heterogeneity.csv", private_views)
    write_csv(root / "analysis/tables/symbolic_empirical_association.csv", associations)
    _figures(root, pooled, task_rows, private_views)

    existing_rows = existing_task_diagnostic(source_root)
    display_pooled = []
    for condition in ("Zero", "Private", "F9"):
        row = by_condition[condition]
        display_pooled.append(
            {
                "Condition": "Full F9" if condition == "F9" else condition,
                "n": row["n"],
                "Truth": row["truth"],
                "Truth rate": f"{float(row['truth_rate']):.1%}",
                "95% CI": f"[{float(row['ci95_low']):.1%}, {float(row['ci95_high']):.1%}]",
            }
        )
    display_existing = [
        {
            "Task": row["task_id"],
            "Gold": row["gold"],
            "Margin": row["margin"],
            "Zero": f"{float(row['zero_truth_rate']):.1%}",
            "Private": f"{float(row['private_truth_rate']):.1%}",
            "Full": f"{float(row['full_truth_rate']):.1%}",
            "Worst private M": f"{float(row['worst_private_M']):.3f}",
        }
        for row in existing_rows
    ]
    display_tasks = [
        {
            "Task": row["task_id"],
            "Gold": row["gold"],
            "Margin": row["margin"],
            "Zero": f"{float(row['zero_truth_rate']):.1%}",
            "Private": f"{float(row['private_truth_rate']):.1%}",
            "Full": f"{float(row['full_truth_rate']):.1%}",
            "Private agents mean/min/max": (
                f"{float(row['private_agent_mean']):.1%} / "
                f"{float(row['private_agent_min']):.1%} / "
                f"{float(row['private_agent_max']):.1%}"
            ),
            "Worst private M": f"{float(row['worst_private_M']):.3f}",
        }
        for row in task_rows
    ]
    existing_highest_private = max(
        existing_rows, key=lambda row: float(row["private_truth_rate"])
    )
    highest_private = max(task_rows, key=lambda row: float(row["private_truth_rate"]))
    most_full_failures = min(task_rows, key=lambda row: float(row["full_truth_rate"]))
    m_assoc = next(
        row for row in associations if row["symbolic_metric"] == "symbolic_M"
    )
    h_assoc = next(
        row for row in associations if row["symbolic_metric"] == "symbolic_Hbar"
    )
    margin_values = [float(row["margin"]) for row in task_rows]
    full_values = [float(row["full_truth_rate"]) for row in task_rows]
    margin_correlation = _correlation(margin_values, full_values)
    source_journal = source_root / "behavioral_validation/raw_calls.jsonl"
    new_journal = root / "behavioral_validation/new_raw_calls.jsonl"
    source_attempts = sum(
        row.get("event") == "request_started" for row in read(source_journal)
    )
    new_attempts = sum(
        row.get("event") == "request_started" for row in read(new_journal)
    )
    unparsed_new = sum(
        row["observation_source"] == "replication_01" and not bool(row["parse_success"])
        for row in observations
    )

    report = f"""# MuSR Symbolic Ambiguity Replication 01

## A. Motivation

The frozen calibration produced Zero = 36.7%, Private = 45.8%, and Full F9 = 80.0%. The Private condition missed its preregistered ceiling by 0.8 percentage points, while Full passed exactly at its lower bound. This replication tests stability and task heterogeneity without redesigning the benchmark.

## B. Frozen benchmark

The study retains six semantic tasks with two gold answers per allocation, private breadth k=4, symbolic M<=0.45, normalized entropy Hbar>=0.90, score margin>=2, prompt P2, Full Profile F9, and game-playing model `gwdg/openai-gpt-oss-120b`. Hidden worlds, evidence, private assignments, option shuffling, temperature, parser, and retry behavior are unchanged.

## C. Existing per-task diagnostic

{_markdown_table(display_existing, ("Task", "Gold", "Margin", "Zero", "Private", "Full", "Worst private M"))}

Before new calls, `{existing_highest_private["task_id"]}` was the strongest Private task and `task_003` caused most Full-information failures. The six tasks were therefore not uniform.

## D. Replication design

The replication appended 216 Private calls, 60 Zero calls, and 60 Full calls: 336 new calls total. The combined sample has 120 Zero, 432 Private, and 120 Full observations. Existing calls were retained unchanged. New repetition IDs are Private 3-5 and Zero/Full 10-19.

## E. Final pooled results

{_markdown_table(display_pooled, ("Condition", "n", "Truth", "Truth rate", "95% CI"))}

Chance is 1/3 = 33.3%. The confidence intervals are 95% Wilson binomial intervals.

## F. Task heterogeneity

{_markdown_table(display_tasks, ("Task", "Gold", "Margin", "Zero", "Private", "Full", "Private agents mean/min/max", "Worst private M"))}

1. The highest final Private accuracy is `{highest_private["task_id"]}` at {float(highest_private["private_truth_rate"]):.1%}; the pooled result is {"still concentrated in particular tasks" if max(float(row["private_truth_rate"]) for row in task_rows) - min(float(row["private_truth_rate"]) for row in task_rows) >= 0.25 else "not dominated by a large task spread"}.
2. The most Full-information failures occur for `{most_full_failures["task_id"]}`, whose Full accuracy is {float(most_full_failures["full_truth_rate"]):.1%}.
3. The six tasks are {"not qualitatively consistent" if max(float(row["private_truth_rate"]) for row in task_rows) - min(float(row["private_truth_rate"]) for row in task_rows) >= 0.25 or max(float(row["full_truth_rate"]) for row in task_rows) - min(float(row["full_truth_rate"]) for row in task_rows) >= 0.25 else "qualitatively consistent at this resolution"}.
4. The descriptive Pearson relation between score margin and Full accuracy is {margin_correlation if margin_correlation is not None else "undefined"}. This is not reliable because five tasks have margin 2 and only one has margin 3.
5. Symbolic M has Pearson r={m_assoc["pearson_r"]} and Spearman rho={m_assoc["spearman_rho"]} with empirical Private accuracy. Its range is narrow, so this relation is descriptive only.

## G. Agent-view heterogeneity

All 72 task-agent Private views have six repetitions. Their mean empirical accuracy is {statistics.fmean(float(row["empirical_truth_rate"]) for row in private_views):.1%}, with minimum {min(float(row["empirical_truth_rate"]) for row in private_views):.1%} and maximum {max(float(row["empirical_truth_rate"]) for row in private_views):.1%}. The full distribution and Wilson intervals are in `agent_view_heterogeneity.csv` and Figure 3.

## H. Symbolic versus empirical relation

For M, Pearson r={m_assoc["pearson_r"]} and Spearman rho={m_assoc["spearman_rho"]}. For Hbar, Pearson r={h_assoc["pearson_r"]} and Spearman rho={h_assoc["spearman_rho"]}. These associations are descriptive, not causal estimates. Six repeated calls per view reduce sampling noise compared with the original three, but only six semantic worlds and a very narrow symbolic-metric range remain.

## I. Gate and separation

The original gate is Zero <=45%, Private <=45%, Full >=80%. The strict result is **{"PASS" if strict else "FAIL"}**.

- Full - Private = {separations["full_minus_private"]:.1%}
- Private - Zero = {separations["private_minus_zero"]:.1%}
- Full - Zero = {separations["full_minus_zero"]:.1%}

## J. Final recommendation

**{recommendation}**

{"Freeze the benchmark and proceed to the first blackboard population study." if recommendation != "FAIL" else "Do not redesign automatically. The remaining problem is the observed endpoint/task pathology or inadequate pooled separation documented above."}

The broader recommendation follows the rule frozen before the new calls: strict PASS requires the original gate and no task with Full below 50%; BORDERLINE requires Zero<=45%, Private<=50%, Full>=80%, Full-Private>=25 percentage points, and no such task; otherwise the result is FAIL. Pathological Full tasks under that rule: {", ".join(pathologies) if pathologies else "none"}.

## K. Limitations

The benchmark has only six semantic worlds. Results are subject to provider stochasticity. Calls repeat identical task-agent views. The total call count is larger than the number of independent semantic worlds. Symbolic M and Hbar vary over a narrow range. The score margin has almost no variation.

## Provenance

The final dataset contains 672 observations: 336 retained and 336 new. The retained journal contains {source_attempts} provider attempts; the replication journal contains {new_attempts} provider attempts. There were {unparsed_new} completed but unparsable new response(s), retained and conservatively counted as incorrect under the frozen parser rule. New raw prompts, responses, parsed answers, mappings, provider metadata, requested seeds, token usage, and failures are archived in `behavioral_validation/new_raw_calls.jsonl`.
"""
    report_path = root / "analysis/symbolic_ambiguity_replication_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return {
        "report": str(report_path),
        "strict_gate": "PASS" if strict else "FAIL",
        "decision": recommendation,
        "pooled": pooled,
        "per_task": task_rows,
        "associations": associations,
        "separations": separations,
        "observations": len(observations),
    }


__all__ = [
    "build_outputs",
    "existing_task_diagnostic",
    "load_new_observations",
    "load_source_observations",
]
