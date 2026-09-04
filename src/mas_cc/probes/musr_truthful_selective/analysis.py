"""Reproducible tables, plots, and acceptance report for local calibration."""

from __future__ import annotations

import csv
import json
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mas_cc.musr_team_allocation_generator.validation_study import wilson_interval
from mas_cc.probes.musr_prompt_solvability.execution import read, terminal

from .config import TruthfulSelectiveConfig


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _observations(root: Path) -> list[dict[str, Any]]:
    rows = terminal(root / "behavioral_local/raw_oss_calls.jsonl")
    output = []
    for call_id in sorted(rows):
        row = rows[call_id]
        output.append(
            {
                "call_id": call_id,
                "task_id": row["task_id"],
                "condition": row["condition"],
                "agent_id": row.get("agent_id"),
                "budget": row.get("budget"),
                "subset_id": row.get("subset_id"),
                "repetition": row["repetition"],
                "evidence_ids": "|".join(row.get("evidence_ids") or ()),
                "parsed_semantic_answer": row.get("parsed_semantic_answer"),
                "truth": bool(row.get("correct")),
                "false_target": bool(row.get("false_target_selected")),
                "parse_success": bool(row.get("parse_success")),
                "input_tokens": (row.get("usage") or {}).get("input_tokens"),
                "output_tokens": (row.get("usage") or {}).get("output_tokens"),
            }
        )
    return output


def summarize(
    rows: Sequence[Mapping[str, Any]], keys: Sequence[str]
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key) for key in keys)].append(row)
    output = []
    for values, group in sorted(
        groups.items(), key=lambda item: tuple(str(value) for value in item[0])
    ):
        truth = sum(bool(row["truth"]) for row in group)
        false = sum(bool(row["false_target"]) for row in group)
        other = len(group) - truth - false
        truth_low, truth_high = wilson_interval(truth, len(group))
        false_low, false_high = wilson_interval(false, len(group))
        other_low, other_high = wilson_interval(other, len(group))
        output.append(
            {
                **dict(zip(keys, values, strict=True)),
                "n": len(group),
                "truth_frequency": truth / len(group),
                "false_target_frequency": false / len(group),
                "third_option_frequency": other / len(group),
                "truth_ci95_low": truth_low,
                "truth_ci95_high": truth_high,
                "false_ci95_low": false_low,
                "false_ci95_high": false_high,
                "other_ci95_low": other_low,
                "other_ci95_high": other_high,
                "parse_rate": sum(bool(row["parse_success"]) for row in group)
                / len(group),
            }
        )
    return output


def _plots(root: Path, summary: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = root / "analysis/figures"
    figures.mkdir(parents=True, exist_ok=True)
    order = (
        "ZERO",
        "PRIVATE",
        "C3",
        "C6",
        "C12",
        "C24",
        "DECISIVE",
        "C3+D",
        "C6+D",
        "C12+D",
        "C24+D",
        "FULL",
    )
    pooled = {
        str(row["condition"]): row for row in summary if row.get("task_id") is None
    }
    selected = [pooled[key] for key in order if key in pooled]
    labels = [str(row["condition"]) for row in selected]
    truth = [float(row["truth_frequency"]) for row in selected]
    false = [float(row["false_target_frequency"]) for row in selected]
    fig, ax = plt.subplots(figsize=(11, 5))
    x = range(len(labels))
    truth_low = [
        value - float(row["truth_ci95_low"])
        for value, row in zip(truth, selected, strict=True)
    ]
    truth_high = [
        float(row["truth_ci95_high"]) - value
        for value, row in zip(truth, selected, strict=True)
    ]
    false_low = [
        value - float(row["false_ci95_low"])
        for value, row in zip(false, selected, strict=True)
    ]
    false_high = [
        float(row["false_ci95_high"]) - value
        for value, row in zip(false, selected, strict=True)
    ]
    ax.errorbar(
        x, truth, yerr=[truth_low, truth_high], marker="o", capsize=3, label="truth"
    )
    ax.errorbar(
        x,
        false,
        yerr=[false_low, false_high],
        marker="o",
        capsize=3,
        label="false target",
    )
    ax.set_xticks(list(x), labels, rotation=45, ha="right")
    ax.set(ylabel="Choice frequency", ylim=(0, 1.02))
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "oss_choice_frequencies_by_condition.png", dpi=180)
    plt.close(fig)

    symbolic_rows = []
    for task_dir in sorted((root / "tasks").glob("task_*")):
        profiles = {}
        for filename in (
            "zero_profile.json",
            "controller_profiles.json",
            "decisive_profiles.json",
            "mixed_profiles.json",
            "full_profile.json",
        ):
            value = json.loads(
                (task_dir / "symbolic" / filename).read_text(encoding="utf-8")
            )
            if "posterior_vector" in value:
                profiles[filename.split("_")[0].upper()] = value
            else:
                profiles.update(value)
        task = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
        truth_index = int(str(task["gold_target"]).rsplit("_", 1)[1])
        false_index = int(str(task["false_target"]).rsplit("_", 1)[1])
        for condition, profile in profiles.items():
            symbolic_rows.append(
                (
                    condition,
                    profile["posterior_vector"][truth_index],
                    profile["posterior_vector"][false_index],
                )
            )
    symbolic = {}
    for condition in {row[0] for row in symbolic_rows}:
        values = [row for row in symbolic_rows if row[0] == condition]
        symbolic[condition] = (
            statistics.fmean(row[1] for row in values),
            statistics.fmean(row[2] for row in values),
        )
    symbolic_names = {
        "ZERO": "ZERO",
        "C3": "CONTROLLER_b03",
        "C6": "CONTROLLER_b06",
        "C12": "CONTROLLER_b12",
        "C24": "CONTROLLER_b24",
        "DECISIVE": "DECISIVE",
        "C3+D": "CONTROLLER_b03+DECISIVE",
        "C6+D": "CONTROLLER_b06+DECISIVE",
        "C12+D": "CONTROLLER_b12+DECISIVE",
        "C24+D": "CONTROLLER_b24+DECISIVE",
        "FULL": "FULL",
    }
    symbolic_selected = [
        (condition, symbolic[symbolic_names[condition]])
        for condition in order
        if condition in symbolic_names and symbolic_names[condition] in symbolic
    ]
    fig, ax = plt.subplots(figsize=(11, 5))
    x = range(len(symbolic_selected))
    ax.plot(x, [value[0] for _, value in symbolic_selected], marker="o", label="truth")
    ax.plot(
        x,
        [value[1] for _, value in symbolic_selected],
        marker="o",
        label="false target",
    )
    ax.set_xticks(
        list(x),
        [condition for condition, _ in symbolic_selected],
        rotation=45,
        ha="right",
    )
    ax.set(ylabel="Exact posterior probability", ylim=(0, 1.02))
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "symbolic_posterior_profile.png", dpi=180)
    plt.close(fig)
    comparisons = (("false", 2), ("truth", 1))
    for name, position in comparisons:
        points = []
        for condition in order:
            symbolic_name = symbolic_names.get(condition)
            if condition in pooled and symbolic_name in symbolic:
                points.append(
                    (
                        symbolic[symbolic_name][position - 1],
                        float(
                            pooled[condition][
                                f"{name if name == 'truth' else 'false_target'}_frequency"
                            ]
                        ),
                        condition,
                    )
                )
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter([row[0] for row in points], [row[1] for row in points])
        for x_value, y_value, label in points:
            ax.annotate(label, (x_value, y_value), fontsize=8)
        ax.set(
            xlabel=f"Symbolic {name} probability",
            ylabel=f"OSS {name} frequency",
            xlim=(0, 1),
            ylim=(0, 1),
        )
        fig.tight_layout()
        fig.savefig(figures / f"symbolic_vs_oss_{name}_response.png", dpi=180)
        plt.close(fig)

    for metric, filename in (
        ("false_target_frequency", "controller_dose_p_false.png"),
        ("truth_frequency", "controller_dose_p_truth.png"),
    ):
        rows = [
            pooled[f"C{budget}"] for budget in (3, 6, 12, 24) if f"C{budget}" in pooled
        ]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(
            [int(row["condition"][1:]) for row in rows],
            [float(row[metric]) for row in rows],
            marker="o",
        )
        ax.set(
            xlabel="Controller fact budget",
            ylabel=metric.replace("_", " "),
            ylim=(0, 1),
        )
        fig.tight_layout()
        fig.savefig(figures / filename, dpi=180)
        plt.close(fig)

    private = [row for row in summary if row.get("agent_id") is not None]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(private)), [float(row["truth_frequency"]) for row in private])
    ax.set(xlabel="Task-agent private packet", ylabel="Truth frequency", ylim=(0, 1))
    fig.tight_layout()
    fig.savefig(figures / "private_packet_heterogeneity.png", dpi=180)
    plt.close(fig)

    alternative = [row for row in summary if row.get("subset_id") is not None]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(
        range(len(alternative)),
        [float(row["false_target_frequency"]) for row in alternative],
    )
    ax.set(
        xlabel="Alternative controller subset",
        ylabel="False-target frequency",
        ylim=(0, 1),
    )
    fig.tight_layout()
    fig.savefig(figures / "controller_subset_robustness.png", dpi=180)
    plt.close(fig)

    decisive_order = ("ZERO", "DECISIVE", "C3+D", "C6+D", "C12+D", "C24+D", "FULL")
    decisive_rows = [pooled[key] for key in decisive_order if key in pooled]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(
        range(len(decisive_rows)),
        [float(row["truth_frequency"]) for row in decisive_rows],
        marker="o",
    )
    ax.set_xticks(
        range(len(decisive_rows)),
        [str(row["condition"]) for row in decisive_rows],
        rotation=35,
        ha="right",
    )
    ax.set(ylabel="Truth frequency", ylim=(0, 1))
    fig.tight_layout()
    fig.savefig(figures / "corrective_evidence_truth_response.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    decisive_only = [
        pooled[key] for key in ("ZERO", "DECISIVE", "FULL") if key in pooled
    ]
    values = [float(row["truth_frequency"]) for row in decisive_only]
    lows = [
        value - float(row["truth_ci95_low"])
        for value, row in zip(values, decisive_only, strict=True)
    ]
    highs = [
        float(row["truth_ci95_high"]) - value
        for value, row in zip(values, decisive_only, strict=True)
    ]
    ax.errorbar(range(len(values)), values, yerr=[lows, highs], marker="o", capsize=4)
    ax.set_xticks(range(len(values)), [str(row["condition"]) for row in decisive_only])
    ax.set(ylabel="Truth frequency", ylim=(0, 1))
    fig.tight_layout()
    fig.savefig(figures / "decisive_recovery.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    mixed_only = [
        pooled[key] for key in ("C3+D", "C6+D", "C12+D", "C24+D") if key in pooled
    ]
    values = [float(row["truth_frequency"]) for row in mixed_only]
    lows = [
        value - float(row["truth_ci95_low"])
        for value, row in zip(values, mixed_only, strict=True)
    ]
    highs = [
        float(row["truth_ci95_high"]) - value
        for value, row in zip(values, mixed_only, strict=True)
    ]
    ax.errorbar(range(len(values)), values, yerr=[lows, highs], marker="o", capsize=4)
    ax.set_xticks(range(len(values)), [str(row["condition"]) for row in mixed_only])
    ax.set(ylabel="Truth frequency", ylim=(0, 1))
    fig.tight_layout()
    fig.savefig(figures / "controller_plus_decisive_recovery.png", dpi=180)
    plt.close(fig)

    full_rows = [
        row
        for row in summary
        if row.get("task_id") is not None and row["condition"] == "FULL"
    ]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(
        [str(row["task_id"]) for row in full_rows],
        [float(row["truth_frequency"]) for row in full_rows],
    )
    ax.axhline(0.8, color="black", linestyle="--", label="development minimum")
    ax.axhline(0.9, color="green", linestyle=":", label="preferred")
    ax.set(ylabel="Full-information truth frequency", ylim=(0, 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "full_information_solvability.png", dpi=180)
    plt.close(fig)

    diversity_path = root / "analysis/controller_diversity_audit.json"
    if diversity_path.is_file():
        diversity = json.loads(diversity_path.read_text(encoding="utf-8"))
        fig, ax = plt.subplots(figsize=(10, 5))
        labels, values = [], []
        for task in diversity:
            for family, count in task["relation_family_counts"].items():
                labels.append(f"{task['task_id']}\n{family}")
                values.append(int(count))
        ax.bar(range(len(values)), values)
        ax.set_xticks(range(len(labels)), labels, rotation=70, ha="right")
        ax.set(ylabel="Controller facts", title="Controller fact source coverage")
        fig.tight_layout()
        fig.savefig(figures / "controller_fact_diversity.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 5))
        for task in diversity:
            rows = task["marginal_controller_order"]
            ax.plot(
                [row["rank"] for row in rows],
                [row["delta_p_false"] for row in rows],
                marker="o",
                markersize=3,
                label=task["task_id"],
            )
        for budget in (3, 6, 12, 24):
            ax.axvline(budget, color="black", alpha=0.2, linestyle="--")
        ax.set(
            xlabel="Fact rank in frozen controller order",
            ylabel="Marginal change in symbolic p_false",
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures / "controller_marginal_symbolic_contribution.png", dpi=180)
        plt.close(fig)


def build_outputs(config: TruthfulSelectiveConfig, root: Path) -> dict[str, Any]:
    observations = _observations(root)
    by_task = summarize(observations, ("task_id", "condition"))
    pooled = summarize(observations, ("condition",))
    for row in pooled:
        row["task_id"] = None
    private = summarize(
        [row for row in observations if row["condition"] == "PRIVATE"],
        ("task_id", "condition", "agent_id"),
    )
    alternatives = summarize(
        [row for row in observations if row["subset_id"] is not None],
        ("task_id", "condition", "subset_id"),
    )
    all_summary = [*pooled, *by_task, *private, *alternatives]
    write_csv(root / "behavioral_local/observation_results.csv", observations)
    write_csv(root / "behavioral_local/condition_summary.csv", all_summary)
    write_csv(root / "analysis/condition_summary.csv", all_summary)
    symbolic_behavioral = []
    condition_key = {
        "ZERO": "ZERO",
        "C3": "CONTROLLER_b03",
        "C6": "CONTROLLER_b06",
        "C12": "CONTROLLER_b12",
        "C24": "CONTROLLER_b24",
        "DECISIVE": "DECISIVE",
        "C3+D": "CONTROLLER_b03+DECISIVE",
        "C6+D": "CONTROLLER_b06+DECISIVE",
        "C12+D": "CONTROLLER_b12+DECISIVE",
        "C24+D": "CONTROLLER_b24+DECISIVE",
        "FULL": "FULL",
    }
    for task_dir in sorted((root / "tasks").glob("task_*")):
        task = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
        truth_index = int(task["gold_target"].rsplit("_", 1)[1])
        false_index = int(task["false_target"].rsplit("_", 1)[1])
        profiles = {
            "ZERO": json.loads(
                (task_dir / "symbolic/zero_profile.json").read_text(encoding="utf-8")
            ),
            "DECISIVE": json.loads(
                (task_dir / "symbolic/decisive_profiles.json").read_text(
                    encoding="utf-8"
                )
            )["DECISIVE"],
            "FULL": json.loads(
                (task_dir / "symbolic/full_profile.json").read_text(encoding="utf-8")
            ),
            **json.loads(
                (task_dir / "symbolic/controller_profiles.json").read_text(
                    encoding="utf-8"
                )
            ),
            **json.loads(
                (task_dir / "symbolic/mixed_profiles.json").read_text(encoding="utf-8")
            ),
        }
        empirical = {
            str(row["condition"]): row
            for row in by_task
            if row["task_id"] == task_dir.name
        }
        for condition, symbolic_name in condition_key.items():
            row = empirical[condition]
            profile = profiles[symbolic_name]
            symbolic_behavioral.append(
                {
                    "task_id": task_dir.name,
                    "condition": condition,
                    "symbolic_p_truth": profile["posterior_vector"][truth_index],
                    "oss_p_truth": row["truth_frequency"],
                    "symbolic_p_false": profile["posterior_vector"][false_index],
                    "oss_p_false": row["false_target_frequency"],
                    "oss_p_other": row["third_option_frequency"],
                    "parse_rate": row["parse_rate"],
                }
            )
        private_profiles = json.loads(
            (task_dir / "symbolic/private_profiles.json").read_text(encoding="utf-8")
        )
        private_empirical = {
            str(row["agent_id"]): row
            for row in private
            if row["task_id"] == task_dir.name
        }
        for profile in private_profiles:
            row = private_empirical[str(profile["agent_id"])]
            symbolic_behavioral.append(
                {
                    "task_id": task_dir.name,
                    "condition": f"PRIVATE:{profile['agent_id']}",
                    "symbolic_p_truth": profile["posterior_vector"][truth_index],
                    "oss_p_truth": row["truth_frequency"],
                    "symbolic_p_false": profile["posterior_vector"][false_index],
                    "oss_p_false": row["false_target_frequency"],
                    "oss_p_other": row["third_option_frequency"],
                    "parse_rate": row["parse_rate"],
                }
            )
    write_csv(root / "analysis/symbolic_vs_oss.csv", symbolic_behavioral)
    _plots(root, all_summary)

    task_decisions = []
    for task_dir in sorted((root / "tasks").glob("task_*")):
        task_id = task_dir.name
        rows = {
            str(row["condition"]): row for row in by_task if row["task_id"] == task_id
        }
        zero = rows.get("ZERO", {})
        private_rows = [row for row in private if row["task_id"] == task_id]
        controller_rows = [
            rows.get(f"C{budget}", {}) for budget in config.symbolic.controller_budgets
        ]
        private_behaviorally_decisive = any(
            max(
                float(row["truth_frequency"]),
                float(row["false_target_frequency"]),
                float(row["third_option_frequency"]),
            )
            >= 0.80
            for row in private_rows
        )
        gates = {
            "zero_ambiguous": bool(
                config.behavioral.zero_min_truth
                <= float(zero.get("truth_frequency", 2))
                <= config.behavioral.zero_max_truth
                and config.behavioral.zero_min_false
                <= float(zero.get("false_target_frequency", 2))
                <= config.behavioral.zero_max_false
            ),
            "private_ambiguous": bool(
                statistics.fmean(float(row["truth_frequency"]) for row in private_rows)
                <= config.behavioral.private_max_truth
                and statistics.fmean(
                    float(row["false_target_frequency"]) for row in private_rows
                )
                <= config.behavioral.private_max_false
                and not private_behaviorally_decisive
            ),
            "controller_lift": any(
                float(row.get("false_target_frequency", 0))
                > float(zero.get("false_target_frequency", 1))
                for row in controller_rows
            ),
            "controller_partial": all(
                float(row.get("false_target_frequency", 1))
                < config.behavioral.controller_max_false
                for row in controller_rows
            ),
            "decisive_recovery": float(
                rows.get("DECISIVE", {}).get("truth_frequency", 0)
            )
            >= config.behavioral.decisive_min_truth,
            "mixed_recovery": all(
                float(rows.get(f"C{budget}+D", {}).get("truth_frequency", 0))
                >= config.behavioral.mixed_min_truth
                for budget in config.symbolic.controller_budgets
            ),
            "full_recovery_development": float(
                rows.get("FULL", {}).get("truth_frequency", 0)
            )
            >= config.behavioral.development_full_min_truth,
            "full_recovery_preferred": float(
                rows.get("FULL", {}).get("truth_frequency", 0)
            )
            >= config.behavioral.full_min_truth,
            "parse_complete": all(
                float(row["parse_rate"]) == 1.0 for row in rows.values()
            ),
        }
        passed = bool(
            rows
            and all(
                value
                for key, value in gates.items()
                if key != "full_recovery_preferred"
            )
        )
        task_decisions.append(
            {
                "task_id": task_id,
                "decision": "PASS" if passed else "FAIL",
                "gates": gates,
            }
        )
    decision = (
        "PASS"
        if task_decisions and all(row["decision"] == "PASS" for row in task_decisions)
        else "FAIL"
    )
    scan = json.loads(
        (root / "symbolic_scan/scan_summary.json").read_text(encoding="utf-8")
    )
    generation = json.loads(
        (root / "generation/terra_generation_manifest.json").read_text(encoding="utf-8")
    )
    diversity = json.loads(
        (root / "analysis/controller_diversity_audit.json").read_text(encoding="utf-8")
    )
    journal = read(root / "behavioral_local/raw_oss_calls.jsonl")
    attempts = sum(row.get("event") == "request_started" for row in journal)
    parse_rate = (
        sum(row["parse_success"] for row in observations) / len(observations)
        if observations
        else 0
    )
    report = f"""# Truthful Selective Task Calibration Report

## 1. Scientific objective

Construct truthful Team Allocation tasks where selected true facts raise a fixed false target, while decisive true facts and full information recover the unique truth.

## 2. Exact Team Allocation world

The scan used the repository's exact nine-value Team Allocation prior and exact scorer. It scanned **{scan["candidate_worlds_scanned"]:,}** worlds.

## 3. Truthful selective-disclosure semantics

Every controller statement is a canonical proposition that evaluates true in the same frozen hidden world. No controller text is generated at runtime.

## 4. Posterior-profile definitions

Each task stores the complete three-answer posterior, maximum answer probability, normalized entropy, and compatible-world count for ZERO, every private packet, C3/C6/C12/C24, DECISIVE, every controller-plus-decisive condition, and FULL.

## 5. Symbolic candidate scan

- Passes: **{scan["symbolic_pass_count"]}** ({scan["symbolic_pass_rate"]:.2%})
- Failure reasons: `{json.dumps(scan["failure_reasons"], sort_keys=True)}`

## 6. Controller-compatible fact geometry

Each selected task has at least 24 logically distinct facts. Exact predicate signatures prevent paraphrases from counting as new facts.

`analysis/controller_diversity_audit.json` records latent-source coverage, predicate-family counts, implication/subsumption pairs, and every marginal change in symbolic false-target probability.

## 7. Decisive corrective facts

The decisive set and every C-budget-plus-decisive set recover the gold answer with symbolic probability 1.

## 8. Terra generation validation

Terra model: `{generation["model"]}`. Logical calls: {generation["logical_calls"]}. Provider attempts: {generation["provider_attempts"]}. Retries: {generation["retry_count"]}. Validation: **{generation["generation_validation_status"]}**.

## 9. Exact local gameplay prompt

The local test uses the production P2 ballot renderer and parser with no board messages, controller identity, current vote, or social history.

## 10. OSS local stress-test results

OSS attempts: {attempts}. Parse rate: {parse_rate:.2%}.

## 11. Symbolic vs behavioral comparison

See `analysis/symbolic_vs_oss.csv`, `analysis/figures/symbolic_vs_oss_false_response.png`, and `analysis/figures/symbolic_vs_oss_truth_response.png`.

## 12. Private-packet heterogeneity

See `analysis/figures/private_packet_heterogeneity.png` and `behavioral_local/condition_summary.csv`.

## 13. Controller-dose response

See the two controller-dose plots under `analysis/figures/`.

## 14. Corrective-evidence response

DECISIVE and each C+D condition are reported in the condition summary.

## 15. Robustness to controller subset choice

Alternative subset outcomes are retained by task and subset ID.

## Controller fact diversity summary

{json.dumps([{key: row[key] for key in ("task_id", "controller_fact_ids", "distinct_logical_signatures", "distinct_latent_indices", "implication_pair_count")} for row in diversity], indent=2)}

## 16. Acceptance thresholds

Thresholds were frozen in the copied run configuration before provider calls.

## 17. PASS / FAIL per development task

{json.dumps(task_decisions, indent=2)}

## 18. Recommended production task-bank specification

Accept the first 9–12 future tasks passing the same symbolic and behavioral gates, balanced across gold and false semantic targets.

## 19. Limitations

Generated prose receives deterministic structural and leakage validation. Exact semantic faithfulness remains auditable through the stored canonical proposition and branch/leaf provenance and should be reviewed before production freezing.

## Overall decision

**{decision}**
"""
    report_path = root / "analysis/truthful_selective_task_calibration_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    for task_dir in sorted((root / "tasks").glob("task_*")):
        local = task_dir / "behavioral_local"
        local.mkdir(parents=True, exist_ok=True)
        task_id = task_dir.name
        write_csv(
            local / "observation_results.csv",
            [row for row in observations if row["task_id"] == task_id],
        )
        write_csv(
            local / "condition_summary.csv",
            [row for row in all_summary if row.get("task_id") == task_id],
        )
        (local / "local_stress_test_report.md").write_text(
            f"# {task_id} local stress test\n\nDecision: **{next(row['decision'] for row in task_decisions if row['task_id'] == task_id)}**\n",
            encoding="utf-8",
        )
        with (local / "raw_oss_calls.jsonl").open("w", encoding="utf-8") as stream:
            for row in journal:
                if row.get("task_id") == task_id:
                    stream.write(
                        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                    )
        source_prompts = root / "behavioral_local/prompt_examples.md"
        if source_prompts.is_file():
            (local / "prompt_examples.md").write_text(
                source_prompts.read_text(encoding="utf-8"), encoding="utf-8"
            )
    return {
        "report": str(report_path),
        "decision": decision,
        "tasks": task_decisions,
        "parse_rate": parse_rate,
    }


__all__ = ["build_outputs", "summarize", "write_csv"]
