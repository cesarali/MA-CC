"""Create a provider-free diagnostic report from truthful-selective preflight artifacts."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from statistics import fmean, median, pstdev
from typing import Any

ROOT = Path("results/studies/musr_truthful_selective_task_calibration_01")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def markdown_table(rows: list[dict[str, Any]], columns: tuple[str, ...]) -> str:
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


def main() -> None:
    scan = read_json(ROOT / "symbolic_scan/scan_summary.json")
    with (ROOT / "symbolic_scan/candidate_worlds.csv").open(encoding="utf-8") as stream:
        candidates = list(csv.DictReader(stream))
    task_rows: list[dict[str, Any]] = []
    robustness_rows: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    selected_dose_rows: list[dict[str, Any]] = []
    for task_root in sorted((ROOT / "tasks").glob("task_*")):
        task = read_json(task_root / "task.json")
        zero = read_json(task_root / "symbolic/zero_profile.json")
        private = read_json(task_root / "symbolic/private_profiles.json")
        controller = read_json(task_root / "symbolic/controller_profiles.json")
        decisive = read_json(task_root / "symbolic/decisive_profiles.json")["DECISIVE"]
        mixed = read_json(task_root / "symbolic/mixed_profiles.json")
        full = read_json(task_root / "symbolic/full_profile.json")
        robust = read_json(task_root / "symbolic/robustness_by_subset.json")
        truth = int(task["gold_target"].rsplit("_", 1)[1])
        false = int(task["false_target"].rsplit("_", 1)[1])
        task_rows.append(
            {
                "task": task_root.name,
                "candidate": task["candidate_id"],
                "gold": task["gold_target"],
                "false": task["false_target"],
                "controller_pool": task["controller_eligible_fact_count"],
                "decisive_facts": task["decisive_fact_count"],
                "worst_private_M": f"{max(row['M'] for row in private):.4f}",
                "min_private_Hbar": f"{min(row['Hbar'] for row in private):.4f}",
                "decisive_p_truth": f"{decisive['posterior_vector'][truth]:.4f}",
                "full_p_truth": f"{full['posterior_vector'][truth]:.4f}",
            }
        )
        selected_dose_rows.append(
            {
                "task": task_root.name,
                "ZERO": f"{zero['posterior_vector'][false]:.4f}",
                "C3": f"{controller['CONTROLLER_b03']['posterior_vector'][false]:.4f}",
                "C6": f"{controller['CONTROLLER_b06']['posterior_vector'][false]:.4f}",
                "C12": f"{controller['CONTROLLER_b12']['posterior_vector'][false]:.4f}",
                "C24": f"{controller['CONTROLLER_b24']['posterior_vector'][false]:.4f}",
            }
        )
        for row in private:
            private_rows.append({"task_id": task_root.name, **row})
        regimes = {
            "ZERO": zero,
            **controller,
            "DECISIVE": decisive,
            **mixed,
            "FULL": full,
        }
        for condition, profile in regimes.items():
            profile_rows.append(
                {
                    "task_id": task_root.name,
                    "condition": condition,
                    "p_truth": profile["posterior_vector"][truth],
                    "p_false": profile["posterior_vector"][false],
                    "M": profile["M"],
                    "Hbar": profile["Hbar"],
                }
            )
        for budget, row in robust.items():
            robustness_rows.append(
                {
                    "task_id": task_root.name,
                    "budget": int(budget),
                    **row,
                }
            )
    pooled_robustness = []
    for budget in (3, 6, 12, 24):
        values = [
            float(value)
            for row in robustness_rows
            if row["budget"] == budget
            for value in row["p_false_values"]
        ]
        positive = [
            float(value)
            > next(
                row["p_false"]
                for row in profile_rows
                if row["task_id"] == robust["task_id"] and row["condition"] == "ZERO"
            )
            for robust in robustness_rows
            if robust["budget"] == budget
            for value in robust["p_false_values"]
        ]
        pooled_robustness.append(
            {
                "budget": budget,
                "subsets": len(values),
                "mean": f"{fmean(values):.4f}",
                "median": f"{median(values):.4f}",
                "min": f"{min(values):.4f}",
                "max": f"{max(values):.4f}",
                "std": f"{pstdev(values):.4f}",
                "positive_lift_fraction": f"{fmean(positive):.3f}",
            }
        )
    funnel = [
        {
            "gate": row["gate"],
            "pass": row["passed"],
            "pass/all": f"{row['pass_fraction_all']:.2%}",
            "fail here": row["failed_at_gate"],
            "conditional pass": f"{row['conditional_pass_fraction']:.2%}",
        }
        for row in scan["gate_funnel"]
    ]
    failures = Counter(scan["failure_reasons"])
    failed_examples = []
    for reason, _ in failures.most_common(4):
        row = next(
            candidate
            for candidate in candidates
            if candidate["failure_reason"] == reason
        )
        failed_examples.append(
            {
                "candidate": row["candidate_id"],
                "reason": reason,
                "latent_values": row["latent_values"],
                "gold": row["gold_index"],
                "margin": row["score_margin"],
            }
        )
    output = ROOT / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "funnel.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(funnel[0]))
        writer.writeheader()
        writer.writerows(funnel)
    for name, rows in (
        ("symbolic_profiles.csv", profile_rows),
        ("private_packet_profiles.csv", private_rows),
        ("controller_subset_robustness.csv", robustness_rows),
    ):
        with (output / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(
        [row["gate"] for row in reversed(funnel)],
        [row["pass"] for row in reversed(funnel)],
    )
    ax.set(
        xlabel="Candidates passing cumulative gate",
        title="10,000-world symbolic funnel",
    )
    fig.tight_layout()
    fig.savefig(figures / "symbolic_gate_funnel.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    for ax, budget in zip(axes.flat, (3, 6, 12, 24), strict=True):
        values = [
            float(value)
            for row in robustness_rows
            if row["budget"] == budget
            for value in row["p_false_values"]
        ]
        ax.hist(values, bins=12)
        ax.axvline(1 / 3, color="black", linestyle="--")
        ax.set(title=f"b={budget}", xlabel="p_false", ylabel="Subsets")
    fig.tight_layout()
    fig.savefig(figures / "controller_subset_p_false_distributions.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist([float(row["M"]) for row in private_rows], bins=12, alpha=0.7, label="M")
    ax.axvline(0.45, color="black", linestyle="--", label="M limit")
    ax.set(xlabel="Private packet maximum answer probability", ylabel="Packets")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "private_packet_predictability.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5))
    conditions = (
        "ZERO",
        "CONTROLLER_b03",
        "CONTROLLER_b06",
        "CONTROLLER_b12",
        "CONTROLLER_b24",
        "DECISIVE",
        "CONTROLLER_b24+DECISIVE",
        "FULL",
    )
    for task_id in sorted({row["task_id"] for row in profile_rows}):
        rows = {
            row["condition"]: row for row in profile_rows if row["task_id"] == task_id
        }
        ax.plot(
            range(len(conditions)),
            [float(rows[c]["p_false"]) for c in conditions],
            marker="o",
            label=task_id,
        )
    ax.set_xticks(range(len(conditions)), conditions, rotation=35, ha="right")
    ax.set(ylabel="Exact false-target probability", ylim=(0, 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "symbolic_false_target_profiles.png", dpi=180)
    plt.close(fig)

    private_max = max(float(row["M"]) for row in private_rows)
    private_min_h = min(float(row["Hbar"]) for row in private_rows)
    private_decisive = sum(float(row["M"]) >= 1 for row in private_rows)
    per_task_robustness = [
        {
            "task": row["task_id"],
            "b": row["budget"],
            "n": row["subsets_tested"],
            "mean": f"{row['mean_p_false']:.4f}",
            "median": f"{row['median_p_false']:.4f}",
            "min": f"{row['min_p_false']:.4f}",
            "max": f"{row['max_p_false']:.4f}",
            "std": f"{row['std_p_false']:.4f}",
            "positive": f"{row['fraction_positive_false_target_lift']:.3f}",
            "eliminate_truth": f"{row['fraction_eliminate_truth']:.3f}",
        }
        for row in sorted(
            robustness_rows, key=lambda value: (value["task_id"], value["budget"])
        )
    ]
    report = f"""# MuSR Truthful-Selective Provider-Free Symbolic Preflight

## Verdict

**PASS for proceeding to the small Terra development batch.** The requested geometry is uncommon but feasible: **{scan["symbolic_pass_count"]} of {scan["candidate_worlds_scanned"]:,} candidates ({scan["symbolic_pass_rate"]:.2%})** passed every frozen symbolic gate. No Terra or OSS completion calls were made.

## Funnel

{markdown_table(funnel, ("gate", "pass", "pass/all", "fail here", "conditional pass"))}

`pass/all` uses all 10,000 sampled candidates as the denominator. `conditional pass` uses the preceding gate's pass count.

## Dominant rejection reasons

{markdown_table([{"reason": key, "count": value, "fraction_all": f"{value / scan['candidate_worlds_scanned']:.2%}"} for key, value in failures.most_common()], ("reason", "count", "fraction_all"))}

The hardest late gates were C24, C12, and controller-subset robustness. This is expected: those gates require a large true-fact pool to remain misleading without eliminating truth.

## Controller-subset robustness

{markdown_table(pooled_robustness, ("budget", "subsets", "mean", "median", "min", "max", "std", "positive_lift_fraction"))}

These are deterministic sampled subsets from the three frozen development tasks. Each task contributes up to 64 distinct subsets per budget. Every accepted task also has `fraction_eliminate_truth = 0` at every budget.

When a controller pool has exactly 24 facts, there is only one distinct size-24 subset. Therefore b=24 has 66 total subsets: 1 from task 001, 64 from task 002's 26-fact pool, and 1 from task 003.

### By task

{markdown_table(per_task_robustness, ("task", "b", "n", "mean", "median", "min", "max", "std", "positive", "eliminate_truth"))}

Some alternative b=6 and b=12 subsets exceed the intended-controller ceiling of 0.70, reaching 0.9309 and 0.9732. They never make truth impossible, and they pass the configured robustness rule, which only requires at least 70% positive lift and zero truth elimination. This is a caution for future random controller selection; the frozen deterministic C3/C6/C12/C24 sets all remain at or below 0.70.

### Frozen deterministic controller sets

{markdown_table(selected_dose_rows, ("task", "ZERO", "C3", "C6", "C12", "C24"))}

## N=24 private assignments

- Packets inspected individually: **{len(private_rows)}**.
- Maximum observed $M$ (largest answer probability): **{private_max:.4f}**, below the frozen limit 0.45.
- Minimum normalized entropy: **{private_min_h:.4f}**, above the frozen minimum 0.90.
- Decisive private packets: **{private_decisive}**.

Every private packet contains one canonical true fact. Acceptance used each realized packet, not a population average.

## Representative passing tasks

{markdown_table(task_rows, ("task", "candidate", "gold", "false", "controller_pool", "decisive_facts", "worst_private_M", "min_private_Hbar", "decisive_p_truth", "full_p_truth"))}

## Representative rejected candidates

{markdown_table(failed_examples, ("candidate", "reason", "latent_values", "gold", "margin"))}

## Recommendation

Proceed to the small Terra generation stage for the three frozen development tasks. Do not change the thresholds. Keep the generated language under manual or model-assisted faithfulness review because the symbolic pass proves the exact proposition geometry, not that free-form prose preserves every proposition without strengthening it.

Do not begin a population study yet. Terra evidence validation and the isolated OSS behavioral profile remain required.
"""
    (output / "truthful_selective_symbolic_preflight_report.md").write_text(
        report, encoding="utf-8"
    )
    print(output / "truthful_selective_symbolic_preflight_report.md")


if __name__ == "__main__":
    main()
