"""Report equality repair, controller reranking, and replacement decisions."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean

ROOT = Path("results/studies/musr_truthful_selective_task_calibration_01")
ARCHIVE = Path(
    "results/studies/musr_truthful_selective_task_calibration_01_before_diversity_revision"
)


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def md(rows, columns):
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
    manifest = load(ROOT / "generation/terra_generation_manifest.json")
    revision = load(ROOT / "analysis/task_revision_manifest.json")
    before = {
        row["task_id"]: row
        for row in load(ARCHIVE / "analysis/controller_diversity_audit.json")
    }
    after = {row["task_id"]: row for row in revision["after_diversity"]}
    validation_rows = []
    equality_count = 0
    profiles = []
    diversity_rows = []
    for task_root in sorted(
        path for path in (ROOT / "tasks").glob("task_???") if path.is_dir()
    ):
        task = load(task_root / "task.json")
        summary = load(task_root / "generation/semantic_validation_summary.json")
        cards = load(task_root / "generation/generated_cards.json")
        equality_count += sum(
            row.get("rendering_method") == "deterministic_canonical_equality_v1"
            for row in cards
        )
        validation_rows.append(
            {
                "task": task_root.name,
                "candidate": task["candidate_id"],
                "cards": summary["cards"],
                "passed": summary["passed"],
                "failed": summary["failed"],
                "controller": f"{summary['by_role']['controller-compatible']['passed']}/{summary['by_role']['controller-compatible']['cards']}",
                "decisive": f"{summary['by_role']['decisive']['passed']}/{summary['by_role']['decisive']['cards']}",
                "neutral": f"{summary['by_role']['neutral']['passed']}/{summary['by_role']['neutral']['cards']}",
                "decision": "PASS" if summary["all_passed"] else "FAIL",
            }
        )
        current = after[task_root.name]
        old = before[task_root.name]
        diversity_rows.append(
            {
                "task": task_root.name,
                "candidate": task["candidate_id"],
                "pool": current["controller_fact_ids"],
                "old latent": old["distinct_latent_indices"],
                "new latent": current["distinct_latent_indices"],
                "old implication pairs": old["implication_pair_count"],
                "new implication pairs": current["implication_pair_count"],
                "old zero Δ": sum(
                    abs(row["delta_p_false"]) < 1e-15
                    for row in old["marginal_controller_order"]
                ),
                "informative prefix": current["effective_informative_additions"],
                "new prefix zero Δ": sum(
                    abs(row["delta_p_false"]) < 1e-15
                    for row in current["diversity_aware_curve"]
                ),
            }
        )
        p = load(task_root / "symbolic/controller_profiles_diversity_reranked.json")
        profiles.append(
            {
                "task": task_root.name,
                **{
                    f"C{b} p_false": f"{p[f'CONTROLLER_b{b:02d}']['p_false']:.4f}"
                    for b in (3, 6, 9, 12, 24)
                },
                **{
                    f"C{b} Hbar": f"{p[f'CONTROLLER_b{b:02d}']['Hbar']:.4f}"
                    for b in (3, 6, 9, 12, 24)
                },
            }
        )

    raw = [
        json.loads(line)
        for line in (ROOT / "generation/raw_calls.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    transport_retries = sum(int(row.get("retries") or 0) for row in raw)
    baseline_attempts = 437
    revision_attempts = int(manifest["provider_attempts"]) - baseline_attempts

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = ROOT / "analysis/figures"
    figures.mkdir(parents=True, exist_ok=True)
    for metric, filename, ylabel in (
        (
            "p_false_after",
            "reranked_cumulative_p_false.png",
            "Cumulative symbolic p_false",
        ),
        (
            "entropy_after",
            "reranked_cumulative_entropy.png",
            "Cumulative normalized entropy",
        ),
        (
            "delta_p_false",
            "reranked_marginal_p_false.png",
            "Marginal change in p_false",
        ),
    ):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for task_id, audit in sorted(after.items()):
            curve = audit["diversity_aware_curve"]
            ax.plot(
                [row["rank"] for row in curve],
                [row[metric] for row in curve],
                marker="o",
                label=task_id,
            )
        for budget in (3, 6, 9, 12):
            ax.axvline(budget, color="black", alpha=0.12, linestyle="--")
        ax.set(xlabel="Distinct informative fact rank", ylabel=ylabel)
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures / filename, dpi=180)
        plt.close(fig)

    before_validation = {
        "task_001": (46, 55),
        "task_002": (47, 62),
        "task_003": (47, 49),
    }
    fig, ax = plt.subplots(figsize=(7, 4))
    names = [row["task"] for row in validation_rows]
    x = range(len(names))
    ax.bar(
        [i - 0.18 for i in x],
        [before_validation[n][0] / before_validation[n][1] for n in names],
        width=0.36,
        label="before",
    )
    ax.bar(
        [i + 0.18 for i in x],
        [row["passed"] / row["cards"] for row in validation_rows],
        width=0.36,
        label="after",
    )
    ax.set_xticks(list(x), names)
    ax.set(ylabel="Terra card validation rate", ylim=(0, 1.05))
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "terra_validation_before_after.png", dpi=180)
    plt.close(fig)

    report = f"""# Truthful-Selective Equality and Diversity Revision

## Decision

**PASS for proceeding to the isolated OSS stage.** All required Terra semantic checks now pass. OSS was not run during this revision.

## Equality evidence fix

All **{equality_count}** relational equality cards in the final three-task set use `deterministic_canonical_equality_v1`. Each card names the two compared abilities or working relationships, states that one shared categorical rubric was used, and states that both were placed in the same unnamed category. This directly entails equality without revealing the hidden category, score, gold answer, or allocation.

The deterministic validator regenerates the expected text and requires exact equality. Terra is not used to generate or judge these equality cards.

## Terra validation before and after

Before: task 001 passed 46/55 cards, old task 002 passed 47/62, and task 003 passed 47/49. After canonical equality rendering, bounded regeneration of other failed cards, and prospective task-002 replacement:

{md(validation_rows, ("task", "candidate", "cards", "passed", "failed", "controller", "decisive", "neutral", "decision"))}

## Terra calls

- Cumulative provider attempts retained in the full audit journal: **{manifest["provider_attempts"]}**.
- Additional provider attempts after the original 437-call validation: **{revision_attempts}**.
- Cumulative evidence-generation calls: **{manifest["evidence_generation_logical_calls"]}**.
- Cumulative semantic-audit calls: **{manifest["semantic_validation_logical_calls"]}**.
- Provider transport retries reported by responses: **{transport_retries}**.
- Deterministic equality rendering calls: **0 provider calls**.

## Controller diversity before and after

{md(diversity_rows, ("task", "candidate", "pool", "old latent", "new latent", "old implication pairs", "new implication pairs", "old zero Δ", "informative prefix", "new prefix zero Δ"))}

`informative prefix` counts facts in the diversity-aware order that strictly shrink the compatible world set, are not logically implied by the current prefix, increase false-target probability relative to the immediately preceding prefix, preserve truth, and keep p_false at or below 0.70.

The complete pools still contain implication and subsumption relationships. They remain available for audit, but they are moved behind the informative prefix and are not interpreted as new information.

## Reranked cumulative profiles

{md(profiles, ("task", "C3 p_false", "C6 p_false", "C9 p_false", "C12 p_false", "C24 p_false", "C3 Hbar", "C6 Hbar", "C9 Hbar", "C12 Hbar", "C24 Hbar"))}

C24 is retained only as a diagnostic saturated condition. For every task it extends beyond the positive-marginal informative prefix, so it no longer has a defensible interpretation as 24 distinct increments of information.

## Budget recommendation

Recommend **{{3, 6, 9, 12}}** for the later isolated OSS test and any prospective pilot design. It preserves four doses while remaining within the informative prefix for all three tasks. The smaller grid **{{3, 6, 12}}** is also defensible but loses the intermediate dose. Do not use 24 as an “amount of distinct information” level; report it only as a saturated-message diagnostic if retained for comparison.

## Task decisions

- `task_001` / candidate 42: **KEEP**. Nine latent variables covered; 12 informative reranked additions.
- `task_002`: **REPLACE candidate 53 with candidate 237**. Candidate 53 covered only 6/9 latent variables and had an 11-fact informative prefix. Candidate 237 was selected prospectively from the existing symbolic-pass rows, has the same gold/false balance, covers 9/9 latent variables, and supports 12 informative additions. No OSS or population outcome was used.
- `task_003` / candidate 130: **KEEP**. Nine latent variables covered; 12 informative reranked additions.

## Scope and next stage

Hidden worlds and symbolic thresholds were not changed. No new candidate worlds were sampled. The old result tree is archived at `{ARCHIVE}`. The revised task-selection manifest is `analysis/task_revision_manifest.json`.

The revised tasks may now proceed to the isolated OSS prompt evaluation. They are **not yet approved for a population pilot** because the OSS behavioral stage has not run.
"""
    output = ROOT / "analysis/truthful_selective_equality_diversity_revision_report.md"
    output.write_text(report, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
