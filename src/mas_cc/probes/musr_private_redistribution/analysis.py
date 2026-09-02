"""Structural and behavioral summaries for redistribution calibration."""

from __future__ import annotations
import csv, json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from mas_cc.musr_team_allocation_generator.validation_study import wilson_interval
from mas_cc.probes.musr_prompt_solvability.execution import read
from .assignment import view_diagnostics

ORDER = ("Zero", "NAT", "R2", "R3", "R4", "F9")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def terminal(path: Path) -> list[dict[str, Any]]:
    latest = {
        str(r["call_id"]): r
        for r in read(path)
        if r.get("event") in {"call_finished", "call_failed"}
    }
    return [latest[k] for k in sorted(latest)]


def observation_rows(
    raw: Sequence[Mapping[str, Any]], tasks: Mapping[str, Any]
) -> list[dict[str, Any]]:
    output = []
    for row in raw:
        diag = view_diagnostics(
            tasks[str(row["task_id"])],
            f"agent_{int(row.get('agent_id') or 1):03d}",
            row["evidence_ids"],
        )
        output.append(
            {
                "call_id": row["call_id"],
                "task_id": row["task_id"],
                "agent_id": row.get("agent_id"),
                "regime": row["packet_variant"],
                "repetition": row["repetition"],
                "num_cards": diag["num_cards"],
                "num_latent_values": diag["num_latent_values"],
                "latent_ids": diag["latent_ids"],
                "evidence_ids": diag["evidence_ids"],
                **{
                    k: v
                    for k, v in diag.items()
                    if "allocation_" in k or k == "num_fully_scoreable_allocations"
                },
                "semantic_option_mapping": json.dumps(
                    row["semantic_option_mapping"], sort_keys=True
                ),
                "requested_provider_seed": row["provider_seed"],
                "parsed_semantic_answer": row.get("parsed_semantic_answer"),
                "correct": bool(row.get("correct")),
                "parse_success": bool(row.get("parse_success")),
                "input_tokens": (row.get("usage") or {}).get("input_tokens"),
                "output_tokens": (row.get("usage") or {}).get("output_tokens"),
            }
        )
    return output


def summarize(
    rows: Sequence[Mapping[str, Any]], keys: Sequence[str]
) -> list[dict[str, Any]]:
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(k) for k in keys)].append(row)
    out = []
    for values, group in sorted(
        groups.items(), key=lambda x: tuple(str(v) for v in x[0])
    ):
        correct = sum(bool(r["correct"]) for r in group)
        low, high = wilson_interval(correct, len(group))
        hist = Counter(r.get("parsed_semantic_answer") for r in group)
        out.append(
            {
                **dict(zip(keys, values, strict=True)),
                "n": len(group),
                "truth": correct,
                "truth_rate": correct / len(group),
                "ci95_low": low,
                "ci95_high": high,
                "parse_rate": sum(bool(r["parse_success"]) for r in group) / len(group),
                "answer_ALLOCATION_0": hist["ALLOCATION_0"],
                "answer_ALLOCATION_1": hist["ALLOCATION_1"],
                "answer_ALLOCATION_2": hist["ALLOCATION_2"],
                "unparsed": hist[None],
            }
        )
    return out


def select_regime(development: Sequence[Mapping[str, Any]]) -> str | None:
    rates = {str(r["regime"]): float(r["truth_rate"]) for r in development}
    for regime in ("R4", "R3", "R2"):
        if rates.get(regime, 1.0) <= 0.5:
            return regime
    return None


def plots(
    regime: Sequence[Mapping[str, Any]],
    observations: Sequence[Mapping[str, Any]],
    structural: Sequence[Mapping[str, Any]],
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    ordered = [next(r for r in regime if r["regime"] == name) for name in ORDER]
    values = [r["truth_rate"] for r in ordered]
    lows = [max(0, v - r["ci95_low"]) for v, r in zip(values, ordered)]
    highs = [max(0, r["ci95_high"] - v) for v, r in zip(values, ordered)]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(ORDER, values)
    ax.errorbar(
        range(6), values, yerr=[lows, highs], fmt="none", color="black", capsize=4
    )
    ax.axhline(1 / 3, color="black", linestyle="--")
    ax.set(ylabel="Truth-selection rate", ylim=(0, 1.05))
    fig.tight_layout()
    fig.savefig(output / "regime_truth_rate.png", dpi=180)
    plt.close(fig)
    for field, name, label in (
        ("num_latent_values", "truth_vs_latent_coverage.png", "Distinct latent values"),
        ("num_cards", "truth_vs_card_count.png", "Evidence cards"),
    ):
        groups = summarize(observations, (field,))
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(
            [r[field] for r in groups], [r["truth_rate"] for r in groups], marker="o"
        )
        ax.axhline(1 / 3, color="black", linestyle="--")
        ax.set(xlabel=label, ylabel="Truth-selection rate", ylim=(0, 1.05))
        fig.tight_layout()
        fig.savefig(output / name, dpi=180)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = [r["regime"] for r in structural if r["task_id"] == "task_001"]
    values = [
        r["mean_latent_values_per_agent"]
        for r in structural
        if r["task_id"] == "task_001"
    ]
    ax.bar(labels, values)
    ax.set(ylabel="Mean latent values per agent", ylim=(0, 9))
    fig.tight_layout()
    fig.savefig(output / "latent_coverage_distribution.png", dpi=180)
    plt.close(fig)


def report(
    structural: Sequence[Mapping[str, Any]],
    regime: Sequence[Mapping[str, Any]],
    per_task: Sequence[Mapping[str, Any]],
    selected: str | None,
    heldout: Sequence[Mapping[str, Any]],
    gold_counts: Mapping[str, int],
) -> str:
    def table(rows, cols):
        return "\n".join(
            [
                "| " + " | ".join(cols) + " |",
                "| " + " | ".join("---" for _ in cols) + " |",
                *(
                    "| " + " | ".join(str(r.get(c, "")) for c in cols) + " |"
                    for r in rows
                ),
            ]
        )

    pooled_struct = []
    for name in ("NAT", "R2", "R3", "R4"):
        group = [r for r in structural if r["regime"] == name]
        pooled_struct.append(
            {
                "Regime": name,
                "Mean cards/agent": round(
                    sum(r["mean_cards_per_agent"] for r in group) / len(group), 2
                ),
                "Mean latent values/agent": round(
                    sum(r["mean_latent_values_per_agent"] for r in group) / len(group),
                    2,
                ),
                "Global 9/9?": all(r["global_9_of_9"] for r in group),
                "Min holders/value": min(r["min_holders_per_value"] for r in group),
                "Agents fully scoring all candidates": sum(
                    r["agents_fully_scoring_all_candidates"] for r in group
                ),
            }
        )
    behavior = [
        {
            "Regime": r["regime"],
            "n": r["n"],
            "Truth": r["truth"],
            "Truth rate": f"{r['truth_rate']:.1%}",
            "95% CI": f"[{r['ci95_low']:.1%}, {r['ci95_high']:.1%}]",
        }
        for r in sorted(regime, key=lambda r: ORDER.index(r["regime"]))
    ]
    held = {r["regime"]: r for r in heldout}
    passed = (
        selected is not None
        and held.get(selected, {}).get("truth_rate", 1) <= 0.5
        and held.get("F9", {}).get("truth_rate", 0) >= 0.8
        and held.get("Zero", {}).get("truth_rate", 1) <= 0.5
    )
    nat = next(r for r in pooled_struct if r["Regime"] == "NAT")
    return f"""# MuSR Private Redistribution Calibration 01\n\n## A. Motivation\n\nThe previous calibration selected P2 and F9. On held-out task 003, Zero was 60.0%, Natural private was 78.3%, and F9 was 100.0%. Redistribution tests whether lower-information behavior can be repaired without changing any task or evidence text.\n\n## B. Existing-distribution audit\n\nNatural private views expose an average of **{nat["Mean latent values/agent"]} of 9 latent values**. Across the frozen tasks, many agents can fully score at least one candidate even though none can certify the unique winner under every unknown completion. This makes excessive latent breadth a plausible explanation for high private accuracy.\n\n## C. Redistribution algorithm\n\nR2/R3/R4 give every agent exactly six existing cards while fixing latent breadth at exactly 2/3/4. R2 uses three branches from each of two values; R3 uses two branches from each of three values; R4 uses one branch from four values plus two redundant branches. Assignments are seeded, frozen before calls, collectively complete, and R4 rejects any four-value set equal to a complete candidate score.\n\n## D. Structural validation\n\n{table(pooled_struct, ("Regime", "Mean cards/agent", "Mean latent values/agent", "Global 9/9?", "Min holders/value", "Agents fully scoring all candidates"))}\n\nGold semantic allocation counts across tasks: `{dict(gold_counts)}`.\n\n## E. Behavioral results\n\n{table(behavior, ("Regime", "n", "Truth", "Truth rate", "95% CI"))}\n\nPer-task results are retained in `behavioral/summary_by_task_regime.csv`.\n\n## F. Coverage-response analysis\n\nR2/R3/R4 hold raw card count fixed at six, so differences among them isolate latent breadth more cleanly than the previous dose curve. Detailed truth summaries by latent count, card count, and candidate-term coverage are retained under `analysis/tables/`. These descriptive associations do not prove causal importance for a particular latent subset.\n\n## G. Zero-evidence semantic prior\n\nZero-evidence semantic histograms are retained in `analysis/tables/zero_semantic_preferences.csv`. Because the three worlds have one gold instance of each `ALLOCATION_*` ID, pooled accuracy is less vulnerable to a single semantic-ID preference than the previous one-task diagnostic.\n\n## H. Recommended private regime\n\nSelected development regime: **{selected or "none"}**. The rule prefers the largest regime at or below 50%: R4, then R3, then R2. Held-out results were not used for selection.\n\n## I. PASS / FAIL\n\n**{"PASS — redistribution is sufficient" if passed else "FAIL — redistribution alone does not create a clean distributed-information benchmark"}**\n\n## J. Consequence\n\n{"Freeze " + str(selected) + " and proceed to the blackboard no-control/direct-recommendation/coordination-request comparison." if passed else "Do not scale the blackboard study. A separate task/evidence-generation revision is required."}\n\n## K. Limitations\n\nThis calibration uses three semantic worlds, twelve agents, three private repetitions, ten Zero/F9 repetitions, and one stochastic provider model. Repeated calls from one task-agent view are not fully independent.\n"""


__all__ = [
    "observation_rows",
    "plots",
    "report",
    "select_regime",
    "summarize",
    "terminal",
    "write_csv",
]
