"""Selection rules, summaries, plots, and paper-facing report."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mas_cc.musr_team_allocation_generator.validation_study import wilson_interval
from .execution import read


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def rows(path: Path) -> list[dict[str, Any]]:
    latest = {
        str(row["call_id"]): row
        for row in read(path)
        if row.get("event") in {"call_finished", "call_failed"}
    }
    return [latest[key] for key in sorted(latest)]


def summarize(
    data: Sequence[Mapping[str, Any]], keys: Sequence[str]
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in data:
        groups[tuple(row.get(k) for k in keys)].append(row)
    output = []
    for values, group in sorted(
        groups.items(), key=lambda x: tuple(str(v) for v in x[0])
    ):
        correct = sum(bool(row.get("correct")) for row in group)
        parsed = sum(bool(row.get("parse_success")) for row in group)
        low, high = wilson_interval(correct, len(group))
        hist = Counter(row.get("parsed_semantic_answer") for row in group)
        output.append(
            {
                **dict(zip(keys, values, strict=True)),
                "n": len(group),
                "truth": correct,
                "truth_rate": correct / len(group),
                "ci95_low": low,
                "ci95_high": high,
                "parse_successes": parsed,
                "parse_rate": parsed / len(group),
                "answer_ALLOCATION_0": hist["ALLOCATION_0"],
                "answer_ALLOCATION_1": hist["ALLOCATION_1"],
                "answer_ALLOCATION_2": hist["ALLOCATION_2"],
                "unparsed": hist[None],
                "mean_input_tokens": sum(
                    int((row.get("usage") or {}).get("input_tokens") or 0)
                    for row in group
                )
                / len(group),
                "mean_output_tokens": sum(
                    int((row.get("usage") or {}).get("output_tokens") or 0)
                    for row in group
                )
                / len(group),
            }
        )
    return output


def select_prompt(summary: Sequence[Mapping[str, Any]]) -> str:
    pooled = list(summary)
    eligible = [row for row in pooled if float(row["parse_rate"]) >= 0.95]
    if not eligible:
        raise RuntimeError("no prompt variant reached 95% parse success")
    best = max(float(row["truth_rate"]) for row in eligible)
    near = [row for row in eligible if best - float(row["truth_rate"]) <= 0.0500000001]
    simplicity = {"P0": 0, "P2": 1, "P1": 2, "P3": 3}
    return str(
        min(near, key=lambda row: simplicity[str(row["prompt_variant"])])[
            "prompt_variant"
        ]
    )


def select_packet(summary: Sequence[Mapping[str, Any]]) -> str:
    eligible = [
        row
        for row in summary
        if float(row["truth_rate"]) >= 0.8 and float(row["parse_rate"]) >= 0.95
    ]
    if eligible:
        return str(
            min(
                eligible,
                key=lambda row: {"F9": 9, "F18": 18, "F27": 27}[
                    str(row["packet_variant"])
                ],
            )["packet_variant"]
        )
    return str(max(summary, key=lambda row: float(row["truth_rate"]))["packet_variant"])


def render_plots(
    prompt: Sequence[Mapping[str, Any]],
    packets: Sequence[Mapping[str, Any]],
    heldout: Sequence[Mapping[str, Any]],
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)

    def bar(rows, labels, path, xlabel):
        values = [float(r["truth_rate"]) for r in rows]
        lows = [max(0.0, values[i] - float(r["ci95_low"])) for i, r in enumerate(rows)]
        highs = [
            max(0.0, float(r["ci95_high"]) - values[i]) for i, r in enumerate(rows)
        ]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar(labels, values)
        ax.errorbar(
            range(len(rows)),
            values,
            yerr=[lows, highs],
            fmt="none",
            color="black",
            capsize=4,
        )
        ax.axhline(1 / 3, color="black", linestyle="--")
        ax.set(xlabel=xlabel, ylabel="Truth-selection rate", ylim=(0, 1.05))
        fig.tight_layout()
        fig.savefig(output / path, dpi=180)
        plt.close(fig)

    ordered = sorted(prompt, key=lambda r: str(r["prompt_variant"]))
    bar(
        ordered,
        [r["prompt_variant"] for r in ordered],
        "prompt_ablation_truth_rate.png",
        "Prompt variant",
    )
    ordered = sorted(
        packets, key=lambda r: {"F9": 9, "F18": 18, "F27": 27}[str(r["packet_variant"])]
    )
    bar(
        ordered,
        [r["packet_variant"] for r in ordered],
        "full_profile_packet_truth_rate.png",
        "Full-profile packet",
    )
    order = [
        next(r for r in heldout if r["condition"] == name)
        for name in ("zero", "private", "full")
    ]
    bar(
        order,
        ["Zero", "Private", "Full"],
        "zero_private_full_separation.png",
        "Held-out evidence regime",
    )


def report(
    task_meta: Sequence[Mapping[str, Any]],
    prompt_summary: Sequence[Mapping[str, Any]],
    prompt_task: Sequence[Mapping[str, Any]],
    selected_prompt: str,
    packet_summary: Sequence[Mapping[str, Any]],
    selected_packet: str,
    heldout: Sequence[Mapping[str, Any]],
    examples: Mapping[str, str],
    packet_defs: Mapping[str, Any],
) -> str:
    def table(rows, columns):
        return "\n".join(
            [
                "| " + " | ".join(columns) + " |",
                "| " + " | ".join("---" for _ in columns) + " |",
                *(
                    "| " + " | ".join(str(row.get(c, "")) for c in columns) + " |"
                    for row in rows
                ),
            ]
        )

    full = next(row for row in heldout if row["condition"] == "full")
    zero = next(row for row in heldout if row["condition"] == "zero")
    private = next(row for row in heldout if row["condition"] == "private")
    passed = (
        float(full["truth_rate"]) >= 0.8
        and float(zero["truth_rate"]) <= 0.5
        and float(private["truth_rate"]) <= 0.5
    )
    prompt_rows = [
        {
            "Prompt": r["prompt_variant"],
            "Social caution": "yes" if r["prompt_variant"] in {"P0", "P1"} else "no",
            "Decision scaffold": "yes" if r["prompt_variant"] in {"P1", "P3"} else "no",
            "n": r["n"],
            "Truth": r["truth"],
            "Truth rate": f"{r['truth_rate']:.1%}",
            "95% CI": f"[{r['ci95_low']:.1%}, {r['ci95_high']:.1%}]",
            "Parse rate": f"{r['parse_rate']:.1%}",
        }
        for r in prompt_summary
    ]
    packet_rows = [
        {
            "Packet": r["packet_variant"],
            "Cards": {"F9": 9, "F18": 18, "F27": 27}[r["packet_variant"]],
            "Latent breadth": "9/9",
            "n": r["n"],
            "Truth rate": f"{r['truth_rate']:.1%}",
            "95% CI": f"[{r['ci95_low']:.1%}, {r['ci95_high']:.1%}]",
        }
        for r in packet_summary
    ]
    held = [
        {
            "Task": r.get("task_id", "pooled"),
            "Condition": r["condition"].title(),
            "Evidence regime": {
                "zero": "none",
                "private": "natural distributed",
                "full": selected_packet,
            }[r["condition"]],
            "n": r["n"],
            "Truth rate": f"{r['truth_rate']:.1%}",
            "95% CI": f"[{r['ci95_low']:.1%}, {r['ci95_high']:.1%}]",
        }
        for r in heldout
    ]
    return f"""# MuSR Prompt Solvability Calibration 01\n\n## A. Motivation\n\nLocal full-information solvability is required before population failures can be interpreted as coordination failures. Existing diagnostics were: round-zero natural private 30.6%, zero evidence 33.3%, and a noisy 27-card local-probe result of 66.7%.\n\n## B. Task structure\n\nA hidden skill/cooperation matrix determines exact candidate scores and one gold allocation. Natural-language evidence cards indirectly express those latent values. Hidden scores are evaluation-only and never enter prompts. Development tasks were `task_001` and `task_002`; held-out `task_003` was not used for prompt or packet selection.\n\nScore-margin metadata:\n\n{table(task_meta, ("task_id", "gold_answer", "gold_score", "second_best_score", "score_margin"))}\n\n## C. Prompt variants\n\n- **P0:** exact current game initialization prompt.\n- **P1:** P0 plus the explicit allocation-comparison scaffold.\n- **P2:** P0 with round-zero social caution neutralized.\n- **P3:** P2 plus the scaffold.\n\n### P0 complete example\n\n```text\n{examples["P0"]}\n```\n\n### P1 complete example\n\n```text\n{examples["P1"]}\n```\n\n### P2 complete example\n\n```text\n{examples["P2"]}\n```\n\n### P3 complete example\n\n```text\n{examples["P3"]}\n```\n\n## D. Prompt-ablation results\n\n{table(prompt_rows, ("Prompt", "Social caution", "Decision scaffold", "n", "Truth", "Truth rate", "95% CI", "Parse rate"))}\n\nPer-task results are retained in `prompt_ablation/summary_by_prompt_task.csv`.\n\n## E. Prompt selection\n\nThe frozen prompt is **{selected_prompt}**. Selection used development tasks only: highest pooled truth rate subject to parse rate at least 95%; variants within five percentage points were resolved toward the simpler prompt.\n\n## F. Full Profile packet ablation\n\n{table(packet_rows, ("Packet", "Cards", "Latent breadth", "n", "Truth rate", "95% CI"))}\n\nExact deterministic packets are archived in `tasks/full_profile_packets.json`. Example development-task F9 IDs:\n\n```text\n{chr(10).join(packet_defs["task_001"]["F9"])}\n```\n\n## G. Full Profile definition\n\nThe frozen Full Profile packet is **{selected_packet}**. The rule chooses the smallest packet with development pooled truth at least 80% and parse rate at least 95%; if none passes, it freezes the best observed packet and the held-out gate may fail.\n\n## H. Held-out zero/private/full validation\n\n{table(held, ("Task", "Condition", "Evidence regime", "n", "Truth rate", "95% CI"))}\n\n## I. Acceptance decision\n\n**{"PASS — benchmark satisfies local solvability requirement" if passed else "FAIL — benchmark is not yet ready for blackboard population experiments"}**\n\nHeld-out truth rates were zero {zero["truth_rate"]:.1%}, natural private {private["truth_rate"]:.1%}, and full {full["truth_rate"]:.1%}. The gate requires held-out full truth at least 80% and both zero and natural-private truth no greater than 50% as a conservative operational meaning of substantially lower; 90% or higher full truth is preferred.\n\n## J. Interpretation for the paper\n\n{"The local model solved the held-out task reliably under the frozen Full Profile while lower-information regimes remained lower. This supports interpreting later population behavior primarily through information aggregation and coordination." if passed else "The held-out Full Profile was solvable, but the lower-information regimes did not remain sufficiently low. This task/prompt combination is not yet a clean blackboard benchmark."}\n\n## K. Limitations\n\nThere were two development tasks and one held-out task. Prompt development occurred only on development tasks. Provider behavior is stochastic, and the held-out set contains only one semantic world. All intervals and plots are descriptive.\n"""


__all__ = [
    "render_plots",
    "report",
    "rows",
    "select_packet",
    "select_prompt",
    "summarize",
    "write_csv",
]
