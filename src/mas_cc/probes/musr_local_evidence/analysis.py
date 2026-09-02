"""Tables, descriptive figures, and paired outcomes for the MuSR probe."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mas_cc.musr_team_allocation_generator.validation_study import wilson_interval


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def terminal_rows(raw: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in raw:
        if row.get("event") in {"call_finished", "call_failed"}:
            latest[str(row["call_id"])] = dict(row)
    return [latest[key] for key in sorted(latest)]


def comparison_family(row: Mapping[str, Any]) -> str:
    family = str(row.get("prompt_family", ""))
    if family == "musr_team_allocation_validation":
        return "validation"
    if family == "relational_public_ballot":
        return "game_init"
    return family


def summarize_prompt_equivalence(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    terminal = terminal_rows(rows)
    pairs: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in terminal:
        pairs[str(row["pair_id"])][comparison_family(row)] = row
    paired: list[dict[str, Any]] = []
    for pair_id, values in sorted(pairs.items()):
        validation, game = values.get("validation"), values.get("game_init")
        evaluable = bool(
            validation
            and game
            and validation.get("parse_success")
            and game.get("parse_success")
        )
        paired.append(
            {
                "pair_id": pair_id,
                "agent_id": None if validation is None else validation.get("agent_id"),
                "repetition": None
                if validation is None
                else validation.get("repetition"),
                "evaluable": evaluable,
                "validation_answer": None
                if validation is None
                else validation.get("parsed_semantic_answer"),
                "game_init_answer": None
                if game is None
                else game.get("parsed_semantic_answer"),
                "validation_correct": bool(validation and validation.get("correct")),
                "game_init_correct": bool(game and game.get("correct")),
                "agreement": bool(
                    evaluable
                    and validation.get("parsed_semantic_answer")
                    == game.get("parsed_semantic_answer")
                ),
                "validation_correct_game_wrong": bool(
                    evaluable and validation.get("correct") and not game.get("correct")
                ),
                "game_correct_validation_wrong": bool(
                    evaluable and game.get("correct") and not validation.get("correct")
                ),
            }
        )
    summaries = []
    for agent in sorted({row["agent_id"] for row in paired}):
        group = [row for row in paired if row["agent_id"] == agent]
        evaluable = [row for row in group if row["evaluable"]]
        validation = [
            row
            for row in terminal
            if row.get("agent_id") == agent and comparison_family(row) == "validation"
        ]
        game = [
            row
            for row in terminal
            if row.get("agent_id") == agent and comparison_family(row) == "game_init"
        ]
        summaries.append(_pair_summary(agent, group, evaluable, validation, game))
    summaries.append(
        _pair_summary(
            "pooled",
            paired,
            [r for r in paired if r["evaluable"]],
            [r for r in terminal if comparison_family(r) == "validation"],
            [r for r in terminal if comparison_family(r) == "game_init"],
        )
    )
    return paired, summaries


def _pair_summary(
    agent: Any,
    group: Sequence[Mapping[str, Any]],
    evaluable: Sequence[Mapping[str, Any]],
    validation: Sequence[Mapping[str, Any]],
    game: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    def correct_rate(rows: Sequence[Mapping[str, Any]]) -> float:
        return (
            sum(bool(row.get("correct")) for row in rows) / len(rows) if rows else 0.0
        )

    return {
        "agent_id": agent,
        "scheduled_pairs": len(group),
        "evaluable_pairs": len(evaluable),
        "validation_truth_rate": correct_rate(validation),
        "game_init_truth_rate": correct_rate(game),
        "paired_disagreement_rate": (
            sum(not row["agreement"] for row in evaluable) / len(evaluable)
            if evaluable
            else 0.0
        ),
        "exact_agreement_count": sum(bool(row["agreement"]) for row in evaluable),
        "validation_correct_game_wrong": sum(
            bool(row["validation_correct_game_wrong"]) for row in evaluable
        ),
        "game_correct_validation_wrong": sum(
            bool(row["game_correct_validation_wrong"]) for row in evaluable
        ),
        "validation_parse_rate": sum(
            bool(row.get("parse_success")) for row in validation
        )
        / len(validation)
        if validation
        else 0.0,
        "game_init_parse_rate": sum(bool(row.get("parse_success")) for row in game)
        / len(game)
        if game
        else 0.0,
        **{f"validation_{key}": value for key, value in _hist(validation).items()},
        **{f"game_init_{key}": value for key, value in _hist(game).items()},
    }


def _hist(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(row.get("parsed_semantic_answer") for row in rows)
    return {
        "ALLOCATION_0": counts["ALLOCATION_0"],
        "ALLOCATION_1": counts["ALLOCATION_1"],
        "ALLOCATION_2": counts["ALLOCATION_2"],
        "unparsed": counts[None],
    }


def summarize_doses(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    terminal = terminal_rows(rows)
    observations = []
    for row in terminal:
        observations.append(
            {
                "call_id": row["call_id"],
                "agent_id": row["agent_id"],
                "dose": row["dose"],
                "repetition": row["repetition"],
                "evidence_ids": "|".join(row["evidence_ids"]),
                "distinct_latent_facts": row.get("distinct_latent_facts"),
                "parsed_semantic_answer": row.get("parsed_semantic_answer"),
                "correct": bool(row.get("correct")),
                "parse_success": bool(row.get("parse_success")),
                "input_tokens": (row.get("usage") or {}).get("input_tokens"),
                "output_tokens": (row.get("usage") or {}).get("output_tokens"),
            }
        )
    summary = []
    for dose in sorted({int(row["dose"]) for row in terminal}):
        group = [row for row in terminal if int(row["dose"]) == dose]
        correct = sum(bool(row.get("correct")) for row in group)
        low, high = wilson_interval(correct, len(group))
        counts = _hist(group)
        summary.append(
            {
                "cards": dose,
                "mean_latent_facts_covered": sum(
                    int(row.get("distinct_latent_facts", 0)) for row in group
                )
                / len(group),
                "n": len(group),
                "truth_choices": correct,
                "truth_rate": correct / len(group),
                "ci95_low": low,
                "ci95_high": high,
                "parse_successes": sum(bool(row.get("parse_success")) for row in group),
                "parse_rate": sum(bool(row.get("parse_success")) for row in group)
                / len(group),
                **{f"answer_{key}": value for key, value in counts.items()},
            }
        )
    return observations, summary


def summarize_doses_by_agent(
    observations: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    keys = sorted({(int(row["agent_id"]), int(row["dose"])) for row in observations})
    for agent, dose in keys:
        group = [
            row
            for row in observations
            if int(row["agent_id"]) == agent and int(row["dose"]) == dose
        ]
        correct = sum(bool(row["correct"]) for row in group)
        low, high = wilson_interval(correct, len(group))
        rows.append(
            {
                "agent_id": agent,
                "cards": dose,
                "distinct_latent_facts": group[0]["distinct_latent_facts"],
                "n": len(group),
                "truth_choices": correct,
                "truth_rate": correct / len(group),
                "ci95_low": low,
                "ci95_high": high,
            }
        )
    return rows


def summarize_by_latent_coverage(
    observations: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for coverage in sorted({int(row["distinct_latent_facts"]) for row in observations}):
        group = [
            row for row in observations if int(row["distinct_latent_facts"]) == coverage
        ]
        correct = sum(bool(row["correct"]) for row in group)
        low, high = wilson_interval(correct, len(group))
        rows.append(
            {
                "distinct_latent_facts": coverage,
                "n": len(group),
                "truth_choices": correct,
                "truth_rate": correct / len(group),
                "ci95_low": low,
                "ci95_high": high,
            }
        )
    return rows


def render_plots(
    pair_summary: Sequence[Mapping[str, Any]],
    dose_summary: Sequence[Mapping[str, Any]],
    observations: Sequence[Mapping[str, Any]],
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    x = [row["cards"] for row in dose_summary]
    y = [row["truth_rate"] for row in dose_summary]
    low = [row["truth_rate"] - row["ci95_low"] for row in dose_summary]
    high = [row["ci95_high"] - row["truth_rate"] for row in dose_summary]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(x, y, yerr=[low, high], marker="o", capsize=4)
    ax.axhline(1 / 3, color="black", linestyle="--", label="Chance = 1/3")
    ax.set(xlabel="Evidence cards", ylabel="Truth-selection rate", ylim=(0, 1.05))
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "evidence_dose_truth_curve.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for agent in sorted({int(row["agent_id"]) for row in observations}):
        points = []
        for dose in x:
            group = [
                row
                for row in observations
                if int(row["agent_id"]) == agent and int(row["dose"]) == dose
            ]
            points.append(sum(bool(row["correct"]) for row in group) / len(group))
        ax.plot(x, points, marker="o", label=f"Agent {agent}")
    ax.axhline(1 / 3, color="black", linestyle="--")
    ax.set(xlabel="Evidence cards", ylabel="Truth-selection rate", ylim=(0, 1.05))
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "evidence_dose_by_agent.png", dpi=180)
    plt.close(fig)
    coverage = summarize_by_latent_coverage(observations)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(
        [row["distinct_latent_facts"] for row in coverage],
        [row["truth_rate"] for row in coverage],
        marker="o",
    )
    ax.axhline(1 / 3, color="black", linestyle="--", label="Chance = 1/3")
    ax.set(
        xlabel="Distinct latent facts represented",
        ylabel="Truth-selection rate",
        ylim=(0, 1.05),
        xticks=range(10),
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "truth_by_latent_fact_coverage.png", dpi=180)
    plt.close(fig)
    agents = [row for row in pair_summary if row["agent_id"] != "pooled"]
    positions = range(len(agents))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(
        [p - width / 2 for p in positions],
        [r["validation_truth_rate"] for r in agents],
        width,
        label="Validation",
    )
    ax.bar(
        [p + width / 2 for p in positions],
        [r["game_init_truth_rate"] for r in agents],
        width,
        label="Game init",
    )
    ax.set_xticks(list(positions), [f"Agent {r['agent_id']}" for r in agents])
    ax.set(ylabel="Truth-selection rate", ylim=(0, 1.05))
    ax.axhline(1 / 3, color="black", linestyle="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "prompt_family_comparison.png", dpi=180)
    plt.close(fig)


__all__ = [
    "comparison_family",
    "render_plots",
    "summarize_by_latent_coverage",
    "summarize_doses",
    "summarize_doses_by_agent",
    "summarize_prompt_equivalence",
    "terminal_rows",
    "write_csv",
]
