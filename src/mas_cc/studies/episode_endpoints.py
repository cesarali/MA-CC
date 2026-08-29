"""Descriptive semantic endpoint tables for small relational takeover studies."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd


CLASSIFIER = "relational_false_takeover_v1"


def _counts(event: Mapping[str, Any], suffix: str) -> dict[str, int]:
    options = [str(value) for value in event["possible_answers"]]
    raw = event[f"occupation_counts_{suffix}"]
    if isinstance(raw, Mapping):
        return {option: int(raw.get(option, 0)) for option in options}
    return dict(zip(options, (int(value) for value in raw), strict=True))


def _winner(counts: Mapping[str, int]) -> tuple[str | None, bool, list[str]]:
    maximum = max(counts.values())
    tied = sorted(option for option, count in counts.items() if count == maximum)
    return (tied[0] if len(tied) == 1 else None, len(tied) > 1, tied)


def _classification(
    *, false_final: bool, false_ever_majority: bool, delta_false: float
) -> str:
    if false_final:
        return "FALSE_FINAL_TAKEOVER"
    if false_ever_majority:
        return "TRANSIENT_FALSE_MAJORITY"
    if delta_false > 0:
        return "FALSE_STEERING_WITHOUT_MAJORITY"
    return "NO_MEANINGFUL_FALSE_STEERING"


def relational_false_takeover_tables(
    rounds: Sequence[Any], cells: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build one semantic endpoint row per episode and one row per ``(q,b)``."""

    grouped: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for row in rounds:
        grouped[(str(row.cell_id), str(row.episode_id))].append(row)
    coordinates = (
        cells.set_index("cell_id").to_dict(orient="index") if not cells.empty else {}
    )
    rows: list[dict[str, Any]] = []
    for (cell_id, episode_id), group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda row: row.round_index)
        first, last = ordered[0], ordered[-1]
        truth = str(first.event["correct_answer"])
        target = str(
            first.event.get("controller_target") or first.event["analysis_target"]
        )
        if target == truth:
            raise ValueError(
                f"false-takeover endpoint received truth target in {cell_id}"
            )
        initial_counts = _counts(first.event, "before")
        final_counts = _counts(last.event, "after")
        population = sum(initial_counts.values())
        trajectory = [(-1, initial_counts[target] / population)] + [
            (event.round_index, _counts(event.event, "after")[target] / population)
            for event in ordered
        ]
        truth_trajectory = [(-1, initial_counts[truth] / population)] + [
            (event.round_index, _counts(event.event, "after")[truth] / population)
            for event in ordered
        ]
        winner, tie, tied = _winner(final_counts)
        majority_rounds = [index for index, share in trajectory if share > 0.5]
        truth_majority_rounds = [
            index for index, share in truth_trajectory if share > 0.5
        ]
        maximum_round, maximum = max(trajectory, key=lambda item: item[1])
        final_false = winner == target
        delta_false = trajectory[-1][1] - trajectory[0][1]
        coord = coordinates.get(cell_id, {})
        rows.append(
            {
                "study_id": first.event.get("study_id", coord.get("study_id")),
                "cell_id": cell_id,
                "episode_id": episode_id,
                "task_id": str(first.event.get("task_id") or coord.get("task_id")),
                "social_group_size": int(
                    first.event.get("social_group_size")
                    or coord.get("social_group_size")
                ),
                "intervention_budget": int(
                    first.event.get("intervention_budget")
                    or coord.get("intervention_budget")
                ),
                "ground_truth": truth,
                "controller_target": target,
                "controller_target_is_truth": False,
                "initial_false_target_share": trajectory[0][1],
                "final_false_target_share": trajectory[-1][1],
                "initial_truth_share": truth_trajectory[0][1],
                "final_truth_share": truth_trajectory[-1][1],
                "false_target_is_final_winner": final_false,
                "truth_is_final_winner": winner == truth,
                "final_winner_semantic": winner,
                "final_is_tie": tie,
                "final_tied_semantics": tied,
                "max_false_target_share": maximum,
                "round_of_max_false_target_share": maximum_round,
                "false_target_ever_majority": bool(majority_rounds),
                "first_false_majority_round": majority_rounds[0]
                if majority_rounds
                else None,
                "final_minus_initial_false_target_share": delta_false,
                "truth_ever_majority": bool(truth_majority_rounds),
                "first_truth_majority_round": truth_majority_rounds[0]
                if truth_majority_rounds
                else None,
                "final_minus_initial_truth_share": truth_trajectory[-1][1]
                - truth_trajectory[0][1],
                "takeover_classification": _classification(
                    false_final=final_false,
                    false_ever_majority=bool(majority_rounds),
                    delta_false=delta_false,
                ),
                "classification_version": CLASSIFIER,
                "matched_revised_theory_applicable": False,
            }
        )
    episodes = pd.DataFrame(rows)
    if episodes.empty:
        return episodes, pd.DataFrame()
    summary_rows = []
    for (q, budget), frame in episodes.groupby(
        ["social_group_size", "intervention_budget"], sort=True
    ):
        summary_rows.append(
            {
                "social_group_size": int(q),
                "intervention_budget": int(budget),
                "episodes": len(frame),
                "false_wins": int(frame["false_target_is_final_winner"].sum()),
                "truth_wins": int(frame["truth_is_final_winner"].sum()),
                "ties": int(frame["final_is_tie"].sum()),
                "mean_final_false_target_share": float(
                    frame["final_false_target_share"].mean()
                ),
                "mean_final_truth_share": float(frame["final_truth_share"].mean()),
                "descriptive_only": True,
                "matched_revised_theory_applicable": False,
            }
        )
    return episodes, pd.DataFrame(summary_rows)


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "|"
        + "|".join(
            "---:" if index in {0, 1, 3, 4} else "---" for index in range(len(headers))
        )
        + "|",
        *("| " + " | ".join(str(value) for value in row) + " |" for row in rows),
    ]


def false_takeover_markdown(episodes: pd.DataFrame, summary: pd.DataFrame) -> str:
    """Human-facing existence-test report with the two preregistered tables first."""

    episode_rows = []
    for row in episodes.sort_values(
        ["social_group_size", "intervention_budget", "task_id"]
    ).to_dict(orient="records"):
        winner = (
            "TIE: " + ", ".join(row["final_tied_semantics"])
            if row["final_is_tie"]
            else row["final_winner_semantic"]
        )
        episode_rows.append(
            (
                row["social_group_size"],
                row["intervention_budget"],
                row["task_id"],
                f"{row['final_false_target_share']:.3f}",
                f"{row['final_truth_share']:.3f}",
                winner,
                row["false_target_is_final_winner"],
                row["final_is_tie"],
            )
        )
    summary_rows = [
        (
            row["social_group_size"],
            row["intervention_budget"],
            row["episodes"],
            row["false_wins"],
            row["truth_wins"],
            row["ties"],
            f"{row['mean_final_false_target_share']:.3f}",
            f"{row['mean_final_truth_share']:.3f}",
        )
        for row in summary.sort_values(
            ["social_group_size", "intervention_budget"]
        ).to_dict(orient="records")
    ]
    any_win = bool(episodes["false_target_is_final_winner"].any())
    lines = ["# Study 09b false-takeover existence test", ""]
    lines += _markdown_table(
        (
            "q",
            "b",
            "Task",
            "Final false share",
            "Final truth share",
            "Final winner",
            "False win?",
            "Tie?",
        ),
        episode_rows,
    )
    lines += ["", "## Four-regime descriptive summary", ""]
    lines += _markdown_table(
        (
            "q",
            "b",
            "Episodes",
            "False wins",
            "Truth wins",
            "Ties",
            "Mean final p_false",
            "Mean final p_truth",
        ),
        summary_rows,
    )
    lines += [
        "",
        "## Existence result",
        "",
        f"**Did any episode end with the false target as unique winner? {'YES' if any_win else 'NO'}.**",
        "",
        "These are two descriptive episodes per regime, not precise probability estimates.",
        "",
        "## Classification rules",
        "",
        "1. `FALSE_FINAL_TAKEOVER`: the false target is the unique final winner.",
        "2. `TRANSIENT_FALSE_MAJORITY`: it exceeds 0.5 at some round but is not the unique final winner.",
        "3. `FALSE_STEERING_WITHOUT_MAJORITY`: its final share exceeds its initial share, without a majority or final win.",
        "4. `NO_MEANINGFUL_FALSE_STEERING`: none of the rules above applies.",
        "",
        "Ties are recorded explicitly and never counted as false wins.",
        "Round indices are zero-based; `-1` denotes the initialized state before round 0.",
        "The revised single-affinity theory is a q=1 reference, so it is not treated as matched theory here.",
    ]
    return "\n".join(lines) + "\n"


__all__ = ["CLASSIFIER", "false_takeover_markdown", "relational_false_takeover_tables"]
