"""Descriptive semantic endpoint tables for small relational takeover studies."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd


CLASSIFIER = "relational_false_takeover_v1"
PERSISTENCE_CLASSIFIER = "relational_persistence_exploratory_v1"
PERSISTENCE_REFINEMENT_CLASSIFIER = "relational_persistence_refinement_v1"
PERSISTENCE_TRUTH_CLASSIFIER = "relational_persistence_truth_aligned_v1"


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


def _mean_field(rows: Sequence[Any], field: str) -> float:
    values = [
        float(row.event[field]) for row in rows if row.event.get(field) is not None
    ]
    return float("nan") if not values else sum(values) / len(values)


def _truth_conditionals(event: Mapping[str, Any]) -> tuple[float | None, float | None]:
    """Truth-vote shares with and without a complete active proof."""

    strata = event.get("knowledge_stratum_counts")
    truth = event.get("truth_counts_by_stratum")
    if not isinstance(strata, Sequence) or not isinstance(truth, Sequence):
        return None, None
    counts = [int(value) for value in strata]
    truth_counts = [int(value) for value in truth]
    if not counts or len(counts) != len(truth_counts):
        return None, None
    full_count = counts[-1]
    not_full_count = sum(counts[:-1])
    return (
        None if full_count == 0 else truth_counts[-1] / full_count,
        None if not_full_count == 0 else sum(truth_counts[:-1]) / not_full_count,
    )


def relational_persistence_tables(
    rounds: Sequence[Any],
    cells: pd.DataFrame,
    *,
    classifier: str = PERSISTENCE_CLASSIFIER,
    allow_truth_target: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build finite-persistence episode and ``(rho,b)`` late-time summaries.

    Rounds 21--30 mean zero-based round indices 20--29. These values are
    described as late-time, never stationary.
    """

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
        late = [row for row in ordered if 20 <= int(row.round_index) <= 29]
        if len(late) != 10:
            raise ValueError(
                f"persistence endpoint requires rounds 21-30 for {episode_id}; "
                f"found {len(late)}"
            )
        target = str(
            first.event.get("controller_target") or first.event["analysis_target"]
        )
        truth = str(first.event["correct_answer"])
        if target == truth and not allow_truth_target:
            raise ValueError(f"persistence endpoint received truth target in {cell_id}")
        if target != truth and allow_truth_target:
            raise ValueError(
                f"truth-aligned persistence endpoint received false target in {cell_id}"
            )
        initial_counts = _counts(first.event, "before")
        population = sum(initial_counts.values())
        false_trajectory = [initial_counts[target] / population] + [
            _counts(row.event, "after")[target] / population for row in ordered
        ]
        final_counts = _counts(last.event, "after")
        final_winner, final_is_tie, final_tied = _winner(final_counts)
        late_false = _mean_field(late, "controller_target_share")
        false_final_takeover = final_winner == target
        false_ever_majority = max(false_trajectory) > 0.5
        if false_final_takeover:
            outcome = "FALSE_FINAL_TAKEOVER"
        elif late_false > 0.5:
            outcome = "FALSE_LATE_DOMINANT"
        elif false_ever_majority:
            outcome = "TRANSIENT_FALSE_MAJORITY"
        else:
            outcome = "NO_FALSE_MAJORITY"
        truth_full, truth_not_full = _truth_conditionals(last.event)
        late_truth_full = [
            pair[0]
            for pair in (_truth_conditionals(row.event) for row in late)
            if pair[0] is not None
        ]
        late_truth_not_full = [
            pair[1]
            for pair in (_truth_conditionals(row.event) for row in late)
            if pair[1] is not None
        ]
        coord = coordinates.get(cell_id, {})
        rho = float(
            first.event.get("epistemic_persistence", coord.get("epistemic_persistence"))
        )
        budget = int(
            first.event.get("intervention_budget", coord.get("intervention_budget"))
        )
        q = int(
            first.event.get("social_group_size", coord.get("social_group_size", -1))
        )
        evidence_strategy = str(
            first.event.get("controller_evidence_strategy")
            or coord.get("controller_evidence_strategy")
        )
        receiver_disposition = str(
            first.event.get("receiver_epistemic_disposition")
            or coord.get("receiver_epistemic_disposition")
        )
        target_semantics = str(
            coord.get("target_semantics", "truth" if allow_truth_target else "false")
        )
        rows.append(
            {
                "study_id": first.event.get("study_id", coord.get("study_id")),
                "cell_id": cell_id,
                "episode_id": episode_id,
                "task_id": str(first.event.get("task_id") or coord.get("task_id")),
                "epistemic_persistence": rho,
                "intervention_budget": budget,
                "actuation_fraction": budget / population,
                "social_group_size": q,
                "controller_evidence_strategy": evidence_strategy,
                "receiver_epistemic_disposition": receiver_disposition,
                "target_semantics": target_semantics,
                "ground_truth": truth,
                "controller_target": target,
                "controller_target_is_truth": False,
                "initial_false_target_share": false_trajectory[0],
                "maximum_false_target_share": max(false_trajectory),
                "final_false_target_share": false_trajectory[-1],
                "late_time_mean_false_target_share": late_false,
                "final_truth_share": float(last.event["truth_vote_share"]),
                "late_time_mean_truth_share": _mean_field(late, "truth_vote_share"),
                "final_active_phi": float(
                    last.event["active_full_proof_agent_share_after"]
                ),
                "late_time_mean_active_phi": _mean_field(
                    late, "active_full_proof_agent_share_after"
                ),
                "final_active_kappa": float(
                    last.event["active_mean_supporting_fact_coverage_after"]
                ),
                "late_time_mean_active_kappa": _mean_field(
                    late, "active_mean_supporting_fact_coverage_after"
                ),
                "final_historical_phi": float(
                    last.event["historical_full_proof_agent_share_after"]
                ),
                "late_time_mean_historical_phi": _mean_field(
                    late, "historical_full_proof_agent_share_after"
                ),
                "final_historical_kappa": float(
                    last.event["historical_mean_supporting_fact_coverage_after"]
                ),
                "late_time_mean_historical_kappa": _mean_field(
                    late, "historical_mean_supporting_fact_coverage_after"
                ),
                "advocate_rounds": sum(
                    row.event.get("controller_action") == "ADVOCATE_Z"
                    for row in ordered
                ),
                "no_op_rounds": sum(
                    row.event.get("controller_action") == "NO_OP" for row in ordered
                ),
                "controlled_microscopic_updates": sum(
                    int(row.event.get("controlled_position_count", 0))
                    for row in ordered
                ),
                "fact_acquisitions": sum(
                    int(row.event.get("new_peer_facts", 0))
                    + int(row.event.get("new_controller_facts", 0))
                    for row in ordered
                ),
                "fact_reactivations": sum(
                    int(row.event.get("reactivated_peer_fact_count", 0))
                    + int(row.event.get("reactivated_controller_fact_count", 0))
                    for row in ordered
                ),
                "fact_deactivations": sum(
                    int(row.event.get("persistence_deactivated_fact_count", 0))
                    for row in ordered
                ),
                "final_truth_share_given_active_full_proof": truth_full,
                "final_truth_share_given_not_active_full_proof": truth_not_full,
                "late_time_truth_share_given_active_full_proof": (
                    None
                    if not late_truth_full
                    else sum(late_truth_full) / len(late_truth_full)
                ),
                "late_time_truth_share_given_not_active_full_proof": (
                    None
                    if not late_truth_not_full
                    else sum(late_truth_not_full) / len(late_truth_not_full)
                ),
                "false_target_conditional_on_active_proof_available": False,
                "false_target_is_final_winner": false_final_takeover,
                "false_target_ever_majority": false_ever_majority,
                "final_winner_semantic": final_winner,
                "final_is_tie": final_is_tie,
                "final_tied_semantics": final_tied,
                "outcome_classification": outcome,
                "late_time_round_start_one_based": 21,
                "late_time_round_end_one_based": 30,
                "late_time_label": "late-time",
                "descriptive_only": True,
                "matched_revised_theory_applicable": False,
                "classification_version": classifier,
            }
        )
    episodes = pd.DataFrame(rows)
    if episodes.empty:
        return episodes, pd.DataFrame()
    numeric = [
        column
        for column in episodes.columns
        if column.startswith(("initial_", "maximum_", "final_", "late_time_"))
        and column
        not in {
            "late_time_label",
            "late_time_round_start_one_based",
            "late_time_round_end_one_based",
        }
    ]
    totals = [
        "advocate_rounds",
        "no_op_rounds",
        "controlled_microscopic_updates",
        "fact_acquisitions",
        "fact_reactivations",
        "fact_deactivations",
    ]
    group_columns = [
        "social_group_size",
        "controller_evidence_strategy",
        "receiver_epistemic_disposition",
        "target_semantics",
        "epistemic_persistence",
        "intervention_budget",
        "actuation_fraction",
    ]
    summary = episodes.groupby(group_columns, as_index=False)[numeric + totals].mean(
        numeric_only=True
    )
    counts = episodes.groupby(group_columns, as_index=False).agg(
        episodes=("episode_id", "count"),
        false_final_takeover_count=("false_target_is_final_winner", "sum"),
        false_late_dominant_count=(
            "outcome_classification",
            lambda values: sum(value == "FALSE_LATE_DOMINANT" for value in values),
        ),
        any_false_majority_count=("false_target_ever_majority", "sum"),
        final_tie_count=("final_is_tie", "sum"),
    )
    summary = summary.merge(counts, on=group_columns, how="left")
    summary["false_final_takeover_fraction"] = (
        summary["false_final_takeover_count"] / summary["episodes"]
    )
    summary["false_late_dominant_fraction"] = (
        summary["false_late_dominant_count"] / summary["episodes"]
    )
    summary["any_false_majority_fraction"] = (
        summary["any_false_majority_count"] / summary["episodes"]
    )
    requested = [
        "late_time_mean_false_target_share",
        "late_time_mean_truth_share",
        "late_time_mean_active_phi",
        "late_time_mean_active_kappa",
    ]
    grouped = episodes.groupby(group_columns)
    for column in requested:
        stats = (
            grouped[column]
            .agg(
                mean="mean",
                median="median",
                std="std",
                minimum="min",
                maximum="max",
            )
            .reset_index()
        )
        quartiles = grouped[column].quantile([0.25, 0.75]).unstack()
        quartiles["iqr"] = quartiles[0.75] - quartiles[0.25]
        stats = stats.merge(quartiles[["iqr"]].reset_index(), on=group_columns)
        stem = column.removeprefix("late_time_mean_")
        stats = stats.rename(
            columns={
                name: f"{stem}_{name}"
                for name in [
                    "mean",
                    "median",
                    "std",
                    "iqr",
                    "minimum",
                    "maximum",
                ]
            }
        )
        summary = summary.merge(stats, on=group_columns, how="left")
    summary["late_time_label"] = "late-time"
    summary["descriptive_only"] = True
    summary["matched_revised_theory_applicable"] = False
    return episodes, summary


def relational_persistence_refinement_tables(
    rounds: Sequence[Any], cells: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build Study 09d's repeated persistence-refinement summaries."""

    return relational_persistence_tables(
        rounds, cells, classifier=PERSISTENCE_REFINEMENT_CLASSIFIER
    )


def relational_persistence_truth_tables(
    rounds: Sequence[Any], cells: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build matched truth-aligned late-time persistence summaries."""

    episodes, summary = relational_persistence_tables(
        rounds,
        cells,
        classifier=PERSISTENCE_TRUTH_CLASSIFIER,
        allow_truth_target=True,
    )

    def aligned_names(frame: pd.DataFrame) -> pd.DataFrame:
        renamed = frame.rename(
            columns={
                column: column.replace("false_target", "controller_target").replace(
                    "false_", "truth_"
                )
                for column in frame.columns
            }
        )
        if "outcome_classification" in renamed:
            renamed["outcome_classification"] = (
                renamed["outcome_classification"]
                .astype(str)
                .str.replace("FALSE", "TRUTH", regex=False)
            )
        return renamed

    return aligned_names(episodes), aligned_names(summary)


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
