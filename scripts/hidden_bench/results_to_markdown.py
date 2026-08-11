#!/usr/bin/env python3
"""Render one or two HiddenBench vanilla grid runs as a Markdown table, one
row per task.

A grid run's own artifacts (`grid_summary.json`, per-cell `aggregate.json`)
are aimed at machines, not eyes: none of them puts the paper's actual numbers
(`y_pre`, `y_post`, `accuracy_average`, ...) next to a human-readable task
name in one place. This script does that join.

Usage:
    python scripts/hidden_bench/results_to_markdown.py <run_dir> [<run_dir> ...] [-o OUTPUT.md]

Pass one run dir (e.g. the `profile: full` run) for a single-condition table,
or two (the `profile: full` and `profile: hidden` runs) to merge them by task
and add `gap_to_full = y_post(hidden) - y_full`.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_TASKS = (
    REPO_ROOT
    / "scripts"
    / "local_llms"
    / "hiddenbench_population_pipeline"
    / "data"
    / "hiddenbench"
    / "canonical"
    / "tasks.json"
)


def load_corpus() -> dict[str, dict[str, Any]]:
    payload = json.loads(CORPUS_TASKS.read_text(encoding="utf-8"))
    return {task["name"]: task for task in payload["tasks"]}


def _read_final_csv(path: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                metrics[row["metric_name"]] = float(row["value"])
            except (TypeError, ValueError):
                pass
    return metrics


def _final_round_group_vote(streaming_csv: Path) -> str | None:
    """The option with the largest `population_action_share_per_option` at the last round."""

    best_round = -1
    shares: dict[int, dict[str, float]] = {}
    with streaming_csv.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["metric_name"] != "population_action_share_per_option":
                continue
            round_index = int(row["round_index"])
            best_round = max(best_round, round_index)
            shares.setdefault(round_index, {})[row["series"]] = float(row["value"])
    if best_round not in shares or not shares[best_round]:
        return None
    return max(shares[best_round].items(), key=lambda item: item[1])[0]


def load_run(run_dir: Path) -> tuple[str, dict[str, dict[str, Any]], dict[str, str]]:
    """`(profile, {task_name: row}, {task_name: error})` for one grid run directory.

    A cell with no completed outcome (still queued, or every attempt failed)
    goes into the error dict instead of silently vanishing from the table -
    an incomplete grid is a fact worth showing, not hiding.
    """

    grid_summary = json.loads((run_dir / "grid_summary.json").read_text(encoding="utf-8"))
    profile = None
    rows: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}
    for cell in grid_summary["cells"]:
        task_name = cell["overrides"].get("game.options.task_id")
        if task_name is None:
            continue
        outcomes = [o for o in cell["outcomes"] if o["status"] == "completed"]
        if not outcomes:
            failed = [o for o in cell["outcomes"] if o["status"] == "failed"]
            errors[task_name] = failed[0]["error"] if failed else "no completed episode"
            continue
        episode_id = outcomes[0]["episode_id"]
        episode_dir = run_dir / "cells" / cell["cell_id"] / "data" / "episodes" / episode_id
        final_metrics = _read_final_csv(episode_dir / "metrics" / "final.csv")
        group_vote = _final_round_group_vote(episode_dir / "metrics" / "streaming.csv")
        if profile is None:
            resolved = (run_dir / "cells" / cell["cell_id"] / "resolved_config.yaml").read_text(
                encoding="utf-8"
            )
            for line in resolved.splitlines():
                if line.strip().startswith("profile:"):
                    profile = line.split(":", 1)[1].strip()
                    break
        rows[task_name] = {
            "cell_id": cell["cell_id"],
            "episode_id": episode_id,
            "group_vote": group_vote,
            **final_metrics,
        }
    return profile or "unknown", rows, errors


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def build_table(run_dirs: list[Path]) -> str:
    corpus = load_corpus()
    runs = [load_run(run_dir) for run_dir in run_dirs]
    # Task order follows the first run's grid_summary order; any task only
    # present in a later run is appended after. Tasks with no completed
    # episode in *any* run are reported separately, not as a blank row.
    _, first_rows, _ = runs[0]
    ordered = list(first_rows.keys())
    for _, rows, _ in runs[1:]:
        for name in rows:
            if name not in ordered:
                ordered.append(name)

    lines: list[str] = []
    lines.append(f"# HiddenBench vanilla results ({', '.join(run_dir.name for run_dir in run_dirs)})")
    lines.append("")

    has_full = any(profile == "full" for profile, _, _ in runs)
    has_hidden = any(profile == "hidden" for profile, _, _ in runs)

    header = ["Task", "Correct answer"]
    if has_full:
        header += ["Y_full", "Full-profile vote"]
    if has_hidden:
        header += ["Y_pre", "Y_post", "Improvement", "Post-discussion vote", "Unshared disclosure"]
    if has_full and has_hidden:
        header += ["gap_to_full"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    full_rows = next((rows for profile, rows, _ in runs if profile == "full"), {})
    hidden_rows = next((rows for profile, rows, _ in runs if profile == "hidden"), {})
    full_errors = next((errors for profile, _, errors in runs if profile == "full"), {})
    hidden_errors = next((errors for profile, _, errors in runs if profile == "hidden"), {})

    for name in ordered:
        task = corpus.get(name, {})
        correct = task.get("correct_answer", "-")
        cells = [name, correct]
        full_row = full_rows.get(name)
        hidden_row = hidden_rows.get(name)
        if has_full:
            y_full = full_row.get("y_post") if full_row else None
            cells += [_fmt(y_full), _fmt(full_row.get("group_vote") if full_row else None)]
        if has_hidden:
            cells += [
                _fmt(hidden_row.get("y_pre") if hidden_row else None),
                _fmt(hidden_row.get("y_post") if hidden_row else None),
                _fmt(hidden_row.get("improvement") if hidden_row else None),
                _fmt(hidden_row.get("group_vote") if hidden_row else None),
                _fmt(hidden_row.get("final_unshared_disclosure_rate") if hidden_row else None),
            ]
        if has_full and has_hidden:
            y_full = full_row.get("y_post") if full_row else None
            y_post_hidden = hidden_row.get("y_post") if hidden_row else None
            gap = (y_post_hidden - y_full) if (y_full is not None and y_post_hidden is not None) else None
            cells.append(_fmt(gap))
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    if has_full and not has_hidden:
        lines.append(
            "_Y_full is the ceiling condition: every agent starts holding all information "
            "(`profile: full, rounds: 0`), so the pre-vote is the post-vote._"
        )
    if has_hidden and not has_full:
        lines.append(
            "_Y_pre/Y_post are the paper's asymmetric-information condition "
            "(`profile: hidden`): each agent starts with only its own private slice, "
            "discusses, then votes again._"
        )
    if has_full and has_hidden:
        lines.append(
            "_`gap_to_full = Y_post(hidden) - Y_full` - how much accuracy the group loses "
            "to information asymmetry that discussion did not recover._"
        )
    lines.append("")

    incomplete = sorted(set(full_errors) | set(hidden_errors))
    if incomplete:
        lines.append(f"## Incomplete ({len(incomplete)} task(s) with no completed episode)")
        lines.append("")
        lines.append("| Task | Full-profile run | Hidden-profile run |")
        lines.append("|---|---|---|")
        for name in incomplete:
            lines.append(
                f"| {name} | {full_errors.get(name, '-') if has_full else 'n/a'} "
                f"| {hidden_errors.get(name, '-') if has_hidden else 'n/a'} |"
            )
        lines.append("")
        lines.append(
            "_Re-run the same `experiment run` command against the same `--output-dir`: "
            "`--resume` is the default and only retries episodes that never completed._"
        )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path, help="one or two grid run directories")
    parser.add_argument("-o", "--output", type=Path, default=None, help="output .md path")
    args = parser.parse_args()

    table = build_table(args.run_dirs)
    output = args.output or (args.run_dirs[0] / "results_table.md")
    output.write_text(table, encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
