#!/usr/bin/env python
"""Plot what a relational blackboard grid did, round by round.

``mas-cc analysis relational-round-feedback`` writes tables only, so nothing in
the pipeline draws the one picture this design needs: episodes do not drift
toward the truth, they lock onto a consensus and stay there, and a cell mean
hides that completely.

Reads round_trajectory.jsonl straight from the run; no analysis step required,
so it works on a study that is still executing.

    scripts/local/plot_round_trajectories.py <grid-dir> [--output-dir DIR]

Four figures:
    truth_trajectories.png   truth share per round, one line per episode
    final_truth_hist.png     where episodes end up, pooled by grid factor
    vote_entropy.png         how fast the population stops disagreeing
    knowledge_growth.png     facts per agent, to separate "what they know"
                             from "what they vote"
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _events(path: pathlib.Path) -> list[dict[str, Any]]:
    out = []
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        out.append(row.get("event", row))
    return out


def _load(grid: pathlib.Path) -> tuple[dict[str, dict], dict[str, list[list[dict]]]]:
    cells: dict[str, dict] = {}
    for path in sorted(grid.glob("cells/*/overrides.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        cells[payload["cell_id"]] = payload.get("overrides", {})
    episodes: dict[str, list[list[dict]]] = collections.defaultdict(list)
    for cell in cells:
        directory = grid / "cells" / cell / "data" / "episodes"
        for path in sorted(directory.glob("*/round_trajectory.jsonl")):
            events = _events(path)
            if events:
                episodes[cell].append(events)
    return cells, episodes


def _label(overrides: dict) -> str:
    """A short human label from whatever the grid actually varied."""

    parts = []
    for key, value in sorted(overrides.items()):
        name = key.rsplit(".", 1)[-1]
        short = {
            "communication_profile": "",
            "epistemic_persistence": "rho=",
            "intervention_budget": "b=",
            "target": "target=",
        }.get(name, f"{name}=")
        parts.append(f"{short}{value}")
    return ", ".join(parts) or "run"


def _grid_of_axes(n: int):
    columns = min(4, max(1, n))
    rows = (n + columns - 1) // columns
    figure, axes = plt.subplots(
        rows, columns, figsize=(4.2 * columns, 3.4 * rows), squeeze=False
    )
    return figure, [axes[r][c] for r in range(rows) for c in range(columns)]


def _series(cells, episodes, field, before, ylabel, title, path, hline=None):
    figure, axes = _grid_of_axes(len(cells))
    for axis, cell in zip(axes, sorted(cells)):
        for events in episodes.get(cell, []):
            values = [events[0].get(before)] + [e.get(field) for e in events]
            rounds = list(range(len(values)))
            axis.plot(rounds, values, marker="o", markersize=2.5, linewidth=1.2, alpha=0.85)
        if hline is not None:
            axis.axhline(hline, color="grey", linestyle=":", linewidth=1)
        axis.set_title(f"{cell}\n{_label(cells[cell])}", fontsize=9)
        axis.set_xlabel("round")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    for axis in axes[len(cells):]:
        axis.axis("off")
    figure.suptitle(title, fontsize=12)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print(f"  wrote {path.name}")


def _final_hist(cells, episodes, path):
    # Pool over each varied factor separately: with 5 episodes a cell, a single
    # cell's histogram is noise, but the factor margins are readable.
    factors: dict[str, dict[Any, list[float]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    for cell, overrides in cells.items():
        finals = [events[-1]["truth_vote_share"] for events in episodes.get(cell, [])]
        for key, value in overrides.items():
            factors[key.rsplit(".", 1)[-1]][value].extend(finals)
    if not factors:
        return
    figure, axes = plt.subplots(
        1, len(factors), figsize=(5.0 * len(factors), 3.6), squeeze=False
    )
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    for axis, (name, levels) in zip(axes[0], sorted(factors.items())):
        axis.hist(
            list(levels.values()),
            bins=bins,
            label=[f"{name}={k} (n={len(v)})" for k, v in levels.items()],
        )
        axis.set_xlabel("final truth share")
        axis.set_ylabel("episodes")
        axis.set_title(name, fontsize=10)
        axis.legend(fontsize=7)
        axis.grid(alpha=0.25)
    figure.suptitle(
        "Final truth share is bimodal — episodes end near 0 or near 1", fontsize=12
    )
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print(f"  wrote {path.name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("grid_dir", type=pathlib.Path)
    parser.add_argument("--output-dir", type=pathlib.Path)
    args = parser.parse_args()

    cells, episodes = _load(args.grid_dir)
    if not episodes:
        print(f"no round_trajectory.jsonl under {args.grid_dir}", file=sys.stderr)
        return 1
    total = sum(len(v) for v in episodes.values())
    destination = args.output_dir or (
        args.grid_dir / "relational_imitation_round_feedback_analysis" / "plots"
    )
    destination.mkdir(parents=True, exist_ok=True)
    print(f"{len(cells)} cells, {total} episodes -> {destination}")

    _series(
        cells, episodes, "truth_vote_share", "truth_vote_share_before",
        "truth vote share", "Truth share per round (one line per episode)",
        destination / "truth_trajectories.png", hline=1 / 3,
    )
    _series(
        cells, episodes, "vote_entropy", "vote_entropy_before",
        "vote entropy (bits)", "Disagreement per round",
        destination / "vote_entropy.png",
    )
    _series(
        cells, episodes, "mean_known_fact_count", "mean_known_fact_count",
        "facts per agent", "Evidence accumulated per agent",
        destination / "knowledge_growth.png",
    )
    _final_hist(cells, episodes, destination / "final_truth_hist.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
