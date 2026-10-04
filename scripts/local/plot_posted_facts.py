#!/usr/bin/env python
"""Which facts actually get posted to the blackboard, per episode.

One panel per episode, one bar per fact in the task's fact space. Facts are
ordered decisive first, then controller-reportable, then neutral, so the
question "are the agents circulating the facts that settle the answer?" is
answered by looking at the left edge of each panel.

Agent posts and controller posts are coloured separately. They cannot overlap:
the decisive facts are private to agents and absent from the controller pool.

    scripts/local/plot_posted_facts.py <grid-dir> --task-dir <task> [-n 10]
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
from matplotlib.patches import Patch  # noqa: E402

AGENT_COLOR = "#2f6f9f"
CONTROLLER_COLOR = "#c8622d"
DECISIVE_BAND = "#f2e2c4"
CONTROLLER_BAND = "#e8eef4"
TRUTH_COLOR = "#3f8f5f"
OTHER_COLOR = "#b8b8b8"


def _events(path: pathlib.Path) -> list[dict[str, Any]]:
    out = []
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if line:
            row = json.loads(line)
            out.append(row.get("event", row))
    return out


def fact_order(task_dir: pathlib.Path) -> tuple[list[str], int, int]:
    """Fact ids ordered decisive, then controller-reportable, then neutral."""

    def ids(name: str) -> list[str]:
        payload = json.loads((task_dir / "facts" / f"{name}.json").read_text())
        return [f["fact_id"] if isinstance(f, dict) else f for f in payload]

    decisive = ids("decisive_facts")
    controller = ids("controller_reportable_facts")
    neutral = ids("neutral_facts")
    return decisive + controller + neutral, len(decisive), len(controller)


def episode_counts(episode: pathlib.Path) -> tuple[collections.Counter, collections.Counter]:
    agent: collections.Counter = collections.Counter()
    controller: collections.Counter = collections.Counter()
    trajectory = episode / "trajectory.jsonl"
    if trajectory.is_file():
        for line in trajectory.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            for decision in json.loads(line).get("decisions") or []:
                action = decision.get("action") or decision
                message = (action.get("metadata") or {}).get("public_message")
                if isinstance(message, dict) and message.get("shared_fact_id"):
                    agent[message["shared_fact_id"]] += 1
    rounds = episode / "round_trajectory.jsonl"
    if rounds.is_file():
        for event in _events(rounds):
            for fact in event.get("controller_report_fact_ids") or []:
                controller[fact] += 1
    return agent, controller


def outcome(episode: pathlib.Path) -> dict[str, Any] | None:
    """Final vote split, so a panel shows what the posting produced."""

    path = episode / "round_trajectory.jsonl"
    if not path.is_file():
        return None
    events = _events(path)
    if not events:
        return None
    last, first = events[-1], events[0]
    options = last.get("possible_answers") or []
    counts = last.get("occupation_counts_after") or []
    truth = last.get("correct_answer")
    return {
        "options": options,
        "counts": counts,
        "truth_index": options.index(truth) if truth in options else None,
        "final": last.get("truth_vote_share"),
        "initial": first.get("truth_vote_share_before"),
    }


def label_for(episode: pathlib.Path) -> str:
    for parent in episode.parents:
        candidate = parent / "overrides.json"
        if candidate.is_file():
            o = json.loads(candidate.read_text()).get("overrides", {})
            bits = []
            if "control.options.target" in o:
                target = o["control.options.target"]
                bits.append("truth" if target == "correct" else f"false({target})")
            if "control.options.intervention_budget" in o:
                bits.append(f"b={o['control.options.intervention_budget']}")
            if "game.options.epistemic_persistence" in o:
                bits.append(f"rho={o['game.options.epistemic_persistence']}")
            profile = o.get("game.options.board.communication_profile")
            if profile:
                bits.append(profile.replace("_communication", "").replace("_only", ""))
            return f"{episode.name}  {', '.join(bits)}"
    return episode.name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("grid_dir", type=pathlib.Path)
    parser.add_argument("--task-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("-n", "--episodes", type=int, default=10)
    parser.add_argument(
        "--sort", action="store_true",
        help="order panels by decisive share, strongest first",
    )
    args = parser.parse_args()

    order, n_dec, n_ctrl = fact_order(args.task_dir)
    index = {fact: i for i, fact in enumerate(order)}

    episodes = sorted({p.parent for p in args.grid_dir.rglob("trajectory.jsonl")})
    if not episodes:
        print(f"no episodes under {args.grid_dir}", file=sys.stderr)
        return 1
    episodes = episodes[: args.episodes]
    if args.sort:
        # Ranking by decisive share puts the relationship with the final vote
        # on the page: strong left-edge bars at the top, weak ones at the bottom.
        def decisive_share(ep: pathlib.Path) -> float:
            ag, ct = episode_counts(ep)
            total = sum(ag.values()) + sum(ct.values())
            return sum(ag.get(f, 0) for f in order[:n_dec]) / total if total else 0.0

        episodes = sorted(episodes, key=decisive_share, reverse=True)

    rows = (len(episodes) + 1) // 2
    figure, axes = plt.subplots(rows, 2, figsize=(17, 2.5 * rows), squeeze=False)
    flat = [axes[r][c] for r in range(rows) for c in range(2)]

    for axis, episode in zip(flat, episodes):
        agent, controller = episode_counts(episode)
        xs = range(len(order))
        a = [agent.get(f, 0) for f in order]
        c = [controller.get(f, 0) for f in order]
        # Shade the two pools so the reader can see at a glance which region a
        # bar falls in without reading fact ids.
        axis.axvspan(-0.5, n_dec - 0.5, color=DECISIVE_BAND, zorder=0)
        axis.axvspan(n_dec - 0.5, n_dec + n_ctrl - 0.5, color=CONTROLLER_BAND, zorder=0)
        axis.bar(xs, a, color=AGENT_COLOR, width=0.85, zorder=2)
        axis.bar(xs, c, bottom=a, color=CONTROLLER_COLOR, width=0.85, zorder=2)
        axis.set_title(label_for(episode), fontsize=8, loc="left")
        axis.set_xlim(-0.6, len(order) - 0.4)
        axis.set_xticks(list(xs))
        axis.set_xticklabels(
            [f.replace("cf_", "") for f in order], rotation=90, fontsize=4.2
        )
        axis.tick_params(axis="y", labelsize=6)
        axis.set_ylabel("posts", fontsize=7)
        decisive_posts = sum(agent.get(f, 0) for f in order[:n_dec])
        total = sum(a) + sum(c)
        share = decisive_posts / total if total else 0.0
        result = outcome(episode)
        note = f"decisive posts {share:.0%}"
        if result and result["final"] is not None:
            note += f"     truth vote {result['initial']:.0%} -> {result['final']:.0%}"
        axis.text(
            0.72, 0.95, note, transform=axis.transAxes,
            ha="right", va="top", fontsize=7.5,
        )
        if result and result["counts"]:
            # Final ballot as a small inset: the panel now shows what the
            # posting pattern to its left actually produced.
            inset = axis.inset_axes((0.775, 0.34, 0.2, 0.60))
            colors = [
                TRUTH_COLOR if i == result["truth_index"] else OTHER_COLOR
                for i in range(len(result["counts"]))
            ]
            inset.bar(range(len(result["counts"])), result["counts"], color=colors, width=0.7)
            inset.set_xticks(range(len(result["options"])))
            inset.set_xticklabels(
                [o.replace("ALLOCATION_", "A") for o in result["options"]], fontsize=5.5
            )
            inset.set_ylim(0, max(result["counts"]) * 1.25 or 1)
            inset.tick_params(axis="y", labelsize=5)
            inset.set_title("final vote", fontsize=6, pad=1.5)
            for spine in ("top", "right"):
                inset.spines[spine].set_visible(False)
    for axis in flat[len(episodes):]:
        axis.axis("off")

    figure.suptitle(
        "Which facts reach the blackboard, per episode "
        "(decisive first, then controller-reportable, then neutral)",
        fontsize=13, y=0.998,
    )
    figure.legend(
        handles=[
            Patch(color=AGENT_COLOR, label="posted by an agent"),
            Patch(color=CONTROLLER_COLOR, label="posted by the controller"),
            Patch(color=DECISIVE_BAND, label="decisive facts (agents only)"),
            Patch(color=CONTROLLER_BAND, label="controller-reportable facts"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, 0.982),
        ncol=4, fontsize=9, frameon=False,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.955))
    out = args.output or (args.grid_dir / "posted_facts_by_episode.png")
    figure.savefig(out, dpi=170)
    plt.close(figure)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
