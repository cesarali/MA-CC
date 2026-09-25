#!/usr/bin/env python
"""Fact circulation against the vote, round by round.

One figure per experimental condition, ten episodes in two columns. Each
episode gets a pair of stacked panels sharing the round axis:

    top     how many agents hold each allocation after each round
    bottom  heat map, rows are facts, columns are rounds, colour is how many
            times that fact was posted to the board in that round

Read together they answer "did the board carry the evidence at the moment the
vote moved?". Facts are ordered decisive first, so the band above the rule is
the six facts that jointly determine the answer. A fact never posted in a round
is left white, so the map shows absence as clearly as presence.

Controller posts are ringed rather than recoloured: the two sources can never
share a fact, since the decisive facts are private to agents and absent from
the controller pool.

    scripts/local/plot_fact_heatmap.py <grid-dir> --task-dir <task> --out-dir <dir>
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import io
import shutil
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

TRUTH_COLOR = "#2e7d4f"
OTHER_COLORS = ["#b5482f", "#4a5aa8"]
DECISIVE_LABEL = "#8a5a12"
CONTROLLER_RING = "#0f7bc4"
# Viridis, with unposted cells masked to white so absence reads as blank.
HEAT = plt.get_cmap("viridis").copy()


def use_latex() -> bool:
    ok = all(shutil.which(b) for b in ("latex", "dvipng"))
    if ok:
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "text.latex.preamble": r"\usepackage{amsmath}",
        })
    else:
        plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})
    plt.rcParams.update({"axes.linewidth": 0.6, "savefig.facecolor": "white"})
    return ok


TEX = False


def esc(text: str) -> str:
    """LaTeX chokes on the underscores in fact ids and cell names."""

    return text.replace("_", r"\_") if TEX else text


def _events(path: pathlib.Path) -> list[dict[str, Any]]:
    out = []
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if line:
            row = json.loads(line)
            out.append(row.get("event", row))
    return out


def fact_order(
    task_dir: pathlib.Path, population: int, with_controller: bool
) -> tuple[list[str], set[str]]:
    """The facts that could actually reach this task's board, decisive first.

    Not every fact in the task file is in play. task_004 removes the decisive
    facts by dropping the agents that held them, so listing them would show six
    rows that no participant could ever post. The corpus is what agents hold,
    plus the controller pool only when a controller is running.
    """

    def ids(name: str) -> list[str]:
        payload = json.loads((task_dir / "facts" / f"{name}.json").read_text())
        return [f["fact_id"] if isinstance(f, dict) else f for f in payload]

    assignment = task_dir / "private" / f"N{population}_assignment.json"
    held: set[str] = set()
    if assignment.is_file():
        payload = json.loads(assignment.read_text())
        held = {f for v in payload.get("agent_assignments", {}).values() for f in v}

    corpus = set(held)
    if with_controller:
        corpus |= set(ids("controller_reportable_facts"))
    if not corpus:                       # no assignment on disk: fall back
        corpus = set(ids("controller_reportable_facts")) | set(ids("neutral_facts"))

    decisive = [f for f in ids("decisive_facts") if f in corpus]
    rest = [f for f in ids("all_true_facts") if f in corpus and f not in set(decisive)]
    return decisive + rest, set(decisive)


def overrides_of(episode: pathlib.Path) -> dict[str, Any]:
    for parent in episode.parents:
        candidate = parent / "overrides.json"
        if candidate.is_file():
            return json.loads(candidate.read_text()).get("overrides", {})
    return {}


def condition(episode: pathlib.Path) -> tuple[str, Any, Any]:
    """(arm, rho, budget) - the grouping key, one figure per distinct value."""

    o = dict(overrides_of(episode))
    if "game.options.epistemic_persistence" not in o:
        # Single-run output has no overrides.json; the round record carries the
        # same settings, so fall back to it rather than labelling the figure None.
        events = _events(episode / "round_trajectory.jsonl")
        if events:
            o.setdefault("game.options.epistemic_persistence",
                         events[-1].get("epistemic_persistence"))
            o.setdefault("game.options.board.communication_profile",
                         events[-1].get("communication_profile"))
            budget = events[-1].get("intervention_budget")
            if budget:
                o.setdefault("control.options.intervention_budget", budget)
            o.setdefault("control.options.target", events[-1].get("controller_target"))
    target = o.get("control.options.target")
    arm = ("no-controller" if target is None
           else "truth-controller" if target == "correct"
           else "false-controller")
    return arm, o.get("game.options.epistemic_persistence"), o.get("control.options.intervention_budget")


def episode_data(episode: pathlib.Path, order: list[str], population: int):
    index = {fact: i for i, fact in enumerate(order)}
    agent: collections.Counter = collections.Counter()
    controller: collections.Counter = collections.Counter()
    rounds = 0
    path = episode / "trajectory.jsonl"
    if path.is_file():
        for line in path.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            raw = row.get("interaction_index")
            if raw is None:
                continue
            game_round = ((int(raw) - 1) // population) + 1
            rounds = max(rounds, game_round)
            for decision in row.get("decisions") or []:
                action = decision.get("action") or decision
                message = (action.get("metadata") or {}).get("public_message")
                if isinstance(message, dict) and message.get("shared_fact_id"):
                    fact = message["shared_fact_id"]
                    if fact in index:
                        agent[(index[fact], game_round)] += 1
    events = _events(episode / "round_trajectory.jsonl")
    for event in events:
        game_round = int(event.get("absolute_round") or 0)
        rounds = max(rounds, game_round)
        for fact in event.get("controller_report_fact_ids") or []:
            if fact in index:
                controller[(index[fact], game_round)] += 1
    return agent, controller, rounds, events



def load_from_zip(zip_path: pathlib.Path, order: list[str], per_profile: int):
    """Rebuild the same per-episode structures from a study analysis archive.

    A finished study is distributed as parquet tables, not episode folders.
    Agent posts live one-per-row in micro_slots; controller posts are a JSON
    list on the round record, because the controller is not a population member
    and has no micro slot of its own.
    """

    import zipfile

    import pandas as pd

    index = {fact: i for i, fact in enumerate(order)}
    with zipfile.ZipFile(zip_path) as archive:
        micro = pd.read_parquet(io.BytesIO(archive.read("tables/micro_slots.parquet")))
        rounds = pd.read_parquet(io.BytesIO(archive.read("tables/rounds.parquet")))

    posts = micro[micro["new_message_shared_fact_id"].notna()].copy()
    # episode_id is NOT unique: the same name recurs in different cells, which
    # silently interleaves two episodes. episode_key is the per-episode hash.
    # micro_slots.absolute_round is null throughout this schema; round_index is
    # the usable one and counts from zero, while the round table counts from one.
    posts["round"] = posts["round_index"].astype(int) + 1
    agent_counts = (
        posts.groupby(["episode_key", "round", "new_message_shared_fact_id"])
        .size()
        .reset_index(name="n")
    )

    meta = (
        rounds.sort_values("absolute_round")
        .groupby("episode_key")
        .agg(
            name=("episode_id", "last"),
            profile=("communication_profile", "last"),
            rho=("epistemic_persistence", "last"),
            budget=("intervention_budget", "last"),
            target=("controller_target", "last"),
            correct=("correct_answer", "last"),
        )
        .reset_index()
    )

    def arm_for(row) -> str:
        # A no-control episode has no target at all. Testing it with `not` is
        # wrong: the value is NaN, and `not NaN` is False.
        if pd.isna(row["target"]) or not row["target"] or int(row["budget"] or 0) == 0:
            return "no-controller"
        return "truth-controller" if row["target"] == row["correct"] else "false-controller"

    meta["arm"] = meta.apply(arm_for, axis=1)

    groups: dict[tuple, list[str]] = collections.defaultdict(list)
    for (arm, rho, budget, profile), block in meta.groupby(
        ["arm", "rho", "budget", "profile"], dropna=False
    ):
        key = (arm, rho, None if arm == "no-controller" else int(budget))
        groups[key].extend(sorted(block["episode_key"])[:per_profile])

    wanted = {e for members in groups.values() for e in members}
    agent_by_ep: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for row in agent_counts.itertuples(index=False):
        if row.episode_key in wanted:
            fact = row.new_message_shared_fact_id
            if fact in index:
                agent_by_ep[row.episode_key][(index[fact], int(row.round))] += int(row.n)

    ctrl_by_ep: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    events_by_ep: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    keep = rounds[rounds["episode_key"].isin(wanted)]
    for row in keep.sort_values(["episode_key", "absolute_round"]).itertuples(index=False):
        rnd = int(row.absolute_round)
        for fact in json.loads(row.controller_report_fact_ids or "[]"):
            if fact in index:
                ctrl_by_ep[row.episode_key][(index[fact], rnd)] += 1
        events_by_ep[row.episode_key].append({
            "absolute_round": rnd,
            "possible_answers": json.loads(row.possible_answers),
            "correct_answer": row.correct_answer,
            "occupation_counts_before": json.loads(row.occupation_counts_before),
            "occupation_counts_after": json.loads(row.occupation_counts_after),
        })

    names = dict(zip(meta["episode_key"], meta["name"]))
    profiles = dict(zip(meta["episode_key"], meta["profile"]))
    out: dict[tuple, list] = {}
    for key, members in groups.items():
        rows = []
        for episode in members:
            events = events_by_ep.get(episode) or []
            if not events:
                continue
            rows.append((
                _ZipEpisode(names.get(episode, episode[:12]), profiles.get(episode, "")),
                agent_by_ep[episode], ctrl_by_ep[episode],
                max(e["absolute_round"] for e in events), events,
            ))
        if rows:
            out[key] = rows
    return out


class _ZipEpisode:
    """Stands in for an episode directory when reading an analysis archive."""

    def __init__(self, name: str, profile: str) -> None:
        self.name = name
        self.profile = profile


def draw(
    figure, spec, episode, order, decisive, rows, population, agent, controller,
    rounds, events, vmax, show_x, show_y,
):
    inner = GridSpecFromSubplotSpec(2, 1, subplot_spec=spec, height_ratios=(1, 3.1), hspace=0.06)
    top = figure.add_subplot(inner[0])
    bottom = figure.add_subplot(inner[1], sharex=top)

    options = events[0].get("possible_answers") or []
    truth = events[-1].get("correct_answer")
    truth_index = options.index(truth) if truth in options else None
    series = [events[0].get("occupation_counts_before") or []]
    series += [e.get("occupation_counts_after") or [] for e in events]
    xs = list(range(len(series)))
    for i, option in enumerate(options):
        colour = TRUTH_COLOR if i == truth_index else OTHER_COLORS[i % len(OTHER_COLORS)]
        top.plot(xs, [s[i] if i < len(s) else 0 for s in series], marker="o",
                 markersize=2.6, linewidth=1.4, color=colour,
                 label=esc(option))
    top.set_ylim(0, population * 1.1)
    top.set_yticks([0, population // 2, population])
    top.tick_params(labelbottom=False, labelsize=7)
    top.grid(alpha=0.2, linewidth=0.5)
    profile = getattr(episode, "profile", None)
    if profile is None:
        profile = overrides_of(episode).get("game.options.board.communication_profile", "")
    name = episode.name if len(episode.name) < 26 else episode.name[:24] + ".."
    top.set_title(f"{esc(name)}   {esc(str(profile))}", fontsize=8, loc="left", pad=3)
    if show_y:
        top.set_ylabel("votes", fontsize=8)

    matrix = np.array([[agent.get((i, r), 0) + controller.get((i, r), 0)
                        for r in range(1, rounds + 1)] for i in rows], dtype=float)
    image = bottom.imshow(
        np.ma.masked_equal(matrix, 0), aspect="auto", cmap=HEAT,
        interpolation="nearest", vmin=0, vmax=vmax,
        extent=(0.5, rounds + 0.5, len(rows) - 0.5, -0.5),
    )
    image.cmap.set_bad("white")
    for n, i in enumerate(rows):
        for r in range(1, rounds + 1):
            if controller.get((i, r), 0):
                bottom.plot(r, n, marker="o", markersize=5.5, markerfacecolor="none",
                            markeredgecolor=CONTROLLER_RING, markeredgewidth=1.2)
    n_dec = sum(1 for i in rows if order[i] in decisive)
    if n_dec:
        bottom.axhline(n_dec - 0.5, color=DECISIVE_LABEL, linewidth=0.7,
                       linestyle=(0, (4, 3)))
    bottom.set_xticks(range(1, rounds + 1))
    bottom.tick_params(labelsize=7)
    if show_x:
        bottom.set_xlabel("round", fontsize=9)
    else:
        bottom.tick_params(labelbottom=False)
    bottom.set_yticks(range(len(rows)))
    if show_y:
        bottom.set_yticklabels([esc(order[i].replace("cf_", "")) for i in rows], fontsize=5.5)
        for tick, i in zip(bottom.get_yticklabels(), rows):
            if order[i] in decisive:
                tick.set_color(DECISIVE_LABEL)
        bottom.set_ylabel("fact", fontsize=8)
    else:
        bottom.set_yticklabels([])
    return image


def main() -> int:
    global TEX
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("grid_dir", type=pathlib.Path, nargs="?")
    parser.add_argument("--zip", type=pathlib.Path,
                        help="study analysis archive, instead of a grid directory")
    parser.add_argument("--per-profile", type=int, default=5,
                        help="episodes taken from each communication profile")
    parser.add_argument("--task-dir", type=pathlib.Path, required=True)
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument("--population", type=int, default=24)
    parser.add_argument("--max-episodes", type=int, default=10)
    parser.add_argument("--no-latex", action="store_true")
    args = parser.parse_args()

    TEX = (not args.no_latex) and use_latex()

    if args.zip:
        # A study archive always carries controller arms.
        order, decisive = fact_order(args.task_dir, args.population, True)
        preloaded = load_from_zip(args.zip, order, args.per_profile)
        groups = {k: None for k in preloaded}
    else:
        if not args.grid_dir:
            print("give a grid directory or --zip", file=sys.stderr)
            return 1
        preloaded = None
        groups = collections.defaultdict(list)
        for ep in sorted({p.parent for p in args.grid_dir.rglob("trajectory.jsonl")}):
            groups[condition(ep)].append(ep)
        # The controller pool only belongs in the corpus if a controller ran.
        order, decisive = fact_order(
            args.task_dir, args.population,
            any(key[0] != "no-controller" for key in groups),
        )
    if not groups:
        print("no episodes found", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for key, episodes in sorted(groups.items(), key=lambda kv: str(kv[0])):
        arm, rho, budget = key
        if preloaded is not None:
            loaded = preloaded[key][: args.max_episodes]
        else:
            loaded = [(ep, *episode_data(ep, order, args.population))
                      for ep in episodes[: args.max_episodes]]
            loaded = [row for row in loaded if row[3] and row[4]]
        if not loaded:
            continue
        # One row set and one colour scale for the whole figure, so panels are
        # directly comparable.
        rows = list(range(len(order)))
        vmax = max((max(a.values()) if a else 0) for _, a, _, _, _ in loaded) or 1
        rounds = max(r for _, _, _, r, _ in loaded)

        ncol = 2
        nrow = (len(loaded) + ncol - 1) // ncol
        figure = plt.figure(figsize=(7.6 * ncol, (0.118 * len(rows) + 1.15) * nrow + 0.7))
        # Explicit margins rather than tight_layout: the shared colour bar is a
        # free-floating axes, which tight_layout cannot reason about.
        gs = GridSpec(
            nrow, ncol, figure=figure,
            left=0.085, right=0.885, top=0.935, bottom=0.045,
            hspace=0.30, wspace=0.075,
        )
        image = None
        for n, (ep, agent, controller, _r, events) in enumerate(loaded):
            r, c = divmod(n, ncol)
            image = draw(figure, gs[r, c], ep, order, decisive, rows, args.population,
                         agent, controller, rounds, events, vmax,
                         show_x=(r == nrow - 1), show_y=(c == 0))
        bits = [arm]
        if rho is not None:
            bits.append(rf"$\rho={rho}$" if TEX else f"rho={rho}")
        if budget is not None:
            bits.append(f"$b={budget}$" if TEX else f"b={budget}")
        figure.suptitle(", ".join(bits), fontsize=15, y=0.985)

        options = loaded[0][4][0].get("possible_answers") or []
        truth = loaded[0][4][-1].get("correct_answer")
        handles = [
            Line2D([], [], color=(TRUTH_COLOR if o == truth
                                  else OTHER_COLORS[i % len(OTHER_COLORS)]),
                   marker="o", markersize=3.5, linewidth=1.6,
                   label=esc(o) + (" (truth)" if o == truth else ""))
            for i, o in enumerate(options)
        ]
        handles.append(Line2D([], [], color=CONTROLLER_RING, marker="o", markersize=6,
                              markerfacecolor="none", linestyle="none",
                              label="controller post"))
        figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.485, 0.963),
                      ncol=len(handles), fontsize=9.5, frameon=False)
        if image is not None:
            bar = figure.add_axes((0.905, 0.32, 0.013, 0.36))
            figure.colorbar(image, cax=bar).set_label("posts in that round", fontsize=9)
        name = f"{arm}_rho{rho}" + (f"_b{budget}" if budget is not None else "") + ".png"
        out = args.out_dir / name.replace(".", "p").replace("ppng", ".png")
        figure.savefig(out, dpi=150)
        plt.close(figure)
        print(f"  {out.name}  ({len(loaded)} episodes, {len(rows)} facts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
