"""Q0 of the analysis plan: sanity at scale.

Every cell: messages <= budget; no controller action or post at t >= 30; agents act in the
follow-up; the controller reads only agent posts; budget counts consistent.
Full-table cells: no self-reads; reads only from the 48 most recent eligible posts; every
fact read is active afterwards; replay of reads and forgetting reproduces memories (20 episodes).
The grid's main cells at rate 1 equal the bridge's S4 (gate on, stop on), S5 (gate off, stop on)
and S6 (gate off, stop off) exactly.

Usage, from the repository root:
    .venv/bin/python analysis/task003_llm_free/simulation_2/analysis/q0_sanity.py [--run <run>] [--date <date>]
Output: tables/q0_invariants.csv, tables/q0_bridge_identity.csv; prints totals (all should be 0).
"""
from __future__ import annotations
import argparse, math, multiprocessing, pathlib

import numpy as np
import pandas as pd

from common import cells_table, out_dir, run_dir, write_index
from llmfree_core.setups import load_setup
from llmfree_core.world import World, bits
from common import ROOT

CONTROLLER = -1
WORLD_FILE = ROOT / "llmfree_core" / "world_task003.json"


def check_cell(args) -> dict:
    run, folder, prm = args
    f = pathlib.Path(run) / "cells" / folder
    out = {"folder": folder}
    ctrl = pd.read_parquet(f / "controller.parquet")
    posts = pd.read_parquet(f / "posts.parquet")
    ep = pd.read_parquet(f / "episodes.parquet")
    out["messages_over_budget"] = int((ep.controller_messages > prm["budget"]).sum())
    out["controller_action_after_cutoff"] = int((ctrl.t >= prm["t_control"]).sum())
    out["controller_post_after_cutoff"] = int(((posts.author == CONTROLLER) & (posts.t >= prm["t_control"])).sum())
    out["budget_negative"] = int((ctrl.budget_after < 0).sum())
    n_ctrl_posts = posts[posts.author == CONTROLLER].groupby("episode").size().reindex(ep.episode, fill_value=0)
    out["messages_mismatch"] = int((n_ctrl_posts.to_numpy() != ep.controller_messages.to_numpy()).sum())
    out["episodes_without_followup_posts"] = int(
        ep.episode.nunique() - posts[(posts.author >= 0) & (posts.t > prm["t_control"])].episode.nunique())
    # controller reads only agent posts
    reads = ctrl[["episode", "posts_read"]].explode("posts_read").dropna()
    reads = reads.astype({"posts_read": "int64"}).merge(posts[["episode", "post_id", "author"]],
                                                         left_on=["episode", "posts_read"], right_on=["episode", "post_id"])
    out["controller_read_own_post"] = int((reads.author == CONTROLLER).sum())
    if (f / "actions.parquet").exists():
        out.update(check_full(f, prm, posts))
    return out


def check_full(f: pathlib.Path, prm: dict, posts: pd.DataFrame) -> dict:
    act = pd.read_parquet(f / "actions.parquet")
    fg = pd.read_parquet(f / "forgetting.parquet")
    out = {"self_read": 0, "read_outside_window": 0, "read_fact_not_active": 0, "replay_mismatch": 0}
    out["read_fact_not_active"] = int(((act.active_after & act.facts_read) != act.facts_read).sum())
    by_ep = {e: g.sort_values("post_id") for e, g in posts.groupby("episode")}
    sample = act[act.episode < 20]
    for r in sample.itertuples():
        ids = list(r.posts_read)
        p = by_ep.get(r.episode)
        if not ids:
            continue
        earlier = p[(p.t < r.t) & (p.author != r.agent)]
        window = set(earlier.post_id.iloc[-prm["W"]:])
        read = p[p.post_id.isin(ids)]
        out["self_read"] += int((read.author == r.agent).sum())
        out["read_outside_window"] += int(not set(ids) <= window)
    # replay memories on 20 episodes
    world = World(WORLD_FILE)
    setup = load_setup(prm["setup"], ROOT / "experimental_setup", world)
    for e in range(20):
        ev = [(r.t, 4, "f", r) for r in fg[fg.episode == e].itertuples()]
        ev += [(r.t, 5, "a", r) for r in act[act.episode == e].itertuples()]
        ev.sort(key=lambda x: (x[0], x[1]))
        active = [1 << x for x in setup.agent_facts]
        for _, _, kind, r in ev:
            if kind == "f":
                if not (active[r.agent] >> r.fact) & 1:
                    out["replay_mismatch"] += 1
                active[r.agent] &= ~(1 << r.fact)
            else:
                if active[r.agent] != r.active_before:
                    out["replay_mismatch"] += 1
                    active[r.agent] = r.active_before
                active[r.agent] |= int(r.facts_read)
    return out


def bridge_identity(run: pathlib.Path, cells: pd.DataFrame) -> pd.DataFrame:
    bridge = ROOT / "results" / "simulation_2" / "2026-10-09_sim2_bridge" / "cells"
    rows = []
    for step, gate, stop in (("S4", "on", "on"), ("S5", "off", "on"), ("S6", "off", "off")):
        for setup in ("task003_symmetric", "task003_nosolution"):
            for arm in ("truth", "false"):
                c = cells[(cells.setup == setup) & (cells.arm == arm) & (cells.q == 6) & (cells.qc == 12)
                          & (cells.rho == 0.75) & (cells.lambda_c == 1.0) & (cells.gate == gate) & (cells.stop == stop)]
                if c.empty or not (bridge / f"{step}__{setup}__{arm}").exists():
                    continue
                same = True
                for t in ("snapshots", "controller", "posts", "episodes"):
                    a = pd.read_parquet(run / "cells" / c.folder.iloc[0] / f"{t}.parquet")
                    b = pd.read_parquet(bridge / f"{step}__{setup}__{arm}" / f"{t}.parquet")
                    n = min(a.episode.max(), b.episode.max())                  # episodes both runs have
                    a, b = (x[x.episode <= n].reset_index(drop=True) for x in (a, b))
                    try:
                        pd.testing.assert_frame_equal(a, b)
                    except AssertionError:
                        same = False
                rows.append({"bridge_cell": f"{step}__{setup}__{arm}", "grid_cell": c.cell.iloc[0], "identical": same})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--date")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    run = pathlib.Path(a.run).resolve() if a.run else run_dir("sim2_grid")
    d = out_dir(a.date)
    cells = cells_table(run)
    ran = cells[cells.cell == cells.folder]                                   # the cells that ran
    prm_cols = ["setup", "budget", "t_control", "W"]
    jobs = [(str(run), r.folder, {k: getattr(r, k) for k in prm_cols}) for r in ran.itertuples()]
    with multiprocessing.Pool(12) as pool:
        res = pd.DataFrame(pool.map(check_cell, jobs))
    res.to_csv(d / "tables" / f"{a.tag}q0_invariants.csv", index=False)
    totals = res.drop(columns="folder").fillna(0).sum().astype(int)
    print("INVARIANT VIOLATIONS (all should be 0):")
    print(totals.to_string())
    print(f"cells checked: {len(res)}; with full tables: {res.self_read.notna().sum()}")
    bi = bridge_identity(run, cells)
    bi.to_csv(d / "tables" / f"{a.tag}q0_bridge_identity.csv", index=False)
    print(f"\ngrid main cells identical to bridge S4-S6: {int(bi.identical.sum())} of {len(bi)}")
    write_index(d, [{"file": f"tables/{a.tag}q0_invariants.csv", "question": "Q0", "what": "invariant violations per cell"},
                    {"file": f"tables/{a.tag}q0_bridge_identity.csv", "question": "Q0", "what": "grid vs bridge identity"}])


if __name__ == "__main__":
    main()
