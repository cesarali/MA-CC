"""Q8 of the analysis plan (exploratory): mechanisms, on the full-table cells (q 6, qc 12, rho 0.75).

Per cell, means over episodes:
  - proof rates of A0 at t = 30: from all active facts, and from the agents' original facts only;
  - mean active facts at t = 30;
  - per agent action: new facts read, reactivated facts read (forgotten, read again), posts read;
  - agent posts that relay a controller fact (a pool fact no agent started with);
  - share of controller posts among all posts during control (t < 30);
  - facts forgotten per agent per time unit.

Usage, from the repository root:
    .venv/bin/python analysis/task003_llm_free/simulation_2/analysis/q8_mechanisms.py [--run <run>] [--date <date>]
Output: tables/q8_mechanisms.csv
"""
from __future__ import annotations
import argparse, multiprocessing, pathlib

import numpy as np
import pandas as pd

from common import ROOT, cells_table, out_dir, run_dir, write_index
from llmfree_core.setups import load_setup
from llmfree_core.world import World

WORLD_FILE = ROOT / "llmfree_core" / "world_task003.json"


def popcount(a: pd.Series) -> np.ndarray:
    return np.array([int(x).bit_count() for x in a])


def cell_mechanisms(args) -> dict:
    run, folder, setup_name, arm = args
    f = pathlib.Path(run) / "cells" / folder
    world = World(WORLD_FILE)
    setup = load_setup(setup_name, ROOT / "experimental_setup", world)
    agent_facts = set(setup.agent_facts)
    pool = set(setup.pools[0]) | set(setup.pools[2])
    controller_only = pool - agent_facts
    snap = pd.read_parquet(f / "snapshots.parquet")
    s30 = snap[snap.t == 30.0]
    act = pd.read_parquet(f / "actions.parquet", columns=["episode", "t", "known_before", "active_before", "facts_read",
                                                          "posts_read", "posted_fact"])
    new = act.facts_read.to_numpy() & ~act.known_before.to_numpy()
    react = act.facts_read.to_numpy() & act.known_before.to_numpy() & ~act.active_before.to_numpy()
    posts = pd.read_parquet(f / "posts.parquet")
    ag = posts[posts.author >= 0]
    fg = pd.read_parquet(f / "forgetting.parquet", columns=["episode"])
    n_ep = snap.episode.nunique()
    control = posts[posts.t < 30.0]
    return {
        "folder": folder,
        "proof_rate_A0_t30": s30.proof_rate_A0.mean(),
        "proof_rate_A0_own_evidence_t30": s30.proof_rate_A0_own_evidence.mean(),
        "mean_active_facts_t30": s30.mean_active_facts.mean(),
        "new_facts_per_action": popcount(pd.Series(new)).mean(),
        "reactivated_facts_per_action": popcount(pd.Series(react)).mean(),
        "posts_read_per_action": act.posts_read.map(len).mean(),
        "agent_posts_relaying_controller_facts": ag.fact.isin(controller_only).mean() if len(ag) else np.nan,
        "controller_share_of_posts_t_lt_30": (control.author < 0).mean(),
        "facts_forgotten_per_agent_per_unit": len(fg) / n_ep / 24 / 40,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--date")
    a = ap.parse_args()
    run = pathlib.Path(a.run).resolve() if a.run else run_dir("sim2_grid")
    d = out_dir(a.date)
    cells = cells_table(run)
    full = cells[cells.cell == cells.folder]                                  # the cells that ran
    full = full[[(run / "cells" / x / "actions.parquet").exists() for x in full.folder]]
    with multiprocessing.Pool(12) as pool:
        res = pd.DataFrame(pool.map(cell_mechanisms, [(str(run), r.folder, r.setup, r.arm) for r in full.itertuples()]))
    res = full[["folder", "setup", "arm", "lambda_c", "gate", "stop"]].merge(res, on="folder")
    res.to_csv(d / "tables" / "q8_mechanisms.csv", index=False)
    write_index(d, [{"file": "tables/q8_mechanisms.csv", "question": "Q8", "what": "mechanism measures, main cells"}])
    print(res.drop(columns="folder").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
