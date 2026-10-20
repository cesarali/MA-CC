"""Step 1.3: what agents remember and what the board carries, from the existing full records (q = 6, qc = 12,
rate 1, gate on; rho 0.75 and 1; both setups), and, at rho = 1 in nosolution, the board against q (posts only).

Fact classes (fixed by the setup files): agent-origin = the facts the 24 agents start with; controller-only = facts
of the false controller's pool (target A2) that no agent starts with (symmetric: 10 of 12; nosolution: 12 of 12).

Per episode, at t in TIMES, from agent_snapshots (each agent's remembered facts as a bitmask) and posts:
  mem_facts      mean facts an agent remembers
  mem_agent      mean agent-origin facts an agent remembers
  mem_ctrl       mean controller-only facts an agent remembers
  share_ctrl_any share of agents remembering at least one controller-only fact
  cover_agent    distinct agent-origin facts remembered by anyone (collective coverage, of 18 or 15)
  p_A0, p_A2     mean exact belief; entropy (bits) of each agent's belief, averaged
  proof          share of agents whose memory proves A0 (symmetric only can be > 0)
  board_ctrl     share of the 48 newest posts that carry a controller-only fact (the readable window, approx.)
  board_agent_distinct  distinct agent-origin facts among those 48 posts
Writes results/key_phenomena/mechanism.parquet (episode means, by cell and time) and mechanism_board_q.parquet.

Usage, from the repository root:
    .venv/bin/python analysis/task003_llm_free/simulation_2/key_phenomena_hypothesis/scripts/s1_mechanism.py
"""
from __future__ import annotations
import math, multiprocessing, pathlib, sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from llmfree_core.world import World, bits  # noqa: E402
from llmfree_core.setups import load_setup  # noqa: E402

RES = ROOT / "results" / "simulation_2"
OUT = ROOT / "results" / "key_phenomena"
TIMES = [1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 31.0, 32.0, 35.0, 40.0]
W = 48
CELLS = [  # (setup, rho, arm, run, cell)
    ("symmetric", 0.75, "false", "2026-10-09_sim2_grid", "task003_symmetric__false__q6__qc12__rho0.75__lc1__gateon__stopon"),
    ("symmetric", 0.75, "silent", "2026-10-09_sim2_grid", "task003_symmetric__silent__q6__rho0.75"),
    ("symmetric", 1.0, "false", "2026-10-12_sim2_main_rho1_full", "task003_symmetric__false__lc1__gateon__stopon"),
    ("symmetric", 1.0, "silent", "2026-10-12_sim2_main_rho1_full", "task003_symmetric__silent"),
    ("nosolution", 0.75, "false", "2026-10-09_sim2_grid_confirm_seed2027", "task003_nosolution__false__lc1__gateon__stopon"),
    ("nosolution", 0.75, "silent", "2026-10-09_sim2_grid_confirm_seed2027", "task003_nosolution__silent"),
    ("nosolution", 1.0, "false", "2026-10-12_sim2_main_rho1_full", "task003_nosolution__false__lc1__gateon__stopon"),
    ("nosolution", 1.0, "silent", "2026-10-12_sim2_main_rho1_full", "task003_nosolution__silent"),
]
BOARD_Q_RUN = "2026-10-09_sim2_rho1_confirm_seed2027"     # nosolution, rho 1, q 3/6/12, posts recorded


def classes(setup):
    world = World(ROOT / "llmfree_core" / "world_task003.json")
    s = load_setup(f"task003_{setup}", ROOT / "experimental_setup", world)
    agent = 0
    for f in s.agent_facts:
        agent |= 1 << f
    ctrl = 0
    for f in s.pools[2]:
        if not (agent >> f) & 1:
            ctrl |= 1 << f
    return world, agent, ctrl


def entropy(p):
    return -sum(x * math.log2(x) for x in p if x > 0)


def board_stats(posts: pd.DataFrame, agent: int, ctrl: int, times) -> pd.DataFrame:
    """Per episode and time: the 48 newest posts at that time (all authors; readers exclude their own posts, so this
    is the window up to one post)."""
    rows = []
    for ep, d in posts.groupby("episode", sort=False):
        tt, ff = d.t.to_numpy(), d.fact.to_numpy()
        for t in times:
            k = np.searchsorted(tt, t, side="right")
            win = ff[max(0, k - W):k]
            if len(win) == 0:
                continue
            isc = np.array([(ctrl >> int(f)) & 1 for f in win], bool)
            isa = np.array([(agent >> int(f)) & 1 for f in win], bool)
            rows.append((ep, t, isc.mean(), len(set(win[isa].tolist()))))
    return pd.DataFrame(rows, columns=["episode", "t", "board_ctrl", "board_agent_distinct"])


def run_cell(spec):
    setup, rho, arm, run, cell = spec
    world, agent, ctrl = classes(setup)
    folder = RES / run / "cells" / cell
    snaps = pd.read_parquet(folder / "agent_snapshots.parquet", columns=["episode", "grid_index", "agent", "active"])
    grid = {round(t / 0.25): t for t in TIMES}
    snaps = snaps[snaps.grid_index.isin(grid)]
    post = {}
    for m in snaps.active.unique():
        m = int(m)
        p = world.posterior(m)
        post[m] = (bin(m).count("1"), bin(m & agent).count("1"), bin(m & ctrl).count("1"), p[0], p[2], entropy(p),
                   float(p[0] == 1.0))
    v = np.array([post[int(m)] for m in snaps.active.to_numpy()])
    a = pd.DataFrame(v, columns=["mem_facts", "mem_agent", "mem_ctrl", "p_A0", "p_A2", "entropy", "proof"])
    a["share_ctrl_any"] = (a.mem_ctrl > 0).astype(float)
    a["episode"], a["t"] = snaps.episode.to_numpy(), snaps.grid_index.map(grid).to_numpy()
    a["active"] = snaps.active.to_numpy()
    per = a.groupby(["episode", "t"]).mean(numeric_only=True).drop(columns="active")
    cover = a.groupby(["episode", "t"]).active.agg(lambda s: bin(int(np.bitwise_or.reduce(s.to_numpy())) & agent).count("1"))
    per["cover_agent"] = cover
    posts = pd.read_parquet(folder / "posts.parquet", columns=["episode", "t", "fact"]).sort_values(["episode", "t"])
    per = per.reset_index().merge(board_stats(posts, agent, ctrl, TIMES), on=["episode", "t"], how="left")
    per.insert(0, "setup", setup)
    per.insert(1, "rho", rho)
    per.insert(2, "arm", arm)
    return per


def run_board_q(spec):
    q, arm, cell = spec
    _, agent, ctrl = classes("nosolution")
    posts = pd.read_parquet(RES / BOARD_Q_RUN / "cells" / cell / "posts.parquet", columns=["episode", "t", "fact"])
    b = board_stats(posts.sort_values(["episode", "t"]), agent, ctrl, TIMES)
    b.insert(0, "q", q)
    b.insert(1, "arm", arm)
    return b


def main():
    with multiprocessing.Pool(8) as pool:
        parts = pool.map(run_cell, CELLS)
        bq = pool.map(run_board_q, [(q, "false", f"task003_nosolution__false__q{q}__qc12__lc1__gateon")
                                    for q in (3, 6, 12)] +
                      [(q, "silent", f"task003_nosolution__silent__q{q}") for q in (3, 6, 12)])
    pd.concat(parts, ignore_index=True).to_parquet(OUT / "mechanism.parquet", index=False)
    pd.concat(bq, ignore_index=True).to_parquet(OUT / "mechanism_board_q.parquet", index=False)
    print("->", OUT / "mechanism.parquet", OUT / "mechanism_board_q.parquet")


if __name__ == "__main__":
    main()
