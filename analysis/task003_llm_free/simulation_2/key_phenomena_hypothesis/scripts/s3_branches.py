"""Step 3 (Campaign B): what carries the persistence, from the run <date>_sim2_persistence_branches (both setups,
rho 0.25 and 0.75, q 6, rate 1, qc 12, gate on, t_end 130; post_stop_mode normal / no_comm / frozen_board /
purge_controller_facts, each with its own silent arm).

1. Validation: the normal cells must equal the fine runs for t <= 40 (snapshots) and in the controller table.
2. delta(t) = mean over episodes of [x_A2(false, mode) - x_A2(silent, same mode)], every 0.25 to t = 130, with
   paired standard errors; half-time after t = 30 relative to delta(30) (censored at 130 when not reached).
3. Memory and board composition at TIMES for every cell (as s1_mechanism.py). For the purge mode the purged
   controller-only posts (t <= 30) are left out of the board before taking the 48 newest posts.
Writes results/key_phenomena/s3_validation.csv, s3_delta.parquet, s3_summary.csv, s3_composition.parquet.

Usage, from the repository root:
    .venv/bin/python analysis/task003_llm_free/simulation_2/key_phenomena_hypothesis/scripts/s3_branches.py
"""
from __future__ import annotations
import json, multiprocessing, pathlib, sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
from s1_mechanism import board_stats, classes, entropy  # noqa: E402

RES = ROOT / "results" / "simulation_2"
OUT = ROOT / "results" / "key_phenomena"
MODES = ["normal", "no_comm", "frozen_board", "purge_controller_facts"]
TIMES = [30.0, 30.5, 31.0, 32.0, 35.0, 40.0, 50.0, 60.0, 80.0, 100.0, 130.0]
FINE = {"task003_symmetric": "2026-10-15_sim2_symmetric_fine", "task003_nosolution": "2026-10-16_sim2_nosolution_fine"}


def run_dir():
    runs = sorted(RES.glob("*_sim2_persistence_branches"))
    assert runs, "run sim2_persistence_branches first"
    return runs[-1]


def cells():
    out = {}
    for c in (run_dir() / "cells").iterdir():
        prm = json.loads((c / "params.json").read_text())
        out[(prm["setup"], float(prm["rho"]), prm["post_stop_mode"], prm["arm"])] = c
    return out


def validate(item):
    (setup, rho, mode, arm), cell = item
    if mode != "normal":
        return None
    base = RES / FINE[setup] / "cells"
    old = base / (f"{setup}__silent__q6__rho{rho:g}" if arm == "silent"
                  else f"{setup}__false__q6__qc12__rho{rho:g}__lc1__gateon")
    a, b = pd.read_parquet(cell / "snapshots.parquet"), pd.read_parquet(old / "snapshots.parquet")
    same = a[a.grid_index <= 160].reset_index(drop=True).equals(b.reset_index(drop=True))
    same &= pd.read_parquet(cell / "controller.parquet").equals(pd.read_parquet(old / "controller.parquet"))
    return (setup, rho, arm, bool(same))


def x2(cell):
    s = pd.read_parquet(cell / "snapshots.parquet", columns=["episode", "grid_index", "t", "share_A2", "share_voted"])
    s["x2"] = np.where(s.share_voted > 0, s.share_A2 / s.share_voted.clip(lower=1e-12), np.nan)
    return s[["episode", "grid_index", "t", "x2"]]


def delta(item):
    (setup, rho, mode), (f, s) = item
    j = x2(f).merge(x2(s), on=["episode", "grid_index", "t"], suffixes=("", "_s"))
    j["d"] = j.x2 - j.x2_s
    g = j.groupby("t").d.agg(["mean", "std", "count"]).reset_index()
    g["se"] = g["std"] / np.sqrt(g["count"])
    g.insert(0, "setup", setup.replace("task003_", ""))
    g.insert(1, "rho", rho)
    g.insert(2, "mode", mode)
    return g.rename(columns={"mean": "delta", "count": "n"})[["setup", "rho", "mode", "t", "delta", "se", "n"]]


def composition(item):
    (setup, rho, mode, arm), cell = item
    world, agent, ctrl = classes(setup.replace("task003_", ""))
    grid = {round(t / 0.25): t for t in TIMES}
    snaps = pd.read_parquet(cell / "agent_snapshots.parquet", columns=["episode", "grid_index", "active"])
    snaps = snaps[snaps.grid_index.isin(grid)]
    post = {}
    for m in snaps.active.unique():
        m = int(m)
        p = world.posterior(m)
        post[m] = (bin(m & agent).count("1"), bin(m & ctrl).count("1"), p[0], p[2], entropy(p))
    v = np.array([post[int(m)] for m in snaps.active.to_numpy()])
    a = pd.DataFrame(v, columns=["mem_agent", "mem_ctrl", "p_A0", "p_A2", "entropy"])
    a["episode"], a["t"] = snaps.episode.to_numpy(), snaps.grid_index.map(grid).to_numpy()
    a["active"] = snaps.active.to_numpy()
    g = a.groupby(["episode", "t"])
    per = g.mean(numeric_only=True).drop(columns="active")
    per["cover_agent"] = g.active.agg(lambda s: bin(int(np.bitwise_or.reduce(s.to_numpy())) & agent).count("1"))
    posts = pd.read_parquet(cell / "posts.parquet", columns=["episode", "t", "fact"]).sort_values(["episode", "t"])
    if mode == "purge_controller_facts":
        is_ctrl = np.array([(ctrl >> int(f)) & 1 for f in posts.fact], bool)
        posts = posts[~(is_ctrl & (posts.t <= 30.0).to_numpy())]
    per = per.reset_index().merge(board_stats(posts, agent, ctrl, TIMES), on=["episode", "t"], how="left")
    m = per.drop(columns="episode").groupby("t").mean().reset_index()
    for i, (k, val) in enumerate((("setup", setup.replace("task003_", "")), ("rho", rho), ("mode", mode), ("arm", arm))):
        m.insert(i, k, val)
    return m


def summary(D):
    rows = []
    for (s, r, mode), d in D.groupby(["setup", "rho", "mode"]):
        d = d.sort_values("t")
        d30 = d.loc[d.t == 30.0, "delta"].iloc[0]
        after = d[d.t >= 30.0]
        below = after[after.delta <= d30 / 2]
        row = dict(setup=s, rho=r, mode=mode, delta30=d30, half_time=(below.t.iloc[0] - 30) if len(below) else np.nan)
        for tt in (31.0, 35.0, 40.0, 50.0, 80.0, 130.0):
            x = d[d.t == tt].iloc[0]
            row[f"delta{int(tt)}"], row[f"se{int(tt)}"] = x.delta, x.se
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    cs = cells()
    with multiprocessing.Pool(12) as pool:
        val = [v for v in pool.map(validate, cs.items()) if v is not None]
        pairs = {(s, r, m): (c, cs[(s, r, m, "silent")]) for (s, r, m, a), c in cs.items() if a == "false"}
        D = pd.concat(pool.map(delta, pairs.items()), ignore_index=True)
        comp = pd.concat(pool.map(composition, cs.items()), ignore_index=True)
    v = pd.DataFrame(val, columns=["setup", "rho", "arm", "identical"])
    v.to_csv(OUT / "s3_validation.csv", index=False)
    print(f"validation: {int(v.identical.sum())}/{len(v)} normal cells identical to the fine runs up to t = 40")
    D.to_parquet(OUT / "s3_delta.parquet", index=False)
    summary(D).to_csv(OUT / "s3_summary.csv", index=False)
    comp.to_parquet(OUT / "s3_composition.parquet", index=False)
    print("->", OUT)


if __name__ == "__main__":
    main()
