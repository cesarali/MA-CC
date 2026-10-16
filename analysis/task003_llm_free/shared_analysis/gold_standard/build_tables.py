"""Summary tables of the 15 October symmetric runs (Simulations 1 and 2), for the reports of 15 October.

Reads the light records:
  results/simulation_1/2026-10-15_sim1_pm_symmetric_fine  (days.parquet: one row per episode and day)
  results/simulation_2/2026-10-15_sim2_symmetric_fine     (snapshots.parquet: one row per episode every 0.25)
and writes results/gold_standard/<date>/cells_time.parquet: one row per (cell, time) with

  identity   sim, arm, q, qc, x (b or rate), rho, gate, t
  votes      x0, x1, x2: mean share of votes for A0, A1, A2 among agents who have voted (Simulation 2:
             agents who have not acted yet have no vote and are left out; Simulation 1: everyone votes daily)
  beliefs    m0, m2: mean belief in A0 and A2 (each agent's exact posterior given its facts)
  proofs     proof_any, proof_own: share of agents proving A0 from any facts / from their original facts
  cost       facts: mean facts the controller has posted up to t
  gain       G, G_se: paired susceptibility on the controller's own target (A0 for target A0, A2 for
             target A2): mean over episodes of [x_target(controlled) - x_target(silent)], same episode,
             with its standard error; silent rows have G = 0
  states     distributions of the state X over episodes (each episode contributes a probability vector):
             vote_*   one agent's vote (A0, A1, A2): the episode's vote shares
             plur_*   plurality of the last votes (A0, A1, A2 wins; a k-way tie gives 1/k to each)
             s1_*     S1: m0 in 4 bins at 0.33 / 0.50 / 0.75
             s3_*     S3: (m0, m2), each in 3 equal bins on [0, 1] (9 cells, index 3*bin(m0) + bin(m2))

Times: Simulation 1 days 1..30 (end of each day); Simulation 2 t = 0.25, 0.5, 0.75, 1, 2, ..., 40. At t = 0 nobody
has voted; the figures start every vote share and gain at 0 there.

Usage, from the repository root:
    .venv/bin/python analysis/task003_llm_free/shared_analysis/gold_standard/build_tables.py [--date 2026-10-15] [--workers 12]
"""
from __future__ import annotations
import argparse, json, multiprocessing, pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
RES = ROOT / "results"
RUNS = {"sim1": RES / "simulation_1/2026-10-15_sim1_pm_symmetric_fine/cells",
        "sim2": RES / "simulation_2/2026-10-15_sim2_symmetric_fine/cells"}
N = 24
T2 = [0.25, 0.5, 0.75] + [float(t) for t in range(1, 41)]


def load(sim: str, folder: pathlib.Path) -> pd.DataFrame:
    """Per episode and time: vote shares among voters, beliefs, proofs, cumulative facts posted."""
    if sim == "sim1":
        d = pd.read_parquet(folder / "days.parquet")
        d = d.sort_values(["episode", "day"])
        d["t"] = d.day + 1.0
        # facts posted on the nights before day t (they are on the board the agents read on day t)
        d["facts"] = d.groupby("episode").controller_posts_tonight.cumsum() - d.controller_posts_tonight
        voted = pd.Series(1.0, index=d.index)
    else:
        d = pd.read_parquet(folder / "snapshots.parquet")
        d = d[d.t.isin(T2)].copy()
        d["facts"] = d.controller_messages.astype(float)
        voted = d.share_voted
    out = pd.DataFrame({"episode": d.episode.to_numpy(), "t": d.t.to_numpy(),
                        "m0": d.mean_p_A0.to_numpy(), "m2": d.mean_p_A2.to_numpy(),
                        "proof_any": d.proof_rate_A0.to_numpy(), "proof_own": d.proof_rate_A0_own_evidence.to_numpy(),
                        "facts": d.facts.to_numpy(), "voted": voted.to_numpy()})
    for k in range(3):
        out[f"x{k}"] = np.where(out.voted > 0, d[f"share_A{k}"].to_numpy() / np.maximum(out.voted, 1e-12), np.nan)
    return out


def states(e: pd.DataFrame) -> pd.DataFrame:
    """Per-episode probability vectors of the coarse states."""
    s = pd.DataFrame(index=e.index)
    n = np.rint(e[["x0", "x1", "x2"]].to_numpy() * np.rint(e.voted.to_numpy() * N)[:, None])
    top = n == n.max(axis=1, keepdims=True)
    pl = top / top.sum(axis=1, keepdims=True)
    for k in range(3):
        s[f"vote_{k}"] = e[f"x{k}"]
        s[f"plur_{k}"] = pl[:, k]
    b1 = np.digitize(e.m0, [0.33, 0.50, 0.75])
    for k in range(4):
        s[f"s1_{k}"] = (b1 == k) * 1.0
    b3 = 3 * np.digitize(e.m0, [1 / 3, 2 / 3]) + np.digitize(e.m2, [1 / 3, 2 / 3])
    for k in range(9):
        s[f"s3_{k}"] = (b3 == k) * 1.0
    return s


def summarise(job):
    sim, folder, silent_folder = job
    prm = json.loads((folder / "params.json").read_text())
    e = load(sim, folder)
    e = e[e.voted > 0]
    st = states(e)
    keys = ["x0", "x1", "x2", "m0", "m2", "proof_any", "proof_own", "facts"]
    g = pd.concat([e[["t"] + keys], st], axis=1).groupby("t").mean()
    g["episodes"] = e.groupby("t").episode.nunique()
    arm = prm["arm"]
    if arm == "silent":
        g["G"], g["G_se"] = 0.0, 0.0
    else:
        s = load(sim, silent_folder)
        k = 0 if arm == "truth" else 2
        j = e[["episode", "t", f"x{k}"]].merge(s[["episode", "t", f"x{k}"]], on=["episode", "t"], suffixes=("", "_s"))
        diff = (j[f"x{k}"] - j[f"x{k}_s"]).groupby(j.t)
        g["G"], g["G_se"] = diff.mean(), diff.std() / np.sqrt(diff.count())
    g = g.reset_index()
    g.insert(0, "sim", sim)
    g.insert(1, "cell", folder.name)
    g.insert(2, "arm", arm)
    g.insert(3, "q", prm["q"])
    g.insert(4, "qc", prm.get("qc"))
    g.insert(5, "x", prm.get("b") if sim == "sim1" else prm.get("lambda_c"))
    g.insert(6, "rho", float(prm["rho"]))
    gate = prm.get("controller_gate", "votes")
    g.insert(7, "gate", None if arm == "silent" else ("on" if gate == "votes" else "off"))
    return g


def jobs():
    out = []
    for sim, cells in RUNS.items():
        for f in sorted(p for p in cells.iterdir() if p.is_dir()):
            prm = json.loads((f / "params.json").read_text())
            if prm["arm"] == "silent":
                silent = None
            elif sim == "sim2":
                silent = cells / prm["silent_cell"]
            else:
                silent = cells / f"task003_symmetric__silent__q{prm['q']}__rho{float(prm['rho'])}"
            assert silent is None or silent.exists(), silent
            out.append((sim, f, silent))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2026-10-15")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    out = RES / "gold_standard" / a.date
    out.mkdir(parents=True, exist_ok=True)
    js = jobs()
    with multiprocessing.Pool(a.workers) as pool:
        parts = pool.map(summarise, js, chunksize=8)
    t = pd.concat(parts, ignore_index=True)
    t.to_parquet(out / "cells_time.parquet", index=False)
    print(f"{len(js)} cells, {len(t)} rows -> {out / 'cells_time.parquet'}")


if __name__ == "__main__":
    main()
