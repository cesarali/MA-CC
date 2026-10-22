"""Benchmark targets for the theory, with the pre-registered development / held-out split
(calibration/holdout_preregistration.md). Reads only existing runs; no simulation.

Targets (all Simulation 2, q_c 12, rate 1, gate on, B 30, W 48, false controller and silent arm):
  G_D        vote susceptibility G(rho, q) at t = 30 and reading increments D(q1 -> q2) with 95% bootstrap intervals
             (copied from tables/step2b_G.csv, step2b_D.csv; method in reports/2026-10-21_step2_reading_mechanism)
  G_time     G(t) every 0.25 to t = 40 for every (setup, rho, q)          (results/key_phenomena/s2_G.parquet)
  facts      per fact f: m_f = share of agents remembering f, b_f = share of the 48 newest posts carrying f, at
             t = 1, 2, 5, 10, 20, 30, 40, every (setup, rho, q, arm)      (raw 2026-10-20_sim2_reading_mechanism)
  post_stop  delta(t) to t = 130 for the four post-stop rules               (results/key_phenomena/s3_delta.parquet)
Each row gets split = "development" or "held_out" by the rule in holdout_preregistration.md.
Writes results/key_phenomena/theory_benchmarks/{G_D,G_time,facts,post_stop}.parquet and small CSV copies of G_D in
calibration/ (development rows) and held_out_validation/ (held-out rows).

Usage, from the repository root:
    caffeinate -i .venv/bin/python analysis/task003_llm_free/simulation_2/key_phenomena_hypothesis/calibration/build_benchmarks.py
"""
from __future__ import annotations
import multiprocessing, pathlib, sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(BASE / "scripts"))
from s1_mechanism import classes  # noqa: E402
from s2_mechanism import cells  # noqa: E402

KP = ROOT / "results" / "key_phenomena"
OUT = KP / "theory_benchmarks"
DEV_RHO, DEV_Q = {0.25, 0.75, 1.0}, {3, 6}
TIMES = [1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 40.0]
W = 48


def split_point(rho, q, t=None):
    """Development: rho in {0.25, 0.75, 1}, q in {3, 6}, t <= 30. Everything else is held out."""
    dev = rho in DEV_RHO and q in DEV_Q and (t is None or t <= 30.0)
    return "development" if dev else "held_out"


def split_step(rho, step):
    q1, q2 = map(int, step.split("->"))
    return "development" if rho in DEV_RHO and {q1, q2} <= DEV_Q else "held_out"


def facts(item):
    (setup, rho, q, arm), cell = item
    s = setup.replace("task003_", "")
    _, agent, ctrl = classes(s)
    universe = [f for f in range(64) if ((agent | ctrl) >> f) & 1]
    grid = {round(t / 0.25): t for t in TIMES}
    a = pd.read_parquet(cell / "agent_snapshots.parquet", columns=["episode", "grid_index", "active"])
    a = a[a.grid_index.isin(grid)]
    act = a.active.to_numpy(np.int64)
    rows = []
    gi = a.grid_index.to_numpy()
    for k, t in grid.items():
        sel = act[gi == k]
        for f in universe:
            rows.append((s, rho, q, arm, t, f, "agent" if (agent >> f) & 1 else "controller",
                         "m", float(((sel >> f) & 1).mean())))
    p = pd.read_parquet(cell / "posts.parquet", columns=["episode", "t", "fact"]).sort_values(["episode", "t"])
    share = {(t, f): [] for t in TIMES for f in universe}
    for _, d in p.groupby("episode", sort=False):
        tt, ff = d.t.to_numpy(), d.fact.to_numpy()
        for t in TIMES:
            k = np.searchsorted(tt, t, side="right")
            win = ff[max(0, k - W):k]
            if len(win) == 0:
                continue
            c = np.bincount(win, minlength=64)
            for f in universe:
                share[(t, f)].append(c[f] / len(win))
    for (t, f), v in share.items():
        rows.append((s, rho, q, arm, t, f, "agent" if (agent >> f) & 1 else "controller", "b",
                     float(np.mean(v)) if v else np.nan))
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tab = BASE / "tables"
    G = pd.read_csv(tab / "step2b_G.csv")
    G["split"] = [split_point(r, q) for r, q in zip(G.rho, G.q)]
    D = pd.read_csv(tab / "step2b_D.csv")
    D["split"] = [split_step(r, s) for r, s in zip(D.rho, D.step)]
    gd = pd.concat([G.assign(kind="G"), D.assign(kind="D")], ignore_index=True)
    gd.to_parquet(OUT / "G_D.parquet", index=False)
    for name, folder in (("development", HERE), ("held_out", BASE / "held_out_validation")):
        G[G.split == name].round(5).to_csv(folder / f"benchmark_G_{name}.csv", index=False)
        D[D.split == name].round(5).to_csv(folder / f"benchmark_D_{name}.csv", index=False)
    gt = pd.read_parquet(KP / "s2_G.parquet")
    gt["split"] = [split_point(r, q, t) for r, q, t in zip(gt.rho, gt.q, gt.t)]
    gt.to_parquet(OUT / "G_time.parquet", index=False)
    ps = pd.read_parquet(KP / "s3_delta.parquet")
    ps["split"] = np.where(ps.t <= 30.0, "pre_stop_reference", "held_out")
    ps.to_parquet(OUT / "post_stop.parquet", index=False)
    with multiprocessing.Pool(12) as pool:
        parts = pool.map(facts, cells().items())
    fa = pd.DataFrame([r for p in parts for r in p],
                      columns=["setup", "rho", "q", "arm", "t", "fact", "fact_class", "quantity", "value"])
    fa["split"] = [split_point(r, q, t) for r, q, t in zip(fa.rho, fa.q, fa.t)]
    fa.to_parquet(OUT / "facts.parquet", index=False)
    print({k: int((d.split == "development").sum()) for k, d in (("G_D", gd), ("G_time", gt), ("facts", fa))},
          "development rows ->", OUT)


if __name__ == "__main__":
    main()
