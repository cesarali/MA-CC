"""Step 1.1-1.2: the reading reversal from existing light records, with paired episode-level uncertainty, and the
dose (messages posted) at each reading level q.

For every setting (setup, case, rho, qc, rate, gate) and time t, and each q in {3, 6, 12}:
    g_e(q) = x_A2(e; false arm, q) - x_A2(e; silent arm, q)        (paired susceptibility of episode e)
    G(q)   = mean_e g_e(q),  G_se = sd_e / sqrt(n)
    F(q)   = mean messages the controller has posted by t
and for each pair of reading levels (3->6, 6->12, 3->12):
    D      = mean_e [g_e(q2) - g_e(q1)]                              (finite difference in q)
    D_se   = sd_e[g_e(q2) - g_e(q1)] / sqrt(n)                       (paired: same episode index at both q)
    D_se_unpaired = sqrt(G_se(q1)^2 + G_se(q2)^2)                    (for comparison)
Pairing across q is valid because every random stream is keyed by (seed, episode, purpose), never by q or arm
(simulation_2/simulator/sim2/simulation_2.py, keyed_rng): episode e at q = 3 and at q = 12 share clocks, lifetimes and
initial order; their reads differ only because q differs.

Vote shares x are among agents who have voted (build_tables.load). Cases: standard (B 30, W 48), budget100, window24,
window240. Writes results/key_phenomena/reversal.parquet (one row per setting, time and q-pair) and
reversal_G.parquet (one row per setting, time and q).

Usage, from the repository root:
    .venv/bin/python analysis/task003_llm_free/simulation_2/key_phenomena_hypothesis/scripts/s1_reversal.py [--workers 12]
"""
from __future__ import annotations
import argparse, multiprocessing, pathlib, sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]                                        # analysis/task003_llm_free
sys.path.insert(0, str(ROOT / "shared_analysis" / "gold_standard"))
from build_tables import load  # noqa: E402

RES = ROOT / "results"
OUT = RES / "key_phenomena"
RUNS = ["2026-10-15_sim2_symmetric_fine", "2026-10-16_sim2_symmetric_fine_extra_rho", "2026-10-16_sim2_nosolution_fine",
        "2026-10-18_sim2_symmetric_budget100", "2026-10-18_sim2_symmetric_window24",
        "2026-10-18_sim2_symmetric_window240", "2026-10-18_sim2_nosolution_budget100",
        "2026-10-18_sim2_nosolution_window24", "2026-10-18_sim2_nosolution_window240"]
QS = [3, 6, 12]
PAIRS = [(3, 6), (6, 12), (3, 12)]


def case_of(r):
    return "budget100" if r.budget != 30 else (f"window{int(r.W)}" if r.W != 48 else "standard")


def groups():
    inv = pd.read_csv(OUT / "inventory_cells.csv")
    inv = inv[inv.run.isin(RUNS)].copy()
    inv["case"] = inv.apply(case_of, axis=1)
    inv["folder"] = [str(RES / "simulation_2" / r / "cells" / c) for r, c in zip(inv.run, inv.cell)]
    sil = {(r.run, r.q, r.rho): r.folder for r in inv[inv.arm == "silent"].itertuples()}
    fa = inv[inv.arm == "false"]
    jobs = []
    for key, g in fa.groupby(["setup", "case", "rho", "qc", "lambda_c", "gate"]):
        g = g.set_index("q")
        assert sorted(g.index) == QS, key
        jobs.append((key, [(q, g.loc[q, "folder"], sil[(g.loc[q, "run"], q, key[2])]) for q in QS]))
    return jobs


def per_episode(folder, silent):
    c, s = load("sim2", pathlib.Path(folder)), load("sim2", pathlib.Path(silent))
    j = c[["episode", "t", "x2", "facts"]].merge(s[["episode", "t", "x2"]], on=["episode", "t"], suffixes=("", "_s"))
    j["g"] = j.x2 - j.x2_s
    return j[["episode", "t", "g", "facts"]].dropna()


def run(job):
    key, cells = job
    ep = {q: per_episode(f, s) for q, f, s in cells}
    grows, drows = [], []
    for q, d in ep.items():
        a = d.groupby("t").agg(G=("g", "mean"), sd=("g", "std"), n=("g", "size"), F=("facts", "mean"))
        for t, r in a.iterrows():
            grows.append((*key, q, t, r.G, r.sd / np.sqrt(r.n), r.F, int(r.n)))
    for q1, q2 in PAIRS:
        j = ep[q1].merge(ep[q2], on=["episode", "t"], suffixes=("1", "2"))
        j["d"] = j.g2 - j.g1
        a = j.groupby("t").agg(D=("d", "mean"), sd=("d", "std"), n=("d", "size"), v1=("g1", "var"), v2=("g2", "var"))
        for t, r in a.iterrows():
            drows.append((*key, f"{q1}->{q2}", q1, q2, t, r.D, r.sd / np.sqrt(r.n),
                          np.sqrt((r.v1 + r.v2) / r.n), int(r.n)))
    return grows, drows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    jobs = groups()
    print(len(jobs), "settings")
    with multiprocessing.Pool(a.workers) as pool:
        parts = pool.map(run, jobs, chunksize=4)
    k = ["setup", "case", "rho", "qc", "lambda_c", "gate"]
    G = pd.DataFrame([r for p in parts for r in p[0]], columns=k + ["q", "t", "G", "G_se", "F", "n"])
    D = pd.DataFrame([r for p in parts for r in p[1]],
                     columns=k + ["step", "q1", "q2", "t", "D", "D_se", "D_se_unpaired", "n"])
    G.to_parquet(OUT / "reversal_G.parquet", index=False)
    D.to_parquet(OUT / "reversal.parquet", index=False)
    print(len(G), "G rows,", len(D), "D rows ->", OUT)


if __name__ == "__main__":
    main()
