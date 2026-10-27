"""Step 3 runs of the finite model (decision_log.md, 2026-10-28): F1-F6.
  reversal grid   rho 0.1..0.9, q 3/6, 400 paired episodes: G, D(3->6) with standard errors; diversity at T
  persistence     rho 0.25/0.75, q 6, four continuations, to t = 130 (record every 2 time units)
  scaling         rho 0.5, q 6: R copies of the swarm (N, W, lam0 x R; facts fixed) and K copies of everything
Mean-field comparison from fact_flow.py at the same parameters (eps = 1).
Writes findings/step3_reversal.csv, step3_persistence.csv, step3_scaling.csv.

Usage, from this folder:  caffeinate -i ../../../.venv/bin/python step3_finite_runs.py --workers 7
"""
import argparse, multiprocessing, pathlib
from dataclasses import replace

import numpy as np
import pandas as pd

from finite_model import FP, episode
from fact_flow import P, G_at_T

HERE = pathlib.Path(__file__).resolve().parent
EP = 400
RHOS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
GRID_T = tuple(float(t) for t in range(30, 131, 2))


def cell(args):
    p, q, rho, mode, t_end, eps_n, tag = args
    rows = []
    for s in range(eps_n):
        for controlled in (True, False):
            rec = episode(p, q, rho, controlled, seed=10_000 * s + 7, t_end=t_end, mode=mode,
                          record=GRID_T if t_end > 30 else (30.0,))
            for t, m in rec.items():
                rows.append(dict(tag=tag, q=q, rho=rho, mode=mode, seed=s, arm="false" if controlled else "silent", t=t, **m))
    return rows


def paired(df, by):
    piv = df.pivot_table(index=by + ["seed"], columns="arm", values="x2")
    return (piv["false"] - piv["silent"]).rename("d").reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=7)
    a = ap.parse_args()
    p = FP()
    jobs = [(p, q, r, "normal", 30.0, EP, "base") for r in RHOS for q in (3, 6)]
    jobs += [(p, 6, r, m, 130.0, EP, "base") for r in (0.25, 0.75) for m in ("normal", "frozen", "no_comm", "purge")]
    for R in (1, 2, 4):
        jobs.append((replace(p, N=24 * R, W=48 * R, lam0=1.0 * R), 6, 0.5, "normal", 30.0, EP, f"R{R}"))
    for K in (2, 4):
        jobs.append((replace(p, N=24 * K, W=48 * K, lam0=1.0 * K, F_A=16 * K, F_C=4 * K), 6, 0.5, "normal", 30.0, EP, f"K{K}"))
    with multiprocessing.Pool(a.workers) as pool:
        parts = pool.map(cell, jobs, chunksize=1)
    d = pd.DataFrame([r for part in parts for r in part])
    d.to_parquet(HERE / "findings" / "step3_raw.parquet", index=False)

    # reversal grid
    rv = d[(d.tag == "base") & (d["mode"] == "normal") & np.isclose(d.t, 30.0) & d.rho.isin(RHOS)]
    rv = rv[~((rv.q == 6) & rv.rho.isin([0.25, 0.75]))] if False else rv
    g = paired(rv, ["q", "rho"])
    rows = []
    pm = replace(P(), eps=1.0)
    for rho in RHOS:
        g3 = g[(g.q == 3) & np.isclose(g.rho, rho)].set_index("seed").d
        g6 = g[(g.q == 6) & np.isclose(g.rho, rho)].set_index("seed").d
        dd = (g6 - g3).dropna()
        div = rv[np.isclose(rv.rho, rho) & (rv.q == 6)].groupby("arm")[["distinct_A", "copies_A", "holders_A", "distinct_C"]].mean()
        rows.append(dict(rho=rho, G3=g3.mean(), G6=g6.mean(), G6_se=g6.std() / np.sqrt(len(g6)), D36=dd.mean(),
                         D36_se=dd.std() / np.sqrt(len(dd)), G6_meanfield=G_at_T(pm, 6, rho)[0],
                         D36_meanfield=G_at_T(pm, 6, rho)[0] - G_at_T(pm, 3, rho)[0],
                         distinct_A_false=div.loc["false", "distinct_A"], distinct_A_silent=div.loc["silent", "distinct_A"],
                         copies_A_false=div.loc["false", "copies_A"], holders_A_false=div.loc["false", "holders_A"],
                         distinct_C_false=div.loc["false", "distinct_C"]))
    r1 = pd.DataFrame(rows)
    r1.to_csv(HERE / "findings" / "step3_reversal.csv", index=False)

    # persistence
    ps = d[(d.tag == "base") & (d.t >= 30) & d.rho.isin([0.25, 0.75]) & (d.q == 6)]
    ps = ps[(ps["mode"] != "normal") | (ps.t > 30.0) | True]
    pp = paired(ps[ps.groupby(["mode"]).t.transform("max") > 30], ["rho", "mode", "t"])
    r2 = pp.groupby(["rho", "mode", "t"]).d.agg(["mean", "std", "count"]).reset_index()
    r2["se"] = r2["std"] / np.sqrt(r2["count"])
    r2.to_csv(HERE / "findings" / "step3_persistence.csv", index=False)

    # scaling
    sc = d[d.tag.str.match(r"^[RK]\d") & np.isclose(d.t, 30.0)]
    gs = paired(sc, ["tag"])
    r3 = gs.groupby("tag").d.agg(["mean", "std", "count"]).reset_index()
    r3["se"] = r3["std"] / np.sqrt(r3["count"])
    r3["G_meanfield"] = G_at_T(pm, 6, 0.5)[0]
    r3.to_csv(HERE / "findings" / "step3_scaling.csv", index=False)

    pd.set_option("display.width", 250)
    print(r1.round(4).to_string(index=False))
    summ = r2[r2.t.isin([30.0, 32.0, 40.0, 80.0, 130.0])].pivot_table(index=["rho", "mode"], columns="t", values="mean")
    print(summ.round(4).to_string())
    print(r3.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
