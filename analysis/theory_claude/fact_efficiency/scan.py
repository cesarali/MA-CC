"""Fact efficiency of the controller (pre-registered in ../../theory_blackboard/2026-11-08_claude.md, section 3).

How much effect does a controller get from F_C distinct facts and a posting budget? Fact-flow model, P0 except
N = 15, W = 30, F_A = 15 (the 15-agent campaign), eps = 1, T = 30, continuation to t = 130.

  scan A (fixed strength):  beta_C = 0.5; F_C in {1,2,4,8,16}; lam0 in {0.5,1,2}
  scan B (fixed total evidence F_C beta_C = 2): beta_C = 2/F_C; lam0 = 1
  mean field: q in {1,2,4,8,16}; finite: q in {2,4,8} (400 paired episodes); rho in {0.135,0.37,0.61,0.78,0.88}
Outputs: mf.csv, finite.csv (per cell: G30, G130, se, retained, controller facts circulating).

Usage, from the repository root:
    caffeinate -i .venv/bin/python analysis/theory_claude/fact_efficiency/scan.py --workers 12
"""
import argparse, multiprocessing, pathlib, sys
from dataclasses import replace

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "phenomenological_fact_flow"))
from fact_flow import P, phi2, run  # noqa: E402
from finite_model import FP, episode  # noqa: E402

FC = [1, 2, 4, 8, 16]
LAM = [0.5, 1.0, 2.0]
RHO = [0.135, 0.37, 0.61, 0.78, 0.88]
EP = 400


def mf_cells():
    rows = []
    for scan, lams in (("A", LAM), ("B", [1.0])):
        for F in FC:
            beta = 0.5 if scan == "A" else 2.0 / F
            p = replace(P(), W=30.0, F_A=15.0, F_C=float(F), beta_C=beta, eps=1.0)
            for lam in lams:
                p2 = replace(p, lam0=lam)
                for q in (1, 2, 4, 8, 16):
                    for rho in RHO:
                        te = np.array([30.0, 130.0])
                        c1, c2 = run(p2, q, rho, True, t_end=130.0, t_eval=te)
                        s1, s2 = run(p2, q, rho, False, t_end=130.0, t_eval=te)
                        g30 = phi2(p2, c1.y[0, -1], c1.y[1, -1]) - phi2(p2, s1.y[0, -1], s1.y[1, -1])
                        g130 = phi2(p2, c2.y[0, -1], c2.y[1, -1]) - phi2(p2, s2.y[0, -1], s2.y[1, -1])
                        rows.append(dict(scan=scan, F_C=F, beta_C=beta, lam0=lam, q=q, rho=rho, G30=g30, G130=g130,
                                         retained=g130 / g30 if g30 > 1e-9 else np.nan, M_C30=F * c1.y[1, -1],
                                         M_C130=F * c2.y[1, -1]))
    return pd.DataFrame(rows)


def finite_cell(args):
    scan, F, beta, lam, q, rho = args
    p = replace(FP(), N=15, W=30, F_A=15, F_C=F, beta_C=beta, lam0=lam, eps=1.0)
    rows = []
    for s in range(EP):
        for controlled in (True, False):
            rec = episode(p, q, rho, controlled, seed=10_000 * s + 7, t_end=130.0, record=(30.0, 130.0))
            for t, m in rec.items():
                rows.append(dict(scan=scan, F_C=F, beta_C=beta, lam0=lam, q=q, rho=rho, seed=s,
                                 arm="c" if controlled else "s", t=t, x2=m["x2"], distinct_C=m["distinct_C"],
                                 distinct_A=m["distinct_A"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    mf = mf_cells()
    mf.to_csv(HERE / "mf.csv", index=False)
    print("mean field done", len(mf), flush=True)
    jobs = [("A", F, 0.5, lam, q, rho) for F in FC for lam in LAM for q in (2, 4, 8) for rho in RHO]
    jobs += [("B", F, 2.0 / F, 1.0, q, rho) for F in FC for q in (2, 4, 8) for rho in RHO]
    with multiprocessing.Pool(a.workers) as pool:
        parts = pool.map(finite_cell, jobs, chunksize=1)
    d = pd.DataFrame([r for part in parts for r in part])
    d.to_parquet(HERE / "finite_episodes.parquet", index=False)
    key = ["scan", "F_C", "beta_C", "lam0", "q", "rho"]
    piv = d.pivot_table(index=key + ["seed", "t"], columns="arm", values="x2").reset_index()
    piv["g"] = piv.c - piv.s
    g = piv.groupby(key + ["t"]).g.agg(["mean", lambda x: x.std(ddof=1) / np.sqrt(len(x))]).reset_index()
    g.columns = key + ["t", "G", "se"]
    w = g.pivot_table(index=key, columns="t", values=["G", "se"]).reset_index()
    w.columns = [c[0] if c[1] == "" else f"{c[0]}{int(c[1])}" for c in w.columns]
    dc = d[d.arm == "c"].groupby(key + ["t"]).distinct_C.mean().unstack("t").reset_index()
    dc.columns = key + ["C_alive30", "C_alive130"]
    w = w.merge(dc, on=key)
    w["retained"] = np.where(w.G30 > 2 * w.se30, w.G130 / w.G30, np.nan)
    w.to_csv(HERE / "finite.csv", index=False)
    print("finite done", len(w))


if __name__ == "__main__":
    main()
