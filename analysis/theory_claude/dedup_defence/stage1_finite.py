"""Stage 1 of the deduplicated-reading defence (pre-registered in 00_preregistration.md): finite fact-flow model,
P0, eps = 1, standard ("posts") against deduplicated ("dedup") reading.
  reversal cells:    q 3/6 x rho 0.25/0.5/0.75, 400 paired episodes, measured at t = 30
  persistence cells: q 6 x rho 0.25/0.75, normal continuation to t = 130 (every 10 time units)
Writes stage1_episodes.csv and stage1_summary.csv.

Usage, from the repository root:
    caffeinate -i .venv/bin/python analysis/theory_claude/dedup_defence/stage1_finite.py --workers 7
"""
import argparse, multiprocessing, pathlib, sys
from dataclasses import replace

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "phenomenological_fact_flow"))
from finite_model import FP, episode  # noqa: E402

EP = 400
GRID_T = tuple(float(t) for t in range(30, 131, 10))


def cell(args):
    read_mode, q, rho, t_end = args
    p = replace(FP(), read_mode=read_mode)
    rows = []
    for s in range(EP):
        for controlled in (True, False):
            rec = episode(p, q, rho, controlled, seed=10_000 * s + 7, t_end=t_end,
                          record=GRID_T if t_end > 30 else (30.0,))
            for t, m in rec.items():
                rows.append(dict(read_mode=read_mode, q=q, rho=rho, t_end=t_end, seed=s,
                                 arm="false" if controlled else "silent", t=t, **m))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=7)
    a = ap.parse_args()
    jobs = [(m, q, r, 30.0) for m in ("posts", "dedup") for q in (3, 6) for r in (0.25, 0.5, 0.75)]
    jobs += [(m, 6, r, 130.0) for m in ("posts", "dedup") for r in (0.25, 0.75)]
    with multiprocessing.Pool(a.workers) as pool:
        parts = pool.map(cell, jobs, chunksize=1)
    d = pd.DataFrame([r for part in parts for r in part])
    d.to_csv(HERE / "stage1_episodes.csv", index=False)
    key = ["read_mode", "q", "rho", "t_end", "t"]
    piv = d.pivot_table(index=key + ["seed"], columns="arm", values="x2").reset_index()
    piv["G"] = piv["false"] - piv["silent"]
    g = piv.groupby(key).agg(G=("G", "mean"), G_se=("G", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
                             silent_A2=("silent", "mean")).reset_index()
    div = d[d.arm == "silent"].groupby(key).distinct_A.mean().rename("native_circulating_silent").reset_index()
    divc = d[d.arm == "false"].groupby(key).distinct_A.mean().rename("native_circulating_controlled").reset_index()
    s = g.merge(div, on=key).merge(divc, on=key)
    s.to_csv(HERE / "stage1_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(s.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
