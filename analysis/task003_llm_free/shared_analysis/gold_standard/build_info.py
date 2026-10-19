"""Readability I(Z;X), cost C, undirected part R and efficiency eta for every setting and time, per setup.

Same definitions as the gold-standard report (Eqs. 3-7): Z = the controller's target (A0 or A2, fair coin); X = the
vote of a randomly chosen agent; P_s = mean over episodes of the vote shares in arm s (silent 0, target A0 +,
target A2 -); every distribution gets a pseudo-count of 0.5 episodes per value before any KL; bits;
I = 1/2 KL(P+||Q) + 1/2 KL(P-||Q), Q = (P+ + P-)/2; C = 1/2 KL(P+||P0) + 1/2 KL(P-||P0) = I + KL(Q||P0); eta = I/C,
undefined when C < 0.01 bits. Reads results/gold_standard/<date>/cells_time.parquet (build_tables.py) and writes
info.parquet next to it.

Usage, from the repository root:
    .venv/bin/python analysis/task003_llm_free/shared_analysis/gold_standard/build_info.py --date 2026-10-16
"""
from __future__ import annotations
import argparse, pathlib, sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from build_tables import RES  # noqa: E402

ALPHA, C_MIN = 0.5, 0.01
V = ["vote_0", "vote_1", "vote_2"]


def kl(p, q):
    nz = p > 0
    return (p[nz] * np.log2(p[nz] / q[nz])).sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2026-10-16")
    a = ap.parse_args()
    d = RES / "gold_standard" / a.date
    t = pd.read_parquet(d / "cells_time.parquet")
    if "case" not in t.columns:                     # tables built before the case column existed
        t = t.assign(case="standard")
    key = ["sim", "setup", "case", "q", "qc", "x", "rho", "gate", "t"]
    tr = t[t.arm == "truth"].set_index(key)
    fa = t[t.arm == "false"].set_index(key)
    si = t[t.arm == "silent"].set_index(["sim", "setup", "case", "q", "rho", "t"])

    def sm(row):
        p = row[V].to_numpy(float)
        p = p / p.sum()
        return (row.episodes * p + ALPHA) / (row.episodes + ALPHA * 3)

    rows = []
    for k in tr.index.intersection(fa.index):
        a_, b_ = tr.loc[k], fa.loc[k]
        s_ = si.loc[(k[0], k[1], k[2], k[3], k[6], k[8])]
        pp, pm, p0 = sm(a_), sm(b_), sm(s_)
        q = 0.5 * (pp + pm)
        I = 0.5 * kl(pp, q) + 0.5 * kl(pm, q)
        C = 0.5 * kl(pp, p0) + 0.5 * kl(pm, p0)
        rows.append(dict(zip(key, k), I=I, C=C, R=kl(q, p0), eta=I / C if C >= C_MIN else np.nan))
    out = pd.DataFrame(rows)
    out.to_parquet(d / "info.parquet", index=False)
    print(len(out), "rows ->", d / "info.parquet", "| max |C - I - R|:", float((out.C - out.I - out.R).abs().max()))


if __name__ == "__main__":
    main()
