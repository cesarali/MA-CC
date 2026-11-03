"""Metastability and tipping in the 15-agent campaign (exploratory; 3 Nov 2026).

For every cell, read each episode's vote shares among voters at t = 30 and t = 130 and classify the episode:
  "A_k lock-in" if share_Ak / share_voted >= threshold (2/3 primary, 0.8 robustness), else "mixed".
Tipping probability: P(A2 lock-in | false arm) - P(A2 lock-in | paired silent arm), per (setup, q, rho, read_mode, t).
Also records P(A0 lock-in) in both arms (the "unintended diversion" toward A0), and a bimodality summary: the share of
episodes with A2 share <= 0.2 and >= 0.8.
Writes <out>/table_tipping.csv. Usage, from the repository root:
    .venv/bin/python analysis/15_agent_simulation/analysis/tipping.py --runs <run> <run> --out <folder> [--workers 12]
"""
from __future__ import annotations
import argparse, json, multiprocessing, pathlib

import numpy as np
import pandas as pd

TIMES = (30.0, 130.0)
THRESH = (2 / 3, 0.8)


def one(cell_dir):
    c = pathlib.Path(cell_dir)
    p = json.loads((c / "params.json").read_text())
    d = pd.read_parquet(c / "snapshots.parquet", columns=["episode", "t", "share_A0", "share_A1", "share_A2", "share_voted"])
    rows = []
    for t in TIMES:
        x = d[np.isclose(d.t, t)]
        v = x.share_voted.where(x.share_voted > 0)
        s = {k: (x[f"share_{k}"] / v).to_numpy() for k in ("A0", "A1", "A2")}
        row = dict(setup=p["setup"], arm=p["arm"], q=p["q"], rho=p["rho"], read_mode=p["read_mode"], t=t, n=len(x),
                   low_A2=float(np.nanmean(s["A2"] <= 0.2)), high_A2=float(np.nanmean(s["A2"] >= 0.8)))
        for th in THRESH:
            tag = "23" if th < 0.7 else "80"
            for k in ("A0", "A1", "A2"):
                row[f"lock_{k}_{tag}"] = float(np.nanmean(s[k] >= th - 1e-12))
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    cells = [str(c) for r in a.runs for c in sorted((pathlib.Path(r) / "cells").iterdir())]
    with multiprocessing.Pool(a.workers) as pool:
        parts = pool.map(one, cells, chunksize=4)
    d = pd.DataFrame([r for part in parts for r in part])
    key = ["setup", "q", "rho", "read_mode", "t"]
    f = d[d.arm == "false"].set_index(key)
    s = d[d.arm == "silent"].set_index(key)
    out = f.join(s, lsuffix="_false", rsuffix="_silent", how="inner").reset_index()
    for tag in ("23", "80"):
        out[f"tip_A2_{tag}"] = out[f"lock_A2_{tag}_false"] - out[f"lock_A2_{tag}_silent"]
        out[f"div_A0_{tag}"] = out[f"lock_A0_{tag}_false"] - out[f"lock_A0_{tag}_silent"]
    o = pathlib.Path(a.out)
    o.mkdir(parents=True, exist_ok=True)
    out.to_csv(o / "table_tipping.csv", index=False)
    print("wrote", o / "table_tipping.csv", len(out), "rows")


if __name__ == "__main__":
    main()
