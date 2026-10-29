"""Aggregate a 15-agent campaign run into compact per-cell time series (pre-registered measures,
../00_preregistration.md).

For every cell and grid time t:
  vote shares among voters, share_Ak / share_voted (mean over episodes; episodes with no voter are skipped)
For every controlled (false) cell, paired with its silent cell episode by episode:
  g_e(t) = share_A2/share_voted (false) - share_A2/share_voted (silent);  G(t) = mean_e g_e(t), se = sd/sqrt(n)
Times are thinned to every 1.0 time unit (and t = 30, 130 exactly). Writes <run>/aggregates/cells.parquet and
<run>/aggregates/gain.parquet (gitignored with the run).

Usage, from the repository root:
    .venv/bin/python analysis/15_agent_simulation/analysis/aggregate.py <run folder> [--workers 12]
"""
from __future__ import annotations
import argparse, json, multiprocessing, pathlib

import numpy as np
import pandas as pd

COLS = ["episode", "grid_index", "t", "share_A0", "share_A1", "share_A2", "share_voted", "mean_p_A0", "mean_p_A2",
        "mean_active_facts"]
STEP = 1.0


def _snap(cell: pathlib.Path) -> pd.DataFrame:
    d = pd.read_parquet(cell / "snapshots.parquet", columns=COLS)
    d = d[np.isclose(d.t / STEP, np.round(d.t / STEP))].copy()
    v = d.share_voted.where(d.share_voted > 0)
    for k in ("A0", "A1", "A2"):
        d[f"vote_{k}"] = d[f"share_{k}"] / v
    return d


def _one(args):
    cell_dir, silent_dir = args
    cell = pathlib.Path(cell_dir)
    prm = json.loads((cell / "params.json").read_text())
    d = _snap(cell)
    agg = d.groupby("t").agg(vote_A0=("vote_A0", "mean"), vote_A1=("vote_A1", "mean"), vote_A2=("vote_A2", "mean"),
                             share_voted=("share_voted", "mean"), mean_p_A0=("mean_p_A0", "mean"),
                             mean_p_A2=("mean_p_A2", "mean"), active=("mean_active_facts", "mean")).reset_index()
    meta = {k: prm[k] for k in ("setup", "arm", "q", "rho", "read_mode")}
    agg = agg.assign(cell=cell.name, **meta)
    gain = None
    if prm["arm"] != "silent":
        s = _snap(pathlib.Path(silent_dir))[["episode", "t", "vote_A2", "vote_A0"]]
        m = d[["episode", "t", "vote_A2", "vote_A0"]].merge(s, on=["episode", "t"], suffixes=("_c", "_s")).dropna()
        m["g"] = m.vote_A2_c - m.vote_A2_s
        m["g0"] = m.vote_A0_c - m.vote_A0_s
        gain = m.groupby("t").agg(G=("g", "mean"), G_sd=("g", "std"), n=("g", "size"),
                                  G_A0=("g0", "mean")).reset_index()
        gain["G_se"] = gain.G_sd / np.sqrt(gain.n)
        gain = gain.assign(cell=cell.name, silent_cell=pathlib.Path(silent_dir).name, **meta)
    return agg, gain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    run = pathlib.Path(a.run)
    cells = sorted((run / "cells").iterdir())
    jobs = []
    for c in cells:
        prm = json.loads((c / "params.json").read_text())
        jobs.append((str(c), str(run / "cells" / prm["silent_cell"]) if prm["arm"] != "silent" else ""))
    with multiprocessing.Pool(a.workers) as pool:
        parts = pool.map(_one, jobs, chunksize=4)
    out = run / "aggregates"
    out.mkdir(exist_ok=True)
    pd.concat([p[0] for p in parts]).to_parquet(out / "cells.parquet", index=False)
    pd.concat([p[1] for p in parts if p[1] is not None]).to_parquet(out / "gain.parquet", index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
