"""Marginal susceptibility with paired standard errors: the slope of G between neighbouring strengths.

For each simulation, arm (target A0 or A2), setting (q, qc, rho, gate), pair of neighbouring strengths
(b_k, b_k+1 or rate_k, rate_k+1) and time t:
    slope = mean over episodes of [x_target(e; strength_k+1) - x_target(e; strength_k)] / (strength_k+1 - strength_k),
    slope_se = standard deviation of the same per-episode difference / sqrt(episodes) / (strength_k+1 - strength_k).
The silent arm cancels in the difference, so this equals [G(strength_k+1) - G(strength_k)] / step. Episode e uses the
same random draws at every strength (the streams are keyed by seed and episode), so the per-episode difference is
paired and its error is smaller than that of two independent means.

Vote shares as in build_tables.py (Simulation 2: among agents who have voted). Writes
results/gold_standard/<date>/slopes.parquet.

Usage, from the repository root:
    .venv/bin/python analysis/task003_llm_free/shared_analysis/gold_standard/build_slopes.py [--date 2026-10-15] [--workers 12]
"""
from __future__ import annotations
import argparse, json, multiprocessing, pathlib, sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from build_tables import RES, RUNS, load  # noqa: E402

XS = {"sim1": [1, 2, 3, 4, 5, 6, 7, 8, 9], "sim2": [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]}


def strength(sim, prm):
    return float(prm["b"] if sim == "sim1" else prm["lambda_c"])


def job_list():
    jobs = []
    for sim, cells in RUNS.items():
        groups = {}
        for f in sorted(p for p in cells.iterdir() if p.is_dir()):
            prm = json.loads((f / "params.json").read_text())
            if prm["arm"] == "silent":
                continue
            gate = "on" if prm.get("controller_gate", "votes") == "votes" else "off"
            key = (sim, prm["arm"], prm["q"], prm["qc"], float(prm["rho"]), gate)
            groups.setdefault(key, {})[strength(sim, prm)] = f
        for key, by_x in groups.items():
            xs = XS[key[0]]
            assert sorted(by_x) == [float(x) for x in xs], key
            jobs.append((key, [by_x[float(x)] for x in xs]))
    return jobs


def run(job):
    (sim, arm, q, qc, rho, gate), folders = job
    k = 0 if arm == "truth" else 2
    xs = [float(x) for x in XS[sim]]
    frames = [load(sim, f)[["episode", "t", f"x{k}"]].rename(columns={f"x{k}": "y"}) for f in folders]
    rows = []
    for i in range(len(xs) - 1):
        j = frames[i].merge(frames[i + 1], on=["episode", "t"], suffixes=("_lo", "_hi")).dropna()
        d = (j.y_hi - j.y_lo).groupby(j.t)
        step = xs[i + 1] - xs[i]
        m, sd, n = d.mean(), d.std(), d.count()
        for t in m.index:
            rows.append((sim, arm, q, qc, rho, gate, xs[i], xs[i + 1], float(t), m[t] / step, sd[t] / np.sqrt(n[t]) / step))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2026-10-15")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    with multiprocessing.Pool(a.workers) as pool:
        parts = pool.map(run, job_list(), chunksize=2)
    cols = ["sim", "arm", "q", "qc", "rho", "gate", "x_lo", "x_hi", "t", "slope", "slope_se"]
    out = RES / "gold_standard" / a.date / "slopes.parquet"
    pd.DataFrame([r for p in parts for r in p], columns=cols).to_parquet(out, index=False)
    print("->", out)


if __name__ == "__main__":
    main()
