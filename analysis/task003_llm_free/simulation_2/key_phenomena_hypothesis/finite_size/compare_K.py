"""Fact scaling: K = 1, 2, 4 projects (copies per fact constant) against the replicated-agent series.

For each K, setup, rho, q at t = 30 (q_c 12; rate and budget x K; vote shares among agents who have voted):
  G = paired mean of x_A2(false) - x_A2(silent), with standard error; silent A2 and A0 shares; messages per project;
  D(3 -> 6) paired per episode (common random streams across q, as at K = 1).
K = 1: the fine runs (2026-10-15_sim2_symmetric_fine, 2026-10-16_sim2_nosolution_fine; rate 1, budget 30, W 48).
K = 2, 4: the latest *_sim2_kprojects_K2 / _K4 runs.
Writes key_phenomena_hypothesis/tables/scaling_K.csv and prints it.

Usage, from the repository root:
    .venv/bin/python analysis/task003_llm_free/simulation_2/key_phenomena_hypothesis/finite_size/compare_K.py
"""
from __future__ import annotations
import json, pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RES = ROOT / "results" / "simulation_2"
OUT = HERE.parent / "tables"
T30 = 120


def x_at(cell):
    s = pd.read_parquet(cell / "snapshots.parquet", columns=["episode", "grid_index", "share_A0", "share_A2",
                                                               "share_voted", "controller_messages"])
    s = s[s.grid_index == T30].set_index("episode").sort_index()
    return pd.DataFrame({"x0": s.share_A0 / s.share_voted, "x2": s.share_A2 / s.share_voted,
                         "msgs": s.controller_messages})


def cells_of(run, K):
    out = {}
    for c in (run / "cells").iterdir():
        if not (c / "episodes.parquet").exists() or not (c / "params.json").exists():
            continue                                    # cell still running
        p = json.loads((c / "params.json").read_text())
        if p["arm"] == "false" and (p.get("qc") != 12 or p.get("controller_gate") != "votes"
                                    or float(p.get("lambda_c", 0)) != float(K)):
            continue
        base = p["setup"].replace("task003_", "").split("_K")[0]
        out[(base, float(p["rho"]), p["q"], p["arm"])] = c
    return out


def main():
    runs = {1: [RES / "2026-10-15_sim2_symmetric_fine", RES / "2026-10-16_sim2_nosolution_fine"]}
    for K in (2, 4):
        found = sorted(RES.glob(f"*_sim2_kprojects_K{K}"))
        if found:
            runs[K] = [found[-1]]
    rows = []
    for K, rs in runs.items():
        cs = {}
        for r in rs:
            cs.update(cells_of(r, K))
        for (setup, rho, q, arm), cell in cs.items():
            if arm != "false" or q not in (3, 6) or rho not in (0.25, 0.5, 0.75, 1.0):
                continue
            if (setup, rho, q, "silent") not in cs:
                continue
            f, s = x_at(cell), x_at(cs[(setup, rho, q, "silent")])
            g = (f.x2 - s.x2).dropna()
            rows.append(dict(K=K, setup=setup, rho=rho, q=q, G=g.mean(), G_se=g.std() / np.sqrt(len(g)),
                             silent_x2=s.x2.mean(), silent_x0=s.x0.mean(), msgs_per_project=f.msgs.mean() / K,
                             episodes=len(g), _g=g))
    d = pd.DataFrame(rows)
    out = []
    for (K, setup, rho), grp in d.groupby(["K", "setup", "rho"]):
        g = grp.set_index("q")
        if not {3, 6} <= set(g.index):
            continue
        rec = dict(K=K, setup=setup, rho=rho)
        for q in (3, 6):
            for c in ("G", "G_se", "silent_x2", "silent_x0", "msgs_per_project"):
                rec[f"{c}_q{q}"] = g.loc[q, c]
        dd = (g.loc[6, "_g"] - g.loc[3, "_g"]).dropna()
        rec["D_3_6"], rec["D_3_6_se"] = dd.mean(), dd.std() / np.sqrt(len(dd))
        rec["episodes"] = int(g.loc[3, "episodes"])
        out.append(rec)
    o = pd.DataFrame(out).sort_values(["setup", "rho", "K"])
    o.round(4).to_csv(OUT / "scaling_K.csv", index=False)
    pd.set_option("display.width", 250)
    print(o[["setup", "rho", "K", "G_q3", "G_q6", "G_se_q6", "D_3_6", "D_3_6_se", "silent_x2_q6", "silent_x0_q6",
             "msgs_per_project_q6", "episodes"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
