"""Object Placements: pre-registered model predictions and their score against the simulation.

  predict --tag M4     writes object_placements/preregistered_<tag>.csv from the model run
                       results/key_phenomena/theory_runs/<tag>_op.parquet: G at t = 30 per (setup, rho, q) and
                       D(q1 -> q2) for 3->6, 6->9, 9->12. Commit this file BEFORE running `score`.
  data                 writes object_placements/op_data.csv from the simulation run *_sim2_op_main: G at t = 30 (paired
                       per episode, x_A2 among agents who have voted) and D for the three steps with paired standard
                       errors (episodes are paired across q: common random streams). 95% interval = +-1.96 se.
  score --tag M4       compares a committed prediction file with op_data.csv: resolved signs (data interval excludes
                       0; model |D| >= 0.01), split by data sign, misses, crossover brackets, mean |G error|.
Writes object_placements/scorecard_op_<tag>.csv.

Usage, from the repository root:
    .venv/bin/python analysis/task003_llm_free/simulation_2/key_phenomena_hypothesis/object_placements/op_predictions_and_score.py predict --tag M4
"""
from __future__ import annotations
import argparse, json, pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUNS = ROOT / "results" / "key_phenomena" / "theory_runs"
SIM = ROOT / "results" / "simulation_2"
STEPS = [(3, 6), (6, 9), (9, 12)]
TOL = 0.01


def predict(tag):
    d = pd.read_parquet(RUNS / f"{tag}_op.parquet")
    d = d[np.isclose(d.t, 30.0)]
    p = d.pivot_table(index=["setup", "rho", "q"], columns="arm", values="x2")
    g = (p["false"] - p["silent"]).rename("G").reset_index()
    rows = []
    for (s, r), x in g.groupby(["setup", "rho"]):
        x = x.set_index("q").G
        for q in (3, 6, 9, 12):
            rows.append(dict(setup=s, rho=r, quantity="G", step=f"q{q}", value=x[q]))
        for a, b in STEPS:
            rows.append(dict(setup=s, rho=r, quantity="D", step=f"{a}->{b}", value=x[b] - x[a]))
    out = pd.DataFrame(rows)
    out.round(5).to_csv(HERE / f"preregistered_{tag}.csv", index=False)
    print(len(out), "predictions ->", HERE / f"preregistered_{tag}.csv")


def data():
    run = sorted(SIM.glob("*_sim2_op_main"))[-1]
    cells = {}
    for c in (run / "cells").iterdir():
        p = json.loads((c / "params.json").read_text())
        cells[(p["setup"], float(p["rho"]), p["q"], p["arm"])] = c

    def g_ep(setup, rho, q):
        def x2(c):
            s = pd.read_parquet(c / "snapshots.parquet", columns=["episode", "grid_index", "share_A2", "share_voted"])
            s = s[s.grid_index == 120].set_index("episode").sort_index()
            return s.share_A2 / s.share_voted
        return x2(cells[(setup, rho, q, "false")]) - x2(cells[(setup, rho, q, "silent")])

    rows = []
    for setup in ("op_symmetric", "op_nosolution"):
        for rho in sorted({k[1] for k in cells}):
            ep = {q: g_ep(setup, rho, q) for q in (3, 6, 9, 12)}
            for q, v in ep.items():
                rows.append(dict(setup=setup, rho=rho, quantity="G", step=f"q{q}", value=v.mean(),
                                 se=v.std() / np.sqrt(len(v))))
            for a, b in STEPS:
                d = (ep[b] - ep[a]).dropna()
                rows.append(dict(setup=setup, rho=rho, quantity="D", step=f"{a}->{b}", value=d.mean(),
                                 se=d.std() / np.sqrt(len(d))))
    out = pd.DataFrame(rows)
    out["sign"] = np.where(out.value > 1.96 * out.se, "+", np.where(out.value < -1.96 * out.se, "-", "0"))
    out.round(5).to_csv(HERE / "op_data.csv", index=False)
    print("->", HERE / "op_data.csv", "from", run.name)


def bracket(seq):
    res = [(r, s) for r, s in seq if s != "0"]
    return "; ".join(f"{a:g}-{b:g} ({s}->{t})" for (a, s), (b, t) in zip(res[:-1], res[1:]) if s != t) or "none"


def score(tag):
    pr = pd.read_csv(HERE / f"preregistered_{tag}.csv")
    da = pd.read_csv(HERE / "op_data.csv", dtype={"sign": str})
    m = da.merge(pr, on=["setup", "rho", "quantity", "step"], suffixes=("", "_model"))
    D = m[m.quantity == "D"].copy()
    D["model_sign"] = np.where(D.value_model >= TOL, "+", np.where(D.value_model <= -TOL, "-", "0"))
    D["resolved"] = D.sign != "0"
    D["correct"] = D.resolved & (D.sign == D.model_sign)
    rows = []
    for s, d in D.groupby("setup"):
        r = d[d.resolved]
        G = m[(m.quantity == "G") & (m.setup == s)]
        rows.append(dict(setup=s, resolved=len(r), correct=int(r.correct.sum()),
                         pos=int((r.sign == "+").sum()), pos_correct=int((r.correct & (r.sign == "+")).sum()),
                         neg=int((r.sign == "-").sum()), neg_correct=int((r.correct & (r.sign == "-")).sum()),
                         G_mae=float((G.value_model - G.value).abs().mean()),
                         misses="; ".join(f"{x.rho:g} {x.step} data {x.sign} model {x.model_sign}"
                                          for x in r[~r.correct].itertuples())))
        for a, b in STEPS:
            x = d[d.step == f"{a}->{b}"].sort_values("rho")
            rows.append(dict(setup=s, step=f"{a}->{b}", data=bracket(list(zip(x.rho, x.sign))),
                             model=bracket(list(zip(x.rho, x.model_sign)))))
    out = pd.DataFrame(rows)
    out.to_csv(HERE / f"scorecard_op_{tag}.csv", index=False)
    D.round(4).to_csv(HERE / f"scorecard_op_{tag}_cells.csv", index=False)
    pd.set_option("display.width", 250, "display.max_colwidth", 120)
    print(out.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["predict", "data", "score"])
    ap.add_argument("--tag")
    a = ap.parse_args()
    {"predict": lambda: predict(a.tag), "data": data, "score": lambda: score(a.tag)}[a.what]()
