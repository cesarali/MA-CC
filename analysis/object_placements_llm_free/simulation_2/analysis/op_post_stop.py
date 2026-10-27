"""Object Placements, persistence: pre-registered model predictions of delta(t) after the controller stops, the
simulation's delta(t), and the score. Same measures as the task003 post-stop scorecard
(task003_llm_free/simulation_2/key_phenomena_hypothesis/held_out_validation/score_post_stop.py).

delta(t) = mean over paired episodes of [x_A2(false) - x_A2(silent)] at time t, x_A2 = share of A2 among agents who
have voted. Four continuations from t_c = 30: normal, no_comm, frozen_board, purge_controller_facts; both setups,
rho 0.25 and 0.75, q 6, to t = 130.

  predict --tag M4   writes preregistered_post_stop_<tag>.csv (delta(t) on the 0.25 grid) from the model run
                     analysis/theory_claude/results/theory_runs/<tag>_op_post_stop.parquet (run_models.py --set op_post_stop).
                     Commit this file BEFORE running the simulation `sim2_op_post_stop`.
  data               writes op_post_stop_data.csv (delta(t), its standard error, episodes) from the newest run
                     results/simulation_2/*_sim2_op_post_stop.
  score --tag M4     compares a committed prediction with the data: delta at t = 30, 31, 40, 80, 130; the time to half
                     the t = 30 value (censored at 130); the sign of the residual delta(130) after removal; and the
                     ordering of the four continuations at t = 40 and 130. Writes scorecard_op_post_stop_<tag>.csv and
                     scorecard_op_post_stop_<tag>_order.csv.

Usage, from the repository root:
    .venv/bin/python analysis/object_placements_llm_free/simulation_2/analysis/op_post_stop.py predict --tag M4
"""
from __future__ import annotations
import argparse, json, pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]                                            # analysis/object_placements_llm_free
RUNS = ROOT.parent / "theory_claude" / "results" / "theory_runs"
SIM = ROOT / "results" / "simulation_2"
TIMES = [30.0, 31.0, 40.0, 80.0, 130.0]
MODES = ["normal", "no_comm", "frozen_board", "purge_controller_facts"]
GRID = 0.25


def half_time(d):
    d = d.sort_values("t")
    d30 = d.loc[np.isclose(d.t, 30.0), "delta"].iloc[0]
    after = d[d.t > 30.0]
    below = after[after.delta <= d30 / 2]
    return float(below.t.iloc[0] - 30.0) if len(below) else np.nan


def predict(tag):
    m = pd.read_parquet(RUNS / f"{tag}_op_post_stop.parquet", columns=["setup", "rho", "mode", "arm", "t", "x2", "seed"])
    p = m.pivot_table(index=["setup", "rho", "mode", "t"], columns="arm", values="x2")
    out = (p["false"] - p["silent"]).rename("delta").reset_index()
    out["realisations"] = m.seed.nunique()
    out.round(5).to_csv(HERE / f"preregistered_post_stop_{tag}.csv", index=False)
    print(len(out), "rows ->", HERE / f"preregistered_post_stop_{tag}.csv")
    summary(out, "model")


def data():
    run = sorted(SIM.glob("*_sim2_op_post_stop"))[-1]
    cells = {}
    for c in (run / "cells").iterdir():
        p = json.loads((c / "params.json").read_text())
        cells[(p["setup"], float(p["rho"]), p["post_stop_mode"], p["arm"])] = c

    def x2(c):
        s = pd.read_parquet(c / "snapshots.parquet", columns=["episode", "grid_index", "share_A2", "share_voted"])
        s = s.assign(x2=s.share_A2 / s.share_voted.where(s.share_voted > 0))
        return s.pivot_table(index="episode", columns="grid_index", values="x2")

    rows = []
    for (setup, rho, mode, arm) in sorted(cells):
        if arm != "false":
            continue
        d = x2(cells[(setup, rho, mode, "false")]) - x2(cells[(setup, rho, mode, "silent")])
        for g in d.columns:
            v = d[g].dropna()
            rows.append(dict(setup=setup, rho=rho, mode=mode, t=round(g * GRID, 6), delta=v.mean(),
                             se=v.std() / np.sqrt(len(v)) if len(v) > 1 else np.nan, n=len(v)))
    out = pd.DataFrame(rows)
    out.round(6).to_csv(HERE / "op_post_stop_data.csv", index=False)
    print("->", HERE / "op_post_stop_data.csv", "from", run.name)
    summary(out, "data")


def summary(d, label):
    rows = []
    for (s, r, mode), g in d.groupby(["setup", "rho", "mode"]):
        row = dict(setup=s, rho=r, mode=mode, half=half_time(g))
        for t in TIMES:
            row[f"t{int(t)}"] = float(g.loc[np.isclose(g.t, t), "delta"].iloc[0])
        rows.append(row)
    pd.set_option("display.width", 250)
    print(label)
    print(pd.DataFrame(rows).round(3).to_string(index=False))


def score(tag):
    m = pd.read_csv(HERE / f"preregistered_post_stop_{tag}.csv")
    d = pd.read_csv(HERE / "op_post_stop_data.csv")
    rows = []
    for (s, r, mode), g in d.groupby(["setup", "rho", "mode"]):
        h = m[(m.setup == s) & np.isclose(m.rho, r) & (m["mode"] == mode)]
        row = dict(setup=s, rho=r, mode=mode, half_data=half_time(g), half_model=half_time(h))
        for t in TIMES:
            row[f"data_{int(t)}"] = float(g.loc[np.isclose(g.t, t), "delta"].iloc[0])
            row[f"se_{int(t)}"] = float(g.loc[np.isclose(g.t, t), "se"].iloc[0])
            row[f"model_{int(t)}"] = float(h.loc[np.isclose(h.t, t), "delta"].iloc[0])
        if mode == "purge_controller_facts":
            sd = "+" if row["data_130"] > 1.96 * row["se_130"] else ("-" if row["data_130"] < -1.96 * row["se_130"] else "0")
            sm = "+" if row["model_130"] >= 0.01 else ("-" if row["model_130"] <= -0.01 else "0")
            row.update(residual_sign_data=sd, residual_sign_model=sm)
        rows.append(row)
    out = pd.DataFrame(rows)
    order = []
    for (s, r), g in out.groupby(["setup", "rho"]):
        for t in (40, 130):
            od = list(g.sort_values(f"data_{t}")["mode"])
            om = list(g.sort_values(f"model_{t}")["mode"])
            order.append(dict(setup=s, rho=r, t=t, data_order=" < ".join(od), model_order=" < ".join(om), same=od == om))
    out.round(4).to_csv(HERE / f"scorecard_op_post_stop_{tag}.csv", index=False)
    pd.DataFrame(order).to_csv(HERE / f"scorecard_op_post_stop_{tag}_order.csv", index=False)
    pd.set_option("display.width", 250, "display.max_colwidth", 90)
    print(out.round(3).to_string(index=False))
    print(pd.DataFrame(order).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["predict", "data", "score"])
    ap.add_argument("--tag")
    a = ap.parse_args()
    {"predict": lambda: predict(a.tag), "data": data, "score": lambda: score(a.tag)}[a.what]()
