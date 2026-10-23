"""Post-stop scorecard (held-out, pre-registered): a model's delta(t) after the controller stops, against the four
archived branches (normal, no_comm, frozen_board, purge_controller_facts), rho 0.25 and 0.75, both worlds, q = 6.

Model delta(t) = mean x2(false) - mean x2(silent) over the model's realisations, per (setup, rho, mode, t), from
analysis/theory/results/theory_runs/<tag>_post_stop.parquet. Data: theory_benchmarks/post_stop.parquet.
Reports delta at t = 30, 31, 40, 80, 130 (model and data), the time to half the t = 30 value (censored at 130),
and whether the ordering of the four branches at t = 40 and 130 matches.
Writes held_out_validation/scorecard_<tag>_post_stop.csv and prints a summary.

Usage, from the repository root:
    .venv/bin/python analysis/task003_llm_free/simulation_2/key_phenomena_hypothesis/held_out_validation/score_post_stop.py --tag M3
"""
from __future__ import annotations
import argparse, pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
KP = HERE.parents[2] / "results" / "key_phenomena"
RUNS = HERE.parents[3] / "theory" / "results" / "theory_runs"     # analysis/theory
TIMES = [30.0, 31.0, 40.0, 80.0, 130.0]


def half_time(d):
    d = d.sort_values("t")
    d30 = d.loc[np.isclose(d.t, 30.0), "delta"].iloc[0]
    after = d[d.t > 30.0]
    below = after[after.delta <= d30 / 2]
    return float(below.t.iloc[0] - 30.0) if len(below) else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    m = pd.read_parquet(RUNS / f"{a.tag}_post_stop.parquet")
    m = m.pivot_table(index=["setup", "rho", "mode", "t"], columns="arm", values="x2")
    m = (m["false"] - m["silent"]).rename("delta").reset_index()
    d = pd.read_parquet(KP / "theory_benchmarks" / "post_stop.parquet")[["setup", "rho", "mode", "t", "delta"]]
    rows = []
    for (s, r, mode), g in d.groupby(["setup", "rho", "mode"]):
        h = m[(m.setup == s) & (m.rho == r) & (m["mode"] == mode)]
        row = dict(setup=s, rho=r, mode=mode, half_data=half_time(g), half_model=half_time(h))
        for t in TIMES:
            row[f"data_{int(t)}"] = float(g.loc[np.isclose(g.t, t), "delta"].iloc[0])
            row[f"model_{int(t)}"] = float(h.loc[np.isclose(h.t, t), "delta"].iloc[0])
        rows.append(row)
    out = pd.DataFrame(rows)
    order = []
    for (s, r), g in out.groupby(["setup", "rho"]):
        for t in (40, 130):
            od = list(g.sort_values(f"data_{t}")["mode"])
            om = list(g.sort_values(f"model_{t}")["mode"])
            order.append(dict(setup=s, rho=r, t=t, data_order=" < ".join(od), model_order=" < ".join(om), same=od == om))
    out.round(4).to_csv(HERE / f"scorecard_{a.tag}_post_stop.csv", index=False)
    pd.DataFrame(order).to_csv(HERE / f"scorecard_{a.tag}_post_stop_order.csv", index=False)
    pd.set_option("display.width", 250, "display.max_colwidth", 90)
    print(out.round(3).to_string(index=False))
    print(pd.DataFrame(order).to_string(index=False))


if __name__ == "__main__":
    main()
