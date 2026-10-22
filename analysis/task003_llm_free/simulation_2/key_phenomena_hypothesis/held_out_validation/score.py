"""Scorecard of a model version against the pre-registered targets (calibration/holdout_preregistration.md).

Input: a predictions file results/key_phenomena/theory_runs/<tag>_all.parquet (run_models.py --set all), with vote
shares among voters x2 per (setup, rho, q, arm, t). Model G = x2(false) - x2(silent) at t = 30; model
D(q1 -> q2) = G(q2) - G(q1). A model increment is called positive / negative when |D| >= TOL, else zero (TOL covers
the model's own sampling noise; 0.01 for the mean-field models, see theory/M1_M2_mean_field.md).

Per split (development / held_out) and world:
  - resolved data increments (95% interval excludes 0): how many the model gets right in sign, separately for positive
    and negative data increments; list of misses
  - crossover brackets per reading step: the rho interval in which the data increment changes sign versus the model's
  - mean absolute error of G at t = 30
Writes held_out_validation/scorecard_<tag>.csv (cell level) and scorecard_<tag>_summary.csv.

Usage, from the repository root:
    .venv/bin/python analysis/task003_llm_free/simulation_2/key_phenomena_hypothesis/held_out_validation/score.py --tag M2 [--tol 0.01]
"""
from __future__ import annotations
import argparse, pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUNS = ROOT / "results" / "key_phenomena" / "theory_runs"
BENCH = ROOT / "results" / "key_phenomena" / "theory_benchmarks" / "G_D.parquet"
STEPS = ["3->6", "6->9", "9->12"]


def model_G(tag):
    d = pd.read_parquet(RUNS / f"{tag}_all.parquet")
    d = d[np.isclose(d.t, 30.0)]
    p = d.pivot_table(index=["setup", "rho", "q"], columns="arm", values="x2")
    return (p["false"] - p["silent"]).rename("G_model").reset_index()


def crossover(seq):
    """seq: list of (rho, sign) sorted by rho; the bracket between the last resolved positive cell and the next
    resolved negative cell (the data's convention: amplifying at low rho, protective at high rho)."""
    out = []
    res = [(r, s) for r, s in seq if s != "0"]
    for (r1, s1), (r2, s2) in zip(res[:-1], res[1:]):
        if s1 != s2:
            out.append(f"{r1:g}-{r2:g} ({s1}->{s2})")
    return "; ".join(out) or "none"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--tol", type=float, default=0.01)
    a = ap.parse_args()
    b = pd.read_parquet(BENCH)
    G = b[b.kind == "G"][["setup", "rho", "q", "G_vote", "split"]].merge(model_G(a.tag), on=["setup", "rho", "q"])
    D = b[b.kind == "D"][["setup", "rho", "step", "D_vote", "D_vote_lo", "D_vote_hi", "D_vote_sign", "split"]].copy()
    gm = G.set_index(["setup", "rho", "q"]).G_model
    D["D_model"] = [gm[(s, r, int(st.split("->")[1]))] - gm[(s, r, int(st.split("->")[0]))]
                    for s, r, st in zip(D.setup, D.rho, D.step)]
    D["model_sign"] = np.where(D.D_model >= a.tol, "+", np.where(D.D_model <= -a.tol, "-", "0"))
    D["data_sign"] = D.D_vote_sign.astype(str)
    D["resolved"] = D.data_sign != "0"
    D["correct"] = D.resolved & (D.model_sign == D.data_sign)
    D.round(4).to_csv(HERE / f"scorecard_{a.tag}.csv", index=False)
    rows = []
    for (split, s), d in D.groupby(["split", "setup"]):
        r = d[d.resolved]
        g = G[(G.split == split) & (G.setup == s)]
        row = dict(split=split, setup=s, resolved=len(r), correct=int(r.correct.sum()),
                   pos_data=int((r.data_sign == "+").sum()), pos_correct=int((r.correct & (r.data_sign == "+")).sum()),
                   neg_data=int((r.data_sign == "-").sum()), neg_correct=int((r.correct & (r.data_sign == "-")).sum()),
                   G_mae=float((g.G_model - g.G_vote).abs().mean()) if len(g) else np.nan,
                   misses="; ".join(f"{x.rho:g} {x.step} data {x.data_sign} model {x.model_sign}"
                                    for x in r[~r.correct].itertuples()))
        rows.append(row)
    for s in ("symmetric", "nosolution"):
        for st in STEPS:
            d = D[(D.setup == s) & (D.step == st)].sort_values("rho")
            rows.append(dict(split="crossover", setup=s, step=st,
                             data=crossover(list(zip(d.rho, d.data_sign))),
                             model=crossover(list(zip(d.rho, d.model_sign)))))
    out = pd.DataFrame(rows)
    out.to_csv(HERE / f"scorecard_{a.tag}_summary.csv", index=False)
    pd.set_option("display.width", 250, "display.max_colwidth", 120)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
