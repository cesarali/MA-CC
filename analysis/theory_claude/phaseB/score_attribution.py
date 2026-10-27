"""Phase B: score M3s with the existing scoring code and compute the attribution f = (err_M3 - err_M3s) / (err_M3 - err_M4).

  task003 held-out      held_out_validation/score.py --tag M3s       (writes its scorecards there, like M1-M4)
  task003 post-stop     held_out_validation/score_post_stop.py --tag M3s
  Object Placements     reversal: the scoring of op_predictions_and_score.py; persistence: that of op_post_stop.py;
                        both applied to M3s model outputs, written here (not as "preregistered" files: M3s is post-hoc)
Writes attribution.csv and prints the comparison of M3, M3s and M4.

Usage, from the repository root:
    .venv/bin/python analysis/theory_claude/phaseB/score_attribution.py [TAG ...]   (default M3 M3s M4)
"""
from __future__ import annotations
import pathlib, subprocess, sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
A = HERE.parents[1]
REPO = A.parent
KP = A / "task003_llm_free/simulation_2/key_phenomena_hypothesis"
OPA = A / "object_placements_llm_free/simulation_2/analysis"
RUNS = A / "theory_claude/results/theory_runs"
PY = str(REPO / ".venv/bin/python")
sys.path.insert(0, str(OPA))
import op_post_stop as ops  # noqa: E402
from op_predictions_and_score import STEPS, TOL, bracket  # noqa: E402


def op_reversal_predictions(tag):
    d = pd.read_parquet(RUNS / f"{tag}_op.parquet", columns=["setup", "rho", "q", "arm", "t", "x2"])
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
    return pd.DataFrame(rows)


def op_reversal_score(pr):
    da = pd.read_csv(OPA / "op_data.csv", dtype={"sign": str})
    m = da.merge(pr, on=["setup", "rho", "quantity", "step"], suffixes=("", "_model"))
    D = m[m.quantity == "D"].copy()
    D["model_sign"] = np.where(D.value_model >= TOL, "+", np.where(D.value_model <= -TOL, "-", "0"))
    out = {}
    for s, d in D.groupby("setup"):
        r = d[d.sign != "0"]
        G = m[(m.quantity == "G") & (m.setup == s)]
        x = d[d.step == "3->6"].sort_values("rho")
        out[s] = dict(signs=f"{int((r.sign == r.model_sign).sum())}/{len(r)}", G_err=float((G.value_model - G.value).abs().mean()),
                      cross36=bracket(list(zip(x.rho, x.model_sign))))
    return out


def op_post_stop_score(tag):
    m = pd.read_parquet(RUNS / f"{tag}_op_post_stop.parquet", columns=["setup", "rho", "mode", "arm", "t", "x2"])
    p = m.pivot_table(index=["setup", "rho", "mode", "t"], columns="arm", values="x2")
    pred = (p["false"] - p["silent"]).rename("delta").reset_index()
    pred.to_csv(HERE / f"op_post_stop_{tag}_model.csv", index=False)
    d = pd.read_csv(OPA / "op_post_stop_data.csv")
    order_right, half_right, n_half = 0, 0, 0
    for (s, r), g in d.groupby(["setup", "rho"]):
        h = pred[(pred.setup == s) & np.isclose(pred.rho, r)]
        for t in (40.0, 130.0):
            od = list(g[np.isclose(g.t, t)].sort_values("delta")["mode"])
            om = list(h[np.isclose(h.t, t)].sort_values("delta")["mode"])
            order_right += od == om
        gd, hm = g[g["mode"] == "no_comm"], h[h["mode"] == "no_comm"]
        n_half += 1
        half_right += ops.half_time(gd) == ops.half_time(hm)
    return dict(order=f"{order_right}/8", no_comm_half_exact=f"{half_right}/{n_half}")


def task003_post_stop_order(tag):
    o = pd.read_csv(KP / "held_out_validation" / f"scorecard_{tag}_post_stop_order.csv")
    return f"{int(o.same.sum())}/{len(o)}"


def main():
    tags = sys.argv[1:] or ["M3", "M3s", "M4"]
    for tag in tags:
        if tag in ("M3s", "M4b"):           # models added after the original scorecards
            for script in ("score.py", "score_post_stop.py"):
                subprocess.run([PY, str(KP / "held_out_validation" / script), "--tag", tag], check=True, capture_output=True)
    rows = []
    for tag in tags:
        s = pd.read_csv(KP / "held_out_validation" / f"scorecard_{tag}_summary.csv")
        h = s[s.split == "held_out"].set_index("setup")
        cr = s[(s.split == "crossover") & (s.step == "3->6")].set_index("setup")
        opr = op_reversal_score(op_reversal_predictions(tag))
        ops_ = op_post_stop_score(tag)
        rows.append(dict(model=tag,
                         t3_signs_sym=f"{int(h.loc['symmetric','correct'])}/{int(h.loc['symmetric','resolved'])}",
                         t3_signs_nos=f"{int(h.loc['nosolution','correct'])}/{int(h.loc['nosolution','resolved'])}",
                         t3_G_err=float(h.G_mae.mean()), t3_cross36_sym=cr.loc["symmetric", "model"],
                         t3_cross36_nos=cr.loc["nosolution", "model"], t3_post_order=task003_post_stop_order(tag),
                         op_signs_sym=opr["op_symmetric"]["signs"], op_signs_nos=opr["op_nosolution"]["signs"],
                         op_G_err=float(np.mean([opr[k]["G_err"] for k in opr])),
                         op_cross36_sym=opr["op_symmetric"]["cross36"], op_post_order=ops_["order"],
                         op_no_comm_half=ops_["no_comm_half_exact"]))
    r = pd.DataFrame(rows).set_index("model")
    out = HERE / ("attribution.csv" if tags == ["M3", "M3s", "M4"] else "comparison_" + "_".join(tags) + ".csv")
    r.to_csv(out)
    if not {"M3", "M3s", "M4"} <= set(tags):
        pd.set_option("display.width", 250, "display.max_columns", 30)
        print(r.T.to_string())
        return
    f = {k: (r.loc["M3", k] - r.loc["M3s", k]) / (r.loc["M3", k] - r.loc["M4", k]) for k in ("t3_G_err", "op_G_err")}
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(r.T.to_string())
    print("f_controller (share of the M3 -> M4 G-error gain already achieved by M3s):", {k: round(v, 2) for k, v in f.items()})


if __name__ == "__main__":
    main()
