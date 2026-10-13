"""Build the draft LaTeX report on the operational frontier (tables from the result files).
Usage, from the repository root (after frontier.py and frontier_phase.py):
    .venv/bin/python analysis/task003_llm_free/efficiency_estimates/reports/make_frontier_draft.py
"""
import pathlib
import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "2026-10-13_frontier_draft"
D = HERE.parents[1] / "results" / "efficiency_estimates" / "2026-10-13"
SN = {"sim1": "Sim.~1", "sim2": "Sim.~2"}


def f(v, nd=1):
    return "--" if not np.isfinite(v) else f"{v:.{nd}f}"


t = pd.read_csv(D / "frontier_thresholds.csv")
t = t[(t.rule == "plurality") & (t.q == 6) & (t["T"] == 30) & (t.outcome == "target")]
rows = []
for (sim, setup, rho, arm), g in t.groupby(["sim", "setup", "rho", "arm"]):
    g = g.set_index("threshold")
    if not np.isfinite(g.least_cost).any():
        continue
    cells = [f"{f(g.loc[d, 'least_cost'])} / {f(g.loc[d, 'least_cost_mix'])}" for d in (0.05, 0.1, 0.2, 0.3)]
    rows.append(f"{SN[sim]} & {setup[8:]} & {rho:g} & {'A0' if arm == 'truth' else 'A2'} & " + " & ".join(cells) + r" \\")
thr = "\n".join(rows)

ds = pd.read_csv(D / "frontier_downsample.csv")
ds = ds[ds.q == 6]
prows = []
for (sim, outcome), g in ds.groupby(["sim", "outcome"]):
    cells = []
    for m in (50, 100, 300):
        h = g[g.n_sub == m]
        cells += [f"{h.delta_rmse.median():.3f}", f"{h.eta_cost_rmse.median():.3f}"]
    prows.append(f"{SN[sim]} & {'target attainment' if outcome == 'target' else 'task correctness'} & " + " & ".join(cells) + r" \\")
prec = "\n".join(prows)

p = pd.read_csv(D / "frontier_points.csv")
x = p[(p.rule == "plurality") & (p["T"] == 30) & (p.q == 6) & (p.stop == "on") & (p.outcome == "target")]
srows = []
for (sim, setup, rho, arm), g in x.groupby(["sim", "setup", "rho", "arm"]):
    if g.delta.abs().max() < 0.02:
        continue
    e = g.eta_cost
    srows.append(f"{SN[sim]} & {setup[8:]} & {rho:g} & {'A0' if arm == 'truth' else 'A2'} & {g.p_silent.iloc[0]:.2f} & "
                 f"{g.delta.min():+.2f} to {g.delta.max():+.2f} & {f(g.room_share.median(), 2)} & {g.facts.min():.0f}--{g.facts.max():.0f} & "
                 f"{int((e > 0.999).sum())}/{len(g)} & {f(e.median(), 2)} & {f(e.min(), 2)} \\\\")
summ = "\n".join(srows)

tex = (HERE / "frontier_draft_template.tex").read_text()
tex = tex.replace("%THRESHOLDS%", thr).replace("%PRECISION%", prec).replace("%SUMMARY%", summ)
(OUT / "2026-10-13_frontier_draft.tex").write_text(tex)
print("written", OUT / "2026-10-13_frontier_draft.tex")
