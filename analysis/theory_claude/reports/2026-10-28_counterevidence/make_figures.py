"""Figures of the counter-evidence report. Inputs: phenomenological_fact_flow/findings/step2_*.csv; reruns the model D
curves with the measured strengths.
Usage, from the repository root: .venv/bin/python analysis/theory_claude/reports/2026-10-28_counterevidence/make_figures.py"""
import pathlib, sys
from dataclasses import replace
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
FF = HERE.parents[1] / "phenomenological_fact_flow"
sys.path.insert(0, str(FF))
from fact_flow import P, G_at_T  # noqa: E402

OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "legend.fontsize": 7})
s = pd.read_csv(FF / "findings" / "step2_strengths.csv")
mid = {"0.25-0.375": 0.3125, "0.5-0.625": 0.5625, "0.625-0.875": 0.75, "0.75-1": 0.875}
col = {("task003", "symmetric"): "#3b75af", ("task003", "no-proof"): "#9ecae1",
       ("Object Placements", "symmetric"): "#d1495b", ("Object Placements", "no-proof"): "#f4a3ad"}
fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.6))
for _, r in s.iterrows():
    lo, hi = [float(v) for v in r.sim_cross_36.split("-")]
    ax[0].errorbar(r.beta_A, mid[r.sim_cross_36], yerr=[[mid[r.sim_cross_36] - lo], [hi - mid[r.sim_cross_36]]], fmt="o",
                   color=col[(r.task, r.setup)], capsize=3, ms=6, label=f"{r.task}, {r.setup}")
ax[0].set_xlabel("measured counter-evidence strength β_A")
ax[0].set_ylabel("simulator crossover ρ (3→6)")
ax[0].set_title("(a) simulator: stronger counter-evidence,\nlower crossover", fontsize=8)
ax[0].legend(frameon=False, fontsize=6)
rh = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
for _, r in s.iterrows():
    p = replace(P(), eps=1.0, h0=float(r.h0), beta_A=float(r.beta_A), beta_C=float(r.beta_C))
    D = [G_at_T(p, 6, x)[0] - G_at_T(p, 3, x)[0] for x in rh]
    ax[1].plot(rh, D, "-o", ms=2.3, color=col[(r.task, r.setup)], label=f"{r.task}, {r.setup}")
ax[1].axhline(0, color="0.6", lw=0.6)
ax[1].set_xlabel("memory ρ")
ax[1].set_ylabel("model D(3→6)")
ax[1].set_title("(b) model with the measured strengths\n(nothing fitted to D)", fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "fig1_counterevidence.pdf")
plt.close(fig)
print("->", OUT)
