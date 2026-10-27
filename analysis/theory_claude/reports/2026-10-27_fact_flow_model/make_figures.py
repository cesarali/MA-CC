"""Figures of the fact-flow report (fact_flow_report.tex). Regenerates the model's figures with
phenomenological_fact_flow/stageE_figures.py, copies them here, and draws the Stage C (proof) figure from
findings/stageC_curves.csv. Run stageC_proof.py and rev3_grid.py first.

Usage, from the repository root:
    .venv/bin/python analysis/theory_claude/reports/2026-10-27_fact_flow_model/make_figures.py
"""
import pathlib, shutil, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
FF = HERE.parents[1] / "phenomenological_fact_flow"
sys.path.insert(0, str(FF))
import stageE_figures as sf  # noqa: E402

OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
sf.fig1(); sf.fig2(); sf.fig3(); sf.fig4(); sf.fig5()
for src, dst in (("fig1_reversal", "fig2_reversal"), ("fig5_dose_sensitivity", "fig3_dose_sensitivity"),
                 ("fig4_threshold", "fig4_threshold"), ("fig2_continuations", "fig5_continuations"),
                 ("fig3_trajectories", "fig6_trajectories")):
    shutil.copy(FF / "findings" / f"{src}.pdf", OUT / f"{dst}.pdf")

c = pd.read_csv(FF / "findings" / "stageC_curves.csv")
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
                     "legend.fontsize": 7})
col = {"no proof (z = 0)": "#555555", "task003-like proof (l = 6, P = 12)": "#3b75af", "OP-like proof (l = 3, P = 2)": "#d1495b"}
fig, ax = plt.subplots(1, 3, figsize=(6.8, 2.3))
for lab, g in c.groupby("variant", sort=False):
    ax[0].plot(g.rho, g.D36, "-o", ms=2.3, color=col[lab], label=lab)
    ax[1].plot(g.rho, g.G6, "-", color=col[lab])
    ax[2].plot(g.rho, g.z_silent_q6, "-", color=col[lab])
for a, t, y in ((ax[0], "(a) reading increment D(3→6)", "D"), (ax[1], "(b) controller effect G (q = 6)", "G"),
                (ax[2], "(c) share holding a proof", "z")):
    a.set_title(t, fontsize=8)
    a.set_xlabel("memory ρ")
    a.set_ylabel(y)
ax[0].axhline(0, color="0.6", lw=0.6)
h, l = ax[0].get_legend_handles_labels()
fig.tight_layout(rect=(0, 0.1, 1, 1))
fig.legend(h, l, frameon=False, ncol=3, loc="lower center", fontsize=7)
fig.savefig(OUT / "fig7_proofs.pdf")
plt.close(fig)
print("figures ->", OUT)
