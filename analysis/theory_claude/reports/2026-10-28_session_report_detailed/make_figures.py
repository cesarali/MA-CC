"""Figures for session_report_detailed.tex (28 Oct 2026). Engine-free model runs plus stored CSVs; no simulator runs.
Usage, from the repository root:
    .venv/bin/python analysis/theory_claude/reports/2026-10-28_session_report_detailed/make_figures.py"""
import pathlib, shutil, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

HERE = pathlib.Path(__file__).resolve().parent
FC = HERE.parents[1] / "fading_chain"
sys.path.insert(0, str(FC))
import selection_model as sm  # noqa: E402

OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
for f in ("fig3_dose_sensitivity.pdf", "fig4_threshold.pdf"):     # from the previous session report
    shutil.copy(HERE.parent / "2026-10-28_session_report" / "figures" / f, OUT / f)
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
                     "legend.fontsize": 7})

# fig: board kernels (sliding window against exponential fading), same mean turnover N/W
fig, ax = plt.subplots(figsize=(3.4, 2.3))
tau = 48 / 24
a = np.linspace(0, 4 * tau, 400)
ax.plot(a, np.where(a <= tau, 1 / tau, 0), color="#3b75af", label="sliding window of W posts (simulator)")
ax.plot(a, np.exp(-a / tau) / tau, color="#d1495b", ls="--", label="exponential fading (Eq. 2)")
ax.set_xlabel("age of a post (time units)")
ax.set_ylabel("weight in the board")
ax.legend(frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig(OUT / "fig_kernels.pdf")
plt.close(fig)

# fig: selection model mean coherence, and the data's board coherence
fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.6))
p = dict(sm.P0)
ts = np.linspace(30, 130, 201)
for name, c, col in (("coherent", sm.SETS["coherent"], "#3b75af"), ("ambivalent", sm.SETS["ambivalent"], "#d1495b")):
    for eps, ls in ((1.0, ":"), (0.75, "--"), (0.5, "-")):
        y0 = np.concatenate([[1 / 16], np.zeros(4), [0], np.zeros(4)])
        g = -np.log(0.25)
        s1 = solve_ivp(sm.rhs, (0, 30), y0, args=(c, p, eps, 6, g, True), rtol=1e-9, atol=1e-12, max_step=0.1)
        s2 = solve_ivp(sm.rhs, (30, 130), s1.y[:, -1], args=(c, p, eps, 6, g, True), t_eval=ts, rtol=1e-9, atol=1e-12,
                       max_step=0.1)
        cb = [(lambda bf: (bf * c).sum() / bf.sum())(s2.y[6:10, k]) for k in range(len(ts))]
        ax[0].plot(ts, cb, color=col, ls=ls, label=f"{name}, ε = {eps:g}")
ax[0].set_xlabel("time (controller stopped at 30)")
ax[0].set_ylabel("mean coherence on the board, c̄")
ax[0].set_title("(a) selection model, ρ = 0.25", fontsize=8)
ax[0].legend(frameon=False, fontsize=6, ncol=2, loc="center right")
d = pd.read_csv(FC / "board_coherence.csv")
d = d[~((d.task == "Object Placements") & (d.setup == "symmetric") & (d.rho == 0.75))]
lab = [f"{'t003' if r.task == 'task003' else 'OP'} {'no-pr' if r.setup == 'nosolution' else 'sym'} {r.rho:g}"
       for r in d[d.t == 30].itertuples()]
x = np.arange(len(lab))
for k, (t, col) in enumerate(((30.0, "0.6"), (130.0, "#d1495b"))):
    e = d[d.t == t]
    ax[1].bar(x + (k - 0.5) * 0.38, e.cbar, 0.38, yerr=1.96 * e.cbar_se, color=col, label=f"t = {t:g}", capsize=1.5)
ax[1].set_xticks(x)
ax[1].set_xticklabels(lab, rotation=30, ha="right")
ax[1].set_ylim(0.3, 1.0)
ax[1].set_ylabel("world coherence of facts on the board")
ax[1].set_title("(b) simulator, controlled arm (post-hoc)", fontsize=8)
ax[1].legend(frameon=False)
fig.tight_layout()
fig.savefig(OUT / "fig_selection.pdf")
plt.close(fig)
print("->", OUT)
