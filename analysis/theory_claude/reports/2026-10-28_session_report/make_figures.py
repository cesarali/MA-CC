"""Figures of the pooled session report (session_report.tex). Copies figures already produced during the session and
draws three new ones from stored results (no new model runs).
Usage, from the repository root: .venv/bin/python analysis/theory_claude/reports/2026-10-28_session_report/make_figures.py"""
import pathlib, shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
T = HERE.parents[1]
FF = T / "phenomenological_fact_flow" / "findings"
R = T / "reports"
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
for src, dst in ((FF / "fig1_reversal.pdf", "fig2_mf_reversal.pdf"), (FF / "fig5_dose_sensitivity.pdf", "fig3_dose_sensitivity.pdf"),
                 (FF / "fig4_threshold.pdf", "fig4_threshold.pdf"), (R / "2026-10-28_finite_model/figures/fig1_reversal.pdf", "fig5_finite_reversal.pdf"),
                 (R / "2026-10-28_finite_model/figures/fig4_scaling.pdf", "fig7_scaling.pdf"),
                 (R / "2026-10-28_counterevidence/figures/fig1_counterevidence.pdf", "fig8_counterevidence.pdf")):
    shutil.copy(src, OUT / dst)
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "legend.fontsize": 7})

# fig6: surviving facts, finite model with the two analytic approximations
b = pd.read_csv(FF / "step3_branching.csv")
fig, ax = plt.subplots(figsize=(3.4, 2.6))
ax.plot(b.rho, b.Feff_finite_silent, "-o", ms=3, color="#d1495b", label="finite model (silent arm)")
ax.plot(b.rho, b.Feff_finite_controlled, "-o", ms=3, color="#d1495b", mfc="white", label="finite model (controlled arm)")
ax.plot(b.rho, b.Feff_branching, "--", color="#3b75af", label="independent lineages")
ax.plot(b.rho, b.Feff_drift, ":", color="#3b75af", lw=1.6, label="neutral drift, conserved total")
ax.axhline(16, color="0.6", lw=0.8, label="mean field (all 16)")
ax.set_xlabel("memory ρ")
ax.set_ylabel("agent facts circulating at t = 30")
ax.set_ylim(0, 17)
ax.legend(frameon=False, fontsize=6, loc="upper left")
fig.tight_layout()
fig.savefig(OUT / "fig6_diversity.pdf")
plt.close(fig)

# fig9: vote gating in the finite model
g = pd.read_csv(FF / "step2_gating_finite.csv")
col = {1.0: "#222222", 0.75: "#e6a532", 0.5: "#d1495b"}
fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.4))
for a, rho in zip(ax, (0.25, 0.75)):
    for eps, c in col.items():
        x = g[(g.eps == eps) & (g.rho == rho)].sort_values("t")
        a.errorbar(x.t, x["mean"], yerr=1.96 * x.se, fmt="-o", ms=3, color=c, capsize=1.5, label=f"ε = {eps:g}")
    a.set_title(f"finite model, normal continuation, ρ = {rho}")
    a.set_xlabel("time (controller stopped at 30)")
ax[0].set_ylabel("δ(t)")
ax[0].legend(frameon=False)
fig.tight_layout()
fig.savefig(OUT / "fig9_gating.pdf")
plt.close(fig)

# fig10: susceptibility probe
s = pd.read_csv(T / "mixture_probe" / "probe_summary.csv")
s = s[s.t == 130.0].copy()
ret = {("Object Placements", "nosolution", 0.25): 0.370, ("Object Placements", "nosolution", 0.75): 0.333,
       ("Object Placements", "symmetric", 0.25): 0.670, ("task003", "nosolution", 0.25): 0.970,
       ("task003", "nosolution", 0.75): 1.049, ("task003", "symmetric", 0.25): 0.966, ("task003", "symmetric", 0.75): 0.751}
s["retained"] = [ret.get((a, b_, c), np.nan) for a, b_, c in zip(s.task, s.setup, s.rho)]
s["relS"] = s.S_t / s.S_30
fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.7))
for _, r in s.dropna(subset=["retained"]).iterrows():
    c = "#d1495b" if r.task == "Object Placements" else "#3b75af"
    m = "o" if r.setup == "symmetric" else "s"
    ax[0].scatter(r.relS, r.retained, color=c, marker=m, s=30)
    lab = f"{'OP' if r.task != 'task003' else 't003'} {'sym' if r.setup == 'symmetric' else 'no-pr'} {r.rho:g}"
    off = {"t003 no-pr 0.25": (-48, -9), "t003 sym 0.25": (-44, 4)}.get(lab, (3, 2))
    ax[0].annotate(lab, (r.relS, r.retained), fontsize=5.5, xytext=off, textcoords="offset points")
ax[0].plot([0, 1.1], [0, 1.1], color="0.7", lw=0.6, ls=":")
ax[0].set_xlabel("S(130)/S(30): susceptibility to the encountered mixture")
ax[0].set_ylabel("retained δ(130)/δ(30), simulation")
ax[0].set_title("(a) susceptibility tracks persistence", fontsize=8)
sub = s[(s.task == "Object Placements") & (s.setup == "nosolution")]
x = np.arange(len(sub))
ax[1].bar(x - 0.3, sub.dS, 0.2, color="#555555", label="total change ΔS")
ax[1].bar(x - 0.1, sub.mix_term, 0.2, color="#d1495b", label="mixture (order 1)")
ax[1].bar(x + 0.1, sub.mix_term_alt, 0.2, color="#f4a3ad", label="mixture (order 2)")
ax[1].bar(x + 0.3, sub.ctx_term_alt, 0.2, color="#3b75af", label="context (order 2)")
ax[1].set_xticks(x)
ax[1].set_xticklabels([f"OP no-proof ρ {r:g}" for r in sub.rho])
ax[1].axhline(0, color="0.6", lw=0.6)
ax[1].set_title("(b) the split depends on the order", fontsize=8)
h, l = ax[1].get_legend_handles_labels()
fig.tight_layout(rect=(0, 0.1, 1, 1))
fig.legend(h, l, frameon=False, fontsize=6.5, ncol=4, loc="lower right", bbox_to_anchor=(0.99, 0.0))
fig.savefig(OUT / "fig10_susceptibility.pdf")
plt.close(fig)
print("->", OUT)
