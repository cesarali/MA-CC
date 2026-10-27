"""Figures of the finite-model report. Inputs: phenomenological_fact_flow/findings/step3_*.csv and finite_size_numbers.csv;
simulator references (qualitative, read-only): task003 benchmarks G_D.parquet and the earlier replicated-swarm diagnostic
(theory_claude/M1_M2_mean_field.md section 3) and K data (key_phenomena_hypothesis/tables/scaling_K.csv).
Usage, from the repository root: .venv/bin/python analysis/theory_claude/reports/2026-10-28_finite_model/make_figures.py"""
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
A = HERE.parents[2]
F = A / "theory_claude/phenomenological_fact_flow/findings"
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "legend.fontsize": 7})
CF, CM, CS = "#d1495b", "#3b75af", "#222222"

r = pd.read_csv(F / "step3_reversal.csv")
b = pd.read_parquet(A / "task003_llm_free/results/key_phenomena/theory_benchmarks/G_D.parquet")
gs = b[(b.kind == "G") & (b.setup == "symmetric") & (b.q == 6)].sort_values("rho")
ds = b[(b.kind == "D") & (b.setup == "symmetric") & (b.step == "3->6")].sort_values("rho")
fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.6))
ax[0].errorbar(r.rho, r.G6, yerr=1.96 * r.G6_se, fmt="-o", ms=3, color=CF, capsize=1.5, label="finite model")
ax[0].plot(r.rho, r.G6_meanfield, "--", color=CM, label="mean-field model")
ax[0].plot(gs.rho, gs.G_vote, "s", ms=3.5, color=CS, mfc="white", label="simulator, task003 symmetric")
ax[1].errorbar(r.rho, r.D36, yerr=1.96 * r.D36_se, fmt="-o", ms=3, color=CF, capsize=1.5, label="finite model")
ax[1].plot(r.rho, r.D36_meanfield, "--", color=CM, label="mean-field model")
ax[1].plot(ds.rho, ds.D_vote, "s", ms=3.5, color=CS, mfc="white", label="simulator, task003 symmetric")
ax[1].axhline(0, color="0.6", lw=0.6)
ax[0].set_title("(a) controller effect G (q = 6)")
ax[1].set_title("(b) reading increment D(3→6)")
for a in ax:
    a.set_xlabel("memory ρ")
ax[0].legend(frameon=False, fontsize=6.5)
fig.tight_layout()
fig.savefig(OUT / "fig1_reversal.pdf")
plt.close(fig)

n = pd.read_csv(F / "finite_size_numbers.csv")
fig, ax = plt.subplots(figsize=(3.4, 2.6))
ax.plot(r.rho, r.distinct_A_false, "-o", ms=3, color=CF, label="finite model, controlled arm")
ax.plot(r.rho, r.distinct_A_silent, "-o", ms=3, color=CF, mfc="white", label="finite model, silent arm")
ax.axhline(16, color=CM, ls="--", label="mean-field model (all 16)")
sim = n[n.kind == "agent"]
ax.plot(sim.rho, sim.distinct_held, "s", ms=4, color=CS, mfc="white", label="simulator, 4 setups (controlled arm,\nheld by an agent)")
ax.set_xlabel("memory ρ")
ax.set_ylabel("distinct agent facts circulating at t = 30")
ax.set_title("diversity collapse")
ax.set_ylim(0, 17)
ax.legend(frameon=False, fontsize=6, loc="center left", bbox_to_anchor=(0.0, 0.62))
fig.tight_layout()
fig.savefig(OUT / "fig2_diversity.pdf")
plt.close(fig)

p = pd.read_csv(F / "step3_persistence.csv")
MC = {"normal": "#222222", "frozen": "#4292c6", "no_comm": "#e6a532", "purge": "#d1495b"}
ML = {"normal": "normal", "frozen": "frozen board", "no_comm": "no communication", "purge": "controller facts removed"}
fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.5), sharey=False)
for a, rho in zip(ax, (0.25, 0.75)):
    for m in MC:
        x = p[(p.rho == rho) & (p["mode"] == m)].sort_values("t")
        a.plot(x.t, x["mean"], color=MC[m], label=ML[m])
        a.fill_between(x.t, x["mean"] - 1.96 * x.se, x["mean"] + 1.96 * x.se, color=MC[m], alpha=0.15, lw=0)
    a.axhline(0, color="0.6", lw=0.6)
    a.set_title(f"finite model, ρ = {rho}")
    a.set_xlabel("time (controller stopped at 30)")
ax[0].set_ylabel("δ(t)")
h, l = ax[0].get_legend_handles_labels()
fig.tight_layout(rect=(0, 0, 1, 0.9))
fig.legend(h, l, ncol=4, frameon=False, loc="upper center")
fig.savefig(OUT / "fig3_persistence.pdf")
plt.close(fig)

s = pd.read_csv(F / "step3_scaling.csv").set_index("tag")
k = pd.read_csv(A / "task003_llm_free/simulation_2/key_phenomena_hypothesis/tables/scaling_K.csv")
ksim = k[(k.setup == "symmetric") & (k.rho == 0.5)].sort_values("K")
fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.5), sharey=True)
R = [1, 2, 4]
ax[0].errorbar(R, [s.loc[f"R{i}", "mean"] for i in R], yerr=[1.96 * s.loc[f"R{i}", "se"] for i in R], fmt="-o", color=CF, capsize=2, label="finite model")
ax[0].plot([1, 2, 4, 8], [0.322, 0.204, 0.144, 0.137], "s--", color=CS, mfc="white", label="simulator (earlier diagnostic)")
ax[0].axhline(s.G_meanfield.iloc[0], color=CM, ls="--", label="mean-field model")
ax[0].set_xscale("log", base=2)
ax[0].set_xticks([1, 2, 4, 8])
ax[0].set_xticklabels(["1", "2", "4", "8"])
ax[0].set_xlabel("R copies of the swarm (facts fixed)")
ax[0].set_ylabel("G (ρ = 0.5, q = 6)")
ax[0].set_title("(a) more copies per fact → mean field")
ax[0].legend(frameon=False, fontsize=6.5)
K = [1, 2, 4]
gk = [s.loc["R1", "mean"]] + [s.loc[f"K{i}", "mean"] for i in (2, 4)]
ek = [s.loc["R1", "se"]] + [s.loc[f"K{i}", "se"] for i in (2, 4)]
ax[1].errorbar(K, gk, yerr=[1.96 * e for e in ek], fmt="-o", color=CF, capsize=2, label="finite model")
ax[1].plot(ksim.K, ksim.G_q6, "s--", color=CS, mfc="white", label="simulator, K projects")
ax[1].axhline(s.G_meanfield.iloc[0], color=CM, ls="--", label="mean-field model")
ax[1].set_xscale("log", base=2)
ax[1].set_xticks([1, 2, 4])
ax[1].set_xticklabels(["1", "2", "4"])
ax[1].set_xlabel("K copies of everything (agents and facts)")
ax[1].set_title("(b) same copies per fact → stays finite")
fig.tight_layout()
fig.savefig(OUT / "fig4_scaling.pdf")
plt.close(fig)

m = pd.read_csv(F / "step3_mechanism_equal_evidence.csv")
fig, ax = plt.subplots(figsize=(3.4, 2.5))
x = np.arange(2)
for j, (fc, lab) in enumerate(((4, "controller in 4 facts"), (16, "controller in 16 facts"))):
    g = m[m.F_C == fc].sort_values("rho")
    ax.bar(x + (j - 0.5) * 0.38, g["mean"], 0.38, color=CF if fc == 4 else "#f4a3ad", label=f"finite, {lab}")
    ax.scatter(x + (j - 0.5) * 0.38, g.G_meanfield, color=CM, marker="_", s=200, zorder=3, label="mean field" if j == 0 else None)
ax.set_xticks(x)
ax.set_xticklabels(["ρ = 0.25", "ρ = 0.5"])
ax.set_ylabel("G (q = 6)")
ax.set_title("same controller evidence, spread or concentrated", fontsize=7.5)
ax.legend(frameon=False, fontsize=6)
fig.tight_layout()
fig.savefig(OUT / "fig5_concentration.pdf")
plt.close(fig)
print("->", OUT)
