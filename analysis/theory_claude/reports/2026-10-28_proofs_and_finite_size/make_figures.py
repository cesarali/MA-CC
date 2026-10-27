"""Figures of the 28 Oct report: proofs with co-retention (model against simulator) and effective finite sizes.
Inputs: phenomenological_fact_flow/findings/{step1_curves,step1_simulator_check,finite_size_numbers}.csv.
Usage, from the repository root: .venv/bin/python analysis/theory_claude/reports/2026-10-28_proofs_and_finite_size/make_figures.py"""
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
F = HERE.parents[1] / "phenomenological_fact_flow" / "findings"
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "legend.fontsize": 7})

c = pd.read_csv(F / "step1_curves.csv")
s = pd.read_csv(F / "step1_simulator_check.csv")
s = s[s.arm == "silent"]
fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.5))
sty = {"task003-like, independent": ("#3b75af", ":"), "task003-like, co-retention": ("#3b75af", "-"),
       "OP-like, independent": ("#d1495b", ":"), "OP-like, co-retention": ("#d1495b", "-"), "no proof": ("#555555", "-")}
for lab, g in c.groupby("variant", sort=False):
    col, ls = sty[lab]
    ax[0].plot(g.rho, g.D36, ls=ls, color=col, label=lab)
    if lab != "no proof":
        ax[1].plot(g.rho, g.z, ls=ls, color=col, label=f"model, {lab}")
for task, col in (("task003", "#3b75af"), ("Object Placements", "#d1495b")):
    x = s[s.task == task]
    ax[1].plot(x.rho, x.proof_share, "o", ms=5, color=col, mfc="white", label=f"simulator, {task}")
ax[0].axhline(0, color="0.6", lw=0.6)
ax[0].set_title("(a) model: reading increment D(3→6)")
ax[1].set_title("(b) share of agents holding a complete proof")
for a in ax:
    a.set_xlabel("memory ρ")
ax[0].set_ylabel("D")
ax[1].set_ylabel("share (silent arm, q = 6, t = 30)")
ax[0].legend(frameon=False, fontsize=6)
ax[1].legend(frameon=False, fontsize=6)
ax[1].set_xlim(0.2, 0.8)
fig.tight_layout()
fig.savefig(OUT / "fig1_proofs.pdf")
plt.close(fig)

n = pd.read_csv(F / "finite_size_numbers.csv")
fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.8))
lab = n.task.str.replace("Object Placements", "OP") + "\n" + n.setup.str.replace("nosolution", "no-proof").str.replace("symmetric", "sym.") + "\nρ " + n.rho.astype(str)
n = n.assign(cell=lab)
cells = list(dict.fromkeys(n.cell))
import numpy as np
x = np.arange(len(cells))
for k, (kind, col, nm) in enumerate((("agent", "#3b75af", "agent facts"), ("ctrl", "#d1495b", "controller facts"))):
    g = n[n.kind == kind].set_index("cell").reindex(cells)
    ax[0].bar(x + (k - 0.5) * 0.38, g.distinct_on_board, 0.38, color=col, label=f"{nm} (of {'10–12' if kind == 'ctrl' else '15–18'})")
    ax[1].bar(x + (k - 0.5) * 0.38, g.copies_per_board_fact, 0.38, color=col, label=nm)
ax[0].set_title("(a) distinct facts still on the board")
ax[1].set_title("(b) board copies per surviving fact")
for a in ax:
    a.set_xticks(x)
    a.set_xticklabels(cells, fontsize=6)
ax[0].legend(frameon=False)
fig.tight_layout()
fig.savefig(OUT / "fig2_finite_sizes.pdf")
plt.close(fig)
print("->", OUT)
