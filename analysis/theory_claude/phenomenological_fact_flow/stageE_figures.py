"""Stage E figures of the fact-flow model (revision 3 = plan closure with softened vote gating eps). Two settings, both
P0 otherwise: eps = 1.0 ("high eligibility") and eps = 0.75 ("lower eligibility").
  fig1  D(rho; q1 -> q2) for 3->6, 6->9, 9->12, both settings
  fig2  delta(t) for the four continuations, rho 0.25 and 0.75, both settings
  fig3  m_A, m_C, b_A, b_C trajectories (controlled arm, normal continuation), rho 0.25, both settings
  fig4  retained fraction delta(130)/delta(30) against R_eff (all revision-3 grid points), with the threshold R = 1
  fig5  dose-against-sensitivity split of D(3->6) at eps = 1: full D, its small-effect approximation and the two terms
Writes findings/fig*.pdf.
"""
import pathlib
from dataclasses import replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fact_flow import P, G_at_T, run, phi2, sigma
from stageD_persistence import delta_path

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "findings"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
                     "legend.fontsize": 7})
SETS = {"high eligibility (ε = 1)": replace(P(), eps=1.0), "lower eligibility (ε = 0.75)": replace(P(), eps=0.75)}
FIG1_Y = "D(q₁→q₂) (reading increment)"
RH = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
MC = {"normal": "#222222", "frozen": "#4292c6", "no_comm": "#e6a532", "purge": "#d1495b"}
ML = {"normal": "normal", "frozen": "frozen board", "no_comm": "no communication", "purge": "controller facts removed"}


def fig1():
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.4), sharey=False)
    for a, (lab, p) in zip(ax, SETS.items()):
        G = {q: np.array([G_at_T(p, q, r)[0] for r in RH]) for q in (3, 6, 9, 12)}
        for (q1, q2), c in zip(((3, 6), (6, 9), (9, 12)), ("#9ecae1", "#2171b5", "#08306b")):
            a.plot(RH, G[q2] - G[q1], "o-", ms=2.5, color=c, label=f"{q1}→{q2}")
        a.axhline(0, color="0.6", lw=0.6)
        a.set_title(f"{lab}")
        a.set_xlabel("memory ρ")
    ax[0].set_ylabel(FIG1_Y)
    ax[0].legend(frameon=False, title="reading step")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_reversal.pdf")
    plt.close(fig)


def fig2():
    fig, ax = plt.subplots(2, 2, figsize=(6.8, 4.0), sharex=True)
    for j, (lab, p) in enumerate(SETS.items()):
        for i, rho in enumerate((0.25, 0.75)):
            a = ax[i, j]
            for mode in MC:
                t, d, _ = delta_path(p, 6, rho, mode)
                a.plot(t, d, color=MC[mode], label=ML[mode])
            a.axvline(30, color="0.5", ls=":", lw=0.6)
            a.axhline(0, color="0.6", lw=0.6)
            a.set_xlim(20, 130)
            a.set_title(f"{lab}, ρ = {rho}")
            if j == 0:
                a.set_ylabel("δ(t)")
            if i == 1:
                a.set_xlabel("time (controller stops at 30)")
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.legend(h, l, ncol=4, frameon=False, loc="upper center")
    fig.savefig(OUT / "fig2_continuations.pdf")
    plt.close(fig)


def fig3():
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.4), sharey=True)
    for a, (lab, p) in zip(ax, SETS.items()):
        f, s = run(p, 6, 0.25, True, t_end=130.0, mode="normal", t_eval=np.linspace(0, 130, 1301))
        y = np.concatenate([f.y[:, :-1], s.y], axis=1)
        t = np.concatenate([f.t[:-1], s.t])
        for k, (name, c, ls) in enumerate((("m_A", "#3b75af", "-"), ("m_C", "#d1495b", "-"), ("b_A", "#3b75af", "--"),
                                           ("b_C", "#d1495b", "--"))):
            a.plot(t, y[k], color=c, ls=ls, label=name)
        a.plot(t, phi2(p, y[0], y[1]), color="black", lw=1, label="Φ₂ (controlled)")
        a.axvline(30, color="0.5", ls=":", lw=0.6)
        a.set_title(f"{lab}, ρ = 0.25 (controlled arm)", fontsize=7.5)
        a.set_xlabel("time")
    ax[0].legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_trajectories.pdf")
    plt.close(fig)


def fig4():
    g = pd.read_csv(OUT / "rev3_grid.csv")
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    for rho, c in ((0.25, "#3b75af"), (0.75, "#d1495b")):
        ax.scatter(g[f"Reff_{rho}"], g[f"ret_{rho}"], color=c, s=14, label=f"ρ = {rho}")
    ax.axvline(1, color="0.5", ls=":", lw=0.8)
    ax.set_xlabel("R_eff at the stop (growth number of controller facts)")
    ax.set_ylabel("retained δ(130)/δ(30)")
    ax.set_title("persistence follows R_eff > 1")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_threshold.pdf")
    plt.close(fig)


def fig5():
    p = SETS["high eligibility (ε = 1)"]
    rows = []
    for r in RH:
        _, c1, s1 = G_at_T(p, 3, r)
        _, c2, s2 = G_at_T(p, 6, r)
        D = G_at_T(p, 6, r)[0] - G_at_T(p, 3, r)[0]
        MCm = 0.5 * p.F_C * (c1[1] + c2[1])
        Phi = sigma(p.h0 - p.beta_A * p.F_A * 0.5 * (s1[0] + s2[0]))
        sp = Phi * (1 - Phi)
        dose = p.beta_C * sp * p.F_C * (c2[1] - c1[1])
        sens = -p.beta_C * sp * (1 - 2 * Phi) * p.beta_A * MCm * p.F_A * (s2[0] - s1[0])
        rows.append((r, D, dose + sens, dose, sens))
    a = np.array(rows)
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    ax.plot(a[:, 0], a[:, 1], "k-o", ms=2.5, label="D(3→6), full model")
    ax.plot(a[:, 0], a[:, 2], "k--", label="small-effect formula")
    ax.plot(a[:, 0], a[:, 3], color="#d1495b", label="dose term (+)")
    ax.plot(a[:, 0], a[:, 4], color="#3b75af", label="sensitivity term (−)")
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set_xlabel("memory ρ")
    ax.set_title("why the sign changes (ε = 1)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_dose_sensitivity.pdf")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    fig1(); fig2(); fig3(); fig4(); fig5()
    print("figures ->", OUT)
