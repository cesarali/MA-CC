"""Step 1 figures and small summary tables, from the outputs of s1_reversal.py, s1_mechanism.py, s1_persistence.py and
s1_backfire.py (results/key_phenomena/). Writes into key_phenomena_hypothesis/figures/ and tables/:

  fig1_reversal_map.pdf      left: G(t = 30) against q, one line per rho; right: the paired increments D(3->6) and
                             D(6->12) against rho with 95% intervals; rows: symmetric, nosolution
  fig2_reversal_time.pdf     D(3->6) and D(6->12) over time for every rho (main setting)
  fig3_persistence.pdf       top: delta(s) / delta(0) after the last controller post (rate 8), one line per rho;
                             bottom: share of readable posts carrying a controller-only fact over time (q 6, rho 0.75, 1)
  tables/reversal_t30.csv    G and D with paired errors at t = 30, main setting
  tables/reversal_sign_counts.csv  per rho and step, settings (of 72: qc x rate x gate) with D > 0, ~0, < 0
  tables/persistence.csv     one row per setting (rates 1 and 8, all cases)
  tables/mechanism_t30_t40.csv  memory and board composition at t = 30 and 40
  tables/backfire.csv        nosolution gain against messages and distinct facts

Main setting: q_c = 12, rate 1, gate on, B = 30, W = 48.

Usage, from the repository root:
    .venv/bin/python analysis/task003_llm_free/simulation_2/key_phenomena_hypothesis/scripts/s1_figures.py
"""
from __future__ import annotations
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = HERE.parents[2]
IN = ROOT / "results" / "key_phenomena"
FIG, TAB = BASE / "figures", BASE / "tables"
SETUPS = ["symmetric", "nosolution"]
RHOS = [1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125]
RCOL = dict(zip(RHOS, plt.cm.viridis(np.linspace(0, 0.92, len(RHOS)))))
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "legend.fontsize": 6.5, "figure.dpi": 150})


def main_setting(d):
    return d[(d.case == "standard") & (d.qc == 12) & (d.lambda_c == 1.0) & (d.gate == "on")]


def fig_reversal(G, D):
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.2))
    g, d = main_setting(G), main_setting(D)
    for i, s in enumerate(SETUPS):
        ax = axes[i, 0]
        for rho in RHOS:
            x = g[(g.setup == s) & (g.rho == rho) & (g.t == 30.0)].sort_values("q")
            ax.errorbar(x.q, x.G, yerr=1.96 * x.G_se, color=RCOL[rho], marker="o", ms=3, lw=1, capsize=1.5,
                        label=f"$\\rho$ = {rho:g}")
        ax.set_xscale("log", base=2)
        ax.set_xticks([3, 6, 12], ["3", "6", "12"])
        ax.set_xlabel("posts read per action $q$")
        ax.set_ylabel(f"{s}\n$G$ at $t = 30$, target A2")
        ax = axes[i, 1]
        for step, mk, col in (("3->6", "o", "tab:blue"), ("6->12", "s", "tab:red")):
            x = d[(d.setup == s) & (d.step == step) & (d.t == 30.0)].sort_values("rho")
            ax.errorbar(x.rho, x.D, yerr=1.96 * x.D_se, color=col, marker=mk, ms=3, lw=1, capsize=1.5,
                        label=f"$D$({step.replace('->', '→')})")
        ax.axhline(0, color="0.5", lw=0.6)
        ax.set_xlabel("memory $\\rho$")
        ax.set_ylabel("paired increment $D$ at $t = 30$")
    axes[0, 0].legend(fontsize=5.5, ncol=2)
    axes[0, 1].legend(fontsize=6)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "fig1_reversal_map.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_reversal_time(D):
    d = main_setting(D)
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.0), sharex=True)
    for i, s in enumerate(SETUPS):
        for j, step in enumerate(("3->6", "6->12")):
            ax = axes[i, j]
            for rho in RHOS:
                x = d[(d.setup == s) & (d.step == step) & (d.rho == rho)].sort_values("t")
                ax.plot(x.t, x.D, color=RCOL[rho], lw=1, label=f"$\\rho$ = {rho:g}")
                ax.fill_between(x.t, x.D - 1.96 * x.D_se, x.D + 1.96 * x.D_se, color=RCOL[rho], alpha=0.12, lw=0)
            ax.axhline(0, color="0.5", lw=0.6)
            ax.axvline(30, color="0.7", ls="--", lw=0.6)
            ax.set_title(f"{s}: $D$({step.replace('->', '→')})")
            if i == 1:
                ax.set_xlabel("time $t$")
    axes[0, 0].legend(fontsize=5.5, ncol=2)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "fig2_reversal_time.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_persistence(C, M):
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.0))
    c = C[(C.case == "standard") & (C.lambda_c == 8.0) & (C.n >= 100)]
    for j, s in enumerate(SETUPS):
        ax = axes[0, j]
        for rho in RHOS:
            x = c[(c.setup == s) & (c.rho == rho)].sort_values("s")
            if len(x) and abs(x.delta.iloc[0]) > 0.01:
                ax.plot(x.s, x.delta / x.delta.iloc[0], color=RCOL[rho], lw=1, label=f"$\\rho$ = {rho:g}")
        ax.axhline(0, color="0.5", lw=0.6)
        ax.axhline(1, color="0.8", lw=0.6, ls=":")
        ax.set_title(f"{s}: effect after the last post ($\\lambda_c$ = 8)")
        ax.set_xlabel("time since the controller's last post $s$")
        ax.set_ylabel("$\\delta(s)/\\delta(0)$")
        ax = axes[1, j]
        m = M[M.setup == s].groupby(["rho", "arm", "t"]).board_ctrl.mean().reset_index()
        for rho, ls in ((0.75, "-"), (1.0, "--")):
            x = m[(m.rho == rho) & (m.arm == "false")]
            ax.plot(x.t, x.board_ctrl, color="tab:red", ls=ls, marker="o", ms=2.5, lw=1, label=f"$\\rho$ = {rho:g}")
        ax.axvline(30, color="0.7", ls="--", lw=0.6)
        ax.set_ylim(0, 0.5)
        ax.set_xlabel("time $t$")
        ax.set_ylabel("readable posts with a\ncontroller-only fact")
        ax.set_title(f"{s}: board after the controller stops ($q$ = 6, $\\lambda_c$ = 1)")
    axes[0, 0].legend(fontsize=5.5, ncol=2)
    axes[1, 0].legend(fontsize=6)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "fig3_persistence.pdf", bbox_inches="tight")
    plt.close(fig)


def tables(G, D, P, M, B):
    g = main_setting(G)
    g = g[g.t == 30.0].pivot_table(index=["setup", "rho"], columns="q", values=["G", "G_se"])
    g.columns = [f"{a}_q{b}" for a, b in g.columns]
    d = main_setting(D)
    d = d[d.t == 30.0].pivot_table(index=["setup", "rho"], columns="step", values=["D", "D_se", "D_se_unpaired"])
    d.columns = [f"{a}_{b}" for a, b in d.columns]
    g.join(d).reset_index().round(4).to_csv(TAB / "reversal_t30.csv", index=False)
    s = D[(D.case == "standard") & (D.t == 30.0) & (D.step != "3->12")].copy()
    s["sign"] = np.where(s.D > 1.96 * s.D_se, "positive", np.where(s.D < -1.96 * s.D_se, "negative", "not_significant"))
    s.groupby(["setup", "step", "rho"]).sign.value_counts().unstack(fill_value=0).reset_index().to_csv(
        TAB / "reversal_sign_counts.csv", index=False)
    P[P.lambda_c.isin([1.0, 8.0])].round(4).to_csv(TAB / "persistence.csv", index=False)
    M[M.t.isin([30.0, 40.0])].groupby(["setup", "rho", "arm", "t"]).mean(numeric_only=True).drop(
        columns="episode").round(4).reset_index().to_csv(TAB / "mechanism_t30_t40.csv", index=False)
    B.round(4).to_csv(TAB / "backfire.csv", index=False)


def main():
    FIG.mkdir(exist_ok=True)
    TAB.mkdir(exist_ok=True)
    G, D = pd.read_parquet(IN / "reversal_G.parquet"), pd.read_parquet(IN / "reversal.parquet")
    P, C = pd.read_parquet(IN / "persistence.parquet"), pd.read_parquet(IN / "persistence_curves.parquet")
    M, B = pd.read_parquet(IN / "mechanism.parquet"), pd.read_parquet(IN / "backfire.parquet")
    fig_reversal(G, D)
    fig_reversal_time(D)
    fig_persistence(C, M)
    tables(G, D, P, M, B)
    print("->", FIG, TAB)


if __name__ == "__main__":
    main()
