"""Step 2b figures and checks, from s2b_decomposition.py outputs (tables/step2b_*.csv) and s2_mechanism.py
(results/key_phenomena/s2_composition.parquet). Writes into key_phenomena_hypothesis/figures/ and tables/:

  step2b_fig1_heatmaps.pdf   2 x 3: D_direct, D_indirect, D_total,belief (columns) for symmetric and nosolution (rows),
                             step in q (x) and rho (y); one diverging scale centred on 0; grey number: 95% interval
                             includes 0; corner marker on D_total,belief: sign of D_total,vote (▲ +, ▼ -, ○ unresolved)
  step2b_fig2_vote_vs_belief.pdf  E = D_vote - D_belief per setup (grey: interval includes 0; corner: '=' signs of
                             D_vote and D_belief both resolved and equal, '~' at least one unresolved)
  step2b_fig3_vote_belief_ratio.pdf  G_vote / G_belief at each (rho, q) against the expected share of remembered facts
                             forgotten since the agent's last vote, gamma / (1 + gamma)
  tables/step2b_checks.csv   correlations used in the report; tables/step2b_ambiguous_cells.csv

Usage, from the repository root:
    .venv/bin/python analysis/task003_llm_free/simulation_2/key_phenomena_hypothesis/scripts/s2b_figures.py
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
FIG, TAB = BASE / "figures", BASE / "tables"
COMP = HERE.parents[2] / "results" / "key_phenomena" / "s2_composition.parquet"
SETUPS = ["symmetric", "nosolution"]
STEPS = ["3->6", "6->9", "9->12"]
RHOS = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "figure.dpi": 150})
MARK = {"+": "▲", "-": "▼", "0": "○"}


def grid(d, col):
    return d.pivot_table(index="rho", columns="step", values=col, aggfunc="first").reindex(index=RHOS, columns=STEPS)


def panel(ax, d, col, vmax, cmap="RdBu_r", corner=None):
    m, sg = grid(d, col), grid(d, f"{col}_sign")
    im = ax.imshow(m.to_numpy(float), origin="lower", cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    for a in range(len(RHOS)):
        for b in range(len(STEPS)):
            v = m.iat[a, b]
            ax.text(b, a, f"{v:+.3f}", ha="center", va="center", fontsize=5.6,
                    color="0.6" if sg.iat[a, b] == "0" else "black")
            if corner is not None:
                ax.text(b + 0.42, a + 0.38, corner.iat[a, b], ha="right", va="top", fontsize=5, color="0.15")
    ax.set_xticks(range(3), [s.replace("->", "→") for s in STEPS])
    ax.set_yticks(range(len(RHOS)), [f"{r:g}" for r in RHOS])
    return im


def main():
    D = pd.read_csv(TAB / "step2b_D.csv", dtype={"D_direct_sign": str, "D_indirect_sign": str, "D_belief_sign": str,
                                                  "D_vote_sign": str, "E_sign": str})
    D["E_sign"] = D.E_sign.astype(str)
    vmax = float(np.ceil(D[["D_direct", "D_indirect", "D_belief"]].abs().max().max() * 100) / 100)
    fig = plt.figure(figsize=(7.6, 6.4))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.06], wspace=0.28, hspace=0.18)
    axes = np.array([[fig.add_subplot(gs[i, j]) for j in range(3)] for i in range(2)])
    cols = (("D_direct", "$D_{\\rm direct}$"), ("D_indirect", "$D_{\\rm indirect}$"),
            ("D_belief", "$D_{\\rm total,belief}$ (corner: $D_{\\rm total,vote}$ sign)"))
    for i, s in enumerate(SETUPS):
        d = D[D.setup == s]
        vote = grid(d, "D_vote_sign").apply(lambda c: c.map(MARK))
        for j, (col, lab) in enumerate(cols):
            im = panel(axes[i, j], d, col, vmax, corner=vote if j == 2 else None)
            if i == 0:
                axes[i, j].set_title(lab, fontsize=7)
            if i == 1:
                axes[i, j].set_xlabel("reading step")
            if j == 0:
                axes[i, j].set_ylabel(f"{s}\nmemory $\\rho$")
            else:
                axes[i, j].set_yticklabels([])
    fig.colorbar(im, cax=fig.add_subplot(gs[:, 3]), label="$D$ (positive: more reading amplifies)")
    fig.savefig(FIG / "step2b_fig1_heatmaps.pdf", bbox_inches="tight")
    plt.close(fig)

    emax = float(np.ceil(D.E.abs().max() * 100) / 100)
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.2))
    for j, s in enumerate(SETUPS):
        d = D[D.setup == s]
        agree = pd.DataFrame("~", index=RHOS, columns=STEPS)
        bs, vs = grid(d, "D_belief_sign"), grid(d, "D_vote_sign")
        for a in RHOS:
            for b in STEPS:
                if bs.at[a, b] != "0" and vs.at[a, b] != "0":
                    agree.at[a, b] = "=" if bs.at[a, b] == vs.at[a, b] else "≠"
        im = panel(axes[j], d, "E", emax, cmap="PuOr_r", corner=agree)
        axes[j].set_title(f"{s}: $E = D_{{\\rm total,vote}} - D_{{\\rm total,belief}}$", fontsize=7)
        axes[j].set_xlabel("reading step")
        axes[j].set_ylabel("memory $\\rho$")
    fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02, label="$E$")
    fig.savefig(FIG / "step2b_fig2_vote_vs_belief.pdf", bbox_inches="tight")
    plt.close(fig)

    G = pd.read_csv(TAB / "step2b_G.csv")
    G = G[(G.rho < 1) & (G.G_belief.abs() > 0.02)].copy()
    G["ratio"] = G.G_vote / G.G_belief
    G["loss"] = -np.log(G.rho) / (1 - np.log(G.rho))
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    for s, c in zip(SETUPS, ("tab:red", "tab:blue")):
        g = G[G.setup == s]
        ax.scatter(g.loss, g.ratio, s=12, color=c, label=s)
    ax.axhline(1, color="0.6", lw=0.6)
    ax.set_xlabel("expected share of facts forgotten since\nthe last vote, $\\gamma/(1+\\gamma)$")
    ax.set_ylabel("$G_{\\rm total,vote} / G_{\\rm total,belief}$")
    ax.legend(fontsize=6)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "step2b_fig3_vote_belief_ratio.pdf", bbox_inches="tight")
    plt.close(fig)

    # checks quoted in the report
    C = pd.read_parquet(COMP)
    C = C[C.t == 30.0]
    p = C.pivot_table(index=["setup", "rho", "q"], columns="arm", values="mem_agent")
    gap = (p.silent - p.false).unstack("q")
    rows = [(s, r, f"{lo}->{hi}", g[hi] - g[lo]) for (s, r), g in gap.iterrows() for lo, hi in ((3, 6), (6, 9), (9, 12))]
    x = pd.DataFrame(rows, columns=["setup", "rho", "step", "d_count_gap"]).merge(D, on=["setup", "rho", "step"])
    x = x[x.rho < 1]
    checks = []
    for s in SETUPS:
        y, g = x[x.setup == s], G[G.setup == s]
        checks += [(s, "corr(D_indirect, change of agent-fact count gap silent-controller)",
                    np.corrcoef(y.D_indirect, y.d_count_gap)[0, 1]),
                   (s, "corr(G_vote/G_belief, gamma/(1+gamma))", np.corrcoef(g.ratio, g.loss)[0, 1])]
        r = D[(D.setup == s) & (D.rho < 1)]
        both = r[(r.D_vote_sign != "0") & (r.D_belief_sign != "0")]
        checks += [(s, "cells with D_vote and D_belief both resolved", len(both)),
                   (s, "of which signs agree", int((both.D_vote_sign == both.D_belief_sign).sum()))]
    pd.DataFrame(checks, columns=["setup", "check", "value"]).round(3).to_csv(TAB / "step2b_checks.csv", index=False)
    amb = D[(D.rho < 1) & ((D.D_vote_sign == "0") | (D.D_belief_sign == "0"))]
    amb[["setup", "rho", "step", "D_vote", "D_vote_lo", "D_vote_hi", "D_vote_sign", "D_belief", "D_belief_lo",
         "D_belief_hi", "D_belief_sign"]].round(4).to_csv(TAB / "step2b_ambiguous_cells.csv", index=False)
    print("->", FIG, TAB)


if __name__ == "__main__":
    main()
