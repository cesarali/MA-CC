"""Figures of the side-by-side comparison report (task_comparison.tex): task003 and Object Placements, both setups.
Writes figures/fig*.pdf. Run carrier_after_stop.py, board_persuasion.py, context_value.py, ctrl_diversity.py and
task_structure.py first (their CSVs are inputs here).

Usage, from the repository root:
    .venv/bin/python analysis/theory_claude/reports/2026-10-23_task_comparison/make_figures.py
"""
from __future__ import annotations
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
A = HERE.parents[2]
BENCH = A / "task003_llm_free" / "results" / "key_phenomena" / "theory_benchmarks"
OPA = A / "object_placements_llm_free" / "simulation_2" / "analysis"
OUT = HERE / "figures"

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42, "lines.linewidth": 1.3})
QCOL = {3: "#9ecae1", 6: "#4292c6", 9: "#2171b5", 12: "#08306b"}
SCOL = {"3->6": "#4292c6", "6->9": "#2171b5", "9->12": "#08306b"}
MC = {"normal": "#222222", "frozen_board": "#4292c6", "no_comm": "#e6a532", "purge_controller_facts": "#d1495b"}
ML = {"normal": "normal", "frozen_board": "frozen board", "no_comm": "no communication",
      "purge_controller_facts": "controller facts removed"}
TASKS = ["task003", "Object Placements"]
SETUPS = [("symmetric", "symmetric"), ("nosolution", "no-proof")]
TC = {"task003": "#555555", "Object Placements": "#d1495b"}
W2 = 6.8


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print("->", OUT / name)


def zero(ax):
    ax.axhline(0, color="0.6", lw=0.6, zorder=0)


def load_GD():
    """G (q 3..12) and D (three steps) with 95% intervals, both tasks, one table."""
    d = pd.read_parquet(BENCH / "G_D.parquet")
    g1 = d[d.kind == "G"].assign(task="task003", value=lambda x: x.G_vote, lo=lambda x: x.G_vote_lo,
                                 hi=lambda x: x.G_vote_hi, key=lambda x: "q" + x.q.astype(int).astype(str))
    d1 = d[d.kind == "D"].assign(task="task003", value=lambda x: x.D_vote, lo=lambda x: x.D_vote_lo,
                                 hi=lambda x: x.D_vote_hi, key=lambda x: x.step, sign=lambda x: x.D_vote_sign)
    o = pd.read_csv(OPA / "op_data.csv", dtype={"sign": str})
    o = o.assign(task="Object Placements", setup=o.setup.str.replace("op_", ""), lo=o.value - 1.96 * o.se,
                 hi=o.value + 1.96 * o.se, key=o.step)
    cols = ["task", "setup", "rho", "key", "value", "lo", "hi"]
    G = pd.concat([g1[cols], o[o.quantity == "G"][cols]])
    D = pd.concat([d1[cols + ["sign"]], o[o.quantity == "D"][cols + ["sign"]]])
    return G, D


def grid(title_fmt, sharey="row"):
    fig, ax = plt.subplots(2, 2, figsize=(W2, 4.0), sharex=True, sharey=sharey)
    for i, (s, sl) in enumerate(SETUPS):
        for j, t in enumerate(TASKS):
            ax[i, j].set_title(title_fmt.format(task=t, setup=sl))
    return fig, ax


def fig_G(G):
    fig, ax = grid("{task}, {setup}: $G$")
    for i, (s, _) in enumerate(SETUPS):
        for j, t in enumerate(TASKS):
            for q in (3, 6, 9, 12):
                x = G[(G.task == t) & (G.setup == s) & (G.key == f"q{q}")].sort_values("rho")
                ax[i, j].plot(x.rho, x.value, "o-", ms=2.5, color=QCOL[q], label=f"$q={q}$")
                ax[i, j].fill_between(x.rho, x.lo, x.hi, color=QCOL[q], alpha=0.2, lw=0)
            zero(ax[i, j])
        ax[i, 0].set_ylabel("$G$ at $t=30$")
    for j in (0, 1):
        ax[1, j].set_xlabel(r"memory $\rho$")
    ax[0, 1].legend(frameon=False, ncol=2)
    fig.tight_layout()
    save(fig, "fig1_G_side_by_side.pdf")


def fig_D(D):
    fig, ax = grid("{task}, {setup}: $D$")
    for i, (s, _) in enumerate(SETUPS):
        for j, t in enumerate(TASKS):
            for st, c in SCOL.items():
                x = D[(D.task == t) & (D.setup == s) & (D.key == st)].sort_values("rho")
                ax[i, j].errorbar(x.rho, x.value, yerr=[x.value - x.lo, x.hi - x.value], fmt="o-", ms=2.5,
                                  capsize=1.5, color=c, label=st.replace("->", r"$\to$"))
            zero(ax[i, j])
        ax[i, 0].set_ylabel(r"$D(q_1\to q_2)$")
    for j in (0, 1):
        ax[1, j].set_xlabel(r"memory $\rho$")
    ax[0, 1].legend(frameon=False, title="reading step", title_fontsize=7)
    fig.tight_layout()
    save(fig, "fig2_D_side_by_side.pdf")


def brackets(seq):
    """rho intervals in which the resolved sign changes: [(lo, hi, from, to)]."""
    res = [(r, s) for r, s in seq if s in "+-"]
    return [(a, b, s, t) for (a, s), (b, t) in zip(res[:-1], res[1:]) if s != t]


def fig_crossovers(D):
    """Where the reading increment changes sign, per task, setup and step."""
    fig, ax = plt.subplots(figsize=(W2, 2.3))
    y, labels = 0, []
    for s, sl in SETUPS:
        for st in SCOL:
            for t in TASKS:
                x = D[(D.task == t) & (D.setup == s) & (D.key == st)].sort_values("rho")
                br = brackets(list(zip(x.rho, x.sign)))
                for lo, hi, a, b in br:
                    ax.plot([lo, hi], [y, y], "-", lw=6, color=TC[t], solid_capstyle="butt", alpha=0.85)
                    ax.text(hi + 0.01, y, f"{a}$\\to${b}", va="center", fontsize=6.5, color=TC[t])
                if not br:
                    ax.text(0.13, y, "no resolved sign change", va="center", fontsize=6.5, color=TC[t])
                labels.append(f"{sl} {st.replace('->', '→')} — {t}")
                y -= 1
            y -= 0.5
    ax.set_yticks([0, -1, -2.5, -3.5, -5, -6, -7.5, -8.5, -10, -11, -12.5, -13.5][:len(labels)])
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlim(0.1, 1.05)
    ax.set_ylim(y + 0.6, 0.6)
    ax.set_xlabel(r"memory $\rho$: interval in which $D$ changes sign")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    save(fig, "fig3_crossovers.pdf")


def load_post_stop():
    d1 = pd.read_parquet(BENCH / "post_stop.parquet").assign(task="task003")
    d2 = pd.read_csv(OPA / "op_post_stop_data.csv").assign(task="Object Placements")
    d2["setup"] = d2.setup.str.replace("op_", "")
    return pd.concat([d1[["task", "setup", "rho", "mode", "t", "delta", "se"]],
                      d2[["task", "setup", "rho", "mode", "t", "delta", "se"]]])


def fig_post_stop(P, rho, name):
    fig, ax = grid("{task}, {setup}: $\\delta(t)$, $\\rho=" + f"{rho}" + "$")
    for i, (s, _) in enumerate(SETUPS):
        for j, t in enumerate(TASKS):
            a = ax[i, j]
            for mode in MC:
                x = P[(P.task == t) & (P.setup == s) & np.isclose(P.rho, rho) & (P["mode"] == mode) & (P.t >= 20)].sort_values("t")
                a.plot(x.t, x.delta, color=MC[mode], lw=1.3, label=ML[mode])
                a.fill_between(x.t, x.delta - 1.96 * x.se, x.delta + 1.96 * x.se, color=MC[mode], alpha=0.15, lw=0)
            a.axvline(30, color="0.5", lw=0.6, ls=":")
            zero(a)
            a.set_xscale("function", functions=(lambda u: np.where(u < 30, u, 30 + 8 * np.log1p(np.maximum(u - 30, 0) / 3)),
                                                lambda v: np.where(v < 30, v, 30 + 3 * np.expm1(np.maximum(v - 30, 0) / 8))))
            a.set_xticks([20, 30, 31, 35, 40, 60, 130])
        ax[i, 0].set_ylabel(r"$\delta(t)$")
    for j in (0, 1):
        ax[1, j].set_xlabel("time $t$ (controller stops at 30; stretched after 30)")
    h, lab = ax[0, 0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.legend(h, lab, frameon=False, ncol=4, loc="upper center")
    save(fig, name)


def fig_mechanism():
    """Why the imprint fades in Object Placements: circulation versus persuasion after the stop (false arm, normal)."""
    car = pd.read_csv(HERE / "carrier_after_stop.csv")
    bp = pd.read_csv(HERE / "board_persuasion.csv").pivot_table(index=["task", "setup", "rho", "t"], columns="arm",
                                                                 values="read_p2").reset_index()
    bp["diff"] = bp["false"] - bp["silent"]
    cv = pd.read_csv(HERE / "context_value.csv")
    dv = pd.read_csv(HERE / "ctrl_diversity.csv")
    car = car[car.arm == "false"]
    panels = [(car, "board_ctrl", "(a) share of the board carrying controller-only facts"),
              (bp, "diff", "(b) $P(A2\\mid$one read of 6 posts$)$, false $-$ silent"),
              (dv, "distinct_ctrl_board", "(c) distinct controller-only facts on one board"),
              (cv, "value_alone", "(d) controller facts of a read, alone: $P(A2\\mid\\cdot)-1/3$")]
    fig, ax = plt.subplots(2, 2, figsize=(W2, 4.2))
    ax = ax.ravel()
    for a, (d, col, title) in zip(ax, panels):
        for t in TASKS:
            for s, sl in SETUPS:
                x = d[(d.task == t) & (d.setup == s) & np.isclose(d.rho, 0.25)].sort_values("t")
                a.plot(x.t, x[col], "o-" if s == "symmetric" else "s--", ms=3, color=TC[t], label=f"{t}, {sl}")
        a.set_title(title.replace("\n", " "), fontsize=7.5)
        a.set_xlabel("time $t$ (controller stopped at 30)")
        a.set_xticks([30, 55, 80, 105, 130])
        a.set_ylim(bottom=0)
    h, lab = ax[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.legend(h, lab, frameon=False, ncol=4, loc="upper center")
    save(fig, "fig6_mechanism_after_stop.pdf")


def fig_structure():
    """Task structure from the worlds alone: single-fact P(A2) of agent and controller-only facts."""
    s = pd.read_csv(HERE / "single_fact_p2.csv")
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.2), sharey=True)
    rng = np.random.default_rng(0)
    for a, (st, sl) in zip(ax, SETUPS):
        for k, (t, kind) in enumerate([(t, kind) for t in TASKS for kind in ("agent", "controller-only")]):
            x = s[(s.task == t) & (s.setup == st) & (s.kind == kind)].p2
            a.scatter(np.full(len(x), k) + rng.uniform(-0.15, 0.15, len(x)), x, s=10, color=TC[t],
                      alpha=0.8 if kind == "controller-only" else 0.35, marker="o" if kind == "agent" else "D")
            a.plot([k - 0.25, k + 0.25], [np.median(x)] * 2, color="black", lw=1)
        a.axhline(1 / 3, color="0.5", ls=":", lw=0.8)
        a.set_xticks(range(4))
        a.set_xticklabels(["task003\nagent", "task003\ncontroller", "Obj. Plac.\nagent", "Obj. Plac.\ncontroller"], fontsize=6.5)
        a.set_title(f"{sl}: $P(A2\\mid\\{{f\\}})$ of single facts (line: median)", fontsize=7.5)
    ax[0].set_ylabel(r"$P(A2\mid\{f\})$ (dotted: prior 1/3)")
    fig.tight_layout()
    save(fig, "fig7_task_structure.pdf")


def main():
    G, D = load_GD()
    fig_G(G)
    fig_D(D)
    fig_crossovers(D)
    P = load_post_stop()
    fig_post_stop(P, 0.25, "fig4_persistence_rho025.pdf")
    fig_post_stop(P, 0.75, "fig5_persistence_rho075.pdf")
    fig_mechanism()
    fig_structure()


if __name__ == "__main__":
    main()
