"""Figures for the metastability report (exploratory; 3 Nov 2026). Reads <out>/table_tipping.csv (tipping.py) and,
for the histograms, the runs' snapshots. Usage, from the repository root:
    .venv/bin/python analysis/15_agent_simulation/analysis/tipping_figures.py --runs <task003 run> <op run> --out <folder>
"""
import argparse, pathlib, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from figures import Q, RHO, SETUPS, grid, heat  # noqa: E402

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})


def four(path, fn, xlabel="q (posts read per action)"):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0))
    for ax, (s, label) in zip(axes.flat, SETUPS):
        fn(ax, s, label)
    for ax in axes[1]:
        ax.set_xlabel(xlabel)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs=2, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    F = out / "figures"
    F.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(out / "table_tipping.csv")
    p = d[d.read_mode == "posts"]

    for t, tag in ((130.0, "t130"), (30.0, "t30")):
        x = p[p.t == t]
        four(F / f"fig01_tipping_A2_{tag}.pdf",
             lambda ax, s, l: heat(ax, grid(x[x.setup == s], "tip_A2_23"), f"{l}: tipping into A2 lock-in",
                                   diverging=True, vmax=0.9, fmt="{:+.2f}"))
        four(F / f"fig02_diversion_A0_{tag}.pdf",
             lambda ax, s, l: heat(ax, grid(x[x.setup == s], "div_A0_23"), f"{l}: change of A0 lock-in",
                                   diverging=True, vmax=0.7, fmt="{:+.2f}"))

    # fig03: the most frequent end state of the false arm at t = 130 (0 mixed, 1 A0 lock-in, 2 A2 lock-in)
    x = p[p.t == 130.0].copy()
    x["mixed"] = 1 - x.lock_A0_23_false - x.lock_A1_23_false - x.lock_A2_23_false
    x["state"] = np.argmax(x[["mixed", "lock_A0_23_false", "lock_A2_23_false"]].to_numpy(), axis=1)
    x["state_share"] = x[["mixed", "lock_A0_23_false", "lock_A2_23_false"]].max(axis=1)
    cmap = ListedColormap(["#cccccc", "#3b75af", "#d1495b"])

    def f3(ax, s, l):
        m = grid(x[x.setup == s], "state")
        sh = grid(x[x.setup == s], "state_share")
        ax.imshow(m, origin="lower", cmap=cmap, vmin=-0.5, vmax=2.5, aspect="auto")
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                ax.text(j, i, f"{sh[i, j]:.2f}", ha="center", va="center", fontsize=5)
        ax.set_xticks(range(len(Q)))
        ax.set_xticklabels(Q)
        ax.set_yticks(range(len(RHO)))
        ax.set_yticklabels([f"{r:g}" for r in RHO], fontsize=6)
        ax.set_title(f"{l}: most frequent end state (false arm)", fontsize=8)
    four(F / "fig03_end_state_map.pdf", f3)

    # fig04: per-episode A2 share at t = 130, silent against false, for chosen cells
    runs = {"task003": pathlib.Path(a.runs[0]), "op": pathlib.Path(a.runs[1])}
    picks = [("task003", "task003_symmetric_15", 6, 0.61), ("task003", "task003_symmetric_15", 6, 0.135),
             ("task003", "task003_nosolution_uniform_15", 6, 0.94), ("task003", "task003_nosolution_uniform_15", 6, 0.37),
             ("op", "op_symmetric_15", 6, 0.37), ("op", "op_nosolution_uniform_15", 6, 0.94)]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.4))
    for ax, (task, s, q, rho) in zip(axes.flat, picks):
        for arm, col in (("silent", "0.55"), ("false", "#d1495b")):
            c = runs[task] / "cells" / f"{s}__{arm}__q{q}__rho{rho:g}__readposts"
            z = pd.read_parquet(c / "snapshots.parquet", columns=["t", "share_A2", "share_voted"])
            z = z[np.isclose(z.t, 130)]
            v = (z.share_A2 / z.share_voted.where(z.share_voted > 0)).dropna()
            ax.hist(v, bins=np.linspace(0, 1, 16), color=col, alpha=0.6, label=arm)
        name = dict(SETUPS)[s]
        ax.set_title(f"{name}\nq {q}, ρ {rho:g}", fontsize=7)
        ax.set_xlabel("A2 share among voters at t = 130", fontsize=6)
    axes[0, 0].legend(frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(F / "fig04_end_state_histograms.pdf")
    plt.close(fig)

    # fig05: bistability index of the false arm at t = 130: min(P(A2 share <= 0.2), P(A2 share >= 0.8)); large only when
    # episodes split between both extremes
    x5 = p[p.t == 130.0].copy()
    x5["bistable"] = np.minimum(x5.low_A2_false, x5.high_A2_false)
    four(F / "fig05_bistability.pdf",
         lambda ax, s, l: heat(ax, grid(x5[x5.setup == s], "bistable"), f"{l}: both extremes populated",
                               cmap="viridis", vmax=0.4, fmt="{:.2f}"))
    print("max bistability index:", x5.groupby("setup").bistable.max().round(2).to_dict())

    # robustness: 80% threshold against 2/3
    r = p[p.t == 130.0]
    print("corr tip 2/3 vs 0.8:", round(np.corrcoef(r.tip_A2_23, r.tip_A2_80)[0, 1], 3),
          "| corr div 2/3 vs 0.8:", round(np.corrcoef(r.div_A0_23, r.div_A0_80)[0, 1], 3))
    print("max tip t30/t130 by setup:", p.groupby(["setup", "t"]).tip_A2_23.max().round(2).to_dict())
    print("max div A0 t130 by setup:", p[p.t == 130].groupby("setup").div_A0_23.max().round(2).to_dict())
    print("figures ->", F)


if __name__ == "__main__":
    main()
