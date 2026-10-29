"""Numbered figures for a 15-agent campaign report (all four setups). Reads <run>/aggregates/ (aggregate.py) of the
task003 and Object Placements runs and theory_grid.csv; writes PDFs into <out>/figures/ and key tables into <out>/.

  fig01_vote_shares      vote shares among voters against time, silent and false arms (q 6, rho 0.61, standard reading)
  fig02_gain_time        G(t) for q 6 across rho
  fig03_gain_map         phase diagram: G(30) over (q, rho)
  fig04_increment_map    phase diagram: D(q_i -> q_i+1) at t = 30 (sign = reading reversal); dots: |D| > 2 se
  fig05_crossovers       crossover memory against the q pair, all setups and the theory
  fig06_theory_maps      the mean-field theory's G(30) and D on the same grid
  fig07_dedup_map        relative change of G(30) under deduplicated reading
  fig08_retention_map    phase diagram: retained G(130)/G(30) (blank where G(30) < 2 se)
  fig09_gain_after_stop  G(t)/G(30) after the stop for q 6, all rho
  fig10_dedup_controlled_A2  change of the controlled arm's A2 share under deduplicated reading
  fig11_dedup_silent     change of silent accuracy (symmetric: A0 share; no-solution: A2 share) under deduplication
  fig12_theory_retention the theory's retained fraction on the same grid
Usage, from the repository root:
    .venv/bin/python analysis/15_agent_simulation/analysis/figures.py --task003 <run> --op <run> --out <report folder>
"""
from __future__ import annotations
import argparse, pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
Q = [1, 2, 3, 4, 6, 8, 12, 16]
RHO = [0.018, 0.135, 0.37, 0.61, 0.78, 0.88, 0.94, 1.0]
TAU = ["0.25", "0.5", "1", "2", "4", "8", "16", "∞"]
SETUPS = [("task003_symmetric_15", "task003 symmetric"), ("task003_nosolution_uniform_15", "task003 no-solution"),
          ("op_symmetric_15", "OP symmetric"), ("op_nosolution_uniform_15", "OP no-solution")]
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
                     "legend.fontsize": 7})


def load(runs):
    cells, gain = [], []
    for r in runs:
        if r:
            a = pathlib.Path(r) / "aggregates"
            cells.append(pd.read_parquet(a / "cells.parquet"))
            gain.append(pd.read_parquet(a / "gain.parquet"))
    return pd.concat(cells), pd.concat(gain)


def grid(df, value, read="posts"):
    """(rho x q) matrix of a column at t = 30 or of a precomputed column."""
    m = np.full((len(RHO), len(Q)), np.nan)
    for _, r in df[df.read_mode == read].iterrows() if "read_mode" in df else df.iterrows():
        if r.q in Q and any(np.isclose(r.rho, RHO)):
            m[int(np.argmin(np.abs(np.array(RHO) - r.rho))), Q.index(int(r.q))] = r[value]
    return m


def heat(ax, m, title, cmap="viridis", diverging=False, dots=None, fmt="{:.2f}", vmax=None):
    if diverging:
        lim = vmax or np.nanmax(np.abs(m)) or 1
        im = ax.imshow(m, origin="lower", cmap="RdBu_r", norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    else:
        im = ax.imshow(m, origin="lower", cmap=cmap, aspect="auto", vmax=vmax)
    ax.set_xticks(range(len(Q)))
    ax.set_xticklabels(Q)
    ax.set_yticks(range(len(RHO)))
    ax.set_yticklabels([f"{r:g} (τ {t})" for r, t in zip(RHO, TAU)], fontsize=6)
    ax.set_title(title, fontsize=8)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if np.isfinite(m[i, j]):
                ax.text(j, i, fmt.format(m[i, j]), ha="center", va="center", fontsize=4.5,
                        color="black" if diverging else "white")
            if dots is not None and dots[i, j]:
                ax.plot(j + 0.35, i + 0.3, marker="o", ms=1.6, color="black")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    return im


def four(fname, fn, out, figsize=(7.2, 6.0), xlabel="q (posts read per action)", legend=None):
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    for ax, (s, label) in zip(axes.flat, SETUPS):
        fn(ax, s, label)
    for ax in axes[1]:
        ax.set_xlabel(xlabel)
    if legend is not None:
        fig.legend(*legend, frameon=False, ncol=len(legend[0]), loc="lower center", fontsize=7)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        fig.savefig(out / f"{fname}.pdf")
        plt.close(fig)
        return
    fig.tight_layout()
    fig.savefig(out / f"{fname}.pdf")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task003")
    ap.add_argument("--op")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    F = out / "figures"
    cells, gain = load([a.task003, a.op])
    theory = pd.read_csv(HERE / "theory_grid.csv")
    have = set(cells.setup)
    g30 = gain[np.isclose(gain.t, 30)].copy()
    g130 = gain[np.isclose(gain.t, 130)][["cell", "G"]].rename(columns={"G": "G130"})
    g30 = g30.merge(g130, on="cell", how="left")
    g30["retained"] = np.where(g30.G > 2 * g30.G_se, g30.G130 / g30.G, np.nan)
    g30 = g30.sort_values(["setup", "read_mode", "rho", "q"])
    g30["D"] = g30.groupby(["setup", "read_mode", "rho"]).G.shift(-1) - g30.G
    g30["D_se"] = np.sqrt(g30.groupby(["setup", "read_mode", "rho"]).G_se.shift(-1) ** 2 + g30.G_se ** 2)
    g30.to_csv(out / "table_gain_t30.csv", index=False)

    def skip(ax, label):
        ax.text(0.5, 0.5, f"{label}: no data", ha="center", transform=ax.transAxes)

    # fig01: vote shares
    def f1(ax, s, label):
        if s not in have:
            return skip(ax, label)
        c = cells[(cells.setup == s) & (cells.q == 6) & np.isclose(cells.rho, 0.61) & (cells.read_mode == "posts")]
        for arm, ls in (("silent", "--"), ("false", "-")):
            x = c[c.arm == arm]
            for k, col in (("A0", "#3b75af"), ("A1", "0.5"), ("A2", "#d1495b")):
                ax.plot(x.t, x[f"vote_{k}"], ls=ls, color=col, lw=1, label=f"{k}, {arm}")
        ax.axvline(30, color="0.6", lw=0.6, ls=":")
        ax.set_title(f"{label} (q 6, ρ 0.61)", fontsize=8)
        ax.set_ylabel("share of voters")
        ax.set_ylim(0, 1)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=c, lw=1.2) for c in ("#3b75af", "0.5", "#d1495b")] + [
        Line2D([], [], color="k", lw=1, ls="-"), Line2D([], [], color="k", lw=1, ls="--")]
    four("fig01_vote_shares", f1, F, xlabel="time (controller stops at t = 30)",
         legend=(handles, ["A0 (correct)", "A1", "A2 (controller's target)", "false-controller arm", "silent arm"]))

    # fig02: G(t), q 6, all rho
    cmap = plt.get_cmap("viridis")

    def f2(ax, s, label):
        if s not in have:
            return skip(ax, label)
        for i, rho in enumerate(RHO):
            x = gain[(gain.setup == s) & (gain.q == 6) & np.isclose(gain.rho, rho) & (gain.read_mode == "posts")]
            ax.plot(x.t, x.G, color=cmap(i / 7), lw=1, label=f"ρ {rho:g}")
        ax.axvline(30, color="0.6", lw=0.6, ls=":")
        ax.axhline(0, color="0.7", lw=0.5)
        ax.set_title(f"{label} (q 6)", fontsize=8)
        ax.set_ylabel("G(t)")
    four("fig02_gain_time", f2, F, xlabel="time (controller stops at t = 30)")

    # fig03-04: phase diagrams G and D
    four("fig03_gain_map", lambda ax, s, l: heat(ax, grid(g30[g30.setup == s], "G"), l, fmt="{:.2f}")
         if s in have else skip(ax, l), F)

    def f4(ax, s, l):
        if s not in have:
            return skip(ax, l)
        d = g30[g30.setup == s]
        m, se = grid(d, "D"), grid(d, "D_se")
        heat(ax, m[:, :-1], l + ": D(q→next)", diverging=True, dots=(np.abs(m) > 2 * se)[:, :-1], fmt="{:+.2f}")
        ax.set_xticks(range(len(Q) - 1))
        ax.set_xticklabels([f"{a}→{b}" for a, b in zip(Q[:-1], Q[1:])], fontsize=6)
    four("fig04_increment_map", f4, F)

    # fig05: crossover memory against q pair (log-lifetime interpolation)
    tau = np.array([0.25, 0.5, 1, 2, 4, 8, 16, 64.0])          # rho = 1 plotted at tau 64 for display

    def crossing(Ds):
        for i in range(len(Ds) - 1):
            if np.isfinite(Ds[i]) and np.isfinite(Ds[i + 1]) and Ds[i] > 0 >= Ds[i + 1]:
                w = Ds[i] / (Ds[i] - Ds[i + 1])
                return float(np.exp(np.log(tau[i]) + w * (np.log(tau[i + 1]) - np.log(tau[i]))))
        return np.nan
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
    ax, ax2 = axs
    qmid = np.sqrt(np.array(Q[:-1]) * np.array(Q[1:]))      # geometric mean of each reading step
    rows = []
    for s, label in SETUPS:
        if s not in have:
            continue
        m = grid(g30[g30.setup == s], "D")
        xs = [crossing(m[:, j]) for j in range(len(Q) - 1)]
        rows += [dict(setup=s, pair=f"{Q[j]}->{Q[j + 1]}", tau_cross=x) for j, x in enumerate(xs)]
        ax.plot(range(len(Q) - 1), xs, "-o", ms=3, label=label)
        ax2.plot(range(len(Q) - 1), np.array(xs) * qmid, "-o", ms=3, label=label)
    tm = theory.pivot(index="rho", columns="q", values="D_next").reindex(RHO)[Q[:-1]].to_numpy()
    xs = [crossing(tm[:, j]) for j in range(len(Q) - 1)]
    rows += [dict(setup="theory", pair=f"{Q[j]}->{Q[j + 1]}", tau_cross=x) for j, x in enumerate(xs)]
    ax.plot(range(len(Q) - 1), xs, "k--", lw=1, label="mean-field theory (P0)")
    ax2.plot(range(len(Q) - 1), np.array(xs) * qmid, "k--", lw=1, label="mean-field theory (P0)")
    ax2.set_xticks(range(len(Q) - 1))
    ax2.set_xticklabels([f"{a}→{b}" for a, b in zip(Q[:-1], Q[1:])])
    ax2.set_xlabel("reading step")
    ax2.set_ylabel("τ* × q̄ (posts read per memory lifetime)")
    ax2.set_ylim(0, 30)                                     # a noisy crossing (|D| < 2 se on both sides) can lie far above
    ax.set_title("(a) crossover memory", fontsize=8)
    ax2.set_title("(b) crossover in reads per memory lifetime", fontsize=8)
    ax.set_yscale("log")
    ax.set_xticks(range(len(Q) - 1))
    ax.set_xticklabels([f"{a}→{b}" for a, b in zip(Q[:-1], Q[1:])])
    ax.set_xlabel("reading step")
    ax.set_ylabel("crossover memory lifetime τ*")
    ax.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(F / "fig05_crossovers.pdf")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(out / "table_crossovers.csv", index=False)

    # fig06: theory maps
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
    tg = theory.rename(columns={"g30": "G"}).assign(read_mode="posts")
    heat(ax[0], grid(tg, "G"), "theory: g(30)")
    tD = grid(tg.rename(columns={"D_next": "D"}), "D")
    heat(ax[1], tD[:, :-1], "theory: D(q→next)", diverging=True, fmt="{:+.3f}")
    ax[1].set_xticks(range(len(Q) - 1))
    ax[1].set_xticklabels([f"{a}→{b}" for a, b in zip(Q[:-1], Q[1:])], fontsize=6)
    for x in ax:
        x.set_xlabel("q")
    fig.tight_layout()
    fig.savefig(F / "fig06_theory_maps.pdf")
    plt.close(fig)

    # fig07: dedup effect
    def f7(ax, s, l):
        if s not in have:
            return skip(ax, l)
        d = g30[g30.setup == s]
        rel = (grid(d, "G", "distinct_facts") - grid(d, "G")) / grid(d, "G")
        heat(ax, np.clip(rel, -1, 1), l + ": ΔG/G (dedup)", diverging=True, vmax=1, fmt="{:+.2f}")
    four("fig07_dedup_map", f7, F)

    # fig08: retention
    four("fig08_retention_map", lambda ax, s, l: heat(ax, grid(g30[g30.setup == s], "retained"), l + ": G(130)/G(30)",
                                                       cmap="magma", vmax=1.2) if s in have else skip(ax, l), F)

    # fig09: G after the stop
    def f9(ax, s, l):
        if s not in have:
            return skip(ax, l)
        for i, rho in enumerate(RHO):
            x = gain[(gain.setup == s) & (gain.q == 6) & np.isclose(gain.rho, rho) & (gain.read_mode == "posts") & (gain.t >= 30)]
            if len(x):
                ax.plot(x.t, x.G / max(x.G.iloc[0], 1e-9), color=cmap(i / 7), lw=1, label=f"ρ {rho:g}")
        ax.axhline(1, color="0.7", lw=0.5)
        ax.set_ylim(-0.2, 1.6)
        ax.set_title(f"{l} (q 6): G(t)/G(30)", fontsize=8)
    four("fig09_gain_after_stop", f9, F, xlabel="time")
    # fig10: absolute outcomes under the two reading modes (a smaller gap alone is not a defence)
    c30 = cells[np.isclose(cells.t, 30)]

    def f10(ax, s, l):
        if s not in have:
            return skip(ax, l)
        d = c30[c30.setup == s]
        ctrl = lambda rm: grid(d[(d.arm == "false")], "vote_A2", rm)
        heat(ax, ctrl("distinct_facts") - ctrl("posts"), l + ": Δ controlled A2 share (dedup − standard)",
             diverging=True, vmax=0.3, fmt="{:+.2f}")
    four("fig10_dedup_controlled_A2", f10, F)

    def f11(ax, s, l):
        if s not in have:
            return skip(ax, l)
        d = c30[(c30.setup == s) & (c30.arm == "silent")]
        col = "vote_A0" if "symmetric" in s else "vote_A2"
        what = "silent A0 share (accuracy)" if "symmetric" in s else "silent A2 share"
        heat(ax, grid(d, col, "distinct_facts") - grid(d, col), l + f": Δ {what}", diverging=True, vmax=0.3,
             fmt="{:+.2f}")
    four("fig11_dedup_silent", f11, F)

    # fig12: retention, theory against data (task003 symmetric shown with theory for scale)
    fig, ax = plt.subplots(1, 1, figsize=(3.8, 3.0))
    heat(ax, grid(theory.assign(read_mode="posts"), "retained"), "theory: g(130)/g(30)", cmap="magma", vmax=1.2)
    ax.set_xlabel("q")
    fig.tight_layout()
    fig.savefig(F / "fig12_theory_retention.pdf")
    plt.close(fig)
    abs_rows = c30.pivot_table(index=["setup", "q", "rho", "read_mode"], columns="arm",
                               values=["vote_A0", "vote_A2"]).reset_index()
    abs_rows.columns = ["_".join(str(x) for x in c if x) for c in abs_rows.columns]
    abs_rows.to_csv(out / "table_absolute_t30.csv", index=False)
    print("figures ->", F)


if __name__ == "__main__":
    main()
