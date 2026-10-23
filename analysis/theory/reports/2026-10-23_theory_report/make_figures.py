"""Figures of the theory report (theory_report.tex). Writes figures/fig*.pdf (vector, numbered as in the report).

Inputs (none are recomputed here):
  task003 data      task003_llm_free/results/key_phenomena/theory_benchmarks/{G_D,post_stop}.parquet (calibration/build_benchmarks.py)
  model outputs     theory/results/theory_runs/<model>_{all,post_stop}.parquet (theory/models/run_models.py)
  diagnostics       theory/results/theory_runs/diag_scale_check.csv; pinned-fact numbers from theory/M1_M2_mean_field.md §5
  K scaling         key_phenomena_hypothesis/tables/scaling_K_models.csv (finite_size/score_models_K.py)
  Object Placements object_placements_llm_free/simulation_2/analysis/{op_data,preregistered_M3,preregistered_M4}.csv
  scorecards        held_out_validation/scorecard_M*_summary.csv, scorecard_op_M*.csv

Usage, from the repository root:
    .venv/bin/python analysis/theory/reports/2026-10-23_theory_report/make_figures.py
"""
from __future__ import annotations
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
A = HERE.parents[2]                                                   # analysis/
KP = A / "task003_llm_free" / "simulation_2" / "key_phenomena_hypothesis"
BENCH = A / "task003_llm_free" / "results" / "key_phenomena" / "theory_benchmarks"
RUNS = A / "theory" / "results" / "theory_runs"
OPA = A / "object_placements_llm_free" / "simulation_2" / "analysis"
OUT = HERE / "figures"

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42, "lines.linewidth": 1.3})
COL = {"data": "black", "M1": "#bbbbbb", "M2": "#777777", "M3": "#3b75af", "M4": "#d1495b"}
QCOL = {3: "#9ecae1", 6: "#4292c6", 9: "#2171b5", 12: "#08306b"}
SETUPS = [("symmetric", "symmetric"), ("nosolution", "no-proof")]
W1, W2 = 3.3, 6.8                                                     # one / two column widths (inches)


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print("->", OUT / name)


def zero(ax):
    ax.axhline(0, color="0.6", lw=0.6, zorder=0)


# ------------------------------------------------------------------------------------------------ inputs
def bench():
    d = pd.read_parquet(BENCH / "G_D.parquet")
    G = d[d.kind == "G"][["setup", "rho", "q", "G_vote", "G_vote_lo", "G_vote_hi"]]
    D = d[d.kind == "D"][["setup", "rho", "step", "D_vote", "D_vote_lo", "D_vote_hi", "D_vote_sign"]]
    return G, D


def model_G(tag, which="all", t=30.0):
    f = RUNS / f"{tag}_{which}.parquet"
    m = pd.read_parquet(f, columns=["setup", "rho", "q", "arm", "mode", "t", "x2"])
    m = m[np.isclose(m.t, t)]
    p = m.groupby(["setup", "rho", "q", "arm"]).x2.mean().unstack("arm")
    return (p["false"] - p["silent"]).rename("G").reset_index()


# ------------------------------------------------------------------------------------------------ figures
def fig_data_task003(G, D):
    """Fig. 3: the two phenomena's first half in task003: G(q) and the reading increments D by memory."""
    fig, ax = plt.subplots(2, 2, figsize=(W2, 3.9), sharex=True)
    for j, (s, lab) in enumerate(SETUPS):
        g = G[G.setup == s]
        for q in (3, 6, 9, 12):
            x = g[g.q == q].sort_values("rho")
            ax[0, j].plot(x.rho, x.G_vote, "o-", ms=2.5, color=QCOL[q], label=f"$q={q}$")
            ax[0, j].fill_between(x.rho, x.G_vote_lo, x.G_vote_hi, color=QCOL[q], alpha=0.2, lw=0)
        ax[0, j].set_title(f"task003, {lab}: controller effect $G$")
        ax[0, j].set_ylabel("$G$ at $t=30$")
        zero(ax[0, j])
        d = D[D.setup == s]
        for st, c in (("3->6", QCOL[6]), ("6->9", QCOL[9]), ("9->12", QCOL[12])):
            x = d[d.step == st].sort_values("rho")
            ax[1, j].errorbar(x.rho, x.D_vote, yerr=[x.D_vote - x.D_vote_lo, x.D_vote_hi - x.D_vote], fmt="o-",
                              ms=2.5, capsize=1.5, color=c, label=st.replace("->", r"$\to$"))
        zero(ax[1, j])
        ax[1, j].set_title(f"task003, {lab}: reading increment $D$")
        ax[1, j].set_ylabel(r"$D(q_1\to q_2)$")
        ax[1, j].set_xlabel(r"memory $\rho$")
    ax[0, 0].legend(frameon=False, ncol=2)
    ax[1, 0].legend(frameon=False, title="reading step", title_fontsize=7)
    fig.tight_layout()
    save(fig, "fig3_task003_data.pdf")


def fig_diagnostics():
    """Fig. 4: why M1/M2 fail. (a) copies per fact; (b) extinction switched off."""
    d = pd.read_csv(RUNS / "diag_scale_check.csv")
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.3))
    for s, lab in SETUPS:
        x = d[d.setup == s].pivot_table(index=["R", "seed"], columns="arm", values="x2")
        g = (x["false"] - x["silent"]).groupby("R")
        mean, se = g.mean(), g.std() / np.sqrt(g.count())
        c = COL["M4"] if s == "symmetric" else COL["M3"]
        ax[0].errorbar(mean.index, mean.values, yerr=1.96 * se.values, fmt="o-", ms=3, capsize=2, color=c, label=lab)
        mf = {"symmetric": 0.11, "nosolution": 0.12}[s]               # M2 mean-field value, M1_M2_mean_field.md §3
        ax[0].axhline(mf, color=c, ls=":", lw=1)
    ax[0].set_xscale("log", base=2)
    ax[0].set_xticks([1, 2, 4, 8])
    ax[0].set_xticklabels(["1\n(N=24)", "2", "4", "8\n(N=192)"])
    ax[0].set_xlabel("copies of the swarm $R$ (holders per fact $\\approx 1.3R$)")
    ax[0].set_ylabel("$G$ ($\\rho=0.5$, $q=6$)")
    ax[0].set_title("(a) more copies per fact: $G$ falls to mean field (dotted)")
    ax[0].legend(frameon=False)
    ax[0].set_ylim(0, 0.37)
    # (b) pinned own fact: M1_M2_mean_field.md §5 (models/diag_pin_check.py, 300 paired episodes per cell)
    cells = ["symmetric\n$\\rho=0.25$", "symmetric\n$\\rho=0.5$", "no-proof\n$\\rho=0.25$", "no-proof\n$\\rho=0.5$"]
    normal, pinned = [0.441, 0.322, 0.304, 0.288], [0.089, 0.074, 0.197, 0.212]
    xx = np.arange(4)
    ax[1].bar(xx - 0.18, normal, 0.36, color="0.35", label="normal")
    ax[1].bar(xx + 0.18, pinned, 0.36, color="0.75", label="own fact never forgotten")
    ax[1].set_xticks(xx)
    ax[1].set_xticklabels(cells, rotation=0)
    ax[1].set_ylabel("$G$ ($q=6$)")
    ax[1].set_title("(b) no agent fact can go extinct: $G$ collapses")
    ax[1].legend(frameon=False)
    fig.tight_layout()
    save(fig, "fig4_diagnostics.pdf")


def fig_models_task003(G, D):
    """Fig. 5: the four models against the task003 data, G(q=6) and D(3->6)."""
    models = {t: model_G(t) for t in ("M1", "M2", "M3", "M4")}
    fig, ax = plt.subplots(2, 2, figsize=(W2, 4.0), sharex=True)
    for j, (s, lab) in enumerate(SETUPS):
        g = G[(G.setup == s) & (G.q == 6)].sort_values("rho")
        ax[0, j].errorbar(g.rho, g.G_vote, yerr=[g.G_vote - g.G_vote_lo, g.G_vote_hi - g.G_vote], fmt="o",
                          ms=3.5, color=COL["data"], capsize=1.5, label="simulation", zorder=5)
        d = D[(D.setup == s) & (D.step == "3->6")].sort_values("rho")
        ax[1, j].errorbar(d.rho, d.D_vote, yerr=[d.D_vote - d.D_vote_lo, d.D_vote_hi - d.D_vote], fmt="o",
                          ms=3.5, color=COL["data"], capsize=1.5, label="simulation", zorder=5)
        for t, m in models.items():
            m = m[m.setup == s].pivot_table(index="rho", columns="q", values="G")
            ls = "--" if t in ("M1", "M2") else "-"
            ax[0, j].plot(m.index, m[6], ls, color=COL[t], label=t)
            ax[1, j].plot(m.index, m[6] - m[3], ls, color=COL[t], label=t)
        for i in (0, 1):
            zero(ax[i, j])
        ax[0, j].set_title(f"task003, {lab}: $G$ at $q=6$")
        ax[1, j].set_title(f"task003, {lab}: $D(3\\to 6)$")
        ax[1, j].set_xlabel(r"memory $\rho$")
        # development cells (rho 0.25, 0.75, 1): mark them
        for i in (0, 1):
            for r in (0.25, 0.75, 1.0):
                ax[i, j].axvline(r, color="#f2e394", lw=5, alpha=0.5, zorder=-1)
    ax[0, 0].set_ylabel("$G$")
    ax[1, 0].set_ylabel("$D$")
    ax[0, 1].legend(frameon=False, ncol=2)
    fig.tight_layout()
    save(fig, "fig5_models_task003.pdf")


def fig_post_stop():
    """Fig. 6: persistence after the controller stops: data and M4 for the four continuations."""
    data = pd.read_parquet(BENCH / "post_stop.parquet")
    m = pd.read_parquet(RUNS / "M4_post_stop.parquet", columns=["setup", "rho", "mode", "arm", "t", "x2"])
    m = m.groupby(["setup", "rho", "mode", "t", "arm"]).x2.mean().unstack("arm")
    m = (m["false"] - m["silent"]).rename("delta").reset_index()
    MC = {"normal": "#222222", "frozen_board": "#4292c6", "no_comm": "#e6a532", "purge_controller_facts": "#d1495b"}
    ML = {"normal": "normal", "frozen_board": "frozen board", "no_comm": "no communication",
          "purge_controller_facts": "controller facts removed"}
    fig, ax = plt.subplots(2, 2, figsize=(W2, 3.9), sharex=True)
    for i, (s, lab) in enumerate(SETUPS):
        for j, rho in enumerate((0.25, 0.75)):
            a = ax[i, j]
            for mode in MC:
                x = data[(data.setup == s) & (data.rho == rho) & (data["mode"] == mode) & (data.t >= 20)].sort_values("t")
                a.plot(x.t, x.delta, color=MC[mode], lw=1.4, label=ML[mode])
                a.fill_between(x.t, x.delta - 1.96 * x.se, x.delta + 1.96 * x.se, color=MC[mode], alpha=0.15, lw=0)
                y = m[(m.setup == s) & (m.rho == rho) & (m["mode"] == mode) & (m.t >= 20)].sort_values("t")
                y = y.assign(sm=y.delta.rolling(8, center=True, min_periods=1).mean())
                a.plot(y.t, y.sm, color=MC[mode], lw=1.0, ls=(0, (3, 1.5)))
            a.axvline(30, color="0.5", lw=0.6, ls=":")
            zero(a)
            a.set_xscale("function", functions=(lambda t: np.where(t < 30, t, 30 + 8 * np.log1p(np.maximum(t - 30, 0) / 3)),
                                                lambda u: np.where(u < 30, u, 30 + 3 * np.expm1(np.maximum(u - 30, 0) / 8))))
            a.set_xticks([20, 30, 31, 35, 40, 60, 130])
            a.set_title(f"{lab}, $\\rho={rho}$")
            if j == 0:
                a.set_ylabel(r"$\delta(t)$")
            if i == 1:
                a.set_xlabel("time $t$ (controller stops at 30; stretched axis after 30)")
    h = [plt.Line2D([], [], color=MC[m], lw=1.6) for m in MC]
    h += [plt.Line2D([], [], color="0.3", lw=1.4), plt.Line2D([], [], color="0.3", lw=1.0, ls=(0, (3, 1.5)))]
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.legend(h, [ML[m] for m in MC] + ["simulation (95% band)", "M4"], frameon=False, ncol=6, loc="upper center",
               fontsize=6.5, handlelength=1.8, columnspacing=1.2)
    save(fig, "fig6_post_stop.pdf")


def fig_scaling_K():
    """Fig. 7: K independent projects: data and M4 (post-hoc)."""
    d = pd.read_csv(KP / "tables" / "scaling_K_models.csv")
    d = d[d.model == "M4"]
    se = pd.read_csv(KP / "tables" / "scaling_K.csv")
    d = d.merge(se[["K", "setup", "rho", "G_se_q6", "D_3_6_se"]], on=["K", "setup", "rho"], how="left")
    KC = {1: "#9ecae1", 2: "#4292c6", 4: "#08306b"}
    fig, ax = plt.subplots(2, 2, figsize=(W2, 3.8), sharex=True)
    for j, (s, lab) in enumerate(SETUPS):
        for K in (1, 2, 4):
            x = d[(d.setup == s) & (d.K == K)].sort_values("rho")
            ax[0, j].errorbar(x.rho, x.G6_data, yerr=1.96 * x.G_se_q6, fmt="o", ms=3.5, color=KC[K], capsize=1.5)
            ax[0, j].plot(x.rho, x.G6_model, "-", color=KC[K], lw=1.0)
            ax[1, j].errorbar(x.rho, x.D_data, yerr=1.96 * x.D_3_6_se, fmt="o", ms=3.5, color=KC[K], capsize=1.5)
            ax[1, j].plot(x.rho, x.D_model, "-", color=KC[K], lw=1.0)
        for i in (0, 1):
            zero(ax[i, j])
        ax[0, j].set_title(f"{lab}: $G$ at $q=6$")
        ax[1, j].set_title(f"{lab}: $D(3\\to 6)$")
        ax[1, j].set_xlabel(r"memory $\rho$")
    ax[0, 0].set_ylabel("$G$")
    ax[1, 0].set_ylabel("$D$")
    h = [plt.Line2D([], [], color=KC[K], lw=4) for K in (1, 2, 4)]
    h += [plt.Line2D([], [], color="0.3", marker="o", ls="", ms=3.5), plt.Line2D([], [], color="0.3", lw=1.0)]
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.legend(h, ["$K=1$", "$K=2$", "$K=4$", "simulation", "M4"], frameon=False, ncol=5, loc="upper center")
    save(fig, "fig7_scaling_K.pdf")


def fig_op():
    """Fig. 8: Object Placements, pre-registered M3 and M4 against the simulation."""
    d = pd.read_csv(OPA / "op_data.csv", dtype={"sign": str})
    pr = {t: pd.read_csv(OPA / f"preregistered_{t}.csv") for t in ("M3", "M4")}
    fig, ax = plt.subplots(2, 2, figsize=(W2, 3.8), sharex=True)
    for j, (s, lab) in enumerate((("op_symmetric", "symmetric"), ("op_nosolution", "no-proof"))):
        for i, (qty, step, title) in enumerate((("G", "q6", "$G$ at $q=6$"), ("D", "3->6", "$D(3\\to 6)$"))):
            x = d[(d.setup == s) & (d.quantity == qty) & (d.step == step)].sort_values("rho")
            ax[i, j].errorbar(x.rho, x.value, yerr=1.96 * x.se, fmt="o", ms=3.5, color=COL["data"], capsize=1.5,
                              label="simulation (opened after)", zorder=5)
            for t, p in pr.items():
                y = p[(p.setup == s) & (p.quantity == qty) & (p.step == step)].sort_values("rho")
                ax[i, j].plot(y.rho, y.value, "-", color=COL[t], label=f"{t} (committed before)")
            zero(ax[i, j])
            ax[i, j].set_title(f"Object Placements, {lab}: {title}")
        ax[1, j].set_xlabel(r"memory $\rho$")
    ax[0, 0].set_ylabel("$G$")
    ax[1, 0].set_ylabel("$D$")
    h, lab = ax[0, 0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.legend(h, lab, frameon=False, ncol=3, loc="upper center")
    save(fig, "fig8_object_placements.pdf")


def fig_scoreboard():
    """Fig. 9: one summary of every test: share of resolved increment signs right, and mean |G error|."""
    rows = []
    for t in ("M1", "M2", "M3", "M4"):
        s = pd.read_csv(KP / "held_out_validation" / f"scorecard_{t}_summary.csv")
        s = s[s.split == "held_out"]
        for _, r in s.iterrows():
            rows.append(dict(model=t, test=f"task003\n{'symmetric' if r.setup == 'symmetric' else 'no-proof'}",
                             right=r.correct / r.resolved, err=r.G_mae))
    k = pd.read_csv(KP / "tables" / "scaling_K_models.csv")
    for (t, K), x in k[k.K > 1].groupby(["model", "K"]):
        res = x[x.data_sign != "0"]
        err = pd.concat([(x.G3_model - x.G3_data).abs(), (x.G6_model - x.G6_data).abs()]).mean()
        rows.append(dict(model=t, test=f"task003\n$K={K}$", right=(res.data_sign == res.model_sign).mean(), err=err))
    for t in ("M3", "M4"):
        s = pd.read_csv(OPA / f"scorecard_op_{t}.csv")
        s = s[s.resolved.notna()]
        for _, r in s.iterrows():
            rows.append(dict(model=t, test=f"Obj. Plac.\n{'symmetric' if 'symm' in r.setup else 'no-proof'}",
                             right=r.correct / r.resolved, err=r.G_mae))
    r = pd.DataFrame(rows)
    tests = list(dict.fromkeys(r.test))
    fig, ax = plt.subplots(2, 1, figsize=(W2, 3.8), sharex=True)
    xx = np.arange(len(tests))
    for i, t in enumerate(("M1", "M2", "M3", "M4")):
        y = r[r.model == t].set_index("test").reindex(tests)
        ax[0].bar(xx + (i - 1.5) * 0.2, y.right, 0.2, color=COL[t], label=t)
        ax[1].bar(xx + (i - 1.5) * 0.2, y.err, 0.2, color=COL[t], label=t)
    ax[1].set_xticks(xx)
    ax[1].set_xticklabels(tests, fontsize=7)
    for a in ax:
        for x0 in (1.5, 3.5):
            a.axvline(x0, color="0.85", lw=0.8)
    ax[0].axhline(0.5, color="0.5", ls=":", lw=0.8)
    ax[0].set_ylabel("resolved signs of $D$ right")
    ax[0].set_ylim(0, 1.05)
    ax[0].set_title("(a) direction of the reading effect (held-out | post-hoc | pre-registered)")
    ax[1].set_ylabel("mean $|G_{\\rm model}-G_{\\rm sim}|$")
    ax[1].set_title("(b) size of the controller effect")
    ax[0].legend(frameon=False, ncol=4, loc="upper left", fontsize=6.5)
    fig.tight_layout()
    save(fig, "fig9_scoreboard.pdf")
    r.round(3).to_csv(OUT / "fig9_scoreboard.csv", index=False)


def main():
    G, D = bench()
    fig_data_task003(G, D)
    fig_diagnostics()
    fig_models_task003(G, D)
    fig_post_stop()
    fig_scaling_K()
    fig_op()
    fig_scoreboard()


if __name__ == "__main__":
    main()
