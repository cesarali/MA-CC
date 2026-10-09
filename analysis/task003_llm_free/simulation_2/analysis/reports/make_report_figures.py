"""Figures for the two short Simulation 2 reports (report1_symmetric, report2_nosolution).

Reads the analysis tables in results/simulation_2/analysis/2026-10-09/tables/ and the run data,
writes PDF figures into each report's figures/ folder, and copies two phase diagrams per report.

Usage, from the repository root:
    .venv/bin/python analysis/task003_llm_free/simulation_2/analysis/reports/make_report_figures.py
"""
from __future__ import annotations
import pathlib, shutil, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import RESULTS, SIM1_RESULTS  # noqa: E402

A = RESULTS / "analysis" / "2026-10-09"
T = A / "tables"
GRID = RESULTS / "2026-10-09_sim2_grid" / "cells"
SIM1 = SIM1_RESULTS / "2026-10-07_sim1_pm" / "cells"
OUT = {"symmetric": HERE / "report1_symmetric" / "figures", "nosolution": HERE / "report2_nosolution" / "figures"}
RATES = [0.5, 1.0, 2.0, 4.0, 8.0]
RC = dict(zip(RATES, plt.cm.plasma(np.linspace(0, .85, 5))))
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "legend.fontsize": 7, "figure.dpi": 150})


def setup_full(s):
    return f"task003_{s}"


def gains_over_time(s: str):
    tr = pd.read_csv(T / "gain_trajectories_main.csv")
    tr = tr[tr.setup == setup_full(s)]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.1), sharex=True)
    for j, (arm, gate) in enumerate([(a, g) for a in ("truth", "false") for g in ("on", "off")]):
        ax = axes[j]
        for lc in RATES:
            x = tr[(tr.arm == arm) & (tr.gate == gate) & (tr.lambda_c == lc)]
            ax.plot(x.t, x["mean"], color=RC[lc], lw=1, label=f"{lc:g}")
            ax.fill_between(x.t, x["mean"] - 1.96 * x.se, x["mean"] + 1.96 * x.se, color=RC[lc], alpha=.2, lw=0)
        ax.axvline(30, color="0.5", ls="--", lw=0.7)
        ax.set_title(f"target {'A0' if arm == 'truth' else 'A2'}, gate {gate}")
        ax.set_xlabel("time $t$")
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("gain $G(t)$")
    axes[0].legend(title="$\\lambda_c$", fontsize=6, title_fontsize=6, ncol=1, loc="lower right")
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT[s] / "fig_gain_time.pdf")
    plt.close(fig)


def rate_curves(s: str, rhos=(0.75,)):
    g = pd.read_csv(T / "gains.csv")
    g = g[(g.setup == setup_full(s)) & (g.q == 6) & (g.qc == 12) & (g.stop == "on")]
    q1 = pd.read_csv(T / "q1_gains.csv")
    q1 = q1[(q1.setup == setup_full(s)) & (q1.q == 6) & (q1.qc == 12)]
    fig, axes = plt.subplots(1, 2 * len(rhos), figsize=(3.6 * len(rhos) + 0.2, 2.3))
    axes = np.atleast_1d(axes)
    for i, rho in enumerate(rhos):
        ax, ax2 = axes[2 * i], axes[2 * i + 1]
        for arm, col in (("truth", "tab:blue"), ("false", "tab:red")):
            for gate, ls, mk in (("on", "-", "o"), ("off", "--", "s")):
                x = g[(g.arm == arm) & (g.gate == gate) & (g.rho == rho)].sort_values("lambda_c")
                lab = f"target {'A0' if arm == 'truth' else 'A2'}, gate {gate}"
                ax.errorbar(x.lambda_c, x.G30, yerr=[x.G30 - x.G30_lo, x.G30_hi - x.G30], color=col, ls=ls,
                            marker=mk, ms=3, capsize=1.5, lw=1, label=lab)
                ax2.plot(x.lambda_c, x.messages, color=col, ls=ls, marker=mk, ms=3, lw=1)
            ref = q1[(q1.arm == arm) & (q1.rho == rho)]
            if len(ref):
                ax.axhline(ref.G30_sim1.iloc[0], color=col, lw=0.8, ls=":")
        ax.set_xscale("log", base=2)
        ax2.set_xscale("log", base=2)
        ax.set_xticks(RATES, [f"{r:g}" for r in RATES])
        ax2.set_xticks(RATES, [f"{r:g}" for r in RATES])
        ax.set_xlabel("$\\lambda_c$")
        ax2.set_xlabel("$\\lambda_c$")
        ax.set_title(f"gain $G(30)$, $\\rho = {rho:g}$")
        ax2.set_title(f"messages, $\\rho = {rho:g}$")
        ax2.set_ylim(0, 31)
        ax.set_ylim(bottom=0)
    axes[0].legend(fontsize=5.5, loc="lower left")
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT[s] / "fig_rate.pdf")
    plt.close(fig)


def silent_vs_sim1():
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.1), sharey=True)
    for j, q in enumerate((3, 6, 12)):
        ax = axes[j]
        s = pd.read_parquet(GRID / f"task003_symmetric__silent__q{q}__rho0.75" / "snapshots.parquet",
                            columns=["t", "mean_p_A0", "mean_p_A2"]).groupby("t").mean()
        s1 = pd.read_parquet(SIM1 / f"task003_symmetric__silent__q{q}__rho0.75" / "days.parquet").groupby("day")[["mean_p_A0", "mean_p_A2"]].mean()
        for k, col in (("A0", "tab:blue"), ("A2", "tab:red")):
            ax.plot(s.index, s[f"mean_p_{k}"], color=col, lw=1.1, label=f"core, $m_{{{k}}}$")
            ax.plot(s1.index + 1, s1[f"mean_p_{k}"], color=col, lw=0, marker="o", ms=1.8, label=f"Simulation 1, $m_{{{k}}}$")
        ax.set_title(f"silent, $q = {q}$, $\\rho = 0.75$")
        ax.set_xlabel("time $t$ (Simulation 1: day)")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("mean belief")
    axes[0].legend(fontsize=5.5, loc="center right")
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT["symmetric"] / "fig_silent.pdf")
    plt.close(fig)


def own_proof():
    m = pd.read_csv(T / "q8_mechanisms.csv")
    m = m[m.setup == "task003_symmetric"]
    sil = m[m.arm == "silent"].iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.1), sharey=True)
    for j, arm in enumerate(("truth", "false")):
        ax = axes[j]
        for gate, ls in (("on", "-"), ("off", "--")):
            x = m[(m.arm == arm) & (m.gate == gate) & (m.stop == "on")].sort_values("lambda_c")
            ax.plot(x.lambda_c, x.proof_rate_A0_t30, color="k", ls=ls, marker="o", ms=3, lw=1, label=f"from all facts, gate {gate}")
            ax.plot(x.lambda_c, x.proof_rate_A0_own_evidence_t30, color="tab:green", ls=ls, marker="s", ms=3, lw=1,
                    label=f"from own evidence, gate {gate}")
        ax.axhline(sil.proof_rate_A0_t30, color="0.5", lw=0.8, ls=":", label="silent (both)")
        ax.set_xscale("log", base=2)
        ax.set_xticks(RATES, [f"{r:g}" for r in RATES])
        ax.set_xlabel("$\\lambda_c$")
        ax.set_title(f"target {'A0' if arm == 'truth' else 'A2'}")
    axes[0].set_ylabel("share of agents proving A0, $t = 30$")
    axes[1].legend(fontsize=5.5)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT["symmetric"] / "fig_own_proof.pdf")
    plt.close(fig)


def distinct_facts():
    """Nosolution: G(30) against distinct pool facts posted per episode, every q, qc, rate, gate."""
    g = pd.read_csv(T / "gains.csv")
    g = g[(g.setup == "task003_nosolution") & (g.stop == "on")].copy()
    vals = []
    for r in g.itertuples():
        c = pd.read_parquet(GRID / r.folder / "controller.parquet", columns=["episode", "posted_fact"])
        vals.append(c[c.posted_fact >= 0].groupby("episode").posted_fact.nunique().reindex(range(1000), fill_value=0).mean())
    g["distinct_facts"] = vals
    g[["cell", "arm", "q", "qc", "rho", "lambda_c", "gate", "G30", "messages", "distinct_facts"]].to_csv(
        T / "report2_distinct_facts.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.3), sharex=True)
    for j, rho in enumerate((0.75, 1.0)):
        ax = axes[j]
        for arm, col in (("truth", "tab:blue"), ("false", "tab:red")):
            x = g[(g.arm == arm) & (g.rho == rho)]
            ax.scatter(x.distinct_facts, x.G30, s=5, color=col, alpha=.6,
                       label=f"target {'A0' if arm == 'truth' else 'A2'}")
            if len(x) > 2:
                b = np.polyfit(x.distinct_facts, x.G30, 1)
                xs = np.linspace(x.distinct_facts.min(), x.distinct_facts.max(), 10)
                ax.plot(xs, np.polyval(b, xs), color=col, lw=0.8)
        ax.set_title(f"$\\rho = {rho:g}$")
        ax.set_xlabel("distinct pool facts posted per episode")
    axes[0].set_ylabel("gain $G(30)$")
    axes[0].legend(fontsize=6)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT["nosolution"] / "fig_distinct.pdf")
    plt.close(fig)
    return g


def sensing(s: str):
    q7 = pd.read_csv(T / "q7_sensing.csv")
    q7 = q7[(q7.setup == setup_full(s)) & (q7.q == 6) & (q7.rho == 0.75) & (q7.stop == "on")]
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.1), sharey=True)
    for j, arm in enumerate(("truth", "false")):
        ax = axes[j]
        for qc, col in zip((3, 6, 12, 24), plt.cm.viridis(np.linspace(0, .85, 4))):
            for gate, ls in (("on", "-"), ("off", "--")):
                x = q7[(q7.arm == arm) & (q7.qc == qc) & (q7.gate == gate)].sort_values("lambda_c")
                ax.errorbar(x.lambda_c, x.error_mean, yerr=1.96 * x.error_se, color=col, ls=ls, marker="o", ms=2.5,
                            lw=0.9, capsize=1.5, label=f"$q_c$ {qc}, gate {gate}" if j == 0 else None)
        ax.axhline(0, color="0.6", lw=0.7)
        ax.set_xscale("log", base=2)
        ax.set_xticks(RATES, [f"{r:g}" for r in RATES])
        ax.set_xlabel("$\\lambda_c$")
        ax.set_title(f"target {'A0' if arm == 'truth' else 'A2'}")
    axes[0].set_ylabel("$\\hat v$ $-$ true current share")
    axes[0].legend(fontsize=5, ncol=2)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT[s] / "fig_sensing.pdf")
    plt.close(fig)


def main():
    for d in OUT.values():
        d.mkdir(parents=True, exist_ok=True)
    for s in ("symmetric", "nosolution"):
        gains_over_time(s)
        sensing(s)
        for arm in ("truth", "false"):
            shutil.copy(A / "figures" / f"F1_G30_{s}_{arm}_gateon.png", OUT[s] / f"phase_{arm}_gateon.png")
    rate_curves("symmetric", rhos=(0.75,))
    rate_curves("nosolution", rhos=(0.75, 1.0))
    silent_vs_sim1()
    own_proof()
    distinct_facts()
    print("figures written to", *OUT.values())


if __name__ == "__main__":
    main()
