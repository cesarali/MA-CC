"""Compact figures for the two short Simulation 1 reports (report1_symmetric, report2_nosolution).

Reads the analysis tables in results/simulation_1/analysis/2026-10-07/tables/ and the run data, and
writes vector PDFs into each report's figures/ folder. The phase diagrams are copied
from the analysis figures.

From the repository root:
    .venv/bin/python analysis/task003_llm_free/simulation_1/analysis/reports/make_report_figures.py
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
from common import RESULTS, VARIANT_COLOR, cells_table  # noqa: E402

A = RESULTS / "analysis" / "2026-10-07"
R1, R2 = HERE / "report1_symmetric" / "figures", HERE / "report2_nosolution" / "figures"
for d in (R1, R2):
    d.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "legend.fontsize": 7, "figure.dpi": 150})
LABEL = {"sim1_base": "argmax + supporting posts", "sim1_pm": "probability matching",
         "sim1_uniform_post": "argmax + uniform posts"}
T2 = pd.read_parquet(A / "tables" / "q2_time.parquet")
T5 = pd.read_parquet(A / "tables" / "q5_time.parquet")
S4 = pd.read_csv(A / "tables" / "q4_sensing.csv")
C = cells_table()


def share(variant, setup, arm, a, q, qc, b, rho):
    """Mean vote share of allocation a per day under `arm` (silent: from the silent twin)."""
    base = T2[(T2.variant == variant) & (T2.setup == setup) & (T2.q == q) & (T2.qc == qc) & (T2.b == b) & (T2.rho == rho)]
    sil = base[(base.quantity == "share_silent") & (base.a == a)].sort_values("day")
    if arm == "silent":
        return sil.day.to_numpy(), sil["mean"].to_numpy(), None, None
    g = base[(base.quantity == "G") & (base.a == a) & (base.arm == arm)].sort_values("day")
    return g.day.to_numpy(), sil["mean"].to_numpy() + g["mean"].to_numpy(), \
        sil["mean"].to_numpy() + g.lo.to_numpy(), sil["mean"].to_numpy() + g.hi.to_numpy()


def days_mean(path, col):
    d = pd.read_parquet(pathlib.Path(path) / "days.parquet", columns=["day", col])
    return d.groupby("day")[col].mean().to_numpy()


PREVIEW = None          # set to a folder to also write PNG previews (not used by the reports)


def save(fig, folder, name, layout=True):
    if layout:
        fig.tight_layout()
    fig.savefig(folder / name)
    if PREVIEW:
        fig.savefig(pathlib.Path(PREVIEW) / f"{folder.parent.name}_{name.replace('.pdf', '.png')}", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- report 1: symmetric
def r1_a2_share_time():
    """A2 share over days: silent vs false controller, three variants, rho 0.75 and 1."""
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.1), sharey=True)
    for ax, rho in zip(axes, (0.75, 1.0)):
        for v in ("sim1_base", "sim1_pm", "sim1_uniform_post"):
            day, m, lo, hi = share(v, "task003_symmetric", "false", 2, 6, 12, 3, rho)
            ax.plot(day, m, color=VARIANT_COLOR[v], lw=1.6, label=f"false controller, {LABEL[v]}")
            ax.fill_between(day, lo, hi, color=VARIANT_COLOR[v], alpha=0.2, lw=0)
            day, s, _, _ = share(v, "task003_symmetric", "silent", 2, 6, 12, 3, rho)
            ax.plot(day, s, color=VARIANT_COLOR[v], lw=1, ls=":")
        ax.set_title(f"ρ = {rho}" + ("  (agents forget)" if rho < 1 else "  (full memory)"))
        ax.set_xlabel("day")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("share of agents voting A2")
    axes[0].plot([], [], color="k", ls=":", lw=1, label="silent (same variant)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    save(fig, R1, "fig_a2_share_time.pdf", layout=False)


def r1_proof_time():
    """Share of agents whose memory proves A0, silent vs false controller (base, q=6, qc=12, b=3)."""
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.1), sharey=True)
    main = C[(C.purpose == "main") & (C.variant == "sim1_base") & (C.setup == "task003_symmetric") & (C.q == 6)]
    for ax, rho in zip(axes, (0.75, 1.0)):
        sil = main[(main.arm == "silent") & (main.rho == rho)].path.iloc[0]
        fal = main[(main.arm == "false") & (main.qc == 12) & (main.b == 3) & (main.rho == rho)].path.iloc[0]
        days = np.arange(1, 31)
        ax.plot(days, days_mean(sil, "share_A0"), color="#777777", lw=1, ls="--", label="silent: votes A0")
        ax.plot(days, days_mean(sil, "proof_rate_A0"), color="#777777", lw=1.6, label="silent: holds a proof of A0")
        ax.plot(days, days_mean(fal, "share_A0"), color="#d1495b", lw=1, ls="--", label="false ctrl: votes A0")
        ax.plot(days, days_mean(fal, "proof_rate_A0"), color="#d1495b", lw=1.6,
                label="false ctrl: proof of A0, any facts")
        ax.plot(days, days_mean(fal, "proof_rate_A0_own_evidence"), color="#d1495b", lw=1.6, ls=":",
                label="false ctrl: proof of A0, own evidence only")
        ax.set_title(f"ρ = {rho}")
        ax.set_xlabel("day")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("share of agents")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    save(fig, R1, "fig_proof_time.pdf", layout=False)


def r1_sensing():
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.4))
    qcs = (3, 6, 12, 24)
    for v in ("sim1_base", "sim1_pm", "sim1_uniform_post"):
        s = S4[(S4.variant == v) & (S4.setup == "task003_symmetric")].groupby("qc")
        axes[0].plot(range(4), s.rmse.mean().reindex(qcs), "o-", color=VARIANT_COLOR[v], label=LABEL[v])
        axes[1].plot(range(4), 100 * s.wrongly_silent.mean().reindex(qcs), "o-", color=VARIANT_COLOR[v])
        axes[1].plot(range(4), 100 * s.wrongly_acting.mean().reindex(qcs), "s--", color=VARIANT_COLOR[v], mfc="white")
    for ax in axes:
        ax.set_xticks(range(4), [str(q) for q in qcs])
        ax.set_xlabel("qc (posts the controller reads per night)")
    axes[0].set_ylabel("error of v̂ (root mean square)")
    axes[0].legend(frameon=False)
    axes[1].set_ylabel("% of nights")
    axes[1].plot([], [], "ko-", label="silent although true share < 0.75")
    axes[1].plot([], [], "ks--", mfc="white", label="posting although true share ≥ 0.75")
    axes[1].legend(frameon=False)
    save(fig, R1, "fig_sensing.pdf")


def r1_own_proof():
    """Rule off minus rule on: proof from all facts vs from own evidence, over days."""
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.5), sharey=True)
    for ax, v in zip(axes, ("sim1_base", "sim1_pm")):
        for k, color, label in (("proof_rate_A0", "#1b6ca8", "holds a proof (any facts)"),
                                ("proof_rate_A0_own_evidence", "#d1495b", "holds a proof from own evidence")):
            t = T5[(T5.variant == v) & (T5.q == 6) & (T5.qc == 12) & (T5.b == 1) & (T5.rho == 0.75) & (T5.quantity == k)]
            ax.plot(t.day, t["mean"], color=color, lw=1.6, label=label)
            ax.fill_between(t.day, t.lo, t.hi, color=color, alpha=0.2, lw=0)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(f"{LABEL[v]}, ρ = 0.75")
        ax.set_xlabel("day")
    axes[0].set_ylabel("rule OFF − rule ON (share of agents)")
    axes[0].legend(frameon=False, loc="lower left")
    save(fig, R1, "fig_own_proof.pdf")


# ---------------------------------------------------------------- report 2: no-solution
def r2_drift():
    """Silent arm: A2 - A0 margin over days (variants), and its spread across episodes on day 30."""
    q1 = pd.read_csv(A / "tables" / "q1_trajectories.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), gridspec_kw={"width_ratios": [1, 1, 1.1]})
    for ax, rho in zip(axes[:2], (0.75, 1.0)):
        for v in ("sim1_pm", "sim1_base", "sim1_uniform_post"):
            t = q1[(q1.variant == v) & (q1.setup == "task003_nosolution") & (q1.rho == rho) & (q1.q == 6)
                   & (q1.quantity == "margin_A2_minus_A0")]
            ax.plot(t.day, t["mean"], color=VARIANT_COLOR[v], lw=1.6, label=LABEL[v])
            ax.fill_between(t.day, t.lo, t.hi, color=VARIANT_COLOR[v], alpha=0.2, lw=0)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(f"silent, ρ = {rho}, q = 6")
        ax.set_xlabel("day")
        ax.set_ylim(-0.35, 0.6)
    axes[0].set_ylabel("share A2 − share A0")
    axes[0].legend(frameon=False, loc="upper left")
    main = C[(C.purpose == "main") & (C.setup == "task003_nosolution") & (C.arm == "silent") & (C.q == 6) & (C.rho == 1.0)]
    bins = np.arange(-24.5, 25.5, 1)
    for v in ("sim1_pm", "sim1_base"):
        d = pd.read_parquet(pathlib.Path(main[main.variant == v].path.iloc[0]) / "days.parquet",
                            columns=["day", "share_A0", "share_A2"])
        last = d[d.day == 29]
        axes[2].hist(np.rint(24 * (last.share_A2 - last.share_A0)), bins=bins, color=VARIANT_COLOR[v], alpha=0.65,
                     label=LABEL[v], density=True)
    axes[2].set_title("silent, ρ = 1, q = 6: day 30")
    axes[2].set_xlabel("n(A2) − n(A0), agents")
    axes[2].set_ylabel("share of episodes")
    axes[2].legend(frameon=False, loc="upper center")
    save(fig, R2, "fig_drift.pdf")


def r2_gain_time():
    """Probability matching: own-target paired gain over days by b (q = 6, qc = 12)."""
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.6), sharey=True)
    colors = dict(zip((1, 2, 3, 6, 9), plt.cm.viridis(np.linspace(0, 0.9, 5))))
    for ax, arm in zip(axes, ("truth", "false")):
        a = 0 if arm == "truth" else 2
        for b in (1, 3, 6, 9):
            for rho, ls in ((0.75, "-"), (1.0, "--")):
                t = T2[(T2.variant == "sim1_pm") & (T2.setup == "task003_nosolution") & (T2.arm == arm) & (T2.quantity == "G")
                       & (T2.a == a) & (T2.q == 6) & (T2.qc == 12) & (T2.b == b) & (T2.rho == rho)]
                ax.plot(t.day, t["mean"], color=colors[b], ls=ls, lw=1.4, label=f"b = {b}" if rho == 0.75 else None)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(f"{arm} controller (target A{a}), probability matching")
        ax.set_xlabel("day")
    axes[0].set_ylabel(f"paired gain in target's share")
    axes[0].plot([], [], "k-", label="ρ = 0.75")
    axes[0].plot([], [], "k--", label="ρ = 1")
    axes[0].legend(frameon=False, ncol=2)
    save(fig, R2, "fig_gain_time.pdf")


def r2_sensing():
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    qcs = (3, 6, 12, 24)
    for v in ("sim1_pm", "sim1_base"):
        s = S4[(S4.variant == v) & (S4.setup == "task003_nosolution")].groupby("qc")
        ax.plot(range(4), s.rmse.mean().reindex(qcs), "o-", color=VARIANT_COLOR[v], label=LABEL[v])
    ax.set_xticks(range(4), [str(q) for q in qcs])
    ax.set_xlabel("qc")
    ax.set_ylabel("error of v̂ (rms)")
    ax.legend(frameon=False)
    save(fig, R2, "fig_sensing.pdf")


def copy_phase():
    src = A / "figures"
    shutil.copy(src / "q2_phase_own_gain_30_task003_symmetric_sim1_base_false.png", R1 / "phase_false_base.png")
    shutil.copy(src / "q2_phase_own_gain_30_task003_nosolution_sim1_pm_false.png", R2 / "phase_false_pm.png")
    shutil.copy(src / "q2_phase_own_gain_30_task003_nosolution_sim1_pm_truth.png", R2 / "phase_truth_pm.png")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--preview":
        PREVIEW = sys.argv[2]
        pathlib.Path(PREVIEW).mkdir(parents=True, exist_ok=True)
    r1_a2_share_time(); r1_proof_time(); r1_sensing(); r1_own_proof()
    r2_drift(); r2_gain_time(); r2_sensing(); copy_phase()
    print("figures written to", R1, "and", R2)
