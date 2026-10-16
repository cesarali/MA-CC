"""Figures for the four reports of 14 October 2026 (Simulations 1 and 2, task003-symmetric and
task003-nosolution). The reports of 7, 9 and 12 October are left untouched: this script writes only into
the four new folders 2026-10-14_report{1_symmetric,2_nosolution}/figures/ and copies there the figures
it reuses from the 12 October folders.

The two simulations get the same figures, with the same layout and colours (silent grey, target A0 blue,
target A2 red; controller strength b or rate in one colour scale):
  fig_votes_time.pdf     share of agents voting A0 and A2 over time, silent and both controllers, rho 0.75 / 1
  fig_gain_time.pdf      gain over time by b (Simulation 1) or rate (Simulation 2), both targets, rho 0.75 / 1
  fig_phase_gain.pdf     gain at t = 30 over qc (y) and b / rate (x), every q, both targets
  fig_strength.pdf       Simulation 1 only: gain and facts posted against b (Simulation 2: fig_rate.pdf, copied)
  fig_proof_time.pdf     symmetric: share voting A0 and proving A0, silent vs false controller
  fig_eff_time2.pdf      one setting over time: readability I, eta and eta_JS, agents' uncertainty
  fig_eff_phase_I.pdf    readability I (target vs one agent's vote) at t = 30 over qc x b / rate
  fig_success_facts.pdf  change in P(target wins) against facts posted, t = 30 and 5, rho 0.75 / 1
  tab_eff_precision.tex  the 12 October precision rows without progress and focus

Usage, from the repository root (after frontier.py; the efficiency tables of 12 October must exist):
    .venv/bin/python analysis/task003_llm_free/shared_analysis/efficiency_estimates/make_2026-10-14_report_figures.py
"""
from __future__ import annotations
import pathlib, shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from frontier import hull  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RES = ROOT / "results"
SIM1_CELLS = RES / "simulation_1" / "2026-10-07_sim1_pm" / "cells"
SIM2_CELLS = RES / "simulation_2" / "2026-10-09_sim2_grid" / "cells"
SIM1_TIME = RES / "simulation_1" / "analysis" / "2026-10-07" / "tables" / "q2_time.parquet"
SIM2_GAINS = RES / "simulation_2" / "analysis" / "2026-10-09" / "tables" / "gains.csv"
EFF = RES / "efficiency_estimates" / "2026-10-12"
FRONTIER = RES / "efficiency_estimates" / "2026-10-13" / "frontier_points.csv"
REP = {"sim1": ROOT / "simulation_1" / "analysis" / "reports", "sim2": ROOT / "simulation_2" / "analysis" / "reports"}
NAME = {"symmetric": "report1_symmetric", "nosolution": "report2_nosolution"}

GREY, BLUE, RED = "0.45", "tab:blue", "tab:red"
TCOL = {"silent": GREY, "truth": BLUE, "false": RED}
TLAB = {"truth": "target A0", "false": "target A2"}
XS = {"sim1": [1, 2, 3, 6, 9], "sim2": [0.5, 1.0, 2.0, 4.0, 8.0]}
XLAB = {"sim1": "facts per night $b$", "sim2": "controller rate $\\lambda_c$"}
TLABEL = {"sim1": "day", "sim2": "time $t$"}
QCS = [3, 6, 12, 24]
STRENGTH_COL = dict(zip(range(5), plt.cm.plasma(np.linspace(0, .85, 5))))
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "legend.fontsize": 6.5, "figure.dpi": 150})


def out(sim, setup, date="2026-10-14"):
    d = REP[sim] / f"{date}_{NAME[setup]}" / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ------------------------------------------------------------------ raw per-episode series
def sim1_cell(setup, arm, rho, q=6, qc=12, b=3):
    name = f"task003_{setup}__silent__q{q}__rho{rho:g}" if arm == "silent" else \
        f"task003_{setup}__{arm}__q{q}__qc{qc}__b{b}__rho{rho:g}"
    if rho == 1.0:
        name = name.replace("rho1", "rho1.0")
    d = pd.read_parquet(SIM1_CELLS / name / "days.parquet")
    d["t"] = d.day + 1                                  # end of day d+1
    return d


def sim2_cell(setup, arm, rho, q=6, qc=12, lc=1.0, gate="on"):
    name = f"task003_{setup}__silent__q{q}__rho{rho:g}" if arm == "silent" else \
        f"task003_{setup}__{arm}__q{q}__qc{qc}__rho{rho:g}__lc{lc:g}__gate{gate}__stopon"
    return pd.read_parquet(SIM2_CELLS / name / "snapshots.parquet")


def cell(sim, setup, arm, rho, x=None, **kw):
    if sim == "sim1":
        return sim1_cell(setup, arm, rho, b=int(x or 3), **kw)
    return sim2_cell(setup, arm, rho, lc=float(x or 1.0), **kw)


def mean_band(d, col):
    g = d.groupby("t")[col]
    m, se = g.mean(), g.std() / np.sqrt(g.count())
    return m.index.to_numpy(), m.to_numpy(), 1.96 * se.to_numpy()


def paired_gain(c, s, col):
    j = c[["episode", "t", col]].merge(s[["episode", "t", col]], on=["episode", "t"], suffixes=("_c", "_s"))
    j["g"] = j[f"{col}_c"] - j[f"{col}_s"]
    return mean_band(j, "g")


# ------------------------------------------------------------------ figures
def votes_time(sim, setup):
    """2 x 2: share voting A0 (top) and A2 (bottom), rho 0.75 (left) and 1 (right); silent and both arms."""
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.0), sharex=True, sharey=True)
    for j, rho in enumerate((0.75, 1.0)):
        data = {arm: cell(sim, setup, arm, rho) for arm in ("silent", "truth", "false")}
        for i, k in enumerate(("A0", "A2")):
            ax = axes[i, j]
            for arm, d in data.items():
                t, m, h = mean_band(d, f"share_{k}")
                lab = "silent" if arm == "silent" else f"controller, {TLAB[arm]}"
                ax.plot(t, m, color=TCOL[arm], lw=1.2, ls=":" if arm == "silent" else "-", label=lab)
                ax.fill_between(t, m - h, m + h, color=TCOL[arm], alpha=.18, lw=0)
            if sim == "sim2":
                ax.axvline(30, color="0.6", ls="--", lw=0.7)
            ax.set_ylim(0, 1)
            ax.set_title(f"share voting {k}, $\\rho = {rho:g}$" + (" (agents forget)" if rho < 1 else " (full memory)"))
            if i == 1:
                ax.set_xlabel(TLABEL[sim])
            if j == 0:
                ax.set_ylabel(f"share of agents voting {k}")
    axes[0, 0].legend(loc="best")
    fig.tight_layout(pad=0.4)
    fig.savefig(out(sim, setup) / "fig_votes_time.pdf")
    plt.close(fig)


def gain_time(sim, setup):
    """1 x 4: target A0 at rho 0.75 / 1, target A2 at rho 0.75 / 1; one line per b or rate (gate on)."""
    col = "share" if sim == "sim1" else "mean_p"
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.2), sharey=True)
    for j, (arm, rho) in enumerate([(a, r) for a in ("truth", "false") for r in (0.75, 1.0)]):
        ax = axes[j]
        k = "A0" if arm == "truth" else "A2"
        s = cell(sim, setup, "silent", rho)
        for n, x in enumerate(XS[sim]):
            t, m, h = paired_gain(cell(sim, setup, arm, rho, x), s, f"{col}_{k}")
            ax.plot(t, m, color=STRENGTH_COL[n], lw=1, label=f"{x:g}")
            ax.fill_between(t, m - h, m + h, color=STRENGTH_COL[n], alpha=.18, lw=0)
        ax.axhline(0, color="0.6", lw=0.6)
        if sim == "sim2":
            ax.axvline(30, color="0.6", ls="--", lw=0.7)
        ax.set_title(f"{TLAB[arm]}, $\\rho = {rho:g}$")
        ax.set_xlabel(TLABEL[sim])
    axes[0].set_ylabel("gain in target's vote share" if sim == "sim1" else "gain $G(t)$ (mean belief)")
    axes[0].legend(title="$b$" if sim == "sim1" else "$\\lambda_c$", fontsize=5.5, title_fontsize=6, loc="best")
    fig.tight_layout(pad=0.4)
    fig.savefig(out(sim, setup) / "fig_gain_time.pdf")
    plt.close(fig)


def gains_t30(sim, setup):
    """Gain at t = 30 with 95% interval per (arm, q, qc, x, rho); gate on, proof stop on."""
    if sim == "sim1":
        t = pd.read_parquet(SIM1_TIME)
        t = t[(t.variant == "sim1_pm") & (t.setup == f"task003_{setup}") & (t.quantity == "G") & (t.day == 30)]
        t = t[((t.arm == "truth") & (t.a == 0)) | ((t.arm == "false") & (t.a == 2))]
        return pd.DataFrame({"arm": t.arm, "q": t.q, "qc": t.qc.astype(int), "x": t.b.astype(float), "rho": t.rho,
                             "g": t["mean"], "lo": t.lo, "hi": t.hi})
    g = pd.read_csv(SIM2_GAINS)
    g = g[(g.setup == f"task003_{setup}") & (g.gate == "on") & (g.stop == "on") & (g.arm != "silent")]
    return pd.DataFrame({"arm": g.arm, "q": g.q, "qc": g.qc, "x": g.lambda_c, "rho": g.rho,
                         "g": g.G30, "lo": g.G30_lo, "hi": g.G30_hi})


def phase_gain(sim, setup):
    """Rows: rho; columns: target x q; tiles qc (y) x b / rate (x). Grey number: interval includes 0."""
    g = gains_t30(sim, setup)
    rhos = (0.75, 1.0) if setup == "nosolution" else (0.75,)
    cols = [(arm, q) for arm in ("truth", "false") for q in (3, 6, 12)]
    vmax = max(0.05, float(np.nanmax(np.abs(g.g))))
    fig, axes = plt.subplots(len(rhos), 6, figsize=(8.0, 1.5 * len(rhos) + 0.55), squeeze=False)
    xs = XS[sim]
    for i, rho in enumerate(rhos):
        for j, (arm, q) in enumerate(cols):
            ax = axes[i, j]
            m = np.full((4, 5), np.nan)
            ci0 = np.zeros((4, 5), bool)
            for r in g[(g.rho == rho) & (g.arm == arm) & (g.q == q)].itertuples():
                a, b = QCS.index(int(r.qc)), xs.index(float(r.x) if sim == "sim2" else int(r.x))
                m[a, b] = r.g
                ci0[a, b] = r.lo <= 0 <= r.hi
            im = ax.imshow(m, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
            for a in range(4):
                for b in range(5):
                    if np.isfinite(m[a, b]):
                        ax.text(b, a, f"{m[a, b]:.2f}".replace("0.", ".", 1), ha="center", va="center", fontsize=4.9,
                                color="0.55" if ci0[a, b] else ("white" if abs(m[a, b]) > 0.6 * vmax else "black"))
            ax.set_xticks(range(5), [f"{x:g}" for x in xs], fontsize=5)
            ax.set_yticks(range(4), QCS if j == 0 else [], fontsize=5)
            if i == 0:
                ax.set_title(f"{TLAB[arm]}, $q = {q}$", fontsize=6.5)
            if i == len(rhos) - 1:
                ax.set_xlabel("$b$" if sim == "sim1" else "$\\lambda_c$", fontsize=6.5)
            if j == 0:
                ax.set_ylabel(f"$\\rho = {rho:g}$\n$q_c$", fontsize=6.5)
    fig.colorbar(im, ax=axes, shrink=0.85, pad=0.015,
                 label="gain, vote share" if sim == "sim1" else "gain $G(30)$, mean belief")
    fig.savefig(out(sim, setup) / "fig_phase_gain.pdf", bbox_inches="tight")
    plt.close(fig)


def strength_sim1(setup):
    """Simulation 1 analogue of Simulation 2's fig_rate: gain at day 30 and facts posted against b (qc 12, q 6)."""
    g = gains_t30("sim1", setup)
    g = g[(g.q == 6) & (g.qc == 12)]
    f = pd.read_csv(FRONTIER)
    f = f[(f.sim == "sim1") & (f.setup == f"task003_{setup}") & (f.q == 6) & (f.qc == 12) & (f["T"] == 30)
          & (f.rule == "plurality") & (f.outcome == "target") & (f.stop == "on")]
    rhos = (0.75, 1.0) if setup == "nosolution" else (0.75,)
    fig, axes = plt.subplots(1, 2 * len(rhos), figsize=(3.6 * len(rhos) + 0.2, 2.3))
    axes = np.atleast_1d(axes)
    for i, rho in enumerate(rhos):
        ax, ax2 = axes[2 * i], axes[2 * i + 1]
        for arm in ("truth", "false"):
            x = g[(g.arm == arm) & (g.rho == rho)].sort_values("x")
            ax.errorbar(x.x, x.g, yerr=[x.g - x.lo, x.hi - x.g], color=TCOL[arm], marker="o", ms=3, capsize=1.5, lw=1,
                        label=TLAB[arm])
            y = f[(f.arm == arm) & (f.rho == rho)].sort_values("x")
            ax2.plot(y.x, y.facts, color=TCOL[arm], marker="o", ms=3, lw=1)
        for a in (ax, ax2):
            a.set_xticks(XS["sim1"])
            a.set_xlabel("$b$")
        ax.axhline(0, color="0.6", lw=0.6)
        ax.set_title(f"gain at day 30, $\\rho = {rho:g}$")
        ax2.set_title(f"facts posted, $\\rho = {rho:g}$")
        ax2.set_ylim(bottom=0)
    axes[0].legend(fontsize=6, loc="best")
    fig.tight_layout(pad=0.4)
    fig.savefig(out("sim1", setup) / "fig_strength.pdf")
    plt.close(fig)


def proof_time(sim):
    """Symmetric: share voting A0 (dashed), proving A0 from any facts (solid) and from own evidence (dotted)."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
    for ax, rho in zip(axes, (0.75, 1.0)):
        for arm in ("silent", "false"):
            d = cell(sim, "symmetric", arm, rho)
            who = "silent" if arm == "silent" else "false controller"
            for c, ls, lab in (("share_A0", "--", "votes A0"), ("proof_rate_A0", "-", "proves A0, any facts"),
                               ("proof_rate_A0_own_evidence", ":", "proves A0, own evidence only")):
                t, m, _ = mean_band(d, c)
                ax.plot(t, m, color=TCOL[arm], ls=ls, lw=1.3, label=f"{who}: {lab}")
        if sim == "sim2":
            ax.axvline(30, color="0.6", ls="--", lw=0.7)
        ax.set_title(f"$\\rho = {rho:g}$")
        ax.set_xlabel(TLABEL[sim])
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("share of agents")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=6)
    fig.tight_layout(rect=(0, 0.14, 1, 1), pad=0.4)
    fig.savefig(out(sim, "symmetric") / "fig_proof_time.pdf")
    plt.close(fig)


SPEC = {"sim1": "b3", "sim2": "lc1_gateon"}


def eff_time2(e, u, sim, setup):
    """At one setting: readability I (swarm, one agent), pair efficiency, agents' uncertainty, over time."""
    e = e[(e.sim == sim) & (e.setup == f"task003_{setup}") & (e.rho == 0.75) & (e.setting == SPEC[sim])]
    u = u[(u.sim == sim) & (u.setup == f"task003_{setup}") & (u.rho == 0.75) & (u.setting == SPEC[sim])]
    av = e[e.resolution == "agent_votes"].sort_values("t")
    sw = e[e.resolution == "swarm_beliefs_k3"].sort_values("t")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.0))
    for x, c, lab in ((sw, "k", "swarm (3 bins)"), (av, "0.55", "one agent (votes)")):
        axes[0].plot(x.t, x.I, color=c, lw=1, label=lab)
        axes[0].fill_between(x.t, x.I_lo, x.I_hi, color=c, alpha=.2, lw=0)
    axes[0].set_title("readability $I(t)$ (bits)")
    y = av[av.C >= 0.01]
    axes[1].plot(y.t, y.eta, color="tab:green", lw=1, label="$\\eta$, one agent (votes)")
    axes[1].fill_between(y.t, y.eta_lo, y.eta_hi, color="tab:green", alpha=.2, lw=0)
    axes[1].plot(sw.t, sw.eta_JS, color="tab:purple", lw=1, label="$\\eta_{JS}$, swarm")
    axes[1].fill_between(sw.t, sw.eta_JS_lo, sw.eta_JS_hi, color="tab:purple", alpha=.2, lw=0)
    axes[1].set_ylim(0, 1.03)
    axes[1].set_title("pair efficiency")
    for arm in ("truth", "false"):
        for m, ls, mlab in (("Hbar", "-", "$\\Delta\\bar H$"), ("JSD", "--", "$\\Delta$JSD")):
            x = u[(u.arm == arm) & (u.measure == m)].sort_values("t")
            axes[2].plot(x.t, x["diff"], color=TCOL[arm], ls=ls, lw=1, label=f"{mlab}, {TLAB[arm]}")
    axes[2].axhline(0, color="0.6", lw=0.6)
    axes[2].set_title("agents' uncertainty (bits)")
    for ax in axes:
        ax.set_xlabel(TLABEL[sim])
        if sim == "sim2":
            ax.axvline(30, color="0.6", ls="--", lw=0.6)
        ax.legend(fontsize=5, loc="best")
    fig.tight_layout(pad=0.3)
    fig.savefig(out(sim, setup) / "fig_eff_time2.pdf")
    plt.close(fig)


def eff_phase_I(ph, sim, setup):
    """Readability I = I(target; one agent's vote) at t = 30 over qc (y) and b / rate (x); gate on and off."""
    ph = ph[(ph.sim == sim) & (ph.setup == f"task003_{setup}") & (ph.resolution == "agent_votes")]
    rhos = (0.75, 1.0) if setup == "nosolution" else (0.75,)
    gates = ["on", "off"] if sim == "sim2" else ["on"]
    vmax = float(np.nanmax(ph.I)) if len(ph) else 1
    fig, axes = plt.subplots(len(rhos), len(gates), figsize=(2.3 * len(gates) + 0.9, 1.7 * len(rhos) + 0.4), squeeze=False)
    xs = XS[sim]
    for i, rho in enumerate(rhos):
        for j, gate in enumerate(gates):
            ax = axes[i, j]
            m = np.full((4, 5), np.nan)
            for r in ph[(ph.rho == rho) & (ph.gate == gate)].itertuples():
                m[QCS.index(int(r.qc)), xs.index(float(r.x) if sim == "sim2" else int(r.x))] = r.I
            im = ax.imshow(m, origin="lower", cmap="viridis", vmin=0, vmax=vmax, aspect="auto")
            for a in range(4):
                for b in range(5):
                    if np.isfinite(m[a, b]):
                        ax.text(b, a, f"{m[a, b]:.3f}", ha="center", va="center", fontsize=5.2,
                                color="white" if m[a, b] < 0.55 * vmax else "black")
            ax.set_xticks(range(5), [f"{x:g}" for x in xs], fontsize=6)
            ax.set_yticks(range(4), QCS, fontsize=6)
            if i == 0:
                ax.set_title("$I$(target; vote)" + (f", gate {gate}" if sim == "sim2" else ""), fontsize=7)
            if i == len(rhos) - 1:
                ax.set_xlabel("$b$" if sim == "sim1" else "$\\lambda_c$", fontsize=7)
            if j == 0:
                ax.set_ylabel(f"$\\rho = {rho:g}$\n$q_c$", fontsize=7)
    fig.colorbar(im, ax=axes, shrink=0.85, pad=0.03, label="bits")
    fig.savefig(out(sim, setup) / "fig_eff_phase_I.pdf", bbox_inches="tight")
    plt.close(fig)


def success_facts(fr, sim, setup):
    """2 x 2: t = 30 (top) and 5 (bottom), rho 0.75 (left) and 1 (right); x from 0 to 500 (linear below 1)."""
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.4), sharex=True)
    for i, T in enumerate((30, 5)):
        for j, rho in enumerate((0.75, 1.0)):
            ax = axes[i, j]
            for arm in ("truth", "false"):
                g = fr[(fr.sim == sim) & (fr.setup == f"task003_{setup}") & (fr.rho == rho) & (fr.arm == arm)
                       & (fr["T"] == T)]
                if g.empty:
                    continue
                for gate, mk in (("on", "o"), ("off", "s")):
                    h = g[g.gate == gate]
                    if h.empty:
                        continue
                    ax.errorbar(h.facts, h.delta, yerr=[h.delta - h.delta_lo, h.delta_hi - h.delta], fmt=mk, ms=2.6,
                                color=TCOL[arm], alpha=0.6, lw=0.6, capsize=0,
                                label=TLAB[arm] + (f", gate {gate}" if sim == "sim2" else ""))
                c, dl = g.facts.to_numpy(), g.delta.to_numpy()
                o = np.argsort(c)
                stair = np.maximum.accumulate(np.maximum(dl[o], 0))
                ax.step(np.r_[0, c[o]], np.r_[0, stair], where="post", color=TCOL[arm], lw=1)
                hx, hy = hull(dl, c)
                ax.plot(hx, hy, color=TCOL[arm], lw=0.8, ls="--")
            ax.axhline(0, color="0.6", lw=0.5)
            ax.set_xscale("symlog", linthresh=1, linscale=0.4)
            ax.set_xlim(0, 500)
            ax.set_xticks([0, 1, 10, 100, 500], ["0", "1", "10", "100", "500"])
            when = f"day {T}" if sim == "sim1" else f"$t = {T}$"
            ax.set_title(f"{when}, $\\rho = {rho:g}$")
            if i == 1:
                ax.set_xlabel("facts posted (mean per run)")
            if j == 0:
                ax.set_ylabel("$\\Delta$ P(target wins)")
    axes[0, 0].legend(loc="best", fontsize=5.5)
    fig.tight_layout(pad=0.4)
    fig.savefig(out(sim, setup) / "fig_success_facts.pdf")
    plt.close(fig)


def precision_rows(sim, setup):
    src = REP[sim] / f"2026-10-12_{NAME[setup]}" / "figures" / "tab_eff_precision.tex"
    keep = [l for l in src.read_text().splitlines() if "progress" not in l and "focus" not in l]
    (out(sim, setup) / "tab_eff_precision.tex").write_text("\n".join(keep) + "\n")


COPY = {("sim1", "nosolution"): ["fig_drift.pdf", "fig_sensing.pdf", "fig_eff_time.pdf", "fig_eff_phase.pdf",
                                 "fig_eff_settings.pdf"],
        ("sim1", "symmetric"): ["fig_sensing.pdf", "fig_own_proof.pdf", "phase_false_base.png", "fig_eff_time.pdf",
                                "fig_eff_phase.pdf", "fig_eff_settings.pdf"],
        ("sim2", "nosolution"): ["fig_rate.pdf", "fig_distinct.pdf", "fig_sensing.pdf", "fig_eff_time.pdf",
                                 "fig_eff_phase.pdf", "fig_eff_settings.pdf"],
        ("sim2", "symmetric"): ["fig_rate.pdf", "fig_own_proof.pdf", "fig_sensing.pdf", "fig_eff_time.pdf",
                                "fig_eff_phase.pdf", "fig_eff_settings.pdf"]}


def main():
    e = pd.read_csv(EFF / "efficiency.csv")
    u = pd.read_csv(EFF / "uncertainty.csv")
    ph = pd.read_csv(EFF / "efficiency_phase.csv")
    fr = pd.read_csv(FRONTIER)
    fr = fr[(fr.q == 6) & (fr.rule == "plurality") & (fr.stop == "on") & (fr.outcome == "target")]
    for sim in ("sim1", "sim2"):
        for setup in ("nosolution", "symmetric"):
            for f in COPY[(sim, setup)]:
                shutil.copy(REP[sim] / f"2026-10-12_{NAME[setup]}" / "figures" / f, out(sim, setup) / f)
            votes_time(sim, setup)
            gain_time(sim, setup)
            phase_gain(sim, setup)
            if sim == "sim1":
                strength_sim1(setup)
            if setup == "symmetric":
                proof_time(sim)
            eff_time2(e, u, sim, setup)
            eff_phase_I(ph, sim, setup)
            success_facts(fr, sim, setup)
            precision_rows(sim, setup)
            print("written", out(sim, setup))


if __name__ == "__main__":
    main()
