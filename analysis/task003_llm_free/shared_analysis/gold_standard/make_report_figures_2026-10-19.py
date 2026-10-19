"""Figures for the 19 October reports: Simulation 2 variants against the gold standard, one report per setup, plus
one figure added to a 19 October copy of the gold-standard report.

Cases (column `case` of the tables):
  standard   the gold-standard core: budget B = 30 messages, readers sample the W = 48 newest posts
             (tables results/gold_standard/2026-10-16/, restricted to qc 12 and rho 1, 0.75, 0.5, 0.25)
  budget100  B = 100                                  (tables results/gold_standard/2026-10-18/)
  window24   W = 24, a post stays readable about 1 time unit
  window240  W = 240, about 10 time units
Each variant report has two parts; every figure is made once per part, with the part's cases side by side:
  part "budget": gold standard, B = 100          (files fig_*_budget.pdf)
  part "window": W = 24, gold standard, W = 240  (files fig_*_window.pdf)
Target A2 only (the false controller); I and eta (in the phase diagrams) use both targets by definition. q = 6, qc = 12, gate on unless stated:
  fig_votes_<part>         A0 and A2 shares over time with the false controller at rate 1 (solid) and 8 (dashed),
                           and without a controller (dotted); rows rho, columns case
  fig_gain_time_<part>     G (target A2) over time, rows rho, columns case, one curve per rate
  fig_per_fact_<part>      G/F (solid) and messages posted F (dashed), rows rho, columns case
  fig_phase_rho_q_<part>   rows case, columns G, I, eta over rho x q (rate 1), t = 30
  fig_phase_rho_x_<part>   rows case, the same over rho x rate (q 6), t = 30
  fig_gain_rate_<part>     G against the rate (top) and its slope with paired 95% intervals (bottom), columns rho,
                           one colour per case, t = 30
  fig_gain_messages_<part> G against messages posted by t = 30, one point per rate, columns rho, one colour per case
  fig_strength_time_<part> G and its slope over time x rate, rho 0.5, rows case
  fig_proofs_<part>        (symmetric only) share proving A0 from any facts (solid) or own facts (dashed), rows case
Gold-standard copy (simulation_2/analysis/reports/2026-10-19_report1_symmetric/figures/):
  fig_gain_messages.pdf    G on t = 30 against messages posted by then, one labelled point per rate, columns rho
                           (2026-10-15 tables: the gold-standard data)

Usage, from the repository root (after build_tables.py, build_slopes.py and build_info.py for 2026-10-16 and 2026-10-18):
    .venv/bin/python analysis/task003_llm_free/shared_analysis/gold_standard/make_report_figures_2026-10-19.py
"""
from __future__ import annotations
import pathlib, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import make_report_figures as M  # noqa: E402

PARTS = {"budget": ["standard", "budget100"], "window": ["window24", "standard", "window240"]}
CLAB = {"standard": "gold standard ($B$ 30, $W$ 48)", "budget100": "$B = 100$", "window24": "$W = 24$",
        "window240": "$W = 240$"}
CCOL = {"standard": "black", "budget100": "tab:green", "window24": "tab:orange", "window240": "tab:purple"}
REPORTS = {"task003_symmetric": "2026-10-19_report3_variants_symmetric",
           "task003_nosolution": "2026-10-19_report4_variants_nosolution"}
GOLD_COPY = "2026-10-19_report1_symmetric"
SIM = "sim2"
XS = M.XS[SIM]
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "legend.fontsize": 6.5, "figure.dpi": 150})


def tables():
    """Standard rows from the 2026-10-16 tables plus the 2026-10-18 variants, Simulation 2, qc 12, four rho."""
    old, new = M.RES / "2026-10-16", M.RES / "2026-10-18"

    def keep(d):          # silent cells have no controller, so whatever qc they record does not matter
        silent = d.arm == "silent" if "arm" in d.columns else False
        return d[(d.sim == SIM) & d.rho.isin(M.RHOS) & ((d.qc == M.QC) | silent)]
    t = pd.concat([keep(pd.read_parquet(old / "cells_time.parquet")).assign(case="standard"),
                   pd.read_parquet(new / "cells_time.parquet")], ignore_index=True)
    sl = pd.concat([keep(pd.read_parquet(old / "slopes.parquet")).assign(case="standard"),
                    pd.read_parquet(new / "slopes.parquet")], ignore_index=True)
    mi = pd.concat([keep(pd.read_parquet(old / "info.parquet")).assign(case="standard"),
                    pd.read_parquet(new / "info.parquet")], ignore_index=True)
    return t, sl, mi


def width(cases):
    return 2.0 + 1.85 * len(cases)


def grid(cases, h=6.6, **kw):
    fig, axes = plt.subplots(len(M.RHOS), len(cases), figsize=(width(cases), h), sharex=True, **kw)
    return fig, np.atleast_2d(axes)


def frame_rho_case(axes, cases, ylabel):
    for i, rho in enumerate(M.RHOS):
        for j, case in enumerate(cases):
            ax = axes[i, j]
            ax.axvline(30, color="0.7", ls="--", lw=0.6)
            if i == 0:
                ax.set_title(CLAB[case], fontsize=7)
            if i == len(M.RHOS) - 1:
                ax.set_xlabel("time $t$")
        axes[i, 0].set_ylabel(f"$\\rho = {rho:g}$\n{ylabel}")


def fig_votes(t, cases, out, name):
    fig, axes = grid(cases, sharey=True)
    z = pd.DataFrame({"t": [0.0], "x0": [0.0], "x2": [0.0]})          # nobody has voted at t = 0
    for i, rho in enumerate(M.RHOS):
        for j, case in enumerate(cases):
            ax = axes[i, j]
            s = pd.concat([z, M.sel(t, case=case, arm="silent", q=M.Q, rho=rho).sort_values("t")])
            ax.plot(s.t, s.x0, color="tab:blue", ls=":", lw=1, label="no controller: A0")
            ax.plot(s.t, s.x2, color="tab:red", ls=":", lw=1, label="no controller: A2")
            for x, ls in ((1.0, "-"), (8.0, "--")):
                d = pd.concat([z, M.sel(t, case=case, arm="false", q=M.Q, x=x, rho=rho, gate="on").sort_values("t")])
                ax.plot(d.t, d.x0, color="tab:blue", ls=ls, lw=1, label=f"target A2, $\\lambda_c$ {x:g}: A0")
                ax.plot(d.t, d.x2, color="tab:red", ls=ls, lw=1, label=f"target A2, $\\lambda_c$ {x:g}: A2")
            ax.set_ylim(0, 1)
    frame_rho_case(axes, cases, "vote share")
    axes[0, 0].legend(fontsize=5, loc="center right")
    fig.tight_layout(pad=0.3)
    M.save(fig, out, name)


def fig_rate_curves(src, col, cases, out, name, ylabel, ylim=None, band=False):
    fig, axes = grid(cases, sharey=True)
    for i, rho in enumerate(M.RHOS):
        for j, case in enumerate(cases):
            ax = axes[i, j]
            for k, x in enumerate(XS):
                d = M.sel(src, case=case, q=M.Q, x=x, rho=rho, gate="on").sort_values("t")
                if col == "G":
                    d = pd.concat([pd.DataFrame({"t": [0.0], "G": [0.0], "G_se": [0.0]}), d])
                ax.plot(d.t, d[col], color=M.SCOL[k], lw=0.9)
                if band:
                    ax.fill_between(d.t, d.G - 1.96 * d.G_se, d.G + 1.96 * d.G_se, color=M.SCOL[k], alpha=.1, lw=0)
            if col == "G":
                ax.axhline(0, color="0.6", lw=0.5)
            if ylim:
                ax.set_ylim(*ylim)
    frame_rho_case(axes, cases, ylabel)
    M.strength_legend(axes[0, 0], SIM, ncol=2)
    fig.tight_layout(pad=0.3)
    M.save(fig, out, name)


def fig_per_fact(t, cases, out, name):
    fig, axes = grid(cases)
    fmax = float(M.sel(t, arm="false", q=M.Q, gate="on").facts.max())
    last = len(cases) - 1
    for i, rho in enumerate(M.RHOS):
        for j, case in enumerate(cases):
            ax = axes[i, j]
            ax2 = ax.twinx()
            for k, x in enumerate(XS):
                d = M.sel(t, case=case, arm="false", q=M.Q, x=x, rho=rho, gate="on").sort_values("t")
                ok = d[d.facts >= 0.5]
                ax.plot(ok.t, ok.G / ok.facts, color=M.SCOL[k], lw=0.9)
                ax2.plot(d.t, d.facts, color=M.SCOL[k], lw=0.6, ls="--")
            ax.set_yscale("symlog", linthresh=0.003)
            ax.set_ylim(-0.0005, 0.25)
            ax2.set_ylim(0, fmax * 1.05)
            if j < last:
                ax2.set_yticklabels([])
            else:
                ax2.set_ylabel("$F_t$ (dashed)", fontsize=6)
            if j > 0:
                ax.set_yticklabels([])
    frame_rho_case(axes, cases, "$G_t/F_t$")
    M.strength_legend(axes[0, 0], SIM, ncol=2)
    fig.tight_layout(pad=0.3)
    M.save(fig, out, name)


def fig_phase(t, mi, cases, out, name, ys, xs, xlab, kw, title, gmax):
    """Rows: case. Columns: G (target A2), I(Z;X), eta, over rho (y) and a second parameter (x), t = 30, gate on."""
    fig, axes = plt.subplots(len(cases), 3, figsize=(7.6, 1.95 * len(cases) + 0.4))
    ts = M.sel(t, arm="false", gate="on", t=M.T_PHASE)
    ms = M.sel(mi, gate="on", t=M.T_PHASE)
    specs = [("$G$, target A2", ts, "G", "RdBu_r", -gmax, gmax), ("$I(Z;X)$, bits", ms, "I", "viridis", 0, 0.6),
             ("$\\eta$", ms, "eta", "viridis", 0, 1)]
    for r, case in enumerate(cases):
        for c, (lab, src, col, cmap, vmin, vmax) in enumerate(specs):
            ax = axes[r, c]
            m = np.full((len(ys), len(xs)), np.nan)
            c0 = np.zeros_like(m, bool)
            for a, yv in enumerate(ys):
                for b, xv in enumerate(xs):
                    d = M.sel(src, case=case, **kw(yv, xv))
                    if len(d):
                        m[a, b] = d.iloc[0][col]
                        if col == "G":
                            c0[a, b] = abs(d.iloc[0].G) <= 1.96 * d.iloc[0].G_se
            im = M.tile(ax, m, xs, ys, ".2f", vmin, vmax, cmap, c0 if col == "G" else None)
            if r == 0:
                ax.set_title(lab, fontsize=7.5)
            if r == len(cases) - 1:
                ax.set_xlabel(xlab)
            ax.set_ylabel(f"{CLAB[case]}\n$\\rho$" if c == 0 else "$\\rho$", fontsize=6.5)
            fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    fig.suptitle(title, fontsize=7.5, y=1.01)
    fig.tight_layout(pad=0.3)
    M.save(fig, out, name)


def fig_gain_rate(t, sl, cases, out, name):
    fig, axes = plt.subplots(2, 4, figsize=(7.6, 3.9), sharex=True, sharey="row")
    xs = np.array(XS, float)
    for j, rho in enumerate(M.RHOS):
        for n, case in enumerate(cases):
            off = (n - (len(cases) - 1) / 2) * 0.08
            d = M.sel(t, case=case, arm="false", q=M.Q, rho=rho, gate="on", t=M.T_PHASE).set_index("x").reindex(xs)
            axes[0, j].errorbar(xs + off, d.G, yerr=1.96 * d.G_se, color=CCOL[case], marker="o", ms=2.2, lw=1,
                                capsize=1.2, label=CLAB[case])
            e = M.sel(sl, case=case, arm="false", q=M.Q, rho=rho, gate="on", t=M.T_PHASE).sort_values("x_lo")
            axes[1, j].errorbar(0.5 * (e.x_lo + e.x_hi) + off, e.slope, yerr=1.96 * e.slope_se, color=CCOL[case],
                                marker="o", ms=2.2, lw=1, capsize=1.2)
        axes[0, j].set_title(f"$\\rho = {rho:g}$")
        axes[0, j].axhline(0, color="0.6", lw=0.5)
        axes[1, j].axhline(0, color="0.6", lw=0.5)
        axes[1, j].set_xlabel("$\\lambda_c$")
    axes[0, 0].set_ylabel("$G$ at $t = 30$, target A2")
    axes[1, 0].set_ylabel("slope $\\Delta G/\\Delta\\lambda_c$")
    axes[0, 0].legend(fontsize=5.5)
    fig.tight_layout(pad=0.3)
    M.save(fig, out, name)


def messages_panel(ax, d, color, label, every=False):
    """One case in one panel: G on t = 30 against messages posted by then, one point per rate, joined in rate order."""
    d = d.sort_values("x")
    ax.errorbar(d.facts, d.G, yerr=1.96 * d.G_se, color=color, marker="o", ms=2.4, lw=0.9, capsize=1.2, label=label)
    for _, r in d.iterrows():
        if every or r.x in (0.5, 1.0, 8.0):
            ax.annotate(f"{r.x:g}", (r.facts, r.G), fontsize=4.8, color=color, xytext=(2, 2),
                        textcoords="offset points")


def fig_gain_messages(t, cases, out, name):
    fig, axes = plt.subplots(1, 4, figsize=(7.6, 2.3), sharey=True)
    for j, rho in enumerate(M.RHOS):
        ax = axes[j]
        for case in cases:
            messages_panel(ax, M.sel(t, case=case, arm="false", q=M.Q, rho=rho, gate="on", t=M.T_PHASE),
                           CCOL[case], CLAB[case])
        ax.axhline(0, color="0.6", lw=0.5)
        ax.set_title(f"$\\rho = {rho:g}$")
        ax.set_xlabel("messages posted by $t = 30$")
    axes[0].set_ylabel("$G$ at $t = 30$, target A2")
    axes[0].legend(fontsize=5.5)
    fig.tight_layout(pad=0.3)
    M.save(fig, out, name)


def fig_strength_time(t, sl, cases, out, name, gmax, smax, rho=0.5):
    times = sorted(tt for tt in t.t.unique() if tt >= 1 and float(tt).is_integer())
    fig, axes = plt.subplots(len(cases), 2, figsize=(7.6, 1.8 * len(cases) + 0.3), sharex=True)
    for i, case in enumerate(cases):
        d = M.sel(t, case=case, arm="false", q=M.Q, rho=rho, gate="on")
        G = d.pivot_table(index="x", columns="t", values="G").reindex(index=XS, columns=times).to_numpy()
        slope = np.diff(G, axis=0) / np.diff(np.array(XS, float))[:, None]
        ext = [times[0] - 0.5, times[-1] + 0.5, -0.5, len(XS) - 0.5]
        im0 = axes[i, 0].imshow(G, origin="lower", aspect="auto", cmap="Reds", vmin=0, vmax=gmax, extent=ext,
                                interpolation="nearest")
        axes[i, 0].set_yticks(range(len(XS)), [f"{x:g}" for x in XS], fontsize=5.5)
        sext = [ext[0], ext[1], -0.5, len(XS) - 1.5]
        im1 = axes[i, 1].imshow(slope, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-smax, vmax=smax,
                                extent=sext, interpolation="nearest")
        axes[i, 1].set_yticks(range(len(XS) - 1), [f"{a:g}–{b:g}" for a, b in zip(XS[:-1], XS[1:])], fontsize=5)
        e = M.sel(sl, case=case, arm="false", q=M.Q, rho=rho, gate="on")
        se = e.pivot_table(index="x_lo", columns="t", values="slope_se").reindex(
            index=[float(x) for x in XS[:-1]], columns=times).to_numpy()
        ns = np.where(np.abs(slope) <= 1.96 * se, 1.0, np.nan)
        axes[i, 1].imshow(ns, origin="lower", aspect="auto", cmap="Greys", vmin=0, vmax=1, alpha=0.55, extent=sext,
                          interpolation="nearest")
        axes[i, 0].set_ylabel(f"{CLAB[case]}\n$\\lambda_c$", fontsize=6.5)
        axes[i, 1].set_ylabel("$\\lambda_c$ step", fontsize=6.5)
        for ax in axes[i]:
            ax.axvline(30, color="k", ls="--", lw=0.6)
        fig.colorbar(im0, ax=axes[i, 0], shrink=0.9, pad=0.02)
        fig.colorbar(im1, ax=axes[i, 1], shrink=0.9, pad=0.02)
    for ax in axes[-1]:
        ax.set_xlabel("time $t$")
    axes[0, 0].set_title(f"susceptibility $G$, target A2, $\\rho = {rho:g}$")
    axes[0, 1].set_title("slope $\\Delta G/\\Delta\\lambda_c$")
    fig.tight_layout(pad=0.3)
    M.save(fig, out, name)


def fig_proofs(t, cases, out, name):
    fig, axes = plt.subplots(len(cases), 4, figsize=(7.6, 1.6 * len(cases) + 0.3), sharex=True, sharey=True)
    for i, case in enumerate(cases):
        for j, rho in enumerate(M.RHOS):
            ax = axes[i, j]
            for arm in ("silent", "false"):
                d = M.sel(t, case=case, arm=arm, q=M.Q, rho=rho) if arm == "silent" else \
                    M.sel(t, case=case, arm=arm, q=M.Q, x=1.0, rho=rho, gate="on")
                d = d.sort_values("t")
                ax.plot(d.t, d.proof_any, color=M.ARMC[arm], lw=1, label=M.ARML[arm])
                ax.plot(d.t, d.proof_own, color=M.ARMC[arm], lw=0.8, ls="--")
            ax.axvline(30, color="0.7", ls="--", lw=0.6)
            ax.set_ylim(0, 1)
            if i == 0:
                ax.set_title(f"$\\rho = {rho:g}$")
            if i == len(cases) - 1:
                ax.set_xlabel("time $t$")
        axes[i, 0].set_ylabel(f"{CLAB[case]}\nshare proving A0", fontsize=6.5)
    axes[0, 0].legend(fontsize=5.5, loc="upper left")
    fig.tight_layout(pad=0.3)
    M.save(fig, out, name)


def fig_gold_messages():
    """The gold standard alone (2026-10-15 tables, Simulation 2): G of the false controller on t = 30 against the
    messages it posted by then (bottom), and those messages against the rate (top); one colour per rate."""
    t = pd.read_parquet(M.RES / "2026-10-15" / "cells_time.parquet")
    out = M.REPORT[SIM] / GOLD_COPY / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(7.6, 4.5), sharey="row")
    for j, rho in enumerate(M.RHOS):
        d = M.sel(t, sim=SIM, arm="false", q=M.Q, qc=M.QC, rho=rho, gate="on", t=M.T_PHASE).sort_values("x")
        top, ax = axes[0, j], axes[1, j]
        top.plot(d.x, d.facts, color="0.6", lw=0.8, zorder=1)
        ax.plot(d.facts, d.G, color="0.6", lw=0.8, zorder=1)
        for k, (_, r) in enumerate(d.iterrows()):          # one colour per rate, as in the other figures
            top.plot(r.x, r.facts, "o", color=M.SCOL[k], ms=3.2, zorder=2)
            ax.errorbar(r.facts, r.G, yerr=1.96 * r.G_se, color=M.SCOL[k], marker="o", ms=3.2, capsize=1.2, zorder=2)
        top.axhline(30, color="0.7", ls="--", lw=0.6)
        top.set_ylim(0, 33)
        top.set_xlabel("$\\lambda_c$")
        top.set_title(f"$\\rho = {rho:g}$")
        ax.axhline(0, color="0.6", lw=0.5)
        ax.axvline(30, color="0.7", ls="--", lw=0.6)
        ax.set_xlim(0, 33)
        ax.set_xlabel("messages posted by $t = 30$")
    axes[1, 0].set_ylim(-0.03, 0.5)
    M.strength_legend(axes[1, 0], SIM, ncol=2)
    axes[0, 0].set_ylabel("messages posted\nby $t = 30$")
    axes[1, 0].set_ylabel("$G$ at $t = 30$, target A2")
    fig.tight_layout(pad=0.3)
    M.save(fig, out, "fig_gain_messages.pdf")
    print("figure ->", out / "fig_gain_messages.pdf")


def main():
    t_all, sl_all, mi_all = tables()
    for setup, report in REPORTS.items():
        t, sl, mi = (d[d.setup == setup] for d in (t_all, sl_all, mi_all))
        out = M.REPORT[SIM] / report / "figures"
        out.mkdir(parents=True, exist_ok=True)
        fa = t[t.arm == "false"]
        gmax = float(np.nanmax(np.abs(fa[fa.t == M.T_PHASE].G)))
        gall = float(np.nanmax(np.abs(fa.G)))
        p = M.sel(fa, q=M.Q, rho=0.5, gate="on")
        smax = max(float(np.nanmax(np.abs(np.diff(
            M.sel(p, case=c).pivot_table(index="x", columns="t", values="G").to_numpy(), axis=0)
            / np.diff(np.array(XS, float))[:, None]))) for c in PARTS["budget"] + PARTS["window"])
        for part, cases in PARTS.items():
            f = lambda stem: f"{stem}_{part}.pdf"  # noqa: E731
            fig_votes(t, cases, out, f("fig_votes"))
            fig_rate_curves(M.sel(t, arm="false"), "G", cases, out, f("fig_gain_time"), "$G_t$, target A2", band=True)
            fig_per_fact(t, cases, out, f("fig_per_fact"))
            fig_phase(t, mi, cases, out, f("fig_phase_rho_q"), M.RHOS[::-1], [3, 6, 12], "$q$",
                      lambda r, q: dict(q=q, x=1.0, rho=r), "$q_c = 12$, $\\lambda_c = 1$, $t = 30$", gmax)
            fig_phase(t, mi, cases, out, f("fig_phase_rho_x"), M.RHOS[::-1], XS, "$\\lambda_c$",
                      lambda r, x: dict(q=M.Q, x=x, rho=r), "$q = 6$, $q_c = 12$, $t = 30$", gmax)
            fig_gain_rate(t, sl, cases, out, f("fig_gain_rate"))
            fig_gain_messages(t, cases, out, f("fig_gain_messages"))
            fig_strength_time(t, sl, cases, out, f("fig_strength_time"), gall, smax)
            if setup == "task003_symmetric":
                fig_proofs(t, cases, out, f("fig_proofs"))
        print("figures ->", out)
    fig_gold_messages()


if __name__ == "__main__":
    main()
