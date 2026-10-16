"""Figures for the 16 October versions of the two mirrored task003-symmetric reports (Simulations 1 and 2).

Same data and tables as the 15 October reports (cells_time.parquet, mi_table.csv with the pseudo-count of 0.5);
only the presentation changes:
  fig_votes.pdf              votes over time (as on 15 October)
  fig_timeseries.pdf         4 x 4: G target A0, G target A2, I(Z;X), eta (rows) x rho (columns), one curve per strength
  fig_per_fact.pdf           2 x 4: G_t/F_t (left axis, solid) and facts posted F_t (right axis, dashed), per target
  fig_phase_rho_q.pdf        1 x 3: G (target A2), I, eta over rho x q (qc 12, main strength), t = 30
  fig_phase_rho_x.pdf        1 x 3: same over rho x strength (q 6, qc 12)
  fig_phase_qc_x.pdf         1 x 3: same over qc x strength (rho 0.5, q 6)
  fig_gain_strength.pdf      G against strength and its slope (as on 15 October)
  fig_strength_time.pdf      2 x 2: G (target A2) and its slope over time (x) and strength (y), rho 0.5 and 0.75
  fig_proofs.pdf             who holds the proof (as on 15 October)

Usage, from the repository root (after build_tables.py and make_report_figures.py):
    .venv/bin/python analysis/task003_llm_free/symmetric_fine/make_report_figures_2026-10-16.py
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

DATE_DATA = "2026-10-15"
DATE_REPORT = "2026-10-16"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "legend.fontsize": 6.5, "figure.dpi": 150})


def add_zero(d, cols):
    z = pd.DataFrame({"t": [0.0], **{c: [0.0] for c in cols}})
    return pd.concat([z, d.sort_values("t")])


def fig_timeseries(t, mi, sim, out):
    fig, axes = plt.subplots(4, 4, figsize=(7.4, 7.0), sharex=True, sharey="row")
    for j, rho in enumerate(M.RHOS):
        for k, x in enumerate(M.XS[sim]):
            c = M.SCOL[k]
            for i, arm in enumerate(("truth", "false")):
                d = add_zero(M.sel(t, sim=sim, arm=arm, q=M.Q, qc=M.QC, x=x, rho=rho, gate="on"), ["G", "G_se"])
                axes[i, j].plot(d.t, d.G, color=c, lw=1)
                axes[i, j].fill_between(d.t, d.G - 1.96 * d.G_se, d.G + 1.96 * d.G_se, color=c, alpha=.12, lw=0)
            d = M.sel(mi, sim=sim, q=M.Q, qc=M.QC, x=x, rho=rho, gate="on", state="vote").sort_values("t")
            axes[2, j].plot(d.t, d.I, color=c, lw=1)
            axes[3, j].plot(d.t, d.eta, color=c, lw=1)
        axes[0, j].set_title(f"$\\rho = {rho:g}$")
        axes[3, j].set_xlabel(M.TNAME[sim])
        axes[3, j].set_ylim(0, 1.02)
        for i in range(4):
            if i < 2:
                axes[i, j].axhline(0, color="0.6", lw=0.5)
            if sim == "sim2":
                axes[i, j].axvline(30, color="0.7", ls="--", lw=0.6)
    for i, lab in enumerate(("$G_t$, target A0", "$G_t$, target A2", "$I(Z;X)$ (bits)", "$\\eta$")):
        axes[i, 0].set_ylabel(lab)
    M.strength_legend(axes[1, 0], sim)
    fig.tight_layout(pad=0.3)
    M.save(fig, out, "fig_timeseries.pdf")


def fig_per_fact(t, sim, out):
    fig, axes = plt.subplots(2, 4, figsize=(7.6, 3.9), sharex=True)
    fmax = float(M.sel(t, sim=sim, q=M.Q, qc=M.QC, gate="on").facts.max())
    for j, rho in enumerate(M.RHOS):
        for i, arm in enumerate(("truth", "false")):
            ax = axes[i, j]
            ax2 = ax.twinx()
            for k, x in enumerate(M.XS[sim]):
                d = M.sel(t, sim=sim, arm=arm, q=M.Q, qc=M.QC, x=x, rho=rho, gate="on").sort_values("t")
                ok = d[d.facts >= 0.5]
                ax.plot(ok.t, ok.G / ok.facts, color=M.SCOL[k], lw=1)
                ax2.plot(d.t, d.facts, color=M.SCOL[k], lw=0.7, ls="--")
            ax.set_yscale("symlog", linthresh=0.003)
            ax.set_ylim(-0.0005, 0.25)
            ax2.set_ylim(0, fmax * 1.05)
            if sim == "sim2":
                ax.axvline(30, color="0.7", ls="--", lw=0.6)
            if j < 3:
                ax2.set_yticklabels([])
            else:
                ax2.set_ylabel("facts posted $F_t$ (dashed)")
            if j > 0:
                ax.set_yticklabels([])
            if i == 0:
                ax.set_title(f"$\\rho = {rho:g}$")
            if i == 1:
                ax.set_xlabel(M.TNAME[sim])
            if j == 0:
                ax.set_ylabel(f"$G_t/F_t$, {M.ARML[arm]} (solid)")
    M.strength_legend(axes[1, 0], sim)
    fig.tight_layout(pad=0.3)
    M.save(fig, out, "fig_per_fact.pdf")


def phase_triple(t, mi, sim, out, name, ys, xs, ylab, xlab, kw, title, gmax):
    """1 x 3 tiles: G (target A2), I(Z;X), eta, over the same two parameters, t = 30, gate on."""
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.3 if len(xs) > 3 else 2.2))
    ts = M.sel(t, sim=sim, arm="false", gate="on", t=M.T_PHASE)
    ms = M.sel(mi, sim=sim, gate="on", t=M.T_PHASE, state="vote")
    specs = [("$G$, target A2", ts, "G", "RdBu_r", -gmax, gmax), ("$I(Z;X)$, bits", ms, "I", "viridis", 0, 0.6),
             ("$\\eta$", ms, "eta", "viridis", 0, 1)]
    for ax, (lab, src, col, cmap, vmin, vmax) in zip(axes, specs):
        m = np.full((len(ys), len(xs)), np.nan)
        c0 = np.zeros_like(m, bool)
        for a, yv in enumerate(ys):
            for b, xv in enumerate(xs):
                d = M.sel(src, **kw(yv, xv))
                if len(d):
                    m[a, b] = d.iloc[0][col]
                    if col == "G":
                        c0[a, b] = abs(d.iloc[0].G) <= 1.96 * d.iloc[0].G_se
        im = M.tile(ax, m, xs, ys, ".2f", vmin, vmax, cmap, c0 if col == "G" else None)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_title(lab, fontsize=7.5)
        fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    fig.suptitle(title, fontsize=7.5, y=1.02)
    fig.tight_layout(pad=0.3)
    M.save(fig, out, name)


def fig_strength_time(t, sim, out, gmax, smax):
    xs = M.XS[sim]
    times = sorted(tt for tt in M.sel(t, sim=sim).t.unique() if tt >= 1 and float(tt).is_integer())
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 3.9), sharex=True)
    for i, rho in enumerate((0.5, 0.75)):
        d = M.sel(t, sim=sim, arm="false", q=M.Q, qc=M.QC, rho=rho, gate="on")
        G = d.pivot_table(index="x", columns="t", values="G").reindex(index=xs, columns=times).to_numpy()
        slope = np.diff(G, axis=0) / np.diff(np.array(xs, float))[:, None]
        ext = [times[0] - 0.5, times[-1] + 0.5, -0.5, len(xs) - 0.5]
        im0 = axes[i, 0].imshow(G, origin="lower", aspect="auto", cmap="Reds", vmin=0, vmax=gmax, extent=ext,
                                interpolation="nearest")
        axes[i, 0].set_yticks(range(len(xs)), [f"{x:g}" for x in xs], fontsize=5.5)
        im1 = axes[i, 1].imshow(slope, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-smax, vmax=smax,
                                extent=[ext[0], ext[1], -0.5, len(xs) - 1.5], interpolation="nearest")
        axes[i, 1].set_yticks(range(len(xs) - 1), [f"{a:g}–{b:g}" for a, b in zip(xs[:-1], xs[1:])], fontsize=5)
        axes[i, 0].set_ylabel(f"$\\rho = {rho:g}$\n{M.XNAME[sim]}")
        axes[i, 1].set_ylabel(f"{M.XNAME[sim]} step")
        for ax in axes[i]:
            if sim == "sim2":
                ax.axvline(30, color="k", ls="--", lw=0.6)
        fig.colorbar(im0, ax=axes[i, 0], shrink=0.9, pad=0.02)
        fig.colorbar(im1, ax=axes[i, 1], shrink=0.9, pad=0.02)
    for ax in axes[1]:
        ax.set_xlabel(M.TNAME[sim])
    axes[0, 0].set_title("susceptibility $G$, target A2")
    axes[0, 1].set_title(f"slope $\\Delta G/\\Delta${M.XNAME[sim]}")
    fig.tight_layout(pad=0.3)
    M.save(fig, out, "fig_strength_time.pdf")


def main():
    d = M.RES / DATE_DATA
    t = pd.read_parquet(d / "cells_time.parquet")
    mi = pd.read_csv(d / "mi_table.csv")
    at30 = t[(t.t == M.T_PHASE) & (t.arm == "false")]
    gmax = float(np.nanmax(np.abs(at30.G)))
    gall = float(np.nanmax(np.abs(t[t.arm == "false"].G)))
    smax = 0.0
    for sim in ("sim1", "sim2"):
        for rho in (0.5, 0.75):
            p = M.sel(t, sim=sim, arm="false", q=M.Q, qc=M.QC, rho=rho, gate="on").pivot_table(index="x", columns="t", values="G")
            smax = max(smax, float(np.nanmax(np.abs(np.diff(p.to_numpy(), axis=0) / np.diff(p.index.to_numpy(float))[:, None]))))
    for sim in ("sim1", "sim2"):
        out = M.REPORT[sim] / f"{DATE_REPORT}_report1_symmetric" / "figures"
        out.mkdir(parents=True, exist_ok=True)
        ts, ms = t[t.sim == sim], mi[mi.sim == sim]
        M.fig_votes(ts, sim, out, "fig_votes.pdf", "on", M.MAIN[sim])
        fig_timeseries(ts, ms, sim, out)
        fig_per_fact(ts, sim, out)
        xs, qs, qcs, rhos = M.XS[sim], [3, 6, 12], [3, 6, 12, 24], M.RHOS[::-1]
        xm = M.MAIN[sim]
        phase_triple(ts, ms, sim, out, "fig_phase_rho_q.pdf", rhos, qs, "$\\rho$", "$q$",
                     lambda r, q: dict(q=q, qc=M.QC, x=xm, rho=r), f"$q_c = {M.QC}$, {M.label_x(sim, xm)}, $t = 30$", gmax)
        phase_triple(ts, ms, sim, out, "fig_phase_rho_x.pdf", rhos, xs, "$\\rho$", M.XNAME[sim],
                     lambda r, x: dict(q=M.Q, qc=M.QC, x=x, rho=r), f"$q = {M.Q}$, $q_c = {M.QC}$, $t = 30$", gmax)
        phase_triple(ts, ms, sim, out, "fig_phase_qc_x.pdf", qcs, xs, "$q_c$", M.XNAME[sim],
                     lambda qc, x: dict(q=M.Q, qc=qc, x=x, rho=M.RHO_REP), f"$\\rho = {M.RHO_REP:g}$, $q = {M.Q}$, $t = 30$", gmax)
        M.fig_gain_strength(ts, sim, out, "fig_gain_strength.pdf")
        fig_strength_time(ts, sim, out, gall, smax)
        M.fig_proofs(ts, sim, out, "fig_proofs.pdf", "on", M.MAIN[sim])
        print("figures ->", out)


if __name__ == "__main__":
    main()
