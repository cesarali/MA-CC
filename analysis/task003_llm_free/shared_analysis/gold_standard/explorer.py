"""Plot functions for the interactive notebook `explorer.ipynb`: the gold-standard figures with rho (and, in the phase
diagrams, the parameters held fixed) as sliders. Every function draws one view and returns the figure; the notebook
wires them to sliders. Works for both simulations and both setups.

Data: results/gold_standard/2026-10-16/{cells_time.parquet, info.parquet, slopes.parquet}, built by build_tables.py,
build_info.py and build_slopes.py with --date 2026-10-16 (symmetric and nosolution, rho 1, 0.875, ..., 0.125).
"""
from __future__ import annotations
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "results" / "gold_standard" / "2026-10-16"
XS = {"sim1": [1, 2, 3, 4, 5, 6, 7, 8, 9], "sim2": [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]}
XNAME = {"sim1": "$b$", "sim2": "$\\lambda_c$"}
TNAME = {"sim1": "day", "sim2": "time $t$"}
RHOS = [1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125]
QS, QCS = [3, 6, 12], [3, 6, 12, 24]
ARMC = {"silent": "0.45", "truth": "tab:blue", "false": "tab:red"}
ARML = {"silent": "silent", "truth": "target A0", "false": "target A2"}
SCOL = plt.cm.plasma(np.linspace(0, 0.88, 9))
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "legend.fontsize": 7, "figure.dpi": 110})

_T = _I = _S = None


def data():
    """Load the three tables once."""
    global _T, _I, _S
    if _T is None:
        _T = pd.read_parquet(DATA / "cells_time.parquet")
        _I = pd.read_parquet(DATA / "info.parquet")
        _S = pd.read_parquet(DATA / "slopes.parquet")
    return _T, _I, _S


def sel(df, **kw):
    m = np.ones(len(df), bool)
    for k, v in kw.items():
        m &= np.isclose(df[k].to_numpy(float), v) if isinstance(v, float) else (df[k] == v).to_numpy()
    return df[m]


def times(sim):
    t, _, _ = data()
    return sorted(float(x) for x in sel(t, sim=sim).t.unique())


def _zero(d, cols):
    return pd.concat([pd.DataFrame({"t": [0.0], **{c: [0.0] for c in cols}}), d.sort_values("t")])


def _legend_strength(ax, sim):
    for i, x in enumerate(XS[sim]):
        ax.plot([], [], color=SCOL[i], lw=1.2, label=f"{x:g}")
    ax.legend(title=XNAME[sim], fontsize=6.5, title_fontsize=7, ncol=2, loc="best")


# ------------------------------------------------------------------ Fig. 1, votes over time
def votes(sim, setup, rho, x, gate="on", q=6, qc=12):
    t, _, _ = data()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for arm in ("silent", "truth", "false"):
        d = sel(t, sim=sim, setup=setup, arm=arm, q=q, rho=rho) if arm == "silent" else \
            sel(t, sim=sim, setup=setup, arm=arm, q=q, qc=qc, x=float(x), rho=rho, gate=gate)
        d = _zero(d, ["x0", "x2"])
        for ax, c in zip(axes, ("x0", "x2")):
            ax.plot(d.t, d[c], color=ARMC[arm], ls=":" if arm == "silent" else "-", lw=1.4, label=ARML[arm])
    for ax, lab in zip(axes, ("A0", "A2")):
        ax.set_title(f"share voting {lab}")
        ax.set_xlabel(TNAME[sim])
        ax.set_ylim(0, 1)
        if sim == "sim2":
            ax.axvline(30, color="0.7", ls="--", lw=0.7)
    axes[0].legend()
    fig.suptitle(f"{setup}, $\\rho = {rho:g}$, {XNAME[sim]} = {x:g}, q = {q}, $q_c$ = {qc}, gate {gate}")
    fig.tight_layout()
    plt.close(fig)      # shown once, as the return value (interact displays it); an open figure is shown again
    return fig


# ------------------------------------------------------------------ Fig. 2, observables as time series
def timeseries(sim, setup, rho, gate="on", q=6, qc=12):
    t, inf, _ = data()
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.0))
    for k, x in enumerate(XS[sim]):
        for ax, arm in zip(axes[:2], ("truth", "false")):
            d = _zero(sel(t, sim=sim, setup=setup, arm=arm, q=q, qc=qc, x=float(x), rho=rho, gate=gate), ["G", "G_se"])
            ax.plot(d.t, d.G, color=SCOL[k], lw=1)
            ax.fill_between(d.t, d.G - 1.96 * d.G_se, d.G + 1.96 * d.G_se, color=SCOL[k], alpha=.12, lw=0)
        d = sel(inf, sim=sim, setup=setup, q=q, qc=qc, x=float(x), rho=rho, gate=gate).sort_values("t")
        axes[2].plot(d.t, d.I, color=SCOL[k], lw=1)
        axes[3].plot(d.t, d.eta, color=SCOL[k], lw=1)
    for ax, lab in zip(axes, ("$G_t$, target A0", "$G_t$, target A2", "$I(Z;X)$ (bits)", "$\\eta$")):
        ax.set_title(lab)
        ax.set_xlabel(TNAME[sim])
        if sim == "sim2":
            ax.axvline(30, color="0.7", ls="--", lw=0.7)
    axes[3].set_ylim(0, 1.02)
    _legend_strength(axes[1], sim)
    fig.suptitle(f"{setup}, $\\rho = {rho:g}$, q = {q}, $q_c$ = {qc}, gate {gate}")
    fig.tight_layout()
    plt.close(fig)      # shown once, as the return value (interact displays it); an open figure is shown again
    return fig


# ------------------------------------------------------------------ Fig. 3, gain per fact and facts posted
def per_fact(sim, setup, rho, gate="on", q=6, qc=12):
    t, _, _ = data()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, arm in zip(axes, ("truth", "false")):
        ax2 = ax.twinx()
        for k, x in enumerate(XS[sim]):
            d = sel(t, sim=sim, setup=setup, arm=arm, q=q, qc=qc, x=float(x), rho=rho, gate=gate).sort_values("t")
            ok = d[d.facts >= 0.5]
            ax.plot(ok.t, ok.G / ok.facts, color=SCOL[k], lw=1)
            ax2.plot(d.t, d.facts, color=SCOL[k], lw=0.7, ls="--")
        ax.set_yscale("symlog", linthresh=0.003)
        ax.set_title(f"{ARML[arm]}: $G_t/F_t$ (solid), $F_t$ (dashed, right)")
        ax.set_xlabel(TNAME[sim])
        ax2.set_ylabel("facts posted $F_t$")
    _legend_strength(axes[0], sim)
    fig.suptitle(f"{setup}, $\\rho = {rho:g}$, q = {q}, $q_c$ = {qc}, gate {gate}")
    fig.tight_layout()
    plt.close(fig)      # shown once, as the return value (interact displays it); an open figure is shown again
    return fig


# ------------------------------------------------------------------ Figs. 4-6, phase diagrams (G, I, eta)
def _tiles(ax, m, xs, ys, fmt, vmin, vmax, cmap):
    im = ax.imshow(m, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    for a in range(m.shape[0]):
        for b in range(m.shape[1]):
            v = m[a, b]
            ax.text(b, a, "–" if not np.isfinite(v) else format(v, fmt).replace("0.", ".", 1), ha="center",
                    va="center", fontsize=6)
    ax.set_xticks(range(len(xs)), [f"{x:g}" for x in xs])
    ax.set_yticks(range(len(ys)), [f"{y:g}" for y in ys])
    return im


def _phase(sim, setup, ys, xs, ylab, xlab, kw, title, tt, gate):
    t, inf, _ = data()
    ts = sel(t, sim=sim, setup=setup, arm="false", gate=gate, t=float(tt))
    ms = sel(inf, sim=sim, setup=setup, gate=gate, t=float(tt))
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    gmax = max(0.05, float(np.nanmax(np.abs(sel(t, sim=sim, setup=setup, arm="false").G))))
    for ax, (lab, src, col, cmap, lo, hi) in zip(axes, [("$G$, target A2", ts, "G", "RdBu_r", -gmax, gmax),
                                                        ("$I(Z;X)$, bits", ms, "I", "viridis", 0, 0.6),
                                                        ("$\\eta$", ms, "eta", "viridis", 0, 1)]):
        m = np.full((len(ys), len(xs)), np.nan)
        for a, yv in enumerate(ys):
            for b, xv in enumerate(xs):
                d = sel(src, **kw(yv, xv))
                if len(d):
                    m[a, b] = d.iloc[0][col]
        im = _tiles(ax, m, xs, ys, ".2f", lo, hi, cmap)
        ax.set_title(lab)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        fig.colorbar(im, ax=ax, shrink=0.85)
    fig.suptitle(f"{setup}, {title}, {TNAME[sim]} = {tt:g}, gate {gate}")
    fig.tight_layout()
    plt.close(fig)      # shown once, as the return value (interact displays it); an open figure is shown again
    return fig


def phase_rho_q(sim, setup, qc, x, tt, gate="on"):
    return _phase(sim, setup, RHOS[::-1], QS, "$\\rho$", "$q$",
                  lambda r, q: dict(q=q, qc=qc, x=float(x), rho=r), f"$q_c$ = {qc}, {XNAME[sim]} = {x:g}", tt, gate)


def phase_rho_x(sim, setup, q, qc, tt, gate="on"):
    return _phase(sim, setup, RHOS[::-1], XS[sim], "$\\rho$", XNAME[sim],
                  lambda r, x: dict(q=q, qc=qc, x=float(x), rho=r), f"q = {q}, $q_c$ = {qc}", tt, gate)


def phase_qc_x(sim, setup, rho, q, tt, gate="on"):
    return _phase(sim, setup, QCS, XS[sim], "$q_c$", XNAME[sim],
                  lambda c, x: dict(q=q, qc=c, x=float(x), rho=rho), f"$\\rho$ = {rho:g}, q = {q}", tt, gate)


# ------------------------------------------------------------------ Fig. 7, G against strength and its slope
def gain_strength(sim, setup, rho, tt, q=6, qc=12):
    t, _, sl = data()
    xs = np.array(XS[sim], float)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for gate, ls, mk in (("on", "-", "o"), ("off", "--", "s")):
        d = sel(t, sim=sim, setup=setup, arm="false", q=q, qc=qc, rho=rho, gate=gate, t=float(tt)).set_index("x").reindex(xs)
        axes[0].errorbar(xs, d.G, yerr=1.96 * d.G_se, color="tab:red", ls=ls, marker=mk, ms=3, capsize=2, label=f"gate {gate}")
        e = sel(sl, sim=sim, setup=setup, arm="false", q=q, qc=qc, rho=rho, gate=gate, t=float(tt)).sort_values("x_lo")
        axes[1].errorbar(0.5 * (e.x_lo + e.x_hi), e.slope, yerr=1.96 * e.slope_se, color="tab:red", ls=ls, marker=mk,
                         ms=3, capsize=2)
    axes[0].set_ylabel("$G$, target A2")
    axes[1].set_ylabel(f"slope $\\Delta G/\\Delta${XNAME[sim]} (paired 95% intervals)")
    axes[1].axhline(0, color="0.6", lw=0.6)
    for ax in axes:
        ax.set_xlabel(XNAME[sim])
    axes[0].legend()
    fig.suptitle(f"{setup}, $\\rho = {rho:g}$, {TNAME[sim]} = {tt:g}, q = {q}, $q_c$ = {qc}")
    fig.tight_layout()
    plt.close(fig)      # shown once, as the return value (interact displays it); an open figure is shown again
    return fig


# ------------------------------------------------------------------ Fig. 8, time x strength maps
def strength_time(sim, setup, rho, gate="on", q=6, qc=12):
    t, _, sl = data()
    xs = XS[sim]
    ts = [x for x in times(sim) if x >= 1 and float(x).is_integer()]
    d = sel(t, sim=sim, setup=setup, arm="false", q=q, qc=qc, rho=rho, gate=gate)
    G = d.pivot_table(index="x", columns="t", values="G").reindex(index=[float(x) for x in xs], columns=ts).to_numpy()
    e = sel(sl, sim=sim, setup=setup, arm="false", q=q, qc=qc, rho=rho, gate=gate)
    S = e.pivot_table(index="x_lo", columns="t", values="slope").reindex(index=[float(x) for x in xs[:-1]], columns=ts).to_numpy()
    SE = e.pivot_table(index="x_lo", columns="t", values="slope_se").reindex(index=[float(x) for x in xs[:-1]], columns=ts).to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
    ext = [ts[0] - 0.5, ts[-1] + 0.5, -0.5, len(xs) - 0.5]
    im0 = axes[0].imshow(G, origin="lower", aspect="auto", cmap="Reds", vmin=0, extent=ext, interpolation="nearest")
    axes[0].set_yticks(range(len(xs)), [f"{x:g}" for x in xs])
    smax = float(np.nanmax(np.abs(S))) or 1
    ext1 = [ext[0], ext[1], -0.5, len(xs) - 1.5]
    im1 = axes[1].imshow(S, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-smax, vmax=smax, extent=ext1, interpolation="nearest")
    axes[1].imshow(np.where(np.abs(S) <= 1.96 * SE, 1.0, np.nan), origin="lower", aspect="auto", cmap="Greys", vmin=0,
                   vmax=1, alpha=0.55, extent=ext1, interpolation="nearest")
    axes[1].set_yticks(range(len(xs) - 1), [f"{a:g}–{b:g}" for a, b in zip(xs[:-1], xs[1:])], fontsize=7)
    axes[0].set_title("$G$, target A2")
    axes[1].set_title(f"slope $\\Delta G/\\Delta${XNAME[sim]} (faded: interval includes 0)")
    for ax in axes:
        ax.set_xlabel(TNAME[sim])
        if sim == "sim2":
            ax.axvline(30, color="k", ls="--", lw=0.7)
    axes[0].set_ylabel(XNAME[sim])
    fig.colorbar(im0, ax=axes[0])
    fig.colorbar(im1, ax=axes[1])
    fig.suptitle(f"{setup}, $\\rho = {rho:g}$, q = {q}, $q_c$ = {qc}, gate {gate}")
    fig.tight_layout()
    plt.close(fig)      # shown once, as the return value (interact displays it); an open figure is shown again
    return fig


# ------------------------------------------------------------------ Fig. 8b, fitted derivative (Simulation 2)
def fit_saturating(xs, g, se):
    """Weighted least-squares fit of G(lambda) = Ginf (1 - exp(-lambda / lambda0)); lambda0 by grid search,
    Ginf in closed form. Returns (Ginf, lambda0, weighted residual)."""
    w = 1 / np.maximum(se, 1e-4) ** 2
    best = (np.nan, np.nan, np.inf)
    for l0 in np.geomspace(0.02, 50, 400):
        f = 1 - np.exp(-xs / l0)
        ginf = np.sum(w * f * g) / np.sum(w * f * f)
        r = np.sum(w * (g - ginf * f) ** 2)
        if r < best[2]:
            best = (ginf, l0, r)
    return best


def fitted_derivative(setup, rho, tt, gate="on", q=6, qc=12):
    """Simulation 2: G against the rate at time tt with the fitted curve (left); dG/dlambda from the fit over time and
    rate (centre); the fitted scale lambda0(t), the rate where returns have fallen by e (right)."""
    t, _, _ = data()
    xs = np.array(XS["sim2"])
    d = sel(t, sim="sim2", setup=setup, arm="false", q=q, qc=qc, rho=rho, gate=gate)
    ts = [x for x in times("sim2") if x >= 1 and float(x).is_integer()]
    fine = np.linspace(0.25, 8, 120)
    D = np.full((len(fine), len(ts)), np.nan)
    l0s, ginfs = [], []
    for j, tj in enumerate(ts):
        e = sel(d, t=float(tj)).set_index("x").reindex(xs)
        ginf, l0, _ = fit_saturating(xs, e.G.to_numpy(), e.G_se.to_numpy())
        D[:, j] = ginf / l0 * np.exp(-fine / l0)
        l0s.append(l0)
        ginfs.append(ginf)
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
    e = sel(d, t=float(tt)).set_index("x").reindex(xs)
    ginf, l0, _ = fit_saturating(xs, e.G.to_numpy(), e.G_se.to_numpy())
    axes[0].errorbar(xs, e.G, yerr=1.96 * e.G_se, fmt="o", color="tab:red", ms=3, capsize=2, label="data")
    axes[0].plot(fine, ginf * (1 - np.exp(-fine / l0)), color="k", lw=1,
                 label=f"fit: $G_\\infty$ = {ginf:.2f}, $\\lambda_0$ = {l0:.2f}")
    axes[0].set_xlabel("$\\lambda_c$")
    axes[0].set_ylabel("$G$, target A2")
    axes[0].set_title(f"$t$ = {tt:g}")
    axes[0].legend(fontsize=7)
    vmax = np.nanpercentile(D, 99)
    im = axes[1].imshow(D, origin="lower", aspect="auto", cmap="Reds", vmin=0, vmax=vmax,
                        extent=[ts[0] - 0.5, ts[-1] + 0.5, fine[0], fine[-1]])
    axes[1].set_xlabel("time $t$")
    axes[1].set_ylabel("$\\lambda_c$")
    axes[1].set_title("fitted $dG/d\\lambda_c$")
    axes[1].axvline(30, color="k", ls="--", lw=0.7)
    fig.colorbar(im, ax=axes[1])
    axes[2].plot(ts, l0s, color="k")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("time $t$")
    axes[2].set_ylabel("$\\lambda_0$ (rate scale of the fit)")
    axes[2].axvline(30, color="0.7", ls="--", lw=0.7)
    fig.suptitle(f"Simulation 2, {setup}, $\\rho = {rho:g}$, q = {q}, $q_c$ = {qc}, gate {gate}; "
                 "fit $G = G_\\infty(1 - e^{-\\lambda_c/\\lambda_0})$ at each time")
    fig.tight_layout()
    plt.close(fig)      # shown once, as the return value (interact displays it); an open figure is shown again
    return fig


# ------------------------------------------------------------------ Fig. 9, who holds the proof
def proofs(sim, setup, rho, x, gate="on", q=6, qc=12):
    t, _, _ = data()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for arm in ("silent", "truth", "false"):
        d = sel(t, sim=sim, setup=setup, arm=arm, q=q, rho=rho) if arm == "silent" else \
            sel(t, sim=sim, setup=setup, arm=arm, q=q, qc=qc, x=float(x), rho=rho, gate=gate)
        d = d.sort_values("t")
        for ax, c in zip(axes, ("proof_any", "proof_own")):
            ax.plot(d.t, d[c], color=ARMC[arm], ls=":" if arm == "silent" else "-", lw=1.4, label=ARML[arm])
    axes[0].set_title("proves A0, any facts")
    axes[1].set_title("proves A0, own facts only")
    for ax in axes:
        ax.set_xlabel(TNAME[sim])
        ax.set_ylim(0, 1)
    axes[0].legend()
    fig.suptitle(f"{setup}, $\\rho = {rho:g}$, {XNAME[sim]} = {x:g}, q = {q}, $q_c$ = {qc}, gate {gate}")
    fig.tight_layout()
    plt.close(fig)      # shown once, as the return value (interact displays it); an open figure is shown again
    return fig
