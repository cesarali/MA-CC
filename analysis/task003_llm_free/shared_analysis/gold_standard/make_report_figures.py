"""Figures for the two mirrored reports of 15 October 2026 on task003-symmetric (Simulations 1 and 2).

Reads results/gold_standard/<date>/cells_time.parquet (build_tables.py) and writes, for each simulation,
into simulation_<n>/analysis/reports/<date>_report1_symmetric/figures/ the figures listed in
shared_analysis/gold_standard/2026-10-15_plan_reports.md, plus mi_table.csv (readability and efficiency for every
setting, state and time) and numbers.json (values quoted in the text).

Usage, from the repository root:
    .venv/bin/python analysis/task003_llm_free/shared_analysis/gold_standard/make_report_figures.py [--date 2026-10-15]
"""
from __future__ import annotations
import argparse, json, pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "gold_standard"
REPORT = {"sim1": ROOT / "simulation_1/analysis/reports", "sim2": ROOT / "simulation_2/analysis/reports"}
XS = {"sim1": [1, 2, 3, 4, 5, 6, 7, 8, 9], "sim2": [0.5, 1, 2, 3, 4, 5, 6, 7, 8]}
XNAME = {"sim1": "$b$", "sim2": "$\\lambda_c$"}
XLONG = {"sim1": "facts per night $b$", "sim2": "controller rate $\\lambda_c$"}
TNAME = {"sim1": "day", "sim2": "time $t$"}
MAIN = {"sim1": 1, "sim2": 1}
STRONG = {"sim1": 9, "sim2": 8}
RHOS = [1.0, 0.75, 0.5, 0.25]
RHO_REP = 0.5
Q, QC, T_PHASE = 6, 12, 30.0
ARMC = {"silent": "0.45", "truth": "tab:blue", "false": "tab:red"}
ARML = {"silent": "silent", "truth": "target A0", "false": "target A2"}
SCOL = plt.cm.plasma(np.linspace(0, 0.88, 9))
STATES = {"vote": ("one agent's vote", 3), "plur": ("plurality of votes", 3), "s1": ("S1: $m_{A0}$, 4 bins", 4),
          "s3": ("S3: $(m_{A0}, m_{A2})$, 3$\\times$3 bins", 9)}
C_MIN = 0.01
ALPHA = 0.5          # pseudo-count (episodes) added to every cell of every arm before computing KL (Eq. smoothing)
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "legend.fontsize": 6.5, "figure.dpi": 150})


# ------------------------------------------------------------------ information
def kl(p, q):
    nz = p > 1e-15
    if np.any(q[nz] <= 1e-15):
        return np.inf
    return float(np.sum(p[nz] * np.log2(p[nz] / q[nz])))


def mi_table(t: pd.DataFrame, alpha: float = ALPHA) -> pd.DataFrame:
    """I(Z; X), C, R and eta for every setting, time and state (Z: target A0 or A2, fair coin)."""
    key = ["sim", "q", "qc", "x", "rho", "gate", "t"]
    tr = t[t.arm == "truth"].set_index(key)
    fa = t[t.arm == "false"].set_index(key)
    si = t[t.arm == "silent"].set_index(["sim", "q", "rho", "t"])
    rows = []
    for k in tr.index.intersection(fa.index):
        a, b = tr.loc[k], fa.loc[k]
        s = si.loc[(k[0], k[1], k[4], k[6])]
        for st, (_, n) in STATES.items():
            cols = [f"{st}_{i}" for i in range(n)]
            pp, pm, p0 = a[cols].to_numpy(float), b[cols].to_numpy(float), s[cols].to_numpy(float)
            pp, pm, p0 = pp / pp.sum(), pm / pm.sum(), p0 / p0.sum()
            if alpha > 0:       # the same pseudo-count in every arm: (E P + alpha) / (E + alpha K)
                pp = (a.episodes * pp + alpha) / (a.episodes + alpha * n)
                pm = (b.episodes * pm + alpha) / (b.episodes + alpha * n)
                p0 = (s.episodes * p0 + alpha) / (s.episodes + alpha * n)
            qm = 0.5 * (pp + pm)
            I = 0.5 * kl(pp, qm) + 0.5 * kl(pm, qm)
            C = 0.5 * kl(pp, p0) + 0.5 * kl(pm, p0)
            R = kl(qm, p0)
            eta = I / C if (np.isfinite(C) and C >= C_MIN) else np.nan
            rows.append(dict(zip(key, k), state=st, I=I, C=C, R=R, eta=eta))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ helpers
def sel(df, **kw):
    m = np.ones(len(df), bool)
    for k, v in kw.items():
        m &= (df[k] == v).to_numpy()
    return df[m]


def save(fig, out, name):
    fig.savefig(out / name, bbox_inches="tight")
    plt.close(fig)


def label_x(sim, x):
    return f"{XNAME[sim]} = {x:g}"


def strength_legend(ax, sim, ncol=1):
    for i, x in enumerate(XS[sim]):
        ax.plot([], [], color=SCOL[i], lw=1.2, label=f"{x:g}")
    ax.legend(title=XNAME[sim], fontsize=5.5, title_fontsize=6, ncol=ncol, loc="best")


def tile(ax, m, xs, ys, fmt, vmin, vmax, cmap, ci0=None):
    im = ax.imshow(m, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    cm = plt.get_cmap(cmap)
    for a in range(m.shape[0]):
        for b in range(m.shape[1]):
            v = m[a, b]
            if not np.isfinite(v):
                ax.text(b, a, "–", ha="center", va="center", fontsize=5)
                continue
            r, g, bl, _ = cm((min(max(v, vmin), vmax) - vmin) / (vmax - vmin) if vmax > vmin else 0.5)
            col = "white" if 0.299 * r + 0.587 * g + 0.114 * bl < 0.5 else "black"
            if ci0 is not None and ci0[a, b]:
                col = "0.6"
            ax.text(b, a, format(v, fmt).replace("0.", ".", 1), ha="center", va="center", fontsize=4.6, color=col)
    ax.set_xticks(range(len(xs)), [f"{x:g}" for x in xs], fontsize=5.5)
    ax.set_yticks(range(len(ys)), [f"{y:g}" for y in ys], fontsize=5.5)
    return im


# ------------------------------------------------------------------ figures
def fig_votes(t, sim, out, name, gate, x):
    fig, axes = plt.subplots(2, 4, figsize=(7.4, 3.6), sharex=True, sharey=True)
    for j, rho in enumerate(RHOS):
        for arm in ("silent", "truth", "false"):
            d = sel(t, sim=sim, arm=arm, q=Q, rho=rho) if arm == "silent" else \
                sel(t, sim=sim, arm=arm, q=Q, qc=QC, x=x, rho=rho, gate=gate)
            d = pd.concat([pd.DataFrame({"t": [0.0], "x0": [0.0], "x2": [0.0]}), d.sort_values("t")])  # nobody has voted at t = 0
            for i, col in enumerate(("x0", "x2")):
                axes[i, j].plot(d.t, d[col], color=ARMC[arm], lw=1.2, ls=":" if arm == "silent" else "-",
                                label=ARML[arm] if (i, j) == (0, 0) else None)
        for i, lab in enumerate(("A0", "A2")):
            ax = axes[i, j]
            ax.set_ylim(0, 1)
            if sim == "sim2":
                ax.axvline(30, color="0.7", ls="--", lw=0.6)
            if i == 0:
                ax.set_title(f"$\\rho = {rho:g}$")
            if i == 1:
                ax.set_xlabel(TNAME[sim])
            if j == 0:
                ax.set_ylabel(f"share voting {lab}")
    axes[0, 0].legend(loc="center right")
    fig.tight_layout(pad=0.3)
    save(fig, out, name)


def fig_gain_time(t, sim, out, name, gate):
    fig, axes = plt.subplots(2, 4, figsize=(7.4, 3.6), sharex=True, sharey="row")
    for j, rho in enumerate(RHOS):
        for i, arm in enumerate(("truth", "false")):
            ax = axes[i, j]
            for k, x in enumerate(XS[sim]):
                d = sel(t, sim=sim, arm=arm, q=Q, qc=QC, x=x, rho=rho, gate=gate).sort_values("t")
                d = pd.concat([pd.DataFrame({"t": [0.0], "G": [0.0], "G_se": [0.0]}), d])          # all arms identical at t = 0
                ax.plot(d.t, d.G, color=SCOL[k], lw=1)
                ax.fill_between(d.t, d.G - 1.96 * d.G_se, d.G + 1.96 * d.G_se, color=SCOL[k], alpha=.15, lw=0)
            ax.axhline(0, color="0.6", lw=0.5)
            if sim == "sim2":
                ax.axvline(30, color="0.7", ls="--", lw=0.6)
            if i == 0:
                ax.set_title(f"$\\rho = {rho:g}$")
            if i == 1:
                ax.set_xlabel(TNAME[sim])
            if j == 0:
                ax.set_ylabel(f"$G_t$, {ARML[arm]}")
    strength_legend(axes[1, 0], sim)
    fig.tight_layout(pad=0.3)
    save(fig, out, name)


def phase_panels(sim, getter, fmt, vmin, vmax, cmap, clabel, xmain, out, name, ci=False):
    """(a) qc x strength at rho_rep, q 6; (b) rho x strength at q 6, qc 12; (c) rho x q at qc 12, main strength."""
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.2), gridspec_kw={"width_ratios": [9, 9, 3.6]})
    xs = XS[sim]
    qcs, qs = [3, 6, 12, 24], [3, 6, 12]
    panels = [("a", [(qc, x) for qc in qcs for x in xs], qcs, xs, "$q_c$", XNAME[sim],
               lambda qc, x: dict(q=Q, qc=qc, x=x, rho=RHO_REP), f"$\\rho = {RHO_REP:g}$, $q = {Q}$"),
              ("b", [(r, x) for r in RHOS[::-1] for x in xs], RHOS[::-1], xs, "$\\rho$", XNAME[sim],
               lambda r, x: dict(q=Q, qc=QC, x=x, rho=r), f"$q = {Q}$, $q_c = {QC}$"),
              ("c", [(r, q) for r in RHOS[::-1] for q in qs], RHOS[::-1], qs, "$\\rho$", "$q$",
               lambda r, q: dict(q=q, qc=QC, x=xmain, rho=r), f"$q_c = {QC}$, {label_x(sim, xmain)}")]
    for ax, (tag, cells, ys, xs_, ylab, xlab, kw, title) in zip(axes, panels):
        m = np.full((len(ys), len(xs_)), np.nan)
        c0 = np.zeros_like(m, bool)
        for (yv, xv) in cells:
            v, z = getter(**kw(yv, xv))
            m[ys.index(yv), xs_.index(xv)] = v
            c0[ys.index(yv), xs_.index(xv)] = z
        im = tile(ax, m, xs_, ys, fmt, vmin, vmax, cmap, c0 if ci else None)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_title(f"({tag}) {title}", fontsize=7)
    fig.subplots_adjust(wspace=0.32)
    fig.colorbar(im, ax=axes, shrink=0.9, pad=0.01, label=clabel)
    save(fig, out, name)


def fig_gain_strength(t, sim, out, name):
    fig, axes = plt.subplots(2, 4, figsize=(7.4, 3.8), sharex=True)
    cols = [(30.0, r) for r in RHOS] if sim == "sim1" else [(5.0, 0.5), (30.0, 0.5), (5.0, 0.75), (30.0, 0.75)]
    xs = np.array(XS[sim], float)
    for j, (tt, rho) in enumerate(cols):
        for gate, ls, mk in (("on", "-", "o"), ("off", "--", "s")):
            d = sel(t, sim=sim, arm="false", q=Q, qc=QC, rho=rho, gate=gate, t=tt).set_index("x").reindex(xs)
            axes[0, j].errorbar(xs, d.G, yerr=1.96 * d.G_se, color="tab:red", ls=ls, marker=mk, ms=2.5, lw=1,
                                capsize=1.5, label=f"gate {gate}")
            slope = np.diff(d.G.to_numpy()) / np.diff(xs)
            axes[1, j].plot(0.5 * (xs[1:] + xs[:-1]), slope, color="tab:red", ls=ls, marker=mk, ms=2.5, lw=1)
        axes[0, j].set_title(f"{TNAME[sim] if sim == 'sim1' else '$t$'} {tt:g}, $\\rho = {rho:g}$".replace("time $t$", "day"))
        axes[1, j].axhline(0, color="0.6", lw=0.5)
        axes[1, j].set_xlabel(XNAME[sim])
    axes[0, 0].set_ylabel("$G$, target A2")
    axes[1, 0].set_ylabel(f"slope $\\Delta G/\\Delta${XNAME[sim]}")
    axes[0, 0].legend(fontsize=6)
    fig.tight_layout(pad=0.3)
    save(fig, out, name)


def fig_gain_per_fact(t, sim, out, name, gate):
    fig, axes = plt.subplots(2, 4, figsize=(7.4, 3.6), sharex=True, sharey="row")
    for j, rho in enumerate(RHOS):
        for i, arm in enumerate(("truth", "false")):
            ax = axes[i, j]
            for k, x in enumerate(XS[sim]):
                d = sel(t, sim=sim, arm=arm, q=Q, qc=QC, x=x, rho=rho, gate=gate).sort_values("t")
                d = d[d.facts >= 0.5]
                ax.plot(d.t, d.G / d.facts, color=SCOL[k], lw=1)
            ax.axhline(0, color="0.6", lw=0.5)
            ax.set_yscale("symlog", linthresh=0.003)
            if sim == "sim2":
                ax.axvline(30, color="0.7", ls="--", lw=0.6)
            if i == 0:
                ax.set_title(f"$\\rho = {rho:g}$")
            if i == 1:
                ax.set_xlabel(TNAME[sim])
            if j == 0:
                ax.set_ylabel(f"$G_t/F_t$, {ARML[arm]}")
    strength_legend(axes[1, 0], sim)
    fig.tight_layout(pad=0.3)
    save(fig, out, name)


def fig_proofs(t, sim, out, name, gate, x):
    fig, axes = plt.subplots(2, 4, figsize=(7.4, 3.6), sharex=True, sharey=True)
    for j, rho in enumerate(RHOS):
        for arm in ("silent", "truth", "false"):
            d = sel(t, sim=sim, arm=arm, q=Q, rho=rho) if arm == "silent" else \
                sel(t, sim=sim, arm=arm, q=Q, qc=QC, x=x, rho=rho, gate=gate)
            d = d.sort_values("t")
            for i, col in enumerate(("proof_any", "proof_own")):
                axes[i, j].plot(d.t, d[col], color=ARMC[arm], lw=1.2, ls=":" if arm == "silent" else "-",
                                label=ARML[arm] if (i, j) == (0, 0) else None)
        for i in range(2):
            axes[i, j].set_ylim(0, 1)
            if sim == "sim2":
                axes[i, j].axvline(30, color="0.7", ls="--", lw=0.6)
        axes[0, j].set_title(f"$\\rho = {rho:g}$")
        axes[1, j].set_xlabel(TNAME[sim])
    axes[0, 0].set_ylabel("proves A0,\nany facts")
    axes[1, 0].set_ylabel("proves A0,\nown facts only")
    axes[0, 0].legend(loc="upper left")
    fig.tight_layout(pad=0.3)
    save(fig, out, name)


def fig_info_time(mi, sim, out, name, state, gate, rows=("I",)):
    lab = {"I": "$I(Z;X)$ (bits)", "eta": "$\\eta$"}
    fig, axes = plt.subplots(len(rows), 4, figsize=(7.4, 1.85 * len(rows) + 0.2), sharex=True, sharey="row", squeeze=False)
    for i, qty in enumerate(rows):
        for j, rho in enumerate(RHOS):
            ax = axes[i, j]
            for k, x in enumerate(XS[sim]):
                d = sel(mi, sim=sim, q=Q, qc=QC, x=x, rho=rho, gate=gate, state=state).sort_values("t")
                ax.plot(d.t, d[qty], color=SCOL[k], lw=1)
            if sim == "sim2":
                ax.axvline(30, color="0.7", ls="--", lw=0.6)
            if qty == "eta":
                ax.set_ylim(0, 1.02)
            if i == 0:
                ax.set_title(f"$\\rho = {rho:g}$")
            if i == len(rows) - 1:
                ax.set_xlabel(TNAME[sim])
            if j == 0:
                ax.set_ylabel(lab[qty])
    strength_legend(axes[0, 0], sim)
    fig.tight_layout(pad=0.3)
    save(fig, out, name)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2026-10-15")
    a = ap.parse_args()
    t = pd.read_parquet(RES / a.date / "cells_time.parquet")
    mi_file = RES / a.date / "mi_table.csv"
    if mi_file.exists() and mi_file.stat().st_mtime > (RES / a.date / "cells_time.parquet").stat().st_mtime:
        mi = pd.read_csv(mi_file)
    else:
        mi = mi_table(t)
        mi.to_csv(mi_file, index=False)
        key = ["sim", "q", "qc", "x", "rho", "gate", "t", "state"]
        sub = t[t.t == T_PHASE]
        sens = mi_table(sub, 1.0)[key + ["I", "C", "eta"]].merge(mi_table(sub, 0.0)[key + ["I", "C", "eta"]], on=key,
                                                                 suffixes=("_a1", "_a0"))
        sens.merge(mi[mi.t == T_PHASE][key + ["I", "C", "eta"]], on=key).to_csv(RES / a.date / "mi_smoothing_sensitivity.csv", index=False)
    # one colour range per quantity, shared by the two simulations (mirrored figures)
    at30 = t[(t.t == T_PHASE) & (t.arm == "false")]
    gmax = float(np.nanmax(np.abs(at30.G)))
    imax = {st: float(np.nanmax(mi[(mi.t == T_PHASE) & (mi.state == st)].I)) for st in STATES}
    for sim in ("sim1", "sim2"):
        out = REPORT[sim] / f"{a.date}_report1_symmetric" / "figures"
        out.mkdir(parents=True, exist_ok=True)
        ts, ms = t[t.sim == sim], mi[mi.sim == sim]

        def gain_get(gate):
            def g(**kw):
                d = sel(ts, arm="false", gate=gate, t=T_PHASE, **kw)
                if d.empty:
                    return np.nan, False
                r = d.iloc[0]
                return r.G, abs(r.G) <= 1.96 * r.G_se
            return g

        def info_get(gate, state, qty):
            def g(**kw):
                d = sel(ms, gate=gate, t=T_PHASE, state=state, **kw)
                return (np.nan, False) if d.empty else (d.iloc[0][qty], False)
            return g

        for gate, suffix in (("on", ""), ("off", "_gateoff")):
            fig_votes(ts, sim, out, f"fig_votes{suffix}.pdf", gate, MAIN[sim])
            fig_gain_time(ts, sim, out, f"fig_gain_time{suffix}.pdf", gate)
            phase_panels(sim, gain_get(gate), ".2f", -gmax, gmax, "RdBu_r", "$G$ at $t = 30$, target A2",
                         MAIN[sim], out, f"fig_phase_gain{suffix}.pdf", ci=True)
            for st in STATES:
                tag = "" if st == "vote" else f"_{st}"
                phase_panels(sim, info_get(gate, st, "I"), ".2f", 0, imax[st], "viridis", "$I(Z;X)$, bits",
                             MAIN[sim], out, f"fig_phase_I{tag}{suffix}.pdf")
                phase_panels(sim, info_get(gate, st, "eta"), ".2f", 0, 1, "viridis", "$\\eta$",
                             MAIN[sim], out, f"fig_phase_eta{tag}{suffix}.pdf")
                if st == "vote":
                    fig_info_time(ms, sim, out, f"fig_I_time{suffix}.pdf", st, gate, ("I",))
                    fig_info_time(ms, sim, out, f"fig_eta_time{suffix}.pdf", st, gate, ("eta",))
                fig_info_time(ms, sim, out, f"fig_Ieta_time{tag or '_vote'}{suffix}.pdf", st, gate, ("I", "eta"))
        fig_gain_strength(ts, sim, out, "fig_gain_strength.pdf")
        fig_gain_per_fact(ts, sim, out, "fig_gain_per_fact.pdf", "on")
        fig_proofs(ts, sim, out, "fig_proofs.pdf", "on", MAIN[sim])
        fig_votes(ts, sim, out, "fig_votes_strong.pdf", "on", STRONG[sim])
        fig_proofs(ts, sim, out, "fig_proofs_strong.pdf", "on", STRONG[sim])
        print("figures ->", out)


if __name__ == "__main__":
    main()
