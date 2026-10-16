"""Figures for simulation_2/analysis/reports/2026-10-16_sham_baseline_symmetric (sham-baseline measures).

Reads results/symmetric_fine/<date>/sham_table.parquet (sham_baseline.py). Writes:
  fig_time_rho{1,0.75,0.5,0.25}.pdf   2 x 4 per memory rho: top chi, eta_post, eta_info, eta_goal; bottom I(Z;X),
                                      I(Z;Y), C_X; one curve per rate; q 6, qc 12, gate on
  fig_phase_rho_q.pdf, fig_phase_rho_x.pdf, fig_phase_qc_x.pdf   2 x 4 tiles at t = 30, same seven quantities

Usage, from the repository root:
    .venv/bin/python analysis/task003_llm_free/symmetric_fine/sham_baseline_figures.py
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

OUT = M.ROOT / "simulation_2/analysis/reports/2026-10-16_sham_baseline_symmetric/figures"
TABLE = M.RES / "2026-10-15" / "sham_table.parquet"
XS = M.XS["sim2"]
Q, QC, T = 6, 12, 30.0
QTY = [("chi", "$\\chi_{\\rm correct}$", "RdBu_r"), ("eta_post", "$\\eta_{\\rm post}$ (per fact)", "magma_r"),
       ("eta_info", "$\\eta_{\\rm info}$", "viridis"), ("eta_goal", "$\\eta_{\\rm goal}$", "viridis"),
       ("I_X", "$I(Z;X)$ (bits)", "viridis"), ("I_Y", "$I(Z;Y)$ (bits)", "viridis"), ("C_X", "$C_X$ (bits)", "viridis")]
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "legend.fontsize": 6.5, "figure.dpi": 150})


def ranges(t):
    s = t[t.rho < 1]
    r = {}
    for c, _, _ in QTY:
        v = s[c].replace([np.inf, -np.inf], np.nan)
        r[c] = (0.0, float(np.nanmax(v)))
    r["chi"] = (-r["chi"][1], r["chi"][1])
    r["eta_info"] = r["eta_goal"] = (0.0, 1.0)
    return r


def fig_time(t, rho, rng):
    fig, axes = plt.subplots(2, 4, figsize=(7.6, 3.9), sharex=True)
    for k, x in enumerate(XS):
        d = t[(t.q == Q) & (t.qc == QC) & (t.x == x) & (t.rho == rho) & (t.gate == "on")].sort_values("t")
        z = pd.DataFrame({"t": [0.0], "chi": [0.0], "chi_se": [0.0]})
        dz = pd.concat([z, d])
        for ax, (c, lab, _) in zip(list(axes[0]) + list(axes[1][:3]), QTY):
            src = dz if c == "chi" else d
            ax.plot(src.t, src[c], color=M.SCOL[k], lw=1)
            if c == "chi":
                ax.fill_between(src.t, src.chi - 1.96 * src.chi_se, src.chi + 1.96 * src.chi_se, color=M.SCOL[k],
                                alpha=.12, lw=0)
    for ax, (c, lab, _) in zip(list(axes[0]) + list(axes[1][:3]), QTY):
        ax.set_title(lab)
        ax.axvline(30, color="0.7", ls="--", lw=0.6)
        lo, hi = rng[c]
        ax.set_ylim(min(0, lo) if c != "chi" else min(0, -0.02), hi * 1.05 if hi > 0 else 1)
    for ax in axes[1]:
        ax.set_xlabel("time $t$")
    axes[1, 3].axis("off")
    for i, x in enumerate(XS):
        axes[1, 3].plot([], [], color=M.SCOL[i], lw=1.2, label=f"{x:g}")
    axes[1, 3].legend(title="$\\lambda_c$", loc="center", ncol=2, fontsize=6.5, title_fontsize=7)
    fig.suptitle(f"$\\rho = {rho:g}$", y=1.0, fontsize=9)
    fig.tight_layout(pad=0.3)
    M.save(fig, OUT, f"fig_time_rho{rho:g}.pdf")


def fig_phase(t, name, ys, xs, ylab, xlab, kw, title, rng):
    rng = ranges(t[t.t == T])            # colour ranges from t = 30 values only, shared by the three diagrams
    fig, axes = plt.subplots(2, 4, figsize=(7.8, 4.0))
    s = t[(t.t == T) & (t.gate == "on")]
    for ax, (c, lab, cmap) in zip(list(axes[0]) + list(axes[1][:3]), QTY):
        m = np.full((len(ys), len(xs)), np.nan)
        c0 = np.zeros_like(m, bool)
        for a, yv in enumerate(ys):
            for b, xv in enumerate(xs):
                d = M.sel(s, **kw(yv, xv))
                if len(d):
                    m[a, b] = d.iloc[0][c]
                    if c == "chi":
                        c0[a, b] = abs(d.iloc[0].chi) <= 1.96 * d.iloc[0].chi_se
        lo, hi = rng[c]
        fmt = ".3f" if c in ("eta_post", "I_X", "I_Y", "C_X") else ".2f"
        im = M.tile(ax, m, xs, ys, fmt, lo, hi, cmap, c0 if c == "chi" else None)
        ax.set_title(lab, fontsize=7.5)
        ax.set_xlabel(xlab, fontsize=7)
        ax.set_ylabel(ylab, fontsize=7)
        fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    axes[1, 3].axis("off")
    fig.suptitle(title, fontsize=8, y=1.0)
    fig.tight_layout(pad=0.3)
    M.save(fig, OUT, name)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t = pd.read_parquet(TABLE)
    rng = ranges(t)
    for rho in M.RHOS:
        fig_time(t, rho, rng)
    rhos = M.RHOS[::-1]
    fig_phase(t, "fig_phase_rho_q.pdf", rhos, [3, 6, 12], "$\\rho$", "$q$",
              lambda r, q: dict(q=q, qc=QC, x=1, rho=r), "$q_c = 12$, $\\lambda_c = 1$, $t = 30$", rng)
    fig_phase(t, "fig_phase_rho_x.pdf", rhos, XS, "$\\rho$", "$\\lambda_c$",
              lambda r, x: dict(q=Q, qc=QC, x=x, rho=r), "$q = 6$, $q_c = 12$, $t = 30$", rng)
    fig_phase(t, "fig_phase_qc_x.pdf", [3, 6, 12, 24], XS, "$q_c$", "$\\lambda_c$",
              lambda qc, x: dict(q=Q, qc=qc, x=x, rho=0.5), "$\\rho = 0.5$, $q = 6$, $t = 30$", rng)
    print("figures ->", OUT)


if __name__ == "__main__":
    main()
