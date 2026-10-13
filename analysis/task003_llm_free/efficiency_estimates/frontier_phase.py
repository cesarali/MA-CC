"""Phase diagrams of the frontier measures (q 6, plurality, proof stop on): over qc x rate
(Simulation 2, gate on and off) and qc x b (Simulation 1). Tiles: change in P(target wins) (Delta),
share of the available room (Delta / (1 - silent success)), facts posted and eta_cost, for each target.

Usage, from the repository root (after frontier.py):
    .venv/bin/python analysis/task003_llm_free/efficiency_estimates/frontier_phase.py --out <folder> [--date 2026-10-13]
"""
from __future__ import annotations
import argparse, pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from frontier import RES  # noqa: E402

plt.rcParams.update({"font.size": 6.5, "axes.titlesize": 6.5})
QCS = [3, 6, 12, 24]
MEASURES = [("delta", "Δ P(target wins)", "RdBu", None, ".2f"), ("room_share", "share of room", "RdBu", (-1, 1), ".2f"),
            ("facts", "facts posted", "Greys", None, ".0f"), ("eta_cost", "η_cost", "viridis", (0, 1), ".2f")]


def tiles(ax, m, xs, fmt, cmap, vmin, vmax):
    im = ax.imshow(m, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    cm = plt.get_cmap(cmap)
    for a in range(m.shape[0]):
        for b in range(m.shape[1]):
            v = m[a, b]
            if np.isnan(v):
                ax.text(b, a, "–", ha="center", va="center", fontsize=5)
                continue
            r, g, bl, _ = cm((min(max(v, vmin), vmax) - vmin) / (vmax - vmin) if vmax > vmin else 0.5)
            ax.text(b, a, format(v, fmt), ha="center", va="center", fontsize=4.6,
                    color="white" if 0.299 * r + 0.587 * g + 0.114 * bl < 0.5 else "black")
    ax.set_xticks(range(len(xs)), [f"{x:g}" for x in xs], fontsize=5)
    ax.set_yticks(range(len(QCS)), QCS, fontsize=5)
    return im


def figure(p, sim, setup, T, out, name):
    xs = [0.5, 1.0, 2.0, 4.0, 8.0] if sim == "sim2" else [1, 2, 3, 6, 9]
    g = p[(p.sim == sim) & (p.setup == setup) & (p["T"] == T)]
    rows = [(rho, gate) for rho in (0.75, 1.0) for gate in (["on", "off"] if sim == "sim2" else ["on"])]
    if setup == "task003_symmetric":
        rows = [r for r in rows if r[0] == 0.75]
    cols = [(m, arm) for m in MEASURES for arm in ("truth", "false")]
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(1.05 * len(cols) + 0.6, 1.05 * len(rows) + 0.5), squeeze=False)
    for j, ((col, lab, cmap, rng, fmt), arm) in enumerate(cols):
        sub = g[g.arm == arm]
        if rng is None:
            v = sub[col].replace([np.inf, -np.inf], np.nan)
            lo, hi = float(np.nanmin(v)) if v.notna().any() else 0, float(np.nanmax(v)) if v.notna().any() else 1
            if col == "delta":
                m_ = max(abs(lo), abs(hi), 1e-3)
                lo, hi = -m_, m_
        else:
            lo, hi = rng
        for i, (rho, gate) in enumerate(rows):
            m = np.full((len(QCS), len(xs)), np.nan)
            for r in sub[(sub.rho == rho) & (sub.gate == gate)].itertuples():
                v = getattr(r, col)
                if np.isfinite(v):
                    m[QCS.index(r.qc), xs.index(r.x)] = v
            ax = axes[i, j]
            tiles(ax, m, xs, fmt, cmap, lo, hi)
            if i == 0:
                ax.set_title(f"{lab}\ntarget {'A0' if arm == 'truth' else 'A2'}")
            if i == len(rows) - 1:
                ax.set_xlabel("λc" if sim == "sim2" else "b", fontsize=6)
            if j == 0:
                ax.set_ylabel(f"ρ {rho:g}" + (f", gate {gate}" if sim == "sim2" else "") + "\nqc", fontsize=6)
    fig.tight_layout(pad=0.3)
    fig.savefig(out / name)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2026-10-13")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    p = pd.read_csv(RES / "efficiency_estimates" / a.date / "frontier_points.csv")
    p = p[(p.q == 6) & (p.rule == "plurality") & (p.stop == "on") & (p.outcome == "target")]
    for sim in ("sim1", "sim2"):
        for setup in ("task003_nosolution", "task003_symmetric"):
            figure(p, sim, setup, 30, out, f"phase_{sim}_{setup[8:]}_T30.pdf")
    figure(p, "sim2", "task003_nosolution", 5, out, "phase_sim2_nosolution_T5.pdf")
    print("phase diagrams in", out)


if __name__ == "__main__":
    main()
