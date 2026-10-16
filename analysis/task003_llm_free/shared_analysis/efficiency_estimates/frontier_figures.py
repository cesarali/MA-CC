"""Figures for the frontier analysis: success gain against facts posted (q 6), each configuration
with its 95% interval, the best-achievable curve over tested configurations (solid staircase) and the
curve reachable by random mixing of configurations per run (dashed: concave hull through (0, 0)).

Usage, from the repository root (after frontier.py):
    .venv/bin/python analysis/task003_llm_free/shared_analysis/efficiency_estimates/frontier_figures.py [--date 2026-10-13]
"""
from __future__ import annotations
import argparse, pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from frontier import RES, hull  # noqa: E402

plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "legend.fontsize": 6})
COL = {"truth": "tab:blue", "false": "tab:red"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2026-10-13")
    a = ap.parse_args()
    d = RES / "efficiency_estimates" / a.date
    p = pd.read_csv(d / "frontier_points.csv")
    p = p[(p.q == 6) & (p.rule == "plurality") & (p.stop == "on")]
    panels = [(s, st, r) for s in ("sim1", "sim2") for st in ("task003_nosolution", "task003_symmetric") for r in (0.75, 1.0)]
    for T in (30, 5):
        for outcome in ("target", "task"):
            fig, axes = plt.subplots(2, 4, figsize=(10, 5))
            for ax, (sim, setup, rho) in zip(axes.flat, panels):
                for arm in ("truth", "false"):
                    g = p[(p.sim == sim) & (p.setup == setup) & (p.rho == rho) & (p.arm == arm) & (p.outcome == outcome) & (p["T"] == T)]
                    if g.empty:
                        continue
                    for gate, mk in (("on", "o"), ("off", "s")):
                        h = g[g.gate == gate]
                        ax.errorbar(h.facts, h.delta, yerr=[h.delta - h.delta_lo, h.delta_hi - h.delta], fmt=mk, ms=2.5,
                                    color=COL[arm], alpha=0.6, lw=0.6, capsize=0,
                                    label=f"target {'A0' if arm == 'truth' else 'A2'}" + (f", gate {gate}" if sim == "sim2" else "")
                                    if (gate == "on" or sim == "sim2") else None)
                    c, dl = g.facts.to_numpy(), g.delta.to_numpy()
                    order = np.argsort(c)
                    stair = np.maximum.accumulate(np.maximum(dl[order], 0))
                    ax.step(np.r_[0, c[order]], np.r_[0, stair], where="post", color=COL[arm], lw=1)
                    xs, ys = hull(dl, c)
                    ax.plot(xs, ys, color=COL[arm], lw=0.8, ls="--")
                ax.axhline(0, color="0.6", lw=0.5)
                ax.set_xscale("symlog", linthresh=1)
                ax.set_title(f"{'Simulation 1' if sim == 'sim1' else 'Simulation 2'}, {setup[8:]}, ρ = {rho:g}")
                ax.set_xlabel("facts posted (mean per run)")
                ax.set_ylabel(f"Δ P({'target' if outcome == 'target' else 'correct'} wins)")
            axes.flat[0].legend(loc="lower right")
            fig.suptitle(f"{'Target attainment' if outcome == 'target' else 'Task correctness'} gain against facts posted, "
                         f"t = {T}, q = 6. Solid: best-achievable curve (tested configurations); dashed: with random mixing per run.", fontsize=8)
            fig.tight_layout()
            fig.savefig(d / f"frontier_{outcome}_T{T}.png", dpi=140)
            plt.close(fig)
    print("figures in", d)


if __name__ == "__main__":
    main()
