"""Q2 figures, drawn from the tables q2_compute.py writes (ANALYSIS_PLAN.md §8).

F1  paired gain over days: own target, one line per b at qc = 12 (the clean gate),
    and one line per qc at b = 3; panels rho (rows) x q (columns).
F1b shared part M and directional part D for truth (a = 0) over days, lines per b.
F2  vote shares over days for silent, truth and false at qc = 12, b = 3.
F3  final-day phase diagrams: a 4 x 5 grid of equal tiles, qc (x) by b (y), the value
    printed in each tile; panels q (columns) x rho (rows). Quantities: own-target gain
    (tiles whose 95% interval includes 0 are hatched), delivered dose, target share,
    A1 share; and D for truth. A line separates qc in {3, 6} (stricter effective gate)
    from {12, 24}.

From the repository root:
    .venv/bin/python analysis/task003_llm_free/simulation_1/analysis/q2_figures.py
"""
from __future__ import annotations
import pathlib, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm, TwoSlopeNorm  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import VARIANT_LABEL, VARIANTS, out_dir, write_index  # noqa: E402

SETUP_LABEL = {"task003_nosolution": "task003-nosolution", "task003_symmetric": "task003-symmetric"}
ARM_LABEL = {"truth": "truth controller (target A0)", "false": "false controller (target A2)"}
TARGET = {"truth": 0, "false": 2}
QS, RHOS, QCS, BS = (3, 6, 12), (0.75, 1.0), (3, 6, 12, 24), (1, 2, 3, 6, 9)
B_COLOR = dict(zip(BS, plt.cm.viridis(np.linspace(0, 0.9, 5))))
QC_COLOR = dict(zip(QCS, plt.cm.plasma(np.linspace(0, 0.85, 4))))
ARM_COLOR = {"silent": "#777777", "truth": "#1b6ca8", "false": "#d1495b"}


def panels(title):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
    fig.suptitle(title, fontsize=11)
    return fig, axes


def save(fig, d, name, entries, what, cells):
    fig.tight_layout()
    fig.savefig(d / "figures" / name, dpi=115)
    plt.close(fig)
    entries.append({"question": "Q2", "file": f"figures/{name}", "what": what, "cells": cells})


def lines(ax, t, color, label):
    ax.plot(t.day, t["mean"], color=color, lw=1.6, label=label)
    ax.fill_between(t.day, t.lo, t.hi, color=color, alpha=0.18, lw=0)


def main():
    d = out_dir()
    T = pd.read_parquet(d / "tables" / "q2_time.parquet")
    F = pd.read_csv(d / "tables" / "q2_final.csv")
    entries = []
    sel = lambda df, **kw: df.loc[np.logical_and.reduce([df[k] == v for k, v in kw.items()])]

    for setup in SETUP_LABEL:
        for variant in VARIANTS:
            # ---- F1: own-target gain over days --------------------------------------------
            for arm in ("truth", "false"):
                a = TARGET[arm]
                for along, fixed, values, colors in (("b", ("qc", 12), BS, B_COLOR), ("qc", ("b", 3), QCS, QC_COLOR)):
                    fig, axes = panels(f"Q2 · {SETUP_LABEL[setup]} · {VARIANT_LABEL[variant]} · {ARM_LABEL[arm]}: "
                                       f"paired gain in A{a} share over days, by {along} ({fixed[0]} = {fixed[1]})")
                    for i, rho in enumerate(RHOS):
                        for j, q in enumerate(QS):
                            ax = axes[i, j]
                            for v in values:
                                t = sel(T, variant=variant, setup=setup, arm=arm, quantity="G", a=a, q=q, rho=rho,
                                        **{along: v, fixed[0]: fixed[1]})
                                lines(ax, t, colors[v], f"{along} = {v}")
                            ax.axhline(0, color="k", lw=0.7)
                            ax.set_title(f"ρ = {rho}, q = {q}", fontsize=9)
                    axes[0, 0].legend(fontsize=7)
                    for ax in axes[-1]:
                        ax.set_xlabel("day")
                    for ax in axes[:, 0]:
                        ax.set_ylabel(f"G (A{a} share) vs silent")
                    save(fig, d, f"q2_gain_time_{setup}_{variant}_{arm}_by_{along}.png", entries,
                         f"own-target paired gain over days, lines by {along}", f"{setup} {variant} {arm}")

            # ---- F1b: M and D for truth support over days -------------------------------------
            for qty in ("M", "D"):
                fig, axes = panels(f"Q2 · {SETUP_LABEL[setup]} · {VARIANT_LABEL[variant]}: {qty} for A0 over days "
                                   f"({'shared part (G⁺+G⁻)/2' if qty == 'M' else 'directional part (G⁺−G⁻)/2'}), qc = 12")
                for i, rho in enumerate(RHOS):
                    for j, q in enumerate(QS):
                        ax = axes[i, j]
                        for b in BS:
                            t = sel(T, variant=variant, setup=setup, arm="both", quantity=qty, a=0, q=q, rho=rho, qc=12, b=b)
                            lines(ax, t, B_COLOR[b], f"b = {b}")
                        ax.axhline(0, color="k", lw=0.7)
                        ax.set_title(f"ρ = {rho}, q = {q}", fontsize=9)
                axes[0, 0].legend(fontsize=7)
                save(fig, d, f"q2_{qty}_time_{setup}_{variant}.png", entries, f"{qty} for A0 over days, lines by b",
                     f"{setup} {variant}")

            # ---- F2: vote shares over days, silent vs truth vs false --------------------------
            fig, axes = panels(f"Q2 · {SETUP_LABEL[setup]} · {VARIANT_LABEL[variant]}: share A0 (solid) and A2 "
                               f"(dashed) over days, qc = 12, b = 3")
            for i, rho in enumerate(RHOS):
                for j, q in enumerate(QS):
                    ax = axes[i, j]
                    for k, style in ((0, "-"), (2, "--")):
                        s = sel(T, variant=variant, setup=setup, quantity="share_silent", a=k, q=q, rho=rho, qc=12, b=3)
                        ax.plot(s.day, s["mean"], color=ARM_COLOR["silent"], ls=style, lw=1.5,
                                label=f"silent A{k}" if (i, j) == (0, 0) else None)
                        for arm in ("truth", "false"):
                            g = sel(T, variant=variant, setup=setup, arm=arm, quantity="G", a=k, q=q, rho=rho, qc=12, b=3)
                            ax.plot(g.day, s["mean"].to_numpy() + g["mean"].to_numpy(), color=ARM_COLOR[arm], ls=style,
                                    lw=1.5, label=f"{arm} A{k}" if (i, j) == (0, 0) else None)
                    ax.set_ylim(0, 1)
                    ax.set_title(f"ρ = {rho}, q = {q}", fontsize=9)
            axes[0, 0].legend(fontsize=7, ncol=2)
            save(fig, d, f"q2_shares_time_{setup}_{variant}.png", entries, "vote shares over days per arm",
                 f"{setup} {variant} qc12 b3")

            # ---- F3: phase diagrams ----------------------------------------------------------
            specs = [("own_gain_30", "own-target paired gain, day 30", "gain"),
                     ("dose_29", "facts actually posted per episode", "dose"),
                     ("target_share_30", "target's vote share, day 30", "share"),
                     ("A1_share_30", "A1 vote share, day 30", "share")]
            for arm in ("truth", "false"):
                sub = F[(F.variant == variant) & (F.setup == setup) & (F.arm == arm)]
                for col, title, kind in specs:
                    phase(d, entries, sub, col, kind,
                          f"Q2 · {SETUP_LABEL[setup]} · {VARIANT_LABEL[variant]} · {ARM_LABEL[arm]}: {title}",
                          f"q2_phase_{col}_{setup}_{variant}_{arm}.png", f"{setup} {variant} {arm}")
            sub = F[(F.variant == variant) & (F.setup == setup) & (F.arm == "both") & (F.target == 0)].rename(
                columns={"D_30_lo": "own_gain_30_lo", "D_30_hi": "own_gain_30_hi"})
            phase(d, entries, sub, "D_30", "gain",
                  f"Q2 · {SETUP_LABEL[setup]} · {VARIANT_LABEL[variant]}: directional part D for A0, day 30",
                  f"q2_phase_D_30_{setup}_{variant}.png", f"{setup} {variant}")
    write_index(d, entries)
    print(f"{len(entries)} figures written")


def phase(d, entries, sub, col, kind, title, name, cells):
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    fig.suptitle(title + "  (x = qc, y = b; tiles equal size, not to scale)", fontsize=11)
    if kind == "gain":
        norm, cmap = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1), "RdBu_r"
    elif kind == "dose":
        norm, cmap = LogNorm(vmin=0.3, vmax=300), "magma_r"
    else:
        norm, cmap = plt.Normalize(0, 1), "viridis"
    im = None
    for i, rho in enumerate(RHOS):
        for j, q in enumerate(QS):
            ax = axes[i, j]
            grid = np.full((len(BS), len(QCS)), np.nan)
            null = np.zeros_like(grid, dtype=bool)
            for r in sub[(sub.q == q) & (sub.rho == rho)].itertuples():
                y, x = BS.index(int(r.b)), QCS.index(int(r.qc))
                grid[y, x] = getattr(r, col)
                if kind == "gain":
                    null[y, x] = r.own_gain_30_lo <= 0 <= r.own_gain_30_hi
            shown = np.where(np.isnan(grid), np.nan, np.maximum(grid, 0.3)) if kind == "dose" else grid
            im = ax.imshow(shown, origin="lower", cmap=cmap, norm=norm, aspect="auto")
            for y in range(len(BS)):
                for x in range(len(QCS)):
                    v = grid[y, x]
                    if np.isnan(v):
                        continue
                    txt = f"{v:.0f}" if kind == "dose" else f"{v:.2f}"
                    ax.text(x, y, txt, ha="center", va="center", fontsize=7.5,
                            color="#999999" if null[y, x] else "black")
                    if null[y, x]:
                        ax.add_patch(plt.Rectangle((x - .5, y - .5), 1, 1, fill=False, hatch="///", lw=0,
                                                   edgecolor="#bbbbbb"))
            ax.axvline(1.5, color="k", lw=1.2)
            ax.set_xticks(range(len(QCS)), [str(v) for v in QCS])
            ax.set_yticks(range(len(BS)), [str(v) for v in BS])
            ax.set_title(f"ρ = {rho}, q = {q}", fontsize=9)
            if i == 1:
                ax.set_xlabel("qc  (left of line: stricter effective gate)")
            if j == 0:
                ax.set_ylabel("b")
    fig.colorbar(im, ax=axes, shrink=0.8)
    fig.savefig(d / "figures" / name, dpi=115)
    plt.close(fig)
    entries.append({"question": "Q2", "file": f"figures/{name}", "what": f"phase diagram of {col}", "cells": cells})


if __name__ == "__main__":
    main()
