"""Q1 — what drives the A2 drift? (ANALYSIS_PLAN.md §4). Silent cells only: no controller.

Compares the four variants at fixed setup, q and rho:
  sim1_base            argmax votes, post a fact supporting the vote
  sim1_uniform_post    argmax votes, post a random active fact (vote ignored)
  sim1_pm              probability-matching votes, supporting posts
  sim1_pm_post_always  probability-matching votes, supporting posts, never abstain
Prediction if "argmax + vote-conditioned posting" drives the drift: it shrinks
under sim1_uniform_post and under sim1_pm.

Outputs (results/simulation_1/analysis/<date>/): tables/q1_final.csv, tables/q1_trajectories.csv,
figures/q1_*.png; listed in index.csv.

From the repository root:
    .venv/bin/python analysis/task003_llm_free/simulation_1/analysis/q1_drift.py
"""
from __future__ import annotations
import pathlib, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import (ALLOC_COLOR, VARIANT_COLOR, VARIANT_LABEL, VARIANTS, bootstrap_mean,  # noqa: E402
                    cells_table, out_dir, write_index)

SETUP_LABEL = {"task003_nosolution": "task003-nosolution", "task003_symmetric": "task003-symmetric"}
PANELS = [(rho, q) for rho in (0.75, 1.0) for q in (3, 6, 12)]


def load(cell_path: str) -> dict:
    d = pd.read_parquet(pathlib.Path(cell_path) / "days.parquet",
                        columns=["episode", "day", "share_A0", "share_A1", "share_A2"])
    d = d.sort_values(["episode", "day"])
    E, T = d.episode.nunique(), d.day.nunique()
    return {k: d[f"share_A{k}"].to_numpy().reshape(E, T) for k in range(3)}


def plurality(x: dict) -> np.ndarray:
    """Per episode and day: 0, 1, 2 for the allocation with the most votes, 3 for a tie at the top."""
    s = np.stack([x[0], x[1], x[2]])                          # (3, E, T)
    top = s.max(axis=0)
    winners = (np.isclose(s, top)).sum(axis=0)
    return np.where(winners > 1, 3, s.argmax(axis=0))


def main():
    d = out_dir()
    cells = cells_table()
    silent = cells[(cells.arm == "silent") & (cells.purpose == "main")]
    data = {(r.variant, r.setup, r.rho, r.q): load(r.path) for r in silent.itertuples()}
    days = np.arange(1, 31)

    # ---- tables ----------------------------------------------------------------------
    traj, final = [], []
    for (v, s, rho, q), x in data.items():
        margin = x[2] - x[0]
        for k, arr in ((0, x[0]), (1, x[1]), (2, x[2]), ("A2-A0", margin)):
            m, lo, hi = bootstrap_mean(arr)
            name = f"share_A{k}" if k != "A2-A0" else "margin_A2_minus_A0"
            traj += [{"variant": v, "setup": s, "rho": rho, "q": q, "day": t + 1, "quantity": name,
                      "mean": m[t], "lo": lo[t], "hi": hi[t]} for t in range(len(days))]
            final.append({"variant": v, "setup": s, "rho": rho, "q": q, "quantity": name,
                          "day30_mean": m[-1], "day30_lo": lo[-1], "day30_hi": hi[-1],
                          "day1_mean": m[0]})
        pl = plurality(x)[:, -1]
        final.append({"variant": v, "setup": s, "rho": rho, "q": q, "quantity": "episodes_led_by_A2_minus_A0",
                      "day30_mean": (pl == 2).mean() - (pl == 0).mean()})
    traj, final = pd.DataFrame(traj), pd.DataFrame(final)
    # unanimity: share of episodes where all 24 agents vote the same way on the last day
    unan = []
    for (v, s, rho, q), x in data.items():
        unan.append({"setup": s.split("_")[1], "rho": rho, "q": q, "variant": v,
                     "all_A0": (x[0][:, -1] == 1).mean(), "all_A2": (x[2][:, -1] == 1).mean(),
                     "A0_majority": (x[0][:, -1] > .5).mean(), "A2_majority": (x[2][:, -1] > .5).mean()})
    pd.DataFrame(unan).to_csv(d / "tables" / "q1_unanimity.csv", index=False)
    traj.to_csv(d / "tables" / "q1_trajectories.csv", index=False, float_format="%.5f")
    final.to_csv(d / "tables" / "q1_final.csv", index=False, float_format="%.5f")

    entries = [{"question": "Q1", "file": "tables/q1_trajectories.csv",
                "what": "mean vote shares and A2-A0 margin per day with 95% intervals", "cells": "silent, 4 variants"},
               {"question": "Q1", "file": "tables/q1_unanimity.csv",
                "what": "share of episodes unanimous on A0 / A2, and majority, on the last day", "cells": "silent, 4 variants"},
               {"question": "Q1", "file": "tables/q1_final.csv",
                "what": "day-30 shares and margin with 95% intervals; episodes led by A2 minus by A0", "cells": "silent, 4 variants"}]

    for setup in ("task003_nosolution", "task003_symmetric"):
        # ---- F2a: the A2-A0 margin over days, all variants per panel ----------------------
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
        for ax, (rho, q) in zip(axes.flat, PANELS):
            for v in VARIANTS:
                t = traj[(traj.variant == v) & (traj.setup == setup) & (traj.rho == rho) & (traj.q == q)
                         & (traj.quantity == "margin_A2_minus_A0")]
                ax.plot(t.day, t["mean"], color=VARIANT_COLOR[v], label=VARIANT_LABEL[v], lw=1.8)
                ax.fill_between(t.day, t.lo, t.hi, color=VARIANT_COLOR[v], alpha=0.2, lw=0)
            ax.axhline(0, color="k", lw=0.8)
            ax.set_title(f"ρ = {rho}, q = {q}", fontsize=10)
        for ax in axes[-1]:
            ax.set_xlabel("day")
        for ax in axes[:, 0]:
            ax.set_ylabel("share A2 − share A0")
        axes[0, 0].legend(fontsize=8, loc="best")
        fig.suptitle(f"Q1 · {SETUP_LABEL[setup]} · silent arm: vote margin A2 − A0 over days "
                     f"(mean over 1,000 episodes, 95% band)", fontsize=11)
        fig.tight_layout()
        f = f"figures/q1_margin_{setup}.png"
        fig.savefig(d / f, dpi=130)
        plt.close(fig)
        entries.append({"question": "Q1", "file": f, "what": "A2-A0 vote margin over days, 4 variants",
                        "cells": f"{setup} silent"})

        # ---- F2b: the three shares over days, one row per variant -------------------------
        fig, axes = plt.subplots(4, 6, figsize=(18, 10), sharex=True, sharey=True)
        for i, v in enumerate(VARIANTS):
            for j, (rho, q) in enumerate(PANELS):
                ax = axes[i, j]
                for k in range(3):
                    t = traj[(traj.variant == v) & (traj.setup == setup) & (traj.rho == rho) & (traj.q == q)
                             & (traj.quantity == f"share_A{k}")]
                    ax.plot(t.day, t["mean"], color=ALLOC_COLOR[k], lw=1.6, label=f"A{k}")
                    ax.fill_between(t.day, t.lo, t.hi, color=ALLOC_COLOR[k], alpha=0.2, lw=0)
                ax.set_ylim(0, 1)
                if i == 0:
                    ax.set_title(f"ρ = {rho}, q = {q}", fontsize=10)
                if j == 0:
                    ax.set_ylabel(VARIANT_LABEL[v], fontsize=8)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(f"Q1 · {SETUP_LABEL[setup]} · silent arm: vote shares over days", fontsize=12)
        fig.tight_layout()
        f = f"figures/q1_shares_{setup}.png"
        fig.savefig(d / f, dpi=110)
        plt.close(fig)
        entries.append({"question": "Q1", "file": f, "what": "mean vote shares A0/A1/A2 over days",
                        "cells": f"{setup} silent"})

        # ---- F2c: spread across episodes: heat strip of n_A2 - n_A0 per day --------------
        fig, axes = plt.subplots(4, 6, figsize=(18, 10), sharex=True, sharey=True)
        bins = np.arange(-24.5, 25.5, 1)
        for i, v in enumerate(VARIANTS):
            for j, (rho, q) in enumerate(PANELS):
                ax = axes[i, j]
                x = data[(v, setup, rho, q)]
                margin = np.rint(24 * (x[2] - x[0])).astype(int)       # (E, T) in agents
                H = np.stack([np.histogram(margin[:, t], bins=bins)[0] for t in range(30)], axis=1)
                ax.imshow(H / H.sum(axis=0, keepdims=True), origin="lower", aspect="auto", cmap="magma_r",
                          extent=[0.5, 30.5, -24.5, 24.5], vmin=0, vmax=0.3)
                ax.axhline(0, color="#3d9970", lw=0.6)
                if i == 0:
                    ax.set_title(f"ρ = {rho}, q = {q}", fontsize=10)
                if j == 0:
                    ax.set_ylabel(VARIANT_LABEL[v] + "\nn_A2 − n_A0", fontsize=8)
        for ax in axes[-1]:
            ax.set_xlabel("day")
        fig.suptitle(f"Q1 · {SETUP_LABEL[setup]} · silent arm: how the vote margin n_A2 − n_A0 is spread across "
                     f"episodes each day (darker = more episodes)", fontsize=12)
        fig.tight_layout()
        f = f"figures/q1_spread_{setup}.png"
        fig.savefig(d / f, dpi=110)
        plt.close(fig)
        entries.append({"question": "Q1", "file": f, "what": "distribution of n_A2 - n_A0 across episodes per day",
                        "cells": f"{setup} silent"})

        # ---- consensus split: which allocation leads, share of episodes per day ----------
        fig, axes = plt.subplots(4, 6, figsize=(18, 10), sharex=True, sharey=True)
        for i, v in enumerate(VARIANTS):
            for j, (rho, q) in enumerate(PANELS):
                ax = axes[i, j]
                pl = plurality(data[(v, setup, rho, q)])
                fr = np.stack([(pl == k).mean(axis=0) for k in (0, 1, 2, 3)])
                ax.stackplot(days, fr, colors=[ALLOC_COLOR[0], ALLOC_COLOR[1], ALLOC_COLOR[2], "#e8e8e8"],
                             labels=["A0 leads", "A1 leads", "A2 leads", "tie at top"])
                ax.set_ylim(0, 1)
                if i == 0:
                    ax.set_title(f"ρ = {rho}, q = {q}", fontsize=10)
                if j == 0:
                    ax.set_ylabel(VARIANT_LABEL[v], fontsize=8)
        axes[0, 0].legend(fontsize=7, loc="lower left")
        fig.suptitle(f"Q1 · {SETUP_LABEL[setup]} · silent arm: share of episodes led by each allocation, per day",
                     fontsize=12)
        fig.tight_layout()
        f = f"figures/q1_leader_{setup}.png"
        fig.savefig(d / f, dpi=110)
        plt.close(fig)
        entries.append({"question": "Q1", "file": f, "what": "share of episodes led by A0/A1/A2 per day",
                        "cells": f"{setup} silent"})
    write_index(d, entries)

    # ---- console summary --------------------------------------------------------------
    m = final[final.quantity == "margin_A2_minus_A0"]
    piv = m.pivot_table(index=["setup", "rho", "q"], columns="variant", values="day30_mean")[list(VARIANTS)]
    print("DAY-30 MARGIN share A2 − share A0 (silent; >0 means drift to A2)")
    print(piv.round(3).to_string())
    lo = m.pivot_table(index=["setup", "rho", "q"], columns="variant", values="day30_lo")[list(VARIANTS)]
    hi = m.pivot_table(index=["setup", "rho", "q"], columns="variant", values="day30_hi")[list(VARIANTS)]
    print("\n95% intervals")
    print((lo.round(3).astype(str) + " .. " + hi.round(3).astype(str)).to_string())
    a1 = final[final.quantity == "share_A1"].pivot_table(index=["setup", "rho", "q"], columns="variant",
                                                           values="day30_mean")[list(VARIANTS)]
    print("\nDAY-30 share A1")
    print(a1.round(3).to_string())


if __name__ == "__main__":
    main()
