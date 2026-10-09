"""Q1 of the analysis plan: Simulation 1 against the asynchronous core.

Silent populations: m_A0(30), m_A2(30) at every setup, q, rho (Simulation 1: end of day 30).
Control: G(30) in the core (rate 1, gate on, stop on) against Simulation 1 (b = 1, same q, qc, rho).
Each code's gain is paired within its own run; the two codes are bootstrapped independently.

Usage, from the repository root (after build_tables.py):
    .venv/bin/python analysis/task003_llm_free/simulation_2/analysis/q1_sim1_vs_core.py [--date <date>] [--run <run>]
Output: tables/q1_silent.csv, tables/q1_gains.csv, figures/F4_*.png
"""
from __future__ import annotations
import argparse, pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import (SETUP_NAME, SIM1_RESULTS, TARGET, boot_index, out_dir, run_dir, verdict,  # noqa: E402
                    write_index)

SIM1_RUN = SIM1_RESULTS / "2026-10-07_sim1_pm" / "cells"


def sim1_cell(setup, arm, q, qc, rho):
    rho_s = f"{rho:.2f}".rstrip("0") if rho != 1 else "1.0"
    if arm == "silent":
        return SIM1_RUN / f"{setup}__silent__q{q}__rho{rho_s}"
    return SIM1_RUN / f"{setup}__{arm}__q{q}__qc{qc}__b1__rho{rho_s}"


def sim1_final(cell: pathlib.Path) -> pd.DataFrame:
    d = pd.read_parquet(cell / "days.parquet", columns=["episode", "day", "mean_p_A0", "mean_p_A2"])
    return d[d.day == 29].sort_values("episode").reset_index(drop=True)          # end of day 30


def diff_ci(a: np.ndarray, b: np.ndarray, name: str):
    """mean(a) - mean(b), a and b from different codes: independent resampling."""
    ia, ib = boot_index(len(a), name + "|a"), boot_index(len(b), name + "|b")
    bs = a[ia].mean(axis=1) - b[ib].mean(axis=1)
    return float(a.mean() - b.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date")
    ap.add_argument("--run")
    a = ap.parse_args()
    d = out_dir(a.date)
    run = pathlib.Path(a.run).resolve() if a.run else run_dir("sim2_grid")
    cells = pd.read_csv(d / "tables" / "cells.csv")
    ev = pd.read_parquet(d / "tables" / "episode_values.parquet")
    V = {f: g.sort_values("episode").reset_index(drop=True) for f, g in ev.groupby("folder")}

    silent_rows = []
    for c in cells[cells.arm == "silent"].itertuples():
        s1 = sim1_final(sim1_cell(c.setup, "silent", c.q, None, c.rho))
        x = V[c.folder]
        for k in ("A0", "A2"):
            m, lo, hi = diff_ci(x[f"mean_p_{k}_t30"].to_numpy(), s1[f"mean_p_{k}"].to_numpy(), f"silent{c.cell}{k}")
            silent_rows.append({"setup": c.setup, "q": c.q, "rho": c.rho, "belief": k,
                                "core": x[f"mean_p_{k}_t30"].mean(), "sim1": s1[f"mean_p_{k}"].mean(),
                                "core_minus_sim1": m, "lo": lo, "hi": hi, "verdict": verdict(m, lo, hi)})
    silent = pd.DataFrame(silent_rows)
    silent.to_csv(d / "tables" / "q1_silent.csv", index=False)

    folder_of = dict(zip(cells.cell, cells.folder))
    core = cells[(cells.arm != "silent") & (cells.lambda_c == 1.0) & (cells.gate == "on") & (cells.stop == "on")]
    rows = []
    for c in core.itertuples():
        k = TARGET[c.arm]
        x, s = V[c.folder], V[folder_of[c.silent_cell]]
        g_core = (x[f"mean_p_A{k}_t30"] - s[f"mean_p_A{k}_t30"]).to_numpy()
        c1, s1 = sim1_final(sim1_cell(c.setup, c.arm, c.q, c.qc, c.rho)), sim1_final(sim1_cell(c.setup, "silent", c.q, None, c.rho))
        g_sim1 = (c1[f"mean_p_A{k}"] - s1[f"mean_p_A{k}"]).to_numpy()
        m, lo, hi = diff_ci(g_core, g_sim1, f"gain{c.cell}")
        rows.append({"setup": c.setup, "arm": c.arm, "q": c.q, "qc": c.qc, "rho": c.rho,
                     "G30_core": g_core.mean(), "G30_sim1": g_sim1.mean(), "core_minus_sim1": m, "lo": lo, "hi": hi,
                     "verdict": verdict(m, lo, hi)})
    gains = pd.DataFrame(rows)
    gains.to_csv(d / "tables" / "q1_gains.csv", index=False)

    # F4a: silent trajectories, Simulation 1 (days) vs core (continuous time)
    index = []
    fig, axes = plt.subplots(2, 6, figsize=(17, 6), sharex=True, sharey=True)
    for i, setup in enumerate(("task003_symmetric", "task003_nosolution")):
        for j, (q, rho) in enumerate([(q, r) for r in (0.75, 1.0) for q in (3, 6, 12)]):
            ax = axes[i, j]
            c = cells[(cells.arm == "silent") & (cells.setup == setup) & (cells.q == q) & (cells.rho == rho)].iloc[0]
            s = pd.read_parquet(run / "cells" / c.folder / "snapshots.parquet", columns=["t", "mean_p_A0", "mean_p_A2"])
            s = s.groupby("t").mean()
            s1 = pd.read_parquet(sim1_cell(setup, "silent", q, None, rho) / "days.parquet").groupby("day")[["mean_p_A0", "mean_p_A2"]].mean()
            for k, col in (("A0", "tab:blue"), ("A2", "tab:red")):
                ax.plot(s.index, s[f"mean_p_{k}"], color=col, lw=1.2, label=f"core m_{k}")
                ax.plot(s1.index + 1, s1[f"mean_p_{k}"], color=col, lw=0, marker="o", ms=2.5, label=f"Simulation 1 m_{k}")
            ax.axvline(30, color="0.6", ls="--", lw=0.8)
            ax.set_title(f"{SETUP_NAME[setup]}, q {q}, ρ {rho:g}", fontsize=8)
            ax.set_ylim(0, 1)
    axes[0, 0].legend(fontsize=6)
    for ax in axes[1]:
        ax.set_xlabel("time (Simulation 1: day)")
    fig.suptitle("Silent populations: Simulation 1 (dots, end of each day) vs the asynchronous core (lines)", fontsize=11)
    fig.tight_layout()
    fig.savefig(d / "figures" / "F4a_silent_sim1_vs_core.png", dpi=130)
    plt.close(fig)
    index.append({"file": "figures/F4a_silent_sim1_vs_core.png", "question": "Q1", "what": "silent trajectories"})

    # F4b: G(30) core vs Simulation 1
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), sharex=True, sharey=True)
    for j, (setup, arm) in enumerate([(s, a) for s in ("task003_symmetric", "task003_nosolution") for a in ("truth", "false")]):
        ax = axes[j]
        g = gains[(gains.setup == setup) & (gains.arm == arm)]
        for rho, mk in ((0.75, "o"), (1.0, "s")):
            h = g[g.rho == rho]
            ax.scatter(h.G30_sim1, h.G30_core, c=np.log2(h.qc), cmap="viridis", marker=mk, s=18 + 6 * h.q,
                       label=f"ρ {rho:g}")
        ax.plot([-0.1, 0.7], [-0.1, 0.7], color="0.6", lw=0.8)
        ax.set_title(f"{SETUP_NAME[setup]}, target {'A0' if arm == 'truth' else 'A2'}", fontsize=9)
        ax.set_xlabel("G(30), Simulation 1 (b = 1)")
    axes[0].set_ylabel("G(30), core (λc = 1, gate on)")
    axes[0].legend(fontsize=7)
    fig.suptitle("Control gain: Simulation 1 vs core, every q, qc, ρ (colour: qc; size: q)", fontsize=11)
    fig.tight_layout()
    fig.savefig(d / "figures" / "F4b_gain_sim1_vs_core.png", dpi=130)
    plt.close(fig)
    index.append({"file": "figures/F4b_gain_sim1_vs_core.png", "question": "Q1", "what": "G(30) core vs Simulation 1"})
    index += [{"file": "tables/q1_silent.csv", "question": "Q1", "what": "silent levels core vs Simulation 1"},
              {"file": "tables/q1_gains.csv", "question": "Q1", "what": "G(30) core vs Simulation 1"}]
    write_index(d, index)
    print(silent.round(3).to_string(index=False))
    print(gains.groupby(["setup", "arm", "rho"])[["G30_core", "G30_sim1", "core_minus_sim1"]].mean().round(3))
    print(gains.groupby(["setup", "arm"]).verdict.value_counts().unstack(fill_value=0))


if __name__ == "__main__":
    main()
