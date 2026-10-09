"""Q7 of the analysis plan: how well does the controller sense the population?

Per controller action: v_hat (share of the posts it read whose author voted for the target
WHEN POSTING) against the true current share (all 24 agents' last votes). Sensing error =
v_hat - true share. Recorded in every controlled cell, also with the gate off (where v_hat
is not used).

Usage, from the repository root:
    .venv/bin/python analysis/task003_llm_free/simulation_2/analysis/q7_sensing.py [--run <run>] [--date <date>]
Output: tables/q7_sensing.csv, figures/F6_sensing.png
"""
from __future__ import annotations
import argparse, multiprocessing, pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import SETUP_NAME, cells_table, out_dir, run_dir, write_index  # noqa: E402

WINDOWS = [(0, 5), (5, 15), (15, 30)]


def cell_sensing(args) -> dict:
    run, folder = args
    c = pd.read_parquet(pathlib.Path(run) / "cells" / folder / "controller.parquet",
                        columns=["episode", "t", "v_hat", "true_target_share", "share_voted", "decision"])
    c = c[c.v_hat.notna()]
    err = c.v_hat - c.true_target_share
    out = {"folder": folder, "actions_with_reads": len(c), "error_mean": err.mean(), "abs_error_mean": err.abs().mean(),
           "v_hat_mean": c.v_hat.mean(), "true_share_mean": c.true_target_share.mean(),
           "share_gate_decisions": (c.decision == "gate").mean()}
    per_ep = err.groupby(c.episode).mean()
    out["error_se"] = per_ep.std() / np.sqrt(len(per_ep)) if len(per_ep) > 1 else np.nan
    for lo, hi in WINDOWS:
        w = (c.t >= lo) & (c.t < hi)
        out[f"error_t{lo}_{hi}"] = err[w].mean()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--date")
    a = ap.parse_args()
    run = pathlib.Path(a.run).resolve() if a.run else run_dir("sim2_grid")
    d = out_dir(a.date)
    cells = cells_table(run)
    ctrl = cells[(cells.arm != "silent") & (cells.cell == cells.folder)]        # the cells that ran
    with multiprocessing.Pool(12) as pool:
        res = pd.DataFrame(pool.map(cell_sensing, [(str(run), f) for f in ctrl.folder]))
    res = ctrl[["folder", "setup", "arm", "q", "qc", "rho", "lambda_c", "gate", "stop"]].merge(res, on="folder")
    res.to_csv(d / "tables" / "q7_sensing.csv", index=False)

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), sharey=True)
    for j, (setup, arm) in enumerate([(s, x) for s in ("task003_symmetric", "task003_nosolution") for x in ("truth", "false")]):
        ax = axes[j]
        for gate, ls in (("on", "-"), ("off", "--")):
            for qc, col in zip((3, 6, 12, 24), plt.cm.viridis(np.linspace(0, .85, 4))):
                s = res[(res.setup == setup) & (res.arm == arm) & (res.q == 6) & (res.rho == 0.75) & (res.qc == qc)
                        & (res.gate == gate) & (res.stop == "on")].sort_values("lambda_c")
                ax.errorbar(s.lambda_c, s.error_mean, yerr=1.96 * s.error_se, color=col, ls=ls, marker="o", ms=3,
                            capsize=2, label=f"qc {qc}, gate {gate}" if j == 0 else None)
        ax.axhline(0, color="0.6", lw=0.8)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("λc")
        ax.set_title(f"{SETUP_NAME[setup]}, target {'A0' if arm == 'truth' else 'A2'}", fontsize=9)
    axes[0].set_ylabel("v̂ − true current share")
    axes[0].legend(fontsize=6, ncol=2)
    fig.suptitle("Sensing error per controller action, q 6, ρ 0.75 (bars: ±1.96 s.e. over episodes)", fontsize=11)
    fig.tight_layout()
    fig.savefig(d / "figures" / "F6_sensing.png", dpi=130)
    plt.close(fig)
    write_index(d, [{"file": "tables/q7_sensing.csv", "question": "Q7", "what": "sensing error per cell"},
                    {"file": "figures/F6_sensing.png", "question": "Q7", "what": "sensing error against the rate"}])
    print(res.groupby(["setup", "arm", "gate"])[["error_mean", "abs_error_mean"]].mean().round(3))


if __name__ == "__main__":
    main()
