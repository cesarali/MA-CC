"""Figures F1-F3 and F5 of the analysis plan (§9), from the tables of build_tables.py.

  F1 phase diagrams: tiles rate (x) x qc (y), G(30) printed, grey where the interval includes 0;
     panels q x rho; one figure per setup x target x gate (stop on). Companion: messages.
  F2 rate curves: G(30) and messages against the rate, one line per qc, panels q x rho.
  F3 gains over time for the main cells (q 6, qc 12, rho 0.75): G(t), 0-40, one line per rate.
     Bands: mean +- 1.96 standard errors over episodes (normal approximation).
  F5 gate effect (off - on) against the rate; proof-stop effect.

Usage, from the repository root:
    .venv/bin/python analysis/task003_llm_free/simulation_2/analysis/make_figures.py [--run <run folder>] [--date <date>] [--tag <prefix>]
"""
from __future__ import annotations
import argparse, pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import SETUP_NAME, SIZE, TARGET, out_dir, run_dir, write_index  # noqa: E402

RATES = [0.5, 1.0, 2.0, 4.0, 8.0]
QCS = [3, 6, 12, 24]
QS = [3, 6, 12]
RHOS = [0.75, 1.0]
TNAME = {"truth": "target A0 (truth)", "false": "target A2 (false)"}


def phase(g: pd.DataFrame, value: str, d: pathlib.Path, tag: str, vmax: float, cmap: str, fmt: str, label: str,
          grey_ci: bool) -> list[dict]:
    index = []
    for (setup, arm, gate), sub in g[g.stop == "on"].groupby(["setup", "arm", "gate"]):
        fig, axes = plt.subplots(2, 3, figsize=(11, 6.2), sharex=True, sharey=True)
        for i, rho in enumerate(RHOS):
            for j, q in enumerate(QS):
                ax = axes[i, j]
                m = np.full((len(QCS), len(RATES)), np.nan)
                grey = np.zeros_like(m, dtype=bool)
                for r in sub[(sub.q == q) & (sub.rho == rho)].itertuples():
                    a, b = QCS.index(r.qc), RATES.index(r.lambda_c)
                    m[a, b] = getattr(r, value)
                    if grey_ci:
                        grey[a, b] = getattr(r, f"{value}_lo") <= 0 <= getattr(r, f"{value}_hi")
                im = ax.imshow(m, origin="lower", cmap=cmap, vmin=-vmax if cmap == "RdBu_r" else 0, vmax=vmax,
                               aspect="auto")
                for a in range(len(QCS)):
                    for b in range(len(RATES)):
                        if np.isnan(m[a, b]):
                            continue
                        ax.text(b, a, format(m[a, b], fmt), ha="center", va="center", fontsize=8,
                                color="0.55" if grey[a, b] else "black")
                        if grey[a, b]:
                            ax.add_patch(plt.Rectangle((b - .5, a - .5), 1, 1, color="0.92", zorder=0.5))
                ax.set_xticks(range(len(RATES)), [f"{x:g}" for x in RATES])
                ax.set_yticks(range(len(QCS)), QCS)
                ax.set_title(f"q = {q}, ρ = {rho:g}", fontsize=9)
                if i == 1:
                    ax.set_xlabel("controller rate λc")
                if j == 0:
                    ax.set_ylabel("qc")
        fig.colorbar(im, ax=axes, shrink=0.8, label=label)
        fig.suptitle(f"{SETUP_NAME[setup]}, {TNAME[arm]}, vote gate {gate}: {label}", fontsize=11)
        name = f"figures/{tag}F1_{value}_{SETUP_NAME[setup]}_{arm}_gate{gate}.png"
        fig.savefig(d / name, dpi=130, bbox_inches="tight")
        plt.close(fig)
        index.append({"file": name, "question": "Q2", "what": f"phase diagram of {value}"})
    return index


def rate_curves(g: pd.DataFrame, d: pathlib.Path, tag: str) -> list[dict]:
    index = []
    for (setup, arm, gate), sub in g[g.stop == "on"].groupby(["setup", "arm", "gate"]):
        fig, axes = plt.subplots(2, 6, figsize=(16, 5.5), sharex=True)
        for i, rho in enumerate(RHOS):
            for j, q in enumerate(QS):
                ax, ax2 = axes[i, 2 * j], axes[i, 2 * j + 1]
                for qc, col in zip(QCS, plt.cm.viridis(np.linspace(0, .85, len(QCS)))):
                    s = sub[(sub.q == q) & (sub.rho == rho) & (sub.qc == qc)].sort_values("lambda_c")
                    ax.errorbar(s.lambda_c, s.G30, yerr=[s.G30 - s.G30_lo, s.G30_hi - s.G30], color=col,
                                marker="o", ms=3, capsize=2, label=f"qc {qc}")
                    ax2.plot(s.lambda_c, s.messages, color=col, marker="o", ms=3)
                ax.axhline(0, color="0.6", lw=0.8)
                ax.set_xscale("log", base=2)
                ax2.set_xscale("log", base=2)
                ax.set_title(f"G(30), q {q}, ρ {rho:g}", fontsize=8)
                ax2.set_title(f"messages, q {q}, ρ {rho:g}", fontsize=8)
                ax2.set_ylim(0, 31)
        axes[0, 0].legend(fontsize=7)
        for ax in axes[1]:
            ax.set_xlabel("λc")
        fig.suptitle(f"{SETUP_NAME[setup]}, {TNAME[arm]}, vote gate {gate}: gain and messages against the rate", fontsize=11)
        fig.tight_layout()
        name = f"figures/{tag}F2_rate_{SETUP_NAME[setup]}_{arm}_gate{gate}.png"
        fig.savefig(d / name, dpi=120)
        plt.close(fig)
        index.append({"file": name, "question": "Q3", "what": "G(30) and messages against the rate"})
    return index


def gains_over_time(run: pathlib.Path, cells: pd.DataFrame, d: pathlib.Path, tag: str) -> list[dict]:
    """F3 for the main cells; per-episode differences at every measurement time."""
    main = cells[(cells.q == 6) & (cells.rho == 0.75) & (cells.arm != "silent") & (cells.qc == 12) & (cells.stop == "on")]
    folder_of = dict(zip(cells.cell, cells.folder))
    cache = {}

    def snap(folder):
        if folder not in cache:
            s = pd.read_parquet(run / "cells" / folder / "snapshots.parquet",
                                columns=["episode", "t", "mean_p_A0", "mean_p_A2"])
            cache[folder] = s.sort_values(["t", "episode"])
        return cache[folder]

    rows, index = [], []
    for (setup, arm, gate), sub in main.groupby(["setup", "arm", "gate"]):
        k = TARGET[arm]
        for c in sub.itertuples():
            x, s = snap(c.folder), snap(folder_of[c.silent_cell])
            diff = x[f"mean_p_A{k}"].to_numpy() - s[f"mean_p_A{k}"].to_numpy()
            t = x.t.to_numpy()
            df = pd.DataFrame({"t": t, "d": diff}).groupby("t").d.agg(["mean", "std", "count"]).reset_index()
            df["se"] = df["std"] / np.sqrt(df["count"])
            df["setup"], df["arm"], df["gate"], df["lambda_c"] = setup, arm, gate, c.lambda_c
            rows.append(df)
    traj = pd.concat(rows)
    traj.to_csv(d / "tables" / f"{tag}gain_trajectories_main.csv", index=False)
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.5), sharex=True)
    for i, gate in enumerate(["on", "off"]):
        for j, (setup, arm) in enumerate([(s, a) for s in ("task003_symmetric", "task003_nosolution") for a in ("truth", "false")]):
            ax = axes[i, j]
            for lc, col in zip(RATES, plt.cm.plasma(np.linspace(0, .85, len(RATES)))):
                s = traj[(traj.setup == setup) & (traj.arm == arm) & (traj.gate == gate) & (traj.lambda_c == lc)]
                ax.plot(s.t, s["mean"], color=col, lw=1.2, label=f"λc {lc:g}")
                ax.fill_between(s.t, s["mean"] - 1.96 * s.se, s["mean"] + 1.96 * s.se, color=col, alpha=.2, lw=0)
            ax.axvline(30, color="0.5", ls="--", lw=0.8)
            ax.axhline(0, color="0.6", lw=0.8)
            ax.set_title(f"{SETUP_NAME[setup]}, {TNAME[arm]}, gate {gate}", fontsize=9)
            if i == 1:
                ax.set_xlabel("time t")
            if j == 0:
                ax.set_ylabel("gain G(t) in mean belief")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Gain over time, q 6, qc 12, ρ 0.75 (controller stops at t = 30; bands ±1.96 s.e.)", fontsize=11)
    fig.tight_layout()
    name = f"figures/{tag}F3_gain_over_time_main.png"
    fig.savefig(d / name, dpi=130)
    plt.close(fig)
    index.append({"file": name, "question": "Q4", "what": "G(t) for the main cells"})
    index.append({"file": f"tables/{tag}gain_trajectories_main.csv", "question": "Q4", "what": "G(t), main cells"})
    return index


def gate_and_stop(d: pathlib.Path, tag: str) -> list[dict]:
    ge = pd.read_csv(d / "tables" / f"{tag}gate_effect.csv")
    ge = ge[ge.stop == "on"]
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.5), sharex=True)
    for j, (setup, arm) in enumerate([(s, a) for s in ("task003_symmetric", "task003_nosolution") for a in ("truth", "false")]):
        for i, (col_name, lab) in enumerate([("effect_G30", "gate off − on: G(30)"), ("effect_messages", "gate off − on: messages")]):
            ax = axes[i, j]
            for qc, col in zip(QCS, plt.cm.viridis(np.linspace(0, .85, len(QCS)))):
                s = ge[(ge.setup == setup) & (ge.arm == arm) & (ge.q == 6) & (ge.rho == 0.75) & (ge.qc == qc)].sort_values("lambda_c")
                ax.errorbar(s.lambda_c, s[col_name], yerr=[s[col_name] - s[f"{col_name}_lo"], s[f"{col_name}_hi"] - s[col_name]],
                            color=col, marker="o", ms=3, capsize=2, label=f"qc {qc}")
            ax.axhline(0, color="0.6", lw=0.8)
            if i == 0:
                for y in (SIZE, -SIZE):
                    ax.axhline(y, color="0.8", ls=":", lw=0.8)
            ax.set_xscale("log", base=2)
            ax.set_title(f"{SETUP_NAME[setup]}, {TNAME[arm]}", fontsize=9)
            ax.set_ylabel(lab, fontsize=8)
    axes[0, 0].legend(fontsize=7)
    for ax in axes[1]:
        ax.set_xlabel("λc")
    fig.suptitle("Turning the vote gate off, q 6, ρ 0.75 (dotted: ±1/24)", fontsize=11)
    fig.tight_layout()
    name = f"figures/{tag}F5_gate_effect.png"
    fig.savefig(d / name, dpi=130)
    plt.close(fig)
    return [{"file": name, "question": "Q5", "what": "gate effect against the rate"}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--date")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    run = pathlib.Path(a.run).resolve() if a.run else run_dir("sim2_grid")
    d = out_dir(a.date)
    g = pd.read_csv(d / "tables" / f"{a.tag}gains.csv")
    cells = pd.read_csv(d / "tables" / f"{a.tag}cells.csv")
    index = []
    index += phase(g, "G30", d, a.tag, vmax=0.5, cmap="RdBu_r", fmt=".2f", label="gain G(30)", grey_ci=True)
    index += phase(g, "messages", d, a.tag, vmax=30, cmap="Greys", fmt=".0f", label="messages per episode", grey_ci=False)
    index += rate_curves(g, d, a.tag)
    index += gains_over_time(run, cells, d, a.tag)
    index += gate_and_stop(d, a.tag)
    write_index(d, index)
    print(f"{len(index)} figures and tables in {d}")


if __name__ == "__main__":
    main()
