"""Build the consolidated preanalysis PDF (figures + LaTeX)."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "report"
FIGS = REPORT / "figures"


def tex_env() -> dict[str, str]:
    """PATH with a pdflatex that actually runs on this machine placed first."""
    for root in ("/opt/homebrew/bin", "/usr/local/bin", "/Library/TeX/texbin"):
        exe = Path(root) / "pdflatex"
        if not exe.exists():
            continue
        probe = subprocess.run([str(exe), "--version"], capture_output=True)
        if probe.returncode == 0:
            return {**os.environ, "PATH": root + ":" + os.environ.get("PATH", "")}
    raise RuntimeError("no working pdflatex found")


def figures() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    curves = pd.read_csv(ROOT / "40_information_estimates" / "estimates_curves.csv")
    tests = pd.read_csv(ROOT / "30_coarse_graining" / "four_tests.csv")
    sweep = pd.read_csv(ROOT / "30_coarse_graining" / "bin_sweep.csv")

    # --- Figure 1: path KL vs horizon -----------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), sharex=True)
    for ax, rho in zip(axes, (0.75, 1.0)):
        d = curves[curves.rho == rho]
        ax.plot(d.h, d.K_A2, "o-", color="#1f77b4", lw=1.8, ms=3.5, label="A2 arm (decoy)")
        ax.plot(d.h, d.K_A0, "s--", color="#d62728", lw=1.8, ms=3.5, label="A0 arm (confounded)")
        ax.set_title(f"persistence $\\rho$ = {rho:g}")
        ax.set_xlabel("horizon $h$ (rounds)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("path KL from silence (nats)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_pathkl.pdf")
    plt.close(fig)

    # --- Figure 2: mutual informations ----------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
    for rho, style in ((0.75, "o-"), (1.0, "s--")):
        d = curves[curves.rho == rho]
        axes[0].plot(d.h, d.I_assigned, style, lw=1.8, ms=3.5, label=f"$\\rho$ = {rho:g}")
        axes[1].plot(d.h, d.I_target_confounded, style, lw=1.8, ms=3.5, label=f"$\\rho$ = {rho:g}")
    axes[0].set_ylabel("I(intervention; state) [bits]")
    axes[0].set_title("is intervention visible?")
    axes[1].set_ylabel("I(target; state) [bits]")
    axes[1].set_title("is the chosen target visible?")
    for ax in axes:
        ax.set_xlabel("horizon $h$ (rounds)")
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
    axes[1].set_ylim(0, 0.03)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_mi.pdf")
    plt.close(fig)

    # --- Figure 3: the four tests ---------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2))
    labels = [f"{r.rho:g}\n{r.arm}" for r in tests.itertuples()]
    x = range(len(tests))
    colors = ["#2ca02c" if (r.rho == 1.0 and r.arm == "silent") else "#8c8c8c"
              for r in tests.itertuples()]

    axes[0].bar(x, tests.order_gain, yerr=tests.order_sd, color=colors, capsize=3)
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].set_title("Test 1: order-2 gain")
    axes[0].set_ylabel("nats / step")

    for col, mk in (("ck_n2", "o-"), ("ck_n3", "s--"), ("ck_n5", "^:")):
        axes[1].plot(x, tests[col], mk, lw=1.6, ms=4, label=col.replace("ck_n", "$n$ = "))
    axes[1].set_title("Test 2: Chapman--Kolmogorov gap")
    axes[1].set_ylabel("total variation")
    axes[1].legend(frameon=False, fontsize=8)

    axes[2].bar(x, tests.cmi_excess, color=colors)
    axes[2].set_title("Test 3: CMI above permutation null")
    axes[2].set_ylabel("bits")

    for ax in axes:
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, fontsize=7)
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_tests.pdf")
    plt.close(fig)

    # --- Figure 4: bin sweep --------------------------------------------
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for (scheme, arm), d in sweep.groupby(["scheme", "arm"]):
        ax.plot(d.bins, d.gain, "o-", lw=1.6, ms=4, label=f"{scheme}, {arm}")
    ax.set_xlabel("number of bins")
    ax.set_ylabel("order-2 gain (nats/step)")
    ax.set_title("how many states?")
    ax.axvline(4, color="k", ls=":", lw=1)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_bins.pdf")
    plt.close(fig)


def main() -> None:
    figures()
    env = tex_env()
    for _ in range(2):
        proc = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
            cwd=REPORT, env=env, capture_output=True, text=True,
        )
    if proc.returncode != 0:
        tail = "\n".join(proc.stdout.splitlines()[-40:])
        raise SystemExit(f"pdflatex failed:\n{tail}")
    shutil.copy(REPORT / "main.pdf", ROOT / "preanalysis_task003_report.pdf")
    print("wrote", ROOT / "preanalysis_task003_report.pdf")


if __name__ == "__main__":
    main()
