#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "state_matched_response_summary.csv")

fig, ax = plt.subplots(figsize=(5.25, 3.35))
yerr = [df["chi"] - df["chi_ci_low"], df["chi_ci_high"] - df["chi"]]
ax.errorbar(
    df["x_0"], df["chi"], yerr=yerr,
    marker="o", capsize=4, linewidth=1.6,
    label="LLM: ADVOCATE - NOOP",
)
ax.plot(
    df["x_0"], df["qvoter_q1_chi"],
    marker="s", linestyle="--", linewidth=1.4,
    label="Exact $q=1$ reference",
)
ax.axhline(0.0, linewidth=0.8)
ax.set_xlabel(r"Initial target fraction $x_0=n_Z(0)/N$")
ax.set_ylabel(r"Signed response $\chi(x_0)$")
ax.set_xticks(df["x_0"].tolist())
ax.set_xlim(0.225, 0.525)
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(ROOT / "figures" / "state_matched_response.pdf", bbox_inches="tight")
fig.savefig(ROOT / "figures" / "state_matched_response.png", dpi=220, bbox_inches="tight")
