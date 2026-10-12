"""Figures and tables for the efficiency sections of the extended reports (12 October 2026).

For each simulation and setup writes, into the new report's figures/ folder:
  fig_eff_settings.pdf  steering I, efficiency eta and the agents' uncertainty against the control
                        setting (Simulation 1: b; Simulation 2: rate, gate on/off), at t = 30
  fig_eff_time.pdf      I(t) and eta(t) over time, main setting family
  tab_eff_precision.tex values at 1,000 paths and RMSE at 50 / 100 / 300 paths (LaTeX rows)

Usage, from the repository root (after efficiency.py and downsample.py):
    .venv/bin/python analysis/task003_llm_free/efficiency_estimates/make_report_figures.py [--date 2026-10-12]
"""
from __future__ import annotations
import argparse, pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
RES = ROOT / "results" / "efficiency_estimates"
REPORTS = {("sim1", "symmetric"): ROOT / "simulation_1/analysis/reports/2026-10-12_report1_symmetric",
           ("sim1", "nosolution"): ROOT / "simulation_1/analysis/reports/2026-10-12_report2_nosolution",
           ("sim2", "symmetric"): ROOT / "simulation_2/analysis/reports/2026-10-12_report1_symmetric",
           ("sim2", "nosolution"): ROOT / "simulation_2/analysis/reports/2026-10-12_report2_nosolution"}
SWARM = "swarm_beliefs_k3"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "legend.fontsize": 6.5, "figure.dpi": 150})
TCOL = {"truth": "tab:blue", "false": "tab:red"}


def xvals(sim, setting):
    if sim == "sim1":
        return int(setting[1:]), "on"
    lc, gate = setting.split("_")
    return float(lc[2:]), gate[4:]


def settings_figure(e, u, sim, setup, out):
    e = e[(e.sim == sim) & (e.setup == f"task003_{setup}") & (e.t == 30)].copy()
    u = u[(u.sim == sim) & (u.setup == f"task003_{setup}") & (u.t == 30)].copy()
    e[["x", "gate"]] = e.setting.apply(lambda s: pd.Series(xvals(sim, s)))
    u[["x", "gate"]] = u.setting.apply(lambda s: pd.Series(xvals(sim, s)))
    rhos = [0.75, 1.0] if setup == "nosolution" else [0.75]
    gates = ["on"] if sim == "sim1" else ["on", "off"]
    fig, axes = plt.subplots(len(rhos), 3, figsize=(7.2, 2.15 * len(rhos)), squeeze=False)
    for i, rho in enumerate(rhos):
        a0, a1, a2 = axes[i]
        for gate, ls in zip(gates, ("-", "--")):
            gl = "" if sim == "sim1" else f", gate {gate}"
            for res, mk, lab in ((SWARM, "o", "swarm"), ("agent_votes", "s", "one agent (votes)")):
                x = e[(e.rho == rho) & (e.gate == gate) & (e.resolution == res)].sort_values("x")
                a0.errorbar(x.x, x.I, yerr=[x.I - x.I_lo, x.I_hi - x.I], ls=ls, marker=mk, ms=3, lw=1, capsize=1.5,
                            color="k" if res == SWARM else "0.55", label=f"{lab}{gl}")
            for res, q, col, lab in (("agent_votes", "eta", "tab:green", "η, one agent (votes)"),
                                     ("agent_beliefs", "eta", "tab:olive", "η, one agent (beliefs)"),
                                     (SWARM, "eta_JS", "tab:purple", "η_JS, swarm")):
                x = e[(e.rho == rho) & (e.gate == gate) & (e.resolution == res)].sort_values("x")
                x = x[(x.C >= 0.01) | (q == "eta_JS")]
                a1.errorbar(x.x, x[q], yerr=[x[q] - x[f"{q}_lo"], x[f"{q}_hi"] - x[q]], ls=ls, marker="o", ms=2.5,
                            lw=1, capsize=1.5, color=col, label=f"{lab}{gl}")
            for arm in ("truth", "false"):
                for m, mk in (("Hbar", "o"), ("JSD", "^")):
                    x = u[(u.rho == rho) & (u.gate == gate) & (u.arm == arm) & (u.measure == m)].sort_values("x")
                    a2.plot(x.x, x["diff"], ls=ls, marker=mk, ms=3, lw=1, color=TCOL[arm],
                            label=f"Δ{'H̄' if m == 'Hbar' else 'JSD'}, target {'A0' if arm == 'truth' else 'A2'}{gl}")
        for ax in (a0, a1, a2):
            if sim == "sim2":
                ax.set_xscale("log", base=2)
                ax.set_xticks([0.5, 1, 2, 4, 8], ["0.5", "1", "2", "4", "8"])
            else:
                ax.set_xticks([1, 2, 3, 6, 9])
            ax.set_xlabel("controller rate $\\lambda_c$" if sim == "sim2" else "facts per night $b$")
        a2.axhline(0, color="0.6", lw=0.7)
        a0.set_title(f"steering $I$ (bits), $\\rho = {rho:g}$")
        a1.set_title(f"efficiency, $\\rho = {rho:g}$")
        a2.set_title(f"agents' uncertainty (bits), $\\rho = {rho:g}$")
        a0.set_ylim(bottom=0)
        a1.set_ylim(0, 1.02)
        if i == 0:
            a0.legend(loc="best", fontsize=5.5)
            a1.legend(loc="lower left", fontsize=5.5)
            a2.legend(loc="best", fontsize=5, ncol=1)
    fig.tight_layout(pad=0.4)
    fig.savefig(out / "fig_eff_settings.pdf")
    plt.close(fig)


def time_figure(e, sim, setup, out):
    e = e[(e.sim == sim) & (e.setup == f"task003_{setup}") & (e.rho == 0.75)]
    fam = [f"b{b}" for b in (1, 2, 3, 6, 9)] if sim == "sim1" else [f"lc{r}_gateon" for r in ("0.5", "1", "2", "4", "8")]
    cols = plt.cm.plasma(np.linspace(0, .85, len(fam)))
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.0))
    for s, c in zip(fam, cols):
        lab = s if sim == "sim1" else f"$\\lambda_c$ {s[2:].split('_')[0]}"
        for ax, res, q in ((axes[0], SWARM, "I"), (axes[1], "agent_votes", "eta"), (axes[2], SWARM, "eta_JS")):
            x = e[(e.setting == s) & (e.resolution == res)].sort_values("t")
            if q == "eta":
                x = x[x.C >= 0.01]
            ax.plot(x.t, x[q], color=c, lw=1, label=lab)
            if q != "eta":
                ax.fill_between(x.t, x[f"{q}_lo"], x[f"{q}_hi"], color=c, alpha=.2, lw=0)
    axes[0].set_title("steering $I(t)$, swarm (bits)")
    axes[1].set_title("efficiency $\\eta(t)$, one agent (votes)")
    axes[2].set_title("efficiency $\\eta_{JS}(t)$, swarm")
    for ax in axes:
        ax.set_xlabel("day" if sim == "sim1" else "time $t$")
        if sim == "sim2":
            ax.axvline(30, color="0.5", ls="--", lw=0.7)
    axes[1].set_ylim(0, 1.02)
    axes[2].set_ylim(0, 1.02)
    axes[0].legend(fontsize=5.5, ncol=1)
    fig.tight_layout(pad=0.4)
    fig.savefig(out / "fig_eff_time.pdf")
    plt.close(fig)


def precision_table(e, ds, sim, setup, out):
    main = "b1" if sim == "sim1" else "lc1_gateon"
    rows = [("steering $I$, swarm (3 bins)", SWARM, "I", None), ("steering $I$, one agent (votes)", "agent_votes", "I", None),
            ("efficiency $\\eta$, one agent (votes)", "agent_votes", "eta", None),
            ("efficiency $\\eta_{JS}$, swarm", SWARM, "eta_JS", None),
            ("$\\Delta\\bar H$, target A0", "paired_truth", "dHbar", None), ("$\\Delta\\bar H$, target A2", "paired_false", "dHbar", None),
            ("$\\Delta$JSD, target A0", "paired_truth", "dJSD", None), ("$\\Delta$JSD, target A2", "paired_false", "dJSD", None)]
    lines = []
    for lab, res, q, _ in rows:
        d = ds[(ds.sim == sim) & (ds.setup == f"task003_{setup}") & (ds.rho == 0.75) & (ds.setting == main)
               & (ds.t == 30) & (ds.resolution == res) & (ds.quantity == q)]
        if d.empty:
            continue
        full = d.full.iloc[0]
        if q in ("eta",) and not np.isfinite(full):
            continue
        err = {int(r.n_sub): r.rmse for r in d.itertuples()}
        inf = {int(r.n_sub): r.share_not_finite for r in d.itertuples()}
        cells = []
        for m in (50, 100, 300):
            txt = f"{err[m]:.3f}" if np.isfinite(err[m]) else "--"
            if inf[m] > 0.005:
                txt += f" ({inf[m]:.0%}$^\\ast$)".replace("%", "\\%")
            cells.append(txt)
        lines.append(f"{lab} & {full:.3f} & " + " & ".join(cells) + " \\\\")
    (out / "tab_eff_precision.tex").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2026-10-12")
    a = ap.parse_args()
    d = RES / a.date
    e = pd.read_csv(d / "efficiency.csv")
    u = pd.read_csv(d / "uncertainty.csv")
    ds = pd.read_csv(d / "downsample.csv")
    for (sim, setup), rep in REPORTS.items():
        out = rep / "figures"
        out.mkdir(parents=True, exist_ok=True)
        settings_figure(e, u, sim, setup, out)
        time_figure(e, sim, setup, out)
        precision_table(e, ds, sim, setup, out)
        print("written", out)


if __name__ == "__main__":
    main()
