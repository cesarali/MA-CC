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
import matplotlib.colors  # noqa: E402
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
    main = "b3" if sim == "sim1" else "lc1_gateon"
    rows = [("steering $I$, swarm (3 bins)", SWARM, "I", None), ("steering $I$, one agent (votes)", "agent_votes", "I", None),
            ("efficiency $\\eta$, one agent (votes)", "agent_votes", "eta", None),
            ("efficiency $\\eta_{JS}$, swarm", SWARM, "eta_JS", None),
            ("progress $\\pi$, target A0 (votes)", "agent_votes", "progress_plus", None),
            ("progress $\\pi$, target A2 (votes)", "agent_votes", "progress_minus", None),
            ("focus $\\varphi$, target A0 (votes)", "agent_votes", "focus_plus", None),
            ("focus $\\varphi$, target A2 (votes)", "agent_votes", "focus_minus", None),
            ("$\\Delta\\bar H$, target A0", "paired_truth", "dHbar", None), ("$\\Delta\\bar H$, target A2", "paired_false", "dHbar", None),
            ("$\\Delta$JSD, target A0", "paired_truth", "dJSD", None), ("$\\Delta$JSD, target A2", "paired_false", "dJSD", None)]
    lines = []
    for lab, res, q, _ in rows:
        d = ds[(ds.sim == sim) & (ds.setup == f"task003_{setup}") & (ds.rho == 0.75) & (ds.setting == main)
               & (ds.t == 30) & (ds.resolution == res) & (ds.quantity == q)]
        if d.empty:
            continue
        full = d.full.iloc[0]
        if q in ("eta", "progress_plus", "progress_minus", "focus_plus", "focus_minus") and not np.isfinite(full):
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


def phase_figure(ph, sim, setup, out):
    """eta (one agent, beliefs) and eta_JS (swarm) over qc x rate (Simulation 2) or qc x b (Simulation 1)."""
    ph = ph[(ph.sim == sim) & (ph.setup == f"task003_{setup}")]
    rhos = [0.75, 1.0] if setup == "nosolution" else [0.75]
    xs = [0.5, 1.0, 2.0, 4.0, 8.0] if sim == "sim2" else [1, 2, 3, 6, 9]
    qcs = [3, 6, 12, 24]
    panels = [("agent_beliefs", "eta", "η, one agent", g) for g in (["on", "off"] if sim == "sim2" else ["on"])]
    panels += [("swarm_beliefs_k3", "eta_JS", "η$_{JS}$, swarm", g) for g in (["on", "off"] if sim == "sim2" else ["on"])]
    fig, axes = plt.subplots(len(rhos), len(panels), figsize=(1.85 * len(panels) + 0.7, 1.65 * len(rhos) + 0.35),
                             squeeze=False)
    for i, rho in enumerate(rhos):
        for j, (res, q, lab, gate) in enumerate(panels):
            ax = axes[i, j]
            m = np.full((len(qcs), len(xs)), np.nan)
            for r in ph[(ph.rho == rho) & (ph.resolution == res) & (ph.gate == gate)].itertuples():
                if q == "eta" and not (np.isfinite(r.C) and r.C >= 0.01):
                    continue
                m[qcs.index(r.qc), xs.index(r.x)] = getattr(r, q)
            im = ax.imshow(m, origin="lower", cmap="viridis", vmin=0, vmax=1, aspect="auto")
            for a_ in range(len(qcs)):
                for b_ in range(len(xs)):
                    txt = "–" if np.isnan(m[a_, b_]) else f"{m[a_, b_]:.2f}"
                    ax.text(b_, a_, txt, ha="center", va="center", fontsize=5.5,
                            color="white" if (np.isnan(m[a_, b_]) or m[a_, b_] < 0.55) else "black")
            ax.set_xticks(range(len(xs)), [f"{x:g}" for x in xs], fontsize=6)
            ax.set_yticks(range(len(qcs)), qcs, fontsize=6)
            gl = f", gate {gate}" if sim == "sim2" else ""
            if i == 0:
                ax.set_title(f"{lab}{gl}", fontsize=6.5)
            if i == len(rhos) - 1:
                ax.set_xlabel("$\\lambda_c$" if sim == "sim2" else "$b$", fontsize=7)
            if j == 0:
                ax.set_ylabel(f"$\\rho$ = {rho:g}\n$q_c$", fontsize=7)
    fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    fig.savefig(out / "fig_eff_phase.pdf", bbox_inches="tight")
    plt.close(fig)


SPEC = {"sim1": "b3", "sim2": "lc1_gateon"}        # the one setting shown over time (rho 0.75, qc 12)


def time2_figure(e, u, sim, setup, out):
    """At one setting: steering I, progress of each controller, directedness, agents' uncertainty, over time."""
    st = SPEC[sim]
    e = e[(e.sim == sim) & (e.setup == f"task003_{setup}") & (e.rho == 0.75) & (e.setting == st)]
    u = u[(u.sim == sim) & (u.setup == f"task003_{setup}") & (u.rho == 0.75) & (u.setting == st)]
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 1.95))
    av = e[e.resolution == "agent_votes"].sort_values("t")
    sw = e[e.resolution == SWARM].sort_values("t")
    ax = axes[0]
    for x, col, lab in ((sw, "k", "swarm"), (av, "0.55", "one agent")):
        ax.plot(x.t, x.I, color=col, lw=1, label=lab)
        ax.fill_between(x.t, x.I_lo, x.I_hi, color=col, alpha=.2, lw=0)
    ax.set_title("steering $I(t)$ (bits)")
    ax = axes[1]
    for k, col, lab in (("plus", TCOL["truth"], "target A0"), ("minus", TCOL["false"], "target A2")):
        ax.plot(av.t, av[f"progress_{k}"], color=col, lw=1, label=lab)
        ax.fill_between(av.t, av[f"progress_{k}_lo"], av[f"progress_{k}_hi"], color=col, alpha=.2, lw=0)
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set_title("progress $\\pi_z(t)$")
    ax = axes[2]
    for q, x, col, lab in (("focus_plus", av, TCOL["truth"], "focus, A0"), ("focus_minus", av, TCOL["false"], "focus, A2"),
                           ("eta", av, "tab:green", "$\\eta$, one agent"), ("eta_JS", sw, "tab:purple", "$\\eta_{JS}$, swarm")):
        y = x[x.C >= 0.01] if q == "eta" else x
        ax.plot(y.t, y[q], color=col, lw=1, label=lab)
    ax.set_ylim(min(0, float(np.nanmin(av[["focus_plus", "focus_minus"]].to_numpy())) - 0.05), 1.03)
    ax.set_title("directedness")
    ax = axes[3]
    for arm, col in (("truth", TCOL["truth"]), ("false", TCOL["false"])):
        for m, ls in (("Hbar", "-"), ("JSD", "--")):
            x = u[(u.arm == arm) & (u.measure == m)].sort_values("t")
            mlab = "$\\Delta\\bar H$" if m == "Hbar" else "$\\Delta$JSD"
            tlab = "A0" if arm == "truth" else "A2"
            ax.plot(x.t, x["diff"], color=col, ls=ls, lw=1, label=f"{mlab}, {tlab}")
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set_title("agents' uncertainty (bits)")
    for ax in axes:
        ax.set_xlabel("day" if sim == "sim1" else "time $t$")
        if sim == "sim2":
            ax.axvline(30, color="0.5", ls="--", lw=0.6)
        ax.legend(fontsize=4.8, loc="best")
    fig.tight_layout(pad=0.3)
    fig.savefig(out / "fig_eff_time2.pdf")
    plt.close(fig)


def _tiles(ax, m, xs, qcs, fmt, vmin, vmax, cmap):
    im = ax.imshow(m, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    for a_ in range(len(qcs)):
        for b_ in range(len(xs)):
            v = m[a_, b_]
            txt = "–" if np.isnan(v) else format(v, fmt)
            dark = (not np.isnan(v)) and (0.299 * plt.get_cmap(cmap)((min(max(v, vmin), vmax) - vmin) / (vmax - vmin))[0]
                                          + 0.587 * plt.get_cmap(cmap)((min(max(v, vmin), vmax) - vmin) / (vmax - vmin))[1]
                                          + 0.114 * plt.get_cmap(cmap)((min(max(v, vmin), vmax) - vmin) / (vmax - vmin))[2]) < 0.5
            ax.text(b_, a_, txt, ha="center", va="center", fontsize=5.0, color="white" if dark else "black")
    ax.set_xticks(range(len(xs)), [f"{x:g}" for x in xs], fontsize=5.5)
    ax.set_yticks(range(len(qcs)), qcs, fontsize=5.5)
    return im


def phase_grid(ph, sim, setup, out, name, panels, rows_rho, cmap, vmin, vmax, fmt, clabel):
    ph = ph[(ph.sim == sim) & (ph.setup == f"task003_{setup}") & (ph.resolution == "agent_votes")]
    xs = [0.5, 1.0, 2.0, 4.0, 8.0] if sim == "sim2" else [1, 2, 3, 6, 9]
    qcs = [3, 6, 12, 24]
    fig, axes = plt.subplots(len(rows_rho), len(panels), figsize=(1.6 * len(panels) + 0.7, 1.45 * len(rows_rho) + 0.35),
                             squeeze=False)
    for i, rho in enumerate(rows_rho):
        for j, (col, lab, gate) in enumerate(panels):
            m = np.full((len(qcs), len(xs)), np.nan)
            for r in ph[(ph.rho == rho) & (ph.gate == gate)].itertuples():
                v = getattr(r, col)
                if np.isfinite(v):
                    m[qcs.index(r.qc), xs.index(r.x)] = v
            im = _tiles(axes[i, j], m, xs, qcs, fmt, vmin, vmax, cmap)
            if i == 0:
                axes[i, j].set_title(lab + (f", gate {gate}" if sim == "sim2" else ""), fontsize=6)
            if i == len(rows_rho) - 1:
                axes[i, j].set_xlabel("$\\lambda_c$" if sim == "sim2" else "$b$", fontsize=6.5)
            if j == 0:
                axes[i, j].set_ylabel(f"$\\rho$ = {rho:g}\n$q_c$", fontsize=6.5)
    fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02, label=clabel)
    fig.savefig(out / name, bbox_inches="tight")
    plt.close(fig)


def phase_new(ph, sim, setup, out):
    rhos = [0.75, 1.0] if setup == "nosolution" else [0.75]
    gates = ["on", "off"] if sim == "sim2" else ["on"]
    prog = [(f"progress_{k}", f"$\\pi$, {t}", g) for k, t in (("plus", "target A0"), ("minus", "target A2")) for g in gates]
    phase_grid(ph, sim, setup, out, "fig_eff_phase_progress.pdf", prog, rhos, "RdBu", -0.5, 0.5, ".2f",
               "progress $\\pi_z$")
    fp = [("focus_plus", "$\\varphi$, target A0", "on"), ("focus_minus", "$\\varphi$, target A2", "on")]
    phase_grid(ph, sim, setup, out, "fig_eff_phase_focus.pdf", fp, rhos, "RdBu", -1, 1, ".2f", "focus $\\varphi_z$")
    pr = [("S_per_fact_plus", "bits/fact, target A0", "on"), ("S_per_fact_minus", "bits/fact, target A2", "on")]
    vmax = float(np.nanpercentile(ph[(ph.sim == sim) & (ph.setup == f"task003_{setup}") & (ph.resolution == "agent_votes")]
                                  [["S_per_fact_plus", "S_per_fact_minus"]].replace(np.inf, np.nan).to_numpy(), 100))
    phase_grid(ph, sim, setup, out, "fig_eff_phase_price.pdf", pr, rhos, "magma_r", 0, vmax, ".3f", "bits per fact posted")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2026-10-12")
    a = ap.parse_args()
    d = RES / a.date
    e = pd.read_csv(d / "efficiency.csv")
    u = pd.read_csv(d / "uncertainty.csv")
    ds = pd.read_csv(d / "downsample.csv")
    ph = pd.read_csv(d / "efficiency_phase.csv")
    for (sim, setup), rep in REPORTS.items():
        out = rep / "figures"
        out.mkdir(parents=True, exist_ok=True)
        settings_figure(e, u, sim, setup, out)
        time_figure(e, sim, setup, out)
        precision_table(e, ds, sim, setup, out)
        phase_figure(ph, sim, setup, out)
        time2_figure(e, u, sim, setup, out)
        phase_new(ph, sim, setup, out)
        print("written", out)


if __name__ == "__main__":
    main()
