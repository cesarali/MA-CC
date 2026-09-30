"""LaTeX report: figures (matplotlib -> PDF), booktabs tables, equations, analysis text.

    python -m bound_estimators.report_tex --config <cfg>

Writes ``<output_dir>/report/{main.tex, figures/, tables/}``, compiles with latexmk and
copies the PDF to ``<output_dir>/control_efficiency_bound_estimators_report.pdf``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TEXTWIDTH = 6.3  # inches (letter, 1in margins)
LN2 = float(np.log(2))
SETTINGS = [(3, 0.7), (3, 1.0), (12, 0.7), (12, 1.0)]
POLICIES = [("always", 3), ("always", 12), ("sensing", 3), ("sensing", 12)]
COLORS = {("always", 3): "#6baed6", ("always", 12): "#08519c", ("sensing", 3): "#fd8d3c", ("sensing", 12): "#a63603"}
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "figure.dpi": 150, "axes.grid": True,
                     "grid.alpha": 0.3, "grid.linewidth": 0.5, "lines.linewidth": 1.1, "lines.markersize": 3})


# ----------------------------------------------------------------------------- helpers
def _key_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Split comparison key into q, rho, schedule, budget display columns."""
    out = df.copy()
    out["q"] = out["comparison"].str.extract(r"q(\d+)_")[0].astype(int)
    out["rho"] = out["comparison"].str.extract(r"rho([\d.]+)_")[0].astype(float)
    out["schedule"] = out["comparison"].str.extract(r"_(always|sensing)_")[0]
    out["budget"] = out["comparison"].str.extract(r"_b(\d+)$")[0].astype(int)
    return out


def _sort(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["q", "rho", "schedule", "budget"])


def f3(v, d=3):
    return "" if pd.isna(v) else f"{v:.{d}f}"


def ci(x, lo, hi, d=2):
    if pd.isna(x):
        return ""
    return f"{x:.{d+1}f} [{lo:.{d}f}, {hi:.{d}f}]"


def sched_short(s):
    return "alw." if s == "always" else "sens."


def esc(s: str) -> str:
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def model_tex(name: str) -> str:
    """'quadratic-l0.1' -> 'quad, $\\lambda$=0.1'; 'gru-w4-l0.01' -> 'GRU-4, $\\lambda$=0.01'."""
    parts = name.split("-")
    fam = {"constant": "const", "linear": "lin", "quadratic": "quad", "mlp": "MLP", "flat": "flat",
           "gru": "GRU", "gru_mean": "GRU-mean", "gru_direct": "GRU-dir"}[parts[0]]
    w = [p[1:] for p in parts if p.startswith("w")]
    lam = [p[1:] for p in parts if p.startswith("l")]
    s = fam + (f"-{w[0]}" if w else "")
    if lam:
        s += f", $\\lambda$={lam[0]}"
    return s


def booktabs(df: pd.DataFrame, colspec: str, header: list[str], caption: str, label: str,
             size: str = "footnotesize", note: str | None = None, longtable: bool = False,
             extra_header: str | None = None, fit: bool = False) -> str:
    rows = [" & ".join(str(v) for v in r) + r" \\" for r in df.values.tolist()]
    head = " & ".join(header) + r" \\"
    if longtable:
        body = "\n".join([
            f"\\begin{{{size}}}",
            r"\setlength{\tabcolsep}{3pt}",
            f"\\begin{{longtable}}{{{colspec}}}",
            f"\\caption{{{caption}}}\\label{{{label}}}\\\\",
            r"\toprule", *( [extra_header] if extra_header else []), head, r"\midrule", r"\endfirsthead",
            r"\toprule", *( [extra_header] if extra_header else []), head, r"\midrule", r"\endhead",
            r"\midrule", r"\multicolumn{" + str(len(header)) + r"}{r}{\emph{continued on next page}}\\", r"\endfoot",
            r"\bottomrule", r"\endlastfoot",
            *rows,
            r"\end{longtable}",
            f"\\end{{{size}}}"])
        return body
    lines = [r"\begin{table}[htbp]", r"\centering", f"\\{size}", f"\\caption{{{caption}}}\\label{{{label}}}"]
    if note:
        lines += [r"\begin{threeparttable}"]
    if fit:
        lines.append(r"\begin{adjustbox}{max width=\textwidth}")
    lines += [f"\\begin{{tabular}}{{{colspec}}}", r"\toprule"]
    if extra_header:
        lines.append(extra_header)
    lines += [head, r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"]
    if fit:
        lines.append(r"\end{adjustbox}")
    if note:
        lines += [r"\begin{tablenotes}\footnotesize", r"\item " + note, r"\end{tablenotes}", r"\end{threeparttable}"]
    lines += [r"\end{table}"]
    return "\n".join(lines)


def setting_grid(nrows=2, ncols=2, height=4.2, sharex=True, sharey=False):
    fig, axes = plt.subplots(nrows, ncols, figsize=(TEXTWIDTH, height), sharex=sharex, sharey=sharey)
    return fig, np.atleast_2d(axes)


# ----------------------------------------------------------------------------- figures
def fig_info_horizon(ep: pd.DataFrame, path: Path):
    fig, axes = setting_grid(height=4.4)
    for ax, (q, rho) in zip(axes.ravel(), SETTINGS):
        for sched, b in POLICIES:
            d = ep[(ep.q == q) & np.isclose(ep.rho, rho) & (ep.schedule == sched) & (ep.budget == b)].sort_values("horizon")
            ax.plot(d.horizon, d.score, marker="o", color=COLORS[(sched, b)], label=f"{sched}, $b$={b}")
            ax.fill_between(d.horizon, d.score_lo, d.score_hi, color=COLORS[(sched, b)], alpha=0.15, lw=0)
        ax.axhline(LN2, color="k", lw=0.7, ls="--"); ax.axhline(0, color="gray", lw=0.6)
        ax.set_title(f"$q$={q}, $\\rho$={rho:.2f}"); ax.set_ylim(-0.12, 0.72)
    for ax in axes[-1]:
        ax.set_xlabel("horizon $h$ (rounds after checkpoint)")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\widehat L_I(h)$ [nats]")
    axes[0, 0].legend(loc="upper left", ncol=2)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_direction(direction: pd.DataFrame, path: Path):
    d = _sort(direction[direction.horizon == 10])
    labels = [f"{r.q}/{r.rho:.1f}/{sched_short(r.schedule)}/{r.budget}" for r in d.itertuples()]
    x = np.arange(len(d)); w = 0.27
    fig, axes = plt.subplots(2, 1, figsize=(TEXTWIDTH, 4.0), sharex=True)
    for ax, a in zip(axes, (0, 2)):
        ax.bar(x - w, d[f"silent_mean_N{a}"], w, color="#999999", label="silent")
        ax.bar(x, d[f"target0_mean_N{a}"], w, color="#2c7bb6", label="target 0 (correct)")
        ax.bar(x + w, d[f"target2_mean_N{a}"], w, color="#d7191c", label="target 2 (false)")
        ax.set_ylabel(f"mean share on ALLOCATION\\_{a}" if False else f"mean share on allocation {a}"); ax.set_ylim(0, 1.02)
    axes[0].legend(ncol=3, loc="lower left"); axes[1].set_xticks(x); axes[1].set_xticklabels(labels, rotation=60, ha="right", fontsize=6.5)
    axes[1].set_xlabel("comparison ($q$ / $\\rho$ / schedule / $b$)")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_cost_horizon(c0: pd.DataFrame, c2: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(2, 4, figsize=(TEXTWIDTH, 3.6), sharex=True)
    for row, (z, df) in enumerate(((0, c0), (2, c2))):
        for col, (q, rho) in enumerate(SETTINGS):
            ax = axes[row, col]
            for sched, b in POLICIES:
                d = df[(df.q == q) & np.isclose(df.rho, rho) & (df.schedule == sched) & (df.budget == b)].sort_values("horizon")
                ax.plot(d.horizon, d.nwj_cap5, marker="o", color=COLORS[(sched, b)], label=f"{sched}, $b$={b}")
                ax.plot(d.horizon, d.dv_raw.clip(-0.5, 12.5), ls=":", color=COLORS[(sched, b)], lw=0.9)
            ax.axhline(0, color="gray", lw=0.6)
            if row == 0:
                ax.set_title(f"$q$={q}, $\\rho$={rho:.2f}")
            if col == 0:
                ax.set_ylabel(f"target {z}: score [nats]")
            if row == 1:
                ax.set_xlabel("$h$")
    axes[0, 0].legend(loc="upper left", fontsize=6)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_heatmap(df: pd.DataFrame, path: Path, vmin: float, vmax: float, title: str, height: float, cbar_label: str):
    d = _sort(df)
    cand = [c for c in d.columns if c.startswith("cand:") and d[c].notna().any()]
    M = d[cand].to_numpy().T
    fig, ax = plt.subplots(figsize=(TEXTWIDTH, height))
    im = ax.imshow(np.clip(M, vmin, vmax), aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_yticks(range(len(cand))); ax.set_yticklabels([c[5:] for c in cand], fontsize=6.5)
    ax.set_xticks(range(len(d))); ax.set_xticklabels([f"{r.q}/{r.rho:.1f}/{sched_short(r.schedule)}/{r.budget}" for r in d.itertuples()],
                                                     rotation=60, ha="right", fontsize=6.5)
    ax.grid(False); ax.set_title(title)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, f"{v:.2f}" if abs(v) < 10 else ("<" if v < 0 else ">"), ha="center", va="center", fontsize=4.6,
                    color="white" if (np.clip(v, vmin, vmax) - vmin) / (vmax - vmin) < 0.55 else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01); cb.set_label(cbar_label, fontsize=7); cb.ax.tick_params(labelsize=6)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_ess(c0: pd.DataFrame, c2: pd.DataFrame, path: Path):
    both = pd.concat([c0.assign(target=0), c2.assign(target=2)])
    both = both[both.horizon == 10]
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.6))
    for ax, col, lab in zip(axes, ("nwj_raw", "nwj_cap5"), ("raw logits", "critic capped at $M=5$")):
        for sched, mk in (("always", "s"), ("sensing", "o")):
            for z, colr in ((0, "#2c7bb6"), (2, "#d7191c")):
                d = both[(both.schedule == sched) & (both.target == z)]
                ax.scatter(d.ess_raw, d[col].clip(-3, 9), marker=mk, color=colr, s=16, alpha=0.85, label=f"{sched}, target {z}")
        ax.set_xscale("log"); ax.set_xlabel("effective sample size of $e^{f}$ on silent paths"); ax.set_ylabel(f"NWJ score, {lab} [nats]")
        ax.axhline(0, color="gray", lw=0.6); ax.axvline(5, color="gray", lw=0.6, ls=":")
    axes[0].legend(fontsize=6, loc="lower right")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_efficiency(eff: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.8), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    for h, mk in ((5, "^"), (10, "o")):
        for sched, colr in (("always", "#08519c"), ("sensing", "#a63603")):
            d = eff[(eff.horizon == h) & (eff.schedule == sched)]
            ax.scatter(d.cost_cap5, d["info"], marker=mk, color=colr, s=18, label=f"{sched}, $h$={h}")
    xs = np.linspace(0.05, 9, 100)
    xmax, ymax = 9.3, 0.48
    for eta in (0.02, 0.05, 0.1, 0.2):
        ax.plot(xs, eta * xs, color="gray", lw=0.6, ls="--")
        # label where the line leaves the axes, not at a fixed x: a steep line is already far
        # above the top of the panel by x=8.6, so a fixed x puts the label outside the axes.
        lx = min(xmax, ymax / eta)
        ax.text(lx, eta * lx, f"$\\eta$={eta}", fontsize=6, color="gray", va="bottom", ha="right")
    ax.set_xlim(-0.2, 9.5); ax.set_ylim(-0.06, 0.5); ax.set_xlabel(r"$\widehat{C}$ (NWJ, cap 5) [nats]"); ax.set_ylabel(r"$\widehat L_I$ [nats]")
    # upper left: the points sit low and left-to-centre, and "upper right" collided with the
    # right panel's row labels, which are drawn in the gap between the two panels.
    ax.legend(fontsize=6, loc="upper left", framealpha=0.9)
    ax = axes[1]
    d = _sort(eff[(eff.horizon == 10) & (eff.eta_raw_supported == True)].copy())
    y = np.arange(len(d))
    ax.errorbar(d.eta_raw, y, xerr=[d.eta_raw - d.eta_raw_lo, d.eta_raw_hi - d.eta_raw], fmt="o", color="k", ms=3, capsize=2, lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels([f"{r.q}/{r.rho:.1f}/{sched_short(r.schedule)}/{r.budget}" for r in d.itertuples()], fontsize=6.5)
    ax.axvline(0, color="gray", lw=0.6); ax.set_xlabel(r"$\widehat{\eta}_{end}(10)$ with 95% interval"); ax.invert_yaxis()
    fig.tight_layout(w_pad=2.0); fig.savefig(path); plt.close(fig)


def fig_synthetic(syn: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, 2.3))
    specs = [("exact_info_end_h3", "est_info_end", r"$I(Z;Y_3)$"), ("exact_cost", "est_cost_nwj", r"$C$ (NWJ)"),
             ("exact_info_path", "est_info_path", r"$I(Z;\Gamma)$")]
    markers = {"identical": "o", "generic": "s", "opposite": "^", "transient": "v", "rare_tail": "D", "reversed": "P"}
    for ax, (ex, es, lab) in zip(axes, specs):
        for sc, mk in markers.items():
            for m_, colr in ((40, "#fd8d3c"), (200, "#08519c")):
                d = syn[(syn.scenario == sc) & (syn.m == m_)]
                ax.scatter(d[ex], d[es], marker=mk, color=colr, s=16, alpha=0.85, label=f"{sc}" if m_ == 40 else None)
        lim = [-0.05, max(syn[ex].max(), syn[es].max()) * 1.08]
        ax.plot(lim, lim, "k--", lw=0.7); ax.set_xlim(lim); ax.set_ylim(lim); ax.set_xlabel("exact [nats]"); ax.set_title(lab)
    axes[0].set_ylabel("estimate [nats]")
    axes[2].legend(fontsize=5.5, loc="lower right", title="scenario (orange $m$=40, blue $m$=200)", title_fontsize=5.5)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_swaps(swaps: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, 2.3))
    for ax, prob, lim in zip(axes, ("endpoint", "cost0", "cost2"), ((-0.06, 0.4), (-1.5, 9.5), (-1.5, 9.5))):
        d = swaps[swaps.problem == prob]
        for h, colr in ((1, "#fd8d3c"), (10, "#08519c")):
            dd = d[d.horizon == h]
            ax.scatter(dd.null_q95.clip(*lim), dd.observed.clip(*lim), color=colr, s=14, label=f"$h$={h}")
        ax.plot(lim, lim, "k--", lw=0.7); ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("null 95th percentile"); ax.set_title({"endpoint": r"endpoint $\widehat L_I$", "cost0": r"cost, target 0", "cost2": r"cost, target 2"}[prob])
    axes[0].set_ylabel("observed score"); axes[0].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_memory(mem: pd.DataFrame, full: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(TEXTWIDTH * 0.62, 2.4))
    xs = [1, 2, 4, 10]
    allv = []
    for (comp, prob), g in mem.groupby(["comparison", "problem"]):
        g = g.sort_values("history")
        f = full[(full.comparison == comp) & (full.problem == prob)]
        ys = list(g.nwj_cap5) + [float(f.nwj_cap5.iloc[0]) if len(f) else np.nan]
        allv.append(ys)
        ax.plot(xs, ys, color="#2c7bb6" if prob == "cost0" else "#d7191c", alpha=0.25, lw=0.7)
    arr = np.array(allv)
    ax.plot(xs, np.nanmedian(arr, axis=0), color="k", marker="o", lw=1.4, label="median over 32 problems")
    ax.set_xscale("log"); ax.set_xticks(xs); ax.set_xticklabels(["1", "2", "4", "10 (full)"])
    ax.set_xlabel(r"history length $\ell$ (rounds fed to the GRU, plus $Y_0$)"); ax.set_ylabel("NWJ score, cap 5 [nats]")
    ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_curves(curves: dict[str, Any], path: Path):
    keys = list(curves)[:8]
    fig, axes = plt.subplots(2, 4, figsize=(TEXTWIDTH, 3.2))
    for ax, k in zip(axes.ravel(), keys):
        c = curves[k]
        ax.plot(c["epoch"], c["train"], label="train"); ax.plot(c["epoch"], c["val"], label="inner validation")
        comp, prob, name = k.split("/")
        q, rho, sched, b = comp.split("_")
        ax.set_title(f"{q[1:]}/{rho[3:]}/{sched[:4]}/{b[1:]}\n{prob}: {name}", fontsize=6.5)
        ax.tick_params(labelsize=6); ax.set_xlabel("epoch", fontsize=6.5)
    axes[0, 0].legend(fontsize=6); axes[0, 0].set_ylabel("BCE"); axes[1, 0].set_ylabel("BCE")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


# ----------------------------------------------------------------------------- tables
def tab_sample(audit: dict) -> str:
    cc = audit["comparison_counts"]
    rows = []
    for q, rho in SETTINGS:
        rows.append([q, f"{rho:.2f}"] + [cc.get(f"q{q}_rho{rho:.2f}_{s}_b{b}", 0) for s, b in POLICIES])
    return booktabs(pd.DataFrame(rows), "cc cccc", ["$q$", r"$\rho$", "always, $b$=3", "always, $b$=12", "sensing, $b$=3", "sensing, $b$=12"],
                    "Parents with a complete silent / target-0 / target-2 triplet of ten-round paths, per comparison.", "tab:sample", size="small")


def tab_models() -> str:
    rows = [
        [r"constant", "endpoint / path", "--", "0", r"$g=\tfrac12$ or $f=0$; scores exactly zero"],
        [r"linear logistic", "endpoint", r"$u=N_0/24,\ v=N_2/24$", "3", r"$\lambda\in\{0.01,0.1,1,10\}$"],
        [r"quadratic logistic", "endpoint", r"$u,v,u^2,uv,v^2$", "6", "same $\\lambda$ grid"],
        [r"MLP", "endpoint", r"$u,v \to \tanh(4\ \text{or}\ 8) \to \sigma$", "17 / 33", r"$\lambda\in\{0.01,0.1\}$, 3 seeds averaged"],
        [r"linear logistic", "path", r"$Y_0/24,\ Y_h/24,\ \overline{Y_{1:h}}/24$ (6 numbers)", "7", "same $\\lambda$ grid"],
        [r"quadratic logistic", "path", "the 6 summaries and their 21 products", "28", "same $\\lambda$ grid"],
        [r"flat MLP", "path", r"$(u_t,v_t,t/10)_{t=0..h}$ flattened $\to \tanh(4/8)$", "141 / 281", r"$\lambda\in\{0.01,0.1\}$"],
        [r"GRU", "path", r"sequence $(u_t,v_t,t/10)$, hidden 4 / 8, last state $\to$ linear", "113 / 321", r"$\lambda\in\{0.01,0.1\}$"],
        [r"GRU-mean", "path", "as GRU, mean-pooled hidden states, hidden 4", "113", "RNEEP-style comparison"],
        [r"GRU-direct", "path", r"GRU-4 trained on $L_{\rm NWJ}$, $f = 5\tanh(s/5)$", "113", "separate output"],
    ]
    return booktabs(pd.DataFrame(rows), r"l l p{5.2cm} c p{3.6cm}", ["model", "problem", "inputs", "params", "notes"],
                    "Candidate models. Endpoint models output $P(Z=0\\mid Y_h)$; path models output a logit that serves as the critic $f$. All standardisation is fitted on training parents only.", "tab:models", size="footnotesize")


def tab_endpoint(ep: pd.DataFrame, swaps: pd.DataFrame) -> str:
    rows = []
    sw = swaps[(swaps.problem == "endpoint")]
    for comp, g in _sort(ep).groupby("comparison", sort=False):
        r = g.iloc[0]
        g = g.set_index("horizon")
        p10 = sw[(sw.comparison == comp) & (sw.horizon == 10)].p_value
        rows.append([r.q, f"{r.rho:.2f}", sched_short(r.schedule), r.budget,
                     ci(g.loc[1, "score"], g.loc[1, "score_lo"], g.loc[1, "score_hi"]),
                     ci(g.loc[5, "score"], g.loc[5, "score_lo"], g.loc[5, "score_hi"]),
                     ci(g.loc[10, "score"], g.loc[10, "score_lo"], g.loc[10, "score_hi"]),
                     model_tex(g.loc[10, "selected_mode"]), f3(g.loc[10, "accuracy"], 2),
                     f3(float(p10.iloc[0]), 3) if len(p10) else ""])
    return booktabs(pd.DataFrame(rows), "cclc lll l cc",
                    ["$q$", r"$\rho$", "sched.", "$b$", r"$\widehat L_I(1)$", r"$\widehat L_I(5)$", r"$\widehat L_I(10)$", "model ($h$=10)", "acc.", "$p_{\\rm swap}$"],
                    r"Endpoint target-information score $\widehat L_I(h)$ (nats; maximum $\log 2=0.693$) with 95\% paired-parent bootstrap intervals, the model selected at $h=10$ (mode over outer folds), its held-out accuracy, and the whole-parent label-swap $p$-value at $h=10$ (200 swaps; $0.005$ is the smallest attainable value).",
                    "tab:endpoint", size="scriptsize", fit=True)


def tab_direction(direction: pd.DataFrame) -> str:
    rows = []
    for r in _sort(direction[direction.horizon == 10]).itertuples():
        rows.append([r.q, f"{r.rho:.2f}", sched_short(r.schedule), r.budget,
                     f3(r.silent_mean_N0, 2), f3(r.silent_mean_N2, 2), f3(r.target0_mean_N0, 2), f3(r.target0_mean_N2, 2),
                     f3(r.target2_mean_N0, 2), f3(r.target2_mean_N2, 2),
                     f"{r.gain_toward_target0:+.2f} ({r.gain_toward_target0_se:.2f})", f"{r.gain_toward_target2:+.2f} ({r.gain_toward_target2_se:.2f})"])
    extra = r"\multicolumn{4}{c}{} & \multicolumn{2}{c}{silent} & \multicolumn{2}{c}{target 0} & \multicolumn{2}{c}{target 2} & \multicolumn{2}{c}{paired gain (s.e.)} \\ \cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}\cmidrule(lr){11-12}"
    return booktabs(pd.DataFrame(rows), "cclc cc cc cc cc",
                    ["$q$", r"$\rho$", "sched.", "$b$", "$N_0/24$", "$N_2/24$", "$N_0/24$", "$N_2/24$", "$N_0/24$", "$N_2/24$", r"$\to 0$", r"$\to 2$"],
                    r"Direction check at $h=10$: mean vote shares on the correct answer (allocation 0) and the false target (allocation 2) under silence and under each target, and the paired gain toward each target relative to silence, $\frac1m\sum_i\big[N_z^{(i,z)}(10)-N_z^{(i,\rm base)}(10)\big]/24$.",
                    "tab:direction", size="scriptsize", fit=True, extra_header=extra)


def tab_cost(c0: pd.DataFrame, c2: pd.DataFrame, direct: pd.DataFrame) -> str:
    rows = []
    both = pd.concat([c0.assign(z=0), c2.assign(z=2)])
    both = _sort(both[both.horizon == 10]).sort_values(["q", "rho", "schedule", "budget", "z"])
    for r in both.itertuples():
        dd = direct[(direct.comparison == r.comparison) & (direct.problem == f"cost{r.z}")]
        rows.append([r.q, f"{r.rho:.2f}", sched_short(r.schedule), r.budget, r.z,
                     f3(r.nwj_raw, 2), f3(r.nwj_cap5, 2), f3(r.dv_raw, 2), f3(r.plugin_raw, 2),
                     f3(float(dd.nwj_cap5.iloc[0]), 2) if len(dd) else "", f3(r.accuracy, 2), f3(r.ess_raw, 1), model_tex(r.selected_mode)])
    return booktabs(pd.DataFrame(rows), "cclcc cccc c cc l",
                    ["$q$", r"$\rho$", "sched.", "$b$", "$z$", "NWJ", "NWJ$_5$", "DV", "plug-in", "direct$_5$", "acc.", "ESS", "model"],
                    r"Cost scores at $h=10$ for each target $z$: held-out NWJ with raw logits, with the critic capped at $M=5$ (NWJ$_5$), Donsker--Varadhan, plug-in $\frac1m\sum_i f(\Gamma_{i,z})$, the separately trained direct NWJ critic (cap 5), held-out classification accuracy, the effective sample size of $e^{f}$ over the $m$ silent paths, and the selected BCE model. All in nats.",
                    "tab:cost", size="scriptsize", fit=True)


def tab_efficiency(eff: pd.DataFrame) -> str:
    rows = []
    for r in _sort(eff[eff.horizon.isin([5, 10])]).sort_values(["q", "rho", "schedule", "budget", "horizon"]).itertuples():
        eta = ci(r.eta_raw, r.eta_raw_lo, r.eta_raw_hi) if r.eta_raw_supported else "--"
        rows.append([r.q, f"{r.rho:.2f}", sched_short(r.schedule), r.budget, r.horizon,
                     ci(r.info, r.info_lo, r.info_hi), ci(r.cost_raw, r.cost_raw_lo, r.cost_raw_hi, 1), f3(r.cost_raw_nonpositive_frac, 2),
                     f3(r.cost_cap5, 2), eta, f3(r.eta_cap5, 3) if r.eta_cap5_supported else "--"])
    return booktabs(pd.DataFrame(rows), "cclcc l l c c l c",
                    ["$q$", r"$\rho$", "sched.", "$b$", "$h$", r"$\widehat L_I$", r"$\widehat{\mathcal C}$ (raw)", r"$P(\widehat{\mathcal C}\le0)$", r"$\widehat{\mathcal C}_5$", r"$\widehat\eta_{\rm end}$ (raw)", r"$\widehat\eta_{\rm end}$ (cap 5)"],
                    r"Exploratory endpoint efficiency at $h=5$ and $h=10$. $P(\widehat{\mathcal C}\le0)$ is the fraction of 2000 paired parent resamples with a non-positive cost; a ratio is reported only when this fraction is zero. Numerator and denominator use the same resamples.",
                    "tab:efficiency", size="scriptsize", longtable=True)


def tab_decomposition(pathinfo, mixture, eff) -> str:
    rows = []
    pi = pathinfo[pathinfo.horizon == 10].set_index("comparison")
    mx = mixture[mixture.horizon == 10].set_index("comparison")
    e = eff[eff.horizon == 10].set_index("comparison")
    for comp in _sort(eff[eff.horizon == 10]).comparison:
        r = e.loc[comp]
        rows.append([r.q, f"{r.rho:.2f}", sched_short(r.schedule), r.budget, f3(r["info"], 3),
                     ci(pi.loc[comp, "score"], pi.loc[comp, "score_lo"], pi.loc[comp, "score_hi"]), model_tex(pi.loc[comp, "selected_mode"]),
                     f3(mx.loc[comp, "nwj_raw"], 2), f3(mx.loc[comp, "nwj_cap5"], 2),
                     f3(pi.loc[comp, "score"] + mx.loc[comp, "nwj_cap5"], 2), f3(r.cost_cap5, 2)])
    return booktabs(pd.DataFrame(rows), "cclc c l l cc cc",
                    ["$q$", r"$\rho$", "sched.", "$b$", r"$\widehat L_I$", r"$\widehat L_{\Gamma}$", "model", r"$\widehat D_Q$", r"$\widehat D_{Q,5}$", r"$\widehat L_\Gamma+\widehat D_{Q,5}$", r"$\widehat{\mathcal C}_5$"],
                    r"Decomposition diagnostics at $h=10$: endpoint information $\widehat L_I$, full-path target information $\widehat L_\Gamma$ (lower bound on $I(Z;\Gamma)$), the mixture-versus-silent NWJ score $\widehat D_Q$ (lower bound on $D_{\rm KL}(Q\Vert p_{\rm base})$; raw and cap 5), their sum, and the cost score. In population $\mathcal C = I(Z;\Gamma)+D_{\rm KL}(Q\Vert p_{\rm base})$; independently fitted lower bounds need not add up exactly.",
                    "tab:decomposition", size="scriptsize", fit=True)


def tab_synthetic(syn: pd.DataFrame) -> str:
    rows = []
    order = ["identical", "generic", "opposite", "reversed", "transient", "rare_tail"]
    for sc in order:
        for m_ in (40, 200):
            d = syn[(syn.scenario == sc) & (syn.m == m_)]
            if d.empty:
                continue
            g = d.iloc[0]
            eta_est = d.est_eta.mean() if d.est_eta.notna().any() else np.nan
            rows.append([sc.replace("_", r"\_"), m_, f3(g.exact_info_end_h3, 3), f"{d.est_info_end.mean():.3f} $\\pm$ {d.est_info_end.std(ddof=0):.3f}",
                         f3(d.est_info_end_freq_alpha0.mean(), 3), f3(g.exact_cost, 2), f"{d.est_cost_nwj.mean():.2f} $\\pm$ {d.est_cost_nwj.std(ddof=0):.2f}",
                         f3(g.exact_info_path, 3), f3(d.est_info_path.mean(), 3), f3(g.exact_eta, 3), f3(eta_est, 3), f"{d.gain_toward_target0.mean():+.2f}"])
    return booktabs(pd.DataFrame(rows), "l c cc c cc cc cc c",
                    ["scenario", "$m$", "$I(Z;Y_3)$", r"$\widehat L_I$", r"$\widehat I_{\rm freq}$", r"$\mathcal C$", r"$\widehat{\mathcal C}$", r"$I(Z;\Gamma)$", r"$\widehat L_\Gamma$", r"$\eta_{\rm end}$", r"$\widehat\eta_{\rm end}$", r"gain$\to$0"],
                    r"Synthetic known-law benchmark (8 agents, horizon 3, two replicates per size; mean $\pm$ spread over replicates). Exact values by path enumeration. $\widehat I_{\rm freq}$ is the in-sample raw frequency estimate. The efficiency is shown when the bootstrap supported it in at least one replicate.",
                    "tab:synthetic", size="scriptsize")


def tab_frequency(freq: pd.DataFrame) -> str:
    rows = []
    for h in (1, 5, 10):
        d = freq[freq.horizon == h]
        rows.append([h] + [f3(d[c].mean(), 3) for c in ("insample_mi_alpha0", "insample_mi_alpha1", "insample_mi_alpha10", "insample_mi_alpha100",
                                                         "heldout_score_alpha1", "heldout_score_alpha10", "heldout_score_alpha100", "classifier_score")])
    extra = r"& \multicolumn{4}{c}{in-sample $\widehat I_{\rm freq}$, pseudocount $\alpha$} & \multicolumn{3}{c}{held-out smoothed posterior} & \\ \cmidrule(lr){2-5}\cmidrule(lr){6-8}"
    return booktabs(pd.DataFrame(rows), "c cccc ccc c", ["$h$", "0", "1", "10", "100", "1", "10", "100", "classifier"],
                    r"Frequency / smoothing comparison, averaged over the 16 comparisons (nats). In-sample plug-in MI from (smoothed) endpoint frequencies versus the held-out score of the same smoothed densities used as a Bayes posterior, and the held-out selected-classifier score.",
                    "tab:frequency", size="small", extra_header=extra)


def tab_selection(ep, c0, c2) -> str:
    def counts(df):
        fam = df.selected_mode.str.split("-").str[0].replace({"gru_mean": "GRU-mean"})
        return fam.value_counts()
    e = counts(ep); c = counts(pd.concat([c0, c2]))
    fams = ["constant", "linear", "quadratic", "mlp", "flat", "gru", "GRU-mean"]
    rows = [[f, int(e.get(f, 0)), int(c.get(f, 0))] for f in fams]
    rows.append(["total problems", int(e.sum()), int(c.sum())])
    return booktabs(pd.DataFrame(rows), "l cc", ["family", "endpoint (160 problems)", "cost (96 problems)"],
                    "Model family selected by the one-standard-error rule (mode over the five outer folds), counted over all comparisons and horizons.",
                    "tab:selection", size="small")


def tab_memory(mem: pd.DataFrame, full: pd.DataFrame) -> str:
    rows = []
    for hist in (1, 2, 4):
        d = mem[mem.history == hist]
        rows.append([f"$Y_0$ + last {hist}", f3(d.nwj_cap5.mean(), 2), f3(d.nwj_cap5.median(), 2), f3(d.nwj_raw.median(), 2), model_tex(d.selected_mode.mode().iloc[0])])
    d = full
    rows.append(["full path (selected model)", f3(d.nwj_cap5.mean(), 2), f3(d.nwj_cap5.median(), 2), f3(d.nwj_raw.median(), 2), "--"])
    return booktabs(pd.DataFrame(rows), "l ccc l", ["observation", "mean NWJ$_5$", "median NWJ$_5$", "median NWJ", "typical model"],
                    r"Memory comparison at $h=10$ over the 32 cost problems (16 comparisons $\times$ 2 targets): GRU-4 critic fed $Y_0$ (to the readout) plus the last $\ell$ vote vectors, versus the full-path selected model.",
                    "tab:memory", size="small")


def tab_appendix_endpoint(ep: pd.DataFrame) -> str:
    rows = []
    for r in _sort(ep).sort_values(["q", "rho", "schedule", "budget", "horizon"]).itertuples():
        rows.append([r.q, f"{r.rho:.2f}", sched_short(r.schedule), r.budget, r.horizon, r.m, f3(r.score, 3), f3(r.score_lo, 3), f3(r.score_hi, 3),
                     model_tex(r.selected_mode), f3(r.log_loss, 3), f3(r.brier, 3), f3(r.accuracy, 2)])
    return booktabs(pd.DataFrame(rows), "cclcc c ccc l ccc",
                    ["$q$", r"$\rho$", "sched.", "$b$", "$h$", "$m$", r"$\widehat L_I$", "lo", "hi", "model", "log loss", "Brier", "acc."],
                    "Endpoint information at every horizon (all 160 problems).", "tab:app-endpoint", size="scriptsize", longtable=True)


def tab_appendix_cost(c0, c2) -> str:
    both = pd.concat([c0.assign(z=0), c2.assign(z=2)])
    rows = []
    for r in _sort(both).sort_values(["q", "rho", "schedule", "budget", "z", "horizon"]).itertuples():
        rows.append([r.q, f"{r.rho:.2f}", sched_short(r.schedule), r.budget, r.z, r.horizon, f3(r.nwj_raw, 2), f3(r.score_lo, 2), f3(r.score_hi, 2),
                     f3(r.nwj_cap2, 2), f3(r.nwj_cap5, 2), f3(r.nwj_cap10, 2), f3(r.dv_raw, 2), f3(r.ess_raw, 1), f3(r.epoch_cap_fraction, 2), model_tex(r.selected_mode)])
    return booktabs(pd.DataFrame(rows), "cclcc c ccc ccc c cc l",
                    ["$q$", r"$\rho$", "sched.", "$b$", "$z$", "$h$", "NWJ", "lo", "hi", "NWJ$_2$", "NWJ$_5$", "NWJ$_{10}$", "DV", "ESS", "cap frac.", "model"],
                    "Cost scores at every horizon and cap (all 96 problems). `cap frac.' is the share of neural refits that hit the 1000-epoch limit.",
                    "tab:app-cost", size="scriptsize", longtable=True)


def tab_appendix_swaps(swaps: pd.DataFrame) -> str:
    s = _key_cols(swaps)
    rows = []
    for r in _sort(s).sort_values(["problem", "q", "rho", "schedule", "budget", "horizon"]).itertuples():
        rows.append([r.problem, r.q, f"{r.rho:.2f}", sched_short(r.schedule), r.budget, r.horizon, f3(np.clip(r.observed, -99, 99), 3),
                     f3(np.clip(r.null_mean, -99, 99), 3), f3(np.clip(r.null_q95, -99, 99), 3), f3(np.clip(r.null_max, -99, 99), 3), f3(r.p_value, 3)])
    return booktabs(pd.DataFrame(rows), "l cclcc ccccc", ["problem", "$q$", r"$\rho$", "sched.", "$b$", "$h$", "observed", "null mean", "null $q_{95}$", "null max", "$p$"],
                    "Whole-parent label-swap diagnostics (200 swaps each; cheap constant/linear/quadratic pipeline). Values clipped to $\\pm99$ for display.",
                    "tab:app-swaps", size="scriptsize", longtable=True)


# ----------------------------------------------------------------------------- numbers for the text
def numbers(ep, c0, c2, eff, direction, swaps, syn, freq, mem, direct, audit) -> dict[str, Any]:
    n: dict[str, Any] = {}
    e10 = ep[ep.horizon == 10]
    sens = e10[e10.schedule == "sensing"]; alw = e10[e10.schedule == "always"]
    n["sens_min"], n["sens_max"] = sens.score.min(), sens.score.max()
    n["sens_ci_excl0"] = int((sens.score_lo > 0).sum())
    n["alw_min"], n["alw_max"] = alw.score.min(), alw.score.max()
    n["alw_acc_min"], n["alw_acc_max"] = alw.accuracy.min(), alw.accuracy.max()
    n["sens_acc_min"], n["sens_acc_max"] = sens.accuracy.min(), sens.accuracy.max()
    sw10 = swaps[(swaps.problem == "endpoint") & (swaps.horizon == 10)].merge(e10[["comparison", "schedule"]], on="comparison")
    n["sens_swap_sig"] = int((sw10[sw10.schedule == "sensing"].p_value <= 0.01).sum())
    n["alw_swap_sig"] = int((sw10[sw10.schedule == "always"].p_value < 0.05).sum())
    n["alw_swap_minp"] = sw10[sw10.schedule == "always"].p_value.min()
    best = sens.sort_values("score", ascending=False).iloc[0]
    n["best"] = best
    d10 = direction[direction.horizon == 10]
    n["gain0_neg"] = int((d10.gain_toward_target0 < 0).sum()); n["gain2_pos"] = int((d10.gain_toward_target2 > 0).sum())
    n["gain0_min"], n["gain0_max"] = d10.gain_toward_target0.min(), d10.gain_toward_target0.max()
    n["gain2_min"], n["gain2_max"] = d10.gain_toward_target2.min(), d10.gain_toward_target2.max()
    ex = d10[d10.comparison == "q12_rho0.70_always_b12"].iloc[0]; n["ex"] = ex
    alwd = d10[d10.schedule == "always"]; n["alw_t0_N2_min"], n["alw_t0_N2_max"] = alwd.target0_mean_N2.min(), alwd.target0_mean_N2.max()
    both10 = pd.concat([c0.assign(z=0), c2.assign(z=2)]); both10 = both10[both10.horizon == 10]
    n["ess_le5"] = int((both10.ess_raw <= 5).sum()); n["n_cost10"] = len(both10)
    n["maxshare_ge50"] = int((both10.maxshare_raw >= 0.5).sum())
    n["nwj_neg"] = int((both10.nwj_raw < 0).sum())
    n["cap5_min"], n["cap5_max"] = both10.nwj_cap5.min(), both10.nwj_cap5.max()
    n["acc_alw_min"] = both10[both10.schedule == "always"].accuracy.min()
    n["acc_sens0_min"], n["acc_sens0_max"] = both10[(both10.schedule == "sensing") & (both10.z == 0)].accuracy.min(), both10[(both10.schedule == "sensing") & (both10.z == 0)].accuracy.max()
    n["epoch_cap_mean"] = pd.concat([c0, c2]).epoch_cap_fraction.mean()
    n["epoch_cap_max"] = pd.concat([c0, c2]).epoch_cap_fraction.max()
    n["direct_min"], n["direct_max"] = direct.nwj_cap5.min(), direct.nwj_cap5.max()
    n["direct_ess_min"], n["direct_ess_max"] = direct.ess_raw.min(), direct.ess_raw.max()
    n["n_eta"] = int(eff.eta_raw_supported.sum()); n["n_eff"] = len(eff)
    sup = eff[eff.eta_raw_supported == True]
    n["eta_min"], n["eta_max"] = sup.eta_raw.min(), sup.eta_raw.max()
    n["eta_supported_sens"] = int((sup.schedule == "sensing").sum())
    n["eta_supported_q12"] = int((sup.q == 12).sum())
    f10 = freq[freq.horizon == 10]
    n["freq_insample"], n["freq_heldout1"], n["freq_heldout100"], n["freq_clf"] = f10.insample_mi_alpha0.mean(), f10.heldout_score_alpha1.mean(), f10.heldout_score_alpha100.mean(), f10.classifier_score.mean()
    n["mem"] = {h: mem[mem.history == h].nwj_cap5.median() for h in (1, 2, 4)}
    n["mem_full"] = both10.nwj_cap5.median()
    sy = syn.groupby(["scenario", "m"]).mean(numeric_only=True)
    n["syn"] = sy
    n["syn_ident_freq40"] = syn[(syn.scenario == "identical") & (syn.m == 40)].est_info_end_freq_alpha0.mean()
    n["n_excl"] = len(audit["exclusions"])
    return n


# ----------------------------------------------------------------------------- document
def write_tex(out: Path, n: dict[str, Any], tabs: dict[str, str], cfg: dict[str, Any], audit: dict) -> None:
    b = n["best"]; ex = n["ex"]; sy = n["syn"]
    prov = audit["provenance"]
    tex = r"""\documentclass[10pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb,booktabs,longtable,graphicx,caption,subcaption,threeparttable,array,multirow,xcolor,microtype,adjustbox}
\usepackage[hidelinks]{hyperref}
\captionsetup{font=small,labelfont=bf}
\setlength{\parskip}{3pt}
\newcommand{\KL}{D_{\mathrm{KL}}}
\newcommand{\Ex}{\mathbb{E}}
\newcommand{\Cost}{\mathcal{C}}
\newcommand{\Lhat}{\widehat L}
\graphicspath{{figures/}}
\title{Blackboard control efficiency: offline bound estimators\\ \large Study \texttt{blackboard\_checkpoint\_ensemble\_01}, task 003}
\author{Prepared with \texttt{src/bound\_estimators} (offline; no provider calls)}
\date{\today}
\begin{document}
\maketitle

\begin{abstract}
We implement and run the estimator programme of \emph{Control efficiency: derivation and estimators} (Part II) on the frozen paired-checkpoint archive of the blackboard checkpoint ensemble. The endpoint target information $I(Z;Y_h)$ is estimated by cross-fitted probabilistic classification; the trajectory cost $\Cost=\tfrac12\KL(p_0\Vert p_{\rm base})+\tfrac12\KL(p_2\Vert p_{\rm base})$ by controlled-versus-silent density-ratio critics scored with held-out variational objectives. Both are population lower bounds. Three results stand out. (i) Under the \emph{sensing} schedule the endpoint reveals the assigned target: $\Lhat_I(10)$ between """ + f"{n['sens_min']:.2f}" + r""" and """ + f"{n['sens_max']:.2f}" + r""" nats (maximum $\log 2=0.69$), """ + f"{n['sens_ci_excl0']}" + r""" of 8 cells with intervals excluding zero and all 8 beating a whole-parent label-swap null. (ii) Under the \emph{always} schedule the endpoint carries no target information because both targets drive the population toward the false answer: the truthful-report controller aimed at the correct answer leaves only """ + f"{100*n['alw_t0_N2_min']:.0f}--{100*n['alw_t0_N2_max']:.0f}\\%" + r""" of the agents \emph{off} the false answer, so its trajectories are a large target-independent disturbance. (iii) The cost is large (capped NWJ scores """ + f"{n['cap5_min']:.1f}--{n['cap5_max']:.1f}" + r""" nats at $h=10$) but not stably estimable with 38--40 silent paths: one or two silent paths dominate the variational objective in """ + f"{n['ess_le5']}" + r""" of """ + f"{n['n_cost10']}" + r""" cost problems, and a synthetic benchmark shows the same pipeline under-estimates the exact cost by 20--60\% at this sample size. Exploratory efficiencies are therefore reported only where the cost bootstrap is bounded away from zero (""" + f"{n['n_eta']}" + r""" of """ + f"{n['n_eff']}" + r""" cells) and are small, $\widehat\eta_{\rm end}\le """ + f"{n['eta_max']:.2f}" + r"""$.
\end{abstract}

\tableofcontents

\section{Scope}
This report covers the offline analysis only: no new simulations and no provider requests were made. The two input archives were verified by SHA-256 against the values declared in the specification and were not modified. Everything below is computed from the vote-count paths reconstructed from the archived round records; controller messages, activation flags and any other branch-identifying metadata were never used as features.

\paragraph{Reductions relative to the specification.} They are listed here so that the numbers are read with the right weight. (1)~One grouped fold partition (the specification asks for three repeats). (2)~Cost estimated at horizons $h\in\{1,5,10\}$ only; endpoint information at every $h=1,\dots,10$. (3)~The full-refit parent bootstrap (200--1000 complete refits) was not run; intervals come from resampling the saved held-out per-parent contributions, which omits training variability. (4)~Label swaps rerun the cheap constant/linear/quadratic pipeline, not the neural candidates. (5)~The synthetic benchmark uses 8 agents, horizon 3 and two replicates per sample size. (6)~The micro-update grid was not analysed.

\section{Experiment and data}
\subsection{Design}
A population of $N=24$ LLM agents discusses a fixed MuSR team-allocation task (\texttt{task\_003}, correct answer \texttt{ALLOCATION\_0}) on a shared message board. Each round every agent reads a uniform sample of $q$ board messages, updates its beliefs (facts persist with probability $\rho$), and votes for one of three allocations. A \emph{parent} is one independent two-round preparation run; its state after round~2 is frozen as a checkpoint. From the checkpoint nine ten-round continuations are launched: one silent (no controller) and eight controlled, crossing four branch policies with two posting budgets $b\in\{3,12\}$. The controller can only post canonical \emph{true} facts from the task pool, selected to favour its target $z$: $z=0$ (the correct answer, policies \texttt{*\_truth}) or $z=2$ (a false answer, policies \texttt{*\_false}). The \emph{always} schedule posts every round; the \emph{sensing} schedule samples 12 votes and posts only when the target's share is below $0.5$ (soft threshold, slope~4).

A \emph{setting} is a pair $(q,\rho)\in\{3,12\}\times\{0.70,1.00\}$, one per archived config. A \emph{comparison} additionally fixes the schedule and the budget; there are 16. Inside a comparison every retained parent contributes exactly one silent path and one path per target, all starting from the same checkpoint.

\subsection{Canonical arrays and validation}
Let $N_a(t)$ be the number of agents voting for allocation $a\in\{0,1,2\}$ after continuation round $t$, and
\begin{equation}
Y_t=\big(N_0(t),N_1(t),N_2(t)\big),\qquad N_0(t)+N_1(t)+N_2(t)=24,\qquad \Gamma_h=(Y_0,Y_1,\dots,Y_h).
\end{equation}
$Y_0$ is the before-vector of continuation round~1 (the checkpoint state); $Y_t$ for $t\ge1$ is the after-vector of round~$t$. Counts were rebuilt from the JSON label lists \texttt{population\_state\_before/after} in the archived round-records table (\texttt{checkpoint\_complete\_\allowbreak round\_records.parquet}), not from the endpoint table (which stores a single target's count and lists each path twice). Every path was checked for: horizons $1..10$ present exactly once, integer counts summing to 24, before-vector of round $t$ equal to after-vector of round $t-1$, one checkpoint hash per parent, and agreement between the branch's recorded controller target and its policy name. A failing path is excluded with a reason, never repaired. Within each triplet the three $Y_0$ vectors were verified identical.

\subsection{Sample}
The archive holds """ + f"{audit['n_parents_total']}" + r""" parents and """ + f"{audit['complete_paths']}" + r""" complete ten-round paths. Two parents in the $(q,\rho)=(12,1.00)$ setting lack their target-2 branches, giving """ + f"{n['n_excl']}" + r""" exclusion notes and 38 instead of 40 parents in that setting's four comparisons (Table~\ref{tab:sample}). All results are complete-case: the estimand is the population of parents with all three branches. The 342\,720 micro-updates in the archive do not add independent populations; the parent is the only independent unit throughout.

""" + tabs["sample"] + r"""

\section{Quantities}
Fix the task $x$ (suppressed below), a comparison, and a horizon $h$. The target $Z\in\{0,2\}$ has weights $w_0=w_2=\tfrac12$. Write $p_z(\gamma)$ for the law of $\Gamma_h$ under the controller with target $z$, $p_{\rm base}(\gamma)$ for its law under silence, and
\begin{equation}
q(\gamma)=\sum_z w_z\,p_z(\gamma)
\end{equation}
for the target mixture $Q$ (run the controller with a target drawn from $w$, then hide the target). The trajectory-divergence cost and its exact decomposition are
\begin{align}
\Cost &= \sum_z w_z\,\KL\!\big(p_z\,\Vert\,p_{\rm base}\big)
 = I(Z;\Gamma_h) + \KL\!\big(q\,\Vert\,p_{\rm base}\big), \label{eq:decomp}\\
I(Z;\Gamma_h) &= I(Z;Y_h) + I(Z;\Gamma_h\mid Y_h), \label{eq:chain}
\end{align}
so that $0\le I(Z;Y_h)\le I(Z;\Gamma_h)\le\Cost$ and the endpoint efficiency
\begin{equation}
\eta_{\rm end}(h)=\frac{I(Z;Y_h)}{\Cost}\in[0,1]
\qquad\text{with}\qquad
1-\eta_{\rm end}(h)=\frac{I(Z;\Gamma_h\mid Y_h)+\KL(q\Vert p_{\rm base})}{\Cost}. \label{eq:eta}
\end{equation}
The two loss terms have distinct meanings: target information present in the path but not in its endpoint, and target-\emph{independent} deviation from silence. All logarithms are natural (nats); with two balanced targets $I(Z;\cdot)\le\log2=0.693$. Equations~\eqref{eq:decomp}--\eqref{eq:eta} hold for the population laws; the estimates below are finite-sample lower-bound \emph{scores} and need not obey them.

\section{Estimators}
\subsection{Endpoint information by classification}
By Bayes' rule the true posterior is $r(z\mid y)=w_z p_z^h(y)/q_h(y)$ and $I(Z;Y_h)=\Ex\log\big[r(Z\mid Y_h)/w_Z\big]$. For any classifier $g_\theta(z\mid y)$ the predictive score
\begin{equation}
L_I(g_\theta)=\Ex_{Z,Y_h}\log\frac{g_\theta(Z\mid Y_h)}{w_Z}
= I(Z;Y_h)-\Ex_{Y_h}\KL\big(r(\cdot\mid Y_h)\,\Vert\,g_\theta(\cdot\mid Y_h)\big)\ \le\ I(Z;Y_h) \label{eq:LI}
\end{equation}
is a population lower bound, tight for the true posterior. With $m$ held-out parents, each providing both targets,
\begin{equation}
\Lhat_I(h)=\frac1m\sum_{i=1}^m\sum_{z\in\{0,2\}}\tfrac12\log\frac{g^{(-i)}(z\mid Y_{i,z,h})}{1/2}, \label{eq:LIhat}
\end{equation}
where $g^{(-i)}$ was trained and selected without parent $i$. Probabilities are clipped to $[\epsilon,1-\epsilon]$ with $\epsilon=10^{-6}$ (also $10^{-4},10^{-8}$; the saturation fraction is recorded and was below 4\% everywhere). A finite realisation can be negative and is reported as such. Candidates: constant $\tfrac12$, linear and quadratic logistic regression on $(u,v)=(N_0/24,N_2/24)$, and a one-hidden-layer MLP (Table~\ref{tab:models}). Logistic regression minimises the explicit objective
\begin{equation}
\frac1n\sum_j \mathrm{CE}\big(y_j,\sigma(\beta_0+\beta^\top x_j)\big)+\frac{\lambda}{2}\|\beta\|_2^2,\qquad \lambda\in\{0.01,0.1,1,10\},
\end{equation}
intercept unpenalised, features standardised on the training parents, solved to convergence with L-BFGS.

\paragraph{Frequency and smoothing comparison.} With $c_z(y)$ the count of parents with endpoint $y$ under target $z$ over the 325 possible count vectors,
\begin{equation}
\widehat p_{z,\alpha}(y)=\frac{c_z(y)+\alpha/325}{m+\alpha},\qquad
\widehat I_{\rm freq}(\alpha)=\sum_z w_z\sum_y \widehat p_{z,\alpha}(y)\log\frac{\widehat p_{z,\alpha}(y)}{\sum_{z'}w_{z'}\widehat p_{z',\alpha}(y)},\qquad \alpha\in\{0,1,10,100\}.
\end{equation}
This in-sample plug-in is reported alongside the held-out score obtained by fitting $\widehat p_{z,\alpha}$ on training folds and scoring the resulting Bayes posterior with~\eqref{eq:LIhat}.

\subsection{Trajectory cost by density-ratio critics}
For target $z$, label controlled paths $B=1$ and silent paths $B=0$ with balanced prior $a=\tfrac12$. The Bayes classifier satisfies $p_z(\gamma)/p_{\rm base}(\gamma)=\frac{d(\gamma)}{1-d(\gamma)}\frac{1-a}{a}$, so the logit of a fitted classifier,
\begin{equation}
f_z(\gamma)=\log\frac{\widehat d_z(\gamma)}{1-\widehat d_z(\gamma)},
\end{equation}
is an estimate of the log density ratio. It is scored with the held-out variational objectives
\begin{align}
\Lhat_{\rm NWJ,z}&=\frac1m\sum_{i=1}^m\Big[f_z^{(-i)}(\Gamma_{i,z})-\exp f_z^{(-i)}(\Gamma_{i,\rm base})+1\Big],\label{eq:nwj}\\
\Lhat_{\rm DV,z}&=\frac1m\sum_{i=1}^m f_z^{(-i)}(\Gamma_{i,z})-\log\frac1m\sum_{i=1}^m\exp f_z^{(-i)}(\Gamma_{i,\rm base}),\label{eq:dv}\\
\Lhat_{\rm plug,z}&=\frac1m\sum_{i=1}^m f_z^{(-i)}(\Gamma_{i,z}),\label{eq:plug}
\end{align}
where for any critic $f$ the population versions obey $L_{\rm NWJ}(f)\le L_{\rm DV}(f)\le\KL(p_z\Vert p_{\rm base})$ with equality at the true log ratio; the plug-in~\eqref{eq:plug} equals the KL only if the ratio is exact and is not a bound. The cost score is
\begin{equation}
\widehat\Cost=\tfrac12\Lhat_{\rm NWJ,0}+\tfrac12\Lhat_{\rm NWJ,2},
\end{equation}
computed parent by parent so that the shared silent paths' covariance is retained. Because $\exp f$ on a single silent path can dominate~\eqref{eq:nwj}, every score is also evaluated with the bounded critic
\begin{equation}
f_M=M\tanh(f/M),\qquad M\in\{2,5,10\},
\end{equation}
which remains a valid population lower bound, and the concentration of the silent-path weights $v_i=\exp f(\Gamma_{i,\rm base})$ is reported through
\begin{equation}
\mathrm{ESS}=\frac{(\sum_i v_i)^2}{\sum_i v_i^2},\qquad \max_i\frac{v_i}{\sum_j v_j}.
\end{equation}
ESS is a weight-concentration diagnostic, not an independent-sample count. Path candidates (Table~\ref{tab:models}) receive the normalised vote fractions $(N_0(t)/24,N_2(t)/24,t/10)$ for $t=0..h$ and nothing else; summary-logistic critics see only $Y_0$, $Y_h$ and the mean over rounds $1..h$, so their scores are restricted-critic bounds on the declared path divergence.

Two further outputs use the same machinery. \emph{Direct critics} train the GRU by maximising the NWJ objective itself (with $f=5\tanh(s/5)$) and are selected by inner NWJ; they are reported separately from the BCE-trained models and never mixed with them in model selection. \emph{Memory comparison}: the GRU sees $Y_0$ (fed to the readout) plus only the last $\ell\in\{1,2,4\}$ vote vectors; these are nested coarsenings of the full path, so exact projected KL cannot increase as rounds are removed.

\subsection{Decomposition diagnostics}
A target classifier on the full path scored with~\eqref{eq:LIhat} gives $\Lhat_\Gamma\le I(Z;\Gamma_h)$. A balanced classifier between the mixture $Q$ (both controlled paths of a parent, weight $\tfrac12$ each) and the silent path (weight 1), scored with~\eqref{eq:nwj} using $\tfrac12[f(\Gamma_{i,0})+f(\Gamma_{i,2})]$ on the controlled side, gives $\widehat D_Q\le\KL(q\Vert p_{\rm base})$. By~\eqref{eq:decomp} the population quantities add to $\Cost$; independently fitted lower bounds need not.

\section{Training and validation protocol}
\paragraph{Folds.} Master seed """ + f"{cfg['master_seed']}" + r""". For each comparison the parents are split into 5 outer folds; the outer-training parents into 3 inner folds. All branches, horizons and examples of a parent stay in one fold. Fold assignments were written to disk before fitting.
\paragraph{Selection.} Every candidate is trained on the inner-training parents and scored on the inner-validation parents by weighted log loss (BCE-trained models) or by negative NWJ (direct critics). With mean $\bar\ell_c$ and standard error $s_c$ over the three inner folds, the \emph{one-standard-error rule} picks the simplest candidate with $\bar\ell_c\le\min_{c'}\bar\ell_{c'}+s_{c^\star}$, simplicity ordered constant $<$ linear $<$ quadratic $<$ MLP/flat $<$ GRU, then smaller width, then larger $\lambda$. The constant model is always eligible.
\paragraph{Neural training.} Adam, learning rate $10^{-3}$, full batch, at most 1000 epochs, gradient-norm clipping at 1, $L_2$ penalty $\lambda\|W\|^2/2$ on weights only, $\tanh$ hidden units, standard GRU gates, no dropout or normalisation. Inner-validation objective evaluated every 10 epochs; early stopping after 10 checks without an improvement of $10^{-4}$ nats per observation; the validation-optimal weights are kept. Three initialisation seeds; predicted probabilities (or critic values) are averaged. The outer refit uses the median inner-selected epoch count on all outer-training parents, with preprocessing refitted, and is evaluated once on the outer-test parents. Every candidate is refitted and scored out-of-fold, not only the selected one, so model sensitivity is fully visible.
\paragraph{Uncertainty.} 2000 paired parent resamples of the saved held-out per-parent contributions; percentile intervals; numerator and denominator drawn on the same resamples. A ratio $\widehat\eta_{\rm end}$ is produced only when no resample gives $\widehat\Cost\le0$. It is not forced into $[0,1]$. This interval is conditional on the fitted predictions; it omits training variability and is not a coverage-certified bound (the synthetic benchmark quantifies this).
\paragraph{Null diagnostics.} 200 whole-parent label swaps per comparison at $h\in\{1,10\}$: for the endpoint both controlled paths of a random half of the parents are exchanged; for each cost problem the controlled and silent path are exchanged. The whole fitting/selection pipeline (cheap candidates) is rerun on every swap with identical folds, and $p=(1+\#\{\text{null}\ge\text{observed}\})/(1+200)$.

""" + tabs["models"] + r"""

\section{Synthetic known-law benchmark}
Before interpreting archive numbers, the full pipeline was run on small models with exactly computable answers: 8 agents, three allocations, horizon 3. Each agent redraws its vote each round from a softmax over $\log(\text{current fraction}+0.05)$ plus a law-specific pull; the count vector is then a Markov chain on 45 states, all $45^4$ paths are enumerated, and $\KL$, $I(Z;\Gamma)$, $I(Z;Y_h)$ and $\KL(Q\Vert p_{\rm base})$ are exact. Six scenarios follow the specification: \emph{identical} (controlled = silent), \emph{generic} (both targets share one perturbation: $I=0$, $\Cost>0$), \emph{opposite} (targets pull apart), \emph{reversed} (target 0 pulls toward allocation 2 and vice versa: same information, opposite direction), \emph{transient} (target pull in round 1 only, then a memoryless reset in the last round: $I(Z;\Gamma)>0$ but $I(Z;Y_3)=0$), and \emph{rare\_tail} (strong pull to states the baseline almost never visits). Parents are sampled with the same shared-$Y_0$ triplet structure, at $m=40$ and $m=200$.

""" + tabs["synthetic"] + r"""

\begin{figure}[htbp]\centering
\includegraphics[width=\textwidth]{fig_synthetic.pdf}
\caption{Synthetic benchmark: estimate versus exact value for endpoint information, cost and path information. Points below the diagonal are the expected lower-bound behaviour.}
\label{fig:synthetic}
\end{figure}

\paragraph{Reading.} The endpoint score recovers the truth well and from below: $""" + f"{sy.loc[('opposite', 40), 'est_info_end']:.2f}" + r"""$ and $""" + f"{sy.loc[('opposite', 200), 'est_info_end']:.2f}" + r"""$ against $0.489$ (opposite), $""" + f"{sy.loc[('rare_tail', 40), 'est_info_end']:.2f}" + r"""$ against $0.693$ (rare tail), and $|\Lhat_I|<0.01$ where the truth is zero. The in-sample frequency plug-in is not usable at $m=40$: it reads $""" + f"{n['syn_ident_freq40']:.2f}" + r"""$ nats on the \emph{identical} law, where the truth is exactly zero, because most endpoints occur once. The path-information score also tracks the truth ($""" + f"{sy.loc[('transient', 40), 'est_info_path']:.2f}" + r"""$--$""" + f"{sy.loc[('transient', 200), 'est_info_path']:.2f}" + r"""$ vs $0.481$ in the transient scenario, where the endpoint score is correctly zero), so the estimators separate ``information in the path'' from ``information at the endpoint''. The cost score is the weak link: it under-estimates by 20--60\% at $m=40$ ($""" + f"{sy.loc[('generic', 40), 'est_cost_nwj']:.2f}" + r"""$ vs $2.58$ generic; $""" + f"{sy.loc[('rare_tail', 40), 'est_cost_nwj']:.2f}" + r"""$ vs $6.64$ rare tail) and still by 15--50\% at $m=200$, and the conditional bootstrap interval rarely contains the exact cost. Because the denominator is biased low, the estimated efficiency can exceed the truth: $""" + f"{sy.loc[('opposite', 40), 'est_eta']:.2f}" + r"""$ and $""" + f"{sy.loc[('opposite', 200), 'est_eta']:.2f}" + r"""$ against $0.335$ (opposite), $""" + f"{sy.loc[('rare_tail', 40), 'est_eta']:.2f}" + r"""$ against $0.104$ (rare tail). Finally, the reversed scenario reproduces the opposite scenario's information and cost exactly while the gain toward target 0 changes sign, confirming that efficiency is blind to direction and the direction table is a separate, required check.

\section{Results on the archive}
\subsection{Endpoint target information}
Figure~\ref{fig:info} and Table~\ref{tab:endpoint} give $\Lhat_I(h)$ for every comparison.

\begin{figure}[htbp]\centering
\includegraphics[width=\textwidth]{fig_info_horizon.pdf}
\caption{Endpoint target-information score $\Lhat_I(h)$ (selected model, out-of-fold) with 95\% paired-parent bootstrap bands, by setting. Dashed line: the maximum $\log 2$. Blue: always schedule; orange: sensing schedule; darker shade: budget 12.}
\label{fig:info}
\end{figure}

""" + tabs["endpoint"] + r"""

\paragraph{Sensing schedule.} At $h=10$ the score ranges from $""" + f"{n['sens_min']:.3f}" + r"""$ to $""" + f"{n['sens_max']:.3f}" + r"""$ nats; """ + f"{n['sens_ci_excl0']}" + r""" of the 8 sensing cells have a lower interval limit above zero, and all 8 beat the label-swap null at $p=0.005$. The largest value, $""" + f"{b.score:.3f}" + r"""$ $[""" + f"{b.score_lo:.2f},{b.score_hi:.2f}" + r"""]$ at $q=""" + f"{b.q}" + r"""$, $\rho=""" + f"{b.rho:.2f}" + r"""$, $b=""" + f"{b.budget}" + r"""$, is half of the one-bit maximum, with held-out accuracy $""" + f"{b.accuracy:.2f}" + r"""$. Information is essentially zero at $h=1$ and grows monotonically over the horizon in every sensing cell; the target becomes legible in the endpoint only after several rounds of selective posting. The $q=12$ cells (agents read 12 board messages per round) carry more information than $q=3$ cells at the same budget, and $\rho=1$ (no forgetting) more than $\rho=0.7$. Held-out accuracy is """ + f"{n['sens_acc_min']:.2f}--{n['sens_acc_max']:.2f}" + r""".

\paragraph{Always schedule.} Scores at $h=10$ lie between $""" + f"{n['alw_min']:.3f}" + r"""$ and $""" + f"{n['alw_max']:.3f}" + r"""$, accuracy """ + f"{n['alw_acc_min']:.2f}--{n['alw_acc_max']:.2f}" + r""", and """ + f"{n['alw_swap_sig']}" + r""" of 8 cells reach $p<0.05$ against the swap null (smallest $p=""" + f"{n['alw_swap_minp']:.3f}" + r"""$, not surviving any correction for 16 tests). The endpoint of an always-controlled population does not tell which target the controller had. The reason is in Section~\ref{sec:direction}.

\paragraph{Model selection.} Over the 160 endpoint problems the rule chose linear 63 times, quadratic 40, MLP 30 and constant 27 (Table~\ref{tab:selection}). Where information is present the quadratic boundary on $(u,v)$ was typically enough; the MLP's out-of-fold score never exceeded the best logistic score by more than the inner standard error (Figure~\ref{fig:heat-endpoint}). Complexity did not earn its place here, and the linear-versus-nonlinear question is settled empirically in favour of quadratic-or-simpler.

\paragraph{Frequency plug-in.} Averaged over the 16 comparisons at $h=10$ the raw in-sample frequency estimate is $""" + f"{n['freq_insample']:.2f}" + r"""$ nats, against a held-out smoothed-density score of $""" + f"{n['freq_heldout1']:.2f}" + r"""$ ($\alpha=1$) and $""" + f"{n['freq_heldout100']:.2f}" + r"""$ ($\alpha=100$), and a held-out classifier score of $""" + f"{n['freq_clf']:.2f}" + r"""$ (Table~\ref{tab:frequency}). The plug-in is inflated by unique endpoints, exactly as on the synthetic identical law, and should not be quoted.

""" + tabs["frequency"] + r"""

\begin{figure}[htbp]\centering
\includegraphics[width=\textwidth]{fig_heat_endpoint.pdf}
\caption{Out-of-fold information score of every endpoint candidate at $h=10$ (rows) for every comparison (columns; $q/\rho/$schedule$/b$). The selected model in each column is within one inner standard error of the best.}
\label{fig:heat-endpoint}
\end{figure}

\subsection{Direction: what the controller actually does}\label{sec:direction}
Efficiency measures target \emph{dependence}, not steering success, so Table~\ref{tab:direction} and Figure~\ref{fig:direction} report where the populations end up.

""" + tabs["direction"] + r"""

\begin{figure}[htbp]\centering
\includegraphics[width=\textwidth]{fig_direction.pdf}
\caption{Mean share of agents on the correct answer (top) and on the false target (bottom) at $h=10$, under silence and under each assigned target, for all 16 comparisons.}
\label{fig:direction}
\end{figure}

Silent populations converge toward the correct answer (73--100\% on allocation 0 at $h=10$). Under the \emph{always} schedule both targets push the population toward allocation~2: in the example $q=12$, $\rho=0.70$, $b=12$ the silent branch ends with $""" + f"{100*ex.silent_mean_N0:.0f}" + r"""\%$ on the correct answer, the target-0 branch with $""" + f"{100*ex.target0_mean_N2:.0f}" + r"""\%$ on allocation~2 and the target-2 branch with $""" + f"{100*ex.target2_mean_N2:.0f}" + r"""\%$ on allocation~2. Across all 16 comparisons the paired gain toward the correct answer is negative in """ + f"{n['gain0_neg']}" + r""" (range $""" + f"{n['gain0_min']:+.2f}" + r"""$ to $""" + f"{n['gain0_max']:+.2f}" + r"""$) and the gain toward the false target is positive in all """ + f"{n['gain2_pos']}" + r""" (range $""" + f"{n['gain2_min']:+.2f}" + r"""$ to $""" + f"{n['gain2_max']:+.2f}" + r"""$). This was cross-checked against the archive's own \texttt{checkpoint\_branch\_endpoints} table, which gives identical shares. Whatever the mechanism---selectively posted true facts favouring allocation~0 being read as evidence for allocation~2, or the mere presence of reports destabilising a converging population---the effect is a large shift that does not depend on the target. This is the synthetic \emph{generic} scenario ($I\approx0$, $\Cost\gg0$) and it explains the always-schedule result of the previous section without any appeal to estimator failure. The sensing controller acts less, disturbs less, and leaves the target visible: its target-0 branches stay near the silent endpoint (gain $-0.12$ to $+0.02$) while its target-2 branches move part of the way.

\subsection{Trajectory cost}
Figure~\ref{fig:cost} shows the capped NWJ score by horizon; Table~\ref{tab:cost} gives all variants at $h=10$; Figure~\ref{fig:ess} shows the weight-concentration diagnostic.

\begin{figure}[htbp]\centering
\includegraphics[width=\textwidth]{fig_cost_horizon.pdf}
\caption{Cost scores by horizon for target 0 (top) and target 2 (bottom): NWJ with the critic capped at $M=5$ (solid, markers) and DV with raw logits (dotted, clipped to $[-0.5,12.5]$). Selected BCE model in each cell.}
\label{fig:cost}
\end{figure}

""" + tabs["cost"] + r"""

\begin{figure}[htbp]\centering
\includegraphics[width=\textwidth]{fig_ess.pdf}
\caption{Weight concentration versus score at $h=10$. Left: raw NWJ; right: capped at $M=5$. The vertical line marks $\mathrm{ESS}=5$. Points with $\mathrm{ESS}\le5$ have their objective dominated by a handful of silent paths.}
\label{fig:ess}
\end{figure}

\paragraph{Distinguishability is easy; magnitude is not.} Controlled and silent paths separate with held-out accuracy at least $""" + f"{n['acc_alw_min']:.2f}" + r"""$ in every always cell and in every target-2 cell; target-0 sensing branches, which rarely act because the group already favours allocation~0, separate at only """ + f"{n['acc_sens0_min']:.2f}--{n['acc_sens0_max']:.2f}" + r""". The capped NWJ scores at $h=10$ range from $""" + f"{n['cap5_min']:.2f}" + r"""$ to $""" + f"{n['cap5_max']:.2f}" + r"""$ nats and rank the cells sensibly: always $>$ sensing, budget 12 $>$ budget 3, target 2 $>$ target 0 under sensing, $\rho=1$ (no forgetting, hence a more stable silent baseline) $>$ $\rho=0.7$.

\paragraph{But the scores are not stable.} In """ + f"{n['ess_le5']}" + r""" of the """ + f"{n['n_cost10']}" + r""" cost problems at $h=10$ the effective sample size of $e^{f}$ over the 38--40 silent paths is at most 5, and in """ + f"{n['maxshare_ge50']}" + r""" a single silent path carries at least half of the weight. Weakly regularised logistic critics assign very large logits to one silent path, $e^f$ explodes, and the raw NWJ score collapses ($""" + f"{n['nwj_neg']}" + r"""$ of 32 problems have a negative raw score at $h=10$; the $q=3$, $\rho=1.00$, always, $b=12$ cell reads $-10.5$ and $-5.8$). The capped critic removes the collapse but changes the estimand to a bounded-critic bound; DV is systematically larger than NWJ and more so where the weights concentrate, which is its finite-sample bias, not extra information. The plug-in $\frac1m\sum f(\Gamma_{i,z})$ is close to the capped NWJ where the ESS is healthy and far above it where it is not. Across candidates (Figure~\ref{fig:heat-cost}) the \emph{median} score at $h=10$ is 1.2--2.3 nats for every non-constant family, while the mean is dominated by a few blow-ups; this is the signature of tail extrapolation with too few silent paths, and the synthetic rare-tail scenario shows that the true cost can be twice the recovered score in this regime.

\paragraph{Direct critics and memory.} The separately trained direct NWJ GRU critics give capped scores of """ + f"{n['direct_min']:.2f}--{n['direct_max']:.2f}" + r""" nats with ESS """ + f"{n['direct_ess_min']:.1f}--{n['direct_ess_max']:.1f}" + r""" (Table~\ref{tab:cost}, column direct$_5$); they are smoother than the BCE-selected critics in the worst cells and agree with them where the ESS is above 10. Feeding the GRU only $Y_0$ and the last round already recovers a median capped score of $""" + f"{n['mem'][1]:.2f}" + r"""$ nats against $""" + f"{n['mem'][2]:.2f}" + r"""$ (last 2), $""" + f"{n['mem'][4]:.2f}" + r"""$ (last 4) and $""" + f"{n['mem_full']:.2f}" + r"""$ for the full-path selected model (Table~\ref{tab:memory}, Figure~\ref{fig:memory}): most of the distinguishability is visible in the endpoint, consistent with the small conditional term in Section~\ref{sec:decomp}. The fitted scores do not decrease monotonically with $\ell$ as the exact projected divergences must, which is a reminder that these are lower-bound scores with fitting noise.

\paragraph{Neural fits.} """ + f"{100*n['epoch_cap_mean']:.0f}" + r"""\% of the neural cost refits (up to """ + f"{100*n['epoch_cap_max']:.0f}" + r"""\% in some cells) hit the 1000-epoch limit at learning rate $10^{-3}$, i.e.\ the flat MLP and GRU critics were still improving when stopped (Appendix~\ref{app:curves}). The one-standard-error rule preferred regularised logistic summaries in 72 of 96 cost problems and a GRU in only 2 (Table~\ref{tab:selection}); given the epoch cap this is a statement about the training budget as much as about the architectures.

""" + tabs["memory"] + r"""

\begin{figure}[htbp]\centering
\includegraphics[width=0.62\textwidth]{fig_memory.pdf}
\caption{Memory comparison at $h=10$: capped NWJ score of the GRU-4 critic against the number of most recent rounds it sees (plus $Y_0$), for all 32 cost problems (thin lines; blue target 0, red target 2) and their median (black). The right-most point is the full-path selected model.}
\label{fig:memory}
\end{figure}

\begin{figure}[htbp]\centering
\includegraphics[width=\textwidth]{fig_heat_cost.pdf}
\caption{Out-of-fold NWJ score (raw logits, clipped to $[-1,8]$ for colour; `$<$' marks values below $-10$) of every path candidate at $h=10$ for target 0 (top) and target 2 (bottom).}
\label{fig:heat-cost}
\end{figure}

""" + tabs["selection"] + r"""

\subsection{Exploratory efficiency}
Table~\ref{tab:efficiency} joins numerator and denominator on the same parent resamples. A ratio is shown only when no resample gives $\widehat\Cost\le0$: this holds in """ + f"{n['n_eta']}" + r""" of the """ + f"{n['n_eff']}" + r""" (comparison, horizon) cells, """ + f"{n['eta_supported_q12']}" + r""" of them at $q=12$ and """ + f"{n['eta_supported_sens']}" + r""" under sensing. Where both numerator and denominator are non-trivial the ratio is small: the largest is $\widehat\eta_{\rm end}(10)=""" + f"{n['eta_max']:.3f}" + r"""$ at $q=12$, $\rho=1.00$, sensing, $b=3$, and the other sensing cells give 0.04--0.07 (Figure~\ref{fig:eff}). Always cells give $0.00\pm0.01$: a numerator of zero over a denominator of several nats. Two cautions apply. The numerator is well estimated and the denominator is biased low, so these ratios are more likely too large than too small (the synthetic opposite scenario gave 0.31--0.49 for a true 0.335). And the cells where the ratio is \emph{not} shown are not cells with low efficiency; they are cells where the cost score has bootstrap mass at or below zero and no ratio is meaningful.

""" + tabs["efficiency"] + r"""

% the efficiency table is a longtable, which cannot float; clear it before the figure so
% the float is not squeezed onto an already-full page (overfull vbox, clipped figure).
\clearpage
\begin{figure}[htbp]\centering
\includegraphics[width=\textwidth]{fig_efficiency.pdf}
\caption{Left: endpoint information against capped cost at $h=5$ and $h=10$; dashed lines are constant $\eta$. Right: $\widehat\eta_{\rm end}(10)$ with 95\% paired intervals for the cells where the raw cost bootstrap is bounded away from zero.}
\label{fig:eff}
\end{figure}

\subsection{Decomposition diagnostics}\label{sec:decomp}
""" + tabs["decomposition"] + r"""

The two independently fitted lower bounds add up to roughly the fitted cost in the well-behaved cells (e.g.\ $0.19+1.94=2.13$ vs $1.85$ at $q=12$, $\rho=0.70$, sensing, $b=12$) and the mixture term dominates everywhere: 70--99\% of the recovered cost is target-\emph{independent} deviation from silence, $\KL(Q\Vert p_{\rm base})$. Full-path target information exceeds endpoint information by at most 0.2 nats (largest gap $0.449$ vs $0.239$ at $q=12$, $\rho=0.70$, sensing, $b=3$), so the first loss term in~\eqref{eq:eta}, $I(Z;\Gamma_h\mid Y_h)$, is small: little target information is lost by looking only at the endpoint. Both facts point the same way. In this experiment the price of control is paid almost entirely as generic disturbance, and the endpoint efficiency is low because the controller's footprint is much larger than its message.

\subsection{Null diagnostics}
\begin{figure}[htbp]\centering
\includegraphics[width=\textwidth]{fig_swaps.pdf}
\caption{Observed score against the 95th percentile of 200 whole-parent label swaps (cheap pipeline), at $h=1$ and $h=10$; points above the diagonal exceed the null. Axes clipped.}
\label{fig:swaps}
\end{figure}
Under the swap null the endpoint score is centred slightly below zero (as a held-out log score of an uninformative predictor should be) with a 95th percentile of $0.002$--$0.013$ nats; all sensing cells at $h=10$ exceed it by an order of magnitude, all always cells sit inside it. For the cost problems the null 95th percentile is $0.01$--$0.08$ nats and the observed scores exceed it in every cell where the raw score is positive; the cells with $p\approx1$ are exactly the collapsed raw scores discussed above, which the null cannot rescue. These are exchangeability diagnostics: they say the target and branch labels are not exchangeable within parents, not that the branch generation is free of non-policy differences.

\section{Discussion}
\paragraph{What the archive supports.} A quantitative, null-checked statement that the sensing controller's endpoint carries $0.15$--$0.34$ nats about its target after ten rounds, growing with horizon, board sample size and persistence; a clear negative statement for the always controller; and a direction table showing that the truthful-report controller, at least when it posts every round, steers the population toward the false answer regardless of the target it was given. The latter is not an estimator artefact and is, for the paper's purposes, probably the most consequential single finding in this archive.

\paragraph{What it does not support.} A stable trajectory-cost estimate or, therefore, a certified efficiency. Thirty-eight to forty silent paths per comparison are not enough to characterise the tails of $p_{\rm base}$ where the controlled paths live; the ESS diagnostic makes this visible in two thirds of the cost problems, and the synthetic benchmark shows the resulting bias is downward and of order 2$\times$ in the rare-tail regime. No architecture choice repairs this---the binding constraint is silent parents, not model capacity. If a tighter cost is wanted, the experiment needs more silent continuations per checkpoint (they are also the cheapest branch to run).

\paragraph{How to report.} (a) Endpoint information with intervals and swap $p$-values, cell by cell (Table~\ref{tab:endpoint}). (b) The direction table (Table~\ref{tab:direction}). (c) The cost as a range across critic caps and training objectives together with the ESS, not as a single number (Table~\ref{tab:cost}). (d) Efficiency only for the $q=12$ sensing cells, with the caveat that the ratio is a well-estimated numerator over an under-estimated denominator.

\paragraph{Limitations of this run.} One fold partition; no full-refit bootstrap; neural critics under-trained at the 1000-epoch cap; cost at three horizons; synthetic coverage statements based on two replicates. None of these change the direction of the conclusions, but the interval widths should be read as optimistic.

\appendix
\section{Training curves}\label{app:curves}
\begin{figure}[htbp]\centering
\includegraphics[width=\textwidth]{fig_curves.pdf}
\caption{Training and inner-validation binary cross-entropy for neural candidates (outer fold 0, inner fold 0, seed 0, $h=10$) in two comparisons. The GRU curves are still descending at the epoch cap.}
\end{figure}

\section{Full tables}
""" + tabs["app_endpoint"] + "\n\n" + tabs["app_cost"] + "\n\n" + tabs["app_swaps"] + r"""

\section{Provenance}
\begin{itemize}\footnotesize
\item Specification: \texttt{CONTROL\_EFFICIENCY\_DERIVATION\_AND\_ESTIMATORS.md}, 21 September 2026, Part II.
\item Analysis archive SHA-256: \texttt{""" + prov["archive_sha256"] + r"""}.
\item Frozen inputs archive SHA-256: \texttt{""" + prov.get("frozen_inputs_sha256", "n/a") + r"""}.
\item Code: \texttt{src/bound\_estimators/} (data, folds, models, objectives, crossfit, pipelines, uncertainty, synthetic, tables, report\_tex); tests in \texttt{tests/bound\_estimators/}. Config: \url{""" + prov["config"] + r"""}. Package version """ + prov["package_version"] + r""".
\item Outputs: \url{results/bound_estimators/blackboard_checkpoint_ensemble_01/} (audit, canonical paths, fold manifests, per-job pickles with out-of-fold logits and per-parent contributions, CSV tables) and \url{results/bound_estimators/synthetic/}.
\item References: Kim, Bae, Lee, Jeong (2020) \emph{Learning entropy production via neural networks}, arXiv:2003.04166 (GRU architecture background only); Nguyen, Wainwright, Jordan (2010) \emph{Estimating divergence functionals and the likelihood ratio by convex risk minimization}; Belghazi et al.\ (2018) \emph{MINE}; Poole et al.\ (2019) \emph{On variational bounds of mutual information}.
\end{itemize}
\end{document}
"""
    out.write_text(tex)


# ----------------------------------------------------------------------------- main
def _tex_env() -> dict[str, str]:
    """PATH with a TeX that actually runs on this machine placed first.

    A TeX install built for one CPU cannot run on another (an x86-only TeX Live on an
    Apple Silicon Mac needs Rosetta, and fails with "Bad CPU type in executable" once
    Rosetta is gone).  Being on PATH is therefore not enough: each candidate root is
    probed by running ``pdflatex --version`` and the first one that succeeds wins.
    """
    roots = [d for d in ("/opt/homebrew/bin", "/usr/local/bin", "/Library/TeX/texbin")
             if (Path(d) / "pdflatex").exists()]
    tried = []
    for root in roots:
        try:
            probe = subprocess.run([str(Path(root) / "pdflatex"), "--version"],
                                   capture_output=True, text=True, timeout=30)
        except OSError as err:
            tried.append(f"{root}: {err}")
            continue
        if probe.returncode == 0:
            return {**os.environ, "PATH": root + ":" + os.environ.get("PATH", "")}
        tried.append(f"{root}: exit {probe.returncode} {probe.stderr.strip()[:80]}")
    if shutil.which("pdflatex") and not tried:
        return {**os.environ}
    raise RuntimeError("no working pdflatex found; tried:\n  " + "\n  ".join(tried or ["(none on PATH)"]))


def build(cfg: dict[str, Any], out_dir: Path, synthetic_dir: Path) -> Path:
    tdir = out_dir / "tables"
    rdir = out_dir / "report"; fdir = rdir / "figures"; fdir.mkdir(parents=True, exist_ok=True)
    est = pd.read_csv(tdir / "estimates.csv")
    eff = pd.read_csv(tdir / "efficiency.csv")
    freq = pd.read_csv(tdir / "frequency.csv")
    swaps = pd.read_csv(tdir / "label_swaps.csv")
    mem = pd.read_csv(tdir / "memory.csv")
    direction = pd.read_csv(tdir / "direction.csv")
    audit = json.loads((out_dir / "audit.json").read_text())
    curves = json.loads((tdir / "curves.json").read_text())
    syn = pd.read_csv(synthetic_dir / "synthetic_results.csv")
    ep = est[(est.problem == "endpoint") & (est.set == "main")]
    c0 = est[(est.problem == "cost0") & (est.set == "main")]
    c2 = est[(est.problem == "cost2") & (est.set == "main")]
    direct = est[est.set == "direct"]
    pathinfo = est[est.problem == "pathinfo"]; mixture = est[est.problem == "mixture"]
    full10 = pd.concat([c0, c2]); full10 = full10[full10.horizon == 10]

    fig_info_horizon(ep, fdir / "fig_info_horizon.pdf")
    fig_direction(direction, fdir / "fig_direction.pdf")
    fig_cost_horizon(c0, c2, fdir / "fig_cost_horizon.pdf")
    fig_heatmap(ep[ep.horizon == 10], fdir / "fig_heat_endpoint.pdf", -0.05, 0.4, "endpoint candidates, $h=10$", 3.4, r"$\widehat L_I$ [nats]")
    # cost heatmap: two stacked panels
    fig, axes = plt.subplots(2, 1, figsize=(TEXTWIDTH, 7.2))
    for ax, (z, df) in zip(axes, ((0, c0), (2, c2))):
        d = _sort(df[df.horizon == 10]); cand = [c for c in d.columns if c.startswith("cand:") and d[c].notna().any()]
        M = d[cand].to_numpy().T
        im = ax.imshow(np.clip(M, -1, 8), aspect="auto", cmap="viridis", vmin=-1, vmax=8)
        ax.set_yticks(range(len(cand))); ax.set_yticklabels([c[5:] for c in cand], fontsize=6)
        ax.set_xticks(range(len(d))); ax.set_xticklabels([f"{r.q}/{r.rho:.1f}/{sched_short(r.schedule)}/{r.budget}" for r in d.itertuples()], rotation=60, ha="right", fontsize=6)
        ax.grid(False); ax.set_title(f"path candidates, target {z}, $h=10$")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                ax.text(j, i, f"{v:.1f}" if v > -10 else "<", ha="center", va="center", fontsize=4.5, color="white" if (np.clip(v, -1, 8) + 1) / 9 < 0.55 else "black")
        cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01); cb.set_label("NWJ [nats]", fontsize=7); cb.ax.tick_params(labelsize=6)
    fig.tight_layout(); fig.savefig(fdir / "fig_heat_cost.pdf"); plt.close(fig)
    fig_ess(c0, c2, fdir / "fig_ess.pdf")
    fig_efficiency(eff, fdir / "fig_efficiency.pdf")
    fig_synthetic(syn, fdir / "fig_synthetic.pdf")
    fig_swaps(swaps, fdir / "fig_swaps.pdf")
    fig_memory(mem, full10, fdir / "fig_memory.pdf")
    fig_curves(curves, fdir / "fig_curves.pdf")

    tabs = {
        "sample": tab_sample(audit), "models": tab_models(), "endpoint": tab_endpoint(ep, swaps),
        "direction": tab_direction(direction), "cost": tab_cost(c0, c2, direct), "efficiency": tab_efficiency(eff),
        "decomposition": tab_decomposition(pathinfo, mixture, eff), "synthetic": tab_synthetic(syn),
        "frequency": tab_frequency(freq), "selection": tab_selection(ep, c0, c2), "memory": tab_memory(mem, full10),
        "app_endpoint": tab_appendix_endpoint(ep), "app_cost": tab_appendix_cost(c0, c2), "app_swaps": tab_appendix_swaps(swaps),
    }
    (rdir / "tables").mkdir(exist_ok=True)
    for k, v in tabs.items():
        (rdir / "tables" / f"{k}.tex").write_text(v)
    n = numbers(ep, c0, c2, eff, direction, swaps, syn, freq, mem, direct, audit)
    write_tex(rdir / "main.tex", n, tabs, cfg, audit)
    env = _tex_env()
    res = subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"], cwd=rdir, env=env,
                         capture_output=True, text=True)
    if res.returncode != 0:
        log = (rdir / "main.log").read_text(errors="ignore") if (rdir / "main.log").exists() else res.stdout
        raise RuntimeError("LaTeX failed:\n" + log[-4000:])
    final = out_dir / "control_efficiency_bound_estimators_report.pdf"
    shutil.copy(rdir / "main.pdf", final)
    print("report:", final)
    return final


def main(argv=None):
    from .run import load_config
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    build(cfg, Path(cfg["output_dir"]), Path(cfg.get("synthetic_dir", "results/bound_estimators/synthetic")))


if __name__ == "__main__":
    main()
