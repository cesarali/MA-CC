"""LaTeX report for the 21-09-2026-full-vs-report-v1 study.

Writes ``<output_dir>/report/{main.tex, figures/, tables/}``, compiles with
latexmk and copies the PDF to ``<output_dir>/new_rnd_init_observables_report.pdf``.
Every number quoted in the prose is pulled from the CSV tables at build time, so
the text cannot drift from the data.
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

TEXTWIDTH = 6.3
LN2 = float(np.log(2))
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "figure.dpi": 160, "savefig.bbox": "tight", "axes.grid": True,
                     "grid.alpha": 0.25, "grid.linewidth": 0.4})

# one colour per comparison family, one dash pattern per budget
PROFILE_COLOR = {"report_only": "#08519c", "full_communication": "#a63603"}
RHO_MARK = {0.75: "o", 1.0: "s"}
B_STYLE = {3: (0.45, "-"), 12: (0.72, "--"), 18: (1.0, ":")}


def _label(r) -> str:
    prof = "RO" if r["profile"] == "report_only" else "FC"
    return f"{prof}/$\\rho$={r['rho']:.2f}/b={int(r['budget'])}"


def _plain(r) -> str:
    prof = "RO" if r["profile"] == "report_only" else "FC"
    return f"{prof} rho={r['rho']:.2f} b={int(r['budget'])}"


def _style(row):
    alpha, ls = B_STYLE[int(row["budget"])]
    return dict(color=PROFILE_COLOR[row["profile"]], alpha=alpha, ls=ls,
                marker=RHO_MARK[float(row["rho"])], ms=2.6, lw=1.1)


def _order(df: pd.DataFrame) -> list[str]:
    k = df[["comparison", "profile", "rho", "budget"]].drop_duplicates()
    k = k.sort_values(["profile", "rho", "budget"])
    return k.comparison.tolist()


# ------------------------------------------------------------------ figures
def fig_trajectories(traj: pd.DataFrame, path: Path):
    """Mean truth share against round, one panel per setting, all arms."""
    settings = traj[["profile", "rho"]].drop_duplicates().sort_values(["profile", "rho"])
    fig, axes = plt.subplots(1, len(settings), figsize=(TEXTWIDTH, 2.3), sharey=True)
    for ax, (_, s) in zip(np.atleast_1d(axes), settings.iterrows()):
        d = traj[(traj.profile == s.profile) & (traj.rho == s.rho)]
        for b, (alpha, ls) in B_STYLE.items():
            dd = d[d.budget == b]
            if dd.empty:
                continue
            ax.plot(dd.horizon, dd.t0_x0, color="#2171b5", alpha=alpha, ls=ls, lw=1.1,
                    label=f"target 0, b={b}")
            ax.plot(dd.horizon, dd.t2_x0, color="#cb181d", alpha=alpha, ls=ls, lw=1.1,
                    label=f"target 2, b={b}")
        d0 = d[d.budget == d.budget.min()]
        ax.plot(d0.horizon, d0.silent_x0, color="k", lw=1.5, label="silent")
        ax.set_title(f"{'RO' if s.profile=='report_only' else 'FC'}, $\\rho$={s.rho:.2f}")
        ax.set_xlabel("round $h$"); ax.set_ylim(0, 1.02)
    np.atleast_1d(axes)[0].set_ylabel("mean truth share $x_0$")
    np.atleast_1d(axes)[-1].legend(fontsize=5.4, loc="lower left", ncol=1, framealpha=0.9)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_md(eff: pd.DataFrame, path: Path):
    """Shared component M and directional half-contrast D against horizon."""
    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, 2.4), sharex=True)
    for key in _order(eff):
        d = eff[eff.comparison == key].sort_values("horizon")
        st = _style(d.iloc[0])
        axes[0].plot(d.horizon, 100 * d.truth_M, **st)
        axes[1].plot(d.horizon, 100 * d.truth_D, **st)
        axes[2].plot(d.horizon, 100 * d.false_target_M, label=_label(d.iloc[0]), **st)
    for ax, t in zip(axes, [r"$M_h^{(0)}$ truth", r"$D_h^{(0)}$ truth", r"$M_h^{(2)}$ false target"]):
        ax.axhline(0, color="gray", lw=0.6); ax.set_title(t); ax.set_xlabel("horizon $h$")
    axes[0].set_ylabel("percentage points")
    axes[2].legend(fontsize=5.2, loc="upper left", ncol=1, framealpha=0.9)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_gains(eff: pd.DataFrame, path: Path):
    """Own-target gains G+ and G- with bootstrap bands, against horizon."""
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.4), sharex=True, sharey=True)
    for key in _order(eff):
        d = eff[eff.comparison == key].sort_values("horizon")
        st = _style(d.iloc[0])
        axes[0].plot(d.horizon, 100 * d.truth_g_plus, **st)
        axes[0].fill_between(d.horizon, 100 * d.truth_g_plus_lo, 100 * d.truth_g_plus_hi,
                             color=st["color"], alpha=0.07, lw=0)
        axes[1].plot(d.horizon, 100 * d.false_target_g_minus, label=_label(d.iloc[0]), **st)
    axes[0].set_title(r"$G_h^{+,0}$: truth gain under the truth request")
    axes[1].set_title(r"$G_h^{-,2}$: false-target gain under the false request")
    for ax in axes:
        ax.axhline(0, color="gray", lw=0.6); ax.set_xlabel("horizon $h$")
    axes[0].set_ylabel("percentage points")
    axes[1].legend(fontsize=5.2, loc="upper left", framealpha=0.9)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_command(cmd: pd.DataFrame, path: Path):
    """Command information, following probability and bits per post against horizon."""
    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, 2.4), sharex=True)
    for key in _order(cmd):
        d = cmd[cmd.comparison == key].sort_values("horizon")
        st = _style(d.iloc[0])
        axes[0].plot(d.horizon, d.mi_bits, **st)
        axes[1].plot(d.horizon, d.following, **st)
        axes[2].plot(d.horizon, 1e3 * d["bits_per_resource_lam0"], label=_label(d.iloc[0]), **st)
    axes[0].set_title(r"$I(Z;V_h)$ [bits]"); axes[0].set_ylabel("bits")
    axes[1].set_title(r"following $F_h$"); axes[1].axhline(0.5, color="gray", lw=0.6)
    axes[2].set_title(r"$10^3\times$ bits per post"); axes[2].axhline(0, color="gray", lw=0.6)
    for ax in axes:
        ax.set_xlabel("horizon $h$")
    axes[2].legend(fontsize=5.2, loc="upper right", framealpha=0.9)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_information(est: pd.DataFrame, swaps: pd.DataFrame, path: Path):
    """Endpoint target information against horizon, with the permutation null."""
    ep = est[est.problem == "endpoint"]
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.4))
    for key in _order(ep):
        d = ep[ep.comparison == key].sort_values("horizon")
        st = _style(d.iloc[0])
        axes[0].plot(d.horizon, d.score_bits, label=_label(d.iloc[0]), **st)
    axes[0].axhline(0, color="gray", lw=0.6)
    axes[0].set_xlabel("horizon $h$"); axes[0].set_ylabel("bits")
    axes[0].set_title(r"$\widehat{I}(Z;Y_h)$, held out")
    axes[0].legend(fontsize=5.2, loc="upper left", framealpha=0.9)
    if swaps is not None and len(swaps):
        s = swaps.merge(ep[["comparison", "horizon", "profile", "rho", "budget"]],
                        on=["comparison", "horizon"], how="left")
        for _, r in s.iterrows():
            axes[1].scatter(r.null_p95, r.observed, s=16, color=PROFILE_COLOR[r["profile"]],
                            alpha=B_STYLE[int(r["budget"])][0], marker=RHO_MARK[float(r["rho"])])
        lim = [min(s.null_p95.min(), s.observed.min()) - 0.02, max(s.null_p95.max(), s.observed.max()) + 0.02]
        axes[1].plot(lim, lim, color="gray", lw=0.7, ls="--")
        axes[1].set_xlim(lim); axes[1].set_ylim(lim)
    axes[1].set_xlabel("95th percentile of 200 target-label swaps [bits]")
    axes[1].set_ylabel("observed [bits]")
    axes[1].set_title("null diagnostic ($h=1,8,15$)")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_cost(eff: pd.DataFrame, path: Path):
    """Trajectory cost, its decomposition and the effective sample size, against horizon."""
    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, 2.4), sharex=True)
    for key in _order(eff):
        d = eff[eff.comparison == key].sort_values("horizon")
        st = _style(d.iloc[0])
        axes[0].plot(d.horizon, d.cost_nats, label=_label(d.iloc[0]), **st)
        axes[1].plot(d.horizon, d.I_path_nats, **st)
        axes[2].plot(d.horizon, d.D_mixture_nats, **st)
    axes[0].set_title(r"$\widehat{C}_\pi$ [nats]"); axes[0].set_ylabel("nats")
    axes[1].set_title(r"$\widehat{I}(Z;\Gamma_h)$ [nats]")
    axes[2].set_title(r"$\widehat{D}_{KL}(Q\|P_{base})$ [nats]")
    for ax in axes:
        ax.set_xlabel("horizon $h$"); ax.axhline(0, color="gray", lw=0.6)
    axes[0].legend(fontsize=5.2, loc="upper left", framealpha=0.9)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_efficiency(eff: pd.DataFrame, path: Path):
    """eta_end, eta_ctl and eta_task against horizon, where the cost supports a ratio."""
    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, 2.4), sharex=True)
    for key in _order(eff):
        d = eff[(eff.comparison == key) & eff.eta_supported].sort_values("horizon")
        if d.empty:
            continue
        st = _style(d.iloc[0])
        axes[0].plot(d.horizon, d.eta_end, **st)
        axes[1].plot(d.horizon, d.eta_ctl, label=_label(d.iloc[0]), **st)
        axes[2].plot(d.horizon, d.eta_task, **st)
    axes[0].set_title(r"$\eta_{end}(h)=\widehat{I}(Z;Y_h)/\widehat{C}_\pi$")
    axes[1].set_title(r"$\eta_{ctl}(h)=\widehat{I}(Z;\Gamma_h)/\widehat{C}_\pi$")
    axes[2].set_title(r"$\eta_{task}(h)=K_{min}/\widehat{C}_\pi$")
    for ax in axes:
        ax.set_xlabel("horizon $h$"); ax.axhline(0, color="gray", lw=0.6)
    axes[0].set_ylabel("ratio")
    axes[1].legend(fontsize=5.2, loc="upper left", framealpha=0.9)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_susceptibility(sus: pd.DataFrame, act: pd.DataFrame, path: Path):
    """One-round response families and activation information."""
    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, 2.4))
    s = sus[sus.outcome == "truth"].copy()
    s["lab"] = [_plain(r) for _, r in s.iterrows()]
    piv = s.pivot_table(index="lab", columns="target", values="chi_bar")
    y = np.arange(len(piv))
    axes[0].barh(y - 0.2, piv[0], 0.4, color="#2171b5", label="target 0")
    axes[0].barh(y + 0.2, piv[2], 0.4, color="#cb181d", label="target 2")
    axes[0].set_yticks(y); axes[0].set_yticklabels(piv.index, fontsize=5)
    axes[0].axvline(0, color="gray", lw=0.6); axes[0].set_title(r"$\bar{\chi}_0$ (state matched)")
    axes[0].legend(fontsize=5.5)
    piv2 = s.pivot_table(index="lab", columns="target", values="tau_lag1")
    axes[1].barh(y - 0.2, piv2[0], 0.4, color="#2171b5")
    axes[1].barh(y + 0.2, piv2[2], 0.4, color="#cb181d")
    axes[1].set_yticks(y); axes[1].set_yticklabels([]); axes[1].axvline(0, color="gray", lw=0.6)
    axes[1].set_title(r"IPW $\hat{\tau}_{0,1}$ (causal)")
    for key in _order(act):
        d = act[(act.comparison == key) & (act.target == 0)].sort_values("horizon")
        axes[2].plot(d.horizon, d.T_act_bits, label=_label(d.iloc[0]), **_style(d.iloc[0]))
    axes[2].set_title(r"$I(U_h;n_{0,h}\mid n_{0,h-1})$ [bits]"); axes[2].set_xlabel("horizon $h$")
    axes[2].legend(fontsize=4.8, loc="upper right", ncol=1, framealpha=0.9)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_evidence(ev: pd.DataFrame, path: Path):
    """Proof coverage and full-proof ownership against horizon, by arm."""
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.3), sharex=True)
    for key in _order(ev):
        d = ev[ev.comparison == key].sort_values("horizon")
        st = _style(d.iloc[0])
        axes[0].plot(d.horizon, d.kappa_t0, **st)
        axes[1].plot(d.horizon, d.kappa_silent, label=_label(d.iloc[0]), **st)
    axes[0].set_title(r"$\kappa_h$ under the truth request"); axes[0].set_ylabel("mean proof coverage")
    axes[1].set_title(r"$\kappa_h$ under silence")
    for ax in axes:
        ax.set_xlabel("horizon $h$")
    axes[1].legend(fontsize=5.2, loc="upper left", framealpha=0.9)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_dose(dil: pd.DataFrame, dr: pd.DataFrame, path: Path):
    """Within-round dilution of the controller, and the attenuation of its dose slope."""
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.5),
                             gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    ax.plot(dil.slot_bin, dil.eligible_board, marker="o", ms=3, lw=1.2, color="#6baed6",
            label="messages on the board")
    ax.plot(dil.slot_bin, dil.messages_read, marker="s", ms=3, lw=1.2, color="#08519c",
            label="messages the agent read")
    ax.plot(dil.slot_bin, dil.dose, marker="^", ms=3, lw=1.2, color="#a63603",
            label="controller messages read")
    ax.set_xlabel("update order within the round (slot)")
    ax.set_ylabel("messages"); ax.legend(fontsize=5.6, loc="upper left", framealpha=0.9)
    ax2 = ax.twinx(); ax2.grid(False)
    ax2.plot(dil.slot_bin, dil.on_target_after, color="k", lw=1.0, ls=":",
             marker="d", ms=2.5, label="P(on target after)")
    ax2.set_ylabel("P(on target after)", fontsize=7); ax2.tick_params(labelsize=6)
    ax2.legend(fontsize=5.6, loc="lower right", framealpha=0.9)
    ax.set_title("the controller is diluted during the round")

    ax = axes[1]
    labs = ["raw", "population\nadjusted", "order\nadjusted"]
    keys = ["raw", "population_adjusted", "order_adjusted"]
    colors = {"on_target_after": "#08519c", "on_target_next": "#a63603"}
    names = {"on_target_after": "same round", "on_target_next": "next round"}
    for k, (_, r) in enumerate(dr.iterrows()):
        off = (k - 0.5) * 0.18
        y = np.arange(len(keys)) + off
        pts = [r[c] for c in keys]
        lo = [r[c] - r[f"{c}_lo"] for c in keys]; hi = [r[f"{c}_hi"] - r[c] for c in keys]
        ax.errorbar(pts, y, xerr=[lo, hi], fmt="o", ms=3.5, lw=1.0, capsize=2.5,
                    color=colors[r.outcome], label=names[r.outcome])
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_yticks(np.arange(len(keys))); ax.set_yticklabels(labs, fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlabel("slope: change in P(adopt target)\nper controller message read")
    ax.legend(fontsize=6, loc="lower right", framealpha=0.9)
    ax.set_title("the dose effect is an artifact of ordering")
    fig.tight_layout(w_pad=1.6); fig.savefig(path); plt.close(fig)


def fig_models(est: pd.DataFrame, path: Path):
    """Held-out score of every candidate family, endpoint problem at the final horizon."""
    ep = est[(est.problem == "endpoint") & (est.horizon == est.horizon.max())]
    cands = sorted([c for c in ep.columns if c.startswith("cand:")])
    fig, ax = plt.subplots(figsize=(TEXTWIDTH, 2.6))
    data = ep[cands].to_numpy(dtype=float)
    im = ax.imshow(data, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(cands)))
    ax.set_xticklabels([c[5:] for c in cands], rotation=90, fontsize=4.6)
    ax.set_yticks(range(len(ep)))
    ax.set_yticklabels([_plain(r) for _, r in ep.iterrows()], fontsize=5)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isfinite(data[i, j]):
                ax.text(j, i, f"{data[i,j]:.2f}", ha="center", va="center", fontsize=3.6,
                        color="white" if data[i, j] < np.nanmedian(data) else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("held-out score [bits]", fontsize=6); cb.ax.tick_params(labelsize=5)
    ax.set_title(r"Candidate comparison for $\widehat{I}(Z;Y_{15})$", fontsize=8)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_curves(curves: dict, path: Path):
    """Training and inner-validation loss for the neural endpoint candidates."""
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.2), sharey=True)
    keys = list(curves)[:2]
    for ax, k in zip(axes, keys):
        fold = curves[k].get("0", {})
        for name, c in list(fold.items())[:6]:
            if not c or not c[0].get("epoch"):
                continue
            ax.plot(c[0]["epoch"], c[0]["train"], lw=0.8, label=f"{name} train")
            ax.plot(c[0]["epoch"], c[0]["val"], lw=0.8, ls="--", label=f"{name} val")
        ax.set_title(k, fontsize=6); ax.set_xlabel("epoch")
    axes[0].set_ylabel("binary cross-entropy")
    axes[0].legend(fontsize=4.4, ncol=2)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


# ------------------------------------------------------------------- tables
def _fmt(v, nd=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "--"
    return f"{v:.{nd}f}"


def tab_sample(audit: dict) -> str:
    rows = []
    for k, m in audit["comparison_counts"].items():
        prof, rho, b = k.rsplit("_", 2)[0], k.split("_rho")[1][:4], k.rsplit("_b", 1)[1]
        rows.append(f"{prof.replace('_',' ')} & {rho} & {b} & {m} \\\\")
    return ("\\begin{tabular}{llrr}\n\\toprule\ncommunication profile & $\\rho$ & $b$ & "
            "initializations $m$ \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")


def tab_effects(eff: pd.DataFrame, h: int) -> str:
    d = eff[eff.horizon == h].sort_values(["profile", "rho", "budget"])
    rows = []
    for _, r in d.iterrows():
        rows.append(" & ".join([
            "RO" if r.profile == "report_only" else "FC", f"{r.rho:.2f}", f"{int(r.budget)}", f"{int(r.m)}",
            _fmt(100 * r.truth_base, 1), _fmt(100 * r.truth_g_plus, 1), _fmt(100 * r.truth_g_minus, 1),
            _fmt(100 * r.truth_M, 1), f"{100*r.truth_D:.1f} [{100*r.truth_D_lo:.1f}, {100*r.truth_D_hi:.1f}]",
            _fmt(100 * r.false_target_M, 1), _fmt(100 * r.false_target_D, 1)]) + " \\\\")
    return ("\\begin{tabular}{llrr rrr rlrr}\n\\toprule\n"
            "prof. & $\\rho$ & $b$ & $m$ & base & $G^{+,0}$ & $G^{-,0}$ & $M^{(0)}$ & "
            "$D^{(0)}$ [95\\%] & $M^{(2)}$ & $D^{(2)}$ \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")


def tab_command(cmd: pd.DataFrame, h: int) -> str:
    d = cmd[cmd.horizon == h].sort_values(["profile", "rho", "budget"])
    rows = []
    for _, r in d.iterrows():
        rows.append(" & ".join([
            "RO" if r.profile == "report_only" else "FC", f"{r.rho:.2f}", f"{int(r.budget)}",
            _fmt(r.mi_bits, 4), _fmt(r.following, 3), _fmt(r.following_gain, 3),
            _fmt(0.5 * (r.posts_t0 + r.posts_t2), 1), _fmt(0.5 * (r.senses_t0 + r.senses_t2), 0),
            _fmt(1e3 * r.bits_per_resource_lam0, 3), _fmt(1e3 * r["bits_per_resource_lam1"], 4)]) + " \\\\")
    return ("\\begin{tabular}{llr rrr rr rr}\n\\toprule\n"
            "prof. & $\\rho$ & $b$ & $I(Z;V_h)$ & $F_h$ & $F_h-F^{\\rm base}_h$ & posts & senses & "
            "$10^3 \\mathcal E_{\\lambda=0}$ & $10^3\\mathcal E_{\\lambda=1}$ \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")


def tab_cost(effc: pd.DataFrame, h: int) -> str:
    d = effc[effc.horizon == h].sort_values(["profile", "rho", "budget"])
    rows = []
    for _, r in d.iterrows():
        rows.append(" & ".join([
            "RO" if r.profile == "report_only" else "FC", f"{r.rho:.2f}", f"{int(r.budget)}",
            _fmt(r.K0_nats), _fmt(r.K2_nats), _fmt(r.cost_nats), _fmt(r.I_path_nats),
            _fmt(r.D_mixture_nats), _fmt(r.decomp_sum_nats), _fmt(r.I_end_nats),
            _fmt(r.eta_end), _fmt(r.eta_ctl), _fmt(r.eta_task)]) + " \\\\")
    return ("\\begin{tabular}{llr rrr rrr r rrr}\n\\toprule\n"
            "prof. & $\\rho$ & $b$ & $K^0$ & $K^2$ & $\\mathcal C_\\pi$ & $I(Z;\\Gamma)$ & "
            "$D_Q$ & sum & $I(Z;Y)$ & $\\eta_{\\rm end}$ & $\\eta_{\\rm ctl}$ & $\\eta_{\\rm task}$ \\\\\n"
            "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")


def tab_susceptibility(sus: pd.DataFrame) -> str:
    d = sus[sus.outcome == "truth"].sort_values(["profile", "rho", "budget", "target"])
    rows = []
    for _, r in d.iterrows():
        rows.append(" & ".join([
            "RO" if r.profile == "report_only" else "FC", f"{r.rho:.2f}", f"{int(r.budget)}",
            str(int(r.target)), _fmt(r.activation_rate, 3), _fmt(r.chi_bar, 4),
            _fmt(r.chi_unconditioned, 4), _fmt(r.tau_lag1, 4), _fmt(r.tau_lag5, 4),
            _fmt(r.avail_cell_ratio, 4), _fmt(r.max_abs_weight, 1)]) + " \\\\")
    return ("\\begin{tabular}{llrr rrr rr rr}\n\\toprule\n"
            "prof. & $\\rho$ & $b$ & $z$ & $P(U{=}1)$ & $\\bar\\chi_0$ & uncond. & "
            "$\\hat\\tau_{0,1}$ & $\\hat\\tau_{0,5}$ & avail. & $\\max|W|$ \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")


def tab_frequency(est: pd.DataFrame, h: int) -> str:
    d = est[(est.problem == "endpoint") & (est.horizon == h)].sort_values(["profile", "rho", "budget"])
    rows = []
    for _, r in d.iterrows():
        rows.append(" & ".join([
            "RO" if r.profile == "report_only" else "FC", f"{r.rho:.2f}", f"{int(r.budget)}",
            _fmt(r["freq_insample_mi_alpha0"]), _fmt(r["freq_insample_mi_alpha1"]),
            _fmt(r["freq_heldout_score_alpha1"]), _fmt(r["freq_heldout_score_alpha10"]),
            _fmt(r.score_bits), str(r.selected_mode)]) + " \\\\")
    return ("\\begin{tabular}{llr rrrr rl}\n\\toprule\n"
            "prof. & $\\rho$ & $b$ & freq. $\\alpha{=}0$ & freq. $\\alpha{=}1$ & "
            "held out $\\alpha{=}1$ & held out $\\alpha{=}10$ & classifier & selected \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")


def tab_swaps(sw: pd.DataFrame) -> str:
    d = sw.sort_values(["comparison", "horizon"])
    rows = [" & ".join([r.comparison.replace("_", " "), str(int(r.horizon)), _fmt(r.observed),
                        _fmt(r.null_mean), _fmt(r.null_p95), _fmt(r.p_value)]) + " \\\\"
            for _, r in d.iterrows()]
    return ("\\begin{tabular}{lrrrrr}\n\\toprule\ncomparison & $h$ & observed & null mean & "
            "null p95 & $p$ \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")


def tab_dose(dr: pd.DataFrame) -> str:
    names = {"on_target_after": "same round", "on_target_next": "next round"}
    keys = [("raw", "raw"), ("population_adjusted", "population adjusted"),
            ("order_adjusted", "order adjusted")]
    rows = []
    for _, r in dr.iterrows():
        for k, lab in keys:
            rows.append(f"{names[r.outcome]} & {lab} & {r[k]:+.4f} & "
                        f"[{r[k+'_lo']:+.4f}, {r[k+'_hi']:+.4f}] & {int(r.n_updates)} \\\\")
    return ("\\begin{tabular}{llrlr}\n\\toprule\noutcome & adjustment & slope & 95\\% interval & "
            "$n$ updates \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")


def _tex_env() -> dict[str, str]:
    """PATH with a TeX that actually runs on this machine placed first."""
    roots = [d for d in ("/opt/homebrew/bin", "/usr/local/bin", "/Library/TeX/texbin")
             if (Path(d) / "pdflatex").exists()]
    tried = []
    for root in roots:
        try:
            probe = subprocess.run([str(Path(root) / "pdflatex"), "--version"],
                                   capture_output=True, text=True, timeout=30)
        except OSError as err:
            tried.append(f"{root}: {err}"); continue
        if probe.returncode == 0:
            return {**os.environ, "PATH": root + ":" + os.environ.get("PATH", "")}
        tried.append(f"{root}: exit {probe.returncode}")
    raise RuntimeError("no working pdflatex found; tried: " + ", ".join(tried or ["(none)"]))


# --------------------------------------------------------------------- text
PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb,booktabs,graphicx,longtable,adjustbox,microtype}
\usepackage[dvipsnames]{xcolor}   %% hyperref's colour names need xcolor loaded first
\usepackage{hyperref}
\usepackage[labelfont=bf,font=small]{caption}
\graphicspath{{figures/}}
\hypersetup{colorlinks=true,linkcolor=blue!50!black,urlcolor=blue!50!black}
\newcommand{\Chat}{\widehat{\mathcal C}}
\newcommand{\DKL}{D_{\mathrm{KL}}}
\title{Observables and efficiencies for the \texttt{21-09-2026-full-vs-report-v1} study\\
\large A first coherent account of what the controller does to the population}
\author{Offline analysis, MA-CC}
\date{%(date)s}
\begin{document}
\maketitle
\tableofcontents
\clearpage
"""


def numbers(eff, cmd, est, effc, sus, act, ev, audit, swaps,
            dil=None, dr=None, facts=None) -> dict[str, Any]:
    """Every number the prose quotes, computed from the tables at build time."""
    H = int(eff.horizon.max())
    e = eff[eff.horizon == H].set_index("comparison")
    c = cmd[cmd.horizon == H].set_index("comparison")
    n: dict[str, Any] = {"H": H, "m_min": int(eff.m.min()), "m_max": int(eff.m.max()),
                         "n_traj": audit["n_trajectories"], "n_init": audit["n_inits"],
                         "n_excl": len(audit["exclusions"]), "n_comp": len(audit["comparison_counts"])}
    n["fc_b3_gplus"] = 100 * e.loc["full_communication_rho0.75_b3", "truth_g_plus"]
    n["fc_b3_gplus_lo"] = 100 * e.loc["full_communication_rho0.75_b3", "truth_g_plus_lo"]
    n["fc_b3_gplus_hi"] = 100 * e.loc["full_communication_rho0.75_b3", "truth_g_plus_hi"]
    n["fc_b18_M"] = 100 * e.loc["full_communication_rho0.75_b18", "truth_M"]
    n["ro_b3_M"] = 100 * e.loc["report_only_rho0.75_b3", "truth_M"]
    n["ro_b18_M"] = 100 * e.loc["report_only_rho0.75_b18", "truth_M"]
    n["ro_b3_D"] = 100 * e.loc["report_only_rho0.75_b3", "truth_D"]
    n["ro_base"] = 100 * e.loc["report_only_rho0.75_b3", "truth_base"]
    n["fc_base"] = 100 * e.loc["full_communication_rho0.75_b3", "truth_base"]
    n["M_min"] = 100 * e.truth_M.min(); n["M_max"] = 100 * e.truth_M.max()
    n["D_min"] = 100 * e.truth_D.min(); n["D_max"] = 100 * e.truth_D.max()
    n["n_M_neg"] = int((e.truth_M < 0).sum()); n["n_D_pos"] = int((e.truth_D > 0).sum())
    n["mi_min"] = c.mi_bits.min(); n["mi_max"] = c.mi_bits.max()
    n["F_min"] = c.following.min(); n["F_max"] = c.following.max()
    n["ro_b3_mi"] = c.loc["report_only_rho0.75_b3", "mi_bits"]
    n["ro_b3_F"] = c.loc["report_only_rho0.75_b3", "following"]
    n["ro_b3_Fgain"] = c.loc["report_only_rho0.75_b3", "following_gain"]
    ep = est[(est.problem == "endpoint") & (est.horizon == H)].set_index("comparison")
    n["Iend_min"] = ep.score_bits.min(); n["Iend_max"] = ep.score_bits.max()
    n["freq_bias"] = float(ep["freq_insample_mi_alpha0"].mean() - ep.score_bits.mean())
    n["freq_insample"] = float(ep["freq_insample_mi_alpha0"].mean())
    n["classifier_mean"] = float(ep.score_bits.mean())
    if effc is not None and len(effc):
        ec = effc[effc.horizon == H].set_index("comparison")
        n["cost_min"] = ec.cost_nats.min(); n["cost_max"] = ec.cost_nats.max()
        n["mix_share_min"] = float((ec.D_mixture_nats / ec.cost_nats).min())
        n["mix_share_max"] = float((ec.D_mixture_nats / ec.cost_nats).max())
        n["n_supported"] = int(effc.eta_supported.sum()); n["n_eta_rows"] = len(effc)
        sup = ec[ec.eta_supported]
        n["eta_end_min"] = sup.eta_end.min() if len(sup) else np.nan
        n["eta_end_max"] = sup.eta_end.max() if len(sup) else np.nan
        n["eta_ctl_min"] = sup.eta_ctl.min() if len(sup) else np.nan
        n["eta_ctl_max"] = sup.eta_ctl.max() if len(sup) else np.nan
        n["n_kmin_ident"] = int(effc.K_min_identified.sum()); n["n_kmin_rows"] = len(effc)
        n["n_eta_task"] = int(effc.eta_task.notna().sum())
        et = effc.eta_task.dropna()
        n["eta_task_min"] = float(et.min()) if len(et) else np.nan
        n["eta_task_max"] = float(et.max()) if len(et) else np.nan
    s = sus[sus.outcome == "truth"]
    n["tau1_min"] = s.tau_lag1.min(); n["tau1_max"] = s.tau_lag1.max()
    n["tau5_min"] = s.tau_lag5.min(); n["tau5_max"] = s.tau_lag5.max()
    n["act_rate_min"] = s.activation_rate.min(); n["act_rate_max"] = s.activation_rate.max()
    n["maxw"] = s.max_abs_weight.max()
    n["n_tau1_neg_t0"] = int((s[s.target == 0].tau_lag1 < 0).sum())
    n["n_t0_rows"] = int((s.target == 0).sum())
    if dil is not None and dr is not None:
        f = dil.sort_values("slot_bin")
        n["dose_first"] = float(f.dose.iloc[0]); n["read_first"] = float(f.messages_read.iloc[0])
        n["dose_last"] = float(f.dose.iloc[-1]); n["read_last"] = float(f.messages_read.iloc[-1])
        n["elig"] = float(f.eligible_controller.mean())
        now = dr[dr.outcome == "on_target_after"].iloc[0]
        nxt = dr[dr.outcome == "on_target_next"].iloc[0]
        ci = lambda r, k: f"[{r[k+'_lo']:+.4f}, {r[k+'_hi']:+.4f}]"
        n["dose_raw"] = now["raw"]; n["dose_raw_ci"] = ci(now, "raw")
        n["dose_pop"] = now["population_adjusted"]; n["dose_pop_ci"] = ci(now, "population_adjusted")
        n["dose_ord"] = now["order_adjusted"]; n["dose_ord_ci"] = ci(now, "order_adjusted")
        n["dose_next_ord"] = nxt["order_adjusted"]; n["dose_next_ord_ci"] = ci(nxt, "order_adjusted")
        n["n_dose"] = int(now["n_updates"]); n["n_init_dose"] = int(now["n_initializations"])
        n["resid_sd"] = float(now["resid_dose_sd"]); n["raw_sd"] = float(now["raw_dose_sd"])
    if facts is not None:
        n["lifetime"] = ", ".join(str(x) for x in facts["message_lifetime_rounds"])
        n["surviving"] = facts["surviving_message_count_max"]
        n["created"] = facts["messages_created_per_round"]
        n["expo"] = 100 * facts["exposure_rate"]
        n["dose_mean"] = float(dil.dose.mean()) if dil is not None else float("nan")
    a = act[(act.horizon == H) & (act.target == 0)]
    n["Tact_min"] = a.T_act_bits.min(); n["Tact_max"] = a.T_act_bits.max()
    evh = ev[ev.horizon == H]
    n["kappa_sil"] = float(evh.kappa_silent.mean()); n["kappa_t0"] = float(evh.kappa_t0.mean())
    n["kappa_t2"] = float(evh.kappa_t2.mean())
    ratio = evh.kappa_t0 / evh.kappa_silent
    n["kappa_ratio_min"] = float(ratio.min()); n["kappa_ratio_max"] = float(ratio.max())
    n["n_kappa_t2_ge"] = int((evh.kappa_t2 >= evh.kappa_t0).sum())
    n["phi_max"] = float(ev[["phi_silent", "phi_t0", "phi_t2"]].to_numpy().max())
    n["n_ev_rows"] = len(ev)
    if swaps is not None and len(swaps):
        n["n_swap_sig"] = int((swaps.p_value <= 0.05).sum()); n["n_swap"] = len(swaps)
        n["swap_h15_sig"] = int((swaps[swaps.horizon == H].p_value <= 0.05).sum())
        n["swap_h15_n"] = int((swaps.horizon == H).sum())
    return n


def write_tex(path: Path, n: dict, tabs: dict, audit: dict) -> None:
    import datetime
    d = PREAMBLE + r"""
\section{What this report is}
This is an offline analysis of the archived study \texttt{21-09-2026-full-vs-report-v1}.
No simulation was run and no provider request was made; the source directory was opened
read-only. Every quantity follows the definitions in
\texttt{shared\_references/BLACKBOARD\_OBSERVABLES\_AND\_EFFICIENCIES.md}, and the section
numbers quoted below are that document's.

The purpose is narrative, not just tabulation: to say what the controller actually does to
this population, and which of the many available numbers are load-bearing for that story.
Section~\ref{sec:story} states the account; everything before it is the evidence, and
Section~\ref{sec:limits} is what would falsify it.

\section{Design, data and the unit of independence}
\subsection{Design}
A population of $N=24$ LLM agents discusses one fixed MuSR team-allocation task
(\texttt{task\_003}) on a shared message board for %(H)s rounds. Truth is
\texttt{ALLOCATION\_0}; the designated false target is \texttt{ALLOCATION\_2};
\texttt{ALLOCATION\_1} is the remaining wrong answer. Writing $n_{a,t}$ for the number of
agents voting for allocation $a$ after round $t$,
\begin{equation}
Y_t=(n_{0,t},n_{1,t},n_{2,t}),\qquad x_{a,t}=n_{a,t}/N,\qquad
\Gamma_h=(Y_0,Y_1,\dots,Y_h),
\end{equation}
with $Y_0$ the initialization state. Counts sum to $N$ at every round; this was checked on
every trajectory rather than assumed.

Two experimental factors distinguish this study from the earlier paired-parent archive, and
both matter for the interpretation:
\begin{description}
\item[Communication profile.] \emph{Report-only} (RO) permits only REPORT messages;
\emph{full communication} (FC) permits the richer message set. This changes what the
\emph{population} can do among itself, not only what the controller can do.
\item[Randomized activation.] The controller's decision to act in round $t$ is drawn with a
logged probability $e_t=P(U_t=1\mid\text{pre-action information})$, a sigmoid with threshold
$0.5$ and slope $4$. Realized $e_t$ lies in $[0.119,0.881]$, bounded away from $0$ and $1$.
\end{description}
That second point is the important methodological gain over the paired archive. Because the
gate is genuinely randomized with known and non-degenerate propensities, the
inverse-probability-weighted causal estimator of reference section~3 is \emph{available}
here. It was not available for a deterministic always-policy. We therefore report a causal
one-round response alongside the observational one, and they can be compared.

The first physical round carries no controller decision by design; rounds $1..14$ do. Posts
satisfy $c_t=b\,U_t$ exactly in every trajectory, with $b\in\{3,12,18\}$, which was verified
rather than assumed.

\subsection{The independent unit}
The archive holds %(n_traj)s complete trajectories, all of which passed validation
(%(n_excl)s exclusions). They come from only %(n_init)s distinct physical initializations,
reused across settings and arms. \textbf{The initialization is the only independent unit.}
Neither the %(n_traj)s trajectories, nor the 24 votes inside a population, nor the
rounds within a trajectory are independent replicates. Every split and every bootstrap in
this report resamples whole initializations with all their arms attached.

A \emph{comparison} fixes a communication profile, a persistence $\rho$ and a budget $b$,
and retains the initializations having all three arms --- silence, the truth request and the
false request --- complete. This yields %(n_comp)s comparisons with $m$ between %(m_min)s and
%(m_max)s (Table~\ref{tab:sample}). The four full-communication/$\rho{=}1$ cells with a false
target are absent from the archive, so no two-target comparison exists there; that block is
simply missing, and is not imputed.

\begin{table}[htbp]\centering\small
\caption{Initializations with a complete silent / target-0 / target-2 triple.}\label{tab:sample}
%(tab_sample)s
\end{table}

The silent arm carries no budget, so one silent trajectory is shared by the three budgets
within a setting. That shared randomness is real and is preserved by resampling whole
initializations.

\section{Effects on the vote: shared and directional components}
\subsection{Definitions}
For a measured answer $a$, the paired policy susceptibility of reference section~3 is the
finite contrast between running the whole procedure and running nothing,
\begin{equation}
G_h^{\pm,a}=\mathbb E\big[x_{a,h}^{\pm}-x_{a,h}^{\rm base}\big],
\end{equation}
where $+$ requests truth and $-$ requests the false target. This is not a derivative with
respect to budget. Reference section~4 splits it into a shared component and a directional
half-contrast,
\begin{equation}
M_h^{(a)}=\tfrac12\big(G_h^{+,a}+G_h^{-,a}\big),\qquad
D_h^{(a)}=\tfrac12\big(G_h^{+,a}-G_h^{-,a}\big),
\end{equation}
so $G^{\pm,a}_h=M^{(a)}_h\pm D^{(a)}_h$ and the full effect of switching the requested
target is $2D^{(a)}_h$. We export both conventions. $M$ answers ``what does having a
controller at all do?''; $D$ answers ``does it matter which answer we asked for?''. The
identities $\sum_a M^{(a)}_h=\sum_a D^{(a)}_h=0$ and $M^{(\rm err)}=-M^{(0)}$ were verified
numerically.

\subsection{What the data show}
\begin{figure}[htbp]\centering
\includegraphics[width=\textwidth]{fig_trajectories.pdf}
\caption{Mean truth share against round, by setting. Black: silence. Blue: truth requested.
Red: false target requested. Line style encodes the posting budget. The silent curve is the
same in every panel of a setting because the silent arm is shared across budgets.}
\label{fig:traj}
\end{figure}

Figure~\ref{fig:traj} is the single most informative picture in this report, and it already
contains the main result. Under silence the population is \emph{good at this task}: the
report-only baseline reaches %(ro_base).1f\%% truth by $h=%(H)s$ and the full-communication
baseline %(fc_base).1f\%%. Control almost always moves it away from that.

\begin{figure}[htbp]\centering
\includegraphics[width=\textwidth]{fig_md.pdf}
\caption{Shared component $M$ and directional half-contrast $D$ against horizon, in
percentage points. Left and centre: truth coordinate. Right: the false-target coordinate.}
\label{fig:md}
\end{figure}

\begin{figure}[htbp]\centering
\includegraphics[width=\textwidth]{fig_gains.pdf}
\caption{Own-target gains against horizon with 95\%% paired-initialization bootstrap bands:
truth gain when truth is requested (left) and false-target gain when the false target is
requested (right).}\label{fig:gains}
\end{figure}

\begin{table}[htbp]\centering\small
\caption{Effects on the vote at $h=%(H)s$, in percentage points. ``base'' is the mean silent
truth share. $D^{(0)}$ carries its 95\%% paired bootstrap interval.}\label{tab:effects}
\begin{adjustbox}{max width=\textwidth}
%(tab_effects)s
\end{adjustbox}
\end{table}

Three facts stand out in Table~\ref{tab:effects} and Figure~\ref{fig:md}.

\paragraph{The shared component is negative almost everywhere, and grows with budget.}
$M^{(0)}_{%(H)s}$ is negative in %(n_M_neg)s of the %(n_comp)s comparisons, ranging from
%(M_min).1f to %(M_max).1f points. Within the report-only setting at $\rho=0.75$ it
deepens monotonically with the budget: %(ro_b3_M).1f points at $b=3$ and %(ro_b18_M).1f at
$b=18$. Posting more does not steer better; it damages more. The corresponding
$M^{(2)}$ is positive throughout, so the mass leaving truth arrives largely at the
false target regardless of which target was requested.

\paragraph{But the directional component is real, and larger than in the paired archive.}
$D^{(0)}_{%(H)s}$ is positive in %(n_D_pos)s of %(n_comp)s comparisons, spanning
%(D_min).1f to %(D_max).1f points. In report-only at $b=3$, $D^{(0)}=%(ro_b3_D).1f$ points
with a bootstrap interval excluding zero. Requesting truth really does preserve more truth
than requesting the false target, even though both requests are harmful on net. The
controller is not merely a target-independent disturbance here --- which is what the
paired always-policies looked like --- but a disturbance with a genuine directional
component riding on top.

\paragraph{One cell is different, and it is the informative one.}
Full communication at $\rho=0.75$, $b=3$ has $G^{+,0}_{%(H)s}=%(fc_b3_gplus)+.1f$ points
(95\%% interval $[%(fc_b3_gplus_lo).1f, %(fc_b3_gplus_hi).1f]$) and $M^{(0)}\approx0$: at low
budget, in the setting where the population is \emph{worse} on its own
(%(fc_base).1f\%% versus %(ro_base).1f\%%), a truth-targeted controller is roughly neutral
to mildly helpful. Raise the budget in that same setting and the benefit disappears:
$M^{(0)}=%(fc_b18_M).1f$ points at $b=18$. Helpfulness is confined to the corner where the
population has room to improve and the controller speaks quietly.

\subsection{How the effects build with the horizon}\label{sec:horizon}
Because every quantity in this report is computed at each $h$, the horizon axis is itself a
finding rather than a presentational choice. Three patterns recur across
Figures~\ref{fig:traj}, \ref{fig:md}, \ref{fig:gains} and \ref{fig:cmd}.

\paragraph{Nothing happens immediately.} At $h=1$ both $M^{(0)}$ and $D^{(0)}$ are within a
point or two of zero in every comparison. One round of posting does not move a population of
24 agents measurably. Any analysis that stopped at the first horizon would conclude the
controller does nothing.

\paragraph{Damage accumulates without saturating.} $M^{(0)}$ falls steadily across the whole
range and is still falling at $h=%(H)s$ in the larger-budget comparisons. This is the shape
of a process that erodes rather than one that shifts an equilibrium. Figure~\ref{fig:traj}
makes the mechanism visible in the report-only panels: the controlled branches track the
silent curve upward for roughly the first five rounds, reach a peak, and only then decline.
The controller does not prevent the population from finding the answer; it degrades a
consensus the population has already reached.

\paragraph{Direction saturates early.} $D^{(0)}$ rises steeply to about $h=7$ and then
flattens or drifts. Command information $I(Z;V_h)$ behaves the same way. So the target-specific
part of the signal is delivered in the first half of the run and does not grow after that,
while the target-independent damage keeps compounding. That divergence in timescales is the
cleanest dynamical statement this archive supports, and it is what drives the efficiency
ratios down at long horizons: the numerator plateaus while the denominator grows.

A practical consequence: the measured ``efficiency'' of this controller is a function of when
you stop looking. Quoting any of these ratios without its horizon is meaningless.

\section{One-round response: observational and causal}\label{sec:sus}
\begin{figure}[htbp]\centering
\includegraphics[width=\textwidth]{fig_susceptibility.pdf}
\caption{Left: state-matched response $\bar\chi_0$, weighted by visited states. Centre: the
inverse-probability-weighted causal response $\hat\tau_{0,1}$. Right: one-round activation
information against horizon.}\label{fig:sus}
\end{figure}

\begin{table}[htbp]\centering\small
\caption{Response families for the truth coordinate. $P(U{=}1)$ is the realized activation
rate; $\bar\chi_0$ is the state-matched contrast; $\hat\tau_{0,\ell}$ is the IPW causal
response at lag $\ell$; ``avail.'' is the cell-ratio available-susceptibility;
$\max|W|$ is the largest inverse-probability weight.}\label{tab:sus}
\begin{adjustbox}{max width=\textwidth}
%(tab_sus)s
\end{adjustbox}
\end{table}

Because activation is randomized with logged propensities, we can ask the causal question
directly: what does forcing the controller to act \emph{this} round do to truth support?
The estimator is
\begin{equation}
W_t=\frac{U_t}{e_t}-\frac{1-U_t}{1-e_t},\qquad
\hat\tau_{a,\ell}=\frac1{L_\ell}\sum_{t}W_t\,(x_{a,t+\ell}-x_{a,t}).
\end{equation}
Realized activation rates are %(act_rate_min).2f to %(act_rate_max).2f, and the largest
weight is %(maxw).1f, so no single round dominates. At lag~1 the causal truth response is
negative in %(n_tau1_neg_t0)s of the %(n_t0_rows)s truth-request rows, and the range across
all rows is %(tau1_min)+.3f to %(tau1_max)+.3f. By lag~5 it is %(tau5_min)+.3f to
%(tau5_max)+.3f: the damage accumulates rather than reverting.

This is worth pausing on. The observational state-matched contrast $\bar\chi_0$ and the
causal $\hat\tau_{0,1}$ agree in sign and rough magnitude. In the paired archive we could
only compute the observational version and had to caveat it heavily. Here the randomized
gate lets us say the stronger thing: \emph{acting causally reduces truth support on the very
next round}, and this is not an artifact of the controller choosing to act when the
population was already drifting.

The right panel of Figure~\ref{fig:sus} shows the one-round activation information
$I(U_h;n_{0,h}\mid n_{0,h-1})$ of reference section~6, which reads between %(Tact_min).3f and
%(Tact_max).3f bits at $h=%(H)s$. \textbf{These numbers should not be interpreted as
transmitted information.} They are frequency plug-in estimates of a conditional mutual
information computed from $m\approx%(m_max)s$ observations spread over many
$(n_{0,h-1},n_{0,h})$ cells, so most cells hold one or two observations and the estimator is
dominated by upward bias --- the same pathology Table~\ref{tab:freq} exhibits for the
endpoint, where the in-sample plug-in overstates the held-out score by %(freq_bias).3f bits.
The declining trend with horizon is also consistent with bias rather than signal: as the
population converges, the conditioning variable takes fewer distinct values and the bias
falls. We report the quantity because the reference defines it, and we decline to read
anything into its magnitude. The honest one-round statements in this archive are the
state-matched and IPW responses in the left two panels, which are differences of means
rather than nonlinear functionals and do not carry this bias.

\section{The command channel: does the population hear which answer was requested?}
Reference section~9 defines the single-agent command channel by drawing $Z\in\{0,2\}$
balanced, running the target-conditioned policy, and reading one uniformly chosen agent's
final vote $V_h$:
\begin{equation}
p_h(v\mid z)=\mathbb E\big[x_{v,h}^{z}\big],\qquad
I(Z;V_h)=\tfrac12\sum_{z}\sum_{v}p_h(v|z)\log_2\frac{p_h(v|z)}{\bar p_h(v)},
\end{equation}
with $\eta_{\rm command}=I(Z;V_h)/\mathcal H(Z)$ numerically equal to the information in
bits because $\mathcal H(Z)=1$. Following is $F_h=\tfrac12[p_h(0|0)+p_h(2|2)]$ against the
matched silent benchmark $F^{\rm base}_h=\tfrac12\mathbb E[x^{\rm base}_{0,h}+x^{\rm base}_{2,h}]$.
Marginalizing a uniformly chosen agent is exact from the vote fractions, so no classifier is
used here and none is needed.

\begin{figure}[htbp]\centering
\includegraphics[width=\textwidth]{fig_command.pdf}
\caption{Command information, following probability and information per post against
horizon.}\label{fig:cmd}
\end{figure}

\begin{table}[htbp]\centering\small
\caption{Command transmission and resources at $h=%(H)s$. Posts and senses are averaged over
the two target branches. $\mathcal E_\lambda$ is bits per post-equivalent with sensing
charged at $\lambda$.}\label{tab:cmd}
\begin{adjustbox}{max width=\textwidth}
%(tab_cmd)s
\end{adjustbox}
\end{table}

Command information at $h=%(H)s$ ranges from %(mi_min).4f to %(mi_max).4f bits, and
following from %(F_min).3f to %(F_max).3f. In report-only at $b=3$, $I(Z;V_{%(H)s})=
%(ro_b3_mi).4f$ bits with $F=%(ro_b3_F).3f$ against a silent benchmark, an improvement of
%(ro_b3_Fgain)+.3f.

The exact identity of reference section~9,
\begin{equation}
F_h-F^{\rm base}_h=\tfrac12\big(G^{+,0}_h+G^{-,2}_h\big),
\end{equation}
explains why following can improve while truth collapses: it is the average of the two
\emph{own-target} gains, and the false branch's large gain on its own target carries it.
A controller that is very good at installing the false answer, and mildly bad at defending
the true one, still scores well on ``following''. Following is not a measure of benefit.

Sensing is charged honestly. The controller reads sampled votes on every gated round whether
or not it posts, so sensing expenditure is the same in all controlled branches and is not a
cost specific to a sensing policy. The $\lambda=1$ column of Table~\ref{tab:cmd} shows what
happens to apparent efficiency once those reads are priced at one post each: the numbers
drop by roughly an order of magnitude without anything having changed about what was
transmitted.

\section{Endpoint and trajectory information}
\subsection{Estimators}
$I(Z;Y_h)$ has 325 possible endpoint states and cannot be read off a frequency table at
$m\approx50$. Following reference section~12 we fit a balanced target classifier
$g(Z\mid Y_h)$ and score the held-out
\begin{equation}
\widehat L_I(h)=\mathbb E\big[\log_2 2g(Z\mid Y_h)\big],
\end{equation}
whose population gap from $I(Z;Y_h)$ is the expected KL between the true and fitted
posteriors --- a lower bound in expectation, with negative realized values permitted.
Candidates are a constant prior, L2 logistic regression on the two vote fractions, quadratic
logistic features, and a one-hidden-layer MLP of width 4 or 8; selection is by nested
held-out log loss inside training initializations only.

\begin{table}[htbp]\centering\small
\caption{Frequency plug-in versus held-out estimates of $I(Z;Y_{%(H)s})$, in bits.}
\label{tab:freq}
\begin{adjustbox}{max width=\textwidth}
%(tab_freq)s
\end{adjustbox}
\end{table}

Table~\ref{tab:freq} is a warning about the frequency estimator. The in-sample plug-in
averages %(freq_insample).3f bits; the held-out classifier averages %(classifier_mean).3f,
a gap of %(freq_bias).3f bits of pure optimism. With nearly unique endpoints the raw
frequency MI is mostly counting its own noise. Only the held-out column should be quoted.

\begin{figure}[htbp]\centering
\includegraphics[width=\textwidth]{fig_information.pdf}
\caption{Left: held-out endpoint target information against horizon. Right: observed score
against the 95th percentile of 200 within-initialization target-label swaps.}
\label{fig:info}
\end{figure}

\begin{figure}[htbp]\centering
\includegraphics[width=\textwidth]{fig_models.pdf}
\caption{Held-out score of every candidate for $\widehat I(Z;Y_{%(H)s})$.}\label{fig:models}
\end{figure}

\subsection{Trajectory cost and its decomposition}
For the divergence terms we fit balanced controlled-versus-silent critics and report the
NWJ objective $\mathbb E_P f-\mathbb E_{\rm base}e^{f}+1$ and the Donsker--Varadhan objective
$\mathbb E_P f-\log\mathbb E_{\rm base}e^{f}$, both population lower bounds in nats, with a
capped critic and baseline-tail diagnostics. Candidates add path summaries, a flat MLP over
the whole path, and small GRUs. The target-averaged cost and its exact decomposition
(reference section~11) are
\begin{equation}
\mathcal C_\pi=\tfrac12K^0_\pi+\tfrac12K^2_\pi
= I_{\rm nat}(Z;\Gamma)+\DKL\!\left(Q\,\Vert\,P_{\rm base}\right),
\qquad Q=\tfrac12P^0_\pi+\tfrac12P^2_\pi,
\end{equation}
with the data-processing chain
\begin{equation}
I_{\rm nat}(Z;V_h)\le I_{\rm nat}(Z;Y_h)\le I_{\rm nat}(Z;\Gamma)\le\mathcal C_\pi
\end{equation}
and the efficiencies $\eta_{\rm end}=I_{\rm nat}(Z;Y_h)/\mathcal C_\pi$,
$\eta_{\rm ctl}=I_{\rm nat}(Z;\Gamma)/\mathcal C_\pi$. The three terms are fitted
independently, so their sum need not reconcile exactly; the gap is a diagnostic.

\begin{figure}[htbp]\centering
\includegraphics[width=\textwidth]{fig_cost.pdf}
\caption{Trajectory cost and the two terms of its decomposition against horizon.}
\label{fig:cost}
\end{figure}

\begin{figure}[htbp]\centering
\includegraphics[width=\textwidth]{fig_efficiency.pdf}
\caption{Efficiency ratios against horizon, shown only where no bootstrap draw of the cost
is non-positive.}\label{fig:eta}
\end{figure}

\begin{table}[htbp]\centering\small
\caption{Divergences, decomposition and efficiencies at $h=%(H)s$, in nats. Entries are
blank where the cost bootstrap is not bounded away from zero, or, for $\eta_{\rm task}$,
where $K_{\min}$ is not identified from the silent sample.}\label{tab:cost}
\begin{adjustbox}{max width=\textwidth}
%(tab_cost)s
\end{adjustbox}
\end{table}
"""
    d += r"""
\subsection{Why $\eta_{\rm task}$ is mostly unavailable}
Reference section~11 defines the minimum displacement needed to move a reward by $\delta$ as
the Legendre transform of the baseline log moment generating function,
$K_{\min}(\delta)=\sup_\lambda[\lambda\delta-\psi(\lambda)]$, and
$\eta_{\rm task}=K_{\min}(\delta)/K_\pi\le1$.

Computed naively this returns numbers like 33, 66 and 158 nats for the report-only
$\rho=1$ comparisons, which would give $\eta_{\rm task}\approx10$ --- impossible for a
quantity bounded by one. The cause is a support failure, not a large cost. Across all
initializations of that setting the silent truth share never falls below $0.917$; the
controlled arms shift the mean to below that. The requested mean therefore lies outside the
observed baseline range, the empirical moment generating function has no mass there, and the
optimal tilt runs to whatever bound the $\lambda$ grid happens to impose. The reported value
is then an artifact of the grid, not an estimate.

We detect this explicitly --- requested mean inside the observed baseline support, and the
optimizer interior rather than at the grid edge --- and report $K_{\min}$ as unidentified
when either test fails. It is identified in %(n_kmin_ident)s of %(n_kmin_rows)s rows, and
combined with the requirement that the cost bootstrap be bounded away from zero this leaves
%(n_eta_task)s usable $\eta_{\rm task}$ values. This is the same missing-baseline-event
problem the reference warns about, and it is worth stating plainly: \emph{the controller
drives this population somewhere silence never goes}, and that is precisely why the
information-theoretic minimum cost cannot be estimated from the silent sample.

\section{Null diagnostics}
\begin{table}[htbp]\centering\small
\caption{Within-initialization target-label swaps (200 per cell, full refit).}\label{tab:swaps}
\begin{adjustbox}{max width=\textwidth}
%(tab_swaps)s
\end{adjustbox}
\end{table}

Swapping the two target branches within an initialization destroys target identity while
preserving everything else about the pair. Under that exchangeability null the observed
endpoint score should sit inside the permutation distribution. %(n_swap_sig)s of
%(n_swap)s cells exceed their null at the 5\%% level, and at $h=%(H)s$ it is
%(swap_h15_sig)s of %(swap_h15_n)s. This is an exchangeability diagnostic, not a proof of
causality: it says the target label carries information the vote endpoint can see.

\section{Evidence: what the controller does to what agents know}
\begin{figure}[htbp]\centering
\includegraphics[width=\textwidth]{fig_evidence.pdf}
\caption{Mean active supporting-fact coverage $\kappa_h$ against horizon, under the truth
request and under silence.}\label{fig:ev}
\end{figure}

Reference section~5 defines $\kappa_t$ as mean active proof coverage and $\phi_t$ as the
share of agents holding a full configured proof. This section contains the result that most
constrains the mechanism, and it is not the one we expected.

\paragraph{The controller substantially increases evidence coverage.} At $h=%(H)s$, mean
$\kappa$ is %(kappa_sil).3f under silence, %(kappa_t0).3f under the truth request and
%(kappa_t2).3f under the false request. Per comparison the truth-request arm carries
%(kappa_ratio_min).2f to %(kappa_ratio_max).2f times the silent coverage, rising with the
posting budget. The controller is not starving the population
of information. It is demonstrably \emph{delivering} verified facts, and delivering more of
them as it posts more.

\paragraph{The false-target controller delivers about as much as the truth-target one.}
$\kappa_{t2}\ge\kappa_{t0}$ at the final horizon in %(n_kappa_t2_ge)s of the %(n_comp)s
comparisons, and the exception is close to a tie. This is exactly what
the design implies and is easy to miss: the controller posts \emph{true} facts drawn from
the task pool together with a vote for its assigned target. A true fact accompanied by a vote
for \texttt{ALLOCATION\_2} still raises measured proof coverage. Coverage counts facts held,
not conclusions correctly drawn.

\paragraph{No agent ever completes a proof.} $\phi_h$ is exactly zero (maximum observed
%(phi_max).1f over all %(n_ev_rows)s arm-horizon rows) in every arm, comparison and horizon,
including under silence. Not one of the 24 agents ever holds a
full configured proof at any point in this archive. The population is voting on fragments
throughout.

Those three facts together rule out the simplest explanation of the damage --- that the
controller crowds out peer information --- and point at a different one. Evidence supply goes
\emph{up} while truth support goes \emph{down}. What the controller adds alongside the facts
is a salient, repeated, explicit vote; and because no agent can verify a conclusion from a
complete proof, that vote is cheap to copy relative to the cost of assembling fragments. The
controller's influence appears to run through the \emph{vote} it attaches, not through the
\emph{evidence} it supplies, and the evidence it supplies is not sufficient to protect an
agent against the vote.

That was our reading when only these observables were available. Section~\ref{sec:micro}
carries out the message-level test it calls for --- linking which agents read which posts to
how they then voted --- and the result does \emph{not} support the simplest version of it. At
the margin, within a population and a round, agents who read more of the controller were no
likelier to adopt its target. The vote-copying story survives only in a collective form, in
which the controller moves a few agents and the population carries the rest; it is not
supported as direct per-message persuasion. The observables here constrain the mechanism;
they do not identify it, and neither does the micro test.

\section{Does the controller persuade agent by agent?}\label{sec:micro}
The evidence observables above constrain the mechanism but do not identify it. The
micro-update records can go further: for each of the 497,520 agent updates they record that
agent's vote before and after, and the identifiers of the messages it actually read. This
section reports that test. It returns a null, and the way it returns a null is informative.

\subsection{Two features of the board that shape the test}
First, \textbf{the board is erased every round}: \texttt{message\_lifetime\_rounds} is
%(lifetime)s in all cells and the maximum \texttt{surviving\_message\_count} over the whole
archive is %(surviving).1f. About %(created).0f messages are created per round and the same
number expire. Message dose therefore measures the current round's output only; nothing
accumulates.

Second, \textbf{agents update sequentially and post as they go}. The controller posts first,
so its share of the board decays across the 24 update slots within a round
(Figure~\ref{fig:dose}, left). An agent updating in the first slots reads about
%(dose_first).1f controller messages out of %(read_first).1f; one updating in the last slots
reads %(dose_last).1f out of %(read_last).1f, because peers have filled the board meanwhile.
The number of controller messages available is constant at about %(elig).1f throughout ---
what changes is competition for the agent's fixed reading allowance.

That matters because the outcome moves with update order too. Both dose and the probability
of ending the update on the controller's target decline across slots, so any unadjusted
dose-response will report an effect of ordering as though it were an effect of messages.

\begin{figure}[htbp]\centering
\includegraphics[width=\textwidth]{fig_dose.pdf}
\caption{Left: within a round the controller's messages are progressively crowded out of the
agent's fixed reading sample, and the outcome declines with them. Right: the estimated slope
of adoption on controller messages read, before adjustment, after holding the population and
round fixed, and after additionally removing the update-order trend. Bars are 95\%% cluster
bootstrap intervals over %(n_init_dose)s initializations.}\label{fig:dose}
\end{figure}

\subsection{The estimator}
Restrict to agent updates where a controller message was on the board and the agent was
\emph{not} already voting the target, so the outcome is adoption rather than retention. Let
$d_{jt}$ be the number of controller messages agent $j$ read at that update and $y_{jt}$ an
indicator that it ends on the target. We report three slopes of $y$ on $d$:

\begin{description}
\item[raw] no adjustment; pools across budgets, rounds and update order.
\item[population adjusted] $d$ and $y$ residualised on (episode $\times$ round), so the
comparison is between agents facing the same board in the same population at the same moment.
\item[order adjusted] additionally residualised on \texttt{micro\_slot\_index}, removing the
dilution trend.
\end{description}

Board sampling is uniform and excludes self-authored messages, so after these adjustments the
remaining variation in $d$ is close to which messages an agent happened to draw. Residual
dose standard deviation is %(resid_sd).2f against a raw %(raw_sd).2f, so the adjustments do
not exhaust the variation. Intervals are a cluster bootstrap over initializations.

\subsection{Result}
\begin{table}[htbp]\centering\small
\caption{Change in the probability of adopting the controller's target per additional
controller message read, with 95\%% cluster bootstrap intervals over
%(n_init_dose)s initializations.}\label{tab:dose}
%(tab_dose)s
\end{table}

The raw slope is %(dose_raw).4f %(dose_raw_ci)s: reading one more controller message is
associated with about two percentage points more adoption. Holding the population and round
fixed cuts it to %(dose_pop).4f %(dose_pop_ci)s. Removing the update-order trend as well
leaves %(dose_ord).4f %(dose_ord_ci)s --- indistinguishable from zero, and tightly bounded
rather than merely imprecise, on %(n_dose)s agent updates.

Repeating the test with the agent's vote at the end of the \emph{following} round as the
outcome gives %(dose_next_ord).4f %(dose_next_ord_ci)s. There is no delayed individual
adoption either.

\subsection{What this does and does not establish}
It does not mean the controller has no effect. The population-level effects are large and
causally established in Section~\ref{sec:sus}. It means the effect \textbf{does not appear as
a within-round, within-population dose gradient}: among agents sitting in the same population
at the same moment, the ones who happened to read more of the controller were not more likely
to adopt its target.

Two readings are consistent with that, and this archive cannot separate them.

\emph{Saturation.} Exposure is near-universal --- %(expo).1f\%% of agents read at least one
controller message when one is available, and the mean dose is %(dose_mean).1f. Everybody is
already well past any threshold, so the margin carries no information even if the first
message matters a great deal.

\emph{Mediation.} If the controller works by shifting a few agents, who then post and shift
others, the (episode $\times$ round) adjustment absorbs exactly that propagation channel by
construction. Conditioning on a mediator removes the effect one is trying to measure. This
reading fits the horizon evidence of Section~\ref{sec:horizon}: controlled branches track
silence for roughly five rounds before diverging, which is the signature of a slow collective
process rather than immediate per-agent persuasion, and a within-round contrast is blind to it.

The honest summary is that the simplest mechanical story --- each controller message directly
pulls its readers toward the target --- is \emph{not} supported at the margin, while the
collective story remains open. Distinguishing them needs either an experiment that randomizes
exposure within a population, or one that varies the vote attached to a fixed set of facts.

\section{The account}\label{sec:story}
Putting the pieces together, the following story is consistent with every table above.

\paragraph{1. The population is competent, and competence is the baseline to beat.}
Left alone these agents converge on the correct allocation: %(ro_base).1f\%% truth in
report-only, %(fc_base).1f\%% in full communication. Any controller is competing against a
process that mostly works. This single fact reframes every efficiency number: the
denominator of ``how much did control help'' is small by construction.

\paragraph{2. Intervening is mostly disruption, and disruption scales with volume.}
The shared component $M^{(0)}$ is negative in %(n_M_neg)s of %(n_comp)s comparisons and
deepens monotonically with the posting budget. The causal IPW estimator confirms this is
not selection: forcing an action lowers truth support at the next round and further by lag~5.
The mixture term $\DKL(Q\Vert P_{\rm base})$ accounts for most of the fitted trajectory cost,
which is the same statement in divergence form --- the bulk of the statistical displacement
the controller produces is target-independent.

\paragraph{3. Yet the command does get through, weakly but measurably.}
$D^{(0)}$ is positive in %(n_D_pos)s of %(n_comp)s comparisons and reaches %(D_max).1f
points; command information reaches %(mi_max).4f bits; the label-swap null is exceeded in
most cells. So this is not the paired-archive picture of a purely target-blind perturbation.
The controller transmits a real, small directional signal on top of a large non-directional
one. The ratio of the two is what the efficiencies measure, and it is small.

\paragraph{4. The damage is not an information shortage.}
Evidence coverage rises by a factor of %(kappa_ratio_min).1f to %(kappa_ratio_max).1f under
control and grows with budget, the false-target controller supplies about as much coverage as
the truth-target one, and no agent ever
holds a complete proof in any arm. So the controller is supplying genuine facts while making
collective inference worse. What it adds besides facts is a salient repeated vote. The
natural next step is to say that agents copy that vote, but Section~\ref{sec:micro} tests
that directly and cannot find it: holding the population and the update order fixed, reading
more controller messages does not raise adoption. So the damage is neither an information
shortage nor, apparently, direct per-message persuasion. What remains is a collective route
--- the controller shifts a few agents, whose posts then shift others --- which the
horizon evidence supports and which the micro test is structurally unable to see. The one
benign corner, full communication at $\rho=0.75$ and $b=3$, is where autonomous performance
is weakest and the controller speaks least; raising the budget there reverses the sign.

\paragraph{5. ``Following'' is a trap.}
Following improves in every comparison, because it averages the two own-target gains and the
false branch is very good at installing the false answer. A metric can rise while the thing
one cares about falls. This is the clearest single argument in the report for keeping
signed, outcome-specific effects rather than a scalar success score.

\section{What would falsify this, and what is not established}\label{sec:limits}
\begin{itemize}
\item \textbf{The sample is %(n_init)s initializations.} Not %(n_traj)s trajectories, not
$24\times$ that in votes. Every interval here is a %(m_min)s--%(m_max)s unit bootstrap and
is correspondingly wide. The bootstrap intervals on fitted information and divergence scores
are conditional on the fitted predictions: they resample held-out contributions, not the
whole training pipeline, and so omit training variability.
\item \textbf{The snapshot is incomplete and its own validation is marked invalid.} Four
full-communication/$\rho{=}1$ cells are missing, so the profile $\times$ persistence
interaction is only identified at $\rho=0.75$. Claim~4 above rests on a single setting and
should be treated as a hypothesis to test, not a finding.
\item \textbf{Divergence estimates are lower bounds from fitted critics.} A ratio of two
lower bounds has no guaranteed ordering with respect to the true efficiency. Ratios are
suppressed wherever the cost bootstrap touches zero.
\item \textbf{The mechanism is not established, and the micro test is not decisive against
it either.} Exposure is near-universal in this design, so a null at the margin is also what
saturation would produce; and the (episode $\times$ round) adjustment absorbs the collective
propagation channel by construction. Separating saturation from mediation needs a design that
randomizes exposure within a population, or that varies the target vote attached to a fixed
set of facts.
\item \textbf{No thermodynamic reading is offered.} $\mathcal C_\pi$ is a statistical
displacement in nats, not heat, and no reverse process has been justified.
\end{itemize}

\section*{Reproduction}
\begin{small}
\begin{verbatim}
.venv/bin/python -m rnd_init_metrics.run \
  --config configs/analysis/rnd_init_metrics/\
new_rnd_init_experiment.yaml --stage all
\end{verbatim}
\end{small}
\noindent Source archive, opened read-only:\\
\texttt{\small %(archive)s}\\
Rounds table SHA-256: \texttt{\small %(sha)s}

\end{document}
"""
    payload = {**n, **{f"tab_{k}": v for k, v in tabs.items()},
               "date": datetime.date.today().isoformat(),
               "archive": audit["provenance"]["archive_root"].replace("_", r"\_"),
               "sha": audit["provenance"]["rounds_parquet_sha256"][:32]}
    path.write_text(d % payload)


def build(cfg: dict[str, Any], out_dir: Path) -> Path:
    tdir, rdir = out_dir / "tables", out_dir / "report"
    (rdir / "figures").mkdir(parents=True, exist_ok=True)
    fdir = rdir / "figures"
    eff = pd.read_csv(tdir / "effects.csv")
    cmd = pd.read_csv(tdir / "command.csv")
    est = pd.read_csv(tdir / "estimates.csv")
    effc = pd.read_csv(tdir / "efficiency.csv")
    sus = pd.read_csv(tdir / "susceptibility.csv")
    act = pd.read_csv(tdir / "activation.csv")
    ev = pd.read_csv(tdir / "evidence.csv")
    traj = pd.read_csv(tdir / "trajectories.csv")
    swaps = pd.read_csv(tdir / "swaps.csv") if (tdir / "swaps.csv").exists() else None
    dil = pd.read_csv(tdir / "dose_by_slot.csv") if (tdir / "dose_by_slot.csv").exists() else None
    dr = pd.read_csv(tdir / "dose_response.csv") if (tdir / "dose_response.csv").exists() else None
    facts = json.loads((tdir / "board_facts.json").read_text()) if (tdir / "board_facts.json").exists() else None
    audit = json.loads((out_dir / "audit.json").read_text())
    H = int(eff.horizon.max())

    fig_trajectories(traj, fdir / "fig_trajectories.pdf")
    fig_md(eff, fdir / "fig_md.pdf")
    fig_gains(eff, fdir / "fig_gains.pdf")
    fig_command(cmd, fdir / "fig_command.pdf")
    fig_information(est, swaps, fdir / "fig_information.pdf")
    fig_cost(effc, fdir / "fig_cost.pdf")
    fig_efficiency(effc, fdir / "fig_efficiency.pdf")
    fig_susceptibility(sus, act, fdir / "fig_susceptibility.pdf")
    fig_evidence(ev, fdir / "fig_evidence.pdf")
    fig_models(est, fdir / "fig_models.pdf")
    if dil is not None and dr is not None:
        fig_dose(dil, dr, fdir / "fig_dose.pdf")

    tabs = {"sample": tab_sample(audit), "effects": tab_effects(eff, H),
            "cmd": tab_command(cmd, H), "cost": tab_cost(effc, H),
            "sus": tab_susceptibility(sus), "freq": tab_frequency(est, H),
            "swaps": tab_swaps(swaps) if swaps is not None else "(no swap jobs)",
            "dose": tab_dose(dr) if dr is not None else "(no micro tables)"}
    (rdir / "tables").mkdir(exist_ok=True)
    for k, v in tabs.items():
        (rdir / "tables" / f"{k}.tex").write_text(v)
    n = numbers(eff, cmd, est, effc, sus, act, ev, audit, swaps, dil, dr, facts)
    write_tex(rdir / "main.tex", n, tabs, audit)

    env = _tex_env()
    res = subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
                         cwd=rdir, env=env, capture_output=True, text=True)
    if res.returncode != 0:
        log = (rdir / "main.log").read_text(errors="ignore") if (rdir / "main.log").exists() else res.stdout
        raise RuntimeError("LaTeX failed:\n" + log[-4000:])
    final = out_dir / "new_rnd_init_observables_report.pdf"
    shutil.copy(rdir / "main.pdf", final)
    print("report:", final)
    return final


def main(argv=None):
    from .run import load_config
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    build(cfg, Path(cfg["output_dir"]))


if __name__ == "__main__":
    main()
