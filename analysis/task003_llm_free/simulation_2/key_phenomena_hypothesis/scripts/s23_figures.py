"""Figures and small tables for steps 2 and 3, from the outputs of s2_mechanism.py and s3_branches.py
(results/key_phenomena/). Writes into key_phenomena_hypothesis/figures/ and tables/:

Step 2 (q = 3, 6, 9, 12; 8 rho; t = 30 unless stated):
  step2_fig1_G_and_D.pdf       left: G against q, one line per rho; right: D for 3->6, 6->9, 9->12 against rho
  step2_fig2_D_map.pdf         D(step, rho) as a map with its value; grey: 95% interval includes 0
  step2_fig3_memory.pdf        agent-origin facts per agent, controller-only facts per agent and swarm coverage of
                               agent-origin facts against q (false: solid, silent: dashed) for rho 0.25, 0.5, 0.75, 1
  step2_fig4_split.pdf         the belief shift split into direct (controller facts) and indirect (which agent facts are
                               kept) against q, and the increments of each against D
  tables/step2_D.csv, step2_split.csv
Step 3 (rho 0.25 and 0.75; post-stop modes; t to 130):
  step3_fig1_delta.pdf         delta(t) for the four modes, one panel per setup and rho
  step3_fig2_carriers.pdf      controller-only facts per agent, swarm coverage of agent-origin facts and the
                               controller-only share of readable posts over time, false arm, per mode
  tables/step3_summary.csv, step3_validation.csv, step2_validation.csv

Usage, from the repository root:
    .venv/bin/python analysis/task003_llm_free/simulation_2/key_phenomena_hypothesis/scripts/s23_figures.py
"""
from __future__ import annotations
import pathlib, shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
BASE = HERE.parent
IN = HERE.parents[2] / "results" / "key_phenomena"
FIG, TAB = BASE / "figures", BASE / "tables"
SETUPS = ["symmetric", "nosolution"]
RHOS = [1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125]
RCOL = dict(zip(RHOS, plt.cm.viridis(np.linspace(0, 0.92, len(RHOS)))))
STEPS = ["3->6", "6->9", "9->12"]
MODES = ["normal", "no_comm", "frozen_board", "purge_controller_facts"]
MLAB = {"normal": "normal", "no_comm": "no reading or posting", "frozen_board": "frozen board",
        "purge_controller_facts": "controller-only facts removed"}
MCOL = {"normal": "black", "no_comm": "tab:gray", "frozen_board": "tab:blue", "purge_controller_facts": "tab:red"}
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "legend.fontsize": 6.5, "figure.dpi": 150})


def save(fig, name):
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)


def split_table(C):
    c = C[C.t == 30.0].pivot_table(index=["setup", "rho", "q"], columns="arm", values=["p_A2", "p_A2_agentonly"])
    x = pd.DataFrame({"direct": c[("p_A2", "false")] - c[("p_A2_agentonly", "false")],
                      "indirect": c[("p_A2_agentonly", "false")] - c[("p_A2", "silent")]})
    x["total"] = x.direct + x.indirect
    return x.reset_index()


def step2(G, D, C):
    g, d = G[G.t == 30.0], D[D.t == 30.0]
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.2))
    for i, s in enumerate(SETUPS):
        ax = axes[i, 0]
        for rho in RHOS:
            x = g[(g.setup == s) & (g.rho == rho)].sort_values("q")
            ax.errorbar(x.q, x.G, yerr=1.96 * x.G_se, color=RCOL[rho], marker="o", ms=2.5, lw=1, capsize=1.2,
                        label=f"$\\rho$ = {rho:g}")
        ax.set_xticks([3, 6, 9, 12])
        ax.set_xlabel("posts read per action $q$")
        ax.set_ylabel(f"{s}\n$G$ at $t = 30$, target A2")
        ax = axes[i, 1]
        for step, col, mk in zip(STEPS, ("tab:blue", "tab:purple", "tab:red"), "os^"):
            x = d[(d.setup == s) & (d.step == step)].sort_values("rho")
            ax.errorbar(x.rho, x.D, yerr=1.96 * x.D_se, color=col, marker=mk, ms=2.5, lw=1, capsize=1.2,
                        label=f"$D$({step.replace('->', '→')})")
        ax.axhline(0, color="0.5", lw=0.6)
        ax.set_xlabel("memory $\\rho$")
        ax.set_ylabel("paired increment $D$ at $t = 30$")
    axes[0, 0].legend(fontsize=5.5, ncol=2)
    axes[0, 1].legend(fontsize=6)
    save(fig, "step2_fig1_G_and_D.pdf")

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for j, s in enumerate(SETUPS):
        x = d[d.setup == s]
        m = x.pivot_table(index="rho", columns="step", values="D").reindex(index=RHOS[::-1], columns=STEPS)
        se = x.pivot_table(index="rho", columns="step", values="D_se").reindex(index=RHOS[::-1], columns=STEPS)
        ax = axes[j]
        im = ax.imshow(m.to_numpy(), origin="lower", cmap="RdBu_r", vmin=-0.2, vmax=0.2, aspect="auto")
        for a in range(m.shape[0]):
            for b in range(m.shape[1]):
                v, e = m.iat[a, b], se.iat[a, b]
                ax.text(b, a, f"{v:+.3f}", ha="center", va="center", fontsize=5.5,
                        color="0.55" if abs(v) <= 1.96 * e else "black")
        ax.set_xticks(range(3), [s_.replace("->", "→") for s_ in STEPS])
        ax.set_yticks(range(len(RHOS)), [f"{r:g}" for r in RHOS[::-1]])
        ax.set_xlabel("step in $q$")
        ax.set_ylabel("memory $\\rho$")
        ax.set_title(s)
        fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02, label="$D$")
    save(fig, "step2_fig2_D_map.pdf")

    c = C[C.t == 30.0]
    sel = [0.25, 0.5, 0.75, 1.0]
    cols = [("mem_agent", "agent-origin facts per agent"), ("mem_ctrl", "controller-only facts per agent"),
            ("cover_agent", "distinct agent-origin facts\nin the swarm")]
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.6))
    for i, s in enumerate(SETUPS):
        for j, (col, lab) in enumerate(cols):
            ax = axes[i, j]
            for rho in sel:
                for arm, ls in (("false", "-"), ("silent", "--")):
                    if col == "mem_ctrl" and arm == "silent":
                        continue
                    x = c[(c.setup == s) & (c.rho == rho) & (c.arm == arm)].sort_values("q")
                    ax.plot(x.q, x[col], color=RCOL[rho], ls=ls, marker="o", ms=2.5, lw=1,
                            label=f"$\\rho$ = {rho:g}" if arm == "false" else None)
            ax.set_xticks([3, 6, 9, 12])
            ax.set_title(f"{s}: {lab}", fontsize=6.5)
            if i == 1:
                ax.set_xlabel("$q$")
    axes[0, 0].legend(fontsize=5.5)
    save(fig, "step2_fig3_memory.pdf")

    sp = split_table(C)
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.6))
    for i, s in enumerate(SETUPS):
        for j, part in enumerate(("direct", "indirect")):
            ax = axes[i, j]
            for rho in RHOS:
                x = sp[(sp.setup == s) & (sp.rho == rho)].sort_values("q")
                ax.plot(x.q, x[part], color=RCOL[rho], marker="o", ms=2.5, lw=1, label=f"$\\rho$ = {rho:g}")
            ax.axhline(0, color="0.5", lw=0.6)
            ax.set_xticks([3, 6, 9, 12])
            ax.set_title(f"{s}: {part} part of the belief shift", fontsize=6.5)
            if i == 1:
                ax.set_xlabel("$q$")
        ax = axes[i, 2]
        rows = []
        for rho in RHOS:
            x = sp[(sp.setup == s) & (sp.rho == rho)].set_index("q")
            for step in STEPS:
                q1, q2 = map(int, step.split("->"))
                e = d[(d.setup == s) & (d.rho == rho) & (d.step == step)].iloc[0]
                rows.append((e.D, e.D_se, x.loc[q2, "direct"] - x.loc[q1, "direct"],
                             x.loc[q2, "indirect"] - x.loc[q1, "indirect"]))
        r = np.array(rows)
        sig = np.abs(r[:, 0]) > 1.96 * r[:, 1]
        ax.scatter(r[sig, 2], r[sig, 0], s=10, color="tab:orange", label="Δ direct")
        ax.scatter(r[sig, 3], r[sig, 0], s=10, color="tab:green", marker="s", label="Δ indirect")
        ax.axhline(0, color="0.5", lw=0.6)
        ax.axvline(0, color="0.5", lw=0.6)
        ax.set_xlabel("change of the part over the $q$ step")
        ax.set_ylabel("$D$ (significant only)")
        ax.set_title(f"{s}: increments", fontsize=6.5)
    axes[0, 0].legend(fontsize=5, ncol=2)
    axes[0, 2].legend(fontsize=5.5)
    save(fig, "step2_fig4_split.pdf")

    d[["setup", "rho", "step", "D", "D_se"]].round(4).to_csv(TAB / "step2_D.csv", index=False)
    sp.round(4).to_csv(TAB / "step2_split.csv", index=False)
    g.pivot_table(index=["setup", "rho"], columns="q", values="G").round(4).reset_index().to_csv(
        TAB / "step2_G.csv", index=False)


def step3(S3, C3):
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 4.8), sharex=True)
    for i, s in enumerate(SETUPS):
        for j, rho in enumerate((0.25, 0.75)):
            ax = axes[i, j]
            for mode in MODES:
                x = S3[(S3.setup == s) & (S3.rho == rho) & (S3["mode"] == mode)].sort_values("t")
                ax.plot(x.t, x.delta, color=MCOL[mode], lw=1, label=MLAB[mode])
                ax.fill_between(x.t, x.delta - 1.96 * x.se, x.delta + 1.96 * x.se, color=MCOL[mode], alpha=0.15, lw=0)
            ax.axhline(0, color="0.5", lw=0.6)
            ax.axvline(30, color="0.7", ls="--", lw=0.6)
            ax.set_title(f"{s}, $\\rho$ = {rho:g}")
            ax.set_ylabel("$\\delta(t)$, target A2")
            if i == 1:
                ax.set_xlabel("time $t$ (controller stops at 30)")
    axes[0, 0].legend(fontsize=6)
    save(fig, "step3_fig1_delta.pdf")

    c = C3[C3.arm == "false"]
    cs = C3[(C3.arm == "silent") & (C3["mode"] == "normal")]
    cols = [("mem_ctrl", "controller-only facts per agent"), ("cover_agent", "distinct agent-origin facts in the swarm"),
            ("board_ctrl", "readable posts with a controller-only fact")]
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.6), sharex=True)
    for i, s in enumerate(SETUPS):
        for j, (col, lab) in enumerate(cols):
            ax = axes[i, j]
            for rho, ls in ((0.25, "-"), (0.75, "--")):
                for mode in MODES:
                    x = c[(c.setup == s) & (c.rho == rho) & (c["mode"] == mode)].sort_values("t")
                    ax.plot(x.t, x[col], color=MCOL[mode], ls=ls, marker="o", ms=1.8, lw=0.9,
                            label=f"{MLAB[mode]}, $\\rho$ {rho:g}" if j == 0 and i == 0 else None)
                if col == "cover_agent":
                    x = cs[(cs.setup == s) & (cs.rho == rho)].sort_values("t")
                    ax.plot(x.t, x[col], color="tab:green", ls=ls, lw=0.9,
                            label=None if i or j != 1 else f"silent, normal, $\\rho$ {rho:g}")
            ax.set_title(f"{s}: {lab}", fontsize=6.5)
            if i == 1:
                ax.set_xlabel("time $t$")
    axes[0, 0].legend(fontsize=4.5)
    axes[0, 1].legend(fontsize=4.5)
    save(fig, "step3_fig2_carriers.pdf")


def main():
    FIG.mkdir(exist_ok=True)
    TAB.mkdir(exist_ok=True)
    G, D, C = (pd.read_parquet(IN / f"s2_{k}.parquet") for k in ("G", "D", "composition"))
    step2(G, D, C)
    step3(pd.read_parquet(IN / "s3_delta.parquet"), pd.read_parquet(IN / "s3_composition.parquet"))
    for f in ("s3_summary.csv", "s3_validation.csv", "s2_validation.csv"):
        shutil.copy(IN / f, TAB / f.replace("s3_", "step3_").replace("s2_", "step2_"))
    print("->", FIG, TAB)


if __name__ == "__main__":
    main()
