"""Figures for both reports."""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent/"10_task_and_facts"))
from engine import World
from meanfield import MeanFieldV2
from exact_game import ExactGame, GameRules, AgentPolicy, ControllerPolicy
OUT = HERE/"report"/"figures"; OUT.mkdir(parents=True, exist_ok=True)
# The setups these results were produced with (2 October single-pool designs). The keys
# are kept because saved results are named after them; the symmetric file is superseded.
SETUP_FILES = {
    "task003_symmetric_v2": HERE.parent / "50_next_experiment" / "archive" / "task003_symmetric_sharedpool_superseded.json",
    "task003_nosolution_v2": HERE.parent / "50_next_experiment" / "designs" / "task003_nosolution.json",
}
W = World((3,1,1,2,2,1,1,1,2))
SET = {"task003_symmetric_v2":"symmetric-v2","task003_nosolution_v2":"nosolution-v2"}

df = pd.read_parquet(HERE/"results"/"trajectories.parquet")

# --- Fig 1: trajectories, both rules ---------------------------------------
fig, axes = plt.subplots(2, 4, figsize=(13, 5.6), sharey=True, sharex=True)
for r, (setup, nm) in enumerate(SET.items()):
    for c, (rho, rule) in enumerate([(0.75,"lru"),(0.75,"greedy_posterior"),
                                     (1.0,"lru"),(1.0,"greedy_posterior")]):
        ax = axes[r, c]
        for arm, col in (("silent","#555"),("truth","#1f77b4"),("false","#d62728")):
            g = df[(df.setup==setup)&(df.rho==rho)&(df.rule==rule)&(df.arm==arm)]
            m = g.groupby("round_index")["e"].mean()
            s = g.groupby("round_index")["e"].std()
            ax.plot(m.index, m.values, color=col, lw=1.8, label=arm)
            ax.fill_between(m.index, m-s, m+s, color=col, alpha=.15, lw=0)
        ax.axhline(1/3, color="k", ls=":", lw=.8)
        ax.grid(alpha=.3)
        if r==0: ax.set_title(f"$\\rho$={rho:g}, {'LRU' if rule=='lru' else 'oracle'}", fontsize=10)
        if c==0: ax.set_ylabel(f"{nm}\n$e$ = mean $P(A_0|K_i)$", fontsize=9)
        if r==1: ax.set_xlabel("round")
axes[0,0].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(OUT/"fig_trajectories.pdf"); plt.close(fig)

# --- Fig 2: the LRU result -------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
for ax, rule, ttl in zip(axes, ("lru","greedy_posterior"),
                         ("LRU rule (= `deterministic` authoring)","oracle rule")):
    lab, gap = [], []
    for setup in SET:
        for rho in (0.75,1.0):
            t = df[(df.setup==setup)&(df.rho==rho)&(df.rule==rule)&(df.arm=="truth")&(df.round_index==29)]["e"].mean()
            f = df[(df.setup==setup)&(df.rho==rho)&(df.rule==rule)&(df.arm=="false")&(df.round_index==29)]["e"].mean()
            lab.append(f"{SET[setup][:4]}\n$\\rho$={rho:g}"); gap.append(t-f)
    ax.bar(range(len(gap)), gap, color=["#2ca02c" if g>.05 else "#bbb" for g in gap])
    ax.set_xticks(range(len(lab))); ax.set_xticklabels(lab, fontsize=8)
    ax.set_title(ttl, fontsize=10); ax.set_ylabel("$e$(truth) $-$ $e$(false)")
    ax.axhline(0, color="k", lw=.8); ax.set_ylim(-.05, .95); ax.grid(alpha=.3, axis="y")
fig.tight_layout(); fig.savefig(OUT/"fig_steering.pdf"); plt.close(fig)

# --- Fig 3: coarse-grained state occupancy ---------------------------------
occ = pd.read_csv(HERE/"results"/"state_occupancy.csv")
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
for ax, setup in zip(axes, SET):
    sub = occ[(occ.setup==setup)&(occ.rule=="greedy_posterior")]
    labs, bot = [], np.zeros(len(sub))
    cols = ["#d9d9d9","#9ecae1","#4292c6","#084594"]
    for k in range(4):
        v = sub[str(k)].to_numpy() if str(k) in sub else np.zeros(len(sub))
        ax.bar(range(len(sub)), v, bottom=bot, color=cols[k],
               label=f"state {k}" if setup==list(SET)[0] else None)
        bot += v
    ax.set_xticks(range(len(sub)))
    ax.set_xticklabels([f"{a}\n$\\rho$={r:g}" for a,r in zip(sub.arm, sub.rho)], fontsize=7)
    ax.set_title(SET[setup], fontsize=10)
axes[0].set_ylabel("occupancy"); axes[0].legend(frameon=False, fontsize=8, ncol=4)
fig.tight_layout(); fig.savefig(OUT/"fig_states.pdf"); plt.close(fig)

# --- Fig 4: path KL curves -------------------------------------------------
cur = pd.read_csv(HERE/"results"/"path_kl_curves.csv")
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
for ax, rule in zip(axes, ("lru","greedy_posterior")):
    for setup in SET:
        for arm, ls in (("truth","-"),("false","--")):
            g = cur[(cur.setup==setup)&(cur.rule==rule)&(cur.arm==arm)&(cur.rho==0.75)]
            if len(g): ax.plot(g.h, g.path_kl, ls, lw=1.6, label=f"{SET[setup][:4]} {arm}")
    ax.set_xlabel("horizon $h$"); ax.grid(alpha=.3)
    ax.set_title("LRU" if rule=="lru" else "oracle", fontsize=10)
axes[0].set_ylabel("path KL from silence (nats)"); axes[0].legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(OUT/"fig_pathkl.pdf"); plt.close(fig)

# --- Fig 5: theory vs simulation -------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(9, 5.6), sharex=True, sharey=True)
for r, setup in enumerate(SET):
    d = json.loads(SETUP_FILES[setup].read_text())
    asg, pool = d["agents"]["agent_assignments"], d["controller_pool"]["fact_ids"]
    for c, rho in enumerate((0.75, 1.0)):
        ax = axes[r, c]
        for arm, tgt, col in (("silent",None,"#555"),("false",2,"#d62728")):
            g = ExactGame(W, asg, pool, GameRules(rounds=30, rho=rho), AgentPolicy(),
                          ControllerPolicy(target=tgt,budget=3,gate="always",rule="greedy_posterior"))
            sim = np.mean([[x["e"] for x in g.run_episode(s,setup)] for s in range(25)],axis=0)
            th = MeanFieldV2(W, asg, pool, rho=rho, target=tgt, mc=300).run(30)
            ax.plot(sim, color=col, lw=2, label=f"sim {arm}")
            ax.plot(th, color=col, lw=1.4, ls="--", label=f"theory {arm}")
        ax.grid(alpha=.3); ax.set_title(f"{SET[setup]}, $\\rho$={rho:g}", fontsize=9)
        if r==1: ax.set_xlabel("round")
        if c==0: ax.set_ylabel("$e$ / $M$")
axes[0,0].legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(OUT/"fig_theory.pdf"); plt.close(fig)
print("figures written:", sorted(p.name for p in OUT.iterdir()))
