"""Fading chain (pre-registered in ../../theory_blackboard/2026-10-28_claude_12.md, step 1).

When the controller's effect fades after the stop (Object Placements) or does not (task003), which link of the chain
exposure -> distinct controller facts held -> operating point -> log-odds strength of encountered facts moves?

Stored normal-continuation runs, q 6, rho 0.25/0.75, four setups, controlled arm ("false") and silent arm,
t in {30, 40, 80, 130}, first 300 episodes per cell. The exact posterior is used only to measure stored memories.

Per episode and time:
  (a) b_C: share of the 48 newest posts carrying a controller-exclusive fact; F_C^board, F_C^swarm: distinct
      controller-exclusive facts on the board / on the board or in any memory; M_C: distinct controller-exclusive facts
      per agent; M_A: native facts per agent (both arms)
  (b) x = logit P(A2 | M), clipped to +-12 (controlled and silent arms)
  (c) theta = logit P(A2 | M + f) - logit P(A2 | M) for encountered controller-exclusive f not held, weighted by w_t(f):
      share positive, mean positive theta, mean |theta| among negative, share of (M, f) pairs already held
  (d) S(t) (add-one probability increment, as in the mixture probe) and the counterfactual
      S_theta30(t) = E_{M ~ mu_t} E_{theta ~ nu_30}[sigma(x + theta) - sigma(x)]   (primary, as pre-registered)
      and a held-adjusted secondary version that multiplies by the share of encountered pairs not already held at t
  (e) Z(t) = E_episode[b_C(t) S(t)], 0 when no controller post is on the board (unconditional)
Writes chain_episodes.csv and chain_summary.csv.

Usage, from the repository root:
    caffeinate -i .venv/bin/python analysis/theory_claude/fading_chain/fading_chain.py
"""
from __future__ import annotations
import json, multiprocessing, pathlib, sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
A = HERE.parents[1]
sys.path.insert(0, str(A / "task003_llm_free"))
from llmfree_core.kworld import load_world  # noqa: E402
from llmfree_core.setups import load_setup  # noqa: E402

RUNS = {"task003": (A / "task003_llm_free/results/simulation_2/2026-10-20_sim2_persistence_branches",
                    A / "task003_llm_free/llmfree_core/world_task003.json", A / "task003_llm_free/experimental_setup"),
        "Object Placements": (A / "object_placements_llm_free/results/simulation_2/2026-10-23_sim2_op_post_stop",
                              A / "object_placements_llm_free/world/world_op1.json",
                              A / "object_placements_llm_free/experimental_setup")}
GI = {120: 30.0, 160: 40.0, 320: 80.0, 520: 130.0}
W, EPISODES, XCLIP = 48, 300, 12.0
TH_BINS = np.linspace(-2 * XCLIP, 2 * XCLIP, 481)
TH_MID = 0.5 * (TH_BINS[1:] + TH_BINS[:-1])


def sigma(x):
    return 1.0 / (1.0 + np.exp(-x))


def logit(p):
    p = min(max(p, 1e-300), 1 - 1e-16)
    return float(np.clip(np.log(p) - np.log1p(-p), -XCLIP, XCLIP))


def popcount(m, mask):
    return bin(m & mask).count("1")


def load(cpath):
    ag = pd.read_parquet(cpath / "agent_snapshots.parquet", columns=["episode", "grid_index", "active"],
                         filters=[("grid_index", "in", list(GI))])
    ag = ag[ag.episode < EPISODES]
    posts = pd.read_parquet(cpath / "posts.parquet", columns=["episode", "t", "fact"])
    posts = posts[posts.episode < EPISODES].sort_values(["episode", "t"])
    mem = {(int(e), GI[int(gi)]): [int(m) for m in g.active.to_numpy()] for (e, gi), g in ag.groupby(["episode", "grid_index"])}
    return mem, {int(e): g.fact.to_numpy() for e, g in posts.groupby("episode")}, {int(e): g.t.to_numpy() for e, g in posts.groupby("episode")}


def cell(args):
    task, ccell, scell = args
    run, wf, sd = RUNS[task]
    p = json.loads((pathlib.Path(ccell) / "params.json").read_text())
    w = load_world(str(wf))
    st = load_setup(p["setup"], sd, w)
    ctrl = sorted(set(st.pools[2]) - set(st.agent_facts))
    cmask = sum(1 << f for f in ctrl)
    nmask = sum(1 << f for f in set(st.agent_facts))
    cache = {}

    def P2(m):
        if m not in cache:
            cache[m] = w.posterior(m)[2]
        return cache[m]

    cmem, cfact, ctime = load(pathlib.Path(ccell))
    smem, _, _ = load(pathlib.Path(scell))
    rows, nu30 = [], np.zeros(len(TH_MID))
    per_ep = {}
    for e in sorted({k[0] for k in cmem}):
        for t in GI.values():
            Ms = cmem.get((e, t))
            if Ms is None:
                continue
            f_all = cfact.get(e, np.array([], dtype=int))
            t_all = ctime.get(e, np.array([]))
            win = f_all[t_all <= t][-W:]
            cw = [int(f) for f in win if int(f) in ctrl]
            bC = len(cw) / len(win) if len(win) else 0.0
            mix = {f: cw.count(f) / len(cw) for f in set(cw)} if cw else None
            swarm = set(cw)
            for m in Ms:
                swarm |= {f for f in ctrl if m >> f & 1}
            xs = [logit(P2(m)) for m in Ms]
            sm = smem.get((e, t), [])
            row = dict(task=task, setup=p["setup"].split("_", 1)[1], rho=float(p["rho"]), episode=e, t=t, b_C=bC,
                       F_C_board=len(set(cw)), F_C_swarm=len(swarm), M_C=np.mean([popcount(m, cmask) for m in Ms]),
                       M_A_c=np.mean([popcount(m, nmask) for m in Ms]),
                       M_A_s=np.mean([popcount(m, nmask) for m in sm]) if sm else np.nan,
                       x_c=np.mean(xs), x_c_sd=np.std(xs), P2_c=np.mean([P2(m) for m in Ms]),
                       x_s=np.mean([logit(P2(m)) for m in sm]) if sm else np.nan,
                       P2_s=np.mean([P2(m) for m in sm]) if sm else np.nan, avail=mix is not None)
            if mix is not None:
                S, held, wpos, wneg, sth_pos, sth_neg = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                hist = np.zeros(len(TH_MID))
                for m, x in zip(Ms, xs):
                    for f, wf_ in mix.items():
                        bit = 1 << f
                        if m & bit:
                            held += wf_
                            continue
                        S += wf_ * (P2(m | bit) - P2(m))
                        th = logit(P2(m | bit)) - x
                        if th > 0:
                            wpos += wf_; sth_pos += wf_ * th
                        elif th < 0:
                            wneg += wf_; sth_neg += wf_ * (-th)
                        hist[min(max(np.searchsorted(TH_BINS, th) - 1, 0), len(TH_MID) - 1)] += wf_
                n = len(Ms)
                notheld = hist.sum()                # total weight of encountered pairs not already held
                row.update(S=S / n, held=held / n, share_pos=wpos / notheld if notheld else np.nan,
                           theta_pos=sth_pos / wpos if wpos else np.nan, theta_neg=sth_neg / wneg if wneg else np.nan,
                           Z=bC * S / n)
                if t == 30.0 and notheld:
                    nu30 += hist / notheld          # each episode contributes equally to the pooled nu_30
            else:
                row.update(S=np.nan, held=np.nan, share_pos=np.nan, theta_pos=np.nan, theta_neg=np.nan, Z=0.0)
            per_ep[(e, t)] = xs
            rows.append(row)
    nu30 = nu30 / nu30.sum()
    for r in rows:                             # (d) operating-point counterfactual with the pooled t = 30 strengths
        xs = np.asarray(per_ep[(r["episode"], r["t"])])
        resp = (sigma(xs[:, None] + TH_MID[None, :]) - sigma(xs[:, None])) @ nu30
        r["S_theta30"] = float(resp.mean())
        r["S_theta30_held"] = float(resp.mean() * (1 - r["held"])) if r["avail"] else np.nan
    return rows


def main():
    jobs = []
    for task, (run, _, _) in RUNS.items():
        cells = {}
        for c in sorted((run / "cells").iterdir()):
            p = json.loads((c / "params.json").read_text())
            if p["post_stop_mode"] == "normal" and int(p["q"]) == 6:
                cells[(p["setup"], p["rho"], p["arm"])] = str(c)
        for (setup, rho, arm), c in cells.items():
            if arm == "false":
                jobs.append((task, c, cells[(setup, rho, "silent")]))
    with multiprocessing.Pool(min(8, len(jobs))) as pool:
        parts = pool.map(cell, jobs, chunksize=1)
    d = pd.DataFrame([r for part in parts for r in part])
    d.to_csv(HERE / "chain_episodes.csv", index=False)
    key = ["task", "setup", "rho", "t"]
    allep = d.groupby(key).agg(episodes=("episode", "count"), b_C=("b_C", "mean"), F_C_board=("F_C_board", "mean"),
                               F_C_swarm=("F_C_swarm", "mean"), M_C=("M_C", "mean"), M_A_c=("M_A_c", "mean"),
                               M_A_s=("M_A_s", "mean"), x_c=("x_c", "mean"), x_s=("x_s", "mean"), P2_c=("P2_c", "mean"),
                               P2_s=("P2_s", "mean"), Z=("Z", "mean"), S_theta30=("S_theta30", "mean"),
                               avail_share=("avail", "mean")).reset_index()
    cond = d[d.avail].groupby(key).agg(S_cond=("S", "mean"), held_cond=("held", "mean"), share_pos_cond=("share_pos", "mean"),
                                       theta_pos_cond=("theta_pos", "mean"), theta_neg_cond=("theta_neg", "mean"),
                                       S_theta30_cond=("S_theta30", "mean"), S_theta30_held_cond=("S_theta30_held", "mean"),
                                       Z_check=("Z", "mean")).reset_index()
    s = allep.merge(cond, on=key, how="left")
    s["delta"] = s.P2_c - s.P2_s
    s.to_csv(HERE / "chain_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(s.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
