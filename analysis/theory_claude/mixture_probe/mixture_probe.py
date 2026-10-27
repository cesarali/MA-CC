"""Mixture probe (pre-registered in ../../theory_blackboard/2026-10-28_claude_09.md; copy in 00_preregistration_copy.md).

After the controller stops (normal continuation, controlled arm, q 6, rho 0.25/0.75, four setups), how much of the change
in the population's susceptibility to the encountered controller mixture comes from the mixture changing, and how much
from receiver contexts changing?

  S(w, M) = mean over agents of sum_f w(f) * dP(M_agent, f)
  dP(M, f) = P(A2 | M + f) - P(A2 | M)          (add-one probe; zero if f already held)
  dPdel(M, f) = P(A2 | M + f) - P(A2 | M - f)   (deletion-then-addition probe; removes redundancy)
  w_t(f) = share of controller-exclusive posts among the 48 newest posts at t that carry fact f
Decomposition from 30 to t, per episode, both orders. Reads stored data and worlds only.
Writes probe_cells.csv, probe_summary.csv.

Usage, from the repository root:
    caffeinate -i .venv/bin/python analysis/theory_claude/mixture_probe/mixture_probe.py
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
W, EPISODES = 48, 300


def cell(args):
    task, cpath = args
    run, wf, sd = RUNS[task]
    c = pathlib.Path(cpath)
    p = json.loads((c / "params.json").read_text())
    w = load_world(str(wf))
    st = load_setup(p["setup"], sd, w)
    ctrl = sorted(set(st.pools[2]) - set(st.agent_facts))
    cache = {}

    def P2(m):
        if m not in cache:
            cache[m] = w.posterior(m)[2]
        return cache[m]

    ag = pd.read_parquet(c / "agent_snapshots.parquet", columns=["episode", "grid_index", "active"],
                         filters=[("grid_index", "in", list(GI))])
    ag = ag[ag.episode < EPISODES]
    posts = pd.read_parquet(c / "posts.parquet", columns=["episode", "t", "fact"])
    posts = posts[posts.episode < EPISODES].sort_values(["episode", "t"])
    pe = {e: g for e, g in posts.groupby("episode")}
    rows = []
    for e, g in ag.groupby("episode"):
        mix, mem, expo = {}, {}, {}
        for gi, t in GI.items():
            pp = pe.get(e)
            win = pp[pp.t <= t].fact.to_numpy()[-W:] if pp is not None else np.array([], dtype=int)
            cw = [int(f) for f in win if int(f) in ctrl]
            expo[t] = len(cw) / len(win) if len(win) else 0.0
            mix[t] = None if not cw else {f: cw.count(f) / len(cw) for f in set(cw)}
            mem[t] = [int(m) for m in g[g.grid_index == gi].active.to_numpy()]

        def S(wt, Ms, deletion=False):
            tot, held = 0.0, 0.0
            for m in Ms:
                for f, wf_ in wt.items():
                    bit = 1 << f
                    if deletion:
                        tot += wf_ * (P2(m | bit) - P2(m & ~bit))
                    else:
                        tot += wf_ * (P2(m | bit) - P2(m))
                    held += wf_ * bool(m & bit)
            return tot / len(Ms), held / len(Ms)

        for t in (40.0, 80.0, 130.0):
            row = dict(task=task, setup=p["setup"].split("_", 1)[1], rho=float(p["rho"]), episode=int(e), t=t,
                       expo30=expo[30.0], expo_t=expo[t], avail=mix[30.0] is not None and mix[t] is not None)
            if row["avail"]:
                for name, dl in (("", False), ("del_", True)):
                    s_tt, h_tt = S(mix[t], mem[t], dl)
                    s_30t, _ = S(mix[30.0], mem[t], dl)
                    s_3030, h_3030 = S(mix[30.0], mem[30.0], dl)
                    s_t30, _ = S(mix[t], mem[30.0], dl)
                    row.update({f"{name}S_t": s_tt, f"{name}S_30": s_3030,
                                f"{name}mix_term": s_tt - s_30t, f"{name}ctx_term": s_30t - s_3030,
                                f"{name}mix_term_alt": s_t30 - s_3030, f"{name}ctx_term_alt": s_tt - s_t30})
                    if not dl:
                        row.update(held_t=h_tt, held_30=h_3030)
            rows.append(row)
    return rows


def main():
    jobs = []
    for task, (run, _, _) in RUNS.items():
        for c in sorted((run / "cells").iterdir()):
            p = json.loads((c / "params.json").read_text())
            if p["post_stop_mode"] == "normal" and p["arm"] == "false" and int(p["q"]) == 6:
                jobs.append((task, str(c)))
    with multiprocessing.Pool(7) as pool:
        parts = pool.map(cell, jobs, chunksize=1)
    d = pd.DataFrame([r for part in parts for r in part])
    d.to_csv(HERE / "probe_cells.csv", index=False)
    av = d[d.avail]
    s = av.groupby(["task", "setup", "rho", "t"]).agg(
        episodes=("episode", "count"), S_30=("S_30", "mean"), S_t=("S_t", "mean"), mix_term=("mix_term", "mean"),
        ctx_term=("ctx_term", "mean"), mix_term_alt=("mix_term_alt", "mean"), ctx_term_alt=("ctx_term_alt", "mean"),
        mix_se=("mix_term", lambda x: x.std() / np.sqrt(len(x))), del_S_30=("del_S_30", "mean"), del_S_t=("del_S_t", "mean"),
        del_mix_term=("del_mix_term", "mean"), del_ctx_term=("del_ctx_term", "mean"), held_30=("held_30", "mean"),
        held_t=("held_t", "mean"), expo30=("expo30", "mean"), expo_t=("expo_t", "mean")).reset_index()
    unav = d.groupby(["task", "setup", "rho", "t"]).avail.apply(lambda x: 1 - x.mean()).rename("unavailable_share")
    s = s.merge(unav.reset_index(), on=["task", "setup", "rho", "t"])
    s["dS"] = s.S_t - s.S_30
    s["mix_share"] = s.mix_term.abs() / s.dS.abs().where(s.dS.abs() > 0)
    s.to_csv(HERE / "probe_summary.csv", index=False)
    pd.set_option("display.width", 260)
    print(s[s.t == 130.0][["task", "setup", "rho", "episodes", "unavailable_share", "expo30", "expo_t", "S_30", "S_t", "dS",
                           "mix_term", "mix_se", "ctx_term", "mix_term_alt", "ctx_term_alt", "del_mix_term", "del_ctx_term",
                           "held_30", "held_t"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
