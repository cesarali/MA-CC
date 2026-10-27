"""Phase A4 (00_preregistration.md): the no-communication and frozen-board branches predicted from the state at t = 30.

Given the t = 30 state (memories, last votes, board), agents in these branches are independent of each other.
  no_comm   exact: each fact of M_i(30) survives to t with probability pi = exp(-gamma (t - 30)); the agent's vote at t is
            its t = 30 vote with probability exp(-(t - 30)), otherwise drawn at its last action time t - u
            (density exp(-u)) from P(. | memory then). E[P(A2 | kept subset)] = sum_j pi^j (1 - pi)^(k - j) A_j, with
            A_j the sum of P(A2 | S) over subsets S of size j (exact enumeration); the u integral on a 400-point grid.
  frozen    each agent is simulated as an independent Markov chain with the simulator's rules (rate-1 actions, read q
            posts without replacement from its frozen window = the 48 newest posts at t = 30 not its own, renew read
            facts, forget each fact at rate gamma, vote by probability matching). One chain per agent and episode.
  check     frozen board, analytic: each fact's long-run holding share r_f / (r_f + gamma), r_f = 1 - C(W_i - k_f, q) /
            C(W_i, q), against the simulated chains and the data at t = 130.
Output: delta(t) = mean x_A2(false) - mean x_A2(silent) at t = 30.25 ... 130 for model and data, written to
solvable_branches.csv; summary at t = 31, 40, 80, 130 and the no-communication half-time printed.

Usage, from the repository root:
    caffeinate -i .venv/bin/python analysis/theory_claude/phaseA/solvable_branches.py --workers 7
"""
from __future__ import annotations
import argparse, itertools, json, math, multiprocessing, pathlib, sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from evidence_map import facts  # noqa: E402

A = HERE.parents[1]
RUNS = {"task003": A / "task003_llm_free/results/simulation_2/2026-10-20_sim2_persistence_branches",
        "Object Placements": A / "object_placements_llm_free/results/simulation_2/2026-10-23_sim2_op_post_stop"}
W, GRID = 48, 0.25
TIMES = np.round(np.arange(30.25, 130.0001, GRID), 6)


def bits(m):
    out, i = [], 0
    while m:
        if m & 1:
            out.append(i)
        m >>= 1
        i += 1
    return out


def state_at_30(cell):
    ag = pd.read_parquet(cell / "agent_snapshots.parquet", columns=["episode", "grid_index", "agent", "active", "last_vote"],
                         filters=[("grid_index", "==", 120)])
    return ag


def data_delta(cells):
    out = {}
    for arm, c in cells.items():
        s = pd.read_parquet(c / "snapshots.parquet", columns=["episode", "t", "share_A2", "share_voted"])
        s = s[s.t >= 30.0]
        s = s.assign(x=s.share_A2 / s.share_voted.where(s.share_voted > 0))
        out[arm] = s.pivot_table(index="episode", columns="t", values="x")
    d = out["false"] - out["silent"]
    return d.mean(axis=0), d.std(axis=0) / np.sqrt(d.count(axis=0))


def no_comm(task, setup, rho, cell):
    w, _, _ = facts(task, setup)
    gamma = -math.log(rho)
    ag = state_at_30(cell)
    cache, pcache = {}, {}

    def post2(mm):
        if mm not in pcache:
            pcache[mm] = w.posterior(mm)[2]
        return pcache[mm]

    def coeffs(m):
        if m not in cache:
            fs = bits(m)
            k = len(fs)
            Aj = np.zeros(k + 1)
            for j in range(k + 1):
                for S in itertools.combinations(fs, j):
                    mm = 0
                    for f in S:
                        mm |= 1 << f
                    Aj[j] += post2(mm)
            cache[m] = (k, Aj)
        return cache[m]

    # one curve per distinct (memory, last vote): E[x_A2](t) = exp(-s) 1[v0 = A2] + int_0^s exp(-u) Ep(s - u) du, s = t - 30,
    # with Ep(tau) = sum_j pi(tau)^j (1 - pi(tau))^(k - j) A_j, pi(tau) = exp(-gamma tau); the integral is a convolution on
    # a fine uniform grid (step 0.01), read off at the report times
    du = 0.01
    tau = np.arange(0, 100.0 + du / 2, du)
    ker = np.exp(-tau)
    idx = np.round((TIMES - 30.0) / du).astype(int)
    x = np.zeros(len(TIMES))
    n = 0
    groups = pd.DataFrame(dict(m=ag.active.astype("int64"), v0=ag.last_vote)).query("v0 >= 0").value_counts()
    for (m, v0), cnt in groups.items():
        k, Aj = coeffs(int(m))
        pi = np.exp(-gamma * tau)
        Ep = sum(pi ** j * (1 - pi) ** (k - j) * Aj[j] for j in range(k + 1))
        integ = np.convolve(ker, Ep)[: len(tau)] * du - 0.5 * du * (ker[0] * Ep + ker * Ep[0])   # trapezoid correction
        x += cnt * (np.exp(-(TIMES - 30.0)) * (v0 == 2) + integ[idx])
        n += cnt
    return x / n


def frozen(task, setup, rho, q, cell, seed):
    w, _, _ = facts(task, setup)
    gamma = -math.log(rho)
    rng = np.random.default_rng(seed)
    ag = state_at_30(cell)
    posts = pd.read_parquet(cell / "posts.parquet", columns=["episode", "t", "author", "fact"])
    posts = posts[posts.t <= 30.0].sort_values(["episode", "t"])
    pe = {e: g for e, g in posts.groupby("episode")}
    cache = {}

    def P(m):
        if m not in cache:
            cache[m] = w.posterior(m)
        return cache[m]

    x = np.zeros(len(TIMES))
    hold_obs = {}
    n = 0
    for e, g in ag.groupby("episode"):
        pp = pe.get(e)
        for a, m, v0 in zip(g.agent.to_numpy(), g.active.to_numpy(), g.last_vote.to_numpy()):
            win = pp[pp.author != a].fact.to_numpy()[-W:] if pp is not None else np.array([], dtype=int)
            mem = {f: 30.0 + rng.exponential(1 / gamma) for f in bits(int(m))}
            t, vote, ti = 30.0, int(v0), 0
            while True:
                t_next = t + rng.exponential(1.0)
                while ti < len(TIMES) and TIMES[ti] < t_next:
                    x[ti] += vote == 2
                    ti += 1
                if ti >= len(TIMES):
                    break
                t = t_next
                mem = {f: d for f, d in mem.items() if d > t}
                if len(win):
                    for f in win[rng.choice(len(win), size=min(q, len(win)), replace=False)]:
                        mem[int(f)] = t + rng.exponential(1 / gamma)
                mm = 0
                for f in mem:
                    mm |= 1 << f
                p = P(mm)
                vote = int(rng.choice(3, p=np.asarray(p) / sum(p)))
            for f, d in mem.items():
                if d > 130.0:                                   # still held at t = 130
                    hold_obs[f] = hold_obs.get(f, 0) + 1
            n += 1
    return x / n, {f: c / n for f, c in hold_obs.items()}


def job(args):
    task, setup, rho, mode = args
    run = RUNS[task]
    cells = {}
    for c in (run / "cells").iterdir():
        p = json.loads((c / "params.json").read_text())
        if p["setup"] == setup and float(p["rho"]) == rho and p["post_stop_mode"] == mode:
            cells[p["arm"]] = c
    q = int(json.loads((cells["false"] / "params.json").read_text())["q"])
    if mode == "no_comm":
        model = {arm: no_comm(task, setup, rho, c) for arm, c in cells.items()}
        hold = None
    else:
        res = {arm: frozen(task, setup, rho, q, c, seed=11 if arm == "false" else 12) for arm, c in cells.items()}
        model = {arm: r[0] for arm, r in res.items()}
        hold = res["false"][1]
    dm, dse = data_delta(cells)
    print('done', task, setup, rho, mode, flush=True)
    out = pd.DataFrame(dict(task=task, setup=setup.split("_", 1)[1], rho=rho, mode=mode, t=TIMES,
                            model=model["false"] - model["silent"], data=dm.reindex(TIMES).to_numpy(),
                            data_se=dse.reindex(TIMES).to_numpy()))
    d30 = dm.get(30.0, np.nan)
    out["data_30"] = d30
    return out, (task, setup, rho, mode, hold)


def analytic_hold(task, setup, rho, q, cell):
    """Long-run holding share of each fact on a frozen board, r_f / (r_f + gamma), averaged over agents (false arm)."""
    gamma = -math.log(rho)
    ag = state_at_30(cell)
    posts = pd.read_parquet(cell / "posts.parquet", columns=["episode", "t", "author", "fact"])
    posts = posts[posts.t <= 30.0].sort_values(["episode", "t"])
    pe = {e: g for e, g in posts.groupby("episode")}
    acc, n = {}, 0
    for e, g in ag.groupby("episode"):
        pp = pe.get(e)
        for a in g.agent.to_numpy():
            win = pp[pp.author != a].fact.to_numpy()[-W:]
            Wi = len(win)
            vals, cnt = np.unique(win, return_counts=True)
            for f, k in zip(vals, cnt):
                r = 1 - math.comb(Wi - k, min(q, Wi)) / math.comb(Wi, min(q, Wi))
                acc[int(f)] = acc.get(int(f), 0.0) + r / (r + gamma)
            n += 1
    return {f: v / n for f, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=7)
    a = ap.parse_args()
    jobs = [(task, setup, rho, mode) for task, setups in (("task003", ["task003_symmetric", "task003_nosolution"]),
                                                         ("Object Placements", ["op_symmetric", "op_nosolution"]))
            for setup in setups for rho in (0.25, 0.75) for mode in ("no_comm", "frozen_board")]
    with multiprocessing.Pool(a.workers) as pool:
        res = pool.map(job, jobs, chunksize=1)
    out = pd.concat([r[0] for r in res])
    out.round(5).to_csv(HERE / "solvable_branches.csv", index=False)
    rows = []
    for (task, setup, rho, mode), g in out.groupby(["task", "setup", "rho", "mode"]):
        g = g.set_index("t")
        row = dict(task=task, setup=setup, rho=rho, mode=mode, data_30=g.data_30.iloc[0])
        for t in (31.0, 40.0, 80.0, 130.0):
            row[f"model_{int(t)}"], row[f"data_{int(t)}"], row[f"se_{int(t)}"] = g.model[t], g.data[t], g.data_se[t]
        if mode == "no_comm":
            for col in ("model", "data"):
                below = g[g[col] <= row["data_30"] / 2]
                row[f"half_{col}"] = float(below.index[0] - 30.0) if len(below) else np.nan
        z = ((g.model - g.data) / g.data_se.where(g.data_se > 0)).abs().dropna()   # times where the data vary
        row["max_abs_z"] = float(z.max())
        row["share_within_2se"] = float((z <= 2).mean())
        row["times_scored"] = int(len(z))
        row["max_abs_diff"] = float((g.model - g.data).abs().max())
        rows.append(row)
    s = pd.DataFrame(rows)
    s.round(4).to_csv(HERE / "solvable_branches_summary.csv", index=False)
    pd.set_option("display.width", 260)
    print(s.round(3).to_string(index=False))
    # analytic holding check (false arm, frozen board): chains and data at t = 130
    chk = []
    for (task, setup, rho, mode, hold) in [r[1] for r in res]:
        if mode != "frozen_board":
            continue
        run = RUNS[task]
        cell = [c for c in (run / "cells").iterdir() if json.loads((c / "params.json").read_text()) | {} and
                json.loads((c / "params.json").read_text())["setup"] == setup and
                float(json.loads((c / "params.json").read_text())["rho"]) == rho and
                json.loads((c / "params.json").read_text())["post_stop_mode"] == "frozen_board" and
                json.loads((c / "params.json").read_text())["arm"] == "false"][0]
        an = analytic_hold(task, setup, rho, 6, cell)
        ag = pd.read_parquet(cell / "agent_snapshots.parquet", columns=["grid_index", "active"],
                             filters=[("grid_index", "==", 520)])
        data_hold = {f: float(np.mean([(int(m) >> f) & 1 for m in ag.active.to_numpy()])) for f in an}
        for f in an:
            chk.append(dict(task=task, setup=setup, rho=rho, fact=f, analytic=an[f], chains=hold.get(f, 0.0), data=data_hold[f]))
    c = pd.DataFrame(chk)
    c.round(4).to_csv(HERE / "frozen_holding_check.csv", index=False)
    print(c.groupby(["task", "setup", "rho"]).apply(
        lambda g: pd.Series(dict(facts=len(g), mae_analytic_vs_data=(g.analytic - g.data).abs().mean(),
                                 mae_chains_vs_data=(g.chains - g.data).abs().mean())), include_groups=False).round(4).to_string())


if __name__ == "__main__":
    main()
