"""Phase C2, revision 1 (00_preregistration.md, change log): N2R, neutral drift in two reservoirs.

From the stored state at t = 30 (memories, last votes, board with authors), each episode is continued to t = 130 with
the simulator's rules (normal continuation, no controller): 24 agents act at rate 1, read q posts from the 48 newest
posts not their own, renew read facts, forget each fact at rate gamma, vote by probability matching, and post one fact.
The one change: the posted fact is drawn uniformly from the agent's memory (preferring facts not just read), not from
the facts that support its vote. With --support the simulator's own rule (support in context, fallback, prefer not
read) is used instead: a check of this code path, which should reproduce the data.

Outputs (tag n2r or support): <tag>_delta.csv, <tag>_summary.csv (same checks as neutral_board_drift.py).

Usage, from the repository root:
    caffeinate -i .venv/bin/python analysis/theory_claude/phaseC/two_reservoir_drift.py --workers 2 [--support]
"""
from __future__ import annotations
import argparse, heapq, json, math, multiprocessing, pathlib, random, sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
A = HERE.parents[1]
sys.path.insert(0, str(A / "theory_claude/phaseA"))
sys.path.insert(0, str(A / "task003_llm_free/simulation_2/simulator"))
from evidence_map import facts  # noqa: E402
from solvable_branches import RUNS, TIMES, W, data_delta, bits  # noqa: E402
from sim2.simulation_2 import choose_post, NO_POST  # noqa: E402

DIV = A / "theory_claude/reports/2026-10-23_task_comparison/ctrl_diversity.csv"
CHECK_T = (40.0, 80.0, 130.0)


def run_arm(task, setup, rho, q, cell, seed, support):
    w, agent_facts, ctrl = facts(task, setup)
    ctrl = set(ctrl)
    gamma = -math.log(rho)
    rng = np.random.default_rng(seed)
    prng = random.Random(seed)
    ag = pd.read_parquet(cell / "agent_snapshots.parquet", columns=["episode", "grid_index", "agent", "active", "last_vote"],
                         filters=[("grid_index", "==", 120)])
    posts = pd.read_parquet(cell / "posts.parquet", columns=["episode", "t", "author", "fact"]).sort_values(["episode", "t"])
    pe = {e: g for e, g in posts[posts.t <= 30.0].groupby("episode")}
    cache = {}

    def P(m):
        if m not in cache:
            cache[m] = w.posterior(m)
        return cache[m]

    x = np.zeros(len(TIMES))
    n_vote = np.zeros(len(TIMES))
    variety = {t: [] for t in CHECK_T}
    for e, g in ag.groupby("episode"):
        board = [(int(a), int(f)) for a, f in zip(pe[e].author, pe[e].fact)] if e in pe else []
        agents = g.agent.to_numpy()
        mem = {int(a): {f: 30.0 + rng.exponential(1 / gamma) for f in bits(int(m))} for a, m in zip(agents, g.active)}
        vote = {int(a): int(v) for a, v in zip(agents, g.last_vote)}
        heap = [(30.0 + rng.exponential(1.0), int(a)) for a in agents]
        heapq.heapify(heap)
        ti, ci = 0, 0
        while heap:
            t, a = heapq.heappop(heap)
            while ti < len(TIMES) and TIMES[ti] < t:
                vs = np.array([v for v in vote.values() if v >= 0])
                x[ti] += (vs == 2).mean() if len(vs) else 0.0
                n_vote[ti] += 1
                ti += 1
            while ci < len(CHECK_T) and CHECK_T[ci] < t:
                win = [f for _, f in board[-W:]]
                variety[CHECK_T[ci]].append(len(set(win) & ctrl))
                ci += 1
            if t > 130.0:
                break
            heapq.heappush(heap, (t + rng.exponential(1.0), a))
            m = {f: d for f, d in mem[a].items() if d > t}
            win = [f for au, f in board if au != a][-W:]
            read = set()
            if win:
                for i in rng.choice(len(win), size=min(q, len(win)), replace=False):
                    read.add(win[i])
            for f in read:
                m[f] = t + rng.exponential(1 / gamma)
            mem[a] = m
            mm = 0
            for f in m:
                mm |= 1 << f
            p = P(mm)
            v = int(rng.choice(3, p=np.asarray(p) / sum(p)))
            vote[a] = v
            if support:
                rmask = 0
                for f in read:
                    rmask |= 1 << f
                f = choose_post(w, mm, v, rmask, True, prng)[0]
                if f != NO_POST:
                    board.append((a, int(f)))
            elif m:
                fresh = [f for f in m if f not in read]
                cand = fresh if fresh else list(m)
                board.append((a, int(cand[int(rng.integers(len(cand)))])))
    return x / np.maximum(n_vote, 1), {t: float(np.mean(v)) for t, v in variety.items()}


def job(args):
    task, setup, rho, support = args
    cells = {}
    for c in (RUNS[task] / "cells").iterdir():
        p = json.loads((c / "params.json").read_text())
        if p["setup"] == setup and float(p["rho"]) == rho and p["post_stop_mode"] == "normal":
            cells[p["arm"]] = c
    q = int(json.loads((cells["false"] / "params.json").read_text())["q"])
    res = {arm: run_arm(task, setup, rho, q, c, 31 if arm == "false" else 32, support) for arm, c in cells.items()}
    dm, dse = data_delta(cells)
    d = pd.DataFrame(dict(task=task, setup=setup.split("_", 1)[1], rho=rho, t=TIMES,
                          model=res["false"][0] - res["silent"][0], data=dm.reindex(TIMES).to_numpy(),
                          data_se=dse.reindex(TIMES).to_numpy(), data_30=dm.get(30.0, np.nan)))
    var = dict(task=task, setup=setup.split("_", 1)[1], rho=rho, **{f"model_distinct_{int(t)}": v for t, v in res["false"][1].items()})
    print("done", task, setup, rho, flush=True)
    return d, var


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--support", action="store_true")
    a = ap.parse_args()
    tag = "support" if a.support else "n2r"
    jobs = [(t, s, r, a.support) for t, ss in (("task003", ["task003_symmetric", "task003_nosolution"]),
                                               ("Object Placements", ["op_symmetric", "op_nosolution"]))
            for s in ss for r in (0.25, 0.75)]
    with multiprocessing.Pool(a.workers) as pool:
        res = pool.map(job, jobs, chunksize=1)
    d = pd.concat([r[0] for r in res])
    d.round(5).to_csv(HERE / f"{tag}_delta.csv", index=False)
    v = pd.DataFrame([r[1] for r in res])
    div = pd.read_csv(DIV).pivot_table(index=["task", "setup", "rho"], columns="t", values="distinct_ctrl_board")
    rows = []
    for (task, setup, rho), g in d.groupby(["task", "setup", "rho"]):
        g = g.set_index("t")
        d30 = g.data_30.iloc[0]
        vv = v[(v.task == task) & (v.setup == setup) & (v.rho == rho)].iloc[0]
        row = dict(task=task, setup=setup, rho=rho, data_30=d30, model_31=g.model[31.0], data_130=g.data[130.0],
                   model_130=g.model[130.0], data_retained=g.data[130.0] / d30, model_retained=g.model[130.0] / d30,
                   scored=not (task == "Object Placements" and setup == "symmetric" and rho == 0.75))
        for t in (80.0, 130.0):
            row[f"distinct_data_{int(t)}"] = div.loc[(task, setup, rho), t]
            row[f"distinct_model_{int(t)}"] = vv[f"model_distinct_{int(t)}"]
        row["variety_ok"] = all(abs(row[f"distinct_model_{t}"] / row[f"distinct_data_{t}"] - 1) <= 0.25 for t in (80, 130))
        row["persistence_ok"] = abs(row["model_retained"] - row["data_retained"]) <= 0.2
        rows.append(row)
    s = pd.DataFrame(rows)
    s.round(4).to_csv(HERE / f"{tag}_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(s.round(3).to_string(index=False))
    sc = s[s.scored]
    print(f"{tag}: variety ok {int(sc.variety_ok.sum())}/{len(sc)}; persistence ok {int(sc.persistence_ok.sum())}/{len(sc)}")


if __name__ == "__main__":
    main()
