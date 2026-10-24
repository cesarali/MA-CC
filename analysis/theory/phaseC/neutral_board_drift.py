"""Phase C2 (00_preregistration.md): neutral board drift (NBD), a board-only model of persistence under normal
continuation, with no fitted parameter.

From t = 30, per episode and arm: the board starts as the stored board (the W = 48 newest posts at t = 30). New posts
arrive at rate nu (the agents' posting rate measured in this cell for 20 <= t <= 30), each carrying the fact of a post
drawn uniformly from the current window (neutral copying). Agents start from their stored memory and last vote at
t = 30 and follow the exact single-agent chain of Phase A4 on the drifting board (rate-1 actions, read q posts,
forget at rate gamma, probability-matching votes); their posts do not feed the board.

Outputs: nbd_delta.csv (delta(t) model and data, t = 30.25 ... 130), nbd_variety.csv (distinct controller-only facts per
board at t = 40, 80, 130, model), nbd_summary.csv (retained fraction delta(130)/delta(30), variety against
../reports/2026-10-23_task_comparison/ctrl_diversity.csv, and the pre-registered checks).

Usage, from the repository root:
    caffeinate -i .venv/bin/python analysis/theory/phaseC/neutral_board_drift.py --workers 2
"""
from __future__ import annotations
import argparse, bisect, json, math, multiprocessing, pathlib, sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
A = HERE.parents[1]
sys.path.insert(0, str(A / "theory/phaseA"))
from evidence_map import facts  # noqa: E402
from solvable_branches import RUNS, TIMES, W, data_delta, bits  # noqa: E402

DIV = A / "theory/reports/2026-10-23_task_comparison/ctrl_diversity.csv"
CHECK_T = (40.0, 80.0, 130.0)


def run_arm(task, setup, rho, q, cell, seed):
    w, agent, ctrl = facts(task, setup)
    ctrl = set(ctrl)
    gamma = -math.log(rho)
    rng = np.random.default_rng(seed)
    ag = pd.read_parquet(cell / "agent_snapshots.parquet", columns=["episode", "grid_index", "agent", "active", "last_vote"],
                         filters=[("grid_index", "==", 120)])
    posts = pd.read_parquet(cell / "posts.parquet", columns=["episode", "t", "author", "fact"]).sort_values(["episode", "t"])
    pre = posts[(posts.t >= 20.0) & (posts.t <= 30.0) & (posts.author >= 0)]
    nu = len(pre) / 10.0 / posts.episode.nunique()
    pe = {e: g for e, g in posts[posts.t <= 30.0].groupby("episode")}
    cache = {}

    def P(m):
        if m not in cache:
            cache[m] = w.posterior(m)
        return cache[m]

    x = np.zeros(len(TIMES))
    n = 0
    variety = {t: [] for t in CHECK_T}
    for e, g in ag.groupby("episode"):
        win = list(pe[e].fact.to_numpy()[-W:]) if e in pe else []
        # neutral drift of the board from 30 to 130: event times and the window after each event
        times, wins = [30.0], [np.array(win, dtype=int)]
        t = 30.0
        cur = list(win)
        while len(cur):
            t += rng.exponential(1 / nu)
            if t > 130.0:
                break
            cur.append(cur[int(rng.integers(len(cur)))])
            if len(cur) > W:
                cur.pop(0)
            times.append(t)
            wins.append(np.array(cur, dtype=int))
        for tc in CHECK_T:
            wv = wins[bisect.bisect_right(times, tc) - 1]
            variety[tc].append(len(set(wv.tolist()) & ctrl))
        for m, v0 in zip(g.active.to_numpy(), g.last_vote.to_numpy()):
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
                wv = wins[bisect.bisect_right(times, t) - 1]
                if len(wv):
                    for f in wv[rng.choice(len(wv), size=min(q, len(wv)), replace=False)]:
                        mem[int(f)] = t + rng.exponential(1 / gamma)
                mm = 0
                for f in mem:
                    mm |= 1 << f
                p = P(mm)
                vote = int(rng.choice(3, p=np.asarray(p) / sum(p)))
            n += 1
    return x / n, {t: float(np.mean(v)) for t, v in variety.items()}, nu


def job(args):
    task, setup, rho = args
    cells = {}
    for c in (RUNS[task] / "cells").iterdir():
        p = json.loads((c / "params.json").read_text())
        if p["setup"] == setup and float(p["rho"]) == rho and p["post_stop_mode"] == "normal":
            cells[p["arm"]] = c
    q = int(json.loads((cells["false"] / "params.json").read_text())["q"])
    res = {arm: run_arm(task, setup, rho, q, c, seed=21 if arm == "false" else 22) for arm, c in cells.items()}
    dm, dse = data_delta(cells)
    d = pd.DataFrame(dict(task=task, setup=setup.split("_", 1)[1], rho=rho, t=TIMES,
                          model=res["false"][0] - res["silent"][0], data=dm.reindex(TIMES).to_numpy(),
                          data_se=dse.reindex(TIMES).to_numpy(), data_30=dm.get(30.0, np.nan)))
    var = dict(task=task, setup=setup.split("_", 1)[1], rho=rho, nu_false=res["false"][2], nu_silent=res["silent"][2],
               **{f"model_distinct_{int(t)}": v for t, v in res["false"][1].items()})
    print("done", task, setup, rho, flush=True)
    return d, var


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    jobs = [(t, s, r) for t, ss in (("task003", ["task003_symmetric", "task003_nosolution"]),
                                    ("Object Placements", ["op_symmetric", "op_nosolution"]))
            for s in ss for r in (0.25, 0.75)]
    with multiprocessing.Pool(a.workers) as pool:
        res = pool.map(job, jobs, chunksize=1)
    d = pd.concat([r[0] for r in res])
    d.round(5).to_csv(HERE / "nbd_delta.csv", index=False)
    v = pd.DataFrame([r[1] for r in res])
    div = pd.read_csv(DIV).pivot_table(index=["task", "setup", "rho"], columns="t", values="distinct_ctrl_board")
    rows = []
    for (task, setup, rho), g in d.groupby(["task", "setup", "rho"]):
        g = g.set_index("t")
        d30 = g.data_30.iloc[0]
        vv = v[(v.task == task) & (v.setup == setup) & (v.rho == rho)].iloc[0]
        row = dict(task=task, setup=setup, rho=rho, data_30=d30, model_31=g.model[31.0], data_130=g.data[130.0],
                   model_130=g.model[130.0], data_retained=g.data[130.0] / d30, model_retained=g.model[130.0] / d30,
                   nu=vv.nu_false, scored=not (task == "Object Placements" and setup == "symmetric" and rho == 0.75))
        for t in (80.0, 130.0):
            row[f"distinct_data_{int(t)}"] = div.loc[(task, setup, rho), t]
            row[f"distinct_model_{int(t)}"] = vv[f"model_distinct_{int(t)}"]
        row["variety_ok"] = all(abs(row[f"distinct_model_{t}"] / row[f"distinct_data_{t}"] - 1) <= 0.25 for t in (80, 130))
        row["persistence_ok"] = abs(row["model_retained"] - row["data_retained"]) <= 0.2
        rows.append(row)
    s = pd.DataFrame(rows)
    s.round(4).to_csv(HERE / "nbd_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(s.round(3).to_string(index=False))
    sc = s[s.scored]
    print(f"variety ok {int(sc.variety_ok.sum())}/{len(sc)}; persistence ok {int(sc.persistence_ok.sum())}/{len(sc)}")


if __name__ == "__main__":
    main()
