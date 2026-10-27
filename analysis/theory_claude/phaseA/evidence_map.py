"""Phase A2: the compact evidence map, built from the world and setups alone (no simulation data).

State of a memory M: s = (z, c, a) with z = 1 if M proves A0, c = distinct controller-only facts in M (capped at 4),
a = distinct agent facts in M (capped at 6). Phi_hat(s) = mean of P(A2 | M) over memories drawn uniformly among
subsets of the circulating facts with c controller-only and a agent facts; for a capped bin the exact count is drawn
uniformly in the capped range (00_preregistration.md, change log). 20,000 draws per (c, a).

Writes evidence_map.csv: task, setup, z, c, a, phi_hat, n_draws (draws that landed in this z), and the share of z = 1
per (c, a).

Usage, from the repository root:
    caffeinate -i .venv/bin/python analysis/theory_claude/phaseA/evidence_map.py --workers 7
"""
from __future__ import annotations
import argparse, multiprocessing, pathlib, sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
A = HERE.parents[1]
sys.path.insert(0, str(A / "task003_llm_free"))
from llmfree_core.kworld import load_world  # noqa: E402
from llmfree_core.setups import load_setup  # noqa: E402

WORLDS = {"task003": (A / "task003_llm_free/llmfree_core/world_task003.json", A / "task003_llm_free/experimental_setup",
                      ["task003_symmetric", "task003_nosolution"]),
          "Object Placements": (A / "object_placements_llm_free/world/world_op1.json",
                                A / "object_placements_llm_free/experimental_setup", ["op_symmetric", "op_nosolution"])}
C_CAP, A_CAP, DRAWS = 4, 6, 20000
_W = {}


def facts(task, name):
    wf, sd, _ = WORLDS[task]
    if task not in _W:
        _W[task] = load_world(str(wf))
    w = _W[task]
    st = load_setup(name, sd, w)
    agent = sorted(set(st.agent_facts))
    ctrl = sorted(set(st.pools[2]) - set(agent))
    return w, agent, ctrl


def state(mask, agent_mask, ctrl_mask, world):
    z = int(world.posterior(mask)[0] >= 1 - 1e-12)
    c = min(bin(mask & ctrl_mask).count("1"), C_CAP)
    a = min(bin(mask & agent_mask).count("1"), A_CAP)
    return z, c, a


def one(args):
    task, name, c, a, seed = args
    w, agent, ctrl = facts(task, name)
    rng = np.random.default_rng(seed)
    out = {}
    c_range = range(c, len(ctrl) + 1) if c == C_CAP else [c]
    a_range = range(a, len(agent) + 1) if a == A_CAP else [a]
    for _ in range(DRAWS):
        cc, aa = int(rng.choice(list(c_range))), int(rng.choice(list(a_range)))
        m = 0
        for f in rng.choice(ctrl, size=cc, replace=False) if cc else []:
            m |= 1 << int(f)
        for f in rng.choice(agent, size=aa, replace=False) if aa else []:
            m |= 1 << int(f)
        p = w.posterior(m)
        z = int(p[0] >= 1 - 1e-12)
        s, n = out.get(z, (0.0, 0))
        out[z] = (s + p[2], n + 1)
    return [dict(task=task, setup=name, z=z, c=c, a=a, phi_hat=s / n, n_draws=n, share_z_in_ca=n / DRAWS)
            for z, (s, n) in out.items()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=7)
    a = ap.parse_args()
    jobs = []
    for task, (_, _, names) in WORLDS.items():
        for name in names:
            _, agent, ctrl = facts(task, name)
            for c in range(min(C_CAP, len(ctrl)) + 1):
                for aa in range(min(A_CAP, len(agent)) + 1):
                    jobs.append((task, name, c, aa, 1000 * c + aa))
    with multiprocessing.Pool(a.workers) as pool:
        rows = [r for part in pool.map(one, jobs, chunksize=1) for r in part]
    out = pd.DataFrame(rows).sort_values(["task", "setup", "c", "a", "z"])
    out.round(5).to_csv(HERE / "evidence_map.csv", index=False)
    pd.set_option("display.width", 200)
    for (t, s), g in out.groupby(["task", "setup"]):
        print(t, s)
        print(g.pivot_table(index=["z", "c"], columns="a", values="phi_hat").round(3).to_string())


if __name__ == "__main__":
    main()
