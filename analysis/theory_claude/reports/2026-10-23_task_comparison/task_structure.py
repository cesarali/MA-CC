"""Task-structure statistics computed from the world and setups alone (no simulation data), for both tasks and setups:
  single-fact P(A2 | {f}) for agent facts and for controller-only facts (the false controller's pool minus agent facts)
  pair synergy among controller-only facts: P(A2 | {f, g}) - 1/3 - [(P(A2|f) - 1/3) + (P(A2|g) - 1/3)]
      (> 0: the pair pushes A2 more than its two facts separately; additive evidence gives about 0)
  best single / best pair / whole pool: max P(A2) over single controller-only facts, pairs, and all of them together
  symmetric setups: smallest number of distinct agent facts that proves A0, and how many proofs of that size exist
Writes task_structure.csv (summary) and single_fact_p2.csv (per fact).

Usage, from the repository root:
    .venv/bin/python analysis/theory_claude/reports/2026-10-23_task_comparison/task_structure.py
"""
from __future__ import annotations
import itertools, pathlib, sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from carrier_after_stop import RUNS, load_world, load_setup  # noqa: E402

SETUPS = {"task003": ["task003_symmetric", "task003_nosolution"], "Object Placements": ["op_symmetric", "op_nosolution"]}


def main():
    rows, single = [], []
    for task, (_, wf, sd) in RUNS.items():
        world = load_world(str(wf))
        post = lambda fs: world.posterior(sum(1 << int(f) for f in fs))  # noqa: E731
        for name in SETUPS[task]:
            st = load_setup(name, sd, world)
            agent = sorted(set(st.agent_facts))
            ctrl = sorted(set(st.pools[2]) - set(agent))
            for kind, fs in (("agent", agent), ("controller-only", ctrl)):
                for f in fs:
                    p = post([f])
                    single.append(dict(task=task, setup=name.split("_", 1)[1], kind=kind, fact=world.fact_ids[f],
                                       p0=p[0], p2=p[2]))
            s1 = {f: post([f])[2] - 1 / 3 for f in ctrl}
            pairs = [(f, g, post([f, g])[2] - 1 / 3) for f, g in itertools.combinations(ctrl, 2)]
            syn = [v - s1[f] - s1[g] for f, g, v in pairs]
            row = dict(task=task, setup=name.split("_", 1)[1], n_agent_facts=len(agent), n_ctrl_only=len(ctrl),
                       ctrl_single_p2_median=float(np.median([s1[f] + 1 / 3 for f in ctrl])),
                       ctrl_single_p2_max=max(s1.values()) + 1 / 3,
                       ctrl_neutral_share=float(np.mean([abs(s1[f]) < 1e-9 for f in ctrl])),
                       ctrl_pair_p2_max=max(v for *_, v in pairs) + 1 / 3,
                       ctrl_pair_synergy_mean=float(np.mean(syn)), ctrl_pair_synergy_pos_share=float(np.mean(np.array(syn) > 0.01)),
                       ctrl_pool_p2=post(ctrl)[2],
                       agent_single_p0_median=float(np.median([post([f])[0] for f in agent])),
                       agent_single_p2_median=float(np.median([post([f])[2] for f in agent])),
                       agents_joint=tuple(round(x, 3) for x in post(agent)))
            if "symmetric" in name:
                for k in range(1, len(agent) + 1):
                    proofs = [c for c in itertools.combinations(agent, k) if post(c)[0] >= 1 - 1e-12]
                    if proofs:
                        row.update(min_proof_size=k, n_min_proofs=len(proofs))
                        break
            rows.append(row)
            print(task, name, flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(HERE / "task_structure.csv", index=False)
    pd.DataFrame(single).round(4).to_csv(HERE / "single_fact_p2.csv", index=False)
    pd.set_option("display.width", 250)
    print(r.T.to_string())


if __name__ == "__main__":
    main()
