"""Fact prevalence in Simulation 2, task003-symmetric: n_j(t) = number of the 24 agents holding fact j.

A reasoner-free description of the swarm's knowledge: it uses only which facts each agent remembers
(recorded in agent_snapshots.parquet as a bitmask), never a belief. Cells with per-agent tables:
q 6, qc 12, rate 1, gate on, proof stop on; rho 0.75 (2026-10-09 grid) and rho 1 (2026-10-12 full run).

Facts are labelled by their role in the setup (from the frozen setup files, not from any reasoner):
  decisive    the 6 agent facts that together prove A0
  rival       the other 12 agent facts (4 of them are also pool facts)
  A0 pool     the 10 facts only the A0 pool brings
  A2 pool     the 10 facts only the A2 pool brings

Writes into results/coarse_graining/<date>/:
  prevalence_mean.csv   mean and 95% interval of n_j(t) per (rho, arm, fact, t)
  prevalence_t30.csv    n_j at t = 30 per fact: silent, truth, false, and paired differences
  prevalence_time.png   mean n_j(t), one panel per (rho, arm), one line per fact, coloured by role
  prevalence_groups.png summed over each role group, the three arms on one panel per (rho, group)

Usage, from the repository root:
    .venv/bin/python analysis/task003_llm_free/coarse_graining/code/prevalence.py [--date 2026-10-15]
"""
from __future__ import annotations
import argparse, json, pathlib, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]          # analysis/task003_llm_free
sys.path.insert(0, str(ROOT))
from llmfree_core.world import World  # noqa: E402
from llmfree_core.setups import load_setup  # noqa: E402

RES = ROOT / "results"
CELLS = {
    0.75: {"silent": RES / "simulation_2/2026-10-09_sim2_grid/cells/task003_symmetric__silent__q6__rho0.75",
           "truth": RES / "simulation_2/2026-10-09_sim2_grid/cells/task003_symmetric__truth__q6__qc12__rho0.75__lc1__gateon__stopon",
           "false": RES / "simulation_2/2026-10-09_sim2_grid/cells/task003_symmetric__false__q6__qc12__rho0.75__lc1__gateon__stopon"},
    1.0: {"silent": RES / "simulation_2/2026-10-12_sim2_main_rho1_full/cells/task003_symmetric__silent",
          "truth": RES / "simulation_2/2026-10-12_sim2_main_rho1_full/cells/task003_symmetric__truth__lc1__gateon__stopon",
          "false": RES / "simulation_2/2026-10-12_sim2_main_rho1_full/cells/task003_symmetric__false__lc1__gateon__stopon"},
}
ROLE_COLOR = {"decisive": "tab:green", "rival": "0.55", "A0 pool": "tab:blue", "A2 pool": "tab:red"}
ARM_COLOR = {"silent": "0.45", "truth": "tab:blue", "false": "tab:red"}
ARM_LABEL = {"silent": "silent", "truth": "controller, target A0", "false": "controller, target A2"}


def facts_and_roles(world: World):
    setup = load_setup("task003_symmetric", ROOT / "experimental_setup", world)
    agents = json.loads((ROOT / "experimental_setup/task003_symmetric/agents.json").read_text())
    decisive = {world.index[f["fact_id"]] for f in agents_facts(agents) if f.get("decisive")}
    held = set(setup.agent_facts)
    roles = {}
    for j in held:
        roles[j] = "decisive" if j in decisive else "rival"
    for k, name in ((0, "A0 pool"), (2, "A2 pool")):
        for j in setup.pools[k]:
            roles.setdefault(j, name)
    assert sum(r == "decisive" for r in roles.values()) == 6 and len(roles) == 38
    in_pool = {j: ("A0" if j in setup.pools[0] else "A2" if j in setup.pools[2] else "") for j in roles}
    return roles, in_pool, setup


def agents_facts(agents: dict):
    """The fact records in agents.json, wherever they are listed."""
    for key in ("facts", "fact_details", "agent_facts"):
        if key in agents and isinstance(agents[key], list):
            return agents[key]
    for v in agents.values():
        if isinstance(v, list) and v and isinstance(v[0], dict) and "decisive" in v[0]:
            return v
    raise KeyError("no fact list with a 'decisive' field in agents.json")


def prevalence(cell: pathlib.Path, facts: list[int]) -> pd.DataFrame:
    """Per episode and snapshot time: n_j = number of agents holding fact j."""
    d = pd.read_parquet(cell / "agent_snapshots.parquet", columns=["episode", "grid_index", "active"])
    act = d.active.to_numpy().astype(np.int64)
    out = d[["episode", "grid_index"]].copy()
    for j in facts:
        out[j] = (act >> j) & 1
    return out.groupby(["episode", "grid_index"], sort=True)[facts].sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2026-10-15")
    a = ap.parse_args()
    outdir = RES / "coarse_graining" / a.date
    outdir.mkdir(parents=True, exist_ok=True)
    world = World(ROOT / "llmfree_core/world_task003.json")
    roles, in_pool, _ = facts_and_roles(world)
    order = sorted(roles, key=lambda j: (list(ROLE_COLOR).index(roles[j]), world.fact_ids[j]))
    dt = 0.25

    per = {}
    rows = []
    for rho, arms in CELLS.items():
        for arm, cell in arms.items():
            n = prevalence(cell, order)
            per[(rho, arm)] = n
            g = n.groupby(level="grid_index")
            m, s, c = g.mean(), g.std(), g.count()
            for j in order:
                for gi in m.index:
                    se = s.at[gi, j] / np.sqrt(c.at[gi, j])
                    rows.append((rho, arm, world.fact_ids[j], roles[j], in_pool[j], gi * dt, m.at[gi, j],
                                 m.at[gi, j] - 1.96 * se, m.at[gi, j] + 1.96 * se))
            print(f"rho {rho} {arm}: {n.index.get_level_values(0).nunique()} episodes, "
                  f"{n.index.get_level_values(1).nunique()} times")
    mean = pd.DataFrame(rows, columns=["rho", "arm", "fact", "role", "also_in_pool", "t", "mean", "lo", "hi"])
    mean.to_csv(outdir / "prevalence_mean.csv", index=False)

    # t = 30: per fact, each arm and the paired differences (same episodes, same random draws)
    gi30 = int(round(30 / dt))
    t30 = []
    for rho in CELLS:
        s = per[(rho, "silent")].xs(gi30, level="grid_index")
        for j in order:
            row = {"rho": rho, "fact": world.fact_ids[j], "role": roles[j], "also_in_pool": in_pool[j],
                   "silent": s[j].mean()}
            for arm in ("truth", "false"):
                c = per[(rho, arm)].xs(gi30, level="grid_index")
                diff = (c[j] - s[j].reindex(c.index)).dropna()
                row[arm] = c[j].mean()
                row[f"{arm}_minus_silent"] = diff.mean()
                row[f"{arm}_minus_silent_se"] = diff.std() / np.sqrt(len(diff))
            t30.append(row)
    pd.DataFrame(t30).to_csv(outdir / "prevalence_t30.csv", index=False)

    # figure 1: every fact over time
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.6), sharex=True, sharey=True)
    for i, rho in enumerate(CELLS):
        for k, arm in enumerate(("silent", "truth", "false")):
            ax = axes[i, k]
            x = mean[(mean.rho == rho) & (mean.arm == arm)]
            for f, h in x.groupby("fact", sort=False):
                ax.plot(h.t, h["mean"], color=ROLE_COLOR[h.role.iloc[0]], lw=0.8, alpha=0.8)
            ax.axvline(30, color="0.6", ls="--", lw=0.6)
            ax.set_title(f"{ARM_LABEL[arm]}, $\\rho = {rho:g}$", fontsize=8)
            if i == 1:
                ax.set_xlabel("time $t$")
            if k == 0:
                ax.set_ylabel("agents holding the fact, $n_j$")
    for role, col in ROLE_COLOR.items():
        axes[0, 0].plot([], [], color=col, label=role)
    axes[0, 0].legend(fontsize=6.5, loc="upper left")
    fig.tight_layout(pad=0.4)
    fig.savefig(outdir / "prevalence_time.png", dpi=140)
    plt.close(fig)

    # figure 2: summed over each role group
    fig, axes = plt.subplots(2, 4, figsize=(10, 4.8), sharex=True)
    for i, rho in enumerate(CELLS):
        for k, role in enumerate(ROLE_COLOR):
            ax = axes[i, k]
            js = [j for j in order if roles[j] == role]
            for arm in ("silent", "truth", "false"):
                tot = per[(rho, arm)][js].sum(axis=1).groupby(level="grid_index")
                m, se = tot.mean(), tot.std() / np.sqrt(tot.count())
                ax.plot(m.index * dt, m, color=ARM_COLOR[arm], lw=1.1, label=ARM_LABEL[arm])
                ax.fill_between(m.index * dt, m - 1.96 * se, m + 1.96 * se, color=ARM_COLOR[arm], alpha=.2, lw=0)
            ax.axvline(30, color="0.6", ls="--", lw=0.6)
            ax.set_title(f"{role} ({len(js)} facts), $\\rho = {rho:g}$", fontsize=8)
            if i == 1:
                ax.set_xlabel("time $t$")
            if k == 0:
                ax.set_ylabel("$\\sum_j n_j$ over the group")
    axes[0, 0].legend(fontsize=6)
    fig.tight_layout(pad=0.4)
    fig.savefig(outdir / "prevalence_groups.png", dpi=140)
    plt.close(fig)
    print("written to", outdir)


if __name__ == "__main__":
    main()
