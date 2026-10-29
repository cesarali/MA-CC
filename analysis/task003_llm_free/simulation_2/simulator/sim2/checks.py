"""Start-up checks on the frozen setups (spec §1). Each check uses the exact posterior engine,
not file names. A failure stops the run: the setup is reported, never rebuilt."""
from __future__ import annotations
import re

from llmfree_core.setups import Setup
from llmfree_core.world import World

EXPECTED = {
    "task003_symmetric": {"agents": 24, "distinct_facts": 18, "pool_size": 12, "shared_pool": False},
    "task003_nosolution": {"agents": 24, "distinct_facts": 15, "pool_size": 12, "shared_pool": True},
    # MuSR Object Placements world (llmfree_core/op_world.py), built with the same rules
    "op_symmetric": {"agents": 24, "distinct_facts": 18, "pool_size": 12, "shared_pool": False},
    "op_nosolution": {"agents": 24, "distinct_facts": 15, "pool_size": 12, "shared_pool": True},
    # 15-agent campaign (analysis/15_agent_simulation, 29 Oct 2026): no-solution agents are uniform, not (0.5, 0, 0.5)
    "task003_symmetric_15": {"agents": 15, "distinct_facts": 15, "pool_size": 12, "shared_pool": False},
    "task003_nosolution_uniform_15": {"agents": 15, "distinct_facts": 15, "pool_size": 12, "shared_pool": True,
                                      "uniform_tolerance": 0.02},
    "op_symmetric_15": {"agents": 15, "distinct_facts": 15, "pool_size": 12, "shared_pool": False},
    "op_nosolution_uniform_15": {"agents": 15, "distinct_facts": 15, "pool_size": 12, "shared_pool": True,
                                 "uniform_tolerance": 0.02},
}


class SetupCheckFailed(RuntimeError):
    pass


def check_setup(world: World, setup: Setup) -> dict:
    """Returns the measured properties; raises SetupCheckFailed listing every failed check."""
    exp = EXPECTED.get(setup.name)
    k = re.search(r"_K(\d+)$", setup.name)
    if exp is None and k:                    # K-projects setups: K copies of the base setup, one per project
        K, base = int(k.group(1)), EXPECTED[setup.name[:k.start()]]
        exp = {"agents": base["agents"] * K, "distinct_facts": base["distinct_facts"] * K,
               "pool_size": base["pool_size"] * K, "shared_pool": base["shared_pool"], "K": K}
    if exp is None:
        raise SetupCheckFailed(f"no checks defined for setup {setup.name!r}")
    agents = 0
    for f in setup.agent_facts:
        agents |= 1 << f
    pooled = world.posterior(agents)
    got = {
        "agents": len(setup.agent_facts),
        "distinct_facts": len(set(setup.agent_facts)),
        "pool_size": {k: len(v) for k, v in setup.pools.items()},
        "shared_pool": setup.pools[0] == setup.pools[2],
        "pooled_agent_posterior": pooled,
    }
    failed = []
    if got["agents"] != exp["agents"]:
        failed.append(f"{got['agents']} agents, expected {exp['agents']}")
    if got["distinct_facts"] != exp["distinct_facts"]:
        failed.append(f"{got['distinct_facts']} distinct agent facts, expected {exp['distinct_facts']}")
    if any(n != exp["pool_size"] for n in got["pool_size"].values()):
        failed.append(f"pool sizes {got['pool_size']}, expected {exp['pool_size']} each")
    if got["shared_pool"] != exp["shared_pool"]:
        failed.append(f"one pool for both targets: {got['shared_pool']}, expected {exp['shared_pool']}")
    if setup.name.startswith(("task003_symmetric", "op_symmetric")):
        if pooled[0] != 1.0:
            failed.append(f"pooled agent facts should prove A0, posterior {pooled}")
    else:
        # K = 1: the agents' evidence is exactly balanced. K > 1: summing scores over projects breaks that balance
        # (the pooled posterior leans to A2); only "no proof" is required there.
        if "uniform_tolerance" in exp:
            if max(abs(x - 1 / 3) for x in pooled) > exp["uniform_tolerance"]:
                failed.append(f"pooled agent posterior {pooled}, expected uniform within {exp['uniform_tolerance']}")
        elif "K" not in exp and any(abs(x - y) > 1e-12 for x, y in zip(pooled, (0.5, 0.0, 0.5))):
            failed.append(f"pooled agent posterior {pooled}, expected (0.5, 0, 0.5)")
        everything = agents
        for f in setup.pools[0]:
            everything |= 1 << f
        if max(world.posterior(everything)) == 1.0:
            failed.append("agent facts plus the pool prove an allocation; the setup should have no proof")
    if failed:
        raise SetupCheckFailed(f"{setup.name}: " + "; ".join(failed))
    return got
