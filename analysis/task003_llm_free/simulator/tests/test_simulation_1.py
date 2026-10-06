"""Checks for Simulation 1. Run from the repository root:
    .venv/bin/python -m pytest analysis/task003_llm_free/simulator/tests -q
"""
from __future__ import annotations
import json, pathlib, sys

import pytest
import yaml

SIM = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
from llmfree.setups import load_setup  # noqa: E402
from llmfree.simulation_1 import NIGHT_COLUMNS, Params, run_episode  # noqa: E402
from llmfree.world import World, bits  # noqa: E402
from run_simulation_1 import build_cells  # noqa: E402

SETUPS = SIM.parent / "experimental_setup"
WORLD = World(SIM / "world_task003.json")
SETUP = {s: load_setup(s, SETUPS, WORLD) for s in ("task003_symmetric", "task003_nosolution")}
CONFIG = yaml.safe_load((SIM / "simulation_1_config.yaml").read_text())


def params(setup, arm, **kw):
    base = dict(setup=setup, arm=arm, q=6, qc=None if arm == "silent" else 12,
                b=None if arm == "silent" else 3, rho=0.75)
    base.update(kw)
    return Params(**base)


# --- the world and the setups --------------------------------------------------------

@pytest.mark.parametrize("setup", list(SETUP))
def test_agents_joint_posterior_matches_frozen_setup(setup):
    """The simulator's posterior engine reproduces the frozen setup's own numbers."""
    agents = json.loads((SETUPS / setup / "agents.json").read_text())
    s = 0
    for f in SETUP[setup].agent_facts:
        s |= 1 << f
    assert [round(x, 6) for x in WORLD.posterior(s)] == agents["properties"]["joint_posterior_A0_A1_A2"]


def test_pools_match_files():
    sym = SETUP["task003_symmetric"]
    for k, name in ((0, "A0"), (2, "A2")):
        ids = json.loads((SETUPS / "task003_symmetric" / f"controller_pool_{name}.json").read_text())["fact_ids"]
        assert sorted(WORLD.fact_ids[i] for i in sym.pools[k]) == sorted(ids)
    assert not set(sym.pools[0]) & set(sym.pools[2])
    nos = SETUP["task003_nosolution"]
    assert nos.pools[0] == nos.pools[2]


def test_prior_is_one_third():
    assert WORLD.posterior(0) == (1 / 3, 1 / 3, 1 / 3)


# --- the config ----------------------------------------------------------------------

def test_grid_has_exactly_156_cells():
    cells = build_cells(CONFIG)
    assert len(cells) == CONFIG["cells"]["expected_total"] == 156
    assert sum(p.arm == "silent" for p in cells) == 12
    assert all(p.qc is None and p.b is None for p in cells if p.arm == "silent")


def test_arm_names_are_strings():
    assert CONFIG["grid"]["arm"] == ["silent", "truth", "false"]


# --- the dynamics --------------------------------------------------------------------

def test_same_seed_same_episode():
    p = params("task003_nosolution", "false")
    assert run_episode(WORLD, SETUP[p.setup], p, 7) == run_episode(WORLD, SETUP[p.setup], p, 7)


@pytest.mark.parametrize("setup", list(SETUP))
def test_arms_share_day_one(setup):
    """The controller first acts on night 1, so day 1 must be identical across arms (spec §8)."""
    day1 = [[r for r in run_episode(WORLD, SETUP[setup], params(setup, arm), 3)[0] if r[1] == 0]
            for arm in ("silent", "truth", "false")]
    assert day1[0] == day1[1] == day1[2]


@pytest.mark.parametrize("setup", list(SETUP))
@pytest.mark.parametrize("arm", ["silent", "truth", "false"])
def test_invariants(setup, arm):
    p = params(setup, arm)
    for episode in range(5):
        agents, nights, days = run_episode(WORLD, SETUP[setup], p, episode)
        assert len(days) == p.M and len(agents) == p.M * 24
        for r in agents:
            active, read, n_read, posted = r[4], r[5], r[6], r[11]
            assert n_read <= p.q
            assert read & ~active == 0                      # read facts are active
            if posted >= 0:
                assert (active >> posted) & 1              # posts only an active fact
        for row in nights:
            n = dict(zip(NIGHT_COLUMNS, row))
            assert n["n_posted"] in (0, p.b)               # all or nothing
            assert n["n_posts_read"] <= p.qc
            if n["decision"] in ("proved", "gate"):
                assert n["n_posted"] == 0
            if n["decision"] == "proved":
                assert n["p_target_given_read"] == 1.0
            pool = set(SETUP[setup].pools[{"truth": 0, "false": 2}[arm]])
            assert set(bits(n["posted_facts"])) <= pool    # posts only from its pool
        if arm == "silent":
            assert nights == []


def test_rho_one_never_forgets():
    p = params("task003_symmetric", "silent", rho=1.0)
    agents = run_episode(WORLD, SETUP[p.setup], p, 0)[0]
    last = {}
    for r in agents:
        a, active = r[3], r[4]
        if a in last:
            assert last[a] & ~active == 0                   # nothing ever leaves memory
        last[a] = active


def test_votes_gate_with_uncleared_board_warns():
    assert params("task003_nosolution", "truth", is_board_cleared=False).check()
    assert not params("task003_nosolution", "truth").check()


def test_unimplemented_options_refuse():
    with pytest.raises(NotImplementedError):
        params("task003_nosolution", "truth", controller_memory="accumulate").check()
    with pytest.raises(NotImplementedError):
        params("task003_nosolution", "truth", board_read_weighting="recency").check()


@pytest.mark.parametrize("mode", ["probability_matching", "softmax"])
def test_other_vote_modes_run(mode):
    p = params("task003_nosolution", "false", agent_sampling_mode=mode)
    agents = run_episode(WORLD, SETUP[p.setup], p, 0)[0]
    assert {r[10] for r in agents} <= {0, 1, 2}
