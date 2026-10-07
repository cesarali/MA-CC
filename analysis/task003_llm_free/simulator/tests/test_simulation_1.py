"""Checks for Simulation 1. Run from the repository root:
    .venv/bin/python -m pytest analysis/task003_llm_free/simulator/tests -q
"""
from __future__ import annotations
import glob, json, pathlib, sys

import pytest
import yaml

SIM = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
from llmfree.setups import load_setup  # noqa: E402
from llmfree.simulation_1 import (ABSTAINED, AGENT_COLUMNS, CONTROLLER_CODE, DAY_COLUMNS,  # noqa: E402
                                  NIGHT_COLUMNS, NO_MEMORY, UNIFORM, Params, decode_post, run_episode)
from llmfree.world import World, bits  # noqa: E402
from run_simulation_1 import build_cells, check_record, dirty_check_args  # noqa: E402

SETUPS = SIM.parent / "experimental_setup"
WORLD = World(SIM / "world_task003.json")
SETUP = {s: load_setup(s, SETUPS, WORLD) for s in ("task003_symmetric", "task003_nosolution")}
CONFIGS = {pathlib.Path(f).stem: yaml.safe_load(open(f)) for f in glob.glob(str(SIM / "configs" / "*.yaml"))}
TARGET = {"truth": 0, "false": 2}


def params(setup, arm, **kw):
    base = dict(setup=setup, arm=arm, q=6, qc=None if arm == "silent" else 12,
                b=None if arm == "silent" else 3, rho=0.75)
    base.update(kw)
    return Params(**base)


def episode(p, e=0):
    a, n, d = run_episode(WORLD, SETUP[p.setup], p, e)
    return ([dict(zip(AGENT_COLUMNS, r)) for r in a], [dict(zip(NIGHT_COLUMNS, r)) for r in n],
            [dict(zip(DAY_COLUMNS, r)) for r in d])


# --- the world and the setups --------------------------------------------------------

@pytest.mark.parametrize("setup", list(SETUP))
def test_agents_joint_posterior_matches_frozen_setup(setup):
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


# --- the configs ---------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_config_cell_count(name):
    cfg = CONFIGS[name]
    cells = build_cells(cfg)
    assert len(cells) == cfg["cells"]["expected_total"]
    assert all(p.qc is None and p.b is None for p in cells if p.arm == "silent")
    assert all(isinstance(a, str) for a in cfg["grid"]["arm"])
    check_record(cfg)
    for p in cells:
        p.check()


def test_step1_configs_are_492_and_differ_only_as_intended():
    base = CONFIGS["sim1_base"]
    assert len(build_cells(base)) == 492
    expect = {"sim1_pm": {"agent_sampling_mode": "probability_matching"},
              "sim1_pm_post_always": {"agent_sampling_mode": "probability_matching", "agent_post_always": True},
              "sim1_uniform_post": {"agent_post_rule": "uniform_active"}}
    for name, changes in expect.items():
        cfg = CONFIGS[name]
        assert cfg["grid"] == base["grid"]
        diff = {k: v for k, v in cfg["fixed"].items() if base["fixed"].get(k) != v}
        assert diff == changes, name


def test_step2_configs_change_only_the_stopping_rule():
    for name in ("sim1_base", "sim1_pm", "sim1_pm_post_always", "sim1_uniform_post"):
        step1, step2 = CONFIGS[name], CONFIGS[f"{name}_no_proof_stop"]
        diff = {k: v for k, v in step2["fixed"].items() if step1["fixed"].get(k) != v}
        assert diff == {"silent_when_target_proved": False}
        assert step2["grid"]["setup"] == ["task003_symmetric"] and step2["grid"]["arm"] == ["truth"]
        assert len(build_cells(step2)) == 120


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
@pytest.mark.parametrize("mode", ["argmax", "probability_matching"])
def test_invariants(setup, arm, mode):
    p = params(setup, arm, agent_sampling_mode=mode)
    own = 0
    for f in SETUP[setup].agent_facts:
        own |= 1 << f
    for e in range(4):
        agents, nights, days = episode(p, e)
        assert len(days) == p.M and len(agents) == p.M * 24
        end_of_yesterday = {}
        for r in agents:
            # memory bookkeeping
            assert r["memory_after_dawn"] & ~r["known_before_reading"] == 0
            assert r["active_facts"] == r["memory_after_dawn"] | r["facts_read"]
            if r["agent"] in end_of_yesterday:
                assert r["memory_after_dawn"] & ~end_of_yesterday[r["agent"]] == 0   # dawn only removes
            end_of_yesterday[r["agent"]] = r["active_facts"]
            # posts read match the recorded fact set and count
            decoded = [decode_post(c) for c in r["posts_read"]]
            assert len(decoded) == r["n_posts_read"] <= p.q
            fs = 0
            for author, fact in decoded:
                fs |= 1 << fact
                assert author != r["agent"]                         # never reads its own posts
            assert fs == r["facts_read"]
            # posting
            if r["posted_fact"] >= 0:
                assert (r["active_facts"] >> r["posted_fact"]) & 1
            else:
                assert r["post_reason"] in (ABSTAINED, NO_MEMORY)
            # proof measured two ways
            assert r["proves_A0"] or not r["proves_A0_own_evidence"]
        for n in nights:
            assert n["day"] < p.M - 1                               # no controller on the final night
            assert n["n_posted"] in (0, p.b)
            assert n["n_posts_read"] == len(n["posts_read"]) <= p.qc
            assert all(a != CONTROLLER_CODE for a, _ in map(decode_post, n["posts_read"]))
            if n["decision"] in ("proved", "gate"):
                assert n["n_posted"] == 0
            if n["decision"] == "proved":
                assert n["p_target_given_read"] == 1.0
            assert set(bits(n["posted_facts"])) <= set(SETUP[setup].pools[TARGET[arm]])
            day_votes = [r["vote"] for r in agents if r["episode"] == e and r["day"] == n["day"]]
            assert n["true_target_share"] == day_votes.count(TARGET[arm]) / 24
        if arm == "silent":
            assert nights == []


def test_argmax_never_abstains_when_supported():
    """With argmax the vote is the most probable answer, so some fact always supports it
    (in context or alone) unless memory is empty."""
    agents, _, _ = episode(params("task003_nosolution", "silent"))
    assert all(r["post_reason"] != ABSTAINED for r in agents)


def test_uniform_post_rule():
    agents, _, _ = episode(params("task003_nosolution", "silent", agent_post_rule="uniform_active"))
    assert {r["post_reason"] for r in agents} <= {UNIFORM, NO_MEMORY}


def test_final_night_option():
    p = params("task003_nosolution", "false", controller_acts_on_final_night=True)
    _, nights, _ = episode(p)
    assert max(n["day"] for n in nights) == p.M - 1


def test_no_proof_stop_keeps_posting():
    """Symmetric truth controller: with the rule on it stays silent when the board proves A0;
    with it off it posts in those same situations."""
    on = params("task003_symmetric", "truth", controller_gate="always")
    off = params("task003_symmetric", "truth", controller_gate="always", silent_when_target_proved=False)
    seen_on, seen_off = set(), set()
    for e in range(10):
        seen_on |= {n["decision"] for n in episode(on, e)[1]}
        seen_off |= {n["decision"] for n in episode(off, e)[1]}
    assert "proved" in seen_on and "posted_proved" not in seen_on
    assert "proved" not in seen_off and "posted_proved" in seen_off


def test_rho_one_never_forgets():
    agents, _, _ = episode(params("task003_symmetric", "silent", rho=1.0))
    last = {}
    for r in agents:
        if r["agent"] in last:
            assert last[r["agent"]] & ~r["active_facts"] == 0
        last[r["agent"]] = r["active_facts"]


def test_votes_gate_with_uncleared_board_warns():
    assert params("task003_nosolution", "truth", is_board_cleared=False).check()
    assert not params("task003_nosolution", "truth").check()


def test_unimplemented_options_refuse():
    with pytest.raises(NotImplementedError):
        params("task003_nosolution", "truth", controller_memory="accumulate").check()
    with pytest.raises(NotImplementedError):
        params("task003_nosolution", "truth", board_read_weighting="recency").check()


def test_softmax_runs():
    agents, _, _ = episode(params("task003_nosolution", "false", agent_sampling_mode="softmax"))
    assert {r["vote"] for r in agents} <= {0, 1, 2}


def test_dirty_check_ignores_run_bookkeeping():
    """runs_index.csv and run_summaries/ change after every run; they must not mark the code dirty."""
    args = dirty_check_args()
    assert ":(exclude)runs_index.csv" in args and ":(exclude)run_summaries" in args
