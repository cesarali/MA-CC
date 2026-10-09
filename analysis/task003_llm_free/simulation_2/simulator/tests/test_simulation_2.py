"""Checks for Simulation 2 (spec §13). Run from the repository root:
    .venv/bin/python -m pytest analysis/task003_llm_free/simulation_2/simulator/tests -q
"""
from __future__ import annotations
import dataclasses, math, pathlib, random, sys

import pandas as pd
import pytest
import yaml

SIM = pathlib.Path(__file__).resolve().parents[1]                 # simulation_2/simulator
ROOT = SIM.parents[1]                                              # analysis/task003_llm_free
sys.path.insert(0, str(SIM))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "simulation_1" / "simulator"))      # only to compare rules with Simulation 1
from llmfree_core.setups import load_setup  # noqa: E402
from llmfree_core.world import World, bits  # noqa: E402
from sim2.checks import SetupCheckFailed, check_setup  # noqa: E402
from sim2.simulation_2 import (CONTROLLER, COLUMNS, Params, choose_post, controller_decision,  # noqa: E402
                               read_window, run_episode)
from run_simulation_2 import build_cells, dirty_check_args, resolve_steps  # noqa: E402
import sim1.simulation_1 as s1  # noqa: E402

SETUPS = ROOT / "experimental_setup"
WORLD = World(ROOT / "llmfree_core" / "world_task003.json")
SETUP = {s: load_setup(s, SETUPS, WORLD) for s in ("task003_symmetric", "task003_nosolution")}
BRIDGE = yaml.safe_load((SIM / "configs" / "sim2_bridge.yaml").read_text())
STEPS = resolve_steps(BRIDGE)
FIXED = BRIDGE["fixed"]
TARGET = {"truth": 0, "false": 2}


def params(step: str, setup: str, arm: str, **kw) -> Params:
    return Params(setup=setup, arm=arm, step=step, **{**FIXED, **STEPS[step], **kw})


def tables(p: Params, episodes=range(3)) -> dict[str, pd.DataFrame]:
    rows = {k: [] for k in COLUMNS}
    for e in episodes:
        r = run_episode(WORLD, SETUP[p.setup], p, e)
        for k in rows:
            rows[k] += r[k]
    return {k: pd.DataFrame(v, columns=COLUMNS[k]) for k, v in rows.items()}


# ---------------------------------------------------------------- 1. setups

@pytest.mark.parametrize("setup", sorted(SETUP))
def test_setup_checks_pass(setup):
    check_setup(WORLD, SETUP[setup])


def test_setup_check_fails_loudly():
    s = SETUP["task003_nosolution"]
    broken = dataclasses.replace(s, agent_facts=(s.agent_facts[0],) * 24)
    with pytest.raises(SetupCheckFailed):
        check_setup(WORLD, broken)


def test_prior_is_one_third():
    """The fallback threshold P(vote | no facts) equals Simulation 1's 1/3 (spec §7)."""
    assert WORLD.posterior(0) == (1 / 3, 1 / 3, 1 / 3)


# ---------------------------------------------------------------- the bridge config

def test_bridge_has_36_cells_and_reuses_s3_silent():
    cells, reused = build_cells(BRIDGE)
    assert len(cells) == BRIDGE["cells"]["expected_total"] == 36
    for step in ("S4", "S5", "S6"):
        for setup in SETUP:
            assert reused[f"{step}__{setup}__silent"] == f"S3__{setup}__silent"


def test_each_step_changes_one_setting():
    names = list(STEPS)
    for a, b in zip(names, names[1:]):
        changed = {k for k in STEPS[b] if STEPS[a].get(k) != STEPS[b][k]} - {"lambda_c"}
        assert len(changed) == 1, (b, changed)


def test_last_step_is_the_source_document_setting():
    """S6 must equal the rate scan's cell at lambda_c = 1: the defaults of Params."""
    defaults = Params(setup="x", arm="silent", step="x")
    for k, v in STEPS["S6"].items():
        assert getattr(defaults, k) == v, k


def test_cumulative_step_refuses_two_changes():
    cfg = {**BRIDGE, "steps": {"S0": BRIDGE["steps"]["S0"], "S1": {"board": "persistent", "forgetting": "continuous"}}}
    with pytest.raises(ValueError):
        resolve_steps(cfg)


# ---------------------------------------------------------------- 2. stochastic laws

def test_poisson_activation_counts():
    """Each agent acts on average lambda_a * t_end = 40 times, with Poisson variance."""
    d = tables(params("S3", "task003_nosolution", "silent"), range(20))["actions"]
    counts = d.groupby(["episode", "agent"]).size()
    assert len(counts) == 480
    se = math.sqrt(40 / 480)
    assert abs(counts.mean() - 40) < 4 * se
    assert 0.7 < counts.var() / 40 < 1.4


@pytest.mark.parametrize("forgetting", ["continuous", "dawn"])
def test_fact_survival_over_one_unit_is_rho(forgetting):
    """With nothing read (q = 0) each agent keeps only its own fact; the share still active
    after one time unit is rho in both forgetting modes (dawn: after the dawn at t = 1)."""
    step = "S2" if forgetting == "continuous" else "S1"
    p = params(step, "task003_nosolution", "silent", q=0, t_end=3.0)
    snaps = tables(p, range(200))["snapshots"]
    at = lambda t: snaps[(snaps.t - t).abs() < 1e-9].mean_active_facts.mean()
    se = math.sqrt(0.75 * 0.25 / 4800)
    if forgetting == "continuous":
        assert abs(at(1.0) - 0.75) < 4 * se
    else:
        assert at(1.0) == 1.0                         # the measurement at t = 1 comes before the dawn
        assert abs(at(1.25) - 0.75) < 4 * se
    expected = 0.75 ** 2.5 if forgetting == "continuous" else 0.75 ** 2     # dawn: two dawns by t = 2.5
    assert abs(at(2.5) - expected) < 4 * math.sqrt(expected * (1 - expected) / 4800)


# ---------------------------------------------------------------- 3. budget and cutoff

@pytest.mark.parametrize("lam", [0.5, 8.0])
def test_budget_and_cutoff(lam):
    p = params("S6", "task003_nosolution", "false", lambda_c=lam)
    tb = tables(p, range(10))
    ctrl, posts, ep, act = tb["controller"], tb["posts"], tb["episodes"], tb["actions"]
    assert (ep.controller_messages <= 30).all()
    assert (ctrl.t < 30).all() and (ctrl.budget_after >= 0).all()
    assert (posts[posts.author == CONTROLLER].t < 30).all()
    assert (act.t > 30).sum() > 0                     # agents keep acting in the follow-up
    if lam == 8.0:
        assert ep.budget_exhausted_at.notna().all()   # 240 expected opportunities for 30 messages
        assert (ctrl.budget_before > 0).all()


def test_nights_schedule_never_reaches_the_budget():
    tb = tables(params("S0", "task003_symmetric", "false"), range(5))
    assert set(tb["controller"].t) <= {float(d) for d in range(1, 30)}
    assert (tb["episodes"].controller_messages <= 29).all()


# ---------------------------------------------------------------- 4. reading

def test_window_excludes_own_posts_before_taking_the_window():
    board = [(i, float(i), 0 if i % 2 else 1, i % 49, 0) for i in range(100)]   # authors alternate 1, 0
    w = read_window(board, 0, 48, agents_only=False)
    assert len(w) == 48 and all(post[2] == 1 for post in w)
    assert [post[0] for post in w] == list(range(98, 2, -2))
    board.append((100, 100.0, CONTROLLER, 3, -1))
    w = read_window(board, CONTROLLER, 48, agents_only=True)
    assert all(post[2] != CONTROLLER for post in w) and len(w) == 48


@pytest.mark.parametrize("step", ["S0", "S1", "S3", "S6"])
@pytest.mark.parametrize("setup", sorted(SETUP))
def test_reads_follow_the_rules(step, setup):
    p = params(step, setup, "false")
    tb = tables(p, range(3))
    posts = tb["posts"].set_index(["episode", "post_id"])
    for r in tb["actions"].itertuples():
        ids = list(r.posts_read)
        assert len(ids) == len(set(ids)) <= p.q
        read = posts.loc[[(r.episode, i) for i in ids]] if ids else posts.iloc[:0]
        assert (read.author != r.agent).all()                         # no self-reads
        assert (read.t < r.t).all()
        facts = 0
        for f in read.fact:
            facts |= 1 << int(f)
        assert facts == r.facts_read
        assert r.active_after & r.facts_read == r.facts_read          # every fact read is active
        if p.board == "persistent":                                   # inside the 48 most recent eligible
            mine = tb["posts"][(tb["posts"].episode == r.episode) & (tb["posts"].t < r.t)
                               & (tb["posts"].author != r.agent)]
            allowed = set(mine.post_id.nlargest(48))
            assert set(ids) <= allowed
        else:                                                         # today's posts and last night's controller posts
            assert (read.t >= math.floor(r.t)).all()
    for r in tb["controller"].itertuples():
        ids = list(r.posts_read)
        assert len(ids) == len(set(ids)) <= p.qc
        if ids:
            assert (posts.loc[[(r.episode, i) for i in ids]].author != CONTROLLER).all()


def test_days_schedule_everyone_acts_once_a_day():
    a = tables(params("S0", "task003_nosolution", "silent"), range(2))["actions"]
    a["day"] = a.t.astype(int)
    per = a.groupby(["episode", "day", "agent"]).size()
    assert (per == 1).all() and len(per) == 2 * 40 * 24


# ---------------------------------------------------------------- 5. forgetting

@pytest.mark.parametrize("setup", sorted(SETUP))
def test_no_stale_expiry(setup):
    """An expiry belongs to the latest spell: the fact was not reread between the spell's start
    and the expiry, and the spell started at t = 0 (own fact) or at the latest read of it."""
    tb = tables(params("S6", setup, "truth"), range(3))
    act, fg = tb["actions"], tb["forgetting"]
    assert len(fg) > 0
    reads = {}
    for r in act.itertuples():
        for f in bits(int(r.facts_read)):
            reads.setdefault((r.episode, r.agent, f), []).append(r.t)
    for r in fg.itertuples():
        times = [t for t in reads.get((r.episode, r.agent, r.fact), []) if t < r.t]
        assert r.spell_start == (max(times) if times else 0.0)


@pytest.mark.parametrize("step", ["S0", "S6"])
def test_rho_one_never_forgets(step):
    tb = tables(params(step, "task003_nosolution", "false", rho=1.0), range(2))
    assert len(tb["forgetting"]) == 0


# ---------------------------------------------------------------- 6. rules match Simulation 1

def _random_states(n, seed=1):
    rng = random.Random(seed)
    for _ in range(n):
        k = rng.randint(0, 9)
        yield sum(1 << f for f in rng.sample(range(WORLD.n_facts), k)), rng.randrange(3), rng


def test_posting_rule_matches_simulation_1():
    p1 = s1.Params(setup="task003_nosolution", arm="silent", q=6, qc=None, b=None, rho=0.75,
                   agent_sampling_mode="probability_matching")
    for i, (active, vote, rng) in enumerate(_random_states(3000)):
        read_now = active & rng.getrandbits(WORLD.n_facts)
        a = choose_post(WORLD, active, vote, read_now, True, random.Random(i))
        b = s1._choose_post(WORLD, active, vote, read_now, p1, random.Random(i))
        assert (a[0], a[1]) == b


@pytest.mark.parametrize("gate", ["votes", "always"])
@pytest.mark.parametrize("stop", [True, False])
@pytest.mark.parametrize("setup,arm", [("task003_symmetric", "truth"), ("task003_symmetric", "false"),
                                       ("task003_nosolution", "truth"), ("task003_nosolution", "false")])
def test_controller_rule_matches_simulation_1_with_b1(gate, stop, setup, arm):
    target = TARGET[arm]
    pool = SETUP[setup].pools[target]
    p2 = params("S6", setup, arm, controller_gate=gate, silent_when_target_proved=stop)
    p1 = s1.Params(setup=setup, arm=arm, q=6, qc=12, b=1, rho=0.75, controller_gate=gate,
                   silent_when_target_proved=stop)
    agent_facts = list(set(SETUP[setup].agent_facts))
    rng = random.Random(7)
    for _ in range(400):
        read = [(rng.randrange(24), rng.choice(agent_facts + list(pool)), 0, rng.randrange(3))
                for _ in range(rng.randint(0, 12))]
        R = 0
        for post in read:
            R |= 1 << post[1]
        v_hat = (sum(post[3] == target for post in read) / len(read)) if read else None
        d2, f2 = controller_decision(WORLD, pool, target, R, v_hat, p2)
        d1, S1, _, _ = s1._controller_night(WORLD, pool, target, read, p1)
        assert d1 == d2 and S1 == (0 if f2 < 0 else 1 << f2)


# ---------------------------------------------------------------- 7. replay

@pytest.mark.parametrize("step", ["S0", "S2", "S6"])
@pytest.mark.parametrize("setup", sorted(SETUP))
def test_replay_reconstructs_memories(step, setup):
    """Replaying logged reads and forgetting from the initial facts reproduces every recorded
    memory (before and after each action) and every agent snapshot."""
    p = params(step, setup, "false")
    tb = tables(p, range(2))
    for e in range(2):
        ev = []
        for r in tb["forgetting"][tb["forgetting"].episode == e].itertuples():
            ev.append((r.t, 3 if math.isnan(r.spell_start) else 4, "forget", r))
        for r in tb["actions"][tb["actions"].episode == e].itertuples():
            ev.append((r.t, 5, "act", r))
        for r in tb["agent_snapshots"][tb["agent_snapshots"].episode == e].itertuples():
            ev.append((r.grid_index * p.grid_step, 0, "snap", r))
        ev.sort(key=lambda x: (x[0], x[1]))
        active = [1 << f for f in SETUP[setup].agent_facts]
        for _, _, kind, r in ev:
            if kind == "forget":
                assert (active[r.agent] >> r.fact) & 1
                active[r.agent] &= ~(1 << r.fact)
            elif kind == "act":
                assert active[r.agent] == r.active_before
                active[r.agent] |= int(r.facts_read)
                assert active[r.agent] == r.active_after
                assert WORLD.posterior(active[r.agent]) == (r.p_A0, r.p_A1, r.p_A2)
            else:
                assert active[r.agent] == r.active


# ---------------------------------------------------------------- 8, 9. reproducibility

def _same(a: dict, b: dict, which=("posts", "actions", "forgetting", "controller")):
    for k in which:
        pd.testing.assert_frame_equal(a[k], b[k])


def test_same_seed_same_output():
    p = params("S6", "task003_symmetric", "false")
    _same(tables(p), tables(p), which=tuple(COLUMNS))


@pytest.mark.parametrize("step", ["S0", "S3"])
def test_silent_arm_ignores_controller_settings(step):
    base = tables(params(step, "task003_nosolution", "silent"))
    other = tables(params(step, "task003_nosolution", "silent", controller_gate="always", qc=3, budget=5,
                          theta_vote=0.5, silent_when_target_proved=False))
    _same(base, other, which=("posts", "actions", "forgetting", "snapshots"))


def test_silent_s3_equals_silent_s6_settings():
    """Why S4-S6 can reuse S3's silent cells."""
    _same(tables(params("S3", "task003_symmetric", "silent")),
          tables(params("S6", "task003_symmetric", "silent")), which=("posts", "actions", "forgetting", "snapshots"))


@pytest.mark.parametrize("step", ["S0", "S6"])
def test_measurement_grid_does_not_change_the_trajectory(step):
    _same(tables(params(step, "task003_nosolution", "truth")),
          tables(params(step, "task003_nosolution", "truth", grid_step=0.1)))


def test_arms_identical_until_the_first_controller_post():
    s = tables(params("S6", "task003_nosolution", "silent"), [0])["actions"]
    c_tb = tables(params("S6", "task003_nosolution", "false"), [0])
    first = c_tb["posts"][c_tb["posts"].author == CONTROLLER].t.min()
    c = c_tb["actions"]
    cols = ["t", "agent", "facts_read", "vote", "posted_fact"]
    pd.testing.assert_frame_equal(s[s.t <= first][cols].reset_index(drop=True),
                                  c[c.t <= first][cols].reset_index(drop=True))


def test_controller_clock_is_shared_across_rates():
    """Controller opportunities at rate 2 are those at rate 1, at half the times (spec §11)."""
    a = tables(params("S6", "task003_nosolution", "false", lambda_c=1.0, budget=1000), [0])["controller"]
    b = tables(params("S6", "task003_nosolution", "false", lambda_c=2.0, budget=1000), [0])["controller"]
    n = min(len(a), len(b[b.t < 15]))
    assert n > 5
    assert list((a.t[:n] / 2).round(12)) == list(b.t[:n].round(12))


# ---------------------------------------------------------------- settings and bookkeeping

def test_impossible_settings_refuse():
    with pytest.raises(ValueError):
        params("S6", "task003_nosolution", "false", board="cleared_nightly").check()
    with pytest.raises(ValueError):
        params("S6", "task003_nosolution", "false", rho=0.0).check()


def test_votes_gate_with_persistent_board_warns():
    assert params("S1", "task003_nosolution", "false").check()
    assert not params("S0", "task003_nosolution", "false").check()


def test_dirty_check_covers_shared_code_and_ignores_bookkeeping():
    args = dirty_check_args()
    assert ":(exclude)runs_index.csv" in args and ":(exclude)run_summaries" in args
    assert any(a.endswith("llmfree_core") for a in args)
