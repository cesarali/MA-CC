"""Mean-field models M1 and M2 of Simulation 2 (theory programme, phase 2).

Both models replace the 24-agent stochastic system by a large population of agents (24 x R replicas of the setup's
agents) that interact only through a DETERMINISTIC shared board: the board is the expected composition of the W newest
posts, fed by the population's average posting (scaled to 24 agents' posting rate) and by the controller's expected
posting. Each agent uses the simulator's exact rules (imported from sim2.simulation_2): exact posterior, probability-
matching vote, posting rule with support in context / fallback / prefer-not-just-read, and the controller's
deterministic best-fact choice and vote gate.

  M2 (tagged-agent closure): each agent keeps its exact memory set; correlations within an agent's memory (e.g. the
     several facts that together prove A0) are kept; correlations between agents and finite-size effects are dropped.
  M1 (fact-by-fact closure, the plan's eq. M1): identical, except that before each action an agent's remembered set is
     redrawn fact by fact, independently, from the population's current per-fact shares m_f. This destroys all
     joint structure in memory and is the only difference from M2.

Approximations shared by M1 and M2 (documented in theory/M1_M2_mean_field.md):
  (a) infinite population: R replicas of the 24 agents; no extinction of facts, no between-agent correlations;
  (b) deterministic board: readers draw q posts without replacement from their own multinomial realisation of the
      W-post board (finite-board read noise kept, but not shared between readers);
  (c) a reader's own posts are not excluded from its window (they are about 1/24 of the board);
  (d) controller: expected posting rate; at each step its decision probabilities are estimated by sampling q_c agent
      posts from the agent-only window (Monte Carlo); its budget is spent deterministically;
  (e) time step dt (default 0.05): an agent acts with probability dt per step (rate 1), every remembered fact is
      forgotten with probability 1 - rho^dt per step. Exponential lifetimes are memoryless, so renewal on rereading
      needs no bookkeeping: this per-step hazard is exact in the limit dt -> 0.
Post-stop modes from t_control as in the simulator: normal, no_comm, frozen_board, purge_controller_facts.
"""
from __future__ import annotations
import math, pathlib, random, sys
from collections import deque
from dataclasses import dataclass

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1] / "task003_llm_free"                       # shared simulator code and the task003 world
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "simulation_2" / "simulator"))
from llmfree_core.world import World, bits  # noqa: E402,F401
from llmfree_core.kworld import load_world  # noqa: E402
from llmfree_core.setups import load_setup  # noqa: E402
from sim2.simulation_2 import NO_POST, choose_post, vote_probability_matching  # noqa: E402

TASK003 = ROOT / "llmfree_core" / "world_task003.json"
_WORLDS: dict = {}
EPS = 1e-12


def get_world(path):
    path = str(path or TASK003)
    if path not in _WORLDS:
        _WORLDS[path] = load_world(path)
    return _WORLDS[path]


@dataclass
class Spec:
    setup: str                      # "symmetric" | "nosolution" (task003) or a full setup name (e.g. "op_symmetric",
                                    # "task003_symmetric_K2")
    arm: str                        # "silent" | "false"
    rho: float
    q: int
    closure: str = "M2"             # "M2" | "M1" | "M3" (M2 + demographic noise and extinction of fact holder counts)
                                    # | "M3s" (M3 with one stochastic controller instead of the expected one;
                                    #   Phase B attribution of the M3 -> M4 gain)
                                    # | "M4b" (M4 without integer holder counts: an unlimited population of exact
                                    #   agents reading one shared finite board; Phase C1)
                                    # | "M4" (M3 + one shared, realised board of W posts)
    n_swarm: int | None = None      # M3/M4: size of the swarm whose fact holder counts fluctuate (default: its agents)
    world_file: str = ""            # default: task003
    setups_dir: str = ""            # default: task003_llm_free/experimental_setup
    qc: int = 12
    lambda_c: float = 1.0
    gate: bool = True
    budget: float = 30.0
    W: int = 48
    t_control: float = 30.0
    t_end: float = 40.0
    post_stop_mode: str = "normal"
    replicas: int | None = None     # default: 100 for the mean-field board, 1 (N = 24) for the stochastic board
    dt: float = 0.05
    controller_samples: int = 60
    seed: int = 1
    board: str = "mean_field"       # "mean_field" | "stochastic" (diagnostic: real finite board and population)
    pin_own: bool = False           # diagnostic: an agent never forgets its own starting fact (no agent fact can go extinct)


class Model:
    def __init__(self, s: Spec):
        self.s = s
        self.world = get_world(s.world_file)
        name = s.setup if "_" in s.setup else f"task003_{s.setup}"
        st = load_setup(name, s.setups_dir or ROOT / "experimental_setup", self.world)
        self.n_agents = len(st.agent_facts)
        if s.n_swarm is None:
            s.n_swarm = self.n_agents
        self.agent_mask = 0
        for f in st.agent_facts:
            self.agent_mask |= 1 << f
        self.pool = st.pools[2]
        self.ctrl_mask = 0
        for f in self.pool:
            self.ctrl_mask |= 1 << f
        self.ctrl_mask &= ~self.agent_mask
        self.U = sorted(set(bits(self.agent_mask | self.ctrl_mask)))       # facts that can circulate
        self.uidx = {f: i for i, f in enumerate(self.U)}
        nU = len(self.U)
        # board categories: agent posts (fact, vote 0..2) -> 3*i + v; controller posts (fact) -> 3*nU + i
        self.nC = 4 * nU
        self.cat_fact = np.array([self.U[c // 3] if c < 3 * nU else self.U[c - 3 * nU] for c in range(self.nC)])
        self.cat_vote = np.array([c % 3 if c < 3 * nU else -1 for c in range(self.nC)])
        self.stochastic = s.board == "stochastic"
        if s.replicas is None:
            s.replicas = 1 if self.stochastic else 100
        if self.stochastic:
            # finite system of N = 24 R agents; the plan's scaling (section 7): W, controller rate and budget x R
            s.W, s.lambda_c, s.budget = s.W * s.replicas, s.lambda_c * s.replicas, s.budget * s.replicas
            self.posts_list = []        # (category, author), author -1 for the controller
        self.K = self.n_agents * s.replicas
        self.mem = [1 << f for f in st.agent_facts] * s.replicas            # exact memories (python ints)
        self.own = np.array([f for f in st.agent_facts] * s.replicas)
        self.vote = np.full(self.K, -1, dtype=np.int8)
        self.npr = np.random.default_rng(s.seed)
        self.rnd = random.Random(s.seed)
        self.hist = deque()             # (count vector over categories) per step, newest last
        self.frozen = None
        self.messages = 0.0
        self.target = 2 if s.arm == "false" else None
        if s.closure in ("M3", "M3s", "M4"):
            # integer holders of each circulating fact in a swarm of n_swarm agents (the setup's starting holders)
            self.n = np.zeros(len(self.U), dtype=int)
            for f in st.agent_facts:
                self.n[self.uidx[f]] += 1
            self.n = (self.n * s.n_swarm) // self.n_agents
            self.dead = np.zeros(len(self.U), dtype=bool)
        if s.closure in ("M4", "M4b"):
            self.shared = []            # the realised board: (category, is_controller), oldest first

    # ---------------------------------------------------------------- stochastic board (diagnostic)
    def s_window(self, exclude=None, agents_only=False):
        out = []
        for c, a in reversed(self.posts_list):
            if a == exclude and exclude is not None:
                continue
            if agents_only and a == -1:
                continue
            out.append(c)
            if len(out) == self.s.W:
                break
        return out

    def s_read(self, win, k):
        k = min(k, len(win))
        return [win[i] for i in self.npr.choice(len(win), size=k, replace=False)] if k else []

    def s_controller(self, t):
        s = self.s
        if self.target is None or t >= s.t_control or self.messages >= s.budget:
            return
        if self.npr.random() >= s.lambda_c * s.dt:
            return
        win = self.s_window(agents_only=True)
        cats = self.s_read(win, s.qc)
        R = 0
        for c in cats:
            R |= 1 << int(self.cat_fact[c])
        if cats:
            v_hat = float(np.mean([self.cat_vote[c] == self.target for c in cats]))
            if s.gate and v_hat >= 0.75 - EPS:
                return
        best, bv = self.pool[0], None
        for f in self.pool:
            v = self.world.posterior(R | (1 << f))[self.target]
            if bv is None or v > bv + EPS:
                best, bv = f, v
        if bv <= EPS:
            best, bv = self.pool[0], None
            for f in self.pool:
                v = self.world.posterior(1 << f)[self.target]
                if bv is None or v > bv + EPS:
                    best, bv = f, v
        self.posts_list.append((3 * len(self.U) + self.uidx[best], -1))
        self.messages += 1

    # ---------------------------------------------------------------- board
    def window(self, agents_only=False):
        """Expected composition (counts) of the W newest posts (agent posts only for the controller)."""
        tot = np.zeros(self.nC)
        need = float(self.s.W)
        for c in reversed(self.hist):
            v = c.copy()
            if agents_only:
                v[3 * len(self.U):] = 0
            n = v.sum()
            if n <= 0:
                continue
            if n >= need:
                tot += v * (need / n)
                break
            tot += v
            need -= n
        return tot

    def read(self, board, k):
        """k posts without replacement from one multinomial realisation of the expected board."""
        n = board.sum()
        if n < 0.5:
            return np.zeros(0, dtype=int)
        size = int(round(n))
        counts = self.npr.multinomial(size, board / n)
        k = min(k, size)
        picked = self.npr.multivariate_hypergeometric(counts, k)
        return np.repeat(np.arange(self.nC), picked)

    # ---------------------------------------------------------------- controller
    def controller_rate(self, t):
        """Expected posts per fact (over U indices) in this step."""
        s = self.s
        if self.target is None or t >= s.t_control or self.messages >= s.budget:
            return np.zeros(len(self.U))
        board = self.window(agents_only=True)
        out = np.zeros(len(self.U))
        if board.sum() < 0.5:
            return out
        for _ in range(s.controller_samples):
            cats = self.read(board, s.qc)
            R = 0
            for c in cats:
                R |= 1 << int(self.cat_fact[c])
            v_hat = (self.cat_vote[cats] == self.target).mean() if len(cats) else None
            if s.gate and v_hat is not None and v_hat >= 0.75 - EPS:
                continue
            best, bv = self.pool[0], None
            for f in self.pool:
                v = self.world.posterior(R | (1 << f))[self.target]
                if bv is None or v > bv + EPS:
                    best, bv = f, v
            if bv <= EPS:
                best, bv = self.pool[0], None
                for f in self.pool:
                    v = self.world.posterior(1 << f)[self.target]
                    if bv is None or v > bv + EPS:
                        best, bv = f, v
            out[self.uidx[best]] += 1
        rate = out / s.controller_samples * s.lambda_c * s.dt
        room = s.budget - self.messages
        if rate.sum() > room:
            rate *= room / rate.sum()
        self.messages += rate.sum()
        return rate

    def controller_one(self, t):
        """M3s: one stochastic controller on the deterministic board. With probability lambda_c dt it reads q_c agent
        posts from one realisation of the expected agent-only window and posts one whole fact by the simulator's rule
        (gate, best pool fact). Returns posts per fact (over U indices) entering the board in this step."""
        s = self.s
        out = np.zeros(len(self.U))
        if self.target is None or t >= s.t_control or self.messages >= s.budget:
            return out
        if self.npr.random() >= s.lambda_c * s.dt:
            return out
        board = self.window(agents_only=True)
        cats = self.read(board, s.qc) if board.sum() >= 0.5 else np.zeros(0, dtype=int)
        R = 0
        for c in cats:
            R |= 1 << int(self.cat_fact[c])
        if len(cats) and s.gate and (self.cat_vote[cats] == self.target).mean() >= 0.75 - EPS:
            return out
        best, bv = self.pool[0], None
        for f in self.pool:
            v = self.world.posterior(R | (1 << f))[self.target]
            if bv is None or v > bv + EPS:
                best, bv = f, v
        if bv <= EPS:
            best, bv = self.pool[0], None
            for f in self.pool:
                v = self.world.posterior(1 << f)[self.target]
                if bv is None or v > bv + EPS:
                    best, bv = f, v
        out[self.uidx[best]] = 1.0
        self.messages += 1
        return out

    # ---------------------------------------------------------------- one step
    def forget(self):
        if self.s.rho >= 1:
            return
        p = 1 - self.s.rho ** self.s.dt
        mem = self.mem
        lose = self.npr.random((self.K, len(self.U))) < p
        rows, cols = np.nonzero(lose)
        for a, j in zip(rows, cols):
            if self.s.pin_own and self.U[j] == self.own[a]:
                continue
            mem[a] &= ~(1 << self.U[j])

    def marginals(self):
        m = np.zeros(len(self.U))
        for x in self.mem:
            for f in bits(x):
                m[self.uidx[f]] += 1
        return m / self.K

    def step(self, t):
        if self.stochastic:
            return self.s_step(t)
        if self.s.closure in ("M3", "M3s", "M4"):
            before = self.memmat()
            self._step(t)
            return self.m3_noise(before, t)
        return self._step(t)

    def memmat(self):
        """Boolean matrix agents x circulating facts (works for any number of facts)."""
        out = np.zeros((self.K, len(self.U)), dtype=bool)
        for a, x in enumerate(self.mem):
            for f in bits(x):
                j = self.uidx.get(f)
                if j is not None:
                    out[a, j] = True
        return out

    # ---------------------------------------------------------------- M4: one shared, realised board
    def shared_window(self, agents_only=False):
        out = []
        for c, is_ctrl in reversed(self.shared):
            if agents_only and is_ctrl:
                continue
            out.append(c)
            if len(out) == self.s.W:
                break
        return out

    def m4_controller(self, t):
        """One stochastic controller acting on the shared board (the simulator's rule)."""
        s = self.s
        if self.target is None or t >= s.t_control or self.messages >= s.budget:
            return
        if self.npr.random() >= s.lambda_c * s.dt:
            return
        cats = self.s_read(self.shared_window(agents_only=True), s.qc)
        R = 0
        for c in cats:
            R |= 1 << int(self.cat_fact[c])
        if cats and s.gate and float(np.mean([self.cat_vote[c] == self.target for c in cats])) >= 0.75 - EPS:
            return
        best, bv = self.pool[0], None
        for f in self.pool:
            v = self.world.posterior(R | (1 << f))[self.target]
            if bv is None or v > bv + EPS:
                best, bv = f, v
        if bv <= EPS:
            best, bv = self.pool[0], None
            for f in self.pool:
                v = self.world.posterior(1 << f)[self.target]
                if bv is None or v > bv + EPS:
                    best, bv = f, v
        self.shared.append((3 * len(self.U) + self.uidx[best], True))
        self.messages += 1

    def m3_noise(self, before, t):
        """M3: replace the population's deterministic change in the holders of each fact by an integer change in a
        swarm of n_swarm agents. Per step, each holder loses the fact and each non-holder gains it with the
        probabilities measured in the population (binomial draws on the swarm counts); the population is then forced
        to the new count. A fact with no holder and less than half a copy on the board is extinct for good, unless the
        controller can still post it (pool fact while the controller is active)."""
        s, K = self.s, self.K
        after = self.memmat()
        board = None
        controller_active = self.target is not None and t < s.t_control and self.messages < s.budget
        pool = set(self.pool)
        for j, f in enumerate(self.U):
            if self.dead[j]:
                continue
            b0, b1 = before[:, j], after[:, j]
            h0 = int(b0.sum())
            p_loss = int((b0 & ~b1).sum()) / h0 if h0 else 0.0
            p_gain = int((~b0 & b1).sum()) / (K - h0) if K > h0 else 0.0
            n = int(self.n[j])
            n_new = n - self.npr.binomial(n, min(p_loss, 1.0)) + self.npr.binomial(s.n_swarm - n, min(p_gain, 1.0))
            self.n[j] = n_new
            target = int(round(n_new / s.n_swarm * K))
            idx = np.nonzero(b1)[0] if target < b1.sum() else np.nonzero(~b1)[0]
            k = abs(target - int(b1.sum()))
            if k:
                for a in self.npr.choice(idx, size=min(k, len(idx)), replace=False):
                    if target < b1.sum():
                        self.mem[int(a)] &= ~(1 << f)
                    else:
                        self.mem[int(a)] |= 1 << f
            if n_new == 0 and not (controller_active and f in pool):
                if s.closure == "M4":
                    copies = sum(1 for c, _ in self.shared[-s.W:] if int(self.cat_fact[c]) == f)
                else:
                    if board is None:
                        board = self.window()
                    copies = board[self.cat_fact == f].sum()
                if copies < 0.5:
                    self.dead[j] = True
                    for c in self.hist:
                        c[self.cat_fact == f] = 0
                    if s.closure == "M4":
                        self.shared = [(c, k) for c, k in self.shared if int(self.cat_fact[c]) != f]

    def _step(self, t):
        s = self.s
        stopped = t >= s.t_control
        if stopped and s.post_stop_mode == "purge_controller_facts" and self.frozen is None:
            self.mem = [x & ~self.ctrl_mask for x in self.mem]
            for c in self.hist:
                c[(np.isin(self.cat_fact, list(bits(self.ctrl_mask))))] = 0
            if s.closure in ("M3", "M3s", "M4"):
                for f in bits(self.ctrl_mask):
                    self.n[self.uidx[f]] = 0
            if s.closure in ("M4", "M4b"):
                self.shared = [(c, k) for c, k in self.shared if not (self.ctrl_mask >> int(self.cat_fact[c])) & 1]
            self.frozen = "done"
        if stopped and s.post_stop_mode == "frozen_board" and self.frozen is None:
            self.frozen = self.shared_window() if s.closure in ("M4", "M4b") else self.window()
        self.forget()
        m4 = s.closure in ("M4", "M4b")
        if stopped and s.post_stop_mode == "frozen_board":
            board = self.frozen
        else:
            board = self.shared_window() if m4 else self.window()
        posts = np.zeros(self.nC)
        acting = np.nonzero(self.npr.random(self.K) < s.dt)[0]
        m = self.marginals() if s.closure == "M1" else None
        no_read = stopped and s.post_stop_mode == "no_comm"
        no_post = stopped and s.post_stop_mode in ("no_comm", "frozen_board")
        for a in acting:
            a = int(a)
            if m is not None:                                       # M1: redraw memory fact by fact
                draw = self.npr.random(len(self.U)) < m
                x = 0
                for j in np.nonzero(draw)[0]:
                    x |= 1 << self.U[j]
                self.mem[a] = x
            read_now = 0
            if not no_read:
                for c in (self.s_read(board, s.q) if m4 else self.read(board, s.q)):
                    read_now |= 1 << int(self.cat_fact[c])
            self.mem[a] |= read_now
            P = self.world.posterior(self.mem[a])
            v = vote_probability_matching(P, self.rnd)
            self.vote[a] = v
            if not no_post:
                f = choose_post(self.world, self.mem[a], v, read_now, True, self.rnd)[0]
                if f != NO_POST:
                    posts[3 * self.uidx[f] + v] += 1
        if m4:
            # the realised board receives each population post with probability n_agents / K (the swarm's posting
            # stream), in random order, and the posts of one stochastic controller
            new = []
            for c in np.nonzero(posts)[0]:
                new += [int(c)] * int(self.npr.binomial(int(posts[c]), self.n_agents / self.K))
            self.npr.shuffle(new)
            self.shared += [(c, False) for c in new]
            self.m4_controller(t)
            if len(self.shared) > 20 * s.W:
                self.shared = self.shared[-4 * s.W:]
            return
        posts *= self.n_agents / self.K                              # scale to the swarm's posting
        u = self.controller_one(t) if s.closure == "M3s" else self.controller_rate(t)
        posts[3 * len(self.U):] += u
        if not no_post or u.sum() > 0:
            self.hist.append(posts)
            if len(self.hist) > 4000:
                self.hist.popleft()

    def s_step(self, t):
        s = self.s
        stopped = t >= s.t_control
        if stopped and s.post_stop_mode == "purge_controller_facts" and self.frozen is None:
            self.mem = [x & ~self.ctrl_mask for x in self.mem]
            self.posts_list = [(c, a) for c, a in self.posts_list if not (self.ctrl_mask >> int(self.cat_fact[c])) & 1]
            self.frozen = "done"
        if stopped and s.post_stop_mode == "frozen_board" and self.frozen is None:
            self.frozen = list(self.posts_list)
        self.forget()
        self.s_controller(t)
        no_read = stopped and s.post_stop_mode == "no_comm"
        no_post = stopped and s.post_stop_mode in ("no_comm", "frozen_board")
        for a in np.nonzero(self.npr.random(self.K) < s.dt)[0]:
            a = int(a)
            read_now = 0
            if not no_read:
                if stopped and s.post_stop_mode == "frozen_board":
                    saved, self.posts_list = self.posts_list, self.frozen
                    win = self.s_window(exclude=a)
                    self.posts_list = saved
                else:
                    win = self.s_window(exclude=a)
                for c in self.s_read(win, s.q):
                    read_now |= 1 << int(self.cat_fact[c])
            self.mem[a] |= read_now
            P = self.world.posterior(self.mem[a])
            v = vote_probability_matching(P, self.rnd)
            self.vote[a] = v
            if not no_post:
                f = choose_post(self.world, self.mem[a], v, read_now, True, self.rnd)[0]
                if f != NO_POST:
                    self.posts_list.append((3 * self.uidx[f] + v, a))

    # ---------------------------------------------------------------- run
    def run(self, grid_step=0.25):
        s = self.s
        out = []
        n = int(round(s.t_end / s.dt))
        every = int(round(grid_step / s.dt))
        for k in range(n + 1):
            t = k * s.dt
            if k % every == 0:
                out.append(self.snapshot(t))
            if k < n:
                self.step(t)
        return out

    def snapshot(self, t):
        voted = self.vote >= 0
        x = [float((self.vote[voted] == a).mean()) if voted.any() else 0.0 for a in range(3)]
        p = np.array([self.world.posterior(m) for m in self.mem]).mean(axis=0)
        if self.stochastic:
            board = np.bincount(self.s_window(), minlength=self.nC).astype(float)
        elif self.s.closure in ("M4", "M4b"):
            win = self.frozen if isinstance(self.frozen, list) else self.shared_window()
            board = np.bincount(win, minlength=self.nC).astype(float)
        else:
            board = self.frozen if isinstance(self.frozen, np.ndarray) else self.window()
        bf = np.zeros(len(self.U))
        if board.sum() > 0:
            for c in range(self.nC):
                bf[self.uidx[int(self.cat_fact[c])]] += board[c]
            bf /= board.sum()
        return dict(t=round(t, 6), x0=x[0], x1=x[1], x2=x[2], share_voted=float(voted.mean()), p0=p[0], p2=p[2],
                    messages=self.messages, m=self.marginals(), b=bf,
                    ctrl_b=float(sum(bf[self.uidx[f]] for f in bits(self.ctrl_mask))),
                    ctrl_m=float(sum(self.marginals()[self.uidx[f]] for f in bits(self.ctrl_mask))))
