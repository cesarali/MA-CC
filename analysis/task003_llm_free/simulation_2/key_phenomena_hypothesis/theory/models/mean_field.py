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
ROOT = HERE.parents[3]                                            # analysis/task003_llm_free
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "simulation_2" / "simulator"))
from llmfree_core.world import World, bits  # noqa: E402
from llmfree_core.setups import load_setup  # noqa: E402
from sim2.simulation_2 import NO_POST, choose_post, vote_probability_matching  # noqa: E402

WORLD = World(ROOT / "llmfree_core" / "world_task003.json")
EPS = 1e-12


@dataclass
class Spec:
    setup: str                      # "symmetric" | "nosolution"
    arm: str                        # "silent" | "false"
    rho: float
    q: int
    closure: str = "M2"             # "M2" | "M1"
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
        st = load_setup(f"task003_{s.setup}", ROOT / "experimental_setup", WORLD)
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
        self.K = 24 * s.replicas
        self.mem = [1 << f for f in st.agent_facts] * s.replicas            # exact memories (python ints)
        self.own = np.array([f for f in st.agent_facts] * s.replicas)
        self.vote = np.full(self.K, -1, dtype=np.int8)
        self.npr = np.random.default_rng(s.seed)
        self.rnd = random.Random(s.seed)
        self.hist = deque()             # (count vector over categories) per step, newest last
        self.frozen = None
        self.messages = 0.0
        self.target = 2 if s.arm == "false" else None

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
            v = WORLD.posterior(R | (1 << f))[self.target]
            if bv is None or v > bv + EPS:
                best, bv = f, v
        if bv <= EPS:
            best, bv = self.pool[0], None
            for f in self.pool:
                v = WORLD.posterior(1 << f)[self.target]
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
                v = WORLD.posterior(R | (1 << f))[self.target]
                if bv is None or v > bv + EPS:
                    best, bv = f, v
            if bv <= EPS:
                best, bv = self.pool[0], None
                for f in self.pool:
                    v = WORLD.posterior(1 << f)[self.target]
                    if bv is None or v > bv + EPS:
                        best, bv = f, v
            out[self.uidx[best]] += 1
        rate = out / s.controller_samples * s.lambda_c * s.dt
        room = s.budget - self.messages
        if rate.sum() > room:
            rate *= room / rate.sum()
        self.messages += rate.sum()
        return rate

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
        s = self.s
        stopped = t >= s.t_control
        if stopped and s.post_stop_mode == "purge_controller_facts" and self.frozen is None:
            self.mem = [x & ~self.ctrl_mask for x in self.mem]
            for c in self.hist:
                c[(np.isin(self.cat_fact, list(bits(self.ctrl_mask))))] = 0
            self.frozen = "done"
        if stopped and s.post_stop_mode == "frozen_board" and self.frozen is None:
            self.frozen = self.window()
        self.forget()
        if stopped and s.post_stop_mode == "frozen_board":
            board = self.frozen
        else:
            board = self.window()
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
                for c in self.read(board, s.q):
                    read_now |= 1 << int(self.cat_fact[c])
            self.mem[a] |= read_now
            P = WORLD.posterior(self.mem[a])
            v = vote_probability_matching(P, self.rnd)
            self.vote[a] = v
            if not no_post:
                f = choose_post(WORLD, self.mem[a], v, read_now, True, self.rnd)[0]
                if f != NO_POST:
                    posts[3 * self.uidx[f] + v] += 1
        posts *= 24.0 / self.K                                       # scale to 24 agents' posting
        u = self.controller_rate(t)
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
            P = WORLD.posterior(self.mem[a])
            v = vote_probability_matching(P, self.rnd)
            self.vote[a] = v
            if not no_post:
                f = choose_post(WORLD, self.mem[a], v, read_now, True, self.rnd)[0]
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
        p = np.array([WORLD.posterior(m) for m in self.mem]).mean(axis=0)
        if self.stochastic:
            board = np.bincount(self.s_window(), minlength=self.nC).astype(float)
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
