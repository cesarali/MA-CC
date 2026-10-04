"""Mean-field theory for the exact game, and its numerical solution.

State variable: m_f(t), the fraction of agents holding fact f ACTIVE at round t.
This is the natural mean field here because the game's only epistemic dynamics
are decay and exposure-activation, both of which act per (agent, fact).

Derivation (one round, in the order the runtime applies them):

  1. decay          every active fact survives with probability rho
                        m_f  ->  rho * m_f
  2. board          each agent posts one fact it holds and that supports its
                    vote; the controller posts its b facts. The expected number
                    of live copies of f is
                        B_f = N * pi_f + c_f
                    with pi_f the probability a random agent chooses f.
  3. exposure       an agent reads the board and every fact it sees becomes
                    active. With `full` sampling it reads everything, so
                        u_f = 1{B_f >= 1}
                    With `uniform` sampling of q of n messages, u_f is the
                    hypergeometric probability of seeing at least one copy,
                        u_f = 1 - C(n - k_f, q) / C(n, q)
                    which is exactly Eq. (7) of the task004 paper.
  4. combine        a fact is active after the round if it survived decay or
                    was (re)activated by exposure:
                        m_f(t+1) = rho*m_f + (1 - rho*m_f) * u_f

**There is no free parameter.** rho, N, q and b are game settings; pi_f and u_f
are determined by them. Nothing is fitted.

The population coordinate follows by averaging the exact posterior over the
product measure implied by the m_f:

    M(t) = E_{K ~ prod_f Bernoulli(m_f)} [ P(A0 | K) ]

evaluated by Monte Carlo over K, because P(A0 | K) is strongly nonlinear in K
and no product-form closure of it exists.
"""
from __future__ import annotations
import sys
from math import comb
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "10_task_and_facts"))
from engine import World

PRIOR = 1.0 / 3.0


def hyper_read(n: int, k: int, q: int) -> float:
    """P(at least one of k marked items in a sample of q from n)."""
    if n <= 0 or k <= 0:
        return 0.0
    q = min(q, n)
    if n - k < q:
        return 1.0
    return 1.0 - comb(n - k, q) / comb(n, q)


class MeanField:
    def __init__(self, world: World, agent_facts: list[str], pool: list[str],
                 N: int = 24, rho: float = 0.75, b: int = 3,
                 sampling: str = "full", q: int = 23,
                 target: int | None = None, mc: int = 4000, seed: int = 0):
        self.w = world
        self.idx = {f.fact_id: i for i, f in enumerate(world.facts)}
        self.facts = sorted(set(agent_facts) | set(pool))
        self.agent_facts = agent_facts          # multiset: one per agent slot
        self.pool = list(pool)
        self.N, self.rho, self.b = N, rho, b
        self.sampling, self.q = sampling, q
        self.target = target
        self.rng = np.random.default_rng(seed)
        self.mc = mc
        # initial occupancy: fraction of agents starting with each fact active
        self.m = np.array([agent_facts.count(f) / N for f in self.facts])

    # -- the coordinate ----------------------------------------------------

    def M(self) -> float:
        """E[P(A0|K)] over K ~ prod Bernoulli(m_f). Monte Carlo over K."""
        draws = self.rng.random((self.mc, len(self.facts))) < self.m
        tot = 0.0
        for row in draws:
            ids = [self.idx[f] for f, on in zip(self.facts, row) if on]
            tot += self.w.posterior(ids)[0] if ids else PRIOR
        return tot / self.mc

    # -- one round ---------------------------------------------------------

    def step(self) -> None:
        # 1. decay
        m = self.rho * self.m

        # 2. board. An agent posts a fact it holds; approximate the choice as
        #    uniform over its held facts, so the expected number of copies of f
        #    is N * m_f / (expected number held). Controller adds its b facts.
        held = max(m.sum(), 1e-9)
        pi = m / held
        Bf = self.N * pi
        if self.target is not None:
            ctl = self._controller_pick()
            for f in ctl:
                Bf[self.facts.index(f)] += 1.0

        # 3. exposure
        n = Bf.sum()
        if self.sampling == "full":
            u = np.clip(Bf, 0.0, 1.0)              # sees every live message
        else:
            u = np.array([hyper_read(int(round(n)), int(round(k)), self.q)
                          for k in Bf])

        # 4. combine
        self.m = m + (1.0 - m) * u

    def _controller_pick(self) -> list[str]:
        """Oracle: the b pool facts that most raise the target in expectation."""
        if self.target is None:
            return []
        gains = []
        for f in self.pool:
            j = self.facts.index(f)
            gains.append((-(1.0 - self.m[j]), f))   # most absent first
        gains.sort()
        return [f for _, f in gains[: self.b]]

    def run(self, rounds: int = 30) -> list[float]:
        out = [self.M()]
        for _ in range(rounds - 1):
            self.step()
            out.append(self.M())
        return out


# ---------------------------------------------------------------------------
# Version 2: heterogeneous agents and sequential within-round exposure
# ---------------------------------------------------------------------------


class MeanFieldV2:
    """Per-agent occupancies m[i,f], with the within-round build-up of the board.

    Two corrections to `MeanField`, each fixing a specific, identified error:

    **Heterogeneity.** Agents are not exchangeable: each starts with its own
    single fact. Averaging them into one occupancy vector smears the exact
    initial condition into a Poisson-like one that allows an agent to hold no
    fact at all (probability about e^-1), and P(A0 | empty) is the prior, which
    drags M(0) down by roughly 0.4. Tracking m[i,f] per agent removes this.

    **Sequential exposure.** The runtime updates agents one at a time and the
    board *builds up within the round*: the agent in position k reads only the
    k posts made before it, so position 0 reads nothing. A synchronous mean
    field instead lets every posted fact reach every agent in the same round,
    which with `full` sampling drives every occupancy to 1 after one round.
    Averaging the exposure over positions gives, for a per-post probability
    pi_f,

        ubar_f = (1/N) sum_{k=0}^{N-1} [1 - (1-pi_f)^k]
               = 1 - (1 - (1-pi_f)^N) / (N * pi_f)

    a closed form, and it is the damping the first version was missing.
    """

    def __init__(self, world: World, agent_facts: dict[str, list[str]],
                 pool: list[str], rho: float = 0.75, b: int = 3,
                 target: int | None = None, mc: int = 800, seed: int = 0):
        self.w = world
        self.idx = {f.fact_id: i for i, f in enumerate(world.facts)}
        self.agents = sorted(agent_facts)
        self.facts = sorted({f for v in agent_facts.values() for f in v} | set(pool))
        self.fi = {f: j for j, f in enumerate(self.facts)}
        self.N = len(self.agents)
        self.rho, self.b, self.target = rho, b, target
        self.pool = list(pool)
        self.rng = np.random.default_rng(seed)
        self.mc = mc
        self.m = np.zeros((self.N, len(self.facts)))
        for i, a in enumerate(self.agents):
            for f in agent_facts[a]:
                self.m[i, self.fi[f]] = 1.0

    def M(self) -> float:
        tot = 0.0
        for i in range(self.N):
            draws = self.rng.random((self.mc, len(self.facts))) < self.m[i]
            s = 0.0
            for row in draws:
                ids = [self.idx[f] for f, on in zip(self.facts, row) if on]
                s += self.w.posterior(ids)[0] if ids else PRIOR
            tot += s / self.mc
        return tot / self.N

    def step(self) -> None:
        m = self.rho * self.m                       # 1. decay

        # 2. posting: agent i posts one of its held facts, roughly uniformly
        held = np.maximum(m.sum(axis=1, keepdims=True), 1e-9)
        post = m / held                             # P(agent i posts fact f)
        pi = post.mean(axis=0)                      # per-post marginal

        # controller posts land on the board before any agent reads
        ctl = np.zeros(len(self.facts))
        if self.target is not None:
            # oracle rule, matching ControllerPolicy(rule="greedy_posterior"):
            # the pool facts whose marginal lift toward the target is largest,
            # weighted by how many agents do not yet hold them.
            base = self.w.posterior([])[self.target] if False else PRIOR
            scored = []
            for f in self.pool:
                j = self.fi[f]
                lift = self.w.posterior([self.idx[f]])[self.target] - base
                scored.append((-lift * (1.0 - m[:, j].mean()), j))
            scored.sort()
            for _, j in scored[: self.b]:
                ctl[j] = 1.0

        # 3. sequential exposure, averaged over reading position
        with np.errstate(divide="ignore", invalid="ignore"):
            ubar = np.where(
                pi > 1e-12,
                1.0 - (1.0 - (1.0 - pi) ** self.N) / (self.N * np.maximum(pi, 1e-12)),
                0.0,
            )
        # a controller post is on the board from the start, so every agent sees it
        u = 1.0 - (1.0 - ubar) * (1.0 - ctl)

        # 4. combine
        self.m = m + (1.0 - m) * u[None, :]

    def run(self, rounds: int = 30) -> list[float]:
        """The simulation records e AFTER each round's updates, so step first."""
        out = []
        for _ in range(rounds):
            self.step()
            out.append(self.M())
        return out
