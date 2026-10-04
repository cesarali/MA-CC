"""An LLM-free mirror of the blackboard game, exact to the arithmetic.

Everything the real runtime does to *mechanism* is reproduced here. The only
thing replaced is the agent's and controller's **decision**, which in the real
game is a language model reading a prompt and here is the exact posterior over
allocations computed from the 14,388-world enumeration.

No natural language, no fact weights, no logistic noise, no fitted temperature:
a fact is an id, a belief is a count of surviving worlds.

Mechanism reproduced from
``src/mas_cc/games/relational_reasoning/imitation_round_feedback/``:

* ``apply_epistemic_persistence`` -- at each round boundary every **active**
  fact survives independently with probability rho. ``known`` is untouched.
* ``game.py`` exposure rule -- a fact cited in a message the focal agent reads
  becomes **active**: added to ``known`` if new, merely reactivated if it had
  decayed. This is the whole propagation mechanism, and it is why the mean
  posterior rises rather than decaying to the prior.
* sequential within-round updating -- agents are updated one at a time and
  **see posts made earlier in the same round**. This is not a detail: it is the
  source of the within-round ordering artifact that invalidated an earlier
  dose-response result.
* ``board_sampling`` full or uniform of ``social_group_size``, with
  ``exclude_self_authored``.
* ``message_lifetime_rounds`` expiry.
* controller at dawn only, soft-target gate, per-round budget, the
  least-recently-used ranking of ``select_truthful_reports``.

What is a modelling *choice* here, because in the real game a model makes it,
is marked CHOICE in the code and collected in ``AgentPolicy`` /
``ControllerPolicy`` so it can be varied and reported.
"""

from __future__ import annotations

import hashlib
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

_ENGINE = Path(__file__).resolve().parent.parent / "10_task_and_facts"
if str(_ENGINE) not in sys.path:
    sys.path.insert(0, str(_ENGINE))
from engine import World  # noqa: E402

PRIOR = 1.0 / 3.0


# ---------------------------------------------------------------------------
# policies: the only places a language model is replaced by arithmetic
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentPolicy:
    """How an agent turns its active facts into a vote and a post."""

    # CHOICE. "bayes" = argmax of the exact posterior, no noise at all.
    # "softmax" adds one temperature, for sensitivity checks only.
    vote_rule: str = "bayes"
    beta: float = 0.0

    # CHOICE. Which fact to cite when posting.
    #   "vote_aligned"  -- uniform among own active facts that support the vote
    #                      (positive lift), falling back to any active fact.
    #                      Argue for what you believe, but do not optimise.
    #   "vote_aligned_novel" -- as above, preferring facts not already live on
    #                      the board. Closest to "say something relevant and new".
    #   "random_active" -- uniform among own active facts. The null.
    #   "argmax_aligned" -- the globally strongest supporting fact. DEGENERATE:
    #                      every agent converges on the same fact and the board
    #                      carries one distinct fact forever. Kept only to make
    #                      that failure reproducible.
    post_rule: str = "vote_aligned_novel"


@dataclass(frozen=True)
class ControllerPolicy:
    """How the controller selects facts. Target is set per arm."""

    target: int | None = None          # 0 or 2; None = silent arm
    budget: int = 3                    # b, per round
    threshold: float = 0.5
    gate: str = "always"               # "always" or "soft"
    beta: float = 4.0
    cooldown: int = 0
    # CHOICE. "lru" reproduces select_truthful_reports exactly: a
    # least-recently-used rotation in which the target only breaks ties.
    # "greedy_posterior" is the oracle: pick the facts that most raise the
    # target given what the population actually holds now.
    rule: str = "lru"


@dataclass(frozen=True)
class GameRules:
    rounds: int = 30
    rho: float = 0.75
    board_sampling: str = "full"       # "full" or "uniform"
    social_group_size: int = 23
    message_lifetime_rounds: int = 1
    exclude_self_authored: bool = True


@dataclass
class Message:
    author: str
    author_kind: str                   # "agent" or "controller"
    fact_id: str
    round_posted: int


@dataclass
class Agent:
    agent_id: str
    known: set[str]
    active: set[str]
    vote: int | None = None
    observed: set[str] = field(default_factory=set)


# ---------------------------------------------------------------------------
# the game
# ---------------------------------------------------------------------------


class ExactGame:
    def __init__(
        self,
        world: World,
        assignment: dict[str, list[str]],
        pool: list[str],
        rules: GameRules,
        agent_policy: AgentPolicy,
        controller_policy: ControllerPolicy,
    ) -> None:
        self.w = world
        self.idx = {f.fact_id: i for i, f in enumerate(world.facts)}
        self.assignment = assignment
        self.pool = list(pool)
        self.rules = rules
        self.ap = agent_policy
        self.cp = controller_policy
        self._cache: dict[frozenset[str], tuple[float, float, float]] = {}

    # -- exact belief ------------------------------------------------------

    def posterior(self, facts: set[str]) -> tuple[float, float, float]:
        """P(A0, A1, A2 | facts), an exact count over surviving worlds."""
        key = frozenset(facts)
        hit = self._cache.get(key)
        if hit is None:
            hit = (
                (PRIOR, PRIOR, PRIOR)
                if not facts
                else self.w.posterior([self.idx[f] for f in facts])
            )
            self._cache[key] = hit
        return hit

    # -- agent decisions ---------------------------------------------------

    def _vote(self, agent: Agent, rng: random.Random) -> int:
        p = self.posterior(agent.active)
        if self.ap.vote_rule == "bayes":
            best = max(p)
            winners = [k for k in range(3) if p[k] == best]
            # deterministic tie-break, so a tie does not inject noise
            return winners[0] if len(winners) == 1 else rng.choice(winners)
        if self.ap.vote_rule == "softmax_prob":
            # Darius's task004 rule, Eq. (8): softmax over the PROBABILITIES
            # themselves, not their logs, with beta = 20. This is not
            # "artificial confusion": without it the argmax agent is
            # deterministic and the coarse-grained chain collapses onto
            # absorbing states, which an LLM population does not do.
            import math

            z = [self.ap.beta * p[k] for k in range(3)]
            m = max(z)
            e = [math.exp(v - m) for v in z]
            tot = sum(e)
            r = rng.random() * tot
            acc = 0.0
            for k in range(3):
                acc += e[k]
                if r <= acc:
                    return k
            return 2
        if self.ap.vote_rule == "softmax":
            import math

            z = [self.ap.beta * math.log(max(p[k], 1e-12)) for k in range(3)]
            m = max(z)
            e = [math.exp(v - m) for v in z]
            tot = sum(e)
            r = rng.random() * tot
            acc = 0.0
            for k in range(3):
                acc += e[k]
                if r <= acc:
                    return k
            return 2
        raise ValueError(f"unknown vote_rule {self.ap.vote_rule!r}")

    def _post(
        self, agent: Agent, rng: random.Random, live: frozenset[str] = frozenset()
    ) -> str | None:
        """Which fact the agent cites. Grounded: it must hold the fact.

        An agent reports from its own active knowledge. The real game permits
        citing an observed fact too (``report_citation_scope:
        active_or_observed``), but an agent that cites whatever is already on
        the board adds no information, so own-knowledge is the faithful default
        and is what makes facts propagate.
        """
        own = sorted(agent.active)
        if not own:
            return None                       # no_citable_fact_action: none
        rule = self.ap.post_rule

        if rule == "random_active":
            return rng.choice(own)

        if rule == "argmax_aligned":
            # DEGENERATE on purpose; see AgentPolicy.
            k = agent.vote if agent.vote is not None else 0
            base = self.posterior(agent.active)[k]
            return max(own, key=lambda f: self.posterior(agent.active | {f})[k] - base)

        if rule in ("vote_aligned", "vote_aligned_novel"):
            k = agent.vote if agent.vote is not None else 0
            # a fact supports the vote if, read alone, it raises that allocation
            supporting = [f for f in own if self.posterior({f})[k] > PRIOR + 1e-12]
            pick_from = supporting or own
            if rule == "vote_aligned_novel":
                fresh = [f for f in pick_from if f not in live]
                pick_from = fresh or pick_from
            return rng.choice(pick_from)

        raise ValueError(f"unknown post_rule {self.ap.post_rule!r}")

    # -- controller --------------------------------------------------------

    def _controller_fires(self, board: list[Message], agents: list[Agent]) -> bool:
        if self.cp.target is None:
            return False
        if self.cp.gate == "always":
            return True
        # soft_target gate on the sensed population state
        m = sum(self.posterior(a.active)[self.cp.target] for a in agents) / len(agents)
        import math

        return random.random() < 1.0 / (1.0 + math.exp(-self.cp.beta * (self.cp.threshold - m)))

    def _controller_facts(
        self,
        round_index: int,
        episode_seed: int,
        task_id: str,
        live_counts: dict[str, int],
        selected_rounds: dict[str, list[int]],
        agents: list[Agent],
    ) -> list[str]:
        if self.cp.rule == "greedy_posterior":
            # oracle: expected shift in the population's posterior toward target
            scored = []
            for f in self.pool:
                gain = 0.0
                for a in agents:
                    base = self.posterior(a.active)[self.cp.target]
                    gain += self.posterior(a.active | {f})[self.cp.target] - base
                scored.append((-gain / len(agents), f))
            scored.sort()
            return [f for _, f in scored[: self.cp.budget]]

        # "lru": reproduce select_truthful_reports' lexicographic ranking.
        ranked = []
        for f in self.pool:
            prior = selected_rounds.get(f, [])
            cool_ok = (not prior) or (round_index - max(prior) >= self.cp.cooldown)
            lc = live_counts.get(f, 0)
            # base_score is 0 under a balanced/neutral pool, matching the runtime
            base_score = 0.0
            tie = hashlib.sha256(
                f"{episode_seed}:{task_id}:{round_index}:{f}".encode()
            ).hexdigest()
            ranked.append(((not cool_ok, len(prior), lc, -base_score, tie, f), f))
        ranked.sort()
        return [f for _, f in ranked[: self.cp.budget]]

    # -- one episode -------------------------------------------------------

    def run_episode(self, episode_seed: int, task_id: str = "task003") -> list[dict]:
        r = self.rules
        rng = random.Random(episode_seed)
        agents = [
            Agent(aid, known=set(facts), active=set(facts))
            for aid, facts in sorted(self.assignment.items())
        ]
        for a in agents:
            a.vote = self._vote(a, rng)
        board: list[Message] = []
        selected_rounds: dict[str, list[int]] = {}
        rows: list[dict] = []

        for t in range(r.rounds):
            # 1. dawn persistence: active facts decay, known is untouched
            if r.rho < 1.0:
                for a in agents:
                    a.active = {f for f in sorted(a.active) if rng.random() < r.rho}

            # 2. two buffers, matching the runtime. `prev` is last round's
            #    board -- what the controller senses at dawn
            #    (``live_messages(round_index - 1)``). `board` is this round's,
            #    which the agents read as it grows
            #    (``live_messages(round_index)``). With
            #    message_lifetime_rounds = 1, last round's board is gone for the
            #    agents by the time they act.
            prev, board = board, []

            # 3. controller at dawn, sensing the previous board
            n_ctl = 0
            if self._controller_fires(prev, agents):
                live = {}
                for m in prev:
                    live[m.fact_id] = live.get(m.fact_id, 0) + 1
                picks = self._controller_facts(
                    t, episode_seed, task_id, live, selected_rounds, agents
                )
                for f in picks:
                    board.append(Message("controller", "controller", f, t))
                    selected_rounds.setdefault(f, []).append(t)
                n_ctl = len(picks)

            # 4. agents update SEQUENTIALLY and see earlier posts this round
            exposures = 0
            for a in agents:
                eligible = [
                    m
                    for m in board
                    if not r.exclude_self_authored or m.author != a.agent_id
                ]
                if r.board_sampling == "full":
                    sampled = eligible
                else:
                    k = min(r.social_group_size, len(eligible))
                    sampled = rng.sample(eligible, k) if k else []

                # exposure -> activation (game.py rule)
                for m in sampled:
                    a.observed.add(m.fact_id)
                    if m.fact_id not in a.known:
                        a.known.add(m.fact_id)
                        a.active.add(m.fact_id)
                        exposures += 1
                    elif m.fact_id not in a.active:
                        a.active.add(m.fact_id)
                        exposures += 1

                a.vote = self._vote(a, rng)
                f = self._post(a, rng, frozenset(m.fact_id for m in board))
                if f is not None:
                    board.append(Message(a.agent_id, "agent", f, t))

            # 5. record
            post = [self.posterior(a.active) for a in agents]
            e = sum(p[0] for p in post) / len(agents)
            votes = [a.vote for a in agents]
            rows.append(
                {
                    "round_index": t,
                    "e": e,
                    "share_A0": votes.count(0) / len(votes),
                    "share_A1": votes.count(1) / len(votes),
                    "share_A2": votes.count(2) / len(votes),
                    "mean_nfacts": sum(len(a.active) for a in agents) / len(agents),
                    "controller_posts": n_ctl,
                    "activations": exposures,
                    "board_size": len(board),
                }
            )
        return rows
