"""Simulation 1: days and nights, per-round controller budget.

Implements ../simulation_1_spec.md. Section numbers (§) below refer to it.
"""
from __future__ import annotations
import hashlib, itertools, random
from dataclasses import dataclass, asdict

from .setups import TARGETS, Setup
from .world import World, bits

EPS = 1e-12
CONTROLLER = -1                      # author id of controller posts
CONTROLLER_CODE = 24                 # author code used in the posts_read lists
NO_POST = -1

# why an agent posted what it posted (agents table, column post_reason)
POST_REASONS = {0: "in_context", 1: "fallback", 2: "random", 3: "uniform", 4: "abstained", 5: "no_memory"}
IN_CONTEXT, FALLBACK, RANDOM, UNIFORM, ABSTAINED, NO_MEMORY = range(6)


def encode_post(author: int, fact: int) -> int:
    """A post read, as one small int: author * 64 + fact. The controller's author code is 24.
    Example: agent 3 posting fact 17 -> 209."""
    return (CONTROLLER_CODE if author == CONTROLLER else author) * 64 + fact


def decode_post(code: int) -> tuple[int, int]:
    """Inverse of encode_post: (author code, fact index)."""
    return code // 64, code % 64


@dataclass(frozen=True)
class Params:
    # the grid (§3)
    setup: str
    arm: str                          # silent | truth | false
    q: int
    qc: int | None                    # None for the silent arm
    b: int | None                     # None for the silent arm
    rho: float
    # fixed settings (§3)
    M: int = 30
    controller_acts_on_final_night: bool = False
    is_board_cleared: bool = True
    board_read_weighting: str = "uniform"
    read_posts_not_facts: bool = True
    agent_sampling_mode: str = "argmax"
    beta: float = 4.0
    agent_post_rule: str = "supporting"
    agent_post_always: bool = False
    support_rule: str = "in_context"
    support_fallback: str = "read_alone"
    prefer_facts_not_read_today: bool = True
    supporting_fact_choice: str = "uniform"
    controller_gate: str = "votes"
    theta_vote: float = 0.75
    theta_facts: float = 0.8
    silent_when_target_proved: bool = True
    fallback_when_target_ruled_out: str = "strongest_set_alone"
    controller_reads_own_posts: bool = False
    controller_memory: str = "none"
    posts_all_or_nothing: bool = True
    paired_random_streams: bool = True
    seed: int = 2026

    def check(self) -> list[str]:
        """Reject settings the spec defines but this version does not implement; return warnings."""
        if self.arm not in ("silent", "truth", "false"):
            raise ValueError(f"unknown arm {self.arm!r}")
        if self.arm != "silent" and (self.qc is None or self.b is None):
            raise ValueError("a controlled arm needs qc and b")
        todo = {
            "board_read_weighting": (self.board_read_weighting, "uniform"),
            "read_posts_not_facts": (self.read_posts_not_facts, True),
            "support_rule": (self.support_rule, "in_context"),
            "supporting_fact_choice": (self.supporting_fact_choice, "uniform"),
            "controller_memory": (self.controller_memory, "none"),
            "controller_reads_own_posts": (self.controller_reads_own_posts, False),
            "posts_all_or_nothing": (self.posts_all_or_nothing, True),
            "paired_random_streams": (self.paired_random_streams, True),
            "fallback_when_target_ruled_out": (self.fallback_when_target_ruled_out, "strongest_set_alone"),
        }
        for name, (value, supported) in todo.items():
            if value != supported:
                raise NotImplementedError(f"{name}={value!r} is in the spec but not implemented yet")
        if self.agent_sampling_mode not in ("argmax", "probability_matching", "softmax"):
            raise ValueError(f"unknown agent_sampling_mode {self.agent_sampling_mode!r}")
        if self.agent_post_rule not in ("supporting", "uniform_active"):
            raise ValueError(f"unknown agent_post_rule {self.agent_post_rule!r}")
        if self.support_fallback not in ("read_alone", "none"):
            raise ValueError(f"unknown support_fallback {self.support_fallback!r}")
        if self.controller_gate not in ("votes", "facts", "always"):
            raise ValueError(f"unknown controller_gate {self.controller_gate!r}")
        warnings = []
        if self.controller_gate == "votes" and not self.is_board_cleared and self.arm != "silent":
            warnings.append("controller_gate='votes' with is_board_cleared=False: posts keep the vote "
                            "their author had on the day they posted, so the controller reads old "
                            "opinions mixed with new ones (spec §3).")
        return warnings

    def cell_id(self) -> str:
        if self.arm == "silent":
            return f"{self.setup}__silent__q{self.q}__rho{self.rho}"
        return f"{self.setup}__{self.arm}__q{self.q}__qc{self.qc}__b{self.b}__rho{self.rho}"


def _stream(seed: int, episode: int, name: str) -> random.Random:
    """One independent random stream per purpose (§8). It depends on the episode, not
    the arm or cell, so arms of the same episode start from the same draws."""
    h = hashlib.sha256(f"{seed}|{episode}|{name}".encode()).digest()
    return random.Random(int.from_bytes(h[:8], "big"))


def _vote(P, mode: str, beta: float, rng: random.Random) -> int:
    """§5."""
    if mode == "argmax":
        top = max(P)
        winners = [k for k in range(3) if P[k] >= top - EPS]
        return winners[0] if len(winners) == 1 else rng.choice(winners)
    weights = P if mode == "probability_matching" else [p ** beta for p in P]
    r = rng.random() * sum(weights)
    acc = 0.0
    for k in range(3):
        acc += weights[k]
        if r < acc:
            return k
    return max(k for k in range(3) if weights[k] > 0)


def _choose_post(world: World, active: int, vote: int, read_today: int, p: Params,
                 rng: random.Random) -> tuple[int, int]:
    """§6. Returns (fact index or NO_POST, post reason)."""
    if not active:
        return NO_POST, NO_MEMORY
    facts = list(bits(active))
    if p.agent_post_rule == "uniform_active":            # control: ignore the vote entirely
        return rng.choice(facts), UNIFORM
    pk = world.posterior(active)[vote]
    # step 1: support in context -- removing the fact lowers P(vote)
    support = [f for f in facts if pk > world.posterior(active & ~(1 << f))[vote] + EPS]
    reason = IN_CONTEXT
    # step 2: fallback for redundant facts -- the fact alone raises P(vote) above the prior
    if not support and p.support_fallback == "read_alone":
        prior = world.posterior(0)[vote]
        support = [f for f in facts if world.posterior(1 << f)[vote] > prior + EPS]
        reason = FALLBACK
    if support:
        # step 3: prefer facts not read today
        if p.prefer_facts_not_read_today:
            fresh = [f for f in support if not (read_today >> f) & 1]
            if fresh:
                support = fresh
        return rng.choice(support), reason              # step 4: uniform
    if p.agent_post_always:                             # step 5
        return rng.choice(facts), RANDOM
    return NO_POST, ABSTAINED


def _best_set(pool: tuple, b: int, score) -> int:
    """The b-fact subset of the pool with the highest score; ties go to the first set in
    alphabetical order of fact IDs (the pool is sorted by fact ID)."""
    best_set, best_val = 0, None
    for combo in itertools.combinations(pool, b):
        S = 0
        for f in combo:
            S |= 1 << f
        v = score(S)
        if best_val is None or v > best_val + EPS:
            best_set, best_val = S, v
    return best_set


def _controller_night(world: World, pool: tuple, target: int, read: list, p: Params):
    """§7. `read` is the list of agent posts the controller drew.
    Returns (decision, fact set to post, v_hat, P(target | R))."""
    R = 0
    for post in read:
        R |= 1 << post[1]
    v_hat = (sum(post[3] == target for post in read) / len(read)) if read else None
    p_R = world.posterior(R)[target]
    proved = p_R >= 1.0 - EPS
    if proved and p.silent_when_target_proved:               # 1: target already proved
        return "proved", 0, v_hat, p_R
    if p.controller_gate == "votes" and v_hat is not None and v_hat >= p.theta_vote - EPS:
        return "gate", 0, v_hat, p_R                         # 2: gate on votes
    if p.controller_gate == "facts" and p_R >= p.theta_facts - EPS:
        return "gate", 0, v_hat, p_R                         # 2: gate on facts
    if proved:
        # silent_when_target_proved is False and the board already proves the target: every
        # set scores 1, so rank sets by their strength on their own instead (spec §7, step 3b)
        return "posted_proved", _best_set(pool, p.b, lambda S: world.posterior(S)[target]), v_hat, p_R
    S = _best_set(pool, p.b, lambda S: world.posterior(R | S)[target])     # 3: best set
    if world.posterior(R | S)[target] > EPS:
        return "posted", S, v_hat, p_R
    # 4: what it read rules the target out: the set strongest on its own
    return "fallback", _best_set(pool, p.b, lambda S: world.posterior(S)[target]), v_hat, p_R


def run_episode(world: World, setup: Setup, p: Params, episode: int, record_posts_read: bool = True):
    """One episode (§2). Returns three lists of row tuples: agent-days, nights, days."""
    rng = {name: _stream(p.seed, episode, name)
           for name in ("order", "reads", "forget", "vote", "post", "controller_reads")}
    n = len(setup.agent_ids)
    own_evidence = 0                                  # the agents' original facts (proof measured two ways)
    for f in setup.agent_facts:
        own_evidence |= 1 << f
    known = [1 << f for f in setup.agent_facts]
    active = list(known)
    target = TARGETS.get(p.arm)
    pool = setup.pools[target] if target is not None else ()

    # a post is (author, fact, day, author's vote); the controller's author is -1, vote -1
    board: list[tuple[int, int, int, int]] = []
    agent_rows, night_rows, day_rows = [], [], []

    for day in range(p.M):
        # dawn: forgetting (§2), not before the first day
        if day > 0 and p.rho < 1.0:
            for a in range(n):
                kept = 0
                for f in bits(active[a]):
                    if rng["forget"].random() < p.rho:
                        kept |= 1 << f
                active[a] = kept
        morning_controller_posts = sum(1 for post in board if post[0] == CONTROLLER and post[2] == day - 1)

        order = list(range(n))
        rng["order"].shuffle(order)
        posteriors, votes, reasons = [None] * n, [None] * n, [None] * n
        n_agent_posts = 0
        for position, a in enumerate(order):
            after_dawn, known_before = active[a], known[a]
            # 1. read q posts, never its own (§2, §4)
            eligible = [post for post in board if post[0] != a]
            read = rng["reads"].sample(eligible, min(p.q, len(eligible)))
            # 2. absorb: read facts join known and active
            read_today = 0
            for post in read:
                read_today |= 1 << post[1]
            known[a] |= read_today
            active[a] |= read_today
            # 3. update, 4. vote
            P = world.posterior(active[a])
            v = _vote(P, p.agent_sampling_mode, p.beta, rng["vote"])
            # 5. post
            f, reason = _choose_post(world, active[a], v, read_today, p, rng["post"])
            if f != NO_POST:
                board.append((a, f, day, v))
                n_agent_posts += 1
            posteriors[a], votes[a], reasons[a] = P, v, reason
            agent_rows.append((
                episode, day, position, a,
                after_dawn, known_before, active[a], read_today, len(read),
                [encode_post(post[0], post[1]) for post in read] if record_posts_read else None,
                P[0], P[1], P[2], v, f, reason,
                P[0] == 1.0, world.posterior(active[a] & own_evidence)[0] == 1.0))

        # night (§2, §7); none after the final day unless asked
        posted = 0
        if target is not None and (day < p.M - 1 or p.controller_acts_on_final_night):
            agent_posts = [post for post in board if post[0] != CONTROLLER]
            read = rng["controller_reads"].sample(agent_posts, min(p.qc, len(agent_posts)))
            decision, posted, v_hat, p_R = _controller_night(world, pool, target, read, p)
            poster_votes = [post[3] for post in agent_posts]
            night_rows.append((
                episode, day, len(agent_posts), len(read),
                [encode_post(post[0], post[1]) for post in read] if record_posts_read else None,
                v_hat, votes.count(target) / n,
                (poster_votes.count(target) / len(poster_votes)) if poster_votes else None,
                p_R, decision, posted, posted.bit_count()))
        new_controller_posts = [(CONTROLLER, f, day, -1) for f in bits(posted)]
        board = new_controller_posts if p.is_board_cleared else board + new_controller_posts

        day_rows.append((episode, day,
                         sum(P[0] for P in posteriors) / n, sum(P[1] for P in posteriors) / n,
                         sum(P[2] for P in posteriors) / n,
                         votes.count(0) / n, votes.count(1) / n, votes.count(2) / n,
                         sum(P[0] == 1.0 for P in posteriors) / n,
                         sum(world.posterior(active[a] & own_evidence)[0] == 1.0 for a in range(n)) / n,
                         sum(s.bit_count() for s in active) / n,
                         n_agent_posts, sum(r == ABSTAINED for r in reasons) / n,
                         morning_controller_posts, posted.bit_count()))
    return agent_rows, night_rows, day_rows


AGENT_COLUMNS = ["episode", "day", "position", "agent",
                 "memory_after_dawn", "known_before_reading", "active_facts", "facts_read", "n_posts_read",
                 "posts_read", "p_A0", "p_A1", "p_A2", "vote", "posted_fact", "post_reason",
                 "proves_A0", "proves_A0_own_evidence"]
NIGHT_COLUMNS = ["episode", "day", "n_agent_posts_on_board", "n_posts_read", "posts_read", "v_hat",
                 "true_target_share", "target_share_among_posters", "p_target_given_read", "decision",
                 "posted_facts", "n_posted"]
DAY_COLUMNS = ["episode", "day", "mean_p_A0", "mean_p_A1", "mean_p_A2", "share_A0", "share_A1",
               "share_A2", "proof_rate_A0", "proof_rate_A0_own_evidence", "mean_active_facts",
               "agent_posts", "abstention_rate", "controller_posts_on_board_this_morning",
               "controller_posts_tonight"]


def params_dict(p: Params) -> dict:
    return asdict(p)
