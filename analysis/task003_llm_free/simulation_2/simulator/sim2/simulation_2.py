"""Simulation 2: asynchronous activation, with Simulation 1's rules available as options.

Implements ../simulation_2_spec.md (simulation_2/simulator/). Section numbers (§) below refer to it.

One episode is a list of timed events processed in order (§4). Every setting that differs
between Simulation 1 and the source document is an option (§2), so the bridge steps and
the rate scan run through this one code path.
"""
from __future__ import annotations
import hashlib, heapq, math, random
from dataclasses import dataclass, asdict

from llmfree_core.setups import TARGETS, Setup
from llmfree_core.world import World, bits

EPS = 1e-12
CONTROLLER = -1                      # author of controller posts
NO_POST = -1
NO_VOTE = -1

# why an agent posted what it posted (actions table, column post_reason); same codes as Simulation 1
POST_REASONS = {0: "in_context", 1: "fallback", 4: "abstained", 5: "no_memory"}
IN_CONTEXT, FALLBACK, ABSTAINED, NO_MEMORY = 0, 1, 4, 5

# event ranks: at equal times, lower rank first (§4)
MEASURE, CONTROL, CLEAR, DAWN, FORGET, ACT = range(6)

AGENT_SETTINGS = ("board", "forgetting", "agent_schedule")
CONTROLLER_SETTINGS = ("controller_schedule", "lambda_c", "controller_gate", "theta_vote",
                       "silent_when_target_proved", "qc", "budget")


@dataclass(frozen=True)
class Params:
    setup: str
    arm: str                                  # silent | truth | false
    step: str                                 # label from the config, e.g. S3 or rate_2
    # the settings that the bridge switches one at a time (§2)
    board: str = "persistent"                 # cleared_nightly | persistent
    forgetting: str = "continuous"            # dawn | continuous
    agent_schedule: str = "poisson"           # days | poisson
    controller_schedule: str = "poisson"      # nights | poisson
    controller_gate: str = "always"           # votes | always
    silent_when_target_proved: bool = False
    # fixed in both studies (§2)
    lambda_a: float = 1.0
    lambda_c: float | None = 1.0              # poisson controller only
    theta_vote: float = 0.75
    rho: float = 0.75
    q: int = 6
    qc: int = 12
    W: int = 48
    budget: int = 30
    t_control: float = 30.0                   # the controller may act for t < t_control
    t_end: float = 40.0
    grid_step: float = 0.25
    agent_sampling_mode: str = "probability_matching"
    prefer_facts_not_read: bool = True
    seed: int = 20261007

    def check(self) -> list[str]:
        """Reject impossible or unimplemented settings; return warnings."""
        allowed = {"arm": ("silent", "truth", "false"), "board": ("cleared_nightly", "persistent"),
                   "forgetting": ("dawn", "continuous"), "agent_schedule": ("days", "poisson"),
                   "controller_schedule": ("nights", "poisson"), "controller_gate": ("votes", "always"),
                   "agent_sampling_mode": ("probability_matching",)}
        for name, values in allowed.items():
            if getattr(self, name) not in values:
                raise ValueError(f"{name}={getattr(self, name)!r}; allowed: {values}")
        if not 0 < self.rho <= 1:
            raise ValueError("rho must be in (0, 1]")
        if self.board == "cleared_nightly" and self.controller_schedule != "nights":
            raise ValueError("board=cleared_nightly needs controller_schedule=nights (the board is cleared at night)")
        if self.agent_schedule == "days" and self.t_end != int(self.t_end):
            raise ValueError("agent_schedule=days needs a whole number of days")
        if self.controller_schedule == "poisson" and self.arm != "silent" and not (self.lambda_c or 0) > 0:
            raise ValueError("controller_schedule=poisson needs lambda_c > 0")
        if abs(self.t_end / self.grid_step - round(self.t_end / self.grid_step)) > 1e-9:
            raise ValueError("t_end must be a multiple of grid_step")
        warnings = []
        if self.controller_gate == "votes" and self.board == "persistent" and self.arm != "silent":
            warnings.append("controller_gate='votes' with a persistent board: posts keep the vote their "
                            "author had when posting, so the controller reads old opinions (spec §3).")
        return warnings

    def cell_id(self) -> str:
        return f"{self.step}__{self.setup}__{self.arm}"

    def silent_key(self) -> tuple:
        """Settings a silent cell depends on. Two silent cells with the same key are the same
        simulation, so the runner runs only the first (spec §3)."""
        return (self.setup, *(getattr(self, k) for k in AGENT_SETTINGS), self.lambda_a, self.rho, self.q,
                self.W, self.t_end, self.grid_step, self.agent_sampling_mode, self.prefer_facts_not_read,
                self.seed)


def params_dict(p: Params) -> dict:
    return asdict(p)


# ---------------------------------------------------------------- randomness (§11)

def keyed_rng(seed: int, episode: int, *key) -> random.Random:
    """A random stream determined only by (seed, episode, key). The key never contains the arm
    or the step, so arms and steps share draws wherever their rules use the same key."""
    h = hashlib.blake2b(repr((seed, episode) + key).encode(), digest_size=8).digest()
    return random.Random(int.from_bytes(h, "big"))


# ---------------------------------------------------------------- rules shared with Simulation 1

def vote_probability_matching(P, rng: random.Random) -> int:
    """§7 step 3: draw the vote from the posterior."""
    r = rng.random() * sum(P)
    acc = 0.0
    for k in range(3):
        acc += P[k]
        if r < acc:
            return k
    return max(k for k in range(3) if P[k] > 0)


def choose_post(world: World, active: int, vote: int, read_now: int, prefer_not_read: bool,
                rng: random.Random) -> tuple[int, int, int, int, int]:
    """§7 step 4 (Simulation 1 spec §6, `supporting`, agent_post_always false).
    Returns (fact or NO_POST, reason, in-context candidates, fallback candidates, final candidates);
    candidates are fact sets."""
    if not active:
        return NO_POST, NO_MEMORY, 0, 0, 0
    facts = list(bits(active))
    pk = world.posterior(active)[vote]
    support = [f for f in facts if pk > world.posterior(active & ~(1 << f))[vote] + EPS]
    in_context = _mask(support)
    fallback = 0
    reason = IN_CONTEXT
    if not support:
        prior = world.posterior(0)[vote]
        support = [f for f in facts if world.posterior(1 << f)[vote] > prior + EPS]
        fallback = _mask(support)
        reason = FALLBACK
    if not support:
        return NO_POST, ABSTAINED, in_context, fallback, 0
    if prefer_not_read:
        fresh = [f for f in support if not (read_now >> f) & 1]
        if fresh:
            support = fresh
    return rng.choice(support), reason, in_context, fallback, _mask(support)


def _mask(facts) -> int:
    s = 0
    for f in facts:
        s |= 1 << f
    return s


def _best_fact(pool: tuple, score) -> int:
    """The pool fact with the highest score; ties within EPS go to the alphabetically first
    fact ID (the pool is sorted by fact ID)."""
    best, best_val = pool[0], None
    for f in pool:
        v = score(f)
        if best_val is None or v > best_val + EPS:
            best, best_val = f, v
    return best


def controller_decision(world: World, pool: tuple, target: int, R: int, v_hat, p: Params):
    """§8 steps 3-5. Returns (decision, fact or NO_POST)."""
    p_R = world.posterior(R)[target]
    proved = p_R >= 1.0 - EPS
    if proved and p.silent_when_target_proved:
        return "proved", NO_POST
    if p.controller_gate == "votes" and v_hat is not None and v_hat >= p.theta_vote - EPS:
        return "gate", NO_POST
    if proved:      # every fact scores 1: rank by strength alone (Simulation 1 spec §7 step 3b)
        return "posted_proved", _best_fact(pool, lambda f: world.posterior(1 << f)[target])
    f = _best_fact(pool, lambda f: world.posterior(R | (1 << f))[target])
    if world.posterior(R | (1 << f))[target] > EPS:
        return "posted", f
    return "fallback", _best_fact(pool, lambda f: world.posterior(1 << f)[target])


# ---------------------------------------------------------------- reading (§6)

def read_window(board: list, exclude_author: int | None, W: int, agents_only: bool) -> list:
    """The W most recent eligible posts. Own posts (and, for the controller, its posts) are
    removed BEFORE the window is taken. The board is in posting order."""
    out = []
    for post in reversed(board):
        if post[2] == exclude_author or (agents_only and post[2] == CONTROLLER):
            continue
        out.append(post)
        if len(out) == W:
            break
    return out


# ---------------------------------------------------------------- one episode

def run_episode(world: World, setup: Setup, p: Params, episode: int, record_agent_snapshots: bool = True):
    """One episode (§4-§9). Returns a dict of row lists, one per table (see COLUMNS)."""
    n = len(setup.agent_ids)
    seed = p.seed
    target = TARGETS.get(p.arm)
    controlled = target is not None
    pool = setup.pools[target] if controlled else ()
    own_evidence = _mask(setup.agent_facts)
    gamma = -math.log(p.rho) if p.rho < 1 else 0.0
    continuous = p.forgetting == "continuous" and p.rho < 1

    known = [1 << f for f in setup.agent_facts]
    active = list(known)
    last_vote = [NO_VOTE] * n
    last_vote_time = [math.nan] * n
    n_actions = [0] * n
    # continuous forgetting: one lifetime stream per (agent, fact), one draw per spell (§5, §11)
    spell_version: dict[tuple[int, int], int] = {}
    spell_start: dict[tuple[int, int], float] = {}
    life_rng: dict[tuple[int, int], random.Random] = {}

    # a post is (post id, time, author, fact, author's vote); the controller's vote is -1
    board: list[tuple[int, float, int, int, int]] = []
    next_post_id = 0
    messages = controller_reads = 0
    budget_left = p.budget
    exhausted_at = math.nan
    rows = {k: [] for k in COLUMNS}

    heap: list = []
    seq = 0

    def push(t, rank, kind, data=None):
        nonlocal seq
        heapq.heappush(heap, (t, rank, seq, kind, data))
        seq += 1

    def start_spell(a, f, t):
        key = (a, f)
        rng = life_rng.get(key)
        if rng is None:
            rng = life_rng[key] = keyed_rng(seed, episode, "life", a, f)
        life = rng.expovariate(gamma)
        spell_version[key] = spell_version.get(key, 0) + 1
        spell_start[key] = t
        if t + life <= p.t_end:
            push(t + life, FORGET, "forget", (a, f, spell_version[key]))

    if continuous:
        for a in range(n):
            start_spell(a, setup.agent_facts[a], 0.0)

    # --- schedules (§4)
    n_grid = int(round(p.t_end / p.grid_step))
    for k in range(n_grid + 1):
        push(k * p.grid_step, MEASURE, "measure", k)
    n_days = int(p.t_end) if p.agent_schedule == "days" else 0
    if p.agent_schedule == "days":
        for d in range(1, n_days + 1):
            order = list(range(n))
            keyed_rng(seed, episode, "order", d).shuffle(order)
            for j, a in enumerate(order):
                push((d - 1) + (j + 1) / (n + 1), ACT, "act", a)
    else:
        clocks = [keyed_rng(seed, episode, "agent_clock", a) for a in range(n)]
        for a in range(n):
            t = clocks[a].expovariate(p.lambda_a)
            while t < p.t_end:
                push(t, ACT, "act", a)
                t += clocks[a].expovariate(p.lambda_a)
    if controlled:
        if p.controller_schedule == "nights":
            for d in range(1, int(math.ceil(p.t_control))):
                push(float(d), CONTROL, "control")
        else:
            clock = keyed_rng(seed, episode, "controller_clock")
            t = clock.expovariate(1.0) / p.lambda_c        # shared unit draws, scaled by the rate (§11)
            while t < p.t_control:
                push(t, CONTROL, "control")
                t += clock.expovariate(1.0) / p.lambda_c
    for d in range(1, int(math.ceil(p.t_end))):
        if p.board == "cleared_nightly":
            push(float(d), CLEAR, "clear")
        if p.forgetting == "dawn" and p.rho < 1:
            push(float(d), DAWN, "dawn", d)

    agent_posts_so_far = 0
    while heap:
        t, _, _, kind, data = heapq.heappop(heap)

        if kind == "act":
            a = data
            idx = n_actions[a]
            n_actions[a] += 1
            active_before, known_before = active[a], known[a]
            window = read_window(board, a, p.W, agents_only=False)
            read = keyed_rng(seed, episode, "reads", a, idx).sample(window, min(p.q, len(window)))
            read_now = _mask(post[3] for post in read)
            known[a] |= read_now
            active[a] |= read_now
            if continuous:
                for f in bits(read_now):          # newly active or renewed: a fresh lifetime
                    start_spell(a, f, t)
            P = world.posterior(active[a])
            v = vote_probability_matching(P, keyed_rng(seed, episode, "vote", a, idx))
            f, reason, cand_ctx, cand_fb, cand_final = choose_post(
                world, active[a], v, read_now, p.prefer_facts_not_read, keyed_rng(seed, episode, "post", a, idx))
            post_id = NO_POST
            if f != NO_POST:
                post_id = next_post_id
                board.append((post_id, t, a, f, v))
                rows["posts"].append((episode, post_id, t, a, f, v))
                next_post_id += 1
                agent_posts_so_far += 1
            last_vote[a], last_vote_time[a] = v, t
            rows["actions"].append((
                episode, t, a, idx, active_before, known_before, read_now,
                [post[0] for post in read], len(window), active[a], P[0], P[1], P[2], v,
                f, post_id, reason, cand_ctx, cand_fb, cand_final,
                P[0] == 1.0, world.posterior(active[a] & own_evidence)[0] == 1.0))

        elif kind == "forget":
            a, f, version = data
            if spell_version.get((a, f)) == version and (active[a] >> f) & 1:
                active[a] &= ~(1 << f)
                rows["forgetting"].append((episode, t, a, f, spell_start[(a, f)]))

        elif kind == "dawn":
            for a in range(n):
                rng = keyed_rng(seed, episode, "dawn", data, a)
                kept = 0
                for f in bits(active[a]):
                    if rng.random() < p.rho:
                        kept |= 1 << f
                    else:
                        rows["forgetting"].append((episode, t, a, f, math.nan))
                active[a] = kept

        elif kind == "clear":
            board = [post for post in board if post[2] == CONTROLLER and post[1] == t]

        elif kind == "control":
            if t >= p.t_control or budget_left <= 0:
                continue
            idx = len(rows["controller"])          # rows are per episode
            window = read_window(board, CONTROLLER, p.W, agents_only=True)
            read = keyed_rng(seed, episode, "controller_reads", idx).sample(window, min(p.qc, len(window)))
            controller_reads += len(read)
            R = _mask(post[3] for post in read)
            v_hat = (sum(post[4] == target for post in read) / len(read)) if read else None
            p_R = world.posterior(R)[target]
            decision, f = controller_decision(world, pool, target, R, v_hat, p)
            budget_before = budget_left
            post_id = NO_POST
            if f != NO_POST:
                post_id = next_post_id
                board.append((post_id, t, CONTROLLER, f, NO_VOTE))
                rows["posts"].append((episode, post_id, t, CONTROLLER, f, NO_VOTE))
                next_post_id += 1
                messages += 1
                budget_left -= 1
                if budget_left == 0:
                    exhausted_at = t
            rows["controller"].append((
                episode, t, idx, len(window), [post[0] for post in read], R, v_hat,
                sum(v == target for v in last_vote) / n, sum(v != NO_VOTE for v in last_vote) / n,
                p_R, decision, f, post_id, budget_before, budget_left, controller_reads))

        elif kind == "measure":
            Ps = [world.posterior(s) for s in active]
            voted = [v for v in last_vote if v != NO_VOTE]
            rows["snapshots"].append((
                episode, data, t,
                sum(P[0] for P in Ps) / n, sum(P[1] for P in Ps) / n, sum(P[2] for P in Ps) / n,
                last_vote.count(0) / n, last_vote.count(1) / n, last_vote.count(2) / n, len(voted) / n,
                sum(P[0] == 1.0 for P in Ps) / n,
                sum(world.posterior(s & own_evidence)[0] == 1.0 for s in active) / n,
                sum(s.bit_count() for s in active) / n, len(board), agent_posts_so_far, messages,
                controller_reads))
            if record_agent_snapshots:
                for a in range(n):
                    rows["agent_snapshots"].append((episode, data, a, active[a], last_vote[a],
                                                    t - last_vote_time[a]))

    rows["episodes"].append((episode, messages, controller_reads, exhausted_at, sum(n_actions),
                             agent_posts_so_far, len(rows["forgetting"])))
    return rows


COLUMNS = {
    # one row per post, agents and controller
    "posts": ["episode", "post_id", "t", "author", "fact", "author_vote"],
    # one row per agent action (§7)
    "actions": ["episode", "t", "agent", "action_index", "active_before", "known_before", "facts_read",
                "posts_read", "window_size", "active_after", "p_A0", "p_A1", "p_A2", "vote",
                "posted_fact", "post_id", "post_reason", "candidates_in_context", "candidates_fallback",
                "candidates_final", "proves_A0", "proves_A0_own_evidence"],
    # one row per fact lost; spell_start is NaN for dawn forgetting
    "forgetting": ["episode", "t", "agent", "fact", "spell_start"],
    # one row per controller action that could act (t < t_control, budget left)
    "controller": ["episode", "t", "action_index", "window_size", "posts_read", "facts_read", "v_hat",
                   "true_target_share", "share_voted", "p_target_given_read", "decision", "posted_fact",
                   "post_id", "budget_before", "budget_after", "cumulative_reads"],
    # one row per measurement time (§9)
    "snapshots": ["episode", "grid_index", "t", "mean_p_A0", "mean_p_A1", "mean_p_A2", "share_A0", "share_A1",
                  "share_A2", "share_voted", "proof_rate_A0", "proof_rate_A0_own_evidence",
                  "mean_active_facts", "board_size", "agent_posts", "controller_messages", "controller_reads"],
    "agent_snapshots": ["episode", "grid_index", "agent", "active", "last_vote", "vote_age"],
    "episodes": ["episode", "controller_messages", "controller_reads", "budget_exhausted_at",
                 "agent_actions", "agent_posts", "facts_forgotten"],
}
