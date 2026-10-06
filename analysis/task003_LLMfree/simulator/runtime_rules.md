# The real game's rules — the starting point for the LLM-free simulator

The LLM-free simulator replaces each language-model agent with an exact Bayesian
reasoner. Everything **else** should behave like the real game, so that what we
learn carries over. This file lists those rules, where each lives in the real
game's code, and the decisions already taken for the new simulator.

Written 6 October 2026 from the old simulator (`60_exact_simulation/exact_game.py`,
deleted; recover it with `git show ba16739:analysis/preanalysis_task003_new_setup_and_coarsegraining/60_exact_simulation/exact_game.py`).

The real game lives in `src/mas_cc/games/relational_reasoning/imitation_round_feedback/`.
Below, `runtime.py`, `game.py`, `state.py` and `controller.py` mean files in that folder.

---

## 1. Rules to copy from the real game

### Memory: `known` and `active`

Each agent has two sets of facts:

- **known**: every fact it has ever held. It only grows.
- **active**: the facts it currently remembers and reasons with. A subset of known.

At the start, both are the agent's one assigned fact.

### Forgetting (persistence ρ)

At the start of every round, each **active** fact survives independently with
probability ρ (rho). `known` is untouched.
Code: `apply_epistemic_persistence` in `runtime.py`.

Example: an agent with 4 active facts at ρ = 0.75 keeps on average 3 of them.

### Reading makes a fact active again (exposure → activation)

When an agent reads a message citing a fact:

- if the fact is new to it, the fact is added to **known** and **active**;
- if it knew the fact but had forgotten it, the fact becomes **active** again.

Code: the loop over `exposures` in `game.py` (around line 920).
This is the **whole propagation mechanism**. Without it beliefs would only decay.

### One agent at a time, within a round

Agents act **one after another** and each sees the posts made earlier in the
same round. Code: `for within_round_index in range(rules.n_agents)` in
`runtime.py`. This ordering matters: an earlier result was invalidated by it.
The old simulator kept a fixed order; shuffling the order each round is a
sensitivity check.

### Two board buffers

- The **controller** acts at the start of the round and sees **last round's**
  board (`live_messages(round_index - 1)`, `state.py`).
- The **agents** read **this round's** board as it grows
  (`live_messages(round_index)`).

With `message_lifetime_rounds = 1` (the default), last round's board is gone for
the agents by the time they act. So each round, an agent reads the controller's
fresh posts plus whatever earlier agents posted this round.

### What an agent reads

`board_sampling`: `full` (read every eligible message) or `uniform` (read a random
sample of `social_group_size` messages, called **q** below).
`exclude_self_authored`: an agent never reads its own posts.

### Posts carry facts

In the report-only setting every post cites one fact. The real game lets an
agent cite a fact it holds actively **or** one it observed
(`report_citation_scope: active_or_observed`, `game.py`). Since an observed fact
becomes active on reading, "cite from your active facts" covers both.

---

## 2. Decisions already taken for the new simulator (5 October 2026)

### Agents

- **Vote**: the allocation with the highest exact posterior given **all** the
  agent's active facts. Ties: a random choice among the tied allocations.
  (A softmax vote, a random draw favouring more probable allocations, may be
  kept for sensitivity checks only.)
- **Post**: one fact from its active facts that **supports its vote**, chosen at
  random among those supporting facts.
  - **Support is measured in context**: how much P(vote) drops if this fact is
    removed from the agent's active facts. Facts interact (the 6 decisive facts
    prove A0 only together), so "read alone" is not used.
  - **Prefer new facts**, where "new" means **not among the messages this agent
    read** this round, not "not anywhere on the board".
  - **Never "post the strongest fact"**: the old simulator tried it, every agent
    posted the same fact, and knowledge stopped spreading.

### Controller

- **Drop the `lru` rule.** It copied the real scripted controller, which only
  rotates through its pool and was measured to steer not at all (gap 0.000).
- **Keep `greedy_posterior` only as an oracle reference**: it reads every
  agent's private memory, which no real controller can.
- **The main controller** reads **qc** facts from last round's board and posts
  the **b** pool facts that maximise P(target | facts it read + facts it posts).
  It scores whole sets (12 pool facts, b = 3: 220 sets), not single facts.
  - **Fallback** when every set scores 0 (the board already proves the other
    allocation): rank by P(target | posted facts alone).
- **Pools**: task003-symmetric uses the A0 pool when targeting A0 and the A2
  pool when targeting A2; task003-nosolution uses its single pool for both.

### Open: a sensing-feedback rule

Proposed: post nothing when s = P(target | facts read) ≥ θ (for example 0.8).
To decide: hard threshold or soft (post with a probability that falls as s
rises); and how to report dose, since the truth and false arms will then post
different amounts (the archived study's 32% vs 54% firing gap explained most of
its apparent effect).

---

## 3. Known gaps of the old simulator, to address or accept

- **No social influence.** Real agents see peers' **votes** as well as facts;
  here only facts enter the belief. Adding a vote term adds a free parameter.
- **Never validated against the LLM runs** beyond the qualitative rise in mean
  belief.
- **`known` only grows**, as in the real game.
- **`uniform` board sampling was implemented but never tested.**

## 4. Checks the new simulator should have

- The mean belief in the truth **rises** in the silent arm at ρ = 1. The old
  simulator showed this, and so does the archived LLM data (0.43 → 0.63 over 15
  rounds).
- The same seed gives identical results.
- Each agent's posterior, recomputed independently from its recorded active
  facts with `task_and_facts/engine.py`, matches what the simulator used.
