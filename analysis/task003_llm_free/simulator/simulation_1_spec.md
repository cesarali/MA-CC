# Simulation 1 — days and nights, per-round controller budget

**Status: agreed 6 October 2026, implemented; extended 7 October 2026** (final night,
`agent_post_rule`, `silent_when_target_proved: false`, richer recording).

The rules copied from the real game, and where they live in its code, are in
[`runtime_rules.md`](runtime_rules.md). This file defines Simulation 1 exactly
enough to implement it.

---

## 1. What is simulated

24 agents and one controller play on one world, `task_003`, where `ALLOCATION_0`
(A0) is correct. There are no language models and no natural language: agents
and the controller exchange **fact IDs** such as `cf_x00_ge_x04`. Every agent
is an exact Bayesian reasoner: its belief is the exact posterior over A0, A1, A2
given the facts it currently remembers, computed by
[`../task_and_facts/engine.py`](../task_and_facts/engine.py).

The agents and the controller's pool or pools come from the frozen setups:

| setup | agents | controller pool |
|---|---|---|
| task003-symmetric | `../experimental_setup/task003_symmetric/agents.json` | `controller_pool_A0.json` when targeting A0, `controller_pool_A2.json` when targeting A2 |
| task003-nosolution | `../experimental_setup/task003_nosolution/agents.json` | `controller_pool.json` for both targets |

---

## 2. One episode, step by step

An episode is **M rounds**. Each round is a **day** followed by a **night**.

**Start of the episode.** Each agent's `known` and `active` facts are its one
assigned fact. The board is empty.

**Dawn** (every round except the first). Each agent keeps each of its active
facts independently with probability ρ; a forgotten fact stays in `known`.
An agent can forget its original fact and get it back by reading it again.
**(decided)**

**Day.** Agents act **one at a time, in a new random order each day**. Each agent:

1. **reads** q posts drawn at random, without replacement, from the board
   (§4), never its own posts. If fewer than q are available it reads them all.
2. **absorbs** the fact in each post it read: a new fact joins `known` and
   `active`; a forgotten one becomes `active` again. Read facts count for
   today's vote. **(decided)**
3. **updates** its posterior from all its `active` facts.
4. **votes** (§5).
5. **posts** at most one fact (§6). The post carries the agent's vote, which
   only the controller can see.

Later agents in the day read the posts of earlier ones.

**Night.** The controller reads qc posts, decides whether to post, and posts
either **all b facts or none** (§7). Then, if `is_board_cleared` is True, the
board is cleared and **only the controller's new posts remain** for the next
day. **(decided)**

There is no night before day 1, so on day 1 the board starts empty and the
first agent reads nothing. **There is no controller on the final night**
(`controller_acts_on_final_night: false`, decided 7 October): after day M no
agent can read its posts, so they could only add cost.

---

## 3. Settings

| setting | values in the study | default | meaning |
|---|---|---|---|
| `M` | 30 | 30 | rounds per episode |
| `controller_acts_on_final_night` | False | False | whether the controller posts after the last day |
| `q` | 3, 6, 12 | — | posts an agent reads per day |
| `qc` | 3, 6, 12, 24 | — | posts the controller reads per night |
| `b` | 1, 2, 3, 6, 9 | — | facts the controller posts when it acts |
| `rho` | 0.75, 1.0 | — | probability an active fact survives each dawn |
| `arm` | silent, truth (A0), false (A2) | — | silent = no controller |
| `episodes` | 1,000 per cell | 1,000 | |
| `is_board_cleared` | True, False | True | clear the board each night, keeping only the controller's new posts |
| `board_read_weighting` | uniform, recency | uniform | how posts are drawn from an uncleared board (§4) |
| `agent_sampling_mode` | argmax, probability_matching, softmax | argmax | how an agent turns its posterior into a vote (§5) |
| `beta` | — | 4 | sharpness for `softmax` only |
| `agent_post_rule` | supporting, uniform_active | supporting | `supporting`: post a fact supporting the vote (§6); `uniform_active`: post a uniformly random active fact, ignoring the vote (a control) |
| `agent_post_always` | True, False | False | `supporting` rule only: post a random active fact when no fact supports the vote (§6) |
| `controller_gate` | votes, facts, always | votes | what decides whether the controller acts (§7) |
| `theta_vote` | 0.75 (others later) | 0.75 | `votes` gate: stay silent at or above this share |
| `theta_facts` | — | 0.8 | `facts` gate: stay silent at or above this probability |
| `silent_when_target_proved` | True, False | True | stay silent when what it read proves the target (§7 step 1) |
| `controller_memory` | none, accumulate | none | what the controller remembers across nights (later experiments) |

**The studies** live in [`configs/`](configs/), one file per study. The
7 October grid, per voting/posting variant:

- silent: 2 setups × 3 q × 2 ρ = **12 cells** (qc and b mean nothing without a controller)
- controlled: 2 setups × 3 q × 4 qc × 5 b × 2 ρ × 2 targets = **480 cells**
- **492 cells × 1,000 episodes.**

| config | voting | posting |
|---|---|---|
| `sim1_base` | argmax | supporting, abstain when nothing supports the vote |
| `sim1_pm` | probability matching | supporting, abstain |
| `sim1_pm_post_always` | probability matching | supporting, random active fact when nothing supports the vote |
| `sim1_uniform_post` | argmax | uniform_active |

Each has a `_no_proof_stop` twin: task003-symmetric truth cells only (120),
with `silent_when_target_proved: false`.

**The effective vote gate depends on qc.** With `theta_vote` = 0.75 the
controller stays silent when at least ⌈0.75·qc⌉ of the posts it read vote for
its target: 3 of 3 (qc = 3), 5 of 6, 9 of 12, 18 of 24. With abstaining agents
it may read fewer than qc posts, which shifts this again; `n_posts_read` is
recorded each night. b = 12 is excluded: every pool has 12 facts, so the
controller would have no choice.

The uncleared board (`is_board_cleared = False`) is a later, separate study.

**Warning:** if `controller_gate = votes` and `is_board_cleared = False`, the
simulator prints a warning at start-up: posts keep the vote their author had on
the day they posted, so the controller reads old opinions mixed with new ones.
**(decided)**

---

## 4. The board

Each post records: author (an agent ID or `controller`), fact ID, day posted,
and, for agent posts, the author's vote that day.

- **Cleared board (default).** During a day the board holds last night's
  controller posts plus today's agent posts so far.
- **Uncleared board.** Nothing is ever removed. Posts are drawn by
  `board_read_weighting`:
  - `uniform` **(decided default)**: every post ever made is equally likely;
  - `recency`: newer posts more likely (weights to be defined if used).
- **Posts, not distinct facts, are drawn.** A fact posted three times is three
  times as likely to be read. Redundancy is part of the dynamics. **(decided)**

---

## 5. How an agent votes

From its posterior P = (P(A0), P(A1), P(A2)) over its active facts:

| `agent_sampling_mode` | rule | example, P = (0.6, 0.1, 0.3) |
|---|---|---|
| `argmax` **(default)** | the most probable allocation; ties broken at random | always A0 |
| `probability_matching` | draw the vote from P | A0 60%, A1 10%, A2 30% |
| `softmax` | draw from P^β, normalised (β = 1 is probability matching; large β approaches argmax) | β = 4: A0 94% |

An agent with no active facts has P = (⅓, ⅓, ⅓) and votes at random.

---

## 6. Which fact an agent posts

1. **Supporting facts.** A fact *f* supports the vote *k* if removing it lowers
   P(k): P(k | active) > P(k | active without *f*). This is "support in
   context": facts interact, so a fact's support is judged next to the others
   the agent holds.
2. **Fallback for redundant facts (decided).** If no fact passes step 1, use
   "read alone": *f* supports *k* if P(k | *f* only) > ⅓. See §10.
3. **Prefer facts not read today.** Among supporting facts, keep those the
   agent did not read today, if any. A fact absorbed on an earlier day counts
   as the agent's own, so it can be passed on.
4. **Pick one uniformly at random.** **(decided)**
5. **If nothing supports the vote:** post nothing, unless `agent_post_always`
   is True, in which case post a random active fact. An agent with no active
   facts never posts. **(decided)**

**The `uniform_active` control** (`agent_post_rule`, decided 7 October)
replaces steps 1–5: post one active fact chosen uniformly at random, whatever
the vote. It tests whether vote-conditioned posting drives the A2 drift
(§10).

Never "post the strongest fact": the old simulator showed every agent then
posts the same fact and knowledge stops spreading.

---

## 7. The controller

Silent arm: no controller. Otherwise the target is A0 (truth) or A2 (false).

**Reading.** At night the controller draws qc posts from the board, never its
own. If fewer are available it reads them all. Let *R* be the facts in the
posts it read and *v̂* the share of those posts whose author voted for the
target. **(decided: skip its own posts; no memory across nights by default)**

**Deciding, in this order:**

1. **Target already proved.** If P(target | R) = 1 and
   `silent_when_target_proved` is True: post nothing. Only the truth controller
   can reach this; nothing can prove A2. **(decided)**
2. **Gate** (`controller_gate`):
   - `votes` **(default)**: post nothing if *v̂* ≥ `theta_vote`. Votes measure
     the opinion the controller is trying to move.
   - `facts`: post nothing if P(target | R) ≥ `theta_facts`.
   - `always`: never stop here.
   If the controller read no agent posts, *v̂* is undefined and it acts.
3. **Choose what to post.** Among all sets S of b facts from its pool, post
   the one that maximises P(target | R + S). It checks every set (220 for
   b = 3 and a 12-fact pool). Ties go to the alphabetically first set.
   Facts may repeat on later nights, which refreshes fading memories. **(decided)**

   **3b. If P(target | R) = 1 already** (possible only with
   `silent_when_target_proved: false`), every set scores 1, and the
   alphabetical tie-break would post the same facts whatever they are. So in
   this case the controller posts the set that maximises P(target | S) on its
   own, ties alphabetical. Recorded as decision `posted_proved`. Added
   7 October; the default runs never reach it.
4. **Fallback when the target is ruled out.** If every S gives P(target | R + S)
   = 0, post the set S that maximises P(target | S) on its own. **(decided)**

**What *v̂* means.** It is estimated from the posts read, not from the whole
population. Agents who did not post that day are invisible to it. Example: with
qc = 12 and `theta_vote` = 0.75, the controller stays silent when at least 9 of
the 12 posts it read came from agents voting for its target.

---

## 8. Randomness and pairing

Every random choice has its own stream, seeded from (episode number, stream
name): day order, agents' reads, forgetting, votes (for the sampling modes),
posting, controller reads. **(decided)** The silent, truth and false arms of
the same episode number therefore share their starting point and, as long as
the controller has not changed anything, the same draws, so differences
between arms come from the controller and not from chance. Once the controller
changes what is on the board, the runs diverge naturally.

---

## 9. What is recorded

Each run goes to a new dated folder `results/<date>_<study>/` (not in git);
the column-by-column list is in [`README.md`](README.md).

- **per agent per day:** position in the order; memory after the dawn
  forgetting and `known` before reading (to separate new facts, reactivations
  and losses); every post read (author and fact, repeats kept); memory after
  reading; posterior; vote; fact posted and **why** (in context, fallback,
  random, uniform, abstained, empty memory); whether its memory proves A0,
  **measured two ways**: from all its facts, and from its facts that belong to
  the agents' original evidence only. The difference shows whether a proof came
  from the controller's facts.
- **per night:** posts on the board and posts read (author and fact), *v̂*, the
  **true** share of all agents voting for the target that day, the share among
  agents who posted, P(target | R), the decision (proved / gate / posted /
  posted_proved / fallback), facts posted.
- **per day:** population averages, including the abstention rate.

Enough to rebuild the coarse-grained state (mean P(A0), and mean P(A2)) for
`../coarse_graining/`.

---

## 10. Open points

- **Why the redundancy fallback in §6 step 2 exists (decided).** Without it, some of the
  most certain agents fall silent. The test removes **one fact at a time**. If
  every fact the agent holds is left out of at least one complete proof it
  also holds, then P(A0) stays 1 whichever single fact is removed, and no fact
  passes. Real example in task003-symmetric: 8 agent facts plus 2 A0-pool facts
  (`cf_x00_eq_3`, `cf_x04_eq_2`) form a 10-fact memory where removing any one
  fact leaves P(A0) = 1. It can only arise once the truth controller has posted;
  with the agents' facts alone it cannot, because their 12 proofs all overlap.
- **`recency` weights** for an uncleared board: to define only if used.
- **Speed:** a 156-cell grid took 100 s on 12 cores (6 October); b = 6 is the
  slowest case measured, about 30 ms per episode.
- **The A2 drift (7 October).** In the silent no-solution arm the swarm drifts
  to A2. Starting votes are not balanced: expected 7 / 6 / 11 for A0 / A1 / A2,
  because six agents hold facts tied between A1 and A2. A diagnostic by a
  reviewing agent points to argmax voting plus vote-conditioned posting as the
  central mechanism. The four 7 October variants test this.
- **Known limitation.** The controller maximises P(target | sampled board facts
  + its posts), a proxy: that union is nobody's actual memory. The
  `greedy_posterior` oracle of the old simulator is the alternative.
