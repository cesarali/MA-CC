# Simulation 2 — asynchronous activation

**Status: agreed and implemented 9 October 2026.** Code, configs and tests: [`README.md`](README.md).

Source: `agents_control/task003_llm_free/asynchronous_activation/minimal_experimental_setup.md`
(read-only, called "the source document" below). This spec follows it, with the
changes the researcher decided on 8–9 October, listed in §12. Simulation 1's
rules are in [`../../simulation_1/simulator/simulation_1_spec.md`](../../simulation_1/simulator/simulation_1_spec.md)
("the Simulation 1 spec").

> **Update, 9 October 2026 (decided by the researcher): the main study is the grid of §14.**
> The step-by-step ladder below (§3) was run once (`2026-10-09_sim2_bridge`) and is kept as
> history and as the S0 check. Its intermediate steps mix old and new features in ways that
> describe no real system, so they are not analysed. Read §14 first.

Simulation 2 was first planned as two studies, run in this order:

1. **The bridge** (§3): take Simulation 1 to the asynchronous setting **one
   change at a time**, so that each change in the results can be traced to one
   change in the rules.
2. **The rate scan** (§9): the source document's study. Only the controller's
   rate varies.

Both use **one simulator with options** (§2). The bridge's last step is the
rate scan's cell at λc = 1, so the two studies share one code path.

---

## 1. What is simulated

The same world, agents and controller as Simulation 1: `task_003`, 24
exact-Bayesian agents, one controller posting true facts from its pool to a
shared board. Agents and the controller exchange fact IDs, not language.

**Both setups**, loaded from `../../experimental_setup/` with `llmfree_core.setups`:

| setup | agents | controller pool | checked at start-up with the engine |
|---|---|---|---|
| task003-symmetric | `task003_symmetric/agents.json` | `controller_pool_A0.json` (target A0), `controller_pool_A2.json` (target A2) | 24 agents, 18 distinct facts, pooled agent facts prove A0; 12 facts per pool |
| task003-nosolution | `task003_nosolution/agents.json` | `controller_pool.json` (both targets) | 24 agents, 15 distinct facts, pooled posterior (0.5, 0, 0.5), 12-fact pool, no proof of any allocation from all agent + pool facts |

If a check fails, the runner stops and reports it. It never rebuilds a setup.
Input files are hashed into each run's manifest.

**Arms:** silent (no controller), truth (target A0), false (target A2).

**Time.** Time t is continuous. One time unit is one expected action per agent,
the same amount of agent activity as one Simulation 1 day. An episode runs from
t = 0 to t = 40:

- **0 ≤ t < 30: control period.** The controller may act.
- **30 ≤ t ≤ 40: follow-up.** No controller. Agents keep acting and forgetting.
  This measures how long the controller's effect lasts.

---

## 2. The settings (one simulator, options)

Each setting has a Simulation-1-like value and a source-document value. Each
bridge step (§3) switches one of them.

| setting | Simulation 1 value | source-document value | meaning |
|---|---|---|---|
| `board` | `cleared_nightly` | `persistent` | cleared each night (only the controller's new posts survive), or never cleared |
| `forgetting` | `dawn` | `continuous` | each active fact survives each dawn with probability ρ, or each fact has its own random lifetime (§5) |
| `agent_schedule` | `days` | `poisson` | every agent acts once per day in a new random order, or each agent acts at random times (§4) |
| `controller_schedule` | `nights` | `poisson` | the controller acts at t = 1, 2, …, 29, or at random times with rate λc (§4) |
| `controller_gate` | `votes` | `always` | stay silent when v̂ ≥ θ_vote = 0.75 (Simulation 1 spec §7), or never stop here |
| `silent_when_target_proved` | true | false | stay silent when the facts read prove the target |

Fixed in every step and both studies:

| setting | value | note |
|---|---|---|
| ρ | 0.75 | survival of a fact over one time unit |
| q | 6 | posts an agent reads per action |
| qc | 12 | agent posts the controller reads per action |
| W | 48 | readers sample from the 48 most recent eligible posts (§6) |
| facts per controller action | 1 | Simulation 1's b = 1 |
| budget B | 30 | controller messages per episode. Never binds with `nights` (at most 29 nights) |
| voting | probability matching | the vote is drawn from the agent's posterior |
| agent posting | Simulation 1 spec §6, `supporting`, `agent_post_always: false` | identical to the source document's rule; see §7 |
| controller memory | none | nothing remembered between actions |
| episodes | 1,000 per cell | after a timing run (§10) |

---

## 3. The bridge study

Each step keeps everything from the step before and changes one setting:

| step | change | what it isolates |
|---|---|---|
| **S0** | none: Simulation 1's rules in the new code | checks the new code against Simulation 1 (§9) |
| **S1** | `board: persistent` | evidence that stays on the board |
| **S2** | `forgetting: continuous` | when facts are forgotten, at the same average rate |
| **S3** | `agent_schedule: poisson` | losing the "everyone acts once a day" rhythm |
| **S4** | `controller_schedule: poisson`, λc = 1 | irregular controller timing |
| **S5** | `controller_gate: always` | the controller no longer stops when enough votes agree |
| **S6** | `silent_when_target_proved: false` | the controller no longer stops when its target is proved |

S6 is the rate scan's cell at λc = 1.

**Cells.** Each step has 2 setups × 3 arms. The silent arm has no controller,
so it is the same in S3–S6. It is run once, in S3, and reused for S4–S6.
Total: 4 × 2 silent + 7 × 2 × 2 controlled = **36 cells**.

**Two known couplings** (they cannot be separated by a change of order):

- **S1 also changes what the vote gate sees.** On a persistent board, posts
  keep their author's vote from when they were posted. The controller then
  reads old opinions mixed with new ones (Simulation 1 spec §3, warning). Both
  v̂ and the true current vote share are recorded at every controller action,
  so the size of this sensing error is measured in every step.
- **The proof stop matters only for task003-symmetric, truth arm.** Only there
  can the controller's reading prove its target. In every other cell S6
  repeats S5 except for its random draws.

---

## 4. Schedules: when agents and the controller act

**`agent_schedule: days`.** Day d (d = 1 … 40) is the interval (d − 1, d).
Each day the 24 agents act once each, in a new random order. The agent in
position j (j = 0 … 23) acts at t = (d − 1) + (j + 1)/25. So nobody acts at
an integer time. The exact times matter only for continuous forgetting (S2).

**`agent_schedule: poisson`.** Each agent has its own random "alarm clock":
waiting times between its actions are independent Exponential(λa = 1) draws.
The first action is one waiting time after t = 0. Nobody is forced to act at
t = 0. On average each agent acts once per time unit, as in `days`, but some
act twice in a unit and some not at all.

**`controller_schedule: nights`.** The controller acts at t = 1, 2, …, 29:
after days 1–29. There is no night 30, which is the cutoff. This is Simulation 1's
"no controller on the final night".

**`controller_schedule: poisson`.** Waiting times between controller actions are
Exponential(λc). Actions at t ≥ 30, or after the budget is spent, do nothing.

**Ties at the same time.** Only integer times can have ties: nights, dawns,
measurements and the cutoff. Agent actions and continuous forgetting fall on an
exact time with probability zero. At an integer time t, events happen in this
order:

1. **measurement** (§9). It sees the state at the end of the day, like
   Simulation 1's `days.parquet`;
2. **controller**: reads, decides, posts. Not at t ≥ 30;
3. **board clearing** (`cleared_nightly` only);
4. **dawn forgetting** (`dawn` only; not at t = 0 or t = 40).

A measurement never changes the state and uses no random numbers.

---

## 5. Forgetting

Each agent has `known` facts (ever held) and `active` facts (remembered now).
Only active facts enter its posterior. A forgotten fact stays in `known`.
Reading it again makes it active again.

**`dawn`.** As in Simulation 1: at each dawn (t = 1 … 39), every active fact
of every agent survives independently with probability ρ.

**`continuous`.** Each active fact has its own lifetime, drawn from
Exponential(γ) with γ = −ln ρ ≈ 0.288. So the chance of surviving one time
unit is exactly ρ, the same as one dawn on average.

- A lifetime starts when the fact becomes active: at t = 0 for the agent's
  own fact, and when it is read while inactive.
- **Rereading an active fact renews its lifetime.** The old expiry is cancelled
  by a version number, so a stale expiry can never remove a refreshed fact.
- At expiry the fact leaves `active` and the agent's posterior is recomputed,
  so measurements see it. The agent's **last vote does not change**: votes
  change only when the agent acts.

**Note on renewal.** Under `dawn`, rereading an active fact changes nothing,
because survival is decided at each dawn independently. Under `continuous`,
renewal is the natural equivalent, and the exponential lifetime has no memory,
so renewing does not change the odds. This rule is taken from the source
document, §5.

---

## 6. The board and reading

Every post records: post ID (increasing), time, author (agent ID or
`controller`), fact ID, and the author's vote at posting time (none for the
controller). Posts are never edited.

**Reading, for an agent:**

1. **eligible posts**: every post on the board except the reader's own;
2. **window**: the W = 48 most recent eligible posts (by time, then post ID).
   This is "limited attention": older posts stay on the board but cannot be
   read;
3. **sample**: draw up to q = 6 posts uniformly from the window, without
   replacement. If fewer are available, read them all.

Posts, not distinct facts, are drawn: a fact posted three times is three times
as likely to be read.

**For the controller**, the same, with "eligible" meaning **agent posts only**
(never its own) and qc = 12.

**With `cleared_nightly`** the board never holds more than 24 agent posts plus
1 controller post, so the window of 48 never removes anything. S0 therefore
reads exactly as Simulation 1 does.

---

## 7. One agent action

The same as Simulation 1 spec §2 days, steps 1–5:

1. read (§6) and absorb the facts read. New facts join `known` and `active`;
   forgotten ones become active again; active ones are renewed (`continuous`);
2. compute the exact posterior from the active facts;
3. vote by probability matching. The vote becomes the agent's **last
   expressed vote**;
4. post at most one fact by Simulation 1 spec §6 (`supporting`):
   - "support in context": facts whose removal lowers P(vote);
   - if none, the fallback "read alone": facts with P(vote | that fact alone) > 1/3;
   - among these, prefer facts not read in this action;
   - pick one uniformly at random;
   - if there are none, post nothing;
5. the post is on the board at once, so the next reader can see it.

This is the source document's rule too. Its fallback threshold, P(vote | no
facts), equals 1/3 because the world's prior is exactly (1/3, 1/3, 1/3)
(checked 9 October).

Before an agent's first action its last vote is "none". Measurements report the
fraction of agents that have voted.

---

## 8. One controller action

The same as Simulation 1 spec §7 with b = 1, plus the budget:

1. if t ≥ 30 or the budget is spent: do nothing. Nothing is read or recorded
   as a cost;
2. read up to qc agent posts (§6). R = the facts read; v̂ = the share of the
   posts read whose author voted for the target **when posting**;
3. if `silent_when_target_proved` and P(target | R) = 1: post nothing (`proved`);
4. if `controller_gate: votes` and v̂ ≥ 0.75: post nothing (`gate`). If
   nothing was read, v̂ is undefined and the controller acts;
5. otherwise post the pool fact f that maximises P(target | R + f). Ties
   within 1e-12 go to the alphabetically first fact ID. Two exceptions:
   - if P(target | R) = 1 already (only when step 3 is off), use
     P(target | f) alone (Simulation 1 spec §7, step 3b);
   - if every f gives 0, also use P(target | f) alone (`fallback`);
6. one fact posted = one message; the budget drops by 1.

Facts may repeat across actions and may already be in R. A repeat refreshes
fading memories, and it costs a message.

**Why step 5's first exception is needed here.** The source document covers
only task003-nosolution, where nothing can be proved, so it never meets this
case. With task003-symmetric and the proof stop off (S6), a proved R would
make every fact score 1. The alphabetical tie-break would then post the same
fact whatever it is.

---

## 9. What is recorded and measured

**Measurement grid:** t = 0, 0.25, 0.5, …, 40 (161 times), in every step.
At each grid time, from the current state:

- m_k(t): mean posterior of allocation k over all 24 agents, for k = A0, A1, A2;
- vote shares among last expressed votes, and the fraction of agents that have voted;
- proof rates: A0 proved by an agent's active facts, from all its facts and
  from the agents' original evidence only (as in Simulation 1);
- mean active-memory size, board size, controller messages and reads so far.

**Event tables** (one file per type per cell, as in the source document §9):

- posts;
- reads: one row per post read, repeats kept;
- agent actions: memory before reading, facts read (new vs reactivated),
  posterior, vote, candidates and chosen post or reason for none;
- forgetting events: valid expiries only;
- controller actions: posts read, R, v̂, **true current share** of all 24
  agents' last votes for the target, P(target | R), decision, fact posted,
  budget before and after.

**Episode summary:** messages, reads, the time the budget ran out (if it did),
and the outcomes below.

**Outcomes:**

| outcome | definition |
|---|---|
| main | paired gain at t = 30: m_target(30) controlled minus silent, same episode |
| persistence | the same at t = 40 |
| over time | the paired gain at every grid time; its average over [0, 30] by the trapezoid rule on the grid |
| votes | paired gain in the target's vote share at t = 30 and 40 |
| success | P(m_target(30) ≥ 0.75); first grid time m_target ≥ 0.75 (censored at 30) |
| cost | messages, controller reads, time the budget ran out |
| sensing | v̂ minus the true current share, per controller action |

**Uncertainty:** bootstrap over whole episodes. The silent and controlled runs
of the same episode are resampled together. Step-to-step differences are
computed per episode index and bootstrapped the same way. Events, agents and
grid times are never treated as independent trials.

**S0 check against Simulation 1.** S0 is compared with the Simulation 1 run
`2026-10-07_sim1_pm`, cells `q6 / qc12 / b1 / rho0.75` (both setups, both
targets) and `q6 / rho0.75` (silent). The random draws differ, so the check
is statistical, not number for number (decided 9 October). Fixed list:

- at the end of days 1–30, with S0's t = 1, …, 30 vs Simulation 1's
  `days.parquet`: mean P(A0), mean P(A2), vote shares of A0 and A2, A0 proof
  rate, abstention rate;
- per episode: facts posted, nights stopped by the gate, nights stopped by a proof.

For each quantity, z = (S0 mean − Simulation 1 mean) / (standard error of the
difference, from episodes). **S0 passes if no |z| exceeds 4.5 and at most 10%
of |z| exceed 2.** With about 1,000 comparisons, chance alone gives about 5%
above 2 and almost never one above 4.5. Days 31–40 are not compared;
Simulation 1 has none.

---

## 10. The rate scan, and timing

**Rate scan** (after the bridge, after a look at its results): S6's settings,
with λc ∈ {0.5, 1, 2, 4, 8}. Both setups. Silent reused from S3.
2 setups × (1 silent + 5 rates × 2 targets) = 22 cells, of which 4 are
already in the bridge (λc = 1 and silent). The source document's §8 cautions apply:
- the number of messages actually sent is min(Poisson(30 λc), 30). So rates
  differ in dose and in when the budget runs out, not only in rate;
- report cost, the time the budget runs out and effects together;
- never compare only the episodes that spent the whole budget.

**Timing first.** Before any 1,000-episode run: run 20 episodes of the
slowest cell (S6-like, symmetric, truth), time it, and report the projected
time of the bridge and the rate scan to the researcher. The full bridge starts
only after that.

---

## 11. Randomness

Separate random streams, each keyed by (master seed, episode, purpose, agent ID,
action index or fact/spell index), so one agent's draws do not depend on how
many draws another agent made. Purposes: day order, agent clocks, controller
clock, dawn forgetting, fact lifetimes, reads, votes, posting, controller reads.

- The silent and controlled arms of an episode share every key. They stay
  identical until the controller first changes the board.
- The steps share keys too, where the step's rule uses them. This pairs steps
  by episode index and makes step differences less noisy.
- Controller clocks are shared across rates: the same unit-exponential draws,
  divided by λc.
- Master seed 20261007, as in the source document. Measurements use no randomness.
- A rerun with the same seed reproduces every event.

---

## 12. Decisions taken, and where they differ from the source document

| # | decision | date | source document |
|---|---|---|---|
| 1 | code in MA-CC, `analysis/task003_llm_free/simulation_2/` | 8 Oct | its own folder |
| 2 | run names `sim2_…`, own runs index, results in `results/simulation_2/` | 8 Oct | — |
| 3 | bridge study first, then the rate scan | 8 Oct | rate scan only |
| 4 | both setups | 8 Oct | task003-nosolution only |
| 5 | probability matching only | 8 Oct | same |
| 6 | persistent board, window W = 48 | 8 Oct | same |
| 7 | vote gate kept as an option (`controller_gate`) | 8 Oct | no gate |
| 8 | record m_k(t) and vote shares | 8 Oct | same |
| 9 | 1,000 episodes per cell, timed first | 8 Oct | 300 |
| 10 | ladder order S0 → S6 as in §3; ρ = 0.75 only | 9 Oct | — |
| 11 | S0 checked statistically against Simulation 1, days 1–30 | 9 Oct | — |
| 12 | 40 days in S0–S2 (days 31–40 without controller) | 9 Oct | t = 0–40 |
| 13 | proof stop split into its own step S6; budget B = 30 fixed in all steps | 9 Oct | — |
| 14 | step 5 exception "target already proved" kept from Simulation 1 | 9 Oct | not needed for nosolution |

Unchanged from the source document: Poisson clocks, exponential forgetting with
renewal, recency window, one fact per controller action, budget B = 30,
cutoff at 30 and follow-up to 40, the 0.25 measurement grid, keyed streams,
the verification list (its §12), and the cautions on the rate scan (its §8).

---

## 13. Verification before any study

The source document's §12, adapted to both setups and the bridge settings:

1. setup checks (§1) and exact posteriors agree with `llmfree_core.world`;
2. activation counts and fact survival match their laws (statistical tests
   with stated tolerances);
3. budget never exceeded; nothing read or posted at t ≥ 30 or after the budget
   is spent; agents still act during the follow-up;
4. reading: no self-reads, own posts excluded before the window, no post twice
   in one read, every fact read is active right after;
5. stale expiries never remove a refreshed fact; ρ = 1 never forgets (tested
   even though the study uses 0.75);
6. posting and controller rules on hand-built states, including tie-breaks and
   both exceptions;
7. replaying logged reads and forgetting reconstructs the recorded memories
   and posteriors;
8. same seed, same output; the silent arm does not depend on controller settings;
9. adding measurement times does not change the trajectory;
10. `cleared_nightly` + `dawn` + `days` + `nights` reproduces Simulation 1's
    rules: S0 check (§9), run on 1,000 episodes once the code passes 1–9.

---

## 14. The main grid (decided 9 October 2026)

**The core of Simulation 2 is one bundle**, turned on together: agents **and** the
controller act at random times (Poisson clocks), memory fades continuously, and the
board fades (readers sample the 48 most recent posts). These features are not separated:

- with agents at random times, nightly board clearing would cut an agent's view at
  arbitrary clock times; the board has to fade by itself;
- with agents at random times, there are no nights on which the controller could act,
  so it acts at random times too;
- with continuous memory but agents acting in days, how much an agent forgets would
  depend on its place in the day's order.

**Reference point: Simulation 1** itself, through S0 of the bridge (the S0 check passed,
§9) and Simulation 1's own runs at the same q, qc and ρ (`2026-10-07_sim1_pm`, b = 1).

**The grid** ([`configs/sim2_grid.yaml`](configs/sim2_grid.yaml)), core settings fixed:

| factor | values | note |
|---|---|---|
| setup | symmetric, nosolution | |
| arm | silent, truth, false | |
| q | 3, 6, 12 | Simulation 1's scan |
| qc | 3, 6, 12, 24 | Simulation 1's scan |
| ρ | 0.75, 1 | Simulation 1's scan |
| λc | 0.5, 1, 2, 4, 8 | **replaces Simulation 1's b**: both are facts per time unit; here one fact per message (b = 1), spread in time |
| vote gate | on (`votes`), off (`always`) | parallel arms, not steps |
| proof stop | on, off | parallel arms; run separately only for symmetric truth, the only place it can act |

Budget: 30 messages per episode in every cell. At high rates with the gate off, it runs
out early (at λc = 8 by about t = 4); the time it runs out is recorded and reported.

Cells: 12 silent (2 setups × 3 q × 2 ρ; silent cells do not depend on the controller
factors) + 1,200 controlled. 1,000 episodes each. Every table is recorded for the main
cells (q = 6, qc = 12, ρ = 0.75: 52 cells); elsewhere posts, controller actions,
measurements and episode summaries only.

Seed 20261007, as in the bridge: the grid's cell at q 6, qc 12, ρ 0.75, λc 1, gate on, stop on
is identical to the bridge's S4 (tested). Any finding singled out by the analysis is checked
on a confirmation run with seed 2027.
