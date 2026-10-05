# Next symmetric experiment — PRELIMINARY design

**Status: PRELIMINARY.** Nothing here is frozen. Written 2 October 2026 before
the controller choice is settled. Read `prior_art_2026-10-02.md` first: most of
the machinery already exists on `darius-MA-v1` and that changes the design.

---

## 1. What changed from the original sketch

The sketch was: three runs — silent, control toward A0 with a symmetric A0 pool,
control toward A2 with a symmetric A2 pool — then repeat with the six decisive
facts removed from the agents. Six experiments.

**Two of those three choices should change.**

**Drop the two target-specific pools.** *(Superseded 2026-10-02: use the
neutral shared pool in [`designs/`](../designs), not Darius's balanced pool
directly — his whole pool has joint posterior 1.0 for the truth and contains a
size-6 proof, so his truth controller could simply hand over a proof. The
reasoning below for preferring one shared pool over two still holds.)*
Use a single *balanced* pool instead: one 27-fact pool, 9 facts leaning to each allocation, strength-matched
across halves. Two pools means the A0 and A2 arms differ in both *what the
controller may say* and *what it chooses to say*, and those cannot then be
separated. One pool makes the menu identical and attributes every directional
difference to the selection policy. It is also already built, frozen and
audited, and excluded from the pairing hash so the baseline is reusable.

**Keep the decisive-fact factor — but it is not new.** *(Corrected
2026-10-02: an earlier version of this document called it novel. It is not.)*
`task_004` on `darius-MA-v1` **is** `task_003` with the decisive facts removed;
its `task.json` records `derived_from: {task_id: task_003, change: "decisive
facts removed with their holders"}`, and the set arithmetic checks exactly.

The factor is still worth keeping, because Darius's version removes the decisive
facts **together with their holders**, dropping the population from 24 agents to
15. That confounds proof availability with population size, with the number of
distinct facts (21 → 15), and with per-fact redundancy. Holding the population at
24 and *replacing* the six facts isolates the factor. Both corrected setups are
in [`designs/`](../designs).

So the design is **3 arms × 2 decisive conditions**, not 6 separate experiments:

| | agents hold the 6 decisive facts | decisive facts withheld |
|---|---|---|
| **silent** | cell 1 | cell 4 |
| **control → A0 (truth)** | cell 2 | cell 5 |
| **control → A2 (decoy)** | cell 3 | cell 6 |

Crossed with persistence ρ ∈ {0.75, 1.0} → **12 cells**.

---

## 2. Which controller — the decision that is actually open

Both candidates are in `prior_art_2026-10-02.md` §3. The recommendation:

> **First pass: controller B, the deterministic per-round one.**
> `controller_authoring: deterministic`, `advocacy_schedule: soft`,
> `intervention_budget: 3`, `controller_budget_scope: per_round`.

Three reasons, in order of weight:

1. **Our estimator needs it.** The exact path-KL calculation
   (`40_information_estimates/`) assumes the coarse-grained process is Markov.
   An episode-scope budget makes the controller's remaining reservoir a hidden
   state variable that evolves over the episode — a second memory channel on top
   of the one we already failed to capture at ρ = 0.75. A per-round budget
   resets every round and adds no such state.
2. **Deterministic authoring makes the intervention sequence a known function
   of the sensed state** rather than something a language model improvises. The
   treatment is then measured, not estimated.
3. **It is the reference the LLM controller needs to be compared against.**
   Without it, "the LLM controller achieved η = x" has no scale.

**Second pass: controller A**, `llm_authored` with `advocacy_schedule: always`
and `intervention_budget: 90` at episode scope — v0.4's priority 1. This is the
interesting object, and the one worth a paper. Run it once the pipeline is
validated on B, not before.

This ordering inverts v0.4's priority list. v0.4 puts the full-episode budget
first because it is the richer control problem; we put it second because the
*estimator* is the thing most likely to fail, and B is the configuration in
which it is most likely to work. Worth raising with César.

---

## 3. Is 100 episodes × 15 rounds enough?

**No on rounds — use 30. Probably no on episodes either, and 50 is worse.**

### Rounds: 15 → 30

The archived study used 15 rounds (`round_index` 0–14). v0.4 specifies 30 and
Darius's `task_004` configs already use 30. Take 30, for two independent
reasons:

- The path divergence is still climbing at h = 14 in every cell
  (`40_information_estimates/estimates_curves.csv`), so 15 rounds truncates the
  quantity mid-growth. 30 rounds reaches h = 29.
- Consensus needs room to form. v0.4's wording — "giving agents sufficient time
  to exchange information" — is the same judgement.

### Episodes: what the existing uncertainty actually shows

From `40_information_estimates/estimates_chain.csv`, the bias-corrected cluster
bootstrap on the path KL:

| ρ | controlled episodes | K (nats) | CI half-width | as % of K |
|---|---:|---:|---:|---:|
| 0.75 | 337 | 3.045 | 0.257 | **8%** |
| 1.00 | 168 | 1.931 | 0.353 | **18%** |

The bootstrap resamples **whole episodes**, so precision scales roughly as
1/√(episodes), not with total rounds. Extrapolating from the ρ = 1 row:

| episodes per cell | expected CI half-width |
|---|---|
| 168 (what we had) | 18% |
| 100 | ≈ 23% |
| **50 (v0.4 initial)** | **≈ 33%** |

A 33% interval on the headline quantity will not separate two control
directions. **Recommend 100 episodes per cell**, and treat v0.4's 50 as the
pilot it is described as — v0.4 §2 already allows extension to 100 "if the
statistical analysis indicates that the additional repetitions are necessary."
This is that indication, available in advance.

Note the two options are *not* equivalent despite similar totals:

| | transitions per cell | independent units |
|---|---:|---:|
| 100 episodes × 15 rounds | 1,400 | 100 |
| 50 episodes × 30 rounds | 1,450 | 50 |
| **100 episodes × 30 rounds** | **2,900** | **100** |

The first two fit the 4-state kernel about equally well. They differ by a factor
of two in the only thing that sets the error bars. Rounds buy horizon; episodes
buy precision. We need both, and they are not substitutes.

### The unit question that must be settled first

Our analysis established that the **initialization** is the only independent
unit — 60 initializations produced 120–347 episodes, because several episodes
share one initialization. Every split and bootstrap resamples whole
initializations.

So **"50 episodes per cell" in v0.4 is ambiguous**, and the two readings differ
by a factor of several in effective sample size. If it means 50 initializations,
the bootstrap has fewer clusters than the study we just called unreliable. This
needs resolving with César and Darius before anything is frozen; it is the
single highest-leverage question on this page.

### Cost

100 × 30 is double v0.4's 50 × 30. Doubling the bill for this is a judgement
call, not a technical one — but the honest version is that at 50 episodes the
information estimates will likely land where the last batch did: visible
intervention, uninterpretable direction.

---

## 4. What the agents and the controller do

### Agents

Unchanged from the archived setup, which worked:

- 15 agents (`task_004`) or 24 (`task_003`), each holding a private packet of
  facts, each round posting a report to the board and casting a ballot.
- Front-page board: `message_lifetime_rounds: 1`. Last round's board is visible;
  there is no accumulated history.
- Epistemic persistence ρ applied per agent, per fact, per round.
- `communication_profile: report_only` for the first pass. `full` (reports plus
  requests) is a second factor, and v0.4 postpones it to priorities 3–4.

**The one change: the decisive-fact factor.** *(Updated 2026-10-02: this is now
built.)* Each agent holds **exactly one fact**, not a packet — 24 agents, 24
slots. Both setups hold the population at 24 agents with slot-level lean balance
8/8/8, and differ only in whether the decisive facts are present. See
[`designs/`](../designs) for the fact lists, and [`builders/`](../builders) for the
search that produced them.

### Controller

Per §2, first pass:

```yaml
target: ALLOCATION_0        # or ALLOCATION_2
sensing_mode: board
sensor_sample_size: 100
policy: soft_target
threshold: 0.5
beta: 4.0
intervention_budget: 3
controller_budget_scope: per_round
advocacy_schedule: soft
controller_authoring: deterministic
controller_round_budget_mode: exact
controller_fact_pool_mode: balanced
controller_report_selection_strategy: target_preserving_v1
controller_timing: dawn_only
controller_memory_mode: none
```

It senses the board at dawn, and if the population state is on the wrong side of
`threshold` it posts up to 3 facts from the balanced pool chosen to move the
posterior toward its target.

**Report the dose, do not assume it.** `advocacy_schedule: soft` keeps the
activation gate, and the previous batch fired on only 32% / 54% of rounds. Per
`task_003_redesign.md` recommendation 5, measure actual activations and delivered
reports as a separate quantity, and **do not condition on realized firing rounds
and call the result a causal effect.**

---

## 5. Open questions, in priority order

1. **Does "50 episodes per cell" mean 50 initializations or 50 trajectories?**
   (§3) Blocks the sample-size decision.
2. **Controller B first or controller A first?** (§2) We recommend B; v0.4
   implies A. Needs César.
3. **`task_003` or `task_004`?** The balanced pool is frozen for `task_004`
   only, and all our coarse-graining was calibrated on `task_003`. Rebuilding
   the pool for `task_003` is one script run; recalibrating the binning for
   `task_004` is also cheap. Prefer `task_004` — it inherits Darius's frozen
   pool, its 30-round configs and its reusable no-control baselines.
4. **How is "partial-board sensing" defined?** (prior art §4) Probably a reduced
   `sensor_sample_size`, but confirm rather than assume.
5. **Is the frozen-pool target fix actually in the tree?** (prior art §6) It is
   on neither `dev/rsanchez` nor `darius-MA-v1`.
6. **The decisive-fact-withheld packets do not exist yet.** (§4) Needs writing
   and auditing.

---

## 6. What is not in this design, deliberately

- **Multiple worlds.** Everything here is one task. `task_003_redesign.md` is
  right that more repetitions of one world do not substitute for more worlds,
  and `generality.json` shows the 12 additional worlds vary structurally. Single
  world is a stated limitation for this round, not a claim about the family.
- **The full v0.4 crossing.** v0.4's priority study is 1,700 episodes across
  budget protocol × budget level × sensing × direction × ρ. Crossing all of it
  before the estimator is known to work on one cell repeats the last batch's
  mistake. Add one factor at a time, per `task_003_redesign.md`
  recommendation 4.
- **Entropy production.** Still no justified reverse protocol. The irreversibility
  number stays an internal diagnostic and does not go in the paper under that name.
