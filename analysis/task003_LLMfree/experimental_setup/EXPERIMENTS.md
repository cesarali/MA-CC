# The new experiments — what we are running

**Status.** The two setups are **frozen** (5 October 2026); the run settings
below are still preliminary. Nothing launched.

> **Corrections, 5 October.** (1) task003-symmetric now has **two** controller
> pools, one per target, as originally specified on 2 October. An earlier
> version of this file gave both setups one shared neutral pool; that was a
> change made without the author's agreement. (2) `origin/darius-MA-v1` was
> merged on 4 October (commit `4d570b3`); statements below about what is
> "unmerged" describe the branch before that.

This file is the plain explanation. For building the configs, see
[`designs/AGENT_RUNBOOK.md`](designs/AGENT_RUNBOOK.md).

---

## 0. Glossary

Every term used below, defined once.

| term | meaning |
|---|---|
| **post-meeting plan** | `control_efficiency_project_plan_the post-meeting plan_post-meeting_plan.md` — **the team's agreed experiment plan, written after a meeting.** It names leads (Alan + Chris on parallelization, César + Ramses on statistics) and fixes 30 rounds, ρ ∈ {0.75, 1.0}, two budget protocols, full vs report-only communication, full vs partial board sensing, truth vs false control, 50 episodes per cell, and a 1,700-episode first study. "The post-meeting plan says X" means that document specifies X. |
| **plan priority 1–4** | The post-meeting plan's own ordering of four experiments. **1** = full-episode budget `B` + full communication. **2** = per-round budget `b` + full communication. **3** = full-episode `B` + report-only. **4** = per-round `b` + report-only. The plan calls 1 and 2 "the immediate priority" and postpones 3 and 4. |
| **task003-symmetric** | Our setup where the 24 agents **collectively hold a proof** of the right answer. Two task directories, `task003_symmetric_to_a0` and `task003_symmetric_to_a2`, identical except for the controller's pool. |
| **task003-nosolution** | Our setup where they **do not**: the population's posterior is a coin flip between truth and `ALLOCATION_2`. Task directory `task003_nosolution`. |
| **ρ (rho), persistence** | Probability an agent remembers each fact it holds, per round. ρ = 1 is perfect memory. |
| **b** | Control budget **per round**: facts the controller may post each round. |
| **B** | Control budget **per episode**: one reservoir for all 30 rounds; the controller chooses when to spend it. |
| **arm** | silent (no controller) / truth control (`ALLOCATION_0`) / false control (`ALLOCATION_2`). |
| **lean** | The allocation a fact most favours when read alone. |
| **cell** | One (setup, arm, ρ, settings) combination. |
| **pool** | The facts the controller may post. task003-symmetric: **two** 12-fact pools, the A0 pool used only when targeting `ALLOCATION_0`, the A2 pool only when targeting `ALLOCATION_2`. task003-nosolution: **one** 12-fact pool for both targets. |
| **decisive facts** | A 6-fact set that proves `ALLOCATION_0`. Held by the agents in task003-symmetric, absent in task003-nosolution. |

## 1. Two setups, same world

Both are `task_003` — latent vector `(3,1,1,2,2,1,1,1,2)`, scores 8 / 4 / 6,
`ALLOCATION_0` correct, `ALLOCATION_2` the wrong target we steer toward. Same
49-fact universe, same 24 agents, one fact each. **They differ only in which
facts the agents hold.**

| | `task003-symmetric` | `task003-nosolution` |
|---|---|---|
| distinct facts held | 18 | 15 |
| agents per allocation A0/A1/A2 | 8 / 8 / 8 | 8 / 8 / 8 |
| agents' joint posterior | **[1.0, 0, 0]** | **[0.5, 0, 0.5]** |
| decisive facts held | 6 of 6 | 0 of 6 |
| proofs the population can assemble | 12 | **0** |
| **can the population prove the truth?** | **yes** | **no** |
| accuracy ceiling | 1.0 | 0.5 |
| the question it answers | Can the swarm find a proof it collectively owns, and can a controller stop it? | Can a controller tip a swarm that is genuinely undecided? |

**The controller pools differ between the setups.**

- **task003-symmetric: two pools.** The A0 pool (12 facts favouring A0) is used
  only when the controller targets A0; the A2 pool (12 facts favouring A2) only
  when it targets A2. The pools share no fact, contain no decisive fact, cannot
  prove their target, and are matched pair by pair in weight. The agents were
  chosen so both controllers bring the same number (10) and weight of facts the
  agents do not already hold.
- **task003-nosolution: one pool**, 4 facts favouring each allocation, close to
  the prior, proving nothing, sharing no fact with the agents. Both targets use
  it, so the controller's menu is identical whichever way it steers.

Full property tables: [`designs/README.md`](designs/README.md).

---

## 2. Arms

Each setup has **two controlled arms plus one silent baseline** — so three kinds
of run:

| arm | `control` block | `target` |
|---|---|---|
| **silent** (no-control baseline) | omitted entirely | — |
| **truth control** | present | `ALLOCATION_0` |
| **false control** | present | `ALLOCATION_2` |

The silent arm is a property of the *agent* setup, not the controller, so it is
reused across controller settings — but **not across setups**, because the two
setups have different agent fact sets and therefore different uncontrolled
dynamics. Each setup needs its own silent baseline.

Crossed with persistence **ρ ∈ {0.75, 1.0}** (ρ = how likely an agent is to
remember each fact it holds, per round).

---

## 3. Settings: two phases of controller, three phases of run

### Phase 1 — the clean measurement

```
agents:     communication_profile: report_only
controller: controller_actuation_mode: adaptive_communication
            controller_authoring:      deterministic
            message_mode:              recommendation_only
```

Yes — this is exactly the combination for the first 12 cells.

**Why.** Reports are the only channel that carries facts, and facts are the
entire treatment. `adaptive_communication` is one of only two actuation modes
that can post a pool fact at all. `deterministic` keeps a language model out of
the controller loop, so the dose is a known function of the sensed state rather
than something improvised — the treatment is measured, not estimated.
`recommendation_only` keeps the separate peer-slot channel factless, so every
controller fact reaches the agents through the pool, where the design controls
it.

Under `report_only` the controller's only permitted message type is `REPORT`, so
`deterministic` authoring is **genuinely deterministic**: it posts a report every
time it fires.

### Phase 2 — add the LLM controller

```
agents:     communication_profile: report_only       (unchanged)
controller: controller_authoring:      llm_authored  (the only change)
```

**Why second, not first.** Phase 1 is the reference the LLM controller has to be
scored against; "the LLM controller achieved η = x" means nothing without it.
Changing one thing at a time also means any difference is attributable.

Derived policy: `llm_authored_fixed_report_only_v1`.

### Phase 3 — add full communication

```
agents:     communication_profile: full_communication
controller: controller_authoring:      llm_authored
```

**Why these two together.** A `REQUEST` carries no fact, so it adds a channel
the measured quantity is blind to, and `deterministic` authoring under
`full_communication` would introduce a seeded random mode draw (see below).

> ⚠️ **Correction (2026-10-04).** An earlier version of this file said "the post-meeting plan
> agrees on the ordering — report-only is its priority 1–2". **That is backwards.**
> the post-meeting plan's priorities are **1** = `B` + full communication, **2** = `b` + full
> communication, **3** = `B` + report-only, **4** = `b` + report-only. So
> **full communication is the post-meeting plan's immediate priority and report-only is
> postponed** — our phase 1 is the post-meeting plan's *priority 4*, its lowest. See §3d for how
> to reconcile that; it is a team decision, not ours to make silently.

Derived policy: `llm_authored_full_communication_v1`. The controller picks one
mode per round (`REPORT`, `REQUEST` or `DIRECTIVE`) and fills every slot with
it. Adding `controller_allow_mixed_message_types: true` instead permits a
different type per slot and gives
`llm_authored_mixed_full_communication_v1`.

### The fourth combination, and why we skip it

`deterministic` + `full_communication` is **legal but worse than it looks.**
With three permitted message types, the deterministic chooser
(`contextual_weighted_v1`) makes a **seeded random draw** among them with
context-dependent weights. So "deterministic" there means *no language model*,
not *no randomness* — it introduces an unmotivated stochastic mode choice.

Under `report_only` this does not arise, because the allowed set is just
`(REPORT,)` and the draw is trivial.

**So: keep `report_only` for the clean measurement, and when moving to
`full_communication` use `llm_authored`,** where the mode choice is part of the
policy being studied rather than a coin flip. Run the fourth combination only if
a reviewer asks for it.

---

## 3b. The budget protocol — decided

### Which branch has what

| | meaning | available |
|---|---|---|
| **`b`, per round** | the controller gets a fresh allowance of `b` facts every round | **both branches.** On `dev/rsanchez` this is the *only* behaviour and the default: the module docstring describes "an exact budget `b` of randomly placed controlled positions", one decision per round |
| **`B`, per episode** | one reservoir for all 30 rounds; the controller decides when to spend it | **`darius-MA-v1` only** (`controller_budget_scope: episode`) |

So per-round `b` does work on `dev/rsanchez`. What is missing there is the
*option to choose* — `controller_budget_scope` does not exist, so per-round is
simply what you get.

> ### ⚠️ A silent-failure trap
>
> `from_options` on `dev/rsanchez` **accepts and discards** keys it does not
> know, with no error and no warning. Verified by direct call. So on that branch:
>
> - `controller_budget_scope: episode` is **silently ignored** and the run
>   executes **per-round** — a config that looks like the post-meeting plan's priority-1
>   experiment but is actually priority-2.
> - `controller_fact_pool_mode: balanced` is **silently ignored**, and the pool
>   falls back to `controller_report_pool_mode: frozen`, which returns the
>   task's `facts/controller_reportable_facts.json`. If that file is copied from
>   `task_003` unchanged, the run uses **the original defective 24-fact pool** —
>   the exact bug this whole redesign exists to fix, reintroduced invisibly.
>
> Mitigation, and it is cheap: write each directory's 12-fact pool into **both**
> `controller/balanced_fact_pool.json` *and*
> `facts/controller_reportable_facts.json` in the new task directories, so even
> a silent fallback lands on the right pool. See
> [`designs/AGENT_RUNBOOK.md`](designs/AGENT_RUNBOOK.md) §2.

### Decision: `b = 3` per round for phases 1–3

Two reasons, and the first is about our estimator rather than about control.

**An episode reservoir breaks the exact path-KL calculation.** The remaining
budget is a state variable that evolves over the episode and is *not* in our
4-state coarse-graining. It is also strongly non-Markov in a specific way: it
only ever decreases, so it carries the episode's entire intervention history.
The four Markov tests already found about 0.015 nats/step of unexplained memory
at ρ = 0.75 with a *fixed* per-round dose; an episode reservoir adds a second
memory channel on top of that.

To stay Markov we would have to add remaining-budget to the state. At B = 90
that is 91 values; binned to three levels it gives 4 × 3 = 12 states. Our bin
sweep showed **8** states already leaves thinly-estimated rows at ~1,800
transitions per cell, so 12 needs substantially more episodes.

**It also makes the dose endogenous.** A controller that spends early produces
different exposure from one that spends late, so comparing the truth and false
arms would confound policy *timing* with evidence *content* — the same class of
confound as the activation gate we are switching off.

### But yes, `B` is worth pursuing — as phase 4, and for a specific reason

the post-meeting plan is right that it is the richer object: with a reservoir, *temporal
allocation becomes part of the controller policy*. "The controller learned when
to intervene" is a stronger result than "the controller posted three facts every
round."

The clean way to get it is **the matched pair**, which is the post-meeting plan's own design:

```
b = 3  per round   over 30 rounds  =  90 total
B = 90 per episode over 30 rounds  =  90 total
```

Total capacity is identical, so the contrast isolates **one thing: whether the
controller may choose its own timing.** That is a publishable comparison and it
needs no new factor beyond the scope flag. Likewise `b = 6` against `B = 180`.

**Conditions before trusting any path quantity from it:** augment the
coarse-grained state with a 3-level remaining-budget coordinate, and **re-run
the four Markov tests on the augmented chain** before reporting anything. The
machinery is already there — `../coarse_graining/` and
`../coarse_graining/code/four_tests.py`.

**If we can only afford one**, run per-round at full precision rather than both
at half. An uninterpretable episode-scope result is worth nothing, and that is
precisely what happened to the last batch.

## 3c. Do we need the merge? — corrected

**No, not for `b = 3`, and not for phase 1 either.** An earlier version of these
notes said the merge was a prerequisite. That was wrong, and tracing why turned
up a design decision that had been hidden.

### `b = 3` needs nothing

`intervention_budget: 3` is per-round on `dev/rsanchez` and it is the *only*
behaviour there. The merge adds the *choice* of scope, not the per-round
protocol.

### The per-directory pool does not need the merge either

`controller_report_pool_mode: frozen` — the default on both branches — returns
`task.controller_reportable_fact_ids` **with no reference to the target**. It is
already target-independent. The archived bug was never in this mechanism; it was
in the pool's *content*, which had been selected for `ALLOCATION_2` and then
reused.

So the pool is chosen by the task directory: writing the right 12 facts into
each directory's `facts/controller_reportable_facts.json` gives task003-symmetric
its two pools (one directory per target) and task003-nosolution its shared pool,
with no merge. And `validate_truthful_report_task` explicitly permits
the target to be either the task's declared target *or* the ground truth, so
both arms pass. Its code comment is the original justification:

> A frozen adversarial pool is also scientifically valid for the ground-truth
> counterpart: every report remains a true task fact and the decisive subset
> proves the ground truth.

That reasoning is what allowed a decoy-selected pool to serve the truth arm.
With one pool per target, or a neutral shared pool, the same permission becomes
legitimate.

### The thing that actually matters: where target alignment comes from

With **`deterministic`** authoring the facts are chosen by
`select_truthful_reports`, ranked on

```
(not cooldown_eligible, reuse_count, live_count, -base_score, tie_hash, fact_id)
```

where `base_score` comes from `task.controller_fact_scores`. The tie hash is
over `(episode_seed, task_id, round_index, fact_id)` — **not the target.** So if
the scores are absent or equal, **the fact choice is target-blind**: both arms
rotate through the same 12 facts in the same order, and only the recommendation
text differs.

With **`llm_authored`** the model is told the target and picks from the ranked
eligible set, so alignment comes from the model and target-blind scores are
fine.

### And this is what Darius's `balanced` mode does

His change **forces the score to zero** whenever the pool mode is not `frozen`:

```python
base_score = (
    0.0
    if self.controller_fact_pool_mode != CONTROLLER_FACT_POOL_FROZEN
    else float(base_scores.get(fact_id, 0.0))
)
```

That is deliberate and defensible — the task's stored scores were computed for
its declared false target, so reusing them with a balanced pool would re-import
the slant. But the consequence is that **`controller_fact_pool_mode: balanced`
plus `deterministic` authoring cannot select facts toward a target at all.**

Which means his `29-09-2026-task004-classical-b3-gated-balanced` truth and false
arms differ only in the recommendation text, not in which facts get posted.
That may be exactly what he intended; it is worth asking him, and it is worth us
not inheriting it by accident.

### So, concretely

| what we want to run | merge needed? |
|---|---|
| `b = 3` per round | **no** |
| the right pool per target | **no**: `frozen` mode plus each directory's own `facts/controller_reportable_facts.json` |
| `deterministic` authoring that **selects facts toward its target** | **no**, and in fact the merge's `balanced` mode would *prevent* it. Needs per-target `controller_fact_scores` |
| `llm_authored` authoring | **no** |
| **`B`, episode-scope budget** | **yes** |
| `all_nondecisive` pool, `public_ledger` controller memory | yes (we want neither) |

**Phase 1 runs on `dev/rsanchez` today.** The merge is for phase 4.

### The one real cost of going unmerged

`controller_fact_scores` is a **single field per task**, so it cannot encode
alignment toward two opposite targets at once. **task003-symmetric already has
one directory per target, so each scores toward its own target.** For
task003-nosolution, two options:

1. **Two task directories** with identical facts and identical pool, differing
   *only* in the score field (one scored toward A0, one toward A2). Ugly but
   fully auditable: a diff shows one field changed.
2. **Accept target-blind fact choice** and let the arms differ only in the
   recommendation. That is a narrower but perfectly clean experiment — *does a
   recommendation steer, with evidence held constant?* — and it is what Darius's
   balanced configs already do.

Option 1 is what our design intends. Option 2 is worth running *as well*,
because the contrast between them separates **evidence selection** from **bare
recommendation**.

## 3d. Our phases against the post-meeting plan's priorities — the real relationship

the post-meeting plan's four experiments, in its own order:

| the post-meeting plan priority | budget | communication |
|---|---|---|
| **1** — immediate | `B` full-episode | **full communication** |
| **2** — immediate | `b` per-round | **full communication** |
| 3 — postponed | `B` full-episode | report-only |
| 4 — postponed | `b` per-round | report-only |

Mapping our phases onto that:

| our phase | = the post-meeting plan priority |
|---|---|
| phase 1 — report-only, `b`, deterministic | **4** (its lowest) |
| phase 2 — report-only, `b`, llm_authored | **4** |
| phase 3 — full communication, `b`, llm_authored | **2** |
| phase 4 — full communication, `B`, llm_authored | **1** |

So our ordering is **the reverse of the post-meeting plan's**. That is a real disagreement and it
needs team agreement, not a quiet decision by us.

### The honest case for our ordering

It is **not** "report-only is more important". It is that the post-meeting plan's own §4
execution rule says:

> Implement → validate → freeze → run Experiments 1–2 at 50 episodes/cell →
> assess statistical reliability → increase repetitions only if needed

**Our phase 1 is the "validate" step, not a competing experiment.** It is the
cheapest, cleanest cell in the whole design — one message type, no language
model in the controller loop, a dose that is a known function of the sensed
state — and it is the configuration in which our estimator is most likely to
work. The last batch failed precisely because nobody checked that the
information quantities were estimable before running the full grid.

So the proposal is: **phase 1 as the validation gate (12 cells, 1,200
episodes), then straight to the post-meeting plan priority 2, then priority 1.** Phases 3 and 4
*are* the post-meeting plan's priorities 2 and 1.

### If the team wants the post-meeting plan's order kept strictly

Then start at **priority 2** (`b` per-round + full communication), not priority
1. The per-round budget is the part our estimator argument is really about — an
episode reservoir adds a hidden state variable (§3b) — and priority 2 already
gives the post-meeting plan its full communication. That is a one-step departure instead of
four, and it is the compromise worth proposing at the next meeting.

## 3e. Can the silently-ignored options be fixed?

**Partly, and a general fix is riskier than it looks.**

The problem: `from_options` reads the keys it knows with `options.get(...)` and
**ignores the rest silently**. So `controller_budget_scope: episode` on
`dev/rsanchez` runs per-round while the config claims otherwise.

**A naive strict "reject unknown keys" check would break every config in the
repository.** Measured across all configs on both branches, these keys appear in
`control.options` but are *not* read by the relational controller's own
`options.get` calls:

| key | configs using it |
|---|---:|
| `target` | 139 / 197 |
| `sensor_sample_size` | 139 / 197 |
| `threshold` | 139 / 197 |
| `beta` | 136 / 194 |
| `policy` | 131 / 189 |
| `template_version` | 23 |
| `agent_ids`, `forced_value`, `until_interaction` | 3 each |

The first six are read by the **parent** class
(`hidden_bench/imitation/controller.py`), and the last three belong to *other*
control mechanisms. So any strict check must (a) collect the key set across the
whole class hierarchy and (b) be per-mechanism. Doable, but it touches a path
every study in the repo depends on, and five people share this branch.

### Recommended instead: a targeted guard

Two cheap steps that close the actual hazard without that risk:

1. **Merge.** After the merge both dangerous keys exist and are read, so the
   specific footgun disappears.
2. **Assert at build time, not in the library.** The runbook already requires:

   ```python
   c = RelationalRoundBudgetedControl.from_options(opts)
   assert hasattr(c, "controller_budget_scope"), "darius-MA-v1 not merged"
   assert hasattr(c, "controller_fact_pool_mode"), "darius-MA-v1 not merged"
   ```

   This catches the one case that matters, costs nothing, and risks no other
   config.

A proper strict-key validator is worth doing as its own piece of work, with the
MRO walk and per-mechanism key sets, and with the repository's two latent test
breakages fixed first so the change can be validated. See
the merge assessment note (deleted 6 October; see [`../README.md`](../README.md#recovering-deleted-material)).

## 4. How many runs is that?

A **cell** = one (setup, arm, ρ, settings) combination.

| phase | agents | authoring | silent cells | controlled cells | new cells | cumulative |
|---|---|---|---|---|---:|---:|
| **1** | report_only | deterministic | 2 × 2 = **4** | 2 × 2 × 2 = **8** | **12** | **12** |
| **2** | report_only | llm_authored | 0 — **reused from phase 1** | 2 × 2 × 2 = **8** | **8** | **20** |
| **3** | full_communication | llm_authored | 2 × 2 = **4** | 2 × 2 × 2 = **8** | **12** | **32** |

Counting is `setups × arms × ρ`: 2 setups × 2 ρ for silent, and
2 setups × 2 directions × 2 ρ for controlled.

Phase 2 needs no new silent runs because nothing about the agents changed and a
silent run has no controller. Phase 3 does, because the agents' communication
profile changed.

### Episodes

At **100 episodes per cell, 30 rounds each**:

| | cells | episodes |
|---|---:|---:|
| phase 1 alone | 12 | **1,200** |
| phases 1–2 | 20 | 2,000 |
| phases 1–3 | 32 | **3,200** |

**On 100 rather than the post-meeting plan's 50:** the bootstrap resamples whole episodes, so
precision goes as 1/√episodes. We measured a ±18% interval on the path
divergence at 168 episodes; 50 extrapolates to about **±33%**, which will not
separate two control directions. the post-meeting plan permits extending to 100 "if the
statistical analysis indicates" — it does. But this doubles the bill, so it is
César's call. The calculation is in
the 2 October design draft, §3 (deleted 6 October; see [`../README.md`](../README.md#recovering-deleted-material)).

### Later factors, deliberately not in the grid above

Add one at a time, after phase 1 is analysed: budget level (`b = 6`), budget
scope (`episode`, `B = 90` and `180`), partial controller sensing, partial board
reading by agents. Crossing everything at once is what made the last batch
uninterpretable.

---

## 5. Where the settings are

| what | where |
|---|---|
| **the settings themselves**, annotated with a reason per line | [`designs/config_template.yaml`](designs/config_template.yaml) |
| **the phase 2 and 3 overrides**, machine-readable | [`designs/variants.yaml`](designs/variants.yaml) |
| **step-by-step build instructions for an agent** | [`designs/AGENT_RUNBOOK.md`](designs/AGENT_RUNBOOK.md) |
| the two setups' fact lists, every property, and how they were built | [`designs/README.md`](designs/README.md) and the `*.json` beside it |
| why each protocol option was chosen, what Darius built, the sample-size calculation | dated notes, deleted 6 October; see [`../README.md`](../README.md#recovering-deleted-material) |

---

## 6. Two prerequisites

1. **Build the three task directories** (`task003_symmetric_to_a0`,
   `task003_symmetric_to_a2`, `task003_nosolution`); see
   [`designs/AGENT_RUNBOOK.md`](designs/AGENT_RUNBOOK.md) §2. Use
   `controller_report_pool_mode: frozen` with each directory's pool written into
   `facts/controller_reportable_facts.json`. Merge `origin/darius-MA-v1` before
   phase 4 (episode-scope `B`), which needs `controller_budget_scope`.
2. **Decide the scoring question in §3c for task003-nosolution**: per-target
   `controller_fact_scores` in two task directories, or target-blind fact
   choice. task003-symmetric already has one directory per target. This determines whether the arms differ in *which facts* get
   posted or only in the *recommendation*.
