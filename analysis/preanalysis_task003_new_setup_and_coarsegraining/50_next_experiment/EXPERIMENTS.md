# The new experiments — what we are running

**Status: PRELIMINARY.** Nothing frozen, nothing launched. 3 October 2026.

This file is the plain explanation. For building the configs, see
[`designs/AGENT_RUNBOOK.md`](designs/AGENT_RUNBOOK.md).

---

## 1. Two setups, same world

Both are `task_003` — latent vector `(3,1,1,2,2,1,1,1,2)`, scores 8 / 4 / 6,
`ALLOCATION_0` correct, `ALLOCATION_2` the wrong target we steer toward. Same
49-fact universe, same 24 agents, one fact each. **They differ only in which
facts the agents hold.**

| | `task003-symmetric-v2` | `task003-nosolution-v2` |
|---|---|---|
| distinct facts held | 16 | 15 |
| slot lean A0/A1/A2 | 8 / 8 / 8 | 8 / 8 / 8 |
| agents' joint posterior | **[1.0, 0, 0]** | **[0.5, 0, 0.5]** |
| decisive facts held | 6 of 6 | 0 of 6 |
| proofs the population can assemble | 8 | **0** |
| **can the population prove the truth?** | **yes** | **no** |
| accuracy ceiling | 1.0 | 0.5 |
| the question it answers | Can the swarm find a proof it collectively owns, and can a controller stop it? | Can a controller tip a swarm that is genuinely undecided? |

Both use the **same** 12-fact controller pool: lean 4/4/4, strength-matched,
joint posterior on the prior, **proving no allocation**, and **zero overlap**
with what the agents hold. One pool shared by both steering directions, so the
controller's menu is identical whichever way it steers and any directional
difference is the *selection policy*, not the menu.

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

**Why these two together, and why `full_communication` last.** A `REQUEST`
carries no fact, so it adds a channel the measured quantity is blind to. Plan
v0.4 agrees on the ordering — report-only is its priority 1–2, full
communication its 3–4.

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
>   executes **per-round** — a config that looks like v0.4's priority-1
>   experiment but is actually priority-2.
> - `controller_fact_pool_mode: balanced` is **silently ignored**, and the pool
>   falls back to `controller_report_pool_mode: frozen`, which returns the
>   task's `facts/controller_reportable_facts.json`. If that file is copied from
>   `task_003` unchanged, the run uses **the original defective 24-fact pool** —
>   the exact bug this whole redesign exists to fix, reintroduced invisibly.
>
> Mitigation, and it is cheap: write our 12-fact pool into **both**
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

v0.4 is right that it is the richer object: with a reservoir, *temporal
allocation becomes part of the controller policy*. "The controller learned when
to intervene" is a stronger result than "the controller posted three facts every
round."

The clean way to get it is **the matched pair**, which is v0.4's own design:

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
machinery is already there — `../30_coarse_graining/` and
`../scripts/run_four_tests.py`.

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

### The shared neutral pool does not need the merge either

`controller_report_pool_mode: frozen` — the default on both branches — returns
`task.controller_reportable_fact_ids` **with no reference to the target**. It is
already target-independent. The archived bug was never in this mechanism; it was
in the pool's *content*, which had been selected for `ALLOCATION_2` and then
reused.

So writing our 12 neutral facts into
`facts/controller_reportable_facts.json` gives the shared pool directly, on
`dev/rsanchez`, unmerged. And `validate_truthful_report_task` explicitly permits
the target to be either the task's declared target *or* the ground truth, so
both arms pass. Its code comment is the original justification:

> A frozen adversarial pool is also scientifically valid for the ground-truth
> counterpart: every report remains a true task fact and the decisive subset
> proves the ground truth.

That reasoning is what allowed a decoy-selected pool to serve the truth arm.
With a *neutral* pool the same permission becomes legitimate.

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
| shared neutral 12-fact pool, both arms | **no** — `frozen` mode plus our own `facts/controller_reportable_facts.json` |
| `deterministic` authoring that **selects facts toward its target** | **no**, and in fact the merge's `balanced` mode would *prevent* it. Needs per-target `controller_fact_scores` |
| `llm_authored` authoring | **no** |
| **`B`, episode-scope budget** | **yes** |
| `all_nondecisive` pool, `public_ledger` controller memory | yes (we want neither) |

**Phase 1 runs on `dev/rsanchez` today.** The merge is for phase 4.

### The one real cost of going unmerged

`controller_fact_scores` is a **single field per task**, so it cannot encode
alignment toward two opposite targets at once. Two options:

1. **Two task directories per setup** — four in total — with identical facts and
   identical pool, differing *only* in the score field (one scored toward A0,
   one toward A2). Ugly but fully auditable: a diff shows one field changed.
2. **Accept target-blind fact choice** and let the arms differ only in the
   recommendation. That is a narrower but perfectly clean experiment — *does a
   recommendation steer, with evidence held constant?* — and it is what Darius's
   balanced configs already do.

Option 1 is what our design intends. Option 2 is worth running *as well*,
because the contrast between them separates **evidence selection** from **bare
recommendation**, which is a question neither setup currently answers.

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

**On 100 rather than v0.4's 50:** the bootstrap resamples whole episodes, so
precision goes as 1/√episodes. We measured a ±18% interval on the path
divergence at 168 episodes; 50 extrapolates to about **±33%**, which will not
separate two control directions. v0.4 permits extending to 100 "if the
statistical analysis indicates" — it does. But this doubles the bill, so it is
César's call. The calculation is in
[`design_preliminary_2026-10-02.md`](design_preliminary_2026-10-02.md) §3.

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
| the two setups' fact lists and every property | [`designs/README.md`](designs/README.md) and the three `*.json` beside it |
| why each protocol option was chosen, with code references | [`protocols_2026-10-02.md`](protocols_2026-10-02.md) §2, §3, §5b |
| rounds, episodes, which controller first | [`design_preliminary_2026-10-02.md`](design_preliminary_2026-10-02.md) |
| what Darius already built, and what plan v0.4 says | [`prior_art_2026-10-02.md`](prior_art_2026-10-02.md) |

---

## 6. Two prerequisites

1. **Build the two task directories** — see §3c: phase 1 needs **no merge**.
   Use `controller_report_pool_mode: frozen` with our 12 facts written into
   `facts/controller_reportable_facts.json`. Merge `origin/darius-MA-v1` before
   phase 4 (episode-scope `B`), which needs `controller_budget_scope`.
2. **Decide the scoring question in §3c** — per-target
   `controller_fact_scores` in two task directories per setup, or target-blind
   fact choice. This determines whether the arms differ in *which facts* get
   posted or only in the *recommendation*.
