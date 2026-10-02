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

1. **Merge `origin/darius-MA-v1`.** Four controller options do not exist on
   `dev/rsanchez`: `controller_fact_pool_mode` (the target-independent menu),
   `controller_budget_scope`, `controller_round_budget_mode`,
   `controller_memory_mode`. There is no workaround.
2. **Build the two task directories.** Each needs exactly two new files —
   `private/N24_assignment.json` and `controller/balanced_fact_pool.json` — both
   already generated in `designs/*.json`. Everything else is copied from
   `task_003` unchanged.
