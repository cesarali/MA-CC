# The imitation game, end to end: mechanics in plain and technical terms

This document explains **what `hidden_bench_imitation` actually does**, step by
step, from the raw HiddenBench corpus to a single recorded event. Every section
gives the plain-language version first and the precise version second.

It is written for one purpose: to make the mechanics complete enough that the
mathematics of the **non-reasoning (`dynamics_mode: classical`) arm** can be
developed against them without re-reading the source.

**How this fits with the other documents.**

| Document | Answers |
| --- | --- |
| [`README.md`](README.md) | What the three HiddenBench games are and how they differ |
| **this document** | **How the imitation game is played, step by step** |
| [`hidden_bench_imitation.md`](hidden_bench_imitation.md) | Full reference: every config field, event-schema column, metric, estimator, and report file |
| [`classical_dynamics_and_the_imitation_model.md`](classical_dynamics_and_the_imitation_model.md) | What the classical kernel *means* physically, and its relation to the Irisarri *q*-voter paper |
| [`paraphrase_and_factorization_pipeline.md`](paraphrase_and_factorization_pipeline.md) | Full procedure behind the semantic scaling methods |

There is deliberate overlap with the last two. This document is the connective
tissue: it walks the whole pipeline once, so the physics document can assume the
mechanics and the reference document can assume both.

---

## Table of contents

1. [Where the imitation game sits](#1-where-the-imitation-game-sits)
2. [From HiddenBench to a task](#2-from-hiddenbench-to-a-task)
3. [Building a population: replication, paraphrasing, factorization](#3-building-a-population-replication-paraphrasing-factorization)
4. [The episode](#4-the-episode)
5. [The event](#5-the-event)
6. [The controller](#6-the-controller)
7. [The non-reasoning kernel](#7-the-non-reasoning-kernel)
8. [What is recorded](#8-what-is-recorded)
9. [Seams for extending the mathematics](#9-seams-for-extending-the-mathematics)

---

## 1. Where the imitation game sits

### Plain terms

Three games share the HiddenBench information structure. They differ only in
*how agents talk to each other*.

- **`hidden_bench_vanilla`** — everyone in one room, round-robin discussion,
  a vote before and a vote after. The original paper's protocol.
- **`hidden_bench_naming`** — one random pair meets privately per round, swaps
  messages, both commit. Private memory, no global transcript.
- **`hidden_bench_imitation`** — the interaction is stripped to the minimal
  opinion-dynamics event: **pick one agent at random; only that agent may change
  its mind.**

That last reduction is the whole design of this game. It makes the process a
jump chain on the vector of "how many agents hold each option", which is exactly
the object statistical physics knows how to analyse. And it isolates a single
replaceable box — *how does the focal agent decide?* — that can be filled either
by a language model or by a written-down transition kernel, with everything else
held identical.

### Technical

| | `vanilla` | `naming` | `imitation` |
| --- | --- | --- | --- |
| Agents changing state per step | all (final vote) | 2 | **1** |
| Interaction | plenary broadcast | dyad | focal + `q` peers |
| Controller wired | no | no (`control.mechanism: none` only) | **yes** |
| Provider-free mode | no | no | **yes** (`dynamics_mode: classical`) |
| Module | [`vanilla/`](../../../src/mas_cc/games/hidden_bench/vanilla/) | [`naming/`](../../../src/mas_cc/games/hidden_bench/naming/) | [`imitation/`](../../../src/mas_cc/games/hidden_bench/imitation/) |

The controller and the classical dynamics exist **only** in the imitation game.
[`README.md` §6.1](README.md) records why message-level control was not wired
into the other two: `Control.override` returns a single action value, which
fits forcing a vote but not injecting a message.

---

## 2. From HiddenBench to a task

### Plain terms

The Stasser–Titus *hidden profile* setup. There is a scenario — say, which of
three places to evacuate a village to. Some facts everyone receives. Other facts
are split up so each agent holds a piece nobody else has.

The trap is built deliberately:

- the shared facts alone point at a **wrong** answer (the *decoy*);
- shared facts plus any *single* hidden fact **still** point at the decoy;
- only pooling **all** the hidden facts disqualifies the decoy and reveals the
  correct answer.

And the load-bearing detail: **agents are never told the information is
asymmetric.** Each one believes everybody sees what it sees, so nobody has an
obvious reason to recite their own facts out loud.

The paper's finding is that groups reliably fail — not because they reason badly
once they have the facts, but because they never surface them.

### Technical

[`data.py`](../../../src/mas_cc/games/hidden_bench/data.py) reads
`data/hidden_bench/canonical/tasks.json` (65 upstream tasks). A task is

```
task = ( scenario, I_s, I_u = (u_0, ..., u_{C-1}), A, o* )
```

- `I_s` — shared information, given to every agent.
- `I_u` — hidden information, `C` items. **The list index is the evidence
  type.** This convention is load-bearing everywhere downstream and is asserted
  at load time
  ([`data.py:139`](../../../src/mas_cc/games/hidden_bench/data.py#L139)).
- `A` — `possible_answers`, `K = |A|`, in **canonical order**. Count vectors are
  encoded in this order, never dictionary insertion order.
- `o*` — the correct answer.

**Assignment.**
[`assign()`](../../../src/mas_cc/games/hidden_bench/data.py#L381) produces
`{I_i}` for `i = 1..N`, where `I_i = I_s ∪ I_u^(i)`, subject to the

> **union invariant:** the union of every agent's private information reconstructs
> `I_u`.

Checked on every `initialize` by
[`assert_union_invariant`](../../../src/mas_cc/games/hidden_bench/data.py#L489):
literal set equality (on whitespace/case-normalized text) under the identity
schemes, and *coverage of evidence types* under the semantic schemes, where the
reconstruction is semantic rather than literal.

**Two profiles, one code path.**

| `profile` | Meaning |
| --- | --- |
| `hidden` | The split above. |
| `full` | Every agent receives all of `I_u`. The paper's ceiling condition. |

`full` is a **profile flag, not a separate game**, precisely so that `Y_full` and
`Y_post` are produced by the same code and are therefore comparable without
caveat.

**Per-agent presentation.** Each agent's facts are shuffled with a per-agent
derived seed (`shuffle_facts`, default `true`), so ordinal position within the
information block carries no signal about which fact is shared and which is
private.

---

## 3. Building a population: replication, paraphrasing, factorization

### Plain terms

Canonical tasks are written for three or four participants — one hidden fact
each. The imitation game wants `N = 32`. Where do 32 private information sets
come from when there are only four hidden facts?

Three answers, plus two local controls:

- **Exact replication** — deal the four facts round-robin. Eight agents end up
  holding evidence type 0, and all eight hold *the identical sentence*.
- **Paraphrasing** — those eight agents each get a **different wording of the
  same complete fact**. Same meaning, different sentence.
- **Factorization** — one fact is **split into 2–4 pieces** that are jointly
  sufficient and individually insufficient, and the pieces are spread across
  agents.

Paraphrasing changes *how a fact is worded*. Factorization changes *how finely
the information is divided*. They are not variations of one idea.

### Why paraphrasing exists

Exact replication at `N > C` introduces an artifact: eight agents reciting the
same sentence verbatim. Real groups do not do this, and — more concretely for
measurement — a keyword-overlap disclosure detector cannot tell which of the
eight surfaced it.

A worked example from task 1:

> **Source:** The supply truck headed to the village from East Town was stuck in
> the tunnel.
>
> **Accepted variant:** The supply truck traveling from East Town to the village
> was stuck in the tunnel.

Same origin, destination, vehicle, obstruction. A paraphrase is a **standalone
restatement of one complete hidden fact** — not a shortened clue, and not a piece
of the fact.

Contrast the factorization of the same fact:

1. "A supply truck existed."
2. "It was destined for the village."
3. "It had departed from East Town."
4. "It became stuck in a tunnel."

These reproduce the original only once combined.

### Technical

The annotation pipeline is two LLM passes per evidence type
([`paraphrase_and_factorization_pipeline.md`](paraphrase_and_factorization_pipeline.md)):

1. **Generator** receives the full task context — scenario, `I_s`, *all* of
   `I_u`, the options, and `o*`. Full context is supplied so it can *avoid*
   importing another agent's fact or leaking the answer. It proposes ~12
   candidates per batch.
2. **Verifier** checks each candidate for entailment by the source, **reverse**
   entailment (nothing dropped), absence of added answer-relevant content,
   absence of answer leakage, and absence of overlap with another evidence type.
   A candidate is acceptable only when both entailment directions hold and all
   three contamination flags are false.
3. Accepted variants receive stable IDs (`0-000` = evidence type 0, variant 0).
   Target: 10 per type.
4. After a complete traversal the pool is stamped `status: "frozen"`. **The
   population builder refuses unfrozen pools**, so a half-finished annotation run
   can never silently become an experimental condition.
5. `prepare_hiddenbench.py` balances evidence types across `N` agents (counts
   differ by at most one), then deals one distinct variant per holder. Capacity
   without reuse is `variants_per_type × types` — roughly 40 agents for a
   four-type task.

The runtime loader re-validates all of this on every load
([`_scaled_task`](../../../src/mas_cc/games/hidden_bench/data.py#L162)):
unique `variant_id`s unless reuse was explicitly declared,
`transformation == "validated_paraphrase"`, complete evidence-type coverage,
type balance within one, and source-text identity.

### The five schemes

| Scheme | Each agent holds | Availability |
| --- | --- | --- |
| `bijective` | one complete hidden fact, `N == C` exactly | any task |
| `exact_replication` | one complete hidden fact, verbatim duplicated across holders | any `N ≥ C`, derived at run time |
| `paraphrased_replication` | one complete hidden fact, reworded per holder | prebuilt, or opt-in deterministic preparation from validated paraphrases |
| `factorized_evidence` | one or more partial components | prebuilt file only |
| `padded` | first `C` agents carry the bijection; extras get shared information only | any `N ≥ C`, mas_cc-local |
| `decoy` | extras get a shared fact restated as if private | any `N ≥ C`, mas_cc-local |

`exact_replication` is reimplemented in-repo
([`_balanced_type_assignment`](../../../src/mas_cc/games/hidden_bench/data.py#L88))
because its allocation rule is short, deterministic, and LLM-free; a test pins it
against the checked-in `N_32.json` agent-for-agent so the two cannot drift.
The semantic schemes never invent text at run time. `factorized_evidence`
remains prebuilt-only. A paraphrased run may explicitly enable
`population_preparation.auto_build_missing`; then a missing task is appended to
`N_<N>.json` using existing accepted paraphrases. The builder first validates
the task, source-text identity, unique capacity, and evidence coverage, and
fails before provider inference when the source paraphrases are unavailable or
insufficient.

`padded` and `decoy` are group-size controls, not scaling methods. Without
`padded`, `N` and `|I_u|` move together and no group-size result can be
attributed to either. `decoy` adds pooling noise with **no new proposition**, so
a drop in accuracy under it cannot be blamed on new misinformation.

### Two caveats that matter for interpretation

**The annotation pools were self-verified.** The existing files record
`microsoft/gpt-5-mini` as *both* generator and verifier. This is a two-pass
self-verification process, not independent cross-model verification. The
pipeline supports separate models; this particular run did not use them.

**The global pool is unfrozen.** The generation run stopped at task 43. The
`imitation_N` grid works around this with `freeze_paraphrase_subset.py`, which
cuts a *complete and validated subset* (tasks 1–2, `N ∈ {4, 8, 16, 32}`) out of
the unfinished global pool and freezes that subset, validating evidence-type
coverage, variant uniqueness, source-text identity, and capacity `ceil(N_max/E)`.

### Why this matters for the mathematics

`exact_replication` and `paraphrased_replication` produce **the same information
structure** — the same map from agent to evidence type, the same union
invariant, the same balance. They differ only in surface form.

Any difference in dynamics between the two arms is therefore a **semantic or
lexical** effect, not an informational one. That is a controlled contrast
available at zero additional modelling cost.

---

## 4. The episode

### Plain terms

An episode is one task, one seed, one population, run for a fixed number of
population sweeps. Each sweep contains `N` elementary focal-update attempts.
Everyone starts holding some opinion, then updates happen one at a time. The
episode ends when all configured sweeps finish (or, optionally, at consensus).

### Technical

**State.**

```
X(t) ∈ A^N              each agent's current option
n_j(t) = |{ i : X_i(t) = j }|          occupation vector, canonical option order
p_j(t) = n_j(t) / N                    population shares
sum_j n_j(t) = N
```

**Three independently controlled sizes.** This separation is the point of the
scaled protocol
([`hidden_bench_imitation.md` §12](hidden_bench_imitation.md)):

```
N   = game.population_size                  the voting population
q   = game.options.social_group_size        ordinary peers per event
q_c = control.options.sensor_sample_size    agents the controller observes
```

Validated as `1 ≤ q ≤ N-1` and `1 ≤ q_c ≤ N`. Note the asymmetry: social peers
are drawn from the `N-1` agents *other than the focal*, while the controller's
sensor may sample the focal and any social peer. Both overlaps are recorded per
event.

**Initialization** — three modes
([`_provider_free_initial_votes`](../../../src/mas_cc/games/hidden_bench/imitation/game.py#L137)):

| Mode | Behaviour | Provider calls |
| --- | --- | --- |
| explicit `initial_votes` | a literal starting vector | 0 |
| `initial_distribution` | seeded categorical draw per agent | 0 |
| `local_vote` (reasoning default) | each agent votes from its own evidence alone | `N` |

`local_vote` is the `Y_pre` measurement: the answer each agent gives on the least
evidence it will ever have. Classical mode must never require provider
decisions, and the runtime raises if it would
([`runtime.py:131`](../../../src/mas_cc/games/hidden_bench/imitation/runtime.py#L131)).

**Termination**
([`_termination_reason`](../../../src/mas_cc/games/hidden_bench/imitation/game.py#L837)):
`turn >= game.horizon * N`, or — only if `stop_on_consensus` (default **false**)
— the population is unanimous. `game.horizon` always means population sweeps;
there is no elementary-interaction mode. The default is `false` so matched
cells have equal horizons, which several downstream estimators assume.

**The clock.** The event index *is* the clock. There is no physical time.
Normalized sweep time is

```
tau = t / N
```

The configured `game.horizon` is `S`, and the runtime sets `T = S·N`. Thus the
same horizon automatically preserves sweep count when `N` changes. `tau` is
normalized sweep time, **not** Gillespie physical time — see §7 for the
consequence.

---

## 5. The event

### Plain terms

One event is:

1. Pick one agent to be the **focal**. Pick `q` others to be its **peers**.
2. The **controller** peeks at a few agents (possibly including the focal), and
   decides whether to intervene.
3. If it intervenes, its message **takes the place of one peer** — the focal
   still hears exactly `q` voices, but one of them is the controller wearing a
   peer's clothes.
4. The focal listens, then re-votes.
5. Only the focal's opinion changes. Everybody else is frozen.

The only thing that differs between the reasoning and the classical arm is
step 4.

### Technical

Driven by
[`runtime.py:157-345`](../../../src/mas_cc/games/hidden_bench/imitation/runtime.py#L157-L345).

**1 — Schedule.** Focal and peers are drawn in a *single*
`rng.sample(agents, q + 1)`
([`select_participants`](../../../src/mas_cc/games/hidden_bench/imitation/game.py#L198)).
Drawing all `q+1` together keeps `q = 1` byte-for-byte compatible with the
former `rng.sample(population, 2)` dyadic scheduler, so old runs replay.

**2 — Sense.** The controller samples `q_c` agents uniformly **without
replacement from all `N`** — independently of the social draw.

**3 — Decide.** The policy maps the observed target support to
`U_t ∈ {NO_OP, ADVOCATE_Z}`. See §6.

**4 — Actuate.** On `ADVOCATE_Z`, the advocacy **replaces exactly one of the `q`
social slots**: slot 0 when `q = 1` (consuming no extra random draw, preserving
legacy RNG streams), otherwise a slot drawn from its own episode-seeded stream so
sensing stays independent of social replacement. The focal therefore always
receives exactly `q` inputs: `q-1` ordinary peer messages plus one peer-style
controller message. The replaced peer ID and zero-based slot are logged.

**5 — Update.** Mode-dependent, and this is the *only* branch:

- **`reasoning`** — for each retained peer, in the scheduler's logged order, run
  a dyadic exchange (`messages_per_agent` rounds, both parties speak each round);
  then one final `focal_update` call in which the focal sees the scenario, its own
  facts, its bounded private history, its current vote, the dialogue, and — when
  `q > 1` — the per-slot influence inputs. It returns `{vote, rationale}` JSON.
- **`classical`** — no provider call at all. `sample_jump` picks the destination
  from written-down weights. See §7.

**6 — Apply.** With `A = X_f(t)` and `B = X_f(t+1)`:

```
n  ->  n - e_A + e_B
```

Only the focal changes. The controller never joins the occupation vector and
never receives task evidence.

**7 — Record.** One event row, ~90 fields. See §8.

### Population observables

Computed before and after every event by
[`population_observables`](../../../src/mas_cc/games/hidden_bench/imitation/metrics.py#L16):

```
m_Z       = (K * p_Z - 1) / (K - 1)                 target alignment
m_order   = (K * max_j p_j - 1) / (K - 1)           order parameter
H_vote    = -sum_j p_j ln p_j / ln K                normalized vote entropy
```

`m_Z` is a `K`-state magnetization: `1` at consensus on `Z`, `0` at the uniform
state, `-1/(K-1)` at zero support. It is reported twice per event — as `m_truth`
with `Z = o*`, and as `m_ctrl` with `Z` the controller/analysis target.

In an *uncontrolled* comparator cell there is no controller target, so the
analysis target falls back to `o*` while the controller-specific diagnostics stay
null. That keeps a matched grid readable with one column list
([`game.py:530-540`](../../../src/mas_cc/games/hidden_bench/imitation/game.py#L530-L540)).

### Information diffusion

Even though only votes evolve, the game tracks what the population *knows*. Each
agent carries an exposure set `K_i ⊆ {0..C-1}`, initialized to its natively held
evidence types and grown whenever a partner surfaces a fact:

```
K_i  <-  K_i ∪ { c : D_c(message heard) = 1 }
```

with the disclosure detector

```
D_c(msg) = 1[ |W(u_c) ∩ W(msg)| / |W(u_c)| >= 0.6 ]
```

where `W` is the set of normalized content words (lowercased, longer than three
characters, stopwords removed), evaluated against a **single** message rather
than the cumulative transcript
([`disclosed_facts`](../../../src/mas_cc/games/hidden_bench/data.py#L533)).

Derived: `disclosure_reach[c]` = how many agents have been exposed to fact `c`,
counting native holders, so a fact held by four agents starts at reach 4 before
anybody speaks.

> **`unshared_disclosure_rate` is a lower bound by construction.** A faithful
> paraphrase sharing few content words is missed. Every value reads "at least
> this fraction was surfaced", never "exactly". An LLM-judge variant is the
> obvious follow-up and is deliberately not v1 — it would make the benchmark's
> central diagnostic depend on a second, unaudited model.

Note the interaction with §3: under `paraphrased_replication` agents hold
*reworded* facts, but `D` matches against the **canonical source text**.
Detection sensitivity is therefore scheme-dependent, and any diffusion model
built on these counts needs that measurement term carried explicitly.

---

## 6. The controller

Source:
[`imitation/controller.py`](../../../src/mas_cc/games/hidden_bench/imitation/controller.py).

### Plain terms

A partial-observation feedback controller. Each event it peeks at a handful of
agents, sees how many already hold the option it wants, and if that is too few it
inserts a message arguing for that option — **written to be indistinguishable
from an ordinary participant**.

It never forces a vote. It never sees hidden facts. It is not a member of the
population.

### Technical — sensor, policy, actuator

**Sensor.** Uniform sample without replacement, size `q_c`, from `N`. Observed
support:

```
s_t = (1 / q_c) * |{ i in S_t : X_i = Z }|
```

so `q_c * s_t` is hypergeometric:

```
q_c * s_t ~ Hypergeometric(N, n_Z(t), q_c)

E[s_t]   = p_Z
Var(s_t) = p_Z (1 - p_Z) / q_c  *  (N - q_c) / (N - 1)
```

Exact at `q_c = N`. The finite-population correction is why each event logs
`sensor_target_error = s_t - p_Z` alongside its absolute value: the sensor's
own error is a measured quantity, not an assumption.

**Policy.** Two mechanisms, differing only in this step:

```
threshold_target:   U_t = ADVOCATE_Z   iff   s_t < theta

soft_target:        P(U_t = ADVOCATE_Z | s_t) = sigma( beta * (theta - s_t) )
```

with `sigma` the logistic function (implemented overflow-free, so extreme `beta`
saturates rather than raising). `advocacy_probability` is monotonically
decreasing in `s_t`, exactly `0.5` at the threshold, and approaches the hard step
as `beta -> inf`.

The soft policy exists for a specific and non-obvious reason
([`controller.py:447-456`](../../../src/mas_cc/games/hidden_bench/imitation/controller.py#L447-L456)):

> The deterministic policy makes `I(U_t ; n_Z(t+1) | n_Z(t))` **unidentifiable**,
> however many episodes are run. Every event in a given conditioning slice
> carries the same action, so the estimator has no within-slice contrast to
> measure. Giving both actions non-zero probability in comparable population
> states is the entire purpose of the `soft_target` mechanism. It does not change
> what the intervention *means*.

The deterministic policy still reports a probability (`1.0` or `0.0`), so
`controller_advocacy_probability` means the same thing under both mechanisms and
one analysis query audits both. Likewise `beta` is logged as `None` rather than
omitted under `threshold_target`.

**Target resolution.** `"correct"` resolves to `o*`; a string resolves to that
exact option label; a non-negative integer resolves to that zero-based position
in `possible_answers`.

**Actuator.** Mode-dependent, and this asymmetry is the scientific point of the
2×2 design:

| Mode | What `ADVOCATE_Z` becomes |
| --- | --- |
| `reasoning` | **a sentence of English** that the focal agent reads |
| `classical` | **a number added to a transition weight** |

The classical actuator never reads the message text at all. The configs
nevertheless carry a `template_version` in classical cells, with a comment saying
it changes no classical dynamics — so the contrast is not accidentally comparing
two different controllers.

### Message templates

| Version | Form | Status |
| --- | --- | --- |
| 1 | announces itself: "The external controller currently advocates…" | **known leak**, kept only for reproducibility |
| 2 | four peer-style paraphrases: cite a *shared* fact, soft coordination appeal, explicit first-person vote | |
| 3 | one fixed, factual-claim-free advocacy line | default for `soft_target` |

Version 1 is a leak in the precise sense that an agent told an *external
controller* is speaking can discount the turn as an outside voice, so `R_ctrl`
measured under it is **not social control**. It is retained because runs recorded
before 2026-08-11 used it and those numbers have to stay reproducible.

Version 3's single frame is a modelling decision, not a style preference
([`controller.py:209-216`](../../../src/mas_cc/games/hidden_bench/imitation/controller.py#L209-L216)):

> The soft policy already carries the run's stochasticity at `Y_t -> U_t`.
> Drawing a paraphrase *and* a fact on top of it would make `U_t -> M_t` a second
> random channel, and the actuation estimate could then no longer say which of
> the two the population responded to.

### Two invariants enforced in code

1. **The controller never receives hidden facts.** It may cite only `I_s`. A
   controller that manufactured evidence would turn "the population moved toward
   `Z`" into "an agent believed a new fact"
   ([`controller.py:85-90`](../../../src/mas_cc/games/hidden_bench/imitation/controller.py#L85-L90)).
2. **The controller never identifies itself.** `check_peer_style` rejects any
   rendered message containing `controller`, `external`, `experiment`, or
   `simulation`; `check_frames` additionally rejects `system`, `your position`,
   `reconsider`, and `commit` in the *frame* the controller contributes. The
   split matters — nine corpus facts legitimately say "air purification system"
   and two say "committed", so the stricter list can only be checked against the
   frame, never the whole message.

And structurally: `override()` returns `None` unconditionally. **Feedback shapes
the interaction; it never overwrites the vote.**

---

## 7. The non-reasoning kernel

Source:
[`classical.py`](../../../src/mas_cc/games/hidden_bench/imitation/classical.py)
— about 70 lines of substance. The physics reading is in
[`classical_dynamics_and_the_imitation_model.md`](classical_dynamics_and_the_imitation_model.md);
this section states the mechanics and the exact formulas.

### Plain terms

Do not ask anybody. Compute a number for each option the focal agent could switch
to, and draw from those numbers.

The rule in one sentence:

> **You are more likely to switch to an opinion other people already hold** —
> that is herding, and it is the whole of the dynamics. On top of it, the
> controller can add a push toward one specific opinion.

This is not a cheap stand-in for the language model. It is a **null model with
known physics**: the rule comes from a published stochastic-thermodynamics paper,
so entropy production, opinion currents, affinities, and fluctuation theorems are
*defined* for it. Running the identical game with LLMs then asks how far the
model's behaviour sits from a system whose thermodynamics is already known.

### Technical — the transition law

Reference: Irisarri, Trigal, Toral & Manzano, *Stochastic Thermodynamics of
Social Imitation beyond Energetics*,
[arXiv:2511.14006](https://arxiv.org/abs/2511.14006), multi-opinion section
(Eqs. 34a–34b). `kernel: irisarri_multi_opinion` refers to that section, not to
the binary model.

Given focal `f` with `X_f = A` and occupation vector `n`, for every destination
`B != A`:

```
interaction factor
    g(n_B) = (n_B + delta) / N          interaction_factor: destination_count_plus_offset
    g(n_B) = 1                          interaction_factor: constant

base weight
    w_base(A -> B ; n) = r_AB * n_A * g(n_B)

    r_AB = h  if idx(A) < idx(B)         forward_rate
    r_AB = a  otherwise                  reverse_rate

control weight
    w_ctrl(A -> B) = c * n_A * 1[ U_t = ADVOCATE_Z  and  B = Z ]

destination law
    P(B | A, n, U_t) = (w_base + w_ctrl) / sum over C != A of (w_base + w_ctrl)
```

"Forward" and "reverse" are fixed by the **canonical index order of the option
list**, so each unordered pair `{A, B}` gets a two-way road with its own rate
constants. Reaction IDs are logged as `pair-0-1:reverse`, `pair-1-2:forward`, and
prefixed `control+` whenever the control weight fired — giving the
reaction-resolved trajectory the paper's fluctuation theorems need.

**`n_A` cancels.** It multiplies every channel identically, so the destination
law reduces to

```
P(B)  ∝  r_AB * (n_B + delta)/N  +  c * 1[B = Z]
```

which makes the scale of `c` directly readable. The herding term lives in
`(0, 1]`, reaching `1` only when the entire rest of the population already holds
the destination. So `control_strength: 2.0` means **the controller's push is
twice as strong as unanimous peer pressure could ever be** — a strong controller
by construction.

### The symmetric point

With `forward_rate = reverse_rate = 1.0` the affinity is

```
mu_r = ln(h / a) = 0        for every pair
```

**No intrinsic bias toward any opinion.** This is chosen deliberately: if the
kernel also had a built-in preference, any drift toward the target would mix "the
controller worked" with "the physics was already tilted that way". At `mu_r = 0`
every asymmetry in the trajectory is attributable to the controller.

### Why `interaction_offset: 1.0` is not a tuning knob

This looks like a numerical safety hack — without it, `g` would be zero for an
opinion nobody holds, making that opinion unreachable. That is true but
incomplete. The real reason is **local detailed balance**.

The forward jump `A -> B` is evaluated at `n`; its reverse `B -> A` is evaluated
at `n' = n - e_A + e_B`, where `n'_A = n_A - 1` and `n'_B = n_B + 1`. Since the
code recomputes the interaction factor from whatever configuration it is in:

```
w(A->B ; n)      h * n_A * (n_B + delta)
------------  =  -------------------------------
w(B->A ; n')     a * (n_B + 1) * (n_A - 1 + delta)
```

Set `delta = 1`. The numerator becomes `h * n_A * (n_B + 1)`, the denominator
`a * (n_B + 1) * n_A`, and everything cancels:

```
w(A->B ; n) / w(B->A ; n')  =  h / a  =  exp(mu_r)      for every configuration
```

That is exactly the paper's microscopic local detailed balance relation (Eq. 37).
With any other offset the ratio retains a configuration-dependent factor and
`mu_r` is no longer a constant affinity — the thermodynamic interpretation is
lost. Verified over all 15 occupation vectors of `N = 4, K = 3` and all
source/destination pairs; at `offset = 0.5` the ratio is `4/7` at `n = (0,0,4)`
and `7/4` at `n = (0,1,3)`.

The offset also has a clean social reading: `delta/N` is a small **spontaneous
switching** probability that survives when nobody holds the destination opinion —
a minority of one can still appear. That is the role a noise or free-will term
plays in noisy voter models.

> **Rule of thumb.** `interaction_offset: 1.0` is not tunable if you want the
> paper's thermodynamics. It is the value that makes `forward_rate/reverse_rate`
> mean `exp(mu_r)`.

### The controller breaks detailed balance on purpose

The control term is added to **one direction only** and is not paired under time
reversal. The controlled kernel is therefore driven, and the entropy production
it generates is the "demonic" feedback effect of the paper's Maxwell-demon /
Szilard discussion. The uncontrolled kernel is the equilibrium reference; the
controller is the demon.

### Three limitations to know before building on this

**1 — It is the `q = 1` member of the family. There is no `q` knob.**

```
paper, q-voter :  g(n) = [ n / (N-1) ]^q   ->  x^q
this kernel    :  g(n) = (n_B + 1) / N     ->  x^1
```

So the shipped kernel is the **linear voter limit** with a spontaneous-switching
regularizer, generalized to `K` opinions. Everything above — Eq. 34, local
detailed balance, the affinity, reaction-resolved trajectories — holds exactly at
`q = 1`.

What it does **not** give you is the physics the paper's figures are about. The
paper is explicit that the stationary distribution is *always unimodal for
`q <= 1`*, and the critical point `lambda_c = (q+1)/(q-1)` diverges as `q -> 1`.

> With `interaction_factor: destination_count_plus_offset`, the classical arm sits
> in the **no-phase-transition regime**: no polarized-to-consensus transition, no
> bistability, no symmetry breaking. Exactly right for a null model, but the phase
> diagram is not reachable from this config.

Reaching it needs a third `interaction_factor` of the form
`(n_B + delta)^q / N^q`. That is **not implemented** — the validator accepts only
`constant` and `destination_count_plus_offset`. And note that **the `delta = 1`
cancellation argument has to be redone**, because `(n+1)^q` does not cancel
against `n^q`.

**2 — `q` is sampled and logged, but not consumed.**

`ClassicalSocialContext` — peer IDs, peer opinions, replaced slot — is
constructed, passed into `sample_jump`, and immediately discarded
(`_ = social_context`,
[`classical.py:88`](../../../src/mas_cc/games/hidden_bench/imitation/classical.py#L88)).
It is a typed boundary placed in advance so that extending the transition law
does not require redesigning the scheduler or the persisted event schema. This is
the clearest single seam for a `q`-voter kernel.

**3 — The realized chain is not detailed-balanced, even at `mu_r = 0`.**

The simulator picks the focal uniformly and then normalizes destination weights
*within that focal's options*, so the realized transition probability carries a
source-dependent normalizer the underlying rates do not:

```
P(n -> n')  =  (n_A / N) * w(A->B ; n) / Z_A(n)

Z_A(n) = sum over C != A of w(A -> C ; n)
```

`Z_A(n) != Z_B(n')`, so the forward/backward *probability* ratio differs from 1
despite the channel weights individually satisfying local detailed balance.
Checked exhaustively at `N = 4, K = 3`: 48 state/pair combinations break.

This is a property of the simulator, not an error in the rates, and it is
**recoverable offline** — every event logs `classical_candidate_channels` with
each channel's `base_weight`, so the true rates, the affinities, and the
trajectory entropy production can be reconstructed. Reading the paper's
identities *directly* off event counts would need a Gillespie clock: sample the
escape time from the total rate over all agents and record the increment. That is
a v2 change, not a config change. Today `physical_time_increment` is recorded as
`None` and `time_convention` as `focal_conditioned_embedded_jump_chain`.

### Worked example

Config as shipped: `h = a = 1.0`, `delta = 1.0`, `c = 2.0`. State
`n = (West 1, East 2, North 1)`, `N = 4`. Focal holds *East Town*. Target
resolves to *North Hill*.

**Controller silent (`NO_OP`):**

| channel | reaction id | base | control | weight | P |
| --- | --- | --- | --- | --- | --- |
| East -> West | `pair-0-1:reverse` | 1.000 | 0.000 | 1.000 | **0.500** |
| East -> North | `pair-1-2:forward` | 1.000 | 0.000 | 1.000 | **0.500** |

A coin flip, as it must be: `mu_r = 0` and both destinations have one holder.

**Controller advocating North Hill (`ADVOCATE_Z`):**

| channel | reaction id | base | control | weight | P |
| --- | --- | --- | --- | --- | --- |
| East -> West | `pair-0-1:reverse` | 1.000 | 0.000 | 1.000 | **0.167** |
| East -> North | `control+pair-1-2:forward` | 1.000 | 4.000 | 5.000 | **0.833** |

(`base = h * n_A * (n_dest + 1)/N = 1 * 2 * 2/4 = 1.0`;
`control = c * n_A = 2.0 * 2 = 4.0`.) One intervention moves the focal's
probability of adopting the target from 50 % to 83 %.

### What the classical arm buys the reasoning arm

Classical and reasoning are **the same experiment with one component swapped**.
Everything outside the `if dynamics_mode == "reasoning"` branch is shared:

| Stage | Classical | Reasoning |
| --- | --- | --- |
| Pick focal + `q` peers | identical | identical |
| Controller senses and decides | identical | identical |
| Advocacy replaces one social slot | identical | identical |
| **Choose the new opinion** | `sample_jump` weights | LLM reads prompt, returns a vote |
| Apply `-e_A + e_B`, record event | identical | identical |
| Metrics, MI/CMI analysis | identical | identical |

The LLM has a transition kernel too — it simply is not written down anywhere.
When an LLM focal agent reads a peer message and commits to a vote, it is
sampling from some conditional distribution over destinations. **That
distribution is the object of study**, and the classical arm is the ruler:

- **Does the LLM herd?** The classical dependence on `n_dest` is explicit and
  linear. Estimate the LLM's. A steeper-than-linear dependence would be
  *q*-voter-like group pressure *emerging from semantics* rather than being put
  in by hand.
- **What is the LLM's affinity?** The paper gives an inference route: the strong
  fluctuation theorem for the opinion current makes `mu_1 - mu_2` recoverable by
  linear regression on observed jump-count statistics. For the classical arm the
  answer is `0` by construction; for the LLM arm there is no config field to
  read, only a measurement to make.
- **Is semantic control stronger or weaker than mechanical control?** The
  classical controller's grip is exactly `c = 2.0`. The reasoning controller's
  grip is a paragraph of persuasive English. The matched 2×2 grid measures the
  difference.
- **Is reasoning even reversible?** Microscopic reversibility is the paper's
  foundational assumption and the classical kernel satisfies it by construction.
  If an LLM argued into one option can essentially never be argued back out, the
  dynamics has irreversible transitions sitting *outside* the framework — which
  the paper itself flags as needing new techniques. Finding that would be a
  result.

---

## 8. What is recorded

### Plain terms

Every event writes one row containing everything needed to replay it and
everything needed to analyse it — who was picked, what the controller saw and
decided, what was said, what changed, and what the population looked like before
and after.

### Technical

One `event` mapping per elementary step
([`apply_event_transition`](../../../src/mas_cc/games/hidden_bench/imitation/game.py#L424)),
grouped by purpose. The complete column list is in
[`hidden_bench_imitation.md` §11](hidden_bench_imitation.md); the groups are:

| Group | Representative fields |
| --- | --- |
| Identity / clock | `episode_id`, `interaction_index`, `tau`, `seed`, `task_id`, `N`, `K`, `social_group_size`, `dynamics_mode` |
| Schedule | `focal_agent_id`, `social_peer_ids`, `social_peer_votes_before`, `sampled_peer_agent_id`, `replaced_peer_id`, `replaced_peer_slot`, `influence_slots` |
| Sensor | `sensor_agent_ids`, `sensor_observed_opinions`, `sensor_count_vector`, `controller_target_support`, `controller_sensor_social_overlap_ids`, `controller_sensor_includes_focal` |
| Policy | `controller_policy`, `controller_threshold`, `controller_beta`, `controller_advocacy_probability`, `controller_action`, `controller_applied` |
| Actuation | `controller_template_version`, `controller_template_id`, `controller_fact_index`, `controller_message` |
| Population before / after | `population_state_*`, `occupation_counts_*`, `population_shares_*`, `p_truth*`, `p_ctrl*`, `m_truth*`, `m_ctrl*`, `m_order*`, `H_vote*` |
| Response | `delta_m_ctrl`, `delta_m_truth`, `delta_m_order`, `delta_H_vote`, `focal_changed`, `focal_adopted_target`, `focal_left_target`, `truth_current_increment` |
| Disclosure | `disclosed_hidden_facts`, `disclosure_reach`, `disclosure_events`, `unshared_disclosure_rate`, `focal_message`, `peer_message` |
| Classical | `classical_reaction_id`, `classical_source_opinion`, `classical_destination_opinion`, `classical_transition_rate_or_weight`, `classical_total_rate_or_normalizer`, `classical_candidate_channels`, `physical_time_increment`, `time_convention` |

Three recording choices worth noting:

- **`sampled_peer_agent_id` is kept alongside `peer_agent_id`.** The latter is
  `None` on a control event, so replay invariants are only checkable against the
  pre-substitution draw.
- **`classical_*` fields are written as explicit `None` in reasoning mode**
  rather than omitted, so one analysis pass reads both arms without a
  per-mode column list. Same rationale as `controller_beta` under the
  deterministic policy.
- **`focal_message` is recorded so fact citation can be detected in a separate
  pass afterwards.** Asking the agent to list the facts it cited in its own reply
  would itself increase disclosure and contaminate the measurement.

Derived per-episode: `truth_current = sum_t j_t` with

```
j_t = 1[X_f(t) != o*  and  X_f(t+1) == o*]  -  1[X_f(t) == o*  and  X_f(t+1) != o*]
```

Because only the focal changes, the net current **telescopes** to the final minus
initial truth headcount; it is not by itself a volatility measure. The cell
report pairs it with a fluctuation ratio
`F_truth = |mean(J_truth)| / Var_hat(J_truth)` over equal-horizon episodes, with
whole episodes as the bootstrap unit. This ratio is **not** claimed to satisfy a
thermodynamic uncertainty relation.

---

## 9. Seams for extending the mathematics

| Seam | Location | Status |
| --- | --- | --- |
| `q`-voter interaction factor `(n_B + delta)^q / N^q` | validator [`state.py:135-140`](../../../src/mas_cc/games/hidden_bench/imitation/state.py#L135-L140), kernel [`classical.py:49-55`](../../../src/mas_cc/games/hidden_bench/imitation/classical.py#L49-L55) | not implemented; validator accepts two values. **The `delta = 1` local-detailed-balance argument must be redone.** |
| Consuming the `q`-peer context in the kernel | `ClassicalSocialContext`, [`classical.py:88`](../../../src/mas_cc/games/hidden_bench/imitation/classical.py#L88) | typed boundary exists, value discarded |
| Gillespie clock / physical time | [`runtime.py:303-305`](../../../src/mas_cc/games/hidden_bench/imitation/runtime.py#L303-L305) | `physical_time_increment: None`; rates recoverable from `classical_candidate_channels`, per-unit-time quantities are not directly readable from event counts |
| Focal-conditioned normalizer `Z_A(n)` | §7, limitation 3 | breaks realized detailed balance even at `mu_r = 0`; recoverable offline |
| Network topology | `pairing` option, `select_participants` | `uniform_two_distinct` only; complete graph. Single documented drop-in point. |
| Disclosure detector as a measurement model | [`disclosed_facts`](../../../src/mas_cc/games/hidden_bench/data.py#L533) | keyword overlap at 0.6, lower bound, sensitivity depends on the assignment scheme |
| Asymmetric per-pair rates | `forward_rate` / `reverse_rate` | implemented but **one pair of constants for all pairs**; the paper allows `h_r`, `a_r` per unordered pair |

### Two open items to settle before quoting numbers

1. **The paraphrase annotation pools were self-verified** by a single model
   acting as both generator and verifier, and the global pool is unfrozen. Only
   the explicitly frozen subsets are usable, and even those are a two-pass
   self-verification rather than independent cross-model verification.
2. **`control.options.template_version: 1` is a known leak** — an agent told an
   external controller is speaking can discount the turn. `R_ctrl` measured under
   v1 is not social control. Version 1 is retained only so pre-2026-08-11 runs
   stay reproducible.

---

## See also

- [`hidden_bench_imitation.md`](hidden_bench_imitation.md) — full reference:
  every config field, event column, metric, MI/CMI estimator, and report file.
- [`classical_dynamics_and_the_imitation_model.md`](classical_dynamics_and_the_imitation_model.md)
  — the physics reading of the classical kernel and its relation to the Irisarri
  paper.
- [`paraphrase_and_factorization_pipeline.md`](paraphrase_and_factorization_pipeline.md)
  — the complete generation, verification, and allocation procedure.
- [`data_provenance.md`](data_provenance.md) — where the corpus came from and
  what remains unfinished.
- [`README.md`](README.md) — the three HiddenBench games and the hidden-profile
  paradigm they share.
