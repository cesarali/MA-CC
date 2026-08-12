# The classical dynamics mode, and what it has to do with the *q*-voter model

This document explains, in plain terms and then precisely, what the six lines
under `game.options.classical` in
[`hidden_bench_imitation_classical_control_10.yaml`](../../../configs/runs/hidden_bench/hidden_bench_imitation_classical_control_10.yaml)
actually do:

```yaml
classical:
  kernel: irisarri_multi_opinion
  forward_rate: 1.0
  reverse_rate: 1.0
  interaction_factor: destination_count_plus_offset
  interaction_offset: 1.0
  control_strength: 2.0
```

It then explains how that block relates to the *same* game run with the LLMs
switched on (`dynamics_mode: reasoning`).

The reference is Irisarri, Trigal, Toral & Manzano, *Stochastic Thermodynamics
of Social Imitation beyond Energetics*, [arXiv:2511.14006v2](https://arxiv.org/abs/2511.14006)
([local copy](../../../pdfs/Physics/StochasticThermodynamics/Stochastic%20Thermodynamics%20of%20Social%20Imitation%20beyond%20Energetics.pdf)).
Equation numbers below are the paper's. The implementation is
[`classical.py`](../../../src/mas_cc/games/hidden_bench/imitation/classical.py)
(70 lines of substance) and the config validation is in
[`state.py:120-139`](../../../src/mas_cc/games/hidden_bench/imitation/state.py#L120-L139).

This is the *physics* companion to
[`hidden_bench_imitation.md`](hidden_bench_imitation.md), which documents the
mechanics, the event schema, and the metrics. Read that one for "what is
recorded"; read this one for "what it means".

---

## 1. The whole idea in plain words

Four agents each hold one of three opinions — *West City*, *East Town*, *North
Hill*. Repeatedly, one agent is picked at random and changes its mind. The
only question the model has to answer is: **change its mind to what?**

There are two ways this repository answers that question, and they are
deliberately interchangeable:

- **`dynamics_mode: reasoning`** — ask an LLM. It reads the scenario, its own
  private facts, and whatever a peer just said to it, and it commits to a vote.
  The rule it follows is real but implicit: nobody can write it down.
- **`dynamics_mode: classical`** — do not ask anybody. Compute a number for
  each possible new opinion, and draw from it. No provider call is ever made.

The classical mode is not "a cheap stand-in for the LLM". It is a **null model
with known physics**. Its rule is short enough to write on one line, and — this
is the point — it is the rule from a published stochastic-thermodynamics paper,
so the quantities that paper derives (entropy production, opinion currents,
affinities, fluctuation theorems) are *defined* for it. When you then run the
identical game with LLMs, you can ask how far the LLM's behaviour sits from a
system whose thermodynamics you already know.

The plain-words version of the classical rule is:

> **You are more likely to switch to an opinion that other people already
> hold.** That is herding, and it is the whole of the dynamics. On top of it,
> an external controller can push you toward one specific opinion.

---

## 2. The paper's model

### 2.1 The binary version (paper, Sec. II)

The paper starts with `N` agents holding one of two opinions, `A` or `B`, all
connected to all. Opinions change through two *reversible* reactions
(Eqs. 1a–1b):

```
qA + B  <-->  (q+1)A          reaction 1, rates h1 (forward) / a1 (reverse)
A + qB  <-->  (q+1)B          reaction 2, rates h2 (forward) / a2 (reverse)
```

Reading the first one left to right: an agent holding `B`, confronted with `q`
agents holding `A`, converts to `A`. That is **herding** (conformity). Read
right to left it is **anti-conformity**: an agent surrounded by people who
already agree with it switches away out of contrarianism.

Insisting that both directions have non-zero rates is the load-bearing
assumption of the whole paper. It is called **microscopic reversibility**, and
it is what lets a "second law" be derived without ever invoking energy or
temperature. Socially: *if a person can be talked into an opinion, they can be
talked back out of it.*

The transition rates (Eq. 3) all share the shape

```
W = (rate constant) x (how many agents are available to jump) x g(n)
```

where `g(n)` is the **interaction factor** — the nonlinear group-interaction
term. It is the part that says "how much does the crowd matter". For sampling
`q` neighbours with repetition the paper takes

```
g(n) = [ n / (N-1) ]^q                                       (Sec. III)
```

with `n` the number holding the opposite opinion. This is precisely the
**nonlinear (or *q*-) voter model**: `q = 1` is the classical linear voter
model, `q = 2` and above give genuinely nonlinear "you need a *group* to
convince me" dynamics.

Two derived quantities matter later:

- **The affinity / generalized chemical potential** `mu_r = ln(h_r / a_r)`
  (Eq. 5). It measures how much the herding mechanism outweighs the
  anti-conformity one for reaction `r`. `mu_r = 0` means the two are perfectly
  balanced and there is no intrinsic push toward any opinion.
- **The local detailed balance relation** (Eq. 7), which is what makes all the
  thermodynamic machinery apply at all:

  ```
  W(n -> m) / W(m -> n)  =  exp( S_int(m) - S_int(n) + (m-n) mu_r )
  ```

The paper shows this model has a real phase diagram (Fig. 3): a second-order
transition from a **polarized** state (opinions split near `N/2`) to a
**consensus** state as the herding rate rises past `lambda_c = (q+1)/(q-1)`,
plus a first-order line and metastability. Crucially, **this only exists for
`q > 1`** — for `q <= 1` the stationary distribution is always unimodal and no
transition occurs. Keep that sentence; §5 comes back to it.

### 2.2 The multi-opinion generalization (paper, Sec. V.A)

`kernel: irisarri_multi_opinion` refers to this section, not to the binary
model. With `M > 2` opinions the state is no longer one number but an
**occupation vector** `n = (n_A, n_B, ...)` with `sum n_i = N`. One agent
changing from `A` to `B` moves the state by `-e_A + e_B` — subtract one from
the `A` bucket, add one to the `B` bucket.

Each unordered pair of opinions `{A, B}` gets its own reaction with its own
`h_r`, `a_r`, and its own interaction factor `g_r(n)`. The rates are
**Eqs. 34a–34b**, and these are the two lines the kernel implements:

```
W(n -> n - e_A + e_B)  =  h_r * n_A     * g_r(n)          (34a)
W(n - e_A + e_B -> n)  =  a_r * (n_B+1) * g_r(n)          (34b)
```

Note what the paper is careful about: **the same `g_r(n)` multiplies both
directions.** That is what makes the interaction factor cancel in the ratio,
leaving a clean generalized local detailed balance (Eq. 35). The counting
factors differ — `n_A` agents are available to leave `A`, and `n_B + 1` agents
are available to come back, because after the jump the `B` bucket has grown by
one.

In plain words: *multi-opinion just means the bookkeeping is a vector instead of
a single number, and every pair of opinions gets its own two-way road.*

---

## 3. Field by field

### `kernel: irisarri_multi_opinion`

The only implemented kernel; anything else raises at config load
([`state.py:121-123`](../../../src/mas_cc/games/hidden_bench/imitation/state.py#L121-L123)).
It selects the Eq. 34 geometry above: pick a focal agent, keep its current
opinion as the source `A`, and choose one destination `B != A`.

### `forward_rate: 1.0` and `reverse_rate: 1.0`

These are `h_r` and `a_r`. "Forward" and "reverse" are fixed by the **canonical
index order of the option list**, which for this task is

```
0 = West City,   1 = East Town,   2 = North Hill
```

A jump from the lower-indexed to the higher-indexed option of a pair is
*forward* and uses `forward_rate`; the other direction is *reverse* and uses
`reverse_rate` ([`classical.py:52-60`](../../../src/mas_cc/games/hidden_bench/imitation/classical.py#L52-L60)).
Reaction ids are logged as `pair-0-1:reverse`, `pair-1-2:forward`, and so on,
so every recorded jump names the reaction that produced it — exactly the
"reaction-resolved trajectory" the paper needs for its fluctuation theorems
(Eq. 38).

Setting both to `1.0` is a deliberate choice, not a default nobody thought
about. It means

```
mu_r = ln(h_r / a_r) = ln(1) = 0     for every pair
```

**There is no intrinsic bias toward any opinion.** In the paper's parameters
(Eq. 10) this is `lambda = chi = theta = 1`: the fully symmetric, unbiased
point. Herding and anti-conformity exactly balance.

Why do that on purpose? Because this run has a controller. If the kernel also
had a built-in preference, any drift toward *North Hill* would be a mixture of
"the controller worked" and "the physics was already tilted that way". With
`mu_r = 0`, **every asymmetry in the trajectory is attributable to the
controller.** The classical arm is a clean baseline, and the controller is the
only thing breaking symmetry.

### `interaction_factor: destination_count_plus_offset`

This is `g_r(n)`. Two values are accepted:

| value | `g_r(n)` | meaning |
| --- | --- | --- |
| `constant` | `1` | Nobody influences anybody. Pure random switching. |
| `destination_count_plus_offset` | `(n_dest + offset) / N` | Herding: the more people already hold an opinion, the more attractive it is. |

The second is the interesting one and the one configured here. In plain words:
**your chance of adopting an opinion is proportional to the fraction of the
population that already holds it.**

### `interaction_offset: 1.0`

This looks like a numerical safety hack. It is not — it is the exact value that
makes the model thermodynamically consistent, and it deserves the most
attention of anything in the block.

The naive reading is: without the offset, `g` would be zero for an opinion
nobody currently holds, so that opinion could never be re-entered and would be
absorbing. Adding `1` keeps every opinion reachable. True, but incomplete.

The real reason is **local detailed balance**. Work out the ratio of a jump and
its reverse. The forward jump `A -> B` is evaluated at configuration `n`; the
reverse jump `B -> A` is evaluated at `n' = n - e_A + e_B`, where `n'_A = n_A -
1` and `n'_B = n_B + 1`. Since the code recomputes the interaction factor from
whatever configuration it is currently in:

```
w(A->B; n)   =  h * n_A     * (n_B + off) / N
w(B->A; n')  =  a * (n_B+1) * (n_A - 1 + off) / N
```

so

```
w(A->B; n)      h * n_A * (n_B + off)
------------- = -------------------------------
w(B->A; n')     a * (n_B+1) * (n_A - 1 + off)
```

Set `off = 1`. The numerator becomes `h * n_A * (n_B + 1)`, the denominator
`a * (n_B + 1) * n_A`, and everything cancels:

```
w(A->B; n) / w(B->A; n')  =  h / a  =  exp(mu_r)         for every configuration
```

That is **exactly Eq. 37**, the paper's microscopic local detailed balance
relation. With any other offset the ratio keeps a configuration-dependent
factor and `mu_r` is no longer a constant affinity — the thermodynamic
interpretation is lost.

This was verified over all 15 occupation vectors of `N = 4`, `M = 3` and all
source/destination pairs: with `offset = 1.0` the ratio equals `h/a` in every
single case; with `offset = 0.5` it does not (e.g. `4/7` at `n = (0,0,4)`,
`7/4` at `n = (0,1,3)`).

> **Rule of thumb.** `interaction_offset: 1.0` is not tunable if you want the
> paper's thermodynamics. It is the value that makes `forward_rate/reverse_rate`
> mean `exp(mu_r)`. Change it only if you knowingly want a model outside the
> framework.

The offset also has a clean social reading: `off/N` is a small **spontaneous
switching** probability that survives even when nobody holds the destination
opinion — a minority of one can still appear. That is the same role a
"noise"/"free-will" term plays in noisy voter models.

### `control_strength: 2.0`

This is the only ingredient that is *not* in the paper. It is the external
controller's grip.

When the controller decides to advocate a target opinion `Z`, the kernel adds a
weight to the `A -> Z` channel only
([`classical.py:88-94`](../../../src/mas_cc/games/hidden_bench/imitation/classical.py#L88-L94)):

```
w_ctrl(A -> Z) = control_strength * n_A
```

Both `w_base` and `w_ctrl` carry the same factor `n_A`, so `n_A` **cancels** in
the destination choice. The rule the agent actually follows reduces to:

```
P(dest = B)  proportional to  (n_B + 1)/N  +  control_strength * [B is the target]
```

This makes the scale of `control_strength` easy to read. The herding term lives
in `(0, 1]` — it is at most `1.0`, reached only when the whole rest of the
population already holds the destination opinion. So `control_strength = 2.0`
means:

> **The controller's push is twice as strong as unanimous peer pressure could
> ever be.** This is a strong controller by construction.

Thermodynamically, the control term is added to one direction only and is not
paired under time reversal. It therefore **breaks local detailed balance on
purpose** — the controlled kernel is driven, and the entropy production it
generates is exactly the "demonic" feedback effect the paper discusses in
Sec. IV.E (Eq. 31, the Maxwell-demon / Szilard analogy). The uncontrolled
kernel is the equilibrium reference; the controller is the demon.

---

## 4. A worked example with this exact config

State: `initial_votes: [East Town, East Town, North Hill, West City]`, so
`n = (West 1, East 2, North 1)`, `N = 4`. Focal agent 0 holds *East Town*. The
controller's target is `correct`, which for task `evacuation_north_hill`
resolves to **North Hill**. Values below are produced by the real `sample_jump`:

**Controller silent (`NO_OP`):**

| channel | reaction id | base | control | weight | P |
| --- | --- | --- | --- | --- | --- |
| East Town -> West City | `pair-0-1:reverse` | 1.000 | 0.000 | 1.000 | **0.500** |
| East Town -> North Hill | `pair-1-2:forward` | 1.000 | 0.000 | 1.000 | **0.500** |

A coin flip — as it must be, since `mu_r = 0` and both destinations happen to
have one holder each.

**Controller advocating North Hill (`ADVOCATE_Z`):**

| channel | reaction id | base | control | weight | P |
| --- | --- | --- | --- | --- | --- |
| East Town -> West City | `pair-0-1:reverse` | 1.000 | 0.000 | 1.000 | **0.167** |
| East Town -> North Hill | `control+pair-1-2:forward` | 1.000 | 4.000 | 5.000 | **0.833** |

(`base = 1.0` because `h * n_A * (n_dest+1)/N = 1 * 2 * 2/4`; `control = 2.0 * 2
= 4.0`.) One controller intervention moves the focal's probability of adopting
the target from 50 % to 83 %. The reaction id is prefixed `control+` whenever
the control weight fired, so controlled and spontaneous jumps stay separable in
the logs.

---

## 5. So is this the *q*-voter model?

**It is the `q = 1` member of the family**, plus a spontaneous-switching
regularizer, generalized to `M = 3` opinions.

Compare the two interaction factors in the large-`N` limit, writing
`x = n_dest/N` for the fraction already holding the destination opinion:

```
paper, q-voter :  g(n) = [n/(N-1)]^q      ->  x^q
this kernel    :  g(n) = (n_dest + 1)/N   ->  x^1
```

So the configured kernel is the **linear voter limit**, `q = 1`, with the
`+1/N` acting as the noise/spontaneous term. Everything in §§2–4 — Eq. 34, local
detailed balance, the affinity `mu_r`, the reaction-resolved trajectories —
holds exactly at `q = 1`. This is a faithful, if minimal, member of the paper's
family.

What you do **not** get at `q = 1` is the physics the paper's figures are about.
The paper is explicit (Sec. III): the stationary distribution is *always
unimodal for `q <= 1`*, and the critical point `lambda_c = (q+1)/(q-1)` diverges
as `q -> 1`. So:

> With `interaction_factor: destination_count_plus_offset`, the classical arm
> sits in the **no-phase-transition regime**. There is no polarized-to-consensus
> transition, no bistability, no symmetry breaking. That is exactly right for a
> null model — but the phase diagram of Fig. 3 is not reachable from this
> config.

Reaching it would need a `q` exponent, i.e. a third `interaction_factor` value
of the form `(n_dest + offset)^q / N^q`. That is **not implemented** — the
validator accepts only `constant` and `destination_count_plus_offset`
([`state.py:124-129`](../../../src/mas_cc/games/hidden_bench/imitation/state.py#L124-L129)).
It is the natural next extension if the phase behaviour is wanted, and note
that the offset argument of §3 would need redoing, since `(n+1)^q` does not
cancel against `n^q`.

### A second caveat: the clock

The paper's objects — entropy production rate, probability currents `J`,
dynamical activity `K` — are all *per unit time* and defined for a
continuous-time Markov process. This kernel is a **discrete embedded jump
chain**, and says so: `physical_time_increment` is recorded as `None` and
`time_convention` as `focal_conditioned_embedded_jump_chain`
([`runtime.py:255-256`](../../../src/mas_cc/games/hidden_bench/imitation/runtime.py#L255-L256)).
The event index is the clock.

The concrete consequence is worth stating plainly. The simulator picks the focal
agent uniformly and then normalizes the destination weights *within that focal's
options*, so the realized transition probability carries a source-dependent
normalizer `Z_A(n)` that the underlying rates do not:

```
P(n -> n')  =  (n_A/N) * w(A->B; n) / Z_A(n),      Z_A(n) = sum over C != A of w(A->C; n)
```

`Z_A(n)` differs from `Z_B(n')`, so **the realized chain is not detailed-balanced
even at `h = a`**, although the channel weights individually are. Checked
exhaustively on `N = 4, M = 3`: 48 state/pair combinations have a
forward/backward probability ratio different from 1 despite `mu_r = 0`.

This is a property of the simulator, not an error in the rates, and it is
recoverable: every event logs `classical_candidate_channels` with each channel's
`base_weight`, so the true rates — and hence the affinities and the
Eq. 38 trajectory entropy production — can be reconstructed offline. What would
be needed for the paper's identities to be read off *directly* from event counts
is a Gillespie clock: sample the escape time from the total rate over all agents
and record the increment. That is a v2 change, not a config change.

---

## 6. How this connects to the reasoning game

The design point of the whole imitation game is that **classical and reasoning
mode are the same experiment with one component swapped**. Look at the run loop
in [`runtime.py:153-232`](../../../src/mas_cc/games/hidden_bench/imitation/runtime.py#L153-L232):
everything outside the `if rules.dynamics_mode == "reasoning"` branch is shared.

| Stage | Classical | Reasoning |
| --- | --- | --- |
| Pick focal + peer | identical (`select_participants`) | identical |
| Controller senses & decides | identical (`ThresholdTargetControl`) | identical |
| Advocacy **replaces** the peer slot | identical | identical |
| **Choose the new opinion** | `sample_jump` weights | LLM reads prompt, returns a vote |
| Apply `-e_A + e_B`, record event | identical | identical |
| Metrics, MI/CMI analysis | identical | identical |

The controller is genuinely the same object in both arms. Only its *actuation*
differs: in classical mode its advocacy becomes a number added to a transition
weight; in reasoning mode its advocacy becomes **a sentence of text** that the
focal agent reads
([`controller.py:366-376`](../../../src/mas_cc/games/hidden_bench/imitation/controller.py#L366-L376)).
The classical actuator never reads the message text at all — which is why the
config carries `control.options.template_version: 2` with a comment saying it
changes no classical dynamics. It is there so the B-vs-D contrast is not
comparing two different controllers.

### What this buys you

Everything in §§2–5 is a statement about the classical arm because its kernel is
written down. **The LLM has a transition kernel too — it just is not written
down anywhere.** When an LLM focal agent reads a peer message and commits to a
vote, it is sampling from some conditional distribution over destinations. That
distribution is the object of study.

So the classical arm is the ruler, and the questions become empirical:

- **Does the LLM herd?** The classical kernel's dependence on `n_dest` is
  explicit and linear. Estimate the LLM's dependence on `n_dest` from its jumps.
  A steeper-than-linear dependence would be *q*-voter-like group pressure
  emerging from semantics rather than being put in by hand.
- **What is the LLM's affinity?** The paper (Sec. IV.D, Eq. 30) gives an
  inference route: the strong fluctuation theorem for the opinion current means
  `mu_1 - mu_2` is recoverable by linear regression on observed jump-count
  statistics. For the classical arm the answer is known — `0`, by construction.
  For the LLM arm there is no config field to read; you must measure it. That
  is a genuine measurement of *how much a language model herds versus
  differentiates*, on a scale a physics paper defines.
- **Is semantic control more or less effective than mechanical control?** The
  classical arm has a controller with a known, exact grip: `control_strength =
  2.0`, i.e. twice maximal peer pressure. The reasoning arm has a controller
  whose grip is a paragraph of persuasive English. The B-vs-D contrast in the
  2x2 design measures the difference, and the `sensing_mi` /
  `population_actuation_cmi` / `target_actuation_cmi` / `focal_actuation_cmi`
  estimators in the `analysis` block are the instruments.
- **Is reasoning even reversible?** Microscopic reversibility is the paper's
  foundational assumption, and the classical kernel satisfies it by
  construction. Whether an LLM does is an open empirical question: if an LLM
  that has been argued into *North Hill* can essentially never be argued back
  out, the dynamics has irreversible transitions that sit *outside* the
  framework — which the paper itself flags as needing "more advanced and
  specific techniques" (Sec. VI). Finding that would be a result.

The short version:

> The classical mode gives the reasoning experiment a **physical null
> hypothesis**. Any information-theoretic quantity you compute on the LLM runs
> can be compared against the same quantity computed on a system whose
> thermodynamics is exactly known, matched agent-for-agent, controller-for-
> controller, and event-for-event.

---

## 7. Quick reference

| Field | Paper object | Configured | What it means |
| --- | --- | --- | --- |
| `kernel` | Eqs. 34a-34b | `irisarri_multi_opinion` | Multi-opinion occupation-vector rates; only kernel implemented. |
| `forward_rate` | `h_r` | `1.0` | Herding rate, low-index -> high-index option. |
| `reverse_rate` | `a_r` | `1.0` | Anti-conformity rate, the other way. |
| — | `mu_r = ln(h/a)` | `0.0` | No intrinsic bias. All asymmetry comes from control. |
| `interaction_factor` | `g_r(n)` | `destination_count_plus_offset` | Herding: attractiveness proportional to current support. `q = 1`. |
| `interaction_offset` | — | `1.0` | **The exact local-detailed-balance value.** Also the spontaneous-switching term. Do not tune casually. |
| `control_strength` | not in paper | `2.0` | Additive push toward the target; 2x maximal peer pressure. Breaks detailed balance by design. |

### Things to know before changing anything

1. `interaction_offset` away from `1.0` costs you the clean affinity `mu_r =
   ln(h/a)`. Verified numerically in §3.
2. `forward_rate != reverse_rate` introduces a real intrinsic bias, and in a
   controlled run the controller's effect and the kernel's bias become
   confounded. Change it only for uncontrolled runs, or knowingly.
3. `interaction_factor: constant` removes herding entirely — every destination
   equally likely. Useful as a second, even weaker null.
4. There is no `q` knob. The current kernel is `q = 1` and therefore has no
   phase transition (§5).
5. The clock is the event index, not physical time. Rates are recoverable from
   `classical_candidate_channels`; per-unit-time quantities are not directly
   readable from event counts (§5).

---

## See also

- [`hidden_bench_imitation.md`](hidden_bench_imitation.md) — full mechanics,
  event schema, every metric, the MI/CMI estimators, and report files.
- [`imitation_mutual_information.md`](../imitation_mutual_information.md) — the
  information-theoretic analysis applied to this game.
- [`README.md`](README.md) — the three HiddenBench games and the hidden-profile
  paradigm they share.
