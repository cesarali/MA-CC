# Reading `information_estimates.md`: the sensing and actuation channels

This document explains, end to end, where every number in a HiddenBench imitation run's
`information_estimates.md` comes from — what was counted, what table it was counted into, what
formula was applied to that table, and how to read the answer. Each section gives the
**plain-terms** version first and the **technical** version second.

It is the companion to [`sweep_mutual_information.md`](sweep_mutual_information.md), which covers
the *grid-level* estimators. The two files measure different things with the same arithmetic, and
[§14](#14-how-these-differ-from-the-sweep-metrics) explains exactly where they part ways. If you
only read one section, read that one — the difference is not cosmetic, and the two families of
numbers are not comparable.

The worked example throughout is a real run,
`results/hidden_bench_imitation/hidden-bench-imitation-reasoning-control-10/hidden-bench-imitation-reasoning-control-10-20260840/`,
produced by
[`hidden_bench_imitation_reasoning_control_10.yaml`](../../configs/runs/hidden_bench/hidden_bench_imitation_reasoning_control_10.yaml).
Every arithmetic result below was recomputed from that run's files, so you can check any of it.

## Table of contents

1. [The file, and the one question it answers](#1-the-file-and-the-one-question-it-answers)
2. [Where this sits: post-hoc, not tiered](#2-where-this-sits-post-hoc-not-tiered)
3. [The variables, and the four statistics defined](#3-the-variables-and-the-four-statistics-defined)
4. [The estimators, mathematically](#4-the-estimators-mathematically)
5. [`sensing_mi`, worked in full](#5-sensing_mi-worked-in-full)
6. [The three actuation CMIs](#6-the-three-actuation-cmis)
7. [`target_actuation_cmi`, worked in full](#7-target_actuation_cmi-worked-in-full)
8. [The slice-collapse problem](#8-the-slice-collapse-problem)
9. [Episode bootstrap: how precise is that](#9-episode-bootstrap-how-precise-is-that)
10. [Temporal nulls: how much of that is noise](#10-temporal-nulls-how-much-of-that-is-noise)
11. [Reading this run's numbers](#11-reading-this-runs-numbers)
12. [The order-parameter projections](#12-the-order-parameter-projections)
13. [Configuring and recomputing](#13-configuring-and-recomputing)
14. [How these differ from the sweep metrics](#14-how-these-differ-from-the-sweep-metrics)
15. [Where the code is](#15-where-the-code-is)

---

## 1. The file, and the one question it answers

**In plain terms.** An imitation run puts a controller next to a population of agents. The
controller peeks at a couple of agents, decides whether to argue for its preferred answer, and the
population then updates. Two things could go wrong, and these four numbers separate them:

- **Can the controller *see*?** `sensing_mi` — how much its two-agent peek actually tells it about
  the real state of the population.
- **Can the controller *act*?** The three `*_actuation_cmi` numbers — how much its decision to
  argue changes what happens next, measured at three different resolutions (the whole population,
  just the target option's headcount, just the one agent being talked to).

A controller can have a perfect sensor and no lever, or a lever and no sensor. Reporting the two
halves separately is the point of the file.

Both halves are then asked a second time against the **order parameters** — the macroscopic scalars
`m_ctrl`, `m_truth` and `m_order` — instead of the raw population counts. That is six more numbers,
ten in total; they are self-contained in [§12](#12-the-order-parameter-projections) and every
convention in §§2–11 applies to them unchanged.

**Technically.** One interaction event is a causal tuple

```text
population before -> sensor measurement -> controller action -> population after
N_t                  Y_t                  U_t                  N_t1
```

and the four statistics are four channels through that tuple: one unconditional MI on the
*sensing* edge, and three CMIs on the *actuation* edge, each conditioned on the pre-event state so
that ordinary population inertia is not credited to the controller. Every symbol is defined and
every statistic written out as a formula in
[§3](#3-the-variables-and-the-four-statistics-defined). All four are estimated in
**bits** by direct counting over finite discrete alphabets, and all four are written by
[`analysis.py`](../../src/mas_cc/games/hidden_bench/imitation/analysis.py) to
`hidden_bench_imitation_analysis/information_estimates.md`.

The run used here is a single cell:

| | |
| --- | --- |
| Scientific cell | **B** (`dynamics_mode: reasoning`, `control.mechanism: threshold_target`) |
| Task | `evacuation_north_hill`, options `[West City, East Town, North Hill]`, correct `North Hill` |
| Population | $N = 4$ agents, sensor sample $S = 2$, threshold $0.5$, target `correct` → `North Hill` |
| Sample | **5 episodes, 332 events** |

---

## 2. Where this sits: post-hoc, not tiered

**This analysis has no streaming tier.** Unlike the sweep metrics, nothing here is computed while
the episode runs and nothing is refreshed as cells complete. The estimator reads finished
`trajectory.jsonl` files off disk and does all its work at once.

That is a deliberate consequence of two requirements, both stated in the module docstring:

- **Episodes are the bootstrap unit.** A confidence interval that resamples episodes cannot be
  formed until whole episodes exist.
- **The nulls are temporal.** Permuting $U_t$ *within an episode* requires the complete episode in
  memory before it can be perturbed.

```text
data/episodes/<episode-id>/trajectory.jsonl     one JSON line per interaction event
        |
        |  read_imitation_events() + adapt_event()      canonical categorical tuple
        v
ImitationEvent(N_t, Y_t, U_t, N_t1, Z_t, Z_t1, Xf_t, Xf_t1)
        |
        |  information_analysis()                estimate, bootstrap, null, diagnostics
        v
hidden_bench_imitation_analysis/
    information_estimates.md      the four headline numbers, per cell
    information_nulls.csv         one row per statistic per permutation
    support_diagnostics.csv       the sparsity/degeneracy audit
```

The practical consequence: **the analysis is fully re-runnable from a finished run directory** with
different bootstrap counts, permutation counts, or a different subset of statistics, without
issuing a single provider call. See [§13](#13-configuring-and-recomputing).

Estimates are computed **per cell**, never pooled across cells. Two cells with different dynamics
or different controllers are different channels, and averaging them would be meaningless.

---

## 3. The variables, and the four statistics defined

This is the reference section: every symbol used anywhere in the document is defined here, and the
four reported statistics are written out as formulas in those symbols. If you only want to know
*what the file reports*, this section is sufficient on its own.

### 3.1 The cell constants

**In plain terms.** A few numbers are fixed for the whole cell and never vary event to event: how
many options there are, how many agents, how many the controller peeks at, which option it is
pushing, and how stubbornly it pushes.

**Technically.** Fixed for the cell, with this run's values:

| Symbol | Meaning | This run |
| --- | --- | --- |
| $K$ | number of options, in the canonical `possible_answers` order $(o_1, \ldots, o_K)$ | $3$: `(West City, East Town, North Hill)` |
| $N$ | population size | $4$ |
| $S$ | sensor sample size (`control.options.sensor_sample_size`) | $2$ |
| $z$ | index, in canonical order, of the **target** option the controller promotes | $3$ (`North Hill`, here also the correct answer) |
| $\theta$ | controller threshold (`control.options.threshold`) | $0.5$ |

Write $e_k$ for the $k$-th standard basis vector of $\mathbb{Z}^K$, so $e_k$ is "one agent at
option $o_k$".

### 3.2 The per-event variables

**In plain terms.** Each event is boiled down to a handful of small categorical labels. Everything
else in the trajectory — the messages, the reasoning traces, the agent identities — is thrown away
before any information is measured. The estimators only ever see these labels.

**Technically.** [`adapt_event`](../../src/mas_cc/games/hidden_bench/imitation/analysis.py#L154)
builds an `ImitationEvent` with these information-bearing fields:

| Symbol | Field | Support | Definition | Alphabet seen in this run |
| --- | --- | --- | --- | --- |
| $N_t$ | `N_t` | $\{n \in \mathbb{Z}_{\ge 0}^K : \sum_k n^{(k)} = N\}$ | $N_t^{(k)}$ = how many of the $N$ agents vote option $o_k$ **just before** the event. E.g. `(0, 1, 3)` = 0 West City, 1 East Town, 3 North Hill. | 14 states |
| $Y_t$ | `Y_t` | $\{y : \sum_k y^{(k)} = S\}$ | Counts over the $S$ agents drawn uniformly **without replacement** from the population — so $Y_t \mid N_t$ is multivariate hypergeometric. **This is the only thing the controller ever observes.** | 6 states |
| $U_t$ | `U_t` | $\{\texttt{ADVOCATE\_Z}, \texttt{NO\_OP}\}$ | The controller's action, a deterministic function of the sensor alone: $U_t = \texttt{ADVOCATE\_Z}$ iff $Y_t^{(z)} / S < \theta$, else $\texttt{NO\_OP}$. | 2 |
| $X^f_t$ | `Xf_t` | one of the $K$ options | Vote of the **focal** agent — the single agent selected to update this event — before the interaction. | 3 |
| $X^f_{t+1}$ | `Xf_t1` | one of the $K$ options | That same agent's vote after the interaction. | 3 |
| $N_{t+1}$ | `N_t1` | same as $N_t$ | Population counts **after**. Exactly one agent may move, so $N_{t+1} = N_t - e_{X^f_t} + e_{X^f_{t+1}}$. | 14 states |
| $Z_t,\, Z_{t+1}$ | `Z_t`, `Z_t1` | $\{0, 1, \ldots, N\}$ | The **target option's headcount**, i.e. coordinate $z$ of the population vector: $Z_t = N_t^{(z)}$ and $Z_{t+1} = N_{t+1}^{(z)}$. | 5 each ($0\ldots4$) |

The causal chain one event traces out, in these symbols:

```text
   N_t   ──sample S of N──▶   Y_t   ──threshold θ──▶   U_t   ──▶ X^f_t → X^f_{t+1} ──▶   N_{t+1}
population       (hypergeometric)   sensor   (deterministic)  action    one agent updates    population
  before                            reading                              (the focal one)       after

   └────────── sensing_mi ─────────┘         └──────────── the three actuation CMIs ───────────┘
```

Three details that are load-bearing later:

- **$U_t$ is a deterministic function of $Y_t$, and $Y_t$ is a noisy function of $N_t$.** The
  controller has no other input — no memory, no transcript, no knowledge of $N_t$ itself. That one
  fact is the entire cause of [§8](#8-the-slice-collapse-problem).
- **$Z$ is not an independent variable; it is a coordinate of $N$.** $Z_t = N_t^{(z)}$ is a
  deterministic projection, which is why `target_actuation_cmi` is a *coarsening* of
  `population_actuation_cmi` rather than a separate measurement ([§3.4](#34-how-the-three-actuation-cmis-relate)).
- **Only the focal agent can move per event.** $N_{t+1}$ is therefore fully determined by
  $(N_t, X^f_t, X^f_{t+1})$ — which likewise ties `focal_actuation_cmi` to
  `population_actuation_cmi` exactly ([§3.4](#34-how-the-three-actuation-cmis-relate)).

Plus three encoding choices:

- **Counts are tuples in canonical option order**, never dictionaries. `_canonical_counts` projects
  onto `possible_answers` order, so `(0, 1, 3)` always means "0 West City, 1 East Town, 3 North
  Hill". Dictionary insertion order would make the alphabet depend on which option happened to
  appear first in a given episode.
- **The count state, not the labelled state, is the population variable.** The population is
  exchangeable for the purposes of this measurement; using labelled vectors would multiply the
  alphabet by $\binom{N}{\cdot}$ for no informational gain.
- **$Z_t$ is the integer headcount, not `m_ctrl`.** For fixed $N$ and $K$ the two are one-to-one,
  but an integer is a safe contingency-table category and a float is not.

Uncontrolled events have $Y_t = $ `None` and $U_t = $ `None`. They are **excluded** from every
statistic rather than counted as a third action class — a cell with `mechanism: none` produces
behavioral summaries but no sensing or actuation MI at all, by design.

### 3.3 The four statistics, as formulas

This is the table the rest of the document elaborates. All four are in bits.

| Statistic | Formula | Channel input | Channel output | Conditioned on | The question it answers |
| --- | --- | --- | --- | --- | --- |
| `sensing_mi` | $I(N_t \,;\, Y_t)$ | $N_t$ — the true population state | $Y_t$ — the controller's peek | *nothing* | **Can the controller see?** How much does a sample of $S$ of the $N$ agents reveal about how all $N$ are voting? |
| `population_actuation_cmi` | $I(U_t \,;\, N_{t+1} \mid N_t)$ | $U_t$ — the action | $N_{t+1}$ — the whole count vector after | $N_t$ — the whole count vector before | **Does the action move the population at all?** The complete, highest-resolution answer. |
| `target_actuation_cmi` | $I(U_t \,;\, Z_{t+1} \mid Z_t)$ $\;=\; I\big(U_t \,;\, N^{(z)}_{t+1} \mid N^{(z)}_t\big)$ | $U_t$ | $Z_{t+1}$ — headcount on the advocated option, after | $Z_t$ — that same headcount, before | **Does the action move the specific option it is arguing for?** Everything is projected onto coordinate $z$. |
| `focal_actuation_cmi` | $I(U_t \,;\, X^f_{t+1} \mid X^f_t,\, N_t)$ | $U_t$ | $X^f_{t+1}$ — the new vote of the one agent that updated | the pair $(X^f_t, N_t)$ — that agent's old vote *and* the full population it sat in | **Does the action move the individual agent being talked to?** |

In words, one at a time:

- **`sensing_mi`** is the only statistic with **no $U_t$ in it at all**. It measures the
  *instrument*, not the intervention — the quality of the controller's eyes, independent of what it
  does with what it sees. It is also the only unconditional one, and the only one that stays
  meaningful when the controller's policy is degenerate. Worked in full in
  [§5](#5-sensing_mi-worked-in-full).
- **The three `*_actuation_cmi` statistics all share the same channel input, $U_t$.** They differ
  only in *which piece of "what happened next" they look at* and *what they hold fixed*. Each asks:
  given where the population already was, how much does knowing whether the controller argued tell
  you about where it went? The conditioning is what keeps ordinary population inertia — a unanimous
  population staying unanimous — from being credited to the controller. Elaborated in
  [§6](#6-the-three-actuation-cmis), worked in full for the target case in
  [§7](#7-target_actuation_cmi-worked-in-full).

**Scale.** Mutual information is bounded by the entropy of either argument, so:

$$\text{sensing\_mi} \le \min\{H(N_t),\, H(Y_t)\} \qquad\text{and}\qquad \text{every actuation CMI} \le H(U_t \mid \text{its conditioning variable}) \le H(U_t)$$

The second bound is worth internalizing before reading any actuation number as "small": with a
binary action, $H(U_t) \le 1$ bit always, and in this run the conditioning drives the real ceiling
far below even that — see [§8](#8-the-slice-collapse-problem).

> **These four are the microstate family.** Six further statistics ask the same two questions
> against the macroscopic order parameters `m_ctrl`, `m_truth` and `m_order` instead of $N_t$; they
> are defined in [§12](#12-the-order-parameter-projections) and share every estimator, bootstrap
> and null convention described below.

### 3.4 How the three actuation CMIs relate

They are **not** three independent measurements; two exact structural relations connect them, both
following from [§3.2](#32-the-per-event-variables).

**`focal` is `population` with one extra conditioning variable.** Given $N_t$ and $X^f_t$, the map
$X^f_{t+1} \mapsto N_{t+1} = N_t - e_{X^f_t} + e_{X^f_{t+1}}$ is a bijection, so conditioning on
that pair makes the two outcome variables informationally identical:

$$\text{focal\_actuation\_cmi} = I(U_t \,;\, X^f_{t+1} \mid X^f_t, N_t) = I(U_t \,;\, N_{t+1} \mid X^f_t, N_t)$$

Compare that to $\text{population\_actuation\_cmi} = I(U_t \,;\, N_{t+1} \mid N_t)$: **same input,
same outcome, and the only difference is whether the focal agent's own prior vote is in the
conditioning set.** `focal` is the sharper question — "did the action change *this agent's* mind,
given who they already were" — bought at the cost of a conditioning alphabet multiplied by $K$.

**`target` is a projection of `population` on both axes.** $Z_{t+1} = N^{(z)}_{t+1}$ is a
deterministic function of $N_{t+1}$, so the data-processing inequality gives
$I(U_t; Z_{t+1} \mid N_t) \le I(U_t; N_{t+1} \mid N_t)$ — coarsening the *outcome* can only lose
information. But `target_actuation_cmi` also coarsens the *conditioning*, from $N_t$ to $Z_t$, and
changing the conditioning has **no fixed sign**. So

$$\text{target\_actuation\_cmi} \not\le \text{population\_actuation\_cmi} \quad \text{in general.}$$

In practice the outcome-coarsening usually dominates and `target` comes out smaller (0.0158 vs.
0.0347 here). When it does *not*, the usual cause is the conditioning: $N_t$ carries detail beyond
$Z_t$ that fragments the table and starves each slice, while the coarser $Z_t$ conditioning
survives the collapse of [§8](#8-the-slice-collapse-problem) better. Reading the two together is
therefore diagnostic, not redundant.

> **One discrepancy to be aware of.** The `target_actuation_cmi` blurb that
> [`analysis.py`](../../src/mas_cc/games/hidden_bench/imitation/analysis.py#L50) writes into the
> top of `information_estimates.md` describes the outcome as "**whether** the target option's count
> after the interaction differs from before", i.e. an indicator $\mathbb{1}[Z_{t+1} \ne Z_t]$. The
> code does not do that — it passes $Z_{t+1}$ itself, a 5-level variable. The formula in the table
> above is what is computed.

---

## 4. The estimators, mathematically

All estimates are **plug-in** (direct-counting) estimators over finite discrete alphabets, in
**bits**, and they all live in
[`analysis/estimators.py`](../../src/mas_cc/analysis/estimators.py) — the same module the sweep
metrics use.

### 4.1 Entropy

For a count table $n$ with total $N = \sum n$,

$$\hat H(n) = -\sum_{i \,:\, n_i > 0} \frac{n_i}{N} \log_2 \frac{n_i}{N}$$

### 4.2 Mutual information

From a two-dimensional contingency table $n_{xy}$,

$$\hat I(X;Y) = \hat H(n_{x\cdot}) + \hat H(n_{\cdot y}) - \hat H(n_{xy})$$

### 4.3 Conditional mutual information

From a three-dimensional table with axes ordered $(X, Z, Y)$,

$$\hat I(X;Y \mid Z) = \hat H(X,Z) + \hat H(Y,Z) - \hat H(Z) - \hat H(X,Y,Z)$$

Equivalently, in the form that makes [§8](#8-the-slice-collapse-problem) obvious:

$$I(X;Y \mid Z) = \sum_z p(z) \; \underbrace{I(X;Y \mid Z = z)}_{\text{MI inside the slice}}$$

— a $p(z)$-weighted average, over conditioning states actually visited, of the mutual information
*within* that state. **Every slice contributes independently, and a slice can contribute exactly
zero.**

### 4.4 The alphabet is the observed alphabet

[`mutual_information`](../../src/mas_cc/analysis/estimators.py#L47) and
[`conditional_mutual_information`](../../src/mas_cc/analysis/estimators.py#L94) build their levels
with `dict.fromkeys(x)` over the **values actually present**. There is no declared Cartesian
alphabet: a population state that never occurred gets no row.

This matters for smoothing, below, and it means the alphabet is a property of the sample rather
than of the game. Comparing a 5-episode estimate to a 50-episode estimate compares two different
tables.

### 4.5 The three variants, and why `unsmoothed` is the headline here

| Field | What it is |
| --- | --- |
| `unsmoothed` | the raw plug-in estimate — **the main reported value**, duplicated into `estimate` |
| `jeffreys` | $\hat I(n + \tfrac12)$, adding $\tfrac12$ to every cell of the observed-level table |
| `miller_madow` | $\hat I(n)$ with $\hat H \mathrel{+}= \frac{K-1}{2N\ln 2}$ per constituent entropy, $K$ = occupied cells |

The sweep metrics treat **Jeffreys** as the answer. This analysis deliberately does not, and the
reason is visible in this run's own numbers. The Jeffreys prior adds $0.5$ per cell of the *full*
observed-level table, and these conditional tables are far larger than the sample:

| Statistic | Table shape | Cells | Pseudo-counts added | Real events | Pseudo / real |
| --- | --- | ---: | ---: | ---: | ---: |
| `sensing_mi` | $14 \times 6$ | 84 | 42.0 | 332 | 0.13 |
| `target_actuation_cmi` | $2 \times 5 \times 5$ | 50 | 25.0 | 332 | 0.08 |
| `focal_actuation_cmi` | $2 \times 25 \times 3$ | 150 | 75.0 | 332 | 0.23 |
| `population_actuation_cmi` | $2 \times 14 \times 14$ | 392 | **196.0** | 332 | **0.59** |

For `population_actuation_cmi`, Jeffreys injects 196 pseudo-observations against 332 real ones.
Worse, it injects them into *empty* $(u, z, y)$ cells — which **manufactures action variation in
conditioning slices where the controller never varied its action**, and therefore manufactures
CMI. That is why `jeffreys` (0.1460) comes out **four times larger** than `unsmoothed` (0.0347)
here, rather than shrinking it toward independence as it does in a well-sampled table.

Smoothing is a mild regularizer on a dense table and a dominant fiction on a sparse one. Hence the
main value is unsmoothed, `main_estimator_variant: unsmoothed` is written into every row, and the
other two variants are reported beside it as sensitivity checks rather than alternatives.

> **Reading the Miller–Madow column.** Its correction to a CMI is
> $\frac{K_{XZ} + K_{YZ} - K_Z - K_{XYZ}}{2 N \ln 2}$, which can vanish. In this run it is
> $-0.0065$ bits for both `population_` and `target_actuation_cmi`, and exactly $0$ for
> `focal_actuation_cmi` ($K_{XZ}=33$, $K_{YZ}=41$, $K_Z=25$, $K_{XYZ}=49$). `miller_madow`
> matching `unsmoothed` to every decimal is arithmetic, not a bug.

Fewer than two levels on any axis returns `NaN`, not `0.0` — "not yet", not "zero bits".

---

## 5. `sensing_mi`, worked in full

$$\text{sensing\_mi} = I(N_t \,;\, Y_t)$$

**In plain terms.** The controller samples 2 of the 4 agents and sees what they think. How much
does that tell it about what all 4 think? This is the one statistic with no conditioning and no
controller action in it — it measures the *instrument*, not the intervention. It is also the only
one that is meaningful even when the controller's policy is degenerate.

**The table.** Rows are the 14 observed $N_t$ states, columns the 6 observed $Y_t$ states, in
`(West City, East Town, North Hill)` order:

| $N_t$ \ $Y_t$ | `(0,0,2)` | `(0,1,1)` | `(0,2,0)` | `(1,0,1)` | `(1,1,0)` | `(2,0,0)` | total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `(0,0,4)` | 87 | 0 | 0 | 0 | 0 | 0 | 87 |
| `(0,1,3)` | 1 | 3 | 0 | 0 | 0 | 0 | 4 |
| `(0,2,2)` | 1 | 6 | 3 | 0 | 0 | 0 | 10 |
| `(0,3,1)` | 0 | 5 | 4 | 0 | 0 | 0 | 9 |
| `(0,4,0)` | 0 | 0 | 40 | 0 | 0 | 0 | 40 |
| `(1,0,3)` | 4 | 0 | 0 | 1 | 0 | 0 | 5 |
| `(1,1,2)` | 0 | 3 | 0 | 3 | 0 | 0 | 6 |
| `(1,2,1)` | 0 | 2 | 4 | 2 | 0 | 0 | 8 |
| `(2,0,2)` | 0 | 0 | 0 | 5 | 0 | 1 | 6 |
| `(2,1,1)` | 0 | 1 | 0 | 0 | 2 | 0 | 3 |
| `(2,2,0)` | 0 | 0 | 0 | 0 | 2 | 0 | 2 |
| `(3,0,1)` | 0 | 0 | 0 | 2 | 0 | 0 | 2 |
| `(3,1,0)` | 0 | 0 | 0 | 0 | 55 | 50 | 105 |
| `(4,0,0)` | 0 | 0 | 0 | 0 | 0 | 45 | 45 |
| **total** | **93** | **20** | **51** | **13** | **59** | **96** | **332** |

The table has visible hypergeometric structure — every nonzero entry sits where a 2-of-4 draw from
that population *could* land, and the zeros are structural, not sampling gaps. That is the shape a
correct sensor should have.

**The arithmetic:**

$$\hat H(N_t) = 2.74048 \text{ bits}, \qquad \hat H(Y_t) = 2.31717 \text{ bits}, \qquad \hat H(N_t, Y_t) = 3.21702 \text{ bits}$$

$$\hat I(N_t; Y_t) = 2.74048 + 2.31717 - 3.21702 = \mathbf{1.84063} \text{ bits}$$

**How to read the magnitude.** The ceiling is $\min\{H(N_t), H(Y_t)\} = H(Y_t) = 2.317$ bits — the
sensor cannot convey more than its own entropy. We get 1.841, i.e. **79% of the sensor's own
capacity**, and the residual uncertainty about the population is

$$\hat H(N_t \mid Y_t) = 2.74048 - 1.84063 = 0.900 \text{ bits}$$

So a 2-of-4 sample resolves about two-thirds of the population-state uncertainty and leaves
roughly 0.9 bits standing. That is the expected cost of partial observation, and the one place it
visibly bites is $N_t = (3,1,0)$: 105 events split 55/50 between two different sensor readings,
because a 2-draw from three West City and one East Town agent genuinely cannot distinguish them.

**Note the alphabet is not fixed by the game.** The 14 rows are the states this run visited, not
the $\binom{6}{2} = 15$ compositions of 4 agents over 3 options. A longer run visits more states,
$H(N_t)$ rises, and `sensing_mi` typically rises with it. Do not compare `sensing_mi` across runs
of different length without checking `unique_N_t_states` first.

---

## 6. The three actuation CMIs

All three ask the same question at three resolutions: **given where the population already is, how
much does knowing the controller's action tell you about where it goes next?**

The conditioning is the whole point. Without it, the number would mostly measure the fact that a
population which is already unanimous stays unanimous — inertia, not control.

| Statistic | Definition | Conditioning states in this run | Why it exists |
| --- | --- | ---: | --- |
| `population_actuation_cmi` | $I(U_t \,;\, N_{t+1} \mid N_t)$ | 14 | The complete answer: does the action move the full occupation state? Highest resolution, sparsest table. |
| `target_actuation_cmi` | $I(U_t \,;\, Z_{t+1} \mid Z_t)$ | 5 | Projection onto the target headcount. Much denser, and directly about the thing the controller is trying to change. |
| `focal_actuation_cmi` | $I(U_t \,;\, X^f_{t+1} \mid X^f_t, N_t)$ | 25 | Only one agent can change per event — this looks at that agent directly, conditioning on both its own opinion and the population it sits in. |

The three are **not** redundant, and they can disagree informatively — see
[§3.4](#34-how-the-three-actuation-cmis-relate) for the two exact relations that connect them.
In short: `focal_actuation_cmi` is `population_actuation_cmi` with the focal agent's own prior vote
added to the conditioning set (the outcome variables are informationally identical once you
condition on $X^f_t$), and `target_actuation_cmi` coarsens both the outcome and the conditioning
onto coordinate $z$, which usually — but not necessarily — makes it the smallest of the three.

A structural caution: `focal_actuation_cmi`'s conditioning category is the tuple
$(X^f_t, N_t)$, which multiplies alphabets. In this run that is 25 occupied states across 332
events — median 3 events per state, 2.1% of events in singleton states, and
`sparse_conditioning_table: true`. It is the only one of the four flagged sparse here.

---

## 7. `target_actuation_cmi`, worked in full

This is the smallest table of the three, so it can be shown completely — and it makes the central
problem with all of them visible at a glance.

**The table.** $U_t \times Z_t \times Z_{t+1}$, listing each conditioning slice:

| $Z_t$ | $n$ | $p(z)$ | $U_t$ | $n$ | $Z_{t+1}$ distribution |
| ---: | ---: | ---: | --- | ---: | --- |
| 0 | 192 | 0.578 | `ADVOCATE_Z` | 192 | `{0: 190, 1: 2}` |
| | | | `NO_OP` | **0** | — |
| 1 | 22 | 0.066 | `ADVOCATE_Z` | 10 | `{0: 2, 1: 2, 2: 6}` |
| | | | `NO_OP` | 12 | `{0: 3, 1: 6, 2: 3}` |
| 2 | 22 | 0.066 | `ADVOCATE_Z` | 4 | `{2: 2, 3: 2}` |
| | | | `NO_OP` | 18 | `{1: 7, 2: 8, 3: 3}` |
| 3 | 9 | 0.027 | `ADVOCATE_Z` | **0** | — |
| | | | `NO_OP` | 9 | `{2: 3, 3: 4, 4: 2}` |
| 4 | 87 | 0.262 | `ADVOCATE_Z` | **0** | — |
| | | | `NO_OP` | 87 | `{4: 87}` |

**The decomposition.** Applying $I(X;Y\mid Z) = \sum_z p(z)\, I(X;Y \mid Z=z)$ slice by slice:

| $Z_t$ | $p(z)$ | actions present | $I$ inside slice | contribution |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.578 | **1** | 0.00000 | 0.00000 |
| 1 | 0.066 | 2 | 0.10268 | 0.00680 |
| 2 | 0.066 | 2 | 0.13522 | 0.00896 |
| 3 | 0.027 | **1** | 0.00000 | 0.00000 |
| 4 | 0.262 | **1** | 0.00000 | 0.00000 |
| | | | **total** | **0.01576** |

which is `target_actuation_cmi_unsmoothed` exactly.

**Read that table carefully.** The entire statistic — all 0.0158 bits of it — is produced by
**44 of the 332 events**. The other 288 events sit in slices where only one action was ever
observed, and a slice with one action has zero mutual information *by construction*, regardless of
what the population did in it. The 192 events at $Z_t = 0$ contribute nothing not because the
controller failed there, but because there is no contrast to measure.

---

## 8. The slice-collapse problem

**In plain terms.** The controller is a thermostat: it argues exactly when the target is losing.
That is good engineering and terrible for measurement. Because its action is decided *by* the
state we then condition on, conditioning on the state removes almost all the variation in the
action — and mutual information can only see variation.

**Technically.** The `threshold_target` policy is a deterministic function of the sensor,
$U_t = \mathbb{1}[\,y_Z/S < \theta\,]$, and the sensor is a noisy function of the state. With
$N=4$, $S=2$, $\theta=0.5$, the sensor sees $y_Z = 0$ with certainty when $Z_t = 0$ and $y_Z = 2$
with certainty when $Z_t = 4$. So $U_t$ is *deterministic* given $Z_t$ at both ends of the range,
and those ends are where the population spends most of its time.

Measured across the three conditioning schemes in this run:

| Statistic | Conditioning variable | Events in single-action slices |
| --- | --- | ---: |
| `target_actuation_cmi` | $Z_t$ | 288 / 332 = **86.7%** |
| `population_actuation_cmi` | $N_t$ | 296 / 332 = **89.2%** |
| `focal_actuation_cmi` | $(X^f_t, N_t)$ | 300 / 332 = **90.4%** |

**Roughly 90% of the sample is informationally inert for every actuation statistic.** The effective
sample size behind these CMIs is not 332 events; it is 30–45.

**What that costs, in bits.** Recall the ceiling from [§3.3](#33-the-four-statistics-as-formulas):
an actuation CMI cannot exceed $H(U_t \mid \text{conditioning})$. A single-action slice contributes
exactly $0$ to that conditional entropy and a binary slice contributes at most $1$ bit, so the
collapse caps every actuation statistic at roughly the multi-action share of the sample:

| Statistic | $H(U_t)$ unconditional | Ceiling $H(U_t \mid \text{cond.})$ | Estimate | Share of the ceiling actually attained |
| --- | ---: | ---: | ---: | ---: |
| `population_actuation_cmi` | 0.9577 | $\le 36/332 = 0.108$ | 0.03474 | $\ge 32\%$ |
| `target_actuation_cmi` | 0.9577 | $= 0.111$ (exact, from [§7](#7-target_actuation_cmi-worked-in-full)) | 0.01576 | $14\%$ |
| `focal_actuation_cmi` | 0.9577 | $\le 32/332 = 0.096$ | 0.02907 | $\ge 30\%$ |

The controller varied its action by 0.958 bits overall, but **conditioning leaves only about 0.1
bits of that variation available to be informative** — a 9× reduction before the estimator sees a
single transition. So these estimates are not "0.03 bits out of a possible 1.0". They are 0.03 out
of a possible 0.11, and read that way the actuation numbers are small *because the measurement
budget is small*, not obviously because the controller is ineffective. Distinguishing those two
readings is exactly what the permutation null in
[§10](#10-temporal-nulls-how-much-of-that-is-noise) fails to do here, and why the directional
behavioral metrics are the tiebreaker.

This is not a bug in the estimator, and it is not fixed by more episodes alone — more episodes of
the same policy reproduce the same collapse proportionally. The levers that actually change it:

- **Randomize the action.** An $\varepsilon$-greedy or stochastic-threshold controller breaks the
  determinism and puts both actions in every slice. This is the direct fix, and it is what makes
  the sweep-tier design work (see [§14](#14-how-these-differ-from-the-sweep-metrics)).
- **Move the threshold off the boundary.** $\theta = 0.5$ with $S = 2$ means the sensor's three
  possible readings straddle the cutoff at exactly one point. A larger `sensor_sample_size` gives
  the sensor more intermediate readings and spreads the action across more states.
- **Prefer the coarser conditioning.** `target_actuation_cmi` collapses least of the three above,
  which is one reason it is worth reporting alongside the full-state version. Pushed further, this
  is exactly what the order-parameter projections of [§12](#12-the-order-parameter-projections)
  do: conditioning on `m_order` cuts the inert share here from ~52% to ~29%
  ([§12.5](#125-what-this-looks-like-on-real-data)) — at the cost of controlling for less
  ([§12.6](#126-what-the-coarser-conditioning-costs)).

Always read `occupied_conditioning_states`, `min/median/max_events_per_conditioning_state`, and
`fraction_events_singleton_conditioning_states` in the diagnostics table before quoting an
actuation number. In this run:

| Statistic | Occupied states | Events/state (min / median / max) | Singleton share | Sparse |
| --- | ---: | --- | ---: | --- |
| `sensing_mi` | 0 (unconditional) | — | — | no |
| `population_actuation_cmi` | 14 | 2 / 7 / 105 | 0% | no |
| `target_actuation_cmi` | 5 | 9 / 22 / 192 | 0% | no |
| `focal_actuation_cmi` | 25 | 1 / 3 / 87 | 2.1% | **yes** |

> **`controller_degenerate` is a different, coarser check.** It fires only when the controller
> never varied its action *at all* across the whole cell. Here $H(U) = 0.9577$ bits (206
> `ADVOCATE_Z` vs. 126 `NO_OP`), so all four statistics are marked
> `scientifically_interpretable: true` — and yet 90% of the sample is still inert. **A `true`
> interpretability flag does not mean the conditioning survived.** The flag and the sparsity
> diagnostics answer different questions; read both.

---

## 9. Episode bootstrap: how precise is that

**In plain terms.** Re-run the estimate many times on resampled versions of the data to see how
much the answer wobbles.

**The procedure**, from
[`bootstrap_episode_ids`](../../src/mas_cc/games/hidden_bench/imitation/analysis.py#L368):

1. Collect the unique episode IDs in the cell (here: 5).
2. Draw 5 of them **with replacement**.
3. Concatenate every event from each drawn episode, including repeated draws.
4. Recompute the main unsmoothed estimate.
5. Repeat `bootstrap_resamples` times (1000 here) and take percentile bounds at `confidence`.

**Whole episodes are the resampling unit, never individual events.** Events within an episode are a
single dependent trajectory — the same population, carried forward — so resampling events would
treat 332 correlated observations as 332 independent ones and produce an interval several times
too narrow.

The cost of doing it correctly is that **the interval is governed by the episode count, not the
event count**. Five episodes is a very small bootstrap: the resampling distribution has at most
$\binom{9}{4} = 126$ distinct multisets, and a single unusual episode appears in a large fraction
of the draws. That is exactly what the `sensing_mi` interval shows:

| Statistic | Estimate | 95% bootstrap CI | Width |
| --- | ---: | --- | ---: |
| `sensing_mi` | 1.84063 | [0.75263, 1.97000] | 1.217 |
| `population_actuation_cmi` | 0.03474 | [0.01643, 0.06583] | 0.049 |
| `target_actuation_cmi` | 0.01576 | [0.01259, 0.03897] | 0.026 |
| `focal_actuation_cmi` | 0.02907 | [0.00972, 0.06039] | 0.051 |

The `sensing_mi` interval is wildly asymmetric — the point estimate sits at the 1.841 upper end
while the lower bound falls to 0.753. That skew is the signature of a 5-episode bootstrap where
dropping one episode removes several population states from the alphabet entirely. **The lever is
`execution.repetitions`.** Ten or twelve episodes per cell is a sanity-pilot size; 50–100 is what
these intervals need.

---

## 10. Temporal nulls: how much of that is noise

**In plain terms.** Even a controller doing nothing useful produces a positive MI, from random
coincidence in a finite sample. This measures that floor by scrambling the timing of the
controller's actions — keeping every action it took, but detaching each from the moment it took
it — and re-measuring, hundreds of times.

**The procedure**, from
[`_perturb_within_episode`](../../src/mas_cc/games/hidden_bench/imitation/analysis.py#L462):

1. Within **each episode separately**, randomly permute the field of interest across that episode's
   events — $Y_t$ for `sensing_mi`, $U_t$ for the three actuation statistics.
2. Leave every other field untouched: the population transitions, the focal transitions, and the
   episode membership are exactly as observed.
3. Re-estimate.
4. Repeat `null_permutations` times (1000 here); report `null_mean`, `null_ci_low`, `null_ci_high`.

Permuting **within** rather than across episodes is what makes this a *temporal* null: it preserves
each episode's action mix and each episode's trajectory, and destroys only the alignment between
them. An across-episode shuffle would also destroy between-episode composition and test a weaker
hypothesis.

| Statistic | Estimate | Null mean | 95% null CI | Verdict |
| --- | ---: | ---: | --- | --- |
| `sensing_mi` | **1.84063** | 1.31991 | [1.27661, 1.37594] | **clears the null** |
| `population_actuation_cmi` | 0.03474 | 0.06350 | [0.04363, 0.08948] | **below the null band** |
| `target_actuation_cmi` | 0.01576 | 0.03215 | [0.01329, 0.05448] | inside the null band |
| `focal_actuation_cmi` | 0.02907 | 0.03835 | [0.01930, 0.06225] | inside the null band |

**The actuation estimates come in at or below their nulls. That is not a paradox, and it is worth
understanding rather than dismissing.**

The null is not "a controller with no effect". It is "these same actions, reshuffled in time" — and
reshuffling *destroys the thermostat coupling from [§8](#8-the-slice-collapse-problem)*. Verified
directly on this run:

| | Share of events in single-action slices ($Z_t$) | `target_actuation_cmi` |
| --- | ---: | ---: |
| Observed data | 86.7% | 0.01576 |
| Null permutations (mean) | **2.0%** | 0.03066 |

Permuting $U_t$ scatters both actions into every conditioning slice, so the null tables have ~40×
more informationally live events than the real table does — and correspondingly more room for
spurious association. **The null is structurally denser than the data it is being compared to.**

The correct reading is therefore: *this pilot provides no evidence of measurable actuation
information* — not *the controller has a negative effect*. Whether the shortfall is a genuinely
absent effect or the slice collapse masking a real one cannot be resolved from these 5 episodes.
The directional behavioral metrics (`advocacy_delta_m_ctrl`, `target_adoption_lift` in
`cell_summaries.csv`) are the independent check, because they measure a signed response rather
than a symmetric association and do not collapse under deterministic conditioning.

---

## 11. Reading this run's numbers

Put together, the file says one coherent thing:

| Reading | Evidence |
| --- | --- |
| The controller's **sensor works well**. | 1.841 bits against a 1.376-bit null ceiling, and 79% of the sensor's own 2.317-bit capacity. It resolves ~2/3 of population-state uncertainty. |
| The controller's **actions varied**. | $H(U) = 0.958$ bits, 206 advocate vs. 126 no-op — nowhere near degenerate. |
| There is **no measurable actuation information**. | All three CMIs sit at or below their permutation nulls. |
| But that result is **not yet conclusive**. | ~90% of events lie in single-action conditioning slices, so the effective sample behind the CMIs is 30–45 events, not 332 — and the conditioning caps the statistics at ~0.11 bits rather than $H(U)=0.958$, which the estimates attain 14–32% of. |
| The **estimates are imprecise**. | 5-episode bootstrap; the `sensing_mi` CI spans 0.75–1.97. |
| Smoothing is **not usable** on the population table. | 196 Jeffreys pseudo-counts against 332 real events; `jeffreys` is 4× `unsmoothed` for `population_actuation_cmi`. |

Three things these numbers **cannot** tell you:

- **Which direction the controller moved the population.** Mutual information is symmetric and
  label-free. For direction, read `advocacy_delta_m_ctrl` and `target_adoption_lift` in
  `cell_summaries.csv`.
- **Whether reasoning differs from the classical kernel.** That needs the matched 2×2 grid (cells
  A/B/C/D), not a single cell.
- **Whether the absent actuation signal is real.** See above; it needs a randomized controller or
  many more episodes.

The full interpretation checklist is in
[`hidden_bench/hidden_bench_imitation.md` §10](hidden_bench/hidden_bench_imitation.md#10-interpretation-checklist).

---

## 12. The order-parameter projections

Everything up to here measures the sensor and the action against the **microstate**: the full count
vector $N_t$, or one coordinate of it. This section covers the six statistics that ask the same two
questions against the **order parameters** — the macroscopic scalars `m_ctrl`, `m_truth` and
`m_order` that the rest of the pipeline already reports in `cell_summaries.csv`,
`order_parameter_trajectories.csv`, and the aggregate metric plots.

### 12.1 Why project at all

**In plain terms.** $N_t$ is the finest possible description of the population, and that is exactly
its problem. With $N=4$ and $K=3$ it has 15 possible values, this run visited 12 of them, and every
one of those 12 is a separate bucket the estimator has to fill before it can say anything. The
order parameters are one number each — "how much of the population is on the controller's side",
"…on the correct answer", "…on whichever option is winning" — so the same events pile into 3 to 5
buckets instead of 12. Fewer buckets, more events per bucket, and a conditioning variable that no
longer starves.

They also change the *question* into one you actually want answered. A physicist studying this
system does not ask "did the action move the population from $(1,1,2)$ to $(1,0,3)$". They ask "did
the action raise the alignment" or "did the action increase consensus". The order parameters are
those questions; $N_t$ is the raw data underneath them.

**Technically.** Each order parameter is a deterministic function of $N_t$, so on the *sensing* edge
the data-processing inequality applies and each projection is a lower bound on `sensing_mi`. On the
*actuation* edge the projection coarsens both the outcome and the conditioning, so — exactly as in
[§3.4](#34-how-the-three-actuation-cmis-relate) — it has **no fixed sign** relative to
`population_actuation_cmi`. What it reliably does is cut the number of conditioning slices, which
is the direct remedy for the slice collapse of [§8](#8-the-slice-collapse-problem). That remedy is
not free, and [§12.6](#126-what-the-coarser-conditioning-costs) is the price.

### 12.2 The three order parameters, and how they are encoded

**In plain terms.** All three are floats between $-\tfrac{1}{K-1}$ and $1$, and a float is a
terrible category to count with. But for a fixed population size each one is just a rescaled
headcount, so the estimators count the headcount instead. Same buckets, same answer, no float
comparisons.

**Technically.** With shares $p_k = N_t^{(k)}/N$, [`metrics.py`](../../src/mas_cc/games/hidden_bench/imitation/metrics.py#L31)
defines the alignment of an option $o$ as $m_o = \dfrac{K p_o - 1}{K - 1}$, and:

| Order parameter | Definition | Reads as | Integer it is one-to-one with |
| --- | --- | --- | --- |
| `m_ctrl` | $\dfrac{K\,N_t^{(z)}/N - 1}{K-1}$ | alignment with the option the **controller promotes** | $N_t^{(z)}$ — the target headcount, i.e. **$Z_t$** |
| `m_truth` | $\dfrac{K\,N_t^{(c)}/N - 1}{K-1}$, $c$ = index of `correct_answer` | alignment with the **correct answer** | $N_t^{(c)}$ — the correct-answer headcount |
| `m_order` | $\dfrac{K \max_k p_k - 1}{K-1}$ | how **ordered/consensual** the population is, regardless of *which* option won | $\max_k N_t^{(k)}$ — the largest headcount |

For a cell's fixed $N$ and $K$, $n \mapsto \frac{Kn/N - 1}{K-1}$ is affine and strictly increasing,
so each mapping above is a **bijection onto the observed values**. Two events share an order
parameter value iff they share the corresponding integer, which means the contingency tables — and
therefore every entropy, MI and CMI built from them — are *identical* to what the floats would
give. The integer is used because it is an exact hashable category; the float is not. This is the
same encoding rule already stated for $Z_t$ in [§3.2](#32-the-per-event-variables).

The integers appear on every event as `Mtruth_t`/`Mtruth_t1` and `Morder_t`/`Morder_t1` in
`event_metrics.csv`, beside the existing `Z_t`/`Z_t1`. Write $M^{\text{ctrl}}_t = Z_t$,
$M^{\text{truth}}_t = N_t^{(c)}$ and $M^{\text{order}}_t = \max_k N_t^{(k)}$ for the rest of this
section.

`m_order` is the one that is genuinely new in kind. `m_ctrl` and `m_truth` are single coordinates
of $N_t$; `m_order` is **invariant under relabelling the options**. It cannot tell you *which* way
the population went, only *how far it has collapsed onto something*. That makes it the statistic to
read when the question is "does the controller create consensus" rather than "does the controller
win".

### 12.3 The six statistics, as formulas

Three sensing channels, in the pattern of [§5](#5-sensing_mi-worked-in-full):

| Statistic | Formula | The question it answers |
| --- | --- | --- |
| `sensing_mi_m_ctrl` | $I\big(M^{\text{ctrl}}_t \,;\, Y_t\big)$ | How much does the peek tell the controller about **its own objective**? |
| `sensing_mi_m_truth` | $I\big(M^{\text{truth}}_t \,;\, Y_t\big)$ | How much does the peek tell it about **how close the population is to being right**? |
| `sensing_mi_m_order` | $I\big(M^{\text{order}}_t \,;\, Y_t\big)$ | How much does the peek tell it about **how converged the population is**? |

Three actuation channels, in the pattern of [§6](#6-the-three-actuation-cmis):

| Statistic | Formula | The question it answers |
| --- | --- | --- |
| `m_ctrl_actuation_cmi` | $I\big(U_t \,;\, M^{\text{ctrl}}_{t+1} \mid M^{\text{ctrl}}_t\big)$ | Given the current alignment, does the action move **the controller's objective**? |
| `m_truth_actuation_cmi` | $I\big(U_t \,;\, M^{\text{truth}}_{t+1} \mid M^{\text{truth}}_t\big)$ | …does it move the population **toward or away from the truth**? |
| `m_order_actuation_cmi` | $I\big(U_t \,;\, M^{\text{order}}_{t+1} \mid M^{\text{order}}_t\big)$ | …does it change **how much consensus** there is? |

Everything else is unchanged: same plug-in estimators ([§4](#4-the-estimators-mathematically)),
same `unsmoothed` headline variant, same episode bootstrap ([§9](#9-episode-bootstrap-how-precise-is-that)),
same within-episode temporal null ([§10](#10-temporal-nulls-how-much-of-that-is-noise)) — $Y_t$
permuted for the sensing three, $U_t$ for the actuation three.

### 12.4 Three exact relations, and two aliases

Read these before treating the six as six independent results.

**1. The sensing projections are bounded by `sensing_mi`.** Each $M_t$ is a deterministic function
of $N_t$, so the data-processing inequality gives, with **no exceptions**:

$$\max\big\{\text{sensing\_mi\_m\_ctrl},\; \text{sensing\_mi\_m\_truth},\; \text{sensing\_mi\_m\_order}\big\} \;\le\; \text{sensing\_mi}$$

A projection strictly below `sensing_mi` means the sensor resolves detail of $N_t$ that that
particular order parameter throws away. That gap is informative, not wasteful.

**2. `m_ctrl_actuation_cmi` *is* `target_actuation_cmi`.** $Z_t$ is defined as the target
headcount, and `m_ctrl` is one-to-one with the target headcount, so the two statistics are the same
estimate computed twice under two names. It is kept so the order-parameter family is complete and
so a config that asks for the order parameters gets all three without having to know the microstate
naming. **Do not report both as if they were corroborating evidence.**

> Their *nulls* and *bootstrap CIs* will differ slightly, and that is not an inconsistency. Seeding
> is per statistic index ([§13](#13-configuring-and-recomputing)): bootstrap draws use
> `seed + name_index` and permutation $p$ uses `seed + 10000 * (index + 1) + p`, so two identically
> defined statistics at different positions in `analysis.estimators` draw different random numbers.
> In the run below they land at 0.0693 and 0.0708 for the same 0.0623 estimate — Monte-Carlo
> spread, nothing more.

**3. `m_truth` collapses onto `m_ctrl` whenever the controller targets the truth.** With
`control.options.target: correct`, $c = z$ and the truth pair is numerically identical to the ctrl
pair. The two separate only under a config where the controller promotes something other than the
correct answer — which is precisely the configuration in which "does the controller help or hurt
accuracy" becomes a real question. **In a `target: correct` run, treat `m_truth` as a duplicate.**

### 12.5 What this looks like on real data

The worked example in §§5–11 is the v1 5-episode run, whose directory no longer exists. The numbers
below are therefore from a **different, smaller run** —
`hidden-bench-imitation-reasoning-control-10-v2-20260840`, **3 episodes, 90 controlled events**,
$H(U_t) = 0.699$ bits (17 advocate, 73 no-op), target `North Hill` = correct answer. Do not mix
them with the §11 table; they are here to show the *shape* of the result, not to restate it.

**Sensing.** All estimates unsmoothed, in bits, against $H(Y_t) = 2.183$:

| Statistic | $H(X)$ | Estimate | $H(X \mid Y_t)$ | Share of $H(X)$ resolved |
| --- | ---: | ---: | ---: | ---: |
| `sensing_mi` ($X = N_t$) | 3.011 | **1.339** | 1.672 | 44% |
| `sensing_mi_m_ctrl` | 2.115 | **0.993** | 1.123 | 47% |
| `sensing_mi_m_truth` | 2.115 | **0.993** | 1.123 | 47% |
| `sensing_mi_m_order` | 1.517 | **0.773** | 0.744 | 51% |

Relation 1 holds as it must ($0.993, 0.773 \le 1.339$), and `m_truth` duplicates `m_ctrl` as
predicted by relation 3. The interesting column is the last one: the coarser the projection, the
**larger the fraction of it the sensor resolves** — 44% of the microstate but 51% of the consensus
level. A 2-of-4 peek is a poor microscope and a decent consensus meter.

**Actuation**, with the conditioning-slice audit of [§8](#8-the-slice-collapse-problem) recomputed
for each conditioning variable:

| Statistic | Conditioning | Slices | Events in single-action slices | Ceiling $H(U_t\mid\cdot)\le$ | Estimate |
| --- | --- | ---: | ---: | ---: | ---: |
| `population_actuation_cmi` | $N_t$ | 12 | 47/90 = **52%** | 0.478 | 0.1415 |
| `focal_actuation_cmi` | $(X^f_t, N_t)$ | 20 | 60/90 = **67%** | 0.333 | 0.0764 |
| `m_ctrl_actuation_cmi` = `target_actuation_cmi` | $M^{\text{ctrl}}_t$ | 5 | 45/90 = **50%** | 0.500 | 0.0623 |
| `m_truth_actuation_cmi` | $M^{\text{truth}}_t$ | 5 | 45/90 = **50%** | 0.500 | 0.0623 |
| `m_order_actuation_cmi` | $M^{\text{order}}_t$ | 3 | 26/90 = **29%** | 0.711 | 0.0127 |

**This is the payoff and the catch in one table.** `m_order` conditioning nearly halves the inert
share — from 52% under $N_t$ to 29% — and raises the measurement ceiling from 0.478 to 0.711 bits.
It is the only conditioning in the whole file under which a majority of events are informationally
live. And it still returns the *smallest* estimate of the five, because on this data the action
genuinely does not predict the consensus level once you know the consensus level.

Worked slice by slice, as in [§7](#7-target_actuation_cmi-worked-in-full):

| $M^{\text{order}}_t$ | $n$ | $p(z)$ | $U_t$ | $n$ | $M^{\text{order}}_{t+1}$ | $I$ in slice | contribution |
| ---: | ---: | ---: | --- | ---: | --- | ---: | ---: |
| 2 | 43 | 0.478 | `ADVOCATE_Z` | 14 | `{2: 12, 3: 2}` | 0.00978 | 0.00467 |
| | | | `NO_OP` | 29 | `{2: 22, 3: 7}` | | |
| 3 | 21 | 0.233 | `ADVOCATE_Z` | 3 | `{2: 1, 3: 2}` | 0.03451 | 0.00805 |
| | | | `NO_OP` | 18 | `{2: 5, 3: 10, 4: 3}` | | |
| 4 | 26 | 0.289 | `ADVOCATE_Z` | **0** | — | 0.00000 | 0.00000 |
| | | | `NO_OP` | 26 | `{4: 26}` | | |
| | | | | | | **total** | **0.01272** |

which is `m_order_actuation_cmi_unsmoothed` exactly. The absorbing slice is still there — at full
consensus the sensor always reads the target as a majority, the controller never advocates, and 26
events contribute nothing — but it is 29% of the sample rather than 87%.

### 12.6 What the coarser conditioning costs

Coarsening the conditioning is not a free improvement, and the doc would be lying if it stopped at
§12.5. **Conditioning on less means controlling for less.**

$I(U_t; M_{t+1} \mid M_t)$ holds only the order parameter fixed. Two events with the same
$M^{\text{order}}_t = 2$ can sit in genuinely different populations — $(2,1,1)$ and $(1,1,2)$ both
have a largest count of 2 — and if the controller's action correlates with *which* of those it is,
that correlation is now free to show up as apparent actuation information. Under $N_t$ conditioning
it could not. So:

- **The microstate CMIs are the causally cleaner statistics.** `population_actuation_cmi` remains
  the reference answer to "does the action move the population"; it is starved, not confounded.
- **The order-parameter CMIs are the better-powered statistics.** They have more live events per
  slice and a higher ceiling, at the price of admitting within-slice confounding.
- **They are diagnostic read together, not substitutes.** A projection that comes out well above
  its microstate counterpart is a flag to check whether residual within-slice structure is doing
  the work, not a stronger result.

Two smaller caveats carry over unchanged from earlier sections:

- **The alphabet is still the observed alphabet** ([§4.4](#44-the-alphabet-is-the-observed-alphabet)).
  `m_order` visited only $\{2,3,4\}$ here; a run that reaches a 1-1-1-1 split adds a level and
  raises $H(M^{\text{order}})$. Check the level counts before comparing across runs.
- **The nulls remain structurally denser than the data** ([§10](#10-temporal-nulls-how-much-of-that-is-noise)).
  Permuting $U_t$ scatters both actions into every slice, and it does so for the coarse
  conditioning too. All three order-parameter actuation estimates sit at or below their nulls in
  this run, and that reads the same way it did in §10: *no evidence of measurable actuation
  information*, not *a negative effect*.

### 12.7 Which of the six to actually read

| If the question is | Read | Ignore |
| --- | --- | --- |
| Can the controller see its own objective? | `sensing_mi_m_ctrl` against `sensing_mi` | — |
| Can it see whether the crowd is converging? | `sensing_mi_m_order` | — |
| Does advocating move the objective? | `m_ctrl_actuation_cmi` **or** `target_actuation_cmi`, once | the other name |
| Does advocating help or hurt accuracy? | `m_truth_actuation_cmi` — **only** if `control.options.target` is not `correct` | it, otherwise |
| Does advocating change consensus? | `m_order_actuation_cmi`, with §12.6 in mind | — |
| Which *direction* any of this went | `advocacy_delta_m_ctrl`, `advocacy_delta_m_truth`, `target_adoption_lift` in `cell_summaries.csv` | all six — MI is unsigned |

The last row is the one people forget. Every statistic in this file is symmetric and label-free:
`m_order_actuation_cmi` cannot tell you whether advocating built consensus or destroyed it. The
signed `delta_m_*` metrics are the only thing in the pipeline that can.

---

## 13. Configuring and recomputing

The relevant block, from the example config:

```yaml
analysis:
  enabled: true
  estimators:                    # any subset; all ten by default
    - sensing_mi                 # the microstate channels, §§5-8
    - population_actuation_cmi
    - target_actuation_cmi
    - focal_actuation_cmi
    - sensing_mi_m_ctrl          # the order-parameter projections, §12
    - sensing_mi_m_truth
    - sensing_mi_m_order
    - m_ctrl_actuation_cmi
    - m_truth_actuation_cmi
    - m_order_actuation_cmi
  options:
    bootstrap_resamples: 1000    # episode-level resampling (§9)
    null_permutations: 1000      # within-episode permutations (§10)
    confidence: 0.95
    seed: 20260813
  comet_export: true             # upload information_estimates.md; needs logging.comet: true
execution:
  repetitions: 5                 # EPISODES — the sample size of every CI in the file
control:
  mechanism: threshold_target    # `none` produces no sensing/actuation MI at all
  options:
    sensor_sample_size: 2
    threshold: 0.5
```

Because the analysis reads only persisted trajectories, **everything in this document can be
recomputed from a finished run directory without re-running a single episode**:

```bash
conda run -n MA-CC --no-capture-output \
  python -m mas_cc.cli.main analysis hidden-bench-imitation \
  --run-dir results/hidden_bench_imitation/hidden-bench-imitation-reasoning-control-10/hidden-bench-imitation-reasoning-control-10-20260840 \
  --bootstrap-resamples 1000 \
  --null-permutations 1000
```

Seeding is deterministic and per statistic: bootstrap draws use `seed + name_index`, and null
permutation $p$ of statistic $i$ uses `seed + 10000 * (i + 1) + p`. Re-running with the same seed
reproduces the file byte for byte; changing the *order* of `analysis.estimators` changes
`name_index` and therefore the bootstrap draws, though not the estimates themselves.

---

## 14. How these differ from the sweep metrics

This is the section to read before putting a number from this file next to a number from
`sweep_metrics.json`. **They are not comparable**, and the reason is not the arithmetic — the
entropy, MI, and CMI functions are literally the same code in
[`analysis/estimators.py`](../../src/mas_cc/analysis/estimators.py). The difference is what plays
the role of the channel input, and it changes everything downstream.

### 14.1 The one difference that causes all the others

| | Sweep metrics (`terminal_mi`, `lagged_cmi`) | Imitation metrics (this file) |
| --- | --- | --- |
| **Channel input** | $C$, the swept grid axis | $U_t$, the controller's action |
| **How the input is generated** | **Exogenously assigned** by the experiment design, once per episode | **Endogenously produced** by a feedback policy, once per event |
| **Input ↔ conditioning state** | Independent by construction — $p(c)$ is the design distribution | **Strongly dependent** — $U_t$ is a deterministic function of a noisy read of the state |

In the sweep design, $C$ is randomized before the episode begins, so every conditioning slice
receives all levels of $C$ and every slice contributes to the CMI. In the imitation design, the
controller *chooses* $U_t$ by looking at the state, so conditioning on the state removes the
action's variation — the slice collapse of [§8](#8-the-slice-collapse-problem). A sweep CMI is an
interventional quantity by design; an imitation actuation CMI is an observational one, and carries
all the usual hazards of measuring a controlled system from its own control signal.

### 14.2 Everything that follows from it

| Aspect | Sweep metrics | Imitation metrics |
| --- | --- | --- |
| **Sampling unit** | Episode (terminal MI) or within-episode round pair (lagged CMI) | Interaction **event**, 332 of them across 5 episodes |
| **When computed** | Live, at every cell completion; a partial grid yields a partial estimate | **Post-hoc only**, from finished `trajectory.jsonl` files |
| **Observed variable** | `macrostate` — the $\arg\max$ option label, one letter per round | Full **count vectors** $N_t$, plus sensor, target-count, and focal projections |
| **Headline variant** | `_estimate` = **Jeffreys**-smoothed | **`unsmoothed`** — smoothing is a dominant fiction on these sparse conditional tables ([§4.5](#45-the-three-variants-and-why-unsmoothed-is-the-headline-here)) |
| **Alphabet handling** | `_levels` **unions across all cells** so the alphabet is stable as the grid fills | Observed levels **within the cell only**; the alphabet grows with the sample |
| **Uncertainty** | None on the estimate itself; the null band is the only spread | **Episode bootstrap CI** on every estimate |
| **Null hypothesis** | **Label shuffle** on the terminal table — destroys $C \leftrightarrow O$ association, holds both marginals fixed | **Within-episode temporal permutation** of $U_t$ or $Y_t$ — destroys timing, and also destroys the policy's state-coupling |
| **Ground truth** | Available for synthetic games in closed form (`mi_ground_truth_gap`) | **None exists** — the reasoning dynamics are an LLM, not a transition matrix |
| **Scope** | One number per **grid** | One row per **cell**, never pooled across cells |

### 14.3 Practical consequences

- **Never compare magnitudes across the two files.** `lagged_cmi` at 0.23 bits and
  `population_actuation_cmi` at 0.03 bits are not "more" and "less" control. Different inputs,
  different outcome variables, different alphabets, different smoothing.
- **The nulls mean opposite things.** A sweep estimate below its null band means "no effect
  detected". An imitation actuation estimate below its null band **may** mean that, or may mean the
  null was structurally denser than the data ([§10](#10-temporal-nulls-how-much-of-that-is-noise)).
  Check the single-action-slice share before concluding.
- **Sample-size levers differ.** For the sweep terminal MI, `execution.repetitions` is the sample
  size. Here `repetitions` sets the *bootstrap* precision, but the *estimate's* effective sample is
  the number of events in multi-action conditioning slices — which more episodes of the same
  deterministic policy scale only proportionally. Randomizing the controller changes it
  qualitatively.
- **Only `sensing_mi` behaves like an ordinary MI.** It has no conditioning, no policy coupling,
  and no degeneracy failure mode. It is the one number in this file that can be read at face value
  — subject to its alphabet caveat in [§5](#5-sensing_mi-worked-in-full).

There is also a third family in the repository: the **exact empowerment** quantities in
[`games/synthetic/effective_empowerment.py`](../../src/mas_cc/games/synthetic/effective_empowerment.py),
computed by linear algebra on a known transition matrix rather than estimated from samples. Those
are not comparable to either estimated family — see
[`sweep_mutual_information.md` §7.1](sweep_mutual_information.md#71-what-it-measures-and-why-the-conditioning-matters).

---

## 15. Where the code is

| Concern | File |
| --- | --- |
| Event adapter, all ten statistics, bootstrap, nulls, diagnostics, report writing | [`games/hidden_bench/imitation/analysis.py`](../../src/mas_cc/games/hidden_bench/imitation/analysis.py) |
| Order-parameter integer encodings (`ORDER_PARAMETER_COUNT_FIELDS`) | [`games/hidden_bench/imitation/analysis.py`](../../src/mas_cc/games/hidden_bench/imitation/analysis.py) |
| `m_ctrl`, `m_truth`, `m_order` definitions (`population_observables`) | [`games/hidden_bench/imitation/metrics.py`](../../src/mas_cc/games/hidden_bench/imitation/metrics.py) |
| Entropy, MI, CMI, Jeffreys, Miller–Madow | [`analysis/estimators.py`](../../src/mas_cc/analysis/estimators.py) |
| Population observables and event-level behavioral metrics | [`games/hidden_bench/imitation/metrics.py`](../../src/mas_cc/games/hidden_bench/imitation/metrics.py) |
| Sensor and threshold policy | [`games/hidden_bench/imitation/controller.py`](../../src/mas_cc/games/hidden_bench/imitation/controller.py) |
| State transitions and event records | [`games/hidden_bench/imitation/game.py`](../../src/mas_cc/games/hidden_bench/imitation/game.py) |
| Grid-level counterparts of the same estimators | [`metrics/sweep.py`](../../src/mas_cc/metrics/sweep.py), [`metrics/cell.py`](../../src/mas_cc/metrics/cell.py) |

**Related documents:** [`sweep_mutual_information.md`](sweep_mutual_information.md) (the grid-level
estimators, and the file this one is the counterpart to),
[`hidden_bench/hidden_bench_imitation.md`](hidden_bench/hidden_bench_imitation.md) (the game
semantics, event schema, control configuration reference, and full metric index), and
[`metrics.md`](metrics.md) (the per-round and per-episode metric reference).
