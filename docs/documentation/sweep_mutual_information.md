# Reading `sweep_metrics.json`: the mutual information and the lagged empowerment

This document explains, end to end, where every number in a run's `sweep_metrics.json` comes
from — what was counted, what table it was counted into, what formula was applied to that
table, and how to read the answer. Each section gives the **plain-terms** version first and
the **technical** version second.

The worked example throughout is a real run,
`results/synthetic_controlled_markov/synthetic-controlled-markov-empowerment-lagged/synthetic-controlled-markov-empowerment-lagged-20260806/`,
produced by
[`synthetic_controlled_markov_empowerment_lagged.yaml`](../../configs/runs/synthetic_games/synthetic_controlled_markov_empowerment_lagged.yaml).
Every arithmetic result below was recomputed from that run's files, so you can check any of it.

## Table of contents

1. [The file, and the one question it answers](#1-the-file-and-the-one-question-it-answers)
2. [The four tiers: from a round to a grid-level number](#2-the-four-tiers-from-a-round-to-a-grid-level-number)
3. [The macrostate: what actually gets counted](#3-the-macrostate-what-actually-gets-counted)
4. [The tables](#4-the-tables)
5. [The estimators, mathematically](#5-the-estimators-mathematically)
6. [`terminal_mi`, worked in full](#6-terminal_mi-worked-in-full)
7. [`lagged_cmi`: the lagged empowerment](#7-lagged_cmi-the-lagged-empowerment)
8. [`mi_null_band`: how much of that is noise](#8-mi_null_band-how-much-of-that-is-noise)
9. [`mi_ground_truth_gap`: the answer key](#9-mi_ground_truth_gap-the-answer-key)
10. [Reading this run's numbers](#10-reading-this-runs-numbers)
11. [Configuring and recomputing](#11-configuring-and-recomputing)
12. [Where the code is](#12-where-the-code-is)

---

## 1. The file, and the one question it answers

**In plain terms.** A grid run sweeps one setting — here, which action an exogenous controller
pushes the population toward — across a few levels, and runs many episodes at each level.
`sweep_metrics.json` answers: *knowing only which level was used, how much can you predict about
what the population did?* If the controller has no influence, the answer is 0 bits. If knowing
the level tells you exactly where the population ended up, the answer is $\log_2(\text{number of
levels})$ bits. Everything in the file is a variation on that one question: measured at the end
of the episode (`terminal_mi`), measured $h$ rounds ahead from wherever the population currently
is (`lagged_cmi`), and two sanity checks on both (`mi_null_band`, `mi_ground_truth_gap`).

**Technically.** The grid *is* a communication channel. The swept axis is the channel input $C$
(the condition variable), the observed macrostate is the output, and the design distribution
$p(c)$ is read off the actual per-cell episode counts rather than assumed uniform — so a resumed
or partially failed sweep with unbalanced cells still gets an honest answer. The file is written
by [`aggregation.py:161`](../../src/mas_cc/experiments/aggregation.py#L161), refreshed on **every
cell completion**, not once at the end, so a partial estimate appears early and tightens as the
grid fills.

The run used here has one axis with three levels:

| Cell | `game.options.control_value` | Meaning |
| --- | --- | --- |
| `cell-0000` | 0 | controller pushes the population toward action **Q** |
| `cell-0001` | 1 | controller pushes toward action **M** |
| `cell-0002` | 2 | `null` — a genuine **do-nothing** arm |

60 episodes per cell, 60 rounds per episode, so $3 \times 60 = 180$ episodes in total.

---

## 2. The four tiers: from a round to a grid-level number

There are four metric tiers in the codebase, and the mutual information lives in the last two.
Nothing at the sweep tier ever opens an episode file — it reads only the small count tables each
cell wrote when it finished. That is what makes it cheap enough to re-run continuously.

| Tier | Emits | Runs when | Class |
| --- | --- | --- | --- |
| `StreamingMetric` | one value per **round** | during the episode | [`metrics/base.py`](../../src/mas_cc/metrics/base.py) |
| `FinalMetric` | one value per **episode** | at episode end | [`metrics/base.py`](../../src/mas_cc/metrics/base.py) |
| `AggregateMetric` | one value per **cell** | at cell completion | [`metrics/cell.py`](../../src/mas_cc/metrics/cell.py) |
| `SweepMetric` | one value per **grid** | on every cell completion | [`metrics/sweep.py`](../../src/mas_cc/metrics/sweep.py) |

```text
metrics/streaming.csv          per-round option shares, written by each worker
        |
        |  _macrostate_series()          dominant option label, per round
        v
cells/cell-000N/aggregate.json  counts: terminal_outcome, macrostate_transition_h<h>
        |
        |  SweepMetric.compute()          stack cells into one array, estimate
        v
sweep_metrics.json              terminal_mi_*, lagged_cmi_*, null band, gap
```

The consequence worth remembering: **a grid killed at 80% yields the correct MI for the 80% that
finished**, because each cell's contribution outlives the cell.

---

## 3. The macrostate: what actually gets counted

**In plain terms.** Every round, each agent is standing on some action. The *macrostate* is
simply "which action is currently the most popular." For a two-action game that is a single
letter per round — `Q` or `M` — so one episode becomes a string like `MMMQQQQQ…`. That string is
all the mutual information ever sees.

**Technically.** [`_macrostate_series`](../../src/mas_cc/metrics/cell.py#L57) reads the recorded
`population_action_share_per_option` metric — the fraction of the population standing on each
option, counting each agent's most recent choice — and takes the $\arg\max$ per round:

$$M_t = \varphi(x_t) = \arg\max_{a} \; s_a(t), \qquad s_a(t) = \frac{\#\{\text{agents last playing } a \text{ at round } t\}}{N}$$

Two details that are load-bearing:

- **Ties break to the first declared option.** Python's `max` returns the first maximal element
  in iteration order and the labels are in declared-option order, which is exactly what the
  offline pipeline's `idxmax` does over rows written in that same order. Any other tie rule would
  make the live grid MI disagree with `analysis/pipeline.py`.
- **This is a coarse-graining, and it is lossy on purpose.** The microstate is which of $2^N$
  configurations the population is in; the macrostate is one of 2 labels. The information you
  measure is information about the *coarse* variable. That is the quantity the closed form in
  [`exact.py`](../../src/mas_cc/games/synthetic/exact.py) also computes, so the two are comparable.

`metrics.enabled: true` is therefore mandatory for any of this. Turn it off and the empowerment
has no input at all.

---

## 4. The tables

[`MacrostateCounts`](../../src/mas_cc/metrics/cell.py#L236) turns each cell's ~60 macrostate
strings into two kinds of count table. It is the cell's *sufficient statistic* — not a curve, not
meant to be looked at directly, just everything the sweep tier needs.

### 4.1 `terminal_outcome` — one count per episode

Take the **last** letter of each episode's macrostate string; tally them.

```text
cell-0000 (push to Q):   {"M": 1,  "Q": 59}     60 episodes
cell-0001 (push to M):   {"M": 56, "Q": 4}      60 episodes
cell-0002 (do nothing):  {"M": 23, "Q": 37}     60 episodes
```

**The sampling unit here is the episode, not the round.** Rounds within one episode are heavily
correlated and buy no independent samples for $I(C;O)$; that is why `execution.repetitions`, not
`game.horizon`, is what tightens this estimate. 180 episodes → $n = 180$, which is exactly
`terminal_mi_episodes` in the output file.

### 4.2 `macrostate_transition_h<h>` — one count per within-episode round pair

For each horizon $h$ in `aggregation.horizons`, slide a window of length $h$ over each episode's
string and tally the `"<current>><future>"` pairs:

```text
episode: M M Q Q Q      h=1  ->  M>M, M>Q, Q>Q, Q>Q
```

Pairs **never cross an episode boundary** — that is what makes them transitions rather than
artifacts of concatenation. An episode of $T$ rounds contributes $T-h$ pairs, so this run's table
totals are $180 \times (60 - h)$:

| $h$ | pairs |
| --- | --- |
| 1 | 10,620 |
| 2 | 10,440 |
| 5 | 9,900 |
| 10 | 9,000 |

Here are the actual $h=1$ tables, as $2\times 2$ matrices with rows $= M_t$ and columns
$= M_{t+1}$, in the order `[M, Q]`:

| Cell | `[[M>M, M>Q], [Q>M, Q>Q]]` | $P(M_{t+1}=Q \mid M_t=M)$ | $P(M_{t+1}=Q \mid M_t=Q)$ |
| --- | --- | --- | --- |
| `cell-0000` (→Q) | `[[27, 31], [17, 3465]]` | 0.534 | 0.995 |
| `cell-0001` (→M) | `[[3205, 89], [112, 134]]` | 0.027 | 0.545 |
| `cell-0002` (none) | `[[1144, 144], [145, 2107]]` | 0.112 | 0.936 |

> **The horizon list lives in one place.** `aggregation.horizons` is read by *both*
> `create_cell_metrics` (which builds the tables) and `create_sweep_metrics` (which estimates from
> them). Add a lag there and both follow. A cell built at $h=1$ against a sweep asking for $h=5$
> would report `NaN` with nothing to indicate why — hence the single source.

---

## 5. The estimators, mathematically

All three estimators are **plug-in** (direct-counting) estimators over finite discrete alphabets,
in **bits**, and they all live in
[`analysis/estimators.py`](../../src/mas_cc/analysis/estimators.py).

### 5.1 Entropy

For a count table $n$ with total $N = \sum n$, the plug-in entropy is

$$\hat H(n) = -\sum_{i \,:\, n_i > 0} \frac{n_i}{N} \log_2 \frac{n_i}{N}$$

### 5.2 Mutual information

From a two-dimensional contingency table $n_{cy}$ (rows $=$ condition, columns $=$ outcome), MI
is computed as a sum of three entropies of *marginalizations of the same table*:

$$\hat I(C;Y) = \hat H(n_{c\cdot}) + \hat H(n_{\cdot y}) - \hat H(n_{cy})$$

which is the standard identity $I(X;Y) = H(X) + H(Y) - H(X,Y)$. See
[`_mi_from_counts`](../../src/mas_cc/analysis/estimators.py#L39).

### 5.3 Conditional mutual information

From a three-dimensional table with axes ordered $(X, Z, Y)$:

$$\hat I(X;Y \mid Z) = \hat H(X,Z) + \hat H(Y,Z) - \hat H(Z) - \hat H(X,Y,Z)$$

Each term is again an entropy of a marginal sum of the same array — `counts.sum(axis=2)`,
`counts.sum(axis=0).T`, `counts.sum(axis=(0,2))`, and the full array. See
[`_cmi_from_counts`](../../src/mas_cc/analysis/estimators.py#L84).

Equivalently, in the form that makes the interpretation obvious:

$$I(X;Y \mid Z) = \sum_z p(z) \; \underbrace{\sum_{x,y} p(x,y \mid z) \log_2 \frac{p(x,y \mid z)}{p(x \mid z)\,p(y \mid z)}}_{\text{MI inside the slice } Z=z}$$

— an average, over the states you actually visit, of the mutual information *within* that state.

### 5.4 The three variants every estimate reports

A plug-in MI is **biased upward**: it is positive in expectation even when $X$ and $Y$ are
independent, because finite samples produce spurious association. Three numbers are therefore
reported side by side rather than one being silently chosen:

| Suffix | What it is | Formula |
| --- | --- | --- |
| `_unsmoothed` | the raw plug-in estimate | $\hat I(n)$ |
| `_estimate` | **Jeffreys-smoothed**, the headline number | $\hat I(n + \tfrac12)$ |
| `_miller_madow` | first-order bias correction | $\hat I(n)$ with $\hat H \mathrel{+}= \frac{K-1}{2N\ln 2}$ per term |

- **Jeffreys** adds $\tfrac12$ to *every* cell of the table before estimating — the Jeffreys
  ($\mathrm{Beta}(\tfrac12,\tfrac12)$ / $\mathrm{Dirichlet}(\tfrac12)$) prior. It pulls the
  estimate toward the independent case and stops a zero cell from claiming certainty. This is the
  number the pipeline treats as the answer.
- **Miller–Madow** instead adds $\frac{K-1}{2N\ln 2}$ to each entropy, with $K$ the number of
  *occupied* cells — an analytic correction for the entropy's downward bias, which translates into
  a downward correction on MI.

They disagree; that disagreement is information. In this run
$\hat I_{\text{unsm}} = 0.5124$, $\hat I_{\text{MM}} = 0.5044$, $\hat I_{\text{Jeff}} = 0.4887$
— a spread of ~0.024 bits, which is a reasonable read on the estimator uncertainty from
smoothing alone.

### 5.5 The alphabet is held fixed across cells

[`_levels`](../../src/mas_cc/metrics/sweep.py#L30) unions the keys seen across **all** cells, so a
cell that never observed outcome `M` still gets a zero column for it. Without that, the alphabet
would change as the grid fills and the MI would move for reasons unrelated to the dynamics.

Fewer than 2 populated cells or fewer than 2 outcome levels returns `NaN`, not `0.0`: one
condition level carries no information *by construction* and a single outcome has no entropy to
share. Both are "not yet", not "zero bits."

---

## 6. `terminal_mi`, worked in full

**In plain terms.** Stack the three `terminal_outcome` tallies into one 3-row table and ask how
much knowing the row tells you about the column.

**The table** — rows are cells, columns are outcomes sorted alphabetically `[M, Q]`:

|  | M | Q | row total |
| --- | --- | --- | --- |
| `cell-0000` (→Q) | 1 | 59 | 60 |
| `cell-0001` (→M) | 56 | 4 | 60 |
| `cell-0002` (none) | 23 | 37 | 60 |
| **column total** | **80** | **100** | **180** |

**The arithmetic** (unsmoothed, to show the mechanism):

$$\hat H(C) = -3 \cdot \tfrac{60}{180}\log_2 \tfrac{60}{180} = \log_2 3 = 1.58496 \text{ bits}$$

$$\hat H(O) = -\tfrac{80}{180}\log_2\tfrac{80}{180} - \tfrac{100}{180}\log_2\tfrac{100}{180} = 0.99108 \text{ bits}$$

$$\hat H(C,O) = 2.06363 \text{ bits} \quad \text{(over the six cells } 1, 59, 56, 4, 23, 37\text{)}$$

$$\hat I(C;O) = 1.58496 + 0.99108 - 2.06363 = \mathbf{0.51240} \text{ bits}$$

which is `terminal_mi_unsmoothed`. Repeating the same three steps on the table with $\tfrac12$
added to all six entries gives `terminal_mi_estimate` $= 0.48870$, and the Miller–Madow variant
gives $0.50439$.

**How to read the magnitude.** The ceiling is $H(C) = \log_2 3 = 1.585$ bits (perfect
recoverability of the condition from the outcome). We get 0.489, i.e. ~31% of it. That is the
do-nothing arm's fault and it should be: `cell-0002` ends on Q 62% of the time and M 38% of the
time, so an observed outcome barely distinguishes it from the other two. The two *active* arms
are nearly deterministic in opposite directions — $P(Q \mid \text{→Q}) = 0.983$,
$P(Q \mid \text{→M}) = 0.067$ — and they are what the 0.489 bits are made of.

---

## 7. `lagged_cmi`: the lagged empowerment

### 7.1 What it measures, and why the conditioning matters

**In plain terms.** *Given where the population is standing right now, how much extra does
knowing the controller's setting tell you about where it will be $h$ rounds from now?* The
"given where it is right now" is the whole point. Without it the number would mostly measure the
trivial fact that a converged population stays where it already is — which is a property of
inertia, not of the controller.

**Technically**, this is quantity **3a** of
[`effective_empowerment.py`](../../src/mas_cc/games/synthetic/effective_empowerment.py):

$$\text{lagged\_cmi}(h) \;=\; I\!\left(C \;;\; M_{t+h} \,\middle|\, M_t\right)$$

with $C$ drawn **once per episode and held fixed** (`control_mode: episode_fixed`). That fixed-
per-episode draw is exactly what makes $C$ a sweepable condition and what makes this estimable
from episodes — which is why it can run inside the run itself.

> **This is not the empowerment-paper quantity.** That one, **3b**, is
> $\mathbb{E}\big[I(a_t ; s_* \mid s_t)\big]$ with $\tau \sim \mathrm{Geom}(1-\gamma)$; it needs
> `control_mode: per_round` and is **exact linear algebra** on the microstate chain rather than an
> estimate, so it comes from `mas-cc synthetic truth --config <file>` and runs no episodes. Do not
> compare a 3a number to a 3b number.

### 7.2 The table

For each $h$, [`LaggedConditionalMutualInformation`](../../src/mas_cc/metrics/sweep.py#L94) builds
one array with axes $(C, M_t, M_{t+h})$ — i.e. $(X, Z, Y)$, matching `_cmi_from_counts`'s
convention. Here that is $3 \times 2 \times 2$: each cell's `"M>Q" -> count` dict is split on `>`
and dropped into `counts[row, index[current], index[future]]`. Then $\hat I(X;Y\mid Z)$ from
§5.3 is applied to it directly.

### 7.3 The results, and the shape of the curve

| $h$ | `_estimate` (Jeffreys) | `_unsmoothed` | pairs |
| --- | --- | --- | --- |
| 1 | 0.05578 | 0.05572 | 10,620 |
| 2 | 0.09589 | 0.09596 | 10,440 |
| 5 | 0.16791 | 0.16845 | 9,900 |
| 10 | 0.23223 | 0.23303 | 9,000 |

**Read it as a curve, not as four numbers.** It rises monotonically, and that rise is the
signature of a control input that acts *slowly*:

- At $h=1$ the macrostate almost never moves. In `cell-0000` the population is on Q 98% of pair-
  starts and stays on Q with probability 0.995. $M_t$ already predicts $M_{t+1}$ almost perfectly,
  so there is very little left for $C$ to explain — **0.056 bits**.
- At $h=10$ the chain has had time to drift toward wherever its condition pushes it. Knowing
  $M_t$ no longer pins down $M_{t+10}$, and the residual uncertainty is precisely what $C$ fills
  in — **0.232 bits**. Compare `cell-0001` at $h=10$: $P(Q_{t+10}\mid M_t=M) = 0.052$ vs.
  `cell-0000`'s $1.000$. The conditional distributions have separated by condition, which is what
  the CMI is picking up.

The saturation ceiling is $H(C) = 1.585$ bits again, and the curve is nowhere near it — the
controller nudges, it does not dictate.

### 7.4 Two caveats you should carry with these numbers

1. **Correlated samples.** The 10,620 pairs at $h=1$ are *not* 10,620 independent observations —
   they come from 180 episodes, and consecutive pairs within an episode overlap. The effective
   sample size is much closer to 180. Do not read the large pair count as precision.
2. **Degrees of freedom grow with the macrostate.** The plug-in bias for the CMI scales as
   $|S|\,(|C|-1)\,(|S|-1)$, so at a finer macrostate the bias can exceed the effect being
   measured. That is exactly why `_unsmoothed` is reported *beside* `_estimate` rather than
   hidden: if the two diverge materially, the alphabet is too fine for the sample. Here they agree
   to three decimals at every horizon, which is the reassuring case.

---

## 8. `mi_null_band`: how much of that is noise

**In plain terms.** Even if the controller did absolutely nothing, the measured MI would come out
positive, just from random imbalance in a finite sample. This metric measures that floor by
destroying the real association and re-measuring, hundreds of times.

**The procedure**, from
[`MutualInformationNullBand`](../../src/mas_cc/metrics/sweep.py#L139) — a **label-shuffle
permutation test** on the terminal table:

1. Pool all 180 episode outcomes into one bag (here: 80 `M`, 100 `Q`).
2. Shuffle the bag, and deal it back out into rows of the **original cell sizes** (60/60/60).
3. Re-estimate $\hat I$ (Jeffreys) on the shuffled table.
4. Repeat `aggregation.null_permutations` times — 500 in this run — and report the mean and the
   95th percentile.

Both marginals — the design $p(c)$ and the outcome mix $p(o)$ — are held **exactly** fixed while
the association between them is destroyed. That is precisely the null hypothesis being tested.

| Scalar | Value |
| --- | --- |
| `terminal_mi_null_mean` | 0.00831 |
| `terminal_mi_null_p95` | 0.02362 |

**How to use it.** The decision rule is: if the estimate is not clearly above `null_p95`, there is
no effect to report. Here $0.489 \gg 0.024$ — the estimate is roughly **21× the noise ceiling**,
so the effect is unambiguous. Use 200 permutations to read a result, 1000 to quote one.

Note this null band covers the **terminal** statistic only; there is no permutation null for
`lagged_cmi` today. For the lagged numbers, the `_unsmoothed`-vs-`_estimate` spread from §7.4 is
the diagnostic you have.

---

## 9. `mi_ground_truth_gap`: the answer key

**In plain terms.** Because this is a *synthetic* game, we know the exact right answer without
running anything — the dynamics are equations we wrote. This metric prints the exact answer and
subtracts the estimate from it, so you can see whether the whole measurement machinery works.

**Technically**, [`grid_ground_truth`](../../src/mas_cc/experiments/aggregation.py#L171) calls
`sweep_ground_truth` on the resolved grid, which propagates the exact microstate transition
matrices $T_c$ forward, lumps them onto the macrostate, and evaluates $I(C;O)$ in closed form
under the design $p(c)$:

$$I(C;O) = H(O) - \sum_c p(c)\,H(O \mid c)$$

The answer is persisted to `sweep_ground_truth.json` so that a later re-aggregation still has it,
since it is derived from the resolved grid and the game object — neither of which a run directory
carries.

| Scalar | Value |
| --- | --- |
| `terminal_mi_ground_truth` | 0.56500 |
| `terminal_mi_estimate` | 0.48870 |
| `terminal_mi_gap` | **−0.07630** |

**Sign convention:** `gap = estimate − ground_truth`. Negative means the estimate came in *below*
the truth. Note that **both** estimates undershoot here — the unsmoothed one, 0.5124, has its own
gap of −0.053 — so the shortfall is not just Jeffreys shrinkage. Decomposing it:

- **−0.024 bits** is the Jeffreys smoothing, i.e. the distance from `_unsmoothed` to `_estimate`.
  That part is deliberate and by construction downward.
- **−0.053 bits** is the unsmoothed estimate's own deviation from the closed form. This is *not*
  plug-in bias: the analytic plug-in bias at these dimensions is $\frac{(|C|-1)(|O|-1)}{2 n \ln 2}
  = \frac{2}{2 \cdot 180 \cdot \ln 2} = +0.008$ bits, and it points the *other* way. A −0.053-bit
  residual over 180 episodes is within ordinary sampling variability for this statistic.

So the honest reading is "the estimate agrees with the closed form to about one standard error,"
not "smoothing explains the gap." If you want to shrink it, the lever is `execution.repetitions` —
the episode count is the sample size of this statistic. The exact plug-in-bias term is itself
reported as `terminal_plugin_bias` by `mas-cc synthetic empowerment`, so you can always check
whether a gap you are looking at is bias-sized or noise-sized.

**Three situations where this metric correctly reports nothing rather than a wrong number:**

- the game is not synthetic (no closed form exists);
- the grid has more than one axis — the closed form is defined for a single condition variable $C$;
- the closed form does not cover this particular sweep (e.g. `population_size` varies without
  `macrostate_bins` set, which would make the macrostate alphabet differ per condition and collapse
  $I(C;S)$ onto $H(C)$ as a pure support artifact).

In all three cases `ground_truth` is `None` and the metric emits nothing at all — **a missing
answer key must never be published as a zero gap.**

You can print the closed form without running any episodes:

```bash
mas-cc synthetic empowerment --config configs/runs/synthetic_games/synthetic_controlled_markov_empowerment_lagged.yaml \
       --condition control_value --values 0 1 2 --repetitions 60 --horizons 1 2 5 10
```

---

## 10. Reading this run's numbers

Put together, the file says one coherent thing:

| Reading | Evidence |
| --- | --- |
| The controller has a **real, large** effect on where the population ends up. | 0.489 bits vs. a 0.024-bit noise ceiling — 21×. |
| It uses about **31%** of the available channel capacity. | $0.489 / \log_2 3 = 0.31$. The do-nothing arm is nearly ambiguous by design, which caps this. |
| The effect is **slow-acting**, not instantaneous. | Lagged CMI rises 0.056 → 0.232 bits from $h=1$ to $h=10$ and is still climbing. |
| The estimator itself is **behaving**. | Gap to closed form is −0.076 bits, of which −0.024 is deliberate Jeffreys shrinkage and the rest is ~one standard error of sampling noise at 180 episodes. |
| The macrostate alphabet is **not too fine** for this sample. | Same point: `_estimate` ≈ `_unsmoothed` at every horizon. |

The one thing these numbers **cannot** tell you is which action the controller pushed toward —
mutual information is symmetric and label-free. For direction, read the per-cell
`terminal_outcome_fraction_*` scalars in each `cells/cell-000N/aggregate.json`.

---

## 11. Configuring and recomputing

The whole block, from the example config:

```yaml
metrics:
  enabled: true            # MANDATORY — the macrostate is rebuilt from
                           # population_action_share_per_option in streaming.csv
aggregation:
  cell_metrics: [...]      # macrostate_counts is added automatically when needed
  sweep_metrics:
    - terminal_mi
    - lagged_cmi
    - mi_null_band
    - mi_ground_truth_gap
  horizons: [1, 2, 5, 10]  # the lags h; read by BOTH tiers, one source of truth
  null_permutations: 500   # 200 to read a result, 1000 to quote one
execution:
  repetitions: 60          # episodes PER CELL — this, not game.horizon, is what
                           # tightens the terminal estimate
grid:
  game.options.control_value: [0, 1, 2]   # ONE axis, required for the ground-truth gap
```

Because the sweep tier reads only the per-cell count tables, **everything in this document can be
recomputed from a finished run directory without re-running a single episode** — with different
horizons, more permutations, or a corrected policy:

```bash
mas-cc experiment aggregate --run-dir results/.../synthetic-controlled-markov-empowerment-lagged-20260806
```

Caveat: changing `horizons` after the fact only works for horizons whose `macrostate_transition_h<h>`
table already exists in the cell files. A *new* horizon requires re-aggregating the cells from
their episode files, which `aggregate_grid_directory` does — it re-reads episodes and rebuilds the
count tables, then re-estimates.

---

## 12. Where the code is

| Concern | File |
| --- | --- |
| Macrostate extraction, per-cell count tables | [`metrics/cell.py`](../../src/mas_cc/metrics/cell.py) — `_macrostate_series`, `MacrostateCounts` |
| The four sweep metrics | [`metrics/sweep.py`](../../src/mas_cc/metrics/sweep.py) |
| Entropy, MI, CMI, Jeffreys, Miller–Madow | [`analysis/estimators.py`](../../src/mas_cc/analysis/estimators.py) |
| Writing `sweep_metrics.json`, wiring the ground truth | [`experiments/aggregation.py`](../../src/mas_cc/experiments/aggregation.py) |
| Exact linear algebra behind the closed form | [`games/synthetic/exact.py`](../../src/mas_cc/games/synthetic/exact.py), [`games/synthetic/empowerment.py`](../../src/mas_cc/games/synthetic/empowerment.py) |
| The two empowerment quantities, 3a vs. 3b | [`games/synthetic/effective_empowerment.py`](../../src/mas_cc/games/synthetic/effective_empowerment.py) |
| Offline, bootstrap-CI counterpart of the same estimates | [`analysis/pipeline.py`](../../src/mas_cc/analysis/pipeline.py) |

The live sweep metrics and the offline pipeline estimate the **same quantities from the same
estimators and must agree**. Where they could diverge — the macrostate tie rule — the cell tier
deliberately matches `analysis/reader.py`.

**Related documents:** [`metrics.md`](metrics.md) (the per-round and per-episode metric reference),
[`05082026_empowerment_ground_truth.md`](../tdd/architecture/05082026_empowerment_ground_truth.md)
(the ground-truth design doc), and
[`06082026_game3_empowerment_extension.md`](../tdd/architecture/06082026_game3_empowerment_extension.md).
