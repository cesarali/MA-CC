# Epistemic coarse-graining: recipe, and what worked

**2 October 2026.** Offline analysis of the archived `21-09-2026-full-vs-report-v1`
study. No simulations were run.

Companion to [`markov_test_report_2026-10-02.md`](markov_test_report_2026-10-02.md).
This note is the procedure plus an honest record of the three things that failed
and why.

## 0. Headline

| | |
|---|---|
| state variable | binned population mean of `P(ALLOCATION_0 \| agent's active facts)` |
| number of bins | **3 or 4** — the data cannot separate them |
| bin edges | **fixed a priori**, not fitted to the run |
| **split by persistence ρ** | **yes — mandatory** |
| Markov gap at ρ = 1.00 | **≈ 0**, the state is sufficient |
| Markov gap at ρ = 0.75 | **0.016 nats/step**, the state is *not* sufficient |
| the vote count as a state | worse on both counts |

The single most important finding is the last two rows: **the coarse-graining is
Markovian at ρ = 1.00 and is not at ρ = 0.75.** Pooling the two hides this.

## 1. Two clarifications that matter

### The 14,388 worlds are the *agent's* uncertainty, not the simulation's

task_003 has exactly one true hidden world,
`V = (3,1,1,2,2,1,1,1,2)`. That never changes, in any episode.

But an **agent** does not know `V`. It holds a handful of true facts and nothing
else. The question `P(ALLOCATION_0 | K)` asks:

> Among all hidden worlds that are consistent with the facts this agent holds,
> in what fraction does ALLOCATION_0 win?

The 14,388 are the **hypothetical worlds an agent could still be in** — the
worlds not yet ruled out by its evidence. They are the agent's epistemic state,
not alternative simulations. Concretely: 3⁹ = 19,683 assignments of nine values
in {1,2,3}, of which 14,388 have a unique winning allocation; the rest are ties
and are dropped.

The true world is always one of the survivors, because every fact in the game is
true of it. That is why `P(ALLOCATION_0) > 0` always, and why no set of true
facts can make a wrong answer certain.

So the measure is *"how close is this agent to being able to work out the
answer"*, computed exactly.

### Bin edges are fixed, visited states are not

Two different things get conflated here.

**The binning rule is fixed a priori.** Cuts at 0.33 (the uniform prior), 0.50
(a majority) and 0.75 are chosen for interpretability, not estimated from the
run. Apply them to any study and you get the same rule. A quantile scheme — cuts
at the empirical 25th/50th/75th percentiles — *would* be run-dependent, which is
one reason it is not recommended (§5).

**Which states get visited does depend on the run**, and sharply so:

| | M₀ | M₁ | M₂ | M₃ |
|---|---:|---:|---:|---:|
| ρ=0.75, silent | 0.05 | 0.76 | 0.19 | — |
| ρ=0.75, A2-target | 0.50 | 0.43 | 0.07 | — |
| ρ=1.00, silent | 0.01 | 0.23 | 0.72 | 0.04 |
| ρ=1.00, A2-target | 0.07 | 0.45 | 0.46 | 0.02 |

Same rule, four very different occupancies. M₃ is essentially unvisited at
ρ = 0.75. This is a property of the dynamics being measured, not of the
coarse-graining, and it is exactly what one wants a state variable to reveal.

## 2. The recipe

For each round of each episode:

1. **Read each agent's active fact set.** Field
   `active_fact_ids_by_agent_after` in the round records: the facts that agent
   can currently reason with, after persistence has acted. Typically 1-6 facts.

2. **Compute that agent's posterior.** `P(ALLOCATION_0 | K)` by filtering the
   14,388 valid worlds to those consistent with every fact in *K*, then counting
   the fraction won by ALLOCATION_0. Exact; no model, no sampling.

3. **Cache on `frozenset(K)`.** Across 497,520 agent-round observations there
   are only **147,443 distinct fact sets**. The theoretical space is 2⁴⁹ ≈
   5.6 × 10¹⁴ and is never enumerated — only what actually occurs is evaluated.

4. **Average over the 24 agents** to get the population epistemic level
   `e(t) ∈ [0,1]`.

5. **Bin it** with fixed cuts 0.33 / 0.50 / 0.75 → four states:

   | state | range | reading | share (pooled) |
   |---|---|---|---:|
   | M₀ | < 0.33 | believes *against* the truth | 27% |
   | M₁ | 0.33 – 0.50 | uninformed | 41% |
   | M₂ | 0.50 – 0.75 | leaning truth | 26% |
   | M₃ | > 0.75 | strongly truth | 6% |

6. **Fit transition kernels separately per (ρ, arm).** Never pool persistence
   levels — see §4.

Cost: 147,443 posterior evaluations, each a handful of bitmask operations over
14,388 worlds. Seconds.

## 3. Why the mean, and not the composition

A *composition* records how many agents sit in each bin, e.g. (6,10,7,1). The
number of such states is

```
C(24 + 4 - 1, 4 - 1) = C(27,3) = 2,925
```

far worse than the 325 states of the vote vector. Three bins would give
C(26,2) = 325 — exactly the vote vector's size, so no gain either.

With roughly 900 round-transitions per arm:

| state space | states | transition probabilities | observations each |
|---|---:|---:|---:|
| 4-bin composition | 2,925 | 8.6 M | hopeless |
| vote vector (current) | 325 | 105,625 | hopeless |
| **4-bin mean** | **4** | **16** | **~55** |

The scalar mean discards the shape of the distribution. That is the price, and it
is worth paying: sparsity is what made the earlier trajectory-KL estimates
unreliable.

## 4. What worked

### Splitting by persistence — and this is the main result

Fitting kernels separately per ρ, the within-arm Markov-order test (held-out
log-likelihood, episode-level splits, 12 seeds, 4 bins):

| split | arm | n rows | Markov gap (nats/step) |
|---|---|---:|---:|
| pooled ρ | silent | 3,600 | 0.0138 ± 0.0050 |
| pooled ρ | A2-target | 7,575 | 0.0168 ± 0.0030 |
| **ρ = 0.75** | silent | 1,800 | **0.0177 ± 0.0098** |
| **ρ = 0.75** | A2-target | 5,055 | **0.0155 ± 0.0027** |
| **ρ = 1.00** | silent | 1,800 | **−0.0050 ± 0.0029** |
| **ρ = 1.00** | A2-target | 2,520 | **0.0060 ± 0.0064** |

The gap is how much a second-order chain improves on a first-order one. Zero
means the current state is sufficient.

**At ρ = 1.00 the state is Markovian.** The silent arm's gap is *negative*
(−0.0050): adding history makes held-out prediction **worse**, which is the
signature of a genuinely first-order process — the extra parameters buy nothing
and cost precision. The controlled arm at ρ = 1.00 is +0.0060, within noise of
zero.

**At ρ = 0.75 it is not**, in either arm (0.016-0.018).

The mechanism is clear. At ρ = 1.00 nothing is forgotten, so an agent's fact set
only grows; the epistemic level is a monotone accumulator and its current value
summarises the whole history. At ρ = 0.75 facts are lost and reacquired
stochastically, so the same mean can arise from different fact sets with
different futures — and the mean alone cannot tell them apart.

**Consequence:** path KL, entropy production and mutual information computed on
this state are well founded at ρ = 1.00. At ρ = 0.75 they are approximations and
must carry the 0.016 nats/step residual.

### The epistemic state beats the vote count

In the controlled arm (pooled ρ, 4 bins):

| state | absolute fit (nats/step) | Markov gap |
|---|---:|---:|
| vote share V | −0.807 | 0.0450 |
| **epistemic M** | **−0.515** | **0.0184** |
| both, MV | −1.274 | 0.0144 |

M predicts better *and* is more nearly Markovian than the vote count, where it
matters. MV has the smallest gap but the worst absolute fit — the extra states
cost more in sparsity than they buy.

## 5. What did not work, and why

### Cross-arm kernel transfer — the premise is false

**Attempted:** fit the kernel on silent episodes, test it on the gate-OFF rounds
of controlled episodes, which "should" be passive.

**Result:** failed for every state variable. For M the "should match" case was
*worse* (TV 0.162) than the "should differ" calibration case (0.112) — no
discriminating power at all.

**Why:** a gate-OFF round inside a controlled episode is not passive in state.

| | agents holding ≥1 controller-pool fact |
|---|---:|
| silent episodes | 0.541 |
| A2 episodes, **gate OFF** | **0.900** |
| A2 episodes, gate ON | 0.899 |

Gate-OFF and gate-ON are identical in exposure. At finite ρ the controller's
facts persist in memory and keep being reactivated, so the population carries its
accumulated influence whether or not it posts this round.

**Lesson:** "the controller did not act this round" is not "the controller was
never here". Do not compare kernels across arms.

### Adding exposure to the state — positivity failure

**Attempted:** add "fraction of agents holding a controller-pool fact" as a
second state dimension, hoping it was the missing variable.

**Result:** worse. Transfer TV went 0.162 → 0.291 → 0.359 as dimensions were
added.

**Why:** silent episodes sit at exposure 0.54 and controlled ones at 0.90, so the
two arms occupy nearly disjoint regions. The transferred kernel has little data
in the states it is asked to predict, and extrapolates. Adding a variable that
*separates* the arms makes a cross-arm comparison less identified, not more.

### Tuning the number of bins — no effect

**Attempted:** sweep 2-6 bins, quantile and semantic schemes, hoping resolution
was the lever.

**Result:** across 12 episode splits, 3 / 4 / 5 bins are statistically
indistinguishable in five of six pairwise comparisons, and the one that "differs"
does so by 0.0019 nats/step.

| bins | silent | A2-target |
|---|---|---|
| 3 | 0.0121 ± 0.0043 | 0.0139 ± 0.0040 |
| 4 | 0.0138 ± 0.0052 | 0.0168 ± 0.0032 |
| 5 | 0.0120 ± 0.0049 | 0.0164 ± 0.0036 |

**Why it does not help:** the residual non-Markovianity is not a binning
artifact. It comes from the mean discarding *which* facts are held, and no
resolution of the mean recovers that.

**Methodological warning:** absolute log-likelihood is **not comparable across
bin counts**. It falls from −0.23 (2 bins) to −0.68 (6 bins) purely because more
states mean more entropy to predict. Only the *gap* is comparable, being a
difference within one state space. An earlier single-seed run appeared to favour
3 bins; across 12 seeds that disappeared.

## 6. Recommended procedure

1. Use the **population mean** of `P(A0 | K_j)`, **4 bins**, fixed cuts
   0.33 / 0.50 / 0.75. Three bins is equally defensible; four keeps M₀
   ("believes against the truth", 27% of observations) as its own state, which is
   where control plausibly acts.
2. **Fit every kernel separately per (ρ, arm).** Never pool persistence.
3. At **ρ = 1.00**, treat the chain as Markovian and compute path KL, entropy
   production and MI directly.
4. At **ρ = 0.75**, report the same quantities as approximations and attach the
   0.016 nats/step residual; check sensitivity against a second-order estimate.
5. **Never transfer a kernel between arms.** Compare separately-fitted kernels.
6. Do not use the composition, and do not use the vote count.

## 7. Caveats

- One task, one model, 60 initializations. Everything here is task_003 with
  `gpt-oss-120b`.
- Bin edges are fixed a priori and so are not fitted, but they were chosen after
  looking at the pooled distribution. A fully clean procedure would select them
  on episodes disjoint from the evaluation set.
- The order test compares two Dirichlet-smoothed estimators (α = 0.5). The prior
  affects absolute values, not the ranking.
- The A0-target arm is excluded throughout: its controller recommends
  ALLOCATION_0 while quoting ALLOCATION_2-favouring evidence, so it cannot
  inform a kernel describing controlled dynamics.
- The ρ = 1.00 result rests on 1,800 silent and 2,520 controlled rows. It is the
  most consequential claim here and deserves replication on the next batch.
