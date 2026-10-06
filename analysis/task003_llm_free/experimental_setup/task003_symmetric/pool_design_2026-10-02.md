# A symmetric controller and an epistemic coarse-graining

> **Status, 5 October 2026.** §2 (the two pools) is frozen as the controller
> pools of `task003-symmetric`. §1 is superseded: the agents were replaced so
> that 8 favour each allocation. §3 developed into `task003-nosolution`. See
> [`../README.md`](../README.md).

Design note, 2 October 2026. Companion to
[`../../task_and_facts/task_003_analysis.md`](../../task_and_facts/task_003_analysis.md),
which establishes the fact inventory this note builds on. Every number here is
exact arithmetic over task_003 or a measurement on the archived
`21-09-2026-full-vs-report-v1` study; no simulations were run.

## Why this is needed

The executed study admits controller facts by one rule:

```
not in the decisive set    AND    ΔP(false target) > 0
```

evaluated once for the **false** target and reused for both controller
configurations. Measured consequences:

- both targets draw from the **identical 24 facts**; the two pools differ by zero
  members, and fact selection never consults the target
- the controller can complete **0 of 7,694** minimal proofs, and its whole pool
  reaches only P(A0) = 0.308
- the four strongest pro-truth facts are in **nobody's** hands
- the remaining directional difference comes mostly from the **activation gate**:
  the truth controller senses its target already at 0.686 and fires 32% of
  rounds, the false controller senses 0.393 and fires 54%

So an arm labelled "steer toward ALLOCATION_0" recommends the truth while
quoting evidence curated to favour ALLOCATION_2, and acts 40% less often. The
comparison is confounded three ways: content, capability and dosage.

---

## 1. Agents' facts stay as they are

The 24 agents keep their frozen one-fact packets, including all six facts of the
decisive set. Rationale: those six are the population's **only** complete route —
all four minimal proofs available to the agents use decisive members, and
removing them drops the swarm from P(A0) = 1.000 to 0.737 with no path to
certainty. Keeping them preserves the one configuration in which "did the
controller help or hinder a swarm that could otherwise succeed?" is answerable.

This also keeps the two controller arms comparable: the population is identical
across arms, so any difference is attributable to the controller.

## 2. Two symmetric controller pools

### Constraints

1. **equal size**
2. **matched epistemic weight** — paired facts of near-identical |ΔP(target)|
3. **no decisive facts**
4. **disjoint** — no fact in both pools
5. **neither pool can prove its target**

Constraints 4 and 5 were not in the original brief but are required. Without 4,
a naive ΔP(target) > 0 rule puts **12 facts in both pools** — the ones that only
rule out ALLOCATION_1 — which dilutes the contrast. Without 5, the asymmetry
survives: strong A0 facts *are* proof components, so a magnitude-matched A0 pool
still reaches P(A0) = 1.000 while the A2 pool caps at 0.982.

Constraint 5 is enforced constructively: add A0 facts one at a time, rejecting
any addition that makes the pool solvable.

### The result

| | A0 pool | A2 pool |
|---|---:|---:|
| size | **12** | **12** |
| Σ ΔP(target) | **1.016** | **1.028** |
| overlap | **0** | |
| contains decisive facts | no | no |
| **can prove its target** | **no** | no |
| max reachable P(target) | 0.976 | 0.935 |
| full pool posted, P(target) | 0.963 | 0.886 |

Magnitude-matched pair by pair; **10 of 12 gaps are exactly 0.000**.

| A0 fact | ΔP(A0) | A2 fact | ΔP(A2) | gap |
|---|---:|---|---:|---:|
| `cf_x00_eq_3` | +0.252 | `cf_x05_eq_1` | +0.252 | 0.000 |
| `cf_x00_ge_x01` | +0.150 | `cf_x04_ge_x05` | +0.150 | 0.000 |
| `cf_x08_ge_2` | +0.114 | `cf_x01_ge_x05` | +0.116 | 0.002 |
| `cf_x01_le_2` | +0.114 | `cf_x05_le_2` | +0.114 | 0.000 |
| `cf_x00_ge_2` | +0.114 | `cf_x04_ge_2` | +0.114 | 0.000 |
| `cf_x02_le_x05` | +0.067 | `cf_x01_ge_x02` | +0.067 | 0.000 |
| `cf_x04_le_2` | +0.063 | `cf_x08_le_2` | +0.063 | 0.000 |
| `cf_x00_ge_x03` | +0.046 | `cf_x02_ge_x05` | +0.046 | 0.000 |
| `cf_x01_le_x02` | +0.046 | `cf_x03_le_x04` | +0.046 | 0.000 |
| `cf_x01_eq_x02` | +0.024 | `cf_x03_eq_x04` | +0.024 | 0.000 |
| `cf_x04_eq_2` | +0.014 | `cf_x02_eq_x05` | +0.024 | 0.010 |
| `cf_x06_le_x07` | +0.013 | `cf_x08_eq_2` | +0.014 | 0.001 |

Membership is stored in [`controller_pool_A0.json`](controller_pool_A0.json) and [`controller_pool_A2.json`](controller_pool_A2.json). These are the two controller pools of the frozen task003-symmetric setup (5 October 2026).

### The fourth confound: the activation gate

Pools fix content and capability. They do **not** fix dosage. The gate fires on
the sensed share of the controller's **own** target, so the truth controller is
quiet precisely because it is winning. This must be made target-independent —
a fixed schedule, or a gate driven by a signal that does not depend on which
target was assigned — or the arms differ in posts-per-episode and no efficiency
comparison is valid.

### Known limitation

Twelve facts is small against budgets of 6, 9, 15: at b = 15 the pool is
exhausted in one round. Either cap budgets at about 6, or relax constraint 4 to
reach roughly 17 per pool and accept some shared facts.

## 3. A second pair: agents without the decisive facts

Remove the six decisive facts from the agent packets, keep the symmetric pools.

| agents hold | P(A0) reachable |
|---|---:|
| all 21 facts | **1.000** |
| 15 facts, decisive removed | **0.737** |

The swarm moves from *can prove it* to *cannot, ever*. The question changes from
"can control beat an achievable proof?" to **"under irreducible uncertainty, does
control help or hurt?"** — arguably closer to any real deployment, since real
collectives rarely hold a complete proof.

Two checks before running it:

- can the agents' remaining 15 facts **plus the A0 pool** prove the answer? If
  so, the truth controller has a uniquely cooperative role worth isolating.
- does the **silent** baseline still converge on A0 at 0.737? If it collapses,
  the reference arm is lost and the contrast becomes uninterpretable.

Run this after the symmetric pools establish the baseline.

## 4. A two-allocation world

Not possible with the current generator: `enumerate_allocations` hard-requires
exactly three people, and an allocation is "who works alone", so three people
imply three allocations structurally.

Two routes:

- **restrict** — give every agent a fact eliminating ALLOCATION_1. Cheap, but it
  is a two-allocation game only in effect; the machinery still carries three.
- **rebuild** — a small 2-allocation generator, e.g. 2 people x 2 jobs, giving
  3^4 = 81 worlds. The posterior engine is already generic over
  (worlds, winner, facts); a new fact catalog and evidence wording are needed.

The rebuild is worth it. A binary task makes the command channel exactly one
bit, removes the third allocation that ΔP can leak into, and eliminates the
"dual-positive" facts that forced constraint 4 above. It is a genuinely new task
and needs its own validation and calibration run.

## 5. An epistemic coarse-graining

### The problem it addresses

A mean-field kernel on the **vote count** alone is not Markovian in this
archive: the passive kernel fitted from the silent arm does not match the one
fitted from gate-OFF rounds of controlled arms. Something outside the vote count
carries state, and the obvious candidate is agent **memory** — which the
controller demonstrably manipulates, since proof coverage rises 1.7-5.4x under
control.

A state variable built from memory may restore Markovianity, and would also
shrink the state space enough for trajectory-KL estimation to become feasible.

### The measure

For agent *j* with active fact set *K*, use the exact posterior

```
e_j(t) = P(ALLOCATION_0 | K_j(t))
```

computable in closed form from the same enumeration used throughout. It is
direction-aware and bounded.

An information-gain measure `log2(3) - H(p)` is **not** suitable: it is
direction-blind, maximal for certainty in ALLOCATION_2 as much as in
ALLOCATION_0.

### What the data actually look like

Measured over 28,800 agent-rounds sampled from the archive:

| statistic | P(A0 \| K) |
|---|---:|
| mean | 0.435 |
| median | 0.430 |
| 10th / 90th percentile | 0.191 / 0.708 |
| maximum | 1.000 |

| threshold | share of agent-rounds |
|---|---:|
| > 0.50 | 31.5% |
| > 0.66 | 14.1% |
| > 0.90 | **1.35%** |
| > 0.99 | **0.77%** |

**A 33/66/99 split does not work.** The top bin would hold 0.77% of
observations — far too sparse for transition estimation — and roughly a quarter
of the mass sits *below* 0.333, which that scheme has no bin for.

### Proposed states: four, not three

| state | range | reading | approx. share |
|---|---|---|---:|
| **M₋** | P(A0) < 0.33 | believes *against* the truth | 27% |
| **M₀** | 0.33 - 0.50 | uninformed / balanced | 41% |
| **M₊** | 0.50 - 0.75 | leaning truth | 26% |
| **M₊₊** | > 0.75 | strongly truth | 6% |

The important addition is **M₋**. A quarter of agent-rounds sit at below-prior
confidence in the truth; a 33/66/99 scheme would merge them with the uninformed,
and that distinction is probably where the controller's effect lives.

Five states is defensible if M₀ is split at the prior 0.3333 to separate "no
information" from "slightly informed". Do not go beyond five: sparsity is what
made the trajectory-KL estimates unreliable in the first place.

### Population state: use the mean, not the composition

A *composition* is how 24 identical agents are distributed across the bins —
how many sit in each, e.g. (6, 10, 7, 1). The number of such distributions is

```
C(24 + 4 - 1, 4 - 1) = C(27, 3) = 2,925
```

which is **worse** than the 325 states of the current vote vector, not better.
(Three bins would give C(26,2) = 325 — exactly the vote vector's size, so no gain
either.)

Use instead the **population mean** of e_j(t), binned into 4-5 levels. With
roughly 900 round-transitions per arm, a 5-state chain has 25 transition
probabilities and about 36 observations each — enough for a path KL. With 325 or
2,925 states it is hopeless, which is precisely why the earlier trajectory-KL
estimates failed their reliability checks.

### Testing whether it is actually Markovian

Memory alone may not be Markovian either: the loop runs memory -> vote ->
messages -> memory. Build two candidate state variables and test both:

- **S1** = binned mean epistemic level
- **S2** = (binned mean epistemic level, binned truth-vote share)

**The test**: estimate the passive transition kernel from the silent arm, then
check whether it predicts the gate-OFF rounds of controlled episodes. If the
state variable is sufficient, those rounds are passive by definition and the
kernels must agree. This is the same test that falsified the vote-count model.

**Which data.** Only the **silent** and **A2-target** arms. The A0-target arm's
dynamics are sound as recorded, but the configuration is the confounded one —
the controller recommends ALLOCATION_0 while quoting ALLOCATION_2-favouring
evidence — so it should not inform a kernel meant to describe controlled
dynamics. The A2 arm is unaffected: its pool was built for its own target.

Available clean data:

| source | transitions |
|---|---:|
| silent arm (passive throughout) | **3,360** |
| A2-target, gate-OFF rounds (passive by design) | **3,273** |
| A2-target, gate-ON rounds (active) | 3,797 |
| *A0-target arm (excluded)* | *9,555* |

Ample for both the training and the held-out comparison, with no reliance on the
compromised arm.

## Sequence

1. **Build the symmetric pools** (ready; membership already computed).
2. **Define the coarse-graining and run the Markov test** on existing data — no
   new simulations, and it determines whether the state variable works before
   compute is spent.
3. Decide on the incomplete-information pair (section 3) and the two-allocation
   world (section 4) with those results in hand.

Steps 1 and 2 are the core result; 3 and 4 are extensions.
