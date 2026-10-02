# Four tests of the epistemic coarse-graining

**2 October 2026.** Offline analysis of `21-09-2026-full-vs-report-v1`. No
simulations were run.

The earlier [`markov_test_report`](markov_test_report_2026-10-02.md) used a
single test (order comparison) and a cross-arm transfer test that turned out to
be invalid. This report runs **four independent tests** and, where the state
fails, identifies **which hidden coordinate is missing**.

## Summary

| | |
|---|---|
| Do the four tests agree? | **Yes, completely** |
| Where is the chain Markovian? | **Only in the silent arm at ρ = 1.00** |
| Where is it worst? | **ρ = 0.75**, both arms |
| What is the missing coordinate? | **Knowledge spread** under silence; **survival of the decisive facts** under control |
| Does adding a coordinate fix it? | **Partly, and only where it was already nearly fine** |

## 1. The four tests

Each probes a different consequence of the Markov property, so agreement between
them is meaningful.

**Order comparison.** Fit order-1 and order-2 chains; compare *held-out*
log-likelihood with episode-level splits (12 seeds). Gain = order-2 minus
order-1, in nats/step. Zero or negative means the previous state adds nothing.

**Chapman–Kolmogorov.** Does the fitted one-step kernel raised to the *n*-th
power match the empirical *n*-step kernel? Reported as weighted total-variation
distance. Probes longer horizons than the order test reaches.

**Conditional independence.** Estimate `I(S(t+1) ; S(t−1) | S(t))` directly, in
bits. Exactly zero under the Markov property. The plug-in estimate is biased
upward, so it is compared against a permutation null that shuffles `S(t−1)`
within each `S(t)` stratum.

**Lumpability.** Within one macrostate, split the microstates by a candidate
hidden feature and compare their next-state distributions. A gap means the
macrostate is not a sufficient summary, and names what is missing.

## 2. Results — all four agree

| ρ | arm | order gain | CK n=2 | CK n=3 | CK n=5 | CMI excess (bits) | CMI *p* |
|---|---|---:|---:|---:|---:|---:|---:|
| 0.75 | silent | 0.0161 | 0.044 | 0.084 | 0.106 | 0.0284 | **0.005** |
| 0.75 | A2 | 0.0146 | 0.047 | 0.082 | 0.109 | 0.0261 | **0.005** |
| 0.75 | A0* | 0.0139 | 0.052 | 0.090 | 0.110 | 0.0225 | **0.005** |
| **1.00** | **silent** | **−0.0050** | **0.009** | **0.019** | **0.033** | **0.0017** | **0.154** |
| 1.00 | A2 | 0.0060 | 0.032 | 0.060 | 0.078 | 0.0130 | **0.005** |
| 1.00 | A0* | 0.0025 | 0.023 | 0.040 | 0.068 | 0.0074 | **0.005** |

\* the confounded arm; included because the *dynamics* are sound.

**One cell passes every test: ρ = 1.00, silent.** Its order gain is negative (a
second-order chain predicts held-out rounds *worse*), its Chapman–Kolmogorov gap
is the smallest at every horizon, and it is the **only** row whose conditional
mutual information is indistinguishable from its permutation null (*p* = 0.154;
every other row is at the 0.005 floor).

Three further patterns:

- **ρ = 0.75 is uniformly worst**, by every test, in every arm.
- **Control degrades the state even at ρ = 1.** The silent arm passes; the
  controlled arms at the same persistence fail (*p* = 0.005). Intervention
  introduces structure the mean does not capture.
- **Chapman–Kolmogorov errors compound.** At ρ = 1 silent the gap grows
  0.009 → 0.019 → 0.033 from 2 to 5 steps; at ρ = 0.75 it reaches 0.11. A state
  adequate for one-step prediction can still drift over a trajectory, which
  matters directly for path KL.

The earlier single-test conclusion ("Markovian at ρ = 1") was too generous. The
correct statement is **"Markovian at ρ = 1 under silence only."**

## 3. Which coordinate is missing

Lumpability, as weighted TV between the two halves of each macrostate:

| ρ | arm | spread | dec_frac | pool_frac | mean_nfacts |
|---|---|---:|---:|---:|---:|
| 0.75 | silent | **0.130** | 0.089 | 0.065 | 0.079 |
| 0.75 | A2 | 0.050 | **0.089** | 0.069 | 0.072 |
| 0.75 | A0 | 0.055 | **0.080** | 0.061 | 0.043 |
| 1.00 | silent | **0.095** | 0.027 | 0.065 | 0.013 |
| 1.00 | A2 | 0.072 | 0.062 | — | **0.111** |
| 1.00 | A0 | 0.051 | **0.124** | 0.016 | 0.102 |

The answer **depends on the arm**, and both answers are mechanistically sensible:

**Under silence, the missing coordinate is spread** (0.130, 0.095) — whether
knowledge is concentrated in a few agents or shared out. Two populations with the
same mean but different concentration differ in whether any single agent can
broadcast a complete argument.

**Under control, it is survival of the decisive facts** (0.089, 0.124) — whether
the six facts that constitute the population's only complete proof are still
alive in anyone's memory. Under intervention, *that* is what distinguishes
futures, not how evenly knowledge is spread.

`pool_frac` is weak everywhere (0.016–0.069) and undefined at ρ=1/A2 where it is
nearly constant at 1. Controller-fact exposure separates *arms* but does not
predict transitions *within* a macrostate — it is a label, not a dynamical
coordinate.

## 4. Does adding the coordinate fix it?

Each candidate was added as an explicit binary second dimension (median split),
taking the state space from 4 to 8, and the order test re-run.

Order gain, lower is more Markovian:

| state | ρ=0.75 silent | ρ=0.75 A2 | ρ=1.00 silent | ρ=1.00 A2 |
|---|---:|---:|---:|---:|
| `e` only (4 states) | 0.0161 | 0.0146 | −0.0050 | 0.0060 |
| `e × spread` (8) | 0.0274 | 0.0274 | **−0.0124** | 0.0097 |
| `e × dec_frac` (8) | 0.0249 | 0.0135 | −0.0050 | 0.0023 |
| `e × pool_frac` (8) | **0.0011** | 0.0207 | **−0.0140** | 0.0060 |
| `e × mean_nfacts` (8) | 0.0328 | **0.0133** | **−0.0087** | **−0.0009** |

**Where the state was already nearly sufficient, adding a coordinate helps.** At
ρ = 1 every augmentation drives the gain further negative, and `e × mean_nfacts`
turns the controlled arm negative too (−0.0009 from +0.0060). At ρ = 1 a
two-dimensional state is genuinely Markovian in both arms.

**At ρ = 0.75 it mostly does not.** `e × pool_frac` in the silent arm looks
spectacular (0.0161 → 0.0011), but `spread` and `mean_nfacts` make things *worse*
there, and in the controlled arm the best improvement is marginal
(0.0146 → 0.0133). Doubling the state space halves the data per transition, and
at ρ = 0.75 that cost roughly cancels the gain.

Note the tension with §3: lumpability says `spread` is the strongest missing
coordinate under silence, yet adding it makes the order test *worse* there. Both
can be true — `spread` carries real information about the next step, but a binary
median split is a poor encoding of it, and the sparsity cost exceeds the signal
recovered. This is a reason to prefer lumpability for *diagnosis* and the order
test for *deciding what to use*.

## 5. What to use

**At ρ = 1.00:** use `e × mean_nfacts`, 8 states. Markovian in both arms by the
order test, and still small enough to estimate (≈ 8 states, 64 transition
probabilities, ~900 transitions per arm).

**At ρ = 0.75:** use `e` alone, 4 states, and treat the chain as an
approximation carrying ≈ 0.015 nats/step of unexplained memory. No augmentation
tested here reliably improves it, and the Chapman–Kolmogorov gap of 0.11 at five
steps means path quantities over long horizons should be read as indicative.

**Never pool persistence levels.** The two regimes differ in which coordinate is
missing, in how Markovian they are, and in which states they occupy.

## 6. Why finite persistence is the harder regime

Forgetting does not reduce the system's memory in the sense that matters. A
coarse-graining is Markovian when every microstate sharing a macrostate has the
same distribution over next macrostates (lumpability). Forgetting *widens* those
fibres: at ρ = 1 each agent's fact set only grows, so the mean tracks a genuine
one-way progress coordinate; at ρ = 0.75 the same mean arises from populations
with knowledge concentrated or dispersed, holding decisive facts or decoy facts,
and these evolve differently.

The underlying principle is **timescale separation**. Coarse-graining works when
a slow collective coordinate exists and the fast degrees of freedom equilibrate.
At ρ = 0.75 forgetting runs at 25% per round — comparable to acquisition — so
there is no slow coordinate to project onto.

A caveat on the ρ = 1 result: that chain is also **less mobile**, changing state
in 7.6% of rounds against 15.5% at ρ = 0.75 (mean diag(T) 0.80 vs 0.66). A chain
that moves less is easier to call Markovian. The result is not an artifact —
diag(T) = 0.80 is far from frozen — but part of the gap is reduced mobility
rather than a better summary.

## 7. Caveats

- One task, one model, 60 initializations.
- The CMI permutation null shuffles within `S(t)` strata, which tests conditional
  independence given the *binned* state; a finer binning could in principle
  change the verdict, though §5c of the earlier report found bin count does not
  matter.
- Chapman–Kolmogorov at n = 5 uses fewer, overlapping windows and so is noisier
  than n = 2.
- Augmentation used median splits, the crudest possible encoding. A better
  encoding of `spread` might help where the binary one did not.
- The A0 arm is confounded and is reported for dynamics only.
