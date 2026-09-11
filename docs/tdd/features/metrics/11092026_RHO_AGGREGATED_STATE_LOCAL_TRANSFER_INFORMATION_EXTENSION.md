# Extension Plan: Proper Persistence Aggregation for State-Local Transfer Information

## Purpose

This extension clarifies and formalizes how to construct a statistically useful state-resolved transfer-information map

\[
\bar T_\pi(x,b)
\]

by aggregating the existing persistence-resolved estimates

\[
T_\pi(x,b,\rho)
\]

across \(\rho\).

This is an **analysis-layer extension only**. It does not modify the simulation, controller, prompts, persistence dynamics, or primary physical-cell estimators.

The core principle is:

> **Estimate transfer information inside each physical persistence cell first, then aggregate the already-estimated state-local quantities across \(\rho\). Do not pool raw trajectories across \(\rho\) and recompute one CMI for the headline phase map.**

---

# 1. Quantity being aggregated

The existing state-local transfer information is

\[
T_\pi(x,b,\rho,s)
=
I(U_k;n_{Z,k+1}\mid x_k=x,b,\rho,s),
\]

where:

- \(U_k\in\{0,1\}\) is the binary controller intervention;
- \(n_Z\) is the population support for the controller target;
- \(x=n_Z/N\) is the target-support fraction;
- \(b\) is intervention budget;
- \(\rho\) is epistemic persistence;
- \(s\) denotes target semantics, e.g. truth-control or false-control.

The desired persistence-aggregated map is

\[
\boxed{
\bar T_\pi(x,b,s)
=
\mathbb E_{\rho\mid x,b,s}
\left[
T_\pi(x,b,\rho,s)
\right].
}
\]

This is a **descriptive state-local aggregate across persistence conditions**.

---

# 2. Recommended weighting across rho

For the \(x\times b\) phase map, use support/occupancy weighting.

Define

\[
N_{x,b,\rho,s}
\]

as the number of usable state-local observations contributing to the estimate at a given \((x,b,\rho,s)\).

Then

\[
\boxed{
w_{\rho}(x,b,s)
=
\frac{
N_{x,b,\rho,s}
}{
\sum_{\rho'}N_{x,b,\rho',s}
}.
}
\]

The persistence-aggregated state-local estimator is

\[
\boxed{
\bar T_\pi(x,b,s)
=
\sum_\rho
w_{\rho}(x,b,s)\,
T_\pi(x,b,\rho,s).
}
\]

Interpretation:

> At state \(x\) and budget \(b\), average the transfer-information values from the persistence conditions that actually visit and support that state, with more strongly sampled persistence conditions receiving more weight.

This is useful because different persistence values populate different parts of the phase space.

---

# 3. Why this is preferable to raw pooling across rho

Do **not** define the headline state-local map by concatenating all transitions from all persistence values and computing

\[
I(U;n'_Z\mid x,b)
\]

on the pooled records.

In general,

\[
\boxed{
I(U;n'_Z\mid x,b)
\neq
\sum_\rho
p(\rho\mid x,b)
I(U;n'_Z\mid x,b,\rho).
}
\]

Persistence is a scientific condition that changes the transition dynamics and the distribution of controller actions.

Pooling raw observations across \(\rho\) can therefore create or destroy apparent dependence through between-\(\rho\) mixture effects.

For the main phase map, preserve the physical-cell conditioning and aggregate the cell-local estimators afterward.

---

# 4. Distinguish two different rho-aggregated objects

The implementation should expose both quantities if useful, but they must have different names and interpretations.

## 4.1 Recommended headline state-local aggregate

\[
\boxed{
T_{\pi,\mathrm{rho\_avg}}(x,b,s)
=
\sum_\rho
w_\rho(x,b,s)T_\pi(x,b,\rho,s)
}
\]

Suggested name:

```text
target_actuation_cmi_state_local_rho_aggregated
```

Metadata:

```text
aggregation_scope = state_local_rho_marginalized
aggregation_weight = n_observations
descriptive_only = true
```

This should be the default \(x\times b\) phase map.

---

## 4.2 Optional pooled-record diagnostic

If desired, separately compute

\[
T_{\pi,\mathrm{pooled}}(x,b,s)
=
I(U;n'_Z\mid x,b,s)
\]

after concatenating records across \(\rho\).

Suggested name:

```text
target_actuation_cmi_state_local_rho_pooled_records
```

Metadata:

```text
aggregation_scope = pooled_records_across_rho
descriptive_only = true
```

This is **not** the same estimator and should not replace the recommended rho-averaged map.

Its main use is diagnostic:

> Does pooling across persistence materially change the apparent transfer information?

---

# 5. Truth-control and false-control maps

First compute the persistence-aggregated maps separately:

\[
\bar T_\pi(x,b,\mathrm{truth}),
\]

\[
\bar T_\pi(x,b,\mathrm{false}).
\]

These should remain available even if a pooled controller-target map is also produced.

This allows the analysis to detect asymmetry between truth-aligned and false-target control.

---

# 6. Pool truth and false control only in controller-target coordinates

For the final target-semantics-aggregated map, use the controller-target coordinate \(x=n_Z/N\).

Do not pool truth and false conditions in truth coordinates.

Define

\[
\bar T_\pi(x,b)
=
\sum_s
w_s(x,b)\,
\bar T_\pi(x,b,s),
\]

where \(s\in\{\mathrm{truth},\mathrm{false}\}\).

Two weighting options should be supported.

## 6.1 Observation-weighted pooled map

Recommended for phase-space coverage:

\[
w_s(x,b)
=
\frac{
N_{x,b,s}
}{
\sum_{s'}N_{x,b,s'}
}.
\]

This gives

\[
\boxed{
\bar T_\pi^{\rm occ}(x,b)
=
\sum_s
w_s(x,b)
\bar T_\pi(x,b,s).
}
\]

Interpretation:

> Transfer information in controller-target coordinates across all observed controlled trajectories.

---

## 6.2 Balanced truth/false pooled map

Optional secondary summary:

\[
w_{\rm truth}=w_{\rm false}=\frac12
\]

provided both semantic arms have support at that state.

This prevents the more frequently visited semantic arm from dominating.

Suggested name:

```text
target_actuation_cmi_state_local_target_semantics_balanced
```

If one arm is unsupported at a given \((x,b)\), do not silently replace it with zero.

Mark the balanced estimate as incomplete/unsupported or expose a coverage flag.

---

# 7. Null aggregation

If state-local null estimates exist for each \((x,b,\rho,s)\), aggregate them using the **same weights** used for the observed statistic.

For null replicate \(r\),

\[
\bar T_{\pi,\rm null}^{(r)}(x,b,s)
=
\sum_\rho
w_\rho^{(r)}(x,b,s)
T_{\pi,\rm null}^{(r)}(x,b,\rho,s).
\]

Then compute

\[
\boxed{
\Delta \bar T_\pi(x,b,s)
=
\bar T_\pi(x,b,s)
-
E_r[
\bar T_{\pi,\rm null}^{(r)}(x,b,s)
].
}
\]

Preferred outputs:

```text
raw_T_pi
null_mean
null_sd
T_pi_minus_null
permutation_p_value
```

### Important

Do not average cell-wise permutation p-values.

If state-local null replicates are unavailable, do not fabricate a null-adjusted map. Report raw \(T_\pi\) plus support and say that null-adjusted state-local inference is unavailable.

---

# 8. Bootstrap uncertainty

For uncertainty on the rho-aggregated state-local map:

1. resample complete episodes within each physical \((\rho,b,s)\) cell;
2. recompute each state-local \(T_\pi(x,b,\rho,s)\);
3. recompute the support weights;
4. aggregate across \(\rho\);
5. store the resulting \(\bar T_\pi(x,b,s)\).

Repeat for all bootstrap replicates.

Do **not** independently bootstrap each state-local value and then combine only CI endpoints.

The aggregate must be recomputed inside each bootstrap replicate.

Output:

```text
estimate
ci_low
ci_high
bootstrap_sd
```

---

# 9. Support requirements

Every aggregated \((x,b)\) row should carry explicit support diagnostics.

At minimum:

```text
x_bin_index
x_bin_lower
x_bin_upper
x_bin_center
intervention_budget

n_observations
n_rho_contributing
rho_values_contributing
n_target_semantics_contributing

dual_action_supported
support_status

aggregation_weight
aggregation_scope
```

Recommended support interpretation:

```text
adequate:
    enough observations and both U=0/U=1 represented

limited:
    estimate exists but support is weak

unsupported:
    no reliable state-local estimator
```

Unsupported bins must be blank/hatched in plots, not treated as zero.

---

# 10. Minimum number of contributing rho values

Do not require all \(\rho\) values to visit every state.

That would defeat the purpose of this aggregation.

Instead expose:

```text
n_rho_contributing
```

for every \((x,b)\).

Recommended plotting convention:

- metric heatmap = aggregated estimate;
- adjacent support heatmap = number of contributing \(\rho\) values;
- optionally annotate bins supported by only one \(\rho\).

A bin supported by one persistence value can be shown descriptively but should not be interpreted as a robust persistence-aggregated result.

---

# 11. Main output tables

Create:

```text
state_local_rho_aggregated_metrics.parquet
state_local_rho_aggregated_metrics.csv
```

Recommended schema:

```text
metric

target_semantics
intervention_budget

x_bin_index
x_bin_lower
x_bin_upper
x_bin_center

estimate
ci_low
ci_high

null_mean
null_sd
null_adjusted_estimate
permutation_p_value

n_observations
n_rho_contributing
rho_values_json

aggregation_scope
aggregation_weight
support_status
descriptive_only
```

Create a second table for truth+false pooled target-coordinate maps:

```text
state_local_rho_target_aggregated_metrics.parquet
state_local_rho_target_aggregated_metrics.csv
```

---

# 12. Required plots

Produce these for:

```text
T_pi
T_pi - T_null     # only if valid state-local nulls exist
chi
eta_IF
eta_IR
```

## A. Truth control

\[
(x,b)
\]

after rho aggregation.

## B. False control

\[
(x,b)
\]

after rho aggregation.

## C. Truth + false pooled in controller-target coordinates

\[
(x,b)
\]

after rho and target-semantics aggregation.

For every metric family, also provide:

```text
n_observations(x,b)
n_rho_contributing(x,b)
dual_action_support(x,b)
```

Use identical x and b axes across all three semantic views.

---

# 13. Apply the same rho aggregation to chi

For susceptibility,

\[
\chi(x,b,\rho,s)
\]

aggregate as

\[
\boxed{
\bar\chi(x,b,s)
=
\sum_\rho
w_\rho(x,b,s)\chi(x,b,\rho,s).
}
\]

Use the same support/occupancy weights as for \(T_\pi\) unless the susceptibility estimator has a clearly different effective sample count.

If it does, use the estimator-specific observation count and document it.

---

# 14. Apply the same logic to eta_IF and eta_IR carefully

Do **not** simply average the efficiency ratios if their components are available.

For each \((x,b,\rho,s)\), retain components.

## eta_IF

If

\[
\eta_{\rm IF}
=
\frac{T_\pi}{H(U\mid x)},
\]

then the rho-aggregated state-local quantity should be

\[
\boxed{
\bar\eta_{\rm IF}(x,b,s)
=
\frac{
\sum_\rho w_\rho T_\pi(x,b,\rho,s)
}{
\sum_\rho w_\rho H(U\mid x,b,\rho,s)
}.
}
\]

Do not calculate

\[
\sum_\rho w_\rho \eta_{\rm IF}(x,b,\rho,s)
\]

as the canonical aggregate.

---

## eta_IR

If

\[
\eta_{\rm IR}
=
\frac{B_{\rm IR}}{T_\pi},
\]

then aggregate components first:

\[
\boxed{
\bar\eta_{\rm IR}(x,b,s)
=
\frac{
\sum_\rho w_\rho B_{\rm IR}(x,b,\rho,s)
}{
\sum_\rho w_\rho T_\pi(x,b,\rho,s)
}.
}
\]

Again, do not average the ratios.

---

# 15. Relationship to the whole-cell per-budget metric

Keep the following distinction explicit.

## State-local rho-aggregated map

\[
\bar T_\pi(x,b)
\]

answers:

> Where in controller-target state space is transfer information observed, after using all persistence conditions that visit that state?

## Whole-cell study-level metric

\[
\bar T_\pi(b)
\]

answers:

> How much transfer information is observed overall at budget \(b\), after marginalizing state occupancy and selected study dimensions?

These are complementary.

The whole-cell quantity should normally be the stronger inferential statistic.

The \(x\times b\) map should explain the structure behind it.

---

# 16. Validation check: aggregate back over x

As a diagnostic only, compare the rho-aggregated state-local map reweighted over state occupancy against the independently computed whole-cell aggregate.

Compute

\[
T_{\pi,\rm reconstructed}(b,s)
=
\sum_x
p(x\mid b,s)\,
\bar T_\pi(x,b,s).
\]

Compare with the physical-cell-derived study aggregate

\[
\bar T_\pi(b,s).
\]

They may not be numerically identical if:

- state bins are coarse;
- different support rules are applied;
- some states are dropped;
- the local estimator and whole-cell CMI are not algebraically identical after binning.

But they should be broadly consistent.

Export the discrepancy:

```text
state_local_reconstruction
whole_cell_aggregate
difference
relative_difference
```

This is a useful sanity check, not a requirement that the values match exactly.

---

# 17. Implementation rule for the current adaptive study

For the current adaptive-communication dataset:

1. keep the five physical persistence values separate during estimation;
2. reconstruct valid state-local estimates from the primary estimator table;
3. ignore the broken gray placeholder export if it incorrectly says `structural_cell_not_run`;
4. aggregate valid state-local estimates across rho using observation/support weights;
5. generate truth, false, and truth+false target-coordinate maps;
6. regenerate support maps;
7. add the results to the full report.

No provider calls and no new simulation episodes are required for this extension.

---

# 18. Tests

Add at least these tests.

## Test 1 — weighted rho average

Synthetic values:

```text
rho=.7: T=0.1, n=10
rho=.8: T=0.3, n=30
```

Expected observation-weighted aggregate:

\[
0.25.
\]

---

## Test 2 — not equal to raw pooled CMI

Construct a synthetic fixture where persistence changes the action distribution.

Verify numerically that:

```text
weighted mean of rho-conditioned CMI
```

and

```text
CMI from pooled raw observations
```

are different.

The implementation must keep the two outputs distinct.

---

## Test 3 — unsupported rho is not zero

If one persistence value has no supported state-local estimate, exclude it from the weighted estimate and record reduced coverage.

Never replace it with zero.

---

## Test 4 — truth/false pooling uses target coordinates

Equivalent target-coordinate dynamics with opposite truth semantics should give the same pooled result.

---

## Test 5 — eta_IF ratio of components

Verify aggregate \(\eta_{\rm IF}\) is ratio of weighted numerator/denominator components, not weighted mean of ratios.

---

## Test 6 — eta_IR ratio of components

Same for \(\eta_{\rm IR}\).

---

## Test 7 — bootstrap reaggregation

Each bootstrap replicate must recompute:

```text
state-local estimates
weights
rho aggregate
```

before constructing the CI.

---

# 19. Documentation wording to add to metrics.md

Suggested text:

> **Persistence-aggregated state-local metrics.**  
> State-local quantities such as \(T_\pi(x,b,\rho)\) may have sparse support because different persistence values visit different regions of population state space. For descriptive \(x\times b\) phase maps, the analysis first estimates each quantity inside the physical \(\rho\) cell and then averages the supported state-local estimates across persistence using observation/support weights. This differs from concatenating records across persistence and recomputing a single CMI, which can mix between-\(\rho\) differences in dynamics and controller activation. The persistence-aggregated map therefore represents the expected local metric over the persistence conditions that actually support a given state.

Also add:

> Truth-target and false-target phase maps may subsequently be pooled only in controller-target coordinates. Separate semantic-arm maps remain available because epistemic outcomes can differ even when target-coordinate controllability is similar.

---

# 20. Acceptance criteria

This extension is complete when:

```text
[ ] T_pi(x,b,rho,s) remains the physical-cell estimator
[ ] T_pi(x,b,s) can be aggregated across rho
[ ] observation/support weights are explicit
[ ] raw pooling across rho is not used as the headline map
[ ] optional pooled-record CMI is separately named
[ ] truth and false maps are retained
[ ] truth+false target-coordinate pooled map exists
[ ] null aggregation uses the same weights
[ ] p-values are not averaged
[ ] bootstrap recomputes aggregate per replicate
[ ] support maps are produced
[ ] n_rho_contributing is exported
[ ] unsupported rho values are not treated as zero
[ ] eta_IF is aggregated from components
[ ] eta_IR is aggregated from components
[ ] x-aggregated reconstruction is compared with whole-cell summary
[ ] metrics.md is updated
[ ] current adaptive study is reaggregated without provider calls
```

---

# Scientific interpretation

The purpose of this extension is not to claim that persistence is irrelevant.

The purpose is to use the fact that different persistence regimes explore different portions of the population state space.

The resulting map

\[
\boxed{
\bar T_\pi(x,b)
=
\mathbb E_{\rho\mid x,b}[T_\pi(x,b,\rho)]
}
\]

should be interpreted as:

> **the transfer information observed at population state \(x\) and budget \(b\), averaged over the persistence regimes that actually support that state.**

This gives a much better-supported phase-space picture while preserving the physical-cell estimator and avoiding spurious dependencies caused by indiscriminate raw pooling across persistence.
