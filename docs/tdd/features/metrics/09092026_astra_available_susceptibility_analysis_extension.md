# ASTRA analysis extension — available causal susceptibility

**Scope:** offline analysis only  
**Goal:** add an availability-normalized version of the existing propensity-weighted immediate causal response, without changing the simulation, controller, prompts, or recorded runtime behavior.

## 1. Concept

The current Phase-2 causal response estimator is

\[
\psi_{t,h}
=
\left[
\frac{U_t}{e_t}
-
\frac{1-U_t}{1-e_t}
\right]
(x_{t+h}-x_t),
\]

with \(e_t=P(U_t=1\mid I_t)\).

For the immediate response \(h=1\), define the **available causal susceptibility score**

\[
\psi^{\mathrm{avail}}_{t,1}
=
\frac{\psi_{t,1}}{1-x_t},
\qquad x_t<1.
\]

Here \(1-x_t\) is the pre-intervention fraction of agents not yet supporting the controller target. Because it is measured before treatment, this normalization does not condition on a post-intervention variable.

Interpretation:

> causal target-share response per unit of population mass still available to move toward the controller target.

At \(x_t=1\), the quantity is undefined and must remain `NaN` / unsupported rather than being set to zero or clipped.

## 2. Implementation location

Extend the existing offline causal-response analysis, preferably in:

```text
src/mas_cc/analysis/causal_response.py
```

and wire the new outputs through the existing study aggregation in:

```text
src/mas_cc/studies/aggregation.py
```

Do **not** modify:

- the game runtime;
- controller action sampling;
- prompts;
- episode generation;
- provider calls;
- canonical recording semantics.

The required fields already exist in canonical round data:

```text
U_t
e_t
x_t
x_{t+1}
episode / round identity
shared-initialization block identity
cell coordinates such as persistence rho and budget b
```

## 3. Required estimators

### A. State-local available causal susceptibility

For each eligible round with \(x_t<1\), compute

\[
\psi^{\mathrm{avail}}_{t,1}
=
\left[
\frac{U_t}{e_t}
-
\frac{1-U_t}{1-e_t}
\right]
\frac{x_{t+1}-x_t}{1-x_t}.
\]

Aggregate this using the same eight \(x\)-bin convention already used by the study:

```text
B(x) = min(floor(8*x), 7)
```

For each scientific cell and \(x\)-bin, report:

- mean available causal susceptibility;
- number of contributing rounds;
- number of excluded saturated rounds \(x_t=1\);
- action/silence counts;
- propensity support diagnostics;
- identified initialization-block count.

Use the **actual round-level \(1-x_t\)** before averaging. Do not divide a bin-level mean response by the bin center.

### B. Stable cell-level summary

Also report the available-mass-weighted cell summary

\[
\chi_{\mathrm{avail}}^{\mathrm{mass}}
=
\frac{\sum_t \psi_{t,1}}
{\sum_t (1-x_t)},
\]

over eligible rounds in the cell.

Equivalently,

\[
\chi_{\mathrm{avail}}^{\mathrm{mass}}
=
\frac{E[\psi_{t,1}]}{E[1-x_t]}.
\]

This should be the preferred one-number-per-cell summary because it avoids giving extreme leverage to individual near-saturated rounds.

Keep this distinct from

\[
E\!\left[\frac{\psi_{t,1}}{1-x_t}\right],
\]

which is the average state-local normalized score.

## 4. Uncertainty

Reuse the current Phase-2 causal-response resampling contract:

- resample whole shared-initialization blocks;
- preserve all matched cells/episodes belonging to a block;
- recompute numerator and denominator inside every bootstrap replicate;
- use the configured confidence level and number of bootstrap draws;
- never treat rounds as independent bootstrap units.

For the ratio-of-sums summary, compute the ratio **inside each bootstrap replicate** rather than bootstrapping a precomputed scalar ratio.

## 5. Outputs

Add compact tables such as:

```text
analysis/tables/available_causal_susceptibility_state_local.parquet
analysis/tables/available_causal_susceptibility_summary.parquet
```

Recommended fields include:

```text
cell_id
persistence
budget
x_bin
estimate
ci_low
ci_high
n_rounds
n_saturated_excluded
n_action
n_silence
n_initialization_blocks
propensity_min
propensity_median
propensity_max
available_mass
estimator_name
```

Use an explicit estimator name such as:

```text
propensity_weighted_available_susceptibility
```

For the cell-level ratio-of-sums summary, use a distinct name such as:

```text
available_mass_weighted_causal_susceptibility
```

## 6. Plots

Add the following offline plots:

1. **\(x\)-vs-\(b\) phase diagram** of available causal susceptibility for each persistence value \(\rho\).
2. **Available causal susceptibility vs budget** for representative \(x\)-bins.
3. **Cell summary heatmap** over \((\rho,b)\) using the ratio-of-sums summary.
4. If the blackboard theory is available in the same analysis package, optionally add:
   \[
   \hat\chi_{\mathrm{avail}}^{\mathrm{causal}}(x,b)
   \quad\text{vs}\quad
   \chi_{\mathrm{avail}}^{\mathrm{BB}}(x,b)
   =
   \frac{\chi_{\mathrm{BB}}(x,b)}{1-x}.
   \]

Do not redefine or replace `round_target_susceptibility`, `eta_ir`, `T_pi`, or thermodynamic efficiency.

## 7. Validation and tests

Add tests showing that:

1. a synthetic randomized example with known causal response recovers the correct available susceptibility;
2. \(x_t=1\) is excluded and reported, never coerced to zero;
3. the implementation uses the logged propensity \(e_t\);
4. state-local values use each row's exact \(1-x_t\), not bin centers;
5. lag lookup does not cross episode boundaries;
6. block bootstrap preserves shared initializations;
7. the ratio-of-sums summary is recomputed inside every bootstrap replicate;
8. row order and execution sharding do not change the result;
9. existing study packages can be reaggregated without provider calls;
10. existing susceptibility, CMI, \(\eta_{IR}\), and causal-response outputs remain unchanged.

## 8. Documentation

Update the task-003 metrics tutorial and aggregation contract with one short subsection explaining:

- this is an **offline derived analysis**;
- it is the availability-normalized version of the \(h=1\) propensity-weighted causal response;
- \(1-x_t\) is pre-intervention available target-convertible population mass;
- the quantity is undefined at \(x_t=1\);
- it is a susceptibility normalization, **not** a new information or thermodynamic efficiency.

## 9. Acceptance criterion

The extension is complete when an already finished canonical study can be reaggregated and produces the new tables and plots with no simulation rerun and no provider calls, while all pre-existing outputs remain bit-for-bit or numerically unchanged except for the addition of the new analysis artifacts.
