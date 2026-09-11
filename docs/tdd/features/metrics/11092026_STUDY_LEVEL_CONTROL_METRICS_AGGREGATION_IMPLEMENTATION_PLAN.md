# Implementation Plan: Study-Level Aggregation of Transfer Information, Susceptibility, and Efficiencies

## Purpose

Extend the current blackboard analysis pipeline so that the information/control metrics already defined in `metrics.md` can be summarized at statistically stronger aggregation levels **without rerunning the LLM simulations**.

The main motivation is that the fully state-resolved quantity

\[
T_\pi(x,\rho,b,s)
\]

can be very sparse, because some population states \(x\) are rarely or never visited for particular persistence values \(\rho\), budgets \(b\), and controller-target semantics \(s\). This does **not** mean the whole experiment is unusable. The current analysis already computes whole-cell conditional mutual information, which integrates over the empirical occupancy of the visited population states.

The implementation should therefore preserve the detailed state-local maps as mechanistic diagnostics, while adding progressively more aggregated study-level summaries that use much more of the available data.

Here:

- \(x\) = current fraction/count supporting the **controller target**;
- \(\rho\) = epistemic persistence;
- \(b\) = intervention/communication budget;
- \(s\) = controller-target semantics, e.g. truth-target or false-target;
- \(U\in\{0,1\}\) = binary controller intervention;
- \(n_Z\) = population count supporting the controller target.

The existing target-actuation CMI is

\[
T_\pi
=
I(U_k;n_{Z,k+1}\mid n_{Z,k}).
\]

This should remain the primary transfer-information estimator.

---

# 1. Do not replace the current physical-cell estimators

The current `metrics.md` contract is correct and should remain intact:

- primary estimators are computed **inside physical scientific cells**;
- cells with different scientific conditions are not silently pooled;
- `round_target_actuation_cmi` is the target-coordinate transfer-information estimator;
- `round_target_susceptibility` / `susceptibility_occupancy_weighted` quantify target response;
- `round_target_information_fraction` / `eta_IF` use the available action-entropy ceiling;
- `eta_ir` is derived from a Pinsker numerator and the transfer-information denominator;
- support diagnostics must accompany conditional-MI estimates;
- nulls are generated from the retained observations;
- uncertainty is based on whole-episode resampling.

The new quantities described below are therefore **derived study-level summaries**, not replacements for physical-cell estimates.

Do not alter the meaning of `round_target_actuation_cmi`.

---

# 2. Important observation: x is already marginalized in the whole-cell CMI

The difficult phase diagram is the state-local decomposition

\[
T_\pi(x,\rho,b,s).
\]

However, the existing whole-cell estimator

\[
T_\pi(\rho,b,s)
=
I(U;n'_Z\mid n_Z,\rho,b,s)
\]

already conditions on the current target state and then averages over the empirical occupancy of the states that were actually visited.

Conceptually,

\[
T_\pi(\rho,b,s)
=
\sum_x
p(x\mid \rho,b,s)
\,
T_\pi(x,\rho,b,s).
\]

Therefore:

- the state-local map answers **where in state space control information appears**;
- the whole-cell CMI answers **how much target-control information is transmitted in that experimental condition overall**.

The whole-cell quantity is statistically easier because it uses all supported visited states in the cell.

This distinction should be made explicit in `metrics.md`.

---

# 3. Aggregation hierarchy to implement

Implement four principal levels.

## Level 0 — State-local diagnostic

Keep the current state-local estimator:

\[
T_\pi(x,\rho,b,s).
\]

This is the most detailed and statistically hardest quantity.

It should remain available for phase diagrams, but sparse/unsupported bins must be shown as unsupported rather than interpreted as zero.

---

## Level 1 — Physical-cell / x-marginalized estimate

This already exists:

\[
\boxed{
T_\pi(\rho,b,s)
=
I(U;n'_Z\mid n_Z,\rho,b,s)
}
\]

Use the existing `round_target_actuation_cmi`.

Do not implement a duplicate estimator under a new name.

Likewise keep the existing physical-cell versions of:

\[
\chi(\rho,b,s),
\qquad
\eta_{\rm IF}(\rho,b,s),
\qquad
\eta_{\rm IR}(\rho,b,s).
\]

---

## Level 2 — Persistence-marginalized summary

For fixed budget and target semantics define:

\[
\boxed{
\bar T_\pi(b,s)
=
\sum_{\rho}
w_\rho
T_\pi(\rho,b,s).
}
\]

This is the first important new derived metric.

Recommended headline weighting:

\[
w_\rho=\frac{1}{N_\rho}
\]

over the persistence values represented by the experimental design, i.e. **equal physical-cell weighting across \(\rho\)**.

This prevents a persistence value from dominating merely because it happened to retain more rounds or visit more states.

Name suggestion:

```text
target_actuation_cmi_rho_aggregated
```

or, if study-level names are preferred,

```text
study_target_actuation_cmi_rho_aggregated
```

---

## Level 3 — Target-semantics-marginalized summary

For fixed \((\rho,b)\), aggregate truth-target and false-target control in **controller-target coordinates**:

\[
\boxed{
\bar T_\pi(\rho,b)
=
\sum_s w_s T_\pi(\rho,b,s).
}
\]

Recommended default:

\[
w_{\rm truth}
=
w_{\rm false}
=
\frac12.
\]

This is scientifically meaningful because `round_target_actuation_cmi` uses \(n_Z\): the population count supporting **whatever target the controller is trying to increase**.

This aggregation answers:

> How much transfer information does the binary controller transmit toward its selected target, independent of whether the target is epistemically correct?

Do **not** use `round_truth_actuation_cmi` for this pooled controller-target quantity.

Name suggestion:

```text
target_actuation_cmi_target_semantics_aggregated
```

---

## Level 4 — Main per-budget study-level transfer information

Aggregate over both persistence and controller-target semantics:

\[
\boxed{
\bar T_\pi(b)
=
\sum_{\rho,s}
w_{\rho,s}
T_\pi(\rho,b,s).
}
\]

Recommended default: balanced design weighting,

\[
w_{\rho,s}
=
\frac{1}{N_\rho N_s}.
\]

This should become the statistically strongest **per-budget transfer-information summary**.

Name suggestion:

```text
target_actuation_cmi_budget_summary
```

The output is still indexed by budget \(b\); “budget summary” means other selected study dimensions have been marginalized, not that budget itself has been averaged away.

---

# 4. Two weighting modes must be distinguished

There are two scientifically useful weighting schemes. Do not mix them silently.

## 4.1 Balanced design weighting — headline study summaries

Use equal weight across physical experimental conditions being marginalized.

Examples:

\[
\bar T_\pi(b,s)
=
\frac1{N_\rho}
\sum_\rho T_\pi(\rho,b,s)
\]

and

\[
\bar T_\pi(b)
=
\frac1{N_\rho N_s}
\sum_{\rho,s}T_\pi(\rho,b,s).
\]

Use this for headline per-budget comparisons.

Rationale: every designed condition gets the same influence.

Suggested metadata:

```text
aggregation_weight = balanced_cell
```

---

## 4.2 Occupancy / observation weighting — descriptive phase-space summaries

For state-local maps aggregated across persistence, use the amount of actual support at each state.

For example,

\[
\bar T_\pi(x,b,s)
=
\sum_\rho
p(\rho\mid x,b,s)
T_\pi(x,\rho,b,s).
\]

The weights can be proportional to the number of supported observations contributing to the state-local estimate.

This answers:

> Across the dynamics that actually visit state \(x\), how much transfer information is observed there?

This is useful for filling the phase-space picture because different \(\rho\) values populate different regions of \(x\).

Suggested metadata:

```text
aggregation_weight = n_observations
aggregation_scope = rho_marginalized_state_local
descriptive_only = true
```

Do not present this occupancy-weighted map as the same object as the balanced study-level summary.

---

# 5. Null-adjusted transfer information

Every new transfer-information aggregate must be paired with its null.

For a physical cell:

\[
\Delta T_\pi(\rho,b,s)
=
T_\pi(\rho,b,s)
-
E[T_{\pi,\rm null}(\rho,b,s)].
\]

For a study-level aggregate, **do not subtract separately averaged headline numbers after the fact if the null draws can be recomputed**.

For null replicate \(r\):

\[
\bar T_{\pi,\rm null}^{(r)}(b)
=
\sum_{\rho,s}
w_{\rho,s}
T_{\pi,\rm null}^{(r)}(\rho,b,s).
\]

Then define

\[
\boxed{
\Delta \bar T_\pi(b)
=
\bar T_\pi(b)
-
E_r[
\bar T_{\pi,\rm null}^{(r)}(b)
].
}
\]

The aggregate permutation p-value should be computed from the **aggregate null distribution**.

For example, with the usual +1 correction,

\[
p_{\rm perm}
=
\frac{
1+\#\{
\bar T_{\pi,\rm null}^{(r)}
\ge
\bar T_\pi
\}
}{
R+1
}.
\]

### Critical rule

**Never average permutation p-values across cells.**

The aggregate statistic needs its own aggregate null distribution.

Suggested fields:

```text
estimate_raw_bits
null_mean_bits
null_sd_bits
null_adjusted_bits
permutation_p_value
n_null_draws
```

---

# 6. Bootstrap uncertainty for the aggregates

Use a stratified whole-episode bootstrap.

For every bootstrap replicate:

1. stay inside each physical cell;
2. resample complete episodes with replacement;
3. recompute the physical-cell estimator;
4. combine the recomputed physical-cell values using the requested aggregate weights;
5. save the aggregate replicate.

Then compute the aggregate 95% CI from the aggregate bootstrap distribution.

This preserves the existing rule that rounds from the same episode are not treated as independent.

Recommended default:

```text
bootstrap_unit = episode
bootstrap_scope = stratified_by_physical_cell
```

If a later analysis explicitly wants to exploit common-random-number / paired-initialization structure across cells, implement that as a separate optional mode. Do not make it the first version.

---

# 7. Susceptibility aggregation

The current framework already produces:

```text
susceptibility_occupancy_weighted
```

for each physical cell.

Keep that quantity.

Add the same study-level aggregation hierarchy:

\[
\bar\chi(b,s)
=
\sum_\rho w_\rho \chi(\rho,b,s),
\]

\[
\bar\chi(\rho,b)
=
\sum_s w_s \chi(\rho,b,s),
\]

\[
\boxed{
\bar\chi(b)
=
\sum_{\rho,s}w_{\rho,s}\chi(\rho,b,s).
}
\]

Because \(\chi\) is already in a common target-fraction scale, a weighted average is appropriate.

Report:

```text
estimate
ci_low
ci_high
n_contributing_cells
```

Do not conflate:

```text
small chi
```

with

```text
poorly estimated chi
```

The report should show magnitude and uncertainty separately.

---

# 8. Information fraction η_IF

The existing target information fraction is conceptually

\[
\eta_{\rm IF}
=
\frac{
T_\pi
}{
H(U\mid n_Z)
}.
\]

Do **not** average cell-wise efficiencies.

For a study-level aggregate calculate the ratio of weighted components:

\[
\boxed{
\bar\eta_{\rm IF}(b)
=
\frac{
\sum_c w_c T_{\pi,c}
}{
\sum_c w_c H_c(U\mid n_Z)
}.
}
\]

Here \(c\) indexes the physical cells included in the requested aggregation.

The bootstrap must recompute numerator, denominator, and ratio inside every replicate.

Export both components:

```text
eta_if
eta_if_numerator_T_bits
eta_if_denominator_action_entropy_bits
```

### Optional diagnostic

A null-adjusted information-fraction diagnostic may be exported as

\[
\eta_{\rm IF}^{\rm excess}
=
\frac{
T_\pi-T_{\rm null}
}{
H(U\mid n_Z)
}.
\]

If implemented, label it explicitly as a **null-adjusted diagnostic**, not as the canonical bounded information fraction, because it can be negative.

Suggested name:

```text
eta_if_null_adjusted_diagnostic
```

---

# 9. Information-response efficiency η_IR

The existing `metrics.md` already exposes:

```text
eta_ir
eta_ir_pinsker_numerator_bits
eta_ir_denominator_T_bits
```

Do not average cell-wise `eta_ir`.

For a requested aggregate:

\[
\boxed{
\bar\eta_{\rm IR}
=
\frac{
\sum_c w_c B_{{\rm IR},c}
}{
\sum_c w_c T_{\pi,c}
},
}
\]

where \(B_{\rm IR}\) denotes the existing Pinsker numerator.

The aggregate bootstrap must recompute the entire ratio in each replicate.

Export:

```text
eta_ir
eta_ir_pinsker_numerator_bits
eta_ir_denominator_T_bits
```

Do not create a null-adjusted \(\eta_{\rm IR}\) in the first implementation. The denominator \(T_\pi-T_{\rm null}\) can be close to zero or negative and would produce an unstable quantity without a separate theoretical justification.

---

# 10. Support diagnostics for every aggregation

A headline aggregate is only useful if its support is transparent.

For every aggregated row export at least:

```text
n_contributing_cells
n_expected_cells
cell_coverage_fraction
n_episodes
n_round_observations
weighted_dual_action_state_fraction
weighted_dual_action_event_fraction
weighted_single_action_slice_fraction
weighted_singleton_fraction
conditioning_state_count_sum
min_cell_observations
max_cell_observations
support_status
```

For state-local aggregated maps also export:

```text
n_rho_contributing
n_target_semantics_contributing
n_observations
state_occupancy
```

Suggested support labels:

```text
adequate
limited
unsupported
```

Reuse the existing support rules where possible rather than inventing a second independent support system.

---

# 11. New tables

Add a derived study-level table, for example:

```text
analysis/tables/study_aggregated_metrics.parquet
```

Recommended schema:

```text
metric
estimate
ci_low
ci_high
units

aggregation_level
aggregation_weight
marginalized_dimensions

social_group_size
intervention_budget
epistemic_persistence          # null when marginalized
target_semantics               # null when marginalized

n_contributing_cells
n_expected_cells
n_episodes
n_observations
support_status

null_mean
null_sd
null_adjusted_estimate
permutation_p_value
n_null_draws

component_numerator
component_denominator

analysis_semantics_version
```

Also write a convenient CSV:

```text
analysis/tables/study_aggregated_metrics.csv
```

The Parquet table remains authoritative; CSV is for inspection and plotting.

---

# 12. State-local aggregation table

Create or repair:

```text
analysis/tables/state_local_aggregated_metrics.parquet
```

It should support:

### rho-marginalized

```text
(x, b, target_semantics)
```

### target-semantics-marginalized

```text
(x, rho, b)
```

### rho + target-semantics marginalized

```text
(x, b)
```

For every row retain:

```text
metric
estimate
x_bin_index
x_bin_lower
x_bin_upper
x_bin_center
b
rho
target_semantics
n_observations
n_contributing_cells
aggregation_weight
support_status
descriptive_only
```

For the state-local maps, observation/occupancy weighting should be the default descriptive aggregation.

---

# 13. Analysis recipe extension

Extend the study `analysis.yaml` with an explicit derived-aggregation section rather than hard-coding one particular study.

Illustrative design:

```yaml
derived_study_aggregates:
  enabled: true

  metrics:
    - round_target_actuation_cmi
    - susceptibility_occupancy_weighted
    - round_target_information_fraction
    - eta_ir

  groupings:

    - name: rho_aggregated_by_budget_target
      group_by:
        - intervention_budget
        - target_semantics
      marginalize:
        - epistemic_persistence
      weighting: balanced_cell

    - name: target_aggregated_by_rho_budget
      group_by:
        - epistemic_persistence
        - intervention_budget
      marginalize:
        - target_semantics
      weighting: balanced_cell

    - name: fully_aggregated_by_budget
      group_by:
        - intervention_budget
      marginalize:
        - epistemic_persistence
        - target_semantics
      weighting: balanced_cell

  state_local:
    enabled: true
    weighting: n_observations
    groupings:
      - group_by: [intervention_budget, target_semantics, target_fraction_bin_index]
        marginalize: [epistemic_persistence]

      - group_by: [epistemic_persistence, intervention_budget, target_fraction_bin_index]
        marginalize: [target_semantics]

      - group_by: [intervention_budget, target_fraction_bin_index]
        marginalize: [epistemic_persistence, target_semantics]

  bootstrap:
    unit: episode
    stratify_by_physical_cell: true

  null:
    aggregate_permutation_draws: true
```

Important: every scientific dimension **not explicitly marginalized** must remain a grouping coordinate.

Do not accidentally mix:

```text
different task IDs
different models
different q
different N
different controller policies
different prompt versions
```

unless a recipe explicitly requests that marginalization.

---

# 14. Plot set

The reaggregation should automatically produce the following.

## 14.1 Headline per-budget plots

For each budget \(b\):

\[
\bar T_\pi(b),
\qquad
\bar T_{\pi,\rm null}(b),
\qquad
\Delta\bar T_\pi(b).
\]

Plot:

```text
raw T_pi vs b
null T_pi vs b
T_pi - T_null vs b
```

with 95% bootstrap intervals.

Also plot:

```text
chi vs b
eta_IF vs b
eta_IR vs b
```

for the fully aggregated study-level summaries.

---

## 14.2 Persistence-resolved whole-cell maps

Produce \(\rho\times b\) maps for truth and false control separately:

```text
T_pi
T_pi - T_null
chi
eta_IF
eta_IR
support
```

These are based on whole-cell estimates and are statistically easier than the x-resolved maps.

---

## 14.3 rho-marginalized x × b maps

Produce:

```text
T_pi(x,b)
T_pi - T_null(x,b) if state-local null exists
chi(x,b)
eta_IF(x,b)
eta_IR(x,b)
occupancy/support
```

separately for:

```text
truth target
false target
truth + false pooled in target coordinates
```

These should use observation/occupancy weighting across rho.

---

## 14.4 Support overlays

Every phase-map family should have a matching support panel.

At minimum show:

```text
n_observations
dual-action support
number of contributing rho values / cells
```

Unsupported regions should be blank/hatched, not numerically zero.

---

# 15. Add a sample-size / stability diagnostic

This directly addresses the question:

> Is the transfer-information effect genuinely small, or are there not enough statistics?

This diagnostic does not create new information, but it shows whether the current dataset has stabilized.

Using the already completed episodes, run repeated **whole-episode subsampling** at available sample sizes.

For example, if 30 episodes per cell are available:

```text
N = 5, 10, 15, 20, 25, 30
```

For every \(N\):

1. subsample complete episodes without replacement inside each physical cell;
2. recompute the requested aggregate statistic;
3. repeat many times;
4. record its sampling spread.

Do this for:

```text
T_pi
T_pi - T_null
chi
```

at least for the fully aggregated per-budget level.

Export:

```text
sample_size
metric
budget
mean_estimate
sd_estimate
ci_width
sign_stability
fraction_permutation_significant
```

Recommended plots:

```text
CI width vs number of episodes
SD(T_pi - T_null) vs number of episodes
sign stability vs number of episodes
permutation-detection frequency vs number of episodes
```

Label this as:

```text
empirical sample-size stability diagnostic
```

not as a formal prospective power calculation unless the assumptions required for a formal power analysis are added.

---

# 16. Reporting hierarchy

The final report should separate **effect size**, **statistical precision**, and **finite-sample null bias**.

For transfer information report, in this order:

### A. Main statistically strongest result

\[
\boxed{
\Delta\bar T_\pi(b)
=
\bar T_\pi(b)-\bar T_{\pi,\rm null}(b)
}
\]

with aggregate permutation p-value and bootstrap CI.

### B. Raw transfer information

\[
\bar T_\pi(b)
\]

shown beside the null, never alone.

### C. Whole-cell \((\rho,b)\) structure

Show where the budget-level signal comes from across persistence.

### D. State-local \(x\times b\) structure

Use this as a mechanistic/descriptive phase-space view.

Do not require every state-local pixel to be individually significant before discussing the aggregate signal.

---

# 17. Truth + false aggregation semantics

Pooling truth-target and false-target conditions is allowed only in **controller-target coordinates**.

Use:

```text
controller target count
controller target fraction
m_ctrl
round_target_actuation_cmi
round_target_susceptibility
```

Do not pool via:

```text
truth count
m_truth
round_truth_actuation_cmi
```

for the target-agnostic control summary.

The report must state clearly:

> Truth-target and false-target conditions are pooled only after expressing the dynamics relative to the controller's selected target. This aggregate measures controllability toward a target and does not imply that truth and false control have identical epistemic consequences.

Always retain separate truth/false results as secondary diagnostics.

---

# 18. η_th remains unchanged

Do not infer or fabricate thermodynamic efficiency from these new aggregations.

If `eta_th` is unsupported because the blackboard actuator lacks a valid calibrated \(h\), leave it unsupported.

The new work concerns:

```text
T_pi
T_pi null
T_pi - null
chi
eta_IF
eta_IR
support
```

It does not solve the affinity calibration problem.

---

# 19. Suggested implementation location

Before adding new code, inspect the existing study aggregation architecture.

Likely relevant sources from `metrics.md`:

```text
src/mas_cc/studies/aggregation.py
src/mas_cc/studies/canonical.py
src/mas_cc/studies/validation.py
src/mas_cc/analysis/single_affinity.py
src/mas_cc/games/hidden_bench/imitation_round_feedback/analysis.py
```

Prefer adding a dedicated reusable study-derived aggregation module rather than embedding this logic in plotting code.

Possible location:

```text
src/mas_cc/studies/derived_aggregation.py
```

or an equivalent existing analysis module if one already serves this role.

Keep:

```text
estimation
aggregation
plotting
reporting
```

as separate steps.

---

# 20. Required tests

Add unit/integration tests for at least the following.

## Test 1 — Physical cells remain unchanged

Enabling study-level aggregation must not change the existing physical-cell estimates.

---

## Test 2 — Balanced rho aggregation

Given known cell values:

```text
rho1: T=0.1
rho2: T=0.3
```

balanced aggregation must give:

```text
0.2
```

regardless of different observation counts.

---

## Test 3 — Occupancy-weighted state-local aggregation

Given local estimates with different observation counts, verify the descriptive state-local aggregate uses the requested observation weights.

---

## Test 4 — No p-value averaging

Aggregate permutation significance must be derived from aggregate null replicates.

There must be no code path that averages cell-wise permutation p-values.

---

## Test 5 — η_IF ratio of sums

Verify:

```text
aggregate_eta_IF
!= mean(cell_eta_IF)
```

in a synthetic case where the denominators differ.

The expected result must be:

\[
\frac{\sum w_i T_i}{\sum w_i H_i}.
\]

---

## Test 6 — η_IR ratio of sums

Verify the same principle for the Pinsker numerator and transfer-information denominator.

---

## Test 7 — Target-coordinate truth/false pooling

Create truth- and false-target fixture cells with different semantic labels but equivalent target-coordinate dynamics.

The pooled target-coordinate estimate should be invariant to which semantic answer is called “truth”.

---

## Test 8 — Shard/order invariance

Changing scheduler shard IDs, file order, or cell discovery order must not change any study-level aggregate.

---

## Test 9 — Unsupported cells

If a requested contributing cell is unsupported, the aggregate must:

- record that fact;
- expose reduced cell coverage;
- follow a documented support rule;
- never silently replace the cell with zero.

---

## Test 10 — Bootstrap recomputation

Verify numerator/denominator ratios are recomputed inside each bootstrap replicate rather than combining separate confidence intervals.

---

# 21. Documentation update to metrics.md

Add a new section after the existing derived quantities, for example:

```text
7.4 Study-level aggregated control metrics
```

Explain in simple terms:

1. state-local estimates show where response/information occurs;
2. whole-cell estimates already average over visited current states;
3. study-level summaries can marginalize persistence and target semantics after physical-cell estimation;
4. balanced and occupancy-weighted aggregation answer different questions;
5. raw T_pi must be paired with its permutation null;
6. aggregate p-values come from aggregate nulls, not averaged p-values;
7. eta_IF and eta_IR must be recomputed from aggregate components;
8. truth and false control may only be pooled in controller-target coordinates.

Also add the new output tables to the “Main tables” section.

---

# 22. Reaggregate the current adaptive study

After implementation, rerun aggregation on the current adaptive-communication study **without making provider calls**.

Produce at least:

```text
A. per-cell:
   T_pi(rho,b,s)
   T_null(rho,b,s)
   T_pi-T_null
   chi
   eta_IF
   eta_IR

B. rho-marginalized:
   T_pi(b,s)
   T_null(b,s)
   T_pi-T_null
   chi(b,s)
   eta_IF(b,s)
   eta_IR(b,s)

C. target-semantics-marginalized:
   T_pi(rho,b)
   T_null(rho,b)
   T_pi-T_null
   chi(rho,b)
   eta_IF(rho,b)
   eta_IR(rho,b)

D. fully aggregated per budget:
   T_pi(b)
   T_null(b)
   T_pi-T_null
   permutation p-value
   chi(b)
   eta_IF(b)
   eta_IR(b)

E. state-local descriptive:
   T_pi(x,b,s), rho-marginalized
   chi(x,b,s), rho-marginalized
   eta_IF(x,b,s), rho-marginalized
   eta_IR(x,b,s), rho-marginalized

F. state-local truth+false pooled:
   T_pi(x,b)
   chi(x,b)
   eta_IF(x,b)
   eta_IR(x,b)

G. support:
   occupancy
   n observations
   dual-action support
   number of contributing cells/rho values
```

If a state-local permutation null cannot be reconstructed from current retained data, state that explicitly and omit `T_pi-T_null(x,b)` rather than inventing it.

---

# 23. Acceptance criteria

The extension is complete when:

```text
[ ] existing physical-cell estimates are unchanged
[ ] x-marginal whole-cell T_pi is clearly documented
[ ] rho-marginal per-budget metrics exist
[ ] truth/false target-coordinate aggregates exist
[ ] fully aggregated per-budget metrics exist
[ ] balanced and occupancy-weighted aggregation are distinct
[ ] aggregate null distributions are recomputed correctly
[ ] permutation p-values are never averaged
[ ] whole-episode stratified bootstrap works
[ ] chi aggregates include uncertainty
[ ] eta_IF is ratio of aggregate components
[ ] eta_IR is ratio of aggregate components
[ ] state-local rho-aggregated maps are regenerated
[ ] truth+false pooled state-local maps are generated
[ ] support/occupancy maps accompany phase maps
[ ] sample-size stability diagnostic is available
[ ] eta_th remains unsupported unless independently calibrated
[ ] metrics.md is updated
[ ] current adaptive study reaggregates without provider calls
[ ] plots and report compile successfully
```

---

# 24. Scientific interpretation intended by this extension

The extension should let us distinguish three separate statements:

\[
\boxed{\text{effect magnitude}}
\]

from

\[
\boxed{\text{statistical precision}}
\]

from

\[
\boxed{\text{finite-sample estimator/null bias}}.
\]

For transfer information specifically:

- `T_pi` = raw conditional predictive information;
- `T_null` = what the same estimator produces under the null with finite data;
- `T_pi - T_null` = null-adjusted excess transfer information;
- bootstrap width = how precisely the quantity is estimated;
- permutation p-value = how surprising the observed statistic is relative to the null;
- state-local maps = where the effect occurs;
- per-budget aggregates = the statistically stronger headline estimate.

The goal is therefore **not** to discard the \(x\times b\) phase diagrams. The goal is to put them in the correct hierarchy:

\[
\boxed{
\text{per-budget aggregate evidence}
\rightarrow
(\rho,b)\text{ structure}
\rightarrow
(x,b)\text{ mechanistic phase-space structure}.
}
\]

This should allow the existing simulations to support much stronger statistical conclusions without requiring every sparsely visited state-space pixel to be individually well estimated.
