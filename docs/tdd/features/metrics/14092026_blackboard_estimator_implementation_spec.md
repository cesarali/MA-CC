# Blackboard empirical estimator implementation specification

Date: 2026-09-14  
Status: Proposed implementation contract; not a claim that these estimators have been implemented or run.

## 1. Objective and evidence boundary

Add a reproducible offline calibration pipeline for the blackboard model's exposure probability, microscopic response channels, effective active-branch parameters, finite-round relaxation, and predicted susceptibility:

\[
w_b,\quad (p_0,\gamma_0,h_0),\quad(p_1,\gamma_1,h_1),\quad
(p_b,\gamma_b,h_b),\quad r_0,r_b,\quad\chi_b(x),\chi_{b,\mathrm{avail}}(x).
\]

The implementation must distinguish measured transition statistics from parameters inferred under the finite-compliance model and from predictions made using those parameters. It must preserve existing empirical susceptibility, causal-response, information, and efficiency estimators.

This specification is based on three supplied documents:

1. **agent_metrics_reference.md**, reconciled 2026-09-14: current documented estimator, retention, identity, resampling, and packaging contracts. Relevant sections: “Retention, identity, and canonicalization,” “Aggregation mathematics and estimators,” “Causal, communication, and symbolic families,” and “Recipe switches, products, and readiness.”
2. **blackboard_feedback_report_with_available_susceptibility(1).pdf**, dated 8 September 2026: §§4–5 define exposure mixtures and relaxation; §12.1 and §14 give empirical calibration procedures; §15 states model limitations. Its numerical examples are illustrative, not fitted blackboard results.
3. **iclr2027_conference.pdf**, Appendix C.2, pp. 25–27: existing single-affinity controlled-slot calibration. Reported values of approximately `h_eff = 3.83` and `gamma_eff = 0.372` must not be reused as fitted exposed/unexposed blackboard parameters.

Only these documents were audited. Repository code, companion documentation, recipes, and run artifacts were not inspected. All new names, fields, tables, and configuration keys below are **proposed**, unless explicitly identified as existing. The implementing agent must inspect the repository and read its referenced study-workflow skill before executing or changing study analysis.

## 2. Existing functionality and additions

| Capability | Documented current status | Required work |
| --- | --- | --- |
| `effective_affinity`, `kinetic_compliance` | Existing estimators restricted to controlled ADVOCATE micro-slots | Reuse verified counting primitives; preserve existing selection and semantics |
| Focal before/after transitions and exposure records | Documented in retained micro-records | Verify exact field mapping and sampling boundary |
| `controller_message_directly_exposed` | Existing streaming indicator | Verify it means at least one actual sampled controller message for the focal update before using it |
| Exposed/unexposed response calibration | Procedure specified in blackboard report; separate study outputs not listed | Implement new estimator family |
| Empirical and sampling-model exposure probability | Inputs partly documented; dedicated calibration output not listed | Implement observed frequency, sampling prediction, and residuals |
| Effective mixture and relaxation parameters | Formulas specified; named outputs not listed | Implement derived outputs with dependency and model checks |
| Empirical susceptibility, available causal susceptibility, MI/CMI, information efficiencies | Existing families documented | Join for validation; do not redefine or silently replace them |

Absence from the reference is not proof of absence from code. Complete a code inventory before adding duplicate functionality.

## 3. Definitions and population of updates

| Symbol | Meaning |
| --- | --- |
| `Z` | Controller target, with documented gold fallback only when applicable |
| `N` | Number of agents |
| `M` | Number of microscopic population updates in the modeled round; do not assume `M = N` |
| `U` | Round action: silence `0`, active posting `1` |
| `E` | Actual exposure: at least one controller-authored message sampled into the focal update's context |
| `z_before`, `z_after` | Indicators that the focal vote equals `Z` before and after its update |
| `x` | Population target share before the round for round predictions |
| `B_t`, `C_t` | Eligible peer and controller message counts immediately before update `t` samples its context |
| `m_t` | Actual number of messages sampled in update `t` |
| `b_budget` | Configured intervention cap or budget |
| `b_posted` | Realized number of newly posted controller messages in the round |
| `b` in the frozen-board theory | Number of controller messages eligible on that board, not automatically the configured budget or new-post count |

Count all eligible focal update opportunities, including unchanged votes. A switch between two non-target options remains `z_before = z_after = 0` and is not a target entry or exit.

Use completed canonical episodes only for final estimation. Failed, superseded, duplicate, or interrupted-prefix records must follow the existing validation/censoring contract. Never condition inclusion on whether a vote changed, whether a controller fact was acquired, or whether the focal agent posted afterward.

Keep exposure class `E` separate from controller action `U`. Persistent messages can produce `E = 1` in a silent round; active rounds can contain `E = 0`. Consequently, “unexposed channel” and “silent branch” are different data selections.

## 4. Required canonical inputs

Create a versioned adapter from retained records to the following logical schema. These are semantic requirements, not assertions that these exact column names already exist.

| Input group | Required information | Use |
| --- | --- | --- |
| Identity | Study/run/qualified-cell/episode/round/micro-slot identities; task; shared-initialization block | Deduplication, joins, resampling |
| Scientific condition | Prompt family, target and truth, `N`, persistence, board lifetime and sampling rules, communication mode, budget | Valid grouping and model applicability |
| Transition | Focal agent identity; vote immediately before and after its update; valid-update status | Entry/exit counts and opportunities |
| Exposure | Sampled message identities with author/source labels, or a verified equivalent `E` indicator | Observed exposure frequency and channel classification |
| Eligible board | Eligible peer/controller counts before sampling, actual sample size, exclusions | Per-update sampling-model exposure probability |
| Round | `U`, action propensity, `b_posted`, target counts before/after, actual update count | Branch comparison, causal validation, relaxation |
| Optional pre-action state | Active proof coverage, full-proof share, symbolic solvability, initial board signature | Diagnostics and prespecified heterogeneity checks |

Audit whether sampled messages include old controller posts, whether the focal agent's own posts are excluded, and when newly posted content becomes eligible. Total board size and a “message posted” flag alone do not establish actual exposure or eligible composition.

If valid `E` and transitions exist but eligible composition is missing, estimate empirical exposure and channel rates; mark the combinatorial comparison unavailable. Do not discard supported components because another component lacks inputs. Unknown exposure must remain unknown, never be converted to `E = 0`.

## 5. Estimator definitions

### 5.1 Observed exposure probability

Within a declared physical cell, action, budget/post-count group, and optional state slice, let `T` be the updates with valid exposure records:

\[
\widehat w=\frac{\sum_{t\in T}E_t}{|T|}.
\]

The primary active-branch quantity `w_b` uses `U = 1` and an explicit budget/post-count grouping. Emit separate silent-branch exposure frequency and unknown-exposure coverage. Record whether the estimate pools varying eligible boards; call such a value an observed average, not a fitted frozen-board constant.

Also emit exposure frequencies conditional on focal starting vote:

\[
\widehat w_{b,\mathrm{nonZ}}=P_n(E=1\mid U=1,z_{\mathrm{before}}=0),\qquad
\widehat w_{b,Z}=P_n(E=1\mid U=1,z_{\mathrm{before}}=1).
\]

These diagnose whether a single exposure weight can represent both entry and exit opportunities. Empirical frequencies are update-weighted by default. An equal-episode summary, if requested, must have a separate variant label.

### 5.2 Sampling-model exposure prediction

For uniform sampling without replacement from the eligible board:

\[
w_t^{\mathrm{sampling}}=1-\frac{\binom{B_t}{m_t}}{\binom{B_t+C_t}{m_t}}.
\]

Use actual `m_t`, with `0 <= m_t <= B_t + C_t`. An empty sample has exposure probability zero. When `m_t > B_t`, the numerator is zero. Compute the ratio stably using log-combinations or a product; do not overflow factorials.

Aggregate `w_t` over exactly the same rows and weights as observed exposure and emit `mean(E_t - w_t)`. Do not insert mean board counts into the nonlinear formula. For frozen `B`, `C = b`, and `m = min(q, B+b)`, this reduces to the report's `w_b`.

If the actual sampler is weighted, stratified, with replacement, or otherwise different, mark this formula inapplicable. Add a separately versioned sampler-specific prediction only after its law has been verified.

### 5.3 Microscopic channel transition rates

For exposure class `c` in `{0,1}`, define:

\[
D_{+,c}=\sum_t\mathbf1[E_t=c,z_{t,\mathrm{before}}=0],\qquad
A_{+,c}=\sum_t\mathbf1[E_t=c,z_{t,\mathrm{before}}=0,z_{t,\mathrm{after}}=1],
\]

\[
D_{-,c}=\sum_t\mathbf1[E_t=c,z_{t,\mathrm{before}}=1],\qquad
A_{-,c}=\sum_t\mathbf1[E_t=c,z_{t,\mathrm{before}}=1,z_{t,\mathrm{after}}=0].
\]

Then:

\[
\widehat p_{+,c}=A_{+,c}/D_{+,c},\qquad
\widehat p_{-,c}=A_{-,c}/D_{-,c}.
\]

Retain each numerator, denominator, number of contributing episodes/blocks, and missing-record count. Never divide both transition counts by all updates.

Initially emit channel estimates separately by `U`: silent-unexposed, active-unexposed, active-exposed, and silent-exposed where present. A shared unexposed calibration may pool actions only as an explicitly declared modeling assumption with branch-comparison diagnostics. Lack of a significant difference does not prove equality.

### 5.4 Channel preference, compliance, and affinity

For each supported channel:

\[
\widehat\gamma_c=\widehat p_{+,c}+\widehat p_{-,c},\qquad
\widehat p_c=\frac{\widehat p_{+,c}}{\widehat\gamma_c},\qquad
\widehat h_c=\log\frac{\widehat p_{+,c}}{\widehat p_{-,c}}.
\]

`gamma` is the model's revision-opportunity parameter. It is not the observed fraction of votes that switch; a model revision can leave the vote unchanged. `p_c` is a model-implied target preference, not the raw population target share.

Boundary behavior:

| Condition | Required output |
| --- | --- |
| Either opportunity denominator absent | Missing corresponding rate; joint parameters unsupported |
| Both rates zero with denominators present | `gamma = 0`; `p` and `h` unidentified; identity-kernel predictions remain possible |
| Entry rate zero, exit rate positive | `p = 0`, `h = -infinity`; boundary flag |
| Exit rate zero, entry rate positive | `p = 1`, `h = +infinity`; boundary flag |
| Both rates positive | Finite log-ratio |
| Rate sum greater than one | Retain descriptive rates and algebraic sum; flag incompatibility with the report's `gamma in [0,1]` model |

Do not clip raw estimates or silently apply pseudocounts. A constrained or regularized fit, if later added, must be a separate estimator variant with an explicit objective and prior/constraint. The reference notes unusual zero-rate behavior in the legacy affinity estimator; preserve compatibility there and implement mathematically signed boundaries in this new family.

### 5.5 Effective active-branch mixture

Under shared response channels and a common exposure weight:

\[
\widehat\gamma_b=(1-\widehat w_b)\widehat\gamma_0+\widehat w_b\widehat\gamma_1,
\]

\[
\widehat p_b=
\frac{(1-\widehat w_b)\widehat\gamma_0\widehat p_0+
\widehat w_b\widehat\gamma_1\widehat p_1}{\widehat\gamma_b}.
\]

Implement through rates to handle zero-compliance components safely:

\[
a_b=(1-w_b)p_{+,0}+w_bp_{+,1},\qquad
d_b=(1-w_b)p_{-,0}+w_bp_{-,1},
\]

\[
\gamma_b=a_b+d_b,\qquad p_b=a_b/\gamma_b,\qquad h_b=\log(a_b/d_b).
\]

Never evaluate `0 * NaN` from an unidentified component preference. A component with exactly zero mixture weight does not require a fitted rate. Missing support is not evidence for a zero weight.

Also calculate direct active-branch entry/exit rates from all `U = 1` updates and derive separate `direct_active` parameters. Compare them with `mixture_common_weight` results. These variants estimate different summaries when exposure depends on the starting vote or response channels differ across populations.

For a decomposition of observed active rates, use active-only channel rates and the respective starting-vote exposure weights. This reproduces the observed rate decomposition when support is complete, but it does not establish the report's common-weight homogeneous kernel. Report this as `mixture_start_vote_weighted`, not as validation of a single `w_b`.

### 5.6 Finite-round relaxation

For a supported homogeneous reference with fixed `N` and `M`:

\[
r_0=(1-\gamma_0/N)^M,\qquad r_b=(1-\gamma_b/N)^M.
\]

These are derived model quantities. Do not estimate them as observed persistence of individual votes. For zero compliance or zero updates, set `r = 1`. Never substitute budget `b` for update count `M`.

If `M` varies, compute predictions for each observed `M` and then aggregate under explicit weights. Do not exponentiate using a silently averaged `M`.

### 5.7 Susceptibility prediction and related coefficients

For branch `c`, write `a_c = p_{+,c}` and define the stable mean map:

\[
F_c(x)=r_cx+\frac{a_c}{N}\sum_{j=0}^{M-1}(1-\gamma_c/N)^j.
\]

For positive compliance this equals `p_c + (x-p_c)r_c`; for an identity kernel it equals `x` without needing an identified `p_c`.

Predict:

\[
\chi_b^{\mathrm{model}}(x)=F_b(x)-F_0(x)
=c_b+s_bx,
\]

\[
c_b=p_b(1-r_b)-p_0(1-r_0),\qquad s_b=r_b-r_0,
\]

using the stable mean-map form for zero-compliance branches. For `x < 1`:

\[
\chi_{b,\mathrm{avail}}^{\mathrm{model}}(x)=\frac{\chi_b^{\mathrm{model}}(x)}{1-x}.
\]

At `x = 1`, available susceptibility is undefined, not zero. Do not clip it to `[0,1]`. Optionally emit the zero-response point `x_star = -c_b/s_b` when `s_b != 0`, retaining out-of-range values and flags. If slope is zero, report either no root or all states as roots according to the intercept.

Predicted susceptibility must have a distinct name from measured `round_target_susceptibility` and propensity-weighted causal response. Predict at actual evaluation states before bin averaging; preserve each comparison's weighting and conditioning.

## 6. Model applicability and causal interpretation

The report's closed form requires a frozen eligible board, homogeneous channel response, a common exposure weight, and an unexposed silence baseline over `M` updates. Emit assumption diagnostics alongside predictions:

1. Eligible board composition and exposure probability variation within rounds.
2. Silent-round exposure to old controller messages.
3. Active versus silent unexposed transition-rate differences.
4. Exposure differences by focal starting vote.
5. Residual variation by target share, pre-action evidence state, round, task, or agent where supported.
6. Whether the actual focal selection rule supports the count-kernel approximation.

If the silent branch contains exposure, do not label its dynamics `K_0` merely because `U = 0`. Report direct silent-branch calibration separately. A generalized two-action mixture requires its own declared model variant. Time-varying boards require per-update kernels or another explicit extension; do not claim that the frozen-board matrix power remains exact.

Observed exposed/unexposed contrasts are descriptive calibrations. Exposure is generally post-treatment and not randomized. Randomized round action and recorded propensities identify action effects under the existing design, not automatically the causal effect of reading a controller message. Do not use post-action exposure to form purported causal treatment strata.

Use existing paired branch continuations from identical complete states where available, or the existing propensity-weighted round-response estimator with its eligibility rules, to validate predicted intervention-minus-silence response. Shared initial episode states alone do not establish identical later branch states.

## 7. Uncertainty, support, and calibration splits

Bootstrap whole episodes, or whole shared-initialization blocks when episodes are coupled. Preserve paired branches and all micro-updates in each sampled unit. Never bootstrap individual micro-slots independently. Resample separately within physical cells for cell estimates; any declared multi-cell calibration must retain cell strata and explicit weights.

In every replicate, recompute exposure, transition counts/rates, channel parameters, mixtures, relaxation, and predictions. This propagates dependence and uncertainty through the entire chain. Follow existing confidence/resampling configuration, with reproducible seeds and estimator versions. No null test is required for a parameter estimate; do not emit arbitrary p-values.

Record requested, valid, undefined, and boundary replicate counts per output. Do not silently discard infinite affinity draws to create a finite interval. Use an explicit boundary-aware interval policy; if too few valid replicates or independent units support an interval, emit a missing interval and reason. Degenerate nonparametric intervals at zero events must not imply certainty.

Maintain separate `support_status`, `model_status`, and `uncertainty_status`. Missing opportunity denominators imply unsupported joint parameters. Zero events with positive denominators imply a boundary estimate, not missing data. A single independent block cannot support an empirical between-block uncertainty estimate. Existing causal dual-arm thresholds do not automatically establish sufficient microscopic calibration support; document any new configurable thresholds before implementation.

Calibration/evaluation splitting must occur by independent episode/block, never by micro-slot. To test transfer across budgets, declare training and held-out budgets explicitly. Fit a prespecified shared channel model within a homogeneous scientific regime; do not fit a fresh parameter set to every evaluation heatmap cell and call the result predictive. Save split identities and weights. Pooled calibration is an explicit model fit, separate from ordinary per-cell study estimation.

## 8. Proposed API and output contract

Use one new analysis family with explicit dependencies. Suggested metric identifiers:

| Proposed metric | Scope |
| --- | --- |
| `blackboard_exposure_probability` | Observed exposure by action and declared group |
| `blackboard_sampling_exposure_probability` | Average sampler-predicted exposure |
| `blackboard_exposure_probability_residual` | Observed minus sampler-predicted exposure |
| `blackboard_target_entry_probability` | Per-opportunity target-entry rate |
| `blackboard_target_exit_probability` | Per-opportunity target-exit rate |
| `blackboard_channel_preference` | `p_0`, `p_1`, or action-specific channel variant |
| `blackboard_channel_compliance` | Corresponding `gamma` |
| `blackboard_channel_affinity` | Corresponding `h`, in nats |
| `blackboard_effective_preference` | `p_b`, with mixture/direct variant |
| `blackboard_effective_compliance` | `gamma_b`, with mixture/direct variant |
| `blackboard_effective_affinity` | `h_b`, with mixture/direct variant |
| `blackboard_round_relaxation` | `r_0` or `r_b` |
| `blackboard_model_target_susceptibility` | Predicted target-share response |
| `blackboard_model_available_susceptibility` | Predicted response divided by remaining non-target share |
| `blackboard_model_response_intercept` | `c_b` |
| `blackboard_model_response_slope` | `s_b` |
| `blackboard_model_zero_response_share` | Optional `x_star` |

Use channel, branch, and variant columns rather than overloading the same metric label with undocumented meanings. Suggested new Parquet audit tables:

| Proposed table | Contents |
| --- | --- |
| `blackboard_calibration_counts` | Exposure counts and directional opportunity/event counts |
| `blackboard_calibration_estimates` | Rates and channel/mixture/relaxation parameter estimates |
| `blackboard_calibration_diagnostics` | Missingness, sampler checks, support, model assumptions |
| `blackboard_model_predictions` | Evaluation-state predictions and parameter dependencies |
| `blackboard_model_validation` | Held-out empirical comparisons, residuals, and comparison provenance |

Populate or index the existing standard estimate tables consistently with repository conventions; avoid conflicting duplicate authoritative rows. Every estimate must retain study/run/qualified-cell identity, grouping and conditioning, estimator version/variant, estimate and interval, counts, units, support, analysis hash, source selection, and dependency IDs. Explicitly record training-group identity for predictions that use a multi-cell fit.

Do not join only on metric names. Preserve distinctions between unsupported, structurally absent, unvisited, boundary, model-incompatible, and provisional results.

## 9. Proposed recipe and execution flow

The following is a **schema proposal, not currently executable configuration**. Add parser/schema validation and documentation before advertising it as supported.

```yaml
blackboard_calibration_outputs:
  enabled: true
  estimator_version: blackboard_calibration_v1
  exposure_definition: sampled_controller_authored_message
  calibration_scope: physical_cell
  action_stratification: true
  shared_unexposed_baseline: false
  exposure_prediction: uniform_without_replacement
  active_parameter_variants:
    - direct_active
    - mixture_common_weight
    - mixture_start_vote_weighted
  model_predictions:
    enabled: false  # Enable with an explicit calibration and evaluation design.
    reference: frozen_board_homogeneous
  diagnostics:
    starting_vote_exposure: true
    silent_exposure: true
    unexposed_branch_comparison: true

resampling:
  bootstrap_resamples: 1000
  confidence: 0.95
  seed: 1
```

Execution order:

1. Validate source identities, completed episodes, required records, and semantic field mappings.
2. Canonicalize micro-updates and join verified pre-action round metadata.
3. Create declared scientific groups and independent calibration/evaluation splits.
4. Calculate exposure frequencies, sampler predictions, and transition counts.
5. Calculate action-stratified response channels and support diagnostics.
6. Calculate requested mixture/direct variants and model applicability flags.
7. Evaluate relaxation and susceptibility only for explicitly enabled reference variants.
8. Bootstrap the full dependency chain and compare with supported empirical estimands.
9. Publish tables and provenance through existing analysis packaging and finalization.

Integration points to inspect, from the reference: `src/mas_cc/analysis/effective_affinity.py`, `single_affinity.py`, `causal_response.py`; `src/mas_cc/studies/{canonical,validation,aggregation,table_io}.py`; relational game recording/runtime; and recipe models. Choose final module placement after inspecting these files. Estimation belongs in analysis, not report rendering.

Do not wire calibrated `h_b` directly into the old single-affinity thermodynamic efficiency. The blackboard report has a nontrivial silent branch and requires separate attribution assumptions. This specification adds calibration and susceptibility predictions; exact branch information kernels and thermodynamic decomposition require a separate, explicitly versioned extension.

## 10. Required validation and acceptance criteria

Tests must verify scientific semantics as well as arithmetic:

| Case | Expected result |
| --- | --- |
| Known counts: 20 entries / 100 non-target opportunities; 5 exits / 50 target opportunities | `p_plus = 0.2`, `p_minus = 0.1`, `gamma = 0.3`, `p = 2/3`, `h = ln(2)` |
| Non-target option changes and unchanged updates | Correct target coarse-graining and denominator inclusion |
| All boundary and missing-opportunity cases in §5.4 | Correct signed infinity, unidentified values, and support flags |
| Empty sample, no controller messages, sample larger than peer board | Exposure predictions respectively `0`, `0`, and `1` where applicable |
| Report fixture: `N=M=24`, `B=18`, `q=3`, `b=9`, `(gamma0,p0)=(0.35,0.70)`, `(gamma1,p1)=(0.65,0.90)` | `w ≈ 0.7210256410`, `gamma_b ≈ 0.5663076923`, `p_b ≈ 0.8655166169` |
| Equal compliance | Predicted raw susceptibility has zero slope in `x` |
| Zero exposure or identical channels | Mixture response relative to the shared unexposed baseline is zero |
| Zero compliance or zero updates | Identity mean map without `NaN` propagation |
| `x = 1` | Raw response remains evaluable; available response undefined |
| Action/exposure mismatch and persistent posts | Silent-exposed and active-unexposed records remain distinct |
| Starting-vote-dependent exposure | Direct rates match the conditional-weight decomposition; common-weight mismatch is visible |
| Variable `M`, board growth, and self-exclusion | No silent substitution of average counts or fixed-board exactness |
| Duplicates, superseded attempts, and interrupted prefixes | Existing canonicalization and provisional rules preserved |
| Shared-initialization bootstrap and held-out splits | Coupled episodes stay together; deterministic seed behavior; no leakage |
| Legacy estimator regression | Existing controlled-slot affinity/compliance semantics remain unchanged |

Acceptance requires: a completed source-field audit; validated new recipe selection; auditable counts for every estimate; explicit missing/boundary handling; reproducible block/episode uncertainty; truthful model status; held-out validation when predictive claims are made; and updated metrics/epistemics documentation. A successful command submission is not proof of completed analysis—inspect the published validation and analysis manifests.

Offline reaggregation is sufficient when retained data contain the required observations. If only eligible-board composition is missing, observed exposure and channel calibration may still be possible. New simulations are needed only for genuinely absent required measurements, unsupported scientific conditions, or a new experimental validation design. No numerical blackboard calibration is supplied by this specification.
