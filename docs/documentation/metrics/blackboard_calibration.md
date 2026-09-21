# Offline blackboard calibration

`blackboard_calibration_v1` implements the physical-cell calibration and held-out
reference-prediction path in the [implementation specification](../../tdd/features/metrics/14092026_blackboard_estimator_implementation_spec.md).
Select it in a study's `analysis.yaml`, then use the existing `mas-cc study
aggregate --study-dir <study>` command. No provider calls are needed when the
required canonical observations are retained.

```yaml
theoretical_reference: none
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
    enabled: false
  diagnostics:
    starting_vote_exposure: true
    silent_exposure: true
    unexposed_branch_comparison: true
resampling:
  bootstrap_resamples: 1000
  confidence: 0.95
  seed: 1
```

The selector is a validated mapping. Unknown settings and unsupported estimator,
sampler, pooling or reference variants raise errors. `exposure_prediction: none`
disables the combinatorial comparison. Diagnostics are mandatory in v1.

## Source-field audit (2026-09-14)

The versioned adapter is `blackboard_retained_updates_v1` in
[`blackboard_calibration.py`](../../../src/mas_cc/analysis/blackboard_calibration.py).
The existing `effective_affinity.py` and `single_affinity.py` count only controlled
ADVOCATE slots. Their selection, bootstrap stabilization and legacy zero-rate
behavior cannot be reused for all board updates; those implementations remain
unchanged. Existing causal-response input validation is reused for held-out
randomized-action comparisons. No information estimator is added or replaced.

| Logical input | Audited source and semantics |
| --- | --- |
| Completed identity | `studies/canonical.py` selects completed/skipped-resumed episodes and the final trajectory per coordinate before this analysis. Adapter joins qualified `cell_id`, `episode_id`, `round_index`, `micro_slot_index`; unexpected duplicates fail instead of guessing retry order. Interrupted prefixes are excluded. |
| Target | Round `analysis_target`, then explicit controller target; runtime supplies the gold fallback in `analysis_target`. Micro target disagreement fails. No inferred gold fallback from option order. |
| Transition | `focal_opinion_before/after` from `game.apply_round_event_transition`, immediately surrounding the focal update. All unchanged updates and non-target-to-non-target changes remain opportunities. |
| Action | Round `U_k` / `controller_sampled_U`, with explicit action-label fallback. Missing action stays unknown. Action and exposure are separate. |
| Exposure | `sampled_controller_message_ids` from `social_sources` with `author_kind == controller`, including still-live old posts. JSON strings and native lists are supported; missing/malformed records remain unknown. |
| Direct recommendation | `controller_message_directly_exposed` specifically means a **transient** control source. It is not an E indicator. Because legacy ID lists also include transient recommendations, these updates have unknown board-only E and an inapplicable sampler comparison. |
| Eligible board | `runtime.py` computes live messages immediately before sampling, after any controller insertion and after configured focal-self exclusion. New retained `eligible_peer_message_count` and `eligible_controller_message_count` count all author kinds, including controller REQUESTs, REPORTs and DIRECTIVEs. |
| Sampling | `BlackboardState.sample_live` uses `rng.sample(eligible, min(count,len(eligible)))`. New `board_sample_size` is the actual board sample size, before any transient source insertion. `board_sampling=uniform` is required for the hypergeometric reference. |
| Eligibility timing | Dawn posts are eligible for subsequent updates. Microscopic controller posts become eligible immediately before that update's sample. A participant's new post occurs after its update, so it can affect later updates. Live-message expiration is round-based. |
| N and M | N comes from round/cell metadata. M counts all canonical micro-update records in each actual round; new `actual_update_count` verifies this count. The current runtime executes N updates, but analysis never substitutes N or budget for M. |
| Focal selection | Board runtime draws one focal agent uniformly with replacement per update; new `focal_selection_rule` records this. Older records remain unverified for this assumption. |
| Budget/posts | `intervention_budget` is the group coordinate; `actual_controller_posts` / `controller_posts` is a separate realized-post diagnostic. Neither is substituted for eligible C. |
| Independent units | `physical_initial_state_hash`, then `initialization_artifact_hash`; otherwise the qualified episode identity. Paired-artifact data without a block hash fail. Every episode's block must be constant. |
| State/outcome | Round target shares checked against retained target-oriented count vectors when available. Task, agent, round, starting share and realized posts are diagnostic slices, not exposure-based causal strata. |

`results_only` previously retained message IDs but omitted eligible composition.
Those archives still support exposure and transition calibration. Their sampling
comparison is missing, not zero. New micro fields are retained by the recorder
and public semantic stream; canonical tables preserve them for offline reuse.

## Estimands and support

Each fit groups by qualified physical cell and configured budget. The primary
exposure estimate is update-weighted, with separate action and focal-start-vote
rows. Sampling averages use the exact subset with both known E and valid eligible
composition. `sampling_n`, `sampling_observed` and `residual` expose that matching;
no formula is evaluated at mean board counts.

Channel counts retain `D_plus`, `A_plus`, `D_minus`, `A_minus`, episode/block counts
including directional contributors, unknown exposure and missing transitions.
Entry divides by non-target starts; exit divides by target starts. Channels are
stratified by both action and exposure. Direct active/silent rates also retain
valid transitions with unknown exposure.

`gamma=a+d`, `p=a/gamma`, `h=log(a/d)`. No clipping or pseudocounts are applied.
Missing directional opportunities leave joint parameters unsupported; zero rates
with opportunities are boundaries. Both-zero rates identify gamma=0 but not p or
h. Entry-only and exit-only kernels have positive and negative infinite affinity,
respectively. Gamma greater than one remains a descriptive algebraic result with
`incompatible_gamma` model status.

`mixture_common_weight` uses the active average exposure and active channel rates.
`shared_unexposed_baseline: true` explicitly pools unexposed actions and publishes
a separately identified shared calibration. `mixture_start_vote_weighted` always
uses active-only channels and directional exposure weights. It decomposes direct
rates when exposure coverage is complete; it does not validate a homogeneous
common-weight kernel. Zero-weight components do not require fitted parameters.

Whole blocks are resampled within each physical group. Each draw recomputes the
counts, exposures, rates, mixtures and enabled predictions. There are no p-values
or retained bootstrap draws. The fixed v1 interval policy requires at least two
contributing independent units, two valid draws and 80% valid requested draws.
It uses extended-real empirical order statistics (including signed infinities).
Undefined and boundary replicate counts are retained. A constant zero-event
boundary bootstrap receives a missing interval with `degenerate_boundary`, not a
claim of certainty. These are computational support thresholds, not guarantees
of adequate statistical power.

## Held-out reference predictions

Enable predictions only with an explicit design:

```yaml
blackboard_calibration_outputs:
  enabled: true
  model_predictions:
    enabled: true
    reference: frozen_board_homogeneous
    evaluation_fraction: 0.3
    assume_homogeneous_channels: true
```

A reproducible split holds out whole initialization blocks within each physical
cell/budget before fitting. `blackboard_calibration_splits` saves identities,
assignment, seed and update weights. With predictions enabled, calibration rows
refer to training blocks only. At least one training and one evaluation block
are required; one-block data produce unsupported predictions. This is within-cell
held-out evaluation, **not evidence of transfer across budgets**. V1 deliberately
rejects multi-cell/prespecified-budget pooled fits and equal-episode variants;
those require a separately declared fitting and weighting contract.

The stable finite-round mean map uses each evaluation round's actual N, M and x.
Zero compliance and zero updates give the identity map; x=1 has a defined raw
response and undefined available susceptibility. No response is clipped.
Prediction rows retain relaxation, intercept, slope, root and root status,
training identity and rate dependency IDs. Affinity is never fed into the legacy
single-affinity thermodynamic efficiency.

Predictions are explicitly assumed homogeneous references. Diagnostics identify
changing eligible boards, varying M, unverified focal selection, silence exposure,
branch differences among unexposed updates and starting-vote exposure differences.
Exposed silence suppresses the frozen-unexposed-baseline prediction. Time-varying
boards are labeled reference-only; the matrix-power formula is not claimed exact.

Held-out validation joins canonical round identities with the existing
`build_causal_response_inputs(..., lags=(1,))` eligibility and HT contribution.
It compares the predicted response at each actual state before forming a
round-weighted residual summary, with both-arm support diagnostics. Summary
uncertainty independently resamples training and evaluation blocks within their
fixed splits and recomputes the full prediction chain. Shared initial states do
not establish identical later branch states; no paired-continuation claim is made.
Exposure contrasts are descriptive post-treatment calibrations, not causal
reading effects. Available model response rows are not replacements for the
existing causal available-mass ratio or empirical state-local susceptibility.

## Published products

All products are compressed Parquet under `analysis/tables`:

| Table | Role |
| --- | --- |
| `blackboard_calibration_inputs` | Versioned joined micro-update audit inputs |
| `blackboard_calibration_splits` | Saved block split and weights |
| `blackboard_calibration_counts` | Opportunity/event counts and matched exposure support |
| `blackboard_calibration_estimates` | Long-format measured and inferred parameters with intervals |
| `blackboard_calibration_diagnostics` | Missingness, model assumptions and heterogeneity slices |
| `blackboard_model_predictions` | Held-out reference predictions with rate dependencies |
| `blackboard_model_validation` | Matched HT contributions and joint-bootstrap residual summaries |

Calibration estimates are also indexed in `primary_estimates` with the same
`estimate_id`; they are not distinct estimands to count twice. Prediction tables
are authoritative for evaluation-state outputs. Every table records estimator
version, analysis hash and provisional status. Study validation and analysis
manifests include family identity and row counts. Standard packaging includes
these products, with no new SLURM launcher or persistent resampling cache.

Tests cover arithmetic/boundaries, action/exposure separation, incomplete and
duplicate records, deterministic block splitting, uncertainty and a real mock
study through retention, canonicalization, Parquet publication, ZIP packaging,
and offline reaggregation after removing the source run tree. These tests are
implementation evidence; no real blackboard calibration result is asserted.
