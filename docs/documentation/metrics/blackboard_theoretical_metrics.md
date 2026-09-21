# Calibrated blackboard theoretical metrics

`study aggregate` can calculate exact finite-state blackboard predictions after
empirical estimators, derived observables, causal analysis, and blackboard
calibration. This stage is opt-in and makes no provider calls. Existing empirical
tables and estimator versions are preserved; the new results live in separate
Parquet tables and are included in the normal analysis archive.

```yaml
# analysis.yaml
# Use this reference, or retain `none` and explicitly enable the family below.
theoretical_reference: calibrated_blackboard_finite_state_v1

blackboard_calibration_outputs:
  enabled: true
  shared_unexposed_baseline: true

blackboard_theory_outputs:
  enabled: true
  model: frozen_board_exposure_mixture_v1
  assume_homogeneous_channels: true
  evaluation_modes: [matched_empirical_design]
  matched_policy_source: empirical_action_frequency
  families:
    - target_response
    - causal_response
    - action_entropy
    - target_actuation_information
    - information_efficiencies
    - scalar_sensing
    - currents
  causal_lags: [1, 2, 3]
  uncertainty: joint_data_and_calibration
  comparisons: true
```

The assumption flag declares a homogeneous calibrated reference. It does not
establish that a persistent board or an LLM population is Markov in target count.
The exposure-mixture variant consumes the implemented `shared_unexposed` and
`mixture_common_weight` directional rates. It rejects missing opportunities,
compliance above one, exposed silent rounds, unverified uniform focal selection,
and missing or time-varying eligible board composition. It mixes microscopic
channels before taking the round matrix power. Actual retained population update
count `M` supplies that exponent, independently of posting budget.

Alternatively, `model: action_branch_calibrated` consumes `direct_silent` and
`direct_active` rates. This is a descriptive homogeneous branch reference, with
no exposure-based causal identification claim. It still requires the homogeneous
assumption, valid rates, and uniform focal selection. No branch uses the active
round's realized posts as the silent counterfactual.

Run the normal aggregation command after updating the recipe:

```bash
mas-cc study aggregate --study-dir <study-result-root>
```

Reaggregation can use retained canonical Parquet observations after source run
removal. The new analysis hash includes scientific input identity, calibration
hash, model version, settings, and resampling configuration.

## Outputs and empirical contracts

| Table | Contents |
| --- | --- |
| `theory_model_manifest` | Configuration, calibration dependencies, split blocks, policy source, evaluation identities, compatibility map |
| `theory_state_metrics` | Count-state occupancy, policy probability, both branch increments, susceptibility, information/entropy, Pinsker numerator, ratios, currents |
| `theory_primary_estimates` | Matched primary, causal and sensing metrics, or explicit missing-dependency statuses |
| `theory_derived_observables` | Component-based efficiencies and horizon/episode currents |
| `theory_empirical_comparison` | Recomputed existing empirical adapters on the evaluation slice, theoretical estimates, residuals, paired intervals |
| `theory_validation` | Numerical checks, model applicability and unsupported families |

The manifest contains a machine-readable compatibility map. Each comparison
records its metric, qualified cell, evaluation rows, mask, grouping, conditioning,
units, lag, policy, split, calibration IDs, model version, and analysis hash.
Empirical comparisons recompute the established estimator on the selected split;
they must not be joined to an all-data estimate by metric name alone.

* Susceptibility uses the exact target-count dual-action mask and renormalizes
  its occupancy mass. Signed actuation multiplies that response by `K/(K-1)`.
  Marginal signed response instead uses action-specific occupancies.
* Target information is exact `I(U; n_next | n_before)` in bits. Target information
  efficiency divides it by `H(U | n_before)`. Marginal action entropy and
  full-population conditional action entropy are separate quantities. The latter
  uses empirical frequencies within its full population key and is currently
  supported for `empirical_action_frequency` only.
* `eta_ir` divides the unrenormalized dual-action Pinsker numerator by target
  information. It exports identified occupancy and flags a numerator/denominator
  support mismatch under a policy source that assigns both actions to empirically
  single-action states. Model support never upgrades empirical support.
* Causal contrasts reuse the existing canonical propensity/endpoint eligibility
  adapter. Lag one is the branch difference without an additional propensity
  factor. Later lags intervene on the first action only, then recompute feedback
  at reached states. Available susceptibility excludes `x=1`: the eight-bin local
  result is a mean of ratios, while the cell summary is a ratio of sums.
* Matched controlled currents mirror the existing pooled state-policy frequencies
  and dual-action mask, then average across round indices or sum over the horizon.
  Expected episode currents instead propagate the complete dynamics from each
  episode's initial count and compare final-minus-initial counts. Cell values
  average episodes equally.
* Exact sensing reuses the existing hypergeometric engine. The nats calculation
  already used by `target_sensing_information_nats` is the same calculation, not
  an independent validation. Sensor MAE/MSE use the existing target-fraction error.

The exact joint-law engine supports explicit context and outcome mappings,
including coarse bins, and validates stochastic kernels, information bounds and
entropy/current identities. The published adapter currently implements exact
count conditioning and the existing eight-bin available-susceptibility contract.
It does not silently project arbitrary empirical conditioning recipes onto counts.

## Policy, sensing and forward predictions

Policy sources never fall back to one another:

* `empirical_action_frequency`: observed action frequency at the exact target count.
* `recorded_propensity_average`: average retained pre-action probability at that count.
* `sensor_policy_exact`: integrate the explicitly configured sensor and feedback law.
* `complete_policy_table`: use `complete_policy_table: [...]`, with `N+1` probabilities.

For scalar sensing and an exact sensor policy, provide `sensor_policy` containing
`sampling_law: uniform_without_replacement`, `rule: logistic_target_fraction`,
`q_c`, `beta`, and `theta`. Use the study's actual values. Scalar comparisons check
recorded sensor sample sizes; the exact policy source additionally verifies the
recorded `soft_target` policy and parameters. The pure sensor engine handles
`q_c=0` only with an explicit `no_observation_probability`; an empirical mirror
still requires compatible recorded policy metadata.

A multi-round calculation requires `assume_count_markov: true` plus a complete
policy on future states. Missing empirical policy states produce a status rather
than extrapolation. Unmodeled board/evidence history remains an explicit coarse
model assumption even with this flag.

To request an autonomous forward ensemble, add `forward_model_ensemble` to
`evaluation_modes`, set `forward_policy_source` to `sensor_policy_exact` or
`complete_policy_table`, and supply `forward_initial_occupancy` (an `N+1` probability
vector) and positive integer `forward_horizon`. Forward outputs carry their own
mode and occupancy provenance and are not paired with matched empirical estimates.
Forward total currents telescope to expected final-minus-initial count. Accumulated
local excess is not labeled as the global feedback-versus-all-silent intervention.

## Uncertainty and limitations

The existing calibration pipeline's retained shared-initialization blocks are
resampled and refitted using its own calibration functions. With
`joint_data_and_calibration`, evaluation occupancy, empirical policy frequencies,
empirical estimates, theoretical components, and paired residuals are recomputed
in every replicate. In-sample training/evaluation use the same block multiplicities;
held-out splits remain disjoint. `calibration_only` holds evaluation rows fixed;
`none` publishes point predictions. Resampling count, confidence and seed come
from the normal recipe `resampling` section. Insufficient blocks or valid draws
produce unavailable intervals. Marginal parameter CIs are never treated as an
independent Gaussian parameter distribution. Bootstrap intermediates are transient.

Without a calibration holdout the output is `in_sample_calibrated`. To obtain a
held-out comparison, enable the existing calibration `model_predictions` settings
with an explicit `evaluation_fraction` and `assume_homogeneous_channels: true`.
Theory consumes that actual block split; it never relabels an in-sample fit.

Optional `cross_cell_weights` maps qualified cell IDs to positive weights. It
aggregates already computed efficiency numerators and denominators, retaining
source IDs and rejecting missing/duplicate components. Cross-cell intervals are
unavailable until a study-level joint block plan is supplied; per-cell intervals
cannot reconstruct that covariance. No aggregate transition matrix is manufactured.

Current unsupported adapters emit reasons: varying update counts within an
evaluation slice require context-specific branch laws; full-vector sensor-action
MI requires the full sensor law; full-population/evidence/truth/order mirrors
require additional state. The optional synthetic estimator, communication cost,
and path-thermodynamic families are not implemented. Empirical MI retains its
finite-sample behavior; exact model MI has no empirical null offset. Residuals
include calibration error, empirical estimation error, and misspecification.

Tests reproduce the specification's `N=M=24` fixture within `1e-8`, exercise
policy/lag/entropy/binning/normalization semantics, and run an offline mock
aggregation and retained-data packaging regression. These are analytical and
mock fixtures, not fitted results from a production study. No study-specific
SLURM launcher or replacement empirical CMI estimator was added.
