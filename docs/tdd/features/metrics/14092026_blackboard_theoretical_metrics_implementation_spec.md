# Theoretical counterparts of empirical blackboard metrics

Date: 2026-09-14  
Status: Implementation specification for a new theoretical analysis layer.  
Prerequisite: The user reports that the preceding blackboard calibration specification has been implemented. Inspect that implementation and consume its actual interfaces and outputs; do not reimplement it from assumed names.

## 1. Goal and scope

Use calibrated exposure and transition-channel parameters to construct exact finite-state blackboard kernels and compute theoretical counterparts of susceptibility, propensity-weighted causal response, action entropy, target actuation information, sensing information, information efficiencies, and currents. Produce empirical/theoretical comparison rows with identical estimand definitions, conditioning, state selection, units, horizons, and weights.

These are **calibrated model predictions**, not new empirical measurements. Exact finite-state summation removes simulation error conditional on the model; it does not remove calibration uncertainty, finite-data estimator bias, or model misspecification.

Default deliverable: an offline analysis family and comparison package, using existing calibration estimates and canonical observations. No provider calls or new LLM experiments are required when dependencies exist. This document specifies implementation; it supplies no fitted values from the user's runs.

Sources and authority:

| Source | Relevant material |
| --- | --- |
| `agent_metrics_reference.md`, reconciled 2026-09-14 | Existing metric definitions, conditioning, causal lag estimand, available normalization, resampling, joins, and aggregation |
| `blackboard_feedback_report_with_available_susceptibility(1).pdf`, 8 September 2026 | §§4–6: kernels, susceptibility, information and efficiencies; §7: excess current; §§8–11: branch path identity; §§12–15: calibration and limitations |
| `iclr2027_conference.pdf`, Appendix C.2 | Earlier controlled-slot calibration; not a substitute for blackboard branch calibration |
| `blackboard_estimator_implementation_spec.md`, preceding conversation deliverable | Calibration design; implementation names must now be verified against code |

The attached reference and reports were reviewed, but current repository code and the user's newly implemented calibration outputs were not available for verification. All new identifiers and schemas below are **proposed**. Existing implementation governs empirical behavior when documentation differs. Read applicable repository instructions and the referenced study-workflow skill before modifying or running study analysis.

## 2. Mandatory distinctions

1. **Exposure channel versus action branch:** channel `c=1` means actual controller-message exposure; action `u=1` means activation. The active round contains an exposure mixture. Silent rounds may contain old controller posts.
2. **Preference versus occupancy:** use `p_0,p_1,p_b` for channel preferences and `omega_k(n)` for pre-round population occupancy. Do not confuse preference `p_0` with an initial state distribution.
3. **Entropy versus horizon:** write `H_action` and `H_action_given_state` for entropy, `L` for the analysis horizon, and `ell` for a causal lag.
4. **Micro-update count versus posting budget:** round kernels use `M` population updates; `b` is not the exponent unless the actual protocol independently establishes `M=b`.
5. **Local information versus binned information:** averaging `I(U;n'|n)` inside an x-bin differs from recomputing `I(U;n'|bin(n))`.
6. **Model estimand versus empirical estimator:** exact theoretical MI is not the expected finite-sample output of a smoothed direct-counting estimator. Compare them transparently; do not label their difference solely as model error.

## 3. Computation modes

Implement the first two modes in the initial release. The third is optional but useful for diagnosing estimator bias.

| Mode | State distribution and policy | Interpretation |
| --- | --- | --- |
| `matched_empirical_design` (default) | Actual empirical evaluation rows, masks and weights; explicitly selected policy source | Model counterpart evaluated where the experiment was observed |
| `forward_model_ensemble` | Declared initial occupancy evolved under theoretical feedback kernels | Model trajectory prediction; occupancy can diverge from observations |
| `synthetic_estimator_check` (optional) | Synthetic episodes from the model, processed through the unchanged empirical estimator | Finite-sample bias, null behavior, and coverage diagnostics |

Store policy source separately:

- `empirical_action_frequency`: use observed advocacy frequency in the exact empirical conditioning state; this matches descriptive CMI/efficiency components documented in the reference.
- `recorded_propensity_average`: average recorded pre-action probabilities in each declared state. This is a design-based policy estimate, not the same finite-sample number as observed action frequency.
- `sensor_policy_exact`: integrate the declared sensor kernel and feedback law. Required for autonomous forward predictions unless a complete policy table is supplied.

Never silently fall back between policy sources. The multi-lag causal counterpart requires a future policy defined at every reachable model state. An empirical policy table supported only at visited states cannot support an unrestricted rollout; report missing policy mass/states rather than invent extrapolation.

Preserve empirical support separately from model availability. A model may predict an unvisited or single-action state while the empirical comparison remains unsupported. Do not mark empirical support adequate because the model supplies both kernels.

## 4. Inputs and dependency resolution

Consume the implemented calibration outputs: directional rates or `(gamma,p)` for both exposure classes, selected mixture/direct branch variants, exposure estimate or sampler model, covariance-preserving resampling inputs, calibration regime, training identities, support, and model-applicability flags.

Also require:

- `N`, number of answer options `K_options`, target/gold identity, round update count `M`, budget and realized/eligible board semantics.
- Qualified study/run/cell/episode/round identities and shared-initialization blocks.
- Canonical target counts, action, propensity, empirical grouping/conditioning definitions, bin edges, eligible masks, lag endpoint availability, horizon, and aggregation weights.
- Exact sensor/policy configuration when requested: sample size `q_c`, sampling law, feedback rule and parameters, plus all policy-relevant pre-action covariates.
- Calibration/evaluation split and provenance. Same-data calibration is allowed for a descriptive fitted reference, but must be labeled `in_sample_calibrated`; held-out prediction requires an actual disjoint split.

Join dependencies by complete scientific identity, grouping, conditioning, calibration variant/version and analysis hash. Missing dependencies must produce explicit per-family statuses. Parameters alone are insufficient to determine cell-level `T_pi`, entropy or efficiencies: those also require action probabilities and occupancy.

Do not use an observed active-branch post count, exposure sequence, or later board trajectory as the silent branch's counterfactual covariate. If realized posting depends on the sensor, action, or hidden state, supply a valid branch-specific distribution or label an explicitly conditional descriptive calculation. A fixed `b` from the report is a model condition, not a license to condition causal comparisons on post-treatment values.

## 5. Exact kernel engine

### 5.1 Microscopic kernel

Use row-stochastic matrices, with source count in the row and destination count in the column. For `n=0,...,N`:

\[
K_{\gamma,p}[n,n+1]=\gamma(1-n/N)p,
\qquad K_{\gamma,p}[n,n-1]=\gamma(n/N)(1-p),
\]

\[
K[n,n]=1-K[n,n+1]-K[n,n-1].
\]

Omit transitions outside `[0,N]`. Construct directly from entry and exit rates `a=gamma*p`, `d=gamma*(1-p)` when preference is unidentified at zero compliance. A zero-rate channel produces the identity kernel. Reject invalid/missing calibration for the selected model; do not silently clip an observed `gamma>1` into the report's `[0,1]` model.

For the report's frozen-board, shared-unexposed-baseline model:

\[
K_b=(1-w_b)K_0+w_bK_1,\qquad Q_0=K_0^M,\qquad Q_1=K_b^M.
\]

Here `K_1` is the exposed microscopic channel; `Q_1` is the complete active round. Avoid naming ambiguity in code (`K_exposed`, `Q_active`). Never replace `Q_1` by `(1-w_b)K_0^M+w_bK_1^M`: mixing per update and mixing whole rounds are different models.

Permit an explicitly named `action_branch_calibrated` variant using directly fitted silent/active microscopic kernels. This can support information and mean-response calculations but is not automatically the report's exposure-based mechanistic model or an identified causal branch law.

If frozen-board assumptions fail, do not call the matrix-power result exact for the experiment. A time-varying extension uses an ordered product of branch kernels derived from a valid exogenous schedule or modeled state; state-dependent schedules require an adequate enlarged state. Replaying a realized post-treatment schedule under both actions is not generally a counterfactual model.

### 5.2 Policy mixture

For a declared policy `a_k(n)=P(U_k=1|n)`:

\[
P_k[n,m]=(1-a_k(n))Q_{0,k}[n,m]+a_k(n)Q_{1,k}[n,m].
\]

Policy probabilities scale **rows**, not columns. For the report's scalar sensor:

\[
S(y|n)=\frac{\binom{n}{y}\binom{N-n}{q_c-y}}{\binom{N}{q_c}},
\qquad a(n)=\sum_y S(y|n)\pi(1|y).
\]

Use the actual implemented policy. For the report fixture only, `pi(1|y)=sigmoid(beta*(theta-y/q_c))`. If `q_c=0`, require the configured no-observation policy instead of dividing by zero. Reject a hypergeometric assumption for a different sampling law.

The simple factorization assumes `Q_u` depends on the sensor only through `u`. If budget, message choice, or compliance depends further on `y`, implement `Q_{u,y}` and marginalize with the correct conditional sensor distribution, or mark that variant unsupported. Do not average over `y` uniformly.

### 5.3 General joint-law adapter

For empirical pre-action context `s` with mass `v_s`, target count `n_s`, propensity/policy probability `a_s`, and branch row `Q_{u,s}[n_s,m]`, construct:

\[
J(c,u,m)=\sum_{s:C(s)=c}v_s\,a_s^u(1-a_s)^{1-u}Q_{u,s}[n_s,m].
\]

`C(s)` is the **exact empirical conditioning key**, and `sum_s v_s=1` within the evaluation slice. Compute MI, entropy and conditional response from this joint law when covariates vary. It preserves action-dependent context composition and prevents incorrect averaging of nonlinear statistics.

For a causal contrast, use the same pre-action context weights in both branches, rather than conditioning separately on realized action. When branch response is heterogeneous within `n`, the descriptive conditional contrast and standardized causal contrast can differ. Preserve that distinction in metrics and Pinsker numerators.

## 6. Metric mapping

Use proposed `theory_` identifiers, with `empirical_metric` and `estimand_variant` metadata. Resolve final names against code. A request for `H` must yield explicitly named entropy quantities, never one ambiguous scalar.

| Existing empirical metric/family | Proposed theoretical counterpart | Requirement |
| --- | --- | --- |
| `round_target_susceptibility` | `theory_round_target_susceptibility` | State-matched mean target-share contrast; same dual-action mask/weights for comparison |
| `round_target_signed_actuation` | `theory_round_target_signed_actuation` | Multiply matching fraction contrast by `K_options/(K_options-1)` |
| `round_target_signed_response_share` | `theory_round_target_signed_response_share` | Marginal action-conditioned mean increment difference, not standardized susceptibility |
| `propensity_weighted_causal_response` | `theory_propensity_weighted_causal_response` | Expected IPW estimand at lag `ell`; §§7–8 |
| Available causal susceptibility tables | `theory_available_causal_susceptibility_*` | Preserve local mean-of-ratios versus cell ratio-of-sums |
| `round_target_actuation_cmi` / `T_pi` | `theory_round_target_actuation_cmi` | Exact conditional channel information in bits |
| `round_controller_action_entropy` | `theory_round_controller_action_entropy` | Marginal `H(U)` |
| `round_controller_action_entropy_given_population` | `theory_round_controller_action_entropy_given_population` | Match full conditioning key; target-count reduction requires sufficiency |
| Target conditional action entropy audit component | `theory_action_entropy_given_target` | `H(U|n_Z)`; explicit denominator when appropriate |
| `round_target_information_fraction` / `eta_IF` | `theory_round_target_information_fraction` | Ratio of matching information and entropy components |
| `eta_ir`, `eta_ir_state_local` | `theory_eta_ir`, `theory_eta_ir_state_local` | Same-state Pinsker numerator / matching `T_pi` |
| `round_target_sensing_mi` | `theory_round_target_sensing_mi` | Scalar count sensor law and occupancy |
| `round_sensor_action_mi` | `theory_round_sensor_action_mi` | Sensor-policy joint law with matching sensor representation |
| `round_sensor_mae`, `round_sensor_mse` | `theory_round_sensor_mae`, `theory_round_sensor_mse` | Exact expectation of the existing sensor-error definition and units |
| `controlled_current`, horizon counterpart | `theory_controlled_current`, `theory_controlled_current_horizon` | Controller-excess current, not all active-branch motion |
| `episode_current`, `cell_current` | `theory_expected_episode_current`, `theory_expected_cell_current` | Full forward expected final-minus-initial target count |
| Sensing information in nats/horizon | `theory_target_sensing_information_nats` and horizon counterpart | Exact scalar sensor; conversion and horizon sum |

Existing aliases must resolve to one canonical estimand without duplicate conflicting output rows.

## 7. Response and available normalization

Let `f(n)=n/N`, `d_u(n)=(Q_u f)(n)-f(n)`. Then:

\[
\chi(n)=(Q_1-Q_0)f(n)=d_1(n)-d_0(n).
\]

Cross-check against the calibration closed form:

\[
\chi(x)=p_b(1-r_b)-p_0(1-r_0)+(r_b-r_0)x,
\quad r_c=(1-\gamma_c/N)^M.
\]

Use matrix/stable rate-based formulas for zero-compliance cases. Exact-state susceptibility is independent of `a(n)`; policy changes its visitation, aggregation, and information consequences.

For an empirical state-matched summary, average the matching conditional contrasts over the same eligible state/event mass. If heterogeneous contexts induce selection within a state, derive the descriptive branch conditional means from `J`, not by averaging standardized context contrasts and renaming them.

For the marginal response-share estimator:

\[
R_{\mathrm{marg}}=E_J[f(n')-f(n)\mid U=1]-E_J[f(n')-f(n)\mid U=0].
\]

Compute with action-specific state/context occupancies. Generally it differs from `sum_n omega(n)*chi(n)` under feedback.

For causal rows `t` with `x_t<1` and supported lag endpoints, let `delta_{t,ell}` be the theoretical action contrast in §8. With normalized empirical row weights `v_t`:

\[
\chi_{\mathrm{avail,local}}^{\mathrm{theory}}
=\sum_t v_t\frac{\delta_{t,ell}}{1-x_t},
\]

\[
\chi_{\mathrm{avail,summary}}^{\mathrm{theory}}
=\frac{\sum_t v_t\delta_{t,ell}}{\sum_t v_t(1-x_t)}.
\]

Apply the same exclusion and grouping rules as the empirical implementation. Exclude `x=1`; never encode undefined normalization as zero or clip a result exceeding one. If existing available susceptibility supports only one-round outcomes, keep that mapping and label additional lags as new variants.

## 8. Theoretical counterpart of propensity-weighted causal response

The documented empirical estimator is:

\[
\widehat\tau_\ell=\frac1{|T_\ell|}\sum_{t\in T_\ell}
\left[\frac{U_t}{e_t}-\frac{1-U_t}{1-e_t}\right]
(x_{t+\ell}-x_t).
\]

Its model counterpart is the expected effect of changing **the action at t only**, with later actions following the specified feedback policy. It is not the difference between “always advocate for ell rounds” and “always silent for ell rounds.”

For Markov count-state dynamics define `V_{k,0}=f` and recursively:

\[
V_{k,j}=P_kV_{k+1,j-1},\qquad
\delta_{k,\ell}=(Q_{1,k}-Q_{0,k})V_{k+1,\ell-1}.
\]

For homogeneous rounds this becomes:

\[
\delta_\ell=(Q_1-Q_0)P^{\ell-1}f.
\]

Evaluate each row at its initial state/context and aggregate using the exact empirical lag-eligible row weights:

\[
\tau_\ell^{\mathrm{theory}}=\sum_{t\in T_\ell}v_t\delta_{t,\ell}(n_t).
\]

At `ell=1`, `delta=chi`. Propensity factors cancel in expectation when action is randomized with the recorded `0<e_t<1`, the modeled branch law is valid conditional on the relevant pre-action information, and the evaluation selection is valid:

\[
E\left[\left(\frac Ue-\frac{1-U}{1-e}\right)(x'-x)\mid s\right]
=E[x'-x\mid do(U=1),s]-E[x'-x\mid do(U=0),s].
\]

Do not divide the theoretical contrast by `e`, multiply it by `a`, or apply IPW a second time. Retain recorded propensities for eligibility, audit, and synthetic validation. Return theoretical branch increments as well as their difference when the empirical output exposes both arms.

If persistent boards, memory or controller state create additional history dependence, multi-lag predictions require a valid transition law for that state. A scalar count model can still be emitted as an explicitly labeled coarse reference; it cannot be claimed to incorporate unmodeled board/evidence evolution. Do not freeze future observed covariates when they are affected by the initial action.

Check termination/censoring semantics: using empirical completed-row masks provides a matched descriptive evaluation set, but treatment-dependent endpoint selection can change the causal estimand. Exact treatment-policy claims require a fixed horizon with valid terminal-state handling or an explicitly modeled selection process. Never extrapolate missing lag outcomes silently.

## 9. Information, entropy, and efficiency

### 9.1 Exact-state action channel

For each count state with action probability `a=a(n)`:

\[
T_\pi(n)=(1-a)\sum_m Q_0[n,m]\log_2\frac{Q_0[n,m]}{P[n,m]}
+a\sum_m Q_1[n,m]\log_2\frac{Q_1[n,m]}{P[n,m]}.
\]

Use `0 log 0 = 0`; skip zero-weight arms before evaluating ratios. With deterministic action `T=0` even if branch kernels differ.

Also expose explicit outcome-entropy audit components, since an unlabeled `H` may otherwise be confused with action entropy:

\[
H_{next|n}(n)=-\sum_m P[n,m]\log_2P[n,m],
\]

\[
H_{next|n,U}(n)=(1-a)H(Q_0[n,:])+aH(Q_1[n,:]).
\]

Validate `T_pi(n)=H_next_given_n(n)-H_next_given_n_and_U(n)` against the divergence calculation. These outcome entropies can exceed one bit; the one-bit bound below applies to binary action entropy and action information. Name them `theory_next_target_entropy_given_target` and `theory_next_target_entropy_given_target_action`, with exact local or occupancy-averaged scope. Do not confuse entropy of the random target count with the existing within-population vote-entropy metric.

\[
H_2(a)=-a\log_2a-(1-a)\log_2(1-a),\qquad
B_{IR}(n)=\frac{2a(1-a)\chi(n)^2}{\ln2}.
\]

\[
\eta_{IF}(n)=T_\pi(n)/H_2(a(n)),\qquad
\eta_{IR}(n)=B_{IR}(n)/T_\pi(n).
\]

Both are undefined when their denominator is zero. Return null plus reason, not zero or one. Validate `0 <= B_IR <= T_pi <= H_2(a) <= 1` within numerical tolerance. Do not clip material violations.

For heterogeneous contexts, use the **same conditional outcome distributions** for `T` and `chi` when forming this bound. A standardized causal effect paired with a differently conditioned observational channel does not automatically satisfy this Pinsker calculation.

### 9.2 Occupancy summaries

On one normalized evaluation population:

\[
T=\sum_n\omega(n)T_\pi(n),\quad
H_{U|n}=\sum_n\omega(n)H_2(a(n)),\quad
H_U=H_2\left(\sum_n\omega(n)a(n)\right),
\]

\[
B=\sum_n\omega(n)B_{IR}(n),\qquad
\eta_{IF}=T/H_{U|n},\qquad\eta_{IR}=B/T.
\]

`H(U)` and `H(U|n)` are different. Verify the actual empirical `eta_IF` denominator selector: `H(U|full population)` is equivalent to `H(U|n_Z)` only when the policy is conditionally sufficient in `n_Z` under the declared model. If not, preserve the full empirical key through the joint-law adapter or mark the mirror incompatible. Never silently rename one entropy as the other.

Export all numerator/denominator components and identified occupancy mass. Do not average local efficiency ratios. If empirical susceptibility uses dual-action states, mirror the exact mask and normalization used by its derived `eta_ir` implementation; report any numerator/denominator support mismatch. Additional full-support model values must be separate variants.

### 9.3 Bins, conditioning and mixtures

Inspect whether each existing local estimator filters to an x-bin but still conditions on exact `n`, or actually coarsens the conditioning variable. For the former, filter/renormalize occupancy then compute exact-state conditional information. For the latter, form `J(bin(n),u,m)` and recompute conditional information. Preserve the next-state variable too: count and count-bin are different output alphabets.

A plot's bin center is a label, not a replacement input for nonlinear calculations. Apply deterministic mappings to the theoretical joint law first, then calculate the matching statistic. Never average cell transition matrices before computing cell metrics unless a pooled-channel estimand is explicitly requested.

### 9.4 Sensing and policy information

With `p_Y(y)=sum_n omega(n)S(y|n)`:

\[
I_{sens}^{bits}=\sum_{n,y}\omega(n)S(y|n)\log_2\frac{S(y|n)}{p_Y(y)}.
\]

Convert with `I_nats = ln(2)*I_bits`. Compute `I(Y;U)` from `p_Y(y)pi(u|y)`, or an appropriately augmented joint law if policy has extra context. A policy table `a(n)` alone does not identify `I(Y;U)` or sensing MI.

For sensor errors, obtain the exact existing error function `g(n,y)` from code and return `sum omega*S*abs(g)` or `sum omega*S*g^2`. Do not guess whether errors compare fractions, counts, or scaled estimates. The scalar target sensor does not automatically model the full option-count sensor vector.

The existing `target_sensing_information_nats` already uses the exact scalar hypergeometric sensor with empirical occupancy according to the reference. Reuse that engine and provenance; avoid presenting an identical calculation as an independent model validation.

## 10. Currents and forward ensembles

Let `mu_u(n)=N*d_u(n)`. For round `k`:

\[
J_{ex,k}=\sum_n\omega_k(n)a_k(n)[\mu_1(n)-\mu_0(n)],
\]

\[
J_{sil,k}=\sum_n\omega_k(n)\mu_0(n),\quad
J_{tot,k}=\sum_n\omega_k(n)[(1-a_k(n))\mu_0(n)+a_k(n)\mu_1(n)].
\]

Validate `J_tot - J_sil = J_ex`. The factor `a` belongs in controller-excess current; it does not belong in the action-contrast causal response of §8.

In forward mode propagate row vectors `omega_{k+1}=omega_k P_k`. Then:

\[
E[J_{episode}]=N(\omega_Lf-\omega_0f)=\sum_{k=0}^{L-1}J_{tot,k}.
\]

Use each empirical episode's initial state and matching horizon for a matched expected episode-current comparison, with valid terminal-state dynamics where required. Cell current uses the existing equal-episode weighting. Do not compare total episode current directly with `sum J_ex`.

Horizon quantities sum round components; round summaries follow the empirical round weighting. `sum_k J_ex,k` is accumulated local excess evaluated along the chosen occupancy trajectory. It is generally **not** the final difference between a full feedback rollout and an all-silent rollout, because those rollouts develop different occupancies. Offer that global intervention comparison only with a separate explicit name.

## 11. Metrics requiring additional modeled state

The scalar target-count kernel does not supply a theoretical mirror for every registered metric. Emit `not_modeled` with the missing state/law instead of fabricating values.

| Family | Additional requirement |
| --- | --- |
| Full population-vector sensing/actuation MI | Joint option-count population and sensor kernels |
| Truth/order response and MI | Joint target/truth/order dynamics; target equals gold is a direct special case, and binary complements require an explicit valid mapping |
| Memory/phi/kappa/susceptible-conditioned response or CMI | Pre-action strata and justified conditional branch laws; future evidence dynamics required for multi-lag rollout |
| Symbolic solvability, fragmentation, evidence gains or capture timing | Joint epistemic transition model, not just fitted scalar compliance |
| Communication efficiency, readers, fact acquisition/reactivation | Activation-cost outcome model with the empirical cost definition |
| Provider token costs | Separate resource model; not inferred from board exposure |

A static epistemic-stratum conditional one-round reference is possible with calibrated kernels and observed stratum weights. Label its assumptions; it does not model how that stratum changes. In general it is not a causal effect of knowledge.

Optional communication extension: with a justified expected activation cost `c_1(s)=E[C|do(U=1),s]`, mirror the denominator `E[U*C/e]` by `sum_s v_s c_1(s)`. For a genuinely fixed frozen board, the expected count of updates exposed is `M*w_b`; that is neither distinct-reader count nor expected number of controller-message presentations. Only map it to a cost metric whose definition matches. Divide matching causal-response and activation-cost components and propagate joint uncertainty.

## 12. Optional branch thermodynamic accounting

Do not mirror the legacy `eta_th` by simply substituting calibrated `h_b` into `h*J_ex/(h*J_ex+I)`. The report explicitly rejects this general blackboard attribution.

For the frozen homogeneous reversible branch model with finite affinities, define:

\[
A_k=\sum_n\omega_k(n)[(1-a_k(n))h_0\mu_0(n)+a_k(n)h_b\mu_1(n)],
\]

\[
S_{sys}[\omega]=-\sum_n\omega(n)\ln\omega(n)+\sum_n\omega(n)\ln\binom Nn,
\]

\[
\Sigma_k=S_{sys}[\omega_{k+1}]-S_{sys}[\omega_k]+I_{sens,k}^{nats}+A_k.
\]

Validate against direct forward/reverse path KL from report §§9–10 before enabling these outputs. Use the internally model-propagated `omega_{k+1}=omega_kP_k` even in a one-round matched calculation; observed next occupancy cannot be substituted into this exact identity.

Suggested separate metrics: `theory_affinity_flow_nats`, `theory_path_irreversibility_nats`, `theory_system_entropy_change_nats`, and `theory_path_transport_fraction = A/(A+I)` only when `A>=0` and denominator positive. These are coarse model quantities, not measured physical energy.

The neutral-baseline special case `h_0=0` gives `A=h_b*J_adv`, where `J_adv=sum omega*a*mu_1`; it does not generally give `A=h_b*J_ex`. Only expose its controller-associated efficiency under the explicit model condition and nonnegative expenditure. Missing/ infinite affinities or nonreversible generalized kernels require a separate supported treatment, not `infinity*0` arithmetic or clipping.

## 13. Uncertainty and residuals

Deterministic exact summation has no Monte Carlo confidence interval by itself. Label uncertainty source: `calibration_only`, `joint_data_and_calibration`, or `synthetic_sampling`.

Reuse the existing episode/shared-initialization bootstrap. For joint intervals, in every replicate recompute calibration, empirical evaluation weights and selected empirical policy estimates, theoretical kernels/components, empirical estimates, and their paired differences. Preserve training/evaluation splits and coupled branches. Reuse stored paired calibration draws if actually available; marginal CIs alone cannot reconstruct parameter covariance.

If only point estimates remain, either re-estimate from retained counts/records using the implemented calibration pipeline or publish point predictions with uncertainty unavailable. Do not assume independent Gaussian parameters from their marginal intervals.

For forward mode bootstrap empirical initial occupancy if it is part of the estimated design, and propagate it anew. Across physical cells resample independently except where the actual experimental design contains shared blocks that must be preserved.

Comparison rows should include empirical estimate, theoretical estimate, residual `empirical - theoretical`, paired residual interval where valid, units, support and conditioning compatibility. Avoid relative residuals when theoretical values are zero or near zero. Do not label in-sample residual intervals as held-out predictive coverage.

Exact model `T` has no empirical null offset. Keep empirical raw `T`, null mean and `T-null_mean` distinct; do not subtract empirical null mean from the theoretical channel, and do not create a null-adjusted `eta_ir`. Optional synthetic checks must run the actual empirical estimator, sample size, smoothing and null scheme; never equate MI of expected counts with expected estimated MI.

## 14. Aggregation and output schema

For explicit cross-cell weights `w_c`, aggregate already computed components:

\[
\eta_{IF,agg}=\frac{\sum_cw_cT_c}{\sum_cw_cH_c},\qquad
\eta_{IR,agg}=\frac{\sum_cw_cB_c}{\sum_cw_cT_c}.
\]

Preserve designed equal-cell versus observation weights, declared marginalized axes, and compatible units. Never average efficiencies, affinities or parameter estimates to manufacture an aggregate kernel. If cells have different `N`, derive per-cell fraction responses and count currents before aggregation and label the resulting units. Retain qualified identities even when local cell labels match.

Suggested new tables under the existing analysis package:

| Proposed table | Content |
| --- | --- |
| `theory_model_manifest` | Calibration, policy, occupancy, kernel, training/evaluation and version provenance |
| `theory_state_metrics` | Exact-state branch moments, `a`, `T`, `H`, `B`, ratios, and current components |
| `theory_primary_estimates` | Empirical-compatible grouped primary/causal/sensing outputs |
| `theory_derived_observables` | Efficiencies, horizon currents, aggregation components and optional path quantities |
| `theory_empirical_comparison` | Matched empirical/theoretical pairs, residuals and compatibility statuses |
| `theory_validation` | Kernel invariants, model applicability, missing dependencies and numerical checks |

Required row metadata: metric and empirical counterpart; study/run/qualified-cell identity; grouping/conditioning JSON; units; lag/horizon; mask and binning identity; theory mode; policy and occupancy source; calibration variant/version/hash; model version; split status; numerator/denominator; empirical/model support; extrapolated mass; estimate and uncertainty metadata; analysis hash and source/dependency IDs.

Large kernels may be cached in the existing resumable work area with a content hash; a new heavy permanent artifact is not mandatory. Published outputs must remain reproducible from retained parameters, model configuration and provenance. Follow existing packaging and validation conventions, including provisional status. Reporting reads prepared rows and must not estimate, rebin or synthesize theory on the fly.

## 15. Proposed configuration and implementation sequence

This YAML is a **schema proposal**, not a claim of currently accepted keys. Adapt final names to the implemented calibration family and parser.

```yaml
blackboard_theory_outputs:
  enabled: true
  model: frozen_board_exposure_mixture_v1
  calibration_source: implemented_blackboard_calibration
  evaluation_modes:
    - matched_empirical_design
    - forward_model_ensemble
  matched_policy_source: empirical_action_frequency
  forward_policy_source: sensor_policy_exact
  empirical_contract: reuse_exact_masks_conditioning_and_weights
  families:
    - target_response
    - causal_response
    - action_entropy
    - target_actuation_information
    - information_efficiencies
    - scalar_sensing
    - currents
  causal_lags: [1, 2, 3]
  unsupported_dependencies: emit_status
  uncertainty: joint_data_and_calibration
  comparisons: true
  optional:
    synthetic_estimator_check: false
    communication_cost_model: false
    branch_path_accounting: false
```

Implementation order:

1. Inventory actual calibration outputs and empirical dispatch; map exact masks, weighting, units, endpoint and estimator variants. Write a machine-readable compatibility map.
2. Implement a pure kernel/policy/occupancy engine with validated inputs and numerically stable primitives; reuse existing exact sensing code.
3. Implement joint-law transformations and response/information component functions.
4. Add causal lag recursion and forward propagation with explicit state/policy completeness checks.
5. Add empirical-compatible adapters, support diagnostics, component aggregation and comparisons.
6. Add joint bootstrap integration, recipe schema, packaging and documentation. Preserve existing empirical outputs and estimator versions.
7. Run analytical fixtures and one supported retained-data analysis. Inspect validation/manifests and comparison provenance; report any blocked family instead of fabricating values.

The current reference says board mode uses `theoretical_reference: none`. Add a distinct blackboard reference identifier and explicit opt-in applicability rules. Do not globally turn on legacy q-voter/single-affinity references for board or forgetting regimes. A homogeneous calibrated blackboard reference can be useful under forgetting, but it does not make epistemic dynamics modeled.

## 16. Numerical fixtures and acceptance tests

### 16.1 Reproduced report fixture

Use `N=M=24`, `B=18`, `q=3`, `b=9`, `q_c=12`, `beta=4`, `theta=0.5`, `(gamma_0,p_0)=(0.35,0.70)`, `(gamma_1,p_1)=(0.65,0.90)`. These are illustrative test parameters, never defaults for user calibration.

They imply `w_b=0.721025641025641`, `gamma_b≈0.5663076923`, `p_b≈0.8655166169`. With exact hypergeometric sensing and the policy above:

| n | x | a(n) | chi(n) | T_pi bits | H(U given n) bits | eta_IF | eta_IR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 6 | 0.25 | 0.7253678932 | 0.1347957916 | 0.3284661709 | 0.8480331716 | 0.3873270314 | 0.0317962711 |
| 12 | 0.50 | 0.5000000000 | 0.1000215251 | 0.2772913903 | 1.0000000000 | 0.2772913903 | 0.0260252615 |
| 18 | 0.75 | 0.2746321068 | 0.0652472586 | 0.1339128654 | 0.8480331716 | 0.1579099378 | 0.0182732758 |

The values above were independently computed while preparing this specification using finite matrix powers and exact finite sums, and agree with the report's rounded examples. Compare printed values within `1e-8`; test internal identities around `1e-10` for this small fixture with float64, with a declared scale-aware tolerance for other dimensions.

### 16.2 Required tests

| Test | Expected property |
| --- | --- |
| Kernel entries, boundaries, row sums and ordered matrix products | Valid probabilities and correct time direction |
| Matrix-power mean versus closed form | Agreement across all `n` in the fixture |
| Mixture placement | Detect incorrect mixture of powers versus power of microscopic mixture |
| Identical branch kernels | `chi=T=B=0`; `eta_IR` undefined; `eta_IF=0` if action entropy positive |
| Deterministic action | `H(U|n)=T=B=0`; ratios with zero denominators undefined; causal model contrast can still exist |
| Equal compliance | Raw susceptibility independent of `x` |
| Positive information, zero mean contrast | `T>0` with `eta_IR=0` is allowed; response alone cannot determine information |
| Information bounds | `0<=B<=T<=H<=1` for matching exact-state laws |
| Entropy conditioning | Distinguish `H(U)`, `H(U|n)` and any full-context entropy |
| Coarse bins | Detect difference between averaging exact-state CMI and conditioning only on a bin |
| Ratios | Aggregate numerators/denominators, not mean ratios |
| IPW expectation | At propensities such as `0.2` and `0.7`, enumerate actions/outcomes and recover the branch contrast |
| Lags 1, 2, 3 | First-action intervention plus feedback recurrence; distinguish from sustained-action powers |
| Policy depends on state | Recompute future policy at reached states, not the initial state |
| Available normalization | Preserve local mean-of-ratios versus summary ratio-of-sums; `x=1` excluded |
| Currents | `J_tot-J_sil=J_ex`; total forward increments telescope to final-minus-initial count |
| Missing future policy / unmodeled evidence | Explicit unsupported/extrapolation status |
| Zero compliance / boundary preference | Stable identity kernels; no `0*NaN` propagation |
| State and cell masks | Exact empirical row identities, units, bins and weights retained |
| Bootstrap | Coupled blocks remain intact; covariance and paired residuals preserved |
| Optional path family | Direct path KL equals entropy change plus sensing plus branch affinity flow |
| End-to-end regression | Existing empirical values remain unchanged when theory is enabled |

Acceptance requires the requested core families to emit matched theory rows or precise missing-dependency reasons; all finite-state numerical fixtures and semantic tests to pass; comparison rows to retain valid provenance and support; and a documented distinction between fitted reference, held-out prediction, and optional synthetic estimator behavior. Do not claim that the full empirical registry has a theoretical mirror when its required state is absent from the model.
