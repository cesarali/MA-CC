# Agent prompt: prepare two Santa Fe v3 studies

You are the agent implementing and running the Santa Fe synthetic simulations. Prepare two reproducible studies: **(A) mean-field theory versus microscopic simulations**, and **(B) finite-sample reliability of the entropy/information estimators**. Keep the scientific questions, sample-size axes, outputs, and conclusions separate, while using Study B to determine which information comparisons in Study A are trustworthy.

Use the existing exact Santa Fe v3 simulator, the supplied `santa_fe_theory_runner_v1.zip`, `INTEGRATION_HANDOFF.md`, `santa_fe_game_mechanics.md`, and the current metrics reference. Inspect the repository and reuse its v3 validation, saved trajectories/RNG states, information engine, bootstrap and null machinery. This is the binary synthetic frozen-board game, not the live-board MuSR/LLM game. No LLM/provider calls or parameter fitting are required.

This prompt supersedes earlier sweep suggestions: the main physical axes are **rho, b, q_c and q**. Use exactly three evidence/social sensitivity regimes. Keep controller threshold and sharpness fixed.

## Shared physical specification

| Setting | Values |
|---|---|
| Population | N=24 |
| Facts | Actual F_plus=7, F_minus=3 |
| Persistence rho | 0, 0.4, 0.7, 0.85, 0.95, 1 |
| Active controller budget b | 0, 2, 4, 6, 12, 24 |
| Controller sensing q_c | 1, 3, 6, 12, 24 |
| Agent message sample q | 1, 3, 6, 12 |
| Evidence-weighted regime | beta_evidence=2.0, beta_social=0.5 |
| Balanced regime | beta_evidence=1.0, beta_social=1.3 |
| Social-weighted regime | beta_evidence=0.5, beta_social=2.0 |
| Controller policy | policy_threshold=theta=0.55; policy_beta=beta_C=8.0 |
| Primary controller target | -1 |
| Main horizon | 60 rounds; retain round 0 and all subsequent rounds |
| Main late window | Rounds 41–60, explicitly a finite-horizon window |
| Model | santa_fe_epistemic_feedback_v3 with compatible semantic switches |

Use one declared initial_fact_redundancy setting from the v3 configuration, fixed across the primary grid. Record its value. Initialize theory from the actual saved simulation state and front page, rather than a generic x=0.5 or independent-vote initialization. Use matched initial blocks where valid, but do not assume initial votes remain identical when their evidence-response coefficient changes.

Export actual integer q_c and b after the simulator's fraction rounding/capping; do not guess them from fractions. Verify realized F+/F- rather than assuming truth_fact_fraction produces the requested counts. q and q_c are different physical resources: q changes exposure, acquisition and social-sampling noise; q_c changes controller observation and feedback decisions. Record effective participant sample sizes if a board contains fewer than q messages.

Keep q_c>=1 under the documented simulator; a sensor-blind policy is a separately defined control, not a mislabeled q_c=0 setting. Do not sweep theta or beta_C in these studies. Fix other mechanics and retain resolved configs, commits and seeds.

## Study A — Does the running theory predict the simulator?

### A1. Integrate and label the theory predictions

Implement the adapter described in `INTEGRATION_HANDOFF.md`. Export actual parameters, initial snapshots, canonical microscopic trajectories and pre-action snapshots; run the supplied Langevin package alongside MICRO.

Compare three named reduced-theory variants where feasible:

1. `paper_reduced`: with-replacement participant sampling and paper-diagonal covariance.
2. `sampling_matched_reduced`: without-replacement category sampling and paper-diagonal covariance.
3. `sampling_clock_corrected_reduced`: without-replacement category sampling and fixed-clock covariance.

The package implements the three-variable reduced theory, not the full class-density HMF. Label any separately implemented class-density predictions independently. Known simulator beta coefficients are inputs, not parameters to refit in every cell.

The game samples messages without replacement; the original paper uses a multinomial approximation. Preserve that distinction. Even the sampling-matched runner treats hidden fact identities as exchangeable and retains binomial epistemic, old-vote-independence, aggregate-night and independent-board approximations. Category-law alignment does not make acquisition identity-exact.

Validate factual replay, parameter/state matching, action timing, b=0 causal null, and a small number of local drift/covariance/acquisition checks before scaling up. U_d is chosen after C_d closes and affects population on day d+1. It must not be regressed against a population change already completed on day d as though that were its response.

### A2. Stage the physical grid

The complete design is 3 beta regimes × 6 rho × 6 b × 5 q_c × 4 q = **2,160 physical cells**, before episodes and theory variants. Prepare the full manifest, but stage execution rather than launching it indiscriminately.

- **Pilot:** a small balanced-regime selection spanning low/high persistence and budget, including b=0. Include q=1 and q=12 cases to assess theory enumeration/runtime, not only q=3. Approximately 64 independent simulation episodes per pilot cell is a starting point.
- **Core phase-map study:** all rho and b, all three beta regimes, q=3 and q_c=12: **108 physical cells**.
- **Communication/sensing interaction study:** all q and q_c, all three beta regimes, rho in {0.4,0.7,0.95,1}, b in {6,24}: **480 physical cells**, 24 already in the core study, hence **564 unique cells** in the combined staged design.
- **Expansion:** fill other cells only if needed to resolve interactions or uncertain regime boundaries. Preserve the full 2,160-cell manifest as a possible later stage.

Initially plan about 256 independent simulation episodes per core cell, with additional replication toward 1,024 at selected uncertain cells if justified by precision. Theory replicas per actual initialization are a separate knob: choose them using measured theory Monte Carlo error and runtime. Do not multiply many replicas by all initializations without assessing cost. Replication values are provisional, not a guaranteed power calculation.

Use separate seed namespaces for pilots, final evaluation, estimator-reference pools and null calibration. Prevent reruns of an existing identical cell unless additional replication is intended. Estimate CPU, storage and retained microscopic-event volume before expansion. Preserve raw events for selected validation/branching cells and the necessary state/summary data elsewhere according to the v3 retention contract.

### A3. Observables and comparison metrics

Primary dynamics and regime observables:

- Truth share x and controller-target share m=(1+c(2x-1))/2.
- kappa_plus, kappa_minus and active fact counts F_plus*kappa_plus, F_minus*kappa_minus.
- Mean individual normalized evidence separately from the ratio of mean counts.
- Peer-board and next-front-page target shares separately from population target share.
- Policy activation frequency, effective actuation, actual posts bU, and controller exposure.
- Trajectory variance/quantiles, late-window mean target support, final majority probability P(m_T>1/2).

Compare mean trajectories, late-window errors, bias/RMSE and selected outcome-distribution distances. Keep final-round mean, late-window mean, majority probability and persistent-capture probability distinct. If reporting capture, define its threshold and persistence duration before examining results.

Causal and information observables:

- Next-day forced-action susceptibility chi_target and its state dependence.
- Sensing information I_sens, action entropy H_action=H(U|S), action-to-output information T_pi, eta_IF=T_pi/H_action, and eta_IR=2a(1-a)chi_target²/T_pi with consistent outputs and units.
- If reporting output entropy, name it explicitly H_output; do not confuse it with action entropy or the sampled social field H.
- Optional availability-normalized susceptibility, with its denominator and x=1 exclusions specified.

Use natural logs/nats consistently, converting any repository bit-valued outputs. Aggregate efficiencies as ratios of weighted components, not averages of local ratios. Report undefined or unsupported estimates rather than forcing zeros or clipping ratios into a plausible range. Do not infer a thermodynamic efficiency from these information ratios.

### A4. Required plot families

**Figure A1: persistence–budget regime maps.** At q=3,q_c=12, show rows for the three beta regimes and clearly labeled MICRO/theory/residual panels. Use a common response color scale and a diverging theory-minus-MICRO scale centered at zero. Add uncertainty/precision markings and operational majority contours. Additional q/q_c slices belong in a structured supplement or interactive report, not an unreadable single figure.

**Figure A2: communication–sensing maps.** Show (q,q_c) responses at selected rho,b, with simulation, theory and residuals. State whether displayed quantity is support, response or information. This tests whether controller sensing matters differently when agents read little versus much.

**Figure A3: trajectories and fluctuations.** Overlay simulation and theory x/m, coverage and board/action trajectories in representative resistant, controlled and crossover cells. Distinguish confidence intervals for means from trajectory dispersion/prediction bands. Add selected outcome distributions or variance comparisons.

**Figure A4: causal and information response.** Compare chi, T_pi and efficiencies in selected cells using matched pre-action branch ensembles. Include Q0/Q1 distribution comparisons. Only report information estimates at sampling levels supported by Study B.

**Figure A5: information, response and cost.** Plot I_sens, target support, chi and actual spending versus q_c at selected q,rho,b; and analogous q slices at fixed q_c. More information is not automatically better control. Claims of feedback advantage require matched-cost comparisons or a spending–performance frontier.

Supplementary diagnostics: drift/covariance residuals, fact diversity/overlap, empirical versus binomial class distributions, timestep/clipping sensitivity and selected horizon checks. Call these finite-horizon regime maps unless a separate stability and size-scaling study supports actual phase transitions.

## Study B — How much data do the information estimators need?

### B1. Separate physical variation from estimator sample size

Hold the physical cell, conditioning definition, output partition and target estimand fixed while varying sample size. Do not interpret changes caused by varying q or q_c as estimator convergence.

Choose a manageable calibration panel spanning all three beta regimes, low/high q, low/high q_c, weak/strong response, and a crossover. An initial balanced panel can use rho in {0.4,0.95}, b in {0,6,24}, q in {1,12}, q_c in {1,24}; then add a few beta-extreme and crossover cases selected by a declared pilot rule. Freeze the selection before final testing. Not every estimator-calibration cell needs a full phase map.

### B2. Use two complementary estimation experiments

**Fixed-snapshot channel experiment.** Select saved complete pre-action snapshots. From each, repeatedly force U_d=0 or U_d=1 and simulate the next night/day. Study per-arm sample sizes R in {32,64,128,256,512,1024}, extending selected reference conditions to 4096 or more if needed and feasible. Here the conditioning state is fixed, avoiding sparse-state CMI as an initial confound.

**Trajectory-engine experiment.** Separately vary the number of independent episodes supplied to the existing empirical MI/CMI pipeline, for example the same 32–1024 sequence where feasible. Keep horizon, bins, analysis window and state-conditioning rules fixed. Report transition counts, occupied states, dual-action coverage and effective block counts. Do not call correlated rounds independent samples. This experiment tests the actual downstream estimator, including sparse conditioning and overlap.

A single fixed sensed state has no nontrivial I_sens across states. To calibrate sensing MI, prescribe a fixed ensemble prior over peer target counts or snapshots, apply the known hypergeometric sensor kernel, and compare empirical estimates against its computed MI. Holding the prior fixed separates estimation error from feedback-induced changes to state occupancy.

### B3. Define references and estimands before estimating

Use exact known finite-channel fixtures as software/estimator controls, plus independent high-replication simulator/theory reference banks for physically relevant channels. For a simulator bank, reference values remain Monte Carlo estimates; provide their uncertainty. Do not call a high-sample estimate exact truth. Keep reference/calibration/evaluation draws independent, or explicitly account for shared data in uncertainty.

The policy gives exact sensor-averaged a(S) at a complete state and hence exact H_action=h_binary(a). Use this as a reference; optionally test empirical action-entropy estimation against it. For T_pi, Qbar=(1-a)Q0+aQ1 and T_pi=(1-a)KL(Q0||Qbar)+a KL(Q1||Qbar). If pooling hidden states, define common weights and mix Q_u before computing the corresponding coarse-conditioned channel; averaging finely conditioned JSDs changes the estimand.

The Langevin output is continuous and the simulator count output is discrete. Specify a common output partition for direct information comparisons, with edges fixed from independent pilot/reference design. Include bin-resolution sensitivity separately from the primary sample-size curve. Never fabricate a binomial SDE output distribution from its mean or round samples and call the result the exact continuous channel.

For an information-response bound computed on binned outputs, use a bounded statistic determined by the same bins for its mean contrast. Retain the unbinned physical chi as a separate metric. Keep target-count-only, reduced-state and complete-snapshot conditioning labels distinct.

### B4. Nulls that respect the experiment

Use both:

1. **Physical no-actuation null:** b=0, for which forced U=0 and U=1 population laws must coincide when U has no other effect. If common-random-number branches produce identical paired outcomes, that is an excellent causal implementation check but an artificially easy estimator-bias test. For estimator calibration, independently sample the two arm datasets from a common null channel as well.
2. **Conditional statistical null:** construct equal Q0/Q1 channels at fixed snapshots, or generate surrogate actions independently of the output conditional on the declared state using the correct a(S). For trajectory estimators preserve the documented conditioning/propensity and episode structure; do not globally permute U across different feedback states. Validate the null-generation assumptions.

If permutation is used with paired samples, preserve the pairing/exchangeability structure; do not apply an independence-based pooled permutation to strongly coupled trajectories. Use the repository's existing null machinery when it matches the intended estimand, and document any modifications.

Compare conditional T_pi to its null, not arbitrary unconditional I(U;next state): feedback can make the latter nonzero even without a causal action effect through shared state dependence. B=0 is not a universal null for sensing information; I_sens can remain positive without actuation.

### B5. Quantify accuracy and detection performance

For each physical condition and sample size, repeat independent estimation experiments or properly designed repeated subsampling. An initial 100 outer repetitions and 200 null replicates can support a pilot; increase where false-positive/power uncertainty is too large. Repeated subsamples of one bank are correlated and are not 100 independent physical datasets. Report the distinction. Exact channels permit inexpensive independent outer trials for error-rate calibration.

Measure:

- Bias, standard deviation, RMSE and reference uncertainty.
- Confidence-interval coverage and width.
- Null false-positive rate at a declared nominal level, e.g. 0.05, with binomial uncertainty for independently repeated tests.
- Detection power under specified nonzero alternatives, with uncertainty.
- Fraction of undefined/unsupported efficiency estimates, near-zero denominators and inadequate-overlap cases.
- Sensitivity to state/output bins, rare outcomes and action imbalance, as separately labeled analyses.

Do not average p-values across cells. Distinguish exploratory many-cell testing from preregistered individual tests, and use an explicit multiple-testing policy when reporting discovery maps. Do not define significance solely by exceeding the mean null estimate.

Retain raw T_pi, its null distribution and any separately labeled null-adjusted diagnostic. Do not construct an unestablished null-adjusted eta_IR. Near T_pi=0, ratio instability itself is a result and may require reporting components rather than an efficiency point estimate.

### B6. Required estimator plots and decision output

**Figure B1:** estimate versus sample size for H_action, optional H_output, T_pi, chi and efficiencies as appropriate, with reference values/uncertainty and estimator spread. Plot per-arm rollout counts and episode counts on separate panels.

**Figure B2:** null and alternative estimator distributions at selected sample sizes, with the actual test threshold.

**Figure B3:** false-positive rate and power versus sample size, including their uncertainty and the nominal false-positive line.

**Figure B4:** bias/RMSE and interval coverage versus sample size; add binning/overlap diagnostics.

Deliver a sampling recommendation table by estimator and representative physical regime. State what precision or power target the recommendation achieves, where uncertainty remains, and where no supported recommendation is possible. Avoid one universal required sample count for all conditions. Sample-size stability alone is not a power analysis.

## Joint execution order and deliverables

1. Prepare manifests, configuration generation, analysis definitions and adapters for both studies. Reuse the supplied theory runner and existing metrics engine; do not implement competing pipelines unnecessarily.
2. Verify mechanics, initialization, clocks, sampling conventions and factual replay; run a small physical pilot and estimator-null pilot.
3. Freeze key estimands, binning/weighting, practical error tolerances, pilot-selected calibration cells and independent evaluation seeds.
4. Use the estimator pilot to choose supported sampling levels for selected information comparisons; expand core Study A dynamics maps in parallel if the theory adapter is validated.
5. Refine uncertain physical cells or unreliable estimators selectively. Do not confuse lack of statistical power with successful theory validation.
6. Deliver separate `study_A_theory_validation` and `study_B_estimator_calibration` result roots, plus a shared provenance/cell manifest. Save raw comparison data, vector figures, uncertainty/support columns, configuration/seed records and runnable analysis commands.
7. Write two separate results summaries and a short synthesis: where the theory predicts MICRO, which approximations explain residuals, and how estimator limitations constrain those statements. If updating the paper, keep the compact PRL main text and place detailed methods/diagnostics in its supplement.

Before large-scale execution, provide the staged job counts, estimated runtime/storage and the existing outputs that can be reused. Prepare the workflows and run inexpensive checks/pilots within the established synthetic-compute workflow; do not silently launch the entire 2,160-cell design without resource assessment. Do not modify v3 production semantics or refit known response coefficients to make plots agree. The goal is an honest comparison between the running theory and the running simulator, accompanied by evidence that the information estimators are reliable at the chosen sample sizes.
