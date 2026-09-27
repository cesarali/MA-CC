# Agent prompt: validate the mean-field theory against Santa Fe v3 simulations

You are the agent responsible for running the Santa Fe simulations and their analysis. Implement and execute a reproducible validation study of the heterogeneous stochastic mean-field theory and its reduced closure against the **existing exact finite-population Santa Fe v3 simulator**. The objective is to identify which theoretical predictions hold, which approximations fail, and in which parameter regimes. Report discrepancies honestly; do not change the simulator or refit parameters merely to force agreement.

This is the synthetic binary Santa Fe game, with no LLM/provider calls. Do not substitute the live-board, three-answer MuSR/LLM protocol from earlier documents. This prompt supersedes those parts of the earlier generic `AGENT_VALIDATION_PROTOCOL.md` that concerned a different game. The current mechanics document is the starting point; verify it against the implementation before running.

## Inputs and existing implementation

Read the supplied `santa_fe_game_mechanics.md`, the v3 paper and revised theory/supplement if available, and `agent_metrics_reference.md`. Locate and inspect these repository paths named by the mechanics document:

- `src/santa_fe/game.py`
- `src/santa_fe/v3_game.py`
- `src/santa_fe/state.py`
- `src/santa_fe/v3_validation.py`
- `configs/santa_fe/v3_pilot.yaml`
- `tdd/santa_fe/santa_fe_v3_simulation_update_plan.md`

Resolve paths against the actual repository; these are document-provided locations, not a guarantee that every current symbol or CLI matches. Reuse the existing simulator, post-hoc validator, retained trajectories, shared information engine, bootstrap machinery, and RNG replay capability. Inspect and extend missing pieces rather than implementing a competing simulator. Follow repository instructions and use available cluster resources appropriately. Do not claim to have run anything until commands and results exist.

The theory parameters are known simulator inputs. The primary study is **prediction without parameter fitting**. New closure parameters, if proposed later, must be calibrated on separate episodes and tested on held-out episodes/conditions.

## 1. Lock the model and align every comparison

Require `model_version: santa_fe_epistemic_feedback_v3` and the compatible semantic switches from the existing pilot YAML. Do not invent configuration keys; add new options explicitly if needed. Legacy v2 results must stay separate.

Save a resolved protocol manifest containing the commit, configuration, seed schedule, model version, actual F+/F-, actual integer b and q_c after rounding/capping, initial fact redundancy, response coefficients, policy parameters, and output stage/index definitions. Budget/sensing fractions are input conveniences; plots and analysis must use realized integers. In particular, the documented sensor has a one-message minimum: do not label a condition “q_c=0” unless a genuine no-sensing control is separately implemented.

Preserve these v3 rules:

1. Global independent fact thinning once at the start of every round.
2. Exactly N focal update slots, selecting agents with replacement.
3. All slots read the same frozen front page B_d; peer emissions go to a separate C_d.
4. Acquire sampled fact IDs before calculating evidence and drawing the new vote.
5. Post a fact aligned with the **new** vote, or a factless vote if unavailable.
6. Sense target votes in **peer messages C_d**, not the final agent population.
7. Sample U_d; when active, append b target-vote/target-fact posts with uniformly sampled fact IDs, with replacement.
8. These posts affect the next day. U_d cannot change the population already recorded at the end of day d.

At b=0 retain both the policy action U and the effective posting action. Forcing U=1 and U=0 must have identical population transition laws when nothing is posted and U has no other effect.

### Two alignment issues to test explicitly

**Participant sampling.** The simulator samples q distinct messages without replacement; the paper's multinomial focal kernel samples with replacement. Keep the default simulator unchanged. Compare:

- the paper's multinomial kernel, labeled as the original sampling approximation;
- a theory kernel using the simulator's multivariate hypergeometric category sample;
- optionally an isolated simulator ablation with replacement, if implemented as a separately named test mode.

For a board of M messages, use q_eff=min(q,M). If M_j counts category j, the without-replacement category law is proportional to the product of choose(M_j,n_j), divided by choose(M,q_eff), over feasible count vectors. Sampling distinct messages does **not** imply distinct fact IDs. Handle duplicate facts in acquisition.

For a particular missing fact f appearing in M_f messages, its exact encounter probability is

`alpha_f = 1 - choose(M-M_f, q_eff) / choose(M, q_eff)`.

With replacement it is `1-(1-M_f/M)^q`. The paper's `1-(1-c_sign/F_sign)^q` additionally assumes exchangeable identity frequencies. Do not call it an exact finite-board acquisition probability. Treat M=0 explicitly if reachable; no normalization by zero.

**Initialization.** In this simulator each fact has a fixed number of distinct initial holders and initial votes depend on local evidence. The earlier theoretical illustrations used a different initialization. For validation, project each actual saved round-0 population into z_0, x_0, kappa_0 and its actual board B_0. Initialize each theoretical branch from those projections. Do not regenerate independent votes or invent a new initial board. When the reduced theory discards vote/evidence correlation, identify that as a closure assumption rather than hiding it in initialization mismatch.

## 2. Compare a hierarchy of theoretical models

Keep labels visible in every output:

- **MICRO:** existing identity-resolved finite-N v3 simulator, the reference.
- **HMF:** class-density theory z_(r,s,v), with its declared fact-identity/exchangeability approximation, frozen-board emission, night kernel, and feedback.
- **REDUCED:** x, kappa_plus, kappa_minus closure, using full finite-sample and post-acquisition voting averages.
- **MEAN-INPUT:** the simplified sigmoid-of-mean-inputs approximation, as an explicit ablation only.

Within feasible HMF/REDUCED comparisons, separate participant sampling, diagonal versus full covariance, and Poisson versus fixed-update clocks. If the full stochastic HMF is not implemented or infeasible, report that and provide deterministic HMF/local moment comparisons; do not label REDUCED as HMF. Never use the MICRO transition rule to evolve a state and then present it as an independently validated mean-field prediction.

A full microscopic state is Markov. Neither the class vector nor the reduced state is automatically Markov after hiding identity overlap and board details. Preserve the exact night kernel at class level where feasible. At reduced level, declare how thinning and boundaries are approximated. Record clipping/projection events and check time-step sensitivity before interpreting theory–simulation differences.

## 3. Stage A: verify the mechanics and local predictions

Start with retained pilot trajectories. At selected complete snapshots, sample repeated independent one-slot continuations or enumerate exact choices when tractable. Distinguish checks of the coded microscopic rule from tests of coarse-graining.

Required checks and metrics:

- **Voting:** observed new-positive frequency versus recorded sigmoid probability, with calibration bins and binomial uncertainty. Recompute E from post-acquisition active IDs and H from the actual sampled messages. This verifies mechanics, not the adequacy of a closure.
- **Persistence:** survivors versus the exact conditional binomial law for held facts. At rho=0 all pre-night active facts are removed, but board facts remain available for reacquisition; at rho=1 there is no night forgetting.
- **Acquisition:** expected/observed distinct new facts by sign; compare identity-level alpha_f, finite-board category approximation, and exchangeable alpha_sign.
- **Emission:** probabilities of (+,+), (+,0), (-,-), (-,0); cross-sign peer posts must be absent. Check uniform selection among aligned active IDs.
- **Sensing/action:** the conditional hypergeometric sensor distribution and Bernoulli action probability. Full q_c=N determines C_d's count exactly, not the agents' target count.
- **Replay:** factual next-day replay reproduces the saved path before using any counterfactual branches.

For the population projection define per-event v=(Delta truth indicator, acquired positive facts/F+, acquired negative facts/F-). Estimate A=E[v] and raw second moment D_P=E[vv^T]. For one scheduled update use D_F=D_P-AA^T. Population increments are v/N; their covariance per unit within-day time is D_F/N for the fixed-slot clock. Do not compare a centered empirical covariance directly to the paper's raw Poisson second moment. Also compare conditional old-vote rates and vote/acquisition cross-covariances.

Use repeated continuations from the **same full state** when estimating conditional covariance. Pooling a broad bin introduces between-state variability and must be identified as such. Compare finite-board and exchangeable predictions on the same snapshots. Do not treat a transition kernel measured on those snapshots as an out-of-sample prediction.

**Plot 1 — local validation:** calibration panels for voting, acquisition, emission, and sensing; predicted-versus-measured drift; covariance matrices or elementwise residuals. Show uncertainty and separate raw/centered clock conventions. The important scientific result is where the class/reduced predictions depart from an otherwise verified microscopic rule.

## 4. Stage B: predict trajectories from matched initial states

Use the same physical parameters and the actual initial projections for all models. Compare ensembles, not individual stochastic paths expected to match exactly. For conditional predictions from one microscopic initialization, run several downstream replicas. For unconditional cell predictions, average over the simulator's initialization ensemble with the same weighting for all models.

Record at consistent stages, ideally before night, after night, after day, and after controller posting:

- x_d (truth share), m_d=(1+c(2x_d-1))/2 (controller-target share).
- kappa_plus, kappa_minus, and kappa_ctrl.
- Mean active fact counts F+ kappa_plus and F- kappa_minus.
- Mean individual normalized evidence E_bar and the ratio-of-mean-counts proxy separately; they need not agree.
- Four peer-message shares, combined-board shares, and board target share separately from population target share.
- Action frequency, actual bU, distinct controller fact IDs posted, and realized controller exposure.
- Variance and quantiles of population share, not only its mean.

Report model bias/RMSE across time, normalized count-distribution Wasserstein distance (or another declared distribution metric), ensemble variance ratio, and confidence/prediction interval calibration where appropriate. Keep Monte Carlo confidence intervals for means distinct from trajectory prediction bands.

**Plot 2 — dynamics:** rows for x, both coverages/active counts, and board share/action rate; MICRO confidence bands with HMF/REDUCED predictions overlaid. Select representative truth-resistant, controller-dominated, and crossover regimes using a declared pilot rule. Include distribution insets at early and late times.

## 5. Stage C: regime maps and parameter interactions

Use the paper's reference parameters initially, provided the repository supports them: N=24, F+=7, F-=3, q=3, policy_beta=8, policy_threshold=0.55, and negative controller target. Explicitly set actual fact counts rather than assuming rounded truth_fact_fraction produces them. Use the simulator's redundancy-based initialization with its value reported. It is not necessary to fit beta_evidence or beta_social: they are known inputs here.

Recommended staged design, not a command to launch an unrestricted Cartesian product:

1. **Pilot:** balanced sensitivities (1,1.3), rho in {0.4,0.7,0.95,1}, b in {0,6,24}, q_c=12, 64 independent initializations per cell, 60 rounds. Establish runtime, output size, and Monte Carlo variance.
2. **Core maps:** rho in {0,0.4,0.7,0.85,0.95,1}; b in {0,2,4,6,12,24}; q_c=12; sensitivity pairs (2,0.5), (1,1.3), (0.5,2). Start at 256 episodes/cell, extending selected uncertain cells toward 1,024 as justified by uncertainty.
3. **Sensing slices:** q_c in {1,3,6,12,24}, rho in {0.4,0.7,0.95,1}, b in {6,24}, initially balanced sensitivities.
4. **Focused beta sweep:** beta_evidence and beta_social in {0.25,0.5,1,2,4}, at b=6 and selected persistence levels, after the local/trajectory checks pass.
5. **Robustness:** selected cases at N in {24,48,96,192}, holding b/N and q_c/N fixed and q,F fixed. For initialization, separately hold initial coverage R/N approximately fixed, and identify that design; holding redundancy R fixed instead changes coverage with N. Check 30/60/120-round horizons and declared initial-condition variations without conflating them with the natural initialization ensemble.

Choose practical error tolerances before examining final results, e.g. two percentage points of target-share error as an initial scientific target if appropriate. These replication values are starting points, not power guarantees. Additional replicates reduce sampling error, not closure bias.

For each cell distinguish late-window mean m (e.g. rounds 41–60), final-round mean, final target-majority probability P(m_T>1/2), and persistent-capture probability under a prespecified threshold/run length. Do not substitute one for another. Compare early/late windows before calling a distribution stationary.

**Plot 3 — regime maps with errors:** for each sensitivity regime, show MICRO, HMF/REDUCED predictions, and signed theory-minus-MICRO residuals in (rho,b). Use common support scales, a residual scale centered at zero, uncertainty/insufficient-precision markings, and separately labeled majority contours. Use actual parameter coordinates or clearly categorical tiles. Interpolated contours are guides; do not invent precise critical boundaries between grid cells.

**Plot 4 — sensing, response, and spending:** target share, activation rate/actual posts, and sensing information versus q_c at fixed rho,b. Include always-on and sensor-blind controls. If claiming an advantage of feedback, compare matched expected spending or a performance–spending frontier. Greater sensing information alone is not evidence of greater controllability.

## 6. Stage D: paired action-conditioned kernels — the decisive causal test

Exploit the saved downstream RNG boundary state and the existing validator. Select complete end-of-day snapshots **after C_d is closed and before U_d reshapes B_(d+1)**. State selection must use only pre-action information.

At each snapshot:

1. Preserve agent facts/votes, C_d, fact signs, and all relevant controller/RNG state.
2. Create forced U_d=0 and U_d=1 branches, with the configured budget. In the active branch draw controller identities according to uniform_with_replacement, not a specially chosen fact set.
3. Advance the next night/day using the resulting frozen B_(d+1). Read population x_(d+1) before its subsequent controller action can matter.
4. Use common downstream streams where valid, but use independent seed pairs across repetitions. Audit random-stream alignment: forcing/skipping an action must not accidentally shift later randomness. Marginal branch laws must remain correct even if exact coupling is unavailable.
5. Average over controller-identity randomness as well as future population randomness. Reusing one factual active set of controller IDs changes the estimand to a fact-conditioned contrast; label it separately if desired.

Estimate the distributions Q_u(n'_target | complete snapshot), means, variances, and chi_target = E[m'|do(U=1)]-E[m'|do(U=0)]. Compare with theoretical kernels initialized from that snapshot's z or reduced projection and C_d. This is a next-day effect, not a contemporaneous effect of U_d on x_d.

Use, for example, 24–48 snapshots stratified by pre-action target support, coverage, board composition and fact diversity; start with 256 branch pairs per snapshot and increase selected cases to 1,024 or more as required for information estimates. Preserve the snapshot sampling weights. Diversity-stratified sampling is not the same as natural occupancy weighting.

Different microscopic snapshots with the same coarse state may have different kernels. Measure both the average coarse-conditioned kernel and between-snapshot dispersion to test sufficiency of z or (x,kappa+,kappa-,B). Do not confuse variation across hidden states with repeated noise from one complete state.

### Causal and information metrics

For each declared conditioning state compute:

- chi_target and uncertainty; truth-coordinate chi has opposite sign for c=-1.
- Optional available susceptibility chi_target/(1-m), only when m<1; distinguish local ratios from a ratio of weighted components.
- The sensor-averaged propensity a(S)=sum_y P(y|n_c) sigmoid(policy_beta*(policy_threshold-y/q_c)), using the peer board. It differs from the realized conditional propensity P(U=1|Y).
- Qbar=(1-a)Q0+aQ1 and T_pi=(1-a)KL(Q0||Qbar)+a KL(Q1||Qbar), in nats.
- H_action=h_binary(a); eta_IF=T_pi/H_action where H_action>0.
- B_IR=2a(1-a)chi_target^2; eta_IR=B_IR/T_pi where T_pi>0.

Keep the complete snapshot, full class state, reduced state, and target-count-only conditioning labels separate. They define different information quantities. Do not compare a target-count-only empirical CMI to a complete-state theoretical JSD and call their difference model error.

Aggregate eta_IF as sum(w T)/sum(w H) and eta_IR as sum(w B_IR)/sum(w T), not means of ratios. If kernels are mixed over hidden states, form the required mixed Q_u before computing their JSD; averaging snapshot JSDs generally gives a different, more finely conditioned quantity. Convert bits and nats explicitly. Report zero denominators as undefined. T_pi=0 at b=0 is an exact null under the stated action semantics.

Reuse the existing information engine and episode bootstrap where applicable. Assess plug-in bias with replication curves and declared null/resampling procedures. A Pinsker bound satisfied by probabilities estimated from the same kernels is a consistency check, not independent validation of the estimator. Do not normalize bias-corrected T_pi into an unestablished “corrected eta_IR.” Do not invent thermodynamic efficiency or entropy-production claims from these ratios.

Use the existing `propensity_weighted_causal_response` as a cross-check on naturally randomized trajectories, with recorded e=P(U=1|Y), positivity, correct day alignment, and episode/block uncertainty. Forced branches identify effects even in rarely activated states; observational estimates still need overlap. Never condition on post-action exposure/acquisition to estimate the total effect.

**Plot 5 — causal response and transition channels:** MICRO versus theory chi parity plot with intervals; selected Q0/Q1 count histograms overlaid with predicted distributions; T_pi and eta_IR comparisons under identical conditioning, units and weights. Add budget/persistence susceptibility maps only after branch support is adequate.

## 7. Diagnose failures rather than merely report disagreement

The retained identities and class vectors permit direct closure tests:

- Compare empirical z_(r,s,v) and vote-conditioned count distributions with the assumed independent-binomial law. Measure count variance/Fano ratios, r–s correlation, and vote–evidence dependence. Use TV/JSD with finite-sample calibration where interpretable; sparse empirical tables otherwise produce apparent distances even under a correct model. Do not compare a time-pooled distribution to a single instantaneous closure without matching the mixture.
- Track identity frequencies, fact-holder overlap, diversity/effective number of facts, and extinction/reintroduction. A fact absent from agents can remain on the frozen/next board and be reacquired. Define loss in agents separately from loss in all currently transmissible reservoirs. Controller-aligned facts may be reintroduced from its pool; opposite-sign facts generally cannot. The redundancy initialization ensures initial holders when R>0, unlike the earlier Bernoulli initialization.
- Compare mean individual evidence with the ratio-of-mean-counts approximation.
- Test board count dispersion and joint population–emission covariance using repeated whole-day continuations from the same start. The independent multinomial board is an approximation; conditioning on the full realized population path can itself alter the emission law, so define the conditioning experiment carefully.
- Examine residuals versus q/M, identity concentration, coverage, old-vote dependence, and time. This helps distinguish sampling-law errors from identity closure, binomial closure, covariance, and numerical errors.

**Plot 6 — why the approximation fails:** predicted-versus-observed epistemic distributions, identity-diversity/overlap trajectories, and theory residuals against independently defined closure diagnostics. Associations do not establish a single causal source. Targeted ablations are needed for causal attribution, and should be labeled as modified models.

**Supplementary plots:** N/horizon convergence; time-step and clipping sensitivity; full covariance comparisons; controller exposure and distinct-acquisition funnel; susceptibility/CMI overlap and information-estimator replication curves; true versus false controller targets in target coordinates.

## 8. Deliverables, priorities, and acceptance

Execute in stages: verify v3/replay and local mechanisms; match initialization/sampling; run a pilot; compare trajectories and regime maps; estimate forced-action kernels; then expand only where uncertainty or failure diagnosis requires it. Do not spend the whole budget on a huge grid before validating the clock.

Deliver:

1. `protocol_alignment.md` and a machine-readable manifest, with every theory/simulator mismatch and how it is handled.
2. Reproducible resolved configurations and seed/block manifests; separate output roots for legacy, v3, and ablations.
3. Validated local drift/covariance/acquisition/emission reports and factual replay checks.
4. Per-cell and per-state tables carrying model label, initialization/snapshot weights, sample sizes, uncertainties, units, time stages, conditioning variables, and numerical settings.
5. The six plot families above, with vector PDFs, machine-readable data, and self-contained captions. Prioritize trajectory overlays, regime-map residuals, and paired causal-kernel comparisons as the main paper figures; local mechanics and closure diagnoses can support them in the supplement.
6. A concise scientific report classifying each discrepancy as implementation error, protocol mismatch, numerical error, Monte Carlo uncertainty, or structural approximation error. Where unresolved, say so.
7. A manuscript-ready results section explaining rho, b, sensing capacity, and closure validity, using only completed computations. Preserve PRL formatting if editing the paper; put detailed methods and secondary diagnostics in supplementary material. Do not replace a submitted draft silently.

A successful study need not show agreement everywhere. It should establish an observable-specific validity range, or a reproducible failure of a named approximation, under matched conditions and independently checked uncertainty. Call finite-N/finite-horizon plots regime maps unless a separate stability and size-scaling analysis supports a phase-transition claim.

Before large jobs, report the resolved semantics, planned run counts, estimated runtime/storage, and reused existing outputs. Then continue with authorized synthetic validation. No LLM simulation, external-provider calls, or empirical calibration is required for this Santa Fe study.
