# Blackboard steering: checkpoint ensemble and longitudinal continuations

Date: 2026-09-16. Status: implementation and experiment-planning handoff.

## 1. Purpose and scope

Implement a paired checkpoint experiment asking: under what conditions can a dashboard controller steer a population of reasoning agents toward the correct answer or a fixed incorrect answer? Measure immediate response, subsequent policy effects, and information about the assigned intervention carried by population outcomes.

Prepare configurations, checkpoint support, analysis, validation, and a costed execution plan. This document is not by itself authorization to launch paid simulations. It specifies the proposed experiment; distinguish recommended defaults, pilot-dependent decisions, and optional extensions in the resolved plan. Do not silently reduce sample sizes or omit conditions.

The primary proposal uses 120 independent parents per reading-budget/persistence setting, two preparation rounds, ten continuation rounds, and one continuation copy per branch. Run BOTH reading budgets, 12 and 3. The original cost argument covered ONE reading budget: the expanded two-budget study has a larger total cost unless sample size is reduced. Section 10 gives the accounting and a budget-constrained alternative.

## 2. Read these sources and map their notation

Repository root on the user's machine: `/Users/rsanchez/Projects/MA-CC`.

Reference scientific configurations, relative to that root:

- `configs/runs/relational_reasoning/blackboard_game/astra_task003_false_control_30x30_potsdam_rho3/false_control_llm.yaml`
- `configs/runs/relational_reasoning/blackboard_game/astra_task003_truth_control_30x30_potsdam_rho3/truth_control_llm.yaml`
- `configs/runs/relational_reasoning/blackboard_game/astra_task003_no_control_30x30_potsdam_rho3/no_control.yaml`

Read `/Users/rsanchez/Projects/agents_control/EXPERIMENT_PLAN_MEMORY.md` and `/Users/rsanchez/Projects/agents_control/TASK003_METRICS_AUDIT_HANDOFF.md`. The latter records incomplete false-control data, reporting defects, and information-estimator limitations. Resolve these against the current code rather than assuming every historical defect remains present.

Follow the repository's `AGENTS.md`, `.codex/skills/ma-cc-study-workflow/SKILL.md`, and the architecture and handoff documents named by that skill. Reuse the generic study orchestration and established estimator infrastructure. Put scientific parameters in YAML and analysis choices in the analysis recipe. Do not invent unsupported YAML fields; identify required schema/code extensions explicitly.

The notation below is local to this specification. Produce a mapping from these definitions to actual configuration keys and retained-data columns. Never infer semantic equivalence from matching variable names alone.

## 3. Notation and recommended parameters

| Symbol or name | Definition | Recommendation |
|---|---|---|
| `N` | Number of ordinary agents in a population | 24 |
| `q` | Maximum number of eligible dashboard messages sampled for an ordinary agent's update; repository `game.options.social_group_size` | Run 12 and repeat at 3 |
| `rho` | Repository evidence-persistence parameter; preserve its exact retention semantics | 0.70 and 1.00 |
| `K` | Independently generated parent populations WITHIN one fixed `(q, rho)` setting | 120 in full proposal |
| `L` | Number of uncontrolled preparation rounds before saving a checkpoint | 2; pilot-dependent exception below |
| `M` | Number of population rounds after branching | 10 |
| `B` | Number of independently randomized continuation copies of each policy/budget branch per parent | 1; configurable for future use |
| `b` | Maximum allowed controller posts in an active round, not a guaranteed realized count | 3 and 12; none has 0 |
| `s` | Number of agent votes sampled by the controller sensor | 12 |
| `theta` | Target-share threshold in the original activation rule | 0.50 |
| `beta` | Slope parameter of the original probabilistic activation rule | 4.0 |
| `i` | Parent identifier within a fixed `(q, rho)` setting | Integers 1 through K |
| `j` | Continuation-copy identifier within a branch | Integers 1 through B |
| `r` | Absolute population-round index, starting at 1 | Preparation ends at r=L |
| `h` | Number of rounds elapsed since branching | h=0 at checkpoint; primary outcomes h=1 and h=M |
| `p` | Branch policy label | Five labels defined below |
| `v` | Semantic answer whose support is being measured | Correct answer or fixed false target |
| `n_i(v)` | Number supporting answer v in parent i's checkpoint | Integer 0 through N |
| `m_{i,p,b,j,h}(v)` | Number supporting v after h rounds in a specified continuation | Integer 0 through N |

Uppercase `B` and lowercase `b` have different meanings. The controller sensor size `s` and ordinary-agent reading budget `q` are also different quantities.

A population round contains the repository's sequence of N ordinary-agent updates. One-round response means a full population round, not one individual update. An episode in this study must be qualified as either a parent preparation episode or a branch continuation episode. The independent statistical block is the parent together with ALL its descendants.

## 4. Parent generation and checkpoint boundary

For each of the four settings `(q, rho)` in `{12,3} x {0.70,1.00}`:

1. Generate K independent stochastic initial populations using the original task and private-information allocation protocol. Preserve `paired_local_vote` semantics, but generate enough new initialization artifacts. Copying the same old 30 artifacts four times does not create 120 independent parents.
2. Run each parent for L rounds with no controller. Use the same q and rho that its continuations will use.
3. Save a complete immutable checkpoint at the END of round L, BEFORE the next round's normal message-expiry and evidence-persistence operations.
4. Fork the checkpoint into the policies below. Apply the next round's boundary operations exactly once in each continuation. Do not reset votes, evidence, dashboard state, or the ordinary-agent interaction history at branching.

Recommended default: independently generated parent ensembles across `(q, rho)` settings. Pair policies and posting budgets within each setting. Do not claim a cross-q effect holding checkpoint states fixed: changing q during preparation changes the starting distribution as well as later dynamics. A common-preparation, continuation-only q intervention would be a distinct future experiment.

The checkpoint must include every state component needed by future dynamics: individual votes, evidence and evidence history, agent memory or conversation context where used, dashboard entries with lifetimes/authorship/order, round and micro-update position, random-stream state, and any posting eligibility/cooldown history. Before branching, controller history should represent a controller that has not acted. Set branch-specific controller state explicitly and identically except for the intended policy/target/budget differences.

Do not assume `storage.checkpoint_mode: episode` already supports arbitrary round-level forks. Inspect and extend save/restore semantics if needed. Demonstrate checkpoint continuation equivalence using a deterministic/mock provider.

Use distinct documented continuation random streams indexed by setting, parent, policy, budget, and copy. Sharing a checkpoint is required; sharing future random draws is not the default. Do not claim that an LLM provider is replayable merely because local random seeds agree.

## 5. Five policies and the meaning of sensing

| Policy label | Controller target | Activation |
|---|---|---|
| `always_truth` | Correct answer | Every continuation round |
| `always_false` | Fixed incorrect answer | Every continuation round |
| `sensing_truth` | Correct answer | Original sensor-dependent stochastic gate |
| `sensing_false` | Same fixed incorrect answer | Original sensor-dependent stochastic gate |
| `none` | No acting controller | Never |

Run each of the four controller policies at b=3 and b=12. Run only one no-control branch per parent/copy; reuse its outcomes across the two budget comparisons. This gives nine branch continuations per parent when B=1, not ten. The shared baseline induces correlation between comparisons; preserve it in resampling.

Use semantic answer IDs, not display letters, in analysis. In the reference task, the correct answer is `ALLOCATION_0` and the false target is `ALLOCATION_2`; verify these against the task hash and resolved configs. Letters A/B/C are prompt display labels and must be mapped to semantic IDs.

### Sensor and activation rule

Let `Y_r` denote the sampled votes observed by the controller in round r. Let `z_r` be the fraction of these s sampled votes supporting that controller's target. Let `U_r` be 1 when activation occurs and 0 otherwise. Let `e_r` be the activation probability conditional on the sensor observation. Define the logistic function `sigma(x)=1/(1+exp(-x))`.

For sensing policies, preserve the original rule:

`e_r = sigma(beta * (theta - z_r))`, followed by `U_r ~ Bernoulli(e_r)`.

The controller is more likely to activate when its sampled target support is low. Preserve the repository's sensor sampling mechanism and timing, and record the sampled votes, e_r, and U_r. For always-active policies, U_r=1 and e_r=1. For none, U_r=0 and e_r=0; activation propensity estimators are not applicable to these deterministic branches.

### Critical distinction: always-active is not sensor-free

The existing `advocacy_schedule: always` bypasses the stochastic activation decision, but the sensor still runs. The adaptive communication controller can receive sampled opinions when choosing what message to send. Therefore, do NOT call these branches genuinely sensor-free.

The primary comparison asks whether USING SENSING TO GATE ACTIVATION improves steering relative to always activating. Keep the active-round communication procedure, available information, and other rules matched between the always-active and sensing policies. Differences later in the run may also reflect the histories produced by those policies; that is part of the longitudinal policy effect.

A genuinely blind controller is an OPTIONAL SEPARATE EXTENSION. It would require removing sensor observations from message selection as well as from activation. Also audit other inputs: public votes, prior dashboard contents, and population summaries can reveal the population state. Specify whether 'blind' means only 'no sampled-vote sensor' or 'no population-state information at all.' A fixed message protocol based only on predeclared facts/target/budget is one way to implement the latter. Merely selecting `always` is insufficient. Do not silently substitute this extension for the five primary policies.

## 6. Communication and ordinary dynamics

Preserve these reference settings unless an explicit pilot decision changes them:

- Task `task_003`, original dataset and private-assignment/controller-pool hashes; do not generalize results to all tasks.
- Original provider/model `gwdg/openai-gpt-oss-120b`, temperature 1.0, maximum output tokens 4096, prompt family and version 4, response schema version 2. Record any unavailable-model substitution as a design change.
- Complete interaction topology, board social mode, uniform eligible-message sampling, self-authored messages excluded, participant requests allowed, and no-post allowed.
- Board message lifetime one round; public votes; vigilant receiver disposition; no early stopping at consensus.
- Adaptive controller communication, dawn-only controller timing, LLM structured communication policy version 1 and original fallback, requests/directives allowed, report cooldown one round, report maximum posts per fact three, target-preserving report selection.
- Original response validation and retry rules, with failures retained in diagnostics. Preserve provider load control and cost limits rather than copying old concurrency settings without preflight.

In adaptive communication, b is a MAXIMUM. Reports may use multiple slots, subject to eligibility restrictions. A request or directive is posted once rather than duplicated to fill b slots. Active rounds can therefore use less than the allowed budget. Preserve this behavior for continuity, and report realized posts, communication act types, and actual participant reads.

The false controller steers toward a false answer under the existing evidence/message constraints; do not introduce fabricated facts or a new attack mechanism.

Message expiry and evidence persistence are separate mechanisms. Apply the original one-round eligibility rule to dashboard messages; archived logs may retain expired text. Apply rho to evidence according to the original evidence-retention implementation. Do not replace either mechanism with a blanket dashboard reset or erase all persistent evidence after a round. Inherited checkpoint messages must retain their correct ages.

The q=3 study is a full repeat of the same proposed protocol, not a reuse of the old 30-round studies. Hold all other chosen scientific settings fixed when possible. Higher q changes ordinary evidence exchange as well as controller exposure; it is not a pure controller-exposure manipulation.

## 7. Primary estimands and clearer estimation

### Paired response / susceptibility

For target v and horizon h, first average over continuation copies. Define `bar_m_{i,p,b,h}(v)` as the arithmetic mean of `m_{i,p,b,j,h}(v)` over j=1,...,B. Define `bar_m_{i,none,h}(v)` analogously from the baseline. With B=1, these are simply the observed counts.

The primary estimator within each fixed `(q,rho)` setting is:

`chi_hat_{p,b,h}(v) = (1/K) * sum_i [bar_m_{i,p,b,h}(v) - bar_m_{i,none,h}(v)] / N`.

Primary horizons: h=1 and h=M=10. Report fractions and percentage points with explicit units. The first is an immediate policy response; the second is a cumulative policy effect on the endpoint, not a sum of one-round susceptibilities or a derivative with respect to b. For sensing policies, the comparison includes silent rounds and estimates adopting the entire policy, not forced activation.

Report correct-answer support for all policies, and fixed false-target support separately. They are not complements in a three-option task. Include time courses for all h=0,...,M, but predeclare h=1 and h=M as primary.

Why this is clearer than the old design:

- Each policy begins from exactly the same complete state as its comparator, not merely the same target count.
- There is no need to find historical count bins containing both activation and silence.
- Whole-policy paired contrasts need no inverse-propensity weighting.
- Shorter trajectories and fewer parameter settings allow more independent parents; many rounds from one trajectory do not substitute for independent parents.
- Policy-induced changes in state occupancy are part of the longitudinal effect, instead of being obscured by averaging immediate effects over different visited distributions.

This design does not eliminate LLM randomness, weak effects, task specificity, or uncertainty in information estimation.

For confidence intervals, bootstrap whole parents with all branches/copies/horizons kept together; recommended 2,000 bootstrap replicates for the mean response. Never bootstrap individual rounds as independent observations. Report K and missingness per comparison. Pointwise 95% intervals are not simultaneous coverage over all plotted comparisons. Predeclare any family of formal hypothesis tests and its multiplicity handling.

Increasing B can reduce continuation noise conditional on a checkpoint, but does not create additional independent parent states. Keep B configurable; B=1 is the primary resource allocation.

### Assigned-policy information

For EACH controller-policy-versus-none comparison, define a balanced binary label `A`: A=1 selects the controller branch and A=0 the paired no-control branch, each with probability one half. This defines the comparison distribution even though both branches are actually simulated.

Let `n_0` be the checkpoint target count and `m_h` the target count h rounds later in the selected branch. The primary information quantity, in bits, is:

`T_{p,b,h}(v) = I(A; m_h | n_0)`.

Here I denotes conditional mutual information. Because both labels are represented equally for every parent, `P(A=1|n_0)=1/2` in the intended design. T measures conditional distinguishability of intervention outcomes, not whether steering was toward truth, and not the full information flow through all messages. Combine it with signed chi. An always-active branch can carry assigned-policy information relative to none even though its internal U_r is constant.

Use all K parents, including counts unique to a parent. Do not condition the empirical frequency estimator on exact parent identity; that creates a different, extremely sparse estimation problem. Complete-state pairing supports causal policy comparisons, while n_0 is the deliberately chosen coarse conditioning variable for T. Do not assert that n_0 is a Markov state.

For a cross-fitted classifier, let `g_{-f}(a|n,m)` be the probability it assigns to label a given counts n,m, trained without fold f. Let `f(i)` identify parent i's held-out fold. For B=1, estimate the predictive score:

`T_hat_classifier = (1/(2K)) * sum_i sum_{a in {0,1}} log2[ g_{-f(i)}(a | n_i(v), m_{i,a,h}(v)) / (1/2) ]`.

Here `m_{i,1,h}` is the selected controller-policy/budget outcome and `m_{i,0,h}` its baseline. With B>1, average within parent and label before averaging parents, so every parent retains equal weight.

Use five outer folds grouped by parent, with grouped inner validation for regularization/model selection. All descendants and copies of a parent must stay in one fold. Fit a modest regularized logistic model using scaled starting count, count change, and a preregistered low-order interaction/polynomial feature set; include the constant 1/2 classifier as a candidate. Record feature definitions, penalty grid, seeds, clipping convention, calibration and held-out log loss. Reuse the prior classifier analysis where appropriate, but REMOVE the old sensor-propensity baseline: this new label A has balanced prior 1/2.

The expected classifier score is a lower bound on T when evaluated independently of training, with equality only for a correct posterior model. Finite-sample estimates can be negative. Do not clip reported scores to zero or present them as guaranteed recovery of true T. Repeat grouped splits to diagnose sensitivity; split variation is not a confidence interval. If computing bootstrap uncertainty for the fitted procedure, resample parents and refit the pipeline, keeping duplicate instances of the same original parent in the same fold to prevent leakage.

Compare with the existing frequency conditional-information estimator and uniform-smoothed conditional transition estimators. For explicit smoothing definitions, let `C_a(n,m)` count outcomes m under label a from starting count n, and let `C_a(n)=sum_m C_a(n,m)`. Use total row pseudocount `lambda` distributed uniformly across the N+1 possible target counts:

`p_hat_lambda(m|n,a) = [C_a(n,m) + lambda/(N+1)] / [C_a(n)+lambda]`.

Use lambda=1 and lambda=12.5 as sensitivity checks; lambda=0 gives the unsmoothed frequency estimator on observed starting counts. With equal branch weights, define `p_hat_lambda(m|n) = (p_hat_lambda(m|n,0)+p_hat_lambda(m|n,1))/2`, and average `sum_a sum_m (1/2)*p_hat_lambda(m|n,a)*log2[p_hat_lambda(m|n,a)/p_hat_lambda(m|n)]` over the empirical parent distribution of n. Use the established estimator engine/adapters rather than creating a competing CMI implementation. State any engine convention that differs from this definition.

Report support counts, singleton starting states, held-out scores, and estimator sensitivity. A paired label-swap null (swap the two whole continuation records within independently chosen parents) can diagnose apparent information under an exchangeability null; label it with that assumption, preserve all within-parent dependence, and refit the classifier. It is not a universal exact test of the weaker count-conditioned null when hidden-state effects cancel in aggregate. Do not reuse the old sequential action-redraw null as though it were this new experiment.

Validate information-estimator behavior on an LLM-free model with known T before claiming accurate recovery. Ground-truth T is not known for the LLM data. No claim that K=120 guarantees adequate information-estimation precision is authorized by this design.

### Optional activation-level analysis

Within sensing branches only, retain the quantities needed for the old randomized immediate effect. Define `D_r` as the within-round change in target fraction, and `R` as the number of eligible rounds. The lag-one estimator is `tau_hat_1 = (1/R)*sum_r [U_r/e_r - (1-U_r)/(1-e_r)]*D_r`. It estimates current activation response over visited histories under the randomized gate, not the primary paired whole-policy effect. Cluster by parent, and report it separately. Never apply this formula where e_r is 0 or 1.

## 8. Required retained data

Counts alone suffice to form some outcome estimators but are insufficient for restart, exposure auditing, or mechanism diagnosis. Retain:

- Study/config/code/model/prompt/task/private-assignment/controller-pool versions and hashes.
- Explicit q, rho, N, K, L, M, B, b, policy, target, correct answer, parent/checkpoint/copy IDs and checkpoint hashes.
- Separate absolute round index r and post-branch index h, and micro-update order.
- Full semantic option-count vector and individual votes at h=0 and after EVERY continuation round; retain preparation history too.
- Sensor samples and sampled IDs where available, sampled target share, activation probability, realized activation, communication act, selected fact IDs, actual posts, controller decision/retry/fallback records.
- Board messages, authorship, timestamps/expiry, sampled eligible messages per agent update, and delivered controller-message counts.
- Evidence available for reasoning, facts ever received, persistence events, and required memory/context for restart. Map repository evidence fields explicitly; do not treat ever-received facts as necessarily still available for reasoning.
- Random-stream provenance, checkpoint state, failures/interruption/resume status, provider requests/tokens/cost and retries by parent/branch.

Canonical tables must preserve the parent-child mapping across separate execution jobs. Do not duplicate the shared baseline as independent data during pooled analysis. Retain both starting and ending counts with unambiguous timing.

## 9. Pilot and protocol freeze

Run credential-free preflight and deterministic/mock tests first. When real pilot execution is authorized, recommend a separate pilot of 10 parents per `(q,rho)` setting using the full branch structure, B=1, L=2, M=10. Include pilot cost in the resource plan. These pilot parents are separate from the main K unless a protocol was fixed in advance and an explicit inclusion rule is documented.

Check checkpoint-state distributions, ceiling/floor effects, actual posts and reads, invalid responses, token cost and latency. More reading need not strengthen control: it can strengthen ordinary truth discovery.

If q=12 already produces near-consensus after two preparation rounds, assess L=1 in the pilot and report the checkpoint distributions. Prefer the same finalized L for both q studies to preserve a clean protocol comparison. Do not filter out consensus parents or choose L to maximize a favorable treatment effect. Freeze L and all scientific settings before the main ensemble.

Estimate achievable confidence-interval width from parent-level variability. More independent parents improve mean-response precision approximately with the square root of K, but pilot variability is uncertain and T has additional estimation bias. Finalize the costed K before main collection; do not stop when significance is achieved.

## 10. Workload and honest comparison with the old cost

The retained old archives contain 540 truth-control episodes, 458 false-control episodes, and 90 no-control episodes: 1,088 completed episodes. Each has 30 rounds and 24 ordinary updates per round, giving 783,360 ordinary updates. Their logged provider-request totals sum to 797,885. These measures are different: requests include other calls/retries, and ordinary-update count is a workload proxy. The originally planned complete grid would have required 842,400 ordinary updates. Use retained actual usage as the conservative comparison and account for any failed/unarchived spend separately if available.

For workload notation, let `Q` be the number of reading-budget settings and `P` the number of persistence settings. There are nine continuation branches per parent/copy. Ordinary-update workload is:

`W = Q * P * K * N * (L + 9*B*M)`.

This excludes initialization calls, controller calls, retries, pilot runs, and any longer-prompt cost. Preparation is performed once per parent, not once per branch.

| Design | Q | P | K per setting | Parents | Continuations (B=1) | Ordinary updates W |
|---|---:|---:|---:|---:|---:|---:|
| Earlier proposal, q=12 only | 1 | 2 | 120 | 240 | 2,160 | 529,920 |
| Full proposal, q=12 AND q=3 | 2 | 2 | 120 | 480 | 4,320 | 1,059,840 |
| Budget-constrained alternative, both q values | 2 | 2 | 60 | 240 | 2,160 | 529,920 |
| Separate suggested pilot, both q values | 2 | 2 | 10 | 40 | 360 | 88,320 |

Why the earlier proposal could fit: a single-q main study used about 68% of the old retained ordinary updates, leaving room for other calls and larger q prompts. That was a plausibility argument, not a monetary guarantee.

Why the full expanded proposal does NOT automatically fit: adding the q=3 repeat at K=120 doubles that main workload to about 135% of the old retained ordinary updates, before overhead and pilot cost. Do not say that both full studies fit the old computation budget.

If the old total cost is a strict constraint, propose K=60 per `(q,rho)` setting as the starting alternative, with both q values and all branches retained. With the separate 10-parent pilot, this totals 618,240 ordinary updates, about 79% of the old retained workload before overhead. This still requires token-cost validation. Relative to K=120, sampling uncertainty for mean response is approximately sqrt(2) larger at otherwise equal variance. This alternative is not an already approved replacement for the full proposal.

Use pilot-measured input/output tokens, controller activity and retries to cost the complete study, not only request counts. Set explicit global and per-stage ceilings. If even the reduced design exceeds the intended cost, present a revised balanced sample allocation; do not let late-scheduled false-control branches become the accidental missing cells.

## 11. Implementation validation and execution contract

Required before main execution:

1. Map notation and all old/new config settings; identify any scientific change.
2. Validate expected parents, branches, copies and scientific cells independently of scheduler shards.
3. Verify all children share the exact parent state hash before branch-specific setup, with no premature expiry/persistence or duplicated preparation calls.
4. Verify save/restore preserves state and boundary timing in deterministic/mock runs, including the first continuation round.
5. Check always-active and sensing gates, logged probabilities, sensor-information availability, post-count caps, and correct semantic targets.
6. Verify none is reused correctly, copies have distinct future streams, and no parent leaks across classifier folds.
7. Run the actual retention profile at reduced scale and confirm every required field survives canonical export and reaggregation.
8. Use known-outcome synthetic cases to check paired response, balanced label weighting, constant-label classifier baseline, information-estimator support diagnostics, and bootstrap grouping.
9. Preflight resource totals and provider concurrency using the generic study workflow. On Potsdam follow the existing dedicated environment and results-root rules; do not invent a study-specific scheduler script.
10. Require complete expected branch coverage for final paired analysis. Resume recoverable failures. Do not replace missing outcomes by zero or silently analyze a selected set of surviving parents. Mark exploratory partial results, show missingness, and state the resulting analysis population.

The baseline condition has no meaningful posting-budget sweep. There are 36 branch conditions across both q and both rho settings: four settings times nine branches. Each condition has K continuations at B=1. Parent preparation artifacts are an additional stage, not another control branch. Physical scheduling may combine or split jobs without altering these scientific identities.

## 12. Deliverables and reporting

Deliver resolved scientific configs, study and analysis manifests, notation-to-schema mapping, preflight and pilot report, checkpoint implementation/validation evidence, frozen main protocol and cost estimate, canonical paired tables, estimator configuration/provenance, and an analysis handoff.

Primary figures: paired truth-support and false-target effects at h=1 and h=10, trajectories with parent-block uncertainty, assigned-policy information estimates with frequency/smoothing comparisons, and actual exposure/resource usage. Show q and rho separately. With only two posting budgets, show their contrast; do not fit or imply a finely resolved susceptibility curve.

Explain sensing outcomes alongside active-round counts, actual posts and token usage. Similar final outcomes with fewer interventions can be scientifically meaningful. Do not condition the primary effect estimate on realized exposure, since exposure is a post-intervention mediator; exposure-conditioned comparisons are descriptive unless separately identified.

Use the task-specific scope honestly. This experiment tests steering by an externally assigned controller, not spontaneous emergence of leadership. The main scientific story is how persistence, reading, and activation policy change the direction, magnitude, persistence, distinguishability, and resource cost of control.
