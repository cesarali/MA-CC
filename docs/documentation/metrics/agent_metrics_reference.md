# Metrics and aggregation: agent master reference

Scope: `relational_imitation_round_feedback`, MuSR team allocation, `social_mode: board`,
`prompt_family` / response contract `relational_blackboard_ballot`. Study packaging and reporting
are generic; evidence and controller semantics below belong to this game family.
Reconciled against implementation and checked-in recipes on 2026-09-14. This is the entry point;
“implemented” means present in code, not that a study has collected sufficient data or passed a
fresh execution test. A study's resolved recipe, manifest, retained fields, and support determine
what is actually available.

## Authority and navigation

| Need | Authority / detailed reference |
| --- | --- |
| Agent overview, exact names, dispatch, readiness | This document |
| Live metrics and ordinary estimator interpretation | [metrics.md](metrics.md) |
| Evidence semantics, symbolic solver, causal formulas, plot catalogue | [metrics_epistemics.md](metrics_epistemics.md) |
| Validation, identity, retention, reaggregation | [study_aggregation_contract.md](study_aggregation_contract.md) |
| Report configuration and presentation constraints | [report_creation.md](report_creation.md) |
| Scientific design | Submitted experiment YAML + resolved cell overrides |
| Requested analysis | `study_manifest.json: analysis_recipe` → study `analysis.yaml`; emitted snapshot is `analysis/analysis_recipe.yaml` |
| Actual outputs and status | `analysis/validation.json`, `analysis_manifest.json`, source table rows |

Use implementation for current behavior when older design/handoff text conflicts. In particular,
older `eta_ir` descriptions using aligned magnetization, persistent-cache language, and proposals
for heterogeneous pooled CMI do not define the current contract. Update this reference and the
relevant companion when changing semantics. Read the
[study workflow skill](../../../.codex/skills/ma-cc-study-workflow/SKILL.md) before executing or
changing study analysis.

## Execution layers and calls

```text
resolved scientific cell → repeated episodes → round + micro-slot records
  → validated canonical observations → per-cell/state estimates → derived observables
  → explicitly weighted cross-cell summaries → plots/reports/ZIP
```

| Layer | Call / selector | Product and boundary |
| --- | --- | --- |
| Episode measurements | Game `METRICS`; recorder and runtime | Streaming trajectories plus richer retained records; no offline estimator implied |
| Generic cell summaries | Experiment `aggregation:`; `mas-cc experiment aggregate --run-dir <run>` | `cells/<cell>/aggregate.json`, available `metrics/plots/`; trajectory percentiles, not study CMI |
| Relational run analysis | `mas-cc analysis relational-round-feedback --run-dir <run>` | CSV/Markdown diagnostics; optional convenience `pooled` rows are not final study estimates |
| Final study analysis | `mas-cc study aggregate --study-dir <study> [--backend auto\|local\|slurm]` | Validated `analysis/` package; offline, no provider calls |
| Intentional partial analysis | Add `--allow-incomplete` | Provisional package; never upgrades completeness |
| Legacy analysis format migration | `mas-cc study compact-analysis --study-dir <study>` | CSV → Parquet package; no estimator recomputation |
| Run retention migration | `mas-cc experiment compact --run-dir <run> --profile results_only` | Changes retained run artifacts; distinct from study aggregation |
| Presentation | `mas-cc study report --config <report.yaml>` | Separate `report.md`, `.tex`, `.pdf`, manifest, `figures/` |

Experiment `analysis: {enabled: false, estimators: []}` disables automatic run analysis, not the
study's separate recipe. `metrics.enabled` is not a reliable global recording switch in current
wiring. `metrics.comet_export: []` selects no episode metrics for Comet; local files are authoritative.

Backend semantics: `auto` uses local execution except on a detected Potsdam runtime with `sbatch`
available, outside an existing SLURM allocation (and outside pytest). There it submits a detached
prepare → information-group array → finalizer graph. `local` forces synchronous execution;
`slurm` explicitly submits. Local execution can use `SLURM_CPUS_PER_TASK` for cell parallelism.
A submitted command returning 0 means submission succeeded, not that analysis completed. Follow
returned job IDs/progress and inspect the published package. Local CLI exits: 0 complete, 1
provisional, 2 validation/configuration/technical failure. Strict validation writes diagnostics
before stopping; an older package may still exist and must not be mistaken for new output.

Local work uses the existing project environment. On Potsdam only, prefix commands with
`/home/ojedamarin/.local/share/miniforge3/bin/conda run --live-stream -n MA-CC`;
scientific outputs/logs stay under `/work`. See the skill for runtime-directory and submission rules.

## State and measurement vocabulary

`N`: agents; `K`: answer options; `Z`: analysis target (controller target, or gold fallback);
`n_Z`: target votes; `x=n_Z/N`; `U∈{0,1}`: silence/advocacy; `e=P(U=1|sensor)`:
recorded propensity (`P_U1_given_Y`, action field `U_k`); `b`: intervention budget;
`rho`: epistemic persistence. `m_Z=(K*x-1)/(K-1)`, hence `Δm=K/(K-1)*Δx`.
Never interchange target and truth coordinates for false-target control.

The registry has 27 streaming metrics, no game-specific final metrics:

| Family | Exact registered names | Meaning |
| --- | --- | --- |
| Votes (8) | `population_action_share_per_option`, `agent_current_action`, `dominant_action_share`, `truth_vote_share`, `m_truth`, `m_ctrl`, `m_order`, `normalized_vote_entropy` | Option/agent votes, dominant/gold shares, normalized alignments, normalized vote entropy |
| Latest update (5) | `delta_m_truth`, `delta_m_ctrl`, `delta_m_order`, `focal_changed`, `focal_adopted_target` | Alignment increments; focal switch/entry onto target |
| Evidence (8) | `mean_supporting_fact_coverage`, `full_proof_agent_share`, `peer_fact_exposures`, `controller_fact_exposures`, `new_peer_facts`, `new_controller_facts`, `reactivated_peer_facts`, `reactivated_controller_facts` | Active proof coverage/ownership; presented, newly acquired, and reactivated fact counts |
| Board (6) | `q_effective`, `board_size_before`, `board_size_after`, `focal_posted_message`, `controller_message_posted`, `controller_message_directly_exposed` | Realized social sample, live message counts, posting/direct-delivery indicators |

Evidence state: active inventory `A_i=active_fact_ids` is a subset of cumulative historical
`H_i=known_fact_ids`. Each active agent-fact pair survives forgetting independently with probability
`rho`; history does not shrink. Exposure counts presentations, acquisition adds a fact absent from
history, reactivation restores a historical but inactive fact. Public prose may convey inferences
not represented by acquired fact IDs.

Proof coverage `c_i` is the fraction of configured proof requirements held actively. With grouped
alternatives, one concrete fact suffices per latent group; empty proof means coverage 1.
`kappa=mean(c_i)`; `phi=mean(c_i>=1)`; `1-phi` is incomplete-proof share, not behavioral mobility.
Reach counts concrete-fact holders, whereas coverage may count latent groups.
`knowledge_stratum_counts[k]` counts agents covering k requirements; `truth_counts_by_stratum[k]`
counts their gold votes. `knowledge_share_k{k}=n_k/N`; `truth_share_k{k}=t_k/n_k`, missing if `n_k=0`.
These helpers are retained observables, not additional registered metrics.

Retained active summaries use `active_*_before`, `*_after_interactions`, `*_after`; historical
coverage uses `historical_*`. Current unsuffixed `mean_supporting_fact_coverage` and
`full_proof_agent_share` mean active; round `supporting_fact_reach` and `mean_known_fact_count`
mean historical. Helper scope can differ: inspect the field boundary, not just its name.
Live `reactivated_*_facts` corresponds to retained `reactivated_*_fact_count`.

## Retention, identity, and canonicalization

| Profile | Retained contract |
| --- | --- |
| `results_only` | `scientific_events.parquet`, rich `round_records/<episode>/round_trajectory.jsonl`, compact `micro_slot_trajectory.jsonl`, manifests/summaries/seals/hashes; compact numeric metric JSON. Omits private prompts/responses, metric CSVs, and `agent_current_action` payload |
| `dashboard_semantic` | Same scientific retention plus public `dashboard_semantic.jsonl` and hash/count completion file; no private reasoning |
| `full` | Verbose episode trajectories, metric CSVs, prompts, checkpoints, richer inspection artifacts |

`scientific_events.parquet` alone is insufficient for all relational estimators. Round records retain
before/after votes/counts, target/gold, policy/action/propensity/budget, evidence inventories and
flows, communication, and shared-initialization identity. Micro records retain focal transitions,
controlled-slot eligibility, exposure/posting, message identities, and fact effects.
Symbolic reconstruction specifically needs `initial_active_fact_ids_by_agent`,
`active_fact_ids_by_agent_after`, `persistence_deactivated_pairs`, and the frozen task dataset.

Required study identity: `study_manifest.json` + `submission_manifest.csv`. Discover outputs from
manifest paths, not an assumed `runs/` tree. Qualified `cell_id` includes source identity; a local
`cell-0000` is not globally unique. SLURM IDs/shards are never scientific grouping coordinates.
Validate expected cells/episodes, schemas, duplicates, config/artifact hashes, seals, and required
round/micro records. Select completed canonical episodes; remove superseded/failed attempts.
Interrupted prefixes are censored diagnostics, not inputs to completed-episode estimators.

All following table names mean `analysis/tables/<name>.parquet`; readers accept legacy CSV:

| Contract | Tables |
| --- | --- |
| Canonical observations | `cells`, `episodes`, `rounds`, `micro_slots` |
| Core analysis (possibly empty) | `primary_estimates`, `information_estimates`, `support_diagnostics`, `derived_observables` |
| Partial-run diagnostics when applicable | `available_round_prefixes`, `available_micro_slot_prefixes`, `interrupted_episode_diagnostics`, `interrupted_episode_summary` |

Estimate rows carry identity, `metric`, estimator version/variant, `grouping_json`,
`conditioning_json`, `estimate`, CI/confidence, null summaries/p-value, resampling counts,
`n_observations`, `n_episodes`, `units`, `support_status`, `analysis_hash`. Join dependencies at
identical study/run/cell/grouping/conditioning; never join only on a metric label.

Reaggregation can use the four retained canonical tables and validation/provenance without run
trees. It recomputes estimator resampling; absent measurements cannot be recovered. Symbolic
outputs additionally require the frozen task dataset to remain resolvable. Final packages retain
compact summaries, not individual bootstrap/null draws or analysis caches. Detached SLURM work
uses resumable `analysis/.work/<generation>/` inputs/group fragments/progress until successful
publication; those intermediates are not part of the final ZIP.

## Aggregation mathematics and estimators

Generic cell summaries default to `forward_fill: absorbing`, `relabel_by_winner: true`,
`percentiles: [10,50,90]`, `rolling_window: 20`, and cell metrics
`[dominant_action_share, action_share_relabelled, active_fraction, consensus_round, converged_fraction]`.
Smooth each episode, align lengths, then calculate across-episode percentiles. Winner relabeling is
per episode; `active_fraction` exposes how many trajectories really reached each round before fill.
These curves do not replace raw-round scientific estimation.

Study estimates run independently within each physical cell and requested state slice. MI/CMI use
the established direct-counting engine (including its supported smoothing variants), in bits.
Ordinary uncertainty resamples whole episodes. Sensing nulls shuffle sensor observations;
actuation nulls redraw each action from its recorded probability. Never bootstrap independent
rounds, average cell p-values, or treat CMI as causal identification.

| Exact estimator name(s) | Definition / distinction |
| --- | --- |
| `round_sensing_mi`, `round_target_sensing_mi` | Empirical `I(population counts;sensor counts)` versus scalar `I(n_Z;sampled n_Z)` |
| `round_sensor_action_mi` | `I(sensor;U)` policy channel |
| `round_population_actuation_cmi`, `round_target_actuation_cmi`, `round_truth_actuation_cmi`, `round_order_actuation_cmi` | `I(U;next state\|current state)` for full counts, target count, gold count, or majority strength; target channel is `T_pi` |
| `round_{memory,epistemic,phi,susceptible,kappa}_target_actuation_cmi` | Five actual names obtained by expanding braces; condition on current target count plus exact proof-depth histogram, joint `(kappa,phi)` bins, phi bin, `(1-phi)` bin, or kappa bin |
| `round_controller_action_entropy`, `round_controller_action_entropy_given_population` | `H(U)` and `H(U\|population)` |
| `round_population_information_fraction`, `round_target_information_fraction` | Channel CMI / corresponding conditional action entropy; target fraction is `eta_IF` |
| `round_target_susceptibility` | Event-weighted state-matched `chi(n)=E[Δx\|U=1,n]-E[Δx\|U=0,n]`; only dual-action states |
| `round_{target,truth,order}_signed_actuation` | Matched response in aligned-order units, not fraction units |
| `round_target_signed_response_share` | Marginal target-fraction difference, without matching states |
| `round_{memory,epistemic,phi,susceptible,kappa}_target_signed_response` | Fraction response matched on the same augmented states as the respective CMI |
| `round_sensor_mae`, `round_sensor_mse` | Absolute/squared sensor error |
| `episode_current`, `cell_current` | Final minus initial target count per episode; mean episode current + episode bootstrap |
| `effective_affinity`, `kinetic_compliance` | Controlled ADVOCATE micro-slots: `p+=entries/non-target opportunities`, `p-=exits/target opportunities`; `h_eff=ln(p+/p-)` nats, `gamma_eff=p++p-` |

Braces above are compact enumeration notation, not literal YAML names. Compatibility aliases:
`round_target_actuation_cmi_memory` → `round_memory_target_actuation_cmi`;
`round_target_actuation_cmi_memory_phi` → `round_phi_target_actuation_cmi`;
`target_signed_actuation` → `round_target_signed_actuation`.
Run-only `micro_slot_focal_actuation_cmi` and `micro_slot_target_signed_response` are not accepted
study estimator names. Raw affinity currently returns positive infinity if either observed rate
is zero; missing opportunity denominators give NaN. Inspect counts, even with non-unsupported status.

Scalar phi/susceptible/kappa bins: `[0,1/3)`, `[1/3,2/3)`, `[2/3,1]`; joint kappa/phi uses four
bins per axis by default. `state_local: [x]` uses configurable target-share bins;
`x_phi`/`x_kappa` use exact target count plus coarse evidence bin. For B x-bins,
`j=min(floor(B*x),B-1)`, center `(j+0.5)/B`. Do not confuse these with symbolic phi-star bands.

Support statistics: `round_dual_action_state_fraction`, `round_dual_action_event_fraction`,
`round_single_action_slice_fraction`, `round_conditioning_state_count`, `round_singleton_fraction`.
Ordinary/conditioned CMI: unsupported if fewer than two observed actions or zero dual-state fraction;
limited if dual-state fraction <0.25 or singleton fraction >0.5; otherwise adequate. Causal/symbolic
local contrasts instead require both arms: zero in either → unsupported; one in either → limited;
at least two each → adequate. These are implementation labels, not universal sufficiency guarantees.

## Derived quantities and cross-cell summaries

Let `p(n)` be occupancy and `a(n)` empirical advocacy frequency from the same rows as `T_pi`.
Current single-affinity definitions:

```text
B_IR(n) = 2*a(n)*(1-a(n))*chi(n)^2 / ln(2)                   [bits]
eta_ir  = sum_n p(n)*B_IR(n) / I(U;next n_Z | n_Z)          [dimensionless]
J_c,k   = N * sum_n p_k(n)*a(n)*chi(n)                      [target count/round]
eta_th  = h*J_c,horizon / (h*J_c,horizon + I_sens,horizon)  [dimensionless]
```

Use fraction-response `chi`, not `delta_m_ctrl` response, in `eta_ir`. Identified occupancy mass,
positive denominator, and support travel with the ratio. Exported audit components include
`eta_ir_pinsker_numerator_bits`, `eta_ir_denominator_T_bits`; local ratio is `eta_ir_state_local`.
`target_sensing_information_nats` uses scalar target sensing with the exact hypergeometric sensor
kernel and empirical occupancy, not empirical full-vector `round_sensing_mi`.
`controlled_current` is the round mean; `controlled_current_horizon` and
`target_sensing_information_horizon_nats` sum over the analysis horizon.
Other outputs: `susceptibility_occupancy_weighted`, `affinity_weighted_current_nats`,
`thermodynamic_control_expenditure_nats`, `eta_th_signed`, `eta_th_bounded`.
Thermodynamic quantities require valid calibrated affinity; measured `h_eff`/`gamma_eff` are outcomes,
not sweep knobs. Board mode or persistence <1 uses `theoretical_reference: none`; the names alone
assert neither matched q-voter theory nor measured physical energy.

`derived_study_aggregates` combines already-estimated physical cells. `balanced_cell` gives designed
conditions equal weight; `n_observations` local maps are explicitly descriptive. Marginalized axes
must be declared. Truth/false targets may combine only in controller-target coordinates.
Bootstrap episodes separately within every cell, recompute cell estimates/components/weights, then
combine. Aggregate null draws first, then compute the aggregate p-value; never average p-values.
Report raw `T_pi`, null, and `T_pi-null_mean`. Aggregate efficiencies are ratios of summed components:
`eta_IF_bar=sum(w*T)/sum(w*H)` and `eta_IR_bar=sum(w*B)/sum(w*T)`, not means of ratios.
No null-adjusted `eta_ir` is defined. Occupancy-reweighted local reconstruction can differ from
whole-cell estimation through binning/support. Sample-size stability is empirical, not power analysis.

## Causal, communication, and symbolic families

The opt-in [`blackboard_calibration_outputs`](blackboard_calibration.md) family adds
action/exposure-stratified microscopic calibration, sampling-exposure diagnostics,
whole-block uncertainty and explicitly enabled held-out homogeneous-reference
predictions. It uses sampled controller-message identities, **not** the transient
`controller_message_directly_exposed` flag. New eligible-board counts are retained
for sampling comparisons; older records still support empirical components.
Its `blackboard_model_*` outputs remain distinct from measured susceptibility and
causal response, and do not alter single-affinity thermodynamic efficiencies.

Randomized response (`propensity_weighted_causal_response`) uses
`w_t=U_t/e_t-(1-U_t)/(1-e_t)` and `tau_h=mean[w_t*(x_{t+h}-x_t)]` over eligible completed-episode
rows with a lag outcome (normally h=1,2,3). Require binary action, `0<e<1`, ordered rounds and both
arms. Bootstrap shared-initialization blocks. Only pre-action state may define causal groups;
realized message mode, posts, exposures, readers and evidence gains are post-treatment outcomes.

Available susceptibility excludes `x=1`: state-local value is mean `w*Δx/(1-x)`;
cell summary is `sum(w*Δx)/sum(1-x)`. These are different estimands.
Communication efficiency divides causal response by expected activation cost `mean(U*C/e)`,
rebuilding numerator and denominator in each bootstrap. Costs: actual posts, exposures, per-round
unique readers, new controller facts, reactivations. Provider tokens are not public communication
cost; per-round unique readers do not establish episode-wide unique reach.

Symbolic analysis evaluates exact finite MuSR worlds at
`post_forgetting_pre_intervention_delivery`. Participant active union only: exclude historical
inactive facts, controller-private facts and unacquired board facts. `G=collective_solvable` means
union uniquely determines gold; `phi*=symbolic_individual_solvability_share` is individually solvable
share; `fragmentation_gap=G-phi*`. Configured full-proof `phi` is a different definition.
Outputs also retain union size/fraction, agent-fact occurrences, inventory sizes, holder redundancy,
compatible-world counts, gold probability, normalized answer entropy and after-round deltas.

Robustness: a fact with h holders survives somewhere with probability `1-(1-rho)^h`; Monte Carlo
samples surviving union facts and calls the exact solver. Report configured/reference persistence,
uncertainty and draw counts; currently unsolvable → NaN, currently solvable at rho=1 → 1.
Parameter summaries average rounds within episodes then weight episodes equally. Occupancy groups
by cell, x-bin, phi-star-band and explicit G. Joint drift supplies propensity-weighted activation,
silence and contrast in `(Δx,Δphi*)`. Modulation fits weighted response on x, phi-star, normalized
round; reports `0.1*beta_phi`, rejects rank failure, condition number >1e8 or phi-star range <0.1.
It is activation-effect heterogeneity, not a causal knowledge effect.
Capture uses the first start of m consecutive x≥threshold rounds, evidence loss the first G=0;
shortest common within-cell horizon and explicit censoring. Defaults: threshold .75, m=3.
These finite-horizon maps do not establish phase transitions or a new symbolic `eta_ir`.

## Recipe switches, products, and readiness

All switches below belong at the top level of study `analysis.yaml`, not experiment `aggregation:`.
Core settings: `estimators: [...]`, `derived: [...]`, `resampling: {bootstrap_resamples: 1000,
null_permutations: 1000, confidence: 0.95, seed: 1}` (shown resampling values are defaults).
Optional output presence requires selection, retained inputs, and applicable support.

| Selector / capability | Tables / readiness |
| --- | --- |
| `state_local: [x]`, `state_local_x_bins: 8` | Implemented local estimator rows; `x_phi`, `x_kappa` also implemented but not selected in usual blackboard recipes |
| Available target-state rows and local estimates | `state_occupancy`, `state_occupancy_binned`, `state_local_phase_maps` when applicable; these do not require the population-output switch |
| `blackboard_population_outputs: true` | `blackboard_diagnostics`, `cell_summary` and population views; standard blackboard recipes exercise this family |
| `rho_aggregated_descriptive: true` | `rho_aggregated_state_local_maps`, `rho_aggregated_state_occupancy`, `rho_aggregated_descriptive_summary`; descriptive persistence summaries |
| Both ordinary and phi-conditioned target CMI available | `phi_conditioning_comparison`, including raw/null/adjusted estimates and support |
| `derived_study_aggregates: {enabled: true, ...}` | `study_aggregated_metrics`, optional `state_local_aggregated_metrics`, `state_local_reconstruction`, `sample_size_stability`; Q3 recipe exercises these |
| `blackboard_phase2_outputs: {enabled: true, strata: [...]}` or supported causal/communication requests | `causal_response_round_inputs`, `causal_response_effects`, `causal_response_support`, `available_causal_susceptibility_state_local`, `available_causal_susceptibility_summary`, `communication_funnel`, `communication_efficiency`, `response_cost_frontier`, `communication_mode_descriptive_response` |
| `blackboard_epistemic_phase_outputs: {enabled: true, task_dataset_dir: <frozen-tasks>, ...}` | Required dataset path; `epistemic_round_timeseries`, `epistemic_parameter_summary`, `epistemic_state_occupancy`, `epistemic_joint_drift`, `epistemic_causal_susceptibility`, `epistemic_modulation`, `epistemic_capture_timing`, `epistemic_capture_summary`; ASTRA recipes exercise this family |
| Conditioned signed responses / susceptible CMI | Implemented estimator names; usual blackboard recipes do not request them |
| Microscopic CMI/response names above | Run-level diagnostics only; not standard study recipe estimators |
| `study report` | Implemented section kinds: `validation_summary`, `state_budget_phase_suite`; arbitrary outcome tables, line sections and free-form narrative templates are not implemented |

Symbolic settings default to `robustness_draws: 500`, `reference_persistence: 0.85`, `x_bins: 8`,
`phi_bands: 3`, `capture_threshold: 0.75`, `capture_consecutive_rounds: 3`.
Existing recipe exemplars (inspect before adapting; none enables everything):

- [Q3 balanced aggregates and stability](../../../configs/runs/relational_reasoning/blackboard_game/blackboard_adaptive_communication_q3_deepinfra/analysis.yaml).
- [ASTRA symbolic analysis](../../../configs/runs/relational_reasoning/blackboard_game/astra_task003_false_control_30x30/analysis.yaml).
- [ASTRA causal/communication analysis](../../../configs/runs/relational_reasoning/blackboard_game/astra_task003_false_control_30x30_extended_b24/analysis.yaml).
- [Maintained report config](../../../configs/reports/state_budget_phase_report.example.yaml).

## Packaging and reporting contract

Final `analysis/`: `validation.json`, `validation.md`, `analysis_manifest.json`, resolved
`analysis_recipe.yaml`, `tables/`, `plots/`, `reports/`, `provenance/`, `<study-id>_analysis.zip`.
The ZIP excludes source run trees, provider logs, scheduler artifacts, checkpoints and transient
resampling work. Built-in aggregation reports are distinct from the separately configured report.

Report schema: `schema_version: 1`; `report: {id, title, source_analysis, output_dir}` and `sections`.
Input must be a study root or extracted `analysis/` directory, not a ZIP path. Output must lie
outside the source package. `latexmk` and a working LaTeX installation are required; PDF failure
is a command failure. Phase suites require existing bins over `[0,1]` (minimum eight), resolved or
already-aggregated views, and facets preserving remaining varying scientific coordinates.
Reporting selects/filter/reshapes source rows, converts display units, or subtracts a same-row null;
it does not estimate MI/CMI, bootstrap, rebin, average duplicate coordinates or synthesize missing
metrics. Missing/unsupported cells are gray, never zero. Keep structural absence, unvisited state,
limited support and unsupported estimates distinguishable. Provisional status carries into reports;
every displayed value must have a source row recorded in the report ledger.

Agent acceptance order: validation → manifest/recipe → cell/episode identities → support → primary
and derived rows → plots/report → provenance/archive. For missing outputs first inspect recipe
selection, required fields, support and qualified-cell joins. Change supported analysis requests and
reaggregate offline when observations exist; new simulation is needed only for absent source data
or new scientific conditions. Do not substitute a nearby metric or rebuild an estimator in reporting.

## Implementation and verification map

Paths below are repository-relative; test files are existing coverage, not a claim they were run
while reconciling this documentation.

| Concern | Implementation | Existing tests |
| --- | --- | --- |
| Registry / retained evidence | `src/mas_cc/games/relational_reasoning/imitation_round_feedback/{metrics,runtime,state,game}.py`; `src/mas_cc/observability/recorder.py` | `tests/mas_cc/test_results_only_resume.py` |
| Generic summaries | `src/mas_cc/metrics/cell.py`, `src/mas_cc/experiments/aggregation.py`; defaults in `config/models.py` | — |
| MI/CMI/conditioned response | `src/mas_cc/games/hidden_bench/imitation_round_feedback/analysis.py`; relational adapter in corresponding relational `analysis.py` | `tests/mas_cc/test_relational_round_feedback_analysis.py` |
| Affinity / derived theory | `src/mas_cc/analysis/{effective_affinity,single_affinity}.py` | `tests/mas_cc/test_single_affinity_consistency.py` |
| Causal / symbolic | `src/mas_cc/analysis/{causal_response,epistemic_phase}.py` | `tests/mas_cc/test_causal_response.py`, `test_epistemic_phase.py` |
| Study / cross-cell aggregation | `src/mas_cc/studies/{aggregation,canonical,validation,derived_aggregation,table_io}.py` | `tests/mas_cc/test_studies.py`, `test_derived_study_aggregation.py` |
| Detached analysis | `src/mas_cc/studies/{analysis_slurm,analysis_worker}.py` | `tests/mas_cc/test_analysis_slurm.py` |
| Reporting / CLI | `src/mas_cc/studies/reporting.py`, `src/mas_cc/cli/main.py` | `tests/mas_cc/test_study_reporting.py` |
