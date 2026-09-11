# Metrics and analysis for the relational blackboard game

This page covers only the current blackboard experiment family:

```yaml
prompt:
  prompt_family: relational_blackboard_ballot
  response_contract:
    type: relational_blackboard_ballot

game:
  type: relational_imitation_round_feedback
  options:
    task_family: musr_team_allocation
    social_mode: board
```

The settings around this core can vary. For example, an experiment may change population size,
epistemic persistence, controller policy, communication mode, budget, model, or number of rounds.
This page therefore describes the family in general rather than one config.

The central pipeline is:

```text
one episode's metrics and detailed event records
    -> repeated episodes in one scientific cell
    -> validated study tables
    -> per-cell estimates and derived quantities
    -> plots, reports, and a packaged analysis
```

A **scientific cell** is one fixed combination of experimental conditions. Repeated episodes in
that cell change the random seed, not the conditions being tested.

## Contents

1. [What is measured during an episode](#1-what-is-measured-during-an-episode)
2. [Important data that is not a metric](#2-important-data-that-is-not-a-metric)
3. [What each storage profile keeps](#3-what-each-storage-profile-keeps)
4. [How repeated episodes are summarized](#4-how-repeated-episodes-are-summarized)
5. [How final study analysis works](#5-how-final-study-analysis-works)
6. [Statistical estimates in simple words](#6-statistical-estimates-in-simple-words)
7. [Derived quantities and blackboard outputs](#7-derived-quantities-and-blackboard-outputs)
8. [Commands and output files](#8-commands-and-output-files)
9. [Important interpretation rules](#9-important-interpretation-rules)
10. [Source map](#10-source-map)

---

## 1. What is measured during an episode

The game registers 27 **streaming metrics**, meaning values recorded after each game update. The
list is `METRICS` in
`src/mas_cc/games/relational_reasoning/imitation_round_feedback/metrics.py`.

There are no game-specific final metrics in that list.

### 1.1 Votes and population state

| Metric | Plain meaning |
| --- | --- |
| `population_action_share_per_option` | Fraction of agents whose current vote is each answer option. One series is recorded per option. |
| `agent_current_action` | Current vote of each individual agent. |
| `dominant_action_share` | Fraction supporting the currently most common answer. |
| `truth_vote_share` | Fraction currently voting for the correct answer. |
| `m_truth` | Alignment with the correct answer. Uniform voting maps to 0 and complete truth agreement maps to 1. |
| `m_ctrl` | Alignment with the controller target. When analysis has no controller target, it uses the correct answer as the analysis target. |
| `m_order` | Alignment with whichever answer currently has the most votes. |
| `normalized_vote_entropy` | How spread out the votes are, from 0 for unanimity to 1 for the maximum possible spread. Entropy means uncertainty in a distribution. |

The three `m_*` values are normalized order parameters. An **order parameter** is one number that
summarizes the population's collective state.

### 1.2 Changes caused by the latest update

| Metric | Plain meaning |
| --- | --- |
| `delta_m_truth` | Change in truth alignment during the latest update. |
| `delta_m_ctrl` | Change in controller-target alignment during the latest update. |
| `delta_m_order` | Change in population order during the latest update. |
| `focal_changed` | Whether the focal agent changed its vote. The **focal agent** is the agent updated by the current event. |
| `focal_adopted_target` | Whether that agent moved from a non-target vote onto the analysis target. |

### 1.3 Evidence movement

An agent's **active facts** are facts it can currently use. A known fact can become inactive because
of epistemic persistence, the rule controlling whether learned evidence remains available.

| Metric | Plain meaning |
| --- | --- |
| `mean_supporting_fact_coverage` | Mean fraction of the required proof that agents currently hold as active facts. |
| `full_proof_agent_share` | Fraction of agents whose active facts cover the complete proof. |
| `peer_fact_exposures` | Number of fact exposures caused by participant messages during the update. |
| `controller_fact_exposures` | Number of fact exposures caused by controller communication. |
| `new_peer_facts` | Number of previously unknown facts learned from participants. |
| `new_controller_facts` | Number of previously unknown facts learned from the controller. |
| `reactivated_peer_facts` | Number of known but inactive facts brought back into active use by participants. |
| `reactivated_controller_facts` | Number of known but inactive facts brought back into active use by the controller. |

### 1.4 Blackboard and social sampling

| Metric | Plain meaning |
| --- | --- |
| `q_effective` | Number of social observations actually available or used. This can be smaller than the requested group size. |
| `board_size_before` | Number of live blackboard messages before the focal update. |
| `board_size_after` | Number of live blackboard messages after the focal update. |
| `focal_posted_message` | Whether the focal participant posted a message. |
| `controller_message_posted` | Whether the controller posted a message. |
| `controller_message_directly_exposed` | Whether a controller message was directly included in the focal agent's sample. |

The metric module also calculates richer evidence summaries such as `supporting_fact_reach`,
`mean_known_fact_count`, `knowledge_share_k*`, `truth_share_k*`,
`knowledge_stratum_counts`, and `truth_counts_by_stratum`. These are helper observables used in
round records and later analysis. They are not entries in the 27-item `METRICS` list.

### 1.5 Symbolic epistemic state metrics

Studies can enable `blackboard_epistemic_phase_outputs` in `analysis.yaml`. **Symbolic** means the
program checks every compatible finite task world instead of asking a language model whether the
evidence looks sufficient. These values are calculated offline from exact active fact identifiers.

The main output, `epistemic_round_timeseries.parquet`, contains one row per episode and round. Its
pre-action boundary is after that round's forgetting step and before controller communication is
delivered. The evidence scope is only the union of participant active inventories. It excludes the
controller's private pool, inactive historical facts, and blackboard facts that no participant has
acquired.

| Metric | Plain meaning |
| --- | --- |
| `collective_solvable` | Whether the union of all active participant facts uniquely determines the gold answer. This is $G_t$. |
| `symbolic_individual_solvability_share` | Fraction of agents whose own active facts uniquely determine the gold answer. This is $\phi_t^*$. |
| `fragmentation_gap` | $G_t-\phi_t^*$: evidence is collectively available but not assembled inside agents when this is large. |
| `solvable_agent_count` | Number of individually solvable agents. |
| `unsolvable_agent_share` | Fraction of agents not individually solvable. This is bookkeeping, not automatically the fraction able to change vote. |
| `active_union_fact_count` | Number of distinct facts active anywhere in the population. |
| `active_union_fact_fraction` | Active union size divided by the frozen task's full true-fact catalog size. |
| `active_fact_occurrence_count` | Number of active agent-fact pairs, so duplicate holders count separately. |
| `active_mean_fact_count` | Mean active facts per agent. |
| `active_min_fact_count`, `active_max_fact_count` | Smallest and largest active inventory sizes. |
| `active_mean_holder_redundancy` | Mean number of active holders for each fact currently in the population union. |
| `active_min_holder_redundancy` | Smallest holder count among facts in the active union. |
| `collective_compatible_world_count` | Number of valid task worlds consistent with the population's active union. |
| `collective_gold_probability` | Exact gold-answer probability among compatible valid worlds under the frozen prior. |
| `collective_normalized_entropy` | Remaining uncertainty over the three answers, scaled from 0 to 1. |
| `configured_robustness` | Chance current collective solvability survives one independent forgetting boundary at the cell's configured persistence. |
| `reference_robustness` | The same one-boundary check at one fixed declared persistence, permitting redundancy comparisons across cells. |
| `delta_symbolic_individual_solvability_share` | Immediate after-round change in $\phi^*$; this is an outcome, never a causal grouping variable. |
| `symbolic_minus_recorded_full_proof_share` | Difference between symbolic solvability and the older all-supporting-facts coverage measure. |

`collective_solvable = false` means current active evidence is insufficient. It does not mean that
the truth cannot be guessed or recovered later.

Additional tables separate different questions:

- `epistemic_parameter_summary.parquet`: equal-episode budget-versus-persistence summaries.
- `epistemic_state_occupancy.parquet`: observed $x$-versus-$\phi^*$ state support, with $G$ kept explicit.
- `epistemic_joint_drift.parquet`: silence, activation, and randomized causal-displacement arrows
   for vote share and symbolic proof ownership.
- `epistemic_causal_susceptibility.parquet`: raw binned randomized response over pre-action
   $x$ and $\phi^*$.
- `epistemic_modulation.parquet`: the fitted change in randomized response per 0.1 increase in
   $\phi^*$, with rank and collinearity checks.
- `epistemic_capture_timing.parquet`: per-episode first evidence loss and persistent false-capture
   timing.
- `epistemic_capture_summary.parquet`: cell-level timing and censoring summaries.

These are finite-horizon regime maps. Smooth colors do not by themselves establish a phase
transition. They are also kept separate from the existing `eta_IR` efficiency calculation because
that quantity would need a consistent new derivation before accepting these responses.

---

## 2. Important data that is not a metric

The final analysis does not depend only on the 27 metric objects. The game deliberately saves rich
round and micro-slot records. A **micro-slot** is one individual agent-update opportunity inside a
population round.

This distinction matters:

- A metric is a convenient trajectory summary.
- A retained field is exact source data from which later estimates can be recalculated.
- An estimator combines many retained observations after the simulation has finished.

### 2.1 Round records

The round records preserve:

- votes and option counts before and after each round;
- the correct answer and controller target;
- target, truth, order, and vote-uncertainty values before and after;
- controller action, action probability, sensor sample, policy settings, and remaining budget;
- active and historical evidence coverage;
- proof-depth counts such as how many agents hold zero, one, or all supporting facts;
- facts exposed, learned, reactivated, or deactivated;
- blackboard posts, exposures, reports, directives, requests, and readers;
- initialization identity, which links episodes that started from the same physical state.

These records support information, response, current, causal, communication, occupancy, and
endpoint analyses without rerunning the language model.

### 2.2 Micro-slot records

The compact micro-slot records preserve the focal agent and vote before and after, population counts,
controller action and target, whether the slot was controlled, target-entry and target-exit changes,
sampled blackboard message identities, controller-message exposure, newly learned or reactivated
controller facts, posting behavior, and board sizes.

These fields support effective affinity, kinetic compliance, microscopic response, and
communication-funnel checks.

---

## 3. What each storage profile keeps

### 3.1 `results_only`

This is the compact scientific profile used by large studies. It keeps:

- atomic `scientific_events.parquet`, where **Parquet** means a compressed table format for analysis;
- rich `round_trajectory.jsonl` records under each cell's `round_records/<episode-id>/` directory;
- compact `micro_slot_trajectory.jsonl` records;
- manifests, summaries, completion seals, and file hashes;
- numeric population and option metrics inside compact JSON fields. JSON means JavaScript Object
  Notation, a structured data format.

It does not keep full private prompts, private model responses, per-episode metric CSV files, or the
agent-level `agent_current_action` metric payload.

The Parquet table alone does not contain every field needed by relational analysis. Study
aggregation also reads the retained round and micro-slot JSONL records. **JSONL** means one JSON
record per line.

### 3.2 `dashboard_semantic`

This profile keeps the same compact scientific data as `results_only` and adds a dashboard stream:

```text
round_records/<episode-id>/dashboard_semantic.jsonl
round_records/<episode-id>/dashboard_semantic_complete.json
```

The stream contains public semantic state: public votes, blackboard messages, active and known fact
identifiers, sampled message identities, public actions, controller sensing and policy fields, and
validation issue codes. It deliberately excludes private prompts, private model responses, and
private reasoning traces.

The completion file records the row count and SHA-256 hash. **SHA-256** is a content fingerprint
used to detect missing or changed data.

### 3.3 `full`

This inspection profile additionally keeps verbose episode files, prompts, checkpoints, metric CSV
files, full trajectories, and richer intermediate records. Current short smoke and prompt-inspection
runs may use it, but compact profiles are the normal choice for larger studies.

---

## 4. How repeated episodes are summarized

The run's `aggregation:` section controls generic cell summaries. Current blackboard configs do not
normally provide this section, so they inherit these defaults:

```yaml
aggregation:
  forward_fill: absorbing
  relabel_by_winner: true
  percentiles: [10, 50, 90]
  rolling_window: 20
  cell_metrics:
    - dominant_action_share
    - action_share_relabelled
    - active_fraction
    - consensus_round
    - converged_fraction
  sweep_metrics: []
```

The generic cell summary:

1. uses completed compact episodes and excludes failed or `skipped_aborted` full-profile episodes;
2. smooths each episode separately over the configured rolling window;
3. aligns episodes with different lengths;
4. calculates the requested percentiles across repeated episodes.

`action_share_relabelled` calls each episode's eventual winner `option_1`, its runner-up `option_2`,
and so on. This prevents different answer labels from cancelling one another when episodes are
averaged.

`active_fraction` shows the fraction of episodes that originally reached each round. It should be
read beside a forward-filled curve, because an ended episode's final value can be carried forward.

Each cell writes `aggregate.json`. The program also attempts to write available aggregate curves
under `metrics/plots/`.

No current blackboard config requests generic run-level sweep metrics such as terminal mutual
information. Cross-cell scientific comparisons are performed by study analysis instead.

---

## 5. How final study analysis works

The authoritative final command is:

```text
mas-cc study aggregate --study-dir <study-result-root>
```

Study aggregation performs these steps:

1. Read the study and submission manifests.
2. Discover ordinary or execution-sharded run and cell outputs.
3. Validate expected cells, episodes, completion seals, schemas, identities, and file hashes.
4. Build canonical tables for cells, episodes, rounds, and micro-slots. **Canonical** means these are
   the standard retained tables from which analysis can be reproduced.
5. Group data by physical scientific cell. SLURM (the cluster job scheduler) task numbers and job
   identifiers are never scientific grouping variables.
6. Calculate each requested estimate independently in each cell or requested state group. Data from
   different conditions is not mixed into one estimator.
7. Calculate uncertainty using the resampling unit required by each estimator. Most estimates use a
   **bootstrap**, which repeatedly samples complete episodes with replacement. Causal-response
   analysis resamples complete shared-initialization blocks so matched episodes remain together.
8. Run requested null models. A **null model** breaks the tested relationship to show the estimate
   expected from chance and limited data. Sensing nulls shuffle sensor observations. Actuation nulls
   draw replacement actions from each observation's recorded controller-action probability.
9. Calculate derived quantities and blackboard-specific summaries.
10. Render plots and reports, save provenance, and create a ZIP compressed archive.

Aggregation makes no provider or language-model calls. Bootstrap and permutation draws are temporary
and are not retained. Reaggregation recomputes them from the canonical observations.

### Main tables

New packages write compressed Parquet tables under `analysis/tables/`:

| File | Contents |
| --- | --- |
| `cells.parquet` | One row per scientific cell and its coordinates. |
| `episodes.parquet` | One row per episode. |
| `rounds.parquet` | One row per population round. |
| `micro_slots.parquet` | One row per individual update opportunity. |
| `primary_estimates.parquet` | Estimates, confidence intervals, null summaries, sample counts, units, and grouping information. |
| `information_estimates.parquet` | Information estimates such as mutual information and conditional mutual information. |
| `support_diagnostics.parquet` | Evidence showing whether enough states and both controller actions were observed. |
| `derived_observables.parquet` | Quantities calculated from primary estimates. |
| `study_aggregated_metrics.parquet` | Balanced study summaries built after estimating each physical cell, with aggregate bootstrap intervals, aggregate permutation nulls, component values, and support coverage. |
| `state_local_aggregated_metrics.parquet` | Observation-weighted descriptive state maps after explicitly selected study dimensions have been averaged. |
| `sample_size_stability.parquet` | Empirical checks of how estimates and their spread change as more complete episodes are used. |

Recipes can add state-local, occupancy, causal-response, communication, initialization, endpoint,
and blackboard diagnostic tables. The package also contains `validation.json`, `validation.md`, the
resolved analysis recipe, `analysis_manifest.json`, plots, reports, provenance, and an analysis ZIP.

---

## 6. Statistical estimates in simple words

The study folder's `analysis.yaml` selects these estimates. The current blackboard recipes use
subsets of the following list.

### 6.1 Sensing and actuation information

**MI** means mutual information: how much observing one variable reduces uncertainty about another.
**CMI** means conditional mutual information: the same idea after specified current-state information
has been held fixed.

| Estimator | Question it answers |
| --- | --- |
| `round_sensing_mi` | How much does the controller's complete sample reveal about the complete population vote counts? |
| `round_target_sensing_mi` | How much does sampled target count reveal about true target count? |
| `round_sensor_action_mi` | How strongly does controller action depend on what the controller observed? |
| `round_population_actuation_cmi` | Does controller action predict the next complete population state after current population state is held fixed? |
| `round_target_actuation_cmi` | Does controller action predict next target count after current target count is held fixed? |
| `round_truth_actuation_cmi` | Does controller action predict next truth count after current truth count is held fixed? |
| `round_order_actuation_cmi` | Does controller action predict next majority strength after current majority strength is held fixed? |

Additional-conditioning estimates hold current target count fixed and also hold one evidence summary
fixed:

| Estimator | Additional evidence summary |
| --- | --- |
| `round_memory_target_actuation_cmi` | Full histogram of how many supporting facts each agent holds. |
| `round_epistemic_target_actuation_cmi` | Joint coarse bin of $\kappa$ and $\phi$. |
| `round_phi_target_actuation_cmi` | Coarse bin of $\phi$, the fraction of agents holding the full proof. |
| `round_susceptible_target_actuation_cmi` | Coarse bin of $1-\phi$, the fraction not yet holding the full proof. |
| `round_kappa_target_actuation_cmi` | Coarse bin of $\kappa$, the mean fraction of supporting facts held. |

### 6.2 Controller uncertainty and estimator support

| Estimator | Plain meaning |
| --- | --- |
| `round_controller_action_entropy` | Overall uncertainty in controller actions. |
| `round_controller_action_entropy_given_population` | Action uncertainty remaining among events with the same population state. |
| `round_population_information_fraction` | Population-actuation CMI divided by the action uncertainty available to carry information. |
| `round_target_information_fraction` | Target-actuation CMI divided by its action-uncertainty ceiling. |
| `round_dual_action_state_fraction` | Fraction of visited conditioning states observed with both controller actions. |
| `round_dual_action_event_fraction` | Fraction of events lying in those dual-action states. |
| `round_single_action_slice_fraction` | Fraction of visited states observed with only one controller action. |
| `round_conditioning_state_count` | Number of distinct conditioning states visited. |
| `round_singleton_fraction` | Fraction of observations in states seen only once. |

The support diagnostics must be shown beside CMI estimates. A number can be mathematically available
but scientifically weak when nearly every state was observed with only one action.

### 6.3 Signed response and sensor error

| Estimator | Plain meaning |
| --- | --- |
| `round_target_signed_actuation` | Within matching current target-count states, difference between mean change in target alignment after advocacy and after no action. |
| `round_truth_signed_actuation` | The corresponding state-matched difference in mean truth-alignment change. |
| `round_order_signed_actuation` | The corresponding state-matched difference in mean order change. |
| `round_target_susceptibility` | State-matched difference in target-fraction change. This is $\chi$ (chi), the response value used by the current single-affinity theory. |
| `round_target_signed_response_share` | Difference in target-fraction change without matching current states. |
| `round_memory_target_signed_response` | Target response compared within matching target-count and evidence-memory states. |
| `round_epistemic_target_signed_response` | Target response compared within matching target-count and joint $\kappa$/$\phi$ bins. |
| `round_phi_target_signed_response` | Target response compared within matching target-count and $\phi$ bins. |
| `round_susceptible_target_signed_response` | Target response compared within matching target-count and $1-\phi$ bins. |
| `round_kappa_target_signed_response` | Target response compared within matching target-count and $\kappa$ bins. |
| `round_sensor_mae` | Mean absolute sensor error. |
| `round_sensor_mse` | Mean squared sensor error, which gives large errors more weight. |

### 6.4 Population current and microscopic dynamics

| Estimator | Plain meaning |
| --- | --- |
| `episode_current` | One episode's final target count minus its initial target count. |
| `cell_current` | Mean episode current with a whole-episode bootstrap confidence interval. |
| `effective_affinity` | Natural logarithm of the controlled target-entry rate divided by the controlled target-exit rate. It measures directional bias in microscopic vote changes and is reported in nats. A nat is an information unit based on natural logarithms. The current implementation reports positive infinity when either observed rate is zero, so the transition counts must be inspected with the estimate. |
| `kinetic_compliance` | Sum of observed target-entry and target-exit probabilities, $p_+ + p_-$, in controlled micro-slots where the controller advocates the target. It measures how readily votes move, not which direction they favor. |

The run-level relational analyzer also calculates `micro_slot_focal_actuation_cmi` and
`micro_slot_target_signed_response` in `micro_slot_diagnostics.csv`. They are not currently accepted
as estimator names in the standard study recipe.

### 6.5 Causal response

`propensity_weighted_causal_response` estimates the effect of randomized controller activation on
later target vote share, normally at lags 1, 2, and 3. **Propensity** means the recorded probability
that the controller took the active action in that state.

This is the causal estimator. CMI describes conditional predictive dependence and is not, by itself,
proof of causation.

Causal analysis requires both action and silence to have non-zero probability. A cell without both
possibilities is marked unsupported. It also keeps episodes with the same starting state together
during resampling.

---

## 7. Derived quantities and blackboard outputs

### 7.1 Single-affinity quantities

The current single-affinity analysis models control using one measured directional bias. Requesting
one member of its coupled family can produce:

- `susceptibility_occupancy_weighted`: target susceptibility averaged using how often each state was visited;
- `eta_ir`: information-response efficiency;
- `eta_ir_pinsker_numerator_bits` and `eta_ir_denominator_T_bits`: the numerator and denominator used to audit `eta_ir`;
- `eta_ir_state_local`: state-specific information-response efficiency where support is sufficient;
- `target_sensing_information_nats` and `target_sensing_information_horizon_nats`;
- `controlled_current` and `controlled_current_horizon`;
- `affinity_weighted_current_nats`;
- `thermodynamic_control_expenditure_nats`;
- `eta_th`, `eta_th_signed`, and `eta_th_bounded`.

The thermodynamic names express an energy-and-information analogy. They are not direct physical
energy measurements. `effective_affinity` and `kinetic_compliance` are measured outcomes, not grid
settings to sweep.

### 7.2 Communication response and cost

When a recipe requests blackboard phase-two outputs, aggregation can write:

- `causal_response_round_inputs`;
- `causal_response_effects` and `causal_response_support`;
- state-local and cell-level available-susceptibility tables;
- `communication_funnel`;
- `communication_efficiency`;
- `response_cost_frontier`;
- `communication_mode_descriptive_response`.

Communication cost is measured from public intervention outcomes such as controller posts, message
exposures, unique readers within a round, newly learned controller facts, and reactivated controller
facts. Provider token usage includes private prompting and must not be substituted for public
communication cost.

The communication funnel links controller activation, message delivery, and population response.
Realized message mode, posts, exposures, and readers happen after controller activation, so they are
descriptive outputs rather than valid pre-action groups for a causal estimate.

### 7.3 Blackboard diagnostics

Blackboard recipes can also produce `blackboard_diagnostics.parquet`, `cell_summary.parquet`, phase
maps, state-occupancy tables, initialization diagnostics, and other recipe-specific tables. These
summarize the detailed retained fields; they are not additional live metric objects.

### 7.4 Study-level aggregated control metrics

A **physical cell** is one fixed experimental condition, such as one persistence, budget, and
controller target. The estimators above are still calculated inside those cells first. Study-level
aggregation is a later summary step; it does not replace or redefine a physical-cell estimate.

The state-local transfer information $T_\pi(x,\rho,b,s)$ says where in state space controller
information appears. It is detailed but can be sparse. The whole-cell
`round_target_actuation_cmi` already holds the current target count fixed and then averages over the
visited counts:

$$
T_\pi(\rho,b,s)=I(U;n'_Z\mid n_Z,\rho,b,s)
=\sum_x p(x\mid\rho,b,s)T_\pi(x,\rho,b,s).
$$

Here, $U$ is the binary controller action and $n_Z$ is the number of agents supporting the
controller's selected target. The whole-cell value therefore uses all supported visited states in
one condition. Unsupported state-local bins remain unsupported rather than becoming zero.

Recipes may then average physical-cell estimates over persistence $\rho$, target meaning $s$, or
both. `balanced_cell` gives each designed condition equal weight. This is the headline study
summary. `n_observations` weights a state-local map by how many observations supported each local
estimate. It answers what happened across dynamics that actually visited that state and is marked
`descriptive_only: true`. The two weightings answer different questions and are never silently
mixed.

Every aggregated transfer-information value is paired with an aggregated policy-randomization
null. A **policy-randomization null** redraws the action from its recorded state-dependent action
probability while leaving the retained trajectory otherwise fixed. Null replicate $r$ is first
combined across cells,

$$
\bar T_{\pi,\mathrm{null}}^{(r)}=\sum_c w_cT_{\pi,\mathrm{null},c}^{(r)},
$$

and the aggregate permutation p-value is calculated from those combined draws. Cell p-values are
never averaged. The main reported transfer result is
$\Delta\bar T_\pi=\bar T_\pi-E[\bar T_{\pi,\mathrm{null}}]$, shown beside the raw estimate and null.

Uncertainty uses a stratified whole-episode bootstrap. **Stratified** means complete episodes are
sampled with replacement separately inside every physical cell. Each cell estimate is recomputed,
and only then are the cell results combined. This keeps rounds from the same episode together.

The two efficiencies are ratios of aggregated components, not averages of cell efficiencies:

$$
\bar\eta_{\rm IF}=\frac{\sum_cw_cT_{\pi,c}}
{\sum_cw_cH_c(U\mid n_Z)},\qquad
\bar\eta_{\rm IR}=\frac{\sum_cw_cB_{{\rm IR},c}}
{\sum_cw_cT_{\pi,c}}.
$$

The numerator, denominator, and ratio are rebuilt inside every bootstrap replicate. The table
exports both components so the calculation can be checked. No null-adjusted `eta_ir` is created,
because a near-zero or negative adjusted denominator would be unstable without a separate theory.

Truth-target and false-target cells may be combined only in controller-target coordinates. This
uses `target_count_before`, `target_count_after`, `delta_p_ctrl`, and
`round_target_actuation_cmi`. It measures controllability toward the selected target. It does not
say that truth and false control have the same consequences for knowledge. Separate truth and false
rows remain available. `round_truth_actuation_cmi` is never used for this pooled quantity.

Each aggregate reports cell coverage, episode and round counts, action-overlap diagnostics, and an
`adequate`, `limited`, or `unsupported` label. A small susceptibility and an imprecisely estimated
susceptibility are different claims; the estimate and its interval remain separate. Thermodynamic
efficiency `eta_th` is unchanged and stays unsupported when no calibrated affinity $h$ exists.

The optional `sample_size_stability` output repeatedly subsamples complete episodes inside cells.
It reports how spread, interval width, sign stability, and null-detection frequency change with the
available episode count. This is an empirical stability check, not a prospective power calculation.

---

## 8. Commands and output files

### Check one config without model calls

```text
mas-cc experiment preflight --config <config.yaml> --output-dir <inspection-directory>
```

### Rebuild generic summaries for one completed run

```text
mas-cc experiment aggregate --run-dir <run-directory>
```

### Run the dedicated relational analyzer for one run

```text
mas-cc analysis relational-round-feedback --run-dir <run-directory>
```

This run-level analyzer writes CSV and Markdown files for round information, null results, support,
controller actions, epistemic trajectories, currents, and optional theory comparisons. It can also
write a convenience `pooled` row across cells. That pooled row is not the final scientific study
estimate.

### Produce the final study package

```text
mas-cc study aggregate --study-dir <study-result-root>
```

Use `--allow-incomplete` only for intentionally provisional analysis. Strict aggregation is required
for a final result.

Most current study configs contain:

```yaml
analysis: {enabled: false, estimators: []}
```

That disables automatic run-level analysis. Their study folder's separate `analysis.yaml` is the
authoritative final analysis recipe. Some standalone pilots instead enable run-level analysis. They
can request `blackboard_episode_inspection`, which writes episode-inspection artifacts for checking
prompts, responses, votes, evidence, and board activity.

For the complete operational workflow, see
[`study_aggregation_contract.md`](study_aggregation_contract.md).

---

## 9. Important interpretation rules

1. **Metrics are not estimators.** Metrics describe one trajectory. MI, CMI, causal response,
   effective affinity, and efficiencies combine many retained observations later.
2. **Not every analysis input is a metric object.** Round and micro-slot records contain the exact
   state and action fields needed to recompute estimates.
3. **Final study estimates stay inside physical cells.** Scheduler shards and cells with different
   scientific conditions are not pooled together.
4. **CMI is predictive, not automatically causal.** Use the propensity-weighted estimator for the
   randomized controller effect.
5. **Causal groups must be measured before treatment.** A **treatment** is the controller action
   whose effect is being estimated. Posts, exposures, readers, and realized message mode happen
   after that action and cannot define causal comparison groups.
6. **Unique readers are exact per round only.** Compact records do not prove episode-wide unique
   readership or repeated exposure of one reader to one message across rounds.
7. **`round_target_susceptibility` and `round_target_signed_actuation` use different scales.** The
   former is target-fraction change and feeds `eta_ir`; the latter uses aligned-order units.
8. **`target_sensing_information_nats` is not `round_sensing_mi`.** The first is the scalar target
   channel derived from the exact sensor rule. The second is empirical information about the full
   option-count vector.
9. **Effective affinity needs microscopic support.** Study support status is based on whether both
   target-entry and target-exit opportunities were observed. Zero realized transitions can still
   produce limited or adequate support and an infinite estimate, so inspect the exposure and
   transition counts together.
10. **Board mode has no matched q-voter theory claim.** Finite-memory board analysis uses
    `theoretical_reference: none`; so does any run with epistemic persistence below 1.
11. **`dashboard_semantic` adds a dashboard stream to compact scientific retention.** It does not
    replace the scientific, round, or micro-slot records.
12. **Aggregation makes no provider calls.** Existing retained observations are enough to change
    estimators, uncertainty settings, derived outputs, plots, and reports.
13. **`metrics.enabled` is not currently a reliable master switch.** The blackboard configs set it
    to `true`. Current wiring can still record registered metrics when it is false.
14. **`metrics.comet_export: []` means no episode metric is selected for Comet.** Comet is an
    optional remote dashboard; local retained files remain authoritative.

---

## 10. Source map

| Concern | Source of truth |
| --- | --- |
| Registered streaming metrics and evidence helpers | `src/mas_cc/games/relational_reasoning/imitation_round_feedback/metrics.py` |
| Round and micro-slot event construction | `src/mas_cc/games/relational_reasoning/imitation_round_feedback/runtime.py` |
| Storage profiles | `src/mas_cc/config/models.py` |
| Metric recording and compact retention | `src/mas_cc/observability/recorder.py` |
| Dashboard stream | `src/mas_cc/storage/dashboard_semantic.py` |
| Generic cell aggregation | `src/mas_cc/metrics/cell.py`, `src/mas_cc/experiments/aggregation.py` |
| Round-information and response estimators | `src/mas_cc/games/hidden_bench/imitation_round_feedback/analysis.py` |
| Relational run analyzer and currents | `src/mas_cc/games/relational_reasoning/imitation_round_feedback/analysis.py`, `current.py` |
| Effective affinity and kinetic compliance | `src/mas_cc/analysis/effective_affinity.py` |
| Causal response and communication efficiency | `src/mas_cc/analysis/causal_response.py` |
| Single-affinity derived quantities | `src/mas_cc/analysis/single_affinity.py` |
| Study validation and canonical tables | `src/mas_cc/studies/validation.py`, `src/mas_cc/studies/canonical.py` |
| Study orchestration and packaging | `src/mas_cc/studies/aggregation.py` |
| Standard study workflow | `.codex/skills/ma-cc-study-workflow/SKILL.md` |
