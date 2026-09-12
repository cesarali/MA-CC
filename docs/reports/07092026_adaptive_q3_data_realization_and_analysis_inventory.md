# Adaptive-Communication q=3: Data Realization and Analysis Inventory

**Study:** `musr_blackboard_adaptive_communication_q3_deepinfra`  
**Source archive:** `results/musr_blackboard_adaptive_communication_q3_deepinfra_analysis.zip`  
**Extracted copy used for inspection:** `results/musr_blackboard_adaptive_communication_q3_deepinfra_analysis_source/`  
**Derived report:** `results/musr_adaptive_q3_rebuilt_report/`  
**Status:** incomplete and provisional: 723 of 750 planned episodes are present

## 1. Purpose

This note records what data the archive actually contains, how its tables fit together, which quantities are direct measurements, which quantities are estimates, and what was added during the rebuilt analysis.

The central distinction is:

- `episodes.parquet` is mainly an episode index and execution record.
- `rounds.parquet` is the main table of population-level scientific measurements.
- `micro_slots.parquet` is the main table of individual update-level measurements.
- Estimator and summary tables are calculated from those retained measurements.

No new game episodes were run during the reconstruction. The source Parquet files were read without modification.

## 2. Realized study size

The planned study contains three arms:

| Arm | Scientific cells | Planned episodes | Completed episodes | Sealed cells |
|---|---:|---:|---:|---:|
| No control | 5 | 50 | 47 | 3 |
| Truth control | 35 | 350 | 339 | 25 |
| False control | 35 | 350 | 337 | 22 |
| **Total** | **75** | **750** | **723** | **50** |

A **scientific cell** is one fixed combination of experimental settings. For the controlled arms, the main varied settings are persistence `rho` and nominal intervention budget `b`.

The retained hierarchy is internally regular:

- 723 completed episodes;
- exactly 10 round rows for every completed episode;
- 7,230 round rows in total;
- exactly 24 microscopic update rows for every round;
- 173,520 microscopic update rows in total.

There are 27 missing planned episodes, but no retained failed or aborted episodes. The archive must therefore be treated as a partial realization of the planned grid.

## 3. Data hierarchy and safe join keys

The data form this hierarchy:

```text
scientific cell
  -> episode
    -> population round
      -> microscopic agent-update slot
```

Use these composite keys:

| Table | One row represents | Safe key |
|---|---|---|
| `cells.parquet` | One scientific cell | `cell_id` |
| `episodes.parquet` | One realized episode | (`cell_id`, `episode_id`) |
| `rounds.parquet` | One population round | (`cell_id`, `episode_id`, `round_index`) |
| `micro_slots.parquet` | One microscopic update | (`cell_id`, `episode_id`, `round_index`, `micro_slot_index`) |

Do **not** join on `episode_id` alone. Episode labels are reused across cells: there are only 350 distinct `episode_id` strings among 723 episode rows. The composite keys above are unique and contain no null key values.

Some scientific coordinates are attached at cell level. In particular, target semantics should be recovered from `cells.parquet` through `cell_id` rather than guessed from a run name. The value `False` in the source means false-target control; it should be normalized to the text label `false` during analysis.

## 4. Canonical retained tables

### 4.1 `cells.parquet`: experimental coordinates and completeness

**Shape:** 75 rows x 42 columns.

This is the scientific-cell index. It records:

- study, source run, and source cell identities;
- source and resolved configuration hashes;
- task identity;
- expected, completed, and failed episode counts;
- whether the cell is sealed;
- target semantics;
- persistence `rho`;
- intervention budget `b`;
- population size, social group size, and sensor size;
- controller mode, policy, and message settings;
- participant/controller REQUEST and DIRECTIVE permissions;
- board sampling and message lifetime;
- controller threshold and inverse-temperature parameter `beta`.

This table should be the authority for arm labels and fixed cell coordinates.

### 4.2 `episodes.parquet`: episode identity and execution metadata

**Shape:** 723 rows x 23 columns.

The columns are:

```text
study_id
source_extension_index
source_submission_attempt
source_config_index
source_run_id
source_run_path
cell_id
source_cell_id
episode_id
source_episode_id
cell_key
repetition_index
episode_key
episode_seed
status
interaction_count
usage_requests
usage_input_tokens
usage_output_tokens
started_at
finished_at
termination_reason
scientific_schema_version
```

This table answers questions such as:

- Which episode was realized?
- Which scientific cell does it belong to?
- Which random seed and repetition index were used?
- Did it complete?
- How many requests and tokens did it consume?
- When did it start and finish?

It does not contain most opinion, communication, evidence, or controller measurements. Those are in the round and microscopic tables.

### 4.3 `rounds.parquet`: principal population-level measurements

**Shape:** 7,230 rows x 267 columns.

This is the main table for most later analysis. Each row describes one complete population round, including the state before the round, controller decision, communication events, and state after the round.

#### Identity and provenance

Examples include:

```text
study_id, source_run_id, source_run_path, cell_id, source_cell_id,
episode_id, source_episode_id, episode_key, repetition_index,
round_index, record_source, schema_version
```

#### Experimental coordinates

Examples include:

```text
task_id, task_family, N, K, social_group_size, social_mode,
epistemic_persistence, intervention_budget, sensor_sample_size,
controller_beta, controller_threshold, receiver_epistemic_disposition,
message_lifetime_rounds, board_sampling
```

Here `N=24` is population size and `K=3` is the number of possible answers. The controlled cells use social group size `q=3`, sensor size `q_c=12`, and budgets `b` in `{3, 6, 9, 12, 15, 18, 21}`.

#### Population opinion state and outcomes

Important columns include:

```text
possible_answers
correct_answer
analysis_target
controller_target
occupation_counts_before
occupation_counts_after
controller_target_share_before
controller_target_share
truth_vote_share_before
truth_vote_share
m_ctrl_before
m_ctrl_after
m_truth_before
m_truth_after
m_order_before
m_order_after
vote_entropy_before
vote_entropy
n_k
n_k_plus_1
```

These fields support initial/final truth share, target share, plurality outcomes, trajectories, target-count state bins, and response calculations.

#### Binary controller decision and sensing

Important columns include:

```text
Y_k
U_k
P_U1_given_Y
controller_action
controller_action_probability
controller_probability_U1_given_Y
controller_sampled_U
controller_sensor_Y
sensor_agent_ids
sensor_observed_opinions
sensor_count_vector
sensor_target_share
controlled_positions
controlled_position_count
```

`U_k` is the binary theoretical control action: `1` means the controller acts and `0` means no action. This binary variable remains the action used in transfer-information calculations. REQUEST, REPORT, and DIRECTIVE are communication modes chosen only when the binary controller acts; they must not replace `U_k` in the main information estimator.

#### Adaptive communication and board activity

Important columns include:

```text
chosen_message_mode
allowed_message_modes
communication_choice_reason
controller_posts
actual_controller_posts
controller_post_ids
controller_message_mode
controller_message_exposures
controller_unique_readers
request_count
report_count
dawn_directive_count
directive_count
controller_direct_replies
board_mean_size
board_peak_size
board_messages_created
board_messages_expired
message_type_counts
```

These fields support communication-mode probabilities, realized budget, saturation, participant posting, and public-board diagnostics.

#### Exposure, adoption, and reply measurements

Important columns include:

```text
peer_fact_exposures
controller_fact_exposures
peer_evidence_exposures
controller_report_exposures
controller_report_unique_readers
controller_report_target_adoptions
controlled_adoption_count
controlled_target_adoption_rate
new_evidence_acquisitions
reactivated_peer_fact_count
reactivated_controller_fact_count
directive_report_reply_count
directive_attributed_acquisitions
directive_attributed_refreshes
```

These fields support questions about whether messages were merely posted, actually read, followed by evidence acquisition, or followed by target adoption.

#### Epistemic state and evidence coverage

The archive retains both active evidence and historical evidence. **Active evidence** means evidence still available under the persistence process. **Historical evidence** means evidence acquired at any earlier point, even if it is no longer active.

Important columns include:

```text
active_fact_ids_by_agent_after
known_fact_ids_by_agent_after
active_mean_fact_count_before
active_mean_fact_count_after
active_full_proof_agent_share_before
active_full_proof_agent_share_after
active_mean_supporting_fact_coverage_before
active_mean_supporting_fact_coverage_after
historical_full_proof_agent_share_before
historical_full_proof_agent_share_after
historical_mean_supporting_fact_coverage_before
historical_mean_supporting_fact_coverage_after
knowledge_stratum_counts_before
knowledge_stratum_counts
truth_counts_by_stratum_before
truth_counts_by_stratum
persistence_deactivated_fact_count
```

These fields support evidence-retention, proof availability, knowledge-stratum, and cooperation analyses.

#### Known round-table caveats

Eight round columns are entirely null in this archive:

```text
controlled_target_adoption_rate
controller_episode_fact_id
controller_evidence_strategy
controller_fact_id
controller_fact_text
derived_epistemic_condition
epistemic_condition
micro_slot_index
```

`micro_slot_index` is correctly absent at round resolution; use `micro_slots.parquet` for that coordinate. Analyses should check actual non-null support rather than assuming that a column name guarantees measurements.

### 4.4 `micro_slots.parquet`: microscopic update measurements

**Shape:** 173,520 rows x 59 columns.

Each population round contains 24 microscopic slots, one for each sequential update position. This table records the local state and message exposure around each update.

Important groups are:

- **Identity:** `cell_id`, `episode_id`, `round_index`, `micro_slot_index`.
- **Control placement:** `controlled_slot`, `controlled_positions_hash_or_id`, `round_controller_action`, `intervention_budget`.
- **Opinion transition:** `focal_opinion_before`, `focal_opinion_after`, `occupation_counts_before`, `occupation_counts_after`.
- **Currents:** `delta_m_ctrl`, `delta_m_truth`, `delta_m_order`, `truth_current_increment`.
- **Board state:** `board_size_before`, `board_size_after`.
- **Message exposure:** `sampled_message_ids`, `sampled_message_types`, `sampled_message_authors`, `sampled_message_ages`.
- **Controller exposure:** `controller_message_posted`, `controller_message_directly_exposed`, `sampled_controller_message_ids`, `sampled_controller_report_ids`.
- **New public message:** `new_message`, `new_message_type`, `new_message_id`, `new_message_reply_to`, `new_message_shared_fact_id`.
- **Evidence changes:** `new_controller_fact_ids`, `reactivated_controller_fact_ids`.

This is the appropriate source for microscopic currents, controlled transition frequencies, message-level exposure attribution, and any future effective-affinity calculation.

`controller_message_id` is entirely null, but the plural/list-valued controller message fields contain retained information. Code should not infer that controller exposure is absent solely from the singular null column.

## 5. Stored estimates and derived products

The archive also contains tables that are outputs of analysis rather than raw observations.

| Table | Shape | Role |
|---|---:|---|
| `primary_estimates.parquet` | 4,418 x 67 | Primary information, response, sensing, and current estimates |
| `information_estimates.parquet` | 1,540 x 61 | Information-estimator subset |
| `support_diagnostics.parquet` | 1,540 x 56 | State/action support checks for estimators |
| `transfer_information.parquet` | 730 x 67 | Transfer-information views |
| `sensing_information.parquet` | 70 x 67 | Sensor-information views |
| `susceptibility.parquet` | 590 x 117 | Whole-cell and state-local response estimates |
| `derived_observables.parquet` | 1,565 x 103 | Quantities constructed from primary estimates |
| `efficiencies.parquet` | 1,040 x 103 | Information-response and unsupported thermodynamic-efficiency views |
| `blackboard_diagnostics.parquet` | 75 x 74 | Per-cell communication and exposure totals |
| `rho_b_summary.parquet` | 75 x 58 | Per-cell summaries indexed by persistence and budget |
| `thermodynamic_efficiency_diagnostics.parquet` | 75 x 35 | Why thermodynamic efficiency is unsupported |
| `phi_conditioning_comparison.parquet` | 70 x 53 | Comparison with additional proof-state conditioning |

### Principal estimated quantities

#### Transfer information `T_pi`

The principal action-to-next-state quantity is

```text
round_target_actuation_cmi
```

which represents conditional mutual information

```text
T_pi = I(U_k ; target state at k+1 | target state at k).
```

**Conditional mutual information (CMI)** is a number measuring how much knowing the controller action tells us about the next target state after accounting for the current target state.

At whole-cell resolution, the archive contains:

- the raw direct-count estimate;
- a 95% whole-episode bootstrap interval;
- a policy-conditional action-randomization null mean and standard deviation;
- a permutation p-value;
- observation and episode counts;
- support diagnostics.

The bootstrap resamples complete episodes, not isolated rounds. This preserves within-episode dependence.

At state-bin resolution, the archive contains point estimates and support counts, but it does **not** contain state-local bootstrap intervals or state-local permutation-null estimates. Consequently, raw local `T_pi` maps are supported, but local `T_pi - T_null` maps are not recoverable from this archive alone.

#### Susceptibility `chi`

The source metric is:

```text
round_target_susceptibility
```

It measures the action-conditioned difference in next-round target fraction at matched current target state. It is a behavioral response estimate, not a raw column.

The archive contains state-local point estimates and whole-cell occupancy-weighted estimates. Whole-cell rows have whole-episode bootstrap intervals. Sparse state slices can be marked `limited` or `unsupported`.

#### Information fraction `eta_IF`

The source metric is:

```text
round_target_information_fraction
```

It compares realized target transfer information with the action-information ceiling available under the observed controller policy. It is calculated from estimator outputs rather than directly measured in a game row.

#### Information-response efficiency `eta_IR`

The state-local source metric is:

```text
eta_ir_state_local
```

The whole-cell source metric is:

```text
eta_ir
```

`eta_IR` combines matched transfer-information, susceptibility, and action-frequency estimates. It measures how much of a rigorous information-response bound is expressed as mean target motion. It is not an energetic efficiency.

Whole-cell `eta_IR` has retained whole-episode bootstrap intervals. State-local `eta_IR` has point estimates but no state-local confidence intervals in this archive.

#### Thermodynamic efficiency `eta_th`

`eta_th` is not available. The adaptive blackboard actuator lacks a supported effective-affinity calibration `h`. Related columns such as `effective_affinity`, `affinity_weighted_current_nats`, and `thermodynamic_control_expenditure_nats` are null or explicitly unsupported. A later analysis must not silently reuse a direct-actuation calibration.

## 6. State-local realization

The state coordinate is current controller-target fraction:

```text
x = target_count_before / N.
```

The archived local estimator uses eight bins:

| Bin index | Range | Center |
|---:|---:|---:|
| 0 | [0.000, 0.125) | 0.0625 |
| 1 | [0.125, 0.250) | 0.1875 |
| 2 | [0.250, 0.375) | 0.3125 |
| 3 | [0.375, 0.500) | 0.4375 |
| 4 | [0.500, 0.625) | 0.5625 |
| 5 | [0.625, 0.750) | 0.6875 |
| 6 | [0.750, 0.875) | 0.8125 |
| 7 | [0.875, 1.000] | 0.9375 |

The primary and derived estimator tables contain 520 rows for each of `T_pi`, `chi`, `eta_IF`, and `eta_IR`, for 2,080 local-metric rows in total. Not every planned `(arm, rho, b, x-bin)` position appears because some scientific cells are incomplete and some states were never visited.

Support and occupancy must always be shown beside local estimates. A large point estimate based on one or two rounds is not equivalent to a large estimate supported by many episodes.

## 7. The broken phase-map export

The bundled files

```text
state_local_phase_maps.parquet
state_resolved_x_b.parquet
state_occupancy_binned.parquet
rho_aggregated_state_local_maps.parquet
rho_aggregated_state_occupancy.parquet
```

were intended as convenient plotting views. Their state-local rows were incorrectly labelled `structural_cell_not_run`, producing gray plots.

The underlying estimator values are not absent. `primary_estimates.parquet` and `derived_observables.parquet` contain finite state-local records. The failure arose in the downstream export layer when expected cell labels did not match the resolved hashed cell identifiers used by the estimator rows.

Therefore:

- use `primary_estimates.parquet` as the source for `T_pi`, `chi`, and `eta_IF`;
- use `derived_observables.parquet` for `eta_IR`;
- use recorded `target_fraction_bin_*` fields exactly;
- use the estimator's `support_status` and `n_observations`;
- do not use the broken `phase_status` as scientific evidence of absence.

## 8. What the rebuilt analysis did

The reconstruction in `results/musr_adaptive_q3_rebuilt_report/` performed only derived analysis:

1. Loaded the source Parquet tables with `pyarrow`.
2. Confirmed the 723/750 incomplete status and retained hierarchy.
3. Recovered target semantics and scientific coordinates from `cells.parquet`.
4. Rebuilt state-local maps directly from primary and derived estimator rows.
5. Preserved unsupported states as blank/hatched rather than converting them to zero.
6. Produced persistence-resolved maps for truth and false control.
7. Produced observation-weighted descriptive maps aggregated over persistence.
8. Produced observation-weighted truth-plus-false control maps, labelled as target-semantic aggregates rather than evidence of symmetry.
9. Recomputed episode endpoints from the first round's `occupation_counts_before` and the final round's `occupation_counts_after`.
10. Summarized adaptive REQUEST, REPORT, and DIRECTIVE choices conditional on `U_k=1`.
11. Calculated same-round target-share changes by communication mode.
12. Compared nominal budget with realized controller posts, exposures, readers, and adoptions.
13. Summarized participant REQUEST, REPORT, reply, evidence-acquisition, and exposure measurements.
14. Used retained whole-cell bootstrap and permutation summaries rather than resampling individual rounds.
15. Added a descriptive comparison with the prior report-only study, clearly noting that prompt version and initialization differ.

The persistence-aggregated maps are weighted means of available local estimates using `n_observations`. They are descriptive summaries, not newly pooled CMI estimators.

## 9. Direct measurements, estimates, and descriptive reconstructions

| Quantity | Status | Main source |
|---|---|---|
| Episode identity, seed, completion, token use | Direct retained record | `episodes.parquet` |
| Votes and target/truth shares before and after a round | Direct retained measurement | `rounds.parquet` |
| Binary controller action and action probability | Direct retained measurement | `rounds.parquet` |
| Chosen REQUEST/REPORT/DIRECTIVE mode | Direct retained measurement | `rounds.parquet` |
| Posts, exposures, readers, requests, replies, adoptions | Direct retained measurement/count | `rounds.parquet` |
| Individual opinion transition and slot exposure | Direct retained measurement | `micro_slots.parquet` |
| Initial/final episode outcomes | Deterministic reconstruction from retained counts | `rounds.parquet` |
| `T_pi` | Statistical estimate | `primary_estimates.parquet` |
| `T_null` and permutation p-value | Randomization estimate at whole-cell resolution | `primary_estimates.parquet` |
| `chi` | State-matched behavioral estimate | `primary_estimates.parquet` / `susceptibility.parquet` |
| `eta_IF` | Derived from information estimates | `primary_estimates.parquet` |
| `eta_IR` | Derived from matched information and response estimates | `derived_observables.parquet` |
| Persistence-aggregated local map | Observation-weighted descriptive reconstruction | rebuilt report tables |
| Truth-plus-false pooled map | Observation-weighted descriptive reconstruction | rebuilt report tables |
| `eta_th` | Unsupported | thermodynamic diagnostics |

## 10. Limits of the present archive

1. **Incomplete grid:** 27 planned episodes are missing. Twenty-five cells are not sealed.
2. **No state-local resampling:** local confidence intervals and local permutation nulls were not retained or run.
3. **Sparse high-target states:** upper `x` bins often contain few observations.
4. **Observational mode comparison:** REQUEST, REPORT, and DIRECTIVE are selected by policy, not randomly assigned. Differences by mode are not causal effects.
5. **Cross-study comparison is not an ablation:** the prior report-only study used prompt version 3 and a different paired-initialization archive; the adaptive study used prompt version 4.
6. **No valid thermodynamic efficiency:** effective affinity is unsupported for this actuator.
7. **Nested data dependence:** rounds from the same episode are dependent. Analyses must bootstrap or split by whole episode.
8. **Identifier reuse:** `episode_id` alone is not globally unique.
9. **List-valued fields:** several columns store vectors or JSON-like lists. They must be parsed without losing answer order or agent identity.
10. **Column presence is not support:** some retained columns are entirely null. Every analysis should check non-null counts and semantic support.

## 11. Recommended preparation for the next analysis

Before extending the scientific interpretation:

1. Fill the 27 missing episode keys and rerun strict aggregation.
2. Preserve the four canonical tables unchanged as the auditable data layer.
3. Build one explicit coordinate table keyed by `cell_id`, with normalized arm labels.
4. Validate composite-key uniqueness before every join.
5. Pre-register whether each new result is a direct summary, a statistical estimator, or a descriptive weighted aggregate.
6. Add state-local whole-episode bootstrap intervals.
7. Add state-local policy-conditional action randomization, retaining `T_null` and p-values by `(arm, rho, b, x-bin)`.
8. Keep occupancy and estimator support adjacent to every local map.
9. For causal communication-mode claims, run a randomized or explicitly matched REQUEST/REPORT/DIRECTIVE experiment rather than relying on policy-selected modes.
10. For a valid adaptive-versus-report-only ablation, use the same prompt, initialization artifacts, seeds, task, and grid.
11. Treat participant REQUEST ON/OFF as a planned experimental axis if its protective role against false control is the target question.
12. Develop a new affinity calibration before attempting `eta_th` for the adaptive blackboard actuator.

## 12. Practical loading example

```python
from pathlib import Path
import pandas as pd

root = Path(
    "results/musr_blackboard_adaptive_communication_q3_deepinfra_analysis_source/tables"
)

cells = pd.read_parquet(root / "cells.parquet")
episodes = pd.read_parquet(root / "episodes.parquet")
rounds = pd.read_parquet(root / "rounds.parquet")
micro_slots = pd.read_parquet(root / "micro_slots.parquet")

# Attach authoritative arm labels to round rows.
coordinates = cells[[
    "cell_id",
    "target_semantics",
    "epistemic_persistence",
    "intervention_budget",
]].copy()
coordinates["target_semantics"] = (
    coordinates["target_semantics"].astype(str).str.lower()
)

rounds_with_arm = rounds.merge(
    coordinates,
    on="cell_id",
    how="left",
    suffixes=("", "_cell"),
)

# Safe episode grouping: never group by episode_id alone.
episode_groups = rounds_with_arm.groupby(["cell_id", "episode_id"])
```

The source ZIP is sufficient for extensive new offline analysis because it retains complete round and microscopic records for every realized episode. The main additions still needed are completion of the planned episodes and state-local uncertainty/null calculations.
