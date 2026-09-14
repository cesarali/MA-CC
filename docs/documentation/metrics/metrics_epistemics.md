# Epistemic metrics and plots for the relational blackboard game

Agent entry point: [Metrics and aggregation master reference](README.md).

This is the detailed epistemic companion to [`metrics.md`](metrics.md). It covers only the current
relational blackboard family:

```yaml
game:
  type: relational_imitation_round_feedback
  options:
    task_family: musr_team_allocation
    social_mode: board

prompt:
  prompt_family: relational_blackboard_ballot
```

**Epistemic** means related to what evidence agents know and can currently use. This document
separates five layers that should not be confused:

1. live metrics recorded during an episode;
2. exact evidence fields retained in round and micro-slot records;
3. statistical estimators calculated after simulation;
4. derived tables and diagnostics;
5. plots made from those tables.

Plots are views of stored numbers. They are not additional measurements.

## Contents

1. [Agent knowledge state](#1-agent-knowledge-state)
2. [Coverage, proof ownership, and reach](#2-coverage-proof-ownership-and-reach)
3. [Evidence movement and persistence](#3-evidence-movement-and-persistence)
4. [Knowledge and truth strata](#4-knowledge-and-truth-strata)
5. [Registered epistemic streaming metrics](#5-registered-epistemic-streaming-metrics)
6. [Retained epistemic fields](#6-retained-epistemic-fields)
7. [Epistemically conditioned estimators](#7-epistemically-conditioned-estimators)
8. [Support diagnostics](#8-support-diagnostics)
9. [State-local and persistence-level analysis](#9-state-local-and-persistence-level-analysis)
10. [Exact symbolic epistemic analysis](#10-exact-symbolic-epistemic-analysis)
11. [Causal response and available susceptibility](#11-causal-response-and-available-susceptibility)
12. [Communication and evidence funnel](#12-communication-and-evidence-funnel)
13. [Epistemic tables](#13-epistemic-tables)
14. [Epistemic plots](#14-epistemic-plots)
15. [Current recipe coverage](#15-current-recipe-coverage)
16. [Interpretation rules](#16-interpretation-rules)
17. [Source map](#17-source-map)

---

## 1. Agent knowledge state

For agent $i$ at time $t$, the game distinguishes three states:

- $X_i(t)$: the agent's current vote, stored as `committed_action`;
- $H_i(t)$: every fact legitimately received so far, stored as `known_fact_ids`;
- $K_i(t)$: facts currently available for reasoning, stored as `active_fact_ids`.

They satisfy:

$$
K_i(t)\subseteq H_i(t).
$$

### Active knowledge

Active knowledge, $K_i(t)$, is evidence the agent can currently use. It can shrink when epistemic
persistence removes facts and grow when new or previously inactive evidence is delivered.

### Historical knowledge

Historical knowledge, $H_i(t)$, is cumulative. It does not shrink. A fact can remain historically
known while no longer being active.

### Why both are needed

Suppose an agent once learned facts A and B, then B became inactive:

```text
historical knowledge H = {A, B}
active knowledge K     = {A}
```

Historical coverage says the information reached the agent at some point. Active coverage says what
the agent can use now. A current-response analysis should normally use active state.

### Board-message caveat

`active_fact_ids` records exact acquisition of source facts. Public message prose can also express
semantic conclusions. The fact inventory does not claim to encode every inference an agent could
make from that prose.

---

## 2. Coverage, proof ownership, and reach

Let $S$ be the configured set of supporting facts required by the task.

### 2.1 Individual proof coverage

Without grouped alternatives, agent $i$'s coverage is:

$$
c_i=\frac{|K_i\cap S|}{|S|}.
$$

If the configured proof set is empty, coverage is defined as 1.

MuSR team-allocation tasks can represent one latent proof requirement with several alternative
concrete facts. Let $G$ be the set of latent evidence groups. A group counts as covered when the
agent holds at least one concrete fact in that group:

$$
c_i=
\frac{
\sum_{g\in G}\mathbf 1[K_i\cap g\neq\varnothing]
}{|G|}.
$$

The denominator is then the number of latent requirements, not the number of concrete fact cards.

### 2.2 Mean supporting-fact coverage: $\kappa$

For population size $N$:

$$
\kappa=\frac{1}{N}\sum_{i=1}^{N}c_i.
$$

The live metric name is `mean_supporting_fact_coverage`. It answers:

> On average, what fraction of the configured proof can one agent actively use?

A high $\kappa$ does not imply that any one agent has the entire proof.

### 2.3 Full-proof agent share: $\phi$

$$
\phi=\frac{1}{N}\sum_{i=1}^{N}\mathbf 1[c_i\geq1].
$$

The live metric name is `full_proof_agent_share`. It answers:

> What fraction of agents individually hold every configured proof requirement?

The associated susceptible fraction is:

$$
s=1-\phi.
$$

Here **susceptible** only means “does not hold the complete configured proof.” It does not prove
that an agent will change its vote.

### 2.4 Supporting-fact reach

For each concrete supporting fact $f$:

$$
r_f=\sum_{i=1}^{N}\mathbf 1[f\in K_i].
$$

`supporting_fact_reach` is the vector of these holder counts, ordered by sorted concrete fact IDs.
Grouped coverage and reach use different bases: coverage can count latent groups, while reach counts
concrete facts.

### 2.5 Mean fact count

For active and historical inventories:

$$
\overline{|K|}=\frac{1}{N}\sum_i|K_i|,
\qquad
\overline{|H|}=\frac{1}{N}\sum_i|H_i|.
$$

The helper function calls this value `mean_known_fact_count` even when asked to inspect active facts.
Round fields remove the ambiguity with names such as `active_mean_fact_count_after`.

---

## 3. Evidence movement and persistence

### 3.1 Exposure

An exposure is one presented fact occurrence. Repeating the same fact counts as another exposure,
even if the receiver's knowledge does not change.

### 3.2 New acquisition

Fact $f$ is newly acquired when:

$$
f\notin H_i(t).
$$

The transition adds it to both inventories:

$$
H_i(t+1)=H_i(t)\cup\{f\},
\qquad
K_i(t+1)=K_i(t)\cup\{f\}.
$$

Source-specific fields include:

- `new_peer_fact_ids` and `new_peer_facts`;
- `new_controller_fact_ids` and `new_controller_facts`.

If one new fact is presented more than once in the same update, every occurrence is an exposure but
only the first can be a new acquisition.

### 3.3 Reactivation

Fact $f$ is reactivated when:

$$
f\in H_i(t),\qquad f\notin K_i(t).
$$

It returns to active knowledge but is not added to historical knowledge again.

Relevant fields include:

- `reactivated_peer_fact_ids`;
- `reactivated_controller_fact_ids`;
- `reactivated_peer_fact_count`;
- `reactivated_controller_fact_count`.

The live metric names use `reactivated_peer_facts` and `reactivated_controller_facts`.

### 3.4 Deactivation and epistemic persistence

Let $\rho$ be `epistemic_persistence`. Each active agent-fact pair survives one forgetting boundary
independently with probability $\rho$:

$$
P\bigl(f\in K_i(t^+)\mid f\in K_i(t^-)\bigr)=\rho.
$$

Historical knowledge is unchanged. Retained fields include:

- `epistemic_persistence`;
- `epistemic_persistence_seed`;
- `persistence_deactivated_pairs`;
- `persistence_deactivated_fact_count`;
- `persistence_deactivated_supporting_fact_count`.

At $\rho=1$, no active fact is forgotten. The implementation does not consume a random draw at
that boundary.

### 3.5 Timing

In the current dawn-board protocol, forgetting occurs before controller communication. The symbolic
pre-action boundary is therefore:

```text
post_forgetting_pre_intervention_delivery
```

Round records separately preserve before-round, after-interaction, and after-persistence summaries
where needed. Do not compare values from different boundaries as if they described the same state.

---

## 4. Knowledge and truth strata

Let proof depth $L$ be the number of concrete proof facts or latent grouped requirements. For each
$k=0,\ldots,L$:

- `knowledge_stratum_counts[k]`: agents covering exactly $k$ requirements;
- `truth_counts_by_stratum[k]`: those agents currently voting for the correct answer;
- `knowledge_share_k{k}`:

$$
P(K=k)=\frac{n_k}{N};
$$

- `truth_share_k{k}`:

$$
P(\text{truth vote}\mid K=k)=\frac{t_k}{n_k}.
$$

When $n_k=0$, `truth_share_k{k}` is `None`, not zero. This distinguishes “nobody occupies this
knowledge stratum” from “the stratum is occupied but nobody votes correctly.”

The round records retain before and after versions:

- `knowledge_stratum_counts_before`, `truth_counts_by_stratum_before`;
- `knowledge_stratum_counts`, `truth_counts_by_stratum`;
- corresponding `knowledge_share_k*` and `truth_share_k*` values.

These are raw observables, not registered live metric objects.

---

## 5. Registered epistemic streaming metrics

The game registers these evidence-related live metrics:

| Metric | Meaning |
| --- | --- |
| `mean_supporting_fact_coverage` | Mean active proof coverage, $\kappa$. |
| `full_proof_agent_share` | Active full-proof share, $\phi$. |
| `peer_fact_exposures` | Peer-source fact presentations in the latest update. |
| `controller_fact_exposures` | Controller-source fact presentations. |
| `new_peer_facts` | Previously unknown facts acquired from peers. |
| `new_controller_facts` | Previously unknown facts acquired from the controller. |
| `reactivated_peer_facts` | Inactive historical facts returned to active use by peers. |
| `reactivated_controller_facts` | Inactive historical facts returned to active use by the controller. |

Communication metrics commonly interpreted beside them are:

- `q_effective`;
- `board_size_before`, `board_size_after`;
- `focal_posted_message`;
- `controller_message_posted`;
- `controller_message_directly_exposed`.

These describe one episode trajectory. They are not CMI, causal, or efficiency estimators.

---

## 6. Retained epistemic fields

The analysis uses richer round records than the registered metric list.

### 6.1 Exact inventories

Important source fields include:

- `initial_active_fact_ids_by_agent`;
- `initial_known_fact_ids_by_agent`;
- `active_fact_ids_by_agent_after`;
- `known_fact_ids_by_agent_after`;
- `persistence_deactivated_pairs`;
- `supporting_fact_ids`.

These permit reconstruction of active inventories at the symbolic pre-action boundary.

### 6.2 Active summaries

- `active_mean_supporting_fact_coverage_before`;
- `active_mean_supporting_fact_coverage_after_interactions`;
- `active_mean_supporting_fact_coverage_after`;
- `active_full_proof_agent_share_before`;
- `active_full_proof_agent_share_after_interactions`;
- `active_full_proof_agent_share_after`;
- `active_supporting_fact_reach_before`;
- `active_supporting_fact_reach_after_interactions`;
- `active_supporting_fact_reach_after`;
- `active_mean_fact_count_before`;
- `active_mean_fact_count_after_interactions`;
- `active_mean_fact_count_after`.

Compatibility aliases such as `mean_supporting_fact_coverage` and `full_proof_agent_share` refer to
active knowledge in current records.

### 6.3 Historical summaries

- `historical_mean_supporting_fact_coverage_before` and `_after`;
- `historical_full_proof_agent_share_before` and `_after`;
- `supporting_fact_reach_before` and `supporting_fact_reach`;
- `mean_known_fact_count`.

The unsuffixed reach and mean-known-count fields are historical.

### 6.4 Evidence-flow fields

- `peer_fact_exposures`, `controller_fact_exposures`;
- `new_peer_facts`, `new_controller_facts`;
- `new_evidence_acquisitions`;
- `reactivated_peer_fact_count`, `reactivated_controller_fact_count`.

Here:

$$
\text{new evidence acquisitions}
=
\text{new peer facts}+\text{new controller facts}.
$$

The `results_only` and `dashboard_semantic` profiles retain rich round JSONL and compact micro-slot
JSONL records in addition to `scientific_events.parquet`. The generic Parquet table alone is not the
complete epistemic analysis input.

---

## 7. Epistemically conditioned estimators

### 7.1 Conditional mutual information

**CMI** means conditional mutual information: predictive dependence remaining after current-state
information is held fixed.

The target-channel family has the form:

$$
I(U_t;n_{Z,t+1}\mid n_{Z,t},E_t),
$$

where:

- $U_t$ is controller activation;
- $n_{Z,t}$ is the number of votes for the analysis target;
- $E_t$ is an optional epistemic state.

| Estimator | Additional pre-action state $E_t$ |
| --- | --- |
| `round_memory_target_actuation_cmi` | Exact histogram `knowledge_stratum_counts_before`. |
| `round_epistemic_target_actuation_cmi` | Joint binned $(\kappa,\phi)$ state. |
| `round_phi_target_actuation_cmi` | Coarse $\phi$ state. |
| `round_susceptible_target_actuation_cmi` | Coarse $1-\phi$ state. |
| `round_kappa_target_actuation_cmi` | Coarse $\kappa$ state. |

The exact memory state is:

$$
E_t=(n_t^{(0)},n_t^{(1)},\ldots,n_t^{(L)}).
$$

The scalar $\phi$, $1-\phi$, and $\kappa$ conditionings use three bins:

- low: $[0,1/3)$;
- medium: $[1/3,2/3)$;
- high: $[2/3,1]$.

The joint $(\kappa,\phi)$ diagnostic uses four bins per axis by default.

Actuation CMI uses a policy-conditional null. For each event, a replacement action is drawn from its
recorded controller-action probability. Ordinary uncertainty intervals resample complete episodes,
not individual rounds.

CMI is predictive dependence. It is not automatically a causal effect.

### 7.2 Epistemically conditioned signed response

For conditioning state $z$:

$$
\Delta(z)=
E[\Delta p_Z\mid U=1,z]
-
E[\Delta p_Z\mid U=0,z].
$$

Only states observed with both actions contribute. The final response weights each contributing
state by its event count:

$$
\widehat\Delta=
\frac{
\sum_{z\in D}(n_{1,z}+n_{0,z})
(\overline{\Delta p}_{1,z}-\overline{\Delta p}_{0,z})
}{
\sum_{z\in D}(n_{1,z}+n_{0,z})
}.
$$

Implemented names are:

- `round_memory_target_signed_response`;
- `round_epistemic_target_signed_response`;
- `round_phi_target_signed_response`;
- `round_susceptible_target_signed_response`;
- `round_kappa_target_signed_response`.

These use target-fraction change. They are distinct from
`round_target_signed_actuation`, which uses aligned-order units. For $K$ answer options:

$$
\Delta m=\frac{K}{K-1}\Delta p.
$$

The estimator engine supports all five conditioned responses. Current recipes under
`configs/runs/relational_reasoning/blackboard_game/` do not request them, but some current
`relational_imitation_round_feedback` population-study recipes outside that folder do.

---

## 8. Support diagnostics

Adding epistemic conditioning creates more states and therefore increases the risk of sparse data.
Each estimate must be read with its own support diagnostics.

For visited conditioning states:

$$
\text{dual-state fraction}
=
\frac{\#\text{states observed with both actions}}
{\#\text{visited states}},
$$

$$
\text{dual-event fraction}
=
\frac{\#\text{events in dual-action states}}
{\#\text{controlled events}},
$$

$$
\text{single-action slice fraction}=1-\text{dual-state fraction},
$$

$$
\text{singleton fraction}
=
\frac{\#\text{events in states observed once}}
{\#\text{controlled events}}.
$$

Related metrics are:

- `round_dual_action_state_fraction`;
- `round_dual_action_event_fraction`;
- `round_single_action_slice_fraction`;
- `round_conditioning_state_count`;
- `round_singleton_fraction`.

For ordinary CMI, conditioned CMI, and derived study aggregation, support is `unsupported` when
fewer than two actions are observed or the dual-action-state fraction is zero; `limited` when the
dual-action-state fraction is below 0.25 or the singleton fraction exceeds 0.5; otherwise it is
`adequate`.

Phase-two causal and symbolic state-local outputs use a different count rule: `unsupported` when
either action or silence has zero observations, `limited` when either has exactly one observation,
and `adequate` when both have at least two. These are implementation rules, not universal
statistical laws.

---

## 9. State-local and persistence-level analysis

### 9.1 State-local coordinates

Supported resolutions are:

| Resolution | Coordinates |
| --- | --- |
| `x` | Current controller-target share. |
| `x_phi` | Exact target count plus coarse $\phi$ bin. |
| `x_kappa` | Exact target count plus coarse $\kappa$ bin. |

For $B$ target-share bins:

$$
j=\min(\lfloor Bx\rfloor,B-1),
\qquad
x_{\text{center}}=\frac{j+1/2}{B}.
$$

Current blackboard recipes request `state_local: [x]` with eight bins. `x_phi` and `x_kappa` are
implemented but are not currently selected by these recipes.

Within each slice, analysis can calculate target actuation CMI, target information fraction, signed
target actuation, and target susceptibility.

### 9.2 Occupancy

`state_occupancy.parquet` counts observations and episodes by physical cell and exact pre-action
target count. `state_occupancy_binned.parquet` represents every expected cell and target-share bin.
Its status distinguishes:

- `structural_cell_not_run`;
- `state_not_visited`;
- `visited`.

The phase-map status further distinguishes adequate, limited, and insufficient estimator support.
Missing or unsupported states are not converted to zero.

### 9.3 Descriptive persistence aggregation

With `rho_aggregated_descriptive: true`, the package can contain:

- `rho_aggregated_state_local_maps`;
- `rho_aggregated_state_occupancy`;
- `rho_aggregated_descriptive_summary`.

Here $\rho$ is epistemic persistence. These are explicitly descriptive summaries. They do not create
one pooled CMI estimate across heterogeneous cells.

### 9.4 Derived study aggregation

The four current Task003 recipes additionally enable paired causal summaries
and all ten equal-episode symbolic parameter summaries across persistence.
See [`weighted_study_summaries.md`](weighted_study_summaries.md) for estimands,
support rules, bootstrap pairing, and the quantities kept at cell level.

`derived_study_aggregates` first estimates each physical cell, then combines estimates. It supports:

- `round_target_actuation_cmi`;
- `susceptibility_occupancy_weighted`;
- `round_target_information_fraction`;
- `eta_ir`.

Balanced-cell whole-study summaries weight cells equally. State-local descriptive maps can use
observation weights. Aggregate efficiencies are ratios of aggregated components, not averages of
cell efficiency ratios.

Outputs include:

- `study_aggregated_metrics`;
- `state_local_aggregated_metrics`;
- `state_local_reconstruction`;
- `sample_size_stability`.

`state_local_reconstruction` compares occupancy-reweighted local values with a separately estimated
whole-cell result. A discrepancy is a support and binning diagnostic, not necessarily an error.

---

## 10. Exact symbolic epistemic analysis

Enable this family with:

```yaml
blackboard_epistemic_phase_outputs:
  enabled: true
  task_dataset_dir: <path-to-frozen-task-dataset>  # required
```

The dataset must remain resolvable when reaggregating from canonical tables.
The analysis uses an exact finite-world solver for MuSR team-allocation tasks. It makes no language
model calls.

### 10.1 Evidence boundary

The symbolic state is evaluated after forgetting and before intervention delivery. It uses the union
of participant active inventories. It excludes:

- inactive historical facts;
- controller-private evidence;
- blackboard facts not acquired by a participant.

### 10.2 Configured proof versus symbolic proof

The existing $\phi$ asks whether an agent holds every configured supporting requirement. Symbolic
$\phi^*$ asks whether that agent's active evidence uniquely determines the gold answer across all
valid finite task worlds.

These are not assumed to be equal.

### 10.3 Main symbolic values

| Field | Meaning |
| --- | --- |
| `collective_solvable` | Whether the active population union uniquely determines gold. Denoted $G_t$. |
| `symbolic_individual_solvability_share` | Fraction of individually solvable agents. Denoted $\phi_t^*$. |
| `fragmentation_gap` | $G_t-\phi_t^*$: collective availability without individual assembly. |
| `solvable_agent_count` | Number of individually solvable agents. |
| `unsolvable_agent_share` | $1-\phi_t^*$. |
| `active_union_fact_count` | Distinct active facts anywhere in the population. |
| `active_union_fact_fraction` | Union size divided by the frozen true-fact catalog size. |
| `active_fact_occurrence_count` | Number of active agent-fact pairs. Duplicate holders count separately. |
| `active_mean_fact_count` | Mean active inventory size. |
| `active_min_fact_count`, `active_max_fact_count` | Active inventory-size extremes. |
| `active_mean_holder_redundancy` | Mean active holder count over facts present in the union. |
| `active_min_holder_redundancy` | Minimum holder count over facts present in the union. |
| `collective_compatible_world_count` | Valid task worlds compatible with the active union. |
| `collective_gold_probability` | Gold probability among compatible worlds under the frozen prior. |
| `collective_normalized_entropy` | Remaining uncertainty over the three answers, from 0 to 1. |
| `delta_symbolic_individual_solvability_share` | After-round $\phi^*$ minus pre-action $\phi^*$. |
| `delta_collective_solvable` | After-round $G$ minus pre-action $G$. |
| `symbolic_minus_recorded_full_proof_share` | $\phi^*-\phi$, an explicit cross-check of the two proof definitions. |

`collective_solvable = false` means current active evidence is insufficient for a unique symbolic
answer. It does not mean agents cannot guess correctly or acquire sufficient evidence later.

### 10.4 Robustness to forgetting

If fact $f$ currently has $h_f$ active holders, its probability of surviving one independent
forgetting boundary is:

$$
P(f\text{ survives somewhere})=1-(1-\rho)^{h_f}.
$$

Monte Carlo draws retain each union fact with this probability and rerun exact symbolic solvability.
Outputs include configured and reference robustness, Monte Carlo standard errors, confidence limits,
and draw counts.

If the current state is not collectively solvable, robustness is `NaN`, not zero. At $\rho=1$,
robustness is exactly 1 for a currently solvable state.

### 10.5 Symbolic occupancy and parameter summaries

`epistemic_state_occupancy` groups by physical cell, binned target share $x$, binned $\phi^*$, and
explicit collective-solvability state $G$.

`epistemic_parameter_summary` first averages rounds inside each episode, then gives every episode
equal weight inside a cell. Longer episodes therefore do not receive extra weight.

### 10.6 Joint drift

`epistemic_joint_drift` describes movement in $(x,\phi^*)$ for silence, activation, and randomized
contrast:

$$
\Delta x=x_{t+1}-x_t,
\qquad
\Delta\phi^*=\phi^*_{\text{after}}-\phi^*_{\text{before}}.
$$

The randomized contrast uses:

$$
w_t=\frac{U_t}{e_t}-\frac{1-U_t}{1-e_t},
$$

where $e_t$ is the recorded activation probability. Uncertainty resamples complete
shared-initialization blocks.

The three reported branches are:

$$
D_{\mathrm{activation}}=E\!\left[\frac{U_t}{e_t}\Delta\right],
\qquad
D_{\mathrm{silence}}=E\!\left[\frac{1-U_t}{1-e_t}\Delta\right],
\qquad
D_{\mathrm{contrast}}=E[w_t\Delta],
$$

applied separately to $\Delta x$ and $\Delta\phi^*$.

### 10.7 Epistemic modulation

The modulation model is:

$$
w_t\Delta x_t
=
\beta_0+\beta_xx_t+\beta_\phi\phi_t^*+\beta_rr_t+\epsilon_t.
$$

The reported coefficient is $0.1\beta_\phi$: change in randomized response per 0.1 increase in
$\phi^*$.

Here $r_t$ is the round index divided by the maximum round index in that cell, using a denominator
floor of 1.

The code refuses interpretation when the design matrix lacks full rank, its condition number exceeds
$10^8$, or observed $\phi^*$ range is below 0.1. This measures heterogeneity of the randomized
activation effect. It is not the causal effect of knowledge itself.

### 10.8 Capture timing

With capture threshold $\tau$ and required consecutive run length $m$:

- first evidence loss is the first round with $G_t=0$;
- capture is the first start of $m$ consecutive rounds with $x_t\geq\tau$.

Current symbolic recipes use $\tau=0.75$ and $m=3$. Analysis uses the shortest common episode
horizon in each cell and reports censoring explicitly.

---

## 11. Causal response and available susceptibility

For randomized controller activation, the lag-$h$ response is:

$$
\widehat\tau_h=
\frac{1}{M_h}\sum_t
\left[
\frac{U_t}{e_t}-\frac{1-U_t}{1-e_t}
\right](x_{t+h}-x_t).
$$

Requirements include binary action, $0<e_t<1$, valid round ordering, and both action and silence.
Resampling uses complete shared-initialization blocks.

$M_h$ is the number of eligible rows from complete episodes for which the lag-$h$ outcome exists.
Incomplete episodes and rows lacking that lag are excluded. The estimator is the mean of the
retained Horvitz–Thompson contributions, meaning inverse-probability-weighted outcome changes.

Available target mass is $a_t=1-x_t$. For nonsaturated rows:

$$
\chi_{\text{available},t}
=
\frac{w_t(x_{t+1}-x_t)}{1-x_t}.
$$

Rows with $x_t=1$ are saturated and excluded rather than assigned zero. The cell summary uses a
ratio of sums:

$$
\chi_{\text{available,cell}}
=
\frac{\sum_tw_t(x_{t+1}-x_t)}{\sum_t(1-x_t)}.
$$

Inside each eight-bin target-share slice, `available_causal_susceptibility_state_local` is the
arithmetic mean of the finite row-level values

$$
\frac{w_t(x_{t+1}-x_t)}{1-x_t}.
$$

This differs from `available_causal_susceptibility_summary`, which is the whole-cell ratio of summed
causal response to summed available mass.

Outputs include:

- `causal_response_round_inputs`;
- `causal_response_effects`;
- `causal_response_support`;
- `available_causal_susceptibility_state_local`;
- `available_causal_susceptibility_summary`.

---

## 12. Communication and evidence funnel

The blackboard funnel separates delivery from epistemic effect:

```text
controller activation
  -> controller post
  -> message exposure
  -> reader reached
  -> new fact or reactivation
  -> later vote response
```

### 12.1 Relevant retained fields

Controller action and posting:

- `U_k`, `P_U1_given_Y`;
- `chosen_message_mode`;
- `actual_controller_posts`, `controller_posts`, `controller_post_ids`.

Delivery:

- `controller_message_exposures`;
- `controller_unique_readers`;
- `controller_report_exposures`;
- `controller_report_unique_readers`;
- `directive_exposed_focal_updates`;
- `realized_directive_exposure_fraction`.

Evidence effect:

- `controller_report_fact_acquisitions`;
- `controller_report_fact_reactivations`;
- `new_evidence_acquisitions`;
- `directive_attributed_acquisitions`;
- `directive_attributed_refreshes`.

Vote effect:

- `controller_report_off_target_exposures`;
- `controller_report_target_adoptions`;
- `controller_report_target_adoption_rate`.

The adoption-rate denominator is off-target report exposures:

$$
\frac{\text{target adoptions after report exposure}}
{\text{report exposures where the focal vote began off target}}.
$$

### 12.2 Communication costs

Supported public cost measures are:

- actual posts;
- exposures;
- unique readers per round;
- new controller facts;
- reactivated controller facts.

Expected activation cost is:

$$
E\left[\frac{U_tC_t}{e_t}\right].
$$

Communication efficiency is causal response divided by expected activation cost. Response and cost
are both recalculated inside each bootstrap draw before taking their ratio.

Provider token usage is not public communication cost because it includes private prompts and model
responses.

Reader uniqueness is exact within one round only. Compact retention does not establish episode-wide
unique readership.

Realized message mode, posts, exposures, and readers happen after activation. They can be descriptive
outputs but cannot define pre-action causal groups.

---

## 13. Epistemic tables

### Live and retained state

| Table or file | Epistemic role |
| --- | --- |
| `rounds.parquet` | Canonical population-round state and evidence fields. |
| `micro_slots.parquet` | Canonical individual updates and controller evidence effects. |
| `round_trajectory.jsonl` | Rich source round records retained by the simulation. |
| `micro_slot_trajectory.jsonl` | Rich or compact source update records. |

### Conditioned estimators and support

| Table | Contents |
| --- | --- |
| `primary_estimates.parquet` | Epistemically conditioned CMI and response estimates when requested. |
| `information_estimates.parquet` | Information-estimator rows and variants. |
| `support_diagnostics.parquet` | Action overlap, state counts, singleton rate, and support status. |
| `phi_conditioning_comparison.parquet` | Per-cell comparison of target CMI with and without $\phi$ conditioning, including raw, policy-null, null-adjusted, confidence-interval, entropy-ceiling, and dual-action-support fields. It is produced when both estimates exist. |
| `state_local_phase_maps.parquet` | State-local $T_\pi$, susceptibility, and efficiencies with status. |
| `state_occupancy.parquet` | Exact target-state occupancy. |
| `state_occupancy_binned.parquet` | Complete expected cell-by-$x$ support grid. |

### Persistence and study aggregation

- `rho_aggregated_state_local_maps.parquet`;
- `rho_aggregated_state_occupancy.parquet`;
- `rho_aggregated_descriptive_summary.parquet`;
- `study_aggregated_metrics.parquet`;
- `state_local_aggregated_metrics.parquet`;
- `state_local_reconstruction.parquet`;
- `sample_size_stability.parquet`.

### Symbolic epistemic family

- `epistemic_round_timeseries.parquet`;
- `epistemic_parameter_summary.parquet`;
- `epistemic_state_occupancy.parquet`;
- `epistemic_joint_drift.parquet`;
- `epistemic_causal_susceptibility.parquet`;
- `epistemic_modulation.parquet`;
- `epistemic_capture_timing.parquet`;
- `epistemic_capture_summary.parquet`.

### Communication and blackboard summaries

- `blackboard_diagnostics.parquet`;
- `cell_summary.parquet`;
- `communication_funnel.parquet`;
- `communication_efficiency.parquet`;
- `response_cost_frontier.parquet`;
- `communication_mode_descriptive_response.parquet`.

Not every recipe creates every optional table.

---

## 14. Epistemic plots

### 14.1 Coverage and delivery lines

| Plot recipe | What to read |
| --- | --- |
| `active_coverage_by_budget` | Mean currently usable latent-proof coverage against intervention budget. |
| `historical_coverage_by_budget` | Mean cumulative latent-proof coverage against budget. |
| `realized_exposure_by_budget` | Fraction of focal updates actually exposed to a directive against budget. |

A gap between historical and active coverage shows evidence that reached agents but was not currently
available. A gap between exposure and acquisition shows repeated or already-known evidence.

### 14.2 State-local phase maps

Common plots are:

- `chi_x_b_by_rho`;
- `T_pi_x_b_by_rho`;
- `eta_IF_x_b_by_rho`;
- `eta_IR_x_b_by_rho`;
- `occupancy_x_b_by_rho`.

Their usual coordinates are:

- horizontal: intervention budget $b$;
- vertical: target-share bin center $x$;
- facet: epistemic persistence $\rho$;
- series: truthful-target versus false-target semantics.

Always inspect occupancy and support beside a phase map. Structural absence, unvisited state, and
insufficient estimator support are different conditions and must not be read as zero.

### 14.3 Aggregate and stability plots

Supported aggregate recipes can plot:

- whole-cell raw and null-adjusted $T_\pi$ over persistence and budget;
- study-level $T_\pi$, null, excess information, $\chi$, $\eta_{IF}$, and $\eta_{IR}$ by budget;
- persistence-aggregated local maps;
- observation counts, contributing persistence values, and dual-action support;
- confidence-interval width, estimate spread, sign stability, and permutation-detection rate by
  sample size.

These plots distinguish a stable effect from a visually strong estimate supported by too little data.

### 14.4 Causal communication plots

When phase-two outputs are enabled, fixed plots can include:

- `causal_response_by_lag.png`;
- `response_vs_nominal_budget.png`;
- `response_vs_actual_posts.png`;
- `response_vs_exposures.png`;
- `response_vs_new_evidence.png`;
- `response_cost_frontier.png`;
- `available_causal_susceptibility_x_b_by_rho.png`;
- `available_causal_susceptibility_vs_budget.png`;
- `available_causal_susceptibility_cell_summary.png`.

The response-cost frontier is an operational efficiency view, not thermodynamic efficiency.

### 14.5 Symbolic epistemic plots

When `blackboard_epistemic_phase_outputs` is enabled:

| Plot | Meaning |
| --- | --- |
| `epistemic_parameter_maps.png` | Cell-level collective solvability, fragmentation gap, and reference robustness. |
| `epistemic_causal_displacement.png` | Randomized contrast arrows in $(x,\phi^*)$. Silence and activation branches remain available in `epistemic_joint_drift.parquet`. |
| `epistemic_causal_susceptibility.png` | Randomized response over pre-action $(x,\phi^*)$ bins. |
| `epistemic_capture_timing.png` | False capture relative to evidence loss and collective solvability. |
| `epistemic_central_figure.png` | Combined evidence availability, state motion, and response summary. |

These are finite-horizon regime maps. Smooth colors do not establish a mathematical phase transition.

---

## 15. Current recipe coverage

No single current recipe enables every implemented epistemic capability.

### Standard blackboard population recipes

Supported standard recipes commonly request:

- exact-memory, joint-epistemic, $\phi$-, and $\kappa$-conditioned CMI;
- ordinary target susceptibility and signed response;
- support diagnostics;
- $x$-local analysis with eight bins;
- active and historical coverage plots;
- descriptive persistence aggregation;
- blackboard diagnostic tables.

They generally do not request susceptible-conditioned CMI, the five conditioned signed-response
estimators, or `x_phi`/`x_kappa` state-local coordinates.

### Adaptive communication Q3

The adaptive Q3 recipe additionally requests balanced study-level aggregation across persistence and
target semantics, state-local aggregate maps, aggregate nulls, reconstruction checks, and sample-size
stability diagnostics.

### ASTRA false-control studies

Current ASTRA recipes exercise different subsets:

- phase-two randomized causal and communication outputs;
- exact symbolic epistemic outputs;
- or both.

The recipe's `analysis.yaml` is the authority for what a particular study package should contain.
An implemented table or plot is not guaranteed to appear unless the recipe requests its family.

---

## 16. Interpretation rules

1. **$\phi$ is not $\phi^*$.** $\phi$ means complete configured proof coverage. $\phi^*$ means exact
   unique-gold symbolic solvability.
2. **Active is not historical.** Historical evidence may no longer be usable.
3. **Exposure is not acquisition.** A repeated presentation can increase exposure without changing
   either knowledge inventory.
4. **Reactivation is not acquisition.** It changes active knowledge but not historical knowledge.
5. **Grouped coverage is not concrete-fact reach.** They use different denominators.
6. **Susceptible does not mean behaviorally movable.** $1-\phi$ is evidence bookkeeping.
7. **Vote state and evidence state are separate.** A fully informed agent can vote incorrectly, and
   a poorly informed agent can guess correctly.
8. **CMI is not a causal effect.** Use randomized propensity-weighted response for causal claims.
9. **Post-action communication fields cannot define causal strata.** Posts and exposures occur after
   controller activation.
10. **More conditioning needs more data.** Read support diagnostics beside every epistemic CMI.
11. **$\phi$ and $1-\phi$ can give identical CMI.** They can induce the same state partition.
12. **Local and whole-cell estimates answer different questions.** Reconstruction differences are
   diagnostics, not automatically errors.
13. **Persistence-averaged maps are descriptive unless a declared derived aggregation combines
   already-estimated cells.**
14. **Symbolic maps are finite-horizon summaries.** Smooth heatmaps alone do not prove a phase
   transition.
15. **Unique readers are per-round.** Compact records do not establish episode-wide unique reach.
16. **Provider tokens are not public communication cost.** They include private prompt and response
   material.
17. **Aggregation is offline.** Existing retained records are enough; no provider calls are made.

---

## 17. Source map

| Concern | Source of truth |
| --- | --- |
| Agent active and historical knowledge | `src/mas_cc/games/relational_reasoning/imitation_round_feedback/state.py` |
| Coverage, reach, and knowledge strata | `src/mas_cc/games/relational_reasoning/imitation_round_feedback/metrics.py` |
| Acquisition and reactivation transitions | `src/mas_cc/games/relational_reasoning/imitation_round_feedback/game.py` |
| Persistence and round-record construction | `src/mas_cc/games/relational_reasoning/imitation_round_feedback/runtime.py` |
| Conditioned information and signed-response engine | `src/mas_cc/games/hidden_bench/imitation_round_feedback/analysis.py` |
| Relational record adapter and run analysis | `src/mas_cc/games/relational_reasoning/imitation_round_feedback/analysis.py` |
| Causal response and communication funnel | `src/mas_cc/analysis/causal_response.py` |
| Exact symbolic epistemic analysis | `src/mas_cc/analysis/epistemic_phase.py` |
| Derived study aggregation | `src/mas_cc/studies/derived_aggregation.py` |
| State-local, occupancy, table, and plot orchestration | `src/mas_cc/studies/aggregation.py` |
| Storage retention | `src/mas_cc/observability/recorder.py`, `src/mas_cc/config/models.py` |

For the complete blackboard metric catalogue, see [`metrics.md`](metrics.md). For validation,
canonical-table, packaging, and reaggregation rules, see
[`study_aggregation_contract.md`](study_aggregation_contract.md).

## Blackboard microscopic calibration

See [Offline blackboard calibration](blackboard_calibration.md) for the versioned
source-field audit, recipe, support and held-out reference-prediction contract.
Exposure is post-treatment: exposed/unexposed response rates are descriptive
calibrations, not causal effects of reading. Model preference is not observed
target share; compliance is not the observed switch fraction. Silent rounds may
still expose old controller posts. Missing exposure stays unknown, and available
model susceptibility is undefined at full target saturation.
