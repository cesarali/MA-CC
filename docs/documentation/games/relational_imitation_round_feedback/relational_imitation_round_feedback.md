# Relational imitation with round feedback

This document describes the `relational_imitation_round_feedback` game and the
measurement plan used by
`configs/runs/relational_reasoning/population_study_04/`. It covers the frozen
task data, episode generation, runtime implementation, all 30 configured
analysis statistics, matched classical theory, current estimators, and output
artifacts.

The authoritative implementations are:

- [`game.py`](../../../../src/mas_cc/games/relational_reasoning/imitation_round_feedback/game.py),
  [`runtime.py`](../../../../src/mas_cc/games/relational_reasoning/imitation_round_feedback/runtime.py),
  and [`controller.py`](../../../../src/mas_cc/games/relational_reasoning/imitation_round_feedback/controller.py)
  for the game and controller;
- [`metrics.py`](../../../../src/mas_cc/games/relational_reasoning/imitation_round_feedback/metrics.py)
  for online vote and knowledge observables;
- [`analysis.py`](../../../../src/mas_cc/games/relational_reasoning/imitation_round_feedback/analysis.py)
  for relational record adaptation and post-processing;
- the shared round-feedback
  [`analysis.py`](../../../../src/mas_cc/games/hidden_bench/imitation_round_feedback/analysis.py)
  and generic [`estimators.py`](../../../../src/mas_cc/analysis/estimators.py)
  for direct-counting MI/CMI, bootstrap, null, and overlap diagnostics;
- [`theory.py`](../../../../src/mas_cc/games/relational_reasoning/imitation_round_feedback/theory.py)
  and [`current.py`](../../../../src/mas_cc/games/relational_reasoning/imitation_round_feedback/current.py)
  for the matched finite-`N` q-voter reference and finite-horizon current
  analysis.

## 1. Basic explanation

The game asks whether a population of LLM agents can solve a spatial reasoning
problem when the facts needed for the answer are distributed among them. Each
agent has two separate state variables:

```text
X_i(t) = the relation currently voted for
K_i(t) = the exact set of fact IDs currently known by the agent
```

Agents repeatedly see a small social sample and update one at a time. A ballot
contains a vote, a private recorded reason, and at most one explicitly shared
fact. The reason is never shown to another agent. Consequently,
`shared_fact_id` is the only inter-agent evidence channel and changes to
`K_i(t)` are exactly auditable.

Once per population round, a controller:

1. samples `q_c` agents and reads their votes only;
2. chooses either `NO_OP` or `ADVOCATE_Z`;
3. if advocating, replaces one ordinary social slot at exactly `b` of the
   round's `N` microscopic updates with a recommendation for target `Z`.

The controller cannot inspect knowledge sets. Study 04 uses
`message_mode: recommendation_only`, so it also supplies no facts. It can move
votes through recommendation, but cannot directly increase agent knowledge.

The main scientific question is whether and how information about the
controller's action appears in the next population state as sensing `q_c` and
actuation budget `b` vary.

## 2. Study 04 design

Study 04 is one scientific 3 x 3 sensing/actuation grid split into three YAML
files for operational convenience:

| Config | Fixed sensor size `q_c` | Swept budget `b` | Tasks | Repetitions | Episodes |
|---|---:|---|---|---:|---:|
| `relational_population_study04_qc06.yaml` | 6 | 6, 12, 18 | 0001, 0002 | 30/cell | 180 |
| `relational_population_study04_qc12.yaml` | 12 | 6, 12, 18 | 0001, 0002 | 30/cell | 180 |
| `relational_population_study04_qc18.yaml` | 18 | 6, 12, 18 | 0001, 0002 | 30/cell | 180 |
| **Whole study** | **6, 12, 18** | **6, 12, 18** | **2** | **30/cell** | **540** |

Every episode has ten rounds and every round has 24 microscopic updates. This
produces 5,400 round records across the study. With `local_vote`
initialization, every episode uses 24 initial provider calls and 240 update
calls, or 264 nominal calls. The full study therefore has 142,560 nominal
provider calls.

Parameters held fixed are:

| Parameter | Study 04 value | Meaning |
|---|---:|---|
| `N` | 24 | population size and microscopic updates per round |
| `L` | 2 | supporting-fact chain length |
| `r` | 6 | initial holders of each supporting fact |
| `K` | 3 | answer relations |
| `q` | 1 | ordinary social slots per focal update |
| rounds | 10 | controller decisions and population rounds per episode |
| `theta` | 0.5 | policy target-share threshold |
| `beta` | 4.0 | logistic policy steepness |
| target | option index 2 | fixed incorrect semantic relation in each task |
| schedule | `soft` | stochastic closed-loop advocacy |
| message | `recommendation_only` | controller transmits a vote and no fact |
| initialization | `local_vote` | one initial LLM decision from private facts |
| stop on consensus | false | all episodes retain ten round observations |

The target is `EAST` for task 0001 and `SOUTH` for task 0002. The correct
answers are respectively `WEST` and `SOUTHWEST`. Option index 2 is not a
general property of the dataset; it is verified only for the selected tasks.

### Paired randomization

All three YAML files use execution seed `20260821` and preserve the same grid
axis order. Thus a matching `(b, task, repetition)` across the `q_c` files has
the same episode seed and initial population. This is a common-random-numbers,
paired comparison across the sensing axis.

The following streams are derived independently inside an episode:

- focal-agent and peer selection;
- controller sensing and soft-policy sampling;
- controller slot replacement;
- controlled-position schedules, derived separately for each round;
- per-call semantic option shuffling.

Sensed agents are identities, whereas controlled positions are positions in
the round's update schedule. They are separate draws and must not be equated.

## 3. How the task data is generated

The game does not invent a problem during a run. It reads frozen JSON from
`src/mas_cc/relational_task_generator/relational_task_generator/datasets/pop24_L2_r06/`.
The dataset manifest records:

```text
dataset seed             20260818
tasks                    10
population size N        24
reasoning depth L        2
support redundancy r     6
distractors              4
distractor redundancy    1
answer options K         3
no single-agent solution true
```

Study 04 uses only `task_0001.json` and `task_0002.json` from this ten-task
dataset.

The standalone, standard-library-only generator constructs each task as
follows:

1. Build a self-avoiding chain of `L` exact two-dimensional spatial
   relations. Integer coordinates are the symbolic ground truth.
2. Derive the query answer from the net displacement of the chain endpoints.
3. Generate four internally consistent distractor facts in a disconnected
   component, so they cannot alter or shorten the proof.
4. Choose `K` distinct compass relations with exactly one correct answer.
5. Assign every supporting fact to exactly `r` distinct agents and every
   distractor to its configured number of agents. With
   `no_single_agent_solution`, no agent initially receives the complete proof,
   while the population union does.
6. Render symbolic facts into canonical natural language without an LLM.
7. Validate world consistency, reasoning depth, correct answer, option
   uniqueness, allocation constraints, rendering, and references.
8. Write each canonical task, per-task SHA-256 fingerprints, and a full-dataset
   fingerprint to `manifest.json`.

Task seeds are deterministic children of `(dataset_seed, task_index)`.
Regeneration with the stored parameters is checked by the dataset validator.
The dataset seed governs task construction; the separate execution seed
governs episode sampling and LLM interaction.

To reproduce the dataset shape from the generator directory:

```bash
python generate_dataset.py \
  --num-tasks 10 \
  --population-size 24 \
  --reasoning-depth 2 \
  --support-redundancy 6 \
  --distractors 4 \
  --distractor-redundancy 1 \
  --num-options 3 \
  --seed 20260818 \
  --no-single-agent-solution \
  --output datasets/pop24_L2_r06

python validate_dataset.py datasets/pop24_L2_r06
```

Do not overwrite the checked-in dataset merely to run the study; the configs
consume its frozen files directly.

## 4. How an episode is generated

### Initialization

The loader validates the task schema and population size, then initializes
each `K_i(0)` exactly from the task's `agents` mapping. Under `local_vote`, all
24 agents independently receive a prompt containing only their initial facts
and make one ballot. The returned display letter is immediately mapped back to
a semantic relation; persistent state never stores `A`, `B`, or `C`.

Options are shuffled per call from a deterministic seed. A peer vote is
re-rendered into the receiving call's current letter mapping. This prevents a
globally stable letter from becoming an artificial social attractor.

### One population round

Let `N_k` be the length-`K` vote-count vector before round `k`, `Y_k` the
sensor count vector, `U_k` the controller action, and `N_{k+1}` the count vector
after the round:

```text
N_k -> Y_k -> U_k -> N microscopic updates -> N_(k+1)
```

The sensor samples `q_c` distinct agents without replacement and sees votes,
not facts. If `y_Z` sampled agents vote for the target, the sensed share is
`p_Z(Y_k) = y_Z/q_c`. The configured soft policy is

```text
P(U_k = ADVOCATE_Z | Y_k)
  = sigmoid(beta * (theta - p_Z(Y_k))).
```

For Study 04, `beta=4` and `theta=0.5`. Low observed target support therefore
makes advocacy more likely, while the policy remains stochastic. The
stochasticity is essential: an `always` schedule makes `H(U_k)=0`, so all
action-information quantities are zero by construction.

On an advocacy round, exactly `b` distinct update positions are sampled
uniformly without replacement. On a `NO_OP` round there are exactly zero
controlled positions.

### One microscopic update

At each of the 24 positions:

1. sample one focal agent and `q=1` distinct peer;
2. if the position is controlled, replace the peer slot with the persistent
   controller participant; otherwise show the sampled peer;
3. render the focal agent's private facts, current vote, and visible social
   source;
4. make one LLM call and validate its ballot;
5. update the focal vote immediately;
6. add any fact explicitly exposed in the visible source to the focal
   knowledge set, recording peer/controller provenance.

The controller is rendered as `Agent N+1` and is not identified as a special
authority. In Study 04 it exposes no fact. An ordinary peer can expose at most
one fact it already knows. Reasons are recorded but never rendered socially or
fed back to their author.

### Persisted data

`trajectory.jsonl` or its compact equivalent records microscopic focal
updates when retained by the artifact profile. The essential analysis input is
`round_records/<episode>/round_trajectory.jsonl`. Each row includes:

- before/after vote counts and order parameters;
- controller action, logged action probability, sensor identities and counts;
- controlled positions, count, seed, and schedule hash;
- `q_c`, `q_c/N`, `b`, `b/N`, `beta`, and `theta`;
- before/after knowledge coverage, full-proof share, and exact knowledge
  stratum histogram;
- peer/controller fact exposures and newly acquired facts;
- task, answer, target, seed, episode, and round provenance.

With `artifact_profile: results_only`, these round records are deliberately
retained because the offline analysis can be rerun without provider calls.

## 5. Implementation map

| Component | Responsibility |
|---|---|
| `relational_reasoning/data.py` | Loads and rejects invalid frozen tasks; maps task JSON to immutable task data. |
| `imitation_round_feedback/state.py` | Stores agent votes, known fact IDs, ballots, provenance, task metadata, and history. |
| `game.py` | Initializes agents; creates initial and social ballot requests; shuffles options; validates citations; applies vote and knowledge transitions. |
| `prompts.py` | Renders private knowledge, current vote, public vote, and explicitly shared evidence; keeps reasons out of the social channel. |
| `controller.py` | Extends the shared soft target controller with relational message modes and deterministic controller-fact selection. |
| `runtime.py` | Runs the two clocks, derives RNG streams, senses, selects the round action, schedules exact-budget control, executes calls, and emits micro/round records. |
| `metrics.py` | Reuses vote observables and adds exact knowledge observables and streaming metric adapters. |
| relational `analysis.py` | Converts relational round rows to shared `RoundEvent`s; adds epistemic conditioning; runs per-cell and pooled analysis; writes theory comparisons. |
| shared round `analysis.py` | Defines all 30 statistic names, direct-counting estimates, whole-episode bootstrap, policy/sensor nulls, entropy bounds, and support diagnostics. |
| `theory.py` | Computes the deterministic matched finite-`N` q-voter kernels and transfer entropy. |
| `current.py` | Computes repeated-episode net target current and exact finite-horizon theory moments. |

## 6. Core recorded metrics

Let `p_j=n_j/N` be the share voting for option `j`, and let `K` be the number
of options. Vote order parameters are:

```text
m_truth = (K p_truth - 1)/(K - 1)
m_ctrl  = (K p_target - 1)/(K - 1)
m_order = (K max_j p_j - 1)/(K - 1)
H_vote  = -sum_j p_j ln(p_j) / ln(K)
```

The aligned magnetizations equal 0 at a uniform ballot and 1 when the named
option has full consensus. They can be negative when the named option has less
than the uniform share. `m_order` measures concentration without caring which
option leads. `H_vote` is normalized to `[0,1]`, with 0 at consensus and 1 at
the uniform distribution. `delta_*` fields are after minus before for one
micro update or one round, depending on the record.

For supporting-fact set `S` and agent knowledge `K_i`:

```text
coverage_i = |K_i intersect S| / |S|
kappa      = mean_i coverage_i
phi        = count_i[K_i contains S] / N
susceptible = 1 - phi
E_k[j]     = number of agents knowing exactly j supporting facts
```

Other knowledge fields are `supporting_fact_reach` (holder count for each
supporting fact), `mean_known_fact_count`, exposure counts, and counts of facts
that were newly acquired rather than already known.

The registered streaming metrics are:

| Family | Metrics |
|---|---|
| Votes/state | per-option action share, `agent_current_action`, `dominant_action_share`, `truth_vote_share` |
| Order | `m_truth`, `m_ctrl`, `m_order`, `normalized_vote_entropy` |
| Change/event | `delta_m_truth`, `delta_m_ctrl`, `delta_m_order`, `focal_changed`, `focal_adopted_target` |
| Knowledge | `mean_supporting_fact_coverage`, `full_proof_agent_share` |
| Transmission | `peer_fact_exposures`, `controller_fact_exposures`, `new_peer_facts`, `new_controller_facts` |

## 7. Statistical estimator conventions

The analysis treats complete discrete states as categorical values. It does
not impose a continuous model or fit a regression.

For discrete variables `X`, `Y`, and `Z`, the reported definitions are

```text
I(X;Y)   = sum_(x,y) p(x,y) log2[p(x,y)/(p(x)p(y))]

I(X;Y|Z) = sum_(x,y,z) p(x,y,z)
             log2[p(x,y|z)/(p(x|z)p(y|z))].
```

Observed contingency-table frequencies supply the probabilities. Every MI/CMI
row contains three variants:

| Variant | Calculation | Role in Study 04 |
|---|---|---|
| `unsmoothed` | plug-in entropy from nonzero observed cells | **Primary estimate** (`main_estimator_variant`) |
| `jeffreys` | add 0.5 to every cell before computing entropy | sensitivity column |
| `miller_madow` | add `(r-1)/(2 n ln 2)` to each empirical entropy, where `r` is its occupied-cell count | finite-sample bias-correction column |

Non-information diagnostics use their direct formula in the `unsmoothed`
column; their other variant columns are `NaN`.

### Uncertainty and nulls

- The 95% interval is a percentile bootstrap with 1,000 resamples. Whole
  episode IDs are sampled with replacement, preserving dependence among the
  ten rounds of one episode.
- Each actuation CMI gets 1,000 policy-conditional randomizations. For every
  row, a new action is drawn from that row's logged advocacy probability
  `p_k`; population outcomes and conditioning states stay fixed.
- Sensing MI gets 1,000 permutations of sensor count vectors among eligible
  rows.
- Diagnostics that are not MI/CMI do not receive a randomization null.
- Estimates are produced for every cell and for a pooled row. The pooled
  empirical row combines the six cells within a launched YAML. Interpret it
  carefully because cells have different `b` values.

These are observational channel estimates under the logged policy, not a
claim that CMI alone proves a causal effect. Sparse conditioning and action
overlap must be checked alongside every estimate.

## 8. All 30 configured analysis statistics

Notation used below:

```text
N_k       complete K-option count vector before round k
Y_k       q_c-sample count vector
U_k       NO_OP or ADVOCATE_Z
n_Z,k     target-vote count
n_T,k     correct-answer vote count
o_k       max_j n_j,k
Delta x   x_(k+1) - x_k
E_k       exact knowledge-stratum histogram
B4(kappa,phi) joint pair of 4-bin indices on [0,1]
B3(x)     low [0,1/3), medium [1/3,2/3), high [2/3,1]
```

For a signed response matched on state `z`, the common estimator is the
event-count-weighted average over slices containing both actions:

```text
sum_z w_z [mean(Delta | ADVOCATE,z) - mean(Delta | NO_OP,z)] / sum_z w_z,
```

where `w_z` is the number of events in that dual-action slice.

| # | Configured statistic | Definition / estimator | How to read it |
|---:|---|---|---|
| 1 | `round_sensing_mi` | `I(N_k; Y_k)` in bits | Information the sample carries about the full population state. Larger is better sensing, but depends on visited-state diversity. |
| 2 | `round_sensor_mae` | `mean(abs(y_Z/q_c - n_Z/N))` | Mean absolute target-share sensing error; lower is more accurate. |
| 3 | `round_sensor_mse` | `mean((y_Z/q_c - n_Z/N)^2)` | Squared target-share sensing error; penalizes large errors. |
| 4 | `round_controller_action_entropy` | `H(U_k)` in bits | Whether both actions occur. Zero means actuation information is structurally unidentifiable. |
| 5 | `round_controller_action_entropy_given_population` | `H(U_k | N_k)` in bits | Action variation left after the full population state is known; ceiling for the full-state population CMI. |
| 6 | `round_population_actuation_cmi` | `I(U_k; N_(k+1) | N_k)` in bits | Action information in the complete next vote-count vector. |
| 7 | `round_target_actuation_cmi` | `I(U_k; n_Z,k+1 | n_Z,k)` in bits | Primary controller-to-target channel. This is the empirical quantity compared with matched q-voter transfer entropy. |
| 8 | `round_truth_actuation_cmi` | `I(U_k; n_T,k+1 | n_T,k)` in bits | Action information in correct-answer support. Important because Study 04 targets an incorrect answer. |
| 9 | `round_order_actuation_cmi` | `I(U_k; o_(k+1) | o_k)` in bits | Action information in population concentration, regardless of winner identity. |
| 10 | `round_population_information_fraction` | `I(U;N'|N) / H(U|N)` | Fraction of available conditional action entropy expressed in the full next state; undefined if denominator is effectively zero. |
| 11 | `round_target_information_fraction` | `I(U;n_Z'|n_Z) / H(U|n_Z)` | Target-channel information normalized by its action-entropy ceiling; undefined at zero ceiling. |
| 12 | `round_conditioning_state_count` | Number of distinct full `N_k` slices | Size of the conditioning state space used by the general support diagnostic. |
| 13 | `round_dual_action_state_fraction` | fraction of observed `N_k` slices containing both actions | State-level overlap; near zero warns that conditional contrasts are weakly supported. |
| 14 | `round_dual_action_event_fraction` | fraction of action-bearing rows located in dual-action `N_k` slices | Observation-level overlap. |
| 15 | `round_single_action_slice_fraction` | `1 - dual_action_state_fraction` | Fraction of state slices in which only one action occurred. |
| 16 | `round_singleton_fraction` | fraction of action-bearing rows whose `N_k` slice has exactly one row | Direct measure of conditioning sparsity. |
| 17 | `round_target_signed_response_share` | Unmatched `E[Delta(n_Z/N)|ADV] - E[Delta(n_Z/N)|NO_OP]` | Direction of raw target-share movement; positive favors the controller target. Unlike the next statistic, it does not state-match. |
| 18 | `round_target_signed_actuation` | Signed response of `Delta m_ctrl`, matched on `n_Z,k` | Directional target effect in aligned-magnetization units. Since `Delta m_ctrl = K/(K-1) Delta(n_Z/N)`, it rescales target-share change. |
| 19 | `round_truth_signed_actuation` | Signed response of `Delta m_truth`, matched on `n_T,k` | Directional effect on truth. It may oppose target response because the target is wrong. |
| 20 | `round_order_signed_actuation` | Signed response of `Delta m_order`, matched on `o_k` | Whether advocacy increases or decreases consensus/concentration. |
| 21 | `round_memory_target_actuation_cmi` | `I(U;n_Z' | n_Z,E_k)` | High-dimensional exact epistemic-memory-conditioned target channel. Scientifically preferred when supported, often sparse. |
| 22 | `round_epistemic_target_actuation_cmi` | `I(U;n_Z' | n_Z,B4(kappa,phi))` | Coarse joint knowledge-state diagnostic, four bins per axis. |
| 23 | `round_phi_target_actuation_cmi` | `I(U;n_Z' | n_Z,B3(phi))` | Target channel after separately conditioning on full-proof share. |
| 24 | `round_susceptible_target_actuation_cmi` | `I(U;n_Z' | n_Z,B3(1-phi))` | Same question expressed through the fraction still lacking the full proof. It normally equals the phi estimate because the bins induce a relabelled partition. |
| 25 | `round_kappa_target_actuation_cmi` | `I(U;n_Z' | n_Z,B3(kappa))` | Target channel after conditioning on mean proof coverage. |
| 26 | `round_memory_target_signed_response` | Target-share signed response matched on `(n_Z,E_k)` | Direction corresponding to statistic 21. |
| 27 | `round_epistemic_target_signed_response` | Target-share signed response matched on `(n_Z,B4(kappa,phi))` | Direction corresponding to statistic 22. |
| 28 | `round_phi_target_signed_response` | Target-share signed response matched on `(n_Z,B3(phi))` | Direction corresponding to statistic 23. |
| 29 | `round_susceptible_target_signed_response` | Target-share signed response matched on `(n_Z,B3(1-phi))` | Direction corresponding to statistic 24; normally identical to the phi partition result. |
| 30 | `round_kappa_target_signed_response` | Target-share signed response matched on `(n_Z,B3(kappa))` | Direction corresponding to statistic 25. |

The standalone support-statistic rows (12-16) use the full population vector
`N_k`. In addition, every estimate row carries support columns. For the five
epistemic CMI/signed-response families, those columns are recalculated using
that statistic's own widened conditioning state.

## 9. Support and overlap diagnostics

Read these fields before interpreting a small CMI:

| Output field | Definition |
|---|---|
| `n_episodes`, `n_rounds` | eligible sample size and its episode clustering |
| `unique_population_states` | number of distinct full vote-count vectors |
| `unique_sensor_states` | number of distinct observed sensor vectors |
| `number_of_actions_observed` | one or two controller actions actually present |
| `min/median/max_rounds_per_population_state` | occupancy of full population-state slices |
| `round_conditioning_state_count` | number of slices under the relevant conditioning |
| `round_dual_action_state_fraction` | fraction of slices with both actions |
| `round_dual_action_event_fraction` | fraction of rows in those overlapping slices |
| `round_single_action_slice_fraction` | fraction of slices with only one action |
| `round_singleton_fraction` | fraction of rows in one-observation slices |
| `conditional_action_entropy_bits` | statistic-specific `H(U|Z)`, the CMI ceiling |
| `entropy_bound_satisfied` | numerical check that estimated CMI does not exceed that ceiling |

A zero CMI with no dual-action support is not evidence of no response. It says
the action contrast is unobserved within the chosen conditioning slices. The
exact `E_k` estimator should be retained as the scientific high-dimensional
reference even when sparse; the joint and scalar binnings show whether a
coarser epistemic description is estimable.

## 10. Matched finite-N q-voter analysis

The relational analysis automatically compares each cell with a deterministic
binary q-voter reference at the same `(N,q,q_c,b,beta,theta)`. The relational
population is projected to `target` versus `not target`; this is the only
coarse-graining needed to compare its three-option state with the binary
reference.

The reference computes:

- the exact ordinary microscopic kernel `K0` and controlled kernel `K1`;
- `R0 = K0^N` for a `NO_OP` round;
- an exact dynamic-programming average `R1` over schedules with exactly `b`
  controlled positions, rather than independent Bernoulli control;
- the hypergeometric sensing law `S(y|n)`;
- the sensor-averaged policy
  `a_n = sum_y S(y|n) sigmoid(beta(theta-y/q_c))`;
- state-local transfer entropy `T_qv(n)`, the weighted Jensen-Shannon
  divergence between `R1[n,:]` and `R0[n,:]`;
- empirical-occupancy-weighted classical TE, exact `q=1` signed response, and
  empirical-minus-theory residuals.

Important theory outputs include:

| Output | Meaning |
|---|---|
| `theory_te_emp_occ_bits` | exact local q-voter TE averaged over states visited by the LLM episodes; primary comparator |
| `theory_te_self_occ_bits` | q-voter TE under the reference process's own occupancy; secondary occupancy diagnostic |
| `mean_field_te_bits` | approximation only; flagged when outside its entropy ceiling |
| `delta_te_bits` | empirical target CMI minus empirical-occupancy classical TE |
| `te_ratio` | empirical/classical diagnostic ratio; explicitly not an efficiency |
| `delta_mu_empirical`, `delta_mu_theory` | empirical and exact classical signed target response |
| `policy_mae`, `policy_rmse` | statewise empirical advocacy frequency versus exact sensor-averaged policy |
| `delta_te_*_ci` | whole-episode bootstrap interval for the empirical-minus-theory difference; the deterministic theory curve itself is not bootstrapped |
| `theory_interpretation` | classifier distinguishing degeneracy, policy mismatch, comparable channels, kernel departure, or occupancy departure |

Each config pools cells with three different `b` values. Therefore its pooled
theory row correctly has `theory_applicable=false`: no single matched parameter
tuple describes that pool. Use the six per-cell theory rows. Pooled empirical
MI/CMI values are still calculated.

## 11. Finite-horizon target current estimators

Current analysis is also produced automatically, although its names are not
members of the YAML's 30-entry `analysis.estimators` list. For an episode of
`R` completed rounds,

```text
J = n_Z,R - n_Z,0.
```

Microscopic target increments are summed when microscopic records are present
and checked against this terminal difference. Across the 30 repeated episodes
of a task/cell, the empirical estimators are:

| Metric | Formula | Meaning |
|---|---|---|
| `current_mean` | `mean(J)` | average net agents gained by the target |
| `current_variance` | sample variance `Var(J)` with `ddof=1` | between-episode fluctuation |
| `current_fano_dispersion` | `Var(J)/abs(mean(J))` | Fano-like dispersion |
| `current_precision_irisarri` | `abs(mean(J))/Var(J)` | inverse-Fano/Irisarri-style precision |
| `current_snr2` | `mean(J)^2/Var(J)` | squared signal-to-noise ratio |

Zero mean and zero variance are left explicitly undefined or infinite as
appropriate and accompanied by degeneracy flags; no smoothing is introduced.
Whole-episode bootstrap intervals use the same 1,000-resample setting. Exact
finite-horizon q-voter moments are computed from the matched closed-loop kernel
and the empirical distribution of initial target counts, then reported beside
empirical-minus-theory differences.

## 12. Output files and how to read them

The default offline command is:

```bash
conda run -n MA-CC --no-capture-output python -m mas_cc.cli.main \
  analysis relational-round-feedback \
  --run-dir results/relational_imitation_round_feedback/relational-study04-qc06-resource-grid/<run-id>
```

Repeat for `qc12` and `qc18`. The analysis is provider-free.

| File | Contents |
|---|---|
| `round_information_estimates.csv` | all configured estimates, estimator variants, bootstrap intervals, null summaries, entropy ceilings, and support fields, per cell plus pooled |
| `round_information_estimates.md` | compact estimate table followed by matched-theory interpretation |
| `round_information_nulls.csv` | one row per MI/CMI null randomization |
| `round_support_diagnostics.csv` | overall per-cell and pooled support summaries |
| `controller_action_summary.csv` | action counts, advocacy frequency/probability, and realized controlled positions |
| `episode_epistemic_regime.csv` | per-episode knowledge regime and fact-exposure summaries |
| `round_epistemic_trajectory.csv` | analysis-ready round sequence with target/truth counts, `kappa`, `phi`, `E_k`, and bin labels |
| `theory_comparison.csv` | per-cell empirical/classical comparison and scientific coordinates |
| `theory_state_curves.csv` | state-resolved sensing policy, response, local TE, and occupancy curves |
| `currents/episode_currents.csv` | one net target current per episode and optional microscopic telescoping check |
| `currents/cell_current_summary.csv` | empirical and exact-theory current moments, ratios, differences, and intervals |
| `currents/**/current_analysis.md` | human-readable current report per task/cell |
| `analysis_summary.json` | settings, counts, theory summaries, epistemic support warnings, current summaries, and Comet export status |

A practical reading order is:

1. Confirm `number_of_actions_observed`, action entropy, and dual-action support.
2. Read sensing MI with sensor MAE/MSE to separate information from accuracy.
3. Read target, truth, population, and order CMIs with their entropy ceilings
   and null means.
4. Use signed responses to determine direction.
5. Compare unconditioned, exact-memory, joint-bin, and scalar-bin target CMIs.
6. Compare the target CMI and signed response with the matched classical row.
7. Use current mean/variance/SNR metrics for the finite-horizon repeated-episode
   outcome.

## 13. Running and validation

Preflight validates config structure, all estimator names, cost bounds, and the
grid without making model calls:

```bash
conda run -n MA-CC --no-capture-output python -m mas_cc.cli.main \
  experiment preflight \
  --config configs/runs/relational_reasoning/population_study_04/relational_population_study04_qc06.yaml \
  --output-dir results/inspection/relational_study04_qc06_preflight
```

Launch one row with:

```bash
conda run -n MA-CC --no-capture-output python -m mas_cc.cli.main \
  experiment run \
  --config configs/runs/relational_reasoning/population_study_04/relational_population_study04_qc06.yaml
```

Use the corresponding `qc12` and `qc18` paths for the other rows. Do not
reorder grid axes in only one file, and use `--no-resume` or a clean output
directory after changing a config; checkpoints reject a mismatched resolved
config hash rather than mixing experiments.

Relevant test suites are:

```bash
pytest -q \
  tests/mas_cc/test_relational_imitation_round_feedback.py \
  tests/mas_cc/test_relational_round_feedback_analysis.py \
  tests/mas_cc/test_relational_current_analysis.py \
  tests/mas_cc/test_relational_matched_theory.py
```

## 14. Interpretation cautions

- `q_c` is sensing capacity and `b` is actuation capacity. Equal numerical
  values do not make them the same resource.
- MI measures statistical dependence and is unsigned. Always pair a CMI with
  its signed response and support diagnostics.
- A larger sensing MI need not imply smaller target-share error because MI also
  reflects diversity of visited states.
- A zero information estimate under a deterministic action schedule is
  structural, not evidence that advocacy has no behavioral effect.
- `phi` and `1-phi` induce equivalent conditioning partitions in the usual
  case, so equal phi/susceptible results are expected.
- The pooled empirical row mixes intervention budgets; the pooled matched
  theory row is intentionally inapplicable.
- The q-voter is a matched control-protocol reference, not a claim that LLM
  reasoning follows a q-voter mechanism.
- Study 04 uses an incorrect controller target. Target success and truth
  success are deliberately different outcomes.
