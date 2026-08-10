# HiddenBench imitation game, metrics, and information analysis

This document describes the implemented and executable
`hidden_bench_imitation` game. It is a reference for the game semantics, the
recorded event schema, all currently exposed metrics, and the offline discrete
mutual-information analysis.

The relevant implementation is in:

- `src/mas_cc/games/hidden_bench/imitation/game.py` — state transitions and
  event records;
- `runtime.py` — reasoning/classical execution and provider-call separation;
- `controller.py` — partial-observation threshold controller;
- `classical.py` — provider-free multi-opinion jump kernel;
- `metrics.py` — population observables and streaming metrics;
- `analysis.py` — event adapter, episode/cell summaries, MI/CMI, bootstrap,
  nulls, support diagnostics, and report files.

No InfoNCE estimator is implemented or emitted.

## Table of contents

1. [Scientific purpose](#1-scientific-purpose)
2. [State and event schedule](#2-state-and-event-schedule)
   - [Reasoning dynamics](#21-reasoning-dynamics)
   - [Classical/no-reasoning dynamics](#22-classicalno-reasoning-dynamics)
3. [Control configuration reference](#3-control-configuration-reference)
   - [Supported mechanisms](#31-supported-mechanisms)
   - [`none`: uncontrolled behavior](#32-none-uncontrolled-behavior)
   - [`threshold_target`: measured feedback](#33-threshold_target-measured-feedback)
   - [Every `threshold_target` field](#34-every-threshold_target-field)
   - [Choosing another target](#35-choosing-another-target)
   - [Worked sensor/action examples](#36-worked-sensoraction-examples)
   - [Effect in reasoning and classical modes](#37-effect-in-reasoning-and-classical-modes)
   - [Unsupported and compatibility forms](#38-unsupported-and-compatibility-forms)
4. [Population observables](#4-population-observables)
   - [Per-option counts and shares](#41-per-option-counts-and-shares)
   - [Truth and controller-target projections](#42-truth-and-controller-target-projections)
   - [Population order and vote entropy](#43-population-order-and-vote-entropy)
   - [Other standing metrics](#44-other-standing-metrics)
5. [Event-level behavioral metrics](#5-event-level-behavioral-metrics)
   - [Changes in population observables](#51-changes-in-population-observables)
   - [Focal transition indicators](#52-focal-transition-indicators)
   - [Controller and sensor diagnostics](#53-controller-and-sensor-diagnostics)
6. [Episode-level and pooled summaries](#6-episode-level-and-pooled-summaries)
   - [Population response](#61-population-response)
   - [Controller action statistics](#62-controller-action-statistics)
   - [Sensor quality](#63-sensor-quality)
   - [Action-conditioned response](#64-action-conditioned-behavioral-response)
   - [Target-adoption response](#65-target-adoption-response)
   - [Cell summary convention](#66-cell-summary-convention)
   - [Four-grid contrasts](#67-four-grid-contrasts)
7. [Implemented discrete information analysis](#7-implemented-discrete-information-analysis)
   - [Canonical event adapter](#71-canonical-event-adapter)
   - [Four MI/CMI statistics](#72-four-statistics)
   - [MI and CMI definitions](#73-mi-and-cmi-definitions)
   - [Estimator variants](#74-estimator-variants)
   - [Episode bootstrap](#75-episode-bootstrap)
   - [Temporal nulls](#76-temporal-nulls)
   - [Support diagnostics](#77-support-and-sparsity-diagnostics)
8. [Analysis output files](#8-analysis-output-files)
9. [Ready-to-run configurations](#9-ready-to-run-configurations)
   - [Separate reasoning run](#91-separate-reasoning-run-10-episodes)
   - [Separate classical run](#92-separate-classicalno-reasoning-run-10-episodes)
   - [Matched 2 x 2 grid](#93-matched-2-x-2-grid)
   - [Viewing aggregates in the console](#94-viewing-aggregate-metrics-in-the-console)
10. [Interpretation checklist](#10-interpretation-checklist)
11. [Consolidated metric index](#11-consolidated-metric-index)
    - [Standing and event-level metrics](#111-standing-and-event-level-metrics)
    - [Episode and pooled metrics](#112-episode-and-pooled-cell-metrics)
    - [Information metrics and diagnostics](#113-information-metrics-and-their-diagnostics)

## 1. Scientific purpose

The game holds the HiddenBench task, population, evidence allocation, option
alphabet, initial state, controller, and recorded observables fixed while
switching the population transition mechanism:

1. `dynamics_mode: reasoning` uses LLM messages and an LLM focal vote update;
2. `dynamics_mode: classical` uses a provider-free stochastic multi-opinion
   imitation kernel.

This makes it possible to compare semantic reasoning with a classical opinion
process under the same feedback-controller measurement and action.

One event records the causal tuple

```text
population before -> sensor measurement -> controller action -> population after
N_t / X_t         -> Y_t                -> U_t              -> N_t1 / X_t1
```

where:

- `X_t` is the labelled vector of all agent opinions;
- `N_t` is the occupation-count vector;
- `Y_t` is the sensor count vector;
- `U_t` is `ADVOCATE_Z` or `NO_OP`;
- `X_t1` and `N_t1` are the post-event states.

The count state `N_t` is the primary population variable for the implemented
information analysis. The labelled state remains persisted because an agent's
identity determines which private HiddenBench evidence it holds.

## 2. State and event schedule

Let:

- `N` be the number of agents;
- `K` be the number of task options;
- `X_i(t)` be agent `i`'s current option;
- `n_j(t)` be the number of agents on option `j`;
- `p_j(t) = n_j(t) / N` be its population share;
- `Z` be the controller/analysis target;
- `o*` be the correct HiddenBench answer.

The option ordering is always `possible_answers` from the task. Count vectors
are encoded in this canonical order, never dictionary insertion order.

The shipped first experiments use:

```yaml
task_id: evacuation_north_hill
population_size: 4
horizon: 20
initialization:
  mode: explicit
  initial_votes: [East Town, East Town, North Hill, West City]
```

Therefore both truth support and controller-target support initially equal
`1/4`.

### 2.1 Reasoning dynamics

At each event:

1. a focal agent and peer are sampled uniformly without replacement;
2. under `NO_OP`, the pair privately exchanges the configured number of
   messages and the focal agent makes a new vote;
3. under `ADVOCATE_Z`, the ordinary peer exchange is replaced by a fixed
   controller advocacy message and the focal agent makes its own new vote;
4. only the focal opinion may change.

The controller does not overwrite or force the reasoning agent's vote. Its
message contains target advocacy but no hidden task evidence.

With explicit initial votes, a horizon of 20 steps, and one message per agent, the
conservative reasoning call plan is 60 provider requests per episode: two
message decisions plus one focal update per event. Advocacy events may use
fewer calls because they replace the peer exchange, but preflight retains the
conservative bound.

### 2.2 Classical/no-reasoning dynamics

The classical mode makes exactly zero provider calls. A uniformly selected
focal agent leaves source opinion `A` for a destination `B != A`. For the
default `destination_count_plus_offset` interaction factor,

```text
interaction_factor(A -> B) = (n_B + interaction_offset) / N
```

and the base channel weight is

```text
w_base(A -> B) = rate_direction * n_A * interaction_factor(A -> B).
```

The direction-specific rate is `forward_rate` or `reverse_rate`, based on the
canonical indices of `A` and `B`. If the controller advocates `B`, the kernel
adds

```text
w_ctrl(A -> B) = control_strength * n_A.
```

The destination is sampled proportionally to `w_base + w_ctrl`. This is a
focal-conditioned embedded jump chain: event index is the v1 clock, not
Gillespie physical time. The event records each candidate channel, its base
and control weights, the selected reaction, and the total normalizer.

## 3. Control configuration reference

The `control` section chooses whether each event receives a measured feedback
signal. This is separate from `game.options.classical.control_strength`, which
only says how strongly a classical transition reacts *after* an advocacy
signal exists.

### 3.1 Supported mechanisms

For `hidden_bench_imitation`, use one of these effective mechanisms:

| `control.mechanism` | Sensor? | Action? | Effect |
| --- | --- | --- | --- |
| `none` | No | No | Runs the ordinary reasoning or classical dynamics without a controller. |
| `threshold_target` | Yes | `ADVOCATE_Z` or `NO_OP` | Samples current opinions and advocates the configured target when sampled support is below a threshold. |

The first matched grid varies this field between `none` and
`threshold_target` while holding the initial state and dynamics mode fixed.

### 3.2 `none`: uncontrolled behavior

```yaml
control:
  mechanism: none
  options: {}
```

`options` may be omitted. No sensor sample is drawn, no controller message or
classical control weight is applied, and the controller-only event metrics are
NA. This is the proper uncontrolled comparison; do not imitate it by setting
`control_strength: 0` on a live controller, because that would still measure
the population and produce controller actions.

For matched analysis, the no-control events still project `m_ctrl` onto the
correct answer. That permits direct `m_ctrl` comparisons with the shipped
controlled cells, whose target is `correct`.

### 3.3 `threshold_target`: measured feedback

The complete canonical block is:

```yaml
control:
  mechanism: threshold_target
  options:
    target: correct
    sensor_sample_size: 2
    policy: threshold_target
    threshold: 0.5
```

This means:

1. resolve `target` to one task option `Z`;
2. sample `sensor_sample_size` agents uniformly without replacement;
3. calculate the fraction of sampled opinions currently equal to `Z`;
4. emit `ADVOCATE_Z` if that fraction is strictly below `threshold`;
5. otherwise emit `NO_OP`.

The controller only observes the sampled current opinions and agent IDs. It
does not receive the full occupation state through its sensor and does not see
or inject hidden task evidence.

### 3.4 Every `threshold_target` field

| YAML field | Legal values | Default | Meaning and effect |
| --- | --- | --- | --- |
| `control.mechanism` | `threshold_target` | No implicit game-level default should be relied on | Selects the HiddenBench imitation feedback controller. |
| `options.target` | `correct`, an exact option-label string, or a non-negative zero-based integer index | `correct` | The option advocated by the controller. Resolution occurs against the current task. |
| `options.sensor_sample_size` | Integer `>= 1` and `<= N` | `1` | Number of agents sampled without replacement at every event. Larger samples are more accurate but reduce sensor variation; `N` reveals the full population share. |
| `options.policy` | Exactly `threshold_target` | `threshold_target` | Audit label and policy selector. No other policy is implemented; another value is rejected during control creation. |
| `options.threshold` | Number in `[0, 1]` | `0.5` | Advocacy cutoff. The comparison is strict: advocate when measured target share `< threshold`. |

Useful threshold edge cases:

- `threshold: 0.0` can never trigger advocacy because a share cannot be below
  zero; the controller always emits `NO_OP`;
- `threshold: 1.0` advocates unless every sampled agent is already on the
  target;
- raising the threshold generally increases advocacy frequency;
- lowering it generally increases `NO_OP` frequency.

Both actions need nontrivial support for meaningful actuation MI/CMI. Always
inspect `controller_action_entropy_bits`, `n_advocate`, and `n_noop` before
interpreting an actuation estimate.

### 3.5 Choosing another target

`target: correct` does **not** mean the literal option named `correct`. It is a
special keyword resolved at runtime to the task's `correct_answer`.

For the shipped task, the canonical option order is:

| Zero-based index | Exact option label |
| ---: | --- |
| `0` | `West City` |
| `1` | `East Town` |
| `2` | `North Hill` |

and `evacuation_north_hill` has `correct_answer: North Hill`.

These three target declarations are therefore equivalent for that task:

```yaml
# Resolve whatever answer is correct for the selected task.
target: correct
```

```yaml
# Select by exact label. Quotes are recommended for labels containing spaces.
target: "North Hill"
```

```yaml
# Select by zero-based position in possible_answers.
target: 2
```

To advocate a wrong or alternative option deliberately:

```yaml
control:
  mechanism: threshold_target
  options:
    target: "East Town"
    sensor_sample_size: 2
    policy: threshold_target
    threshold: 0.5
```

or equivalently for this task:

```yaml
target: 1
```

An integer outside `[0, K)` is rejected with a clear range error. A string
must match an option label exactly, including capitalization and spaces;
otherwise the run fails when resolving population observables. `correct` is
the portable choice when changing `task_id`, while a label or index deliberately
fixes a task-specific target.

Changing `target` changes all target-directed quantities, including
`p_ctrl`, `m_ctrl`, `focal_adopted_target`, `focal_left_target`, sensor error,
target-adoption lift, `Z_t`, and `target_actuation_cmi`. It does not change
`p_truth`, `m_truth`, or which answer HiddenBench considers correct.

### 3.6 Worked sensor/action examples

For the shipped settings, `N = 4`, sample size `S = 2`, target `North Hill`,
and threshold `0.5`. Let `y_Z` be the number of sampled North Hill opinions.

| Sampled target count `y_Z` | Measured share `y_Z / 2` | Test `< 0.5` | Controller action |
| ---: | ---: | --- | --- |
| `0` | `0.0` | true | `ADVOCATE_Z` |
| `1` | `0.5` | false | `NO_OP` |
| `2` | `1.0` | false | `NO_OP` |

The comparison is strictly less-than, so a sample exactly at the threshold is
`NO_OP`.

For general sample size `S`, let `y_Z(t)` be sampled target count. Then:

```text
sensor_target_share = y_Z(t) / S

ADVOCATE_Z  if y_Z(t) / S < threshold
NO_OP       otherwise.
```

Sampling is without replacement, so conditional on the true target count
`n_Z(t)`, `y_Z(t)` follows a finite-population hypergeometric distribution.

### 3.7 Effect in reasoning and classical modes

The same `threshold_target` measurement and action semantics feed both
dynamics modes, but the actuator differs.

In reasoning mode:

- `ADVOCATE_Z` replaces the ordinary peer exchange with the fixed message
  `The external controller currently advocates option <Z>. Reconsider your
  current position before committing your next vote.`;
- the focal LLM still chooses and validates its own vote;
- `NO_OP` leaves the ordinary focal/peer exchange in place;
- the controller never directly overwrites an opinion.

In classical mode:

- `ADVOCATE_Z` adds
  `classical.control_strength * n_source` to each eligible transition channel
  whose destination is `Z`;
- `NO_OP` adds zero control weight;
- destination sampling still occurs stochastically from all candidate channel
  weights;
- the mode remains provider-free.

Therefore `classical.control_strength` is meaningful only with
`threshold_target` advocacy events. It is not another control mechanism and
does not affect reasoning-mode votes.

### 3.8 Unsupported and compatibility forms

The repository-wide registry also contains `forced_action`, but
`hidden_bench_imitation` does not consume the action-override hook used by that
mechanism. Do not use it for this game: it will not provide the measured
message/transition control described here. The supported imitation choices are
`none` and `threshold_target`.

An older compatibility form exists under `game.options.controller`:

```yaml
game:
  options:
    controller:
      enabled: true
      target: correct
      sensor_sample_size: 2
      threshold: 0.5
```

New configurations should not use it. Prefer the top-level `control` section,
because that is the canonical experiment/grid axis and is recorded consistently
in cell overrides. Do not configure both forms.

## 4. Population observables

### 4.1 Per-option counts and shares

`occupation_counts_before` and `occupation_counts_after` store every `n_j`.
`population_shares_before` and `population_shares_after` store every `p_j`.

The streaming metric `population_action_share_per_option` writes one row per
option per event. Its values sum to one.

### 4.2 Truth and controller-target projections

The raw target shares are:

```text
p_truth = p_o*
p_ctrl  = p_Z.
```

The aligned order parameter for any target `q` is

```text
m(q) = (K * p_q - 1) / (K - 1).
```

Consequently:

```text
m_truth = (K * p_o* - 1) / (K - 1)
m_ctrl  = (K * p_Z  - 1) / (K - 1).
```

This normalization is zero at uniform support, one at unanimous support, and
negative when support is below `1/K`.

### 4.3 Population order and vote entropy

The target-independent order parameter is

```text
m_order = (K * max_j(p_j) - 1) / (K - 1).
```

The normalized vote entropy is

```text
H_vote = -sum_j p_j ln(p_j) / ln(K),
```

where zero-probability terms are omitted. It is zero at consensus and one at a
uniform population. In `metrics/streaming.csv`, this field is named
`normalized_vote_entropy`; in event and analysis files it is `H_vote`.

### 4.4 Other standing metrics

| Metric | Definition |
| --- | --- |
| `agent_current_action` | Current option held by each labelled agent. |
| `dominant_action_share` | Population share of the most common option. |
| `unshared_disclosure_rate` | Fraction of HiddenBench hidden facts detected in messages so far. Detection is keyword-based, so treat it as a lower bound. |
| `disclosure_reach` | One series per hidden fact: number of distinct agents whose recorded knowledge includes that fact. |

## 5. Event-level behavioral metrics

Every item below is stored in `trajectory.jsonl`, included in the emitted
imitation-transition event, written to `event_metrics.csv` by offline analysis,
and exposed as a population-scope row in each episode's
`metrics/streaming.csv`.

Pre-event quantities use the state before sensor/controller actuation. The
unsubscripted `m_*` and `H_vote` fields in the event are post-event values.

### 5.1 Changes in population observables

```text
delta_m_ctrl  = m_ctrl(t+1)  - m_ctrl(t)
delta_m_truth = m_truth(t+1) - m_truth(t)
delta_m_order = m_order(t+1) - m_order(t)
delta_H_vote  = H_vote(t+1)  - H_vote(t).
```

Interpretation:

- positive `delta_m_ctrl` means movement toward the controller target;
- positive `delta_m_truth` means movement toward the correct answer;
- positive `delta_m_order` means greater concentration on whichever option is
  currently dominant;
- negative `delta_H_vote` means reduced vote diversity.

When `target: correct`, `m_ctrl` and `m_truth` are numerically identical, but
both are retained so later experiments can target a non-truth option without
changing the schema.

### 5.2 Focal transition indicators

```text
focal_changed = 1[focal_opinion_after != focal_opinion_before]

focal_adopted_target =
    1[focal_opinion_before != Z and focal_opinion_after == Z]

focal_left_target =
    1[focal_opinion_before == Z and focal_opinion_after != Z].
```

These are integer indicators, not probabilities. Adoption probabilities are
computed later by averaging eligible indicators.

### 5.3 Controller and sensor diagnostics

```text
u_advocate = 1[U_t == ADVOCATE_Z]

population_target_share = p_Z(t)

sensor_target_error =
    sensor_target_share - population_target_share

sensor_target_abs_error = abs(sensor_target_error).
```

`population_target_share` is explicitly the pre-actuation truth against which
the sensor is compared. Under a real controller event, `u_advocate` is zero or
one. Under no control, all five controller/sensor-only diagnostic values are
NA rather than misleading zeros.

## 6. Episode-level and pooled summaries

Offline analysis sorts events by `interaction_index` and reconstructs a state
trajectory containing the initial state plus every post-event state.

### 6.1 Population response

| Metric | Definition |
| --- | --- |
| `initial_m_ctrl` | Target projection at state 0. |
| `final_m_ctrl` | Target projection after the final event. |
| `delta_final_m_ctrl` | `final_m_ctrl - initial_m_ctrl`. |
| `initial_m_truth` | Truth projection at state 0. |
| `final_m_truth` | Truth projection after the final event. |
| `delta_final_m_truth` | `final_m_truth - initial_m_truth`. |
| `mean_m_ctrl` | Arithmetic mean over state 0 and every post-event state. |
| `mean_m_truth` | Same convention for `m_truth`. |
| `mean_m_order` | Same convention for `m_order`. |
| `mean_H_vote` | Same convention for `H_vote`. |
| `auc_m_ctrl` | V1 equal-event-spacing mean of the `m_ctrl` trajectory. |
| `auc_m_truth` | V1 equal-event-spacing mean of the `m_truth` trajectory. |

The exact stored convention is
`equal_event_spacing_mean_including_initial_state`. `auc_*` is therefore a
trajectory-average summary, not a physical-time integral.

### 6.2 Controller action statistics

For controlled events:

```text
controller_advocacy_rate = n_advocate / (n_advocate + n_noop)
controller_noop_rate     = n_noop     / (n_advocate + n_noop).
```

The binary controller action entropy in bits is

```text
H(U) = -sum_u p(u) log2 p(u).
```

The emitted names are:

- `controller_action_entropy_bits`;
- `n_advocate`;
- `n_noop`;
- `controller_degenerate`.

An episode/cell is degenerate if both actions do not occur or if `H(U)` is at
most `1e-6` bits. Actuation MI from a degenerate action distribution is
reported with an explicit non-interpretability flag.

### 6.3 Sensor quality

```text
sensor_target_bias = mean(sensor_target_error)
sensor_target_mae  = mean(sensor_target_abs_error).
```

### 6.4 Action-conditioned behavioral response

The analysis emits:

- `mean_delta_m_ctrl_advocate`;
- `mean_delta_m_ctrl_noop`;
- `mean_delta_m_truth_advocate`;
- `mean_delta_m_truth_noop`.

When both actions have support:

```text
advocacy_delta_m_ctrl =
    E[delta_m_ctrl | ADVOCATE_Z] - E[delta_m_ctrl | NO_OP]

advocacy_delta_m_truth =
    E[delta_m_truth | ADVOCATE_Z] - E[delta_m_truth | NO_OP].
```

### 6.5 Target-adoption response

Only events whose focal agent is not already on `Z` enter the adoption
denominator:

```text
target_adoption_probability_advocate =
    P(focal_adopted_target | ADVOCATE_Z, Xf_t != Z)

target_adoption_probability_noop =
    P(focal_adopted_target | NO_OP, Xf_t != Z)

target_adoption_lift =
    target_adoption_probability_advocate
    - target_adoption_probability_noop.
```

These directional response metrics are essential: positive mutual information
alone says that control perturbed the population, not that it moved the
population toward `Z`.

### 6.6 Cell summary convention

Population trajectory fields in `cell_summaries.csv` are means of the episode
summaries. Controller, sensor, conditional-delta, and adoption fields are
pooled across all eligible event rows in the cell/run. The file also records:

- `n_episodes` and `n_events`;
- `initial_state`;
- `all_episodes_share_initial_state`;
- `dynamics_mode` and `control_mechanism`.

### 6.7 Four-grid contrasts

For the matched grid, `cell_contrasts.csv` computes:

```text
control_effect_within_reasoning  = B - A
control_effect_within_classical  = D - C
reasoning_effect_without_control = A - C
reasoning_effect_under_feedback  = B - D.
```

Each contrast is emitted for `final_m_ctrl`, `delta_final_m_ctrl`,
`final_m_truth`, `delta_final_m_truth`, `mean_m_ctrl`, `mean_m_truth`,
`auc_m_ctrl`, and `auc_m_truth`.

## 7. Implemented discrete information analysis

The information analysis is post-hoc over persisted `trajectory.jsonl` files.
It does not consume independent streaming rows and does not make provider
calls.

### 7.1 Canonical event adapter

For every event, the adapter constructs:

```text
N_t   := tuple(occupation_counts_before in possible_answers order)
N_t1  := tuple(occupation_counts_after  in possible_answers order)
Y_t   := tuple(sensor_count_vector      in possible_answers order)
U_t   := controller_action
Z_t   := target count n_Z(t)
Z_t1  := target count n_Z(t+1)
Xf_t  := focal_opinion_before
Xf_t1 := focal_opinion_after.
```

`Z_t` uses the integer target count rather than floating-point `m_ctrl`.
For fixed `N` and `K` they are one-to-one, and integer categories are safer for
direct contingency counting.

No-control events have `Y_t = NA` and `U_t = NA`; they therefore produce
behavioral/population summaries but not fabricated sensing or actuation MI.

### 7.2 Four statistics

#### Sensing mutual information

```text
sensing_mi = I(N_t ; Y_t).
```

This measures how much the partial sensor observation says about the true
population occupation state.

#### Population actuation conditional mutual information

```text
population_actuation_cmi = I(U_t ; N_t1 | N_t).
```

This asks whether the controller action predicts the full next occupation
state after conditioning on the current occupation state.

#### Target-projected actuation CMI

```text
target_actuation_cmi = I(U_t ; Z_t1 | Z_t).
```

This is the lower-dimensional target-count projection. It is less sparse than
the full occupation-state table and directly addresses movement in target
support.

#### Focal actuation CMI

```text
focal_actuation_cmi = I(U_t ; Xf_t1 | Xf_t, N_t).
```

The conditioning category is the tuple `(Xf_t, N_t)`. This local diagnostic
uses the one agent that can change during an event and is often less sparse
than the full population transition.

### 7.3 MI and CMI definitions

For discrete variables, the unsmoothed mutual information is

```text
I(X;Y) = H(X) + H(Y) - H(X,Y)
       = sum_x,y p(x,y) log2[p(x,y) / (p(x)p(y))].
```

The implemented conditional mutual information uses

```text
I(X;Y|C) = H(X,C) + H(Y,C) - H(C) - H(X,Y,C).
```

All information values are in bits.

### 7.4 Estimator variants

Every information row retains the repository's three direct-counting
variants:

| Output field | Definition |
| --- | --- |
| `unsmoothed` | Plug-in estimate from empirical contingency counts. This is the main reported estimate and is duplicated in `estimate`. |
| `jeffreys` | Adds `0.5` to every cell of the complete observed-level contingency table before computing MI/CMI. |
| `miller_madow` | Computes the count estimate with the Miller–Madow correction applied to each constituent entropy. |
| `observations` | Number of event observations entering the table. |

The output identifies `main_estimator_variant: unsmoothed` and
`estimator_variant: direct_counting` explicitly. Smoothing is not silently
substituted for the main result.

### 7.5 Episode bootstrap

Confidence intervals resample complete episodes, not event rows:

1. collect unique episode IDs in a cell;
2. draw the same number of episode IDs with replacement;
3. concatenate all events from each drawn episode, including repeated draws;
4. recompute the main unsmoothed estimate;
5. take percentile bounds at the configured confidence level.

The output fields are `bootstrap_ci_low` and `bootstrap_ci_high`.

### 7.6 Temporal nulls

Each null transformation preserves episode membership and event-level
marginals while breaking the relevant temporal alignment:

- sensing null: randomly permute `Y_t` within each episode while keeping
  `N_t` fixed;
- actuation nulls: randomly permute `U_t` within each episode while keeping
  population/focal transitions fixed.

Each permutation and all three estimator variants are written to
`information_nulls.csv`. The estimate table also records the main-estimator
`null_mean`, `null_ci_low`, and `null_ci_high`.

### 7.7 Support and sparsity diagnostics

Every information estimate reports:

- `n_episodes` and `n_events`;
- `unique_N_t_states` and `unique_Y_t_states`;
- `number_of_U_t_classes_observed`;
- `H_U_bits`;
- `occupied_conditioning_states`;
- minimum, median, and maximum events per conditioning state;
- `fraction_events_singleton_conditioning_states`;
- `sparse_conditioning_table`;
- `controller_degenerate`;
- `scientifically_interpretable`.

The table is flagged sparse when the median conditioning-state count is below
five or any conditioning state is a singleton. For sensing MI, controller
degeneracy is not applicable. For actuation CMI, fewer than two observed
actions or effectively zero `H(U)` sets `scientifically_interpretable` false.
The numeric estimate remains available for audit; the flag prevents it from
being silently treated as evidence of control.

## 8. Analysis output files

`mas_cc analysis hidden-bench-imitation` writes:

| File | Contents |
| --- | --- |
| `analysis_summary.json` | Run/cell/episode/event counts, matched-initial-state check, main estimator, and AUC convention. |
| `event_metrics.csv` | Canonical `N_t`, `N_t1`, `Y_t`, `U_t`, target/focal encodings, and all 12 derived event diagnostics. |
| `episode_summaries.csv` | One row per episode with population, controller, sensor, conditional-response, and adoption summaries. |
| `cell_summaries.csv` | Pooled summary for each grid cell, or one pooled row for a standalone run. |
| `option_share_trajectories.csv` | Long-form option-share trajectories including state 0. |
| `order_parameter_trajectories.csv` | `m_ctrl`, `m_truth`, `m_order`, and `H_vote` trajectories including state 0. |
| `information_estimates.csv` | Four MI/CMI estimates, bootstrap intervals, null summaries, and support fields. |
| `information_nulls.csv` | One row per statistic and null permutation. |
| `support_diagnostics.csv` | Compact support, sparsity, entropy, and interpretability view. |
| `cell_contrasts.csv` | Four-grid contrasts for final and trajectory-average response metrics. |

Within each completed episode, raw and streaming files remain available under:

```text
data/episodes/<episode-id>/trajectory.jsonl
data/episodes/<episode-id>/events.jsonl
data/episodes/<episode-id>/metrics/streaming.csv
```

## 9. Ready-to-run configurations

Run commands below from the repository root. The university provider requires:

```bash
export POTSDAM_API_KEY='...'
export BASE_POTSDAM_LLM_URL='...'
```

`--no-capture-output` and `logging.console: true` keep banners, progress, and
completion messages visible.

### 9.1 Separate reasoning run: 10 episodes

Preflight cost and demand:

```bash
conda run -n MA-CC --no-capture-output \
  python -m mas_cc.cli.main experiment preflight \
  --config configs/runs/hidden_bench/hidden_bench_imitation_reasoning_control_10.yaml \
  --output-dir inspection/hidden_bench_imitation_reasoning_control_10_preflight
```

The reasoning preflight uses live university price metadata and therefore
requires the configured base URL to be reachable. It sends no completion
requests.

Run after inspecting the report:

```bash
conda run -n MA-CC --no-capture-output \
  python -m mas_cc.cli.main experiment run \
  --config configs/runs/hidden_bench/hidden_bench_imitation_reasoning_control_10.yaml \
  --output-dir results \
  --approve-preflight inspection/hidden_bench_imitation_reasoning_control_10_preflight/preflight_id.txt
```

The run config enables all four MI/CMI statistics, so the experiment command
automatically writes them to `hidden_bench_imitation_analysis/`. To recompute
the analysis later with different settings without rerunning episodes, use:

```bash
conda run -n MA-CC --no-capture-output \
  python -m mas_cc.cli.main analysis hidden-bench-imitation \
  --run-dir results/hidden_bench_imitation/hidden-bench-imitation-reasoning-control-10/hidden-bench-imitation-reasoning-control-10-20260810 \
  --bootstrap-resamples 1000 \
  --null-permutations 1000
```

### 9.2 Separate classical/no-reasoning run: 10 episodes

Preflight:

```bash
conda run -n MA-CC --no-capture-output \
  python -m mas_cc.cli.main experiment preflight \
  --config configs/runs/hidden_bench/hidden_bench_imitation_classical_control_10.yaml \
  --output-dir inspection/hidden_bench_imitation_classical_control_10_preflight
```

The classical configuration retains the same university-provider declaration
for parity but has hard budgets of zero requests and zero tokens. Its
provider-free preflight is permitted without live pricing.

Run:

```bash
conda run -n MA-CC --no-capture-output \
  python -m mas_cc.cli.main experiment run \
  --config configs/runs/hidden_bench/hidden_bench_imitation_classical_control_10.yaml \
  --output-dir results \
  --approve-preflight inspection/hidden_bench_imitation_classical_control_10_preflight/preflight_id.txt
```

This config also computes all four MI/CMI statistics automatically. To
recompute them later without rerunning episodes, use:

```bash
conda run -n MA-CC --no-capture-output \
  python -m mas_cc.cli.main analysis hidden-bench-imitation \
  --run-dir results/hidden_bench_imitation/hidden-bench-imitation-classical-control-10/hidden-bench-imitation-classical-control-10-20260810 \
  --bootstrap-resamples 1000 \
  --null-permutations 1000
```

### 9.3 Matched 2 x 2 grid

The first grid contains exactly:

| Cell | Dynamics | Control |
| --- | --- | --- |
| A | reasoning | none |
| B | reasoning | threshold target |
| C | classical | none |
| D | classical | threshold target |

It uses 12 episodes per cell and the same initial vote vector in every cell.

```bash
conda run -n MA-CC --no-capture-output \
  python -m mas_cc.cli.main experiment preflight \
  --config configs/runs/hidden_bench/hidden_bench_imitation_first_control_grid.yaml \
  --output-dir inspection/hidden_bench_imitation_first_control_grid_preflight

conda run -n MA-CC --no-capture-output \
  python -m mas_cc.cli.main experiment run \
  --config configs/runs/hidden_bench/hidden_bench_imitation_first_control_grid.yaml \
  --output-dir results \
  --approve-preflight inspection/hidden_bench_imitation_first_control_grid_preflight/preflight_id.txt
```

Analyze the resulting grid:

```bash
conda run -n MA-CC --no-capture-output \
  python -m mas_cc.cli.main analysis hidden-bench-imitation \
  --run-dir results/hidden_bench_imitation/hidden-bench-imitation-first-control-grid/hidden-bench-imitation-first-control-grid-20260810 \
  --bootstrap-resamples 1000 \
  --null-permutations 1000
```

### 9.4 Viewing aggregate metrics in the console

The analysis command prints its JSON run summary. To print the complete pooled
metric row for the reasoning run:

```bash
conda run -n MA-CC --no-capture-output python -c \
"import pandas as pd; p='results/hidden_bench_imitation/hidden-bench-imitation-reasoning-control-10/hidden-bench-imitation-reasoning-control-10-20260810/hidden_bench_imitation_analysis/cell_summaries.csv'; print(pd.read_csv(p).to_string(index=False))"
```

Change both `reasoning` path components to `classical` for the classical run.

## 10. Interpretation checklist

Before treating an MI/CMI value as a control result, check in this order:

1. both `ADVOCATE_Z` and `NO_OP` occur;
2. `controller_action_entropy_bits` is not approximately zero;
3. sensor target shares vary and sensor error is consistent with the finite
   sample size;
4. `advocacy_delta_m_ctrl` or `target_adoption_lift` has the intended sign;
5. actuation CMI exceeds its within-episode action-permutation null;
6. conditioning tables are not dominated by singleton states;
7. the result reproduces across episodes and, eventually, a larger run.

Typical diagnoses:

| Observation | Interpretation |
| --- | --- |
| `H(U) ~= 0` | Sensor/policy action is degenerate; MI is not informative about controllability. |
| Sensing MI is positive but actuation CMI and behavioral response are absent | The sensor works, but the intervention does not affect updates. |
| Behavioral response exists but CMI is unstable | Sample size or conditioning sparsity is the likely issue; inspect target/focal projections first. |
| Actuation CMI exceeds its null but `m_ctrl` does not move toward `Z` | The controller perturbs dynamics but does not provide useful directional control. |
| Reasoning and classical responses are indistinguishable | Semantic reasoning has not changed the measured macroscopic controllability under this pilot. |

Ten or twelve episodes per cell are sanity-pilot sizes, not final MI sample
sizes. Use the emitted support and degeneracy diagnostics before scaling to
50–100 episodes. Nonzero MI alone is not a scientific green light.

## 11. Consolidated metric index

This section is a name-oriented index for finding a metric quickly. The
detailed definitions and interpretation rules remain in Sections 4–7.

### 11.1 Standing and event-level metrics

| Exact metric name | Level | Meaning |
| --- | --- | --- |
| `population_action_share_per_option` | event, one row per option | Current fraction of the population holding each option. |
| `agent_current_action` | event, one row per agent | Current option held by a labelled agent. |
| `dominant_action_share` | event | Share of the most common option. |
| `m_truth` | event/state | Order parameter aligned to the correct answer. |
| `m_ctrl` | event/state | Order parameter aligned to the controller target. |
| `m_order` | event/state | Target-independent population concentration. |
| `normalized_vote_entropy` | streaming event | Normalized vote entropy; the same quantity is called `H_vote` in trajectory and analysis files. |
| `delta_m_ctrl` | transition | Post-event minus pre-event `m_ctrl`. |
| `delta_m_truth` | transition | Post-event minus pre-event `m_truth`. |
| `delta_m_order` | transition | Post-event minus pre-event `m_order`. |
| `delta_H_vote` | transition | Post-event minus pre-event normalized vote entropy. |
| `focal_changed` | transition indicator | One when the selected focal agent changes opinion. |
| `focal_adopted_target` | transition indicator | One when a focal agent not previously on the target adopts it. |
| `focal_left_target` | transition indicator | One when a focal agent previously on the target leaves it. |
| `u_advocate` | controlled transition indicator | One for `ADVOCATE_Z`, zero for `NO_OP`, and NA without a controller. |
| `sensor_target_share` | controlled transition | Fraction of the sensor sample holding the target before actuation. |
| `population_target_share` | controlled transition | True population target share before actuation. |
| `sensor_target_error` | controlled transition | `sensor_target_share - population_target_share`. |
| `sensor_target_abs_error` | controlled transition | Absolute sensor target error. |
| `unshared_disclosure_rate` | event | Keyword-detected fraction of hidden facts disclosed in messages so far. |
| `disclosure_reach` | event, one series per fact | Number of agents whose recorded knowledge contains a hidden fact. |

The twelve transition/controller metrics requested for direct inspection are
therefore:

```text
delta_m_ctrl, delta_m_truth, delta_m_order, delta_H_vote,
focal_changed, focal_adopted_target, focal_left_target, u_advocate,
sensor_target_share, population_target_share, sensor_target_error,
sensor_target_abs_error
```

They appear in `trajectory.jsonl`, `event_metrics.csv`, and the episode
streaming metrics as described in Section 5. Controller-only fields are NA in
an uncontrolled cell by design.

### 11.2 Episode and pooled cell metrics

| Exact metric name | Meaning |
| --- | --- |
| `initial_m_ctrl`, `final_m_ctrl`, `delta_final_m_ctrl` | Initial, final, and final-minus-initial controller-target order. |
| `initial_m_truth`, `final_m_truth`, `delta_final_m_truth` | Initial, final, and final-minus-initial truth-aligned order. |
| `mean_m_ctrl`, `mean_m_truth`, `mean_m_order`, `mean_H_vote` | State-trajectory means including the initial state. |
| `auc_m_ctrl`, `auc_m_truth` | Equal-event-spacing trajectory means including the initial state. |
| `controller_advocacy_rate`, `controller_noop_rate` | Pooled fractions of controlled events using each action. |
| `controller_action_entropy_bits` | Binary controller action entropy in bits. |
| `n_advocate`, `n_noop` | Counts of the two controller actions. |
| `controller_degenerate` | True when both actions lack support or action entropy is effectively zero. |
| `sensor_target_bias`, `sensor_target_mae` | Mean signed and mean absolute sensor target error. |
| `mean_delta_m_ctrl_advocate`, `mean_delta_m_ctrl_noop` | Mean target-order change conditional on each action. |
| `mean_delta_m_truth_advocate`, `mean_delta_m_truth_noop` | Mean truth-order change conditional on each action. |
| `advocacy_delta_m_ctrl`, `advocacy_delta_m_truth` | Advocate conditional mean minus no-op conditional mean. |
| `target_adoption_probability_advocate` | Target-adoption probability among eligible focal agents under advocacy. |
| `target_adoption_probability_noop` | Target-adoption probability among eligible focal agents under no-op. |
| `target_adoption_lift` | Advocate adoption probability minus no-op adoption probability. |
| `n_episodes`, `n_events` | Number of episodes and transitions represented by the summary. |
| `initial_state`, `all_episodes_share_initial_state` | Initial-state audit fields for matched comparisons. |
| `dynamics_mode`, `control_mechanism` | Cell identity fields. |

Episode rows are written to `episode_summaries.csv`. Pooled run/cell rows are
written to `cell_summaries.csv`. See Section 6.6 for the difference between
episode-mean and event-pooled aggregation.

### 11.3 Information metrics and their diagnostics

| Exact name | Meaning |
| --- | --- |
| `sensing_mi` | `I(N_t; Y_t)`: information in the sensor count vector about the population count vector. |
| `population_actuation_cmi` | `I(U_t; N_t1 | N_t)`: action information about the full next population state, conditional on its current state. |
| `target_actuation_cmi` | `I(U_t; Z_t1 | Z_t)`: action information about next target support, conditional on current target support. |
| `focal_actuation_cmi` | `I(U_t; Xf_t1 | Xf_t, N_t)`: action information about the focal agent's next opinion given its current opinion and population state. |

Each statistic has `estimate`/`unsmoothed`, `jeffreys`, `miller_madow`, and
`observations` fields. Its uncertainty/null fields are `bootstrap_ci_low`,
`bootstrap_ci_high`, `null_mean`, `null_ci_low`, and `null_ci_high`.

The most important audit fields are `number_of_U_t_classes_observed`,
`H_U_bits`, `occupied_conditioning_states`,
`fraction_events_singleton_conditioning_states`,
`sparse_conditioning_table`, `controller_degenerate`, and
`scientifically_interpretable`. Do not interpret an actuation estimate as
control evidence when `scientifically_interpretable` is false.
