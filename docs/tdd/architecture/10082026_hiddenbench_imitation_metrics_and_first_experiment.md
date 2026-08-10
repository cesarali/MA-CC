# HiddenBench Imitation — Metrics + First Control Experiment Implementation Brief

Date: 2026-08-10

## Purpose

The `hidden_bench_imitation` game is already implemented. Do not redesign it.

The goal now is to add the smallest set of behavioral/control diagnostics needed for the first scientific pilot, wire a first **reasoning ON vs classical OFF** experiment, and make the stored outputs immediately usable for the first discrete information-theoretic analysis.

Do **not** implement InfoNCE yet. Do **not** add a controller zoo. Do **not** change the scientific semantics of the existing game unless a bug is found.

The current implementation already records the critical event tuple

```text
population state -> controller measurement -> controller action -> next population state
n_t / X_t       -> Y_t                    -> U_t              -> n_{t+1} / X_{t+1}
```

and already emits `p_truth`, `m_truth`, `p_ctrl`, `m_ctrl`, `m_order`, `H_vote`, per-option population shares, current agent actions, `unshared_disclosure_rate`, and `disclosure_reach`.

Therefore the missing work is mostly **derived metrics, aggregate diagnostics, and an offline information-analysis adapter**. Do not duplicate data already in `events.jsonl` / `trajectory.jsonl`.

## 1. Scientific state used by the first analysis

For the first control analysis, use the mesoscopic occupation state as the primary population variable:

```text
n_t = (n_1(t), ..., n_K(t))
p_t = n_t / N
```

The full labelled state `X_t = (X_1(t), ..., X_N(t))` must still be logged because agent identity is tied to private HiddenBench evidence, but it is not the primary count-based variable.

The first information analysis should work at three levels:

```text
1. sensing:
   I(n_t ; Y_t)

2. population actuation:
   I(U_t ; n_{t+1} | n_t)

3. target-directed projection:
   I(U_t ; m_ctrl(t+1) | m_ctrl(t))
```

Also retain the less sparse local diagnostic:

```text
I(U_t ; X_focal(t+1) | X_focal(t), n_t)
```

Internally, for the target projection prefer the integer target count `n_Z(t)` instead of floating-point `m_ctrl(t)`; for fixed `N,K` they are one-to-one and integer categories are safer.

## 2. Add event-level behavioral metrics

Derive these from the already-recorded pre/post state:

```text
delta_m_ctrl  = m_ctrl(t+1)  - m_ctrl(t)
delta_m_truth = m_truth(t+1) - m_truth(t)
delta_m_order = m_order(t+1) - m_order(t)
delta_H_vote  = H_vote(t+1)  - H_vote(t)
```

If pre-event order parameters are not explicit fields, compute them from `occupation_counts_before` / `population_shares_before`.

Add focal transition indicators:

```text
focal_changed
focal_adopted_target
focal_left_target
```

with

```text
focal_changed =
    1[focal_opinion_after != focal_opinion_before]

focal_adopted_target =
    1[focal_opinion_before != Z and focal_opinion_after == Z]

focal_left_target =
    1[focal_opinion_before == Z and focal_opinion_after != Z]
```

Add controller/sensor diagnostics:

```text
u_advocate
sensor_target_share
population_target_share      # p_Z(t), BEFORE actuation
sensor_target_error
sensor_target_abs_error
```

For no-controller events, use the repository's normal missing/NA convention rather than misleading zeros.

## 3. Add episode-level / cell-level summaries

### Population response

Compute:

```text
initial_m_ctrl
final_m_ctrl
delta_final_m_ctrl
initial_m_truth
final_m_truth
delta_final_m_truth
mean_m_ctrl
mean_m_truth
mean_m_order
mean_H_vote
auc_m_ctrl
auc_m_truth
```

A simple equal-event-spacing trajectory mean is sufficient for the v1 AUC-style summary; document the convention.

### Controller action statistics

Compute:

```text
controller_advocacy_rate
controller_noop_rate
controller_action_entropy_bits
n_advocate
n_noop
```

For binary `U_t` use the standard Shannon entropy in bits.

Flag a controller-degenerate episode/cell when both actions do not occur or when action entropy is extremely small. Do not silently present an MI value as scientifically meaningful when `H(U)` is effectively zero.

### Sensor quality

Compute:

```text
sensor_target_bias = mean(sensor_target_error)
sensor_target_mae  = mean(sensor_target_abs_error)
```

RMSE is optional.

### Conditional behavioral response

When both controller actions have support, compute:

```text
E[delta_m_ctrl  | ADVOCATE_Z]
E[delta_m_ctrl  | NO_OP]
E[delta_m_truth | ADVOCATE_Z]
E[delta_m_truth | NO_OP]

advocacy_delta_m_ctrl =
    E[delta_m_ctrl | ADVOCATE_Z] - E[delta_m_ctrl | NO_OP]

advocacy_delta_m_truth =
    E[delta_m_truth | ADVOCATE_Z] - E[delta_m_truth | NO_OP]
```

Also compute:

```text
P(focal_adopted_target | ADVOCATE_Z, focal_before != Z)
P(focal_adopted_target | NO_OP,       focal_before != Z)

target_adoption_lift =
    P(adopt | ADVOCATE_Z) - P(adopt | NO_OP)
```

These are essential because positive MI alone does not mean useful control.

## 4. Offline discrete information analysis

Implement this post-hoc from persisted event files, not as a streaming metric.

Reuse the repository's current direct-counting MI / CMI estimators and surrogate machinery. Do not reimplement entropy estimation.

Create a small HiddenBench-imitation event adapter:

```text
N_t   := tuple(occupation_counts_before in canonical option order)
N_t1  := tuple(occupation_counts_after in canonical option order)
Y_t   := tuple(sensor_count_vector in canonical option order)
U_t   := controller_action
Z_t   := integer target count n_Z(t)
Z_t1  := integer target count n_Z(t+1)
Xf_t  := focal_opinion_before
Xf_t1 := focal_opinion_after
```

Compute:

```text
sensing_mi:
I(N_t ; Y_t)

population_actuation_cmi:
I(U_t ; N_t1 | N_t)

target_actuation_cmi:
I(U_t ; Z_t1 | Z_t)

focal_actuation_cmi:
I(U_t ; Xf_t1 | Xf_t, N_t)
```

Use the estimator variants already implemented in the repository and clearly identify the main reported estimate.

### Uncertainty and nulls

Use **episode-level bootstrap**, not independent event-row bootstrap.

For nulls:

```text
system -> controller:
shuffle/circularly perturb Y_t relative to N_t

controller -> system:
shuffle/circularly perturb U_t relative to the transition sequence
```

Reuse existing permutation/circular-shift infrastructure where possible.

For every estimate report:

```text
n_episodes
n_events
unique N_t states
unique Y_t states
number of U_t classes observed
H(U_t)
occupied conditioning states
min / median / max events per conditioning state
fraction of events in singleton conditioning states
estimator variant
bootstrap interval
null mean / interval
```

If tables are sparse, flag the problem rather than hiding it with smoothing.

## 5. First experiment: 2 x 2 factorial pilot

Create an executable config:

```text
configs/runs/hidden_bench/hidden_bench_imitation_first_control_grid.yaml
```

Use the repository's **current grid syntax as the source of truth**. Clone/adapt an existing executable grid config rather than inventing a new schema.

The grid must contain exactly four scientific cells:

```text
A. reasoning + no control
B. reasoning + threshold_target control
C. classical + no control
D. classical + threshold_target control
```

Comparisons:

```text
control effect within reasoning:  B - A
control effect within classical:  D - C
reasoning effect without control: A - C
reasoning effect under feedback:  B - D
```

## 6. Matched initial state

Do not allow reasoning and classical cells to initialize differently in this first grid.

Use:

```text
task_id: evacuation_north_hill
```

with explicit initial votes:

```yaml
initialization:
  mode: explicit
  initial_votes:
    - East Town
    - East Town
    - North Hill
    - West City
```

This yields initial correct/controller-target support of `1/4`, avoiding a controller-saturated boundary.

This is a controlled-dynamics pilot, not a reproduction of the natural HiddenBench pre-discussion distribution. A later experiment can obtain `X_0` from natural LLM initialization and replay the same realized state in matched ON/OFF cells.

## 7. Controller settings

Controlled cells:

```yaml
control:
  mechanism: threshold_target
  options:
    target: correct
    sensor_sample_size: 2
    policy: threshold_target
    threshold: 0.5
```

For `N=4` and the proposed initial state, sampling two agents without replacement gives substantial sensor/action variation.

Use the already-implemented classical control weight:

```yaml
classical:
  control_strength: 2.0
```

No-control cells must use the actual `none` / `NoneControl` mechanism, not a zero-strength fake controller.

## 8. Dynamics and horizon

Use:

```yaml
game:
  type: hidden_bench_imitation
  population_size: 4
  options:
    task_set: vanilla
    task_id: evacuation_north_hill
    profile: hidden
    assignment_scheme: bijective
    interactions: 20
    pairing: uniform_two_distinct
    messages_per_agent: 1
    memory_size: 0
    allow_relay: true
    stop_on_consensus: false
```

Classical settings:

```yaml
classical:
  kernel: irisarri_multi_opinion
  forward_rate: 1.0
  reverse_rate: 1.0
  interaction_factor: destination_count_plus_offset
  interaction_offset: 1.0
  control_strength: 2.0
```

Keep equal horizon across all cells.

## 9. Provider

Use exactly:

```yaml
llm_provider:
  type: university
  model: gwdg/qwen3-30b-a3b-instruct-2507
  credentials_env: POTSDAM_API_KEY
  base_url_env: BASE_POTSDAM_LLM_URL
  timeout_seconds: 60
  max_retries: 2
  request_concurrency: 10
  temperature: 0.0
  max_output_tokens: 128
  options:
    estimated_latency_seconds: 3.0
```

Classical cells must still plan and execute with zero provider calls.

Temperature `0.0` is intentional: remove one unnecessary LLM sampling source while preserving pair-selection, sensor, and classical-process stochasticity.

## 10. Episode count

For the first live sanity pilot use:

```text
12 episodes per cell
```

This is not the final MI sample size. It is enough to inspect trajectories, controller action support, target-specific response, table sparsity, provider behavior, and cost.

If the platform passes the acceptance criteria, rerun the identical scientific design at 50–100 episodes per cell subject to preflight cost.

## 11. Required first report

Produce:

1. per-option share trajectories by cell;
2. `m_ctrl`, `m_truth`, `m_order`, `H_vote` trajectories;
3. controller action frequency and `H(U)`;
4. sensor target share vs true population target share;
5. `delta_m_ctrl` and `delta_m_truth` split by `ADVOCATE_Z` / `NO_OP`;
6. target-adoption probability split by controller action;
7. final and trajectory-average response summaries for the four cells;
8. the four discrete information estimates;
9. bootstrap intervals and null distributions;
10. contingency/support diagnostics.

Do not create InfoNCE outputs.

## 12. Acceptance criteria

The platform passes if:

- all dynamical invariants remain valid;
- classical mode has zero provider calls;
- controlled reasoning never directly forces votes;
- both `ADVOCATE_Z` and `NO_OP` occur with nontrivial frequency;
- controller action entropy is not approximately zero;
- sensor observations vary and are consistent with finite-population sampling;
- at least one target-specific behavioral response is measurable;
- MI/CMI estimators produce finite values with transparent support diagnostics;
- null transformations reduce the corresponding information signal;
- reasoning and classical cells start from the exact same `X_0`.

A **scientific green light** requires more:

```text
1. controlled vs uncontrolled gives a reproducible target-specific shift
   in m_ctrl / target adoption,

AND

2. controller -> population information survives the appropriate null,

AND

3. the size or structure of the response differs between reasoning and
   classical dynamics.
```

Nonzero MI alone is not enough.

## 13. Failure diagnosis

```text
H(U) ~ 0
-> controller/sensor policy is degenerate; change sensor/threshold before MI work.

I(N_t;Y_t) works, but no behavioral response and no U->population CMI
-> sensing works; actuator/controller message is ineffective.

Behavioral response exists, but CMI is unstable
-> sample-size / conditioning sparsity problem; increase episodes or use the
   focal/target-count projection before changing the game.

U->population CMI > null, but m_ctrl does not move toward target
-> controller perturbs dynamics but is not useful control.

Reasoning and classical responses are indistinguishable
-> semantic reasoning has not changed macroscopic controllability; inspect
   evidence transmission/update behavior before scaling N.

Reasoning differs from classical and controlled response survives nulls
-> freeze v1 and proceed to scale N / sensor quality before InfoNCE.
```

## 14. Tests

Add focused tests for:

- exact `delta_m_*` on hand-created states;
- target-adoption indicators;
- `H(U)=0` for constant actions and `H(U)=1` for a 50/50 binary fixture;
- sensor target share/error;
- stable target-count encoding for 3- and 4-option tasks;
- canonical option ordering in the event extractor;
- independent MI fixture -> near zero;
- deterministic measurement fixture -> positive sensing MI;
- permuted controller action -> actuation CMI collapses;
- episode bootstrap resamples episodes, not rows;
- no-control cells do not fabricate controller metrics;
- classical grid cells make zero provider calls.

Do not fix unrelated stale test failures unless required.

## 15. Deliverables

Finish with:

```text
1. code changes for derived metrics;
2. offline HiddenBench-imitation information-analysis adapter;
3. tests;
4. executable config:
   configs/runs/hidden_bench/hidden_bench_imitation_first_control_grid.yaml
5. a short handoff with:
   - files changed;
   - exact metric names;
   - exact analysis command;
   - preflight/run commands;
   - resolved four-cell grid;
   - known limitations.
```
