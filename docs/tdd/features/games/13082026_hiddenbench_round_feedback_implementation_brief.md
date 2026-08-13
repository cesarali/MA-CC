# Implementation Brief — New HiddenBench Round-Level Budgeted Feedback Game

**Date:** 2026-08-13

## Purpose

Implement a **new game derived from the existing `hidden_bench_imitation` game**. Do **not** silently change the semantics of the existing game, because the current event-level experiments and reports must remain reproducible.

Recommended new game type:

```text
hidden_bench_imitation_round_feedback
```

Recommended package:

```text
src/mas_cc/games/hidden_bench/imitation_round_feedback/
```

The new game keeps the existing HiddenBench task/evidence machinery, one-focal-agent imitation geometry, reasoning/classical switch, prompts where reusable, population observables, recorder infrastructure, and post-hoc information estimators. The essential change is the **feedback timescale**:

> The controller senses and chooses its behavior once per population round. A round contains exactly `N` microscopic focal-update opportunities. If the controller chooses `ADVOCATE_Z`, exactly `b` of those `N` update positions are selected uniformly at random in advance and receive one controller influence slot. No further sensing or controller decision occurs until the next round.

This is a new scientific game, not a cosmetic regrouping of existing events.

---

# 1. Existing implementation to reuse

The current game is implemented in:

```text
src/mas_cc/games/hidden_bench/imitation/
├── __init__.py
├── README.md
├── analysis.py
├── classical.py
├── controller.py
├── game.py
├── metrics.py
├── prompts.py
├── runtime.py
└── state.py
```

Current responsibilities are already well separated:

- `game.py`: HiddenBench task loading, initialization, focal transitions, state/event records.
- `runtime.py`: reasoning/classical execution and current event-level feedback.
- `controller.py`: hypergeometric sensor, target policy, peer-style advocacy message.
- `classical.py`: provider-free classical transition code.
- `metrics.py`: order parameters and behavioral metrics.
- `analysis.py`: event adapter, direct-counting MI/CMI, bootstrap, nulls, support diagnostics.
- shared MI/entropy estimators: `src/mas_cc/analysis/estimators.py`.

Reuse the existing HiddenBench infrastructure rather than duplicating it:

- canonical task loader and answer ordering;
- hidden/full profile semantics;
- evidence assignment schemes;
- validated scalable population construction;
- initial vote logic;
- privacy checks;
- vote parsing/validation;
- disclosure utilities;
- generic experiment/grid/planning/storage stack;
- existing order parameters and current/activity code where compatible.

**Do not modify `hidden_bench_imitation` behavior.** Existing configs must continue to replay unchanged.

---

# 2. Scientific contract of the new game

For a task with answer alphabet

```text
A = {a_1, ..., a_K}
```

do not hard-code `K = 3`. The repository already supports HiddenBench tasks with different option counts.

Each agent has:

```text
E_i                      # shared + assigned hidden evidence
X_i                      # current committed option
H_i                      # private semantic history in reasoning mode
```

The labelled population state is:

```text
X = (X_1, ..., X_N)
```

The mesoscopic state is the canonical occupation-count tuple:

```text
n = (n_1, ..., n_K)
sum(n) = N
```

Only one focal agent may change its committed opinion at a microscopic update.

The two dynamics modes remain:

```text
reasoning -> focal update decided by the LLM
classical -> provider-free explicit stochastic transition rule
```

The controller must never directly overwrite a committed vote.

---

# 3. The two clocks

Introduce two explicit indices:

```text
round_index = k
within_round_index = r  # 0, ..., N-1
```

One population round contains **exactly `N` microscopic update opportunities**.

At the round boundary:

```text
n_k = population occupation state before the round
```

After all `N` microscopic updates:

```text
n_{k+1} = population occupation state after the round
```

Do not call the controller inside the microscopic loop.

The episode horizon for the new game should be configured in **rounds**, not in elementary interactions:

```yaml
rounds: 10
```

The total number of microscopic focal-update attempts is then:

```text
rounds * N
```

Keep a derived microscopic event counter for logging and replay.

---

# 4. Three separate control/social scales

The new game has three structurally distinct resources:

```text
q    = number of social influence slots at each microscopic focal update
q_c  = number of population agents sensed once at the start of each round
b    = exact number of controlled microscopic positions in an ADVOCATE round
```

Derived intensive quantities:

```text
sensing_fraction   = q_c / N
actuation_fraction = b / N
```

Validation:

```text
1 <= q <= N - 1
1 <= q_c <= N
0 <= b <= N
```

`q_c` and `b` must not be conflated. One is observation capacity; the other is actuation capacity.

---

# 5. Recommended configuration surface

Use a new game type and a new round-level control mechanism so old configs remain untouched.

Example:

```yaml
game:
  type: hidden_bench_imitation_round_feedback
  population_size: 32
  options:
    task_set: vanilla
    task_id: null
    profile: hidden
    assignment_scheme: exact_replication

    dynamics_mode: reasoning      # reasoning | classical
    rounds: 10
    social_group_size: 2          # q

    initialization:
      mode: local_vote
      initial_votes: null
      initial_distribution: null

    classical:
      # Keep this interface pluggable.
      # Another theory task may replace the current classical kernel.
      kernel: controlled_imitation_round_reference

control:
  mechanism: round_soft_target_budgeted
  options:
    target: correct
    sensor_sample_size: 8         # q_c
    policy: soft_target
    threshold: 0.5
    beta: 6.0
    intervention_budget: 8        # b
    template_version: 3

analysis:
  enabled: true
  estimators:
    - round_sensing_mi
    - round_population_actuation_cmi
    - round_target_actuation_cmi
    - round_truth_actuation_cmi
    - round_controller_action_entropy
    - round_controller_action_entropy_given_population
    - round_population_information_fraction
    - round_target_information_fraction
    - round_dual_action_state_fraction
    - round_target_signed_actuation
    - round_sensor_mae
    - round_sensor_mse
  options:
    bootstrap_resamples: 1000
    null_permutations: 1000
    confidence: 0.95
    seed: 20260813
```

If the repository control abstraction cannot express a round action, add the **smallest backward-compatible hook**, for example:

```python
Control.round_signal(...)
```

with an inert default implementation. Existing `override()` and `interaction_signal()` behavior must remain unchanged.

Do not force this new controller through the old per-event `interaction_signal()` semantics.

---

# 6. Exact round protocol

For every round `k`:

## 6.1 Snapshot the pre-round state

Record:

```text
X_k
n_k
population shares
m_truth_k
m_ctrl_k
m_order_k
H_vote_k
```

## 6.2 Sense exactly once

Sample:

```text
sensor_agent_ids
```

uniformly without replacement from all `N` agents, with exactly `q_c` sampled agents.

The controller receives only their current opinions/counts.

It must not receive:

- hidden evidence;
- task rationales;
- private memories;
- the full population state.

Store the sensor count vector in canonical option order.

For target `Z`, define:

```text
sensor_target_share = (# sampled agents holding Z) / q_c
```

## 6.3 Choose exactly one round action

The default policy is:

```text
P(ADVOCATE_Z | Y_k)
    = sigmoid(beta * (threshold - sensor_target_share))
```

Sample once:

```text
U_k in {NO_OP, ADVOCATE_Z}
```

Log the actual action probability:

```text
controller_advocate_probability
```

This is needed for a policy-preserving conditional-randomization null later.

No policy call is allowed again during this round.

## 6.4 Preallocate the intervention schedule

If:

```text
U_k == NO_OP
```

then:

```text
controlled_positions = []
```

If:

```text
U_k == ADVOCATE_Z
```

then draw exactly:

```text
b
```

unique integers from:

```text
{0, ..., N - 1}
```

uniformly without replacement.

This draw occurs **before microscopic update 0**.

Log the sorted set/list:

```text
controlled_positions
```

and its deterministic seed/replay metadata.

Do not choose specific controlled agents in advance. Choose **update positions**. Focal identities are still sampled dynamically at each microscopic update. This preserves the population-level exchangeability needed by the classical theory.

---

# 7. Microscopic update protocol

For each `r = 0, ..., N-1`:

1. Sample one focal agent and `q` distinct ordinary peers from the **current** population state.
2. Set:

   ```text
   controlled_slot = (r in controlled_positions)
   ```

3. If `controlled_slot == false`, use all `q` ordinary peer inputs.
4. If `controlled_slot == true`, replace exactly one of the `q` influence slots by the controller advocacy message for `Z`.

The focal must always see exactly `q` influence slots.

Therefore:

```text
ordinary step:
    q ordinary peer slots

controlled step:
    (q - 1) ordinary peer slots + 1 controller slot
```

For `q = 1`, a controlled update contains the controller slot and zero ordinary peer slots.

The peer that is displaced can be chosen uniformly from the `q` sampled peer slots. Log both the original sampled peer set and the effective influence set so replay and auditing remain possible.

Only the focal agent may update its opinion.

---

# 8. Reasoning-mode behavior

Reuse the existing HiddenBench imitation reasoning machinery as far as possible:

- focal evidence;
- bounded/unbounded private history;
- current vote;
- social input;
- vote parser and validator.

The controller advocacy message should reuse the current safe peer-style semantics:

- no hidden facts;
- no claim that it is an external controller;
- no direct vote overwrite;
- provider-free controller message;
- preferably the current fixed template version that avoids introducing a second random message channel.

The focal LLM is free to:

```text
accept the advocacy
reject it
retain its current vote
switch to a third option
```

Do not reward truth or reveal `o*` in the focal prompt.

---

# 9. Classical-mode interface

The new runtime must support a classical kernel that sees the same social structure as reasoning mode.

Do **not** reuse the old behavior where `q` is sampled/logged but discarded by the kernel.

Provide a clean interface, for example:

```python
classical_transition(
    *,
    population_state,
    focal_agent_id,
    focal_opinion,
    peer_agent_ids,
    peer_opinions,
    controlled_slot,
    controller_target,
    rng,
) -> ClassicalTransition
```

The theory agent may define the actual transition law separately.

Important requirements:

- no provider calls;
- one focal update at most;
- actual `q`-peer context must be available to the kernel;
- controlled and ordinary steps must be distinguishable;
- controller semantics must be slot replacement, not a hidden additive `control_strength` unless the final theory explicitly specifies that;
- reaction/detailed-balance metadata may be logged if useful, but **microscopic reversibility is not a requirement of this new game**.

---

# 10. New round record — required analysis layer

The new coarse-graining must be explicit in persisted data.

Write **one round-level record per round** using the existing recorder infrastructure. Prefer adding a new record/event type rather than adding a new global storage subsystem.

Recommended record type:

```text
imitation_round_feedback
```

Required fields:

```text
episode_id
round_index
seed
task_id
K
N
dynamics_mode

social_group_size                 # q
sensor_sample_size                # q_c
intervention_budget               # b
sensing_fraction                  # q_c / N
actuation_fraction                # b / N

controller_enabled
controller_target
controller_policy
controller_threshold
controller_beta
controller_action
controller_advocate_probability

sensor_agent_ids
sensor_observed_opinions
sensor_count_vector
sensor_target_share

controlled_positions
controlled_position_count

population_state_before
occupation_counts_before
population_shares_before
target_count_before
truth_count_before
m_ctrl_before
m_truth_before
m_order_before
H_vote_before

population_state_after
occupation_counts_after
population_shares_after
target_count_after
truth_count_after
m_ctrl_after
m_truth_after
m_order_after
H_vote_after

delta_m_ctrl
delta_m_truth
delta_m_order
delta_H_vote
```

The round record should be the source of the new round-level information analysis.

Do not derive the main round-level CMI by summing microscopic CMI values.

---

# 11. Extend microscopic event logging

Keep the rich microscopic trajectory because it is still needed for local response, activity, disclosure, and later memory analysis.

Every microscopic row in the new game should additionally include:

```text
round_index
within_round_index
round_controller_action
round_controller_target
round_controller_advocate_probability
controlled_slot
intervention_budget
controlled_positions_hash_or_id
sampled_peer_agent_ids
effective_peer_agent_ids
replaced_peer_agent_id              # null on ordinary steps
```

Keep existing pre/post focal opinion, occupation counts, reasoning messages, disclosure fields, and classical metadata where applicable.

---

# 12. New metrics layer

The existing event-level information metrics belong to the old controller timescale. Do not silently reuse their names for round-level quantities.

Implement new metrics with an explicit `round_` prefix.

## 12.1 Round sensing

### `round_sensing_mi`

```text
I(N_k ; Y_k)
```

where `N_k` is the canonical occupation-count tuple and `Y_k` is the canonical sensor-count tuple.

Report in bits.

Also add direct sensor diagnostics:

```text
round_sensor_mae
round_sensor_mse
```

using target-share error:

```text
sensor_target_share - true_target_share_at_round_start
```

Raw sensing MI must not be interpreted as directly comparable across different `q_c` without an alphabet/estimator caveat.

---

## 12.2 Headline round actuation / transfer entropy

### `round_population_actuation_cmi`

```text
I(U_k ; N_{k+1} | N_k)
```

This is the principal controller-to-population information quantity.

For a Markov classical round process it is the one-step transfer entropy at the feedback timescale.

For reasoning mode it is the first-order empirical round-level directed-information statistic. Do not claim the LLM process is Markov until history tests support that.

---

## 12.3 Lower-dimensional round actuation

### `round_target_actuation_cmi`

```text
I(U_k ; n_Z(k+1) | n_Z(k))
```

Use integer target counts, not floating-point magnetizations.

### `round_truth_actuation_cmi`

```text
I(U_k ; n_truth(k+1) | n_truth(k))
```

When the controller target is correct this equals the target projection by construction; still keep separate output names for wrong-target experiments.

Optional but useful:

```text
round_order_actuation_cmi
```

based on a stable discrete encoding of the order parameter or maximum occupation count.

---

## 12.4 Controller entropy and information fractions

### `round_controller_action_entropy`

```text
H(U_k)
```

### `round_controller_action_entropy_given_population`

```text
H(U_k | N_k)
```

Check numerically:

```text
round_population_actuation_cmi
    <= round_controller_action_entropy_given_population
```

up to estimator/tolerance issues.

### `round_population_information_fraction`

```text
I(U_k ; N_{k+1} | N_k) / H(U_k | N_k)
```

### `round_target_information_fraction`

```text
I(U_k ; n_Z(k+1) | n_Z(k)) / H(U_k | n_Z(k))
```

These are normalization diagnostics only. Do not call them thermodynamic efficiencies.

---

## 12.5 Action-overlap diagnostics

The coarser clock yields far fewer samples, so overlap diagnostics are mandatory.

Add:

```text
round_dual_action_state_fraction
round_dual_action_event_fraction
round_single_action_slice_fraction
round_conditioning_state_count
round_singleton_fraction
```

Definitions should parallel the existing imitation analysis but use **round-boundary states**.

A small CMI in a state with only one observed action is not interpretable as weak actuation.

---

## 12.6 Signed round response

CMI is unsigned. Add a state-adjusted signed response.

### `round_target_signed_actuation`

Conceptually:

```text
E[delta_m_ctrl | ADVOCATE_Z, state]
-
E[delta_m_ctrl | NO_OP, state]
```

aggregated only over conditioning states supporting both actions.

Also add, if useful:

```text
round_truth_signed_actuation
round_order_signed_actuation
```

Keep the exact weighting convention documented and deterministic.

---

# 13. Microscopic randomized-slot diagnostics

The preallocated schedule creates a useful local randomized intervention inside ADVOCATE rounds.

Define:

```text
C_{k,r} = 1 if microscopic position r is controlled
```

Within rounds with `U_k = ADVOCATE_Z`, optional local diagnostics are:

### `micro_slot_focal_actuation_cmi`

```text
I(C_{k,r} ; X_focal_after |
  X_focal_before, N_{k,r}, U_k = ADVOCATE_Z)
```

### `micro_slot_target_signed_response`

Compare `delta_m_ctrl` for controlled versus ordinary positions within ADVOCATE rounds, state adjusted where possible.

These are **local intervention diagnostics**, not the primary round-level transfer entropy.

Because exactly `b` positions are sampled uniformly in advance, schedule labels can be resampled/permuted within an ADVOCATE round while preserving the exact budget for a local randomization null.

---

# 14. Episode-level current and activity

Retain the current episode quantities:

```text
truth_current
truth_current_mean
truth_current_variance
truth_current_fano
```

and add/retain explicit path activity:

```text
truth_activity
```

with:

```text
truth_current
    = #(switches toward truth) - #(switches away from truth)

truth_activity
    = #(switches toward truth) + #(switches away from truth)
```

The net current telescopes under one-focal updates, so activity is required to measure path traffic/volatility.

Bootstrap whole episodes for cell-level current statistics.

---

# 15. Estimation conventions

Reuse:

```text
src/mas_cc/analysis/estimators.py
```

for direct-counting entropy/MI/CMI.

Use:

- **bits** throughout;
- canonical option ordering;
- pooled round rows within one cell;
- whole episodes as the bootstrap unit;
- no pooling across grid cells.

Do **not** compute MI independently from one short episode. The sample consists of repeated round records across episodes.

The new round clock reduces sample count by approximately a factor of `N` relative to the old event-level analysis. Therefore the agent must report support/sparsity diagnostics and must not hide failures of the full-state estimator.

For `N = 32` and `K >= 3`, expect the full-state `round_population_actuation_cmi` to require substantially more repetitions than the target-count projection.

---

# 16. Preferred actuation null: preserve the known policy

The soft controller gives the action-assignment probability explicitly.

For every observed round, log:

```text
p_k = P(U_k = ADVOCATE_Z | Y_k)
```

For the preferred round-level null, generate:

```text
U_k* ~ Bernoulli(p_k)
```

independently for each observed round while leaving:

```text
N_k
Y_k
N_{k+1}
```

fixed.

Then recompute the round actuation statistic.

This conditional-randomization null preserves:

```text
population -> sensor -> policy
```

but breaks the observed:

```text
action -> next population
```

association.

Keep the existing temporal/permutation null only as an optional compatibility diagnostic; do not make it the only null for the new game.

For local controlled-slot diagnostics, resample exactly `b` slot labels uniformly within each observed ADVOCATE round.

---

# 17. Analysis outputs

Create a dedicated post-hoc output directory, for example:

```text
hidden_bench_imitation_round_feedback_analysis/
```

Recommended files:

```text
round_information_estimates.csv
round_information_estimates.md
round_information_nulls.csv
round_support_diagnostics.csv
round_behavioral_summary.csv
micro_slot_diagnostics.csv
episode_currents.csv
cell_summaries.csv
```

The analysis must be rerunnable from persisted trajectories/round records without provider calls.

---

# 18. No-control cells

For:

```text
control.mechanism: none
```

do not fabricate `U_k = NO_OP` as if it were an experimentally randomized controller action.

No-control cells should report population/order/current metrics, but controller sensing/actuation information quantities should be null/absent exactly as in the current implementation philosophy.

---

# 19. Reasoning ON/OFF matched comparison

The principal 2x2 experiment remains:

```text
                    no feedback      round feedback
reasoning ON             A                B
reasoning OFF            C                D
```

Match where possible:

```text
task
initial vote vector
N
q
round count
seed
target
q_c
b
```

Only the focal response mechanism should differ between reasoning and classical controlled arms.

---

# 20. Tests — mandatory

Add focused tests for the scientific invariants.

## Controller timescale

- For `S` rounds, controller sensing is called exactly `S` times, not `S*N`.
- Exactly one `U_k` is sampled per round.
- No controller resensing occurs inside the microscopic loop.
- Every microscopic event in a round carries the same `round_controller_action`.

## Budget schedule

- `NO_OP` round -> exactly 0 controlled positions.
- `ADVOCATE_Z` round -> exactly `b` unique controlled positions.
- Controlled positions are sampled before the first microscopic update.
- Fixed seed reproduces the same controlled-position set.
- All size-`b` schedules are generated by uniform sampling without replacement.

## Social slots

- Ordinary update -> exactly `q` ordinary peer slots.
- Controlled update -> exactly `q-1` ordinary peer slots plus 1 controller slot.
- `q = 1` controlled update -> 0 ordinary peer slots + 1 controller slot.
- Only the focal committed vote may change.

## Sensor

- Exactly `q_c` distinct sensor agents are sampled from all `N`.
- Sensor sample may overlap focal/peer identities because sensing occurs at the round boundary independently of later microscopic draws.
- Controller never receives hidden evidence or private histories.

## Round boundaries

- `occupation_counts_before` of round `k` equals the microscopic state before `r=0`.
- `occupation_counts_after` equals the microscopic state after `r=N-1`.
- Round `k+1` pre-state equals round `k` post-state.

## Reasoning/classical

- Classical mode performs zero provider calls.
- Reasoning mode uses the same controller schedule semantics.
- New classical interface receives actual `q`-peer opinions/context.
- Existing `hidden_bench_imitation` tests remain unchanged and pass.

## Information fixtures

- Constant round action -> `H(U)=0`.
- 50/50 binary round action fixture -> `H(U)≈1` bit.
- Independent `U_k` and `N_{k+1}` conditional on `N_k` -> actuation CMI near zero.
- Deterministic action-dependent next-state fixture -> positive round actuation CMI.
- Policy-resampling null collapses a synthetic action-outcome signal.
- Bootstrap resamples episode IDs, not individual round rows.
- Target-count encoding is stable for 3- and 4-option tasks.
- No-control cells do not emit fabricated controller MI.

---

# 21. Backward compatibility

This task must not alter the scientific meaning of:

```text
hidden_bench_imitation
threshold_target
soft_target
existing event-level information metrics
existing reports
```

The old event-level controller and its results remain a separate experimental object.

Do not rename old metrics to the new round names. Do not reinterpret old `population_actuation_cmi` as `round_population_actuation_cmi`.

---

# 22. Suggested implementation order

1. Create/register the new game package and config type.
2. Reuse HiddenBench task/population initialization.
3. Implement round loop and round-boundary state.
4. Add round-level controller hook/policy.
5. Implement exact preallocated `b`-position schedule.
6. Reuse/extend microscopic focal+`q` peer update logic.
7. Implement reasoning controlled-slot replacement.
8. Add pluggable classical controlled-slot interface.
9. Add round record persistence.
10. Add `round_` analysis adapter and metrics.
11. Add policy-resampling null and support diagnostics.
12. Add episode current/activity aggregation.
13. Add tests.
14. Add one small executable smoke config for reasoning and one provider-free classical config.
15. Add one matched 2x2 grid config.

Do not implement InfoNCE in this task.

---

# 23. Deliverables

Finish with:

```text
1. new game package:
   src/mas_cc/games/hidden_bench/imitation_round_feedback/

2. registry/config integration

3. new round-level control mechanism:
   round_soft_target_budgeted

4. round-level persisted record

5. round-level information/entropy/signed-response analysis

6. local controlled-slot diagnostics

7. current + activity outputs

8. tests

9. smoke configs

10. matched reasoning/classical control grid

11. concise handoff listing:
    - files changed
    - exact config fields
    - exact metric names
    - analysis command
    - preflight/run commands
    - known limitations
```

---

# 24. Scientific interpretation to preserve

The new game should support the following causal decomposition:

```text
round-boundary population
        |
        v
finite sensing (q_c)
        |
        v
one stochastic controller decision U_k
        |
        v
fixed actuation budget b
        |
        v
N fast microscopic social updates of size q
        |
        v
next round-boundary population
```

The main scientific quantities are therefore:

```text
sensing:       I(N_k ; Y_k)

round control: I(U_k ; N_{k+1} | N_k)

local slot:    controlled-vs-ordinary focal response inside ADVOCATE rounds

trajectory:    truth current + truth activity + final order/truth alignment
```

The purpose of reasoning ON vs OFF is to test whether semantic reasoning changes the relationship between sensing, control information, local response, and macroscopic trajectory reliability.

Microscopic reversibility may be studied as a special property of a classical reference kernel, but it is **not** a required design constraint for this game.
