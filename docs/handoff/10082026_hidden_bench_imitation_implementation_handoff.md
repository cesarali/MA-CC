# HiddenBench Imitation Game — Implementation Handoff

Date: 2026-08-10  
Implementation plan: [`10082026_hidden_bench_imitation_implementation_plan.md`](10082026_hidden_bench_imitation_implementation_plan.md)

## 1. Status

The `hidden_bench_imitation` project is implemented and runnable through the
normal MAS-CC experiment stack.

The game supports the two matched experimental conditions requested in the
plan:

```text
reasoning = on   -> LLM-mediated focal opinion updates
reasoning = off  -> provider-free classical multi-opinion jumps
```

Both modes use the same:

- HiddenBench task and evidence allocation;
- finite, task-defined answer alphabet;
- population representation;
- explicit initial vote vector;
- one-focal-agent event geometry;
- controller measurement and action interface;
- macroscopic observables;
- event and trajectory schema.

The main switching variable is `game.options.dynamics_mode`, whose values are
`reasoning` and `classical`.

## 2. Main implementation files

The new game package is:

```text
src/mas_cc/games/hidden_bench/imitation/
├── __init__.py
├── README.md
├── classical.py
├── controller.py
├── game.py
├── metrics.py
├── prompts.py
├── runtime.py
└── state.py
```

Responsibilities:

| File | Responsibility |
| --- | --- |
| `game.py` | Task loading, evidence assignment, initialization, private observations, reasoning requests, focal transitions, termination, and call planning. |
| `runtime.py` | Executes reasoning or classical episodes, controls provider access, runs sensing/feedback, and reports interactions to the recorder. |
| `classical.py` | Provider-free focal-conditioned embedded multi-opinion jump kernel. |
| `controller.py` | Hypergeometric population sensor, threshold policy, fixed advocacy message, and controller config adapter. |
| `state.py` | Validated imitation rules plus state and transition records. |
| `prompts.py` | Versioned local-vote, private-message, and focal-update prompts. |
| `metrics.py` | Order-parameter calculations and streaming metric registration. |
| `README.md` | Short operational and scientific protocol description. |

Shared repository integration changed in the following places:

| File | Change |
| --- | --- |
| `src/mas_cc/games/registry.py` | Registers the game and its three prompt families. |
| `src/mas_cc/experiments/orchestrator.py` | Dispatches imitation runs through their observer-aware runtime and supplies the configured control. |
| `src/mas_cc/control/protocols.py` | Adds the backward-compatible optional `interaction_signal()` control hook. |
| `src/mas_cc/control/registry.py` | Registers `threshold_target`. |
| `src/mas_cc/planning/game_preflight.py` | Permits and reports genuinely provider-free call plans with zero requests. |
| `src/mas_cc/observability/recorder.py` | Adds local-only rich `trajectory.jsonl` recording. |
| `src/mas_cc/config/loader.py` | Keeps old run-config paths replayable after configs moved under `runs/hidden_bench/` or `runs/old/`. |

## 3. Scientific state and event semantics

An agent's committed opinion is stored as `committed_action`. The microscopic
state is the population vote vector:

```text
X_t = (X_1(t), ..., X_N(t))
```

The mesoscopic state is recorded as an occupation count for every task option:

```text
n_t = (n_1(t), ..., n_K(t))
```

`K` is always derived from `len(task.possible_answers)`. It is not fixed at
three. End-to-end tests cover both three-option and four-option tasks.

Every dynamical event selects one focal agent. Only this agent's committed vote
may be replaced. All non-focal committed votes remain unchanged.

For a classical event, the destination must differ from the source, so every
event is an explicit one-agent jump:

```text
n -> n - e_A + e_B,  A != B
```

For a reasoning event, the focal LLM may retain its current vote. Thus at most
one vector component changes; the controller never writes that component
directly.

## 4. Initialization

### Reasoning mode

With the default initialization:

```yaml
initialization:
  mode: local_vote
  initial_votes: null
```

all agents make private local-information votes concurrently before event 1.
The realized state is stored as `initial_state`, and the vote vector is exposed
as `initial_votes` for reuse in a matched classical run.

Supplying `initial_votes` in reasoning mode skips provider-backed local
initialization and starts directly from that matched state.

### Classical mode

Classical initialization never calls a provider. It supports:

- an explicit `initial_votes` vector; or
- a provider-free weighted `initial_distribution` mapping.

If neither is supplied in classical mode, a seeded uniform distribution over
the task options is used.

The episode seed and realized initial vote vector are recorded.

## 5. Reasoning dynamics

A normal reasoning event performs:

1. Uniformly sample an ordered focal/peer pair of distinct agents.
2. Run `messages_per_agent` private exchange steps. Both participants speak
   from the same frozen pre-message state.
3. Construct a focal vote prompt from the focal agent's own evidence, bounded
   private history, current vote, and current private exchange.
4. Ask only the focal agent for the new committed vote.
5. Update private memories for the participants and the focal committed vote.

The prompt does not reward correctness. Ground truth is retained as an
observable only.

Private-history behavior:

- an agent sees only interactions it participated in;
- `memory_size: 0` means unbounded memory;
- a positive `memory_size` exposes only that many most recent entries;
- `allow_relay` is a prompt instruction, matching the existing HiddenBench
  naming-game convention;
- controller messages contain no hidden evidence or rationale.

Prompt families:

```text
hidden_bench_imitation_initial
hidden_bench_imitation_message
hidden_bench_imitation_update
```

## 6. Classical dynamics

Classical mode is not a mock LLM. The runtime does not enter the provider at
all, including during initialization.

The v1 kernel is documented as a:

```text
focal-conditioned embedded jump chain
```

It uses event index rather than Gillespie physical time. Accordingly:

```text
physical_time_increment: null
```

For every unordered option pair, the code exposes forward and reverse reaction
channels. Given a focal source option and a candidate destination, the base
weight uses:

```text
rate_constant * n_source * g(n)
```

The default interaction factor is:

```text
g(n) = (n_destination + interaction_offset) / N
```

The following parameters are explicit and recorded:

```yaml
classical:
  kernel: irisarri_multi_opinion
  forward_rate: 1.0
  reverse_rate: 1.0
  interaction_factor: destination_count_plus_offset
  interaction_offset: 1.0
  control_strength: 2.0
```

`interaction_factor: constant` is also available. Forward and reverse rates
must be positive in v1, preserving support in both directions.

Each classical trajectory event logs all candidate destination channels, the
selected reaction ID, base and controller weights, selected transition weight,
and the local normalization total.

This implementation does not claim that reasoning-mode transitions obey
microscopic reversibility.

## 7. Feedback controller

The shared `Control` abstraction still supports action forcing through
`override()`. Existing `NoneControl` and `ForcedActionControl` behavior is
unchanged.

The new optional hook is:

```python
Control.interaction_signal(...)
```

It returns an `InteractionControlSignal` containing a measured observation,
control action, optional local message, target, and metadata. Controls that do
not implement the hook inherit an inert default.

The pilot mechanism is selected with:

```yaml
control:
  mechanism: threshold_target
  options:
    target: correct
    sensor_sample_size: 1
    policy: threshold_target
    threshold: 0.5
```

For compatibility with the implementation plan, the nested
`game.options.controller` form is also recognized.

Controller targets may be:

- `correct`;
- an exact option label; or
- a non-negative zero-based option index.

### Measurement: `X_t -> Y_t`

At every event, the controller samples `sensor_sample_size` agents without
replacement. It receives only sampled opinions and their count vector.

The policy does not receive:

- hidden facts;
- task rationale;
- private memories;
- the full population state.

Sampled agent IDs are logged for audit but are not used by the policy.

### Policy: `Y_t -> U_t`

The deterministic pilot policy is:

```text
if sampled target support < threshold:
    ADVOCATE_Z
else:
    NO_OP
```

Sensor sampling makes the resulting action stochastic across trajectories even
though the threshold rule itself is deterministic.

### Actuation: `U_t -> X_{t+1}`

In reasoning mode, `ADVOCATE_Z` substitutes a fixed, provider-free controller
message for the ordinary peer interaction. The focal LLM remains free to
accept or reject it.

In classical mode, advocacy adds the explicit configured `control_strength`
to the local transition weight toward the target. The contribution is logged
separately from the classical base weight.

## 8. Metrics

Every completed event emits per-option population shares and:

```text
p_truth = share on the correct option
m_truth = (K * p_truth - 1) / (K - 1)

p_ctrl = share on the controller target
m_ctrl = (K * p_ctrl - 1) / (K - 1)

m_order = (K * max_option_share - 1) / (K - 1)

H_vote = -sum(p_a * log(p_a)) / log(K)
```

The implementation keeps `m_truth` and `m_ctrl` separate, so adversarial target
success cannot be confused with truth alignment.

The metric shelf also includes:

- `population_action_share_per_option`;
- `agent_current_action`;
- `dominant_action_share`;
- `unshared_disclosure_rate`;
- `disclosure_reach`.

## 9. Trajectory recording

Two local trajectory surfaces are written during normal experiment runs.

### `events.jsonl`

Each `imitation_transition` row includes the information-analysis tuple and
macroscopic state:

```text
X_t -> Y_t -> U_t -> X_{t+1}
```

Important fields include:

```text
episode_id
interaction_index
tau
seed
task_id
K
N
dynamics_mode

population_state_before
occupation_counts_before
population_shares_before

focal_agent_id
focal_opinion_before
peer_agent_id
peer_opinion_before

controller_enabled
controller_target
sensor_sample_size
sensor_agent_ids
sensor_observed_opinions
sensor_count_vector
controller_policy
controller_action
controller_applied

focal_opinion_after
population_state_after
occupation_counts_after
population_shares_after

m_truth
m_ctrl
m_order
H_vote

disclosed_hidden_facts
unshared_disclosure_rate
disclosure_reach
```

Classical events additionally include:

```text
classical_reaction_id
classical_source_opinion
classical_destination_opinion
classical_transition_rate_or_weight
classical_total_rate_or_normalizer
classical_base_weight
classical_control_weight
classical_candidate_channels
physical_time_increment
time_convention
```

### `trajectory.jsonl`

This richer, local-only file includes the event plus reasoning decisions,
private observation records, prompt fingerprints, raw responses, and parsed
actions. It is intended to make the focal agent's effective input auditable.
It is never sent to Comet.

Streaming metrics remain in the existing `metrics/streaming.csv` format.

## 10. Provider planning and preflight

Reasoning-mode `call_plan()` includes:

- all local initialization votes unless explicit votes are supplied;
- both participants' private messages;
- one focal update per elementary event;
- configured validation retry bounds.

With no feedback substitution, the provider request count is exact. When
feedback may replace a peer exchange, private-exchange demand is documented and
priced conservatively.

Classical-mode call plans declare zero provider-backed requests. Shared
preflight now treats this as a valid provider-free game rather than rejecting
it. It returns:

```text
provider requests: 0 / 0 / 0
launch status: permitted
pricing status: provider_free
```

## 11. Shipped smoke configurations

### Provider-free controlled classical run

```text
configs/runs/hidden_bench/hidden_bench_imitation_classical.yaml
```

This uses an explicit initial vote vector, 20 classical jumps, and the
`threshold_target` controller.

Run it with:

```bash
conda run -n MA-CC mas-cc experiment preflight \
  --config configs/runs/hidden_bench/hidden_bench_imitation_classical.yaml

conda run -n MA-CC mas-cc experiment run \
  --config configs/runs/hidden_bench/hidden_bench_imitation_classical.yaml
```

### Mock reasoning run

```text
configs/runs/hidden_bench/hidden_bench_imitation_reasoning_mock.yaml
```

This performs four local initial votes and four focal updates using the mock
provider. `messages_per_agent` is zero in this cheap smoke configuration; set
it to one for the minimal private exchange described in the plan.

Run it with:

```bash
conda run -n MA-CC mas-cc experiment run \
  --config configs/runs/hidden_bench/hidden_bench_imitation_reasoning_mock.yaml
```

## 12. Tests and verification

The focused acceptance tests are in:

```text
tests/mas_cc/test_hidden_bench_imitation.py
```

They cover:

- registry and config loading;
- hand-checkable order parameters;
- zero provider calls in classical mode;
- deterministic fixed-seed trajectories;
- one-focal-agent classical jumps;
- explicit initial vote vectors;
- dynamic four-option tasks;
- N=32 exact replication initialization;
- pre-interaction reasoning votes;
- focal-only reasoning updates;
- exact uncontrolled reasoning call planning;
- hidden-information privacy;
- stochastic sensor measurements;
- both controller actions in a realistic trajectory;
- complete feedback tuple logging;
- controller advocacy not forcing the reasoning vote;
- `NoneControl` reproducing the uncontrolled path.

Verification completed during implementation:

```text
new imitation acceptance tests:                 10 passed
relevant HiddenBench/control/experiment suite: 96 passed
full repository suite:                         571 passed, 3 failed
```

The three remaining full-suite failures are existing/stale checks outside the
new game:

1. a registry test expects a tuple that already omits the pre-existing
   `hidden_bench_vanilla` and `hidden_bench_naming` registrations, and now also
   omits `hidden_bench_imitation`;
2. two Phase 6 inspection tests fail the pre-existing frozen naming-prompt wire
   parity check.

The new implementation and all directly affected regression tests are green.

Useful verification commands:

```bash
conda run -n MA-CC python -m pytest -q \
  tests/mas_cc/test_hidden_bench_imitation.py

conda run -n MA-CC python -m pytest -q \
  tests/mas_cc/test_hidden_bench_imitation.py \
  tests/mas_cc/test_hidden_bench_games.py \
  tests/mas_cc/test_hidden_bench_grid.py \
  tests/mas_cc/test_control.py \
  tests/mas_cc/test_experiments.py \
  tests/mas_cc/test_metrics.py
```

## 13. Intentional limitations and next work

The following remain deliberately out of scope, matching the plan:

- no InfoNCE model;
- no transfer-entropy estimator inside the game;
- no stochastic-thermodynamic efficiency claim;
- no claim of microscopic reversibility for LLM updates;
- no Gillespie holding-time simulation;
- no broad model or benchmark sweep;
- no dependency on unfinished paraphrased or factorized evidence datasets;
- no additional controller policy zoo.

Recommended next steps:

1. Read `trajectory.jsonl` into a discrete event table.
2. Verify empirical variation in `Y_t` and `U_t` over repeated seeds.
3. Adapt the existing direct-counting MI/conditional-MI estimators to estimate
   the two directions separately:

   ```text
   system -> controller: I(X_t ; Y_t | controller history)
   controller -> system: I(U_t ; X_{t+1} | X_t)
   ```

4. Run matched reasoning/classical episodes with identical task, allocation,
   population, seed policy, controller target, and explicit `initial_votes`.
5. Only after the mock pilot is coherent, substitute a real provider and run a
   very small validated pilot.

## 14. Practical matched-run recipe

To compare the two kernels from the same initial state:

1. Run reasoning initialization once or read `initial_votes` from a completed
   reasoning result/checkpoint.
2. Copy that exact vector into both configs:

   ```yaml
   initialization:
     mode: explicit
     initial_votes: [ ... ]
   ```

3. Keep the task, assignment scheme, population, controller options, event
   count, and execution seed fixed.
4. Change only:

   ```yaml
   game:
     options:
       dynamics_mode: reasoning  # or classical
   ```

5. Compare the event trajectories using `tau`, `m_truth`, `m_ctrl`, `m_order`,
   vote entropy, and transition frequencies.

This is the intended experimental use of the implementation.
