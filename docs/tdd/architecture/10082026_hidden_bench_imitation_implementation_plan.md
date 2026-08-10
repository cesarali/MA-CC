# HiddenBench Imitation Game — Monday/Tuesday Implementation Plan

## Purpose

Implement a new HiddenBench game called **`hidden_bench_imitation`** under:

```text
src/mas_cc/games/hidden_bench/imitation/
```

Do **not** remove or repurpose the existing `hidden_bench_vanilla` or `hidden_bench_naming` games. The new game is the scientific object we want to use for the current control/feedback study.

The game should combine two already-established objects:

1. **HiddenBench / Hidden Profile** supplies the semantic reasoning problem: shared evidence, distributed private evidence, a finite set of answer options, and a known correct answer.
2. **Irisarri et al. social imitation dynamics** supplies the dynamical skeleton: a population occupies a finite set of opinions, and elementary dynamics are one-agent, one-step opinion changes. For more than two opinions the mesoscopic state is an occupation vector, and a jump from opinion `A` to `B` changes it by `n -> n - e_A + e_B`.

The central experimental switch must be:

```text
reasoning = on   -> LLM-mediated opinion update over HiddenBench evidence/history
reasoning = off  -> provider-free classical multi-opinion imitation dynamics
```

The state space, task, answer options, population size, observables, controller interface, and event logging should remain as matched as possible between the two modes. The scientific question is then: **what changes in the controlled population dynamics when semantic reasoning is present?**

This document is an implementation directive for Monday and Tuesday. Do not expand the scope into InfoNCE, a full stochastic-thermodynamic derivation, or a large benchmark study yet.

---

## 1. Scientific contract to preserve

For task `r`, let the answer alphabet be

```text
A_r = {a_1, ..., a_K}
```

where `K` must be read from the task. Do **not** hard-code three choices. The current HiddenBench source has 65 tasks: most have 3 options, but 6 tasks have 4 options.

Each population agent `i` has:

```text
E_i = shared evidence + assigned hidden evidence
X_i(t) in A_r                # current committed opinion / vote
H_i(t)                       # private semantic interaction history, reasoning mode only
```

The microscopic population state is:

```text
X_t = (X_1(t), ..., X_N(t))
```

The mesoscopic occupation state is:

```text
n_t = (n_1(t), ..., n_K(t))
```

with

```text
n_a(t) = number of agents whose current opinion is a
sum_a n_a(t) = N
p_a(t) = n_a(t) / N
```

An elementary state change must affect **one focal agent only**:

```text
X_i(t) = a  ->  X_i(t+1) = b
```

so that:

```text
n_{t+1} = n_t - e_a + e_b
```

This one-agent jump geometry is important because it is the bridge to the classical imitation model.

---

## 2. New game location and reuse of existing HiddenBench code

Create:

```text
src/mas_cc/games/hidden_bench/imitation/
```

Suggested internal organization:

```text
imitation/
├── __init__.py
├── game.py                  # Game implementation and orchestration of one event
├── state.py                 # game-specific state/event dataclasses if useful
├── prompts.py               # reasoning-mode prompts only
├── classical.py             # provider-free imitation kernel
├── controller.py            # game-side adapter for stochastic feedback control
├── metrics.py               # imitation-specific metrics / adapters
└── README.md                # concise description of the implemented protocol
```

Reuse rather than duplicate the existing HiddenBench infrastructure where possible:

- task loading and canonical task objects;
- `profile: hidden | full` semantics;
- assignment schemes (`bijective`, `exact_replication`, etc.);
- shared/private evidence construction;
- vote parsing and validation;
- privacy checks;
- existing generic choice metrics;
- existing HiddenBench-specific disclosure utilities where applicable;
- experiment/grid/config/planning/recording abstractions.

The current completed scalable data that can be used immediately is `exact_replication`, including the existing `N_32.json`. Paraphrased replication and factorized evidence are not yet frozen; they must **not** become dependencies of this game implementation.

---

## 3. Configuration surface

Add a normal game config for `hidden_bench_imitation`. Keep names consistent with the existing config system.

At minimum support:

```yaml
game:
  type: hidden_bench_imitation
  population_size: 4
  horizon: 20                  # maximum elementary focal-agent update steps
  options:
    task_set: vanilla           # vanilla | expanded
    task_id: null
    profile: hidden             # hidden | full
    assignment_scheme: bijective

    dynamics_mode: reasoning    # reasoning | classical

    pairing: uniform_two_distinct
    messages_per_agent: 1       # reasoning mode only
    memory_size: 0              # 0 = unbounded, if current conventions use this
    allow_relay: true
    stop_on_consensus: false    # default false for dynamics/control experiments

    initialization:
      mode: local_vote          # reasoning mode default
      initial_votes: null       # explicit matched initial state when provided
      initial_distribution: null

    classical:
      kernel: irisarri_multi_opinion
      # kinetic parameters / reaction specification live here
      # choose defaults that are explicit and documented, never hidden

    controller:
      enabled: false
      target: correct           # correct | explicit option label/index
      sensor_sample_size: 1
      policy: threshold_target  # pilot policy; pluggable
      threshold: 0.5
```

Use the repository's existing `control:` abstraction/config wherever possible rather than creating an unrelated second control system. If the existing `Control.override()` interface is insufficient for measurement + message-level intervention, make the **smallest backward-compatible extension** required. Existing games and `ForcedActionControl` must continue to work unchanged.

The exact API shape of the extension can be chosen after inspecting the current runtime, but the scientific semantics below are mandatory.

---

## 4. Initialization

### 4.1 Reasoning mode

Before any pair interaction, every agent should make a private local-information vote using only its own HiddenBench observation:

```text
X_i(0) ~ pi_LLM(. | E_i)
```

This is a genuine pre-interaction state. It is preferable to defining an agent's initial state only when that agent happens to be sampled for the first time.

Record all initial votes.

### 4.2 Classical mode

Classical mode must be capable of running with **zero provider / LLM calls**.

Support explicit initial votes:

```yaml
initialization:
  initial_votes: [...]
```

This is the preferred mechanism for matched `reasoning on` vs `reasoning off` comparisons: take a fixed population state and evolve it under the two different kernels.

Also support a provider-free fallback such as an explicit `initial_distribution` if needed for synthetic/classical-only tests.

Do not silently call an LLM during `classical` initialization.

### 4.3 Matched-run support

Make it easy to export/reuse a realized initial vote vector. The goal is to be able to compare:

```text
same task
same evidence allocation
same N
same X_0
same controller target/policy
reasoning dynamics vs classical dynamics
```

Random seeds and initial states must be recorded.

---

## 5. Elementary interaction in reasoning mode

One interaction is an ordered focal-source event.

1. Sample focal population agent `i_t`.
2. Select a normal peer `j_t != i_t`, unless the controller substitutes a control interaction.
3. If it is a normal peer interaction, allow a small bounded private exchange using the existing HiddenBench/naming memory and prompt machinery where possible.
4. Update conversational memories according to the implemented exchange.
5. **Only the focal agent commits a new vote** at the end of the event.
6. All non-focal committed opinions remain unchanged.

A minimal reasoning event may use one message from each population participant followed by one focal vote. Reuse existing abstractions rather than hard-coding provider calls in the game.

The focal update should have conceptual form:

```text
P_LLM(X_i(t+1) | X_i(t), E_i, H_i(t), current interaction)
```

The action alphabet is exactly the current HiddenBench task's answer-option set.

No extra reward for correctness should be injected into the prompt merely to make the dynamics move toward truth. Truth must remain an observable, not a reward artifact.

---

## 6. Classical mode: reasoning OFF

Classical mode is not a mock LLM. It is a real provider-free stochastic imitation process over the same finite opinion alphabet.

Implement a classical kernel based on the **multi-opinion extension of Irisarri et al.** For `M > 2`, their mesoscopic state is the occupation vector `n`, and an elementary `A -> B` update is a one-step jump:

```text
n -> n - e_A + e_B
```

For a reaction `r` connecting two opinions, the paper gives reversible rates of the form:

```text
W_r(n - e_A + e_B, n) = h_r * n_A * g_r(n)
W_r(n, n - e_A + e_B) = a_r * (n_B + 1) * g_r(n)
```

Implement the classical kernel so that:

- it supports arbitrary `K = len(task.options)`;
- all possible one-agent opinion jumps are explicit;
- forward and reverse reaction channels have nonzero support when configured as such;
- kinetic parameters and the interaction factor `g_r` are explicit config, not hidden constants;
- the provider is never called;
- every sampled classical jump is logged with source opinion, destination opinion, reaction/channel identifier, transition weight/rate, and current occupation vector.

If the implementation uses the embedded jump chain rather than exact continuous-time Gillespie holding times, name/document that clearly. If adding a stored `physical_time` via Gillespie is easy and isolated, it is welcome, but it must not delay the core game. The first requirement is that the classical mode is a transparent multi-opinion one-step stochastic process that can be compared to the reasoning mode.

Do **not** claim in code/docs that the LLM mode obeys microscopic reversibility. That is a later empirical/theoretical question.

---

## 7. Macroscopic observables / order parameters

The game should emit the standard per-option population shares plus the following imitation-specific observables at every elementary event.

Let `K = len(options)` and `Y*` be the correct option.

### 7.1 Truth-aligned order parameter

```text
p_truth(t) = fraction of population currently voting for Y*

m_truth(t) = (K * p_truth(t) - 1) / (K - 1)
```

Properties:

```text
uniform population across K options -> m_truth = 0
truthful consensus                 -> m_truth = 1
zero support for truth             -> m_truth = -1/(K-1)
```

### 7.2 Controller-aligned order parameter

Let `Z` be the controller target.

```text
p_ctrl(t) = fraction of population currently voting for Z

m_ctrl(t) = (K * p_ctrl(t) - 1) / (K - 1)
```

If control is disabled, this metric can be absent/NA unless a target is still supplied for analysis.

Keep `m_truth` and `m_ctrl` distinct. In an adversarial-control experiment `Z != Y*`, the controller can become more successful while truth alignment becomes worse.

### 7.3 Generic collective order

```text
m_order(t) = (K * max_a p_a(t) - 1) / (K - 1)
```

This measures ordering/consensus regardless of which option dominates.

### 7.4 Normalized vote entropy

```text
H_vote(t) = - sum_a p_a(t) log p_a(t) / log(K)
```

Use the usual convention `0 log 0 = 0`.

### 7.5 Evidence diagnostics

Preserve/reuse where appropriate:

- `unshared_disclosure_rate`;
- `disclosure_reach` for dyadic diffusion;
- per-option population share;
- current action per agent.

Do not add a large new metric zoo for v1.

---

## 8. Stochastic feedback controller

The controller must be implemented as a **measurement + feedback loop**, not as an independent Bernoulli coin whose only purpose is to create entropy.

The abstract architecture is:

```text
X_t  ->  Y_t  ->  U_t  ->  X_{t+1}
```

where:

```text
X_t  = population state
Y_t  = stochastic / partial controller observation of the population
U_t  = control action chosen from the observation/history
X_t+1 = post-interaction population state
```

This is the structure required for later analysis of both information directions:

```text
system -> controller : X -> Y
controller -> system : U -> X
```

### 8.1 Measurement channel `X -> Y`

For the pilot, use a natural finite-population measurement rather than artificial Gaussian noise.

At each elementary event, sample `q_c = sensor_sample_size` population agents **without replacement** and observe only their current committed opinions.

Represent the controller observation as both:

```text
sampled agent IDs        # logged for audit, not necessarily exposed to policy
sampled opinion counts   # policy input
```

Given occupation vector `n_t`, this is a multivariate-hypergeometric observation channel. The randomness therefore comes from partial population sampling.

The controller must not see:

- hidden facts it was not granted;
- task rationale;
- ground-truth explanation;
- full private memories;
- full population state, unless a later explicit sensor condition enables this.

For the initial pilot, the controller should use only the sampled opinion-count vector.

### 8.2 Pilot control action `U_t`

Use a fixed target `Z` for an episode. The first action space can be:

```text
U_t in {NO_OP, ADVOCATE_Z}
```

The key point is that `U_t` is produced from `Y_t`, not drawn as an unrelated Bernoulli variable.

Provide a simple deterministic pilot policy such as:

```text
if sampled support for Z < threshold:
    U_t = ADVOCATE_Z
else:
    U_t = NO_OP
```

Because `Y_t` is stochastic, the resulting `U_t` is also stochastic across trajectories/states even if the policy itself is deterministic.

Keep the policy pluggable. `threshold_target` is a pilot policy, not a scientific commitment.

### 8.3 Local actuation

Control acts locally on the focal interaction.

If `U_t = NO_OP`:

```text
focal i_t interacts with ordinary peer j_t
```

If `U_t = ADVOCATE_Z`:

```text
focal i_t receives a controller interaction advocating Z
```

The controller must not directly overwrite `X_i(t+1)` in the reasoning mode. It supplies an intervention; the focal LLM can accept, reject, or reinterpret it.

The first controller should be **provider-free** and use a fixed, auditable message template. It should not inject task-specific hidden evidence. Example semantics:

```text
The external controller currently advocates option <Z>. Reconsider your current position before committing your next vote.
```

The exact wording belongs in a versioned prompt block.

In classical mode, the same control event should modify the classical transition mechanism in an explicit, documented way consistent with the target `Z` (for example as an external influence/reaction channel toward `Z`). The classical control effect must be provider-free and separately parameterized/logged. Do not bury the controller inside arbitrary prompt behavior.

### 8.4 Existing `Control` abstraction

The current repository `Control.override(...)` only forces an action. That is insufficient for this feedback design because we need to record a stochastic measurement and optionally inject a local interaction/message without forcing the resulting vote.

Extend the shared Control protocol only as much as needed, with these constraints:

- backward compatible with `NoneControl` and `ForcedActionControl`;
- no HiddenBench-specific assumptions in the generic protocol;
- still selectable from normal config and sweepable in experiment grids;
- controller observation/action metadata must reach the recorder;
- the game should not bypass the repository runtime/provider abstractions.

Prefer a generic optional interaction/message hook over special-casing `hidden_bench_imitation` inside the runtime.

---

## 9. Event logging required for Wednesday's information analysis

Do **not** implement the final transfer-entropy/InfoNCE estimator now. Instead make the trajectory logs sufficient to estimate it later without rerunning expensive LLM episodes.

For every elementary event record at least:

```text
episode_id
interaction_index
seed
task_id
K
N
dynamics_mode                     # reasoning | classical

population_state_before           # full vote vector or reconstructable representation
occupation_counts_before
population_shares_before

focal_agent_id
focal_opinion_before
peer_agent_id                     # null for controller interaction if appropriate
peer_opinion_before               # when ordinary peer exists

controller_enabled
controller_target
sensor_sample_size
sensor_agent_ids
sensor_observed_opinions
sensor_count_vector               # Y_t
controller_policy
controller_action                 # U_t = NO_OP / ADVOCATE_Z
controller_applied

focal_opinion_after
population_state_after
occupation_counts_after
population_shares_after

m_truth
m_ctrl
m_order
H_vote

disclosed evidence metadata / reach metadata where already available
```

For reasoning-mode events also retain the existing auditable conversation/prompt artifacts required to reconstruct what the focal agent saw.

For classical events also retain:

```text
classical_reaction_id
classical_source_opinion
classical_destination_opinion
classical_transition_rate_or_weight
classical_total_rate_or_normalizer
physical_time_increment            # if Gillespie is implemented
```

The point is to support later discrete estimates such as:

```text
system -> controller:
I(X_t ; Y_t | relevant controller history)

controller -> system:
I(U_t ; X_{t+1} | X_t)
```

and later history-aware variants, without changing the game again.

The repository already has direct-counting MI/conditional-MI estimators, bootstrap CIs, permutation/circular-shift nulls, and label-swap checks under `analysis/`; do not replace them now. Wednesday's work can adapt them to the new event table.

---

## 10. Time convention

For raw logs, `interaction_index = t` is sufficient.

Also expose normalized population time:

```text
tau = t / N
```

so later plots can compare different population sizes on an approximately per-agent interaction scale.

Do not stop early by consensus in the default dynamics/control experiment. Equal-horizon trajectories are more useful for the first information analysis.

---

## 11. Population sizes for the first implementation

The code must support arbitrary valid `N`, but Monday/Tuesday only needs smoke tests at small scale plus one scaled case.

Recommended implementation checks:

```text
N = 4   # canonical HiddenBench-style scale
N = 5   # if a compatible assignment scheme/task supports it
N = 32  # existing exact-replication population, smoke test only
```

Do not claim phase transitions or thermodynamic-limit behavior from `N = 4` or `5`. Small populations are being used first to answer a different question: **does the feedback/control-information signal exist and is the game behaving coherently?**

---

## 12. Tests / acceptance criteria

Add focused tests under the existing test layout. At minimum, acceptance requires all of the following.

### Game and data

- [ ] `hidden_bench_imitation` is registered and runnable from the standard CLI.
- [ ] Existing `hidden_bench_vanilla` and `hidden_bench_naming` tests remain green.
- [ ] Task option count is dynamic; at least one 3-option and one 4-option HiddenBench task pass end-to-end tests.
- [ ] Hidden information privacy invariants still hold.
- [ ] `bijective` works for canonical populations where valid.
- [ ] `exact_replication` works for a scaled population.

### Reasoning mode

- [ ] Initial local votes occur before pair interactions.
- [ ] An interaction updates only the focal committed opinion.
- [ ] Conversation/history is private according to the intended dyadic semantics.
- [ ] Provider call planning is exact and preflight can price the run.
- [ ] Controller messages never directly force the focal vote.

### Classical mode

- [ ] Zero LLM/provider calls.
- [ ] Supports arbitrary task `K`.
- [ ] Every state transition is a one-agent `A -> B` jump.
- [ ] Reverse reaction channels are available when configured.
- [ ] Fixed seed gives deterministic reproducibility.
- [ ] Explicit initial vote vectors work.

### Controller

- [ ] `controller.enabled: false` exactly reproduces the no-controller path.
- [ ] Measurement `Y_t` is sampled from the population and logged.
- [ ] The pilot policy produces `U_t` from `Y_t`.
- [ ] `U_t` can take at least `NO_OP` and `ADVOCATE_Z` in realistic trajectories.
- [ ] The controller does not receive hidden facts/rationale by accident.
- [ ] The generic Control extension is backward compatible.

### Metrics/logging

- [ ] `m_truth`, `m_ctrl`, `m_order`, and normalized vote entropy are correct for hand-checkable states.
- [ ] Event traces contain enough fields to reconstruct `(X_t, Y_t, U_t, X_{t+1})`.
- [ ] Existing experiment storage/aggregation and Comet conventions remain intact.

---

## 13. Minimal smoke experiment after implementation

Do not launch a broad benchmark yet. Run one or two validated tasks with a cheap/mock provider first, then one real-provider pilot if the implementation is sound.

Suggested pilot grid:

```text
dynamics_mode in {classical, reasoning}
controller.enabled in {false, true}
N in {4, 5}                      # where task/allocation permits
controller.target in {correct, one explicit wrong option}
```

Keep the controller sensor size and threshold fixed in the first pilot.

The first plots/tables should answer only:

1. Does the uncontrolled reasoning game produce sensible opinion trajectories?
2. Does classical mode produce the expected one-step imitation trajectories with no provider calls?
3. Does control visibly change `m_ctrl`, `m_truth`, `m_order`, or transition frequencies?
4. Do `Y_t` and `U_t` vary enough across repeated trajectories to make information estimation plausible?
5. Are the effects qualitatively different with reasoning ON vs OFF?

If these answers are sensible, the game is ready for Wednesday's discrete information analysis.

---

## 14. Explicitly out of scope for Monday/Tuesday

Do **not** spend implementation time on the following yet:

- training an InfoNCE estimator;
- embedding conversation histories;
- claiming a thermodynamic efficiency;
- deriving a new second law;
- proving microscopic reversibility for LLM transitions;
- large `N` finite-size scaling;
- broad multi-model evaluation;
- finishing paraphrased/factorized HiddenBench preprocessing;
- inventing additional controller policies beyond the minimal pluggable pilot baseline;
- replacing the existing analysis stack.

The immediate goal is to obtain a clean, auditable dynamical game and a valid feedback trajectory:

```text
X_t -> Y_t -> U_t -> X_{t+1}
```

under both:

```text
reasoning ON  (LLM-mediated HiddenBench update)
reasoning OFF (classical multi-opinion imitation update)
```

---

## 15. Definition of done for today

Monday + Tuesday can be considered complete when:

1. `hidden_bench_imitation` exists as a normal registered game;
2. the state/transition semantics above are documented and tested;
3. both reasoning and provider-free classical modes run;
4. the stochastic measurement/controller loop runs without forcing LLM actions;
5. the four order/information-relevant state variables are logged;
6. a trajectory file contains reconstructable `X_t, Y_t, U_t, X_{t+1}` for every interaction;
7. at least one small reasoning/classical/controller smoke comparison completes through the normal experiment stack.

After that, stop changing the game unless the pilot exposes a real conceptual bug. The next task is statistical: determine whether the two directional information signals are estimable from these trajectories before building any richer estimator.

---

## Source anchors used for this plan

- Li, Naito & Shirado, **Systematic Failures in Collective Reasoning under Distributed Information in Multi-Agent LLMs** (HIDDENBENCH): established distributed-evidence reasoning task with finite answer options and known ground truth.
- Irisarri, Trigal, Toral & Manzano, **Stochastic Thermodynamics of Social Imitation beyond Energetics**: one-step stochastic opinion dynamics and the multi-opinion occupation-vector extension.
- Horowitz & Sandberg, **Second-law-like inequalities with information and their interpretations**: measurement-feedback architecture motivating the separation of system-to-controller sensing and controller-to-system actuation information.
- Repository `architecture_overview.md`: existing `Game`, runtime, metrics, experiment, control, storage, and direct-counting information-analysis abstractions.
- Repository HiddenBench `README.md`: current `vanilla` and `naming` games, assignment schemes, metrics, dyadic memory semantics, and the known limitation of the current action-only `Control.override()` interface.
- `paraphrase_and_factorization_pipeline.md`: current scalable population preprocessing status; only exact replication is immediately complete/frozen for large-N use.
