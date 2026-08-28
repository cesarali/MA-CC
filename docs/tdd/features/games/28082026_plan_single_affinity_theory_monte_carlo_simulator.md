# Implementation Plan: Monte Carlo Simulator for the Revised Single-Affinity Theory

## Scope

Implement a **provider-free Monte Carlo simulator of the revised finite-state theory** defined in:

```text
src/mas_cc/games/relational_reasoning/imitation_round_feedback/theory_revised.py
```

This simulator is the **only classical simulation requested in this task**.

Do **not** implement a second full-game q-voter baseline, do **not** add a classical agent backend to the relational LLM runtime, and do **not** introduce a second population model.

The intended scientific structure is:

```text
                         LLM experiment
                              |
                              | coarse-grained comparison in n
                              |
                              v
              single-affinity finite-compliance theory
                         /             \
                        /               \
                       v                 v
              exact finite-state      Monte Carlo
                  calculation          trajectories
```

The exact and Monte Carlo branches describe the **same mathematical model**. The Monte Carlo branch exists to:

1. provide a trajectory-level realization of the theory;
2. validate the exact matrix/closed-form implementation numerically;
3. enable simulation-based exploration of transient occupancies, fluctuations, currents, and efficiencies;
4. provide a clean classical reference without adding another conceptual baseline to the paper.

`theory_revised.py` remains the authoritative theory source.

---

# 1. Non-negotiable conceptual rule

The simulator must realize exactly

```text
n_k -> Y_k -> U_k -> n_(k+1)
```

as defined by `theory_revised.py`.

It is **not** a simulation of the full relational game.

Therefore its state and channels are:

```text
state:
    n_k in {0, ..., N}

sensor:
    Y_k

controller action:
    U_k in {0,1}

controlled microscopic kernel:
    K_{h,gamma}

one-cycle NoOp kernel:
    Q0 = I

one-cycle advocacy kernel:
    Q1 = K_{h,gamma}^b
```

It has **no**:

```text
LLM provider
prompts
language
agents with identities
peer ballots
semantic relations
fact sets K_i
relational task generator
ordinary q-voter social updates
natural-language reasoning
controller fact messages
```

Do not introduce fake agents or fake prompts merely to reuse the existing LLM runtime.

---

# 2. Authoritative theory semantics

The simulator must use the public API and definitions already implemented in `theory_revised.py`.

## Parameters

```python
TheoryParameters(
    N,
    q_c,
    b,
    beta,
    theta,
    h,
    gamma,
)
```

Interpretation:

| Parameter | Meaning |
|---|---|
| `N` | population size / maximum target count |
| `q_c` | number of agents sensed |
| `b` | controlled microscopic opportunities when advocacy is active |
| `beta` | feedback-policy gain |
| `theta` | feedback threshold |
| `h` | directional affinity of the controlled revision channel |
| `gamma` | kinetic compliance of a controlled opportunity |

The revised theory fixes the controlled layer to `q=1`; `q` must not become a simulation sweep parameter.

Do not add mutation, temperature, lapse probability, an opposing affinity, or any other population-response parameter.

---

# 3. Exact stochastic process to simulate

## 3.1 Sensor

At count state `n`, sample

```text
Y ~ S(. | n)
```

with the exact finite-population hypergeometric sensor law:

```text
S(y | n) = P(Y=y | n).
```

Use the semantics already encoded by:

```python
sensor_law(...)
sensor_kernel(...)
```

Do not replace sampling without replacement by a binomial approximation.

## 3.2 Feedback policy

Given the realized sensor observation `y`, draw

```text
U ~ Bernoulli(pi(1 | y))
```

with

```text
pi(1 | y) = sigmoid(beta * (theta - y/q_c)).
```

Use the same semantics as:

```python
policy_advocacy_vector(...)
advocacy_probability_curve(...)
```

The runtime draw must depend on **realized `y`**, not directly on the true population count `n`.

The marginalized curve

```text
a_n = P(U=1 | n)
```

is for theory and validation. It should not replace the explicit causal sequence `n -> Y -> U` in Monte Carlo trajectories.

## 3.3 Controlled microscopic opportunity

For one controlled opportunity at current count `n`, the theory defines

```text
P(n -> n+1) = gamma * (N-n)/N * sigma(h)
P(n -> n-1) = gamma * n/N * [1-sigma(h)]
P(n -> n)   = 1 - P(up) - P(down).
```

This is the kernel returned by:

```python
controlled_kernel(parameters)
```

Both directions must remain possible for finite `h` and positive `gamma`.

Do not replace this channel by deterministic advocacy and do not infer its behavior from the old q-voter code.

## 3.4 One feedback cycle

For each cycle `k`:

```text
given n_k

1. sample Y_k ~ S(. | n_k)
2. sample U_k ~ pi(. | Y_k)

3a. if U_k = 0:
        n_(k+1) = n_k

3b. if U_k = 1:
        starting from n_k,
        perform exactly b sequential draws from K_{h,gamma}
        and set the final count to n_(k+1)
```

This realizes

```text
Q0 = I
Q1 = K_{h,gamma}^b.
```

### Preferred simulation mode

The default Monte Carlo implementation should execute the `b` microscopic controlled opportunities sequentially rather than drawing the final state directly from `Q1`.

This gives an independent numerical check that

```text
Monte Carlo composition of K over b steps -> K^b
```

within sampling error.

A direct-`Q1` sampler may be added as an optional fast/debug mode, but it should not be the only implementation.

---

# 4. Relationship to the LLM experiment

The simulator is **not required to use the existing `Game` decision loop**.

The relational game has a rich state

```text
{X_i, K_i}_{i=1}^N,
```

while the revised theory deliberately studies only

```text
n_k = number of agents supporting the controller target.
```

Therefore:

- keep the LLM experiment code unchanged;
- do not add `dynamics_mode: classical` to the relational runtime for this task;
- keep `theory_revised.py` deterministic and authoritative;
- implement the simulator as a separate theory-level component;
- compare LLM results with theory only through the existing coarse target-count coordinate and derived observables.

This avoids creating an unnecessary third model.

---

# 5. Recommended module placement

Preferred structure:

```text
src/mas_cc/games/relational_reasoning/imitation_round_feedback/
├── theory_revised.py              # existing exact theory, authoritative
├── theory_simulation.py           # NEW Monte Carlo trajectory engine
├── theory_simulation_analysis.py  # optional, only if analysis becomes nontrivial
└── ...
```

If repository naming conventions strongly suggest another filename, adapt it, but keep the simulator beside `theory_revised.py`.

Avoid `classical.py` if that would be confused with the older q-voter reference.

Do not replace or branch the existing relational `runtime.py`.

---

# 6. Dependency direction

The dependency must be one-way:

```text
theory_revised.py
        |
        v
theory_simulation.py
```

Never make `theory_revised.py` depend on simulation code.

The simulator should import the theory's public objects rather than duplicate formulas. In particular, prefer using:

```python
TheoryParameters
SingleAffinityReference
single_affinity_reference
controlled_kernel
sensor_law
policy_advocacy_vector
```

or the assembled reference fields:

```text
reference.S
reference.pi1
reference.advocacy
reference.K
reference.Q0
reference.Q1
reference.chi
reference.T_pi
reference.eta_IR
reference.closed_loop_kernel
```

Do not rewrite the hypergeometric law, logistic policy, affinity kernel, susceptibility formula, Pinsker bound, system entropy, or thermodynamic identities inside the simulator.

---

# 7. Core data structures

Keep the trajectory API small and explicit.

A possible configuration object:

```python
@dataclass(frozen=True, slots=True)
class TheorySimulationConfig:
    parameters: TheoryParameters
    rounds: int
    episodes: int
    seed: int
    initial_condition: ...
    record_microsteps: bool = False
```

A possible cycle record:

```python
@dataclass(frozen=True, slots=True)
class TheoryCycleRecord:
    episode_id: int
    round_index: int
    n_before: int
    y: int
    action: int
    advocacy_probability_given_y: float
    advocacy_probability_given_n: float
    n_after: int
    current: int
```

Optional microscopic record:

```python
@dataclass(frozen=True, slots=True)
class ControlledMicrostepRecord:
    episode_id: int
    round_index: int
    controlled_step: int
    n_before: int
    n_after: int
    delta_n: int
```

The essential scientific trajectory is:

```text
(n_k, Y_k, U_k, n_(k+1)).
```

Do not make the simulation record mimic the relational-agent trajectory schema.

---

# 8. Initialization

Support explicit initial distributions over `n_0`.

At minimum:

## Fixed count

```yaml
initialization:
  type: fixed_count
  n0: 8
```

## Binomial ensemble

Use the existing helper:

```python
binomial_ensemble(N, x0)
```

for configurations such as:

```yaml
initialization:
  type: binomial
  x0: 0.33
```

## Explicit probability vector

```yaml
initialization:
  type: distribution
  probabilities: [...]
```

For each episode, sample `n_0 ~ p_0` using the simulation RNG.

Do not use LLM `local_vote` initialization inside this theory simulator.

If later the theory should be initialized from an empirical LLM `n_0` distribution, implement that as an adapter/analysis feature rather than as a dependency of the core simulator.

---

# 9. RNG and reproducibility

Reproducibility is mandatory.

Use an explicit NumPy generator or the repository's equivalent seeded abstraction:

```python
rng = np.random.default_rng(seed)
```

Do not use global `np.random` or `random` state.

If convenient, use deterministic child streams for:

```text
initialization
sensor draws
policy draws
controlled microsteps
```

At minimum, the same

```text
resolved config + seed
```

must reproduce the complete trajectory exactly.

Persist the relevant theory provenance already exposed by `theory_revised.py`:

```python
THEORY_REFERENCE
THEORY_SEMANTICS_VERSION
THEORY_API_VERSION
THEORY_MODULE
```

along with the parameter tuple, seed, episode index, and round index.

---

# 10. Simulation engine API

Implement small testable functions before CLI integration.

Suggested functions:

```python
simulate_controlled_opportunity(...)
simulate_cycle(...)
simulate_episode(...)
simulate_ensemble(...)
```

For example:

```python
def simulate_cycle(
    n: int,
    reference: SingleAffinityReference,
    rng: np.random.Generator,
    *,
    record_microsteps: bool = False,
) -> TheoryCycleRecord:
    ...
```

Keep sampling logic separate from estimation/analysis logic.

---

# 11. Preserve the explicit measurement path

This is an important correctness requirement.

Wrong:

```python
u = Bernoulli(reference.advocacy[n])
```

Correct:

```python
y = sample(reference.S[n])
u = Bernoulli(reference.pi1[y])
```

Although both produce the same marginalized `P(U|n)`, only the second retains the realized measurement variable needed for:

```text
I(n_k ; Y_k)
```

and the finite-time feedback path accounting.

The Monte Carlo simulator must explicitly realize:

```text
n -> Y -> U.
```

---

# 12. Trajectory-level current

For one sampled cycle define

```text
j_c,k = n_(k+1) - n_k.
```

Under the isolated controlled layer:

```text
U=0 -> Q0=I -> j_c,k=0,
```

so every displacement belongs to the controlled channel.

The ensemble mean must converge to:

```python
reference.current(p_k)
```

which equals the theory's mean controlled target-count current.

Keep the distinction:

```text
j_c,k    trajectory random variable
J_c,k    ensemble mean current.
```

The affinity-weighted contributions are correspondingly `h*j_c,k` and `h*J_c,k`.

---

# 13. Monte Carlo occupancy

For `E` episodes estimate at each finite cycle:

```text
p_hat_k(n) = count{episodes with n_k=n} / E.
```

The theory is nonstationary, so retain the full sequence:

```text
p_hat_0, p_hat_1, ..., p_hat_H.
```

Do not pool all cycles into one stationary histogram by default.

Compare against repeated exact propagation through:

```python
reference.propagate(...)
reference.self_occupancy(...)
```

or the equivalent exact finite-horizon helper.

---

# 14. Exact-vs-Monte-Carlo validation suite

The simulator is not complete until it is validated against the deterministic theory.

## 14.1 Sensor law

For selected `(N,n,q_c)`, estimate

```text
P_hat(Y=y | n)
```

and compare with:

```python
sensor_law(N, n, q_c)
```

## 14.2 Policy

Estimate both:

```text
P_hat(U=1 | Y=y)
P_hat(U=1 | n)
```

and compare with:

```python
reference.pi1[y]
reference.advocacy[n]
```

## 14.3 One controlled opportunity

From fixed `n`, repeatedly simulate one controlled opportunity and compare empirical transition probabilities with:

```python
reference.K[n]
```

Test `n=0` and `n=N` explicitly.

## 14.4 Advocacy kernel `Q1`

From fixed `n`, execute exactly `b` sequential controlled opportunities and estimate:

```text
Q1_hat(m | n).
```

Compare with:

```python
reference.Q1[n, :]
```

This validates `K` composed `b` times against `K^b`.

## 14.5 NoOp kernel `Q0`

Under `U=0`, assert exactly:

```text
n_(k+1) == n_k.
```

No tolerance is needed.

## 14.6 Closed-loop one-cycle kernel

For fixed `n`, simulate the complete sensor-policy-actuation cycle many times and compare:

```text
P_hat(n_(k+1)=m | n_k=n)
```

with:

```python
reference.closed_loop_kernel[n, :]
```

This validates the complete causal composition.

---

# 15. Susceptibility validation

For each state `n`, estimate:

```text
chi_hat(n/N)
 = E_hat[n_(k+1)/N | U=1, n_k=n]
 - E_hat[n_(k+1)/N | U=0, n_k=n].
```

Compare with:

```python
reference.chi[n]
```

and therefore indirectly with the closed form:

```text
chi_{h,gamma}(x)
 = [sigma(h)-x] * [1-(1-gamma/N)^b].
```

Do not duplicate the closed-form calculation inside the simulation code.

Test states on both sides of the zero-response set point:

```text
x* = sigma(h).
```

---

# 16. State-local action information `T_pi`

For each fixed state `n`, estimate from simulated cycles:

```text
I_hat(U ; n_(k+1) | n_k=n)
```

using the repository's existing discrete MI estimator if it can be reused cleanly.

Compare with:

```python
reference.T_pi[n]
```

in **bits**.

Use enough samples and suitable confidence/tolerance because the empirical MI estimator has finite-sample bias.

The exact deterministic curve remains authoritative.

Also verify:

```text
T_pi(n) <= h2(a_n).
```

---

# 17. Information-response efficiency

The exact state-local efficiency is already:

```python
reference.eta_IR
```

The Monte Carlo side should estimate:

```text
a_hat_n
chi_hat(n)
T_hat_pi(n)
```

and form the simulation diagnostic using the same definition.

For the finite horizon, follow the paper's aggregation rule:

> accumulate occupancy-weighted numerator and denominator first, then take the ratio.

Do **not** average state-local ratios.

For horizon `H`, compare the Monte Carlo result with the exact finite-horizon construction based on:

```text
sum_k sum_n p_k(n) B_IR(n)
sum_k sum_n p_k(n) T_pi(n).
```

If useful, report separately:

```text
MC occupancy + exact local theory functions
fully empirical MC estimate
```

but label them clearly.

---

# 18. Sensing information

At each finite cycle estimate:

```text
I_hat_sens,k = I(n_k ; Y_k)
```

in **nats** for thermodynamic comparison.

Compare with:

```python
sensing_information_nats(p_k, reference.S)
```

using the exact occupancy.

The simulator may additionally report bits for convenience, but never mix the units of:

```text
T_pi       bits
I_sens     nats in thermodynamic accounting
Sigma      nats
h J_c      nats
C_th       nats.
```

---

# 19. Finite-time thermodynamic validation

This is a major acceptance criterion.

For exact occupancy `p_k`, the theory already returns:

```python
reference.one_cycle(p_k)
```

with:

```text
J_c
I_sens_nats
delta_S_sys_nats
Sigma_nats
C_th_nats
eta_th.
```

The Monte Carlo ensemble should estimate the corresponding quantities from:

```text
p_hat_k
p_hat_(k+1)
J_hat_c
I_hat_sens.
```

Use the existing exact definition:

```python
system_entropy(...)
```

on empirical occupancy vectors rather than inventing a second entropy definition.

Then compute:

```text
Sigma_hat_identity
 = Delta S_hat_sys + h * J_hat_c + I_hat_sens
```

and:

```text
C_hat_th = h * J_hat_c + I_hat_sens.
```

For target-directed operation:

```text
eta_hat_th = h * J_hat_c / C_hat_th.
```

These estimates must converge to the exact `one_cycle` quantities as episode count increases.

---

# 20. Optional direct path-space irreversibility estimator

If feasible without excessive complexity, add a second Monte Carlo estimator of `Sigma` from sampled path log-likelihood ratios.

For each sampled coarse path element:

```text
(n, y, u, m)
```

evaluate the forward/reverse reference likelihood ratio once `p_k`, `p_(k+1)`, and the sensor marginal are known.

The average sampled quantity:

```text
log(P_F / P_R)
```

should converge to:

```python
CycleThermodynamics.Sigma_direct_KL_nats
```

and to the decomposition:

```text
Delta S_sys + h J_c + I_sens.
```

This is recommended as a second-stage validation, but should not block the core simulator if it requires substantial additional infrastructure.

---

# 21. Finite-horizon thermodynamics

For `H` cycles retain the full nonstationary occupancy sequence.

Estimate:

```text
sum_k J_c,k
sum_k I_sens,k
S_sys[p_H] - S_sys[p_0]
sum_k Sigma_k.
```

Compare with:

```python
finite_horizon_thermodynamics(
    reference,
    initial,
    rounds=H,
)
```

The simulation must reproduce the telescoping identity:

```text
S_sys[p_H] - S_sys[p_0]
+ h * sum_k J_c,k
+ sum_k I_sens,k
=
sum_k Sigma_k.
```

The horizon-level thermodynamic efficiency must be the ratio of accumulated terms:

```text
eta_th^(H)
 = h * sum_k J_c,k
   /
   [h * sum_k J_c,k + sum_k I_sens,k].
```

Do not use the arithmetic average of per-cycle efficiencies as the primary quantity.

---

# 22. Current fluctuations

The Monte Carlo simulator naturally gives current fluctuations not visible from mean trajectories alone.

For each episode define:

```text
J_episode = n_H - n_0.
```

Report at minimum:

```text
mean
variance
standard deviation
SNR^2 = mean^2 / variance
```

where defined.

Compare with:

```python
finite_horizon_current_moments(
    reference.closed_loop_kernel,
    initial_distribution,
    rounds=H,
)
```

For mixed initial counts/horizons, use the existing helper:

```python
finite_horizon_current_moments_for_episodes(...)
```

Do not create a new definition of current variance.

---

# 23. Parameter sweeps

The simulator should make it easy to explore:

```text
N
q_c
b
beta
theta
h
gamma
initial distribution
H
```

The principal scientific axes are:

```text
sensing:      q_c / N
actuation:    b / N
policy:       beta, theta
response:     h, gamma.
```

Do not add `q` as a sweep axis.

---

# 24. Calibration compatibility

The exact theory already supports:

```python
calibrate_affinity_compliance(...)
calibrate_affinity_compliance_from_counts(...)
```

with:

```text
gamma = p_plus + p_minus
h     = ln(p_plus / p_minus).
```

The simulation runner should accept `h` and `gamma` directly.

A convenience adapter may accept calibration counts and call the existing theory functions, but the Monte Carlo engine must not fit `h` or `gamma` itself.

Calibration and simulation remain separate operations.

---

# 25. Suggested configuration

Follow repository configuration conventions rather than inventing a large new framework.

A conceptual example:

```yaml
simulation:
  type: single_affinity_theory
  seed: 20260828
  episodes: 100000
  rounds: 10

theory:
  N: 24
  q_c: 12
  b: 18
  beta: 4.0
  theta: 0.5
  h: 2.0
  gamma: 0.35

initialization:
  type: binomial
  x0: 0.33

artifacts:
  record_cycles: true
  record_microsteps: false
```

If an existing analysis-job config abstraction already fits, use it.

Avoid changing the main `GameConfig` unless there is a strong repository-level reason.

---

# 26. CLI integration

Add the smallest clean CLI surface consistent with the existing package.

Possible forms:

```bash
mas-cc analysis relational-theory-simulate --config <config.yaml>
```

or:

```bash
mas-cc theory simulate --config <config.yaml>
```

Choose the form that best matches the repository after inspection.

Requirements:

```text
provider-free
no LLM configuration required
reproducible
cluster-friendly
explicit output directory
safe to rerun
must not mutate existing LLM experiment results
```

Do not overload the agent-game runtime if doing so requires pretending this is a relational agent simulation.

---

# 27. Output artifacts

Recommended minimal output layout:

```text
theory_simulation/
├── resolved_config.json
├── metadata.json
├── cycle_trajectories.csv          # optional for very large runs
├── occupancy_by_round.csv
├── state_local_validation.csv
├── horizon_summary.csv
└── validation_summary.json
```

## `cycle_trajectories`

Useful columns:

```text
episode
round
n_before
y
action
pi_advocate_given_y
a_n
n_after
current
seed/provenance
```

Do not store microscopic rows by default for very large simulations.

## `occupancy_by_round`

```text
round
n
count
probability
exact_probability
difference
```

## `state_local_validation`

```text
n
x
a_exact
a_mc
chi_exact
chi_mc
T_pi_exact_bits
T_pi_mc_bits
eta_IR_exact
eta_IR_mc
```

## `horizon_summary`

```text
H
J_exact
J_mc
Var_J_exact
Var_J_mc
I_sens_exact_nats
I_sens_mc_nats
Sigma_exact_nats
Sigma_mc_nats
eta_IR_exact
eta_IR_mc
eta_th_exact
eta_th_mc
```

Keep exact and Monte Carlo quantities side by side.

---

# 28. Memory and performance

The state space is tiny, so large episode counts should be practical.

Avoid retaining all trajectories in memory for large `E`.

Use streaming accumulators for:

```text
occupancies
sensor contingency counts
action contingency counts
transition counts
current moments
```

Allow trajectory persistence only when requested.

A useful default is:

```text
record_cycles = false
```

for high-sample validation/sweeps, while still retaining aggregate sufficient statistics.

Reuse the cached exact reference from:

```python
single_affinity_reference(parameters)
```

rather than recomputing matrices per episode.

---

# 29. Statistical uncertainty

Monte Carlo comparisons need uncertainty.

For transition probabilities, use appropriate multinomial/binomial standard errors or confidence intervals.

For episode-level observables such as current moments and horizon efficiencies, use whole-episode bootstrap where useful.

Do not bootstrap deterministic exact theory quantities.

Exact theory is the target; uncertainty belongs to Monte Carlo estimates.

---

# 30. Tests

Add a focused test module such as:

```text
tests/mas_cc/test_single_affinity_theory_simulation.py
```

Required checks:

```text
[ ] deterministic reproducibility for fixed seed
[ ] fixed-count initialization
[ ] distribution initialization
[ ] sensor samples respect the exact sensor support
[ ] policy uses realized y rather than true n
[ ] NoOp leaves n unchanged exactly
[ ] controlled microstep never leaves [0,N]
[ ] one-step frequencies match K
[ ] b-step frequencies match Q1
[ ] closed-loop frequencies match closed_loop_kernel
[ ] Monte Carlo chi converges to reference.chi
[ ] Monte Carlo T_pi converges to reference.T_pi
[ ] occupancy propagation converges to exact propagation
[ ] finite-horizon current mean/variance converge to exact moments
[ ] finite-time thermodynamic quantities converge to reference.one_cycle
[ ] finite-horizon quantities converge to finite_horizon_thermodynamics
[ ] bits and nats are never silently mixed
[ ] no provider imports/calls are required
```

Use moderate sample sizes in the default test suite and choose tolerances from expected sampling error rather than arbitrary exact-equality thresholds.

Large statistical regression tests may be marked separately if necessary.

---

# 31. Numerical edge cases

Explicitly test:

## `b = 0`

Then:

```text
Q1 = I
chi = 0
T_pi = 0
J_c = 0.
```

Efficiencies may be undefined according to the current theory semantics. Do not coerce undefined ratios to zero unless `theory_revised.py` already does so.

## `gamma = 0`

No controlled revision occurs:

```text
K = I
Q1 = I.
```

## Large positive or negative `h`

Reuse the numerically stable logistic implementation from the theory module.

## Boundary states

```text
n = 0
n = N
```

must never produce invalid counts.

## Very large `beta`

The policy may become nearly deterministic away from the threshold and local action information may collapse. Preserve this behavior; do not smooth it away.

---

# 32. Backward compatibility

This feature must be additive.

Do not change the semantics of:

```text
relational LLM runtime
existing experiment configs
completed result directories
existing analysis outputs
old theory module
theory_revised.py exact outputs
```

unless a separate verified bug requires its own fix.

Existing analysis must not start depending on Monte Carlo results.

The exact theory remains the primary/reference calculation.

---

# 33. Documentation

Add a short repository document, for example:

```text
docs/.../single_affinity_theory_simulation.md
```

It should explain:

1. what is simulated;
2. what is explicitly not simulated;
3. the causal cycle `n -> Y -> U -> n'`;
4. the role of `h` and `gamma`;
5. `Q0=I` and `Q1=K^b`;
6. initialization;
7. exact-vs-Monte-Carlo validation;
8. outputs;
9. example command.

Include an explicit statement such as:

> This simulator is a Monte Carlo realization of the coarse single-affinity finite-compliance reference theory. It is not a q-voter agent simulation and is not a provider-free reimplementation of the full relational reasoning game.

---

# 34. Recommended implementation sequence

## Phase 1 — Core trajectory sampler

Implement:

```text
config/data structures
initial-state sampling
sensor draw
policy draw
controlled microscopic draw
cycle simulation
episode simulation
ensemble simulation
```

Validate reproducibility and state boundaries.

## Phase 2 — Exact-kernel validation

Add tests for:

```text
S(y|n)
pi(1|y)
a_n
K
Q0
Q1
closed-loop kernel
```

Only proceed once these agree with `theory_revised.py`.

## Phase 3 — Scientific observables

Add Monte Carlo estimates for:

```text
occupancy
chi
T_pi
eta_IR
J_c
current variance
I_sens
Delta S_sys
Sigma via identity
C_th
eta_th
```

## Phase 4 — Finite-horizon aggregation

Implement the exact paper conventions:

```text
eta_IR^(H): ratio after accumulating occupancy-weighted terms
eta_th^(H): ratio after accumulating current and sensing information
```

Validate against the existing finite-horizon theory API.

## Phase 5 — CLI/config/artifacts

Wire the simulator into the smallest suitable CLI/config layer.

Do not modify the relational agent runtime.

## Phase 6 — Optional direct path-KL estimator

Add trajectory log-ratio validation of `Sigma` if useful.

---

# 35. Scientific acceptance criteria

The implementation is complete when all of the following are true.

## Model identity

```text
[ ] the simulator has state n only
[ ] it realizes n -> Y -> U -> n'
[ ] sensor draws are hypergeometric
[ ] policy draws use realized Y
[ ] NoOp is Q0=I
[ ] advocacy executes exactly b K_{h,gamma} opportunities
[ ] no ordinary q-voter dynamics are present
[ ] no LLM/agent/fact dynamics are present
```

## Theory agreement

```text
[ ] MC sensor law agrees with exact S
[ ] MC action frequencies agree with pi and a_n
[ ] MC one-step transitions agree with K
[ ] MC b-step transitions agree with Q1=K^b
[ ] MC closed-loop transition law agrees with Q_pi
[ ] MC susceptibility agrees with chi
[ ] MC action information agrees with T_pi
[ ] MC occupancy agrees with exact transient occupancy
[ ] MC finite-horizon current mean/variance agree with exact moments
[ ] MC sensing information agrees with exact I_sens
[ ] MC thermodynamic accounting converges to exact one-cycle values
[ ] MC finite-horizon eta_IR agrees with exact aggregation
[ ] MC finite-horizon eta_th agrees with exact aggregation
```

## Engineering

```text
[ ] exact theory remains deterministic and authoritative
[ ] simulation is provider-free
[ ] simulation is reproducible
[ ] existing LLM experiments are unaffected
[ ] output carries theory semantic/API version
[ ] very large episode counts do not require storing all trajectories
```

---

# 36. Final intended architecture

```text
src/mas_cc/games/relational_reasoning/imitation_round_feedback/

    theory_revised.py
        |
        | authoritative exact finite-state theory
        |
        +---- TheoryParameters
        +---- S(y|n), pi(1|y), a_n
        +---- K_{h,gamma}
        +---- Q0, Q1
        +---- chi
        +---- T_pi
        +---- eta_IR
        +---- one_cycle thermodynamics
        +---- finite_horizon_thermodynamics
        |
        v
    theory_simulation.py
        |
        | Monte Carlo realization of exactly the same model
        |
        +---- sample n_0
        +---- sample Y_k
        +---- sample U_k
        +---- execute b controlled K steps if U=1
        +---- record n_(k+1)
        +---- repeat for H cycles and E episodes
        |
        v
    simulation summaries / validation
        |
        +---- exact vs MC occupancy
        +---- exact vs MC chi
        +---- exact vs MC T_pi
        +---- exact vs MC current moments
        +---- exact vs MC eta_IR
        +---- exact vs MC Sigma
        +---- exact vs MC eta_th
```

The project-level comparison remains:

```text
LLM experiment
      |
      | empirical coarse-grained quantities
      v
single-affinity finite-compliance reference
      |
      +---- exact analytical/matrix calculation
      |
      +---- Monte Carlo trajectory realization
```

There is **no additional full-game classical baseline**.

That simplicity is intentional and should be preserved during implementation.
