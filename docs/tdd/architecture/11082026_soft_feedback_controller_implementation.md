# Soft Feedback Controller for HiddenBench Imitation

## Goal

Implement one new controller option for the HiddenBench imitation game: a **stochastic soft-feedback controller**.

The purpose is to replace the current deterministic threshold policy

\[
U_t =
\begin{cases}
\texttt{ADVOCATE\_Z}, & \hat p_Z < \theta,\\
\texttt{NO\_OP}, & \text{otherwise},
\end{cases}
\]

with a probabilistic policy that preserves the same feedback logic but gives both actions non-zero probability in comparable population states.

This is needed because the current deterministic threshold controller produces severe action-support collapse when estimating quantities such as

\[
I(U_t; n_Z(t+1)\mid n_Z(t)).
\]

The new controller should make these conditional-information estimates statistically identifiable without changing the scientific meaning of the intervention.

The controller should remain a **local measurement-feedback controller**:

\[
n_t \rightarrow Y_t \rightarrow U_t \rightarrow n_{t+1}.
\]

For reasoning runs, the action `ADVOCATE_Z` is realized as a fixed natural-language advocacy message. For classical runs, the same abstract action should use the existing provider-free classical actuation mechanism.

---

## 1. New controller mechanism

Add a controller mechanism with a clear name such as:

```yaml
control:
  mechanism: soft_target
```

or, if naming conventions in the repository suggest otherwise:

```yaml
control:
  mechanism: stochastic_threshold_target
```

Prefer the shortest name that is consistent with the existing controller registry.

Do **not** replace the existing `threshold_target` controller. Keep both mechanisms available so that old runs remain reproducible.

---

## 2. Policy

Let the controller sample the population exactly as it already does.

Let

```text
sample_size = q_c
target_count_in_sample = y_Z
sampled_target_share = y_Z / q_c
```

and let `threshold` be the desired target-support threshold.

The controller chooses

```text
ADVOCATE_Z
NO_OP
```

stochastically according to

\[
P(U_t=\texttt{ADVOCATE\_Z}\mid Y_t)
=
\sigma\!\left[
\beta\left(\theta-\hat p_Z(Y_t)\right)
\right],
\]

where

\[
\sigma(x)=\frac{1}{1+e^{-x}}.
\]

Thus:

- if sampled support for the target is far below the threshold, advocacy is very likely;
- if sampled support is far above the threshold, `NO_OP` is very likely;
- close to the threshold, both actions occur with appreciable probability.

`beta` controls how deterministic the policy is.

Large `beta` approaches the old hard-threshold controller. Smaller `beta` produces more overlap between actions.

A reasonable first implementation should make `beta` configurable rather than selecting a scientifically final value.

Example:

```yaml
control:
  mechanism: soft_target
  options:
    target: correct
    sensor_sample_size: 2
    threshold: 0.5
    beta: 4.0
```

The implementation must use the run/episode RNG already used by the experiment framework. Do not create an unseeded independent RNG.

---

## 3. Natural-language realization in reasoning mode

The stochasticity belongs to the **choice of controller action**, not to generation of the controller message.

For v1, `ADVOCATE_Z` should map to one fixed, versioned natural-language message template.

The controller must **not fabricate evidence** and must **not claim access to HiddenBench facts** that were not assigned to an agent.

The controller is applying social/argumentative pressure toward the target, not injecting misinformation.

A suitable template is:

```text
I think we should reconsider {target} before settling on another option.
Based on the discussion so far, I still favor {target} and think it deserves
more weight. I'm voting {target}.
```

A shorter equivalent is also acceptable if it better matches the existing prompt style.

Important requirements:

1. The template contains no new task-specific factual claim.
2. It does not say things such as "I just learned that ...".
3. It does not pretend to possess private evidence.
4. It clearly advocates the configured target.
5. It should be deterministic/versioned for the first experiments.
6. It should appear to the focal LLM as a natural-language interaction, while the trajectory log still records the abstract controller action as `ADVOCATE_Z`.

Conceptually, reasoning mode becomes

\[
Y_t
\rightarrow
U_t
\rightarrow
M_t
\rightarrow
X_{i,t+1},
\]

where `M_t` is the fixed language realization of `ADVOCATE_Z`.

Do not introduce an LLM-generated controller message in this implementation.

---

## 4. Classical reasoning-OFF realization

The same abstract controller policy must also work with `dynamics_mode: classical`.

The measurement and stochastic action selection should be identical:

```text
population state
    -> stochastic population sample
    -> sampled target share
    -> soft policy
    -> ADVOCATE_Z or NO_OP
```

Only the actuator differs.

For classical mode:

- `NO_OP`: use the normal classical imitation transition.
- `ADVOCATE_Z`: use the existing provider-free classical controller actuation toward target `Z`.

Do not add any language-model call in classical mode.

This is important because reasoning OFF is intended to remain the physics-level special case of the experiment.

---

## 5. Recommended implementation strategy

Before editing code, locate the existing HiddenBench imitation implementation of:

```text
threshold_target
```

and reuse as much of its sensor, target resolution, logging, and actuation code as possible.

The new controller should differ from `threshold_target` only at the policy-selection step.

A clean design is conceptually:

```python
measurement = sensor.observe(population_state)

p_target = measurement.target_count / measurement.sample_size

advocacy_probability = sigmoid(
    beta * (threshold - p_target)
)

if rng.random() < advocacy_probability:
    action = ADVOCATE_Z
else:
    action = NO_OP
```

Then reuse the existing downstream actuation path.

Avoid duplicating:

- target resolution (`correct`, explicit option, etc.);
- sensor sampling;
- event construction;
- reasoning/classical branching;
- trajectory logging;
- target message construction infrastructure, if a suitable prompt block already exists.

If the current design has a controller base class or protocol, implement the new mechanism as another registered controller rather than adding conditionals throughout the game.

The repository already uses registry/config patterns for control mechanisms, so preserve that architecture rather than introducing a special-case flag. fileciteturn2file0

---

## 6. Configuration

The controller should support at least:

```yaml
control:
  mechanism: soft_target
  options:
    target: correct
    sensor_sample_size: 2
    threshold: 0.5
    beta: 4.0
```

Required fields:

```text
target
sensor_sample_size
threshold
beta
```

Validation:

- `sensor_sample_size >= 1`
- `sensor_sample_size <= population_size`
- `0 <= threshold <= 1`
- `beta > 0`
- resolved target must be one of the game's answer options

Do not silently fall back to a default target if target resolution fails.

A default `beta` may be provided if the current config system normally uses defaults, but it must appear in the fully resolved run configuration.

---

## 7. Logging requirements

Every controlled event should continue to log the fields required by the existing information analysis.

At minimum preserve:

```text
N_t
Y_t
U_t
N_t1
Z_t
Z_t1
focal agent before/after state
episode id
event/round index
```

In addition, add the following controller diagnostics if they are not already recorded:

```text
sampled_target_share
controller_advocacy_probability
controller_threshold
controller_beta
controller_target
controller_mechanism
```

`controller_advocacy_probability` is particularly important. It allows us to audit whether the implemented policy agrees with the intended soft policy.

For reasoning mode, also preserve enough provenance to determine that the message came from the controller and which versioned controller prompt/template was used.

Do not change the existing MI/CMI estimator for this task. The point of this controller is to improve the data-generating process, not to invent a new estimator.

---

## 8. Required tests

### Unit test: policy monotonicity

For fixed `threshold` and `beta`:

```text
lower sampled target support
    -> higher ADVOCATE_Z probability
```

For example, with threshold `0.5`:

```text
P(ADVOCATE | p_Z=0.0)
>
P(ADVOCATE | p_Z=0.5)
>
P(ADVOCATE | p_Z=1.0)
```

### Unit test: threshold midpoint

At

```text
sampled_target_share == threshold
```

verify

```text
P(ADVOCATE_Z) == 0.5
```

up to floating-point tolerance.

### Unit test: reproducibility

With the same seed, initial state, and config, the sequence of controller actions must be reproducible.

### Unit test: reasoning message

When `ADVOCATE_Z` fires in reasoning mode:

- the controller message contains the target;
- it contains no injected HiddenBench fact;
- it does not overwrite the focal agent's vote;
- the focal agent still produces the actual state transition through the normal LLM decision path.

### Unit test: classical mode

With `dynamics_mode: classical`:

- zero provider calls occur;
- the soft policy still produces both `ADVOCATE_Z` and `NO_OP`;
- the existing classical actuation mechanism is used.

### Regression test

Existing `threshold_target` runs must remain unchanged.

---

## 9. Smoke experiment

Before running expensive LLM episodes, run a cheap classical/mock experiment.

Use approximately:

```yaml
population_size: 4
sensor_sample_size: 2
threshold: 0.5
beta: 4.0
target: correct
```

Run enough events to inspect action overlap.

The key diagnostic is not merely

```text
H(U) > 0
```

but whether both actions occur within the same relevant conditioning states.

For the target CMI, inspect each `Z_t` slice and verify that substantially more slices contain both:

```text
ADVOCATE_Z
NO_OP
```

than under the hard threshold controller.

The old deterministic controller produced roughly 90% of events in single-action conditioning slices. The soft controller should reduce this substantially.

Do not proceed to a costly provider sweep if the action-support collapse remains essentially unchanged.

---

## 10. First real comparison

Once the smoke test passes, run the same matched HiddenBench imitation experiment with:

```text
reasoning ON  + soft feedback
reasoning OFF + soft feedback
```

and their corresponding no-control baselines.

Keep matched:

```text
task
initial vote vector
population size
controller target
sensor sample size
threshold
beta
horizon
random seeds where appropriate
```

The existing information analysis can then continue to compute:

```text
sensing_mi
population_actuation_cmi
target_actuation_cmi
focal_actuation_cmi
```

with `target_actuation_cmi`

\[
I(U_t;n_Z(t+1)\mid n_Z(t))
\]

remaining the primary low-dimensional actuation statistic.

The larger scientific study distinguishes information acquired by the controller from information associated with its actuation. The stochastic controller is being introduced specifically so the latter can be estimated under better action overlap.

---

## 11. Non-goals for this implementation

Do **not** implement any of the following now:

- fabricated evidence or misinformation;
- LLM-generated controller arguments;
- multiple persuasion styles;
- message embeddings;
- InfoNCE;
- history-conditioned transfer entropy;
- thermodynamic efficiency;
- new MI estimators;
- automatic optimization of `beta`;
- automatic optimization of the threshold.

Those can become later experimental axes.

For the present implementation, the intervention should remain:

```text
one stochastic policy
+
one fixed language actuator
+
the existing classical actuator
```

This keeps the experiment interpretable.

---

## 12. Acceptance criteria

The implementation is complete when all of the following hold:

- `soft_target` is selectable through normal run configuration.
- It uses the same stochastic population sensor as the current controller.
- It computes the configured sigmoid advocacy probability correctly.
- Its random decision is seeded/reproducible.
- Both `ADVOCATE_Z` and `NO_OP` occur with nontrivial probability.
- Reasoning mode uses a fixed advocacy-only message with no fabricated evidence.
- Classical mode performs zero LLM calls.
- The controller never directly overwrites an agent's committed answer.
- Existing information-analysis files can consume the resulting trajectories without modification.
- Logs contain the realized action and the probability with which it was chosen.
- Existing `threshold_target` behavior remains unchanged.
- A smoke run demonstrates materially improved within-state action overlap.

## Bottom line

Implement the stochasticity at the **abstract controller-action level**:

\[
Y_t \rightarrow U_t.
\]

Keep the natural-language realization of `ADVOCATE_Z` fixed and factual-claim-free:

\[
U_t \rightarrow M_t.
\]

Then let the reasoning agent decide whether that linguistic pressure changes its opinion:

\[
M_t + \text{evidence} + \text{history}
\rightarrow
X_{i,t+1}.
\]

This gives the experiment the statistical overlap needed for actuation CMI while preserving the intended distinction between classical imitation and semantic reasoning.
