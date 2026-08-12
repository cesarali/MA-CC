# Implementation Prompt — Controller Entropy, Information-Fraction, and Signed Actuation Metrics

## Context

This task is part of the **post-hoc analysis** for the HiddenBench imitation control experiments.

The existing analysis already computes the following information-theoretic quantities from completed imitation trajectories:

- `sensing_mi`
- `population_actuation_cmi`
- `target_actuation_cmi`
- `focal_actuation_cmi`
- `sensing_mi_m_ctrl`
- `sensing_mi_m_truth`
- `sensing_mi_m_order`
- `m_ctrl_actuation_cmi`
- `m_truth_actuation_cmi`
- `m_order_actuation_cmi`

The purpose of this task is to add a small set of **controller entropy diagnostics**, **normalized information fractions**, and **signed behavioral actuation metrics** so that the existing CMI values can be interpreted correctly.

These metrics belong to the **analysis layer**, not to the game runtime or streaming metrics.

After implementing them, expose/configure them in the `analysis` section of:

```text
configs/runs/hidden_bench/hidden_bench_imitation_reasoning_control_task_grid.yaml
```

Do not modify the scientific meaning of the existing MI/CMI estimators.

---

# 1. Why these metrics are needed

For the current hard threshold controller, the controller action

```text
U_t ∈ {ADVOCATE_Z, NO_OP}
```

can become almost deterministic once the current population state is known.

For any conditional actuation quantity,

\[
I(U_t; X_{t+1} \mid S_t)
\leq
H(U_t \mid S_t).
\]

Therefore a very small CMI can arise for two different reasons:

1. the controller has little effect on the next state; or
2. the controller has almost no remaining action entropy after conditioning on the current state.

The new metrics must distinguish these two cases.

The analysis should therefore report three complementary pieces:

1. **How much action uncertainty/freedom the controller has** — entropy.
2. **How much of that available action information is predictive of the next state** — CMI normalized by conditional controller entropy.
3. **Whether the population actually moves in the intended direction** — signed behavioral response.

---

# 2. Controller entropy metrics

Use the same discrete empirical/direct-counting conventions already used by the current information analysis.

All entropies must be reported in **bits**.

## 2.1 Unconditional controller action entropy

Add:

```text
controller_action_entropy
```

Definition:

\[
H(U_t).
\]

Interpretation:

> How variable is the controller action overall?

For a binary controller:

- `0 bits` means the controller always takes the same action.
- `1 bit` is the maximum, achieved when the two actions occur equally often.

This metric alone is not enough because a controller can have high overall entropy while being deterministic conditional on state.

## 2.2 Conditional controller entropy given full population state

Add:

```text
controller_action_entropy_given_population
```

Definition:

\[
H(U_t \mid N_t).
\]

This is the entropy budget for:

\[
I(U_t;N_{t+1}\mid N_t).
\]

The inequality

\[
I(U_t;N_{t+1}\mid N_t)
\leq
H(U_t\mid N_t)
\]

should hold up to numerical tolerance.

Interpretation:

> After the current population state is known, how much uncertainty remains in which controller action will be taken?

If this value is approximately zero, then a small `population_actuation_cmi` is structurally expected.

## 2.3 Conditional controller entropy given controller-alignment order parameter

Add:

```text
controller_action_entropy_given_m_ctrl
```

Definition:

\[
H(U_t\mid m_{\mathrm{ctrl},t}).
\]

In implementation, use the same discrete encoding already used for `m_ctrl`, i.e. the target headcount when appropriate.

This is the entropy budget for:

\[
I(U_t;m_{\mathrm{ctrl},t+1}\mid m_{\mathrm{ctrl},t}).
\]

## 2.4 Conditional controller entropy given truth-alignment order parameter

Add:

```text
controller_action_entropy_given_m_truth
```

Definition:

\[
H(U_t\mid m_{\mathrm{truth},t}).
\]

This is the entropy budget for:

\[
I(U_t;m_{\mathrm{truth},t+1}\mid m_{\mathrm{truth},t}).
\]

When the controller target is the correct answer, `m_ctrl` and `m_truth` coincide and these conditional entropies should match.

## 2.5 Conditional controller entropy given consensus/order parameter

Add:

```text
controller_action_entropy_given_m_order
```

Definition:

\[
H(U_t\mid m_{\mathrm{order},t}).
\]

This is the entropy budget for:

\[
I(U_t;m_{\mathrm{order},t+1}\mid m_{\mathrm{order},t}).
\]

Because `m_order` is a coarse projection of the population state, this value may remain substantially larger than \(H(U_t\mid N_t)\).

---

# 3. Information-fraction diagnostics

Add normalized diagnostics that compare each actuation CMI with the amount of conditional controller entropy available in the same conditioning space.

These are **diagnostics**, not thermodynamic efficiencies.

Use neutral names ending in `*_information_fraction`.

Do not call them efficiency in code, tables, or documentation.

## 3.1 Population information fraction

Add:

```text
population_actuation_information_fraction
```

Definition:

\[
ho_{\mathrm{population}}
=
rac{
I(U_t;N_{t+1}\mid N_t)
}{
H(U_t\mid N_t)
}.
\]

Interpretation:

> What fraction of the controller's available action information, after fixing the current population state, is predictive of the next population state?

## 3.2 Controller-alignment information fraction

Add:

```text
m_ctrl_actuation_information_fraction
```

Definition:

\[
ho_{\mathrm{ctrl}}
=
rac{
I(U_t;m_{\mathrm{ctrl},t+1}\mid m_{\mathrm{ctrl},t})
}{
H(U_t\mid m_{\mathrm{ctrl},t})
}.
\]

## 3.3 Truth-alignment information fraction

Add:

```text
m_truth_actuation_information_fraction
```

Definition:

\[
ho_{\mathrm{truth}}
=
rac{
I(U_t;m_{\mathrm{truth},t+1}\mid m_{\mathrm{truth},t})
}{
H(U_t\mid m_{\mathrm{truth},t})
}.
\]

## 3.4 Consensus/order information fraction

Add:

```text
m_order_actuation_information_fraction
```

Definition:

\[
ho_{\mathrm{order}}
=
rac{
I(U_t;m_{\mathrm{order},t+1}\mid m_{\mathrm{order},t})
}{
H(U_t\mid m_{\mathrm{order},t})
}.
\]

## 3.5 Zero-denominator handling

If the corresponding conditional entropy is approximately zero, report the information fraction as:

```text
NaN
```

or the project's established missing-value representation.

Do **not** return `0.0`, because the ratio is undefined rather than zero.

Also expose an interpretability flag/reason indicating that the controller is deterministic given the conditioning variable.

Do not silently clip ratio values to `[0, 1]`; finite-sample or bias-correction pathologies should remain visible.

---

# 4. Signed behavioral actuation metrics

CMI is unsigned. It tells us whether the controller action is associated with/predictive of a transition but not whether the transition moves in the desired direction.

Add signed behavioral response metrics.

The preferred estimator should be **state-adjusted**, not merely an unconditional difference between `ADVOCATE_Z` and `NO_OP`, because the controller policy depends on the current state.

## 4.1 Controller-target signed response

Add:

```text
m_ctrl_signed_actuation
```

For each current controller-alignment state \(s=m_{\mathrm{ctrl},t}\) where **both** controller actions are observed, compute:

\[
\Delta_{\mathrm{ctrl}}(s)
=
E[
m_{\mathrm{ctrl},t+1}-m_{\mathrm{ctrl},t}
\mid
U_t=\mathrm{ADVOCATE\_Z},
m_{\mathrm{ctrl},t}=s
]
-
E[
m_{\mathrm{ctrl},t+1}-m_{\mathrm{ctrl},t}
\mid
U_t=\mathrm{NO\_OP},
m_{\mathrm{ctrl},t}=s
].
\]

Then aggregate over valid overlap states:

\[
\Delta_{\mathrm{ctrl}}
=
\sum_{s\in\mathcal S_{\mathrm{overlap}}}
w_s\,
\Delta_{\mathrm{ctrl}}(s).
\]

Use empirical state-frequency weights over overlap-supported data unless the existing analysis already uses a more appropriate established weighting rule.

Interpretation:

- positive: advocacy produces more movement toward the controller target than `NO_OP`;
- zero: no measurable directional difference;
- negative: advocacy is associated with less movement toward the controller target.

## 4.2 Truth-alignment signed response

Add:

```text
m_truth_signed_actuation
```

Analogously:

\[
\Delta_{\mathrm{truth}}(s)
=
E[
m_{\mathrm{truth},t+1}-m_{\mathrm{truth},t}
\mid
U_t=\mathrm{ADVOCATE\_Z},
m_{\mathrm{truth},t}=s
]
-
E[
m_{\mathrm{truth},t+1}-m_{\mathrm{truth},t}
\mid
U_t=\mathrm{NO\_OP},
m_{\mathrm{truth},t}=s
].
\]

Aggregate only across states with action overlap.

This metric is important for wrong-target experiments because controller success and epistemic success can have opposite signs.

## 4.3 Consensus/order signed response

Add:

```text
m_order_signed_actuation
```

Analogously:

\[
\Delta_{\mathrm{order}}(s)
=
E[
m_{\mathrm{order},t+1}-m_{\mathrm{order},t}
\mid
U_t=\mathrm{ADVOCATE\_Z},
m_{\mathrm{order},t}=s
]
-
E[
m_{\mathrm{order},t+1}-m_{\mathrm{order},t}
\mid
U_t=\mathrm{NO\_OP},
m_{\mathrm{order},t}=s
].
\]

Interpretation:

- positive: advocacy increases consensus/order relative to `NO_OP`;
- negative: advocacy decreases consensus/order;
- zero: no directional effect on order.

Do not interpret this as target success.

---

# 5. Action-overlap diagnostics

The signed metrics above require both actions to be observed in comparable conditioning states.

For each conditioning representation used above, also report:

```text
dual_action_conditioning_states
occupied_conditioning_states
fraction_conditioning_states_with_both_actions
fraction_events_in_dual_action_conditioning_states
```

At minimum provide these for:

```text
N_t
m_ctrl
m_truth
m_order
```

The most important quantity is:

```text
fraction_events_in_dual_action_conditioning_states
```

because it tells us what fraction of the data actually contributes to a within-state comparison between `ADVOCATE_Z` and `NO_OP`.

This diagnostic is especially important when comparing the old deterministic threshold controller with the new soft controller.

Do not treat an overall nonzero `H(U)` as sufficient evidence of action overlap.

---

# 6. Bootstrap uncertainty

Use **whole episodes as the bootstrap unit**, consistent with the existing information analysis.

Do not bootstrap individual interaction events.

Provide a 95% bootstrap interval for:

- all controller entropy metrics;
- all information-fraction metrics;
- all signed actuation metrics.

For ratio metrics, recompute numerator and denominator inside each bootstrap resample and then take the ratio.

Do not construct a ratio CI by dividing CI endpoints.

If a bootstrap resample has zero conditional controller entropy, return `NaN` for that resample and follow the project's existing policy for partially undefined bootstrap distributions. Record the number of valid bootstrap replicates.

---

# 7. Null-model handling

Do not invent a new null model unless it is naturally supported by the existing analysis framework.

For entropy metrics:

- no permutation null is necessary; these are controller-policy diagnostics.

For information fractions:

- if the existing actuation-CMI permutation machinery can consistently recompute the ratio under the same perturbed action sequence, reuse it;
- otherwise report bootstrap uncertainty only and label the quantity as a normalization diagnostic.

For signed actuation metrics:

- if the existing temporal/permutation null can be applied by permuting `U_t` within episode and recomputing the full state-adjusted statistic, do so;
- otherwise bootstrap CI plus overlap diagnostics are sufficient for the first implementation.

Do not introduce a new causal-inference estimator in this task.

---

# 8. Output/reporting

Add these quantities to the existing HiddenBench imitation analysis output.

Suggested grouping:

```text
Controller entropy / available action information
-------------------------------------------------
controller_action_entropy
controller_action_entropy_given_population
controller_action_entropy_given_m_ctrl
controller_action_entropy_given_m_truth
controller_action_entropy_given_m_order

Actuation information fractions
-------------------------------
population_actuation_information_fraction
m_ctrl_actuation_information_fraction
m_truth_actuation_information_fraction
m_order_actuation_information_fraction

Signed behavioral actuation
---------------------------
m_ctrl_signed_actuation
m_truth_signed_actuation
m_order_signed_actuation

Action-overlap diagnostics
--------------------------
...
```

For every normalized information fraction, print:

```text
CMI numerator
conditional entropy denominator
ratio
```

so the result is auditable.

---

# 9. Required sanity checks

## 9.1 Entropy bounds

Verify numerically:

\[
I(U_t;N_{t+1}\mid N_t)
\leq
H(U_t\mid N_t)
\]

and analogously for each order parameter, up to numerical/estimator tolerance.

Do not silently clip violations.

## 9.2 Deterministic controller

Construct a toy sequence where:

```text
U_t = deterministic function of conditioning state
```

and verify:

```text
H(U | state) = 0
information_fraction = NaN
```

even if outcomes differ strongly across states.

## 9.3 Randomized controller

Construct a toy case where both actions occur at the same conditioning state and verify:

```text
H(U | state) > 0
```

and the information fraction is finite.

## 9.4 Signed response direction

Construct a toy case in which `ADVOCATE_Z` always increases `m_ctrl` relative to `NO_OP` and verify:

```text
m_ctrl_signed_actuation > 0
```

Reverse the effect and verify that the sign flips.

## 9.5 Controller target equals truth

When:

```text
controller_target == correct_answer
```

verify that, modulo identical encodings:

```text
controller_action_entropy_given_m_ctrl
==
controller_action_entropy_given_m_truth
```

and that the corresponding CMI/information-fraction/signed-response quantities coincide where mathematically expected.

---

# 10. Configuration requirement

After implementing the metrics, add them to the `analysis` section of:

```text
configs/runs/hidden_bench/hidden_bench_imitation_reasoning_control_task_grid.yaml
```

Follow the existing configuration style already used in that file.

The configuration should explicitly enable the new analysis quantities rather than relying on hidden defaults.

Do not alter unrelated experiment parameters.

The final resolved config should make it obvious that the run requests:

- controller entropy diagnostics;
- actuation information fractions;
- signed actuation responses;
- action-overlap diagnostics.

---

# 11. Scope / non-goals

Do not implement in this task:

- a new MI estimator;
- InfoNCE;
- text embeddings;
- history-conditioned transfer entropy;
- thermodynamic efficiency;
- a new controller;
- controller optimization;
- causal inference beyond the simple overlap-adjusted signed response defined above.

This task is only about making the **existing actuation CMI results interpretable**.

---

# 12. Acceptance criteria

The task is complete when:

1. All five controller entropy metrics are computed in bits.
2. The four normalized actuation information fractions are computed with correct zero-denominator handling.
3. The three state-adjusted signed behavioral actuation metrics are computed.
4. Action-overlap diagnostics are reported for the relevant conditioning variables.
5. Whole-episode bootstrap uncertainty is provided.
6. Existing CMI values remain unchanged.
7. Sanity tests for entropy bounds, deterministic policies, randomized policies, and response sign pass.
8. The new metrics appear in the normal HiddenBench imitation analysis report.
9. The metrics are explicitly enabled in the `analysis` section of:

```text
configs/runs/hidden_bench/hidden_bench_imitation_reasoning_control_task_grid.yaml
```

10. No provider/LLM calls are added by the analysis implementation.

## Bottom line

For each actuation channel, the analysis should now answer:

```text
How much controller-action uncertainty exists at fixed state?
        -> H(U | state)

How much of that available action information predicts the next state?
        -> I(U ; next_state | state) / H(U | state)

Does the controller move the relevant macroscopic variable in the intended direction?
        -> signed state-adjusted actuation response
```

These are diagnostics for interpreting the current CMI results. They should not yet be described as thermodynamic efficiencies.
