# HiddenBench Imitation Control — Research Handoff Summary

## 1. Current scientific objective

We are building a **controlled HiddenBench imitation game** to study how semantic reasoning changes the controllability and phase behavior of a multi-agent imitation process.

The core comparison is:

\[
\text{reasoning OFF: classical imitation}
\qquad\text{vs}\qquad
\text{reasoning ON: LLM-mediated imitation}.
\]

HiddenBench provides the distributed-information reasoning task. The imitation framework provides the population dynamics.

The main population state is the full opinion-count vector:

\[
N_t.
\]

We also track three macroscopic order parameters:

\[
m_{\rm ctrl},\qquad
m_{\rm truth},\qquad
m_{\rm order}.
\]

Interpretation:

- `m_ctrl`: alignment with the controller's target option.
- `m_truth`: alignment with the correct HiddenBench answer.
- `m_order`: degree of consensus/order, independent of which option wins.

The long-term scientific question is:

> Does semantic reasoning change how efficiently and in what direction a population responds to feedback information?

---

## 2. Current controller design

The original controller is a measurement-feedback loop:

\[
N_t \rightarrow Y_t \rightarrow U_t \rightarrow N_{t+1},
\]

where:

- \(N_t\): true population opinion counts,
- \(Y_t\): noisy sample from the population,
- \(U_t\): controller action,
- \(N_{t+1}\): next population state.

The action space is:

```text
ADVOCATE_Z
NO_OP
```

where \(Z\) is the controller target.

The current implemented controller is the hard threshold controller:

```text
threshold_target
```

Conceptually:

\[
U_t =
\begin{cases}
\mathrm{ADVOCATE}_Z, & \hat p_Z(Y_t) < \theta,\\
\mathrm{NO\_OP}, & \text{otherwise}.
\end{cases}
\]

This policy is often nearly deterministic once the current state is known.

That creates an important statistical problem:

\[
I(U_t;N_{t+1}\mid N_t)
\leq
H(U_t\mid N_t).
\]

If

\[
H(U_t\mid N_t)\approx 0,
\]

then the population actuation CMI must also be close to zero, even if the controller has a strong causal effect.

Therefore a very small CMI under the hard controller does **not automatically imply weak control**.

---

## 3. Soft stochastic controller

A new soft-feedback controller is being implemented.

The intended policy is:

\[
P(U_t=\mathrm{ADVOCATE}_Z\mid Y_t)
=
\sigma\!\left[
\beta\left(\theta-\hat p_Z(Y_t)\right)
\right],
\]

where:

\[
\sigma(x)=\frac{1}{1+e^{-x}}.
\]

Interpretation:

- if sampled support for the target is far below the threshold, advocacy is likely;
- if sampled support is far above the threshold, `NO_OP` is likely;
- near the threshold, both actions occur with appreciable probability.

The purpose of this controller is **not to artificially increase CMI**.

Its purpose is to create action overlap at comparable states so that quantities such as

\[
I(U_t;N_{t+1}\mid N_t)
\]

and

\[
I(U_t;m_{{\rm ctrl},t+1}\mid m_{{\rm ctrl},t})
\]

are statistically identifiable.

Implementation brief:

```text
soft_feedback_controller_implementation.md
```

---

## 4. Natural-language realization of control

For reasoning-ON runs, the controller action should be separated from its linguistic realization:

\[
Y_t
\rightarrow
U_t
\rightarrow
M_t
\rightarrow
X_{i,t+1}.
\]

Here:

- \(U_t\) is the abstract action,
- \(M_t\) is the natural-language controller message,
- \(X_{i,t+1}\) is the focal agent's next opinion.

For the first implementation, `ADVOCATE_Z` should produce one fixed, versioned natural-language advocacy message.

The controller must **not fabricate HiddenBench evidence** and must not pretend to possess private facts.

The intervention is social/argumentative pressure, not misinformation.

Example:

```text
I think we should reconsider North Hill before settling on another option.
I still favor North Hill and think it deserves more weight.
I'm voting North Hill.
```

In classical reasoning-OFF mode, the same abstract `ADVOCATE_Z` action should instead modify the provider-free imitation transition toward \(Z\).

This gives a clean correspondence:

\[
\text{classical transition bias toward } Z
\quad\leftrightarrow\quad
\text{linguistic advocacy toward } Z.
\]

---

## 5. Information-theoretic quantities already implemented

The current analysis contains:

### Sensing

\[
I(N_t;Y_t)
\]

implemented as:

```text
sensing_mi
```

This measures how much the noisy controller sensor reveals about the true population state.

Projected sensing quantities are also available:

```text
sensing_mi_m_ctrl
sensing_mi_m_truth
sensing_mi_m_order
```

These naturally tend to be smaller than full-state sensing MI because they are projections of \(N_t\).

### Population actuation

\[
I(U_t;N_{t+1}\mid N_t)
\]

implemented as:

```text
population_actuation_cmi
```

### Target actuation

\[
I(U_t;Z_{t+1}\mid Z_t)
\]

implemented as:

```text
target_actuation_cmi
```

### Focal-agent actuation

\[
I(U_t;X^f_{t+1}\mid X^f_t,N_t)
\]

implemented as:

```text
focal_actuation_cmi
```

### Order-parameter actuation

```text
m_ctrl_actuation_cmi
m_truth_actuation_cmi
m_order_actuation_cmi
```

with corresponding definitions:

\[
I(U_t;m_{{\rm ctrl},t+1}\mid m_{{\rm ctrl},t}),
\]

\[
I(U_t;m_{{\rm truth},t+1}\mid m_{{\rm truth},t}),
\]

\[
I(U_t;m_{{\rm order},t+1}\mid m_{{\rm order},t}).
\]

`target_actuation_cmi` and `m_ctrl_actuation_cmi` are identical by construction because `m_ctrl` is a one-to-one transform of target headcount.

When the controller target equals the correct answer:

\[
m_{\rm ctrl}=m_{\rm truth},
\]

so the corresponding CMI values are identical as well.

---

## 6. Important result from the large hard-controller run

A larger run was completed using:

- reasoning ON,
- hard `threshold_target`,
- 100 episodes,
- 100 interaction steps per episode,
- 10,000 total events.

The estimated actuation CMIs were:

\[
I(U;N'|N)=0.00732\text{ bits},
\]

\[
I(U;Z'|Z)=0.00349\text{ bits},
\]

\[
I(U;X_f'|X_f,N)=0.00782\text{ bits},
\]

\[
I(U;m_{\rm order}'|m_{\rm order})=0.04330\text{ bits}.
\]

This is much more informative than the earlier 2–3 episode pilot runs.

The earlier \(0.05\)-\(0.2\) bit actuation estimates were largely finite-sample effects.

The one-step actuation information under the hard threshold controller appears genuinely small.

However, the crucial caveat is:

\[
I(U_t;S_{t+1}\mid S_t)
\leq
H(U_t\mid S_t).
\]

We have not yet measured the corresponding conditional controller entropies.

Therefore we do not yet know whether the tiny CMI means:

1. weak actuation, or
2. almost zero conditional controller-action entropy.

This is the immediate technical question.

---

## 7. New metrics to add

A new analysis implementation task has been specified in:

```text
hidden_bench_controller_entropy_and_actuation_metrics.md
```

These quantities belong to the **post-hoc analysis** and should be exposed in the `analysis` section of:

```text
configs/runs/hidden_bench/hidden_bench_imitation_reasoning_control_task_grid.yaml
```

### 7.1 Controller entropy diagnostics

Add:

```text
controller_action_entropy
controller_action_entropy_given_population
controller_action_entropy_given_m_ctrl
controller_action_entropy_given_m_truth
controller_action_entropy_given_m_order
```

corresponding to:

\[
H(U_t),
\]

\[
H(U_t\mid N_t),
\]

\[
H(U_t\mid m_{\rm ctrl,t}),
\]

\[
H(U_t\mid m_{\rm truth,t}),
\]

\[
H(U_t\mid m_{\rm order,t}).
\]

These quantify how much action variability is actually available to the controller under each conditioning representation.

---

## 8. Information-fraction diagnostics

For each actuation CMI, add a normalized diagnostic.

For example:

\[
\rho_N
=
\frac{
I(U_t;N_{t+1}\mid N_t)
}{
H(U_t\mid N_t)
}.
\]

Analogously:

\[
\rho_{\rm ctrl}
=
\frac{
I(U_t;m_{{\rm ctrl},t+1}\mid m_{{\rm ctrl},t})
}{
H(U_t\mid m_{{\rm ctrl},t})
},
\]

and likewise for truth and order.

These should be called **information fractions**, not efficiencies.

They answer:

> Of the controller-action information available at fixed state, what fraction predicts the next state?

If the denominator is zero, the ratio is undefined and should be reported as `NaN`, not zero.

---

## 9. Signed behavioral actuation metrics

CMI is unsigned.

It measures predictive dependence, not whether the population moves toward the controller target.

Therefore add state-adjusted signed responses.

For example:

\[
\Delta_{\rm ctrl}(s)
=
E[
m_{{\rm ctrl},t+1}-m_{{\rm ctrl},t}
\mid
U_t=\mathrm{ADVOCATE}_Z,
m_{{\rm ctrl},t}=s
]
-
E[
m_{{\rm ctrl},t+1}-m_{{\rm ctrl},t}
\mid
U_t=\mathrm{NO\_OP},
m_{{\rm ctrl},t}=s
].
\]

Then aggregate over states where both actions are observed.

Add:

```text
m_ctrl_signed_actuation
m_truth_signed_actuation
m_order_signed_actuation
```

Interpretation:

- positive `m_ctrl_signed_actuation`: controller moves population toward its target;
- positive `m_truth_signed_actuation`: controller moves population toward the correct answer;
- positive `m_order_signed_actuation`: controller increases consensus/order.

These quantities become especially important in wrong-target experiments.

---

## 10. Action-overlap diagnostics

The soft controller is being introduced because the old threshold controller frequently produces only one action at a given conditioning state.

Therefore report overlap diagnostics such as:

```text
dual_action_conditioning_states
occupied_conditioning_states
fraction_conditioning_states_with_both_actions
fraction_events_in_dual_action_conditioning_states
```

for at least:

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

because it tells us how much of the data supports a genuine within-state comparison between `ADVOCATE_Z` and `NO_OP`.

This should be one of the primary diagnostics when comparing the hard and soft controllers.

---

## 11. Interpretation of the order-parameter CMIs

The order parameters are projections of the full state.

Therefore lower MI values are generally expected.

For example:

\[
N_t=(2,1,1)
\]

and

\[
N_t=(1,1,2)
\]

may have the same target headcount or same consensus magnitude while representing different population states.

Thus projecting:

\[
N_t\rightarrow m_t
\]

throws information away.

`m_ctrl_actuation_cmi` is especially clean because it corresponds directly to the target headcount.

`m_order_actuation_cmi` requires more care because conditioning only on `m_order` does not fully specify the population composition.

For example:

\[
(3,1,0)
\]

and

\[
(0,3,1)
\]

have the same order magnitude but different relationships to the controller target.

Therefore `m_order_actuation_cmi` should currently be viewed mainly as a coarse-grained predictive diagnostic rather than a clean causal actuation quantity.

---

## 12. Immediate next steps

### Step 1 — Finish the new analysis metrics

On the existing 100-episode hard-controller run, compute:

\[
H(U),
\]

\[
H(U\mid N),
\]

\[
H(U\mid m_{\rm ctrl}),
\]

\[
H(U\mid m_{\rm truth}),
\]

\[
H(U\mid m_{\rm order}),
\]

plus:

- information fractions,
- signed responses,
- action-overlap diagnostics.

This should immediately tell us whether the tiny hard-controller CMI is mainly an entropy-budget issue.

### Step 2 — Interpret the hard controller

If:

\[
H(U\mid N)\approx 0,
\]

then:

\[
I(U;N'|N)\approx 0
\]

is structurally expected.

If:

\[
H(U\mid N)
\]

is substantial but the CMI remains around \(10^{-3}\)-\(10^{-2}\) bits, then the one-step actuation really is weak.

### Step 3 — Finish and validate the soft controller

First run cheap classical/mock experiments.

The main success criterion is not larger CMI.

It is substantially improved action overlap:

\[
P(U=\mathrm{ADVOCATE}\mid S_t)
\notin\{0,1\}
\]

for relevant states.

### Step 4 — Run a large soft-controller experiment

Then compare hard vs soft using:

\[
H(U\mid S),
\]

\[
I(U;S'|S),
\]

\[
\frac{I(U;S'|S)}{H(U\mid S)},
\]

and signed actuation.

If the soft controller has good overlap and the CMI remains tiny, then that becomes a meaningful negative result about one-step controller actuation.

---

## 13. Main scientific experiment after controller validation

Run the matched comparison:

\[
\begin{array}{c|cc}
& \text{no control} & \text{feedback}\\
\hline
\text{reasoning ON} & A & B\\
\text{reasoning OFF} & C & D
\end{array}
\]

Keep matched:

- HiddenBench task,
- initial vote state,
- population size,
- controller target,
- sensor sample size,
- threshold,
- soft-controller parameter \(\beta\),
- interaction horizon,
- seeds where appropriate.

The key comparison is whether reasoning changes:

- the available controller-information budget,
- one-step information transfer,
- directional response,
- consensus behavior,
- eventual phase behavior.

---

## 14. Wrong-target experiment

A key experiment will deliberately choose:

\[
Z\neq Y^\ast,
\]

where \(Y^\ast\) is the correct HiddenBench answer.

This separates:

\[
m_{\rm ctrl}
\]

from:

\[
m_{\rm truth}.
\]

An effective adversarial controller could produce:

\[
m_{\rm ctrl}\uparrow
\]

while:

\[
m_{\rm truth}\downarrow.
\]

This is important because controller success and reasoning success are not the same thing.

It also provides a strong test of whether semantic reasoning resists or amplifies incorrect social influence.

---

## 15. Longer-term theory direction

Once the empirical framework is stable, return to the theoretical formulation.

The intended program is:

1. construct an exact classical controlled Markov model;
2. define the sensor kernel \(S(y\mid n)\);
3. define the stochastic policy \(\pi(u\mid y)\);
4. define the controlled transition kernel \(P_u(n'\mid n)\);
5. compute exact stationary distributions and information quantities;
6. compare analytical values against simulation;
7. connect the classical limit to stochastic thermodynamics;
8. investigate whether a Horowitz/Sandberg-style information-feedback bound or a related discrete-time formulation can be derived;
9. compare the reasoning-ON LLM dynamics against that classical benchmark.

The classical controlled process can be written schematically as:

\[
P(n',u,y\mid n)
=
S(y\mid n)\,
\pi(u\mid y)\,
P_u(n'\mid n).
\]

The eventual goal is to understand whether reasoning modifies the information/control trade-offs predicted by the classical stochastic process.

---

# Current central questions

The immediate technical question is:

\[
\boxed{
\text{Are the tiny hard-controller CMIs caused by weak actuation, or by }H(U\mid S)\approx0?
}
\]

The next experimental question is:

\[
\boxed{
\text{Does the soft stochastic controller restore enough action overlap to estimate actuation cleanly?}
}
\]

The main scientific question remains:

\[
\boxed{
\text{Does reasoning change how efficiently and in what direction a population responds to feedback information?}
}
\]

Resume the next discussion from these three questions.
