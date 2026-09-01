# Agent handoff: generalize the single-affinity finite-time feedback theory from `q = 1` to arbitrary integer `q >= 1`

## 0. Objective

Extend the current finite-time single-affinity theory of feedback control from the special `q = 1` controlled layer to a **general finite-`N`, integer `q >= 1` theory** that accounts for the fact that, at a controlled microscopic update, the controller replaces only one social slot and the focal agent still sees `q-1` ordinary peers.

The new theory must preserve the conceptual architecture, terminology, notation, coding style, numerical exactness, report style, and figure philosophy of the current theory as closely as possible.

The two authoritative inputs are:

1. `22082026_feedback_control_single_affinity_report_revised.pdf`
2. `src/mas_cc/games/relational_reasoning/imitation_round_feedback/theory_revised.py`

Treat those as the baseline specification. Do **not** redesign the theory from scratch.

The key scientific goal is to determine whether the empirical `q > 1` behavior can be explained primarily as a change in the **kinetic/traffic part** of the controlled channel while retaining the same measurable controller affinity `h`.

The central sanity condition is strict:

> **At `q = 1`, the generalized theory must reduce numerically and algebraically to the current report and `theory_revised.py`.**

This should be tested at the kernel, susceptibility, information, current, finite-time thermodynamic, finite-horizon, and figure-data levels.

---

# 1. Preserve the current conceptual chain

The current report is organized around

\[
\text{sensing/policy}
\longrightarrow
\chi
\longrightarrow
T_\pi(n)
\longrightarrow
J_c
\longrightarrow
\Sigma
\longrightarrow
\eta_{\rm th}.
\]

Keep this chain.

For the generalized theory, use the same structure but make the dependence on `q` explicit:

\[
\boxed{
\text{sensing/policy}
\longrightarrow
K^{(q)}
\longrightarrow
\chi_q
\longrightarrow
T_{\pi,q}(n)
\longrightarrow
J_{c,q}
\longrightarrow
\Sigma_q
\longrightarrow
\eta_{{\rm th},q}
}
\]

The sensing and policy pieces should remain unchanged. The principal modification is the controlled microscopic kernel.

Do not collapse all quantities into one efficiency. Preserve the distinct interpretation of:

- sensing information `I(n_k;Y_k)`;
- susceptibility `chi_q`;
- action-to-population information `T_{pi,q}`;
- information-response efficiency `eta_IR,q`;
- directed controlled current `J_c,q`;
- finite-time path irreversibility `Sigma_q`;
- non-storage control expenditure `C_th,q`;
- thermodynamic control efficiency `eta_th,q`.

---

# 2. Scope: what `q` means here

In the runtime, `q` is the number of ordinary social slots shown to the focal agent at a microscopic update.

When the controller acts at one controlled update position, **one** of these slots is replaced by the controller. Therefore:

- `q = 1`: focal agent sees only the controller at a controlled position;
- `q = 2`: focal agent sees controller + 1 ordinary peer;
- `q = 3`: focal agent sees controller + 2 ordinary peers;
- etc.

The current theory is exact for the isolated `q = 1` controlled layer. For `q > 1`, the controller is embedded in a residual social context.

The new theory should model exactly this structure:

\[
\boxed{\text{controller} + (q-1)\text{ ordinary peers}.}
\]

Do **not** interpret `q` as sensor size. Sensor size remains `q_c`.

Do **not** confuse:

- `q`: ordinary social group size per microscopic LLM update;
- `q_c`: controller sensing sample size;
- `b`: number of controlled update opportunities when advocacy occurs.

Primary implementation support should be for integer

\[
1 \le q \le N.
\]

Do not silently introduce continuous/non-integer `q` in the core implementation. If a continuous analytic extrapolation is later useful for an illustrative closure, label it explicitly as such.

---

# 3. Important modeling decision: generalize the kinetics first, not the controller affinity

The preferred extension should preserve the single measurable controller affinity `h` if at all possible.

The working hypothesis is:

> Increasing `q` changes the **kinetic accessibility / compliance / traffic** of controlled transitions because the controller is embedded in additional peer context, while the controller's directional affinity remains `h`.

This is attractive for three reasons:

1. it preserves the current interpretation of `h` as the log directional odds of the controlled revision channel;
2. it is consistent with the stochastic-thermodynamic separation between directional affinity and symmetric activity/traffic;
3. it gives a natural explanation for empirical regimes in which `q > 1` changes `T_pi` strongly without producing a proportionate increase in coherent mean response.

The generalized theory should therefore be built in two layers:

### Primary model: single-affinity, context-dependent kinetics

Retain one global controller affinity `h`, while allowing the residual `q-1` peers to modulate the kinetic compliance/traffic.

### Diagnostic fallback: context-dependent affinity

Also implement a diagnostic that tests whether the empirical directional odds remain approximately constant across peer contexts. If this assumption fails strongly, provide a clearly separated optional extension with context/state-dependent affinity. Do **not** silently replace the single-affinity model.

The main report should remain centered on the single-affinity generalization unless diagnostics force otherwise.

---

# 4. Keep Sections 1-3 of the current report essentially unchanged

The following definitions remain exactly as in the current report.

## 4.1 Population state

\[
n_k \in \{0,1,\ldots,N\},
\qquad
x_k = n_k/N.
\]

`n_k` is the number of agents supporting the controller target immediately before sensing/control in feedback cycle `k`.

## 4.2 Population ensemble

\[
p_k(n)=P(n_k=n).
\]

No stationarity or equilibrium assumption.

## 4.3 Sensor

The controller samples `q_c` agents without replacement:

\[
Y_k|n_k=n \sim \operatorname{Hypergeom}(N,n,q_c),
\]

with the same sensor kernel

\[
S(y|n)
=
\frac{\binom{n}{y}\binom{N-n}{q_c-y}}{\binom{N}{q_c}}.
\]

## 4.4 Feedback policy

Keep

\[
\pi(1|y)
=
\sigma\!\left[\beta\left(\theta-\frac{y}{q_c}\right)\right]
\]

and

\[
a_n
=
P(U_k=1|n_k=n)
=
\sum_y S(y|n)\pi(1|y).
\]

No `q` dependence should be inserted into sensing or policy unless it is already present operationally in the runtime. The new `q` belongs to the actuation/social-context layer.

---

# 5. General controlled microscopic layer for `q >= 1`

This is the central extension.

At a controlled microscopic opportunity, one social slot is occupied by the controller and `q-1` residual ordinary peers remain.

## 5.1 Residual-peer context

For an edge between population counts `n` and `n+1`, define

\[
J \in \{0,1,\ldots,q-1\}
\]

as the number of target-supporting agents among the `q-1` residual ordinary peers.

For an upward transition `n -> n+1`, the focal agent is non-target. Excluding that focal agent leaves:

- `n` target agents;
- `N-n-1` non-target agents.

Therefore, for sampling without replacement,

\[
H_q(j|n)
=
\frac{
\binom{n}{j}
\binom{N-n-1}{q-1-j}
}{
\binom{N-1}{q-1}
}.
\]

This same residual-peer composition law appears in the reverse edge `n+1 -> n`: when the focal agent is target in state `n+1`, excluding it again leaves exactly `n` target and `N-n-1` non-target agents.

This edge-pair identity is extremely important. It is what allows the peer-context factor to cancel from the directional forward/reverse ratio if it enters symmetrically as a kinetic factor.

Implement `H_q(j|n)` exactly and test normalization for all supported `N,n,q`.

Boundary convention: `H_q(j|n)` is only needed for edges `n=0,...,N-1`. Impossible combinatorial terms are zero.

## 5.2 Context-dependent kinetic compliance

Introduce a nonnegative context-compliance function

\[
\gamma_q(j) \in [0,1].
\]

Interpretation:

> `gamma_q(j)` measures how kinetically responsive a controlled microscopic opportunity is when `j` of the residual `q-1` peers support the target.

This is **not** an additional directional affinity.

Define the edge-averaged kinetic factor

\[
G_q(n)
=
\sum_{j=0}^{q-1} H_q(j|n)\,\gamma_q(j).
\]

For `q=1`, the residual peer set is empty. Require

\[
\gamma_1(0)=\gamma,
\qquad
G_1(n)=\gamma.
\]

Thus the old theory must be recovered exactly.

### API design

The implementation should support at least these kinetic-context modes:

1. `constant`:
   \[
   \gamma_q(j)=\gamma.
   \]
   This is a null model: `q` has no kinetic effect.

2. `context_table`:
   user supplies the exact vector
   \[
   (\gamma_q(0),\ldots,\gamma_q(q-1)).
   \]
   This is the preferred model for operational calibration from LLM data.

3. `canonical_group_gate` (illustrative only):
   provide one clearly documented, bounded, monotone/group-interaction closure for theoretical plots. The exact choice should be explicitly labeled as an illustrative closure, not an empirical law of the LLMs.

A natural option is a standard residual-group gate in which the edge activity is written directly as a bounded `g_q(n)` satisfying `g_1(n)=1`; for example, a without-replacement q-voter-style residual-group factor can be used for illustration. However, the **general theorems must not depend on this particular closure**.

Prefer implementing the theory in terms of `G_q(n)` / `gamma_q(j)` first, and only then adding named closures.

---

# 6. Generalized controlled kernel

Keep the same single directional affinity

\[
p_h=\sigma(h).
\]

Define the generalized one-opportunity controlled birth-death kernel by **edge activity**:

\[
K_q(n+1|n)
=
\frac{N-n}{N}\,G_q(n)\,p_h,
\qquad n=0,\ldots,N-1,
\]

and

\[
K_q(n-1|n)
=
\frac{n}{N}\,G_q(n-1)\,(1-p_h),
\qquad n=1,\ldots,N.
\]

Then

\[
K_q(n|n)
=
1-K_q(n+1|n)-K_q(n-1|n).
\]

This construction has three intended properties:

1. `q` changes kinetics through the edge activities `G_q`;
2. the controller affinity `h` is unchanged;
3. each forward/reverse edge uses the same kinetic factor `G_q(n)`.

Require `0 <= G_q(n) <= 1` so that, together with `gamma <= 1`-style bounded compliance, the kernel remains stochastic. Validate all rows numerically.

## 6.1 Exact `q=1` reduction

For `q=1`, `G_1(n)=gamma`, hence

\[
K_1(n+1|n)
=
\gamma\frac{N-n}{N}\sigma(h),
\]

\[
K_1(n-1|n)
=
\gamma\frac{n}{N}\sigma(-h),
\]

which must be exactly the current `controlled_kernel`.

This is the first non-negotiable parity test.

---

# 7. Action-conditioned cycle kernels

Preserve the current isolated controlled-layer semantics:

\[
Q_0 = I,
\qquad
Q_1^{(q)} = K_q^b.
\]

Do **not** replace `Q_0` by a full uncontrolled social-round kernel in the primary theory. The existing report deliberately isolates the controlled channel, and the requested `q=1` parity would be broken if ordinary q-voter diffusion were inserted into `Q_0`.

The interpretation for `q>1` is:

> `K_q` is the incremental controlled microscopic layer after averaging over the `q-1` residual peers that coexist with the controller at a controlled update position.

This is the closest extension of the current semantics.

Later, if desired, a separate full-round theory may be built, but it is outside this task.

---

# 8. Exact generalized susceptibility

For `q=1`, preserve the current closed form

\[
\chi_1(x)
=
[\sigma(h)-x]
\left[1-\left(1-\frac{\gamma}{N}\right)^b\right].
\]

For general `q`, do **not** force an artificial closed form if the state-dependent kinetic factors make one unavailable.

Define the exact finite-`N` susceptibility from the matrix kernel:

\[
\boxed{
\chi_q(n/N)
=
E[x'|U=1,n,q]-E[x'|U=0,n,q]
=
\sum_{m=0}^N
\left(\frac{m-n}{N}\right)
Q_1^{(q)}(m|n)
}
\]

because `Q_0=I`.

Equivalently,

\[
\chi_q(n/N)
=
(Q_1^{(q)}x)_n-x_n,
\qquad x_n=n/N.
\]

This should become the primary exact definition for all `q`.

Then show explicitly that for `q=1` it collapses to the old analytic formula.

## 8.1 One-step drift

Also derive the one-opportunity drift

\[
d_q(n)
=
E[\Delta n|n]
=
K_q(n+1|n)-K_q(n-1|n).
\]

Write it explicitly in terms of `G_q(n)` and `G_q(n-1)`.

This is useful for interpreting why `q>1` may shift or distort the susceptibility landscape even while the edge affinity remains `h`.

## 8.2 Zero-response point

For `q=1`, retain

\[
x^*=\sigma(h).
\]

For `q>1`, do **not** assume the zero-response point remains exactly `sigma(h)` when edge activities are state dependent. Compute it from the exact `chi_q` curve.

This distinction should be highlighted in the report:

- `h` still fixes the edge-level directional bias;
- heterogeneous kinetic accessibility can distort the finite-step mean response and shift the state where the net controlled response vanishes.

If a chosen kinetic closure happens to preserve `x^*=sigma(h)`, show that as a special case rather than assuming it generally.

---

# 9. Exact state-local action-to-population information

Keep exactly the current definition, replacing `Q1` by the generalized `Q1^(q)`:

\[
Q_\pi^{(q)}(m|n)
=
(1-a_n)Q_0(m|n)
+a_n Q_1^{(q)}(m|n).
\]

Then

\[
\boxed{
T_{\pi,q}(n)
=
I(U_k;n_{k+1}|n_k=n,q)
=
\sum_{u\in\{0,1\}}P(u|n)
\sum_m
Q_u^{(q)}(m|n)
\log_2\frac{Q_u^{(q)}(m|n)}{Q_\pi^{(q)}(m|n)}
}
\]

with

\[
Q_0^{(q)}=I,
\qquad
Q_1^{(q)}=K_q^b.
\]

All existing interpretations remain valid:

- action variability is required;
- kernel separation is required;
- strong actuation with deterministic policy can still yield small action information;
- large action information need not imply large signed mean response.

This last point should receive more emphasis in the generalized report because it is a central reason for introducing `q`.

## 9.1 Ensemble weighting

Retain

\[
I(U_k;n_{k+1}|n_k,q)
=
\sum_n p_k(n)T_{\pi,q}(n).
\]

Also retain the arithmetic state average used for occupancy-free theoretical summaries.

---

# 10. Information-response efficiency

The Pinsker argument should remain valid because the next-state target fraction is still bounded in `[0,1]`.

Use the exact generalized susceptibility:

\[
T_{\pi,q}(n)
\ge
\frac{2a_n(1-a_n)}{\ln 2}
\chi_q(n/N)^2.
\]

Therefore

\[
\boxed{
\eta_{{\rm IR},q}(n)
=
\frac{
2a_n(1-a_n)\chi_q(n/N)^2
}{
(\ln 2)T_{\pi,q}(n)
}
\le 1
}
\]

whenever `T_{pi,q}(n) > 0`.

Keep the same semantic distinction from `eta_th`.

This quantity is particularly important for the `q > 1` theory because it can expose a regime with:

- larger `T_pi,q`;
- smaller `|chi_q|`;
- therefore less coherent conversion of action-dependent distributional change into directed mean motion.

Do not claim that this pattern must occur; the generalized theory should make it possible to study it.

---

# 11. Generalized controlled current

Use the same occupancy/action-weighted definition:

\[
\boxed{
J_{c,q,k}
=
N\sum_n p_k(n)a_n\chi_q(n/N).
}
\]

This is the exact controlled target-count current for the isolated generalized controlled layer.

Keep the same sign convention:

- positive: net movement toward controller target;
- negative: net movement away from controller target.

At `q=1`, verify exact agreement with the existing closed-form current.

---

# 12. Central thermodynamic result: the transition ratio should retain the same single affinity

This is the most important derivation to establish.

For each neighboring edge `n <-> n+1`, the generalized kernel has

\[
K_q(n+1|n)
=
\frac{N-n}{N}G_q(n)\sigma(h),
\]

and

\[
K_q(n|n+1)
=
\frac{n+1}{N}G_q(n)\sigma(-h).
\]

Therefore the kinetic factor cancels:

\[
\frac{K_q(n+1|n)}{K_q(n|n+1)}
=
\frac{N-n}{n+1}
\frac{\sigma(h)}{\sigma(-h)}
=
\frac{N-n}{n+1}e^h.
\]

With

\[
S_{\rm mix}(n)=\ln\binom{N}{n},
\]

obtain

\[
\boxed{
\ln\frac{K_q(n+1|n)}{K_q(n|n+1)}
=
S_{\rm mix}(n+1)-S_{\rm mix}(n)+h.
}
\]

This identity should be shown to hold for **every integer `q >= 1` and every allowed context-compliance profile** as long as the same edge activity enters both directions.

This is the key conceptual result:

> `q` changes the traffic/kinetics of controlled transitions while the directional affinity remains `h`.

Then define the same weight

\[
w_h(n)=\binom{N}{n}e^{hn}.
\]

Verify

\[
w_h(n)K_q(m|n)=w_h(m)K_q(n|m)
\]

for supported transitions.

Because weighted reversibility is preserved by matrix powers, prove

\[
w_h(n)Q_1^{(q)}(m|n)
=
w_h(m)Q_1^{(q)}(n|m),
\]

and therefore

\[
\boxed{
\ln\frac{Q_1^{(q)}(m|n)}{Q_1^{(q)}(n|m)}
=
S_{\rm mix}(m)-S_{\rm mix}(n)+h(m-n).
}
\]

If this derivation goes through, the finite-time thermodynamic identity retains exactly the same form as in the current report. This should be the main theoretical payoff of the generalization.

---

# 13. Forward/reverse finite-time feedback paths

Keep the existing definitions, replacing only the action-conditioned controlled kernel.

Forward path:

\[
P_F^{(k,q)}(n,y,u,m)
=
p_k(n)S(y|n)\pi(u|y)Q_u^{(q)}(m|n).
\]

Forward final ensemble:

\[
p_{k+1}(m)
=
\sum_{n,y,u}P_F^{(k,q)}(n,y,u,m).
\]

Sensor marginal:

\[
p_{Y,k}(y)=\sum_n p_k(n)S(y|n).
\]

Reverse reference:

\[
P_R^{(k,q)}(m,y,u,n)
=
p_{k+1}(m)p_{Y,k}(y)\pi(u|y)Q_u^{(q)}(n|m).
\]

Keep the same operational interpretation of the reverse reference.

Do not impose stationarity.

---

# 14. General `q` finite-time path identity

Define

\[
\Sigma_{k,q}^*
=
\ln\frac{P_F^{(k,q)}(n,y,u,m)}{P_R^{(k,q)}(m,y,u,n)}.
\]

With the same pointwise sensing information

\[
i_k(n;y)=\ln\frac{S(y|n)}{p_{Y,k}(y)},
\]

and stochastic system entropy

\[
s_{\rm sys}(n;p_k)
=-\ln p_k(n)+S_{\rm mix}(n),
\]

use the generalized transition ratio to prove

\[
\boxed{
\Sigma_{k,q}^*
=
\Delta s_{\rm sys}
+h\,j_c
+i_k(n;y),
}
\]

where

\[
j_c=u(m-n).
\]

Average over the forward paths:

\[
\boxed{
\Delta S_{{\rm sys},k}
+hJ_{c,q,k}
+I(n_k;Y_k)
=
\Sigma_{k,q}
=
D_{\rm KL}(P_F^{(k,q)}\Vert P_R^{(k,q)})
\ge 0.
}
\]

This should be demonstrated both algebraically and numerically.

If successful, emphasize strongly:

> The finite-time second-law-like identity survives arbitrary `q >= 1` under the single-affinity kinetic-context construction. `q` changes the controlled transition traffic and therefore susceptibility, action information, current, final ensemble, and irreversibility, but it does not introduce a new directional thermodynamic force.

That is likely the main theorem/result of the generalized report.

---

# 15. General `q` thermodynamic control efficiency

Retain the same non-storage expenditure

\[
C_{{\rm th},q,k}
=
\Sigma_{k,q}-\Delta S_{{\rm sys},k}
=
hJ_{c,q,k}+I(n_k;Y_k).
\]

For target-directed operation `h J_c >= 0`, define

\[
\boxed{
\eta_{{\rm th},q,k}
=
\frac{hJ_{c,q,k}}
{hJ_{c,q,k}+I(n_k;Y_k)}
\le 1.
}
\]

Do not modify the boundedness interpretation unless the derivation requires it.

If `hJ_c < 0`, retain the same signed diagnostic behavior as the current code/report.

The important scientific question becomes how the same information cost can be converted differently when `q` changes the kinetic response landscape.

---

# 16. Operational calibration for general `q`

The current report calibrates

\[
p_+=P(\text{non-Z}\to Z|\text{controlled}),
\qquad
p_-=P(Z\to\text{non-Z}|\text{controlled}),
\]

and obtains

\[
\gamma=p_++p_-,
\qquad
h=\ln(p_+/p_-).
\]

For general `q`, extend this calibration by residual peer context `j`.

At fixed `(q,j)`, estimate

\[
p_+^{(q,j)}
=
P(\text{non-Z}\to Z|\text{controlled},q,J=j),
\]

\[
p_-^{(q,j)}
=
P(Z\to\text{non-Z}|\text{controlled},q,J=j).
\]

Define empirical contextual quantities

\[
\gamma_{q,j}^{\rm eff}
=
p_+^{(q,j)}+p_-^{(q,j)},
\]

\[
h_{q,j}^{\rm eff}
=
\ln\frac{p_+^{(q,j)}}{p_-^{(q,j)}}.
\]

The **single-affinity hypothesis** predicts approximately

\[
h_{q,j}^{\rm eff}\approx h
\quad\text{for all sufficiently supported }j,
\]

while allowing

\[
\gamma_{q,j}^{\rm eff}
\]

to vary strongly with `j` and `q`.

This should be treated as a major empirical diagnostic, not merely a calibration detail.

## 16.1 Required calibration diagnostics

Implement utilities that report, by `(q,j)`:

- plus eligible count;
- plus transition count;
- minus eligible count;
- minus transition count;
- `p_plus`;
- `p_minus`;
- `gamma_eff`;
- `h_eff`;
- uncertainty / support warning when reverse transitions are too sparse;
- weighted pooled `h` across supported contexts;
- residuals `h_eff(q,j)-h_pooled`.

Do not invent a finite affinity when `p_minus=0`. Use either:

- explicit undefined/infinite diagnostics; or
- a clearly labeled Bayesian/pseudocount sensitivity estimate separate from the primary raw estimate.

Do not silently smooth the main calibration.

## 16.2 Building the theory kernel from calibration

Preferred empirical mode:

1. estimate a common `h` from all supported controlled transitions;
2. estimate the context compliances `gamma_q(j)`;
3. construct `G_q(n)` exactly from the hypergeometric residual-peer law;
4. build `K_q`, `Q1_q`, `chi_q`, `T_pi,q`, etc.

This creates a direct operational bridge between LLM micro-transitions and the generalized finite-state theory.

---

# 17. Diagnostic fallback if one affinity is not enough

Do not make this the main model initially, but prepare the mathematics/code sufficiently that the failure mode is explicit.

If the empirical `h_eff(q,j)` varies systematically with peer context beyond sampling uncertainty, then the kinetic-only single-affinity assumption is incomplete.

In that case define a generalized edge affinity

\[
A_q(n)
=
\ln\frac{K_q(n+1|n)}{K_q(n|n+1)}
-
\left[S_{\rm mix}(n+1)-S_{\rm mix}(n)\right].
\]

Because the population coordinate is one-dimensional, define a discrete potential

\[
\Psi_q(0)=0,
\qquad
\Psi_q(n+1)-\Psi_q(n)=A_q(n).
\]

Then the weighted reversible measure becomes

\[
w_q(n)=\binom{N}{n}e^{\Psi_q(n)}.
\]

If `Q1 = K_q^b` remains reversible with respect to this weight, derive

\[
\ln\frac{Q_1(m|n)}{Q_1(n|m)}
=
S_{\rm mix}(m)-S_{\rm mix}(n)
+\Psi_q(m)-\Psi_q(n).
\]

The path identity then contains a generalized directed contribution

\[
\Delta \Psi_q
\]

instead of `h(m-n)`.

This is the mathematically correct fallback, but keep it clearly separated from the preferred single-affinity theory.

A report conclusion such as “the single-affinity kinetic generalization is sufficient/insufficient” is scientifically useful. Do not force sufficiency.

---

# 18. Do not put persistence `rho` into the fundamental `q` theory in this task

Persistence is important empirically, but do not add it as another microscopic thermodynamic parameter here.

For this task, interpret persistence and epistemic dynamics as mechanisms that alter:

- the occupied state ensemble `p_k(n)`;
- possibly the empirically calibrated kinetic context profile `gamma_q(j)`;
- possibly the quality of the Markov coarse-graining in `n`.

The generalized `q` theory itself should remain conditional on the current population state and calibrated controlled channel.

This keeps the theory modular and prevents the parameter space from exploding.

If later needed, occupancy-weighted quantities can be written as

\[
\bar\chi(q,\rho,b)
=
\sum_n p_{q,\rho}(n)\chi_q(n/N;b),
\]

without inserting `rho` into the local kernel by fiat.

---

# 19. Implementation strategy

## 19.1 Do not destroy the current implementation

Keep `theory_revised.py` unchanged initially.

Create a new module, preferably

```text
src/mas_cc/games/relational_reasoning/imitation_round_feedback/theory_general_q.py
```

or another clearly named sibling.

Only after all `q=1` parity tests pass should the project consider replacing or aliasing the old implementation.

## 19.2 Suggested data structures

A possible parameter object:

```python
@dataclass(frozen=True, slots=True)
class GeneralQTheoryParameters:
    N: int
    q: int
    q_c: int
    b: int
    beta: float
    theta: float
    h: float
    # kinetic context specification
    context_mode: str
    gamma: float | None = None
    gamma_by_context: tuple[float, ...] | None = None
```

The exact API can differ, but preserve the current style:

- immutable dataclasses;
- explicit validation;
- deterministic finite-state calculations;
- cached reference objects;
- flat `as_fields()` outputs for analysis tables.

Suggested exact reference object:

```python
GeneralQSingleAffinityReference
```

containing at least:

- parameters;
- sensor kernel `S`;
- policy `pi1`;
- statewise advocacy `a_n`;
- residual-peer context kernel `H`;
- edge activity `G_q(n)`;
- microscopic controlled kernel `K_q`;
- `Q0`;
- `Q1_q`;
- exact `chi_q`;
- exact `T_pi_q`;
- Pinsker lower bound;
- `eta_IR_q`.

Keep compatibility aliases where they are scientifically harmless.

## 19.3 Functions to implement

At minimum:

```text
residual_peer_law(N, n, q)
residual_peer_kernel(N, q)
context_compliance(...)
edge_activity_curve(...)
controlled_kernel_general_q(...)
susceptibility_curve_general_q(...)
local_action_information(...)
information_response_lower_bound(...)
information_response_efficiency(...)
mean_controlled_current(...)
single_affinity_general_q_reference(...)
finite_horizon_thermodynamics(...)
finite_horizon_current_moments(...)
```

Reuse the existing sensing, policy, entropy, KL, finite-horizon, and numerical helper logic whenever possible instead of duplicating it unnecessarily.

## 19.4 Numerical policy

Preserve existing conventions:

- thermodynamic logs in nats;
- `T_pi` in bits;
- no Monte Carlo in the reference theory;
- undefined ratios remain `NaN` rather than being artificially set to zero;
- floating-point clipping only for tiny numerical residue;
- material bound violations should raise errors.

---

# 20. Required `q=1` parity suite

This is mandatory and should be extensive.

For many parameter tuples over the domains used in the report, compare the new generalized implementation at `q=1` against `theory_revised.py`.

At minimum test equality of:

1. sensor law;
2. sensor kernel;
3. policy advocacy vector;
4. statewise advocacy probability `a_n`;
5. microscopic controlled kernel `K`;
6. `Q0`;
7. `Q1 = K^b`;
8. susceptibility curve;
9. kernel mean response;
10. `T_pi`;
11. action-entropy ceiling;
12. Pinsker bound;
13. `eta_IR`;
14. closed-loop kernel;
15. one-cycle propagation;
16. controlled current `J_c`;
17. sensing information;
18. system entropy;
19. `Delta S_sys`;
20. direct path KL;
21. decomposed `Sigma`;
22. identity residual;
23. `C_th`;
24. `eta_th`;
25. finite-horizon thermodynamic totals;
26. finite-horizon current moments.

Use tight numerical tolerances, ideally around `1e-12` where matrix arithmetic allows and no worse than the current test tolerances.

The generalized code should fail loudly if `q=1` parity is broken.

## 20.1 Figure-data parity

Recompute the numerical data underlying every current report figure using the generalized code at `q=1` and compare with the old code.

Record maximum absolute and relative differences.

The goal is not pixel-perfect rendering; it is equality of the numerical curves/surfaces before plotting.

---

# 21. General `q` invariants and tests

For `q = 1,2,3,...` over representative finite `N`, test:

## Probability/kernel tests

- every residual-peer law sums to 1;
- every `G_q(n)` lies in `[0,1]`;
- every kernel entry is nonnegative;
- every `K_q` row sums to 1;
- every `Q1_q` row sums to 1;
- propagated ensembles remain normalized.

## Thermodynamic structure

For every supported edge verify numerically

\[
\ln\frac{K_q(n+1|n)}{K_q(n|n+1)}
-
[S_{\rm mix}(n+1)-S_{\rm mix}(n)]
=h.
\]

Verify weighted reversibility of `K_q` and `K_q^b`.

## Information bounds

Verify

\[
0\le T_{\pi,q}(n)\le h_2(a_n)
\]

and

\[
T_{\pi,q}(n)
\ge
\frac{2a_n(1-a_n)}{\ln2}\chi_q(n/N)^2.
\]

Verify `eta_IR <= 1` wherever defined.

## Finite-time identity

For arbitrary normalized ensembles `p_k`, verify

\[
\Delta S_{\rm sys}
+hJ_{c,q}
+I(n;Y)
=
D_{KL}(P_F||P_R)
\]

to floating-point precision.

## Thermodynamic efficiency

When `hJ_c >= 0` and denominator positive, verify

\[
0\le\eta_{\rm th,q}\le1.
\]

---

# 22. Generalized report: preserve the existing style and organization

Produce a new report rather than editing the old PDF destructively.

Suggested title:

> **Finite-Time Feedback Control with Social Context**  
> *Single-affinity control with arbitrary interaction size `q >= 1`*

or a similarly restrained title.

The report should visually and structurally resemble the current revised report:

- same notation style;
- same definition-first progression;
- same finite-time emphasis;
- same boxed key identities;
- similar figure aesthetics;
- concise captions that say exactly what changes;
- no unnecessary literature review expansion.

The current report should remain readable as the `q=1` special case of the new one.

## Suggested section structure

1. Problem and one-cycle architecture
2. Population ensembles and finite-time notation
3. Sensing and feedback decision
4. Controlled actuation with `q-1` residual peers
5. Residual-peer context and kinetic compliance
6. Exact generalized susceptibility `chi_q`
7. Exact state-local action-to-population information `T_pi,q`
   - bounded information-response conversion
   - baseline landscapes
   - parameter sweeps
8. Controlled current `J_c,q`
9. Transition-ratio identity for arbitrary `q`
10. Forward and reverse finite-time feedback paths
11. Finite-time path irreversibility and second-law identity
12. Finite-time thermodynamic control efficiency
   - current-irreversibility view
13. Direct finite-sum computation recipe
14. Role of group size and kinetic context
15. Operational calibration from microscopic LLM transitions
16. `q=1` reduction and validation
17. Scope and modeling limitations

This can be adjusted, but preserve the report's existing logical rhythm.

---

# 23. Figures: reproduce the old figure philosophy and add `q` explicitly

The purpose is not to create many decorative plots. Each figure should expose one structural dependency.

Use the same baseline values as the current report whenever possible:

```text
N = 24
h = 2
q_c = 12
beta = 4
theta = 1/2
b = 12 or swept
```

For context-dependent kinetics, clearly state the illustrative closure/profile used in purely theoretical figures.

## Figure 1 - generalized susceptibility

Reproduce the role of old Figure 1, but include `q`.

Recommended version:

- x-axis: target fraction `x=n/N`;
- y-axis: exact `chi_q(x)`;
- curves: `q = 1,2,3,4` at fixed `h,b` and one stated kinetic-context closure;
- mark the old `q=1` analytic curve and confirm exact overlap;
- optionally use a second panel varying `gamma` for a fixed `q` to retain the old compliance interpretation.

The caption should distinguish:

- edge-level affinity set by `h`;
- q-dependent kinetic distortion of the finite-step response.

## Figure 2 - state-local `T_pi,q` landscape

Reproduce the old `(x,b)` heatmap, but make `q` visible.

Preferred layout: small multiples for `q = 1,2,3,4` with a common color scale.

This is a key figure because it can show that group context may increase or redistribute action visibility even when the signed response is weaker.

## Figures 3-6 - existing parameter sweeps

Retain the four sweeps over:

- `h`;
- `q_c`;
- `beta`;
- `theta`.

Do not explode the number of panels unnecessarily. Pick one representative `q>1` (for example `q=2`) for the main reproduction, while always including `q=1` as a reference where useful.

## New explicit `q` sweep

Add one figure dedicated to varying `q` at fixed values of the other parameters.

At minimum show one of:

- `T_pi,q(n)` heatmaps across `q`;
- arithmetic state-average `T_pi` versus `b` for several `q`;
- occupancy-weighted `T_pi` if a theoretical ensemble is specified.

## Summary state-average figure

Extend the old Figure 7 to include a `q` sweep as an additional panel.

The existing four panels (`h`, `q_c`, `beta`, `theta`) should be preserved in spirit.

## Thermodynamic efficiency figure

Reproduce the old Figure 8 using the same illustrative

\[
p_0(n)=\operatorname{Binomial}(24,0.25)
\]

and compare `eta_th` versus `b` for several `q` values.

A clean option is one panel per `q`, each containing the old `q_c={6,12,18}` curves.

Ensure the `q=1` panel reproduces the current report numerically.

## Current-versus-irreversibility figure

Reproduce the old Figure 9 for multiple `q` values.

Use consistent axes where possible so the change in the current/irreversibility geometry is directly visible.

## Optional new response-information plane

If it adds insight without distracting from the report style, add a final diagnostic figure showing points/curves in

\[
(T_{\pi,q}, |\chi_q|)
\]

or

\[
(T_{\pi,q}, J_{c,q}).
\]

This is useful for visualizing the distinction between action visibility and directed control response.

Do not make this optional figure a substitute for the core report plots.

---

# 24. Theoretical data outputs

As with the current workflow, save the numerical data underlying figures.

At minimum export tables containing:

```text
q
n
x
b
q_c
beta
theta
h
context_mode
G_q(n)
a_n
chi_q(n)
T_pi_q(n)
pinsker_bound_q(n)
eta_IR_q(n)
```

For ensemble plots additionally save:

```text
J_c_q
I_sens
Delta_S_sys
Sigma_q
Sigma_direct_KL
C_th_q
eta_th_q
identity_residual
```

This makes all plots reproducible and allows direct comparison with LLM analysis.

---

# 25. Comparison outputs with the old theory

Create a dedicated parity/validation artifact, e.g.

```text
q1_parity_report.md
```

It should summarize:

- tested parameter tuples;
- maximum kernel difference;
- maximum susceptibility difference;
- maximum `T_pi` difference;
- maximum thermodynamic-quantity difference;
- direct-KL identity residual;
- finite-horizon differences;
- figure-data differences.

The expected scientific conclusion should be something like:

> The generalized `q >= 1` implementation reduces to the previous single-affinity theory at `q=1` to numerical precision.

Do not state this unless the tests demonstrate it.

---

# 26. Relationship to the classical q-voter literature

The new theory can use the same structural insight that nonlinear group interactions can enter through a symmetric activity/traffic factor while canceling from the directional affinity ratio.

However:

- do not claim that the LLM agents literally implement the classical q-voter rule;
- do not force a unanimity mechanism unless it is explicitly labeled as an illustrative closure;
- do not change the operational meaning of runtime `q`.

The generalized theory should remain a **matched finite-state control reference**, not a behavioral claim that the LLM is a q-voter.

---

# 27. Modeling limitations to state explicitly

The final report should preserve and update the limitations section.

In particular:

1. `n_k` remains a coarse-grained population coordinate.
2. LLM transitions may depend on hidden epistemic state, active facts, dialogue history, reasoning, and persistence.
3. The residual-peer context is summarized only through the chosen kinetic-context variable/profile.
4. The single-affinity hypothesis is empirical and testable, not guaranteed.
5. `q` may alter both occupancy and local transition kinetics in the actual LLM process.
6. The theory is finite-time and does not require equilibrium/stationarity.
7. The theory remains a reference model for the controlled population channel, not a full thermodynamics of the language-model population.

---

# 28. Recommended workflow for the agent

Follow this order. Do not start by rewriting the report.

## Phase A - understand and freeze the baseline

1. Read the current PDF carefully.
2. Read `theory_revised.py` carefully.
3. Run existing tests for `theory_revised.py`.
4. Create a small script that reproduces the old report's numerical figure data.
5. Freeze those values as the `q=1` reference.

## Phase B - implement the residual-peer and edge-activity layer

1. Implement `H_q(j|n)`.
2. Implement `gamma_q(j)` / context profile interface.
3. Implement `G_q(n)`.
4. Implement generalized `K_q`.
5. Verify row stochasticity.
6. Verify the edge transition-ratio identity.

Do not proceed until these are solid.

## Phase C - reconstruct the current theory objects from `K_q`

1. `Q0` and `Q1_q`.
2. exact `chi_q` by kernel expectation.
3. `T_pi,q`.
4. Pinsker lower bound and `eta_IR,q`.
5. closed-loop kernel and propagation.
6. `J_c,q`.
7. one-cycle thermodynamics.
8. finite-horizon thermodynamics/current moments.

## Phase D - prove and test `q=1` parity

Run the full parity suite before making generalized plots.

If parity fails, fix the theory/code first. Do not compensate in plotting code.

## Phase E - generalized figures

Reproduce the old plots using the new implementation.

First generate the `q=1` versions and confirm parity.

Then add `q>1` panels/sweeps.

## Phase F - report

Only after the implementation and checks are stable, write the generalized report, closely following the structure and tone of the current report.

---

# 29. Acceptance criteria

The task is complete only if all of the following are true.

### Mathematical

- A well-defined finite-`N` controlled kernel exists for every integer `1 <= q <= N`.
- `q` enters through residual social context / kinetic traffic.
- The single controller affinity `h` is preserved in the primary model.
- Exact susceptibility is defined for general `q`.
- Exact `T_pi,q`, `eta_IR,q`, current, propagation, and finite-time thermodynamics are defined.
- The forward/reverse path ratio is derived.
- The exact finite-time second-law-like identity is verified.

### Reduction

- `q=1` reduces to the current theory algebraically where possible.
- `q=1` matches `theory_revised.py` numerically across the full test suite.

### Software

- deterministic finite-state implementation;
- no Monte Carlo in the reference;
- robust validation and tests;
- saved figure data;
- old implementation remains available until parity is demonstrated.

### Report

- style and notation closely follow the current revised report;
- old figures are reproduced in spirit and numerically at `q=1`;
- new `q` dependence is shown explicitly;
- limitations and the single-affinity diagnostic are stated clearly.

### Scientific interpretation

The report should make it possible to answer the following question cleanly:

> Can the observed `q>1` departures be explained by a controller with the same directional affinity but a different, context-dependent kinetic response/traffic?

If yes, show this clearly.

If not, demonstrate where the single-affinity hypothesis fails and report the context-dependent-affinity fallback separately.

---

# 30. Final deliverables expected from the agent

Please produce:

1. `theory_general_q.py` (or equivalently named implementation);
2. unit tests for general `q` and `q=1` parity;
3. a `q1_parity_report.md`;
4. scripts/notebook/module for generating the generalized theory figures and CSV data;
5. a generalized report source and rendered PDF, closely matching the current report style;
6. a concise implementation summary explaining:
   - the new definitions;
   - which identities remain unchanged;
   - what `q` changes;
   - whether `q=1` parity passed;
   - whether the constant-`h` assumption remains empirically plausible when context-conditioned calibration is available.

Do not modify unrelated experiment/runtime code as part of this task unless a minimal read-only adapter is required for calibration.

---

# 31. One-sentence design principle

The whole extension should be organized around this principle:

> **Generalize `q` by letting the `q-1` residual peers modulate the symmetric kinetic accessibility of controlled transitions, while preserving the controller's directional affinity and the finite-time path-thermodynamic structure; recover the current theory exactly at `q=1`.**
