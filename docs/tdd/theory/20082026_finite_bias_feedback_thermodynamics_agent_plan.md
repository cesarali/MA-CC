# Thermodynamic Extension of the Controlled q-Voter / LLM Population Model
## Agent brief, derivation plan, constraints, and target closed-form efficiency theory
**Date:** 20 August 2026

---

# 0. Mission

The goal is to extend the **existing controlled q-voter reference theory** into the smallest possible **genuine stochastic-thermodynamic theory of feedback control** that remains faithful to the existing LLM population experiment.

The empirical experiment is **not to be redesigned** for this task.

The desired endpoint is a theory in which we can define, derive, and—where possible—obtain in closed form:

1. a finite-bias reversible control mechanism;
2. forward and reverse trajectory probabilities;
3. entropy production / irreversibility;
4. controller-induced currents and activities;
5. generalized control work;
6. a useful output or load work;
7. the information term produced by feedback;
8. a second-law-like bound;
9. one or more **dimensionless efficiencies bounded by 1**;
10. exact or nearly exact finite-\(N\), \(q=1\) formulas that can be evaluated beside the existing LLM results.

The central scientific target is

\[
\boxed{
\text{How efficiently can directed information and finite control bias drive a population against its natural/epistemic tendency?}
}
\]

The work must be mathematically conservative. Do **not** call a ratio a thermodynamic efficiency merely because it looks reasonable. A thermodynamic efficiency must follow from a derived inequality or a clearly specified stochastic-thermodynamic construction.

---

# 1. Non-negotiable scope constraints

## 1.1 Do not change the empirical LLM experiment

The current LLM experiment should remain exactly as it is.

The thermodynamic work is initially a **theoretical extension and interpretation layer**.

Do not request:
- continuous-time LLM calls;
- new interaction semantics;
- a different ballot;
- a new controller;
- a different information allocation;
- a redesign of the relational task.

A later validation experiment may be proposed only if the theory produces a parameter that can be independently estimated from existing or minimally extended trajectories.

## 1.2 Keep discrete population rounds

The natural time variable is the round

\[
k=0,1,\ldots,K.
\]

The current feedback cycle is

\[
N_k \longrightarrow Y_k \longrightarrow U_k \longrightarrow N_{k+1}.
\]

This is a legitimate repeated-feedback architecture. Do **not** convert the primary theory to continuous time merely for aesthetic reasons.

A continuous-time limit may be derived later as an optional asymptotic result if it follows naturally.

## 1.3 Start from finite \(N\)

The reference experiment uses \(N=24\). The theory already has exact finite-\(N\) round kernels.

Therefore:
- derive exact finite-\(N\) quantities first whenever practical;
- use mean-field / diffusion / large-\(N\) formulas only as secondary approximations;
- state approximation assumptions explicitly.

## 1.4 Solve \(q=1\) first

The current strongest empirical study is in the \(q=1\) regime.

The first thermodynamic theory must close for \(q=1\).

Do not expand to \(q\ge2\) until:
- the finite-bias \(q=1\) construction is internally consistent;
- entropy production is finite and well-defined;
- at least one second-law-like bound is derived;
- at least one efficiency has a clear interpretation.

\(q\ge2\) is an optional extension, not a prerequisite.

## 1.5 Do not invent physical units

This is a social / information-processing model, not a molecular system.

Use dimensionless stochastic-thermodynamic quantities by default, e.g.

\[
k_B T = 1
\]

only as a mathematical convention if needed.

Prefer terms such as:
- generalized affinity;
- generalized work;
- entropy production;
- chemical-work analogue;
- information resource;

unless a physical mapping has genuinely been established.

Never report Joules, Watts, or a physical temperature.

---

# 2. Existing empirical LLM system

The empirical system is a population of LLM agents solving a synthetic relational reasoning task.

Typical Study 04 configuration:

\[
N=24,\qquad L=2,\qquad K=3,\qquad q=1,
\]

with:
- exact ground truth;
- supporting facts distributed among agents;
- no single-agent solution initially;
- persistent public votes;
- explicit per-agent knowledge state;
- a controller that may advocate a target answer \(Z\);
- stochastic finite-population sensing;
- round-level action commitment;
- fixed actuation budget \(b\).

The current target fraction is

\[
x_k = \frac{n_{Z,k}}{N}.
\]

Use \(x\) consistently for the normalized target state and \(n_Z\) for the count.

Do not switch notation to \(p_Z\).

## 2.1 Empirical epistemic variables

The empirical population contains a real internal/epistemic state in addition to the vote count.

Important observables include

\[
\phi_k
=
\frac{\#\{\text{agents possessing the complete proof}\}}{N},
\]

and

\[
\kappa_k
=
\frac{1}{N}
\sum_i
\frac{|K_i(k)\cap S|}{|S|},
\]

where \(S\) is the set of supporting facts.

The LLM population is therefore not genuinely Markov in \(n_Z\) alone.

This fact must be respected when mapping theory to data.

## 2.2 Existing empirical directed-information quantity

The principal round-level action-to-population information quantity is

\[
T_{\rm emp}
=
I(U_k;n_{Z,k+1}\mid n_{Z,k}).
\]

State-local estimates are

\[
T_{\rm emp}(n)
=
I(U_k;n_{Z,k+1}\mid n_{Z,k}=n).
\]

Memory / epistemic conditioned versions already considered include

\[
I(U_k;n_{Z,k+1}\mid n_{Z,k},\phi_k),
\]

\[
I(U_k;n_{Z,k+1}\mid n_{Z,k},\kappa_k),
\]

and

\[
I(U_k;n_{Z,k+1}\mid n_{Z,k},n_{Z,k-1}).
\]

Important empirical fact: conditioning on epistemic state and one-step history attenuates part of the state-local information landscape, but the strongest high-\(b\) interior band survives.

---

# 3. Existing controlled q-voter reference theory

The current classical state is the target count

\[
N_k=n
\]

or equivalently

\[
x_k=n/N.
\]

The controller senses a finite sample

\[
Y_k\mid N_k=n
\sim
\mathrm{Hypergeom}(N,n,q_c).
\]

The soft feedback policy is

\[
P(U_k=\mathrm{ADV}\mid Y_k=y)
=
\sigma\!\left[
\beta\left(\theta-\frac{y}{q_c}\right)
\right],
\]

where

\[
\sigma(z)=\frac{1}{1+e^{-z}}.
\]

Therefore the exact state-level advocacy probability is

\[
a_n
=
P(U_k=\mathrm{ADV}\mid N_k=n)
=
\sum_y
S(y\mid n)
\sigma\!\left[
\beta\left(\theta-\frac{y}{q_c}\right)
\right].
\]

Interpretation of controller parameters:

\[
q_c = \text{sensing budget},
\]

\[
\theta = \text{feedback set point},
\]

\[
\beta = \text{decision gain / softness},
\]

\[
b = \text{actuation budget}.
\]

The current main study uses

\[
\theta=0.5,\qquad \beta=4,
\]

with

\[
q_c\in\{6,12,18\},\qquad
b\in\{6,12,18\}.
\]

## 3.1 Existing exact whole-round kernels

Let

\[
R_0(m\mid n)
\]

be the exact finite-\(N\) whole-round kernel under NoOp, and

\[
R_1(m\mid n)
\]

the exact finite-\(N\) whole-round kernel under the current one-sided Advocate mechanism.

The policy-mixed kernel is

\[
\bar R(m\mid n)
=
(1-a_n)R_0(m\mid n)
+
a_n R_1(m\mid n).
\]

These exact kernels already exist in the current theory/post-processing machinery.

Reuse them. Do not replace them with an unnecessary approximation.

## 3.2 Existing exact local transfer entropy

The exact classical state-local directed information is

\[
\boxed{
T_{\rm qv}(n)
=
I(U_k;N_{k+1}\mid N_k=n)
=
JS_{a_n}
\left(
R_0(\cdot\mid n),
R_1(\cdot\mid n)
\right).
}
\]

The whole-round quantity is

\[
I(U_k;N_{k+1}\mid N_k).
\]

This already produces a nontrivial state-local information ridge.

That result must be preserved under the finite-bias extension.

## 3.3 Existing exact \(q=1\) response

For the current deterministic one-sided controlled update,

\[
\boxed{
\Delta\mu_{\infty}(x)
=
(1-x)
\left[
1-
\left(1-\frac1N\right)^b
\right].
}
\]

Here

\[
\Delta\mu(x)
=
E[x_{k+1}\mid x_k=x,\mathrm{ADV}]
-
E[x_{k+1}\mid x_k=x,\mathrm{NOOP}].
\]

The notation \(\infty\) is useful because this current controlled update will become the \(h_c\to+\infty\) limit of the finite-bias theory below.

---

# 4. Why the present controller is thermodynamically singular

The current theoretical controlled interaction is effectively one-sided.

Conditional on a controlled \(q=1\) slot:

\[
\text{non-target}\rightarrow Z
\]

is allowed, while the reverse controller-mediated transition

\[
Z\rightarrow \text{non-target}
\]

is forbidden.

Thus the corresponding transition-rate ratio is formally

\[
\frac{W^c_{0\to1}}{W^c_{1\to0}}
\to\infty.
\]

An affinity defined by

\[
h_c
=
\ln
\frac{W^c_{0\to1}}{W^c_{1\to0}}
\]

therefore diverges.

This is the sense in which the current controller is an **infinite-bias controller**.

This is not a problem for transfer entropy.

It is a problem for a clean finite entropy-production construction because local detailed-balance expressions contain logarithms of forward/backward transition ratios.

---

# 5. Required finite-bias controller construction

The preferred minimal extension is a finite-compliance / finite-bias controlled microscopic update.

Let

\[
s_i\in\{0,1\},
\]

where \(1\) means support for controller target \(Z\).

For a controlled microscopic update, define

\[
p_h
=
\sigma(h_c),
\qquad
1-p_h
=
\sigma(-h_c).
\]

Then the controlled birth/death probabilities at count \(n\) are

\[
\boxed{
K_{h_c}(n+1\mid n)
=
\frac{N-n}{N}\sigma(h_c),
}
\]

\[
\boxed{
K_{h_c}(n-1\mid n)
=
\frac{n}{N}\sigma(-h_c),
}
\]

and

\[
K_{h_c}(n\mid n)
=
1-
K_{h_c}(n+1\mid n)
-
K_{h_c}(n-1\mid n).
\]

Interpretation:
- \(h_c>0\): the controller biases toward \(Z\);
- \(h_c=0\): unbiased finite-compliance update;
- \(h_c<0\): bias away from \(Z\);
- \(h_c\to+\infty\): recover the current idealized controller.

The limit must be checked explicitly:

\[
\lim_{h_c\to\infty}
K_{h_c}(n+1\mid n)
=
\frac{N-n}{N},
\]

\[
\lim_{h_c\to\infty}
K_{h_c}(n-1\mid n)
=
0.
\]

---

# 6. Key local-detailed-balance identity

Since

\[
\frac{\sigma(h_c)}{\sigma(-h_c)}
=
e^{h_c},
\]

we obtain

\[
\frac{
K_{h_c}(n+1\mid n)
}{
K_{h_c}(n\mid n+1)
}
=
\frac{N-n}{n+1}e^{h_c}.
\]

Define the combinatorial system entropy

\[
S_{\rm mix}(n)
=
\ln {N\choose n}.
\]

Then

\[
S_{\rm mix}(n+1)-S_{\rm mix}(n)
=
\ln\frac{N-n}{n+1}.
\]

Therefore

\[
\boxed{
\ln
\frac{
K_{h_c}(n+1\mid n)
}{
K_{h_c}(n\mid n+1)
}
=
\Delta S_{\rm mix}(n\to n+1)
+
h_c.
}
\]

This is the central local-detailed-balance candidate.

Interpret \(h_c\) as a dimensionless generalized chemical potential / social control affinity.

The agent must verify this derivation carefully and determine the exact stochastic-thermodynamic interpretation.

A useful sanity check is that the controlled-only chain has equilibrium-like stationary measure

\[
\pi_{h_c}(n)
\propto
{N\choose n}e^{h_c n},
\]

equivalently a binomial distribution with single-agent target probability

\[
p_h=\sigma(h_c).
\]

Prove this.

---

# 7. Exact finite-bias \(q=1\) response: derive this first

For one controlled microscopic update,

\[
E[x_{j+1}-x_j\mid x_j=x]
=
\frac{\sigma(h_c)-x}{N}.
\]

Hence

\[
E[x_{j+1}\mid x_j]
=
\left(1-\frac1N\right)x_j
+
\frac{\sigma(h_c)}{N}.
\]

After \(b\) controlled opportunities,

\[
E[x_b\mid x_0=x]
=
\sigma(h_c)
+
\left[x-\sigma(h_c)\right]
\left(1-\frac1N\right)^b.
\]

Because the ordinary \(q=1\) voter dynamics are a martingale in \(x\), the candidate exact whole-round mean response is

\[
\boxed{
\Delta\mu_{h_c}(x)
=
\left[
\sigma(h_c)-x
\right]
\left[
1-\left(1-\frac1N\right)^b
\right].
}
\]

The agent must verify this against the exact round schedule/composition used by the existing theory.

Required limiting checks:

\[
h_c\to+\infty
\quad\Rightarrow\quad
\Delta\mu_{h_c}(x)
\to
(1-x)
\left[
1-\left(1-\frac1N\right)^b
\right].
\]

At

\[
x=\sigma(h_c),
\]

the mean controlled response vanishes.

This finite-bias model therefore has an intrinsic control set point.

Do not confuse this with the policy threshold \(\theta\).

They are different:

\[
\theta
=
\text{when the controller decides to act},
\]

\[
h_c
=
\text{how strongly a controlled interaction biases an agent}.
\]

---

# 8. Whole-round finite-bias kernel

Construct

\[
R_{h_c}(m\mid n)
\]

using the **same round construction as the existing exact theory**, replacing the deterministic controlled microscopic kernel by \(K_{h_c}\).

The actuation budget remains exactly \(b\).

Do not replace exact-\(b\) control by independent Bernoulli control unless explicitly deriving an approximation.

The finite-bias policy-mixed kernel is

\[
\boxed{
\bar R_{h_c}(m\mid n)
=
(1-a_n)R_0(m\mid n)
+
a_nR_{h_c}(m\mid n).
}
\]

The finite-bias exact state-local TE is therefore

\[
\boxed{
T_{h_c}(n)
=
JS_{a_n}
\left(
R_0(\cdot\mid n),
R_{h_c}(\cdot\mid n)
\right).
}
\]

Required limits:

\[
h_c\to+\infty
\quad\Rightarrow\quad
R_{h_c}\to R_1,
\]

and

\[
T_{h_c}(n)\to T_{\rm qv}(n).
\]

---

# 9. Important warning: finite bias alone will not explain every LLM departure

For the minimal finite-bias \(q=1\) response,

\[
\frac{\partial}{\partial x}
\Delta\mu_{h_c}(x)
=
-
\left[
1-\left(1-\frac1N\right)^b
\right]
<0.
\]

Therefore finite \(h_c\) by itself does **not** reverse the theoretical state-response slope.

The empirical LLM susceptibility currently tends to rise with existing target support over the observed state range.

Do not force the finite-bias thermodynamic extension to explain that result.

That discrepancy may require:
- epistemic state;
- memory;
- state-dependent social coupling;
- nonlinear reinforcement;
- a richer effective state.

The thermodynamic extension and the social-reinforcement extension are logically distinct.

---

# 10. The missing ingredient for a genuine efficiency: an opposing load

A finite-bias controller gives a force/current/entropy-production structure, but a useful **motor-like efficiency** requires an output.

The current neutral \(q=1\) voter model does not contain a thermodynamic load corresponding to “truth.”

The empirical LLM system does.

Agents with stronger evidence tend to move toward truth, and full-proof agents are almost completely resistant to the wrong controller.

Therefore introduce, if mathematically necessary, a **minimal opposing epistemic/load affinity**

\[
h_e\ge0
\]

that favors truth / opposes the wrong controller target.

Start with the simplest constant \(h_e\).

Only after the constant-load theory closes should one consider a knowledge-dependent version

\[
h_e=h_e(K),
\]

or a coarse two-state form such as

\[
h_e=
\begin{cases}
h_{\rm weak}, & \text{incomplete proof},\\
h_{\rm full}, & \text{complete proof}.
\end{cases}
\]

The purpose is not to fit every LLM detail. It is to construct the smallest thermodynamically meaningful competition:

\[
\boxed{
\text{controller bias toward }Z
\quad\text{versus}\quad
\text{epistemic/load bias toward truth}.
}
\]

---

# 11. Candidate combined local-detailed-balance form

For a target-directed transition \(n\to n+1\), a natural target form is

\[
\boxed{
\ln
\frac{
W_{n+1,n}
}{
W_{n,n+1}
}
=
\Delta S_{\rm mix}
+
u\,h_c
-
h_e,
}
\]

where

\[
u=
\begin{cases}
1, & U=\mathrm{ADV},\\
0, & U=\mathrm{NOOP}.
\end{cases}
\]

This is a candidate, not a result.

The agent must derive a microscopic transition rule that produces this relation rather than simply postulating the ratio.

A minimal Glauber/logistic candidate is

\[
P(0\to1\mid u)
=
\sigma(u h_c-h_e),
\]

\[
P(1\to0\mid u)
=
\sigma(h_e-u h_c).
\]

Check whether this is compatible with:
- the existing q-voter social update;
- exact-\(b\) actuation;
- the desired \(h_c\to\infty\) limit;
- a meaningful NoOp limit.

If not, construct a better mechanism.

Do not hide inconsistencies.

---

# 12. Thermodynamic dictionary

The theory should aim for the following correspondence.

| Langevin / stochastic thermodynamics | Population control model |
|---|---|
| coordinate \(x\) | target count \(n_Z\) or fraction \(x=n_Z/N\) |
| velocity / particle current | opinion current \(J\) |
| external force \(f_c\) | control affinity \(h_c\) |
| load force \(f_{\rm load}\) | epistemic/truth affinity \(h_e\) |
| work increment \(f_c\,dx\) | generalized control work \(h_c\,d\mathcal J_c\) |
| load work | \(h_e\,d\mathcal J\) for target current against truth |
| heat / entropy flow | log forward/backward path-weight contribution |
| measurement | \(Y_k\sim S(\cdot\mid N_k)\) |
| feedback protocol | \(U_k\sim\pi(\cdot\mid Y_k)\) |
| controller information | derive from measurement/feedback trajectory |
| entropy production | forward/reverse path log-ratio |

Reserve \(I(\cdot;\cdot)\) for mutual information.

Use \(\mathcal J\) for integrated current.

---

# 13. Controller current, activity, and generalized work

For controller-mediated elementary transitions, define

\[
\mathcal J_c
=
N^c_+ - N^c_-,
\]

where:
- \(N^c_+\) counts target-directed controlled jumps;
- \(N^c_-\) counts reverse controlled jumps.

The mean current per round is

\[
J_c
=
E[\mathcal J_c].
\]

Define activity

\[
A_c
=
E[N^c_+ + N^c_-].
\]

The generalized controller work candidate is

\[
\boxed{
\mathcal W_c[\gamma]
=
h_c\,\mathcal J_c[\gamma]
}
\]

for constant \(h_c\).

Mean control work:

\[
\boxed{
W_c
=
h_c J_c.
}
\]

This is the discrete-jump analogue of

\[
W=\int f\,dx.
\]

Important:

\[
b\neq h_c.
\]

The actuation budget \(b\) controls how many opportunities the force has to act.

The affinity \(h_c\) controls the directional strength of one controlled interaction.

---

# 14. Useful output / load work

If \(h_e>0\) opposes target-directed motion, then a positive target current

\[
J>0
\]

works against the epistemic load.

The simplest load-work candidate is

\[
\boxed{
W_{\rm out}
=
h_e J.
}
\]

The agent must determine whether \(J\) should be:
- total target current;
- controller-mediated current only;
- excess current relative to NoOp;
- a cycle current in a mechanism-resolved network.

Do not choose this arbitrarily.

The correct definition should follow from the mechanism decomposition and the stochastic-thermodynamic balance.

---

# 15. The central target inequality

The hoped-for steady/cyclic bound has the schematic structure

\[
\boxed{
W_{\rm out}
\le
W_{\rm ctrl}
+
\mathcal I_{\rm fb},
}
\]

where:
- \(W_{\rm out}\) is useful work against the epistemic load;
- \(W_{\rm ctrl}\) is generalized work supplied by the controller bias;
- \(\mathcal I_{\rm fb}\) is the correct information resource generated/consumed by feedback.

In the simple one-current picture this would resemble

\[
\boxed{
h_e J
\le
h_c J
+
\mathcal I_{\rm fb}.
}
\]

Equivalently,

\[
\boxed{
(h_e-h_c)J
\le
\mathcal I_{\rm fb}.
}
\]

This is only a **target structure**.

Do not state it as proven until it is derived from forward/reverse trajectory probabilities.

---

# 16. Target thermodynamic efficiency

If the above bound is valid, define

\[
\boxed{
\eta_{\rm th}
=
\frac{W_{\rm out}}
{W_{\rm ctrl}+\mathcal I_{\rm fb}}
\le1.
}
\]

In the one-current constant-affinity case,

\[
\boxed{
\eta_{\rm th}
=
\frac{h_e J}
{h_c J+\mathcal I_{\rm fb}}
\le1.
}
\]

This is the desired end product: a genuine dimensionless efficiency whose upper bound follows from the second law / feedback inequality.

If free-energy or boundary terms appear, include them.

For finite-time nonstationary dynamics the denominator or numerator may require

\[
\Delta F
\]

or a nonequilibrium free-energy change.

Do not suppress those terms merely to obtain a cleaner formula.

---

# 17. Which information quantity belongs in the thermodynamic bound?

This must be **derived**, not selected for convenience.

Existing quantities include:

### Sensing mutual information
\[
I(N_k;Y_k).
\]

### Action-to-population conditional mutual information
\[
I(U_k;N_{k+1}\mid N_k).
\]

### State-local TE
\[
I(U_k;N_{k+1}\mid N_k=n).
\]

### Directed-information / repeated-feedback sums
Potentially

\[
\sum_k
I(U_k;N_{k+1}\mid N_{\le k},U_{<k}),
\]

or a reduced Markov version.

The second-law derivation must decide which information quantity appears.

It is completely acceptable if:
- the thermodynamic bound uses sensing information;
- the empirical TE remains a separate controller-to-population channel diagnostic.

Do not force the thermodynamic bound to use the existing TE if the trajectory derivation says otherwise.

---

# 18. Forward trajectory probability

For one round, start from

\[
P_F(n,y,u,m)
=
p_k(n)\,
S(y\mid n)\,
\pi(u\mid y)\,
R_u(m\mid n).
\]

For multiple rounds,

\[
\boxed{
P_F[\gamma]
=
p_0(n_0)
\prod_{k=0}^{K-1}
S(y_k\mid n_k)
\pi(u_k\mid y_k)
R_{u_k}(n_{k+1}\mid n_k).
}
\]

For the finite-bias theory use

\[
R_{\mathrm{ADV}}=R_{h_c}.
\]

The reverse experiment must be defined explicitly.

Do not merely write

\[
P_R[\tilde\gamma]
\]

without specifying:
- reversed system protocol;
- treatment of the measurement record;
- treatment of the feedback decisions;
- whether the reverse controller is causal;
- initial distribution of the reverse experiment.

---

# 19. Pathwise entropy production

The fundamental object should be

\[
\boxed{
\Sigma[\gamma]
=
\ln
\frac{P_F[\gamma]}
{P_R[\tilde\gamma]}.
}
\]

Then

\[
\boxed{
\langle \Sigma\rangle
=
D_{\rm KL}(P_F\Vert P_R)
\ge0.
}
\]

Decompose \(\Sigma\) into the cleanest possible terms, ideally

\[
\Sigma
=
\Delta S_{\rm sys}
+
\Sigma_{\rm env}
+
\Sigma_{\rm info},
\]

or an equivalent feedback form.

The exact decomposition depends on the reverse protocol.

The agent must derive the signs.

Do not import a remembered second-law formula without checking that its measurement protocol matches this model.

---

# 20. Required feedback-thermodynamics derivation strategy

Use the literature on repeated discrete feedback and information thermodynamics as guidance, especially the Horowitz–Sandberg framework.

But the derivation must be self-contained for this model.

At minimum:

1. write the forward joint path measure;
2. define a legitimate reverse/reference path measure;
3. compute the log-ratio;
4. identify system entropy change;
5. identify generalized work / affinity-current terms;
6. identify the information term;
7. average;
8. obtain a nonnegative KL divergence;
9. rearrange into a second-law-like inequality;
10. define efficiency from that inequality.

---

# 21. Mechanism-resolved route versus coarse-grained route

There are two acceptable routes.

## Route A — preferred: mechanism-resolved thermodynamics

Track separately:
- ordinary social/voter transitions;
- controller-mediated transitions;
- epistemic/load transitions if introduced.

For each mechanism \(r\), define forward/backward rates or transition probabilities and affinities.

Then define currents

\[
J^{(r)}_{m,n}
=
P_nW^{(r)}_{m,n}
-
P_mW^{(r)}_{n,m},
\]

activities

\[
A^{(r)}_{m,n}
=
P_nW^{(r)}_{m,n}
+
P_mW^{(r)}_{n,m},
\]

and entropy production

\[
\boxed{
\dot\Sigma
=
\frac12
\sum_{r,m,n}
J^{(r)}_{m,n}
\ln
\frac{
P_nW^{(r)}_{m,n}
}{
P_mW^{(r)}_{n,m}
}
\ge0.
}
\]

This route gives the cleanest force-current interpretation.

## Route B — fallback: round-kernel irreversibility

If the mechanism-resolved model becomes ill-defined because the ordinary voter kernel has absorbing/one-way transitions, work with the complete finite-bias round kernel

\[
\bar R_{h_c}(m\mid n)
\]

and define coarse-grained trajectory irreversibility from

\[
\ln
\frac{
p_k(n)\bar R_{h_c}(m\mid n)
}{
p_{k+1}(m)\bar R^{R}_{h_c}(n\mid m)
}.
\]

This may give a rigorous entropy production but a weaker decomposition into control work and load work.

If Route B is used, state clearly that the resulting EP is coarse-grained and can underestimate hidden/microscopic irreversibility.

---

# 22. Ordinary q-voter reversibility problem: do not ignore it

The ordinary voter process can possess absorbing boundaries.

Therefore a fully mechanism-resolved stochastic thermodynamics may encounter transitions whose reverse probability is zero.

The finite-bias controller fixes the controller singularity, but it may not automatically regularize every ordinary-social transition.

The agent must explicitly check:

\[
W_{m,n}>0
\quad\Rightarrow\quad
W_{n,m}>0
\]

for every mechanism used in the entropy-production formula.

If this fails, consider the following options in this order:

1. determine whether the **full policy-mixed round kernel** is nevertheless dynamically reversible on the occupied state space;
2. restrict a finite-time derivation to a mutually accessible communicating class if mathematically legitimate;
3. introduce a minimal spontaneous-revision / trembling parameter
   \[
   \varepsilon>0
   \]
   only if necessary;
4. study the limit
   \[
   \varepsilon\to0^+.
   \]

If \(\varepsilon\) is introduced, it must have a semantic interpretation such as independent reasoning / spontaneous opinion revision.

Do **not** insert an arbitrary \(\varepsilon\) and hide it.

If entropy production diverges as \(\varepsilon\to0\), report that as a genuine singular irreversible limit.

---

# 23. A rigorous information-response bound that should be derived independently

Even before the thermodynamic construction is complete, the current exact kernels imply a useful bounded information-response relation.

At fixed state \(n\),

\[
T(n)
=
JS_{a_n}(R_0,R_1).
\]

Let

\[
\Delta\mu(n)
=
E_{R_1}[x']
-
E_{R_0}[x'].
\]

Since \(x'\in[0,1]\),

\[
|\Delta\mu(n)|
\le
\operatorname{TV}(R_0,R_1).
\]

Weighted Pinsker gives, in bits,

\[
\boxed{
T(n)
\ge
\frac{2a_n(1-a_n)}{\ln2}
\operatorname{TV}(R_0,R_1)^2.
}
\]

Therefore,

\[
\boxed{
T(n)
\ge
\frac{2a_n(1-a_n)}{\ln2}
[\Delta\mu(n)]^2.
}
\]

This yields the rigorous bounded information-response efficiency

\[
\boxed{
\eta_{\rm IR}(n)
=
\frac{
2a_n(1-a_n)[\Delta\mu(n)]^2
}{
(\ln2)T(n)
}
\le1.
}
\]

Tasks:
- prove this carefully;
- check bit/nat conventions;
- generalize from \(R_1\) to finite-bias \(R_{h_c}\);
- derive occupancy-averaged versions;
- determine whether a sharper bound is possible for the birth-death structure.

This is not yet a thermodynamic efficiency, but it provides a rigorous baseline bound that should remain in the final theory.

---

# 24. Desired closed-form calculations

Obtain closed forms as far as possible for \(q=1\).

Priority order:

## 24.1 Finite-bias response
Already targeted:

\[
\boxed{
\Delta\mu_{h_c}(x)
=
[\sigma(h_c)-x]
\left[
1-\left(1-\frac1N\right)^b
\right].
}
\]

Verify exactly.

## 24.2 Mean controller current

Derive

\[
J_c(x;h_c,b,N)
\]

and, if possible, its exact finite-\(N\) whole-round expression.

At minimum derive the expected integrated current over \(b\) controlled updates.

Because

\[
N\Delta x = \Delta n,
\]

the expected net target current produced by the controlled component should be closely related to

\[
N\Delta\mu_{h_c}(x).
\]

Clarify exact distinctions between:
- controlled current;
- total current;
- excess current relative to NoOp.

## 24.3 Activity

Derive

\[
A_c(x;h_c,b,N)
=
E[N_+^c+N_-^c].
\]

Try to obtain a closed form for \(q=1\).

This is useful for kinetic uncertainty relations and for separating:
- directional response;
- total dynamical traffic.

## 24.4 Entropy production

Derive state-local or round-level

\[
\Sigma(x;h_c,b,q_c,\beta,\theta,h_e,\ldots).
\]

Where possible separate

\[
\Sigma
=
\Sigma_{\rm social}
+
\Sigma_{\rm control}
+
\Sigma_{\rm load}
+
\Sigma_{\rm info}.
\]

## 24.5 Work terms

Derive

\[
W_{\rm ctrl}=h_cJ_c
\]

or the correct mechanism-resolved analogue.

Derive

\[
W_{\rm out}=h_eJ
\]

or the correct load-work expression.

## 24.6 Information term

Derive the exact information term appearing in the feedback inequality.

Then compare it to

\[
I(N;Y),
\]

\[
I(U;N'|N),
\]

and the existing state-local TE.

## 24.7 Efficiency

Obtain the cleanest exact formula possible:

\[
\eta_{\rm th}
=
\frac{W_{\rm out}}
{W_{\rm ctrl}+\mathcal I_{\rm fb}+\text{boundary terms}}
\le1.
\]

Then simplify in:
- stationary limit;
- periodic/cyclic limit;
- weak-control limit;
- large-\(N\) limit;
- \(h_c\to\infty\) limit;
- zero-information limit;
- perfect-measurement limit.

The final report should explicitly state which versions are exact and which are asymptotic.

---

# 25. Weak-control expansion

The existing theory has a weak-control scaling of the form

\[
T_{\rm MF}(x)
\approx
\frac{
a(x)[1-a(x)]
}{
2\ln2
}
Nc^2
\frac{
[\Delta f_q(x)]^2
}{
\nu_0(x)
},
\qquad
c=b/N.
\]

Extend this systematically to finite \(h_c\).

Determine whether the finite-bias response modifies

\[
\Delta f_q(x)
\]

by a multiplicative compliance factor or changes the state dependence more substantially.

Derive weak-\(h_c\) and strong-\(h_c\) expansions if useful.

---

# 26. Information ridge under finite bias

Compute

\[
T_{h_c}(n;b,q_c)
\]

over

\[
n=0,\ldots,N
\]

for representative values of

\[
h_c.
\]

Determine how the theoretical state-local information ridge moves with:
- \(b\);
- \(q_c\);
- \(h_c\);
- \(\beta\);
- \(\theta\).

Do not tune these parameters to mimic the empirical heatmap.

The purpose is to understand structural predictions.

---

# 27. Relation to the empirical LLM system

The empirical LLM does not literally obey the classical micro-kernel.

The correct interpretation is

\[
\boxed{
\text{finite-bias q-voter}
=
\text{thermodynamically tractable effective reference model}.
}
\]

The theory should explain:
- resource dependence;
- information-flow structure;
- possible state-local ridges;
- how finite compliance changes the control response;
- what thermodynamic bounds would hold for a process with those transition rules.

It does not have to reproduce every empirical trajectory.

---

# 28. Estimating \(h_c\) from LLM data

If possible, identify an empirical estimate of finite compliance from controlled exposures.

For example, conditional on a controller exposure and a sufficiently resolved pre-state, estimate

\[
P(\text{move toward }Z\mid\mathrm{ADV}),
\]

and

\[
P(\text{move away from }Z\mid\mathrm{ADV}).
\]

A naive effective estimate would be

\[
\hat h_c
=
\ln
\frac{
\hat P(0\to1\mid\mathrm{ADV})
}{
\hat P(1\to0\mid\mathrm{ADV})
}.
\]

But do not use this formula without checking:
- focal-state conditioning;
- ordinary peer effects;
- knowledge state;
- exposure definition;
- whether transitions are truly controller-mediated.

If current logs do not identify this cleanly, say so.

Do not infer \(h_c\) from the same aggregate outcome that the theory is later asked to predict.

---

# 29. Epistemic load estimation

If a load field \(h_e\) is introduced, seek an independently interpretable empirical analogue.

Possible source:

\[
P(\text{move toward truth}\mid\text{knowledge state}, U=\mathrm{NOOP}).
\]

For a two-state logistic approximation, one could define an effective load log-odds ratio.

Again, this is only legitimate if the conditioning is sufficient enough to separate epistemic drive from social exposure.

A particularly important empirical fact is

\[
P(\text{truth}\mid \text{full proof})
\approx1.
\]

Therefore the full-proof state corresponds qualitatively to a very strong truth-directed field.

Do not set \(h_e=\infty\) merely because empirical accuracy is 1 in a finite sample.

Use finite confidence intervals / regularization if estimating.

---

# 30. Memory and coarse-graining

The classical model is Markov in \(n\).

The LLM system is not.

Therefore any thermodynamic quantity computed directly from the observed coarse variable \(n_Z\) should be called a **coarse-grained entropy production / information flow** unless state sufficiency is established.

At minimum, compare:
- state \(n_Z\);
- state \((n_Z,\phi)\);
- state \((n_Z,\kappa)\);
- short trajectory history.

---

# 31. Controller soft policy versus finite bias

Keep these completely separate.

### Soft policy
\[
P(U=\mathrm{ADV}\mid Y)
=
\sigma[\beta(\theta-Y/q_c)].
\]

This controls **whether the controller intervenes**.

### Finite interaction bias
\[
h_c
=
\ln
\frac{\text{toward-target controlled tendency}}
{\text{away-from-target controlled tendency}}.
\]

This controls **how strongly an intervention biases the focal agent**.

### Actuation budget
\[
b
\]

controls **how many update opportunities receive that intervention**.

These three parameters must never be conflated.

---

# 32. Sensing resource versus actuation resource

Current simple interaction accounting is

\[
C_{\rm int}
=
q_c+bP(\mathrm{ADV}).
\]

This remains useful operationally, but it is not automatically thermodynamic work.

Preserve the distinction:

\[
q_c
=
\text{measurement effort / sample size},
\]

\[
b
=
\text{number of actuation opportunities},
\]

\[
h_c
=
\text{thermodynamic/generalized force strength},
\]

\[
h_cJ_c
=
\text{generalized work}.
\]

A later engineering/control efficiency may normalize by \(q_c+bP(\mathrm{ADV})\), but do not call that quantity thermodynamic efficiency.

---

# 33. Candidate efficiency family

The final theory may contain several clearly named efficiencies.

## 33.1 Thermodynamic efficiency
Only if derived from a second-law-like bound:

\[
\boxed{
\eta_{\rm th}
=
\frac{W_{\rm out}}
{W_{\rm ctrl}+\mathcal I_{\rm fb}+\cdots}
\le1.
}
\]

## 33.2 Information-response efficiency
Rigorous but non-thermodynamic:

\[
\boxed{
\eta_{\rm IR}(n)
=
\frac{
2a_n(1-a_n)[\Delta\mu(n)]^2
}{
(\ln2)T(n)
}
\le1.
}
\]

## 33.3 Information utilization
Descriptive:

\[
\eta_{\rm use}
=
\frac{\Delta\mu}{T}.
\]

Do not claim a universal bound.

## 33.4 Information yield per interaction
Descriptive:

\[
Y_I
=
\frac{T}
{q_c+bP(\mathrm{ADV})}.
\]

## 33.5 Response per interaction
Descriptive:

\[
\eta_{\rm int}
=
\frac{\Delta\mu}
{q_c+bP(\mathrm{ADV})}.
\]

These descriptive ratios are useful but must not be confused with \(\eta_{\rm th}\).

---

# 34. Check fluctuation relations

If the pathwise EP is well defined, test whether the construction yields

\[
\left\langle e^{-\Sigma}\right\rangle=1
\]

or the appropriate feedback-corrected integral fluctuation theorem.

This is an important internal consistency check.

If an information term appears pathwise,

\[
\Sigma+\mathcal I
\]

may be the quantity entering the exponential identity.

Derive, do not assume.

---

# 35. Current/activity bounds and TUR/KUR connection

Once finite affinities and reversible transitions exist, examine whether the model admits thermodynamic or kinetic uncertainty relations.

Possible targets include

\[
\frac{\mathrm{Var}(\mathcal J)}
{\langle \mathcal J\rangle^2}
\Sigma
\ge
\text{constant},
\]

or activity-based analogues.

The project already tracks:
- current means;
- variances;
- Fano factors;
- SNR\(^2\);
- activity.

Therefore a clean finite-bias thermodynamics could unify existing current statistics with entropy production.

This is secondary to obtaining the main efficiency bound.

Do not allow TUR derivations to delay the primary objective.

---

# 36. Required limiting cases

Every proposed formula must be tested in at least these limits.

### No actuation
\[
b=0.
\]

Expected:
\[
J_c=0,\qquad W_c=0,\qquad T=0.
\]

### No control bias
\[
h_c=0.
\]

Interpret carefully: the action may occur, but the controlled interaction has no target preference.

### Infinite bias
\[
h_c\to+\infty.
\]

Recover the existing one-sided controller response and kernel.

### Deterministic policy
\[
\beta\to\infty.
\]

Conditional action entropy may collapse away from the sensing threshold.

### Random policy
\[
\beta\to0.
\]

Action becomes nearly state independent.

### Perfect sensing
\[
q_c=N.
\]

### No load
\[
h_e=0.
\]

There is no motor-like useful work against a competing force.

### Stall condition
Find parameter values for which

\[
J=0.
\]

This should define a thermodynamic control threshold.

A clean stall relation involving \(h_c,h_e,\mathcal I\) would be especially valuable.

---

# 37. Desired theoretical result

A particularly elegant final structure would be

\[
\boxed{
\text{feedback information}
+
\text{control work}
\ge
\text{useful work against epistemic load}
}
\]

or, in the simplest steady one-current form,

\[
\boxed{
\mathcal I_{\rm fb}
+
h_cJ
\ge
h_eJ.
}
\]

Then

\[
\boxed{
\eta_{\rm th}
=
\frac{h_eJ}
{h_cJ+\mathcal I_{\rm fb}}
\le1.
}
\]

The agent should aim for this form only if it follows from a rigorous derivation.

If the exact bound contains extra terms, keep them.

Beauty is subordinate to correctness.

---

# 38. Numerical validation requirements

Every analytic result must be checked numerically.

At minimum:

1. construct the finite-\(N\) transition matrices;
2. verify row normalization;
3. verify positivity;
4. verify detailed-balance identities where claimed;
5. compare analytic mean response to matrix propagation;
6. compare \(h_c\to\infty\) to the existing exact controller kernel;
7. compare exact TE from distributions to direct MI computation;
8. simulate trajectories and verify current means;
9. verify entropy-production nonnegativity;
10. verify any integral fluctuation theorem;
11. verify the efficiency never exceeds 1 within numerical tolerance when theorem assumptions hold.

Use multiple random parameter points, not only the Study 04 values.

---

# 39. Symbolic algebra requirements

Use symbolic algebra where it genuinely helps.

Try to derive closed forms for:
- \(\Delta\mu_{h_c}(x)\);
- expected controller current;
- controlled activity;
- stationary distribution of controlled-only kernel;
- elementary affinity;
- stall condition;
- weak-control TE;
- simple efficiency in the constant-load model.

If a full finite-\(N\) closed form becomes unreadable, provide:
1. an exact finite sum / matrix expression;
2. a clean asymptotic approximation;
3. numerical verification.

“Closed form as close as possible” is preferable to forcing an artificial expression.

---

# 40. Deliverables

Create a self-contained theory package containing at least:

## 40.1 Main derivation report
Suggested name:

`20082026_finite_bias_feedback_thermodynamics_derivation.md`

It must contain:
- definitions;
- all assumptions;
- derivations;
- exact formulas;
- bounds;
- limiting cases;
- interpretation;
- unresolved issues.

## 40.2 Formula summary
Suggested name:

`20082026_finite_bias_thermodynamics_formula_sheet.md`

This should be concise and list:
- microscopic kernels;
- round kernels;
- affinities;
- current/activity;
- entropy production;
- information terms;
- work;
- efficiencies;
- bounds.

## 40.3 Validation code

Add a standalone analysis script or module that:
- evaluates the exact finite-\(N\) formulas;
- checks limits;
- produces comparison plots;
- reports numerical violations if any.

Do not modify the empirical experiment code unless necessary for reading existing outputs.

## 40.4 Plots

At minimum:

1. \(\Delta\mu_{h_c}(x)\) versus \(x\) for several \(h_c\);
2. state-local \(T_{h_c}(x,b)\);
3. entropy production versus \(x,b,h_c\);
4. current versus \(h_c-h_e\);
5. efficiency versus control strength;
6. efficiency versus information resource;
7. stall boundary;
8. optional current-precision / TUR plot.

---

# 41. Red flags / forbidden shortcuts

Do not do any of the following.

### 41.1 Do not call transfer entropy “energy”
TE is information.

### 41.2 Do not call \(b\) “work”
\(b\) is an actuation count/resource.

### 41.3 Do not call \(q_c\) “thermodynamic cost”
It is a sensing sample size unless a cost model is derived.

### 41.4 Do not invent an effective energy only to make detailed balance work
The microscopic transition semantics must justify the affinity.

### 41.5 Do not hide zero reverse probabilities
If a reverse transition is impossible, explicitly handle the singularity.

### 41.6 Do not tune \(h_c\), \(h_e\), \(\beta\), or \(\theta\) to reproduce an attractive empirical plot
Theory parameters require independent interpretation.

### 41.7 Do not claim the LLM is Markov in \(n_Z\)
It is not.

### 41.8 Do not claim a phase transition from an interior ridge
An information ridge is not automatically a phase transition.

### 41.9 Do not assume continuous time is more physical
The empirical controller is round-based.

### 41.10 Do not sacrifice the current exact theory
The finite-bias construction must reduce to the existing model in a controlled limit.

---

# 42. Success criteria

The thermodynamic extension is successful if the agent can deliver all of the following.

### Minimum success
1. finite-bias reversible controlled kernel;
2. exact \(q=1\) response;
3. finite local affinity;
4. pathwise entropy-production definition;
5. numerical EP validation;
6. rigorous information-response bound.

### Strong success
Additionally:
7. mechanism-resolved control current and activity;
8. load/epistemic affinity;
9. derived feedback second law;
10. bounded thermodynamic efficiency.

### Ideal success
Additionally:
11. clean closed-form or nearly closed-form \(\eta_{\rm th}\);
12. stall condition;
13. finite-\(N\) state-local efficiency landscape;
14. connection to empirical Study 04 without fitting away discrepancies;
15. optional TUR/KUR relation for current precision.

---

# 43. Failure criteria / when to stop

Stop and report clearly if obtaining a “thermodynamic efficiency” requires any of the following:

- an arbitrary reverse process;
- an arbitrary energy function unrelated to the transition rules;
- a hidden regularization parameter that controls the answer;
- an information measure chosen only because it gives the desired inequality;
- changing the empirical LLM experiment;
- assuming away the observed memory dependence;
- treating the wrong-target q-voter as if it already contains a truth force when it does not.

In that case, preserve the successful pieces:
- finite-bias control theory;
- entropy production if legitimate;
- information-response bound;
- descriptive efficiencies.

Do not overclaim.

---

# 44. Recommended order of work

Follow this order strictly.

## Phase I — finite-bias mechanics
1. derive \(K_{h_c}\);
2. prove local detailed balance;
3. derive stationary measure;
4. derive exact \(q=1\) response;
5. construct exact \(R_{h_c}\);
6. reproduce \(h_c\to\infty\) existing theory.

## Phase II — information
7. derive \(T_{h_c}(n)\);
8. reproduce state-local information landscapes;
9. prove the information-response bound;
10. study \(h_c\) dependence.

## Phase III — irreversibility
11. define forward trajectories;
12. define reverse trajectories;
13. derive pathwise EP;
14. check absorbing-boundary issues;
15. validate fluctuation relation.

## Phase IV — load and work
16. introduce the minimal epistemic/load mechanism;
17. derive its affinity;
18. define controller work;
19. define load/output work;
20. derive stall condition.

## Phase V — feedback second law
21. derive the correct information term;
22. derive the averaged inequality;
23. rearrange into useful-work bound;
24. define \(\eta_{\rm th}\).

## Phase VI — closed forms and empirical bridge
25. simplify \(q=1\) efficiency;
26. obtain weak-control / large-\(N\) limits;
27. compare theory surfaces with Study 04;
28. identify which parameters can be estimated independently from LLM trajectories.

Only then consider \(q\ge2\).

---

# 45. Final conceptual target

The desired theory should make the following statement precise:

> A feedback controller obtains partial information about an interacting population, decides whether to intervene, and applies a finite social affinity through a limited number of microscopic interactions. The resulting opinion current can oppose an epistemic load. The forward/reverse trajectory asymmetry defines entropy production, while the feedback record supplies an information resource. A second-law-like inequality limits how much useful population-level control can be obtained from the combined informational and actuation resources.

The ideal final equation is structurally of the form

\[
\boxed{
W_{\rm out}
\le
W_{\rm ctrl}
+
\mathcal I_{\rm fb}
}
\]

leading to

\[
\boxed{
0\le
\eta_{\rm th}
=
\frac{W_{\rm out}}
{W_{\rm ctrl}+\mathcal I_{\rm fb}}
\le1.
}
\]

But the exact formula must come from the derivation.

The theory should be elegant because the microscopic assumptions make it elegant—not because terms were chosen after the fact to produce a pretty bound.
