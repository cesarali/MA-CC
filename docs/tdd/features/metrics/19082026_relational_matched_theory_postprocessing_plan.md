# Matched Classical-Theory Post-processing for Relational Round-Feedback Experiments

**Purpose:** Implementation plan for extending the existing relational round-feedback MI/CMI analysis so that **every completed run is automatically compared against the matched finite-\(N\) controlled q-voter theory**.

**Core requirement:** This must live **inside the existing MI/CMI post-processing and report bundle**. Do **not** create a parallel theory-analysis pipeline or a separate top-level theory report. The same command/report that currently produces the relational MI/CMI results should also compute, store, and display the matched classical-theory quantities described below.

---

## 1. Scientific objective

The relational LLM experiments and the classical controlled q-voter should be treated as two realizations of the same high-level control protocol:

\[
\text{current population state}
\rightarrow
\text{finite controller sensing}
\rightarrow
\text{stochastic ADVOCATE/NO\_OP decision}
\rightarrow
\text{budgeted actuation}
\rightarrow
\text{next population state}.
\]

The controller protocol is matched through the run parameters

\[
(N,q,q_c,b,\beta,\theta),
\]

where:

- \(N\): population size,
- \(q\): number of social inputs used in the population update,
- \(q_c\): controller sensor sample size,
- \(b\): number of controlled microscopic positions on an ADVOCATE round,
- \(c=b/N\): actuation coverage,
- \(\beta\): softness / inverse-noise parameter of the controller decision,
- \(\theta\): controller intervention threshold.

The **population-response kernel** is what differs:

- classical reference: explicit q-voter/unanimity kernel;
- relational experiment: implicit LLM response with persistent knowledge / epistemic memory.

The point of this post-processing is therefore not to claim that the LLM *is* a q-voter. It is to ensure that **every empirical information-flow result has an immediately available classical reference at the same controller parameters**.

The analysis should answer, for each run/cell:

1. Does the empirical controller policy match the theoretical controller policy?
2. Is the empirical response to ADVOCATE qualitatively and quantitatively similar to the classical response?
3. Is the empirical controller-to-population CMI/TE of the same scale and state dependence as the exact classical TE?
4. Where does the LLM system depart from the classical reference?
5. Can later epistemic conditioning (\(\phi,\kappa,\ldots\)) explain those departures?

---

## 2. Non-negotiable implementation constraints

### 2.1 Reuse existing machinery

Inspect and reuse the existing implementation for:

- relational round-feedback analysis;
- direct-counting MI/CMI estimators;
- whole-episode bootstrap;
- policy-conditional nulls;
- sensor-permutation nulls;
- support/overlap diagnostics;
- entropy ceilings;
- signed response;
- per-cell and pooled aggregation;
- current MI/CMI report generation.

Do **not** reimplement any of these.

The theory layer should be a deterministic calculation/adaptor that plugs into the same analysis workflow.

### 2.2 Same reports, not a separate analysis product

Wherever the current relational MI/CMI report writes summary statistics, append the theory quantities and empirical-theory comparison quantities there.

If state-resolved curves require an auxiliary table because they cannot fit in a single summary row, that table may be written **inside the same analysis/report directory**, and the main MI report must reference/include it. Do not create a separate `theory_analysis/` tree disconnected from the MI results.

### 2.3 Per-cell and pooled

Compute theory comparisons:

- for every individual grid cell;
- for the pooled analysis if the existing relational analysis already produces a pooled result.

Do not pool cells with different \((N,q,q_c,b,\beta,\theta)\) into one theoretical reference. If pooled cells differ in theory parameters, either stratify by unique parameter tuple or clearly mark the pooled theory comparison as not applicable.

### 2.4 No Monte Carlo theory when the exact finite-\(N\) theory applies

The finite-\(N\) q-voter reference is deterministic once the parameters are fixed. Use the exact formulas/kernels below. Do not simulate the q-voter merely to estimate quantities that are analytically/numerically exact.

---

# 3. State used for the theory comparison

For the main comparison use the **controller target count**

\[
n \equiv n_Z \in \{0,\ldots,N\},
\qquad
x=\frac{n}{N}.
\]

For the relational experiment, \(n_Z\) is the number of agents currently voting for the controller's target semantic relation.

This target/non-target projection is intentional. The relational task has three answer options, whereas the classical reference is binary. The comparable coarse state is therefore:

\[
\text{target} \quad \text{vs} \quad \text{not target}.
\]

Do not silently use the full three-option occupation vector in the binary classical formulas.

Record in the report that the classical comparison is a **target-count coarse graining** of the relational population.

---

# 4. Exact controller sensing and policy

## 4.1 Hypergeometric sensor

The controller samples \(q_c\) distinct agents without replacement. At true target count \(n\), let

\[
Y_k=y
\]

be the number of sampled agents currently supporting the controller target.

The exact sensor law is

\[
\boxed{
S(y\mid n)
=
\Pr(Y_k=y\mid N_k=n)
=
\frac{\binom{n}{y}\binom{N-n}{q_c-y}}
{\binom{N}{q_c}}
}
\]

for valid \(y\), with invalid combinations contributing zero.

## 4.2 Soft stochastic controller

Given measured target fraction \(y/q_c\),

\[
\boxed{
\Pr(U_k=\mathrm{ADVOCATE}\mid Y_k=y)
=
\sigma\!\left[
\beta\left(\theta-\frac{y}{q_c}\right)
\right]
}
\]

with

\[
\sigma(z)=\frac{1}{1+e^{-z}}.
\]

The exact state-conditioned advocacy probability is

\[
\boxed{
a_n
=
\Pr(U_k=\mathrm{ADVOCATE}\mid N_k=n)
=
\sum_y
S(y\mid n)
\sigma\!\left[
\beta\left(\theta-\frac{y}{q_c}\right)
\right].
}
\]

This quantity should be calculated for all \(n=0,\ldots,N\).

### Required empirical comparison

From the relational rounds estimate

\[
\widehat a_n
=
\Pr_{\rm emp}(U_k=\mathrm{ADVOCATE}\mid n_{Z,k}=n)
\]

where support permits.

The report should include:

- exact \(a_n\);
- empirical \(\widehat a_n\);
- counts of ADVOCATE/NO\_OP observations at each \(n\);
- empirical minus theoretical policy residual;
- an aggregate calibration metric such as occupancy-weighted MAE and/or RMSE.

This policy comparison is primarily a **controller implementation/calibration check**. If the controller machinery is matched, empirical action frequencies should fluctuate around the exact \(a_n\).

---

# 5. Classical microscopic q-voter kernels

The classical population consists of binary target/non-target opinions. The focal agent switches only when the \(q\) social inputs unanimously support the opposite opinion.

Let \(K_0\) denote one ordinary microscopic update and \(K_1\) one controlled microscopic update.

## 5.1 Ordinary microscopic update

\[
\boxed{
K_0(n+1\mid n)
=
\frac{N-n}{N}
\frac{\binom{n}{q}}{\binom{N-1}{q}}
}
\]

and

\[
\boxed{
K_0(n-1\mid n)
=
\frac{n}{N}
\frac{\binom{N-n}{q}}{\binom{N-1}{q}}.
}
\]

The diagonal probability is

\[
K_0(n\mid n)
=
1-K_0(n+1\mid n)-K_0(n-1\mid n).
\]

## 5.2 Controlled microscopic update

At a controlled position, the controller occupies one social-input slot and advocates for the target. Therefore only \(q-1\) ordinary target supporters are needed to produce a non-target \(\to\) target switch:

\[
\boxed{
K_1(n+1\mid n)
=
\frac{N-n}{N}
\frac{\binom{n}{q-1}}{\binom{N-1}{q-1}}
}
\]

and

\[
\boxed{
K_1(n-1\mid n)=0.
}
\]

Again,

\[
K_1(n\mid n)=1-K_1(n+1\mid n).
\]

Use robust combinatorial conventions so impossible transitions have probability zero.

---

# 6. Exact whole-round kernels

The empirical information analysis is performed on the **round clock**, so the classical reference must also use whole-round kernels.

## 6.1 NO_OP round

With \(N\) ordinary microscopic positions,

\[
\boxed{
R_0=K_0^N.
}
\]

## 6.2 ADVOCATE round

On an ADVOCATE round, exactly \(b\) of the \(N\) positions are controlled and are uniformly preallocated.

Do **not** approximate this as independent Bernoulli control of each position.

Use the exact dynamic recursion over the number of positions processed and controlled positions already used.

Initialize

\[
F_{0,0}=I.
\]

After \(r\) positions, if \(j\) controlled positions have already occurred, the probability that position \(r+1\) is controlled is

\[
\frac{b-j}{N-r},
\]

and the probability it is ordinary is

\[
\frac{N-b-r+j}{N-r}.
\]

Therefore update

\[
\boxed{
F_{r+1,j+1}
\mathrel{+}=
\frac{b-j}{N-r}
F_{r,j}K_1
}
\]

and

\[
\boxed{
F_{r+1,j}
\mathrel{+}=
\frac{N-b-r+j}{N-r}
F_{r,j}K_0.
}
\]

After all \(N\) positions,

\[
\boxed{
R_1=F_{N,b}.
}
\]

This gives the exact finite-\(N\) ADVOCATE whole-round kernel, averaged over all uniformly preallocated schedules without enumerating them.

Cache \(R_0\), \(R_1\), and \(a_n\) by the unique theory-parameter tuple

\[
(N,q,q_c,b,\beta,\theta)
\]

so repeated cells do not recompute them unnecessarily.

---

# 7. Exact state-local theoretical transfer entropy

At fixed current state \(n\), define the action-marginalized next-state distribution

\[
\boxed{
M_n(m)
=
[1-a_n]R_0(m\mid n)
+
a_nR_1(m\mid n).
}
\]

The exact state-local transfer entropy is

\[
\boxed{
\begin{aligned}
T_{\rm qv}(n)
=&
[1-a_n]
\sum_m
R_0(m\mid n)
\log_2
\frac{R_0(m\mid n)}{M_n(m)}
\\
&+
a_n
\sum_m
R_1(m\mid n)
\log_2
\frac{R_1(m\mid n)}{M_n(m)}.
\end{aligned}
}
\]

Equivalently,

\[
\boxed{
T_{\rm qv}(n)
=
\mathrm{JS}_{a_n}
\left(
R_0(\cdot\mid n),
R_1(\cdot\mid n)
\right).
}
\]

This is an exact theoretical quantity. There is no empirical MI estimator in this calculation.

It must obey

\[
\boxed{
0\le T_{\rm qv}(n)\le h_2(a_n)
}
\]

where

\[
h_2(a)
=
-a\log_2a-(1-a)\log_2(1-a).
\]

Add numerical tests for normalization and this ceiling.

---

# 8. Occupancy-matched theoretical TE

This is one of the most important outputs.

The scalar empirical CMI is collected over the population states actually visited by the LLM run. Therefore the fairest primary theory scalar should use the **same empirical state occupancy**.

## 8.1 Round-specific empirical occupancy

Let

\[
\widehat P_k(n)
\]

be the empirical distribution of \(n_{Z,k}\) at the beginning of round \(k\), estimated across episodes in the cell.

Then calculate

\[
\boxed{
T_{{\rm qv},k}^{\rm emp-occ}
=
\sum_n
\widehat P_k(n)T_{\rm qv}(n).
}
\]

## 8.2 Finite-horizon empirical-occupancy theory value

For \(K\) rounds,

\[
\boxed{
\overline T_{\rm qv}^{\rm emp-occ}
=
\frac{1}{K}
\sum_{k=0}^{K-1}
T_{{\rm qv},k}^{\rm emp-occ}.
}
\]

This should be the **primary scalar theoretical comparator** shown beside the empirical target-actuation CMI.

It answers:

> If the classical q-voter experienced the same controller parameters and we evaluated its local information channel over the states that the LLM population actually visited, how much TE would the classical kernel predict?

## 8.3 Optional secondary theoretical-self-occupancy calculation

It is also useful, but secondary, to propagate the classical closed-loop dynamics under its own state occupancy.

The closed-loop classical kernel is

\[
R(m\mid n)
=
[1-a_n]R_0(m\mid n)+a_nR_1(m\mid n).
\]

Given an initial distribution \(P_0(n)\),

\[
P_{k+1}(m)
=
\sum_nP_k(n)R(m\mid n),
\]

and

\[
T_{{\rm qv},k}^{\rm self}
=
\sum_nP_k(n)T_{\rm qv}(n).
\]

If implemented, initialize \(P_0\) from the empirical initial-state distribution unless a cell defines a more natural exact initial law.

**Label this clearly as a secondary quantity.** Do not confuse:

- classical TE evaluated on LLM-visited states; and
- classical TE under the classical process's own evolving occupancy.

The first is the direct matched local-channel comparison. The second reveals differences in where the two systems travel through state space.

---

# 9. Empirical target CMI to place beside theory

Retain the existing empirical quantity unchanged:

\[
\boxed{
T_{\rm emp}
=
I(U_k;n_{Z,k+1}\mid n_{Z,k}).
}
\]

Use the existing estimator, bootstrap, null model, entropy ceiling, and support diagnostics.

Do not replace or modify the estimator in order to make it agree with theory.

The report should place directly beside it:

\[
\overline T_{\rm qv}^{\rm emp-occ}.
\]

Also record diagnostic comparisons:

\[
\boxed{
\Delta T
=
T_{\rm emp}
-
\overline T_{\rm qv}^{\rm emp-occ}
}
\]

and, where the denominator is safely nonzero,

\[
\boxed{
\rho_T
=
\frac{T_{\rm emp}}
{\overline T_{\rm qv}^{\rm emp-occ}}.
}
\]

**Do not call \(\rho_T\) an efficiency.** For now it is only an empirical/classical information-channel ratio.

If the theoretical denominator is zero or numerically negligible, report the ratio as undefined rather than forcing a number.

---

# 10. Exact q=1 response reference

For \(q=1\), the theory has a particularly clean exact mean-response result.

Under NO_OP,

\[
\boxed{
\mathbb E[x_{k+1}\mid x_k=x,\mathrm{NO\_OP}]
=
x.
}
\]

Under ADVOCATE with exactly \(b\) controlled positions,

\[
\boxed{
\mathbb E[x_{k+1}\mid x_k=x,\mathrm{ADVOCATE}]
=
1-(1-x)
\left(1-\frac1N\right)^b.
}
\]

Therefore the exact theoretical action-induced mean separation is

\[
\boxed{
\Delta\mu_{\rm qv}^{(q=1)}(x)
=
(1-x)
\left[
1-
\left(1-\frac1N\right)^b
\right].
}
\]

For \(b=cN\) and large \(N\),

\[
\Delta\mu_{\rm qv}^{(q=1)}(x)
\rightarrow
(1-x)(1-e^{-c}).
\]

### Empirical response

For each sufficiently supported target state \(n\), calculate from the relational rounds

\[
\boxed{
\Delta\mu_{\rm emp}(n)
=
\mathbb E[x_{k+1}-x_k
\mid U_k=\mathrm{ADVOCATE},n_{Z,k}=n]
-
\mathbb E[x_{k+1}-x_k
\mid U_k=\mathrm{NO\_OP},n_{Z,k}=n].
}
\]

At fixed \(n\), this is equivalent to comparing the two conditional next-state means.

For \(q=1\), report side by side:

- \(x=n/N\);
- \(\Delta\mu_{\rm emp}(n)\);
- \(\Delta\mu_{\rm qv}^{(q=1)}(x)\);
- empirical minus theory residual;
- number of ADVOCATE samples;
- number of NO_OP samples;
- existing dual-action/support flag;
- empirical bootstrap interval if available from the existing pipeline.

Also produce an occupancy-weighted response comparison if support permits.

This response comparison is especially important because CMI itself has no sign. The response tells us whether the observed information channel acts in the controller's intended direction.

---

# 11. General-q mean-field reference

For general \(q\), retain the exact finite-\(N\) kernels as the primary theoretical reference.

The following large-\(N\) expressions are useful as interpretable diagnostics.

Ordinary drift:

\[
\boxed{
f_0(x)
=
(1-x)x^q
-
x(1-x)^q.
}
\]

Controlled-position drift:

\[
\boxed{
f_1(x)
=
(1-x)x^{q-1}.
}
\]

Actuation advantage:

\[
\boxed{
\Delta f_q(x)
=
f_1(x)-f_0(x)
=
x^{q-1}(1-x)^2
+
x(1-x)^q.
}
\]

With \(c=b/N\), the weak-control mean separation is approximately

\[
\boxed{
\Delta\mu(x)
\simeq
c\,\Delta f_q(x).
}
\]

Ordinary local noise coefficient:

\[
\boxed{
\nu_0(x)
=
(1-x)x^q
+
x(1-x)^q
-
f_0(x)^2.
}
\]

The weak-separation TE approximation is

\[
\boxed{
T_{\rm MF}(x)
\simeq
\frac{a(x)[1-a(x)]}{2\ln 2}
\frac{
Nc^2[\Delta f_q(x)]^2
}{
\nu_0(x)
}.
}
\]

This should be reported as an **approximation / scaling diagnostic**, never in place of the exact finite-\(N\) \(T_{\rm qv}(n)\).

If the approximation is evaluated, clearly identify regimes where it is not expected to be reliable, especially support-opening/boundary cases such as \(q=1,n=0\).

---

# 12. Resource coordinates to record with every result

Every MI/theory summary row should include the controller resource coordinates

\[
\boxed{
r_{\rm sense}=\frac{q_c}{N},
\qquad
r_{\rm act}=\frac{b}{N}=c.
}
\]

Also retain the raw values \(q_c\) and \(b\).

For the current Study 03 pilot these happen to satisfy \(q_c=b=12\), but the code must not assume equality.

Also record:

- empirical ADVOCATE fraction \(\widehat p_{\rm ADV}\);
- theoretical occupancy-weighted ADVOCATE probability;
- realized average actuation slots per round
  \[
  b\,\widehat p_{\rm ADV};
  \]
- sensing observations per round \(q_c\).

These are resource descriptors only. **Do not yet define a thermodynamic or information efficiency from them.**

---

# 13. Memory-aware empirical results remain empirical extensions

Retain all existing/new empirical epistemic-conditioned quantities, e.g.

\[
I(U_k;n_{Z,k+1}\mid n_{Z,k},E_k),
\]

\[
I(U_k;n_{Z,k+1}\mid n_{Z,k},\phi_k^{\rm bin}),
\]

\[
I(U_k;n_{Z,k+1}\mid n_{Z,k},\kappa_k^{\rm bin}),
\]

and the susceptible-fraction analogue.

Do **not** modify the classical q-voter theory by inserting \(\phi\), \(\kappa\), or \(E_k\) ad hoc. The q-voter is intentionally the memoryless classical baseline.

The scientific interpretation should be:

\[
\text{classical reference}
\quad\longrightarrow\quad
\text{LLM empirical channel}
\quad\longrightarrow\quad
\text{epistemic conditioning explains departures}.
\]

A useful optional diagnostic is to stratify the empirical response by \(\phi\)- or \(\kappa\)-bin and compare each stratum with the same classical \(n\)-dependent response reference. Label this as a stratified empirical comparison, not as a new theoretical formula.

---

# 14. Required report layout

The existing MI/CMI report should gain a section resembling the following.

## A. Empirical information channel

Keep the existing fields:

- empirical target-actuation CMI;
- bootstrap CI;
- policy-conditional null;
- null \(p\)-value / exceedance measure;
- entropy ceiling;
- action entropy;
- support diagnostics;
- signed response;
- memory-aware CMI variants;
- sensing MI and other existing information measures.

## B. Matched classical q-voter reference

Add:

- `theory_N`
- `theory_q`
- `theory_qc`
- `theory_b`
- `theory_c`
- `theory_beta`
- `theory_theta`
- `theory_sensing_fraction`
- `theory_actuation_fraction`
- exact theoretical \(a_n\) curve reference
- exact theoretical \(T_{\rm qv}(n)\) curve reference
- occupancy-matched finite-horizon theoretical TE
- optional classical-self-occupancy finite-horizon TE
- for \(q=1\): occupancy-matched exact theoretical response
- optional mean-field TE diagnostic

## C. Empirical-vs-theory comparison

Add at minimum:

- empirical target CMI;
- occupancy-matched classical TE;
- CMI minus classical TE;
- CMI / classical TE diagnostic ratio;
- empirical signed response;
- matched classical response;
- response residual;
- policy calibration MAE/RMSE;
- state-support warning flags.

The report prose should explicitly distinguish:

1. **controller mismatch** — empirical action policy does not follow theoretical \(a_n\);
2. **population-kernel mismatch** — controller is calibrated but empirical response/TE differs from q-voter;
3. **occupancy mismatch** — LLM and classical process visit different parts of state space.

These are scientifically different phenomena.

---

# 15. State-resolved tables and plots

Within the **same MI report bundle**, produce state-resolved diagnostics.

## 15.1 Policy plot/table

Horizontal axis:

\[
x=n_Z/N.
\]

Show:

- exact \(a_n\);
- empirical \(\widehat a_n\);
- empirical uncertainty/support.

## 15.2 Response plot/table

Show:

- empirical \(\Delta\mu_{\rm emp}(n)\);
- exact q=1 response when \(q=1\), otherwise exact-kernel conditional mean separation if convenient;
- mean-field response as an optional dashed/secondary reference;
- support counts.

## 15.3 Local TE plot/table

Show:

- exact \(T_{\rm qv}(n)\);
- empirical state-local information estimate only if the existing estimator/support makes this statistically defensible.

Do not manufacture noisy per-state empirical MI estimates merely to have a curve. If support is insufficient, show only the exact local theory curve plus the empirical occupancy histogram and the global empirical CMI.

## 15.4 Occupancy plot

Show the empirical state occupancy over \(x=n_Z/N\). If classical self-evolution is computed, overlay the corresponding classical occupancy separately.

This plot is important because two systems can have similar local control laws but very different finite-horizon TE simply because they visit different regions.

---

# 16. Bootstrap treatment of theory comparisons

The exact local theory curve has no sampling uncertainty.

However, the **empirical-occupancy-weighted** theoretical scalar depends on the sampled LLM occupancy.

Reuse the existing whole-episode bootstrap. For each bootstrap replicate:

1. resample episodes exactly as the current analysis already does;
2. recompute empirical occupancy \(\widehat P_k(n)\);
3. recompute
   \[
   \overline T_{\rm qv}^{\rm emp-occ};
   \]
4. recompute empirical CMI using the existing estimator;
5. compute the replicate residual
   \[
   \Delta T^{(r)}
   =
   T_{\rm emp}^{(r)}
   -
   \overline T_{\rm qv}^{(r),\rm emp-occ};
   \]
6. optionally compute the ratio when numerically safe.

This gives a meaningful confidence interval for the **empirical-vs-theory difference**, while correctly treating the local theory itself as exact.

Do the analogous thing for an occupancy-weighted response comparison if implemented.

Do not bootstrap the exact \(R_0,R_1,T(n)\) curves themselves.

---

# 17. Support and identifiability

The existing overlap diagnostics remain mandatory.

For empirical response/CMI comparisons, report:

- total observations;
- number of conditioning states;
- singleton fraction;
- dual-action-state fraction;
- fraction of observations lying in conditioning states containing both ADVOCATE and NO_OP;
- per-state ADVOCATE and NO_OP counts where state-resolved quantities are shown.

The theory exists for every state, but the empirical estimate does not.

Do not interpret a large empirical/theory residual in a state with effectively no dual-action support.

Use the existing report's sparsity flags rather than inventing a second unrelated support criterion.

---

# 18. Special handling of deterministic/open-loop controller runs

The postprocessor should also work on older open-loop runs.

If the controller always chooses ADVOCATE or always chooses NO_OP, then empirically there is no action variation and

\[
I(U_k;n_{Z,k+1}\mid n_{Z,k})=0
\]

by construction.

For such runs:

- still compute/report the classical response reference;
- still record \(q_c,b,c\) where meaningful;
- clearly mark TE/CMI comparison as non-identifiable or degenerate because \(H(U\mid n)=0\);
- do not interpret zero TE as zero behavioral control.

This distinction is important because the older open-loop studies are response/susceptibility experiments, not stochastic-feedback TE experiments.

---

# 19. Tests

Add tests covering at least:

1. hypergeometric sensor probabilities sum to 1;
2. \(a_n\in[0,1]\);
3. \(K_0\) and \(K_1\) rows sum to 1;
4. \(R_0\) and \(R_1\) rows sum to 1;
5. \(R_1\) uses exactly-\(b\) preallocated-control semantics, not Bernoulli-\(c\) semantics;
6. \(T_{\rm qv}(n)\ge0\);
7. \(T_{\rm qv}(n)\le h_2(a_n)\);
8. \(T_{\rm qv}(n)=0\) when \(a_n\in\{0,1\}\);
9. \(T_{\rm qv}(n)=0\) when \(R_0(\cdot|n)=R_1(\cdot|n)\);
10. q=1 exact mean response agrees numerically with the exact round kernels;
11. occupancy-weighted TE agrees with direct summation;
12. episode-bootstrap resampling changes only empirical occupancy/empirical estimates, not the local exact theory curve;
13. existing MI/CMI outputs are unchanged except for intentionally appended fields/sections;
14. existing analysis commands remain backward compatible.

---

# 20. Expected analysis command behavior

The user should continue to run the existing relational round-feedback analysis command, e.g.

```bash
python -m mas_cc.cli.main analysis relational-round-feedback --run-dir <run-dir>
```

The same command should now produce:

\[
\boxed{
\text{existing MI/CMI results}
+
\text{matched classical theory}
+
\text{empirical-vs-theory diagnostics}.
}
\]

Do not require a second manual theory command for routine analysis.

If a flag is technically necessary for backward compatibility, theory comparison should default to **on** whenever the run contains the required controller parameters, and the report should clearly state if theory comparison was skipped and why.

---

# 21. Minimum summary block to add to each MI report

Each cell's report should contain a compact summary similar to:

```text
MATCHED CLASSICAL REFERENCE
---------------------------
N=...
q=...
q_c=...
b=...
c=b/N=...
beta=...
theta=...

Empirical target CMI:
    I(U_k ; n_Z,k+1 | n_Z,k) = ...
    bootstrap 95% CI = ...
    null = ...
    null exceedance/p = ...

Classical exact TE, empirical-occupancy weighted:
    T_qv_emp_occ = ...

Difference:
    empirical - classical = ...

Diagnostic ratio:
    empirical / classical = ...
    [NOT an efficiency]

Empirical signed response:
    Delta_mu_emp = ...

Matched classical response:
    Delta_mu_qv = ...

Controller-policy calibration:
    empirical ADV fraction = ...
    theory occupancy-weighted ADV probability = ...
    statewise policy MAE/RMSE = ...

Support:
    dual-action-state fraction = ...
    singleton fraction = ...
    overlap observation fraction = ...

Epistemic CMI:
    full-E = ...
    phi-conditioned = ...
    susceptible-conditioned = ...
    kappa-conditioned = ...
```

The precise formatting should follow the existing report style rather than forcing this literal text.

---

# 22. Interpretation rules for the generated report

The generated report should not make strong scientific claims automatically, but it may include deterministic descriptive labels.

Use the following logic:

### Case A: policy matches, empirical response and TE close to classical

Interpretation:
> The LLM collective exhibits a control channel similar in scale/shape to the matched classical imitation reference over the visited state range.

### Case B: policy matches, response differs strongly

Interpretation:
> The controller is calibrated, but the LLM population response kernel departs from the q-voter reference.

This is likely where reasoning/semantic/epistemic effects live.

### Case C: local theory looks comparable, finite-horizon scalar differs because occupancies differ

Interpretation:
> Much of the discrepancy is due to different state-space trajectories rather than a purely local channel-strength difference.

### Case D: empirical CMI disappears after epistemic conditioning

Interpretation:
> The apparent controller-to-population information channel is largely explained by the population's epistemic state at the chosen coarse graining.

### Case E: empirical CMI remains above null after epistemic conditioning

Interpretation:
> A directed controller-to-population information channel remains after accounting for the selected epistemic macrostate.

Do not use the terms "thermodynamic efficiency", "information cost", or "second-law bound" unless those quantities are separately derived and justified.

---

# 23. Why this should become standard post-processing

The desired permanent workflow is:

\[
\boxed{
\text{completed relational run}
\rightarrow
\text{MI/CMI analysis}
\rightarrow
\text{matched finite-\(N\) q-voter reference}
\rightarrow
\text{epistemic deviation analysis}.
}
\]

The classical theory must therefore not live as an isolated manuscript calculation. Every stochastic-controller experiment should immediately report where it sits relative to the matched classical control channel.

This gives each empirical result three levels of interpretation:

1. **Behavioral:** did the controller steer the population?
2. **Information-theoretic:** how much directed dependence is measurable?
3. **Theoretical:** how does that response/information channel compare with a controlled classical imitation process at the same sensing and actuation resources?

That three-layer comparison is the purpose of this implementation.

---

# 24. What the agent should report back after implementation

Return:

- files changed;
- new theory module(s), if any;
- exact existing MI/CMI components reused;
- names of all new fields/statistics appended to the existing reports;
- confirmation that the ordinary relational analysis command now computes theory automatically;
- example output from the completed Study 03 run;
- exact \((N,q,q_c,b,\beta,\theta)\) read from that run;
- empirical target CMI;
- occupancy-matched classical TE;
- empirical-vs-theory residual/ratio;
- empirical and theoretical response comparison;
- controller-policy calibration result;
- support diagnostics;
- all test results.

Do not launch new LLM experiments for this task. This is a post-processing/theory-integration task only.
