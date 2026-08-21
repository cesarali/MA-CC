# 20-08-2026 — Relational current analysis: mean, variance, Fano, and SNR² with matched q-voter theory

## Friendly overview

This task adds one compact **current-analysis report per completed grid cell**.

The goal is deliberately compact. For each cell we want to answer four related questions:

1. **Mean current:** on average, how far does the population move toward the analysis target?
2. **Current variance:** how variable is that net movement across repeated episodes?
3. **Fano-like dispersion:** how much fluctuation is observed per unit net current?
4. **Current SNR²:** how large and reproducible is the directed population movement relative to its fluctuations?

For each of these quantities, the report must show:

- the **empirical value** reconstructed from the repeated LLM simulations; and
- the corresponding **exact finite-\(N\) controlled q-voter prediction** at matched simulation parameters.

Do not create separate empirical and theory reports. The output should be **one current report per cell**, with empirical and theory values shown side by side.

This is separate from the MI/CMI/transfer-entropy analysis. Do not modify those quantities here.

No new LLM calls are required.

---

# 1. Scope

Implement the following primary current quantities:

\[
\boxed{
\mathbb E[J]
}
\]

\[
\boxed{
\operatorname{Var}(J)
}
\]

\[
\boxed{
F_J^{\rm disp}
=
\frac{\operatorname{Var}(J)}
{|\mathbb E[J]|}
}
\]

and

\[
\boxed{
\mathrm{SNR}_J^2
=
\frac{\mathbb E[J]^2}{\operatorname{Var}(J)}
}
\]

The empirical estimates and theoretical predictions must use the same definition of current.

Also report, as a clearly named secondary field, the inverse-Fano / Irisarri-style precision

\[
\boxed{
P_J^{\rm Iri}
=
\frac{|\mathbb E[J]|}
{\operatorname{Var}(J)}
=
\frac{1}{F_J^{\rm disp}}
}
\]

whenever defined.

This avoids the nomenclature ambiguity around the word *Fano*: many fields use variance/mean, while the Irisarri current-precision convention uses the inverse ratio.

Do **not** add activity, coherence, entropy production, TURs, or thermodynamic-efficiency quantities in this task.

The purpose is to keep the result scientifically compact while retaining all directly related fluctuation/precision summaries.

---

# 2. Analysis target

For the matched classical theory, use the **controller target** \(Z\).

Let

\[
n_Z(k)
\]

be the number of agents voting for \(Z\) at the beginning of population round \(k\).

The binary theory uses the coarse graining

\[
\boxed{
Z
\quad\text{vs}\quad
\text{not }Z.
}
\]

For an adversarial controller, \(Z\) is the wrong controller target.

For a truthful controller, \(Z\) is the truth.

The report may additionally show truth current as an empirical diagnostic, but the primary empirical-vs-theory comparison must use the controller-target current.

---

# 3. Empirical episode current

For completed episode \(e\), define the finite-horizon net controller-target current

\[
\boxed{
J_Z^{(e)}
=
n_Z^{(e)}(K)-n_Z^{(e)}(0),
}
\]

where \(K\) is the final population round.

Equivalently, at the microscopic level,

\[
J_Z^{(e)}
=
\sum_t j_{Z,t}^{(e)},
\]

where

\[
j_{Z,t}^{(e)}
=
\mathbf 1[X_{f,t}\neq Z,\;X_{f,t+1}=Z]
-
\mathbf 1[X_{f,t}=Z,\;X_{f,t+1}\neq Z].
\]

If microscopic transition records are available, verify numerically that

\[
\sum_t j_{Z,t}^{(e)}
=
n_Z^{(e)}(K)-n_Z^{(e)}(0).
\]

The terminal-count difference is sufficient for the primary metric.

---

# 4. Empirical mean current

Suppose the same cell has \(R\) repeated episodes:

\[
J_Z^{(1)},\ldots,J_Z^{(R)}.
\]

Define

\[
\boxed{
\widehat{\mu}_{J,\mathrm{emp}}
=
\frac{1}{R}
\sum_{e=1}^{R}J_Z^{(e)}.
}
\]

Use the explicit result name:

```text
current_mean_empirical
```

Interpretation:

> Average net number of agents gained by the controller target over one complete episode.

A positive value means net motion toward the controller target.

A negative value means net motion away from it.

---

# 5. Empirical current variance

Across repeated episodes in the same matched cell, calculate the sample variance

\[
\boxed{
\widehat{V}_{J,\mathrm{emp}}
=
\frac{1}{R-1}
\sum_{e=1}^{R}
\left(
J_Z^{(e)}-\widehat{\mu}_{J,\mathrm{emp}}
\right)^2.
}
\]

Use the result name:

```text
current_variance_empirical
```

Do not mix different tasks/worlds into this variance before computing the within-task repeated-episode statistic.

If a grid cell contains more than one task/world, calculate the repeated-episode statistics per task first and keep the task identity in the output.

---

# 6. Empirical Fano-like dispersion and inverse-Fano precision

Define the Fano-like current dispersion

\[
\boxed{
F_{J,\mathrm{emp}}^{\rm disp}
=
\frac{
\widehat{V}_{J,\mathrm{emp}}
}{
|\widehat{\mu}_{J,\mathrm{emp}}|
}.
}
\]

Use:

```text
current_fano_dispersion_empirical
```

Interpretation:

> fluctuation magnitude per unit net current.

Also report the inverse ratio, corresponding to the Irisarri-style current-precision convention,

\[
\boxed{
P_{J,\mathrm{emp}}^{\rm Iri}
=
\frac{
|\widehat{\mu}_{J,\mathrm{emp}}|
}{
\widehat{V}_{J,\mathrm{emp}}
}.
}
\]

Use:

```text
current_precision_irisarri_empirical
```

The two are exact reciprocals whenever both are finite and nonzero.

If \(\widehat{\mu}_{J,\mathrm{emp}}=0\), treat the Fano-like dispersion as undefined/infinite according to the repository's numeric conventions and add an explicit diagnostic flag. Do not silently regularize the denominator.

---

# 6. Empirical current SNR²

Define the squared signal-to-noise ratio

\[
\boxed{
\mathrm{SNR}_{J,\mathrm{emp}}^2
=
\frac{
\widehat{\mu}_{J,\mathrm{emp}}^2
}{
\widehat{V}_{J,\mathrm{emp}}
}.
}
\]

Use the explicit result name:

```text
current_snr2_empirical
```

Interpretation:

- large SNR²: strong and reproducible net population motion;
- small SNR²: weak and/or highly variable net population motion.

This is a dimensionless precision statistic.

Do **not** call it a thermodynamic efficiency.

### Degenerate cases

If

\[
\widehat{V}_{J,\mathrm{emp}}=0
\]

and

\[
\widehat{\mu}_{J,\mathrm{emp}}\neq0,
\]

report SNR² as infinite and set

```text
current_snr2_zero_variance_nonzero_mean = true
```

If both mean and variance are zero, report SNR² as undefined/NaN and set

```text
current_snr2_degenerate_zero_current = true
```

Do not silently add an epsilon to the denominator.

---

# 8. Repetition warning

Mean current can be inspected with very few repetitions.

Variance and SNR² require repeated trajectories.

Always record

```text
n_repetitions
```

and a qualitative support flag, for example

```text
current_precision_support = descriptive_only | limited | adequate
```

For the present pilot-style cells with only two repetitions, variance and SNR² are mathematically computable but should be marked **descriptive only**.

Reuse the existing whole-episode bootstrap infrastructure when enough repetitions are available.

---

# 9. Matched controlled q-voter theory

Use the exact finite-\(N\) controlled q-voter already defined for the project.

The theory must use the run parameters

\[
(N,q,q_c,b,\beta,\theta)
\]

where applicable.

For open-loop `always ADVOCATE` cells, the finite-horizon current theory only requires the exact ADVOCATE round kernel \(R_1\).

For open-loop `NO_OP`, use \(R_0\).

For stochastic feedback, use the state-dependent closed-loop kernel defined below.

Do not estimate the q-voter theory with Monte Carlo when exact finite-\(N\) matrix propagation applies.

---

# 10. Exact microscopic kernels

## Ordinary microscopic kernel \(K_0\)

\[
\boxed{
K_0(n+1\mid n)
=
\frac{N-n}{N}
\frac{\binom{n}{q}}{\binom{N-1}{q}}
}
\]

\[
\boxed{
K_0(n-1\mid n)
=
\frac{n}{N}
\frac{\binom{N-n}{q}}{\binom{N-1}{q}}
}
\]

and

\[
\boxed{
K_0(n\mid n)
=
1-K_0(n+1\mid n)-K_0(n-1\mid n).
}
\]

## Controlled microscopic kernel \(K_1\)

\[
\boxed{
K_1(n+1\mid n)
=
\frac{N-n}{N}
\frac{\binom{n}{q-1}}{\binom{N-1}{q-1}}
}
\]

\[
\boxed{
K_1(n-1\mid n)=0
}
\]

and

\[
\boxed{
K_1(n\mid n)
=
1-K_1(n+1\mid n).
}
\]

Use zero for impossible combinatorial transitions.

---

# 11. Exact whole-round kernels

## NO_OP round

One population round contains exactly \(N\) microscopic positions:

\[
\boxed{
R_0=K_0^N.
}
\]

## ADVOCATE round

Exactly \(b\) of the \(N\) microscopic positions are controlled and uniformly preallocated.

Initialize

\[
F_{0,0}=I.
\]

After \(r\) positions, if \(j\) controlled positions have already occurred,

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

This is exact finite-\(N\).

Do not approximate exactly-\(b\) actuation by independent Bernoulli-\(c\) actuation.

---

# 12. Closed-loop kernel for stochastic-feedback experiments

When the controller is stochastic, compute

\[
S(y\mid n)
=
\frac{
\binom{n}{y}\binom{N-n}{q_c-y}
}{
\binom{N}{q_c}
}
\]

and

\[
a_n
=
\sum_y
S(y\mid n)
\sigma\left[
\beta\left(\theta-\frac{y}{q_c}\right)
\right].
\]

Then define the row-wise closed-loop kernel

\[
\boxed{
R(m\mid n)
=
[1-a_n]R_0(m\mid n)
+
a_nR_1(m\mid n).
}
\]

For open-loop cells:

```text
always ADVOCATE -> use R1
always NO_OP    -> use R0
```

---

# 13. Exact finite-horizon theoretical current

Let

\[
\mathcal R
\]

denote the appropriate whole-round kernel:

\[
\mathcal R=
\begin{cases}
R_0,&\text{NO_OP},\\
R_1,&\text{always ADVOCATE},\\
R,&\text{stochastic feedback}.
\end{cases}
\]

Let \(P_0(n_0)\) be the initial target-count distribution.

If the initial target count is deterministic, \(P_0\) is a point mass.

If several repeated LLM episodes have different initial target counts, use their empirical initial distribution for the matched theory comparison.

After \(K\) rounds,

\[
\boxed{
P(N_K=m\mid N_0=n_0)
=
[\mathcal R^K]_{n_0m}.
}
\]

The finite-horizon current is

\[
\boxed{
J^{(K)}=N_K-N_0.
}
\]

---

# 14. Exact theoretical mean current

Define

\[
\boxed{
\mu_{J,\mathrm{theory}}
=
\mathbb E_{\mathrm{qv}}[J^{(K)}]
=
\sum_{n_0,m}
P_0(n_0)
[\mathcal R^K]_{n_0m}
(m-n_0).
}
\]

Use the result name:

```text
current_mean_theory
```

This is an exact finite-\(N\), finite-horizon theoretical quantity.

---

# 15. Exact theoretical current variance

First calculate

\[
\boxed{
M_{2,J,\mathrm{theory}}
=
\mathbb E_{\mathrm{qv}}[(J^{(K)})^2]
=
\sum_{n_0,m}
P_0(n_0)
[\mathcal R^K]_{n_0m}
(m-n_0)^2.
}
\]

Then

\[
\boxed{
V_{J,\mathrm{theory}}
=
\operatorname{Var}_{\mathrm{qv}}(J^{(K)})
=
M_{2,J,\mathrm{theory}}
-
\mu_{J,\mathrm{theory}}^2.
}
\]

Use the result name:

```text
current_variance_theory
```

This is also exact finite-\(N\), finite-horizon theory.

---

# 16. Exact theoretical Fano-like dispersion and inverse-Fano precision

From the exact finite-horizon theory moments define

\[
\boxed{
F_{J,\mathrm{theory}}^{\rm disp}
=
\frac{
V_{J,\mathrm{theory}}
}{
|\mu_{J,\mathrm{theory}}|
}.
}
\]

Use:

```text
current_fano_dispersion_theory
```

Also report the inverse-Fano / Irisarri-style precision

\[
\boxed{
P_{J,\mathrm{theory}}^{\rm Iri}
=
\frac{
|\mu_{J,\mathrm{theory}}|
}{
V_{J,\mathrm{theory}}
}.
}
\]

Use:

```text
current_precision_irisarri_theory
```

Again, these are reciprocal whenever both are finite and nonzero.

Apply explicit zero-mean/zero-variance handling rather than epsilon regularization.

---

# 17. Exact theoretical current SNR²

Define

\[
\boxed{
\mathrm{SNR}_{J,\mathrm{theory}}^2
=
\frac{
\mu_{J,\mathrm{theory}}^2
}{
V_{J,\mathrm{theory}}
}.
}
\]

Use the result name:

```text
current_snr2_theory
```

Apply the same zero-variance rules as for the empirical statistic.

This gives a direct matched comparison:

\[
\boxed{
\mathrm{SNR}_{J,\mathrm{emp}}^2
\quad\text{vs}\quad
\mathrm{SNR}_{J,\mathrm{theory}}^2.
}
\]

---

# 18. Genuine closed-form result for \(q=1\)

Whenever \(q=1\), show this result explicitly in the report.

For NO_OP,

\[
\boxed{
\mathbb E[N_{k+1}\mid N_k=n,\mathrm{NO\_OP}]
=
n
}
\]

so the one-round mean current is

\[
\boxed{
\mu_{J,0}^{(q=1)}(n)=0.
}
\]

For ADVOCATE with exactly \(b\) controlled positions,

\[
\boxed{
\mathbb E[N_{k+1}\mid N_k=n,\mathrm{ADVOCATE}]
=
N-(N-n)
\left(1-\frac1N\right)^b.
}
\]

Therefore

\[
\boxed{
\mu_{J,1}^{(q=1)}(n)
=
(N-n)
\left[
1-\left(1-\frac1N\right)^b
\right].
}
\]

Because the NO_OP mean is zero,

\[
\boxed{
\Delta\mu_J^{(q=1)}(n)
=
(N-n)
\left[
1-\left(1-\frac1N\right)^b
\right].
}
\]

In population-fraction coordinates \(x=n/N\),

\[
\boxed{
\Delta\mu_x^{(q=1)}(x)
=
(1-x)
\left[
1-\left(1-\frac1N\right)^b
\right].
}
\]

For \(b=cN\),

\[
\boxed{
\Delta\mu_x^{(q=1)}(x)
\longrightarrow
(1-x)(1-e^{-c})
}
\]

as \(N\to\infty\).

The implementation must test that the finite-\(N\) kernel calculation agrees with this closed-form expression.

Use the names:

```text
q1_current_mean_noop_closed_form_theory
q1_current_mean_advocate_closed_form_theory
q1_current_response_closed_form_theory
q1_current_response_fraction_closed_form_theory
```

Do not call the general finite-horizon matrix formulas “closed form”; they are exact finite-\(N\) calculations.

---

# 19. Per-cell summary table

Write one

```text
currents/cell_current_summary.csv
```

with one row per task/cell and both empirical and theory quantities.

Required fields:

```text
cell_id
task_id
analysis_target
N
q
q_c
b
c
beta
theta
K
n_repetitions

current_mean_empirical
current_variance_empirical
current_fano_dispersion_empirical
current_precision_irisarri_empirical
current_snr2_empirical

current_mean_theory
current_variance_theory
current_fano_dispersion_theory
current_precision_irisarri_theory
current_snr2_theory

current_mean_empirical_minus_theory
current_variance_empirical_minus_theory
current_fano_dispersion_empirical_minus_theory
current_precision_irisarri_empirical_minus_theory
current_snr2_empirical_minus_theory

current_precision_support
current_snr2_zero_variance_nonzero_mean
current_snr2_degenerate_zero_current
```

For \(q=1\), append the closed-form fields from Section 16.

Also preserve the episode-level currents in

```text
currents/episode_currents.csv
```

with at minimum:

```text
cell_id
task_id
episode_id
initial_target_count
final_target_count
episode_current
```

---

# 20. One human-readable report per cell

Write exactly one:

```text
currents/current_analysis.md
```

per cell.

Start with a friendly explanation such as:

> This report measures the net population current toward the controller target over repeated LLM episodes and compares it with the exact finite-\(N\) controlled q-voter at the same matched parameters. The primary quantities are the mean current, current variance, Fano-like dispersion, and the squared signal-to-noise ratio SNR². The report also gives the inverse-Fano / Irisarri-style precision explicitly so there is no convention ambiguity.

Then show:

## Empirical repeated-episode current

```text
episodes: ...
mean current: ...
variance: ...
Fano-like dispersion Var/|mean|: ...
Irisarri-style precision |mean|/Var: ...
SNR²: ...
support/repetition warning: ...
```

## Exact matched q-voter current

```text
theory mode: ...
N=...
q=...
q_c=...
b=...
c=...
beta=...
theta=...
K=...

mean current: ...
variance: ...
Fano-like dispersion Var/|mean|: ...
Irisarri-style precision |mean|/Var: ...
SNR²: ...
```

## Direct comparison

Use a compact table:

| Quantity | Empirical | Exact q-voter | Empirical - theory |
|---|---:|---:|---:|
| Mean current | ... | ... | ... |
| Current variance | ... | ... | ... |
| Fano-like dispersion Var/|mean| | ... | ... | ... |
| Irisarri-style precision |mean|/Var | ... | ... | ... |
| Current SNR² | ... | ... | ... |

If \(q=1\), print the closed-form result from Section 16 immediately below this table.

The report should never require the reader to open a second theory document.

---

# 21. Comet integration

Expose the same three quantities to Comet whenever Comet is enabled.

Use the existing master/post-hoc Comet path. Do not log from episode workers.

Recommended keys:

```text
current/mean_empirical
current/mean_theory

current/variance_empirical
current/variance_theory

current/fano_dispersion_empirical
current/fano_dispersion_theory

current/precision_irisarri_empirical
current/precision_irisarri_theory

current/snr2_empirical
current/snr2_theory

current/n_repetitions
```

For \(q=1\):

```text
current/q1_response_closed_form_theory
current/q1_closed_form_matches_kernel
```

The filesystem report remains authoritative. Comet is only a synchronized visualization/logging layer.

---

# 22. Bootstrap

Reuse the existing whole-episode bootstrap machinery.

For each bootstrap replicate:

1. resample complete episodes;
2. recompute the empirical current mean;
3. recompute empirical current variance;
4. recompute empirical Fano-like dispersion and Irisarri-style precision;
5. recompute empirical SNR²;
6. if the matched theory uses the empirical initial \(P_0\), recompute \(P_0\) from the bootstrap sample and then recompute the exact theory quantities;
7. compute empirical-minus-theory differences.

Do not bootstrap the exact transition kernels themselves.

With very small \(R\), report the raw values but mark variance/SNR² as descriptive.

---

# 23. Required tests

At minimum:

1. episode current equals final target count minus initial target count;
2. microscopic current sum equals the terminal-count difference when microscopic records exist;
3. empirical mean matches a hand-computed fixture;
4. empirical sample variance matches a hand-computed fixture;
5. empirical Fano-like dispersion matches variance/absolute-mean;
6. empirical Irisarri-style precision matches absolute-mean/variance;
7. empirical SNR² matches mean²/variance;
8. Fano-like dispersion and Irisarri-style precision are reciprocal when both are finite and nonzero;
9. zero-variance and zero-mean cases are handled explicitly;
10. different tasks are not silently pooled into one variance estimate;
11. \(K_0\) rows normalize;
12. \(K_1\) rows normalize;
13. \(R_0\) rows normalize;
14. \(R_1\) rows normalize;
15. \(R_1\) uses exactly-\(b\) schedule semantics;
16. \(q=1\) kernel mean agrees with the closed-form expression;
17. finite-horizon theory mean/variance agree with brute-force enumeration for a tiny system;
18. theoretical Fano-like dispersion matches exact variance/absolute-mean;
19. theoretical Irisarri-style precision matches absolute-mean/exact variance;
20. theoretical SNR² equals exact mean²/exact variance;
21. empirical and theory fields all appear in the same cell summary;
22. the same `current_analysis.md` contains empirical and theory results;
23. Comet-off analysis still writes all local results;
24. mocked Comet logging receives empirical and theory keys from the master/post-hoc layer only;
25. no provider/LLM calls are made.

---

# 24. Expected command behavior

Extend the existing relational current/post-processing entry point rather than introducing a separate theory command.

A completed cell should be transformable as

\[
\boxed{
\text{completed repeated episodes}
\rightarrow
\text{episode currents}
\rightarrow
(\widehat\mu_J,\widehat V_J,\widehat F_J^{\rm disp},
\widehat P_J^{\rm Iri},\widehat{\mathrm{SNR}}_J^2)
\rightarrow
(\mu_J^{\rm qv},V_J^{\rm qv},F_{J,\rm qv}^{\rm disp},
P_{J,\rm qv}^{\rm Iri},\mathrm{SNR}_{J,\rm qv}^2)
\rightarrow
\text{one current report}.
}
\]

No new simulations are required.

---

# 25. What the coding agent should report back

Return:

```text
1. files changed;
2. existing readers/bootstrap/reporting code reused;
3. exact theory module/functions reused or added;
4. confirmation that mean, variance, Fano-like dispersion, Irisarri-style precision, and SNR² are all reported;
5. one example current_analysis.md;
6. one example cell_current_summary.csv;
7. q=1 closed-form validation result;
8. empirical/theory Comet keys;
9. all tests and results;
10. confirmation that no LLM/provider calls were made.
```

The final scientific product should remain deliberately simple:

\[
\boxed{
\textbf{Current mean}
\qquad
\textbf{Current variance}
\qquad
\textbf{Fano-like dispersion}
\qquad
\textbf{Irisarri-style precision}
\qquad
\textbf{Current SNR}^2
}
\]

with the empirical LLM result and the exact matched q-voter prediction shown directly beside one another.
