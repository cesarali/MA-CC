# Implementation plan: one canonical single-affinity treatment across theory, simulation, and empirical estimation

**Date:** 27 August 2026  
**Status:** implementation contract  
**Canonical theory:** `theory_revised.py` + revised single-affinity report / current ICLR paper  
**Primary goal:** make the quantities reported from LLM trajectories mathematically identical, in definition and units, to the quantities exposed by the single-affinity theory.

---

## 0. Non-negotiable scientific contract

The codebase must have **one canonical meaning** for each symbol:

\[
n_k = \text{number of agents voting for controller target } Z,
\qquad
x_k = n_k/N.
\]

The controller senses the binary target projection,

\[
Y_k = \text{number of sampled agents voting for } Z,
\]

and acts according to

\[
a_n \equiv P(U_k=1\mid n_k=n)
=\sum_y S(y\mid n)\,\pi(1\mid y).
\]

The single-affinity controlled microscopic kernel is

\[
K_{h,\gamma}(n+1\mid n)
=\gamma\frac{N-n}{N}\sigma(h),
\]

\[
K_{h,\gamma}(n-1\mid n)
=\gamma\frac{n}{N}\sigma(-h),
\]

with

\[
Q_0=I,
\qquad
Q_1=K_{h,\gamma}^{\,b}.
\]

The canonical theory objects are

\[
\chi(n/N)
=E[x_{k+1}\mid U_k=1,n_k=n]
-E[x_{k+1}\mid U_k=0,n_k=n],
\]

\[
T_\pi(n)=I(U_k;n_{k+1}\mid n_k=n),
\]

\[
\eta_{\mathrm{IR}}(n)
=\frac{2a_n(1-a_n)\chi(n/N)^2}
{(\ln 2)T_\pi(n)}\le1,
\]

\[
J_{c,k}=N\sum_n p_k(n)a_n\chi(n/N),
\]

\[
I_{\mathrm{sens},k}=I(n_k;Y_k),
\]

\[
\Delta S_{\mathrm{sys},k}+hJ_{c,k}+I_{\mathrm{sens},k}
=\Sigma_k\ge0,
\]

and

\[
\eta_{\mathrm{th},k}
=\frac{hJ_{c,k}}
{hJ_{c,k}+I_{\mathrm{sens},k}}
=\frac{hJ_{c,k}}{\Sigma_k-\Delta S_{\mathrm{sys},k}}
\le1
\]

when \(hJ_{c,k}\ge0\) and the denominator is positive.

Everything below exists to enforce these definitions without silent re-scaling or estimator drift.

---

# 1. Architectural separation: preserve the three scientific layers

The repository must keep the following layers distinct.

## 1.1 LLM runtime / empirical process

The actual relational LLM game remains the empirical system. It contains ordinary social/reasoning updates plus controller interventions. **Do not rewrite the LLM runtime to force it to obey the single-affinity kernel.**

The LLM data are used to estimate \(h\), \(\gamma\), \(\chi\), \(T_\pi\), \(J_c\), and the efficiencies.

## 1.2 Single-affinity finite-state theory

`theory_revised.py` remains deterministic and provider-free. It is the canonical mathematical implementation of the revised theory:

```text
TheoryParameters
  -> sensor law / policy / a_n
  -> K_{h,gamma}, Q0, Q1
  -> chi(n)
  -> T_pi(n)
  -> eta_IR(n)
  -> occupancy p_k
  -> J_c, I_sens, Delta S_sys, Sigma, C_th, eta_th
```

No empirical estimator, bootstrap, smoothing convention, or trajectory reader belongs in this module.

## 1.3 Matched q-voter reference

The existing matched q-voter theory remains a **separate classical null/reference**. It must not be silently reinterpreted as the single-affinity theory.

Recommended naming:

```text
theory mode: single_affinity
classical reference: matched_qvoter
```

Do not use `reasoning` as the name of a classical population kernel.

## 1.4 Provider-free simulation, if/when enabled

If a provider-free `single_affinity` dynamics mode is exposed, it must call the same canonical kernel construction used by `theory_revised.py`; do **not** duplicate the formulas in a second implementation.

A simulation kernel and a theory kernel that independently restate \(K_{h,\gamma}\) are not acceptable because they can drift.

---

# 2. First repair: define one canonical empirical susceptibility

This must be completed before changing either efficiency.

## 2.1 Current inconsistency

There are currently two response quantities:

```text
round_target_signed_response_share
    outcome: delta_p_ctrl
    units: target fraction
    currently not state-matched

round_target_signed_actuation
    outcome: delta_m_ctrl
    units: aligned magnetization
    state-matched on target_count_before
```

For \(K=3\),

\[
m_{\mathrm{ctrl}}=\frac{3p_{\mathrm{ctrl}}-1}{2},
\qquad
\Delta m_{\mathrm{ctrl}}=\frac32\Delta p_{\mathrm{ctrl}}.
\]

Therefore the second estimator differs from the target-fraction response by a factor \(3/2\).

The theory does **not** define \(\chi\) in magnetization units. It defines it in target-fraction units.

## 2.2 Add the canonical estimator

Add a new estimator with an unambiguous name, preferably

```text
round_target_susceptibility
```

with definition

\[
\widehat\chi(n)
=
E[\Delta x\mid U=1,n]
-
E[\Delta x\mid U=0,n],
\qquad
\Delta x=\frac{n_{k+1}-n_k}{N}.
\]

Implementation requirements:

- reuse the existing `_signed_response`/state-slicing machinery;
- condition on `target_count_before`;
- use `delta_p_ctrl`, **not** `delta_m_ctrl`;
- preserve dual-action/support diagnostics;
- preserve episode-aware bootstrap behavior;
- expose state-resolved estimates when the existing analysis representation supports them;
- expose an occupancy-weighted summary only as a summary of the state-resolved object.

## 2.3 Preserve old estimators as diagnostics

Do **not** delete:

```text
round_target_signed_response_share
round_target_signed_actuation
```

They remain useful legacy/diagnostic observables. However:

- neither should be the canonical theoretical \(\chi\);
- `round_target_signed_actuation` must never be inserted directly into the revised Pinsker formula;
- plots/tables should label magnetization and target-fraction response explicitly.

## 2.4 Mandatory cross-check

For any identical state-matching calculation on a \(K\)-option task,

\[
\chi_m=\frac{K}{K-1}\chi_x.
\]

For Study 08 with \(K=3\), test numerically that

\[
\chi_m=1.5\chi_x
\]

up to floating-point tolerance whenever both quantities use the same rows and same state matching.

---

# 3. Repair empirical information-response efficiency `eta_ir`

The current empirical `eta_ir` must be audited because it currently consumes the magnetization response while the revised derivation assumes target fraction.

## 3.1 State-local theoretical bound

The implemented theory is

\[
T_\pi(n)
\ge
B_{\mathrm{IR}}(n)
\equiv
\frac{2a_n(1-a_n)}{\ln2}\chi(n)^2.
\]

Thus

\[
\eta_{\mathrm{IR}}(n)
=\frac{B_{\mathrm{IR}}(n)}{T_\pi(n)}.
\]

The range argument in the derivation uses

\[
f(n')=n'/N\in[0,1].
\]

Therefore **the empirical response entering `eta_ir` must be the new target-fraction susceptibility**.

## 3.2 Do not repair by multiplying the existing result afterward

Although for \(K=3\)

\[
\chi_x=\frac23\chi_m,
\]

and therefore the old numerator is inflated by \((3/2)^2=2.25\), the implementation should not patch `eta_ir` with a hard-coded `4/9` factor.

Instead, route `eta_ir` through `round_target_susceptibility` so the units are correct by construction for arbitrary \(K\).

## 3.3 Canonical occupancy-level empirical `eta_ir`

The state-local inequality can be occupancy weighted:

\[
\sum_n p(n)B_{\mathrm{IR}}(n)
\le
\sum_n p(n)T_\pi(n)
=I(U;n_{k+1}\mid n_k).
\]

Define the headline occupancy-level efficiency as

\[
\boxed{
\eta_{\mathrm{IR}}^{\mathrm{occ}}
=
\frac{
\sum_n \widehat p(n)
\frac{2\widehat a_n(1-\widehat a_n)}{\ln2}
\widehat\chi(n)^2
}
{
\widehat I(U;n_{k+1}\mid n_k)
}
}
\]

when the denominator is positive.

This is preferred to either

```text
mean_n eta_IR(n)
```

or

```text
2 * global_a * (1-global_a) * global_chi^2 / ((ln 2) * global_T)
```

because the rigorous state-local inequality survives the occupancy sum directly.

## 3.4 Action mixing weight

The mixing weight used in the numerator must match the action channel used in the empirical denominator.

Primary recommendation:

- compute \(\widehat a_n=P(U=1\mid n)\) from the same empirical rows used by the CMI / state-local transition estimator;
- retain the exact sensor-policy curve `a_n` from `theory_revised.py` as a **policy-calibration comparator**.

If the analysis instead reconstructs empirical \(Q_0,Q_1\) and explicitly evaluates their Jensen-Shannon divergence at the exact protocol \(a_n\), then the exact protocol weight may be used. Do not mix an empirical-CMI denominator using one action weight with a Pinsker numerator using another without documenting it.

## 3.5 Output fields

At minimum expose

```text
eta_ir
eta_ir_ci_low
eta_ir_ci_high
eta_ir_pinsker_numerator_bits
eta_ir_denominator_T_bits
eta_ir_support_mass
eta_ir_valid
```

and preserve the current `round_target_actuation_cmi` as a separate empirical information diagnostic.

---

# 4. Add the thermodynamic sensing quantity in the correct state space

The single-affinity thermodynamic theory uses the binary target coordinate:

\[
I_{\mathrm{sens},k}=I(n_{Z,k};Y_{Z,k}).
\]

It does **not** use the full \(K\)-option population vector.

## 4.1 Do not reuse full-vector `round_sensing_mi` blindly

If existing `round_sensing_mi` is

\[
I(\mathbf N_k;\mathbf Y_k),
\]

keep it as a useful general sensing metric, but do not put it into `eta_th`.

Add a scalar target-channel quantity, e.g.

```text
round_target_sensing_mi
```

for

\[
I(n_{Z,k};Y_{Z,k}).
\]

## 4.2 Canonical thermodynamic estimate: exact sensor channel on empirical occupancy

Because the sensing mechanism is known exactly, the preferred thermodynamic estimator should use the empirical population occupancy and the exact hypergeometric channel:

\[
\widehat I_{\mathrm{sens},k}
=
\sum_{n,y}
\widehat p_k(n)S(y\mid n)
\ln
\frac{S(y\mid n)}
{\sum_{n'}\widehat p_k(n')S(y\mid n')}.
\]

This quantity is:

- empirical with respect to the visited LLM population distribution;
- exact with respect to the known sensor mechanism;
- naturally in **nats**;
- substantially less noisy than estimating the same MI solely from realized sensor draws.

Recommended name:

```text
target_sensing_information_nats
```

Keep a direct-counting target sensing MI from realized `(n_Z, Y_Z)` pairs as a validation diagnostic if desired.

## 4.3 Finite-time / round dependence

Do not silently impose stationarity by pooling all rounds into one occupancy when computing the thermodynamic quantity.

For a horizon of \(H\) rounds, compute

\[
I_{\mathrm{sens},k}
\]

from the empirical occupancy \(p_k(n)\) at each round index and aggregate

\[
I_{\mathrm{sens}}^{(H)}
=\sum_{k=0}^{H-1}I_{\mathrm{sens},k}.
\]

A per-round average may also be reported, but the finite-horizon sum is the primary quantity entering the horizon efficiency.

---

# 5. Effective affinity and kinetic compliance

These quantities are already conceptually aligned with the revised theory and should remain based on **controlled microscopic transitions**.

Define

\[
p_+
=P(\neg Z\rightarrow Z\mid\text{controlled, eligible}),
\]

\[
p_-
=P(Z\rightarrow\neg Z\mid\text{controlled, eligible}).
\]

Then

\[
\boxed{
\widehat\gamma=p_++p_-
}
\]

and

\[
\boxed{
\widehat h=\ln\frac{p_+}{p_-}.
}
\]

## 5.1 Unit convention

`effective_affinity` must use a natural logarithm. It is a dimensionless affinity in **nats per net target-count revision** for the path accounting.

## 5.2 Zero transition handling

Do not silently add an arbitrary epsilon only to make `eta_th` finite.

Primary unsmoothed behavior:

- if either direction has zero eligible support, affinity is undefined;
- if a direction is eligible but has zero observed transitions, the unsmoothed finite affinity is not identified;
- `eta_th` should then be reported as undefined for that estimator variant.

If the existing pipeline already exposes Jeffreys or another explicit smoothing variant, it may be retained as a **sensitivity estimate**, clearly labelled as such.

## 5.3 Calibration consistency test

The empirical calibration function and `theory_revised.py::calibrate_affinity_compliance_from_counts` must agree on the same controlled transition table.

---

# 6. Define the empirical controlled current from susceptibility

Do **not** use the existing episode terminal current

```text
cell_current = n_Z,H - n_Z,0
```

inside `eta_th`.

That terminal difference mixes controller-induced motion with ordinary social/reasoning dynamics and is a different scientific observable.

## 6.1 Canonical current

The theory defines

\[
J_{c,k}=N\sum_n p_k(n)a_n\chi(n).
\]

The empirical analogue must therefore be

\[
\boxed{
\widehat J_{c,k}
=N\sum_n\widehat p_k(n)\widehat a_n\widehat\chi(n)
}
\]

or, if susceptibility is estimated separately by round,

\[
\widehat J_{c,k}
=N\sum_n\widehat p_k(n)\widehat a_{n,k}\widehat\chi_k(n).
\]

This is a g-computation / response-based current. It is the current consistent with the revised theory.

## 6.2 Do not factor the expectation

Do **not** approximate this by

\[
N\,\bar a\,\bar\chi.
\]

In general

\[
E[a_n\chi_n]\neq E[a_n]E[\chi_n].
\]

Both the policy and susceptibility are state dependent.

## 6.3 Horizon current

For \(H\) feedback cycles,

\[
\widehat J_c^{(H)}
=\sum_{k=0}^{H-1}\widehat J_{c,k}.
\]

Also report the per-cycle average if convenient:

\[
\overline J_c^{(H)}=\widehat J_c^{(H)}/H.
\]

## 6.4 Direct microscopic current as a cross-check only

If microscopic records permit, compute

\[
J_{\mathrm{ctrl,micro}}
=\sum_{\text{controlled positions}}
[\mathbf 1(X_{after}=Z)-\mathbf 1(X_{before}=Z)]
\]

as a useful diagnostic.

Do not substitute it automatically for the response-based \(J_c\): the headline current must remain tied to the same causal susceptibility used by the rest of the efficiency family.

---

# 7. Implement empirical thermodynamic efficiency `eta_th`

Once the canonical \(\widehat h\), \(\widehat J_c\), and \(\widehat I_{\mathrm{sens}}\) exist, add the derived thermodynamic family.

## 7.1 One-cycle quantities

\[
W_{c,k}^{\mathrm{aff}}
\equiv
\widehat h\,\widehat J_{c,k},
\]

\[
\widehat C_{\mathrm{th},k}
=
\widehat h\widehat J_{c,k}
+
\widehat I_{\mathrm{sens},k},
\]

\[
\boxed{
\widehat\eta_{\mathrm{th},k}
=
\frac{\widehat h\widehat J_{c,k}}
{\widehat h\widehat J_{c,k}+\widehat I_{\mathrm{sens},k}}
}
\]

when defined.

## 7.2 Primary finite-horizon quantity

For the repeated multi-round studies, the headline empirical thermodynamic efficiency should be the **ratio of accumulated terms**:

\[
\boxed{
\widehat\eta_{\mathrm{th}}^{(H)}
=
\frac{
\widehat h\sum_{k=0}^{H-1}\widehat J_{c,k}
}{
\widehat h\sum_{k=0}^{H-1}\widehat J_{c,k}
+
\sum_{k=0}^{H-1}\widehat I_{\mathrm{sens},k}
}.
}
\]

Do not use

\[
\frac1H\sum_k\widehat\eta_{\mathrm{th},k}
\]

as the primary efficiency. The ratio of total directed expenditure to total non-storage expenditure is the finite-horizon quantity consistent with the theory.

## 7.3 Output the decomposition, not only the ratio

At minimum write

```text
effective_affinity
kinetic_compliance
controlled_current
controlled_current_horizon
target_sensing_information_nats
target_sensing_information_horizon_nats
affinity_weighted_current_nats
thermodynamic_control_expenditure_nats
eta_th
eta_th_ci_low
eta_th_ci_high
eta_th_target_directed
eta_th_valid
```

This makes the ratio auditable.

## 7.4 Validity condition

The bounded interpretation requires

\[
\widehat h\widehat J_c\ge0,
\qquad
\widehat C_{\mathrm{th}}>0.
\]

If these conditions fail:

- retain the signed numerator as a diagnostic;
- do not clip the ratio to `[0,1]`;
- set `eta_th_valid=false`;
- use `NaN` for the bounded-efficiency headline if appropriate.

---

# 8. Empirical support and identifiability must propagate into both efficiencies

The response \(\chi(n)\) requires both controller actions to be observed at a state.

For every cell / task / analysis slice record:

```text
chi_state_count
chi_dual_action_state_fraction
chi_dual_action_event_fraction
chi_identified_occupancy_mass
eta_ir_identified_occupancy_mass
eta_th_identified_occupancy_mass
```

No estimator should silently pretend that unsupported states have \(\chi(n)=0\).

Recommended behavior:

1. compute state-local \(\chi(n)\) only on dual-action states;
2. propagate an explicit support mask into `eta_ir` and response-based `J_c`;
3. report the occupancy mass represented by that mask;
4. distinguish a support-restricted estimate from a fully identified one;
5. do not silently renormalize away missing occupancy without labelling the result.

Keep the existing overlap/support diagnostics in the output and add derived-metric-specific support fields if needed.

---

# 9. Bootstrap the whole derived estimator, not its ingredients independently

The two efficiencies are nonlinear functions of correlated estimates. Confidence intervals must therefore be obtained by **whole-episode bootstrap of the entire pipeline**.

For bootstrap replicate \(r\):

1. resample complete episodes with replacement;
2. rebuild empirical round occupancies \(\widehat p_k^{(r)}(n)\);
3. recompute state-matched \(\widehat\chi^{(r)}(n)\);
4. recompute empirical action weights / channel quantities used by `eta_ir`;
5. recompute \(\widehat T^{(r)}\) and \(\widehat\eta_{\mathrm{IR}}^{(r)}\);
6. recompute microscopic \(p_+^{(r)},p_-^{(r)},\widehat h^{(r)},\widehat\gamma^{(r)}\);
7. recompute \(\widehat J_c^{(r)}\);
8. recompute roundwise and horizon sensing information;
9. recompute \(\widehat\eta_{\mathrm{th}}^{(r)}\).

Then form percentile intervals from the bootstrap distribution.

Do **not** combine independently bootstrapped confidence intervals for \(h\), \(J_c\), and \(I_{\mathrm{sens}}\).

---

# 10. Required file-level implementation work

The coding agent should inspect the repository first and adapt exact paths if they have moved. The following responsibilities should remain separated.

## 10.1 `theory_revised.py`

Keep it pure and deterministic.

Verify/retain:

```text
susceptibility / susceptibility_curve
kernel_mean_response
local_action_information (T_pi)
information_response_lower_bound
information_response_efficiency
sensing_information_nats
mean_controlled_current
thermodynamic_efficiency
SingleAffinityReference.one_cycle
finite_horizon_thermodynamics
calibrate_affinity_compliance_from_counts
```

Optional useful additions:

```text
occupancy_weighted_information_response_bound(...)
occupancy_weighted_information_response_efficiency(...)
```

only if they are generic deterministic theory utilities. Do not insert empirical data handling here.

## 10.2 Round-feedback analysis (`hidden_bench/.../analysis.py` and relational adapter)

Add / expose:

```text
round_target_susceptibility
round_target_sensing_mi
```

Ensure the relational adapter supplies:

```text
episode_id
round_index
N
target_count_before
target_count_after
target_fraction_before
delta_p_ctrl
controller_action
logged_advocacy_probability
sensor_target_count
sensor_sample_size
beta
theta
b
```

and all fields needed by the existing microscopic affinity/compliance estimator.

## 10.3 Study aggregation (`src/mas_cc/studies/aggregation.py` or current equivalent)

Implement dedicated derived builders, rather than embedding formulas inline:

```text
_build_eta_ir(...)
_build_target_sensing_information(...)
_build_controlled_current(...)
_build_eta_th(...)
```

Each builder should return both the headline estimate and its audit components / validity metadata.

## 10.4 Study recipe (`population_study_08/analysis.yaml`)

Add the real derived quantities to `derived:`. The exact schema should follow the current recipe system, but semantically it should include:

```text
round_target_susceptibility
eta_ir
target_sensing_information_nats
controlled_current
affinity_weighted_current_nats
thermodynamic_control_expenditure_nats
eta_th
```

Do not rely on `experiment.metadata.primary_analysis_family` to trigger computation.

## 10.5 Study 08 config metadata

After the estimators are genuinely implemented, update `primary_analysis_family` in both Study 08 configs so the metadata truthfully names the headline outputs.

Recommended family:

```text
round_target_susceptibility
round_target_actuation_cmi
eta_ir
effective_affinity
kinetic_compliance
controlled_current
target_sensing_information_nats
eta_th
```

Keep legacy response/magnetization quantities elsewhere in diagnostics rather than presenting them as the canonical \(\chi\).

## 10.6 Plotting / reports

Add at least:

1. `chi(x)` empirical state profile in target-fraction units, with theory overlay when calibrated \((h,\gamma)\) is available;
2. `eta_ir` versus sensing/actuation resources;
3. `J_c` versus resources;
4. `I_sens` in nats versus resources;
5. `eta_th` versus resources;
6. optional decomposition plot showing `h*J_c` and `I_sens` separately.

Every plot axis must state units.

---

# 11. Unit discipline: enforce this in names, docs, and tests

| Quantity | Canonical units | Notes |
|---|---|---|
| \(x=n/N\) | fraction | `[0,1]` |
| \(\chi\) | target-fraction change | **not magnetization** |
| \(T_\pi\) | bits | empirical action channel convention |
| Pinsker numerator for `eta_ir` | bits | contains `/ ln(2)` |
| \(h\) | natural-log affinity | dimensionless / nats convention |
| \(\gamma\) | probability scale | `[0,1]` |
| \(J_c\) | target-count change per cycle | multiply fraction response by `N` |
| \(hJ_c\) | nats per cycle | directed term |
| \(I_{sens}\) | nats per cycle | natural logarithm |
| \(C_{th}\) | nats per cycle | `hJ_c + I_sens` |
| \(\eta_{IR}\) | dimensionless | `[0,1]` when defined |
| \(\eta_{th}\) | dimensionless | `[0,1]` in target-directed regime |

Add assertions/test helpers where feasible so bit/nat mismatches cannot silently pass.

---

# 12. Theory-simulation-empirical consistency tests

The implementation is not complete until these tests pass.

## 12.1 Pure theory invariants

For representative grids over \(N,q_c,b,\beta,\theta,h,\gamma\):

- every sensor row sums to one;
- every controlled-kernel row sums to one;
- `Q0 == identity`;
- `Q1 == matrix_power(K, b)`;
- `susceptibility_curve == kernel_mean_response(Q0,Q1,N)`;
- `pinsker_bound <= T_pi + tolerance` statewise;
- `T_pi <= binary_entropy_bits(a_n) + tolerance` statewise;
- `0 <= eta_IR <= 1` wherever defined;
- direct path KL equals `Delta S_sys + h J_c + I_sens`;
- `C_th == Sigma - Delta S_sys == h J_c + I_sens`;
- `0 <= eta_th <= 1` whenever the target-directed validity condition holds.

## 12.2 Calibration tests

Generate known finite values \((h,\gamma)\), set

\[
p_+=\gamma\sigma(h),\qquad p_-=\gamma\sigma(-h),
\]

and verify calibration recovers \((h,\gamma)\).

Also test the observed calibration example from the paper/report approximately:

```text
plus: 208 / 572
minus: 4 / 508
```

which should give approximately

```text
h_eff ~= 3.83
gamma_eff ~= 0.372
```

## 12.3 Susceptibility-unit test

Construct synthetic round data where the state-matched fraction response is known.

Verify:

```text
round_target_susceptibility == known fraction response
round_target_signed_actuation == K/(K-1) * round_target_susceptibility
```

when both use identical slices.

## 12.4 `eta_ir` regression test

Using a synthetic empirical channel with known \(Q_0,Q_1,a_n,p(n)\):

1. sample or enumerate events;
2. estimate state-matched fraction susceptibility;
3. compute empirical target CMI;
4. compute occupancy-level `eta_ir`;
5. verify convergence to the deterministic theory result as sample size increases;
6. verify replacing \(\chi_x\) with magnetization would produce the expected incorrect factor and is rejected by the implementation.

## 12.5 Sensing test

For a known occupancy \(p(n)\):

- `target_sensing_information_nats` from empirical occupancy + exact sensor kernel must match `theory_revised.sensing_information_nats`;
- direct-counting target sensing MI should converge to the same value as sample size increases;
- full-vector sensing MI must remain a different named metric.

## 12.6 Current test

For synthetic state-resolved \(p(n),a_n,\chi(n)\), verify

\[
J_c=N\sum_n p(n)a_n\chi(n).
\]

Add a regression test that deliberately creates correlated \(a_n\) and \(\chi_n\) and verifies the implementation does **not** use

\[
N\bar a\bar\chi.
\]

## 12.7 `eta_th` end-to-end synthetic test

Generate data from the single-affinity kernel itself. From the generated trajectories, estimate:

```text
h_hat
gamma_hat
chi_hat(n)
J_c_hat
I_sens_hat
eta_th_hat
```

and verify convergence to the exact values from `single_affinity_reference(...).one_cycle(...)` / finite-horizon theory.

This is the most important integration test because it closes the complete loop:

```text
simulation -> empirical estimators -> exact theory
```

## 12.8 Bootstrap test

Use fixed deterministic seeds and verify that episode bootstrap:

- resamples whole episodes rather than rounds independently;
- recomputes all nonlinear ingredients inside each replicate;
- produces deterministic CI output under a fixed bootstrap seed.

---

# 13. Acceptance criteria for Study 08

Before considering the migration complete, run the Study 08 offline analysis on a small existing or inspection result set and confirm all of the following.

### Response

- `round_target_susceptibility` exists;
- it is state-matched;
- it is in target-fraction units;
- magnetization response remains separately available.

### Information efficiency

- `eta_ir` uses the fraction-valued susceptibility;
- the numerator is built statewise and occupancy weighted;
- the result is not the legacy magnetization-scaled value;
- the output includes numerator, denominator, validity, and support fields.

### Thermodynamic ingredients

- `effective_affinity` is estimated from controlled microscopic transition odds with natural logs;
- `kinetic_compliance` is estimated from the same transitions;
- `controlled_current` is built from \(N\sum p a\chi\), not terminal `cell_current`;
- `target_sensing_information_nats` uses the scalar target channel, not full-vector sensing MI;
- finite-time sensing is aggregated by round rather than through an implicit stationary pool.

### Thermodynamic efficiency

- `eta_th` is actually computed and written to output tables;
- it is included in `derived:` rather than only metadata;
- it carries a bootstrap interval;
- its components `h*J_c` and `I_sens` are present in the same output;
- invalid or non-target-directed cases are flagged, not clipped.

### Theory comparison

For a calibrated cell, write side-by-side:

```text
empirical chi vs single-affinity chi
empirical T_pi / CMI vs single-affinity T_pi
empirical eta_ir vs single-affinity eta_IR
empirical J_c vs single-affinity J_c
empirical I_sens vs single-affinity I_sens
empirical eta_th vs single-affinity eta_th
```

The matched q-voter comparison remains separate.

---

# 14. Required output provenance

Every derived table row containing `eta_ir` or `eta_th` should carry enough metadata to reconstruct what was done:

```text
theory_semantics_version = single_affinity_v1
response_coordinate = target_fraction
response_conditioning = target_count_before
sensing_coordinate = target_count
sensing_log_base = e
actuation_information_log_base = 2
affinity_log_base = e
current_units = target_count_per_cycle
eta_ir_aggregation = occupancy_ratio_of_sums
eta_th_aggregation = finite_horizon_ratio_of_sums
bootstrap_unit = episode
```

This may live in columns, JSON metadata, or a versioned analysis manifest, but it must be persisted somewhere machine-readable.

---

# 15. Backward compatibility and migration policy

Do not silently overwrite legacy outputs with different semantics under the same name unless the project explicitly accepts that breaking change.

Recommended migration:

- add `round_target_susceptibility` as the new canonical name;
- keep `round_target_signed_actuation` and `round_target_signed_response_share` as legacy diagnostics;
- update `eta_ir` implementation but record an analysis-version field so old and corrected outputs are distinguishable;
- add `eta_th` as a genuinely new derived observable;
- leave `cell_current` unchanged and explicitly document that it is **not** the thermodynamic controlled current;
- leave `round_sensing_mi` unchanged if it is full-vector and add the scalar target sensing quantity under a new name.

Re-running offline analysis must be sufficient. No provider calls should be required if the existing retained records contain all needed round and microscopic fields.

If an existing artifact profile omitted a required microscopic field for \(h,\gamma\), fail clearly and state which field is missing rather than substituting a theory value.

---

# 16. Implementation order

Implement in this order so each stage has an independent correctness check.

## Phase A — canonicalize response

1. inspect current `_signed_response` implementation and Study 08 adapters;
2. add `round_target_susceptibility`;
3. add unit/factor tests against magnetization;
4. update plots/tables to expose the canonical \(\chi\).

## Phase B — repair `eta_ir`

1. replace magnetization response with canonical statewise fraction response;
2. implement occupancy-weighted Pinsker numerator;
3. ensure denominator/action mixing semantics match;
4. bootstrap the full derived calculation;
5. regression-test against `theory_revised.py` synthetic data.

## Phase C — thermodynamic ingredients

1. add scalar target sensing information in nats;
2. verify/reuse empirical \(h,\gamma\) calibration;
3. implement response-based \(J_c\);
4. add support/provenance fields.

## Phase D — empirical `eta_th`

1. implement one-cycle and finite-horizon decomposition;
2. add `h*J_c`, `C_th`, and `eta_th` outputs;
3. add whole-episode bootstrap;
4. add validity flags;
5. add resource-grid plots.

## Phase E — integration / Study 08 recipe

1. add new objects to `analysis.yaml::derived`;
2. update Study 08 metadata only after computation exists;
3. run offline analysis on inspection data;
4. run targeted tests;
5. run full existing test suite to ensure no regression;
6. document corrected-vs-legacy `eta_ir` semantics in analysis output/README.

---

# 17. Final scientific invariant

After this migration, the project should have one consistent chain:

\[
\boxed{
\widehat\chi(n)
\longrightarrow
\begin{cases}
\widehat T_\pi \longrightarrow \widehat\eta_{\mathrm{IR}},\\[2mm]
\widehat J_c,\widehat h,\widehat I_{\mathrm{sens}}
\longrightarrow \widehat\eta_{\mathrm{th}}.
\end{cases}
}
\]

with the same coordinates and units as the exact single-affinity theory:

\[
\chi:\text{ target fraction},
\qquad
T_\pi:\text{ bits},
\qquad
hJ_c, I_{\mathrm{sens}}:\text{ nats}.
\]

The empirical estimators, provider-free reference theory, and any future single-affinity simulator must therefore agree on definitions **before** numerical agreement is assessed.

The matched q-voter remains a separate classical reference, and the terminal episode current remains a separate behavioral outcome. Neither is to be substituted into the single-affinity efficiency formulas.

---

# 18. Definition of done

This task is complete only when all of the following are true in code, not merely in metadata or documentation:

- [ ] `round_target_susceptibility` is computed from state-matched target-fraction response.
- [ ] corrected empirical `eta_ir` consumes that susceptibility and passes the occupancy-weighted Pinsker bound tests.
- [ ] scalar target sensing information is available in nats.
- [ ] empirical `effective_affinity` and `kinetic_compliance` are calibrated from controlled microscopic transitions.
- [ ] empirical controlled current is computed as \(N\sum p a\chi\).
- [ ] empirical `eta_th` is computed from \(hJ_c/(hJ_c+I_{sens})\).
- [ ] finite-horizon `eta_th` uses a ratio of accumulated terms.
- [ ] both efficiencies are bootstrapped by resampling whole episodes and recomputing the complete estimator.
- [ ] support/overlap and validity diagnostics propagate into both efficiencies.
- [ ] Study 08 `derived:` actually requests the new computed objects.
- [ ] Study 08 headline metadata matches what is genuinely computed.
- [ ] theory-vs-empirical comparison tables contain \(\chi,T_\pi,\eta_{IR},J_c,I_{sens},\eta_{th}\).
- [ ] matched-q-voter outputs remain clearly separate.
- [ ] `cell_current` remains available but is never used as thermodynamic \(J_c\).
- [ ] full-vector `round_sensing_mi` remains available but is never used as single-affinity \(I_{sens}\) unless its definition is explicitly changed and versioned.
- [ ] synthetic single-affinity data recover the exact `theory_revised.py` quantities within sampling error.
- [ ] existing repository tests still pass.

