# Paper update handoff: finite-horizon aggregation for nonstationary control efficiencies

## Purpose

Update the current ICLR 2027 paper so that the manuscript explicitly matches the nonstationary aggregation semantics already used by the empirical single-affinity analysis.

The current paper correctly defines the theory **cycle by cycle**, but it does not yet explicitly define the **finite-horizon experiment-level efficiencies** that should be reported when the population distribution changes across rounds.

The goal of this edit is therefore to make the paper's reporting hierarchy explicit:

\[
\text{state-local theory}
\;\longrightarrow\;
\text{cycle-wise quantities}
\;\longrightarrow\;
\text{finite-horizon reported efficiencies}.
\]

Do **not** restructure the paper. Preserve the current section organization, notation, plots, and theoretical framing as much as possible. This is a targeted clarification/update.

---

# 1. Why this update is needed

The current theory is explicitly nonstationary.

At feedback cycle \(k\), the population distribution is

\[
p_k(n)=\Pr(n_k=n),
\]

and this distribution generally changes with \(k\).

The paper already defines:

- the state-local susceptibility \(\chi(n)\),
- the state-local action-to-population information \(T_\pi(n)\),
- the state-local information-response efficiency \(\eta_{\rm IR}(n)\),
- the cycle-wise controlled current \(J_{c,k}\),
- the cycle-wise sensing information \(I_{{\rm sens},k}\),
- the cycle-wise thermodynamic efficiency \(\eta_{{\rm th},k}\).

The current finite-time identity is

\[
\Delta S_{{\rm sys},k}
+hJ_{c,k}
+I_{{\rm sens},k}
=
\Sigma_k.
\]

However, the manuscript does not yet state explicitly how these quantities are aggregated over a finite experimental horizon of \(H\) rounds.

That omission matters because:

1. the population is nonstationary, so \(p_k(n)\) changes with \(k\);
2. efficiency is a nonlinear ratio, so the correct horizon-level quantity is not an arithmetic average of per-cycle efficiencies;
3. sensing mutual information is nonlinear in the occupancy distribution, so in general it cannot be computed once from the pooled finite-horizon occupancy.

The paper should therefore define finite-horizon aggregation explicitly.

---

# 2. Canonical hierarchy to preserve throughout the paper

The revised manuscript should make the following distinction clear.

## 2.1 State-local quantities

For fixed controller parameters and a fixed current target count \(n\),

\[
\chi(n),\qquad
T_\pi(n),\qquad
\eta_{\rm IR}(n)
\]

are state-local quantities.

They describe the geometry of the controller channel at a given state.

They do not require stationarity.

---

## 2.2 Cycle-wise quantities

Once the transient occupancy \(p_k(n)\) enters, the quantities become cycle dependent:

\[
J_{c,k}
=
N\sum_n p_k(n)a_n\chi(n),
\]

\[
I_{{\rm sens},k}
=
I(n_k;Y_k),
\]

and

\[
\eta_{{\rm th},k}
=
\frac{hJ_{c,k}}
{hJ_{c,k}+I_{{\rm sens},k}}.
\]

These are finite-time cycle-wise quantities.

---

## 2.3 Finite-horizon quantities

For an experiment of \(H\) feedback cycles, the primary reported efficiencies should be formed from accumulated numerator and denominator terms.

The finite-horizon thermodynamic efficiency is

\[
\boxed{
\eta_{\rm th}^{(H)}
=
\frac{
h\sum_{k=0}^{H-1}J_{c,k}
}{
h\sum_{k=0}^{H-1}J_{c,k}
+
\sum_{k=0}^{H-1}I_{{\rm sens},k}
}
}
\]

and **not**

\[
\frac1H\sum_{k=0}^{H-1}\eta_{{\rm th},k}.
\]

Likewise, the finite-horizon information-response efficiency should be a ratio of accumulated state-weighted bound terms to accumulated action-to-population information.

---

# 3. Exact paper edits

## Edit A — Section 5.4: Information-response efficiency

### Current location

Section 5.4 currently defines the state-local quantity

\[
\eta_{\rm IR}(n)
=
\frac{
2a_n(1-a_n)\chi_{h,\gamma}(n/N)^2
}{
(\ln2)T_\pi(n)
}
\le1.
\]

This state-local definition is correct and should remain unchanged.

### Required addition

Add a short paragraph at the end of Section 5.4, after the current explanation of the Pinsker efficiency and before Section 5.5.

### Suggested text

> **Finite-horizon aggregation.** The efficiency above is defined locally at a fixed population state. In a nonstationary experiment, the state occupancy changes from cycle to cycle. Let
> \[
> B_{\rm IR}(n)
> \equiv
> \frac{2a_n(1-a_n)}{\ln2}\chi_{h,\gamma}(n/N)^2
> \]
> denote the state-local Pinsker response term. For a finite horizon of \(H\) feedback cycles, we define the experiment-level information-response efficiency by accumulating the state-weighted numerator and denominator before forming the ratio:
> \[
> \eta_{\rm IR}^{(H)}
> =
> \frac{
> \sum_{k=0}^{H-1}\sum_n p_k(n)B_{\rm IR}(n)
> }{
> \sum_{k=0}^{H-1}\sum_n p_k(n)T_\pi(n)
> }.
> \]
> Because \(B_{\rm IR}(n)\le T_\pi(n)\) state by state, the finite-horizon ratio remains bounded,
> \[
> 0\le\eta_{\rm IR}^{(H)}\le1.
> \]
> This quantity is distinct from an arithmetic average of the state-local ratios \(\eta_{\rm IR}(n)\): it measures the fraction of the total action-dependent information transmitted over the evaluation horizon that is expressed through the response lower bound.

### Why this belongs in the main body

Section 6.4 currently refers to “occupancy-averaged empirical values,” but the aggregation rule is not explicitly defined. The text above makes the reported empirical quantity mathematically unambiguous and matches the corrected implementation.

---

# 4. Edit B — Section 5.5: finite-horizon thermodynamic efficiency

## Current location

Section 5.5 correctly defines, for one cycle,

\[
C_{{\rm th},k}
=
\Sigma_k-\Delta S_{{\rm sys},k}
=
hJ_{c,k}+I_{{\rm sens},k},
\]

and

\[
\eta_{{\rm th},k}
=
\frac{
hJ_{c,k}
}{
hJ_{c,k}+I_{{\rm sens},k}
}
\le1.
\]

The section also correctly notes that no stationarity assumption is required.

## Required addition

Add a paragraph near the end of Section 5.5, after the cycle-wise efficiency definition and before Section 6.

## Suggested text

> **Finite-horizon aggregation.** The quantities above are defined for one feedback cycle because the population distribution \(p_k(n)\) generally changes with \(k\). For an experiment of \(H\) feedback cycles, we therefore aggregate the thermodynamic accounting before forming the efficiency ratio. Summing the cycle-wise identity gives
> \[
> S_{\rm sys}[p_H]-S_{\rm sys}[p_0]
> +
> h\sum_{k=0}^{H-1}J_{c,k}
> +
> \sum_{k=0}^{H-1}I_{{\rm sens},k}
> =
> \sum_{k=0}^{H-1}\Sigma_k,
> \]
> where the system-entropy increments telescope. The corresponding finite-horizon non-storage expenditure is
> \[
> C_{\rm th}^{(H)}
> =
> h\sum_{k=0}^{H-1}J_{c,k}
> +
> \sum_{k=0}^{H-1}I_{{\rm sens},k},
> \]
> and the finite-horizon thermodynamic efficiency is
> \[
> \boxed{
> \eta_{\rm th}^{(H)}
> =
> \frac{
> h\sum_{k=0}^{H-1}J_{c,k}
> }{
> h\sum_{k=0}^{H-1}J_{c,k}
> +
> \sum_{k=0}^{H-1}I_{{\rm sens},k}
> }.
> }
> \]
> Thus the experiment-level efficiency is a ratio of accumulated directed current to accumulated non-storage expenditure, rather than an arithmetic mean \(H^{-1}\sum_k\eta_{{\rm th},k}\). This distinction is essential in the nonstationary regime because the control expenditure can vary substantially across cycles.

## Important interpretation

The primary experiment-level thermodynamic efficiency should be the ratio above.

Do not redefine the cycle-wise \(\eta_{{\rm th},k}\). Both quantities are useful:

- \(\eta_{{\rm th},k}\): cycle-wise diagnostic;
- \(\eta_{\rm th}^{(H)}\): finite-horizon headline quantity.

---

# 5. Edit C — Appendix A.2.4: formal finite-horizon derivation

## Current location

Appendix A.2.4 currently derives

\[
\eta_{{\rm th},k}
=
\frac{hJ_{c,k}}
{hJ_{c,k}+I_{{\rm sens},k}}
\]

and then gives the stationary limit

\[
\eta_{\rm th}=\frac{hJ_c}{\Sigma}.
\]

## Required addition

Add a short derivation immediately after the current cycle-wise efficiency and before or after the stationary-limit paragraph.

## Suggested derivation

Start from the cycle-wise identity

\[
\Delta S_{{\rm sys},k}
+hJ_{c,k}
+I_{{\rm sens},k}
=
\Sigma_k.
\]

Summing over \(k=0,\ldots,H-1\),

\[
\sum_{k=0}^{H-1}\Delta S_{{\rm sys},k}
+
h\sum_{k=0}^{H-1}J_{c,k}
+
\sum_{k=0}^{H-1}I_{{\rm sens},k}
=
\sum_{k=0}^{H-1}\Sigma_k.
\]

Because

\[
\sum_{k=0}^{H-1}\Delta S_{{\rm sys},k}
=
S_{\rm sys}[p_H]-S_{\rm sys}[p_0],
\]

we obtain

\[
S_{\rm sys}[p_H]-S_{\rm sys}[p_0]
+
hJ_c^{(H)}
+
I_{\rm sens}^{(H)}
=
\Sigma^{(H)},
\]

where

\[
J_c^{(H)}
\equiv
\sum_{k=0}^{H-1}J_{c,k},
\qquad
I_{\rm sens}^{(H)}
\equiv
\sum_{k=0}^{H-1}I_{{\rm sens},k},
\qquad
\Sigma^{(H)}
\equiv
\sum_{k=0}^{H-1}\Sigma_k.
\]

The accumulated non-storage expenditure is therefore

\[
C_{\rm th}^{(H)}
=
hJ_c^{(H)}+I_{\rm sens}^{(H)},
\]

and

\[
\eta_{\rm th}^{(H)}
=
\frac{hJ_c^{(H)}}{C_{\rm th}^{(H)}}.
\]

Add the explicit sentence:

> In general,
> \[
> \eta_{\rm th}^{(H)}
> \neq
> \frac1H\sum_{k=0}^{H-1}\eta_{{\rm th},k},
> \]
> because the thermodynamic efficiency is a nonlinear ratio and the cycle-wise expenditures need not be equal.

This derivation should remain concise. The main body should carry the definition; the appendix should justify it formally.

---

# 6. Edit D — Appendix D.3: clarify nonstationary occupancy averaging

## Current location

Appendix D.3 currently introduces

\[
\rho_k(n)=\Pr(n_k=n)
\]

and the finite-horizon visitation distribution

\[
\bar\rho_H(n)
=
\frac1H\sum_{k=0}^{H-1}\rho_k(n).
\]

It correctly states that the operational design problem averages over transient states rather than over an equilibrium distribution.

## Required clarification

Add a paragraph immediately after the definition of \(\bar\rho_H\).

## Suggested text

> The visitation distribution \(\bar\rho_H\) can be used directly for observables that are linear in the occupancy, such as the expected controlled response and current. For example,
> \[
> \frac1H\sum_{k=0}^{H-1}J_{c,k}
> =
> N\sum_n\bar\rho_H(n)a_n\chi(n).
> \]
> Sensing information is different because mutual information is nonlinear in the state distribution. In general,
> \[
> I_{\rm sens}[\bar\rho_H]
> \neq
> \frac1H\sum_{k=0}^{H-1}I_{\rm sens}[\rho_k].
> \]
> Finite-horizon thermodynamic accounting therefore evaluates \(I_{{\rm sens},k}\) from each transient distribution \(\rho_k\) and accumulates the resulting cycle-wise contributions before forming \(\eta_{\rm th}^{(H)}\). This preserves the finite-time interpretation without imposing stationarity or replacing the transient sequence by a single pooled ensemble.

This paragraph is important.

Do not use a single pooled occupancy to compute the finite-horizon thermodynamic sensing term.

---

# 7. Results-section wording that should be updated

## Section 6.4

The phrase “occupancy-averaged empirical values” should be made more precise.

Replace or augment it with wording such as:

> For each controller configuration, we report the finite-horizon information-response efficiency \(\eta_{\rm IR}^{(H)}\), obtained as the ratio of the accumulated occupancy-weighted Pinsker response terms to the accumulated state-local action-to-population information over the evaluation horizon.

This avoids suggesting that the reported quantity is simply the arithmetic mean of state-local efficiencies.

---

## Section 6.5

When empirical \(\eta_{\rm th}\) is inserted, explicitly describe it as a finite-horizon quantity.

Suggested wording:

> For the empirical trajectories, we report the finite-horizon thermodynamic efficiency \(\eta_{\rm th}^{(H)}\). The affinity-weighted controlled current and scalar target-sensing information are accumulated cycle by cycle and the efficiency ratio is formed only after this accumulation. We therefore do not average the cycle-wise efficiencies, and we do not compute sensing information from a single pooled occupancy distribution.

This wording should be adapted to the final empirical results once Study 08 numbers are available.

---

# 8. Important implementation-paper consistency requirements

The paper should remain aligned with the empirical single-affinity implementation on all of the following points.

## 8.1 Susceptibility coordinate

The canonical susceptibility is the state-matched response in **target-fraction units**:

\[
\chi(n)
=
E[\Delta x\mid U=1,n]
-
E[\Delta x\mid U=0,n],
\qquad
x=\frac{n_Z}{N}.
\]

Do not describe aligned magnetization as the theoretical susceptibility.

---

## 8.2 Information-response efficiency

The state-local theory remains

\[
\eta_{\rm IR}(n)
=
\frac{
2a_n(1-a_n)\chi(n)^2
}{
(\ln2)T_\pi(n)
}.
\]

The finite-horizon headline is a **ratio of accumulated terms**, not an arithmetic average of local efficiencies.

---

## 8.3 Controlled current

The empirical/theoretical current corresponding to the single-affinity theory is

\[
J_{c,k}
=
N\sum_n p_k(n)a_n\chi(n).
\]

Do not substitute terminal episode drift or a generic cell-level current for \(J_c\).

---

## 8.4 Sensing information

The thermodynamic sensing quantity is the scalar target channel

\[
I_{{\rm sens},k}
=
I(n_{Z,k};Y_{Z,k})
\]

in **nats**.

It is not the full \(K\)-option vector sensing mutual information.

---

## 8.5 Affinity

The empirical affinity is

\[
h_{\rm eff}
=
\ln\frac{p_+}{p_-}
\]

from controlled microscopic target/non-target transitions.

The same measured affinity enters susceptibility calibration, current, and the thermodynamic efficiency.

---

## 8.6 Nonstationary horizon

The finite-horizon thermodynamic efficiency must use

\[
\sum_k I_{{\rm sens},k},
\]

not

\[
I_{\rm sens}[\bar p_H].
\]

The latter is generally different because mutual information is nonlinear in the occupancy.

---

# 9. What should remain unchanged

Do not unnecessarily rewrite the following:

- Section 5.1 sensor law and policy;
- Section 5.2 single-affinity finite-compliance kernel;
- the exact susceptibility derivation;
- the state-local \(T_\pi(n)\) definition;
- the Pinsker derivation in Appendix A.1;
- the cycle-wise path-space identity;
- the definition of the coarse system entropy;
- the target-directed condition \(hJ_c\ge0\);
- the calibrated \(h_{\rm eff}\) and \(\gamma_{\rm eff}\) discussion;
- the existing state-local theory figures;
- the resource-constrained design section, except for the nonstationary clarification in D.3.

This task is an aggregation/interpretation clarification, not a new theoretical model.

---

# 10. Equation-numbering guidance

Do not hard-code the equation numbers in this handoff.

The current PDF has approximately:

- Section 5.4 local \(\eta_{\rm IR}\): current Eq. (33),
- Section 5.5 cycle-wise thermodynamic identity/efficiency: current Eqs. (36)–(38),
- Appendix A.2 cycle-wise identity: current Eqs. (94)–(100),
- Appendix D.3 visitation distribution: current Eq. (113).

Adding equations will renumber downstream equations automatically.

Use LaTeX labels and references consistently with the repository's current conventions.

---

# 11. Definition of done

The paper update is complete when all of the following are true:

- [ ] Section 5.4 defines the finite-horizon \(\eta_{\rm IR}^{(H)}\) as a ratio of accumulated terms.
- [ ] Section 5.5 defines the finite-horizon \(\eta_{\rm th}^{(H)}\).
- [ ] Section 5.5 explicitly states that \(\eta_{\rm th}^{(H)}\) is not the arithmetic mean of cycle-wise efficiencies.
- [ ] Appendix A.2 formally derives the horizon-level thermodynamic identity by summing over cycles.
- [ ] The telescoping boundary term
  \[
  S_{\rm sys}[p_H]-S_{\rm sys}[p_0]
  \]
  appears explicitly.
- [ ] Appendix D.3 explains that occupancy averaging is valid directly for linear observables but not generally for sensing MI.
- [ ] Appendix D.3 explicitly states
  \[
  I_{\rm sens}[\bar\rho_H]
  \neq
  H^{-1}\sum_k I_{{\rm sens},k}
  \]
  in general.
- [ ] Section 6.4 no longer leaves “occupancy-averaged \(\eta_{\rm IR}\)” mathematically ambiguous.
- [ ] Section 6.5 describes empirical thermodynamic efficiency as a finite-horizon ratio when Study 08 results are inserted.
- [ ] The manuscript remains explicit that the theory is finite-time and does not assume stationarity.
- [ ] No existing state-local definitions are silently changed.

---

# 12. Conceptual summary for the paper

The final paper should make the following three-level distinction transparent:

\[
\boxed{
\text{state-local}
\quad\rightarrow\quad
\text{cycle-wise}
\quad\rightarrow\quad
\text{finite-horizon}
}
\]

with

\[
\chi(n),\;
T_\pi(n),\;
\eta_{\rm IR}(n)
\]

at the state-local level,

\[
J_{c,k},\;
I_{{\rm sens},k},\;
\eta_{{\rm th},k}
\]

at the cycle level, and

\[
\eta_{\rm IR}^{(H)},\;
\eta_{\rm th}^{(H)}
\]

as the experiment-level reported efficiencies.

The nonstationarity is not a nuisance to hide or average away. It is handled explicitly by keeping the transient occupancy \(p_k(n)\), evaluating the relevant cycle-wise quantities at each \(k\), and only then forming finite-horizon accumulated ratios.

That is the aggregation semantics the paper should state explicitly so that theory, simulation, and empirical estimation remain fully aligned.
