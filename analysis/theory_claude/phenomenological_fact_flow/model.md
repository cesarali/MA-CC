# Fact-flow model: equations and derived results

Phenomenological model following the researcher's plan (`phenomenological_fact_flow_theory_plan.md`, 27 Oct 2026), with
one revision of the reposting closure (§1.4). No exact Bayesian inference, no task world, no M1–M4 code.

## 1. Model

**Classes.** A = agents' evidence, which supports the correct answer; C = controller evidence, which supports the wrong
answer A2. Both are true facts. Each class has F_α exchangeable facts.

**State.**
- m_α(t): the probability that an agent holds a given class-α fact;
- b_α(t): the share of the readable board carrying class-α posts;
- M_α = F_α m_α: the expected number of distinct class-α facts per agent.

Agents act at rate 1 and read q posts. Each fact is forgotten at rate γ = −ln ρ. The board turns over at rate N/W, and
the controller posts at rate λ(t) = λ₀ for t < T.

### 1.1 Memory (plan Eq. 1)
$$\dot m_\alpha=(1-m_\alpha)\,r_\alpha-\gamma m_\alpha,\qquad r_\alpha=1-\Big(1-\frac{b_\alpha}{F_\alpha}\Big)^q .$$

### 1.2 Board (plan Eq. 2)
$$\dot b_\alpha=\frac NW\big(S_\alpha-b_\alpha\big)+\frac{\lambda(t)}W\big(u_\alpha-b_\alpha\big),\qquad u_C=1,\ u_A=0.$$

### 1.3 Vote response (plan Eq. 3, with z = 0: no proof variable)
$$\Phi_2=\sigma(x),\qquad x=h_0+\beta_C M_C-\beta_A M_A,\qquad \sigma(x)=\frac1{1+e^{-x}}.$$

### 1.4 Reposting (plan Eq. 5, revised: softened vote gating ε)
$$w_C=\kappa_C M_C\,g_C,\quad w_A=\kappa_A M_A\,g_A,\quad g_C=\varepsilon+(1-\varepsilon)\Phi_2,\quad g_A=\varepsilon+(1-\varepsilon)(1-\Phi_2),$$
$$S_C=\frac{w_C}{w_A+w_C+\kappa_0},\qquad S_A=\frac{w_A}{w_A+w_C+\kappa_0}.$$

- ε = 0 is the plan's closure (5): only agents who believe A2 repost controller facts.
- ε = 1 is neutral reposting.
- ε is the share of reposting eligibility that does not depend on the vote. In the simulator, a held controller fact is
  eligible about 0.8 of the time in task003 and 0.55 in Object Placements.

### 1.5 Observables (plan Eq. 6)
The controlled arm has λ₀ > 0. The silent arm has λ = 0 and m_C = b_C = 0. Then
G = Φ₂(controlled) − Φ₂(silent) at T = 30, and D(q₁→q₂) = G(q₂) − G(q₁). After T, δ(t) = the same difference, with both
arms under the same continuation.

**Continuations from T:**
- normal: λ = 0;
- frozen board: b fixed at b(T), no ordinary posting;
- no communication: r = 0, no ordinary posting;
- controller facts removed: m_C = b_C = 0 at T, then normal.

**Parameters (P0):** N/W = 0.5, F_A = 16, F_C = 4, β_A = β_C = 0.5, h₀ = ln ½ (prior P(A2) = 1/3), κ_A = κ_C = 1,
κ₀ = 0.2, λ₀ = 1, W = 48, T = 30; initial m_A = 1/F_A, m_C = b = 0.

## 2. Derived results

### 2.1 Frozen board: exact holding share
With b fixed, Eq. (1.1) is linear: m_α(t) → m_α* = r_α/(r_α + γ) > 0 for q > 0, γ > 0. This is the same form as the
simulator's exact single-fact result (Phase A4). δ stays at its value at T, because both arms relax to their own fixed
board and the board carries the controller's facts.

### 2.2 No communication: decay and a transient rise
With r = 0, m_α(T+s) = m_α(T) e^{−γs}, so x(s) = h₀ + e^{−γs}(x(T) − h₀) in each arm, and δ → σ(h₀) − σ(h₀) = 0. The
initial slope is
$$\frac{d\delta}{ds}\Big|_0=-\gamma\Big[\sigma'(x_c)(x_c-h_0)-\sigma'(x_s)(x_s-h_0)\Big].$$
It is **positive** when the silent arm's evidence term, weighted by its slope, outweighs the controlled arm's. Then
both arms' evidence fades and both slide back toward the prior, but the controlled arm's A2 belief rises faster, because
the agents' evidence that held it down is fading. The result is a transient rise of δ before it decays. This appears in
the model at ρ = 0.75 (δ: 0.027 → peak 0.063), and in the simulator's symmetric setups at ρ = 0.75. The model has no
proofs, so here the cause is fading evidence, not broken proofs.

### 2.3 The reading reversal: dose against sensitivity
For a small effect, G ≈ σ′(x_s) β_C M_C: the population's **sensitivity** σ′ at the silent arm's operating point,
times the controller's **dose** M_C. Differencing in q:
$$D\approx\beta_C\,\sigma'(x_s)\Big[\Delta M_C-(1-2\Phi_s)\,\beta_A\,M_C\,\Delta M_A\Big].$$
- **Dose term (+):** reading spreads the controller's evidence.
- **Sensitivity term (−, when Φ_s < 1/2):** reading also adds agents' evidence, moving everyone away from the tipping
  point, where the sigmoid is flatter.

With short memory ΔM_A is small, and the dose term wins (D > 0). With long memory agents accumulate their own evidence
(ΔM_A large), and the sensitivity term wins (D < 0). As memory goes to 1 both terms vanish, because everyone is
confident.

The formula gets the sign of the full model's D except where D ≈ 0, and its size within a factor of about 2
(`findings/stageA_approx.csv`, Fig. 5).

**What the reversal does not need:**
- a proof variable (z = 0 throughout);
- selective reposting (it holds at every ε, and for neutral reposting);
- concentrated controller evidence (it holds with F_C = F_A = 16).

It needs a decision rule with a tipping point, an operating point below it (Φ_s < 1/2), and memory that lets agents'
own evidence accumulate.

### 2.4 Persistence: a growth threshold for controller facts after the stop
Linearise Eqs. (1.1)–(1.2) for the controller class around a state with few controller facts, after the stop (λ = 0):
$$\dot m_C\simeq\frac q{F_C}b_C-\gamma m_C,\qquad \dot b_C\simeq\frac NW\Big(\frac{\kappa_C F_C\,g_C}{H}m_C-b_C\Big),\qquad H=\kappa_A M_A g_A+\kappa_0 .$$
The trace of the 2×2 matrix is negative. The determinant is positive iff the growth number
$$\boxed{\mathcal R_C=\frac{q\,\kappa_C\,g_C}{\gamma\,H}<1}.$$
So controller facts die out when R_C < 1 and keep circulating when R_C > 1. N/W sets the speed, not the threshold. This
is the plan's Eq. (8), with the vote-gating factor g_C = ε + (1 − ε)Φ₂ in place of Φ₂.

Numerically (revision-3 grid), the retained fraction δ(130)/δ(30) is about 0 when R_C, evaluated at the stop, is below
1, and it rises with R_C above 1 (Fig. 4). In the mean-field model "kept circulating" means a stable mixed state, in which
δ relaxes in 5–9 time units to a plateau. In a finite swarm, this state would eventually be lost to extinction (not
modelled here).

**The role of selection.** With full vote gating (ε = 0), g_C = Φ₂ is small unless the controller has already convinced
many agents. Then R_C < 1 and controller facts die out. With strong reposting advantage (κ_C/κ_A ≥ 4), the model can
instead switch into a controller-dominated state (bistability, `findings/rev2_grid.csv`). Vote-independent eligibility
(ε near 1) gives **moderate, stable persistence**, which is the simulator's pattern. Lowering ε lowers R_C and the
retained fraction: high eligibility gives task003-like persistence, lower eligibility Object-Placements-like partial
fading.
