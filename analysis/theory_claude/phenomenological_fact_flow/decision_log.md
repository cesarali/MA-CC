# Decision log — phenomenological fact-flow model

Plan: researcher's `phenomenological_fact_flow_theory_plan.md` (27 Oct 2026). Equations numbered as in the plan. One
entry per decision, newest last. Every version of the model is kept.

## 2026-10-27 — Stage A setup (written before any run)

**Model V1 = plan Eqs. (1), (2), (3), (5), (6) with z = 0** (no proof variable). Classes A (agents' evidence) and C
(controller evidence).
- Per-fact memory m_α and board share b_α.
- Reading: r_α = 1 − (1 − b_α/F_α)^q.
- Board turnover N/W.
- Controller injection λ₀ = 1 for t < T = 30.
- Vote response Φ₂ = σ(h₀ + β_C F_C m_C − β_A F_A m_A).
- Selective reposting: S_α ∝ κ_α M_α × (Φ₂ for C, 1 − Φ₂ for A), with background weight κ₀.
- Silent arm: λ = 0 and m_C = b_C = 0 throughout.
- G = Φ₂(controlled) − Φ₂(silent) at T. D = G(q₂) − G(q₁).

**Illustrative parameter set P0 (chosen before any run, not tuned):**
- N/W = 24/48 = 0.5;
- F_A = 16, F_C = 4 (the controller recycles few facts; agents' evidence is spread);
- β_A = β_C = 0.5 (equal strength per fact);
- h₀ = ln(1/2) (prior P(A2) = 1/3 against the other two answers);
- κ_A = κ_C = 1, κ₀ = 0.2.

Initial state: each agent holds one A fact on average (m_A(0) = 1/F_A); m_C(0) = 0; board empty (b_A = b_C = 0).
Grid: ρ ∈ (0.05, 0.99), q ∈ {3, 6, 9, 12}.

**Expectations stated in advance:**
- E1 (reversal, no proof): with F_C ≪ F_A, D(3→6) > 0 at small ρ and < 0 at large ρ (differential saturation:
  concentrated controller evidence saturates in memory first when memory is long). If true, the reversal should
  disappear when F_C = F_A, with everything else equal.
- E2 (persistence): with selective reposting, controller information is replenished after T and δ decays much more
  slowly than e^{−γt}. With neutral reposting (S_α ∝ M_α), the controller has no advantage after T.
- E3 (frozen board / no communication): the frozen board keeps m_α* = r_α/(r_α + γ) > 0; no communication lets
  memories decay at rate γ. These follow from the equations and are checks of the code.

Stop rule (plan §7): if E1 fails, vary one interpretable mechanism at a time (κ_C/κ_A, β_C/β_A, F_C/F_A, N/W) and log
each revision with the requirement it addresses. No q-by-ρ switching, no tuning of the sign of D.

## 2026-10-27 — Stage A and D results at P0; revision 1 (written before running it)

**Stage A (reversal) at P0: present.** D(3→6) = +0.0033 at ρ 0.05 and −0.0045 at ρ 0.5. The crossover falls with q
(3→6: 0.2–0.3; 6→9: 0.1–0.2; 9→12: 0.05–0.1), as in the simulator. It holds in 16 of 22 single-parameter variants
(×0.5, ×2), and it fails for β_A ×0.5 or ×2, κ_A ×0.5, κ_C ×2, F_A ×0.5 and h₀ = 0. Effects are tiny (G ≈ 0.016).

**E1's mechanism is wrong.** The reversal survives F_C = F_A, and it survives neutral reposting (even stronger there).
The chain-rule split (`stageA_mechanism.py`) shows the real mechanism, **dose against sensitivity**. For a small
effect, G ≈ σ′(x_s) β_C M_C, and

  D ≈ β_C σ′(x_s) [ΔM_C − (1 − 2Φ_s) β_A M_C ΔM_A].

Reading raises the controller's dose (ΔM_C > 0). It also adds agent evidence to both arms; when Φ_s < 1/2, this moves
the population away from the tipping point and lowers its sensitivity σ′. Long memory makes ΔM_A large, and the second
term wins. The approximation gets the sign right except where D ≈ 0 (`results/stageA_approx.csv`).

**Stage D (persistence) at P0: E3 met, E2 failed.** The frozen board keeps δ exactly; no communication and removal
send it to 0. Under normal continuation, *selective* reposting loses everything within about 10 time units, while
*neutral* reposting keeps 65–75%. That is the opposite of what we need. Eq. (8) explains it: with
R_C = qκ_CΦ₀/(γH₀), the selective rule has Φ₀ ≈ 0.2 (agents rarely believe A2, so they rarely repost C facts), giving
R_C ≈ 0.2 < 1; the neutral rule gives ≈ 1.3 > 1.

**Diagnosis:** P0's controller is far too weak (G ≈ 0.015 against the simulator's 0.1–0.4). With selective reposting,
persistence needs the controller to push enough agents toward A2 that its facts are reposted: a feedback threshold.

**Revision 1 (one interpretable change): controller facts carry more evidence each, β_C/β_A ∈ {1, 2, 3, 4}**,
everything else at P0. Justification: in the simulator the controller *selects* its strongest facts (single-fact
P(A2) up to 0.59, against about 0.35 for agent facts). Check both phenomena at the *same* parameters for each ratio:
- the reversal for 3→6;
- normal-continuation retention at ρ 0.25 and 0.75;
- R_C at the controller-free state and R_eff at the state at T (with Φ and H evaluated at T).

## 2026-10-27 — revision 2 result; revision 3 (written before running it)

**Revision 2 result** (`results/rev2_grid.csv`; κ_C/κ_A × β_C/β_A).
- At κ_C/κ_A = 1–2: the reversal holds (in 6 of 8 points), but there is no persistence anywhere (controller facts die
  out, R_eff < 1).
- At κ_C/κ_A ≥ 4: at ρ 0.25 the controlled population switches into a **controller-dominated state** after the
  injection and stays there (G 0.6–0.8, retention 1.0). At ρ 0.75 it relaxes. This is **bistability** (lock-in), and
  the sign change of D there comes from crossing the switching point, a different mechanism from P0's.
- Both phenomena together: only κ_C/κ_A = 8, β_C/β_A = 3, a corner whose neighbours fail. **Not an open region.**

Reading: with closure (5)'s full vote gating, persistence exists only as lock-in. The simulator's persistence is a
moderate, nearly flat δ (task003 0.3–0.4), which looks like **stable coexistence** of controller facts with agent facts
rather than a switch. Neutral reposting at P0 gave that pattern (retention 0.65–0.75).

**Revision 3 (the single change announced in revision 2): soften the vote gating.**
- W_C = κ_C M_C [ε + (1 − ε) Φ₂];
- W_A = κ_A M_A [ε + (1 − ε)(1 − Φ₂)].

ε = 0 is the plan's closure (5); ε = 1 is neutral. ε is a vote-independent repost eligibility. In the simulator, a held
controller fact is eligible about 0.8 of the time in task003 and 0.55 in OP, so ε plausibly differs by task. Grid:
ε ∈ {0, 0.25, 0.5, 0.75, 1} × β_C/β_A ∈ {1, 2}, κ_C = κ_A = 1, otherwise P0.

**Prediction stated now:**
- retention rises with ε, through partial fading at intermediate ε up to near-flat persistence at ε → 1 (task003-like
  at high ε, OP-like at lower ε);
- the reversal survives at every ε, because it is a dose-against-sensitivity effect that does not need selection;
- persistence occurs where the post-stop R_eff (with the ε-weights) exceeds 1.

## 2026-10-27 — revision 3 result; stop for this round

**Revision 3 result** (`results/rev3_grid.csv`).
- The reversal holds at every ε, as predicted.
- Retention rises with ε, as predicted: ρ 0.25 / 0.75 = 0/0 (ε = 0), 0/0 (0.25), 0/0.29 (0.5), 0.32/0.63 (0.75),
  0.65/0.75 (1).
- The prediction "persistence where R_eff > 1" holds: retention is ≈ 0 for R_eff < 1 and rises above it (Fig. 4).
- Both phenomena hold for ε ≳ 0.75 at β_C/β_A ∈ {1, 2}.

Revisions used: three, each a single change.

The model now shows both phenomena qualitatively, with derived explanations (dose against sensitivity; growth threshold
R_C). Known mismatches: the persistence time course (the plateau is reached in 5–9 units, while the simulator is flat or
declines slowly); no residual after removal; no proof variable yet. Written up in `phenomenological_theory_report.md`.
Next proposed step: Stage C (complementarity z), then the finite-copy refinement.
