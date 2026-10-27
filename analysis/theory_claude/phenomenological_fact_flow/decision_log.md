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
term wins. The approximation gets the sign right except where D ≈ 0 (`findings/stageA_approx.csv`).

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

**Revision 2 result** (`findings/rev2_grid.csv`; κ_C/κ_A × β_C/β_A).
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

**Revision 3 result** (`findings/rev3_grid.csv`).
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

## 2026-10-27 — Stage C: one complementarity variable (written before running it)

The researcher asked whether the model covers the symmetric setups and Object Placements. The model has no task. z = 0
means no proof can form, as in the no-proof setups. **Stage C adds the plan's single complementarity variable**:
- z = Z(m_A) = 1 − (1 − m_A^ℓ)^P (plan Eq. 4), the chance that an agent holds at least one complete proof (independence
  approximation);
- Φ₂ = (1 − z) σ(x): a complete proof sets the belief in A2 to 0, as in the simulator;
- applied in both arms, and in the reposting weights through Φ₂.

ℓ and P are taken from the worlds, not fitted. They are the size and number of the smallest agent-fact proofs of A0 in
the symmetric setups: **task003 ℓ = 6, P = 12; Object Placements ℓ = 3, P = 2** (`../reports/2026-10-23_task_comparison/
task_structure.csv`). Everything else at the current best version (revision 3, ε = 1, P0).

**Expectations stated in advance:**
- C1: the reversal survives with z.
- C2: proofs make reading protective at shorter memory, so the 3→6 crossover moves to *lower* ρ than with z = 0, as in
  the simulator, where the symmetric crossover is lower than the no-proof one in both tasks.
- C3: the shift is larger for ℓ = 3 (OP-like) than for ℓ = 6 (task003-like), because a 3-fact proof forms at much lower
  memory. In the simulator, OP's symmetric crossover (0.25–0.375) is lower than task003's (0.5–0.625).
- C4: at long memory proofs push G towards 0 (the simulator's symmetric G → 0 at ρ = 1).

## 2026-10-27 — Stage C result

`findings/stageC_proof.csv`:
- **C1 met:** the reversal survives with z (both proof shapes).
- **C4 met:** proofs send G to 0 at long memory (G at ρ 0.99: 0.0014 with no proof, 0 with either proof).
- **C2 and C3 not met:** the 3→6 crossover does not move (0.25–0.3 in all three variants).

**Diagnosis:** in the model, proofs form only at very long memory. The share of agents holding a proof at ρ 0.5 / 0.9 is
0.01 / 0.91 (ℓ = 6) and 0.06 / 0.67 (ℓ = 3), so proofs appear after the crossover and cannot move it. Two causes:
- The per-fact memory m_A is low, because 16 agent facts compete for reading.
- Z treats a proof's facts as held independently. The ChatGPT agent's frozen-board calculation shows independent
  products **under-estimate** proof co-retention by factors of 1.8–2.6 on stored data
  (`../../theory_blackboard/2026-10-27_02_exact_frozen_board_combinations.md`).

The missing mechanism is therefore the correlated co-retention of proof facts, since facts are acquired together in one
read. No further revision in this round (three used). Proposed next: replace the independence approximation in Z by a
co-retention moment.

## 2026-10-28 — Step 1: proofs with co-retention (written before running it)

Researcher: "go ahead" with step 1.

**Change (one):** replace the independence product m_A^ℓ in z by the exact co-retention moment of an ℓ-fact proof,
treating the proof's facts as interchangeable, each with board share p = b_A/F_A (equal copies):
- dh_k/dt = −kγh_k + Σ_{j=1..k} C(k,j) p_k(j) (h_{k−j} − h_k), with h₀ = 1, k = 1..ℓ;
- p_k(j) = Σ_{i=0..j} (−1)^i C(j,i) (1 − (k − j + i) p)^q, the chance that one read (q posts, with replacement) brings
  exactly a given j of the k facts;
- z = 1 − (1 − h_ℓ)^P.

This is the ChatGPT agent's triangular recurrence (`../../theory_blackboard/2026-10-27_02_exact_frozen_board_
combinations.md`) specialised to equal copies, applied on the model's changing board. Check: for k = 1 it is Eq. (1).

**Expectations stated in advance:**
- C5 (mathematical): h_ℓ ≥ m_A^ℓ at all times, because facts acquired in the same read are positively correlated.
- C6: the 3→6 crossover moves to *lower* ρ with proofs, more for ℓ = 3 (OP-like) than for ℓ = 6 (task003-like).
- My honest prior on C6 is weak. Co-retention gains a factor of about 2 in the simulator, while z at ρ 0.25 was about
  10⁻³ under independence; a factor of 2 may not be enough. If C6 fails, the next suspect is the per-fact memory level
  m_A itself (16 facts competing for reading). That will be checked against the simulator's per-fact holding shares in
  the symmetric setups (stored memories), not tuned.

## 2026-10-28 — Step 1 result; finite-size measurements

**Step 1** (`findings/step1_coretention.csv`):
- **C5 met:** co-retention raises proof holding strongly. h_ℓ/m_A^ℓ is 26 (ℓ = 6) and 2.7 (ℓ = 3) at ρ 0.25, and 6.2 and
  1.7 at ρ 0.5. The ℓ = 3 values are close to the ChatGPT agent's simulator values (1.8–2.6). The check h₁ = m_A holds
  exactly.
- **C6 not met:** the crossover still does not move (0.25–0.3).

**Pre-announced check against the simulator** (`findings/step1_simulator_check.csv`, symmetric setups, t = 30, q 6):
- The simulator's per-fact holding share of agent facts is 0.08–0.35, so the model's m_A ≈ 0.15 is realistic: the
  suspect is cleared.
- The model's proof shares with co-retention are now close to the simulator's (OP-like 0.03/0.10/0.32 against OP
  0.03/0.14/0.42 at ρ 0.25/0.5/0.75).
- **But in the simulator, too, almost nobody holds a complete proof near the crossover:** 1% at task003's crossover
  (ρ 0.5–0.625), 3% at OP's (0.25–0.375).
- So completed proofs cannot be what moves the simulator's symmetric crossover. The Stage C expectation C2/C3 rested on
  a wrong premise.

**New candidate** (from the model, not yet tested): the crossover is set by β_A, how strongly agents' own evidence argues
*against* the wrong answer; a larger β_A gives a lower crossover (stage A scan). In the no-proof setups the agents'
evidence is neutral between A0 and A2 (jointly 0.5/0/0.5), so β_A is small; in the symmetric setups it argues strongly
against A2. To be tested by measuring an effective β_A in each setup from stored memories (logit of exact P(A2 | M)
against the number of agent facts held), before comparing with the crossovers.

**Finite sizes** (`findings/finite_size_numbers.csv`, false arm, q 6, t = 30): at ρ 0.25 only about 3 of 15–18 agent
facts survive per episode (held or on the board), with 7–9 board copies and about 7 holders each. At ρ 0.75, 6–9 survive
with 4–5 copies. The controller keeps 3–6 distinct facts with 4–8 copies each. The ODE assumes all 16 agent facts
circulate with 2–4 copies: its effective fact number is wrong at short memory.

## 2026-10-28 — Step 2: does the agents' counter-evidence strength set the crossover? (written before measuring)

**Measurement** (stored simulator memories; read-only; no votes, G or D used).
- In each of the four setups (task003 / OP × symmetric / no-proof), take every agent memory at t = 30 in the **false**
  arm, q = 6, ρ ∈ {0.25, 0.5, 0.75} pooled.
- Compute the exact logit of P(A2 | M), with P clipped to [10⁻³, 1 − 10⁻³] (a proof gives P(A2) = 0).
- Fit by least squares: logit P(A2 | M) = h₀ + β_C·c(M) − β_A·a(M), where a = number of agent facts held and c = number
  of controller-only facts held.
- The fitted β_A is the "effective counter-evidence strength"; β_C and h₀ are measured the same way.
- Secondary measure, reported but not used for the decision: the mean drop in logit P(A2) when one more agent fact is
  added to a memory.

**Predictions fixed now:**
- B1: β_A is larger in the symmetric setup than in the no-proof setup of the same task.
- B2: the four β_A values are ordered as the reverse of the four simulator crossovers for 3→6: OP symmetric (0.25–0.375)
  > task003 symmetric (0.5–0.625) > OP no-proof (0.625–0.875) > task003 no-proof (0.75–1).
- B3: putting the measured (h₀, β_A, β_C) of each setup into the model (revision 3, ε = 1, everything else P0;
  nothing fitted to D) gives crossovers in the same order as the simulator's.

If B1 fails, the counter-evidence explanation is rejected. If B1 holds but B2/B3 fail, the explanation covers the
setup difference but not the task difference.

## 2026-10-28 — Step 2 result

`findings/step2_strengths.csv`, `findings/step2_model_crossovers.csv`. 72,000 memories per setup.

| setup | β_A | β_C | h₀ | mean logit drop per added agent fact | simulator crossover 3→6 | model crossover with measured values |
|---|---|---|---|---|---|---|
| OP symmetric | 1.141 | 0.863 | −0.720 | 0.181 | 0.25–0.375 | below 0.05 (negative everywhere) |
| task003 symmetric | 0.671 | 0.633 | −0.127 | 0.236 | 0.5–0.625 | 0.2–0.25 |
| OP no-proof | −0.074 | 0.140 | −0.707 | −0.037 | 0.625–0.875 | none (positive everywhere, D ≤ 0.003) |
| task003 no-proof | −0.172 | 0.383 | −0.461 | −0.048 | 0.75–1 | 0.85–0.9 |

- **B1 met.** β_A is much larger in the symmetric setups. In the no-proof setups agents' facts do not argue against A2:
  they push slightly towards it, because together they point to (0.5, 0, 0.5).
- **B2 met:** the order of β_A is exactly the reverse of the simulator's crossovers. The secondary measure (mean drop)
  orders the two symmetric setups the other way (0.18 against 0.24); the decision used the primary measure.
- **B3 partly met.**
  - Within each task, symmetric comes below no-proof: met (2 of 2).
  - Among the symmetric setups, OP comes below task003: met.
  - Among the no-proof setups: swapped. OP has no crossover in the model, and its curve is nearly flat.
  - The model's crossovers also sit at lower memory than the simulator's, but positions were not a target.

**Reading.** One measured number, the agents' counter-evidence strength, orders the four setups' reversals in the
simulator, and drives the right setup contrast in the model. The no-proof pair differs by little (β_A −0.07 against
−0.17), so other quantities (β_C, h₀) decide between them in the model.

## 2026-10-28 — Step 3: a finite fact-flow model (written before running it)

**Model F (finite, stochastic):** the same rules as the mean-field model, but with individual agents, individual facts
and a board of whole posts, so a fact can die out.
- N = 24 agents, W = 48 board slots, F_A = 16 agent facts, F_C = 4 controller facts, time step dt = 0.05.
- Each agent starts with one agent fact (24 agents over 16 facts: 8 facts held twice).
- **Per step:** each agent acts with probability dt; each held fact is forgotten with probability 1 − ρ^dt; the
  controller acts with probability λ₀ dt while t < T.
- **An action:** read q posts uniformly from the W newest posts not the agent's own (with replacement, as in the ODE).
  Vote A2 with probability Φ₂ = σ(h₀ + β_C c − β_A a), where a and c are the numbers of agent and controller facts held.
  Then post: an agent fact with weight κ_A a [ε + (1 − ε)·1(vote ≠ A2)], a controller fact with weight
  κ_C c [ε + (1 − ε)·1(vote = A2)], or a background post with weight κ₀; within a class the fact is uniform among those
  held.
- **The controller** posts a uniformly chosen controller fact.
- **Observables:** x_A2 among agents who have voted; G, D and δ(t) as differences between arms (400 episodes per cell;
  each controlled episode is paired with the silent one of the same seed). Continuations as in the ODE.
- **Parameters:** the current best mean-field version (P0, ε = 1). No Bayesian engine, no task world.

**Predictions** (from `../reports/2026-10-28_proofs_and_finite_size/`):
- F1 (diversity collapse): at T, the number of distinct agent facts still circulating (held or on the board) is well
  below 16 at ρ 0.25, and higher at ρ 0.75.
- F2: G is larger than in the mean-field model at the same parameters.
- F3: the reversal is still present (D(3→6) > 0 at short memory, < 0 at long memory).
- F4: after the controller's facts are removed, a residual remains (lasting damage) instead of 0.
- F5: under normal continuation, δ fades over tens of time units rather than settling within 5–9.
- F6 (scaling): with R copies of the swarm (N, W, λ₀ ×R) and the same facts, G approaches the mean-field value. With K
  copies of everything (N, W, λ₀, F_A, F_C ×K), it does not.

## 2026-10-28 — Step 3 result (finite model): F1–F6 met

`findings/step3_reversal.csv`, `step3_persistence.csv`, `step3_scaling.csv` (400 paired episodes per cell; P0, ε = 1;
no fitting).
- **F1 met:** agent facts still circulating at T (controlled arm) fall from 11.5 of 16 at ρ 0.9 to 1.6 at ρ 0.1 (2.5 at
  0.3, 6.8 at 0.7). The simulator has about 3 at ρ 0.25 and 4–8 at 0.75. All 4 controller facts survive (3.8–4.0).
- **F2 met:** G(q 6) is 0.31–0.41 at ρ ≤ 0.5, against 0.07–0.09 in the mean field. The simulator's range is 0.1–0.45.
- **F3 met:** D(3→6) = +0.078 ± 0.011 at ρ 0.1, −0.044 ± 0.007 at ρ 0.8; crossover between 0.5 and 0.6 (the simulator's
  task003 symmetric: 0.5–0.625).
- **F4 met:** removal residual +0.036 at ρ 0.25 (the simulator's task003 symmetric: +0.04) and +0.004 at ρ 0.75.
- **F5:** normal continuation 0.388 → 0.347 at ρ 0.25 (slow, about 10% over 100 units) and flat at 0.75. No fast
  plateau; it resembles task003's flat persistence. No communication decays, with a slight transient rise at ρ 0.75
  (0.147 → 0.150). The frozen board stays flat.
- **F6 met:** R copies: G = 0.307, 0.209, 0.130 for R = 1, 2, 4, approaching the mean field (0.072). The simulator's
  earlier diagnostic gave 0.322, 0.204, 0.144. K copies of everything: 0.434, 0.555 (no approach to the mean field).

**Mechanism check, stated before running:** if the larger finite effect comes from *asymmetric extinction*, then
spreading the controller's evidence over as many facts as the agents' (F_C = 16, same total evidence per agent through
β_C) should shrink G towards the mean field. Concentration keeps controller facts alive; spreading lets them die like
agent facts. Run: ρ 0.25 and 0.5, q 6, 400 paired episodes, F_C = 16 against 4.

**Mechanism check, result as run (execution error):** the run kept β_C = 0.5. I had stated "same total evidence per
agent through β_C" but did not implement it. With 16 controller facts, agents hold more of them (9–10 survive
against 4), so G rises in both models: finite 0.62/0.64, mean field 0.25/0.32 at ρ 0.25/0.5. **The prediction as
stated ("G shrinks towards the mean field") is not met.** Seen after the fact: the finite-to-mean-field ratio halves
(4.2–4.3 → 2.0–2.4). This is consistent with asymmetric extinction, but post-hoc.

**Corrected run (the control described above, not a new hypothesis):** β_C = 0.5 × 4/16 = 0.125 with F_C = 16, so the
controller's total possible evidence (F_C β_C = 2) is unchanged. The prediction is unchanged: G in the finite model moves
towards the mean-field value at the same parameters, i.e. the finite-to-mean-field ratio falls well below 4.

**Corrected mechanism check, result** (`findings/step3_mechanism_equal_evidence.csv`): with equal controller evidence
(F_C β_C = 2), spreading it over 16 facts lowers the finite G from 0.388 to 0.279 (ρ 0.25) and from 0.307 to 0.245
(ρ 0.5). The finite-to-mean-field ratio falls from 4.2–4.3 to 2.4. **Prediction met:** concentration of the controller's
evidence accounts for about half of the finite-size amplification; the collapse of the agents' own evidence accounts for
the rest. Report: `../reports/2026-10-28_finite_model/finite_model.pdf`.

**Not explained:**
- D is 2–3 times smaller than the simulator's (+0.08 against +0.22 at short memory);
- K-projects scaling (the model rises at K = 4; the simulator falls, because single facts there carry less information);
- the persistence task difference (ε not yet varied in the finite model).

## 2026-10-28 — Step 2 (revised plan): vote gating and persistence in the finite model (written before running)

Qualitative check only; no tuning towards the simulator. Finite model F at P0, normal continuation to t = 130,
ε ∈ {1, 0.75, 0.5}, ρ 0.25 and 0.75, q 6, 300 paired episodes.

**Expectation:** lower ε (reposting of controller facts more tied to believing A2) gives a **lower and still-declining**
δ(t) after the stop, i.e. fading over tens of time units like OP, while ε = 1 stays near flat (task003-like). Reported:
δ at 30, 50, 80, 130; retention; distinct controller facts circulating at 130.

**Step 2 result** (`findings/step2_gating_finite.csv`): **expectation met.**

| ε | ρ 0.25 retained | ρ 0.75 retained | controller facts left at 130 (ρ 0.25 / 0.75) |
|---|---|---|---|
| 1 | 0.91 | 0.99 | 2.5 / 3.4 |
| 0.75 | 0.82 | 0.65 | 2.1 / 2.4 |
| 0.5 | 0.55 | 0.30 | 1.1 / 0.9 |

- Lower ε gives a steady decline over the whole 100 time units, still falling at 130. That is OP-like; OP's retained
  fractions are 0.33–0.67. ε = 1 is near-flat, task003-like.
- The decline comes from controller lineages going extinct one at a time once their reposting depends on belief in A2.
  It is a finite-size selection effect, not the mean field's quick plateau.
