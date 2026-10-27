# A minimal fact-flow model of the reading reversal and persistence (27 October 2026)

**Author:** Claude. **Plan followed:** the researcher's `phenomenological_fact_flow_theory_plan.md` (27 Oct).
**Equations and derivations:** [`model.md`](model.md). **Every decision, including failures:**
[`decision_log.md`](decision_log.md). **Code:** `fact_flow.py` (model), `stage*.py` and `rev*_*.py` (runs); figures in
`results/`.

## 1. The answer in brief

A two-class model, about ten lines of ODE, shows both phenomena in a region of parameters, with no Bayesian engine and
no task world. Its classes are the agents' evidence and the controller's evidence, each with a memory density and a
board density. Agents vote through a sigmoid and repost one fact.

- **The reading reversal comes from dose against sensitivity.** Reading spreads the controller's facts (dose ↑). But
  it also lets agents accumulate their own evidence, which moves the population away from the tipping point, where
  extra evidence of either kind moves votes less (sensitivity ↓). With short memory the dose wins; with long memory
  the sensitivity loss wins. It needs no proof and no selective reposting.
- **Persistence comes from controller facts that keep reproducing after the stop.** That happens when their growth
  number R_C = qκ_C g_C/(γH) exceeds 1. Whether it does depends on the reposting rule. If only agents who already
  believe the wrong answer repost controller facts (the plan's closure), R_C < 1 and the effect dies, unless the
  controller is so strong that the system switches into a controller-dominated state. If controller facts are reposted
  partly regardless of the vote (eligibility ε), R_C > 1 and the effect settles at a moderate plateau. Lower ε gives a
  lower plateau.

## 2. Status of each result

| result | kind | evidence |
|---|---|---|
| frozen board: m_α* = r_α/(r_α + γ), δ constant | mathematical | §2.1 of model.md |
| no communication: δ → 0 on the memory time scale, with a transient rise at long memory | mathematical (sign of the initial slope) + numerical | §2.2; Fig. 2 |
| reversal: D ≈ β_Cσ′[ΔM_C − (1 − 2Φ_s)β_A M_C ΔM_A] | approximation (small effect), checked against the full model | §2.3; Fig. 5 |
| reversal exists without proof and without selection | numerical demonstration | stage A scan, rev. 3 grid |
| persistence ⇔ R_C > 1 (linear stability) | mathematical (linearisation) + numerical | §2.4; Fig. 4 |
| high ε ↔ task003, lower ε ↔ Object Placements | suggestive mapping to the simulator | §4 |

## 3. Path taken: what failed and what worked (decision log)

1. **Stage A, P0 (parameters fixed before any run).** The reversal is present (D(3→6) = +0.003 at ρ 0.05, −0.005 at
   ρ 0.5). The crossover falls with q, as in the simulator, and the result holds in 16 of 22 single-parameter changes
   (×0.5, ×2). Effects are small (G ≈ 0.016).
2. **My stated mechanism failed.** I had predicted that the controller's few facts saturate in memory first. The
   reversal survives equal fact counts, and survives neutral reposting. The chain-rule split found the actual mechanism
   (dose against sensitivity, §1).
3. **Stage D, P0: persistence failed.** With the plan's vote-gated reposting, the effect is gone about 10 time units
   after the stop. The plan's own threshold explains this: R_C ≈ 0.2. Neutral reposting keeps 65–75%.
4. **Revision 1: stronger controller facts (β_C/β_A up to 4).** The reversal is kept; persistence is still 0.
5. **Revision 2: reposting advantage κ_C/κ_A × β_C/β_A.** At κ_C/κ_A ≥ 4 the controlled arm switches into a
   controller-dominated state (bistability). Both phenomena hold only at one corner. Not an open region.
6. **Revision 3: softened vote gating ε.** The reversal holds at every ε. Retention rises with ε: 0, 0, 0–0.29,
   0.32–0.63, 0.65–0.75 for ε = 0, 0.25, 0.5, 0.75, 1. It is ≈ 0 when R_C < 1. **Both phenomena hold for ε ≳ 0.75**, at
   β_C/β_A = 1 and 2.

Figures (in `results/`):
- `fig1_reversal.pdf`: D for three reading steps, two eligibilities;
- `fig2_continuations.pdf`: the four continuations;
- `fig3_trajectories.pdf`: m and b over time;
- `fig4_threshold.pdf`: retained fraction against R_C;
- `fig5_dose_sensitivity.pdf`: the reversal's two terms.

## 4. Qualitative comparison with the simulator (no fitting)

| feature | simulator (task003 / Object Placements) | model | agreement |
|---|---|---|---|
| D > 0 at short memory, < 0 at long memory | yes, both tasks and setups | yes | ✓ |
| crossover moves to lower memory as q grows | yes | yes (3→6: 0.2–0.3; 6→9: 0.1–0.2; 9→12: 0.05–0.1) | ✓ |
| reversal without any possible proof | yes (no-proof setups) | yes (z = 0) | ✓ |
| frozen board keeps the effect | yes | yes, exactly | ✓ |
| no communication: δ → 0 within a few memory lifetimes | yes | yes | ✓ |
| transient rise when communication stops, long memory | yes (symmetric ρ 0.75) | yes (ρ 0.75) | ✓ (different explanation: fading evidence, not broken proofs) |
| controller facts removed → most of the effect gone | yes, small residual | yes, residual 0 | partial: mean field has no lasting damage |
| normal continuation: task003 flat, OP fades | flat (~1.0) / slow decline to 0.33–0.67 over 100 units | plateau 0.65–0.75 (ε = 1) / 0.32–0.63 (ε = 0.75), reached in 5–9 units | partial: plateau levels, wrong time course |
| effect size G | 0.1–0.45 | 0.02–0.09 | ✗ (not a target) |
| proof-bearing vs no-proof setups differ (crossover higher without proof) | yes | not modelled (z not introduced) | open |

## 5. What is not explained, and why

- **The slow time course of persistence.** The model relaxes to its plateau in 5–9 time units. The simulator stays
  flat (task003) or declines over about 50 units (OP). Phase C2 found OP's slow fading is caused by selection pruning
  the controller's facts over time, which a two-class mean field cannot represent (it has no within-class variety). A
  finite-copy or multi-type version would be needed.
- **The lasting residual after removing the controller's facts.** That residual is the extinction of agents' own facts
  (Phase C1). In the mean field nothing goes extinct, so the residual is 0.
- **The proof / no-proof difference.** Stage C of the plan (one complementarity variable z) was not needed for the
  reversal and was not attempted. It is the natural next test: does a proof variable move the crossover to lower
  memory, as in the simulator?
- **Magnitudes.** The model's G is 3–10 times smaller; not a target.
- **The mapping ε ↔ task** (high in task003, lower in OP) is suggestive. ε is a model parameter, and its relation to
  the simulator's measured eligibility (0.8 against 0.55) is not derived.

## 6. Next steps (proposed)

1. **Stage C:** add the plan's single complementarity variable z = Z(m_A) and check whether it lowers the crossover
   (proof-bearing against no-proof), without being needed for the reversal.
2. **Analytic crossover.** Solve the stationary memory densities in the short-memory and long-memory limits and derive
   the ρ at which the dose and sensitivity terms balance, as a function of q. This would explain why the crossover
   falls with q.
3. **The minimal finite-copy refinement** (plan §6): integer holders and board copies for a few representative facts.
   It should give the slow decline of persistence and the residual after removal.
4. **Qualitative checks against stored simulator data** of the two new mechanisms:
   - sensitivity: does a single controller fact move fewer votes in agents who hold more of their own evidence?
   - eligibility: does reposting of controller facts by non-A2 voters differ between task003 and OP?
