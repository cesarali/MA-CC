# Plan: two mirrored reports on task003-symmetric (Simulations 1 and 2), 15 October 2026

**Status: plan, for the researcher's approval. Nothing written yet.**

## Data

- **Simulation 1:** `results/simulation_1/2026-10-15_sim1_pm_symmetric_fine` (1,740 cells).
- **Simulation 2:** `results/simulation_2/2026-10-15_sim2_symmetric_fine` (1,740 cells).
- **Summary table:** `results/symmetric_fine/2026-10-15/cells_time.parquet`, built by `symmetric_fine/build_tables.py`. It matches the old runs on overlapping cells: Simulation 1, b 3, day 30, 0.317; Simulation 2, rate 1, t 30, 0.114.

## Output

Two LaTeX reports, next to the older ones and replacing none:
- `simulation_1/analysis/reports/2026-10-15_report1_symmetric/2026-10-15_report1_symmetric.tex`
- `simulation_2/analysis/reports/2026-10-15_report1_symmetric/2026-10-15_report1_symmetric.tex`

Equations and figures are numbered and referenced in the text. One figure script writes both reports' figures with the same layout, colours and axes. The two reports have identical sections and figure numbers, and differ only in the simulation. Figures are vector PDFs.

## Main settings, chosen from the numbers

| | Simulation 1 | Simulation 2 | why |
|---|---|---|---|
| main setting | q 6, q_c 12, b 1 | q 6, q_c 12, λc 1 | b 1 and λc 1 post about the same number of facts (13–29 against 19–28 per episode, by ρ); q 6, q_c 12 is the setting of all earlier reports |
| representative memory | ρ 0.5 | ρ 0.5 | the largest effects: Simulation 1 false-controller gain 0.53–0.57 at ρ 0.5, against 0.31–0.36 at 0.25 and 0.22–0.58 at 0.75; at ρ 1 every gain is 0 by t 30 |
| times | end of each day, 1–30; phase diagrams at day 30 | t 0.5, 1, …, 40; phase diagrams at t 30; Fig. 4 also at t 5 | at t 30 most Simulation 2 controllers have spent their 30 messages, so t 5 shows what speed buys |
| gate | on in the main text; off in Appendix A | same | as requested |
| colours | silent grey, target A0 blue, target A2 red; one colour scale for b or λc | same | as in the 14 Oct reports |

## Structure

**Title, author line, "How the simulator works".**
- The rules in one paragraph, as before.
- The facts at the start, as a table:
  - the agents' 18 distinct facts, held by 24 agents, which together prove A0;
  - the A0 pool and the A2 pool, 12 facts each, of which 10 are new to the agents;
  - the posterior P(A0, A1, A2) of each set: agents (1, 0, 0), A0 pool (0.963, 0.037, 0), A2 pool (0.114, 0, 0.886), agents + either pool (1, 0, 0).
- The grid of settings, as a table.

**Summary.** Written last, after all sections and conclusions.

**§1 Quantities.** Numbered equations:
1. Vote shares x_{a,t} = n_{a,t}/N_voted. In Simulation 2, N_voted counts only agents who have voted; this is a correction to the 9–14 Oct reports.
2. Paired susceptibility (notation of `BLACKBOARD_OBSERVABLES_AND_EFFICIENCIES`, §3): χ^{(a)}_{π,t} = G^{(a)}_t(π) = E[x^π_{a,t} − x^{base}_{a,t}], on the controller's own target (a = A0 for target A0, a = A2 for target A2).
3. Target Z ∈ {A0, A2}, a fair coin.
4. The states X:
   - one agent's vote (main text);
   - the plurality of the last votes, with ties split;
   - S1: m_A0 in 4 bins at 0.33 / 0.50 / 0.75;
   - S3: (m_A0, m_A2) in 3 × 3 bins.

   For each: how P(X | arm) is estimated, as the average over episodes of each episode's probability vector.
5. Mutual information I(Z; X).
6. C = ½KL(P₊‖P₀) + ½KL(P₋‖P₀) = I(Z; X) + KL(Q‖P₀), with Q = ½(P₊ + P₋); η = I/C. Undefined when C < 0.01 bits or C is infinite (a controller reaches a value silence never shows). Undefined values show as a dash or a gap, never as 0.
7. Gain per fact, G_t / F_t, with F_t the facts posted up to t.

**§2 Fig. 1: votes over time.** Two rows by four columns: share voting A0 (top) and A2 (bottom); columns ρ = 1, 0.75, 0.5, 0.25; three curves each (silent, target A0, target A2); main setting; 95% bands.

**§3 Fig. 2: paired susceptibility over time.** Two rows by four columns: target A0 (top) and A2 (bottom) × the four ρ; one curve per b or λc; q 6, q_c 12; bands.

**§4 Fig. 3: phase diagrams of the susceptibility**, target A2, t 30, three panels:
- (a) q_c × b or λc at ρ 0.5, q 6;
- (b) ρ × b or λc at q 6, q_c 12;
- (c) ρ × q at q_c 12 and the main b or λc.

**§5 Fig. 4: susceptibility against strength, and its slope.** Target A2, gate on and off on the same axes. Two rows by four columns:
- **Top row, the susceptibility:**
  - Simulation 1: day 30, one panel per ρ (1, 0.75, 0.5, 0.25);
  - Simulation 2: (t 5, ρ 0.5), (t 30, ρ 0.5), (t 5, ρ 0.75), (t 30, ρ 0.75).
- **Bottom row, the slope:** finite differences ΔG/Δb or ΔG/Δλc between neighbouring values, the marginal susceptibility. This is the derivative read as an efficiency.

**§6 Fig. 5: susceptibility per fact over time.** G_t / F_t. Two rows (targets) by four columns (ρ), one curve per b or λc. Facts posted are also listed in a short table. (A phase diagram was considered; time curves keep the mirror with Fig. 2.)

**§7 Fig. 6: who holds the proof.** Two rows by four columns:
- top: share proving A0 from any facts;
- bottom: share proving A0 from the agents' own original facts.

Columns are the four ρ; three curves each (silent, target A0, target A2); main setting.

**§8 Fig. 7: readability over time, one-agent state.** I(Z; X)(t), four panels (ρ), one curve per b or λc; q 6, q_c 12.

**§9 Fig. 8: efficiency over time, one-agent state.** η(t), same layout.

**§10 Fig. 9: phase diagrams of readability**, one-agent state, t 30, the three panels of Fig. 3.

**§11 Fig. 10: phase diagrams of efficiency**, same.

Each section ends with 3–5 one-line findings, each backed by a number from the figure.

**§12 Conclusions.**

**Appendix A, gate off.** Figs. A1–A3: Figs. 1, 2 and 3 with the gate off. Figs. A4–A5: readability and efficiency over time and in phase, gate off. Short findings, mostly where gate off differs.

**Appendix B, other states X.** For each of the plurality, S1 and S3:
- one figure with I(Z; X)(t) (top) and η(t) (bottom), 2 × 4 by ρ;
- one figure with the three phase panels of I and of η, 2 × 3.

Short findings, including a comparison with the one-agent state.

**Appendix C, strong control.** Figs. 1 and 6 again at b 9 (Simulation 1) or λc 8 (Simulation 2), the
strongest setting, so weak and strong control can be compared directly. (Approved 15 Oct; added at the
researcher's request that results be shown across strengths as well as across ρ.)

## Checks before writing the text

- Gains reproduce the 14 Oct values where settings overlap.
- I ≤ 1 bit; C = I + KL(Q‖P₀) holds numerically; η is in [0, 1].
- Figure numbers match the text; mirrored figures use identical axes and colour scales across the two reports where the ranges allow.
- Every number in the text is read from the table by a script, not copied by hand.
