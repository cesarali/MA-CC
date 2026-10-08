# TODO — task003_llm_free

Deferred work, newest decisions first. Move an item to "Done" with a date and a
pointer when it is finished.

## Setups

- [ ] **Balance task003-nosolution on expected starting votes (8 / 8 / 8).** Under the
  current fact-group rule it is impossible (0 of 10,438 coin-flip agent sets); balancing
  on expected argmax votes works (a random search found sets). Needs a new builder rule,
  a re-check of the pool rules (no shared facts, no joint proof), and new simulations.
  Deferred 2026-10-08: not expected to change conclusions; probability matching already
  starts balanced. See `simulation_1/analysis/reports/report2_nosolution/` §3.
- [ ] Rank the no-solution builder also by the gap between the two population ceilings
  (P(A0) 0.92 vs P(A2) 0.77).
- [ ] task003-symmetric without tied facts (strict lean rule) as a comparison for the
  duplication asymmetry at ρ = 0.75.
- [ ] Per-agent ceilings for task003-symmetric (how far the A2 pool can pull individuals).

## Simulation 1 follow-ups

- [ ] **Large-b backfire: design fixes to discuss.** Let the controller post *up to* b
  facts, or never post facts against its target (Q2 §3, Q3).
- [ ] **Stopping rule with the vote gate off** (`controller_gate: always`), so
  `silent_when_target_proved` is tested on its own (Q5).
- [ ] **Uncleared board** (`is_board_cleared: false`) as a separate study.
- [ ] Overlap as a factor (pool facts the agents already hold: 0 / 4 / 8).
- [ ] Send `simulation_1/analysis/REVIEW_GUIDE.md` to a reviewing agent and act on its report.

## Observables and theory

- [ ] Command channel I(Z; V_H), following F_H, information per post.
- [ ] Information measures (activation, assigned-policy, target information).
- [ ] Rerun the four Markov tests on the Simulation 1 data; re-check bin cuts per setup;
  consider a second coordinate (mean P(A2)).
- [ ] Path KL, C_π, η_ctl, η_task on a validated coarse-graining.

## Done

- 2026-10-07: Simulation 1 grid, analysis Q0–Q6, two short reports (`simulation_1/analysis/`).
