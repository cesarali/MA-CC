# TODO: the single list for MA-CC analysis

Rewritten 10 November 2026 by Claude, after a keep / park / dismiss review with the ChatGPT agent
(`theory_blackboard/2026-11-10_claude.md` §3 and `2026-11-10_chatgpt_01_reply_and_priorities.md`). Item codes (A1, B3,
…) are the ones used in that review.

This file replaces the open items of the older lists:
- `15_agent_simulation/TODO.md`;
- `task003_llm_free/TODO.md`;
- `object_placements_llm_free/TODO.md`.

Those files now point here and keep their history. `theory_chatgpt/TODO.md` stays the ChatGPT agent's own list.

Rules:
- **Keep** means wanted, but not started until the researcher says so.
- **Park** means kept for later, with no current plan.
- **Dismissed** items are kept below with their reason; nothing is erased.
- Move a finished item to "Done" with its date and a pointer.

## Keep

1. **F1. Proposals for the LLM experiment** (collaborator's Design R in swarm-control). The LLM run misses the
   short-memory half of the reading reversal. We propose prompt changes to the LLM experiment; we run no LLM
   experiments ourselves. The LLM runs use only LLM reasoning and are not meant to reproduce the Bayesian runs; the
   Bayesian results help interpret them (no Bayesian surrogates or controls). Each change, e.g. a scoring hint, is
   tested with and without it. Draft prompts: `swarm-control`, branch `dev/rsanchez`,
   `experimental_harness/src/board_harness/live_agent/r_prompting/*-sim2.txt`.
   - Absorbs **E1**, the 16-condition LLM transfer test: the no-solution half is still not covered by any LLM run.
   - Absorbs **A8** (choose cells for the LLM runs) and **F2** (our `llm-sim2` prompt drafts).
2. **A2. Controller timing at equal budget.** Keep budget = 30 and scan the controller rate λ_c, so the same 30 posts
   are delivered over 30/λ_c time units. This is the researcher's request, and also ChatGPT's TODO. Report outcomes at
   a common horizon separately from recovery at equal time since the last controller post. Proposed settings:
   - λ_c ∈ {0.125, 0.25, 0.5, 1, 2, 4, 8};
   - all four 15-agent setups, standard reading, q_c = 8, run to t = 130, 1,000 paths per cell;
   - reduced grid q ∈ {2, 4, 8, 16} × 8 ρ values, about 3 h on 12 cores.

   Pre-register the predictions first.
3. **B1. Do beliefs, not only votes, split into two groups?** Use the stored snapshots: the per-episode mean
   posterior P(A2) at t = 130, and dwell times on the posterior with hysteresis. Condition on *which* controller facts
   survive, not only how many. Pre-register first. Not yet claimed by either agent.
4. **B4. The fact-flow model does not reproduce the Object Placements no-proof reversal.** This limits any claim of a
   common mechanism. Includes **C15** (why forgetting weakens control): the theory gives a qualitative account, not a
   full causal one for every setup.
5. **B3. Fact efficiency in the mean-field theory.** OPEN, incomplete, paused (researcher, 10 Nov). Definitions and
   examples exist: η = g/F, the exact slope factorisation, information measures. The general trade-offs and the
   robustness are unresolved. See:
   - `theory_chatgpt/reports/2026-11-10_mean_field_efficiency/`;
   - `theory_blackboard/2026-11-08_chatgpt_07_mean_field_fact_efficiency.md`.

## Park

- **A1 + C14. Critical mass (dose).** No budget limit: the controller posts until t = 30, so total posts grow with λ_c.
  Measure the smallest λ_c that tips each (q, ρ), with the threshold fixed in advance.
- **A3. Stepping stones** (no-solution setups): target A1 first, then A2, at the same total budget.
- **A4. Asymmetry:** compare truth→false with false→truth attacks, at matched evidence and starting states.
- **A5. A finite-size theory of the crossover constant** q τ\* ≈ 8–14 (task003); the mean field gives 2.5–4. Do not
  build theory only to fit this number.
- **A6. The selection mechanism for OP's weaker persistence,** using the recorded posts (coherence of the controller
  facts in circulation). Related to B1, but a separate analysis.
- **A7. Source refresh as a Simulation 2 option** (the next candidate defence). The theory part is done.
- **B2. Finite-model efficiency diagnostics:**
  - separate the nonlinear readout, native crowding and fact extinction;
  - start with per-agent controller and native counts, before any reintroduction of facts.

  Kept separate from B3.
- **C11. Sensing bias:** the controller's vote estimate overestimates the target's share by +0.03 to +0.05.
- **C12. A controller with memory.** Also relevant to the LLM controller, which has none.
- **C13. Board fading:** window W and time-based fading. First, match W = 48 for the LLM comparison (F1).
- **C16 + C17 + C19. Information measures, as one item:**
  - command channel I(Z; V_H);
  - activation, assigned-policy and target information;
  - path KL, η_ctl, η_task.

  Link them to ChatGPT's on/off channel information, and keep the individual measures distinct.
- **D2. A stronger symmetric OP pool,** built from individually A2-leaning facts (at most about 6 qualify).
- **D3. Another Object Placements story** whose facts lean to A2 one at a time.
- **E2. Scale facts with agents through K independent projects with one shared split.** Needs a new exact belief
  engine.

## Dismissed (10 November 2026)

| item | reason |
|---|---|
| C1. Balance task003 no-solution on starting votes (8/8/8) | replaced by the 15-agent setups |
| C2. Rank the no-solution builder by the gap between population ceilings | same |
| C3. task003 symmetric without tied facts | same |
| C4. Per-agent ceilings for task003 symmetric | same; the analysis stays documented |
| C5. Large-b backfire fixes (Simulation 1) | Simulation 1 is retired |
| C6. Stopping rule with the vote gate off (Simulation 1) | same |
| C7. Uncleared board (Simulation 1) | same |
| C8. Overlap as a factor (Simulation 1) | same |
| C9. Simulation 1 review guide to a reviewing agent | same |
| C10. Simulation 2 review guide to a reviewing agent | replaced by the Claude–ChatGPT cross-reviews |
| C18. The four Markov tests on Simulation 1 data | Simulation 1 is retired |
| D1. Balanced OP no-proof pool | done in effect: the 15-agent OP no-solution pool gives exactly 1/3 each |

## Done

- 2026-11-08: fact-efficiency scans and cross-review (`theory_claude/fact_efficiency/`). Metastability corrections:
  `15_agent_simulation/reports/2026-11-03_metastability/ADDENDUM_2026-11-08.md`.
- 2026-11-03: metastability and tipping analysis, `15_agent_simulation/reports/2026-11-03_metastability/`.
- 2026-10-30: 15-agent campaign runs (1,024 cells) and report, `15_agent_simulation/reports/2026-10-30_campaign_results/`.
- 2026-10-20: key-phenomena study, Campaigns A and B (`task003_llm_free/simulation_2/key_phenomena_hypothesis/`).
- 2026-10-09: Simulation 2 spec, simulator, main grid and reports (`task003_llm_free/simulation_2/`).
- 2026-10-07: Simulation 1 grid and reports (`task003_llm_free/simulation_1/analysis/`).
