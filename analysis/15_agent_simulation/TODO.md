# TODO — 15_agent_simulation

Deferred work for the 15-agent campaign, newest first. Move an item to "Done" with a date and a pointer when it is
finished. New runs get new configs and dated result folders; the runs of 29–30 Oct 2026 are never overwritten.

## Controller-rate scan (deferred 2026-11-03)

Motivation: the paper's critical-mass maps (Flint et al., arXiv:2609.25194). See the metastability report,
`reports/2026-11-03_metastability/`.

The researcher prefers scanning the controller's rate λ_c. Decide first between the two versions below, which answer
different questions.

- [ ] **A. Critical mass (dose).** Remove the budget limit, and let the controller post until t = 30. Total posts
  then grow with λ_c (about 30 λ_c, minus vote-gate skips). Measure the smallest λ_c that tips each (q, ρ): for
  example, tipping probability T_A2 ≥ 0.5, or G above a threshold, fixed in advance.
- [ ] **B. Timing at equal budget.** Keep budget = 30 and scan λ_c: the same 30 posts delivered over 30/λ_c time units.
  This is also ChatGPT's TODO item "does controller timing matter at equal total posting budget?"
  (`../theory_chatgpt/TODO.md`). Report common-horizon outcomes separately from recovery at equal time since the
  last controller post.

Proposed settings (not yet decided):
- λ_c ∈ {0.125, 0.25, 0.5, 1, 2, 4, 8}, log-spaced; the controller's share of posts goes from about 1% to 35%;
- all four setups; standard reading; q_c = 8; to t = 130; 1,000 paths per cell;
- grid:
  - reduced: q ∈ {2, 4, 8, 16} × the 8 ρ values. 896 controlled + 128 silent = 1,024 cells, about 3 h on 12 cores.
    Suggested.
  - full: 8 q × 8 ρ. 2,048 cells, about 6 h.
- A and B together on the reduced grid: about 6 h, one overnight run.
- Pre-register the threshold definition and predictions before running.

## Other open items from the campaign reports

- [ ] **Stepping stones** (no-solution setups): a controller that targets A1 first and A2 later, at the same total
  budget. Tests whether an intermediate answer lowers the cost of reaching A2.
- [ ] **Asymmetry:** compare truth→false with false→truth attacks.
- [ ] **A finite-size theory of the crossover constant** q τ* ≈ 8–14 (task003). The mean field gives 2.5–4.
- [ ] **Test the selection mechanism** for Object Placements' weaker persistence, using the recorded posts (coherence
  of circulating controller facts).
- [ ] **Source refresh** (ChatGPT's proposal) as a Simulation 2 option, as the next candidate defence.
- [ ] **Choose cells for the 100-path LLM runs** along the crossover diagonal; only if the LLM setup is resumed.

## Done

- 2026-10-30: campaign runs (task003, OP; symmetric and uniform no-solution; 1,024 cells) and report
  `reports/2026-10-30_campaign_results/`.
- 2026-11-03: metastability and tipping analysis, `reports/2026-11-03_metastability/`.
