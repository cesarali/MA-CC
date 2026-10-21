# Agent task: Map the direct/indirect mechanisms of the reading reversal

> Copy of the task given by the researcher on 21 October 2026 (`2026-10-21_agent_direct_indirect_heatmaps.md`),
> kept verbatim as the source of the report in this folder. Agreed changes: no new runs; the belief at the last
> vote is reported as a limitation (no per-action records), not computed.

**Priority:** Reanalyse existing simulations. **Do not launch new simulations** unless the requested quantities cannot be recovered from existing saved states/logs; report any such limitation first.

## 1. Scientific motivation

We study 24 Bayesian agents who communicate through a shared blackboard. Agents read `q` posts per activation, forget facts at a rate controlled by `rho`, vote by probability matching, and repost evidence. A controller selects true facts supporting the incorrect answer `A2`. Increasing agent reading can either **amplify manipulation** or **protect against manipulation**. We want to identify *why the sign changes* as a function of memory `rho` and communication intensity `q`.

A companion persistence experiment showed that controller-exclusive facts continue to circulate long after the controller stops, and that deleting those facts largely removes susceptibility, leaving a small residual consistent with the *indirect* belief shift. This motivates investigating the direct/indirect decomposition as a common description of the reading reversal and persistence. **This task is analysis/visualization only, not a new persistence experiment.**

### Existing sources to inspect

- `2026-10-20_step2_reading_mechanism.pdf`, especially Figures 1–4, and the underlying run `results/simulation 2/2026-10-20 sim2 reading mechanism`.
- `2026-10-20_step3_persistence_carrier.pdf`, especially Figures 1–2; use only for interpretation/cross-checks.
- Relevant older baseline reports (`2026-10-19_report1_symmetric.pdf` and `2026-10-17_report2_nosolution.pdf`) for reproducibility checks.

Study **both** setups: `task003-symmetric` (proof available collectively) and `task003-nosolution` (no conclusive proof).

## 2. Data and comparison to retain

Use the *already completed* reading-mechanism run:

- Agent memory survival `rho ∈ {0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1}`.
- Reads per agent activation `q ∈ {3, 6, 9, 12}`.
- Controller aims at `A2`, with `q_c = 12`, `lambda_c = 1`, gate on, `B = 30`, board window `W = 48`.
- 1,000 episodes per setting, with controlled and silent arms; evaluation at `t = 30`.
- The report says that runs at `q = 3, 6, 12` reproduce earlier results. Recheck these as part of validation.

**First audit stored data** to determine where per-agent facts, posterior beliefs, last votes, episode identifiers, and number of agent reads are stored. Confirm whether episode random streams genuinely match **across different q settings**, rather than assuming that paired controlled/silent episodes also imply q-to-q pairing. If unavailable, use statistically correct independent-across-q uncertainty estimation.

## 3. Definitions — use exactly these conventions

For episode `e`, agent `i`, setup `s`, reading level `q`, and memory `rho`:

- `F_i^C` = original **agent-origin** facts currently remembered in the controller arm.
- `H_i^C` = **controller-exclusive** facts currently remembered in the controller arm.
- `F_i^S` = original agent-origin facts currently remembered in the silent arm.
- `p_2(F) = P(A2 | F)` = exact Bayesian posterior of `A2` given fact set `F`.
- `N = 24` agents.

The symmetric controller pool overlaps with agent-origin facts: classify any overlapping facts as **agent-origin**, not controller-exclusive. Use content IDs/provenance consistently with the simulator.

For each episode at `t = 30`, compute the agent-averaged belief components

\[
 g_{\mathrm{direct}}^{(e)}(\rho,q)
 =\frac1N\sum_{i=1}^N\left[p_2(F_i^C\cup H_i^C)-p_2(F_i^C)\right],
\]

\[
 g_{\mathrm{indirect}}^{(e)}(\rho,q)
 =\frac1N\sum_{i=1}^N\left[p_2(F_i^C)-p_2(F_i^S)\right],
\]

\[
 g_{\mathrm{total,belief}}^{(e)}
 =g_{\mathrm{direct}}^{(e)}+g_{\mathrm{indirect}}^{(e)}
 =\frac1N\sum_i\left[p_2(F_i^C\cup H_i^C)-p_2(F_i^S)\right].
\]

Define `G_direct`, `G_indirect`, and `G_total,belief` by averaging the corresponding `g` over episodes. The sum identity is **exact for belief quantities** by construction.

Independently compute the original vote-based susceptibility in each episode:

\[
 g_{\mathrm{total,vote}}^{(e)}(\rho,q)
 =x_{A2}^{C,(e)}(t=30)-x_{A2}^{S,(e)}(t=30),
\]

where `x_A2` is the fraction of agents whose *last recorded votes* are `A2`. Average over episodes to obtain `G_total,vote`.

For each adjacent reading step `q_low -> q_high` in `{3 -> 6, 6 -> 9, 9 -> 12}`, compute

\[
\begin{aligned}
D_{\mathrm{direct}}(\rho;q_{\mathrm{low}}\to q_{\mathrm{high}})
 &=G_{\mathrm{direct}}(\rho,q_{\mathrm{high}})-G_{\mathrm{direct}}(\rho,q_{\mathrm{low}}),\\
D_{\mathrm{indirect}}(\rho;q_{\mathrm{low}}\to q_{\mathrm{high}})
 &=G_{\mathrm{indirect}}(\rho,q_{\mathrm{high}})-G_{\mathrm{indirect}}(\rho,q_{\mathrm{low}}),\\
D_{\mathrm{total,belief}}&=D_{\mathrm{direct}}+D_{\mathrm{indirect}},\\
D_{\mathrm{total,vote}}&=G_{\mathrm{total,vote}}(\rho,q_{\mathrm{high}})-G_{\mathrm{total,vote}}(\rho,q_{\mathrm{low}}).
\end{aligned}
\]

Positive `D` means additional reading **amplifies** the controller-induced `A2` shift; negative `D` means additional reading **protects** against that shift. Never label a cell as significant solely from its mean sign.

## 4. Required plots

**Main deliverable: six heatmaps in one coherent 2 × 3 layout.**

- Rows: `symmetric`, `nosolution`.
- Columns, in order: `D_direct`, `D_indirect`, `D_total,belief`.
- Horizontal axis: incremental reading step `3->6`, `6->9`, `9->12` (these are **steps**, not point values of `q`).
- Vertical axis: all 8 values of `rho`, explicitly labeled.
- Use the **same diverging color scale centered on zero across all six panels**; annotate cell means (at sensible precision) and visually mark cells whose 95% CI includes zero (e.g., gray or hollow annotation). Do not smooth the three-column grid into a continuous-looking phase diagram.

**Vote comparison:** Provide an additional compact table or a two-panel plot (one per setup) showing

\[
E(\rho;q_1\to q_2) = D_{\mathrm{total,vote}}-D_{\mathrm{total,belief}},
\]

with paired/bootstrap confidence intervals. State whether their signs agree in each cell, and explicitly inspect discrepancies near zero. Compare your `D_total,vote` with the existing Figure 2 in the reading-mechanism PDF.

**Optional only if useful:** Overlay cell-level signs of `D_total,vote` on the `D_total,belief` heatmaps instead of creating redundant additional figures.

## 5. Uncertainty and validation

- **Resampling unit must be the episode**, not an individual agent or 0.25-time-unit snapshot: agents and observations within one episode are dependent.
- Within each `rho,q` setting, keep the **controlled and silent arms paired** in resampling.
- For reading differences, pair `q_high` with `q_low` at the episode level **only if** matching episode IDs/common random streams across `q` have been verified. Otherwise bootstrap the two `q` conditions independently while preserving their internal controlled/silent pairing.
- Compute two-sided **95% bootstrap confidence intervals** (e.g., ≥ 2,000 cluster resamples) for each `D_direct`, `D_indirect`, `D_total,belief`, `D_total,vote`, and `E`. Describe bootstrap design and number of resamples.
- Compute `D_total,belief` and its uncertainty by adding *episode-level* direct and indirect estimates in each bootstrap replicate, not by adding independent confidence limits.
- Check the decomposition numerically at episode level to machine precision: `g_total,belief = g_direct + g_indirect`.
- Explain that belief evaluated from the **memory snapshot at `t=30`** need not agree exactly with expected *last recorded votes*: an agent's remembered facts can decay between activations while its most recent vote is unchanged. If possible, separately evaluate posterior belief **at the agent's last voting activation**, aligned with the last vote. This should improve comparability. Do not silently replace one definition by another.
- Report baseline replication checks and any absent/mismatched event logs explicitly.

## 6. Scientific questions the plots must answer

**Q1 — When does the direct term switch sign?** Does the sign-change pattern of `D_direct` coincide with the protective/amplifying boundary measured by `D_total,vote`? Report matches and exceptions, with uncertainty.

**Q2 — When does the indirect term switch sign?** Is `D_indirect` systematically negative at high memory? Are there positive regions? Distinguish differences in the **Bayesian meaning** of retained original facts from just the number of facts remembered.

**Q3 — Which mechanism controls the crossover?** For every well-resolved sign reversal of `D_total`, classify whether:

- **Direct reversal:** `D_direct` itself switches sign;
- **Indirect override:** `D_direct` keeps a positive sign but a sufficiently negative `D_indirect` outweighs it;
- **Mixed/uncertain:** both contributions change or intervals are too broad.

The belief-level crossover obeys `D_direct = -D_indirect`; this is an **identity at zero total response**, not proof that direct and indirect are independent causal pathways. Quantify the characteristic differences between proof-bearing and no-proof setups.

## 7. Outputs, reporting, and stop rule

Return:

1. A reproducible analysis script/notebook, reusing project paths, without changing simulation rules.
2. The six-panel annotated heatmap figure, plus compact belief-versus-vote comparison.
3. Machine-readable tables with episode means, `D` values, 95% intervals, signs, and the bootstrap/pairing method.
4. A short report (roughly 1–2 pages) that answers Q1–Q3, lists ambiguous cells, and states what can/cannot be explained from saved data.
5. A brief audit: whether further runs across `q_c`, `lambda_c`, `B`, `W`, or additional `rho,q` combinations are necessary to answer **this specific mechanistic question**. Prior results suggest broad `q_c` sweeps are low priority; verify against available data before ruling them out.

**Stop once the existing data have been fully analyzed and uncertainties quantified.** Do not conduct parameter sweeps or propose a large simulation campaign by default. Recommend only the smallest new run that addresses a concrete missing measurement or statistically unresolved crossover.

### Connection to the persistence study (interpretation, not an additional task)

Immediately after removing controller-exclusive facts from agent memories, the belief-level direct term is zero and the residual controller-versus-silent belief difference equals the **indirect** term, with the surviving original fact sets held fixed. The observed later vote residual need only approximately match because agents activate asynchronously and retained original evidence continues evolving. The persistence branches show controller-exclusive facts are necessary for most of the sustained response; do not conflate this causal removal with the algebraic identity at a fixed memory snapshot.
