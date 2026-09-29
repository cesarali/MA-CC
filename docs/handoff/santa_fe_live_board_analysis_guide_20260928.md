# Santa Fe live-board study: analysis guide

**Run:** `santa_fe_live_board_v4_full` (28 September 2026)  
**Review archive:** `santa_fe_live_board_full_review_20260928.zip`  
**Full retained result root:** `/shared/home/cesar/work/results/santa_fe_live_board_v4_full`

This guide explains what was run, what each exported number means, and how to read the plots. It describes the **microscopic Santa Fe live-board simulator only**. No stochastic mean-field theory was run for this version, so none of these maps is a theory–simulation difference map. The 60-round maps describe a **finite horizon**; they do not establish phase transitions or equilibrium.

## 1. What happened in one episode

Each physical cell fixes persistence `rho`, active-round posting budget `B`, agent reading capacity `q`, controller sensing capacity `q_c`, one of three `(beta_E,beta_S)` regimes, and a truth (`+1`) or false (`-1`) controller target. The grid has `6 × 9 × 3 × 3 × 3 × 2 = 2,916` cells. Each cell ran **128 independent episodes** of 60 rounds, for **373,248 independent episodes**. A round has 24 focal-agent slots, sampled with replacement. The population has 24 agents and 12 distinct facts: eight favor truth and four favor falsehood. Each fact initially has two holders; each agent starts with one fact. The board starts empty.

At the start of round `t`, record the population and evidence state. Write `K_t` for the number of agents supporting the controller target and `X_t=K_t/24` for its share. The controller samples `q_c` **agents' current votes without replacement**, observes `Y_t` target votes, and draws a gate `U_t` with known propensity

\[
e_t=P(U_t=1\mid Y_t)=\operatorname{sigmoid}\!\left[4\left(\tfrac12-Y_t/q_c\right)\right].
\]

Previous-round posts expire; each active fact persists independently with probability `rho`. If `U_t=1`, the controller appends exactly `B` target-aligned posts; if `U_t=0`, it appends none. Then 24 focal slots read the **live** board, update evidence and votes, and immediately post peer messages visible to later slots. Each read samples up to `q` eligible current messages without replacement and excludes the focal agent's own messages. The resulting population is `K_(t+1)`. At `B=0`, `U_t` remains a logged **virtual** randomized gate but physically posts nothing. The simulator's overload factor attenuates the entire evidence-plus-social voting logit above seven distinct active facts.

The main physical summary first averages `X_41,...,X_60` **within each episode**, then averages those 128 episode summaries. Information estimates pool the 60 pre-action-to-next-state transitions **within a cell**; four primary statistics also have separate per-round estimates. Consecutive rounds and micro-slots are not counted as independent population episodes.

The resolved grid and clock are in `config/full_grid.yaml` and `docs/study_spec.md` inside the ZIP. The script `code/llm_parallel.py` is the exact adapter from Santa Fe records to the shared information estimator.

## 2. Where to start reading the package

| File | What it contains |
|---|---|
| `figures/live_board_maps.pdf` | 390 pages of 6-by-9 `(rho,B)` maps, with nine `(q,q_c)` panels per page. |
| `figures/live_board_maps_index.csv` | Page number by metric, raw/null/excess view, beta regime and target arm. |
| `figures/live_board_maps.csv` | Every numeric cell plotted in the PDF; retains all physical cells. |
| `summaries/cell_metric_summary.csv` | Physical outcome, evidence, board, action and spending estimates with across-episode SD, SE and normal 95% intervals. |
| `summaries/per_run_summary.parquet` | One compact physical summary per independent episode, 373,248 rows. |
| `summaries/gain_vs_b0.csv` | Descriptive late-support gain relative to matched `B=0`, actual mean spending, and gain per actual post where its denominator is stable. |
| `information/round_information_estimates.parquet` | All 25 shared-engine pooled statistics per cell (72,900 rows), plus per-round estimates (772,740 rows total), estimator variants, bootstrap intervals, null summaries and overlap diagnostics. |
| `derived/entropy_information.csv` | Additional action/outcome entropies and the entropy-difference form of CMI at each declared epistemic conditioning. |
| `derived/propensity_response.csv` | Propensity-weighted response at lags 1–3 and the available-susceptibility ratio. |
| `derived/whole_cell_susceptibility_by_conditioning.csv` | Occupancy-weighted response point estimates for eight conditioning levels, with identified occupancy. |
| `derived/state_local_exact_K.csv` | Response point estimates and arm counts at each exact pre-action target count and conditioning level. |
| `derived/derived_diagnostics.jsonl` | One record per cell with the occupancy-level `eta_IR` components and sensing information from the known vote-sampling kernel. |
| `calibration/episode_sensitivity.csv`, `calibration/Tpi_episode_sensitivity.pdf` | The 12-condition estimator pilot described in section 6. |

The map index is the quickest way to find a comparison: select a metric and beta regime, then compare truth/false target pages with the same color scale. Each page's rows are `q={3,12,24}`, columns are `q_c={3,12,24}`, vertical cells are `rho={0.25,0.4,0.55,0.7,0.85,1}`, and horizontal cells are `B={0,3,...,24}`. The map PDF currently covers the physical summaries and **shared-engine** information/response measures. Additional entropy, IPW and state-local response values are in `derived/`; they were calculated but do not yet have their own map pages. Read `figures/live_board_maps.csv` with `keep_default_na=False` in pandas if you want its literal `view="null"` label preserved.

## 3. Physical outcomes and what “cost” means

`late_target_share` and `late_truth_share` are the two late-window support fractions. `final_target_majority` is the fraction of episodes with **strictly more than half** of agents supporting the controller target at `X_60`; ties are not majorities. `late_kappa_plus` and `late_kappa_minus` summarize distinct active truth- and false-favoring fact coverage, while `late_phi_plus` is the share holding **all eight** truth-favoring facts. This `phi_plus` is an ownership statistic, not a proof of logical correctness. `late_overloaded_share` measures agents above seven active facts. The board metrics distinguish target-vote composition, positive-fact composition, board size and distinct controller fact IDs. `late_target_temporal_sd` captures variation **over rounds within an episode**; `sd_across_episodes` in the cell summary captures variation **between independent episodes**.

The nominal budget `B` is posts **when the gate is active**, not total expense. `intervention_frequency` is the mean gate activation rate. `cumulative_spending` sums actual controller posts over 60 rounds. In `gain_vs_b0.csv`, gain is mean late target support at `B` minus the corresponding physical `B=0` mean. Its reported uncertainty treats the episodes as unpaired; shared initialization seeds do not make the complete post-initialization paths identical. `gain_per_actual_post` divides this descriptive gain by mean **actual** spending only when that denominator is positive and more than twice its SE. It is not a theoretical efficiency bound.

For late target support, the declared 95% half-width target was at most `0.02`. **132 of 2,916 cells** missed it at 128 episodes; inspect `precision_sufficient` in `summaries/cell_metric_summary.csv`. The PDF itself does not draw uncertainty on every pixel, so use the CSV alongside it.

## 4. Entropy, MI and CMI: which question each asks

All shared-engine direct-counting information estimates use **base-2 logarithms and bits**. The primary `estimate` is the **unsmoothed plug-in** value. Alternative `jeffreys` and `miller_madow` columns are exported for information statistics; they are not pooled with the primary value. The additional entropy table also uses unsmoothed plug-in counts. `N_t` below denotes the binary population count vector, `Y_t` the controller's sample, and `E_t` an explicitly named pre-action epistemic representation.

| Exported name | Definition and reading |
|---|---|
| `round_sensing_mi` | `I(N_t;Y_t)`: full population-vector to full sensor-vector MI. |
| `round_target_sensing_mi` | `I(K_t;Y_t^target)`: scalar target-count sensing MI. In this binary model the full and target coordinates can be bijective, but the named estimands remain explicit. |
| `round_sensor_action_mi` | `I(Y_t;U_t)`: how the sampled votes inform the stochastic gate. |
| `I_U_Knext` | `I(U_t;K_(t+1))`: **unconditioned** action/outcome MI in `derived/entropy_information.csv`. |
| `round_target_actuation_cmi` (`T_pi`) | `I(U_t;K_(t+1)\mid K_t)`: observational one-round transfer information. `U_t` is at the start of round `t`; its outcome is after that round's 24 slots. |
| `round_population_actuation_cmi`, `round_truth_actuation_cmi` | The same one-round action channel using population and truth-count coordinates. In a fixed binary target arm these count coordinates are bijective; they are mapping checks, not independent effects. |
| `round_order_actuation_cmi` | An order/majority-strength outcome channel, which need not equal the target-count channel. |
| `round_memory_target_actuation_cmi` | `I(U_t;K_(t+1)\mid K_t,E_t)` with the exact histogram of agents' truth-favoring fact counts. Often sparse. |
| `round_kappa_target_actuation_cmi`, `round_phi_target_actuation_cmi`, `round_susceptible_target_actuation_cmi` | The same outcome and lag, additionally conditioned on a three-bin truth-fact-coverage, all-positive-facts ownership, or `1-phi_plus` coordinate. These are **separate** conditions, not one joint condition. |
| `round_epistemic_target_actuation_cmi` | The same outcome and lag, conditioned on the joint four-bin `(kappa_plus,phi_plus)` state. |

The `derived/entropy_information.csv` table also contains `H(K_t)`, `H(U_t)`, `H(U_t\mid K_t,E_t)`, `H(K_(t+1)\mid K_t,E_t)`, and `H(K_(t+1)\mid K_t,E_t,U_t)`. Their last-two-entropy difference checks the corresponding CMI using the same unsmoothed count table. The `none` level means conditioning on `K_t` without extra `E_t`; other levels include exact memory, coarse coverage/ownership, exact `(r,s,vote)` histogram, and overload share. `H(U_t\mid K_t,E_t)` is estimated from action frequencies. The separately named `H_U_given_sensor_full_known_propensity` is the mean binary entropy of **known** `e_t`, so it conditions on the realized sensor and is a different quantity.

For the shared estimator, `round_controller_action_entropy` is `H(U_t)` and `round_controller_action_entropy_given_population` is `H(U_t\mid N_t)`. `round_target_information_fraction` divides raw `T_pi` by `H(U_t\mid K_t)` where defined; it is a descriptive information fraction, not a response efficiency. The full-grid `derived_diagnostics.jsonl` separately contains `eta_IR`, its raw numerator and denominator, identified-state support and a validity flag. Its sensing diagnostic uses the **known hypergeometric vote-sensor kernel** and empirical state occupancy, and is reported in **nats**; do not compare its number directly with a bits-valued MI without converting by `ln(2)`. There is no separately validated thermodynamic efficiency bound in this package.

## 5. Susceptibility and causal response

For a pre-action state `C_t`, define `Delta X_t=X_(t+1)-X_t`. The state-matched susceptibility is

\[
\chi(C)=E[\Delta X_t\mid U_t=1,C_t=C]-E[\Delta X_t\mid U_t=0,C_t=C].
\]

`round_target_susceptibility` takes `C=K_t` **exactly**. It computes the two action-arm means at each visited `K`, discards states lacking either arm, and averages local differences with weights equal to their number of observations. Units are **target-fraction change per round**. It is not a derivative with respect to budget `B`, and it is not the aligned-magnetization response (which has a different scale). `round_memory_target_signed_response` and the `kappa`, `phi`, `susceptible` and joint `epistemic` signed-response columns repeat this matched difference with exactly the corresponding CMI's expanded conditioning. The shared table supplies episode-bootstrap intervals for pooled response estimates and overlap columns such as dual-action event fraction and singleton fraction. A wide exact condition may have little identified occupancy despite a finite point estimate.

`derived/state_local_exact_K.csv` retains local `chi` point estimates by exact `K` and evidence level plus action/silence counts. `derived/whole_cell_susceptibility_by_conditioning.csv` weights identified local strata by their event counts. Both expose `identified_occupancy_mass` and leave unsupported states missing; they do **not** contain episode-bootstrap intervals. Matching on `K` (even with coarse evidence bins) does not by itself prove that the two observed action groups have identical hidden states.

We also calculated a distinct, propensity-weighted response using the **logged randomized gate**:

\[
w_t=U_t/e_t-(1-U_t)/(1-e_t),\qquad
\widehat\tau_h=\frac1{M_h}\sum_t w_t\,[X_(t+h)-X_t],\quad h=1,2,3.
\]

These `tau_ipw_lag_1/2/3` rows are in `derived/propensity_response.csv`, with SEs computed **across complete episodes**. `h=1` ends after the same round's action and peer updates. The `h=2,3` outcomes also include later usual-policy actions; they are not sustained forced-control effects. The available-susceptibility ratio instead sums `w_t Delta X_t` only where `X_t<1` and divides by the sum of `1-X_t` over those same eligible rows; its numerator, denominator and saturated exclusions are exported. That ratio currently has **no interval**. State-local available response is a mean of per-row `w_t Delta X_t/(1-X_t)` and is a different weighting.

There were no matched forced intervention/silence rollout branches in this version. Do not label observational `T_pi` or matched `chi` as a branch experiment. At `B=0`, physical posting has exactly zero effect by construction, but a virtual gate and state-dependent policy remain logged; observational MI/CMI or finite-sample response estimates need not be numerically zero.

## 6. Nulls, uncertainty and the `T_pi` sensitivity pilot

Every cell's pooled shared information estimate has `estimate`, `null_mean`, `estimate_minus_null`, a null tail probability, an episode-bootstrap 95% interval, and occupancy/overlap diagnostics. For actuation CMI, the **policy-conditional randomization** null redraws `U_t` from its logged propensity while holding the recorded trajectory fixed. It tests a conditional observational reference; it does **not** generate a new counterfactual outcome trajectory. Sensing/policy MI uses a **sensor-permutation** null. `estimate_minus_null` may be negative and should not be clipped to zero. A null is not attached to the matched susceptibility column. Full-grid pooled estimates used 99 bootstrap draws and 99 null draws per configured statistic, with unsmoothed direct counting as the primary estimator.

The separate 12-condition, 512-independent-episode calibration bank varied the number of **complete episodes** used per dataset. Its 19-null-draw series covers 8, 16, 32, 64, 128, 256 and 512 episodes; its 99-null-draw series covers 32 through 512. Groups are disjoint **within** each sample size, but the sizes reuse the same episode bank, and at 512 there is only one group. An independent 128-episode pilot bank is also included. `calibration/episode_sensitivity.csv` covers `T_pi`, four extra epistemic CMI conditions, sensing MI, sensor–action MI and susceptibility. `calibration/Tpi_episode_sensitivity.pdf` plots raw `T_pi` and raw-minus-policy-null versus independent episode count. This is real sample-size sensitivity analysis, **not** an independently calibrated bias/RMSE/coverage/power study; it cannot yield one reliable universal minimum `n`.

The pilot found substantial finite-sample positive bias in some raw plug-in CMI results. The shared percentile-bootstrap interval for raw CMI sometimes excludes its own observed estimate, so do not treat it as proven calibrated coverage. For the maps, inspect raw and excess-information pages together with action overlap and conditioning-state counts. The 132 late-support cells missing the 0.02 half-width target need more independent episodes or an explicit precision-insufficient label. Extra entropy and `eta_IR` outputs currently have no validated uncertainty interval.

## 7. Retention and limits on further analysis

All calculations above ran **before** per-cell raw rounds and micro-slots were compacted. Following the storage decision, each cell retains all 128 per-episode **physical summaries**, its information and null-analysis files, its extra entropy/response and local-response files, and eight deterministically chosen illustrative trajectories. The ZIP includes illustrative rounds for 24 selected physical cells, not the whole raw archive. The full result root keeps every cell's compact outputs and individual null draws.

The retained local-response tables contain arm counts and **mean changes**, but not every state/arm outcome histogram or episode identity. Thus new state-local information estimates and new episode-bootstrap intervals for those local response curves cannot be reconstructed from this ZIP alone. Those require new raw data. The archive does contain all already calculated full-grid pooled and per-round information estimates, their summarized null results, full-grid physical summaries, state-local response point estimates and the 12-condition sensitivity pilot.

The files `MANIFEST.json` and `code/` identify what was packed and the scripts/versions used. This tutorial documents the calculations; it does not assert that every visual pattern is statistically or numerically resolved.
