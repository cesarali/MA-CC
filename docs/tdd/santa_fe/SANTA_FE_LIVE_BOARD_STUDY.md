# Santa Fe live-board simulation: revised mechanics and study specification

**Execution update (29 September 2026):** this is the original design document.
The v4 implementation and full-grid run are complete; see the
[implementation and results guide](../../documentation/games/santa_fe/live_board_v4_implementation_and_results.md)
for the resolved configuration, code map, artifact paths, coverage, and limits.
The status line below records the document's status **when written**, before launch.

Date: 27 September 2026. Revision: whole-decision overload correction. Status: implementation specification; no experiments launched.

## 1. Scope and decisions

Implement a new, versioned microscopic Santa Fe model with the live-board clock of the documented LLM blackboard game. Study only this simulator. Do not run, fit, or plot mean-field theory. Do not reinterpret the September 26 frozen-board results as results for this model.

Confirmed by the user: 24 agents; six persistence values from 0.25 to 1; integer budgets every three posts from zero to 24; both false-target and true-target control; 12 distinct facts (eight truth-favoring, four false-favoring); confusion begins above seven active facts. Include the three existing beta regimes, with balanced results presented first.

Interpretation from the preceding discussion: the three reading and sensing values are each {3,12,24}, not a single fixed value of 3. Proposed choices below are explicit defaults, not claims about existing implementation or empirical LLM calibration.

## 2. Exact parameter grid

| Axis | Values | Count |
|---|---|---:|
| Persistence rho | 0.25, 0.40, 0.55, 0.70, 0.85, 1.00 | 6 |
| Active-round posting budget B | 0, 3, 6, 9, 12, 15, 18, 21, 24 | 9 |
| Agent reading q | 3, 12, 24 | 3 |
| Controller sensing q_c | 3, 12, 24 | 3 |
| (beta_E,beta_S) | balanced (1,1.3); evidence-weighted (2,0.5); social-weighted (0.5,2) | 3 |
| Controller target Z | truth +1; false -1 | 2 |

Take the full Cartesian product: **2,916 design cells**. The balanced subset contains **972 cells**. Each fixed (q,q_c,beta,target) has a complete 6-by-9 rho–B map. The user's integer-step correction replaces the original six-budget request; nine budgets are intentional.

Keep B=0 rows for both target labels and every q_c so the analysis grid remains rectangular. Their physical population dynamics are equivalent at fixed rho,q,beta and initialization; target coordinates and sensor observables differ. Avoid counting reused baseline trajectories as independent episodes. Initially prefer separate episodes per design cell for a simple provenance contract.

Use 60 rounds as a proposed continuity default, not an equilibrium claim. Primary population summary: per-episode mean over rounds 41–60; also export all trajectories and final-round distributions. Primary information window: all 60 action–outcome transitions, with round-conditioned and late-window diagnostics. Every table must identify its window; pooling transient rounds defines a time-mixture estimand.

Do not pick an episode count from the old support-only precision criterion. Specify it after a small, separately authorized estimator pilot. For budgeting only, 256 episodes per cell would mean 746,496 episodes (248,832 balanced); these are arithmetic examples, not a launch instruction or a precision guarantee.

## 3. New state and round clock

Agent i has vote v_i in {-1,+1}, active fact set K_i, and historical fact set H_i, with K_i a subset of H_i. Define r_i=|K_i intersect F_plus|, s_i=|K_i intersect F_minus| and L_i=r_i+s_i. Facts have stable IDs. Repetition never creates duplicate knowledge. Reading a forgotten fact reactivates it. Historical memory is audit data and does not enter voting.

Proposed initialization, closer to the LLM document: each agent receives one fact; allocate each of the 12 facts to two randomly selected agent slots, with exactly one fact per agent. Thus all facts are initially represented with redundancy two. Set H_i=K_i; draw initial votes from the voting rule with social field zero. Start with an empty live board. Save exact initial states and use shared initialization IDs when pairing true and false control. Use separate random streams for initialization, sensor, gate, forgetting, focal selection, reading, voting and posting.

Define X_t as population target share at the start of transition t, before sensing and forgetting. One transition is:

1. Record all current votes and pre-action evidence, including exact active and historical inventories and derived summaries.
2. Sense q_c distinct **agents' current votes**, uniformly without replacement. Y_t counts votes for target Z. This replaces sensing peer posts: repeated focal updates can generate several posts from one agent and no posts from another.
3. Draw U_t from Bernoulli(e_t), with proposed LLM-matched gate e_t=sigmoid[4(0.5-Y_t/q_c)]. Log Y_t, e_t and U_t. Keep this gate stochastic in both target arms. This deliberately changes the old threshold 0.55/sharpness 8; record the change in resolved configurations.
4. Expire all previous-round posts for participant delivery. Retain their audit records. Independently retain each active fact with probability rho; historical sets are unchanged. Record this post-forgetting, pre-posting evidence state separately.
5. If U_t=1, append B controller messages to the new live board at dawn. If U_t=0, append none. No later controller posting this round.
6. Perform 24 micro-slots. At each slot choose a focal agent uniformly **with replacement**. Read up to q currently live messages, uniformly without replacement, excluding that agent's own posts. Empty board means zero social field and no evidence acquisition; never substitute population ballots.
7. Form the union of current active facts and facts carried by the sampled messages. Use these facts in the vote calculation, then commit the vote and evidence updates. Immediately append one peer message. Later slots can sample this message during the same round.
8. Record end-of-round population, evidence, messages and actual spending. This state is X_(t+1), the outcome of U_t. Continue even at consensus.

An early agent reads and then posts; a later agent can read that post. An agent selected again can read new posts from others but not its own. Board size is variable throughout the day. q=24 is a capacity, not a guarantee of 24 eligible messages.

For causal/CMI conditioning use the stage explicitly declared by each estimator. Default is the start-of-transition state from step 1. Post-forgetting snapshots from step 4 are permissible secondary conditioning variables only because neither forgetting nor its random stream depends on U_t; label them separately. Never condition a causal response on evidence acquired after controller exposure.

## 4. Reports, targets, and remaining differences from the LLM game

Retain the simple report-only Santa Fe communication rule: each updated agent posts its new vote plus one uniformly chosen active fact aligned with that vote, or a vote-only report if it has no aligned fact. These are modeling choices; the LLM can instead choose not to post and can use richer message types.

The controller's proposed report pool is the fixed set of facts favoring its target. Each of its B posts carries target vote Z and an ID drawn uniformly with replacement from that pool. All messages enter the same sampling pool; controller posts get no sampling priority. This permits B=24 with only four false-favoring or eight truth-favoring IDs. Repeated reports can amplify exposure and social influence but not distinct fact counts.

This repeats IDs within a dawn and therefore differs from the documented LLM distinct-card controller. Capping posts at the distinct pool size would destroy the requested budget meaning; do not silently do so. Record posts and unique posted facts separately. Controller access is its frozen public report pool and sampled votes, not agents' private inventories.

Here “false control” means advocating the false answer using the designated false-favoring facts. Do not describe these as fabricated evidence: their sign is part of this abstract model. This simulator remains binary, with signed evidence and a prescribed posting rule; it does not reproduce three-option reasoning, proof entailment, REQUEST/DIRECTIVE semantics, or an LLM's strategic text selection.

## 5. Voting with reasoning overload

Proposed first implementation:

\[
E_i=\begin{cases}(r_i-s_i)/(r_i+s_i),&L_i>0,\\0,&L_i=0,\end{cases}
\qquad S_i=\text{mean vote in the current sampled messages},
\]

with S_i=0 for an empty sample. Use updated counts after incorporating sampled facts.

\[
g(L)=\frac{1}{1+\alpha\max(0,L-7)},\qquad
P(v_i^{new}=+1)=\sigma\bigl[g(L_i)(\beta_E E_i+\beta_S S_i)\bigr].
\]

Proposed alpha=0.2: g(7)=1, g(8)=5/6, g(10)=5/8 and g(12)=1/2. The overload factor multiplies the ENTIRE voting logit: both evidence and social influence. Above seven facts agents retain their knowledge but make a less reliable overall decision. For a fixed combined field, increasing overload moves the vote probability toward 1/2; it does not reverse the field or force an immediate fair coin. At twelve facts the whole logit is halved. Equivalently, the decision temperature is 1+alpha*max(0,L-7).

Load L counts ALL distinct active facts of both signs, including newly read facts: L=r+s, not only truth-favoring facts or the facts supporting the chosen vote. Previously forgotten facts absent from the current read do not count; repeated IDs count once. This preserves the specified fact-count load rather than introducing a separate message-length or repetition load.

This is a phenomenological hypothesis about reasoning, not an estimate of real LLM capacity. It attenuates the combined evidence-and-social decision field; it does not guarantee that adding any particular fact reduces accuracy, because the evidence and social inputs can also change. Keep the threshold and alpha configurable and log g(L). alpha=0 recovers the same live-board model without overload. A later matched alpha=0 diagnostic can isolate overload; do not add a second full grid by default.

## 6. Shared information-analysis contract

Adapt the existing MA-CC information engine to these records; do not infer execution from the existence of estimator code. Retain implemented estimator names, exact formulas, units, time boundaries, windows, conditioning partitions, counts, uncertainty, nulls and status. Calculate each metric within each physical cell before any averaging across cells. Use bits consistently with the shared engine, with explicit conversion if another export uses nats.

Let K_t=N X_t be target count, V_t the two-component population count vector, Y_t the sensor target count, and U_t the binary gate. Required quantities:

| Family | Required quantities |
|---|---|
| Population/action uncertainty | H(K_t), H(U_t), H(U_t given K_t), H(U_t given K_t,E_t); also mean within-population binary vote entropy, separately labeled from H(K_t) |
| Sensing/policy MI | I(K_t;Y_t), I(V_t;full sensor counts), I(Y_t;U_t) |
| Unconditioned action MI | I(U_t;K_(t+1)) |
| One-step transfer information | T_pi=I(U_t;K_(t+1) given K_t) |
| Epistemic CMI | I(U_t;K_(t+1) given K_t,E_t), for every declared evidence representation below |
| Outcome uncertainty | H(K_(t+1) given K_t,E_t) and H(K_(t+1) given K_t,E_t,U_t); their difference checks CMI |
| Population/truth/order channels | Shared population-, truth- and order-actuation estimators where their input definitions apply |

The one-step quantity conditions on one outcome lag; it is not transfer entropy with arbitrarily long histories. In this binary model truth count, target count and full vote-count vector are bijections within a fixed target arm, so their corresponding count-based information channels can be identical. Report this as a mapping check, not three independent findings. Do not assume the coarse majority-strength channel is equivalent.

Epistemic representations:

- No extra epistemic conditioning.
- Exact histogram of per-agent truth-favoring fact counts, r=0,...,8, for the shared memory estimator.
- kappa_plus=mean(r_i/8), kappa_minus=mean(s_i/4); use kappa_plus for the LLM supporting-truth-fact analogue and retain both signs for extended diagnostics.
- phi_plus=share holding all eight truth-favoring facts. This is complete positive-pool ownership, not a validated logical proof or guaranteed correct vote. Do not relabel it symbolic solvability phi-star.
- Separate coarse kappa_plus, phi_plus and 1-phi_plus conditionings; joint (kappa_plus,phi_plus). Preserve shared partitions: three scalar bins with edges 0,1/3,2/3,1; four bins per axis for the joint diagnostic.
- New extended diagnostics: exact joint histogram of (r_i,s_i,v_i) and coarse overloaded-agent share Pr(L_i>7). Exact histograms may be too sparse; retaining them is mandatory, claiming reliable estimates is not.

Keep evidence semantics anchored to truth in both target arms. Optionally export target-aligned fact coverage separately with an explicit label. Do not silently switch the meaning of kappa or phi in false-control cells.

For each information estimate export raw value, mean policy-null value, raw-minus-mean-null, null replicates or reproducible seed/count, episode bootstrap interval, occupancy, distinct conditioning states, singleton fraction and dual-action event/state fractions. Preserve negative null-adjusted estimates. Estimate-minus-null is the requested estimator diagnostic, not a theory-minus-simulation plot.

For actuation CMI use the documented reference that redraws U from each logged e_t while keeping the observed trajectory fixed. This is a policy-reference diagnostic, not a new counterfactual trajectory and not proof of causality. Use the engine's documented marginal-preserving null for each ordinary MI; never apply a global action shuffle in place of the state-dependent policy reference. Label the null algorithm per metric.

Do not replace H(U|K,E) with mean h(e_t): the latter additionally conditions on the controller's observed sensor information and can be smaller. Compute ceilings with the same conditioning as their numerator.

## 7. Susceptibility and causal response: mandatory separate outputs

Let delta X_t=X_(t+1)-X_t and C_t denote the declared pre-action state partition.

State-matched susceptibility:

\[
\chi(C)=E[\Delta X_t\mid U_t=1,C]-E[\Delta X_t\mid U_t=0,C].
\]

Export state-local and occupancy-weighted whole-cell values, with and without the epistemic conditionings. Units are target-fraction change, not twice that value in alignment units, and not a derivative with respect to B. Coarse matching alone need not remove policy selection.

Also compute logged-propensity causal response:

\[
w_t=U_t/e_t-(1-U_t)/(1-e_t),\qquad
\widehat\tau_h=\frac1{M_h}\sum_t w_t(X_{t+h}-X_t),\quad h=1,2,3.
\]

The h=1 endpoint is the end of the same round in which the dawn action occurs. Larger lags allow subsequent actions under the usual policy, rather than holding later control fixed. Respect episode boundaries. Bootstrap complete episodes or shared-initialization blocks, not micro-slots or individual rounds.

Available susceptibility: for X_t<1, retain row contributions w_t delta X_t/(1-X_t). The state-local summary is their mean; the cell summary is the ratio of summed causal contributions to summed available mass over the same eligible nonsaturated rows. Export excluded saturated counts and denominator. These summaries are different estimands; never fill saturated or unsupported cases with zero.

At B=0 the action has no physical effect. Keep a randomized virtual gate and sensor logs: physical causal response is zero, but finite estimates need not be, and sensor/policy MI can be positive. Coarsely conditioned observational action information may retain state-selection dependence. Do not force every B=0 information estimate to zero.

Export eta_IF and eta_IR only with the shared engine's matched conditioning, weighting, units and component definitions. Keep raw numerator/denominator and reliability flags. Do not substitute propensity response for the susceptibility in an existing eta_IR formula, or use null-adjusted information in that denominator. Unsupported efficiencies and thermodynamic quantities stay unsupported.

## 8. Figures and finite-horizon maps

Required primary figure family: horizontal B, vertical rho, one 6-by-9 map per observable and fixed (q,q_c,beta,target). Present balanced agents first, with separate truth-control and false-control panels. Include target and truth support, final majority probability, evidence coverage, overloaded share, spending, every supported information/entropy family, state-matched susceptibility, causal response, and available susceptibility.

Required secondary views: curves versus B at fixed rho, and versus rho at fixed B. Include uncertainty. Also retain state-local target-share-versus-B maps at fixed rho and the other physical parameters, with matched occupancy/support panels. Aggregate local information using exact target-count conditioning inside a displayed coarse target-share bin; do not silently redefine CMI by replacing its conditioning count with the display bin.

All panel letters must have explicit caption definitions. Use common scales for comparable target arms. Show raw/null/excess-information views separately. Mark unsupported cells visibly; do not interpolate them or equate them with zero. There are no theory curves or theory-difference panels. Call these finite-horizon regime maps; do not claim thermodynamic phase transitions or stationarity from 60 rounds.

## 9. Retention, pilot and implementation acceptance

Retain all per-episode round transitions, micro-slots, exact fact inventories, message IDs/authors/votes/facts/lifetimes, sampled message IDs, sensor samples, actions/propensities, initializations and random seeds. Record actual versus requested posts, exposure, unique acquisitions, reactivations, forgetting, L, g(L), and pre/post state snapshots. Save individual records across the whole grid, not only ensemble averages or 12 selected cells.

Produce canonical round and micro-slot tables usable by the shared analyzer, complete cell/config/seed manifests, estimator coverage/status tables, information and susceptibility tables, uncertainty/null/support outputs, plotted numeric data, a figure index, and an analysis completion seal. The agent must demonstrate that the analyzer consumed the new Santa Fe records; no result is considered delivered merely because it is recoverable later.

A later estimator pilot should vary numbers of complete independent episodes at a few fixed physical settings, preserving the physical model. Report raw, null and excess MI/CMI plus response uncertainty at each count. Nested subsamples are stability checks, not independent replicate datasets. Include all conditioning levels; insufficient support for exact-state estimators must remain explicit. Do not launch this pilot or the production grid as part of preparing this document.

Before launch, implementation checks must establish: same-round post visibility; previous-day expiry; own-message exclusion; sampling without replacement within a read; focal selection with replacement; current-agent-vote sensing; distinct fact-set acquisition; historical memory preservation; correct overload breakpoint and attenuation of BOTH evidence and social terms (including a social-only nonzero field); exact B posts even above pool size; correct true/false target conversion; action–outcome timing; B=0 absence of physical actuation; no accidental dependence of random streams on virtual actions; whole-episode/bootstrap grouping; and verified adapter mappings for every requested metric.

Return the revised game document, resolved grid/configurations, metric compatibility table, meaningful mechanics checks and proposed episode budget for review. Production simulation remains a later task.

## 10. Source basis

- Supplied blackboard_game.md, sections 5–7: active/historical memory, vote sensing, randomized gate, dawn posting, expiry, asynchronous live-board reading and updates.
- Supplied metrics.md, sections 6–7: information, response, entropy, aggregation and null definitions.
- Supplied metrics_epistemics.md, sections 7–11: conditioning levels, support diagnostics, causal response and available susceptibility.
- Supplied study_aggregation_contract.md: canonical records, coverage, provenance and completed-analysis packaging.
- September 26 ZIP, study_A/DEFINITIONS_CONFIGURATIONS.md: old fixed-front-page mechanics, beta pairs, fact pool and clock. It is the baseline being changed, not the specification of this new study.

This document specifies an LLM-inspired microscopic surrogate. It does not claim exact equivalence to LLM reasoning or an already implemented simulator.
