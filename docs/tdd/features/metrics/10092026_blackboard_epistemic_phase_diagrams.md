# Epistemic regimes and control: five focused analysis extensions

## Scientific objective

Test whether false-target control exploits **missing evidence, fragile evidence, distributed evidence, or a failure to translate available evidence into votes**. The central outcome is a map of dynamical regimes, not a longer list of correlations.

Implement as an offline extension of the current study analysis described in `metrics.md`. Reuse `propensity_weighted_causal_response`, canonical rounds, micro-slots, exact inventory snapshots and frozen task definitions. These are proposals to test, not findings from the partial snapshot. No new LLM calls are needed if those sources are retained.

Use two distinct diagram types:

- **Parameter maps:** horizontal axis assigned budget b, vertical axis persistence rho. These compare experimental conditions.
- **State maps:** horizontal axis current target share x, vertical axis one epistemic coordinate. These describe where the system is and how it moves, within fixed experimental conditions.

Call them finite-horizon regime maps; smooth colors alone do not establish a phase transition.

## 1. Collective versus individual truth accessibility

**Question:** Is the correct answer identifiable from the evidence, and is that evidence assembled inside any agent?

Let A_it be agent i's active facts. Define a symbolic predicate S(A)=1 when A uniquely determines the gold answer under the task rules. Evaluate all compatible worlds or use the existing solver; do not require one particular proof if alternative proofs suffice.

$$
G_t=S\!\left(\bigcup_i A_{it}\right),\qquad
\phi_t^*=\frac1N\sum_i S(A_{it}),\qquad
D_t^{\rm frag}=G_t-\phi_t^*.
$$

The last quantity is the gap between collective and individual solvability. At G=1, a large gap means sufficient evidence exists but remains distributed. At G=0 it is zero, so always show G alongside it. Compare phi* with the existing full-proof share; retain the existing name only if their semantics agree.

**Diagram:** b-versus-rho maps of time with G=1 and mean fragmentation gap. Companion x-versus-phi* occupancy map, with G=0 shown separately.

**Interpretation:** Separate epistemically insufficient states, collectively solvable but fragmented states, and individually solvable states. G concerns the union of participant active inventories; it does not include the controller's private pool, inactive historical records, or unacquired board content. Report that scope explicitly. G=0 means insufficient current evidence, not that truth cannot be guessed or later recovered.

**Requires:** exact active IDs, task semantics, gold answer. Cache solver evaluations by task and sorted fact set. Solver failures or empty compatible-world sets are invalid, not G=0.

## 2. Robustness of truth to the next forgetting step

**Question:** Two populations may both have G=1, but is one a single fact loss away from losing solvability?

For a currently solvable state, define

$$
V_\rho(A_t)=\Pr\!\left[S\!\left(\bigcup_i\widetilde A_{it}\right)=1\mid A_t\right],
$$

where each agent-fact occurrence is independently retained with probability rho, exactly matching the configured forgetting rule. This is a **one-boundary robustness diagnostic**, with no new acquisition or communication. It does not simulate a whole future round.

For one proof consisting of distinct required facts f, with n_f active holders,

$$
V_\rho^{\rm proof}=\prod_f[1-(1-\rho)^{n_f}].
$$

This is exact for survival of that proof under independent thinning. With alternative proofs it is generally only a sufficient-proof lower bound; estimate full solvability by inexpensive seeded thinning draws plus the symbolic solver. Report Monte Carlo uncertainty. Where G=0, mark current solvability separately rather than mixing those states into a conditional robustness average.

**Diagram:** b-versus-rho map of mean V_rho among G=1 states, next to the fraction G=1. For comparing redundancy across rho settings, also evaluate the same states at one fixed, declared reference persistence.

**Interpretation:** Distinguishes robust truth availability from precarious availability. This directly tests whether forgetting structurally removes the opportunity for collective reasoning.

## 3. Joint motion of votes and evidence

**Question:** Does control push votes toward its target while spreading, preserving or eroding usable evidence?

For z_t=(x_t,phi*_t), estimate the two immediate branch drifts at comparable pre-action states s:

$$
v_u(s)=E[z_{t+1}-z_t\mid\mathrm{do}(U_t=u),s],\qquad
\delta v(s)=v_1(s)-v_0(s).
$$

Use the existing propensity score multiplier W_t=U_t/e_t-(1-U_t)/(1-e_t). Then average W_t times each component of the change to estimate delta v. Estimate v_1 and v_0 using U/e and (1-U)/(1-e), respectively, over the same target population of records.

**Diagram:** x-versus-phi* maps with coarse drift arrows for silence and activation, plus a separate causal displacement-arrow map. Keep b and rho fixed or facet a few prespecified conditions. Mark unsupported bins.

**Interpretation:** In false-target runs, positive delta v_x together with positive delta v_phi would mean activation increases both false-target support and proof ownership on average. Positive vote response with negative evidence response suggests a different mechanism. These are population-average joint effects, not proof that the same individuals gained proof and changed vote. Drift reversals are candidate regime boundaries; do not call them attractors without additional evidence.

## 4. Epistemic modulation of causal susceptibility

**Question:** Does increasing proof ownership change the strength or sign of control at the same vote state?

$$
\chi(s)=E[W_t(x_{t+1}-x_t)\mid s],\qquad
\mathcal E_\phi=\frac{\partial\chi}{\partial\phi^*}.
$$

Start with a linear score regression on x, phi* and round index inside a physical cell. The phi* coefficient is a first approximation to the gradient. Report its effect per 0.1 increase in phi*, with a block-bootstrap interval. Check against a prespecified low/high-phi* comparison with overlapping x and time support. If x and phi* are nearly collinear, mark the gradient unidentified at useful precision.

**Diagram:** x-versus-phi* heatmap of causal susceptibility, at selected fixed budgets and persistence values; overlay a zero-response contour only where supported. Keep the existing eight x bins, but use only two or three epistemic bands initially. The fitted surface and raw binned estimates must be visibly distinguished.

**Interpretation:** Tests whether proof availability protects against control, leaves it unchanged, or changes its effect. This is heterogeneity of the randomized activation effect, not the causal effect of manipulating knowledge. Do not automatically equate 1-phi* with the fraction able to change votes.

## 5. Timing of false capture relative to evidence loss

**Question:** Does false agreement emerge only after truth becomes unavailable, or can it precede evidence loss and persist alongside solvable evidence?

For episodes starting with G=1, define first loss T_loss=min{t:G_t=0}. Define false capture T_cap as the start of h consecutive rounds with target share at least theta; use a prespecified descriptive rule, for example theta=0.75 and h=3. Require the full h-round window to be observed.

Report the fraction with observed T_cap<T_loss, unresolved/censored episodes, and occupancy of the joint state {x>=theta,G=1}. For initially G=0 episodes, report a separate trajectory category; do not assign an invented loss time. Recovery of G after a loss is possible, so first loss is not an absorbing extinction time.

**Diagram:** b-versus-rho maps of capture-before-first-loss probability by a common horizon and of persistent false capture with G=1 throughout the capture window. Show censoring/support beside them. With incomplete follow-up, report bounds or a competing-event analysis instead of treating missing times as infinity.

**Interpretation:** Addresses temporal ordering and coexistence. It does not identify a causal mediation pathway from forgetting to capture. Compare matched no-control trajectories where available; the randomized score remains the primary immediate causal effect.

## Minimal implementation and reporting contract

1. Establish exact timing: pre-action evidence must include any already-applied forgetting and exclude the current intervention. Do not blindly copy the previous after-state across a forgetting boundary. If exact reconstruction is impossible, use the last verified boundary with an explicit label.
2. Keep task, target identity/direction, controller policy, budget, rho and model identifiable. Do not relabel by eventual winner. Compare parameter cells with equal-episode weighting over a common horizon; label incomplete snapshots provisional.
3. Bootstrap complete shared-initialization/randomness blocks. Display action counts, independent block counts and occupancy. Thousands of micro-slots are not thousands of independent episodes.
4. Use only pre-action coordinates for causal grouping. Later proof state is an outcome in class 3, not a grouping variable. Never classify a present response by a future capture or loss event.
5. Preserve existing metrics and names. Add derived tables and diagnostic figures; do not insert these causal responses into the current eta_IR denominator/numerator combination without a consistent rederivation.

**Priority:** Implement classes 1 and 3 first, then 4. Add class 2 to diagnose persistence vulnerability and class 5 for the temporal narrative. The central figure should combine evidence availability, state-space motion and causal control response. The scientific question is whether control changes the route from distributed evidence to collective decisions, and in which regimes that occurs.
