# Santa Fe v3 — Simulation, Metrics, and Theory-Validation Update Plan

## Purpose

Update the existing Santa Fe synthetic simulator and its post-hoc analysis so that the microscopic simulation matches the current stochastic mean-field theory exactly enough to validate the theory rather than merely resemble it.

The simulator should remain the **microscopic ground-truth stochastic process**. Do **not** replace the simulator with the SDE. The SDE / heterogeneous mean-field model is an approximation that must be tested against the microscopic simulation.

The main model changes are:

1. **Night persistence is applied globally once per round/day**, not each time an agent is selected as focal.
2. **Peer messages are vote-aligned:** an agent first acquires sampled evidence and chooses its new vote; it then posts one active fact supporting that new vote, if one exists.
3. **Controller messages carry target-aligned facts:** when the controller acts, it injects `b` public messages, all voting for the controller target and each carrying a fact aligned with that target.
4. The simulation must expose enough microscopic state and event probabilities to validate the transition kernel, system-size expansion, stochastic board field, reduced closure, controller response, and information bounds.
5. **Reuse the existing MA-CC / Santa Fe estimation, bootstrap, null-model, permutation, support-diagnostic, and plotting infrastructure. Do not reimplement statistical machinery that already exists.** Extend its inputs and recipes only where necessary.

---

## 1. Preserve the old simulator for reproducibility

Do not silently change historical Santa Fe semantics.

Add a new explicit model/version flag, for example:

```yaml
model_version: santa_fe_epistemic_feedback_v3
```

or equivalent configuration switches such as:

```yaml
persistence_clock: round_boundary
peer_posting_mode: vote_aligned_fact
controller_message_mode: target_aligned_fact
board_clock: frozen_front_page
```

Historical configs should retain their previous behavior. New v3 configs should opt into the new semantics explicitly.

Record the resolved model/version and all semantic switches in every run manifest and output package.

---

# 2. Canonical v3 microscopic clock

The code and retained records should implement this exact day/night order.

For round/day `d`:

```text
state at start of boundary: {X_i, K_i}_{i=1}^N and front page B_d

1. NIGHT PERSISTENCE
   Apply rho once to every active fact of every agent.

2. DAYTIME
   Repeat N microscopic update opportunities:
      a. choose one focal agent uniformly
      b. sample q messages from the fixed front page B_d
      c. acquire / reactivate sampled fact IDs
      d. compute the focal evidence signal from its updated active facts
      e. sample the new vote using beta_E and beta_S
      f. choose one active fact aligned with the new vote, if available
      g. append that peer message to the new peer-board buffer C_d

   Important: B_d is read-only during the day.
   Messages generated during day d go to C_d and are not sampled until the next day.

3. CLOSE THE PEER BOARD
   C_d now contains exactly N peer emissions, one per microscopic update opportunity.

4. CONTROLLER SENSING
   Sample q_c messages from C_d without replacement.
   Y_d = number of sampled messages supporting the controller target c.

5. CONTROLLER POLICY
   Draw U_d in {0,1} from the existing soft policy pi(U|Y_d).

6. CONTROL EFFECT
   If U_d = 0: add no controller posts.
   If U_d = 1: add exactly b controller posts.
   Every controller post:
      - votes for target c
      - carries a fact whose direction is c
      - enters the same next-day front page as peer posts

7. NEXT FRONT PAGE
   B_{d+1} = peer posts C_d + controller posts.

8. Repeat.
```

At the sign-level mean-field representation this corresponds to

\[
\mathbf B_{d+1}
=\frac{\mathbf C_d+bU_d\mathbf e_{c,c}}{N+bU_d},
\]

but in code **keep actual message objects and fact IDs**. Normalization is an analysis/theory operation, not a reason to discard discrete messages.

---

# 3. Required microscopic model changes

## 3.1 Persistence: move to the round boundary

### Current issue to inspect

The older synthetic implementation applied persistence during focal updates. If that is still true, an agent selected multiple times in one day can be thinned multiple times and an agent never selected can escape persistence entirely.

### Required behavior

At the beginning of every round/day, for every agent `i` and every active fact `f in K_i`:

```text
keep f active with probability rho
otherwise deactivate it
```

This happens **once and only once per day**.

Historical/ever-known facts may remain stored separately if the simulator already distinguishes active versus historical knowledge; the v3 theory concerns the active set.

### Required retained fields

Per round:

- active fact IDs before persistence;
- active fact IDs after persistence;
- number of `+` facts lost;
- number of `-` facts lost;
- per-agent `(r_i, s_i)` before/after persistence;
- exact persistence log-probability if feasible.

---

## 3.2 Peer posting: vote first, then post a supporting fact

After board sampling and evidence acquisition, compute the focal agent's new vote exactly as before:

\[
P(X_i'=+1)
=\sigma\!\left(\beta_E E_i+\beta_S H_i\right).
\]

Then choose the public fact **conditional on the new vote**.

If `new_vote == +1`:

```python
eligible = [f for f in active_facts if fact_sign[f] == +1]
posted_fact = uniform_choice(eligible) if eligible else None
```

If `new_vote == -1`:

```python
eligible = [f for f in active_facts if fact_sign[f] == -1]
posted_fact = uniform_choice(eligible) if eligible else None
```

There is **no additional alignment parameter**.

Therefore the following must be impossible:

```text
vote + with posted - fact
vote - with posted + fact
```

A vote without any aligned active fact is allowed and produces a factless message.

### Required retained fields per microscopic update

- focal agent ID;
- class before sampling `(r,s,v)`;
- sampled board message IDs;
- sampled vote signs;
- sampled fact IDs and directions;
- newly acquired facts;
- reactivated facts, if historical memory exists;
- class after acquisition;
- evidence signal `E`;
- social signal `H`;
- vote logit;
- `P(vote=+1)`;
- realized new vote;
- eligible supporting fact IDs for that new vote;
- selected posted fact ID or `None`;
- selected posted fact direction;
- class after the complete focal update.

This makes the empirical one-step kernel reconstructible from retained data.

---

## 3.3 Controller fact pool and fact-bearing actuation

The controller now has an explicit pool of synthetic facts aligned with its target.

For controller target

\[
c\in\{-1,+1\},
\]

define

```python
controller_fact_pool = [f for f in all_facts if fact_sign[f] == c]
```

When `U_d = 1`, generate exactly `b` controller messages.

Each message has:

```text
vote = c
fact_sign = c
fact_id = selected from controller_fact_pool
source = controller
```

### Fact identity selection

Implement one explicit reproducible strategy and expose it in config.

Recommended first v3 baseline:

```yaml
controller_fact_selection: uniform_with_replacement
```

This keeps the analytic controller emission law simple and allows `b` to exceed the number of distinct target-aligned facts.

Optionally support a second mode later:

```yaml
controller_fact_selection: uniform_distinct_if_possible
```

but do not mix these semantics inside one study.

### Important causal constraint

Controller facts do **not** directly modify any agent's knowledge set.

The route must remain:

```text
controller action
 -> controller message on B_{d+1}
 -> focal agent samples that message on day d+1
 -> focal agent acquires/reactivates the fact
 -> epistemic state changes
 -> later vote probabilities change
```

### Required retained controller fields

Per round:

- controller target `c`;
- sensor sample IDs;
- `Y_d`;
- `P(U_d=1 | Y_d)`;
- realized `U_d`;
- requested / realized budget `b`;
- controller fact-pool IDs;
- selected controller fact IDs;
- controller message IDs;
- controller message categories `(vote_sign, fact_sign)`;
- later message exposures and acquisitions attributable to controller posts.

---

## 3.4 Frozen front-page board

For v3, the front page used during one day is fixed.

Maintain two buffers explicitly:

```text
B_current = fixed front page read during the current day
C_peer    = new peer emissions accumulated during the current day
```

Microscopic updates read only from `B_current`.

They write only to `C_peer`.

At the controller boundary:

```text
B_next = C_peer + controller_messages
```

Then swap buffers.

Add an invariant test that a peer message created during day `d` is never present in another focal agent's sample during the same day `d`.

---

# 4. Canonical theory state to save

The fundamental heterogeneous state is

\[
a=(r,s,v),
\]

where:

- `r`: number of active `+`-directed facts;
- `s`: number of active `-`-directed facts;
- `v in {-1,+1}`: current vote.

For every round boundary and, ideally, every microscopic slot, compute the class counts

\[
Z_{rsv}=\#\{i:(r_i,s_i,v_i)=(r,s,v)\},
\]

and fractions

\[
z_{rsv}=Z_{rsv}/N.
\]

Save the complete sparse occupancy vector, not just summary moments.

Also save the reduced observables:

\[
x=\sum_{r,s}z_{rs,+},
\]

\[
\kappa_+=\frac1{F_+}\sum_{r,s,v}r z_{rsv},
\qquad
\kappa_-=\frac1{F_-}\sum_{r,s,v}s z_{rsv}.
\]

Additional useful quantities:

- total active fact coverage;
- fraction with all `+` facts;
- fraction with all `-` facts;
- fraction with complete evidence;
- epistemic heterogeneity / variance of `(r,s)`;
- vote share conditioned on `(r,s)`;
- empirical correlation between epistemic state and vote.

---

# 5. Board observables and emission quantities

The v3 theory distinguishes three objects. Preserve these names in analysis where possible.

## 5.1 `R`: one-update peer emission law

For a starting class `a=(r,s,v)` and front page `B`,

\[
R_{\nu\ell\mid a}(B)
\]

is the probability that one focal update emits message category `(nu, ell)` after board sampling, acquisition, vote update, and aligned posting.

Message categories are:

```text
(+,+), (+,0), (-,-), (-,0)
```

The cross-sign categories must have probability zero in v3:

```text
(+,-) = 0
(-,+) = 0
```

Estimate empirical `R` from micro-slot records for state/board bins with enough support.

## 5.2 `R_bar`: mean peer emission distribution over one day

Compute

```text
mean normalized peer message-category counts during day d
```

and retain it as the empirical counterpart of

\[
\overline{\mathbf R}_d.
\]

## 5.3 `C_d`: realized peer-board count vector

Before control, store the actual count vector of the `N` peer posts:

```text
C_(+,+)
C_(+,0)
C_(-,-)
C_(-,0)
```

and the corresponding normalized fractions.

## 5.4 `B_{d+1}`: controlled next front page

Store peer and controller contributions separately and combined:

```text
B_peer_counts
B_controller_counts
B_total_counts
B_total_fractions
```

Never lose source attribution at the raw-data level.

---

# 6. Primitive probabilities to retain for future path-space analysis

Add enough information to reconstruct the probability of every primitive event without rerunning the simulation.

Where practical, save both probability and log-probability.

At minimum:

```text
p_persistence_event
p_board_sample
p_fact_acquisition_given_sample   # often deterministic once sample is fixed
p_vote_realized
p_peer_fact_selection
p_sensor_outcome
p_action_given_sensor
p_controller_fact_selection
```

For deterministic events record probability `1.0` rather than omitting them.

This is not yet a full entropy-production calculation. It is forward-path bookkeeping so that a reverse protocol can be defined later without changing the simulator again.

---

# 7. Analysis / metric extensions

## 7.1 Reuse the existing analysis engine

The existing Santa Fe / MA-CC analysis already contains infrastructure for:

- MI / CMI estimators;
- permutation or policy-resampling nulls;
- bootstrap uncertainty;
- support diagnostics;
- per-round and pooled summaries;
- parallel execution;
- saved plots and tables.

**Do not duplicate these.**

Instead:

1. add the new v3 state variables to the canonical trajectory/table representation;
2. add adapters / estimator recipes using the existing engine;
3. preserve the existing null semantics unless there is a demonstrated reason they are incompatible;
4. make every new estimate report the same support/null diagnostics already used elsewhere.

## 7.2 Core sensing quantities

Retain / calculate:

\[
I(n_d;Y_d),
\]

where `n_d` is the target-message count in the closed peer board used by the sensor.

Also retain:

```text
sensor absolute error
sensor squared error
sensor sample fraction q_c/N
I(sensor; action)
action entropy
```

Compare empirical sensing MI to the exact hypergeometric-channel calculation whenever the state distribution is known.

## 7.3 Action-to-vote information

Primary:

\[
I(U_d;x_{d+1}\mid x_d).
\]

Epistemically conditioned variants:

\[
I(U_d;x_{d+1}\mid x_d,\kappa_{+,d}),
\]

\[
I(U_d;x_{d+1}\mid x_d,\kappa_{-,d}),
\]

\[
I(U_d;x_{d+1}\mid x_d,\kappa_{+,d},\kappa_{-,d}).
\]

Keep the existing null engine and support diagnostics beside every estimate.

## 7.4 New action-to-epistemic information

Because the controller now injects facts, also estimate:

\[
I(U_d;\kappa_{+,d+1}\mid S_d),
\qquad
I(U_d;\kappa_{-,d+1}\mid S_d),
\]

using practical coarse conditioning states compatible with the existing estimator infrastructure.

For a controller with target `c`, also define the aligned coverage

```text
kappa_ctrl = kappa_+ if c=+1 else kappa_-
```

and estimate

\[
I(U_d;\kappa_{{\rm ctrl},d+1}\mid S_d).
\]

This directly measures whether the controller action is expressed in the epistemic state, not only in votes.

## 7.5 Signed response / susceptibility

Continue the vote response

\[
\chi_x(S)
=E[x_{d+1}\mid S,U=1]-E[x_{d+1}\mid S,U=0].
\]

Add epistemic responses:

\[
\chi_{\kappa_+}(S),\qquad
\chi_{\kappa_-}(S),
\]

and target-aligned epistemic response

\[
\chi_{\kappa_{\rm ctrl}}(S).
\]

Where possible, calculate two versions:

1. **observational / state-matched estimator** using the existing analysis engine;
2. **paired counterfactual ground truth** using cloned synthetic states with forced `U=0` and `U=1` and matched downstream random-number streams.

The second is a synthetic-model validation oracle and should not replace the estimator used for the LLM study.

## 7.6 Information-response efficiency

Use the existing `T_pi` and susceptibility pipeline to evaluate

\[
\eta_{\rm IR}(S)
=\frac{2a(S)[1-a(S)]\chi_x(S)^2}{T_\pi(S)}.
\]

Report:

- numerator;
- denominator;
- action propensity `a(S)`;
- `chi`;
- `T_pi`;
- `eta_IR`;
- support diagnostics;
- null results.

Do not hide finite-sample violations of the bound. The theoretical bound applies to the exact distributions; plug-in estimators may violate it because of finite data. Flag such cases and show uncertainty.

---

# 8. Theory-validation suite

Implement validation as a separate analysis package / command, not as ad-hoc plotting inside the simulator.

The validation hierarchy should proceed from microscopic to macroscopic.

## Validation 1 — exact semantic invariants

Unit tests must establish:

- persistence is applied exactly once per agent/fact per day;
- no same-day sampling of newly generated peer posts;
- `vote=+` never posts a `-` fact;
- `vote=-` never posts a `+` fact;
- factless post occurs only when the chosen vote has no aligned active fact;
- controller messages always have `vote=c` and `fact_sign=c` in v3;
- controller facts reach agents only through sampled messages;
- `U=0` creates zero controller messages;
- `U=1` creates exactly `b` controller messages;
- sensor samples exactly `q_c` peer-board messages when `q_c <= N`;
- all random choices are seed-reproducible.

## Validation 2 — one-step transition kernel `T_{a->a'}`

For selected starting classes `a=(r,s,v)` and fixed synthetic boards `B`:

1. clone the same starting configuration many times;
2. perform exactly one focal update;
3. estimate

\[
\widehat T_{a\to a'}(B);
\]

4. compute the analytic v3 transition kernel;
5. compare probabilities with Monte Carlo binomial uncertainty.

Test several representative boards:

```text
balanced board
truth-heavy board
false-heavy board
fact-rich board
fact-poor board
controller-like board
```

Deliverable: table + residual plot `T_empirical - T_theory`.

## Validation 3 — emission kernel `R`

For the same fixed states/boards, compare empirical emitted message-category probabilities to analytic

\[
R_{\nu\ell\mid a}(B).
\]

Cross-sign categories should be exactly zero by construction.

Deliverable: empirical-vs-theoretical probability plot.

## Validation 4 — system-size drift and diffusion

Choose a fixed macroscopic state `(z,B)` and generate many short independent trajectories.

For a short interval `Delta tau`, estimate

\[
\widehat A
=\frac{E[\Delta z]}{\Delta\tau},
\]

and

\[
\widehat D
=\frac{N\,\mathrm{Cov}(\Delta z)}{\Delta\tau}.
\]

Compare to

\[
A(z,B)
=\sum_{a,a'}z_aT_{a\to a'}(B)\nu_{a'a},
\]

\[
D(z,B)
=\sum_{a,a'}z_aT_{a\to a'}(B)
\nu_{a'a}\nu_{a'a}^{\mathsf T}.
\]

Do not require every matrix element to be well-estimated; report support / transition counts and focus on occupied/reachable classes.

Deliverable: drift correlation plot, diffusion correlation plot, residual summaries.

## Validation 5 — `N^{-1/2}` fluctuation scaling

Repeat comparable simulations for e.g.

```yaml
N: [24, 48, 96, 192]
```

while holding macroscopic initial conditions fixed.

For chosen observables such as `x`, `kappa_+`, and `kappa_-`, test

\[
\mathrm{Std}\propto N^{-1/2}.
\]

Fit

```text
log Std = intercept + slope * log N
```

and report the slope with uncertainty. The stochastic mean-field prediction is approximately `-0.5` in regimes where the expansion is valid.

Deliverable: log-log fluctuation-scaling figure.

## Validation 6 — stochastic board field

Condition on / reproduce comparable daily emission laws and test

\[
C_d\sim\mathrm{Multinomial}(N,\bar R_d).
\]

Compare:

\[
E[C_d/N]
\]

with `R_bar`, and empirical covariance with

\[
\frac1N[\mathrm{diag}(\bar R)-\bar R\bar R^T].
\]

Deliverable: board mean/covariance comparison.

## Validation 7 — evidence acquisition / epidemic approximation

For agents missing a specific fact of direction `+` or `-`, measure one-sample acquisition probability as a function of board fact composition.

Compare to the exchangeability approximation

\[
\alpha_\pm(B)
=1-\left(1-\frac{c_\pm(B)}{F_\pm}\right)^q.
\]

Run both uncontrolled and controlled boards so the controller-induced increase in target-aligned infection pressure is visible.

Deliverable: empirical versus theoretical acquisition probability.

## Validation 8 — reduced epistemic closure

The reduced theory assumes approximately

\[
P(r,s)
\approx
\mathrm{Bin}(F_+,\kappa_+)\,
\mathrm{Bin}(F_-,\kappa_-).
\]

From microscopic simulations, compare the empirical distribution `P_sim(r,s)` to this closure.

Report at least:

- total variation distance;
- KL divergence when supported;
- moment errors;
- where in `(rho,b,beta_E,beta_S)` the closure breaks down.

This validation distinguishes failure of the low-dimensional closure from failure of the full heterogeneous theory.

## Validation 9 — paired counterfactual controller response

At sampled end-of-day states `S_d`:

1. clone the state;
2. branch one copy with forced `U=0`;
3. branch the other with forced `U=1`;
4. use matched downstream RNG streams wherever possible;
5. repeat enough times to estimate the action-conditioned next-state laws.

Measure:

\[
\chi_x,
\qquad
\chi_{\kappa_+},
\qquad
\chi_{\kappa_-},
\qquad
\chi_{\kappa_{\rm ctrl}}.
\]

Compare these paired ground-truth responses to the observational/state-matched estimators already used in the analysis pipeline.

Deliverable: estimator-vs-ground-truth calibration plots.

## Validation 10 — `T_pi` and Pinsker bound

At a fixed coarse state `S`, use the forced-action branching above to estimate

\[
Q_0(x'\mid S),\qquad Q_1(x'\mid S).
\]

For action propensity `a`, calculate

\[
\bar Q=(1-a)Q_0+aQ_1,
\]

\[
T_\pi(S)
=(1-a)D_{KL}(Q_0\Vert\bar Q)
+aD_{KL}(Q_1\Vert\bar Q),
\]

and

\[
\chi_x(S)=E_{Q_1}[x']-E_{Q_0}[x'].
\]

Then test

\[
T_\pi(S)\ge 2a(1-a)\chi_x(S)^2.
\]

Also compare the exact/forced-action `T_pi` with the ordinary estimator + existing null model at finite episode counts.

Deliverable: bound diagram, `T_pi` vs Pinsker numerator, `eta_IR` phase map, finite-sample convergence.

## Validation 11 — sensing MI

Because the sensor kernel is known, compute the exact/reference

\[
I(n_d;Y_d)
\]

from the empirical distribution of `n_d` and the hypergeometric channel.

Compare with the normal estimator and existing sensing null pipeline at sample sizes such as

```yaml
n_episodes: [10, 20, 30, 50, 75, 100, 200]
```

Deliverable: estimator bias / variance / null separation versus sample size.

---

# 9. Reuse existing null and bootstrap machinery

The codebase already has substantial work for null models, matrix permutations / resampling, bootstrap intervals, support diagnostics, and parallel analysis.

The implementation rule is:

> **Extend the canonical data inputs and estimator recipes; do not create a second statistics stack.**

Specifically:

- feed `x`, `kappa_+`, `kappa_-`, target-aligned coverage, `Y`, `U`, and next-state values into the existing estimator interface;
- reuse the current action null / policy-resampling semantics for action-to-population CMI;
- reuse the current sensing null for sensing MI;
- reuse episode-level bootstrap where the existing engine already treats episodes as the independent unit;
- reuse current support diagnostics for dual-action occupancy and sparse conditioning states;
- keep raw null draws / summaries in the same format already used by Santa Fe / MA-CC analysis;
- preserve exact parallel/sequential reproducibility tests that already exist.

If the existing engine cannot directly accept a new composite conditioning state such as `(x_bin, kappa_plus_bin, kappa_minus_bin)`, add a state-encoding adapter. Do not duplicate the estimator or permutation implementation.

---

# 10. Suggested canonical output tables

Keep existing table conventions where possible. Add fields rather than inventing an unrelated analysis package.

Suggested outputs:

```text
rounds.parquet
micro_slots.parquet
agent_states.parquet              # optional if compact enough
board_messages.parquet            # peer + controller source attribution
occupancy_states.parquet          # Z_rsv / z_rsv
information_estimates.parquet
support_diagnostics.parquet
response_estimates.parquet
mean_field_validation.parquet
transition_kernel_validation.parquet
board_field_validation.parquet
closure_validation.parquet
paired_counterfactuals.parquet
path_probability_terms.parquet    # optional compact event log / summaries
```

If the repository already has equivalent tables, extend them rather than creating duplicates.

---

# 11. Minimum plots for the v3 validation report

Reuse current plotting infrastructure where possible.

Required figures:

1. `T_empirical` vs `T_theory` for one-step transitions.
2. `R_empirical` vs `R_theory` for peer message emissions.
3. empirical drift vs theoretical drift.
4. empirical diffusion entries vs theoretical diffusion entries.
5. log-log finite-size fluctuation scaling with reference slope `-1/2`.
6. board mean/covariance validation.
7. empirical vs theoretical fact-acquisition probability.
8. closure error across `(rho,b)`.
9. paired ground-truth susceptibility vs ordinary estimator.
10. `T_pi` vs Pinsker numerator with the `y=x` boundary line.
11. `eta_IR` phase map.
12. sensing-MI estimator vs exact/reference value as sample size increases.
13. final vote response phase diagram across `(rho,b)`.
14. final epistemic response phase diagram across `(rho,b)`.

---

# 12. Initial scientific parameter set

Do not optimize parameters while implementing. First reproduce a fixed reference configuration.

Suggested baseline:

```yaml
N: 24
F: 10
F_plus: 7
F_minus: 3
q: 3
q_c: 12
rounds: 30
rho: 0.75
beta_E: 1.0
beta_S: 1.3
controller_target: -1
controller_fact_selection: uniform_with_replacement
persistence_clock: round_boundary
peer_posting_mode: vote_aligned_fact
controller_message_mode: target_aligned_fact
board_clock: frozen_front_page
```

Budget sweep:

```yaml
b: [0, 3, 6, 9, 12, 15, 18, 21, 24]
```

Primary persistence comparison:

```yaml
rho: [0.75, 1.0]
```

For theory-regime checks retain the existing beta pairs:

```text
(0.75, 1.0)
(1.00, 1.3)   # primary balanced baseline
(1.25, 1.6)
(2.00, 0.5)   # evidence dominated
(0.50, 2.0)   # social dominated
```

Do not run the full grid until the one-step and drift/diffusion validations pass.

---

# 13. Implementation sequence

## Phase A — semantics first

Implement only:

- global boundary persistence;
- frozen board buffers;
- vote-aligned peer fact posting;
- target-aligned controller fact posting;
- exact retained micro events;
- deterministic invariants / unit tests.

Run tiny smoke tests.

**Do not start the large sweep yet.**

## Phase B — theory-state logging

Add:

- `(r,s,v)` class tracking;
- `Z_rsv`, `z_rsv`;
- `x`, `kappa_+`, `kappa_-`;
- `R`, `R_bar`, `C`, controlled `B` summaries;
- primitive event probabilities.

Verify retained records reconstruct the episode exactly.

## Phase C — microscopic theory validation

Implement validations 2–7:

- transition kernel;
- emission kernel;
- drift;
- diffusion;
- finite-size scaling;
- board multinomial field;
- acquisition probability.

Only after these work should the reduced closure be treated as scientifically meaningful.

## Phase D — reuse information-analysis pipeline

Wire new variables into the existing estimator/null/bootstrap framework.

Add:

- epistemically conditioned CMI;
- action-to-epistemic CMI;
- signed epistemic susceptibility;
- paired counterfactual reference calculations;
- exact/reference sensing MI;
- Pinsker validation.

Do not create a new null/permutation subsystem.

## Phase E — reduced closure and phase diagrams

Run:

- closure validation;
- `(rho,b)` response maps;
- beta-regime comparisons;
- sample-size calibration;
- comparison of microscopic simulator vs reduced SDE trajectories.

---

# 14. Acceptance criteria

The v3 update is complete only when all of the following are true.

### Model semantics

- Historical configs still reproduce old semantics.
- New v3 configs have explicit provenance/version fields.
- Persistence occurs exactly once per day.
- Same-day peer-board leakage is impossible.
- Peer posted facts are always aligned with the realized new vote when such evidence exists.
- Controller posts always carry target-aligned facts.
- Controller facts affect agents only through actual board exposure.

### Theory observability

- Every microscopic `a -> a'` transition can be reconstructed.
- `Z_rsv` / `z_rsv` are reconstructible from saved state.
- Peer board `C_d` and controlled board `B_{d+1}` are separately reconstructible.
- The controller sensor, policy probability, action, and fact identities are retained.
- Primitive forward event probabilities are retained sufficiently for later path-probability work.

### Validation

- Empirical one-step kernels agree with analytic kernels within Monte Carlo uncertainty.
- Empirical `R` agrees with the aligned-emission theory.
- Drift and diffusion agree with the system-size formulas over supported states.
- Finite-size fluctuations show approximately `N^{-1/2}` scaling in the regime where the approximation should hold.
- Board mean/covariance agree with the multinomial approximation where its assumptions hold.
- Reduced-closure error is explicitly measured rather than assumed small.
- Paired counterfactual response agrees with ordinary response estimators as sample size increases.
- Exact/reference `T_pi` satisfies the Pinsker inequality; finite-sample estimator deviations are reported with uncertainty.
- Exact/reference sensing MI is recovered by the empirical estimator as sample size increases.

### Statistics / software

- Existing null-model, permutation/resampling, bootstrap, and support-diagnostic code is reused.
- Parallel and sequential analyses remain reproducible under fixed seeds.
- New tests cover the v3-specific semantics.
- A compact validation report is produced automatically from a completed study directory.

---

# 15. Important scientific distinction to preserve

Keep three layers separate in code and analysis:

```text
MICROSCOPIC SIMULATOR
    exact finite-N stochastic reference process

HETEROGENEOUS STOCHASTIC MEAN FIELD
    z_rsv drift + N^(-1/2) diffusion approximation

REDUCED CLOSURE
    x, kappa_+, kappa_- approximation
```

A failure of the reduced closure does **not** imply failure of the heterogeneous theory. A failure of the heterogeneous theory does **not** automatically imply a simulator bug. The validation package should make it possible to locate the first level at which disagreement appears.

Likewise, distinguish:

```text
controller decision U_d
controller public messages
controller-message exposure
controller-fact acquisition
population response
```

These are separate stages of the causal communication funnel and should remain separately measurable.

---

# 16. Final deliverables

Please implement and return:

1. v3 simulator changes with backward-compatible configuration;
2. new/updated unit and integration tests;
3. expanded trajectory / canonical-state retention;
4. new estimator recipes using the existing analysis/null/bootstrap engine;
5. paired-counterfactual synthetic validation utilities;
6. theory-validation tables and plots;
7. one `README` or handoff report explaining:
   - what changed;
   - exact v3 round clock;
   - configuration knobs;
   - validation results;
   - any theory approximation that fails and where;
   - which parts of the existing statistics stack were reused.

Please do **not** silently tune parameters to make theory and simulation agree. If a validation fails, report the discrepancy first, isolate whether it comes from implementation, finite-size effects, board correlations, or the reduced closure, and only then propose a model/theory change.
