# ASTRA task-003 metrics tutorial

**Scope:** the metrics requested by
[`astra_task003_false_control_30x30`](../../../configs/runs/relational_reasoning/blackboard_game/astra_task003_false_control_30x30/README.md)

**Code state checked:** 8 September 2026

**Main analysis recipe:**
[`analysis.yaml`](../../../configs/runs/relational_reasoning/blackboard_game/astra_task003_false_control_30x30/analysis.yaml)

This document explains what the study measures, how each value is calculated,
and how to read the output. It is a tutorial and a status record for the
current implementation.

The short version is:

- The study measures what the controller sensed, whether it acted, how the
  population changed, and how its public messages moved through the board.
- CMI (conditional mutual information) measures predictive information after
  accounting for the current state.
- $\eta_{IR}$ (information-response efficiency) compares a response-based
  information bound with target CMI.
- $\eta_{th}$ (thermodynamic efficiency) compares directed control expenditure
  with directed control expenditure plus sensing information.
- The new Phase 2 metrics estimate a causal target-share response at lags one
  through three and divide that response by several observed communication
  costs.
- Every estimate is calculated separately inside each scientific cell. A
  **scientific cell** is one fixed persistence-and-budget combination.

There is no current repository metric literally named `ETS`. If “ETS” means the
eta values, they are $\eta_{IR}$ and $\eta_{th}$ below. If it means “empirical
transfer susceptibility,” the repository keeps the two parts separate:
`round_target_actuation_cmi` is transfer information and
`round_target_susceptibility` is signed response.

## Contents

1. [The study being measured](#1-the-study-being-measured)
2. [Where and when metrics are calculated](#2-where-and-when-metrics-are-calculated)
3. [The variables used in the formulas](#3-the-variables-used-in-the-formulas)
4. [Metric map](#4-metric-map)
5. [Sensing metrics](#5-sensing-metrics)
6. [Controller variation and support](#6-controller-variation-and-support)
7. [All CMI metrics](#7-all-cmi-metrics)
8. [Signed response and susceptibility](#8-signed-response-and-susceptibility)
9. [Eta and single-affinity metrics](#9-eta-and-single-affinity-metrics)
10. [New Phase 2 causal and communication metrics](#10-new-phase-2-causal-and-communication-metrics)
11. [Blackboard mechanism summaries](#11-blackboard-mechanism-summaries)
12. [State-local and persistence summaries](#12-state-local-and-persistence-summaries)
13. [Uncertainty, nulls, and support](#13-uncertainty-nulls-and-support)
14. [Output files and a reading workflow](#14-output-files-and-a-reading-workflow)
15. [Important limits](#15-important-limits)
16. [Implementation map](#16-implementation-map)

---

## 1. The study being measured

The study contains one **false-control arm**. This means the controller tries
to increase support for an answer that is not the correct answer.

| Setting | Current value |
|---|---:|
| Task | MuSR Team Allocation `task_003`, candidate 130 |
| Correct answer | `ALLOCATION_0` |
| Controller target | `ALLOCATION_2` |
| Participant and controller model | DeepInfra `deepseek-ai/DeepSeek-V4-Flash` |
| Population size, $N$ | 24 agents |
| Population rounds | 30 per episode |
| Repetitions | 30 per cell |
| Persistence, $\rho$ | 0.70, 0.775, 0.85, 0.925, 1.00 |
| Intervention budget, $b$ | 3, 6, 9, 12 |
| Scientific cells | $5\times4=20$ |
| Planned episodes | $20\times30=600$ |
| Maximum round records | $600\times30=18{,}000$ |
| Controller sensor sample, $q_c$ | 12 agents |
| Social sample, $q$ | up to 3 board messages per update |

At the start of a round, the controller samples 12 votes. It then draws a
binary action from a coded probability:

$$
P(U_k=1\mid Y_k)=\sigma\!\left(4(0.5-p_{Z,k}^{sample})\right).
$$

Here $\sigma$ is the logistic function, $Y_k$ is the sampled vote vector, and
$p_{Z,k}^{sample}$ is the sampled share supporting the false target. The action
is:

- $U_k=0$: remain silent;
- $U_k=1$: act at dawn.

Only after $U_k=1$ does the controller language model choose `REPORT`,
`REQUEST`, or `DIRECTIVE`. That communication choice does not replace the
binary action used by the CMI and causal estimators.

## 2. Where and when metrics are calculated

Runtime analysis is intentionally disabled in `false_control_llm.yaml`:

```yaml
analysis: {enabled: false, estimators: []}
```

The study first retains round and microscopic-update records. Later,
`mas-cc study aggregate` reads those records and performs the analysis without
making provider calls.

```text
completed episodes
  -> canonical rounds and micro-slots
  -> one estimator run per scientific cell
  -> uncertainty and support checks
  -> derived eta values and communication metrics
  -> Parquet tables, reports, and plots
```

**Canonical** means the one authoritative copy selected for analysis. Only
`completed` and `skipped_resumed` episodes contribute. Failed attempts are
excluded. If retry records duplicate one physical round or microscopic update,
the last copy is retained.

The primary observation tables are:

- `cells.parquet`: one row per scientific cell;
- `episodes.parquet`: one row per episode;
- `rounds.parquet`: one row per population round;
- `micro_slots.parquet`: one row per individual update within a round.

The estimator never combines different budgets or persistence values into one
CMI. Each of the 20 cells is analyzed separately.

## 3. The variables used in the formulas

| Symbol | Plain meaning |
|---|---|
| $k$ | population-round index |
| $N=24$ | number of agents |
| $K=3$ | number of possible answers |
| $U_k$ | binary controller action: act or remain silent |
| $e_k$ | logged probability that $U_k=1$ from the controller's visible information |
| $N_k$ | complete three-answer population count vector before round $k$ |
| $Y_k$ | complete three-answer sensor count vector from the 12 sampled agents |
| $n_{Z,k}$ | number of agents supporting controller target $Z$ before round $k$ |
| $n_{T,k}$ | number of agents supporting the true answer before round $k$ |
| $x_k=n_{Z,k}/N$ | fraction supporting the controller target |
| $E_k$ | exact histogram of how many agents hold 0, 1, … supporting facts |
| $\phi_k$ | fraction of agents holding the complete proof |
| $s_k=1-\phi_k$ | fraction still socially susceptible because they lack the complete proof |
| $\kappa_k$ | mean fraction of supporting facts currently held by agents |
| $\Delta x_k$ | target-share change, $(n_{Z,k+1}-n_{Z,k})/N$ |

For this false-control study, “target-directed” means movement toward
`ALLOCATION_2`, not movement toward truth. Target and truth metrics must
therefore be read separately.

## 4. Metric map

The current `analysis.yaml` requests 23 primary estimators, four derived names,
eight-bin state-local analysis, blackboard summaries, and Phase 2 outputs.
Naming one single-affinity derived metric activates its complete coupled family,
so related intermediate values are emitted together.

| Question | Main metrics |
|---|---|
| Did the sensor describe the population? | `round_sensing_mi`, `round_sensor_mae`, `round_sensor_mse` |
| Did the controller vary its action? | action entropies and overlap diagnostics |
| Did action predict the next state? | the seven configured actuation CMI metrics |
| Which direction did the population move? | susceptibility and signed response metrics |
| How much available action information reached the target state? | `round_target_information_fraction`, $\eta_{IR}$ |
| What sensing and directed-control cost was used? | scalar sensing information, controlled current, affinity, $\eta_{th}$ |
| What was the randomized intervention's causal response? | `propensity_weighted_causal_response` at lags 1–3 |
| How much public communication produced that response? | communication funnel, response per cost, response-cost frontier |
| What happened on the board? | `blackboard_diagnostics.parquet` |

## 5. Sensing metrics

### 5.1 `round_sensing_mi`

This is the empirical full-vector mutual information:

$$
I(N_k;Y_k).
$$

**Mutual information (MI)** measures how much knowing one variable reduces
uncertainty about another. Here it asks how much the sampled 12-agent vote
vector tells us about the full 24-agent vote vector.

It is calculated from observed discrete counts with base-two logarithms, so its
unit is **bits**.

### 5.2 `round_sensor_mae`

For every eligible round, define the sensor error

$$
\epsilon_k=
\text{sensor target share}_k-\frac{n_{Z,k}}{N}.
$$

The mean absolute error (MAE) is

$$
\operatorname{MAE}=\frac{1}{R}\sum_k |\epsilon_k|.
$$

It answers: on average, how far was the sampled target share from the true
population target share?

### 5.3 `round_sensor_mse`

The mean squared error (MSE) is

$$
\operatorname{MSE}=\frac{1}{R}\sum_k \epsilon_k^2.
$$

Squaring gives more weight to large sensor errors.

### 5.4 `target_sensing_information_nats`

This is a different sensing quantity:

$$
I(n_{Z,k};Y_{Z,k}).
$$

It uses only the target count and sampled target count. The sensor mechanism is
known exactly: 12 agents are drawn without replacement from 24. The code uses
the exact hypergeometric sampling law and the observed state occupancy.

Outputs are:

- `target_sensing_information_nats`: mean information per round;
- `target_sensing_information_horizon_nats`: sum over retained rounds.

These use natural logarithms, so the unit is **nats**. Do not compare the number
directly with `round_sensing_mi` without accounting for both the different
channel and the different unit.

## 6. Controller variation and support

A controller that always acts, or always remains silent, provides no empirical
comparison between actions. The following values show whether a CMI or response
has enough action variation to interpret.

### 6.1 Action entropy

`round_controller_action_entropy` is

$$
H(U_k).
$$

It measures overall action variation in bits.

`round_controller_action_entropy_given_population` is

$$
H(U_k\mid N_k).
$$

It measures the action variation left after the complete current population
state is known.

### 6.2 Overlap diagnostics

A **conditioning state** is one value of the state held fixed by a CMI. For
example, target CMI conditions on $n_{Z,k}$.

| Metric | Meaning |
|---|---|
| `round_conditioning_state_count` | Number of occupied conditioning states |
| `round_dual_action_state_fraction` | Fraction of occupied states in which both act and silence occurred |
| `round_dual_action_event_fraction` | Fraction of round observations lying in those dual-action states |
| `round_single_action_slice_fraction` | Fraction of occupied states with only one observed action |
| `round_singleton_fraction` | Fraction of observations in states seen only once |

The support label is:

- `unsupported` when fewer than two actions were observed or no conditioning
  state contains both actions;
- `limited` when the dual-action state fraction is below 0.25 or the singleton
  fraction exceeds 0.5;
- `adequate` otherwise.

These thresholds are warnings about the observed table. They do not prove that
an `adequate` estimate is exact.

## 7. All CMI metrics

### 7.1 What CMI means

**Conditional mutual information (CMI)** measures dependence between two
variables after accounting for a third. For discrete variables:

$$
I(X;Y\mid Z)
=H(X,Z)+H(Y,Z)-H(Z)-H(X,Y,Z).
$$

The code builds a three-dimensional count table with axes $X,Z,Y$ and inserts
the observed frequencies. Its headline estimate is the unsmoothed plug-in
value in bits.

Every CMI below uses the same implementation. Only its outcome or conditioning
state changes.

### 7.2 Configured CMI family

| Metric | Formula | Plain question |
|---|---|---|
| `round_population_actuation_cmi` | $I(U_k;N_{k+1}\mid N_k)$ | Does action predict the complete next vote distribution after the complete current distribution is known? |
| `round_target_actuation_cmi` | $I(U_k;n_{Z,k+1}\mid n_{Z,k})$ | Does action predict the next false-target count after the current false-target count is known? |
| `round_truth_actuation_cmi` | $I(U_k;n_{T,k+1}\mid n_{T,k})$ | Does action predict the next truth count after the current truth count is known? |
| `round_memory_target_actuation_cmi` | $I(U_k;n_{Z,k+1}\mid n_{Z,k},E_k)$ | Does the target relationship remain after exact evidence memory is also held fixed? |
| `round_epistemic_target_actuation_cmi` | $I(U_k;n_{Z,k+1}\mid n_{Z,k},B_\kappa,B_\phi)$ | Does it remain after a coarse joint evidence state is held fixed? |
| `round_phi_target_actuation_cmi` | $I(U_k;n_{Z,k+1}\mid n_{Z,k},B_\phi)$ | Does it remain within low, medium, and high complete-proof groups? |
| `round_kappa_target_actuation_cmi` | $I(U_k;n_{Z,k+1}\mid n_{Z,k},B_\kappa)$ | Does it remain within low, medium, and high mean evidence-coverage groups? |

The scalar $\phi$ and $\kappa$ bins are:

- low: $[0,1/3)$;
- medium: $[1/3,2/3)$;
- high: $[2/3,1]$.

The joint epistemic metric uses four bins per axis. The exact-memory metric can
be much sparser because $E_k$ is a whole histogram rather than one or two bin
labels.

### 7.3 How direct counting works: a small example

Suppose the current target count is 8. Among all observed rounds starting at 8,
we count how often each action leads to each next target count:

| Action | next 7 | next 8 | next 9 |
|---|---:|---:|---:|
| silence | 5 | 8 | 2 |
| act | 1 | 6 | 9 |

The estimator converts these counts into probabilities and measures how
different the next-state distributions are between the two actions. It repeats
this for every current target count and weights by how often each count was
visited. The result is `round_target_actuation_cmi`.

A large value says that action labels help predict the next target count. It
does not say whether acting increased or decreased target support. Signed
response supplies that direction.

### 7.4 Estimator variants

Each information estimate retains three variants:

- `unsmoothed`: direct observed frequencies; this is the headline value;
- `jeffreys`: adds 0.5 to every complete contingency-table cell;
- `miller_madow`: applies the Miller–Madow finite-sample entropy correction.

The variants are sensitivity checks. They can disagree when the table is
sparse. Read the headline estimate with the null and support diagnostics rather
than choosing whichever variant is largest.

### 7.5 `round_target_information_fraction`

Target CMI cannot exceed the action variation available under the same current
target count. The normalized fraction is

$$
\text{target information fraction}
=
\frac{I(U_k;n_{Z,k+1}\mid n_{Z,k})}
     {H(U_k\mid n_{Z,k})}.
$$

The result is missing when the denominator is zero or too close to zero. A
small CMI with a near-zero action-entropy ceiling is structurally expected; a
small CMI with a large ceiling means the available action variation did not
strongly predict the next target count.

## 8. Signed response and susceptibility

CMI has no sign. These metrics say which direction the population moved.

### 8.1 `round_target_susceptibility`

At target count $n$, define

$$
\chi(n)=
E[\Delta x_k\mid U_k=1,n_{Z,k}=n]
-
E[\Delta x_k\mid U_k=0,n_{Z,k}=n].
$$

Only states containing both actions identify $\chi(n)$. The reported scalar
combines those state differences using their observed numbers of rounds.

- Positive: acting is associated with more movement toward the false target
  than silence at the same starting target count.
- Negative: acting is associated with less movement toward the false target.
- Near zero: little state-matched separation was observed.

Its unit is target-fraction change per population round.

### 8.2 `round_target_signed_response_share`

This compares the two action groups without matching the starting target count:

$$
E[\Delta x_k\mid U_k=1]-E[\Delta x_k\mid U_k=0].
$$

It is easy to read, but it can mix together different starting-state
compositions. Susceptibility is the preferred state-matched response.

### 8.3 Magnetization responses

For $K$ answers, aligned magnetization is

$$
m=\frac{Kx-1}{K-1}.
$$

The configured metrics are:

- `round_target_signed_actuation`: state-matched change toward the false target;
- `round_truth_signed_actuation`: state-matched change toward truth.

Their unit is aligned-magnetization change per round. With $K=3$,
$\Delta m=\tfrac32\Delta x$. These values must not be substituted for the
share-unit susceptibility in $\eta_{IR}$.

## 9. Eta and single-affinity metrics

**Single affinity** is the repository's reduced target-versus-non-target model.
The related metrics share one state response, one action weighting, and one
whole-episode bootstrap. Requesting any member of this family causes the
complete family to be calculated so the pieces cannot silently come from
incompatible resamples.

### 9.1 $\eta_{IR}$: information-response efficiency

At target count $n$, let

$$
a(n)=P(U_k=1\mid n_{Z,k}=n)
$$

be the observed action frequency. The response-based Pinsker lower-bound term
is

$$
B_{IR}(n)=\frac{2a(n)[1-a(n)]\chi(n)^2}{\ln 2}.
$$

The headline efficiency is a ratio of sums:

$$
\eta_{IR}
=
\frac{\sum_n \hat p(n)B_{IR}(n)}
     {I(U_k;n_{Z,k+1}\mid n_{Z,k})}.
$$

It is not an average of state-local ratios. Its output includes:

| Metric | Meaning |
|---|---|
| `eta_ir` | Headline dimensionless ratio |
| `eta_ir_pinsker_numerator_bits` | Occupancy-weighted response bound |
| `eta_ir_denominator_T_bits` | Target actuation CMI used below the line |
| `susceptibility_occupancy_weighted` | One-number summary of identified $\chi(n)$ values |
| `eta_ir_state_local` | Ratio within each configured target-state bin |

The output also records identified occupancy mass. A result describing only a
small part of the visited state space must be treated as limited even if its
number is finite.

### 9.2 Effective affinity and kinetic compliance

These values use controlled microscopic updates inside advocating rounds.
Define

$$
p_+=
\frac{\#(\text{non-target}\rightarrow\text{target})}
     {\#(\text{eligible non-target updates})},
$$

$$
p_-=
\frac{\#(\text{target}\rightarrow\text{non-target})}
     {\#(\text{eligible target updates})}.
$$

Then

$$
h_{eff}=\ln\frac{p_+}{p_-},
\qquad
\gamma_{eff}=p_++p_-.
$$

- `effective_affinity`, $h_{eff}$, is the directional log ratio in nats.
- `kinetic_compliance`, $\gamma_{eff}$, is total controlled transition activity.

The calculation is deliberately unsmoothed. If either direction lacks
eligibility or never occurs, the affinity is invalid instead of inventing a
finite value. In this dawn-only board design, microscopic updates may not be
marked as individually controlled, so these values and $\eta_{th}$ may be
unsupported.

### 9.3 Controlled current

For each round,

$$
J_{c,k}=N\sum_n p_k(n)a(n)\chi(n).
$$

This estimates target-count movement associated with the controller's response
model. It is not the episode's final target count minus its initial target
count; ordinary social dynamics also contribute to that raw endpoint change.

Outputs are:

- `controlled_current`: mean target-count current per round;
- `controlled_current_horizon`: sum over the retained horizon.

### 9.4 $\eta_{th}$: thermodynamic efficiency

Let $I_{sens}$ be the horizon scalar sensing information in nats. Then

$$
C_{th}=h_{eff}J_c+I_{sens},
$$

$$
\eta_{th,signed}
=
\frac{h_{eff}J_c}{h_{eff}J_c+I_{sens}}.
$$

The output family is:

| Metric | Meaning |
|---|---|
| `affinity_weighted_current_nats` | $h_{eff}J_c$ |
| `thermodynamic_control_expenditure_nats` | $h_{eff}J_c+I_{sens}$ |
| `eta_th_signed` | Signed ratio whenever numerically defined |
| `eta_th` | Ratio only when the bounded interpretation is valid |
| `eta_th_bounded` | Same bounded value, kept explicitly for auditing |

The bounded result requires target-directed affinity-weighted current, a
positive denominator, and a ratio in $[0,1]$. Invalid cases remain missing and
carry a reason such as `insufficient_support_for_h_calibration`.

## 10. New Phase 2 causal and communication metrics

These are the main newly tracked metrics in the current study recipe.

### 10.1 Propensity-weighted causal response

A **propensity** is the logged probability $e_k$ that the randomized policy
acts. For lag $h\in\{1,2,3\}$, each eligible round contributes

$$
\psi_{k,h}
=
\left[
\frac{U_k}{e_k}-\frac{1-U_k}{1-e_k}
\right]
(x_{k+h}-x_k).
$$

The cell estimate is the mean of these contributions. This is a
Horvitz–Thompson estimate, meaning each observed action is weighted by the
inverse of its assignment probability so the randomized act and silence arms
represent the policy population.

The calculation requires:

- $U_k$ exactly 0 or 1;
- $0<e_k<1$;
- unique round identities;
- target shares in $[0,1]$;
- no lag lookup across episode boundaries;
- a complete episode for inclusion in the cell effect.

The output is `causal_response_effects.parquet`, with one row per cell and lag.
A positive value means that randomized activation increased false-target share
relative to randomized silence. Because this is a false-control arm, that is
movement away from truth.

`causal_response_support.parquet` records action and silence counts, missing
lags, incomplete episodes, initialization blocks, and propensity quantiles.
Near-zero or near-one propensities produce large weights and can make estimates
unstable even when both actions occurred.

### 10.2 Available causal susceptibility

Available causal susceptibility is an **offline derived analysis**, meaning it
is calculated from saved canonical rounds and does not rerun the simulation or
call a model provider. For the immediate response, it divides each round's
propensity-weighted causal contribution by the population share that was still
available to move toward the controller target before treatment:

$$
\psi^{\mathrm{avail}}_{k,1}
=
\frac{\psi_{k,1}}{1-x_k},
\qquad x_k<1.
$$

Here $1-x_k$ is pre-intervention available target-convertible population mass.
The state-local table, `available_causal_susceptibility_state_local.parquet`,
averages these exact round-level values within the existing eight $x$ bins. It
does not divide by a bin center.

At $x_k=1$, no population mass remains available. The value is therefore
undefined, remains missing (`NaN`), and is counted in
`n_saturated_excluded`. It is never set to zero or clipped.

The preferred one-number-per-cell result is in
`available_causal_susceptibility_summary.parquet`:

$$
\chi_{\mathrm{avail}}^{\mathrm{mass}}
=
\frac{\sum_k \psi_{k,1}}{\sum_k(1-x_k)}.
$$

This ratio of sums gives less leverage to individual nearly saturated rounds
than the average of $\psi_{k,1}/(1-x_k)$. The numerator and denominator are
recalculated together inside every shared-initialization-block bootstrap draw.
This is a susceptibility normalization. It does not redefine
`round_target_susceptibility`, $T_\pi$, $\eta_{IR}$, or thermodynamic
efficiency.

### 10.3 Communication funnel

`communication_funnel.parquet` follows each randomized round through:

```text
binary assignment
  -> actual controller posts
  -> message exposures
  -> distinct readers within that round
  -> new controller-fact acquisitions
  -> controller-fact reactivations
  -> target response at lags 1, 2, and 3
```

The cost fields are:

1. `actual_posts`;
2. `exposures`;
3. `unique_readers_per_round`;
4. `new_controller_facts`;
5. `reactivated_controller_facts`.

When detailed microscopic records are present, their message, reader, new-fact,
and reactivated-fact identities audit the round totals. A disagreement causes
aggregation to fail rather than silently accepting two versions.

### 10.4 Expected activation cost

For a communication cost $C_k$, the estimated expected cost under activation
is

$$
\widehat C
=
\frac{1}{R}\sum_k\frac{U_k C_k}{e_k}.
$$

This calculation is repeated for each of the five cost fields above.

### 10.5 Operational response per cost

`communication_efficiency.parquet` reports

$$
\frac{\widehat\psi_h}{\widehat C}
$$

for each cell, lag, and communication cost. Both response and cost are
recalculated inside each bootstrap draw before their ratio is formed.

If expected cost is zero, the ratio is missing and `zero_denominator=true`. It
is never changed to zero.

This is an operational communication-efficiency measure. It is not
$\eta_{IR}$ and not thermodynamic $\eta_{th}$.

### 10.6 Response-cost frontier

`response_cost_frontier.parquet` sorts supported cells by observed expected
cost. A point is on the frontier when its causal response is larger than every
supported response observed at an equal or lower cost.

The frontier describes the best observed trade-off in this study. It is not an
optimization proof and does not predict untested budgets.

### 10.7 Communication mode summary

`communication_mode_descriptive_response.parquet` reports mean observed
response after `REPORT`, `REQUEST`, or `DIRECTIVE` choices.

This table is explicitly **descriptive, not causal**. Mode is chosen after the
randomized gate acts. Conditioning a causal estimate on this later choice
would break the original randomized comparison, so the code rejects realized
mode and realized communication costs as causal strata.

## 11. Blackboard mechanism summaries

`blackboard_diagnostics.parquet` gives one row per cell. It includes:

- rounds and episodes;
- controller posts and reports;
- controller-message exposures;
- the sum of per-round unique readers;
- requests and directives;
- new evidence acquisitions;
- peer and controller fact reactivations;
- active and historical evidence coverage;
- target adoptions after controller reports;
- realized exposure fractions;
- counts and fractions of `REPORT`, `REQUEST`, and `DIRECTIVE` choices.

These are mechanism summaries: they show what physically happened on the
board. They are not replacements for CMI or causal response.

A “unique reader” is unique only within one round. Summing this value across
rounds can count the same agent repeatedly.

The blackboard option also writes convenient views:

- `sensing_information.parquet`;
- `transfer_information.parquet`;
- `susceptibility.parquet`;
- `efficiencies.parquet`;
- `state_resolved_x_b.parquet`;
- `rho_b_summary.parquet`.

These are selections or rearrangements of existing estimates, not additional
estimators.

## 12. State-local and persistence summaries

### 12.1 Eight target-share bins

The recipe sets:

```yaml
state_local: [x]
state_local_x_bins: 8
```

A target share $x$ is assigned to

$$
B(x)=\min(\lfloor8x\rfloor,7).
$$

Within each bin the analysis recalculates:

- target actuation CMI;
- target information fraction;
- target signed actuation;
- target susceptibility.

These state-local rows currently have no bootstrap interval and no permutation
null. Their support and occupancy are therefore especially important.

### 12.2 Persistence-aggregated views

`rho_aggregated_descriptive: true` creates summaries across persistence values.
They are arithmetic descriptions of already-computed cell results. They do not
pool raw rounds from different persistence values into a new CMI.

## 13. Uncertainty, nulls, and support

### 13.1 Ordinary metric bootstrap

A **bootstrap** repeatedly resamples observed units and recalculates a value to
show sampling uncertainty.

For ordinary MI, CMI, entropy, and response metrics:

- 1,000 bootstrap draws are requested;
- the confidence level is 95%;
- complete episodes, not individual rounds, are sampled with replacement;
- the interval uses the 2.5th and 97.5th percentiles.

Rounds from one episode are dependent, so they travel together.

### 13.2 CMI null distribution

A **null distribution** shows what values appear after intentionally breaking
the tested relationship while preserving the policy structure.

For each actuation CMI, the code keeps the recorded state and outcome but
redraws action independently from that round's logged policy probability:

$$
U_k^*\sim\operatorname{Bernoulli}(e_k).
$$

It repeats this 1,000 times. The stored one-sided value is

$$
p=\frac{1+\#\{T^*\ge T_{observed}\}}{1000+1}.
$$

The null type is `policy_conditional_randomization`.

For sensing MI, sensor values are shuffled across eligible rows. Its null type
is `sensor_permutation`.

Direct-counting information estimates have positive finite-sample bias. A raw
CMI above zero is therefore not enough; compare it with `null_mean`, `p_value`,
and support.

### 13.3 Causal-response bootstrap

The Phase 2 estimator resamples complete **shared-initialization blocks**. A
block keeps matched episodes or cells with the same starting population
together. This preserves the study's paired initialization during uncertainty
calculation.

The propensity-weighted response has no permutation null. Its evidence comes
from randomized assignment, confidence intervals, propensity diagnostics, and
action-arm support.

## 14. Output files and a reading workflow

After strict aggregation, start with:

1. `analysis/validation.json` — confirm that expected cells and episodes are
   complete and the retained data passed integrity checks.
2. `analysis/tables/causal_response_support.parquet` — check both action arms,
   propensity range, missing lags, and shared-initialization block counts.
3. `analysis/tables/support_diagnostics.parquet` — check overlap and sparsity
   for each CMI.
4. `analysis/tables/primary_estimates.parquet` — read CMI, entropy, sensing, and
   signed-response values with their uncertainty and null summaries.
5. `analysis/tables/derived_observables.parquet` — read $\eta_{IR}$, sensing,
   current, affinity audit fields, and $\eta_{th}$ validity.
6. `analysis/tables/causal_response_effects.parquet` — compare causal response
   across lags, persistence values, and budgets.
7. `analysis/tables/available_causal_susceptibility_summary.parquet` — compare
   the stable immediate response per available target-convertible mass.
8. `analysis/tables/available_causal_susceptibility_state_local.parquet` —
   inspect how that normalization changes across the eight target-share bins.
9. `analysis/tables/communication_efficiency.parquet` — compare response with
   posts, exposures, readers, acquisitions, and reactivations.
10. `analysis/tables/blackboard_diagnostics.parquet` — explain which board
   mechanisms produced those costs.
11. `analysis/tables/response_cost_frontier.parquet` — inspect the supported
   observed response-cost trade-off.

A practical reading sequence for one cell is:

```text
Did both actions occur?
  -> Is target CMI above its randomized null?
  -> Is susceptibility positive or negative?
  -> Does the propensity-weighted causal response agree in direction?
  -> Which board stage used the communication budget?
  -> What response was obtained per post, exposure, reader, or evidence event?
  -> Are eta_IR and eta_th supported, and what exact ingredients produced them?
```

## 15. Important limits

1. **CMI is predictive, not automatically causal.** Use the propensity-weighted
   response for the randomized causal question.
2. **The target is false.** Positive target response means movement toward
   `ALLOCATION_2`, away from the true `ALLOCATION_0`.
3. **Sparse conditioning can inflate finite-sample CMI.** The exact memory CMI
   is particularly vulnerable. Always read its own null and support row.
4. **Different CMI conditionings answer different questions.** A larger
   memory-conditioned value does not automatically mean memory strengthened
   control; the contingency table also became larger and sparser.
5. **No separate control arm exists in this folder.** Randomized activation is
   compared with randomized silence inside the false-target policy. The study
   does not compare false control with a truth-control or no-controller config.
6. **Lag support shrinks near episode end.** The final rounds cannot contribute
   to every future lag. Missing lag counts are recorded.
7. **Communication modes are post-action choices.** Their summaries are not
   causal mode comparisons.
8. **Reader counts are per round.** They are not episode-wide unique reach.
9. **Provider token totals are not public communication cost.** They include
   prompts, participant replies, and controller decision traffic.
10. **New facts and reactivated facts are different.** A reactivated fact was
    seen earlier and became active again.
11. **Operational response per cost is not thermodynamic efficiency.** They
    have different denominators and different meanings.
12. **$\eta_{th}$ may be unavailable.** The microscopic transition data must
    identify both directions of the effective affinity.
13. **State-local rows have no uncertainty intervals in this recipe.** Use them
    as support-aware maps, not as equally precise point estimates.
14. **Persistence-aggregated tables are descriptive.** They are not pooled
    estimators over heterogeneous cells.
15. **No matched q-voter theory is requested.** `theoretical_reference: none`
    is intentional for this finite-persistence blackboard study.

## 16. Implementation map

| Responsibility | Current source |
|---|---|
| Study aggregation and output views | `src/mas_cc/studies/aggregation.py` |
| Canonical completed-record selection | `src/mas_cc/studies/canonical.py` |
| Shared direct-counting MI and CMI | `src/mas_cc/analysis/estimators.py` |
| Round estimator, bootstrap, null, and support | `src/mas_cc/games/hidden_bench/imitation_round_feedback/analysis.py` |
| Blackboard round-record adapter and epistemic bins | `src/mas_cc/games/relational_reasoning/imitation_round_feedback/analysis.py` |
| $\eta_{IR}$, sensing, affinity, current, and $\eta_{th}$ | `src/mas_cc/analysis/single_affinity.py` |
| Propensity response, funnel, costs, and frontier | `src/mas_cc/analysis/causal_response.py` |
| Scientific recipe | `configs/runs/relational_reasoning/blackboard_game/astra_task003_false_control_30x30/analysis.yaml` |
| Scientific and runtime design | `configs/runs/relational_reasoning/blackboard_game/astra_task003_false_control_30x30/false_control_llm.yaml` |

No replacement MI or CMI estimator was added for this study. All CMI values
reuse the repository's established direct-counting implementation.