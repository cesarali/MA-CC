# Relational reasoning — population study 03 (arm G)

## Stochastic-feedback / transfer-information pilot

A **2-cell, 20-episode** pilot. It is the first relational arm run under a
**closed loop**, and it exists to answer one question that studies 01 and 02
structurally could not:

> When the controller's action is a genuine random variable, how many bits does
> it push into the population's opinion channel — and does conditioning on the
> population's *epistemic memory* change that number?

### Why studies 01/02 could not answer it

Both ran `advocacy_schedule: always`. That is the right choice for a
*controllability* sweep — it fixes the intervention and lets the population
vary — but it pins `H(U_k) = 0`. A controller that acts every round carries no
information by construction, and every actuation CMI is identically zero. Arm G
switches to `advocacy_schedule: soft`, so `U_k` is drawn from

```
P(U_k = ADVOCATE | Y_k) = sigmoid[ beta * (threshold - p_Z(Y_k)) ]
```

with `beta = 4`, `threshold = 0.5`. Nothing about that policy is new; it is
`SoftTargetControl.select_action` / `advocacy_probability`, unchanged.

---

## Files

| path | what it is |
|---|---|
| `relational_population_study03_g_stochastic_feedback_pilot.yaml` | **the pilot.** 2 cells × 10 repetitions = 20 episodes, `university` / `gwdg/openai-gpt-oss-120b` |
| `relational_population_study03_g_stochastic_feedback_pilot_mock.yaml` | the same config with `type: mock`, Comet hard-off, `artifact_profile: full`. Provider-free smoke |
| `README.md` | this file |

The two YAMLs differ in exactly five places — provider, pricing mode, Comet,
artifact profile/output dir, and the experiment name. Game, controller, grid,
seed, repetitions and horizon are identical, so the smoke run exercises the
real round-level machinery.

---

## The round-level ordering

```
N_k  ->  Y_k   sensor, q_c = 12 agents sampled without replacement, votes only
     ->  U_k   in {NO_OP, ADVOCATE}, drawn from the soft policy above
     ->  b = 12 controlled update positions  IF  U_k = ADVOCATE, else 0
     ->  N_{k+1}
```

### `q_c` and `b` are not the same resource

They take the same numerical value **in this pilot and only in this pilot**.

| symbol | value | what it is | where it happens |
|---|---|---|---|
| `q_c` | 12 | how many agents the controller **looks at**, once, at the start of the round | `ThresholdTargetControl.interaction_signal`: `rng.sample(state.agents, sensor_sample_size)` |
| `b` | 12 | how many **update positions** are actuated, only on an ADVOCATE round | `runtime.py`: `sample_controlled_positions(N, b, ...) if action == ADVOCATE_TARGET else ()` |

The 12 sensed agents and the 12 controlled positions come from **separate
seeded streams** (`relational-sensor:*` vs `relational-schedule:<round>`) and
are not required to coincide — they are not even in the same index space
(agent identities vs within-round update slots). The implementation is
exchangeable in both and was left exactly as it was. **Do not collapse them
into one draw**: sensing fraction `q_c/N` and control fraction `c = b/N` are
separate axes that this pilot happens to hold at the same value, `0.5`.

---

## Fixed in both cells

```
N = 24      L = 2       r = 6       q = 1       K = 3 options, 4 distractors
rounds = 10             threshold = 0.5         beta = 4
q_c = 12 -> sensing fraction 0.5      b = 12 -> c = b/N = 0.5
advocacy_schedule: soft               message_mode: recommendation_only
target: 2 (pinned, verified incorrect)
social_distrust: true                 initialization: local_vote
stop_on_consensus: false              semantic option shuffling (per call)
artifact_profile: results_only
```

`stop_on_consensus` must stay `false`: a consensus stop would truncate episodes
to different lengths and put the round-indexed estimates on ragged data.

### The target is pinned, not drawn

`control.options.target: 2` is a zero-based index into the task's frozen
`possible_answers`, **not** `random_incorrect`. `random_incorrect` resolves per
*episode* from a seed derived from the **grid cell index**, so two repetitions
of one cell can advocate two different relations — study 01 arm D shows exactly
that happening. An index is seed- and cell-independent.

Verified against the dataset files:

| world | `possible_answers` | correct | pinned target (index 2) |
|---|---|---|---|
| `task_0001` | `[SOUTHWEST, WEST, EAST]` | WEST (index 1) | **EAST** |
| `task_0002` | `[SOUTHEAST, SOUTHWEST, SOUTH]` | SOUTHWEST (index 1) | **SOUTH** |

"Correct sits at index 1" is a **per-task** fact, not a dataset invariant — in
`pop24_L2_r06` the correct option is at index 2 for `task_0006` and
`task_0010`. Re-check before widening `game.options.task_id`.

---

## What gets estimated, and what it is reused from

**No new estimator was written.** The direct-counting MI/CMI, the whole-episode
bootstrap, the policy-conditional null and the action-support diagnostics are
all the existing HiddenBench round-feedback pipeline
(`mas_cc/games/hidden_bench/imitation_round_feedback/analysis.py`), which in
turn calls `mas_cc/analysis/estimators.py`.

| # | quantity | statistic name | status |
|---|---|---|---|
| 1 | action frequencies, `H(U_k)` | `round_controller_action_entropy` + `controller_action_summary.csv` | existing |
| 2 | `H(U_k \| n_Z,k)`, support/overlap | `round_controller_action_entropy_given_population`, `round_dual_action_*`, `round_singleton_fraction`, `round_conditioning_state_count` | existing |
| 3 | sensing MI `I(N_k ; Y_k)` | `round_sensing_mi` | existing |
| 4 | `I(U_k ; n_Z,k+1 \| n_Z,k)` | `round_target_actuation_cmi` | existing |
| 5 | `I(U_k ; n_truth,k+1 \| n_truth,k)` | `round_truth_actuation_cmi` | existing |
| 6 | bootstrap CIs + nulls | `bootstrap_ci_low/high`, `null_mean`, `round_information_nulls.csv` | existing |
| 7 | `E[Δp_Z \| ADVOCATE] − E[Δp_Z \| NO_OP]` | `round_target_signed_response_share` | **new name, existing `_signed_response`** |
| 8 | `I(U_k ; n_Z,k+1 \| n_Z,k, E_k)` | `round_memory_target_actuation_cmi` | **new name, same CMI estimator** |
| 9 | coarse joint `(κ, φ)` diagnostic | `round_epistemic_target_actuation_cmi` | **new name, labelled separately** |
| 10 | `I(U_k ; n_Z,k+1 \| n_Z,k, φ̄_k)` | `round_phi_target_actuation_cmi` | **new name, same CMI estimator** |
| 11 | `I(U_k ; n_Z,k+1 \| n_Z,k, s̄_k)` | `round_susceptible_target_actuation_cmi` | **new name, same CMI estimator** |
| 12 | `I(U_k ; n_Z,k+1 \| n_Z,k, κ̄_k)` | `round_kappa_target_actuation_cmi` | **new name, same CMI estimator** |
| 13 | matched signed response, one per conditioning above | `round_{memory,epistemic,phi,susceptible,kappa}_target_signed_response` | **new names, existing `_signed_response`** |

All 30 are named explicitly in the config's `analysis.estimators`, so the run
computes them itself — see [The analysis is declared in the config](#the-analysis-is-declared-in-the-config-not-left-implicit).

Note on #7: the pipeline already had `round_target_signed_actuation`, which is
the same comparison **state-matched** on `n_Z,k` and expressed in aligned
magnetization (`Δm = Δp · K/(K−1)`). Both are reported. The new one is the
plain marginal difference in share units, obtained by handing the *existing*
`_signed_response` a constant conditioning state.

### The epistemic memory state

```
E_k = (n_k^(0), n_k^(1), ..., n_k^(L))
```

`n_k^(j)` = agents holding exactly `j` of the `L` supporting facts at the
**start** of round `k`. At `L = 2` this is `(n0, n1, n2)`.

Statistic #8 is the *same* `conditional_mutual_information` call as #4 with a
wider `z` — nothing else. A unit test
(`tests/mas_cc/test_relational_round_feedback_analysis.py`) pins this down from
both directions: a fixture where two memory regimes push the target opposite
ways gives **0.000 bits** on #4 and **1.000 bits** on #8, and a fixture with a
constant `E_k` makes #8 collapse exactly onto #4.

**Sparsity is expected and is reported, never hidden.** `analysis_summary.json`
carries a `memory_conditioning_support` block per cell — conditioning-state
count, rounds, singleton fraction, dual-action state/event fraction, and a
`support_limited` flag. Every statistic's support columns are computed against
**its own** conditioning state, so #8's numbers describe the slices #8 actually
faced.

#9 is a deliberately coarse fallback: `(κ_k, φ_k)` binned jointly into 4×4. It
is reported **under its own name** and never substituted for #8.

### The scalar coarse-grained epistemic conditionings (#10–#12)

`E_k` is three-dimensional at `L = 2` and is expected to be support-limited at
this sample size. #10–#12 are the low-dimensional answer: three **scalar**
variables, each conditioned on **separately** — never jointly, which would
rebuild exactly the sparsity they exist to avoid.

| variable | definition | read from |
|---|---|---|
| `φ_k` | `n_k^(L) / N`, the full-proof fraction | the last stratum of the recorded `E_k` (falls back to `full_proof_agent_share_before`) |
| `s_k` | `1 − φ_k`, the **socially susceptible** fraction — who can still be moved by what they are told | derived from `φ_k` |
| `κ_k` | mean supporting-fact coverage | the already-recorded `mean_supporting_fact_coverage_before`, not recomputed |

Each is discretised into three interpretable bins, half-open below and closed
at the top so `1.0` lands in `high`:

```
low     [0, 1/3)
medium  [1/3, 2/3)
high    [2/3, 1]
```

No repository-wide share-binning utility existed to reuse
(`metrics.interactions.non_overlapping_bins` bins interaction indices,
`synthetic.empowerment.binning_matrix` bins occupation counts), so
`coarse_bin` compares against the edges directly rather than by
multiply-and-truncate — `2/3` is the case that decides it.

**`s` is `1 − φ`, so #10 and #11 will usually be identical.** CMI is invariant
under relabelling of the conditioning variable, and three equal bins make the
two labellings a relabelling of one another on almost any data. This is
expected, not an error; `analysis_summary.json` reports
`phi_susceptible_partition_identical` so the coincidence is stated rather than
rediscovered. `s` is carried anyway because the population it names is a
different scientific object from `φ`.

Every one of #10–#12 gets the **full existing reporting**: estimate, bootstrap
CI, the policy-conditional null, the `H(U_k | Z)` entropy ceiling and its
bound check, conditioning-state count, singleton fraction, dual-action state
fraction and dual-action event fraction — each computed against *its own*
conditioning state. #13 mirrors each one with `E[Δp_Z | ADVOCATE] −
E[Δp_Z | NO_OP]` matched on the same state.

The full `E_k` result (#8) stays in the report as the high-dimensional
reference **even when sparse**, flagged by `support_limited`.

κ and φ themselves are kept alongside the information quantities, per episode
(`episode_epistemic_regime.csv`) and per round
(`round_epistemic_trajectory.csv`). An actuation CMI says how much the
controller moved the opinion channel; it says nothing about whether the
population was epistemically *able* to move. Reading the two apart is how a
null result avoids being misattributed.

---

## Smoke-run evidence (mock provider, provider-free)

Full pilot shape — 2 cells × 10 repetitions × 10 rounds = **200 round records,
20 episodes**:

| check | result |
|---|---|
| both actions occur | ADVOCATE **138**, NO_OP **62** (freq 0.690; policy mean p = 0.663) |
| `q_c = 12` agents sensed | `sensor_sample_size = 12` and 12 **distinct** `sensor_agent_ids` on all 200 rounds |
| `b = 12` on ADVOCATE | `controlled_position_count = 12` on all 138 ADVOCATE rounds |
| 0 on NO_OP | `controlled_position_count = 0` on all 62 NO_OP rounds |
| state–action overlap | pooled: 77 `N_k` slices, dual-action state fraction **0.312**, singleton fraction **0.175**; on the `(n_Z,k, E_k)` conditioning: 14 slices, dual-action state fraction **0.786**, singleton fraction **0.005** |
| entropy bound | `H(U_k) = 0.893` bits, `H(U_k \| N_k) = 0.519`; every CMI ≤ its ceiling |

**A caveat that matters for reading the mock numbers.** The mock returns a
canned ballot with a fixed `reason`, so no fact ever moves between agents:
`E_k` is constant at `(12, 12, 0)`, κ is pinned at 0.25 and φ at 0.0 for all
200 rounds. The memory-aware CMI therefore *correctly* collapses onto the plain
one (both 0.302 bits pooled) — that is the constant-conditioning behaviour the
unit test asserts, and it is **not** a prediction about the real run. Only a
live model produces peer reasons that carry facts, and only then does `E_k`
move. The mock verifies the plumbing and the controller; it cannot verify the
memory channel.

---

## Cost and shape

`experiment preflight`, status **permitted**:

| | value |
|---|---|
| cells | **2** (`task_0001`, `task_0002`) |
| episodes | **20** (10 repetitions per cell) |
| provider calls, nominal | **5,280** (264/episode = 24 initial votes + 24 × 10 focal updates) |
| provider calls, expected | **5,580** (5 % validation-failure allowance) |
| provider calls, conservative | **10,560** (retry bound) |
| rough runtime | ~2,094 s at concurrency 8 |
| cost | 0.00 proxy accounting units (University proxy reports no price) |

`q` does not change the call count: control **replaces** a social slot rather
than adding one, so a focal update is one call whatever `q` is.

---

## Running

Preflight makes no model calls.

```bash
conda run -n MA-CC --no-capture-output python -m mas_cc.cli.main experiment preflight \
  --config configs/runs/relational_reasoning/population_study_03/relational_population_study03_g_stochastic_feedback_pilot.yaml \
  --output-dir results/inspection/relational_study03_g_preflight
```

Launch:

```bash
conda run -n MA-CC --no-capture-output python -m mas_cc.cli.main experiment run \
  --config configs/runs/relational_reasoning/population_study_03/relational_population_study03_g_stochastic_feedback_pilot.yaml
```

Then the analysis, offline and provider-free, over the finished run directory:

```bash
conda run -n MA-CC --no-capture-output python -m mas_cc.cli.main analysis relational-round-feedback \
  --run-dir results/relational_imitation_round_feedback/relational-study03-g-stochastic-feedback-pilot/<run-id>
```

It writes `relational_imitation_round_feedback_analysis/` next to the run:
`round_information_estimates.csv` / `.md`, `round_information_nulls.csv`,
`round_support_diagnostics.csv`, `controller_action_summary.csv`,
`episode_epistemic_regime.csv`, `round_epistemic_trajectory.csv`,
`analysis_summary.json`. Estimates are emitted per cell **and pooled** — with
100 rounds per cell the per-cell tables are thin, and the pooled slice is the
one with a real chance of populating the memory-aware conditioning.

### The analysis is declared in the config, not left implicit

`analysis.enabled` is **`true`**, and `analysis.estimators` lists all 30
statistics by name. That block is the study's measurement plan: `experiment
preflight` validates every name against `ROUND_ANALYSIS_STATISTICS`, so a typo
fails **before** the run spends anything, and the finished run carries its own
report at

```
<run-dir>/relational_imitation_round_feedback_analysis/
```

The analyzer writes local files only — the relational path has no Comet
integration at all, so `analysis` cannot open a remote experiment regardless of
`logging.comet`.

`analysis.options` sets `bootstrap_resamples: 1000`, `null_permutations: 1000`
and `epistemic_bins: 4` (the joint `(κ,φ)` diagnostic only). The whole analysis
takes ~35 s for the 200-round pilot, so there is nothing to economise.

The offline command below reads the same `round_trajectory.jsonl` and runs the
same estimators, and is what to use for re-analysis or for a run made before
the estimator list was declared.

> **If you edit this config and then re-launch, pass `--no-resume` or use a
> clean output directory.** `--resume` defaults to on, and checkpoints written
> under a different `resolved_config_hash` are rejected — the run aborts rather
> than mixing two configurations in one result set.

### Re-running the analysis on results that already exist

The command is idempotent and provider-free, so an already-finished run is
re-analysed by pointing at it again — no re-run of the experiment:

```bash
conda run -n MA-CC --no-capture-output python -m mas_cc.cli.main analysis relational-round-feedback \
  --run-dir results/relational_imitation_round_feedback/relational-study03-g-stochastic-feedback-pilot/<run-id> \
  --output-dir results/relational_imitation_round_feedback/relational-study03-g-stochastic-feedback-pilot/<run-id>/relational_imitation_round_feedback_analysis \
  --bootstrap-resamples 1000 --null-permutations 1000
```

`--epistemic-bins` still controls only the **joint** `(κ, φ)` diagnostic (#9).
The three scalar conditionings are fixed at the three bins above by design.
