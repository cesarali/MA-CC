# Relational reasoning — population study 04

## The sensing/actuation resource grid

Study 03 arm G measured the controller-to-population information channel at
**one** point of the resource plane: `q_c = 12`, `b = 12`, both at half the
population. It answered *"are these quantities estimable at all"* — yes — but a
single point has no slope, so it could say nothing about how the channel
**scales** with the controller's resources.

Study 04 is the same measurement over a 3 × 3 grid:

```
q_c ∈ {6, 12, 18}      sensing:   how many agents the controller LOOKS AT
b   ∈ {6, 12, 18}      actuation: how many positions it ACTS ON
```

over the two matched worlds, 30 repetitions each.

|  | b = 6 (c=0.25) | b = 12 (c=0.50) | b = 18 (c=0.75) |
|---|---|---|---|
| **q_c = 6** (r=0.25) | 2 cells | 2 cells | 2 cells |
| **q_c = 12** (r=0.50) | 2 cells | 2 cells | **← Study 03's point** |
| **q_c = 18** (r=0.75) | 2 cells | 2 cells | 2 cells |

**18 task-specific cells, 540 episodes.**

---

## Files

| path | what it is |
|---|---|
| `relational_population_study04_qc06.yaml` | row `q_c = 6`, sweeps `b ∈ {6,12,18}` × 2 worlds → 6 cells, 180 episodes |
| `relational_population_study04_qc12.yaml` | row `q_c = 12`, same sweep → 6 cells, 180 episodes |
| `relational_population_study04_qc18.yaml` | row `q_c = 18`, same sweep → 6 cells, 180 episodes |
| `README.md` | this file |

The three YAMLs are **byte-identical except for `q_c` and the strings derived
from it** (`sensor_sample_size`, `experiment.name`, tags, metadata, and the
comments that quote the value). They were generated from one template for
exactly that reason: the sensing axis is only a clean comparison if nothing
else moved with it.

### The three-way split is operational, not scientific

Scientifically the three files are **one** 3 × 3 resource grid. They are split
because each is independently launchable and independently resumable, so an
overnight run that dies in row two does not cost rows one and three.

---

## Everything is inherited from Study 03

The base is
`../population_study_03/relational_population_study03_g_stochastic_feedback_pilot.yaml`.
**No controller, estimator, theory, bootstrap, null, or reporting machinery was
written for this study.** Only the sweep changed.

Held fixed at the Study 03 values:

```
N = 24      L = 2       r = 6       q = 1       K = 3 options, 4 distractors
rounds = 10             threshold = 0.5         beta = 4
advocacy_schedule: soft               message_mode: recommendation_only
target: 2 (pinned, verified incorrect)
social_distrust: true                 initialization: local_vote
stop_on_consensus: false              semantic option shuffling (per call)
artifact_profile: results_only
provider / model / temperature / max_output_tokens / concurrency: identical
analysis.estimators: identical, all 32 names
analysis.options: identical (1000 bootstrap, 1000 null, epistemic_bins 4)
```

Changed, and only these:

| field | Study 03 | Study 04 |
|---|---|---|
| `control.options.sensor_sample_size` | `12` | `6` / `12` / `18` (one per file) |
| `control.options.intervention_budget` | `12` | swept `{6, 12, 18}` |
| `execution.repetitions` | `10` | `30` |
| `execution.seed` | `20260820` | `20260821` |
| `grid` | 1 axis (`task_id`) | 2 axes (`intervention_budget` × `task_id`) |
| `experiment.name` / tags / metadata | arm G | study 04, per row |

---

## The round-level ordering (unchanged)

```
N_k  ->  Y_k   sensor, q_c agents sampled without replacement, VOTES ONLY
     ->  U_k   in {NO_OP, ADVOCATE}, drawn from the soft policy
     ->  b controlled update positions  IF  U_k = ADVOCATE, else exactly 0
     ->  N_{k+1}
```

```
P(U_k = ADVOCATE | Y_k) = sigmoid[ beta * (threshold - p_Z(Y_k)) ]
```

with `beta = 4`, `threshold = 0.5` — `SoftTargetControl.select_action` /
`advocacy_probability`, unchanged.

`advocacy_schedule` **must stay `soft`.** Under `always`, `H(U_k) = 0` and every
information quantity in the study is identically zero by construction.

### `q_c` and `b` are independent — that is the point of Study 04

Study 03 held them numerically equal (12 = 12) and its header warned that this
was a coincidence of that pilot. Study 04 breaks the coincidence: each row
contains cells pairing one `q_c` against `b ∈ {6, 12, 18}`, so sensing and
actuation can finally be told apart.

| symbol | what it is | where it happens |
|---|---|---|
| `q_c` | agents the controller **looks at**, once, at the start of the round | `ThresholdTargetControl.interaction_signal`: `rng.sample(state.agents, sensor_sample_size)` |
| `b` | **update positions** actuated, only on an ADVOCATE round | `runtime.py`: `sample_controlled_positions(N, b, ...) if action == ADVOCATE_TARGET else ()` |

The sensed agents and the controlled positions come from **separate seeded
streams** (`relational-sensor:*` vs `relational-schedule:<round>`), are not
required to coincide, and are not even in the same index space (agent
identities vs within-round update slots). **Do not tie them together.**

---

## The seed is shared across the three rows on purpose

All three files use `execution.seed: 20260821` and declare their grid axes in
the **same order**, so cell index *i* is the same `(b, task)` pair in every row:

```
cell-0000 = (b=6,  task_0001)     cell-0003 = (b=12, task_0002)
cell-0001 = (b=6,  task_0002)     cell-0004 = (b=18, task_0001)
cell-0002 = (b=12, task_0001)     cell-0005 = (b=18, task_0002)
```

Episode seeds derive from the cell index
(`Seed(execution.seed).derive("grid-cell:i").derive("episode:j")`), so episode
*j* of `(b=6, task_0001)` starts from the **same initial population** in the
`q_c = 6`, `12` and `18` rows. That is common random numbers across the sensing
axis: the `q_c` comparison is **paired**, and a difference between rows is not
initial-condition noise.

**Do not reorder the axes in one file only** — it would silently break the
pairing while leaving all three configs individually valid.

---

## The target is pinned, not drawn

`control.options.target: 2` is a zero-based index into the task's frozen
`possible_answers`, **not** `random_incorrect`. `random_incorrect` resolves per
*episode* from a seed derived from the grid cell index, so two repetitions of
one cell can advocate two different relations. An index is seed- and
cell-independent, so all 540 episodes advocate one fixed relation per world.

| world | `possible_answers` | correct | pinned target (index 2) |
|---|---|---|---|
| `task_0001` | `[SOUTHWEST, WEST, EAST]` | WEST (index 1) | **EAST** |
| `task_0002` | `[SOUTHEAST, SOUTHWEST, SOUTH]` | SOUTHWEST (index 1) | **SOUTH** |

"Correct sits at index 1" is a **per-task** fact, not a dataset invariant — in
`pop24_L2_r06` the correct option is at index 2 for `task_0006` and
`task_0010`. Re-check before widening `game.options.task_id`.

---

## Optional historical classical null

The matched finite-`N` q-voter reference is deterministic, so an optional null
hypothesis can be computed **before** a single call is made. Uniform-occupancy
mean `T_qv(n)`, in bits, at `N=24, q=1, beta=4, theta=0.5`:

| | b = 6 | b = 12 | b = 18 |
|---|---|---|---|
| **q_c = 6** | 0.1664 | 0.3398 | 0.4688 |
| **q_c = 12** | 0.1645 | 0.3361 | 0.4639 |
| **q_c = 18** | 0.1638 | 0.3347 | 0.4620 |

The classical channel is driven **almost entirely by actuation**: `T_qv` nearly
triples across the `b` axis and moves by ~1.5 % across the `q_c` axis. A
memoryless imitation population barely benefits from a better sensor at fixed
`beta`, because the soft policy already averages the sensor away.

That makes the sensing axis the interesting one: **a strong empirical `q_c`
dependence would be a departure from the classical reference**, not a
confirmation of it. It is not emitted by default and must be selected
explicitly as `matched_qvoter_null`; it is never substituted for revised
theory.

---

## What each cell produces automatically

`analysis.enabled: true` with the estimator family declared, so every finished
run writes its own empirical report plus the revised single-affinity comparison
or an explicit unavailable reason:

- target CMI `I(U_k ; n_Z,k+1 | n_Z,k)`, sensing MI `I(N_k ; Y_k)`, signed response;
- `φ`-, susceptible-, `κ`-conditioned CMIs and their matched signed responses;
- full-`E_k` memory-aware CMI and the joint `(κ,φ)` diagnostic;
- support / overlap / sparsity diagnostics, bootstrap CIs, policy-conditional nulls;
- revised single-affinity `chi`, `T_pi`, `eta_IR`, `J_c`, `I_sens`, and
  `eta_th`, calibrated from retained microscopic transitions.

### The scientific coordinates land on every round record

Which is what makes the resource-space plots possible afterwards:

| coordinate | round-record field |
|---|---|
| `q_c` | `sensor_sample_size` |
| `q_c / N` | `sensing_fraction` |
| `b` | `intervention_budget` |
| `b / N` | `actuation_fraction` |
| `beta` | `controller_beta` |
| `threshold` | `controller_threshold` |

They remain on canonical observations and are attached to the standardized
study-level `single_affinity_theory_comparison.parquet`, whose rows carry the
`single_affinity_revised` reference stamp.

### ⚠ Revised theory is never pooled across physical cells

Each config's empirical analysis emits per-cell rows and a pooled descriptive
row. Because the cells span different `b` values, revised theory comparison is
emitted only per physical cell. No pooled theory tuple is manufactured. The
pooled empirical MI/CMI estimates remain available as descriptive calculations.

---

## Cost and shape

Preflight status **permitted** for all three rows (identical, since neither
`q_c` nor `b` changes the call count):

| | per row | study total |
|---|---|---|
| cells | 6 | **18** |
| episodes | 180 | **540** |
| provider calls, nominal | 47,520 | **142,560** |
| provider calls, expected | 50,220 | **150,660** |
| provider calls, conservative | 95,040 | **285,120** |
| rough runtime @ concurrency 8 | ~18,834 s (**~5.2 h**) | ~15.7 h sequential |
| cost | 0.00 proxy accounting units | 0.00 |

Per episode: 264 nominal calls = 24 initial votes + 24 × 10 focal updates.
Neither `q`, `q_c` nor `b` changes this — control **replaces** a social slot
rather than adding one, sensing reads recorded votes and makes no call, and a
controlled position is still one focal update.

`budget.max_provider_requests` is 200,000 **per run**, which sits above the
95,040 conservative bound for a single row.

### ⚠ Three rows do not fit in one overnight window sequentially

~5.2 h each, ~15.7 h for all three back to back. Options, in preference order:

1. **One row per night** — simplest, keeps concurrency 8 against the proxy.
2. **All three in parallel** — ~5.2 h wall clock, but puts 3 × 8 = 24
   concurrent requests on the University proxy. Only do this if that
   concurrency is known to be acceptable; nothing here was tuned for it.
3. Raise `request_concurrency` / `parallelism` — **deliberately not done**,
   because it is one of the Study 03 settings this study is holding fixed.

---

## Running

Preflight makes no model calls.

```bash
conda run -n MA-CC --no-capture-output python -m mas_cc.cli.main experiment preflight \
  --config configs/runs/relational_reasoning/population_study_04/relational_population_study04_qc06.yaml \
  --output-dir results/inspection/relational_study04_qc06_preflight
```

Launch one row (repeat with `qc12`, `qc18`):

```bash
conda run -n MA-CC --no-capture-output python -m mas_cc.cli.main experiment run \
  --config configs/runs/relational_reasoning/population_study_04/relational_population_study04_qc06.yaml
```

Then the analysis, offline and provider-free:

```bash
conda run -n MA-CC --no-capture-output python -m mas_cc.cli.main analysis relational-round-feedback \
  --run-dir results/relational_imitation_round_feedback/relational-study04-qc06-resource-grid/<run-id>
```

> **If you edit a config and re-launch, pass `--no-resume` or use a clean output
> directory.** `--resume` defaults to on, and checkpoints written under a
> different `resolved_config_hash` are rejected — the run aborts rather than
> mixing two configurations in one result set.
