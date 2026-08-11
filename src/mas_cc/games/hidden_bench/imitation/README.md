# HiddenBench imitation

`hidden_bench_imitation` holds the HiddenBench task, evidence allocation,
population, finite option alphabet, observables, controller, and event schema
fixed while switching the transition mechanism:

- `dynamics_mode: reasoning` performs private local votes before event 1, then
  private focal/peer exchanges followed by one focal LLM vote update.
- `dynamics_mode: classical` performs no provider calls. It runs a
  focal-conditioned embedded multi-opinion jump chain with one `A -> B` change
  per event and explicit forward/reverse reaction weights.

The classical interaction factor defaults to
`(n_destination + interaction_offset) / N`. Forward/reverse rate constants,
the positive offset, and controller strength are all recorded config values.
This v1 uses event index as its clock; `physical_time_increment` is `null` and
logs name the convention `focal_conditioned_embedded_jump_chain`.

Feedback uses the shared `Control.interaction_signal()` hook. The pilot
`threshold_target` mechanism samples agents without replacement, exposes only
sampled opinion counts to its policy, and returns `NO_OP` or `ADVOCATE_Z`.
Targets may be `correct`, an exact option label, or a zero-based option index.
Reasoning mode receives a fixed advocacy message and still chooses its own
vote. Classical mode adds a separately logged local transition weight toward
the target.

`soft_target` is the same sensor and the same two actuators with a stochastic
policy in between: `P(ADVOCATE_Z | Y_t) = sigma[beta * (threshold - sampled
target share)]`, drawn from the episode's seeded sensor RNG. It exists because
`threshold_target` saturates — with everyone off the target it always
advocates, with the target dominant it never does — so those conditioning
slices carry one action and `I(U_t; n_Z(t+1) | n_Z(t))` cannot be estimated
from them at any sample size. `beta` is the inverse policy temperature: large
`beta` recovers `threshold_target`, smaller `beta` buys action overlap.
Advocacy defaults to `template_version: 3`, one fixed line that argues for the
target and asserts nothing about the task. Every event logs
`controller_threshold`, `controller_beta`, and
`controller_advocacy_probability`, so the realized action sequence can be
audited against the policy it claims to follow.

Every event records the full pre/post vote vectors and occupation states,
sensor observation, controller action, focal/peer identities, order parameters,
evidence disclosure diagnostics, and classical channel data. Consequently
`X_t, Y_t, U_t, X_{t+1}` are reconstructable from `events.jsonl`; the richer
`trajectory.jsonl` also retains reasoning observations, prompt fingerprints,
responses, and actions without sending any of them to Comet.

Run the provider-free smoke configuration:

```bash
mas-cc experiment preflight --config configs/runs/hidden_bench/hidden_bench_imitation_classical.yaml
mas-cc experiment run --config configs/runs/hidden_bench/hidden_bench_imitation_classical.yaml
```

The mock reasoning smoke configuration is
`configs/runs/hidden_bench/hidden_bench_imitation_reasoning_mock.yaml`.

The soft controller has its own provider-free smoke configuration,
`configs/runs/hidden_bench/hidden_bench_imitation_classical_soft_control_smoke.yaml`.
Run it before any provider sweep that uses `soft_target`. The thing to check is
not `H(U) > 0` — the hard controller had that too — but whether both actions
occur inside the same `Z_t` slice. On the shipped smoke settings
(`N=4`, `sensor_sample_size=2`, `threshold=0.5`, `beta=4.0`, 1600 events) all
5 target slices and all 15 occupation states see both actions, against 2 of 5
and 7 of 15 under `threshold_target` — 31.7% of events unusable, down to 0%.

## First control pilot and offline report

The matched four-cell pilot is
`configs/runs/hidden_bench/hidden_bench_imitation_first_control_grid.yaml`.
It crosses reasoning/classical dynamics with none/threshold-target control,
uses the same explicit initial votes in every cell, and runs 12 episodes per
cell.  `hidden_bench_imitation_soft_control_grid.yaml` is the same grid with
`soft_target` in place of `threshold_target`, so the two pilots differ in the
policy alone.  Preflight and run either with:

```bash
mas-cc experiment preflight \
  --config configs/runs/hidden_bench/hidden_bench_imitation_first_control_grid.yaml
mas-cc experiment run \
  --config configs/runs/hidden_bench/hidden_bench_imitation_first_control_grid.yaml \
  --approve-preflight <preflight-output>/preflight_id.txt
```

After the grid completes, derive the behavioral report and the four discrete
information estimates without any provider calls:

```bash
mas-cc analysis hidden-bench-imitation \
  --run-dir <completed-grid-run-dir> \
  --bootstrap-resamples 1000 \
  --null-permutations 1000
```

`auc_m_ctrl` and `auc_m_truth` are v1 equal-event-spacing trajectory means,
including the initial state and every post-event state.  The analysis reports
all existing direct-counting variants and identifies the unsmoothed value as
the main estimate; support, sparsity, action-degeneracy, episode-bootstrap, and
within-episode temporal-null diagnostics are stored beside it.  No InfoNCE
estimator or output is created.

## Separate 10-episode behavior runs

For inspecting each dynamics mode independently, these otherwise matched
controlled configs run 10 episodes each. The preflight writes the cost report
and approval ID; `--no-capture-output` plus `logging.console: true` keeps the
run's banner, progress, and completion messages visible.

Reasoning preflight and run:

```bash
conda run -n MA-CC --no-capture-output python -m mas_cc.cli.main experiment preflight \
  --config configs/runs/hidden_bench/hidden_bench_imitation_reasoning_control_10.yaml \
  --output-dir inspection/hidden_bench_imitation_reasoning_control_10_preflight

conda run -n MA-CC --no-capture-output python -m mas_cc.cli.main experiment run \
  --config configs/runs/hidden_bench/hidden_bench_imitation_reasoning_control_10.yaml \
  --output-dir results \
  --approve-preflight inspection/hidden_bench_imitation_reasoning_control_10_preflight/preflight_id.txt
```

Classical/no-reasoning preflight and run:

```bash
conda run -n MA-CC --no-capture-output python -m mas_cc.cli.main experiment preflight \
  --config configs/runs/hidden_bench/hidden_bench_imitation_classical_control_10.yaml \
  --output-dir inspection/hidden_bench_imitation_classical_control_10_preflight

conda run -n MA-CC --no-capture-output python -m mas_cc.cli.main experiment run \
  --config configs/runs/hidden_bench/hidden_bench_imitation_classical_control_10.yaml \
  --output-dir results \
  --approve-preflight inspection/hidden_bench_imitation_classical_control_10_preflight/preflight_id.txt
```

Every episode writes the 12 behavioral diagnostics directly to
`metrics/streaming.csv`. To pool the 10 episodes and produce event,
episode-level, and run-level CSVs, analyze each completed run separately:

```bash
conda run -n MA-CC --no-capture-output python -m mas_cc.cli.main analysis hidden-bench-imitation \
  --run-dir results/hidden_bench_imitation/hidden-bench-imitation-reasoning-control-10/hidden-bench-imitation-reasoning-control-10-20260810

conda run -n MA-CC --no-capture-output python -m mas_cc.cli.main analysis hidden-bench-imitation \
  --run-dir results/hidden_bench_imitation/hidden-bench-imitation-classical-control-10/hidden-bench-imitation-classical-control-10-20260810
```

The most direct outputs are `event_metrics.csv`, `episode_summaries.csv`,
`cell_summaries.csv`, and `order_parameter_trajectories.csv` in each analysis
directory. The classical configuration retains the requested university
provider declaration for parity, but its call plan and runtime make zero
provider requests.
