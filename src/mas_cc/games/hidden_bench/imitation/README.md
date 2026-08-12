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

## Scaled `imitation_N` protocol

`game.options.social_group_size` is the social interaction order `q` and
defaults to `1`. One focal agent and `q` distinct ordinary peers are sampled
without replacement. Under `NO_OP`, all `q` peer slots remain. Under
`ADVOCATE_Z`, one logged slot is replaced by the external controller, leaving
`q-1` ordinary peer inputs and exactly `q` total influence inputs. The
controller remains outside the population. `control.options.sensor_sample_size`
is the independent sensing budget `q_c`; sensor/social overlap and inclusion of
the focal agent are allowed and logged.

Reasoning mode performs one dyadic exchange with every retained peer in sampled
slot order, then one focal update over the ordered group input. Classical mode
samples and logs the same context but deliberately retains the existing linear
`irisarri_multi_opinion` transition kernel: peer opinions do not yet define a
new q-voter rate.

Every event includes `population_size`, `social_group_size`, ordered peer IDs,
votes and messages, ordered `influence_slots`, controller sensor IDs/votes,
replacement ID/slot, and the pre/post occupation state. Raw event time remains
the discrete interaction index; `tau = interaction_index / N` is population
sweep time, not continuous physical time.

Scaled semantic populations use `task_set: expanded` with
`assignment_scheme: paraphrased_replication`. The freezer/build commands are
documented in the `imitation_N` grid config. Agent assignments retain the
frozen `variant_id`, source evidence indices/text, and transformation
provenance.

The post-hoc estimators `truth_current` and `truth_current_fano` add, per
episode, switches toward truth minus switches away and, per cell,
`abs(mean(J_truth)) / sample_variance(J_truth)`. The net current telescopes to
the final minus initial truth headcount, so the directional switch counts are
also stored. Zero dispersion with nonzero mean is reported as `+inf`; zero mean
and zero dispersion is `NaN`. Uncertainty resamples whole episodes and there is
no action-label shuffle null.

The provider-free validation config is
`configs/runs/hidden_bench/hidden_bench_imitation_scaled_q_qc_classical_smoke.yaml`.
The nine-cell reasoning grid is
`configs/runs/hidden_bench/hidden_bench_imitation_N_q_qc_phase_grid.yaml` and
crosses `q in {1,2,4}` with `q_c in {2,8,32}` at `N=32` for ten sweeps. It
selects `evacuation_west_city`; all four evidence types have 10 accepted unique
paraphrases, versus 8 required per type. Its schema-v2 response contract omits
`allowed_values` intentionally: the selected task supplies the answer alphabet
to the concrete runtime prompts and validators.

That grid uses `artifact_profile: results_only`. Each completed cell retains a
compact `scientific_events.parquet`, its aggregate/plots, two sampled prompts,
and immediate human-readable reports under `cells/<cell-id>/reports/`. Report
filenames include the cell ID, task, and resolved `N`, `q`, and `q_c`, for example
`information_estimates__cell-0007__task-evacuation_west_city__N-32__q-4__qc-8.md`.
Per-cell bootstrap/null
analyses are serialized to bound local CPU and memory use; provider execution
in unfinished cells can continue. These local reports do not open extra Comet
experiments. The final combined grid analysis and its single configured Comet
export still run after all cells complete.

Compact scientific rows retain the population/sensor/action/focal variables
needed by every configured MI, entropy, controller diagnostic, and truth-current
estimator. They intentionally omit full transcripts, ordered peer/controller
messages, individual sensor IDs and overlaps, and most per-episode raw files;
use the full-profile smoke run when those protocol-audit details are needed.
