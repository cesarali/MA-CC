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

## First control pilot and offline report

The matched four-cell pilot is
`configs/runs/hidden_bench/hidden_bench_imitation_first_control_grid.yaml`.
It crosses reasoning/classical dynamics with none/threshold-target control,
uses the same explicit initial votes in every cell, and runs 12 episodes per
cell.  Preflight and run it with:

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
