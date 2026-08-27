---
name: ma-cc-study-workflow
description: Create, preflight, submit, monitor, aggregate, or post-process MA-CC experiments and studies using YAML scientific configs and the generic SLURM config-array workflow. Use for Study 04/06 variants, Potsdam runs, per-cell estimator analysis, derived observables, and phase diagrams.
---

# MA-CC Study Workflow

Work from the repository root. Keep scientific design separate from scheduler
topology and reuse the repository's established analysis engines.

## Potsdam dedicated environment

When operating on Potsdam or submitting to its SLURM cluster, use the dedicated
Conda environment `MA-CC` for every Python, `mas-cc`, and pytest command. The
canonical Potsdam Conda executable is
`/home/ojedamarin/.local/share/miniforge3/bin/conda`. Prefer the explicit forms
below because environment activation does not persist between agent tool
calls:

```bash
/home/ojedamarin/.local/share/miniforge3/bin/conda run -n MA-CC python ...
/home/ojedamarin/.local/share/miniforge3/bin/conda run -n MA-CC mas-cc ...
/home/ojedamarin/.local/share/miniforge3/bin/conda run -n MA-CC python -m pytest ...
```

Use `--live-stream` for long-running execution, monitoring, or aggregation
where buffered output would hide progress. On Potsdam, never substitute
`/usr/bin/python`, create a study-specific environment, or install dependencies
into the system interpreter. Before real submission, verify the environment
with a credential-free import check for `mas_cc`, `pandas`, and `pyarrow`.
Ensure the SLURM job inherits or explicitly invokes this same `MA-CC`
environment; the login shell's system Python is not a valid fallback.

Outside Potsdam, use the existing environment and setup conventions of the
local checkout. Do not require the Potsdam environment name or absolute Conda
path on a developer's local machine.

## Read the architecture first

Before designing, submitting, or changing analysis, read:

- `docs/tdd/features/orchestrator/22082026_TDD_standardized_study_submission_and_aggregation.md`
- `docs/handoff/22082026_standardized_study_submission_and_aggregation_handoff.md`

Also inspect the README, preflight report, configs, and analysis recipe for the
closest existing study family. Prefer Study 04 or Study 06 unless another
family is demonstrably closer to the requested scientific design.

## Preserve these invariants

- Do not create a study-specific `.job` file. Use the generic
  `scripts/Potsdam/SLURM/run_config_array.job` for config arrays or
  `scripts/Potsdam/SLURM/run_study_cell_array.job` for planned cell arrays. Add
  another launcher only when scheduler topology genuinely differs, and explain
  that difference.
- Put hypotheses, fixed parameters, sweep axes, seeds, models, budgets,
  retention, and analysis requests in YAML—not shell scripts.
- Group related configs in one study folder with `study.yaml` and, when needed,
  `analysis.yaml`.
- Treat each SLURM array task as an execution shard only. Scheduler IDs and
  shard boundaries are never scientific coordinates.
- Reuse existing MI/CMI, bootstrap, null, and support estimators. Never
  implement another CMI estimator.
- Compute observables at their physical scientific-cell coordinates. Do not
  manufacture a study-wide estimator by mixing heterogeneous parameter cells.
- Treat `effective_affinity` (`h_eff`) and `kinetic_compliance` (`gamma_eff`) as
  measured outcomes, not sweep knobs.
- With `artifact_profile: results_only`, preserve the round and micro-slot
  fields required by every configured estimator. Verify this against an actual
  downscaled execution when a new game/config path changes retention.

## Create or modify a study

1. Copy the closest existing scientific config family and change only the
   requested design parameters.
2. Keep related experiment configs in one folder and list their stable order in
   `study.yaml`.
3. Put primary estimators, resampling, derived observables, and plots in
   `analysis.yaml`.
4. Resolve and preflight every config without provider calls:

   ```bash
   mas-cc experiment preflight --config <config.yaml> --output-dir <inspection-dir>
   ```

5. Before any real submission, report at least:

   - config and cell counts;
   - repetitions and total episodes;
   - nominal, expected, and conservative provider calls when available;
   - scheduler throttle, experiment parallelism, provider request concurrency,
     and the effective concurrency ceiling;
   - token and monetary cost estimates, including their accounting units and
     whether they are bounds or predictions;
   - estimated wall time and its assumptions.

Do not submit if preflight denies launch, required credentials are absent, the
result root is ambiguous, or the requested real-run authorization is missing.

On the Potsdam cluster, real study outputs must live beneath
`/work/ojedamarin/Projects/LanguageGames/MA-CC/results`, never beneath the home
repository. Put the authoritative absolute destination in
`study.yaml: execution.results_root` and guard it with
`execution.require_results_under`. Before submission, verify the resolved
study root and the generated manifest output paths. SLURM stdout/stderr must use
absolute paths under `<study-result-root>/logs`; never rely on Slurm's relative
`slurm-%A_%a.out/.err` default in the repository working directory.

## Plan, submit, and monitor

Treat the scientific cell as the aggregation unit and the episode as the
computational unit. Execution topology must not change cell IDs, overrides,
episode seeds, canonical observations, or estimator results.

Before submission, inspect the generated execution plan and require it to:

- expand workload independently of YAML-file boundaries;
- use cell or cell-bundle shards when config-level tasks underuse the cluster;
- report nodes/tasks, CPUs per task, episode slots, request concurrency, RPM,
  memory, time limit, and expected wall time;
- derive the array throttle from a declared provider RPM target and planning
  latency, with a configurable node ceiling;
- reject a requested throttle above the calculated provider-safe bound;
- request explicit SLURM CPU, memory, and time resources;
- ensure conservative shard duration fits the SLURM time limit.

For remote API workloads, optimize useful request throughput rather than CPU
allocation alone. Each configured episode slot runs one episode at a time;
provider concurrency remains an independent hard semaphore. Prefer a live,
model-specific concurrency probe when available and keep large runs below the
declared RPM target. Published RPM is a ceiling, not guaranteed throughput.
Standardized workers also apply the shared adaptive provider coordinator from
`execution.provider_load_control`. Keep it enabled for real remote-provider
studies: start conservatively, bound it by planned request capacity and target
RPM, and let local/global circuit breakers respond to 429/5xx bursts. Do not
replace it with game-specific throttling or a study-specific job file.

For an authorized real run, use:

```bash
mas-cc study submit --config-dir <folder>
```

Pass `--results-dir <study-result-root>` when overriding a permitted
destination. The command preflights every config again and submits one generic
study array. `execution.mode: auto` should generate execution shards and an
`execution_plan.json`; config-array mode remains a compatibility fallback.
Do not increase a throttle without recalculating provider load.

Monitor the returned job with the site's ordinary `squeue`/`sacct` tools and
verify task exit states and run/cell seals. For a scheduler smoke, use a tiny
mock-provider study; never point a smoke test at a production study folder.

Sealed cell-array workers retain canonical scientific observations; they do not
publish persistent estimator, bootstrap, or permutation caches. In post-hoc
aggregation, request multiple SLURM CPUs. Aggregation uses
`SLURM_CPUS_PER_TASK` and exposes transient live state in
`analysis/progress.json`, which is removed after success. Final study work joins
cell outputs, computes compact estimator summaries, derives observables,
renders plots, and packages the handoff.

The standardized study analysis package contains canonical scientific data,
compact estimator summaries, support diagnostics, plots, reports, validation,
and provenance. Bootstrap/permutation draws and analysis caches are transient
computational intermediates and are not retained. Parquet is authoritative;
do not emit table-wide CSV mirrors or package source run trees and execution
artifacts. Reaggregation recomputes from the canonical Parquet observations.

## Aggregate and extend analysis

After the expected runs and cells are sealed, use strict aggregation:

```bash
mas-cc study aggregate --study-dir <study-result-root>
```

Use `--allow-incomplete` only for explicitly partial exploratory output, never
for a final pooled scientific result.

For a new efficiency, phase diagram, or plot that existing raw records support:

1. add a derived observable and/or analysis recipe entry;
2. rerun `study aggregate`;
3. recompute from the retained canonical Parquet observations;
4. do not rerun LLM calls.

Rerun LLM calls only when the requested result genuinely needs raw fields or
scientific conditions absent from the retained data. State that reason before
launching new data collection.

## Handoff checklist

Report the study folder, result root, preflight totals, submission/job identity
if submitted, aggregation status, test evidence, retained-schema checks, and
any blocker. Explicitly state that no study-specific job or replacement CMI
implementation was added.
