---
name: ma-cc-study-workflow
description: Create, preflight, submit, monitor, aggregate, or post-process MA-CC experiments and studies using YAML scientific configs and the generic SLURM config-array workflow. Use for Study 04/06 variants, Potsdam runs, pooled estimator analysis, derived observables, and phase diagrams.
---

# MA-CC Study Workflow

Work from the repository root. Keep scientific design separate from scheduler
topology and reuse the repository's established analysis engines.

## Read the architecture first

Before designing, submitting, or changing analysis, read:

- `docs/tdd/features/orchestrator/22082026_TDD_standardized_study_submission_and_aggregation.md`
- `docs/handoff/22082026_standardized_study_submission_and_aggregation_handoff.md`

Also inspect the README, preflight report, configs, and analysis recipe for the
closest existing study family. Prefer Study 04 or Study 06 unless another
family is demonstrably closer to the requested scientific design.

## Preserve these invariants

- Do not create a study-specific `.job` file. Use
  `scripts/Potsdam/SLURM/run_config_array.job`. Add a different generic launcher
  only when the scheduler allocation topology genuinely cannot be represented
  by the existing launcher, and explain the topology difference.
- Put hypotheses, fixed parameters, sweep axes, seeds, models, budgets,
  retention, and analysis requests in YAML—not shell scripts.
- Group related configs in one study folder with `study.yaml` and, when needed,
  `analysis.yaml`.
- Treat each SLURM array task as an execution shard only. Scheduler IDs and
  shard boundaries are never scientific coordinates.
- Reuse existing MI/CMI, bootstrap, null, and support estimators. Never
  implement another CMI estimator.
- Compute pooled CMI from pooled canonical observations. Never average
  shard-level or cell-level CMI estimates to manufacture pooled CMI.
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

## Submit and monitor

For an authorized real run, use:

```bash
mas-cc study submit --config-dir <folder>
```

Pass `--results-dir <study-result-root>` when the destination should be
explicit. The command preflights every config again and submits one generic
config array. Let `study.yaml` or the CLI throttle bound array concurrency; do
not increase it without recalculating provider load.

Monitor the returned job with the site's ordinary `squeue`/`sacct` tools and
verify task exit states and run/cell seals. For a scheduler smoke, use a tiny
mock-provider study; never point a smoke test at a production study folder.

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
3. reuse cached primary estimates where their data and specification match;
4. do not rerun LLM calls.

Rerun LLM calls only when the requested result genuinely needs raw fields or
scientific conditions absent from the retained data. State that reason before
launching new data collection.

## Handoff checklist

Report the study folder, result root, preflight totals, submission/job identity
if submitted, aggregation status, test evidence, retained-schema checks, and
any blocker. Explicitly state that no study-specific job or replacement CMI
implementation was added.
