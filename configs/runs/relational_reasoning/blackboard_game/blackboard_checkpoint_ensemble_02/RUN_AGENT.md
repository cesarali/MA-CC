# Runner handoff: augment the paired-parent archive with batch 02

## Purpose and scope

Generate additional samples from the same simulation design as:

/Users/rsanchez/Projects/MA-CC/configs/runs/relational_reasoning/blackboard_game/blackboard_checkpoint_ensemble_01

The user confirms that the simulation code/pipeline has not changed. Keep it unchanged. This new bundle changes sample count, simulation seed and batch/output identity. It does not propose a new controller, task or estimator. The present request prepared configuration files only; no run was submitted. Obtain the user's execution authorization in the run task before incurring provider costs, following the repository workflow. Instructions here are a handoff, not evidence that submission occurred.

The local handoff directory is /Users/rsanchez/Projects/agents_control/paired_parent_exp_2. Copy its YAML files and this guide together to the execution host. On Potsdam, a suitable repository config destination is configs/runs/relational_reasoning/blackboard_game/blackboard_checkpoint_ensemble_02. Run commands from that host's MA-CC repository root. Do not submit the old ensemble_01 config directory by accident.

Read the repository AGENTS.md and .codex/skills/ma-cc-study-workflow/SKILL.md, including its Potsdam university-provider reference. Use generic study launchers. No study-specific SLURM script is needed.

## Scientific design: preserve exactly

- Four settings: (q,rho)=(3,.70),(3,1.00),(12,.70),(12,1.00).
- q is the ordinary agents' board-message reading allowance; controller sensing size remains 12 for every q.
- Population size N=24, task_003, original frozen task facts, original provider/model and prompt settings.
- 60 independently generated parent bundles per setting. execution.repetitions and ensemble.parent_count are both 60 and must agree.
- Evolve each parent for exactly two rounds without a controller. Save its full state once at that checkpoint.
- Fork nine continuations: none; always_truth at b=3,12; always_false at b=3,12; sensing_truth at b=3,12; sensing_false at b=3,12.
- One continuation copy per branch. Each continuation evolves for ten rounds. The base horizon and rounds remain 12, not 11 or 92. Total computational work is 2+9*10=92 population rounds per parent because preparation is shared.
- Truth target ALLOCATION_0, selected false target ALLOCATION_2. A false target is one specified incorrect allocation, not all wrong answers combined.
- Keep controller_actuation_mode truthful_strategic_report, dawn_only timing, fixed report pool/selector, sample size 12, threshold .5, beta 4, and all repetition/cooldown options as copied. No LLM controller authoring. Ordinary agents remain LLM-driven and may REPORT, REQUEST or post NONE.
- Keep results_only retention and parent artifacts. Do not reuse an old checkpoint as a new parent. Do not replace new parents with extra continuation copies of old parents.

All llm_provider, prompt, game, control, storage, budget and pricing sections are copied unchanged. The ensemble changes only parent_count. analysis.yaml changes only expected_parents from 160 to 240; its resampling seed remains unchanged intentionally. The simulation seed is different. Do not copy the original report.yaml: it points to the first batch's saved analysis and is not needed to collect batch 02.

## New seeds: augmentation, not replay

Every new run config uses execution.seed=20260922 instead of 20260916. Preserve the existing per-setting stream derivation and within-parent branching procedure. The same new root seed across the four files preserves the original seed arrangement across settings; do not create a fresh arbitrary seed every time a failed batch resumes.

Local checks found 60 unique derived episode seeds per new config and zero overlap with the first 40 episode seeds for both standalone and grid-cell-0 derivation paths in the current code. This is not a comparison against an independently regenerated remote execution manifest. Before submission, compare the resolved new episode seeds and scientific identities against the retained batch-01 manifest. Record the seed schedule and reject accidental old-seed reuse. Different names or output directories alone do not guarantee new stochastic streams.

Use study name blackboard-checkpoint-ensemble-02 and the new output root. On a genuine resume of batch 02, keep its seeds and output root unchanged so completed work is reused. Do not resubmit batch 01, overwrite its results, or relabel copied old paths as new data. LLM stochasticity does not guarantee byte-identical outputs from identical seeds; the point is to generate a deliberately new population ensemble rather than rely on such variability.

## Inputs and execution paths

Configured relative dataset path:

results/studies/musr_truthful_selective_task_calibration_01/tasks

Required task file:

results/studies/musr_truthful_selective_task_calibration_01/tasks/task_003/base_task.json

This path was preserved from batch 01. It is absent from the local checkout used to prepare this bundle, so full local preflight stopped at this prerequisite. Stage the exact original dataset on the execution host and verify its task/fact/report-pool hashes against batch 01. Do not regenerate task_003, substitute a new task with the same name, or silently alter prompts to make preflight pass.

Authoritative scientific output root in study.yaml:

/work/ojedamarin/Projects/LanguageGames/MA-CC/results/studies/blackboard_checkpoint_ensemble_02

It is guarded by execution.require_results_under. On Potsdam the generic launcher must set the runtime repository directory to /home/ojedamarin/Projects/LanguageGames/MA-CC; results and SLURM logs belong under /work. Confirm absolute stdout/stderr paths under the new result root's logs directory. Credentials must remain in the host's existing environment, never copied into this handoff.

## Preflight and authorized execution

On Potsdam use the dedicated MA-CC Conda environment. Before submission, check imports without printing credentials:

```bash
/home/ojedamarin/.local/share/miniforge3/bin/conda run -n MA-CC python -c 'import mas_cc, pandas, pyarrow; print(mas_cc.__file__); print(pandas.__file__); print(pyarrow.__file__)'
```

After staging the bundle at the suggested host config destination, run from the repository root:

```bash
/home/ojedamarin/.local/share/miniforge3/bin/conda run -n MA-CC mas-cc study preflight --config-dir configs/runs/relational_reasoning/blackboard_game/blackboard_checkpoint_ensemble_02 --output-dir /work/ojedamarin/Projects/LanguageGames/MA-CC/results/studies/blackboard_checkpoint_ensemble_02/preflight
```

Check all four configs, expected 240 parent-bundle scheduler episodes, 2,160 scientific continuations, the output root, seeds, manifests, input hashes and complete retention. Do not treat a typed-loader pass as launch approval. Refresh university-provider model availability, accounting units and limits using the repository workflow on the host.

Report nominal/expected/conservative requests, tokens, cost with units, predicted wall time and operational limits before launch. Copied limits: 4 active nodes/shards maximum, 4 throttle, 20 parent episode slots and 20 provider request concurrency per run, study-wide shared adaptive cap 80, target 600 RPM, 8 CPUs and 12 GiB per shard, 24-hour time limit. Effective simultaneous request ceiling is no greater than 80 and may be reduced by adaptive throttling/RPM and branch serialization. Do not automatically increase these values or spending limits.

Design arithmetic: 535,680 nominal no-retry requests including initialization. The original static assumptions imply about 630,164 expected requests and 2,678,400 conservative requests for 240 parents, but fresh executable preflight takes precedence. At a sustained 600 RPM the nominal request-only floor is about 14.9 hours; the expected-request equivalent is about 17.5 hours. These exclude queue time and do not predict actual duration. Check that the conservative execution plan fits 24-hour allocations; use the generic supported scheduling/resume workflow if not, without changing scientific parameters. A monetary estimate is unavailable from the local preparation; proxy_accounting_unit is not USD. Do not weaken budget checks merely to pass preflight.

After the user's execution authorization and successful preflight:

```bash
/home/ojedamarin/.local/share/miniforge3/bin/conda run --live-stream -n MA-CC mas-cc study submit --config-dir configs/runs/relational_reasoning/blackboard_game/blackboard_checkpoint_ensemble_02
```

Monitor scientific completeness as well as scheduler exit status. Preserve completed parents/checkpoints during recovery. Once complete, aggregate the new batch strictly:

```bash
/home/ojedamarin/.local/share/miniforge3/bin/conda run --live-stream -n MA-CC mas-cc study aggregate --study-dir /work/ojedamarin/Projects/LanguageGames/MA-CC/results/studies/blackboard_checkpoint_ensemble_02
```

Incomplete exploratory aggregation must be labeled explicitly. Do not claim 60 complete parents per setting until all required branch records verify.

## Combine with the first batch only after separate validation

The original archive has 160 parent populations and 1,428 complete continuations out of 1,440 intended. Batch 02 adds 240 parents and 2,160 intended continuations. With complete new data, totals are 400 parents across settings and 3,588 complete continuations; the old missing 12 paths remain missing unless recovered separately. Per setting, most comparisons reach 100 pairs, and affected old comparisons reach 98.

Preserve batch identity, original parent_id, config, seed and source manifest. Use a qualified parent key such as (batch_id, original_parent_id), checking seed and checkpoint provenance before deduplicating. Some IDs can repeat across runs without meaning the same physical population. Conversely, renamed files are not new observations.

Combine canonical observations within identical scientific settings, not precomputed MI/CMI values or plot averages. Parent-level folds and bootstrap groups must include all descendants of each parent. Keep this batch identifiable so it can support independent validation of models developed on batch 01; predeclare any held-out new parents before model selection.

The repository's complementary-study merge is not a generic same-cell repetition concatenator. Read docs/documentation/metrics/merging_complementary_studies.md before any combining operation and verify an episode-aware method that retains both non-overlapping repetition sets; do not use a cell-preference merge that discards one batch. Combined analysis is a subsequent task, not required to collect and preserve batch 02.

## Deliverables for the run agent

Return the submitted config hashes, code revision, task hashes, seed/identity checks, preflight report, job ID, exact output root, per-setting completeness and exclusions, and a separately named batch-02 analysis package. Retain full parent artifacts/checkpoints as configured, canonical round and micro-update records, and frozen analysis inputs needed for later model training. Do not mix this data with rnd_init_experiment.
