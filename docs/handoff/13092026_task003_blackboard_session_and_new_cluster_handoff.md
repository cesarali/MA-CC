# Task003 blackboard studies: session and new-cluster handoff

**Date:** 2026-09-13  
**Purpose:** give an engineer on another cluster the scientific and operational
context needed to reproduce the current Task003 blackboard studies without
coupling the game or analysis to Potsdam SLURM.

This document supersedes the workload numbers in
`docs/handoff/09092026_blackboard_execution_portability_handoff.md`. That
document remains useful architectural background, but it describes an earlier
20-cell/180-concurrency run. The configurations listed here are the current
acceptance anchors.

## 1. The central separation

Keep these layers independent:

```text
frozen scientific inputs
        -> YAML scientific design
        -> deployment/execution topology
        -> canonical observations
        -> estimators and phase products
        -> compact analysis package
```

The deployment may be Potsdam SLURM, another batch scheduler, a single
20–30-CPU server, or Kubernetes. It must not redefine the game, seeds, cells,
episode identities, retained observations, or estimators.

The scientific hierarchy is:

```text
study -> parameter cell -> episode -> round -> microscopic slot
```

- A **cell** is a fixed scientific coordinate, currently one `(rho, b)` pair.
- An **episode** is one deterministic repetition inside a cell and is the unit
  of runtime work.
- A scheduler shard, process, host, or pod is only an execution container.
- Estimation, completeness, bootstrap membership, nulls, and support
  diagnostics remain organized by the scientific cell.

The current Potsdam launcher assigns one cell per array shard and advances
multiple episodes inside it concurrently. A new backend may distribute
episodes differently, provided stable cell/episode identities and atomic
completion are preserved.

## 2. Current configuration anchors

### DeepInfra q=12 false-control

```text
configs/runs/relational_reasoning/blackboard_game/
astra_task003_false_control_q12_deepinfra_rho3/
```

Files:

```text
study.yaml
false_control_llm.yaml
analysis.yaml
```

### DeepInfra q=12 truth-control

```text
configs/runs/relational_reasoning/blackboard_game/
astra_task003_truth_control_q12_deepinfra_rho3/
```

Files:

```text
study.yaml
truth_control_llm.yaml
analysis.yaml
```

The truth arm is the matched counterpart. It uses the supported
`control.options.target: correct` mechanism; it does not introduce another
controller mode.

### Potsdam q=3 false- and truth-control

```text
configs/runs/relational_reasoning/blackboard_game/
astra_task003_false_control_30x30_potsdam_rho3/

configs/runs/relational_reasoning/blackboard_game/
astra_task003_truth_control_30x30_potsdam_rho3/
```

Use the DeepInfra q=12 pair when reproducing the latest external-provider
experiment. Use the Potsdam q=3 pair when reproducing the university-provider
comparison.

## 3. Frozen inputs that must travel with the configs

The portable input archive is:

```text
/work/ojedamarin/Projects/LanguageGames/MA-CC/results/handoffs/
astra_task003_execution_inputs_20260910.zip
```

SHA-256:

```text
69abb34d8ba00246d58ce4378671dcdb05a9a94e4dbac59c15dd2b9928a3aebb
```

It contains:

```text
task_003/
astra_task003_false_control_30x30_initializations/
```

Place `task_003` beneath a dataset directory so the final layout is:

```text
<repo>/results/studies/musr_truthful_selective_task_calibration_01/tasks/
    task_003/
```

The experiment then resolves:

```yaml
task_dataset_dir: results/studies/musr_truthful_selective_task_calibration_01/tasks
task_id: task_003
```

`task_003` is the complete frozen world: public task, hidden truth, generated
true facts, N=24 private assignment, controller-reportable evidence and budget
subsets, plus symbolic/generation provenance. The correct answer is
`ALLOCATION_0`; the false target is `ALLOCATION_2`.

Place the initialization directory on durable storage visible to every
worker, then update the YAML's absolute `artifact_dir`:

```yaml
initialization:
  mode: paired_local_vote
  artifact_dir: <shared-results>/studies/astra_task003_false_control_30x30_initializations
  require_artifact: true
```

The bank has 30 `episode-seed-*.json` files plus
`initialization_manifest.json`. Each artifact fixes the pre-controller local
votes and initial evidence-sharing decisions for one repetition. Reusing it
matches starting populations across rho, budget, and truth/false arms. The
historical `false_control` directory name does not mean the starting state is
scientifically restricted to the false arm.

The Potsdam q=3 configs currently reference a separately located but analogous
bank named `astra_task003_false_control_30x30_potsdam_rho3_initializations`.
Do not silently mix banks. Use the artifact path and identity required by the
chosen anchor config.

## 4. Current scientific design

Shared fixed design:

- Task: `task_003`, frozen Task3 team-allocation world.
- Population: N=24.
- Rounds: 30.
- Repetitions: 30 per cell.
- Complete topology and public votes.
- Reasoning dynamics with blackboard social interaction.
- Receiver disposition: `vigilant`.
- Controller policy: `soft_target`, beta=4.0, theta=0.50.
- Sensor sample size: 12.
- Controller communication: `adaptive_communication`, dawn timing, structured
  LLM communication policy, requests and directives enabled.
- Board messages live one round, self-authored messages are excluded, and
  participants may ask questions.
- Consensus does not stop an episode.
- Paired deterministic initializations and common seeds are reused across the
  parameter grid.

The persistence axis is:

```text
rho = [0.70, 0.85, 1.00]
```

`rho` governs survival of active knowledge. Historical knowledge remains
recorded separately; inactive knowledge can be reactivated only by a valid
later exposure. Do not conflate active and historically known facts.

DeepInfra q=12 grid:

```text
b = [6, 9, 12, 18, 24]
3 rho x 5 b = 15 cells
15 x 30 repetitions = 450 episodes per truth/false arm
```

Potsdam q=3 grid:

```text
b = [3, 6, 9, 12, 18, 24]
3 rho x 6 b = 18 cells
18 x 30 repetitions = 540 episodes per truth/false arm
```

Here `q` is `game.options.social_group_size`: how many blackboard messages are
sampled into a focal participant's social context. It is not the number of
provider calls or the controller's budget. A focal participant still produces
one decision/update at a time. `b` is the controller intervention budget.

## 5. Proven execution topologies

### DeepInfra

Current anchor:

```text
model: deepseek-ai/DeepSeek-V4-Flash
array shards: 15, one cell per shard
array throttle / active cells: 5
episode slots per shard: 20
study-wide request concurrency ceiling: 100
target RPM: 1000
CPUs per shard: 8
memory per shard: 12G
wall time per shard: 20:00:00
```

This means up to 100 episodes advance concurrently, but each episode has at
most one provider request outstanding at a time. The 40 allocated CPUs at full
occupancy support local validation/serialization; they are not the source of
the 100-way network concurrency.

Observed throughput was often roughly 200–400 RPM despite a 1,000-RPM target.
That was mostly model-response latency, not a low configured ceiling. Do not
increase CPU count merely to chase RPM. Measure ready episode work, occupied
provider leases, response latency, token throughput, and successful RPM.

The q=12 false arm took almost exactly 24 hours across three five-cell waves.
It produced 409/450 completed episodes; 41 ended after strict model-decision
validation was exhausted.

The q=12 truth arm produced 263/450 completed episodes before DeepInfra
returned HTTP 402. The coordinator recorded 178 HTTP 402 responses; 146
episodes stopped with provider errors and 41 with validation exhaustion. This
was an account credit/payment condition, not concurrency overload or a SLURM
coordination failure. Their validated partial trajectories and failure
checkpoints were retained for targeted recovery.

### Potsdam university provider

Current anchor:

```text
model: gwdg/openai-gpt-oss-120b
array shards: 18, one cell per shard
array throttle / active cells: 3
episode slots per shard: 20
study-wide request concurrency ceiling: 60
target RPM: 600
CPUs per shard: 8
memory per shard: 12G
wall time per shard: 16:00:00
```

The university provider and DeepInfra use separate provider coordinators and
credentials. Their limits must not be merged. The Potsdam topology was chosen
for reliability after higher active-cell concurrency caused provider trouble.

Session snapshot on 2026-09-13 at 21:25 CEST:

- Potsdam false arm: 18/18 cell seals present.
- Potsdam truth arm: 15/18 cell seals present; the last three shards were
  running.

## 6. Provider coordination and failure semantics

All workers for one provider study share one adaptive provider controller. It
tracks leases, successful/failed requests and recent dispatches; begins at the
planned safe ceiling; decreases on retryable service failures; and recovers
gradually after stable responses.

Important distinctions:

- **HTTP 429/5xx, timeouts, dropped connections:** retryable provider/service
  conditions. Keep retrying inside the same episode for the bounded provider
  retry window (currently five minutes) and checkpoint at the failure boundary.
- **HTTP 402:** payment/credit refusal. Lower concurrency does not fix it.
- **Malformed semantic decision:** model returned an invalid ballot, fact ID,
  or structured response. Send the contract-authored correction prompt and
  require a complete replacement. Never coerce invalid identifiers.
- **SLURM completion:** only says the worker process exited cleanly. It does
  not prove all episodes completed. Inspect episode status and cell seals.

Mid-episode failure checkpoints preserve the validated trajectory prefix,
runtime state, usage and failed-call identity. Recovery must reuse valid
completed episodes and continue/retry only missing or interrupted identities.
Do not wipe a study root to recover provider failures.

Two workers must never own the same episode concurrently. Completion should be
published atomically: retained records first, then the episode/cell completion
seal.

## 7. Retention and dashboard observability

The current Task003 studies use `artifact_profile: dashboard_semantic`, not the
leanest `results_only` profile. This intentionally retains enough semantic
state for the hierarchical dashboard:

- study -> cells -> episodes;
- vote/target/truth trajectories;
- blackboard messages and controller communications;
- active/historical epistemic state where emitted;
- three prompt examples per cell for beginning/middle/end inspection;
- exact structured decision metadata without exposing credentials.

The dashboard is operational observability, not scientific identity. A
dashboard label such as `control` is source provenance and does not mean the
controller necessarily identifies itself that way in the agent prompt.

Do not enable uncontrolled full prompt/provider logging for every interaction.
The chosen semantic profile was a compromise between live inspection and
storage size.

## 8. Submission workflow

On Potsdam, always use the dedicated environment:

```bash
/home/ojedamarin/.local/share/miniforge3/bin/conda run -n MA-CC python ...
/home/ojedamarin/.local/share/miniforge3/bin/conda run -n MA-CC mas-cc ...
```

This absolute Conda path is Potsdam-specific. A new cluster must use its own
installed project environment and must not reproduce the Potsdam path.

Before a real launch:

1. Copy and validate the frozen input bundle.
2. Change only deployment paths and provider secrets/settings required by the
   new cluster.
3. Verify `mas_cc`, `pandas`, and `pyarrow` imports.
4. Run experiment preflight and inspect resolved cells, episodes, calls,
   tokens, cost, storage and time estimates.
5. Ensure the result root and scheduler logs are on durable shared storage.

Standard submission:

```bash
mas-cc study submit --config-dir <study-config-folder>
```

The generic launcher is:

```text
scripts/Potsdam/SLURM/run_study_cell_array.job
```

Do not create one `.job` file per study. On another scheduler, implement one
generic adapter that consumes the generated execution manifest. The YAML and
scientific plan remain authoritative.

Potsdam has one deployment-specific idiosyncrasy: the job runs with the source
repository as working directory so `.env` and repository-relative task data
resolve consistently, while results and SLURM logs are written under `/work`.
Do not copy that absolute layout into another deployment; preserve the
separation using that platform's own shared-storage paths.

## 9. Aggregation and the scientific handoff

The user-facing command remains one command:

```bash
mas-cc study aggregate --study-dir <study-result-root>
```

On the current Potsdam installation, the `auto` backend submits an internal
detached SLURM graph:

```text
prepare immutable canonical input
        -> parallel per-cell information/resampling workers
        -> one finalizer
        -> final analysis tree and ZIP
```

Relevant implementation files:

```text
src/mas_cc/studies/analysis_slurm.py
src/mas_cc/studies/analysis_worker.py
scripts/Potsdam/SLURM/run_study_analysis.job
src/mas_cc/studies/aggregation.py
tests/mas_cc/test_analysis_slurm.py
```

This parallelization changes computation placement only. It must preserve
estimator mathematics, deterministic cell seeds, compact result ordering and
the final delivery contract. It does not require one permanent CPU per cell.

Canonical final data are compressed Parquet tables, especially:

```text
analysis/tables/cells.parquet
analysis/tables/episodes.parquet
analysis/tables/rounds.parquet
analysis/tables/micro_slots.parquet
analysis/tables/primary_estimates.parquet
analysis/tables/information_estimates.parquet
analysis/tables/support_diagnostics.parquet
analysis/tables/derived_observables.parquet
```

Configured epistemic, causal-response, thermodynamic and blackboard tables and
plots are also emitted. Bootstrap confidence intervals and permutation/null
procedures remain scientifically unchanged, but individual draws and analysis
caches are transient. The standard ZIP excludes run trees, provider logs,
SLURM logs, `.resume`, raw draws, persistent caches and redundant CSV mirrors.

Use strict aggregation only after every expected cell and episode is valid:

```bash
mas-cc study aggregate --study-dir <study-result-root>
```

Use this only for an explicitly exploratory partial snapshot:

```bash
mas-cc study aggregate --study-dir <study-result-root> --allow-incomplete
```

Partial/censored prefixes may support interruption diagnostics, but established
MI/CMI, bootstrap, takeover and final-state estimators consume completed
episodes only. Never describe a partial package as the final study.

## 10. Analysis invariants

- MI, CMI, bootstrap, randomization/permutation nulls, signed response,
  susceptibility, information efficiency, eta_IR, eta_th, currents, effective
  affinity, compliance and support diagnostics use the existing estimators.
- Do not implement a second CMI estimator in a deployment backend.
- State-local and cell-level products use canonical scientific `cell_id`, not
  array indices or directory labels.
- No heterogeneous study-wide pooled estimator should be presented as a
  physical observable.
- `theoretical_reference: none` is required for these finite-persistence
  blackboard studies; do not apply the old q=1 non-persistence theory.
- Support diagnostics and empty state bins are first-class. Do not fabricate
  observations for unvisited population states.
- Future observables should be recomputed from canonical cells/episodes/rounds/
  micro-slots rather than by retaining large estimator caches.

## 11. New-cluster recommendation

Prefer the simplest architecture that can keep the provider safely occupied:

1. Start on one durable 20–30-CPU host if it can sustain the required logical
   episode tasks and network requests.
2. Separate CPU process count, logical episode concurrency, provider request
   concurrency, token throughput and RPM; they are not interchangeable.
3. Use async tasks or threads for waiting provider calls and a bounded process
   pool for local CPU work. Do not require one CPU per outstanding request.
4. Use one global provider controller per provider/account.
5. Add Redis or another network coordinator only if workers span hosts without
   a reliable shared coordination primitive.
6. Use Kubernetes only if it is already the supported/simple platform or if
   multi-host elasticity and replacement justify its overhead. Prefer bundled
   worker jobs over one pod per episode.
7. Keep canonical output storage and aggregation scheduler-agnostic.

Acceptance should demonstrate identical resolved cells and episode seeds,
zero duplicate ownership, atomic completion, safe restart of missing work,
compatible canonical tables, numerically equivalent estimators, and provider
limits never exceeded.

## 12. Repository changes from this session to preserve

Before deploying from another Git revision, ensure it contains the relevant
work represented by these paths:

```text
.codex/skills/ma-cc-study-workflow/SKILL.md
AGENTS.md
scripts/Potsdam/SLURM/run_config_array.job
scripts/Potsdam/SLURM/run_study_cell_array.job
scripts/Potsdam/SLURM/run_study_analysis.job
src/mas_cc/cli/main.py
src/mas_cc/games/relational_reasoning/imitation_round_feedback/controller.py
src/mas_cc/games/relational_reasoning/imitation_round_feedback/runtime.py
src/mas_cc/studies/aggregation.py
src/mas_cc/studies/analysis_slurm.py
src/mas_cc/studies/analysis_worker.py
tests/mas_cc/test_analysis_slurm.py
tests/mas_cc/test_selective_truth_controller.py
```

The controller change permits the frozen truthful controller evidence pool to
be used when the configured target is the task's correct relation, while still
rejecting arbitrary mismatched targets. The runtime work preserves recoverable
mid-episode failures. The analysis additions provide detached, parallel,
single-command SLURM aggregation.

Do not assume that a locally present uncommitted file exists on the new
cluster. Commit and push the chosen coherent revision, then record that commit
in the study provenance before launch.

