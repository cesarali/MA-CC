# MAS-CC Phase 9 — Experiment Orchestration Plan

**Status:** proposed, replaces Phase 9 of the V6 plan
**Revision date:** 2026-08-03
**Relative to:** `02082026_MAS_CC_Reorganization_and_Validation_Plan_v3_Final_Metrics_Recording_Revision_V6.md`
(Phases 1–8.0 there remain the adopted, frozen record of what is already built.)
**Target package:** `mas_cc`
**Primary principle:** an experiment is many episodes of one game, run in parallel,
priced before they run, and legible on the console while they run.

---

## 0. Why this document exists

V6's Phase 9 bundles three things into one phase: (a) running many episodes of a
game concurrently, (b) a committee/control-policy layer, and (c) binding
preflight approval to offline-analysis provenance (extractor versions, planned-
analysis manifests) that Phase 8.1 was supposed to have defined first. In
practice:

- **8.1 is its own project.** Control policies and information-theoretic
  offline analysis (MI, conditional MI, InfoNCE, null models) have their own
  validation story — synthetic fixtures, estimator diagnostics — independent of
  whether episodes can run concurrently. It is deferred, not deleted; see
  [§5](#5-deferred-not-deleted-relationship-to-81).
- **"Committee" has no settled definition yet.** The working idea — a
  committee is just extra per-game agents running an alternate, game-defined
  decision rule, with their own metrics — is real but not designed. Phase 9
  does not need it: an experiment of ordinary episodes is useful on its own,
  and nothing below assumes committees exist.
- **The V6 Phase 9 tasks are hard to act on.** Preflight bound to
  "extractor versions" and "planned-analysis manifests" that don't exist yet
  is exactly the kind of undifferentiated abstraction the user has flagged
  before (see the `user-over-architecting-tendency` memory) — this plan trims
  Phase 9 down to what the current codebase can actually support.

This document is the sole authority for Phase 9. It supersedes the "Phase 9"
section of V6 without modifying that file.

---

## 1. Scope

**In scope:**

1. `Experiment` = N `Episode` runs of one resolved game + prompt + provider
   configuration, executed concurrently, checkpointed per episode, resumable.
2. A preflight command that estimates total provider calls, tokens, cost, and
   runtime for the *whole* experiment before any provider call is made, and
   fails closed the same way single-episode preflight already does.
3. Console output the user can actually watch: a short banner describing what
   is about to run, then a live progress bar over episodes (and, inside it,
   over rounds) as they complete.
4. Aggregated results written under the existing `results/<game>/<experiment>/<run_id>/`
   convention (`storage/results.py`), one shard per episode plus an
   experiment-level summary.

**Out of scope (this phase does not implement or block on):**

- Committee/control policies, forced actions, intervention schedules.
- Offline information-theoretic analysis (MI/CMI/InfoNCE/null models).
- Experiment *grids* over multiple game/prompt/provider configurations —
  Phase 9 runs one resolved configuration N times. A grid sweep is a thin
  wrapper over this (loop the CLI, or a follow-up phase) and is not designed
  here.
- Binding preflight approval to recording-plan/extractor/analysis fingerprints.
  Phase 9 preflight binds to what Phase 9 actually touches: resolved config,
  prompt definition hash, pricing snapshot, episode count.

---

## 2. What already exists (reuse, do not rebuild)

The repository already has most of the hard parts. Phase 9's job is composition,
not invention.

| Need | Existing piece |
| --- | --- |
| Run one episode of a game | `games/runner.py::run_game` |
| Per-episode reproducible seeding | `core/random.py::Seed.derive(namespace)` |
| Typed experiment/run identifiers | `core/ids.py::ExperimentId`, `RunId` |
| Repetition + concurrency config | `config/models.py::ExecutionConfig.repetitions`, `.parallelism` |
| Experiment identity (name/description/tags) | `config/models.py::ExperimentConfig` |
| Per-call cost/token estimate, budget fail-closed | `planning/preflight.py::static_preflight` |
| Per-episode call-plan cost estimate (lower/expected/max) | `planning/game_preflight.py::static_game_preflight` |
| Results directory convention | `storage/results.py::results_run_dir` (defined, **not yet wired into a live run command** — this phase is what wires it) |
| Atomic per-episode checkpointing | `storage/checkpoints.py::AtomicCheckpointStore` (currently scoped to rounds *within* one episode; Phase 9 adds an episode-shard layer, see §4.3) |
| Streaming/final metrics per episode | `metrics/base.py`, `games/<game>/metrics.py` |
| Audit/console logging conventions | `observability/audit.py`, `config/models.py::LoggingConfig` |
| tqdm dependency | already in `pyproject.toml` |
| Precedent for the exact UX being asked for | `src/naming_game/empowerment_experiment.py::run_experiment` — banner log lines, `tqdm.auto` episode + round bars, `asyncio.Semaphore` concurrency, atomic per-episode parquet shards with resume-by-existence-check. This is frozen legacy code and is not imported, but its shape is the reference for §4.4. |

Nothing in this table needs a new abstraction layer. `static_game_preflight`
already prices *one* episode's full call plan (lower/expected/maximum,
per-decision-stage); an experiment estimate is that number multiplied by
episode count, not a new pricing model.

---

## 3. Core concepts

### 3.1 Episode

One episode is exactly what `run_game` already produces: a `GameResult` from
one seeded playthrough of a resolved game config. Phase 9 adds nothing to the
episode's internal execution — the game/prompt/provider/metrics stack from
Phases 3–8.0 is untouched.

### 3.2 Experiment

An experiment is:

- one resolved `RunConfig` (game + prompt + provider + execution + storage +
  metrics), the same shape already validated by Phases 2–8.0;
- a count of episodes to run — `config.execution.repetitions`;
- a concurrency bound — `config.execution.parallelism`;
- a deterministic per-episode seed: `Seed(config.execution.seed).derive(f"episode:{i}")`
  for `i in range(repetitions)`, so re-running the same config reproduces the
  same per-episode seeds regardless of completion order or concurrency;
- an `ExperimentId`/`RunId` pair identifying where results land:
  `results/<game>/<experiment.name>/<run_id>/`.

No new config section is required beyond what `ExecutionConfig` and
`ExperimentConfig` already declare. `episode_id` is `f"{run_id}-{i:04d}"`.

### 3.3 Two commands, matching the existing `provider test` / `game run` pattern

```bash
mas-cc experiment preflight --config configs/runs/<name>.yaml --output-dir inspection/...
mas-cc experiment run       --config configs/runs/<name>.yaml --output-dir results/... [--resume]
```

`experiment run` refuses to launch a paid provider without either a passing
preflight in the same process or `--approve-preflight <preflight_id>` — same
fail-closed posture as Phase 4's per-call budget guard, applied once at the
experiment level instead of re-derived per phase.

---

## 4. Design

### 4.1 Experiment preflight

New: `planning/experiment_preflight.py`.

```python
@dataclass(frozen=True, slots=True)
class ExperimentPreflightEstimate:
    game_type: str
    provider: str
    model: str
    episode_count: int
    per_episode: GamePreflightEstimate      # unchanged, from game_preflight.py
    total_provider_requests: EstimateRange  # per_episode.provider_requests * episode_count
    total_input_tokens: EstimateRange
    total_output_tokens: EstimateRange
    total_costs: MonetaryEstimateRange
    rough_runtime_seconds: float            # accounts for config.execution.parallelism
    launch_status: str                      # same values as PreflightEstimate
    warnings: tuple[str, ...]

def static_experiment_preflight(
    plan: GameCallPlan,
    config: RunConfig,
    *,
    pricing_quote: PricingQuote | None = None,
    system_budget: BudgetLimits | None = None,
    run_budget: BudgetLimits | None = None,
    explicit_override: bool = False,
    allow_stale_pricing: bool = False,
) -> ExperimentPreflightEstimate: ...
```

Implementation is multiplication, not re-derivation: call
`static_game_preflight` once, multiply request/token/cost ranges by
`config.execution.repetitions`, and re-check the multiplied conservative cost
against the *same* budget resolution path `static_preflight` already uses (so
a budget ceiling means the same thing whether you're pricing one call or one
experiment). Runtime estimate divides total expected calls by
`config.execution.parallelism` (bounded by provider concurrency limits, same
as `runtime_estimation.py` already does per-call).

No network I/O, no model load, no credentials — same guarantee as the
existing preflight path, just scaled up. This produces the "diagnosis of how
much this is going to cost" the user asked for, before touching University or
any paid provider.

### 4.2 Experiment orchestrator

New: `experiments/orchestrator.py`.

```python
async def run_experiment(
    config: RunConfig,
    provider: LLMProvider,
    output_dir: str | Path,
    *,
    resume: bool = True,
    show_progress: bool = True,
) -> ExperimentResult: ...
```

- Builds `run_id`, resolves `results_run_dir(...)`.
- Derives `repetitions` per-episode seeds via `Seed.derive`.
- Runs episodes under `asyncio.Semaphore(config.execution.parallelism)`,
  each episode calling the existing `run_game`.
- Each completed episode is written atomically (temp file + rename) as its own
  shard under `data/episodes/<episode_id>.*` before being counted as done —
  same pattern as the legacy orchestrator's parquet shards, just pointed at
  the mas_cc storage/metrics writers instead of ad hoc parquet.
- On resume, an episode already present on disk is skipped, not re-run.
- After all episodes complete, writes an experiment-level summary
  (`experiment_summary.csv`/`.json`): episode count, completed/failed counts,
  aggregate metric values (reusing whatever `FinalMetric`s the game already
  declares), total actual calls/tokens/cost from the provider's own
  accounting, and a comparison against the preflight estimate.
- `config.execution.fail_fast` controls whether one episode's exception
  cancels the remaining episodes or is recorded and skipped.

Nothing here reaches into game internals; the orchestrator's only contract
with a game is the same `Game` ABC every other phase already uses.

### 4.3 Checkpointing

`storage/checkpoints.py::AtomicCheckpointStore` today checkpoints *rounds
within one episode*. Phase 9 adds a second, independent checkpoint layer at
the *episode* level: an episode is "done" purely by the existence of its
completed shard files (interactions + episode summary), matching the legacy
orchestrator's resume-by-existence-check rather than a separate manifest file
that could drift from the data. The two layers compose but don't share a
schema: within-episode round checkpoints remain a `run_game` concern, and
experiment-level resume is a directory-listing concern in the orchestrator.

### 4.4 Console UX

This is a stated requirement, not a nice-to-have, so it's specified concretely
rather than left to "add some logging later."

**Banner**, printed once at experiment start (to stdout, and mirrored into
`logs/experiment.log` if `config.logging.audit` is enabled):

```
Experiment: naming-convention-paper-w2-gpt54nano
  Game:          naming_convention v3
  Provider:      university / gpt-5.4-nano
  Episodes:      10  (parallelism: 4)
  Prompt:        naming_convention_decision v3  [def:a91f... ]
  Budget:        max $12.00 (run-specific)
  Preflight:     expected $3.42 / conservative $7.10  — within budget
```

**Progress**, via `tqdm.auto`, two bars exactly as the legacy precedent does
it (position 0 = episodes, position 1 = current-round-within-active-episode),
active only when `show_progress=True` (the CLI's default; disabled under
`--no-progress` or when stdout isn't a TTY, so CI logs stay clean):

```
Episodes:  40%|████        | 4/10 [02:15<03:22, episode]
Rounds:    62%|██████▏     | 31/50 [round 12/12 | episode ep-0004]
```

`show_progress=False` still logs one line per completed episode
(`LOGGER.info("episode %s complete: %d rounds, %.2fs", ...)`) so the run
remains legible without a TTY.

### 4.5 Approval and revalidation

Kept minimal relative to V6: approval binds to the resolved config hash,
prompt definition hash, and pricing snapshot version — the things Phase 9
actually estimates. It does **not** bind to recording-plan/extractor/analysis
fingerprints, because Phase 9 doesn't produce those. If 8.1 lands later and an
experiment needs to declare control-policy or analysis requirements, that's an
additive field on the same approval record, not a redesign of it.

---

## 5. Deferred, not deleted: relationship to 8.1

Nothing in this plan forecloses 8.1. The pieces that would eventually plug in:

- A control policy wraps `Game.select_participants` / `construct_observations`
  / `build_decision_requests` — all already-generic hooks on the `Game` ABC.
  The orchestrator calls a `Game`; it does not care whether that `Game` is
  wrapped by a control policy.
- Offline analysis reads completed episode shards from
  `results/<game>/<experiment>/<run_id>/data/`. Phase 9 already writes that
  data in a stable, versioned-enough shape (bound prompt hashes, per-round
  metric values) for 8.1 to consume later without a Phase 9 rewrite.

When 8.1 is designed, it should be designed against what Phase 9 actually
produces, not the other way around.

---

## 6. Tasks

1. `planning/experiment_preflight.py`: `ExperimentPreflightEstimate`,
   `static_experiment_preflight` (multiply `GamePreflightEstimate` by
   `repetitions`, re-check budget, estimate runtime under `parallelism`).
2. `experiments/orchestrator.py`: `ExperimentResult`, `run_experiment`
   (semaphore-bounded concurrent episodes, per-episode seed derivation,
   atomic shard writes, resume-by-existence, fail-fast toggle).
3. Wire `storage/results.py::results_run_dir` into `run_experiment` as the
   actual output location (first live use of that function).
4. Experiment-level summary writer: aggregate `FinalMetric` values across
   episodes, actual-vs-estimated call/token/cost comparison.
5. Console UX: banner formatter + `tqdm.auto` dual progress bars +
   no-progress fallback logging, per §4.4.
6. CLI: `mas-cc experiment preflight` and `mas-cc experiment run`
   (`--config`, `--output-dir`, `--resume`/`--no-resume`,
   `--approve-preflight`, `--no-progress`), following the existing
   `provider test` / `game run` argument conventions in `cli/main.py`.
7. Tests: seed determinism across concurrency levels, resume skips completed
   episodes and only completed episodes, fail-fast vs. continue-on-error,
   preflight multiplication matches N independent `static_game_preflight`
   calls, budget fail-closed at experiment scale mirrors single-call
   fail-closed behavior.

---

## 7. Inspection commands

```bash
conda run -n MA-CC mas-cc experiment preflight \
  --config configs/runs/naming_convention_paper_w2_gpt54nano_v3.yaml \
  --output-dir inspection/phase_09/preflight

conda run --live-stream -n MA-CC mas-cc experiment run \
  --config configs/runs/naming_convention_paper_w2_gpt54nano_v3.yaml \
  --approve-preflight inspection/phase_09/preflight/preflight_id.txt \
  --output-dir results
```

## 8. Inspection artifacts

```text
inspection/phase_09/
├── report.md
├── manifest.json
└── preflight/
    ├── resolved_config.yaml
    ├── per_episode_estimate.json
    ├── experiment_estimate.json
    ├── pricing_snapshot.json
    ├── budget_status.json
    └── preflight_id.txt

results/<game>/<experiment>/<run_id>/
├── manifest.json
├── resolved_config.yaml
├── logs/experiment.log
├── checkpoints/               # per-episode shard existence = resume state
├── metrics/{streaming.csv,final.csv,plots/}
├── data/episodes/<episode_id>.*
├── experiment_summary.csv
└── experiment_summary.json    # includes estimated-vs-actual comparison
```

## 9. Manual checks

- Run preflight twice; confirm identical output with no provider/network access.
- Run an experiment with the mock provider, kill it mid-run, resume it; confirm
  completed episodes are skipped and per-episode seeds are unchanged.
- Confirm the banner and progress bars show the right episode count, provider,
  and budget before any call is made.
- Compare `experiment_summary.json`'s actual totals against the preflight
  estimate.
- Force a budget-exceeding config and confirm `experiment run` refuses to
  launch without an explicit override, exactly as single-call preflight does.

## 10. Acceptance criterion

An experiment of N episodes over one resolved game/prompt/provider
configuration can be priced before launch without provider access, launched
with bounded concurrency, watched on the console with a clear banner and
live progress, interrupted and resumed without re-running completed episodes,
and summarized with an actual-vs-estimated comparison — using only components
that exist today plus the composition described above.
