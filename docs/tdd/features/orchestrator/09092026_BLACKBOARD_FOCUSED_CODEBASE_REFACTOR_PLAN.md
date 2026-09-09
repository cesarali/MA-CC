# Blackboard-focused codebase refactor

Status: proposed implementation plan; no runtime changes or deployments.

## Objective

Make the supported application focus on the blackboard game, frozen task
generation and validation, study execution, dashboard inspection, and scientific
aggregation. Reduce installed dependencies and maintenance burden while keeping
Potsdam SLURM and DeepInfra operational. Deployment remains independent of game
semantics and estimator mathematics. Prefer one ordinary Python package and the
existing launch mechanisms; Kubernetes is optional future work.

## Findings from the current repository

- Blackboard is a mode of
  `games/relational_reasoning/imitation_round_feedback/`, rather than an isolated
  game package. Current configs use `relational_imitation_round_feedback`.
- Its prompts import JSON extraction from HiddenBench; its game/runtime import
  population observables and controller constants from HiddenBench; its
  controller imports HiddenBench controller helpers.
- The relational analysis, currents, and `studies/aggregation.py` depend on
  `games/hidden_bench/imitation_round_feedback/analysis.py`. That file contains
  an authoritative estimator engine, not merely obsolete game implementation.
- `games/registry.py` registers old games and eagerly imports their prompt
  factories. `cli/main.py` also imports HiddenBench and exposes historical tools.
- MuSR generation spans `musr_team_allocation_generator/` and
  `probes/musr_truthful_selective/`. The latter imports helpers from
  `probes/musr_prompt_solvability/`; deleting probe directories indiscriminately
  would break the retained workflow.
- `pyproject.toml` still describes a naming-game benchmark, exposes the
  `naming-game` entry point, discovers `src/naming_game`, and requires torch and
  sentence-transformers in the base installation. Their actual retained uses
  must be traced before removing dependencies.

Therefore extract shared dependencies before deleting old game families.
An import search is a starting inventory, not proof that dynamic registrations,
CLI branches, fixture dependencies, or serialized identities are unused.

## Supported scope

Keep:

- Current blackboard runtime, participant/controller prompts and strict repair
  contracts, persistence, initialization, communication modes, and metrics.
- MuSR task creation, symbolic validation, evidence generation, distribution,
  calibration and frozen dataset loading. Preserve licenses and attribution.
- Generic config, seeds/identities, provider runtime, budgets, coordination,
  retry behavior, retention, studies, extension/resume, and SLURM launchers.
- Empirical MI/CMI variants, causal response, uncertainty/null summaries,
  efficiencies, currents, support diagnostics, plots and packaging.
- Hierarchical dashboard, semantic logging, prompt examples, and offline views.
- DeepInfra and the Potsdam provider adapter, plus a lightweight fake provider
  for deterministic validation. Audit other provider adapters individually.

Removal candidates after dependency extraction:

- `src/naming_game`, naming-specific CLI/entry point and scripts.
- HiddenBench, naming, and other unused production game registrations/modules.
- Non-board relational runtime branches and the old spatial task generator
  once retained schemas and dataset loading no longer depend on them.
- Unused benchmark/probe commands, configs, launch scripts, and dependencies.
- Obsolete theory branches only after separating them from empirical analysis
  and recording which existing analysis recipes require them.

Retain small synthetic estimator fixtures in tests even when synthetic games
leave the public application. Remove neither scientific fixtures nor historical
datasets solely because their originating game is no longer supported.

## Stage 1 — Freeze contracts and inventory dependencies

Work in an isolated checkout with its own environment. Active Potsdam jobs use
the existing repository/environment and must not see files moved underneath
them, including files imported lazily after a shard starts.

Inventory imports, registrations, CLI commands, scripts, test fixtures,
scientific identity hashes and serialized module references. Produce an explicit
keep/extract/remove table before deleting code. Include runtime discovery paths.

Capture compact baselines for representative blackboard conditions: false,
truth and no control; report and adaptive communication; q=1 and q=3; persistence
below one and at one. Reuse existing deterministic fixtures and frozen artifacts.
Capture canonical observations, prompt hashes, seeds, estimator summaries,
completion manifests and dashboard responses. Do not make paid calls for this.

Deliverable: dependency map, removal inventory and scientific regression fixtures.

## Stage 2 — Extract shared scientific helpers

Move the existing implementations, preserving algorithms and call ordering:

- JSON response extraction into the shared response/validation layer.
- Population observables into the existing metrics layer.
- Shared controller constants and policy primitives into the control layer.
- Round information estimation and support/resampling helpers into the existing
  analysis layer; keep one authoritative implementation.
- Shared probe I/O and configuration helpers into a small common generation
  support module where a real shared dependency exists.

Update imports, registries and estimator provenance together. Moving Python
modules must not silently change scientific protocol fingerprints, seed streams,
estimator versions, prompt hashes, or eligibility for study-extension reuse.
If persisted references need adaptation, add one documented boundary adapter;
do not keep duplicate engines. Verify empirical execution with
`theoretical_reference: none` independently of optional theory imports.

Deliverable: blackboard and aggregation run without imports from retired games.

## Stage 3 — Simplify the application surface and installation

Keep the existing package name `mas_cc` and current blackboard game identifier
initially. Renaming directories and config keys simultaneously with dependency
removal makes reuse and provenance unnecessarily difficult.

Narrow the default registry and CLI to supported workflows. Delete obsolete
production code after the dependency gate passes. Preserve historical code in
Git history with a recorded release commit; do not ship an archive tree or a
second legacy application inside the package. Historical configs should either
remain supported or point clearly to their reproducible historical revision.

Audit runtime dependencies. Remove unused ones; put heavy optional local-model
dependencies behind extras if still supported. Keep scientific numerical and
Parquet dependencies required for analysis. Verify a clean installation can
import and execute blackboard workflows without optional GPU/embedding packages.
Update the package description, entry points, README, examples and dependency
files. A distribution-name migration is separate and unnecessary for this step.

Deliverable: lean installation and a small, explicit supported command surface.

## Stage 4 — Establish simple boundaries for portability

Keep the existing directories where practical:

```text
mas_cc/
  games/relational_reasoning/       supported blackboard implementation
  musr_team_allocation_generator/   frozen task generation and validation
  probes/                          retained calibration workflows only
  core/ config/ control/ metrics/  shared scientific contracts
  llm_runtime/                     providers, validation, load control
  experiments/ runtime/ studies/  episode execution and study orchestration
  storage/ analysis/               canonical data and scientific results
  blackboard_dashboard/ cli/       inspection and user commands
```

Scheduler concerns stay at submission/worker boundaries. Games consume frozen
tasks and provider interfaces; analysis consumes canonical observations. Neither
should import SLURM, Kubernetes, or deployment-specific absolute paths.

Preserve Potsdam's dedicated MA-CC environment instructions and generic SLURM
scripts. A future single-server runner can reuse episode identities and workers;
Kubernetes, Redis, object storage, and a new distributed queue are not prerequisites
for this cleanup. Do not combine a concurrency redesign with scientific code
extraction. CPU allocation, active episodes and provider concurrency remain
separate controls; comparable throughput on fewer CPUs needs measurement.

Deliverable: documented execution and storage boundaries with the current
SLURM implementation still working.

## Stage 5 — Verify and cut over

Required acceptance checks:

1. Clean install, CLI discovery, frozen task loading, and generator validation.
2. Deterministic game parity: ballots, board events, active knowledge, controller
   actions, prompt/repair contracts and seeds match the captured baselines.
3. Canonical tables and retained scientific fields remain compatible.
4. Estimator variants, bootstrap intervals, null summaries, support, currents,
   affinities and causal response match established numerical tolerances.
5. Aggregation and plots work from retained completed data without provider calls;
   dashboard study/cell/episode navigation and semantic content still work.
6. Interrupted execution and study extension preserve valid completed episodes,
   retain scientific identities and schedule only the intended missing work.
7. Provider retry, concurrency and RPM coordination tests continue passing.
8. A generic SLURM fake-provider smoke passes in an isolated result directory.
   A paid smoke or production relaunch needs separate operational authorization.

Keep changes reviewable: separate commits for baselines, shared-helper moves,
registry/CLI pruning, deletions, dependency cleanup and documentation. Record the
validated release/environment for future jobs. Leave active runs on their current
revision until completion; completed result trees and task hashes remain intact.

Deliverable: validation report, removed-path list, final supported commands,
installation instructions, measured dependency reduction, and rollback revision.

## Completion criteria and limits

The reduced application supports task generation through final analysis and
dashboard inspection using existing scientific definitions and retained data.
Potsdam remains operational; deployment adapters can be added independently.
No new estimator, game mechanism, seeding framework, execution architecture,
legacy archive package, or mandatory cluster platform is introduced by this plan.

This is a staged refactor rather than a folder deletion. The shared HiddenBench
estimator/controller dependencies are the first critical extraction to validate.
