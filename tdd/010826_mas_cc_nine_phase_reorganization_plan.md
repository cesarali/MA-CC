# MAS-CC Reorganization and Validation Plan

**Status:** implementation plan  
**Target package:** `mas_cc`  
**Legacy package:** `naming_game` remains operational and unchanged during the migration  
**Maximum phases:** 9  
**Primary principle:** every phase must be independently runnable and manually inspectable

---

## 1. Purpose

The immediate goal is not to rewrite the existing `naming_game` package. The goal is to construct a clean successor package, `mas_cc`, in parallel with it.

The successor should support:

- multiple language games;
- several interchangeable LLM backends;
- compositional “Lego-like” prompts;
- reproducible experiment configurations;
- cost and runtime estimation before execution;
- local and remote execution;
- detailed logging and audit traces;
- Comet ML monitoring;
- committee/control interventions;
- offline information-theoretic analysis;
- complete experiment orchestration.

The architecture must distinguish the following objects.

### 1.1 Game

A **game** defines the dynamical system:

- agent state;
- information available to each agent;
- interaction topology;
- pair or group selection;
- prompts used during an interaction;
- legal actions;
- state transitions;
- payoff or success rules;
- stopping conditions.

Examples may include:

```text
games/
├── naming_basic/
├── naming_convention/
├── hidden_bench/
├── epistemic_signaling/
└── future_game/
```

`naming_convention` means the specific Naming Convention Game. It is not a general synonym for all games.

### 1.2 Experiment

An **experiment** specifies how a game is studied:

- selected game;
- selected LLM provider and model;
- population sizes;
- committee sizes;
- policies;
- seeds;
- repetitions;
- horizons;
- parallelism;
- logging;
- checkpoints;
- outcome definitions;
- estimators.

A game should be executable independently of an experiment grid.

### 1.3 LLM provider

An **LLM provider** is the adapter that executes a provider-independent request using:

- OpenAI;
- the University of Potsdam proxy;
- local Gemma;
- a deterministic mock backend.

The directory should therefore be named `llm_providers/`, not merely `providers/`.

### 1.4 Prompt

A prompt is a composition of reusable blocks:

```text
task description
+ game rules
+ private information
+ agent memory
+ current interaction
+ decision instruction
+ output contract
```

The same compiled prompt must be sent through every LLM provider.

### 1.5 Control policy

A control policy defines when and how a committee or another intervention mechanism modifies agent decisions. It is not part of the provider and should not be hard-coded into the base game engine.

---

## 2. Non-negotiable implementation rules

1. `src/naming_game/` is not modified during Phases 1–9 except for an explicitly approved critical bug fix.
2. Importing `mas_cc` must not:
   - load Gemma;
   - create an API client;
   - read credentials;
   - start a run;
   - open Comet.
3. Game code must not contain provider-specific conditionals.
4. LLM providers must not contain game logic or agent memory.
5. Prompts must be versioned.
6. Resolved run configurations must be saved with every run.
7. Real secrets belong only in `.env` or an external secret manager.
8. `.env.example` contains variable names and placeholders only. It must never contain a real OpenAI, University, Hugging Face, or Comet key.
9. Every phase must produce:
   - one human-readable report;
   - one machine-readable manifest or summary;
   - one or more phase-specific artifacts.
10. A phase is complete only after its inspection command and acceptance tests pass.

---

## 3. Proposed repository structure

```text
repository/
├── src/
│   ├── naming_game/                 # unchanged legacy package
│   └── mas_cc/
│       ├── __init__.py
│       │
│       ├── core/
│       │   ├── ids.py
│       │   ├── records.py
│       │   ├── exceptions.py
│       │   ├── random.py
│       │   └── validation.py
│       │
│       ├── llm_providers/
│       │   ├── protocols.py
│       │   ├── requests.py
│       │   ├── responses.py
│       │   ├── capabilities.py
│       │   ├── registry.py
│       │   ├── pricing.py
│       │   ├── budget.py
│       │   └── adapters/
│       │       ├── mock.py
│       │       ├── openai.py
│       │       ├── university.py
│       │       └── gemma_local.py
│       │
│       ├── prompts/
│       │   ├── blocks.py
│       │   ├── composer.py
│       │   ├── context.py
│       │   ├── contracts.py
│       │   ├── registry.py
│       │   ├── versions.py
│       │   └── plugins/
│       │       └── basic_binary_choice.py
│       │
│       ├── games/
│       │   ├── protocols.py
│       │   ├── registry.py
│       │   ├── naming_basic/
│       │   ├── naming_convention/
│       │   └── toy_coordination/
│       │
│       ├── control/
│       │   ├── protocols.py
│       │   ├── committees.py
│       │   ├── policies.py
│       │   ├── schedules.py
│       │   └── interventions.py
│       │
│       ├── runtime/
│       │   ├── lifecycle.py
│       │   ├── concurrency.py
│       │   ├── events.py
│       │   └── progress.py
│       │
│       ├── storage/
│       │   ├── schemas.py
│       │   ├── jsonl.py
│       │   ├── parquet.py
│       │   ├── checkpoints.py
│       │   └── artifacts.py
│       │
│       ├── observability/
│       │   ├── logging.py
│       │   ├── audit.py
│       │   ├── heartbeat.py
│       │   └── comet.py
│       │
│       ├── planning/
│       │   ├── call_graph.py
│       │   ├── token_estimation.py
│       │   ├── cost_estimation.py
│       │   ├── runtime_estimation.py
│       │   └── preflight.py
│       │
│       ├── analysis/
│       │   ├── trajectories.py
│       │   ├── discrete_mi.py
│       │   ├── infonce.py
│       │   ├── diagnostics.py
│       │   ├── plots.py
│       │   └── reports.py
│       │
│       ├── experiments/
│       │   ├── specs.py
│       │   ├── grids.py
│       │   ├── episodes.py
│       │   ├── runner.py
│       │   └── committee_empowerment.py
│       │
│       └── cli/
│           ├── main.py
│           ├── inspect.py
│           ├── provider.py
│           ├── game.py
│           ├── analyze.py
│           └── experiment.py
│
├── configs/
│   ├── components/
│   │   ├── llm_providers/
│   │   ├── prompts/
│   │   ├── games/
│   │   ├── logging/
│   │   └── storage/
│   └── runs/
│       ├── provider_smoke_test.yaml
│       ├── toy_game_smoke_test.yaml
│       ├── naming_convention_smoke_test.yaml
│       └── committee_empowerment_pilot.yaml
│
├── tests/
│   ├── naming_game/                 # existing tests
│   └── mas_cc/
│
├── inspection/
│   ├── phase_01/
│   ├── phase_02/
│   └── ...
│
├── .env
└── .env.example
```

---

## 4. Configuration design

Two types of configuration should be kept separate.

### 4.1 Reusable component configurations

These describe reusable components:

```text
configs/components/llm_providers/openai.yaml
configs/components/llm_providers/university.yaml
configs/components/llm_providers/gemma_local.yaml
configs/components/prompts/basic_binary_choice.yaml
configs/components/games/naming_convention.yaml
configs/components/logging/comet.yaml
```

### 4.2 Run configurations

A run configuration composes components into one executable run:

```text
configs/runs/naming_convention_smoke_test.yaml
configs/runs/committee_empowerment_pilot.yaml
```

Every execution must save a fully resolved configuration:

```text
resolved_config.yaml
```

This file must contain all inherited defaults and component references expanded into explicit values, excluding secrets.

---

## 5. Standard manual-inspection contract

Every phase must expose one main inspection command:

```bash
mas-cc inspect phase <N> --output-dir inspection/phase_<NN>
```

The command may delegate to phase-specific subcommands, but it must produce a stable inspection directory.

Every inspection directory should contain at least:

```text
inspection/phase_<NN>/
├── report.md              # what was executed and what to inspect
├── manifest.json          # files, hashes, versions, pass/fail status
└── resolved_config.yaml   # when configuration is involved
```

Additional files depend on the phase.

The phase report should state:

- command executed;
- code paths exercised;
- inputs;
- outputs;
- expected behavior;
- deviations or warnings;
- exact files to inspect manually.

No phase should require searching through an unstructured terminal transcript to understand whether it worked.

---

# Phase 1 — Freeze the legacy package and create the successor shell

## Objective

Create `mas_cc` beside `naming_game` without changing existing behavior.

## Tasks

1. Run and record the complete existing test suite.
2. Save the current Git commit and environment information.
3. Create `src/mas_cc/__init__.py`.
4. Configure package discovery so both packages install from the same repository.
5. Add an initial `mas-cc` CLI with:
   - `mas-cc --help`;
   - `mas-cc version`;
   - `mas-cc inspect phase 1`.
6. Add a guard test verifying that importing `mas_cc` performs no external work.
7. Create the target directories with minimal `__init__.py` files.
8. Do not copy implementation code from `naming_game` yet.

## Inspection command

```bash
conda run -n MA-CC mas-cc inspect phase 1 \
  --output-dir inspection/phase_01
```

## Inspection artifacts

```text
inspection/phase_01/
├── report.md
├── manifest.json
├── environment.json
├── package_imports.txt
└── legacy_test_summary.txt
```

## Manual checks

- `import naming_game` still works.
- `import mas_cc` works.
- `mas-cc --help` works.
- No model is loaded.
- No `.env` file is required for import.
- Existing tests remain unchanged and pass.

## Acceptance criterion

The old package remains operational, and the new package is importable but behaviorally empty.

---

# Phase 2 — Core records and validated configuration system

## Objective

Establish the typed, provider-independent records and the two-level configuration system.

## Tasks

1. Implement immutable core records:
   - IDs;
   - messages;
   - seeds;
   - timestamps;
   - validation results.
2. Implement configuration models for:
   - LLM provider;
   - prompt;
   - game;
   - execution;
   - logging;
   - storage;
   - analysis;
   - experiment.
3. Support reusable component configs and run configs.
4. Implement:
   - YAML loading;
   - environment-variable references;
   - default resolution;
   - schema validation;
   - resolved config export.
5. Explicitly prevent secret values from appearing in the resolved configuration.
6. Add helpful validation errors with the exact invalid field.
7. Add config schema versioning.

## Example provider component

```yaml
schema_version: 1
type: university
model: gwdg/qwen3-30b-a3b-instruct-2507
credentials_env: POTSDAM_API_KEY
base_url_env: BASE_POTSDAM_LLM_URL
timeout_seconds: 60
max_retries: 2
request_concurrency: 10
```

## Inspection command

```bash
conda run -n MA-CC mas-cc inspect phase 2 \
  --config configs/runs/provider_smoke_test.yaml \
  --output-dir inspection/phase_02
```

## Inspection artifacts

```text
inspection/phase_02/
├── report.md
├── manifest.json
├── input_config.yaml
├── resolved_config.yaml
├── config_schema.json
└── validation_examples.md
```

## Manual checks

- Inspect inherited values in `resolved_config.yaml`.
- Confirm that API keys are absent.
- Intentionally run one invalid configuration and inspect the validation message.
- Confirm that the same config loads deterministically.

## Acceptance criterion

All future components can be constructed from validated, versioned configuration objects.

---

# Phase 3 — Compositional prompt system

## Objective

Implement provider-independent prompts as inspectable Lego-like blocks.

## Tasks

1. Implement:
   - `PromptBlock`;
   - `PromptContext`;
   - `PromptComposer`;
   - `PromptInstance`;
   - `ResponseContract`;
   - prompt registry;
   - prompt versioning.
2. Create one example prompt plugin:
   - `basic_binary_choice`.
3. Compose the example from separate blocks:
   - task description;
   - game rules;
   - private state;
   - memory;
   - current interaction;
   - decision instruction;
   - output format.
4. Make block order explicit in YAML.
5. Record token counts per block when a tokenizer is available.
6. Produce both:
   - structured messages as JSON;
   - human-readable rendered prompt text.
7. Ensure prompt compilation has no provider dependency.

## Example prompt configuration

```yaml
schema_version: 1
prompt_family: basic_binary_choice
prompt_version: 1
blocks:
  - task_description
  - game_rules
  - private_state
  - recent_memory
  - current_interaction
  - decision_instruction
  - output_contract
response_contract:
  type: choice_only
  allowed_values: [A, B]
```

## Inspection command

```bash
conda run -n MA-CC mas-cc inspect phase 3 \
  --prompt configs/components/prompts/basic_binary_choice.yaml \
  --output-dir inspection/phase_03
```

## Inspection artifacts

```text
inspection/phase_03/
├── report.md
├── manifest.json
├── prompt_context.json
├── prompt_blocks.json
├── compiled_messages.json
├── rendered_prompt.md
└── token_breakdown.csv
```

## Manual checks

- Read every prompt block separately.
- Read the final compiled prompt.
- Verify that changing one block changes only its intended section.
- Confirm that no global population or committee information appears unless explicitly included in the context.
- Confirm that the compiled messages are identical regardless of the future provider.

## Acceptance criterion

One prompt can be composed, inspected, versioned, and tested without calling any LLM.

---

# Phase 4 — LLM providers, unified smoke tests, and static preflight

## Objective

Send the same compiled prompt through mock, OpenAI, University, and local Gemma using one normalized interface.

## Tasks

1. Define:
   - `LLMProvider`;
   - `CompletionRequest`;
   - `CompletionResponse`;
   - `ProviderCapabilities`;
   - `ProviderUsage`;
   - `ProviderError`.
2. Implement a provider registry.
3. Implement adapters for:
   - deterministic mock;
   - OpenAI;
   - University proxy;
   - local Gemma.
4. Keep provider-specific concerns inside each adapter:
   - credentials;
   - endpoint discovery;
   - retries;
   - usage parsing;
   - rate limits;
   - local model loading;
   - GPU checks.
5. Load Gemma lazily.
6. Ensure the caller receives the same normalized response type from all providers.
7. Implement a provider smoke-test CLI.
8. Implement static preflight estimates:
   - input token count;
   - assumed output tokens;
   - logical call count supplied by a test spec;
   - expected cost;
   - conservative cost bound;
   - rough latency estimate.
9. Query live pricing or budget only when the provider exposes an authorized mechanism. Otherwise use a versioned pricing configuration and an optional user-defined budget ceiling.
10. Never print request headers or credentials.

## Unified call

```python
provider = create_llm_provider(config)
response = await provider.complete(request)
```

## Inspection commands

```bash
conda run -n MA-CC mas-cc provider test \
  --provider mock \
  --prompt configs/components/prompts/basic_binary_choice.yaml \
  --output-dir inspection/phase_04/mock
```

Equivalent commands should exist for:

```text
openai
university
gemma_local
```

## Inspection artifacts per provider

```text
inspection/phase_04/<provider>/
├── report.md
├── manifest.json
├── request.json
├── normalized_response.json
├── raw_response_redacted.json
├── usage.json
├── preflight_estimate.json
└── timing.csv
```

## Manual checks

- Compare `request.json` across all providers.
- Confirm that provider-specific fields do not leak into game or prompt objects.
- Inspect Gemma load timing separately from inference timing.
- Confirm that failures are normalized and readable.
- Compare actual token usage with the static estimate.
- Confirm that secrets are absent from every artifact.

## Acceptance criterion

The same compiled prompt can run through every backend, and every backend returns a common response record.

---

# Phase 5 — Generic game interface and a minimal reference game

## Objective

Define what a game is operationally and validate the interface with a very small deterministic coordination game.

The reference game exists to test architecture, not to produce scientific results.

## Tasks

1. Define:
   - `GameSpec`;
   - `GameState`;
   - `AgentState`;
   - `Observation`;
   - `DecisionRequest`;
   - `Action`;
   - `Transition`;
   - `InteractionRecord`;
   - `GameResult`.
2. Define a game protocol with explicit methods such as:
   - initialize;
   - select participants;
   - construct observations;
   - build decision requests;
   - validate actions;
   - apply transition;
   - detect termination.
3. Implement a registry for future games.
4. Implement `toy_coordination`:
   - two or more agents;
   - actions `A` and `B`;
   - one simple matching payoff;
   - short finite horizon;
   - deterministic mock behavior.
5. Keep provider execution outside the state-transition function.
6. Make game transitions pure where possible.
7. Produce one trace that can be read interaction by interaction.

## Inspection command

```bash
conda run -n MA-CC mas-cc game run \
  --config configs/runs/toy_game_smoke_test.yaml \
  --output-dir inspection/phase_05
```

## Inspection artifacts

```text
inspection/phase_05/
├── report.md
├── manifest.json
├── resolved_config.yaml
├── initial_state.json
├── interactions.jsonl
├── final_state.json
├── trajectory.csv
└── trajectory.png
```

## Manual checks

- Inspect the initial state.
- Follow one interaction from observation to prompt, response, action, and transition.
- Verify that the transition follows the game rule.
- Re-run with the same seed and compare file hashes.
- Change the provider from mock to a live provider without changing game code.

## Acceptance criterion

A complete game can run through the generic interface and produce an inspectable trajectory.

---

# Phase 6 — First real game: Naming Convention Game

## Objective

Implement the first scientifically relevant game in `mas_cc` without changing the legacy implementation.

This phase creates a new `mas_cc.games.naming_convention` implementation or a temporary read-only adapter. Behavioral parity with selected legacy fixtures should be tested, but the old package remains the reference during migration.

## Tasks

1. Implement explicit naming-convention state:
   - private agent memory;
   - cumulative score;
   - available actions;
   - optional committed action;
   - topology;
   - global evaluator history.
2. Implement random-sequential pair selection.
3. Make the two decisions within one pair simultaneous.
4. Preserve the information boundary:
   - agents see only their allowed information;
   - population state is not leaked;
   - committee metadata is not leaked;
   - partner identity is hidden when required.
5. Implement prompt construction using Phase 3 blocks.
6. Implement response validation and retry accounting.
7. Implement payoff and memory updates.
8. Implement a short smoke run with:
   - small population;
   - short horizon;
   - deterministic seed;
   - mock provider first;
   - one optional live-provider run.
9. Save individual prompts and responses for selected interactions.
10. Compare normalized trajectories with fixed legacy fixtures where equivalent behavior exists.

## Inspection command

```bash
conda run -n MA-CC mas-cc game run \
  --config configs/runs/naming_convention_smoke_test.yaml \
  --output-dir inspection/phase_06
```

## Inspection artifacts

```text
inspection/phase_06/
├── report.md
├── manifest.json
├── resolved_config.yaml
├── agents_initial.json
├── interactions.jsonl
├── selected_audit_traces.jsonl
├── trajectory.csv
├── action_share.png
├── coordination_rate.png
└── agents_final.json
```

Each selected audit trace should contain:

```text
pre-interaction private memory
compiled messages
provider response
parsed action
validation status
payoff
post-interaction private memory
```

## Manual checks

- Read several complete interaction traces.
- Confirm simultaneous decisions use the same pre-interaction state.
- Confirm that each agent sees only its permitted memory.
- Inspect action-share and coordination plots.
- Compare mock reruns for deterministic equality.
- Confirm that no committee logic is required to run the base game.

## Acceptance criterion

The Naming Convention Game runs independently, produces transparent prompts and transitions, and remains separate from experiment orchestration.

---

# Phase 7 — Storage, audit logging, heartbeats, and Comet ML

## Objective

Make multi-hour runs observable and recoverable after the first real game exists.

## Tasks

1. Implement a structured internal event stream:
   - run started;
   - decision requested;
   - decision completed;
   - retry;
   - invalid response;
   - interaction completed;
   - checkpoint written;
   - heartbeat;
   - run completed.
2. Implement separate sinks:
   - console;
   - local log file;
   - JSONL audit;
   - Comet;
   - in-memory metrics.
3. Implement stable storage schemas for:
   - interactions;
   - provider attempts;
   - episodes or runs;
   - audit traces.
4. Add schema versions.
5. Add atomic checkpoint writing.
6. Add resume checks.
7. Add configurable deterministic audit sampling.
8. Add heartbeat output at a fixed interval.
9. Add Comet configuration:
   - enabled/disabled;
   - project;
   - workspace if needed;
   - tags;
   - offline mode;
   - metric cadence.
10. Load `COMET_API_KEY` from `.env`.
11. Keep a placeholder only in `.env.example`:

```dotenv
COMET_API_KEY=replace-with-real-key
```

12. Log aggregated metrics to Comet, not every full prompt by default.
13. Keep complete scientific traces in local artifacts.

## Suggested Comet metrics

### Operational

- interactions completed;
- logical decisions;
- provider requests;
- retries;
- invalid-response rate;
- request latency mean and p90;
- tokens;
- estimated accumulated cost;
- decisions per minute;
- checkpoint age.

### Online scientific diagnostics

- rolling action share;
- rolling coordination rate;
- rolling mean payoff;
- current macrostate;
- forced-action count once interventions are introduced.

## Inspection command

```bash
conda run --live-stream -n MA-CC mas-cc inspect phase 7 \
  --config configs/runs/naming_convention_smoke_test.yaml \
  --output-dir inspection/phase_07
```

## Inspection artifacts

```text
inspection/phase_07/
├── report.md
├── manifest.json
├── resolved_config.yaml
├── experiment.log
├── events.jsonl
├── api_call_status.jsonl
├── audit_traces.jsonl
├── checkpoint_manifest.json
├── local_metrics.csv
├── comet_summary.json
└── observability_dashboard.png
```

## Manual checks

- Watch the heartbeat during a deliberately slowed mock run.
- Open the Comet experiment and compare its metrics with `local_metrics.csv`.
- Kill a run after a checkpoint and resume it.
- Verify that logging failure does not modify game decisions.
- Confirm that no secret appears in local files or Comet parameters.
- Confirm that Comet can be disabled without affecting the run.

## Acceptance criterion

A long run is visible in real time, locally auditable, restartable, and monitorable in Comet.

---

# Phase 8 — Control policies and information-theoretic analysis primitives

## Objective

Implement the committee/control abstraction and the first two information-theoretic estimators independently of the full experiment grid.

## Part A: control layer

### Tasks

1. Define:
   - `ControlPolicy`;
   - `Committee`;
   - `CommitteeSchedule`;
   - `Intervention`;
   - `ControlBudget`;
   - `ControlObservation`.
2. Separate:
   - committee membership;
   - committee observation privileges;
   - policy;
   - action forcing;
   - duration;
   - budget.
3. Implement initial policies:
   - no committee;
   - always A;
   - always B;
   - support incumbent;
   - promote alternative;
   - temporary pulse.
4. Ensure forced actions:
   - skip the LLM call;
   - remain visible in the affected agents’ real memory;
   - are explicitly logged.
5. Run one manually configured intervention without an experiment grid.

## Part B: analysis primitives

### Discrete/histogram mutual information

Implement a discrete estimator for variables such as:

\[
I(G;Y)
\]

and:

\[
I(G;S_{t+h}\mid S_t).
\]

The module should support:

- contingency-table construction;
- unsmoothed plug-in estimate;
- Jeffreys smoothing;
- optional Miller–Madow sensitivity estimate;
- bootstrap over independent episode IDs;
- null permutations;
- diagnostic tables.

### InfoNCE estimator

Implement an estimator based on the selected embedding model and an InfoNCE objective.

The architecture should separate:

```text
text or trajectory extraction
→ embedding model
→ positive/negative pair construction
→ InfoNCE objective
→ estimate and diagnostics
```

The embedding model must be loaded through an explicit analysis component, not through an LLM game provider.

### Testing strategy

Before integrating with full experiments, test both estimators on:

- independent synthetic variables;
- perfectly correlated synthetic variables;
- partially dependent synthetic variables;
- a small recorded naming-convention run.

## Inspection commands

```bash
conda run -n MA-CC mas-cc control demo \
  --config configs/runs/naming_convention_control_demo.yaml \
  --output-dir inspection/phase_08/control
```

```bash
conda run -n MA-CC mas-cc analyze information \
  --config configs/runs/information_estimator_validation.yaml \
  --output-dir inspection/phase_08/analysis
```

## Inspection artifacts

```text
inspection/phase_08/
├── report.md
├── manifest.json
├── control/
│   ├── intervention_schedule.csv
│   ├── interactions.jsonl
│   ├── forced_actions.csv
│   └── action_share.png
└── analysis/
    ├── synthetic_dataset.parquet
    ├── discrete_mi_results.csv
    ├── contingency_tables.csv
    ├── infonce_results.csv
    ├── null_results.csv
    ├── estimator_diagnostics.md
    ├── discrete_mi_validation.png
    └── infonce_validation.png
```

## Manual checks

- Verify the exact interactions in which control was applied.
- Confirm that forced decisions skipped provider calls.
- Inspect contingency tables directly.
- Confirm that MI is near zero for independent synthetic variables.
- Confirm that MI increases for dependent synthetic variables.
- Inspect InfoNCE positive and negative pair construction.
- Confirm that analysis never calls the game LLM provider.

## Acceptance criterion

Control policies can be applied to a single game run, and both information estimators work on controlled synthetic fixtures before full experiment orchestration.

---

# Phase 9 — Full experiment orchestration and committee empowerment pilot

## Objective

Integrate all previous components into a reproducible experiment runner.

This is intentionally the final phase.

## Tasks

1. Implement:
   - `ExperimentSpec`;
   - `EpisodeSpec`;
   - experimental strata;
   - grids;
   - repetitions;
   - deterministic episode IDs;
   - seed derivation;
   - episode concurrency;
   - global request concurrency;
   - checkpoint/resume;
   - compaction;
   - post-run analysis invocation.
2. Compose:
   - game;
   - LLM provider;
   - prompt;
   - control policy;
   - execution policy;
   - logging;
   - storage;
   - analysis.
3. Implement the committee-empowerment pilot grid.
4. Implement exact preflight from the resolved experiment:
   - number of episodes;
   - maximum interactions;
   - logical LLM decisions;
   - forced decisions;
   - expected provider requests;
   - expected input tokens;
   - expected output tokens;
   - expected cost;
   - conservative cost;
   - configured or queried budget;
   - expected runtime range;
   - expected disk use.
5. Require an explicit launch flag after preflight for live providers.
6. Record the approved preflight with the run.
7. Run episodes concurrently without changing within-episode causal order.
8. Compact only complete episode shards.
9. Run offline analysis only after simulation artifacts are safely stored.
10. Produce a final experiment report and plots.

## Example run configuration

```yaml
schema_version: 1

experiment:
  name: committee_empowerment_pilot
  seed: 1234
  replicates: 20

game:
  type: naming_convention
  population_sizes: [10, 20]
  memory_size: 5
  actions: [A, B]
  match_payoff: 100
  mismatch_payoff: -50
  max_population_rounds: 30

llm_provider:
  component: university
  model: gwdg/qwen3-30b-a3b-instruct-2507

prompt:
  component: naming_convention_v1

control:
  committee_sizes: [0, 1, 2, 4]
  policies:
    - no_committee
    - support_incumbent
    - promote_alternative
    - alternative_pulse

execution:
  episode_concurrency: 2
  request_concurrency: 10
  heartbeat_seconds: 60

storage:
  output_dir: results/committee_empowerment_pilot
  resume: true
  schema_version: 1

observability:
  comet:
    enabled: true
    project_name: mas-committee-empowerment

analysis:
  discrete_mutual_information: true
  infonce: true
  horizons_population_rounds: [1, 3, 5]
```

## Preflight command

```bash
conda run -n MA-CC mas-cc experiment preflight \
  --config configs/runs/committee_empowerment_pilot.yaml \
  --output-dir inspection/phase_09/preflight
```

## Pilot command

```bash
conda run --live-stream -n MA-CC mas-cc experiment run \
  --config configs/runs/committee_empowerment_pilot.yaml \
  --approve-preflight inspection/phase_09/preflight/preflight_id.txt \
  --output-dir inspection/phase_09/run
```

## Inspection artifacts

```text
inspection/phase_09/
├── report.md
├── manifest.json
├── preflight/
│   ├── resolved_config.yaml
│   ├── experiment_grid.csv
│   ├── call_count_breakdown.csv
│   ├── token_estimate.csv
│   ├── cost_estimate.json
│   ├── runtime_estimate.json
│   ├── budget_status.json
│   └── preflight_report.md
└── run/
    ├── resolved_config.yaml
    ├── approved_preflight.json
    ├── experiment.log
    ├── events.jsonl
    ├── interactions.parquet
    ├── provider_attempts.parquet
    ├── episodes.parquet
    ├── experiment_summary.csv
    ├── discrete_mi_results.parquet
    ├── infonce_results.parquet
    ├── null_results.parquet
    ├── summary.md
    └── plots/
        ├── outcomes_by_policy.png
        ├── action_share_trajectories.png
        ├── committee_effect.png
        ├── discrete_empowerment.png
        ├── dynamical_empowerment.png
        └── infonce_empowerment.png
```

## Manual checks

- Inspect the expanded experiment grid before launching.
- Compare estimated and actual calls, tokens, cost, and runtime.
- Open selected interaction traces from different policies.
- Verify committee schedules and forced-action counts.
- Inspect episode completeness and checkpoint recovery.
- Compare Comet metrics with local metrics.
- Inspect discrete MI contingency tables.
- Inspect InfoNCE diagnostics.
- Confirm that re-running offline analysis makes no LLM calls.

## Acceptance criterion

A complete committee-empowerment pilot can be configured, priced, approved, executed, monitored, resumed, analyzed, and manually audited.

---

## 6. Recommended work for today

A sensible target for the first day is:

### Minimum

- Phase 1 complete;
- Phase 2 complete;
- Phase 3 complete.

### Strong target

- Phase 1 complete;
- Phase 2 complete;
- Phase 3 complete;
- Phase 4 mock adapter complete;
- Phase 4 one live adapter started or completed.

Do not begin the Naming Convention Game implementation until:

- the configuration model is stable;
- one prompt is inspectable;
- one normalized provider call works;
- secrets are excluded from artifacts.

---

## 7. Cross-phase test strategy

Each phase should add three test categories.

### Unit tests

Test pure functions and individual records.

### Contract tests

Verify that interchangeable components satisfy the same interface:

- all LLM providers;
- all prompt blocks;
- all games;
- all control policies;
- all event sinks.

### Inspection tests

Run the phase inspection command and verify that:

- expected files exist;
- schemas validate;
- secrets are absent;
- hashes are stable when determinism is expected;
- plots are non-empty;
- reports contain no unhandled errors.

---

## 8. Dependency direction

The intended dependency direction is:

```text
core
  ↓
configuration
  ↓
prompts and LLM-provider protocols
  ↓
LLM-provider adapters
  ↓
games
  ↓
runtime and storage
  ↓
observability
  ↓
control and analysis
  ↓
experiments
  ↓
CLI
```

More operationally:

```text
LLM providers do not know games.
Games do not know OpenAI, University, Gemma, Comet, or Parquet.
Prompts do not execute providers.
Control policies do not write files.
Storage does not define scientific transitions.
Comet is never the source of truth.
Analysis never calls a game LLM.
Experiments compose already tested components.
```

---

## 9. Definition of completion

The reorganization is complete when the following workflow succeeds:

```text
1. Select or write a game configuration.
2. Select a compositional prompt.
3. Select an LLM provider.
4. Render and inspect the exact prompt.
5. Estimate calls, tokens, cost, runtime, and budget.
6. Run a small game.
7. Inspect individual interactions and plots.
8. Enable local logging and Comet.
9. Add a committee policy.
10. Validate MI and InfoNCE on known fixtures.
11. Expand into an experiment grid.
12. Execute with checkpoints and monitoring.
13. Re-run analysis without model access.
14. Reproduce the run from the saved resolved configuration.
```

The architecture should make each of these steps independently visible rather than hiding them inside one large experiment command.
