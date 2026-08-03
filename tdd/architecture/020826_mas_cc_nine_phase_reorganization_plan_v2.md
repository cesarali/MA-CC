# MAS-CC Reorganization and Validation Plan — Version 2

**Status:** revised implementation plan
**Plan version:** 2
**Revision date:** 2026-08-02
**Supersedes:** `010826_mas_cc_nine_phase_reorganization_plan.md` without modifying it
**Current migration boundary:** Phases 1–3 and the original Phase 4 remain historically inspectable; the Phase 4 amendment in this version must pass before Phase 5 begins
**Target package:** `mas_cc`
**Legacy package:** `naming_game` remains operational and unchanged during the migration
**Maximum phases:** 9
**Primary principle:** every phase must be independently runnable and manually inspectable

---

## Revision notes for Version 2

This revision keeps the nine-phase structure and preserves all artifacts produced
under Version 1. It adds a retroactive Phase 4 amendment after review of the
University of Potsdam proxy's live model-information and account-information
endpoints.

The amendment establishes:

- live, cached, and offline pricing sources with provenance and freshness;
- currency/accounting-unit-safe cost records instead of an unconditional USD assumption;
- cached-input, long-context, and provider-specific price dimensions;
- pre-launch budget checks and a basic concurrency-safe runtime budget guard;
- a planning boundary in which games and prompts describe resource demand,
  providers describe monetary rates, and preflight composes them;
- prospective game call-plan contracts in Phases 5–6;
- accumulated actual-cost logging in Phase 7;
- full experiment-grid estimation and approval in Phase 9.

The original `inspection/phase_04` evidence remains evidence for the original
Phase 4 acceptance criteria. The amendment produces a separate
`inspection/phase_04_amendment` bundle. Phases 1–3 are not repeated, but their
tests must be rerun to catch regressions caused by shared type or configuration
changes.

---

## 1. Purpose

The immediate goal is not to rewrite the existing `naming_game` package. The goal is to construct a clean successor package, `mas_cc`, in parallel with it.

The successor should support:

- multiple language games;
- several interchangeable LLM backends;
- compositional “Lego-like” prompts;
- reproducible experiment configurations;
- cost and runtime estimation before execution for every resolvable game run and experiment;
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

### 1.6 Resource demand, pricing, and budget

Cost estimation is a composition of separate responsibilities:

- games describe interactions, decision stages, stopping bounds, and logical LLM demand;
- prompts compile representative and bounded request contexts and expose token estimates;
- control policies describe forced decisions, avoided requests, and additional evaluations;
- experiments expand configurations, grids, repetitions, and episodes;
- provider pricing sources describe availability, rates, accounting units, context tiers, and limits;
- planning combines demand and rates into lower, expected, and conservative estimates;
- runtime budget enforcement prevents new requests when an approved limit would be exceeded.

Games and prompts must not contain provider prices. Providers must not contain
game logic. A provider account budget and a MAS-CC per-run budget are distinct
controls.

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
11. Live-provider execution must not proceed with an unknown price or accounting unit unless the resolved run configuration contains an explicit, auditable override.
12. A run-specific budget may lower but must not silently raise the configured system-wide per-run limit.
13. Pricing refreshes must never serialize credentials, request headers, internal deployment URLs, or account identity into artifacts.

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

# Phase 4 — LLM providers, unified smoke tests, and provider-economics preflight

## Objective

Send the same compiled prompt through mock, OpenAI, University, and local Gemma
using one normalized interface, then establish the pricing, accounting-unit,
preflight, and basic runtime-budget foundations required by every later game.

Phase 4 has two inspection scopes:

1. the original normalized-provider implementation; and
2. the Version 2 provider-economics amendment, which must pass before Phase 5.

## Original provider tasks

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
8. Never print request headers or credentials.

## Version 2 Phase 4 amendment tasks

1. Replace unconditional USD-only pricing and budget fields with typed monetary
   or accounting-unit records containing:
   - amount;
   - unit such as `USD`, `EUR`, `proxy_accounting_unit`, or `unknown`;
   - unit source;
   - provider and exact model ID;
   - source URI or source description;
   - retrieval time;
   - catalog or quote version.
2. Define a provider-independent `PricingSource` or equivalent interface with:
   - `live` mode;
   - `cached` mode;
   - `offline` static-catalog mode;
   - explicit freshness and fallback policy.
3. Implement a University pricing source that performs read-only requests to:
   - `GET /models` for current availability;
   - `GET /v1/model/info` for current prices and limits;
   - optionally `GET /user/info` for account budget information.
4. Query University live metadata once during live preflight and revalidate it
   immediately before launch. Do not query pricing before every completion.
5. Save only a sanitized pricing snapshot containing the selected model's public
   planning metadata, retrieval time, source, and content hash. Exclude API
   headers, keys, internal `api_base` values, deployment identifiers, account
   identity, and unrelated account records.
6. Preserve the versioned static catalog as an auditable offline fallback.
   Unknown models deliberately remain unknown rather than receiving a guessed rate.
7. Represent the price dimensions required by supported chat runs:
   - ordinary input;
   - cached input read;
   - cache creation when applicable;
   - output;
   - context-threshold overrides;
   - applicable RPM, TPM, maximum input, and maximum output limits.
   Non-chat modalities may remain unsupported but must be recognized and rejected
   clearly rather than priced as ordinary chat.
8. Correct cached-input arithmetic:
   - uncached input = total input minus cached input;
   - uncached input uses the ordinary input rate;
   - cached input uses the cache rate;
   - output uses the output rate.
9. Extend preflight results to distinguish:
   - pricing known, partial, stale, unavailable, or unit-unknown;
   - lower, expected, and conservative token/cost estimates;
   - provider account budget;
   - system-wide per-run limit;
   - run-specific limit;
   - permitted, denied, or explicit-override-required status.
10. Retain the Phase 4 logical-call input as a provider-smoke-test fixture. Do
    not add game-specific call formulas to the provider layer.
11. Add a basic concurrency-safe runtime budget guard around live provider
    requests:
    - atomically reserve a conservative request cost before dispatch;
    - release or reconcile the reservation after normalized usage is available;
    - stop scheduling before the approved cost, request, input-token, or
      output-token limit can be exceeded;
    - fail closed when a live paid request cannot be bounded, unless an explicit
      resolved-config override permits it.
12. Keep provider-account budget and MAS-CC per-run budget separate.
13. Make all new imports side-effect free: static/offline preflight must not read
    credentials, create clients, start the VPN bridge, or contact a provider.
14. Preserve backward compatibility where practical. If Phase 2 configuration
    or core types change, rerun Phases 1–3 tests but do not replace their
    historical inspection artifacts.

## Unified call

```python
provider = create_llm_provider(config)
response = await provider.complete(request)
```

The budget guard may wrap this interface, but callers still receive the same
normalized response type.

## Original inspection commands

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

## Amendment inspection command

```bash
conda run -n MA-CC mas-cc inspect phase 4 \
  --amendment provider-economics-v2 \
  --provider university \
  --pricing-mode live \
  --output-dir inspection/phase_04_amendment
```

The command is preflight-only by default and must not send a billable
completion. A separate explicit flag is required for any live completion test.

## Original inspection artifacts per provider

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

## Amendment inspection artifacts

```text
inspection/phase_04_amendment/
├── report.md
├── manifest.json
├── resolved_config.yaml
├── selected_model_availability.json
├── pricing_snapshot.json
├── pricing_snapshot.sha256
├── preflight_estimate.json
├── budget_status.json
├── runtime_guard_scenarios.json
└── regression_test_summary.json
```

## Manual checks

- Confirm the original Phase 4 provider artifacts remain unchanged.
- Compare normalized requests and responses across providers.
- Confirm that live, cached, and offline pricing modes identify their provenance.
- Confirm that the University snapshot contains no credentials, internal
  deployment URLs, account identity, or unrelated account details.
- Confirm that an unavailable model, stale price, unknown unit, and unknown paid
  price each produce explicit safe statuses.
- Compare live selected-model rates with the cached sanitized snapshot.
- Confirm that no exchange-rate conversion occurs without an explicit source,
  timestamp, and target currency.
- Verify cached-token and long-context arithmetic with deterministic fixtures.
- Run concurrent mock requests and confirm atomic reservations cannot overspend
  the configured run ceiling.
- Confirm static preflight performs no external work.
- Rerun all Phase 1–4 tests and record their results.

## Acceptance criterion

The original normalized-provider contract still passes. In addition, every
supported live chat provider can supply an auditable price quote or an explicit
unknown status; preflight preserves the quote's accounting unit and provenance;
and the basic runtime guard prevents request scheduling beyond approved
cost/request/token limits under concurrency.

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
8. Define a provider-independent game planning contract that reports:
   - fixed, expected, and maximum interactions when knowable;
   - logical decision stages;
   - requests per stage;
   - forced or provider-free decisions;
   - retry bounds;
   - stopping-condition assumptions;
   - representative and maximum prompt contexts required from the prompt layer.
9. Make `toy_coordination` emit this plan without knowing provider prices.

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
├── game_call_plan.json
├── trajectory.csv
└── trajectory.png
```

## Manual checks

- Inspect the initial state.
- Follow one interaction from observation to prompt, response, action, and transition.
- Verify that the transition follows the game rule.
- Re-run with the same seed and compare file hashes.
- Change the provider from mock to a live provider without changing game code.
- Inspect `game_call_plan.json` and confirm it is identical across provider selections.
- Confirm the planner can combine that demand with Phase 4 mock and live-price fixtures.

## Acceptance criterion

A complete game can run through the generic interface, produce an inspectable
trajectory, and expose provider-independent resource demand that Phase 4
pricing can convert into a run estimate.

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
11. Implement a Naming Convention Game call plan broken down by decision stage,
    population round, participant count, forced decisions, validation retries,
    and stopping assumptions.
12. Ask the prompt layer for representative and bounded token estimates at
    relevant history/memory sizes instead of extrapolating every call from only
    the first prompt.
13. Produce lower, expected, and conservative run-demand estimates without
    importing or selecting a provider.

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
├── game_call_plan.json
├── prompt_token_scenarios.csv
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
- Compare planned and actual logical calls in the deterministic mock run.
- Inspect token scenarios for empty, representative, and maximum configured memory/history.

## Acceptance criterion

The Naming Convention Game runs independently, produces transparent prompts
and transitions, remains separate from experiment orchestration, and exposes
stage-aware resource demand that can be priced without provider knowledge.

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
14. Integrate the Phase 4 runtime budget guard with execution lifecycle events.
15. Persist, per provider attempt:
    - estimated and reserved cost;
    - normalized actual input, cached-input, and output usage;
    - reconciled accumulated cost;
    - accounting unit and price-snapshot hash;
    - retry and invalid-response attribution.
16. Emit explicit budget-reserved, budget-reconciled, budget-near-limit, and
    budget-exhausted events.
17. Ensure checkpoint/resume restores accumulated usage and reservations safely
    and cannot reset or duplicate the approved run budget.

## Suggested Comet metrics

### Operational

- interactions completed;
- logical decisions;
- provider requests;
- retries;
- invalid-response rate;
- request latency mean and p90;
- input, cached-input, and output tokens;
- estimated, reserved, reconciled, and accumulated cost;
- remaining per-run budget;
- budget-guard stop count;
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
├── usage_cost.jsonl
├── budget_events.jsonl
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
- Compare reservations, reconciled usage, and accumulated cost with deterministic fixtures.
- Resume near the run limit and confirm that the remaining budget is preserved.
- Confirm a budget stop leaves a valid checkpoint and a clearly classified run status.

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
6. Make every control policy expose a provider-independent adjustment to the
   underlying game call plan, including forced decisions that skip provider
   requests and any policy-specific additional evaluations.

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
│   ├── control_call_plan_adjustment.json
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
- Compare the base game call plan with the control-adjusted call plan.
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
4. Implement resolved-experiment preflight by composing the Phase 5–6 game
   demand, prompt token scenarios, control-policy effects, experiment expansion,
   and Phase 4 price quote:
   - number of strata and episodes;
   - fixed, expected, and maximum interactions;
   - logical LLM decisions by stage;
   - forced and provider-free decisions;
   - expected provider requests and retry bounds;
   - lower, expected, and conservative input/output token estimates;
   - cached-token and context-tier assumptions;
   - lower, expected, and conservative cost in the quote's accounting unit;
   - pricing mode, provenance, retrieval time, freshness, and snapshot hash;
   - provider account budget when available;
   - system-wide and run-specific limits;
   - expected runtime range;
   - expected disk use;
   - unknowns and explicit overrides.
5. Treat grid sizes and bounded call counts as exact where possible, but never
   label dynamic token use or cost as exact.
6. Require an explicit launch flag and approved preflight ID for live providers.
7. Bind the preflight ID to the resolved configuration, game/prompt versions,
   experiment grid, pricing snapshot, limits, and estimation assumptions.
8. Revalidate model availability and price immediately before launch. If any
   bound input changes, require a new preflight approval.
9. Record the approved preflight and runtime budget state with the run.
10. Run episodes concurrently without changing within-episode causal order or
    weakening atomic budget reservations.
11. Compact only complete episode shards.
12. Run offline analysis only after simulation artifacts are safely stored.
13. Produce a final experiment report and plots, including estimated-versus-actual
    calls, tokens, cost, runtime, and disk use.

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

pricing:
  mode: live
  require_fresh_at_launch: true
  unknown_price_policy: deny

budget:
  accounting_unit: proxy_accounting_unit
  max_cost_per_run: 25
  max_provider_requests: 100000
  max_input_tokens: 50000000
  max_output_tokens: 10000000

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
│   ├── game_call_plans/
│   ├── call_count_breakdown.csv
│   ├── prompt_token_scenarios.csv
│   ├── token_estimate.csv
│   ├── pricing_snapshot.json
│   ├── pricing_snapshot.sha256
│   ├── cost_estimate.json
│   ├── runtime_estimate.json
│   ├── budget_status.json
│   ├── preflight_id.txt
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

- Inspect the expanded experiment grid, game call plans, prompt token scenarios,
  pricing provenance, accounting unit, and all estimation assumptions before launching.
- Confirm that a changed config, prompt version, game version, pricing snapshot,
  or limit invalidates the preflight ID.
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

## 6. Current revision boundary and next work

At the adoption of Version 2:

- retain the existing Phase 1–4 inspection artifacts as historical evidence;
- treat the original Phase 4 normalized-provider work as implemented but not
  sufficient for the revised Phase 4 acceptance criterion;
- implement and inspect the provider-economics amendment in
  `inspection/phase_04_amendment`;
- rerun all Phase 1–4 tests and record the regression summary;
- begin Phase 5 only after the amendment acceptance criterion passes.

Do not begin the Naming Convention Game implementation until:

- the provider-economics amendment is complete;
- live, cached, and offline pricing modes are inspectable;
- accounting-unit semantics and unknown-price behavior are explicit;
- the concurrency-safe runtime budget guard passes deterministic tests;
- the generic Phase 5 game planning contract is stable;
- secrets and internal provider metadata are excluded from artifacts.

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
5. Resolve provider-independent game demand and prompt token scenarios.
6. Fetch or select an auditable price quote with an explicit accounting unit.
7. Estimate lower, expected, and conservative calls, tokens, cost, runtime, and budget.
8. Approve a preflight bound to the resolved inputs and limits.
9. Run a small game under the runtime budget guard.
10. Inspect individual interactions, usage, budget events, and plots.
11. Enable local logging and Comet.
12. Add a committee policy.
13. Validate MI and InfoNCE on known fixtures.
14. Expand into an experiment grid.
15. Execute with checkpoints, monitoring, and preserved budget state.
16. Compare estimated and actual calls, tokens, cost, runtime, and disk use.
17. Re-run analysis without model access.
18. Reproduce the run from the saved resolved configuration and approved preflight.
```

The architecture should make each of these steps independently visible rather than hiding them inside one large experiment command.
