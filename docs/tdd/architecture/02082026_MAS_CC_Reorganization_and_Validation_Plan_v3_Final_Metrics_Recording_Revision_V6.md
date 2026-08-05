# MAS-CC Reorganization and Validation Plan — Version 3

**Status:** adopted; Phase 3–6 realignment accepted  
**Plan version:** 3  
**Revision date:** 2026-08-02  
**Supersedes:** `020826_mas_cc_nine_phase_reorganization_plan_v2.md` without modifying it  
**Companion implementation plan:** `02082026_realigningment_phase_3456.md`  
**Current migration boundary:** Phase 3–6 Version 3 gates passed; Phase 7 is unblocked  
**Target package:** `mas_cc`  
**Legacy package:** `naming_game` remains operational and unchanged  
**Maximum phases:** 9  
**Primary principle:** every phase must be independently runnable, manually inspectable, and connected through narrow contracts

---

## Table of contents

| Phase | Focus |
| --- | --- |
| [Foundation](#foundation) | Revision notes, purpose, architecture, and configuration conventions |
| [Phase 1](#phase-1) | Freeze the legacy package and create the successor shell |
| [Phase 2](#phase-2) | Core records and validated configuration |
| [Phase 3](#phase-3) | Abstract value-bearing prompt composition |
| [Phase 4](#phase-4) | Normalized providers and provider economics |
| [Phase 5](#phase-5) | Generic game interface and toy game under game-owned prompts |
| [Phase 6](#phase-6) | Naming Convention Game redesigned around concrete blocks |
| [Phase 7](#phase-7) | Storage, audit logging, heartbeats, and Comet ML |
| [Phase 8.0](#phase-80) | Recording contracts, generic metrics, and metric logging |
| [Phase 8.1](#phase-81) | Control policies and offline information-theoretic analysis |
| [Phase 9](#phase-9) | Full experiment orchestration and committee empowerment pilot |
| [Wrap-up](#wrap-up) | Adoption boundary, test strategy, dependency direction, and completion definition |

<a id="foundation"></a>

## Revision notes for Version 3

Version 3 corrects a prompt-architecture misconception in Versions 1–2.

The earlier design treated `PromptContext` as a universal game-shaped envelope
with fields such as task description, rules, private state, memory, and current
interaction. It then treated `PromptBlock` mainly as a renderer over that
envelope. This creates three problems:

1. context fields and block names appear to represent the same concepts;
2. prompt-specific schemas are hidden inside untyped mappings such as
   `private_state["presented_actions"]`;
3. games must populate universal fields even when their prompts do not use them.

Version 3 replaces that model with two abstract prompt-layer contracts:

- `PromptBlock[T]` is a value-bearing, independently renderable prompt component;
- `FullPrompt` is an immutable, ordered composition of concrete blocks plus a
  response contract.

Concrete games define their own concrete full prompts and blocks. For example:

```text
NamingConventionFullPrompt
├── DescriptionBlock
├── RulesBlock
├── PresentedActionsBlock
├── VisibleMemoryBlock
└── VisibleScoreBlock
```

The generic prompt package knows how to bind immutable values, validate required
blocks, render blocks, group messages, count tokens, fingerprint instances, and
compile normalized messages. It does not know what a score, memory, evacuation
fact, committee, or naming game is.

Version 3 preserves the Version 2 provider-economics amendment. Provider
adapters already consume normalized `CompletionRequest.messages`, so they must
not be redesigned around games or concrete prompts.

This final revision also separates raw recording, streaming/final metrics, and
offline scientific analyses. A versioned `RecordingPlan` now guarantees before
launch that selected metrics, controls, stopping rules, and planned analyses can
be reconstructed from persisted artifacts. Runtime metrics use abstract metric
contracts; Phase 8.1 analyses use a separate offline-analysis contract.

Historical inspection artifacts remain immutable evidence. Realigned artifacts
are written to new directories suffixed `_v3`.

---

## 1. Purpose

The goal remains to construct `mas_cc` beside the unchanged `naming_game`
package. The successor must support multiple games, interchangeable LLM
providers, game-owned compositional prompts, reproducible experiments,
pre-launch resource estimation, observable execution, control policies, and
offline analysis.

The architecture distinguishes the following objects.

### 1.1 Game

A game defines:

- state and legal transitions;
- information available to each agent;
- participant selection and topology;
- decision stages;
- the concrete `FullPrompt` used by each decision stage;
- legal actions, payoff rules, and stopping conditions.

A game constructs a new immutable bound prompt for each agent decision. It does
not mutate a shared prompt in place.

### 1.2 Prompt block

A prompt block is one semantic, value-bearing Lego piece. It owns:

- a stable name and version;
- a message role;
- a typed or validated value;
- required/optional status;
- fixed/dynamic binding policy;
- sensitivity/audit policy where needed;
- its rendering method.

An unbound value is distinct from a valid empty value. `UNBOUND` means a required
runtime value has not been supplied; `()` can mean a supplied but empty memory.

### 1.3 Full prompt

`FullPrompt` is the abstract game-independent composition mechanism. A concrete
full prompt owns:

- prompt family and version;
- an ordered tuple of concrete blocks;
- a response contract;
- message grouping rules;
- immutable binding;
- validation;
- block and total token estimation;
- definition and instance fingerprints;
- compilation to provider-independent messages.

One game may define more than one full prompt. HiddenBench needs at least a
discussion prompt and a vote prompt. A naming-convention game needs a decision
prompt. Prompt stage identity must therefore be explicit.

### 1.4 Compiled prompt

A compiled prompt is an immutable artifact containing:

- every rendered block in order;
- omitted optional blocks and reasons;
- normalized system/user messages;
- response contract;
- per-block and total token estimates;
- prompt family/version;
- definition hash and bound-instance hash.

Only compiled messages cross the LLM-provider boundary.

### 1.5 LLM provider

An LLM provider executes a provider-independent `CompletionRequest` using
OpenAI, the University proxy, local Gemma, or a deterministic mock. Providers do
not receive game objects, `FullPrompt`, or prompt blocks.

### 1.6 Experiment and control

Experiments expand games, providers, repetitions, seeds, and policies. Control
policies may change decisions or skip calls, but they do not mutate prompt
definitions or access private blocks without an explicit game-granted view.

### 1.7 Resource demand, pricing, and budget

- games describe logical decision demand;
- concrete full prompts provide bound token scenarios;
- providers describe monetary rates and operational limits;
- planning combines demand, compiled prompt estimates, and prices;
- runtime budget enforcement guards normalized provider requests.

No prompt or game contains provider prices. No provider contains game or prompt
rendering logic.

---

## 2. Non-negotiable implementation rules

1. `src/naming_game/` is not modified except for an explicitly approved critical fix.
2. Importing `mas_cc` must not load a model, read credentials, create a client,
   start a run, or open Comet.
3. `PromptContext` is not the active Phase 3–9 prompt input contract after the
   Version 3 migration.
4. Abstract prompt machinery lives in `mas_cc.prompts`; concrete game prompts
   live with their games or an explicitly temporary benchmark plugin.
5. Every concrete prompt and block is versioned and immutable after binding.
6. Binding returns a new object. Concurrent agent decisions must never share
   mutable block values.
7. `UNBOUND`, `None`, and valid empty values have distinct documented semantics.
8. A required unbound block fails before provider creation or dispatch.
9. Optional omitted blocks remain visible in inspection metadata.
10. Providers receive only normalized messages and request parameters.
11. Game code contains no provider-specific conditionals or prices.
12. Providers contain no game state, prompt block, or agent-memory logic.
13. Resolved configurations and exact prompt versions are saved with every run.
14. Real secrets belong only in `.env` or an external secret manager.
15. Live paid execution fails closed for unknown price/accounting unit unless an
    explicit auditable override is present.
16. A run-specific budget cannot silently raise the system-wide limit.
17. Historical inspection artifacts are never overwritten by Version 3 work.
18. Every phase produces a report, manifest, and phase-specific inspectable artifacts.

---

## 3. Target repository structure

```text
src/mas_cc/
├── core/
├── config/
├── prompts/
│   ├── blocks.py             # abstract PromptBlock, UNBOUND, bound-state rules
│   ├── full_prompt.py        # abstract FullPrompt and immutable bind/compile flow
│   ├── compiled.py           # RenderedPromptBlock and CompiledPrompt
│   ├── contracts.py          # response contracts and validation
│   ├── messages.py           # grouping policy if not retained in composer.py
│   ├── registry.py           # family/version -> concrete FullPrompt factory
│   ├── tokenization.py       # tokenizer protocol and deterministic estimator
│   ├── fingerprints.py       # definition and bound-instance hashes
│   └── examples/
│       └── basic_choice.py
├── llm_providers/
│   ├── requests.py
│   ├── responses.py
│   ├── pricing.py
│   ├── budget.py
│   ├── registry.py
│   └── adapters/
├── games/
│   ├── protocols.py
│   ├── registry.py
│   ├── toy_coordination/
│   │   ├── game.py
│   │   └── prompts.py
│   ├── naming_convention/
│   │   ├── game.py
│   │   ├── prompts.py
│   │   ├── records.py
│   │   ├── parsing.py
│   │   └── runtime.py
│   └── hidden_bench/         # prospective game; not implied complete by prompt fixtures
│       ├── game.py
│       └── prompts.py
├── planning/
├── runtime/
├── storage/
├── observability/
├── control/
├── analysis/
├── experiments/
└── cli/
```

Temporary paper fixtures may remain under `prompts/plugins/` during migration,
but a concrete game prompt becomes owned by `games/<game>/prompts.py` when that
game is implemented.

---

## 4. Prompt architecture contract

### 4.1 Abstract block contract

The exact Python spelling may change, but the contract must express:

```python
class PromptBlock(ABC, Generic[T]):
    name: str
    version: int
    role: MessageRole
    value: T | Unbound
    required: bool
    binding: Literal["fixed", "dynamic"]

    @abstractmethod
    def validate_value(self, value: object) -> ValidationResult: ...

    @abstractmethod
    def render(self) -> str: ...

    def bind(self, value: T) -> Self: ...  # returns a new block
```

Block rendering must be pure and deterministic for a given bound value and
version.

### 4.2 Abstract full-prompt contract

```python
class FullPrompt(ABC):
    family: str
    version: int
    blocks: tuple[PromptBlock[Any], ...]
    response_contract: ResponseContract
    message_mode: str

    def bind(self, **values: object) -> Self: ...
    def validate(self) -> ValidationResult: ...
    def compile(self, token_counter: TokenCounter | None = None) -> CompiledPrompt: ...
```

The concrete prompt owns authoritative block order. Configuration selects a
family/version and permitted presentation options; it must not silently reorder
a scientific prompt without creating a new prompt version.

### 4.3 Configuration

Prompt component schema Version 2 should resemble:

```yaml
schema_version: 2
prompt_family: naming_convention_decision
prompt_version: 1
message_mode: merge_consecutive_roles
response_contract:
  type: paper_choice_reason
  allowed_values: [Q, M]
```

The resolved configuration exports the authoritative ordered block manifest
from the concrete full prompt:

```yaml
resolved_blocks:
  - description@1
  - rules@1
  - presented_actions@1
  - visible_memory@1
  - visible_score@1
```

During migration, prompt schema Version 1 may be readable only through an
explicit compatibility adapter. New runs and Version 3 artifacts use schema 2.

### 4.4 Prompt fingerprints

- definition hash: family, prompt version, block classes, block versions, order,
  roles, required/optional status, message policy, and response contract shape;
- instance hash: definition hash plus canonical bound values and rendered messages;
- audit hash: may redact sensitive values but must remain linked to a local
  protected full record.

---

## 5. Configuration and inspection conventions

Reusable component and run configurations remain separate. Every execution
saves a fully resolved, secret-free configuration.

Every phase continues to expose:

```bash
mas-cc inspect phase <N> --output-dir inspection/phase_<NN>_v3
```

Every inspection directory contains at least:

```text
report.md
manifest.json
resolved_config.yaml        # when configuration is involved
```

Phase reports state command, code paths, inputs, outputs, expected behavior,
warnings, and exact files for manual inspection.

### 5.1 Required end-to-end tutorial notebook

Version 3 must deliver:

```text
notebooks/tutorial_create_full_prompt_new_game.ipynb
```

The notebook must not merely call an existing game prompt helper. It must teach
how to create a prompt for a hypothetical new game from the abstract contracts.
Use a small pedagogical game not otherwise implemented in `mas_cc`, with at
least:

```text
NewGameFullPrompt
├── description          fixed and required
├── rules                fixed and required
├── available_actions    dynamic and required
├── private_signal       dynamic and required
├── visible_memory       dynamic and required; empty is valid
└── optional_hint        dynamic and optional
```

The notebook must show, in separate executable cells:

1. where `PromptBlock`, `FullPrompt`, `ResponseContract`, compiled messages,
   token estimation, provider configuration, and preflight are implemented;
2. how to subclass `PromptBlock` for each semantic block;
3. how each block validates and renders its own value;
4. how to subclass `FullPrompt` and define authoritative order;
5. how fixed, dynamic, required, optional, `UNBOUND`, and empty values differ;
6. how immutable binding creates independent prompts for two agents;
7. how to inspect raw values, rendered blocks, omitted blocks, messages, token
   breakdown, definition hash, and instance hash;
8. how to construct the normalized `CompletionRequest`;
9. how to configure and preflight the University provider using live metadata;
10. how to configure and preflight the standard OpenAI provider using an
    auditable price source;
11. a separate runnable University completion section;
12. a separate runnable OpenAI completion section;
13. response-contract validation and actual cost from normalized usage;
14. how the notebook classes would be moved into `games/<new_game>/prompts.py`
    when the hypothetical game becomes production code.

The same compiled prompt messages must be sent to both providers. The committed
notebook contains no outputs, keys, account data, or internal URLs. Live calls
are enabled in clearly named controls for interactive use, while automated tests
override them without editing the committed notebook. A failed University call
must not prevent the separate OpenAI section from being run.

Notebook validation must include:

- JSON/`nbformat` validation;
- compilation of all Python cells, including top-level `await`;
- a complete non-network execution path;
- deterministic rendered-message and hash assertions;
- one optional manual University live run;
- one optional manual OpenAI live run;
- secret scanning of saved notebook content and generated public artifacts.

---

<a id="phase-1"></a>

# Phase 1 — Freeze the legacy package and create the successor shell

## Objective

Keep `mas_cc` beside the unchanged `naming_game` package and preserve import safety.

## Tasks

1. Preserve the baseline Git revision, environment, and historical inspection hashes.
2. Keep both packages discoverable and importable.
3. Keep the `mas-cc` CLI and phase inspection entry point.
4. Verify import creates no API client, credential read, model load, run, or Comet session.
5. Do not modify `src/naming_game`.
6. Rerun import-safety and legacy regression tests after Phase 3–6 realignment.
7. Record new results in a Version 3 directory without replacing old evidence.

## Inspection command

```bash
conda run -n MA-CC mas-cc inspect phase 1 \
  --output-dir inspection/phase_01_v3
```

## Inspection artifacts

```text
inspection/phase_01_v3/
├── report.md
├── manifest.json
├── environment.json
├── package_imports.txt
├── legacy_test_summary.txt
└── historical_artifact_hash_check.json
```

## Acceptance criterion

Both `naming_game` and `mas_cc` import without external work, and legacy tests
remain green.

---

<a id="phase-2"></a>

# Phase 2 — Core records and validated configuration

## Objective

Maintain immutable core records and secret-safe reusable component/run
configuration while adding the Version 3 prompt component schema.

## Tasks

1. Preserve immutable IDs, messages, seeds, timestamps, and validation results.
2. Preserve provider, game, execution, logging, storage, analysis, experiment,
   pricing, and budget configuration models.
3. Preserve component references, overrides, environment expansion, defaults,
   strict validation, and secret-free resolved export.
4. Add prompt component schema Version 2.
5. Remove `PromptContext` from new configuration and schema documentation.
6. Resolve a prompt family/version to a concrete full-prompt factory.
7. Export the resolved block manifest without serializing live private values.
8. Keep schema Version 1 readable only as a temporary migration input.
9. Produce precise errors for unknown prompt family/version, invalid options,
   and incompatible response contracts.
10. Verify configuration loading itself performs no prompt binding or provider work.

## Inspection command

```bash
conda run -n MA-CC mas-cc inspect phase 2 \
  --config configs/runs/provider_smoke_test_v3.yaml \
  --output-dir inspection/phase_02_v3
```

## Inspection artifacts

```text
inspection/phase_02_v3/
├── report.md
├── manifest.json
├── prompt_schema_v2.json
├── v1_to_v2_migration_examples.md
├── resolved_prompt_component.yaml
└── secret_scan.json
```

## Acceptance criterion

Configuration can select and inspect a concrete full prompt without importing a
game provider, reading credentials, or serializing dynamic private block values.

---

<a id="phase-3"></a>

# Phase 3 — Abstract value-bearing prompt composition

## Objective

Implement game-independent abstract prompt machinery in which concrete,
value-bearing blocks are composed by concrete `FullPrompt` classes.

## Tasks

1. Introduce `UNBOUND` with serialization-safe diagnostics.
2. Define abstract immutable `PromptBlock[T]`.
3. Define abstract immutable `FullPrompt`.
4. Define `RenderedPromptBlock` and `CompiledPrompt`.
5. Retain and adapt `ResponseContract` and local response validation.
6. Replace the registry mapping from `PromptDefinition` to renderer collections
   with family/version to concrete full-prompt factory.
7. Make block order authoritative in the concrete full prompt.
8. Implement immutable `.bind()` for dynamic values.
9. Distinguish required unbound, optional omitted, and valid empty values.
10. Render blocks independently and record per-block token estimates.
11. Compile ordered blocks into normalized messages using an explicit grouping policy.
12. Add definition and instance fingerprints.
13. Implement a game-neutral basic-choice example with fixed and dynamic blocks.
14. Demonstrate that prompt construction and token estimation require no provider.
15. Add contract tests for concurrency safety and absence of cross-agent value leakage.
16. Deprecate `PromptContext`, `PromptDefinition`, and renderer-only block APIs.
17. Remove those APIs from active Phase 5–9 paths after migration.
18. Implement the required new-game tutorial notebook through prompt compilation;
    provider execution cells are completed and validated with the Phase 4 adaptation.

## Example

```python
base = BasicChoiceFullPrompt(
    description="Choose one value.",
    rules=("Return a legal value.",),
    available_actions=UNBOUND,
)

bound = base.bind(available_actions=("A", "B"))
compiled = bound.compile(RegexTokenCounter())
```

## Inspection command

```bash
conda run -n MA-CC mas-cc inspect phase 3 \
  --prompt configs/components/prompts/basic_choice_v2.yaml \
  --output-dir inspection/phase_03_v3
```

## Inspection artifacts

```text
inspection/phase_03_v3/
├── report.md
├── manifest.json
├── full_prompt_definition.json
├── unbound_prompt.json
├── bound_prompt.json
├── block_manifest.json
├── rendered_blocks.json
├── omitted_blocks.json
├── compiled_messages.json
├── rendered_prompt.md
├── token_breakdown.csv
├── fingerprints.json
└── validation_examples.md
```

## Manual checks

- Read each concrete block value and rendered output separately.
- Confirm changing one bound value changes only its block and final instance hash.
- Confirm fixed blocks cannot be rebound silently.
- Confirm empty memory renders differently from an unbound memory block.
- Confirm required missing values fail before provider construction.
- Confirm two concurrently bound prompts do not share values.
- Confirm compiled messages are provider-independent.

## Acceptance criterion

A concrete full prompt can be constructed, partially bound, validated, rendered,
token-estimated, fingerprinted, and inspected without a game or provider, and no
active path requires the universal game-shaped `PromptContext`. The tutorial
notebook demonstrates the same construction without relying on an existing game helper.

---

<a id="phase-4"></a>

# Phase 4 — Normalized providers and provider economics

## Objective

Preserve the normalized provider and Version 2 economics contracts while
adapting their input fixture from `PromptComposer(PromptContext)` to a compiled
`FullPrompt`.

## Tasks

1. Keep `LLMProvider`, `CompletionRequest`, `CompletionResponse`, capabilities,
   usage, errors, registries, and adapters unchanged unless tests expose a real
   boundary violation.
2. Add a single conversion:

   ```python
   compiled_prompt.to_completion_request(...)
   ```

   or an equivalent runtime-owned factory.
3. Ensure `CompletionRequest` contains only normalized messages and request metadata.
4. Keep OpenAI, University, Gemma, and mock adapters unaware of full prompts and blocks.
5. Preserve live/cached/offline price sources, accounting units, freshness,
   cached-input arithmetic, long-context pricing, limits, and budget guards.
6. Make static preflight accept compiled prompt token information or the resulting request.
7. Bind preflight artifacts to prompt definition and instance hashes.
8. Send the same compiled messages through all providers in smoke tests.
9. Preserve secret-redaction and preflight-only defaults.
10. Rerun all provider-economics concurrency and unknown-price tests.
11. Complete and validate the University and OpenAI execution sections of
    `notebooks/tutorial_create_full_prompt_new_game.ipynb`.

## Inspection artifacts

```text
inspection/phase_04_v3/
├── report.md
├── manifest.json
├── full_prompt_definition.json
├── compiled_prompt.json
├── request.json
├── normalized_response.json
├── usage.json
├── pricing_snapshot.json
├── preflight_estimate.json
├── budget_status.json
└── provider_boundary_diff.md
```

## Manual checks

- Confirm adapters import no concrete game prompt modules.
- Compare exact request messages across providers.
- Confirm no block values or private game records leak into wire metadata.
- Confirm Phase 4 price and runtime arithmetic is unchanged for identical messages.

## Acceptance criterion

All providers execute the same normalized request produced from a bound full
prompt, the Version 2 pricing/budget guarantees remain intact, and the tutorial
notebook can exercise University and OpenAI independently.

---

<a id="phase-5"></a>

# Phase 5 — Generic game interface and toy game under game-owned prompts

## Objective

Define the generic game boundary so a decision request carries a concrete bound
full prompt rather than a universal prompt context.

## Tasks

1. Retain generic game state, observation, action, transition, and result records.
2. Replace `DecisionRequest.prompt_context` with a provider-independent bound
   `FullPrompt` or narrow `CompilablePrompt` protocol.
3. Keep observation construction responsible for information visibility.
4. Make each game convert its observation into its own concrete full prompt.
5. Implement `ToyCoordinationFullPrompt` and its concrete blocks in the toy game package.
6. Keep provider execution outside pure game transitions.
7. Compile the bound prompt in the generic runtime immediately before request creation.
8. Store prompt definition/instance hashes on the logical decision and attempt.
9. Replace `PromptContextScenario` in planning with `PromptScenario` containing a
   bound full prompt plus assumptions.
10. Estimate lower, representative, and maximum prompt scenarios by binding
    appropriate block values, not by fabricating universal context fields.
11. Verify deterministic state, prompt binding, and trajectory replay.

## Inspection artifacts

```text
inspection/phase_05_v3/
├── report.md
├── manifest.json
├── resolved_config.yaml
├── initial_state.json
├── observations.jsonl
├── bound_prompts.jsonl
├── compiled_prompts.jsonl
├── interactions.jsonl
├── final_state.json
├── game_call_plan.json
├── prompt_scenarios.json
└── trajectory.csv
```

## Manual checks

- Follow state → observation → bound full prompt → compiled messages → response → transition.
- Confirm the game owns its concrete blocks but not provider execution.
- Confirm the same game prompt runs unchanged with mock and live providers.
- Confirm prompt scenarios and call plans are identical across provider selections.

## Acceptance criterion

A complete toy game runs through the generic interface while carrying concrete,
game-owned bound prompts through a provider-neutral runtime.

---

<a id="phase-6"></a>

# Phase 6 — Naming Convention Game redesigned around concrete blocks

## Objective

Realign the implemented Naming Convention Game so its prompt is a concrete
`FullPrompt` with independently inspectable fixed and dynamic blocks.

## Required concrete prompt

```text
NamingConventionFullPrompt
├── description          fixed, required
├── rules                fixed, required
├── presented_actions    dynamic per agent decision, required
├── visible_memory       dynamic per agent decision, required; empty is valid
└── visible_score        dynamic per agent decision, required
```

The response contract is owned by the full prompt and produces the final
decision/output instruction. If additional semantic content is needed, it must
be introduced as a separately named, versioned block rather than folded into a
large `fixed_game` block.

## Tasks

1. Implement the five concrete block classes in
   `games/naming_convention/prompts.py`.
2. Implement `NamingConventionFullPrompt` with authoritative order.
3. Move fixed description and rules into fixed bound blocks.
4. Bind `presented_actions`, `visible_memory`, and `visible_score` separately for
   each agent from the same pre-interaction state.
5. Preserve anonymous local roles and the information boundary.
6. Keep empty memory distinct from an unbound memory block.
7. Ensure simultaneous pair decisions receive separate immutable prompt instances.
8. Remove the active `ashery_2025` `fixed_game`/context-renderer path after parity review.
9. Preserve or deliberately version the paper-faithful rendered request. Any
   intentional text difference requires a new prompt version and documented fixture.
10. Update response parsing, validation, and retries to consume the full prompt's contract.
11. Save selected block values, rendered blocks, compiled messages, hashes,
    responses, parsed actions, payoffs, and post-memory.
12. Build empty, representative, and maximum memory prompt scenarios using
    actual bound `VisibleMemoryBlock` values.
13. Preserve provider-independent call plans and Version 2 cost composition.
14. Compare deterministic trajectories and exact prompt fixtures before/after migration.

## Inspection command

```bash
conda run -n MA-CC mas-cc game run \
  --config configs/runs/naming_convention_smoke_test_v3.yaml \
  --output-dir inspection/phase_06_v3
```

## Inspection artifacts

```text
inspection/phase_06_v3/
├── report.md
├── manifest.json
├── resolved_config.yaml
├── full_prompt_definition.json
├── agents_initial.json
├── interactions.jsonl
├── selected_audit_traces.jsonl
├── selected_block_traces.jsonl
├── game_call_plan.json
├── prompt_token_scenarios.csv
├── prompt_parity_report.md
├── trajectory.csv
├── action_share.png
├── coordination_rate.png
└── agents_final.json
```

## Manual checks

- Inspect all five blocks independently for several agents and rounds.
- Change visible memory and confirm only the memory block and instance hash change.
- Change visible score and confirm only the score block and instance hash change.
- Confirm fixed description/rules remain identical across decisions.
- Confirm presented action order remains agent-specific and seeded.
- Confirm no agent IDs, global state, committee data, or counterpart current action leak.
- Confirm simultaneous prompts are bound from identical pre-transition state.
- Compare planned and actual calls and token scenarios.

## Acceptance criterion

The Naming Convention Game runs with a concrete five-block full prompt, preserves
its scientific information boundary and deterministic dynamics, and exposes
independently inspectable prompt values, rendering, tokens, and hashes.

---

<a id="phase-7"></a>

# Phase 7 — Storage, audit logging, heartbeats, and Comet ML

## Objective

Make multi-hour runs observable, recoverable, and auditable, with bound prompts
as first-class attempt artifacts.

## Tasks

1. Implement structured run, decision, retry, invalid-response, interaction,
   checkpoint, heartbeat, budget, and completion events.
2. Implement console, local log, JSONL audit, Comet, and in-memory metric sinks.
3. Add stable versioned schemas for interactions, provider attempts, runs, prompt
   traces, usage, and budget events.
4. Add atomic checkpoints and resume checks.
5. Add configurable deterministic, count-based detailed-audit selection. Routine
   decision, usage, cost, status, heartbeat, and error summaries are logged for
   every event; rendered prompts, prompt blocks, compiled messages, and complete
   audit paths are logged only when selected by the detailed-audit policy.
6. Make the detailed-audit policy understandable in experiment terms: it must
   support `log_every_n_rounds`, `always_log_first_n_rounds`,
   `max_logged_prompts_per_game`, and `max_logged_prompts_per_run`. It must also
   support always logging provider errors and invalid responses when configured.
   The selection must be deterministic for a fixed configuration and seed.
7. Enforce the per-game and per-run prompt-record caps before writing a detailed
   record. After a cap is reached, continue lightweight summary logging and
   record the number and reason for omitted detailed records. Write selected
   detailed records incrementally; do not retain an unbounded collection of them
   in memory. When detailed logging is disabled, do not create detailed-audit
   JSONL files merely as empty placeholders.
8. Keep Comet optional and obtain its credential only from the
   `COMET_API_KEY` environment variable. Local development may load that
   variable from the repository's untracked `.env` file; `.env.example` documents
   the required variable name and must contain only a placeholder, never a real
   key. Do not place the key in run configuration, checkpoints, logs, audit
   traces, manifests, or Comet metadata.
9. Log aggregated metrics remotely; keep complete scientific/private traces local.
10. Integrate the Phase 4 runtime budget guard and persist reservations,
   reconciliations, accumulated cost, accounting unit, and price-snapshot hash.
11. Restore accumulated usage and reservations safely on resume.
12. Persist prompt family/version, definition hash, and instance hash per attempt.
13. Persist block name/version, binding state, rendered content, and token count
   under the configured audit-sampling policy.
14. Mark sensitive/private blocks and prevent them from being sent to Comet by default.
15. Restore no mutable prompt state from checkpoints; reconstruct immutable bound
   prompts from state and resolved configuration.
16. Attribute estimated and actual usage to prompt stage and definition hash.
17. As the final Phase 7 end-to-end inspection, run the Naming Convention Game
   with normal local logging and Comet enabled. The fixture must complete at
   least three pair interactions with exactly one provider attempt per logical
   decision, hence two base attempts per ordinary pair interaction. Forced
   decisions and retries must be classified separately. Save the local inspection
   artifacts plus an inspectable Comet run reference or summary. This is an
   integration demonstration, not a substitute for the mock-based unit and
   resume checks.

### Example detailed-audit policy

```yaml
observability:
  detailed_prompt_audit:
    enabled: true
    log_every_n_rounds: 10
    always_log_first_n_rounds: 2
    always_log_provider_errors: true
    always_log_invalid_responses: true
    max_logged_prompts_per_game: 25
    max_logged_prompts_per_run: 1000
```

This example records the first two rounds of each game, then every tenth round,
as well as configured failures, until either prompt-record count cap is reached.
The caps are counts of detailed prompt records, not memory or file-size limits.

## Inspection command

```bash
conda run --live-stream -n MA-CC mas-cc inspect phase 7 \
  --config configs/runs/naming_convention_smoke_test_v3.yaml \
  --output-dir inspection/phase_07_v3
```

## Inspection artifacts

```text
inspection/phase_07_v3/
├── report.md
├── manifest.json
├── resolved_config.yaml
├── experiment.log
├── events.jsonl
├── api_call_status.jsonl
├── audit_traces.jsonl (only when detailed auditing selects records)
├── prompt_block_traces.jsonl (only when detailed auditing selects records)
├── usage_cost.jsonl
├── budget_events.jsonl
├── checkpoint_manifest.json
├── local_metrics.csv
├── comet_summary.json
└── observability_dashboard.png
```

## Manual checks

- Watch heartbeats and deliberately slowed mock requests.
- Compare Comet aggregates with local metrics.
- Kill and resume after a checkpoint without changing prompt hashes or budget state.
- Confirm no secret or private block content appears in Comet.
- Follow one sampled decision through observation, blocks, messages, usage, and transition.
- Run enough rounds to reach both detailed-prompt caps; confirm the configured
  first-round, interval, and failure records exist, later detailed records are
  omitted with a reason, and lightweight event/usage logging continues.
- Disable detailed prompt auditing and confirm `audit_traces.jsonl` and
  `prompt_block_traces.jsonl` are not created.
- Inspect the final Naming Convention Game Comet run: it must contain at least
  three pair interactions and two successful base prompt/provider attempts per
  ordinary interaction, and its aggregate metrics must agree with the corresponding local logs and `comet_summary.json`.
- Confirm a budget stop leaves a valid checkpoint and classified run status.

## Acceptance criterion

A long run is observable and restartable, and every detailed-audit record
selected within the configured per-game and per-run count caps can be traced
from game observation through bound blocks and compiled messages without
leaking private content to remote monitoring. Detailed logging remains bounded
by understandable round and prompt-record controls while lightweight operational
logging continues for the full run. The phase concludes with an inspectable,
three-or-more-interaction Naming Convention Game run using both Comet and normal
local logging, with exactly one provider attempt per logical decision and therefore
two base attempts per ordinary pair interaction, except for forced decisions or
auditable retries.

---

<a id="phase-80"></a>

# Phase 8.0 — Recording contracts, generic metrics, and metric logging

## Objective

Introduce a game-independent recording and metrics layer that makes new scientific
quantities easy to add without changing the game loop, storage backend, provider
layer, experiment orchestrator, or Comet adapter. This phase defines what is
persisted during execution, which metrics are computed during or at the end of an
episode, and how metric outputs are logged. Phase 8.1 remains a separate offline
analysis layer over completed datasets.

## Architectural separation

The architecture must distinguish four different objects:

1. **Raw records** are immutable facts emitted by the simulation and persisted
   incrementally, such as actions, participants, state snapshots, payoffs,
   interventions, and experimental conditions. Raw records are not metrics.
2. **Streaming metrics** consume the ordered raw-record stream and may emit values
   during execution, such as rolling action share or rolling coordination rate.
3. **Final metrics** are computed when an episode or experiment closes, such as
   first-consensus time, terminal convention, or mean final score.
4. **Offline analyses** consume completed datasets after execution, such as mutual
   information, conditional mutual information, InfoNCE, bootstrap intervals, and
   permutation nulls. These belong to Phase 8.1 and are not runtime metrics.

```text
simulation transition
    -> immutable raw record
    -> incremental persistence
    -> optional streaming-metric updates
    -> episode completion
    -> episode-final metric computation
    -> experiment completion
    -> experiment-final metric computation
    -> Phase 8.1 offline analyses over completed artifacts
```

A metric may be replayed from persisted raw records for validation or
recomputation. Replay is an execution mode of the same metric implementation; it
does not turn that metric into a Phase 8.1 scientific analysis.

## Recording plan

Before execution, the framework must build a versioned `RecordingPlan`. The plan
collects requirements from:

- the game and its mandatory audit contract;
- selected streaming and final metrics;
- stopping conditions and online control observations;
- configured control policies and intervention schedules;
- every Phase 8.1 analysis declared in the experiment's planned-analysis section.

For every required variable, the plan records whether it is:

- persisted directly in a named raw-record schema;
- deterministically derivable from persisted fields, with a versioned extractor;
- unavailable.

Preflight fails when any required input is unavailable. It must not silently
replace a missing variable with an approximation. The approved preflight is bound
to the recording-plan hash, raw-schema versions, extractor versions, metric
instances, and planned-analysis requirements.

The exact Python spelling may change, but the contract should express:

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class RecordingRequirement:
    record_type: str
    field: str
    minimum_schema_version: int
    cadence: str
    reason: str
    access_level: str

@dataclass(frozen=True)
class RecordingBinding:
    requirement: RecordingRequirement
    source: str
    mode: str  # "direct", "derived", or "unavailable"
    schema_version: int | None
    extractor: str | None
    extractor_version: int | None

@dataclass(frozen=True)
class RecordingPlan:
    bindings: tuple[RecordingBinding, ...]
    raw_schema_manifest: "RawSchemaManifest"
    fingerprint: str
```

Games expose versioned raw-record schemas rather than arbitrary mutable state.
Possible record families include:

- `InteractionRecord`;
- `DecisionRecord`;
- `StateSnapshotRecord`;
- `ControlEventRecord`;
- `EpisodeRecord`;
- `ProviderAttemptRecord`.

A game does not need to emit every record family. The resolved recording plan
selects only the records and cadence required for that run. State snapshots may,
for example, be persisted after every interaction, every population round, only
at episode boundaries, or not at all when the required state is reconstructible
from interactions.

## Abstract metric contracts

Production code must use abstract base classes or an equivalently strict typed
interface. Concrete scientific metrics are subclasses; their parameters create
immutable metric instances.

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")
MetricStateT = TypeVar("MetricStateT")

@dataclass(frozen=True)
class MetricSpec:
    name: str
    version: int
    scope: str
    timing: str
    required_inputs: tuple[RecordingRequirement, ...]
    output_schema: "MetricOutputSchema"
    access_level: str
    export_capability: str

class Metric(ABC, Generic[InputT, OutputT]):
    spec: MetricSpec

    @abstractmethod
    def validate_input(self, value: InputT) -> "ValidationResult": ...

class StreamingMetric(
    Metric[InputT, OutputT],
    Generic[InputT, OutputT, MetricStateT],
):
    @abstractmethod
    def initialize(self, context: "MetricContext") -> MetricStateT: ...

    @abstractmethod
    def update(
        self,
        state: MetricStateT,
        record: InputT,
    ) -> tuple[MetricStateT, tuple[OutputT, ...]]: ...

    @abstractmethod
    def finalize(self, state: MetricStateT) -> tuple[OutputT, ...]: ...

class FinalMetric(Metric[InputT, OutputT], ABC):
    @abstractmethod
    def compute_final(self, value: InputT) -> OutputT: ...

class AgentMetric(Metric["AgentMetricView", OutputT], ABC):
    pass

class SystemMetric(Metric["SystemMetricView", OutputT], ABC):
    pass

class InteractionMetric(Metric["InteractionMetricView", OutputT], ABC):
    pass

class EpisodeMetric(FinalMetric["EpisodeMetricView", OutputT], ABC):
    pass

class ExperimentMetric(FinalMetric["ExperimentMetricView", OutputT], ABC):
    pass

class AggregateMetric(Metric["MetricOutputCollection", OutputT], ABC):
    @abstractmethod
    def reduce(self, values: "MetricOutputCollection") -> OutputT: ...
```

Scope and timing are orthogonal. A concrete metric may therefore be an agent-level
streaming metric, a system-level streaming metric, an episode-final metric, or an
aggregate over lower-level metric outputs. The implementation may use ABC mixins,
composed protocols, or another strict typed design, but these distinctions must
remain explicit.

Examples of configured instances are:

```python
RollingActionShare(action="Q", window_interactions=30)
RollingCoordinationRate(window_interactions=30)
AgentStateIndicator(target_state="two_items")
MeanAcrossAgents(source_metric="agent_state_indicator")
FirstConsensusTime(threshold=0.95, window_interactions=30)
```

An aggregate over agent metrics is explicit and compositional. Direct system
metrics remain necessary for quantities such as consensus, network connectivity,
polarization, or global coordination that are not correctly represented as a
reduction of independent agent values.

All metric outputs include metric name, version, instance ID, scope, entity or
episode ID, logical time when applicable, parameter hash, output-schema version,
and provenance hash.

## Metric ownership and registration

1. The generic package owns the abstract classes, typed input views, registry,
   execution engines, persistence, validation, sink adapters, and manifests.
2. Each game owns its concrete scientific metrics under
   `games/<game>/metrics.py` or a game-local metrics package.
3. Cross-game operational metrics such as latency, retries, tokens, cost, and
   throughput live in a generic operational-metrics package.
4. `MetricRegistry` resolves `name@version` to a metric class and constructs an
   immutable parameterized instance.
5. Games expose available and recommended metric sets. Experiments explicitly
   select, remove, and parameterize metrics.
6. Metric implementations never import Comet or write files directly. They return
   typed values to generic sinks.
7. Phase 8.1 analyses use a separate `AnalysisRegistry` and do not masquerade as
   runtime metric subclasses.

## Run, episode, and planned-analysis configuration

The resolved run or experiment configuration selects raw recording, metric
instances, and planned offline analyses. Episodes inherit the run-level metric
configuration. Episode-specific overrides are permitted only when they are part
of the experimental design and are included in the episode fingerprint.

Each metric instance configures independently:

- implementation name and version;
- stable instance ID and parameters;
- streaming, episode-final, or experiment-final timing;
- entity selection, such as all agents or selected agent IDs;
- computation cadence where applicable;
- local persistence and flush cadence;
- console logging;
- Comet export;
- summary aggregation;
- optional plot or report rendering.

Example:

```yaml
recording:
  schema_version: 1
  raw:
    interactions:
      enabled: true
    state_snapshots:
      enabled: auto
      cadence:
        every_n_interactions: 1
    control_events:
      enabled: auto
  planned_analyses:
    - id: terminal_policy_information
      analysis: information.discrete_mutual_information@1
      variables:
        x: committee_policy
        y: terminal_convention
    - id: lagged_policy_information
      analysis: information.conditional_mutual_information@1
      variables:
        x: committee_policy
        y: macrostate_future
        condition: macrostate_now
      parameters:
        horizon_population_rounds: 1

metrics:
  defaults:
    local:
      enabled: true
      format: parquet
    console:
      enabled: false
    comet:
      enabled: false

  instances:
    - id: action_share_q
      metric: naming_convention.rolling_action_share@1
      timing: streaming
      scope: population
      parameters:
        action: Q
        window_interactions: 30
      computation_cadence:
        every_n_interactions: 1
      sinks:
        local:
          enabled: true
          flush_every_n_values: 20
        comet:
          enabled: true
          key: convention/action_share_q
          log_every_n_values: 5

    - id: agent_7_q_count
      metric: naming_convention.agent_action_count@1
      timing: streaming
      scope: agent
      entity_selector:
        agent_ids: [7]
      parameters:
        action: Q
      sinks:
        local:
          enabled: true
        comet:
          enabled: false

    - id: first_consensus
      metric: naming_convention.first_consensus_time@1
      timing: episode_final
      scope: episode
      parameters:
        threshold: 0.95
        window_interactions: 30
      sinks:
        local:
          enabled: true
        comet:
          enabled: true
          key: convention/first_consensus_time
          summary_only: true
```

The configuration resolver and preflight must fail when:

- a metric or planned-analysis name/version is unknown;
- parameters or entity selectors are invalid;
- a required raw field is unavailable and cannot be deterministically derived;
- an extractor or raw-schema version is incompatible;
- a streaming cadence is impossible for the selected record source;
- Comet is requested for a non-scalar or restricted metric;
- a requested sink exceeds the metric's access or export policy.

## Metric computation, persistence, and Comet policy

Metric computation and metric emission are separate concerns. A metric may be:

- computed during execution and persisted locally as a time series;
- computed during execution only for a stopping condition or control observation;
- finalized at episode completion and stored as one episode-level result;
- finalized at experiment completion and stored as one aggregate result;
- replayed after the run from raw records to verify deterministic parity;
- emitted to console for debugging;
- sent to Comet as an approved scalar time series or summary;
- stored locally without any remote export.

Computation cadence and sink cadence are distinct. For example, a rolling metric
may update after every interaction, flush locally every 20 values, and send every
fifth value to Comet.

The metric class declares an upper-bound export capability:

```text
local_only
scalar_aggregate_only
scalar_time_series
public_artifact
```

Configuration may choose a stricter policy but never a more permissive one. Comet
export is opt-in per metric instance unless a run-level default enables a named,
approved set. Per-agent identifiers, private memory, evaluator-only records,
vectors, tables, and sensitive values remain local by default. Raw records are
not sent to Comet through the metric sink.

If a streaming metric participates in a stopping condition or control policy,
its class version, parameters, update cadence, and state-restoration semantics
become part of the data-generating configuration and episode fingerprint.

## Data access and scientific safety

Each metric consumes the narrowest immutable typed view it requires:
`AgentMetricView`, `SystemMetricView`, `InteractionMetricView`,
`EpisodeMetricView`, or `ExperimentMetricView`. Metrics do not reach into mutable
live game objects.

Evaluator-only or private state may be used for permitted local scientific
metrics, but metric values never enter ordinary agent prompts, transitions, or
control policies unless the experiment explicitly declares that metric as an
observable control input. Such use is separately configured, fingerprinted, and
audited.

## Storage and provenance

Every run saves both a recording manifest and a metric manifest.

The recording manifest contains:

- raw record families, schema versions, and persistence cadence;
- every requested field and the component that requested it;
- direct versus derived bindings;
- extractor names, versions, and hashes;
- unavailable requirements and preflight status;
- recording-plan fingerprint.

The metric manifest contains:

- selected metric classes, versions, and instance IDs;
- scope, timing, entity selectors, parameters, and hashes;
- required input fields and bound recording sources;
- computation and sink cadence;
- output schemas and artifact locations;
- requested and effective export policies;
- Comet keys and summary/time-series policy;
- success, skipped, failed, and non-computable status with reasons.

Canonical production results use:

```text
results/
└── <game_name>/
    └── <experiment_name>/
        └── <run_id>/
            ├── manifest.json
            ├── config/
            │   ├── requested_config.yaml
            │   ├── resolved_config.yaml
            │   ├── recording_plan.json
            │   ├── metric_manifest.json
            │   └── planned_analysis_manifest.json
            ├── data/
            │   ├── recording_manifest.json
            │   ├── interactions.parquet
            │   ├── decisions.parquet              # only when selected
            │   ├── state_snapshots.parquet         # only when selected
            │   ├── control_events.parquet          # only when selected
            │   ├── provider_attempts.parquet
            │   └── episodes.parquet
            ├── metrics/
            │   ├── manifest.json
            │   ├── streaming/
            │   ├── episode_final/
            │   ├── experiment_final/
            │   ├── comet_export_manifest.json
            │   └── plots/
            ├── logs/
            ├── audit/
            ├── checkpoints/
            └── analysis/
```

Files marked "only when selected" are omitted when the recording plan does not
require them. Checkpoint shards are recoverable intermediate state. Compacted raw
records under `data/` remain the scientific source of truth.

## Alignment with the Naming Convention and empowerment datasets

For the Naming Convention Game, the raw recording plan required by the baseline
metrics and the Phase 8.1 empowerment analyses must preserve or make
reconstructible at least:

- experiment, condition, episode, replicate, seed, and design-stratum identity;
- interaction index and population-round index;
- both participant IDs and both selected actions;
- payoff, success, and pre/post score information required by configured metrics;
- committee policy, committee membership when scientifically required, schedule,
  intervention status, and forced-action flags;
- topology or graph version when it varies;
- episode start, completion, failure, and terminal-status information;
- enough ordered trajectory information to reconstruct configured macrostates.

Rolling action shares, rolling coordination, dominant convention, macrostates,
and resolution labels are derived metric or extractor outputs. They may be stored
for convenience and efficiency, but the raw dataset must contain enough
information to recompute them under their recorded versions.

For the main Phase 8.1 estimands, the recording plan must be able to construct:

- `G`: the configured committee policy or intervention regime;
- `Y`: the versioned terminal outcome extractor;
- `S_t` and `S_{t+h}`: versioned state extractors over the ordered trajectory;
- experimental strata used for conditional estimation and null permutations.

The old `interactions.parquet` and `episodes.parquet` remain conceptually valid,
but their schemas must be versioned and validated against this explicit recording
plan rather than relied upon implicitly.

## Baseline Naming Convention metric fixture

The first concrete metric set implements:

- rolling action share for every legal action;
- rolling coordination rate;
- dominant action and dominant-action share;
- per-agent action and participation counts;
- payoff summaries;
- population aggregates over agent-level metrics;
- resolved convention and first-resolution time as episode-final metrics;
- across-episode convention and time-to-resolution summaries as experiment-final
  metrics.

These are game-owned subclasses. They are not hard-coded into runtime, storage,
reporting, or Comet integration.

## Tasks

1. Implement versioned raw-record schemas and immutable metric input views.
2. Implement `RecordingRequirement`, `RecordingBinding`, `RecordingPlan`, and a
   preflight builder that unions game, metric, control, stopping-rule, and planned-
   analysis requirements.
3. Implement direct and deterministic-derived input bindings with versioned
   extractors; fail closed for unavailable inputs.
4. Implement `MetricSpec`, metric value/provenance records, and output schemas.
5. Implement abstract `Metric`, `StreamingMetric`, `FinalMetric`, `AgentMetric`,
   `SystemMetric`, `InteractionMetric`, `EpisodeMetric`, `ExperimentMetric`, and
   `AggregateMetric` contracts.
6. Implement `MetricRegistry` and immutable parameterized instances.
7. Implement configuration for raw recording, metric selection, entity selectors,
   computation cadence, sink cadence, persistence, Comet export, and planned
   analyses.
8. Execute streaming metrics only from immutable emitted records and checkpoint
   their internal state when required for exact resume.
9. Execute episode-final and experiment-final metrics only after the corresponding
   dataset boundary is complete.
10. Support deterministic metric replay from persisted raw records and compare it
    with online outputs.
11. Implement aggregation over agent metrics and direct system metrics as distinct
    paths.
12. Persist recording and metric manifests, values, statuses, failures, schema
    versions, extractor versions, and effective export policies.
13. Add generic console, local-file, in-memory, and Comet sinks. Metric subclasses
    contain no sink-specific code.
14. Add optional plot/report renderers through a separate registry.
15. Implement the Naming Convention baseline metric fixture and a toy agent metric
    with a population aggregate.
16. Register Phase 8.1 analysis requirements so preflight can validate future
    analyzability before an experiment begins.
17. Test that a new metric requires no changes to runtime, providers, storage,
    experiment orchestration, or Comet code.
18. Test versioning, missing inputs, derivation bindings, invalid selectors,
    deterministic replay, resume, cadence, privacy, sink failures, and export
    restrictions.
19. Test that Comet may be enabled for one metric and disabled for another while
    local persistence continues for both.
20. Test that episode overrides, metric-controlled stopping rules, and recording-
    plan changes alter the appropriate fingerprints.

## Inspection command

```bash
conda run -n MA-CC mas-cc inspect metrics \
  --config configs/runs/naming_convention_baseline_v3.yaml \
  --output-dir inspection/phase_08_0_v3
```

## Inspection artifacts

```text
inspection/phase_08_0_v3/
├── report.md
├── manifest.json
├── resolved_config.yaml
├── recording_plan.json
├── recording_requirement_matrix.csv
├── raw_schema_manifest.json
├── extractor_manifest.json
├── metric_registry.json
├── selected_metric_manifest.json
├── planned_analysis_manifest.json
├── raw_data_fixture/
├── metric_values/
├── streaming_replay_parity.csv
├── sink_delivery_status.json
├── comet_export_manifest.json
├── naming_convention_baseline/
└── toy_metric_fixture/
```

## Manual checks

- Add a new concrete metric subclass without editing runtime, provider, storage,
  experiment orchestration, or Comet code.
- Instantiate one class with different parameters and confirm distinct instance
  IDs and provenance hashes.
- Compute one agent metric and aggregate it over all agents.
- Confirm a direct system metric and an aggregate over agent metrics remain distinct.
- Enable Comet for one metric and disable it for another while storing both locally.
- Confirm non-scalar and restricted values cannot be exported to Comet.
- Confirm computation, persistence, console, and Comet cadences are independent.
- Kill and resume a streaming metric without changing its emitted sequence.
- Recompute deterministic streaming metrics from raw Parquet records and confirm
  equality with the online outputs.
- Remove one field required by a planned Phase 8.1 analysis and confirm preflight
  fails with a precise missing-input report.
- Confirm derived variables name their extractor and version rather than appearing
  as unexplained columns.
- Confirm metric values never enter prompts, transitions, or controls without an
  explicit audited observation contract.

## Acceptance criterion

A researcher can add a game-specific or cross-game metric by implementing one
versioned metric subclass, registering it, and selecting a parameterized instance
in configuration. The framework supports agent, system, interaction, aggregate,
streaming, episode-final, and experiment-final metrics. Every run has an explicit
recording plan that guarantees the raw or deterministically derivable inputs
required by its selected metrics, controls, stopping conditions, and planned
Phase 8.1 analyses. Each metric independently configures local persistence,
logging cadence, and approved Comet export. No new metric requires changes to the
game loop, provider, storage backend, experiment orchestrator, or Comet adapter.

---

<a id="phase-81"></a>

# Phase 8.1 — Control policies and offline information-theoretic analysis

## Objective

Implement control policies and a separate offline-analysis framework for mutual
information, conditional mutual information, InfoNCE, uncertainty estimates, and
null models. Analyses consume completed, versioned artifacts produced under the
Phase 8.0 recording plan; they do not participate in game transitions or provider
execution.

## Offline analysis contract

Offline analyses are not subclasses of runtime metrics. They have their own
versioned abstraction because they operate on completed datasets, may combine
multiple episode strata, and can produce tables, distributions, diagnostics, and
plots rather than a scalar stream.

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass(frozen=True)
class AnalysisSpec:
    name: str
    version: int
    required_inputs: tuple[RecordingRequirement, ...]
    required_metric_inputs: tuple[str, ...]
    output_schemas: tuple["AnalysisOutputSchema", ...]
    access_level: str

class OfflineAnalysis(ABC):
    spec: AnalysisSpec

    @abstractmethod
    def validate_dataset(
        self,
        dataset: "CompletedExperimentDataset",
    ) -> "ValidationResult": ...

    @abstractmethod
    def run(
        self,
        dataset: "CompletedExperimentDataset",
    ) -> "AnalysisResultBundle": ...
```

An analysis may consume raw records, versioned extractors, and explicitly named
metric outputs. It may not assume that a convenient derived column is scientifically
well defined merely because it exists in a Parquet file. Every derived variable
must identify its extractor or metric version and parameters.

## Dependency on Phase 8.0

Each configured analysis declares its requirements before the experiment starts.
Phase 8.0 includes those requirements in the `RecordingPlan`. After the run,
Phase 8.1 validates the completed recording manifest and refuses analysis when:

- required raw fields are missing;
- raw schema or extractor versions are incompatible;
- episodes are incomplete or mixed across incompatible fingerprints;
- a required metric output cannot be recomputed or located;
- strata, policy labels, time indices, or trajectory ordering are ambiguous.

The analysis stage may be rerun repeatedly without model or provider access. It
must never mutate raw records or retroactively alter episode completion status.

## Control requirements

Control policies remain runtime components, but their scientific variables must
be recordable through Phase 8.0. A policy declares:

- membership and selection rules;
- observation privileges;
- intervention schedule and duration;
- forcing or advisory semantics;
- provider-call adjustments;
- raw fields and control events required for later analysis.

Forced actions skip prompt binding, compilation, and provider requests unless an
audit-only prompt is explicitly requested. Any policy that exposes additional
information to an agent must do so through the game's observation construction.

## Information-variable extraction

The Naming Convention empowerment analyses use versioned extractors for:

```text
G     = committee policy or intervention regime
Y     = terminal outcome in {A, B, unresolved}
S_t   = macrostate extracted from the trajectory at time t
S_t+h = macrostate after the configured horizon
Z     = experimental stratum or conditioning variables
```

The extractor manifest records rolling-window definitions, thresholds, tie rules,
time units, horizon conversion, action labels, and missing-data behavior. Changing
any of these creates a new extractor version or parameter hash.

## Tasks

1. Define control policies, committees, schedules, interventions, budgets, and
   observations.
2. Separate membership, observation privileges, policy, forcing, duration, and
   budget.
3. Make every control policy declare Phase 8.0 recording requirements and provider-
   independent call-plan adjustments.
4. Implement no-control, always-A/B, incumbent-support, alternative-promotion,
   and temporary-pulse fixtures.
5. Implement `AnalysisSpec`, `OfflineAnalysis`, `AnalysisRegistry`, completed-
   dataset views, validation, manifests, and result bundles.
6. Implement versioned extractors for `G`, `Y`, binary and three-state `S_t`,
   `S_{t+h}`, and experimental strata.
7. Implement discrete mutual information with plug-in, Jeffreys, optional Miller-
   Madow, episode bootstrap, null permutations, and diagnostics.
8. Implement conditional mutual information for lagged trajectory analyses.
9. Implement an InfoNCE pipeline separating extraction, embeddings, pair
   construction, objective, estimate, and diagnostics.
10. Validate estimators on independent, perfectly dependent, partially dependent,
    and recorded Naming Convention fixtures.
11. Bind each analysis result to the recording-plan fingerprint, input artifact
    hashes, extractor versions, analysis version, and parameters.
12. Store analysis outputs separately from raw data and runtime metric outputs.
13. Permit optional Comet export only through an analysis-specific, opt-in summary
    sink; complete tables and private records remain local by default.
14. Ensure analysis reads no live game objects, invokes no game provider, and does
    not alter simulation checkpoints or completion status.
15. Fail with a precise missing/incompatible-data report rather than silently
    dropping required variables or episodes.

## Inspection commands

```bash
conda run -n MA-CC mas-cc control demo \
  --config configs/runs/naming_convention_control_demo_v3.yaml \
  --output-dir inspection/phase_08_1_v3/control

conda run -n MA-CC mas-cc analyze information \
  --config configs/runs/information_estimator_validation.yaml \
  --output-dir inspection/phase_08_1_v3/analysis
```

## Inspection artifacts

```text
inspection/phase_08_1_v3/
├── report.md
├── manifest.json
├── control/
│   ├── intervention_schedule.csv
│   ├── control_recording_requirements.json
│   ├── interactions.jsonl
│   ├── forced_actions.csv
│   └── control_call_plan_adjustment.json
└── analysis/
    ├── analysis_manifest.json
    ├── input_validation.json
    ├── recording_plan_link.json
    ├── extractor_manifest.json
    ├── synthetic_dataset.parquet
    ├── discrete_mi_results.csv
    ├── conditional_mi_results.csv
    ├── contingency_tables.csv
    ├── infonce_results.csv
    ├── null_results.csv
    └── estimator_diagnostics.md
```

## Manual checks

- Verify exact controlled interactions and skipped prompt/provider work.
- Confirm all control variables needed for analysis appear in the recording plan.
- Compare base and adjusted call plans.
- Inspect the `G`, `Y`, `S_t`, and `S_{t+h}` extraction tables and fingerprints.
- Inspect contingency tables, bootstrap episode IDs, null permutations, and
  InfoNCE pairs.
- Confirm expected estimator behavior on synthetic fixtures.
- Remove a required field and confirm the analysis fails explicitly.
- Change an extractor threshold and confirm the result fingerprint changes.
- Confirm the same completed dataset can be analyzed repeatedly without provider
  access and without modifying raw artifacts.

## Acceptance criterion

Control policies have auditable effects on observations, actions, and provider
call demand, and they declare every raw variable required for later evaluation.
Offline analyses are separately versioned components that run only on completed
artifacts validated against the Phase 8.0 recording plan. Mutual information,
conditional mutual information, InfoNCE, bootstrap, and null analyses can be
reproduced from persisted records without game or provider access, and missing or
incompatible inputs cause an explicit failure rather than an implicit approximation.

---

<a id="phase-9"></a>

# Phase 9 — Full experiment orchestration and committee empowerment pilot

## Objective

Compose all validated components into reproducible, priced, approved experiments.

## Tasks

1. Implement experiment/episode specs, strata, grids, repetitions, deterministic
   IDs and seeds, concurrency, checkpoint/resume, compaction, and post-run analysis.
2. Compose game, concrete full prompt, provider, control, execution, logging,
   recording plan, metrics, storage, pricing, budget, and offline-analysis components.
3. Implement the committee-empowerment pilot grid.
4. Expand concrete game prompt types and versions with the experiment grid.
5. Build token scenarios from bound full prompts for each decision stage and
   representative dynamic-block regime.
6. Resolve lower/expected/conservative interactions, calls, retries, input/output
   tokens, cost, runtime, disk use, provider limits, and unknowns.
7. Bind preflight approval to:
   - resolved game and experiment configuration;
   - full-prompt family/version;
   - ordered block manifest and definition hash;
   - scenario instance hashes;
   - provider/model and pricing snapshot;
   - limits and estimation assumptions.
8. Bind approval to the recording-plan fingerprint, raw-schema and extractor
   versions, selected metric instances, and planned-analysis manifest.
9. Invalidate approval when a block class/version/order, response contract,
   prompt binding assumption, recording requirement, metric instance, analysis
   requirement, extractor version, or price quote changes.
10. Require explicit live launch and approved preflight ID.
11. Revalidate model availability/pricing immediately before launch.
12. Run episodes concurrently without weakening within-episode causal order or
    atomic budget reservations.
13. Compact only complete episode shards, validate the recording manifest, and run
   offline analysis only after safe storage.
14. Compare estimated and actual calls, block tokens, message tokens, cost,
   runtime, and disk use.
15. Produce final scientific, operational, recording, metric, analysis-provenance,
   and cost reports.

## Inspection commands

```bash
conda run -n MA-CC mas-cc experiment preflight \
  --config configs/runs/committee_empowerment_pilot_v3.yaml \
  --output-dir inspection/phase_09_v3/preflight

conda run --live-stream -n MA-CC mas-cc experiment run \
  --config configs/runs/committee_empowerment_pilot_v3.yaml \
  --approve-preflight inspection/phase_09_v3/preflight/preflight_id.txt \
  --output-dir inspection/phase_09_v3/run
```

## Inspection artifacts

```text
inspection/phase_09_v3/
├── report.md
├── manifest.json
├── preflight/
│   ├── resolved_config.yaml
│   ├── experiment_grid.csv
│   ├── game_call_plans/
│   ├── prompt_definitions/
│   ├── prompt_token_scenarios.csv
│   ├── recording_plan.json
│   ├── metric_manifest.json
│   ├── planned_analysis_manifest.json
│   ├── pricing_snapshot.json
│   ├── cost_estimate.json
│   ├── runtime_estimate.json
│   ├── budget_status.json
│   └── preflight_id.txt
└── run/
    ├── approved_preflight.json
    ├── events.jsonl
    ├── data/
    │   ├── recording_manifest.json
    │   ├── interactions.parquet
    │   ├── provider_attempts.parquet
    │   └── episodes.parquet
    ├── metrics/
    ├── experiment_summary.csv
    ├── analysis_results/
    └── summary.md
```

## Manual checks

- Inspect grid, prompt definitions/scenarios, prices, units, limits, and assumptions.
- Confirm any prompt, recording-plan, metric, extractor, planned-analysis, or
  value-assumption change invalidates approval.
- Compare estimated and actual calls, tokens, costs, runtime, and disk use.
- Inspect interaction/control traces, checkpoint recovery, Comet/local parity,
  MI tables, and InfoNCE diagnostics.

## Acceptance criterion

A complete experiment can be configured, prompt-resolved, recording-planned,
metric-configured, priced, approved, executed, monitored, resumed, analyzed, and
reproduced from saved artifacts.

---

<a id="wrap-up"></a>

## 6. Version 3 adoption boundary

At adoption:

1. Freeze current Phase 1–6 tests and inspection artifacts as Version 2 baseline evidence.
2. Implement `02082026_realigningment_phase_3456.md` in order.
3. Do not begin Phase 7 until Phase 3–6 Version 3 gates all pass.
4. Treat Phase 4 provider adapters/economics as stable unless a contract test fails.
5. Do not modify `src/naming_game`.
6. Keep temporary compatibility adapters clearly marked and remove them from
   active runtime paths before the realignment is accepted.
7. Create the required new-game tutorial notebook after the production Version 3
   prompt API is stable, then migrate the earlier HiddenBench provider notebook.

---

## 7. Cross-phase test strategy

### Unit tests

- block value validation and rendering;
- immutable binding and fixed-value protection;
- required/optional/empty semantics;
- message grouping and token arithmetic;
- definition and instance hashing;
- game transitions and provider economics.

### Contract tests

- every concrete block satisfies `PromptBlock`;
- every concrete game prompt satisfies `FullPrompt`/`CompilablePrompt`;
- every game decision supplies a fully bound prompt;
- every provider accepts the same normalized request;
- planning uses bound prompt scenarios without provider imports;
- control and storage respect prompt privacy;
- recording plans satisfy all selected metric and planned-analysis requirements;
- runtime metrics and offline analyses remain separate contracts.

### Inspection tests

- expected artifacts exist and validate;
- block order and hashes are deterministic;
- sensitive values and secrets are absent from public artifacts;
- compiled messages match approved fixtures;
- plots and reports are non-empty;
- historical directories remain unchanged.

### Regression matrix

```text
Phase 3 change -> rerun Phases 1–6 tests
Phase 4 change -> rerun provider, pricing, budget, and game runtime tests
Phase 5 change -> rerun toy and naming generic-game contract tests
Phase 6 change -> rerun naming parity, concurrency, planning, and inspection tests
Phase 8.0 change -> rerun recording, metric replay, resume, sink, and Phase 8.1 input tests
Phase 8.1 change -> rerun analysis-requirement preflight and estimator validation tests
```

---

## 8. Dependency direction

```text
core
  ↓
configuration
  ↓
abstract prompt contracts and provider protocols
  ↓
concrete game prompts + pure game mechanics
  ↓
provider adapters and generic execution runtime
  ↓
planning, storage, observability, and recording plans
  ↓
runtime metrics and control
  ↓
offline analysis
  ↓
experiments
  ↓
CLI
```

Operational rules:

```text
Abstract prompts know no games.
Concrete game prompts know no providers.
Games own observations and concrete prompt binding.
Providers know only normalized requests.
Planning composes demand, compiled token scenarios, and prices.
Storage records prompt provenance but does not render prompts.
Recording plans bind scientific requirements to versioned persisted schemas.
Metrics consume immutable records and know no sink implementation.
Offline analyses consume completed datasets and never call providers.
Control changes decisions/observations, not shared prompt objects.
```

---

## 9. Definition of completion

The reorganization is complete when this workflow succeeds:

```text
1. Select a game and its concrete full-prompt version.
2. Inspect the prompt's ordered block definition.
3. Construct an agent observation.
4. Bind a new immutable full prompt from that observation.
5. Validate required, optional, and empty block states.
6. Render and inspect each block independently.
7. Compile normalized messages and estimate block/total tokens.
8. Follow the tutorial notebook to create a new concrete prompt from scratch.
9. Send its identical compiled messages through University and OpenAI sections.
10. Build provider-independent game call and prompt-scenario plans.
11. Fetch an auditable provider price quote.
12. Estimate lower, expected, and conservative tokens, cost, and runtime.
13. Approve a preflight bound to prompt/game/provider fingerprints.
14. Execute through the normalized provider and runtime budget guard.
15. Audit selected state → observation → blocks → messages → response → transition paths.
16. Resume without restoring mutable cross-agent prompt state.
17. Build and inspect a recording plan covering game, metric, control, stopping,
    and planned-analysis requirements.
18. Compute configured streaming and final metrics, persist them locally, and
    export only approved scalar values to Comet.
19. Replay deterministic metrics from raw artifacts and verify parity.
20. Apply controls and validate separately versioned offline information analyses.
21. Expand to a complete experiment grid.
22. Compare estimated and actual demand and cost.
23. Reproduce the run from saved configuration, schemas, extractors, versions,
    manifests, and fingerprints.
```

The architecture is accepted only when each step remains independently visible
and no universal game-shaped prompt context is required.
