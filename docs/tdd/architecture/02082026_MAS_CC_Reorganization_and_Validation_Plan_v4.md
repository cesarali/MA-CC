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

# Phase 8.0 — Baseline convention metrics and canonical results organization

## Objective

Establish the descriptive scientific metrics and production-results contract
required to determine whether a convention emerges before introducing control
policies or information-theoretic estimators.

## Metric definitions

For a trailing window of `L` pair interactions, let `x[u,1]` and `x[u,2]` be the
two actions produced at interaction `u`. For every legal action `a`, compute:

```text
rolling_action_share[a] =
    count of action a across both player outputs in the window
    / number of player outputs in the window
```

Compute the rolling coordination rate as:

```text
rolling_coordination_rate =
    count of interactions with x[u,1] == x[u,2] in the window
    / number of interactions in the window
```

Also record:

- `dominant_action`: the action with the largest rolling action share;
- `dominant_action_share`: that largest share;
- `resolved`: whether one population-wide convention has emerged;
- `resolved_action`: the convention label when resolved.

The default resolution window is `L = 3 * base_population_size`. A convention is
resolved when both the rolling coordination rate and one action's rolling share
are at least the configured threshold, with `0.95` as the default. A value of
`1.0` means every action produced in the window used the same convention. Do not
interpret this game as giving every agent a persistent categorical
`current_convention` state unless a separate, explicitly defined estimator is
introduced.

## Tasks

1. Persist raw interaction observables before any derived analysis, including
   episode ID, interaction index, population round, both agent IDs, both actions,
   success, payoff, forced-decision flags, and prompt/provider-attempt references.
2. Derive per-interaction or per-population-round trajectories containing the
   rolling coordination rate, one rolling share per legal action, dominant action,
   dominant share, mean payoff, resolved status, and resolved action.
3. Produce one episode summary containing whether consensus was reached, the
   consensus action, first consensus interaction and population round, terminal
   coordination rate, terminal action shares, terminal dominant action, and
   terminal dominant share.
4. Across repeated episodes, report mean coordination and action-share
   trajectories, probability of resolution by time, distribution of final
   conventions, time-to-resolution statistics, and the unresolved fraction at
   the configured horizon.
5. Keep these descriptive metrics independent of Phase 8.1 control and
   information-theoretic estimators. Phase 8.1 may consume them but must not
   redefine them.
6. Count one logical decision for each agent action. An ordinary pair interaction
   therefore produces two prompt instances and two base provider attempts.
   Forced decisions skip the corresponding provider attempt; validation or
   provider retries are recorded separately from base logical decisions.
7. Implement one canonical production-results hierarchy under a unique run ID.
   Inspection directories remain software-validation evidence and must not be
   treated as the primary scientific run directory.

## Canonical production-results hierarchy

```text
results/
└── <game_name>/
    └── <experiment_name>/
        └── <run_id>/
            ├── manifest.json
            ├── config/
            │   ├── requested_config.yaml
            │   ├── resolved_config.yaml
            │   ├── versions.json
            │   └── prompt_definition.json
            ├── data/
            │   ├── interactions.parquet
            │   ├── provider_attempts.parquet
            │   ├── episodes.parquet
            │   └── trajectory.parquet
            ├── metrics/
            │   ├── online_metrics.parquet
            │   ├── episode_metrics.parquet
            │   └── experiment_summary.json
            ├── plots/
            │   ├── coordination_rate.png
            │   ├── action_share.png
            │   ├── resolution_probability.png
            │   └── final_convention_distribution.png
            ├── logs/
            │   ├── experiment.log
            │   ├── events.jsonl
            │   ├── api_call_status.jsonl
            │   ├── usage_cost.jsonl
            │   └── budget_events.jsonl
            ├── audit/
            │   ├── audit_traces.jsonl
            │   └── prompt_block_traces.jsonl
            ├── checkpoints/
            │   ├── checkpoint_manifest.json
            │   └── episode_shards/
            │       └── <experiment_fingerprint>/
            └── analysis/
```

Checkpoint shards are recoverable intermediate state. The compacted files under
`data/` are the primary scientific datasets. Detailed audit files are created
only when selected by the Phase 7 audit policy.

## Inspection command

```bash
conda run -n MA-CC mas-cc analyze convention-baseline \
  --config configs/runs/naming_convention_baseline_v3.yaml \
  --output-dir inspection/phase_08_0_v3
```

## Inspection artifacts

```text
inspection/phase_08_0_v3/
├── report.md
├── manifest.json
├── resolved_config.yaml
├── interactions.parquet
├── trajectory.parquet
├── episode_metrics.parquet
├── experiment_summary.json
├── coordination_rate.png
├── action_share.png
├── resolution_probability.png
├── final_convention_distribution.png
└── results_layout_validation.json
```

## Manual checks

- Recompute action shares and coordination rates from raw interaction rows and
  confirm exact agreement with the stored trajectory.
- Verify that high pairwise coordination without one dominant action does not
  count as a resolved population-wide convention.
- Confirm the first resolution time uses a complete trailing window.
- Confirm ordinary pair interactions contain two logical decisions and two base
  provider attempts, except where an action is forced.
- Confirm checkpoints can be deleted after successful archival without removing
  the compacted scientific datasets.
- Confirm two executions never write into the same `run_id` directory.

## Acceptance criterion

A base Naming Convention experiment can determine, without Phase 8.1 controls or
information estimators, whether and when a population-wide convention emerged,
which convention emerged, and how reliably this occurred across episodes. Raw
interactions, derived metrics, plots, provider-attempt accounting, and recoverable
checkpoint state are organized under one validated and reproducible run layout.

---

# Phase 8.1 — Control policies and information-theoretic analysis

## Objective

Implement control policies and discrete/InfoNCE information estimators under the
new prompt boundary, independently of a full experiment grid.

## Tasks

1. Define control policies, committees, schedules, interventions, budgets, and observations.
2. Separate membership, observation privileges, policy, forcing, duration, and budget.
3. Implement no-control, always-A/B, incumbent support, alternative promotion,
   and temporary-pulse fixtures.
4. Make control policies expose provider-independent call-plan adjustments.
5. Implement discrete mutual information with plug-in, Jeffreys, optional
   Miller–Madow, episode bootstrap, null permutations, and diagnostics.
6. Implement an InfoNCE pipeline separating extraction, embeddings, pairs,
   objective, estimate, and diagnostics.
7. Validate estimators on independent, perfectly dependent, partially dependent,
   and recorded naming-run fixtures.
8. Control policies operate on game/control observations, not prompt blocks.
9. A control policy cannot mutate a shared full prompt.
10. Forced actions skip prompt binding, compilation, and provider requests unless
   an audit-only prompt is explicitly requested.
11. Any policy that adds information to an agent must do so through the game's
   observation construction, after which the game binds a new prompt normally.
12. Call-plan adjustments count skipped or added prompt instances by stage.
13. Offline analysis may read stored rendered prompts but never calls a game provider.

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
│   ├── interactions.jsonl
│   ├── forced_actions.csv
│   └── control_call_plan_adjustment.json
└── analysis/
    ├── synthetic_dataset.parquet
    ├── discrete_mi_results.csv
    ├── contingency_tables.csv
    ├── infonce_results.csv
    ├── null_results.csv
    └── estimator_diagnostics.md
```

## Manual checks

- Verify exact controlled interactions and skipped prompt/provider work.
- Compare base and adjusted call plans.
- Inspect contingency tables and InfoNCE pairs.
- Confirm expected estimator behavior on synthetic fixtures.
- Confirm analysis never calls the game provider.

## Acceptance criterion

Control and analysis remain independent of prompt implementation details while
forced and additional decisions have auditable effects on prompt/call demand.

---

# Phase 9 — Full experiment orchestration and committee empowerment pilot

## Objective

Compose all validated components into reproducible, priced, approved experiments.

## Tasks

1. Implement experiment/episode specs, strata, grids, repetitions, deterministic
   IDs and seeds, concurrency, checkpoint/resume, compaction, and post-run analysis.
2. Compose game, concrete full prompt, provider, control, execution, logging,
   storage, pricing, budget, and analysis components.
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
8. Invalidate approval when a block class/version/order, response contract,
   prompt binding assumption, or price quote changes.
9. Require explicit live launch and approved preflight ID.
10. Revalidate model availability/pricing immediately before launch.
11. Run episodes concurrently without weakening within-episode causal order or
    atomic budget reservations.
12. Compact only complete episode shards and run analysis after safe storage.
13. Compare estimated and actual calls, block tokens, message tokens, cost,
   runtime, and disk use.
14. Produce final scientific, operational, prompt-provenance, and cost reports.

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
│   ├── pricing_snapshot.json
│   ├── cost_estimate.json
│   ├── runtime_estimate.json
│   ├── budget_status.json
│   └── preflight_id.txt
└── run/
    ├── approved_preflight.json
    ├── events.jsonl
    ├── interactions.parquet
    ├── provider_attempts.parquet
    ├── episodes.parquet
    ├── experiment_summary.csv
    ├── analysis_results/
    └── summary.md
```

## Manual checks

- Inspect grid, prompt definitions/scenarios, prices, units, limits, and assumptions.
- Confirm any prompt definition/value-assumption change invalidates approval.
- Compare estimated and actual calls, tokens, costs, runtime, and disk use.
- Inspect interaction/control traces, checkpoint recovery, Comet/local parity,
  MI tables, and InfoNCE diagnostics.

## Acceptance criterion

A complete experiment can be configured, prompt-resolved, priced, approved,
executed, monitored, resumed, analyzed, and reproduced from saved artifacts.

---

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
- control and storage respect prompt privacy.

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
planning, storage, and observability
  ↓
control and analysis
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
17. Apply controls and validate offline information estimators.
18. Expand to a complete experiment grid.
19. Compare estimated and actual demand and cost.
20. Reproduce the run from saved configuration, versions, and fingerprints.
```

The architecture is accepted only when each step remains independently visible
and no universal game-shaped prompt context is required.
