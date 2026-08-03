# Phase 3–6 Prompt Architecture Realignment Plan

**Status:** implemented and accepted  
**Date:** 2026-08-02  
**Parent plan:** `020826_mas_cc_nine_phase_reorganization_plan_v3.md`  
**Scope:** retroactive realignment of Phases 3, 4, 5, and 6  
**Historical baseline:** existing Version 2 implementations and inspection artifacts remain preserved  
**Required completion boundary:** all gates in this document pass before Phase 7 begins

---

## 1. Objective

Replace the active universal `PromptContext` plus renderer-only `PromptBlock`
design with:

- an abstract immutable value-bearing `PromptBlock[T]`;
- an abstract immutable ordered `FullPrompt`;
- concrete prompt blocks and full prompts owned by each game;
- a compiled prompt artifact that remains the only prompt-layer input to request creation;
- bound prompt scenarios for game planning and token/cost preflight.

Realign the toy and Naming Convention games without redesigning provider
adapters or changing scientific transitions.

---

## 2. Why this realignment is required

The current implementation has a misleading split:

```text
PromptContext = values in universal game-shaped containers
PromptBlock   = renderer that reads those containers
```

Examples of the hidden contracts are:

```python
context.private_state["information"]
context.private_state["presented_actions"]
context.private_state["visible_score"]
context.current_interaction["local_round"]
```

The top-level context is nominally generic, but the real schemas are implicit,
prompt-specific dictionary keys. Some mandatory context fields are unused by
some prompt families. Some current blocks, such as `fixed_game`, combine
multiple independently meaningful prompt pieces.

The intended model is:

```text
PromptBlock = semantic value + validation + rendering + role + version
FullPrompt  = ordered immutable collection of those blocks + response contract
```

---

## 3. Scope and non-goals

### In scope

- Phase 3 prompt records, registry, binding, compilation, token reporting, and inspection;
- Phase 2 prompt component schema amendment required by the new contract;
- Phase 4 request-fixture and preflight integration;
- Phase 5 generic decision and planning contracts;
- Phase 5 toy prompt ownership;
- Phase 6 Naming Convention concrete blocks, runtime, planning, and audit traces;
- tests, CLI inspection paths, configurations, documentation, and notebook migration;
- removal of `PromptContext` from active Phase 5–9 runtime paths.

### Out of scope

- rewriting `src/naming_game`;
- changing provider HTTP transports or response normalization without a failing contract test;
- changing University/OpenAI price semantics;
- changing Naming Convention participant selection, payoffs, memory update, or stopping rules;
- implementing a full HiddenBench game runner as part of this realignment;
- starting Phase 7 storage/Comet work before the realignment gates pass.

---

## 4. Required architectural decisions

These decisions must be recorded before code changes begin.

### Decision A — Block value state

Use an explicit `UNBOUND` sentinel.

```text
UNBOUND       value has not been supplied
None          a supplied null value if the block type allows it
() / [] / {}  supplied empty values
```

Required `UNBOUND` blocks fail validation. Optional `UNBOUND` blocks are omitted
and recorded. Empty bound values render normally.

### Decision B — Immutability

`PromptBlock.bind()` and `FullPrompt.bind()` return new objects. Fixed blocks
cannot be rebound unless an explicit constructor creates a new prompt version or
definition. No runtime mutates shared prompt instances.

### Decision C — Ownership

```text
mas_cc.prompts                 abstract mechanics
mas_cc.games.<game>.prompts    concrete game prompt and block classes
```

A temporary benchmark-only prompt may remain under `prompts/plugins`, but it is
not evidence that a complete game exists.

### Decision D — Ordering

Concrete `FullPrompt.blocks` is the authoritative ordered composition. YAML
selects family/version and presentation options. It does not silently reorder a
scientific prompt. A changed order requires a new prompt version.

### Decision E — Response contract

`ResponseContract` belongs to `FullPrompt`, not to a fake output-format context
field. It produces or validates the decision/output instruction according to the
concrete prompt contract.

### Decision F — Provider boundary

Providers accept only `CompletionRequest`. A provider must never import
`FullPrompt`, a concrete block, a game, or agent state.

### Decision G — Planning boundary

Games expose bound prompt scenarios. The prompt layer compiles and counts them.
Planning combines their demand with provider pricing.

---

## 5. Proposed contracts

### 5.1 PromptBlock

Minimum behavior:

```python
class PromptBlock(ABC, Generic[T]):
    name: str
    version: int
    role: MessageRole
    value: T | Unbound
    required: bool
    binding: Literal["fixed", "dynamic"]

    def bind(self, value: T) -> Self: ...
    def validate(self) -> ValidationResult: ...

    @abstractmethod
    def render(self) -> str: ...
```

Required invariants:

- non-empty stable name;
- positive block version;
- deterministic pure rendering;
- binding type validation;
- fixed-value protection;
- no provider or game execution;
- safe serialization of binding state;
- token count applies to rendered content, not raw value representation.

### 5.2 FullPrompt

Minimum behavior:

```python
class FullPrompt(ABC):
    family: str
    version: int
    blocks: tuple[PromptBlock[Any], ...]
    response_contract: ResponseContract
    message_mode: Literal["per_block", "merge_consecutive_roles"]
    block_separator: str

    def bind(self, **values: object) -> Self: ...
    def validate(self) -> ValidationResult: ...
    def compile(self, token_counter: TokenCounter | None = None) -> CompiledPrompt: ...
```

Required invariants:

- unique block names;
- authoritative stable block order;
- family/version registry identity;
- unknown bind key rejected with a dotted field error;
- all required blocks bound before compilation;
- new immutable object returned from binding;
- deterministic definition and instance hashes.

### 5.3 CompiledPrompt

Required fields:

```text
prompt family/version
definition hash
instance hash
rendered blocks in order
omitted optional blocks
normalized messages
response contract
tokenizer identity
per-block token counts
message/total token estimate
```

### 5.4 Narrow game/runtime protocol

The generic game/runtime layer should depend on a narrow protocol rather than
every concrete prompt class:

```python
class CompilablePrompt(Protocol):
    family: str
    version: int

    def compile(self, token_counter: TokenCounter | None = None) -> CompiledPrompt: ...
```

---

## 6. Current-to-target mapping

| Current object/path | Target action |
|---|---|
| `prompts/context.py::PromptContext` | Deprecate, migrate all active use, then remove/export only in compatibility module if required |
| renderer-only `PromptBlock` | Replace with abstract value-bearing generic block |
| `PromptDefinition` | Replace with concrete `FullPrompt` class/factory registration |
| `PromptComposer.compose(config, context)` | Replace with `full_prompt.bind(...).compile(counter)` |
| `PromptInstance` | Rename/evolve into `CompiledPrompt` |
| YAML `blocks:` ordering | Move authoritative order to concrete prompt; export resolved manifest |
| `PromptContextScenario` | Replace with `PromptScenario(bound_prompt=...)` |
| `DecisionRequest.prompt_context` | Replace with `DecisionRequest.prompt` using `CompilablePrompt` |
| `plugins/ashery_2025.py::fixed_game` | Split into concrete naming blocks |
| HiddenBench notebook manual context | Rewrite to construct and bind concrete HiddenBench full prompts |

---

## 7. Implementation sequence and gates

No later workstream starts until the preceding gate passes, except for
documentation or test-fixture preparation that does not modify shared contracts.

### Workstream 0 — Freeze and inventory

#### Tasks

1. Record the current Git revision and dirty-worktree inventory.
2. Run all existing tests.
3. Run Phase 3–6 inspection commands and preserve artifacts.
4. Record exact current naming compiled-message fixtures for:
   - empty memory;
   - one memory item;
   - maximum configured memory;
   - both presented-action orders.
5. Record current toy compiled messages and planning estimates.
6. Inventory every `PromptContext`, `PromptDefinition`, `PromptComposer`, and
   `PromptContextScenario` reference using `rg`.

#### Artifacts

```text
inspection/realignment_v3/baseline/
├── report.md
├── manifest.json
├── git_status.txt
├── test_summary.json
├── prompt_api_references.txt
├── naming_prompt_fixtures.json
├── toy_prompt_fixtures.json
└── artifact_hashes.json
```

#### Gate 0

Baseline tests pass or every pre-existing failure is documented. Historical
artifacts are hashed and will not be overwritten.

---

### Workstream 1 — Add the Version 3 prompt kernel side by side

#### Files expected to change or be added

```text
src/mas_cc/prompts/
├── blocks.py
├── full_prompt.py
├── compiled.py
├── contracts.py
├── registry.py
├── tokenization.py
├── fingerprints.py
└── compatibility.py       # temporary only if needed
```

#### Tasks

1. Implement `UNBOUND`.
2. Implement abstract immutable `PromptBlock[T]`.
3. Implement abstract immutable `FullPrompt`.
4. Implement `CompiledPrompt` and rendered-block records.
5. Adapt tokenizer protocol and regex estimator.
6. Implement canonical deterministic fingerprints.
7. Implement registry factory lookup.
8. Retain response validation without game knowledge.
9. Add a basic-choice concrete example.
10. Add optional compatibility adapters without routing new code through them.

#### Required unit tests

- fixed block renders and cannot be rebound;
- dynamic block binds immutably;
- required `UNBOUND` fails;
- optional `UNBOUND` omits with a record;
- empty tuple renders rather than omits;
- invalid value type produces a field-specific issue;
- block order is stable;
- per-block and total token counts reconcile;
- message grouping is deterministic;
- definition hash changes for definition changes;
- instance hash changes for value changes;
- two concurrent binds share no mutable values;
- import performs no I/O.

#### Gate 1

The new Phase 3 kernel passes its unit/inspection tests without importing a game
or provider. Existing runtime still passes through the temporary old path.

---

### Workstream 2 — Prompt configuration Version 2

#### Tasks

1. Define prompt component schema Version 2.
2. Make family/version the main selection fields.
3. Define permitted message-policy options.
4. Remove hand-maintained required block ordering from new YAML.
5. Export resolved authoritative block order from the registered concrete prompt.
6. Implement clear Version 1 migration diagnostics.
7. Update secret checks for serialized block defaults and options.

#### Example

```yaml
schema_version: 2
prompt_family: naming_convention_decision
prompt_version: 1
message_mode: merge_consecutive_roles
response_contract:
  type: paper_choice_reason
  allowed_values: [Q, M]
```

#### Gate 2

Both configuration loading and resolved export are deterministic; new artifacts
contain the resolved block manifest and no dynamic private values.

---

### Workstream 3 — Phase 4 request and economics adaptation

#### Tasks

1. Create normalized requests from `CompiledPrompt.messages`.
2. Carry prompt family/version and fingerprints only in local request metadata.
3. Confirm `wire_messages()` excludes audit metadata.
4. Adapt provider smoke fixtures to a bound basic-choice full prompt.
5. Adapt static preflight to use the compiled request/token estimate.
6. Preserve all pricing, accounting-unit, limit, and runtime-budget logic.
7. Prove adapters import no prompt/game implementation modules.
8. Rerun mock, OpenAI-compatible, University discovery, Gemma lazy-load,
   pricing-source, cached-token, long-context, and budget-concurrency tests.

#### Gate 3

For identical normalized messages, Version 2 and Version 3 provider/preflight
results differ only in prompt provenance fields. No adapter redesign occurred.

---

### Workstream 4 — Phase 5 generic game contract and toy migration

#### Tasks

1. Define `CompilablePrompt` protocol.
2. Replace `DecisionRequest.prompt_context` with `DecisionRequest.prompt`.
3. Replace `PromptContextScenario` with bound `PromptScenario`.
4. Implement toy-specific concrete blocks.
5. Implement `ToyCoordinationFullPrompt`.
6. Bind a new prompt from each private observation.
7. Compile in the generic runtime, not in pure transition code.
8. Store definition/instance hashes on attempts.
9. Create lower/representative/maximum bound prompt scenarios.
10. Update game inspection and planning artifacts.

#### Required contract test

```text
state
→ private observation
→ game-owned bound prompt
→ independently rendered blocks
→ compiled normalized messages
→ mock response
→ validated action
→ pure transition
```

#### Gate 4

Toy trajectories and call demand remain deterministic. Changing providers does
not change the bound prompt, compiled messages, or game call plan.

---

### Workstream 5 — Phase 6 Naming Convention migration

#### Target concrete classes

```text
DescriptionBlock
RulesBlock
PresentedActionsBlock
VisibleMemoryBlock
VisibleScoreBlock
NamingConventionFullPrompt
```

#### Tasks

1. Add `games/naming_convention/prompts.py`.
2. Implement `DescriptionBlock` as fixed and required.
3. Implement `RulesBlock` as fixed and required.
4. Implement `PresentedActionsBlock` as dynamic and required.
5. Implement `VisibleMemoryBlock` as dynamic and required, accepting empty memory.
6. Implement `VisibleScoreBlock` as dynamic and required.
7. Put response formatting/validation on `NamingConventionFullPrompt.response_contract`.
8. Bind one prompt for each selected agent from the same immutable pre-transition state.
9. Preserve independently seeded presented-action ordering.
10. Preserve bounded private memory and anonymous local roles.
11. Update decision records to contain the bound prompt protocol and prompt hashes.
12. Update runtime compilation and retries.
13. Update empty/representative/maximum memory planning scenarios.
14. Update selected audit traces with block values/renderings/token counts.
15. Compare exact rendered messages with the frozen baseline.
16. If text must change because previously combined blocks are separated, create
    a new prompt version and document the scientific compatibility decision.
17. Remove active use of `plugins/ashery_2025.py` renderer-only definitions.

#### Naming invariants that must not change

- pair selection remains uniform over two distinct agents;
- both pair decisions are simultaneous;
- both prompts use the same pre-interaction state;
- action ordering remains seeded and per-agent;
- visible memory remains bounded;
- global population state and identities remain hidden;
- payoff remains +100/-50 under the configured paper profile;
- invalid-response retry accounting remains explicit;
- no provider logic enters the game.

#### Gate 5

All naming invariants pass. The five blocks are independently inspectable. Mock
reruns are deterministic. Provider selection does not affect prompt construction
or game transitions except through returned model content.

---

### Workstream 6 — Remove active legacy prompt paths

#### Tasks

1. Migrate CLI prompt inspection.
2. Migrate provider smoke CLI.
3. Migrate game preflight.
4. Migrate prompt reporting/Markdown logging.
5. Migrate HiddenBench tutorial notebook to concrete full prompts.
6. Update examples and tests.
7. Remove `PromptContext` exports from the primary API.
8. Remove renderer-only `PromptDefinition` from the primary API.
9. Move any unavoidable transition adapter to `prompts.compatibility`.
10. Add a guard test that active `games`, `planning`, and `runtime` modules do not
    import `PromptContext`.
11. Run `rg` and save remaining references with a justification for each.

#### Gate 6

No active Phase 5–9 code uses `PromptContext`. Compatibility code, if retained,
is isolated, documented, unregistered by default, and scheduled for deletion.

---

### Workstream 7 — Required new-game FullPrompt tutorial notebook

#### Deliverable

```text
notebooks/tutorial_create_full_prompt_new_game.ipynb
```

#### Tutorial scenario

Use a small hypothetical private-signal choice game that is not implemented by
an existing helper. The notebook itself defines:

```text
PrivateSignalChoiceFullPrompt
├── DescriptionBlock
├── RulesBlock
├── AvailableActionsBlock
├── PrivateSignalBlock
├── VisibleMemoryBlock
└── OptionalHintBlock
```

This is a prompt tutorial, not a new production game implementation. A final
section explains how these classes would move into
`games/private_signal_choice/prompts.py` and how a future game observation would
bind them.

#### Required notebook sections

1. Architecture map from abstract block to provider request.
2. Imports and links to every production class used.
3. Definition of each concrete block subclass in full notebook code.
4. Per-block value validation and rendering.
5. Definition of `PrivateSignalChoiceFullPrompt` and authoritative order.
6. Fixed values bound at construction.
7. Dynamic required values initially set to `UNBOUND`.
8. Optional `UNBOUND` hint omission.
9. Empty visible memory rendering, contrasted with unbound memory failure.
10. Immutable binding for two different agents and a non-leakage assertion.
11. Independent block inspection.
12. Compiled system/user messages.
13. Per-block and total token estimates.
14. Definition and instance hashes, including one-value-change demonstrations.
15. Response contract and local validation examples.
16. Normalized `CompletionRequest` construction.
17. Secret-safe University and OpenAI provider configuration display.
18. University live availability/pricing preflight.
19. OpenAI auditable pricing preflight.
20. Separate University completion section.
21. Separate OpenAI completion section.
22. Normalized usage and actual-cost reporting for each provider.
23. Promotion-to-production checklist for a real new game.

#### Interactive controls

The committed notebook must visibly default to:

```python
USE_LIVE_UNIVERSITY_PRICING = True
CALL_UNIVERSITY = True
CALL_OPENAI = True
```

The two completion cells remain separate so the user can run University and
OpenAI independently. Both receive the exact same compiled wire messages. A
University connection/pricing/completion failure is caught and reported safely
and does not prevent the OpenAI section from executing.

#### Credential contract

```dotenv
POTSDAM_API_KEY=replace-with-real-key
BASE_POTSDAM_LLM_URL=replace-with-proxy-base-url
OPENAI_API_KEY=replace-with-real-key
POTSDAM_MODEL=optional-model-override
OPENAI_MODEL=optional-model-override
```

Only variable names and configured/not-configured booleans may be displayed.
Keys, resolved private URLs, headers, account identities, and raw unredacted
responses must never be stored in the notebook.

#### Automated validation

Automated validation overrides live controls in memory without editing the
committed notebook, then verifies:

- `nbformat` validity;
- every code cell compiles, including top-level `await`;
- the complete non-network notebook path executes;
- blocks bind/render as documented;
- two agent prompts remain isolated;
- block token totals reconcile;
- fingerprints are deterministic;
- both provider sections use identical request messages;
- saved notebook content passes a secret scan.

Optional manual validation executes one University call and one OpenAI call and
records only sanitized summaries under:

```text
inspection/realignment_v3/tutorial_live/
├── report.md
├── manifest.json
├── university_summary.json
├── openai_summary.json
├── preflight_comparison.json
└── secret_scan.json
```

#### Gate 7

The notebook teaches construction without calling a ready-made game prompt,
passes the automated non-network path, and its two live sections are structurally
ready for independent University and OpenAI execution.

---

### Workstream 8 — Reinspection and adoption

#### Commands

```bash
conda run -n MA-CC mas-cc inspect phase 3 \
  --output-dir inspection/phase_03_v3

conda run -n MA-CC mas-cc inspect phase 4 \
  --output-dir inspection/phase_04_v3

conda run -n MA-CC mas-cc game run \
  --config configs/runs/toy_game_smoke_test_v3.yaml \
  --output-dir inspection/phase_05_v3

conda run -n MA-CC mas-cc game run \
  --config configs/runs/naming_convention_smoke_test_v3.yaml \
  --output-dir inspection/phase_06_v3
```

#### Gate 8

- all Phase 1–6 tests pass;
- all four Version 3 inspection bundles pass schema and secret scans;
- historical artifact hashes remain unchanged;
- provider-economics results remain correct;
- the Version 3 parent plan is marked adopted;
- the required tutorial notebook passes Gate 7;
- Phase 7 is unblocked.

---

## 8. Phase-specific artifact requirements

### Phase 3

```text
full_prompt_definition.json
unbound_prompt.json
bound_prompt.json
block_manifest.json
rendered_blocks.json
omitted_blocks.json
compiled_messages.json
token_breakdown.csv
fingerprints.json
validation_examples.md
```

### Phase 4

```text
compiled_prompt.json
request.json
normalized_response.json
usage.json
pricing_snapshot.json
preflight_estimate.json
budget_status.json
provider_boundary_diff.md
```

### Phase 5

```text
observations.jsonl
bound_prompts.jsonl
compiled_prompts.jsonl
interactions.jsonl
game_call_plan.json
prompt_scenarios.json
```

### Phase 6

```text
full_prompt_definition.json
selected_block_traces.jsonl
selected_audit_traces.jsonl
prompt_token_scenarios.csv
prompt_parity_report.md
interactions.jsonl
trajectory.csv
```

---

## 9. Test matrix

| Area | Required evidence |
|---|---|
| Block semantics | fixed/dynamic/required/optional/unbound/empty tests |
| Immutability | concurrent bind isolation and deep immutability |
| Rendering | golden block and compiled-message fixtures |
| Token estimation | block totals reconcile with compiled estimate |
| Fingerprints | stable canonical hashes and expected invalidation |
| Configuration | schema 2 validation and schema 1 migration diagnostics |
| Provider boundary | identical wire requests; no prompt/game imports |
| Pricing | cached, long-context, unit, freshness, unknown, budget tests |
| Toy game | deterministic full trace through bound prompt |
| Naming game | information boundary, simultaneous decisions, memory, action order, retry, transition |
| Planning | bound lower/representative/maximum prompt scenarios |
| Inspection | files, schemas, hashes, secret scan, readable report |
| Legacy | unchanged `naming_game` and historical artifact hashes |

---

## 10. Migration risks and mitigations

### Risk: mutable values leak between concurrent agents

Mitigation: frozen blocks/full prompts, deep freezing, bind-return-new semantics,
and explicit concurrent pair tests.

### Risk: separating blocks changes paper-faithful whitespace or wording

Mitigation: freeze exact baseline wire messages, define separators explicitly,
and either prove parity or bump the prompt version with a documented reason.

### Risk: YAML order and class order diverge

Mitigation: concrete full prompt is authoritative; resolved configuration exports
but does not independently redefine the ordered manifest.

### Risk: optional omission is confused with empty memory

Mitigation: explicit `UNBOUND` sentinel and dedicated empty-value tests.

### Risk: provider code becomes coupled to prompts during convenience refactors

Mitigation: import-boundary tests and normalized `CompletionRequest` as the only adapter input.

### Risk: private block values leak into remote observability

Mitigation: sensitivity metadata, local-only detailed traces, redacted public
manifests, and no full block values in Comet by default.

### Risk: preflight approval survives a prompt definition change

Mitigation: bind approvals to prompt definition hashes, block versions/order,
scenario hashes, provider quote, and resolved configuration.

### Risk: compatibility layer becomes permanent

Mitigation: isolate it, keep it unregistered by default, list every remaining
reference in Gate 6, and prohibit it from Phase 5–9 active paths.

---

## 11. Review checklist before implementation

- [x] Confirm the `UNBOUND`/empty/null semantics.
- [x] Confirm concrete full prompt owns authoritative order.
- [x] Confirm response contract is a `FullPrompt` concern, not a normal game-value block.
- [x] Confirm the five Naming Convention blocks and their roles.
- [x] Preserve exact old wire text while changing only local prompt provenance.
- [x] Confirm prompt schema Version 2 migration policy.
- [x] Confirm concrete prompts live in game packages.
- [x] Confirm no full HiddenBench game is implied by the prompt/notebook fixture.
- [x] Confirm provider adapters remain untouched unless contract tests fail.
- [x] Confirm the required tutorial defines a genuinely new concrete FullPrompt and blocks.
- [x] Confirm both live provider controls default to true in separate sections.
- [x] Confirm Phase 7 remained paused until Gate 8 passed.

---

## 12. Definition of realignment completion

The realignment is complete only when:

1. abstract prompt machinery contains no game vocabulary;
2. concrete games own their full prompts and blocks;
3. Naming Convention exposes the five agreed semantic blocks;
4. every agent decision receives a separately bound immutable prompt;
5. block rendering, message composition, token estimation, and fingerprints are inspectable;
6. generic runtime and providers consume only narrow protocols/normalized messages;
7. planning uses bound prompt scenarios;
8. Phase 4 pricing and budget behavior remains correct;
9. `PromptContext` is absent from active Phase 5–9 paths;
10. Version 3 Phase 3–6 inspection bundles and regression tests pass;
11. historical artifacts and `src/naming_game` remain unchanged;
12. the new-game tutorial runs through the non-network path and is ready for
    independent University and OpenAI live calls;
13. Phase 7 can begin without another prompt-boundary redesign.
