# CPU Task: Convention Output Formats and One-Call Gemma Decisions

## 1. Objective

Add explicit convention-decision output formats so the existing JSON behavior remains
available while smaller/local models can use simpler action-first responses.

Implement three modes:

| Mode | Model-facing response | Reason | Constrained Gemma probabilities |
|---|---|---:|---:|
| `json_reason` | `{"value":"A","reason":"..."}` | yes | not required; legacy path |
| `choice_reason` | first line `A`, second line `Reason: ...` | yes | yes |
| `choice_only` | exactly `A` | no | yes |

`json_reason` must remain the default so existing OpenAI/University experiments do not
change silently. The two new modes must eliminate the current mismatch in which the
prompt requests JSON but the Gemma runtime scores bare action strings.

For `choice_reason`, the game must receive the selected action, normalized allowed-choice
probabilities, and the textual reason from **one public client call and one returned
response**. The local runtime may perform several internal forward/generation operations
when required for multi-token choices, but the game must not make one client call for the
choice and a second client call for the reason.

For `choice_only`, the model returns/selects only the legal action. No JSON, delimiter,
rationale generation, or second call is permitted.

## 2. Execution boundary for the CPU implementation agent

This task is implemented and tested without loading or downloading Gemma.

The CPU agent must not:

- call any model/tokenizer `from_pretrained()` path;
- run either Gemma live script;
- download a model, tokenizer, snapshot, or configuration;
- modify `.env`, authenticate with Hugging Face, or change CUDA packages;
- claim that token boundaries, combined generation, or probabilities work with the real
  checkpoint; or
- run a live `gemma_local` convention/empowerment experiment.

Use injected fake runtimes and dependency-free tests. Keep PyTorch and Transformers
imports behind the existing real-runtime construction boundary. At the end, update
`tdd/gemma4_gpu_validation_handoff.md` with the exact new live checks and commands.

## 3. Current repository behavior that must be addressed

The relevant implementation is currently split across:

```text
src/naming_game/local_model_types.py
src/naming_game/gemma_local_client.py
src/naming_game/naming_convention_game.py
src/naming_game/empowerment_experiment.py
src/naming_game/interaction.py
configs/empowerment_pilot.yaml
tdd/gemma4_gpu_validation_handoff.md
```

Known facts and problems:

1. `build_convention_messages()` currently requests JSON containing `value` and
   `reason`.
2. The Gemma constrained path scores bare actions against that JSON prompt. At the first
   assistant position the prompt encourages `{`, so those bare-action probabilities are
   not a clean measurement of action preference.
3. The prompt displays a randomized `action_order`, but the current constrained call
   passes `config.actions` and records that fixed order. The exact displayed
   `action_order` must be used for choices, scoring, tie-breaking, and logs.
4. The current code uses `max(config.temperature, 1.0)`, silently changing valid choice
   temperatures such as `0.5` to `1.0`.
5. The normal `ConventionDecision` can hold a reason and constrained scores, but the
   primary empowerment interaction rows currently omit reasons and constrained
   distributions.
6. Reasons are not part of agent memory and must remain absent from future-round prompts.
   Agent memory continues to contain only actions, payoff, and success information.
7. The existing `complete_constrained()` method returns probabilities but no reason.
8. The current live handoff tests the basic Naming Game; it does not exercise a tiny
   `NamingConventionGame`/empowerment run.

Fix these issues as part of this task without changing payoff rules, pairing, committee
interventions, memory semantics, or simultaneous within-pair decisions.

## 4. Public configuration contract

Introduce dependency-free enums or validated literals with stable serialized values:

```python
DecisionOutputFormat = Literal[
    "json_reason",
    "choice_reason",
    "choice_only",
]

ChoiceSelectionPolicy = Literal[
    "argmax",
    "sample",
]
```

Add the necessary fields to `ConventionGameConfig` and
`EmpowermentExperimentConfig`, for example:

```yaml
decision_output_format: json_reason
choice_selection_policy: argmax
choice_temperature: 1.0
```

Semantics:

- `decision_output_format` defaults to `json_reason` for backward compatibility.
- `choice_selection_policy=argmax` chooses the first maximum-probability action in the
  displayed caller order.
- `choice_selection_policy=sample` samples from the normalized legal-choice
  distribution using the request seed.
- `choice_temperature` must be finite and strictly positive and controls normalization
  and optional categorical sampling over legal actions.
- Existing `temperature` continues to control ordinary text generation, including the
  reason continuation.
- Do not clamp any positive configured temperature to `1.0`.
- If legacy `temperature == 0`, constrained scores may still use the explicit positive
  `choice_temperature`; deterministic behavior comes from `argmax`, not a zero softmax
  temperature.
- Unknown mode/policy values fail during configuration loading.

Output format and selection policy can change model behavior and are scientific inputs.
They must appear in:

- the experiment configuration JSON;
- the prompt version/hash;
- experiment fingerprints and episode identity inputs where scientifically appropriate;
- execution/audit metadata; and
- the final summarized configuration.

Resuming an output directory created with a different decision format/policy must not
silently mix shards.

## 5. Exact response formats

### 5.1 `json_reason` — legacy

Preserve the current instruction and parser behavior:

```text
{"value":"A","reason":"A coordinated successfully in recent rounds."}
```

This mode continues through ordinary `LLMClient.complete()` for all providers unless a
future explicitly tested JSON-aware constrained implementation is added. Do not score
bare `A`/`B` candidates against a JSON prompt.

Existing JSON extraction, validation retries, raw responses, and reason parsing must
remain backward compatible.

### 5.2 `choice_reason` — action first, reason second

Use exactly this conceptual shape:

```text
A
Reason: A coordinated successfully in recent rounds.
```

Prompt requirements:

- state that the first line must be exactly one displayed legal action;
- state that no text, whitespace-only preamble, JSON, Markdown fence, bullet, or label
  may precede the action;
- require a newline after the action;
- require the second line to begin with `Reason:`;
- request a short reason;
- continue to describe the same game, visible history, payoffs, and objective; and
- continue to put the decision before the explanation.

Parser requirements for ordinary remote generation:

- ignore leading/trailing blank lines but not arbitrary prose before the action;
- require the first non-blank line, after surrounding whitespace, to equal one legal
  action exactly;
- parse the remaining text only after an exact `Reason:` marker;
- reject a missing/blank reason in this mode according to the existing bounded retry
  policy; and
- never infer an action by substring search in the reason.

For a constrained local response, `selected_choice` is authoritative. A malformed or
missing rationale must never replace or invalidate the action used for the state
transition. Record rationale validity separately and either preserve `reason=None` or
apply a bounded reason-only repair policy without making another game-level decision
call.

### 5.3 `choice_only` — bare legal action

The response is exactly:

```text
A
```

Prompt requirements:

- ask for exactly one displayed legal action;
- forbid JSON, explanations, punctuation, labels, and Markdown; and
- do not ask the model to “think step by step” or provide a reason.

For ordinary remote generation, accept only a stripped exact legal action. For Gemma,
the selected constrained action is authoritative and the public response content should
be the exact action string.

## 6. Prompt construction

Refactor prompt construction so the scientific context is defined once and only the
final response instruction varies by `decision_output_format`. Avoid maintaining three
copies of the payoff/history text.

Suggested internal structure:

```python
build_convention_context(...)
build_convention_response_instruction(output_format, action_order)
build_convention_messages(..., output_format)
```

Every mode must display and pass the same randomized `action_order`. The following must
all agree:

- action order shown in the prompt;
- `choices` passed to a constrained client;
- deterministic tie order;
- `ConventionDecision.action_order`;
- allowed choices stored in logs; and
- reconstructed audit prompts.

Add the output format to the prompt version. A prompt hash for `json_reason` must differ
from `choice_reason` and `choice_only`.

## 7. Gemma client API

Preserve the existing public `complete()` and `complete_constrained()` methods.
`complete_constrained()` remains the no-reason compatibility capability and must keep
supporting one-token and multi-token choices.

Add a typed combined-decision capability rather than overloading an untyped dictionary.
Names may be refined during implementation, but the contract should be equivalent to:

```python
@dataclass(frozen=True)
class ConstrainedDecisionResponse:
    selected_choice: str
    scores: tuple[ChoiceScore, ...]
    content: str
    reason: str | None
    reason_valid: bool | None
    output_format: Literal["choice_reason", "choice_only"]
    model: str
    latency_seconds: float
    usage: TokenUsage
    choice_temperature: float
    selection_policy: Literal["argmax", "sample"]


class ConstrainedDecisionClient(Protocol):
    async def complete_decision(
        self,
        messages: list[dict[str, str]],
        *,
        choices: Sequence[str],
        output_format: Literal["choice_reason", "choice_only"],
        choice_temperature: float = 1.0,
        selection_policy: Literal["argmax", "sample"] = "argmax",
        generation_temperature: float = 0.0,
        max_reason_tokens: int = 32,
        seed: int | None = None,
    ) -> ConstrainedDecisionResponse:
        ...
```

Required behavior:

- one `complete_decision()` call counts as one logical request attempt;
- the client's existing inference semaphore covers the complete score/select/reason
  operation;
- scores preserve displayed caller order;
- `choice_only` performs no rationale generation;
- `choice_reason` returns one action-first content string and one parsed reason;
- `argmax` is deterministic and first-in-order on exact ties;
- `sample` is reproducible for a fixed seed and changes only selection, not returned
  probabilities;
- the selected action is always one of the supplied choices;
- reason tokens are included in token usage only when a reason is generated;
- prompt/scoring work semantics are documented separately from generated-token usage;
- failures update request statistics exactly once; and
- cancellation and `close()` retain existing safe behavior.

Implement `complete_constrained()` as a stable adapter to the `choice_only` decision
path when this does not break its established sequence-scoring semantics. Otherwise
share validation/normalization/selection helpers while retaining both public methods.

The game must use capability checks, not provider-name checks alone. A fake or future
provider implementing `ConstrainedDecisionClient` should work without pretending to be
Gemma. Legacy remote clients remain supported through ordinary generation and the
format-specific parsers.

## 8. Local runtime algorithm to write but not execute on CPU

Add one runtime operation corresponding to the public combined decision. It receives
already mode-aligned messages and semantic choices.

### 8.1 Common scoring and selection

1. Apply the real chat template once.
2. Determine the exact candidate tokenization at the assistant continuation boundary.
3. Obtain finite log-likelihoods for every legal choice.
4. Normalize over only those choices using `choice_temperature`.
5. Select using `argmax` or seeded categorical sampling.
6. Preserve semantic choice strings independently from their rendered/tokenized forms.

For single-token choices such as `A` and `B`, prefer one next-token forward pass and
gather the allowed token logits. For multi-token choices, retain the existing
teacher-forced sequence-scoring fallback. Do not silently use only the first token of a
multi-token action.

The GPU agent must validate whether the exact boundary uses `"A"`, `" A"`, or another
contextual rendering. CPU fakes cannot establish that fact.

### 8.2 `choice_only`

After selection, return exactly the semantic selected choice as `content`; do not call
`model.generate()` for a reason. This is the cheapest path and should be used when the
experiment does not need explanations.

### 8.3 `choice_reason`

After selection, generate a short continuation conditioned on the authoritative choice
and the exact delimiter:

```text
<SELECTED_CHOICE>\nReason: 
```

Return a single synthesized/decoded response in the exact action-first format. The
implementation may either:

- constrain the first generation step to legal actions, capture the unnormalized
  allowed logits before/while masking, and continue generation; or
- score/select first, then continue generation from the prompt plus authoritative
  action/delimiter, reusing cache when safely supported.

Both approaches are one public client call. Prefer correctness and auditability before
KV-cache optimization. Do not reparse the generated explanation to choose a different
action.

Prevent the reason continuation from generating an unbounded response. Respect
`max_reason_tokens` and the model's normal stop tokens. Do not include a second action in
the returned choice distribution.

## 9. Convention-game and empowerment integration

Update `_request_decision()` to branch on explicit output format:

- `json_reason`: ordinary completion plus current JSON parser/retries;
- `choice_reason` with combined capability: one `complete_decision()` call;
- `choice_only` with combined capability: one `complete_decision()` call and no reason;
- either text mode without combined capability: ordinary completion plus strict text
  parser/retries.

Do not use `provider_name == "gemma_local"` as the sole capability decision.

For local constrained modes:

- pass `choices=action_order`, not `config.actions`;
- make the constrained selected action authoritative;
- construct `LLMResponse.content` from the exact returned action-first content;
- keep the complete constrained scores on `ConventionDecision`;
- keep both players' decision calls concurrent as they are now; and
- keep forced/committed actions as zero-call decisions.

Reasons remain observational metadata. Do not add them to agent memory or future
prompts, and do not let rationale text alter payoff or state transitions.

Apply the same principle to the basic Naming Game speaker path so it no longer scores
bare choices against a JSON-only instruction. It may retain its own legacy JSON default,
but any constrained bare-choice call must use a matching bare-choice prompt.

## 10. Persistence and audit contract

The new information is useful only if it survives the empowerment-specific row builder.
Add nullable, backward-compatible fields to primary interaction records for both players:

```text
decision_output_format_i / decision_output_format_j
decision_method_i / decision_method_j
reason_i / reason_j
reason_valid_i / reason_valid_j
allowed_choices_i / allowed_choices_j
choice_log_likelihoods_i / choice_log_likelihoods_j
choice_probabilities_i / choice_probabilities_j
selected_choice_probability_i / selected_choice_probability_j
choice_entropy_i / choice_entropy_j
```

Use a deterministic JSON representation for mappings/lists stored in Parquet scalar
columns if the current schema does not support nested values consistently. Forced and
committed decisions use null probability/reason fields and their existing method flags.

Preserve the existing sampled audit traces. Their reconstructed prompt must use the
actual decision output format and displayed action order. Store:

- raw/synthesized response content;
- authoritative parsed action;
- reason and reason validity;
- selection policy and temperatures;
- constrained scores/probabilities when available; and
- exact output format.

Bump the interaction/prompt schema version when required. Update analysis column
allowlists without making the new nullable columns mandatory for reading legacy runs.

## 11. Tests-first implementation slices

### Slice 1 — configuration and prompt snapshots

Add failing tests for:

- `json_reason` default/backward compatibility;
- parsing/validation of all format and policy values;
- positive finite `choice_temperature`;
- output format included in fingerprints and prompt hashes;
- three prompts sharing identical scientific context;
- JSON instruction appearing only in `json_reason`;
- `Reason:` instruction appearing only in `choice_reason`;
- no reason/thinking/JSON instruction in `choice_only`; and
- displayed `action_order` preserved exactly.

### Slice 2 — strict format parsers

Test:

- valid legacy JSON with reason;
- valid `A\nReason: ...`;
- valid bare `A`;
- rejection of prose before the action;
- rejection of an action mentioned only inside a reason;
- rejection of blank required reasons;
- rejection of JSON in text modes;
- rejection of explanations in `choice_only`; and
- multi-character configured actions without substring ambiguity.

### Slice 3 — combined client contract with fakes

Test:

- choice order, probabilities, log-likelihoods, and token IDs;
- action-first combined content;
- no reason generation in `choice_only`;
- one logical attempt/success per combined call;
- deterministic tie order;
- reproducible seeded sampling;
- different seeds can select different actions under a non-degenerate fake
  distribution;
- choice and reason token usage;
- validation and non-finite failures;
- initialization once under concurrent first calls;
- inference serialization; and
- imports remain free of Transformers/CUDA initialization.

### Slice 4 — game integration

With fake combined and ordinary clients, test all three modes:

- `json_reason` makes the same request and parses the same response as before;
- `choice_reason` makes one logical call per deciding player, returns a reason, and uses
  the constrained action for payoff/state;
- `choice_only` makes one logical call per deciding player and has `reason=None`;
- two player decisions within a pair still overlap;
- displayed randomized `action_order` equals constrained choice order and logged order;
- a reversed order reverses exact-tie selection;
- configured `choice_temperature=0.5` remains `0.5`;
- malformed reason never changes the authoritative constrained action;
- forced/committed players make no model call; and
- remote clients can use strict text modes without constrained probabilities.

### Slice 5 — empowerment persistence

Run tiny fake-backed episodes and assert:

- new fields are present in primary interaction rows/Parquet;
- reason fields are populated only when applicable;
- probability dictionaries preserve displayed order and sum to one;
- selected-choice probability matches the chosen action;
- entropy is finite and correct;
- legacy histories remain readable;
- prompt/config fingerprints separate all modes; and
- audit reconstruction uses the actual mode and pre-interaction memory.

### Slice 6 — regression and live artifacts

Run the focused tests and full suite without the real model. Implement but do not run an
opt-in public-API live diagnostic covering both `choice_only` and `choice_reason`.

Create the smallest practical Gemma empowerment smoke configuration, for example:

```yaml
population_size: 2
names: [A, B]
memory_length: 1
max_population_rounds: 1
committee_sizes: [0]
pulse_rounds: [1]
regimes: [neutral]
replications:
  unit: per_stratum
  count: 1
auto_analyze: false
provider: gemma_local
model: google/gemma-4-12B-it
request_concurrency: 1
episode_concurrency: 1
decision_output_format: choice_only
choice_selection_policy: argmax
choice_temperature: 1.0
```

Fill in every other field required by the actual loader. This config is a live GPU
artifact and must never be executed by the CPU implementation agent.

## 12. Required update to the GPU handoff

Completion of this CPU task requires updating
`tdd/gemma4_gpu_validation_handoff.md`. Do not create a competing handoff file.

The updated handoff must:

1. state that the previous JSON-prompt/bare-choice mismatch has been removed;
2. list all three output modes and identify `json_reason` as legacy/default;
3. record the CPU/fake test results and clearly state that real token boundaries remain
   unvalidated;
4. include commands for focused tests and the full suite before model loading;
5. run the existing known-good logits smoke test or compare its exact environment facts;
6. run the public `choice_only` diagnostic and verify no rationale generation occurs;
7. run the public `choice_reason` diagnostic and verify one action-first response,
   normalized probabilities, and a non-empty reason;
8. verify one logical request statistic for each combined decision;
9. inspect raw token IDs for every displayed action at the real assistant boundary;
10. verify the single-token fast path for `A`/`B` and the multi-token fallback with at
    least one diagnostic multi-token choice;
11. verify argmax ties/order behavior and seeded categorical sampling;
12. verify combined generation does not change the authoritative selected action;
13. verify model/runtime reuse, dtype, device map, CUDA memory, and close behavior;
14. run the tiny `NamingConventionGame`/empowerment Gemma smoke config—not only the
    basic Naming Game;
15. inspect primary Parquet rows for reasons and constrained distributions;
16. rerun the full regression suite after narrow GPU compatibility fixes; and
17. confirm no `.env`, token, cache, tokenizer, snapshot, model weights, or generated
    live results enter Git.

Explicitly ask the GPU agent to make narrow tokenizer/generation compatibility fixes
based on observations. Material changes to output semantics, selection policy, or
scientific fingerprints must be reported rather than silently introduced.

## 13. CPU validation commands

Use the already-configured environment. Do not install or load Gemma.

```bash
pytest tests/test_convention_output_formats.py \
  tests/test_constrained_client_contract.py \
  tests/test_gemma_local_client_unit.py \
  tests/test_gemma_language_game_integration.py \
  tests/test_empowerment_experiment.py

pytest

python -c "import naming_game; import naming_game.gemma_local_client"
python -m naming_game.cli --help
git status --short
```

Test filenames may be split differently if repository conventions make that clearer,
but all requirements above must remain covered. No command in the CPU execution log may
invoke a real model loader.

## 14. Definition of done

- [ ] `json_reason` preserves legacy prompts, parsing, and defaults.
- [ ] `choice_reason` returns action, probabilities, and reason through one client call.
- [ ] `choice_only` returns/selects only the action and performs no reason generation.
- [ ] Every constrained prompt explicitly requests the continuation being scored.
- [ ] Displayed `action_order` is used for scoring, ties, decisions, and logs.
- [ ] Positive choice temperatures are never silently clamped.
- [ ] Argmax and seeded sampling policies are explicit and tested.
- [ ] One-token fast-path and multi-token fallback code are implemented behind lazy
  runtime loading.
- [ ] Reasons remain absent from agent memory and never determine the state transition.
- [ ] Primary empowerment rows retain reasons and constrained distributions.
- [ ] Scientific fingerprints distinguish output modes/policies.
- [ ] Remote clients retain a strict generated-text path for every format.
- [ ] Existing mock/OpenAI/University behavior remains backward compatible.
- [ ] Fake unit/integration tests and the full regression suite pass.
- [ ] Opt-in live API and tiny empowerment artifacts are written but not executed.
- [ ] `tdd/gemma4_gpu_validation_handoff.md` is updated in place.
- [ ] No model/cache/credential artifact appears in Git status.

## 15. Required final report from the CPU agent

Report separately:

- files and public contracts added or changed;
- exact legacy-compatibility behavior retained;
- fake tests and full-suite results;
- confirmation that `choice_reason` is one logical client call;
- confirmation that `choice_only` invokes no rationale generation;
- persistence/schema changes;
- unresolved tokenizer/runtime questions transferred to the GPU handoff;
- path to the updated `tdd/gemma4_gpu_validation_handoff.md`; and
- confirmation that no real Gemma load or download was attempted.

The correct pre-GPU conclusion is:

> The three decision-output formats and combined local-client contract are implemented
> and verified against deterministic fakes; real Gemma tokenization, logits, combined
> generation, and CUDA behavior remain pending GPU validation.
