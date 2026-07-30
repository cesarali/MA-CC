# Plan: Internal Gemma 4 API for Language Games and Constrained Choices

## 1. Outcome

Build a repository-internal Python API that loads `google/gemma-4-12B-it` once
from the existing `HF_HOME`, serves the same stateless asynchronous completion
contract already used by the language games, and additionally exposes normalized
scores for an explicit set of allowed answers.

The first end-to-end milestone is deliberately small:

1. load the cached Gemma checkpoint once on the A100;
2. answer a simple reasoning question through the internal API;
3. request allowed answers `A`, `B`, and `C` through that same API;
4. verify that callers can inspect raw sequence log-likelihoods and normalized
   constrained probabilities;
5. run a small existing Naming Game with `provider=gemma_local` without changing
   the game engine's provider-independent behavior.

This is an internal Python API in phase 1, not a network service. Keeping the
model in the same process avoids authentication, serialization, and deployment
work while the contract is still being validated. A local HTTP service can be
added later behind the same client interface if multiple processes must share one
GPU-resident model.

## 2. Existing repository boundary to preserve

The games already depend on `LLMClient` in `src/naming_game/api_client.py`:

```python
async def complete(
    messages: list[dict[str, str]],
    *,
    temperature: float,
    max_tokens: int,
    seed: int | None = None,
) -> LLMResponse:
    ...
```

`SequentialNamingGame`, `SynchronousParallelNamingGame`, the reasoning game,
the convention game, the benchmark runner, and the empowerment experiment all
receive a client rather than constructing a provider directly. Provider creation
is concentrated in `src/naming_game/cli.py`. The Gemma implementation should use
these boundaries instead of introducing a separate game framework.

The successful smoke test in `scripts/gemma4_logits_test/` establishes the
working implementation details:

- Python 3.11.15;
- PyTorch 2.13.0 with CUDA 12.9;
- torchvision 0.28.0 from the matching conda-forge CUDA build;
- Transformers 5.14.1 and accelerate 1.14.0;
- `AutoProcessor` and `AutoModelForMultimodalLM`;
- BF16 and `device_map="auto"` on the A100;
- text-only chat templates with `enable_thinking=False`;
- a 262,144-token vocabulary;
- approximately 22.3 GiB allocated for model weights and 22.4 GiB peak for the
  smoke-test prompt;
- correct single-token and teacher-forced sequence scoring;
- checkpoint storage under `/work/ojedamarin/hf_models` through the existing
  ignored `.env`, never through a repository-local cache.

## 3. Proposed API contract

### 3.1 Preserve ordinary completions

Add `GemmaLocalAsyncLLMClient`, implementing the existing `LLMClient` protocol.
Its `complete()` method returns the existing `LLMResponse`, so current games can
use Gemma without knowing it is local.

Required behavior:

- `model == "google/gemma-4-12B-it"` by default;
- `provider_name == "gemma_local"`;
- stateless requests: no agent histories or conversation state in the client;
- model and processor loaded exactly once per client/service lifetime;
- deterministic generation when `temperature == 0`;
- seeded sampling when sampling is requested and supported;
- decode only newly generated tokens;
- populate prompt, completion, and total token usage;
- reuse the repository's request statistics and safe error conventions;
- `close()` releases references and optionally clears GPU resources only at
  process shutdown, never between requests.

### 3.2 Add constrained-choice scoring explicitly

Do not overload `complete()` with an untyped response dictionary. Introduce a
second capability protocol and typed response objects:

```python
@dataclass(frozen=True)
class ChoiceScore:
    choice: str
    token_ids: tuple[int, ...]
    log_likelihood: float
    probability: float


@dataclass(frozen=True)
class ConstrainedLLMResponse:
    selected_choice: str
    scores: tuple[ChoiceScore, ...]
    model: str
    latency_seconds: float
    usage: TokenUsage
    temperature: float


class ConstrainedLLMClient(Protocol):
    async def complete_constrained(
        self,
        messages: list[dict[str, str]],
        *,
        choices: Sequence[str],
        temperature: float = 1.0,
        seed: int | None = None,
    ) -> ConstrainedLLMResponse:
        ...
```

The returned `scores` must preserve the caller's choice order. Probabilities are
normalized only over the supplied choices and sum to one. `selected_choice` is
the highest-probability choice for deterministic use; a later optional sampling
policy may sample from this constrained distribution using `seed`.

Expose sequence log-likelihoods as the stable public primitive. Single-token
next-logit scores are a diagnostic optimization and may also be returned in a
separate optional field later, but they must not be the only implementation
because valid answers can contain multiple tokens.

### 3.3 Validation rules

Both API methods must reject:

- empty or malformed messages;
- empty, blank, or duplicate choices;
- non-finite or non-positive constrained temperature;
- unavailable CUDA unless CPU use was explicitly enabled;
- non-finite logits, log-probabilities, or normalized probabilities;
- inconsistent tokenizer/model vocabulary sizes;
- unsupported model IDs or Transformers versions.

The API must never print tokens or copy model weights into the repository.

## 4. Proposed source layout

```text
src/naming_game/
├── api_client.py                 # existing protocols/remote/mock clients
├── local_model_types.py          # ChoiceScore and ConstrainedLLMResponse
└── gemma_local_client.py         # GemmaLocalAsyncLLMClient

scripts/gemma4_api_test/
├── README.md
└── test_internal_api.py          # live A100 reasoning/API smoke test

tests/
├── test_gemma_local_client_unit.py
├── test_constrained_client_contract.py
└── test_gemma_language_game_integration.py
```

Keep heavyweight imports (`torch`, `transformers`, `torchvision`) inside
`gemma_local_client.py` and preferably behind initialization. Importing
`naming_game`, running mock tests, or using a remote provider must not initialize
CUDA or require the Gemma optional dependencies.

## 5. Gemma runtime design

### 5.1 Initialization

Extract reusable, tested helpers from the smoke-test script rather than importing
the script itself:

1. find and load the repository-root `.env` before Hugging Face imports;
2. validate and create `HF_HOME`;
3. check disk space and CUDA visibility;
4. report the model ID, GPU, dtype, device map, and memory once at startup;
5. instantiate the processor and model once;
6. call `model.eval()`;
7. retain a single runtime object for all requests.

Use dependency injection for unit tests: permit a fake processor/model runtime to
be passed to the client without touching CUDA or Hugging Face.

### 5.2 Message formatting

Use the checkpoint's chat template for all calls:

```python
processor.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True,
    enable_thinking=False,
)
```

Retry without `enable_thinking` only for a documented compatibility exception.
Keep the request text-only. Move tensors to the model's input device derived from
`hf_device_map`, not from a hard-coded `cuda:0` assumption.

### 5.3 Ordinary generation

Run `model.generate()` inside `torch.inference_mode()`:

- `temperature == 0`: `do_sample=False`;
- `temperature > 0`: `do_sample=True`, pass temperature, and use a local seeded
  `torch.Generator` where Transformers supports it;
- set `max_new_tokens=max_tokens`;
- decode only tokens after the prompt length;
- return the raw decoded content and exact token counts.

Do not reload or quantize the model per request.

### 5.4 Constrained sequence scoring

For every choice:

1. tokenize the complete candidate without special tokens;
2. append those candidate IDs to the already templated prompt IDs;
3. teacher-force one forward pass;
4. take logits from `prompt_length - 1` through the token before the final
   candidate token;
5. compute `log_softmax(..., dtype=float32)`;
6. gather only the candidate token log-probabilities;
7. sum them to obtain `log p(choice | prompt)`;
8. normalize all candidate scores with
   `softmax(sequence_log_likelihoods / temperature)` in float32.

Assert finite values and an approximately unit probability sum. Preserve the
token IDs for auditability. Do not expose full 262,144-element logit tensors by
default; they are large and callers need choice scores. A debug-only method may
return top-k token records or CPU logits later if a research use case requires
them.

### 5.5 Concurrency and GPU safety

The current games can issue concurrent requests, but one 40 GB A100 has limited
headroom above a 22.3 GiB model. Begin with an `asyncio.Semaphore(1)` around GPU
inference and move blocking model work into `asyncio.to_thread()` so the event
loop remains responsive. Advertise `concurrency=1` for this provider.

Do not launch simultaneous forward passes during phase 1. Record queue time and
inference time separately if performance analysis needs them. Dynamic batching
is a later optimization only after correctness and peak-memory tests pass.

## 6. Language-game integration

### 6.1 Provider selection

Extend the centralized provider factory in `src/naming_game/cli.py`:

```text
provider: gemma_local
model: google/gemma-4-12B-it
request_concurrency: 1
```

Add `gemma_local` to configuration validation and exports in
`src/naming_game/__init__.py`. Remote and mock provider behavior must remain
unchanged. The model should be created once per experiment process and shared by
all agents, consistent with the repository's stateless-client design.

### 6.2 First compatibility milestone

Run the existing game unchanged through `GemmaLocalAsyncLLMClient.complete()`.
This proves that prompts, JSON validation/repair, interaction accounting, logs,
and summaries work with a local provider.

Use a tiny deterministic run first:

- 2 agents;
- sequential update mode;
- 2 interactions;
- no reasoning interactions;
- temperature 0;
- provider concurrency 1;
- fixed seed.

Assert that the run completes, makes four successful model calls, writes normal
artifacts, records `api_backend/provider == gemma_local`, and never initializes
more than one model.

### 6.3 Constrained game decisions

After compatibility is proven, use `complete_constrained()` at decisions whose
legal action set is already known:

- binary speaker selection: the speaker's current inventory (`A`, `B`, or both);
- convention game action: `config.actions` in the randomized display order;
- other explicitly enumerated experimental actions.

Do not infer allowed choices by parsing prompt text. The game engine must pass the
legal choices explicitly. Store the complete constrained distribution in the
interaction log so later analyses can use uncertainty, entropy, margins, or
counterfactual choice probabilities.

For responses that also require a reason, use a two-stage design:

1. score/select the legal action with `complete_constrained()`;
2. optionally generate a short rationale conditioned on that fixed action.

The action used to update the game must come from the constrained call, never
from reparsing the rationale. Basic Naming Game interactions can skip rationale
generation entirely.

Add optional log fields rather than changing existing required columns:

```text
decision_method: generated | constrained_sequence
allowed_choices: [A, B]
choice_log_likelihoods: {A: ..., B: ...}
choice_probabilities: {A: ..., B: ...}
selected_choice_probability: ...
choice_entropy: ...
```

## 7. Test strategy

### 7.1 Fast unit tests without the real model

Use fake tokenizer/model objects with tiny vocabularies to test:

- prompt tokens are excluded from sequence scores;
- one-token and multi-token choices are handled correctly;
- candidate token alignment is correct;
- choice order is preserved;
- float32 normalization sums to one;
- duplicate/blank choices and invalid temperatures fail clearly;
- non-finite logits fail;
- deterministic argmax selection works;
- the async semaphore limits active inference to one;
- request statistics and token usage are correct;
- construction and multiple calls load the fake model only once;
- `close()` is idempotent;
- ordinary `complete()` still satisfies `LLMClient`.

Add a contract test that both `MockAsyncLLMClient` and the fake-backed Gemma
client satisfy the ordinary completion behavior. Add constrained behavior to a
mock constrained client so game integration tests do not need the A100.

### 7.2 Live internal-API reasoning test

Create `scripts/gemma4_api_test/test_internal_api.py` with an explicit live-test
marker or CLI entry. Use this prompt:

```text
Question: Which number is larger, 7 or 3?

A. 7
B. 3
C. They are equal

Return only A, B, or C.
```

The script must call the public client API, not private runtime helpers:

```python
response = await client.complete_constrained(
    messages,
    choices=["A", "B", "C"],
    temperature=1.0,
)
```

Acceptance checks:

- model is `google/gemma-4-12B-it`;
- provider is `gemma_local`;
- selected choice is `A`;
- all three choices and token IDs are returned;
- all log-likelihoods and probabilities are finite;
- probabilities sum to one within `1e-5`;
- `A` has the greatest probability;
- an ordinary `complete()` call also returns a non-empty response;
- startup and post-request memory diagnostics are printed;
- a second request reuses the existing model instance;
- no cache or model file appears in Git status.

This is an infrastructure assertion, not a benchmark of Gemma's reasoning
quality. Only numerical correctness and API behavior should be generalized from
it.

### 7.3 Game integration tests

With a fake constrained client:

- run a two-agent sequential game;
- verify allowed choices exactly match each speaker inventory;
- verify the selected constrained choice drives the state update;
- verify probabilities are logged;
- test an inventory containing only one legal action;
- run the synchronous game and confirm provider concurrency remains bounded;
- prove seeded runs are reproducible;
- prove existing mock/remote tests remain unchanged.

Then add one opt-in live A100 test for the two-interaction Gemma game. Never run
the 24 GB live test as part of ordinary CI.

## 8. Configuration and CLI work

Add only the options needed for the local provider:

```text
--provider gemma_local
--model google/gemma-4-12B-it
--local-dtype bfloat16
--local-device-map auto
--decision-mode generated|constrained
```

Recommended defaults for `gemma_local`:

```text
request_concurrency: 1
local_dtype: bfloat16
local_device_map: auto
decision_mode: constrained
enable_thinking: false
```

Validate incompatible combinations early. For example, reject local concurrency
above the tested limit unless an explicit unsafe/experimental batching option is
later introduced. Keep `HF_HOME` exclusively in `.env`; do not place cache paths
or tokens in YAML configuration.

## 9. Implementation phases

### Phase 1 — Typed constrained API

- Add response dataclasses and the `ConstrainedLLMClient` protocol.
- Extract pure token-scoring helpers from the smoke test.
- Cover them with tiny fake-model unit tests.

Exit criterion: fast tests prove correct one-token and multi-token probabilities.

### Phase 2 — Local Gemma client

- Implement lazy heavyweight imports and one-time runtime initialization.
- Implement `complete()` and `complete_constrained()`.
- Add serialization, usage, statistics, error handling, and memory diagnostics.
- Run the live reasoning smoke test on the A100.

Exit criterion: `A` wins the reasoning prompt; both API methods work through the
public client; the model is loaded once.

### Phase 3 — Provider and game compatibility

- Add `gemma_local` to CLI/config provider selection.
- Run the existing basic game through ordinary `complete()` first.
- Add fake-backed integration tests and one opt-in live two-interaction run.

Exit criterion: normal game artifacts are produced with the local provider and
existing remote/mock tests still pass.

### Phase 4 — Constrained decisions in games

- Pass legal actions explicitly from game state.
- Use constrained sequence scoring for speaker/convention decisions.
- Persist distributions and derived entropy/margin fields.
- Separate optional rationale generation from authoritative action selection.

Exit criterion: a game run exposes auditable per-action probabilities and uses
the constrained selection for state transitions.

### Phase 5 — Performance and optional service boundary

- Profile queue, tokenize, forward, scoring, and generation times.
- Measure peak memory for realistic prompts and choice counts.
- Consider batched candidate scoring and request batching.
- If multiple processes need the same model, wrap the runtime in a small local
  HTTP service and implement an `LLMClient` adapter without changing game code.

Exit criterion: only pursue this phase when correctness tests are stable and a
measured workload justifies the added complexity.

## 10. Acceptance checklist

- [ ] `.env` is loaded before Hugging Face initialization.
- [ ] Model files exist only below `HF_HOME`.
- [ ] Gemma model and processor load exactly once per process/client lifetime.
- [ ] Existing `LLMClient.complete()` remains backward compatible.
- [ ] `complete_constrained()` supports one-token and multi-token choices.
- [ ] Prompt tokens are excluded from sequence likelihoods.
- [ ] Returned probabilities are finite, ordered, and sum to one.
- [ ] The reasoning API test selects `A` for 7 versus 3.
- [ ] Ordinary generation works through the same public client.
- [ ] Local GPU inference is serialized initially.
- [ ] A tiny existing Naming Game completes with `provider=gemma_local`.
- [ ] Constrained legal-action probabilities are accessible to callers and logs.
- [ ] The action used by the game comes from the constrained call.
- [ ] Existing mock, University, and OpenAI clients remain unaffected.
- [ ] Fast tests do not download or initialize Gemma.
- [ ] Live A100 tests are explicit and excluded from normal CI.
- [ ] Git status contains no `.env`, token, cache, or safetensors artifact.

## 11. Suggested validation commands

```bash
conda activate MA-CC

# Fast contract/unit/integration tests (no model load)
pytest tests/test_constrained_client_contract.py \
  tests/test_gemma_local_client_unit.py \
  tests/test_gemma_language_game_integration.py

# Live internal API and logits test
python scripts/gemma4_api_test/test_internal_api.py

# Existing regression suite
pytest

# Tiny live game; exact CLI flags should follow the implemented provider option
python -m naming_game.cli run \
  --provider gemma_local \
  --model google/gemma-4-12B-it \
  --num-agents 2 \
  --update-mode sequential \
  --interactions 2 \
  --temperature 0 \
  --concurrency 1

git status --short
```

The CLI example is a target interface for implementation; verify the current
parser's final option names when Phase 3 is implemented.
