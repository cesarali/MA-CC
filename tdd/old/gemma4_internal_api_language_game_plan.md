# CPU-Only Implementation Task: Internal Gemma 4 API for Language Games

## 1. Task and execution boundary

Implement the repository code, tests, CLI integration, and live-test scripts for an
internal `google/gemma-4-12B-it` client **without downloading, loading, or running the
real model**. This task is intentionally assigned to an environment that may have no
GPU, no cached checkpoint, and no Hugging Face authentication.

The implementation agent must treat the existing successful smoke test as the runtime
reference:

```text
scripts/gemma4_logits_test/test_gemma4_logits.py
scripts/gemma4_logits_test/README.md
```

Extract and adapt the established behavior; do not import the smoke-test script as a
library and do not try to reproduce its live result locally.

### Non-negotiable restrictions for this task

The CPU-only implementation agent must **not**:

- call `AutoProcessor.from_pretrained()` or any model/tokenizer `from_pretrained()`;
- run `scripts/gemma4_logits_test/test_gemma4_logits.py`, including with `--cpu`;
- run the future live internal-API test or a live `gemma_local` game;
- download Gemma weights, tokenizer files, configuration files, or snapshots;
- authenticate with Hugging Face or inspect/copy another user's credentials;
- run `huggingface-cli login`, `hf download`, `git lfs`, `snapshot_download`, `curl`,
  or equivalent model-fetching commands;
- install or upgrade CUDA, PyTorch, Transformers, torchvision, or other packages merely
  to perform the live test;
- modify `.env`, create a new model cache, or copy model artifacts into the repository;
- weaken or remove CUDA/model validation just to make an unvalidated live path appear
  successful on CPU;
- claim that Gemma compatibility, tokenization, generation, numerical output, memory
  use, or model reuse has been live-validated.

If an optional dependency is unavailable in the active shell, keep imports lazy and
continue with the repository tests and pure/fake-runtime tests that can run. Record any
unexecuted CPU tensor tests in the handoff. Do not solve a missing package by downloading
the model or changing the machine's ML stack.

### What the CPU-only agent is expected to finish

The CPU-only agent should implement all normal application code and as much automated
verification as possible using dependency injection, tiny deterministic fakes, and
CPU tensors where an already-configured CPU PyTorch environment is available. It must
also write the live scripts but must not execute them.

The later GPU agent should be left with validation and small compatibility corrections,
not an unwritten feature.

## 2. Definition of done for the CPU-only agent

The CPU-only task is complete when:

1. the typed constrained-choice API exists;
2. `GemmaLocalAsyncLLMClient` implements the existing ordinary completion contract;
3. the real runtime is lazy and cannot initialize during ordinary imports or fake tests;
4. fake-runtime tests cover lifecycle, concurrency, scoring behavior, statistics, and
   failures;
5. provider selection and artifact metadata understand `gemma_local`;
6. the language games can use fake constrained decisions and log their distributions;
7. opt-in live scripts are implemented but have not been run;
8. existing mock, University, and OpenAI behavior still passes its regression tests;
9. `tdd/gemma4_gpu_validation_handoff.md` is created with exact commands and unresolved
   live questions for a GPU-enabled agent; and
10. no model, cache, credential, or generated live output appears in Git status.

“Done” in this document always means **CPU implementation complete, live Gemma
validation pending**.

## 3. Existing repository boundary to preserve

The games depend on `LLMClient` in `src/naming_game/api_client.py`:

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

`SequentialNamingGame`, `SynchronousParallelNamingGame`, the reasoning game, the
convention game, the benchmark runner, and the empowerment experiment receive a client
instead of constructing a provider. Preserve that dependency direction.

Provider creation is partly centralized in `src/naming_game/cli.py`, but the current
repository has two gaps that this task must fix:

- the `run` subcommand currently has `--mock` but no general `--provider` option; and
- `src/naming_game/runner.py` currently labels every non-mock backend as
  `university_proxy` instead of using `client.provider_name`.

The current run option is `--num-interactions`, not `--interactions`. All new docs and
scripts must use the implemented option name.

## 4. Runtime facts already established by the reference smoke test

Use these as implementation inputs, not as an invitation to rerun the model:

- model ID: `google/gemma-4-12B-it`;
- text-only loading uses `AutoProcessor` and `AutoModelForMultimodalLM`;
- the validated setup used BF16 and `device_map="auto"` on an A100;
- the checkpoint chat template accepted text messages with
  `enable_thinking=False`;
- the smoke test observed a 262,144-token vocabulary;
- weights used approximately 22.3 GiB and peak allocated memory was approximately
  22.4 GiB for the diagnostic prompt;
- ordinary generation, next-token logits, and teacher-forced sequence scoring worked;
- checkpoint storage came from `HF_HOME` in the ignored repository `.env`.

Do not hard-code the observed vocabulary size as the only accepted value. The real
runtime should check tokenizer/model consistency and report the observed size. The GPU
agent will confirm the value for the installed checkpoint and library versions.

## 5. Public API contract

### 5.1 Preserve ordinary completions

Add `GemmaLocalAsyncLLMClient`, structurally compatible with the existing `LLMClient`
protocol. Its `complete()` method returns the existing `LLMResponse`.

Required behavior:

- default `model == "google/gemma-4-12B-it"`;
- `provider_name == "gemma_local"`;
- `concurrency == 1` for the initial implementation;
- requests remain stateless;
- the runtime factory is invoked at most once per client lifetime, including when
  multiple first requests race;
- `temperature == 0` requests deterministic generation;
- `temperature > 0` requests sampling and uses the supplied seed when the installed
  Transformers version safely supports it;
- only newly generated tokens are decoded;
- prompt, completion, and total token usage are populated;
- request statistics follow the existing repository conventions;
- `close()` is idempotent and does not clear CUDA memory between requests.

Seeded real generation is a live compatibility item. Implement the most conservative
version justified by the working script and Transformers interface, cover argument
forwarding with a fake runtime, and list the real seeded-generation check in the GPU
handoff. Do not manipulate process-global RNG state outside a serialized inference
section.

### 5.2 Add constrained-choice scoring as a separate capability

Add immutable shared types, preferably in `src/naming_game/local_model_types.py`:

```python
from dataclasses import dataclass
from collections.abc import Sequence
from typing import Protocol

from .models import TokenUsage


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

Inspect provider identity through `client.provider_name`; it does not need to be
duplicated in every response.

Required semantics:

- returned scores preserve caller choice order;
- probabilities are normalized only over the supplied choices;
- probabilities are finite and sum to one within `1e-5`;
- `selected_choice` is the deterministic highest-probability choice;
- ties use caller order, so selection is reproducible;
- sequence log-likelihood is the stable scoring primitive;
- multi-token choices are supported;
- `seed` is reserved for a later explicitly selected constrained-sampling policy and
  must not silently change deterministic argmax behavior now.

Do not expose a full 262,144-element logit tensor through the public response.

### 5.3 Validation

Reject clearly:

- empty messages;
- message objects whose keys are not exactly `role` and `content`;
- non-string or blank roles/content where the existing client contract requires text;
- `max_tokens < 1`;
- non-finite or negative generation temperature;
- empty, blank, or exactly duplicated choices;
- non-finite or non-positive constrained temperature;
- candidates that tokenize to an empty sequence;
- non-finite logits, log-probabilities, scores, or normalized probabilities;
- tokenizer/model vocabulary inconsistencies;
- unsupported model IDs and unsupported Transformers versions;
- unavailable CUDA unless CPU loading was explicitly requested by a human caller.

The implementation must never log prompt contents, generated token contents, access
tokens, or full logits by default.

## 6. Source layout and dependency isolation

Target layout:

```text
src/naming_game/
├── api_client.py
├── local_model_types.py
└── gemma_local_client.py

scripts/gemma4_api_test/
├── README.md
└── test_internal_api.py

tests/
├── test_constrained_client_contract.py
├── test_gemma_local_client_unit.py
└── test_gemma_language_game_integration.py

tdd/
└── gemma4_gpu_validation_handoff.md
```

Heavy or optional imports (`torch`, `transformers`, `torchvision`, and related Hugging
Face components) must be behind real-runtime initialization. These operations must not
happen when:

- importing `naming_game`;
- importing `GemmaLocalAsyncLLMClient`;
- constructing a client with an injected fake runtime/factory;
- using mock, University, or OpenAI providers;
- collecting ordinary tests; or
- displaying CLI help.

Add a small internal runtime protocol or equivalent dependency-injection boundary.
Fake runtimes should return deterministic generated tokens and deterministic candidate
scores without importing Transformers or touching CUDA. Test the public client through
that boundary rather than creating a second fake-only implementation of the client.

## 7. Real Gemma runtime implementation to write but not execute

### 7.1 Initialization

Adapt the already-working helpers from
`scripts/gemma4_logits_test/test_gemma4_logits.py`:

1. find and load the repository-root `.env` before Hugging Face imports;
2. validate `HF_HOME` without printing credentials;
3. check CUDA visibility unless explicit CPU loading was requested;
4. load `AutoProcessor` and `AutoModelForMultimodalLM` once;
5. select `dtype` versus `torch_dtype` according to the installed Transformers API;
6. call `model.eval()`;
7. derive the input device from `hf_device_map`, not a hard-coded `cuda:0`;
8. retain the runtime for all requests; and
9. report model/device/dtype/memory metadata once at startup.

The CPU-only agent writes this path by following the reference implementation but must
not cause it to run. Isolate model construction in a factory that tests can replace and
assert is invoked once.

### 7.2 Message formatting

Use the known text-only template call:

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

Only retry without `enable_thinking` for the narrow documented compatibility exception
already handled by the smoke test. Preserve the processor's returned attention mask and
any other required text input fields rather than replacing them indiscriminately.

### 7.3 Ordinary generation

Inside `torch.inference_mode()`:

- use `do_sample=False` for `temperature == 0`;
- use `do_sample=True` and pass temperature for `temperature > 0`;
- set `max_new_tokens=max_tokens`;
- keep inference serialized;
- slice output tokens after the input prompt length; and
- decode only the new tokens.

The GPU agent must validate real seeded sampling, exact decode behavior, stop tokens,
and token counts.

### 7.4 Teacher-forced choice scoring

Implement the algorithm demonstrated by the smoke test:

1. format and tokenize the prompt once;
2. tokenize each candidate without special tokens;
3. append candidate IDs to prompt IDs;
4. extend the original attention mask for the appended candidate tokens;
5. preserve or deliberately extend other model input fields where applicable;
6. teacher-force a forward pass for each candidate;
7. select logits from `prompt_length - 1` through the position before the final
   candidate token;
8. compute `log_softmax` in float32;
9. gather the candidate token log-probabilities;
10. sum them into `log p(choice | prompt)`; and
11. normalize candidate sequence scores with
    `softmax(sequence_log_likelihoods / temperature)` in float32.

Use the smoke test's current candidate rendering—encoding the supplied choice exactly,
without silently adding a leading space—as the initial implementation. Mark the answer
boundary as a live-validation item. The CPU-only agent must test the generic alignment
algorithm with a tiny tokenizer/model but must not claim that fake tokenization proves
Gemma's exact boundary behavior.

The later GPU agent must inspect and, if necessary, correct:

- whether `"A"` or `" A"` is the correct rendered continuation after the real chat
  template;
- whether tokenizing a candidate separately produces the same continuation IDs as
  contextual tokenization;
- which processor fields besides `input_ids` and `attention_mask` must be extended for
  text-only Gemma;
- tokenizer/model vocabulary agreement; and
- one-token versus multi-token behavior with the real tokenizer.

Do not hide these questions by stripping choices or changing their public values.

### 7.5 Concurrency and lifecycle

Use an `asyncio.Semaphore(1)` around all local inference and run blocking inference via
`asyncio.to_thread()`. Ensure concurrent first calls cannot create two runtimes. Do not
launch simultaneous forward passes in phase 1.

For ordinary generation, token usage is the prompt length plus newly generated tokens.
For constrained scoring, document and test usage as actual unbatched scoring work:

```text
prompt_tokens     = prompt_length * number_of_choices
completion_tokens = sum(candidate_token_lengths)
total_tokens      = prompt_tokens + completion_tokens
```

One public `complete_constrained()` invocation counts as one request attempt even though
the initial runtime performs one model forward pass per candidate.

## 8. Provider and CLI integration

Add `gemma_local` without changing remote/mock behavior.

For the `run` subcommand:

- add `--provider` with supported values such as `university`, `openai`, and
  `gemma_local`;
- retain `--mock` for backward compatibility, but reject or clearly resolve conflicting
  `--mock`/`--provider` combinations;
- use `--num-interactions`, the repository's existing option;
- default local concurrency to or require it to equal `1`;
- support only the local dtype/device-map flags that the implementation actually uses;
- do not instantiate the real runtime while parsing arguments or displaying help.

Extend the centralized provider factory used by configured experiments. Validate
`gemma_local` configuration and fallback combinations. A local provider must not be used
as an automatic fallback that unexpectedly downloads or loads a model.

Update `src/naming_game/runner.py` to derive artifact backend/provider identity from
`client.provider_name`, with a stable compatibility mapping only if existing artifact
formats require it. Assert that fake-backed local runs record `gemma_local`, not
`university_proxy`.

Export the new public client and types from `src/naming_game/__init__.py` only if doing so
does not trigger heavy imports.

## 9. Language-game integration

### 9.1 Ordinary compatibility

First prove with an injected fake runtime that the existing game can call
`GemmaLocalAsyncLLMClient.complete()` without provider-specific changes. Use:

- 2 agents;
- sequential mode;
- 2 interactions;
- no reasoning interactions;
- temperature 0;
- concurrency 1; and
- a fixed seed.

Assert four successful model calls, normal output artifacts, provider identity
`gemma_local`, and one runtime construction. This is a fake-backed integration test, not
a live model test.

### 9.2 Constrained decisions

Where the legal action set is already known, pass it explicitly:

- speaker selection uses the speaker's inventory;
- convention-game selection uses `config.actions` in display order; and
- other experimental actions use their enumerated state/configuration values.

Never infer allowed choices by parsing prompt text. The constrained response must be the
authoritative state-transition action.

For a rationale-producing interaction:

1. select the action using `complete_constrained()`;
2. optionally generate a rationale conditioned on that fixed action; and
3. never reparse the rationale to replace the constrained action.

Add optional log fields without changing established required columns:

```text
decision_method: generated | constrained_sequence
allowed_choices: [A, B]
choice_log_likelihoods: {A: ..., B: ...}
choice_probabilities: {A: ..., B: ...}
selected_choice_probability: ...
choice_entropy: ...
```

Keep remote clients on generated decisions unless a separate constrained adapter is
explicitly implemented. Do not assume every `LLMClient` implements
`ConstrainedLLMClient`; use explicit configuration and capability checks.

## 10. CPU/fake test requirements

Tests must not patch around an accidental real load after it begins. They must inject a
fake before any real-runtime factory or Transformers import can be reached.

Cover:

- ordinary `LLMClient` structural compatibility;
- choice-order preservation;
- deterministic first-in-order tie handling;
- one-token and multi-token choices;
- correct teacher-forcing offset/alignment with tiny CPU tensors when available;
- prompt tokens excluded from candidate sequence log-likelihoods;
- float32 normalization and an approximately unit probability sum;
- blank/duplicate choices and invalid temperatures;
- empty candidate tokenization;
- non-finite logits, scores, and probabilities;
- tokenizer/model vocabulary mismatch;
- exact message validation;
- ordinary generation slices off prompt tokens;
- exact token-usage accounting under the documented semantics;
- request success/failure statistics;
- semaphore-limited active inference of one;
- racing first requests initialize one runtime;
- several calls reuse the same runtime;
- idempotent `close()`;
- imports and CLI help do not import Transformers or initialize CUDA;
- a two-agent sequential fake-backed local game;
- synchronous fake-backed requests remain serialized;
- the selected constrained action drives state updates;
- single-action inventories;
- constrained decision logging; and
- existing mock/remote regression behavior.

Register a marker such as `gemma_live` if a pytest live test is added. Ordinary `pytest`
must never select the live test and must never fetch a checkpoint.

## 11. Live scripts to implement but not run

Create `scripts/gemma4_api_test/test_internal_api.py` and its README. The script must
call only the public client API and use this diagnostic:

```text
Question: Which number is larger, 7 or 3?

A. 7
B. 3
C. They are equal

Return only A, B, or C.
```

It should call:

```python
response = await client.complete_constrained(
    messages,
    choices=["A", "B", "C"],
    temperature=1.0,
)
```

The written live script should check:

- client model and provider identity;
- all choices and token IDs are present;
- scores/probabilities are finite and ordered;
- probabilities sum to one;
- `A` has the greatest probability;
- ordinary `complete()` returns non-empty content;
- a second request reuses the runtime; and
- startup/post-request memory diagnostics are available.

The CPU-only agent must syntax-check or inspect this script without executing its main
function. Do not add a test that accidentally runs merely because pytest imports it.

## 12. Required GPU handoff document

Before finishing, create `tdd/gemma4_gpu_validation_handoff.md`. It must be a direct,
copy-pasteable task for an agent that has the cached model, compatible dependencies, and
an A100-class GPU.

It must contain:

1. a statement that CPU implementation is complete but live validation is pending;
2. the commit/worktree state and a concise list of implemented files;
3. the exact environment assumptions taken from the existing logits README;
4. commands to run the focused CPU tests first;
5. commands to run the public internal-API live test;
6. the exact tiny live game command using `--num-interactions`;
7. a requirement to compare behavior with
   `scripts/gemma4_logits_test/test_gemma4_logits.py` without rewriting known-good
   behavior unnecessarily;
8. explicit tokenizer-boundary questions from section 7.4;
9. seeded generation, processor fields, device placement, dtype, vocabulary, token
   counts, model-reuse, and CUDA-memory checks;
10. expected acceptance results and space to record observed values;
11. failure triage for authentication, missing cache, Transformers compatibility,
    torchvision, CUDA, and OOM errors;
12. permission to make narrow compatibility fixes supported by actual observations;
13. a requirement to rerun the complete regression suite after fixes; and
14. a final `git status --short` check proving no `.env`, token, cache, tokenizer,
    config, or model-weight artifacts were added.

The GPU agent must not redesign the API merely because a narrow runtime compatibility
fix is needed. Material API or experiment-semantics changes should be reported rather
than silently introduced.

## 13. CPU-only validation commands

Use the already-configured repository environment if available. Do not install Gemma or
invoke a loader.

```bash
# Focused CPU/fake tests. These names are targets and should match created files.
pytest tests/test_constrained_client_contract.py \
  tests/test_gemma_local_client_unit.py \
  tests/test_gemma_language_game_integration.py

# Existing regression suite. It must not collect a live Gemma invocation.
pytest

# Static import/help checks that must not initialize the model.
python -c "import naming_game; import naming_game.gemma_local_client"
python -m naming_game.cli --help
python -m naming_game.cli run --help

git status --short
```

Do **not** include a live Gemma command in the CPU agent's executed validation log. Live
commands belong in `tdd/gemma4_gpu_validation_handoff.md` for the later agent.

## 14. CPU implementation checklist

- [ ] No Gemma/Hugging Face download or real loader call was made.
- [ ] No ML packages or CUDA components were installed or changed for live validation.
- [ ] Typed constrained API is implemented.
- [ ] Local client preserves ordinary `LLMClient.complete()` behavior.
- [ ] Heavy imports and runtime initialization are lazy.
- [ ] Fake dependency injection reaches all public-client behavior.
- [ ] Runtime construction is race-safe and occurs once per client.
- [ ] Constrained scoring supports one-token and multi-token candidates in CPU/fake tests.
- [ ] Tokenizer-boundary assumptions are clearly deferred to live validation.
- [ ] Local inference is serialized.
- [ ] Provider selection includes `gemma_local`.
- [ ] Direct-run CLI and backend artifact identity are corrected.
- [ ] Fake-backed ordinary and constrained games pass.
- [ ] Constrained probabilities and derived fields are logged.
- [ ] Live API script and README are written but not run.
- [ ] Existing regression tests pass, or unrelated/pre-existing failures are documented.
- [ ] GPU validation handoff is complete and uses actual implemented paths/flags.
- [ ] Git status contains no credentials, caches, tokenizer files, or model artifacts.

## 15. Final report required from the CPU-only agent

Report separately:

- implementation completed;
- CPU/fake tests executed and their results;
- tests not executable because an optional dependency was absent;
- live scripts written but intentionally not executed;
- Gemma-specific assumptions awaiting validation;
- the path to `tdd/gemma4_gpu_validation_handoff.md`; and
- confirmation that no model download or real model initialization was attempted.

Never phrase the result as “Gemma works.” The correct conclusion before GPU handoff is:

> The Gemma-local integration is implemented and verified against deterministic fakes;
> compatibility with the real checkpoint remains to be validated on the GPU host.
