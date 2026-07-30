# Pre-GPU Remediation Plan for Gemma Convention Output Formats

## 1. Purpose

Finish and verify the CPU-testable work from
`tdd/gemma_convention_output_formats_task.md` before granting an implementation agent
GPU access and following `tdd/gemma4_gpu_validation_handoff.md`.

The current implementation contains much of the intended public surface, but it is not
ready for expensive live validation. The existing 77-test suite passes, while the
required output-format coverage is absent and several handoff commands or assertions
cannot succeed as written.

This plan is the corrective work item. The original task remains the source of truth for
output semantics. The GPU handoff must be corrected only after the CPU acceptance gate
in this plan passes.

## 2. Observed gaps

The remediation agent must reproduce and address these facts:

1. `tests/test_convention_output_formats.py` is referenced by the handoff but does not
   exist.
2. The current fake contract tests exercise `complete_constrained()` but not the new
   `complete_decision()` contract.
3. The basic Naming Game constrained speaker path still depends on
   `provider_name == "gemma_local"` instead of capability alone.
4. That path still uses `max(temperature, 1.0)`, changing valid positive temperatures
   such as `0.5`.
5. `_TransformersGemmaRuntime.score()` performs one model forward pass per choice. The
   required single-token, one-forward-pass fast path does not exist.
6. `choice_reason` continuation construction has not been validated through a fake
   runtime contract that proves the authoritative action and exact `\nReason: ` prefix
   are preserved.
7. The prompt hash changes with the decision format, but the persisted prompt-version
   value itself does not identify the format.
8. The public live diagnostic reads nonexistent statistics keys (`attempts` and
   `successes`) rather than the keys returned by `_RequestStats.snapshot()`.
9. The GPU handoff invokes a nonexistent `empowerment` CLI subcommand instead of
   `experiment`.
10. The handoff currently states that deterministic fake tests and the full CPU task
    pass even though the required tests are absent.

## 3. Execution boundary

All remediation work in Sections 4–6 is CPU-only. The remediation agent must not:

- instantiate `_TransformersGemmaRuntime`;
- call a model or tokenizer `from_pretrained()` method;
- run `scripts/gemma4_logits_test/test_gemma4_logits.py`;
- run `scripts/gemma4_api_test/test_internal_api.py`;
- run a live `gemma_local` experiment;
- download or inspect a real checkpoint, tokenizer, or model configuration;
- modify `.env`, Hugging Face credentials, `HF_HOME`, CUDA, PyTorch, or Transformers;
- claim that real assistant-boundary tokenization or generation works; or
- weaken assertions merely to make a fake test pass.

Use injected runtimes and dependency-free types. Heavy imports must remain inside the
real-runtime construction boundary.

## 4. Required implementation order

### Slice 1 — establish failing output-format tests

Create `tests/test_convention_output_formats.py` before changing implementation. Cover:

- `json_reason` remains the default in both public configuration objects;
- all three output formats and both selection policies load successfully;
- unknown formats and policies fail during configuration loading;
- zero, negative, infinite, and NaN `choice_temperature` values fail;
- the three prompts share identical game/history/payoff context;
- only the response instruction changes between modes;
- `json_reason` retains the legacy JSON instruction and parser behavior;
- `choice_reason` requires the first exact legal action and a non-blank `Reason:` body;
- `choice_only` accepts only a stripped exact legal action;
- JSON, prose prefixes, Markdown, labels, blank reasons, and substring-only actions are
  rejected in the text modes;
- multi-character actions cannot match by substring;
- displayed action order is preserved exactly; and
- prompt hashes and persisted prompt versions distinguish all three formats.

Do not update snapshots until the expected semantics have been reviewed against the
original task.

### Slice 2 — complete the combined-client fake contract

Expand `tests/test_constrained_client_contract.py` and, where appropriate,
`tests/test_gemma_local_client_unit.py` with an instrumented fake runtime. Test:

- `complete_decision()` satisfies `ConstrainedDecisionClient`;
- caller choice order, token IDs, log likelihoods, and normalized probabilities are
  unchanged in the response;
- exact ties select the first displayed choice under `argmax`;
- seeded `sample` is repeatable;
- at least two fixed seeds select different choices for a non-degenerate distribution;
- sampling changes only the selection, not the probabilities;
- `choice_only` returns exactly the authoritative choice and never calls generation;
- `choice_reason` makes one public call, performs one score/select/reason operation under
  one semaphore acquisition, and returns `<choice>\nReason: <text>`;
- blank or malformed rationale state never changes the authoritative choice;
- prompt, candidate, and reason token accounting is explicit and correct;
- request attempts, successes, failures, and latency are updated exactly once;
- invalid output format, policy, choices, temperatures, token counts, score lengths, and
  non-finite scores fail predictably;
- concurrent first calls initialize the runtime once;
- inference remains serialized;
- cancellation and repeated `close()` retain safe behavior; and
- importing the public modules does not import Transformers or initialize CUDA.

Refactor shared validation, normalization, and selection helpers so
`complete_constrained()` and `complete_decision()` cannot silently diverge.

### Slice 3 — implement an explicit scoring fast path and fallback

Refactor the real runtime behind a small, fakeable scoring boundary:

1. Apply the chat template once per decision.
2. Resolve each semantic choice to its candidate token sequence at the assistant
   continuation boundary.
3. If every candidate is exactly one token, perform one model forward pass and gather
   all allowed logits from the same next-token vector.
4. If any candidate is multi-token, use the teacher-forced sequence-scoring fallback
   without truncating to the first token.
5. Preserve semantic choices independently from token IDs.
6. Return scores in displayed caller order.

CPU tests must use fake tensor/runtime adapters to prove dispatch and call counts. They
must not claim whether Gemma renders a semantic action as `"A"`, `" A"`, or another
boundary-specific sequence. That remains a GPU observation.

If the real boundary cannot be represented cleanly without tokenizer evidence, expose a
narrow boundary-resolution hook and defer only its Gemma-specific result—not the
single-forward/fallback control flow—to GPU validation.

### Slice 4 — make reason continuation auditable

Define one internal runtime operation for a combined decision rather than assembling an
unverified assistant message in the public client. Its contract must:

- accept the already formatted messages and semantic choices;
- score and select before generating a reason;
- condition continuation on the authoritative choice plus exact `\nReason: ` delimiter;
- cap generation at `max_reason_tokens` and normal stop tokens;
- return the generated reason separately from the synthesized action-first content;
- mark an empty or malformed rationale invalid without changing the selected action;
- perform no generation at all for `choice_only`; and
- remain one logical public request even if multiple internal operations are necessary.

Fake tests must inspect the exact continuation input. Real chat-template prefix support
remains a GPU validation item.

### Slice 5 — repair game integration

Add tests to `tests/test_naming_convention_game.py` and
`tests/test_gemma_language_game_integration.py` for:

- all three convention output modes;
- ordinary OpenAI-compatible clients using strict generated-text parsing;
- any client implementing the combined capability using the combined path, regardless
  of provider name;
- the displayed randomized order being passed to scoring, tie-breaking, decisions,
  persistence, and audit reconstruction;
- `choice_temperature=0.5` reaching the constrained call unchanged;
- simultaneous calls for the two players in a pair;
- malformed reasons never changing payoff or state;
- forced and committed actions making no model request; and
- reasons remaining absent from agent memory and later prompts.

For the basic Naming Game speaker path:

- select the constrained path by callable capability, not provider name;
- score against a prompt requesting the same bare continuation;
- preserve every configured positive temperature unchanged;
- handle legacy generation temperature `0` through an explicit, documented positive
  constrained temperature rather than `max(temperature, 1.0)`; and
- test a capable fake provider whose name is not `gemma_local`.

Do not change pair selection, simultaneous decisions, payoffs, listener behavior,
committee interventions, or memory semantics.

### Slice 6 — repair scientific identity and persistence

Add fake-backed empowerment tests proving:

- the format, policy, and choice temperature appear in configuration JSON,
  fingerprints, execution metadata, audit traces, and final summaries;
- persisted prompt-version values identify the actual decision format;
- prompt hashes differ across all formats;
- a different format or policy uses a different shard fingerprint;
- primary interaction rows contain both players' output format, method, reason validity,
  displayed allowed choices, log likelihoods, probabilities, selected probability, and
  entropy where applicable;
- displayed allowed choices remain available as audit metadata for remote generated
  decisions even when constrained probabilities are null;
- forced and committed decisions have null rationale/probability fields and accurate
  method flags;
- deterministic JSON serialization preserves displayed order;
- legacy Parquet histories remain readable; and
- audit reconstruction uses immutable pre-interaction memory and the actual format.

Use a format-aware prompt-version helper instead of one constant value shared by all
three prompt contracts. Keep the interaction schema backward compatible.

### Slice 7 — repair live artifacts without running them

Update `scripts/gemma4_api_test/test_internal_api.py` so static review and fake/unit
coverage prove that it:

- reads `actual_calls`, `successful_calls`, and `failed_calls` from client statistics;
- checks one logical success after `choice_only` and exactly one additional success
  after `choice_reason`;
- checks that `choice_only` caused zero rationale-generation calls;
- prints the semantic choices, token IDs, log likelihoods, probabilities, selected
  choice, reason validity, usage, and runtime diagnostics;
- includes a multi-token diagnostic choice;
- exercises deterministic ties and seeded sampling without assuming sampling changes
  probabilities;
- confirms runtime reuse and safe close behavior; and
- never prints credentials, environment values, or cache paths containing secrets.

Correct the GPU command to:

```bash
python -m naming_game.cli experiment \
  --config configs/empowerment_gemma4_gpu_smoke.yaml \
  --no-resume \
  --output-dir results/gemma4_empowerment_smoke
```

Validate the YAML through `load_experiment_config()` on CPU, but do not execute it.

## 5. CPU acceptance gate

The implementation is ready to hand to the GPU agent only when every command below
passes in the configured Python 3.11 environment without loading Gemma:

```bash
conda run -n MA-CC pytest \
  tests/test_convention_output_formats.py \
  tests/test_constrained_client_contract.py \
  tests/test_gemma_local_client_unit.py \
  tests/test_gemma_language_game_integration.py \
  tests/test_naming_convention_game.py \
  tests/test_empowerment_experiment.py

conda run -n MA-CC pytest

conda run -n MA-CC python -c \
  "import sys; import naming_game; import naming_game.gemma_local_client; assert 'transformers' not in sys.modules"

conda run -n MA-CC python -m naming_game.cli --help
git diff --check
git status --short
```

The remediation report must include exact test counts and results. It must explicitly
state that no Gemma model, tokenizer, configuration, or weights were loaded or
downloaded.

## 6. Required rewrite of the GPU handoff

After the CPU gate passes, update `tdd/gemma4_gpu_validation_handoff.md` in place. Remove
claims based on missing tests and record:

- the exact CPU commit/tree state and passing test counts;
- the corrected commands;
- the implemented single-token and multi-token code paths;
- the fake evidence for one logical call and zero reason generation in `choice_only`;
- the fake evidence for authoritative selection plus one combined response in
  `choice_reason`;
- all remaining unknowns that only the real tokenizer/checkpoint can resolve; and
- a stop condition requiring the GPU agent to report semantic changes instead of making
  them silently.

Do not create another competing GPU handoff after this plan. The existing handoff is the
one the GPU agent will execute.

## 7. GPU-agent safety envelope

Once the CPU gate passes, the GPU agent may perform only the live validation described
by the corrected handoff. It must:

1. Begin from the recorded clean/intended tree and run the CPU gate before model load.
2. Use the already configured environment and credentials; never edit `.env` or install
   CUDA packages.
3. Run the known-good logits smoke before the combined API diagnostic.
4. Inspect assistant-boundary renderings and token IDs before interpreting
   probabilities.
5. Verify the one-token fast path with `A`/`B` and the fallback with a genuinely
   multi-token boundary choice.
6. Verify `choice_only` before attempting reason generation.
7. Run the tiny empowerment config only after public API checks pass.
8. Stop on authentication, model-access, environment, CUDA, or OOM failures; do not
   bypass them or silently change dtype/device/model.
9. Make only narrow tokenizer/template/generation compatibility fixes supported by
   observed evidence.
10. Keep live outputs under ignored `results/` paths and confirm that no credential,
    cache, tokenizer, configuration snapshot, or model artifact enters Git.

## 8. Definition of done

- [ ] All originally required output-format tests exist and pass.
- [ ] The combined client contract is tested entirely with deterministic fakes.
- [ ] `choice_only` performs no reason generation.
- [ ] `choice_reason` is one logical public call with an authoritative selected action.
- [ ] Single-token scoring uses one forward pass; multi-token scoring uses the fallback.
- [ ] Positive temperatures are not clamped.
- [ ] Capability detection does not require a provider-name match.
- [ ] Prompt versions, hashes, fingerprints, rows, traces, and summaries preserve the
      scientific configuration.
- [ ] The live diagnostic uses valid statistics fields.
- [ ] The GPU handoff uses valid CLI commands and makes no unsupported success claims.
- [ ] The full CPU suite passes without importing Transformers or loading Gemma.
- [ ] The corrected GPU handoff contains only the remaining live-checkpoint questions.

The correct conclusion before GPU access is:

> The output-format implementation and combined-decision behavior are verified against
> deterministic CPU fakes, including fast-path dispatch and persistence. Real Gemma
> assistant-boundary tokenization, logits, chat-template prefix continuation, CUDA
> behavior, and checkpoint-specific output remain unvalidated and are ready for the
> controlled GPU handoff.
