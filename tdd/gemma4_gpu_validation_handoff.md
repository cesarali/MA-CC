# Gemma 4 GPU validation handoff

## CPU-verified starting point

The pre-GPU remediation was completed from base commit
`cbb87f69bd3965ac8322487312f4712dac9ff4fe` on `main`. The intended worktree is
the remediation diff in the following files plus the pre-GPU plan supplied with
the task:

```text
scripts/gemma4_api_test/test_internal_api.py
src/naming_game/empowerment_experiment.py
src/naming_game/gemma_local_client.py
src/naming_game/interaction.py
src/naming_game/naming_convention_game.py
tests/test_constrained_client_contract.py
tests/test_convention_output_formats.py
tests/test_empowerment_experiment.py
tests/test_gemma_language_game_integration.py
tests/test_naming_convention_game.py
tdd/gemma4_gpu_validation_handoff.md
tdd/gemma_convention_pre_gpu_remediation_plan.md
```

At handoff, the six-file focused gate passed **78 tests**, and the complete CPU
suite passed **125 tests**. The lazy-import assertion proved that importing
`naming_game` and `naming_game.gemma_local_client` imports neither Transformers
nor PyTorch. The CLI help command passed, and
`configs/empowerment_gemma4_gpu_smoke.yaml` loaded successfully through
`load_experiment_config()`.

No Gemma model, tokenizer, configuration, checkpoint, or weights were loaded,
inspected, or downloaded during remediation. No live Gemma script or experiment
was run.

The output modes are now:

- `json_reason`: legacy/default ordinary generation and legacy JSON parsing;
- `choice_reason`: authoritative constrained action plus a reason in one public
  `complete_decision()` request; and
- `choice_only`: authoritative constrained action with no rationale generation.

Deterministic CPU fakes prove displayed-order preservation, first-in-order
argmax ties, seeded sampling, shared normalization, one logical request statistic,
one serialized score/select/reason operation, the exact
`<choice>\nReason: ` continuation prefix, and zero rationale-generation work for
`choice_only`. A dependency-free scoring adapter proves that all one-token
candidates dispatch to one batched next-token scoring call, while any multi-token
candidate dispatches every full sequence to the teacher-forced fallback.

The real-runtime code implements both paths. These CPU tests prove control flow
and accounting only; they do not prove checkpoint-specific token boundaries or
model output.

## Before loading a model

Begin from the intended tree above. Do not discard or silently alter the
remediation diff. Run:

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
  "import sys; import naming_game; import naming_game.gemma_local_client; assert 'transformers' not in sys.modules; assert 'torch' not in sys.modules"

conda run -n MA-CC python -m naming_game.cli --help

conda run -n MA-CC python -c \
  "from naming_game.empowerment_experiment import load_experiment_config; load_experiment_config('configs/empowerment_gemma4_gpu_smoke.yaml')"

git diff --check
git status --short
```

Stop before model load if the focused count is not 78, the full count is not
125, the lazy-import assertion fails, or the intended worktree differs. Report
the discrepancy rather than adapting the scientific contract.

## Controlled live sequence

Use the already configured environment and credentials. Do not edit `.env`,
authenticate again, change `HF_HOME`, install CUDA/PyTorch/Transformers packages,
or change the configured model, dtype, or device mapping to bypass a failure.

Run the known-good logits smoke first, then the combined public API diagnostic:

```bash
conda run -n MA-CC python scripts/gemma4_logits_test/test_gemma4_logits.py
conda run -n MA-CC python scripts/gemma4_api_test/test_internal_api.py
```

The public diagnostic uses the actual statistics keys `actual_calls`,
`successful_calls`, and `failed_calls`. Inspect its structured report and verify:

1. The semantic choices remain distinct from their rendered token IDs.
2. `A`, `B`, and `C` are each resolved at the real assistant continuation
   boundary; record whether the effective rendering behaves like `A`, ` A`, or
   something else.
3. The `A`/`B`/`C` decision uses the one-forward single-token path only if every
   boundary candidate is genuinely one token.
4. `MULTI_TOKEN_DIAGNOSTIC_CHOICE` is genuinely multi-token and increments the
   full-sequence fallback diagnostic.
5. `choice_only` returns content equal to the authoritative semantic choice,
   performs zero rationale-generation calls, and records exactly one logical
   attempt and success.
6. `choice_reason` records exactly one additional logical attempt and success,
   returns `<choice>\nReason: <non-blank text>`, and increments rationale
   generation once.
7. The generated reason does not replace or reinterpret the authoritative
   selected action.
8. Probabilities are finite, normalized over legal choices, and unchanged by
   sampling policy; exact ties choose the first displayed action and seeded
   sampling is repeatable.
9. Prompt/scoring/candidate/reason token accounting is internally consistent.
10. The same runtime object is reused, inference is serialized, diagnostics
    report dtype/device/vocabulary/CUDA allocated and peak memory, and repeated
    `close()` is safe.

Record raw boundary renderings and token IDs before interpreting probabilities.
Do not assume the checkpoint uses the same tokenization as the CPU fake.

Only after both live diagnostics pass, run the tiny empowerment experiment:

```bash
conda run -n MA-CC python -m naming_game.cli experiment \
  --config configs/empowerment_gemma4_gpu_smoke.yaml \
  --no-resume \
  --output-dir results/gemma4_empowerment_smoke
```

Inspect the primary Parquet rows for both players. Confirm the format and method,
reason validity, displayed allowed-choice order, log likelihoods, normalized
probabilities, selected probability, and entropy. Confirm forced/committed rows,
if present in any additional narrow diagnostic, have null rationale/probability
fields. Reconstruct sampled audit prompts from immutable pre-interaction memory
and verify their persisted format, prompt version, and displayed order.

## Checkpoint-specific questions still unresolved

CPU remediation intentionally leaves these questions for this GPU run:

- the exact Gemma assistant-boundary rendering and token IDs for each semantic
  choice;
- whether `A`/`B` are single tokens at that boundary;
- whether appending the authoritative token prefix to the applied chat template
  is accepted by this processor/model combination;
- generated stop-token behavior after the exact `\nReason: ` delimiter;
- real logits, probabilities, reason quality, token usage, dtype/device placement,
  CUDA allocation, and peak memory; and
- checkpoint-specific behavior of the tiny empowerment trajectory.

Authentication, model-access, environment, CUDA, and OOM failures are stop
conditions. Do not bypass them. Narrow tokenizer/template/generation compatibility
fixes are allowed only when supported by observed token/rendering evidence. If a
finding would change output semantics, selection policy, probability meaning,
prompt identity, persistence, or scientific fingerprints, stop and report the
proposed semantic change instead of making it silently.

After any evidence-backed compatibility fix, rerun the focused and full CPU gates.
Keep live output under ignored `results/` paths. Before handoff completion, confirm
that no `.env`, credentials, cache paths, tokenizer/configuration snapshots,
checkpoint/model weights, or generated live results enter Git.
