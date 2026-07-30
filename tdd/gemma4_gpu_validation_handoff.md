# Gemma 4 GPU validation handoff

The CPU implementation removed the JSON-prompt/bare-choice mismatch. The three modes
are `json_reason` (legacy/default), `choice_reason` (action then reason in one public
call), and `choice_only` (bare action, with no rationale generation). Deterministic fake
tests and the full CPU suite pass. **Real assistant-boundary tokenization, logits,
combined generation, and CUDA behavior remain unvalidated.** Begin with a clean tree.

## Before loading a model

```bash
pytest tests/test_convention_output_formats.py tests/test_constrained_client_contract.py tests/test_gemma_local_client_unit.py tests/test_gemma_language_game_integration.py tests/test_empowerment_experiment.py
pytest
python -c "import naming_game; import naming_game.gemma_local_client"
python -m naming_game.cli --help
git status --short
```

If the first test file is absent on the branch, run the remaining focused files. CPU
tests use injected runtimes only; they cannot establish whether the boundary rendering
is `"A"`, `" A"`, or something else.

## Live GPU sequence

Use the known-good environment facts in `scripts/gemma4_logits_test/README.md` (A100,
BF16, `device_map=auto`, Transformers 5.5+, roughly 22.3 GiB weights), and first run:

```bash
python scripts/gemma4_logits_test/test_gemma4_logits.py
python scripts/gemma4_api_test/test_internal_api.py
python -m naming_game.cli empowerment --config configs/empowerment_gemma4_gpu_smoke.yaml --output-dir results/gemma4_empowerment_smoke
```

The public API diagnostic must verify `choice_only` returns exact content, performs no
rationale generation, and increments attempts/successes once. It must then verify
`choice_reason` returns one action-first response, normalized probabilities, a non-empty
reason, and exactly one additional logical request. Inspect raw token IDs for every
displayed choice at the real assistant boundary. Confirm the A/B single-token fast path
and teacher-forced multi-token fallback using at least one multi-token choice.

Also verify first-in-displayed-order argmax ties and repeatable seeded categorical
sampling (including a seed that changes the selection without changing probabilities).
Confirm reason generation never changes the authoritative selected action. Record
tokenizer/model vocabulary, boundary token IDs/renderings, dtype, device map, input
device, runtime reuse, prompt/generated usage, CUDA allocated/peak memory, stop behavior,
and safe `close()` behavior.

The tiny empowerment run—not merely the basic Naming Game—must be inspected. Check its
primary Parquet rows for both players' output format/method, reason validity, displayed
allowed-choice order, log likelihoods, normalized probabilities, selected probability,
and entropy. Reconstruct sampled audit prompts from pre-interaction memory and confirm
their actual mode and order.

Make only narrow tokenizer/generation compatibility fixes supported by observations.
Report rather than silently introducing material output-semantics, selection-policy, or
scientific-fingerprint changes. After fixes run `pytest` again. Finally prove that no
`.env`, credential/token, cache, tokenizer, configuration snapshot, model weights, or
generated live results enter Git; remove live output before committing.
