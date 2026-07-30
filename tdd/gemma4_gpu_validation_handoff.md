# Gemma 4 GPU validation handoff

CPU implementation is complete and live checkpoint validation is pending. The commit
contains the dependency-free API types, lazy client/runtime, CLI/provider metadata,
fake tests, and `scripts/gemma4_api_test/` diagnostic. Begin from a clean worktree.

## Host assumptions

Use the same environment as `scripts/gemma4_logits_test/README.md`: accepted access to
`google/gemma-4-12B-it`, repository `.env` containing `HF_HOME` on high-capacity
storage, compatible Transformers (5.5+), PyTorch/torchvision/accelerate, and an
A100-class CUDA GPU. The known BF16 `device_map="auto"` load used about 22.3 GiB for
weights and 22.4 GiB peak. Do not print tokens or credentials.

## Commands

```bash
pytest tests/test_constrained_client_contract.py tests/test_gemma_local_client_unit.py tests/test_gemma_language_game_integration.py
pytest
python scripts/gemma4_api_test/test_internal_api.py
python -m naming_game.cli run --provider gemma_local --model google/gemma-4-12B-it --update-mode sequential --num-agents 2 --num-interactions 2 --reasoning-fraction 0 --temperature 0 --concurrency 1 --seed 7 --output-dir results/gemma4_live_tiny
git status --short
```

Compare observations to `scripts/gemma4_logits_test/test_gemma4_logits.py`; do not
rewrite its known-good behavior unnecessarily. Do not redesign the public API for a
narrow compatibility issue. Narrow fixes supported by observed behavior are allowed;
report material API or experiment-semantics changes instead.

## Questions and observations to record

- Is `"A"` or `" A"` the correct continuation at the real answer boundary?
- Does separate candidate encoding equal contextual continuation tokenization?
- Which processor fields beyond `input_ids`/`attention_mask` need extension?
- Record tokenizer and model vocabulary sizes (the reference observed 262,144).
- Validate one- and multi-token alignment, prompt exclusion, and exact token counts.
- Validate seeded sampling and whether `generator=` is accepted by this version.
- Record device map, input device, actual dtype, generated-token slicing/stops.
- Confirm a second request reuses the runtime and record startup/post-request/peak CUDA memory.

Expected: focused and full tests pass; A wins the diagnostic; probabilities are finite
and normalized; ordinary output is non-empty; runtime identity is stable. Record:
`transformers=___`, `torch=___`, `vocab=___`, `device=___`, `dtype=___`,
`prompt/completion=___`, `allocated/peak=___`, `boundary IDs=___`.

## Failure triage

Authentication/gating means model terms or host credentials/cache require attention;
do not copy credentials. A missing cache must be handled according to host policy.
For Transformers signature or processor-field errors, compare the reference script
and make the smallest version-supported correction. For torchvision import errors,
repair the GPU environment rather than adding eager imports. For missing CUDA stop;
do not fake success with CPU. For OOM, verify no competing process, BF16, and the
device map before changing code.

After any fix rerun the complete regression suite. Finally inspect `git status --short`
and prove no `.env`, credential/token, cache, tokenizer, config, snapshot, or model
weight artifacts were added; remove generated live outputs before committing.
