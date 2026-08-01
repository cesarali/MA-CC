# Gemma 4 internal API live diagnostic

This is an **opt-in GPU test**, not part of pytest. It requires the accepted Gemma
license, an already configured `HF_HOME`, Transformers 5.5+, and an A100-class GPU.
Run only on the validation host:

```bash
python scripts/gemma4_api_test/test_internal_api.py
```

It exercises only the public client API, checks constrained sequence scoring,
ordinary generation, runtime reuse, and memory diagnostics. It may initialize the
real checkpoint and must never be run on a CPU-only implementation host.
