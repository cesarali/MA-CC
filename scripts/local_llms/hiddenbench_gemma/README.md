# Canonical HiddenBench with local Gemma

This opt-in runner adapts the canonical source-population HiddenBench protocol to the
new `mas_cc` provider and prompt interfaces. It uses the checked-in canonical
65-task dataset, the versioned paper-style discussion/vote prompts, and the lazy
`gemma_local` provider.

The default command is preflight-only. It does not import Torch or Transformers,
check CUDA, construct a provider, or load Gemma:

```bash
conda run --live-stream -n MA-CC python \
  scripts/local_llms/hiddenbench_gemma/run_hiddenbench_canonical.py
```

An eventual one-task GPU smoke run is deliberately explicit:

```bash
conda run --live-stream -n MA-CC python \
  scripts/local_llms/hiddenbench_gemma/run_hiddenbench_canonical.py \
  --task-ids 1 \
  --sessions 1 \
  --rounds 2 \
  --output-dir results/hiddenbench_canonical_gemma_smoke \
  --execute
```

Run the complete paper condition only after inspecting `preflight.json`:

```bash
conda run --live-stream -n MA-CC python \
  scripts/local_llms/hiddenbench_gemma/run_hiddenbench_canonical.py \
  --output-dir results/hiddenbench_canonical_gemma_full \
  --execute \
  --confirm-full-benchmark
```

The full default condition is 65 tasks × 10 sessions. The checked-in canonical
dataset has 58 four-agent tasks and seven three-agent tasks. With 15 discussion
events, that is 17,340 logical calls in total. A four-agent session makes 27
calls: four pre-votes, 15 messages, four post-votes, and four full-profile votes.

Outputs are:

```text
output-dir/
├── resolved_run.json
├── preflight.json
├── audit.jsonl
└── results.json
```

`audit.jsonl` stores provider-independent requests, normalized responses,
redacted raw responses, usage, and validation results. `results.json` is written
atomically after each completed task. Use `--resume` to skip those completed
tasks after an interruption. Correct answers are used only after inference for
evaluation and are never inserted into a provider request.
