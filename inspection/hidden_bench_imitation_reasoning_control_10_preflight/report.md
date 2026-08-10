# Phase 9 experiment preflight

- Status: **PASS** (`permitted`)
- Experiment: `hidden-bench-imitation-reasoning-control-gpt54nano`; game `hidden_bench_imitation`; provider `university` / `microsoft/gpt-5.4-nano`.
- Episodes: 5 (concurrency 5).
- Expected total cost: 0.27 proxy_accounting_unit; conservative: 0.27 proxy_accounting_unit.
- Rough total runtime: 90.0s.
- Preflight ID: `0829d87c6485f340123a03d3fb8535d5ff3a889c5b88d8ff93fa96f91adae49e` — pass this to `mas-cc experiment run --approve-preflight` to bind the launch to this estimate.

## Warnings

- Lower/expected scenarios use representative contexts; conservative demand uses each stage's maximum context and retry bound.
- Token counts use a deterministic regex estimate, not the provider tokenizer.
- Runtime is a rough planning estimate, not a service guarantee.
