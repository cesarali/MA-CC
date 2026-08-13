# Phase 9 grid preflight

- Status: **PASS** (`permitted`)
- Experiment: `imitation-random-incorrect-gpt-oss-120b-smoke`; game `hidden_bench_imitation`; provider `university` / `gwdg/openai-gpt-oss-120b`.
- Cells: 1; total episodes: 2 (shared concurrency 2).
- Axes:
- `game.options.social_group_size`: [1]
- `control.options.sensor_sample_size`: [2]
- Expected total cost: 0.00 proxy_accounting_unit; conservative: 0.00 proxy_accounting_unit.
- Rough total runtime: 84.0s.
- Preflight ID: `8295bfa01c775e54931a42b77c9cb0835e81d41e780f8de71fdf578a9a88700b` — pass this to `mas-cc experiment run --approve-preflight` to bind the launch to this estimate.

## Warnings

- Lower/expected scenarios use representative contexts; conservative demand uses each stage's maximum context and retry bound.
- Token counts use a deterministic regex estimate, not the provider tokenizer.
- Runtime is a rough planning estimate, not a service guarantee.
