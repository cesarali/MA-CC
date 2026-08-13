# Phase 9 grid preflight

- Status: **PASS** (`permitted`)
- Experiment: `imitation-random-incorrect-gpt-oss-120b-smoke`; game `hidden_bench_imitation`; provider `university` / `gwdg/openai-gpt-oss-120b`.
- Cells: 1; total episodes: 2 (shared concurrency 2).
- Axes:
- `game.options.social_group_size`: [1]
- `control.options.sensor_sample_size`: [2]
- Expected total cost: 0.00 proxy_accounting_unit; conservative: 0.00 proxy_accounting_unit.
- Rough total runtime: 96.0s.
- Preflight ID: `063cab2dbac40f9fa37bd8991b45ccfdb3542116d3315d847c7f22d0c0b0ddad` — pass this to `mas-cc experiment run --approve-preflight` to bind the launch to this estimate.

## Warnings

- Lower/expected scenarios use representative contexts; conservative demand uses each stage's maximum context and retry bound.
- Token counts use a deterministic regex estimate, not the provider tokenizer.
- Runtime is a rough planning estimate, not a service guarantee.
