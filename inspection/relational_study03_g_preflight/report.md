# Phase 9 grid preflight

- Status: **PASS** (`permitted`)
- Experiment: `relational-study03-g-stochastic-feedback-pilot`; game `relational_imitation_round_feedback`; provider `university` / `gwdg/openai-gpt-oss-120b`.
- Cells: 2; total episodes: 20 (shared concurrency 8).
- Axes:
- `game.options.task_id`: ['task_0001', 'task_0002']
- Expected total cost: 0.00 proxy_accounting_unit; conservative: 0.00 proxy_accounting_unit.
- Rough total runtime: 2094.0s.
- Preflight ID: `2da94281f7d642f2e6e2c302ca25e5c014152dfff2dd383d24dfc3d60fa31a97` — pass this to `mas-cc experiment run --approve-preflight` to bind the launch to this estimate.

## Warnings

- Lower/expected scenarios use representative contexts; conservative demand uses each stage's maximum context and retry bound.
- Token counts use a deterministic regex estimate, not the provider tokenizer.
- Runtime is a rough planning estimate, not a service guarantee.
