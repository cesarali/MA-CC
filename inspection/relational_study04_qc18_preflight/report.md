# Phase 9 grid preflight

- Status: **PASS** (`permitted`)
- Experiment: `relational-study04-qc18-resource-grid`; game `relational_imitation_round_feedback`; provider `university` / `gwdg/openai-gpt-oss-120b`.
- Cells: 6; total episodes: 180 (shared concurrency 8).
- Axes:
- `control.options.intervention_budget`: [6, 12, 18]
- `game.options.task_id`: ['task_0001', 'task_0002']
- Expected total cost: 0.00 proxy_accounting_unit; conservative: 0.00 proxy_accounting_unit.
- Rough total runtime: 18834.0s.
- Preflight ID: `7aa66581af8f0c98a5d3b1c088aa5b3d04b500a8e367a445c253d8a079bc73cb` — pass this to `mas-cc experiment run --approve-preflight` to bind the launch to this estimate.

## Warnings

- Lower/expected scenarios use representative contexts; conservative demand uses each stage's maximum context and retry bound.
- Token counts use a deterministic regex estimate, not the provider tokenizer.
- Runtime is a rough planning estimate, not a service guarantee.
