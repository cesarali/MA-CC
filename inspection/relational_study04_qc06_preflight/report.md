# Phase 9 grid preflight

- Status: **PASS** (`permitted`)
- Experiment: `relational-study04-qc06-resource-grid`; game `relational_imitation_round_feedback`; provider `university` / `gwdg/openai-gpt-oss-120b`.
- Cells: 6; total episodes: 180 (shared concurrency 8).
- Axes:
- `control.options.intervention_budget`: [6, 12, 18]
- `game.options.task_id`: ['task_0001', 'task_0002']
- Expected total cost: 0.00 proxy_accounting_unit; conservative: 0.00 proxy_accounting_unit.
- Rough total runtime: 18834.0s.
- Preflight ID: `3153f68b61e157b4ad6356128c4a9ff503768670e213ccdd9b2455b79872b2d3` — pass this to `mas-cc experiment run --approve-preflight` to bind the launch to this estimate.

## Warnings

- Lower/expected scenarios use representative contexts; conservative demand uses each stage's maximum context and retry bound.
- Token counts use a deterministic regex estimate, not the provider tokenizer.
- Runtime is a rough planning estimate, not a service guarantee.
