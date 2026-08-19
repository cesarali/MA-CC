# Phase 9 grid preflight

- Status: **PASS** (`permitted`)
- Experiment: `relational-study01-c-epistemic-control-overnight`; game `relational_imitation_round_feedback`; provider `university` / `gwdg/openai-gpt-oss-120b`.
- Cells: 40; total episodes: 80 (shared concurrency 8).
- Axes:
- `game.options.task_dataset_dir`: ['src/mas_cc/relational_task_generator/relational_task_generator/datasets/pop24_L2_r01', 'src/mas_cc/relational_task_generator/relational_task_generator/datasets/pop24_L2_r03', 'src/mas_cc/relational_task_generator/relational_task_generator/datasets/pop24_L2_r06', 'src/mas_cc/relational_task_generator/relational_task_generator/datasets/pop24_L2_r12']
- `game.options.task_id`: ['task_0001', 'task_0002', 'task_0003', 'task_0004', 'task_0005']
- `control.options.intervention_budget`: [6, 24]
- Expected total cost: 0.00 proxy_accounting_unit; conservative: 0.00 proxy_accounting_unit.
- Rough total runtime: 8370.0s.
- Preflight ID: `bba9c5ee0d4e05d60ab09eb204ca3580cf106b7de35323dea76501422ca20b81` — pass this to `mas-cc experiment run --approve-preflight` to bind the launch to this estimate.

## Warnings

- Lower/expected scenarios use representative contexts; conservative demand uses each stage's maximum context and retry bound.
- Token counts use a deterministic regex estimate, not the provider tokenizer.
- Runtime is a rough planning estimate, not a service guarantee.
