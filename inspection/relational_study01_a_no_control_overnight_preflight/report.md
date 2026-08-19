# Phase 9 grid preflight

- Status: **PASS** (`permitted`)
- Experiment: `relational-study01-a-no-control-overnight`; game `relational_imitation_round_feedback`; provider `university` / `gwdg/openai-gpt-oss-120b`.
- Cells: 20; total episodes: 40 (shared concurrency 8).
- Axes:
- `game.options.task_dataset_dir`: ['src/mas_cc/relational_task_generator/relational_task_generator/datasets/pop24_L2_r01', 'src/mas_cc/relational_task_generator/relational_task_generator/datasets/pop24_L2_r03', 'src/mas_cc/relational_task_generator/relational_task_generator/datasets/pop24_L2_r06', 'src/mas_cc/relational_task_generator/relational_task_generator/datasets/pop24_L2_r12']
- `game.options.task_id`: ['task_0001', 'task_0002', 'task_0003', 'task_0004', 'task_0005']
- Expected total cost: 0.00 proxy_accounting_unit; conservative: 0.00 proxy_accounting_unit.
- Rough total runtime: 4185.0s.
- Preflight ID: `53a984c1cfc1bf8cf25f28124eb0819efb682f8277b042f00c774a44a1ba9d42` — pass this to `mas-cc experiment run --approve-preflight` to bind the launch to this estimate.

## Warnings

- Lower/expected scenarios use representative contexts; conservative demand uses each stage's maximum context and retry bound.
- Token counts use a deterministic regex estimate, not the provider tokenizer.
- Runtime is a rough planning estimate, not a service guarantee.
