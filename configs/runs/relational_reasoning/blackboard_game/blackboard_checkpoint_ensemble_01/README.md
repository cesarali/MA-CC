# Blackboard checkpoint ensemble 01

This folder is the runnable Phase 2 design for task `task_003`. The four
ordinary configs fix `(q, rho)` at `(3, .70)`, `(3, 1.00)`, `(12, .70)`, and
`(12, 1.00)`. Each of their 120 scheduler episodes is one independent parent
bundle: two uncontrolled preparation rounds, one immutable checkpoint, and the
nine ten-round continuations declared in `ensemble`.

Use the generic workflow only:

```bash
mas-cc study preflight --config-dir . --output-dir ./preflight
mas-cc study submit --config-dir . --output-dir ./submission
```

The second command is documentation, not authorization. Do not submit the
pilot or main design without explicit approval. On Potsdam, follow the
repository `AGENTS.md` Conda and `/work` rules.

Aggregation reads `analysis.yaml`, validates complete parent blocks, writes
canonical Parquet tables, and computes controlled-minus-none responses,
assigned-policy CMI using the established estimator engine, grouped
cross-fitted classifier scores, and whole-parent label-swap diagnostics.

The frozen task dataset and any initialization inputs must be staged at their
declared paths before execution. Config resolution and static preflight do not
send provider requests.
