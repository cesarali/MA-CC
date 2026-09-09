# ASTRA task-003 false-control Potsdam rho-3 study

This is a fresh Potsdam-provider replication of
`astra_task003_false_control_30x30`, reduced to persistence values 0.70, 0.85,
and 1.00. The scientific runtime, task, budgets, repetitions, controller, and
analysis recipe are otherwise unchanged. Provider/model is
`gwdg/openai-gpt-oss-120b`.

The design has 18 structural cells and 540 episodes. It uses three active
single-cell shards with 20 episode workers each, a shared concurrency ceiling
of 60, and a 600 RPM ceiling. Results and paired initializations use new paths
under `/work`; no DeepInfra results are reused.

Use the generic study launcher; no study-specific SLURM job is required.
