# ASTRA task-003 truth-control Potsdam rho-3 study

This is the truth-aligned counterpart of
`astra_task003_false_control_30x30_potsdam_rho3`. It preserves the task,
provider/model, population, q=3 blackboard dynamics, persistence and budget
grids, repetitions, rounds, paired initializations, controller policy,
analysis recipe, retention, and execution topology. The only scientific change
is that the controller target resolves to the task's correct answer.

The design has 18 structural cells and 540 episodes. It uses three active
single-cell shards with 20 episode workers each, a shared concurrency ceiling
of 60, and a 600 RPM ceiling.

Use the generic study launcher; no study-specific SLURM job is required.
