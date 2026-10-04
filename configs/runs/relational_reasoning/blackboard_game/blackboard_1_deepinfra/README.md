# Blackboard Population Study 01 (`blackboard_1_deepinfra`)

This folder defines the first population study using the frozen MuSR Team Allocation blackboard protocol.

## Design

- Task: `task_001`
- Model: `gwdg/openai-gpt-oss-120b`
- Population: 24 ordinary agents
- Rounds: 30
- Public-message sample size `q`: 1
- Controller sensor sample size `q_c`: 12
- Board lifetime: 1 round
- Controller policy: frozen soft-target policy with `beta=4`, `theta=0.5`
- Controlled timing: night sensing, dawn actuation, autonomous day
- Persistence: `rho = {0.74, 0.85, 1.00}`
- Dawn directive budgets: `b = {3, 6, 12, 24}`
- Repetitions: 10 per structural cell

The three configs expand to 3 no-control cells, 12 truth-control cells, and 12 false-control cells: 27 cells and 270 episodes in total. The false target is the first canonical non-gold semantic option, `ALLOCATION_1`; truth is `ALLOCATION_0`. Display letters are shuffled per prompt and do not define either target.

All arms use `paired_local_vote`, meaning one saved initial vote state per repetition is reused across the 27 structural conditions. This removes accidental differences in starting states while allowing later model responses and trajectories to differ naturally. Generate these ten initialization artifacts before submission.

Invalid model responses retain the existing correction path. Each logical decision may be retried up to three times, and failed plus corrected attempts remain in the audit records. A decision that never becomes valid fails rather than silently creating a default vote.

## Preparation and execution

On Amarel, use `conda run -p /scratch/df630/conda_envs/MA-CC` with `PYTHONPATH=/scratch/df630/MA-CC/src` before each command.

1. Provider-free design and cost preflight:

   `mas-cc study preflight --config-dir configs/runs/relational_reasoning/blackboard_game/blackboard_1_deepinfra --output-dir /scratch/df630/MA-CC-results/inspection/musr_blackboard_population_01_deepinfra_preflight`

2. Generate matched initial states. This step makes provider calls but does not run the dynamics:

   `mas-cc study initialize --config-dir configs/runs/relational_reasoning/blackboard_game/blackboard_1_deepinfra --output-dir /scratch/df630/MA-CC-results/studies/musr_blackboard_population_01_deepinfra_initializations`

3. Full study submission:

   `mas-cc study submit --config-dir configs/runs/relational_reasoning/blackboard_game/blackboard_1_deepinfra`

4. Resume an interrupted study by running the same submit command again. Existing completed episodes are validated and skipped.

5. Analysis and report generation:

   `mas-cc study aggregate --study-dir /scratch/df630/MA-CC-results/studies/musr_blackboard_population_01_deepinfra`

No study-specific SLURM job is used. `execution.mode: auto` selects the generic cell-array launcher.
