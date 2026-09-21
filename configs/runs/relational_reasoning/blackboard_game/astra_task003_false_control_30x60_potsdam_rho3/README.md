# ASTRA task-003 false-control Potsdam rho-3 extension to 60 episodes

This folder is the complete desired target for extending the existing
`astra_task003_false_control_30x30_potsdam_rho3` study lineage from 30 to 60
episodes per scientific cell. It is not a standalone replacement study.

The scientific design remains unchanged: 18 cells spanning persistence values
0.70, 0.85, and 1.00 and intervention budgets 3, 6, 9, 12, 18, and 24. The
target contains 1,080 episodes. The extension planner must reuse the 540 valid
existing episodes and schedule only repetition indices 30 through 59, for 540
new episodes.

The `experiment.metadata.repetitions` field remains at 30 because it is part of
the indexed legacy protocol fingerprint. The authoritative target count is
`execution.repetitions: 60`; changing the duplicated legacy metadata field
would incorrectly classify otherwise identical cells as a new protocol.

Preview with:

```bash
/home/ojedamarin/.local/share/miniforge3/bin/conda run -n MA-CC \
  mas-cc study extend \
  --study-dir /work/ojedamarin/Projects/LanguageGames/MA-CC/results/studies/astra_task003_false_control_30x30_potsdam_rho3 \
  --config-dir configs/runs/relational_reasoning/blackboard_game/astra_task003_false_control_30x60_potsdam_rho3 \
  --dry-run
```

Use the generic study cell-array launcher; no study-specific SLURM job is
required.
