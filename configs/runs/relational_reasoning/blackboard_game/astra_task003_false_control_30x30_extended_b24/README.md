# ASTRA task-003 false-control high-budget extension

Prepared only; this folder is not a standalone replacement study and has not
been submitted. It is the complete desired target for `mas-cc study extend`.

The target contains:

- the unchanged original block: rho = 0.70, 0.775, 0.85, 0.925, 1.00 and
  b = 3, 6, 9, 12;
- the added matched-rho block: rho = 0.70, 0.85, 1.00 and b = 18, 24;
- 30 repetitions per cell;
- 26 total cells and 780 target episodes.

Only the six high-budget cells (180 episodes) are scientifically new. The
extension protocol must reuse valid observations already present in
`astra_task003_false_control_30x30` and plan only missing episode identities.

Preview after the active source run is quiescent:

```bash
/home/ojedamarin/.local/share/miniforge3/bin/conda run -n MA-CC \
  mas-cc study extend \
  --study-dir /work/ojedamarin/Projects/LanguageGames/MA-CC/results/studies/astra_task003_false_control_30x30 \
  --config-dir configs/runs/relational_reasoning/blackboard_game/astra_task003_false_control_30x30_extended_b24 \
  --dry-run
```

Do not run the non-dry command while the original study still has active
workers. No study-specific SLURM script is required.
