# ASTRA Phase 1 task-003 smoke

This is a small matched DeepInfra live smoke. It compares silence, the existing algorithmic communication chooser, and the optional LLM communication chooser. All arms use `deepseek-ai/DeepSeek-V4-Flash`, the same MuSR task, participant prompt, decoding settings, root seed, and paired initialization artifact.

The smoke is operational. Do not interpret one episode per arm as a scientific effect estimate, and do not launch a full study from this folder.

## Frozen task inputs

- Task: `task_003`, candidate 130
- Gold allocation: `ALLOCATION_0`
- Controller target: `ALLOCATION_2`
- `task.json`: `1ec61ccc9578e2bcbd6a027035231dcd1ee39d44892d95b8b6ae2e9f18fdfbdb`
- `private/N24_assignment.json`: `aad3e64f7a713534667e04984c9911423b94c96381ec9b8ff21f90714aaea0a4`
- `controller/ranked_fact_pool.json`: `b90a3267c26212194e916777795db0751008d8802d45e35a67088fb8a0ce49d0`

## Potsdam commands

Use the repository's required `MA-CC` Conda environment.

1. Verify the task hashes and run the offline study preflight:

```bash
cd /home/ojedamarin/Projects/LanguageGames/MA-CC
sha256sum \
  results/studies/musr_truthful_selective_task_calibration_01/tasks/task_003/task.json \
  results/studies/musr_truthful_selective_task_calibration_01/tasks/task_003/private/N24_assignment.json \
  results/studies/musr_truthful_selective_task_calibration_01/tasks/task_003/controller/ranked_fact_pool.json
/home/ojedamarin/.local/share/miniforge3/bin/conda run -n MA-CC --live-stream \
  mas-cc study preflight \
  --config-dir configs/runs/relational_reasoning/blackboard_game/astra_phase1_task003_smoke \
  --output-dir /work/ojedamarin/Projects/LanguageGames/MA-CC/results/inspection/astra_phase1_task003_smoke
```

2. Materialize the one shared starting snapshot:

```bash
/home/ojedamarin/.local/share/miniforge3/bin/conda run -n MA-CC --live-stream \
  mas-cc study initialize \
  --config-dir configs/runs/relational_reasoning/blackboard_game/astra_phase1_task003_smoke \
  --output-dir /work/ojedamarin/Projects/LanguageGames/MA-CC/results/studies/astra_phase1_task003_smoke_initializations
```

3. Submit the smoke through the generic study launcher:

```bash
/home/ojedamarin/.local/share/miniforge3/bin/conda run -n MA-CC --live-stream \
  mas-cc study submit \
  --config-dir configs/runs/relational_reasoning/blackboard_game/astra_phase1_task003_smoke
```

Inspect the returned job with the site's normal `squeue` and `sacct` commands. A returned job ID is not proof of completion.

4. After every arm is complete, aggregate strictly:

```bash
/home/ojedamarin/.local/share/miniforge3/bin/conda run -n MA-CC --live-stream \
  mas-cc study aggregate \
  --study-dir /work/ojedamarin/Projects/LanguageGames/MA-CC/results/studies/astra_phase1_task003_smoke
```

Do not use `--allow-incomplete` for the final smoke assessment. Do not submit a larger study yet.
