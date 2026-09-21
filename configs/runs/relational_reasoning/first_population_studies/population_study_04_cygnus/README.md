# Study 04, q_c = 6 row — Cygnus variant

The Potsdam row lives in `../population_study_04/` and is Cesar's design of record.
This folder is the Cygnus launch of the same row with the settings Allen approved on
2026-09-12: the task grid widened from 2 to 20 tasks (dataset `pop24_L2_r06_n40`, the
first 20 of the 26 tasks whose target index 2 is verified incorrect), `invalid_response_retries` 5,
and the provider block pointed at the cluster gateway (`LLM_BASE` / `LLM_KEY`). Repetitions are
2 (smoke); Cesar's row design is 30. `*_SMOKE.yaml` trims the grid to the two original tasks.

```bash
mkdir -p results/slurm-logs
mas-cc study preflight --config-dir configs/runs/relational_reasoning/first_population_studies/population_study_04_cygnus --output-dir results/inspection/study04_qc06_cygnus
mas-cc study submit --config-dir configs/runs/relational_reasoning/first_population_studies/population_study_04_cygnus --job-script scripts/Cygnus/SLURM/run_config_array.job
mas-cc study aggregate --study-dir results/studies/study04_qc06_cygnus
```

Provider components for the other gateway models are in `configs/components/cygnus/`.
