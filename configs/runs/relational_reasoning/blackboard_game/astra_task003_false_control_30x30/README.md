# ASTRA task-003 false-control study

Prepared configuration only. Do not submit until the cluster agent completes preflight and the separate Phase 1 live smoke.

Design:

- one false-control arm;
- MuSR task 003, candidate 130;
- `gwdg/openai-gpt-oss-120b` for participants and controller;
- 30 population rounds and 30 repetitions;
- persistence values: 0.70, 0.775, 0.85, 0.925, 1.00;
- report budgets: 3, 6, 9, 12;
- 20 cells and 600 episodes;
- LLM-selected communication after the unchanged randomized binary gate.

The cluster agent must:

1. verify the three task hashes recorded in `false_control_llm.yaml`;
2. run `mas-cc study preflight` and adjust only execution limits if required;
3. materialize the 30 paired initialization artifacts with `mas-cc study initialize`;
4. complete the separate `astra_phase1_task003_smoke` study first;
5. inspect controller prompts, choices, fallback use, posts, and token accounting;
6. request explicit authorization before submitting this full study;
7. submit with `mas-cc study submit --config-dir configs/runs/relational_reasoning/blackboard_game/astra_task003_false_control_30x30`;
8. monitor the returned SLURM job and aggregate strictly after all cells seal.

Strict aggregation now also produces the Phase 2 propensity-weighted response
at lags 1–3, propensity/support diagnostics, communication funnel, operational
response-per-cost summaries, response-cost frontier, and their plots. These are
computed offline from canonical Parquet records and make no provider calls.

```bash
/home/ojedamarin/.local/share/miniforge3/bin/conda run -n MA-CC --live-stream \
  mas-cc study aggregate \
  --study-dir /work/ojedamarin/Projects/LanguageGames/MA-CC/results/studies/astra_task003_false_control_30x30
```

Check `analysis/validation.json` first, then the Phase 2 tables under
`analysis/tables/` and the lag/frontier plots under `analysis/plots/`.

Use the generic study launcher. Do not create a study-specific SLURM job.
