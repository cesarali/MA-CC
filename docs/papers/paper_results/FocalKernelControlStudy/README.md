# Atomic-control calibration: paper reproducibility package

This package reproduces the reported tables and plots for six LLMs evaluated on
600 frozen prompts: 10 HiddenBench tasks × 10 matched base states × 6 social
prompt families. The frozen dataset SHA-256 is
`cd16a40b9c47649ff6216abc95c40d3fd4e8478b5ccade58ae049c6f9c934802`.
The underlying repository commit was
`3515435b95aa3a70f86a349f20c1b2f71bbe4d57`; the exact study and recovery
scripts used after that commit are included in `code/`.

## Contents

- `analysis/`: publication tables, plot-source CSVs, Markdown summaries, and PNG
  figures. `effective_valid_responses.csv` is the analysis-ready observation
  table used to reproduce every reported rate and plot.
- `code/`: generation, execution, recovery, inspection, and analysis scripts,
  plus the submitted Slurm jobs.
- `prompt_examples/`: one exact matched example for each of the six prompt
  families, all using `state_0001` so only the social framing changes.
- `provenance/`: environment specifications, dataset and prompt manifests,
  source data, base states, and per-family manifests.
- `METRICS_AND_METHODS.md`: definitions, denominators, recovery policy, and
  interpretation caveats.
- `PROMPT_FAMILY_EXAMPLES.md`: family definitions and links to the exact examples.

## Reproduce the analysis

From a checkout containing the original response directories:

```bash
python code/analyze_atomic_control_calibration.py \
  --responses-dir results/atomic_control_calibration/responses \
                  results/atomic_control_calibration/recovery_responses \
  --output-dir reproduced_analysis \
  --bootstrap-repetitions 2000 \
  --seed 20260814
```

To restyle figures without raw provider responses, use the CSVs in `analysis/`:

- `effective_valid_responses.csv`: one row per accepted model response;
- `model_metrics.csv`: model-level rates and coverage;
- `model_task_metrics.csv`: model × task rates;
- `model_task_bucket_metrics.csv`: model × task × prompt-family rates;
- `controllability_metrics.csv`: model × prompt-family rates and bootstrap CIs;
- paired-difference CSVs: matched model and prompt-family contrasts.

## Coverage

There are 3,226 accepted observations out of 3,600 planned. Coverage is complete
for GPT-OSS 120B and GPT-5 Mini, 99.8% for GPT-4o, 99.7% for Qwen3 30B A3B,
97.3% for Gemma4 31B, and 40.8% for Kimi K2.6. Comparisons involving Kimi may
be affected by response-selection bias and must report coverage.

## Task names

| Task ID | Exact task name | Plot label |
|---:|---|---|
| 1 | `evacuation_west_city` | Evacuation |
| 4 | `toma_butera_2009` | Traffic accident |
| 9 | `critical_hospital_transfer` | Hospital transfer |
| 13 | `Laboratory Theft Deduction` | Laboratory theft |
| 16 | `Crisis Backup Decision` | Backup datacenter |
| 23 | `community_banquet_venue_decision` | Banquet venue |
| 27 | `research_station_site_selection` | Research station |
| 30 | `the_lead_investor_decision` | Lead investor |
| 36 | `datacenter_emergency_migration` | Datacenter migration |
| 41 | `Space Evacuation Decision` | Space evacuation |