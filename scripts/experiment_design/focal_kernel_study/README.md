# HiddenBench focal-kernel study

This directory contains the maintained generation, execution, progress-inspection, and
postprocessing code for the atomic-control calibration study. Generated prompts, responses,
and analysis artifacts live under `results/atomic_control_calibration/`, which is git-ignored.

Every atomic state deliberately mixes one canonical HiddenBench hidden fact with one accepted
repository paraphrase from a different evidence type. All six social-context buckets reuse the
same 100 underlying states.

## Generate and freeze the prompts

From the repository root:

```bash
conda activate MA-CC
python scripts/experiment_design/focal_kernel_study/generate_atomic_control_calibration.py
```

The default output is `results/atomic_control_calibration/`. The current generated dataset has:

- 10 tasks and 100 base states
- 6 matched buckets and 600 prompts
- 50 truth-aligned and 50 incorrect control targets
- 50 no-history and 50 one-step-history states
- dataset SHA-256 `cd16a40b9c47649ff6216abc95c40d3fd4e8478b5ccade58ae049c6f9c934802`

Workers must consume `results/atomic_control_calibration/frozen_prompts/`. They verify the
dataset and per-prompt hashes before making any provider request.

## Run one model

```bash
python scripts/experiment_design/focal_kernel_study/run_atomic_control_calibration.py \
  --input-dir results/atomic_control_calibration/frozen_prompts \
  --output-dir results/atomic_control_calibration/responses/qwen3_30b \
  --provider university \
  --model gwdg/qwen3-30b-a3b-instruct-2507 \
  --shard-index 0 --num-shards 1 --concurrency 10 \
  --temperature 0 --max-output-tokens 64 --invalid-response-retries 2
```

Launch one command per model with a unique output directory to run models in parallel. Each
process has its own `tqdm` bar and atomically updates `PROGRESS.json`. Completed prompt tuples
are skipped on resume.

For a one-shot overview of every active model, or a live view with `watch`:

```bash
python scripts/experiment_design/focal_kernel_study/inspect_atomic_control_progress.py \
  --responses-dir results/atomic_control_calibration/responses

watch -n 2 python scripts/experiment_design/focal_kernel_study/inspect_atomic_control_progress.py \
  --responses-dir results/atomic_control_calibration/responses
```

## Analyze completed models

```bash
python scripts/experiment_design/focal_kernel_study/analyze_atomic_control_calibration.py \
  --responses-dir results/atomic_control_calibration/responses/qwen3_30b \
                  results/atomic_control_calibration/responses/another_model \
  --output-dir results/atomic_control_calibration/analysis
```

Analysis remains separate by provider, model, bucket, and matched state. It may be rerun over
any subset of completed model directories as new workers finish.
