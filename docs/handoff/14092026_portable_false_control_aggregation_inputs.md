# Portable false-control inputs — 2026-09-14

Scope: **DeepInfra false and Potsdam false only**. This is simulation-input
data for reaggregation, not the finished results package. No LLM calls needed.

## Transfer from Potsdam

Copy both files to the destination cluster's scratch storage:

```text
/work/ojedamarin/Projects/LanguageGames/MA-CC/results/exports/false_control_aggregation_inputs_20260914.zip
/work/ojedamarin/Projects/LanguageGames/MA-CC/results/exports/false_control_aggregation_inputs_20260914.zip.sha256
```

ZIP size: 118,936,333 bytes (~119 MB). It contains canonical Parquet data,
validation/provenance, analysis recipes, task-003 epistemic facts, the source
snapshot with the semantic-label fix, and a portable aggregation helper.
No credentials, provider logs, full run trees, or resampling caches are included.

## Destination layout

Keep extracted inputs together and write results to separate, fresh directories:

```text
<scratch>/ma-cc-transfer/
  false_control_aggregation_inputs_20260914.zip
  false_control_aggregation_inputs_20260914.zip.sha256
  false_control_aggregation_inputs_20260914/
    README.md
    aggregate_portable.py
    bundle_manifest.json
    environment_versions.json
    code/                 # bundled fixed implementation
    tasks/task_003/        # symbolic facts for epistemic analysis
    studies/              # two studies: input/, analysis.yaml, provenance/
  outputs/
    deepinfra_false/       # created by the helper; do not pre-create
    potsdam_false/         # created by the helper; do not pre-create
```

From `<scratch>/ma-cc-transfer/`:

```bash
sha256sum -c false_control_aggregation_inputs_20260914.zip.sha256
unzip false_control_aggregation_inputs_20260914.zip
cd false_control_aggregation_inputs_20260914
python aggregate_portable.py --check
```

Use the destination machine's existing Python 3.11+ project environment;
dependencies/observed versions are included. Do not use the Potsdam-specific
Conda path elsewhere. The helper loads the bundled source and relocates recipe,
task, and discovery paths; historical Potsdam paths in provenance stay inert.

## Aggregation order and outputs

Obtain a normal compute allocation on the destination cluster, then run
**DeepInfra false first, Potsdam false second**. Suggested allocation per study:
4 CPUs, 16 GB RAM, 6 hours; not a runtime guarantee. Do not run on a login node.
From the extracted bundle directory:

```bash
python aggregate_portable.py astra_task003_false_control_q12_deepinfra_rho3 --output-dir ../outputs/deepinfra_false --workers 4
python aggregate_portable.py astra_task003_false_control_30x30_potsdam_rho3 --output-dir ../outputs/potsdam_false --workers 4
```

The helper runs locally inside that allocation (`backend=local`), without
submitting to Potsdam. It rejects existing output directories. To inspect
relocation without computation, use `--prepare-only` with a separate test path.

Each final ZIP appears at:
`<output-dir>/analysis/<original-study-name>_analysis.zip`.
Live progress: `<output-dir>/analysis/progress.json`.

DeepInfra false has **409/450 completed episodes** and remains provisional;
the helper explicitly enables incomplete aggregation only for that study.
Potsdam false has **540/540** and uses strict validation. Bootstrap settings
remain 1000 resamples, 95% confidence, seed 1; epistemic and calibration outputs
are enabled. Information resampling and finalizer work recompute from inputs.

Archive integrity, all included file hashes, canonical schemas, symbolic task
loading, bundled imports, and relocation were checked on a separate extraction.
Full aggregation was not rerun as part of the transfer check.

Exporting did **not** cancel the Potsdam retries 1878067/1878068. Avoid counting
duplicate analyses as independent data. Retry context:
[false-control finalizer handoff](14092026_false_control_finalizer_retry.md).
