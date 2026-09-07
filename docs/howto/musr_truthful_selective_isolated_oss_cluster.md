# MuSR isolated OSS cluster launch

This launch uses the frozen package prepared locally. It does not regenerate Terra evidence and does not run population dynamics.

## Paths on the cluster

Repository:

`/home/ojedamarin/Projects/LanguageGames/MA-CC`

Required result root:

`/work/ojedamarin/Projects/LanguageGames/MA-CC/results/studies/musr_truthful_selective_isolated_oss_01`

Dedicated environment:

`/home/ojedamarin/.local/share/miniforge3/bin/conda run -n MA-CC`

## Transfer and verification

Copy `preparation/transfer_bundle.tar.gz` to the repository, extract it into the result root, then verify every entry in `preparation/checksum_manifest.json`. Do not transfer credentials. The package contains only task IDs 001–003 with candidates 42, 237, and 130.

Run the offline preflight on the cluster before any model call:

`/home/ojedamarin/.local/share/miniforge3/bin/conda run -n MA-CC --live-stream mas-cc probe preflight --config configs/probes/musr_truthful_selective_isolated_oss_01.yaml --output-dir /work/ojedamarin/Projects/LanguageGames/MA-CC/results/studies/musr_truthful_selective_isolated_oss_01`

## Smoke

Submit the generic probe launcher. Add the site's required account/partition flags; none are invented here. This remote API job needs no GPU.

`sbatch --cpus-per-task=4 --mem=8G --time=04:00:00 --output=<RESULT_ROOT>/logs/smoke-%j.out --error=<RESULT_ROOT>/logs/smoke-%j.err scripts/Potsdam/SLURM/run_probe.job configs/probes/musr_truthful_selective_isolated_oss_01.yaml <RESULT_ROOT> <RESULT_ROOT>/preparation/preflight_id.txt smoke smoke`

Inspect job completion and the four atomic checkpoint files. A scientific answer pattern is not required for smoke success.

## Full run and resume

`sbatch --cpus-per-task=4 --mem=8G --time=04:00:00 --output=<RESULT_ROOT>/logs/full-%j.out --error=<RESULT_ROOT>/logs/full-%j.err scripts/Potsdam/SLURM/run_probe.job configs/probes/musr_truthful_selective_isolated_oss_01.yaml <RESULT_ROOT> <RESULT_ROOT>/preparation/preflight_id.txt full cluster`

Use exactly the same command to resume. Completed smoke and full request IDs are skipped. Do not launch two writers against the same result root.

The cluster profile starts at 60 in-flight requests and 600 requests per minute. This matches the successfully completed Potsdam run on 2026-09-06 and remains below the live 2,000 RPM account limit observed during the 2026-09-07 preflight. Reduce these values before launch if other jobs are sharing the account.

## Aggregate

`/home/ojedamarin/.local/share/miniforge3/bin/conda run -n MA-CC --live-stream mas-cc probe analyze --config configs/probes/musr_truthful_selective_isolated_oss_01.yaml --output-dir <RESULT_ROOT>`

The report is `<RESULT_ROOT>/analysis/isolated_oss_report.md`.
