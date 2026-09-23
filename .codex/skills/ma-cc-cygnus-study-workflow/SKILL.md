---
name: ma-cc-cygnus-study-workflow
description: Preflight, submit, monitor, aggregate, relocate, or post-process MA-CC studies on the Cygnus SLURM cluster. Use for Cygnus or slurm-login execution; do not use for Potsdam.
---

# MA-CC Cygnus Study Workflow

Use this skill as the Cygnus compute-site overlay for
`ma-cc-study-workflow`. The general skill controls scientific study design and
provider-specific routing; this skill controls the Cygnus environment,
scheduler, launchers, paths, aggregation, and cluster-specific failure modes.

## Read the Cygnus runbook

Before planning or performing a Cygnus preflight, submission, monitoring,
aggregation, frozen-bundle relocation, cross-study aggregation, or
post-processing operation, read the complete checked-in runbook:

- `docs/handoff/cygnus-end-to-end.md`

Use that file directly rather than reconstructing commands from memory. Select
the procedure matching the requested operation and replace placeholders only
with paths and identifiers verified in the current environment. Its sections
cover prerequisites, study preflight and cell-array submission, monitoring,
both aggregation backends, incomplete-study handling, frozen-bundle
relocation/finalization, joint cross-study aggregation, publishing, optional
System One analysis, byte-identity regression checks, and known Cygnus
failures.

## Keep Cygnus separate from Potsdam

Treat `slurm-login`, `/shared/home/<user>`, `scripts/Cygnus/SLURM`, and the
Cygnus in-cluster gateway as Cygnus context. In that context:

- Do not use Potsdam's `/work/ojedamarin` result root, `/home/ojedamarin`
  repository or Conda path, `scripts/Potsdam/SLURM` launchers, or
  Potsdam-specific environment-loading rules.
- Use the Cygnus Python, repository, result roots, gateway settings, launchers,
  memory limits, and node choices specified by the runbook, after verifying
  that referenced paths exist for the current user.
- Keep compute-site selection separate from provider selection. If a config
  uses DeepInfra, also follow the general workflow's DeepInfra provider
  reference; do not treat DeepInfra as a cluster.
- If the target cluster is genuinely ambiguous, resolve it before any
  submission. Read-only inspection may continue meanwhile.

If this skill conflicts with a generic or Potsdam site example, this skill and
the Cygnus runbook govern Cygnus operations. Scientific invariants and
authorization requirements from `ma-cc-study-workflow` still apply.

## Launch and aggregation gate

Before a real `sbatch` or `mas_cc ... study submit`, verify the runbook's
prerequisites, resolved Python imports, absolute study and result paths,
preflight permission, execution mode, matching Cygnus launcher, resource
request, provider concurrency, and explicit user authorization. Submission
permission does not imply permission to publish results or invoke paid optional
analysis.

For aggregation, determine which runbook procedure applies before calling it:

- ordinary strict study aggregation in one allocation;
- the detached Slurm analysis graph;
- relocation and finalization of a frozen bundle;
- joint paired cross-study aggregation; or
- explicitly provisional incomplete-study aggregation.

Verify sealed inputs and the expected publish directory, monitor
`analysis/progress.json` when applicable, and use the runbook's byte-identity
gate for performance changes. Report the exact command, job ID, output path,
completion/validation state, and any provisional status in the handoff.
