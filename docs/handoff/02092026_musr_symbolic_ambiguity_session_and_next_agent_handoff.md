# MuSR Symbolic Ambiguity Session Summary and Next-Agent Start Guide

**Handoff date:** 2026-09-02  
**Completed study:** `musr_symbolic_ambiguity_calibration_01`  
**Repository:** MA-CC  
**Scientific status:** complete, **FAIL** under the preregistered behavioral gate

This document is the operational handoff for the implementation and execution
of:

```text
docs/tdd/features/games/01092026_MUSR_SYMBOLIC_AMBIGUITY_GENERATOR_HANDOFF.md
```

It is intended for a new agent who needs to understand what was done, how the
work was conducted, which standards are binding, and exactly where a new
revision should begin.

---

## 1. Read this first: the current stopping point

The requested symbolic ambiguity machinery was implemented, tested, and run
end to end. The symbolic construction gate succeeded, but the frozen benchmark
missed the behavioral Private ceiling by 0.83 percentage points:

| Condition | n | Truth rate | 95% Wilson CI | Gate |
|---|---:|---:|---:|---|
| Zero | 60 | 36.7% | 25.6–49.3% | pass: `<= 45%` |
| Private | 216 | 45.8% | 39.3–52.5% | **fail: `<= 45%`** |
| Full F9 | 60 | 80.0% | 68.2–88.2% | pass at boundary: `>= 80%` |

The final scientific decision is therefore:

```text
FAIL — symbolic ambiguity filtering alone is insufficient under the
prespecified behavioral gate.
```

Do **not** start by scaling the blackboard experiment. Do **not** tune or
replace individual frozen worlds after observing their model results. The next
scientific revision must predeclare a change to evidence generation or task
complexity and must use a new study/config/output identity, normally `_02`.

The authoritative scientific report is:

```text
results/studies/musr_symbolic_ambiguity_calibration_01/
    analysis/symbolic_ambiguity_calibration_report.md
```

The frozen task pack is:

```text
results/studies/musr_symbolic_ambiguity_calibration_01/accepted_tasks/
```

The study is complete. It does not need additional calls to repair or finish
it.

---

## 2. Mandatory reading order for a starting agent

Before editing or running anything, read these files in order:

1. `AGENTS.md` at the repository root.
2. `.codex/skills/ma-cc-study-workflow/SKILL.md` in full.
3. `docs/tdd/features/games/01092026_MUSR_SYMBOLIC_AMBIGUITY_GENERATOR_HANDOFF.md`.
4. This handoff.
5. `results/studies/musr_symbolic_ambiguity_calibration_01/README.md`.
6. `results/studies/musr_symbolic_ambiguity_calibration_01/preflight/report.md`.
7. `results/studies/musr_symbolic_ambiguity_calibration_01/analysis/symbolic_ambiguity_calibration_report.md`.
8. `configs/probes/musr_symbolic_ambiguity_calibration_01.yaml`.
9. The implementation map in section 7 below.

Also read the standardized study architecture before converting this probe
into a SLURM study or adding study-level aggregation:

```text
docs/tdd/features/orchestrator/
    22082026_TDD_standardized_study_submission_and_aggregation.md
docs/handoff/
    22082026_standardized_study_submission_and_aggregation_handoff.md
```

Then inspect the worktree:

```bash
git status --short
git diff --check
```

The current checkout is intentionally dirty and contains earlier MuSR and
blackboard work alongside this implementation. Those changes belong to the
user. Never reset, discard, or overwrite them merely to obtain a clean tree.

---

## 3. What was frozen before provider calls

The offline scan sampled 10,000 nine-value latent worlds from the generator's
actual prior:

```text
support = [1, 2, 3]
prior   = independent uniform categorical values
```

Each world contains:

```text
6 individual skill values
3 pairwise cooperation values
```

The existing exact allocation scorer was reused. No second scoring rule was
introduced.

For a private view with visible indices `I`, the implementation exactly
enumerates all completions and calculates:

```text
p_I(a)  = P(A* = a | z_I)
M_I     = max_a p_I(a)
Hbar_I  = [-sum_a p_I(a) log p_I(a)] / log(3)
```

Completion weights use the generator prior conditioned on:

```text
the visible values
unique exact optimum
the tested minimum score-margin rule
```

Tied and sub-margin completions are excluded before posterior
renormalization. There is no Monte Carlo approximation in the completion
calculation.

All subsets of sizes 2, 3, and 4 were evaluated. The preferred and fallback
criteria remained separate. The fallback threshold was not used.

The frozen rule selected from symbolic feasibility was:

```text
private breadth             k = 4
maximum predictability      M <= 0.45
minimum normalized entropy  Hbar >= 0.90
minimum exact score margin  Delta >= 2
gold balance                2 / 2 / 2 across ALLOCATION_0/1/2
population size             N = 12
minimum holders/value       2
```

For the selected margin-2, k-4 preferred gate:

```text
494 / 10,000 sampled candidates passed                 = 4.94%
494 / 3,255 structurally valid margin-2 candidates     = 15.18%
gold counts among passing sampled candidates           = 163 / 164 / 167
```

Six unique worlds were selected deterministically, exactly two per semantic
gold allocation. Every realized private view satisfies the frozen symbolic
threshold. Every latent value has multiple holders, and the population union
covers all nine values.

Natural-language evidence was generated only after this selection was sealed.

---

## 4. Frozen model, prompt, and evidence rules

These components were not behaviorally retuned:

```text
population game-playing model  gwdg/openai-gpt-oss-120b
local ballot prompt             P2
full packet                     F9
decoding temperature            1.0
population size                 12
```

Evidence generation used:

```text
provider/model  university / microsoft/gpt-5.6-terra
calls           54
branches/value  3
statements/card 2
tree depth      2
```

Terra was used only to produce indirect evidence for already accepted worlds.
It was never used as a population game-playing model.

F9 contains one deterministic first-sorted evidence card for each of the nine
latent values. Card choice is independent of gpt-oss performance.

Each private agent receives exactly six evidence cards drawn only from its
four assigned latent values. Extra cards add branches within an assigned value
rather than increasing latent breadth. Across the population, all 27 generated
cards are represented.

Behavioral prompts were checked to exclude:

```text
skill_matrix
cooperation_matrix
candidate_scores
gold_answer
hidden_claim
max_predictability
normalized_entropy
```

Answer letters were permuted while semantic scoring remained in the stable
`ALLOCATION_0/1/2` namespace.

---

## 5. Calls, timing, and observed usage

The planned logical work was:

```text
54  Terra evidence-generation calls
336 gpt-oss behavioral observations
390 completed logical calls total
```

There were 391 actual provider attempts:

```text
54  generation attempts
337 behavioral attempts
```

One gpt-oss response failed at the provider transport/schema layer. It was
retried with the same call identity, prompt, option mapping, and provider seed.
Incorrect model answers were not retried. Both the failed attempt and its
replacement remain in the append-only raw journal.

Observed token usage:

| Stage | Attempts/observations | Input tokens | Output tokens |
|---|---:|---:|---:|
| Evidence generation | 54 attempts | 25,567 | 27,341 |
| Behavioral validation | 337 attempts / 336 observations | 299,315 | 200,334 |

The generation budget guard recorded an observed cost of approximately
`0.379226 proxy_accounting_unit`. The retained gpt-oss pricing snapshot had
zero per-token proxy rates. These are proxy-accounting units, not a claim about
currency expenditure.

Timing at request concurrency 4:

```text
10,000-world symbolic scan      about 47 seconds
evidence generation             2 minutes 10 seconds
initial behavioral run          7 minutes 1 second
transport repair attempt        about 4 seconds
active scan/provider work       about 10 minutes
first provider call to repair   10 minutes 45 seconds
```

The longer final elapsed value includes the manual audit pause before the
single transport repair.

---

## 6. How the session was operated

The operating sequence was deliberately staged:

```text
read specifications and prior calibration
        |
        v
inspect dirty worktree without changing unrelated files
        |
        v
implement exact symbolic machinery and tests
        |
        v
run 10,000-world provider-free scan
        |
        v
freeze rule, worlds, assignments, hashes, and call budgets
        |
        v
write and inspect preflight + approval identity
        |
        v
generate evidence only for accepted worlds
        |
        v
freeze base tasks, F9 packets, and private distributions
        |
        v
render and inspect every behavioral prompt
        |
        v
run gpt-oss Zero / Private / F9 validation
        |
        v
repair transport failure without answer filtering
        |
        v
analyze, render figures, seal hashes, and report PASS/FAIL
```

Important operating behavior:

- Read-only inspection and provider-free tests came before external calls.
- The symbolic rule was selected before evidence generation and behavior.
- Provider calls required a matching preflight approval identity.
- Long-running work was monitored through append-only journals.
- Intermediate rates were treated as diagnostics only; the run continued
  unchanged to the prespecified sample size.
- A near miss was reported as FAIL rather than relaxed post hoc.
- No individual task, evidence card, or assignment was edited after seeing
  behavioral outcomes.
- Existing user changes in overlapping files were preserved.

---

## 7. Implementation map

### Exact generator and ambiguity core

| File | Responsibility |
|---|---|
| `src/mas_cc/musr_team_allocation_generator/latent_problem.py` | Authoritative support/prior, canonical nine-value conversion, exact problem construction, existing scorer reuse |
| `src/mas_cc/musr_team_allocation_generator/ambiguity.py` | Exact completion enumeration, conditional allocation posterior, `M_I`, normalized entropy, cached k-subset scans, ambiguity-qualified assignment selection |
| `src/mas_cc/musr_team_allocation_generator/generate.py` | Public option-row construction used by the post-filter task builder |
| `src/mas_cc/musr_team_allocation_generator/__init__.py` | Public exports for ambiguity and latent-world APIs |

### Calibration probe

| File | Responsibility |
|---|---|
| `configs/probes/musr_symbolic_ambiguity_calibration_01.yaml` | Scientific thresholds, seeds, frozen providers, sample sizes, concurrency, storage, and budget limits |
| `src/mas_cc/probes/musr_symbolic_ambiguity/config.py` | Strict config validation and call-count contracts |
| `src/mas_cc/probes/musr_symbolic_ambiguity/symbolic.py` | 10,000-world scan, subset archive, feasibility comparison, frozen-rule and balanced-world selection |
| `src/mas_cc/probes/musr_symbolic_ambiguity/tasks.py` | Post-selection Terra evidence generation, six-card private packets, base/distribution hashes |
| `src/mas_cc/probes/musr_symbolic_ambiguity/design.py` | Deterministic 336-observation Zero/Private/F9 call plan |
| `src/mas_cc/probes/musr_symbolic_ambiguity/analysis.py` | Observation tables, Wilson intervals, symbolic/empirical comparison, diagnostics, figures, final scientific report |
| `src/mas_cc/probes/musr_symbolic_ambiguity/runner.py` | Preflight, approval, staged execution, resume, sealing, usage accounting |
| `src/mas_cc/cli/probe.py` | Configured-probe CLI dispatch |

### Shared retry support and tests

| File | Responsibility |
|---|---|
| `src/mas_cc/probes/musr_prompt_solvability/execution.py` | Optional `retry_failed=True` support for terminal transport failures; default behavior remains unchanged for other probes |
| `tests/mas_cc/test_musr_symbolic_ambiguity.py` | Exact enumeration, tie handling, priors, entropy bounds, certainty, k scans, assignment thresholds/coverage, letter invariance, config freeze, task-pack and prompt-leakage tests |

The earlier addition of `candidate_score_terms()` in
`musr_team_allocation_generator/validate.py` is reused by the redistribution
calibration and remains relevant context.

---

## 8. Result and provenance layout

The complete result root is:

```text
results/studies/musr_symbolic_ambiguity_calibration_01/
```

Important contents:

```text
README.md
config.yaml
manifest.json

preflight/
    preflight.json
    preflight_id.txt
    pricing_snapshot.json
    behavioral_call_plan.json
    report.md

symbolic_scan/
    candidate_worlds.csv
    acceptance_summary.csv
    subset_metrics.parquet
    ambiguity_by_k.csv
    margin_ambiguity_tradeoff.csv
    frozen_selection.json

generation/
    raw_calls.jsonl

accepted_tasks/
    generation_manifest.json
    full_profile_packets.json
    private_assignments.json
    task_001/ ... task_006/
        base_task.json
        distribution_N12.json

behavioral_validation/
    raw_calls.jsonl
    observation_level_results.csv
    summary_by_task_condition.csv
    summary_pooled.csv

analysis/
    symbolic_ambiguity_calibration_report.md
    tables/
    figures/
```

The final manifest reports:

```text
status               complete
acceptance_decision  FAIL
artifact hashes      44 retained artifacts, all verified
```

The raw journals contain complete prompts and responses. The generation
journal contains hidden generation targets by design; the behavioral journal
must not contain hidden matrices or symbolic metrics in model-visible prompts.

The local `results/` tree may be ignored by Git. Do not assume the scientific
artifacts are versioned merely because they exist in this workspace. Before
moving machines or handing results to another repository clone, explicitly
archive or copy the result root through the project's approved data path.

---

## 9. Commands for inspection and reproduction

### Current local machine

The tested local interpreter is:

```text
/home/cesarali/miniconda3/envs/MA-CC/bin/python
```

Use the local environment only on this developer machine. Do not reproduce
this absolute path on another system.

Focused verification:

```bash
/home/cesarali/miniconda3/envs/MA-CC/bin/python -m pytest -q \
  tests/mas_cc/test_musr_symbolic_ambiguity.py \
  tests/mas_cc/test_musr_team_allocation_generator.py \
  tests/mas_cc/test_musr_team_allocation_validation_study.py \
  tests/mas_cc/test_musr_prompt_solvability.py \
  tests/mas_cc/test_musr_private_redistribution.py \
  tests/mas_cc/test_relational_musr_blackboard.py \
  tests/mas_cc/test_cli_and_inspection.py \
  tests/mas_cc/test_import_safety.py \
  tests/mas_cc/test_relational_blackboard.py \
  tests/mas_cc/test_musr_local_evidence_probe.py
```

Latest result:

```text
82 passed
```

Config parsing and retained-result checks do not require provider calls.

### Existing `_01` preflight command

The original command was:

```bash
/home/cesarali/miniconda3/envs/MA-CC/bin/python -m mas_cc.cli.main \
  probe preflight \
  --config configs/probes/musr_symbolic_ambiguity_calibration_01.yaml
```

Do not rerun this in the `_01` output directory merely to inspect it. Preflight
rewrites planned artifacts and the manifest. Read the retained preflight
instead.

### Existing `_01` run command

The authorized command was:

```bash
/home/cesarali/miniconda3/envs/MA-CC/bin/python -m mas_cc.cli.main \
  probe run \
  --config configs/probes/musr_symbolic_ambiguity_calibration_01.yaml \
  --approve-preflight \
  results/studies/musr_symbolic_ambiguity_calibration_01/preflight/preflight_id.txt
```

The current journal is complete, so a normal rerun resolves no outstanding
behavioral calls. Nonetheless, do not invoke live-provider commands casually:
they perform live pricing/account checks and future code changes could alter
resume behavior.

### Potsdam environment

On Potsdam, every Python, test, preflight, worker, aggregation, and
post-processing command must use:

```bash
/home/ojedamarin/.local/share/miniforge3/bin/conda run -n MA-CC python ...
/home/ojedamarin/.local/share/miniforge3/bin/conda run -n MA-CC mas-cc ...
```

Use `--live-stream` for long-running commands. Before a real submission, verify
that this environment imports `mas_cc`, `pandas`, and `pyarrow` from the
expected environment/repository.

Production Potsdam outputs must be beneath:

```text
/work/ojedamarin/Projects/LanguageGames/MA-CC/results
```

Never place a production cluster result under the home checkout.

---

## 10. Standards that must remain binding

### Scientific standards

1. Select symbolic worlds before generating language.
2. Freeze thresholds before behavioral outcomes.
3. Keep preferred and fallback criteria explicitly separate.
4. Reuse the exact scorer; never create a second scoring implementation.
5. State the latent prior and every conditioning rule.
6. Exclude and count tied/invalid completions explicitly.
7. Preserve semantic gold IDs independently of displayed answer letters.
8. Balance final semantic gold allocations before provider calls.
9. Validate the exact realized private views, not only population averages.
10. Keep dangerous-view analysis diagnostic; do not hand-edit tasks from it.
11. Never filter worlds or cards based on favorable gpt-oss answers unless a
    new behavioral selection rule is preregistered and every rejection is
    reported.
12. Apply the published PASS/FAIL gate literally, including at near-boundary
    results.

### Provider and preflight standards

1. Run provider-free scans and tests before external calls.
2. Put models, seeds, decoding, concurrency, budgets, and thresholds in YAML.
3. Report nominal, expected, and conservative call counts.
4. Report token and cost units and distinguish predictions, ceilings, and hard
   runtime guards.
5. Require an exact preflight approval identity for a paid/remote run.
6. Archive rendered prompts, raw responses, usage, request IDs, and failures.
7. Retry only transport/service failures under an explicit policy. Preserve
   the failed record and use the same scientific call identity and seed.
8. Never retry or discard an incorrect answer because it hurts the result.
9. Count unparsable completed responses conservatively unless a predeclared
   rule says otherwise.
10. Resume from append-only journals rather than deleting partial output.

### Repository and study standards

1. Preserve unrelated dirty-worktree changes.
2. Use `apply_patch` for hand-edited source/docs changes.
3. Use `rg`/`rg --files` for repository discovery.
4. Keep scientific design separate from scheduler topology.
5. Do not create a study-specific SLURM `.job` file. Use the generic launchers
   if a future version becomes a standard study.
6. Scheduler task IDs are execution provenance, never scientific coordinates.
7. Do not implement a replacement CMI/MI/bootstrap/null engine.
8. Keep canonical machine-readable tables alongside narrative reports.
9. Hash retained artifacts and verify hashes after the last report render.
10. A complete manifest and a PASS decision are different concepts. This
    study is operationally complete and scientifically FAIL.

### Communication standards for an agent

1. State assumptions and the current phase before a long operation.
2. Give progress updates during provider calls; do not leave a long run opaque.
3. Separate interim diagnostics from final prespecified results.
4. Lead the handoff with the scientific outcome, including failure.
5. Report blockers and baseline test failures with concrete evidence.
6. Do not imply that a close numerical result passes.

---

## 11. Test evidence and known repository baseline

The final affected/adjacent command above passed:

```text
82 passed
```

It covers the symbolic machinery, generator, validation-study adapter, prompt
calibration, redistribution, blackboard loading/prompts, CLI/inspection,
import safety, and local-evidence probe.

Additional checks passed:

```text
Python compileall on the changed generator/probe modules
git diff --check
44/44 final artifact hashes
manifest self-hash verification
336/336 terminal behavioral observations parse successfully
```

A repository-wide `pytest -q` was attempted and is not currently green for
unrelated baseline reasons. The dominant failures include:

```text
missing legacy configs such as:
  configs/runs/hidden_bench_grid.yaml
  configs/runs/hidden_bench_vanilla.yaml
  configs/runs/hidden_bench/hidden_bench_imitation_classical.yaml

an old game-registry assertion expecting fewer registered games

an unrelated NameError in:
  src/mas_cc/studies/episode_endpoints.py
  allow_truth_target is not defined
```

Do not “fix” those broad failures as part of the next MuSR revision unless the
user explicitly expands scope.

`ruff` was not installed in the tested environment; do not claim a Ruff pass.

---

## 12. Known operational nuances

### Token estimates

The retained `_01` generation preflight estimated 15,495 nominal input tokens
and 46,485 under its semantic-retry multiplier. Actual generation input was
25,567, so the conservative figure held but the nominal estimate was low. The
estimator was subsequently corrected to include the generator's forbidden
phrase block.

The behavioral compiled-prompt estimate was 239,043 tokens versus 299,315
provider-reported input tokens. This reflects tokenizer/accounting differences;
the hard configured behavioral input cap was 4,000,000.

Future preflights should report both the prompt-estimator prediction and the
hard provider budget rather than presenting the prediction as a guarantee.

### Retry semantics

`execute(..., retry_failed=True)` retries journal entries whose latest terminal
event is `call_failed`. It does not retry `call_finished` rows with an incorrect
semantic answer. Other probes retain the prior behavior because the option
defaults to `False`.

### Reanalysis and hashes

The final `probe run` path rebuilds analysis and reseals artifact hashes. If a
future agent edits analysis code and uses a lower-level analysis entry point,
it must reseal and reverify `manifest.json` afterward. Do not leave a rewritten
report with stale artifact hashes.

### Results durability

The study root is locally complete but may not be tracked by Git. Before
deleting the workspace, changing machines, or handing off to the cluster,
confirm an explicit durable copy exists.

---

## 13. When and how the next agent should start

### If the request is only to understand or report `_01`

Start immediately with read-only inspection of the retained report, summary
CSV files, raw journals, and manifest. Do not rerun provider calls.

Useful files:

```text
analysis/symbolic_ambiguity_calibration_report.md
behavioral_validation/summary_pooled.csv
behavioral_validation/summary_by_task_condition.csv
analysis/tables/symbolic_vs_empirical.csv
analysis/tables/dangerous_partial_views.csv
manifest.json
```

### If the request is to improve the benchmark

Start a new `_02` design only after the user approves the scientific change.
The next design should address evidence generation or task complexity because
the symbolic gate itself worked while empirical Private accuracy remained
slightly high and F9 only reached its minimum boundary.

A safe sequence is:

1. Diagnose `_01` from retained data without provider calls.
2. Write a new TDD/handoff that names exactly one primary construction change.
3. Predeclare thresholds and whether P2/F9/gpt-oss remain frozen.
4. Copy the config to a new `_02` filename and change the output directory to
   `musr_symbolic_ambiguity_calibration_02`.
5. Add or update tests before running a provider.
6. Run the symbolic scan into the new result root.
7. Inspect acceptance counts, gold balance, assignments, budgets, prompts, and
   the new preflight ID.
8. Obtain explicit authorization for the real run.
9. Generate language only for accepted worlds.
10. Run the full prespecified Zero/Private/F9 sample.
11. Report PASS or FAIL without editing worlds after the fact.

Candidate next hypotheses include increasing task complexity or changing how
indirect evidence maps latent strength to language. These are hypotheses, not
authorization. The next agent must not choose a modification merely because it
is convenient or likely to improve the observed number.

### If the request is to start the blackboard study

Stop and surface the `_01` FAIL decision. Proceed only if the user explicitly
overrides the gate or supplies a new validated task family. An override should
be documented as an override, not relabeled as a PASS.

### If the request is to move execution to Potsdam/SLURM

First convert the scientific design into repository-native study YAML and use
the generic study launchers. Verify the Potsdam environment, production result
root, cell/episode plan, provider RPM/concurrency, SLURM resources, and
preflight totals. Do not write a MuSR-specific `.job` file unless the scheduler
topology genuinely cannot be represented by the generic launchers.

---

## 14. Minimal start checklist

A starting agent should be able to copy this checklist into its working notes:

```text
[ ] Read AGENTS.md and ma-cc-study-workflow/SKILL.md.
[ ] Read the original symbolic handoff, this handoff, and the final report.
[ ] Run git status --short; preserve the dirty worktree.
[ ] Confirm `_01` manifest status=complete and decision=FAIL.
[ ] Confirm the requested work is review, `_02` design, explicit override, or cluster migration.
[ ] Do not make provider calls for inspection or reanalysis.
[ ] Do not mutate `_01`; use a new config/result identity for new science.
[ ] Keep P2/F9/gpt-oss frozen unless the new TDD explicitly changes them.
[ ] Run focused tests before preflight.
[ ] Inspect calls, tokens, cost units, concurrency, wall-time assumptions, and result root.
[ ] Require real-run authorization and matching preflight approval.
[ ] Archive every prompt/response/failure and preserve resume identities.
[ ] Apply the declared PASS/FAIL gate literally.
[ ] Verify final hashes and report no study-specific job or replacement CMI implementation.
```

---

## 15. Final handoff statement

The repository now has an exact, tested symbolic ambiguity implementation and
a complete calibration artifact family. The symbolic construction achieved
the intended mathematical ambiguity with k=4 and a decisive margin-2 full
world. The fixed gpt-oss behavioral experiment produced the desired broad
separation but missed the Private hard gate and only met the Full gate at its
minimum boundary.

The correct next action is not to reinterpret `_01`. Preserve it as a complete
negative result, use it to motivate a preregistered `_02` construction change,
and keep the same audit, preflight, immutability, and reporting standards.

No study-specific SLURM job or replacement CMI implementation was added in
this work.
