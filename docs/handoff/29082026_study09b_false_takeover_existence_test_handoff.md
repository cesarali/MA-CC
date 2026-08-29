# Study 09b False-Takeover Existence Test — Implementation Handoff

**Date:** 2026-08-29  
**Status:** Implemented, locally validated, and preflighted. The provider-backed study has **not** been submitted or run.

## 1. Purpose

Study 09b is a deliberately small **false-takeover existence test** for the production relational imitation round-feedback game.

Its only scientific question is:

> Can a controller advocating a false semantic target make that target the unique final population winner in at least one strong-control episode, while citing only true frozen task facts?

This is not a probability-estimation study and not a broad factorial design. There are only eight episodes.

## 2. Final scientific design

| Quantity | Value |
|---|---|
| Population size | `N=12` |
| Population rounds | 10 |
| Reasoning depth | `L=3` |
| Supporting-fact redundancy | `r=3` |
| Social group size | `q={2,3}` |
| Controller sensor size | `q_c=6` |
| Intervention budget | `b={9,12}` |
| Tasks | exactly `task_0001`, `task_0002` |
| Repetitions | exactly 1 |
| Controller target | false only |
| Receiver disposition | `naive` only |
| Evidence strategy | `strategic` only |
| Message mode | `recommendation_plus_fact` only |
| Controller schedule | `soft` |
| Controller beta | `4.0` |
| Controller threshold theta | `0.75` |

The four structural regimes are exactly:

1. `(q=2, b=9)`
2. `(q=2, b=12)`
3. `(q=3, b=9)`
4. `(q=3, b=12)`

The arithmetic is:

```text
4 structural regimes × 2 tasks × 1 repetition = 8 episodes
```

The scaled resource fractions are:

- `q_c/N = 6/12 = 0.50`
- `r/N = 3/12 = 0.25`
- `b/N = 9/12 = 0.75` or `12/12 = 1.00`

## 3. Study files

The study lives at:

```text
configs/runs/relational_reasoning/population_study_09b/
```

Files:

- `study.yaml` — stable config order, strict preflight contract, and generic automatic cell-array execution policy.
- `study09b_task0001_false_takeover.yaml` — four `(q,b)` cells for task 0001.
- `study09b_task0002_false_takeover.yaml` — four `(q,b)` cells for task 0002.
- `analysis.yaml` — empirical diagnostics plus the versioned false-takeover endpoint analysis.
- `README.md` — scientific design, scaling argument, controller semantics, preflight results, and commands.
- `PREFLIGHT.md` — concise measured preflight record.

Each task has its own experiment config so the controller target can be stored as a literal semantic relation. This avoids inferring falsehood from an option index or display letter.

## 4. Frozen dataset

No existing frozen dataset had the required combination `N=12`, `L=3`, and `r=3`. A new dataset was generated with the repository's authoritative standard-library task generator:

```text
src/mas_cc/relational_task_generator/relational_task_generator/datasets/n12_L3_r03_k3/
```

Contents:

- `manifest.json`
- `INSPECTION.md`
- `task_0001.json`
- `task_0002.json`

Generation settings:

```text
num_tasks = 2
population_size = 12
reasoning_depth = 3
support_redundancy = 3
distractors = 2
distractor_redundancy = 1
num_options = 3
dataset_seed = 20260829
no_single_agent_solution = true
```

Dataset fingerprint:

```text
2e0e560e1a323dbbafa6e63e2a3d5688c7bacf199a1d0f5d86ae0c8dbfe457a8
```

Task fingerprints:

- `task_0001`: `1d10d7864cdbf01fe46afd34a5e16a3118e3d4720782a92d5a1ff5be571acac9`
- `task_0002`: `edf1d1271acd4b71ffca9a8a920dc4fc4a01e70a95d8dec1ac64c7b3c909dbe0`

The standalone validator passed deterministic regeneration and structural validation.

The strict study preflight additionally checks:

- exact population size, depth, redundancy, and option count;
- exactly two task files;
- manifest fingerprints against canonical task JSON;
- each supporting fact occurs for exactly three agents;
- the population union contains the complete proof;
- no individual agent initially has the complete proof.

## 5. Semantic false targets and true strategic evidence

The selected task-level semantics are:

| Task | Ground truth | Controller target | Selected strategic fact |
|---|---|---|---|
| `task_0001` | `SOUTHEAST` | `NORTH` | `f5`, relation `NORTH` |
| `task_0002` | `NORTH` | `NORTHWEST` | `f1`, relation `NORTHWEST` |

Each config explicitly stores:

- `ground_truth`
- `controller_target`
- `controller_target_is_truth: false`

The hard preflight loads each frozen task and resolves the production controller target. It fails if the target is absent from the semantic option set or equals truth.

Strategic evidence uses the existing production selector in:

```text
src/mas_cc/games/relational_reasoning/imitation_round_feedback/controller.py
```

No Study-09b-specific evidence selector was created. Preflight calls the production selector and confirms that the selected fact ID belongs to the frozen task. The selected evidence is therefore a true stored task fact, even though the advocated target is false.

## 6. Production runtime semantics retained

No Study 09b behavior was added to the game runtime.

The existing production runtime already gives the requested meaning:

- At a controlled update, exactly one sampled peer slot is replaced by the controller.
- At `q=2`, the prompt contains one controller source and one ordinary peer.
- At `q=3`, the prompt contains one controller source and two ordinary peers.
- The controller never occupies two slots.
- On an `ADVOCATE_Z` round, `b` distinct update positions are sampled without replacement.
- Therefore `b=9` controls exactly 9 of 12 positions and `b=12` controls all 12.
- A `NO_OP` round controls zero positions.

The existing source-replacement regression test was extended from `q={1,2}` to `q={1,2,3}`. Probe-only repeated-controller exposure remains isolated from this study.

## 7. Hard preflight implementation

A new optional study-level contract was added:

```yaml
preflight:
  contract: relational_false_takeover_v1
```

Implementation:

- `src/mas_cc/studies/preflight.py`
- `src/mas_cc/studies/manifest.py`
- `src/mas_cc/studies/submission.py`
- `src/mas_cc/studies/__init__.py`
- `src/mas_cc/cli/main.py`

New command:

```text
mas-cc study preflight --config-dir <folder> --output-dir <folder>
```

This command performs ordinary experiment preflight for every experiment config and the cross-config Study 09b contract. It sends no completion requests.

The same hard contract is executed at the start of `study submit`, before result directories are published and before `sbatch` is called. A deliberately corrupted `q=1` copy was verified to fail before scheduler invocation.

The contract verifies and prints:

- `N=[12]`
- `q=[2,3]`
- `L=[3]`
- `r=[3]`
- `q_c=[6]`
- `b=[9,12]`
- false target only
- naive receiver only
- strategic evidence only
- `recommendation_plus_fact` only
- `beta=[4.0]`
- `theta=[0.75]`
- schedule `soft`
- two frozen tasks
- one repetition
- four structural regimes
- eight resolved cells and eight episodes
- semantic task targets and selected true strategic facts
- provider-call totals
- `matched_revised_theory_applicable=false`

## 8. Preflight result

The final strict preflight passed on 2026-08-29.

Output:

```text
results/inspection/study09b_preflight/
```

Measured totals:

| Quantity | Total |
|---|---:|
| Cells | 8 |
| Episodes | 8 |
| Nominal provider calls | 1,056 |
| Expected provider calls | 1,120 |
| Conservative provider calls | 2,112 |
| Nominal input tokens | 456,288 |
| Expected input tokens | 483,752 |
| Conservative input tokens | 912,576 |
| Nominal output tokens | 1,056 |
| Expected output tokens | 4,587,520 |
| Conservative output tokens | 8,650,752 |

The nominal request count matches the expected local-vote arithmetic:

```text
8 × (12 initialization calls + 12 agents × 10 rounds) = 1,056
```

Live University metadata reported zero proxy accounting units. This is not a currency-valued prediction.

Automatic execution planning resolves to:

- 8 scientific-cell shards;
- array throttle 8;
- 8 episode slots and 8 request permits per shard;
- 64 total request permits;
- approximately 384 requests/minute at the assumed 10-second latency;
- 8 CPUs and 8 GB per active shard;
- two-hour task time limit.

## 9. Takeover endpoint analysis

A reusable versioned endpoint implementation was added at:

```text
src/mas_cc/studies/episode_endpoints.py
```

Standard aggregation invokes it only when the analysis recipe contains:

```yaml
episode_endpoints:
  classifier: relational_false_takeover_v1
```

It writes:

```text
analysis/tables/episode_endpoints.parquet
analysis/tables/episode_endpoint_summary.parquet
analysis/reports/false_takeover.md
```

The episode table records:

- initial/final false-target share;
- initial/final truth share;
- semantic final winner;
- explicit tie flag and tied semantics;
- false-target and truth final-win flags;
- maximum false-target share and its round;
- whether and when the false target first exceeded 0.5;
- whether truth ever exceeded 0.5;
- final-minus-initial false and truth shares;
- deterministic descriptive classification;
- `matched_revised_theory_applicable=false`.

Winner and tie calculations use semantic vote counts, not display letters. A false win requires the false target to have a strictly larger final count than every competitor.

Classification rules, in priority order:

1. `FALSE_FINAL_TAKEOVER`: false target is the unique final winner.
2. `TRANSIENT_FALSE_MAJORITY`: false share exceeded 0.5, but it is not the unique final winner.
3. `FALSE_STEERING_WITHOUT_MAJORITY`: final false share exceeds initial false share without either condition above.
4. `NO_MEANINGFUL_FALSE_STEERING`: none of the above.

Round indices are zero-based. The initialized state is represented as round `-1` when it is the first or maximum majority state.

The report begins with the required eight-row episode table, followed by the four-regime descriptive summary and the direct existence answer.

## 10. Plots

When aggregation has completed data, it creates only:

1. `false_target_share_over_rounds.png`
2. `truth_share_over_rounds.png`
3. `final_outcomes.png`

The trajectory plots retain individual episode lines and mark the 0.5 majority threshold. The final plot compares the four `(q,b)` regimes. No broad Study 08-style plot suite is requested.

## 11. Theory handling

`analysis.yaml` sets:

```yaml
theoretical_reference: none
```

Study 09b uses `q=2,3`. The revised single-affinity theory is explicitly a `q=1` controlled-layer reference, so this study does not modify `theory_revised.py`, report it as matched theory, or generate new theoretical efficiency definitions.

Empirical controller diagnostics remain requested, including sensing, action entropy, target response, target conditional mutual information where estimable, support diagnostics, and terminal current. With one episode per cell, many inferential estimates may be unsupported or descriptive only; the takeover endpoint remains primary.

## 12. Tests and checks completed

The focused command was:

```text
PYTHONPATH=src ./.venv/bin/python -m pytest tests/mas_cc/test_study09b.py tests/mas_cc/test_relational_epistemic_factorization.py tests/mas_cc/test_relational_imitation_round_feedback.py tests/mas_cc/test_studies.py -q
```

Result: **157 tests passed**.

Coverage includes:

- exact eight-episode Study 09b contract;
- exact grid axes and values;
- semantic false targets;
- real strategic facts;
- task manifest properties;
- fail-before-submission behavior for an injected `q=1` error;
- exact `b=9` and `b=12` schedule sizes;
- semantic winner/tie handling;
- transient-majority classification;
- production one-slot replacement through `q=3`;
- existing epistemic and standardized-study regressions.

Editor diagnostics reported no errors in the changed Python files or Study 09b YAML folder.

## 13. Commands for the next operator

### Local no-completion-call preflight

```text
PYTHONPATH=src ./.venv/bin/python -m mas_cc.cli.main study preflight --config-dir configs/runs/relational_reasoning/population_study_09b --output-dir results/inspection/study09b_preflight
```

### Potsdam environment check before submission

```text
/home/ojedamarin/.local/share/miniforge3/bin/conda run -n MA-CC python -c "import mas_cc, pandas, pyarrow; print(mas_cc.__file__, pandas.__file__, pyarrow.__file__)"
```

### Authorized Potsdam submission

```text
/home/ojedamarin/.local/share/miniforge3/bin/conda run -n MA-CC --live-stream mas-cc study submit --config-dir configs/runs/relational_reasoning/population_study_09b
```

The configured result root is:

```text
/work/ojedamarin/Projects/LanguageGames/MA-CC/results/studies/relational_population_study_09b
```

### Aggregation after all eight cells seal

```text
/home/ojedamarin/.local/share/miniforge3/bin/conda run -n MA-CC --live-stream mas-cc study aggregate --study-dir /work/ojedamarin/Projects/LanguageGames/MA-CC/results/studies/relational_population_study_09b
```

Do not use `--allow-incomplete` for the final report.

## 14. Current stopping point

Implementation intentionally stopped after:

- frozen dataset generation;
- deterministic dataset validation;
- configuration creation;
- strict design validation;
- focused tests;
- live-pricing preflight.

No provider-backed episode was run. No SLURM job was submitted. No Study-09b-specific SLURM job was created. No production controller or evidence semantics were changed. No replacement mutual-information estimator or theory extension was added.
