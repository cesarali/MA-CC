# Part 1 — Prepare MuSR isolated OSS data and code locally

Companion: `20260907_MuSR_02_Cluster_Launch.md`. Complete this stage on the home computer without live provider calls. Prepare the runner and launch assets needed by Part 2 as well as the frozen data.

## Task for the implementing agent

Prepare the code, frozen evaluation manifest, offline verification, analysis, and cluster launch instructions for the next isolated prompt evaluation. Development happens on the user's home computer; live OSS calls will be launched by the user on the university cluster. Do not launch live provider calls during implementation. Complete all preparation without requiring another design discussion for routine choices.

Respect repository AGENTS.md instructions and existing MAS-CC provider, prompt, configuration, recorder, preflight, concurrency, and resume abstractions. Inspect the current implementation before choosing paths or commands. Extend existing facilities rather than building a separate experiment framework. This handoff specifies desired behavior; any command/config names below are requirements to implement or map to existing equivalents, not claims about current APIs.

## Scientific aim and current status

We want to establish whether the gameplay model solves these tasks with complete evidence, remains uncertain with distributed private evidence, and shifts toward a designated incorrect answer when given strategically selected true reports. A matched random-report comparison tests the contribution of strategic selection.

This is a one-shot evidence-to-answer calibration. It is not a population episode, feedback-policy evaluation, or thermodynamic-efficiency experiment. Earlier strategic-controller population experiments have already finished and their statistics are being aggregated on the cluster. Preserve those results and keep this new task calibration separate.

The user reports that Terra semantic validation has passed, but OSS has not run for these revised tasks. Source result directory: `musr_truthful_selective_task_calibration_01`. Resolve its actual repository path. Read:

- `truthful_selective_equality_diversity_revision_report.md`
- `task_revision_manifest.json`
- Frozen task JSONs, controller report rankings, symbolic checks, and relevant generator/evaluation code.
- Archived predecessor: `musr_truthful_selective_task_calibration_01_before_diversity_revision` (provenance only; do not use its superseded tasks).

Approved candidates:

| Task | Candidate | Terra semantic checks |
|---|---:|---:|
| task_001 | 42 | 55/55 |
| task_002 | 237 | 60/60 |
| task_003 | 130 | 49/49 |

Candidate 53 for task_002 was replaced prospectively, without OSS/population outcome selection. The audit records 321 generation + 344 semantic-validation logical calls, 665 provider attempts, and no transport retries. Do not rerun Terra or regenerate tasks as part of this stage.

All 24 equality facts now use canonical deterministic wording: identify the two compared quantities, state a shared categorical rubric, and state the same unnamed category without disclosing its value. Preserve this wording and existing evidence provenance.

The controller ranking supplies a 12-report informative prefix, covers all nine latent variables, and has no zero-marginal additions within that prefix. Main budgets are **3, 6, 9, 12**. C24 is excluded from the default run; it is an optional saturation diagnostic, contains redundancy, and violates the intended partial-disclosure ceiling for task_002.

## Freeze before live evaluation

1. Verify task/candidate identities, gold answer, false target, complete evidence, private assignments, report pool, and prefix rankings against the revision manifest.
2. Record immutable hashes of all source artifacts, resolved configuration, prompt version, evaluation manifest, and repository commit.
3. Verify the claimed symbolic gates offline. Record the exact definition and prior/weighting used for symbolic `p_false`. It is not an OSS response probability. In particular, distinguish prefix-only symbolic calculations from private-evidence-plus-prefix calculations.
4. Preserve all three approved tasks regardless of subsequent steering strength. Do not rerank reports, replace tasks, alter equality wording, or tune prompts using these evaluation outcomes. If evaluation discovers an actual defect, report it and version any later development separately.
5. Use the existing approved private-evidence assignments/population size. Do not invent a fresh N=12 versus N=24 sweep. If both assignments exist and neither is designated for calibration, use N=12 for this small stage, document that choice before calls, and keep N=24 out of the default grid.

## Evaluation conditions

Every evaluation is an independent fresh context, with no memory from previous calls. Reuse the gameplay-compatible task and report rendering, response schema, and answer parser, without running the board scheduler. Do not label reports as strategic, random, controller-authored, truthful, or misleading in the model-visible prompt. Never expose hidden states, candidate scores, symbolic probabilities, gold answers, or target labels.

| Condition | Model-visible evidence | Purpose |
|---|---|---|
| Full evidence | All approved evidence needed to recover the unique optimum | Solvability by OSS |
| Private only | One actual agent's assigned evidence | Initial ambiguity and answer bias |
| Strategic b | Same private evidence + first b frozen strategic reports | Selective-disclosure response |
| Random b | Same private evidence + b true reports sampled from the eligible pool | Matched message-budget reference |

Budgets: b in {3,6,9,12}. Use all agents in the selected frozen assignment. Full-evidence checks are task-level; do not duplicate them once per agent. Run private-only baselines once per agent/order combination and reuse them in paired comparisons across budgets.

For random controls, use the same legal source pool and report rendering as the strategic controller, with uniform sampling without replacement. Construct one seeded random permutation of that pool per task/random replicate and use nested prefixes across budgets. Default: five random permutations per task, reused across agents and answer orders. Freeze seeds and packets before live calls. Do not reject random packets based on their effects or symbolic lift. Apply any existing controller eligibility restrictions identically to both conditions.

Preserve the established treatment of reports that overlap private evidence. Do not silently remove or replace them: b is the number of delivered reports. Record unique fact count, overlap with private facts, latent-variable coverage, predicate families, report length, and symbolic compatible-world reduction when available. Equal report counts need not imply equal information or tokens; interpret the random comparison accordingly.

Use all six answer-label permutations for the three-option tasks. Map responses back to semantic allocations before scoring. Keep a given permutation identical across paired conditions and budgets. Check whether permutations are merely label changes or also reorder displayed alternatives, and describe the implemented counterbalancing precisely.

Use `gwdg/openai-gpt-oss-120b` through the existing provider adapter, resolving the actual configured provider/model alias. Never substitute Terra or another gameplay model silently. Preserve the approved gameplay decoding settings, including temperature=0.0 if that is current. At temperature zero, counterbalanced prompts and different evidence packets provide variation; identical reruns are not independent scientific replicates. Record any provider seed support honestly.

## Manifest size and preflight

With T=3 tasks, A agents per task, P=6 answer orders, B=4 budgets, and R=5 random packets:

`logical_calls = T * P + T * A * P * (1 + B + B*R)`

This is 5,418 calls for A=12 and 10,818 calls for A=24, before retries. For unequal assignments, use the actual sum of agents. This is a bounded calibration grid, not a huge population sweep. Provide a smaller explicitly named smoke subset drawn from the same manifest; successful smoke requests must be reusable on resume, not repeated in the full run.

Preflight must print exact call counts by condition, input/output token assumptions, estimated cost when provider pricing is configured, rough duration based on observed or clearly assumed latency, and output location. Include retry allowance separately. Do not invent provider pricing or assume a free endpoint. Allow a configured call/token/cost cap to prevent accidental grid expansion. Any smaller scientific grid must be chosen before viewing live outcomes.

## Local implementation and meaningful offline checks

Prepare manifest construction, rendering, runner, resume, aggregation, configuration, and documentation entirely offline with mock responses. Tests should resolve concrete risks:

- Correct approved candidates/hashes; replaced candidate cannot enter the manifest.
- Prefix nesting and exact message counts; truthful source provenance; no answer/hidden-state leakage.
- Semantic scoring invariant to all six answer permutations.
- Random packets deterministic from seeds and independent of scheduling order.
- Correct paired keys and exact planned request counts.
- Resume skips completed request IDs; retry attempts cannot become extra observations.
- A failed request remains missing/failed, never converted into a wrong or random answer.
- Mock mixed outcomes produce correct denominators, missing-pair counts, and task-level summaries.

Run the relevant existing checks rather than broad unrelated test suites. Do not claim live-model validation from mocks.

## Records and outputs

Create a new distinct results directory, for example `musr_truthful_selective_isolated_oss_01`, resolved using repository conventions. Never overwrite the calibration source or already-completed population results.

Deliver:

- Frozen resolved config, input hashes, evaluation manifest, and exact source commit.
- Exact model-visible prompt archive and raw/parsed response records (no credentials).
- Attempt ledger and request-level table with task, candidate, agent/evidence bundle, condition, budget, answer order, random-packet seed, semantic answer, gold/false/other flags, and completion status.
- Paired comparison tables, per-task summaries, aggregate summaries, and missingness/retry diagnostics.
- Markdown report and reusable plotting/aggregation entry point. Use existing report tooling for PDF if available; do not introduce a PDF subsystem merely for this stage.
- Plot accuracy/false-target response versus b, with private-only and strategic/random comparisons and individual task traces. Label symbolic `p_false` separately if displayed.

Primary quantities:

1. Full-evidence correct fraction, per task and answer order.
2. Private-only correct, false-target, and other-answer fractions.
3. Strategic-minus-private and strategic-minus-random changes in false-target choice, paired by task, agent, and answer order; average random packets within that matched unit first.
4. Corresponding changes in correct-answer choice.
5. Invalid-response and provider-failure fractions with explicit denominators.

Report both pooled counts and equal-task-weight summaries, identifying which is primary (default equal-task-weight). Counterbalanced orders and random packets are repeated measurements, not independent new tasks. With only three tasks, emphasize all three task results and avoid strong population-wide generalization. If showing bootstrap intervals, state the resampling unit, preserve pairing/dependence, and distinguish within-task conditional uncertainty from uncertainty across tasks; a three-task bootstrap is intrinsically weak. Do not report naive binomial intervals treating every call as independent. Completed-case summaries must disclose missing pairs; do not hide failures in denominators.

Do not calculate MI, susceptibility, or thermodynamic efficiency from this one-shot calibration as though it were a closed-loop population experiment. Report shifts as isolated evidence effects.

## Interpretation and decision rules

The desired pattern is high full-evidence solvability, nontrivial private-only uncertainty, and increased false-target choice under selective evidence relative to private-only and random evidence. A strictly monotonic OSS response across all four budgets is not required.

Freeze any existing quantitative acceptance thresholds before live calls; otherwise report per-task diagnostics rather than inventing a retrospective pass cutoff. Poor full-information performance signals a reasoning/prompt/task problem to investigate. Good full-information performance with weak selective steering is a valid scientific finding, not permission to replace tasks until steering appears. Near-chance private-only accuracy alone does not prove an individual evidence bundle is symbolically ambiguous: retain the symbolic evidence checks too.

## Transfer contract for Part 2

Deliver the frozen task inputs, evaluation manifest, prompt archive/version, resolved model and decoding configuration, checksum manifest, source commit, offline test report, and exact executable commands for preflight, smoke, full run, resume, and aggregation. Include a cluster launch script using existing repository conventions. Implement the concurrency, shared rate limiting, retries, and checkpoint requirements specified in Part 2; verify them offline. Write a small machine-readable handoff containing these paths and commands so the cluster agent does not reconstruct the scientific design. Include all inputs that are not tracked in Git in a checksummed transfer bundle. No credentials belong in the bundle.

Default scientific design: three frozen tasks, budgets 3/6/9/12, six answer orders, five random packets per task, all agents in the selected approved assignment. Preserve already-completed population studies. Stop after preparing and verifying the runnable package; live execution belongs to Part 2.
