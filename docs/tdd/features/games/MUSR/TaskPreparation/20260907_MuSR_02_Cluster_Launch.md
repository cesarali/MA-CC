# Part 2 — Launch and aggregate MuSR isolated OSS evaluation on the cluster

## Task for the cluster agent

Use the package produced under `20260907_MuSR_01_Local_Preparation.md`. This is the operational launch handoff: verify the transfer, run a small live OSS smoke subset, launch the frozen full manifest, and aggregate the results. When the user invokes this launch handoff on the cluster, proceed with these steps without requesting another routine confirmation. Do not launch anything while merely preparing these instructions on the home computer.

Do not regenerate tasks, run Terra, tune prompts, rerank reports, or modify the scientific grid. Earlier population experiments are complete and their aggregation must remain undisturbed. This new evaluation is isolated prompt calibration using `gwdg/openai-gpt-oss-120b`; it does not run population dynamics.

## Verify the transferred package

Read repository AGENTS.md and the machine-readable handoff from Part 1. Confirm the source commit, input/prompt/config/manifest hashes, and offline test report. Approved candidates are task_001=42, task_002=237, task_003=130. Confirm budgets {3,6,9,12}; C24 is outside the default grid. Use the recorded private assignment, all six answer orders, and five frozen random permutations per task. Use existing secret/environment configuration without printing credentials. Missing or inconsistent artifacts must be repaired from the preparation package before any live calls.

The four conditions are full evidence, private only, private plus strategic prefix, and private plus matched random prefix. Source definitions and immutable request IDs live in the manifest. At N=12 the default design has 5,418 logical calls; at N=24 it has 10,818, excluding retries. Use the actual preflight count as authoritative.

## Parallelism and cluster execution

Use one orchestrator with a bounded asynchronous request pool and one shared provider rate limiter. Avoid launching separate per-task processes each with the full rate allowance. All attempts, including retries, consume the same limiter. Reuse repository implementations where possible.

Prepare these configurable profiles as starting values, not claims about the current account quota:

| Profile | In-flight requests | Requests/minute |
|---|---:|---:|
| Smoke | 4 | 60 |
| Default cluster | 30 | 300 |

Concurrency and requests/minute are separate limits. The user previously requested approximately 30 concurrent calls, but rate allowances have varied. Use the current configured account/project quota and reduce the default if necessary. Since other cluster work may share the provider, account for its traffic; never assume this evaluation owns the whole quota. Expose overrides so 500 RPM can be used only when the actual quota and remaining shared capacity support it. Honor token-per-minute limits too if available.

Reuse bounded exponential backoff with jitter, Retry-After support, finite timeouts, and limited attempts for transient timeouts/429/5xx. Reduce pressure on sustained rate limiting. Authentication/configuration failures should stop with a useful error instead of a retry storm. Keep provider failures distinct from schema-validation retries and record both. Do not ask the model to repair answers using gold information.

Checkpoint each completed logical request atomically with deterministic identity derived from the frozen manifest. Persist separate attempt records, response status, latency, token usage, and provider metadata. On interruption, drain or safely cancel in-flight work and permit resume. Detect incompatible manifest/model/prompt hashes before reusing outputs. Avoid concurrent writers targeting the same run.

Reuse the repository's cluster launcher. If the cluster uses a scheduler, provide a job script with required site-specific fields clearly documented; do not invent partition/account settings. If the established workflow is tmux, document it only for the node where university policy permits execution. A remote API evaluation should not request a GPU unless another existing component needs one.

## Launch sequence

1. Run the prepared offline preflight on the cluster. Check call count, configured caps, model, output location, permissions, and remaining shared rate capacity. Use the existing cluster execution node/scheduler conventions.
2. Run the prepared smoke subset with concurrency 4 and 60 RPM, within the actual quota. Check authentication, response parsing, option mapping, complete records, and checkpoint/resume behavior. Smoke success is an operational check, not a requirement for a particular answer or steering effect.
3. If smoke passes operationally, run the complete frozen manifest with the default starting profile of 30 concurrent calls and 300 RPM, reduced if current shared quota requires. Reuse successful smoke request IDs. Do not spend calls repeating identical finished evaluations.
4. Monitor completion, retries, invalid responses, latency, tokens, and remaining estimates. Transport problems may justify concurrency/rate changes; scientific outcomes do not justify changing tasks or prompts. Stop on persistent authentication or artifact-integrity errors. Preserve completed results for resume.
5. Resume the same compatible run after interruption using the prepared command. Do not start a fresh duplicate run unless explicitly requested.
6. Aggregate completed records with the offline analysis entry point. Disclose failed/missing conditions and incomplete pairs. Generate the report and plots even if scientific steering is weak.

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

## Completion report

Return actual run/manifest identifiers, model, commit, concurrency and rate limits used, planned/completed/failed logical calls, retries and tokens, exact resume command if incomplete, and paths to tables, plots, and the Markdown report (PDF too if existing tooling supports it). Summarize full-information solvability and paired strategic-versus-random response by task. Provide the exact established command for downloading the final report/archive to the home computer. Do not infer successful execution from the launcher merely returning a job ID; inspect terminal job status and recorded outputs.
