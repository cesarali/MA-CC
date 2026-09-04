# MuSR Blackboard DeepInfra Stress Run — Repair Handoff

**Date:** 2026-09-04
**Study:** `musr_blackboard_01_deepinfra_truthful_false_q3_stress`
**Config folder:** `configs/runs/relational_reasoning/blackboard_game/blackboard_1_deepinfra_q3_stress`

## Outcome

The stress run failed twice and now runs clean. Three separate defects were
found, each verified against captured evidence rather than inferred. The study
is resubmitted as SLURM job **1864299**.

The original goal — measure DeepInfra's sustained concurrency and the wall time
of one episode and one cell — is answered in *Measurements* below. DeepInfra
itself was never the problem.

## Defect 1 — every episode died at update 5 (`TypeError`)

**Symptom.** All 210 episodes of job 1864184 failed. Every one died at
`interaction-0004` with `final_cursor {round_index: 0, within_round_index: 2}`.

**Cause.** `orchestrator.py:1593` copied `game.options["board"]` into the
metadata of every retained prompt example. The config loader hands out a
`mappingproxy` (Python's read-only view of a dict), which `json.dump` refuses:

```
TypeError: Object of type mappingproxy is not JSON serializable
```

**Why update 5, deterministically.** `prompt_examples.count: 3` makes
`_CellPromptSampler.capture` write only at beginning / middle / end, which for
`rounds: 10` are indices 0, 4 and 9. Initialization comes from a pre-computed
artifact (`paired_local_vote`), so no prompt is issued at index 0. Index 4 is
therefore the first write — and it crashed.

**Evidence.** 202 half-written `prompt_candidates.json.gz.tmp` files and zero
finished ones. Each stops mid-value immediately after `"board":`.

**Fix.** `dict(...)` at the call site; `default=str` on the sampler's
`json.dump` so a prompt example can never again destroy a paid episode.

**Note on visibility.** `orchestrator.py:1606` drops `str(exc)` under
`semantic_dashboard` retention, leaving only `TypeError`. That retention rule is
correct — an exception message can quote a raw provider response — so it was
left alone. Instead the shard log now records the crash *site*
(`file:line in function`), which carries no provider text.

## Defect 2 — the study dashboard could not see a running study

**Symptom.** Cells showed as RUNNING with no episodes and no trajectories.

**Cause.** `mas_cc/studies/discovery.py:65` accepts a run root only if it holds
`manifest.json` plus a resolved config. Those are written when a shard *seals*.
A running shard has neither, so a live study discovers zero cells.

**Fix.** A dashboard-local `_live_runs` in
`blackboard_dashboard/study_data.py` that also accepts a root with a `cells/`
directory of resolved configs and no manifest. Roots that *do* carry a manifest
are skipped, so strict and live discovery never overlap.
`mas_cc/studies/discovery.py` was deliberately **not** weakened — its strictness
is what stops aggregation counting an unfinished shard as science.

Prompt examples got the same treatment: `dashboard_prompt_examples.json` is only
rendered at cell close, so `_live_prompt_samples` reads the per-episode
`.resume/*/prompt_candidates.json.gz` candidates instead, with the same
beginning/middle/end selection and the same output shape. Live samples are
flagged `"live": true`.

## Defect 3 — 34.4% of all LLM calls failed the ballot contract

**Symptom.** Job 1864255 produced no completed episodes. 0 TypeErrors, but
`RelationalDecisionFailed` climbing steadily.

**Cause.** Established from raw responses captured with `artifact_profile: full`
and `logging.options.detailed_prompt_audit.enabled: true`, which writes
`audit_traces.jsonl`. Seven captured failures, three distinct causes:

| Shape | Count | Owner |
|---|---|---|
| Ballot in a ` ```json ` fence behind a broken opening fragment | 1 | our parser |
| Ballot wrapped in `{"role":"assistant","content":"{...}"}` | 1 | our parser |
| Valid ballot, `private_reason` over the 600-character cap | 2 | our prompt |
| Vote key emitted as `"  "` (two spaces) instead of `"vote"` | 2 | the model |
| Degenerate junk (`{"other_agents": [...]}`, 12 tokens) | 1 | the model |

Two of the three causes were ours. The 600-character cap was never stated in the
prompt — it said only `"<brief private reason>"` — so agents were failed for
breaking a rule they were not given.

**Fixes.**

1. `extract_json_object` (`hidden_bench/vanilla/prompts.py`) gained an opt-in
   `required_keys` parameter. Without it, behaviour is byte-identical, so no
   existing caller changes. With it, the search scans fenced blocks first, then
   balanced brace spans, descending one level into string values. It only ever
   *locates* an object the model really sent — it never renames a key, fills a
   field, or infers a vote, so a corrupted ballot still fails rather than being
   quietly repaired into data. The three relational ballot parsers pass
   `required_keys=("vote",)`.
2. `MAX_REASON_CHARACTERS` 600 → 2000 in
   `relational_reasoning/imitation_round_feedback/prompts.py`. This constant is
   local to the relational game; hidden_bench keeps its own 600.
3. The prompt now states the budget: "a few sentences, and at most 2000
   characters", in the decision instruction and in both JSON skeletons.
4. `invalid_response_retries: 3 → 4` (5 trials per decision).

## Measurements (from job 1864255, 5,825 real calls)

**DeepInfra sustained the load without complaint.** The shared adaptive
controller state (`runtime/provider-control/job-1864255/state.json`) shows
`limit` still 100, `last_decrease_at` 0.0, `global_pause_until` 0.0,
`node_pauses {}`, `retry_counts {}`, `last_error null`. Every event is
`status_code 200`.

| Quantity | Measured |
|---|---|
| Call latency | p50 8.1s · p90 15.5s · p95 19.2s · p99 29.5s |
| Latency by attempt number | 8.1 / 8.0 / 8.1 / 7.9s — no degradation under retry |
| One completed decision | 13.6s median |
| One episode (10 rounds × 24 agents = 240 decisions) | ~54 min *(projected)* |
| One cell (10 episodes on 10 parallel slots) | ~54 min *(projected)* |
| Achieved throughput | 230 requests/min |
| Real cost per call | $0.00011 → **$11.02 per 100,000 calls** |

**The 1000 RPM target was never approached, and that is a workload limit, not a
provider limit.** Agents act one at a time inside an episode, so each live
episode contributes at most one in-flight request. Load equals the number of
live episodes, capped by (running shards × 10 episode slots). To probe
DeepInfra's actual ceiling, raise the array throttle above 10 — raising
`target_rpm` will not do it.

The per-episode and per-cell figures are extrapolations from the measured
per-decision rate. No episode had completed when they were taken.

## Runs and archives

| Path / job | State |
|---|---|
| SLURM **1864184** | all 210 episodes failed — `mappingproxy` |
| SLURM **1864255** | cancelled — 0 completions at a 34.4% call failure rate |
| SLURM **1864299** | **current run** |
| `..._q3_stress_failed_mappingproxy_20260904` | archive of 1864184 |
| `..._q3_stress_cancelled_highfailrate_20260904` | archive of 1864255 |
| `..._q3_stress_initializations_prompt600_20260904` | archive of the pre-prompt-change initializations |
| `..._q3_stress_initializations` | regenerated for the new prompt |

The paired initializations had to be regenerated: their compatibility key covers
`prompt_definition_hashes_hash`, which the prompt change alters. Submission
refuses to proceed on a mismatch, which is the check working correctly.

## Verification

- Full suite: **116 failures before and after, identical set.** All 116 are
  pre-existing, caused by config YAML files absent from this checkout
  (`hidden_bench_vanilla.yaml`, `study09j_*.yaml`, `task_002/base_task.json`).
- Two tests were updated because they asserted the behaviour intentionally
  changed: `test_generic_relational_repair_guidance_does_not_use_fragile_slots_super`
  hardcoded `"x" * 601` and now tracks `MAX_REASON_CHARACTERS`;
  `test_prompt_snapshot` now expects the stated limit.
- Dashboard tests: 22 passed.
- Defect 1 fix confirmed by one real DeepInfra episode reaching
  `interaction-0063` and writing a finished `prompt_candidates.json.gz`
  containing the `middle` sample at round 4 — the exact write that killed 210
  episodes.
- Defect 3 fixes replayed against the seven captured responses: **4 of 7
  recovered.** The three that still fail are the two `"  "` key corruptions and
  the junk response, which is the intended outcome.

## Result so far on job 1864299

882 calls in: per-call invalid rate **23.8%**, down from 34.4%.
`private_reason` failures are effectively gone (821 → 5). `response.vote`
remains dominant at 167, which is the `"  "` corruption we deliberately do not
paper over.

**This is worse than the ~15% predicted from the sample of seven.** Predicted
episode survival at 5 trials was 98%; at the measured 23.8% it is about 84%.
Expect roughly one episode in six still to be lost. That is a large improvement
on 3.4% but it is not the number that was forecast, and the forecast came from
a 7-response sample that was too small.

## Open items

1. **`response.vote` at ~19% of calls is the remaining cost.** It is
   `deepseek-ai/DeepSeek-V4-Flash` emitting a corrupted key. Worth testing
   another model before treating this workload as production-ready — retry
   padding is masking a model defect, not fixing it.
2. **The prompt change alters `prompt_definition_hash`.** These 210 episodes are
   no longer directly comparable with earlier `prompt_version 2` runs. Fine for
   a throughput probe; a decision to make consciously before any science.
3. **The parser change makes previously-invalid responses valid.** Opt-in, so
   only the relational blackboard game is affected today, but it does change what
   gets counted.
4. **`assets/app.js` and `assets/style.css` carry changes this handoff's author
   did not make** — a "Reload examples" button and prompt-loading race fixes,
   added by someone else during the same session. They complement the live
   prompt work but were not reviewed or tested here.
5. All changes are **uncommitted** in the working tree.

No study-specific SLURM job file was added; the run uses the generic planned
cell-array launcher. No CMI estimator was replaced or reimplemented.
