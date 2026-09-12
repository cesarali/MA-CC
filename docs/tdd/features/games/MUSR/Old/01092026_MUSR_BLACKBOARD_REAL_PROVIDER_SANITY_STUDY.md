# Handoff: Small Real-Provider MuSR Blackboard Sanity Study

**Date:** 2026-09-01  
**Purpose:** define a short, qualitative real-provider test of the finite-memory q-message blackboard before designing a larger experiment.

This document is an implementation and execution brief for another agent. It does not authorize changing the scientific design below without documenting why.

---

## 1. Goal

Run the relational imitation round-feedback game on one already validated MuSR Team Allocation task using the new finite-memory public blackboard.

This is a **sanity study**, meaning a small run intended to verify that the implementation behaves correctly and produces interpretable communication traces. It is not powered for statistical claims.

Compare three arms:

1. `no_control`
2. `direct_recommendation`
3. `coordination_request`

The game-playing population must use:

```text
gwdg/openai-gpt-oss-120b
```

through the existing MAS-CC University provider stack.

Do **not** use `microsoft/gpt-5.6-terra` for any population decision. Terra was used only for offline MuSR task generation and validation.

---

## 2. Fixed scientific design

Use these values in every arm:

```yaml
game:
  type: relational_imitation_round_feedback
  population_size: 12
  horizon: 2
  topology: complete
  options:
    task_family: musr_team_allocation
    task_dataset_dir: results/studies/musr_team_allocation_validation_01/tasks
    task_id: task_001
    dynamics_mode: reasoning
    rounds: 2
    social_group_size: 1
    social_mode: board
    board:
      sampling: uniform
      message_lifetime_rounds: 1
      exclude_self_authored: true
      allow_no_post: true
    vote_visibility: public
    prompt_version: 1
    receiver_epistemic_disposition: vigilant
    stop_on_consensus: false
    invalid_response_retries: 1
    expected_validation_failure_rate: 0.05
    initialization:
      mode: uniform_random
```

Use **2 independent replicas per arm**. This is the shortened default. It gives:

```text
3 arms × 2 replicas × 2 rounds × 12 focal updates = 144 logical provider calls
```

There are no provider calls for initialization because `uniform_random` is used.

With one validation retry, the hard attempt ceiling is 288 HTTP requests, although valid first responses should keep the observed count near 144.

Use a third replica per arm only if a technical failure invalidates a replica. Do not add replicas merely to obtain a preferred semantic pattern.

---

## 3. Validated task

Use exactly:

```text
results/studies/musr_team_allocation_validation_01/tasks/task_001/base_task.json
results/studies/musr_team_allocation_validation_01/tasks/task_001/distribution_N12.json
```

Expected task properties:

```text
task family: musr_team_allocation
population: 12 agents
options: ALLOCATION_0, ALLOCATION_1, ALLOCATION_2
gold answer: ALLOCATION_2
evidence cards: 27
latent evidence groups: 9
no-single-agent violations: 0
```

Before running, verify both files against the hashes in:

```text
results/studies/musr_team_allocation_validation_01/manifest.json
```

The adapter must preserve each evidence card as one atomic evidence item. It must not expose the hidden skill matrix, cooperation matrix, hidden claims, candidate scores, or gold answer in agent prompts.

Prompts must include:

- the public scenario;
- the public question;
- the three full allocation descriptions;
- only the focal agent's assigned/acquired evidence cards;
- only the sampled live board messages.

Internal IDs such as `ALLOCATION_2` may remain in stored state, but the model-facing options and controller requests must use readable allocation descriptions.

---

## 4. Provider configuration

Use the existing University provider implementation with:

```yaml
llm_provider:
  type: university
  model: gwdg/openai-gpt-oss-120b
  credentials_env: POTSDAM_API_KEY
  base_url_env: BASE_POTSDAM_LLM_URL
  timeout_seconds: 180
  max_retries: 2
  request_concurrency: 2
  temperature: 1.0
  max_output_tokens: 4096
  options:
    estimated_latency_seconds: 10.0
```

Why these settings:

- `request_concurrency: 2` allows the two replicas in one arm to run concurrently without creating unnecessary load.
- `max_output_tokens: 4096` avoids truncating GPT-OSS reasoning before its final JSON response.
- `temperature: 1.0` matches the successful MuSR validation run.

Before preflight, query the live model list and model-info endpoint through the documented repository path. Confirm that `gwdg/openai-gpt-oss-120b` is currently available. Do not print or persist credentials, request headers, or the private base URL.

Run the three arms sequentially. Do not launch three separate processes concurrently because each process has its own provider-concurrency limit.

---

## 5. Arm definitions

### 5.1 No control

```yaml
control:
  mechanism: none
```

This arm tests ordinary message production, sampling, replies, evidence transfer, and expiration without intervention.

### 5.2 Direct recommendation

```yaml
control:
  mechanism: relational_round_budgeted
  options:
    target: correct
    sensor_sample_size: 12
    policy: soft_target
    threshold: 0.5
    beta: 4.0
    intervention_budget: 4
    advocacy_schedule: always
    message_mode: recommendation_only
    controller_actuation_mode: direct_recommendation
```

`advocacy_schedule: always` is intentional in this small test. It guarantees that the delivery path is exercised.

Expected behavior per round:

- exactly `b = 4` controlled microscopic positions;
- each controlled position receives one transient recommendation;
- the recommendation replaces the only ordinary board slot because `q = 1`;
- the recommendation is not appended to the board;
- exactly four direct controller exposures occur.

### 5.3 Coordination request

Use the same controller settings, except:

```yaml
controller_actuation_mode: coordination_request
```

Expected behavior per round:

- exactly `b = 4` controller-authored `REQUEST` messages are appended;
- each post occurs before sampling at its scheduled position;
- posts carry no `shared_fact_id`;
- the focal samples normally from all eligible live board messages;
- the new request may or may not be sampled immediately;
- total controller exposures need not equal four;
- one request can be sampled by multiple later agents;
- ordinary agents may reply, report results, correct, ignore, or disagree.

---

## 6. Configuration and output layout

Create:

```text
configs/runs/relational_reasoning/musr_blackboard_sanity_01/
├── study.yaml
├── no_control.yaml
├── direct_recommendation.yaml
├── coordination_request.yaml
└── analysis.yaml
```

Use `execution.repetitions: 2`, `execution.parallelism: 2`, and `storage.artifact_profile: full` in each experiment config.

Use this dedicated result root:

```text
results/studies/musr_blackboard_sanity_01/
```

The final result folder must retain:

```text
results/studies/musr_blackboard_sanity_01/
├── configs/                         # exact source configs used
├── preflight/                       # preflight reports and IDs
├── runs/
│   ├── no_control/
│   ├── direct_recommendation/
│   └── coordination_request/
├── analysis/
│   ├── board_message_trace.jsonl
│   ├── board_message_trace.csv
│   ├── summary_by_arm.csv
│   ├── invariant_checks.json
│   └── sanity_report.md
└── task_provenance.json             # task paths and verified hashes
```

The normal MAS-CC run directories under `runs/` must keep their full episode artifacts, including:

- resolved configuration;
- provider usage and status records;
- prompt examples;
- raw microscopic trajectory;
- round trajectory;
- episode result and manifest.

Do not create a study-specific SLURM job. This small study may be run locally as three sequential `experiment run` commands. If it is run on Potsdam, follow `AGENTS.md` and use the generic study launcher plus the dedicated `MA-CC` Conda environment.

Set the study analysis recipe to:

```yaml
theoretical_reference: none
estimators: []
derived: []
plots: []
```

The current q-voter theory assumes contemporaneous peer sampling and is not an exact theory for board mode.

---

## 7. Required preflight checks

Before any real provider request:

1. Verify the selected Python environment imports `mas_cc`, `pandas`, and `pyarrow`.
2. Verify task and distribution hashes against the validation manifest.
3. Run focused MuSR adapter and blackboard tests.
4. Run preflight for all three configs.
5. Confirm all three configs resolve to:
   - model `gwdg/openai-gpt-oss-120b`;
   - `N=12`;
   - `q=1`;
   - two rounds;
   - two replicas;
   - board lifetime one;
   - theoretical reference `none`.
6. Record nominal, expected, and conservative request counts, token bounds, cost estimate, and estimated wall time.
7. Confirm that the result root is the dedicated folder above and does not overlap the validation study.

Stop before real calls if any preflight is denied or if the model is unavailable.

---

## 8. Trace extraction

After all six episodes complete, build one normalized board trace row per microscopic update. Preserve at least:

```text
arm
replica / episode_id
round_index
within_round_index
global_update_index
focal_agent_id
vote_before
vote_after
focal_changed
q_requested
q_effective
sampled_message_ids
sampled_message_authors
sampled_message_types
sampled_message_ages
sampled_controller_message_ids
peer_exposed_fact_ids
new_peer_fact_ids
focal_posted_message
new_message_id
new_message_type
new_message_reply_to
new_message_shared_fact_id
board_size_before
board_size_after
controlled_position
controller_action
controller_actuation_mode
controller_message_posted
controller_message_id
controller_message_directly_exposed
```

Include complete public message objects where available so the report can quote their text. Never publish private `reason` fields as if they were public communication.

---

## 9. Mandatory invariant checks

Write machine-readable results to `analysis/invariant_checks.json`. Every check must include `passed`, the number of violations, and the relevant episode/message IDs.

### 9.1 Lifetime

For `message_lifetime_rounds = 1`, verify from actual sampled-message records:

```text
sampled message round_created == focal update round_index
```

No message created in round `r` may appear in a sample from round `r+1`.

Also verify round-level expiration counts against messages whose `expires_after_round` equals that round.

### 9.2 Controller budget and reach

For every `coordination_request` advocacy round:

```text
controller_posts == b == 4
```

Verify separately:

```text
controller_message_exposures
controller_unique_readers
controller_direct_replies
controller_reply_descendants
```

Do not require exposures to equal posts. The expected causal path is:

```text
posts → sampled readers → optional replies/results → later sampled descendants
```

For every `direct_recommendation` advocacy round, verify:

```text
controlled positions == 4
transient direct exposures == 4
controller-authored persistent board posts == 0
```

### 9.3 Evidence semantics

For every sampled ordinary message:

- If `shared_fact_id` is absent, reading the prose must not add a new item to `K_i`.
- If `shared_fact_id = e`, `e` must already have been known by the author when posted.
- A reader may acquire `e` only when the message carrying `e` was actually sampled.
- Acquisition provenance must name that message ID.
- The rendered evidence text must exactly match the frozen evidence card.

Free-form `CLAIM` and `RESULT` prose may coincide with a later vote change. Record this as a possible semantic influence, not as an exact evidence acquisition and not as a causal proof.

### 9.4 Privacy and hidden-state safety

Search every retained prompt and public message. Confirm that none contains:

```text
skill_matrix
cooperation_matrix
candidate_scores
hidden_claim
gold_answer
```

Also confirm that one agent's private `reason` is never rendered in another agent's prompt.

### 9.5 Provider identity

Verify from resolved configs and provider audit records that every game decision used:

```text
gwdg/openai-gpt-oss-120b
```

No population call may use Terra.

---

## 10. Semantic summaries

Create `analysis/summary_by_arm.csv` with one row per arm and at least:

```text
episodes
rounds
logical_calls
validation_retries
QUESTION
REQUEST
RESULT
REPLY
CORRECTION
CLAIM
ordinary_messages
controller_posts
controller_exposures
controller_unique_readers
controller_direct_replies
controller_reply_descendants
exact_evidence_exposures
new_exact_evidence_acquisitions
vote_changes
```

Also inspect these patterns manually:

1. `QUESTION → RESULT` or `QUESTION → REPLY` chains.
2. Exact evidence moving through `shared_fact_id`.
3. Public prose followed by a vote change without an exact evidence acquisition.
4. A controller request being sampled and receiving a direct response.
5. A response to a controller request being sampled by a later agent.

Absence is a valid finding. Do not rerun selectively until a preferred pattern appears, and do not describe temporal association as proven causation.

---

## 11. Required report

Write:

```text
results/studies/musr_blackboard_sanity_01/analysis/sanity_report.md
```

The report must state first whether the implementation passed the sanity checks.

Include:

1. **Design:** task, model, `N`, `q`, lifetime, rounds, replicas, `q_c`, and `b`.
2. **Provider evidence:** live model availability, actual model IDs, logical calls, retries, token usage, and reported cost/accounting units.
3. **Invariant results:** lifetime, controller budget, exposure semantics, evidence honesty, privacy, and one-call-per-update checks.
4. **Counts by arm:** all six message types and controller/reply quantities listed above.
5. **One readable round-by-round episode from each arm:** no control, direct recommendation, and coordination request.
6. **Representative chains:** quote public messages and list IDs, authors, types, reply targets, sampled readers, evidence IDs, and subsequent vote changes.
7. **Interpretation:** explicitly say whether the traces contain:
   - meaningful `QUESTION → RESULT/REPLY` behavior;
   - exact evidence propagation;
   - second-order activity after controller coordination requests;
   - apparent prose-only influence without false `K_i` growth.
8. **Limitations:** six episodes are enough for semantic validation only, not effect-size or statistical conclusions.
9. **Recommendation:** either proceed to full experiment design or list the implementation defects that must be fixed first.

For each round-by-round example, use a compact table with columns similar to:

```text
step | focal | sampled messages | types | reply links | evidence acquired |
controller event | posted message | vote before → after
```

---

## 12. Failure policy

If a semantic invariant fails:

1. Stop the remaining real-provider runs if continuing would produce invalid data.
2. Preserve the failed trace under the dedicated result root.
3. Add a focused regression test reproducing the defect.
4. Fix the producer of the incorrect state or record.
5. Re-run the focused mock tests.
6. Re-run only invalidated real-provider replicas, clearly recording which outputs were replaced and why.

Do not hide malformed responses, silently coerce invalid message types, infer evidence IDs from prose, or edit generated traces by hand.

---

## 13. Acceptance criteria

The sanity study is complete when:

- six valid episodes exist: two per arm;
- all population calls used `gwdg/openai-gpt-oss-120b`;
- all raw and resolved artifacts are retained under the dedicated result root;
- lifetime-one messages never cross into the next round's samples;
- each coordination advocacy round has exactly four persistent controller posts;
- coordination exposures are measured independently of posts;
- each direct advocacy round has exactly four transient direct exposures and no persistent controller posts;
- public prose never creates an exact `K_i` item without a sampled `shared_fact_id`;
- exact evidence acquisitions have valid author and message provenance;
- message/reply counts are summarized by arm;
- one readable episode from each arm is presented;
- the report makes only qualitative claims;
- board-mode theory remains disabled.
