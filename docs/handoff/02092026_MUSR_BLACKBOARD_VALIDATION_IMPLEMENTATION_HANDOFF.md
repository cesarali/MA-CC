# Handoff A: Implement the Blackboard Prompt Validation Harness (Development + Small Smoke Test Only)

## Purpose

Implement and validate the **blackboard prompt/dynamic-state validation harness**, but **do not run the full behavioral experiment**.

This handoff is for a development agent with repository access. Its job is to:

1. implement the controlled `S0 / S1 / S2` validation states using the **actual blackboard runtime renderer**;
2. implement all logging, analysis, concurrency/rate-limit configuration, and report-generation machinery;
3. run only a **small real-provider smoke test** to verify the harness end-to-end;
4. leave a frozen, ready-to-run config and exact execution command for a second compute agent with access to more CPUs.

The full behavioral run will be executed separately.

---

## 1. Freeze the benchmark

Do not modify:

```text
tasks = current six frozen symbolic-ambiguity tasks
private breadth = k=4
symbolic ambiguity gate = M <= 0.45
Hbar >= 0.90
minimum score margin = 2
round-zero prompt = P2
Full Profile = F9
game-playing model = gwdg/openai-gpt-oss-120b
```

Do not regenerate tasks or evidence.

Do not change the blackboard or controller semantics.

---

## 2. Implement the actual blackboard validation states

Use the actual runtime renderer/data structures, not a manually reconstructed prompt.

Use the current equivalent of:

```text
relational_blackboard_ballot
```

Construct three deterministic states for each selected task-agent pair.

### S0 — Initial/private

```text
original private evidence
no sampled board messages
no newly acquired evidence
```

### S1 — Intermediate blackboard

```text
original private evidence
+
1–2 exact evidence cards acquired from earlier messages
+
q sampled board messages
+
current previous vote
```

Target approximately:

```text
5–7 / 9 latent values represented
```

Include:

- at least one message with a valid `shared_fact_id`;
- at least one semantic-only message without new exact evidence.

### S2 — Near/full blackboard

```text
8–9 / 9 latent values represented
+
realistic sampled blackboard messages
+
current previous vote
```

This must use the **actual later-round blackboard prompt**, not the static F9 prompt.

---

## 3. Freeze state definitions before any behavioral run

Save deterministic state definitions containing:

```text
task_id
agent_id
state_id
current_vote
original_evidence_ids
acquired_evidence_ids
total_evidence_ids
latent_values_covered
sampled_message_ids
sampled_message_types
sampled_message_texts
sampled_shared_fact_ids
reply_to structure
```

Create:

```text
states/frozen_state_definitions.json
states/state_summary.csv
states/rendered_prompt_examples.md
```

The second compute agent must be able to run the experiment from these frozen definitions without reconstructing them.

---

## 4. Use q=1 for the primary probe

Freeze:

```text
q = 1
```

Do not add q=3 yet.

---

## 5. Implement the full-run design, but do not execute it here

Prepare the harness for:

```text
6 tasks
× 4 fixed agents/task
× 3 states
× 5 repetitions
= 360 logical calls
```

The full-run config must be saved and ready for another agent.

---

## 6. Implement parallel execution controls

Expose explicit configuration for:

```text
local_workers = 4
max_concurrency = 30
max_rpm = 500
```

The full-run compute agent may use more local CPUs, but provider concurrency must still respect:

```text
<= 30 concurrent requests
<= 500 RPM globally
```

Prefer repository-native provider concurrency/rate limiting.

The limiter must be global.

Log:

```text
configured worker count
configured concurrency
configured RPM cap
observed peak concurrency
observed sustained RPM
```

---

## 7. Implement retry behavior

Reuse existing provider retry logic.

A retry:

- keeps the same logical call ID;
- preserves requested seed where possible;
- is archived as another provider attempt;
- does not count as another designed observation.

Support concurrency fallback if instability occurs:

```text
30 -> 20 -> 10
```

---

## 8. Mandatory semantic/unit checks

Before real calls, implement automated checks that verify:

1. `RESULT` renders correctly.
2. `REPLY` has valid `reply_to`.
3. semantic-only messages do not create exact evidence.
4. valid `shared_fact_id` adds exact evidence to private memory.
5. expired board messages are absent.
6. previously acquired exact evidence persists after board expiry.
7. private `reason` is not rendered publicly.
8. output schema allows valid public message emission.
9. S0/S1/S2 latent coverage is as intended.
10. no hidden matrix values leak into prompts.

Run relevant existing regression tests.

---

## 9. Small real-provider smoke test only

This development agent **may call the real provider**, but only to verify the harness.

Use:

```text
gwdg/openai-gpt-oss-120b
```

Recommended smoke:

```text
2 tasks
× 1 agent/task
× 3 states
× 2 repetitions
= 12 logical calls
```

Maximum allowed development smoke:

```text
24 logical calls
```

Do not run the 360-call experiment.

The smoke test must verify:

- provider calls succeed;
- prompt rendering is correct;
- outputs parse;
- evidence memory behaves correctly;
- logging works;
- analysis script runs;
- report generation works.

The smoke result is not a scientific estimate.

---

## 10. Implement the analysis pipeline

Prepare code that, after the full run, computes:

```text
truth rate by S0/S1/S2
95% Wilson intervals
per-task results
per-agent results
truth vs latent coverage
truth vs exact evidence-card count
truth vs message count
```

Also implement optional paired static-vs-blackboard analysis if the execution config enables it.

---

## 11. Report template

Create the report generator/template:

```text
analysis/blackboard_prompt_validation_report.md
```

The development smoke may populate it with clearly marked `SMOKE TEST` values.

The full compute run must be able to regenerate/overwrite the scientific result sections from the completed data.

Required sections:

```text
A. Motivation
B. Frozen benchmark
C. Actual blackboard runtime prompt
D. S0/S1/S2 state construction
E. Parallel execution
F. Behavioral results
G. Task heterogeneity
H. Static-vs-blackboard comparison (if enabled)
I. Evidence-response analysis
J. Blackboard semantic checks
K. PASS / BORDERLINE PASS / FAIL
L. Limitations
```

---

## 12. Required development outputs

Create a study directory such as:

```text
results/studies/musr_blackboard_prompt_validation_01/
```

At minimum, leave:

```text
config_full.yaml
config_smoke.yaml
states/frozen_state_definitions.json
states/state_summary.csv
states/rendered_prompt_examples.md
sanity/evidence_memory_checks.csv
sanity/message_schema_checks.csv
sanity/board_lifetime_checks.csv
behavioral/smoke_raw_calls.jsonl
analysis/blackboard_prompt_validation_report.md
```

Also provide an exact command for the full run.

Example form only:

```text
<repo-native command> --config results/studies/musr_blackboard_prompt_validation_01/config_full.yaml
```

Use the actual repository CLI.

---

## 13. Development acceptance criteria

This handoff is complete when:

1. the real blackboard renderer is used;
2. S0/S1/S2 are implemented and frozen;
3. full-run config is ready for 360 calls;
4. concurrency/RPM controls are implemented;
5. semantic checks pass;
6. a 12–24 call real-provider smoke succeeds;
7. raw smoke prompts/responses are archived;
8. the analysis pipeline runs on smoke data;
9. the report template is generated;
10. an exact full-run command is printed;
11. **the 360-call full experiment has NOT been run**.

At completion print:

```text
implementation status
tests passed/failed
smoke logical calls
smoke provider attempts
smoke wall-clock time
full-run logical call count
full-run config path
full-run command
results directory
report path
```
