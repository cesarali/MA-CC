# Handoff B: Execute the Frozen Blackboard Prompt Validation Run

**Implementation status:** ready for the 360-call full run  
**Development smoke:** complete and PASS as an engineering check  
**Full-run status at handoff:** not started

## Purpose

Run the implemented and smoke-tested **blackboard prompt validation harness**
at full scale. The harness tests the actual later-round blackboard prompt, not a
manually reconstructed prompt.

Do not redesign or modify the benchmark unless execution is impossible because
of a demonstrated implementation bug. A scientific failure is a result, not a
reason to edit the tasks or states.

---

## 1. Read this first

Work from the repository root. Read:

1. `AGENTS.md`.
2. `.codex/skills/ma-cc-study-workflow/SKILL.md`.
3. `docs/handoff/02092026_MUSR_BLACKBOARD_VALIDATION_IMPLEMENTATION_HANDOFF.md`.
4. This handoff.
5. `results/studies/musr_blackboard_prompt_validation_01/README.md`.
6. `results/studies/musr_blackboard_prompt_validation_01/preflight/full_report.md`.
7. `results/studies/musr_blackboard_prompt_validation_01/config_full.yaml`.
8. `results/studies/musr_blackboard_prompt_validation_01/states/rendered_prompt_examples.md`.

Then inspect the worktree without cleaning it:

```bash
git status --short
git diff --check
```

The checkout is intentionally dirty. Preserve all existing user changes.

The source calibration and replication both had strict scientific `FAIL`
results. This blackboard validation is an explicitly requested follow-up. It
does not turn either earlier result into a pass.

---

## 2. Frozen implementation and artifacts

The implemented probe name is:

```text
musr_blackboard_prompt_validation
```

The code is under:

```text
src/mas_cc/probes/musr_blackboard_prompt_validation/
```

The CLI entry point is the existing probe command in `src/mas_cc/cli/probe.py`.

The authoritative study root is:

```text
results/studies/musr_blackboard_prompt_validation_01/
```

Required frozen inputs:

```text
config_full.yaml
states/frozen_state_definitions.json
preflight/full_call_plan.json
preflight/full_preflight.json
preflight/full_preflight_id.txt
sanity/evidence_memory_checks.csv
sanity/message_schema_checks.csv
sanity/board_lifetime_checks.csv
```

Current identities are:

| Artifact | SHA-256 |
|---|---|
| `config_full.yaml` | `bea29a08d320b667b77f1f63408e92d9d2ec4fe59f8b1b15761b5d830b1e5ea2` |
| `states/frozen_state_definitions.json` | `8e53d2d87cc08ca404b9d689417df0397f6860c9e8a7a0a5f922a9456c35dc57` |
| `preflight/full_call_plan.json` | `e883535bd24f882b9ae83ade834c936d17eed5c2bb7a2a1464ac536bb6a19302` |
| `preflight/full_preflight.json` | `d377e277c8fe541aaa3579c89833380ff8cdc129ef96a0e647500ade1a2776f5` |
| `preflight/full_preflight_id.txt` | `2af6491330bce26beee0b246f2debf178593b0985eb78b9ba8496efb7767d4ee` |

The preflight ID value inside the final file is:

```text
adea4a26282be5c7103b6c36e2fb23933268e9a7b69838662f6ec4b9f2ca8978
```

If any identity differs, stop and diagnose. Do not rerun preflight merely to
make a changed identity look approved. Re-preflight is appropriate only after a
deliberate implementation fix, followed by review of the new state/prompt
identities.

The state file contains exactly:

```text
72 states = 24 S0 + 24 S1 + 24 S2
```

The call plan contains exactly:

```text
360 unique logical calls = 120 S0 + 120 S1 + 120 S2
```

At handoff, this file must not exist:

```text
behavioral/full_raw_calls.jsonl
```

Its absence means the full provider run has not started.

---

## 3. Do not change the experiment

Keep fixed:

```text
6 frozen tasks
4 frozen agents per task
S0 / S1 / S2 frozen state definitions
q = 1 visible board message
5 repetitions per state
gwdg/openai-gpt-oss-120b
temperature = 1.0
maximum output tokens = 4096
P2 as the frozen source/private prompt
relational_blackboard_ballot as the actual S0/S1/S2 runtime prompt
static comparison = disabled
```

The exact agents are already listed in `config_full.yaml`. Do not reselect
agents. Do not regenerate states, tasks, evidence, messages, option mappings,
or previous votes.

The frozen states mean:

- **S0:** four represented latent values, original private evidence, no
	acquired evidence, and no visible board message.
- **S1:** six represented latent values, two acquired exact cards, and one live
	semantic-only `RESULT` message.
- **S2:** nine represented latent values, five acquired exact cards, and one
	live semantic-only `REPLY` with a valid `reply_to` target in board history.

The exact-evidence acquisition messages are expired in S1/S2, but their cards
remain in private evidence memory. This is intentional.

---

## 4. Environment

### Local developer/compute machine

Use the checkout's selected project environment. In the implementation
workspace, the tested interpreter was:

```text
/home/cesarali/LanguageGames/MA-CC/.venv/bin/python
```

Do not reproduce that absolute path on another machine. Use that machine's
existing project environment.

The University provider requires these environment variables:

```text
POTSDAM_API_KEY
BASE_POTSDAM_LLM_URL
```

Do not print either value. If the checkout uses a local `.env`, load it in the
shell before running the command.

### Potsdam

On Potsdam, every Python command must use:

```text
/home/ojedamarin/.local/share/miniforge3/bin/conda run -n MA-CC python ...
```

Use `--live-stream` if progress must remain visible. First verify, without
printing credentials, that this environment imports `mas_cc`, `pandas`, and
`pyarrow` from the expected environment/repository.

This probe is a single local process with asynchronous provider requests. It is
not a standardized multi-node study and needs no study-specific SLURM job.

---

## 5. Provider and parallel limits

The frozen full config specifies:

```text
configured local_workers = 4
provider request_concurrency = 30
global maximum concurrency = 30
global RPM cap = 500
fallback tiers = 30 -> 20 -> 10
```

`local_workers` is recorded provenance for local orchestration. The workload is
network-bound. Do not edit it to claim more provider capacity.

The implemented runner has:

- one provider instance with a 30-request semaphore;
- one shared coordinator enforcing 30 concurrent requests and 500 requests per
	rolling minute;
- a process-local exact fallback gate with tiers 30, 20, and 10;
- append-only logical-call and validation-attempt records.

Never exceed 30 concurrent requests or 500 requests per minute.

---

## 6. Retry and resume semantics

A logical call is one designed observation. A provider attempt is one actual
request. These counts differ when validation repair or transport retry occurs.

The frozen full design has:

```text
360 logical calls
up to 2 validation attempts per logical call
720 provider attempts for one complete pass
1440 configured provider-attempt ceiling
```

The larger 1,440 ceiling allows one archived outer retry of every logical call
if a run ends in `call_failed`. It is a hard guard, not a request target.

Retries preserve:

```text
logical call ID
requested provider seed
frozen state identity
prompt identity
option-letter mapping
```

Incorrect valid answers are never retried. A schema-invalid answer may receive
the one configured validation-repair attempt. A terminal `call_failed` entry is
eligible for an outer resume attempt with the same identity.

The append-only journal is:

```text
behavioral/full_raw_calls.jsonl
```

If execution stops, rerun the exact same command. The runner skips completed
logical IDs and retries only failed or absent IDs. Do not delete or truncate the
journal.

---

## 7. Pre-run checks

Do not make provider calls until all checks below pass.

### 7.1 Verify frozen counts and absence of a prior full run

Confirm:

```text
72 frozen states
360 call-plan entries
360 unique call IDs
no behavioral/full_raw_calls.jsonl
```

If a full journal already exists, inspect it. Do not start a second result tree
or delete it. Resume the existing journal if incomplete; analyze it if all 360
logical calls are complete.

### 7.2 Verify the retained preflight

Read:

```text
preflight/full_report.md
preflight/full_preflight.json
```

The retained preflight must say:

```text
passed = true
states = 72
logical calls = 360
maximum validated provider attempts = 720
model = gwdg/openai-gpt-oss-120b
provider concurrency = 30
RPM cap = 500
static comparison = disabled
```

Retained planning estimates are:

```text
input estimate including validation retries = 738,410 tokens
output-token ceiling = 2,949,120 tokens
expected wall time = 2.0 minutes
conservative wall time = 0.60 hours
cost unit = proxy_accounting_unit
```

The expected wall time assumes 10 seconds per request and 30-way concurrency.
It is only a planning estimate. The smoke showed that validation repair can
increase real attempts and elapsed time.

### 7.3 Run focused provider-free tests

Run the same focused gate used by the implementation agent:

```text
tests/mas_cc/test_musr_blackboard_prompt_validation.py
tests/mas_cc/test_relational_blackboard.py
tests/mas_cc/test_relational_musr_blackboard.py
tests/mas_cc/test_musr_symbolic_ambiguity.py
tests/mas_cc/test_musr_symbolic_ambiguity_replication.py
tests/mas_cc/test_provider_load_control.py
tests/mas_cc/test_cli_and_inspection.py
```

The latest result was:

```text
59 passed
```

Repository-wide tests have unrelated known baseline failures. Do not expand
this task to repair them.

---

## 8. Exact full-run command

From the repository root, with provider environment variables loaded, run:

```bash
python -m mas_cc.cli.main probe run \
	--config results/studies/musr_blackboard_prompt_validation_01/config_full.yaml \
	--approve-preflight \
	results/studies/musr_blackboard_prompt_validation_01/preflight/full_preflight_id.txt
```

On the implementation machine, replace `python` with:

```text
/home/cesarali/LanguageGames/MA-CC/.venv/bin/python
```

On Potsdam, use:

```bash
/home/ojedamarin/.local/share/miniforge3/bin/conda run -n MA-CC \
	--live-stream python -m mas_cc.cli.main probe run \
	--config results/studies/musr_blackboard_prompt_validation_01/config_full.yaml \
	--approve-preflight \
	results/studies/musr_blackboard_prompt_validation_01/preflight/full_preflight_id.txt
```

Do not run `probe preflight` immediately before execution unless an audited
implementation change requires a new approval identity. The retained full
preflight already approved the frozen state and call-plan hashes.

---

## 9. Monitor and resume

The runner does not print per-call progress. Monitor the append-only journal
read-only.

Useful counts are:

```text
request_started
call_finished
call_failed
unique terminal call IDs
```

Completion requires:

```text
360 unique terminal logical IDs
360 successful parsed observations
0 latest-terminal call_failed IDs
```

Archived older failures may remain in the journal after a successful resume.
That is expected provenance.

If the command exits with incomplete calls:

1. Inspect the latest event for each failed logical ID.
2. Confirm the failure is transport or schema related, not an incorrect valid
	 answer.
3. Rerun the exact same command.
4. Do not change prompt, task, state, seed, or option mapping.

The runner automatically applies the configured 30 → 20 → 10 fallback after
unstable provider failures. Do not edit the config mid-run to create a new
scientific identity.

---

## 10. Implemented output layout

The implemented filenames differ slightly from the conceptual names in the
original request. The authoritative full-run outputs are:

```text
results/studies/musr_blackboard_prompt_validation_01/
├── config_full.yaml
├── manifest.json
├── states/
│   ├── frozen_state_definitions.json
│   ├── state_summary.csv
│   └── rendered_prompt_examples.md
├── sanity/
│   ├── evidence_memory_checks.csv
│   ├── message_schema_checks.csv
│   └── board_lifetime_checks.csv
├── behavioral/
│   ├── full_raw_calls.jsonl
│   └── observation_level_results.csv
├── analysis/
│   ├── blackboard_prompt_validation_report.md
│   └── tables/
│       ├── truth_by_state.csv
│       ├── truth_by_task_state.csv
│       ├── truth_by_agent_state.csv
│       ├── truth_by_latent_coverage.csv
│       ├── truth_by_evidence_card_count.csv
│       └── truth_by_message_count.csv
└── runtime/
		└── provider-control/
```

The six analysis CSV files are the canonical summaries. There are currently no
separate figure files in the implemented harness. Do not invent figures or
rename tables during execution. If figures are later required, generate them
downstream from the retained CSV tables without rerunning provider calls.

Do not overwrite `states/frozen_state_definitions.json`.

---

## 11. Analysis and decision rule

The run command automatically creates the observation table, summary tables,
report, and sealed manifest after all calls complete.

The compute agent must additionally report:

```text
S2 - S0 truth-rate improvement
```

The implemented tables provide every input needed to calculate it. Do not
rerun provider calls for this subtraction.

Use the scientific rule from this handoff, even if the implementation's
generic report label differs:

### PASS

Use when:

```text
S2 >= 0.80
S2 - S0 >= 0.25
all semantic and memory checks pass
S0 remains a weak/private condition
```

### BORDERLINE PASS

Use when S2 clearly improves over S0 and semantic checks pass, but S2 is
modestly below 0.80 or prompt degradation is moderate.

### FAIL

Use when S2 does not substantially outperform S0, the later-round board prompt
destroys near-full solvability, or semantic/memory checks fail.

Do not redesign automatically after a fail.

The development smoke had 12/12 parsed logical calls and passed as an
engineering test. Its 100% state rates are not scientific estimates and must
not be combined with the full run.

---

## 12. Final verification

After completion, verify:

```text
manifest status = complete
execution scheduled = 360
execution successful = 360
observation rows = 360
state summary rows = 3
task-state rows = 18
task-agent-state rows = 72
all retained artifact hashes match
manifest self-hash matches
```

Also confirm:

```text
configured provider concurrency <= 30
configured RPM cap = 500
observed peak concurrency <= 30
no task/evidence/state files changed
```

`runtime/provider-control/` is mutable execution state and is intentionally not
part of the sealed scientific artifact hashes.

---

## 13. Final execution summary

At completion print:

```text
logical calls completed
actual provider attempts
configured local workers
configured provider concurrency
observed peak concurrency
configured RPM cap
observed sustained RPM
wall-clock time

S0 truth rate and 95% CI
S1 truth rate and 95% CI
S2 truth rate and 95% CI
S2-S0 improvement

semantic checks PASS/FAIL
overall PASS / BORDERLINE PASS / FAIL
results directory
report path
```

Also state explicitly:

```text
no tasks, evidence, frozen states, or assignments were regenerated
no study-specific SLURM job was added
```

The deliverable is the **full scientific blackboard prompt-validation result**
using the already frozen and smoke-tested harness.
