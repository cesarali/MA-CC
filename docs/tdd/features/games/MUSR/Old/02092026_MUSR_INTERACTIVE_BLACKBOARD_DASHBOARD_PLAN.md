# Plan: Interactive MuSR Blackboard Dashboard

**Date:** 2026-09-02  
**Status:** implemented initial interactive version  
**Primary probe:** completed `task_001`, N=24, five-round blackboard pilot

## 1. Objective

Build a small interactive, read-only dashboard for inspecting a blackboard
episode during execution and after completion. The dashboard must make it easy
to move through rounds and microscopic updates, inspect the public blackboard,
compare active and historical evidence coverage, and inspect the exact decision
context recorded for any agent.

The dashboard is an observability layer only. It must not alter game state,
scientific identities, episode seeds, provider requests, prompts, controller
behavior, checkpoints, or estimator inputs.

The first version should remain deliberately lightweight. It does not need a
general analytics platform, a database, WebSockets, or a frontend framework.

## 2. Primary user questions

At any point in an episode, the interface should answer:

1. What round and microscopic update has been reached?
2. What does the public blackboard contain at this point?
3. Which messages have expired, and which are still visible?
4. What is the population vote distribution and truth-vote share?
5. What evidence is active or historical for each agent?
6. What exactly did a selected agent see?
7. What did that agent decide and publish?
8. What private rationale did the model return?
9. What did the controller sense and do?
10. Where did exact evidence move, disappear from active memory, or refresh?

The phrase **private model rationale** should be used in the UI instead of
"internal thoughts" or "chain of thought." It is the rationale explicitly
returned and recorded by the model, not access to an unobserved mental state.

## 3. Current run as the primary probe

Yes, the completed pilot is a strong development and acceptance probe. Its run
directory is:

```text
/work/ojedamarin/Projects/LanguageGames/MA-CC/results/studies/
  musr_blackboard_task001_pilot_01/
  relational_imitation_round_feedback/
  musr-blackboard-task001-pilot-01/
  musr-blackboard-task001-pilot-01-20260902/
```

It provides:

- 24 agents;
- 24 local-initialization decisions;
- five rounds and 120 microscopic social updates;
- 145 archived provider attempts, including one retry;
- full rendered prompts and raw responses;
- REQUEST, REPORT, and DIRECTIVE messages;
- controller actions and sensor observations;
- active and historical evidence snapshots;
- exact acquisitions, refreshes, and persistence deactivations;
- board creation and expiry metadata;
- a completed checkpoint and sealed episode manifest.

This run must be treated as read-only. Dashboard development must never rewrite
its scientific or audit artifacts. Generated dashboard files may be written to
a separate development/output directory until acceptance.

The probe supports deterministic replay and post-run UI testing without any new
provider calls. A separate tiny mock-provider episode should be used later to
test live-follow behavior.

## 4. User interface

### 4.1 Persistent navigation controls

The top of the page should contain:

- episode selector;
- round slider, including an explicit initialization phase;
- microscopic-update slider;
- agent selector;
- view selector;
- live auto-follow toggle;
- connection/run status indicator;
- last-refresh timestamp.

Moving the round or update control must update every visible panel to the same
logical point in the episode. Manual navigation should disable auto-follow
until the user enables it again.

Keyboard navigation is desirable:

- left/right: previous or next microscopic update;
- shift+left/right: previous or next round;
- space: pause/resume live auto-follow.

### 4.2 Overview view

At the selected time point, show:

- episode status;
- round and update;
- completed updates versus expected updates;
- vote counts by option;
- truth-vote share;
- live blackboard size;
- cumulative REQUEST, REPORT, and DIRECTIVE counts;
- exact acquisitions and refreshes;
- mean active and historical evidence counts;
- controller action, target, and sensed votes;
- provider-attempt, retry, invalid-response, and budget status.

This view should be compact and useful while the run is still executing.

### 4.3 Blackboard view

Render the blackboard as it existed at the selected round and microscopic
update. Each message should show:

- message ID;
- author and author role;
- message type;
- public text;
- shared evidence ID, when present;
- `reply_to`, when present;
- creation round/update;
- expiry round;
- whether it is live, newly posted, or expired at the selected time.

Required interactions:

- filter by REQUEST, REPORT, and DIRECTIVE;
- toggle expired-message history;
- highlight controller DIRECTIVEs;
- visually connect replies to their parent messages;
- click an author to open the corresponding agent inspector;
- click an evidence-bearing message to highlight its transfer history.

Private reasoning must never appear on the public-blackboard panel.

### 4.4 Evidence coverage view

Render an interactive agent-by-latent matrix, initially 24 by 9 for the pilot.
Support:

- active evidence;
- historical evidence;
- active and historical side by side;
- changed-since-previous-update overlay;
- newly acquired evidence;
- refreshed evidence;
- evidence removed from active memory by persistence;
- hover/click details for agent, latent, exact evidence ID, and event source.

The view should also display population-level active and historical latent
coverage and mean evidence counts. It should be derived from retained state and
events, not from a new scientific estimator.

### 4.5 Agent inspector

For the selected agent at the selected time point, display:

- current and previous vote;
- whether the agent is focal at this update;
- active evidence IDs and rendered evidence text;
- historical-but-inactive evidence IDs;
- currently visible blackboard source(s);
- current controller exposure, if any;
- exact compiled prompt messages;
- raw model response;
- parsed vote;
- recorded private model rationale;
- parsed public message;
- validation result and retry history;
- evidence acquired, refreshed, shared, or forgotten;
- a compact timeline of the agent's decisions across the episode.

If the selected agent was not asked to decide at the selected update, the UI
should say so and show the most recent prior decision rather than implying that
a new decision occurred.

### 4.6 Controller inspector

For each round, show:

- sampled agents and sensed votes;
- controller statistic/state;
- probability of intervention;
- sampled action;
- target;
- controlled positions;
- generated DIRECTIVE text and message IDs;
- agents that sampled the directive;
- replies to the directive;
- downstream exact evidence movement.

The controller panel should explicitly distinguish a stochastic no-action from
a disabled controller or an unavailable record.

## 5. Timeline and state semantics

Use a single canonical cursor:

```text
episode + phase + round_index + within_round_index/global_update_index
```

Initialization is a separate phase and must not be misleadingly presented as a
normal social round. A social-update cursor determines all downstream panels:

```text
selected cursor
    -> population and votes
    -> public blackboard
    -> active/historical evidence
    -> controller state
    -> selected agent's decision context
```

State reconstruction must define whether each panel represents the state before
or after the selected update. The UI should offer a clear before/after toggle or
choose one default and label it everywhere. The recommended default is **after
selected update**, with before-state fields available in the detail drawer.

## 6. Data sources

Reuse existing append-only and checkpoint artifacts:

- `trajectory.jsonl` for microscopic state transitions;
- `round_trajectory.jsonl` for round summaries and persistence transitions;
- `audit_traces.jsonl` for compiled prompts, responses, parsed decisions, and
  validation attempts;
- `api_call_status.jsonl` for provider-attempt status;
- `events.jsonl` for operational/scientific events;
- `.checkpoints/checkpoint.json` for the latest durable state;
- episode and run `manifest.json` files for lifecycle status;
- `usage_cost.jsonl` and `budget_events.jsonl` for live budget status;
- the resolved config and initial-assignment artifact for provenance.

The existing reconstruction and normalization logic in
`pilot_artifacts.py` should be extracted into reusable, side-effect-free reader
functions. The final static artifact builder and live server must call the same
reader layer so they cannot silently disagree.

## 7. Dashboard snapshot schema

Define and version one JSON snapshot contract. A suggested top-level form is:

```json
{
  "schema_version": 1,
  "run": {},
  "cursor": {},
  "population": {},
  "blackboard": [],
  "coverage": {},
  "agent": {},
  "controller": {},
  "evidence_events": [],
  "provider_status": {},
  "available_cursors": []
}
```

The schema should retain scientific IDs and exact integer indices. Human labels
belong in the view layer. Missing/incomplete live data should be represented
explicitly rather than replaced with zeros.

## 8. Live server

Add a read-only CLI such as:

```bash
mas-cc blackboard dashboard \
  --run-dir <run-directory> \
  --host 127.0.0.1 \
  --port 8765 \
  --follow
```

Recommended first implementation:

- Python standard-library HTTP server;
- vanilla HTML, CSS, and JavaScript;
- JSON endpoints;
- browser polling every one or two seconds;
- no database;
- no WebSockets;
- no writes to scientific artifacts.

Suggested endpoints:

```text
GET /api/status
GET /api/timeline
GET /api/state?phase=round&round=2&step=14
GET /api/agent/agent_007?phase=round&round=2&step=14
GET /api/prompt/<audit-id>
GET /
```

The server should bind to `127.0.0.1` by default. Binding to all interfaces must
require an explicit unsafe/remote flag and a warning, because prompts and
private rationales are sensitive.

## 9. Safe incremental reading

Live writers may be appending while the dashboard reads. Readers must:

- consume only complete newline-terminated JSONL records;
- ignore a partial trailing line and retry it on the next poll;
- tolerate files that do not exist yet;
- detect truncation/replacement after resume;
- key records by stable identifiers so polling does not duplicate them;
- reopen atomic checkpoint replacements rather than retaining stale handles;
- never lock or delay the experiment writer;
- expose the latest durable cursor separately from the latest observed audit.

Malformed completed records should surface as dashboard errors. Incomplete
trailing writes should surface as a normal `waiting_for_writer` state.

## 10. Static export

After episode completion, reuse the same frontend and snapshot schema to create
a portable offline dashboard under:

```text
analysis/dashboard/
```

The completed export should include all required JSON data locally and must not
require the Python server. It should replace or subsume the current static
table-based `index.html`, while preserving a simple no-JavaScript summary or
the Markdown report as a fallback.

The existing output paths should remain stable where practical:

```text
analysis/dashboard/index.html
analysis/prompts/*.md
analysis/task001_pilot_report.md
```

## 11. SSH tunnel workflow

Start the server on Potsdam with the dedicated environment:

```bash
/home/ojedamarin/.local/share/miniforge3/bin/conda run -n MA-CC \
  mas-cc blackboard dashboard \
  --run-dir <run-directory> \
  --host 127.0.0.1 \
  --port 8765 \
  --follow
```

From the user's computer, establish a local tunnel:

```bash
ssh -L 8765:127.0.0.1:8765 ojedamarin@login1.hpc.uni-potsdam.de
```

Then open:

```text
http://127.0.0.1:8765
```

If the experiment/dashboard runs on a compute node, document the required
jump-host or two-hop forwarding arrangement. Do not expose the dashboard on a
public interface merely to avoid configuring the tunnel.

## 12. Implementation phases

### Phase A: reusable data model

1. Define the cursor and snapshot schemas.
2. Extract pure artifact-reading and normalization functions from
   `pilot_artifacts.py`.
3. Reconstruct blackboard liveness at any micro-step.
4. Reconstruct per-agent active/historical evidence at any micro-step.
5. Join audit attempts to their corresponding initialization decision or
   social update.
6. Represent incomplete live data explicitly.

### Phase B: completed-run explorer

1. Implement the overview, timeline controls, blackboard, coverage, agent, and
   controller views.
2. Develop entirely against the completed `task_001` probe.
3. Verify deterministic snapshots at selected known cursors.
4. Export a portable offline dashboard.

### Phase C: live-follow server

1. Add the localhost-only CLI server.
2. Add incremental readers and polling endpoints.
3. Implement connection state and auto-follow.
4. Verify resume, file appearance, partial lines, and atomic checkpoint
   replacement.
5. Confirm the dashboard has negligible effect on the experiment writer.

### Phase D: live mock probe

1. Run a tiny mock-provider blackboard episode.
2. Start the dashboard before the first artifact exists.
3. Observe initialization, round transitions, messages, evidence changes, and
   completion.
4. Compare live snapshots with the final static export.
5. Do not use a paid provider merely to validate dashboard mechanics.

### Phase E: acceptance against the real pilot

1. Replay the completed `task_001` run.
2. Verify all five rounds and 120 updates are selectable.
3. Verify all 24 agents are selectable.
4. Verify all 145 prompt attempts are reachable, including the retry.
5. Compare summary counts with `artifact_manifest.json` and the Markdown
   report.
6. Confirm no source artifact changed by hashing before and after inspection.

## 13. Tests

### Unit tests

- complete and partial JSONL parsing;
- cursor ordering and initialization semantics;
- before/after state reconstruction;
- board message creation, visibility, reply linkage, and expiry;
- active versus historical evidence reconstruction;
- acquisition, refresh, and persistence classification;
- audit-to-update and retry joins;
- controller no-action versus disabled distinction;
- stable snapshot serialization.

### API tests

- valid and invalid run directories;
- status before artifacts exist;
- state lookup at valid and invalid cursors;
- agent and prompt lookup;
- correct content type and schema version;
- localhost binding default;
- read-only behavior.

### UI tests

- round/update selectors synchronize every panel;
- manual selection disables auto-follow;
- agent selection updates prompts and decisions;
- active/historical toggle updates coverage;
- message filters and reply navigation work;
- loading, waiting, completed, failed, and malformed states are distinct;
- keyboard navigation works;
- useful layout at ordinary laptop widths.

### Probe assertions

For the current pilot, assert at least:

- five rounds;
- 120 social updates;
- 24 initialization decisions;
- 145 prompt attempts;
- 49 REQUEST messages;
- 58 REPORT messages;
- 12 DIRECTIVE messages;
- 43 exact acquisitions;
- nine refreshes;
- initial truth share `0.6667` within display precision;
- final truth share `0.4583` within display precision.

## 14. Security and privacy

- Bind to localhost by default.
- Do not add provider keys, authorization headers, or `.env` contents to API
  responses or static exports.
- Treat prompts, raw responses, and private rationales as sensitive research
  artifacts.
- Escape all model-generated text before inserting it into HTML.
- Apply a restrictive content-security policy to the generated page.
- Do not load JavaScript, fonts, or analytics from third-party CDNs.
- Make server endpoints read-only.
- Prevent path traversal in run, prompt, and artifact lookup.

## 15. Performance constraints

The pilot is small, but the design should not reread every large JSONL file on
every browser poll. Maintain an in-memory index keyed by file identity, byte
offset, and stable record ID. Rebuild only when files are replaced or
truncated.

The dashboard must not create a second scientific dataset or estimator cache.
Its index is transient operational state. The offline export may retain only
the normalized data required to render the selected run.

## 16. Acceptance criteria

The feature is complete when:

1. Any available round and microscopic update can be selected.
2. The blackboard matches recorded message liveness at that exact point.
3. Any agent can be selected and its latest/active decision context inspected.
4. Exact rendered prompts, raw responses, parsed outputs, and retry status are
   reachable.
5. Active and historical evidence are visibly distinct.
6. Acquisition, refresh, and forgetting events are inspectable.
7. Controller sensing, action, directives, readers, and replies are visible.
8. A running mock episode updates without restarting the dashboard.
9. A completed run works as a portable offline export.
10. The completed pilot reproduces its existing manifest/report counts.
11. Inspection does not modify source scientific artifacts.
12. The server listens only on localhost unless explicitly overridden.

## 17. Non-goals for the first version

- editing or steering a running episode;
- sending provider requests from the dashboard;
- changing controller policy or game parameters;
- replacing Comet or study-level operational monitoring;
- recomputing MI/CMI or other scientific estimators;
- comparing many heterogeneous studies in one browser session;
- authentication for public deployment;
- a general-purpose agent framework UI.

## 18. Recommended delivery

Deliver the feature as:

- reusable blackboard artifact readers and state reconstruction;
- a small dashboard server and CLI command;
- bundled local frontend assets;
- an upgraded static exporter;
- unit/API/UI tests;
- validation against the existing completed pilot;
- one live tiny mock-provider demonstration;
- brief tunnel and usage documentation.

No new scientific run, provider call, study-specific SLURM job, or replacement
estimator is required for implementation or acceptance.

## 19. Implemented interface and usage

The initial implementation is available through:

```bash
mas-cc blackboard dashboard \
  --run-dir <run-directory> \
  --host 127.0.0.1 \
  --port 8765
```

It provides synchronized round/update navigation, live-edge polling, agent
selection, overview cards, vote distributions, blackboard filtering and expiry,
active/historical coverage matrices, exact prompt and raw-response inspection,
private-model-rationale display, per-agent decision timelines, and controller
inspection.

A completed run can be exported without altering its source artifacts:

```bash
mas-cc blackboard export \
  --run-dir <run-directory> \
  --output-dir <dashboard-directory>
```

The export embeds every available cursor and every agent view, so round, update,
coverage, message, and agent controls work without a running Python process.

The primary implementation is under:

```text
src/mas_cc/blackboard_dashboard/
```

Focused tests are under:

```text
tests/mas_cc/test_blackboard_dashboard.py
```
