# TDD: Correct the Study-Wide MuSR Blackboard Dashboard

**Date:** 2026-09-03  
**Status:** corrective implementation plan  
**Priority:** restore episode inspection and truthful live status before adding presentation features

## 1. Problem statement

The first study-wide dashboard implementation does not satisfy the intended
extension contract. It added a study and cell view, but regressed the most
important existing behavior and currently presents misleading status.

Observed failures against the live blackboard population study:

1. A running episode cannot reliably be opened in the existing detailed
   episode inspector.
2. The study header reports `0/270 episodes complete` despite durable completed
   episode work being present.
3. Episode vote sparklines are embedded directly in the episode-list table,
   making the navigation/status view crowded and mixing two separate tasks.
4. The full flattened configuration is always expanded on the cell page and
   cannot be collapsed again.
5. Operational SLURM state and scientific episode state are insufficiently
   separated, which makes the displayed totals confusing during recovery and
   resume runs.

These are acceptance-blocking defects. The dashboard must not be described as
complete until they are corrected.

## 2. Product principle

The study-wide feature must be an outer navigator around the established
episode dashboard:

```text
study overview
    -> cell overview
        -> episode selector
            -> original detailed episode dashboard
```

Opening an episode must restore the same controls and inspectors that worked
before the study extension: round/update navigation, overview, blackboard,
evidence, agent, controller, prompt/response, validation/retry information,
and live follow.

Do not redesign or reduce the existing episode inspector. The study layer
selects an episode and hands its canonical directory to the existing
`BlackboardRunReader`.

## 3. Diagnose path resolution first

The current study reader assumes compact artifacts are directly beneath an
execution shard, for example:

```text
<shard>/data/episodes/<episode>
<shard>/round_records/<episode>
<shard>/cell_summary.json
```

Real standardized shards may contain the ordinary nested run hierarchy:

```text
<shard>/<game>/<experiment>/<dated-run>/cells/<cell>/...
```

Before changing the UI, document the exact live paths for:

- execution shard root;
- discovered run root;
- discovered scientific cell root;
- round-record episode root;
- full episode/audit root;
- episode manifest;
- cell summary and completion seal;
- results-only resume artifact.

Use existing standardized study discovery functions and manifest identities to
resolve these paths. Do not infer the scientific cell merely from directory
depth or use a recursive “first match” search.

Create one canonical `ResolvedDashboardCellPaths` object that carries the
actual paths needed by status, vote-series extraction, and episode inspection.
All three features must use the same resolution result.

## 4. Restore the existing episode inspector

Every episode with sufficient retained full-dashboard artifacts must expose an
`Inspect episode` action.

That action must:

1. resolve the exact episode/run directory safely;
2. instantiate the existing `BlackboardRunReader`;
3. switch to the original episode shell;
4. populate the existing timeline and agent controls;
5. show all original detailed tabs;
6. support live follow while the selected episode is still growing;
7. provide reliable Back to cell and Back to study navigation.

If detail is unavailable because the artifact profile genuinely does not
retain required data, state the missing artifact explicitly. “Detail
unavailable” must never be caused by looking in the wrong directory.

Add an integration test that starts from a standardized study root, selects a
running cell, opens a running episode, and compares its detailed snapshot with
loading the same episode directly through `BlackboardRunReader`.

## 5. Correct completion and progress accounting

Display three clearly named layers instead of one ambiguous status:

### 5.1 Scientific episode outcomes

From canonical episode manifests, summaries, seals, and established resume
semantics:

- completed;
- failed;
- aborted;
- incomplete/unknown.

Recovered successful episodes must count as completed even before every other
episode in their cell finishes, if the repository's durable episode contract
marks them reusable. A cell seal may certify the whole cell, but lack of a
cell seal must not erase known durable per-episode completion.

### 5.2 Live execution activity

From growing retained records and explicit runtime state:

- actively advancing episode;
- started but currently unchanged;
- not started.

Do not classify every episode in a running SLURM shard as running. A shard may
run several episode workers, resume completed episodes, or contain failed
attempts.

### 5.3 Scheduler state

Show SLURM task status separately:

- running, pending, held, or terminal;
- array index, node, and elapsed time.

Scheduler state annotates the cell; it does not determine episode completion.

The study headline should use explicit wording such as:

```text
17 durable episodes complete · 40 actively running · 4 SLURM cell tasks active
```

Never present `0 complete` when completed episode manifests are discoverable.

## 6. Separate navigation from trajectory analysis

### 6.1 Episode list

The default cell page should contain a compact episode table only:

- repetition;
- episode ID;
- seed;
- durable status;
- live progress/current round;
- last update time or elapsed time where supported;
- Inspect button.

Do not render a chart inside every episode row.

### 6.2 Trajectories view

Move vote dynamics into a dedicated `Trajectories` tab or panel on the cell
page. It should provide:

- all episode vote trajectories as selectable overlays or small multiples;
- truth and controller-target shares with an explicit legend;
- repetition filters;
- optional descriptive cell mean, labeled with the available episode count;
- a click from a trajectory to the corresponding episode inspector.

The episode list answers “what is running and what can I inspect?” The
trajectory view answers “how are votes evolving?” They must not compete for the
same table space.

## 7. Make parameters concise and collapsible

Show a small primary parameter summary by default:

- controller condition;
- `rho`;
- `b`;
- task;
- population size;
- rounds;
- relevant controller target and truth;
- other actual sweep coordinates.

Place the complete resolved configuration inside a native `<details>` element
or an accessible equivalent:

```html
<details>
  <summary>All resolved parameters</summary>
  ...
</details>
```

It must open and close repeatedly, support keyboard operation, and preserve its
state during the two-second live refresh. Refresh must update the contents
without forcibly reopening the section.

Avoid duplicating the same value under flattened short and fully-qualified
keys in the default summary. Full provenance may retain all keys inside the
collapsed detail.

## 8. Refresh and navigation behavior

The two-second polling loop must not reset user interface state.

Preserve across refresh:

- current study filters and sorting;
- selected cell;
- selected episode;
- selected episode tab;
- selected round/update and agent when live follow is disabled;
- trajectory selections;
- parameter disclosure open/closed state;
- scroll position where practical.

Refreshing a cell should update its model in place. It must not reconstruct the
entire page and destroy disclosure or selection state.

Use URL/hash state sufficient to reopen a selected cell and episode after a
browser refresh.

## 9. Required implementation order

1. Add a realistic nested standardized-study fixture matching the live result
   layout.
2. Fix canonical shard/run/cell/episode path resolution.
3. Fix per-episode durable completion accounting.
4. Fix active episode versus active SLURM task accounting.
5. Restore study-to-episode routing through `BlackboardRunReader`.
6. Add end-to-end API tests for a running episode.
7. Simplify the default cell episode list.
8. Move trajectories into a separate cell view.
9. Add compact parameters plus persistent collapsible full parameters.
10. Verify refresh-state preservation.
11. Smoke against the live study and the original single-episode pilot.

Do not begin visual polish until steps 1–6 pass.

## 10. Required tests

Add or update tests for:

1. nested standardized shard/run/cell path resolution;
2. multiple configs containing identically named local cells;
3. direct episode loading versus study-selected episode equivalence;
4. opening a running episode in the complete original inspector;
5. opening a completed episode in the complete original inspector;
6. explicit missing-artifact reason for genuinely compact-only episodes;
7. completed episode counting before cell sealing;
8. recovered/resumed completed episode counting;
9. failed attempt followed by successful retry/resume;
10. active episode detection without marking all episodes in a shard active;
11. scheduler running, pending, and held state shown independently;
12. truthful study and cell headline totals;
13. episode table contains no embedded trajectory plots;
14. dedicated trajectory view retains all episode series;
15. partial descriptive means state their numerator/denominator;
16. primary parameter summary contains only relevant scientific coordinates;
17. full parameter section can open and close;
18. disclosure state survives polling refresh;
19. cell/episode/tab/cursor state survives appropriate refreshes;
20. original direct single-episode dashboard regression behavior;
21. source artifacts remain unchanged.

## 11. Acceptance criteria

The correction is complete only when:

1. The live study shows scientifically correct completed/running/pending/failed
   counts.
2. The difference between episode state and SLURM task state is obvious.
3. Any suitably retained running episode can be opened and inspected with all
   previous episode-dashboard functionality.
4. Direct and study-selected views of the same episode produce equivalent
   timeline and snapshot data.
5. The episode table is compact and contains no trajectory chart column.
6. Vote time series appear in a separate trajectories view.
7. Full parameters are collapsed by default, can be hidden again, and remain
   stable during refresh.
8. The dashboard remains read-only and performs no execution control.
9. Focused tests and live-study smoke tests pass.

## 12. Non-goals

- changing game or controller behavior;
- changing retention, resume, aggregation, or estimator semantics;
- adding job-control buttons;
- treating SLURM state as scientific truth;
- dropping the original episode inspector;
- placing analytical plots in the episode navigation table;
- concealing inconsistent artifacts by coercing them into a plausible status.

## 13. Handoff report

After implementation, report:

- the exact path-resolution bug and fix;
- before/after episode-state counts on the live study;
- proof that a running episode opens in the original detailed inspector;
- the new cell page layout;
- refresh-state behavior;
- tests and live smoke results;
- confirmation that no scientific or execution artifacts were modified.
