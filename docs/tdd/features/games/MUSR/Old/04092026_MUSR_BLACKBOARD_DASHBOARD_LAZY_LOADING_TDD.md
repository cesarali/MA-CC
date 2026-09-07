# TDD: Fast Study Dashboard Index and Lazy Episode Loading

**Date:** 2026-09-04  
**Status:** implementation plan  
**Scope:** MUSR blackboard study-wide dashboard only

## 1. Problem

The completed `musr_blackboard_population_01` dashboard takes approximately
102 seconds to answer its initial study request. The current study endpoint
walks the complete lineage and reads semantic trajectory data for hundreds of
episodes before the overview can render.

The overview needs only study, condition, cell, parameter, episode-status, and
latest-progress metadata. Full blackboard, epistemic, prompt/response, and
time-series records are needed only after the user opens one episode.

## 2. Goal

Make the hierarchy responsive without removing any existing inspection
capability:

- show all configs, cells, parameters, and episode statuses quickly;
- preserve navigation from study to cell to episode;
- load one episode's semantic timeline only when that episode is selected;
- cache stable completed-run metadata;
- continue refreshing genuinely live episodes;
- do not add heavy logs or copy scientific data;
- do not change game, runtime, retention, or analysis behavior.

## 3. API separation

Split the current eager study response into two responsibilities.

### 3.1 Lightweight study index

The overview endpoint returns only compact metadata:

- study/config/cell identity;
- resolved scientific coordinates;
- expected and discovered episode IDs;
- scheduler-backed cell/episode status;
- completion state;
- latest round/update when cheaply available;
- whether semantic detail is available;
- a reason when detail is unavailable.

It must not read or serialize complete semantic JSONL trajectories.

### 3.2 Lazy episode detail

An episode-detail endpoint loads only the selected episode and returns the
existing:

- visible state and epistemic state;
- blackboard/message evolution;
- controller and peer events;
- exact prompt/raw response where the retention profile permits them;
- compact semantic records where full content was not retained;
- episode time-series trajectory;
- validation and malformed-response diagnostics allowed by retention.

Changing tabs inside an already loaded episode should reuse the fetched
episode payload rather than repeat disk reads.

## 4. Index construction and caching

Build the hierarchy from authoritative manifests, extension target manifests,
execution manifests, cell seals, and compact completion markers. Respect the
persisted lineage cell and episode identities; do not re-derive them from
mutable source configs.

Maintain a small in-memory cache keyed by study root and a deterministic
fingerprint of relevant manifests/seals. For a completed study, the cached
index may be reused until one relevant file changes. For a live study, refresh
only scheduler state and cells whose marker modification times changed.

Optionally persist a small atomic dashboard index beneath a runtime/dashboard
directory if startup speed requires it. It must contain metadata only, be
rebuildable, and never enter the scientific analysis ZIP.

Do not cache full episode semantic payloads globally. A small bounded LRU of
recently opened episode details is acceptable.

## 5. Front-end behavior

- Render the hierarchy immediately after the lightweight index arrives.
- Fetch episode details only after the user selects an episode.
- Show an explicit loading indicator in the episode panel.
- Keep the selected config, cell, episode, and tab stable during refresh.
- Do not replace the whole hierarchy while one episode is loading.
- Report `Completed`, `Running`, `Failed`, or `Not started` consistently;
  completed studies must not appear actively advancing merely because old
  semantic files exist.
- Keep cell parameter panels collapsible.
- Keep time-series plots inside the selected episode view as well as any
  aggregate cell view.

## 6. Mixed-retention lineage behavior

The current lineage contains 390 episodes: 297 have the newer semantic
dashboard records and 93 reused episodes predate that retention profile.

- All 390 must appear in the hierarchy.
- The 297 detailed episodes must remain inspectable.
- Older episodes without semantic records must show a truthful unavailable
  explanation, not a broken panel or `Not started` if scientific completion is
  known from canonical data/seals.
- Do not fabricate prompts, raw responses, or epistemic histories absent from
  retained files.

## 7. Performance targets

On a completed 39-cell/390-episode study:

- initial index response: target under 3 seconds, hard acceptance under 5;
- cached index response: target under 500 ms;
- opening one retained episode: target under 2 seconds for ordinary files;
- refresh work proportional to changed cells/episodes, not total semantic
  trajectory size;
- memory remains bounded independently of the number of episodes opened over
  the server lifetime.

## 8. Tests

1. The overview endpoint does not call the full episode timeline loader.
2. The index lists every config, cell, and expected episode across extensions.
3. Persisted lineage identities remain authoritative.
4. Selecting an episode triggers exactly one detail request.
5. Re-selecting a cached episode avoids redundant parsing.
6. Cache invalidation occurs when a relevant seal/manifest changes.
7. Live refresh touches only changed cells plus scheduler state.
8. Completed episodes without semantic retention are labeled completed with
   detail unavailable.
9. Existing detailed episode panels and trajectory plots remain functional.
10. Prompt/raw-response visibility continues to follow retention policy.
11. A 39-cell/390-episode fixture meets the response-time acceptance bound.
12. Dashboard runtime caches are excluded from scientific packages.

## 9. Acceptance criteria

- The full study hierarchy becomes usable within five seconds.
- Episode semantic data are loaded lazily and remain scientifically unchanged.
- All 390 episodes are represented with correct completion status.
- Mixed old/new retention is explained accurately.
- No scientific files, estimators, configs, or SLURM execution behavior change.
- No persistent heavy cache or duplicated trajectory data is introduced.
