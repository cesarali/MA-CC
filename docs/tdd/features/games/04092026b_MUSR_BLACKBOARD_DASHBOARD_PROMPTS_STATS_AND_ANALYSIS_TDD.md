# TDD: MUSR Blackboard Dashboard Prompt Samples, Cell Statistics, and Analysis

**Date:** 2026-09-04  
**Status:** implementation plan  
**Scope:** study-wide and episode MUSR blackboard dashboard

## 1. Motivation

The current dashboard blocks while loading
`musr_blackboard_population_01`; the hierarchy can take roughly 102 seconds to
materialize, so the user cannot reliably open even one cell. The dashboard also
lacks a lean view of the exact prompts agents receive, useful cell-level outcome
and intervention summaries, and direct access to the already-computed study
analysis and thermodynamic products.

This plan extends
`04092026_MUSR_BLACKBOARD_DASHBOARD_LAZY_LOADING_TDD.md`. It does not change the
game, controller, estimators, or scientific observations.

## 2. Delivery sequence

Implement in two independently testable phases.

1. **Optimization and prompt samples:** unblock the hierarchy, lazy-load
   episode detail, and retain exactly three representative prompt examples per
   newly executed cell.
2. **Statistics and analysis:** add cell/episode summaries, aggregated
   scientific plots, thermodynamic results, and downloads.

Phase 1 must ship first. Phase 2 must not delay the dashboard performance fix.

# Phase 1: Optimization and lean prompt inclusion

## 3. Fast hierarchy and lazy detail

Use the lightweight cached index and lazy episode endpoint specified in the
companion TDD:

- the initial study endpoint reads manifests, expected episode plans, cell
  seals, and compact completion markers only;
- it never parses every `dashboard_semantic.jsonl` file;
- semantic timelines are read only when the user opens an episode;
- completed studies use a small invalidatable metadata cache;
- live studies refresh scheduler state and changed cells only;
- the initial 39-cell/390-episode hierarchy must render in under five seconds.

The overview and detail endpoints must fail independently. One corrupt or
unavailable episode detail must not prevent the cell list from rendering.

## 4. Three exact prompt examples per cell

For future executions, retain exactly three representative **rendered prompt
examples per scientific cell**, chosen at:

1. beginning: first eligible model request in round 1;
2. middle: first eligible request in the middle round
   (`ceil(rounds / 2)`);
3. end: first eligible request in the final round.

Use a deterministic representative episode and request order: lowest planned
repetition index, then lowest update/micro-slot index, then stable agent ID.
If that episode fails before a sampling point, select the next-lowest completed
repetition deterministically. Retries do not create additional examples; retain
the final valid attempt associated with the selected logical request.

Each sample contains only:

- scientific cell ID and source config;
- episode/repetition ID;
- round, update/micro-slot, and agent ID;
- condition/controller role and resolved public game parameters;
- exact rendered prompt sent to the model;
- prompt template/schema version and content hash;
- model/provider identity;
- whether repair guidance was included.

Raw responses and private chain-of-thought/reasoning are not part of this
three-prompt artifact. Existing detailed-audit policy remains authoritative if
raw responses are retained elsewhere.

Write one small atomic cell artifact, for example:

```text
cell/dashboard_prompt_examples.json
```

Do not write one prompt log per request or duplicate the examples beneath every
episode. Cap the artifact at three samples. Include its schema/version and hash
in the cell seal or cell operational provenance without making remote Comet
delivery part of scientific completion.

## 5. Existing-run compatibility

The current study did not retain enough information to reconstruct every exact
rendered prompt. Therefore:

- all current cells and episodes must remain visible;
- existing semantic episode views must work without prompt samples;
- show `Prompt examples unavailable: not retained by this run` where absent;
- never fabricate prompts from templates and current state;
- the absence of prompt examples must not mark a completed episode/cell
  incomplete;
- newly extended/retried cells can expose prompt samples without requiring old
  cells to be rerun.

## 6. Prompt UI

Add a cell-level **Prompt examples** panel with three labeled cards:
`Beginning`, `Middle`, and `End`. Each card shows compact provenance first and
the exact rendered prompt in a collapsible, copyable monospace block. Keep the
parameter panel independently collapsible. Fetch this artifact only when the
panel is opened.

# Phase 2: Cell statistics, reports, and thermodynamics

## 7. Cell overview statistics

For each cell, calculate compact summaries from canonical episode/round/micro
data or already-generated analysis tables. Display raw counts as well as
fractions.

### 7.1 Outcomes across episodes

- completed, failed, and aborted episodes;
- final unique winner: truth, controller target, other option, and tie;
- truth-win and controller-target-win counts, e.g. `7/10`;
- final and late-time truth/target share: mean, median, standard deviation, and
  interquartile range;
- final/late active phi and kappa where retained;
- episode duration and provider-request summaries where available.

Labels must follow the actual condition. A truth-aligned controller target must
not be double-counted as two independent outcome classes; show both semantic
roles while reporting the unique final option once.

### 7.2 Episode control activity

For each episode show:

- total microscopic updates;
- controlled microscopic updates;
- controlled-update fraction;
- controller `ADVOCATE` and `NO_OP` counts;
- recommendation and fact-sharing counts where retained;
- fact acquisitions, reactivations, and deactivations;
- validation-repair count and terminal malformed-response status;
- final winner, final truth share, final controller-target share, active phi,
  and active kappa.

Add compact distributions or sortable tables across repetitions. Do not expose
provider payloads, credentials, private reasoning, or heavy execution logs.

## 8. Additional plots

Inside a cell provide:

- existing per-episode time series;
- cell trajectory summary with median and interquartile band across episodes;
- final-winner count/fraction chart;
- controlled-update counts/fractions by episode;
- final and late-time truth/target-share distributions;
- active phi/kappa summaries where supported;
- compact fact acquisition/reactivation/deactivation summaries.

Use clear `n` and support labels. Do not imply precision beyond the repetition
count, and do not interpolate missing or unsupported states.

## 9. Aggregated scientific analysis

When `<study>/analysis/validation.json` exists and is valid, add a study-level
**Analysis** section sourced from the canonical aggregation outputs—not from a
second dashboard estimator implementation.

Expose:

- configured phase diagrams and scientific plots;
- primary estimates and derived observables;
- information estimates and support diagnostics;
- thermodynamic/efficiency outputs including the repository-defined chi,
  transfer/information quantity, eta_IR or eta_IF naming used by the recipe,
  eta_th, currents, h_eff, and gamma_eff where present;
- bootstrap confidence summaries and compact null summaries;
- validation, summary report, methods report, and provenance.

Use estimator metadata from `analysis_manifest.json` and reports to map display
labels to repository estimator names. Do not invent paper-symbol mappings or
recompute MI/CMI, nulls, bootstrap, susceptibility, or thermodynamic quantities
in the dashboard.

Unsupported metrics must be hidden or marked unavailable with their support
reason. Plots continue to respect support masks.

## 10. Downloads

Provide explicit read-only downloads for:

- the final analysis ZIP;
- validation and concise reports;
- canonical result/estimator CSV tables;
- individual configured plot files.

Resolve files beneath the validated study analysis directory only. Prevent
path traversal and arbitrary filesystem access. Do not expose run trees,
SLURM logs, provider logs, caches, credentials, or unretained prompt/response
content.

The dashboard must not rebuild aggregation automatically. If analysis is absent
or invalid, show the exact status and the normal `mas-cc study aggregate`
command needed to create it.

## 11. API shape

Keep payloads separate and lazy:

- study index: hierarchy and compact statuses;
- cell summary: aggregate episode statistics and plot-ready compact arrays;
- episode detail: one semantic timeline and episode statistics;
- prompt examples: at most three samples for one cell;
- analysis catalog: validation, available tables/plots/reports, estimator
  metadata, and safe download identifiers;
- file download: allowlisted analysis artifact only.

Do not return canonical rounds/micro-slots or all episode timelines in a single
JSON response.

## 12. Tests

### Phase 1

1. The overview does not parse semantic timelines.
2. A 39-cell/390-episode fixture renders within the five-second bound.
3. Episode detail is fetched only after selection.
4. One broken episode cannot break the hierarchy.
5. Exactly three deterministic prompt samples are retained per new cell.
6. Retry attempts do not exceed the three-sample cap.
7. Failed representative episodes fall back deterministically.
8. Existing cells without prompts show a graceful unavailable state.
9. Prompt artifacts contain no raw response, private reasoning, or secret.
10. Prompt samples are not uploaded to Comet.

### Phase 2

11. Winner/tie counts equal canonical episode outcomes and sum to completed
    episodes.
12. Truth-aligned target roles are represented without double counting.
13. Controlled-update and controller-action counts equal canonical micro data.
14. Cell summary statistics match trusted fixture calculations.
15. Existing episode and cell time-series behavior remains available.
16. Analysis values are read from canonical result tables without recomputation.
17. Unsupported estimates retain their support status.
18. Downloads are restricted to allowlisted analysis artifacts.
19. Traversal and arbitrary-path download attempts fail.
20. Missing/invalid analysis is reported without breaking experiment views.
21. Dashboard caches and prompt samples do not enter the scientific analysis
    ZIP unless prompt examples are explicitly declared as minimal provenance;
    the normal package remains lean.

## 13. Acceptance criteria

- The current study opens to its cell hierarchy in under five seconds.
- Users can drill from study to cell to any retained episode without loading
  unrelated semantic timelines.
- New cells retain exactly three deterministic prompt examples; old cells work
  without them.
- Cell views show truth/controller/other/tie outcomes and meaningful control,
  epistemic, and trajectory summaries.
- Completed aggregation, thermodynamic products, reports, plots, tables, and
  final ZIP are visible and safely downloadable.
- No scientific estimator, runtime/game/controller behavior, or existing
  canonical data contract changes.
