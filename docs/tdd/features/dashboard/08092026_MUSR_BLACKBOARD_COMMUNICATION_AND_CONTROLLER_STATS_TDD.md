# MuSR Blackboard Communication and Controller Statistics

**Date:** 2026-09-08  
**Status:** implementation plan

## Goal

Make communication and controller participation immediately visible in the
existing study → cell → episode dashboard without changing the game runtime,
controller behavior, scientific estimators, or retained-data contract.

The existing `dashboard_semantic.jsonl` records are authoritative. The work is
primarily dashboard normalization, aggregation, and presentation. Missing
historical fields must be shown as unavailable rather than inferred.

## Scope

Implement five focused additions:

1. general message-type statistics;
2. message-type totals by author category;
3. a controller-only blackboard filter;
4. a controller actuation/exposure timeline;
5. population vote shares with controller-event markers.

Do not add unrelated plots or persistent analysis caches.

## 1. Clarify the participant selector

Rename the episode-view `Agent` selector to `Inspect participant` and make its
default `All participants`.

The available scopes should be:

- all participants;
- participants only;
- controller only;
- one named participant.

The controller remains a distinct author category and must not be presented as
an ordinary population member. The selected scope filters communication views;
it must not silently change population-level scientific statistics.

## 2. General communication bars

Add a compact stacked bar chart by round for all authored communication:

- `REPORT`;
- `REQUEST`;
- `DIRECTIVE`;
- replies, when distinguishable;
- no-message actions.

Add adjacent totals for the whole episode. Both views must respond to the
author-scope selector. Counts must be derived from semantic events and deduped
by stable message/event identity so refreshes and resumed attempts do not
double-count records.

## 3. Author-category totals

Show grouped or stacked totals comparing:

- ordinary participants;
- controller.

At minimum report total messages and counts by message type. Also show messages
per observed round. Keep raw counts visible in tooltips or labels rather than
showing percentages alone.

## 4. Controller-only blackboard view

Add a visible toggle to the existing blackboard:

```text
All messages | Controller messages
```

The filtered view must retain the controller's exact stored text, message type,
round, selected/shared fact identifier, and reply target where available. It
must expose historical controller messages even after they expire from the
current board snapshot. Never reconstruct missing text from prompts or facts.

## 5. Controller tab

Keep controller-specific quantities in the existing Controller tab. Add small
summary cards for:

- controller rounds observed;
- `ADVOCATE_Z` count and fraction;
- `NO_OP` count and fraction;
- reports, requests, and directives;
- total posts;
- requested budget `b` versus realized posts;
- LLM fallback count;
- direct/recorded controller exposures, where available.

Add a round-level event strip using consistent colors:

- grey: `NO_OP`;
- blue: report;
- orange: request;
- red: directive.

Selecting an event opens the exact controller messages for that round.

Where supported, add three aligned per-round bars:

1. controller posts;
2. agents exposed to controller messages;
3. agents whose next recorded vote moved toward the controller target.

The third quantity is descriptive temporal response, not a causal estimate,
and must be labelled accordingly.

## 6. Vote-share plot with controller markers

In the episode view, retain the population time series and show stacked or
aligned shares for:

- truth;
- controller target;
- other options.

Overlay controller-actuation markers by message type. The plot must remain
usable when no controller exists and must distinguish missing data from zero.

Do not use financial candlestick charts. Discrete bars, stacked shares, and
event markers match these data more directly.

## Data and API changes

Extend the existing dashboard normalization/API layer rather than parsing files
in browser code. Return compact derived structures for:

- communication counts by round, author category, and message type;
- controller round events and posts;
- controller exposure counts;
- population shares and controller markers.

Use existing study and episode endpoints where practical. If a new endpoint is
needed, keep it episode-scoped and read-only. Cache by path, modification time,
and size using the existing dashboard cache; do not create persistent cache
files.

Live refresh must not navigate away from the selected cell, episode, tab,
participant, or blackboard filter. Manual refresh remains authoritative.

## Attempt and resume semantics

The dashboard must distinguish:

- historical failed attempts;
- a currently advancing retry;
- the final durable completed attempt.

Do not combine duplicated events from retries. Prefer stable attempt/event IDs.
If records cannot be attributed safely, show an explicit support warning rather
than presenting a misleading total.

## Implementation areas

Expected files are primarily:

- `src/mas_cc/blackboard_dashboard/study_data.py`;
- `src/mas_cc/blackboard_dashboard/server.py`;
- `src/mas_cc/blackboard_dashboard/assets/index.html`;
- `src/mas_cc/blackboard_dashboard/assets/app.js`;
- `src/mas_cc/blackboard_dashboard/assets/style.css`;
- dashboard-focused tests under `tests/mas_cc/`.

Do not modify production game/controller code merely to satisfy the dashboard.

## Verification

Add focused fixtures/tests proving:

1. message counts are correct by type, round, and author category;
2. controller messages are independently filterable;
3. expired controller messages remain visible historically;
4. requested and realized controller posts remain distinct;
5. exposure and subsequent-vote counts use the correct event ordering;
6. retries do not double-count prior events;
7. no-controller and results-only episodes degrade cleanly;
8. participant selection does not alter population statistics;
9. live/manual refresh preserves navigation and filters;
10. the existing detailed episode inspector remains functional.

Run the focused Python dashboard tests and `node --check` for the browser code.
Smoke-test against one completed DeepInfra study and one live study root.

## Acceptance criteria

The feature is complete when a user can enter any retained episode and answer,
without reading raw JSONL:

- how much communication occurred;
- how many reports, requests, and directives each author class produced;
- exactly what the controller said;
- when the controller acted or did nothing;
- how many controller posts were requested and realized;
- how often agents were exposed to controller messages;
- how population support changed around controller actions.

The dashboard must remain read-only, responsive, and scientifically explicit
about unavailable or merely descriptive quantities.
