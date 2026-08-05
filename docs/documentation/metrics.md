# Metrics reference: what exists, and how to call each one

This is a **reference**, not a tutorial: every metric the naming-convention game computes today,
explained in plain terms, and exactly which config turns each one on. Want to add a new metric
instead? See [`docs/howto/building/building_a_metric.md`](../howto/building/building_a_metric.md).

**Accompanying config:**
[`configs/runs/naming_convention_tutorial_university_v3.yaml`](../../configs/runs/naming_convention_tutorial_university_v3.yaml) —
already sets up every metric below; see its `metrics:` section for a working example.

## Table of contents

1. [The metrics that exist today](#1-the-metrics-that-exist-today)
2. [How to see any of them](#2-how-to-see-any-of-them)
3. [Where to go next](#3-where-to-go-next)

---

## 1. The metrics that exist today

All eight live in one file: `src/mas_cc/games/naming_convention/metrics.py`. This is genuinely all
of them — nothing else is available unless someone adds it (see the building guide).

| Name | Kind | What it actually tells you |
| --- | --- | --- |
| `population_action_share_q` | live, whole population | Of everyone who has played at least once, what fraction's *most recent* action was Q. |
| `population_action_share_m` | live, whole population | Same, for M. |
| `dominant_action_share` | live, whole population | Whichever of Q/M is currently more common, what share is it? (Always ≥ 0.5.) |
| `agent_current_action` | live, one number per agent | Literally: what did this specific agent play most recently? (Nothing yet → blank.) |
| `first_consensus_time` | end-of-episode summary | Which round was the *first* time 95% of the population agreed on the same action? Blank if that never happened. |
| `rolling_coordination_rate` | live, whole population | Of just the **last few** interactions (not the whole episode) — how many were successful matches? A short-term "how are things going right now" signal, separate from the episode-wide trend. |
| `rolling_action_share_q` | live, whole population | Same short recent window, but "what share of recently-played actions were Q." |
| `rolling_action_share_m` | live, whole population | Same, for M. |

**Live vs. rolling, the distinction that matters:** `population_action_share_q` looks at every
agent's *current* stance, no matter how long ago they last played — once someone settles on Q they
count as Q forever, until they change. The `rolling_*` metrics instead only look at what actually
got played in roughly the last "one interaction per agent" stretch — they can react and wobble
early on, while the population-wide ones are stickier and smooth out over the whole episode.

---

## 2. How to see any of them

Every metric, once turned on, is always written to files — `metrics/streaming.csv` for the live
ones, `metrics/final.csv` for the end-of-episode ones — not optional, and cheap. **Live,
whole-population metrics also always get a plot** (`metrics/plots/<name>.png`) — that's six of the
eight above (everything except `agent_current_action`, which is per-agent, and
`first_consensus_time`, which is an end-of-episode summary, not a per-round line to plot). What you
choose is whether a metric *also* prints to your screen or goes to Comet:

```yaml
metrics:
  enabled: true          # off entirely = none of this happens, nothing computed at all
  available:
    population_action_share_q:
      comet: true         # this one also goes to Comet, every round
    rolling_coordination_rate:
      comet: false        # this one stays local: file + plot only, no Comet
logging:
  options:
    show_metrics: true    # print every metric's value once, when the episode finishes
```

Run `mas-cc game episode --config <path>` and check the end of its output — it tells you exactly
how many plots got written and exactly which metric names went to Comet, so you don't have to trust
the config file alone. See
[`creating_a_game_config.md` §1](../howto/configuring/creating_a_game_config.md#1-which-metrics-you-get--and-the-one-youre-probably-looking-for-that-isnt-here)
for the full config recipe.

---

## 3. Where to go next

- **Adding a metric that isn't on this list:** [`building_a_metric.md`](../howto/building/building_a_metric.md).
- **Everything else about the config:** [`creating_a_game_config.md`](../howto/configuring/creating_a_game_config.md).
- **Running an episode day to day:** [`running_an_episode.md`](../howto/launch/running_an_episode.md).
- **The real files:** `src/mas_cc/metrics/{base,generic}.py` (the shared, cross-game library),
  `src/mas_cc/games/naming_convention/metrics.py` (this game's own adapter + rolling metrics),
  `src/mas_cc/observability/recorder.py` (what actually calls a metric and writes its value down).
