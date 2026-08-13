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
3. [The binned trajectory metrics](#3-the-binned-trajectory-metrics)
4. [Where to go next](#4-where-to-go-next)

---

## 1. The metrics that exist today

All eight live in one file: `src/mas_cc/games/naming_convention/metrics.py`. This is genuinely all
of them — nothing else is available unless someone adds it (see the building guide). Two further
metrics, computed per *bin* rather than per round, are described in [section 3](#3-the-binned-trajectory-metrics).

| Name | Kind | What it actually tells you |
| --- | --- | --- |
| `population_action_share_per_option` | live, one curve per option | For **each** option, what fraction of the population currently stands on it — counting every agent's *most recent* choice, over everyone who has chosen at all. The values sum to 1. |
| `dominant_action_share` | live, whole population | Whichever option is currently most common, what share is it? (Always ≥ 1/number of options.) |
| `agent_current_action` | live, one value per agent | Literally: what did this specific agent play most recently? (Nothing yet → blank.) |
| `first_consensus_time_by_action_share` | end-of-episode summary | Which round was the *first* time 95% of agents were **holding** the same action? Blank if that never happened. |
| `first_consensus_time_by_success_rate` | end-of-episode summary | Which interaction was the first where 95% of the **last 3N conversations succeeded**? The paper's criterion. Blank if never. |
| `consensus_action_by_success_rate` | end-of-episode summary | *Which* word won at that moment — a label like `Q`, not a number. Blank if consensus was never reached. |
| `rolling_coordination_rate` | live, whole population | Of just the **last few** interactions (not the whole episode) — how many were successful matches? A short-term "how are things going right now" signal, separate from the episode-wide trend. |
| `rolling_action_share_per_option` | live, one curve per option | Same short recent window, but "what share of recently-*played* actions was each option." |

**One metric, several curves.** The two `*_per_option` entries are each a *single* metric that emits
one value per option per round, not one metric per option. In `metrics/streaming.csv` that is one
row per option per round, with the option name in the **`series`** column; on Comet each option
arrives under its own suffixed key (`population_action_share_per_option_Q`, `..._M`). You list and
toggle it once, and it keeps working unchanged if you run the game with ten options instead of two.

**Two ways to say "they reached consensus."** `first_consensus_time_by_action_share` asks *are
enough agents holding the same word?*; `first_consensus_time_by_success_rate` asks *are recent
conversations succeeding?* They normally agree in the end but rarely on the same round, so each
name says which quantity it counts. The second one cannot tell you which word won on its own —
that is why `consensus_action_by_success_rate` exists beside it.

**Standing vs. rolling, the distinction that matters:** `population_action_share_per_option` looks at
every agent's *current* stance, no matter how long ago they last played — once someone settles on Q
they count as Q forever, until they change. That is what makes it a readable population statistic
when only two agents act per round. `rolling_action_share_per_option` instead only looks at what
actually got played in roughly the last "one interaction per agent" stretch — it reacts and wobbles
early, and it is the one that moves *first* when a convention starts to tip, while the standing
share is stickier and smooths out over the episode.

---

## 2. How to see any of them

Every metric, once turned on, is always written to files — `metrics/streaming.csv` for the live
ones, `metrics/final.csv` for the end-of-episode ones — not optional, and cheap. **Every live
metric that isn't per-agent also always gets a plot** (`metrics/plots/<name>.png`) — that's four of
the eight above (everything except `agent_current_action`, which is per-agent, and the three
end-of-episode summaries, which are not per-round lines to plot). A
`*_per_option` metric gets **all of its option curves on one axis**, with a legend — the point of
that plot is the curves crossing each other. Pass `separate_options=True` to
`plot_streaming_metrics` if you also want one file per option
(`population_action_share_per_option_Q.png`, `..._M.png`). What you choose is whether a metric
*also* prints to your screen or goes to Comet:

```yaml
metrics:
  enabled: true          # off entirely = none of this happens, nothing computed at all
  available:
    population_action_share_per_option:
      comet: true         # this one also goes to Comet, every round (one key per option)
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

## 3. The binned trajectory metrics

These two are the Ashery–Baronchelli trajectory statistics, specified in
[`05082026_ashery_success_rate_and_production_probability.md`](../tdd/misselaneous/05082026_ashery_success_rate_and_production_probability.md).
Unlike everything above they are computed **per bin**, not per round — one bin is one *population
round*, i.e. N pair interactions for population size N. They land in their own files:

| File | One row per | What it says |
| --- | --- | --- |
| `metrics/success_rate.csv` | bin | Of the N interactions in this bin, what fraction had both agents say the same word? |
| `metrics/production_probability.csv` | bin × word | Of all the individual words spoken in this bin, what fraction were this word? |

**The denominators are deliberately different, and this is the whole point.** A bin of L
interactions contributes **L** observations to the success rate (one outcome per conversation) but
**2L** to the production probability (both agents speak every time). Confusing them is the mistake
the spec is written to prevent:

```text
(M,M) (M,Q) (Q,Q) (M,M)        <- 4 interactions, 8 spoken words

success rate            = 3/4 = 0.75    <- how often they agreed
production probability  M = 5/8 = 0.625 <- how often M was said
                        Q = 3/8 = 0.375
```

Production probability alone does **not** measure coordination. `(M,M),(M,M),(Q,Q),(Q,Q)` and
`(M,Q),(M,Q),(Q,M),(Q,M)` both give P(M) = 0.5, but success rates of 1 and 0 respectively.

Both files keep the **raw counts** next to the normalized value (`success_count`, `action_count`,
`eligible_output_count`), so any number can be rechecked or re-derived later without the original
trajectory.

### Configuring them

```yaml
metrics:
  bin_size_interactions: null   # null = one population round (N interactions). Set a number to pin it.
  partial_final_bin: drop       # drop | include | error - what to do with a short final bin
  exclude_committed_outputs: false
```

`partial_final_bin` matters because a short final bin is a smaller sample, so its point is noisier
than the rest of the curve. Use one fixed policy across every run you intend to compare.

### `exclude_committed_outputs`, and what "committed" means

A **committed** agent is one hard-wired to always say the same word, set up through
`control.mechanism: forced_action`. In the Ashery minority experiments these are the stubborn
agents whose job is to try to flip the population.

They distort the production probability by construction — they say their word every single time
they are picked. Set `exclude_committed_outputs: true` to drop their outputs from **both** the
numerator and the denominator, leaving the question you actually care about: *did the free agents
adopt the word?*

```text
6 agents, 2 of them forced to always say M, the 4 free agents all choose Q

exclude_committed_outputs: false  ->  P(Q) = 0.625, P(M) = 0.375
exclude_committed_outputs: true   ->  P(Q) = 1.0,   P(M) = 0.0
```

The filter is applied **per output, not per interaction** — when a committed agent is paired with
a free one, the free agent's word is kept and only the committed agent's is dropped.

The success rate is left over the actual pair outcomes either way: a conversation either matched or
it did not, regardless of who was in it.

---

## 4. Where to go next

- **Adding a metric that isn't on this list:** [`building_a_metric.md`](../howto/building/building_a_metric.md).
- **Everything else about the config:** [`creating_a_game_config.md`](../howto/configuring/creating_a_game_config.md).
- **Running an episode day to day:** [`running_an_episode.md`](../howto/launch/running_an_episode.md).
- **The real files:** `src/mas_cc/metrics/{base,generic}.py` (the shared, cross-game library),
  `src/mas_cc/metrics/interactions.py` (the binned trajectory metrics, as pure functions you can
  call directly on stored records), `src/mas_cc/games/naming_convention/metrics.py` (this game's own
  adapter + rolling metrics), `src/mas_cc/observability/recorder.py` (what actually calls a metric
  and writes its value down).
