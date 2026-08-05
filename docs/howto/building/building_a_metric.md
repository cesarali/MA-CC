# Building a metric

A how-to for adding a **new** metric. For the metrics that already exist and how to turn them on,
see [`docs/documentation/metrics.md`](../../documentation/metrics.md) instead — this file doesn't
repeat that list.

## Table of contents

1. [What a metric is, in plain terms](#1-what-a-metric-is-in-plain-terms)
2. [Adding a new metric, step by step](#2-adding-a-new-metric-step-by-step)
3. [Where to go next](#3-where-to-go-next)

---

## 1. What a metric is, in plain terms

A metric is a number (or a word) computed from what happened in the game — nothing about prompts,
nothing about the model, purely about who played what and when. There are two kinds:

- **A live number, updated every round** — "right now, what fraction of the population is playing
  Q?" This kind is called every round and its value can change round to round. In code, this is a
  `StreamingMetric`.
- **A one-time summary at the end** — "which round did everyone finally agree?" This kind only
  makes sense once the whole episode is over, so it's computed once, at the end, looking back over
  everything that happened. In code, this is a `FinalMetric`.

Every metric is population-wide (one number for the whole group) or per-agent (one number per
agent) — never both. Both kinds live under one small shared contract,
`src/mas_cc/metrics/base.py` — `Metric`, `StreamingMetric`, `FinalMetric` — deliberately just those
three, nothing more elaborate.

---

## 2. Adding a new metric, step by step

Worked example: a metric that doesn't exist yet — **"what fraction of the population has played at
least once so far"** (call it `participation_rate`). Simple on purpose, so the steps stay visible.

### Step 1 — decide live or end-of-episode

"What fraction has played so far" makes sense to check every round → **live** (a `StreamingMetric`).
If instead you wanted "how many rounds did it take before everyone had played at least once," that
only makes sense once you can look back over the whole episode → **end-of-episode**
(a `FinalMetric`, like the existing `first_consensus_time`).

### Step 2 — check whether it already exists, generically

Before writing a new class, check `src/mas_cc/metrics/generic.py` — metrics written there work for
*any* game that reduces to "each round, each agent has a current value" (most games do). It already
has `ValueShare`, `AgentCurrentValue`, `DominantValueShare`, `FirstConsensusTime`,
`AgentAbsoluteError`, `MeanAbsoluteError`. Only write a new class if none of these, with different
constructor arguments, already say what you need.

### Step 3 — write the class

For naming-convention-specific metrics, add it to `src/mas_cc/games/naming_convention/metrics.py`,
next to the other metric classes there:

```python
class ParticipationRate(StreamingMetric):
    """Population share of agents who have played at least once so far."""

    def __init__(self, *, name: str = "participation_rate") -> None:
        super().__init__(name, scope="population")

    def compute_round(self, view: RoundView) -> Mapping[AgentId | None, Any]:
        if not view.agent_values:
            return {None: 0.0}
        played = [value for value in view.agent_values.values() if value is not None]
        return {None: len(played) / len(view.agent_values)}
```

A few things this example is quietly relying on, worth knowing before you write your own:

- `view.agent_values` is a dict of `{agent_id: their most recent action, or None if they haven't
  played yet}` — that's all `compute_round` ever gets to look at for the simple cases.
- The return shape is always a dict. `{None: <value>}` means "one number for the whole population."
  A per-agent metric (like `agent_current_action`) instead returns `{agent_id: value, ...}`, one
  entry per agent, and is declared with `scope="agent"`.
- If your metric needs more than "everyone's current action" — e.g. a rolling-window metric needs
  *which two agents just played and whether they matched*, not just current state — that's what
  `view.recent_history` is for; see `RollingCoordinationRate` in the same file for a worked example
  of that case (tail-slicing a list of past interaction outcomes rather than needing its own
  memory).

### Step 4 — register it

One line, in the same file's `build_metrics()`:

```python
def build_metrics(actions=("Q", "M"), population_size=4) -> list:
    return [
        ...,
        ParticipationRate(),   # <- add it here
    ]
```

### Step 5 — that's it

Nothing else to wire up. `mas-cc game episode` reads `METRICS` from this file automatically — your
new metric now prints with `logging.options.show_metrics: true`, gets its own row in
`metrics/streaming.csv` (or `final.csv`), gets its own plot if it's live and population-wide, and
can be listed under `metrics.available` for Comet export — exactly like the built-in ones, with
zero other code changes. See
[`docs/documentation/metrics.md`](../../documentation/metrics.md) for exactly how those are turned
on.

**What I actually validated:** I added exactly this `ParticipationRate` class, registered it, ran
`mas-cc game episode` against a mock provider, confirmed `participation_rate` appeared in the
console's `Metrics:` block and that the plot count went up by one (a new `participation_rate.png`),
then removed the class again — the shipped `metrics.py` is unchanged by writing this doc. Copy the
class above yourself to actually add it permanently.

---

## 3. Where to go next

- **What already exists, and how to turn any of it on:** [`docs/documentation/metrics.md`](../../documentation/metrics.md).
- **Everything else about the config:** [`creating_a_game_config.md`](../configuring/creating_a_game_config.md).
- **The real files:** `src/mas_cc/metrics/{base,generic}.py` (the shared, cross-game library),
  `src/mas_cc/games/naming_convention/metrics.py` (this game's own adapter + rolling metrics),
  `src/mas_cc/observability/recorder.py` (what actually calls a metric and writes its value down).
