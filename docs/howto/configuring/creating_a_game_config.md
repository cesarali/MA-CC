# What you can put in a game config

A cookbook, not a reference: "I want X, here's the YAML for it." Each recipe is a small change to
[`configs/runs/naming_convention_tutorial_university_v3.yaml`](../../../configs/runs/naming_convention_tutorial_university_v3.yaml).
Run it with `mas-cc game episode --config <path>` — see
[`running_an_episode.md`](../launch/running_an_episode.md) for the command itself.

## Table of contents

1. [Which metrics you get — and the two you're probably looking for that aren't here](#1-which-metrics-you-get--and-the-two-youre-probably-looking-for-that-arent-here)
2. [See example prompts](#2-see-example-prompts)
3. [Choose the answer format: JSON+reason vs. one word](#3-choose-the-answer-format-jsonreason-vs-one-word)
4. [Change the population and number of rounds](#4-change-the-population-and-number-of-rounds)
5. [Change the model / provider](#5-change-the-model--provider)
6. [Turn Comet on or off](#6-turn-comet-on-or-off)
7. [Change the money/request limits](#7-change-the-moneyrequest-limits)
8. [Force specific agents' answers instead of asking the LLM](#8-force-specific-agents-answers-instead-of-asking-the-llm)
9. [Change where output is written](#9-change-where-output-is-written)
10. [Things you cannot change](#10-things-you-cannot-change)

---

## 1. Which metrics you get — and the one you're probably looking for that isn't here

There are exactly **eight** metrics this game computes per episode. That's the whole menu — you
can't add new ones from the config, only turn them on/off and choose which reach Comet. What each
one actually means, in plain terms, is a full reference on its own:
[`docs/documentation/metrics.md`](../../documentation/metrics.md). This section is just the config
recipe for turning them on.

**To see them printed in the console when the episode finishes:**

```yaml
logging:
  options:
    show_metrics: true
```

**Every metric gets its own `comet: true`/`false` right next to its name** — this is the
authoritative list of what exists *and* what leaves the machine, in one place, not two things you
have to cross-reference by hand:

```yaml
metrics:
  enabled: true
  available:
    population_action_share_q:
      comet: true
    population_action_share_m:
      comet: true
    dominant_action_share:
      comet: true
    first_consensus_time:
      comet: true
    rolling_coordination_rate:
      comet: true
    rolling_action_share_q:
      comet: false
    rolling_action_share_m:
      comet: false
    agent_current_action:
      comet: false   # agent-scoped - Comet only ever receives population-level values, regardless
```

Omitting a metric from `available` (or leaving `comet` unset/`false`) just keeps it local — still
computed and written to `metrics/streaming.csv`/`final.csv` if `enabled: true`, just not sent
anywhere. `game episode`'s closing summary prints exactly which names actually got exported
(`Metrics exported to Comet (N): ...`), so you don't have to take the config's word for it either.
The accompanying tutorial config lists all eight explicitly, as a working example.

**If you turn metrics off entirely** (`metrics.enabled: false`), none of the above computes or
sends anything.

**The rolling metrics also get plotted automatically** — `game episode` writes one PNG per
population-scope streaming metric to `<output-dir>/.../metrics/plots/`, using the same
`metrics/streaming.csv` the console/Comet numbers come from. No separate command needed for this
anymore.

**Mutual information is still not here, on purpose.** `mas-cc analysis empowerment` estimates
mutual information between a swept condition (e.g. varying `game.horizon`) and the outcome, across
**many** episodes — it needs a completed **grid** (`mas-cc experiment run` with a `grid:` section)
as input, not one episode's data. That's a different statistical object from anything in the list
above (a per-episode `Metric` literally cannot compute it — there's nothing to average over inside
one episode), so it stays a separate, later step rather than a config option here.

---

## 2. See example prompts

```yaml
logging:
  options:
    prompt_examples:
      count: 3   # writes 3 rounds' prompts as Markdown to <output-dir>/.../prompts/
```

`count: 0` (or leaving this out) writes none. This is separate from the raw JSON audit trail below
— use this one unless you specifically want JSON:

```yaml
logging:
  options:
    detailed_prompt_audit:
      enabled: true
      log_every_n_rounds: 1   # every round, both players' attempts
```

Rejected attempts (the model's answer didn't parse) are **always** written as Markdown to
`<output-dir>/.../failures/`, regardless of either setting above — you don't need to turn anything
on to see those.

---

## 3. Choose the answer format: JSON+reason vs. one word

```yaml
prompt:
  response_contract:
    type: choice_only        # or: paper_choice_reason
game:
  options:
    response_contract: choice_only     # must match the line above
    parser_contract: choice_only_v1    # must match this pair
```

The default (`paper_choice_reason` / `json_reason` / `tolerant_paper_object_v1`) asks for a reason
alongside the choice — closer to the source paper, but easier for a model's answer to get truncated
or malformed. `choice_only` asks for just the word `Q` or `M`, nothing else — no reason captured,
but far less likely to fail. **All three lines have to agree** — a mismatch is now a clear error at
launch, not a silent no-op.

---

## 4. Change the population and number of rounds

```yaml
game:
  population_size: 6   # at least 2
  horizon: 12           # number of pair-interactions the episode runs
```

More agents/rounds means more provider requests, so re-check `mas-cc game preflight` and
`budget.max_cost_per_run` after changing either.

---

## 5. Change the model / provider

```yaml
llm_provider:
  type: university          # or: openai, gemma_local, mock
  model: gwdg/qwen3-30b-a3b-instruct-2507
  max_output_tokens: 256    # raise this if answers get cut off (finish_reason: length)
```

---

## 6. Turn Comet on or off

```yaml
logging:
  comet: true
  options:
    comet_project: mas-cc
```

Only metrics marked `comet: true` under `metrics.available` (recipe 1) are ever sent — never
prompts or responses.

---

## 7. Change the money/request limits

```yaml
budget:
  max_cost_per_run: 0.50
  max_provider_requests: 60
```

A run that would exceed these refuses to launch (or stops mid-run) rather than overspending.

---

## 8. Force specific agents' answers instead of asking the LLM

```yaml
control:
  mechanism: forced_action
  options:
    agent_ids: [agent-000, agent-001]
    forced_value: Q
    until_interaction: 3   # optional - omit to force for the whole episode
```

Forced agents never call the LLM — zero cost for their decisions.

---

## 9. Change where output is written

```yaml
storage:
  output_dir: results
```

`mas-cc game episode --config <path>` writes here by default; pass `--output-dir <dir>` on the
command line to override just for one run without editing the file.

---

## 10. Things you cannot change

These are in the config schema but locked to one value for this game — setting anything else
raises an error immediately, before any provider call:

- `game.topology` — must be `complete`
- `game.options.pair_sampling` — must be `uniform_two_distinct`
- `game.options.success_payoff` / `failure_payoff` — must be `100` / `-50`
- `game.options.stop_on_convergence` — must be `false` (no early stopping, fixed horizon only)

If you need any of those to actually vary, that's a code change, not a config one.
