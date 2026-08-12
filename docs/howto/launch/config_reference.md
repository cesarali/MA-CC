# The annotated config file — every key, every metric, what changing it does

This is the document the other two files in this folder were supposed to be: one config, fully
expanded, with a comment on every single key saying **what values are legal, what the default is,
and what actually happens if you change it** — not a cookbook of isolated snippets. If you want
"I want X, here's the diff," see [`creating_a_game_config_2.md`](creating_a_game_config_2.md)
instead. If you want the *meaning* of a metric's number in plain terms, see
[`docs/documentation/metrics.md`](../../documentation/metrics.md). This document is the map of the
whole schema, so you can find any key in one pass instead of cross-referencing four files.

The worked example below is
[`configs/runs/naming_convention_tutorial_university_v3.yaml`](../../../configs/runs/naming_convention_tutorial_university_v3.yaml)
verbatim, with commentary. It is a real, runnable config — `mas-cc game episode --config
configs/runs/naming_convention_tutorial_university_v3.yaml` launches it as-is. Every section below
(`llm_provider`, `prompt`, `execution`, `logging`, `storage`, `analysis`, `control`, `aggregation`,
`observability`, `pricing`, `budget`, `experiment`) is **generic** — every game uses the identical
schema for it. Only two things are game-specific: `game.options` (each game defines and validates
its own keys) and the metric names under `metrics.available` (each game declares its own list, in
its `games/<type>/metrics.py`). Section 8 catalogs the metric names for every game currently in the
repo, so you don't have to go source-diving for a game other than naming-convention.

Source of truth for all of this, if you ever suspect this document has drifted:
`src/mas_cc/config/models.py` (the dataclasses), `src/mas_cc/config/schema.py` (the JSON Schema a
loaded config is validated against), and `src/mas_cc/config/loader.py` (YAML → those dataclasses).

## Table of contents

1. [`llm_provider` — which model answers, and how](#1-llm_provider--which-model-answers-and-how)
2. [`prompt` — how the question is phrased and parsed](#2-prompt--how-the-question-is-phrased-and-parsed)
3. [`game` — population, horizon, and per-game rules](#3-game--population-horizon-and-per-game-rules)
4. [`execution`, `pricing`, `budget` — how it runs, and what it's allowed to cost](#4-execution-pricing-budget--how-it-runs-and-what-its-allowed-to-cost)
5. [`logging` — console, audit trail, and Comet](#5-logging--console-audit-trail-and-comet)
6. [`storage` — where files land](#6-storage--where-files-land)
7. [`control` — forcing specific agents' answers](#7-control--forcing-specific-agents-answers)
8. [`metrics` — every episode-level metric, by game](#8-metrics--every-episode-level-metric-by-game)
9. [`analysis`, `aggregation`, `observability` — grid-level and offline analysis](#9-analysis-aggregation-observability--grid-level-and-offline-analysis)
10. [`experiment` — labels only](#10-experiment--labels-only)
11. [Fields that exist but currently do nothing](#11-fields-that-exist-but-currently-do-nothing)

---

## 1. `llm_provider` — which model answers, and how

```yaml
llm_provider:
  type: university               # REQUIRED. One of: university | openai | gemma_local | mock.
                                  # Changes which adapter class handles every request — see
                                  # Appendix B for what's specific to each. Cannot be swept in a
                                  # grid (see section 9): one grid uses one provider client.
  model: gwdg/qwen3-30b-a3b-instruct-2507
                                  # REQUIRED, non-empty string. The model identifier your chosen
                                  # `type` expects (a GWDG/University model path, an OpenAI model
                                  # name, a local HF model id for gemma_local, or anything for
                                  # mock — mock never calls a real model). Also cannot be swept.
  credentials_env: POTSDAM_API_KEY
                                  # Name of an environment variable holding the API key — NOT the
                                  # key itself; the config never carries a secret. Must match
                                  # `^[A-Za-z_][A-Za-z0-9_]*$`. Omit (or null) for providers that
                                  # need no credential, e.g. mock or a fully local gemma_local.
  base_url_env: BASE_POTSDAM_LLM_URL
                                  # Same idea, for the endpoint URL. Required by `university` (a
                                  # proxy with no fixed public URL); irrelevant for `openai`
                                  # (fixed endpoint) and `gemma_local`/`mock` (no HTTP call at all).
  timeout_seconds: 60            # Float > 0. Per-request HTTP timeout passed straight to the
                                  # client. Too low on a slow model → requests fail with a timeout
                                  # error and get retried (see max_retries) or counted as failures.
  max_retries: 2                 # Integer >= 0. Retries per request on a transient provider error
                                  # (timeout, 5xx, rate limit) before that decision fails. Raising
                                  # this trades latency/cost for resilience against flaky providers;
                                  # each retry is a new billed request.
  request_concurrency: 10        # Integer >= 1. Max in-flight requests to this provider at once.
                                  # Higher = faster wall-clock time, more load on the provider (rate
                                  # limits!) and higher peak memory for gemma_local (competing GPU
                                  # forward passes) — the code forces this to 1 for gemma_local
                                  # regardless of what you set, since one local model instance can't
                                  # usefully serve concurrent requests.
  temperature: 0.0                # Float >= 0. Sampling temperature. 0.0 is as close to
                                  # deterministic as the provider allows — the standard choice for
                                  # reproducible episodes. Raising it increases response diversity
                                  # (and, for a fixed prompt, the invalid-response rate — watch
                                  # `game.options.expected_validation_failure_rate` below).
  max_output_tokens: 128          # Integer >= 1. Hard cap on tokens the model can generate per
                                  # call. Too low → answers get cut off mid-JSON
                                  # (finish_reason: length) and fail to parse, burning a retry
                                  # (game.options.invalid_response_retries) for nothing. Also feeds
                                  # `budget.max_output_tokens` accounting at preflight time.
  options:                       # Free-form per-adapter bag; each adapter reads its own keys and
                                  # ignores the rest. See Appendix B for which keys which adapter
                                  # understands (e.g. `university`'s estimated_latency_seconds,
                                  # `gemma_local`'s dtype/device_map/allow_cpu, `mock`'s
                                  # response/artificial_latency_seconds).
    estimated_latency_seconds: 3.0
                                  # Used only by `mas-cc game preflight`'s wall-clock estimate for
                                  # university/openai — has zero effect on the actual run. Raise it
                                  # if your preflight time estimate is consistently too optimistic.
```

---

## 2. `prompt` — how the question is phrased and parsed

```yaml
prompt:
  schema_version: 2              # 1 or 2. Almost always 2 (the current prompt-component schema,
                                  # where a registered `FullPrompt` owns block order). 1 is the
                                  # legacy schema that requires a `blocks:` list here instead —
                                  # supported for old configs, not for new ones; schema 2 explicitly
                                  # forbids `blocks:` being present at all.
  prompt_family: naming_convention_decision
                                  # REQUIRED. Which registered prompt template family to use — this
                                  # is what actually determines prompt wording, independent of the
                                  # game's own logic. Must be one this game's runtime recognizes
                                  # (naming_convention's is `naming_convention_decision`); an
                                  # unregistered family fails at launch, not silently.
  prompt_version: 1               # Integer >= 1. Selects a specific revision of that family, for
                                  # when the wording changes but you want old runs' configs to keep
                                  # pointing at the wording they actually used.
  message_mode: merge_consecutive_roles
                                  # `per_block` or `merge_consecutive_roles`. Whether consecutive
                                  # same-role prompt blocks become separate chat messages or one
                                  # merged message. Some providers/models behave better with fewer,
                                  # larger messages; this is a presentation choice, not a content
                                  # change — merging never changes what's said, just how many
                                  # message objects say it.
  block_separator: "\n\n"        # String. Inserted between merged blocks when message_mode is
                                  # merge_consecutive_roles. Cosmetic only.
  response_contract:
    type: choice_only            # `choice_only` or `paper_choice_reason`. `choice_only` asks for
                                  # just the action word (`Q`/`M`) — far less likely to fail to
                                  # parse. `paper_choice_reason` asks for JSON with a reason
                                  # alongside the choice, closer to the source paper, more prone to
                                  # truncation/malformed JSON at low max_output_tokens.
                                  # MUST agree with game.options.response_contract below, and with
                                  # game.options.parser_contract's family — a mismatch is a
                                  # validation error at launch, not a silent no-op.
    allowed_values: [Q, M]       # The literal legal answers. Must match game.options.actions.
```

---

## 3. `game` — population, horizon, and per-game rules

```yaml
game:
  type: naming_convention        # REQUIRED. Which registered game to run — changes literally
                                  # everything below `options:` (each game validates its own keys
                                  # and ignores unknown ones under a different game). Currently
                                  # registered: toy_coordination, naming_convention,
                                  # synthetic_bernoulli, synthetic_markov,
                                  # synthetic_controlled_markov, hidden_bench_vanilla,
                                  # hidden_bench_naming (see `mas_cc.games.registry` for the
                                  # authoritative list — new games get added there). Cannot be
                                  # swept in a grid.
  population_size: 5             # Integer >= 2. Number of agents. More agents → more provider
                                  # requests per interaction is unaffected (still 2 agents talk per
                                  # pair-interaction for naming_convention) but more agents means
                                  # more pairings are possible, so convergence typically takes
                                  # longer relative to a fixed horizon. Also feeds
                                  # `metrics.resolved_bin_size` (see section 8) when
                                  # `bin_size_interactions` is left null.
  horizon: 20                    # Integer >= 1. Number of pair-interactions the episode runs — the
                                  # episode's fixed length; naming_convention never stops early
                                  # (stop_on_convergence is locked to false, see below). More
                                  # interactions = more requests = more cost; re-check
                                  # `mas-cc game preflight` and `budget.*` after changing this or
                                  # population_size.
  topology: complete             # For naming_convention this is LOCKED to "complete" (every agent
                                  # can be paired with every other) — any other value is a
                                  # validation error at launch, before any provider call. Other
                                  # games may support other topologies; this is a per-game
                                  # constraint, not a global one.
  options:                       # Everything below is naming_convention-specific. A different
                                  # game.type validates a completely different set of keys — see
                                  # section 8 for pointers into other games' option sets.
    actions: [Q, M]               # List of >= 2 unique, non-empty single-line strings — the
                                  # convention's action alphabet. Must match
                                  # prompt.response_contract.allowed_values. More actions makes
                                  # consensus slower and each metric's per-option breakdown wider
                                  # (one curve per action, e.g. population_action_share_per_option
                                  # gets one series per entry here).
    memory_size: 3                 # Integer >= 0. How many past interactions each agent's prompt
                                  # recalls. 0 = fully memoryless (each decision independent of
                                  # history — closer to a static game); larger values give the
                                  # model more context but a longer, costlier prompt.
    success_payoff: 100           # LOCKED to 100 for this game's paper-faithful profile — any
                                  # other value is a launch-time error.
    failure_payoff: -50           # LOCKED to -50, same reasoning.
    pair_sampling: uniform_two_distinct
                                  # LOCKED to this value — uniform random pairing of two distinct
                                  # agents each interaction. No alternative is implemented for this
                                  # game yet.
    simultaneous_pair_decisions: true
                                  # LOCKED to true — both paired agents answer before either sees
                                  # the other's answer (no first-mover advantage).
    randomize_presented_action_order: true
                                  # LOCKED to true — the order [Q, M] vs [M, Q] shown in the prompt
                                  # is randomized per presentation, so a model's answer can't be an
                                  # artifact of always picking "the first option listed."
    prompt_contract: naming_convention_decision
                                  # LOCKED — must equal game.type's own family name; a mismatch
                                  # means an unrelated prompt template got wired to this game.
    response_contract: choice_only
                                  # Must equal prompt.response_contract.type above (section 2) —
                                  # this is the game engine's own copy of the same decision, checked
                                  # for agreement at launch.
    parser_contract: choice_only_v1
                                  # Must be the parser that matches response_contract:
                                  # choice_only_v1 for choice_only, or tolerant_paper_object_v1 for
                                  # paper_choice_reason. Picks which function turns raw model text
                                  # into a validated action; the wrong pairing either fails to parse
                                  # a perfectly good answer or accepts malformed ones.
    invalid_response_retries: 2   # Integer >= 0. How many times a single decision gets re-asked
                                  # after a response fails to parse, before it's recorded as a
                                  # failure (written to <output-dir>/.../failures/ regardless of any
                                  # logging setting). Each retry is a new billed request — this is
                                  # exactly why the budget comment in the accompanying config sizes
                                  # max_provider_requests at 3x the base request count.
    expected_validation_failure_rate: 0.0
                                  # Float in [0, 1]. Purely a preflight cost-estimate input — it
                                  # inflates the expected request count by this fraction to account
                                  # for anticipated retries. Has no effect on the actual run; only
                                  # on how conservative `mas-cc game preflight`'s estimate is.
    representative_memory_size: 1 # Integer >= 0. How many recent interactions feed the
                                  # "representative" summary structure used internally; distinct
                                  # from memory_size (which sizes what's shown in the prompt).
    stop_on_convergence: false    # LOCKED to false — the episode always runs the full horizon,
                                  # even after the population visibly converges. This is what makes
                                  # every episode's row count comparable; there is no early-stopping
                                  # mode for this game.
```

---

## 4. `execution`, `pricing`, `budget` — how it runs, and what it's allowed to cost

```yaml
execution:
  seed: 20260808                 # Integer >= 0. The RNG seed for pairing/order randomization
                                  # (and provider sampling seed, where the provider honors one).
                                  # Same seed + same everything else = same pairing sequence and
                                  # action-order randomization; the model's actual answers can still
                                  # vary run to run unless temperature: 0.0 and the provider is
                                  # itself deterministic.
  repetitions: 1                  # Integer >= 1. How many independent episodes one `experiment run`
                                  # (not `game episode`, which always runs exactly one) executes for
                                  # this config. Each repetition gets a different seed derived from
                                  # this one. Irrelevant to `game episode`.
  parallelism: 1                  # Integer >= 1. How many repetitions/cells an experiment runs
                                  # concurrently. Bounded in practice by request_concurrency and
                                  # provider rate limits, not just this number.
  fail_fast: true                 # Boolean. If true, one repetition's unrecoverable error stops the
                                  # whole experiment; if false, other repetitions keep going and the
                                  # failure is recorded per-repetition.
  # timeout_seconds is also a valid key here but is not currently read anywhere —
  # see section 11.
pricing:
  mode: live                      # live | cached | offline. `live` fetches a fresh price quote
                                  # from the provider before launch (only meaningful for
                                  # `university`, which has a live pricing endpoint); `cached` reuses
                                  # a quote from cache_path if fresh enough; `offline` never calls
                                  # out and uses the repo's static pricing table. Changing this
                                  # changes whether preflight can fail due to a pricing-lookup
                                  # network error.
  max_age_seconds: 600             # Float >= 0. How old a cached/live quote may be before it's
                                  # considered stale. Only relevant when mode is cached, or when
                                  # live mode's own quote needs to still be fresh at launch time
                                  # (see require_fresh_at_launch).
  require_fresh_at_launch: true    # Boolean. If true, a stale quote (older than max_age_seconds)
                                  # blocks launch instead of silently proceeding with a stale price.
  fallback_policy: deny            # deny | offline | allow_stale. What happens when a live/cached
                                  # quote can't be obtained: refuse to launch, fall back to the
                                  # static offline table, or proceed with whatever stale quote
                                  # exists anyway (least safe — cost estimate may be wrong).
  explicit_unknown_price_override: false
                                  # Boolean. Normally an unpriced model blocks launch. Setting this
                                  # true is an explicit "I know this model has no price entry, run
                                  # it anyway" override — cost tracking for it will read as 0 or
                                  # unknown, so budget.max_cost_per_run stops being a meaningful
                                  # guard for this run.
budget:
  # A run whose *preflight* demand exceeds a limit below refuses to launch. Past that point
  # only the cost ceiling halts a run mid-flight by default — see "Prefer a cost ceiling to
  # a token ceiling" below, which is the single most important thing in this section.
  accounting_unit: proxy_accounting_unit
                                  # String label only — records what unit the cost numbers below
                                  # are in (a real currency code, or a proxy unit for a provider
                                  # without real billing, like the University proxy). Purely
                                  # descriptive; changing it doesn't rescale anything.
  system_max_cost_per_run: 0.25    # Float >= 0 or null. An operator-level ceiling, meant to be set
                                  # once for a deployment and not overridden per-experiment casually.
  max_cost_per_run: 0.25           # Float >= 0 or null. The per-run cost ceiling actually enforced;
                                  # null means unbounded (dangerous — combine with
                                  # allow_unbounded_paid_requests below only deliberately).
  live_spend_poll_seconds: 120     # Integer >= 10 or null. When set (University provider only),
                                  # the run polls the proxy's own `/user/info` accounting on this
                                  # interval and stops when the provider says THIS RUN has spent
                                  # more than max_cost_per_run. It is a delta measured from the
                                  # account's spend at launch, so other traffic on a shared key
                                  # neither borrows from nor donates to this run's ceiling. Null
                                  # disables it and leaves only the guard's own arithmetic.
  max_provider_requests: null      # Integer >= 0 or null. Hard cap on total provider calls,
                                  # independent of cost. See the note below before setting it.
  max_input_tokens: null           # Integer >= 0 or null. Cumulative input-token ceiling.
  max_output_tokens: null          # Integer >= 0 or null. Cumulative output-token ceiling — distinct
                                  # from llm_provider.max_output_tokens, which caps *one call*.
  allow_unbounded_paid_requests: false
                                  # Boolean. Must be explicitly set true to launch a paid-provider
                                  # run with any of the above limits set to null — a deliberate
                                  # speed bump against accidentally launching an unbounded paid run.
```

### Prefer a cost ceiling to a token ceiling

`max_provider_requests`, `max_input_tokens`, and `max_output_tokens` behave differently
depending on whether you set them:

- **Left `null`** (recommended for real experiments) they are *advisory*. Usage is still
  tracked and written into every run's `actual_budget_status`, and the first time usage
  passes what preflight predicted you get one warning per resource on the
  `mas_cc.budget` logger. Nothing stops.
- **Set to a number** they are *hard*, exactly as before: the run halts the moment the
  counter crosses. Keep them set only where the demand is genuinely known ahead of time —
  smoke tests, fixed-length probes, a mock provider where you want a runaway retry loop
  caught immediately.

The reason for the default is that a token count is a prediction, and preflight's
prediction comes from one representative prompt. Any game whose prompts grow — anything
with `memory_size: 0`, any long discussion phase — will legitimately exceed it, and a
wrong prediction does not save money, it destroys a run. `results/DIAGNOSIS.md` records
what that costs in practice: a 50-cell grid hit its `max_input_tokens` partway through
cell 0001 and burned its remaining 4,235 episodes on refused calls.

Cost does not have that problem. The guard prices every call from the *actual* reported
token usage, and `live_spend_poll_seconds` cross-checks that arithmetic against the
provider's own books.

### What happens when a budget does stop a run

A budget stop ends the run rather than failing each queued episode. The episode that hit
the ceiling is recorded `failed`; every episode after it is recorded `skipped_aborted`
with `error_type: BudgetStop` and the reason in `error`. This holds regardless of
`execution.fail_fast`, because once the guard denies, no later call can succeed. Results
already written stay on disk and `resume: true` will pick up from them after you raise
the ceiling.

---

## 5. `logging` — console, audit trail, and Comet

```yaml
logging:
  level: INFO                     # DEBUG | INFO | WARNING | ERROR | CRITICAL. Standard Python
                                  # logging level for console/file logs; does not affect what gets
                                  # written to results files or Comet.
  console: true                   # Boolean. Whether progress/summary prints to stdout at all.
  audit: true                     # Boolean. Whether the structured per-round audit log is written
                                  # (separate from the Markdown prompt-example/failure files below).
  comet: true                     # Boolean. Master on/off switch for sending anything to Comet at
                                  # all. When false, nothing under `options.comet_project` or any
                                  # metric's `comet: true` (section 8) has any effect — no network
                                  # call to Comet happens.
  options:
    comet_project: mas-cc          # String. The Comet project name results are grouped under.
                                  # Only consulted when comet: true above.
    show_metrics: true             # Boolean. Prints the final metric values to the console when the
                                  # episode finishes. Purely a display convenience — independent of
                                  # whether metrics are computed/stored (that's `metrics.enabled`,
                                  # section 8) or sent to Comet.
    prompt_examples:
      count: 3                     # Integer >= 0. Writes this many distinct rounds' prompts as
                                  # Markdown to <output-dir>/.../prompts/round_NNN.md — one file per
                                  # distinct round, the first decision seen in each. 0 (or omitting
                                  # this whole block) writes none. Independent of
                                  # detailed_prompt_audit below — use this one unless you
                                  # specifically want the raw JSON audit trail instead.
      scope: episode               # episode | cell. `episode` preserves the full-profile behavior
                                  # above. `cell` deterministically samples the lowest-ID successful
                                  # episode and writes all selected examples into one
                                  # prompt_examples.md. results_only defaults to/requires cell.
    detailed_prompt_audit:
      enabled: false                # Boolean. Turns on a second, JSON-based prompt/response audit
                                  # trail (as opposed to prompt_examples' Markdown). Off by default
                                  # because it's more verbose; turn on only when you need every
                                  # logged prompt's full structure, not just a few examples.
      log_every_n_rounds: 1         # Integer >= 1 or omitted. When enabled, log every Nth round
                                  # (1 = every round, both players' attempts).
      always_log_first_n_rounds: 0  # Integer >= 0. Always audit-log at least this many initial
                                  # rounds regardless of log_every_n_rounds' sampling.
      max_logged_prompts_per_game: null   # Integer >= 0 or null. Caps how many prompts get logged
                                  # for one game/episode; null = unbounded.
      max_logged_prompts_per_run: null    # Same, but across an entire multi-episode experiment run.
      always_log_provider_errors: true    # Boolean. Provider-level errors are always captured
                                  # regardless of the sampling settings above — you should almost
                                  # never turn this off.
      always_log_invalid_responses: true  # Boolean. Same, for responses that failed to parse.
                                  # Note: rejected/invalid attempts are ALSO always written as
                                  # Markdown to <output-dir>/.../failures/ independent of any
                                  # logging.* setting at all — you never need to turn anything on to
                                  # see those.
```

---

## 6. `storage` — where files land

```yaml
storage:
  output_dir: results             # String. Root directory artifacts are written under. Default is
                                  # "results" (already covered by .gitignore, so a default run never
                                  # stages output files for commit). `--output-dir <dir>` on the
                                  # command line overrides this for one invocation, without editing
                                  # the file — precedence is flag > this key > the "results" default.
                                  # Both are optional for `game episode`, `game preflight`,
                                  # `experiment preflight`, `experiment run`, and every
                                  # `synthetic *` subcommand. The one exception is `game run`
                                  # (the Phase 5/6 inspection-bundle command), which still requires
                                  # `--output-dir` explicitly — it's pinned that way as a
                                  # phase-regression test fixture, not because it can't fall back.
  format: jsonl                   # String. The full recorder still uses its established JSONL/CSV
                                  # files. results_only always publishes its versioned scientific
                                  # table as Parquet; this legacy field does not override that schema.
  artifact_profile: full          # full | results_only | timing_study. full preserves the verbose per-episode
                                  # recorder tree. results_only retains atomic compact Parquet,
                                  # aggregate/analysis results, bounded prompt samples, budget
                                  # state, and the manifests needed to understand/resume the run.
                                  # This changes local retention only; master Comet reporting is
                                  # identical.
                                  # timing_study uses the same compact scientific retention as
                                  # results_only and additionally writes per-episode and per-request
                                  # timing CSVs plus timing_study.md. results_only writes only the
                                  # compact Markdown timing summary.
  checkpoint_mode: episode        # off | episode. episode makes a completed episode shard durable
                                  # and skips it after restart. An episode that was in flight starts
                                  # again from round zero with its original seed. This is not a
                                  # mid-round restore and no prompt object is restored.
  # checkpoints: true             # Transitional alias only: true -> episode, false -> off. Do not
                                  # specify it together with checkpoint_mode. Resolved configs emit
                                  # only checkpoint_mode. Historical checkpoints: true wrote round
                                  # snapshots but the orchestrator never restored a partial episode.
  overwrite: false                 # Boolean. Parsed but not currently wired to the recorder's own
                                  # write path — see section 11.
  wipe_and_recompute: false        # Boolean. If true, the entire run (or grid) output directory
                                  # under `output_dir` is deleted before the run starts, ignoring
                                  # `resume`/`checkpoint_mode` entirely — nothing is treated as
                                  # already-completed. Use this when you need a clean recompute
                                  # instead of the default resume-by-episode behavior.
```

For long paid cells, use `results_only` without changing Comet:

```yaml
logging:
  comet: true
  options:
    prompt_examples: {count: 2, scope: cell}
storage:
  artifact_profile: results_only
  checkpoint_mode: episode
  overwrite: true
  wipe_and_recompute: false
```

To recover space from a completed legacy run without new model calls, preview first and
then opt into deletion:

```bash
mas-cc experiment compact --run-dir <exact-run-dir> --profile results_only
mas-cc experiment compact --run-dir <exact-run-dir> --profile results_only --delete-raw
mas-cc experiment compact --run-dir <exact-run-dir> --profile results_only --delete-raw --archive
```

The first command validates and mutates nothing. The second reads back compact outputs
before deleting only the documented raw-artifact allowlist inside the resolved run
directory. Repeating it is safe.
`--archive` requires `--delete-raw` and writes a sibling ZIP without
`:Zone.Identifier` transfer sidecars.

---

## 7. `control` — forcing specific agents' answers

```yaml
control:
  mechanism: none                  # `none` or `forced_action`. `none` (the default) never
                                  # overrides anyone — every agent's decision goes to the LLM.
                                  # `forced_action` pins a fixed set of agents to a fixed answer
                                  # instead of calling the LLM for their decisions — those agents
                                  # cost nothing.
  options:                        # Only meaningful (and only validated) when mechanism is
                                  # forced_action; ignored under none.
    agent_ids: [agent-000, agent-001]
                                  # Non-empty list of agent ID strings to force. An unknown/invalid
                                  # ID format is a launch-time error.
    forced_value: Q                 # Non-empty string — must be one of game.options.actions for a
                                  # choice-family game, or the run will produce actions the game
                                  # doesn't recognize.
    until_interaction: 3            # Positive integer or omitted. If set, these agents are forced
                                  # only through this interaction index, then behave normally
                                  # (return to ordinary LLM-backed decisions) for the rest of the
                                  # episode — a "pulse" intervention rather than forced for life.
                                  # Omit for "forced the whole episode."
```

This is the mechanism behind a **committed-minority** experiment: force a small fixed fraction of
the population to one word and watch whether/how fast the rest converges onto it. Pair with
`metrics.exclude_committed_outputs: true` (section 8) to measure only the free agents' adoption,
not the forced agents' presence inflating the count.

---

## 8. `metrics` — every episode-level metric, by game

```yaml
metrics:
  enabled: true                    # Boolean. Master switch. false means NOTHING below computes,
                                  # is written to metrics/streaming.csv or metrics/final.csv, is
                                  # sent to Comet, or is plotted — including the binned trajectory
                                  # tables further down. There is no partial-off state.
  available:                      # Every metric name is listed here explicitly, each with its own
                                  # `comet: true/false` right next to it — this mapping IS the
                                  # authoritative answer to "what metrics exist for this game" and
                                  # "what leaves the machine," in one place. Omitting a metric here
                                  # (or leaving comet unset/false) keeps it local-only: still
                                  # computed and written to the CSVs above (controlled only by
                                  # `enabled`), just never sent to Comet. `game episode`'s closing
                                  # summary prints exactly which names got exported
                                  # ("Metrics exported to Comet (N): ..."), so you can verify this
                                  # without trusting the config file alone.
    population_action_share_per_option:
      comet: true                  # Standing per-option share of the population (sums to 1).
                                  # ONE metric, ONE curve per option — reaches Comet as one key per
                                  # option (…_Q, …_M), toggled once here for all of them.
    dominant_action_share:
      comet: true                  # Share held by whichever option currently leads — the "how
                                  # close to consensus" curve, symmetric under which option wins.
    first_consensus_time_by_action_share:
      comet: true                  # Final metric: round index where the leading option's standing
                                  # share first reaches 95%, or null if it never does.
    first_consensus_time_by_success_rate:
      comet: true                  # Final metric: the paper's own §7 criterion — first interaction
                                  # where a trailing window of 3N interactions is >=95% successful
                                  # (both agents matched). A different criterion from the one above;
                                  # they usually agree eventually but not on the same round.
    consensus_action_by_success_rate:
      comet: false                 # The word that won under the criterion above. false because it's
                                  # a label, not a number Comet can chart — leave this off unless
                                  # your Comet workflow specifically consumes categorical fields.
    rolling_coordination_rate:
      comet: true                  # Streaming: success rate over the trailing `window` interactions
                                  # (window = population_size) — the *flow* of coordination, moves
                                  # as soon as behavior changes rather than lagging like a cumulative
                                  # average would.
    rolling_action_share_per_option:
      comet: false                 # Streaming, flow counterpart to population_action_share_per_
                                  # option: share of each option among what was actually PLAYED in
                                  # the trailing window, not where the population currently stands.
    agent_current_action:
      comet: false                 # Agent-scoped passthrough of each agent's current action. This
                                  # one is NEVER sent to Comet regardless of what you set here —
                                  # Comet only ever receives population/option-level values, never
                                  # per-agent ones. Setting comet: true here has no effect.
  bin_size_interactions: null      # Positive integer or null. Configures the binned-trajectory
                                  # tables (success rate + production probability), NOT the metrics
                                  # above. null (recommended) tracks game.population_size
                                  # automatically — one bin = one population round (N interactions)
                                  # — so it stays correct if you change population_size later.
                                  # Pinning a number decouples the bin from population size, which
                                  # you'd only want for a specific cross-run comparison.
  partial_final_bin: drop          # drop | include | error. What happens to a trailing bin with
                                  # fewer than the bin size's worth of interactions left over.
                                  # `drop` omits it (default — a partial bin is a noisier, smaller
                                  # sample than the rest); `include` computes it anyway;
                                  # `error` refuses to produce the tables at all rather than mix bin
                                  # sizes silently.
  exclude_committed_outputs: false # Boolean. When true, drops outputs from agents pinned by
                                  # control.mechanism: forced_action out of BOTH the success-rate
                                  # and production-probability tables' numerator and denominator —
                                  # "did the free agents adopt this word" without the forced agents
                                  # inflating the answer. Applies per output, so a committed/ordinary
                                  # pairing still keeps the ordinary agent's word. Leave false
                                  # outside committed-minority analyses (see section 7).
```

The binned trajectory tables write `metrics/success_rate.csv` and
`metrics/production_probability.csv` — one row per bin, and per (bin, action) respectively, each
keeping its raw counts beside the normalized value. All population-scope streaming metrics also get
one PNG per metric under `<output-dir>/.../metrics/plots/`, generated automatically from the same
`metrics/streaming.csv` the console/Comet numbers come from — no separate plotting command needed.

**Every metric name that exists, by game** — `available:`'s keys must come from the list for
whatever `game.type` you're running; an unrecognized name is silently never populated (it isn't
validated against the game's own list at config-parse time, only produced or not by that game's
`build_metrics()`):

| Game (`game.type`) | Episode metrics recorded (from `games/<type>/metrics.py`) |
| --- | --- |
| `naming_convention` | `population_action_share_per_option`, `agent_current_action`, `dominant_action_share`, `first_consensus_time_by_action_share`, `rolling_coordination_rate`, `rolling_action_share_per_option`, `first_consensus_time_by_success_rate`, `consensus_action_by_success_rate` |
| `hidden_bench_vanilla` | `population_action_share_per_option`, `agent_current_action`, `dominant_action_share`, `accuracy_average`, `accuracy_majority`, `decoy_share`, `unshared_disclosure_rate`, `first_consensus_time_by_action_share`, `y_pre`, `y_post`, `improvement`, `final_disclosure_rate` |
| `hidden_bench_naming` | `population_action_share_per_option`, `agent_current_action`, `dominant_action_share`, `accuracy_average`, `accuracy_majority`, `decoy_share`, `unshared_disclosure_rate`, `disclosure_reach`, `rolling_coordination_rate`, `rolling_action_share_per_option`, `first_consensus_time_by_action_share`, `first_commitment_accuracy`, `final_accuracy` |
| `synthetic_bernoulli`, `synthetic_markov`, `synthetic_controlled_markov` | Each calls its own `build_metrics()` in `games/synthetic/<name>/metrics.py`; open that file for the exact list — they share the same generic shelf (`ActionSharePerOption`, etc.) with a per-game `ACTION_METRIC_NAME` adapter, sized for the synthetic ground-truth work rather than the LLM games above. |
| `toy_coordination` | See `games/toy_coordination/metrics.py` — the minimal illustrative game, useful as the smallest possible `available:` example. |

Every metric name above is defined once, in one of these shared modules — reading the class
docstring there is the authoritative definition:

- `src/mas_cc/metrics/generic.py` — `ActionSharePerOption`, `AgentCurrentValue`,
  `DominantValueShare`, `FirstConsensusTime`, `AgentAbsoluteError`, `MeanAbsoluteError` (the two
  error metrics are for numeric, non-choice games — unused by any choice-family game above).
- `src/mas_cc/metrics/rolling.py` — `RollingCoordinationRate`, `RollingActionSharePerOption`,
  `ConsensusFlipBySuccessRate` (the source of `first_consensus_time_by_success_rate` /
  `consensus_action_by_success_rate`).
- `games/hidden_bench/*/metrics.py` — the HiddenBench-specific ones (`accuracy_average`,
  `decoy_share`, `disclosure_reach`, etc.) are not cross-game yet; open that file if you're
  configuring a HiddenBench run and need their exact semantics.

**What's deliberately not in this list:** mutual information between a swept condition and the
outcome. `mas-cc analysis empowerment` computes that over a completed **grid**
(`mas-cc experiment run` with a `grid:` section), not from one episode — see section 9's
`aggregation.sweep_metrics` for the config knob that's actually involved.

---

## 9. `analysis`, `aggregation`, `observability` — grid-level and offline analysis

These three sections only matter once you're running a **grid** (`mas-cc experiment run` with a
`grid:` block sweeping some field, e.g. `game.horizon`, across many episodes) — not for a single
`game episode`.

```yaml
aggregation:
  forward_fill: absorbing          # absorbing | truncate | none. How a shorter episode's curve is
                                  # extended to align with longer ones before averaging across
                                  # episodes in a cell. `absorbing` repeats its last value forward
                                  # (appropriate once a metric is known to only move one direction,
                                  # e.g. after consensus); `truncate` drops that episode from later
                                  # rounds' average instead of inventing a value; `none` leaves gaps.
  relabel_by_winner: true          # Boolean. When true, `action_share_relabelled` (a cell-level
                                  # metric — see below) aligns every episode on its OWN winning
                                  # option before averaging, so the winner's and runner-up's curves
                                  # don't cancel out into a flat 0.5 line across a symmetric game.
                                  # false keeps the raw per-option curves under their real labels.
  percentiles: [10, 50, 90]        # List of integers in [0, 100]. Which percentiles get computed
                                  # for banded cell-level curves (median + a spread band, by
                                  # default the 10th/90th).
  rolling_window: 20                # Integer >= 1. Window used by cell-level aggregation's own
                                  # rolling computations — separate from any per-episode rolling
                                  # metric's own `window` (section 8), which is fixed by the game's
                                  # `build_metrics()`, not by this key.
  cell_metrics:                    # List of names from CELL_METRICS (table below). Computed once
                                  # per grid cell, from that cell's finished episodes' recorded
                                  # files — re-derivable from disk at any time, so re-aggregating
                                  # after changing forward_fill/percentiles/rolling_window above
                                  # doesn't require re-running episodes.
    - dominant_action_share
    - action_share_relabelled
    - active_fraction
    - consensus_round
    - converged_fraction
  sweep_metrics: []                # List of names from SWEEP_METRICS (table below) — grid-level
                                  # metrics computed across ALL finished cells (the mutual-
                                  # information family). Empty list = no sweep-level statistics
                                  # computed; this is the actual knob behind "empowerment" analysis.
                                  # Requesting any of these silently adds `macrostate_counts` to
                                  # cell_metrics if you didn't list it yourself — a sweep metric
                                  # needs that table and would otherwise produce NaN.
  horizons: [1]                    # List of positive integers. Time-lags used by `lagged_cmi`
                                  # (ignored by the other sweep metrics). Each entry adds one
                                  # `lagged_cmi_h<N>_estimate` scalar.
  null_permutations: 200           # Integer >= 0. Label-shuffle permutations used by
                                  # `mi_null_band` to estimate how much of the raw MI estimate is
                                  # finite-sample noise rather than a real effect. More permutations
                                  # = a tighter, slower null-band estimate.
analysis:
  enabled: false                   # Boolean. Only consumed for game.type: hidden_bench_imitation —
                                  # runs the four MI/CMI statistics below automatically after the run
                                  # finishes. For every other game, aggregation.sweep_metrics above
                                  # plus the `mas-cc analysis empowerment` CLI command are the real
                                  # grid-level MI machinery instead.
  estimators: []                   # List of names. Required (non-empty) when enabled is true.
                                  # MI/CMI channels: sensing_mi, population_actuation_cmi,
                                  # target_actuation_cmi, focal_actuation_cmi, and their
                                  # order-parameter projections sensing_mi_m_{ctrl,truth,order} and
                                  # m_{ctrl,truth,order}_actuation_cmi.
                                  # Controller diagnostics, which make those CMIs readable:
                                  # controller_action_entropy plus
                                  # controller_action_entropy_given_{population,m_ctrl,m_truth,m_order}
                                  # (the ceiling each CMI is bounded by),
                                  # {population,m_ctrl,m_truth,m_order}_actuation_information_fraction
                                  # (CMI over that ceiling — NaN, never 0, when the ceiling
                                  # vanishes; a normalization diagnostic, not an efficiency),
                                  # m_{ctrl,truth,order}_signed_actuation (state-adjusted
                                  # ADVOCATE_Z-minus-NO_OP direction, which CMI cannot give), and
                                  # {population,m_ctrl,m_truth,m_order}_action_overlap (how much
                                  # data supports a within-state comparison at all).
                                  # The two families are listed together here and land in
                                  # information_estimates.md side by side.
  options: {}                      # bootstrap_resamples, null_permutations, confidence, seed — see
                                  # docs/documentation/hidden_bench/hidden_bench_imitation.md section 7.
  comet_export: false              # Boolean. Upload the rendered information_estimates.md report to
                                  # Comet as a run asset once the analysis finishes. Only takes effect
                                  # when logging.comet (the master switch) is also true.
observability:
  comet:
    writer: master_only             # LOCKED to master_only — the only legal value. Workers write
                                  # episode files only; the grid's master process is the sole writer
                                  # to Comet, so there's exactly one writer per Comet experiment key
                                  # and no race on its step counter.
    heartbeat_seconds: 60.0          # Float > 0. How often the master pings Comet with a liveness
                                  # heartbeat during a long grid run — a timer, not tied to episode
                                  # completions, so it can distinguish "dead master" from "slow
                                  # master."
    progress_metrics: []             # Optional metric-name allowlist for the master heartbeat.
                                  # Empty means all progress, rate, ETA, and budget metrics. Use
                                  # [episodes_done] for a completion-count-only live dashboard.
    grid_image_every_n_episodes: 25  # Integer >= 1. How often the master renders and uploads a
                                  # grid-progress image to Comet, in units of completed episodes.
    sweep_experiment: true           # Boolean. Whether the grid-wide (sweep-level) Comet experiment
                                  # is created at all.
    cell_reporting: experiments      # experiments | master | disabled. Where a finished cell's
                                  # curves, scalars, and metric PNGs are published.
                                  #   experiments — one child Comet experiment per cell, named
                                  #     `<run>/<cell_id>`, with bare metric names (`m_ctrl`). This
                                  #     is what lets Comet overlay one cell against another.
                                  #   master — all of it onto the sweep experiment instead, along
                                  #     with the post-run analysis report, prefixed by cell id
                                  #     (`cell-0000_m_ctrl`). No child experiments are created.
                                  #   disabled — no cell curves uploaded; the master keeps its
                                  #     progress series and grid image.
                                  # NOTE: write `disabled`, not `off` — YAML reads a bare `off`
                                  # (and `no`) as boolean false.
    metric_plots: false              # Boolean. Upload each cell's locally rendered aggregate metric
                                  # PNGs to the experiment that carries its curves, once per cell
                                  # as that cell's last episode lands. Disabled by default to keep
                                  # image upload opt-in.
```

`cell_reporting` decides how many experiments a run writes to. Under
`experiments` (the default) one run writes to up to three — the master, one per
completed cell, and one for the post-run analysis. The aggregate plots and MI
estimates live on the latter two, so opening only the master shows little more
than the progress counter. Under `master` there is exactly one experiment
holding everything, which is right for a single-cell run, and right for any grid
you would rather read in one place than overlay.

`cell_reporting` is independent of `sweep_experiment`, which decides whether the
master exists at all. The grid-progress image belongs to the master and is
published under all three modes.

**Cell-level metrics** (`aggregation.cell_metrics`, from `src/mas_cc/metrics/cell.py`):

| Name | What it reports |
| --- | --- |
| `dominant_action_share` | Band (median + percentile spread) of the leading option's share across the cell's episodes — the curve that converges to 1.0. |
| `population_action_share_per_option` | Per-option population-share bands under the actual option labels. |
| `action_share_relabelled` | Per-*rank* share bands (`_option_1`, `_option_2`, …) after aligning each episode on its own winner — keeps the loser's trajectory visible instead of washing it out. |
| `active_fraction` | Fraction of the cell's episodes still running at each round — distinguishes "these runs agree" from "these runs are mostly padding" after forward-fill. |
| `consensus_round` | Median/IQR of the round each episode converged on, over the converged subset only. |
| `converged_fraction` | Fraction of the cell's episodes that reached consensus at all. |
| `macrostate_counts` | Not a curve — the raw contingency tables (`terminal_outcome`, `macrostate_transition_h<N>`) a sweep metric needs. Auto-added when any `sweep_metrics` entry is requested. |
| `m_ctrl`, `m_truth`, `m_order` | Percentile bands across episodes for the controller-target, truth-aligned, and winner-agnostic population order parameters. |

**Sweep-level (grid) metrics** (`aggregation.sweep_metrics`, from `src/mas_cc/metrics/sweep.py`) —
this is where mutual-information estimates actually live:

| Name | What it reports |
| --- | --- |
| `terminal_mi` | `I(cell ; terminal outcome)` over cells finished so far — the primary "did this swept condition affect the outcome" number, reported both smoothed (Jeffreys) and unsmoothed/Miller-Madow. |
| `lagged_cmi` | `I(cell ; S_{t+h} \| S_t)` at each `aggregation.horizons` lag — conditions on the current macrostate so the statistic measures influence, not persistence. |
| `mi_null_band` | Label-shuffle null distribution for `terminal_mi` (95th percentile + mean) — the noise floor a raw MI estimate should be compared against before calling it a real effect. |
| `mi_ground_truth_gap` | Estimate minus a known closed-form answer — only meaningful for a synthetic game with a computable ground truth; emits nothing if none is supplied. |

---

## 10. `experiment` — labels only

```yaml
experiment:
  name: naming-convention-tutorial-university-walkthrough
                                  # String. Human-facing identifier — shows up in Comet, in
                                  # results-directory naming, in preflight banners. Purely
                                  # descriptive; changing it has no effect on execution.
  description: Small live N=4, H=6 example...
                                  # Free text, purely descriptive.
  tags: [tutorial, university, howto-manual]
                                  # List of strings, purely descriptive/for filtering in Comet.
```

---

## 11. Fields with limited implementations

Parsed successfully, present in the schema, but not read by any code path today — set them if you
like for future-proofing or clarity, but don't expect a behavior change:

- `execution.timeout_seconds` — the actual per-request timeout that's enforced is
  `llm_provider.timeout_seconds` (section 1).
- `storage.format` — the recorder always writes JSONL regardless of this value.
- `storage.overwrite` — not wired into the recorder's write path (a separate, unrelated
  `overwrite` flag exists internally for the prompt-example reporter).
- `analysis.enabled` / `analysis.estimators` — for `hidden_bench_imitation`, enabling this section
  runs the selected sensing/actuation MI statistics automatically after an experiment or grid and
  writes `hidden_bench_imitation_analysis/`. Other games still use
  `aggregation.sweep_metrics` (section 9) or their explicit `mas-cc analysis ...` command.

If you're relying on any of these for something, that's a sign the feature needs wiring up in code
— not a config problem.
