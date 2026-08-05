# Creating a game run config — the full field reference

This is a reference, not a tutorial: every field a `naming_convention` run config can have, its
type, its default, and — where the field only accepts specific values — exactly which ones. For
*why* these sections exist and how they fit together, see
[`building_a_game.md`](../building/building_a_game.md) (single episode) and
[`running_an_episode.md`](../launch/running_an_episode.md) (the CLI that runs one). This document
is what to open when you're staring at a YAML file asking "what can I actually put here."

**Source of truth:** every value below is read directly from the dataclasses that validate it —
`src/mas_cc/config/models.py` for the generic `RunConfig` sections, and
`src/mas_cc/games/naming_convention/game.py`'s `NamingConventionGameSpec` for `game.options`. If
this document and the code ever disagree, the code wins; file an issue or fix this file to match.

**Scope:** `naming_convention` is the only game with every section (metrics, a production runtime,
this level of validation) wired up, so it's the reference game here. `toy_coordination` exists too,
with a smaller/looser field set — see `src/mas_cc/games/toy_coordination/` directly for it.

**A complete, working example:** [`configs/runs/naming_convention_tutorial_university_v3.yaml`](../../../configs/runs/naming_convention_tutorial_university_v3.yaml).

## Table of contents

1. [The shape of a run config](#1-the-shape-of-a-run-config)
2. [`llm_provider`](#2-llm_provider)
3. [`prompt`](#3-prompt)
4. [`game`](#4-game)
5. [`game.options` — naming_convention-specific](#5-gameoptions--naming_convention-specific)
6. [`execution`](#6-execution)
7. [`pricing`](#7-pricing)
8. [`budget`](#8-budget)
9. [`logging`](#9-logging)
10. [`storage`](#10-storage)
11. [`control`](#11-control)
12. [`metrics`](#12-metrics)
13. [`analysis`](#13-analysis)
14. [`experiment`](#14-experiment)
15. [Minimal valid config](#15-minimal-valid-config)

---

## 1. The shape of a run config

A resolved `RunConfig` (`src/mas_cc/config/models.py:403`) has exactly these top-level keys. All
but the first three are optional — omit a whole section to get its defaults.

| Section | Required? | Default if omitted |
| --- | --- | --- |
| `llm_provider` | **yes** | — |
| `prompt` | **yes** | — |
| `game` | **yes** | — |
| `execution` | no | seed 0, repetitions 1, parallelism 1, fail_fast true |
| `pricing` | no | offline, deny-on-stale |
| `budget` | no | everything unbounded |
| `logging` | no | INFO, console on, Comet off |
| `storage` | no | `output_dir: results` |
| `control` | no | `mechanism: none` (no intervention) |
| `metrics` | no | enabled, nothing exported to Comet |
| `analysis` | no | disabled |
| `experiment` | no | name `unnamed-experiment` |

`schema_version` appears at the top level and inside every section; the loader defaults it to `1`
everywhere except `prompt.schema_version`, where `2` selects the current block-based prompt schema
(`1` is a legacy shape — see `PromptConfig.migration_diagnostics()`). Leave it out unless you
specifically need `prompt.schema_version: 2`.

Config files may also use `component:`/`overrides:` per section instead of writing values inline —
see any file under `configs/components/` for the pattern. Everything below describes the
**resolved** shape either way.

---

## 2. `llm_provider`

`LLMProviderConfig`, `src/mas_cc/config/models.py:36`.

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `type` | string | **required** | One of `mock`, `openai`, `university`, `gemma_local` (`src/mas_cc/llm_providers/registry.py:64`). Anything else raises `ConfigurationError` naming the available list. |
| `model` | string | **required** | Passed through verbatim to the adapter/pricing lookup. |
| `credentials_env` | string \| null | `null` | Name of the environment variable holding the API key. Read at provider-construction time, never stored in the config. |
| `base_url_env` | string \| null | `null` | Name of the environment variable holding the base URL (used by `university`/OpenAI-compatible adapters). |
| `timeout_seconds` | float | `60.0` | Per-request transport timeout. |
| `max_retries` | int | `2` | Transport-level retries (connection/5xx), separate from `game.options.invalid_response_retries` (content-validation retries). |
| `request_concurrency` | int | `1` | Caps in-flight requests to this provider. |
| `temperature` | float | `0.0` | Sent on every request. |
| `max_output_tokens` | int | `256` | Sent on every request. **The most common cause of `InvalidConventionResponse`/`DecisionLoopExhausted`**: too low and the model gets cut off mid-answer (`finish_reason: length`) before a valid response is even possible — see `failures/*.md` in [`running_an_episode.md` §5](../launch/running_an_episode.md#5-response-format-paper-faithful-jsonreason-or-a-bare-choice). |
| `options` | mapping | `{}` | Provider-specific, free-form. See below. |

**`options` keys actually consulted**, by provider:

| Key | Provider(s) | Effect |
| --- | --- | --- |
| `response` | `mock` | The fixed string every completion returns. Default `"A"` — not a legal `naming_convention` action, so leaving it unset with the mock provider will always fail validation; set it to something the game's `response_contract` expects (e.g. `'{"value":"Q","reason":"..."}'` for `json_reason`, or `'Q'` for `choice_only`). |
| `artificial_latency_seconds` | `mock` | Sleep before returning, for testing concurrency/timeouts. Default `0.0`. |
| `dtype`, `device_map`, `allow_cpu` | `gemma_local` | Model loading parameters (`torch_dtype`, HF `device_map`, whether to allow CPU inference). |
| `top_k` | any | **Recorded and audited, but not actually sent** — the normalized request adapters currently omit it. It shows up in `interactions.jsonl`/audit records under `unsupported_or_adapter_omitted_parameters` so this is visible, not silent, but don't rely on it constraining sampling. |
| `estimated_latency_seconds` | any | Used only by the preflight estimator (`src/mas_cc/planning/preflight.py:304`) to project rough total runtime; has no effect on the actual run. |
| `comet_project` | — | This one lives under `logging.options`, not `llm_provider.options` — listed here only because it's easy to misplace. See §9. |

---

## 3. `prompt`

`PromptConfig`, `src/mas_cc/config/models.py:80`.

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `prompt_family` | string | **required** | Must equal `"naming_convention_decision"` — checked against `game.options.prompt_contract` at run start (`runtime.py:201`); a mismatch raises before any provider call. |
| `prompt_version` | int | **required**, ≥ 1 | Provenance label; the actual compiled prompt is always version `1` internally (`prompts.py`'s `naming_convention_prompt`). |
| `schema_version` | int | `1` if omitted | `1` or `2`. **Omitting this field defaults to `1`** (legacy, requires non-empty `blocks`) — the loader's blanket "missing schema_version defaults to 1" rule applies here too, so you must set `schema_version: 2` explicitly to get the block-based schema this whole game is actually built on. This is the one place in the accompanying tutorial config where `schema_version` is written out rather than omitted, precisely because leaving it out changes behavior here. |
| `message_mode` | string \| null | `null` | Must be `null` or `"merge_consecutive_roles"` for this game — `"per_block"` is rejected by `runtime.py:205` (structurally valid per the generic schema, but `naming_convention` requires merged messages). |
| `block_separator` | string \| null | `null` | Must be `null` or exactly `"\n\n"` — enforced the same way. |
| `response_contract` | mapping | `{}` | See below. **Read this before touching it** — it looks like it controls the actual prompt/parsing behavior; it mostly doesn't. |
| `blocks` | tuple of strings | `()` | Legacy (`schema_version: 1`) only; forbidden when `schema_version: 2`. |
| `options` | mapping | `{}` | Not consulted by `naming_convention` today. |

**`response_contract`** (a plain mapping, not validated as strictly as `game.options`):

| Key | Type | Notes |
| --- | --- | --- |
| `type` | string | Should be `"paper_choice_reason"` (matches `game.options.response_contract: json_reason`) or `"choice_only"` (matches `game.options.response_contract: choice_only`). **As of this doc, mismatched with `game.options.response_contract` raises a `ValueError` at run start** naming both fields (`runtime.py`) — added specifically because editing only this field used to look like it switched modes while silently doing nothing; see [`running_an_episode.md` §5](../launch/running_an_episode.md#5-response-format-paper-faithful-jsonreason-or-a-bare-choice) for the full story. |
| `allowed_values` | list of strings | Must equal `game.options.actions` as a set (`runtime.py:211`) — same action pool, both places. |

**The actual instruction text and parser are controlled entirely by `game.options.response_contract`/`parser_contract` (§5), not by this section.** This section exists because `PromptConfig` is the generic, game-agnostic schema — `naming_convention` just doesn't use most of it.

---

## 4. `game`

`GameConfig`, `src/mas_cc/config/models.py:158` — the generic fields every game shares.

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `type` | string | **required** | `"naming_convention"` for this reference. (Also registered: `"toy_coordination"`.) |
| `population_size` | int | **required**, ≥ 2 | Total agents. Every round samples one uniformly-random distinct pair from this population. |
| `horizon` | int | **required** | Number of pair interactions (rounds) the episode runs for. Fixed-horizon only — see `stop_on_convergence` in §5, which is currently locked to `false`. |
| `topology` | string | `"complete"` | **Only `"complete"` is accepted** — anything else raises `"the Phase 6 paper profile supports complete topology only"`. Present in the schema for future games/topologies, not currently adjustable here. |
| `options` | mapping | `{}` | naming_convention-specific — all of §5. |

---

## 5. `game.options` — naming_convention-specific

Read from `NamingConventionGameSpec.from_config`/`.validate()` (`src/mas_cc/games/naming_convention/game.py:49-131`). **Several of these look like options but are currently locked to one paper-faithful value** — set anything else and construction raises immediately, before any provider call. They're listed so you know what's actually there and why changing them fails, not because they're meant to be tuned.

### Actually adjustable

| Key | Type | Default | Constraint |
| --- | --- | --- | --- |
| `actions` | list of strings | `["Q", "M"]` | At least 2, unique, non-empty, single-line. |
| `memory_size` | int | `5` | ≥ 0. How many past interactions are visible to an agent when deciding (a rolling window over its full private history, which is unbounded and always fully recorded regardless of this cap). |
| `invalid_response_retries` | int | `2` | ≥ 0. Total attempts per decision = this + 1. Exhausting it raises `InvalidConventionResponse`/`DecisionLoopExhausted` and aborts the episode. |
| `expected_validation_failure_rate` | float | `0.0` | In `[0, 1]`. **Only used by the preflight cost estimator** (expected attempts per request = `Σ p^k` for `k` in `0..invalid_response_retries`) — has no effect on actual retry behavior. |
| `response_contract` | string | `"json_reason"` | `"json_reason"` (paper-faithful, requires a reason) or `"choice_only"` (bare value, no reason — a deliberate departure from paper fidelity). Must pair with `parser_contract` below; see §3 for the matching `prompt.response_contract.type`. |
| `parser_contract` | string | `"tolerant_paper_object_v1"` | Must pair with `response_contract`: `"strict_json_reason_v1"` or `"tolerant_paper_object_v1"` for `json_reason`; `"choice_only_v1"` for `choice_only`. `tolerant_paper_object_v1` additionally accepts a JSON object, a Python literal, or the paper's semicolon-separated `{'value': '..'; 'reason': '..'}` form; `strict_json_reason_v1` accepts only real JSON. |
| `representative_memory_size` | int | `1` | ≥ 0, capped at `memory_size`. How many memory entries the *representative* prompt scenario simulates for cost/token estimation (`game.call_plan`, used by both preflight commands). Does not affect actual gameplay — purely a preflight-accuracy knob. |
| `selected_audit_interactions` | list of ints | `[1]` | Which interaction indices get included in `selected_audit_traces.jsonl`/`selected_block_traces.jsonl`. Only consulted by `mas-cc game run`/`mas-cc inspect phase 6` (`src/mas_cc/cli/naming_convention.py:348`) — **not** read by `mas-cc game episode`, which uses `logging.options.prompt_examples`/`detailed_prompt_audit` instead (see [`running_an_episode.md` §3](../launch/running_an_episode.md#3-everything-else-is-config-not-flags)). |

### Locked to one value (present in the schema, not currently adjustable)

| Key | Required value | What happens otherwise |
| --- | --- | --- |
| `pair_sampling` | `"uniform_two_distinct"` | `ValueError: pair_sampling must be uniform_two_distinct` |
| `simultaneous_pair_decisions` | `true` | `ValueError: naming_convention requires simultaneous pair decisions` |
| `randomize_presented_action_order` | `true` | `ValueError: the paper-faithful profile requires randomized action order` |
| `prompt_contract` | `"naming_convention_decision"` | `ValueError: prompt_contract must be naming_convention_decision` |
| `success_payoff` | `100` | `ValueError: the Phase 6 paper-faithful payoff profile is +100/-50` |
| `failure_payoff` | `-50` | same as above |
| `stop_on_convergence` | `false` | `ValueError: Phase 6 base profile uses fixed-horizon stopping` — no early-stopping-on-consensus is implemented yet, only fixed horizon |

---

## 6. `execution`

`ExecutionConfig`, `src/mas_cc/config/models.py:187`.

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `seed` | int | `0` | Root seed; per-request seeds are deterministically derived from it (`Seed.derive(...)`), so a run is fully reproducible given the same config + provider responses. |
| `repetitions` | int | `1` | Episodes per `mas-cc experiment run`/`preflight`. **Ignored by `mas-cc game episode`/`game preflight`**, which always run exactly one episode using `seed` directly — see [`running_an_episode.md` §1](../launch/running_an_episode.md#1-the-two-commands). |
| `parallelism` | int | `1` | Concurrent episodes in the experiment/grid path. Not applicable to `game episode` (one episode, no batching). |
| `fail_fast` | bool | `true` | Experiment/grid path only: abort remaining episodes on the first failure. |
| `timeout_seconds` | float \| null | `null` | Declared but not currently enforced anywhere in the codebase — don't rely on it. |

---

## 7. `pricing`

`PricingConfig`, `src/mas_cc/config/models.py:354`. Resolving this performs no I/O by itself; it only decides *how* a price quote gets fetched.

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `mode` | string | `"offline"` | `"offline"` (a static, checked-in price table), `"cached"` (a previously-fetched snapshot from `cache_path`), or `"live"` (fetch fresh — currently only implemented for `llm_provider.type: university`). |
| `cache_path` | string \| null | `null` | Required when `mode: cached`. |
| `max_age_seconds` | float | `86400.0` | Freshness window for `cached`/`live` quotes. |
| `require_fresh_at_launch` | bool | `true` | If true, a stale quote blocks launch instead of being silently accepted. |
| `fallback_policy` | string | `"deny"` | What happens when `mode: live` isn't available for the configured provider: `"deny"` (refuse to launch) or `"offline"` (fall back to the static table). |
| `explicit_unknown_price_override` | bool | `false` | Set `true` to launch anyway when pricing can't be determined (e.g. `mock`/`gemma_local` under `mode: offline` — there's no real price table entry for them). Needed for every mock-provider smoke test in this doc's companion manuals. |

---

## 8. `budget`

`BudgetConfig`, `src/mas_cc/config/models.py:378`. Enforced, not advisory — a run that would exceed any of these fails closed before spending anything past the limit.

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `accounting_unit` | string | `"unknown"` | Must match the unit the resolved pricing quote reports in, or budget checks raise `"Request estimate and budget use different accounting units."` (`USD` for most local/offline setups, `proxy_accounting_unit` for the University proxy). |
| `system_max_cost_per_run` | float \| null | `null` (unbounded) | A system-wide ceiling, separate from the run-specific one below — see `_budgets` in `src/mas_cc/cli/game.py` for how the two combine. |
| `max_cost_per_run` | float \| null | `null` | Run-specific cost ceiling. |
| `max_provider_requests` | int \| null | `null` | Hard cap on completion requests (including validation retries). |
| `max_input_tokens` / `max_output_tokens` | int \| null | `null` | Token ceilings, checked against the estimator in `src/mas_cc/planning/`. |
| `allow_unbounded_paid_requests` | bool | `false` | Set `true` to permit requests whose cost can't be bounded in advance — normally refused. |

---

## 9. `logging`

`LoggingConfig`, `src/mas_cc/config/models.py:209`.

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `level` | string | `"INFO"` | Stored/serialized only — no `logging.basicConfig()` call anywhere in the codebase consumes it. Console noise is controlled by the `logging.options` keys below, not this. |
| `console` | bool | `true` | Same as above — not currently wired to anything. |
| `audit` | bool | `true` | **Not wired to anything either.** Looks like the master audit switch; isn't. Use `logging.options.detailed_prompt_audit.enabled` instead. |
| `comet` | bool | `false` | Honored by `mas-cc game episode`/`mas-cc inspect phase 7`. **Hardcoded off** by `mas-cc experiment run` regardless of this value (one `mas_cc` experiment would otherwise fan out into N remote Comet experiments). Only aggregate round metrics are ever sent — never prompt/response content. |

`logging.options` (free-form mapping — every key below is opt-in, add only what you need):

| Key | Type | Default | Effect |
| --- | --- | --- | --- |
| `comet_project` | string | `"mas-cc"` | Comet project name. |
| `progress` | bool | `true` | `mas-cc game episode`: nested tqdm (Rounds/Decisions) when stdout is a real TTY; one flushed `round N/H complete` line per round otherwise (e.g. under `conda run`, which never attaches a TTY even with `--live-stream`). |
| `show_metrics` | bool | `false` | `mas-cc game episode`: print the game's declared metrics once at the end. Requires `metrics.enabled: true` (the default, §12). |
| `prompt_examples.count` | int | `0` | `mas-cc game episode`: write this many rounds' prompts as readable Markdown to `<output-dir>/.../prompts/round_NNN.md`, via `PromptMarkdownLogger`. One file per distinct round (first decision seen in each), up to `count` rounds. |
| `detailed_prompt_audit` | mapping | `{enabled: false}` | The raw-JSON path — `audit_traces.jsonl`/`prompt_block_traces.jsonl`. See `DetailedAuditPolicy` (`src/mas_cc/observability/audit.py`) for the full sub-schema: `enabled`, `log_every_n_rounds`, `always_log_first_n_rounds`, `max_logged_prompts_per_game`, `max_logged_prompts_per_run`, `always_log_provider_errors` (default `true`), `always_log_invalid_responses` (default `true`). |

**Always on, regardless of any of the above:** rejected attempts (invalid response or provider
error) are *always* written as Markdown to `<output-dir>/.../failures/round_NNN_attempt_K_*.md` by
`mas-cc game episode` — not config-gated, because you don't know you need that record until the
failure has already happened. See [`running_an_episode.md` §3](../launch/running_an_episode.md#3-everything-else-is-config-not-flags).

---

## 10. `storage`

`StorageConfig`, `src/mas_cc/config/models.py:234`.

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `output_dir` | string | `"results"` | Default destination for `mas-cc game episode`/`mas-cc game preflight` when `--output-dir` is omitted. |
| `format` | string | `"jsonl"` | Declared, not currently branched on anywhere — every writer uses JSONL/CSV/Markdown directly regardless of this value. |
| `checkpoints` | bool | `true` | Whether `RunRecorder` writes `.checkpoints/checkpoint.json` after each interaction (used for experiment-path resume; a single `game episode` doesn't itself resume). |
| `overwrite` | bool | `false` | Declared, not currently checked by `game episode`/`experiment run` (both target a fresh, timestamp/seed-qualified directory each time). |
| `options` | mapping | `{}` | Not consulted today. |

---

## 11. `control`

`ControlConfig`, `src/mas_cc/config/models.py:281`. A provider-independent way to force specific agents' decisions instead of asking the LLM — the "committee control layer."

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `mechanism` | string | `"none"` | `"none"` (never overrides anything) or `"forced_action"` (see below). Unknown mechanisms raise via the control registry (`src/mas_cc/control/registry.py`), same pattern as `llm_provider.type`. |
| `options` | mapping | `{}` | Mechanism-specific. |

**`forced_action` options** (`src/mas_cc/control/forced_action.py`):

| Key | Type | Required | Notes |
| --- | --- | --- | --- |
| `agent_ids` | list of strings | yes | Non-empty. These agents never call the LLM — every decision for them is `forced_value`, at zero token/request cost. |
| `forced_value` | string | yes | Non-empty; should be one of `game.options.actions`, though this isn't cross-checked against the action pool at construction time. |
| `until_interaction` | int \| null | no | `null` (forced for the whole episode) or a positive interaction index — a "pulse" that expires after that round, after which those agents resume ordinary LLM-backed decisions. |

---

## 12. `metrics`

`MetricsConfig`, `src/mas_cc/config/models.py:306`. The metric *instances* live in code
(`src/mas_cc/games/naming_convention/metrics.py`'s `METRICS`); this section only gates whether
they run and what may leave the machine.

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `enabled` | bool | `true` | Whether streaming/final metrics get computed at all (`metrics/streaming.csv`, `metrics/final.csv`, and `logging.options.show_metrics`'s console printout all depend on this). |
| `comet_export` | list of strings | `[]` | Metric *names* (e.g. `population_action_share_q`, `dominant_action_share`) allowed to reach Comet as aggregate values. Everything not listed here stays local even if `logging.comet: true`. |

---

## 13. `analysis`

`AnalysisConfig`, `src/mas_cc/config/models.py:258`. Offline, post-hoc analysis over a *completed* grid/experiment — not part of running one episode.

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `enabled` | bool | `false` | **Not read anywhere in the codebase today**, by any command — `mas-cc analysis empowerment` takes its input directory as a CLI flag (`--grid-dir`), not from this config section. Kept in the schema for a planned config-driven analysis step; currently vestigial, same as `logging.audit`/`storage.format`. |
| `estimators` | list of strings | `[]` | Same — currently unread. |
| `options` | mapping | `{}` | Same — currently unread. |

---

## 14. `experiment`

`ExperimentConfig`, `src/mas_cc/config/models.py:330`. Pure human-facing identity — nothing here is validated against game mechanics.

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `name` | string | `"unnamed-experiment"` | Used to build the output path (`results/<game.type>/<name>/<name>-<execution.seed>/...`) and, if `logging.comet: true`, the Comet run name. |
| `description` | string | `""` | Free text. |
| `tags` | list of strings | `[]` | Free text, no validated vocabulary. |
| `metadata` | mapping | `{}` | Free-form, secret-free (checked before it can reach any artifact). |

---

## 15. Minimal valid config

Every `**required**` field from §2–4, everything else at its default — this is the smallest config
that constructs without error (still needs real credentials to actually run against a paid
provider; swap `llm_provider` for the `mock` shape in §2 to run it for free):

```yaml
llm_provider:
  type: university
  model: gwdg/qwen3-30b-a3b-instruct-2507
  credentials_env: POTSDAM_API_KEY
  base_url_env: BASE_POTSDAM_LLM_URL
prompt:
  schema_version: 2   # required explicitly - omitting it defaults to legacy schema 1
  prompt_family: naming_convention_decision
  prompt_version: 1
  response_contract:
    type: paper_choice_reason
    allowed_values: [Q, M]
game:
  type: naming_convention
  population_size: 4
  horizon: 6
```

Everything from §6 onward (`execution`, `pricing`, `budget`, `logging`, `storage`, `control`,
`metrics`, `analysis`, `experiment`) is optional and takes its documented default if omitted —
though in practice you'll want at least `pricing.mode`/`budget.max_cost_per_run` set deliberately
rather than left at their defer-everything defaults before running against a real, paid provider.
