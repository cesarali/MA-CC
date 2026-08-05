# Running one episode in `mas_cc` — a manual

This is the practical, CLI-first companion to [`building_a_game.md`](../building/building_a_game.md)
(what one episode of a game actually *is*, walked through in explicit Python) and
[`running_an_experiment.md`](running_an_experiment.md) (many episodes, priced and batched). This
manual covers the two commands that exist for exactly one thing: **run one episode, or price one
episode, from the command line.**

**Accompanying config:**
[`configs/runs/naming_convention_tutorial_university_v3.yaml`](../../../configs/runs/naming_convention_tutorial_university_v3.yaml) —
4 agents, 6 interactions, live against the University proxy. The same file `building_a_game.md`
walks through section by section.

## Table of contents

1. [The two commands](#1-the-two-commands)
2. [Running the example](#2-running-the-example)
3. [Everything else is config, not flags](#3-everything-else-is-config-not-flags)
4. [Reading the output](#4-reading-the-output)
5. [Response format: paper-faithful JSON+reason, or a bare choice](#5-response-format-paper-faithful-jsonreason-or-a-bare-choice)
6. [What still exists but isn't this](#6-what-still-exists-but-isnt-this)
7. [Where to go next](#7-where-to-go-next)

---

## 1. The two commands

```bash
mas-cc game preflight --config <path> [--output-dir <dir>]
mas-cc game episode    --config <path> [--output-dir <dir>] [--no-preflight]
```

Almost the whole CLI surface for this layer is two flags each: `--config`, and an optional
`--output-dir` (defaults to `config.storage.output_dir` — the tutorial config sets this to
`results`). Everything else about *how* the episode runs — progress bars, Comet, metrics, how many
example prompts get written and in what format — is read from the config, not from a flag. Section
3 is the full list. `--no-preflight` on `game episode` is the one deliberate exception; see below.

**`game preflight` makes zero provider calls.** It resolves the config, builds the call plan, and
reports a cost/token/request estimate — the only network call it makes is a live *price quote*
lookup (metadata, not a completion) if `pricing.mode: live`.

**`game episode` really runs the episode**, and runs that same preflight estimate first, by
default, printing it in a banner before anything is sent — experiment name, game/provider/model,
population and horizon, the resolved budget ceiling, the preflight estimate, and where output is
landing, so you know what's about to run and where to look for it afterward. It refuses to launch
if the estimate isn't `permitted`. Pass **`--no-preflight`** to skip that estimate step and its
gate — useful when you're iterating quickly and already trust the config. This does *not* turn off
spend protection: the runtime `RuntimeBudgetGuard` still enforces `budget.*` live, during execution,
regardless of this flag — `--no-preflight` only skips the *upfront estimate*, not the enforcement.

Both are implemented in `src/mas_cc/cli/episode.py`, and both currently require
`game.type: naming_convention` — the only game with a metrics module and a production runtime
wired up today.

---

## 2. Running the example

Requires `POTSDAM_API_KEY` and `BASE_POTSDAM_LLM_URL` (same as `building_a_game.md`), since the
accompanying config points at the real University provider.

```bash
# Price it first - no provider call happens here.
conda run --live-stream -n MA-CC mas-cc game preflight \
  --config configs/runs/naming_convention_tutorial_university_v3.yaml

# Then actually run it, bounded by the config's budget.max_cost_per_run: 0.25.
conda run --live-stream -n MA-CC mas-cc game episode \
  --config configs/runs/naming_convention_tutorial_university_v3.yaml
```

**Use `--live-stream` (alias `--no-capture-output`).** Without it, `conda run` buffers the whole
subprocess's stdout/stderr and dumps it all at once when the process exits — you'd see nothing
until the episode is already over, defeating the point of live progress. `--live-stream` streams
it as it's produced instead.

That alone isn't quite enough, though: `conda run` never attaches a real TTY to the subprocess
even with `--live-stream` — it only stops *buffering*, it doesn't allocate a pty — and the nested
tqdm bars from section 3 need a real TTY to redraw in place. `mas-cc game episode` accounts for
this: when `logging.options.progress` is on (the default) but stdout isn't a TTY, it falls back to
printing one flushed `round N/H complete` line per round instead of silently doing nothing. So
under `conda run --live-stream` you'll see plain progress lines rather than bars — both are driven
by the same `logging.options.progress` switch, you don't need to choose between them.

Both commands land under
`results/naming_convention/naming-convention-tutorial-university-walkthrough/<run-id>/` (the first
from the config's `experiment.name`, the second `<experiment.name>-<execution.seed>`) —
`preflight`'s estimate in a `preflight/` subdirectory, `episode`'s real output as siblings of it.
Pass `--output-dir some/other/path` on either command if you want to redirect a one-off run
without editing the config.

`game episode` opens with a banner — what's about to run and where it's landing — before it sends
a single request:

```text
Episode: naming-convention-tutorial-university-walkthrough
  Game:          naming_convention v1  (population 4, horizon 6)
  Provider:      university / gwdg/qwen3-30b-a3b-instruct-2507
  Prompt:        naming_convention_decision v1  [def:33232f05...]
  Budget:        0.25 proxy_accounting_unit
  Preflight:     expected ... / conservative ... — permitted
  Output:        results/naming_convention/naming-convention-tutorial-university-walkthrough/naming-convention-tutorial-university-walkthrough-20260803
```

With `--no-preflight`, the last two lines read `Preflight: expected skipped / conservative skipped
— skipped (--no-preflight)` instead — everything else is unchanged. Then progress (section 3), then
a short summary when it finishes — how many interactions ran, why it stopped, where the output
landed, and the Comet status.

---

## 3. Everything else is config, not flags

All of these live under `logging.options` in the run config (a free-form mapping — no schema
change needed to add or omit any of them). Every one defaults to off/minimal if you don't set it.

| Config key | Default | What it controls |
| --- | --- | --- |
| `logging.comet` | `false` | Whether this episode reports aggregate metrics to Comet (`logging.options.comet_project` names the project). Prompt content is never sent — see `CometMetricSink` in `src/mas_cc/observability/recorder.py`. |
| `logging.options.progress` | `true` | Nested tqdm: an outer **Rounds** bar (total = `game.horizon`) and an inner **Decisions** bar (resets each round). Auto-off if stdout isn't a TTY, regardless of this setting. |
| `logging.options.show_metrics` | `false` | Print the game's declared metrics (`src/mas_cc/games/naming_convention/metrics.py`) once at the end — final values for `StreamingMetric`s, computed values for `FinalMetric`s. Requires `metrics.enabled: true` (the default). |
| `logging.options.prompt_examples.count` | `0` | Write this many example prompts as **Markdown**, one per distinct round (first attempt encountered in each of the first *N* rounds), to `<output-dir>/.../prompts/round_NNN.md`. Uses `PromptMarkdownLogger` (`src/mas_cc/prompts/reporting.py`) — human-readable messages, block provenance, response contract, no JSON to parse. |
| `logging.options.detailed_prompt_audit` | `{enabled: false}` | The raw-JSON path: `audit_traces.jsonl` and `prompt_block_traces.jsonl`. Independent of `prompt_examples` — turn this on instead of/alongside it if you want the full structured record rather than a curated Markdown sample. Same `DetailedAuditPolicy` shape used by the experiment/grid path; see `src/mas_cc/observability/audit.py` for `log_every_n_rounds`/`max_logged_prompts_per_run`/etc. |

**One thing on this list isn't config-gated, on purpose:** every rejected attempt — an invalid
response the parser couldn't turn into an action, or a provider error — gets written as Markdown to
`<output-dir>/.../failures/round_NNN_attempt_K_*.md`, **always**, regardless of `prompt_examples`
or `detailed_prompt_audit`. Each file has the exact sent messages, the raw response text received,
and the specific reason it was rejected (`response.value: must contain an answer-first quoted value
field`, etc.). If `game.options.invalid_response_retries` is exhausted for a decision, the episode
raises `InvalidConventionResponse`/`DecisionLoopExhausted` and stops — before that exception
propagates, `game episode` prints how many rejected attempts were recorded and where. The reasoning
is the same as `DetailedAuditPolicy.always_log_invalid_responses` upstream: you don't know you need
this record until the failure has already happened, so it isn't something you can opt into after
the fact.

Example — everything on, for a debugging session:

```yaml
logging:
  comet: false
  options:
    progress: true
    show_metrics: true
    prompt_examples:
      count: 2
```

**One field that looks related but isn't:** `logging.audit` (the top-level boolean, not
`logging.options.detailed_prompt_audit`) is not wired to anything — it's read into
`RunConfig.logging.audit` and serialized back out in `resolved_config.yaml`, and nothing else in
the codebase branches on it. Don't reach for it; use `logging.options.detailed_prompt_audit` for
the JSON traces or `prompt_examples.count` for Markdown.

**What's deliberately *not* config-controllable here:** the lightweight bookkeeping streams —
`events.jsonl`, `api_call_status.jsonl`, `usage_cost.jsonl`, `budget_events.jsonl` — are always
written. They back the budget guard's accounting and contain no prompt or response content, only
event types, hashes, token counts, and cost. Making those opt-in too would mean changing
`RunRecorder` itself (`src/mas_cc/observability/recorder.py`), which is shared with the
experiment/grid path — out of scope here since that path was explicitly left for later.

---

## 4. Reading the output

```text
results/naming_convention/naming-convention-tutorial-university-walkthrough/naming-convention-tutorial-university-walkthrough-20260803/
├── resolved_config.yaml         # secret-free, fully expanded RunConfig — written before the first
│                                 # provider call, so it's there even if the episode fails; this is
│                                 # what you'd hand to `mas-cc game episode --config` to replay it
├── events.jsonl                 # every event, operational fields only, no prompt content
├── api_call_status.jsonl, usage_cost.jsonl, budget_events.jsonl
├── local_metrics.csv            # per-round success/payoff/attempt counts (always written)
├── checkpoint_manifest.json, .checkpoints/checkpoint.json
├── comet_summary.json           # "disabled"/"active"/"unavailable" - never prompt content
├── metrics/
│   ├── streaming.csv            # the game's StreamingMetrics, one row per (round, agent, metric)
│   └── final.csv                # the game's FinalMetrics
├── prompts/                     # only if prompt_examples.count > 0
│   └── round_001.md, round_002.md, ...
├── failures/                    # only if something was rejected — always on when it happens
│   └── round_NNN_attempt_K_*.md   # exact sent messages + exact raw response + rejection reason
├── audit_traces.jsonl           # only if detailed_prompt_audit.enabled
└── prompt_block_traces.jsonl    # only if detailed_prompt_audit.enabled
```

I ran this exact layout twice with a mock-provider copy of the accompanying config (no live
network call or spend): once with `prompt_examples.count: 2` and `show_metrics: true` — confirmed
both `round_001.md`/`round_002.md` are readable Markdown (system/user messages, block provenance,
response contract, no JSON), the console printed a final metrics block and nothing else, and — with
`detailed_prompt_audit` left at its default — no `audit_traces.jsonl`/`prompt_block_traces.jsonl`
file existed at all, and none of `events.jsonl`/`usage_cost.jsonl`/`checkpoint_manifest.json`/
`comet_summary.json` contained any prompt text. Once more with the mock provider forced to return
an unparseable response — reproduced the same `InvalidConventionResponse`/`DecisionLoopExhausted`
you get from a real model that won't follow the answer-first contract, confirmed `failures/`
contained one Markdown file per rejected attempt with the raw response text and rejection reason
intact, confirmed the CLI printed the failure count and path to stderr before the exception
propagated, and confirmed `resolved_config.yaml` was present in that failed run's output directory
too (it's written before the first provider call now, not after a successful finish — a run that
fails is exactly when you most want the exact config that produced it). Running the example
yourself, with your own University credentials, is how you get a real model-driven episode.

---

## 5. Response format: paper-faithful JSON+reason, or a bare choice

If an episode fails with `InvalidConventionResponse`/`DecisionLoopExhausted` ("produced no valid
action after N validation attempts"), check `failures/` from section 4 first — it has the exact
raw text the model returned, and its **finish reason**. Two different failures look identical from
the error message alone but need different fixes:

- **`Finish reason: length`** — the response was *cut off* before its closing `}`, not malformed.
  The model was mid-sentence when `llm_provider.max_output_tokens` ran out. Fix: raise
  `max_output_tokens` (128 is tight for a full reasoned JSON answer from a verbose model), or switch
  to `choice_only` below so there's no reason text to run out of room for.
- **`Finish reason: stop`** (or anything else) with genuinely broken text — the model finished on
  its own but didn't follow the contract. That's the model struggling with the format itself.

`game.options.response_contract` now has two supported values:

| Value | Instruction the model receives | Parser | Reason captured? |
| --- | --- | --- | --- |
| `json_reason` (default) | The paper's exact answer-first form: `{'value': '<ACTION>'; 'reason': '<YOUR REASON>'}` | `strict_json_reason_v1` or `tolerant_paper_object_v1` | Yes |
| `choice_only` | "Reply with exactly one word... nothing else." | `choice_only_v1` | No — `parsed_reason` is `null` |

```yaml
game:
  options:
    response_contract: choice_only
    parser_contract: choice_only_v1
```

Both fields have to change together — `game.py`'s `NamingConventionGameSpec` rejects a mismatched
pair (`choice_only` response_contract with a JSON parser, or vice versa) at config-load time,
before any provider call.

**This is a genuine departure from paper fidelity, not a free technical toggle** — the
`json_reason` default exists because the game is scoped as a faithful replication of a specific
paper's protocol, which calls for a justified, reasoned choice, not just an action. `choice_only`
trades that away for a format small/weaker models are far less likely to fail on (and are much less
likely to get truncated on, since there's no reason text to fill `max_output_tokens` with). Decide
which one you want on scientific grounds, not just to make an error go away — if you need reasons
for your analysis, `choice_only` isn't a substitute for prompting a more capable model.

**The `prompt.response_contract.type` field (top-level `prompt:` section) does not control this —
`game.options.response_contract`/`parser_contract` do, and only those two.** Editing just
`prompt.response_contract.type` looks like it switches modes — it doesn't; the message actually
sent is unchanged. This bit me once while building this feature and it bit the first person who
tried it too, so it's no longer just documented here: the two fields are now checked for
consistency at the start of every real run (`run_naming_convention_game` in
`src/mas_cc/games/naming_convention/runtime.py`), and a mismatch — `prompt.response_contract.type:
choice_only` next to `game.options.response_contract: json_reason`, or vice versa — raises a clear
`ValueError` naming both fields, before any provider call, instead of silently running the wrong
mode. Keep all three in sync:

```yaml
prompt:
  response_contract:
    type: choice_only        # must match game.options.response_contract below
game:
  options:
    response_contract: choice_only
    parser_contract: choice_only_v1
```

I verified this three ways with the mock provider: `response_contract: choice_only` end to end (the
rendered prompt's instruction changed to the one-word form, parsing succeeded, episode completed
normally); the unmodified `json_reason` config still runs exactly as before; and a deliberately
mismatched pair (`prompt.response_contract.type: choice_only` with `game.options.response_contract:
json_reason` left at its default) now fails immediately with the new `ValueError`, config-load-time
fast, not three retries and a wasted budget later.

---

## 6. What still exists but isn't this

`mas-cc game run`, `mas-cc inspect phase N`, and `mas-cc experiment run --config <path>` with
`execution.repetitions: 1` are all still in the codebase, unchanged. They're pinned down by
`tests/mas_cc/test_cli_and_inspection.py` and `tests/mas_cc/test_phase7_observability.py` as
phase-regression fixtures from how this codebase was built — each carries pass/fail checks against
a specific frozen fixture (`frozen_prompt_wire_parity`, etc.) that only make sense for their own
smoke-test configs, not general use. `mas-cc game episode`/`mas-cc game preflight` are the
intended replacement for running *your* configs; the phase commands stay as internal regression
tooling, not as alternative user-facing options.

---

## 7. Where to go next

- **What an episode actually is, underneath this command:** [`building_a_game.md`](../building/building_a_game.md).
- **Running many episodes as a priced, concurrent, resumable batch:** [`running_an_experiment.md`](running_an_experiment.md) —
  unchanged by this manual; the CLI consolidation here is single-episode only. `response_contract`/
  `parser_contract` apply there too, unchanged — they're game-level config, not specific to `game episode`.
- **The real files:** `src/mas_cc/cli/episode.py`, `src/mas_cc/observability/{recorder,audit}.py`,
  `src/mas_cc/prompts/reporting.py`, `src/mas_cc/games/naming_convention/{game,prompts,parsing}.py`.
