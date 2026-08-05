# Building a game in `mas_cc` — a manual

This is the "read and understand" companion. For "read and type," see
[`notebooks/tutorial_build_a_game.ipynb`](../../../notebooks/tutorial_build_a_game.ipynb) — it
builds a brand-new toy game (`SignChoiceGame`) from the same abstract classes described here, live
in the notebook, so you can watch every piece get assembled and then run it.

This manual explains the same abstract classes, but against the **real, already-implemented**
naming-convention game, since that's the one you actually run. Each concept below gets two
passes: a **Technical** explanation (the actual types and behavior) and, right after it, an
**In plain terms** explanation (the intuition, using one running analogy throughout).

**Accompanying config:** [`configs/runs/naming_convention_tutorial_university_v3.yaml`](../../../configs/runs/naming_convention_tutorial_university_v3.yaml) —
a small (4 agents, 6 interactions), cheap, ready-to-run example config against the University
proxy. Section 8 below shows the code that runs it.

## Table of contents

1. [The big picture, and one running analogy](#1-the-big-picture-and-one-running-analogy)
2. [Agents: identity + memory](#2-agents-identity--memory)
3. [The `Game` contract](#3-the-game-contract)
4. [Prompts: `PromptBlock` and `FullPrompt`](#4-prompts-promptblock-and-fullprompt)
5. [The decision loop](#5-the-decision-loop)
6. [Metrics](#6-metrics)
7. [Configuration: anatomy of the accompanying file](#7-configuration-anatomy-of-the-accompanying-file)
8. [Running the example](#8-running-the-example)
9. [Where to go next](#9-where-to-go-next)

---

## 1. The big picture, and one running analogy

| Layer | What it is | Where it lives |
| --- | --- | --- |
| LLM providers | Normalized access to a model (mock, University, OpenAI, local Gemma) | `src/mas_cc/llm_providers/` |
| Prompts | Composable, versioned, value-bearing message construction | `src/mas_cc/prompts/`, `games/naming_convention/prompts.py` |
| Agents | Identity + accumulated memory, generic across every game | `games/protocols.py` |
| The `Game` contract | What every game must implement | `games/protocols.py`, `games/naming_convention/game.py` |
| The decision loop | Ask / validate / retry / log — shared by every game | `src/mas_cc/runtime/loop_runtime.py` |
| Metrics | Scientific quantities computed from what happened | `src/mas_cc/metrics/`, `games/naming_convention/metrics.py` |
| Configuration | One resolved, versioned description of a run | `src/mas_cc/config/`, `configs/runs/*.yaml` |

**In plain terms:** think of this the way you'd think of a physics experiment.

- The **provider** is the raw instrument — it takes a question, returns a reading. It has no idea
  what experiment you're running.
- The **decision loop** is the measurement *procedure*: ask, check whether the reading is
  acceptable, retry if not, write down every attempt. The procedure doesn't care what "acceptable"
  means for your particular experiment — it just needs to be told.
- The **game** is the experiment itself: what's being measured, who the participants are, what
  counts as a valid reading, how the system evolves.
- **Prompts** are how you phrase the instructions and the current state of the sample to the
  instrument for this experiment.
- **Agents** are the samples under study — each one keeps its own logbook page; nobody reads
  anybody else's page unless the experiment's design explicitly allows it.
- **Metrics** are the derived quantities you compute afterward (a rate, a mean, a first-crossing
  time) — not the raw readings themselves.
- The **configuration** file is the lab notebook entry: exactly how this run was set up, so it can
  be reproduced.

Keep this picture in mind — every section below is one row of that table, explained twice.

---

## 2. Agents: identity + memory

**Technical:** `AgentState` (`games/protocols.py`) is a frozen dataclass:

```python
AgentState(agent_id: AgentId, score: float, memory: tuple[Mapping[str, Any], ...], attributes: Mapping[str, Any])
```

It is generic — every game's agents are built from this same class, not a new one per game. A
`GameState` holds a tuple of these, one per agent, each with its *own* `memory` tuple; nothing is
shared between them. Naming-convention subclasses it as `ConventionAgentState`
(`games/naming_convention/records.py`) purely to add typed readers over that same `memory` field:

- `private_history` — turns each raw memory entry into a typed `PrivateMemoryEntry`
  (`own_action`, `partner_action`, `payoff`, `success`).
- `visible_history(window)` — the last `window` entries, for the memory an agent is actually
  shown.
- `lifetime_score` — a typed read of `score`.

- `committed_action` — the choice this agent currently stands on. `None` until the agent's first
  interaction, then updated by `apply_transition` on every round it plays. This is what the
  population-share metrics read, so "where does the population stand" never requires replaying
  anyone's history.

**In plain terms:** an agent is a lab sample with its own logbook page — an ID, a running score,
and a private, append-only history of what happened to it. Two different agents are always two
different logbook pages; reading one never tells you what's on another's.

---

## 3. The `Game` contract

Before the method table, the four objects the methods pass around are easy to conflate because
three of them contain the word "state" or overlap in what they describe. They are not
interchangeable, and the difference matters for understanding what each method actually does.

**Technical — four distinct objects:**

| Object | Scope | What it actually holds | Who constructs it |
| --- | --- | --- | --- |
| `GameState` | **The whole population**, at one instant | `turn` (interaction count so far), the full `agents` tuple (every agent, not just the participants of the current interaction), `terminated`, game-specific `data` | Returned by `initialize` and by `apply_transition` (as `Transition.next_state`) |
| `AgentState` | **One single agent**, at one instant | That agent's own `agent_id`, `score`, `memory`, `attributes` — one element of `GameState.agents` | Never constructed directly by the loop; read out of `GameState.agents` via `state.agent(agent_id)` |
| `Observation` | **One agent's view**, for one interaction | Only what that agent is *allowed to see* this interaction — a subset the game deliberately chose, not the full `GameState` and not even that agent's full `AgentState` | Built by `construct_observations`, one per participant |
| `Transition` | **The result of one interaction** | `actions` taken, `payoffs`, and critically `next_state` — a brand-new `GameState`, plus whether the interaction terminated the run | Returned by `apply_transition`; nothing before it exists yet |

So: `GameState` is the whole world; `AgentState` is one row of it; `Observation` is a deliberately
narrowed *view* onto that world for one agent (not a kind of state at all — it's a read, not a
store); and `Transition` is what you get back after acting, containing the *next* `GameState`.
There is exactly one `GameState` in existence at a time (plus the one about to replace it, inside
a not-yet-returned `Transition`), but many `AgentState`s (one per agent) and many `Observation`s
(one per participating agent per interaction).

**In plain terms:** `GameState` is the whole lab's status board — every sample, every reading,
right now. `AgentState` is one sample's own logbook page, pulled off that board. `Observation` is
what you'd actually hand a specific technician for the next measurement — deliberately not the
whole status board, just the slice they're allowed to see. `Transition` is the lab report after
one measurement: what was recorded, and the *updated* status board that results — the old board
isn't edited, a new one is issued.

---

**Technical — the `Game` contract itself:** `Game` (`games/protocols.py`) is an `abc.ABC`, not a
duck-typed `Protocol` — this is deliberate and was tightened up in this same development pass.
Every concrete game must subclass it and implement all of:

| Method | Takes | Returns | Role |
| --- | --- | --- | --- |
| `spec` (abstract property) | — | `GameSpec` | Stable identity: `game_type`, `version`, `description`, population/topology limits |
| `initialize(config, seed)` | `GameConfig`, seed | `GameState` | Build the *first* `GameState` — every agent starts with empty memory |
| `select_participants(state, config, rng)` | current `GameState` | tuple of `AgentId` | Who acts this interaction (reads `GameState.agents`, decides nothing about any single `AgentState`'s content) |
| `construct_observations(state, participants, config)` | current `GameState`, the selected `AgentId`s | tuple of `Observation` | The one place `AgentState.memory` actually gets read for this decision: `state.convention_agent(agent_id).visible_history(...)`, then folded into each `Observation.visible_state` as `visible_memory`/`visible_score`/`presented_actions` — the information boundary lives here |
| `build_decision_requests(state, observations, config)` | current `GameState`, the `Observation`s just built | tuple of `DecisionRequest` | Bind a concrete prompt to each `Observation` — reads the values back **out of `observation.visible_state`**, not from `AgentState` again; by this point the game has already decided what's visible |
| `parse_action(request, response)` | one `DecisionRequest`, raw model text | `Action` | Turn raw model text into a typed `Action` — no state involved at all |
| `validate_action(state, request, action, config)` | current `GameState`, the `Action` | `ValidationResult` | Is this action legal, given the current `GameState`? |
| `apply_transition(state, participants, actions, config)` | current `GameState`, the `Action`s | `Transition` | Pure, immutable update: consumes the *current* `GameState`, produces a `Transition` whose `next_state` is a *new* `GameState` |
| `detect_termination(state, config)` | current `GameState` | `str \| None` | Has the run finished? |
| `call_plan(config)` | `GameConfig` only — no `GameState` at all | `GameCallPlan` | Token/cost demand estimate for pricing/preflight; never called during actual play, and notably takes no state because it runs *before* any game exists |

`NamingConventionGame(Game)` (`games/naming_convention/game.py`) is the real implementation of all
ten. Because `Game` is an ABC, Python checks at construction time — `NamingConventionGame()` —
that every one of these is actually implemented; a game missing one fails immediately with a
`TypeError` naming exactly what's missing, rather than failing later, silently, the first time
something happens to call the missing piece.

**In plain terms:** `Game` is a recipe card that lists every step a valid experiment must define,
and the table above says, for each step, which of the four objects above it's allowed to touch.
You can't half-follow the recipe — if a step is missing, you find out the moment you try to start
the experiment, not partway through, when it's too late to notice cheaply.

---

## 4. Prompts: `PromptBlock` and `FullPrompt`

**Technical:** `PromptBlock[T]` and `FullPrompt` (`src/mas_cc/prompts/`) are the abstract,
game-independent composition layer: a block owns one named, versioned, typed value plus how to
render it; a `FullPrompt` owns the authoritative ordered tuple of blocks and a response contract.
Binding (`.bind(**values)`) always returns a *new*, immutable prompt — nothing is mutated in
place, so two agents' bound prompts never share state.

`NamingConventionFullPrompt` (`games/naming_convention/prompts.py`) is the concrete realization,
with exactly five blocks:

| Block | Binding | Role |
| --- | --- | --- |
| `description` | fixed | The task, stated once, identical for every agent |
| `rules` | fixed | The rules, stated once |
| `presented_actions` | dynamic | This agent's action choices, in its own randomized order |
| `visible_memory` | dynamic | This agent's own recent history — empty is a valid, distinct state from unbound |
| `visible_score` | dynamic | This agent's own running score |

The response contract, `NamingConventionResponseContract`, requires an answer-first
`{"value": "<Q or M>", "reason": "..."}` object. Compiling a bound prompt (`.compile(token_counter)`)
produces normalized messages, per-block and total token counts, and two hashes: a *definition*
hash (block classes/versions/order — the same for every agent) and an *instance* hash (definition
plus the actual bound values — different per agent, since each agent's memory/score differ).

**In plain terms:** `description` and `rules` are the parts of the protocol you'd read out loud
once to the whole room — every sample gets the identical wording. `presented_actions`,
`visible_memory`, and `visible_score` are what you fill in fresh on the form for *this specific*
sample before handing it to the instrument — same form, different ink each time.

---

## 5. The decision loop

**Technical:** `run_validated_decision` (`src/mas_cc/runtime/loop_runtime.py`) is the one shared
loop every game's decisions go through:

```python
async def run_validated_decision(
    *, game, state, request, game_config, provider, prompt,
    temperature, max_output_tokens, seed_for_attempt, metadata_for_attempt,
    on_attempt=None,
) -> ValidatedDecision: ...
```

It builds a `CompletionRequest`, calls `provider.complete(...)`, then checks the response through
`prompt.response_contract.validate(...)`, `game.parse_action(...)`, and
`game.validate_action(...)` — all already generic `Game`/prompt methods, never anything specific
to this function. If invalid, it retries up to `request.retry_bound`; if the provider call itself
raises (a transport failure), that is *not* retried — it's reported and re-raised immediately.
Every attempt, valid or not, is recorded in the returned `ValidatedDecision.attempts`.

`games/naming_convention/runtime.py` and `games/runner.py` (the Phase 5 generic loop) both call
into this same function — there used to be two independently hand-written, slightly diverged
copies of this loop; that duplication is exactly why it was pulled out into one place.

**In plain terms:** this is the measurement procedure. Ask the instrument, check whether the
reading passes today's acceptance test, ask again if not, write down every attempt including the
failed ones. The procedure has no opinion about what a "good reading" is for your experiment —
that's entirely supplied by the game.

---

## 6. Metrics

**Technical:** `Metric`, `StreamingMetric`, `FinalMetric` (`src/mas_cc/metrics/base.py`) are the
two computation timings: a value per interaction round, or one value computed once at episode end.
Most games reduce to "each round, each agent has a current value," so metrics are written once
against a generic `RoundView` (`src/mas_cc/metrics/generic.py`):

```python
RoundView(
    agent_values: Mapping[AgentId, Any],
    agent_targets: Mapping[AgentId, Any] | None = None,
    options: tuple[str, ...] = (),
)
```

with a small shelf of reusable metrics (`ActionSharePerOption`, `AgentCurrentValue`,
`DominantValueShare`, `FirstConsensusTime`, `AgentAbsoluteError`, `MeanAbsoluteError`). A game only
needs to write a small adapter — `games/naming_convention/metrics.py::to_round_view` reads each
agent's `committed_action` and the game's option set into a `RoundView` — and pick metrics off the
shelf via `build_metrics()`: `population_action_share_per_option`, `agent_current_action`,
`dominant_action_share`, `first_consensus_time_by_action_share`.

`ActionSharePerOption` declares `requires_game_family = "choice"`, which is checked against the
game's `GameSpec.game_family` when metrics are attached (`games/registry.py::game_metrics`) — a
share-of-options metric on a game with no options fails at wiring time instead of writing a column
of zeros.

**In plain terms:** the raw readings are what happened in each interaction (who played what,
who scored). Metrics are the numbers you actually report — "45% of the population is playing Q,"
"the population reached consensus by round 12" — computed *from* the readings, not recorded
alongside them as if they were readings themselves.

---

## 7. Configuration: anatomy of the accompanying file

`configs/runs/naming_convention_tutorial_university_v3.yaml` is written **fully expanded and
self-contained** — every value is right there in the file. Elsewhere in this codebase, run configs
usually reference reusable `component:` files (e.g. one shared University provider component,
reused by many runs) plus small `overrides:` on top; that's a real and useful pattern once you
have many run configs sharing pieces, but it means reading one file requires chasing several
others. For a tutorial config that's the opposite of what you want, so this one intentionally
inlines everything instead. Every section below is a required top-level key in a resolved
`RunConfig` (`src/mas_cc/config/models.py`).

| Section | Technical role | In plain terms |
| --- | --- | --- |
| `llm_provider` | Provider/model/sampling params, written out directly (University, default model, `max_output_tokens: 128`) | Which instrument, and how loudly to let it talk |
| `prompt` | Prompt family/version/response contract, written out directly (`naming_convention_decision`, Q/M) | Which form you're using |
| `game` | `population_size: 4`, `horizon: 6`, action set `[Q, M]`, memory sizes, payoffs — every game option written out | The experiment's size and rules: how many samples, how many trials, what a win looks like |
| `execution` | Seed, repetitions, concurrency | The random-number source and how many times to repeat |
| `pricing` | `mode: live` — fetch a fresh price quote from the proxy before launch, refuse to launch on a stale or missing one | Refuse to spend money you can't currently account for |
| `budget` | Hard caps: `max_cost_per_run: 0.25`, `max_provider_requests: 36`, token ceilings | The absolute maximum this run is allowed to spend, enforced, not advisory |
| `logging` | Audit detail, Comet **on** (`comet: true`) | How much of the run gets written down, and where |
| `storage` | Local artifact destination | Where the lab notebook pages get filed |
| `analysis` | Off here — Phase 8.1 offline analyses are a separate, later step | Not part of this run |
| `metrics` | `enabled: true`, nothing allowed to Comet by default | Compute the game's metrics; keep them local unless explicitly exported |
| `experiment` | Name, description, tags — human-facing identity | The label on the lab notebook page |

The `pricing`/`budget` sections are not decoration: `pricing.mode: live` means the run refuses to
start if it can't get a current price quote from the University proxy, and `budget` is an enforced
ceiling, not a suggestion — a run that would exceed it fails closed before spending anything.

**`schema_version` — why the file barely mentions it now.** It's **one integer per section**,
checked by `_schema_version(...)` in `src/mas_cc/config/loader.py:382` against
`SUPPORTED_SCHEMA_VERSIONS = (1,)` (`config/loader.py:42`) — not a global file version, not
something you're expected to reason about per line. When it's omitted, the loader defaults it to
`1` (same function, same line), so leaving it out changes nothing except how much noise is on the
page. The accompanying file now omits it everywhere it would just be `1` — which is everywhere
except one place: `prompt.schema_version: 2`. That one is real and matters — it selects the newer
prompt-component schema (block-based, the one this whole manual is about) instead of the legacy
Version 1 shape. That's the entire point of `schema_version` in one example: most of the time it's
invisible boilerplate at the default, and the one time it isn't, it's telling you which shape of a
section you're actually getting.

---

## 8. Running the example

**There is also a CLI for this** — `mas-cc game run --config <path> --output-dir <dir>` runs any
resolved game and writes trajectory artifacts (it dispatches to a naming-convention-specific path
automatically when `game.type == "naming_convention"`). It's real and it works, but it doesn't wire
Comet — that only exists today in the Phase 7 observability path shown below. Since the point of
this manual is to see what's actually happening rather than trust a black box, everything from here
on is the explicit Python underneath, not the CLI.

It requires `POTSDAM_API_KEY` and `BASE_POTSDAM_LLM_URL` (a repo-root `.env` is the usual place)
since the accompanying config points at the real University provider, plus `COMET_API_KEY` for the
Comet section below.

### 8.1 Resolve and inspect the config — nothing hidden

```python
import json
import os

from mas_cc.config import load_run_config

# Run from the repository root - the config path below is relative to it.
config = load_run_config(
    "configs/runs/naming_convention_tutorial_university_v3.yaml", environment=os.environ
)

# Every value the run will actually use, with schema_version defaults filled
# in. Nothing past this point is implicit.
print(json.dumps(config.to_dict(), indent=2))
```

`config` is a `RunConfig` (`src/mas_cc/config/models.py`) — a plain, inspectable Python object, not
a dict you have to trust blindly. `config.game.population_size`, `config.llm_provider.model`,
`config.budget.max_cost_per_run`, and so on are ordinary attribute reads; the `.to_dict()` print
above is exactly what gets written to `resolved_config.yaml` in a real run, secret-free.

### 8.2 Construct the game and the provider explicitly

```python
from mas_cc.games import create_game
from mas_cc.llm_runtime.providers import create_llm_provider

game = create_game(config.game)  # NamingConventionGame, resolved from config.game.type
provider = create_llm_provider(config.llm_provider, environment=os.environ)  # the University adapter
```

`create_game`/`create_llm_provider` are just registry lookups keyed by `config.game.type` /
`config.llm_provider.type` — nothing about *which* game or provider is guessed; both come straight
from the config values printed above.

### 8.3 Wire in `RunRecorder`, with Comet on

This is the same observability object Phase 7 uses, constructed explicitly instead of hidden
inside a CLI command:

```python
from mas_cc.observability import DetailedAuditPolicy, RunRecorder
from mas_cc.storage import results_run_dir

destination = results_run_dir(
    "results", game=config.game.type, experiment=config.experiment.name,
    run_id=f"{config.experiment.name}-{config.execution.seed}",
)
policy = DetailedAuditPolicy.from_mapping(config.logging.options.get("detailed_prompt_audit"))
recorder = RunRecorder(
    destination, run_id=f"{config.experiment.name}-{config.execution.seed}",
    resolved_config=config.to_dict(), policy=policy,
    comet_enabled=config.logging.comet,  # True in the accompanying config
    project_name=str(config.logging.options.get("comet_project", "mas-cc")),
)
```

`comet_enabled=config.logging.comet` reads straight from the config rather than hardcoding
`True` here, so flipping `logging.overrides.comet` back to `false` in the YAML is enough to turn
it off again — nothing in this code needs to change. If `COMET_API_KEY` isn't set, `RunRecorder`
does not fail; it records `comet.status == "unavailable"` and everything else still works, exactly
as designed in Phase 7 (see the `mas_cc-loop-runtime`/observability session notes).

### 8.4 The adapter `run_naming_convention_game` actually calls

`run_naming_convention_game`'s `observer` parameter is called with exactly three method names —
`event`, `record_attempt`, `record_interaction` — and expects a `budget_status` mapping that the
game runtime itself doesn't know how to produce (correctly: it has no idea what a budget is).
Production code (`cli/phase7.py`) supplies that from a `RuntimeBudgetGuard`; this example uses an
empty one to keep the Comet wiring visible on its own without also pulling in the full pricing/
budget-guard machinery, which is a separate, larger topic:

```python
class ObservedRecorder:
    """Adds a (here, empty) budget_status at the boundary run_naming_convention_game expects.
    A real run should source this from a RuntimeBudgetGuard - see cli/phase7.py."""

    def event(self, event_type, **payload):
        recorder.event(event_type, **payload)

    def record_attempt(self, **payload):
        recorder.record_attempt(**payload, budget_status={})

    def record_interaction(self, **payload):
        recorder.record_interaction(**payload, budget_status={})
```

### 8.5 Run it

```python
from mas_cc.games.naming_convention import build_metrics, run_naming_convention_game_sync, to_round_view

try:
    result = run_naming_convention_game_sync(game, config, provider, observer=ObservedRecorder())
finally:
    provider.close()

summary = recorder.finalize(status="completed", budget_status={})
print(len(result.interactions), "interactions,", result.termination_reason)
print("Comet:", summary["comet"]["status"], "-", destination / "comet_summary.json")

# Fold the shared metrics over every resulting state.
views = tuple(to_round_view(interaction.transition.next_state) for interaction in result.interactions)
metrics = build_metrics()
shares = next(m for m in metrics if m.name == "population_action_share_per_option")
print("final share per option:", shares.compute_round(views[-1]))  # e.g. {"Q": 1.0, "M": 0.0}
```

**What I actually validated, and what I didn't fabricate:** I ran this exact wiring — config
resolution, `RunRecorder` construction, the `ObservedRecorder` adapter, `run_naming_convention_game_sync`,
`recorder.finalize` — with a mock provider standing in for University, so I could confirm none of
it crashes and that it produces 6 completed interactions,
`termination_reason == "fixed_horizon_reached"`, and a real `population_action_share_per_option` value. That
validation run *did* go through to a real, live Comet experiment (this repository's `.env` has a
working `COMET_API_KEY`), which was not intentional — I was only trying to confirm the code path,
not create a live artifact. Running the snippets above yourself, with your own University
credentials, is how you get a real model-driven run; the Comet run it produces will be a genuine
one this time, deliberately, because you asked for it.

---

## 9. Where to go next

- **Run many episodes of this game as a priced, concurrent experiment:**
  [`running_an_experiment.md`](running_an_experiment.md) — the next layer up: preflight cost
  estimation, concurrency, resumable batches.
- **Build a genuinely new game, hands-on:** [`notebooks/tutorial_build_a_game.ipynb`](../../../notebooks/tutorial_build_a_game.ipynb) —
  constructs `SignChoiceGame` from these exact abstract classes, live, then proves
  `isinstance(game, Game)` is really `True`.
- **The real files**, for comparison at production scale: `src/mas_cc/games/naming_convention/{game,prompts,records,runtime,metrics}.py`.
- **University provider details** (models, pricing, endpoints): [`docs/howto/llm_apis/university_llm_api.md`](../llm_apis/university_llm_api.md).
- **A smaller, second real game:** `src/mas_cc/games/toy_coordination/` — the same shape, fewer
  blocks, no memory-windowing subtlety.
- **Promotion checklist** for turning a notebook-built game into a real package (register in the
  game registry, add contract tests, a `to_round_view` module, and a real `call_plan` maximum-memory
  scenario): see the closing section of the tutorial notebook.
