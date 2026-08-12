# Carrying MA-CC's orchestration into an LLM evolutionary-search project

**Date:** 2026-08-12
**Audience:** whoever builds the next repository (AlphaEvolve / LLEMA-style evolutionary search with LLMs).
**Source system:** `MA-CC`, specifically what happens when you run

```
conda run -n MA-CC --no-capture-output python -m mas_cc.cli.main experiment run \
  --config configs/runs/hidden_bench/hidden_bench_imitation_reasoning_control_task_grid.yaml \
  --output-dir results
```

This is not a port plan. It is a description of **which mechanisms in MA-CC are actually load-bearing**, why
each one exists (most were written after a specific failure), and what has to change when the unit of work
stops being an independent episode and becomes a *generation of an evolving population*.

Papers this is meant to inform:

- Novikov et al., *AlphaEvolve: A coding agent for scientific and algorithmic discovery* (`pdfs/Extra/AlphaEvolve.pdf`)
- Abhyankar et al., *LLEMA: Evolutionary Search with LLMs for Multi-Objective Materials Discovery*, ICLR 2026
  (`pdfs/Extra/ICLR-2026-llema-...pdf`)

---

## 0. The one-paragraph summary

MA-CC runs one **master asyncio process**. It expands a config into a Cartesian grid of *cells*, flattens
every cell into a flat list of independent *episodes*, and runs that whole flat list through a single
`asyncio.Semaphore(execution.parallelism)` via one `asyncio.gather`. Inside each episode, the agents that
must speak in the same round are dispatched together with a second `asyncio.gather`. Underneath both, a
single provider client holds a third semaphore (`llm_provider.request_concurrency`) that is the only thing
the HTTP endpoint actually sees. One `RuntimeBudgetGuard` and one pricing quote are shared by everything, so
the run stops on **money**, not on a token guess. One `MasterMonitor` — the only process allowed to talk to
Comet — publishes progress on a daemon-thread heartbeat. Everything of scientific value is on local disk
first; Comet is a *view*, never the store.

For the evolutionary project, the flat-list-of-independent-episodes assumption is the one thing that breaks,
because generation *n+1* depends on generation *n*. Everything else — the semaphore layering, the budget
guard, the seed derivation, the prompt contract, the recorder, the monitor, the resume logic — transfers
almost unchanged. Section 4 is about that one break.

---

## 1. Vocabulary map

| MA-CC | Evolutionary project | Notes |
| --- | --- | --- |
| `RunConfig` (one YAML) | same | single source of truth, hashed for resume |
| **grid cell** (one point in the Cartesian sweep) | one **task × hyperparameter** configuration, e.g. one of LLEMA's 14 discovery tasks × mutation-operator arm | already exactly right |
| **episode** (one independent trajectory, N repetitions per cell) | one **evolutionary run** (a seed replicate of the whole search) | still independent → still embarrassingly parallel |
| **round / turn** inside an episode (`horizon: 10`) | one **generation** `n = 1..N` | *now sequentially dependent — see §4* |
| **agent decision** inside a round (`asyncio.gather` over agents) | one **candidate proposal** inside a generation (LLEMA's batch of `b`; AlphaEvolve's sampled parents) | direct reuse, this is the hot path |
| `Control` (external actuator that can replace an agent's turn) | evolution operator / chemistry rule injector / meta-prompt | same seam: `forced_action` in `loop_runtime` |
| `Game` ABC | `SearchProblem` ABC | see §7 |
| response contract (`hidden_bench_json_vote`) | candidate contract (SEARCH/REPLACE diff block, or CIF JSON) | see §7.2 |
| `scientific_events.parquet` | candidate / lineage table | see §9 |

The single most important line of that table: **a generation is a round, not an episode.** People's first
instinct is to map "generation" onto "episode" because both are the outer loop they think about, and that
produces a scheduler that cannot run more than one generation at a time.

---

## 2. The three tiers of parallelism, as actually implemented

### Tier 1 — the flat task list (`experiments/orchestrator.py:1808-1836`)

`run_experiment_grid` does **not** loop over cells and run each cell in turn. It builds every episode of
every cell into one flat list first:

```python
all_tasks: list[_EpisodeTask] = []
for cell in cells:
    cell_seed = grid_seed.derive(f"grid-cell:{cell.index}")
    for index in range(cell.config.execution.repetitions):
        episode_seed = int(cell_seed.derive(f"episode:{index}"))
        all_tasks.append(_EpisodeTask(
            episode_id=f"{cell.cell_id}-{index:04d}",
            seed=episode_seed,
            config=replace(cell.config, execution=replace(cell.config.execution, seed=episode_seed)),
            episode_dir=episodes_dir / episode_id, cell_id=cell.cell_id, cell_dir=cell_dir,
        ))
```

Then one gather over the lot ([orchestrator.py:1225-1228](../../src/mas_cc/experiments/orchestrator.py#L1225-L1228)):

```python
async def _run_task_batch(tasks, **kwargs):
    return tuple(await asyncio.gather(*(_run_episode_task(task, **kwargs) for task in tasks)))
```

**Why flat matters.** With 6 cells × 4 repetitions and `parallelism: 5`, a per-cell loop would idle 5 slots
whenever a cell had fewer than 5 episodes left, and the slowest episode of each cell would stall the next
cell entirely. Flattening means the semaphore is always saturated until the *whole run* is nearly done. In
the evolutionary project the same argument applies to islands and to seed replicates: **flatten every
independent track into one list, gate it with one semaphore.**

Each task carries its own *fully resolved config*, not a pointer to a shared one plus an override. That is
what makes an episode reproducible in isolation and what makes the resume hash meaningful.

### Tier 2 — inside one unit of work (`games/hidden_bench/imitation/runtime.py:130-202`)

Everything that can happen simultaneously in the same round is gathered:

```python
message_decisions = tuple(await asyncio.gather(*(
    _execute_decision(game, request, state, config, provider, counter, root, observer)
    for request in requests
)))
```

This is the seam the evolutionary project will use hardest: **one generation = one gather over the batch of
candidate proposals.** LLEMA's `\mathcal{M}^b_{j=1} \leftarrow \pi_\theta(\mathbf{p}_n)` (Algorithm 1, line 6)
is literally this line with `requests` = the `b` prompts built from the selected island's memory.

### Tier 3 — the provider client (`llm_runtime/providers/adapters/_openai_compatible.py:92, 181-239`)

```python
self._semaphore = asyncio.Semaphore(config.request_concurrency)
...
async with self._semaphore:
    for retry in range(self._max_retries + 1):
        ...
        if self._is_retryable(response.status_code) and retry < self._max_retries:
            await asyncio.sleep(self._retry_delay(response, retry))
```

- retryable = `429`, any `5xx`, or a transport error with no status
- backoff = `Retry-After` header if present, else `0.5 * 2**retry`
- the blocking `requests` call is pushed off the event loop with `asyncio.to_thread`

**The two semaphores are not redundant.** Tier 1 bounds *how many trajectories are in flight* (memory,
interleaved logging, blast radius of a failure). Tier 3 bounds *how many HTTP requests the endpoint sees*
(the shared-proxy rate limit). In the reference config they are 5 and 10: five episodes each capable of
issuing 2 concurrent agent calls. Sizing rule: **Tier 3 ≈ Tier 1 × typical fan-out per round.** Get this
wrong in the direction of Tier 3 being too small and you silently serialize; too large and you get 429
storms that the backoff turns into wall-clock you paid for anyway.

There is a fourth pool that MA-CC only needs in a small way but the **evolutionary project will need
seriously**: see §5, the evaluator pool.

### Sizing note carried from a real failure

The config comments at
[hidden_bench_imitation_reasoning_control_task_grid.yaml:83-103](../../configs/runs/hidden_bench/hidden_bench_imitation_reasoning_control_task_grid.yaml#L83-L103)
record two production incidents worth inheriting as rules:

1. `timeout_seconds` must cover generating `max_output_tokens` **in the tail under full concurrency**, not
   the median reply. A 30B model behind a shared proxy at concurrency 10 needed 180s where 60s looked
   generous. The run that died had its last good call 57 seconds before the timeout fired.
2. Raising `max_output_tokens` without raising `timeout_seconds` converts truncation failures into timeout
   failures. Truncation at least fails loudly against the response contract; a mid-sentence cut in a *free
   text* field corrupts your measurement silently. For evolution this is worse, not better: a truncated
   diff block or a truncated CIF is a candidate that looks parseable and isn't.

---

## 3. What the master process actually does, end to end

Ordered, because the order encodes safety properties:

1. **Expand the grid** (`config/grid.py`). Cartesian product of dotted-path overrides against one base
   config. `GridSpec.__post_init__` binds the first value of every axis immediately, so a typo in an axis
   path fails once, before anything runs, rather than once per generated cell.
2. **Refuse illegal axes.** `_FORBIDDEN_EXACT = ("llm_provider.type", "llm_provider.model", "game.type",
   "budget", "pricing")`. Every cell shares one provider client, one pricing quote, one budget guard —
   sweeping those would silently invalidate all three. **This is the one rule the evolutionary project must
   deliberately break; see §6 on model ensembles.**
3. **Preflight**: price the whole grid statically, compare against `budget.max_cost_per_run`, refuse to
   launch if `launch_status != "permitted"`. No provider calls are sent.
4. **Re-validate live pricing immediately before launch** and abort if the terms moved between preflight and
   launch. Cheap, and it closes the window where you priced one thing and ran another.
5. **Wipe/create the run directory**, write `resolved_base_config.yaml` and each cell's
   `resolved_config.yaml` + `overrides.json` *before* any work starts. If the run dies in minute one, the
   directory still says exactly what was going to happen.
6. **Partition resume** (`_partition_resume_tasks`, orchestrator.py:781). Already-complete episodes are
   removed from the schedule and replayed into progress/monitor state as `skipped_resumed`.
7. **Construct the provider once**, wrap it in `BudgetGuardedProvider`, start the live-spend watcher.
8. **Run the flat batch** under the semaphore, with the spend watcher running alongside as a side channel.
9. **Close each cell exactly once**, on the episode that completes it (§8).
10. **`finally`:** close the provider, close the progress bar, seal artifacts, render the grid figure
    locally, aggregate, close Comet. Note the ordering — the post-run analysis publishes onto the master
    Comet experiment, so the master is closed **after** the analysis, not before.

---

## 4. The break: generations are not independent

MA-CC's outer loop is `asyncio.gather` over a list known in full before the run starts. Evolution's outer
loop is not: generation *n+1*'s prompts are built from generation *n*'s results. There are two honest
schedulers, and I recommend building (A) first and (B) behind the same interfaces.

### (A) Synchronous generational — LLEMA Algorithm 1

A barrier at every generation. The parallel fan-out is *within* a generation.

```python
async def run_search(problem, config, provider, guard, observer, semaphore):
    population = problem.initial_population(config, seed)
    for generation in range(1, config.search.generations + 1):
        island   = problem.select_island(population, rng)          # Boltzmann sampling, LLEMA §2.4
        prompts  = problem.build_prompts(island, generation, config)  # top-k successes + failures + rules
        # --- the one hot line: the whole batch in flight at once ---
        candidates = await asyncio.gather(*(
            propose_candidate(problem, p, provider, observer) for p in prompts
        ))
        # --- evaluation: a DIFFERENT pool, see §5 ---
        scored = await asyncio.gather(*(
            evaluate(problem, c, evaluator_semaphore) for c in candidates
        ))
        population = problem.update(population, scored)
        observer.record_generation(generation, population, scored)   # ← checkpoint boundary
    return population
```

**What you get for free by keeping the barrier:** `generation` is a clean checkpoint boundary (resume from
generation *k*), a clean logging step (Comet's step axis), and a clean aggregation unit. The cost is a
straggler tax — the generation is as slow as its slowest evaluation.

Note that this maps *exactly* onto MA-CC's existing round loop: `while state.turn < rules.horizon` with a
gather inside. You are reusing the episode runtime shape, not inventing one.

### (B) Asynchronous steady-state — AlphaEvolve §2.6

AlphaEvolve has no generation barrier. It runs a controller plus LLM samplers plus evaluation nodes as one
asyncio pipeline "optimized for throughput rather than the speed of any one computation." The worker loop is
Figure 2:

```python
async def worker(worker_id, database, prompt_sampler, provider, evaluator, stop):
    while not stop.is_set():
        async with semaphore:
            parent, inspirations = await database.sample()      # lock-protected
            prompt   = prompt_sampler.build(parent, inspirations)
            diff     = await provider.complete(prompt)
            child    = apply_diff(parent, diff)
        results = await evaluator.execute(child)                # own pool, §5
        await database.add(child, results)                      # lock-protected
```

Then `asyncio.gather(*(worker(i, ...) for i in range(n_workers)))` with a stop condition on budget,
wall-clock, or total candidates.

**What you lose:** "generation" stops being a real object. You need a synthetic step counter (candidates
evaluated) for logging and checkpointing, and the program database becomes shared mutable state needing an
`asyncio.Lock` — MA-CC has no equivalent, because episodes share nothing.

**Recommendation.** Build (A). Its barrier is what makes checkpoint/resume/aggregate/Comet-step all trivially
correct, and LLEMA's results were produced with it. Define `SearchProblem` so the population update and the
prompt construction are already pure functions of a population object; then (B) is a scheduler swap, not a
rewrite. Do not build (B) first "because AlphaEvolve does" — AlphaEvolve is optimizing throughput at a scale
(thousands of samples, 100 compute-hours per evaluation) that you will not be at on day one.

### Where the outer parallelism goes

In both schedulers, the *independent* tracks that fill Tier 1 are:

- **islands** (LLEMA's `m = 5`, AlphaEvolve's island model) — independent between synchronization points
- **seed replicates** of the whole search (the direct analogue of `execution.repetitions`)
- **grid cells** (task × operator-set × model-role assignment)

Flatten `cells × replicates × islands` into one list exactly as `run_experiment_grid` does, gate with one
semaphore, and let generations be the sequential inner loop. That gives you a saturated provider even though
each individual search is sequential — which is the whole trick, and the reason the generation-as-episode
mistake is expensive.

---

## 5. The evaluator pool — the genuinely new component

MA-CC's only expensive resource is the LLM. In the evolutionary project, evaluation is a first-class,
*separately bounded* resource:

- AlphaEvolve: executing `evaluate` on a candidate program; "on the order of 100 compute-hours"; cascaded
  through stages of increasing difficulty so bad candidates die cheap (§2.4).
- LLEMA: a database lookup, then CGCNN/ALIGNN surrogate inference for out-of-distribution candidates (§2.3);
  hard-constraint violators are scored low and pruned *before* the expensive path.

Design consequences:

1. **A second, independently sized semaphore.** `evaluation.concurrency` is not `execution.parallelism` and
   is not `llm_provider.request_concurrency`. CPU/GPU-bound work belongs in `asyncio.to_thread` or a process
   pool, never inline on the event loop. MA-CC already uses this discipline for its blocking cell
   aggregation ([orchestrator.py:1143-1150](../../src/mas_cc/experiments/orchestrator.py#L1143-L1150)) —
   "aggregating a cell reads every episode's metric files, and blocking the loop for that would stall the
   episodes still running in the other cells." Same reasoning, much higher stakes.
2. **Implement the cascade from day one.** It is a cheap filter that changes the cost curve: syntactic/
   parse validity → hard-constraint check → cheap surrogate → full evaluation. Each stage is a separate
   observable count, and those counts are the most useful diagnostic you will have (LLEMA's *hit-rate* is
   precisely stage-2 pass rate).
3. **A separate budget dimension.** MA-CC's guard tracks money, requests, and tokens. Add compute-seconds.
   The `RuntimeBudgetGuard` reserve/reconcile pattern (§6) generalizes directly.
4. **Evaluation failures are data, not errors.** A candidate that fails to compile is a *result* to be
   written into the failure memory (LLEMA's $\mathbb{M}^-$), not an exception that kills a generation. This
   is the opposite of MA-CC's stance, where a failed decision is an episode failure — call it out explicitly
   in the `SearchProblem` contract so nobody wires it to the exception path.

---

## 6. The budget guard, and the one rule you must break

### Reserve / reconcile (`llm_runtime/providers/budget.py:538-581`)

Every call goes through `BudgetGuardedProvider`:

```python
input_tokens = ceil(estimate_input(request) * multiplier)
cost         = pricing.cost(input_tokens, request.max_output_tokens)   # conservative
reservation  = guard.reserve(conservative_cost=cost, ...)              # BEFORE dispatch
response     = await provider.complete(request)
guard.reconcile(reservation, actual_cost=..., input_tokens=..., output_tokens=...)  # AFTER
```

Reserving the *conservative* cost before dispatch is what makes the guard correct under concurrency: with N
calls in flight, optimistic accounting overshoots the ceiling by N calls' worth. Reconciling afterwards
gives the money back. Transport failures deliberately **fail closed** (keep the reservation) because the
provider may already have billed.

### Budget stop ≠ episode failure (`orchestrator.py:1163-1179`)

This is the single most valuable operational lesson in the repo, and the code carries the incident report:

```python
# A budget stop aborts regardless of `fail_fast`. Once the guard is exhausted or
# stopped its counters only move one way, so every queued episode would dispatch,
# be refused on its first call, and be recorded as a failure. That is how one
# exhausted token budget turned into 4,235 "failed" episodes (results/DIAGNOSIS.md)
# and buried the fact that the run had simply run out of money.
if budget_abort.is_set():
    outcome = EpisodeOutcome(..., "skipped_aborted", error_type="BudgetStop", ...)
```

Two distinct signals, two distinct terminal statuses:

- `abort` — set only when `fail_fast` and something raised. One bad episode.
- `budget_abort` — set when a `ProviderError.code in BUDGET_STOP_CODES`. Structural: no queued work can ever
  succeed, so remaining units are marked **`skipped_aborted`**, not `failed`.

Carry the four-state outcome vocabulary verbatim: `completed | failed | skipped_resumed | skipped_aborted`.
An aggregate that cannot distinguish "we ran out of money" from "the model produced garbage" is an aggregate
that will mislead you at 2am.

### Live spend polling

`_LiveSpendWatcher` polls the provider's own spend endpoint every `budget.live_spend_poll_seconds` (120 in
the reference config) and calls `guard.request_stop` when the real balance crosses the ceiling. It runs as a
side task and is cancelled the instant the episodes finish; it "is a side channel, never a participant" and
cannot fail the run. Token estimates drift; the invoice does not. Related memory: *mas_cc budget philosophy*.

### The rule to break: model ensembles

`GridAxis.__post_init__` forbids sweeping `llm_provider.model`. AlphaEvolve **requires** an ensemble —
Gemini 2.0 Flash for throughput, Gemini 2.0 Pro for occasional breakthroughs (§2.3). So the new project must
support what MA-CC deliberately refused:

```yaml
llm_provider:
  roles:
    fast:    { type: ..., model: ..., request_concurrency: 16, weight: 0.8 }
    strong:  { type: ..., model: ..., request_concurrency: 4,  weight: 0.2 }
```

Requirements this imposes, all of which MA-CC's single-model design let it skip:

- one client **per role**, each with its own Tier-3 semaphore
- one pricing quote per role; preflight sums over roles weighted by the sampling ratio
- **one shared `RuntimeBudgetGuard` across all roles** — the ceiling is on the run, not per model
- the role that produced each candidate is recorded on the candidate (otherwise you cannot ever answer "was
  the expensive model worth it", which is the ablation AlphaEvolve reports in §4)

Do this deliberately and up front. Retrofitting a second model into a system that assumed one pricing quote
is exactly the kind of change that quietly invalidates every cost number you already published.

---

## 7. Prompts

### 7.1 The prompt is an object with a hash, not a format string

`FullPrompt` (`llm_runtime/prompts/full_prompt.py`) is an ABC: an ordered tuple of named `PromptBlock`s plus
a `ResponseContract`. It gives you

- `definition_hash` — the *structure*: family, version, block definitions, contract. Identical across every
  call that used the same template.
- `instance_hash` — the structure **plus the bound values**. Unique per actual prompt sent.
- `bind(**values)` — validated; an unknown block name raises rather than being silently ignored.
- `compile()` — validates, renders, merges consecutive same-role blocks, counts tokens.

`definition_hash` is what preflight prices, what the audit record carries, and what the resume check
compares (`prompt_definition_hashes_hash`). Change a prompt template and the resume logic *refuses* to reuse
old episodes instead of silently mixing two prompt versions in one dataset. This is worth more in the
evolutionary project than here, because AlphaEvolve has **meta-prompt evolution** (§2.2): the prompt itself
is co-evolved in a separate database. Without content-addressed prompts you cannot reconstruct which prompt
produced which candidate.

The config also cross-checks the version in two places and refuses to start if they disagree
([runtime.py:100-108](../../src/mas_cc/games/hidden_bench/imitation/runtime.py#L100-L108)) — "a silent
disagreement would price v1 and run v2."

**Block structure for evolution** maps cleanly onto both papers:

| Block | AlphaEvolve (§2.2, Fig. 3b) | LLEMA (§2.2) |
| --- | --- | --- |
| `task_spec` | problem definition, background | task description + property constraints $\mathcal{C}$ |
| `operators` | system instructions on how to propose changes | chemistry-informed design rules $\mathcal{R}$ |
| `positive_exemplars` | prior programs with scores | success pool $\mathbb{M}^+$ top-k |
| `negative_exemplars` | — | failure pool $\mathbb{M}^-$ top-k |
| `current` | "the current program we are trying to improve" | current candidate |
| `output_contract` | SEARCH/REPLACE diff format | CIF JSON schema |

Note AlphaEvolve's *stochastic formatting*: template placeholders with human-provided alternatives sampled
from a distribution, for diversity. MA-CC has a version of this in the controller's four-paraphrase bank
whose **ID is logged per event** (see the `template_version` comments in the reference config). Log the
sampled variant ID; otherwise a diversity mechanism becomes an unmeasurable confound.

### 7.2 The ask/validate/retry loop (`runtime/loop_runtime.py`)

One provider-agnostic, problem-agnostic loop that every LLM-backed decision goes through:

```python
for attempt_index in range(request.retry_bound + 1):
    response = await provider.complete(completion_request)     # transport errors: re-raise, NOT retried
    prompt.response_contract.validate(response.content).raise_for_errors(...)
    action = game.parse_action(request, response.content)
    game.validate_action(state, request, action, game_config).raise_for_errors(...)
    if attempt.valid: return ValidatedDecision(action, attempts)
raise DecisionLoopExhausted(...)
```

Four properties to carry:

1. **Validation retries and transport retries are different layers.** Transport (429/5xx) is retried inside
   the provider adapter with backoff. Content-validation failure is retried here. A transport failure is
   re-raised immediately — retrying it here would multiply the adapter's retries.
2. **Every attempt is recorded**, including the failures, via `on_attempt`. Malformed-output rate per prompt
   version is a headline metric, not a debug detail. For evolution it is *the* prompt-quality signal.
3. **Validity is defined by the problem, never by the loop.** `parse_action` + `validate_action` live on the
   `Game`/`SearchProblem`. For evolution: does the diff apply cleanly? Is the CIF parseable? Are hard
   constraints satisfied? Keep the loop ignorant of all of it.
4. **`forced_action` short-circuits without calling the provider.** This is the single seam through which an
   external `Control` overrides a decision — and it is where a *non-LLM* mutation operator (crossover, a
   deterministic chemistry rule) plugs in at zero provider cost. Design it in from the start; it is what
   makes "LLM vs. classical operator" an ablation rather than a fork of the codebase.

### 7.3 Prompt examples in the results (`_CellPromptSampler`, orchestrator.py:679-749)

Requested in the config as:

```yaml
logging:
  options:
    prompt_examples:
      count: 2
      scope: cell   # one deterministic early/late sample per cell, not per round per episode
```

Mechanism: during the run, `_RoundTickingObserver.record_attempt` captures the rendered Markdown of the
**first attempt of a round that validated**, appending to a gzipped candidate file under
`.resume/<episode_id>/prompt_candidates.json.gz` (thread-locked, atomic temp-file replace). With `count: 2`
it keeps the first and continually replaces the last, so you end up with an early and a late prompt. At cell
completion, `render()` emits `cells/cell-000N/prompt_examples.md` from the first completed episode, in
sorted order — deterministic regardless of which episode finished first.

The output is the *exact wire messages*, fenced, with the definition hash, message count, estimated tokens,
request metadata, and the model's response — see
`results/hidden_bench_imitation/.../cells/cell-0000/prompt_examples.md`.

**Why it is shaped this way, and why to copy it:** the naive version (dump every prompt) produced tens of
thousands of files and got turned off, after which nobody could see what the model was actually asked.
Bounded, deterministic, one file per cell, on by default, is the version people keep on. For evolution, take
**one prompt per generation-decile plus the prompt that produced the current best candidate** — the second
is the one you will actually paste into a paper.

---

## 8. Logging, recording, and the results tree

### Layers

1. **Console** — `ExperimentProgress`, two bars (episodes, rounds), plus a banner printed *before* any work
   that names the experiment, model, cell count, concurrency, preflight cost, and — unconditionally, up
   front — the results directory. "This is the one line that answers 'where do I look for the results'
   without waiting for the run to finish or grepping logs."
2. **`RunRecorder`** (`observability/recorder.py`) — per unit of work: events, attempts, interactions,
   trajectories, metrics, budget status at each step; `finalize(status=...)` on both success and failure. It
   is constructed with `comet_enabled=False` **always**: "Per-episode Comet experiments are not wired here:
   one experiment would otherwise fan out into N remote experiments."
3. **Audit** — every attempt with its prompt hashes, under `DetailedAuditPolicy`.
4. **Scientific table** — `scientific_events.parquet` per cell, with an identity (`run_id`, `cell_id`,
   `episode_id`, config hash, price hash, schema version) that is *validated on resume*.

### The tree

`results/<game>/<experiment>/<run_id>/` (`storage/results.py`), with `logs/ audit/ checkpoints/ metrics/
metrics/plots/ data/`, and for a grid:

```
<run_id>/
├── resolved_base_config.yaml      # written before any work
├── grid_summary.csv / manifest.json
├── grid_progress.png              # the same figure the dashboard gets, always written locally
├── comet_summary.json             # what was actually published, on disk
└── cells/cell-0000/
    ├── resolved_config.yaml       # this cell's full config
    ├── overrides.json             # which axis values made this cell
    ├── prompt_examples.md         # §7.3
    ├── scientific_events.parquet  # the analysis input
    ├── aggregate.json / cell_summary.json / cell_complete.json
    └── metrics/plots/*.png
```

`artifact_profile: results_only` (the reference config) compacts per-episode directories into the single
per-cell parquet plus summaries, which is what makes a 28-cell × 4-episode grid navigable at all.

For evolution the tree is nearly unchanged; the parquet's rows become candidates:

```
candidate_id, parent_id, island, generation, model_role, prompt_definition_hash,
prompt_instance_hash, proposal_raw, parse_status, eval_stage_reached,
scores{...}, constraints_satisfied, is_pareto, wall_seconds, cost
```

`parent_id` is the addition that MA-CC has no analogue for and that you cannot reconstruct afterwards. Write
it from the first commit — lineage is the object of study.

### Cell completion (orchestrator.py:478-496, 1131-1155)

```python
class _CellCompletion:
    """Fires once per cell, on the episode that brings it to its expected count.

    Counting terminal episodes rather than watching the filesystem keeps the trigger
    exact under `execution.parallelism`: episodes finish out of order, and "the
    directory looks full" is a race while "the last of N reported in" is not.
    """
```

The close-out then runs `await asyncio.to_thread(aggregator.aggregate, cell_id)` — **off the event loop, and
outside the semaphore**, so aggregating a finished cell neither blocks the loop nor holds a concurrency slot
another cell's work could use. Both details are load-bearing. Reuse this for "an island reached its final
generation" and for "a search replicate finished."

---

## 9. Comet

### One writer, always

`MasterMonitor` is "the only process that talks to Comet, for one experiment or grid run." Two switches, as
labelled in the reference config:

```yaml
logging:
  comet: true                    # MASTER LOGGING, SWITCH 1 of 2: ON / OFF
observability:
  comet:
    writer: master_only          # SWITCH 2 of 2: SHAPE (only read when comet: true)
    heartbeat_seconds: 20
    progress_metrics: [episodes_done]
    grid_image_every_n_episodes: 5
    cell_reporting: master       # master | experiments | disabled
    metric_plots: true
```

Properties to carry verbatim:

- **Disabled instances are inert but still safe to call**, so the orchestrator never branches on whether
  monitoring is on. This alone removes a whole class of "works with Comet off, crashes with it on" bugs.
- **Every remote call is wrapped** in `_guarded`: "a Comet outage must not fail a run whose real output is
  already on disk."
- **The heartbeat is a daemon `threading.Thread`, not an asyncio task** — deliberately: "the heartbeat must
  keep ticking while the event loop is blocked, since 'the loop is wedged' is precisely one of the failures
  it exists to reveal." Publishing unconditionally on a timer means a flatlined series distinguishes *dead*
  from *slow*; a metric that only moves on completion cannot.
- **State behind a `threading.RLock`**, since episode completions arrive from arbitrary tasks.
- **ETA is reported as `-1`, never omitted**, before the first completion: "a missing series and an unknown
  ETA look identical on a dashboard, and only one of them is a reason to worry."
- **`describe()` reports the connection, not the config.** It prints whether Comet actually connected, and
  why not if it didn't. "A run that silently uploaded nothing because the key was missing looks identical to
  one that was never meant to upload, which is the confusion this line exists to end."
- **The grid figure is written to local disk too** (`grid_progress.png`), always: "so the picture is
  checkable, and a cluster job with no outbound network is still watchable by tailing this file."
- **`comet_summary.json` is written into the run directory** — the record of what was published belongs on
  disk with the run, not only on the dashboard.

### `cell_reporting: master` vs `experiments`

- `master` — everything on one experiment; cell series get prefixed (`cell-0000_m_ctrl`), so you get one
  link to open but you cannot overlay two cells on one chart.
- `experiments` — one child experiment per cell with bare metric names, which is what lets Comet overlay
  arm A against arm B.

For evolution, the equivalent trade-off is: one experiment for the whole sweep (a `best_score` curve per
island, prefixed) versus one per configuration (overlay operator-set A against B). Expect to want
`experiments` more often than MA-CC does, because the headline evolutionary plot *is* a comparison of
best-so-far curves.

### What to publish, for an evolutionary run

The metric shape changes more than the mechanism does. Step axis = generation (scheduler A) or candidates
evaluated (scheduler B). Per step:

- `best_score`, `population_mean_score`, `population_score_std`
- `hit_rate` (fraction satisfying all hard constraints — LLEMA's headline)
- `parse_failure_rate`, `eval_failure_rate` (prompt/contract health)
- `pareto_front_size`, hypervolume — multi-objective, both papers
- `diversity` (distance in whatever representation space you have; the exploration/exploitation balance is
  explicitly the hard part of the program database in AlphaEvolve §2.5)
- `unique_candidates / total_candidates` (memorization check — LLEMA §2.4 worries about this)
- cost-to-date, candidates-per-minute, ETA, and the same budget scalars MA-CC publishes

Plus, as figures: the grid-progress analogue (islands × generations, filled in as it runs) and a best-so-far
curve per island. And **the current best candidate as a text asset every K generations** — the cheapest,
highest-value thing on the dashboard, and it is what you will actually watch.

### The incident to inherit

Memory *Incident: accidental live Comet run*: a mock-provider validation still triggered a real Comet
upload, because the integration was gated on config rather than independently of the provider mock. **Make
side-effecting integrations disable independently of the LLM provider mock.** A dry run must be dry in every
direction.

---

## 10. Determinism and resume

### Seeds

`Seed` (`core/random.py`) is an immutable int with `derive(namespace) -> Seed` implemented as
`sha256("mas-cc-seed-v1\0{value}\0{namespace}")`. Derivation is hierarchical and order-independent:

```
root ── "grid-cell:3" ── "episode:2"        # orchestrator
     └─ "imitation-focal-and-peer-selection"  # named streams inside a run
     └─ "imitation-classical-transition"
     └─ "imitation-controller-sensor"
```

Separate named streams per stochastic mechanism means adding a new source of randomness does not shift every
other draw. In the evolutionary project you will want at minimum: `island-selection`, `parent-sampling`,
`prompt-stochastic-formatting`, `operator-choice`, `model-role-choice`, `tie-breaking`. Name them the day
you add them.

### Resume

Per-unit manifests carry `resolved_config_hash`, `prompt_definition_hashes_hash`, `pricing_snapshot_hash`,
`scientific_schema_version`. On resume, `_partition_resume_tasks` validates each and **raises** on a
mismatch rather than silently re-running or silently reusing:

```python
raise ValueError(f"incompatible episode checkpoint {task.episode_id}: {field} does not match")
```

Resumed units are replayed into progress and monitor state so the dashboard and the bars are honest about
totals.

**For evolution this is harder and more important.** An episode is stateless; a search is not. Resume means
reloading the *entire program database* at generation *k* — the population, both memory pools, the
per-island state, the RNG streams. Concretely:

- checkpoint the whole population at each generation barrier (scheduler A makes this trivial; this is the
  strongest argument for building A first)
- include the population hash in the resume identity, not just the config hash
- treat "config changed mid-search" as a hard error, exactly as MA-CC does

---

## 11. Configuration shape

Keep the single-YAML, fully-resolved-and-hashed discipline, and keep the reference config's *commenting
style*: the MA-CC config is heavily commented with **why each number is what it is, and which incident set
it**. That file is the most valuable operational document in this repository. Its header block explains the
whole design of the run before a single key appears; copy that habit.

Sketch:

```yaml
llm_provider:
  roles:                          # §6 — the ensemble, the one deliberate break from MA-CC
    fast:   { type: ..., model: ..., request_concurrency: 16, weight: 0.8,
              timeout_seconds: 180, max_retries: 2, max_output_tokens: 4096 }
    strong: { type: ..., model: ..., request_concurrency: 4,  weight: 0.2, ... }

prompt:
  schema_version: 1
  prompt_family: evolve_program_diff
  prompt_version: 3
  response_contract: { type: search_replace_diff }

problem:
  type: program_evolution          # or materials_discovery
  options:
    task_id: matmul_4x4
    objectives: [speed, correctness]
    initial_program: seeds/matmul.py

search:
  generations: 40                  # ← the horizon analogue
  population_size: 64
  islands: 5
  batch_size: 8                    # candidates proposed per generation ← the inner gather
  parents_per_prompt: 3
  island_selection: boltzmann
  temperature: 1.0

evaluation:
  concurrency: 8                   # §5 — NOT execution.parallelism
  cascade: [parse, constraints, surrogate, full]
  timeout_seconds: 600

execution:
  seed: 20260812
  repetitions: 3                   # independent search replicates
  parallelism: 6                   # concurrent searches (cells × replicates × islands, flattened)
  fail_fast: false

pricing:  { mode: live, require_fresh_at_launch: true, fallback_policy: deny }
budget:   { max_cost_per_run: 300.0, live_spend_poll_seconds: 120,
            allow_unbounded_paid_requests: false }

logging:
  comet: true
  options:
    prompt_examples: { count: 3, scope: cell }

observability:
  comet: { writer: master_only, heartbeat_seconds: 20, cell_reporting: experiments,
           grid_image_every_n_generations: 5, metric_plots: true }

storage: { output_dir: results, artifact_profile: results_only,
           checkpoint_mode: generation, overwrite: false }

grid:
  problem.options.task_id: [matmul_4x4, kissing_number_11]
  search.island_selection: [boltzmann, uniform]
```

Keep `preflight` as a separate CLI verb that sends zero provider calls and prints availability, pricing,
request counts, and the budget verdict. It is the habit that prevents the expensive class of mistake.

---

## 12. Build order

1. `SearchProblem` ABC + `Candidate`/`Population` types + response contracts. Enforce it as an **ABC, not a
   Protocol**, so an incomplete implementation fails at construction rather than at generation 30 of a paid
   run (memory: *mas_cc Game ABC enforcement*, *User prefers ABC-enforced contracts*).
2. `FullPrompt` blocks + `definition_hash`/`instance_hash` + the ask/validate/retry loop, including
   `forced_action` for non-LLM operators.
3. Single-search runtime, scheduler (A): the generation loop with the intra-generation gather, plus a mock
   provider. **Verify the whole pipeline end-to-end on the mock, with Comet independently disabled** (§9).
4. Evaluator pool + cascade, with its own semaphore and its own budget dimension.
5. Recorder + results tree + candidate/lineage parquet with `parent_id`.
6. Multi-search orchestrator: flat task list, one semaphore, one gather, `_CellCompletion`-style close-out.
7. Budget guard: reserve/reconcile, `BUDGET_STOP_CODES`, `budget_abort` distinct from `fail_fast`, live
   spend watcher, four-state outcomes.
8. Preflight + pricing, multi-role aware.
9. `MasterMonitor` with the daemon heartbeat, `_guarded` calls, local figure fallback, `comet_summary.json`.
10. Prompt-example sampler.
11. Resume, including full population checkpoints.
12. Only then, if throughput actually demands it, scheduler (B).

Steps 1–5 give a working single search. Steps 6–9 are what make it a *run you can leave overnight on a
cluster with a credit card attached*, which is the part MA-CC learned the hard way and the main reason this
document exists.

---

## Appendix: file index

| Concern | File |
| --- | --- |
| Grid expansion, forbidden axes | `src/mas_cc/config/grid.py` |
| Master orchestration, both schedulers | `src/mas_cc/experiments/orchestrator.py` |
| — flat task list | `orchestrator.py:1808-1836` |
| — the gather | `orchestrator.py:1225-1228` |
| — budget-abort vs fail-fast | `orchestrator.py:1163-1185` |
| — cell completion / off-loop aggregation | `orchestrator.py:478-496, 1131-1155` |
| — prompt example sampler | `orchestrator.py:679-749` |
| — resume validation | `orchestrator.py:781-845` |
| Ask/validate/retry loop | `src/mas_cc/runtime/loop_runtime.py` |
| Provider semaphore, transport retry/backoff | `src/mas_cc/llm_runtime/providers/adapters/_openai_compatible.py` |
| Budget guard, reserve/reconcile | `src/mas_cc/llm_runtime/providers/budget.py` |
| Prompt ABC, hashes, compilation | `src/mas_cc/llm_runtime/prompts/full_prompt.py` |
| Prompt Markdown rendering | `src/mas_cc/llm_runtime/prompts/reporting.py` |
| Comet master monitor | `src/mas_cc/experiments/comet_monitor.py` |
| Recorder | `src/mas_cc/observability/recorder.py` |
| Results tree | `src/mas_cc/storage/results.py`, `storage/scientific.py` |
| Seed derivation | `src/mas_cc/core/random.py` |
| Intra-round gather (the pattern to reuse) | `src/mas_cc/games/hidden_bench/imitation/runtime.py:130-202` |
| The reference config, heavily commented | `configs/runs/hidden_bench/hidden_bench_imitation_reasoning_control_task_grid.yaml` |
