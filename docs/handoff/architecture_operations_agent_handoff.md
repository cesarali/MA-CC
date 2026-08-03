# MA-CC Architecture and Operations Handoff for AI Agents

Implementation snapshot: 2026-08-01.

This document is an operational map of the current repository. It is intended
for an AI coding agent that will help design and incrementally build a successor
package, tentatively imported as `mas_cc`, while the current `naming_game`
package remains usable.

The code is the final source of truth. In particular, consult the tests when a
behavior described here is being migrated. Provider model lists, pricing, rate
limits, and endpoints are runtime facts and must be queried again before a live
run.

## 1. Package identity and entry points

The repository currently defines:

| Concept | Current value |
|---|---|
| Distribution name used by `pip` | `llm-naming-game` |
| Python import package | `naming_game` |
| Source directory | `src/naming_game/` |
| Console command | `naming-game` |
| Console target | `naming_game.cli:main` |
| Minimum Python version | 3.11 |
| Build backend | setuptools |

The project uses a `src` layout and setuptools package discovery. After an
editable install, these are equivalent entry paths:

```bash
python -m naming_game.cli --help
naming-game --help
```

The planned name `mas-cc` is valid as a distribution name, but not as a Python
identifier. The corresponding import package should be `mas_cc`:

```python
import mas_cc
```

Both `src/naming_game/` and `src/mas_cc/` can coexist during migration.

## 2. The most important architectural distinction

This repository contains two related but non-equivalent game systems. Do not
merge their state models or metrics accidentally.

| System | Main implementation | State | Interaction rule | Main purpose |
|---|---|---|---|---|
| Binary inventory Naming Game | `interaction.py`, `sequential_game.py`, `synchronous_game.py` | Each agent owns `{A}`, `{B}`, or `{A,B}` | A speaker transmits a name, then the engine applies the minimal Naming Game inventory update | Compare sequential and synchronous-parallel update dynamics |
| Repeated Naming Convention Game | `naming_convention_game.py` | Each agent owns private interaction history, score, and optional committed action | Both players simultaneously choose an action; matching earns `+100`, mismatch earns `-50` | Convention formation, committee interventions, and empowerment analysis |

The inventory game is what the `run` and `benchmark` CLI commands execute. The
convention game is what the `experiment` command and the empowerment pipeline
execute.

## 3. Repository component map

### Runtime package

| File or area | Responsibility |
|---|---|
| `models.py` | Shared immutable records, inventory validation, run specifications, and benchmark summaries |
| `agent.py` | Mutable inventory-game agents, private histories, snapshots, and seeded initial populations |
| `interaction.py` | One isolated speaker-then-listener inventory-game interaction and authoritative state transition |
| `reasoning_game.py` | Optional evidence-based prompts and reasoning-task configuration for the inventory game |
| `sequential_game.py` | Dependent random-sequential inventory updates |
| `synchronous_game.py` | Immutable round snapshots, disjoint pairs, concurrent execution, and end-of-round update barrier |
| `api_client.py` | Stateless remote and mock LLM clients, endpoint discovery, retry policy, request semaphore, and aggregate request statistics |
| `local_model_types.py` | Provider-independent protocols and result types for constrained candidate decisions |
| `gemma_local_client.py` | Lazy local Gemma 4 runtime, candidate scoring, optional explanation generation, and serialized inference |
| `runner.py` | Run construction, summary calculation, and benchmark artifact writing |
| `benchmark.py` | Matched-budget grid construction and orchestration |
| `naming_convention_game.py` | Repeated payoff game, private bounded prompt memory, simultaneous decisions, interventions, and convergence checks |
| `empowerment_experiment.py` | Episode grid, committee schedules, full-horizon runs, checkpointing, derived trajectory states, and Parquet compaction |
| `audit_logging.py` | Append-only API status logs and deterministic sampled prompt/response audit traces |
| `analysis/` | Offline empowerment estimators, episode metrics, null models, compatibility normalization, reports, and plots |
| `potsdam_network.py` | Optional restricted Windows VPN bridge preparation when running the University client under WSL |
| `cli.py` | Argument parsing and provider/run dispatch |

### Supporting areas

| Area | Responsibility |
|---|---|
| `configs/` | Benchmark and empowerment YAML configurations |
| `tests/` | Offline behavioral contracts using mock or fake clients; no intended live API access |
| `scripts/Potsdam/` | University API diagnostics, budget helpers, WSL/Windows bridge, and cluster job material |
| `docs/university_llm_api.md` | Detailed University proxy operations and dated model information |
| `docs/committee_empowerment_guide.md` | Human-facing convention-game and empowerment guide |
| `external/AI-norms/` | Vendored reference implementation and assets that inspired the repeated convention experiment |
| `results/` | Runtime output location when default paths are used; it is not application state |

## 4. Binary inventory Naming Game design

### 4.1 State and initialization

An inventory is always exactly one of:

```text
{A}
{B}
{A, B}
```

`normalize_inventory()` enforces this invariant. `create_agents()` gives every
agent independent mutable state, history, random generator, and identity.
Default initialization is approximately balanced between singleton A and B.
For odd populations, the final singleton is seeded randomly before the whole
population is shuffled.

An `AgentSnapshot` is the immutable pair-local view passed into an interaction.
This is especially important in synchronous mode: pair tasks never receive a
live mutable `Agent`.

### 4.2 Basic pair interaction

One call to `execute_pair_interaction()` has this dependency chain:

```text
speaker snapshot
    -> speaker LLM call
    -> validate or locally repair selected name
    -> listener LLM call using the selected name
    -> validate listener report
    -> engine-authoritative inventory update
    -> immutable InteractionResult
```

There are normally two logical LLM calls per pair:

1. The speaker must choose a name in its own inventory.
2. The listener reports whether it already knew that name.

The model does not control the basic-game transition. The engine computes
ground truth from the pre-interaction inventories:

- Success: the listener already contains the selected name. Both inventories
  collapse to the selected singleton.
- Failure: the listener does not contain the name. The speaker is unchanged;
  the listener adds the selected name and therefore becomes `{A,B}`.

Malformed speaker output is repaired by a seeded local choice from the
speaker's immutable inventory. A malformed or incorrect listener report is
logged as invalid, but the engine result is still applied. This makes output
validation observable without granting free-form model text authority over
the basic transition.

If a client exposes `complete_constrained()`, the basic speaker decision uses
candidate scoring over its legal inventory names. Otherwise the client uses
the JSON generation contract.

### 4.3 Optional reasoning interaction

`reasoning_fraction` selects what fraction of inventory-game interactions use
the reasoning path. Any positive fraction requires a separate YAML or JSON
reasoning-task file containing:

- a non-empty task;
- claims for both A and B;
- per-agent evidence and/or default evidence.

The reasoning speaker chooses an inventory name and supplies a reason. The
listener sees its own evidence plus the transmitted claim and reason, then
returns a new inventory. Unlike the basic game, the listener's valid reasoning
response can directly determine its next state. Invalid listener output leaves
the listener unchanged. The speaker remains unchanged in this path.

### 4.4 Sequential engine

`SequentialNamingGame` samples an ordered speaker/listener pair, completes the
two-call interaction, mutates those two agents immediately, records their
private history entries, and only then samples the next pair.

Interactions within one trajectory are causally dependent and are therefore
not parallelized. The shared remote client may still serve other independent
work, but a single sequential engine awaits each pair before continuing.

After every interaction, the engine records population counts and consensus.
Inventory-game consensus means all agents are `{A}` or all agents are `{B}`.

### 4.5 Synchronous-parallel engine

`SynchronousParallelNamingGame` implements a real round barrier:

1. Snapshot every agent before creating pair coroutines.
2. Shuffle agent IDs.
3. Form disjoint ordered pairs.
4. Leave one seeded random agent idle when the population is odd.
5. Run all pair interactions with `asyncio.gather()`.
6. Apply every pair result only after all pairs have completed.
7. Record round-level counts, wall time, slowest pair, idle agent, and
   consensus.

Because pairs are disjoint and all read the same immutable round snapshot,
task completion order cannot leak into the state transition.

### 4.6 Matched benchmark budget

For population size `N` and `R` synchronous rounds:

```text
pairs_per_round = floor(N / 2)
matched_pair_interactions = floor(N / 2) * R
expected_logical_LLM_calls = 2 * matched_pair_interactions
```

The sequential and synchronous runs receive the same initial population and
pair-interaction budget. This controls total work; it does not claim that the
dynamics are equivalent. The benchmark grid itself is currently executed
serially across model, population size, round count, replicate, and update
mode.

## 5. Repeated Naming Convention Game design

### 5.1 Agent and game state

`NamingConventionGame` is a random-sequential population engine. Agent IDs are
one-based in this game. A `ConventionAgent` stores:

- complete private history;
- partner IDs from real interactions;
- cumulative score and score history;
- an optional permanently committed action.

Only the last `memory_size` history entries are exposed in the next prompt.
The complete history remains available to the evaluator and output pipeline.
A stateless client can serve the whole population because agent-specific state
is reconstructed explicitly in each request.

The default topology is complete. An explicit adjacency mapping can restrict
neighbors. Pair sampling first chooses a player uniformly from all agents and
then chooses that player's partner from its adjacency list, so the resulting
ordered-edge distribution depends on graph degrees.

### 5.2 One convention interaction

For every interaction:

1. Sample an ordered pair.
2. Independently shuffle the displayed action order for each player to reduce
   presentation-order bias.
3. Freeze each player's visible pre-interaction memory.
4. Ask both players for decisions concurrently with `asyncio.gather()`.
5. Apply an intervention or committed action without an API call when relevant.
6. Award both players `+100` for a match or `-50` for a mismatch.
7. Append the real outcome to both private histories.
8. Store a full immutable audit record.

Population interactions are sequential: the next pair can observe memory
created by the previous pair. Only the two decisions inside one pair are
simultaneous.

### 5.3 Information boundary and prompt contract

An ordinary agent sees:

- the available actions;
- the payoff rule;
- its own bounded recent choices, partner choices, and payoffs;
- a locally renumbered round and score derived from that visible window.

It is deliberately not shown:

- population membership or global state;
- partner identity;
- committee membership or intervention metadata;
- the global interaction index;
- another agent's memory;
- derived macrostates, rolling shares, or target outcomes.

This information boundary is an experimental invariant, not a formatting
detail.

Three response formats exist:

| Format | Expected result | Typical execution path |
|---|---|---|
| `json_reason` | JSON action/value plus reason | Generated text, parsed and retried if invalid |
| `choice_reason` | Legal choice on line one and `Reason:` on line two | Generated text, or constrained choice plus rationale on a capable local provider |
| `choice_only` | Exactly one legal action | Generated text, or constrained candidate scoring |

Generated invalid responses are retried up to
`invalid_response_retries + 1` validation attempts. Exhaustion raises
`InvalidConventionResponse`; no action is inferred from free-form reasoning.

### 5.4 Convergence in the base convention engine

The base game can stop on convergence over a stage-local tail window. The
default window is `3 * base_population_size`. It first requires the fraction
of successful pair interactions in the window to meet the threshold. A
convention label is returned only when the action share also supports that
label at the threshold. When `target_action` is supplied, the target's share
must meet the threshold.

The empowerment experiment intentionally calls this engine with
`stop_on_convergence=False`. Every episode runs its fixed configured horizon,
and separate rolling-share outcome logic is applied afterward.

### 5.5 Committee intervention regimes

The empowerment layer creates deterministic episode specifications for these
regimes:

| Regime | Initial memory | Policies |
|---|---|---|
| `neutral` | Empty | `always_A`, `always_B`, `no_committee` |
| `consensus_attack` | Successful incumbent consensus memory | `support_incumbent`, `promote_alternative`, `no_committee` |
| `pulse` | Successful incumbent consensus memory | `alternative_pulse`, `no_pulse` |

Committee membership is sampled reproducibly from the episode seed. A forced
decision skips the API request but still produces a real experience in both
agents' memories. After a pulse expires, former committee members return to
ordinary LLM decisions with the pulse experiences still in memory.

`swap` and `inject` committed-minority modes also exist in the lower-level game
API. `swap` converts existing agents; `inject` adds agents and rebuilds a
complete graph. The empowerment experiment uses temporary
`CommitteeSchedule` interventions rather than these committed-minority helpers.

### 5.6 Population-round terminology

In the empowerment path:

```text
1 population round = N pair interactions
max interactions per episode = N * max_population_rounds
```

This is not a disjoint-pair synchronous round. It represents `2N` player
participation slots and approximately two participations per agent on average.
Keep this definition explicit in a successor package.

## 6. Parallelism and state-safety model

There are several nested concurrency limits:

| Layer | Current behavior |
|---|---|
| Remote request concurrency | `AsyncLLMClient` owns an `asyncio.Semaphore(concurrency)` |
| One inventory pair | Speaker call completes before the listener call |
| One convention pair | Both player decisions run concurrently |
| One synchronous inventory round | All disjoint pairs run concurrently, then update at a barrier |
| Empowerment episodes | Up to `episode_concurrency` episodes progress concurrently |
| Requests across all empowerment episodes | The one shared client enforces `request_concurrency` globally |
| Local Gemma | Hard-coded concurrency of one; blocking inference is moved to a thread and serialized |
| Matched benchmark trajectories | Grid traversal is currently serial |

Two constraints are essential:

1. Concurrency must not change which state a decision sees. Use immutable
   snapshots and explicit update barriers.
2. Provider clients must not contain agent conversation state. Their shared
   state is limited to transport resources, endpoint/model discovery,
   semaphores, and aggregate request metrics.

The remote implementation uses synchronous `requests` operations through
`asyncio.to_thread()`. A new package should put transport behind an interface
so an async-native HTTP implementation can be adopted without changing game
logic.

## 7. LLM provider layer

### 7.1 Common contract

The minimum `LLMClient` protocol is:

```python
async def complete(
    messages: list[dict[str, str]],
    *,
    temperature: float,
    max_tokens: int,
    seed: int | None = None,
) -> LLMResponse: ...
```

Game code additionally discovers optional capabilities by checking for
`complete_constrained()` and `complete_decision()`. The associated typed
protocols live in `local_model_types.py`. Provider identity is audit metadata;
capability detection determines the decision path.

`LLMResponse` carries content, returned model, latency, retry count, HTTP
status, tokens, finish reason, and an optional raw response. Callers own the
client lifecycle and must call `close()`.

### 7.2 University of Potsdam proxy

`AsyncLLMClient` is the current University adapter. It reads the
repository-root `.env` unless credentials and URL are passed directly:

```dotenv
POTSDAM_API_KEY=replace-with-real-key
BASE_POTSDAM_LLM_URL=https://replace-with-proxy-base-url
POTSDAM_MODEL=gwdg/qwen3-30b-a3b-instruct-2507
```

Never print, log, commit, or put the key into exception text.

Endpoint discovery works as follows:

1. Try `GET {base_url}/models`.
2. If and only if it returns 404, try `GET {base_url}/v1/models`.
3. Use the matching `/chat/completions` or `/v1/chat/completions` path.
4. Require the exact requested model ID to appear in the model list.
5. Stop clearly on 401 or 403; do not probe alternate authentication paths.

The operational proxy endpoints currently documented by this repository are:

| Purpose | Method and path |
|---|---|
| Model discovery | `GET {base_url}/models` |
| Model pricing and limits | `GET {base_url}/v1/model/info` |
| Account budget | `GET {base_url}/user/info` |
| Chat completion | `POST {base_url}/chat/completions` |

Only model discovery and chat completion are used by `AsyncLLMClient`. Pricing,
limits, and account budget should be queried separately before a large run.

The client sends an OpenAI-compatible request body with `model`, `messages`,
`temperature`, `max_tokens`, and an optional `seed`.

Transient 429 and 5xx responses, timeouts, and connection errors receive
bounded retries. `Retry-After` is honored when present; otherwise the client
uses exponential backoff with jitter. Permanent request and authentication
errors are not blindly retried.

Under WSL, requests to the known University proxy host can automatically use a
restricted Windows CONNECT bridge so the Windows VPN route is available. TLS
remains end-to-end, and the bridge only permits the configured University host
on port 443. See `potsdam_network.py` and `scripts/Potsdam/` before changing
this behavior.

### 7.3 University example using the package client

This validates the model list before the first completion:

```python
import asyncio

from naming_game.api_client import AsyncLLMClient


async def main() -> None:
    client = AsyncLLMClient(
        model="gwdg/qwen3-30b-a3b-instruct-2507",
        concurrency=10,
        timeout_seconds=60,
        max_retries=2,
    )
    try:
        available = await client.validate_model()
        print(f"The proxy currently lists {len(available)} models.")
        response = await client.complete(
            [{"role": "user", "content": "Reply with exactly: API works."}],
            temperature=0.0,
            max_tokens=16,
        )
        print(response.content)
        print(response.usage)
    finally:
        client.close()


asyncio.run(main())
```

Run the maintained preflight instead of writing a new diagnostic when
possible:

```bash
conda run --live-stream -n MA-CC python \
  scripts/Potsdam/check_university_api.py

# Add one minimal billable completion:
conda run --live-stream -n MA-CC python \
  scripts/Potsdam/check_university_api.py --chat
```

### 7.4 University example using raw HTTP

Use this only when debugging the provider independently of the package. Do not
print request headers.

```python
import os

import requests
from dotenv import load_dotenv

load_dotenv(".env")
base_url = os.environ["BASE_POTSDAM_LLM_URL"].rstrip("/")
headers = {
    "Authorization": f"Bearer {os.environ['POTSDAM_API_KEY']}",
    "Content-Type": "application/json",
}

models = requests.get(
    f"{base_url}/models",
    headers=headers,
    timeout=60,
)
models.raise_for_status()

response = requests.post(
    f"{base_url}/chat/completions",
    headers=headers,
    json={
        "model": "gwdg/qwen3-30b-a3b-instruct-2507",
        "messages": [
            {"role": "user", "content": "Reply with exactly: API works."}
        ],
        "temperature": 0,
        "max_tokens": 16,
    },
    timeout=120,
)
response.raise_for_status()
print(response.json()["choices"][0]["message"]["content"])
```

The raw example omits the client's 404 `/v1` fallback, model validation,
bounded retries, request semaphore, and WSL bridge. It should not replace the
shared adapter in application code.

### 7.5 University provider in an experiment YAML

```yaml
provider: university
model: gwdg/qwen3-30b-a3b-instruct-2507
fallback_provider: university
allow_fallback: false
request_concurrency: 10
episode_concurrency: 2
timeout_seconds: 60
max_retries: 2
```

The supplied live University examples currently use
`configs/empowerment_pilot_test_university.yaml` and
`configs/empowerment_pilot_university.yaml`. Model availability is dynamic;
query `/models` immediately before a substantial run.

### 7.6 Official OpenAI, mock, and local Gemma providers

- `OpenAIAsyncLLMClient` reads `OPENAI_API_KEY`, fixes the base URL to the
  official OpenAI v1 API, and reuses the remote client's endpoint validation,
  retries, semaphore, schema parsing, and statistics.
- `MockAsyncLLMClient` is deterministic and latency-configurable. It recognizes
  the repository's prompt markers, tracks maximum active requests, and is the
  normal test/benchmark provider.
- `GemmaLocalAsyncLLMClient` supports only `google/gemma-4-12B-it`, lazily loads
  Transformers and the model, requires Transformers 5+, requires `HF_HOME`,
  refuses CPU unless explicitly permitted, and serializes inference. It can
  score candidate sequences and return their normalized probabilities.

## 8. Logging, audit, checkpoints, and artifacts

### 8.1 Inventory benchmark artifacts

Every `run_single()` writes these files under its output directory:

| Artifact | Contents |
|---|---|
| `interactions_<run_id>.jsonl` | One complete interaction audit row per pair |
| `states_<run_id>.csv` | Sequential population counts after every interaction |
| `rounds_<run_id>.csv` | Synchronous population counts and timing after every round |
| `config_<run_id>.json` | Exact run specification, backend, and initial counts |
| `benchmark_summary.csv` | One aggregate row per run; appended by default |

The non-applicable state or round CSV is still created with a header and zero
data rows. This keeps artifact discovery stable without pretending that
sequential steps and synchronous rounds are the same measurement.

The interaction JSONL includes prompt-result validation, pre/post inventories,
model text, request latency, retries, status, tokens, and constrained-choice
probabilities when available. It does not contain the API key.

### 8.2 Empowerment experiment checkpoints and histories

Each episode is first written to two temporary Parquet files and atomically
renamed into:

```text
<output>/.episode_shards/<experiment_fingerprint>/
```

The fingerprint covers data-generating settings but deliberately permits some
grid and concurrency expansion. On resume, an episode is treated as complete
only when both its interaction shard and episode-summary shard exist.

After all requested episodes are checkpointed, shards are compacted and
validated before atomic replacement of:

| Artifact | Contents |
|---|---|
| `interactions.parquet` | Analysis-ready pair trajectory, decisions, memories, intervention flags, and derived state |
| `episodes.parquet` | One row per episode with terminal, takeover, recovery, persistence, and intervention summaries |
| `experiment_config.json` | Resolved experiment configuration |

Checkpoint shards are recoverable state. Do not delete them during an
incomplete run. `clear_completed_shards()` exists for intentional cleanup only
after successful archival.

### 8.3 Empowerment operational and audit logs

When configured, `run_experiment()` installs a file log at:

```text
<output>/logs/experiment.log
```

`ExperimentAuditLogger` also maintains:

| Artifact | Contents |
|---|---|
| `api_call_status.jsonl` | Append-only status rows for attempts, retries, success, skipped calls, and forced no-call decisions |
| `audit_traces.jsonl` | Deterministically sampled detailed request/response traces |
| `audit_report.md` | Human-readable rendering of those sampled traces |

Trace selection is seeded and based on configured population-round
checkpoints. Depending on configuration, a trace may include the reconstructed
prompt, immutable pre-interaction memory, raw provider response, parsed output,
choice scores, payoff, and post-interaction memory.

Logging failures are caught and reported; they do not change agent choices or
the experiment transition. Request headers and credentials are never included.

Progress bars and log messages go to stderr. The final machine-readable JSON
goes to stdout. Use `conda run --live-stream` so progress is visible rather
than buffered.

## 9. Metrics and analysis

### 9.1 Inventory benchmark metrics

`summarize_run()` calculates:

- total wall time and seconds per pair;
- seconds per matched synchronous-round equivalent;
- actual synchronous-round timing where applicable;
- mean, median, p90, and maximum request latency;
- logical call attempts, retries, and throughput;
- prompt, completion, and total tokens when the provider reports them;
- successful and failed basic naming interactions;
- initial and final A/B/AB population counts;
- consensus and the first consensus interaction index;
- pairs per synchronous round and mean slowest-pair time.

Reasoning interactions have `naming_success=None`, so they are not counted as
basic naming successes or failures.

When redesigning this summary, distinguish logical model decisions, HTTP
attempts, validation retries, provider retries, and failures explicitly. The
current structures expose all of these at different layers, but some legacy
summary field names are broader than their calculation.

### 9.2 Derived convention trajectory state

`derive_episode()` processes stored pair rows in order. Its rolling window
contains `window_interactions` pair interactions and therefore twice that many
action observations.

For action A:

```text
rolling_share_A = count(A across both player outputs in the window)
                  / number of player outputs in the window
```

Before a complete window exists, the share is reported from the available
prefix but resolution events cannot trigger. Derived states are:

- `macrostate_binary`: 1 above 0.5, 0 below 0.5, and previous state carried
  through an exact tie;
- `macrostate_three`: `B_dominant` below 0.4, `mixed` from 0.4 through 0.6,
  and `A_dominant` above 0.6;
- `resolved_state`: A at or above `resolution_threshold`, B at or below
  `1 - resolution_threshold`, otherwise `unresolved`, but only with a full
  window.

Episode summaries include:

- first resolved convention and time;
- final convention and terminal A share;
- whether the promoted alternative was ever reached;
- terminal takeover and incumbent survival;
- peak alternative displacement and time to peak;
- pulse recovery time and censoring;
- permanent flip;
- total forced committee actions;
- post-consensus persistence measured at population-round endpoints.

### 9.3 Empowerment estimands

The analysis is offline: it reads compacted Parquet histories and never calls
an LLM.

The intervention/policy variable is `G = committee_policy`. The main terminal
outcome is `Y` in `{A, B, unresolved}`. Within each experimental stratum, the
terminal empowerment estimate is:

```text
I(G; Y)
```

A sensitivity estimate excludes unresolved outcomes and uses only A/B. The
lagged dynamical estimate uses the state now and after a configured horizon:

```text
I(G; S[t + h] | S[t])
```

It is computed for both binary and three-state macrostates. A horizon of `h`
population rounds becomes a row lag of `h * N` pair interactions.

Entropy uses base-2 logarithms, so information values are in bits. Each table
reports:

- Jeffreys-smoothed estimate, using `+0.5` in every complete contingency cell;
- unsmoothed plug-in estimate;
- Miller-Madow sensitivity estimate;
- observation count.

The Jeffreys estimate is the primary reported value. Complete policy and
outcome/state levels are constructed before smoothing so unobserved cells are
not silently removed.

Estimation status is based on completed episodes per policy within a stratum:

| Minimum episodes per policy | Status |
|---|---|
| Fewer than 5, or fewer than two policies | Non-estimable |
| 5 to 9 | Exploratory and highly noisy |
| 10 or more | Estimable under the current reporting rule |

Bootstrap confidence intervals resample whole episode IDs, preserving
within-episode dependence. Binary episode probabilities use Wilson intervals;
recovery medians use episode bootstrap intervals.

Efficiency is defined as terminal Jeffreys information divided by expected
committee actions when that denominator is positive.

### 9.4 Nulls and diagnostics

The analysis performs these diagnostics:

- episode-policy shuffle within design strata for terminal and lagged nulls;
- within-episode circular trajectory shifts for temporal lag diagnostics;
- seeded A/B label swaps on half of balanced episode groups to test numerical
  invariance;
- a zero-committee baseline compared with its shuffle distribution;
- warnings for missing policies, small cells, missing attack directions, and
  other design limitations.

### 9.5 Analysis outputs

`analyze-empowerment` writes:

| Artifact | Contents |
|---|---|
| `empowerment_estimates.parquet` | Terminal and lagged MI/CMI, sensitivity estimates, intervals, null comparison, and efficiency |
| `episode_metrics.parquet` | Grouped outcome, timing, recovery, persistence, and intervention metrics |
| `null_results.parquet` | Shuffle and circular-shift estimates |
| `label_swap_invariance.parquet` | Original/swapped comparison and tolerance check |
| `no_committee_baseline.parquet` | Zero-committee diagnostic |
| `summary.md` | Human-readable results and warnings |
| `plots/experiment_summary.png` | Main summary plot |
| `plots/pulse_summary.png` | Pulse plot when pulse data exist |
| `analysis_config.json` | Exact offline analysis settings |

## 10. Operational commands

### Environment and editable install

```bash
conda env update -n MA-CC -f environment.yml
conda run -n MA-CC python -m pip install -e .
```

### Offline tests

```bash
conda run -n MA-CC pytest
```

### Offline matched benchmark

```bash
conda run -n MA-CC python -m naming_game.cli benchmark --mock
```

### One sequential inventory trajectory

```bash
conda run -n MA-CC python -m naming_game.cli run \
  --update-mode sequential \
  --num-agents 10 \
  --num-interactions 50 \
  --reasoning-fraction 0 \
  --seed 1 \
  --mock
```

### One synchronous inventory trajectory

```bash
conda run -n MA-CC python -m naming_game.cli run \
  --update-mode synchronous_parallel \
  --num-agents 10 \
  --rounds 10 \
  --reasoning-fraction 0 \
  --concurrency 20 \
  --seed 1 \
  --mock
```

### One live University inventory trajectory

This uses the repository-root `.env`, validates the exact model against the
proxy, and is potentially billable:

```bash
conda run --live-stream -n MA-CC naming-game run \
  --provider university \
  --model gwdg/qwen3-30b-a3b-instruct-2507 \
  --update-mode sequential \
  --num-agents 4 \
  --num-interactions 2 \
  --concurrency 2 \
  --seed 1
```

### Offline empowerment pilot

```bash
conda run --live-stream -n MA-CC naming-game experiment \
  --config configs/empowerment_pilot.yaml \
  --mock \
  --output-dir results/empowerment_pilot
```

### University empowerment smoke test

This is live and potentially billable. Connect the University/VPN network and
run the preflight first.

```bash
conda run --live-stream -n MA-CC python \
  scripts/Potsdam/check_university_api.py

conda run --live-stream -n MA-CC naming-game experiment \
  --config configs/empowerment_pilot_test_university.yaml \
  --no-resume \
  --output-dir results/empowerment_university_smoke
```

### Re-run analysis without model access

```bash
conda run --live-stream -n MA-CC naming-game analyze-empowerment \
  --history-dir results/empowerment_pilot \
  --bootstrap-resamples 1000 \
  --null-permutations 1000
```

## 11. Behavioral invariants to preserve during migration

A successor package should preserve or deliberately version these contracts:

1. Importing a module must not create a client, load a model, call an API, or
   start an experiment.
2. Game engines depend on provider protocols, not concrete providers.
3. Remote clients remain stateless with respect to agents and conversations.
4. Credentials never enter logs, artifacts, prompts, or safe exception text.
5. The exact requested provider model is validated; no silent model
   substitution occurs.
6. Inventory-game basic transitions remain engine-authoritative.
7. Convention decisions remain simultaneous within a pair.
8. Synchronous inventory rounds use immutable global snapshots, disjoint
   pairs, and a real update barrier.
9. Agent-visible memory stays separate from complete evaluator history.
10. Population state, committee policy, and evaluator-derived metrics never
    leak into ordinary convention prompts.
11. Forced decisions skip the API but still update real agent memory.
12. Randomness is local and seeded; episode identity includes decision-contract
    settings.
13. Model response validation, provider retries, and local repairs remain
    auditable.
14. Simulation output is stored before offline analysis begins.
15. Analysis failures never retroactively mark a successfully compacted
    simulation as failed.
16. Checkpoint and compacted artifact replacement remains crash-safe.
17. Information metrics preserve episode-level dependence during resampling.
18. The two meanings of "round" remain explicit and typed.

## 12. Suggested `mas_cc` target boundaries

One reasonable destination layout is:

```text
src/mas_cc/
├── __init__.py
├── cli.py
├── core/
│   ├── ids.py
│   ├── random.py
│   ├── records.py
│   └── validation.py
├── providers/
│   ├── base.py
│   ├── university.py
│   ├── openai.py
│   ├── mock.py
│   └── gemma_local.py
├── games/
│   ├── inventory/
│   │   ├── state.py
│   │   ├── interaction.py
│   │   ├── sequential.py
│   │   └── synchronous.py
│   └── convention/
│       ├── state.py
│       ├── prompts.py
│       ├── engine.py
│       └── interventions.py
├── experiments/
│   ├── benchmark.py
│   └── empowerment.py
├── analysis/
│   ├── estimators.py
│   ├── metrics.py
│   ├── nulls.py
│   └── reporting.py
└── storage/
    ├── schemas.py
    ├── checkpoints.py
    ├── audit.py
    └── artifacts.py
```

The exact names can change, but the dependency direction should remain clear:

```text
core records/protocols
    <- providers
    <- game engines
    <- experiment orchestration
    <- storage and offline analysis
    <- CLI
```

Game engines should not import the CLI, plotting, pandas, environment loading,
or a concrete remote provider. Analysis should consume versioned persisted
schemas rather than live game objects.

## 13. Safe staged migration plan

1. Add `src/mas_cc/__init__.py` and install both packages from the same
   distribution.
2. Define new provider protocols and domain records, with tests copied from
   the current behavioral contracts.
3. Move pure validation and transition functions first. These have the least
   operational coupling.
4. Rebuild one game engine at a time behind parity tests. Keep inventory and
   convention domains separate.
5. Move provider adapters after game code depends only on the new protocols.
6. Version artifact schemas before migrating checkpoint or analysis code.
7. Run old and new implementations against identical mock seeds and compare
   normalized trajectories and summaries.
8. Add `naming_game` compatibility wrappers that import from `mas_cc` only
   after the new implementation owns the behavior.
9. Emit deprecation warnings at public entry points, not deep internal calls.
10. Change the distribution name and console entry point only when deployment
    and downstream scripts are ready.

Avoid moving files merely to reproduce the current dependency graph under a
new directory. The migration is the opportunity to separate domain state,
provider capabilities, orchestration, storage schemas, and reporting.

## 14. Tests that define the current contracts

Use these tests as migration guides:

| Test area | Primary contract |
|---|---|
| `test_interaction.py` | Inventory transition authority, validation, and repairs |
| `test_sequential_game.py` | Immediate dependent updates and trajectory records |
| `test_synchronous_game.py` | Snapshot isolation, disjoint concurrency, and update barrier |
| `test_benchmark.py` | Matched budgets, grid behavior, summaries, and artifacts |
| `test_mock_api.py` | Deterministic provider behavior and request isolation |
| `test_naming_convention_game.py` | Prompt privacy, memory, payoffs, response validation, and interventions |
| `test_convention_output_formats.py` | JSON, choice-plus-reason, and choice-only contracts |
| `test_empowerment_experiment.py` | Episode design, derived outcomes, checkpoint schemas, and resume behavior |
| `test_empowerment_estimators.py` | MI/CMI estimators, smoothing, nulls, and invariance |
| `test_empowerment_cli.py` | CLI experiment/analysis integration |
| `test_constrained_client_contract.py` | Capability protocols and constrained result types |
| `test_gemma_local_client_unit.py` | Local lazy loading, serialization, scoring, and validation |
| `test_gemma_language_game_integration.py` | Local-provider integration with both game paths |

## 15. Current design points to revisit explicitly

These are not instructions to change behavior immediately. They are decisions
the new package should make deliberately:

- Use distinct type names for inventory consensus, convention coordination,
  resolved rolling state, and experimental terminal outcome.
- Use distinct types for synchronous inventory rounds and empowerment
  population rounds.
- Separate logical decisions, HTTP attempts, provider retries, validation
  attempts, repaired responses, and permanent failures in metrics.
- Replace duck-typed optional provider methods with an explicit capability
  interface or capability object.
- Version Parquet schemas and migrations independently from prompt versions.
- Keep transport endpoint discovery separate from chat execution and from WSL
  network preparation.
- Decide whether the successor uses an async-native HTTP client while retaining
  the same retry, timeout, and security policies.
- Make log-handler ownership explicit if multiple experiments may run in one
  Python process.
- Keep local-model token-work accounting distinct from remotely billed token
  accounting.

## 16. Further repository references

- General commands: [`../README.md`](../README.md)
- University proxy operations: [`university_llm_api.md`](university_llm_api.md)
- Convention and empowerment details:
  [`committee_empowerment_guide.md`](committee_empowerment_guide.md)
- Package metadata: [`../pyproject.toml`](../pyproject.toml)
- Benchmark grid: [`../configs/speed_test.yaml`](../configs/speed_test.yaml)
- Full empowerment configuration:
  [`../configs/empowerment.yaml`](../configs/empowerment.yaml)
