# Naming Convention Game and Committee-Empowerment Guide

This guide explains the repeated Naming Convention Game, the committee
interventions, every experiment configuration option, how to choose an LLM
provider, how to run an experiment, and where to inspect the results.

## 1. What is the Naming Convention Game?

The game studies how a population of LLM agents develops a shared convention.
In this first probe there are 24 agents and two possible names, `A` and `B`.

At each interaction:

1. Two agents are sampled from the population.
2. Each agent independently chooses `A` or `B` from the same pre-interaction
   state. Their decisions are therefore simultaneous.
3. If the choices match, both agents receive `+100` points.
4. If the choices differ, both agents receive `-50` points.
5. Each agent records its own choice, its partner's choice, and the payoff in
   its private memory.

An ordinary agent sees only its own most recent interactions. It is not shown
the committee, population state, global interaction number, other agents'
memories, rolling convention share, or experimental outcome.

A **population round** is `N` pair interactions, where `N` is the population
size. With `N = 24`, one population round is 24 pair interactions. A 30-round
episode therefore contains 720 pair interactions. An ordinary pair can produce
two LLM API calls; a forced committee decision does not call the LLM.

## 2. Memory, intervention, and measurement are different layers

The implementation deliberately separates three concepts:

- **Agent memory:** the bounded private history used in the agent's next prompt.
- **Committee intervention:** a temporary rule that can force selected agents
  to choose `A` or `B`.
- **Experiment history:** the complete external record used for analysis.

When a committee member is forced to act, the LLM call for that decision is
skipped. The resulting choice, partner choice, and payoff are still recorded in
both agents' normal memories. After a pulse ends, the committee member returns
to ordinary LLM behavior with those real experiences in its memory.

This means that a pulse can continue to influence later behavior through
memory even though the direct intervention is no longer active.

## 3. Experimental regimes

The `regimes` configuration field selects one or more of these experiments.

### `neutral`

All agents begin with empty memories. The episode policy `G` is one of:

| Policy | Behavior |
|---|---|
| `always_A` | Committee members always choose A. |
| `always_B` | Committee members always choose B. |
| `no_committee` | No action is forced. |

This regime asks whether a committee can select which convention emerges from
a neutral start.

### `consensus_attack`

Agents begin with a complete successful memory supporting either A or B. Both
directions are run: consensus A challenged by B and consensus B challenged by
A. The policy is one of:

| Policy | Behavior |
|---|---|
| `support_incumbent` | Committee members reinforce the initial convention. |
| `promote_alternative` | Committee members always choose the opposite convention. |
| `no_committee` | The population runs without forced actions. |

This regime measures takeover, resilience, and possible A/B asymmetry without
assuming in advance that either name is stronger.

### `pulse`

Agents begin at consensus. The policy is one of:

| Policy | Behavior |
|---|---|
| `alternative_pulse` | Committee members promote the alternative only during the configured pulse window. |
| `no_pulse` | Matched control episode with no forced actions. |

After the pulse window, committee members become ordinary agents again. Each
configured pulse duration is analyzed as a separate experimental stratum.

## 4. Configuration reference

The full configuration is
[`configs/empowerment.yaml`](../configs/empowerment.yaml). The inexpensive
starting point is
[`configs/empowerment_pilot.yaml`](../configs/empowerment_pilot.yaml).

### Game and intervention fields

| Field | Meaning and options |
|---|---|
| `population_size` | Number of agents `N`. Must be at least 2. The standard experiment uses 24. |
| `names` | The available conventions. This first probe requires exactly `[A, B]`. |
| `memory_length` | Maximum number of private past interactions included in an ordinary agent's next prompt. `5` means only its five most recent encounters are visible. |
| `max_population_rounds` | Fixed episode duration. Episodes continue for this entire horizon even after consensus first appears. |
| `committee_sizes` | Committee sizes to sweep. `0` is essential as the null intervention baseline. Values cannot exceed `population_size`. |
| `pulse_rounds` | Pulse durations to sweep in the `pulse` regime. Each value is measured in population rounds. |
| `regimes` | Any subset of `neutral`, `consensus_attack`, and `pulse`. Use a one-item list to run only one regime. |
| `temperature` | LLM sampling temperature. `0` is more deterministic; larger values introduce more response variation. Keep it identical across matched cells. |
| `max_tokens` | Maximum tokens allowed for one convention decision. The prompt expects short JSON; `15` is normally sufficient. |
| `seed` | Base seed used for episode identities, committee selection, pair sampling, policy assignment, and provider seed values. |
| `convention_roles` | Optional calibrated `strong_name`, `weak_name`, and provenance `source`. These roles are stored in both Parquets and are never inferred from attack outcomes. Without them, analysis uses neutral labels such as `A_to_B`. |
| `auto_analyze` | Run disk-based analysis after successful Parquet compaction. Defaults to `true`. |
| `quick_bootstrap_resamples` | Episode-bootstrap count for automatic post-run analysis. Defaults to `200`. |
| `quick_null_permutations` | Shuffle/circular-shift permutation count for automatic post-run analysis. Defaults to `200`. |

### Replication fields

```yaml
replications:
  unit: per_policy
  count: 2
```

| `unit` | Interpretation of `count` |
|---|---|
| `per_policy` | Run `count` episodes for every exact policy in each experimental stratum. This gives balanced policy tables and is the canonical design. |
| `per_stratum` | Run `count` total episodes in a stratum and distribute them across policies. This is cheaper but provides fewer episodes per policy. |

The pilot uses two episodes per policy only to verify execution. The full
configuration uses 100 episodes per policy. The full grid contains 17,500
episodes, 12.6 million pair interactions, and at most 25.2 million model calls,
so do not launch it before checking a pilot's runtime and account limits.

### Outcome and trajectory fields

| Field | Meaning |
|---|---|
| `window_interactions` | Number of recent pair interactions used to compute the convention share. The standard value is `3 * N`, which is 72 for `N = 24`. Both agent outputs in every interaction are counted. |
| `resolution_threshold` | Required share for a resolved convention. At `0.95`, A is resolved when at least 95% of recent outputs are A, and B is resolved when at least 95% are B. Otherwise the state is unresolved. |

Before a complete rolling window exists, `rolling_share_A` is computed from the
available prefix, but consensus, takeover, and recovery events cannot trigger.

### Provider and reliability fields

| Field | Meaning and options |
|---|---|
| `provider` | Primary API provider: `university` or `openai`. |
| `model` | Model identifier understood by the selected provider. Model names are provider-specific. |
| `fallback_provider` | Provider to try if primary provider/model validation fails before any episode begins. |
| `fallback_model` | Optional model for the fallback provider. If omitted, `model` is reused. Reusing a model name only works if both providers expose that exact identifier. |
| `allow_fallback` | `false` stops on primary provider failure. `true` permits the explicit pre-run fallback. The provider never changes midway through an episode. |
| `request_concurrency` | Maximum simultaneous model requests across all episodes. Start conservatively to avoid rate limits. |
| `episode_concurrency` | Maximum episodes progressing concurrently. Each active ordinary pair can issue two requests. |
| `timeout_seconds` | Timeout for one HTTP request. |
| `max_retries` | Bounded retries for rate limits, connection failures, and transient server errors. Permanent authentication and request errors are not repeatedly retried. |

## 5. Choosing University or OpenAI

Provider selection is controlled by the YAML file, not by a separate CLI
flag. Copy the pilot configuration if you want to preserve the supplied file,
then edit `provider` and `model`.

### University of Potsdam proxy

The repository-root `.env` must contain:

```dotenv
POTSDAM_API_KEY=your-key
BASE_POTSDAM_LLM_URL=https://your-proxy-base-url
```

Use a model ID currently returned by the proxy's `/models` endpoint:

```yaml
provider: university
model: gwdg/qwen3-30b-a3b-instruct-2507
allow_fallback: false
```

See [`docs/university_llm_api.md`](university_llm_api.md) for proxy-specific
model discovery, budget, and rate-limit information.

### Official OpenAI API

The repository-root `.env` should contain:

```dotenv
OPENAI_API_KEY=your-key
```

Then select OpenAI and an OpenAI model:

```yaml
provider: openai
model: gpt-4o-mini
allow_fallback: false
```

The temporary legacy spelling `OPEN_API_KEY` is recognized, but it produces a
deprecation warning. Rename it to `OPENAI_API_KEY`.

### Explicit University-to-OpenAI fallback

```yaml
provider: university
model: gwdg/qwen3-30b-a3b-instruct-2507
fallback_provider: openai
fallback_model: gpt-4o-mini
allow_fallback: true
```

Fallback occurs only if provider/model validation fails before the experiment.
It is never silently activated during an episode. The actual provider and
model are recorded in every interaction and episode row.

## 6. Running an experiment

Use the repository's Python 3.11 Conda environment:

```bash
conda env update -n MA-CC -f environment.yml
conda run -n MA-CC python -m pip install -e .
```

### Step 1: run an offline mock pilot

The mock provider makes no external API calls and is the safest way to verify
the grid, Parquet output, and analysis pipeline:

For the smallest possible smoke test—one episode, five agents, and five
population rounds—run:

```bash
conda run -n MA-CC python -m naming_game.cli experiment \
  --config configs/empowerment_pilot_test.yaml \
  --mock \
  --output-dir results/empowerment_pilot_test
```

This performs exactly 25 pair interactions and still creates descriptive
plots. Its report says `empowerment not estimable from this run` because only
one policy episode is present. After it succeeds, the more
representative multi-condition pilot is:

```bash
conda run -n MA-CC python -m naming_game.cli experiment \
  --config configs/empowerment_pilot.yaml \
  --mock \
  --output-dir results/empowerment_pilot
```

### Step 2: run a real-provider pilot

Remove `--mock`. The provider and model come from the YAML file:

```bash
conda run -n MA-CC python -m naming_game.cli experiment \
  --config configs/empowerment_pilot.yaml \
  --output-dir results/empowerment_openai_pilot
```

Completed episode shards are checkpoints. Running the same command again with
the same configuration resumes incomplete work. Use `--no-resume` only when
you intentionally want to rerun and replace all episodes in that grid.
Automatic analysis begins only after both compacted Parquets have been written
and validated. Use `experiment --analyze` to force it when `auto_analyze` is
false. Analysis errors are logged but do not invalidate the completed run.

### Step 3: analyze the stored histories

Analysis does not call an LLM or rerun the game:

```bash
conda run -n MA-CC python -m naming_game.cli analyze-empowerment \
  --history-dir results/empowerment_openai_pilot \
  --horizons 1 3 5 10 \
  --bootstrap-resamples 1000 \
  --null-permutations 1000 \
  --seed 1
```

The default destination is `<history-dir>/analysis`; `--output-dir` remains
available when a separate destination is desired.

For a quick pipeline check, use small values such as 10 bootstrap resamples and
10 null permutations. Use the larger values for final reported results.

## 7. Where to inspect simulation results

After the experiment command, the selected output directory contains:

| File | What it contains |
|---|---|
| `interactions.parquet` | One row per pair interaction: participants, outputs, payoff, pre-interaction memories, forced-action flags, rolling share, binary/three-state macrostates, committee metadata, provider, and model. |
| `episodes.parquet` | One row per episode: initial condition, final convention, terminal share, first consensus, takeover, recovery/censoring, permanent flip, peak displacement, persistence, and committee-action count. |
| `experiment_config.json` | Exact resolved configuration used for the run. |
| `.episode_shards/<fingerprint>/` | Recoverable per-episode checkpoint files used for resume and final compaction. |

Read selected columns with pandas:

```bash
conda run -n MA-CC python -c '
import pandas as pd
episodes = pd.read_parquet("results/empowerment_openai_pilot/episodes.parquet")
columns = [
    "regime", "committee_size", "committee_policy", "initial_condition",
    "final_convention", "takeover", "recovery_time_population_rounds",
    "total_committee_actions",
]
print(episodes[columns].head(20).to_string(index=False))
'
```

Useful interaction-level columns include:

- `memory_i_before` and `memory_j_before`: JSON-encoded bounded private memory;
- `forced_i` and `forced_j`: whether the intervention replaced that decision;
- `pulse_active`: whether the alternative pulse was active;
- `rolling_share_A`: recent fraction of outputs equal to A;
- `macrostate_binary`: B-dominant `0`, A-dominant `1`, with tie carry-forward;
- `macrostate_three`: `B_dominant`, `mixed`, or `A_dominant`;
- `resolved_state`: `A`, `B`, `unresolved`, or unavailable during warm-up;
- `terminal_outcome`: episode-level `A`, `B`, or `unresolved` copied onto the
  trajectory for convenient filtering.

## 8. Where to inspect analysis results

After `analyze-empowerment`, the analysis output directory contains:

| File | Interpretation |
|---|---|
| `empowerment_estimates.parquet` | Terminal and lagged mutual information, Jeffreys-smoothed primary estimates, unsmoothed estimates, Miller–Madow sensitivity estimates, bootstrap intervals, and efficiency. |
| `episode_metrics.parquet` | Takeover and terminal probabilities, consensus times, peak displacement, recovery, permanent flips, committee actions, terminal share, and persistence. |
| `null_results.parquet` | Episode-policy shuffle nulls and circular-shift temporal diagnostics. |
| `label_swap_invariance.parquet` | A/B label-swap comparison and numerical invariance check. |
| `no_committee_baseline.parquet` | The `committee_size = 0` near-zero empowerment diagnostic. |
| `analysis_config.json` | Horizons, random seed, bootstrap count, and null-permutation count used for analysis. |

The `plots/` directory contains:

- `experiment_summary.png`, the four-panel convention, coordination, outcome,
  empowerment, and shuffle-null overview;
- `pulse_summary.png` when pulse histories are present.

`summary.md` records episode counts, policies and directions, terminal and
ever-crossed takeover probabilities, terminal empowerment and its null
comparison, consensus/recovery statistics, and data-quality warnings.

`terminal_takeover` means the promoted convention satisfies the resolution
criterion at the fixed endpoint. `ever_crossed` means it satisfied the
criterion at least once and may subsequently have recovered. The primary
takeover curve and the smallest-tested-fraction marker use terminal takeover.

### Interpreting empowerment intuitively

Terminal empowerment asks: **if we know which committee policy was assigned,
how much does that tell us about the final convention?**

- Near zero means the policy provides almost no information about the outcome.
- A larger value means different policies reliably lead to different outcomes.
- High takeover probability alone is not identical to high empowerment: a
  system could always end on the same convention regardless of the policy.

Lagged empowerment asks: **after accounting for the population's current
macrostate, does the committee policy help predict its future macrostate?**

Always compare the observed estimates with bootstrap intervals, the
episode-label-shuffle null, the zero-committee baseline, complementary dynamics
such as takeover/recovery, and the continuous `rolling_share_A` trajectories.

## 9. Common problems

- **Authentication failure:** check the provider-specific environment variable
  and do not mix an OpenAI key with the University proxy configuration.
- **Model not listed:** model names are provider-specific and may change. Select
  one currently exposed by the chosen provider.
- **HTTP 429:** lower `request_concurrency` and `episode_concurrency`, then
  resume the same output directory.
- **Run appears very large:** reduce committee sizes, regimes, pulse durations,
  or replication count in a copied pilot YAML.
- **No recovery value:** the episode did not recover before the fixed horizon;
  inspect `recovery_censored` rather than treating the missing value as zero.
- **Unexpected early consensus:** events are unavailable until a complete
  `window_interactions` window exists; inspect `insufficient_window` and
  `rolling_window_count` in the interaction history.
