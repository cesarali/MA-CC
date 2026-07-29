# Committee-Empowerment Probe Deliverables

**Status:** Implemented and verified  
**Date:** 2026-07-29

## Executive summary

The repository now contains a separate committee-empowerment experiment around
the repeated Naming Convention Game. It separates private agent memory,
temporary committee interventions, stored observations, derived trajectory
states, and statistical analysis.

The existing inventory-based Naming Game and benchmark remain available and
unchanged when the new experiment commands are not used.

## Delivered components

### Game and intervention layer

- Added a temporary intervention interface to `NamingConventionGame`.
- Committee actions override only the selected decision; they do not create a
  separate agent memory type.
- Forced interactions are recorded in both participants' ordinary memories.
- Committee members return to ordinary LLM behavior after a pulse.
- Pre-interaction bounded memories are captured for external trajectory logs.
- Episodes run for a fixed horizon while recording first consensus separately.

### Experiment runner

`src/naming_game/empowerment_experiment.py` provides:

- validated YAML experiment configuration;
- deterministic episode construction and seeding;
- neutral, consensus-attack, and pulse regimes;
- randomized committee identities;
- `per_policy` and `per_stratum` replication modes;
- rolling convention shares and binary/three-state macrostates;
- takeover, recovery, censoring, persistence, and permanent-flip outcomes;
- episode concurrency and global request concurrency;
- configuration-scoped episode checkpoints and resume behavior;
- final interaction-level and episode-level Parquet compaction.

### Provider layer

- Retained the University of Potsdam OpenAI-compatible proxy provider.
- Added an official OpenAI provider.
- Added provider identity, actual returned model, optional request seed, and
  safe lifecycle handling to the common asynchronous interface.
- Added explicit pre-run provider fallback; providers never switch midway
  through an episode.
- Added temporary compatibility with `OPEN_API_KEY`, with a warning to migrate
  to the standard `OPENAI_API_KEY` variable.

### Statistical analysis

The reusable `src/naming_game/analysis/` package includes:

- direct contingency-table mutual information;
- terminal empowerment with unresolved outcomes retained;
- resolved-only terminal sensitivity analysis;
- binary and three-state lagged conditional mutual information;
- Jeffreys `+1/2` smoothing over possible cells;
- unsmoothed and Miller–Madow sensitivity estimates;
- complete-episode bootstrap confidence intervals;
- within-stratum episode-policy shuffle nulls;
- circular-shift temporal diagnostics;
- balanced A/B label-swap diagnostics;
- zero-committee baseline diagnostics;
- takeover, consensus, displacement, recovery, flip, persistence, action-count,
  and efficiency metrics;
- required trajectory and empowerment plots.

Analysis consumes only stored Parquet histories. It does not import the game or
provider layers and does not make model calls.

### CLI commands

Two commands were added:

```text
naming-game experiment
naming-game analyze-empowerment
```

They are also available through:

```text
python -m naming_game.cli experiment
python -m naming_game.cli analyze-empowerment
```

### Configurations

| Configuration | Purpose |
|---|---|
| `configs/empowerment_pilot_test.yaml` | One episode, 5 agents, 5 population rounds, and 25 pair interactions. Intended only as an end-to-end smoke test. |
| `configs/empowerment_pilot.yaml` | Small multi-condition pilot for checking behavior before a full run. |
| `configs/empowerment.yaml` | Canonical 100-episodes-per-policy experiment grid. |

The full configuration contains:

- 17,500 episodes;
- 12.6 million pair interactions;
- at most 25.2 million model calls before accounting for forced committee
  decisions.

It should not be launched before measuring cost, time, and rate limits with a
pilot.

### Documentation

- `docs/committee_empowerment_guide.md` explains the game, episodes, policies,
  configurations, provider selection, commands, outputs, metrics, and common
  problems.
- `docs/university_llm_api.md` contains University proxy-specific notes.
- The main `README.md` links to the new guide and includes quick-start commands.

### Dependencies

The project and Conda environment now declare:

- NumPy;
- pandas;
- PyArrow;
- Matplotlib.

These support direct-counting estimation, Parquet storage, tabular analysis,
and plots.

## Core terminology

- **Episode:** one independent, complete simulation from initialization through
  the fixed terminal horizon. All agents are reset before the next episode.
- **Policy:** the intervention rule assigned to an episode, such as `always_A`,
  `promote_alternative`, or `no_committee`.
- **Committee size:** the number of agents whose decisions can be overridden by
  the policy.
- **No active committee:** obtained with `committee_size: 0`, `no_committee`, or
  `no_pulse`. In these cases no decision is forced.
- **Population round:** `N` pair interactions. Five agents and five population
  rounds produce 25 pair interactions.

## How to run

### 1. One-episode offline smoke test

```bash
conda run -n MA-CC python -m naming_game.cli experiment \
  --config configs/empowerment_pilot_test.yaml \
  --mock \
  --output-dir results/empowerment_pilot_test
```

Expected result: exactly one episode and 25 interaction rows. This verifies the
pipeline but cannot provide a meaningful empowerment estimate because there is
only one policy observation.

### 2. Multi-condition offline pilot

```bash
conda run -n MA-CC python -m naming_game.cli experiment \
  --config configs/empowerment_pilot.yaml \
  --mock \
  --output-dir results/empowerment_pilot
```

### 3. Real-provider pilot

Set `provider` and `model` in the chosen YAML file, configure the corresponding
credentials in `.env`, and omit `--mock`:

```bash
conda run -n MA-CC python -m naming_game.cli experiment \
  --config configs/empowerment_pilot.yaml \
  --output-dir results/empowerment_real_pilot
```

### 4. Analyze existing histories

```bash
conda run -n MA-CC python -m naming_game.cli analyze-empowerment \
  --history-dir results/empowerment_real_pilot \
  --output-dir results/empowerment_real_pilot_analysis \
  --horizons 1 3 5 10 \
  --bootstrap-resamples 1000 \
  --null-permutations 1000 \
  --seed 1
```

## Output artifacts

### Experiment output

| Artifact | Contents |
|---|---|
| `interactions.parquet` | One row per pair interaction, including outputs, memories before interaction, intervention flags, rolling states, provider, and model. |
| `episodes.parquet` | One row per episode with terminal convention, consensus, takeover, recovery, censoring, displacement, persistence, and action counts. |
| `experiment_config.json` | Exact resolved experiment configuration. |
| `.episode_shards/<fingerprint>/` | Recoverable episode checkpoints used for resume. |

### Analysis output

| Artifact | Contents |
|---|---|
| `empowerment_estimates.parquet` | Terminal and lagged MI estimates, sensitivity estimators, confidence intervals, and efficiency. |
| `episode_metrics.parquet` | Complementary episode metrics by experimental cell. |
| `null_results.parquet` | Episode-label shuffle and circular-shift null distributions. |
| `label_swap_invariance.parquet` | A/B relabeling diagnostic. |
| `no_committee_baseline.parquet` | Zero-committee near-zero empowerment check. |
| `analysis_config.json` | Exact analysis settings. |
| `plots/` | Terminal, takeover, lagged, pulse trajectory, and recovery-efficiency figures. |

## Verification completed

- Complete test suite: **53 tests passing**.
- Python compilation check: passing.
- Git whitespace/error check: passing.
- One-episode mock smoke test: **1 episode and 25 interactions written**.
- Mock Parquet experiment followed by offline analysis: passing.
- Live official OpenAI provider smoke test with `gpt-4o-mini`: HTTP 200,
  actual model recorded, and expected response returned.
- Existing Naming Game tests continue to pass.

## Known boundaries

- The one-episode smoke test validates execution only; it does not estimate
  empowerment.
- No full 17,500-episode real-provider experiment has been launched.
- Provider model availability, pricing, budgets, and rate limits must be checked
  immediately before a large run.
- Large bootstrap and null-permutation counts can make analysis expensive; use
  small values for pipeline checks and final values only for report generation.
- `OPEN_API_KEY` is compatibility-only and should be renamed to
  `OPENAI_API_KEY`.

## Recommended commit message

```text
feat: add committee empowerment experiment and analysis pipeline

- separate agent memory from temporary committee interventions
- add neutral, consensus-attack, and pulse experiment regimes
- write resumable interaction and episode histories to Parquet
- add OpenAI provider selection with explicit pre-run fallback
- estimate terminal and lagged empowerment with bootstrap and null tests
- add metrics, plots, pilot configs, documentation, and end-to-end tests
```

## Suggested commit scope

Include the implementation, configurations, tests, dependency declarations,
README changes, and documentation described above. Review the worktree before
staging: unrelated PDFs and Windows `Zone.Identifier` file changes are present
and are not part of this deliverable.
