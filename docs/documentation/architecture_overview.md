# Repository overview: what's implemented, and where

This is a **map**, not a tutorial: what exists in this repository, what layer
it lives in, and — since it's the reason this doc exists — exactly what is
and isn't in place for computing information-theoretic metrics (mutual
information and friends) over agent trajectories. Short version: the
direct-counting MI/empowerment path (Section 3.3) is implemented and wired to
the CLI; the continuous, embedding/InfoNCE-based path (Section 3.2) is not.

For hands-on material, see:
- [`docs/howto/building/building_a_game.md`](howto/building/building_a_game.md) — manual for the `Game`/prompt/agent
  abstractions, using the real naming-convention game.
- [`docs/howto/building/running_an_experiment.md`](howto/building/running_an_experiment.md) — manual for turning a
  resolved game config into a priced, concurrent, resumable experiment (`mas-cc experiment preflight`/`run`).
- [`notebooks/tutorial_build_a_game.ipynb`](../notebooks/tutorial_build_a_game.ipynb) — builds a toy game live.

---

## 1. `mas_cc` layers

| Layer | What it is | Where |
| --- | --- | --- |
| LLM providers | Normalized access to a model (mock, University proxy, OpenAI, local Gemma) | `src/mas_cc/llm_providers/` |
| Prompts | Composable, versioned, value-bearing message construction (`PromptBlock`, `FullPrompt`) | `src/mas_cc/prompts/`, `games/naming_convention/prompts.py` |
| Agents | Identity + accumulated private memory, generic across every game | `games/protocols.py` (`AgentState`) |
| `Game` contract | `abc.ABC` every game must fully implement: `initialize`, `select_participants`, `construct_observations`, `build_decision_requests`, `parse_action`, `validate_action`, `apply_transition`, `detect_termination`, `call_plan` | `games/protocols.py`, `games/naming_convention/game.py` |
| Decision loop | Shared ask/validate/retry/log procedure, used by every game | `runtime/loop_runtime.py` |
| Metrics | Per-round / per-episode scientific quantities computed from what happened | `metrics/`, `games/naming_convention/metrics.py` |
| Config | One resolved, versioned description of a run | `config/`, `configs/runs/*.yaml` |
| Planning | Pre-run token/cost estimation and budget preflight — refuses to launch a run it can't account for | `planning/` |
| Observability | `RunRecorder` — audit trail + optional Comet logging | `observability/` |
| Storage | Local artifact layout (`results/<game>/<experiment>/<run_id>/...`), checkpoints | `storage/` |
| Experiments | Phase 9: N episodes of one resolved config, bounded concurrency, resumable by episode; grid sweeps (`grid:` config section, cartesian product of cells, one shared concurrency/budget pool); one run-level Comet monitor (progress/budget only, episode-level Comet stays off) | `experiments/orchestrator.py`, `experiments/console.py`, `experiments/comet_monitor.py`, `config/grid.py`, `planning/grid_preflight.py` |
| Control | Provider-independent action-forcing: a `Control.override(...)` hook the decision loop checks before asking the LLM for an action. `ForcedActionControl` forces a fixed agent set to a fixed value, optionally expiring after an interaction index; `NoneControl` is the no-op default. Selected via config (`control.mechanism`/`control.options`, mirrors `GameConfig`'s `type`/`options` idiom) and, since it's ordinary config, is itself a sweepable `grid:` axis — this is how an empowerment grid's "condition" gets created. | `control/` (`protocols.py`, `forced_action.py`, `registry.py`) |
| Analysis | **Implemented**: offline, post-hoc information-theoretic analysis over a completed grid directory. Direct-counting MI/conditional-MI estimators (Jeffreys/unsmoothed/Miller-Madow corrected), an end-to-end empowerment pipeline (terminal MI, lagged conditional MI, bootstrap CIs, permutation/circular-shift nulls, label-swap invariance check), a reader that assembles tidy tables from grid output, and shuffle/circular-shift/label-swap surrogates for the null models. Ported near-verbatim from the legacy `naming_game` committee-empowerment pipeline. Wired to `mas-cc analysis empowerment --grid-dir <dir>`. | `analysis/` (`estimators.py`, `pipeline.py`, `reader.py`, `surrogates.py`) |
| CLI | `mas-cc game run`, `mas-cc experiment ...`, `mas-cc prompt ...`, `mas-cc analysis empowerment`, etc. | `cli/` |

---

## 2. The naming-convention game, briefly

Two agents are sampled each interaction; each independently chooses one of
two actions, `Q` or `M`, based only on its own private history (its own past
partners, own actions, own payoffs — never another agent's). Matching choices
score `+1`-scale payoffs, mismatches score negative payoffs; over many
interactions a population-wide convention (everyone converging on `Q` or on
`M`) tends to emerge. A "round" of interactions produces one new `GameState`
via `apply_transition`. This is the standard object of study for the
population-share / consensus metrics already in place (Section 3) and the
natural first target for any new MI metric.

For the full walkthrough (agents, observations, prompts, the decision loop),
see the building manual linked above — this section is deliberately just
enough to make Section 3 legible.

---

## 3. Metrics: what's actually in place

### 3.1 `mas_cc/metrics/` — current, generic, no MI yet

`Metric` / `StreamingMetric` (per round) / `FinalMetric` (per episode) are
minimal ABCs (`metrics/base.py`) — deliberately no registry, no versioned
extractor. Most games reduce to "each round, each agent has a current value,"
so metrics are written once against a generic `RoundView`
(`metrics/generic.py`):

```python
RoundView(
    agent_values: Mapping[AgentId, Any],
    agent_targets: Mapping[AgentId, Any] | None = None,
    options: tuple[str, ...] = (),
)
```

The shelf of ready-made metrics: `ActionSharePerOption`, `AgentCurrentValue`,
`DominantValueShare`, `FirstConsensusTime`, `AgentAbsoluteError`,
`MeanAbsoluteError`. `games/naming_convention/metrics.py::to_round_view` is
the adapter that reads each agent's `committed_action` and the game's option
set into a `RoundView`; `build_metrics()` wires up
`population_action_share_per_option`, `agent_current_action`,
`dominant_action_share`, `first_consensus_time_by_action_share`.

A metric's `scope` decides what its per-round keys mean and which column they
land in: `agent` (keyed by `AgentId` → `agent_id`), `population` (keyed by
`None`), or `option` (keyed by the option label → `series`). An `option`-scope
metric is one metric carrying a family of curves, which is why a per-option
share is a single named quantity rather than one metric per option.
`GameSpec.game_family` (`choice` | `numeric`) is the game-side counterpart:
metrics declaring `requires_game_family` are validated against it in
`games/registry.py::game_metrics`.

**None of this computes mutual information, entropy, or any other
information-theoretic quantity.** It answers "what fraction of the population
plays Q" and "when did we reach consensus," not "how much does knowing
agent A's action tell you about agent B's."

### 3.2 `scripts/embedding_model_tests/` — prep for a continuous, embedding-based MI approach

Per that folder's own `README.md`, these are explicitly **smoke tests for
model-loading pieces needed by a future empowerment estimator** — they do not
implement or train that estimator, and they never fine-tune the backbone.
Three scripts:

- `test_embeddings.py` — downloads/loads any of three candidate Hugging Face
  checkpoints (`intfloat/e5-small-v2`, `intfloat/e5-base-v2`,
  `sentence-transformers/all-MiniLM-L6-v2`), embeds example agent-trajectory
  text, checks normalization and cosine-similarity sanity.
- `test_classification.py` — fits a scikit-learn logistic regression on
  frozen E5 embeddings over a small synthetic trajectory dataset (a probe:
  "is anything linearly recoverable from this embedding space").
- `test_info_nce.py` — the one directly relevant to MI: freezes the E5
  encoder, trains two small projection MLPs (`phi` over "context" sentences,
  `psi` over "outcome" sentences) with a **symmetric InfoNCE / CLIP-style
  contrastive loss** over 8 hardcoded matched `(context, outcome)` sentence
  pairs (`MATCHED_PAIRS`). InfoNCE is the standard variational lower-bound
  estimator of mutual information used when you have continuous embeddings
  instead of a small discrete alphabet.

None of these three scripts read a real `mas_cc` trajectory log yet —
`MATCHED_PAIRS` is a hand-written illustrative dataset, and there's no wiring
from `RunRecorder`/Parquet output into an encoder or an InfoNCE training loop.

### 3.3 The direct-counting path: implemented, in `mas_cc/analysis/`

For the current game (2-symbol action alphabet, `Q`/`M`), the simplest fit —
a direct contingency-table mutual-information estimate (entropy/MI/
conditional-MI from action counts, with the usual small-sample corrections)
— is now implemented as `mas_cc/analysis/`, not as a `Metric`. It deliberately
sits outside `metrics/`'s `Metric`/`RoundView` objects: MI is computed
post-hoc over a *completed* grid directory (bootstrap-resampled over
episodes, compared against permutation nulls), which doesn't fit a
`Metric`'s one-instance-shared-across-every-episode streaming contract. The
pipeline is:

1. **Condition** — a `control:` mechanism (`ForcedActionControl` or a future
   one) swept as a `grid:` axis, so each cell is a different forced-action
   condition. This is the layer's actual reason for existing: without it
   there is nothing to condition the MI estimate on.
2. **Outcome** — `analysis/reader.py::read_grid` reads each cell's episodes
   straight from what a grid run already writes (`cells/<cell>/overrides.json`
   for the condition label, each episode's `metrics/streaming.csv` for the
   already-existing `population_action_share_per_option` StreamingMetric) —
   no new `Metric` classes or `RunRecorder` changes needed.
3. **Estimate** — `analysis/estimators.py` (pure NumPy, ported near-verbatim
   from the legacy `naming_game` committee-empowerment pipeline) computes
   `mutual_information`/`conditional_mutual_information` from contingency
   tables, each with Jeffreys, unsmoothed, and Miller-Madow corrections.
   `analysis/pipeline.py::analyze_grid` runs the full thing end to end:
   terminal-outcome MI, lagged conditional MI at configurable horizons,
   bootstrap confidence intervals, permutation/circular-shift nulls
   (`analysis/surrogates.py`), and a label-swap invariance sanity check —
   and writes CSVs under an `analysis/` output directory.
4. **Entry point** — `mas-cc analysis empowerment --grid-dir <completed grid
   dir>` (`cli/analysis.py`), driven end-to-end by
   `scripts/Potsdam/SLURM/empowerment_grid.job` on the cluster.

The embedding/InfoNCE path (Section 3.2) remains unimplemented and is only
needed if the quantity to condition on becomes free text (a stated "reason,"
a message) rather than the discrete action itself — `test_info_nce.py` is
still the closest scaffold for that case, but it isn't connected to any real
data and nothing here currently calls it.

---

## 4. Quick file map

```
src/mas_cc/                          current package
├── llm_providers/                   provider layer
├── prompts/                         PromptBlock / FullPrompt
├── games/
│   ├── protocols.py                 Game ABC, AgentState, Observation, Transition
│   ├── naming_convention/           the real, current naming game (Q/M)
│   └── toy_coordination/            a second, smaller real game
├── runtime/loop_runtime.py          shared decision loop
├── metrics/                         generic Metric/RoundView shelf — no MI yet
├── analysis/                        offline MI/empowerment analysis over a completed grid
│   ├── estimators.py                contingency-table MI / conditional-MI (Jeffreys, Miller-Madow)
│   ├── pipeline.py                  analyze_grid: terminal + lagged MI, bootstrap, nulls, invariance
│   ├── reader.py                    reads a grid dir's on-disk output into tidy episode/round tables
│   └── surrogates.py                shuffle / circular-shift / label-swap null-model transforms
├── control/                         provider-independent action-forcing (the MI "condition" source)
│   ├── protocols.py                 Control ABC — override() hook checked by the decision loop
│   ├── forced_action.py             ForcedActionControl, NoneControl
│   └── registry.py                  config `control.mechanism` → Control factory lookup
├── config/, planning/, storage/,
│   observability/, experiments/     supporting layers
└── cli/                             `mas-cc` entry points (incl. `analysis empowerment`)

scripts/embedding_model_tests/       smoke tests for a future embedding-based estimator
├── test_embeddings.py               load/validate 3 candidate sentence-transformer models
├── test_classification.py           linear probe on frozen embeddings
└── test_info_nce.py                 InfoNCE contrastive training (MI lower bound), toy data

docs/howto/building/building_a_game.md        mas_cc Game/prompt/agent manual (this doc's companion)
```
