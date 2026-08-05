# Repository overview: what's implemented, and where

This is a **map**, not a tutorial: what exists in this repository, what layer
it lives in, and — since it's the reason this doc exists — exactly what is
and isn't in place for computing information-theoretic metrics (mutual
information and friends) over agent trajectories.

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
| Experiments | Phase 9: N episodes of one resolved config, bounded concurrency, resumable by episode; grid sweeps (`grid:` config section, cartesian product of cells, one shared concurrency/budget pool) | `experiments/orchestrator.py`, `experiments/console.py`, `config/grid.py`, `planning/grid_preflight.py` |
| Control | Reserved for provider-independent intervention/control policies — currently just a docstring, **not implemented** | `control/__init__.py` |
| Analysis | Reserved for **offline trajectory and information-theoretic analysis** — currently just a docstring, **not implemented** | `analysis/__init__.py` |
| CLI | `mas-cc game run`, `mas-cc experiment ...`, `mas-cc prompt ...`, etc. | `cli/` |

`control/` and `analysis/` are real modules with real names already chosen for
them in this codebase, and both are currently empty stubs. `analysis/` in
particular is the module whose docstring already says
*"Offline trajectory and information-theoretic analysis"* — that's the
designated landing spot for MI-style metrics work on the current game, should
you want it to live at that layer rather than inside `metrics/`.

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

### 3.3 The actual gap, stated plainly

For the current game (2-symbol action alphabet, `Q`/`M`), a direct
contingency-table mutual-information estimate (entropy/MI/conditional-MI from
action counts, with the usual small-sample corrections) is likely the
simplest fit and has no existing implementation in `mas_cc` yet. The
embedding/InfoNCE path only becomes necessary if the quantity you want to
condition on is free text (a stated "reason," a message) rather than the
discrete action itself — in that case `test_info_nce.py` is the closest
existing scaffold, but it isn't connected to any real data yet. Neither
approach is currently wired into `mas_cc/metrics/`'s `Metric`/`RoundView`
objects or into the reserved-but-empty `mas_cc/analysis/` module — that
integration, and the choice between the two approaches, is the open work.

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
├── analysis/                        reserved for info-theoretic analysis — EMPTY
├── control/                         reserved for intervention policies — EMPTY
├── config/, planning/, storage/,
│   observability/, experiments/     supporting layers
└── cli/                             `mas-cc` entry points

scripts/embedding_model_tests/       smoke tests for a future embedding-based estimator
├── test_embeddings.py               load/validate 3 candidate sentence-transformer models
├── test_classification.py           linear probe on frozen embeddings
└── test_info_nce.py                 InfoNCE contrastive training (MI lower bound), toy data

docs/howto/building/building_a_game.md        mas_cc Game/prompt/agent manual (this doc's companion)
```
