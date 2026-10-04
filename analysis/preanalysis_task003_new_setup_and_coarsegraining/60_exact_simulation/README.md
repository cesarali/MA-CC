# 60_exact_simulation — the LLM-free mirror of the game

**PRELIMINARY**, 4 October 2026. Working and fast. Not yet a package; it should
move to `src/` once the modelling choices in §3 are settled.

## What this is

A simulation that reproduces the blackboard game's **mechanism exactly** and
replaces only the **decision** — which in the real game is a language model
reading a prompt, and here is the exact posterior over allocations counted from
the 14,388-world enumeration.

No natural language. No fact weights. No logistic temperature. No fitted
parameters. A fact is an id; a belief is a count of surviving worlds.

**This is not `src/santa_fe/`.** That is an abstract ±1 binary-opinion voter
model with signed fact weights, a logistic choice rule
(`beta_evidence`, `beta_social`) and an artificial reasoning-overload term. It
is a model *inspired by* the game. This is a *replica* of the game: three
allocations, the real 49 `task_003` facts by id, the real board mechanics, and
exact Bayesian arithmetic in place of the model.

| | `src/santa_fe/` | this |
|---|---|---|
| opinions | binary ±1 | **3 allocations** A0/A1/A2 |
| facts | abstract, signed ±1 weights | **the real 49 `task_003` fact ids** |
| belief | mean of weights, in [−1,1] | **exact `P(A_k \| K_i)`** over 14,388 worlds |
| decision | `sigmoid(β_e·e + β_s·s)` | **argmax of the exact posterior** |
| free parameters | `β_evidence`, `β_social`, overload | **none** |

## Files

| | |
|---|---|
| `exact_game.py` | the simulator |
| `run.py` | runs both v2 setups across arms and ρ, prints trajectories |

```bash
.venv/bin/python analysis/preanalysis_task003_new_setup_and_coarsegraining/60_exact_simulation/run.py 100
```

## 1. Mechanism reproduced from the runtime

Each read against
`src/mas_cc/games/relational_reasoning/imitation_round_feedback/`:

- **Persistence** (`apply_epistemic_persistence`): at each round boundary every
  **active** fact survives independently with probability ρ. `known` is left
  untouched.
- **Exposure → activation** (`game.py`, the loop over `exposures`): a fact cited
  in a message the focal agent reads **becomes active** — added to `known` if
  new, merely reactivated if it had decayed. *This is the whole propagation
  mechanism.* It is why the mean posterior **rises** instead of decaying to the
  prior, which the archived data confirms (silent arm at ρ = 1: e goes 0.43 →
  0.63 over 15 rounds).
- **Sequential within-round updating**: agents are updated one at a time and
  **see posts made earlier in the same round**
  (`for within_round_index in range(n_agents)` reading
  `live_messages(round_index)`). Not a detail — it is the source of the
  within-round ordering artifact that invalidated an earlier dose-response
  result.
- **Two board buffers**: the controller at dawn senses the *previous* round's
  board (`live_messages(round_index - 1)`); agents read the *current* round's as
  it grows. With `message_lifetime_rounds: 1` last round's board is gone by the
  time the agents act.
- **Board sampling** `full` or `uniform` of `social_group_size`, with
  `exclude_self_authored`.
- **Controller**: dawn only, soft-target gate, per-round budget, and the exact
  lexicographic ranking of `select_truthful_reports`.

## 2. The headline result: `deterministic` authoring cannot steer

`select_truthful_reports` ranks the controller's pool on

```
(not cooldown_eligible, reuse_count, live_count, -base_score, tie_hash, fact_id)
```

sorted ascending. `reuse_count` is the **second** key and `base_score` only the
**fourth**, so the ranking is a **least-recently-used rotation** in which the
target merely breaks ties within a sweep. Measured, 30 episodes per cell, same
pool, only the target differing:

| setup | ρ | controller rule | e (truth arm) | e (false arm) | **gap** |
|---|---|---|---:|---:|---:|
| symmetric-v2 | 0.75 | `lru` *(= `deterministic`)* | 0.601 | 0.601 | **+0.000** |
| symmetric-v2 | 0.75 | `greedy_posterior` *(oracle)* | 0.750 | 0.067 | **+0.682** |
| symmetric-v2 | 1.0 | `lru` | 0.975 | 0.975 | **+0.000** |
| symmetric-v2 | 1.0 | `greedy_posterior` | 0.980 | 0.486 | **+0.494** |
| nosolution-v2 | 0.75 | `lru` | 0.512 | 0.512 | **+0.000** |
| nosolution-v2 | 0.75 | `greedy_posterior` | 0.838 | 0.090 | **+0.748** |
| nosolution-v2 | 1.0 | `lru` | 0.934 | 0.934 | **+0.000** |
| nosolution-v2 | 1.0 | `greedy_posterior` | 0.985 | 0.125 | **+0.860** |

**Exactly zero in every cell** for the LRU rule, and the zero is structural, not
statistical: with a neutral pool `base_score` is 0 for every fact, so the
ranking never consults the target at all.

Two further readings:

- **nosolution-v2 is the better instrument.** Its oracle gaps (+0.75, +0.86)
  exceed symmetric-v2's (+0.68, +0.49), because an undecided population leaves
  more room to move. That is what it was designed for.
- **A targeted controller can defeat a population that collectively owns a
  proof.** In symmetric-v2 at ρ = 0.75 the false oracle drives the mean
  posterior to **0.067** and the A0 vote share to **0.003**, against a
  population whose pooled evidence proves A0. That is the striking claim the
  setup exists to test.

## 3. The modelling choices — these are where a model is replaced, and they matter

Marked `CHOICE` in the code and collected in `AgentPolicy` / `ControllerPolicy`.

| choice | default | alternatives |
|---|---|---|
| vote rule | `bayes` — argmax of the exact posterior, no noise | `softmax` with one β, for sensitivity only |
| posting rule | `vote_aligned_novel` — uniform among own active facts that support the vote, preferring facts not already live | `vote_aligned`, `random_active`, `argmax_aligned` |
| controller rule | `lru` — reproduces the runtime | `greedy_posterior` — the oracle |

**A flaw this surfaced, worth recording.** The first version had agents post the
fact that *most* raises their voted allocation — an argmax. The result was
degenerate: **all 24 agents converge on the same globally strongest fact**, the
board carries exactly one distinct fact forever, and knowledge stops spreading
at |K| = 2. A language model would never do this, because it does not compute a
global argmax. Replacing it with a uniform draw among supporting own facts
restored spreading (|K|: 1 → 6.7 silent, 18.9 controlled at ρ = 1).

`argmax_aligned` is kept so that failure stays reproducible.

**An agent reports from its own active knowledge.** The real game also permits
citing an *observed* fact (`report_citation_scope: active_or_observed`), but an
agent that cites what is already on the board adds no information, so
own-knowledge is the faithful default and is what makes facts propagate.

## 4. Speed — measured

| | |
|---|---|
| per episode (24 agents × 30 rounds) | **~7.5 ms** |
| 12 cells × 20 episodes | **1.8 s** |
| **12 cells × 100 episodes** | **≈ 5 s** |
| the same grid with an LLM | **≈ 288 hours** |

Locally, free, no cluster. The posterior is memoised on the frozenset of active
facts, which is what makes it this fast.

## 5. What it is for

- **Validating the estimators against known ground truth.** The simulator's own
  transition kernel is available, so the true path KL and mutual information can
  be computed and compared with what the 4-state coarse-graining recovers. This
  is the single most valuable use and is what the earlier critic-based
  estimators never had.
- **Sweeping the design space** — pool composition, overlap, assignments, ρ, b —
  for free before spending LLM budget.
- **Checking the mean-field theory** of `../50_next_experiment/research_directions_2026-10-04.md` §3.
- **Exact power analysis** instead of extrapolation.

**Not** a substitute for the LLM experiments. A Bayes-exact agent cannot be
moved by phrasing, cannot be wrong in correlated ways, and has no notion of
trust in a source.

## 6. Known gaps

- No social-influence term. The real agent sees peers' **votes** as well as
  their facts; here only facts enter the belief. Adding a vote term would
  reintroduce a free parameter, so it is deliberately absent — but it means the
  silent arm here has no conformity pressure.
- Not validated against the archived LLM trajectories beyond the qualitative
  rise in e. A quantitative comparison is the obvious next step.
- `uniform` board sampling is implemented but untested.
- Agents never *stop* believing a fact they absorbed: `known` only grows, as in
  the runtime.
