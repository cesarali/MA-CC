# Running the synthetic games

*Implements all three games of `docs/tdd/architecture/05082026_synthetic_games_plan.md`
and the family-2 closed forms of `docs/tdd/architecture/05082026_empowerment_ground_truth.md`.*

## What these are

Games whose answers we derived ourselves, so that every information-theoretic
quantity the system reports can be compared against a closed form. The agents
are not LLMs — they are lookup tables plus coins, with dynamics we specified.

**They are not tests.** Nothing here fails a build. They are a rehearsal of the
full workflow with the answer key in hand: same `Game` contract, same decision
loop, same recorder, same artifacts. If the workflow produces the right answer
end to end on a problem where we can check it, we have grounds to trust it on a
problem where we cannot.

Nothing here says anything about mutual information between real LLM agents. It
is entirely about whether the machinery behaves as it should.

## Game 1 — Bernoulli

No dynamics, no memory. Each round nature draws a latent bit
`Z_t ~ Bern(1/2)`, and agent *i* reports

```
A_i,t = Z_t XOR B_i,t,     B_i,t ~ Bern(eps_i)   private, independent
```

Marginals are uniform by construction, so for any pair

```
q_ij = eps_i (1 - eps_j) + eps_j (1 - eps_i)
I(A_i ; A_j) = 1 - H(q_ij)     bits, exactly
```

Two anchors carry the weight: `eps = 0.5` gives exactly **0 bits**, `eps = 0`
gives exactly **1 bit**.

## First: one episode, or many? (the thing that trips people up)

There is no separate "MI run". **One episode already gives you one mutual
information estimate.** The confusion comes from two different MIs in this
codebase that count different things:

| | sampling unit | one estimate needs | who computes it |
|---|---|---|---|
| **Within-episode MI** — `I(A_i; A_j)`, what Game 1 measures | a **round** | one episode of T rounds = T paired samples | `mas-cc synthetic episode` |
| **Across-episode MI** — `I(swept condition; terminal outcome)` | an **episode** | many episodes, one row each | `mas-cc analysis empowerment --grid-dir` |

Game 1's agents produce a fresh independent pair of actions every round, so a
200-round episode is 200 samples and the MI falls straight out of it. You do not
need repetitions for that.

What extra seeds buy you is the **distribution** of that estimate — the null,
the error bars, the bias. That is `sweep`, and it deliberately does *not* write
per-episode artifacts, because 200 recorded episodes to produce one calibration
curve would be pure cost.

So, in practice:

- **"I want to look at one run's metrics, artifacts, plots, Comet"** →
  `synthetic episode`. One episode, full artifacts. This is the mundane one.
- **"I want error bars on the estimator"** → `synthetic sweep`. Hundreds of
  seeds, no artifacts, seconds.
- **"I want several full runs on disk"** → `synthetic parity --seeds N`. It runs
  N complete fidelity episodes, each with its own artifact directory under
  `episodes/seed-*/`, and checks trajectory agreement for free.

## The five commands

```bash
mas-cc synthetic truth   --config configs/runs/synthetic_bernoulli_fidelity.yaml
mas-cc synthetic sweep   --config configs/runs/synthetic_bernoulli_null.yaml --seeds 200
mas-cc synthetic episode --config configs/runs/synthetic_bernoulli_fidelity.yaml
mas-cc synthetic parity  --config configs/runs/synthetic_bernoulli_fidelity.yaml --seeds 5
mas-cc synthetic empowerment --config configs/runs/synthetic_controlled_markov.yaml \
  --condition control_value --values 0 1 --repetitions 50 --horizons 1 5 --macrostate-bins 4
```

| Command | Mode | What it answers |
|---|---|---|
| `truth` | — | What the closed form says, for this exact config, without running anything |
| `empowerment` | — | Exact `I(C;O)` and `I(C;S_t+h\|S_t)` for a **sweep**, without running it |
| `sweep` | speed | The null distribution and the calibration curve |
| `episode` | fidelity | One episode through prompts, provider, parser, validator, recorder |
| `parity` | both | Do the two modes produce the *identical* trajectory |

`parity` exits non-zero when they disagree — that is a finding, not a warning.

## The two modes, and why the pair matters

**Fidelity mode** runs the whole pipeline: prompts constructed and compiled,
actions crossing the provider boundary as text, `parse_action` /
`validate_action`, the recorder writing the metrics tree. Slow.

**Speed mode** (`game.simulate()`) is the same dynamics vectorized, with no
prompts and no recorder. 500 seeds × 1000 rounds × 8 agents takes ~0.02 s.

They share one thing: **the coin tape**. Every draw an episode needs is made up
front from the episode seed into numpy arrays (`games/synthetic/noise.py`).
Fidelity mode reads it one cell at a time through the pipeline; speed mode
reads the same arrays in one vectorized pass. So the same seed *must* produce
the identical action sequence, and `parity` demands exactly that — no error
bars to argue about. If it fails, the pipeline has a bug, which is the entire
reason these games exist.

## The agent reads the prompt

The synthetic agent (`games/synthetic/provider.py`) is a normal `LLMProvider`.
It receives only a `CompletionRequest` and recovers its decision input by
finding this line in the compiled messages:

```
SYNTHETIC-OBSERVATION-V1 {"actions":["Q","M"],"flip":false,"policy":"bernoulli_xor_v1","round":1,"signal":"Q"}
```

This is deliberate. A prompt that fails to carry this round's observation, or
carries last round's, makes the agent decode the wrong thing and the measured
MI miss its closed form. An agent handed the observation through a side channel
would have *exercised* prompt construction; this one **checks** it.

## Game 2 — Markov

Real dynamics, real coupling, still closed-form. Agent *i* observes one partner
through a coupling graph, pushes it through a 2×2 kernel, then flips with
probability `eps_i`. The microstate is the full action profile, so with `N ≤ 10`
the chain is at most 1024 states and everything is exact linear algebra.

**The coupling graph is more diagnostic than the noise level**, because it lets
us design *structural zeros*. Verified exactly:

| Config | Claim | Measured |
|---|---|---|
| `coupling: chain` (1→2→3) | `I(A₃ᵗ⁺¹; A₁ᵗ \| A₂ᵗ) = 0` | `0.000e+00` |
| `coupling: chain` | `0 < I(A₁;A₃) < I(A₁;A₂)` | `0 < 0.124 < 0.320` |
| `coupling: [0,0,2]` | `TE(1→2) > 0`, `TE(2→1) = 0` | `0.304`, `0.000e+00` |
| `coupling: self` | every cross-agent TE is zero | max `4.4e-16` |

The reverse-direction zero is the one that catches direction errors and
off-by-one round alignment — the classic silent failure.

Ground truth is the **episode-pooled** value, not the stationary one, because
that is what a pooled-count estimator targets: it builds one table from every
round. Pooling counts is averaging the joint distributions and *then* taking
mutual information, which is not the average of per-round values. The
stationary value ships alongside as `stationary_mutual_information`; the two
coincide only after the chain has mixed.

`macrostate_is_lumpable` is reported per config. When it is 0 — which asymmetric
coupling and heterogeneous noise both cause — computing on the lumped
macrostate chain gives a *different and wrong* answer, and only microstate
propagation is correct. Everything here propagates in 2^N and lumps last.

## Game 3 — Controlled Markov (the positive control)

Game 2 plus an exogenous input `u`, held constant for the episode, pushing a
subset of agents toward one action with a given strength.

This game exists because **sweeping a config parameter is not steering**. On
Game 1 the terminal empowerment is exactly zero for any sweep; on symmetric
Game 2 it is zero again. Those zeros are correct and diagnostic, but they are
all nulls. `u` is the one input that actually moves the population, and this is
the only place a *nonzero* empowerment has a known target:

```
Game 2, sweeping epsilon over [0.05, 0.15, 0.3]:
  terminal I(C;O) = 0.000000 bits      <- config sweeps are not a control channel
  I(C;S)          = 0.000000 bits

Game 3, sweeping the control input over [0, 1]:
  terminal I(C;O) = 0.701594 bits      <- genuine steering
  I(C;S)          = 0.693796 bits
```

Control MI is monotone in strength and zero at strength 0 exactly. Two numbers
ship, answering different questions: **design MI** under the grid you actually
swept (what the estimator must reproduce) and **capacity** — `max_p(u) I(U;S)`
by Blahut–Arimoto — the ceiling on control authority regardless of how you
sampled it. A gap between them means the grid was chosen badly, not that the
controller is weak. `control_microstate_capacity` bounds the macrostate one;
the difference is what coarse-graining discards.

## Empowerment — the family-2 answer key

The system computes two families of mutual information. Family 1
(within-episode `I(A_i;A_j)`) had an answer key. Family 2 — `I(C;O)` and
`I(C;S_{t+h}|S_t)`, what `analysis/pipeline.py` estimates — did not.

The objection was that empowerment depends on which conditions you swept and
how many episodes per cell, so it is a design choice rather than a fixed fact.
True, and it does not follow. **The sweep grid *is* the channel's input
distribution `p(c)`** — equal repetitions make it uniform, unequal ones weight
it — and it is a distribution we chose and know exactly. Condition on the
design, which we always can because we made it, and the quantity is determined.

The real obstacle was architectural: a game instance does not know what it is
being swept against, so this cannot live on the game. It lives at
[`empowerment.py`](../../../src/mas_cc/games/synthetic/empowerment.py), taking
the resolved grid and analysis settings:

```bash
mas-cc synthetic empowerment --config configs/runs/synthetic_bernoulli_fidelity.yaml \
  --condition epsilon --values 0.05 0.15 0.3 0.45 \
  --repetitions 50 --horizons 1 2 5 --macrostate-bins 4
```

### What was verified

| Claim | Result |
|---|---|
| Game 1 terminal `I(C;O)` is **exactly 0** for odd N, any sweep | `0.000e+00` at N = 3,5,7,9 |
| Even N reproduces the tie-break artifact `H(p̄) − Σp(c)H(½+½p_tie)` | matches to 1e-12 at N = 4,6,8 |
| Game 1 lagged CMI is **flat in h** | spread `8.9e-16` over h = 1…20 |
| `I(C;S_{t+h}\|S_t) = I(C;S) − I(S_t;S_{t+h})` | holds to 1e-10 |

The tie-break one is worth dwelling on. `reader.py::_dominant_action` resolves
an exact tie with `idxmax`, so every tie lands on the first declared action.
With **even N** ties have positive probability, the outcome marginal stops being
symmetric, and the pipeline reports **empowerment produced entirely by a
tie-break convention**. Run odd N to confirm the exact zero; run even N to
confirm the pipeline produces *precisely* this spurious value and no more. On
the shipped `synthetic_bernoulli_fidelity.yaml` (N = 4) it is 0.0144 bits — with
a plug-in bias of 0.0108 sitting right next to it.

### Two traps the tool refuses to let you walk into

**Sweeping `population_size` against a raw macrostate is refused outright.** The
action share's support is `{0, 1/N, …, 1}` — a different alphabet per condition
— so observing 0.15 identifies N = 20 with certainty, and `I(C;S)` collapses
onto `H(C)`: maximal, and entirely an artifact of alphabet support. Pass
`--macrostate-bins` to put every condition on a common grid. This is a hard
error, not a warning, because it invalidates every population-size result
silently.

**Plug-in bias is reported next to every value, and often exceeds it.** Bias is
`dof/(2·N_eff·ln2)` with `N_eff` the number of **episodes**, not rounds —
within-episode pairs are correlated and buy no independent samples. For the
lagged CMI `dof = |M|(|C|−1)(|M'|−1)`, so at N = 10 unbinned, 2 conditions and
100 episodes that is 0.79 bits of pure bias, enough to swamp any real effect.
The CLI flags it inline:

```
    h=1   0.092230 bits   (plug-in bias 0.1154)  <-- bias exceeds the signal
```

Binning to 3–5 levels cuts it by more than 5×. It is almost certainly not
optional, and the closed form is what lets you tell a real signal from table
inflation.

## Checking the ordinary metrics (not just MI)

The synthetic game declares the **same choice-metric set as the real naming
convention game**, so an episode exercises the whole recorder — `streaming.csv`,
`final.csv`, the binned trajectory tables, the plots and the Comet export —
exactly as a real run does. And because we wrote the dynamics, most of those
metrics also have closed forms, so `synthetic episode` prints them side by side:

```
  metrics vs. ground truth:
    dominant_action_share                          0.6567   expected 0.6562   ok
    population_action_share_per_option [M]         0.5044   expected 0.5000   ok
    rolling_coordination_rate                      0.0419   expected 0.0312   ok   (second half only)
    consensus_action_by_success_rate                 None   expected None     ok
    first_consensus_time_by_action_share               17   expected 32.0000  --   (geometric, one draw)
```

`--` means **deliberately not judged**, never a soft pass. The same table is
written to `metrics/metric_check.csv`.

Tolerances are derived, not picked: these are proportions averaged over *n*
rounds, so the band is `4 * sqrt(p(1-p)/n)`. It judges whether the recorder
wrote what the game did — estimator bias is the MI table's job, at a precision
this deliberately does not attempt.

### Three configs, three predictable answers

| Config | ε | What the metrics must say |
|---|---|---|
| `synthetic_bernoulli_metrics.yaml` | 0.0 | Unanimous **every** round; both success-rate consensus metrics fire |
| `synthetic_bernoulli_never_converges.yaml` | 0.5 | Cannot converge; both must stay `None` |
| `synthetic_bernoulli_degenerate.yaml` | 0, latent pinned | Zero entropy; MI must be `0`, not NaN or 1 |

Two findings worth knowing, both surfaced by these configs:

**The two consensus metrics genuinely disagree, and should.** At ε=0.5 with 6
agents, `first_consensus_time_by_action_share` fires at round 17 while
`first_consensus_time_by_success_rate` correctly reports `None`. That is not a
bug. The first asks "do enough agents hold the same value *right now*" with no
persistence requirement, so a single lucky round satisfies it — and unanimity
happens by chance every ~1/P(unanimity) = 32 rounds. **It is not a convergence
detector for small populations.** The rolling-window criterion, which needs 95%
of a 25-round window, is.

**ε=0 is perfectly coordinated but never settles.** Every round is unanimous
(`rolling_coordination_rate` = 1.0) while the chosen word keeps flipping,
because the *signal* is still a fair coin. This is the concrete case behind
"success rate alone says the population agreed but not on what". A metric
reporting a stable convention here is measuring the wrong thing.

**Jeffreys smoothing invents information.** On the degenerate config the
unsmoothed and Miller–Madow estimators both return exactly 0, while Jeffreys
returns 0.0151 bits — half a count in three structurally empty cells. Not a
bug; the cost of smoothing, made legible.

### Comet

`configs/runs/synthetic_bernoulli_comet.yaml` is the **only** synthetic config
with Comet enabled, kept separate so uploading is something you opt into by
naming that file, never something a sweep inherits.

> ⚠ **This repo has a `COMET_API_KEY` in `.env`, and the recorder reads `.env`
> from the current directory.** Running that config from the repo root *will*
> create a real experiment in the `mas-cc-synthetic` project. Every other
> synthetic config has `comet: false` and cannot upload.

I verified the wiring without uploading, by running it from a directory with no
`.env` and no key — it completes and reports `Comet: unavailable (COMET_API_KEY
is not set)`. That is the safe way to check the plumbing first:

```bash
cd "$(mktemp -d)" && env -u COMET_API_KEY \
  mas-cc synthetic episode --config /abs/path/configs/runs/synthetic_bernoulli_comet.yaml
```

Routing is per-metric, next to each metric's name rather than in a separate
list:

```yaml
metrics:
  available:
    dominant_action_share: {comet: true}
    population_action_share_per_option: {comet: true}
    rolling_coordination_rate: {comet: true}
```

Option-scope metrics arrive as one suffixed series each
(`population_action_share_per_option_Q`, `..._M`). Agent-scope metrics are never
exported — one series per agent is noise. What reaches Comet is aggregate
metrics only; prompts, blocks, messages and responses never leave the machine.

## Ground truth is an artifact, not a comment

`ground_truth()` computes the closed form from **the same resolved config object
that ran**, and `synthetic episode` writes it to `ground_truth.json` before the
first decision. Pull a run off the cluster and the answer is already in the
directory, next to the estimate.

This kills the phantom-bug failure mode: change `epsilon` in the YAML, forget to
update an expected number somewhere else, lose a day debugging an estimator that
was fine. The discrepancy is a column (`estimate`, `truth`, `gap`), not an
investigation.

## Measured results

Numbers below are from this repository, 8 agents, 500 rounds, 200 seeds.

### The significance floor

At `eps = 0.5`, where the true MI is exactly zero:

| estimator | mean | p95 | max |
|---|---|---|---|
| unsmoothed (plug-in) | +0.00144 | 0.00556 | 0.02032 |
| jeffreys | +0.00143 | 0.00551 | 0.02016 |
| miller_madow | +0.00000 | 0.00411 | 0.01888 |

The plug-in mean matches the analytic first-order bias
`(|X|-1)(|Y|-1) / (2 N ln 2) = 1/(2 × 500 × ln 2) = 0.001443` to three
significant figures, and Miller–Madow removes it to five decimal places.

**A plug-in MI below ~0.0056 bits, at this population and round count, is
indistinguishable from zero.** That floor scales as `1/N` in the number of
paired observations, so re-run `sweep` at the shape of the run you actually
care about rather than reusing this number.

### The calibration curve

`gap_unsmoothed_mean` is a **constant +0.00145 bits across the whole sweep** —
constant offset is bias, scatter would be noise. Jeffreys smoothing is the
outlier: it is fine at low MI but biased *downward* by −0.022 bits at 1 bit,
visible in the residual panel of `metrics/plots/calibration.png`.

### Where the time goes

The plan asked for this measurement early, because with no API calls our own
framework is the bottleneck. Fidelity mode, 8 agents:

| rounds | checkpoints | ms/round |
|---|---|---|
| 500 | on | 14.2 |
| 1000 | on | 21.3 |
| 2000 | on | 36.4 |
| 1000 | off | 13.6 |

Two readings, both worth having:

1. **Fidelity mode is the plan's pessimistic branch** (~14–36 ms/round, not
   ~1 ms). So fidelity runs ~20 seeds, not 500 — which is exactly why speed
   mode and the `parity` check exist. Speed mode is free by comparison.
2. **Per-round checkpointing is quadratic in episode length.** Cost per round
   nearly doubles as rounds double, and disappears with `storage.checkpoints:
   false`. `RunRecorder.record_interaction` re-serializes the whole game state
   every round, and the state grows every round. This is **not specific to the
   synthetic games** — `naming_convention` accumulates both per-agent `memory`
   and `evaluator_history` in exactly the same way, so a long real run pays the
   same cost. Left as-is here: changing checkpoint semantics is a design
   decision, not a cleanup.

## Configs

| File | Purpose |
|---|---|
| `configs/components/games/synthetic_bernoulli.yaml` | 8 agents, 500 rounds, `epsilon: 0.5` |
| `configs/components/prompts/synthetic_agent_v1.yaml` | The `synthetic_agent_decision` family |
| `configs/components/llm_providers/synthetic_agent.yaml` | The lookup-table "provider" |
| `configs/runs/synthetic_bernoulli_null.yaml` | The null anchor, for `sweep` |
| `configs/runs/synthetic_bernoulli_fidelity.yaml` | `eps = 0.15`, for `episode` and `parity` |
| `configs/runs/synthetic_bernoulli_metrics.yaml` | `eps = 0`: always unanimous, never settled |
| `configs/runs/synthetic_bernoulli_never_converges.yaml` | `eps = 0.5`: consensus metrics must stay `None` |
| `configs/runs/synthetic_bernoulli_degenerate.yaml` | Zero entropy; MI is a genuine 0/0 |
| `configs/runs/synthetic_bernoulli_comet.yaml` | **The only one that uploads** |
| `configs/runs/synthetic_markov.yaml` | Game 2: 6 agents on a ring |
| `configs/runs/synthetic_controlled_markov.yaml` | Game 3: the positive control |

Set `epsilons: [...]` instead of `epsilon:` for a per-agent asymmetric config;
the ground truth then differs per pair and the comparison table joins on the
pair, not on one shared number.

## Adding Game 2 or 3

`SyntheticGame` (`games/synthetic/protocols.py`) is an ABC with two abstract
methods beyond the `Game` contract — `ground_truth()` and `simulate()` — so a
new synthetic game cannot ship without an answer key and a speed-mode twin.

Concretely, a new game needs:

1. `games/synthetic/<name>/game.py` implementing both, drawing its coins from
   `noise.episode_generator` under new stream names (new names leave every
   existing episode's draws bit-identical).
2. A binder in `games/synthetic/prompts.py` — the prompt *family* is shared;
   what changes is the decoding rule in the `protocol` block and the payload in
   the `observation` block.
3. A decoding policy in `games/synthetic/provider.py::POLICIES`, named in the
   payload's `policy` field.
4. `games/synthetic/<name>/metrics.py` with `METRICS` and `to_round_view`.
5. A line in `games/registry.py::create_default_game_registry`.

Metrics are discovered from the game class's own package, not from its
`game_type` string, so a nested package needs no naming gymnastics.
