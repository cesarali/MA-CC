# Running the synthetic games

*Implements Game 1 of `docs/tdd/architecture/05082026_synthetic_games_plan.md`.*

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

## The four commands

```bash
mas-cc synthetic truth   --config configs/runs/synthetic_bernoulli_fidelity.yaml
mas-cc synthetic sweep   --config configs/runs/synthetic_bernoulli_null.yaml --seeds 200
mas-cc synthetic episode --config configs/runs/synthetic_bernoulli_fidelity.yaml
mas-cc synthetic parity  --config configs/runs/synthetic_bernoulli_fidelity.yaml --seeds 5
```

| Command | Mode | What it answers |
|---|---|---|
| `truth` | — | What the closed form says, for this exact config, without running anything |
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
