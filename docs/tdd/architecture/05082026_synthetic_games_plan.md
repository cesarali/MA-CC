# Synthetic Games — design plan

*05.08.2026*

## The goal, in one paragraph

These are **synthetic games with analytically known answers**, built so that every
information-theoretic quantity the system reports can be compared against a closed-form
value that we derived ourselves. The agents are not LLMs — they are lookup tables plus
coins, with dynamics we specify. Because we defined the dynamics, we know in advance what
the mutual information, the transfer entropy, and the control-input capacity must be. When
the system reports something else, the discrepancy is unambiguous.

The purpose is **confidence in the system and in our use of it** — the estimators, the
adapters, the recorder, the cluster submission path, and the way we read results — at a
stage where the code is fresh and mutual information is the most sensitive thing in it.

## What these are *not*

**They are not tests in the traditional sense.** There is no assertion, no red/green, no
CI gate. Nothing here fails a build.

They are a **rehearsal of the full workflow with the answer key in hand**. We run them the
way we run a real experiment: same `Game` contract, same decision loop, same recorder,
same submission to the university cluster, same artifacts pulled back, same way of looking
at the output. The only difference is that we already know what the numbers should be. If
the workflow produces the right answer end to end on a problem where we can check it, we
have grounds to trust it on a problem where we cannot.

This also means they are **not about interpreting mutual information between LLM agents**.
Nothing here says anything about emergent conventions. It is entirely about whether the
machinery is behaving as it should.

---

## The three games

One parametric family, three members, increasing in structure.

### Game 1 — Bernoulli

**Not the easy version of Game 2.** This is the game where every quantity is either zero or
a known constant. Different job: it establishes the floor and the calibration curve.

**Mechanics.** No dynamics, no memory. Each round, nature draws a latent
$Z_t \sim \mathrm{Bern}(1/2)$. Agent $i$ plays

$$A_{i,t} = Z_t \oplus B_{i,t}, \qquad B_{i,t} \sim \mathrm{Bern}(\varepsilon_i)$$

with $B$ private and independent across agents. Marginals are uniform by construction.

**Ground truth.** $A_1 \oplus A_2 \sim \mathrm{Bern}(q)$ with

$$q = \varepsilon_1(1-\varepsilon_2) + \varepsilon_2(1-\varepsilon_1)$$

so

$$I(A_1; A_2) = 1 - H(q) \quad \text{bits, exactly.}$$

Two anchors matter: $\varepsilon = 0.5$ gives true MI $= 0$; $\varepsilon = 0$ gives 1 bit.

**What it buys us.**

- The **null distribution**. Run the $\varepsilon = 0.5$ case for a few hundred seeds at the
  same $N$ and round count as a real experiment. Plug-in MI is biased upward at finite
  sample; this tells us by how much, and therefore what magnitude of MI means anything at
  all in a real run.
- The **calibration curve**. Sweep $\varepsilon$ on a grid. We get a curve rather than a
  point, and estimator bias shows up as a visible offset from the diagonal rather than as a
  number we have to form a judgement about.

### Game 2 — Markov

Real dynamics, real convergence toward consensus, still closed-form.

**Mechanics.** Binary alphabet $\{Q, M\}$, matching the real naming game. Agent $i$'s next
action depends on the previous round's observed partner action through a fixed $2\times 2$
stochastic kernel, plus private noise $\varepsilon_i$. A coupling graph specifies who
observes whom, and each edge carries a lag. Pairing uses whatever rule the real game uses.

Keep $N \le 10$ so the state — the full action profile — is at most $2^{10} = 1024$ values.

**Ground truth.** Build the exact transition matrix over the profile state space, take the
stationary distribution by eigendecomposition, and every quantity follows from exact linear
algebra with no sampling error to argue about:

- entropy rate
- temporal $I(A_t; A_{t+1})$
- cross-agent $I(A_i; A_j)$
- conditional MI / transfer entropy $I(A_{i,t+1}; A_{j,t} \mid A_{i,t})$

An $N = 8$ case is a $256 \times 256$ matrix — a few milliseconds.

**What it buys us.** The coupling graph is more diagnostic than the noise level, because it
lets us design **structural zeros**. A value that is 5% off is hard to call a bug. A
conditional MI that must be exactly zero and isn't is not ambiguous.

- **Chain $1 \to 2 \to 3$**: $I(A_1; A_3 \mid A_2) = 0$, and
  $0 < I(A_1; A_3) < I(A_1; A_2)$. Tests conditional MI and the data-processing inequality
  together.
- **Asymmetric lag**: $A_{2,t}$ depends on $A_{1,t-1}$, and nothing depends on agent 2.
  Then $TE(1 \to 2) > 0$ and $TE(2 \to 1) = 0$ exactly. This is the one that catches
  direction errors and off-by-one round alignment — the classic silent failure.

### Game 3 — Controlled Markov

Game 2 plus an exogenous control input, for the `control/` side.

**Mechanics.** An external controller sets $u_t$ each round, biasing a specified subset of
agents toward one action with a given strength. Everything else is Game 2.

**Ground truth.** The chain becomes a family of transition matrices indexed by $u$. On a
state space this small, $I(U; S_{t+k})$ and the channel capacity from controller to
population state are computable exactly — Blahut–Arimoto over $\le 1024$ states is
instant. We can sweep $u$ and lag $k$ densely and still be waiting on nothing.

**What it buys us.** A known target for any empowerment-style estimator, and a
capacity-vs-lag curve to check the estimator's shape against, not just a single value.

---

## Additional configurations

Same game, different parameters. Worth having, lower priority than the three above.

| Config | Setup | Ground truth |
|---|---|---|
| Change point | $\varepsilon$ switches at a scheduled round $t^\*$ | Pooling the episode gives a *predictable wrong answer*; correct windowing recovers both segments |
| Degenerate | $\varepsilon = 0$, all agents locked to one action | Zero entropy, MI is $0/0$ — checks we don't silently return NaN or 1.0 |
| High MI | $k$-ary alphabet, low noise, true MI above $\log K$ for the contrastive batch size $K$ | Makes the InfoNCE ceiling visible instead of looking like a measurement |

The change-point config is the one that speaks to a real concern: the naming game
*converges*, so pooling counts across a whole episode mixes regimes. Here we know the
switch time, so we can see exactly what pooling does to the number.

---

## How they run: two modes

**Fidelity mode.** Full pipeline. Prompts constructed and discarded, actions flowing
through `parse_action` / `validate_action` exactly as if a model had produced them,
recorder logging everything. Slow — perhaps 10–20 seeds. This is what exercises the
adapter, the round alignment, the serialization, and the artifact path.

**Speed mode.** Sampler writes actions directly into state, minimal logging, vectorized
across seeds. Thousands of seeds. 500 seeds × 2000 rounds × 8 agents is 8M draws — under a
second in numpy. This is what produces null distributions and calibration curves with
usable error bars.

**The check that makes the pair valid:** run one config in *both* modes and confirm they
agree. If they do, speed mode is a trustworthy proxy and all sweeps can live there. If they
diverge, we have found a pipeline bug — which is the entire point of building this.

### Where the time actually goes

With no API calls, the arithmetic is nothing and **our own framework becomes the
bottleneck**. Worth measuring a single 1000-round synthetic episode early and seeing where
it goes:

- ~1 ms/round of framework overhead → a 500-seed × 2000-round sweep is ~17 minutes
- ~10–50 ms/round (per-round recorder flush, per-agent prompt string work) → hours

That number decides whether fidelity mode runs 20 seeds or 500.

### Determinism

No API in the loop means full reproducibility. Seed the **pairing selection**, not only the
action noise — then any failure replays exactly, which is the difference between debugging
in minutes and debugging in days.

---

## Reading the results

Because this is a rehearsal of the workflow rather than a test suite, the analytic answer
cannot live in an assertion. It has to be visible at the moment we look at results.

**Ground truth as a first-class artifact.** Each synthetic game gets a `ground_truth()`
that computes the closed form *from the same resolved config object that ran* — never from
a hand-written expected value in a separate file. The recorder writes it into the run
artifacts next to the estimate. Pull results off the cluster and the truth is already in
the file.

This kills the phantom-bug failure mode: change $\varepsilon$ in the YAML, forget to update
the expected number, lose a day debugging an estimator that was fine.

The discrepancy is then a column, not an investigation: `estimate`, `truth`, `gap`, error bar.

**One canonical view per game**, decided in advance, so that "did it work" is a look rather
than an analysis:

| Game | The view |
|---|---|
| Bernoulli | Calibration curve — estimated MI vs. $1 - H(q)$ across the $\varepsilon$ sweep, diagonal drawn. Bias is a visible offset. |
| Markov | Estimate vs. exact chain value, with the structural zeros (conditional MI on the chain, reverse-direction TE) reported separately — those are pass/fail, not close/not-close. |
| Controlled Markov | $I(U; S_{t+k})$ vs. $k$, plotted against the exact capacity curve. |

---

## Cluster rehearsal

These are the ideal smoke runs for the cluster path precisely because they cost nothing.
Make one long enough to hit checkpointing, then kill it deliberately mid-run and resume.

Submission, environment, checkpoint, resume, artifact retrieval — this is plumbing we
cannot afford to debug with real API calls behind it, and realistically more time will go
here than into the MI mathematics.

---

## Layout

```
synthetic_games/
├── game_01_bernoulli/
├── game_02_markov/
└── game_03_controlled_markov/
```

Each implements the same `Game` contract as the real games, plus `ground_truth()`.

## Order of work

1. **Game 1**, speed mode, null config → the plug-in bias and the significance floor. Smallest thing that says something useful about what we already have.
2. **Game 1**, calibration sweep → the estimator's bias curve.
3. **Game 1** in fidelity mode, compared against speed mode → validates the proxy, exercises the pipeline.
4. **Game 2**, exact chain ground truth → structural zeros, TE direction, round alignment.
5. **Game 3** → control and empowerment quantities.

Cluster rehearsal can happen from step 3 onward, since by then there is a run worth submitting.
