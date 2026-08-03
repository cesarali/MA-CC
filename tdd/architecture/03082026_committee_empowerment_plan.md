# Committee Empowerment as a Fragility Probe — Implementation Plan

**Date:** 03/08/2026
**Scope:** naming-game committee empowerment; preflight sizing, estimator changes, fragility readouts, headline experiment.
**Audience:** the person doing the work, and the agent implementing it later.

---

## Part I — What is actually going on (the physics, in plain words)

### A population sitting in a valley

A population of agents playing the naming game eventually agrees on one name. Once
that happens, it stays there. Nobody is enforcing it; it is self-sustaining, because
every agent keeps using the name that has been working for them.

Picture a ball resting at the bottom of a valley. The valley is the convention. The
ball is the state of the population. Left alone, the ball stays put.

There are two valleys — one per name — separated by a ridge. To change the
convention, the population has to get over the ridge.

Here is the part that matters: **the two valleys are not the same depth.** Ashery et
al. found that one name (the "strong" convention) sits in a much deeper valley than
the other. Attacking the shallow one takes a committed group of ~2% of the
population. Attacking the deep one can take 67%. Same game, same agents, thirty-fold
difference in how hard it is to overturn — depending only on *which* name the
population happens to have settled on.

**So the interesting quantity is not "did the attack succeed." It is "how deep is
the valley the population is currently sitting in."** A shallow valley is a fragile
convention. That fragility exists before anyone attacks, and we would like to
measure it without waiting for an attack.

### The probe: a test charge

To measure the valley, we push the ball a little and watch how much it moves.

Concretely: we insert one or two agents that are not playing the game normally.
They ignore their own experience and just keep saying one particular name. They are
not part of the population being measured — they are an **instrument applied to it**,
like an external field applied to a magnet, or a test charge dropped into an
electric field.

- Push a ball resting deep in a steep valley → it barely moves.
- Push a ball resting near the ridge → it moves a lot, and may roll over entirely.

**The same probe gives a different reading depending on how fragile the state is.
That is the whole idea.** The reading tells you about the population, not about the
probe.

This is why the probe should stay *small*. We are measuring the landscape, not
bulldozing it. Once the committee gets large enough to reliably cause a takeover,
we have stopped measuring and started driving.

- **Probe regime** (1–2 agents): measuring fragility. This is the new thing.
- **Drive regime** (many agents): finding the critical mass. This is what the legacy
  grid already does.

Both are worth running. Confusing them is how you end up believing you measured a
property of the population when you actually just knocked it over.

### Why randomness enters

You want to know whether a light switch controls a lamp. If you tape the switch to
"on," you learn nothing — the lamp is on, but you cannot tell whether that is because
of the switch.

You have to flip it. Up, down, up, up, down — and watch whether the lamp follows.

A permanently-committed agent is a taped-down switch. It may be enormously
influential, but you cannot *measure* that influence from watching, because nothing
varies.

So across runs, **we vary which name the committee pushes.** Then we ask one question:

> Knowing which way I pushed, how well can I predict where the population ended up?

- Predict perfectly → the committee fully controls the outcome → **1 bit**
- Predict not at all → the committee controls nothing → **0 bits**

The randomness belongs to the experimenter, not to the attacker. A real attacker is
perfectly stubborn; that is fine. We are the ones turning the dial.

### Three signals that should agree

Physics says a system approaching a tipping point becomes sluggish: it takes longer
to recover from disturbances, and it wanders more on its own. This is *critical
slowing down*, and it is the standard early-warning signature for critical
transitions.

That gives us three independent readouts of the same underlying fragility:

| Readout | What it is | Cost |
| --- | --- | --- |
| **Probe empowerment** | bits the probe's direction determines about the outcome | expensive (needs many episodes) |
| **Recovery time** | how long the population takes to return after a pulse is removed | medium |
| **Autocorrelation / variance** | how sluggish and jittery the convention share is on its own | free (already in the data) |

All three should rise together as a convention becomes fragile. Three independent
signals agreeing is a far stronger result than any one alone — and two of the three
cost essentially nothing.

---

## Part II — Why we need a preflight, and what it actually does

### The problem

A full grid was specified at 17,500 episodes / ~25 million model calls and was never
launched. That is the thing to fix. But "just run a smaller version" is not a plan,
because **nobody knows what the right size is.**

### Why the size is unknowable in advance

Measuring information has a resolution limit, and the required sample size depends
*quadratically* on the size of the effect you are looking for:

$$I \approx 0.72\,d^2 \ \text{bits}, \qquad N_{\text{required}} \approx \frac{10\,(K-1)}{2\,I\ln 2}$$

where $d$ is the effect size (0 = no effect, 1 = perfect determination) and $K$ is the
number of cells in the contingency table.

The consequence:

| effect $d$ | resulting bits | episodes needed |
| --- | --- | --- |
| 0.97 | 0.80 | **~15** |
| 0.40 | 0.12 | ~90 |
| 0.10 | 0.007 | ~1,400 |
| 0.05 | 0.002 | ~5,800 |

**A factor of 20 in effect size is a factor of 400 in cost.** So until you know $d$
for each cell of your grid, any episode count you pick is a guess — and it will be
wrong by orders of magnitude in one direction or the other.

### What the surrogate is (and is not)

The **calibrated surrogate provider** is a fake agent that decides using the empirical
decision rule measured from real LLMs (Ashery Table 1), instead of calling a model.
It runs the entire existing pipeline — game, committee intervention, episode
derivation, parquet output, estimator — at zero marginal cost.

**It is not producing scientific results.** Its only job is to estimate $d$ for every
cell of the grid, so the sizing calculation has real numbers in it.

Think of it as a **wind tunnel**. You are not flying anywhere. You are finding out
how much fuel the real flight needs.

### The loop, and the artifact that connects the stages

```
  ┌──────────────────────┐
  │ 1. SURROGATE RUN     │   free      thousands of episodes, no API calls
  │    (wind tunnel)     │
  └──────────┬───────────┘
             │  emits: effect size d per grid cell
             ▼
  ┌──────────────────────┐
  │ 2. SIZING REPORT     │   free      the translation step
  │    sizing.parquet    │             d → bits → episodes → calls → $$
  └──────────┬───────────┘
             │  emits: a RESHAPED grid (cells dropped, merged, resized)
             ▼
  ┌──────────────────────┐
  │ 3. LIVE PILOT        │   cheap     ~50 episodes on 2-3 cells only
  │    (reality check)   │             does real d match surrogate d?
  └──────────┬───────────┘
             │  if predictions hold → proceed;  if not → recalibrate surrogate
             ▼
  ┌──────────────────────┐
  │ 4. FULL LIVE RUN     │   expensive  but now correctly sized
  └──────────────────────┘
```

`sizing.parquet` is the object the user asked about — **the code that translates
preflight results into experiment design.** It is not a summary; it is an input.
Stage 4's grid is generated *from* it, not written by hand.

### What the reshaping actually looks like

The sizing report does not just scale the grid up or down. It changes its shape:

- **Cells with large $d$** (committee of 6+ against a weak convention) need ~15–50
  episodes. Currently over-sampled — cut them hard.
- **Cells with small $d$** (probe of 1 against a strong convention) need thousands.
  Either commit the budget, accept a wide confidence interval and report it as
  exploratory, or drop the cell.
- **Cells with $d$ below resolution at any affordable N** get dropped before spending
  anything. This is the main saving.
- **Merging:** strata that the surrogate shows are statistically indistinguishable
  can be pooled, multiplying effective N.

The expected outcome is not "17,500 episodes → 5,000 episodes." It is
"17,500 uniformly-allocated episodes → ~2,000 unevenly-allocated episodes that
actually resolve the cells we care about."

### Honest limits of the surrogate

- It is calibrated to **one model** (Llama-3.1-70B-Instruct, from Ashery Table 1).
  Other models will differ, sometimes a lot.
- It reproduces the *decision rule*, not the model. Real agents will have
  correlations and quirks the surrogate lacks.
- Therefore treat surrogate $d$ as accurate to **a factor of 2–3**, not exactly.
  Apply a safety factor, and **stage 3 (live pilot) is not optional** — it is the
  check that the wind tunnel resembles the sky.

---

## Part III — The four workstreams

Recommended order: **1 → 2 → 3 → 4.** Workstream 1 gates everything expensive;
2 and 3 are cheap and can overlap.

**Package choice:** build in the legacy `src/naming_game/` package first. The
estimator, surrogates, checkpointing, and episode derivation already exist and are
tested there, so this is the shortest path to a real number. Porting to
`src/mas_cc/analysis/` is a separate decision to make *after* the science works.

---

### Workstream 1 — Calibrated surrogate provider + sizing report

**Goal:** produce `sizing.parquet` — required episodes per grid cell — for free.

#### 1a. The surrogate provider

Implement behind the **existing provider interface**, so `run_experiment` cannot tell
the difference. This is the critical design property: we are validating the *real*
pipeline, not a parallel reimplementation.

```
src/naming_game/providers/surrogate.py
    class CalibratedSurrogateProvider:
        """Ashery-Table-1 empirical decision rule. No API calls."""
        def __init__(self, table: DecisionTable, rng_seed: int): ...
        async def decide(self, prompt_or_memory) -> str: ...
```

**Calibration data — Ashery et al. 2025, Table 1** (Llama-3.1-70B-Instruct, W=2,
pool {Q, M}; **M is the strong convention**). Memory entries are
`(played, observed)`, most recent last.

*Interaction 1 (empty memory):* P(Q)=0.492, P(M)=0.508

*Interaction 2:*

| memory | P(Q) | P(M) |
| --- | --- | --- |
| 1: Q,M | 0.049 | 0.951 |
| 1: M,Q | 0.995 | 0.005 |
| 1: Q,Q | 0.997 | 0.003 |
| 1: M,M | 0.010 | 0.990 |

*Interaction 3:*

| memory | P(Q) | P(M) |
| --- | --- | --- |
| 1: Q,M  2: M,Q | 0.451 | 0.549 |
| 1: M,Q  2: Q,M | 0.152 | 0.848 |
| 1: Q,M  2: M,M | 0.000 | 1.000 |
| 1: M,Q  2: Q,Q | 0.996 | 0.004 |
| 1: Q,Q  2: Q,M | 0.064 | 0.936 |
| 1: M,M  2: M,Q | 0.841 | 0.159 |
| 1: M,M  2: M,M | 0.001 | 0.999 |
| 1: Q,Q  2: Q,Q | 0.989 | 0.011 |

*Beyond interaction 3:* fall back to win-stay (0.994) / lose-shift (0.973), with the
asymmetry carried by the interaction-3 table above.

> **Critical implementation note.** A pure win-stay/lose-shift rule is *symmetric* in
> Q and M and will **not** reproduce the strong/weak asymmetry — which is the entire
> point of Workstream 4. The interaction-2 and interaction-3 tables are what break the
> symmetry. They must be implemented, not approximated.

**Acceptance criteria**
- Populations reach consensus in ~15 population rounds at N=24 (matches Ashery Fig. 1).
- Consensus lands on M more often than Q from a neutral start (asymmetry present).
- A committed minority produces a takeover threshold, and the threshold is **larger
  when attacking M than when attacking Q**.
- Runs the full existing `run_experiment` → `analyze_histories` path unmodified.

#### 1b. The sizing calculator

```
src/naming_game/planning/sizing.py

    def bits_from_effect(d: float) -> float:
        """Exact MI for a symmetric binary channel with accuracy 0.5 + d/2."""

    def required_episodes(bits: float, n_cells: int, margin: float = 10.0) -> int:
        """N such that the plug-in bias floor sits `margin`x below the signal."""
        # floor = (n_cells - 1) / (2 N ln2);  require floor * margin <= bits

    def build_sizing_report(surrogate_estimates: pd.DataFrame,
                            cost_per_episode: float) -> pd.DataFrame:
        """One row per grid cell -> d, bits, N_required, calls, cost, verdict."""
```

`sizing.parquet` columns:

| column | meaning |
| --- | --- |
| `stratum_id` | the grid cell |
| `effect_size_d` | measured on surrogate data |
| `bits_expected` | `bits_from_effect(d)` |
| `n_cells` | contingency table size for this estimand |
| `episodes_required` | with 10× margin |
| `episodes_required_safe` | ×3 surrogate-uncertainty factor |
| `estimated_calls`, `estimated_cost` | feeds existing cost preflight |
| `verdict` | `run` / `merge` / `drop` / `exploratory` |

#### 1c. Grid generation from the report

```
    def grid_from_sizing(sizing: pd.DataFrame, budget: float) -> ExperimentConfig:
        """Emit the live-run config. Greedy allocation under budget,
        prioritising cells needed for the Workstream-4 headline test."""
```

**This is the translation step the whole preflight exists to produce.** The live
config is generated, not authored.

**Deliverable:** `sizing.parquet` + a generated `configs/empowerment_live.yaml`
+ a one-page summary of what was dropped and why.

---

### Workstream 2 — Split the treatment contrast

**Goal:** make the primary number interpretable.

**Problem.** `committee_policy ∈ {always_A, always_B, no_committee}` in a single
contingency table conflates two questions — *which way did I push* and *did I push at
all* — into one number that means neither.

**Change.**

- **Primary estimand:** the 2-level contrast only.
  `I(direction; outcome)` over `{push_A, push_B}`. Bounded at exactly 1 bit, and
  directly readable as "the fraction of the outcome the committee determines."
- **`no_committee` becomes a separate baseline arm**, used for the frequentist
  metrics and the existing negative control — not a treatment level.

**Files:** `analysis/empowerment.py` (`estimate_terminal`, `estimate_lagged`,
`STRATA_COLUMNS`).

**Keep unchanged:** the three smoothing variants, episode-level counting,
bootstrap over `episode_id`, the estimation-status gate, all three surrogates. That
machinery is sound.

**Acceptance criteria**
- Primary estimate never exceeds 1.0 bits (sanity bound).
- `no_committee` arm sits at or below the 95th percentile of its own shuffle null.
- Label-swap invariance still holds to 1e-9.

---

### Workstream 3 — Three fragility readouts

**Goal:** promote fragility to the primary output; add the two cheap signals.

#### 3a. Probe empowerment (fixed small committee)

`E_probe = I(direction; macrostate at t+τ | macrostate at t=0)` at **c = 1 and c = 2**.

Distinct from the existing sweep: committee size is *held fixed* while the
**population's initial state varies**. Reads the medium, not the attacker.

#### 3b. Recovery time (already implemented — promote it)

`recovery_time_*` from the pulse regime is a **critical-slowing-down measure**, not a
descriptive statistic. Recovery time diverges as a system approaches a tipping point.

Cheaper than the MI estimate — a bootstrap median needs far fewer episodes than a
contingency table. Report it as a primary fragility readout.

#### 3c. Autocorrelation and variance (new, free)

```
src/naming_game/analysis/slowing_down.py
    def slowing_indicators(interactions: pd.DataFrame,
                           window: int, detrend: bool = True) -> pd.DataFrame:
        """lag-1 autocorrelation and variance of rolling_share_A per episode."""
```

Computed off `interactions.parquet`, which already exists. **No new episodes.**

> Compute on a *stationary window before* any transition, and detrend first.
> Autocorrelation measured across a transition is meaningless.

#### 3d. The convergence check

One table, three readouts, ordered by fragility. **They should agree.** Disagreement
is informative: if autocorrelation rises but probe empowerment does not, the
population is wandering without becoming steerable — worth knowing.

**Acceptance criteria**
- All three computed per stratum with intervals.
- Rank correlation between the three reported explicitly.
- 3c adds zero episodes.

---

### Workstream 4 — The headline test: strong vs weak asymmetry

**Goal:** show the probe measures basin depth, using Ashery's result as ground truth.

**Claim.**

> A fixed one-agent probe reads **higher against the weak convention than against the
> strong one**, and the ratio tracks the ratio of their critical masses.

If it holds, you can measure convention fragility from short observation instead of
by running attacks to completion. That is the result.

**Dependency — do this first.** *Which* name is strong is **model-dependent**. Before
the main run, a small preliminary experiment must determine it for the model in use:

- neutral-start episodes, no committee, ~100 episodes
- record the distribution of which name wins
- the more frequent winner is the strong convention for that model

Then set `attack_direction` and `incumbent` strata accordingly.

**Design.** Probe c ∈ {1, 2} × incumbent ∈ {strong, weak} × τ ∈ {2, 5, 10, 20}, with
direction randomized. Sized by `sizing.parquet`.

**Falsification.** If probe empowerment is equal against both conventions while
critical masses differ by 10×+, the probe is not measuring basin depth and the
fragility interpretation fails. State this in advance.

**Acceptance criteria**
- Strong/weak difference significant against the shuffle null.
- Direction of the effect matches the critical-mass ordering.
- Effect survives in at least two of the three readouts from Workstream 3.

---

## Part IV — Decision gates

Do not proceed past a gate that fails.

| Gate | Condition | If it fails |
| --- | --- | --- |
| **G1** after 1a | Surrogate reproduces consensus timing + strong/weak asymmetry | Fix the decision table; the interaction-3 rows are the usual culprit |
| **G2** after 1b | Some non-trivial subset of cells is affordable | Redesign for larger effect (bigger τ, smaller N, stronger contrast) — **do not** just buy more episodes |
| **G3** after 2+3 | Full pipeline runs end-to-end on surrogate data; nulls behave | Estimator bug — fix before spending anything |
| **G4** live pilot | Real $d$ within ~3× of surrogate $d$ on 2–3 cells | Recalibrate the surrogate against pilot data, re-run 1b |
| **G5** full run | — | — |

**G2 is the important one.** If the numbers say the experiment is unaffordable, the
answer is to redesign it, not to fund it. Because bits go as $d^2$, a modest
improvement in experimental design beats a large increase in budget — doubling the
effect size is worth quadrupling the sample.

---

## Part V — Suggested order of work

1. **Surrogate provider** (1a) — the gate on everything else.
2. **Sizing calculator + report** (1b) — first real answer to "how big should this be."
3. **Treatment split** (2) — small, makes the primary number interpretable.
4. **Slowing-down indicators** (3c) — free, computed off data that already exists.
5. **Grid generation** (1c) — turns the sizing report into a runnable config.
6. **Strong/weak preliminary** (Workstream 4 dependency) — ~100 live episodes.
7. **Live pilot** (G4) — check the wind tunnel against the sky.
8. **Full run**, sized by the report.

Steps 1–5 cost nothing but time and answer the question that stalled the project.

---

## Appendix — The estimation principles being applied

Kept here because every sizing decision above follows from them.

1. **Ceiling first.** $I(X;Y) \le \min(H(X), H(Y))$. A binary treatment gives at most
   1 bit, no matter what.
2. **MI is quadratic in effect size.** $I \approx 0.72\,d^2$ bits. Halve the effect,
   quarter the bits. (Information is zero at independence and cannot be negative, so
   near independence it must look like a parabola.)
3. **The estimator reads positive on noise.** Floor $\approx (K-1)/(2N\ln 2)$ bits.
   Random fluctuation always creates apparent association, and MI cannot go negative,
   so the bias only goes up.
4. **Conditioning is expensive.** Each conditioning bin multiplies cells and divides
   data per cell — it raises the floor while also removing real signal.
5. **Confounds beat small signals.** A 0.005-bit effect next to a 0.35-bit structural
   correlation is unmeasurable regardless of sample size. Only better conditioning or
   a better contrast fixes it.
6. **The shuffle null is the verdict.** Formulas are for planning. If observed ≈ null,
   stop and redesign.

**Worked example (why per-round per-agent measurement fails).** One agent among 24
shifts the population share by roughly $d \approx 0.07$ five rounds later. So
$I \approx 0.72 \times 0.07^2 \approx 0.0035$ bits — needing ~1,400 samples for a bare
2×2 table, and ~6,000–20,000 once conditioned. Meanwhile the episode-level treatment
signal is ~0.8 bits and needs ~15. **Same estimator, 100× difference in cost, entirely
because of where the measurement is taken.** This is why the primary estimand is at
the episode level.

---

## References

- Song, Gore & Kleiman-Weiner (2026), *Estimating the Empowerment of Language Model Agents*, ICML. arXiv:2509.22504
- Ashery, Aiello & Baronchelli (2025), *Emergent social conventions and collective bias in LLM populations*, Sci. Adv. 11, eadu9368
- Riedl (2026), *Emergent Coordination in Multi-Agent Language Models*, ICLR. arXiv:2510.05174
- Jaques et al. (2019), *Social Influence as Intrinsic Motivation for Multi-Agent Deep RL*, ICML
- Scheffer et al. (2009), *Early-warning signals for critical transitions*, Nature 461
- Lo Iudice, Garofalo & Sorrentino (2015), *Structural permeability of complex networks to control signals*, Nat. Commun. 6:8349
- Yan et al. (2012), *Controlling complex networks: How much energy is needed?*, PRL
