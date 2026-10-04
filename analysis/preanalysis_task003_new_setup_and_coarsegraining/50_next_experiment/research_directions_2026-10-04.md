# Three research directions: posterior-aware control, LLM-free simulation, and a mean-field theory

4 October 2026. Answers to three questions asked on 4 October. **Two of the
three are largely already built in this repository** — that is the main finding.

---

## 1. Should the controller know the posterior probability of its facts?

**Yes, and it is more meaningful than it first looks — but the naive version is
the wrong one.**

### What the controller currently knows

A frozen scalar per fact, `task.controller_fact_scores`, computed once offline.
With `controller_fact_pool_mode: balanced` it is forced to zero, so the
controller selects on novelty and reuse alone (see `SUMMARY_2026-10-04.md` §4).
It is a heuristic, not a policy.

### The naive version, and why it is not enough

Give the controller `ΔP(target | f)` — the **marginal** effect of each fact read
alone — and let it pick the argmax. This is computable exactly from the
14,388-world enumeration.

But our own analysis already shows marginal scores mislead:

- **15 facts raise both A0 and A2** (by weakening A1), so "favours my target"
  is not a property of a fact in isolation.
- **Adding true facts can lower P(target).** The 15-fact contrastive pool is
  *collectively more persuasive* toward A2 than the full 27-fact pool.
- The 9 A2-leaning facts jointly reach 0.933, while the whole pool proves A0.

So a greedy argmax on marginal lift would sometimes post facts that *hurt* its
target once combined with what is already on the board.

### The version worth building

Score the fact **conditionally on the current epistemic state**:

```
score(f) = E_i [ P(target | K_i ∪ {f}) − P(target | K_i) ]
```

the expected shift in the population's posterior, averaged over agents, given
what each currently holds. Every term is an exact count from the same engine.

This is meaningful for three distinct reasons.

**It gives an upper bound on achievable control.** Efficiency η is bits of
steering per nat of intervention, and a ratio is uninterpretable without a
reference. A controller with exact conditional scores and exact population state
is the **optimal-within-this-policy-class** reference. Reporting
η_measured / η_optimal is a far stronger claim than η alone.

**It makes the intervention a deterministic function of observables**, which
*helps* our estimator: the controller's policy no longer has to be inferred from
data, so the path KL loses one estimated component.

**It is a clean policy class to put in a paper** — "greedy one-step Bayesian
influence maximisation under a per-round budget" is a nameable baseline, unlike
the current heuristic.

### The honest caveat

It is **not realistic**. A real influence operation does not know the exact
Bayesian posterior over 14,388 worlds, nor each agent's active memory. So this
is an **oracle reference arm**, not the headline result. Three controller classes
then bracket the problem:

| class | knows | role |
|---|---|---|
| heuristic (current) | novelty, reuse, a frozen score | lower reference |
| **posterior-aware greedy** | exact conditional posteriors, exact state | **upper reference** |
| LLM-authored | the target, the board, its pool | the object of study |

**Partial-information variants are the interesting middle**: give the oracle
only the *sensed* board rather than the true state, and the gap between oracle
and LLM decomposes into "knows less" versus "reasons worse".

---

## 2. An LLM-free simulation

**Largely built already: `src/santa_fe/`.** And the speed answer is dramatic.

### What exists

`src/santa_fe/` — "Synthetic dynamics only; no estimators or output handling" —
is a provider-free epistemic simulator with a controller. Its agent rule
(`game.py`):

```python
e = mean(weights[f] for f in agent.active_facts)      # evidence, in [-1,+1]
s = mean(m.vote for m in sampled_messages)            # social signal
p_truth = sigmoid(beta_evidence * e + beta_social * s)
vote = +1 with probability p_truth else -1
```

Its parameters are exactly our protocol: `N, F_plus, F_minus, q, q_c, b, rho,
beta_evidence, beta_social, policy_beta, policy_threshold`. `v3_game.py` is an
"exact finite-population day/night process with an event ledger".

### What has to change for our setups

Two things, and only two:

| | santa_fe now | needed for task003-*-v2 |
|---|---|---|
| evidence | a **scalar** mean of ± fact weights | the **exact 3-allocation posterior** `P(A_k \| K_i)` from the 14,388-world engine |
| vote | **binary** ±1 via a sigmoid | **3-way categorical**, e.g. `p_k ∝ exp(β_e·log P(A_k\|K) + β_s·share_k)` |

Everything else — persistence, board lifetime, sensing fraction, controller
budget and gate — carries over unchanged.

Note the codebase already anticipated this: `dynamics_mode` has `reasoning` as
the only *implemented* value, with an explicit error saying a provider-free
kernel "would have to define what an exposed fact does to a q-voter jump, which
this version deliberately leaves open." **Defining that update rule is the real
work**, and it is a modelling choice, not an implementation detail.

### Speed — measured, not estimated

Benchmarked `engine.World.posterior` on the actual `task_003` world (14,388
valid worlds, 49 facts):

| active facts per agent | time per exact posterior |
|---:|---:|
| 1 | 61 µs |
| 3 | 36 µs |
| 6 | 50 µs |
| 12 | 23 µs |
| 24 | 22 µs |

Larger fact sets are *faster*, because the surviving-world set shrinks.

The full 12-cell grid at 100 episodes × 30 rounds × 24 agents needs **864,000
posterior evaluations**:

| | wall clock |
|---|---|
| **LLM-free, measured** | **~20–45 seconds** of posterior math |
| LLM, 1 call/agent/round, 12 s latency, 10 concurrent | **~288 hours (12 days)** |

Even with simulation overhead this is **minutes, locally, for free**, against
nearly two weeks and real money. Memoising the posterior on the frozenset of
active fact ids would cut it further, since few distinct knowledge sets occur.

No cluster needed. Dortmund is irrelevant at this scale.

### What it is good for, and what it is not

**Good for:**

- **Validating the estimators against known ground truth.** We can compute the
  true path KL and mutual information analytically from the simulator's own
  transition kernel, then check whether the 4-state coarse-graining recovers
  them. This is the single most valuable use, and exactly what the earlier
  critic-based estimators lacked.
- **Sweeping the design space cheaply** — pool compositions, assignments,
  overlap levels, ρ — before spending LLM budget.
- **Checking the mean-field theory** in §3 against exact finite-population
  dynamics.
- **Power analysis**: how many episodes do we actually need? Answerable exactly
  instead of extrapolated.

**Not good for:** the headline claims. A Bayes-exact or softmax-Bayes agent is a
**different dynamical system** from a language model doing soft reasoning. It
cannot be influenced by phrasing, cannot be wrong in correlated ways, and has no
notion of trust in a source. The LLM-free run is a **testbed**, not a
substitute.

---

## 3. A mean-field theory with two stochastic forces

**Also largely built: `src/santa_fe_theory/`** — "Reduced hybrid Langevin
theory; this module does not evolve individual agents." Its `Parameters` already
carry `policy_beta` and `policy_threshold`, i.e. the controller feedback.

What exists is **binary and four-category** — `(+,+), (+,0), (−,−), (−,0)` with
an epistemic law of independent Binomials. The proposed structure below is the
3-allocation generalisation.

### The process you described, written down

Let `M_t` be the mean epistemic state — our coarse coordinate,
`M_t = (1/N) Σ_i P(A0 | K_i(t))`. Let `S^p_t` be the peer force (epistemic
content of other agents, reaching an agent through the board) and `S^c_t` the
controller force.

```
dM_t  = −θ (M_t − M_∞) dt  +  g_p S^p_t dt  +  g_c S^c_t dt  +  σ_M dW^M_t
dS^p_t = −θ_p (S^p_t − M_t) dt                              +  σ_p dW^p_t
dS^c_t = −θ_c (S^c_t − u·π(M_t)) dt                         +  σ_c dW^c_t
```

Reading each term:

- **`−θ(M − M_∞)`** — forgetting. Each fact survives with probability ρ per
  round, so knowledge decays toward the empty set and `M` toward the **prior**,
  `M_∞ = 1/3`. Hence `θ ≈ −log ρ` per unit time (≈ `1−ρ` for ρ near 1). This is
  the only term present in the silent arm at long times.
- **`g_p S^p`** — peer injection. Agents post what they believe, so the board's
  content **tracks `M`**: the mean-reversion target of `S^p` is `M_t` itself.
  This is what makes the peer force *endogenous* and gives the system its
  consensus dynamics.
- **`g_c S^c`** — controller injection. Its target is `u·π(M_t)`, where `u` is
  the per-post posterior impact of a pool fact and `π` is the **activation
  policy**. For `policy: soft_target` with threshold τ and inverse temperature
  β,
  ```
  π(M) = σ(β (τ − M))      for a controller steering M up
  π(M) = σ(β (M − τ))      for one steering it down
  ```
  **This is the feedback you asked for**: the controller force depends on the
  state the equation tracks.
- the three `dW` are independent: finite-population sampling noise in `M`, board
  sampling noise in `S^p` (`sampling: uniform` makes this explicit; `full`
  shrinks it), and the controller's own stochasticity — zero under
  `deterministic` authoring with `report_only`, non-zero under `llm_authored`.

### Making it a genuine Ornstein–Uhlenbeck process

As written it is non-linear through `π(M)`. Linearise about an operating point
`M*` with `π'(M*) = −βπ(1−π)`:

```
dX_t = −A (X_t − x*) dt + Σ dW_t ,      X = (M, S^p, S^c)ᵗ

        ⎡  θ        −g_p      −g_c ⎤
  A  =  ⎢ −θ_p       θ_p        0  ⎥          Σ = diag(σ_M, σ_p, σ_c)
        ⎣ −θ_c u π'   0        θ_c ⎦
```

That is a 3-dimensional OU process, which is **exactly solvable**:

1. **Stationary covariance** from the Lyapunov equation `A C + C Aᵗ = Σ Σᵗ`.
2. **Stability** from the eigenvalues of `A`. Note the `(3,1)` entry carries the
   sign of the feedback: a truth controller and a false controller differ in the
   sign of `u`, so one *damps* and the other can *destabilise* — the
   bifurcation structure is a prediction worth testing.
3. **The payoff: the path KL has a closed form.** For two OU processes differing
   only in drift, Girsanov gives
   ```
   D(P ‖ Q) = ½ ∫₀ᵗ (b_P − b_Q)ᵗ (ΣΣᵗ)⁻¹ (b_P − b_Q) ds
   ```
   so the divergence between a controlled and a silent trajectory law is an
   **analytic function of `g_c, u, β, τ, ρ`** — directly comparable to the
   numbers we measure empirically (3.045 nats at ρ = 0.75, 1.931 at ρ = 1).

### Two things needing care

**`M ∈ [0,1]` is bounded**, so a linear SDE with additive noise is a local
approximation and will leave the interval. Two standard fixes: work in the logit
`Y = log(M/(1−M))`, which is unbounded and makes the OU form natural; or keep
`M` and use multiplicative noise `σ√(M(1−M))`, a Wright–Fisher / Jacobi
diffusion, which respects the boundaries and is the right object for a
finite population of voters. **The logit version is the one to write first** —
it keeps the OU solvability.

**Timescale separation is an assumption, not a fact.** The OU reduction assumes
`S^p` and `S^c` relax fast compared with `M`. The board clears every round
(`message_lifetime_rounds: 1`), which argues for fast `S^p`. But our four Markov
tests found ~0.015 nats/step of *unexplained memory* at ρ = 0.75 — evidence that
something slow is missing from a one-coordinate description. **The theory should
be tested against the LLM-free simulator of §2 before being trusted**, and the
missing coordinate is a candidate for a fourth equation (knowledge *spread*
under silence, decisive-fact survival under control — see
`../30_coarse_graining/`).

---

## Recommended order of work

1. **Extend `src/santa_fe/` to 3 allocations with the exact posterior** (§2).
   Bounded, local, free to run, and it unblocks everything else.
2. **Validate the 4-state coarse-graining and the path-KL estimator against it**
   — known ground truth for the first time.
3. **Write the logit-OU theory** (§3) and check it against 1 and 2.
4. **Add the posterior-aware oracle controller** (§1) as the upper reference.

None of this blocks the LLM experiments, and steps 1–2 would materially
de-risk them.
