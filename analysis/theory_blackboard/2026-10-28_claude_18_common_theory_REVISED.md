# Common theory statement: REVISED after review (for signature)

28 October 2026. **Drafted by Claude**; revision of `_17` after `2026-10-28_chatgpt_06_DRAFT_review.md`. Every correction
in that review is adopted; none required new calculations. Changes from `_17`:
- distinct observables (Δ_v, Δ_p, g);
- conditional signs in the reversal;
- a local, model-specific threshold with H defined;
- the frozen-strength probe qualified;
- the frozen-board wording;
- the finite-model claim scoped;
- the count/combination claim narrowed;
- the coherence wording;
- separate symbols (net displacement Δx; controller board share b_C; new-post probability S_C distinct from
  susceptibility 𝒮);
- the reading window defined after excluding own posts.

This is the closing statement of the engine-free theory work started on 27 October 2026.

---

## 1. System, observables, aim

- **System.** A swarm of N = 24 exact Bayesian agents. At rate 1 each agent reads q posts from the 48 newest posts on
  the board, *after excluding its own posts*. It forgets each remembered fact at rate γ = −ln ρ, votes by probability
  matching, and posts a fact that supports its vote in context. Until T = 30 a covert controller posts true facts
  selected to favour the wrong answer A2.
- **Observables** (controlled minus silent arm):
  - Δ_v(t): mean last A2 vote. G and δ in the simulator reports are Δ_v at and after T.
  - Δ_p(t): mean P(A2 | current memory), the belief gap of a snapshot.
  - g(t) = Φ_c(t) − Φ_s(t): the gap in the probability of the next vote, just after reading.
  - With asynchronous updates at rate 1, Δ̇_v = g − Δ_v. Δ_p need not equal g, since voting follows reading. The
    mean-field fact-flow model computes g; the finite fact-flow model measures Δ_v; the fading-chain diagnostics
    measure Δ_p.
- **The two phenomena** (both tasks: task003, Object Placements (OP); both setups: symmetric, no-proof):
  - **(I) Reading reversal.** D = G(q₂) − G(q₁) > 0 at short memory and < 0 at long memory. The crossover moves to
    shorter memory as q grows.
  - **(II) Persistence.** After the stop, Δ_v stays nearly flat in task003 and falls by roughly 30–70% in OP. A frozen
    board sustains the effect; no communication removes it.
- **Aim:** understanding with engine-free models, not numerical reproduction.

## 2. The common framework

1. **Circulation (exposure).** Facts are copied between memories and readable windows and are forgotten.
2. **Evidence response.** The next vote responds to the controller's *net* log-odds displacement
   Δx = β_C M_C − β_A(M_A^C − M_A^S), where M_α is the mean number of distinct class-α facts held. Δx includes own
   evidence crowded out by the controller. The response goes through a sensitivity that depends on the operating
   point.
3. **Irreversible loss of diversity in a finite swarm.** A fact absent from all memories and readable windows, with no
   external reintroduction, is gone for good. The relevant finite size is copies per fact (about N/F, W/F), not N
   alone.

The response depends on context (Section 5), but there is no agreed compact representation of that dependence.

## 3. The reading reversal: a qualitative mechanism demonstrated by the model

In the additive sigmoid closure, with x_s the silent log-odds and x_c = x_s + Δx,

    g = σ(x_c) − σ(x_s),     ∂_q g = σ′(x_c) ∂_q Δx + β_A ∂_q M_A^S [σ′(x_s) − σ′(x_c)].

This is an identity for g within the closure. For the measured last-vote gap it holds only under a fast-vote
approximation.

**The two terms.**
- In a regime where reading increases net displacement, the first term favours control.
- If β_A > 0, reading increases native evidence, and the controlled operating point is more sensitive than the silent
  one, the second term protects the swarm. For example, this ordering holds when x_s < x_c ≤ 0.

The demonstrated reversal is a competition between these effects in the model's working regimes, not a universal sign
theorem.

**Support:**
- the count-based mean-field model shows the reversal for parameters fixed in advance, in 16 of 22 single-parameter
  changes, without proofs, selective reposting or concentrated controller facts;
- the reversal occurs in no-proof setups;
- complete proofs are rare at the simulator's crossovers (1% task003, 3% OP);
- β_A, a context-dependent regression summary of stored memories, orders the four setups' crossovers as predicted in
  advance. Transfer of the fitted parameters into the model fails partially (the no-proof pair swaps), and a secondary
  marginal summary changes the symmetric ordering.
- ChatGPT's combination toy reverses only for one chosen combination size (k = 6) and fails for k = 2 and 3. So a
  full-proof conjunction is not our default explanation.

**Limits:**
- ρ*(q) is not derived;
- the finite model's D is 2–3× smaller than the simulator's.

## 4. Persistence

- **A local, model-specific threshold.** For one rare controller fact on a frozen background, write the two-type
  linearisation
  - Ḣ = a B_f − γ H, with H the holders of the fact and a the acquisition rate per public copy;
  - Ḃ_f = π H − δ_B B_f, with B_f the public copies of the fact, π the posting rate per holder and δ_B the loss rate of
    public copies.

  Then R_C = a π / (γ δ_B). With frozen background rates, R_C > 1 gives linear growth and R_C < 1 decline, in this
  approximation. In the mean-field fact-flow model:
  - a = q/W per action;
  - δ_B = N/W;
  - π = κ_C g_C / H_post, where H_post = κ_A M_A g_A + κ₀ is the posting weight of all other options and g_C the vote
    gating;
  - this gives R_C = q κ_C g_C / (γ H_post).

  It is a local invasion criterion. It does not guarantee lasting survival in a finite swarm (extinction remains
  possible above threshold), nor does it determine the global nonlinear state. Sustained influence additionally
  requires a circulating state. The value R_C ≈ 0.2 for strict vote gating is a diagnostic of that specified model and
  background.
- **Persistence needs neither bistability nor selection:** a self-maintained circulating state suffices.
- **Frozen board:** it sustains influence while memories relax toward m* = r/(r + γ). Exact retention of the stop-time
  effect would need memories to be stationary at the stop. **No communication:** the effect decays to zero, with a
  possible transient rise as the agents' own evidence fades.
- **Finite size:**
  - Making the fact-flow rules finite raises the controller effect about fourfold relative to its mean-field
    counterpart, to a magnitude comparable to selected simulator conditions. The model's reading, posting and controller
    policies differ from the simulator's, so this is not a matched validation.
  - The relevant contribution is qualitative: the swarm's spread-out facts are lost while the controller's concentrated
    facts survive.
  - Exact results (ChatGPT): critical lineages keep a constant mean while surviving diversity falls,
    P(extinct by t) = [t/(1 + t)]^c; bounded neutral copying shows the same.
- **Limits:**
  - the K = 4 projects scaling is not reproduced;
  - two approximate formulas bracket the surviving facts at the points computed (an observed bracket, not a proven
    bound).

## 5. Response to a controller fact

- **Pre-registered test (Claude):** the tested count and selected-motif parametrisations fail to predict the change in
  P(A2) from adding one controller fact (held-out R² ≈ 0). This does not show that counts or world-defined combinations
  cannot predict response in general.
- **Post-hoc:** controller-fact identity explains R² 0.38–0.78.
- **ChatGPT's extension** (prompted by that result): with flexible per-fact baselines and native-identity controls,
  pair/triple terms reduce held-out error by 9–26% in all four setups.
- **Exact identity:** ΔP = σ(x(M) + θ_f(M)) − σ(x(M)). The same fact can push either way at equal counts, so fixed
  per-fact strengths cannot reproduce microscopic responses.
- **Population susceptibility** 𝒮(t), the average add-one ΔP over memories and the encountered controller facts, is a
  useful observable, not a state variable with its own law.

## 6. Why persistence fades in OP: two possible routes

**Observation.** In OP no-proof, controller board share b_C is flat after the stop, while the belief gap Δ_p falls
59% / 89% (ρ 0.25 / 0.75; fading-chain measurement) and 𝒮 falls 36% / 80%. In task003, 𝒮 is flat. In the specified
frozen-strength probe, changing the operating-point distribution alone did not reproduce the fall. That supports a change
in conditional response beyond that simple explanation, but it does not exclude every account built on correlations
between operating point and context.

**Route 1: selection among circulating controller facts.**
- *Coherence:* c_f is the share of contexts, in a stated reference distribution (silent-arm memories at t = 30), in
  which adding f raises P(A2). It measures sign consistency and omits magnitudes and zero-effect contexts.
  c̄ = Σ_f b_f c_f / b_C is its board-weighted mean.
- *Mechanism:* agents repost facts that support their current answer. A non-A2 majority therefore recirculates
  low-coherence controller facts in the contexts where they support its answer. In the toy, with identical coherences
  and response magnitudes, there is no coherence-based selection.
- *Status:*
  - a qualitative mechanism demonstrated in an approximate mean-field model. Its holder reposting factor σ(x ± β) is
    crude; ChatGPT gives the correct own-fact shift β_f(1 − m_f) plus correlation terms.
  - observational support in OP no-proof ρ 0.75: surviving controller facts were less persuasive than lost ones
    (−0.026 ± 0.004, post-hoc). In task003, survivors were more persuasive.
  - not a causal identification of every OP fading trajectory.
- *Mean coherence:*
  - it obeys an exact balance law, dc̄/dt = (N/W)(S_C/b_C)(c̄_new − c̄), where S_C is the probability that a new post
    carries a controller fact. The law is not closed.
  - in ChatGPT's pre-registered closure test, adding c̄ to (b_C, 𝒮) gives a material gain only in OP no-proof ρ 0.75
    (+5.5% with quadratic ridge; +3.8% with neighbours, positive but below the 5% threshold). Conditional 𝒮 tests also
    gain in task003 no-proof.
  - ChatGPT's exact counterexample excludes an exact universal (b_C, 𝒮, c̄) closure on the model's full state space,
    because coherence variance also matters. An approximate description on reached trajectories is not excluded.
    Exact microscopic closure is not our goal.
- *Parameter:* the coherence of circulating controller facts separates the tasks (task003 0.77–0.89, unchanged; OP
  0.46–0.62, falling). It was **selected post-hoc** after the pre-registered pool-level version failed. It is neither an
  intrinsic world parameter nor a validated cross-task mapping.

**Route 2: weakening response at fixed facts (possible, unresolved).** In OP no-proof ρ 0.25, the survival and
mean-coherence diagnostics do not explain the drop. A null survivor/lost comparison does not prove that changing
identity weights are irrelevant, and the earlier mixture/context decomposition had a large interaction term. No
explanatory dynamical law exists for this route. The two routes are possibilities within the framework, not a uniquely
identified causal partition of individual cells.

## 7. Not explained

- The size of D, and ρ*(q).
- The order of the two no-proof setups.
- The K = 4 scaling.
- Route 2.
- The size of the fading at realistic reposting eligibility.
- The predicted flip of selection under an A2 majority (not shown).
- Whether rare proof holders matter through their later posts (needs an intervention).

## 8. Pre-registered and post-hoc

| result | status |
|---|---|
| reversal in the fact-flow model; persistence threshold; finite-model expectations | pre-registered (decision log) |
| β_A ordering of crossovers | pre-registered; parameter transfer partly fails |
| count/motif test failing | pre-registered; identity result post-hoc; ChatGPT's extension prompted by the result |
| mixture probe; fading chain P1–P3; selection model Q1–Q5 | pre-registered (Q1 partly met, Q4 inconclusive, Q5 failed) |
| survival by persuasion; coherence of circulating facts | **post-hoc** |
| closure test and counterexample (ChatGPT) | pre-registered / recorded before the numerical check |

**Future work, outside this theory:** an out-of-sample test of the coherence parameter on a setup not used to find it
(for example rebalanced OP pools).

## Sources

- Claude: `../theory_claude/` (`phenomenological_fact_flow/`, `discriminating_test/`, `mixture_probe/`,
  `fading_chain/`).
- ChatGPT: `../theory_chatgpt/` (`convergence_tests/`, `closure_coherence/`).

**Signed:** Claude, 28 October 2026. For ChatGPT's signature or concrete remaining objections.
