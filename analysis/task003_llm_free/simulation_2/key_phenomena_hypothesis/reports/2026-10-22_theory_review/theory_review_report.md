# Theory of the reading reversal and persistence: models M1–M4, scaling, and a second task

**For:** a reviewer agent who will check every claim and propose other candidate models.
**Date:** 22 October 2026. **Branch:** `dev/rsanchez` (commits up to the one that adds this file).
**Scope:** Simulation 2 (continuous time) only. Plan followed: [`../../00_theory_plan_2026-10-21.md`](../../00_theory_plan_2026-10-21.md).
Decision log: [`../../progress.md`](../../progress.md).

## Summary

- **The phenomena.** In a swarm of 24 exact-Bayesian agents sharing a board, reading more posts (larger q) helps a
  false controller when agents forget fast and protects the swarm when they remember (the *reading reversal*); the
  controller's effect persists long after it stops (*persistence*).
- **Deterministic mean-field theories fail** (M1: fact-by-fact; M2: exact single-agent memory, averaged board):
  held-out signs at chance, effects about 3× too small. Cause: the averaged board has infinitely many copies of every
  fact, so facts never go extinct; in the real swarm each fact starts with about 1.3 holders and extinction carries
  most of the effect.
- **M3 = M2 + finite-swarm extinction** (integer holder counts per fact, absorbing at zero): held-out signs 13/17 and
  13/16, post-stop mechanisms right, but magnitudes about 60% and the amplifying side of task003-symmetric missed.
- **M4 = M3 + one shared, realised board** (all readers see one finite board fed by the swarm's posting stream and one
  stochastic controller): held-out G error 0.027 (M3 0.12), all misses at crossover cells, post-stop branch ordering
  7/8 with the data's timescales; reproduces the K = 2 and K = 4 scaling worlds (post-hoc, 7/7 and 8/8 resolved signs, G error ≤ 0.02); and, **pre-registered
  on a new task** (MuSR Object Placements), predicts G within 0.02–0.03 and the main crossover exactly (symmetric) or
  within its bracket (no-proof). M3 fails on the new task.
- **Scaling.** Scaling agents with facts fixed raises copies per fact and shrinks the effect; scaling both (K
  independent projects) keeps the reversal at K = 2 and 4 but moves its boundary to higher memory and changes how
  informative each fact is. No scaling keeps everything constant (section 1.6).

## 0. Setting, definitions, protocol

- **Simulator rules** (exact, from the code): [`../../theory/00_microscopic_rules.md`](../../theory/00_microscopic_rules.md).
  Agents act at rate 1; read q posts without replacement from the W = 48 newest (own posts excluded); remember facts
  with exponential lifetimes (rate γ = −ln ρ, renewed on rereading; own fact forgettable); vote by probability matching
  from the exact posterior; post one fact that supports the vote in the context of their memory, preferring facts not
  just read. The false controller (rate 1, budget 30, gate at 75% of read votes) posts the pool fact that maximises
  P(A2 | what it read). Paired arms: every random stream is keyed by (seed, episode, purpose), never by arm or q.
- **Observables.** G = mean over episodes of [A2 vote share with the controller − same episode without it] at t = 30
  (shares among agents who have voted). D(q₁→q₂) = G(q₂) − G(q₁), paired per episode. δ(t) = the same difference after
  the controller stops. "Resolved" = 95% interval excludes 0. A model increment counts as ± when |D| ≥ 0.01.
- **Hold-out protocol** ([`../../calibration/holdout_preregistration.md`](../../calibration/holdout_preregistration.md),
  committed before any model): model choices may use only development data (task003, ρ ∈ {0.25, 0.75, 1}, q ∈ {3, 6},
  t ≤ 30); everything else is held out and evaluated once per model version. Caveat stated there: the held-out data had
  been seen in earlier reports; the split constrains model choice, not prior exposure. The Object Placements data were
  **not** seen before the M3/M4 predictions were committed (commits `c0d22bc`, `c9d66c2`).
- **Models** share one code base: [`../../theory/models/mean_field.py`](../../theory/models/mean_field.py), runner
  `run_models.py`, scorecards `../../held_out_validation/score.py`, `score_post_stop.py`. All models use the simulator's
  own posterior, vote and posting functions (imported), so rules cannot drift.

## Section 1. task003 (team allocation)

### 1.1 Data recap (K = 1)

Reading reversal (step 2 run `2026-10-20_sim2_reading_mechanism`, 1,000 paired episodes): the crossover of D in (ρ, q)
descends with q; symmetric 3→6 between ρ 0.5 and 0.625, no-proof between 0.75 and 0.875–1 (tables in
`../../tables/step2b_D.csv`). Persistence (`2026-10-20_sim2_persistence_branches`, to t = 130): flat under normal and
frozen-board continuation, gone within 1–5 time units without communication, collapses to a small residual when
controller-only facts are removed.

### 1.2 M1 and M2: deterministic mean field (failed)

[`../../theory/M1_M2_mean_field.md`](../../theory/M1_M2_mean_field.md). Population of 24 × 100 agents with exact rules,
interacting only through a deterministic board (expected composition of the 48 newest posts). M2 keeps each agent's
exact memory; M1 redraws memory fact by fact from population shares before each action (the only difference).

| model | held-out signs symmetric | held-out signs no-proof | held-out G error |
|---|---|---|---|
| M1 | 9/17 | 6/16 | 0.145 / 0.181 |
| M2 | 8/17 | 8/16 | 0.151 / 0.179 |

**Why they fail** (diagnostics, `../../theory/models/diag_*.py`):
1. *Not a bug:* the same code with a real 24-agent board reproduces the simulator (G 0.322 vs 0.33 symmetric, 0.288 vs
   0.29 no-proof at ρ 0.5, q 6).
2. *Not the memory closure:* M1 and M2 fail alike.
3. *Copies per fact:* replicating agents (N = 24R with W, rate, budget × R) shrinks G toward the mean-field value
   (symmetric 0.32 → 0.20 → 0.14 → 0.14 for R = 1, 2, 4, 8 at ρ 0.5, q 6; mean field ≈ 0.11). Researcher's correction: this multiplies agents but not
   facts, so it raises copies per fact (≈1.3 R); it is a redundancy series, not a size series.
4. *Extinction carries the effect:* forbidding agents to forget their own starting fact (so no agent fact can vanish)
   cuts G from 0.44 to 0.09 (symmetric ρ 0.25) and from 0.30 to 0.20 (no-proof ρ 0.25). Caveat: this also gives every
   agent one extra fact.

**Can M1/M2 get a 1/N correction?** A van Kampen system-size expansion (Gaussian fluctuations O(1/√N), 1/N moment
corrections) expands in the wrong small parameter: the relevant one is 1/(copies per fact) ≈ 0.75, and extinction is an
absorbing event, not a small fluctuation. A deterministic cutoff (kill a fact when its expected holders fall below 1)
is a cheap alternative but is a crude version of M3 and was not run.

### 1.3 M3: mean field + finite-swarm extinction (partial success)

[`../../theory/M3_extinction.md`](../../theory/M3_extinction.md). One change from M2: each fact's holders in a swarm of
24 are an integer that changes by binomial draws with the gain/loss probabilities of the population; a fact with no
holder and no board copy is extinct for good. 40 realisations, 720-agent population.

- Held-out signs 13/17 (data + 1/4, − 12/13) symmetric; 13/16 (+ 8/10, − 5/6) no-proof; held-out G error 0.12.
- Post-stop: no-communication half-times right (1.0/1.0, 2.5/2.5, 1.0/1.0, 7.5 vs 5.5); frozen board flat; removal
  residual sign right in 2/4 (wrong in symmetric: −0.03 vs +0.04 at ρ 0.25, +0.01 vs −0.01 at ρ 0.75); normal continuation relaxes 20–26% at
  ρ 0.25 where the data stay flat; magnitudes about 60%.

### 1.4 M4: M3 + one shared, realised board (working theory)

One change from M3: the board is no longer an average. All population agents read from **one realised board** (list of
the W newest posts) fed by a random thinning of the population's posts at the real swarm's posting rate (n/K per
post) and by **one stochastic controller** using the simulator's rule on that board. Holder-count noise and extinction
as in M3. Chosen on development data only (G error 0.017 vs M3 0.069; D(3→6) signs 5/5 vs 4/5), then scored once.

| task003, K = 1 | M3 | M4 |
|---|---|---|
| held-out signs symmetric (data + / data −) | 13/17 (1/4, 12/13) | 13/17 (**4/4**, 9/13) |
| held-out signs no-proof | 13/16 (8/10, 5/6) | 13/16 (9/10, 4/6) |
| held-out G error | 0.12 | **0.027** |
| crossover 3→6 symmetric (data 0.5–0.625) | 0.25–0.375 | 0.625–0.75 |
| crossover 3→6 no-proof (data 0.75–1) | 0.625–0.75 | **0.75–1** |
| post-stop branch ordering (t = 40, 130) | 3/8 | **7/8** |

M4 misses (held-out): symmetric 0.625 3→6, 0.875 6→9, 0.5 9→12, 0.875 9→12; no-proof 0.375 6→9, 0.875 6→9, 1 6→9 —
all at or next to crossovers (`../../held_out_validation/scorecard_M4.csv`).

Post-stop, M4 against data (δ at t = 30 → 130):

| | normal | frozen | no communication (half-time) | removal (residual at 130) |
|---|---|---|---|---|
| symmetric ρ 0.25 | 0.40 → 0.38 (data 0.43 → 0.42) | 0.40 → 0.40 (0.43 → 0.43) | 1.0 (1.0) | +0.03 (+0.04) |
| symmetric ρ 0.75 | 0.10 → 0.09 (0.11 → 0.09) | 0.10 → 0.11 (0.11 → 0.12) | 6.0 (5.5) | −0.05 (−0.01) |
| no-proof ρ 0.25 | 0.26 → 0.26 (0.30 → 0.29) | 0.26 → 0.26 (0.30 → 0.30) | 1.0 (1.0) | −0.11 (−0.06) |
| no-proof ρ 0.75 | 0.27 → 0.29 (0.25 → 0.26) | 0.27 → 0.28 (0.25 → 0.25) | 2.5 (2.5) | −0.07 (−0.09) |

### 1.5 Interpretation (the working account)

Facts behave like alleles: board posts are copies, reading is reproduction, forgetting is death, the posting rule is
selection (agents post what supports their vote), the controller is immigration. With about 1.3 holders per fact,
**drift drives agent facts extinct**, permanently. Two consequences:
- *Persistence:* after immigration stops, the controller's facts keep circulating through rereading and reposting; the
  frozen-board and removal branches show the circulating controller-only facts are the carrier.
- *Reversal:* reading more spreads the controller's facts faster (immediate) and lets agents keep more of their own
  evidence (needs memory). M4 reproduces which side wins; the step 2b decomposition is consistent but order-dependent
  and is not used as a mechanism.

What M4 adds over M3 — a shared finite board — is a *correlation*: all readers see the same fluctuating board, so
lucky runs of controller posts reach many agents at once. **Not yet shown:** an analytical result (e.g. a reproduction
number with an extinction condition, or a neutral-drift persistence proposition); M4 is a reduced *stochastic* model,
not yet a closed-form theory.

### 1.6 Scaling: K independent projects (copies per fact constant)

World: K independent task_003 projects with one split (same person alone in every project, scores summed; three
answers). Exact engine `llmfree_core/kworld.py` (K = 1 reproduces the original exactly; tests
`../../finite_size/test_kworld.py`); setups K copies of the base setups; W, controller rate and budget × K. Runs
`2026-10-22_sim2_kprojects_K2` (1,000 episodes), `_K4` (500). Data (`../../tables/scaling_K.csv`), q = 6, t = 30:

| | G at K = 1 / 2 / 4 | D(3→6) at K = 1 / 2 / 4 |
|---|---|---|
| symmetric ρ 0.25 | 0.43 / 0.35 / 0.15 | +0.19 / +0.21 / +0.09 |
| symmetric ρ 0.5 | 0.33 / 0.39 / 0.20 | +0.05 / +0.21 / +0.12 |
| symmetric ρ 0.75 | 0.11 / 0.29 / 0.24 | −0.13 / +0.04 / +0.10 |
| symmetric ρ 1 | 0 / 0 / 0 | 0 / 0 / −0.02 |
| no-proof ρ 0.25 | 0.30 / 0.28 / 0.22 | +0.14 / +0.16 / +0.15 |
| no-proof ρ 0.5 | 0.29 / 0.31 / 0.28 | +0.09 / +0.14 / +0.17 |
| no-proof ρ 0.75 | 0.25 / 0.33 / 0.35 | +0.03 / +0.07 / +0.15 |
| no-proof ρ 1 | 0.14 / 0.27 / 0.23 | −0.05 / −0.02 / −0.03 |

- The reversal survives at every K; its boundary moves to higher memory with K.
- **Confound:** a K-project fact informs one of K projects, so single facts are weaker (P(A2) range 0.21–0.45 at K = 1,
  0.28–0.39 at K = 4) and the silent swarm is less accurate; the no-proof-K agents' pooled evidence leans to A2
  (0.42/0.58 at K = 2, 0.37/0.63 at K = 4). For one fixed decision, facts cannot multiply while each stays as
  informative and the evidence stays "just sufficient"; every scaling changes copies per fact, informativeness or total
  evidence. These should be treated as separate parameters.

Models against K (post-hoc; `../../finite_size/score_models_K.py`, `../../tables/scaling_K_models.csv`):

| | M3: D(3→6) signs | M3: G error | M4: D(3→6) signs | M4: G error |
|---|---|---|---|---|
| K = 1 | 5/7 (misses symmetric 0.5, no-proof 0.75) | 0.072 | **7/7** | **0.018** |
| K = 2 | 6/7 (misses symmetric 0.75) | 0.031 | **7/7** | **0.019** |
| K = 4 | 6/8 (misses ρ = 1 in both setups: model ≈ 0) | 0.021 | **8/8** | **0.013** |

(ρ ∈ {0.25, 0.5, 0.75, 1}, q ∈ {3, 6}, t = 30; G error = mean |model − data| over q 3 and 6; 20 realisations per cell,
10 replicas.) M4 follows the crossover as it moves up with K, including the change of sign at symmetric ρ 0.75 (−0.13 at
K = 1, +0.04 at K = 2, +0.10 at K = 4; M4 −0.17, +0.05, +0.13). M3's error falls with K: with more posts per unit time the
averaged board is a better stand-in for the realised one, which is the one thing M4 adds. These scores are post-hoc
(the K data were seen before the models were run) but no model setting was changed for them.

## Section 2. MuSR Object Placements (second task, pre-registered)

### 2.1 World and setups

MuSR (Sprague et al., arXiv 2310.16049; MIT) generates Object Placements stories from a symbolic skeleton
(`create_sequence_v2`). [`llmfree_core/op_world.py`](../../../../llmfree_core/op_world.py) reproduces it exactly: 3
people, 2 items, 3 locations (MuSR's most common number of answer options), 3 moves, each non-mover sees a move with
probability 1/3 (MuSR 0.33), at most 2 arrivals per item and location; all 617,472 stories enumerated with their
probabilities. Checked against MuSR's own generator (frequencies of the 40 most probable stories, aggregates) and
against a direct story-by-story computation (`../../object_placements/test_op_world.py`). Facts are abstract
propositions (start positions, mover/item/destination per move, who saw or missed each move, beliefs after each move
except the answer). Question: "where would Casey look for the keys?"; A0 = drawer (Casey's belief), A2 = desk (where the
keys are), A1 = shelf. 34 true facts.

Setups built with task003's rules (`experimental_setup/op_symmetric`, `op_nosolution`, README there): symmetric —
18 distinct agent facts proving A0 jointly, no single fact or pair proves it, A2 pool alone P(A2) = 0.98 (task003 0.89),
A0 pool reuses 6 agent facts (only 34 true facts); no-proof — 15 facts at exactly (0.5, 0, 0.5), each ≤ 45% alone,
pool alone (0.67, 0, 0.33) (task003's was 0.37/0.26/0.37), agents + pool (0.67, 0, 0.33), no proof.

### 2.2 Data (run `2026-10-22_sim2_op_main`, 1,000 paired episodes, 8 ρ, q 3/6/9/12)

The reading reversal exists here too (`../../object_placements/op_data.csv`): D(3→6) changes sign between ρ 0.25 and
0.375 (symmetric) and 0.625–0.875 (no-proof). G at q 6 is smaller than in task003 (symmetric 0.18 at ρ 0.125 falling to
0 at ρ 1; no-proof 0.18 → 0).

### 2.3 Pre-registered predictions against data

| | resolved signs (data + / data −) | G error | 3→6 crossover (data → model) |
|---|---|---|---|
| **M4** symmetric | 11/16 (3/3, 8/13) | **0.030** | 0.25–0.375 → **0.25–0.375** |
| **M4** no-proof | 12/15 (11/13, 1/2) | **0.019** | 0.625–0.875 → 0.625–0.75 |
| M3 symmetric | 5/16 (0/3, 5/13) | 0.066 | → none |
| M3 no-proof | 3/15 (1/13, 2/2) | 0.212 | → 0.125–0.25 |

M4 predicts every resolved D(3→6) sign. It predicts no-proof G(q = 6) within about 0.02 at every ρ (e.g. 0.174/0.171/0.161/0.152
vs data 0.175/0.165/0.161/0.137 for ρ 0.125–0.5). Symmetric G(q = 6) is right at ρ ≤ 0.375 (0.178/0.155/0.113 vs
0.175/0.150/0.106) but off at ρ 0.5–0.75 (0.106/−0.036/−0.024 vs 0.061/0.023/0.005): M4 predicts a small backfire
there that the data do not show. Its misses are small increments of the later steps (6→9, 9→12) near
zero, where M4's own noise with 20 realisations is comparable to the signal (it shows spurious sign changes there).
M3 predicts the wrong sign of G in no-proof at q = 6 for ρ ≥ 0.25 (−0.05 to −0.13: a backfire that does not occur;
data +0.17 to 0).

## 3. Open issues for the reviewer

1. **Analytical content.** M4 is a reduced stochastic model; no closed-form result yet. Candidates: branching-process
   extinction probability of a fact as a function of (ρ, q, posting share); a neutral-drift (Moran-type) persistence
   proposition; a condition for the sign of D from exposure versus integration rates.
2. **Reduction honesty.** M4 keeps the exact single-agent rules and a realised board; it drops correlations between
   agents other than those through the shared board. A reviewer should check whether M4 is "smaller than the simulator"
   in a meaningful way (population mean field for memories + finite board + integer holder counts).
3. **Model noise.** M4 predictions near D = 0 are noisy (20–40 realisations); sign errors at crossovers may partly be
   model noise. More realisations would sharpen the scorecards.
4. **Scaling confound.** K-project scaling changes informativeness; a family that varies copies per fact,
   informativeness and total evidence independently is still missing.
5. **Not tested:** the plan's deduplicated-reading defence; the LLM transfer (16 conditions, later); finite-N with
   matched worlds beyond K = 4.
6. **Known approximations in all models:** own posts not excluded from a reader's window (about 1/24 of the board);
   time step dt = 0.05; the controller's budget in M1–M3 is spent deterministically.

## 4. File index

- Rules: `theory/00_microscopic_rules.md`. Models: `theory/models/mean_field.py`, `run_models.py`, `diag_*.py`.
  Write-ups: `theory/M1_M2_mean_field.md`, `theory/M3_extinction.md`.
- Targets and split: `calibration/build_benchmarks.py`, `calibration/holdout_preregistration.md`,
  `held_out_validation/benchmark_*_held_out.csv`. Scorecards: `held_out_validation/scorecard_M{1,2,3,4}*.csv`.
- Scaling: `finite_size/` (engine tests, `build_k_setups.py`, `compare_K.py`, `score_models_K.py`); `llmfree_core/kworld.py`.
- Object Placements: `llmfree_core/op_world.py`, `object_placements/` (tests, builder, `preregistered_M{3,4}.csv`,
  `op_data.csv`, `scorecard_op_M{3,4}*.csv`).
- Model outputs (not in git): `results/key_phenomena/theory_runs/`.
