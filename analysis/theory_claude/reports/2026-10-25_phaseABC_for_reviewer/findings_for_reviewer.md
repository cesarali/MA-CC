# Theory continuation after review: findings of Phases A–C and proposed next steps

**For:** the reviewer agent who wrote `M4_theory_review_and_analytical_continuation.md`. You should check every claim
below and propose corrections or other candidate models.
**Date:** 25 October 2026. **Branch:** `dev/rsanchez` (commits named below). **Scope:** Simulation 2, task003 and
MuSR Object Placements (OP), both setups (symmetric: the agents can jointly prove the correct answer A0; no-proof:
they cannot).
**Earlier material:**
- `../2026-10-22_theory_review/theory_review_report.md` (M1–M4; with dated addenda of 23 and 24 Oct);
- `../2026-10-23_theory_report/theory_report.pdf`;
- `../2026-10-23_task_comparison/task_comparison.pdf`;
- the Phase C deliverable `../../theory_mechanism_and_validation.md`.

How to read this document:
- **Established**: an exact statement, or a test fixed in advance and passed.
- **Consistent**: agrees with the data but was not tested as a prediction.
- **Rejected**: a test fixed in advance and failed.
- Every phase was pre-registered: rules and expected outcomes were committed before the comparison with data.

---

## 0. Notation

- **Memory and posterior.** Agent i (N = 24) remembers a set of true facts M_i. The exact posterior is
  P(a | M) for a ∈ {A0, A1, A2}, and Φ₂(M) ≡ P(A2 | M).
- **Forgetting.** Each fact is forgotten at rate γ = −ln ρ.
- **An action.** The agent reads q posts S drawn without replacement from the W = 48 newest posts on the board,
  excluding its own posts. It then votes v ~ P(· | M ∪ F(S)), where F(S) are the facts of S.
- **Posting.** The agent posts one fact drawn uniformly from the support set
  C(M, v) = {f ∈ M : P(v | M) > P(v | M \ {f})}, preferring facts not just read. If C is empty, a fallback applies.
- **Controller.** It posts the pool fact that most raises P(A2), at rate 1 until t = 30.
- **Arms.** C (with controller) and S (silent) share all random streams.
- **Measures.**
  - G = E[x_A2^C − x_A2^S] at t = 30, where x_A2 is the share of A2 votes among agents who have voted;
  - D(q₁→q₂) = G(q₂) − G(q₁);
  - δ(t), the same difference for t > 30 after the controller stops.
- **Continuations after the stop.** normal; frozen board (agents read only the t = 30 board, nobody posts); no
  communication (no reading, no posting, agents still vote); controller facts removed.

---

## 1. Summary of findings

| # | finding | status | phase |
|---|---|---|---|
| 1 | Given the board's history, agents are independent | established (exact; validated by #4) | A |
| 2 | Memory beliefs at t = 30 give the sign of D(3→6) at every resolved point (12/12) | established | A1 |
| 3 | Counting facts by kind (proof / #controller / #agent facts) does not predict the sign of D | rejected closure | A2–A3 |
| 4 | No-communication and frozen-board branches follow exactly from the t = 30 state | established | A4 |
| 5 | M4's gain over M3 is about half a finite controller, half a finite shared board | established (post-hoc ablation) | B |
| 6 | A finite board + finite controller nearly suffice for the reversal; finite holder counts are needed after the stop | partly rejected (pre-registered rule not met) | C1 |
| 7 | Persistence and its fading in OP are a **selection** effect of the posting rule, not neutral drift | established | C2 |
| 8 | No closed-form condition for the sign of D was obtained | not achieved | C |

---

## 2. Phase A — stored data only (pre-registration `6a2425c`; map committed before data `06e3ef2`; results `a36e789`)

### A1 — the belief identity and sign sufficiency

$$G_{\text{belief}}(t)=\sum_M\big[\mu_C(M;t)-\mu_S(M;t)\big]\,\Phi_2(M)$$

μ_C and μ_S are the memory distributions over agents and episodes. This is an identity, not a theory. The result is
that sign(D_belief) = sign(D_vote) at all 12 resolved points (task003 7/7, OP 5/5). A memory-level theory therefore
suffices for the sign. It does not suffice for the size: G_belief/G_vote ≈ 0.4 at ρ = 0.25, because a vote is drawn
from P(· | M ∪ F(S)), and at short memory the facts just read dominate.

### A2–A3 — two predictors of sign D(3→6), fixed in advance

- Marginal predictor:
  $$\hat G_{\text{marg}}=\sum_{f\in U}(m_f^C-m_f^S)\big(P(A2\mid\{f\})-\tfrac13\big)$$
  where U is the set of circulating facts and m_f the share of agents holding f.
- Evidence-state predictor:
  $$\hat G_{\text{state}}=\sum_s[\mu_C(s)-\mu_S(s)]\hat\Phi_2(s),\quad s=(z,c,a)$$
  where z = 1 if M proves A0, c = number of controller-only facts (capped at 4), a = number of agent facts (capped at 6).
  Φ̂₂(s) is the mean of Φ₂ over memories drawn uniformly with that (c, a), computed from the world alone.

| predictor | task003 (7 resolved) | OP (5 resolved) |
|---|---|---|
| exact belief (reference) | 7 | 5 |
| marginal | 5 | 1 |
| (z, c, a) | 4 | 3 |

Expectations stated in advance:
- *The marginal predictor fails on OP*: met, because single-fact pulls there are ≈ 0.
- *The state predictor is at least as good as the marginal one in each task*: **not met** in task003.

|Φ̂₂ − Φ₂| per agent is 0.04–0.20 for ρ < 1. By the pre-registered rule the closure is insufficient, and it was not
tuned. Real memories are far from uniform subsets: which facts are held **together** decides.

### A4 — the two solvable branches (no free parameter)

**No communication.** Let s = t − 30, k = |M_i(30)| and π(τ) = e^{−γτ}. Then

$$\mathbb E[x_{A2,i}(t)]=e^{-s}\mathbf 1[v_i(30)=A2]+\int_0^s e^{-u}\sum_{j=0}^{k}\pi(s-u)^j(1-\pi(s-u))^{k-j}A_j\,du,\qquad A_j=\sum_{S\subseteq M_i(30),|S|=j}\Phi_2(S)$$

The integral is evaluated as a convolution on a 0.01 grid; it agrees with a direct integral to 2·10⁻⁶. Agreement with
the simulation, from t = 30.25 to 130 in 8 cells (2 tasks × 2 setups × ρ 0.25 / 0.75, q 6):
- largest difference 0.005;
- within 2 data standard errors at 97–100% of the times where the data vary;
- half-time exact in 7 cells, 13.5 against 14.25 in the 8th.

It also reproduces the transient rise of δ at symmetric ρ 0.75 (OP 0.027 against 0.027; task003 0.122 against 0.125).
A proof of k facts breaks at rate kγ, a single controller fact at rate γ.

**Frozen board.** Each agent is an independent Markov chain on a fixed window. Each fact's long-run holding share is

$$m_f^*=\frac{r_f}{r_f+\gamma},\qquad r_f=1-\binom{W_i-k_f}{q}\Big/\binom{W_i}{q}$$

where k_f is the number of copies of f among the W_i posts agent i can see. The mean error against the data at t = 130
is 0.0007–0.0018. δ(t) from independent single-agent chains is within 0.023 everywhere, and within 2 data standard
errors at 88–99% of times. The data's standard error alone understates agreement, since the chains add their own noise.

**Consequence.** The memory and evidence half of the problem is solved given the board. The open modelling problem is
the board's evolution under posting and control.

*Two bugs in my own diagnostics were found and fixed before reporting (documented in `../../phaseA/phaseA_results.md`).
Neither touched the δ(t) predictions.*

---

## 3. Phase B — attribution of the M3 → M4 gain (plan `1a32d53` before any run; results `058a3b8`; post-hoc)

M4 changed two things relative to M3: a shared realised board, and one stochastic controller. The matched model **M3s**
is M3 plus M4's stochastic controller, with M3's averaged board.

$$f_{\text{controller}}=\frac{e_{M3}-e_{M3s}}{e_{M3}-e_{M4}}=0.41\ \text{(task003 held-out)},\quad 0.52\ \text{(OP)}$$

Here e is the mean |G_model − G_sim|. The finite controller gives about half of the gain; the shared board gives the
rest, plus the OP crossover and the post-stop ordering (M3s 5/8 and 1/8; M4 7/8 and 7/8).

---

## 4. Phase C — mechanism (pre-registration `bece064`; revision logged `9ee5ea0`; results `0b82951`, `c93dec4`)

### C1 — is the board's finiteness enough? M4b = M4 without integer holder counts (post-hoc)

M4b is an unlimited population of exact agents (R = 30 copies of the swarm) reading one shared finite board, with one
finite controller.

| model | finite: holders / board / controller | task003 G error | OP G error | task003 sym. 3→6 crossover (data 0.5–0.625) | post-stop order task003 / OP |
|---|---|---|---|---|---|
| M3 | ✓ / – / – | 0.121 | 0.139 | 0.25–0.375 | 3/8, 1/8 |
| M3s | ✓ / – / ✓ | 0.082 | 0.080 | 0.375–0.5 | 5/8, 1/8 |
| **M4b** | – / ✓ / ✓ | **0.034** | **0.032** | **0.5–0.625** | 6/8, **2/8** |
| M4 | ✓ / ✓ / ✓ | 0.026 | 0.025 | 0.625–0.75 | 7/8, 7/8 |

Rule fixed in advance: errors ≤ 0.04 in both tasks **and** ordering ≥ 6/8 in both. **Not met** (OP 2/8). The diagnosis:
- M4b reproduces OP's fading (no-proof ρ 0.25: 0.151 → 0.058; data 0.165 → 0.061).
- It misses the level at OP no-proof ρ 0.75 (0.024 against 0.079).
- It misses the sign of the residual after removal at symmetric ρ 0.25 (+0.013 against −0.031). That residual is the
  lasting extinction damage that only finite holder counts can produce.

The swarm's own finiteness therefore matters mainly after the stop.

### C2 — what makes the effect persist or fade under normal continuation

Each episode was continued from its stored t = 30 state (memories, last votes, board with authors) to t = 130, with
three posting rules. The checks were fixed in advance, over 7 scored cells:
1. distinct controller-only facts per board within 25% of the data at t = 80 and 130;
2. retained fraction δ(130)/δ(30) within 0.2 of the data.

| new post | variety | persistence | OP no-proof ρ 0.25 (data 0.37) | OP no-proof ρ 0.75 (data 0.33) | task003 sym. ρ 0.25 (data 0.97) |
|---|---|---|---|---|---|
| NBD: f ~ Unif(board posts), neutral | 0/7 | 0/7 | 0.02 | −0.10 | 0.18 |
| N2R: f ~ Unif(M), prefer not just read; neutral | 2/7 | 2/7 | 0.73 | 1.14 | 0.90 |
| real rule: f ~ Unif(C(M, v)) | **7/7** | **7/7** | 0.42 | 0.30 | 0.98 |

- Copying straight from the board drifts far too fast.
- Neutral posting from memories keeps too much variety, and in OP the effect does not fade.
- **Only the selective rule reproduces both tasks.** This refutes the drift explanation (Hypothesis 4) of the 23 Oct
  comparison report; a dated addendum corrects it.

**Microscopic support.** For a held controller-only fact,

$$e=P\big(f\in C(M,v)\mid f\in M\big)\approx 0.78\text{–}0.82\ \text{(task003)},\quad 0.53\text{–}0.56\ \text{(OP)},\qquad \rho=0.25.$$

Controller facts per memory fall by 10% in task003 and 22% in OP from t = 30 to 130. In OP, single facts support a vote
only in particular combinations (median single-fact P(A2) = 1/3 exactly; task003 0.35–0.42). This is consistent with
the result but is not a derivation, because e depends on the whole memory.

### Why no closed form for the sign of D

The per-fact reproduction number

$$\mathcal R_f\approx\frac{\pi_f}{\gamma}\cdot\frac{q}{p_{\text{post}}}\cdot(1-m_f)$$

has π_f, the chance that a holder posts f, defined through C(M, v). It is not a constant of the fact. A threshold
R_f = 1 therefore does not locate the crossover. The stop rule (at most three single-change revisions) was applied after
one revision, because the next compact step would need to track proof-relevant combinations through the posting rule.
That is a model of M4's size.

---

## 5. Corrections made to earlier claims (all as dated addenda; original text unchanged)

1. M4's gain was attributed to the shared board alone: corrected to about half controller, half board (Phase B).
2. M4 equals M3 on held-out *sign counts* (13/17, 13/16). M4's gain is in sizes.
3. The pinned-fact diagnostic does not isolate extinction: it also changes how much evidence each agent holds.
4. M4's OP no-communication half-times are exact in 2 of 4 cells, not all (1.25 against 1.0 at symmetric ρ 0.25).
5. Hypothesis 4 of the comparison report (drift drives the fading) is rejected; selection by the posting rule drives it.

---

## 6. Known limitations and open questions for the reviewer

- **Held-out exposure.** The held-out task003 data had been seen in earlier descriptive reports. The OP reversal and
  the OP persistence predictions were committed before their data (`c0d22bc`, `c9d66c2`, `09208a7`). Phases B and C1
  are post-hoc ablations.
- **OP setups differ from task003 by construction.** The OP no-proof pool alone favours the truth (0.67/0.33 against
  task003's balanced 0.37/0.37). Single OP facts are mostly neutral. Rebuilding the pools is deferred (TODO). Some
  task differences may reflect pool design rather than task type.
- **Model sampling noise.** M4 and M4b predictions near D = 0 use 20–40 realisations; some sign errors at crossovers
  may be model noise.
- **Vote timing.** Sizes need the facts read just before the vote. No reduced theory of G's size has been attempted
  beyond M4.
- **Questions:**
  - Is there a better compact evidence state than proof progress?
  - Does the support-set selection admit a mean-field description in terms of a small number of evidence classes?
  - Is the C1 rule (≥ 6/8 ordering) too strict, given near-ties among branches?

---

## 7. Proposed next steps (in order of value per effort; none started)

1. **A proof-progress closure for the sign of D (one more single-change attempt at a compact model).**
   - **State:** for each agent, the progress vector over the minimal proofs of A0 (symmetric: 12 proofs of 6 facts in
     task003, 2 of 3 in OP), plus the controller-only facts held.
   - **Response:** exact Φ₂ on that state.
   - **Dynamics:** the board-driven single-agent chain of A4, with the board composition taken from M4b.
   - **Test:** the 16 points of Phase A3, with predictors fixed before computing.
   - **Success:** at least the exact-belief sign count minus one in each task, without tuning.
   - **If it fails:** close the compact-theory question and say so in the final report.
2. **A selection-aware theory of persistence.** Predict the retained fraction δ(130)/δ(30) from task structure:
   - estimate the repost eligibility e of controller facts *from the world* (the expected support-set membership under
     memories drawn from the frozen-board stationary distribution m_f* of §2), not from the dynamics;
   - test against the 7 C2 cells, fixed in advance.

   This is the most promising route to a compact statement, of the form "persistence ⇔ e above a threshold".
3. **Out-of-sample test with rebuilt OP pools** (deferred TODO; new setups next to the existing ones).
   - A balanced no-proof pool.
   - A symmetric pool of individually A2-leaning facts.

   Pre-register M4's and step 2's predictions, especially that persistence returns when controller facts carry evidence
   on their own. Then run the reversal and persistence grids. This directly tests the selection explanation.
4. **A theory of vote timing for G's size:** combine the A4 machinery with the just-read facts, so that
   G_vote ≈ E[Φ₂(M ∪ F(S))] differences. Test it against vote-based G at t = 30 on the Phase A cells.
5. **Documentation.** A LaTeX/PDF version of the mechanism note (`../../theory_mechanism_and_validation.md`) for
   circulation, after the reviewer's comments.
6. **Later, unchanged:** the deduplicated-reading defence; the LLM transfer test (16 conditions; LLM code changes
   first).

---

## 8. File index (paths relative to `analysis/`)

- Phase A: `theory/phaseA/` — `00_preregistration.md`, `evidence_map.py` (+ `.csv`), `memory_predictors.py`
  (+ `.csv`), `solvable_branches.py` (+ `solvable_branches_summary.csv`, `frozen_holding_check.csv`),
  `phaseA_results.md`.
- Phase B: `theory/phaseB/` — `00_plan.md`, `score_attribution.py`, `attribution.csv`, `phaseB_results.md`.
- Phase C: `theory/phaseC/` — `00_preregistration.md` (with change log), `neutral_board_drift.py`,
  `two_reservoir_drift.py`, `repost_propensity.py`, `nbd_summary.csv`, `n2r_summary.csv`, `support_summary.csv`,
  `repost_propensity.csv`, `phaseC_results.md`; M4b comparison in `theory/phaseB/comparison_M3_M3s_M4b_M4.csv`.
- Models: `theory/models/mean_field.py` (closures M1, M2, M3, M3s, M4, M4b), `run_models.py`; outputs in
  `theory/results/theory_runs/` (not in git).
- Mechanism note: `theory/theory_mechanism_and_validation.md`.
- Data: task003 runs `task003_llm_free/results/simulation_2/2026-10-20_sim2_{reading_mechanism,persistence_branches}`;
  OP runs `object_placements_llm_free/results/simulation_2/2026-10-2{2_sim2_op_main,3_sim2_op_post_stop}` (not in git).
- Decision log: `task003_llm_free/simulation_2/key_phenomena_hypothesis/progress.md`.
