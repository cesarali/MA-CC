# Mechanism and validation: what explains the reading reversal and persistence (24 October 2026)

**For:** the researcher and the reviewer who proposed the theory-first continuation.
**What this is:** the Phase C deliverable. It collects what has been *established* by Phases A–C, separates it from
what is only *consistent*, and states what was *not* achieved. Details are in [`phaseA/`](phaseA/),
[`phaseB/`](phaseB/) and [`phaseC/`](phaseC/), each with its pre-registration. Earlier model results (M1–M4) are in
[`reports/`](reports/).

## 1. The answer in one paragraph

Both phenomena are finite-population effects of a board, a finite controller and a selective posting rule acting on
exact Bayesian agents.
- **Given the board, agents are independent**, and the evidence and memory part is solved exactly.
- **The reversal** (whether reading more helps or hurts the controller) is reproduced once two things are finite: the
  shared board, and the controller's posts. Its sign is already fixed by the beliefs agents hold in memory. Which facts
  sit *together* in a memory decides it; counting facts by kind does not.
- **Persistence** after the controller stops is set by the posting rule's **selection**: agents repost only facts that
  support their vote in the context of their memory. Neutral drift does not explain it. Where single facts carry
  evidence (task003), selection keeps the controller's imprint alive. Where evidence lives only in combinations (Object
  Placements), selection lets it fade.
- **Lasting damage** after the controller's facts are removed requires the swarm itself to be finite, so that agents'
  own facts can go extinct.

We did not obtain a closed-form condition for the sign of the reversal.

## 2. State, rules, and the exact decomposition

- **State:** the agents' memories M_i (sets of facts), their last votes, and the board (an ordered list of posts;
  agents read q of the W = 48 newest not their own).
- **Rules:** those of the simulator (`00_microscopic_rules.md`): exact posterior P(a | M), probability-matching votes,
  forgetting at rate γ = −ln ρ, posting one fact from the support set C(M, v), and the controller's best-fact rule.

**Exact statement 1 (conditional independence).** Given the board's history, the agents' memory and vote processes are
independent. Each is the single-agent Markov chain driven by its own window of the board. (The one exception: a reader
skips its own posts.) Everything that couples agents passes through the board and the controller.

**Exact statement 2 (belief identity).**
G_belief(t) = Σ_M [μ_C(M; t) − μ_S(M; t)] · P(A2 | M), where μ_C and μ_S are the memory distributions in the
controller and silent arms. It is an identity, not a theory. Its use here: Phase A1 found that this belief-level effect
has the same resolved sign as the vote-based D(3→6) at **every** resolved point, 12 of 12 over both tasks. A memory-level
theory is therefore enough for the sign of the reversal. For its size, the vote must include the facts read just before
it, since a vote is drawn from P(· | M ∪ facts just read).

## 3. Exact results for the two solvable branches (Phase A4; validated, no free parameter)

From the state at t = 30:
- **No communication** (no reading, no posting, agents still vote). Each fact of M_i(30) survives to t with probability
  e^{−γ(t−30)}. The share of votes for A2 at t is
  e^{−(t−30)} · 1[v_i(30) = A2] + ∫₀^{t−30} e^{−s} E[P(A2 | M_i(t − s))] ds,
  computed exactly by enumerating subsets of M_i(30). It matches the simulation within 0.005 at every time. Half-times
  are exact in 7 of 8 cells (13.5 against 14.25 in the eighth). It reproduces the transient rise at symmetric ρ 0.75:
  a proof breaks at rate kγ (k facts), faster than a single controller fact (rate γ).
- **Frozen board** (agents read only the board at t = 30, nobody posts). Agents are independent chains on a fixed board.
  Each fact's long-run holding share is
  m_f* = r_f / (r_f + γ), with r_f = 1 − C(W_i − k_f, q) / C(W_i, q),
  the chance of reading f in one action when k_f of the W_i visible posts carry it. This matches the simulation within
  0.002, and δ(t) matches within 0.023.

## 4. What each finite ingredient does (Phases B, C1 — matched comparisons, post-hoc)

All models below use the exact single-agent rules. They differ in what is finite.

| model | finite: holder counts / board / controller | task003 G error | OP G error | task003 sym. crossover (data 0.5–0.625) | OP sym. crossover (data 0.25–0.375) | post-stop ordering task003 / OP |
|---|---|---|---|---|---|---|
| M2 | no / no / no | 0.17 | — | 0.125–0.25 | — | — |
| M3 | yes / no / no | 0.121 | 0.139 | 0.25–0.375 | none | 3/8, 1/8 |
| M3s | yes / no / yes | 0.082 | 0.080 | 0.375–0.5 | none | 5/8, 1/8 |
| **M4b** | no / yes / yes | **0.034** | **0.032** | **0.5–0.625** | 0.25–0.375 (plus a spurious change) | 6/8, 2/8 |
| M4 | yes / yes / yes | 0.026 | 0.025 | 0.625–0.75 | 0.25–0.375 | 7/8, 7/8 |

Reading:
- **A finite controller** gives about half of the gain over the averaged model (41% task003, 52% OP).
- **A finite shared board** gives the other half, and the crossover locations.
- **Finite holder counts** add little for the reversal (M4b ≈ M4). They are needed after the controller stops: for the
  sign of the residual after removal (lasting extinction damage), and for the effect level at OP no-proof ρ 0.75.
- **Deterministic averaging of everything** (M1/M2) fails at chance.

## 5. Persistence: selection, not drift (Phase C2)

From the stored state at t = 30, normal continuation, the retained fraction δ(130)/δ(30):

| | data | board copying (neutral) | posting from memory (neutral) | real posting rule |
|---|---|---|---|---|
| OP no-proof ρ 0.25 | 0.37 | 0.02 | 0.73 | 0.42 |
| OP no-proof ρ 0.75 | 0.33 | −0.10 | 1.14 | 0.30 |
| OP symmetric ρ 0.25 | 0.67 | 0.26 | 0.90 | 0.64 |
| task003 symmetric ρ 0.25 | 0.97 | 0.18 | 0.90 | 0.98 |
| pre-registered checks passed (variety / persistence, of 7) | | 0/7, 0/7 | 2/7, 2/7 | **7/7, 7/7** |

- Neutral drift fails in both directions. Board-only copying loses variety and the effect; memory-reservoir copying
  keeps too much of both.
- **Only the selective posting rule reproduces** both task003's persistence and Object Placements' fading.

Microscopically, a held controller fact is eligible for reposting 0.78–0.82 of the time in task003 and 0.53–0.56 in
Object Placements (ρ 0.25). Object Placements' facts support a vote only in particular combinations. Facts that support
no current vote are not reposted, and the controller's imprint drains.

## 6. Hypotheses tested and rejected or revised

| hypothesis | where | outcome |
|---|---|---|
| deterministic mean field suffices (M1, M2) | original reports | rejected (held-out signs at chance) |
| fact-by-fact closure vs joint memory decides it | M1 vs M2 | rejected (both fail alike) |
| M4's gain is the shared board's correlations | theory reports | **revised**: half finite controller, half board (Phase B) |
| compact evidence state (proof, # controller facts, # agent facts) | Phase A3 | rejected: worse than the marginal predictor on task003 (4/7 against 5/7) |
| the board is the only finite object that matters | Phase C1 | rejected after the controller stops (residual sign, OP level); nearly true for the reversal |
| persistence fades by neutral drift of variety | comparison report, Hypothesis 4 | **rejected** (Phase C2): selection is needed |
| pinning agents' own facts isolates extinction | original reports | revised: it also changes how much evidence each agent holds |

## 7. What was not achieved, and why

- **No closed-form sign condition for D.** The sign of D is decided by joint memories: which facts are held together,
  and whether they complete a proof (Phase A). It is also decided by a posting propensity that depends on the whole
  memory (Phase C2). Neither reduces to per-fact rates. A reproduction number R_f = (π_f/γ)(q/p_post)(1 − m_f) can be
  written, but π_f is not a constant of the fact, so a threshold R_f = 1 does not predict the crossover.
- **The stop rule.** Phase C allowed at most three single-change revisions. One was used (C2 revision 1), plus a code
  check. We stopped there, because the next step toward a compact closure would require tracking proof-relevant fact
  combinations through the posting rule. That is a model of the same size as M4.
- **Current status:** M4 (and M4b for the reversal) is the validated stochastic surrogate. The exact pieces (§2–3) and
  the ablations (§4–5) are the explanation.

## 8. What is new beyond M4

1. The exact decomposition and the two solved branches (§2–3), validated within sampling error, with no free
   parameter.
2. The attribution of M4's gain to two finite ingredients, and the finding that the swarm's own finiteness matters
   mainly after the controller stops (§4).
3. Persistence is a selection effect of the evidence-supporting posting rule, not neutral drift (§5). This explains why
   it holds in task003 and fades in Object Placements, from the way evidence is spread over facts in each task.
4. Sign sufficiency of memory beliefs, and the failure of fact counts as an evidence state (§2; Phase A).

## 9. Paper-ready explanation (short)

In a swarm of exact Bayesian agents sharing a finite message board, a controller that posts true but selected facts
gains its power from scarcity. Facts are held by about one agent each and are read from a board of a few dozen posts,
so each controller post is a large, correlated perturbation that every reader sees. Reading more spreads the
controller's facts faster. With good memory it also lets agents reassemble the combinations of their own facts that
prove the truth, and which effect wins depends on memory: this is the reading reversal. After the controller stops,
nothing new enters, but agents keep reposting the facts that support their current vote. Where single facts carry
evidence, the controller's facts keep supporting the votes they caused and survive this selection, so the effect
persists. Where evidence lives only in combinations, they support no vote on their own and drain away.

*Status of this paragraph:* the persistence part is established by matched interventions on the posting rule
(§5). The reversal part (spreading against reassembling) is supported by the models and ablations (§4, Phase A), but
it is not derived.
