# Phase A — results (23 October 2026)

Pre-registration: [`00_preregistration.md`](00_preregistration.md) (commit `6a2425c`; one clarification logged before
any memory data were read). Evidence map committed before the data were read (`06e3ef2`). Stored data only; no new
simulations. Every expectation of the pre-registration is marked **met** or **not met** below.

## Summary

- **A1.** The exact beliefs in the stored memories at t = 30 give the sign of the reading increment D(3→6) right at
  every resolved point: 12/12 over both tasks and setups. A theory of memories alone can therefore predict the
  reversal's sign; vote timing is not needed for the sign. It does change the size: the belief-based effect is
  30–100% of the vote-based one, because a vote is cast right after reading and includes the facts just read.
- **A2/A3.** The compact evidence map (z, c, a) — "proves A0 / how many controller facts / how many agent facts" — is
  **not enough**. Its error against the exact belief is 0.04–0.20 per agent, and as a predictor of the sign of D it does
  worse than the marginal predictor in task003 (4/7 against 5/7). The marginal predictor fails on Object Placements as
  expected (1/5). The missing information is **which** facts sit together in a memory.
- **A4.** The no-communication and frozen-board branches are reproduced from the t = 30 state with no free parameter.
  The no-communication decay is computed in closed form and matches to within 0.005. The frozen board is reproduced by
  independent single-agent chains, and each fact's long-run holding share follows r_f/(r_f + γ) to within 0.002. All
  coupling between agents passes through the board; given the board, the memory and evidence part of the problem is
  solved exactly.

## A1 — belief versus vote (`memory_predictors.py`, `memory_predictors.csv`)

D(3→6) at t = 30; episodes paired across arms and q; "sign" = resolved 95% interval of the vote-based D.

| task | setup | ρ | G(q=6) vote | G(q=6) belief | D vote (sign) | D belief |
|---|---|---|---|---|---|---|
| task003 | symmetric | 0.25 | 0.432 | 0.173 | +0.193 (+) | +0.079 |
| task003 | symmetric | 0.5 | 0.330 | 0.204 | +0.045 (+) | +0.039 |
| task003 | symmetric | 0.75 | 0.114 | 0.128 | −0.126 (−) | −0.071 |
| task003 | symmetric | 1 | 0 | 0 | 0 (0) | 0 |
| task003 | no-proof | 0.25 | 0.301 | 0.120 | +0.137 (+) | +0.055 |
| task003 | no-proof | 0.5 | 0.294 | 0.165 | +0.092 (+) | +0.050 |
| task003 | no-proof | 0.75 | 0.252 | 0.189 | +0.033 (+) | +0.020 |
| task003 | no-proof | 1 | 0.144 | 0.142 | −0.047 (−) | −0.047 |
| Obj. Plac. | symmetric | 0.25 | 0.150 | 0.064 | +0.018 (+) | +0.011 |
| Obj. Plac. | symmetric | 0.5 | 0.061 | 0.051 | −0.058 (−) | −0.023 |
| Obj. Plac. | symmetric | 0.75 | 0.005 | 0.031 | −0.044 (−) | −0.022 |
| Obj. Plac. | symmetric | 1 | 0 | 0 | 0 (0) | 0 |
| Obj. Plac. | no-proof | 0.25 | 0.165 | 0.049 | +0.083 (+) | +0.021 |
| Obj. Plac. | no-proof | 0.5 | 0.137 | 0.062 | +0.055 (+) | +0.025 |
| Obj. Plac. | no-proof | 0.75 | 0.079 | 0.054 | −0.008 (0) | −0.004 |
| Obj. Plac. | no-proof | 1 | −0.004 | −0.005 | 0 (0) | 0 |

Belief signs right: task003 7/7, Object Placements 5/5 (every resolved point). The belief effect is smallest relative to
the vote effect at short memory (ρ = 0.25: about 40%), where memories empty between actions and the vote is driven by
what was just read.

## A2 — compact evidence map (`evidence_map.py`, `evidence_map.csv`)

Mean |Φ̂(s(M)) − P(A2 | M)| over the stored memories, by (task, setup): 0.044–0.112 in Object Placements, 0.080–0.202
in task003 (largest in task003 no-proof at ρ 0.75). The map averages over which facts are held; real memories are far
from uniform subsets, so its responses are biased.

## A3 — predictors of the sign of D(3→6)

| predictor | task003 (7 resolved) | Object Placements (5 resolved) |
|---|---|---|
| exact belief (reference, A1) | 7 | 5 |
| marginal: Σ_f (m_f^C − m_f^S)(P(A2 \| {f}) − 1/3) | 5 | 1 |
| evidence state (z, c, a) | 4 | 3 |

Pre-registered expectations:
- *The marginal predictor fails on Object Placements* — **met** (1/5; single-fact pulls are about 0 there).
- *The evidence-state predictor gets at least as many signs right as the marginal one in each task* — **not met**
  (task003: 4 against 5; it gets every task003 no-proof point negative). On Object Placements it does better (3 against 1).
- Decision rule: the (z, c, a) closure is reported as **insufficient**. It was not tuned on these points.

## A4 — the two solvable branches (`solvable_branches.py`, `solvable_branches_summary.csv`, `frozen_holding_check.csv`)

Predictions from the state at t = 30 only (memories, last votes, board), compared with the data at every 0.25 from
t = 30.25 to 130 (both tasks, both setups, ρ 0.25 and 0.75, q 6).

| branch | method | agreement |
|---|---|---|
| no communication | closed form: facts survive independently with e^{−γs}; votes at rate 1 from the current memory | within 2 se at 97–100% of the times where the data vary; largest difference 0.005; half-time exact in 7 cells, 13.5 against 14.25 in the 8th; reproduces the transient rise at symmetric ρ 0.75 (OP 0.027 against 0.027; task003 0.122 against 0.125) |
| frozen board | one independent chain per agent, simulator rules, own frozen window | within 2 se (data se only) at 88–99% of times; largest difference 0.023 |
| frozen board, per fact | analytic long-run holding share r_f/(r_f + γ) | mean error 0.0007–0.0018 against the data at t = 130 (chains: 0.0012–0.0020) |

Expectation: *these reductions are exact given the t = 30 state and match within sampling error* — **met**. The
"within 2 se" counts use the data's standard error only; the chains add their own noise, so they understate agreement.

Two bugs in my own diagnostics were found and fixed before this write-up: the chains' holding count at t = 130 included
facts forgotten after the agent's last action (it showed an error of 0.1); and the no-communication share-within-2-se
counted times at which the data are exactly 0 (standard error 0) as misses. Neither affected the δ(t) predictions.

## What this means for Phase C

1. A memory-level theory is sufficient for the sign of D (A1). Vote timing changes the size and must be modelled for
   quantitative G: a vote is P(· | memory ∪ facts just read).
2. The evidence state must keep **which facts are held together**, at least for the proof-relevant combinations
   (symmetric: the minimal proofs, 12 in task003 and 2 in Object Placements; no-proof: classes of the exact belief).
   Counting facts by kind is not enough (A3).
3. Given the board, agents are independent and the memory/evidence half is solved exactly (A4), including the
   analytic holding share r_f/(r_f + γ). The open modelling problem is the **board**: how its composition evolves when
   agents post (normal continuation) and while the controller acts.

Next: Phase B (matched M3/M4 attribution), then Phase C.
