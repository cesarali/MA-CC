# Model versions M1 and M2: deterministic mean-field closures (failed), and what the failure shows

**Date:** 22 October 2026. **Code:** [`models/mean_field.py`](models/mean_field.py), runner
[`models/run_models.py`](models/run_models.py), scorecard [`../held_out_validation/score.py`](../held_out_validation/score.py).
**Status:** both versions **fail** the pre-registered scorecard. The failure is informative: it locates the missing
ingredient in finite-size stochasticity, not in the joint structure of memory.

## 1. The two models

Both replace the 24-agent system by a large population (24 × R replicas, R = 100) that interacts only through a
**deterministic board**: the expected composition of the W = 48 newest posts, fed by the population's average posting
(scaled to 24 agents) and by the controller's expected posting. Every agent uses the simulator's own rules, imported
from `sim2/simulation_2.py` (exact posterior, probability-matching vote, posting rule). The controller's posting
probabilities are estimated each step by sampling q_c agent posts from the agent-only window and applying its exact
decision rule; its budget is spent deterministically. Votes are simulated per agent, so the vote lag is exact.

- **M2 (tagged-agent closure):** every agent keeps its exact memory set, so the joint structure of an agent's evidence
  (including proofs) is kept. Dropped: correlations *between* agents and all finite-size effects.
- **M1 (fact-by-fact closure, the plan's eq. M1):** identical, except that before each action an agent's memory is
  redrawn fact by fact from the population's per-fact shares m_f. **This is the only difference from M2.**

Shared approximations (`mean_field.py` docstring): infinite population; deterministic board with per-reader
finite-board read noise; own posts not excluded; expected controller; time step dt = 0.05 (exponential lifetimes are
memoryless, so a per-step hazard 1 − ρ^dt is exact as dt → 0).

**Numerical convergence** (symmetric and nosolution, ρ = 0.5, q = 6; `run_models.py --set check`): G changes by at
most about 0.02 between R = 50, 100, 200 and between dt = 0.05 and 0.025.

## 2. Scorecard (pre-registered; evaluated once per version)

Resolved reading increments (data 95% interval excludes 0) whose sign the model gets right; tolerance |D| ≥ 0.01.

| version | world | development | held-out: all | held-out: data + | held-out: data − | G error (held-out, mean abs.) |
|---|---|---|---|---|---|---|
| M1 | symmetric | 2/2 | 9/17 | 1/4 | 8/13 | 0.145 |
| M1 | nosolution | 1/3 | 6/16 | 2/10 | 4/6 | 0.181 |
| M2 | symmetric | 1/2 | 8/17 | 1/4 | 7/13 | 0.151 |
| M2 | nosolution | 2/3 | 8/16 | 4/10 | 4/6 | 0.179 |

Crossovers (data against M2): symmetric 3→6 at ρ 0.5–0.625 against 0.125–0.25; nosolution 3→6 at 0.75–1 against
0.375–0.75; 6→9 nosolution 0.5–0.875 against none. Both models are at chance on held-out signs and underestimate G
roughly threefold. Cell-level tables: `../held_out_validation/scorecard_M1*.csv`, `scorecard_M2*.csv`.

## 3. Diagnosis: is it a bug, the closure, or the large-N limit?

1. **Not a bug.** A diagnostic variant of the same code with a *real* finite board (actual posts, own posts excluded,
   one stochastic controller, N = 24; `board="stochastic"`) reproduces the simulator at ρ = 0.5, q = 6 (300 paired
   episodes): G = 0.322 ± 0.017 (data 0.33) symmetric and 0.288 ± 0.013 (0.29) nosolution; silent A2 share, controller
   messages and board share match too (`models/diag_stoch_check.py`). Remaining small rule-timing differences: at
   symmetric ρ = 0.75 the reimplementation gives D(3→6) = −0.079 ± 0.017 against −0.126 in the data.
2. **Not the joint-memory closure.** M1 and M2 fail alike, so keeping or dropping the joint structure of memory is not
   what decides the score.
3. **The large-N limit.** Scaling the stochastic model to N = 24R with W, controller rate and budget × R (the plan's
   §7 convention; `models/diag_scale_check.py`), G at ρ = 0.5, q = 6 falls toward the mean-field value:

| N | 24 | 48 | 96 | 192 | mean field |
|---|---|---|---|---|---|
| symmetric G | 0.322 | 0.204 | 0.144 | 0.137 | ≈ 0.11 |
| nosolution G | 0.288 | 0.245 | 0.219 | 0.194 | ≈ 0.12 |
| silent A2, symmetric / nosolution | 0.22 / 0.47 | 0.17 / 0.51 | 0.15 / 0.55 | 0.14 / 0.56 | 0.15 / 0.59 |

   **Correction (researcher's note, 22 October): this scaling multiplies agents but not facts.** Every fact starts with
   about 1.3 × R holders instead of 1.3, so what falls with "N" may be falling with **copies per fact (redundancy)**,
   not with swarm size. A proper size scaling needs proportionally more distinct facts (new worlds per scale), which
   the current task generator cannot produce (it is fixed at 9 hidden numbers and 49 true facts; see
   `M3_extinction.md`, section on scaling). The correct reading of this table is therefore: *the effect falls when each
   fact is held by more agents*.
4. **The reversal itself survives at N = 96** (`models/diag_reversal_N.py`, D(3→6), 300 / 150 paired episodes):

| cell | N = 24 | N = 96 |
|---|---|---|
| symmetric ρ = 0.25 | +0.198 ± 0.016 | +0.041 ± 0.014 |
| symmetric ρ = 0.75 | −0.079 ± 0.017 | −0.106 ± 0.008 |
| nosolution ρ = 0.25 | +0.149 ± 0.015 | +0.117 ± 0.014 |
| nosolution ρ = 1 | −0.080 ± 0.014 | −0.063 ± 0.011 |

   The protective side is not a finite-size effect; the amplifying side in symmetric largely is (it shrinks fivefold).

## 4. Consequence for the theory

A deterministic theory (infinite copies of every fact) cannot reproduce the observed system, whatever its memory
closure: it misses most of the effect's size and gets held-out signs at chance. Whether a large swarm with
proportionally more distinct facts behaves like N = 24 is open (see the correction in section 3). The next version must keep **finite-N stochasticity**.
The leading candidate for the finite-size carrier is the extinction of agent-origin facts (section 5).

## 5. Extinction test

In the N = 24 stochastic model, agents are made never to forget their **own starting fact** (`pin_own=True`), so no
agent-origin fact can disappear from the swarm (`models/diag_pin_check.py`; q = 6, 300 paired episodes per cell):

| cell | G, normal | G, pinned | silent A2, normal → pinned |
|---|---|---|---|
| symmetric ρ = 0.25 | 0.441 | 0.089 | 0.27 → 0.19 |
| symmetric ρ = 0.5 | 0.322 | 0.074 | 0.22 → 0.14 |
| nosolution ρ = 0.25 | 0.304 | 0.197 | 0.44 → 0.53 |
| nosolution ρ = 0.5 | 0.288 | 0.212 | 0.47 → 0.56 |

- **Symmetric:** preventing extinction removes about 80% of the effect, below even the mean-field value. The loss of
  agent facts from a finite swarm is the main carrier of the controller's power when a proof exists.
- **Nosolution:** it removes about a third; the rest survives without extinction.
- **Caveat:** pinning also gives every agent one fact it always holds, so it is not a pure "no extinction" switch.

**Next version (proposal, phase 3): M3, finite-swarm theory centred on fact extinction.** One change from M2: make
the fact population finite. The candidate reduced description is a birth–death process for the copy number of each
fact (holders and board copies), with rates from the exact single-agent rules (as in M2) and absorption at zero copies.
It gives analytical handles: each fact's reproduction number (posting × reading × memory lifetime) and its extinction
probability as functions of ρ and q, and therefore a candidate account of the reversal (reading rescues rare agent
facts from extinction and also spreads the controller's facts) and of persistence (extinct agent facts never return).
