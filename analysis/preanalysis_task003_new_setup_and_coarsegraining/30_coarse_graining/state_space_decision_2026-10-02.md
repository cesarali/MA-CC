# Choosing the state space: 4 states, not 8

**2 October 2026.** Addendum to
[`four_tests_report_2026-10-02.md`](four_tests_report_2026-10-02.md), which found
that at ρ = 1 an 8-state augmented chain (`e × mean_nfacts`) is Markovian in both
arms while the 4-state chain is not. Is 8 states affordable?

## The data budget

| ρ | arm | transitions | 4 states: cells used | rows with <30 obs | 8 states: cells used | rows with <30 obs |
|---|---|---:|---:|---:|---:|---:|
| 0.75 | silent | 1,680 | 7/16 | **0** | 25/64 | 1 |
| 0.75 | A2 | 4,718 | 9/16 | **0** | 31/64 | **0** |
| 1.00 | silent | 1,680 | 9/16 | 1 | 25/64 | **3** |
| 1.00 | A2 | 2,352 | 11/16 | **0** | 29/64 | **3** |
| 1.00 | A0 | 4,060 | 11/16 | **0** | 31/64 | 1 |

"Rows with <30 obs" counts macrostates whose outgoing distribution is estimated
from fewer than 30 transitions.

## Decision: use 4 states everywhere

At 8 states, **every ρ = 1 cell has 3 thinly-estimated rows** — precisely the
cells where the 8-state chain looked best on the Markov test. Entropies and
mutual informations are nonlinear functionals and are biased upward when rows are
sparse, so the apparent gain in Markovianity would be paid for in estimator bias
on the quantities we actually want.

At 4 states nearly every cell is comfortably estimated (0–1 thin rows), and only
3–4 of 4 states are visited in any arm, so the effective chain is smaller still.

**The 8-state result stands as a finding** — a second coordinate does restore the
Markov property at ρ = 1 — **but is not adopted for estimation.** It tells us what
the missing structure is; it does not pay for itself when the goal is entropy and
mutual information at this sample size.

If a future batch has 240 initializations rather than 60, the budget changes by a
factor of four and 8 states becomes comfortable. Worth revisiting then.

## Consequence for the estimates

Everything in [`../40_information_estimates/`](../40_information_estimates/) uses
the **4-state** chain, with validity differing by cell: fully Markovian only at
ρ = 1 silent, approximate elsewhere, carrying ~0.015 nats/step of unexplained
memory at ρ = 0.75.
