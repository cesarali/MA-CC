# Phase C — results of C1 and C2 (24 October 2026)

Pre-registration and change log: [`00_preregistration.md`](00_preregistration.md) (rules `bece064`; revision 1 logged
`9ee5ea0` before it ran).

## C1 — is the board's finiteness enough? M4b = M4 without integer holder counts (post-hoc)

| | M3 | M3s | **M4b** | M4 | rule |
|---|---|---|---|---|---|
| task003 held-out G error | 0.121 | 0.082 | **0.034** | 0.026 | ≤ 0.04: met |
| Object Placements G error | 0.139 | 0.080 | **0.032** | 0.025 | ≤ 0.04: met |
| task003 held-out signs, symmetric / no-proof | 13/17, 13/16 | 14/17, 11/16 | 14/17, 13/16 | 13/17, 13/16 | — |
| Object Placements signs, symmetric / no-proof | 5/16, 3/15 | 6/16, 9/15 | 9/16, 12/15 | 11/16, 12/15 | — |
| task003 symmetric 3→6 crossover (data 0.5–0.625) | 0.25–0.375 | 0.375–0.5 | **0.5–0.625** | 0.625–0.75 | — |
| task003 no-proof 3→6 crossover (data 0.75–1) | 0.625–0.75 | 0.625–1 | 0.75–1 | 0.75–1 | — |
| post-stop ordering, task003 (of 8) | 3 | 5 | 6 | 7 | ≥ 6: met |
| post-stop ordering, Object Placements (of 8) | 1 | 1 | **2** | 7 | ≥ 6: **not met** |

**Decision (pre-registered rule): not met.** The swarm's own finiteness (integer holder counts) is also needed.

Where it is needed (Object Placements post-stop, δ(30) → δ(130)):
- **Not for the fading.** M4b reproduces it: no-proof ρ 0.25 gives 0.151 → 0.058 (data 0.165 → 0.061).
- **For the level at no-proof ρ 0.75:** M4b gives δ(30) = 0.024, against 0.079 in the data and 0.090 in M4.
- **For the sign of the residual after removal of the controller's facts:** symmetric ρ 0.25 gives +0.013 in M4b,
  against −0.031 in the data and −0.014 in M4. This is where lasting damage to the agents' own facts (extinction)
  matters, and only finite holder counts can lose a fact for good. The ordering score then fails through near-ties.

For the reversal itself (t ≤ 30), a finite shared board plus a finite controller is nearly enough. M4b even gets
task003's symmetric crossover exactly, where M4 is one step off.

## C2 — what makes the effect persist or fade under normal continuation?

Normal continuation from the stored state at t = 30, both tasks, both setups, ρ 0.25 / 0.75, q 6 (7 scored cells).
Checks fixed in advance:
1. distinct controller-only facts per board within 25% of the data at t = 80 and 130;
2. retained fraction δ(130)/δ(30) within 0.2 of the data.

| model | what it keeps | variety check | persistence check |
|---|---|---|---|
| NBD: new posts copy a random board post | board only, neutral | 0/7 | 0/7 |
| N2R (revision 1): new posts are a random fact of the poster's memory (preferring not just read) | board + memories, neutral | 2/7 | 2/7 |
| the simulator's posting rule (support in context; check of the code path) | board + memories + selection | **7/7** | **7/7** |

Retained fraction δ(130)/δ(30), selected cells:

| cell | data | NBD | N2R | real rule |
|---|---|---|---|---|
| Object Placements no-proof ρ 0.25 | 0.37 | 0.02 | 0.73 | 0.42 |
| Object Placements no-proof ρ 0.75 | 0.33 | −0.10 | 1.14 | 0.30 |
| Object Placements symmetric ρ 0.25 | 0.67 | 0.26 | 0.90 | 0.64 |
| task003 symmetric ρ 0.25 | 0.97 | 0.18 | 0.90 | 0.98 |
| task003 no-proof ρ 0.75 | 1.05 | 0.20 | 0.70 | 1.00 |

What the three models show:
- **Copying straight from the board** drifts far too fast. Each board fixes on one fact within about 50 time units, and
  the effect is lost in both tasks.
- **Neutral posting from memories** keeps *too much* variety, because memories are a second reservoir. In Object
  Placements the effect then does not fade, and in task003 symmetric ρ 0.75 it even grows.
- **The real rule reproduces both tasks.** Agents post only facts that support their vote in the context of their
  memory.

So **the fading in Object Placements, and the persistence in task003, are set by the selection of the posting rule,
not by neutral drift.** This refutes Hypothesis 4 of the 23 October comparison report ("drift removes variety").

**Microscopic support** (`repost_propensity.py`). For held controller-only facts at ρ 0.25, the share that are eligible
for reposting (they support the holder's vote in context) is 0.78–0.82 in task003 and 0.53–0.56 in Object Placements.
Held controller facts per memory fall by 10% in task003 and 22% in Object Placements from t = 30 to 130. In Object
Placements agent facts are equally rarely eligible (0.52–0.65): single facts there support a vote only in particular
combinations. This is consistent with the result, but it is not a derivation: eligibility depends on the whole
memory, so it is not a per-fact constant.
