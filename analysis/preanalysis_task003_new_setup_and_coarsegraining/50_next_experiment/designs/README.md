# designs — the three concrete setups

**PRELIMINARY.** Candidate assignments, not frozen configurations. Nothing here
has been written into a task directory or run.

All three are the **same world** — `task_003`, latent vector
`(3,1,1,2,2,1,1,1,2)`, scores 8 / 4 / 6, `ALLOCATION_0` correct — and the same
49-fact universe. They differ only in *who holds what*.

| file | setup | keep because |
|---|---|---|
| `task003_symmetric_v0_reference.json` | **v0, archived** | the only setup with data (20,730 per-round rows) and calibrated estimators |
| `task003_symmetric_v2.json` | **symmetric-v2** | population *can* prove the truth; tests aggregation |
| `task003_nosolution_v2.json` | **nosolution-v2** | population *cannot*; tests steering of a genuinely undecided swarm |

## Slot lean versus distinct lean

Every agent holds **exactly one fact**. If the same fact is given to two agents
it occupies two **slots** but counts once as a **distinct** fact. Each lean
label is the allocation a fact most favours when read alone.

- **Distinct lean** — how many *different* facts of each lean are in play. This
  sets what the population can *conclude*: the joint posterior, and whether a
  proof exists at all. It is the information content.
- **Slot lean** — how many *agents* hold a fact of each lean. This sets where the
  population *starts*: the initial spread of beliefs and the opening vote split.

They come apart whenever facts are duplicated, and in v0 they come apart badly:

| v0 | A0 | A1 | A2 |
|---|---:|---:|---:|
| distinct lean | 13 | 5 | 3 |
| **slot lean** | **16** | **5** | **3** |

All three duplicated facts in v0 are truth-leaning, so **16 of 24 agents** open
holding truth-leaning evidence against 8 holding rival-leaning evidence. The
initial skew is worse than the distinct count suggests.

Slot multiplicity has a second effect: a fact held by two agents is more likely
to survive epistemic decay at ρ < 1, because it has to be forgotten twice. So
duplication is also a robustness knob, and uneven duplication quietly privileges
whichever facts got doubled. Both v2 designs keep multiplicity in **[1, 2]** and
duplicate evenly within each lean group, by round-robin rather than at random.

## Side by side

| | **v0 (archived)** | **symmetric-v2** | **nosolution-v2** |
|---|---|---|---|
| agents | 24 | 24 | 24 |
| distinct facts | 21 | 16 | 15 |
| distinct lean | 13 / 5 / 3 | 6 / 5 / 5 | **5 / 5 / 5** |
| **slot lean** | **16 / 5 / 3** | **8 / 8 / 8** | **8 / 8 / 8** |
| slot multiplicity | [1, 2] | [1, 2] | [1, 2] |
| agents' joint posterior | [1.0, 0, 0] | [1.0, 0, 0] | **[0.5, 0, 0.5]** |
| decisive facts held | 6 of 6 | **6 of 6** | **0** |
| minimal proofs assemblable | 4 | **8** | **0** |
| controller pool size | 24 | 12 | 12 |
| pool lean | 9 / 6 / 9 | **4 / 4 / 4** | **4 / 4 / 4** |
| pool joint posterior | [0.308, 0, 0.692] | **[⅓, ⅓, ⅓]** | [0.318, 0.364, 0.318] |
| pool proves an allocation? | no | **no** | **no** |
| strength match (W₁) | n/a | 0.046 | **0.009** |
| **agent ↔ pool overlap** | 8 of 21 | **0** | **0** |

## What each design is for

**v0 — reference only.** Its pool is the defective one: admission was evaluated
once for `ALLOCATION_2` and reused for the `ALLOCATION_0` arm, so both arms quote
the same 24 facts and the truth arm recommends A0 while citing evidence chosen to
undermine it. **Never read its truth-arm numbers as evidence about truth-directed
steering.** It is here because it is the only configuration we have data for, and
because any new result needs a bridge back to the existing estimates.

**symmetric-v2 — can the swarm find a proof it owns?** The population's evidence
is collectively sufficient (joint posterior exactly 1.0) but assembling a proof
needs four or more specific facts simultaneously present in one agent's view,
through a board that clears every round and memory that decays. The accuracy
ceiling is exactly 1.0, so the shortfall is pure aggregation-and-control loss and
needs no rescaling. The false controller here must defeat an available
certainty.

**nosolution-v2 — can a controller tip a genuinely undecided swarm?** The
population tops out at a **coin flip between the truth and `ALLOCATION_2`**
(0.5 / 0 / 0.5) and holds no proof at all. Convergence is then real inference
under uncertainty rather than possession of the answer, and control has maximal
leverage. The accuracy ceiling is 0.5, so "fraction correct" needs rescaling
before it is comparable to symmetric.

## Design rules enforced in both v2 setups

1. 24 agents, one fact each, **slot lean 8/8/8**, multiplicity in [1, 2].
2. **One controller pool shared by both targets**, so the menu is identical
   whichever direction is steered and any difference is attributable to the
   selection policy.
3. Pool lean `4/4/4`, strengths matched across truth-leaning and rival-leaning
   halves by Wasserstein-1 distance.
4. **The pool proves no allocation.** This is the constraint Darius's balanced
   pool does not meet — his whole pool has joint posterior 1.0 for the truth and
   contains one size-6 proof, so his truth controller could simply hand over a
   proof, making that arm disclosure rather than persuasion.
5. **Zero agent↔pool overlap.**

## Two knobs, and the trade-offs behind them

**Overlap is a mechanism switch, not just a nuisance.** At zero overlap the
controller can only **inject** facts new to the population; it can never
**refresh** one an agent already holds. Clean for attribution — but refresh is
the best explanation we have for why path divergence was *larger* at ρ = 0.75
than at ρ = 1 despite fewer posts. Treat overlap as a deliberate factor
(0 / 4 / 8) rather than always minimising it.

**Joint neutrality costs strength matching.** symmetric-v2's pool sits exactly on
the prior (⅓ each) but its W₁ is 0.046; nosolution-v2 trades a little neutrality
for W₁ = 0.009. Both cannot be perfect at once at this pool size. Which matters
more depends on whether the claim is about the menu's *composition* or its
*persuasive weight*.

**Why the pool is only 12 facts.** Zero overlap plus balanced agents plus
balanced pool is bounded by scarcity: the task has only 9 A1-leaning and 9
A2-leaning facts in total, so `kr + c ≤ 9` where `kr` is agents' distinct rival
facts per lean and `c` is the pool's per lean. At b = 3 per round over 30 rounds
the controller posts 90 times from a 12-fact menu, so its problem becomes
*timing and selection*, not discovery.

## Known gaps

- `symmetric-v2`'s six A0-leaning distinct facts **are exactly the six decisive
  facts**, so truth-evidence and the proof set coincide perfectly. Setting
  `k0 = 8` would decouple them; see `../builders/scan_design_menu.py`.
- Nothing yet records the **proof-assembly rate** — per round, the fraction of
  agents whose active memory contains a complete proof. That is the quantity
  symmetric-v2 exists to measure.
- The builders read the frozen tasks from a session scratchpad. They need
  repointing at a permanent copy of
  `results/studies/musr_truthful_selective_task_calibration_01/tasks/` from
  `origin/darius-MA-v1` before anyone else can run them.
