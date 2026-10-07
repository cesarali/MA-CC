# task003-symmetric — agents can prove the answer; two controller pools

**Frozen.** Pools: 2 October 2026. Agents: 5 October 2026.

## In one paragraph

24 agents each hold one fact. Together they hold all 6 decisive facts, so the
swarm **collectively owns a proof** that `ALLOCATION_0` (A0) is correct, but no
single agent does. A controller steers either toward A0 (the truth) or toward
`ALLOCATION_2` (A2, false). It has **a different pool for each target**: the A0
pool when steering to A0, the A2 pool when steering to A2. The question: can the
swarm assemble the proof it owns, and can a controller help it or stop it?

## Files

| file | what it is |
|---|---|
| `agents.json` | the 24 agents: `agent_assignments` (one fact each) and every fact's details |
| `controller_pool_A0.json` | the 12-fact pool used **only** when the controller targets A0 |
| `controller_pool_A2.json` | the 12-fact pool used **only** when the controller targets A2 |
| `pool_design_2026-10-02.md` | why the two pools are built the way they are (§2) |
| `build.py` | chooses the agents for the two fixed pools; writes all three JSON files |

Entry *i* of the A0 pool is paired in strength with entry *i* of the A2 pool;
each pool file shows the partner of every fact.

## Properties

| | |
|---|---|
| agents, facts per agent | 24, 1 |
| distinct facts the agents hold | 18 |
| agents per allocation A0 / A1 / A2, by fact group | 8 / 8 / 8 |
| **expected starting votes** A0 / A1 / A2 (argmax, random ties) | **9 / 7 / 8** |
| decisive facts held | 6 of 6 |
| agents' joint posterior (all their facts together) | [1, 0, 0]: proves A0 |
| minimal proofs the agents can assemble (up to 6 facts) | 12 |
| agents per decisive fact / per rival fact | 1.33 / 1.33 |
| mean starting belief A0 / A1 / A2 (each agent reading only its own fact) | .367 / .312 / .321 |
| pool sizes | 12 and 12, no fact in both |
| whole-pool posterior | A0 pool [.963, .038, 0]; A2 pool [.114, 0, .886]: neither proves its target |
| pool facts the agents already hold | 2 in each pool, held by 1 agent each |
| **new facts each controller brings** | **10 and 10** |
| strength of those new facts (ΣΔP toward the target) | A0 .957, A2 .969 |

ΔP: how much one fact, read alone, raises an allocation's probability above ⅓.

> **Expected starting votes (added 7 October).** "8 agents per allocation" counts how
> facts were grouped, not how agents vote. Agents vote by argmax with random ties, and
> some agents hold facts tied between two allocations, so expected starting votes are
> **9 / 7 / 8**: eight agents hold tied facts (four A0/A2, two A0/A1, two A1/A2).

## How the pools were designed (2 October, unchanged since)

Reasoning: [`pool_design_2026-10-02.md`](pool_design_2026-10-02.md) §2.

1. **Equal size**: 12 facts each.
2. **Matched strength**: each A0 fact is paired with an A2 fact of near-identical
   ΔP for its own target; 10 of the 12 pairs match exactly.
3. **No decisive facts.**
4. **No fact in both pools.** Without this rule, 12 facts that only rule out A1
   would land in both.
5. **Neither pool can prove its target.** Without this rule, the A0 pool reaches
   P(A0) = 1 and the A2 pool cannot match it.

Only 12 facts in the whole task raise A2 while lowering A0; the A2 pool is all of
them.

## How the agents were chosen (5 October)

The pools were fixed; `build.py` chose agents to fit them. Its rules, in order:

1. 24 agents, one fact each; every fact **eligible** (read alone, it leaves the
   agent at most 45% sure of any allocation); each fact held by 1 or 2 agents.
2. 8 agents per allocation. An agent counts for allocation k when its fact
   raises P(k) at least as much as any other allocation. **A fact tied between
   two allocations may count for either.**
3. The A0 group is exactly the 6 decisive facts.
4. **Equal new evidence for both controllers.** The pool facts the agents
   already hold must be matched pairs (an A0-pool fact with its A2 partner). So
   both controllers bring the same number of new facts, of the same strength.
5. **Equal duplication.** A fact held by two agents survives forgetting better.
   Decisive and rival facts are held by the same mean number of agents (1.33),
   so at ρ = 0.75 the proof does not fade faster than the evidence against it.
6. Then: fewest tied facts, smallest overlap, rival groups of the most similar
   strength.

45,925 agent sets satisfy rules 1–5. Five others tie with the chosen one; they
swap facts of identical strength.

| group (8 agents each) | facts; **bold** = held by 2 agents |
|---|---|
| A0 | **`cf_x00_ge_x04`**, `cf_x01_le_x04`, `cf_x01_le_x05`, `cf_x02_le_x03`, **`cf_x06_le_x08`**, `cf_x07_le_x08`: the 6 decisive facts |
| A1 | `cf_x00_ge_x03` (tie A0/A1), **`cf_x01_eq_x05`**, `cf_x02_ge_x05` (tie A1/A2), **`cf_x03_le_2`**, `cf_x06_le_2` (tie A0/A1), `cf_x06_le_x07` |
| A2 | `cf_x02_le_2` (tie A0/A2), **`cf_x02_le_x04`**, `cf_x03_eq_2` (tie A0/A2), `cf_x03_ge_x05`, **`cf_x06_ge_x07`**, `cf_x08_eq_2` (tie A1/A2) |

The agents already hold `cf_x00_ge_x03` and `cf_x06_le_x07` from the A0 pool, and
their partners `cf_x02_ge_x05` and `cf_x08_eq_2` from the A2 pool.

### The price of rule 5 (state it in any write-up)

Exact duplication needs 6 tied facts among the 12 rival facts, 4 of them tied
with A0. So "8 agents per allocation" really means: 8 agents favour A0 alone; 8
favour A1 and 8 favour A2, and 4 of those 16 favour A0 equally. The effect on
starting beliefs is small because the tied facts are weak (ΔP .014 to .063): mean
starting belief in A0 is .367, against .358 for the best design without ties.

Two of the decisive facts (`cf_x01_le_x04`, `cf_x02_le_x03`) are themselves tied
between A0 and A2; that was always true of the decisive set.

## History

- **2 October**: the two pools designed, with the archived agents.
- **2 October, later**: the pools were replaced by one shared neutral pool
  without the author's agreement. Undone on 5 October.
- **5 October**: agents rebalanced to 8 per allocation, then re-chosen so both
  controllers bring equal new evidence and duplication is equal.
