# task003-nosolution — agents cannot prove anything; one shared pool

**Frozen 6 October 2026.** Rebuilt on 5 October so every agent fact is eligible, and on
6 October so that no controller can ever complete a proof.

## In one paragraph

24 agents each hold one fact. None holds a decisive fact, and together they sit
at **exactly 50/50** between `ALLOCATION_0` (A0, the truth) and `ALLOCATION_2`
(A2). No proof is available to anyone: not to the agents, and not even if the
controller posted its whole pool to them. A controller steers toward A0 or toward
A2 using **one pool for both targets**, holding 4 facts that favour each of A0,
A1 and A2. The menu is **shared**: both controllers choose from the same facts.
That controls what each may say; it does not make both directions equally easy
(see the ceilings below: the truth can reach 0.92, the false target 0.77). The question: can a controller tip a
swarm that is genuinely undecided?

## Files

| file | what it is |
|---|---|
| `agents.json` | the 24 agents: `agent_assignments` (one fact each) and every fact's details |
| `controller_pool.json` | the 12-fact pool, used for **both** targets |
| `build.py` | searches for the agents and the pool together; writes both JSON files (about 2 minutes) |

## Properties

| | |
|---|---|
| agents, facts per agent | 24, 1 |
| distinct facts the agents hold | 15 (5 per allocation) |
| agents per allocation A0 / A1 / A2, by fact group | 8 / 8 / 8 |
| **expected starting votes** A0 / A1 / A2 (argmax, random ties) | **7 / 6 / 11** |
| decisive facts held | 0 of 6 |
| every agent fact eligible (read alone, at most 45% sure of anything) | yes |
| agents' joint posterior (all their facts together) | [0.5, 0, 0.5]: a coin flip, no proof |
| mean starting belief A0 / A1 / A2 | .345 / .307 / .348 |
| pool | 12 facts, 4 favouring each allocation, none held by any agent |
| whole-pool posterior | [.37, .26, .37]: proves nothing, A0 and A2 equal |
| strength of the pool's A0 facts vs its A2 facts (ΣΔP) | .580 vs .631 |
| agents' facts + the whole pool | [.75, 0, .25]: **still proves nothing** |

### How far a controller can push

If all 24 agents pooled their facts and added the best subset of the pool:

| target | highest probability | pool facts needed |
|---|---|---|
| A0 (truth) | **0.92** | 3 |
| A1 | **0** (the agents' facts rule A1 out; adding facts can never bring it back) | — |
| A2 (false) | **0.77** | 2 |

Individual agents, each seeing only part of the evidence, can be pushed further
and in any direction, A1 included. Actual simulations land between the two.
The truth keeps an edge (0.92 vs 0.77): every fact is true, so the real world
always survives. That edge cannot be designed away, only reduced.

ΔP: how much one fact, read alone, raises an allocation's probability above ⅓.

> **Expected starting votes (added 7 October).** "8 agents per allocation" counts how
> facts were grouped, not how agents vote. Agents vote by argmax with random ties, and
> some agents hold facts tied between two allocations, so expected starting votes are
> **7 / 6 / 11**: six agents hold facts tied between A1 and A2 (counted as A1) and half of
> them vote A2; two hold A0/A1 ties. So the coin-flip swarm starts with more A2 voters.
> This is a candidate cause of the A2 drift in the silent arm (see
> `../../simulator/simulation_1_spec.md` §10).

## How it was built

`build.py` enforces, in order:

1. 24 agents, one fact each, each fact held by 1 or 2 agents.
2. Every agent fact **eligible**. The pool may hold ineligible facts: the rule
   limits what one agent holds, not what the controller says.
3. No decisive fact; 5 distinct facts favouring each allocation, 8 agents each.
   A fact belongs to the allocation it raises most; a tie goes to the
   lower-numbered allocation.
4. The agents' joint posterior is **exactly** (0.5, 0, 0.5).
5. One 12-fact pool, 4 facts per allocation, proving nothing, sharing no fact
   with the agents.
6. **No joint proof**: the agents' facts plus the *whole* pool must prove
   nothing. A proof can only grow as facts are added, so then no subset of the
   pool can complete one either. Without this rule one pool fact completed a
   proof of A0, and the truth controller could simply hand it over: disclosure,
   not persuasion.
7. Among the agent sets meeting rules 4 and 6, minimise the sum of three
   imbalances, all in probability units:
   - the strength gap between the pool's A0 facts and its A2 facts (the pool
     serves both targets, so neither controller should get stronger facts);
   - the whole pool's distance from the prior;
   - the gap between the agents' mean starting belief in A0 and in A2.

   Result: 0.013, 0.037 and 0.002.

**Why the search is small:** only 9 facts favour A1 and 9 favour A2, against 31
for A0. With 5 of each in the agents' hands and 4 in the pool, all 18 are used,
so choosing the agents' rival facts fixes the pool's.

**Why 24 agents:** Darius's `task_004` removed the decisive facts *together with
the agents holding them*, leaving 15 agents, which mixes "no proof available"
with "fewer agents". Here the decisive facts are *replaced*.

The pool holds a strongest-fact mirror pair: `cf_x01_eq_1` for A0 (ΔP .252) and
`cf_x05_eq_1` for A2 (.252). The two facts that each completed a proof
(`cf_x00_eq_3`, `cf_x06_eq_1`) are not in the pool.

**What rule 6 cost:** the pool leans slightly away from A1 (.26), and its A2
facts are a little stronger than its A0 facts (.631 vs .580). Both are reported
rather than hidden; a refinement that also balances the two ceilings is listed
for testing after the first simulations.

## History

- **2 October**: first version. Two agents each held the two strongest facts in
  the task (`cf_x01_eq_1`, `cf_x05_eq_1`, each making an agent 58.5% sure),
  breaking the eligibility rule.
- **5 October**: rebuilt with eligible agent facts only. The rebuild also
  balanced the pool (ΣΔP .627 vs .629, against .335 vs .446) and halved the
  agents' starting gap (.012, against .028). The first version is recoverable
  from git; see [`../../README.md`](../../README.md#recovering-deleted-material).
- **6 October**: both earlier versions let a single pool fact complete a proof
  of A0 for the pooled agents. Rule 6 added; agents and pool rebuilt. The
  5 October version is in git history (commit `6b2ccbf`).
