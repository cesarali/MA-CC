# task003-nosolution — agents cannot prove anything; one unbiased pool

**Frozen 5 October 2026** (rebuilt that day so every agent fact is eligible).

## In one paragraph

24 agents each hold one fact. None holds a decisive fact, and together they sit
at **exactly 50/50** between `ALLOCATION_0` (A0, the truth) and `ALLOCATION_2`
(A2), with no proof available to anyone. A controller steers toward A0 or toward
A2 using **one pool for both targets**, holding 4 facts that favour each of A0,
A1 and A2. The menu is unbiased, so any difference between the two targets comes
from which facts the controller picks. The question: can a controller tip a
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
| agents per allocation A0 / A1 / A2 | 8 / 8 / 8 |
| decisive facts held | 0 of 6 |
| every agent fact eligible (read alone, at most 45% sure of anything) | yes |
| agents' joint posterior (all their facts together) | [0.5, 0, 0.5]: a coin flip, no proof |
| mean starting belief A0 / A1 / A2 | .349 / .291 / .361 |
| pool | 12 facts, 4 favouring each allocation, none held by any agent |
| whole-pool posterior | [.329, .341, .329]: proves nothing, A0 and A2 equal |
| strength of the pool's A0 facts vs its A2 facts (ΣΔP) | .627 vs .629 |

ΔP: how much one fact, read alone, raises an allocation's probability above ⅓.

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
6. Among the 10,438 agent sets meeting rule 4, minimise the sum of three
   imbalances, all in probability units:
   - the strength gap between the pool's A0 facts and its A2 facts (the pool
     serves both targets, so neither controller should get stronger facts);
   - the whole pool's distance from the prior;
   - the gap between the agents' mean starting belief in A0 and in A2.

   Result: 0.0007, 0.004 and 0.012.

**Why the search is small:** only 9 facts favour A1 and 9 favour A2, against 31
for A0. With 5 of each in the agents' hands and 4 in the pool, all 18 are used,
so choosing the agents' rival facts fixes the pool's.

**Why 24 agents:** Darius's `task_004` removed the decisive facts *together with
the agents holding them*, leaving 15 agents, which mixes "no proof available"
with "fewer agents". Here the decisive facts are *replaced*.

The pool includes the strongest A0 fact (`cf_x00_eq_3`, ΔP .252) and its A2
mirror (`cf_x05_eq_1`, .252) as a matched pair.

## History

- **2 October**: first version. Two agents each held the two strongest facts in
  the task (`cf_x01_eq_1`, `cf_x05_eq_1`, each making an agent 58.5% sure),
  breaking the eligibility rule.
- **5 October**: rebuilt with eligible agent facts only. The rebuild also
  balanced the pool (ΣΔP .627 vs .629, against .335 vs .446) and halved the
  agents' starting gap (.012, against .028). The first version is recoverable
  from git; see [`../../README.md`](../../README.md#recovering-deleted-material).
