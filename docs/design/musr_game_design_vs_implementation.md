# MuSR Team Allocation: original design vs our implementation

*2026-09-28. Question: is our fact space biased toward truth, and does our game
match the original MuSR construction?*

Sources: the paper (Sprague et al., ICLR 2024, arXiv 2310.16049, §4.3 and
App. G.4), upstream code at the commit we pin (`b1f4d41`,
`src/dataset_types/team_allocation.py::build_assignment`, `create_facts`), the
released `datasets/team_allocation.json` (250 instances), and our generator in
`src/mas_cc/musr_team_allocation_generator/`. Every "fact leans X" number below
is exact: the posterior over the 3 allocations after conditioning on that fact
alone, from our 14,388-world model (uniform prior, unique winner).

## 1. How the original game is built

**The puzzle.** 3 people, 2 tasks. One person does task 1 alone, the other two
do task 2 together, so there are 3 possible allocations. 9 hidden numbers:
6 skills (each person x each task) and 3 pairwise teamwork scores, each on a
3-level scale. An allocation's score = solo person's task-1 skill + both pair
members' task-2 skills + the pair's teamwork. Highest score wins.

**Scale.** The paper says values 0/1/2; the code uses 1/2/3 (BAD/OKAY/GOOD).
Same thing, shifted by one.

**Latent sampling is constructive, not random.** The code builds the answer in:
person 0 is always the solo, persons 1-2 always the pair. Every entry that does
*not* enter the gold score starts at the floor value (BAD). Only the four
gold-relevant entries are sampled. Then, while gold leads by more than 2, it
raises entries of a rival allocation one at a time (max 10 steps) so a rival
ends up close behind. In the released data, 4.04 of the 5 non-gold entries are
still at the floor.

**The facts.** Exactly 9 "gold facts", one per hidden number, each stating it
directly: "Sarah is bad at making coffee", "Luis and John work well together".
An LLM expands each into an entailment tree whose leaves are story-level
details plus unstated commonsense rules, then writes a narrative containing
the leaves. Table 2: 10 reasoning steps, 9 commonsense facts per instance.

**Not in the design:** no distractors or red herrings for this domain, no
evidence balancing, no notion of facts supporting a wrong answer. The goal is
a solvable puzzle (human accuracy 100%).

### Upstream inconsistency worth knowing

The paper says the gold allocation beats all others "by a score of at least
2". The code does the opposite: it *shrinks* the gap to at most 2. Released
data, gold margin over the runner-up:

| margin | 0 (tie) | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| instances | 15 | 53 | 167 | 8 | 7 |

So 27% of the 250 released instances have margin below 2, and **15 (6%) are
exact ties** — no unique answer under the stated scoring rule. (The code's
`assert delta > 0` only runs inside the update loop, so a tie at the initial
draw passes through. It also raises the *lowest*-scoring rival, not the
second-best the variable name says.)

## 2. How we built it

- **Latents:** i.i.d. uniform over {1,2,3} for all 9 values, reject unless
  there is a unique winner with margin >= `min_margin` (default 1).
  `latent_problem.py::generate_latent_problem`.
- **Facts:** not MuSR's 9 exact values. `symbolic_facts.py` builds a catalog
  of ~99 machine-checkable propositions per world: exact values, >=2 / <=2
  thresholds, and pairwise <= / = / >= comparisons. A task keeps only the
  **true** ones (49 in task_004).
- **Partition** (`selective_design.py`, our own design, no MuSR analogue):
  *decisive* = a greedy set that recovers the truth; *controller-compatible* =
  true facts that individually raise P(false target); *neutral* = the rest.
- **Surface text:** agents in task_004 see canonical sentences
  ("Bruno's skill for build the data pipeline is at most moderate"), not a
  narrative. The LLM entailment-tree path exists in the generator but is not
  what the swarm runs on.

## 3. Do they match?

| | original MuSR | ours (task_004) | match |
|---|---|---|---|
| people / tasks / split | 3 / 2 / 1+2 | same | yes |
| hidden values | 9 on a 3-level scale | same | yes |
| scoring | sum of skills + pair teamwork | same | yes |
| unique winner | intended (6% ties in practice) | enforced | ours is stricter |
| margin | paper: >= 2; code: <= 2 | >= 1 | no |
| latent sampling | constructive, off-gold at floor | i.i.d. + rejection | no |
| facts | 9 exact values | 49 true comparisons / thresholds | no |
| presentation | narrative, soft reasoning | canonical sentences | no |
| fact roles | none | decisive / controller / neutral | ours only |

The game *rules* match. The *generative process* and the *evidence* do not:
we kept MuSR's puzzle and replaced its evidence model with a symbolic,
selectable one. That substitution is what makes posterior-controlled
selection possible, so it is a deliberate departure, but "MuSR-style" should
be stated that way in the paper, not as "MuSR".

## 4. The bias question

Fraction of facts that, alone, most raise the probability of the true
allocation:

| fact set | leans truth | leans a rival |
|---|---|---|
| original MuSR, released data (9 facts x 250) | **71.5%** | 28.5% |
| ours, full task_004 catalog (49 true facts) | 63% (31) | 37% (18) |
| **ours, facts the 15 agents actually hold** | **47% (7)** | **53% (8)** |

So the premise holds for the catalog and even more for the original, but the
task_004 *corpus* is already balanced by count (mean dP(truth) per fact =
+0.009). It still resolves to truth when pooled:

| conditioning on | P(A0 truth, A1, A2) | worlds left |
|---|---|---|
| nothing | 0.33 / 0.33 / 0.33 | 14,388 |
| the 7 truth-leaning facts | 0.86 / 0.06 / 0.08 | 830 |
| the 8 rival-leaning facts | 0.23 / 0.11 / **0.66** | 349 |
| all 15 | **0.74** / 0.00 / 0.26 | 38 |

Two things follow.

1. **The rival-leaning half is genuinely persuasive**: on its own it makes a
   *wrong* answer (A2) the favourite at 0.66. Individually those 8 facts split
   between A1 and A2 (5 and 3); jointly they concentrate on A2.
2. **Count balance is the wrong knob.** All facts are true statements about one
   world, so any large enough set of them is jointly consistent only with
   worlds near the true one, and the pooled posterior drifts to truth no matter
   how individual lean is balanced. With true-only facts this cannot be
   designed away; it can only be tuned. "50/50" has to be defined at the
   corpus level, e.g. the pooled posterior over the agents' facts
   (now 0.74 / 0.26) or the strength of what each half proves on its own
   (now 0.86 vs 0.66).
