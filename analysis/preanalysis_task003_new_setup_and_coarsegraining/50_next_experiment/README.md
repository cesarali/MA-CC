# 50_next_experiment — the next controller study

**The two setups are frozen (5 October 2026). The run settings are not, and
nothing has been launched.**

## Start here

| if you want | read |
|---|---|
| what the two setups are, and why | [`designs/README.md`](designs/README.md) |
| what we are running: arms, phases, cell counts | [`EXPERIMENTS.md`](EXPERIMENTS.md) |
| to build the task directories and configs | [`designs/AGENT_RUNBOOK.md`](designs/AGENT_RUNBOOK.md) |

## The two setups, in one table

| | **task003-symmetric** | **task003-nosolution** |
|---|---|---|
| agents | 24, one fact each, 8 favouring each allocation | same |
| agents hold the 6 decisive facts | **yes**, so they can prove `ALLOCATION_0` | **no**, so they cannot prove anything |
| controller pools | **two**: the A0 pool for a controller targeting `ALLOCATION_0`, the A2 pool for one targeting `ALLOCATION_2` | **one**, for both targets, 4 facts favouring each allocation |
| file | [`designs/task003_symmetric.json`](designs/task003_symmetric.json) | [`designs/task003_nosolution.json`](designs/task003_nosolution.json) |
| built by | [`builders/build_task003_symmetric.py`](builders/build_task003_symmetric.py) | [`builders/build_task003_nosolution.py`](builders/build_task003_nosolution.py) |

## What is in this folder

| folder or file | what it is |
|---|---|
| [`designs/`](designs/) | the frozen setups, their specification, the build runbook, the config template |
| [`builders/`](builders/) | the scripts that produce and check the setups |
| [`EXPERIMENTS.md`](EXPERIMENTS.md) | the experiment plan: arms, phases, budget protocol, cell counts |
| [`notes/`](notes/) | dated working notes, still correct as records: what Darius built ([`prior_art`](notes/prior_art_2026-10-02.md)), every protocol option ([`protocols`](notes/protocols_2026-10-02.md)), the branch merge ([`merge_assessment`](notes/merge_assessment_2026-10-04.md)), the 4 October summary ([`SUMMARY`](notes/SUMMARY_2026-10-04.md)), research ideas ([`research_directions`](notes/research_directions_2026-10-04.md)) |
| [`archive/`](archive/) | superseded material, kept for the record: the 2 October design draft, and the 2 October shared-pool symmetric setup and its builder |

The notes are dated snapshots. Where they disagree with `designs/README.md`, the
designs README wins. In particular, notes written before 5 October may describe
task003-symmetric with a single shared pool; that design was superseded.

## Points that still hold from the planning notes

**The truth/false asymmetry cannot be removed.** Every fact is true about one
world, so a large enough set of facts is consistent only with worlds near the
truth. A truthful false controller can never hand over a proof of a false
answer. Report the comparison as *correction versus selective persuasion*, not
as perfectly symmetric steering.

**Why the decisive-fact factor is worth keeping.** Darius's `task_004` is
`task_003` with the decisive facts removed, but it removed them *together with
the agents holding them* (24 agents → 15), mixing up "no proof available" with
"fewer agents". task003-nosolution keeps 24 agents and replaces the facts.

**30 rounds, and argue for 100 episodes per cell.** 15 rounds cuts the path
divergence off while it is still growing. At the post-meeting plan's 50 episodes
the confidence interval on the main quantity is about ±33%, too wide to separate
two steering directions. The calculation is in
[`archive/design_preliminary_2026-10-02.md`](archive/design_preliminary_2026-10-02.md) §3.

## Open questions

1. Does the post-meeting plan's "50 episodes per cell" mean 50 **initializations**
   or 50 trajectories? Only initializations are independent, so the two readings
   give very different precision.
2. Scripted (deterministic) controller first, or LLM controller first? Our order
   reverses the plan's priorities; this needs team agreement
   ([`EXPERIMENTS.md`](EXPERIMENTS.md) §3d).
