# builders — the scripts that produce and check the two setups

They read the task files from the repository and write nothing into any task
directory.

| script | what it does |
|---|---|
| `build_task003_symmetric.py` | chooses the 24 task003-symmetric agents against the two fixed pools and **writes** `../designs/task003_symmetric.json`. Its rules are in its docstring and in `../designs/README.md` §3 |
| `build_task003_nosolution.py` | chooses the task003-nosolution agents and single pool and **writes** `../designs/task003_nosolution.json`. Its rules are in its docstring and in `../designs/README.md` §4 |
| `scan_design_menu.py` | scans the feasible combinations of (k0, kr, c) for the single-pool design and prints one candidate per combination |

Run from the repository root with `.venv/bin/python`. Each takes a few seconds.

The rest of this file explains the task003-nosolution search. The two
task003-symmetric pools were designed on 2 October; see
`../designs/symmetric_pools_design_2026-10-02.md` §2.

## Parameters

- `k0` — distinct A0-leaning (truth-leaning) facts the agents hold
- `kr` — distinct A1- and A2-leaning facts the agents hold, each
- `c` — facts per allocation in the controller pool (pool size `3c`)
- agents always have **24 slots**, one fact each, filled by repeating facts so
  the *slot-level* lean is 8/8/8

## The binding constraint

The task has only **9 A1-leaning and 9 A2-leaning facts** in total, against 31
A0-leaning. Rival-leaning facts are the scarce resource, and both the agents and
the controller want them. Zero overlap therefore requires

    kr + c <= 9

Darius's balanced pool takes *all 18* rival facts ("keep all rival-leaning
facts"), which is why every one of the 22 facts outside it is A0-leaning, and
why any lean-balanced agent set must overlap it by exactly two thirds.

## Constraints the task003-nosolution search enforces

8 agents per allocation, every agent fact private-eligible; no decisive fact;
joint posterior exactly (0.5, 0, 0.5); pool `4/4/4`, proving no allocation, zero
overlap with the agents. Ranked by the sum of the pool's A0-vs-A2 strength gap,
the pool's distance from the prior, and the agents' A0-vs-A2 starting gap.
With `kr = 5` and `c = 4` the bound is tight (5 + 4 = 9): choosing the agents'
rival facts fixes the pool's.

The 2 October version (two ineligible agent facts) was deleted on 6 October;
see `../../README.md`, 'Recovering deleted material'.
