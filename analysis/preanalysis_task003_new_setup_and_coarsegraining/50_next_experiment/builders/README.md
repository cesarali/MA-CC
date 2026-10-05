# builders — the scripts that produce and check the two setups

They read the task files from the repository and write nothing into any task
directory.

| script | what it does |
|---|---|
| `build_task003_symmetric.py` | chooses the 24 task003-symmetric agents against the two fixed pools and **writes** `../designs/task003_symmetric.json`. Its rules are in its docstring and in `../designs/README.md` §3 |
| `build_task003_nosolution.py` | rebuilds the task003-nosolution agents and single pool, and **checks** them against `../designs/task003_nosolution.json` |
| `scan_design_menu.py` | scans the feasible combinations of (k0, kr, c) for the single-pool design and prints one candidate per combination |

Run from the repository root with `.venv/bin/python`. Each takes a few seconds.

The rest of this file explains the task003-nosolution search. The two
task003-symmetric pools were designed on 2 October; see
`../../20_controller_redesign/symmetric_controller_and_coarse_graining.md` §2.

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

8 agents per allocation; agents hold no decisive fact; joint posterior searched
toward P(A0) = 0.5; pool `4/4/4`, strength-matched by Wasserstein-1 distance (a
measure of how different two sets of numbers are); **pool proves no
allocation**; zero overlap with the agents.

It does **not** check that agent facts are private-eligible; two are not. See
`../designs/README.md` §7.

The superseded 2 October builder, which also produced a single-pool
task003-symmetric, is in `../archive/build_designs_sharedpool_superseded.py`.
