# builders — searching for improved agent assignments and controller pools

**Preliminary.** These scripts search for assignments satisfying the design
constraints discussed on 2026-10-02. They write nothing into any task
directory; they print candidates for review.

| script | what it does |
|---|---|
| `scan_design_menu.py` | scans the feasible frontier over (k0, kr, c) and prints one candidate per combination |
| `build_designs_v2.py` | produces the two recommended designs with full fact lists and proof counts |

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

## Constraints enforced

Both setups: slot lean 8/8/8; pool lean `c/c/c`; pool strength-matched across
halves by Wasserstein-1; **pool must not prove any allocation**; overlap
minimised.

- `task003-symmetric`: agents hold all 6 decisive facts, joint posterior = 1.0
- `task003-nosolution`: agents hold no decisive fact, joint posterior < 1,
  searched toward P(A0) = 0.5

Run with the repo venv and `PYTHONPATH` unset — the scripts add `src/` and
`../10_task_and_facts` themselves. They read the frozen tasks extracted from
`origin/darius-MA-v1`; see `../prior_art_2026-10-02.md` §6.
