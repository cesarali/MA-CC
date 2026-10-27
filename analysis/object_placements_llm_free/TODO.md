# TODO — object_placements_llm_free

Deferred work, newest decisions first. Move an item to "Done" with a date and a pointer when it is finished.
Every rebuilt setup is a **new** setup folder next to the existing ones (`op_symmetric`, `op_nosolution` are used in
runs and stay as they are).

## Setups (deferred 2026-10-23: keep the current setups, continue the theory first)

Why: the A2 pools differ from task003's in how evidence is spread over facts. All values are P(A0, A1, A2) for an
agent holding the whole set (computed 23 Oct from the worlds):

| setup | whole A2 pool | controller-only part | single pool facts: median / max P(A2) |
|---|---|---|---|
| task003 symmetric | (0.114, 0, 0.886) | (0.065, 0, 0.935) | 0.42 / 0.585 |
| task003 no-proof | (0.37, 0.26, 0.37) | same | 0.35 / 0.585 |
| OP symmetric | (0.018, 0, 0.982) | (0.078, 0.098, 0.824) | 0.333 / 0.487 |
| OP no-proof | (0.667, 0, 0.333) | same | 0.333 / 0.487 |

Only 6 of the story's 34 true facts have single-fact P(A2) > 0.40 (task003: 12 of 49).

- [ ] **Balanced no-proof pool** (`op_nosolution_balanced`): add to `simulation_2/analysis/build_op_setups.py` the
  rule that the pool alone is balanced between A0 and A2, as in task003 (0.37 / 0.37). Cheap; closest to task003's design.
- [ ] **Symmetric A2 pool from individually A2-leaning facts** (`op_symmetric_strong`): pick pool facts by single-fact
  P(A2). At most about 6 facts qualify, so this works only partly.
- [ ] **Another story** whose true facts lean to A2 one at a time: search other Object Placements stories (changes the
  world, not only the pools).
- [ ] Options 2–3 change how evidence is spread over facts, the variable the comparison report
  (`analysis/theory_claude/reports/2026-10-23_task_comparison/`) suggests drives the task differences; they can double as a
  test of that explanation. Each needs new simulations (reversal grid and persistence branches).
